#include <cuda.h>
#include <cuda_fp8.h>
#include <dlfcn.h>
#include "kernel.h"
#include <cstdio>
#include <cmath>
#include <cstring>
#include <nvtx3/nvToolsExt.h>

#ifndef CUDART_INF_F
#define CUDART_INF_F (__int_as_float(0x7f800000))
#endif

// =====================================================================
// Fused GEMM1 kernel v2: TMA + 128B-swizzle + double-buffer pipeline
//
// Optimizations vs v1:
//   #2  cp.async.bulk.tensor.2d.tile::gather4  (A matrix TMA gather)
//   #3  cp.async.bulk.tensor.2d.tile           (W matrix TMA tile)
//   #5  cta_group::1 on TMA loads where applicable
//   #6  Double-buffered SMEM with compute/memory overlap pipeline
//   128B SMEM swizzle for bank-conflict-free MMA reads
//
// Architecture: SM 100a (B200 Blackwell)
// =====================================================================

constexpr int TC5_BM     = 128;
constexpr int TC5_BN     = 128;
constexpr int TC5_BK     = 128;
constexpr int TC5_MMA_K  = 32;
constexpr int TC5_KINNER = TC5_BK / TC5_MMA_K;  // 4
constexpr int TC5_KT     = 56;    // 7168 / 128
constexpr int TC5_HIDDEN = 7168;
constexpr int TC5_N_DIM  = 4096;
constexpr int TC5_GATHER4_ITERS = TC5_BM / 4;  // 32

constexpr int TC5_G2_KT = 16;       // K/128 = 2048/128
constexpr int TC5_G2_K  = 2048;
constexpr int TC5_G2_N  = 7168;

constexpr int G2_SMEM_AS       = 0;
constexpr int G2_SMEM_WS       = G2_SMEM_AS + 2 * TC5_BM * TC5_BK;
constexpr int G2_SMEM_MBAR_TMA = G2_SMEM_WS + 2 * TC5_BN * TC5_BK;
constexpr int G2_SMEM_MBAR_MMA = G2_SMEM_MBAR_TMA + 16;
constexpr int G2_SMEM_TMEM     = G2_SMEM_MBAR_MMA + 8;
constexpr int G2_SMEM_TOTAL    = G2_SMEM_TMEM + 12;

// smem .shared::cta address (u32)
__device__ __forceinline__ uint32_t smem_to_cta_addr(const void* ptr) {
  uint32_t addr;
  asm volatile(
    "{ .reg .u64 u64addr;\n"
    "  cvta.to.shared.u64 u64addr, %1;\n"
    "  cvt.u32.u64 %0, u64addr; }\n"
    : "=r"(addr) : "l"(ptr)
  );
  return addr;
}

// SMEM descriptor for 128B-swizzled layout (tcgen05 MMA)
// SBO = 8 rows × 128B atom = 1024 bytes; LBO unused with swizzle
__device__ __forceinline__ uint64_t make_smem_desc_swizzled(uint32_t smem_addr) {
  constexpr uint32_t SBO = 8 * 128;  // 1024
  uint64_t desc = 0;
  desc |= ((uint64_t)((smem_addr & 0x3FFFFu) >> 4)) & 0x3FFFull;         // bits [0:13]  start addr
  desc |= (((uint64_t)((SBO & 0x3FFFFu) >> 4)) & 0x3FFFull) << 32;      // bits [32:45] SBO
  desc |= 1ull << 46;                                                      // version = 0b01
  desc |= 2ull << 61;                                                      // swizzle = 128B
  return desc;
}

// Forward decl: definition is later in the file (used by DIAG dequant kernel)
__device__ __forceinline__ float4 fp8x4_e4m3_to_f32x4(uint32_t packed);

// FP32 → E8M0 (truncation, matches hw block_scale semantics)
__device__ __forceinline__ uint8_t fp32_to_e8m0(float v) {
  uint32_t bits = __float_as_uint(v);
  uint32_t biased_exp = (bits >> 23) & 0xFFu;
  if (biased_exp == 255u) return 0xFFu;
  return static_cast<uint8_t>(biased_exp);
}

// =====================================================================
// GPU prefix sum: counts[32] → offsets[33], expert_Tk[32]
// =====================================================================
__global__ void compute_offsets_kernel(
    const int* __restrict__ counts,
    int* __restrict__ offsets,
    int* __restrict__ expert_Tk)
{
  if (threadIdx.x != 0) return;
  int sum = 0;
  offsets[0] = 0;
  for (int i = 0; i < NUM_LOCAL_EXPERTS; i++) {
    int c = counts[i];
    expert_Tk[i] = c;
    sum += c;
    offsets[i + 1] = sum;
  }
}

void launch_compute_offsets(const int* counts, int* offsets, int* expert_Tk, cudaStream_t stream) {
  compute_offsets_kernel<<<1, 32, 0, stream>>>(counts, offsets, expert_Tk);
  CUDA_CHECK(cudaGetLastError());
}

// =====================================================================
// Main fused kernel: TMA gather4(A) + TMA tile(W) + pipeline + tcgen05 MMA
// =====================================================================
// Dynamic shared memory layout (total ~66 KB, exceeds 48 KB static limit)
// All TMA destination buffers at 128-byte-aligned offsets.
constexpr int SMEM_AS       = 0;                                          // 32768
constexpr int SMEM_WS       = SMEM_AS + 2 * TC5_BM * TC5_BK;            // 32768
constexpr int SMEM_MBAR_TMA = SMEM_WS + 2 * TC5_BN * TC5_BK;            // 16
constexpr int SMEM_MBAR_MMA = SMEM_MBAR_TMA + 16;                        // 8
constexpr int SMEM_TIDS     = SMEM_MBAR_MMA + 8;                         // 512
constexpr int SMEM_TMEM     = SMEM_TIDS + TC5_BM * (int)sizeof(int);     // 12
constexpr int SMEM_TOTAL    = SMEM_TMEM + 12;                            // ~66084

__global__ void __launch_bounds__(128, 1)
fused_gather_blockscale_gemm1_kernel(
    const float*   __restrict__ A_scale,     // [56, T]
    const int*     __restrict__ token_ids,   // [Tk]
    const float*   __restrict__ W13_scale,   // [32, 56]
    float*         __restrict__ G1,          // [Tk, 4096]
    int Tk, int T,
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_W)
{
  extern __shared__ __align__(128) uint8_t smem[];

  uint8_t*  As_raw       = smem + SMEM_AS;
  uint8_t*  Ws_raw       = smem + SMEM_WS;
  uint64_t* mbar_tma     = (uint64_t*)(smem + SMEM_MBAR_TMA);
  uint64_t* mbar_mma_ptr = (uint64_t*)(smem + SMEM_MBAR_MMA);
  int*      s_token_ids  = (int*)(smem + SMEM_TIDS);
  uint32_t* s_tmem_d     = (uint32_t*)(smem + SMEM_TMEM);
  uint32_t* s_tmem_sa    = (uint32_t*)(smem + SMEM_TMEM + 4);
  uint32_t* s_tmem_sb    = (uint32_t*)(smem + SMEM_TMEM + 8);

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const int m_base  = blockIdx.y * TC5_BM;
  const int n_base  = blockIdx.x * TC5_BN;

  if (m_base >= Tk || n_base >= TC5_N_DIM) return;

  // ---- 1. Warp-group register budget ----
  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

  // ---- 2. TMEM allocation (D=128col, scale_A=32col, scale_B=32col) ----
  uint32_t tmem_d;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_d);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_d = *s_tmem_d;

  uint32_t tmem_scale_a;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sa);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_a = *s_tmem_sa;

  uint32_t tmem_scale_b;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sb);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_b = *s_tmem_sb;

  // ---- 3. Preload token IDs for gather4 coordinates ----
  if (tid < TC5_BM) {
    int mg = m_base + tid;
    s_token_ids[tid] = (mg < Tk) ? token_ids[mg] : 0;
  }

  // ---- 4. mbarrier init ----
  if (tid == 0) {
    uint32_t tma0 = smem_to_cta_addr(&mbar_tma[0]);
    uint32_t tma1 = smem_to_cta_addr(&mbar_tma[1]);
    uint32_t mma  = smem_to_cta_addr(mbar_mma_ptr);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;\n" :: "r"(mma));
  }
  __syncthreads();

  float acc[TC5_BN];
  #pragma unroll
  for (int i = 0; i < TC5_BN; i++) acc[i] = 0.0f;

  int tma_phase[2] = {0, 0};
  int mma_phase    = 0;

  constexpr uint32_t TMA_BYTES = TC5_BM * TC5_BK + TC5_BN * TC5_BK;  // 32768

  // Helper: offset into double-buffered arrays
  #define AS_BUF(b) (As_raw + (b) * TC5_BM * TC5_BK)
  #define WS_BUF(b) (Ws_raw + (b) * TC5_BN * TC5_BK)

  // ---- 5. Pipeline priming: issue first TMA load (kt=0 → buf[0]) ----
  if (tid == 0) {
    uint32_t tma0_addr = smem_to_cta_addr(&mbar_tma[0]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(tma0_addr), "r"(TMA_BYTES));

    uint32_t a_dst = smem_to_cta_addr(AS_BUF(0));
    #pragma unroll 4
    for (int g = 0; g < TC5_GATHER4_ITERS; g++) {
      int y0 = s_token_ids[g * 4 + 0];
      int y1 = s_token_ids[g * 4 + 1];
      int y2 = s_token_ids[g * 4 + 2];
      int y3 = s_token_ids[g * 4 + 3];
      asm volatile(
        "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
        :: "r"(a_dst + g * 4 * TC5_BK), "l"(tma_A),
           "r"(0), "r"(y0), "r"(y1), "r"(y2), "r"(y3),
           "r"(tma0_addr)
        : "memory");
    }

    uint32_t w_dst = smem_to_cta_addr(WS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(w_dst), "l"(tma_W), "r"(0), "r"(n_base), "r"(tma0_addr)
      : "memory");
  }

  // ---- 6. K-tile loop ----
  constexpr uint32_t idesc =
      (0u  <<  0)
    | (0u  <<  2)
    | (0u  <<  4)
    | (0u  <<  7)
    | (0u  << 10)
    | (0u  << 13)
    | (0u  << 14)
    | (0u  << 15)
    | (0u  << 16)
    | ((TC5_BN >> 3) << 17)
    | (1u  << 23)
    | (0u  << 24)
    | ((TC5_BM >> 7) << 27)
    | (0u  << 29)
    ;

  for (int kt = 0; kt < TC5_KT; kt++) {
    const int buf      = kt & 1;
    const int next_buf = 1 - buf;

    // (a) Compute E8M0 scales + compensation — issued BEFORE TMA wait
    //     ld.global for A_scale/W13_scale overlaps with in-flight TMA transfer
    float a_comp_val, w_comp;
    {
      int mg = m_base + tid;
      float sa = (mg < Tk) ? A_scale[kt * T + token_ids[mg]] : 1.0f;
      uint32_t sa_bits = __float_as_uint(sa);
      uint32_t sa_exp  = (sa_bits >> 23) & 0xFFu;
      uint32_t my_scale_a = (sa_exp == 255u) ? 0xFFu : sa_exp;

      if (sa_exp == 0u)        a_comp_val = sa * 0x1.0p127f;
      else if (sa_exp < 255u)  a_comp_val = __uint_as_float((sa_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     a_comp_val = 0.0f;

      int n_block = n_base / 128;
      float sb = W13_scale[n_block * TC5_KT + kt];
      uint32_t sb_bits = __float_as_uint(sb);
      uint32_t sb_exp  = (sb_bits >> 23) & 0xFFu;
      uint32_t my_scale_b = (sb_exp == 255u) ? 0xFFu : sb_exp;

      if (sb_exp == 0u)        w_comp = sb * 0x1.0p127f;
      else if (sb_exp < 255u)  w_comp = __uint_as_float((sb_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     w_comp = 0.0f;

      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a));
      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // (b) Wait for current buffer TMA completion
    {
      uint32_t mbar_addr = smem_to_cta_addr(&mbar_tma[buf]);
      uint32_t parity    = tma_phase[buf];
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_TMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_TMA_%=;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(parity));
      tma_phase[buf] ^= 1;
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    // (c) Prefetch next K-tile into other buffer (overlap with MMA compute)
    if (kt + 1 < TC5_KT && tid == 0) {
      uint32_t next_mbar = smem_to_cta_addr(&mbar_tma[next_buf]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
          :: "r"(next_mbar), "r"(TMA_BYTES));

      int next_k = (kt + 1) * TC5_BK;
      uint32_t a_dst = smem_to_cta_addr(AS_BUF(next_buf));
      #pragma unroll 4
      for (int g = 0; g < TC5_GATHER4_ITERS; g++) {
        int y0 = s_token_ids[g * 4 + 0];
        int y1 = s_token_ids[g * 4 + 1];
        int y2 = s_token_ids[g * 4 + 2];
        int y3 = s_token_ids[g * 4 + 3];
        asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global"
          ".mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
          :: "r"(a_dst + g * 4 * TC5_BK), "l"(tma_A),
             "r"(next_k), "r"(y0), "r"(y1), "r"(y2), "r"(y3),
             "r"(next_mbar)
          : "memory");
      }

      uint32_t w_dst = smem_to_cta_addr(WS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(w_dst), "l"(tma_W), "r"(next_k), "r"(n_base), "r"(next_mbar)
        : "memory");
    }

    // (d) 4x MMA — only accumulate within this K-tile, NOT across K-tiles
    uint32_t a_smem_base = smem_to_cta_addr(AS_BUF(buf));
    uint32_t w_smem_base = smem_to_cta_addr(WS_BUF(buf));

    #pragma unroll
    for (int ki = 0; ki < TC5_KINNER; ki++) {
      uint64_t a_desc = make_smem_desc_swizzled(a_smem_base + ki * TC5_MMA_K);
      uint64_t b_desc = make_smem_desc_swizzled(w_smem_base + ki * TC5_MMA_K);

      if (tid == 0) {
        int accumulate = (ki > 0) ? 1 : 0;
        asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %6, 0;\n"
          "  tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
          "  [%0], %1, %2, %3, [%4], [%5], p;\n"
          "}\n"
          :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc),
             "r"(tmem_scale_a), "r"(tmem_scale_b), "r"(accumulate));

        uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
        asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
          :: "r"(mma_mbar) : "memory");
      }
    }

    // (e) Wait for all 4 MMA steps to complete
    {
      uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
      uint32_t parity = mma_phase;
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_MMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_MMA_%=;\n"
        "}\n"
        :: "r"(mma_mbar), "r"(parity));
      mma_phase ^= 1;
    }

    // (f) Read back TMEM results, apply per-K-tile compensation
    float my_comp = a_comp_val * w_comp;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        uint32_t v0, v1, v2, v3;
        asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
          : "r"(tmem_d + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");

        acc[col]     += __uint_as_float(v0) * my_comp;
        acc[col + 1] += __uint_as_float(v1) * my_comp;
        acc[col + 2] += __uint_as_float(v2) * my_comp;
        acc[col + 3] += __uint_as_float(v3) * my_comp;
      }
    }
  } // end K-tile loop

  #undef AS_BUF
  #undef WS_BUF

  // ---- 7. Write results to G1 ----
  {
    const int out_row = m_base + tid;
    if (out_row < Tk) {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        int gn0 = n_base + col;
        if (gn0     < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0]     = acc[col];
        if (gn0 + 1 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 1] = acc[col + 1];
        if (gn0 + 2 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 2] = acc[col + 2];
        if (gn0 + 3 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 3] = acc[col + 3];
      }
    }
  }

  // ---- 8. TMEM dealloc ----
  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(tmem_d));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_a));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_b));
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 232;\n");
}

// =====================================================================
// Host-side: create TMA tensor maps + launch
// =====================================================================

// Dynamically resolve cuTensorMapEncodeTiled from libcuda.so (driver API)
typedef CUresult (*cuTensorMapEncodeTiled_fn)(
    CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*,
    const cuuint64_t*, const cuuint64_t*, const cuuint32_t*, const cuuint32_t*,
    CUtensorMapInterleave, CUtensorMapSwizzle, CUtensorMapL2promotion,
    CUtensorMapFloatOOBfill);

static cuTensorMapEncodeTiled_fn get_cuTensorMapEncodeTiled() {
  static cuTensorMapEncodeTiled_fn fn = nullptr;
  if (!fn) {
    void* h = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!h) h = dlopen("libcuda.so", RTLD_LAZY);
    if (h) fn = (cuTensorMapEncodeTiled_fn)dlsym(h, "cuTensorMapEncodeTiled");
    if (!fn) fprintf(stderr, "FATAL: cannot resolve cuTensorMapEncodeTiled\n");
  }
  return fn;
}

#define CU_CHECK(status) \
  do { \
    CUresult r__ = (status); \
    if (r__ != CUDA_SUCCESS) { \
      fprintf(stderr, "CUDA Driver Error %d at %s:%d\n", (int)r__, __FILE__, __LINE__); \
    } \
  } while (0)

void launch_fused_gather_blockscale_gemm1(
    const uint8_t* A_fp8,
    const float*   A_scale,
    const int*     token_ids,
    const uint8_t* W13_fp8,
    const float*   W13_scale,
    float*         G1,
    int Tk, int T,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_fused_gather_blockscale_gemm1(TMA+pipeline)");
  if (Tk <= 0) { nvtxRangePop(); return; }

  auto encodeFn = get_cuTensorMapEncodeTiled();

  // --- Create CUtensorMap for A (gather4 mode): [T, 7168] uint8 ---
  // dim0 = K (innermost, contiguous), dim1 = T (rows)
  // box = [128, 1] for gather4 (dim1 must be 1)
  CUtensorMap tma_A_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_HIDDEN, (cuuint64_t)T};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_HIDDEN};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, 1};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_A_host,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A_fp8,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  // --- Create CUtensorMap for W (tile mode): [4096, 7168] uint8 ---
  // dim0 = K (innermost), dim1 = N
  // box = [128, 128] → one full BN×BK tile per TMA
  CUtensorMap tma_W_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_HIDDEN, (cuuint64_t)TC5_N_DIM};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_HIDDEN};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BN};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_W_host,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)W13_fp8,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  // --- Copy tensor maps to device (persistent static buffer) ---
  static CUtensorMap* d_tma_buf = nullptr;
  if (!d_tma_buf) {
    CUDA_CHECK(cudaMalloc(&d_tma_buf, 2 * sizeof(CUtensorMap)));
  }
  CUtensorMap h_tma[2] = {tma_A_host, tma_W_host};
  CUDA_CHECK(cudaMemcpyAsync(d_tma_buf, h_tma, 2 * sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));

  // --- Launch ---
  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fused_gather_blockscale_gemm1_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL));
    smem_attr_set = true;
  }

  dim3 block(128);
  dim3 grid(
    (TC5_N_DIM + TC5_BN - 1) / TC5_BN,
    (Tk        + TC5_BM - 1) / TC5_BM
  );
  fused_gather_blockscale_gemm1_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
      A_scale, token_ids, W13_scale, G1, Tk, T,
      d_tma_buf,        // tma_A
      d_tma_buf + 1);   // tma_W
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

// =====================================================================
// Grouped GEMM1: all 32 experts in ONE launch. Grid = (32, max_m_tiles, 32).
// blockIdx.x = n_block, blockIdx.y = m_block, blockIdx.z = expert_id
// Core MMA logic IDENTICAL to fused_gather_blockscale_gemm1_kernel.
// =====================================================================
__global__ void __launch_bounds__(128, 1)
grouped_gemm1_kernel(
    const float*   __restrict__ A_scale,
    const int*     __restrict__ all_token_ids,
    const int*     __restrict__ expert_offsets,
    const int*     __restrict__ expert_Tk,
    const float*   __restrict__ W13_scale_base,
    float*         __restrict__ G1_all,
    int T,
    const CUtensorMap* __restrict__ tma_A,
    const CUtensorMap* __restrict__ tma_W_array)
{
  const int expert_id = blockIdx.z;
  const int Tk = expert_Tk[expert_id];
  const int m_base = blockIdx.y * TC5_BM;
  const int n_base = blockIdx.x * TC5_BN;

  if (Tk == 0 || m_base >= Tk || n_base >= TC5_N_DIM) return;

  const int offset = expert_offsets[expert_id];
  const int* token_ids = all_token_ids + offset;
  float* G1 = G1_all + (int64_t)offset * TC5_N_DIM;
  const float* W13_scale = W13_scale_base + (int64_t)expert_id * NUM_GEMM1_OUT_BLOCKS * TC5_KT;
  const CUtensorMap* tma_W = &tma_W_array[expert_id];

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  As_raw       = smem + SMEM_AS;
  uint8_t*  Ws_raw       = smem + SMEM_WS;
  uint64_t* mbar_tma     = (uint64_t*)(smem + SMEM_MBAR_TMA);
  uint64_t* mbar_mma_ptr = (uint64_t*)(smem + SMEM_MBAR_MMA);
  int*      s_token_ids  = (int*)(smem + SMEM_TIDS);
  uint32_t* s_tmem_d     = (uint32_t*)(smem + SMEM_TMEM);
  uint32_t* s_tmem_sa    = (uint32_t*)(smem + SMEM_TMEM + 4);
  uint32_t* s_tmem_sb    = (uint32_t*)(smem + SMEM_TMEM + 8);

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

  uint32_t tmem_d;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_d);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_d = *s_tmem_d;

  uint32_t tmem_scale_a;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sa);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_a = *s_tmem_sa;

  uint32_t tmem_scale_b;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sb);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_b = *s_tmem_sb;

  if (tid < TC5_BM) {
    int mg = m_base + tid;
    s_token_ids[tid] = (mg < Tk) ? token_ids[mg] : 0;
  }

  if (tid == 0) {
    uint32_t tma0 = smem_to_cta_addr(&mbar_tma[0]);
    uint32_t tma1 = smem_to_cta_addr(&mbar_tma[1]);
    uint32_t mma  = smem_to_cta_addr(mbar_mma_ptr);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;\n" :: "r"(mma));
  }
  __syncthreads();

  float acc[TC5_BN];
  #pragma unroll
  for (int i = 0; i < TC5_BN; i++) acc[i] = 0.0f;

  int tma_phase[2] = {0, 0};
  int mma_phase    = 0;
  constexpr uint32_t TMA_BYTES_G1 = TC5_BM * TC5_BK + TC5_BN * TC5_BK;

  #define GG1_AS_BUF(b) (As_raw + (b) * TC5_BM * TC5_BK)
  #define GG1_WS_BUF(b) (Ws_raw + (b) * TC5_BN * TC5_BK)

  if (tid == 0) {
    uint32_t tma0_addr = smem_to_cta_addr(&mbar_tma[0]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(tma0_addr), "r"(TMA_BYTES_G1));

    uint32_t a_dst = smem_to_cta_addr(GG1_AS_BUF(0));
    #pragma unroll 4
    for (int g = 0; g < TC5_GATHER4_ITERS; g++) {
      int y0 = s_token_ids[g * 4 + 0];
      int y1 = s_token_ids[g * 4 + 1];
      int y2 = s_token_ids[g * 4 + 2];
      int y3 = s_token_ids[g * 4 + 3];
      asm volatile(
        "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
        :: "r"(a_dst + g * 4 * TC5_BK), "l"(tma_A),
           "r"(0), "r"(y0), "r"(y1), "r"(y2), "r"(y3),
           "r"(tma0_addr)
        : "memory");
    }

    uint32_t w_dst = smem_to_cta_addr(GG1_WS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(w_dst), "l"(tma_W), "r"(0), "r"(n_base), "r"(tma0_addr)
      : "memory");
  }

  constexpr uint32_t gg1_idesc =
      (0u  <<  0)
    | (0u  <<  2)
    | (0u  <<  4)
    | (0u  <<  7)
    | (0u  << 10)
    | (0u  << 13)
    | (0u  << 14)
    | (0u  << 15)
    | (0u  << 16)
    | ((TC5_BN >> 3) << 17)
    | (1u  << 23)
    | (0u  << 24)
    | ((TC5_BM >> 7) << 27)
    | (0u  << 29)
    ;

  for (int kt = 0; kt < TC5_KT; kt++) {
    const int buf      = kt & 1;
    const int next_buf = 1 - buf;

    float a_comp_val, w_comp;
    {
      int mg = m_base + tid;
      float sa = (mg < Tk) ? A_scale[kt * T + token_ids[mg]] : 1.0f;
      uint32_t sa_bits = __float_as_uint(sa);
      uint32_t sa_exp  = (sa_bits >> 23) & 0xFFu;
      uint32_t my_scale_a = (sa_exp == 255u) ? 0xFFu : sa_exp;

      if (sa_exp == 0u)        a_comp_val = sa * 0x1.0p127f;
      else if (sa_exp < 255u)  a_comp_val = __uint_as_float((sa_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     a_comp_val = 0.0f;

      int n_block = n_base / 128;
      float sb = W13_scale[n_block * TC5_KT + kt];
      uint32_t sb_bits = __float_as_uint(sb);
      uint32_t sb_exp  = (sb_bits >> 23) & 0xFFu;
      uint32_t my_scale_b = (sb_exp == 255u) ? 0xFFu : sb_exp;

      if (sb_exp == 0u)        w_comp = sb * 0x1.0p127f;
      else if (sb_exp < 255u)  w_comp = __uint_as_float((sb_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     w_comp = 0.0f;

      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a));
      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    {
      uint32_t mbar_addr = smem_to_cta_addr(&mbar_tma[buf]);
      uint32_t parity    = tma_phase[buf];
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_GG1_TMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_GG1_TMA_%=;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(parity));
      tma_phase[buf] ^= 1;
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    if (kt + 1 < TC5_KT && tid == 0) {
      uint32_t next_mbar = smem_to_cta_addr(&mbar_tma[next_buf]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
          :: "r"(next_mbar), "r"(TMA_BYTES_G1));

      int next_k = (kt + 1) * TC5_BK;
      uint32_t a_dst = smem_to_cta_addr(GG1_AS_BUF(next_buf));
      #pragma unroll 4
      for (int g = 0; g < TC5_GATHER4_ITERS; g++) {
        int y0 = s_token_ids[g * 4 + 0];
        int y1 = s_token_ids[g * 4 + 1];
        int y2 = s_token_ids[g * 4 + 2];
        int y3 = s_token_ids[g * 4 + 3];
        asm volatile(
          "cp.async.bulk.tensor.2d.tile::gather4.shared::cta.global"
          ".mbarrier::complete_tx::bytes"
          " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];\n"
          :: "r"(a_dst + g * 4 * TC5_BK), "l"(tma_A),
             "r"(next_k), "r"(y0), "r"(y1), "r"(y2), "r"(y3),
             "r"(next_mbar)
          : "memory");
      }

      uint32_t w_dst = smem_to_cta_addr(GG1_WS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(w_dst), "l"(tma_W), "r"(next_k), "r"(n_base), "r"(next_mbar)
        : "memory");
    }

    uint32_t a_smem_base = smem_to_cta_addr(GG1_AS_BUF(buf));
    uint32_t w_smem_base = smem_to_cta_addr(GG1_WS_BUF(buf));

    #pragma unroll
    for (int ki = 0; ki < TC5_KINNER; ki++) {
      uint64_t a_desc = make_smem_desc_swizzled(a_smem_base + ki * TC5_MMA_K);
      uint64_t b_desc = make_smem_desc_swizzled(w_smem_base + ki * TC5_MMA_K);

      if (tid == 0) {
        int accumulate = (ki > 0) ? 1 : 0;
        asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %6, 0;\n"
          "  tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
          "  [%0], %1, %2, %3, [%4], [%5], p;\n"
          "}\n"
          :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(gg1_idesc),
             "r"(tmem_scale_a), "r"(tmem_scale_b), "r"(accumulate));

        uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
        asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
          :: "r"(mma_mbar) : "memory");
      }
    }

    {
      uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
      uint32_t parity = mma_phase;
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_GG1_MMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_GG1_MMA_%=;\n"
        "}\n"
        :: "r"(mma_mbar), "r"(parity));
      mma_phase ^= 1;
    }

    float my_comp = a_comp_val * w_comp;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        uint32_t v0, v1, v2, v3;
        asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
          : "r"(tmem_d + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        acc[col]     += __uint_as_float(v0) * my_comp;
        acc[col + 1] += __uint_as_float(v1) * my_comp;
        acc[col + 2] += __uint_as_float(v2) * my_comp;
        acc[col + 3] += __uint_as_float(v3) * my_comp;
      }
    }
  }

  #undef GG1_AS_BUF
  #undef GG1_WS_BUF

  {
    const int out_row = m_base + tid;
    if (out_row < Tk) {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        int gn0 = n_base + col;
        if (gn0     < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0]     = acc[col];
        if (gn0 + 1 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 1] = acc[col + 1];
        if (gn0 + 2 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 2] = acc[col + 2];
        if (gn0 + 3 < TC5_N_DIM) G1[out_row * TC5_N_DIM + gn0 + 3] = acc[col + 3];
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(tmem_d));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_a));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_b));
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 232;\n");
}

// =====================================================================
// Grouped GEMM2: all 32 experts in ONE launch. Grid = (56, max_m_tiles, 32).
// blockIdx.x = n_block, blockIdx.y = m_block, blockIdx.z = expert_id
// =====================================================================
__global__ void __launch_bounds__(128, 1)
grouped_gemm2_kernel(
    const float*   __restrict__ C_scale_all,
    const int*     __restrict__ expert_offsets,
    const int*     __restrict__ expert_Tk,
    const float*   __restrict__ W2_scale_base,
    float*         __restrict__ Otmp_all,
    int Tk_stride,
    const CUtensorMap* __restrict__ tma_C,
    const CUtensorMap* __restrict__ tma_W2_array)
{
  const int expert_id = blockIdx.z;
  const int Tk = expert_Tk[expert_id];
  const int m_base = blockIdx.y * TC5_BM;
  const int n_base = blockIdx.x * TC5_BN;

  if (Tk == 0 || m_base >= Tk || n_base >= TC5_G2_N) return;

  const int offset = expert_offsets[expert_id];
  const float* C_scale = C_scale_all;
  const float* W2_scale = W2_scale_base + (int64_t)expert_id * NUM_HIDDEN_BLOCKS * TC5_G2_KT;
  float* O = Otmp_all + (int64_t)offset * TC5_G2_N;
  const CUtensorMap* tma_W2 = &tma_W2_array[expert_id];
  const int tma_m = offset + m_base;

  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  As_raw       = smem + G2_SMEM_AS;
  uint8_t*  Ws_raw       = smem + G2_SMEM_WS;
  uint64_t* mbar_tma     = (uint64_t*)(smem + G2_SMEM_MBAR_TMA);
  uint64_t* mbar_mma_ptr = (uint64_t*)(smem + G2_SMEM_MBAR_MMA);
  uint32_t* s_tmem_d     = (uint32_t*)(smem + G2_SMEM_TMEM);
  uint32_t* s_tmem_sa    = (uint32_t*)(smem + G2_SMEM_TMEM + 4);
  uint32_t* s_tmem_sb    = (uint32_t*)(smem + G2_SMEM_TMEM + 8);

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

  uint32_t tmem_d;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_d);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_d = *s_tmem_d;

  uint32_t tmem_scale_a;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sa);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_a = *s_tmem_sa;

  uint32_t tmem_scale_b;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sb);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_b = *s_tmem_sb;

  if (tid == 0) {
    uint32_t tma0 = smem_to_cta_addr(&mbar_tma[0]);
    uint32_t tma1 = smem_to_cta_addr(&mbar_tma[1]);
    uint32_t mma  = smem_to_cta_addr(mbar_mma_ptr);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;\n" :: "r"(mma));
  }
  __syncthreads();

  float acc[TC5_BN];
  #pragma unroll
  for (int i = 0; i < TC5_BN; i++) acc[i] = 0.0f;

  int tma_phase[2] = {0, 0};
  int mma_phase    = 0;
  constexpr uint32_t TMA_BYTES_G2 = TC5_BM * TC5_BK + TC5_BN * TC5_BK;

  #define GG2_AS_BUF(b) (As_raw + (b) * TC5_BM * TC5_BK)
  #define GG2_WS_BUF(b) (Ws_raw + (b) * TC5_BN * TC5_BK)

  if (tid == 0) {
    uint32_t tma0_addr = smem_to_cta_addr(&mbar_tma[0]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(tma0_addr), "r"(TMA_BYTES_G2));

    uint32_t a_dst = smem_to_cta_addr(GG2_AS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(a_dst), "l"(tma_C), "r"(0), "r"(tma_m), "r"(tma0_addr)
      : "memory");

    uint32_t w_dst = smem_to_cta_addr(GG2_WS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(w_dst), "l"(tma_W2), "r"(0), "r"(n_base), "r"(tma0_addr)
      : "memory");
  }

  constexpr uint32_t gg2_idesc =
      (0u  <<  0)
    | (0u  <<  2)
    | (0u  <<  4)
    | (0u  <<  7)
    | (0u  << 10)
    | (0u  << 13)
    | (0u  << 14)
    | (0u  << 15)
    | (0u  << 16)
    | ((TC5_BN >> 3) << 17)
    | (1u  << 23)
    | (0u  << 24)
    | ((TC5_BM >> 7) << 27)
    | (0u  << 29)
    ;

  for (int kt = 0; kt < TC5_G2_KT; kt++) {
    const int buf      = kt & 1;
    const int next_buf = 1 - buf;

    float a_comp_val, w_comp;
    {
      int mg = m_base + tid;
      float sa = (mg < Tk) ? C_scale[kt * Tk_stride + (offset + mg)] : 1.0f;
      uint32_t sa_bits = __float_as_uint(sa);
      uint32_t sa_exp  = (sa_bits >> 23) & 0xFFu;
      uint32_t my_scale_a = (sa_exp == 255u) ? 0xFFu : sa_exp;

      if (sa_exp == 0u)        a_comp_val = sa * 0x1.0p127f;
      else if (sa_exp < 255u)  a_comp_val = __uint_as_float((sa_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     a_comp_val = 0.0f;

      int n_block = n_base / 128;
      float sb = W2_scale[n_block * TC5_G2_KT + kt];
      uint32_t sb_bits = __float_as_uint(sb);
      uint32_t sb_exp  = (sb_bits >> 23) & 0xFFu;
      uint32_t my_scale_b = (sb_exp == 255u) ? 0xFFu : sb_exp;

      if (sb_exp == 0u)        w_comp = sb * 0x1.0p127f;
      else if (sb_exp < 255u)  w_comp = __uint_as_float((sb_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     w_comp = 0.0f;

      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a));
      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    {
      uint32_t mbar_addr = smem_to_cta_addr(&mbar_tma[buf]);
      uint32_t parity    = tma_phase[buf];
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_GG2_TMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_GG2_TMA_%=;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(parity));
      tma_phase[buf] ^= 1;
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    if (kt + 1 < TC5_G2_KT && tid == 0) {
      uint32_t next_mbar = smem_to_cta_addr(&mbar_tma[next_buf]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
          :: "r"(next_mbar), "r"(TMA_BYTES_G2));

      int next_k = (kt + 1) * TC5_BK;
      uint32_t a_dst = smem_to_cta_addr(GG2_AS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(a_dst), "l"(tma_C), "r"(next_k), "r"(tma_m), "r"(next_mbar)
        : "memory");

      uint32_t w_dst = smem_to_cta_addr(GG2_WS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(w_dst), "l"(tma_W2), "r"(next_k), "r"(n_base), "r"(next_mbar)
        : "memory");
    }

    uint32_t a_smem_base = smem_to_cta_addr(GG2_AS_BUF(buf));
    uint32_t w_smem_base = smem_to_cta_addr(GG2_WS_BUF(buf));

    #pragma unroll
    for (int ki = 0; ki < TC5_KINNER; ki++) {
      uint64_t a_desc = make_smem_desc_swizzled(a_smem_base + ki * TC5_MMA_K);
      uint64_t b_desc = make_smem_desc_swizzled(w_smem_base + ki * TC5_MMA_K);

      if (tid == 0) {
        int accumulate = (ki > 0) ? 1 : 0;
        asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %6, 0;\n"
          "  tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
          "  [%0], %1, %2, %3, [%4], [%5], p;\n"
          "}\n"
          :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(gg2_idesc),
             "r"(tmem_scale_a), "r"(tmem_scale_b), "r"(accumulate));

        uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
        asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
          :: "r"(mma_mbar) : "memory");
      }
    }

    {
      uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
      uint32_t parity = mma_phase;
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_GG2_MMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_GG2_MMA_%=;\n"
        "}\n"
        :: "r"(mma_mbar), "r"(parity));
      mma_phase ^= 1;
    }

    float my_comp = a_comp_val * w_comp;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        uint32_t v0, v1, v2, v3;
        asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
          : "r"(tmem_d + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
        acc[col]     += __uint_as_float(v0) * my_comp;
        acc[col + 1] += __uint_as_float(v1) * my_comp;
        acc[col + 2] += __uint_as_float(v2) * my_comp;
        acc[col + 3] += __uint_as_float(v3) * my_comp;
      }
    }
  }

  #undef GG2_AS_BUF
  #undef GG2_WS_BUF

  {
    const int out_row = m_base + tid;
    if (out_row < Tk) {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        int gn0 = n_base + col;
        if (gn0     < TC5_G2_N) O[out_row * TC5_G2_N + gn0]     = acc[col];
        if (gn0 + 1 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 1] = acc[col + 1];
        if (gn0 + 2 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 2] = acc[col + 2];
        if (gn0 + 3 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 3] = acc[col + 3];
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(tmem_d));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_a));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_b));
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 232;\n");
}

// =====================================================================
// Batched SwiGLU+Quantize for all experts in one launch.
// Grid = (16, Tk_ub, 1).  Operates on contiguous G1_all buffer.
// =====================================================================
__global__ void batched_swiglu_quantize_kernel(
    const float* __restrict__ G1_all,
    int Tk_ub,
    uint8_t* __restrict__ C_fp8_all,
    float* __restrict__ C_scale_all)
{
  const int col_block = blockIdx.x;
  const int row       = blockIdx.y;
  const int tid       = threadIdx.x;
  if (row >= Tk_ub) return;

  const int col_base  = col_block * 128 + tid * 4;
  const float* g1_row = G1_all + row * GEMM1_OUT_SIZE;

  float4 x1 = *reinterpret_cast<const float4*>(g1_row + col_base);
  float4 x2 = *reinterpret_cast<const float4*>(g1_row + col_base + INTERMEDIATE_SIZE);

  float v0 = (x2.x / (1.0f + __expf(-x2.x))) * x1.x;
  float v1 = (x2.y / (1.0f + __expf(-x2.y))) * x1.y;
  float v2 = (x2.z / (1.0f + __expf(-x2.z))) * x1.z;
  float v3 = (x2.w / (1.0f + __expf(-x2.w))) * x1.w;

  float absv = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    absv = fmaxf(absv, __shfl_xor_sync(0xffffffffu, absv, off));
  const float block_max = absv;

  float    ratio      = block_max * (1.0f / 448.0f);
  uint32_t rb         = __float_as_uint(ratio);
  uint32_t mant       = rb & 0x007FFFFFu;
  uint32_t exp_b      = (rb >> 23) & 0xFFu;
  if (mant != 0u) exp_b += 1u;
  exp_b = max(1u, min(exp_b, 254u));
  const uint32_t scale_bits = (exp_b << 23);
  const float    scale_f    = __uint_as_float(scale_bits);
  const float    inv_scale  = 1.0f / scale_f;

  uint8_t b0 = __nv_cvt_float_to_fp8(v0 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b1 = __nv_cvt_float_to_fp8(v1 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b2 = __nv_cvt_float_to_fp8(v2 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b3 = __nv_cvt_float_to_fp8(v3 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint32_t packed = (uint32_t)b0
                  | ((uint32_t)b1 <<  8)
                  | ((uint32_t)b2 << 16)
                  | ((uint32_t)b3 << 24);

  *reinterpret_cast<uint32_t*>(C_fp8_all + row * INTERMEDIATE_SIZE + col_base) = packed;
  if (tid == 0) {
    C_scale_all[col_block * Tk_ub + row] = scale_f;
  }
}

// =====================================================================
// Batched accumulate_weighted_add for all experts in one launch.
// Grid = (ceil(H/256), max_Tk, 32).
// blockIdx.z = expert_id
// =====================================================================
__global__ void batched_accumulate_weighted_add_kernel(
    const float* __restrict__ Otmp_all,
    const int*   __restrict__ all_token_ids,
    const float* __restrict__ all_token_wts,
    const int*   __restrict__ expert_offsets,
    const int*   __restrict__ expert_Tk,
    int H,
    float* __restrict__ output)
{
  const int expert_id = blockIdx.z;
  const int row = blockIdx.y;
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int Tk = expert_Tk[expert_id];
  if (row >= Tk || col >= H) return;

  const int offset = expert_offsets[expert_id];
  const int global_row = offset + row;
  atomicAdd(&output[all_token_ids[global_row] * H + col],
            Otmp_all[global_row * H + col] * all_token_wts[global_row]);
}

// =====================================================================
// Launch wrappers for grouped/batched kernels
// =====================================================================

void launch_grouped_gemm1(
    const uint8_t* A_fp8,
    const float*   A_scale,
    const int*     all_token_ids,
    const int*     expert_offsets,
    const int*     expert_Tk,
    const float*   W13_scale_base,
    const uint8_t* W13_fp8_base,
    float*         G1_all,
    int T, int Tk_ub,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_grouped_gemm1");
  auto encodeFn = get_cuTensorMapEncodeTiled();

  CUtensorMap tma_A_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_HIDDEN, (cuuint64_t)T};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_HIDDEN};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, 1};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_A_host, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)A_fp8,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  CUtensorMap tma_W_hosts[NUM_LOCAL_EXPERTS];
  for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
    const uint8_t* w_ptr = W13_fp8_base + (int64_t)e * GEMM1_OUT_SIZE * HIDDEN_SIZE;
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_HIDDEN, (cuuint64_t)TC5_N_DIM};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_HIDDEN};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BN};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_W_hosts[e], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)w_ptr,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  static CUtensorMap* d_tma_buf = nullptr;
  if (!d_tma_buf) {
    CUDA_CHECK(cudaMalloc(&d_tma_buf, (1 + NUM_LOCAL_EXPERTS) * sizeof(CUtensorMap)));
  }
  CUDA_CHECK(cudaMemcpyAsync(d_tma_buf, &tma_A_host, sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_tma_buf + 1, tma_W_hosts, NUM_LOCAL_EXPERTS * sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));

  static bool smem_set = false;
  if (!smem_set) {
    CUDA_CHECK(cudaFuncSetAttribute(grouped_gemm1_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_TOTAL));
    smem_set = true;
  }

  int max_m_tiles = (Tk_ub + TC5_BM - 1) / TC5_BM;
  dim3 block(128);
  dim3 grid(
    (TC5_N_DIM + TC5_BN - 1) / TC5_BN,
    max_m_tiles,
    NUM_LOCAL_EXPERTS
  );
  grouped_gemm1_kernel<<<grid, block, SMEM_TOTAL, stream>>>(
      A_scale, all_token_ids, expert_offsets, expert_Tk,
      W13_scale_base, G1_all, T,
      d_tma_buf, d_tma_buf + 1);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_batched_swiglu_quantize(
    const float* G1_all, int Tk_ub,
    uint8_t* C_fp8_all, float* C_scale_all,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_batched_swiglu_quantize");
  if (Tk_ub <= 0) { nvtxRangePop(); return; }
  dim3 block(32);
  dim3 grid(INTERMEDIATE_SIZE / 128, Tk_ub);
  batched_swiglu_quantize_kernel<<<grid, block, 0, stream>>>(
      G1_all, Tk_ub, C_fp8_all, C_scale_all);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_grouped_gemm2(
    const uint8_t* C_fp8_all,
    const float*   C_scale_all,
    const uint8_t* W2_fp8_base,
    const float*   W2_scale_base,
    float*         Otmp_all,
    const int*     expert_offsets,
    const int*     expert_Tk,
    int Tk_ub,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_grouped_gemm2");
  auto encodeFn = get_cuTensorMapEncodeTiled();

  CUtensorMap tma_C_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_G2_K, (cuuint64_t)Tk_ub};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_G2_K};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BM};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_C_host, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)C_fp8_all,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  CUtensorMap tma_W2_hosts[NUM_LOCAL_EXPERTS];
  for (int e = 0; e < NUM_LOCAL_EXPERTS; e++) {
    const uint8_t* w_ptr = W2_fp8_base + (int64_t)e * HIDDEN_SIZE * INTERMEDIATE_SIZE;
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_G2_K, (cuuint64_t)TC5_G2_N};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_G2_K};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BN};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_W2_hosts[e], CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)w_ptr,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  static CUtensorMap* d_tma_g2 = nullptr;
  if (!d_tma_g2) {
    CUDA_CHECK(cudaMalloc(&d_tma_g2, (1 + NUM_LOCAL_EXPERTS) * sizeof(CUtensorMap)));
  }
  CUDA_CHECK(cudaMemcpyAsync(d_tma_g2, &tma_C_host, sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_tma_g2 + 1, tma_W2_hosts, NUM_LOCAL_EXPERTS * sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));

  static bool smem_set = false;
  if (!smem_set) {
    CUDA_CHECK(cudaFuncSetAttribute(grouped_gemm2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, G2_SMEM_TOTAL));
    smem_set = true;
  }

  int max_m_tiles = (Tk_ub + TC5_BM - 1) / TC5_BM;
  dim3 block(128);
  dim3 grid(
    (TC5_G2_N + TC5_BN - 1) / TC5_BN,
    max_m_tiles,
    NUM_LOCAL_EXPERTS
  );
  grouped_gemm2_kernel<<<grid, block, G2_SMEM_TOTAL, stream>>>(
      C_scale_all, expert_offsets, expert_Tk,
      W2_scale_base, Otmp_all, Tk_ub,
      d_tma_g2, d_tma_g2 + 1);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_batched_accumulate(
    const float* Otmp_all,
    const int*   all_token_ids,
    const float* all_token_wts,
    const int*   expert_offsets,
    const int*   expert_Tk,
    int Tk_ub, int H,
    float* output,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_batched_accumulate");
  if (Tk_ub <= 0) { nvtxRangePop(); return; }
  dim3 block(256);
  dim3 grid(
    (H + block.x - 1) / block.x,
    Tk_ub,
    NUM_LOCAL_EXPERTS
  );
  batched_accumulate_weighted_add_kernel<<<grid, block, 0, stream>>>(
      Otmp_all, all_token_ids, all_token_wts,
      expert_offsets, expert_Tk, H, output);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

// =====================================================================
// FP16 GEMM2 pipeline (from solution6): kind::f16 tcgen05 MMA
// =====================================================================

// SMEM descriptor for kind::f16 core-matrix-contiguous layout (no swizzle)
__device__ __forceinline__ uint64_t make_smem_desc_f16(uint32_t smem_addr, int height) {
  uint32_t LBO = height * 16;
  constexpr uint32_t SBO = 8 * 16;  // 128
  uint64_t desc = 0;
  desc |= ((uint64_t)((smem_addr & 0x3FFFFu) >> 4)) & 0x3FFFull;
  desc |= (((uint64_t)((LBO & 0x3FFFFu) >> 4)) & 0x3FFFull) << 16;
  desc |= (((uint64_t)((SBO & 0x3FFFFu) >> 4)) & 0x3FFFull) << 32;
  desc |= 1ull << 46;
  return desc;
}

__global__ void swiglu_to_f16_kernel(
    const float* __restrict__ G1,
    const float* __restrict__ row_w,
    int Tk,
    uint16_t* __restrict__ C_f16)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Tk * INTERMEDIATE_SIZE;
  if (idx >= total) return;
  int row = idx / INTERMEDIATE_SIZE;
  int col = idx - row * INTERMEDIATE_SIZE;
  float x1 = G1[row * GEMM1_OUT_SIZE + col];
  float x2 = G1[row * GEMM1_OUT_SIZE + col + INTERMEDIATE_SIZE];
  float val = (x2 / (1.0f + __expf(-x2))) * x1 * row_w[row];
  val = fminf(fmaxf(val, -65504.0f), 65504.0f);
  uint32_t h;
  asm("cvt.rn.f16.f32 %0, %1;\n" : "=r"(h) : "f"(val));
  C_f16[row * INTERMEDIATE_SIZE + col] = (uint16_t)h;
}

__global__ void fp8_dequant_to_f16_kernel(
    const uint8_t* __restrict__ M_fp8,
    const float* __restrict__ S,
    int cols, int S_cols,
    uint16_t* __restrict__ M_f16)
{
  int blk_row = blockIdx.y;
  int blk_col = blockIdx.x;
  float scale = S[blk_row * S_cols + blk_col];
  int row_base = blk_row * 128;
  int col_base = blk_col * 128;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  #pragma unroll
  for (int r = ty; r < 128; r += 8) {
    int row = row_base + r;
    int col = col_base + tx * 4;
    const uint8_t* in_ptr = M_fp8 + row * cols + col;
    uint16_t* out_ptr = M_f16 + row * cols + col;

    uint32_t packed = *reinterpret_cast<const uint32_t*>(in_ptr);
    uint32_t lo_h2, hi_h2;
    asm("{\n"
        "  .reg .b16 lo16, hi16;\n"
        "  mov.b32 {lo16, hi16}, %2;\n"
        "  cvt.rn.f16x2.e4m3x2 %0, lo16;\n"
        "  cvt.rn.f16x2.e4m3x2 %1, hi16;\n"
        "}\n"
        : "=r"(lo_h2), "=r"(hi_h2) : "r"(packed));

    float f0, f1, f2, f3;
    asm("{\n .reg .b16 a,b;\n mov.b32 {a,b},%2;\n cvt.f32.f16 %0,a;\n cvt.f32.f16 %1,b;\n}\n"
        : "=f"(f0), "=f"(f1) : "r"(lo_h2));
    asm("{\n .reg .b16 a,b;\n mov.b32 {a,b},%2;\n cvt.f32.f16 %0,a;\n cvt.f32.f16 %1,b;\n}\n"
        : "=f"(f2), "=f"(f3) : "r"(hi_h2));
    f0 *= scale; f1 *= scale; f2 *= scale; f3 *= scale;

    uint32_t out_lo, out_hi;
    asm("{\n .reg .b16 h0,h1,h2,h3;\n"
        "  cvt.rn.f16.f32 h0,%2;\n cvt.rn.f16.f32 h1,%3;\n"
        "  cvt.rn.f16.f32 h2,%4;\n cvt.rn.f16.f32 h3,%5;\n"
        "  mov.b32 %0,{h0,h1};\n mov.b32 %1,{h2,h3};\n}\n"
        : "=r"(out_lo), "=r"(out_hi)
        : "f"(f0), "f"(f1), "f"(f2), "f"(f3));

    *reinterpret_cast<uint32_t*>(out_ptr)     = out_lo;
    *reinterpret_cast<uint32_t*>(out_ptr + 2) = out_hi;
  }
}

constexpr int G2F_BM     = 128;
constexpr int G2F_BN     = 128;
constexpr int G2F_BK     = 64;
constexpr int G2F_MMA_K  = 16;
constexpr int G2F_KINNER = G2F_BK / G2F_MMA_K;
constexpr int G2F_KT     = 32;
constexpr int G2F_N_DIM  = 7168;
constexpr int G2F_K_DIM  = 2048;
constexpr int G2F_ELEM   = 2;

constexpr int G2FS_AS       = 0;
constexpr int G2FS_WS       = G2FS_AS + 2 * G2F_BM * G2F_BK * G2F_ELEM;
constexpr int G2FS_MBAR_TMA = G2FS_WS + 2 * G2F_BN * G2F_BK * G2F_ELEM;
constexpr int G2FS_MBAR_MMA = G2FS_MBAR_TMA + 16;
constexpr int G2FS_TIDS     = G2FS_MBAR_MMA + 8;
constexpr int G2FS_TMEM     = G2FS_TIDS + G2F_BM * (int)sizeof(int);
constexpr int G2FS_TOTAL    = G2FS_TMEM + 4;

__global__ void __launch_bounds__(128, 1)
fused_f16_gemm2_scatter_kernel(
    const int*     __restrict__ token_ids,
    const float*   __restrict__ token_w,
    float*         __restrict__ output,
    int Tk, int T,
    const CUtensorMap* __restrict__ tma_C,
    const CUtensorMap* __restrict__ tma_W2)
{
  extern __shared__ __align__(128) uint8_t smem[];
  uint8_t*  As_raw   = smem + G2FS_AS;
  uint8_t*  Ws_raw   = smem + G2FS_WS;
  uint64_t* mbar_tma = (uint64_t*)(smem + G2FS_MBAR_TMA);
  uint64_t* mbar_mma = (uint64_t*)(smem + G2FS_MBAR_MMA);
  int*      s_tids   = (int*)(smem + G2FS_TIDS);
  uint32_t* stm_d    = (uint32_t*)(smem + G2FS_TMEM);

  const int tid  = threadIdx.x;
  const int wid  = tid / 32;
  const int lid  = tid & 31;
  const bool is_prod = (wid == 0 && lid == 0);
  const bool is_mma  = (wid == 1 && lid == 0);
  const int m_base = blockIdx.y * G2F_BM;
  const int n_base = blockIdx.x * G2F_BN;

  if (m_base >= Tk || n_base >= G2F_N_DIM) return;

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

  uint32_t tmem_d;
  if (wid == 0) {
    uint32_t a = smem_to_cta_addr(stm_d);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n" :: "r"(a) : "memory");
  }
  __syncthreads();
  tmem_d = *stm_d;

  if (tid < G2F_BM) {
    int mg = m_base + tid;
    s_tids[tid] = (mg < Tk) ? token_ids[mg] : 0;
  }

  if (is_prod) {
    uint32_t t0 = smem_to_cta_addr(&mbar_tma[0]);
    uint32_t t1 = smem_to_cta_addr(&mbar_tma[1]);
    uint32_t m0 = smem_to_cta_addr(&mbar_mma[0]);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(t0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(t1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(m0), "r"(G2F_KINNER));
  }
  __syncthreads();

  float acc[G2F_BN];
  #pragma unroll
  for (int i = 0; i < G2F_BN; i++) acc[i] = 0.0f;

  int tma_ph[2] = {0, 0};
  int mma_phase = 0;
  constexpr uint32_t G2F_TMA_BYTES = G2F_BM * G2F_BK * G2F_ELEM + G2F_BN * G2F_BK * G2F_ELEM;

  #define G2F_AS(b) (As_raw + (b) * G2F_BM * G2F_BK * G2F_ELEM)
  #define G2F_WS(b) (Ws_raw + (b) * G2F_BN * G2F_BK * G2F_ELEM)

  if (is_prod) {
    uint32_t mb = smem_to_cta_addr(&mbar_tma[0]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(mb), "r"(G2F_TMA_BYTES));
    uint32_t ad = smem_to_cta_addr(G2F_AS(0));
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];\n"
      :: "r"(ad), "l"(tma_C), "r"(0), "r"(m_base), "r"(0), "r"(mb) : "memory");
    uint32_t wd = smem_to_cta_addr(G2F_WS(0));
    asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];\n"
      :: "r"(wd), "l"(tma_W2), "r"(0), "r"(n_base), "r"(0), "r"(mb) : "memory");
  }

  constexpr uint32_t g2f_idesc =
      (0u << 0) | (0u << 2) | (1u << 4) | (0u << 7) | (0u << 10)
    | (0u << 13) | (0u << 14) | (0u << 15) | (0u << 16)
    | ((G2F_BN >> 3) << 17) | (0u << 23) | ((G2F_BM >> 4) << 24) | (0u << 29);

  for (int kt = 0; kt < G2F_KT; kt++) {
    const int buf = kt & 1, nb = 1 - buf;

    if (is_prod) {
      uint32_t ma = smem_to_cta_addr(&mbar_tma[buf]);
      uint32_t p = tma_ph[buf];
      asm volatile("{\n .reg .pred p;\n WTMA_G2F_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n @!p bra WTMA_G2F_%=;\n}\n"
        :: "r"(ma), "r"(p));
      tma_ph[buf] ^= 1;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    if (kt + 1 < G2F_KT && is_prod) {
      uint32_t nm = smem_to_cta_addr(&mbar_tma[nb]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" :: "r"(nm), "r"(G2F_TMA_BYTES));
      int nk_slice = (kt + 1) * (G2F_BK / 8);
      uint32_t ad = smem_to_cta_addr(G2F_AS(nb));
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];\n"
        :: "r"(ad), "l"(tma_C), "r"(0), "r"(m_base), "r"(nk_slice), "r"(nm) : "memory");
      uint32_t wd = smem_to_cta_addr(G2F_WS(nb));
      asm volatile("cp.async.bulk.tensor.3d.shared::cta.global.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3,%4}],[%5];\n"
        :: "r"(wd), "l"(tma_W2), "r"(0), "r"(n_base), "r"(nk_slice), "r"(nm) : "memory");
    }

    uint32_t ab = smem_to_cta_addr(G2F_AS(buf));
    uint32_t wb = smem_to_cta_addr(G2F_WS(buf));
    #pragma unroll
    for (int ki = 0; ki < G2F_KINNER; ki++) {
      uint64_t ad = make_smem_desc_f16(ab + ki * G2F_BM * 32, G2F_BM);
      uint64_t bd = make_smem_desc_f16(wb + ki * G2F_BN * 32, G2F_BN);
      if (is_mma) {
        int ac = (ki > 0) ? 1 : 0;
        asm volatile(
          "{\n .reg .pred p;\n setp.ne.b32 p,%4,0;\n"
          "tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}\n"
          :: "r"(tmem_d), "l"(ad), "l"(bd), "r"(g2f_idesc), "r"(ac));
        uint32_t mm = smem_to_cta_addr(&mbar_mma[0]);
        asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n" :: "r"(mm) : "memory");
      }
    }

    if (is_mma) {
      uint32_t mm = smem_to_cta_addr(&mbar_mma[0]);
      uint32_t p = mma_phase;
      asm volatile("{\n .reg .pred p;\n WMMA_G2F_%=: mbarrier.try_wait.parity.shared::cta.b64 p,[%0],%1;\n @!p bra WMMA_G2F_%=;\n}\n"
        :: "r"(mm), "r"(p));
      mma_phase ^= 1;
    }
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    #pragma unroll
    for (int c = 0; c < G2F_BN; c += 4) {
      uint32_t v0, v1, v2, v3;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0,%1,%2,%3},[%4];\n"
        : "=r"(v0),"=r"(v1),"=r"(v2),"=r"(v3) : "r"(tmem_d + c));
      asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
      acc[c]   += __uint_as_float(v0);
      acc[c+1] += __uint_as_float(v1);
      acc[c+2] += __uint_as_float(v2);
      acc[c+3] += __uint_as_float(v3);
    }
  }
  #undef G2F_AS
  #undef G2F_WS

  {
    const int lr = m_base + tid;
    if (lr < Tk) {
      int tr = s_tids[tid];
      #pragma unroll
      for (int c = 0; c < G2F_BN; c += 4) {
        int gn = n_base + c;
        if (gn     < G2F_N_DIM) output[tr * G2F_N_DIM + gn]     += acc[c];
        if (gn + 1 < G2F_N_DIM) output[tr * G2F_N_DIM + gn + 1] += acc[c+1];
        if (gn + 2 < G2F_N_DIM) output[tr * G2F_N_DIM + gn + 2] += acc[c+2];
        if (gn + 3 < G2F_N_DIM) output[tr * G2F_N_DIM + gn + 3] += acc[c+3];
      }
    }
  }

  __syncthreads();
  if (wid == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(tmem_d));
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 232;\n");
}

void launch_swiglu_to_f16(
    const float* G1, const float* row_w, int Tk,
    uint16_t* C_f16, cudaStream_t stream)
{
  nvtxRangePushA("launch_swiglu_to_f16");
  if (Tk <= 0) { nvtxRangePop(); return; }
  int total = Tk * INTERMEDIATE_SIZE;
  int threads = 256;
  int blocks = (total + threads - 1) / threads;
  swiglu_to_f16_kernel<<<blocks, threads, 0, stream>>>(G1, row_w, Tk, C_f16);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_fp8_dequant_to_f16(
    const uint8_t* M_fp8, const float* S, int rows, int cols,
    int S_rows, int S_cols, uint16_t* M_f16, cudaStream_t stream)
{
  nvtxRangePushA("launch_fp8_dequant_to_f16");
  dim3 grid(S_cols, S_rows);
  dim3 block(32, 8);
  fp8_dequant_to_f16_kernel<<<grid, block, 0, stream>>>(M_fp8, S, cols, S_cols, M_f16);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_fused_f16_gemm2(
    const uint16_t* C_f16,
    const int*      token_ids,
    const float*    token_w,
    const uint16_t* w2_f16,
    float*          output,
    int Tk, int T,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_fused_f16_gemm2");
  if (Tk <= 0) { nvtxRangePop(); return; }
  auto encodeFn = get_cuTensorMapEncodeTiled();

  CUtensorMap tma_C_host;
  {
    cuuint64_t gd[3]  = {8, (cuuint64_t)Tk, (cuuint64_t)(G2F_K_DIM / 8)};
    cuuint64_t gs[2]  = {(cuuint64_t)G2F_K_DIM * G2F_ELEM, 16};
    cuuint32_t bd[3]  = {8, (cuuint32_t)G2F_BM, (cuuint32_t)(G2F_BK / 8)};
    cuuint32_t es[3]  = {1, 1, 1};
    CU_CHECK(encodeFn(&tma_C_host, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, (void*)C_f16,
      gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA));
  }
  CUtensorMap tma_W2_host;
  {
    cuuint64_t gd[3]  = {8, (cuuint64_t)G2F_N_DIM, (cuuint64_t)(G2F_K_DIM / 8)};
    cuuint64_t gs[2]  = {(cuuint64_t)G2F_K_DIM * G2F_ELEM, 16};
    cuuint32_t bd[3]  = {8, (cuuint32_t)G2F_BN, (cuuint32_t)(G2F_BK / 8)};
    cuuint32_t es[3]  = {1, 1, 1};
    CU_CHECK(encodeFn(&tma_W2_host, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 3, (void*)w2_f16,
      gd, gs, bd, es, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  static CUtensorMap* d_g2f_tma = nullptr;
  if (!d_g2f_tma) CUDA_CHECK(cudaMalloc(&d_g2f_tma, 2 * sizeof(CUtensorMap)));
  CUtensorMap h[2] = {tma_C_host, tma_W2_host};
  CUDA_CHECK(cudaMemcpyAsync(d_g2f_tma, h, 2 * sizeof(CUtensorMap), cudaMemcpyHostToDevice, stream));

  static bool g2f_smem_set = false;
  if (!g2f_smem_set) {
    CUDA_CHECK(cudaFuncSetAttribute(fused_f16_gemm2_scatter_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize, G2FS_TOTAL));
    g2f_smem_set = true;
  }

  dim3 blk(128);
  dim3 grd((G2F_N_DIM + G2F_BN - 1) / G2F_BN, (Tk + G2F_BM - 1) / G2F_BM);
  fused_f16_gemm2_scatter_kernel<<<grd, blk, G2FS_TOTAL, stream>>>(
    token_ids, token_w, output, Tk, T, d_g2f_tma, d_g2f_tma + 1);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

// =====================================================================
// Fused SwiGLU + FP8 e4m3 quantize (prepares A operand for mxFP8 GEMM2)
// Input : G1      [Tk, 4096]  fp32
// Output: C_fp8   [Tk, 2048]  uint8 (e4m3)
//         C_scale [16, Tk]    fp32  (pure power-of-2, E8M0 compatible)
//
// Layout: block(32), grid(16, Tk). Each CTA processes 1 row × 128 cols.
//   - Each thread handles 4 consecutive elements (float4 LD from G1).
//   - Warp-level amax via shfl_xor; E8M0 scale = 2^ceil(log2(amax/448)).
//   - Quantize via cvt.rn.satfinite.e4m3x2.f32 → packed 4×fp8 in uint32.
// =====================================================================
__global__ void swiglu_quantize_kernel(
    const float* __restrict__ G1,
    int Tk,
    uint8_t* __restrict__ C_fp8,
    float* __restrict__ C_scale)
{
  const int col_block = blockIdx.x;   // 0..15
  const int row       = blockIdx.y;   // 0..Tk-1
  const int tid       = threadIdx.x;  // 0..31
  if (row >= Tk) return;

  const int col_base  = col_block * 128 + tid * 4;
  const float* g1_row = G1 + row * GEMM1_OUT_SIZE;

  float4 x1 = *reinterpret_cast<const float4*>(g1_row + col_base);
  float4 x2 = *reinterpret_cast<const float4*>(g1_row + col_base + INTERMEDIATE_SIZE);

  float v0 = (x2.x / (1.0f + __expf(-x2.x))) * x1.x;
  float v1 = (x2.y / (1.0f + __expf(-x2.y))) * x1.y;
  float v2 = (x2.z / (1.0f + __expf(-x2.z))) * x1.z;
  float v3 = (x2.w / (1.0f + __expf(-x2.w))) * x1.w;

  // amax over this CTA's 128 elements (32 lanes × 4 elems)
  float absv = fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
  #pragma unroll
  for (int off = 16; off > 0; off >>= 1)
    absv = fmaxf(absv, __shfl_xor_sync(0xffffffffu, absv, off));
  const float block_max = absv;

  // scale_f = smallest 2^e such that block_max / 2^e <= FP8_E4M3_MAX (= 448.0)
  // Encode as pure power-of-2 fp32 (mantissa=0) so it is E8M0-compatible.
  float    ratio      = block_max * (1.0f / 448.0f);
  uint32_t rb         = __float_as_uint(ratio);
  uint32_t mant       = rb & 0x007FFFFFu;
  uint32_t exp_b      = (rb >> 23) & 0xFFu;
  if (mant != 0u) exp_b += 1u;                 // ceil
  exp_b = max(1u, min(exp_b, 254u));           // clamp: avoid subnormal / inf
  const uint32_t scale_bits = (exp_b << 23);
  const float    scale_f    = __uint_as_float(scale_bits);
  const float    inv_scale  = 1.0f / scale_f;

  // [DIAG-CVT] Quantize via __nv_cvt_float_to_fp8 (no PTX-operand-order ambiguity)
  // to validate that swiglu math + scale path is correct before optimizing the cvt itself.
  uint8_t b0 = __nv_cvt_float_to_fp8(v0 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b1 = __nv_cvt_float_to_fp8(v1 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b2 = __nv_cvt_float_to_fp8(v2 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint8_t b3 = __nv_cvt_float_to_fp8(v3 * inv_scale, __NV_SATFINITE, __NV_E4M3);
  uint32_t packed = (uint32_t)b0
                  | ((uint32_t)b1 <<  8)
                  | ((uint32_t)b2 << 16)
                  | ((uint32_t)b3 << 24);

  *reinterpret_cast<uint32_t*>(C_fp8 + row * INTERMEDIATE_SIZE + col_base) = packed;
  if (tid == 0) {
    C_scale[col_block * Tk + row] = scale_f;
  }
}

// =====================================================================
// [DIAG] Dequantize C_fp8 + C_scale (per-row × per-128-col block) → C_fp32
// Used to bisect bugs between swiglu_quantize and fused_blockscale_gemm2:
//   if dequant(C_fp8) + cuBLAS sgemm passes → bug is in fused_blockscale_gemm2
//   if it still fails                       → bug is in swiglu_quantize
// =====================================================================
__global__ void diag_dequant_c_kernel(
    const uint8_t* __restrict__ C_fp8,    // [Tk, 2048] uint8 e4m3
    const float*   __restrict__ C_scale,  // [16, Tk] fp32 per-row per-128-col-block
    float*         __restrict__ C_fp32,   // [Tk, 2048] fp32 output
    int Tk)
{
  const int col_block = blockIdx.x;   // 0..15
  const int row       = blockIdx.y;   // 0..Tk-1
  const int tid       = threadIdx.x;  // 0..31
  if (row >= Tk) return;

  const int col_base = col_block * 128 + tid * 4;
  const float scale  = C_scale[col_block * Tk + row];

  uint32_t packed = *reinterpret_cast<const uint32_t*>(
      C_fp8 + row * INTERMEDIATE_SIZE + col_base);
  float4 v = fp8x4_e4m3_to_f32x4(packed);
  v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
  *reinterpret_cast<float4*>(C_fp32 + row * INTERMEDIATE_SIZE + col_base) = v;
}

void launch_diag_dequant_c(
    const uint8_t* C_fp8, const float* C_scale,
    float* C_fp32, int Tk, cudaStream_t stream)
{
  if (Tk <= 0) return;
  dim3 block(32);
  dim3 grid(INTERMEDIATE_SIZE / 128, Tk);   // (16, Tk)
  diag_dequant_c_kernel<<<grid, block, 0, stream>>>(C_fp8, C_scale, C_fp32, Tk);
  CUDA_CHECK(cudaGetLastError());
}

void launch_swiglu_quantize(
    const float* G1, int Tk,
    uint8_t* C_fp8, float* C_scale,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_swiglu_quantize");
  if (Tk <= 0) { nvtxRangePop(); return; }
  dim3 block(32);
  dim3 grid(INTERMEDIATE_SIZE / 128, Tk);   // (16, Tk)
  swiglu_quantize_kernel<<<grid, block, 0, stream>>>(G1, Tk, C_fp8, C_scale);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

// =====================================================================
// Fused GEMM2 kernel: C_fp8 @ W2_fp8.T → O[Tk, 7168]
// mxFP8 MMA path replacing cuBLAS Sgemm + W2 dequant.
//
// Shapes:
//   C_fp8   [Tk, 2048]       uint8,  row-major
//   C_scale [16, Tk]         fp32,   per-row per-128-col-block scale
//   W2_fp8  [7168, 2048]     uint8,  row-major
//   W2_scale[56, 16]         fp32,   per-128x128-block scale
//   O       [Tk, 7168]       fp32,   row-major
//
// Tile: BM=BN=BK=128, MMA_K=32, KINNER=4, KT=16.
// =====================================================================
__global__ void __launch_bounds__(128, 1)
fused_blockscale_gemm2_kernel(
    const float*   __restrict__ C_scale,     // [16, Tk]
    const float*   __restrict__ W2_scale,    // [56, 16]
    float*         __restrict__ O,           // [Tk, 7168]
    int Tk,
    const CUtensorMap* __restrict__ tma_C,
    const CUtensorMap* __restrict__ tma_W2)
{
  extern __shared__ __align__(128) uint8_t smem[];

  uint8_t*  As_raw       = smem + G2_SMEM_AS;
  uint8_t*  Ws_raw       = smem + G2_SMEM_WS;
  uint64_t* mbar_tma     = (uint64_t*)(smem + G2_SMEM_MBAR_TMA);
  uint64_t* mbar_mma_ptr = (uint64_t*)(smem + G2_SMEM_MBAR_MMA);
  uint32_t* s_tmem_d     = (uint32_t*)(smem + G2_SMEM_TMEM);
  uint32_t* s_tmem_sa    = (uint32_t*)(smem + G2_SMEM_TMEM + 4);
  uint32_t* s_tmem_sb    = (uint32_t*)(smem + G2_SMEM_TMEM + 8);

  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const int m_base  = blockIdx.y * TC5_BM;
  const int n_base  = blockIdx.x * TC5_BN;

  if (m_base >= Tk || n_base >= TC5_G2_N) return;

  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n");

  uint32_t tmem_d;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_d);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_d = *s_tmem_d;

  uint32_t tmem_scale_a;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sa);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_a = *s_tmem_sa;

  uint32_t tmem_scale_b;
  if (warp_id == 0) {
    uint32_t sa = smem_to_cta_addr(s_tmem_sb);
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;\n" :: "r"(sa) : "memory");
  }
  __syncthreads();
  tmem_scale_b = *s_tmem_sb;

  if (tid == 0) {
    uint32_t tma0 = smem_to_cta_addr(&mbar_tma[0]);
    uint32_t tma1 = smem_to_cta_addr(&mbar_tma[1]);
    uint32_t mma  = smem_to_cta_addr(mbar_mma_ptr);
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma0));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;\n" :: "r"(tma1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], 4;\n" :: "r"(mma));
  }
  __syncthreads();

  float acc[TC5_BN];
  #pragma unroll
  for (int i = 0; i < TC5_BN; i++) acc[i] = 0.0f;

  int tma_phase[2] = {0, 0};
  int mma_phase    = 0;

  constexpr uint32_t TMA_BYTES = TC5_BM * TC5_BK + TC5_BN * TC5_BK;  // 32768

  #define G2_AS_BUF(b) (As_raw + (b) * TC5_BM * TC5_BK)
  #define G2_WS_BUF(b) (Ws_raw + (b) * TC5_BN * TC5_BK)

  // ---- Priming: TMA load kt=0 ----
  if (tid == 0) {
    uint32_t tma0_addr = smem_to_cta_addr(&mbar_tma[0]);
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
        :: "r"(tma0_addr), "r"(TMA_BYTES));

    uint32_t a_dst = smem_to_cta_addr(G2_AS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(a_dst), "l"(tma_C), "r"(0), "r"(m_base), "r"(tma0_addr)
      : "memory");

    uint32_t w_dst = smem_to_cta_addr(G2_WS_BUF(0));
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global"
      ".mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n"
      :: "r"(w_dst), "l"(tma_W2), "r"(0), "r"(n_base), "r"(tma0_addr)
      : "memory");
  }

  constexpr uint32_t idesc =
      (0u  <<  0)
    | (0u  <<  2)
    | (0u  <<  4)
    | (0u  <<  7)
    | (0u  << 10)
    | (0u  << 13)
    | (0u  << 14)
    | (0u  << 15)
    | (0u  << 16)
    | ((TC5_BN >> 3) << 17)
    | (1u  << 23)
    | (0u  << 24)
    | ((TC5_BM >> 7) << 27)
    | (0u  << 29)
    ;

  for (int kt = 0; kt < TC5_G2_KT; kt++) {
    const int buf      = kt & 1;
    const int next_buf = 1 - buf;

    // (a) Compute E8M0 scales + FP32 compensation
    float a_comp_val, w_comp;
    {
      int mg = m_base + tid;
      float sa = (mg < Tk) ? C_scale[kt * Tk + mg] : 1.0f;
      uint32_t sa_bits = __float_as_uint(sa);
      uint32_t sa_exp  = (sa_bits >> 23) & 0xFFu;
      uint32_t my_scale_a = (sa_exp == 255u) ? 0xFFu : sa_exp;

      if (sa_exp == 0u)        a_comp_val = sa * 0x1.0p127f;
      else if (sa_exp < 255u)  a_comp_val = __uint_as_float((sa_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     a_comp_val = 0.0f;

      int n_block = n_base / 128;
      float sb = W2_scale[n_block * TC5_G2_KT + kt];
      uint32_t sb_bits = __float_as_uint(sb);
      uint32_t sb_exp  = (sb_bits >> 23) & 0xFFu;
      uint32_t my_scale_b = (sb_exp == 255u) ? 0xFFu : sb_exp;

      if (sb_exp == 0u)        w_comp = sb * 0x1.0p127f;
      else if (sb_exp < 255u)  w_comp = __uint_as_float((sb_bits & 0x807FFFFFu) | 0x3F800000u);
      else                     w_comp = 0.0f;

      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a), "r"(my_scale_a));
      asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32 [%0], {%1,%2,%3,%4};\n"
        :: "r"(tmem_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b), "r"(my_scale_b));
    }
    asm volatile("tcgen05.wait::st.sync.aligned;\n" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;\n" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");

    // (b) Wait TMA
    {
      uint32_t mbar_addr = smem_to_cta_addr(&mbar_tma[buf]);
      uint32_t parity    = tma_phase[buf];
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_G2_TMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_G2_TMA_%=;\n"
        "}\n"
        :: "r"(mbar_addr), "r"(parity));
      tma_phase[buf] ^= 1;
    }
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");

    // (c) Prefetch next K-tile
    if (kt + 1 < TC5_G2_KT && tid == 0) {
      uint32_t next_mbar = smem_to_cta_addr(&mbar_tma[next_buf]);
      asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n"
          :: "r"(next_mbar), "r"(TMA_BYTES));

      int next_k = (kt + 1) * TC5_BK;
      uint32_t a_dst = smem_to_cta_addr(G2_AS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(a_dst), "l"(tma_C), "r"(next_k), "r"(m_base), "r"(next_mbar)
        : "memory");

      uint32_t w_dst = smem_to_cta_addr(G2_WS_BUF(next_buf));
      asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];\n"
        :: "r"(w_dst), "l"(tma_W2), "r"(next_k), "r"(n_base), "r"(next_mbar)
        : "memory");
    }

    // (d) MMA
    uint32_t a_smem_base = smem_to_cta_addr(G2_AS_BUF(buf));
    uint32_t w_smem_base = smem_to_cta_addr(G2_WS_BUF(buf));

    #pragma unroll
    for (int ki = 0; ki < TC5_KINNER; ki++) {
      uint64_t a_desc = make_smem_desc_swizzled(a_smem_base + ki * TC5_MMA_K);
      uint64_t b_desc = make_smem_desc_swizzled(w_smem_base + ki * TC5_MMA_K);

      if (tid == 0) {
        int accumulate = (ki > 0) ? 1 : 0;
        asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %6, 0;\n"
          "  tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X "
          "  [%0], %1, %2, %3, [%4], [%5], p;\n"
          "}\n"
          :: "r"(tmem_d), "l"(a_desc), "l"(b_desc), "r"(idesc),
             "r"(tmem_scale_a), "r"(tmem_scale_b), "r"(accumulate));

        uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
        asm volatile(
          "tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];\n"
          :: "r"(mma_mbar) : "memory");
      }
    }

    // (e) Wait MMA
    {
      uint32_t mma_mbar = smem_to_cta_addr(mbar_mma_ptr);
      uint32_t parity = mma_phase;
      asm volatile(
        "{\n"
        "  .reg .pred p;\n"
        "  WAIT_G2_MMA_%=:\n"
        "  mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
        "  @!p bra WAIT_G2_MMA_%=;\n"
        "}\n"
        :: "r"(mma_mbar), "r"(parity));
      mma_phase ^= 1;
    }

    // (f) Read TMEM + compensation
    float my_comp = a_comp_val * w_comp;
    asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
    {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        uint32_t v0, v1, v2, v3;
        asm volatile(
          "tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];\n"
          : "=r"(v0), "=r"(v1), "=r"(v2), "=r"(v3)
          : "r"(tmem_d + col));
        asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");

        acc[col]     += __uint_as_float(v0) * my_comp;
        acc[col + 1] += __uint_as_float(v1) * my_comp;
        acc[col + 2] += __uint_as_float(v2) * my_comp;
        acc[col + 3] += __uint_as_float(v3) * my_comp;
      }
    }
  } // end K-tile loop

  #undef G2_AS_BUF
  #undef G2_WS_BUF

  // ---- Write O [Tk, 7168] ----
  {
    const int out_row = m_base + tid;
    if (out_row < Tk) {
      #pragma unroll
      for (int col = 0; col < TC5_BN; col += 4) {
        int gn0 = n_base + col;
        if (gn0     < TC5_G2_N) O[out_row * TC5_G2_N + gn0]     = acc[col];
        if (gn0 + 1 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 1] = acc[col + 1];
        if (gn0 + 2 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 2] = acc[col + 2];
        if (gn0 + 3 < TC5_G2_N) O[out_row * TC5_G2_N + gn0 + 3] = acc[col + 3];
      }
    }
  }

  __syncthreads();
  if (warp_id == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;\n" :: "r"(tmem_d));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_a));
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;\n"  :: "r"(tmem_scale_b));
  }
  asm volatile("setmaxnreg.dec.sync.aligned.u32 232;\n");
}

void launch_fused_blockscale_gemm2(
    const uint8_t* C_fp8,
    const float*   C_scale,
    const uint8_t* W2_fp8,
    const float*   W2_scale,
    float*         O,
    int Tk,
    cudaStream_t stream)
{
  nvtxRangePushA("launch_fused_blockscale_gemm2");
  if (Tk <= 0) { nvtxRangePop(); return; }

  auto encodeFn = get_cuTensorMapEncodeTiled();

  // --- CUtensorMap for C: [Tk, 2048] uint8, tile [128, 128] ---
  CUtensorMap tma_C_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_G2_K, (cuuint64_t)Tk};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_G2_K};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BM};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_C_host,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)C_fp8,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  // --- CUtensorMap for W2: [7168, 2048] uint8, tile [128, 128] ---
  CUtensorMap tma_W2_host;
  {
    cuuint64_t globalDim[2]     = {(cuuint64_t)TC5_G2_K, (cuuint64_t)TC5_G2_N};
    cuuint64_t globalStrides[1] = {(cuuint64_t)TC5_G2_K};
    cuuint32_t boxDim[2]        = {(cuuint32_t)TC5_BK, (cuuint32_t)TC5_BN};
    cuuint32_t elemStrides[2]   = {1, 1};
    CU_CHECK(encodeFn(
        &tma_W2_host,
        CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, (void*)W2_fp8,
        globalDim, globalStrides, boxDim, elemStrides,
        CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  }

  static CUtensorMap* d_tma_buf = nullptr;
  if (!d_tma_buf) {
    CUDA_CHECK(cudaMalloc(&d_tma_buf, 2 * sizeof(CUtensorMap)));
  }
  CUtensorMap h_tma[2] = {tma_C_host, tma_W2_host};
  CUDA_CHECK(cudaMemcpyAsync(d_tma_buf, h_tma, 2 * sizeof(CUtensorMap),
                              cudaMemcpyHostToDevice, stream));

  static bool smem_attr_set = false;
  if (!smem_attr_set) {
    CUDA_CHECK(cudaFuncSetAttribute(
        fused_blockscale_gemm2_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, G2_SMEM_TOTAL));
    smem_attr_set = true;
  }

  dim3 block(128);
  dim3 grid(
    (TC5_G2_N + TC5_BN - 1) / TC5_BN,
    (Tk       + TC5_BM - 1) / TC5_BM
  );
  fused_blockscale_gemm2_kernel<<<grid, block, G2_SMEM_TOTAL, stream>>>(
      C_scale, W2_scale, O, Tk,
      d_tma_buf, d_tma_buf + 1);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

// =====================================================================
// Below: unchanged kernels (routing, scale, SwiGLU, gather, accum)
// =====================================================================

__device__ __forceinline__ float warp_max(float v) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    float other = __shfl_down_sync(0xffffffffu, v, offset);
    v = fmaxf(v, other);
  }
  return v;
}

__global__ void noaux_routing_topk8_kernel(
    const float* __restrict__ logits,
    const float* __restrict__ bias,
    int T, float routed_scaling_factor,
    int* __restrict__ topk_idx,
    float* __restrict__ topk_w) {

  __shared__ float group_scores[ROUTE_NUM_GROUP];
  __shared__ unsigned int keep_group_mask;
  __shared__ float warpCandVal[ROUTE_NUM_GROUP * ROUTE_TOP_K];
  __shared__ int   warpCandIdx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
  __shared__ float warpCandSNoBias[ROUTE_NUM_GROUP * ROUTE_TOP_K];

  int t = blockIdx.x;
  if (t >= T) return;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int e = warp * ROUTE_GROUP_SIZE + lane;

  float l = logits[t * NUM_EXPERTS_GLOBAL + e];
  float s = 1.f / (1.f + __expf(-l));
  float sb = s + bias[e];

  float v = sb;
  float m1 = warp_max(v);
  unsigned mask1 = __ballot_sync(0xffffffffu, v == m1);
  int idx1_lane = __ffs(mask1) - 1;
  float v2 = (lane == idx1_lane) ? -CUDART_INF_F : v;
  float m2 = warp_max(v2);
  if (lane == 0) group_scores[warp] = m1 + m2;
  __syncthreads();

  if (threadIdx.x == 0) {
    float temp_scores[ROUTE_NUM_GROUP];
    #pragma unroll
    for (int g = 0; g < ROUTE_NUM_GROUP; ++g) temp_scores[g] = group_scores[g];
    unsigned int mask_bits = 0u;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOPK_GROUP; ++j) {
      int best = 0; float bestv = temp_scores[0];
      #pragma unroll
      for (int g = 1; g < ROUTE_NUM_GROUP; ++g)
        if (temp_scores[g] > bestv) { bestv = temp_scores[g]; best = g; }
      mask_bits |= (1u << best);
      temp_scores[best] = -CUDART_INF_F;
    }
    keep_group_mask = mask_bits;
  }
  __syncthreads();

  bool keep = ((keep_group_mask >> warp) & 1u) != 0u;
  float cur = keep ? sb : -CUDART_INF_F;

  #pragma unroll
  for (int j = 0; j < ROUTE_TOP_K; ++j) {
    float m = warp_max(cur);
    unsigned msk = __ballot_sync(0xffffffffu, cur == m);
    int max_lane = __ffs(msk) - 1;
    float s_no_bias_sel = __shfl_sync(0xffffffffu, s, max_lane);
    if (lane == 0) {
      int base = warp * ROUTE_TOP_K + j;
      warpCandVal[base] = m;
      warpCandIdx[base] = warp * ROUTE_GROUP_SIZE + max_lane;
      warpCandSNoBias[base] = s_no_bias_sel;
    }
    if (lane == max_lane) cur = -CUDART_INF_F;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float temp_val[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    int   temp_idx[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    float temp_snb[ROUTE_NUM_GROUP * ROUTE_TOP_K];
    #pragma unroll
    for (int i = 0; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i) {
      temp_val[i] = warpCandVal[i]; temp_idx[i] = warpCandIdx[i]; temp_snb[i] = warpCandSNoBias[i];
    }
    float sel_s[ROUTE_TOP_K]; int sel_idx[ROUTE_TOP_K];
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      int best_i = 0; float best_v = temp_val[0];
      #pragma unroll
      for (int i = 1; i < ROUTE_NUM_GROUP * ROUTE_TOP_K; ++i)
        if (temp_val[i] > best_v) { best_v = temp_val[i]; best_i = i; }
      sel_idx[j] = temp_idx[best_i]; sel_s[j] = temp_snb[best_i];
      temp_val[best_i] = -CUDART_INF_F;
    }
    float sumw = 0.f;
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) sumw += sel_s[j];
    sumw = fmaxf(sumw, 1e-20f);
    #pragma unroll
    for (int j = 0; j < ROUTE_TOP_K; ++j) {
      topk_idx[t * ROUTE_TOP_K + j] = sel_idx[j];
      topk_w[t * ROUTE_TOP_K + j] = (sel_s[j] / sumw) * routed_scaling_factor;
    }
  }
}

__global__ void apply_hidden_block_scale_kernel(
    float* __restrict__ A, const float* __restrict__ S, int T, int H) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int N = T * H;
  for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
    int t = i / H; int h = i - t * H; int hb = h >> 7;
    A[i] *= S[hb * T + t];
  }
}

__global__ void apply_block_scale_128x128_kernel(
    float* __restrict__ M, int rows, int cols,
    const float* __restrict__ S, int Sb_rows, int Sb_cols) {
  int blk_row = blockIdx.y; int blk_col = blockIdx.x;
  float scale = S[blk_row * Sb_cols + blk_col];
  int row_base = blk_row * BLOCK_SIZE_128; int col_base = blk_col * BLOCK_SIZE_128;
  int tx = threadIdx.x; int ty = threadIdx.y;
  for (int r = ty; r < BLOCK_SIZE_128; r += blockDim.y) {
    int row = row_base + r; float* row_ptr = M + row * cols;
    for (int c = tx; c < BLOCK_SIZE_128; c += blockDim.x)
      row_ptr[col_base + c] *= scale;
  }
}

// 4×FP8 E4M3 (packed u32) → float4, via PTX cvt.rn.f16x2.e4m3x2 + cvt.f32.f16
// Matches PyTorch `.to(torch::kFloat32)` for e4m3fn (saturating / no ±inf).
__device__ __forceinline__ float4 fp8x4_e4m3_to_f32x4(uint32_t packed) {
  uint32_t lo_h2, hi_h2;
  asm("{\n"
      "  .reg .b16 lo16, hi16;\n"
      "  mov.b32 {lo16, hi16}, %2;\n"
      "  cvt.rn.f16x2.e4m3x2 %0, lo16;\n"
      "  cvt.rn.f16x2.e4m3x2 %1, hi16;\n"
      "}\n"
      : "=r"(lo_h2), "=r"(hi_h2)
      : "r"(packed));

  float4 r;
  asm("{\n"
      "  .reg .b16 h0, h1;\n"
      "  mov.b32 {h0, h1}, %2;\n"
      "  cvt.f32.f16 %0, h0;\n"
      "  cvt.f32.f16 %1, h1;\n"
      "}\n"
      : "=f"(r.x), "=f"(r.y) : "r"(lo_h2));
  asm("{\n"
      "  .reg .b16 h0, h1;\n"
      "  mov.b32 {h0, h1}, %2;\n"
      "  cvt.f32.f16 %0, h0;\n"
      "  cvt.f32.f16 %1, h1;\n"
      "}\n"
      : "=f"(r.z), "=f"(r.w) : "r"(hi_h2));
  return r;
}

// Fused: read FP8 + apply 128×128 block scale + write FP32 in one pass.
// Thread layout: block(32, 8) = 256 threads; each thread handles 16 rows × 4 cols = 64 elems,
// issuing 1× u32 LD and 1× float4 ST per row iteration (perfectly coalesced).
__global__ void fp8_to_fp32_block_scale_128x128_kernel(
    const uint8_t* __restrict__ M_fp8,
    float* __restrict__ M_fp32,
    int cols,
    const float* __restrict__ S,
    int Sb_cols) {
  const int blk_row = blockIdx.y;
  const int blk_col = blockIdx.x;
  const float scale = S[blk_row * Sb_cols + blk_col];

  const int row_base = blk_row * BLOCK_SIZE_128;
  const int col_base = blk_col * BLOCK_SIZE_128;
  const int tx = threadIdx.x;  // 0..31
  const int ty = threadIdx.y;  // 0..7

  const int col_offset = col_base + tx * 4;  // 4 fp8 bytes per thread per row

  #pragma unroll
  for (int r = ty; r < BLOCK_SIZE_128; r += 8) {
    const int row = row_base + r;
    const uint8_t* in_ptr  = M_fp8  + row * cols + col_offset;
    float*         out_ptr = M_fp32 + row * cols + col_offset;

    uint32_t packed = *reinterpret_cast<const uint32_t*>(in_ptr);
    float4 v = fp8x4_e4m3_to_f32x4(packed);
    v.x *= scale; v.y *= scale; v.z *= scale; v.w *= scale;
    *reinterpret_cast<float4*>(out_ptr) = v;
  }
}

__global__ void count_local_assignments_kernel(
    const int* __restrict__ topk_idx, int T, int local_expert_offset,
    int* __restrict__ counts) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int le = topk_idx[base + k] - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) atomicAdd(&counts[le], 1);
  }
}

__global__ void fill_local_assignments_kernel(
    const int* __restrict__ topk_idx, const float* __restrict__ topk_w,
    int T, int local_expert_offset,
    int* __restrict__ offsets_inout,
    int* __restrict__ token_ids_out, float* __restrict__ token_w_out) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  int base = t * ROUTE_TOP_K;
  #pragma unroll
  for (int k = 0; k < ROUTE_TOP_K; ++k) {
    int le = topk_idx[base + k] - local_expert_offset;
    if ((unsigned)le < (unsigned)NUM_LOCAL_EXPERTS) {
      int pos = atomicAdd(&offsets_inout[le], 1);
      token_ids_out[pos] = t;
      token_w_out[pos] = topk_w[base + k];
    }
  }
}

__global__ void gather_rows_kernel(
    const float* __restrict__ A, const int* __restrict__ token_ids,
    int, int Tk, int H, float* __restrict__ A_out) {
  int row = blockIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  A_out[row * H + col] = A[token_ids[row] * H + col];
}

__global__ void swiglu_kernel(
    const float* __restrict__ G1, int Tk, float* __restrict__ C) {
  int row = blockIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= INTERMEDIATE_SIZE) return;
  const float* g1_row = G1 + row * GEMM1_OUT_SIZE;
  float x1 = g1_row[col]; float x2 = g1_row[col + INTERMEDIATE_SIZE];
  C[row * INTERMEDIATE_SIZE + col] = (x2 / (1.0f + __expf(-x2))) * x1;
}

__global__ void accumulate_weighted_add_kernel(
    const float* __restrict__ O, const int* __restrict__ token_ids,
    const float* __restrict__ weights, int Tk, int H,
    float* __restrict__ output) {
  int row = blockIdx.y; int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= Tk || col >= H) return;
  output[token_ids[row] * H + col] += O[row * H + col] * weights[row];
}

// =====================================================================
// Launchers (unchanged interfaces)
// =====================================================================

void launch_noaux_routing_topk8(
    const float* routing_logits, const float* routing_bias,
    int T, float routed_scaling_factor,
    int* topk_idx, float* topk_w, cudaStream_t stream) {
  nvtxRangePushA("launch_noaux_routing_topk8");
  dim3 block(ROUTE_NUM_GROUP * 32); dim3 grid(T);
  noaux_routing_topk8_kernel<<<grid, block, 0, stream>>>(
      routing_logits, routing_bias, T, routed_scaling_factor, topk_idx, topk_w);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_apply_hidden_block_scale(
    float* A_fp32, const float* hs_scale, int T, cudaStream_t stream) {
  nvtxRangePushA("launch_apply_hidden_block_scale");
  int H = HIDDEN_SIZE;
  int64_t N64 = static_cast<int64_t>(T) * H;
  int threads = 256; int blocks = static_cast<int>((N64 + threads - 1) / threads);
  blocks = max(1, min(blocks, 65535));
  apply_hidden_block_scale_kernel<<<blocks, threads, 0, stream>>>(A_fp32, hs_scale, T, H);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_apply_block_scale_128x128(
    float* M, int rows, int cols,
    const float* S, int S_rows, int S_cols, cudaStream_t stream) {
  nvtxRangePushA("launch_apply_block_scale_128x128");
  dim3 grid(S_cols, S_rows); dim3 block(32, 8);
  apply_block_scale_128x128_kernel<<<grid, block, 0, stream>>>(M, rows, cols, S, S_rows, S_cols);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_fp8_to_fp32_block_scale_128x128(
    const uint8_t* M_fp8, float* M_fp32,
    int rows, int cols,
    const float* S, int S_rows, int S_cols, cudaStream_t stream) {
  nvtxRangePushA("launch_fp8_to_fp32_block_scale_128x128");
  dim3 grid(S_cols, S_rows); dim3 block(32, 8);
  fp8_to_fp32_block_scale_128x128_kernel<<<grid, block, 0, stream>>>(
      M_fp8, M_fp32, cols, S, S_cols);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_count_local_assignments(
    const int* topk_idx, int T, int local_expert_offset,
    int* counts, cudaStream_t stream) {
  nvtxRangePushA("launch_count_local_assignments");
  int threads = 256; int blocks = (T + threads - 1) / threads;
  count_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, T, local_expert_offset, counts);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_fill_local_assignments(
    const int* topk_idx, const float* topk_w, int T, int local_expert_offset,
    int* offsets_inout, int* token_ids_out, float* token_w_out, cudaStream_t stream) {
  nvtxRangePushA("launch_fill_local_assignments");
  int threads = 256; int blocks = (T + threads - 1) / threads;
  fill_local_assignments_kernel<<<blocks, threads, 0, stream>>>(
      topk_idx, topk_w, T, local_expert_offset, offsets_inout, token_ids_out, token_w_out);
  CUDA_CHECK(cudaGetLastError());
  nvtxRangePop();
}

void launch_gather_rows(
    const float* A, const int* token_ids, int, int Tk, int H,
    float* A_out, cudaStream_t stream) {
  nvtxRangePushA("launch_gather_rows");
  dim3 block(256); dim3 grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    gather_rows_kernel<<<grid, block, 0, stream>>>(A, token_ids, 0, Tk, H, A_out);
    CUDA_CHECK(cudaGetLastError());
  }
  nvtxRangePop();
}

void launch_swiglu(const float* G1, int Tk, float* C, cudaStream_t stream) {
  nvtxRangePushA("launch_swiglu");
  dim3 block(256); dim3 grid((INTERMEDIATE_SIZE + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    swiglu_kernel<<<grid, block, 0, stream>>>(G1, Tk, C);
    CUDA_CHECK(cudaGetLastError());
  }
  nvtxRangePop();
}

void launch_accumulate_weighted_add(
    const float* O, const int* token_ids, const float* weights,
    int Tk, int H, float* output, cudaStream_t stream) {
  nvtxRangePushA("launch_accumulate_weighted_add");
  dim3 block(256); dim3 grid((H + block.x - 1) / block.x, Tk);
  if (Tk > 0) {
    accumulate_weighted_add_kernel<<<grid, block, 0, stream>>>(
        O, token_ids, weights, Tk, H, output);
    CUDA_CHECK(cudaGetLastError());
  }
  nvtxRangePop();
}
