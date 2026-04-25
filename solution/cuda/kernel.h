#ifndef MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_
#define MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_

#include <cuda_runtime.h>
#include <cstdint>

// B200-tuned constants for this specialized kernel
static constexpr int HIDDEN_SIZE = 7168;        // H
static constexpr int INTERMEDIATE_SIZE = 2048;  // I
static constexpr int GEMM1_OUT_SIZE = 4096;     // 2 * I
static constexpr int NUM_EXPERTS_GLOBAL = 256;  // E_global
static constexpr int NUM_LOCAL_EXPERTS = 32;    // E_local
static constexpr int BLOCK_SIZE_128 = 128;

static constexpr int NUM_HIDDEN_BLOCKS = 56;         // H / 128
static constexpr int NUM_INTERMEDIATE_BLOCKS = 16;   // I / 128
static constexpr int NUM_GEMM1_OUT_BLOCKS = 32;      // (2*I)/128

// DeepSeek routing constants
static constexpr int ROUTE_TOP_K = 8;
static constexpr int ROUTE_NUM_GROUP = 8;
static constexpr int ROUTE_GROUP_SIZE = 32;    // NUM_EXPERTS_GLOBAL / ROUTE_NUM_GROUP
static constexpr int ROUTE_TOPK_GROUP = 4;

// Error check macro
#define CUDA_CHECK(status) \
  do { \
    cudaError_t err__ = (status); \
    if (err__ != cudaSuccess) { \
      fprintf(stderr, "CUDA Error %s at %s:%d\n", cudaGetErrorString(err__), __FILE__, __LINE__); \
    } \
  } while (0)

// Kernel launchers

// 1) No-aux routing with group-top2 and global top-k=8
void launch_noaux_routing_topk8(
    const float* routing_logits,   // [T, 256]
    const float* routing_bias,     // [256] (float32)
    int T,                         // seq_len
    float routed_scaling_factor,
    int* __restrict__ topk_idx,    // [T, 8] (int32)
    float* __restrict__ topk_w,    // [T, 8] (float32)
    cudaStream_t stream);

// 2) Hidden states block-scale application (after FP8 -> float32 conversion)
void launch_apply_hidden_block_scale(
    float* __restrict__ A_fp32,     // [T, H], in-place
    const float* __restrict__ hs_scale, // [H/128, T] contiguous
    int T,
    cudaStream_t stream);

// 3) Apply 128x128 block scale to 2D matrix (in-place)
void launch_apply_block_scale_128x128(
    float* __restrict__ M,          // [rows, cols], row-major
    int rows,                       // multiple of 128
    int cols,                       // multiple of 128
    const float* __restrict__ S,    // [rows/128, cols/128], row-major
    int S_rows,                     // rows/128
    int S_cols,                     // cols/128
    cudaStream_t stream);

// 3b) Fused: FP8 E4M3 → FP32 + 128x128 block scale (one-pass, eliminates fp32 GMEM round-trip)
void launch_fp8_to_fp32_block_scale_128x128(
    const uint8_t* __restrict__ M_fp8,  // [rows, cols] fp8_e4m3fn, row-major
    float* __restrict__ M_fp32,          // [rows, cols] fp32 output, row-major
    int rows,                            // multiple of 128
    int cols,                            // multiple of 128
    const float* __restrict__ S,         // [S_rows, S_cols], row-major
    int S_rows,                          // rows/128
    int S_cols,                          // cols/128
    cudaStream_t stream);

// 4) Count assignments per local expert
void launch_count_local_assignments(
    const int* __restrict__ topk_idx,  // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ counts,          // [32], zero-initialized
    cudaStream_t stream);

// 5) Fill flat assignment lists using prefix offsets (atomic on-device)
void launch_fill_local_assignments(
    const int* __restrict__ topk_idx,   // [T, 8]
    const float* __restrict__ topk_w,   // [T, 8]
    int T,
    int local_expert_offset,
    int* __restrict__ offsets_inout,    // [32], device-side running offsets (initialized with prefix "offsets")
    int* __restrict__ token_ids_out,    // [total_assignments]
    float* __restrict__ token_w_out,    // [total_assignments]
    cudaStream_t stream);

// Fused) Gather + BlockScale_A + BlockScale_W13 + GEMM1 in one kernel
void launch_fused_gather_blockscale_gemm1(
    const uint8_t* A_fp8,        // [T, 7168] FP8 raw bytes
    const float*   A_scale,      // [56, T]
    const int*     token_ids,    // [Tk]
    const uint8_t* W13_fp8,      // [4096, 7168] FP8 raw bytes
    const float*   W13_scale,    // [32, 56]
    float*         G1,           // [Tk, 4096] output
    int Tk, int T,
    cudaStream_t stream);

// Fused) SwiGLU + FP8 e4m3 quantize (prepares A operand for mxFP8 GEMM2)
//   G1[Tk, 4096] (fp32) -> C_fp8[Tk, 2048] (uint8) + C_scale[16, Tk] (fp32)
//   C_scale encodes a pure power-of-2 (E8M0 compatible) per-row per-128-col-block scale.
void launch_swiglu_quantize(
    const float*   G1,           // [Tk, 4096] fp32
    int            Tk,
    uint8_t*       C_fp8,        // [Tk, 2048] uint8 (e4m3)
    float*         C_scale,      // [16, Tk] fp32 (pow2, E8M0 compatible)
    cudaStream_t   stream);

// [DIAG] Dequantize C_fp8 + per-row per-128-col-block C_scale → C_fp32
// (used to bisect bug between swiglu_quantize and fused_blockscale_gemm2)
void launch_diag_dequant_c(
    const uint8_t* C_fp8,        // [Tk, 2048] uint8 (e4m3)
    const float*   C_scale,      // [16, Tk] fp32
    float*         C_fp32,       // [Tk, 2048] fp32 output
    int            Tk,
    cudaStream_t   stream);

// Fused) BlockScale_C + BlockScale_W2 + GEMM2 in one tcgen05 mxFP8 kernel
//   O[Tk, 7168] = C_fp8 @ W2_fp8.T  (with per-block scales C_scale / W2_scale)
//   Replaces cublasSgemm + 56MB fp32 W2 materialization.
void launch_fused_blockscale_gemm2(
    const uint8_t* C_fp8,        // [Tk, 2048] FP8 raw bytes
    const float*   C_scale,      // [16, Tk] fp32
    const uint8_t* W2_fp8,       // [7168, 2048] FP8 raw bytes
    const float*   W2_scale,     // [56, 16] fp32
    float*         O,            // [Tk, 7168] fp32 output
    int            Tk,
    cudaStream_t   stream);

// GPU prefix sum: counts[32] → offsets[33], expert_Tk[32]
void launch_compute_offsets(
    const int* counts, int* offsets, int* expert_Tk, cudaStream_t stream);

// Grouped GEMM1: all 32 experts in one launch
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
    cudaStream_t stream);

// Batched SwiGLU+Quantize: all rows in one launch
void launch_batched_swiglu_quantize(
    const float* G1_all, int Tk_ub,
    uint8_t* C_fp8_all, float* C_scale_all,
    cudaStream_t stream);

// Grouped GEMM2: all 32 experts in one launch
void launch_grouped_gemm2(
    const uint8_t* C_fp8_all,
    const float*   C_scale_all,
    const uint8_t* W2_fp8_base,
    const float*   W2_scale_base,
    float*         Otmp_all,
    const int*     expert_offsets,
    const int*     expert_Tk,
    int Tk_ub,
    cudaStream_t stream);

// Batched scatter-add: all 32 experts in one launch
void launch_batched_accumulate(
    const float* Otmp_all,
    const int*   all_token_ids,
    const float* all_token_wts,
    const int*   expert_offsets,
    const int*   expert_Tk,
    int Tk_ub, int H,
    float* output,
    cudaStream_t stream);

// FP16 GEMM2 pipeline (from solution6)
void launch_swiglu_to_f16(
    const float* G1, const float* row_w, int Tk,
    uint16_t* C_f16, cudaStream_t stream);

void launch_fp8_dequant_to_f16(
    const uint8_t* M_fp8, const float* S, int rows, int cols,
    int S_rows, int S_cols, uint16_t* M_f16, cudaStream_t stream);

void launch_fused_f16_gemm2(
    const uint16_t* C_f16,
    const int*      token_ids,
    const float*    token_w,
    const uint16_t* w2_f16,
    float*          output,
    int Tk, int T,
    cudaStream_t stream);

// 6) Gather rows from [T, H] by token_ids to a compact [Tk, H]
void launch_gather_rows(
    const float* __restrict__ A,      // [T, H]
    const int* __restrict__ token_ids,// [Tk]
    int T, int Tk, int H,
    float* __restrict__ A_out,        // [Tk, H]
    cudaStream_t stream);

// 7) SwiGLU on GEMM1 output: C = silu(G1[:, I:]) * G1[:, :I]
void launch_swiglu(
    const float* __restrict__ G1,   // [Tk, 4096]
    int Tk,
    float* __restrict__ C,          // [Tk, 2048]
    cudaStream_t stream);

// 8) Accumulate O[Tk,H] into output[T,H] by token_ids and weights (no atomics if sequential per expert)
void launch_accumulate_weighted_add(
    const float* __restrict__ O,        // [Tk, H]
    const int* __restrict__ token_ids,  // [Tk]
    const float* __restrict__ weights,  // [Tk]
    int Tk, int H,
    float* __restrict__ output,         // [T, H]
    cudaStream_t stream);

#endif // MOE_FP8_BLOCK_SCALE_DS_ROUTING_TOPK8_NG8_KG4_E32_H7168_I2048_KERNEL_H_