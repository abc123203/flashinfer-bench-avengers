#include "kernel.h"

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <nvtx3/nvToolsExt.h>

torch::Tensor run(
    torch::Tensor routing_logits,
    torch::Tensor routing_bias,
    torch::Tensor hidden_states,
    torch::Tensor hidden_states_scale,
    torch::Tensor gemm1_weights,
    torch::Tensor gemm1_weights_scale,
    torch::Tensor gemm2_weights,
    torch::Tensor gemm2_weights_scale,
    int64_t local_expert_offset,
    double routed_scaling_factor
) {
  TORCH_CHECK(routing_logits.is_cuda(), "routing_logits must be CUDA");
  TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be CUDA");
  TORCH_CHECK(hidden_states_scale.is_cuda(), "hidden_states_scale must be CUDA");
  TORCH_CHECK(gemm1_weights.is_cuda() && gemm1_weights_scale.is_cuda(), "gemm1 weights must be CUDA");
  TORCH_CHECK(gemm2_weights.is_cuda() && gemm2_weights_scale.is_cuda(), "gemm2 weights must be CUDA");
  TORCH_CHECK(routing_bias.is_cuda(), "routing_bias must be CUDA");

  const int64_t T = routing_logits.size(0);
  TORCH_CHECK(routing_logits.size(1) == NUM_EXPERTS_GLOBAL);
  TORCH_CHECK(hidden_states.size(0) == T && hidden_states.size(1) == HIDDEN_SIZE);
  TORCH_CHECK(hidden_states_scale.size(0) == NUM_HIDDEN_BLOCKS && hidden_states_scale.size(1) == T);
  TORCH_CHECK(gemm1_weights.size(0) == NUM_LOCAL_EXPERTS && gemm1_weights.size(1) == GEMM1_OUT_SIZE && gemm1_weights.size(2) == HIDDEN_SIZE);
  TORCH_CHECK(gemm1_weights_scale.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, NUM_GEMM1_OUT_BLOCKS, NUM_HIDDEN_BLOCKS}));
  TORCH_CHECK(gemm2_weights.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE}));
  TORCH_CHECK(gemm2_weights_scale.sizes() == torch::IntArrayRef({NUM_LOCAL_EXPERTS, NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS}));
  TORCH_CHECK(routing_bias.size(0) == NUM_EXPERTS_GLOBAL);

  c10::cuda::CUDAGuard device_guard(routing_logits.device());
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  auto routing_bias_f32 = routing_bias.to(torch::kFloat32).contiguous();
  auto routing_logits_f32 = routing_logits.contiguous();
  auto hs_scale_c = hidden_states_scale.contiguous();
  auto gemm1_weights_c = gemm1_weights.contiguous();
  auto gemm1_weights_scale_c = gemm1_weights_scale.contiguous();

  // ========== Phase 1: Routing + assignments (all async on GPU) ==========
  auto topk_idx = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  auto topk_w = torch::empty({T, ROUTE_TOP_K}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  launch_noaux_routing_topk8(
      routing_logits_f32.data_ptr<float>(),
      routing_bias_f32.data_ptr<float>(),
      (int)T, static_cast<float>(routed_scaling_factor),
      topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(), stream);

  auto counts = torch::zeros({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  launch_count_local_assignments(topk_idx.data_ptr<int>(), (int)T, (int)local_expert_offset, counts.data_ptr<int>(), stream);

  auto expert_offsets = torch::empty({NUM_LOCAL_EXPERTS + 1}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  auto expert_Tk_dev = torch::empty({NUM_LOCAL_EXPERTS}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  launch_compute_offsets(counts.data_ptr<int>(), expert_offsets.data_ptr<int>(), expert_Tk_dev.data_ptr<int>(), stream);

  auto offsets_working = expert_offsets.slice(0, 0, NUM_LOCAL_EXPERTS).clone();
  int Tk_ub = (int)(T * ROUTE_TOP_K);
  auto all_token_ids = torch::empty({std::max(1, Tk_ub)}, torch::dtype(torch::kInt32).device(routing_logits.device()));
  auto all_token_wts = torch::empty({std::max(1, Tk_ub)}, torch::dtype(torch::kFloat32).device(routing_logits.device()));
  launch_fill_local_assignments(
      topk_idx.data_ptr<int>(), topk_w.data_ptr<float>(),
      (int)T, (int)local_expert_offset,
      offsets_working.data_ptr<int>(),
      all_token_ids.data_ptr<int>(), all_token_wts.data_ptr<float>(), stream);

  // ========== Phase 2: Grouped GEMM1 — all 32 experts, ONE launch ==========
  auto G1_all = torch::empty({std::max(1, Tk_ub), GEMM1_OUT_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));

  launch_grouped_gemm1(
      reinterpret_cast<const uint8_t*>(hidden_states.data_ptr()),
      hs_scale_c.data_ptr<float>(),
      all_token_ids.data_ptr<int>(),
      expert_offsets.data_ptr<int>(),
      expert_Tk_dev.data_ptr<int>(),
      gemm1_weights_scale_c.data_ptr<float>(),
      reinterpret_cast<const uint8_t*>(gemm1_weights_c.data_ptr()),
      G1_all.data_ptr<float>(),
      (int)T, Tk_ub, stream);

  // ========== Phase 3: Sync (latency hidden behind GEMM1 compute) ==========
  CUDA_CHECK(cudaStreamSynchronize(stream));
  auto counts_cpu = counts.cpu();
  auto counts_ptr = counts_cpu.data_ptr<int>();
  std::vector<int> h_counts(NUM_LOCAL_EXPERTS);
  int total_assign = 0, max_Tk = 0;
  for (int i = 0; i < NUM_LOCAL_EXPERTS; ++i) {
    h_counts[i] = counts_ptr[i];
    total_assign += h_counts[i];
    max_Tk = std::max(max_Tk, h_counts[i]);
  }
  std::vector<int> h_offsets(NUM_LOCAL_EXPERTS + 1, 0);
  for (int i = 0; i < NUM_LOCAL_EXPERTS; ++i) h_offsets[i + 1] = h_offsets[i] + h_counts[i];

  // ========== Phase 4: Per-expert FP16 GEMM2 pipeline (no cuBLAS) ==========
  auto output_f32 = torch::zeros({T, HIDDEN_SIZE}, torch::dtype(torch::kFloat32).device(routing_logits.device()));

  int Tk_max = std::max(1, max_Tk);
  auto C_f16 = torch::empty({Tk_max, INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat16).device(routing_logits.device()));
  auto w2_f16 = torch::empty({HIDDEN_SIZE, INTERMEDIATE_SIZE}, torch::dtype(torch::kFloat16).device(routing_logits.device()));

  for (int le = 0; le < NUM_LOCAL_EXPERTS; ++le) {
    int Tk = h_counts[le];
    if (Tk == 0) continue;
    int start = h_offsets[le];

    const int* d_token_ids_le = all_token_ids.data_ptr<int>() + start;
    const float* d_token_w_le = all_token_wts.data_ptr<float>() + start;
    float* g1_ptr = G1_all.data_ptr<float>() + (int64_t)start * GEMM1_OUT_SIZE;

    launch_swiglu_to_f16(
        g1_ptr, d_token_w_le, Tk,
        reinterpret_cast<uint16_t*>(C_f16.data_ptr()), stream);

    auto w2_fp8 = gemm2_weights.select(0, le).contiguous();
    auto s2 = gemm2_weights_scale.select(0, le).contiguous();
    launch_fp8_dequant_to_f16(
        reinterpret_cast<const uint8_t*>(w2_fp8.data_ptr()),
        s2.data_ptr<float>(), HIDDEN_SIZE, INTERMEDIATE_SIZE,
        NUM_HIDDEN_BLOCKS, NUM_INTERMEDIATE_BLOCKS,
        reinterpret_cast<uint16_t*>(w2_f16.data_ptr()), stream);

    launch_fused_f16_gemm2(
        reinterpret_cast<const uint16_t*>(C_f16.data_ptr()),
        d_token_ids_le, d_token_w_le,
        reinterpret_cast<const uint16_t*>(w2_f16.data_ptr()),
        output_f32.data_ptr<float>(),
        Tk, (int)T, stream);
  }

  return output_f32.to(torch::kBFloat16);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run,
        "moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048 (B200-optimized)");
}
