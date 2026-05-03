import os
import shutil
import torch
from torch.utils.cpp_extension import load

os.environ["TORCH_CUDA_ARCH_LIST"] = "10.0a"
shutil.rmtree(os.path.expanduser("~/.cache/torch_extensions"), ignore_errors=True)

current_dir = os.path.dirname(os.path.abspath(__file__))

moe_ext = load(
    name="moe_ext",
    sources=[
        os.path.join(current_dir, "main.cpp"),
        os.path.join(current_dir, "kernel.cu"),
    ],
    extra_cuda_cflags=[
        "-O3",
        "-std=c++20",
        "--expt-relaxed-constexpr",
        "-arch=sm_100a",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ],
    extra_cflags=["-O3", "-std=c++20", "-fPIC"],
    verbose=True
)

def run(
    routing_logits,
    routing_bias,
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    local_expert_offset,
    routed_scaling_factor
):
    return moe_ext.run(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        local_expert_offset,
        routed_scaling_factor
    )
