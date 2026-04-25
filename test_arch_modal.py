import modal
import sys

app = modal.App("test-arch")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .run_commands("pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128")
)

@app.function(image=image)
def test_arch():
    import torch
    from torch.utils.cpp_extension import _get_cuda_arch_flags
    import os
    print("PyTorch version:", torch.__version__)
    try:
        print("Flags for 10.0a:", _get_cuda_arch_flags(["10.0a"]))
    except Exception as e:
        print("Error for 10.0a:", e)
    try:
        print("Flags for 10.0:", _get_cuda_arch_flags(["10.0"]))
    except Exception as e:
        print("Error for 10.0:", e)
    try:
        print("Flags for 100a:", _get_cuda_arch_flags(["100a"]))
    except Exception as e:
        print("Error for 100a:", e)

@app.local_entrypoint()
def main():
    test_arch.remote()
