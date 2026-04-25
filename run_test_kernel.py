import modal
import subprocess
import os

app = modal.App("test-kernel-exec")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .add_local_dir("solution", remote_path="/workspace/solution")
    .add_local_dir(
        "experiments/cuda-ptx-tests",
        remote_path="/workspace/experiments/cuda-ptx-tests",
    )
)

@app.function(image=image, gpu="B200")
def run_test():
    import os
    os.chdir("/workspace")
    print("Compiling test_fused_gemm1.cu...")
    compile_cmd = [
        "nvcc", "-arch=sm_100a", 
        "-O3", "-std=c++17",
        "experiments/cuda-ptx-tests/test_fused_gemm1.cu",
        "-o", "test_fused"
    ]
    
    try:
        result = subprocess.run(compile_cmd, capture_output=True, text=True, check=True)
        print("Compilation successful!")
        print(result.stderr) # print warnings
    except subprocess.CalledProcessError as e:
        print("Compilation failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return
        
    print("\nRunning test_fused...")
    try:
        run_result = subprocess.run(["./test_fused"], capture_output=True, text=True, check=True)
        print("Execution successful!")
        print(run_result.stdout)
    except subprocess.CalledProcessError as e:
        print("Execution failed!")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)

@app.local_entrypoint()
def main():
    run_test.remote()
