import modal
import subprocess
import os

app = modal.App("test-build")
image = modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12").run_commands("pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128").add_local_dir("solution", remote_path="/workspace/solution")

@app.function(image=image)
def run_build():
    os.chdir("/workspace")
    print("Compiling main.cpp syntax check on Modal...")
    compile_cmd = [
        "c++", "-fPIC", "-std=c++17",
        "-D_GLIBCXX_USE_CXX11_ABI=1",
        "-I/usr/local/lib/python3.12/site-packages/torch/include",
        "-I/usr/local/lib/python3.12/site-packages/torch/include/torch/csrc/api/include",
        "-I/usr/local/cuda/include",
        "-I/usr/local/include/python3.12",
        "-c", "solution/cuda/main.cpp",
        "-o", "main.o"
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Compilation failed!")
        print(result.stderr)
    else:
        print("Syntax check passed!")
        
@app.local_entrypoint()
def main():
    run_build.remote()
