"""
B200 上编译运行 CUDA 测试程序。

用法:
  modal run scripts/test_modal.py
  modal run scripts/test_modal.py --test-file test_fused_gemm1.cu
"""

import modal
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
# PTX / tcgen05 探针与 bench 提交的 solution 分离，避免打进 torch 扩展
PTX_TESTS_DIR = PROJECT_ROOT / "experiments" / "cuda-ptx-tests"
SOLUTION_CUDA_DIR = PROJECT_ROOT / "solution" / "cuda"

app = modal.App("cuda-test")

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ninja-build")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64",
    })
)


@app.function(image=image, gpu="B200:1", timeout=600)
def compile_and_run(source_files: dict[str, bytes], test_file: str, extra_args: str = "") -> str:
    """
    在 B200 上编译并运行 CUDA 测试程序。

    Args:
        source_files: {filename: content_bytes} 需要上传的所有源文件
        test_file: 主测试文件名 (如 test_fused_gemm1.cu)
        extra_args: 额外的 nvcc 参数
    """
    import subprocess
    import os

    workdir = "/tmp/cuda_test"
    os.makedirs(workdir, exist_ok=True)

    # 写入所有源文件
    for fname, content in source_files.items():
        fpath = os.path.join(workdir, fname)
        with open(fpath, "wb") as f:
            f.write(content)
        print(f"  Written: {fname} ({len(content)} bytes)")

    # GPU 信息
    result = subprocess.run("nvidia-smi -L", shell=True, capture_output=True, text=True)
    print(f"\nGPU: {result.stdout.strip()}\n")

    # 编译
    output_name = test_file.replace(".cu", "")
    # 检查测试文件是否引用 kernel.h (需要链接 kernel.cu)
    test_content = source_files[test_file].decode('utf-8', errors='ignore')
    needs_kernel = 'kernel.h' in test_content
    link_files = [test_file]
    if needs_kernel and 'kernel.cu' in source_files:
        link_files.append('kernel.cu')
    link_files_str = " ".join(link_files)

    nvtx_flag = "-lnvToolsExt" if needs_kernel else ""
    compile_cmd = (
        f"nvcc -arch=sm_100a -O2 -lineinfo "
        f"-o {output_name} {link_files_str} "
        f"{nvtx_flag} {extra_args}"
    )

    print(f"=== Compile ===")
    print(f"$ {compile_cmd}\n")

    result = subprocess.run(
        compile_cmd, shell=True, cwd=workdir,
        capture_output=True, text=True, timeout=120
    )

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        return f"COMPILE FAILED (exit code {result.returncode})\n{result.stderr}"

    print("Compile OK\n")

    # 运行（用 Popen 流式输出，超时后 kill）
    run_cmd = f"./{output_name}"
    print(f"=== Run ===")
    print(f"$ {run_cmd}\n")

    import time
    proc = subprocess.Popen(
        run_cmd, shell=True, cwd=workdir,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True
    )

    output_lines = []
    start_time = time.time()
    timeout_sec = 60  # 60 秒应该足够

    try:
        while True:
            line = proc.stdout.readline()
            if line:
                print(line, end='', flush=True)
                output_lines.append(line)
            elif proc.poll() is not None:
                break

            if time.time() - start_time > timeout_sec:
                proc.kill()
                output_lines.append(f"\n*** KILLED after {timeout_sec}s timeout ***\n")
                print(f"\n*** KILLED after {timeout_sec}s timeout ***")
                break
    except Exception as e:
        output_lines.append(f"\nException: {e}\n")
        proc.kill()

    remaining = proc.stdout.read()
    if remaining:
        print(remaining, end='')
        output_lines.append(remaining)

    proc.wait()
    output = "".join(output_lines)
    output += f"\nExit code: {proc.returncode}\n"
    print(f"Exit code: {proc.returncode}")

    return output


@app.local_entrypoint()
def main(test_file: str = "test_fused_gemm1.cu", extra_args: str = ""):
    """打包源文件并在 B200 上编译运行。"""

    # 收集需要上传的源文件
    files_to_upload = [test_file]
    # 检查是否需要 kernel.cu/kernel.h
    test_content = (PTX_TESTS_DIR / test_file).read_text()
    if 'kernel.h' in test_content:
        files_to_upload.extend(["kernel.cu", "kernel.h"])

    source_files = {}
    for fname in files_to_upload:
        if fname in ("kernel.cu", "kernel.h"):
            fpath = SOLUTION_CUDA_DIR / fname
        else:
            fpath = PTX_TESTS_DIR / fname
        if not fpath.exists():
            print(f"ERROR: File not found: {fpath}")
            return
        source_files[fname] = fpath.read_bytes()
        print(f"Packed: {fname} ({len(source_files[fname])} bytes)")

    print(f"\nSubmitting to B200...\n")
    output = compile_and_run.remote(source_files, test_file, extra_args)
    print("\n=== Result ===")
    print(output)
