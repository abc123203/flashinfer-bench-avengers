"""
在 Modal B200 上编译并运行 test_tcgen05.cu

用法:
    modal run scripts/run_tcgen05_modal.py              # 编译运行
    modal run scripts/run_tcgen05_modal.py --ncu         # 编译 + NCU profiling
"""

import modal

app = modal.App("tcgen05-test")

trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
MOUNT_PATH = "/mnt/flashinfer-trace"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("ninja-build", "kmod")
    .run_commands("pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128")
    .pip_install("flashinfer-bench", "triton==3.4.0", "numpy")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
    })
    .add_local_file(
        "experiments/cuda-ptx-tests/test_tcgen05.cu", "/root/test_tcgen05.cu"
    )
)


@app.function(image=image, gpu="B200:1", timeout=600, volumes={MOUNT_PATH: trace_volume})
def compile_and_run(run_ncu: bool = False):
    import subprocess, os

    output_dir = f"{MOUNT_PATH}/test"
    os.makedirs(output_dir, exist_ok=True)

    src = "/root/test_tcgen05.cu"
    binary = f"{output_dir}/test_tcgen05"

    r = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print(f"GPU: {r.stdout.strip()}")

    # 编译
    print("\nCompiling: nvcc -arch=sm_100a ...")
    c = subprocess.run(["nvcc", "-arch=sm_100a", "-lineinfo", "-o", binary, src],
                       capture_output=True, text=True)
    if c.returncode != 0:
        print(f"❌ 编译失败:\n{c.stderr}")
        return c.stderr
    print("✅ 编译成功!\n")

    if run_ncu:
        # NCU profiling
        ncu_report = f"{output_dir}/tcgen05_profile.ncu-rep"
        cmd = ["ncu", "--set", "detailed", "-o", ncu_report, binary]
        print(f"Running NCU: {' '.join(cmd)}\n")
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(r.stdout)
        if r.stderr:
            print("STDERR:", r.stderr)
        if r.returncode == 0 and os.path.exists(ncu_report):
            size = os.path.getsize(ncu_report) / 1024
            print(f"\n✅ NCU report: /test/tcgen05_profile.ncu-rep ({size:.1f} KB)")
    else:
        # 普通运行
        r = subprocess.run([binary], capture_output=True, text=True, timeout=30)
        print(r.stdout)
        if r.stderr:
            print("STDERR:", r.stderr)

    trace_volume.commit()
    return r.stdout


@app.local_entrypoint()
def main(ncu: bool = False):
    output = compile_and_run.remote(run_ncu=ncu)
    print("\n" + "=" * 60)
    print(output)
