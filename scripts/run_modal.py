"""
FlashInfer-Bench Modal Cloud Benchmark Runner.

Automatically packs the solution from source files and runs benchmarks
on NVIDIA B200 GPUs via Modal.

``modal run scripts/run_modal.py --test`` — only ``seq_len == 32`` workloads, lighter
BenchmarkConfig, and no ``solution/last_modal_trace_dump.jsonl`` download.

``modal run scripts/run_modal.py --test-large`` — only ``seq_len == 901`` (single workload).

Full runs also write ``solution/last_modal_benchmark_summary.txt``: one line per workload,
sorted by ``seq_len``, with ``[seq_len=…]`` before each UUID prefix; non-PASSED lines use
``0.000 ms`` and ``speedup=0.00x`` while still printing abs/rel when present.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import modal
from flashinfer_bench import Benchmark, BenchmarkConfig, Solution, TraceSet

app = modal.App("flashinfer-bench")


def _workload_seq_len(workload) -> int | None:
    """Resolve seq_len from a Workload-like object (dict or pydantic)."""
    axes = getattr(workload, "axes", None)
    if axes is None:
        return None
    if isinstance(axes, dict):
        v = axes.get("seq_len")
    elif hasattr(axes, "get"):
        v = axes.get("seq_len")
    else:
        v = getattr(axes, "seq_len", None)
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _format_benchmark_summary_line(seq_len: int | None, workload_uuid: str, res: dict) -> str:
    """One-line summary; non-PASSED uses 0 ms and 0.00x speedup like PASSED layout."""
    uuid_short = workload_uuid[:8]
    sl_tag = str(seq_len) if seq_len is not None else "?"
    status = res.get("status") or "UNKNOWN"
    max_abs = res.get("max_abs_error")
    max_rel = res.get("max_rel_error")
    if max_abs is not None:
        rel = max_rel if max_rel is not None else float("nan")
        err_str = f" | abs={max_abs:.2e}, rel={rel:.2e}"
    else:
        err_str = ""

    if status == "PASSED":
        lat = res.get("latency_ms")
        if lat is None:
            lat = 0.0
        spd = res.get("speedup_factor")
        spd_str = f" | speedup={spd:.2f}x" if spd is not None else ""
        return f"  [seq_len={sl_tag}] {uuid_short}: {status} | {lat:.3f} ms{err_str}{spd_str}"

    return (
        f"  [seq_len={sl_tag}] {uuid_short}: {status} | 0.000 ms{err_str} | speedup=0.00x"
    )


def _build_sorted_benchmark_summary(def_name: str, traces: dict) -> str:
    """Sort by seq_len ascending; missing seq_len last."""
    rows: list[tuple[int, str, dict]] = []
    for workload_uuid, res in traces.items():
        sl = res.get("seq_len")
        if sl is None:
            sort_key = 2**31 - 1
        else:
            try:
                sort_key = int(sl)
            except (TypeError, ValueError):
                sort_key = 2**31 - 1
        rows.append((sort_key, workload_uuid, res))
    rows.sort(key=lambda t: (t[0], t[1]))

    lines = [f"{def_name}:"]
    for _sk, workload_uuid, res in rows:
        sl = res.get("seq_len")
        lines.append(_format_benchmark_summary_line(sl, workload_uuid, res))
    return "\n".join(lines) + "\n"


trace_volume = modal.Volume.from_name("flashinfer-trace", create_if_missing=True)
MOUNT_PATH = "/data"

image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .apt_install("ninja-build", "kmod")
    .run_commands(
        "pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128",
        "mv /usr/local/cuda/bin/nvcc /usr/local/cuda/bin/nvcc.real",
        "printf '%s\\n' '#!/bin/bash' 'args=(\"$@\")' 'for i in \"${!args[@]}\"; do' '  args[$i]=\"${args[$i]//sm_100/sm_100a}\"' '  args[$i]=\"${args[$i]//compute_100/compute_100a}\"' '  args[$i]=\"${args[$i]//sm_100aa/sm_100a}\"' '  args[$i]=\"${args[$i]//compute_100aa/compute_100a}\"' 'done' 'exec /usr/local/cuda/bin/nvcc.real \"${args[@]}\"' > /usr/local/cuda/bin/nvcc",
        "chmod +x /usr/local/cuda/bin/nvcc"
    )
    .pip_install("flashinfer-bench", "triton==3.4.0", "numpy")
    .env({
        "CUDA_HOME": "/usr/local/cuda",
        "PATH": "/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "LD_LIBRARY_PATH": "/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64",
        # tcgen05 / block_scale MMA 仅在 sm_100a 可用；仅 sm_100 会导致 ptxas 报 Instruction 'tcgen05.*' not supported
        "TORCH_CUDA_ARCH_LIST": "10.0a",
        "TORCH_NVCC_FLAGS": "-arch=sm_100a",
    })
)


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={MOUNT_PATH: trace_volume})
def run_benchmark(solution: Solution, config: BenchmarkConfig = None, test: bool = False, test_large: bool = False, test_small: bool = False) -> dict:
    """Run benchmark on Modal B200 and return results.

    If ``test`` is True, only workloads with ``axes.seq_len == 32`` are run, and no
    full-trace JSONL is returned (avoids large payload / local dump).
    If ``test_large`` is True, only workloads with ``axes.seq_len == 901`` are run.
    If ``test_small`` is True, only workloads with ``axes.seq_len == 17`` are run.
    """
    import os
    import shutil
    from pathlib import Path

    shutil.rmtree("/root/.cache/flashinfer_bench/cache/torch/", ignore_errors=True)

    # Automatically find the path that contains 'definitions' folder
    trace_set_root = MOUNT_PATH
    found = False
    for root, dirs, _ in os.walk(MOUNT_PATH):
        if "definitions" in dirs:
            trace_set_root = root
            found = True
            break
    
    if not found:
        raise FileNotFoundError(f"Trace set not found in volume mounted at {MOUNT_PATH}")

    os.environ["FIB_DATASET_PATH"] = trace_set_root
    os.environ["FIB_DB_PATH"] = trace_set_root

    if config is None:
        if test:
            config = BenchmarkConfig(
                warmup_runs=1, iterations=10, num_trials=2,
                atol=1.0, rtol=0.3, required_matched_ratio=0.9,
            )
        else:
            config = BenchmarkConfig(
                warmup_runs=3, iterations=100, num_trials=5,
                atol=1.0, rtol=0.3, required_matched_ratio=0.9,
            )

    trace_set = TraceSet.from_path(trace_set_root)
    definition = trace_set.definitions.get(solution.definition)
    if not definition:
        raise ValueError(f"Definition '{solution.definition}' not found in trace set")

    all_workloads = trace_set.workloads.get(solution.definition, [])
    if test_small:
        workloads = []
        for wl_trace in all_workloads:
            wl = getattr(wl_trace, "workload", wl_trace)
            axes = getattr(wl, "axes", None) or {}
            sl = axes.get("seq_len")
            if sl is not None and int(sl) == 80:
                workloads.append(wl_trace)
        if not workloads:
            raise ValueError(
                f"No workload with seq_len==80 found for definition '{solution.definition}'"
            )
    elif test_large:
        workloads = []
        for wl_trace in all_workloads:
            wl = getattr(wl_trace, "workload", wl_trace)
            axes = getattr(wl, "axes", None) or {}
            sl = axes.get("seq_len")
            if sl is not None and int(sl) == 901:
                workloads.append(wl_trace)
        if not workloads:
            raise ValueError(
                f"No workload with seq_len==901 found for definition '{solution.definition}'"
            )
    elif test:
        workloads = []
        for wl_trace in all_workloads:
            wl = getattr(wl_trace, "workload", wl_trace)
            axes = getattr(wl, "axes", None) or {}
            if axes.get("seq_len") == 32:
                workloads.append(wl_trace)
        if not workloads:
            raise ValueError(
                f"No workload with seq_len=32 found for definition '{solution.definition}'"
            )
    else:
        workloads = all_workloads
    bench_trace_set = TraceSet(
        root=trace_set.root,
        definitions={definition.name: definition},
        solutions={definition.name: [solution]},
        workloads={definition.name: workloads},
        traces={definition.name: []},
    )

    benchmark = Benchmark(bench_trace_set, config)
    result_trace_set = benchmark.run_all(dump_traces=True)

    traces = result_trace_set.traces.get(definition.name, [])
    results = {definition.name: {}}

    for trace in traces:
        if trace.evaluation:
            entry = {
                "status": trace.evaluation.status.value,
                "solution": trace.solution,
            }
            log_text = getattr(trace.evaluation, "log", "") or ""
            try:
                eval_dict = trace.evaluation.model_dump()
            except Exception:
                try:
                    eval_dict = trace.evaluation.dict()
                except Exception:
                    eval_dict = {}
            for k in ("compile_log", "build_log", "error", "stderr", "message"):
                if k in eval_dict and eval_dict[k]:
                    log_text += f"\n[{k}]: {eval_dict[k]}"
            if log_text.strip():
                entry["log"] = log_text
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error
            sl = _workload_seq_len(trace.workload)
            if sl is not None:
                entry["seq_len"] = sl
            results[definition.name][trace.workload.uuid] = entry

    if not test:
        # Full FlashInfer traces (JSON Lines) for Trace Viewer — returned to local main() for saving.
        import json as _json

        def _trace_to_jsonable(tr):
            if hasattr(tr, "model_dump"):
                try:
                    return tr.model_dump(mode="json")
                except Exception:
                    return tr.model_dump()
            if hasattr(tr, "dict"):
                return tr.dict()
            return {"_error": "unserializable_trace", "_repr": repr(tr)}

        _jsonl_parts = []
        for tr in traces:
            try:
                _jsonl_parts.append(_json.dumps(_trace_to_jsonable(tr), ensure_ascii=False, default=str))
            except Exception as ex:
                _jsonl_parts.append(
                    _json.dumps({"_error": "trace_serialize_failed", "detail": str(ex)}, default=str)
                )
        results["_trace_dump_jsonl"] = "\n".join(_jsonl_parts)

    return results


@app.function(image=image, gpu="B200:1", timeout=3600, volumes={MOUNT_PATH: trace_volume})
def profile_ncu(solution: Solution, ncu_set: str = "detailed", ncu_page: str = "details", timeout: int = 60) -> str:
    """Run NCU profiling on Modal B200 and return the output."""
    import os
    import subprocess
    from flashinfer_bench import TraceSet
    from flashinfer_bench.agents import flashinfer_bench_run_ncu

    # Automatically find the path that contains 'definitions' folder
    trace_set_root = MOUNT_PATH
    found = False
    for root, dirs, _ in os.walk(MOUNT_PATH):
        if "definitions" in dirs:
            trace_set_root = root
            found = True
            break
    
    if not found:
        raise FileNotFoundError(f"Trace set not found in volume mounted at {MOUNT_PATH}")

    # NCU Library Fix: Use found internal libraries
    print("DEBUG: Refining NCU environment setup...")
    try:
        # Aggressive diagnostic exploration
        print("DEBUG: Listing /opt/nvidia contents...")
        try: subprocess.run("ls -R /opt/nvidia 2>/dev/null | head -n 20", shell=True)
        except: pass

        ncu_bin = "/usr/local/cuda/bin/ncu"
        try:
            found_which = subprocess.check_output("which ncu", shell=True, text=True).strip()
            if found_which: ncu_bin = found_which
            print(f"DEBUG: NCU binary at {ncu_bin}, running ldd...")
            subprocess.run(f"ldd {ncu_bin} | head -n 20", shell=True)
        except: pass

        ncu_internal_dirs = []
        # Full system search - slow but definitive
        print("DEBUG: Searching for libperfworks.so everywhere...")
        try:
            results = subprocess.check_output("find / -name 'libperfworks.so' 2>/dev/null", shell=True, text=True).strip().split('\n')
            for res in results:
                if res:
                    print(f"DEBUG: Found libperfworks.so at: {res}")
                    d = os.path.dirname(res)
                    if d not in ncu_internal_dirs: ncu_internal_dirs.append(d)
        except: pass

        # Basic driver and toolkit paths
        extra_paths = [
            "/usr/lib/x86_64-linux-gnu",
            "/usr/local/cuda/extras/CUPTI/lib64",
            "/usr/local/cuda/lib64",
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/opt/nvidia/nsight-compute/2025.1.0/target/linux-desktop-glibc_2_11_3-x64",
            "/opt/nvidia/nsight-compute/2024.1.1/target/linux-desktop-glibc_2_11_3-x64"
        ] + ncu_internal_dirs
        
        current_ld = os.environ.get("LD_LIBRARY_PATH", "")
        # Combined and de-duplicated library path
        new_ld = ":".join(list(dict.fromkeys(extra_paths)))
        os.environ["LD_LIBRARY_PATH"] = f"{new_ld}:{current_ld}" if current_ld else new_ld
        print(f"DEBUG: LD_LIBRARY_PATH is now: {os.environ['LD_LIBRARY_PATH']}")
        
        # Diagnostics
        try:
            ncu_ver = subprocess.check_output("ncu --version", shell=True, text=True).strip()
            print(f"DEBUG: NCU Version: \n{ncu_ver}")
        except: pass
        
        try:
            subprocess.run("nvidia-smi -L", shell=True, check=True)
        except: pass

    except Exception as e:
        print(f"DEBUG: NCU setup error: {e}")

    os.environ["FIB_DATASET_PATH"] = trace_set_root
    os.environ["FIB_DB_PATH"] = trace_set_root

    trace_set = TraceSet.from_path(trace_set_root)
    workloads = trace_set.workloads.get(solution.definition, [])
    if not workloads:
        raise ValueError(f"No workloads found for definition '{solution.definition}'")
    
    workload = workloads[0].workload

    # Convert paths to absolute for NCU
    def _abs_path(wl, root):
        import copy
        root = os.path.abspath(root)
        try: data = wl.model_dump()
        except: data = wl.dict()
        if "inputs" in data:
            for k, v in data["inputs"].items():
                if isinstance(v, dict) and "path" in v:
                    v["path"] = os.path.normpath(os.path.join(root, v["path"]))
        try: return type(wl).model_validate(data)
        except: return type(wl)(**data)

    workload = _abs_path(workload, trace_set_root)
    
    print(f"Running NCU profile...")
    return flashinfer_bench_run_ncu(solution, workload, set=ncu_set, page=ncu_page, timeout=timeout)


@app.local_entrypoint()
def main(
    profile: bool = False,
    test: bool = False,
    test_large: bool = False,
    test_small: bool = False,
    set: str = "detailed",
    page: str = "details",
    timeout: int = 60,
):
    from scripts.pack_solution import pack_solution
    solution_path = pack_solution()
    solution = Solution.model_validate_json(solution_path.read_text())

    if profile:
        output = profile_ncu.remote(solution, ncu_set=set, ncu_page=page, timeout=timeout)
        print("\n--- NCU Profiling Output ---\n", output, "\n--- End Output ---")
        return

    if test_small:
        test = True
        print("Test-small mode: only seq_len=80; trace JSONL will not be saved locally.")
    elif test_large:
        test = True
        print("Test-large mode: only seq_len=901; trace JSONL will not be saved locally.")
    elif test:
        print("Test mode: only workloads with seq_len=32; trace JSONL will not be saved locally.")

    results = run_benchmark.remote(solution, test=test, test_large=test_large, test_small=test_small)
    if results:
        trace_dump = results.pop("_trace_dump_jsonl", None)
        summary_path = PROJECT_ROOT / "solution" / "last_modal_benchmark_summary.txt"
        if not test and isinstance(trace_dump, str) and trace_dump.strip():
            out_path = PROJECT_ROOT / "solution" / "last_modal_trace_dump.jsonl"
            out_path.write_text(trace_dump if trace_dump.endswith("\n") else trace_dump + "\n", encoding="utf-8")
            print(f"\nFull trace JSONL saved to: {out_path}")
            print("  (open in editor or paste lines into https://bench.flashinfer.ai/viewer )")

        def _sort_key_seq_len(res: dict) -> int:
            v = res.get("seq_len")
            if v is None:
                return 2**31 - 1
            try:
                return int(v)
            except (TypeError, ValueError):
                return 2**31 - 1

        summary_blocks: list[str] = []
        for def_name, traces in results.items():
            block = _build_sorted_benchmark_summary(def_name, traces)
            summary_blocks.append(block.rstrip("\n"))
            print(f"\n{block}", end="")

        if summary_blocks:
            summary_path.write_text("\n\n".join(summary_blocks) + "\n", encoding="utf-8")
            print(f"\nBenchmark summary (seq_len sorted) saved to: {summary_path}")

        for def_name, traces in results.items():
            for _sk, uuid, res in sorted(
                ((_sort_key_seq_len(res), uuid, res) for uuid, res in traces.items()),
                key=lambda t: (t[0], t[1]),
            ):
                status = res.get("status")
                if status and status != "PASSED":
                    sl = res.get("seq_len")
                    sl_tag = str(sl) if sl is not None else "?"
                    print(f"    --- [{def_name}] [seq_len={sl_tag}] {uuid[:8]} evaluation log ---")
                    log_text = res.get("log", "")
                    if isinstance(log_text, str) and log_text.strip():
                        for line in log_text.rstrip("\n").splitlines():
                            print(f"    {line}")
                        print("    --- end log ---")
                    else:
                        import json as _json

                        print("    --- full result (no log) ---")
                        print(f"    {_json.dumps(res, indent=2, default=str)}")
                        print("    --- end result ---")
