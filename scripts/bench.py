"""
Bench CLI — run and benchmark solutions.

Usage:
    python scripts/bench.py list
    python scripts/bench.py run reference
    python scripts/bench.py run --all
    python scripts/bench.py run reference --modal
    python scripts/bench.py run --all --modal
    python scripts/bench.py run reference --warmup 1 --iterations 10 --trials 1
    python scripts/bench.py run --all --modal --force
    python scripts/bench.py sanitize ref_prefill
    python scripts/bench.py sanitize ref_prefill --all-workloads
    python scripts/bench.py sanitize ref_prefill --workload a3f2
    python scripts/bench.py profile ref_prefill
    python scripts/bench.py profile ref_prefill --page source --set full
"""

import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from tabulate import tabulate

PROJECT_ROOT = Path(__file__).parent.parent
SOLUTIONS_DIR = PROJECT_ROOT / "solutions"

# ---------------------------------------------------------------------------
# Discovery & config
# ---------------------------------------------------------------------------


def discover_solutions() -> list[str]:
    """Return sorted list of solution names (subdirectory names under solutions/)."""
    if not SOLUTIONS_DIR.exists():
        return []
    return sorted(
        d.name for d in SOLUTIONS_DIR.iterdir()
        if d.is_dir() and any(d.glob("*.py"))
    )


def load_root_config() -> dict:
    config_path = PROJECT_ROOT / "config.toml"
    with open(config_path, "rb") as f:
        return tomllib.load(f)


def load_solution_config(sol_name: str) -> dict:
    """Merge root config with optional per-solution config.toml overrides."""
    root = load_root_config()
    sol_config_path = SOLUTIONS_DIR / sol_name / "config.toml"
    if sol_config_path.exists():
        with open(sol_config_path, "rb") as f:
            overrides = tomllib.load(f)
        for section in ("solution", "build"):
            if section in overrides:
                root.setdefault(section, {}).update(overrides[section])
    return root


def get_git_info() -> dict:
    """Collect git version metadata for the current repo."""
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
        info["dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT, text=True
        ).strip())
        info["commit_time"] = subprocess.check_output(
            ["git", "log", "-1", "--format=%cI"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return info


# ---------------------------------------------------------------------------
# Solution hashing
# ---------------------------------------------------------------------------


def hash_solution(sol_name: str) -> str:
    """Content-hash all files in a solution directory (deterministic order)."""
    sol_dir = SOLUTIONS_DIR / sol_name
    h = hashlib.sha256()
    for p in sorted(sol_dir.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(sol_dir).as_posix().encode())
            h.update(p.read_bytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def pack_solution(sol_name: str):
    """Pack a solution from solutions/<name>/ into a Solution object."""
    from flashinfer_bench import BuildSpec
    from flashinfer_bench.agents import pack_solution_from_files

    config = load_solution_config(sol_name)
    sol_cfg = config["solution"]
    build_cfg = config["build"]

    source_dir = SOLUTIONS_DIR / sol_name
    if not source_dir.exists():
        raise FileNotFoundError(f"Solution directory not found: {source_dir}")

    spec = BuildSpec(
        language=build_cfg["language"],
        target_hardware=["cuda"],
        entry_point=build_cfg["entry_point"],
        destination_passing_style=build_cfg.get("destination_passing_style", False),
    )

    solution = pack_solution_from_files(
        path=str(source_dir),
        spec=spec,
        name=f"{sol_cfg['name']}-{sol_name}",
        definition=sol_cfg["definition"],
        author=sol_cfg["author"],
    )
    return solution


# ---------------------------------------------------------------------------
# Workload resolution (shared by sanitize / profile)
# ---------------------------------------------------------------------------


def _resolve_workloads(sol_name: str, workload_prefix: str | None, all_workloads: bool):
    """Pack solution, load TraceSet, return (solution, [workload_objects])."""
    from flashinfer_bench import TraceSet

    solution = pack_solution(sol_name)
    trace_set_path = get_trace_set_path()
    ts = TraceSet.from_path(trace_set_path)

    def_name = solution.definition
    traces = ts.workloads.get(def_name, [])
    if not traces:
        print(f"No workloads found for definition '{def_name}'")
        sys.exit(1)

    workloads = [t.workload for t in traces]

    if all_workloads:
        return solution, workloads

    if workload_prefix:
        matches = [w for w in workloads if w.uuid.startswith(workload_prefix)]
        if not matches:
            uuids = ", ".join(w.uuid[:8] for w in workloads)
            print(f"No workload matching prefix '{workload_prefix}'. Available: {uuids}")
            sys.exit(1)
        return solution, matches

    # Default: first workload
    return solution, [workloads[0]]


# ---------------------------------------------------------------------------
# Benchmark execution
# ---------------------------------------------------------------------------


def get_trace_set_path() -> str:
    path = os.environ.get("FIB_DATASET_PATH")
    if not path:
        raise EnvironmentError(
            "FIB_DATASET_PATH not set. Point it at your flashinfer-trace dataset."
        )
    return path


def run_solutions(names: list[str], warmup: int, iterations: int, trials: int, modal: bool = False, force: bool = False):
    """Pack, benchmark, print, and log to W&B."""
    from scripts.wandb_logger import find_cached_result, init_solution_run, log_solution_results

    fidelity = "modal" if modal else "local"
    bench_config = {"warmup_runs": warmup, "iterations": iterations, "num_trials": trials}
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    group = f"bench_{timestamp}"
    git_info = get_git_info()

    # --- Cache check ---
    cached_results = {}  # sol_name -> {def_name: {uuid: entry}}
    cached_set = set()   # sol_names that came from cache
    fresh_names = []     # sol_names that need benchmarking
    sol_hashes = {}      # sol_name -> hash string

    print("Checking cache...")
    for name in names:
        sol_hash = hash_solution(name)
        sol_hashes[name] = sol_hash

        if force:
            fresh_names.append(name)
            print(f"  {name}: forced re-run")
            continue

        hit = find_cached_result(name, sol_hash, fidelity, bench_config)
        if hit is not None:
            print(f"  {name}: cached ({sol_hash})")
            cached_results[name] = hit
            cached_set.add(name)
        else:
            print(f"  {name}: cache miss")
            fresh_names.append(name)

    # --- Benchmark fresh solutions ---
    fresh_results = {}
    fresh_solutions = {}

    if fresh_names and not modal:
        # Local: one solution at a time, W&B wraps each benchmark
        from flashinfer_bench import BenchmarkConfig, TraceSet

        config = BenchmarkConfig(warmup_runs=warmup, iterations=iterations, num_trials=trials)
        base_ts = TraceSet.from_path(get_trace_set_path())

        for name in fresh_names:
            print(f"\nPacking {name}...")
            sol = pack_solution(name)
            fresh_solutions[name] = sol

            run = init_solution_run(
                name, sol, git_info=git_info,
                sol_hash=sol_hashes[name],
                bench_config=bench_config,
                fidelity=fidelity, group=group, timestamp=timestamp,
            )

            bench_error = None
            try:
                result = _run_local_single(sol, config, base_ts)
            except Exception as exc:
                print(f"\n  {name}: benchmark failed ({exc})")
                result = {}
                bench_error = str(exc)

            fresh_results[name] = result

            log_solution_results(
                run, name, result, sol,
                sol_dir=SOLUTIONS_DIR / name,
                error=bench_error,
            )

    elif fresh_names and modal:
        # Modal: batch all solutions in one remote call
        from flashinfer_bench import BenchmarkConfig

        for name in fresh_names:
            print(f"Packing {name}...")
            fresh_solutions[name] = pack_solution(name)

        config = BenchmarkConfig(warmup_runs=warmup, iterations=iterations, num_trials=trials)
        fresh_results = _run_modal(fresh_solutions, config)

        # Log each solution to W&B individually
        print("\nLogging to W&B...")
        for name, result in fresh_results.items():
            run = init_solution_run(
                name, fresh_solutions[name], git_info=git_info,
                sol_hash=sol_hashes[name],
                bench_config=bench_config,
                fidelity=fidelity, group=group, timestamp=timestamp,
            )
            log_solution_results(
                run, name, result, fresh_solutions[name],
                sol_dir=SOLUTIONS_DIR / name,
            )

    # --- Merge results for display ---
    all_results = {}
    for name in names:
        if name in cached_results:
            all_results[name] = cached_results[name]
        elif name in fresh_results:
            all_results[name] = fresh_results[name]

    print_results_table(all_results, cached_set=cached_set)


def _run_local_single(solution, config, base_ts) -> dict:
    """Run benchmark locally for a single solution."""
    from flashinfer_bench import Benchmark, TraceSet

    def_name = solution.definition
    bench_ts = TraceSet(
        root=base_ts.root,
        definitions={def_name: base_ts.definitions[def_name]},
        solutions={def_name: [solution]},
        workloads={def_name: base_ts.workloads.get(def_name, [])},
        traces={def_name: []},
    )

    print(f"Benchmarking (warmup={config.warmup_runs}, iter={config.iterations}, trials={config.num_trials})...")
    result_ts = Benchmark(bench_ts, config).run_all(dump_traces=True)

    result = {}
    for trace in result_ts.traces.get(def_name, []):
        if trace.evaluation is None:
            entry = {"status": "ERROR"}
        else:
            entry = {"status": trace.evaluation.status.value}
            if trace.evaluation.performance:
                entry["latency_ms"] = trace.evaluation.performance.latency_ms
                entry["reference_latency_ms"] = trace.evaluation.performance.reference_latency_ms
                entry["speedup_factor"] = trace.evaluation.performance.speedup_factor
            if trace.evaluation.correctness:
                entry["max_abs_error"] = trace.evaluation.correctness.max_absolute_error
                entry["max_rel_error"] = trace.evaluation.correctness.max_relative_error

        result.setdefault(def_name, {})[trace.workload.uuid] = entry

    return result


def _run_modal(solutions: dict, config) -> dict:
    """Run benchmarks on Modal B200."""
    from scripts.run_modal import app as modal_app, run_benchmark

    sol_list = list(solutions.values())
    name_lookup = {sol.name: sol_name for sol_name, sol in solutions.items()}

    print(f"\nRunning on Modal B200 ({len(sol_list)} solution(s))...")
    with modal_app.run():
        raw = run_benchmark.remote(sol_list, config)

    # Remap packed names back to solution directory names
    all_results = {}
    for packed_name, defs in raw.items():
        sol_name = name_lookup.get(packed_name, packed_name)
        all_results[sol_name] = defs
    return all_results


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def _summarize(workloads: dict) -> dict:
    """Compute summary stats from a {uuid: entry} dict."""
    speedups, latencies, errors = [], [], []
    passed = total = 0
    for entry in workloads.values():
        total += 1
        if entry.get("status", "").upper() == "PASSED":
            passed += 1
        if entry.get("speedup_factor") is not None:
            speedups.append(entry["speedup_factor"])
        if entry.get("latency_ms") is not None:
            latencies.append(entry["latency_ms"])
        if entry.get("max_abs_error") is not None:
            errors.append(entry["max_abs_error"])
    return {
        "pass_rate": f"{passed}/{total}",
        "mean_speedup": f"{sum(speedups)/len(speedups):.2f}x" if speedups else "n/a",
        "mean_latency": f"{sum(latencies)/len(latencies):.3f}ms" if latencies else "n/a",
        "max_error": f"{max(errors):.2e}" if errors else "n/a",
    }


def print_results_table(results: dict, cached_set: set | None = None):
    """Print a summary table across all solutions."""
    cached_set = cached_set or set()
    rows = []
    for sol_name, sol_results in results.items():
        for def_name, workloads in sol_results.items():
            summary = _summarize(workloads)
            note = "(cached)" if sol_name in cached_set else ""
            rows.append([
                sol_name, def_name,
                summary["pass_rate"], summary["mean_speedup"],
                summary["mean_latency"], summary["max_error"],
                note,
            ])
    headers = ["Solution", "Definition", "Pass Rate", "Mean Speedup", "Mean Latency", "Max Error", ""]
    print(f"\n{tabulate(rows, headers=headers, tablefmt='simple')}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Bench CLI — run and benchmark solutions")
    sub = parser.add_subparsers(dest="command")

    # list
    sub.add_parser("list", help="List available solutions")

    # run
    run_p = sub.add_parser("run", help="Run benchmark for one or more solutions")
    run_p.add_argument("names", nargs="*", help="Solution names to run")
    run_p.add_argument("--all", action="store_true", help="Run all solutions")
    run_p.add_argument("--modal", action="store_true", help="Run on Modal B200 (high fidelity)")
    run_p.add_argument("--warmup", type=int, default=3)
    run_p.add_argument("--iterations", type=int, default=100)
    run_p.add_argument("--trials", type=int, default=5)
    run_p.add_argument("--force", action="store_true", help="Skip cache, force re-run")

    # sanitize
    san_p = sub.add_parser("sanitize", help="Run compute-sanitizer memory checks")
    san_p.add_argument("sol_name", help="Solution name")
    san_p.add_argument("--workload", default=None, help="Workload UUID prefix")
    san_p.add_argument("--all-workloads", action="store_true", help="Run on all workloads")
    san_p.add_argument("--types", nargs="+", default=None,
                       help="Sanitizer types (default: all four)")
    san_p.add_argument("--timeout", type=int, default=300, help="Timeout in seconds")

    # profile
    prof_p = sub.add_parser("profile", help="Run Nsight Compute profiling")
    prof_p.add_argument("sol_name", help="Solution name")
    prof_p.add_argument("--workload", default=None, help="Workload UUID prefix")
    prof_p.add_argument("--all-workloads", action="store_true", help="Run on all workloads")
    prof_p.add_argument("--set", default="detailed", help="NCU metric set (default: detailed)")
    prof_p.add_argument("--page", default="details", help="NCU page (default: details)")
    prof_p.add_argument("--timeout", type=int, default=60, help="Timeout in seconds")

    args = parser.parse_args()

    if args.command == "list":
        solutions = discover_solutions()
        if not solutions:
            print("No solutions found in solutions/")
        else:
            print("Available solutions:")
            for s in solutions:
                print(f"  {s}")

    elif args.command == "run":
        if args.all:
            names = discover_solutions()
        elif args.names:
            names = args.names
        else:
            parser.error("Provide solution names or --all")
            return

        # Validate
        available = set(discover_solutions())
        for n in names:
            if n not in available:
                print(f"Unknown solution: '{n}'. Available: {', '.join(sorted(available))}")
                sys.exit(1)

        run_solutions(names, args.warmup, args.iterations, args.trials, modal=args.modal, force=args.force)

    elif args.command == "sanitize":
        from flashinfer_bench.agents import flashinfer_bench_run_sanitizer

        solution, workloads = _resolve_workloads(
            args.sol_name, args.workload, args.all_workloads
        )
        kwargs = {"timeout": args.timeout}
        if args.types:
            kwargs["sanitizer_types"] = args.types
        for i, wl in enumerate(workloads):
            if len(workloads) > 1:
                print(f"\n--- Workload {i+1}/{len(workloads)}: {wl.uuid[:8]} ---")
            output = flashinfer_bench_run_sanitizer(solution, wl, **kwargs)
            print(output)

    elif args.command == "profile":
        from flashinfer_bench.agents import flashinfer_bench_run_ncu

        solution, workloads = _resolve_workloads(
            args.sol_name, args.workload, args.all_workloads
        )
        for i, wl in enumerate(workloads):
            if len(workloads) > 1:
                print(f"\n--- Workload {i+1}/{len(workloads)}: {wl.uuid[:8]} ---")
            output = flashinfer_bench_run_ncu(
                solution, wl, set=args.set, page=args.page, timeout=args.timeout
            )
            print(output)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()