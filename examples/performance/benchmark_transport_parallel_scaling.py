from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import time
from pathlib import Path

import jax
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.problems.transport_matrix.diagnostics import transport_matrix_size_from_rhs_mode
from sfincs_jax.problems.transport_matrix.parallel.runtime import (
    audit_parallel_scaling_claim_scope,
    audit_transport_parallel_scaling_summary,
)
from sfincs_jax.problems.transport_matrix.parallel.runtime import partition_transport_rhs
from sfincs_jax.problems.transport_matrix.solve import solve_v3_transport_matrix_linear_gmres


def _configure_backend_env(*, workers: int, backend: str) -> None:
    backend_l = str(backend).strip().lower()
    if backend_l not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported backend {backend!r}")
    os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND"] = backend_l
    if backend_l == "gpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, int(workers))))
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
        os.environ.pop("SFINCS_JAX_CPU_DEVICES", None)
    else:
        os.environ["SFINCS_JAX_CPU_DEVICES"] = "1"
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)


def _rhs_mode_from_namelist(input_path: Path) -> int:
    nml = read_sfincs_input(input_path)
    general = nml.group("general")
    return int(general.get("RHSMODE", 2))


def _timing_semantics(*, global_warmup: int, per_worker_warmup: int) -> str:
    if int(per_worker_warmup) > 0:
        return "hot_solve"
    if int(global_warmup) > 0:
        return "cache_warm"
    return "cold_start"


def _compile_amortization_gate(
    *,
    timing_semantics: str,
    global_warmup: int,
    per_worker_warmup: int,
    repeats: int,
    cache_dir: Path | None,
    backend: str,
    repo_root: Path,
) -> dict[str, object]:
    backend_l = str(backend).strip().lower()
    cache_configured = cache_dir is not None
    per_worker_hot = int(per_worker_warmup) > 0
    global_cache_warm = int(global_warmup) > 0 and cache_configured
    timed_repeats = max(int(repeats), 1)
    persistent_pool_env = os.environ.get("SFINCS_JAX_TRANSPORT_POOL_PERSIST", "").strip().lower()
    persistent_pool_enabled = backend_l == "cpu" and persistent_pool_env not in {"0", "false", "no", "off"}
    passes = str(timing_semantics) in {"cache_warm", "hot_solve", "warm"} and (
        per_worker_hot or global_cache_warm
    )
    reason = (
        "per-worker warmup excludes worker-side compile/setup from timed repeats"
        if per_worker_hot
        else "global warmup plus persistent compilation cache excludes first compile from timed repeats"
        if global_cache_warm
        else "no warm compile/cache phase is planned before timed repeats"
    )
    gate: dict[str, object] = {
        "passes": bool(passes),
        "timing_semantics": str(timing_semantics),
        "strategy": "per_worker_warmup" if per_worker_hot else "global_persistent_compile_cache",
        "global_warmup": max(int(global_warmup), 0),
        "per_worker_warmup": max(int(per_worker_warmup), 0),
        "timed_repeats": timed_repeats,
        "min_timed_repeats": 1,
        "persistent_worker_pool_enabled": bool(persistent_pool_enabled),
        "persistent_compile_cache": bool(cache_configured),
        "compile_in_timed_region": False,
        "reason": reason,
    }
    if cache_dir is not None:
        gate["compile_cache_dir"] = _display_path(cache_dir, repo_root=repo_root)
    return gate


def _payloads_for_workers(*, rhs_count: int, workers: int) -> list[dict[str, object]]:
    rhs_values = list(range(1, int(rhs_count) + 1))
    active_workers = min(int(workers), int(rhs_count))
    return [{"which_rhs_values": chunk} for chunk in partition_transport_rhs(rhs_values, active_workers)]


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _normalize_worker_counts(*, rhs_count: int, requested_workers: list[int]) -> tuple[list[int], list[int], list[int]]:
    requested = sorted({int(w) for w in requested_workers if int(w) >= 1})
    workers = [w for w in requested if w <= int(rhs_count)]
    skipped = [w for w in requested if w > int(rhs_count)]
    if not workers:
        workers = [1] if int(rhs_count) <= 1 else [1, int(rhs_count)]
    return requested, workers, skipped


def _transport_benchmark_command(
    *,
    input_path: Path,
    workers: list[int],
    repeats: int,
    warmup: int,
    global_warmup: int,
    precond: str,
    backend: str,
    out_dir: Path,
    cache_dir: Path,
    figure_name: str,
    repo_root: Path,
    audit: bool,
) -> list[str]:
    cmd = [
        "python",
        "examples/performance/benchmark_transport_parallel_scaling.py",
        "--input",
        _display_path(input_path, repo_root=repo_root),
        "--workers",
        *[str(int(w)) for w in workers],
        "--repeats",
        str(int(repeats)),
        "--warmup",
        str(int(warmup)),
        "--global-warmup",
        str(int(global_warmup)),
        "--precond",
        str(precond),
        "--backend",
        str(backend),
        "--out-dir",
        _display_path(out_dir, repo_root=repo_root),
        "--cache-dir",
        _display_path(cache_dir, repo_root=repo_root),
        "--figure-name",
        str(figure_name),
    ]
    if audit:
        cmd.append("--audit")
    return cmd


def _build_transport_benchmark_plan(
    *,
    input_path: Path,
    rhs_mode: int,
    rhs_count: int,
    requested_workers: list[int],
    repeats: int,
    warmup: int,
    global_warmup: int,
    precond: str,
    backend: str,
    out_dir: Path,
    cache_dir: Path,
    figure_name: str,
    audit: bool = False,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    requested, workers, skipped_workers = _normalize_worker_counts(
        rhs_count=int(rhs_count),
        requested_workers=requested_workers,
    )
    timing_semantics = _timing_semantics(
        global_warmup=int(global_warmup),
        per_worker_warmup=int(warmup),
    )
    compile_amortization_gate = _compile_amortization_gate(
        timing_semantics=timing_semantics,
        global_warmup=int(global_warmup),
        per_worker_warmup=int(warmup),
        repeats=int(repeats),
        cache_dir=cache_dir,
        backend=str(backend),
        repo_root=repo_root,
    )
    payloads_by_workers = {
        str(w): _payloads_for_workers(rhs_count=int(rhs_count), workers=int(w)) for w in workers
    }
    worker_plan = [
        {
            "workers": int(w),
            "active_workers": min(int(w), int(rhs_count)),
            "payloads": payloads_by_workers[str(w)],
            "warmup_runs": max(int(warmup), 0),
            "timed_repeats": max(int(repeats), 1),
            "finite_task_ideal_speedup": float(rhs_count) / float(np.ceil(float(rhs_count) / float(w))),
        }
        for w in workers
    ]
    benchmark_command = _transport_benchmark_command(
        input_path=input_path,
        workers=workers,
        repeats=int(repeats),
        warmup=int(warmup),
        global_warmup=int(global_warmup),
        precond=str(precond),
        backend=str(backend),
        out_dir=out_dir,
        cache_dir=cache_dir,
        figure_name=str(figure_name),
        repo_root=repo_root,
        audit=bool(audit),
    )
    backend_l = str(backend).strip().lower()
    plan = {
        "artifact_kind": "benchmark_plan",
        "benchmark_kind": "transport_worker_scaling",
        "launches_solves": False,
        "input": input_path.name,
        "input_path": _display_path(input_path, repo_root=repo_root),
        "case": input_path.stem.replace(".input", ""),
        "rhs_mode": int(rhs_mode),
        "rhs_count": int(rhs_count),
        "which_rhs_values": list(range(1, int(rhs_count) + 1)),
        "requested_workers": requested,
        "workers": workers,
        "skipped_workers": skipped_workers,
        "worker_plan": worker_plan,
        "payloads_by_workers": payloads_by_workers,
        "payloads": payloads_by_workers[str(max(workers))],
        "precond": str(precond),
        "backend": backend_l,
        "timing_semantics": timing_semantics,
        "global_warmup": int(global_warmup),
        "per_worker_warmup": int(warmup),
        "repeats": int(repeats),
        "compile_amortization_gate": compile_amortization_gate,
        "estimated_transport_solve_calls": int(max(global_warmup, 0))
        + len(workers) * (max(int(warmup), 0) + max(int(repeats), 1)),
        "deterministic_payload_coverage": True,
        "deterministic_output_check": True,
        "measurement_scope": "transport_solve_only",
        "ideal_speedup_finite_rhs": [
            float(rhs_count) / float(np.ceil(float(rhs_count) / float(w))) for w in workers
        ],
        "release_gate_semantics": {
            "evaluated_by": "audit_transport_parallel_scaling_summary",
            "requires_measured_results": True,
            "min_speedup": 1.2,
            "min_efficiency": 0.5,
            "cold_start_rejected": True,
            "requires_compile_amortization_gate": True,
        },
        "memory_gate_semantics": {
            "status": "not_measured_in_plan",
            "gpu_preallocation_disabled": backend_l == "gpu",
            "gpu_allocator": "cuda_malloc_async" if backend_l == "gpu" else None,
            "cpu_devices_per_worker": 1 if backend_l == "cpu" else None,
        },
        "benchmark_command": benchmark_command,
    }
    plan["parallel_claim_scope"] = asdict(audit_parallel_scaling_claim_scope(plan))
    return plan


def _write_plan_json(plan: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def _run_scaling_audit(payload: dict[str, object]) -> None:
    audit = audit_transport_parallel_scaling_summary(payload)
    if audit.release_scaling_claim:
        print(
            "Transport-worker scaling audit passed "
            f"(workers={audit.claim_workers} speedup={audit.claim_speedup:.3g}x "
            f"efficiency={audit.claim_efficiency:.3g})",
            flush=True,
        )
        return
    failures = "\n".join(f"- {failure}" for failure in audit.failures)
    raise SystemExit(f"Transport-worker scaling audit failed:\n{failures}")


def _run_once(
    input_path: Path,
    *,
    workers: int,
    cache_dir: Path | None,
    precond: str | None,
    backend: str,
) -> float:
    os.environ["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    os.environ["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    if precond:
        os.environ["SFINCS_JAX_TRANSPORT_PRECOND"] = precond
    if cache_dir is not None:
        os.environ["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)
    if workers > 1:
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "process"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = str(workers)
    else:
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] = "off"
        os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS"] = "1"
    _configure_backend_env(workers=workers, backend=backend)

    nml = read_sfincs_input(input_path)
    t0 = time.perf_counter()
    res = solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1e-10,
        input_namelist=input_path,
        collect_transport_output_fields=False,
    )
    jax.block_until_ready(res.transport_matrix)
    return time.perf_counter() - t0


def _write_scaling_figure(
    payload: dict,
    out_dir: Path,
    *,
    figure_name: str = "transport_parallel_scaling.png",
) -> Path | None:
    """Write the transport-worker scaling figure from a benchmark payload."""
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")
        return None

    results = payload["results"]
    w = np.array([r["workers"] for r in results], dtype=int)
    mean_s = np.array([r["mean_s"] for r in results], dtype=float)
    speedup = np.array([r.get("speedup", np.nan) for r in results], dtype=float)
    ideal = np.asarray(payload.get("ideal_speedup_finite_rhs", w), dtype=float)
    if ideal.shape != w.shape:
        ideal = w.astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].plot(w, mean_s, "o-", label="measured")
    axes[0].set_xlabel("workers")
    axes[0].set_ylabel("time (s)")
    axes[0].set_title("Runtime vs workers")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(w, speedup, "o-", label="measured")
    axes[1].plot(w, ideal, "--", label="ideal")
    axes[1].set_xlabel("workers")
    axes[1].set_ylabel("speedup")
    axes[1].set_title("Speedup vs workers")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.suptitle(f"Parallel whichRHS scaling ({payload.get('backend', 'unknown')}): {payload['case']}", y=1.02)
    fig.tight_layout()
    fig_path = Path(out_dir) / str(figure_name)
    fig.savefig(fig_path, dpi=200)
    plt.close(fig)
    print(f"Saved figure -> {fig_path}")
    return fig_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark parallel whichRHS scaling.")
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "transport_parallel_2min.input.namelist"
    default_out = repo_root / "examples" / "performance" / "output" / "transport_parallel_scaling"
    default_cache = default_out / "jax_cache"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to input.namelist for RHSMode=2/3 transport matrix case.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=list(range(1, 5)),
        help="Worker counts to benchmark (default 1..4).",
    )
    parser.add_argument("--repeats", type=int, default=2, help="Repeats per worker count.")
    parser.add_argument("--warmup", type=int, default=0, help="Warmup runs per worker count.")
    parser.add_argument(
        "--global-warmup",
        type=int,
        default=1,
        help="Warmup runs before benchmarking (uses workers=1).",
    )
    parser.add_argument(
        "--precond",
        type=str,
        default="xmg",
        help="Transport preconditioner to use during the benchmark (default: xmg).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="cpu",
        choices=("cpu", "gpu"),
        help="Parallel transport backend to benchmark.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help="Output directory for JSON and figure.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=default_cache,
        help="Persistent JAX cache directory.",
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help="Regenerate JSON and figure from an existing benchmark payload without rerunning solves.",
    )
    parser.add_argument(
        "--figure-name",
        default="transport_parallel_scaling.png",
        help="Figure filename to write inside --out-dir.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Fail fast if the saved/new payload does not pass the transport-worker release scaling gate.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a deterministic benchmark plan JSON without launching solves.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Path for --plan-only JSON (default: --out-dir/transport_parallel_benchmark_plan.json).",
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_json is not None and args.plan_only:
        raise SystemExit("--plan-only cannot be combined with --from-json")

    if args.from_json is not None:
        payload = json.loads(Path(args.from_json).read_text())
        if args.audit:
            _run_scaling_audit(payload)
        _write_scaling_figure(payload, out_dir, figure_name=str(args.figure_name))
        return

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    rhs_mode = _rhs_mode_from_namelist(input_path)
    rhs_count = transport_matrix_size_from_rhs_mode(rhs_mode)

    requested_workers, workers, skipped_workers = _normalize_worker_counts(
        rhs_count=int(rhs_count),
        requested_workers=[int(w) for w in args.workers],
    )

    if args.plan_only:
        plan = _build_transport_benchmark_plan(
            input_path=input_path,
            rhs_mode=int(rhs_mode),
            rhs_count=int(rhs_count),
            requested_workers=requested_workers,
            repeats=int(args.repeats),
            warmup=int(args.warmup),
            global_warmup=int(args.global_warmup),
            precond=str(args.precond),
            backend=str(args.backend),
            out_dir=out_dir,
            cache_dir=args.cache_dir,
            figure_name=str(args.figure_name),
            audit=bool(args.audit),
        )
        plan_path = args.plan_json if args.plan_json is not None else out_dir / "transport_parallel_benchmark_plan.json"
        _write_plan_json(plan, plan_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    cache_dir = args.cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    if skipped_workers:
        print(
            "Skipping worker counts above independent RHS task count "
            f"(rhs_count={int(rhs_count)} skipped={skipped_workers})",
            flush=True,
        )

    if args.global_warmup and args.global_warmup > 0:
        for i in range(int(args.global_warmup)):
            print(
                f"[warmup-global {i + 1}/{int(args.global_warmup)}] workers=1 starting",
                flush=True,
            )
            _run_once(input_path, workers=1, cache_dir=cache_dir, precond=args.precond, backend=args.backend)
            print(
                f"[warmup-global {i + 1}/{int(args.global_warmup)}] workers=1 done",
                flush=True,
            )

    results = []
    for w in workers:
        print(f"[worker {w}] starting warmups={max(args.warmup, 0)} repeats={max(args.repeats, 1)}", flush=True)
        for i in range(max(args.warmup, 0)):
            print(f"[worker {w}] warmup {i + 1}/{max(args.warmup, 0)} starting", flush=True)
            _run_once(input_path, workers=w, cache_dir=cache_dir, precond=args.precond, backend=args.backend)
            print(f"[worker {w}] warmup {i + 1}/{max(args.warmup, 0)} done", flush=True)
        times = []
        for i in range(max(args.repeats, 1)):
            print(f"[worker {w}] repeat {i + 1}/{max(args.repeats, 1)} starting", flush=True)
            dt = _run_once(input_path, workers=w, cache_dir=cache_dir, precond=args.precond, backend=args.backend)
            times.append(dt)
            print(f"[worker {w}] repeat {i + 1}/{max(args.repeats, 1)} done in {dt:.3f}s", flush=True)
        times = np.asarray(times, dtype=float)
        results.append(
            {
                "workers": w,
                "mean_s": float(np.mean(times)),
                "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
                "samples": [float(v) for v in times],
                "payloads": _payloads_for_workers(rhs_count=rhs_count, workers=w),
            }
        )
        print(f"workers={w} mean_s={results[-1]['mean_s']:.3f} std_s={results[-1]['std_s']:.3f}", flush=True)

    # Normalize speedup to 1 worker.
    base = next((r for r in results if r["workers"] == 1), None)
    if base is not None and base["mean_s"] > 0:
        for r in results:
            r["speedup"] = float(base["mean_s"] / r["mean_s"])
    else:
        for r in results:
            r["speedup"] = None

    payload = {
        "benchmark_kind": "transport_worker_scaling",
        "input": input_path.name,
        "case": input_path.stem.replace(".input", ""),
        "rhs_mode": int(rhs_mode),
        "rhs_count": int(rhs_count),
        "which_rhs_values": list(range(1, int(rhs_count) + 1)),
        "workers": workers,
        "results": results,
        "precond": args.precond,
        "backend": args.backend,
        "timing_semantics": _timing_semantics(
            global_warmup=int(args.global_warmup),
            per_worker_warmup=int(args.warmup),
        ),
        "global_warmup": int(args.global_warmup),
        "per_worker_warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "compile_amortization_gate": _compile_amortization_gate(
            timing_semantics=_timing_semantics(
                global_warmup=int(args.global_warmup),
                per_worker_warmup=int(args.warmup),
            ),
            global_warmup=int(args.global_warmup),
            per_worker_warmup=int(args.warmup),
            repeats=int(args.repeats),
            cache_dir=cache_dir,
            backend=str(args.backend),
            repo_root=repo_root,
        ),
        "deterministic_payload_coverage": True,
        "deterministic_output_check": True,
        "measurement_scope": "transport_solve_only",
        "payloads": _payloads_for_workers(rhs_count=rhs_count, workers=max(workers)),
    }
    if str(args.backend).strip().lower() == "gpu":
        payload["gpu_device_count"] = int(max(workers))
        payload["visible_gpu_ids"] = [str(i) for i in range(int(max(workers)))]
    payload["ideal_speedup_finite_rhs"] = [
        float(rhs_count) / float(np.ceil(float(rhs_count) / float(w))) for w in workers
    ]

    json_path = out_dir / "transport_parallel_scaling.json"
    json_path.write_text(json.dumps(payload, indent=2))

    if args.audit:
        _run_scaling_audit(payload)

    _write_scaling_figure(payload, out_dir, figure_name=str(args.figure_name))


if __name__ == "__main__":
    main()
