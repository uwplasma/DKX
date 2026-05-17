from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import jax
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.transport_parallel_policy import (
    audit_parallel_scaling_claim_scope,
    audit_sharded_solve_scaling_summary,
)
from sfincs_jax.transport_parallel_sharding import (
    plan_compiled_sharded_operator_reuse,
    plan_sharded_solve_deterministic_output_gate,
    plan_single_case_sharded_solve,
)
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres


def _normalized_backend(backend: str) -> str:
    value = str(backend).strip().lower()
    if value in {"", "auto"}:
        return "auto"
    if value in {"cpu", "gpu"}:
        return value
    raise ValueError(f"Unsupported backend {backend!r}")


def _configure_backend_env(*, env: dict[str, str], devices: int, backend: str) -> None:
    backend_norm = _normalized_backend(backend)
    if backend_norm == "gpu":
        env.pop("SFINCS_JAX_CPU_DEVICES", None)
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(max(1, int(devices))))
        env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        env.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
        env.setdefault("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", "1")
        return
    env["SFINCS_JAX_CPU_DEVICES"] = str(int(devices))
    if backend_norm == "cpu":
        env.pop("CUDA_VISIBLE_DEVICES", None)


def _configure_solver_env(
    *,
    env: dict[str, str],
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    rhs1_precond: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
) -> None:
    env["SFINCS_JAX_MATVEC_SHARD_AXIS"] = shard_axis
    env["SFINCS_JAX_GMRES_DISTRIBUTED"] = gmres_distributed
    env["SFINCS_JAX_DISTRIBUTED_KRYLOV"] = distributed_krylov
    env["SFINCS_JAX_AUTO_SHARD"] = "0"
    env["SFINCS_JAX_IMPLICIT_SOLVE"] = "1"
    env["SFINCS_JAX_SHARD_PAD"] = "1"
    env["SFINCS_JAX_FORTRAN_STDOUT"] = "0"
    env["SFINCS_JAX_SOLVER_ITER_STATS"] = "0"
    if rhs1_precond:
        env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] = rhs1_precond
    else:
        env.pop("SFINCS_JAX_RHSMODE1_PRECONDITIONER", None)
    if shard_axis in {"theta", "zeta"} and str(gmres_distributed).strip().lower() not in {"", "0", "false", "no", "off"}:
        env.setdefault("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", "1")
    if periodic_stencil_on_sharded == "off":
        env["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "0"
    elif periodic_stencil_on_sharded == "on":
        env["SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED"] = "1"
    else:
        env.pop("SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED", None)
    if schwarz_coarse_levels is None:
        env.pop("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", None)
    else:
        env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS"] = str(int(schwarz_coarse_levels))
    if schwarz_coarse_steps is None:
        env.pop("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS", None)
    else:
        env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS"] = str(int(schwarz_coarse_steps))
    if schwarz_coarse_damp is None:
        env.pop("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP", None)
    else:
        env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP"] = str(float(schwarz_coarse_damp))


def _configure_benchmark_subprocess_env(env: dict[str, str]) -> None:
    """Keep child-process benchmark logs focused on progress and timings."""

    for key in ("TF_CPP_MIN_LOG_LEVEL", "GLOG_minloglevel", "ABSL_MIN_LOG_LEVEL"):
        try:
            current = int(env.get(key, "0"))
        except ValueError:
            current = 0
        env[key] = str(max(current, 2))


def _display_path(path: Path, *, repo_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(repo_root))
    except ValueError:
        return str(path)


def _normalize_device_counts(requested_devices: list[int]) -> list[int]:
    devices = sorted({int(d) for d in requested_devices if int(d) >= 1})
    return devices or [1]


def _timing_semantics(
    *,
    global_warmup: int,
    per_device_warmup: int,
    inner_warmup_solves: int,
) -> str:
    if int(inner_warmup_solves) > 0:
        return "hot_solve"
    if int(per_device_warmup) > 0 or int(global_warmup) > 0:
        return "cache_warm"
    return "cold_start"


def _operator_reuse_gate(
    *,
    timing_semantics: str,
    global_warmup: int,
    per_device_warmup: int,
    inner_warmup_solves: int,
    repeats: int,
    nsolve: int,
    cache_dir: Path | None,
    repo_root: Path,
) -> dict[str, object]:
    return plan_compiled_sharded_operator_reuse(
        benchmark_kind="single_case_sharded_solve",
        timing_semantics=str(timing_semantics),
        global_warmup_runs=int(global_warmup),
        per_device_warmup_runs=int(per_device_warmup),
        inner_warmup_runs=int(inner_warmup_solves),
        timed_repeats=max(int(repeats), 1),
        work_units_per_sample=max(int(nsolve), 1),
        compile_cache_dir=None if cache_dir is None else _display_path(cache_dir, repo_root=repo_root),
        persistent_compile_cache=cache_dir is not None,
        compile_in_timed_region=False,
    ).to_dict()


def _deterministic_output_gate(*, devices: list[int]) -> dict[str, object]:
    return plan_sharded_solve_deterministic_output_gate(
        baseline_devices=1,
        comparison_devices=max(int(d) for d in devices),
    ).to_dict()


def _run_scaling_audit(payload: dict[str, object]) -> None:
    audit = audit_sharded_solve_scaling_summary(payload)
    if audit.ci_gate_pass:
        print(
            "Sharded-solve scaling audit passed "
            f"(experimental; devices={audit.claim_devices} speedup={audit.claim_speedup:.3g}x)",
            flush=True,
        )
        return
    failures = "\n".join(f"- {failure}" for failure in audit.failures)
    raise SystemExit(f"Sharded-solve scaling audit failed:\n{failures}")


def _run_once_args(
    *,
    input_path: Path | str,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    inner_warmup_solves: int,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
) -> list[str]:
    args = [
        "--run-once",
        "--input",
        str(input_path),
        "--nsolve",
        str(int(nsolve)),
        "--inner-warmup-solves",
        str(int(inner_warmup_solves)),
        "--shard-axis",
        str(shard_axis),
        "--gmres-distributed",
        str(gmres_distributed),
        "--distributed-krylov",
        str(distributed_krylov),
        "--periodic-stencil-on-sharded",
        str(periodic_stencil_on_sharded),
        "--backend",
        str(backend),
    ]
    if rhs1_precond:
        args.extend(["--rhs1-precond", str(rhs1_precond)])
    if schwarz_coarse_levels is not None:
        args.extend(["--schwarz-coarse-levels", str(int(schwarz_coarse_levels))])
    if schwarz_coarse_steps is not None:
        args.extend(["--schwarz-coarse-steps", str(int(schwarz_coarse_steps))])
    if schwarz_coarse_damp is not None:
        args.extend(["--schwarz-coarse-damp", str(float(schwarz_coarse_damp))])
    return args


def _run_once_command(
    *,
    input_path: Path,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    inner_warmup_solves: int,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
) -> list[str]:
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *_run_once_args(
            input_path=input_path,
            shard_axis=shard_axis,
            gmres_distributed=gmres_distributed,
            distributed_krylov=distributed_krylov,
            periodic_stencil_on_sharded=periodic_stencil_on_sharded,
            nsolve=nsolve,
            inner_warmup_solves=inner_warmup_solves,
            rhs1_precond=rhs1_precond,
            backend=backend,
            schwarz_coarse_levels=schwarz_coarse_levels,
            schwarz_coarse_steps=schwarz_coarse_steps,
            schwarz_coarse_damp=schwarz_coarse_damp,
        ),
    ]


def _benchmark_command(
    *,
    input_path: Path,
    devices: list[int],
    warmup: int,
    repeats: int,
    nsolve: int,
    inner_warmup_solves: int,
    sample_timeout_s: float,
    global_warmup: int,
    out_dir: Path,
    cache_dir: Path,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
    audit: bool,
    repo_root: Path,
) -> list[str]:
    cmd = [
        "python",
        "examples/performance/benchmark_sharded_solve_scaling.py",
        "--input",
        _display_path(input_path, repo_root=repo_root),
        "--devices",
        *[str(int(d)) for d in devices],
        "--warmup",
        str(int(warmup)),
        "--repeats",
        str(int(repeats)),
        "--nsolve",
        str(int(nsolve)),
        "--inner-warmup-solves",
        str(int(inner_warmup_solves)),
        "--sample-timeout-s",
        str(float(sample_timeout_s)),
        "--global-warmup",
        str(int(global_warmup)),
        "--out-dir",
        _display_path(out_dir, repo_root=repo_root),
        "--cache-dir",
        _display_path(cache_dir, repo_root=repo_root),
        "--shard-axis",
        str(shard_axis),
        "--gmres-distributed",
        str(gmres_distributed),
        "--distributed-krylov",
        str(distributed_krylov),
        "--periodic-stencil-on-sharded",
        str(periodic_stencil_on_sharded),
        "--backend",
        str(backend),
    ]
    if rhs1_precond:
        cmd.extend(["--rhs1-precond", str(rhs1_precond)])
    if schwarz_coarse_levels is not None:
        cmd.extend(["--schwarz-coarse-levels", str(int(schwarz_coarse_levels))])
    if schwarz_coarse_steps is not None:
        cmd.extend(["--schwarz-coarse-steps", str(int(schwarz_coarse_steps))])
    if schwarz_coarse_damp is not None:
        cmd.extend(["--schwarz-coarse-damp", str(float(schwarz_coarse_damp))])
    if audit:
        cmd.append("--audit")
    return cmd


def _device_env_preview(
    *,
    devices: int,
    cache_dir: Path | None,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
    repo_root: Path,
) -> dict[str, str]:
    env: dict[str, str] = {}
    _configure_benchmark_subprocess_env(env)
    _configure_backend_env(env=env, devices=int(devices), backend=backend)
    _configure_solver_env(
        env=env,
        shard_axis=shard_axis,
        gmres_distributed=gmres_distributed,
        distributed_krylov=distributed_krylov,
        periodic_stencil_on_sharded=periodic_stencil_on_sharded,
        rhs1_precond=rhs1_precond,
        schwarz_coarse_levels=schwarz_coarse_levels,
        schwarz_coarse_steps=schwarz_coarse_steps,
        schwarz_coarse_damp=schwarz_coarse_damp,
    )
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = _display_path(cache_dir, repo_root=repo_root)
    return env


def _build_sharded_solve_benchmark_plan(
    *,
    input_path: Path,
    devices: list[int],
    warmup: int,
    repeats: int,
    nsolve: int,
    inner_warmup_solves: int,
    sample_timeout_s: float,
    global_warmup: int,
    out_dir: Path,
    cache_dir: Path,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
    audit: bool = False,
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[2]
    normalized_devices = _normalize_device_counts(devices)
    input_display = _display_path(input_path, repo_root=repo_root)
    timing_semantics = _timing_semantics(
        global_warmup=int(global_warmup),
        per_device_warmup=int(warmup),
        inner_warmup_solves=int(inner_warmup_solves),
    )
    operator_reuse_gate = _operator_reuse_gate(
        timing_semantics=timing_semantics,
        global_warmup=int(global_warmup),
        per_device_warmup=int(warmup),
        inner_warmup_solves=int(inner_warmup_solves),
        repeats=int(repeats),
        nsolve=int(nsolve),
        cache_dir=cache_dir,
        repo_root=repo_root,
    )
    device_plan = []
    for d in normalized_devices:
        sharding_plan = plan_single_case_sharded_solve(
            requested_devices=int(d),
            backend=str(backend),
            available_device_count=int(d),
            available_device_ids=[str(i) for i in range(int(d))],
            rhs_mode=1,
            shard_axis=str(shard_axis),
            benchmark_kind="single_case_sharded_solve",
            task_count=1,
            release_scaling_claim=False,
            experimental_single_case_scaling=True,
            scaling_status="experimental_single_case_sharding",
        )
        env = _device_env_preview(
            devices=d,
            cache_dir=cache_dir,
            shard_axis=str(shard_axis),
            gmres_distributed=str(gmres_distributed),
            distributed_krylov=str(distributed_krylov),
            periodic_stencil_on_sharded=str(periodic_stencil_on_sharded),
            rhs1_precond=str(rhs1_precond),
            backend=str(backend),
            schwarz_coarse_levels=schwarz_coarse_levels,
            schwarz_coarse_steps=schwarz_coarse_steps,
            schwarz_coarse_damp=schwarz_coarse_damp,
            repo_root=repo_root,
        )
        device_plan.append(
            {
                "devices": int(d),
                "warmup_runs": max(int(warmup), 0),
                "timed_repeats": max(int(repeats), 1),
                "nsolve_per_sample": max(int(nsolve), 1),
                "inner_warmup_solves": max(int(inner_warmup_solves), 0),
                "sample_timeout_s": float(sample_timeout_s),
                "sharding_plan": sharding_plan.to_dict(),
                "env": env,
                "run_once_command": [
                    "python",
                    "examples/performance/benchmark_sharded_solve_scaling.py",
                    *_run_once_args(
                        input_path=input_display,
                        shard_axis=str(shard_axis),
                        gmres_distributed=str(gmres_distributed),
                        distributed_krylov=str(distributed_krylov),
                        periodic_stencil_on_sharded=str(periodic_stencil_on_sharded),
                        nsolve=int(nsolve),
                        inner_warmup_solves=int(inner_warmup_solves),
                        rhs1_precond=str(rhs1_precond),
                        backend=str(backend),
                        schwarz_coarse_levels=schwarz_coarse_levels,
                        schwarz_coarse_steps=schwarz_coarse_steps,
                        schwarz_coarse_damp=schwarz_coarse_damp,
                    ),
                ],
            }
        )
    plan = {
        "artifact_kind": "benchmark_plan",
        "benchmark_kind": "single_case_sharded_solve",
        "scaling_status": "experimental_single_case_sharding",
        "experimental_single_case_scaling": True,
        "release_scaling_claim": False,
        "launches_solves": False,
        "input": input_path.name,
        "input_path": input_display,
        "case": input_path.stem.replace(".input", ""),
        "task_count": 1,
        "devices": normalized_devices,
        "device_plan": device_plan,
        "shard_axis": str(shard_axis),
        "gmres_distributed": str(gmres_distributed),
        "distributed_krylov": str(distributed_krylov),
        "periodic_stencil_on_sharded": str(periodic_stencil_on_sharded),
        "nsolve": int(nsolve),
        "inner_warmup_solves": int(inner_warmup_solves),
        "sample_timeout_s": float(sample_timeout_s),
        "rhs1_precond": str(rhs1_precond),
        "backend": str(backend),
        "schwarz_coarse_levels": schwarz_coarse_levels,
        "schwarz_coarse_steps": schwarz_coarse_steps,
        "schwarz_coarse_damp": schwarz_coarse_damp,
        "timing_semantics": timing_semantics,
        "operator_reuse_gate": operator_reuse_gate,
        "global_warmup": int(global_warmup),
        "per_device_warmup": int(warmup),
        "repeats": int(repeats),
        "deterministic_output_check": False,
        "deterministic_output_gate": _deterministic_output_gate(devices=normalized_devices),
        "estimated_child_process_samples": int(max(global_warmup, 0))
        + len(normalized_devices) * (max(int(warmup), 0) + max(int(repeats), 1)),
        "speedup_gate_semantics": {
            "release_scaling_claim": False,
            "evaluated_by": "audit_sharded_solve_scaling_summary",
            "gate_scope": "schema_and_honesty_only",
            "single_case_strong_scaling_is_experimental": True,
            "requires_operator_reuse_gate": True,
            "requires_deterministic_output_gate_for_claim": True,
        },
        "memory_gate_semantics": {
            "status": "bounded_by_child_timeout_and_allocator_settings",
            "sample_timeout_s": float(sample_timeout_s),
            "child_process_timeout_enabled": float(sample_timeout_s) > 0.0,
            "gpu_preallocation_disabled": _normalized_backend(backend) == "gpu",
            "gpu_allocator": "cuda_malloc_async" if _normalized_backend(backend) == "gpu" else None,
        },
        "benchmark_command": _benchmark_command(
            input_path=input_path,
            devices=normalized_devices,
            warmup=int(warmup),
            repeats=int(repeats),
            nsolve=int(nsolve),
            inner_warmup_solves=int(inner_warmup_solves),
            sample_timeout_s=float(sample_timeout_s),
            global_warmup=int(global_warmup),
            out_dir=out_dir,
            cache_dir=cache_dir,
            shard_axis=str(shard_axis),
            gmres_distributed=str(gmres_distributed),
            distributed_krylov=str(distributed_krylov),
            periodic_stencil_on_sharded=str(periodic_stencil_on_sharded),
            rhs1_precond=str(rhs1_precond),
            backend=str(backend),
            schwarz_coarse_levels=schwarz_coarse_levels,
            schwarz_coarse_steps=schwarz_coarse_steps,
            schwarz_coarse_damp=schwarz_coarse_damp,
            audit=bool(audit),
            repo_root=repo_root,
        ),
    }
    plan["parallel_claim_scope"] = asdict(audit_parallel_scaling_claim_scope(plan))
    return plan


def _write_plan_json(plan: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")


def _run_once(
    input_path: Path,
    *,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    inner_warmup_solves: int,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
) -> float:
    _normalized_backend(backend)
    _configure_solver_env(
        env=os.environ,
        shard_axis=shard_axis,
        gmres_distributed=gmres_distributed,
        distributed_krylov=distributed_krylov,
        periodic_stencil_on_sharded=periodic_stencil_on_sharded,
        rhs1_precond=rhs1_precond,
        schwarz_coarse_levels=schwarz_coarse_levels,
        schwarz_coarse_steps=schwarz_coarse_steps,
        schwarz_coarse_damp=schwarz_coarse_damp,
    )
    nml = read_sfincs_input(input_path)
    for _ in range(max(0, int(inner_warmup_solves))):
        warm = solve_v3_full_system_linear_gmres(
            nml=nml,
            tol=1e-10,
        )
        jax.block_until_ready(warm.x)
    t0 = time.perf_counter()
    for _ in range(max(1, int(nsolve))):
        res = solve_v3_full_system_linear_gmres(
            nml=nml,
            tol=1e-10,
        )
        jax.block_until_ready(res.x)
    return time.perf_counter() - t0


def _run_once_subprocess(
    *,
    input_path: Path,
    devices: int,
    cache_dir: Path | None,
    shard_axis: str,
    gmres_distributed: str,
    distributed_krylov: str,
    periodic_stencil_on_sharded: str,
    nsolve: int,
    inner_warmup_solves: int,
    sample_timeout_s: float | None,
    rhs1_precond: str,
    backend: str,
    schwarz_coarse_levels: int | None,
    schwarz_coarse_steps: int | None,
    schwarz_coarse_damp: float | None,
) -> float:
    env = os.environ.copy()
    _configure_benchmark_subprocess_env(env)
    _configure_backend_env(env=env, devices=devices, backend=backend)
    _configure_solver_env(
        env=env,
        shard_axis=shard_axis,
        gmres_distributed=gmres_distributed,
        distributed_krylov=distributed_krylov,
        periodic_stencil_on_sharded=periodic_stencil_on_sharded,
        rhs1_precond=rhs1_precond,
        schwarz_coarse_levels=schwarz_coarse_levels,
        schwarz_coarse_steps=schwarz_coarse_steps,
        schwarz_coarse_damp=schwarz_coarse_damp,
    )
    if cache_dir is not None:
        env["JAX_COMPILATION_CACHE_DIR"] = str(cache_dir)

    cmd = _run_once_command(
        input_path=input_path,
        shard_axis=str(shard_axis),
        gmres_distributed=str(gmres_distributed),
        distributed_krylov=str(distributed_krylov),
        periodic_stencil_on_sharded=str(periodic_stencil_on_sharded),
        nsolve=int(nsolve),
        inner_warmup_solves=int(inner_warmup_solves),
        rhs1_precond=str(rhs1_precond),
        backend=str(backend),
        schwarz_coarse_levels=schwarz_coarse_levels,
        schwarz_coarse_steps=schwarz_coarse_steps,
        schwarz_coarse_damp=schwarz_coarse_damp,
    )
    timeout = None if sample_timeout_s is None or sample_timeout_s <= 0 else float(sample_timeout_s)
    try:
        out = subprocess.check_output(cmd, env=env, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Timed out after {timeout:.1f}s while benchmarking devices={devices}; "
            "increase --sample-timeout-s or use a smaller case."
        ) from exc
    return float(out.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark sharded RHSMode=1 solve scaling.")
    repo_root = Path(__file__).resolve().parents[2]
    default_input = repo_root / "examples" / "performance" / "rhsmode1_sharded.input.namelist"
    default_out = repo_root / "examples" / "performance" / "output" / "sharded_solve_scaling"
    default_cache = default_out / "jax_cache"

    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to RHSMode=1 input.namelist for sharded solve benchmarking.",
    )
    parser.add_argument(
        "--devices",
        type=int,
        nargs="+",
        default=list(range(1, 5)),
        help="CPU device counts to benchmark (default 1..4).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup runs per device count.",
    )
    parser.add_argument("--repeats", type=int, default=1, help="Timing repeats per device count.")
    parser.add_argument(
        "--nsolve",
        type=int,
        default=1,
        help="Number of RHSMode=1 solves per timed sample.",
    )
    parser.add_argument(
        "--inner-warmup-solves",
        type=int,
        default=0,
        help="Untimed solves inside each child process before timing; use 1 to report hot-solve scaling.",
    )
    parser.add_argument(
        "--sample-timeout-s",
        type=float,
        default=0.0,
        help="Optional timeout for each child-process sample; 0 disables the timeout.",
    )
    parser.add_argument(
        "--global-warmup",
        type=int,
        default=1,
        help="Warmup runs before benchmarking (uses devices=1).",
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
        "--run-once",
        action="store_true",
        help="Run once and print wall time (internal).",
    )
    parser.add_argument(
        "--shard-axis",
        type=str,
        default="theta",
        help="Matvec shard axis for distributed solve benchmark (theta/zeta/x/flat/off).",
    )
    parser.add_argument(
        "--gmres-distributed",
        type=str,
        default="1",
        help="Value for SFINCS_JAX_GMRES_DISTRIBUTED (default: 1).",
    )
    parser.add_argument(
        "--distributed-krylov",
        type=str,
        default="auto",
        help="Value for SFINCS_JAX_DISTRIBUTED_KRYLOV (default: auto).",
    )
    parser.add_argument(
        "--periodic-stencil-on-sharded",
        type=str,
        default="auto",
        choices=("auto", "on", "off"),
        help="Control SFINCS_JAX_PERIODIC_STENCIL_ON_SHARDED for this benchmark.",
    )
    parser.add_argument(
        "--rhs1-precond",
        type=str,
        default="",
        help="Optional SFINCS_JAX_RHSMODE1_PRECONDITIONER override for this benchmark.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=("auto", "cpu", "gpu"),
        help="Benchmark backend. 'cpu' uses SFINCS_JAX_CPU_DEVICES; 'gpu' uses CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--schwarz-coarse-levels",
        type=int,
        default=None,
        help="Optional override for SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS.",
    )
    parser.add_argument(
        "--schwarz-coarse-steps",
        type=int,
        default=None,
        help="Optional override for SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS.",
    )
    parser.add_argument(
        "--schwarz-coarse-damp",
        type=float,
        default=None,
        help="Optional override for SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP.",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Fail fast if the saved payload does not pass the sharded-solve schema/honesty gate.",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Write a deterministic benchmark plan JSON without launching child solve processes.",
    )
    parser.add_argument(
        "--plan-json",
        type=Path,
        default=None,
        help="Path for --plan-only JSON (default: --out-dir/sharded_solve_benchmark_plan.json).",
    )
    args = parser.parse_args()

    input_path = args.input
    if not input_path.exists():
        raise FileNotFoundError(str(input_path))

    if args.run_once:
        dt = _run_once(
            input_path,
            shard_axis=str(args.shard_axis),
            gmres_distributed=str(args.gmres_distributed),
            distributed_krylov=str(args.distributed_krylov),
            periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
            nsolve=int(args.nsolve),
            inner_warmup_solves=int(args.inner_warmup_solves),
            rhs1_precond=str(args.rhs1_precond),
            backend=str(args.backend),
            schwarz_coarse_levels=args.schwarz_coarse_levels,
            schwarz_coarse_steps=args.schwarz_coarse_steps,
            schwarz_coarse_damp=args.schwarz_coarse_damp,
        )
        print(f"{dt:.6f}")
        return

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir

    devices = _normalize_device_counts([int(d) for d in args.devices])

    if args.plan_only:
        plan = _build_sharded_solve_benchmark_plan(
            input_path=input_path,
            devices=devices,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            nsolve=int(args.nsolve),
            inner_warmup_solves=int(args.inner_warmup_solves),
            sample_timeout_s=float(args.sample_timeout_s),
            global_warmup=int(args.global_warmup),
            out_dir=out_dir,
            cache_dir=cache_dir,
            shard_axis=str(args.shard_axis),
            gmres_distributed=str(args.gmres_distributed),
            distributed_krylov=str(args.distributed_krylov),
            periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
            rhs1_precond=str(args.rhs1_precond),
            backend=str(args.backend),
            schwarz_coarse_levels=args.schwarz_coarse_levels,
            schwarz_coarse_steps=args.schwarz_coarse_steps,
            schwarz_coarse_damp=args.schwarz_coarse_damp,
            audit=bool(args.audit),
        )
        plan_path = args.plan_json if args.plan_json is not None else out_dir / "sharded_solve_benchmark_plan.json"
        _write_plan_json(plan, plan_path)
        print(json.dumps(plan, indent=2, sort_keys=True))
        return

    cache_dir.mkdir(parents=True, exist_ok=True)

    if args.global_warmup and args.global_warmup > 0:
        for _ in range(int(args.global_warmup)):
            _run_once_subprocess(
                input_path=input_path,
                devices=1,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                inner_warmup_solves=int(args.inner_warmup_solves),
                sample_timeout_s=float(args.sample_timeout_s),
                rhs1_precond=str(args.rhs1_precond),
                backend=str(args.backend),
                schwarz_coarse_levels=args.schwarz_coarse_levels,
                schwarz_coarse_steps=args.schwarz_coarse_steps,
                schwarz_coarse_damp=args.schwarz_coarse_damp,
            )

    results = []
    for d in devices:
        print(
            f"[device {d}] warmups={max(args.warmup, 0)} repeats={max(args.repeats, 1)} "
            f"shard_axis={args.shard_axis} gmres_distributed={args.gmres_distributed} "
            f"distributed_krylov={args.distributed_krylov} "
            f"stencil_on_sharded={args.periodic_stencil_on_sharded} "
            f"nsolve={int(args.nsolve)} inner_warmup_solves={int(args.inner_warmup_solves)} "
            f"rhs1_precond={args.rhs1_precond or 'auto'} "
            f"backend={args.backend} coarse_levels={args.schwarz_coarse_levels if args.schwarz_coarse_levels is not None else 'auto'} "
            f"coarse_steps={args.schwarz_coarse_steps if args.schwarz_coarse_steps is not None else 'auto'} "
            f"coarse_damp={args.schwarz_coarse_damp if args.schwarz_coarse_damp is not None else 'auto'}",
            flush=True,
        )
        for i in range(max(args.warmup, 0)):
            print(f"[device {d}] warmup {i + 1}/{max(args.warmup, 0)} starting", flush=True)
            _run_once_subprocess(
                input_path=input_path,
                devices=d,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                inner_warmup_solves=int(args.inner_warmup_solves),
                sample_timeout_s=float(args.sample_timeout_s),
                rhs1_precond=str(args.rhs1_precond),
                backend=str(args.backend),
                schwarz_coarse_levels=args.schwarz_coarse_levels,
                schwarz_coarse_steps=args.schwarz_coarse_steps,
                schwarz_coarse_damp=args.schwarz_coarse_damp,
            )
            print(f"[device {d}] warmup {i + 1}/{max(args.warmup, 0)} done", flush=True)
        times = []
        for i in range(max(args.repeats, 1)):
            print(f"[device {d}] repeat {i + 1}/{max(args.repeats, 1)} starting", flush=True)
            dt = _run_once_subprocess(
                input_path=input_path,
                devices=d,
                cache_dir=cache_dir,
                shard_axis=str(args.shard_axis),
                gmres_distributed=str(args.gmres_distributed),
                distributed_krylov=str(args.distributed_krylov),
                periodic_stencil_on_sharded=str(args.periodic_stencil_on_sharded),
                nsolve=int(args.nsolve),
                inner_warmup_solves=int(args.inner_warmup_solves),
                sample_timeout_s=float(args.sample_timeout_s),
                rhs1_precond=str(args.rhs1_precond),
                backend=str(args.backend),
                schwarz_coarse_levels=args.schwarz_coarse_levels,
                schwarz_coarse_steps=args.schwarz_coarse_steps,
                schwarz_coarse_damp=args.schwarz_coarse_damp,
            )
            times.append(dt)
            print(f"[device {d}] repeat {i + 1}/{max(args.repeats, 1)} done in {dt:.3f}s", flush=True)
        times = np.asarray(times, dtype=float)
        results.append(
            {
                "devices": d,
                "mean_s": float(np.mean(times)),
                "std_s": float(np.std(times, ddof=1)) if times.size > 1 else 0.0,
                "samples": [float(v) for v in times],
            }
        )
        print(f"devices={d} mean_s={results[-1]['mean_s']:.3f} std_s={results[-1]['std_s']:.3f}", flush=True)

    base = next((r for r in results if r["devices"] == 1), None)
    if base is not None and base["mean_s"] > 0:
        for r in results:
            r["speedup"] = float(base["mean_s"] / r["mean_s"])
    else:
        for r in results:
            r["speedup"] = None

    payload = {
        "benchmark_kind": "single_case_sharded_solve",
        "scaling_status": "experimental_single_case_sharding",
        "experimental_single_case_scaling": True,
        "release_scaling_claim": False,
        "input": input_path.name,
        "case": input_path.stem.replace(".input", ""),
        "task_count": 1,
        "devices": devices,
        "results": results,
        "shard_axis": str(args.shard_axis),
        "gmres_distributed": str(args.gmres_distributed),
        "distributed_krylov": str(args.distributed_krylov),
        "periodic_stencil_on_sharded": str(args.periodic_stencil_on_sharded),
        "nsolve": int(args.nsolve),
        "inner_warmup_solves": int(args.inner_warmup_solves),
        "sample_timeout_s": float(args.sample_timeout_s),
        "rhs1_precond": str(args.rhs1_precond),
        "backend": str(args.backend),
        "schwarz_coarse_levels": args.schwarz_coarse_levels,
        "schwarz_coarse_steps": args.schwarz_coarse_steps,
        "schwarz_coarse_damp": args.schwarz_coarse_damp,
        "timing_semantics": _timing_semantics(
            global_warmup=int(args.global_warmup),
            per_device_warmup=int(args.warmup),
            inner_warmup_solves=int(args.inner_warmup_solves),
        ),
        "operator_reuse_gate": _operator_reuse_gate(
            timing_semantics=_timing_semantics(
                global_warmup=int(args.global_warmup),
                per_device_warmup=int(args.warmup),
                inner_warmup_solves=int(args.inner_warmup_solves),
            ),
            global_warmup=int(args.global_warmup),
            per_device_warmup=int(args.warmup),
            inner_warmup_solves=int(args.inner_warmup_solves),
            repeats=int(args.repeats),
            nsolve=int(args.nsolve),
            cache_dir=cache_dir,
            repo_root=repo_root,
        ),
        "global_warmup": int(args.global_warmup),
        "per_device_warmup": int(args.warmup),
        "repeats": int(args.repeats),
        "deterministic_output_check": False,
        "deterministic_output_gate": _deterministic_output_gate(devices=devices),
    }
    if _normalized_backend(args.backend) == "gpu":
        payload["gpu_device_count"] = int(max(devices))
        payload["visible_gpu_ids"] = [str(i) for i in range(int(max(devices)))]

    json_path = out_dir / "sharded_solve_scaling.json"
    json_path.write_text(json.dumps(payload, indent=2))

    if args.audit:
        _run_scaling_audit(payload)

    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt  # noqa: PLC0415

        d = np.array([r["devices"] for r in results], dtype=int)
        mean_s = np.array([r["mean_s"] for r in results], dtype=float)
        speedup = np.array([r.get("speedup", np.nan) for r in results], dtype=float)

        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
        axes[0].plot(d, mean_s, "o-", label="measured")
        device_label = "GPU devices" if _normalized_backend(args.backend) == "gpu" else "CPU devices"
        axes[0].set_xlabel(device_label)
        axes[0].set_ylabel("time (s)")
        axes[0].set_title("Runtime vs devices")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(d, speedup, "o-", label="measured")
        axes[1].plot(d, d, "--", label="ideal")
        axes[1].set_xlabel(device_label)
        axes[1].set_ylabel("speedup")
        axes[1].set_title("Speedup vs devices")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(frameon=False)

        fig.suptitle(f"Sharded solve scaling: {payload['case']}", y=1.02)
        fig.tight_layout()
        fig_path = out_dir / "sharded_solve_scaling.png"
        fig.savefig(fig_path, dpi=200)
        print(f"Saved figure -> {fig_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"Matplotlib unavailable: {exc}")


if __name__ == "__main__":
    main()
