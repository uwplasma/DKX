#!/usr/bin/env python
"""Bounded PAS-TZ memory-fallback benchmark harness.

The production-resolution geometry-rich PAS lane should not be promoted by
heuristics. This script forces the matrix-free RHSMode=1 solver through the
``pas_tz`` memory-fallback path in short-lived subprocesses, so slow
preconditioner builds are recorded as bounded timeouts instead of hanging a
developer shell or CI job.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import tempfile
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "examples" / "sfincs_examples" / "geometryScheme4_2species_PAS_noEr" / "input.namelist"
DEFAULT_OUT = _REPO_ROOT / "examples" / "performance" / "output" / "pas_tz_memory_fallback_benchmark.json"
RESULT_MARKER = "__SFINCS_JAX_PAS_TZ_RESULT__="
GRID_OVERRIDE_KEYS = ("Ntheta", "Nzeta", "Nxi", "Nx")
MAX_DEFAULT_RUNTIME_S = 600.0
DEFAULT_MAX_RESIDUAL_NORM = 1.0e-3
DEFAULT_MIN_PROMOTION_SPEEDUP = 1.05
DEFAULT_MIN_PROMOTION_MEMORY_REDUCTION = 1.05
BACKEND_ALIASES = {
    "cuda": "gpu",
    "gpu": "gpu",
    "rocm": "gpu",
    "metal": "gpu",
    "tpu": "gpu",
    "cpu": "cpu",
}


def _tail_text(value: str | bytes | None, n: int = 4000) -> str:
    """Return a JSON-serializable tail from subprocess output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-int(n) :]


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be finite and >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark forced PAS-TZ memory fallback variants with hard timeouts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", nargs="+", default=["collision", "hybrid", "zeta", "theta", "tzfft"])
    parser.add_argument("--timeout-s", type=_positive_float, default=60.0)
    parser.add_argument("--stall-s", type=_positive_float, help="Per-row stall gate in seconds; defaults to timeout-s.")
    parser.add_argument("--maxiter", type=_positive_int, default=8)
    parser.add_argument("--restart", type=_positive_int, default=12)
    parser.add_argument("--tol", type=_positive_float, default=1.0e-6)
    parser.add_argument("--solve-method", default="incremental", help="Krylov solve method passed to the child solve.")
    parser.add_argument("--block", type=_positive_int, default=3)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--Ntheta", "--ntheta", dest="Ntheta", type=_positive_int, help="Override resolutionParameters.Ntheta.")
    parser.add_argument("--Nzeta", "--nzeta", dest="Nzeta", type=_positive_int, help="Override resolutionParameters.Nzeta.")
    parser.add_argument("--Nxi", "--nxi", dest="Nxi", type=_positive_int, help="Override resolutionParameters.Nxi.")
    parser.add_argument("--Nx", "--nx", dest="Nx", type=_positive_int, help="Override resolutionParameters.Nx.")
    parser.add_argument(
        "--max-rss-mb",
        type=_nonnegative_float,
        default=0.0,
        help="Fail the memory gate above this peak RSS; 0 disables the threshold.",
    )
    parser.add_argument(
        "--max-residual-norm",
        type=_nonnegative_float,
        default=DEFAULT_MAX_RESIDUAL_NORM,
        help="Fail the residual gate above this true residual norm; 0 requires finite residual only.",
    )
    parser.add_argument(
        "--expected-backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Optional CPU/GPU backend gate for real-solve provenance.",
    )
    parser.add_argument("--allow-solver-churn", action="store_true", help="Record but do not fail solver-path churn.")
    parser.add_argument(
        "--allow-long-run",
        action="store_true",
        help="Allow timeout/stall bounds above 600s. Defaults reject >10 minute probes.",
    )
    parser.add_argument(
        "--require-default-promotion-gate",
        action="store_true",
        help="Require each ok row to prove residual-clean runtime/RSS improvement over a baseline.",
    )
    parser.add_argument(
        "--baseline-elapsed-s",
        type=_positive_float,
        default=None,
        help="Baseline elapsed time used by --require-default-promotion-gate.",
    )
    parser.add_argument(
        "--baseline-rss-mb",
        type=_positive_float,
        default=None,
        help="Baseline peak RSS used by --require-default-promotion-gate.",
    )
    parser.add_argument(
        "--min-runtime-speedup",
        type=_positive_float,
        default=DEFAULT_MIN_PROMOTION_SPEEDUP,
        help="Minimum baseline/candidate runtime speedup for default-promotion eligibility.",
    )
    parser.add_argument(
        "--min-memory-reduction",
        type=_positive_float,
        default=DEFAULT_MIN_PROMOTION_MEMORY_REDUCTION,
        help="Minimum baseline/candidate RSS reduction for default-promotion eligibility.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write planned variants without running subprocesses.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _grid_overrides(args: argparse.Namespace) -> dict[str, int]:
    """Return requested positive grid overrides keyed by namelist variable."""
    overrides: dict[str, int] = {}
    for key in GRID_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        value_i = int(value)
        if value_i <= 0:
            raise ValueError(f"{key} override must be positive, got {value_i}")
        overrides[key] = value_i
    return overrides


def _optional_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return _optional_float(value)
    return value


def _effective_stall_s(args: argparse.Namespace) -> float:
    stall_s = getattr(args, "stall_s", None)
    return float(stall_s if stall_s is not None else args.timeout_s)


def _validate_runtime_bounds(args: argparse.Namespace) -> None:
    if bool(getattr(args, "allow_long_run", False)):
        return
    timeout_s = float(getattr(args, "timeout_s", 0.0))
    stall_s = _effective_stall_s(args)
    over = [f"timeout-s={timeout_s:g}" if timeout_s > MAX_DEFAULT_RUNTIME_S else ""]
    over.append(f"stall-s={stall_s:g}" if stall_s > MAX_DEFAULT_RUNTIME_S else "")
    over = [item for item in over if item]
    if over:
        raise ValueError(
            "default PAS-TZ probes are capped at 600s; pass --allow-long-run "
            f"for explicit longer probes ({', '.join(over)})"
        )


def _normalize_backend(value: Any) -> str:
    key = str(value or "").strip().lower()
    return BACKEND_ALIASES.get(key, key)


def _normalize_solver_method(value: Any) -> str:
    key = str(value or "").strip().lower().replace("-", "_")
    if key in {"", "auto", "default", "incremental", "gmres", "gmres_scipy"}:
        return "gmres"
    if key in {"lgmres", "lgmres_scipy"}:
        return "lgmres"
    if key in {"bicgstab", "short_recurrence", "shortrecurrence"}:
        return "bicgstab"
    return key


def _last_ksp_solver(messages: list[str]) -> str | None:
    pattern = re.compile(r"\bksp_iterations=\d+\s+solver=([A-Za-z0-9_.-]+)")
    for message in reversed(messages):
        match = pattern.search(str(message))
        if match:
            return match.group(1).strip().lower().replace("-", "_")
    return None


def _runtime_metadata(expected_backend: str) -> dict[str, Any]:
    import jax

    devices = list(jax.devices())
    platforms = sorted({str(getattr(device, "platform", "")) for device in devices if getattr(device, "platform", "")})
    return {
        "expected_backend": str(expected_backend),
        "jax_default_backend": str(jax.default_backend()),
        "jax_device_count": int(jax.device_count()),
        "jax_device_platforms": platforms,
    }


def _input_record(input_path: Path) -> str:
    try:
        return str(input_path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(input_path)


def _override_namelist_text(text: str, overrides: dict[str, int]) -> str:
    """Apply simple scalar grid overrides to an existing SFINCS namelist."""
    updated = text
    for key, value in overrides.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)[^\n!]*?(\s*(?:!.*)?)$", re.MULTILINE)
        updated, count = pattern.subn(rf"\g<1>{value}\2", updated, count=1)
        if count != 1:
            raise ValueError(f"Could not find active namelist assignment for {key}")
    return updated


def _write_child_input(input_path: Path, work_dir: Path, overrides: dict[str, int]) -> Path:
    """Write a temporary child input namelist with requested grid overrides."""
    child_input = work_dir / "input.namelist"
    text = input_path.read_text()
    child_input.write_text(_override_namelist_text(text, overrides))
    return child_input


def _variant_env(variant: str, *, block: int, overlap: int, maxiter: int, restart: int) -> dict[str, str]:
    """Return environment overrides for one forced PAS-TZ fallback variant."""
    variant_l = str(variant).strip().lower().replace("-", "_")
    variant_core = variant_l.removesuffix("_lgmres")
    fallback_variant = variant_core
    structured_levels = ""
    if variant_core in {"collision_tzfft", "collision_tzfft_correction", "tzfft_correction"}:
        fallback_variant = "collision"
    elif variant_core in {"tzfft_structured", "tzfft_structured_default", "tzfft_xmg_collision"}:
        fallback_variant = "tzfft"
        structured_levels = "xmg,collision"
    elif variant_core in {"tzfft_xmg", "tzfft_structured_xmg"}:
        fallback_variant = "tzfft"
        structured_levels = "xmg"
    elif variant_core in {"tzfft_collision", "tzfft_structured_collision"}:
        fallback_variant = "tzfft"
        structured_levels = "collision"
    env = {
        "SFINCS_JAX_FORTRAN_STDOUT": "0",
        "SFINCS_JAX_SOLVER_ITER_STATS": "1",
        "SFINCS_JAX_RHSMODE1_PRECONDITIONER": "pas_tz",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES": "1",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK": fallback_variant,
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK": str(int(block)),
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP": str(int(overlap)),
        "SFINCS_JAX_GMRES_MAXITER": str(int(maxiter)),
        "SFINCS_JAX_GMRES_RESTART": str(int(restart)),
    }
    if variant_core in {"collision_tzfft", "collision_tzfft_correction", "tzfft_correction"}:
        env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION"] = "tzfft"
    if structured_levels:
        env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS"] = structured_levels
    return env


def _variant_base(variant: str) -> str:
    """Return the fallback variant name without solver-method suffixes."""
    return str(variant).strip().lower().replace("-", "_").removesuffix("_lgmres")


def _variant_solve_method(variant: str, default: str) -> str:
    """Return the child solve method for a variant name."""
    variant_l = str(variant).strip().lower().replace("-", "_")
    if variant_l.endswith("_lgmres") or variant_l == "lgmres":
        return "lgmres"
    return str(default)


def _variant_provenance(variant: str, default_solve_method: str) -> dict[str, Any]:
    """Return reproducibility metadata for a planned variant."""
    solve_method = _variant_solve_method(variant, default_solve_method)
    variant_l = str(variant).strip().lower().replace("-", "_")
    return {
        "variant": str(variant),
        "base_variant": _variant_base(variant),
        "requested_solve_method": str(default_solve_method),
        "realized_solve_method": solve_method,
        "solve_method_source": "variant_suffix" if variant_l.endswith("_lgmres") or variant_l == "lgmres" else "plan_default",
        "lgmres_opt_in": solve_method == "lgmres" and str(default_solve_method) != "lgmres",
    }


def _phase_record(name: str, start_s: float, end_s: float, *, status: str = "ok") -> dict[str, Any]:
    """Build a compact phase timing record for benchmark JSON."""
    return {
        "name": name,
        "status": status,
        "elapsed_s": float(max(0.0, end_s - start_s)),
    }


def _child_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Run one forced fallback solve in the current process and return metrics."""
    import jax.numpy as jnp

    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.profiling import _resource_maxrss_to_mb
    from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres

    messages: list[str] = []

    def emit(_level: int, msg: str) -> None:
        msg_s = str(msg)
        messages.append(msg_s)
        if "preconditioner" in msg_s or "GMRES" in msg_s or "complete" in msg_s:
            print(msg_s, flush=True)

    phases: list[dict[str, Any]] = []
    t0 = time.perf_counter()
    phase_t0 = time.perf_counter()
    nml = read_sfincs_input(args.input)
    phases.append(_phase_record("read_input", phase_t0, time.perf_counter()))
    phase_t0 = time.perf_counter()
    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        restart=int(args.restart),
        solve_method=str(args.solve_method),
        emit=emit,
    )
    phases.append(_phase_record("solve", phase_t0, time.perf_counter()))
    elapsed_s = time.perf_counter() - t0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = _resource_maxrss_to_mb(float(usage.ru_maxrss))
    metadata = dict(result.metadata or {})
    observed_krylov = (
        _last_ksp_solver(messages)
        or metadata.get("krylov_method")
        or metadata.get("solve_method")
        or args.solve_method
    )
    rhs_norm = float(jnp.linalg.norm(result.rhs))
    target_residual_norm = max(0.0, float(args.tol) * rhs_norm)
    return _json_safe({
        "status": "ok",
        "elapsed_s": float(elapsed_s),
        "max_rss_mb": max_rss_mb,
        "residual_norm": float(result.residual_norm),
        "rhs_norm": rhs_norm,
        "target_residual_norm": target_residual_norm,
        "phase_metadata": phases,
        "tail_metadata": {
            "messages_tail_count": len(messages[-40:]),
            "messages_tail_limit": 40,
        },
        "solver_provenance": {
            "requested_solve_method": str(args.solve_method),
            "realized_solve_method": str(metadata.get("solve_method") or args.solve_method),
            "observed_krylov_method": str(observed_krylov),
            "normalized_expected_krylov_method": _normalize_solver_method(args.solve_method),
            "normalized_observed_krylov_method": _normalize_solver_method(observed_krylov),
            "metadata_solver_kind": metadata.get("solver_kind"),
            "metadata_krylov_method": metadata.get("krylov_method"),
            "fallback_from_krylov_method": metadata.get("fallback_from_krylov_method"),
            "safe_sparse_host_fallback_used": metadata.get("safe_sparse_host_fallback_used"),
        },
        "runtime_metadata": _runtime_metadata(str(getattr(args, "expected_backend", "auto"))),
        "metadata": metadata,
        "messages_tail": messages[-40:],
    })


def _gate(status: str, reason: str, **details: Any) -> dict[str, Any]:
    return _json_safe({"status": status, "reason": reason, **details})


def _phase_elapsed_max(row: dict[str, Any]) -> float | None:
    phase_values: list[float] = []
    for phase in row.get("phase_metadata", []):
        if isinstance(phase, dict):
            value = _optional_float(phase.get("elapsed_s"))
            if value is not None:
                phase_values.append(value)
    row_elapsed = _optional_float(row.get("elapsed_s"))
    if row_elapsed is not None:
        phase_values.append(row_elapsed)
    return max(phase_values) if phase_values else None


def _observed_backend(row: dict[str, Any]) -> str:
    runtime_metadata = row.get("runtime_metadata", {})
    if isinstance(runtime_metadata, dict):
        backend = runtime_metadata.get("jax_default_backend")
        if backend:
            return _normalize_backend(backend)
        platforms = runtime_metadata.get("jax_device_platforms")
        if isinstance(platforms, list) and platforms:
            return _normalize_backend(platforms[0])
    metadata = row.get("metadata", {})
    if isinstance(metadata, dict):
        for key in ("backend", "jax_backend", "platform", "jax_platform"):
            if metadata.get(key):
                return _normalize_backend(metadata.get(key))
    return ""


def _solver_path_observation(row: dict[str, Any]) -> tuple[str, str, list[str]]:
    provenance = row.get("solver_provenance", {})
    if not isinstance(provenance, dict):
        provenance = {}
    metadata = row.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    messages = row.get("messages_tail", [])
    messages_list = [str(item) for item in messages] if isinstance(messages, list) else []
    observed = (
        provenance.get("observed_krylov_method")
        or metadata.get("krylov_method")
        or _last_ksp_solver(messages_list)
        or provenance.get("realized_solve_method")
    )
    expected = (
        provenance.get("normalized_expected_krylov_method")
        or provenance.get("requested_solve_method")
        or provenance.get("realized_solve_method")
    )
    churn: list[str] = []
    fallback_from = provenance.get("fallback_from_krylov_method") or metadata.get("fallback_from_krylov_method")
    if fallback_from:
        churn.append(f"fallback_from_krylov_method={fallback_from}")
    safe_sparse = provenance.get("safe_sparse_host_fallback_used")
    if safe_sparse is None:
        safe_sparse = metadata.get("safe_sparse_host_fallback_used")
    if safe_sparse is True:
        churn.append("safe_sparse_host_fallback_used")
    return _normalize_solver_method(expected), _normalize_solver_method(observed), churn


def result_gates(args: argparse.Namespace, row: dict[str, Any], variant: str) -> dict[str, Any]:
    """Return policy gates for one real-solve benchmark row."""
    status = str(row.get("status", "unknown"))
    stall_s = _effective_stall_s(args)
    elapsed_max = _phase_elapsed_max(row)
    if status == "timeout":
        stall_gate = _gate("fail", "timeout", timeout_s=float(getattr(args, "timeout_s", 0.0)))
    elif status != "ok":
        stall_gate = _gate("fail", "non-ok-status", status=status)
    elif elapsed_max is None:
        stall_gate = _gate("fail", "missing-elapsed")
    elif elapsed_max > stall_s:
        stall_gate = _gate("fail", "stall-threshold-exceeded", elapsed_s=elapsed_max, stall_s=stall_s)
    else:
        stall_gate = _gate("pass", "within-stall-bound", elapsed_s=elapsed_max, stall_s=stall_s)

    residual = _optional_float(row.get("residual_norm"))
    max_residual = float(getattr(args, "max_residual_norm", DEFAULT_MAX_RESIDUAL_NORM))
    metadata = row.get("metadata", {})
    accepted_converged = metadata.get("accepted_converged") if isinstance(metadata, dict) else None
    if status != "ok":
        residual_gate = _gate("fail", "non-ok-status", status=status)
    elif residual is None:
        residual_gate = _gate("fail", "missing-or-nonfinite-residual")
    elif accepted_converged is False:
        residual_gate = _gate("fail", "solver-metadata-nonconverged", residual_norm=residual)
    elif max_residual > 0.0 and residual > max_residual:
        residual_gate = _gate(
            "fail",
            "residual-threshold-exceeded",
            residual_norm=residual,
            max_residual_norm=max_residual,
        )
    else:
        residual_gate = _gate(
            "pass",
            "residual-clean",
            residual_norm=residual,
            max_residual_norm=max_residual if max_residual > 0.0 else None,
        )

    max_rss_mb = _optional_float(row.get("max_rss_mb"))
    max_rss_limit = float(getattr(args, "max_rss_mb", 0.0))
    if status != "ok":
        memory_gate = _gate("fail", "non-ok-status", status=status)
    elif max_rss_mb is None:
        memory_gate = _gate("fail", "missing-or-nonfinite-rss")
    elif max_rss_limit > 0.0 and max_rss_mb > max_rss_limit:
        memory_gate = _gate("fail", "rss-threshold-exceeded", max_rss_mb=max_rss_mb, limit_mb=max_rss_limit)
    else:
        memory_gate = _gate(
            "pass",
            "rss-within-bound" if max_rss_limit > 0.0 else "rss-recorded-threshold-disabled",
            max_rss_mb=max_rss_mb,
            limit_mb=max_rss_limit if max_rss_limit > 0.0 else None,
        )

    expected_backend = str(getattr(args, "expected_backend", "auto")).strip().lower()
    observed_backend = _observed_backend(row)
    if status != "ok":
        backend_gate = _gate("fail", "non-ok-status", status=status)
    elif expected_backend == "auto":
        backend_gate = _gate("pass", "backend-not-constrained", observed_backend=observed_backend or None)
    elif not observed_backend:
        backend_gate = _gate("fail", "missing-backend-metadata", expected_backend=expected_backend)
    elif observed_backend != expected_backend:
        backend_gate = _gate(
            "fail",
            "backend-mismatch",
            expected_backend=expected_backend,
            observed_backend=observed_backend,
        )
    else:
        backend_gate = _gate("pass", "backend-matches", expected_backend=expected_backend, observed_backend=observed_backend)

    expected_solver, observed_solver, churn = _solver_path_observation(row)
    solver_churn_allowed = bool(getattr(args, "allow_solver_churn", False))
    if status != "ok":
        solver_gate = _gate("fail", "non-ok-status", status=status)
    elif not observed_solver:
        solver_gate = _gate("fail", "missing-solver-path-metadata", expected_solver=expected_solver or None)
    elif observed_solver != expected_solver:
        solver_gate = _gate(
            "pass" if solver_churn_allowed else "fail",
            "solver-path-mismatch-allowed" if solver_churn_allowed else "solver-path-mismatch",
            expected_solver=expected_solver,
            observed_solver=observed_solver,
        )
    elif churn:
        solver_gate = _gate(
            "pass" if solver_churn_allowed else "fail",
            "solver-path-churn-allowed" if solver_churn_allowed else "solver-path-churn",
            expected_solver=expected_solver,
            observed_solver=observed_solver,
            churn=churn,
        )
    else:
        solver_gate = _gate("pass", "solver-path-stable", expected_solver=expected_solver, observed_solver=observed_solver)

    if not bool(getattr(args, "require_default_promotion_gate", False)):
        default_promotion_gate = _gate("pass", "default-promotion-gate-not-requested")
    elif status != "ok":
        default_promotion_gate = _gate("fail", "non-ok-status", status=status)
    elif residual_gate["status"] != "pass":
        default_promotion_gate = _gate("fail", "residual-gate-failed", residual_gate=residual_gate)
    else:
        baseline_elapsed_s = _optional_float(getattr(args, "baseline_elapsed_s", None))
        baseline_rss_mb = _optional_float(getattr(args, "baseline_rss_mb", None))
        min_runtime_speedup = float(getattr(args, "min_runtime_speedup", DEFAULT_MIN_PROMOTION_SPEEDUP))
        min_memory_reduction = float(getattr(args, "min_memory_reduction", DEFAULT_MIN_PROMOTION_MEMORY_REDUCTION))
        runtime_speedup = (
            baseline_elapsed_s / elapsed_max
            if baseline_elapsed_s is not None and elapsed_max is not None and elapsed_max > 0.0
            else None
        )
        memory_reduction = (
            baseline_rss_mb / max_rss_mb
            if baseline_rss_mb is not None and max_rss_mb is not None and max_rss_mb > 0.0
            else None
        )
        runtime_win = runtime_speedup is not None and runtime_speedup >= min_runtime_speedup
        memory_win = memory_reduction is not None and memory_reduction >= min_memory_reduction
        if baseline_elapsed_s is None or baseline_rss_mb is None:
            default_promotion_gate = _gate(
                "fail",
                "missing-promotion-baseline",
                baseline_elapsed_s=baseline_elapsed_s,
                baseline_rss_mb=baseline_rss_mb,
            )
        elif elapsed_max is None or max_rss_mb is None:
            default_promotion_gate = _gate(
                "fail",
                "missing-candidate-runtime-or-rss",
                elapsed_s=elapsed_max,
                max_rss_mb=max_rss_mb,
            )
        elif elapsed_max > baseline_elapsed_s and max_rss_mb > baseline_rss_mb:
            default_promotion_gate = _gate(
                "fail",
                "runtime-and-memory-regression",
                elapsed_s=elapsed_max,
                baseline_elapsed_s=baseline_elapsed_s,
                max_rss_mb=max_rss_mb,
                baseline_rss_mb=baseline_rss_mb,
                runtime_speedup=runtime_speedup,
                memory_reduction=memory_reduction,
            )
        elif elapsed_max > baseline_elapsed_s:
            default_promotion_gate = _gate(
                "fail",
                "runtime-regression",
                elapsed_s=elapsed_max,
                baseline_elapsed_s=baseline_elapsed_s,
                runtime_speedup=runtime_speedup,
                memory_reduction=memory_reduction,
            )
        elif max_rss_mb > baseline_rss_mb:
            default_promotion_gate = _gate(
                "fail",
                "memory-regression",
                max_rss_mb=max_rss_mb,
                baseline_rss_mb=baseline_rss_mb,
                runtime_speedup=runtime_speedup,
                memory_reduction=memory_reduction,
            )
        elif not (runtime_win or memory_win):
            default_promotion_gate = _gate(
                "fail",
                "no-runtime-or-memory-win",
                runtime_speedup=runtime_speedup,
                memory_reduction=memory_reduction,
                min_runtime_speedup=min_runtime_speedup,
                min_memory_reduction=min_memory_reduction,
            )
        else:
            default_promotion_gate = _gate(
                "pass",
                "promotion-win-recorded",
                runtime_speedup=runtime_speedup,
                memory_reduction=memory_reduction,
                runtime_win=runtime_win,
                memory_win=memory_win,
                min_runtime_speedup=min_runtime_speedup,
                min_memory_reduction=min_memory_reduction,
            )

    return {
        "stall": stall_gate,
        "residual": residual_gate,
        "memory": memory_gate,
        "backend": backend_gate,
        "solver_path": solver_gate,
        "default_promotion": default_promotion_gate,
        "variant": _gate("pass", "variant-recorded", variant=str(variant)),
    }


def apply_result_gates(args: argparse.Namespace, row: dict[str, Any], variant: str) -> dict[str, Any]:
    gated = dict(row)
    gates = result_gates(args, gated, variant)
    failures = [f"{name}:{gate['reason']}" for name, gate in gates.items() if gate.get("status") != "pass"]
    gated["gates"] = gates
    gated["gate"] = "pass" if not failures else "fail"
    gated["gate_failures"] = failures
    return _json_safe(gated)


def _run_child(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """Run one variant in a subprocess and return a bounded result row."""
    env = os.environ.copy()
    env.update(
        _variant_env(
            variant,
            block=int(args.block),
            overlap=int(args.overlap),
            maxiter=int(args.maxiter),
            restart=int(args.restart),
        )
    )
    input_path = Path(args.input)
    overrides = _grid_overrides(args)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="sfincs-jax-pas-tz-") if overrides else None
    t0 = time.perf_counter()
    try:
        if tmp_ctx is not None:
            input_path = _write_child_input(input_path, Path(tmp_ctx.name), overrides)
        variant_solve_method = _variant_solve_method(str(variant), str(args.solve_method))
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--input",
            str(input_path),
            "--tol",
            str(args.tol),
            "--maxiter",
            str(args.maxiter),
            "--restart",
            str(args.restart),
            "--solve-method",
            variant_solve_method,
            "--expected-backend",
            str(getattr(args, "expected_backend", "auto")),
        ]
        try:
            completed = subprocess.run(
                cmd,
                env=env,
                text=True,
                capture_output=True,
                timeout=float(args.timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            row = {
                "variant": str(variant),
                "variant_provenance": _variant_provenance(str(variant), str(args.solve_method)),
                "status": "timeout",
                "elapsed_s": float(time.perf_counter() - t0),
                "timeout_s": float(args.timeout_s),
                "stdout_tail": _tail_text(exc.stdout),
                "stderr_tail": _tail_text(exc.stderr),
                "tail_metadata": {
                    "stdout_tail_chars": len(_tail_text(exc.stdout)),
                    "stderr_tail_chars": len(_tail_text(exc.stderr)),
                    "tail_limit_chars": 4000,
                },
            }
            return apply_result_gates(args, row, str(variant))
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    payload: dict[str, Any] | None = None
    for line in completed.stdout.splitlines()[::-1]:
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER) :])
            break
    if payload is None:
        payload = {
            "status": "error",
            "returncode": int(completed.returncode),
            "stdout_tail": _tail_text(completed.stdout),
            "stderr_tail": _tail_text(completed.stderr),
        }
    payload["variant"] = str(variant)
    payload["variant_provenance"] = _variant_provenance(str(variant), str(args.solve_method))
    payload["returncode"] = int(completed.returncode)
    payload.setdefault("elapsed_s", float(time.perf_counter() - t0))
    payload.setdefault(
        "tail_metadata",
        {
            "stdout_tail_chars": len(_tail_text(completed.stdout)),
            "stderr_tail_chars": len(_tail_text(completed.stderr)),
            "tail_limit_chars": 4000,
        },
    )
    if completed.returncode != 0 and payload.get("status") == "ok":
        payload["status"] = "error"
    return apply_result_gates(args, payload, str(variant))


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the benchmark plan payload."""
    input_path = Path(args.input)
    overrides = _grid_overrides(args)
    return {
        "input": _input_record(input_path),
        "input_overrides": overrides,
        "timeout_s": float(args.timeout_s),
        "tol": float(args.tol),
        "solve_method": str(args.solve_method),
        "variant_methods": [_variant_provenance(str(variant), str(args.solve_method)) for variant in args.variants],
        "maxiter": int(args.maxiter),
        "restart": int(args.restart),
        "block": int(args.block),
        "overlap": int(args.overlap),
        "variants": list(args.variants),
        "gates": {
            "max_default_runtime_s": MAX_DEFAULT_RUNTIME_S,
            "timeout_s": float(args.timeout_s),
            "stall_s": _effective_stall_s(args),
            "long_run_opt_in": bool(getattr(args, "allow_long_run", False)),
            "max_rss_mb": float(getattr(args, "max_rss_mb", 0.0)),
            "max_residual_norm": float(getattr(args, "max_residual_norm", DEFAULT_MAX_RESIDUAL_NORM)),
            "expected_backend": str(getattr(args, "expected_backend", "auto")),
            "allow_solver_churn": bool(getattr(args, "allow_solver_churn", False)),
            "default_promotion_required": bool(getattr(args, "require_default_promotion_gate", False)),
            "baseline_elapsed_s": _optional_float(getattr(args, "baseline_elapsed_s", None)),
            "baseline_rss_mb": _optional_float(getattr(args, "baseline_rss_mb", None)),
            "min_runtime_speedup": float(getattr(args, "min_runtime_speedup", DEFAULT_MIN_PROMOTION_SPEEDUP)),
            "min_memory_reduction": float(getattr(args, "min_memory_reduction", DEFAULT_MIN_PROMOTION_MEMORY_REDUCTION)),
            "promotion_policy": (
                "residual-clean candidates must not regress elapsed_s or max_rss_mb "
                "against baseline and must still show a material runtime or memory win"
            ),
        },
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize benchmark status and policy-gate outcomes."""
    by_status: dict[str, int] = {}
    by_gate: dict[str, int] = {"pass": 0, "fail": 0}
    failed_variants: list[str] = []
    promotion_eligible_variants: list[str] = []
    failure_reasons: dict[str, int] = {}
    for row in results:
        status = str(row.get("status", "unknown"))
        by_status[status] = by_status.get(status, 0) + 1
        gate = str(row.get("gate", "fail"))
        by_gate[gate] = by_gate.get(gate, 0) + 1
        failures = row.get("gate_failures", [])
        if gate != "pass":
            failed_variants.append(str(row.get("variant", "unknown")))
        if isinstance(failures, list):
            for failure in failures:
                key = str(failure)
                failure_reasons[key] = failure_reasons.get(key, 0) + 1
        gates = row.get("gates", {})
        if isinstance(gates, dict):
            promotion_gate = gates.get("default_promotion", {})
            if isinstance(promotion_gate, dict) and promotion_gate.get("status") == "pass":
                if promotion_gate.get("reason") == "promotion-win-recorded":
                    promotion_eligible_variants.append(str(row.get("variant", "unknown")))
    return {
        "result_count": len(results),
        "by_status": by_status,
        "by_gate": by_gate,
        "all_gates_passed": by_gate.get("fail", 0) == 0,
        "failed_variants": failed_variants,
        "promotion_eligible_variants": promotion_eligible_variants,
        "failure_reasons": failure_reasons,
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        _validate_runtime_bounds(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.child:
        payload = _child_payload(args)
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True, allow_nan=False))
        return 0

    payload: dict[str, Any] = {
        "schema_version": 2,
        "kind": "pas_tz_memory_fallback_benchmark",
        "plan": build_plan(args),
        "results": [],
        "summary": {},
    }
    if not args.dry_run:
        for variant in args.variants:
            row = _run_child(args, str(variant))
            payload["results"].append(row)
            print(
                f"{variant}: {row.get('status')} gate={row.get('gate')} "
                f"elapsed={float(row.get('elapsed_s', 0.0)):.2f}s",
                flush=True,
            )
    payload["summary"] = summarize_results(payload["results"])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
