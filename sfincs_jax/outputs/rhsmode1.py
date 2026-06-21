"""RHSMode=1 output gates and solver-trace schema helpers.

The public writer in :mod:`sfincs_jax.io` builds the physical output fields.
This module owns the smaller policy boundary around production-output safety:
large RHSMode=1 runs must either satisfy the requested residual target or write
an explicit sidecar trace before the main diagnostic file is refused.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..memory_model import estimate_linear_solve_memory
from ..solver_trace import SolverTrace, write_solver_trace_json


def _rhs1_active_size_for_trace(op: Any) -> int | None:
    """Return the reduced RHSMode=1 active size used by matrix-free solves."""

    try:
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int64)
        active_f = (
            int(op.n_species)
            * int(np.sum(nxi_for_x))
            * int(op.n_theta)
            * int(op.n_zeta)
        )
        phi1_size = int(getattr(op, "phi1_size", 0))
        extra_size = int(getattr(op, "extra_size", 0))
        if (
            int(getattr(op, "rhs_mode", 1)) == 1
            and not bool(getattr(op, "include_phi1", False))
            and int(getattr(op, "constraint_scheme", 0)) == 2
            and getattr(op.fblock, "pas", None) is not None
            and phi1_size == 0
        ):
            min_env = os.environ.get("SFINCS_JAX_PAS_PROJECT_MIN", "").strip()
            try:
                project_min = int(min_env) if min_env else 2000
            except ValueError:
                project_min = 2000
            if int(getattr(op, "total_size", active_f)) >= max(0, project_min):
                return active_f
        return active_f + phi1_size + extra_size
    except Exception:
        return None


def _env_flag(name: str, *, default: bool = False) -> bool:
    """Parse a permissive boolean environment variable."""

    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _rhsmode1_result_residual_and_target(
    result: Any,
    *,
    solver_tol: float,
) -> tuple[float | None, float | None]:
    """Extract the true residual norm and target used to decide output safety."""

    residual_norm = None
    if hasattr(result, "residual_norm"):
        try:
            residual_norm = float(np.asarray(getattr(result, "residual_norm")))
        except Exception:
            residual_norm = None

    residual_target = None
    rhs_vec = getattr(result, "rhs", None)
    if rhs_vec is not None:
        try:
            residual_target = max(
                0.0,
                float(solver_tol) * float(np.linalg.norm(np.asarray(rhs_vec))),
            )
        except Exception:
            residual_target = None
    return residual_norm, residual_target


def _should_fail_nonconverged_rhsmode1_output(
    *,
    active_total_size: int,
    residual_norm: float | None,
    residual_target: float | None,
    accepted_converged: bool | None = None,
) -> bool:
    """Return True when a large RHSMode=1 output should be blocked."""

    if _env_flag("SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT", default=False):
        return False
    if accepted_converged is True:
        return False
    min_env = os.environ.get("SFINCS_JAX_NONCONVERGED_FAIL_MIN_SIZE", "").strip()
    try:
        min_size = int(min_env) if min_env else 10_000
    except ValueError:
        min_size = 10_000
    if int(active_total_size) < max(0, min_size):
        return False
    if residual_norm is None or residual_target is None:
        return False
    return (not np.isfinite(float(residual_norm))) or float(residual_norm) > float(
        residual_target
    )


def _raise_for_nonconverged_rhsmode1_output(
    *,
    active_total_size: int,
    residual_norm: float | None,
    residual_target: float | None,
    solve_method: str,
    accepted_converged: bool | None = None,
    acceptance_criterion: str | None = None,
) -> None:
    """Raise a clear production-output error for nonconverged RHSMode=1 solves."""

    if not _should_fail_nonconverged_rhsmode1_output(
        active_total_size=active_total_size,
        residual_norm=residual_norm,
        residual_target=residual_target,
        accepted_converged=accepted_converged,
    ):
        return
    raise RuntimeError(
        "Refusing to write nonconverged RHSMode=1 diagnostics for a production-sized solve: "
        f"active_size={int(active_total_size)} residual_norm={float(residual_norm):.6e} "
        f"target={float(residual_target):.6e} solve_method={solve_method!s}. "
        f"accepted_converged={accepted_converged!s} criterion={acceptance_criterion!s}. "
        "Use a converged solver path such as --solve-method sparse_pc_gmres, lower the resolution, "
        "or set SFINCS_JAX_ALLOW_NONCONVERGED_OUTPUT=1 only for debugging partial states."
    )


def _solver_metadata_dict(result: Any) -> dict[str, Any]:
    """Return Python-only solver metadata attached by explicit host solve paths."""

    metadata = getattr(result, "metadata", None)
    if isinstance(metadata, dict):
        return dict(metadata)
    return {}


def _metadata_int(metadata: dict[str, Any], key: str) -> int | None:
    """Return a finite integer metadata value when present."""

    if key not in metadata:
        return None
    try:
        value = int(metadata[key])
    except (TypeError, ValueError, OverflowError):
        return None
    return value if value >= 0 else None


def _profile_memory_summary(
    profiler: Any | None,
) -> tuple[float | None, float | None, float | None]:
    """Return active RSS, device peak, and process peak memory from profiler entries."""

    if profiler is None or not getattr(profiler, "entries", None):
        return None, None, None
    active_vals: list[float] = []
    device_vals: list[float] = []
    peak_vals: list[float] = []
    for entry in getattr(profiler, "entries"):
        try:
            if entry.get("dpeak_rss_mb") is not None:
                active_vals.append(float(entry["dpeak_rss_mb"]))
            elif entry.get("drss_mb") is not None:
                active_vals.append(float(entry["drss_mb"]))
        except (TypeError, ValueError):
            pass
        try:
            if entry.get("device_mb") is not None:
                device_vals.append(float(entry["device_mb"]))
        except (TypeError, ValueError):
            pass
        for key in ("rss_mb", "peak_rss_mb"):
            try:
                if entry.get(key) is not None:
                    peak_vals.append(float(entry[key]))
            except (TypeError, ValueError):
                pass
    active_rss_mb = max(active_vals) if active_vals else None
    device_peak_mb = max(device_vals) if device_vals else None
    peak_rss_mb = max(peak_vals) if peak_vals else None
    return active_rss_mb, device_peak_mb, peak_rss_mb


def _solver_trace_memory_estimate(
    *,
    total_size: int | None,
    active_size: int | None,
    solver_metadata: dict[str, Any],
    device_count: int | None,
) -> dict[str, int | None] | None:
    """Build conservative memory estimates for solver trace fields."""

    unknowns = total_size if total_size is not None else active_size
    if unknowns is None or int(unknowns) <= 0:
        return None
    restart = (
        _metadata_int(solver_metadata, "gmres_restart")
        or _metadata_int(solver_metadata, "restart")
        or _metadata_int(solver_metadata, "inner_m")
        or 80
    )
    csr_nnz = (
        _metadata_int(solver_metadata, "sparse_pattern_nnz")
        or _metadata_int(solver_metadata, "csr_nnz")
        or None
    )
    estimate = estimate_linear_solve_memory(
        unknowns=int(unknowns),
        gmres_restart=int(restart),
        csr_nnz=csr_nnz,
        preconditioner_nbytes=_metadata_int(
            solver_metadata,
            "sparse_pc_factor_nbytes_estimate",
        ),
        device_count=1 if device_count is None else max(1, int(device_count)),
    )
    return {
        "dense_operator_nbytes": int(estimate.dense_operator_nbytes),
        "csr_operator_nbytes": (
            None
            if estimate.csr_operator_nbytes is None
            else int(estimate.csr_operator_nbytes)
        ),
        "gmres_basis_nbytes": int(estimate.gmres_basis_nbytes),
        "preconditioner_nbytes": estimate.preconditioner_nbytes,
        "dense_total_nbytes": int(estimate.dense_total_nbytes),
        "csr_total_nbytes": (
            None if estimate.csr_total_nbytes is None else int(estimate.csr_total_nbytes)
        ),
        "dense_per_device_nbytes": int(estimate.dense_per_device_nbytes),
        "csr_per_device_nbytes": (
            None
            if estimate.csr_per_device_nbytes is None
            else int(estimate.csr_per_device_nbytes)
        ),
    }


def _write_nonconverged_rhsmode1_solver_trace_json(
    *,
    solver_trace_path: Path,
    input_namelist: Path,
    output_path: Path,
    output_format: str,
    rhs_mode: int,
    geom_scheme_hint: int | None,
    compute_solution: bool,
    compute_transport_matrix: bool,
    differentiable: bool | None,
    result: Any,
    op_fallback: Any,
    solver_tol: float,
    solve_method: str,
    residual_norm: float | None,
    residual_target: float | None,
    active_total_size: int,
    run_t0: float,
    profiler: Any | None = None,
) -> None:
    """Write a JSON trace before refusing nonconverged RHSMode=1 diagnostics."""

    try:
        import jax  # noqa: PLC0415

        backend = str(jax.default_backend())
        device_count = len(jax.devices())
    except Exception:
        backend = "unknown"
        device_count = None

    trace_op = getattr(result, "op", None)
    if trace_op is None:
        trace_op = getattr(result, "op0", None)
    if trace_op is None:
        trace_op = op_fallback

    trace_total_size = None
    trace_active_size = None
    trace_collision_operator = None
    if trace_op is not None:
        try:
            trace_total_size = int(getattr(trace_op, "total_size"))
        except Exception:
            trace_total_size = None
        trace_active_size = _rhs1_active_size_for_trace(trace_op)
        if trace_active_size is None:
            try:
                trace_active_size = int(getattr(trace_op, "active_size"))
            except Exception:
                trace_active_size = trace_total_size
        try:
            trace_collision_operator = str(getattr(trace_op, "collision_operator"))
        except Exception:
            trace_collision_operator = None
    if trace_active_size is None:
        trace_active_size = int(active_total_size)

    solver_metadata = _solver_metadata_dict(result)
    if residual_target is None:
        rhs_vec = getattr(result, "rhs", None)
        if rhs_vec is not None:
            try:
                residual_target = max(
                    0.0,
                    float(solver_tol) * float(np.linalg.norm(np.asarray(rhs_vec))),
                )
            except Exception:
                residual_target = None

    trace_metadata: dict[str, object] = {
        "input_namelist": str(input_namelist.resolve()),
        "output_path": str(output_path.resolve()),
        "output_format": str(output_format),
        "compute_solution": bool(compute_solution),
        "compute_transport_matrix": bool(compute_transport_matrix),
        "differentiable": None if differentiable is None else bool(differentiable),
        "output_refused": True,
        "failure_reason": "nonconverged_rhsmode1_output",
        "solver_metadata": solver_metadata,
    }
    if "accepted_converged" in solver_metadata:
        trace_metadata["accepted_converged"] = bool(
            solver_metadata["accepted_converged"]
        )
    if "acceptance_criterion" in solver_metadata:
        trace_metadata["acceptance_criterion"] = str(
            solver_metadata["acceptance_criterion"]
        )
    if residual_norm is not None and residual_target is not None:
        trace_metadata["converged"] = bool(float(residual_norm) <= float(residual_target))
    if profiler is not None and getattr(profiler, "entries", None):
        trace_metadata["profile_entries"] = list(getattr(profiler, "entries"))

    try:
        from ..profiling import _peak_rss_mb, _rss_mb  # noqa: PLC0415

        peak_rss_mb = _peak_rss_mb()
        if peak_rss_mb is None:
            peak_rss_mb = _rss_mb()
    except Exception:
        peak_rss_mb = None
    active_rss_mb = None
    device_peak_mb = None
    if profiler is not None and getattr(profiler, "entries", None):
        active_rss_mb, device_peak_mb, profiler_peak_rss_mb = _profile_memory_summary(
            profiler
        )
        if profiler_peak_rss_mb is not None:
            peak_rss_mb = profiler_peak_rss_mb

    memory_estimate = _solver_trace_memory_estimate(
        total_size=trace_total_size,
        active_size=trace_active_size,
        solver_metadata=solver_metadata,
        device_count=device_count,
    )
    if memory_estimate is not None:
        trace_metadata["memory_estimate"] = memory_estimate

    trace = SolverTrace(
        backend=backend,
        rhs_mode=int(rhs_mode),
        selected_path="rhsmode1_solution" if bool(compute_solution) else "geometry_only",
        solve_method=str(solve_method),
        preconditioner=(
            None
            if "preconditioner_kind" not in solver_metadata
            else str(solver_metadata["preconditioner_kind"])
        ),
        geometry_scheme=int(geom_scheme_hint) if geom_scheme_hint is not None else None,
        collision_operator=trace_collision_operator,
        total_size=trace_total_size,
        active_size=trace_active_size,
        device_count=device_count,
        residual_norm=residual_norm,
        residual_target=residual_target,
        converged=False if residual_norm is not None and residual_target is not None else None,
        elapsed_s=float(time.perf_counter() - run_t0),
        setup_s=(
            float(solver_metadata["setup_s"]) if "setup_s" in solver_metadata else None
        ),
        solve_s=(
            float(solver_metadata["solve_s"]) if "solve_s" in solver_metadata else None
        ),
        peak_rss_mb=peak_rss_mb,
        active_rss_mb=active_rss_mb,
        device_peak_mb=device_peak_mb,
        estimated_dense_nbytes=(
            None
            if memory_estimate is None
            else int(memory_estimate["dense_operator_nbytes"])
        ),
        estimated_csr_nbytes=(
            None
            if memory_estimate is None or memory_estimate["csr_operator_nbytes"] is None
            else int(memory_estimate["csr_operator_nbytes"])
        ),
        estimated_gmres_basis_nbytes=(
            None if memory_estimate is None else int(memory_estimate["gmres_basis_nbytes"])
        ),
        matvec_count=_metadata_int(solver_metadata, "matvecs"),
        metadata=trace_metadata,
    )
    write_solver_trace_json(solver_trace_path, trace)


__all__ = (
    "_metadata_int",
    "_profile_memory_summary",
    "_raise_for_nonconverged_rhsmode1_output",
    "_rhs1_active_size_for_trace",
    "_rhsmode1_result_residual_and_target",
    "_should_fail_nonconverged_rhsmode1_output",
    "_solver_metadata_dict",
    "_solver_trace_memory_estimate",
    "_write_nonconverged_rhsmode1_solver_trace_json",
)
