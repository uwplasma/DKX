"""Compare compact Fortran-v3 and SFINCS-JAX solver profile artifacts."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def _first_campaign_case(payload: dict[str, Any], *, case_index: int = 0) -> dict[str, Any]:
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        return payload
    row = cases[int(case_index)]
    if not isinstance(row, dict):
        raise TypeError(f"Campaign case {case_index} is not a JSON object")
    return row


def _krylov_progress_from_text(text: str) -> list[dict[str, Any]]:
    progress: list[dict[str, Any]] = []
    for line in str(text).splitlines():
        if "solve_v3_full_system_linear_gmres:" not in line:
            continue
        count_match = re.search(r"\b(?P<kind>iters|matvecs)=(?P<count>\d+)\b", line)
        if count_match is None:
            continue
        item: dict[str, Any] = {
            "line": line,
            "kind": count_match.group("kind"),
            "count": int(count_match.group("count")),
        }
        residual_match = re.search(r"\b(?:ksp_residual|residual)=(?P<residual>[-+0-9.eE]+)\b", line)
        if residual_match is not None:
            item["residual"] = float(residual_match.group("residual"))
        elapsed_match = re.search(r"\belapsed_s=(?P<elapsed>[-+0-9.eE]+)\b", line)
        if elapsed_match is not None:
            item["elapsed_s"] = float(elapsed_match.group("elapsed"))
        progress.append(item)
    return progress


def _jax_summary(payload: dict[str, Any], *, case_index: int = 0) -> dict[str, Any]:
    row = _first_campaign_case(payload, case_index=case_index)
    solve = row.get("solve") if isinstance(row.get("solve"), dict) else {}
    trace = solve.get("solver_trace") if isinstance(solve.get("solver_trace"), dict) else {}
    progress = solve.get("progress") if isinstance(solve.get("progress"), dict) else {}
    result = solve.get("result") if isinstance(solve.get("result"), dict) else {}
    if not trace and "residual_norm" in payload:
        trace = payload
    tail_progress = _krylov_progress_from_text(
        f"{solve.get('stdout_tail', '')}\n{solve.get('stderr_tail', '')}"
    )

    elapsed_s = trace.get("elapsed_s")
    if elapsed_s is None:
        elapsed_s = result.get("elapsed_s")
    residual_norm = trace.get("residual_norm")
    residual_target = trace.get("residual_target")
    residual_ratio = trace.get("residual_ratio")
    if residual_ratio is None and residual_norm is not None and residual_target is not None:
        try:
            residual_ratio = float(residual_norm) / max(float(residual_target), 1.0e-300)
        except (TypeError, ValueError):
            residual_ratio = None

    max_krylov_count = progress.get("max_krylov_count")
    last_krylov_progress = progress.get("last_krylov_progress")
    last_krylov_residual = progress.get("last_krylov_residual")
    min_krylov_residual = progress.get("min_krylov_residual")
    if tail_progress and max_krylov_count is None:
        max_krylov_count = max(int(item["count"]) for item in tail_progress)
        last_krylov_progress = tail_progress[-1]
        residuals = [float(item["residual"]) for item in tail_progress if "residual" in item]
        last_krylov_residual = residuals[-1] if residuals else None
        min_krylov_residual = min(residuals, default=None)

    return {
        "status": solve.get("status") or payload.get("status"),
        "returncode": solve.get("returncode"),
        "timeout_s": solve.get("timeout_s"),
        "elapsed_s": elapsed_s,
        "converged": trace.get("converged"),
        "solve_method": trace.get("solve_method") or payload.get("solve_method"),
        "selected_path": trace.get("selected_path"),
        "active_size": trace.get("active_size"),
        "total_size": trace.get("total_size"),
        "residual_norm": residual_norm,
        "residual_target": residual_target,
        "residual_ratio": residual_ratio,
        "peak_rss_mb": trace.get("peak_rss_mb"),
        "device_peak_mb": trace.get("device_peak_mb"),
        "profile_entry_count": trace.get("profile_entry_count"),
        "max_krylov_count": max_krylov_count,
        "last_krylov_progress": last_krylov_progress,
        "last_krylov_residual": last_krylov_residual,
        "min_krylov_residual": min_krylov_residual,
    }


def compare_solver_profiles(
    *,
    fortran_profile: dict[str, Any],
    jax_profile: dict[str, Any],
    case_index: int = 0,
) -> dict[str, Any]:
    """Return a compact side-by-side comparison for docs and run gates."""
    f_ksp = fortran_profile.get("ksp") if isinstance(fortran_profile.get("ksp"), dict) else {}
    f_mumps = fortran_profile.get("mumps") if isinstance(fortran_profile.get("mumps"), dict) else {}
    f_timings = fortran_profile.get("timings_s") if isinstance(fortran_profile.get("timings_s"), dict) else {}
    jax = _jax_summary(jax_profile, case_index=case_index)

    f_iters = f_ksp.get("iteration_count")
    jax_count = jax.get("max_krylov_count")
    try:
        krylov_count_ratio = float(jax_count) / max(float(f_iters), 1.0) if jax_count is not None else None
    except (TypeError, ValueError):
        krylov_count_ratio = None

    try:
        residual_ratio_vs_fortran_final = (
            float(jax["residual_norm"]) / max(float(f_ksp["final_residual"]), 1.0e-300)
            if jax.get("residual_norm") is not None and f_ksp.get("final_residual") is not None
            else None
        )
    except (TypeError, ValueError):
        residual_ratio_vs_fortran_final = None

    fortran_shape = fortran_profile.get("matrix_shape")
    fortran_active_size = fortran_shape[0] if isinstance(fortran_shape, list) and fortran_shape else None
    jax_active_size = jax.get("active_size")
    same_active_size = (
        bool(jax_active_size == fortran_active_size)
        if jax_active_size is not None and fortran_active_size is not None
        else None
    )

    return {
        "schema_version": 1,
        "fortran": {
            "n_mpi_processes": fortran_profile.get("n_mpi_processes"),
            "solver_package": fortran_profile.get("solver_package"),
            "matrix_shape": fortran_profile.get("matrix_shape"),
            "matrix_nnz": fortran_profile.get("matrix_nnz"),
            "preconditioner_nnz": fortran_profile.get("preconditioner_nnz"),
            "factor_entries": f_mumps.get("factor_entries"),
            "factor_memory_peak_mb": f_mumps.get("factor_memory_peak_mb"),
            "factor_memory_total_mb": f_mumps.get("factor_memory_total_mb"),
            "ksp_iteration_count": f_iters,
            "ksp_initial_residual": f_ksp.get("initial_residual"),
            "ksp_final_residual": f_ksp.get("final_residual"),
            "assemble_preconditioner_s": f_timings.get("assemble_preconditioner"),
            "assemble_jacobian_s": f_timings.get("assemble_jacobian"),
            "mumps_analysis_s": f_timings.get("mumps_analysis_driver"),
            "mumps_factorization_s": f_timings.get("mumps_factorization_driver"),
        },
        "jax": jax,
        "comparison": {
            "jax_krylov_count_per_fortran_ksp_iteration": krylov_count_ratio,
            "jax_residual_norm_per_fortran_final_ksp_residual": residual_ratio_vs_fortran_final,
            "same_active_size": same_active_size,
        },
    }


def compare_solver_profile_files(
    *,
    fortran_profile_path: str | Path,
    jax_profile_path: str | Path,
    case_index: int = 0,
) -> dict[str, Any]:
    """Read two JSON artifacts and return :func:`compare_solver_profiles`."""
    return compare_solver_profiles(
        fortran_profile=_load_json(fortran_profile_path),
        jax_profile=_load_json(jax_profile_path),
        case_index=case_index,
    )
