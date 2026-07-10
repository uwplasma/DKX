"""Solver progress, state, trace, and profile-comparison diagnostics."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import re
from typing import Any

import jax.numpy as jnp
import numpy as np


# User-facing progress messages.
EmitFn = Callable[[int, str], None]
PROGRESS_SIZE_MIN_ENV = "SFINCS_JAX_PROGRESS_SIZE_MIN"


def format_duration(seconds: float) -> str:
    """Format elapsed time for user-facing progress messages."""

    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h{minutes:02d}m"
    days, hours = divmod(hours, 24)
    return f"{days}d{hours:02d}h"


def runtime_scale_hint(*, rhs_mode_hint: int, total_size_hint: int, n_rhs_hint: int | None = None) -> str:
    """Return a coarse runtime class for CLI progress hints.

    The thresholds are intentionally conservative and human-readable. They are
    not used for solver decisions, so changing this helper cannot affect parity.
    """

    size = int(total_size_hint)
    if int(rhs_mode_hint) in {2, 3}:
        n_rhs_use = max(1, int(n_rhs_hint or 1))
        work = size * n_rhs_use
        if work < 40_000:
            return "usually seconds to a few minutes"
        if work < 250_000:
            return "often minutes"
        return "often many minutes or longer"
    if size < 8_000:
        return "usually seconds"
    if size < 50_000:
        return "often tens of seconds to a few minutes"
    if size < 200_000:
        return "often minutes"
    return "often many minutes or longer"


def rhs1_progress_size_min(*, environ: Mapping[str, str] | None = None, default: int = 20_000) -> int:
    """Read the RHSMode=1 progress-note size threshold from the environment."""

    env = os.environ if environ is None else environ
    raw = str(env.get(PROGRESS_SIZE_MIN_ENV, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def rhs1_large_progress_enabled(
    *, rhs_mode: int, total_size: int, environ: Mapping[str, str] | None = None
) -> bool:
    """Whether to emit extra progress notes for a large RHSMode=1 solve."""

    threshold = max(1, rhs1_progress_size_min(environ=environ))
    return int(rhs_mode) == 1 and int(total_size) >= threshold


@dataclass
class RHS1ProgressNotes:
    """One-shot progress-note emitter for large RHSMode=1 solves."""

    emit: EmitFn | None
    enabled: bool
    emitted: set[str] = field(default_factory=set)

    def preconditioner_build(self, kind: str | None) -> None:
        if self.emit is None or not self.enabled or "precond" in self.emitted:
            return
        self.emit(
            0,
            " solve_v3_full_system_linear_gmres: building RHSMode=1 preconditioner "
            f"({kind}); this stage can take a while for large systems.",
        )
        self.emitted.add("precond")

    def krylov_start(self) -> None:
        if self.emit is None or not self.enabled or "solve" in self.emitted:
            return
        self.emit(0, " solve_v3_full_system_linear_gmres: starting Krylov iterations.")
        self.emitted.add("solve")


def transport_progress_message(*, completed: int, total: int, avg_rhs_s: float, elapsed_s: float) -> str:
    """Build the standard transport whichRHS progress line."""
    completed_i = int(completed)
    total_i = int(total)
    remaining_rhs = max(0, total_i - completed_i)
    return (
        f"solve_v3_transport_matrix_linear_gmres: progress {completed_i}/{total_i} "
        f"avg_rhs={format_duration(avg_rhs_s)} elapsed={format_duration(elapsed_s)} "
        f"est_remaining={format_duration(float(avg_rhs_s) * remaining_rhs)}"
    )


# Krylov state signatures and warm-start persistence.
OPERATOR_SHAPE_SIGNATURE_FIELDS = (
    "rhs_mode",
    "total_size",
    "n_species",
    "n_x",
    "n_xi",
    "n_theta",
    "n_zeta",
    "constraint_scheme",
    "include_phi1",
    "include_phi1_in_kinetic",
    "quasineutrality_option",
)


def operator_shape_signature(op) -> tuple[int, ...]:
    """Return the semantic fixed-shape signature for reusable solve state.

    This signature deliberately excludes electric-field values, profile
    amplitudes, geometry values, and matrix entries.  It is a compatibility key
    for warm starts and symbolic setup reuse, not a key for reusing numerical
    matrices or factors.
    """

    return (
        int(op.rhs_mode),
        int(op.total_size),
        int(op.n_species),
        int(op.n_x),
        int(op.n_xi),
        int(op.n_theta),
        int(op.n_zeta),
        int(op.constraint_scheme),
        int(bool(op.include_phi1)),
        int(bool(op.include_phi1_in_kinetic)),
        int(op.quasineutrality_option),
    )


def operator_shape_signature_dict(op) -> dict[str, int]:
    """Return the shape signature as a JSON-friendly dictionary."""

    return dict(zip(OPERATOR_SHAPE_SIGNATURE_FIELDS, operator_shape_signature(op), strict=True))


def _op_signature(op) -> np.ndarray:
    """Backward-compatible ndarray view of :func:`operator_shape_signature`."""

    return np.asarray(
        operator_shape_signature(op),
        dtype=np.int64,
    )


def read_krylov_state_signature(path: str | Path) -> tuple[int, ...] | None:
    """Read only the fixed-shape signature from a saved Krylov state file."""

    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
        sig = np.asarray(data["signature"], dtype=np.int64).reshape((-1,))
    except Exception:
        return None
    if sig.size != len(OPERATOR_SHAPE_SIGNATURE_FIELDS):
        return None
    return tuple(int(v) for v in sig)


def save_krylov_state(
    *,
    path: str | Path,
    op,
    x_full: jnp.ndarray | None = None,
    x_by_rhs: dict[int, jnp.ndarray] | None = None,
    x_history: list[jnp.ndarray] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "signature": _op_signature(op),
        "signature_fields": np.asarray(OPERATOR_SHAPE_SIGNATURE_FIELDS, dtype="U32"),
    }
    if x_full is not None:
        payload["x_full"] = np.asarray(x_full, dtype=np.float64)
    if x_by_rhs is not None:
        which_rhs = np.asarray(sorted(x_by_rhs.keys()), dtype=np.int64)
        x_stack = np.stack([np.asarray(x_by_rhs[int(k)], dtype=np.float64) for k in which_rhs], axis=0)
        payload["which_rhs"] = which_rhs
        payload["x_by_rhs"] = x_stack
    if x_history is not None:
        if isinstance(x_history, (list, tuple)) and x_history:
            x_hist_stack = np.stack([np.asarray(v, dtype=np.float64) for v in x_history], axis=0)
            payload["x_history"] = x_hist_stack
    np.savez(path, **payload)


def load_krylov_state(
    *,
    path: str | Path,
    op,
) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        data = np.load(path, allow_pickle=False)
    except Exception:
        return None
    try:
        sig = np.asarray(data["signature"], dtype=np.int64)
    except Exception:
        return None
    if sig.shape != _op_signature(op).shape:
        return None
    if not np.array_equal(sig, _op_signature(op)):
        return None
    out: dict[str, Any] = {}
    if "x_full" in data:
        out["x_full"] = np.asarray(data["x_full"], dtype=np.float64)
    if "x_by_rhs" in data and "which_rhs" in data:
        which_rhs = np.asarray(data["which_rhs"], dtype=np.int64)
        x_stack = np.asarray(data["x_by_rhs"], dtype=np.float64)
        if x_stack.ndim == 2 and which_rhs.ndim == 1 and x_stack.shape[0] == which_rhs.shape[0]:
            out["x_by_rhs"] = {int(k): x_stack[i, :] for i, k in enumerate(which_rhs)}
    if "x_history" in data:
        x_hist = np.asarray(data["x_history"], dtype=np.float64)
        if x_hist.ndim == 2:
            out["x_history"] = [x_hist[i, :] for i in range(x_hist.shape[0])]
    if not out:
        return None
    return out


# Stable solver-trace records for logs, output files, and benchmarks.
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SolverTraceCandidate:
    """Recorded decision for one candidate solver/preconditioner path."""

    name: str
    accepted: bool
    reasons: tuple[str, ...] = ()
    residual_ratio: float | None = None
    runtime_ratio: float | None = None
    memory_ratio: float | None = None
    memory_metric: str | None = None
    active_rss_mb: float | None = None
    device_peak_mb: float | None = None
    compiled_temp_mb: float | None = None
    candidate_setup_s: float | None = None
    candidate_solve_s: float | None = None


@dataclass(frozen=True)
class SolverTrace:
    """Portable summary of one `sfincs_jax` solve-path decision."""

    backend: str
    rhs_mode: int
    selected_path: str
    schema_version: int = SCHEMA_VERSION
    solve_method: str | None = None
    preconditioner: str | None = None
    geometry_scheme: int | None = None
    collision_operator: str | None = None
    total_size: int | None = None
    active_size: int | None = None
    device_count: int | None = None
    cold_jit: bool | None = None
    residual_norm: float | None = None
    residual_target: float | None = None
    converged: bool | None = None
    elapsed_s: float | None = None
    setup_s: float | None = None
    solve_s: float | None = None
    peak_rss_mb: float | None = None
    active_rss_mb: float | None = None
    device_peak_mb: float | None = None
    compiled_temp_mb: float | None = None
    estimated_dense_nbytes: int | None = None
    estimated_csr_nbytes: int | None = None
    estimated_gmres_basis_nbytes: int | None = None
    matvec_count: int | None = None
    candidate_decisions: tuple[SolverTraceCandidate, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        record = asdict(self)
        record["candidate_decisions"] = [asdict(item) for item in self.candidate_decisions]
        return record

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SolverTrace":
        """Build a trace from a dictionary and validate the schema version."""
        schema_version = int(data.get("schema_version", SCHEMA_VERSION))
        if schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported solver trace schema_version={schema_version}")
        candidates = tuple(
            SolverTraceCandidate(
                name=str(item["name"]),
                accepted=bool(item["accepted"]),
                reasons=tuple(str(reason) for reason in item.get("reasons", ())),
                residual_ratio=item.get("residual_ratio"),
                runtime_ratio=item.get("runtime_ratio"),
                memory_ratio=item.get("memory_ratio"),
                memory_metric=item.get("memory_metric"),
                active_rss_mb=item.get("active_rss_mb"),
                device_peak_mb=item.get("device_peak_mb"),
                compiled_temp_mb=item.get("compiled_temp_mb"),
                candidate_setup_s=item.get("candidate_setup_s"),
                candidate_solve_s=item.get("candidate_solve_s"),
            )
            for item in data.get("candidate_decisions", ())
        )
        return cls(
            schema_version=schema_version,
            backend=str(data["backend"]),
            rhs_mode=int(data["rhs_mode"]),
            selected_path=str(data["selected_path"]),
            solve_method=data.get("solve_method"),
            preconditioner=data.get("preconditioner"),
            geometry_scheme=data.get("geometry_scheme"),
            collision_operator=data.get("collision_operator"),
            total_size=data.get("total_size"),
            active_size=data.get("active_size"),
            device_count=data.get("device_count"),
            cold_jit=data.get("cold_jit"),
            residual_norm=data.get("residual_norm"),
            residual_target=data.get("residual_target"),
            converged=data.get("converged"),
            elapsed_s=data.get("elapsed_s"),
            setup_s=data.get("setup_s"),
            solve_s=data.get("solve_s"),
            peak_rss_mb=data.get("peak_rss_mb"),
            active_rss_mb=data.get("active_rss_mb"),
            device_peak_mb=data.get("device_peak_mb"),
            compiled_temp_mb=data.get("compiled_temp_mb"),
            estimated_dense_nbytes=data.get("estimated_dense_nbytes"),
            estimated_csr_nbytes=data.get("estimated_csr_nbytes"),
            estimated_gmres_basis_nbytes=data.get("estimated_gmres_basis_nbytes"),
            matvec_count=data.get("matvec_count"),
            candidate_decisions=candidates,
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Return a stable JSON representation."""
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, payload: str | bytes) -> "SolverTrace":
        """Build a trace from a JSON payload."""
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")
        return cls.from_dict(json.loads(payload))


def write_solver_trace_json(path: str | Path, trace: SolverTrace) -> None:
    """Write a solver trace JSON sidecar."""
    Path(path).write_text(trace.to_json() + "\n", encoding="utf-8")


def read_solver_trace_json(path: str | Path) -> SolverTrace:
    """Read a solver trace JSON sidecar."""
    return SolverTrace.from_json(Path(path).read_text(encoding="utf-8"))


def write_solver_trace_h5(group: Any, trace: SolverTrace) -> None:
    """Write a solver trace into an HDF5 file/group as a JSON attribute."""
    trace_group = group.require_group("solver_trace")
    trace_group.attrs["schema_version"] = int(SCHEMA_VERSION)
    trace_group.attrs["json"] = trace.to_json()


def read_solver_trace_h5(group: Any) -> SolverTrace:
    """Read a solver trace from an HDF5 file/group written by this module."""
    trace_group = group["solver_trace"]
    payload = trace_group.attrs["json"]
    return SolverTrace.from_json(payload)


# Compact Fortran-v3 and SFINCS-JAX solver profile comparisons.
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


__all__ = [
    "EmitFn",
    "OPERATOR_SHAPE_SIGNATURE_FIELDS",
    "PROGRESS_SIZE_MIN_ENV",
    "RHS1ProgressNotes",
    "SCHEMA_VERSION",
    "SolverTrace",
    "SolverTraceCandidate",
    "compare_solver_profile_files",
    "compare_solver_profiles",
    "format_duration",
    "load_krylov_state",
    "operator_shape_signature",
    "operator_shape_signature_dict",
    "read_krylov_state_signature",
    "read_solver_trace_h5",
    "read_solver_trace_json",
    "rhs1_large_progress_enabled",
    "rhs1_progress_size_min",
    "runtime_scale_hint",
    "save_krylov_state",
    "transport_progress_message",
    "write_solver_trace_h5",
    "write_solver_trace_json",
]
