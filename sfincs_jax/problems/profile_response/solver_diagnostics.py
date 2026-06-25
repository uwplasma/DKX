"""RHSMode=1 solver diagnostic assembly helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
from typing import Any

import jax.numpy as jnp

from sfincs_jax.krylov_dispatch import ksp_iteration_solver_label
from sfincs_jax.solver import (
    GMRESSolveResult,
    bicgstab_solve_with_history_scipy,
    gmres_solve_with_history_scipy,
    lgmres_solve_with_history_scipy,
)
from sfincs_jax.v3_results import V3LinearSolveResult
from .active_dof import finalize_rhs1_linear_solution_cleanup
from .policies import (
    rhs1_scipy_rescue_abs_floor_after_xblock,
)
from .residual import l2_norm_float, residual_converged, residual_target


EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class RHS1SubspaceCorrectionDiagnostics:
    """Diagnostics for one residual/coarse subspace correction hook."""

    steps_requested: int
    direction_counts: Sequence[int] = ()
    direction_names: Sequence[str] = ()
    residual_before: float | None = None
    residual_after: float | None = None
    history: Sequence[float] = ()
    fsavg_lmax: int = 0
    angular_lmax: int = -1
    angular_residual: bool = False
    seed_initialized: bool | None = None
    setup_s: float | None = None
    include_qi_basis: bool | None = None


@dataclass(frozen=True)
class RHS1PostMinresDiagnostics:
    """Diagnostics for the scalar post-minres cleanup hook."""

    steps_requested: int
    alphas: Sequence[float] = ()
    history: Sequence[float] = ()
    residual_before: float | None = None
    residual_after: float | None = None


@dataclass(frozen=True)
class RHS1PreflightDiagnostics:
    """Diagnostics for the optional x-block seed preflight gate."""

    min_improvement: float
    required: bool
    residual_norm: float | None = None
    improvement: float | None = None
    passed: bool | None = None


@dataclass(frozen=True)
class RHS1CachedQICorrectionBasis:
    """Cached QI basis payload for post residual-equation corrections."""

    vectors: jnp.ndarray | None = None
    operator_on_basis: jnp.ndarray | None = None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class RHS1KSPDiagnosticsContext:
    """Static controls for optional RHSMode=1 KSP diagnostic replay."""

    emit: EmitFn | None
    fortran_stdout: bool
    history_max_size: int | None
    history_max_iter: int | None
    iter_stats_enabled: bool
    iter_stats_max_size: int | None


def _subspace_count(values: Sequence[int]) -> int:
    return int(sum(int(value) for value in values))


def emit_profile_response_ksp_history(
    *,
    context: RHS1KSPDiagnosticsContext,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    solve_method_val: str,
) -> list[float] | None:
    """Emit optional PETSc-like KSP residual history for RHSMode=1 solves."""

    return emit_rhs1_ksp_history(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=x0_vec,
        tol_val=tol_val,
        atol_val=atol_val,
        restart_val=restart_val,
        maxiter_val=maxiter_val,
        precond_side=precond_side,
        solver_kind=solver_kind,
        solve_method_val=solve_method_val,
        emit=context.emit,
        fortran_stdout=bool(context.fortran_stdout),
        max_size=context.history_max_size,
        max_history_iter=context.history_max_iter,
    )


def emit_newton_krylov_ksp_history(
    *,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    emit: EmitFn | None,
    fortran_stdout: bool,
    max_size: int | None,
    max_history_iter: int | None,
) -> list[float] | None:
    """Emit PETSc-like GMRES history for bounded Newton-Krylov diagnostics."""

    if emit is None or not fortran_stdout:
        return None
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(max_size)})")
        return None
    if maxiter_val is not None and max_history_iter is not None:
        est_iters = int(maxiter_val) * max(1, int(restart_val))
        if est_iters > int(max_history_iter):
            emit(
                1,
                "fortran-stdout: KSP history skipped "
                f"(estimated_iters={est_iters} > max={int(max_history_iter)})",
            )
            return None
    try:
        _x_hist, _rn, history = gmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
    except Exception as exc:  # noqa: BLE001
        emit(1, f"fortran-stdout: KSP history unavailable ({type(exc).__name__}: {exc})")
        return None
    for k_hist, rn in enumerate(history):
        emit(0, f"{k_hist:4d} KSP Residual norm {rn: .12e} ")
    if history:
        emit(0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
        emit(0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")
    return history


def emit_profile_response_ksp_iter_stats(
    *,
    context: RHS1KSPDiagnosticsContext,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    history: list[float] | None,
    solve_method_val: str,
) -> None:
    """Emit optional bounded KSP iteration-count diagnostics for RHSMode=1."""

    emit_rhs1_ksp_iter_stats(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=x0_vec,
        tol_val=tol_val,
        atol_val=atol_val,
        restart_val=restart_val,
        maxiter_val=maxiter_val,
        precond_side=precond_side,
        solver_kind=solver_kind,
        history=history,
        solve_method_val=solve_method_val,
        emit=context.emit,
        enabled=bool(context.iter_stats_enabled),
        max_size=context.iter_stats_max_size,
    )


def emit_profile_response_ksp_replay_diagnostics(
    *,
    context: RHS1KSPDiagnosticsContext,
    replay_state: Any,
    tol_val: float,
    atol_val: float,
    solve_method_val: str,
) -> list[float] | None:
    """Emit RHSMode=1 KSP replay history and iteration statistics.

    ``replay_state`` is duck-typed so the driver-owned solve state can stay in
    the handoff module without creating an import cycle.
    """

    matvec_fn = getattr(replay_state, "matvec_fn", None)
    b_vec = getattr(replay_state, "b_vec", None)
    if matvec_fn is None or b_vec is None:
        return None

    history = emit_profile_response_ksp_history(
        context=context,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=getattr(replay_state, "precond_fn", None),
        x0_vec=getattr(replay_state, "x0_vec", None),
        tol_val=tol_val,
        atol_val=atol_val,
        restart_val=int(getattr(replay_state, "restart")),
        maxiter_val=getattr(replay_state, "maxiter", None),
        precond_side=getattr(replay_state, "precond_side"),
        solver_kind=getattr(replay_state, "solver_kind"),
        solve_method_val=solve_method_val,
    )
    emit_profile_response_ksp_iter_stats(
        context=context,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=getattr(replay_state, "precond_fn", None),
        x0_vec=getattr(replay_state, "x0_vec", None),
        tol_val=float(tol_val),
        atol_val=float(atol_val),
        restart_val=int(getattr(replay_state, "restart")),
        maxiter_val=getattr(replay_state, "maxiter", None),
        precond_side=getattr(replay_state, "precond_side"),
        solver_kind=getattr(replay_state, "solver_kind"),
        history=history,
        solve_method_val=solve_method_val,
    )
    return history


def build_profile_response_linear_metadata(
    *,
    rhs_mode: int,
    result_residual_norm: float,
    rhs: jnp.ndarray,
    tol: float,
    atol: float,
    metadata_parts: Sequence[Mapping[str, object]],
    post_xblock_accept_floor: float = 0.0,
) -> dict[str, object]:
    """Merge final linear-solve metadata and apply post-xblock acceptance gates."""

    metadata: dict[str, object] = {}
    for part in metadata_parts:
        metadata.update(dict(part))

    floor = float(post_xblock_accept_floor)
    residual_norm = float(result_residual_norm)
    if int(rhs_mode) == 1 and floor > 0.0 and math.isfinite(residual_norm) and residual_norm <= floor:
        target = residual_target(
            atol=float(atol),
            tol=float(tol),
            rhs_norm=l2_norm_float(rhs),
        )
        metadata.update(
            {
                "accepted_converged": True,
                "acceptance_criterion": "post_xblock_abs_floor",
                "true_residual_converged": residual_converged(residual_norm, target),
                "accepted_residual_floor": floor,
            }
        )
    return metadata


def prepare_cached_qi_correction_basis(
    *,
    active: bool,
    include_qi_basis: bool,
    qi_device_state: object | None,
) -> RHS1CachedQICorrectionBasis:
    """Return cached QI basis arrays when a post correction can use them."""

    if not bool(active) or not bool(include_qi_basis) or qi_device_state is None:
        return RHS1CachedQICorrectionBasis()
    metadata = getattr(qi_device_state, "metadata", None)
    if int(getattr(metadata, "rank", 0)) <= 0:
        return RHS1CachedQICorrectionBasis()
    basis = getattr(qi_device_state, "basis")
    basis_metadata = getattr(basis, "metadata")
    return RHS1CachedQICorrectionBasis(
        vectors=jnp.asarray(basis.vectors, dtype=jnp.float64),
        operator_on_basis=jnp.asarray(
            getattr(qi_device_state, "operator_on_basis"),
            dtype=jnp.float64,
        ),
        labels=tuple(str(label) for label in basis_metadata.accepted_labels),
    )


def build_rhs1_xblock_correction_metadata(
    *,
    probe_coarse: RHS1SubspaceCorrectionDiagnostics,
    preflight: RHS1PreflightDiagnostics,
    post_minres: RHS1PostMinresDiagnostics,
    post_coarse: RHS1SubspaceCorrectionDiagnostics,
    post_residual_equation: RHS1SubspaceCorrectionDiagnostics,
) -> dict[str, object]:
    """Build solver-trace metadata for x-block correction hooks.

    Keeping this field assembly out of ``v3_driver.py`` makes output/trace
    compatibility independently testable. The returned keys intentionally match
    the historical solver metadata names.
    """

    metadata: dict[str, object] = {
        "xblock_probe_coarse_steps_requested": int(probe_coarse.steps_requested),
        "xblock_probe_coarse_steps_accepted": int(len(probe_coarse.direction_counts)),
        "xblock_probe_coarse_direction_count": _subspace_count(probe_coarse.direction_counts),
        "xblock_probe_coarse_residual_before": probe_coarse.residual_before,
        "xblock_probe_coarse_residual_after": probe_coarse.residual_after,
        "xblock_probe_coarse_seed_initialized": bool(probe_coarse.seed_initialized),
        "xblock_probe_coarse_s": float(probe_coarse.setup_s or 0.0),
        "xblock_probe_coarse_history": tuple(probe_coarse.history),
        "xblock_probe_coarse_direction_counts": tuple(probe_coarse.direction_counts),
        "xblock_probe_coarse_direction_names": tuple(probe_coarse.direction_names),
        "xblock_probe_coarse_fsavg_lmax": int(probe_coarse.fsavg_lmax),
        "xblock_probe_coarse_angular_lmax": int(probe_coarse.angular_lmax),
        "xblock_probe_coarse_angular_residual": bool(probe_coarse.angular_residual),
        "xblock_preflight_min_improvement": float(preflight.min_improvement),
        "xblock_preflight_required": bool(preflight.required),
        "xblock_preflight_residual_norm": preflight.residual_norm,
        "xblock_preflight_improvement": preflight.improvement,
        "xblock_preflight_passed": preflight.passed,
        "xblock_post_minres_steps_requested": int(post_minres.steps_requested),
        "xblock_post_minres_steps_accepted": int(len(post_minres.alphas)),
        "xblock_post_minres_residual_before": post_minres.residual_before,
        "xblock_post_minres_residual_after": post_minres.residual_after,
        "xblock_post_minres_alphas": tuple(post_minres.alphas),
        "xblock_post_minres_history": tuple(post_minres.history),
        "xblock_post_coarse_steps_requested": int(post_coarse.steps_requested),
        "xblock_post_coarse_steps_accepted": int(len(post_coarse.direction_counts)),
        "xblock_post_coarse_direction_count": _subspace_count(post_coarse.direction_counts),
        "xblock_post_coarse_residual_before": post_coarse.residual_before,
        "xblock_post_coarse_residual_after": post_coarse.residual_after,
        "xblock_post_coarse_history": tuple(post_coarse.history),
        "xblock_post_coarse_direction_counts": tuple(post_coarse.direction_counts),
        "xblock_post_coarse_direction_names": tuple(post_coarse.direction_names),
        "xblock_post_coarse_fsavg_lmax": int(post_coarse.fsavg_lmax),
        "xblock_post_coarse_angular_lmax": int(post_coarse.angular_lmax),
        "xblock_post_coarse_angular_residual": bool(post_coarse.angular_residual),
        "xblock_post_residual_equation_steps_requested": int(
            post_residual_equation.steps_requested
        ),
        "xblock_post_residual_equation_steps_accepted": int(
            len(post_residual_equation.direction_counts)
        ),
        "xblock_post_residual_equation_direction_count": _subspace_count(
            post_residual_equation.direction_counts
        ),
        "xblock_post_residual_equation_residual_before": (
            post_residual_equation.residual_before
        ),
        "xblock_post_residual_equation_residual_after": (
            post_residual_equation.residual_after
        ),
        "xblock_post_residual_equation_history": tuple(post_residual_equation.history),
        "xblock_post_residual_equation_direction_counts": tuple(
            post_residual_equation.direction_counts
        ),
        "xblock_post_residual_equation_direction_names": tuple(
            post_residual_equation.direction_names
        ),
        "xblock_post_residual_equation_fsavg_lmax": int(
            post_residual_equation.fsavg_lmax
        ),
        "xblock_post_residual_equation_angular_lmax": int(
            post_residual_equation.angular_lmax
        ),
        "xblock_post_residual_equation_angular_residual": bool(
            post_residual_equation.angular_residual
        ),
        "xblock_post_residual_equation_include_qi_basis": bool(
            post_residual_equation.include_qi_basis
        ),
    }
    return metadata


def build_rhs1_xblock_correction_metadata_from_driver_state(
    state: Mapping[str, object],
) -> dict[str, object]:
    """Build x-block correction metadata from the driver solve state.

    This keeps the long correction-diagnostics object assembly next to the
    stable metadata schema instead of in the main solve routine.
    """

    return build_rhs1_xblock_correction_metadata(
        probe_coarse=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=int(state["probe_coarse_steps_requested"]),
            direction_counts=state["probe_coarse_direction_counts"],
            direction_names=state["probe_coarse_direction_names"],
            residual_before=state["probe_coarse_residual_before"],
            residual_after=state["probe_coarse_residual_after"],
            history=state["probe_coarse_history"],
            fsavg_lmax=int(state["probe_coarse_fsavg_lmax"]),
            angular_lmax=int(state["probe_coarse_angular_lmax"]),
            angular_residual=bool(state["probe_coarse_include_angular_residual"]),
            seed_initialized=bool(state["probe_coarse_seed_initialized"]),
            setup_s=float(state["probe_coarse_s"]),
        ),
        preflight=RHS1PreflightDiagnostics(
            min_improvement=float(state["preflight_min_improvement"]),
            required=bool(state["preflight_required"]),
            residual_norm=state["preflight_residual_norm"],
            improvement=state["preflight_improvement"],
            passed=state["preflight_passed"],
        ),
        post_minres=RHS1PostMinresDiagnostics(
            steps_requested=int(state["post_minres_steps_requested"]),
            alphas=state["post_minres_alphas"],
            history=state["post_minres_history"],
            residual_before=state["post_minres_residual_before"],
            residual_after=state["post_minres_residual_after"],
        ),
        post_coarse=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=int(state["post_coarse_steps_requested"]),
            direction_counts=state["post_coarse_direction_counts"],
            direction_names=state["post_coarse_direction_names"],
            residual_before=state["post_coarse_residual_before"],
            residual_after=state["post_coarse_residual_after"],
            history=state["post_coarse_history"],
            fsavg_lmax=int(state["post_coarse_fsavg_lmax"]),
            angular_lmax=int(state["post_coarse_angular_lmax"]),
            angular_residual=bool(state["post_coarse_include_angular_residual"]),
        ),
        post_residual_equation=RHS1SubspaceCorrectionDiagnostics(
            steps_requested=int(state["post_residual_equation_steps_requested"]),
            direction_counts=state["post_residual_equation_direction_counts"],
            direction_names=state["post_residual_equation_direction_names"],
            residual_before=state["post_residual_equation_residual_before"],
            residual_after=state["post_residual_equation_residual_after"],
            history=state["post_residual_equation_history"],
            fsavg_lmax=int(state["post_residual_equation_fsavg_lmax"]),
            angular_lmax=int(state["post_residual_equation_angular_lmax"]),
            angular_residual=bool(
                state["post_residual_equation_include_angular_residual"]
            ),
            include_qi_basis=bool(
                state["post_residual_equation_include_qi_basis"]
            ),
        ),
    )

# Consolidated KSP replay diagnostics

_FALSE_TOKENS = {"0", "false", "no", "off"}
_TRUE_TOKENS = {"1", "true", "yes", "on"}
_UNLIMITED_TOKENS = {"none", "inf", "infinite", "unlimited"}


@dataclass(frozen=True)
class RHS1KSPHistoryLimits:
    """Size and iteration caps for optional PETSc-like KSP replay."""

    max_size: int | None
    max_iter: int


@dataclass(frozen=True)
class RHS1KSPIterStatsControls:
    """Controls for optional bounded KSP iteration-count replay."""

    enabled: bool
    max_size: int | None


@dataclass(frozen=True)
class RHS1KSPDiagnosticsControls:
    """Environment-normalized controls shared by RHSMode=1 diagnostics."""

    fortran_stdout: bool
    history_max_size: int | None
    history_max_iter: int
    iter_stats_enabled: bool
    iter_stats_max_size: int | None


def rhs1_fortran_stdout_from_env(*, emit: EmitFn | None) -> bool:
    """Resolve Fortran-style solver stdout from the environment and emit state."""

    env = os.environ.get("SFINCS_JAX_FORTRAN_STDOUT", "").strip().lower()
    if env in _FALSE_TOKENS:
        return False
    if env in _TRUE_TOKENS:
        return True
    return emit is not None


def rhs1_ksp_history_limits_from_env() -> RHS1KSPHistoryLimits:
    """Return bounded replay limits for optional PETSc-like KSP history."""

    max_size_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "").strip().lower()
    if max_size_env in _UNLIMITED_TOKENS:
        max_size = None
    else:
        try:
            max_size = int(max_size_env) if max_size_env else 800
        except ValueError:
            max_size = 800

    max_iter_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "").strip()
    try:
        max_iter = int(max_iter_env) if max_iter_env else 2000
    except ValueError:
        max_iter = 2000
    return RHS1KSPHistoryLimits(max_size=max_size, max_iter=max_iter)


def rhs1_ksp_iter_stats_controls_from_env() -> RHS1KSPIterStatsControls:
    """Return opt-in iteration replay controls for solver diagnostics."""

    enabled_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS", "").strip().lower()
    max_size_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "").strip()
    try:
        max_size = int(max_size_env) if max_size_env else None
    except ValueError:
        max_size = None
    return RHS1KSPIterStatsControls(enabled=enabled_env in _TRUE_TOKENS, max_size=max_size)


def rhs1_ksp_diagnostics_controls_from_env(*, emit: EmitFn | None) -> RHS1KSPDiagnosticsControls:
    """Parse all shared RHSMode=1 diagnostic replay controls."""

    history = rhs1_ksp_history_limits_from_env()
    iter_stats = rhs1_ksp_iter_stats_controls_from_env()
    return RHS1KSPDiagnosticsControls(
        fortran_stdout=rhs1_fortran_stdout_from_env(emit=emit),
        history_max_size=history.max_size,
        history_max_iter=history.max_iter,
        iter_stats_enabled=iter_stats.enabled,
        iter_stats_max_size=iter_stats.max_size,
    )


def emit_rhs1_ksp_history(
    *,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    solve_method_val: str,
    emit: EmitFn | None,
    fortran_stdout: bool,
    max_size: int | None,
    max_history_iter: int | None,
) -> list[float] | None:
    """Emit PETSc-like KSP residual history for bounded RHSMode=1 diagnostics."""
    if emit is None or not fortran_stdout:
        return None
    solver_label = ksp_iteration_solver_label(solver_kind=solver_kind, solve_method=solve_method_val)
    if solver_label not in {"gmres", "lgmres"}:
        return None
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(max_size)})")
        return None
    if maxiter_val is not None and max_history_iter is not None:
        est_iters = int(maxiter_val)
        if solver_label == "gmres":
            est_iters *= max(1, int(restart_val))
        if est_iters > int(max_history_iter):
            emit(
                1,
                "fortran-stdout: KSP history skipped "
                f"(estimated_iters={est_iters} > max={int(max_history_iter)})",
            )
            return None
    try:
        history = _solve_history(
            solver_label=solver_label,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            precond_side=precond_side,
        )
    except Exception as exc:  # noqa: BLE001
        emit(1, f"fortran-stdout: KSP history unavailable ({type(exc).__name__}: {exc})")
        return None
    for k, rn in enumerate(history):
        emit(0, f"{k:4d} KSP Residual norm {rn: .12e} ")
    if history:
        emit(0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
        emit(0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")
    return history


def emit_rhs1_ksp_iter_stats(
    *,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    history: list[float] | None,
    solve_method_val: str,
    emit: EmitFn | None,
    enabled: bool,
    max_size: int | None,
) -> None:
    """Emit bounded RHSMode=1 KSP iteration-count diagnostics."""
    if emit is None or not enabled:
        return
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"ksp_iterations skipped (size={size} > max={int(max_size)})")
        return
    solver_kind_l = str(solver_kind).strip().lower()
    solver_label = ksp_iteration_solver_label(solver_kind=solver_kind_l, solve_method=solve_method_val)
    iter_stats_max_iter = _read_iter_stats_max_iter()
    if maxiter_val is not None and iter_stats_max_iter is not None:
        est_iters = int(maxiter_val)
        if solver_label == "gmres":
            est_iters *= max(1, int(restart_val))
        if est_iters > int(iter_stats_max_iter):
            emit(
                1,
                "ksp_iterations skipped "
                f"(estimated_iters={est_iters} > max={int(iter_stats_max_iter)})",
            )
            return
    try:
        if solver_label in {"gmres", "lgmres"}:
            if history is None:
                history = _solve_history(
                    solver_label=solver_label,
                    matvec_fn=matvec_fn,
                    b_vec=b_vec,
                    precond_fn=precond_fn,
                    x0_vec=x0_vec,
                    tol_val=tol_val,
                    atol_val=atol_val,
                    restart_val=restart_val,
                    maxiter_val=maxiter_val,
                    precond_side=precond_side,
                )
            iters = len(history or [])
        elif solver_kind_l == "bicgstab":
            _x_hist, _rn, history = bicgstab_solve_with_history_scipy(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                maxiter=maxiter_val,
                precondition_side=precond_side,
            )
            iters = len(history or [])
        else:
            return
    except Exception as exc:  # noqa: BLE001
        emit(1, f"ksp_iterations unavailable ({type(exc).__name__}: {exc})")
        return
    emit(0, f"ksp_iterations={iters} solver={solver_label}")


def _solve_history(
    *,
    solver_label: str,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
) -> list[float]:
    if solver_label == "lgmres":
        _x_hist, _rn, history = lgmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
        return history
    _x_hist, _rn, history = gmres_solve_with_history_scipy(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        precondition_side=precond_side,
    )
    return history


def _read_iter_stats_max_iter() -> int:
    env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER", "").strip()
    try:
        return int(env) if env else 2000
    except ValueError:
        return 2000

# Consolidated final RHSMode=1 linear-solve handoff

@dataclass(frozen=True)
class ProfileResponseLinearFinalizationContext:
    """Inputs needed to finalize a v3-compatible profile-response linear solve."""

    op: Any
    rhs: jnp.ndarray
    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    ksp_replay: Any
    ksp_diagnostics_context: RHS1KSPDiagnosticsContext
    tol: float
    atol: float
    solve_method: str
    active_size: int
    used_large_cpu_xblock_shortcut: bool
    used_explicit_fp_xblock_seed: bool
    use_implicit: bool
    backend: str
    metadata_parts: Sequence[Mapping[str, object]]
    emit: EmitFn | None = None
    elapsed_s: Callable[[], float] | None = None


def profile_response_post_xblock_accept_floor(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> float:
    """Return the final metadata acceptance floor for post-xblock RHSMode=1 solves."""

    if int(op.rhs_mode) != 1:
        return 0.0
    return rhs1_scipy_rescue_abs_floor_after_xblock(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=str(backend),
    )


def finalize_profile_response_linear_solve(
    context: ProfileResponseLinearFinalizationContext,
) -> V3LinearSolveResult:
    """Apply final cleanup, diagnostics, metadata, and result wrapping."""

    result = finalize_rhs1_linear_solution_cleanup(
        op=context.op,
        result=context.result,
        rhs=context.rhs,
        residual_vec=context.residual_vec,
    )
    emit_profile_response_ksp_replay_diagnostics(
        context=context.ksp_diagnostics_context,
        replay_state=context.ksp_replay,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        solve_method_val=str(context.solve_method),
    )
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: "
            f"residual_norm={float(result.residual_norm):.6e}",
        )
        if context.elapsed_s is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: "
                f"elapsed_s={float(context.elapsed_s()):.3f}",
            )

    post_xblock_accept_floor = profile_response_post_xblock_accept_floor(
        op=context.op,
        active_size=int(context.active_size),
        used_large_cpu_xblock_shortcut=bool(context.used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(context.used_explicit_fp_xblock_seed),
        use_implicit=bool(context.use_implicit),
        backend=str(context.backend),
    )
    metadata_out = build_profile_response_linear_metadata(
        rhs_mode=int(context.op.rhs_mode),
        result_residual_norm=float(result.residual_norm),
        rhs=context.rhs,
        tol=float(context.tol),
        atol=float(context.atol),
        metadata_parts=context.metadata_parts,
        post_xblock_accept_floor=float(post_xblock_accept_floor),
    )
    return V3LinearSolveResult(
        op=context.op,
        rhs=context.rhs,
        gmres=result,
        metadata=metadata_out or None,
    )
