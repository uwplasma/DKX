"""RHSMode=1 solver diagnostic assembly helpers."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Mapping, Sequence

import jax.numpy as jnp

from ...rhs1_ksp_diagnostics import emit_rhs1_ksp_history, emit_rhs1_ksp_iter_stats


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
