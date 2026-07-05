"""RHSMode=1 solver diagnostic assembly helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import os
import time
from typing import Any

from jax import tree_util as jtu
import jax.numpy as jnp

from sfincs_jax.solvers.krylov_dispatch import ksp_iteration_solver_label
from sfincs_jax.solvers.preconditioner_pas_policy import rhs1_pas_schur_rescue_controls_from_env
from sfincs_jax.solvers.path_policy import (
    SolverAcceptanceCriteria,
    SolverCandidateMetrics,
    solver_candidate_gate,
)
from sfincs_jax.solver import (
    GMRESSolveResult,
    bicgstab_solve_with_history_scipy,
    gmres_solve_with_history_scipy,
    lgmres_solve_with_history_scipy,
)
from .profile_policies import (
    rhs1_scipy_rescue_abs_floor_after_xblock,
)
from .profile_setup import finalize_rhs1_linear_solution_cleanup
from .profile_residual import l2_norm_float, residual_converged, residual_target


EmitFn = Callable[[int, str], None]

# RHSMode=1 candidate replay and acceptance helpers.
#
# These helpers used to live in ``profile_response.handoff``. They now live next
# to the KSP replay/finalization diagnostics so accepted candidates, replay
# state, and final iteration history have one owner.

@dataclass(frozen=True)
class RHS1KSPHandoffState:
    """Krylov replay state to emit iteration history for an accepted solve."""

    matvec_fn: Any
    b_vec: Any
    precond_fn: Any
    x0_vec: Any
    restart: int
    maxiter: int | None
    precond_side: str
    solver_kind: str


@dataclass
class RHS1KSPReplayState:
    """Mutable Krylov replay state used by final RHSMode=1 diagnostics.

    The driver may accept several rescue candidates before the final solution is
    returned. This small state object records the matvec, right-hand side,
    preconditioner, seed, and Krylov controls associated with the currently
    accepted candidate so Fortran-style iteration diagnostics replay the same
    linear problem that produced the returned residual.
    """

    matvec_fn: Any = None
    b_vec: Any = None
    precond_fn: Any = None
    x0_vec: Any = None
    restart: int = 80
    maxiter: int | None = None
    precond_side: str = "none"
    solver_kind: str = "gmres"


@dataclass(frozen=True)
class RHS1SkipPrimaryKrylovSeedContext:
    """Inputs for a skip-primary seed result and replay update."""

    matvec_fn: Any
    b_vec: Any
    precond_fn: Any
    x0_vec: Any
    precond_side: str
    solver_kind: str
    zero_like: Any
    norm: Any
    inf_residual: Any
    result_factory: Any


def rhs1_apply_handoff_to_replay_state(
    replay_state: RHS1KSPReplayState,
    handoff_state: RHS1KSPHandoffState | None,
) -> bool:
    """Update ``replay_state`` from an accepted candidate handoff.

    Returns ``True`` when a handoff was applied. A ``None`` handoff leaves the
    replay state unchanged and returns ``False``; this mirrors existing driver
    behavior where rejected candidates must not perturb the final diagnostics.
    """
    if handoff_state is None:
        return False
    replay_state.matvec_fn = handoff_state.matvec_fn
    replay_state.b_vec = handoff_state.b_vec
    replay_state.precond_fn = handoff_state.precond_fn
    replay_state.x0_vec = handoff_state.x0_vec
    replay_state.restart = int(handoff_state.restart)
    replay_state.maxiter = handoff_state.maxiter
    replay_state.precond_side = str(handoff_state.precond_side)
    replay_state.solver_kind = str(handoff_state.solver_kind)
    return True


def rhs1_record_ksp_replay_problem(
    replay_state: RHS1KSPReplayState,
    *,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    x0_vec: Any,
    precond_side: str,
    solver_kind: str,
    restart: int | None = None,
    maxiter: int | None = None,
) -> None:
    """Record the current linear problem for final KSP diagnostic replay.

    ``restart`` and ``maxiter`` are optional because the driver often keeps the
    original Krylov bounds while updating only the accepted operator/RHS/seed.
    """

    replay_state.matvec_fn = matvec_fn
    replay_state.b_vec = b_vec
    replay_state.precond_fn = precond_fn
    replay_state.x0_vec = x0_vec
    replay_state.precond_side = str(precond_side)
    replay_state.solver_kind = str(solver_kind)
    if restart is not None:
        replay_state.restart = int(restart)
    if maxiter is not None:
        replay_state.maxiter = maxiter


def rhs1_residual_improves(
    *,
    current_residual: float,
    candidate_residual: float,
) -> bool:
    """Return whether a candidate residual is a strict finite improvement.

    Nonfinite candidates are never accepted. A finite candidate may rescue a
    nonfinite incumbent, but equal finite residuals are rejected so repeated
    fallback attempts cannot churn the accepted solve state without making
    progress.
    """
    current = float(current_residual)
    candidate = float(candidate_residual)
    return math.isfinite(candidate) and (
        not math.isfinite(current) or candidate < current
    )


def rhs1_accept_candidate(
    *,
    current_result: Any,
    candidate_result: Any,
    current_residual_vec: Any = None,
    candidate_residual_vec: Any = None,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    x0_vec: Any,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solver_kind: str,
    candidate_metrics: SolverCandidateMetrics | None = None,
    baseline_metrics: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> tuple[Any, Any, RHS1KSPHandoffState | None, bool]:
    """Accept a candidate result and return the updated driver handoff state.

    The return shape intentionally matches the existing ``v3_driver.py`` call
    sites: accepted result, accepted residual vector, optional KSP replay state,
    and a boolean flag. Measured solver-candidate metrics are consulted only
    after the raw residual comparison passes, preserving the driver contract
    that a non-improving solve cannot become accepted because it has favorable
    runtime or memory metadata.
    """
    current_residual = float(current_result.residual_norm)
    candidate_residual = float(candidate_result.residual_norm)
    residual_improved = rhs1_residual_improves(
        current_residual=current_residual,
        candidate_residual=candidate_residual,
    )
    if residual_improved and candidate_metrics is not None:
        gate = solver_candidate_gate(
            candidate_metrics,
            baseline=baseline_metrics,
            criteria=criteria,
        )
        if not gate.accepted:
            residual_improved = False
    if residual_improved:
        accepted_residual_vec = (
            candidate_residual_vec
            if candidate_residual_vec is not None
            else current_residual_vec
        )
        return (
            candidate_result,
            accepted_residual_vec,
            RHS1KSPHandoffState(
                matvec_fn=matvec_fn,
                b_vec=b_vec,
                precond_fn=precond_fn,
                x0_vec=x0_vec,
                restart=int(restart),
                maxiter=maxiter,
                precond_side=str(precond_side),
                solver_kind=str(solver_kind),
            ),
            True,
        )
    return current_result, current_residual_vec, None, False


def rhs1_accept_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    **candidate_kwargs: Any,
) -> tuple[Any, Any, bool]:
    """Accept a candidate and update KSP replay state if it is retained.

    This is the driver-facing form for fallback/rescue branches: accepted
    candidates must update the final KSP replay diagnostics in the same step as
    the residual/result handoff, while rejected candidates leave the replay
    state untouched.
    """

    result, residual_vec, handoff_state, accepted = rhs1_accept_candidate(
        **candidate_kwargs
    )
    rhs1_apply_handoff_to_replay_state(replay_state, handoff_state)
    return result, residual_vec, accepted


def rhs1_solver_candidate_metrics(
    *,
    name: str,
    result: Any,
    target_value: float,
    solve_s: float | None = None,
    setup_s: float | None = None,
    peak_rss_mb: float | None = None,
) -> SolverCandidateMetrics:
    """Build measured solver-policy metrics from a real RHSMode=1 attempt."""
    try:
        residual_norm = float(result.residual_norm)
    except Exception:
        residual_norm = None
    finite = residual_norm is not None and math.isfinite(residual_norm)
    return SolverCandidateMetrics(
        name=str(name),
        residual_norm=residual_norm,
        target=float(target_value),
        setup_s=setup_s,
        solve_s=solve_s,
        peak_rss_mb=peak_rss_mb,
        finite=finite,
    )


def rhs1_accept_measured_candidate(
    *,
    current_result: Any,
    candidate_result: Any,
    current_residual_vec: Any = None,
    candidate_residual_vec: Any = None,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    x0_vec: Any,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solver_kind: str,
    candidate_name: str,
    baseline_name: str,
    target_value: float,
    solve_s: float | None = None,
    setup_s: float | None = None,
    peak_rss_mb: float | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> tuple[Any, Any, RHS1KSPHandoffState | None, bool]:
    """Accept a measured candidate while constructing standard gate metrics."""
    return rhs1_accept_candidate(
        current_result=current_result,
        candidate_result=candidate_result,
        current_residual_vec=current_residual_vec,
        candidate_residual_vec=candidate_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=x0_vec,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
        candidate_metrics=rhs1_solver_candidate_metrics(
            name=candidate_name,
            result=candidate_result,
            target_value=target_value,
            solve_s=solve_s,
            setup_s=setup_s,
            peak_rss_mb=peak_rss_mb,
        ),
        baseline_metrics=rhs1_solver_candidate_metrics(
            name=baseline_name,
            result=current_result,
            target_value=target_value,
            peak_rss_mb=peak_rss_mb,
        ),
        criteria=criteria,
    )


def rhs1_accept_measured_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    **candidate_kwargs: Any,
) -> tuple[Any, Any, bool]:
    """Accept a measured candidate and update replay diagnostics atomically."""

    result, residual_vec, handoff_state, accepted = rhs1_accept_measured_candidate(
        **candidate_kwargs
    )
    rhs1_apply_handoff_to_replay_state(replay_state, handoff_state)
    return result, residual_vec, accepted


def rhs1_accept_sparse_retry_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    candidate_result: Any,
    current_residual_vec: Any,
    candidate_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solver_kind: str,
    candidate_family: str,
    scope: str,
    target_value: float,
    solve_s: float | None = None,
    peak_rss_mb: float | None = None,
) -> tuple[Any, Any, bool]:
    """Accept an already-run sparse retry candidate and update replay state."""

    scope_name = str(scope)
    return rhs1_accept_measured_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        candidate_result=candidate_result,
        current_residual_vec=current_residual_vec,
        candidate_residual_vec=candidate_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=candidate_result.x,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
        candidate_name=f"{candidate_family}_{scope_name}",
        baseline_name=f"current_{scope_name}",
        target_value=target_value,
        solve_s=solve_s,
        peak_rss_mb=peak_rss_mb,
    )


def rhs1_run_fast_post_xblock_polish(
    *,
    current_result: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solve_linear: Any,
    emit: Any = None,
) -> tuple[Any, bool]:
    """Run the bounded post-xblock polish and retain it only if it improves."""

    current_residual = float(current_result.residual_norm)
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: fast post-xblock polish "
            f"(restart={int(restart)} maxiter={maxiter} residual={current_residual:.3e})",
        )
    candidate = solve_linear(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=current_result.x,
        tol_val=float(tol),
        atol_val=float(atol),
        restart_val=int(restart),
        maxiter_val=maxiter,
        solve_method_val="incremental",
        precond_side=str(precond_side),
    )
    if rhs1_residual_improves(
        current_residual=current_residual,
        candidate_residual=float(candidate.residual_norm),
    ):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: fast post-xblock polish improved residual "
                f"{current_residual:.3e} -> {float(candidate.residual_norm):.3e}",
            )
        return candidate, True
    return current_result, False


def rhs1_run_measured_linear_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    candidate_name: str,
    baseline_name: str,
    target_value: float,
    peak_rss_mb: float | None = None,
    returns_residual_vec: bool = False,
    result_ready: Any = None,
) -> tuple[Any, Any, bool, float]:
    """Run and measured-gate a linear retry candidate.

    This helper covers stage2/strong-style retry branches where the driver owns
    policy and progress messages but the candidate acceptance/replay contract is
    identical. Solvers may return either a bare result or ``(result,
    residual_vec)``.
    """

    started = time.perf_counter()
    candidate_output = solve_linear(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=current_result.x,
        tol_val=float(tol),
        atol_val=float(atol),
        restart_val=int(restart),
        maxiter_val=maxiter,
        solve_method_val=str(solve_method),
        precond_side=str(precond_side),
    )
    if returns_residual_vec:
        candidate_result, candidate_residual_vec = candidate_output
    else:
        candidate_result = candidate_output
        candidate_residual_vec = current_residual_vec
    if result_ready is not None:
        candidate_result = result_ready(candidate_result)
    elapsed_s = time.perf_counter() - started
    result, residual_vec, accepted = rhs1_accept_measured_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        candidate_result=candidate_result,
        current_residual_vec=current_residual_vec,
        candidate_residual_vec=candidate_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=candidate_result.x,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
        candidate_name=candidate_name,
        baseline_name=baseline_name,
        target_value=target_value,
        solve_s=elapsed_s,
        peak_rss_mb=peak_rss_mb,
    )
    return result, residual_vec, accepted, elapsed_s


def rhs1_run_linear_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    returns_residual_vec: bool = False,
    result_ready: Any = None,
) -> tuple[Any, Any, bool, float]:
    """Run and strict-improvement-gate a linear retry candidate.

    This is the non-measured counterpart to
    :func:`rhs1_run_measured_linear_candidate_and_update_replay`. It preserves
    the older driver behavior for rescue branches where any finite residual
    improvement is worth retaining, even if the candidate has not yet reached a
    solver-policy target.
    """

    started = time.perf_counter()
    candidate_output = solve_linear(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=current_result.x,
        tol_val=float(tol),
        atol_val=float(atol),
        restart_val=int(restart),
        maxiter_val=maxiter,
        solve_method_val=str(solve_method),
        precond_side=str(precond_side),
    )
    if returns_residual_vec:
        candidate_result, candidate_residual_vec = candidate_output
    else:
        candidate_result = candidate_output
        candidate_residual_vec = current_residual_vec
    if result_ready is not None:
        candidate_result = result_ready(candidate_result)
    elapsed_s = time.perf_counter() - started
    result, residual_vec, accepted = rhs1_accept_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        candidate_result=candidate_result,
        current_residual_vec=current_residual_vec,
        candidate_residual_vec=candidate_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=candidate_result.x,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
    )
    return result, residual_vec, accepted, elapsed_s


def rhs1_run_pas_schur_rescue_if_requested(
    *,
    replay_state: RHS1KSPReplayState,
    controls: Any,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    build_preconditioner: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    target: float,
    emit: Any = None,
) -> tuple[Any, Any, bool, float]:
    """Run the full-system PAS Schur rescue when policy admission requests it."""

    if not bool(getattr(controls, "run", False)):
        return current_result, current_residual_vec, False, 0.0
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: PAS Schur rescue "
            f"(residual={float(current_result.residual_norm):.3e} "
            f"> {float(target) * float(getattr(controls, 'ratio')):.3e})",
        )
    try:
        schur_precond = build_preconditioner()
        return rhs1_run_linear_candidate_and_update_replay(
            replay_state=replay_state,
            current_result=current_result,
            current_residual_vec=current_residual_vec,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=schur_precond,
            tol=float(tol),
            atol=float(atol),
            restart=int(restart),
            maxiter=maxiter,
            solve_method="incremental",
            precond_side=precond_side,
            solve_linear=solve_linear,
            solver_kind=solver_kind,
            returns_residual_vec=True,
        )
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: PAS Schur rescue failed "
                f"({type(exc).__name__}: {exc})",
            )
        return current_result, current_residual_vec, False, 0.0


def rhs1_run_full_pas_schur_rescue_from_env(
    *,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    build_preconditioner: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    n_species: int,
    active_size: int,
    emit: Any = None,
) -> tuple[Any, Any, bool, float]:
    """Resolve PAS-Schur rescue controls and run the full-system rescue."""

    controls = rhs1_pas_schur_rescue_controls_from_env(
        rhs_mode=int(rhs_mode),
        include_phi1=bool(include_phi1),
        has_pas=bool(has_pas),
        n_species=int(n_species),
        residual_norm=float(current_result.residual_norm),
        target=float(target),
        active_size=int(active_size),
        restart=int(restart),
        maxiter=maxiter,
    )
    return rhs1_run_pas_schur_rescue_if_requested(
        replay_state=replay_state,
        controls=controls,
        current_result=current_result,
        current_residual_vec=current_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        build_preconditioner=build_preconditioner,
        tol=float(tol),
        atol=float(atol),
        restart=int(controls.restart),
        maxiter=int(controls.maxiter),
        precond_side=precond_side,
        solve_linear=solve_linear,
        solver_kind=solver_kind,
        target=float(target),
        emit=emit,
    )


def rhs1_run_collision_retry_if_allowed(
    *,
    allowed: bool,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    build_preconditioner: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    target: float,
    returns_residual_vec: bool,
    emit: Any = None,
) -> tuple[Any, Any, Any, bool, float]:
    """Run a collision-preconditioner retry while preserving cache handoff."""

    if not bool(allowed):
        return current_result, current_residual_vec, precond_fn, False, 0.0
    preconditioner = precond_fn
    if preconditioner is None:
        preconditioner = build_preconditioner()
    if preconditioner is None:
        return current_result, current_residual_vec, preconditioner, False, 0.0
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: retry with collision preconditioner "
            f"(residual={float(current_result.residual_norm):.3e} > target={float(target):.3e})",
        )
    result, residual_vec, accepted, elapsed_s = rhs1_run_linear_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        current_residual_vec=current_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=preconditioner,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method="incremental",
        precond_side=precond_side,
        solve_linear=solve_linear,
        solver_kind=solver_kind,
        returns_residual_vec=bool(returns_residual_vec),
    )
    return result, residual_vec, preconditioner, accepted, elapsed_s


def rhs1_run_stage2_retry_if_allowed(
    *,
    allowed: bool,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    preconditioner_enabled: bool,
    build_preconditioner: Any,
    controls: Any,
    tol: float,
    atol: float,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    candidate_name: str,
    baseline_name: str,
    target: float,
    peak_rss_mb: Any,
    returns_residual_vec: bool,
    result_ready: Any = None,
    emit: Any = None,
    label: str = "stage2 GMRES",
) -> tuple[Any, Any, Any, bool, float]:
    """Run a Stage-2 Krylov retry while preserving preconditioner cache state."""

    if not bool(allowed):
        return current_result, current_residual_vec, precond_fn, False, 0.0
    preconditioner = precond_fn
    if preconditioner is None and bool(preconditioner_enabled):
        preconditioner = build_preconditioner()

    restart = int(getattr(controls, "restart"))
    maxiter = int(getattr(controls, "maxiter"))
    method = str(getattr(controls, "method"))
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: "
            f"{label} (residual={float(current_result.residual_norm):.3e} "
            f"> target={float(target):.3e}) restart={restart} "
            f"maxiter={maxiter} method={method}",
        )
    peak_rss_value = peak_rss_mb() if callable(peak_rss_mb) else peak_rss_mb
    result, residual_vec, accepted, elapsed_s = (
        rhs1_run_measured_linear_candidate_and_update_replay(
            replay_state=replay_state,
            current_result=current_result,
            current_residual_vec=current_residual_vec,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=preconditioner,
            tol=float(tol),
            atol=float(atol),
            restart=restart,
            maxiter=maxiter,
            solve_method=method,
            precond_side=precond_side,
            solve_linear=solve_linear,
            solver_kind=solver_kind,
            candidate_name=candidate_name,
            baseline_name=baseline_name,
            target_value=float(target),
            peak_rss_mb=peak_rss_value,
            returns_residual_vec=bool(returns_residual_vec),
            result_ready=result_ready,
        )
    )
    return result, residual_vec, preconditioner, accepted, elapsed_s


def rhs1_run_bicgstab_gmres_fallback_if_allowed(
    *,
    allowed: bool,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    preconditioner_enabled: bool,
    build_preconditioner: Any,
    x0_vec: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solve_linear: Any,
    target: float,
    returns_residual_vec: bool,
    result_ready: Any = None,
    emit: Any = None,
) -> tuple[Any, Any, Any, bool, float]:
    """Retry a BiCGStab solve with GMRES and update replay unconditionally."""

    if not bool(allowed):
        return current_result, current_residual_vec, precond_fn, False, 0.0
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: BiCGStab fallback to GMRES "
            f"(residual={float(current_result.residual_norm):.3e} > target={float(target):.3e})",
        )
    preconditioner = precond_fn
    if preconditioner is None and bool(preconditioner_enabled):
        preconditioner = build_preconditioner()
    started = time.perf_counter()
    candidate_output = solve_linear(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=preconditioner,
        x0_vec=x0_vec,
        tol_val=float(tol),
        atol_val=float(atol),
        restart_val=int(restart),
        maxiter_val=maxiter,
        solve_method_val="incremental",
        precond_side=str(precond_side),
    )
    if returns_residual_vec:
        candidate_result, candidate_residual_vec = candidate_output
    else:
        candidate_result = candidate_output
        candidate_residual_vec = current_residual_vec
    if result_ready is not None:
        candidate_result = result_ready(candidate_result)
    elapsed_s = time.perf_counter() - started
    rhs1_apply_handoff_to_replay_state(
        replay_state,
        RHS1KSPHandoffState(
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=preconditioner,
            x0_vec=x0_vec,
            restart=int(restart),
            maxiter=maxiter,
            precond_side=str(precond_side),
            solver_kind="gmres",
        ),
    )
    return candidate_result, candidate_residual_vec, preconditioner, True, elapsed_s


def rhs1_run_primary_krylov_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    x0_vec: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    returns_residual_vec: bool,
    current_residual_vec: Any = None,
    result_ready: Any = None,
    progress_start: Any = None,
    mark: Any = None,
    mark_start: str | None = None,
    mark_done: str | None = None,
    block_residual_until_ready: bool = False,
    update_krylov_controls: bool = False,
) -> tuple[Any, Any, float]:
    """Run a primary Krylov solve and record the replay problem.

    The legacy driver initialized replay restart/maxiter once near the top of
    the solve. Most primary-solve replay updates only replaced the linear
    problem and solver kind, so ``update_krylov_controls`` stays opt-in.
    """

    if progress_start is not None:
        progress_start()
    if mark is not None and mark_start is not None:
        mark(mark_start)
    started = time.perf_counter()
    candidate_output = solve_linear(
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=x0_vec,
        tol_val=float(tol),
        atol_val=float(atol),
        restart_val=int(restart),
        maxiter_val=maxiter,
        solve_method_val=str(solve_method),
        precond_side=str(precond_side),
    )
    if returns_residual_vec:
        result, residual_vec = candidate_output
    else:
        result = candidate_output
        residual_vec = current_residual_vec
    if result_ready is not None:
        result = result_ready(result)
    if block_residual_until_ready and residual_vec is not None:
        try:
            residual_vec.block_until_ready()
        except Exception:
            pass
    elapsed_s = time.perf_counter() - started
    if mark is not None and mark_done is not None:
        mark(mark_done)
    rhs1_record_ksp_replay_problem(
        replay_state,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=x0_vec,
        precond_side=precond_side,
        solver_kind=solver_kind,
        restart=int(restart) if update_krylov_controls else None,
        maxiter=maxiter if update_krylov_controls else None,
    )
    return result, residual_vec, elapsed_s


def rhs1_seed_skip_primary_krylov_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    context: RHS1SkipPrimaryKrylovSeedContext,
) -> tuple[Any, Any]:
    """Create a seed result and replay problem when primary Krylov is skipped."""

    x0_vec = context.x0_vec
    if x0_vec is None:
        x0_vec = context.zero_like(context.b_vec)
    try:
        residual = context.b_vec - context.matvec_fn(x0_vec)
        residual_norm = context.norm(residual)
    except Exception:
        residual_norm = context.inf_residual()
    result = context.result_factory(x0_vec, residual_norm)
    rhs1_record_ksp_replay_problem(
        replay_state,
        matvec_fn=context.matvec_fn,
        b_vec=context.b_vec,
        precond_fn=context.precond_fn,
        x0_vec=x0_vec,
        precond_side=context.precond_side,
        solver_kind=context.solver_kind,
    )
    return result, x0_vec


def rhs1_skip_primary_krylov_reason(
    *,
    gpu_dkes_sparse_shortcut: bool,
    cpu_large_xblock_shortcut: bool,
    cpu_large_sparse_shortcut: bool,
    backend_name: str,
) -> str:
    """Return the user-facing reason for bypassing the initial Krylov solve."""

    backend = str(backend_name).strip().lower()
    if gpu_dkes_sparse_shortcut:
        return "GPU DKES auto sparse shortcut"
    if cpu_large_xblock_shortcut:
        return (
            "CPU large FP x-block shortcut"
            if backend == "cpu"
            else f"{backend} host-sparse FP x-block shortcut"
        )
    if cpu_large_sparse_shortcut:
        return (
            "CPU large FP sparse-LU shortcut"
            if backend == "cpu"
            else f"{backend} host-sparse FP sparse-LU shortcut"
        )
    return "probe ratio huge, dense disabled"


def rhs1_retry_without_preconditioner_if_nonfinite(
    *,
    allowed: bool,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    x0_vec: Any,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    solve_method: str,
    precond_side: str,
    solve_linear: Any,
    solver_kind: str,
    result_is_finite: Any,
    returns_residual_vec: bool,
    result_ready: Any = None,
    mark: Any = None,
    mark_start: str | None = None,
    mark_done: str | None = None,
    block_residual_until_ready: bool = False,
    emit: Any = None,
    message: str = (
        "solve_v3_full_system_linear_gmres: preconditioned GMRES returned "
        "non-finite result; retrying without preconditioner"
    ),
) -> tuple[Any, Any, bool, float]:
    """Retry without a preconditioner after a nonfinite preconditioned solve."""

    if not bool(allowed) or bool(result_is_finite(current_result)):
        return current_result, current_residual_vec, False, 0.0
    if emit is not None:
        emit(0, message)
    result, residual_vec, elapsed_s = rhs1_run_primary_krylov_and_update_replay(
        replay_state=replay_state,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=None,
        x0_vec=x0_vec,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        solve_method=solve_method,
        precond_side=precond_side,
        solve_linear=solve_linear,
        solver_kind=solver_kind,
        returns_residual_vec=bool(returns_residual_vec),
        current_residual_vec=current_residual_vec,
        result_ready=result_ready,
        mark=mark,
        mark_start=mark_start,
        mark_done=mark_done,
        block_residual_until_ready=bool(block_residual_until_ready),
    )
    return result, residual_vec, True, elapsed_s


def rhs1_accept_smoother_candidate_and_update_replay(
    *,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    smoother: Any,
    result_factory: Any,
    candidate_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solver_kind: str,
    emit: Any = None,
    label: str = "PAS adaptive smoother",
) -> tuple[Any, Any, bool]:
    """Accept an already-run smoother candidate and update replay state.

    The driver still owns the smoother execution and the residual-vector
    routing. ``candidate_residual_vec`` may be either a precomputed vector or a
    callable that receives the constructed candidate result.
    """

    current_residual = float(current_result.residual_norm)
    smoother_residual = float(smoother.residual_norm)
    if not rhs1_residual_improves(
        current_residual=current_residual,
        candidate_residual=smoother_residual,
    ):
        return current_result, current_residual_vec, False
    candidate_result = result_factory(
        x=smoother.x,
        residual_norm=smoother_residual,
    )
    residual_vec_candidate = (
        candidate_residual_vec(candidate_result)
        if callable(candidate_residual_vec)
        else candidate_residual_vec
    )
    if emit is not None:
        emit(
            1,
            f"solve_v3_full_system_linear_gmres: {label} "
            f"accepted {int(getattr(smoother, 'accepted_sweeps', 0))} sweep(s), "
            f"reason={getattr(smoother, 'stop_reason', 'unknown')}, "
            f"residual={current_residual:.3e}->{smoother_residual:.3e}",
        )
    return rhs1_accept_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        candidate_result=candidate_result,
        current_residual_vec=current_residual_vec,
        candidate_residual_vec=residual_vec_candidate,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        x0_vec=candidate_result.x,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
    )


def rhs1_run_adaptive_smoother_and_update_replay(
    *,
    allowed: bool,
    replay_state: RHS1KSPReplayState,
    current_result: Any,
    current_residual_vec: Any,
    smoother_factory: Any,
    result_factory: Any,
    candidate_residual_vec: Any,
    matvec_fn: Any,
    b_vec: Any,
    precond_fn: Any,
    restart: int,
    maxiter: int | None,
    precond_side: str,
    solver_kind: str,
    emit: Any = None,
    label: str = "PAS adaptive smoother",
) -> tuple[Any, Any, bool]:
    """Run an admitted adaptive smoother and accept it only if residual improves."""

    if not bool(allowed):
        return current_result, current_residual_vec, False
    smoother = smoother_factory(current_result)
    return rhs1_accept_smoother_candidate_and_update_replay(
        replay_state=replay_state,
        current_result=current_result,
        current_residual_vec=current_residual_vec,
        smoother=smoother,
        result_factory=result_factory,
        candidate_residual_vec=candidate_residual_vec,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        precond_fn=precond_fn,
        restart=restart,
        maxiter=maxiter,
        precond_side=precond_side,
        solver_kind=solver_kind,
        emit=emit,
        label=label,
    )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3LinearSolveResult:
    """Result of one matrix-free RHSMode=1 profile-response solve."""

    op: Any
    rhs: jnp.ndarray
    gmres: GMRESSolveResult
    metadata: dict[str, object] | None = None

    def tree_flatten(self):
        children = (self.op, self.rhs, self.gmres)
        aux = self.metadata
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, rhs, gmres_result = children
        return cls(op=op, rhs=rhs, gmres=gmres_result, metadata=aux)

    @property
    def x(self) -> jnp.ndarray:
        return self.gmres.x

    @property
    def residual_norm(self) -> jnp.ndarray:
        return self.gmres.residual_norm


def v3_linear_solve_result_from_payload(
    *,
    op: Any,
    rhs: jnp.ndarray,
    payload: Any,
) -> V3LinearSolveResult:
    """Wrap a sparse-PC payload in the profile-response linear-solve result."""

    return V3LinearSolveResult(
        op=op,
        rhs=rhs,
        gmres=GMRESSolveResult(
            x=payload.x,
            residual_norm=payload.residual_norm,
        ),
        metadata=payload.metadata,
    )


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class V3NewtonKrylovResult:
    """Result of a nonlinear Phi1 Newton-Krylov profile-response solve."""

    op: Any
    x: jnp.ndarray
    residual_norm: jnp.ndarray
    n_newton: int
    last_linear_residual_norm: jnp.ndarray

    def tree_flatten(self):
        children = (self.op, self.x, self.residual_norm, self.last_linear_residual_norm)
        aux = int(self.n_newton)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        op, x, residual_norm, last_linear_residual_norm = children
        return cls(
            op=op,
            x=x,
            residual_norm=residual_norm,
            n_newton=int(aux),
            last_linear_residual_norm=last_linear_residual_norm,
        )


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
