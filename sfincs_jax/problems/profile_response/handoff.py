"""Shared RHSMode=1 solve-handoff helpers.

The RHSMode=1 driver tries several bounded rescue and refinement candidates
after a primary Krylov solve. This module keeps the small but repeated
acceptance contract in one pure place: a candidate must strictly improve the
current residual, any measured promotion gate must pass, and only an accepted
candidate may update the residual vector and KSP replay state used for
Fortran-style iteration diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from ...solvers.selection_policy import (
    SolverAcceptanceCriteria,
    SolverCandidateMetrics,
    solver_candidate_gate,
)
from sfincs_jax.solvers.preconditioners.pas.policy import rhs1_pas_schur_rescue_controls_from_env


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


__all__ = [
    "RHS1KSPHandoffState",
    "RHS1KSPReplayState",
    "RHS1SkipPrimaryKrylovSeedContext",
    "rhs1_apply_handoff_to_replay_state",
    "rhs1_accept_candidate",
    "rhs1_accept_candidate_and_update_replay",
    "rhs1_accept_measured_candidate",
    "rhs1_accept_measured_candidate_and_update_replay",
    "rhs1_accept_sparse_retry_candidate_and_update_replay",
    "rhs1_residual_improves",
    "rhs1_record_ksp_replay_problem",
    "rhs1_accept_smoother_candidate_and_update_replay",
    "rhs1_run_adaptive_smoother_and_update_replay",
    "rhs1_retry_without_preconditioner_if_nonfinite",
    "rhs1_run_bicgstab_gmres_fallback_if_allowed",
    "rhs1_run_fast_post_xblock_polish",
    "rhs1_run_collision_retry_if_allowed",
    "rhs1_run_linear_candidate_and_update_replay",
    "rhs1_run_measured_linear_candidate_and_update_replay",
    "rhs1_run_full_pas_schur_rescue_from_env",
    "rhs1_run_pas_schur_rescue_if_requested",
    "rhs1_run_primary_krylov_and_update_replay",
    "rhs1_seed_skip_primary_krylov_and_update_replay",
    "rhs1_skip_primary_krylov_reason",
    "rhs1_run_stage2_retry_if_allowed",
    "rhs1_solver_candidate_metrics",
]
