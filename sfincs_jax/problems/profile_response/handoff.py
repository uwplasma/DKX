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

from ...solver_selection_policy import (
    SolverAcceptanceCriteria,
    SolverCandidateMetrics,
    solver_candidate_gate,
)


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


__all__ = [
    "RHS1KSPHandoffState",
    "RHS1KSPReplayState",
    "rhs1_apply_handoff_to_replay_state",
    "rhs1_accept_candidate",
    "rhs1_accept_candidate_and_update_replay",
    "rhs1_accept_measured_candidate",
    "rhs1_accept_measured_candidate_and_update_replay",
    "rhs1_accept_sparse_retry_candidate_and_update_replay",
    "rhs1_residual_improves",
    "rhs1_accept_smoother_candidate_and_update_replay",
    "rhs1_run_fast_post_xblock_polish",
    "rhs1_run_linear_candidate_and_update_replay",
    "rhs1_run_measured_linear_candidate_and_update_replay",
    "rhs1_solver_candidate_metrics",
]
