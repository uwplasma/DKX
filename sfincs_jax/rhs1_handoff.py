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
from typing import Any

from .solver_selection_policy import (
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


__all__ = [
    "RHS1KSPHandoffState",
    "rhs1_accept_candidate",
    "rhs1_residual_improves",
]
