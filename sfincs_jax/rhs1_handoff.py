"""Shared RHSMode=1 solve-handoff helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
) -> tuple[Any, Any, RHS1KSPHandoffState | None, bool]:
    """Accept a candidate only if it strictly improves the residual norm."""
    if float(candidate_result.residual_norm) < float(current_result.residual_norm):
        accepted_residual_vec = candidate_residual_vec if candidate_residual_vec is not None else current_residual_vec
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


__all__ = ["RHS1KSPHandoffState", "rhs1_accept_candidate"]
