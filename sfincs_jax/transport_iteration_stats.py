"""Optional Krylov iteration diagnostics for transport solves."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp

from .solver import bicgstab_solve_with_history_scipy, gmres_solve_with_history_scipy


EmitFn = Callable[[int, str], None]


def emit_transport_ksp_iteration_stats(
    *,
    which_rhs: int,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    emit: EmitFn | None,
    enabled: bool,
    max_size: int | None,
) -> None:
    """Emit optional SciPy KSP iteration counts without affecting the solve.

    The diagnostics re-run the requested Krylov method on the host for small
    systems only.  Any diagnostic failure is reported and swallowed so that the
    production transport solve remains the source of truth.
    """
    if emit is None or not enabled:
        return
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"whichRHS={which_rhs} ksp_iterations skipped (size={size} > max={int(max_size)})")
        return
    solver_kind_l = str(solver_kind).strip().lower()
    try:
        history = _solve_history(
            solver_kind=solver_kind_l,
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
        emit(1, f"whichRHS={which_rhs} ksp_iterations unavailable ({type(exc).__name__}: {exc})")
        return
    if history is None:
        return
    emit(0, f"whichRHS={which_rhs} ksp_iterations={len(history)} solver={solver_kind_l}")


def _solve_history(
    *,
    solver_kind: str,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
) -> list[Any] | None:
    if solver_kind == "gmres":
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
    if solver_kind == "bicgstab":
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
        return history
    return None


__all__ = [
    "emit_transport_ksp_iteration_stats",
]
