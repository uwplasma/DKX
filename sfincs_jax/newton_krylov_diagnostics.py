"""Optional diagnostics for v3 Newton-Krylov full-system solves."""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from .solver import gmres_solve_with_history_scipy


EmitFn = Callable[[int, str], None]


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


__all__ = ["emit_newton_krylov_ksp_history"]
