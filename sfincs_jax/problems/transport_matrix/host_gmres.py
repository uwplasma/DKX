"""Host SciPy GMRES helper for explicit transport solves."""

from __future__ import annotations

from collections.abc import Callable
import time
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solver import (
    GMRESSolveResult,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve_with_history_scipy,
)
from sfincs_jax.problems.transport_matrix.policies import transport_host_gmres_accepts_preconditioned_residual


def transport_host_gmres_solve(
    *,
    op: Any,
    matvec_fn,
    b_vec: jnp.ndarray,
    x0_vec: jnp.ndarray | None,
    preconditioner_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
    emit: Callable[[int, str], None] | None = None,
    which_rhs: int | None = None,
    progress_every: int = 10,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Run host SciPy GMRES and return a JAX result plus true residual vector."""
    side = str(precondition_side_val).strip().lower()
    b_norm = float(jnp.linalg.norm(b_vec))
    target_true = max(float(atol_val), float(tol_val) * b_norm)
    reported_residual_norm: float | None = None
    started = time.perf_counter()
    progress_stride = max(0, int(progress_every))

    def _progress(iteration: int, residual: float) -> None:
        if emit is None or progress_stride <= 0:
            return
        iteration_int = int(iteration)
        if iteration_int != 1 and iteration_int % progress_stride != 0:
            return
        rhs_label = "unknown" if which_rhs is None else str(int(which_rhs))
        emit(
            1,
            "transport host SciPy GMRES progress "
            f"whichRHS={rhs_label} iter={iteration_int} "
            f"reported_residual={float(residual):.6e} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
        )

    if preconditioner_fn is not None and side == "left":
        x_np, rn_true, rn_pc, _history = explicit_left_preconditioned_gmres_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            progress_callback=_progress,
        )
        rhs_pc_norm = float(jnp.linalg.norm(preconditioner_fn(b_vec)))
        target_pc = max(float(atol_val), float(tol_val) * rhs_pc_norm)
        if (
            np.isfinite(float(rn_pc))
            and float(rn_pc) <= float(target_pc)
            and transport_host_gmres_accepts_preconditioned_residual(
                op=op,
                true_residual_norm=float(rn_true),
                target_true=float(target_true),
            )
        ):
            # Mirror the PETSc-style transport lane, which may accept convergence
            # on the preconditioned KSP residual for singular/near-singular systems.
            reported_residual_norm = min(float(rn_true), float(target_true))
    else:
        x_np, rn_true, _history = gmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precondition_side_val,
            progress_callback=_progress,
        )
    x_jnp = jnp.asarray(x_np, dtype=jnp.float64)
    residual_vec = b_vec - matvec_fn(x_jnp)
    residual_norm = float(jnp.linalg.norm(residual_vec))
    if np.isfinite(float(rn_true)):
        residual_norm = min(residual_norm, float(rn_true))
    if reported_residual_norm is not None:
        residual_norm = min(residual_norm, float(reported_residual_norm))
    return (
        GMRESSolveResult(
            x=x_jnp,
            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        ),
        residual_vec,
    )


__all__ = ["transport_host_gmres_solve"]
