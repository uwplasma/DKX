"""Host-side refinement and polish helpers for direct solve fallbacks.

The direct refinement routines are NumPy-only. The sparse polish helper accepts
a JAX matvec because it uses a host factor as a preconditioner inside SciPy
GMRES. Keeping them isolated makes residual-improvement behavior testable
without importing the full v3 driver.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np

from .solver import gmres_solve_with_history_scipy


def host_sparse_direct_solve_with_refinement(
    *,
    ilu,
    a_csr_full,
    rhs_vec,
    factor_dtype: np.dtype,
    refine_steps: int,
) -> tuple[np.ndarray, float]:
    """Solve with a sparse host factor and monotone iterative refinement.

    Refinement stops when the residual is non-finite, reaches zero, or a trial
    correction would increase the full double-precision residual norm.  This
    mirrors the conservative behavior needed by RHSMode=1 and transport direct
    solve fallbacks.
    """

    rhs64 = np.asarray(rhs_vec, dtype=np.float64).reshape((-1,))
    rhs_factor = np.asarray(rhs_vec, dtype=factor_dtype).reshape((-1,))
    x_np = np.asarray(ilu.solve(rhs_factor), dtype=np.float64)
    residual_np = rhs64 - a_csr_full @ x_np
    residual_norm = float(np.linalg.norm(residual_np))
    for _ in range(max(0, int(refine_steps))):
        if not np.isfinite(residual_norm) or residual_norm == 0.0:
            break
        dx_np = np.asarray(ilu.solve(np.asarray(residual_np, dtype=factor_dtype)), dtype=np.float64)
        x_trial = x_np + dx_np
        residual_trial = rhs64 - a_csr_full @ x_trial
        residual_norm_trial = float(np.linalg.norm(residual_trial))
        if not np.isfinite(residual_norm_trial) or residual_norm_trial >= residual_norm:
            break
        x_np = x_trial
        residual_np = residual_trial
        residual_norm = residual_norm_trial
    return x_np, residual_norm


def host_direct_solve_with_refinement(
    *,
    factor_solve: Callable[[np.ndarray], np.ndarray],
    operator_matrix,
    rhs_vec,
    factor_dtype: np.dtype,
    refine_steps: int,
) -> tuple[np.ndarray, float]:
    """Solve with a host direct factor callback and monotone refinement."""

    rhs64 = np.asarray(rhs_vec, dtype=np.float64).reshape((-1,))
    rhs_factor = np.asarray(rhs_vec, dtype=factor_dtype).reshape((-1,))
    x_np = np.asarray(factor_solve(rhs_factor), dtype=np.float64)
    residual_np = rhs64 - operator_matrix @ x_np
    residual_norm = float(np.linalg.norm(residual_np))
    for _ in range(max(0, int(refine_steps))):
        if not np.isfinite(residual_norm) or residual_norm == 0.0:
            break
        dx_np = np.asarray(factor_solve(np.asarray(residual_np, dtype=factor_dtype)), dtype=np.float64)
        x_trial = x_np + dx_np
        residual_trial = rhs64 - operator_matrix @ x_trial
        residual_norm_trial = float(np.linalg.norm(residual_trial))
        if not np.isfinite(residual_norm_trial) or residual_norm_trial >= residual_norm:
            break
        x_np = x_trial
        residual_np = residual_trial
        residual_norm = residual_norm_trial
    return x_np, residual_norm


def host_sparse_direct_polish(
    *,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    rhs_vec: jnp.ndarray,
    x0_np: np.ndarray,
    ilu,
    factor_dtype: np.dtype,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    gmres_solver: Callable[..., tuple[np.ndarray, float, list[float]]] | None = None,
) -> tuple[np.ndarray, float]:
    """Polish a host sparse-direct solution with preconditioned SciPy GMRES."""

    def _precond_sparse(v: jnp.ndarray) -> jnp.ndarray:
        v_np = np.asarray(v, dtype=factor_dtype).reshape((-1,))
        y_np = ilu.solve(v_np)
        return jnp.asarray(y_np, dtype=jnp.float64)

    solver = gmres_solve_with_history_scipy if gmres_solver is None else gmres_solver
    x_np, _rn_sparse, _history = solver(
        matvec=matvec_fn,
        b=rhs_vec,
        preconditioner=_precond_sparse,
        x0=jnp.asarray(x0_np, dtype=jnp.float64),
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        precondition_side=precondition_side,
    )
    x_polish = np.asarray(x_np, dtype=np.float64)
    residual_vec = rhs_vec - matvec_fn(jnp.asarray(x_polish, dtype=jnp.float64))
    residual_norm = float(jnp.linalg.norm(residual_vec))
    return x_polish, residual_norm


__all__ = [
    "host_direct_solve_with_refinement",
    "host_sparse_direct_polish",
    "host_sparse_direct_solve_with_refinement",
]
