"""Host-side iterative-refinement helpers for sparse and dense direct solves.

These routines are intentionally NumPy-only.  They sit below the JAX driver
paths that build matrices and factors, and above SciPy/SuperLU solve objects.
Keeping them isolated makes the residual-polishing behavior easy to test
without importing the full v3 driver.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


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


__all__ = [
    "host_direct_solve_with_refinement",
    "host_sparse_direct_solve_with_refinement",
]
