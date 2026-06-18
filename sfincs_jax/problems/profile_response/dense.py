"""Host dense reduced-system helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ...solver import GMRESSolveResult, assemble_dense_matrix_from_matvec


@dataclass(frozen=True)
class HostDenseReducedSolveContext:
    """Solve-local inputs for a host dense reduced RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    active_size: int
    constraint_scheme: int
    has_fp: bool
    dense_matrix_cache: np.ndarray | None = None


@dataclass(frozen=True)
class HostDenseFullSolveContext:
    """Solve-local inputs for a host dense full-system RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    total_size: int


def solve_host_dense_reduced(
    *,
    context: HostDenseReducedSolveContext,
    x0: jnp.ndarray | None = None,
) -> GMRESSolveResult:
    """Solve the reduced system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    use_row_scaled = bool(int(context.constraint_scheme) == 0 or (int(context.constraint_scheme) == 1 and context.has_fp))
    if context.dense_matrix_cache is not None:
        a_np = np.asarray(context.dense_matrix_cache, dtype=np.float64)
    else:
        a_dense_jnp = assemble_dense_matrix_from_matvec(
            matvec=context.matvec,
            n=int(context.active_size),
            dtype=context.rhs.dtype,
        )
        a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)

    matvec_residual = context.matvec
    b_dense = jnp.asarray(context.rhs, dtype=jnp.float64)
    if use_row_scaled:
        diag_floor = 1e-12
        diag = np.diag(a_np).astype(np.float64, copy=False)
        diag_abs = np.abs(diag)
        diag_safe = np.where(diag_abs > diag_floor, diag, np.sign(diag) * diag_floor)
        diag_safe = np.where(diag_safe != 0.0, diag_safe, diag_floor)
        scale = (1.0 / diag_safe).astype(np.float64, copy=False)
        a_np = a_np * scale[:, None]
        scale_jnp = jnp.asarray(scale, dtype=jnp.float64)
        b_dense = b_dense * scale_jnp

        def matvec_residual(x_vec: jnp.ndarray) -> jnp.ndarray:
            return scale_jnp * context.matvec(x_vec)

    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(b_dense, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(b_dense, dtype=np.float64)), dtype=np.float64)
        if x0 is not None and x0.shape == context.rhs.shape:
            x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)

    r_dense = b_dense - matvec_residual(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))


def solve_host_dense_full(
    *,
    context: HostDenseFullSolveContext,
    x0: jnp.ndarray | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve the full system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    a_dense_jnp = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.total_size),
        dtype=context.rhs.dtype,
    )
    a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)
    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(context.rhs, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(context.rhs, dtype=np.float64)), dtype=np.float64)
    if x0 is not None and x0.shape == context.rhs.shape:
        x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
    x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    residual_vec = context.rhs - context.matvec(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_vec)), residual_vec


__all__ = [
    "HostDenseFullSolveContext",
    "HostDenseReducedSolveContext",
    "solve_host_dense_full",
    "solve_host_dense_reduced",
]
