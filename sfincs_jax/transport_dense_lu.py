from __future__ import annotations

"""Cached dense LU helpers for bounded transport fallbacks."""

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.scipy.linalg as jla

from .solver import assemble_dense_matrix_from_matvec


def dense_preconditioner_for_matvec(
    *,
    matvec_fn,
    n: int,
    dtype: jnp.dtype,
    cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
    key: tuple[Any, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build or reuse a dense-LU preconditioner for a matrix-free operator."""
    if key in cache:
        return cache[key]
    a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=int(n), dtype=dtype)
    a_dense = jnp.asarray(a_dense, dtype=dtype)
    lu, piv = jla.lu_factor(a_dense)

    def precond(v: jnp.ndarray) -> jnp.ndarray:
        return jla.lu_solve((lu, piv), v)

    cache[key] = precond
    return precond


def dense_solver_for_matvec(
    *,
    matvec_fn,
    n: int,
    dtype: jnp.dtype,
    cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
    key: tuple[Any, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build or reuse a dense-LU direct solver for a matrix-free operator."""
    if key in cache:
        return cache[key]
    a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=int(n), dtype=dtype)
    a_dense = jnp.asarray(a_dense, dtype=dtype)
    lu, piv = jla.lu_factor(a_dense)

    def solve(v: jnp.ndarray) -> jnp.ndarray:
        return jla.lu_solve((lu, piv), v)

    cache[key] = solve
    return solve


__all__ = [
    "dense_preconditioner_for_matvec",
    "dense_solver_for_matvec",
]
