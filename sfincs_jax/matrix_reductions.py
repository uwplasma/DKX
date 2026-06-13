"""Matrix reduction helpers used by simplified preconditioner operators."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def diagonal_only(matrix: jnp.ndarray) -> jnp.ndarray:
    """Return a diagonal-only copy of a square matrix."""

    return jnp.diag(jnp.diag(matrix))


def block_diagonal_only(matrix: jnp.ndarray, block: int) -> jnp.ndarray:
    """Return a block-diagonal copy of a square matrix."""

    if int(block) <= 1:
        return diagonal_only(matrix)
    matrix_np = np.asarray(matrix, dtype=np.float64)
    n = int(matrix_np.shape[0])
    mask = np.zeros((n, n), dtype=bool)
    for start in range(0, n, int(block)):
        end = min(n, start + int(block))
        mask[start:end, start:end] = True
    matrix_np = np.where(mask, matrix_np, 0.0)
    return jnp.asarray(matrix_np, dtype=matrix.dtype)


__all__ = ["block_diagonal_only", "diagonal_only"]
