"""JAX triangular-solve helpers for sparse preconditioner factors."""

from __future__ import annotations

import jax
from jax import config as _jax_config
import jax.numpy as jnp
import numpy as np

_jax_config.update("jax_enable_x64", True)


def inverse_permutation(p: np.ndarray) -> np.ndarray:
    """Return the inverse of a zero-based permutation."""

    p = np.asarray(p, dtype=np.int32).reshape((-1,))
    inv = np.empty_like(p)
    inv[p] = np.arange(int(p.size), dtype=np.int32)
    return inv


def triangular_solve_lower_padded(
    *,
    lower_idx: jnp.ndarray,
    lower_val: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a unit-lower triangular system ``L y = b`` from padded rows.

    ``lower_idx`` and ``lower_val`` define the strictly lower entries per row.
    Padding entries must use index ``-1`` and are ignored.
    """

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    y = jnp.zeros_like(b)
    if lower_idx.size == 0:
        return b

    def _body(i, y_vec):
        idx = lower_idx[i]
        val = lower_val[i]
        mask = idx >= 0
        idx_safe = jnp.where(mask, idx, 0)
        contrib = jnp.sum(jnp.where(mask, val * y_vec[idx_safe], 0.0))
        yi = b[i] - contrib
        return y_vec.at[i].set(yi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, y)


def triangular_solve_upper_padded(
    *,
    upper_idx: jnp.ndarray,
    upper_val: jnp.ndarray,
    upper_diag: jnp.ndarray,
    b: jnp.ndarray,
) -> jnp.ndarray:
    """Solve an upper triangular system ``U x = b`` from padded rows."""

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    x = jnp.zeros_like(b)
    if upper_idx.size == 0:
        return b / upper_diag

    def _body(i, x_vec):
        row = n - 1 - i
        idx = upper_idx[row]
        val = upper_val[row]
        mask = idx >= 0
        idx_safe = jnp.where(mask, idx, 0)
        contrib = jnp.sum(jnp.where(mask, val * x_vec[idx_safe], 0.0))
        xi = (b[row] - contrib) / upper_diag[row]
        return x_vec.at[row].set(xi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, x)


def triangular_solve_lower_csr_rows(
    *,
    indptr: jnp.ndarray,
    indices: jnp.ndarray,
    data: jnp.ndarray,
    b: jnp.ndarray,
    row_base: jnp.ndarray,
) -> jnp.ndarray:
    """Solve a unit-lower triangular block stored as compact CSR rows.

    ``row_base`` points to the first row pointer for this block inside a
    concatenated per-block CSR table. Column indices are local to the block.
    """

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    y = jnp.zeros_like(b)
    if data.size == 0:
        return b

    def _body(i, y_vec):
        row = row_base + i
        start = indptr[row]
        end = indptr[row + 1]

        def _accumulate(k, acc):
            return acc + data[k] * y_vec[indices[k]]

        contrib = jax.lax.fori_loop(start, end, _accumulate, jnp.asarray(0.0, dtype=b.dtype))
        return y_vec.at[i].set(b[i] - contrib, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, y)


def triangular_solve_upper_csr_rows(
    *,
    indptr: jnp.ndarray,
    indices: jnp.ndarray,
    data: jnp.ndarray,
    upper_diag: jnp.ndarray,
    b: jnp.ndarray,
    row_base: jnp.ndarray,
) -> jnp.ndarray:
    """Solve an upper triangular block stored as compact CSR rows."""

    b = jnp.asarray(b, dtype=jnp.float64)
    n = int(b.shape[0])
    x = jnp.zeros_like(b)
    if data.size == 0:
        return b / upper_diag

    def _body(i, x_vec):
        row_local = n - 1 - i
        row = row_base + row_local
        start = indptr[row]
        end = indptr[row + 1]

        def _accumulate(k, acc):
            return acc + data[k] * x_vec[indices[k]]

        contrib = jax.lax.fori_loop(start, end, _accumulate, jnp.asarray(0.0, dtype=b.dtype))
        xi = (b[row_local] - contrib) / upper_diag[row_local]
        return x_vec.at[row_local].set(xi, unique_indices=True)

    return jax.lax.fori_loop(0, n, _body, x)
