"""JAX-native block factor kernels for RHSMode=1 field-split solvers.

These helpers are intentionally small and dependency-free.  They capture the
parts of PETSc/MUMPS-style preconditioning that are useful for SFINCS_JAX:
deterministic block solves, block-Jacobi smoothing, and exact two-field Schur
updates.  Larger RHSMode=1 preconditioners should assemble physics-aware blocks
and call these kernels rather than depending on host sparse direct packages.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


class NativeDenseBlockJacobi(NamedTuple):
    """Regularized inverse factors for equal-sized dense diagonal blocks."""

    block_inverses: jax.Array
    block_size: int
    original_size: int
    padded_size: int
    regularization: float


class NativeTwoFieldSchurFactor(NamedTuple):
    """Exact block-LDU factorization for a two-field dense split."""

    a_ff_inv: jax.Array
    a_fc: jax.Array
    a_cf: jax.Array
    schur_inv: jax.Array
    f_size: int
    c_size: int
    regularization: float


class NativeXEllKineticFactor(NamedTuple):
    """Device-compatible ``(x, ell)`` kinetic line inverse factor.

    The factor mirrors the host CSR kinetic preconditioner: each fixed
    ``(species, theta, zeta)`` line stores a dense inverse over all
    ``(x, ell)`` unknowns, and the non-kinetic tail is either identity or a
    scalar Jacobi inverse.  It is intentionally a PyTree of small arrays plus
    static sizes so solver code can close over it inside JIT-compiled matvecs.
    """

    block_inverses: jax.Array
    block_indices: jax.Array
    inv_tail: jax.Array
    f_size: int
    total_size: int


class NativePaddedIndexedBlockFactor(NamedTuple):
    """Device-compatible inverse factors for padded indexed blocks.

    Each block stores a dense inverse over ``block_indices[b]``.  ``block_mask``
    marks the physically retained rows/columns, allowing variable-size local
    blocks to share one static padded shape.  Overlapping blocks are combined by
    scatter-add and, by default, normalized by their coverage count.  This is
    the JAX-native building block needed for additive-Schwarz and mixed
    angular/velocity-line preconditioners.
    """

    block_inverses: jax.Array
    block_indices: jax.Array
    block_mask: jax.Array
    overlap_weights: jax.Array
    total_size: int
    normalize_overlap: bool
    damping: float


def _regularized_matrix(matrix: jax.Array, regularization: float) -> jax.Array:
    """Return ``matrix + regularization * scale * I`` with a robust scale."""

    mat = jnp.asarray(matrix, dtype=jnp.float64)
    n = int(mat.shape[0])
    if n == 0:
        return mat
    scale = jnp.maximum(jnp.linalg.norm(mat, ord=jnp.inf), jnp.asarray(1.0, dtype=mat.dtype))
    reg = jnp.asarray(abs(float(regularization)), dtype=mat.dtype) * scale
    return mat + reg * jnp.eye(n, dtype=mat.dtype)


def build_native_x_ell_kinetic_factor(
    *,
    block_inverses: jax.Array,
    block_indices: jax.Array,
    f_size: int,
    total_size: int,
    inv_tail: jax.Array | None = None,
) -> NativeXEllKineticFactor:
    """Build a JAX-native factor from extracted ``x_ell`` inverse blocks."""

    inverses = jnp.asarray(block_inverses, dtype=jnp.float64)
    indices = jnp.asarray(block_indices, dtype=jnp.int32)
    if inverses.ndim != 3 or int(inverses.shape[1]) != int(inverses.shape[2]):
        raise ValueError("block_inverses must have shape (n_blocks, block_size, block_size)")
    if indices.ndim != 2 or indices.shape != inverses.shape[:2]:
        raise ValueError("block_indices must have shape (n_blocks, block_size)")
    f_size_i = int(f_size)
    total_size_i = int(total_size)
    if f_size_i <= 0 or total_size_i < f_size_i:
        raise ValueError("total_size must be greater than or equal to a positive f_size")
    if indices.size and (int(jnp.min(indices)) < 0 or int(jnp.max(indices)) >= f_size_i):
        raise ValueError("block_indices must lie inside the kinetic f block")
    if inv_tail is None:
        tail = jnp.zeros((0,), dtype=jnp.float64)
    else:
        tail = jnp.asarray(inv_tail, dtype=jnp.float64).reshape((-1,))
        tail_size = total_size_i - f_size_i
        if int(tail.shape[0]) not in {0, tail_size}:
            raise ValueError("inv_tail must be empty or have one entry per non-kinetic tail row")
    return NativeXEllKineticFactor(
        block_inverses=inverses,
        block_indices=indices,
        inv_tail=tail,
        f_size=f_size_i,
        total_size=total_size_i,
    )


def apply_native_x_ell_kinetic_factor(factor: NativeXEllKineticFactor, rhs: jax.Array) -> jax.Array:
    """Apply a native ``x_ell`` kinetic factor to one or more RHS columns."""

    arr = jnp.asarray(rhs, dtype=jnp.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr_2d = arr[:, None]
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError("rhs must be a vector or a two-dimensional array")
    total = int(factor.total_size)
    f_size = int(factor.f_size)
    if int(arr_2d.shape[0]) != total:
        raise ValueError("rhs leading dimension does not match the factor size")
    n_cols = int(arr_2d.shape[1])
    gathered = arr_2d[:f_size, :][factor.block_indices]
    solved = jnp.einsum("bij,bjk->bik", factor.block_inverses, gathered)
    flat_indices = factor.block_indices.reshape((-1,))
    flat_solved = solved.reshape((-1, n_cols))
    f_out = jnp.zeros((f_size, n_cols), dtype=arr_2d.dtype).at[flat_indices, :].set(flat_solved)
    tail_size = total - f_size
    if tail_size > 0:
        tail_rhs = arr_2d[f_size:, :]
        if int(factor.inv_tail.shape[0]) == tail_size:
            tail_out = factor.inv_tail[:, None] * tail_rhs
        else:
            tail_out = tail_rhs
        out = jnp.concatenate([f_out, tail_out], axis=0)
    else:
        out = f_out
    if len(original_shape) == 1:
        return out[:, 0]
    return out


def build_native_padded_indexed_block_factor(
    *,
    block_inverses: jax.Array,
    block_indices: jax.Array,
    total_size: int,
    block_mask: jax.Array | None = None,
    overlap_weights: jax.Array | None = None,
    normalize_overlap: bool = True,
    damping: float = 1.0,
) -> NativePaddedIndexedBlockFactor:
    """Build a padded indexed block factor from precomputed inverse blocks.

    ``block_indices`` has shape ``(n_blocks, max_block_size)``.  Invalid padded
    slots should be marked ``False`` in ``block_mask``; their index value is
    ignored and safely replaced by zero inside the JAX scatter/gather kernels.
    """

    inverses = jnp.asarray(block_inverses, dtype=jnp.float64)
    indices = jnp.asarray(block_indices, dtype=jnp.int32)
    if inverses.ndim != 3 or int(inverses.shape[1]) != int(inverses.shape[2]):
        raise ValueError("block_inverses must have shape (n_blocks, block_size, block_size)")
    if indices.ndim != 2 or indices.shape != inverses.shape[:2]:
        raise ValueError("block_indices must have shape (n_blocks, block_size)")
    total = int(total_size)
    if total <= 0:
        raise ValueError("total_size must be positive")
    if block_mask is None:
        mask = jnp.ones(indices.shape, dtype=bool)
    else:
        mask = jnp.asarray(block_mask, dtype=bool)
        if mask.shape != indices.shape:
            raise ValueError("block_mask must have shape (n_blocks, block_size)")
    if int(indices.shape[0]) == 0 or int(indices.shape[1]) == 0:
        raise ValueError("at least one non-empty indexed block is required")
    if bool(jnp.any(mask)):
        safe_indices = jnp.where(mask, indices, 0)
        if int(jnp.min(safe_indices)) < 0 or int(jnp.max(safe_indices)) >= total:
            raise ValueError("block_indices must lie inside [0, total_size) wherever block_mask is true")
    else:
        raise ValueError("at least one active block entry is required")

    if overlap_weights is None:
        safe_indices = jnp.where(mask, indices, 0)
        counts = jnp.zeros((total,), dtype=jnp.float64).at[safe_indices.reshape((-1,))].add(
            mask.astype(jnp.float64).reshape((-1,))
        )
        weights = jnp.where(counts > 0.0, counts, 1.0)
    else:
        weights = jnp.asarray(overlap_weights, dtype=jnp.float64).reshape((-1,))
        if int(weights.shape[0]) != total:
            raise ValueError("overlap_weights must have one entry per system row")
        if bool(jnp.any(weights <= 0.0)):
            raise ValueError("overlap_weights entries must be positive")

    return NativePaddedIndexedBlockFactor(
        block_inverses=inverses,
        block_indices=indices,
        block_mask=mask,
        overlap_weights=weights,
        total_size=total,
        normalize_overlap=bool(normalize_overlap),
        damping=float(damping),
    )


def build_native_padded_indexed_block_factor_from_matrix(
    matrix: jax.Array,
    *,
    block_indices: jax.Array,
    block_mask: jax.Array | None = None,
    regularization: float = 0.0,
    normalize_overlap: bool = True,
    damping: float = 1.0,
) -> NativePaddedIndexedBlockFactor:
    """Extract and invert padded indexed blocks from a dense matrix in JAX.

    Invalid padded rows/columns are replaced by identity rows before inversion.
    This keeps the factor shape static while making variable-size local solves
    safe for JIT and GPU execution.
    """

    mat = jnp.asarray(matrix, dtype=jnp.float64)
    if mat.ndim != 2 or int(mat.shape[0]) != int(mat.shape[1]):
        raise ValueError("matrix must be square")
    indices = jnp.asarray(block_indices, dtype=jnp.int32)
    if indices.ndim != 2:
        raise ValueError("block_indices must have shape (n_blocks, block_size)")
    if block_mask is None:
        mask = jnp.ones(indices.shape, dtype=bool)
    else:
        mask = jnp.asarray(block_mask, dtype=bool)
        if mask.shape != indices.shape:
            raise ValueError("block_mask must have shape (n_blocks, block_size)")
    total = int(mat.shape[0])
    # Validate index bounds and derive overlap weights before the dense gathers.
    probe = build_native_padded_indexed_block_factor(
        block_inverses=jnp.tile(jnp.eye(int(indices.shape[1]), dtype=mat.dtype)[None, :, :], (int(indices.shape[0]), 1, 1)),
        block_indices=indices,
        block_mask=mask,
        total_size=total,
        normalize_overlap=normalize_overlap,
        damping=damping,
    )
    safe_indices = jnp.where(mask, indices, 0)
    blocks = mat[safe_indices[:, :, None], safe_indices[:, None, :]]
    row_mask = mask[:, :, None]
    col_mask = mask[:, None, :]
    block_size = int(indices.shape[1])
    eye = jnp.eye(block_size, dtype=mat.dtype)[None, :, :]
    valid_blocks = jnp.where(row_mask & col_mask, blocks, 0.0)
    invalid_identity = eye * (~mask)[:, :, None]
    block_scales = jnp.maximum(
        jnp.max(jnp.sum(jnp.abs(valid_blocks), axis=2), axis=1),
        jnp.asarray(1.0, dtype=mat.dtype),
    )
    reg = jnp.asarray(abs(float(regularization)), dtype=mat.dtype) * block_scales
    valid_identity = eye * mask[:, :, None]
    regularized_blocks = valid_blocks + invalid_identity + reg[:, None, None] * valid_identity
    block_inverses = jax.vmap(jnp.linalg.inv)(regularized_blocks)
    return NativePaddedIndexedBlockFactor(
        block_inverses=block_inverses,
        block_indices=probe.block_indices,
        block_mask=probe.block_mask,
        overlap_weights=probe.overlap_weights,
        total_size=probe.total_size,
        normalize_overlap=probe.normalize_overlap,
        damping=probe.damping,
    )


def apply_native_padded_indexed_block_factor(
    factor: NativePaddedIndexedBlockFactor,
    rhs: jax.Array,
) -> jax.Array:
    """Apply an indexed block factor to one vector or a matrix of RHS columns."""

    arr = jnp.asarray(rhs, dtype=jnp.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr_2d = arr[:, None]
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError("rhs must be a vector or a two-dimensional array")
    total = int(factor.total_size)
    if int(arr_2d.shape[0]) != total:
        raise ValueError("rhs leading dimension does not match the factor size")

    safe_indices = jnp.where(factor.block_mask, factor.block_indices, 0)
    gathered = arr_2d[safe_indices] * factor.block_mask[:, :, None]
    solved = jnp.einsum("bij,bjk->bik", factor.block_inverses, gathered)
    solved = solved * factor.block_mask[:, :, None]
    flat_indices = safe_indices.reshape((-1,))
    flat_solved = solved.reshape((-1, int(arr_2d.shape[1])))
    out = jnp.zeros((total, int(arr_2d.shape[1])), dtype=arr_2d.dtype).at[flat_indices, :].add(flat_solved)
    if bool(factor.normalize_overlap):
        out = out / factor.overlap_weights[:, None]
    out = jnp.asarray(float(factor.damping), dtype=arr_2d.dtype) * out
    if len(original_shape) == 1:
        return out[:, 0]
    return out


def build_dense_block_jacobi(
    matrix: jax.Array,
    *,
    block_size: int,
    regularization: float = 0.0,
) -> NativeDenseBlockJacobi:
    """Build a JAX-native block-Jacobi inverse from a dense matrix.

    The matrix is padded to a multiple of ``block_size``.  Padding entries are an
    identity block, so applying the factor and truncating back to the original
    size is well defined for non-multiple system sizes.
    """

    mat = jnp.asarray(matrix, dtype=jnp.float64)
    if mat.ndim != 2 or int(mat.shape[0]) != int(mat.shape[1]):
        raise ValueError("block-Jacobi matrix must be square")
    if int(block_size) <= 0:
        raise ValueError("block_size must be positive")
    n = int(mat.shape[0])
    b = int(block_size)
    n_blocks = (n + b - 1) // b
    padded = n_blocks * b
    padded_mat = jnp.eye(padded, dtype=mat.dtype)
    padded_mat = padded_mat.at[:n, :n].set(mat)
    blocks = padded_mat.reshape(n_blocks, b, n_blocks, b)
    diag_blocks = jnp.swapaxes(blocks, 1, 2)[jnp.arange(n_blocks), jnp.arange(n_blocks)]
    diag_blocks = jax.vmap(_regularized_matrix, in_axes=(0, None))(diag_blocks, float(regularization))
    block_inverses = jax.vmap(jnp.linalg.inv)(diag_blocks)
    return NativeDenseBlockJacobi(
        block_inverses=block_inverses,
        block_size=b,
        original_size=n,
        padded_size=padded,
        regularization=float(regularization),
    )


def apply_dense_block_jacobi(factor: NativeDenseBlockJacobi, rhs: jax.Array) -> jax.Array:
    """Apply a dense block-Jacobi factor to one vector or a matrix of RHS columns."""

    arr = jnp.asarray(rhs, dtype=jnp.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr_2d = arr[:, None]
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError("rhs must be a vector or a two-dimensional array")
    if int(arr_2d.shape[0]) != int(factor.original_size):
        raise ValueError("rhs leading dimension does not match the factor size")
    pad_rows = int(factor.padded_size) - int(factor.original_size)
    if pad_rows:
        arr_2d = jnp.pad(arr_2d, ((0, pad_rows), (0, 0)))
    b = int(factor.block_size)
    n_blocks = int(factor.block_inverses.shape[0])
    rhs_blocks = arr_2d.reshape(n_blocks, b, int(arr_2d.shape[1]))
    out_blocks = jnp.einsum("bij,bjk->bik", factor.block_inverses, rhs_blocks)
    out = out_blocks.reshape(int(factor.padded_size), int(arr_2d.shape[1]))[: int(factor.original_size), :]
    if len(original_shape) == 1:
        return out[:, 0]
    return out


def build_two_field_schur_factor(
    a_ff: jax.Array,
    a_fc: jax.Array,
    a_cf: jax.Array,
    a_cc: jax.Array,
    *,
    regularization: float = 0.0,
) -> NativeTwoFieldSchurFactor:
    """Build a dense exact two-field Schur factor in pure JAX.

    For a block system

    ``[[A_ff, A_fc], [A_cf, A_cc]] [x_f, x_c]^T = [b_f, b_c]^T``,

    the stored factor applies ``A_ff^{-1}`` and
    ``S^{-1}``, where ``S = A_cc - A_cf A_ff^{-1} A_fc``.
    """

    aff = jnp.asarray(a_ff, dtype=jnp.float64)
    afc = jnp.asarray(a_fc, dtype=jnp.float64)
    acf = jnp.asarray(a_cf, dtype=jnp.float64)
    acc = jnp.asarray(a_cc, dtype=jnp.float64)
    if aff.ndim != 2 or int(aff.shape[0]) != int(aff.shape[1]):
        raise ValueError("a_ff must be square")
    if acc.ndim != 2 or int(acc.shape[0]) != int(acc.shape[1]):
        raise ValueError("a_cc must be square")
    f_size = int(aff.shape[0])
    c_size = int(acc.shape[0])
    if afc.shape != (f_size, c_size) or acf.shape != (c_size, f_size):
        raise ValueError("off-diagonal block shapes are inconsistent")
    aff_reg = _regularized_matrix(aff, regularization)
    a_ff_inv = jnp.linalg.inv(aff_reg)
    schur = acc - acf @ a_ff_inv @ afc
    schur_reg = _regularized_matrix(schur, regularization)
    schur_inv = jnp.linalg.inv(schur_reg)
    return NativeTwoFieldSchurFactor(
        a_ff_inv=a_ff_inv,
        a_fc=afc,
        a_cf=acf,
        schur_inv=schur_inv,
        f_size=f_size,
        c_size=c_size,
        regularization=float(regularization),
    )


def apply_two_field_schur(factor: NativeTwoFieldSchurFactor, rhs: jax.Array) -> jax.Array:
    """Apply the exact two-field Schur factor to one or more RHS columns."""

    arr = jnp.asarray(rhs, dtype=jnp.float64)
    original_shape = arr.shape
    if arr.ndim == 1:
        arr_2d = arr[:, None]
    elif arr.ndim == 2:
        arr_2d = arr
    else:
        raise ValueError("rhs must be a vector or a two-dimensional array")
    expected = int(factor.f_size + factor.c_size)
    if int(arr_2d.shape[0]) != expected:
        raise ValueError("rhs leading dimension does not match the factor size")
    b_f = arr_2d[: int(factor.f_size), :]
    b_c = arr_2d[int(factor.f_size) :, :]
    y_f0 = factor.a_ff_inv @ b_f
    schur_rhs = b_c - factor.a_cf @ y_f0
    y_c = factor.schur_inv @ schur_rhs
    y_f = y_f0 - factor.a_ff_inv @ (factor.a_fc @ y_c)
    out = jnp.concatenate([y_f, y_c], axis=0)
    if len(original_shape) == 1:
        return out[:, 0]
    return out
