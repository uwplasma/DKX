from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

__all__ = [
    "BlockTridiagonalFactorization",
    "apply_block_tridiagonal",
    "block_tridiagonal_to_dense",
    "factor_block_tridiagonal",
    "solve_block_tridiagonal",
]


@jtu.register_pytree_node_class
@dataclass(frozen=True)
class BlockTridiagonalFactorization:
    """Reusable factorization of a block-tridiagonal linear system.

    The blocks are stored in the orientation used for factorization. If
    ``reverse=True`` was requested, the blocks are flipped and the upper/lower
    couplings are swapped before factoring. That lets callers handle a singular
    leading block by factoring from the opposite end while reusing the same
    solve API.
    """

    diagonal: jnp.ndarray
    lower_diagonal: jnp.ndarray
    upper_diagonal: jnp.ndarray
    elimination_blocks: jnp.ndarray
    lu_blocks: jnp.ndarray
    pivot_blocks: jnp.ndarray
    reverse: bool

    def tree_flatten(self):
        children = (
            self.diagonal,
            self.lower_diagonal,
            self.upper_diagonal,
            self.elimination_blocks,
            self.lu_blocks,
            self.pivot_blocks,
        )
        aux = (self.reverse,)
        return children, aux

    @classmethod
    def tree_unflatten(cls, aux, children):
        (reverse,) = aux
        diagonal, lower_diagonal, upper_diagonal, elimination_blocks, lu_blocks, pivot_blocks = children
        return cls(
            diagonal=diagonal,
            lower_diagonal=lower_diagonal,
            upper_diagonal=upper_diagonal,
            elimination_blocks=elimination_blocks,
            lu_blocks=lu_blocks,
            pivot_blocks=pivot_blocks,
            reverse=reverse,
        )

    @property
    def nblocks(self) -> int:
        return int(self.diagonal.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.diagonal.shape[-1])

    def solve(self, rhs: jnp.ndarray) -> jnp.ndarray:
        return solve_block_tridiagonal(self, rhs)

    def apply(self, x: jnp.ndarray) -> jnp.ndarray:
        return apply_block_tridiagonal(self, x)


def _validate_blocks(diagonal: jnp.ndarray, lower_diagonal: jnp.ndarray, upper_diagonal: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    diagonal = jnp.asarray(diagonal)
    lower_diagonal = jnp.asarray(lower_diagonal)
    upper_diagonal = jnp.asarray(upper_diagonal)
    if diagonal.ndim != 3:
        raise ValueError(f"diagonal must have shape (nblocks, block_size, block_size), got {diagonal.shape}")
    if diagonal.shape[1] != diagonal.shape[2]:
        raise ValueError(f"diagonal blocks must be square, got {diagonal.shape}")
    nblocks = int(diagonal.shape[0])
    block_size = int(diagonal.shape[1])
    if lower_diagonal.shape != (max(nblocks - 1, 0), block_size, block_size):
        raise ValueError(
            "lower_diagonal must have shape "
            f"({max(nblocks - 1, 0)}, {block_size}, {block_size}), got {lower_diagonal.shape}"
        )
    if upper_diagonal.shape != (max(nblocks - 1, 0), block_size, block_size):
        raise ValueError(
            "upper_diagonal must have shape "
            f"({max(nblocks - 1, 0)}, {block_size}, {block_size}), got {upper_diagonal.shape}"
        )
    return diagonal, lower_diagonal, upper_diagonal


def _orient_blocks(
    diagonal: jnp.ndarray,
    lower_diagonal: jnp.ndarray,
    upper_diagonal: jnp.ndarray,
    *,
    reverse: bool,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    if not reverse:
        return diagonal, lower_diagonal, upper_diagonal
    return (
        jnp.flip(diagonal, axis=0),
        jnp.flip(upper_diagonal, axis=0),
        jnp.flip(lower_diagonal, axis=0),
    )


def factor_block_tridiagonal(
    diagonal: jnp.ndarray,
    lower_diagonal: jnp.ndarray,
    upper_diagonal: jnp.ndarray,
    *,
    reverse: bool = False,
) -> BlockTridiagonalFactorization:
    r"""Factor a block-tridiagonal matrix for later repeated solves.

    The matrix is defined by the block arrays

    .. math::

       A = \begin{bmatrix}
       D_0 & U_0 \\
       L_0 & D_1 & U_1 \\
           & \ddots & \ddots & \ddots \\
           &        & L_{n-2} & D_{n-1}
       \end{bmatrix}.

    The factorization stores the LU factors of the block-Schur complements and
    the elimination blocks :math:`C_k = S_k^{-1} U_k`, which makes each later
    solve a pair of forward/back substitutions.
    """

    diagonal, lower_diagonal, upper_diagonal = _validate_blocks(diagonal, lower_diagonal, upper_diagonal)
    diagonal, lower_diagonal, upper_diagonal = _orient_blocks(
        diagonal, lower_diagonal, upper_diagonal, reverse=bool(reverse)
    )

    nblocks = int(diagonal.shape[0])
    block_size = int(diagonal.shape[1])
    zero_block = jnp.zeros((block_size, block_size), dtype=diagonal.dtype)

    def factor_step(prev_elim: jnp.ndarray, inputs: tuple[int, jnp.ndarray]) -> tuple[jnp.ndarray, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
        k, diag_k = inputs

        def _schur_zero(_):
            return diag_k

        def _schur_nonzero(_):
            return diag_k - lower_diagonal[k - 1] @ prev_elim

        schur = jax.lax.cond(k == 0, _schur_zero, _schur_nonzero, operand=None)
        lu_k = jax.scipy.linalg.lu_factor(schur)
        def _elim_nonzero(_):
            return jax.scipy.linalg.lu_solve(lu_k, upper_diagonal[k])

        def _elim_zero(_):
            return zero_block

        elim_k = jax.lax.cond(k < nblocks - 1, _elim_nonzero, _elim_zero, operand=None)
        return elim_k, (elim_k, lu_k[0], lu_k[1])

    init_elim = zero_block
    indices = jnp.arange(nblocks)
    _, (elimination_blocks, lu_blocks, pivot_blocks) = jax.lax.scan(
        factor_step,
        init_elim,
        (indices, diagonal),
    )

    return BlockTridiagonalFactorization(
        diagonal=diagonal,
        lower_diagonal=lower_diagonal,
        upper_diagonal=upper_diagonal,
        elimination_blocks=elimination_blocks,
        lu_blocks=lu_blocks,
        pivot_blocks=pivot_blocks,
        reverse=bool(reverse),
    )


def _rhs_to_blocks(rhs: jnp.ndarray, *, nblocks: int, block_size: int, dtype) -> tuple[jnp.ndarray, tuple[int, ...]]:
    rhs = jnp.asarray(rhs, dtype=dtype)
    if rhs.ndim == 1:
        if rhs.size != nblocks * block_size:
            raise ValueError(
                f"rhs must have length {nblocks * block_size}, got {rhs.size}"
            )
        return rhs.reshape((1, nblocks, block_size)), rhs.shape
    if rhs.shape[-2:] == (nblocks, block_size):
        return rhs.reshape((-1, nblocks, block_size)), rhs.shape
    if rhs.shape[-1] == nblocks * block_size:
        reshaped = rhs.reshape(rhs.shape[:-1] + (nblocks, block_size))
        return reshaped.reshape((-1, nblocks, block_size)), rhs.shape
    raise ValueError(
        "rhs must be a flat vector of length nblocks*block_size, a matrix of shape "
        f"(nblocks, block_size), or a batch with trailing dimension nblocks*block_size; got {rhs.shape}"
    )


def _blocks_to_rhs(blocks: jnp.ndarray, *, original_shape: tuple[int, ...], nblocks: int, block_size: int) -> jnp.ndarray:
    if len(original_shape) == 1:
        return blocks.reshape((nblocks * block_size,))
    return blocks.reshape(original_shape)


def _solve_single_rhs(factor: BlockTridiagonalFactorization, rhs_blocks: jnp.ndarray) -> jnp.ndarray:
    if factor.reverse:
        rhs_blocks = jnp.flip(rhs_blocks, axis=0)

    lower_pad = jnp.concatenate(
        [jnp.zeros((1, factor.block_size, factor.block_size), dtype=factor.diagonal.dtype), factor.lower_diagonal],
        axis=0,
    )

    def forward_step(prev_sol: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        lu_k, piv_k, lower_k, rhs_k = inputs
        rhs_eff = rhs_k - lower_k @ prev_sol
        sol_k = jax.scipy.linalg.lu_solve((lu_k, piv_k), rhs_eff)
        return sol_k, sol_k

    init = jnp.zeros((factor.block_size,), dtype=factor.diagonal.dtype)
    _, y_blocks = jax.lax.scan(
        forward_step,
        init,
        (factor.lu_blocks, factor.pivot_blocks, lower_pad, rhs_blocks),
    )

    def back_step(next_sol: jnp.ndarray, inputs: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
        elim_k, y_k = inputs
        sol_k = y_k - elim_k @ next_sol
        return sol_k, sol_k

    init_back = jnp.zeros((factor.block_size,), dtype=factor.diagonal.dtype)
    _, x_blocks = jax.lax.scan(
        back_step,
        init_back,
        (factor.elimination_blocks, y_blocks),
        reverse=True,
    )

    if factor.reverse:
        x_blocks = jnp.flip(x_blocks, axis=0)
    return x_blocks


def solve_block_tridiagonal(factor: BlockTridiagonalFactorization, rhs: jnp.ndarray) -> jnp.ndarray:
    """Solve a previously factored block-tridiagonal system.

    The solver accepts a flat vector, a block-shaped array of shape
    ``(nblocks, block_size)``, or a batch of flat vectors in the trailing axis.
    Repeated calls reuse the same factorization object.
    """

    rhs_blocks, original_shape = _rhs_to_blocks(rhs, nblocks=factor.nblocks, block_size=factor.block_size, dtype=factor.diagonal.dtype)
    if rhs_blocks.shape[0] == 1:
        sol_blocks = _solve_single_rhs(factor, rhs_blocks[0])[None, ...]
    else:
        sol_blocks = jax.vmap(lambda b: _solve_single_rhs(factor, b))(rhs_blocks)
    return _blocks_to_rhs(sol_blocks, original_shape=original_shape, nblocks=factor.nblocks, block_size=factor.block_size)


def apply_block_tridiagonal(factor: BlockTridiagonalFactorization, x: jnp.ndarray) -> jnp.ndarray:
    """Apply the block-tridiagonal operator represented by ``factor``."""

    x_blocks, original_shape = _rhs_to_blocks(x, nblocks=factor.nblocks, block_size=factor.block_size, dtype=factor.diagonal.dtype)
    if factor.reverse:
        x_blocks = jnp.flip(x_blocks, axis=1)

    def apply_single(vec_blocks: jnp.ndarray) -> jnp.ndarray:
        y = jnp.einsum("kij,kj->ki", factor.diagonal, vec_blocks)
        if factor.nblocks > 1:
            y = y.at[:-1].add(jnp.einsum("kij,kj->ki", factor.upper_diagonal, vec_blocks[1:]))
            y = y.at[1:].add(jnp.einsum("kij,kj->ki", factor.lower_diagonal, vec_blocks[:-1]))
        return y

    if x_blocks.shape[0] == 1:
        y_blocks = apply_single(x_blocks[0])[None, ...]
    else:
        y_blocks = jax.vmap(apply_single)(x_blocks)
    if factor.reverse:
        y_blocks = jnp.flip(y_blocks, axis=1)
    return _blocks_to_rhs(y_blocks, original_shape=original_shape, nblocks=factor.nblocks, block_size=factor.block_size)


def block_tridiagonal_to_dense(
    diagonal: jnp.ndarray,
    lower_diagonal: jnp.ndarray,
    upper_diagonal: jnp.ndarray,
) -> jnp.ndarray:
    """Materialize a block-tridiagonal matrix for test comparisons."""

    diagonal, lower_diagonal, upper_diagonal = _validate_blocks(diagonal, lower_diagonal, upper_diagonal)
    nblocks = int(diagonal.shape[0])
    block_size = int(diagonal.shape[1])
    size = nblocks * block_size
    dense = jnp.zeros((size, size), dtype=diagonal.dtype)
    for k in range(nblocks):
        rows = slice(k * block_size, (k + 1) * block_size)
        cols = slice(k * block_size, (k + 1) * block_size)
        dense = dense.at[rows, cols].set(diagonal[k])
        if k < nblocks - 1:
            dense = dense.at[rows, slice((k + 1) * block_size, (k + 2) * block_size)].set(upper_diagonal[k])
            dense = dense.at[slice((k + 1) * block_size, (k + 2) * block_size), cols].set(lower_diagonal[k])
    return dense
