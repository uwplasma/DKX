from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from sfincs_jax.structured_velocity import (
    apply_block_tridiagonal,
    block_tridiagonal_to_dense,
    factor_block_tridiagonal,
)


def _make_well_conditioned_blocks(seed: int = 0):
    rng = np.random.default_rng(seed)
    nblocks = 4
    block_size = 3
    diagonal = np.stack(
        [3.5 * np.eye(block_size) + 0.05 * rng.standard_normal((block_size, block_size)) for _ in range(nblocks)],
        axis=0,
    )
    lower = 0.08 * rng.standard_normal((nblocks - 1, block_size, block_size))
    upper = 0.08 * rng.standard_normal((nblocks - 1, block_size, block_size))
    return diagonal, lower, upper


def test_block_tridiagonal_apply_matches_dense() -> None:
    diagonal, lower, upper = _make_well_conditioned_blocks(seed=1)
    dense = np.asarray(block_tridiagonal_to_dense(diagonal, lower, upper))
    x = np.random.default_rng(2).standard_normal(dense.shape[0])

    y = np.asarray(apply_block_tridiagonal(factor_block_tridiagonal(diagonal, lower, upper), jnp.asarray(x)))

    np.testing.assert_allclose(y, dense @ x, rtol=0.0, atol=1e-11)


def test_block_tridiagonal_solve_matches_dense_and_reuses_factor() -> None:
    diagonal, lower, upper = _make_well_conditioned_blocks(seed=3)
    dense = np.asarray(block_tridiagonal_to_dense(diagonal, lower, upper))
    factor = factor_block_tridiagonal(diagonal, lower, upper)

    rhs1 = np.random.default_rng(4).standard_normal(dense.shape[0])
    rhs2 = np.random.default_rng(5).standard_normal(dense.shape[0])

    sol1 = np.asarray(factor.solve(jnp.asarray(rhs1)))
    sol2 = np.asarray(factor.solve(jnp.asarray(rhs2)))
    sol_batch = np.asarray(factor.solve(jnp.asarray(np.stack([rhs1, rhs2], axis=0))))

    np.testing.assert_allclose(dense @ sol1, rhs1, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(dense @ sol2, rhs2, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(sol1, np.linalg.solve(dense, rhs1), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(sol2, np.linalg.solve(dense, rhs2), rtol=0.0, atol=1e-10)
    np.testing.assert_allclose(sol_batch[0], sol1, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sol_batch[1], sol2, rtol=0.0, atol=1e-12)


def test_block_tridiagonal_reverse_handles_singular_leading_block() -> None:
    diagonal = np.array([[[0.0]], [[3.0]]], dtype=np.float64)
    lower = np.array([[[1.0]]], dtype=np.float64)
    upper = np.array([[[1.0]]], dtype=np.float64)
    dense = np.asarray(block_tridiagonal_to_dense(diagonal, lower, upper))
    rhs = np.array([2.0, 5.0], dtype=np.float64)

    factor = factor_block_tridiagonal(diagonal, lower, upper, reverse=True)
    sol = np.asarray(factor.solve(jnp.asarray(rhs)))

    np.testing.assert_allclose(dense @ sol, rhs, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(sol, np.linalg.solve(dense, rhs), rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(sol))


def test_block_tridiagonal_block_shape_roundtrip() -> None:
    diagonal, lower, upper = _make_well_conditioned_blocks(seed=6)
    factor = factor_block_tridiagonal(diagonal, lower, upper)
    rhs = np.random.default_rng(7).standard_normal(diagonal.shape[0] * diagonal.shape[1]).reshape(diagonal.shape[0], diagonal.shape[1])
    sol = np.asarray(factor.solve(jnp.asarray(rhs)))

    assert sol.shape == rhs.shape
    applied = np.asarray(factor.apply(jnp.asarray(sol))).reshape(-1)
    np.testing.assert_allclose(
        applied,
        np.asarray(block_tridiagonal_to_dense(diagonal, lower, upper)) @ sol.reshape(-1),
        rtol=0.0,
        atol=1e-11,
    )


def test_block_tridiagonal_factor_is_pytree_roundtrip() -> None:
    diagonal, lower, upper = _make_well_conditioned_blocks(seed=8)
    factor = factor_block_tridiagonal(diagonal, lower, upper)
    leaves, treedef = jax.tree_util.tree_flatten(factor)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    rhs = np.random.default_rng(9).standard_normal(diagonal.shape[0] * diagonal.shape[1])
    np.testing.assert_allclose(
        np.asarray(rebuilt.solve(jnp.asarray(rhs))),
        np.asarray(factor.solve(jnp.asarray(rhs))),
        rtol=0.0,
        atol=1e-12,
    )
