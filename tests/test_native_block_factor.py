from __future__ import annotations

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from sfincs_jax.solvers.native_block_factor import (  # noqa: E402
    apply_dense_block_jacobi,
    apply_native_padded_indexed_block_factor,
    apply_two_field_schur,
    build_dense_block_jacobi,
    build_native_padded_indexed_block_factor,
    build_native_padded_indexed_block_factor_from_matrix,
    build_two_field_schur_factor,
)


def test_two_field_schur_matches_dense_solve_and_jit() -> None:
    a_ff = jnp.array([[4.0, 0.3], [-0.2, 3.2]])
    a_fc = jnp.array([[0.4, -0.1, 0.2], [0.0, 0.3, -0.2]])
    a_cf = jnp.array([[0.2, -0.3], [0.1, 0.4], [-0.2, 0.1]])
    a_cc = jnp.array([[2.5, 0.2, -0.1], [0.0, 2.0, 0.3], [0.1, -0.2, 2.8]])
    full = jnp.block([[a_ff, a_fc], [a_cf, a_cc]])
    rhs = jnp.array([1.0, -2.0, 0.5, 1.5, -0.25])
    rhs_cols = jnp.stack([rhs, rhs * 0.25 + 0.1], axis=1)

    factor = build_two_field_schur_factor(a_ff, a_fc, a_cf, a_cc)

    np.testing.assert_allclose(
        apply_two_field_schur(factor, rhs),
        jnp.linalg.solve(full, rhs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        apply_two_field_schur(factor, rhs_cols),
        jnp.linalg.solve(full, rhs_cols),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(lambda vec: apply_two_field_schur(factor, vec))(rhs),
        jnp.linalg.solve(full, rhs),
        rtol=1e-12,
        atol=1e-12,
    )


def test_two_field_schur_grad_is_finite() -> None:
    base_ff = jnp.array([[3.0, 0.2], [0.1, 2.5]])
    delta_ff = jnp.array([[0.7, -0.1], [0.0, 0.2]])
    a_fc = jnp.array([[0.3], [-0.2]])
    a_cf = jnp.array([[0.1, 0.4]])
    a_cc = jnp.array([[1.8]])
    rhs = jnp.array([1.0, -0.5, 0.25])

    def loss(scale: jax.Array) -> jax.Array:
        factor = build_two_field_schur_factor(
            base_ff + scale * delta_ff,
            a_fc,
            a_cf,
            a_cc,
            regularization=1e-12,
        )
        solution = apply_two_field_schur(factor, rhs)
        return jnp.sum(solution**2)

    value = loss(jnp.asarray(0.3))
    gradient = jax.grad(loss)(jnp.asarray(0.3))

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_dense_block_jacobi_matches_block_diagonal_solve_and_jit() -> None:
    block0 = jnp.array([[4.0, 0.5], [-0.25, 3.0]])
    block1 = jnp.array([[2.0, -0.2], [0.3, 2.5]])
    block2 = jnp.array([[7.0]])
    matrix = jnp.block(
        [
            [block0, jnp.zeros((2, 2)), jnp.zeros((2, 1))],
            [jnp.zeros((2, 2)), block1, jnp.zeros((2, 1))],
            [jnp.zeros((1, 2)), jnp.zeros((1, 2)), block2],
        ]
    )
    rhs = jnp.array([1.0, -0.5, 0.25, 2.0, -3.0])
    rhs_cols = jnp.stack([rhs, 0.5 * rhs - 0.1], axis=1)

    factor = build_dense_block_jacobi(matrix, block_size=2)

    np.testing.assert_allclose(
        apply_dense_block_jacobi(factor, rhs),
        jnp.linalg.solve(matrix, rhs),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        apply_dense_block_jacobi(factor, rhs_cols),
        jnp.linalg.solve(matrix, rhs_cols),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(lambda vec: apply_dense_block_jacobi(factor, vec))(rhs),
        jnp.linalg.solve(matrix, rhs),
        rtol=1e-12,
        atol=1e-12,
    )


def test_dense_block_jacobi_rejects_bad_shapes() -> None:
    with pytest.raises(ValueError, match="square"):
        build_dense_block_jacobi(jnp.ones((2, 3)), block_size=2)
    with pytest.raises(ValueError, match="positive"):
        build_dense_block_jacobi(jnp.eye(2), block_size=0)
    factor = build_dense_block_jacobi(jnp.eye(3), block_size=2)
    with pytest.raises(ValueError, match="leading dimension"):
        apply_dense_block_jacobi(factor, jnp.ones(4))


def test_padded_indexed_block_factor_matches_variable_block_solve_and_jit() -> None:
    block0 = jnp.array([[4.0, 0.5], [-0.25, 3.0]])
    block1 = jnp.array([[2.0, -0.2, 0.1], [0.3, 2.5, -0.4], [0.0, 0.2, 1.8]])
    matrix = jnp.block(
        [
            [block0, jnp.zeros((2, 3))],
            [jnp.zeros((3, 2)), block1],
        ]
    )
    indices = jnp.array([[0, 1, 0], [2, 3, 4]], dtype=jnp.int32)
    mask = jnp.array([[True, True, False], [True, True, True]])
    rhs = jnp.array([1.0, -0.5, 0.25, 2.0, -3.0])
    rhs_cols = jnp.stack([rhs, 0.5 * rhs + 0.2], axis=1)

    factor = build_native_padded_indexed_block_factor_from_matrix(
        matrix,
        block_indices=indices,
        block_mask=mask,
    )

    expected = jnp.linalg.solve(matrix, rhs)
    expected_cols = jnp.linalg.solve(matrix, rhs_cols)
    np.testing.assert_allclose(
        apply_native_padded_indexed_block_factor(factor, rhs),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        apply_native_padded_indexed_block_factor(factor, rhs_cols),
        expected_cols,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        jax.jit(lambda vec: apply_native_padded_indexed_block_factor(factor, vec))(rhs),
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


def test_padded_indexed_block_factor_normalizes_overlapping_blocks() -> None:
    indices = jnp.array([[0, 1], [1, 2]], dtype=jnp.int32)
    inverses = jnp.tile(jnp.eye(2, dtype=jnp.float64)[None, :, :], (2, 1, 1))
    rhs = jnp.array([2.0, 4.0, 6.0])

    normalized = build_native_padded_indexed_block_factor(
        block_inverses=inverses,
        block_indices=indices,
        total_size=3,
        normalize_overlap=True,
    )
    unnormalized = build_native_padded_indexed_block_factor(
        block_inverses=inverses,
        block_indices=indices,
        total_size=3,
        normalize_overlap=False,
    )

    np.testing.assert_allclose(
        apply_native_padded_indexed_block_factor(normalized, rhs),
        rhs,
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        apply_native_padded_indexed_block_factor(unnormalized, rhs),
        jnp.array([2.0, 8.0, 6.0]),
        rtol=1e-12,
        atol=1e-12,
    )


def test_padded_indexed_block_factor_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="block_inverses"):
        build_native_padded_indexed_block_factor(
            block_inverses=jnp.ones((2, 2)),
            block_indices=jnp.ones((1, 2), dtype=jnp.int32),
            total_size=2,
        )
    with pytest.raises(ValueError, match="block_indices"):
        build_native_padded_indexed_block_factor(
            block_inverses=jnp.ones((1, 2, 2)),
            block_indices=jnp.ones((2, 2), dtype=jnp.int32),
            total_size=2,
        )
    with pytest.raises(ValueError, match="block_mask"):
        build_native_padded_indexed_block_factor(
            block_inverses=jnp.ones((1, 2, 2)),
            block_indices=jnp.ones((1, 2), dtype=jnp.int32),
            block_mask=jnp.ones((2, 2), dtype=bool),
            total_size=2,
        )
    with pytest.raises(ValueError, match="inside"):
        build_native_padded_indexed_block_factor(
            block_inverses=jnp.eye(2, dtype=jnp.float64)[None, :, :],
            block_indices=jnp.array([[0, 3]], dtype=jnp.int32),
            total_size=2,
        )
    with pytest.raises(ValueError, match="square"):
        build_native_padded_indexed_block_factor_from_matrix(
            jnp.ones((2, 3)),
            block_indices=jnp.ones((1, 2), dtype=jnp.int32),
        )
    factor = build_native_padded_indexed_block_factor(
        block_inverses=jnp.eye(2, dtype=jnp.float64)[None, :, :],
        block_indices=jnp.array([[0, 1]], dtype=jnp.int32),
        total_size=2,
    )
    with pytest.raises(ValueError, match="leading dimension"):
        apply_native_padded_indexed_block_factor(factor, jnp.ones(3))


def test_two_field_schur_rejects_inconsistent_blocks() -> None:
    with pytest.raises(ValueError, match="off-diagonal"):
        build_two_field_schur_factor(
            jnp.eye(2),
            jnp.ones((2, 2)),
            jnp.ones((1, 2)),
            jnp.eye(1),
        )
    factor = build_two_field_schur_factor(
        jnp.eye(2),
        jnp.ones((2, 1)) * 0.1,
        jnp.ones((1, 2)) * 0.2,
        jnp.eye(1),
    )
    with pytest.raises(ValueError, match="leading dimension"):
        apply_two_field_schur(factor, jnp.ones(4))
