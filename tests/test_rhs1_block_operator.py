from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.rhs1_block_operator import (
    RHS1ActiveBlockLayout,
    RHS1ActiveFieldSplitOrdering,
    RHS1BlockCOOBuilder,
    RHS1BlockCOOOperator,
    RHS1BlockLayout,
    RHS1BlockLinearOperator,
    RHS1GalerkinResidualCorrection,
    RHS1MatrixFreeGalerkinResidualCorrection,
    RHS1GroupedBlockDiagonalFactor,
    RHS1UniformBlockDiagonalFactor,
    preflight_rhs1_block_jacobi_candidate,
    preflight_rhs1_line_jacobi_candidate,
    probe_rhs1_block_jacobi_preconditioner,
    probe_rhs1_block_preconditioner,
)
from sfincs_jax.solver import fgmres_solve_with_residual


def _fake_op(
    *,
    n_species: int = 2,
    n_x: int = 3,
    n_xi: int = 4,
    n_theta: int = 5,
    n_zeta: int = 6,
    include_phi1: bool = True,
    constraint_scheme: int = 1,
) -> SimpleNamespace:
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    phi1_size = n_theta * n_zeta + 1 if include_phi1 else 0
    extra_size = 2 * n_species if constraint_scheme == 1 else 0
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=phi1_size,
        extra_size=extra_size,
        total_size=f_size + phi1_size + extra_size,
        constraint_scheme=constraint_scheme,
        include_phi1=include_phi1,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def test_rhs1_block_layout_decodes_v3_kinetic_ordering() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op())

    flat = layout.kinetic_flat_index(species=1, x=2, ell=3, theta=4, zeta=5)
    decoded = layout.decode_kinetic_indices([flat])

    assert layout.f_shape == (2, 3, 4, 5, 6)
    assert layout.component_sizes() == {
        "kinetic": 720,
        "phi1": 31,
        "extra": 4,
        "total": 755,
    }
    assert layout.f_slice == slice(0, 720)
    assert layout.phi1_field_slice == slice(720, 750)
    assert layout.phi1_lambda_slice == slice(750, 751)
    assert layout.extra_slice == slice(751, 755)
    np.testing.assert_array_equal(decoded.species, np.asarray([1]))
    np.testing.assert_array_equal(decoded.x, np.asarray([2]))
    np.testing.assert_array_equal(decoded.ell, np.asarray([3]))
    np.testing.assert_array_equal(decoded.theta, np.asarray([4]))
    np.testing.assert_array_equal(decoded.zeta, np.asarray([5]))


def test_rhs1_block_layout_rejects_invalid_indices() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op(include_phi1=False))

    with pytest.raises(IndexError):
        layout.kinetic_flat_index(species=2, x=0, ell=0, theta=0, zeta=0)
    with pytest.raises(ValueError):
        layout.decode_kinetic_indices([layout.f_size])


def test_rhs1_active_block_layout_detects_contiguous_constraint_tail() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op())
    active = np.concatenate(
        [
            np.asarray([0, 5, 11, layout.f_size - 1], dtype=np.int32),
            np.arange(layout.extra_slice.start, layout.extra_slice.stop, dtype=np.int32),
        ]
    )

    active_layout = RHS1ActiveBlockLayout.from_layout(layout, active)

    assert active_layout.active_size == 8
    assert active_layout.kinetic_count == 4
    assert active_layout.phi1_count == 0
    assert active_layout.extra_count == 4
    assert active_layout.has_contiguous_extra_tail is True
    np.testing.assert_array_equal(active_layout.active_kinetic_indices(), active[:4])


def test_rhs1_active_block_layout_rejects_duplicate_active_indices() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op())

    with pytest.raises(ValueError):
        RHS1ActiveBlockLayout.from_layout(layout, [0, 0, 1])


def test_active_field_split_ordering_maps_active_positions_and_dominant_kinetic_subset() -> None:
    layout = RHS1BlockLayout.from_operator(
        _fake_op(n_species=2, n_x=3, n_xi=4, n_theta=3, n_zeta=2, include_phi1=True)
    )
    full_indices = [
        layout.kinetic_flat_index(species=0, x=0, ell=0, theta=0, zeta=0),
        layout.kinetic_flat_index(species=0, x=0, ell=1, theta=1, zeta=1),
        layout.kinetic_flat_index(species=1, x=2, ell=3, theta=2, zeta=1),
        layout.phi1_field_slice.start,
        layout.extra_slice.start,
        layout.extra_slice.start + 1,
    ]
    ordering = RHS1ActiveFieldSplitOrdering.from_layout(layout, full_indices)

    assert ordering.active_size == len(full_indices)
    assert ordering.kinetic_size == 3
    assert ordering.phi1_size == 1
    assert ordering.extra_size == 2
    np.testing.assert_array_equal(
        ordering.active_positions_for_full_indices([full_indices[1], full_indices[-1]]),
        np.asarray([1, 5], dtype=np.int64),
    )
    dominant = ordering.dominant_kinetic_positions(x_count=1, ell_count=2, max_positions=4)
    np.testing.assert_array_equal(dominant, np.asarray([0, 1], dtype=np.int64))
    assert ordering.to_dict()["kind"] == "active_field_split_symbolic_ordering"


def test_active_field_split_ordering_rejects_bad_active_indices() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op(include_phi1=False))

    with pytest.raises(ValueError, match="duplicates"):
        RHS1ActiveFieldSplitOrdering.from_layout(layout, [0, 0])
    ordering = RHS1ActiveFieldSplitOrdering.from_layout(layout, [0, 2])
    with pytest.raises(ValueError, match="outside"):
        ordering.active_positions_for_full_indices([layout.total_size])


def test_rhs1_block_linear_operator_matvec_matmat_and_metadata() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op(n_species=1, n_x=1, n_xi=1, n_theta=2, n_zeta=2, include_phi1=False))
    matrix = jnp.asarray(np.diag(np.arange(1, layout.total_size + 1, dtype=np.float64)))
    op = RHS1BlockLinearOperator(
        layout=layout,
        matvec_fn=lambda x: matrix @ x,
        dtype=jnp.float64,
        name="diagonal_test",
    )

    x = jnp.ones((layout.total_size,), dtype=jnp.float64)
    y = op.matvec(x)
    yy = op.jitted_matvec()(x)
    batch = op.matmat(jnp.stack([x, 2.0 * x], axis=1))

    np.testing.assert_allclose(np.asarray(y), np.arange(1, layout.total_size + 1, dtype=np.float64))
    np.testing.assert_allclose(np.asarray(yy), np.asarray(y))
    np.testing.assert_allclose(np.asarray(batch[:, 0]), np.asarray(y))
    np.testing.assert_allclose(np.asarray(batch[:, 1]), 2.0 * np.asarray(y))
    assert op.to_dict()["shape"] == (layout.total_size, layout.total_size)


def test_rhs1_block_linear_operator_validates_shapes() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op(n_species=1, n_x=1, n_xi=1, n_theta=1, n_zeta=2, include_phi1=False))
    op = RHS1BlockLinearOperator(layout=layout, matvec_fn=lambda x: x)

    with pytest.raises(ValueError):
        op.matvec(jnp.ones((layout.total_size + 1,), dtype=jnp.float64))
    with pytest.raises(ValueError):
        op.matmat(jnp.ones((layout.total_size + 1, 2), dtype=jnp.float64))


def test_block_coo_operator_matches_dense_matvec_and_matmat() -> None:
    dense = jnp.asarray(
        [
            [2.0, 0.0, 0.5, -1.0],
            [1.0, 3.0, 0.0, 0.25],
            [0.0, 0.0, 4.0, 1.5],
            [-2.0, 0.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    x = jnp.asarray([1.0, -2.0, 3.0, 4.0], dtype=jnp.float64)
    x_batch = jnp.stack([x, -0.5 * x], axis=1)

    np.testing.assert_allclose(np.asarray(op.matvec(x)), np.asarray(dense @ x), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(op.matvec)(x)), np.asarray(dense @ x), rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(
        np.asarray(op.matmat(x_batch)),
        np.asarray(dense @ x_batch),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert op.to_dict()["nnz_blocks"] == 4


def test_block_coo_operator_groups_scalar_entries_without_dense_input() -> None:
    rows = np.asarray([0, 0, 1, 2, 2, 3, 3], dtype=np.int32)
    cols = np.asarray([0, 2, 3, 0, 2, 1, 3], dtype=np.int32)
    values = np.asarray([2.0, 0.5, 0.25, -1.0, 4.0, 1.5, 5.0], dtype=np.float64)
    op = RHS1BlockCOOOperator.from_scalar_coo_entries(
        row_indices=rows,
        col_indices=cols,
        values=values,
        shape=(4, 4),
        block_size=2,
    )
    dense = np.zeros((4, 4), dtype=np.float64)
    dense[rows, cols] += values
    x = jnp.asarray([1.0, -2.0, 3.0, 4.0], dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(op.to_dense_matrix()), dense, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(op.matvec(x)), dense @ np.asarray(x), rtol=1.0e-12, atol=1.0e-12)
    assert op.nnz_blocks == 4
    assert op.data_nbytes == 4 * 2 * 2 * 8


def test_block_coo_operator_projects_block_indices_in_requested_order() -> None:
    dense = np.arange(1, 37, dtype=np.float64).reshape((6, 6))
    dense[dense % 5 == 0] = 0.0
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.asarray(dense), block_size=2)

    projected = op.project_block_indices([2, 0])
    scalar_indices = np.asarray([4, 5, 0, 1], dtype=np.int64)
    expected = dense[np.ix_(scalar_indices, scalar_indices)]

    np.testing.assert_allclose(np.asarray(projected.to_dense_matrix()), expected)
    np.testing.assert_allclose(projected.to_scipy_csr_matrix().toarray(), expected)
    assert projected.shape == (4, 4)
    assert projected.n_block_rows == 2
    assert projected.n_block_cols == 2


def test_block_coo_builder_combines_scalar_entries_and_dense_blocks() -> None:
    builder = RHS1BlockCOOBuilder(shape=(4, 4), block_size=2, dtype=np.float64)
    builder.add_scalar_entries(
        row_indices=[0, 0, 3],
        col_indices=[0, 0, 3],
        values=[1.0, 2.0, 4.0],
    )
    builder.add_dense_block(0, 1, np.asarray([[0.5, 0.0], [0.0, -1.0]], dtype=np.float64))
    builder.add_dense_block(0, 1, np.asarray([[0.25, 0.0], [0.0, 0.5]], dtype=np.float64))
    op = builder.build()
    expected = np.asarray(
        [
            [3.0, 0.0, 0.75, 0.0],
            [0.0, 0.0, 0.0, -0.5],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0],
        ],
        dtype=np.float64,
    )

    np.testing.assert_allclose(np.asarray(op.to_dense_matrix()), expected)
    assert builder.nnz_blocks == 3
    assert builder.data_nbytes_estimate == 3 * 2 * 2 * 8


def test_block_coo_builder_drop_tolerance_eliminates_empty_blocks() -> None:
    builder = RHS1BlockCOOBuilder(shape=(4, 4), block_size=2, dtype=np.float64)
    builder.add_scalar_entries(row_indices=[0, 2], col_indices=[0, 2], values=[1.0e-12, 2.0], drop_tol=1.0e-10)
    op = builder.build(drop_tol=1.0e-10)

    assert op.nnz_blocks == 1
    np.testing.assert_allclose(
        np.asarray(op.to_dense_matrix()),
        np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )


def test_block_coo_builder_adds_tridiagonal_line_stencil_without_dense_assembly() -> None:
    builder = RHS1BlockCOOBuilder(shape=(8, 8), block_size=2, dtype=np.float64)
    diagonal = np.asarray(
        [
            [[4.0, 0.5], [0.25, 3.0]],
            [[5.0, -0.5], [1.0, 4.0]],
            [[6.0, 0.0], [-1.0, 5.0]],
        ],
        dtype=np.float64,
    )
    lower = np.asarray(
        [
            [[-0.5, 0.0], [0.25, -0.25]],
            [[-0.75, 0.1], [0.0, -0.5]],
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            [[0.4, -0.1], [0.0, 0.2]],
            [[0.5, 0.0], [-0.2, 0.3]],
        ],
        dtype=np.float64,
    )
    builder.add_tridiagonal_block_line(
        block_indices=[0, 1, 2],
        diagonal_blocks=diagonal,
        lower_blocks=lower,
        upper_blocks=upper,
    )
    op = builder.build()

    expected = np.zeros((8, 8), dtype=np.float64)
    for i in range(3):
        expected[2 * i : 2 * i + 2, 2 * i : 2 * i + 2] += diagonal[i]
    expected[2:4, 0:2] += lower[0]
    expected[4:6, 2:4] += lower[1]
    expected[0:2, 2:4] += upper[0]
    expected[2:4, 4:6] += upper[1]
    x = jnp.asarray([1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0], dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(op.to_dense_matrix()), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(op.matvec(x)), expected @ np.asarray(x), rtol=1.0e-12, atol=1.0e-12)
    assert op.nnz_blocks == 7


def test_block_coo_builder_tridiagonal_line_accepts_broadcast_couplings() -> None:
    builder = RHS1BlockCOOBuilder(shape=(6, 6), block_size=2, dtype=np.float64)
    diagonal = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64)
    offdiag = np.asarray([[-0.25, 0.0], [0.0, -0.5]], dtype=np.float64)

    builder.add_tridiagonal_block_line(
        block_indices=[0, 1, 2],
        diagonal_blocks=diagonal,
        lower_blocks=offdiag,
        upper_blocks=offdiag,
    )
    op = builder.build()

    assert op.nnz_blocks == 7
    np.testing.assert_allclose(
        np.asarray(op.to_dense_matrix()[0:2, 0:2]),
        diagonal,
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(op.to_dense_matrix()[2:4, 0:2]),
        offdiag,
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_block_coo_builder_tridiagonal_line_validates_inputs() -> None:
    builder = RHS1BlockCOOBuilder(shape=(4, 4), block_size=2, dtype=np.float64)

    with pytest.raises(ValueError):
        builder.add_tridiagonal_block_line(block_indices=[], diagonal_blocks=np.eye(2))
    with pytest.raises(ValueError):
        builder.add_tridiagonal_block_line(block_indices=[0, 0], diagonal_blocks=np.eye(2))
    with pytest.raises(ValueError):
        builder.add_tridiagonal_block_line(block_indices=[0, 2], diagonal_blocks=np.eye(2))
    with pytest.raises(ValueError):
        builder.add_tridiagonal_block_line(
            block_indices=[0, 1],
            diagonal_blocks=np.ones((3, 2, 2)),
        )


def test_block_coo_operator_wraps_as_block_linear_operator() -> None:
    layout = RHS1BlockLayout.from_operator(_fake_op(n_species=1, n_x=1, n_xi=1, n_theta=2, n_zeta=2, include_phi1=False))
    dense = jnp.eye(layout.total_size, dtype=jnp.float64)
    coo = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    wrapped = coo.as_block_linear_operator(layout, name="identity_block_coo")
    x = jnp.arange(layout.total_size, dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(wrapped.matvec(x)), np.asarray(x))
    assert wrapped.to_dict()["name"] == "identity_block_coo"


def test_block_coo_operator_builds_block_jacobi_factor() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.25, 0.0],
            [2.0, 3.0, 0.0, -0.5],
            [1.0, 0.0, 5.0, -1.0],
            [0.0, 0.5, 1.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    factor = op.block_jacobi_factor()
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)
    expected = np.concatenate(
        [
            np.linalg.solve(np.asarray(dense[:2, :2]), np.asarray(rhs[:2])),
            np.linalg.solve(np.asarray(dense[2:, 2:]), np.asarray(rhs[2:])),
        ]
    )

    np.testing.assert_allclose(np.asarray(factor.apply(rhs)), expected, rtol=1.0e-12, atol=1.0e-12)
    assert factor.to_dict()["kind"] == "uniform_block_diagonal"


def test_block_coo_operator_builds_grouped_line_jacobi_factor() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.5, 0.0, 9.0, 9.0, 9.0, 9.0],
            [2.0, 3.0, 0.0, -0.25, 9.0, 9.0, 9.0, 9.0],
            [0.75, 0.0, 5.0, -1.0, 9.0, 9.0, 9.0, 9.0],
            [0.0, 0.5, 1.5, 2.5, 9.0, 9.0, 9.0, 9.0],
            [8.0, 8.0, 8.0, 8.0, 6.0, 1.0, -0.5, 0.0],
            [8.0, 8.0, 8.0, 8.0, -2.0, 4.0, 0.0, 0.75],
            [8.0, 8.0, 8.0, 8.0, 0.25, 0.0, 3.5, 1.0],
            [8.0, 8.0, 8.0, 8.0, 0.0, -0.5, 1.25, 4.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    factor = op.line_jacobi_factor(blocks_per_line=2)
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0], dtype=jnp.float64)
    expected = np.concatenate(
        [
            np.linalg.solve(np.asarray(dense[:4, :4]), np.asarray(rhs[:4])),
            np.linalg.solve(np.asarray(dense[4:8, 4:8]), np.asarray(rhs[4:8])),
        ]
    )

    np.testing.assert_allclose(np.asarray(factor.apply(rhs)), expected, rtol=1.0e-12, atol=1.0e-12)
    assert factor.to_dict()["n_blocks"] == 2
    assert factor.to_dict()["block_size"] == 4


def test_block_coo_operator_line_jacobi_validates_grouping() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(6, dtype=jnp.float64), block_size=2)

    with pytest.raises(ValueError):
        op.line_jacobi_factor(blocks_per_line=0)
    with pytest.raises(ValueError):
        op.line_jacobi_factor(blocks_per_line=2)


def test_block_coo_operator_builds_indexed_grouped_jacobi_factor() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 9.0, 9.0, 0.5, 0.0, 8.0, 8.0],
            [2.0, 3.0, 9.0, 9.0, 0.0, -0.25, 8.0, 8.0],
            [7.0, 7.0, 5.0, -1.0, 6.0, 6.0, 0.25, 0.0],
            [7.0, 7.0, 1.5, 2.5, 6.0, 6.0, 0.0, 0.5],
            [0.75, 0.0, 9.0, 9.0, 6.0, 1.0, 8.0, 8.0],
            [0.0, 0.5, 9.0, 9.0, -2.0, 4.0, 8.0, 8.0],
            [7.0, 7.0, -0.5, 0.0, 6.0, 6.0, 3.5, 1.0],
            [7.0, 7.0, 0.0, 0.75, 6.0, 6.0, 1.25, 4.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    factor = op.grouped_jacobi_factor(block_groups=np.asarray([[0, 2], [1, 3]], dtype=np.int32))
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0], dtype=jnp.float64)

    expected = np.empty((8,), dtype=np.float64)
    group0 = np.asarray([0, 1, 4, 5], dtype=np.int64)
    group1 = np.asarray([2, 3, 6, 7], dtype=np.int64)
    expected[group0] = np.linalg.solve(np.asarray(dense)[np.ix_(group0, group0)], np.asarray(rhs)[group0])
    expected[group1] = np.linalg.solve(np.asarray(dense)[np.ix_(group1, group1)], np.asarray(rhs)[group1])

    np.testing.assert_allclose(np.asarray(factor.apply(rhs)), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(jax.jit(factor.apply)(rhs)), expected, rtol=1.0e-12, atol=1.0e-12)
    assert factor.to_dict()["kind"] == "grouped_block_diagonal"
    assert factor.to_dict()["blocks_per_group"] == 2
    assert factor.to_dict()["block_size"] == 4


def test_block_coo_operator_grouped_jacobi_validates_groups() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(8, dtype=jnp.float64), block_size=2)

    with pytest.raises(ValueError):
        op.grouped_jacobi_factor(block_groups=np.asarray([0, 1], dtype=np.int32))
    with pytest.raises(ValueError):
        op.grouped_jacobi_factor(block_groups=np.asarray([[0, 1], [1, 2]], dtype=np.int32))
    with pytest.raises(ValueError):
        op.grouped_jacobi_factor(block_groups=np.asarray([[0, 4]], dtype=np.int32))
    with pytest.raises(ValueError):
        RHS1GroupedBlockDiagonalFactor.from_dense_blocks(
            np.eye(4, dtype=np.float64)[None, :, :],
            block_groups=np.asarray([[0, 1]], dtype=np.int32),
            n_operator_blocks=1,
            scalar_block_size=2,
        )


def test_galerkin_residual_correction_solves_fixed_coarse_equation() -> None:
    dense = jnp.asarray(
        [
            [4.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, -0.5],
            [0.25, 0.0, 5.0, 0.0],
            [0.0, -0.75, 0.0, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    local = op.block_jacobi_factor()
    basis = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    ) / jnp.sqrt(2.0)
    coarse = RHS1GalerkinResidualCorrection.from_basis(operator=op, basis=basis, regularization=1.0e-12)
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    z0 = local.apply(rhs)
    residual0 = rhs - op.matvec(z0)
    z1 = z0 + coarse.apply(residual0)
    residual1 = rhs - op.matvec(z1)

    assert float(jnp.linalg.norm(residual1)) < float(jnp.linalg.norm(residual0))
    np.testing.assert_allclose(
        np.asarray(jax.jit(coarse.apply)(residual0)),
        np.asarray(coarse.apply(residual0)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert coarse.to_dict()["kind"] == "galerkin_residual_correction"
    assert coarse.to_dict()["n_coarse"] == 2


def test_galerkin_residual_correction_validates_memory_guards() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(4, dtype=jnp.float64), block_size=2)
    basis = jnp.eye(4, dtype=jnp.float64)

    with pytest.raises(ValueError):
        RHS1GalerkinResidualCorrection.from_basis(operator=op, basis=jnp.ones((5, 1), dtype=jnp.float64))
    with pytest.raises(MemoryError, match="basis"):
        RHS1GalerkinResidualCorrection.from_basis(operator=op, basis=basis, max_basis_nbytes=1)
    with pytest.raises(MemoryError, match="coarse size"):
        RHS1GalerkinResidualCorrection.from_basis(operator=op, basis=basis, max_coarse_size=1)


def test_matrix_free_galerkin_residual_correction_matches_dense_basis() -> None:
    dense = jnp.asarray(
        [
            [4.0, 0.0, 1.0, 0.0],
            [0.0, 3.0, 0.0, -0.5],
            [0.25, 0.0, 5.0, 0.0],
            [0.0, -0.75, 0.0, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    basis = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=jnp.float64,
    ) / jnp.sqrt(2.0)
    dense_coarse = RHS1GalerkinResidualCorrection.from_basis(operator=op, basis=basis, regularization=1.0e-12)

    def _prolong(coefficients):
        return basis @ jnp.asarray(coefficients, dtype=jnp.float64)

    def _restrict(vector):
        return basis.T @ jnp.asarray(vector, dtype=jnp.float64)

    matrix_free = RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
        operator=op,
        restrict_fn=_restrict,
        prolong_fn=_prolong,
        n_coarse=2,
        regularization=1.0e-12,
        basis_batch_size=1,
    )
    rhs = jnp.asarray([1.0, -2.0, 0.5, 3.0], dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(matrix_free.coarse_matrix),
        np.asarray(dense_coarse.coarse_matrix),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(matrix_free.coarse_matrix @ matrix_free.coarse_inverse),
        np.eye(2),
        rtol=1.0e-10,
        atol=1.0e-10,
    )
    np.testing.assert_allclose(
        np.asarray(matrix_free.apply(rhs)),
        np.asarray(dense_coarse.apply(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(jax.jit(matrix_free.apply)(rhs)),
        np.asarray(dense_coarse.apply(rhs)),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert matrix_free.to_dict()["kind"] == "matrix_free_galerkin_residual_correction"
    assert matrix_free.to_dict()["basis_storage_nbytes"] == 0
    assert matrix_free.to_dict()["basis_batch_size"] == 1
    assert matrix_free.to_dict()["coarse_inverse_nbytes"] == matrix_free.to_dict()["coarse_nbytes"]
    assert matrix_free.to_dict()["solver_kind"] == "precomputed_dense_inverse"


def test_matrix_free_galerkin_residual_correction_validates_callbacks_and_guards() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(4, dtype=jnp.float64), block_size=2)

    def _bad_prolong(_coefficients):
        return jnp.ones((5, 1), dtype=jnp.float64)

    def _restrict(vector):
        arr = jnp.asarray(vector, dtype=jnp.float64)
        return arr[:2] if arr.ndim == 1 else arr[:2, :]

    with pytest.raises(ValueError):
        RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
            operator=op,
            restrict_fn=_restrict,
            prolong_fn=_bad_prolong,
            n_coarse=2,
        )

    def _prolong(coefficients):
        arr = jnp.asarray(coefficients, dtype=jnp.float64)
        return jnp.vstack([arr, arr])

    with pytest.raises(MemoryError, match="coarse size"):
        RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
            operator=op,
            restrict_fn=_restrict,
            prolong_fn=_prolong,
            n_coarse=2,
            max_coarse_size=1,
        )
    with pytest.raises(MemoryError, match="basis batch"):
        RHS1MatrixFreeGalerkinResidualCorrection.from_callbacks(
            operator=op,
            restrict_fn=_restrict,
            prolong_fn=_prolong,
            n_coarse=2,
            max_basis_batch_nbytes=1,
        )


def test_block_coo_operator_validates_shapes_and_indices() -> None:
    with pytest.raises(ValueError):
        RHS1BlockCOOOperator.from_blocks(
            row_blocks=[0, 2],
            col_blocks=[0, 1],
            blocks=jnp.ones((2, 2, 2), dtype=jnp.float64),
            n_block_rows=2,
            n_block_cols=2,
        )
    with pytest.raises(ValueError):
        RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(3, dtype=jnp.float64), block_size=2)
    with pytest.raises(ValueError):
        RHS1BlockCOOBuilder(shape=(4, 4), block_size=2).add_dense_block(2, 0, np.eye(2))
    with pytest.raises(ValueError):
        RHS1BlockCOOOperator.from_scalar_coo_entries(
            row_indices=[0],
            col_indices=[4],
            values=[1.0],
            shape=(4, 4),
            block_size=2,
        )


def test_uniform_block_diagonal_factor_matches_dense_block_solve() -> None:
    blocks = jnp.asarray(
        [
            [[4.0, 1.0], [2.0, 3.0]],
            [[5.0, -1.0], [1.5, 2.5]],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)
    factor = RHS1UniformBlockDiagonalFactor.from_dense_blocks(blocks)

    got = factor.apply(rhs)
    got_jit = jax.jit(factor.apply)(rhs)
    expected = np.concatenate(
        [
            np.linalg.solve(np.asarray(blocks[0]), np.asarray(rhs[:2])),
            np.linalg.solve(np.asarray(blocks[1]), np.asarray(rhs[2:])),
        ]
    )

    np.testing.assert_allclose(np.asarray(got), expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(got_jit), expected, rtol=1.0e-12, atol=1.0e-12)
    assert factor.to_dict()["kind"] == "uniform_block_diagonal"


def test_uniform_block_diagonal_factor_extracts_dense_matrix_blocks() -> None:
    matrix = jnp.asarray(
        [
            [2.0, 0.5, 9.0, 9.0],
            [0.25, 3.0, 9.0, 9.0],
            [8.0, 8.0, 4.0, 1.0],
            [8.0, 8.0, -1.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    factor = RHS1UniformBlockDiagonalFactor.from_dense_matrix(matrix, block_size=2)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    expected = np.concatenate(
        [
            np.linalg.solve(np.asarray(matrix[:2, :2]), np.asarray(rhs[:2])),
            np.linalg.solve(np.asarray(matrix[2:, 2:]), np.asarray(rhs[2:])),
        ]
    )

    np.testing.assert_allclose(np.asarray(factor.apply(rhs)), expected, rtol=1.0e-12, atol=1.0e-12)


def test_probe_block_preconditioner_accepts_true_residual_reduction() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    factor = op.block_jacobi_factor()
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)

    probe = probe_rhs1_block_preconditioner(
        matvec=op.matvec,
        rhs=rhs,
        preconditioner=factor.apply,
        target_residual_norm=1.0e-12,
        target_ratio=0.25,
        factor_metadata=factor.to_dict(),
    )

    assert probe.accepted is True
    assert probe.reason == "target_residual_met"
    assert probe.residual_after_norm < 1.0e-12
    assert probe.improvement_ratio is not None
    assert probe.to_dict()["factor_metadata"]["kind"] == "uniform_block_diagonal"


def test_probe_block_jacobi_preconditioner_convenience_api() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)

    probe = probe_rhs1_block_jacobi_preconditioner(
        operator=op,
        rhs=rhs,
        target_residual_norm=1.0e-12,
    )

    assert probe.accepted is True
    assert probe.reason == "target_residual_met"
    assert probe.to_dict()["factor_metadata"]["n_blocks"] == 2


def test_probe_block_preconditioner_rejects_weak_step() -> None:
    matrix = jnp.eye(4, dtype=jnp.float64)
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    probe = probe_rhs1_block_preconditioner(
        matvec=lambda x: matrix @ x,
        rhs=rhs,
        preconditioner=lambda r: 0.01 * r,
        target_ratio=0.5,
    )

    assert probe.accepted is False
    assert probe.reason == "insufficient_residual_reduction"
    np.testing.assert_allclose(np.asarray(probe.x_candidate), np.zeros(4))


def test_preflight_block_jacobi_candidate_accepts_jitted_exact_blocks() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)

    probe = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=rhs,
        target_residual_norm=1.0e-12,
        max_data_nbytes=op.data_nbytes,
        require_jit_apply=True,
    )

    assert probe.accepted is True
    assert probe.reason == "target_residual_met"
    assert probe.to_dict()["factor_metadata"]["factor"]["kind"] == "uniform_block_diagonal"


def test_preflight_block_jacobi_candidate_rejects_data_budget_and_shape() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(4, dtype=jnp.float64), block_size=2)

    too_large = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=jnp.ones((4,), dtype=jnp.float64),
        max_data_nbytes=op.data_nbytes - 1,
    )
    bad_shape = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=jnp.ones((5,), dtype=jnp.float64),
    )

    assert too_large.accepted is False
    assert too_large.reason == "data_budget_exceeded"
    assert bad_shape.accepted is False
    assert bad_shape.reason == "shape_mismatch"


def test_preflight_block_jacobi_candidate_rejects_singular_missing_diagonal_without_regularization() -> None:
    op = RHS1BlockCOOOperator.from_blocks(
        row_blocks=[0],
        col_blocks=[1],
        blocks=jnp.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=jnp.float64),
        n_block_rows=2,
        n_block_cols=2,
    )
    rhs = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    rejected = preflight_rhs1_block_jacobi_candidate(operator=op, rhs=rhs, require_jit_apply=False)
    regularized = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=rhs,
        regularization=1.0,
        target_ratio=1.1,
        require_jit_apply=False,
    )

    assert rejected.accepted is False
    assert rejected.reason in {"correction_amplification_exceeded", "nonfinite_residual"}
    assert regularized.accepted is True


def test_preflight_block_jacobi_candidate_rejects_amplified_correction() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(1.0e-12 * jnp.eye(4, dtype=jnp.float64), block_size=2)
    rhs = jnp.ones((4,), dtype=jnp.float64)

    probe = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=rhs,
        max_correction_ratio=1.0e3,
        require_jit_apply=False,
    )

    assert probe.accepted is False
    assert probe.reason == "correction_amplification_exceeded"


def test_preflight_line_jacobi_candidate_keeps_same_line_coupling() -> None:
    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.5, 0.0],
            [2.0, 3.0, 0.0, -0.25],
            [0.75, 0.0, 5.0, -1.0],
            [0.0, 0.5, 1.5, 2.5],
        ],
        dtype=jnp.float64,
    )
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0], dtype=jnp.float64)

    block_probe = preflight_rhs1_block_jacobi_candidate(
        operator=op,
        rhs=rhs,
        target_ratio=1.0e-12,
        require_jit_apply=False,
    )
    line_probe = preflight_rhs1_line_jacobi_candidate(
        operator=op,
        rhs=rhs,
        blocks_per_line=2,
        target_residual_norm=1.0e-12,
        require_jit_apply=True,
    )

    assert block_probe.accepted is False
    assert line_probe.accepted is True
    assert line_probe.reason == "target_residual_met"
    assert line_probe.to_dict()["factor_metadata"]["blocks_per_line"] == 2


def test_block_coo_line_preconditioner_reduces_true_fgmres_residual() -> None:
    """Grouped line factors must improve the physical residual, not only a proxy norm."""

    dense = jnp.asarray(
        [
            [4.0, 1.0, 0.7, 0.2, 0.06, 0.0, 0.0, 0.0],
            [2.0, 3.0, -0.1, -0.4, 0.0, -0.05, 0.0, 0.0],
            [0.8, -0.2, 5.0, -1.0, 0.0, 0.0, 0.05, 0.0],
            [0.1, 0.5, 1.5, 2.5, 0.0, 0.0, 0.0, -0.04],
            [0.04, 0.0, 0.0, 0.0, 6.0, 1.0, -0.5, 0.0],
            [0.0, -0.03, 0.0, 0.0, -2.0, 4.0, 0.0, 0.75],
            [0.0, 0.0, 0.02, 0.0, 0.25, 0.0, 3.5, 1.0],
            [0.0, 0.0, 0.0, -0.01, 0.0, -0.5, 1.25, 4.5],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0], dtype=jnp.float64)
    op = RHS1BlockCOOOperator.from_dense_matrix(dense, block_size=2)
    block_factor = op.block_jacobi_factor()
    line_factor = op.line_jacobi_factor(blocks_per_line=2)

    unpreconditioned, unpreconditioned_residual = fgmres_solve_with_residual(
        matvec=op.matvec,
        b=rhs,
        tol=1.0e-12,
        restart=1,
        maxiter=1,
        precondition_side="none",
    )
    block_jacobi, block_residual = fgmres_solve_with_residual(
        matvec=op.matvec,
        b=rhs,
        preconditioner=block_factor.apply,
        tol=1.0e-12,
        restart=1,
        maxiter=1,
        precondition_side="right",
    )
    line_jacobi, line_residual = fgmres_solve_with_residual(
        matvec=op.matvec,
        b=rhs,
        preconditioner=line_factor.apply,
        tol=1.0e-12,
        restart=1,
        maxiter=1,
        precondition_side="right",
    )

    dense_np = np.asarray(dense)
    rhs_np = np.asarray(rhs)
    np.testing.assert_allclose(
        np.asarray(unpreconditioned_residual),
        rhs_np - dense_np @ np.asarray(unpreconditioned.x),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(block_residual),
        rhs_np - dense_np @ np.asarray(block_jacobi.x),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.asarray(line_residual),
        rhs_np - dense_np @ np.asarray(line_jacobi.x),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert float(block_jacobi.residual_norm) < 0.5 * float(unpreconditioned.residual_norm)
    assert float(line_jacobi.residual_norm) < 0.1 * float(block_jacobi.residual_norm)


def test_preflight_line_jacobi_candidate_rejects_invalid_grouping() -> None:
    op = RHS1BlockCOOOperator.from_dense_matrix(jnp.eye(6, dtype=jnp.float64), block_size=2)

    probe = preflight_rhs1_line_jacobi_candidate(
        operator=op,
        rhs=jnp.ones((6,), dtype=jnp.float64),
        blocks_per_line=2,
        require_jit_apply=False,
    )

    assert probe.accepted is False
    assert "blocks_per_line must divide" in probe.reason
