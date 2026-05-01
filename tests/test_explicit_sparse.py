from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
import scipy.sparse as sp

from sfincs_jax.explicit_sparse import (
    SparseOperatorBundle,
    build_operator_from_blocks,
    build_operator_from_dense,
    build_operator_from_matvec,
    build_operator_from_pattern,
    choose_storage_kind,
    color_pattern_columns,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
    factorize_host_sparse_operator,
)


def test_storage_estimates_are_consistent() -> None:
    assert estimate_dense_nbytes((3, 4), np.float64) == 3 * 4 * 8
    assert estimate_csr_nbytes((3, 4), 5, data_dtype=np.float64, index_dtype=np.int32) == 5 * 12 + 4 * 4


def test_choose_storage_kind_prefers_dense_when_dense_fits_and_is_smaller() -> None:
    decision = choose_storage_kind(
        shape=(2, 2),
        nnz_estimate=4,
        backend="cpu",
        dense_max_mb=1.0,
        csr_max_mb=1.0,
    )
    assert decision.storage_kind == "dense"
    assert "dense" in decision.reason


def test_choose_storage_kind_prefers_csr_on_gpu_when_sparse_is_smaller() -> None:
    decision = choose_storage_kind(
        shape=(3, 3),
        nnz_estimate=3,
        backend="gpu",
        dense_max_mb=1.0,
        csr_max_mb=1.0,
    )
    assert decision.storage_kind == "csr"
    assert "GPU" in decision.reason or "sparse" in decision.reason


def test_choose_storage_kind_falls_back_to_operator_only_when_budgets_exhausted() -> None:
    decision = choose_storage_kind(
        shape=(8, 8),
        nnz_estimate=64,
        backend="cpu",
        dense_max_mb=0.0,
        csr_max_mb=0.0,
    )
    assert decision.storage_kind == "linear_operator"
    assert "operator-only" in decision.reason


def test_build_operator_from_dense_materializes_csr_and_tracks_metadata() -> None:
    dense = jnp.asarray(
        [
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 0.0],
            [4.0, 0.0, 5.0],
        ],
        dtype=jnp.float64,
    )
    bundle = build_operator_from_dense(dense, backend="gpu", dense_max_mb=0.0, csr_max_mb=1.0)
    assert isinstance(bundle, SparseOperatorBundle)
    assert sp.isspmatrix_csr(bundle.matrix)
    assert bundle.metadata.storage_kind == "csr"
    assert bundle.metadata.shape == (3, 3)
    assert bundle.metadata.nnz_estimate == 5
    np.testing.assert_allclose(bundle.matvec(np.array([1.0, 2.0, 3.0])), np.array([7.0, 6.0, 19.0]))


def test_build_operator_from_blocks_assembles_sparse_matrix() -> None:
    blocks = [
        [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[5.0], [6.0]])],
        [None, np.array([[7.0]])],
    ]
    bundle = build_operator_from_blocks(blocks, backend="cpu")
    assert sp.isspmatrix_csr(bundle.matrix)
    assert bundle.metadata.storage_kind == "csr"
    np.testing.assert_allclose(bundle.matrix.toarray(), np.array([[1.0, 2.0, 5.0], [3.0, 4.0, 6.0], [0.0, 0.0, 7.0]]))


def test_build_operator_from_matvec_falls_back_to_operator_only_when_dense_budget_tiny() -> None:
    a = np.array([[4.0, 1.0], [2.0, 3.0]])

    def mv(x):
        return a @ np.asarray(x)

    bundle = build_operator_from_matvec(
        mv,
        n=2,
        backend="gpu",
        dense_max_mb=0.0,
        csr_max_mb=0.0,
        allow_operator_only=True,
    )
    assert bundle.matrix is None
    assert bundle.metadata.storage_kind == "linear_operator"
    np.testing.assert_allclose(bundle.matvec(np.array([1.0, 2.0])), np.array([6.0, 8.0]))


def test_build_operator_from_matvec_can_assemble_csr_from_columns() -> None:
    a = np.diag([4.0, 3.0, 5.0])

    def mv(x):
        return a @ np.asarray(x)

    bundle = build_operator_from_matvec(
        mv,
        n=3,
        backend="gpu",
        dense_max_mb=1.0,
        csr_max_mb=1.0,
        block_cols=1,
    )
    assert sp.isspmatrix_csr(bundle.matrix)
    np.testing.assert_allclose(bundle.matrix.toarray(), a)
    assert bundle.metadata.storage_kind == "csr"


def test_build_operator_from_matvec_uses_block_basis_without_full_eye() -> None:
    a = np.diag([1.0, 2.0, 3.0, 4.0, 5.0])
    seen_shapes: list[tuple[int, int]] = []
    seen_nnz: list[int] = []

    def mv(x):
        return a @ np.asarray(x)

    def mm(cols):
        cols_np = np.asarray(cols)
        seen_shapes.append(tuple(cols_np.shape))
        seen_nnz.append(int(np.count_nonzero(cols_np)))
        return a @ cols_np

    bundle = build_operator_from_matvec(
        mv,
        n=5,
        backend="gpu",
        dense_max_mb=1.0,
        csr_max_mb=1.0,
        block_cols=2,
        matmat=mm,
    )

    assert seen_shapes == [(5, 2), (5, 2), (5, 1)]
    assert seen_nnz == [2, 2, 1]
    np.testing.assert_allclose(bundle.matrix.toarray(), a)


def test_color_pattern_columns_groups_disjoint_row_supports() -> None:
    pattern = sp.eye(5, format="csr")
    colors = color_pattern_columns(pattern)
    assert colors == [[0, 1, 2, 3, 4]]

    tridiagonal_pattern = sp.diags([np.ones(4), np.ones(5), np.ones(4)], offsets=[-1, 0, 1], format="csr")
    tridiagonal_colors = color_pattern_columns(tridiagonal_pattern)
    assert len(tridiagonal_colors) < tridiagonal_pattern.shape[1]
    for cols in tridiagonal_colors:
        rows_seen: set[int] = set()
        for col in cols:
            rows = set(tridiagonal_pattern.tocsc().indices[tridiagonal_pattern.tocsc().indptr[col] : tridiagonal_pattern.tocsc().indptr[col + 1]])
            assert rows_seen.isdisjoint(rows)
            rows_seen.update(rows)


def test_build_operator_from_pattern_uses_one_probe_for_diagonal_pattern() -> None:
    a = np.diag([2.0, 3.0, 5.0, 7.0])
    calls: list[np.ndarray] = []

    def mv(x):
        x_np = np.asarray(x)
        calls.append(x_np.copy())
        return a @ x_np

    bundle = build_operator_from_pattern(mv, pattern=sp.eye(4, format="csr"), backend="cpu")

    assert sp.isspmatrix_csr(bundle.matrix)
    assert len(calls) == 1
    np.testing.assert_allclose(calls[0], np.ones(4))
    np.testing.assert_allclose(bundle.matrix.toarray(), a)
    assert bundle.metadata.storage_kind == "csr"
    assert bundle.metadata.block_cols == 1
    assert "pattern-probed" in bundle.metadata.reason


def test_build_operator_from_pattern_recovers_tridiagonal_with_coloring() -> None:
    a = sp.diags(
        [np.array([-1.0, -2.0, -3.0, -4.0]), np.array([4.0, 5.0, 6.0, 7.0, 8.0]), np.array([1.0, 2.0, 3.0, 4.0])],
        offsets=[-1, 0, 1],
        format="csr",
    ).toarray()
    pattern = sp.csr_matrix(a != 0)
    calls: list[np.ndarray] = []

    def mv(x):
        x_np = np.asarray(x)
        calls.append(x_np.copy())
        return a @ x_np

    bundle = build_operator_from_pattern(mv, pattern=pattern, backend="gpu")

    assert len(calls) == bundle.metadata.block_cols
    assert len(calls) < a.shape[1]
    np.testing.assert_allclose(bundle.matrix.toarray(), a)
    np.testing.assert_allclose(bundle.matvec(np.arange(1.0, 6.0)), a @ np.arange(1.0, 6.0))


def test_build_operator_from_pattern_drops_overapproximated_structural_zeros() -> None:
    a = np.diag([2.0, 3.0, 5.0])
    pattern = sp.csr_matrix(np.ones((3, 3), dtype=bool))

    def mv(x):
        return a @ np.asarray(x)

    bundle = build_operator_from_pattern(mv, pattern=pattern, backend="cpu")

    assert bundle.matrix.nnz == 3
    np.testing.assert_allclose(bundle.matrix.toarray(), a)


def test_build_operator_from_pattern_can_fall_back_to_operator_only_when_budgeted() -> None:
    a = np.eye(3)

    def mv(x):
        return a @ np.asarray(x)

    bundle = build_operator_from_pattern(
        mv,
        pattern=sp.eye(3, format="csr"),
        backend="cpu",
        csr_max_mb=0.0,
        allow_operator_only=True,
    )

    assert bundle.matrix is None
    assert bundle.metadata.storage_kind == "linear_operator"
    np.testing.assert_allclose(bundle.matvec(np.array([1.0, 2.0, 3.0])), np.array([1.0, 2.0, 3.0]))


def test_factorize_host_sparse_operator_solves_exactly() -> None:
    dense = np.array([[4.0, 1.0], [2.0, 3.0]])
    bundle = build_operator_from_dense(dense, backend="cpu", force_sparse=True)
    factor = factorize_host_sparse_operator(bundle, kind="lu")
    rhs = np.array([1.0, 0.0])
    sol = factor.solve(rhs)
    np.testing.assert_allclose(sol, np.linalg.solve(dense, rhs))
    assert factor.kind == "lu"
    assert factor.metadata.storage_kind == "csr"


def test_factorize_host_sparse_operator_accepts_raw_sparse_matrix() -> None:
    matrix = sp.csr_matrix([[3.0, 0.0], [0.0, 5.0]])
    factor = factorize_host_sparse_operator(matrix, kind="lu")
    rhs = jnp.asarray([6.0, 10.0], dtype=jnp.float64)
    np.testing.assert_allclose(factor.solve(rhs), np.array([2.0, 2.0]))


def test_factorize_host_sparse_operator_reports_singular_branch_actionably() -> None:
    matrix = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(RuntimeError, match="Host sparse factorization failed"):
        factorize_host_sparse_operator(matrix, kind="lu")
