from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import scipy.sparse as sp

from sfincs_jax.explicit_sparse import (
    SparseOperatorBundle,
    build_operator_from_blocks,
    build_operator_from_dense,
    build_operator_from_matvec,
    choose_storage_kind,
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
