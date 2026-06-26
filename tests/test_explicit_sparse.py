from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
import scipy.sparse as sp

from sfincs_jax.solvers.explicit_sparse import (
    SparseDecision,
    SparseOperatorBundle,
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    build_operator_from_blocks,
    build_operator_from_dense,
    build_operator_from_matvec,
    build_operator_from_pattern,
    choose_storage_kind,
    color_pattern_columns,
    csr_matvec,
    deterministic_sparse_probe_matrix,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
    estimate_multifrontal_direct_lu_nbytes,
    estimate_superlu_factor_storage,
    factorize_host_sparse_operator,
    host_direct_solve_with_refinement,
    host_sparse_direct_polish,
    host_sparse_direct_solve_with_refinement,
    wrap_sparse_factor_with_coarse_correction,
)


def test_storage_estimates_are_consistent() -> None:
    assert estimate_dense_nbytes((3, 4), np.float64) == 3 * 4 * 8
    assert estimate_csr_nbytes((3, 4), 5, data_dtype=np.float64, index_dtype=np.int32) == 5 * 12 + 4 * 4


def test_csr_matvec_matches_dense_and_rejects_invalid_shapes() -> None:
    data = jnp.asarray([2.0, 3.0, -1.0])
    indices = jnp.asarray([0, 2, 1], dtype=jnp.int32)
    indptr = jnp.asarray([0, 2, 3], dtype=jnp.int32)
    x = jnp.asarray([5.0, 7.0, 11.0])

    y = csr_matvec(data=data, indices=indices, indptr=indptr, x=x, n_rows=2)
    np.testing.assert_allclose(np.asarray(y), np.asarray([43.0, -7.0]))

    with pytest.raises(ValueError, match="indptr must be 1D"):
        csr_matvec(data=data, indices=indices, indptr=jnp.asarray([[0, 1]]), x=x)

    with pytest.raises(ValueError, match="data and indices must be 1D"):
        csr_matvec(data=jnp.asarray([[1.0]]), indices=indices, indptr=indptr, x=x)

    with pytest.raises(ValueError, match="incompatible length"):
        csr_matvec(data=data, indices=indices, indptr=indptr, x=x, n_rows=3)


def test_multifrontal_direct_lu_estimate_tracks_profiled_fill() -> None:
    estimate = estimate_multifrontal_direct_lu_nbytes(
        8_678_219,
        fill_ratio=888_160_169 / 8_678_219,
        overhead=1.0,
    )
    assert estimate == 888_160_169 * 12


def test_sparse_symbolic_analysis_reports_reusable_ordering_metadata() -> None:
    matrix = sp.diags(
        [
            np.ones(5),
            np.array([2.0, 3.0, 4.0, 5.0]),
            np.array([-1.0, -2.0, -3.0, -4.0]),
        ],
        offsets=[0, 1, -1],
        format="csr",
    )

    analysis = analyze_sparse_symbolic_structure(matrix, ordering_kind="rcm", block_size_target=2)
    as_dict = analysis.to_dict()

    assert analysis.shape == (5, 5)
    assert analysis.nnz == matrix.nnz
    assert analysis.pattern_hash
    assert analysis.ordering_kind in {"rcm", "natural"}
    assert analysis.ordering_hash
    assert analysis.diagonal_missing == 0
    assert analysis.row_nnz_max == 3
    assert analysis.block_count == 3
    assert analysis.block_size_max <= 2
    assert analysis.block_nnz_max > 0
    assert as_dict["pattern_hash"] == analysis.pattern_hash
    assert analysis.cache_key()[2] == analysis.pattern_hash


@pytest.mark.parametrize("ordering", ["nested_dissection", "mumps_like", "scotch", "parmetis", "metis"])
def test_sparse_symbolic_analysis_supports_mumps_like_ordering_aliases(ordering: str) -> None:
    n = 12
    left = sp.diags(
        [np.ones(5), -np.ones(4), -np.ones(4)],
        offsets=[0, -1, 1],
        shape=(5, 5),
        format="csr",
    )
    right = sp.diags(
        [2.0 * np.ones(5), -np.ones(4), -np.ones(4)],
        offsets=[0, -1, 1],
        shape=(5, 5),
        format="csr",
    )
    matrix = sp.block_diag((left, right, sp.eye(2, format="csr")), format="lil")
    matrix[4, 10] = 0.25
    matrix[10, 4] = 0.25
    matrix[9, 11] = -0.5
    matrix[11, 9] = -0.5
    matrix = matrix.tocsr()

    analysis = analyze_sparse_symbolic_structure(matrix, ordering_kind=ordering, block_size_target=3)

    assert analysis.shape == (n, n)
    assert analysis.ordering_kind == "nested_dissection"
    assert analysis.permutation is not None
    assert analysis.inverse_permutation is not None
    np.testing.assert_array_equal(np.sort(analysis.permutation), np.arange(n))
    np.testing.assert_array_equal(analysis.inverse_permutation[analysis.permutation], np.arange(n))
    assert analysis.ordering_hash


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


def test_choose_storage_kind_force_paths_and_metadata_dict() -> None:
    with pytest.raises(ValueError, match="force_dense and force_sparse"):
        choose_storage_kind(shape=(2, 2), nnz_estimate=4, force_dense=True, force_sparse=True)

    dense = choose_storage_kind(
        shape=(2, 2),
        nnz_estimate=4,
        backend=None,
        dense_max_mb=0.0,
        csr_max_mb=0.0,
        force_dense=True,
        block_cols=3,
        drop_tol=0.25,
    )
    assert dense.storage_kind == "dense"
    assert dense.to_dict()["block_cols"] == 3
    assert dense.to_dict()["drop_tol"] == pytest.approx(0.25)

    sparse_rejected = choose_storage_kind(
        shape=(2, 2),
        nnz_estimate=4,
        backend="cpu",
        dense_max_mb=1.0,
        csr_max_mb=0.0,
        force_sparse=True,
    )
    assert sparse_rejected.storage_kind == "linear_operator"
    assert "CSR budget unavailable" in sparse_rejected.reason


def test_superlu_factor_storage_handles_dense_and_missing_factor_attributes() -> None:
    class DenseFactors:
        L = np.asarray([[1.0, 0.0], [2.0, 3.0]])
        U = np.asarray([[4.0, 5.0], [0.0, 6.0]])

    nbytes, nnz = estimate_superlu_factor_storage(DenseFactors())
    assert nbytes == DenseFactors.L.nbytes + DenseFactors.U.nbytes
    assert nnz == int(np.count_nonzero(DenseFactors.L) + np.count_nonzero(DenseFactors.U))
    assert estimate_superlu_factor_storage(object()) == (None, None)


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


def test_color_pattern_columns_honors_max_colors_preflight() -> None:
    pattern = sp.eye(4, format="csr")
    assert color_pattern_columns(pattern, max_colors=1) == [[0, 1, 2, 3, 4][:4]]

    dense_pattern = sp.csr_matrix(np.ones((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="max_colors=1"):
        color_pattern_columns(dense_pattern, max_colors=1)


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


def test_build_operator_from_pattern_batches_color_probes_with_matmat() -> None:
    n = 6
    a = np.arange(1.0, n * n + 1.0, dtype=np.float64).reshape((n, n))
    pattern = sp.csr_matrix(np.ones((n, n), dtype=bool))
    calls: list[np.ndarray] = []
    events: list[str] = []

    def mv(_x):
        raise AssertionError("batched pattern probing should use matmat when provided")

    def matmat(x):
        x_np = np.asarray(x)
        calls.append(x_np.copy())
        return a @ x_np

    bundle = build_operator_from_pattern(
        mv,
        pattern=pattern,
        backend="cpu",
        color_batch=3,
        matmat=matmat,
        progress_callback=events.append,
    )

    assert len(calls) == 2
    assert all(call.shape == (n, 3) for call in calls)
    assert events[1] == "pattern-probe coloring complete colors=6 columns=6 color_batch=3"
    np.testing.assert_allclose(bundle.matrix.toarray(), a)
    assert bundle.metadata.block_cols == 6
    assert "color_batch=3" in bundle.metadata.reason


def test_build_operator_from_pattern_reports_progress_for_materialized_probe() -> None:
    n = 11
    a = np.arange(1.0, n * n + 1.0, dtype=np.float64).reshape((n, n))
    pattern = sp.csr_matrix(np.ones((n, n), dtype=bool))
    calls: list[np.ndarray] = []
    events: list[str] = []

    def mv(x):
        x_np = np.asarray(x)
        calls.append(x_np.copy())
        return a @ x_np

    bundle = build_operator_from_pattern(
        mv,
        pattern=pattern,
        backend="cpu",
        progress_callback=events.append,
    )

    assert len(calls) == n
    assert events[0].startswith("pattern-probe preflight shape=11x11 pattern_nnz=121")
    assert events[1] == "pattern-probe coloring complete colors=11 columns=11 color_batch=1"
    assert "pattern-probe colors_done=10/11" in events
    assert events[-2:] == ["pattern-probe colors_done=11/11", "pattern-probe csr built nnz=121"]
    np.testing.assert_allclose(bundle.matrix.toarray(), a)


def test_build_operator_from_pattern_reports_progress_for_empty_pattern_without_probe() -> None:
    events: list[str] = []

    def mv(_x):
        raise AssertionError("empty patterns should not require matvec probes")

    bundle = build_operator_from_pattern(
        mv,
        pattern=sp.csr_matrix((3, 3), dtype=bool),
        backend="cpu",
        progress_callback=events.append,
    )

    assert bundle.matrix.nnz == 0
    assert events[0].startswith("pattern-probe preflight shape=3x3 pattern_nnz=0")
    assert events[1:] == [
        "pattern-probe coloring complete colors=0 columns=3 color_batch=1",
        "pattern-probe csr built nnz=0",
    ]


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


def test_build_operator_from_pattern_enforces_csr_budget_when_materializing() -> None:
    pattern = sp.eye(3, format="csr")

    def mv(x):
        return np.asarray(x)

    with pytest.raises(MemoryError, match="pattern CSR estimate would exceed budget"):
        build_operator_from_pattern(
            mv,
            pattern=pattern,
            backend="cpu",
            csr_max_mb=0.0,
            allow_operator_only=False,
        )


def test_factorize_host_sparse_operator_solves_exactly() -> None:
    dense = np.array([[4.0, 1.0], [2.0, 3.0]])
    bundle = build_operator_from_dense(dense, backend="cpu", force_sparse=True)
    factor = factorize_host_sparse_operator(bundle, kind="lu")
    rhs = np.array([1.0, 0.0])
    sol = factor.solve(rhs)
    np.testing.assert_allclose(sol, np.linalg.solve(dense, rhs))
    assert factor.kind == "lu"
    assert factor.metadata.storage_kind == "csr"
    assert factor.factor_nbytes_estimate is not None and factor.factor_nbytes_estimate > 0
    assert factor.factor_nnz_estimate is not None and factor.factor_nnz_estimate >= bundle.matrix.nnz
    assert factor.factor_s is not None and factor.factor_s >= 0.0
    assert estimate_superlu_factor_storage(factor.factor) == (
        factor.factor_nbytes_estimate,
        factor.factor_nnz_estimate,
    )


def test_factorize_host_sparse_operator_accepts_raw_sparse_matrix() -> None:
    matrix = sp.csr_matrix([[3.0, 0.0], [0.0, 5.0]])
    factor = factorize_host_sparse_operator(matrix, kind="lu")
    rhs = jnp.asarray([6.0, 10.0], dtype=jnp.float64)
    np.testing.assert_allclose(factor.solve(rhs), np.array([2.0, 2.0]))


def test_factorize_host_sparse_operator_supports_jacobi_factor() -> None:
    matrix = sp.csr_matrix([[4.0, 1.0], [2.0, 8.0]])
    factor = factorize_host_sparse_operator(matrix, kind="jacobi")

    np.testing.assert_allclose(factor.solve(np.asarray([8.0, 16.0])), np.asarray([2.0, 2.0]))
    np.testing.assert_allclose(
        factor.factor.solve(np.asarray([[8.0, 4.0], [16.0, 8.0]])),
        np.asarray([[2.0, 1.0], [2.0, 1.0]]),
    )
    assert factor.kind == "jacobi"
    assert factor.factor_nbytes_estimate == 2 * np.dtype(np.float64).itemsize
    assert factor.factor_nnz_estimate == 2
    assert factor.factor_s is not None and factor.factor_s >= 0.0


def test_factorize_host_sparse_operator_symbolic_block_lu_solves_block_diagonal() -> None:
    matrix = sp.block_diag(
        [
            np.array([[4.0, 1.0], [2.0, 3.0]]),
            np.array([[5.0, -1.0], [1.0, 2.0]]),
        ],
        format="csr",
    )
    rhs = np.array([1.0, 2.0, -1.0, 3.0])
    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(matrix.toarray(), rhs), rtol=1e-13, atol=1e-13)
    assert factor.kind == "symbolic_block_lu"
    assert factor.factor_nbytes_estimate is not None and factor.factor_nbytes_estimate > 0
    assert factor.factor_nnz_estimate is not None and factor.factor_nnz_estimate > 0
    assert factor.factor.analysis.block_count == 2


def test_symbolic_block_lu_admission_accepts_exact_block_factor() -> None:
    matrix = sp.block_diag(
        [
            np.array([[4.0, 1.0], [2.0, 3.0]]),
            np.array([[5.0, -1.0], [1.0, 2.0]]),
        ],
        format="csr",
    )
    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )

    assert admission.accepted is True
    assert admission.max_relative_residual < 1.0e-13
    assert admission.probe_count == 4
    assert admission.to_dict()["accepted"] is True


def test_sparse_factor_admission_rejects_missing_or_nonsquare_operator_and_bad_probes() -> None:
    decision = SparseDecision(
        storage_kind="linear_operator",
        reason="test",
        backend="cpu",
        shape=(2, 2),
        dense_nbytes=32,
        csr_nbytes_estimate=32,
        nnz_estimate=None,
    )
    missing_matrix_bundle = SparseOperatorBundle(
        matrix=None,
        operator=build_operator_from_dense(np.eye(2)).operator,
        metadata=decision,
    )
    factor = factorize_host_sparse_operator(sp.eye(2, format="csr"), kind="jacobi")

    missing = admit_sparse_factor_against_operator(missing_matrix_bundle, factor)
    assert missing.accepted is False
    assert missing.reason == "missing_operator_matrix"

    nonsquare = admit_sparse_factor_against_operator(sp.csr_matrix(np.ones((2, 3))), factor)
    assert nonsquare.accepted is False
    assert nonsquare.reason == "operator_not_square"

    with pytest.raises(ValueError, match="probe rows"):
        admit_sparse_factor_against_operator(sp.eye(2, format="csr"), factor, probes=np.ones((3, 1)))


def test_deterministic_sparse_probe_matrix_covers_empty_and_scalar_systems() -> None:
    empty = deterministic_sparse_probe_matrix(0, count=2)
    assert empty.shape == (0, 2)

    scalar = deterministic_sparse_probe_matrix(1, count=4)
    assert scalar.shape == (1, 4)
    np.testing.assert_allclose(np.abs(scalar), np.ones((1, 4)))


def test_symbolic_block_lu_admission_rejects_missing_offblock_coupling() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-2,
        min_improvement_vs_identity=10.0,
    )

    assert admission.accepted is False
    assert admission.max_relative_residual > 1.0e-2
    assert admission.reason == "residual_or_improvement_gate_failed"


def test_symbolic_superblock_lu_retains_coupled_block_edges() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 2.5, 0.0],
            [2.0, 3.0, 0.0, -1.0],
            [3.0, 0.0, 5.0, -1.0],
            [0.0, -0.7, 1.0, 2.0],
        ]
    )
    rhs = np.array([1.0, 2.0, -1.0, 3.0])
    local = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )
    grouped = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_superblock_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_superblock_max_size=4,
        symbolic_superblock_max_blocks=2,
    )

    exact = np.linalg.solve(matrix.toarray(), rhs)
    local_error = np.linalg.norm(local.solve(rhs) - exact)
    grouped_error = np.linalg.norm(grouped.solve(rhs) - exact)
    admission = admit_sparse_factor_against_operator(
        grouped.operator,
        grouped,
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )

    assert grouped.kind == "symbolic_superblock_lu"
    assert grouped.factor.superblock_count == 1
    assert grouped.factor.retained_cross_nnz > 0
    assert grouped.factor.dropped_cross_nnz == 0
    assert grouped_error < 1.0e-12
    assert grouped_error < local_error
    assert admission.accepted is True


def test_symbolic_superblock_lu_admission_rejects_when_size_gate_drops_edges() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_superblock_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_superblock_max_size=2,
        symbolic_superblock_max_blocks=1,
    )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-2,
        min_improvement_vs_identity=10.0,
    )

    assert factor.factor.superblock_count == 2
    assert factor.factor.retained_cross_nnz == 0
    assert factor.factor.dropped_cross_nnz > 0
    assert admission.accepted is False
    assert admission.reason == "residual_or_improvement_gate_failed"


def test_symbolic_superblock_lu_can_reject_low_retained_coupling_before_factorization() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )

    with pytest.raises(RuntimeError, match="retained insufficient cross-block coupling"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_superblock_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=2,
            symbolic_superblock_max_size=2,
            symbolic_superblock_max_blocks=1,
            symbolic_superblock_min_retained_cross_fraction=0.5,
        )


def test_symbolic_superblock_lu_parallel_numeric_tasks_preserve_solution() -> None:
    matrix = sp.block_diag(
        (
            sp.csr_matrix([[4.0, 1.0], [0.5, 3.0]], dtype=np.float64),
            sp.csr_matrix([[5.0, -0.25], [1.0, 2.5]], dtype=np.float64),
        ),
        format="csr",
    )
    rhs = np.array([1.0, -2.0, 0.5, 3.0], dtype=np.float64)

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_superblock_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_superblock_max_size=2,
        symbolic_superblock_max_blocks=1,
        symbolic_numeric_parallel_workers=2,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )

    assert factor.factor.superblock_count == 2
    assert factor.factor.parallel_workers == 2
    assert factor.factor.numeric_factor_tasks == 2
    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-12,
        atol=1.0e-12,
    )
    assert admission.accepted is True


def test_symbolic_frontal_schur_lu_solves_separator_coupled_blocks() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    rhs = np.array([1.0, 2.0, -1.0, 3.0])

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_frontal_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_frontal_max_superblock_size=2,
        symbolic_frontal_max_superblock_blocks=1,
        symbolic_frontal_max_separator_cols=2,
        symbolic_frontal_boundary_width=0,
        symbolic_frontal_high_degree_cols=0,
        symbolic_frontal_regularization_rel=0.0,
        symbolic_frontal_min_cross_separator_fraction=1.0,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(matrix.toarray(), rhs), rtol=1.0e-12, atol=1.0e-12)
    assert factor.kind == "symbolic_frontal_schur_lu"
    assert factor.factor.separator_count == 2
    assert factor.factor.total_cross_nnz > 0
    assert factor.factor.selected_cross_nnz == factor.factor.total_cross_nnz
    assert factor.factor.cross_separator_fraction == 1.0
    assert factor.factor.dense_rhs_entries == 2
    assert factor.factor.peak_dense_rhs_entries == 1
    assert factor.factor.separator_update_columns == 2
    assert admission.accepted is True


def test_symbolic_frontal_schur_lu_rejects_insufficient_separator_coverage() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError, match="selected insufficient cross-block separator coverage"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=2,
            symbolic_frontal_max_superblock_size=2,
            symbolic_frontal_max_superblock_blocks=1,
            symbolic_frontal_max_separator_cols=0,
            symbolic_frontal_boundary_width=0,
            symbolic_frontal_high_degree_cols=0,
            symbolic_frontal_min_cross_separator_fraction=1.0,
        )


def test_symbolic_frontal_schur_lu_rejects_dense_rhs_work_budget() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(RuntimeError, match="dense separator RHS work budget exceeded"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=2,
            symbolic_frontal_max_superblock_size=2,
            symbolic_frontal_max_superblock_blocks=1,
            symbolic_frontal_max_separator_cols=2,
            symbolic_frontal_boundary_width=0,
            symbolic_frontal_high_degree_cols=0,
            symbolic_frontal_min_cross_separator_fraction=1.0,
            symbolic_frontal_max_dense_rhs_entries=1,
        )


_SYMBOLIC_BLR_FRONTAL_SCHUR_LU_KIND = "symbolic_blr_frontal_schur_lu"


def _rank_one_separator_coupled_blr_matrix() -> sp.csr_matrix:
    interior0 = np.array(
        [
            [5.0, 0.8, 0.0],
            [0.4, 4.5, -0.6],
            [0.0, 0.7, 3.8],
        ],
        dtype=np.float64,
    )
    interior1 = np.array(
        [
            [4.8, -0.5, 0.0],
            [0.3, 4.2, 0.9],
            [0.0, -0.4, 4.6],
        ],
        dtype=np.float64,
    )
    separator = np.array([[7.5, 0.4], [-0.2, 6.8]], dtype=np.float64)
    b0 = 0.7 * np.outer(np.array([1.0, -0.5, 0.25]), np.array([1.4, -0.9]))
    c0 = 0.6 * np.outer(np.array([-0.7, 0.55]), np.array([0.8, -0.35, 0.45]))
    b1 = 0.8 * np.outer(np.array([-0.6, 1.0, 0.35]), np.array([1.1, 0.75]))
    c1 = 0.5 * np.outer(np.array([0.6, -0.8]), np.array([-0.5, 0.65, 0.25]))

    matrix = np.zeros((8, 8), dtype=np.float64)
    matrix[:3, :3] = interior0
    matrix[3:6, 3:6] = interior1
    matrix[6:, 6:] = separator
    matrix[:3, 6:] = b0
    matrix[6:, :3] = c0
    matrix[3:6, 6:] = b1
    matrix[6:, 3:6] = c1
    return sp.csr_matrix(matrix)


def _factorize_symbolic_blr_frontal_schur_lu(matrix: sp.spmatrix, **overrides):
    kwargs = {
        "kind": _SYMBOLIC_BLR_FRONTAL_SCHUR_LU_KIND,
        "symbolic_ordering_kind": "natural",
        "symbolic_block_size": 3,
        "symbolic_frontal_tail_size": 2,
        "symbolic_frontal_max_separator_cols": 2,
        "symbolic_frontal_boundary_width": 0,
        "symbolic_frontal_high_degree_cols": 0,
        "symbolic_frontal_max_superblock_size": 3,
        "symbolic_frontal_max_superblock_blocks": 1,
        "symbolic_frontal_min_cross_separator_fraction": 1.0,
        "symbolic_frontal_regularization_rel": 0.0,
    }
    kwargs.update(overrides)
    try:
        return factorize_host_sparse_operator(matrix, **kwargs)
    except ValueError as exc:
        if _SYMBOLIC_BLR_FRONTAL_SCHUR_LU_KIND in str(exc) and "unknown factorization kind" in str(exc):
            pytest.xfail(
                f"{_SYMBOLIC_BLR_FRONTAL_SCHUR_LU_KIND} is not implemented in explicit_sparse dispatch yet"
            )
        raise


def test_symbolic_blr_frontal_schur_lu_solves_rank_one_separator_updates_under_budget() -> None:
    matrix = _rank_one_separator_coupled_blr_matrix()
    rhs = np.asarray([1.0, -2.0, 0.5, 1.5, -0.75, 2.25, -1.0, 0.8], dtype=np.float64)

    # Dense frontal updates need 12 local RHS entries here; rank-one BLR updates need 6.
    factor = _factorize_symbolic_blr_frontal_schur_lu(
        matrix,
        symbolic_frontal_max_dense_rhs_entries=6,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.kind == _SYMBOLIC_BLR_FRONTAL_SCHUR_LU_KIND
    assert factor.factor.separator_count == 2
    assert factor.factor.total_cross_nnz > 0
    assert factor.factor.selected_cross_nnz == factor.factor.total_cross_nnz
    assert factor.factor.cross_separator_fraction == 1.0
    assert factor.factor.metadata is not None
    assert factor.factor.metadata["blr_woodbury_rank"] > 0
    assert factor.factor.metadata["blr_error_estimate_max"] < 1.0e-12
    assert factor.factor_nbytes_estimate is not None and factor.factor_nbytes_estimate > 0


def test_symbolic_blr_frontal_schur_lu_rejects_ill_conditioned_woodbury_core_to_gmres() -> None:
    matrix = _rank_one_separator_coupled_blr_matrix()
    rhs = np.asarray([1.0, -2.0, 0.5, 1.5, -0.75, 2.25, -1.0, 0.8], dtype=np.float64)

    factor = _factorize_symbolic_blr_frontal_schur_lu(
        matrix,
        symbolic_frontal_max_dense_rhs_entries=6,
        symbolic_blr_frontal_woodbury_max_condition=1.0,
        symbolic_blr_frontal_gmres_rtol=1.0e-12,
        symbolic_blr_frontal_gmres_maxiter=20,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.factor.metadata is not None
    assert factor.factor.metadata["blr_woodbury_rank"] == 0
    assert factor.factor.metadata["blr_woodbury_condition"] is not None


def test_symbolic_blr_frontal_schur_lu_rejects_dense_rhs_update_budget() -> None:
    matrix = _rank_one_separator_coupled_blr_matrix()

    # The same rank-one updates should not fit below one compressed 3-row RHS per interior block.
    with pytest.raises(RuntimeError, match="budget exceeded"):
        _factorize_symbolic_blr_frontal_schur_lu(
            matrix,
            symbolic_frontal_max_dense_rhs_entries=5,
        )


def test_symbolic_blr_frontal_schur_lu_admission_accepts_and_rejects_small_matrix() -> None:
    matrix = _rank_one_separator_coupled_blr_matrix()
    accepted_factor = _factorize_symbolic_blr_frontal_schur_lu(
        matrix,
        symbolic_frontal_max_dense_rhs_entries=6,
    )
    rejected_factor = _factorize_symbolic_blr_frontal_schur_lu(
        matrix,
        symbolic_frontal_max_separator_cols=1,
        symbolic_frontal_min_cross_separator_fraction=0.0,
    )

    accepted = admit_sparse_factor_against_operator(
        accepted_factor.operator,
        accepted_factor,
        max_relative_residual=1.0e-10,
        min_improvement_vs_identity=1.0,
    )
    rejected = admit_sparse_factor_against_operator(
        rejected_factor.operator,
        rejected_factor,
        max_relative_residual=1.0e-2,
        min_improvement_vs_identity=10.0,
    )

    assert accepted.accepted is True
    assert accepted.max_relative_residual < 1.0e-10
    assert accepted.reason == "accepted"
    assert rejected_factor.factor.separator_count == 1
    assert rejected.accepted is False
    assert rejected.max_relative_residual > 1.0e-2
    assert rejected.reason == "residual_or_improvement_gate_failed"


def _nested_dissection_tridiagonal_matrix(n: int = 14) -> sp.csr_matrix:
    diagonal = 6.0 + 0.2 * np.arange(n, dtype=np.float64)
    upper = -0.7 + 0.01 * np.arange(n - 1, dtype=np.float64)
    lower = -0.4 - 0.015 * np.arange(n - 1, dtype=np.float64)
    matrix = sp.diags([lower, diagonal, upper], offsets=[-1, 0, 1], format="lil")
    # These same-side longer-range terms keep the toy operator close to the
    # geometry-rich active matrices without jumping across recursive separators.
    for row, col, value in [(0, 2, 0.12), (2, 0, -0.08), (9, 11, 0.09), (11, 9, -0.07)]:
        if row < n and col < n:
            matrix[row, col] = value
    return matrix.tocsr()


def test_symbolic_nd_frontal_schur_lu_recursively_solves_separated_operator() -> None:
    matrix = _nested_dissection_tridiagonal_matrix()
    rhs = np.linspace(-1.0, 2.0, matrix.shape[0], dtype=np.float64)

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_nd_frontal_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=3,
        symbolic_nd_max_leaf_size=3,
        symbolic_nd_max_depth=4,
        symbolic_nd_separator_width=2,
        symbolic_nd_max_separator_cols=3,
        symbolic_nd_high_degree_cols=0,
        symbolic_nd_regularization_rel=0.0,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-11,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.kind == "symbolic_nd_frontal_schur_lu"
    assert factor.factor.node_count > 3
    assert factor.factor.leaf_count >= 4
    assert factor.factor.max_depth_reached >= 2
    assert factor.factor.separator_count_total > factor.factor.max_separator_count
    assert factor.factor.dense_update_entries > 0
    assert factor.factor.separator_update_chunks > 0
    assert factor.factor.metadata["architecture"] == "symbolic_nd_frontal_schur_lu"
    assert factor.factor.metadata["separator_update_mode"] == "csc_column_chunks"
    assert admission.accepted is True


def test_symbolic_nd_frontal_schur_lu_rejects_dense_update_budget() -> None:
    matrix = _nested_dissection_tridiagonal_matrix()

    with pytest.raises(RuntimeError, match="dense separator RHS work budget exceeded"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_nd_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=3,
            symbolic_nd_max_leaf_size=3,
            symbolic_nd_max_depth=4,
            symbolic_nd_separator_width=2,
            symbolic_nd_max_separator_cols=3,
            symbolic_nd_high_degree_cols=0,
            symbolic_nd_regularization_rel=0.0,
            symbolic_nd_max_dense_rhs_entries=1,
        )


def test_symbolic_nd_frontal_schur_lu_parallel_child_setup_preserves_solution() -> None:
    matrix = _nested_dissection_tridiagonal_matrix()
    rhs = np.linspace(2.0, -1.0, matrix.shape[0], dtype=np.float64)

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_nd_frontal_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=3,
        symbolic_nd_max_leaf_size=3,
        symbolic_nd_max_depth=4,
        symbolic_nd_separator_width=2,
        symbolic_nd_max_separator_cols=3,
        symbolic_nd_high_degree_cols=0,
        symbolic_nd_regularization_rel=0.0,
        symbolic_numeric_parallel_workers=2,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-11,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.factor.metadata["parallel_child_workers"] == 2
    assert factor.factor.metadata["parallel_child_nodes"] == 1
    assert factor.factor.metadata["parallel_child_factor_tasks"] == 2
    assert admission.accepted is True


def test_symbolic_nd_frontal_schur_lu_blr_separator_updates_preserve_solution() -> None:
    matrix = _nested_dissection_tridiagonal_matrix()
    rhs = np.linspace(-1.0, 2.0, matrix.shape[0], dtype=np.float64)

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_nd_frontal_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=3,
        symbolic_nd_max_leaf_size=3,
        symbolic_nd_max_depth=4,
        symbolic_nd_separator_width=2,
        symbolic_nd_max_separator_cols=3,
        symbolic_nd_high_degree_cols=0,
        symbolic_nd_regularization_rel=0.0,
        symbolic_nd_compress_updates=True,
        symbolic_nd_parallel_update_workers=2,
        symbolic_blr_frontal_tol=0.0,
        symbolic_blr_frontal_max_rank=8,
        symbolic_blr_frontal_min_cols=1,
        symbolic_blr_frontal_woodbury_max_rank=128,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-11,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.factor.metadata["separator_update_mode"] == "blr_csc_column_chunks"
    assert factor.factor.metadata["parallel_update_workers"] == 2
    assert factor.factor.metadata["blr_update_count"] > 0
    assert factor.factor.metadata["blr_rank_total"] > 0
    assert factor.factor.metadata["blr_error_estimate_max"] == 0.0
    assert factor.factor.metadata["blr_woodbury_rank_total"] > 0
    assert admission.accepted is True


def test_symbolic_nd_frontal_schur_lu_rejects_dense_update_child_budget() -> None:
    matrix = _nested_dissection_tridiagonal_matrix()

    with pytest.raises(RuntimeError, match="dense separator RHS child work budget exceeded"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_nd_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=3,
            symbolic_nd_max_leaf_size=3,
            symbolic_nd_max_depth=4,
            symbolic_nd_separator_width=2,
            symbolic_nd_max_separator_cols=3,
            symbolic_nd_high_degree_cols=0,
            symbolic_nd_regularization_rel=0.0,
            symbolic_nd_max_dense_rhs_entries=1000,
            symbolic_nd_max_dense_rhs_entries_per_child=1,
        )


def test_symbolic_nd_frontal_schur_lu_rejects_oversized_terminal_leaf() -> None:
    matrix = _nested_dissection_tridiagonal_matrix(n=12)

    with pytest.raises(RuntimeError, match="terminal leaf factor size exceeded"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_nd_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=3,
            symbolic_nd_max_leaf_size=2,
            symbolic_nd_max_terminal_factor_size=4,
            symbolic_nd_max_depth=0,
            symbolic_nd_separator_width=2,
            symbolic_nd_max_separator_cols=3,
            symbolic_nd_high_degree_cols=0,
            symbolic_nd_regularization_rel=0.0,
        )


def test_symbolic_nd_frontal_schur_lu_rejects_setup_time_budget() -> None:
    matrix = _nested_dissection_tridiagonal_matrix(n=12)

    with pytest.raises(RuntimeError, match="setup time budget exceeded"):
        factorize_host_sparse_operator(
            matrix,
            kind="symbolic_nd_frontal_schur_lu",
            symbolic_ordering_kind="natural",
            symbolic_block_size=3,
            symbolic_nd_max_leaf_size=3,
            symbolic_nd_max_depth=4,
            symbolic_nd_separator_width=2,
            symbolic_nd_max_separator_cols=8,
            symbolic_nd_high_degree_cols=0,
            symbolic_nd_regularization_rel=0.0,
            symbolic_nd_max_setup_s=1.0e-12,
        )


def test_symbolic_nd_frontal_schur_lu_promotes_cross_graph_edges_to_separator() -> None:
    matrix = _nested_dissection_tridiagonal_matrix(n=12).tolil()
    matrix[1, 10] = 0.25
    matrix[10, 1] = -0.18
    matrix = matrix.tocsr()
    rhs = np.linspace(0.5, -1.5, matrix.shape[0], dtype=np.float64)

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_nd_frontal_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=3,
        symbolic_nd_max_leaf_size=3,
        symbolic_nd_max_depth=4,
        symbolic_nd_separator_width=2,
        symbolic_nd_max_separator_cols=8,
        symbolic_nd_high_degree_cols=0,
        symbolic_nd_regularization_rel=0.0,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-11,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(
        factor.solve(rhs),
        np.linalg.solve(matrix.toarray(), rhs),
        rtol=1.0e-11,
        atol=1.0e-11,
    )
    assert factor.factor.max_separator_count >= 4
    assert factor.factor.metadata["separator_update_mode"] == "csc_column_chunks"
    assert admission.accepted is True


def test_symbolic_block_lu_overlap_retains_boundary_couplings() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 2.5, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [3.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ]
    )
    rhs = np.array([1.0, 2.0, -1.0, 3.0])
    no_overlap = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_block_overlap=0,
    )
    overlap = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_block_overlap=2,
    )

    exact = np.linalg.solve(matrix.toarray(), rhs)
    no_overlap_error = np.linalg.norm(no_overlap.solve(rhs) - exact)
    overlap_error = np.linalg.norm(overlap.solve(rhs) - exact)

    assert overlap.factor.overlap_size == 2
    assert overlap_error < 1.0e-12
    assert overlap_error < no_overlap_error


def test_symbolic_block_lu_coarse_corrects_block_constant_coupling() -> None:
    local = sp.block_diag(
        [
            np.array([[4.0, 1.0], [1.0, 3.0]]),
            np.array([[5.0, -1.0], [-1.0, 2.0]]),
        ],
        format="csr",
    )
    mode = np.array([1.0, 1.0, -1.0, -1.0])
    matrix = (local + 0.75 * sp.csr_matrix(np.outer(mode, mode))).tocsr()
    rhs = np.array([1.0, 2.0, -1.0, 3.0])

    local_factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )
    coarse_factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu_coarse",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_coarse_max_cols=2,
    )

    exact = np.linalg.solve(matrix.toarray(), rhs)
    local_error = np.linalg.norm(local_factor.solve(rhs) - exact)
    coarse_error = np.linalg.norm(coarse_factor.solve(rhs) - exact)
    admission = admit_sparse_factor_against_operator(
        coarse_factor.operator,
        coarse_factor,
        max_relative_residual=2.0e-1,
        min_improvement_vs_identity=10.0,
    )

    assert coarse_factor.kind == "symbolic_block_lu_coarse"
    assert coarse_factor.factor.coarse_size == 2
    assert coarse_error < 0.4 * local_error
    assert admission.accepted is True


def test_symbolic_block_schur_lu_solves_separator_coupled_blocks() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 5.0, -1.0, 1.5],
            [0.0, 0.0, -1.0, 2.0, 0.5],
            [3.0, -2.0, 1.0, 1.0, 7.0],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([1.0, 2.0, -1.0, 3.0, 0.5])

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_schur_tail_size=1,
        symbolic_schur_max_separator_cols=1,
        symbolic_schur_boundary_width=0,
        symbolic_schur_high_degree_cols=0,
        symbolic_schur_regularization_rel=0.0,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )

    np.testing.assert_allclose(factor.solve(rhs), np.linalg.solve(matrix.toarray(), rhs), rtol=1e-12, atol=1e-12)
    assert factor.kind == "symbolic_block_schur_lu"
    assert factor.factor.coarse_size == 1
    assert factor.factor_nbytes_estimate is not None and factor.factor_nbytes_estimate > 0
    assert admission.accepted is True
    assert admission.max_relative_residual < 1.0e-12


def test_symbolic_block_schur_lu_admission_rejects_missing_interior_coupling() -> None:
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 30.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0, -1.0],
            [25.0, 0.0, 5.0, -1.0, 1.5],
            [0.0, 0.0, -1.0, 2.0, 0.5],
            [3.0, -2.0, 1.0, 1.0, 7.0],
        ],
        dtype=np.float64,
    )

    factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_schur_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
        symbolic_schur_tail_size=1,
        symbolic_schur_max_separator_cols=1,
        symbolic_schur_boundary_width=0,
        symbolic_schur_high_degree_cols=0,
        symbolic_schur_regularization_rel=0.0,
    )
    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        max_relative_residual=1.0e-2,
        min_improvement_vs_identity=10.0,
    )

    assert factor.factor.coarse_size == 1
    assert admission.accepted is False
    assert admission.max_relative_residual > 1.0e-2
    assert admission.reason == "residual_or_improvement_gate_failed"


def test_wrap_sparse_factor_with_coarse_correction_uses_supplied_modes() -> None:
    local = sp.block_diag(
        [
            np.array([[4.0, 1.0], [1.0, 3.0]]),
            np.array([[5.0, -1.0], [-1.0, 2.0]]),
        ],
        format="csr",
    )
    mode = np.array([1.0, 1.0, -1.0, -1.0])
    matrix = (local + 0.75 * sp.csr_matrix(np.outer(mode, mode))).tocsr()
    rhs = np.array([1.0, 2.0, -1.0, 3.0])

    local_factor = factorize_host_sparse_operator(
        matrix,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=2,
    )
    exact = np.linalg.solve(matrix.toarray(), rhs)
    local_solution = local_factor.solve(rhs)
    local_error = np.linalg.norm(local_solution - exact)
    error_mode = exact - local_solution
    wrapped = wrap_sparse_factor_with_coarse_correction(
        local_factor,
        sp.csr_matrix(error_mode.reshape((-1, 1))),
    )
    wrapped_error = np.linalg.norm(wrapped.solve(rhs) - exact)

    assert wrapped.kind == local_factor.kind
    assert wrapped.factor_nbytes_estimate is not None
    assert wrapped.factor_nbytes_estimate > local_factor.factor_nbytes_estimate
    assert wrapped_error < 0.4 * local_error


def test_wrap_sparse_factor_with_coarse_correction_noops_for_invalid_basis() -> None:
    factor = factorize_host_sparse_operator(sp.eye(2, format="csr"), kind="jacobi")

    assert wrap_sparse_factor_with_coarse_correction(factor, sp.csr_matrix((2, 0))) is factor
    assert wrap_sparse_factor_with_coarse_correction(factor, sp.csr_matrix((3, 1))) is factor

    metadata = SparseDecision(
        storage_kind="linear_operator",
        reason="operator only",
        backend="cpu",
        shape=(2, 2),
        dense_nbytes=32,
        csr_nbytes_estimate=0,
        nnz_estimate=None,
    )
    operator_only = SparseOperatorBundle(matrix=None, operator=factor.operator.operator, metadata=metadata)
    factor_without_matrix = factorize_host_sparse_operator(sp.eye(2, format="csr"), kind="jacobi")
    factor_without_matrix = factor_without_matrix.__class__(
        factor=factor_without_matrix.factor,
        operator=operator_only,
        metadata=metadata,
        kind=factor_without_matrix.kind,
        factor_nbytes_estimate=factor_without_matrix.factor_nbytes_estimate,
        factor_nnz_estimate=factor_without_matrix.factor_nnz_estimate,
        factor_s=factor_without_matrix.factor_s,
    )
    assert wrap_sparse_factor_with_coarse_correction(factor_without_matrix, sp.eye(2, format="csr")) is factor_without_matrix


def test_factorize_host_sparse_operator_jacobi_floors_zero_diagonal() -> None:
    matrix = sp.csr_matrix([[0.0, 1.0], [2.0, 4.0]])
    factor = factorize_host_sparse_operator(matrix, kind="jacobi")

    out = factor.solve(np.asarray([1.0, 8.0]))
    assert np.isfinite(out).all()
    assert out[0] > 1.0e10
    assert out[1] == pytest.approx(2.0)


def test_factorize_host_sparse_operator_ilu_can_retry_singular_matrix(monkeypatch) -> None:
    matrix = sp.csr_matrix((2, 2), dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_ILU_ATTEMPTS", "2")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_ILU_SINGULAR_REG_REL", "1e-3")

    factor = factorize_host_sparse_operator(matrix, kind="ilu")

    out = factor.solve(np.asarray([1.0, -2.0]))
    assert np.isfinite(out).all()
    assert factor.kind == "ilu"


def test_factorize_host_sparse_operator_reports_singular_branch_actionably() -> None:
    matrix = sp.csr_matrix([[1.0, 0.0], [0.0, 0.0]])

    with pytest.raises(RuntimeError, match="Host sparse factorization failed"):
        factorize_host_sparse_operator(matrix, kind="lu")


def test_host_direct_refinement_keeps_only_residual_improving_steps() -> None:
    matrix = np.diag([2.0, 4.0])
    rhs = np.asarray([2.0, 8.0])
    calls: list[np.ndarray] = []

    def improving_solve(load: np.ndarray) -> np.ndarray:
        calls.append(np.asarray(load, dtype=np.float64).copy())
        # First solve is deliberately under-corrected; residual solve is exact.
        if len(calls) == 1:
            return np.asarray([0.5, 1.5])
        load_np = np.asarray(load, dtype=np.float64)
        return np.asarray([0.5 * load_np[0], 0.25 * load_np[1]])

    x, residual = host_direct_solve_with_refinement(
        factor_solve=improving_solve,
        operator_matrix=matrix,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )

    np.testing.assert_allclose(x, np.asarray([1.0, 2.0]))
    assert residual == pytest.approx(0.0)
    assert len(calls) == 2

    def worsening_solve(load: np.ndarray) -> np.ndarray:
        calls.append(np.asarray(load, dtype=np.float64).copy())
        if len(calls) == 1:
            return np.asarray([0.5, 1.5])
        return -10.0 * np.asarray(load, dtype=np.float64)

    calls.clear()
    x_worse, residual_worse = host_direct_solve_with_refinement(
        factor_solve=worsening_solve,
        operator_matrix=matrix,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=2,
    )
    np.testing.assert_allclose(x_worse, np.asarray([0.5, 1.5]))
    assert residual_worse == pytest.approx(np.linalg.norm(rhs - matrix @ x_worse))
    assert len(calls) == 2


def test_host_sparse_direct_refinement_and_polish_callbacks() -> None:
    matrix = sp.diags([2.0, 4.0], format="csr")

    class _Factor:
        def __init__(self) -> None:
            self.calls = 0

        def solve(self, load: np.ndarray) -> np.ndarray:
            self.calls += 1
            load_np = np.asarray(load, dtype=np.float64)
            if self.calls == 1:
                return np.asarray([0.5, 1.5])
            return np.asarray([0.5 * load_np[0], 0.25 * load_np[1]])

    factor = _Factor()
    rhs = np.asarray([2.0, 8.0])
    x, residual = host_sparse_direct_solve_with_refinement(
        ilu=factor,
        a_csr_full=matrix,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=2,
    )
    np.testing.assert_allclose(x, np.asarray([1.0, 2.0]))
    assert residual == pytest.approx(0.0)

    polish_seen: dict[str, object] = {}

    def fake_gmres_solver(**kwargs):
        polish_seen.update(kwargs)
        return np.asarray([1.0, 2.0]), 0.0, [1.0, 0.0]

    x_polish, residual_polish = host_sparse_direct_polish(
        matvec_fn=lambda x_vec: jnp.asarray(matrix @ np.asarray(x_vec), dtype=jnp.float64),
        rhs_vec=jnp.asarray(rhs, dtype=jnp.float64),
        x0_np=np.asarray([0.0, 0.0]),
        ilu=factor,
        factor_dtype=np.dtype(np.float64),
        tol=1.0e-10,
        atol=1.0e-12,
        restart=5,
        maxiter=7,
        precondition_side="left",
        gmres_solver=fake_gmres_solver,
    )

    np.testing.assert_allclose(x_polish, np.asarray([1.0, 2.0]))
    assert residual_polish == pytest.approx(0.0)
    assert polish_seen["restart"] == 5
    assert polish_seen["maxiter"] == 7
    assert polish_seen["precondition_side"] == "left"
    preconditioned = polish_seen["preconditioner"](jnp.asarray([2.0, 4.0]))
    np.testing.assert_allclose(np.asarray(preconditioned), np.asarray([1.0, 1.0]))
