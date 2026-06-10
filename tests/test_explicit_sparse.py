from __future__ import annotations

import numpy as np
import jax.numpy as jnp
import pytest
import scipy.sparse as sp

from sfincs_jax.explicit_sparse import (
    SparseOperatorBundle,
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    build_operator_from_blocks,
    build_operator_from_dense,
    build_operator_from_matvec,
    build_operator_from_pattern,
    choose_storage_kind,
    color_pattern_columns,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
    estimate_superlu_factor_storage,
    factorize_host_sparse_operator,
    wrap_sparse_factor_with_coarse_correction,
)


def test_storage_estimates_are_consistent() -> None:
    assert estimate_dense_nbytes((3, 4), np.float64) == 3 * 4 * 8
    assert estimate_csr_nbytes((3, 4), 5, data_dtype=np.float64, index_dtype=np.int32) == 5 * 12 + 4 * 4


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
