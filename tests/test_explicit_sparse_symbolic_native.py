from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import sfincs_jax.solvers.explicit_sparse as explicit_sparse


class _FailingFactor:
    def solve(self, _rhs):
        raise RuntimeError("factor failed")


class _NaNFactor:
    def solve(self, rhs):
        return np.full_like(np.asarray(rhs, dtype=np.float64), np.nan)


def _spd_matrix() -> sparse.csr_matrix:
    return sparse.csr_matrix(
        np.asarray(
            [
                [4.0, -1.0, 0.2, 0.0, 0.0, 0.0],
                [-1.0, 4.5, -0.8, 0.0, 0.0, 0.0],
                [0.2, -0.8, 5.0, -0.7, 0.1, 0.0],
                [0.0, 0.0, -0.7, 4.8, -0.6, 0.3],
                [0.0, 0.0, 0.1, -0.6, 4.4, -0.9],
                [0.0, 0.0, 0.0, 0.3, -0.9, 4.2],
            ],
            dtype=np.float64,
        )
    )


def _analysis(matrix: sparse.spmatrix, *, block_size: int = 2) -> explicit_sparse.SparseSymbolicAnalysis:
    return explicit_sparse.analyze_sparse_symbolic_structure(
        matrix,
        ordering_kind="natural",
        block_size_target=block_size,
    )


def test_symbolic_analysis_and_block_factors_are_finite_on_small_spd_matrix() -> None:
    matrix = _spd_matrix()
    analysis = _analysis(matrix)
    rhs = np.linspace(1.0, 2.0, matrix.shape[0])

    assert analysis.cache_key()[0] == matrix.shape
    analysis_dict = analysis.to_dict(include_permutation=True)
    assert analysis_dict["diagonal_missing"] == 0
    assert analysis_dict["block_count"] == 3
    assert analysis_dict["permutation"] == list(range(matrix.shape[0]))

    block_factor, block_bytes, block_nnz = explicit_sparse._build_symbolic_block_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        overlap_size=1,
    )
    block_solution = block_factor.solve(rhs)

    assert block_factor.overlap_size == 1
    assert block_bytes > 0
    assert block_nnz > 0
    assert block_solution.shape == rhs.shape
    assert np.all(np.isfinite(block_solution))

    coarse_factor, coarse_bytes, coarse_nnz = explicit_sparse._build_symbolic_block_coarse_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        overlap_size=1,
        coarse_max_cols=2,
        coarse_probe_cols=2,
        coarse_damping=0.75,
    )
    coarse_solution = coarse_factor.solve(rhs)

    assert 0 < coarse_factor.coarse_size <= 4
    assert coarse_factor.overlap_size == 1
    assert coarse_factor.coarse_matrix_nnz > 0
    assert coarse_bytes >= block_bytes
    assert coarse_nnz >= block_nnz
    assert coarse_solution.shape == rhs.shape
    assert np.all(np.isfinite(coarse_solution))


def test_symbolic_factor_wrappers_fail_soft_and_clean_nonfinite_values() -> None:
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    analysis = _analysis(matrix, block_size=2)
    rhs = np.asarray([1.0, -2.0, 3.0])

    block_factor = explicit_sparse._SymbolicBlockFactor(
        blocks=((0, 2, 0, 2, _NaNFactor()), (2, 3, 2, 3, _FailingFactor())),
        analysis=analysis,
        permutation=np.arange(3, dtype=np.int64),
        inverse_permutation=np.arange(3, dtype=np.int64),
        dtype=np.dtype(np.float64),
        overlap_size=0,
    )
    np.testing.assert_allclose(block_factor.solve(rhs), np.asarray([0.0, 0.0, 3.0]))

    coarse = explicit_sparse._SymbolicBlockCoarseFactor(
        local_factor=block_factor,
        matrix=matrix,
        coarse_basis=sparse.csr_matrix(np.asarray([[1.0], [1.0], [1.0]])),
        coarse_factor=_FailingFactor(),
        coarse_matrix_nnz=1,
        dtype=np.dtype(np.float64),
        coarse_damping=1.0,
    )
    np.testing.assert_allclose(coarse.solve(rhs), np.asarray([0.0, 0.0, 3.0]))

    coarse_empty = explicit_sparse._SymbolicBlockCoarseFactor(
        local_factor=block_factor,
        matrix=matrix,
        coarse_basis=sparse.csr_matrix((3, 0), dtype=np.float64),
        coarse_factor=_FailingFactor(),
        coarse_matrix_nnz=0,
        dtype=np.dtype(np.float64),
    )
    np.testing.assert_allclose(coarse_empty.solve(rhs), np.asarray([0.0, 0.0, 3.0]))


def test_sparse_coarse_and_residual_polish_wrappers_have_bounded_fallbacks() -> None:
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    rhs = np.asarray([2.0, -1.0, 0.5])

    coarse = explicit_sparse._SparseCoarseCorrectionFactor(
        base_factor=_FailingFactor(),
        matrix=matrix,
        coarse_basis=sparse.csr_matrix(np.asarray([[1.0], [0.0], [1.0]])),
        coarse_factor=_FailingFactor(),
        dtype=np.dtype(np.float64),
        damping=0.5,
    )
    np.testing.assert_allclose(coarse.solve(rhs), rhs)

    coarse_nonfinite = explicit_sparse._SparseCoarseCorrectionFactor(
        base_factor=_NaNFactor(),
        matrix=matrix,
        coarse_basis=sparse.csr_matrix(np.asarray([[1.0], [1.0], [1.0]])),
        coarse_factor=explicit_sparse._DenseInverseFactor(np.eye(1)),
        dtype=np.dtype(np.float64),
    )
    np.testing.assert_allclose(coarse_nonfinite.solve(rhs), rhs)

    polish = explicit_sparse._SparseResidualPolishFactor(
        base_factor=explicit_sparse._DenseInverseFactor(0.5 * np.eye(3)),
        matrix=matrix,
        dtype=np.dtype(np.float64),
        steps=3,
        damping=1.0,
    )
    polished = polish.solve(rhs)
    assert np.linalg.norm(matrix @ polished - rhs) < np.linalg.norm(matrix @ (0.5 * rhs) - rhs)

    matrix_rhs = np.column_stack([rhs, 2.0 * rhs])
    polished_matrix = polish.solve(matrix_rhs)
    assert polished_matrix.shape == matrix_rhs.shape
    assert np.all(np.isfinite(polished_matrix))

    polish_nonfinite = explicit_sparse._SparseResidualPolishFactor(
        base_factor=_NaNFactor(),
        matrix=matrix,
        dtype=np.dtype(np.float64),
        steps=2,
    )
    np.testing.assert_allclose(polish_nonfinite.solve(rhs), np.zeros_like(rhs))


def test_blr_schur_factor_uses_woodbury_and_gmres_fallback_paths() -> None:
    dtype = np.dtype(np.float64)
    base_matrix = sparse.eye(3, format="csr", dtype=dtype) * 3.0
    base_factor = explicit_sparse._DenseInverseFactor(np.eye(3, dtype=dtype) / 3.0)
    update = explicit_sparse._compress_update_block(
        np.asarray([[0.2, 0.0], [0.0, 0.1], [0.1, 0.2]], dtype=dtype),
        columns=np.asarray([0, 2], dtype=np.int64),
        tol=1.0e-12,
        max_rank=2,
        dtype=dtype,
    )
    rank0 = explicit_sparse._compress_update_block(
        np.zeros((3, 0), dtype=dtype),
        columns=np.asarray([], dtype=np.int64),
        tol=1.0e-12,
        max_rank=2,
        dtype=dtype,
    )
    z, vt, core_inv, condition = explicit_sparse._build_blr_woodbury_state(
        base_factor,
        (update, rank0),
        separator_size=3,
        dtype=dtype,
        max_rank=4,
        max_condition=1.0e8,
    )

    assert update.rank > 0
    assert rank0.rank == 0
    assert z is not None and vt is not None and core_inv is not None
    assert condition is not None and np.isfinite(condition)

    factor = explicit_sparse._BLRSchurFactor(
        base_matrix=base_matrix,
        base_factor=base_factor,
        updates=(update, rank0),
        dtype=dtype,
        rtol=1.0e-10,
        atol=0.0,
        maxiter=20,
        restart=3,
        woodbury_z=z,
        woodbury_vt=vt,
        woodbury_core_inverse=core_inv,
        woodbury_condition=condition,
    )
    rhs = np.asarray([1.0, -0.5, 0.25], dtype=dtype)
    woodbury_solution = factor.solve(rhs)

    assert factor.woodbury_rank == update.rank
    assert factor.woodbury_nbytes > 0
    assert np.all(np.isfinite(woodbury_solution))
    np.testing.assert_allclose(factor.matvec(woodbury_solution), rhs, rtol=1.0e-8, atol=1.0e-8)

    gmres_factor = explicit_sparse._BLRSchurFactor(
        base_matrix=base_matrix,
        base_factor=base_factor,
        updates=(update,),
        dtype=dtype,
        rtol=1.0e-10,
        atol=0.0,
        maxiter=20,
        restart=3,
    )
    gmres_solution = gmres_factor.solve(np.column_stack([rhs, 2.0 * rhs]))
    assert gmres_solution.shape == (3, 2)
    assert np.all(np.isfinite(gmres_solution))

    rejected = explicit_sparse._build_blr_woodbury_state(
        base_factor,
        (update,),
        separator_size=3,
        dtype=dtype,
        max_rank=0,
        max_condition=1.0e8,
    )
    assert rejected == (None, None, None, None)


def test_symbolic_schur_and_blr_edge_fallbacks_are_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    dtype = np.dtype(np.float64)
    matrix = sparse.eye(3, format="csr", dtype=dtype)
    analysis = _analysis(matrix, block_size=2)
    rhs = np.asarray([2.0, 4.0, -3.0], dtype=dtype)

    no_separator = explicit_sparse._SymbolicBlockSchurFactor(
        blocks=(
            explicit_sparse._SymbolicSchurBlock(
                indices=np.asarray([0, 1], dtype=np.int64),
                factor=explicit_sparse._DenseInverseFactor(0.5 * np.eye(2, dtype=dtype)),
                b_to_separator=sparse.csr_matrix((2, 0), dtype=dtype),
                c_from_separator=sparse.csr_matrix((0, 2), dtype=dtype),
            ),
            explicit_sparse._SymbolicSchurBlock(
                indices=np.asarray([2], dtype=np.int64),
                factor=_FailingFactor(),
                b_to_separator=sparse.csr_matrix((1, 0), dtype=dtype),
                c_from_separator=sparse.csr_matrix((0, 1), dtype=dtype),
            ),
        ),
        separator_indices=np.asarray([], dtype=np.int64),
        schur_factor=_FailingFactor(),
        dtype=dtype,
        n=3,
        analysis=analysis,
        separator_count=0,
    )
    np.testing.assert_allclose(no_separator.solve(rhs), np.asarray([1.0, 2.0, -3.0], dtype=dtype))

    empty_blr = explicit_sparse._BLRSchurFactor(
        base_matrix=sparse.csr_matrix((0, 0), dtype=dtype),
        base_factor=_FailingFactor(),
        updates=tuple(),
        dtype=dtype,
        rtol=1.0e-10,
        atol=0.0,
        maxiter=2,
        restart=1,
    )
    assert empty_blr.solve(np.asarray([], dtype=dtype)).shape == (0,)

    update = explicit_sparse._BLRUpdateBlock(
        columns=np.asarray([0], dtype=np.int64),
        u=np.asarray([[0.1], [0.0]], dtype=dtype),
        vt=np.asarray([[1.0]], dtype=dtype),
        original_shape=(2, 1),
        rank=1,
        relative_error_estimate=0.0,
    )
    invalid_woodbury = explicit_sparse._BLRSchurFactor(
        base_matrix=sparse.eye(2, format="csr", dtype=dtype),
        base_factor=explicit_sparse._DenseInverseFactor(np.eye(2, dtype=dtype)),
        updates=(update,),
        dtype=dtype,
        rtol=1.0e-10,
        atol=0.0,
        maxiter=3,
        restart=2,
        woodbury_z=np.eye(2, dtype=dtype),
        woodbury_vt=np.ones((1, 2), dtype=dtype),
        woodbury_core_inverse=np.ones((3, 3), dtype=dtype),
    )
    invalid_solution = invalid_woodbury.solve(np.asarray([1.0, -1.0], dtype=dtype))
    assert np.all(np.isfinite(invalid_solution))

    import scipy.sparse.linalg as scipy_sparse_linalg

    monkeypatch.setattr(
        scipy_sparse_linalg,
        "gmres",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("synthetic gmres failure")),
    )
    gmres_failure = explicit_sparse._BLRSchurFactor(
        base_matrix=sparse.eye(2, format="csr", dtype=dtype),
        base_factor=explicit_sparse._DenseInverseFactor(0.25 * np.eye(2, dtype=dtype)),
        updates=(update,),
        dtype=dtype,
        rtol=1.0e-10,
        atol=0.0,
        maxiter=3,
        restart=2,
    )
    np.testing.assert_allclose(
        gmres_failure.solve(np.asarray([4.0, 8.0], dtype=dtype)),
        np.asarray([1.0, 2.0], dtype=dtype),
    )


def test_symbolic_schur_superblock_and_nd_factors_solve_with_small_matrices() -> None:
    matrix = _spd_matrix()
    analysis = explicit_sparse.analyze_sparse_symbolic_structure(
        matrix,
        ordering_kind="nested_dissection",
        block_size_target=2,
    )
    rhs = np.linspace(0.5, 1.5, matrix.shape[0])

    separator = explicit_sparse._select_symbolic_schur_separator(
        matrix,
        analysis=analysis,
        max_separator_cols=3,
        tail_size=1,
        boundary_width=1,
        high_degree_cols=1,
    )
    assert 0 < separator.size <= 3

    schur_factor, schur_bytes, schur_nnz = explicit_sparse._build_symbolic_block_schur_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        max_separator_cols=3,
        tail_size=1,
        boundary_width=1,
        high_degree_cols=1,
    )
    schur_solution = schur_factor.solve(rhs)
    assert schur_factor.coarse_size == schur_factor.separator_count
    assert schur_factor.overlap_size == 0
    assert schur_bytes > 0
    assert schur_nnz > 0
    assert schur_solution.shape == rhs.shape
    assert np.all(np.isfinite(schur_solution))

    frontal_factor, frontal_bytes, frontal_nnz = explicit_sparse._build_symbolic_frontal_schur_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        max_superblock_size=4,
        max_superblock_blocks=2,
        max_separator_cols=3,
        high_degree_cols=1,
        compress_updates=True,
        max_dense_rhs_cols_per_block=1,
        blr_max_rank=2,
        blr_min_cols=1,
    )
    frontal_solution = frontal_factor.solve(rhs)
    assert frontal_factor.coarse_size == frontal_factor.separator_count
    assert frontal_bytes > 0
    assert frontal_nnz > 0
    assert np.all(np.isfinite(frontal_solution))

    superblock_factor, superblock_bytes, superblock_nnz = explicit_sparse._build_symbolic_superblock_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        max_superblock_size=4,
        max_superblock_blocks=2,
        min_cross_nnz=1,
        parallel_workers=2,
    )
    super_solution = superblock_factor.solve(rhs)
    assert superblock_factor.superblock_count >= 1
    assert superblock_factor.base_block_count == analysis.block_count
    assert superblock_factor.parallel_workers >= 1
    assert superblock_bytes > 0
    assert superblock_nnz > 0
    assert np.all(np.isfinite(super_solution))

    nd_factor, nd_bytes, nd_nnz = explicit_sparse._build_symbolic_nd_frontal_schur_factor(
        matrix,
        analysis=analysis,
        diag_pivot_thresh=1.0,
        max_leaf_size=2,
        max_terminal_factor_size=3,
        max_depth=2,
        separator_width=1,
        max_separator_cols=3,
        high_degree_cols=1,
        max_dense_rhs_cols_per_child=1,
        compress_updates=True,
        blr_max_rank=2,
        blr_min_cols=1,
    )
    nd_solution = nd_factor.solve(rhs)
    assert nd_factor.global_size == matrix.shape[0]
    assert nd_factor.max_depth_reached <= 2
    assert nd_bytes > 0
    assert nd_nnz > 0
    assert np.all(np.isfinite(nd_solution))


def test_regularized_factorization_and_admission_gates_fail_closed() -> None:
    singular = sparse.csc_matrix((3, 3), dtype=np.float64)
    factor, nbytes, nnz, failures = explicit_sparse._factor_csc_with_regularized_fallback(
        singular,
        dtype=np.dtype(np.float64),
        diag_pivot_thresh=1.0,
        regularization_rel=0.0,
    )
    assert isinstance(factor, explicit_sparse._JacobiFactor)
    assert failures == 1
    assert nbytes > 0
    assert nnz == 3
    assert np.all(np.isfinite(factor.solve(np.ones(3))))

    missing = explicit_sparse.admit_sparse_factor_against_operator(None, factor)
    assert not missing.accepted
    assert missing.reason == "missing_operator_matrix"

    nonsquare = explicit_sparse.admit_sparse_factor_against_operator(
        sparse.csr_matrix(np.ones((2, 3))),
        factor,
    )
    assert not nonsquare.accepted
    assert nonsquare.reason == "operator_not_square"

    identity = sparse.eye(3, format="csr", dtype=np.float64)
    identity_bundle = explicit_sparse.SparseOperatorBundle(
        matrix=identity,
        operator=explicit_sparse._operator_from_matrix(identity),
        metadata=explicit_sparse.SparseDecision(
            storage_kind="csr",
            reason="unit",
            backend="cpu",
            shape=(3, 3),
            dense_nbytes=72,
            csr_nbytes_estimate=52,
            nnz_estimate=3,
        ),
    )
    exact_factor = explicit_sparse.SparseFactorBundle(
        factor=explicit_sparse._DenseInverseFactor(np.eye(3)),
        operator=identity_bundle,
        metadata=identity_bundle.metadata,
        kind="lu",
    )
    admission = explicit_sparse.admit_sparse_factor_against_operator(
        identity_bundle,
        exact_factor,
        probes=np.eye(3),
        max_relative_residual=1.0e-12,
        min_improvement_vs_identity=1.0,
    )
    assert admission.accepted
    assert admission.reason == "accepted"
    assert admission.to_dict()["probe_count"] == 3

    wrapped = explicit_sparse.wrap_sparse_factor_with_coarse_correction(
        exact_factor,
        sparse.csr_matrix(np.asarray([[1.0], [0.0], [1.0]])),
    )
    assert wrapped.factor_nbytes_estimate is None
    assert wrapped.solve(np.asarray([1.0, 2.0, 3.0])).shape == (3,)

    unchanged = explicit_sparse.wrap_sparse_factor_with_coarse_correction(
        exact_factor,
        sparse.csr_matrix((2, 0), dtype=np.float64),
    )
    assert unchanged is exact_factor


def test_deterministic_probe_and_symbolic_helpers_cover_edge_cases() -> None:
    probes = explicit_sparse.deterministic_sparse_probe_matrix(0, count=3)
    assert probes.shape == (0, 3)

    one_probe = explicit_sparse.deterministic_sparse_probe_matrix(1, count=4)
    assert one_probe.shape == (1, 4)
    assert np.all(np.isfinite(one_probe))

    empty_analysis = explicit_sparse.analyze_sparse_symbolic_structure(
        sparse.csr_matrix((0, 0), dtype=np.float64),
        ordering_kind="rcm",
        block_size_target=4,
    )
    assert empty_analysis.to_dict(include_permutation=True)["permutation"] == []

    assert explicit_sparse._safe_percentile(np.asarray([], dtype=np.float64), 95.0) == 0.0
    assert explicit_sparse._bandwidth_and_profile(sparse.csr_matrix((3, 3), dtype=np.float64)) == (0, 0)

    dsu = explicit_sparse._DisjointSet(np.asarray([2, 3, 4]))
    assert dsu.union_if_fits(0, 1, max_rows=5, max_blocks=2)
    assert not dsu.union_if_fits(0, 2, max_rows=5, max_blocks=3)
    assert not dsu.union_if_fits(1, 0, max_rows=10, max_blocks=3)

    bad_update = np.ones((2, 2, 1))
    with pytest.raises(ValueError, match="2D matrix"):
        explicit_sparse._compress_update_block(
            bad_update,
            columns=np.asarray([0, 1]),
            tol=1.0e-6,
            max_rank=2,
            dtype=np.dtype(np.float64),
        )

    with pytest.raises(ValueError, match="column list"):
        explicit_sparse._compress_update_block(
            np.ones((2, 2)),
            columns=np.asarray([0]),
            tol=1.0e-6,
            max_rank=2,
            dtype=np.dtype(np.float64),
        )

    payload = explicit_sparse.SparseDecision(
        storage_kind="dense",
        reason="unit",
        backend="cpu",
        shape=(1, 1),
        dense_nbytes=8,
        csr_nbytes_estimate=16,
        nnz_estimate=1,
        block_cols=1,
        drop_tol=0.0,
    ).to_dict()
    assert payload["storage_kind"] == "dense"
    assert payload["block_cols"] == 1
