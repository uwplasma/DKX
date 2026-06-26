from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

from sfincs_jax.solvers.explicit_sparse import SparseDecision, SparseOperatorBundle
from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.operators.profile_true_operator_rescue import (
    _ResidualCoarseHostSparsePreconditionerBundle,
    _ResidualWindowHostSparsePreconditionerBundle,
    _ReusableTrueActionColumnCache,
    _TrueOperatorActiveSubmatrixPreconditionerBundle,
    _TrueOperatorCoupledCoarseLSQPreconditionerBundle,
    _TrueOperatorWindowLSQPreconditionerBundle,
    _expand_sparse_graph_positions,
    _parse_true_operator_window_specs,
    _rhs1_additive_rescue_nbytes,
    _rhs1_active_reduced_residual_diagnostics,
    _sparse_factor_nbytes_estimate,
    _true_operator_window_positions_from_residual,
    _try_build_residual_coarse_host_sparse_preconditioner,
    _try_build_residual_window_host_sparse_preconditioner,
)


class _IdentityFactor:
    factor_nbytes_estimate = 128

    def solve(self, rhs):
        return np.asarray(rhs, dtype=np.float64)


class _HalfFactor:
    kind = "half_identity"
    factor_nbytes_estimate = 0
    factor_nnz_estimate = 4
    factor_s = 0.0

    def solve(self, rhs):
        return 0.5 * np.asarray(rhs, dtype=np.float64)


class _ZeroFactor:
    factor_nbytes_estimate = 0

    def solve(self, rhs):
        return np.zeros_like(np.asarray(rhs, dtype=np.float64))


def _layout_with_tail() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=1,
        extra_size=1,
        total_size=8,
        constraint_scheme=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def test_residual_coarse_bundle_solves_identity_residual() -> None:
    matrix = sp.eye(3, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit-test",
            backend="cpu",
            shape=(3, 3),
            dense_nbytes=72,
            csr_nbytes_estimate=64,
            nnz_estimate=3,
        ),
    )

    bundle = _ResidualCoarseHostSparsePreconditionerBundle(
        base_factor=_IdentityFactor(),
        operator=operator,
        z_basis=np.eye(3, dtype=np.float64),
        az_basis=np.eye(3, dtype=np.float64),
        coarse_inverse=np.eye(3, dtype=np.float64),
        kind="identity_coarse",
    )

    np.testing.assert_allclose(bundle.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, -2.0, 3.0])


def test_residual_window_bundle_supports_additive_and_lsq_modes() -> None:
    matrix = sp.eye(3, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit-test",
            backend="cpu",
            shape=(3, 3),
            dense_nbytes=72,
            csr_nbytes_estimate=64,
            nnz_estimate=3,
        ),
    )

    additive = _ResidualWindowHostSparsePreconditionerBundle(
        base_factor=_ZeroFactor(),
        operator=operator,
        window_positions=(np.asarray([0, 2]),),
        window_factors=(_IdentityFactor(),),
        kind="window_additive",
        coefficient_mode="additive",
    )
    np.testing.assert_allclose(additive.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])

    least_squares = _ResidualWindowHostSparsePreconditionerBundle(
        base_factor=_ZeroFactor(),
        operator=operator,
        window_positions=(np.asarray([0, 2]),),
        window_factors=(_IdentityFactor(),),
        kind="window_lsq",
        coefficient_mode="least_squares",
        regularization=0.0,
    )
    np.testing.assert_allclose(least_squares.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])


def test_reusable_true_action_column_cache_reuses_one_hot_batches() -> None:
    matrix = np.asarray(
        [
            [2.0, -0.5, 0.25],
            [0.1, 1.8, -0.4],
            [-0.2, 0.3, 1.6],
        ],
        dtype=np.float64,
    )
    calls = {"matmat": 0}

    def true_matmat(x):
        calls["matmat"] += 1
        return matrix @ np.asarray(x, dtype=np.float64)

    cache = _ReusableTrueActionColumnCache(
        true_matvec=lambda x: matrix @ np.asarray(x, dtype=np.float64),
        true_matmat=true_matmat,
        n=3,
        max_nbytes=1024 * 1024,
        enabled=True,
    )
    basis_a = np.zeros((3, 2), dtype=np.float64)
    basis_a[[0, 2], [0, 1]] = 1.0
    basis_b = np.zeros((3, 2), dtype=np.float64)
    basis_b[[2, 1], [0, 1]] = 1.0

    np.testing.assert_allclose(cache.matmat(basis_a), matrix @ basis_a)
    np.testing.assert_allclose(cache.matmat(basis_b), matrix @ basis_b)

    metadata = cache.metadata()
    assert metadata["hits"] == 1
    assert metadata["misses"] == 3
    assert metadata["stored_columns"] == 3
    assert metadata["batches"] == 2
    assert calls["matmat"] == 1


def test_reusable_true_action_column_cache_bypasses_non_one_hot_or_disabled() -> None:
    matrix = np.diag([2.0, 3.0, 4.0])
    cache = _ReusableTrueActionColumnCache(
        true_matvec=lambda x: matrix @ np.asarray(x, dtype=np.float64),
        true_matmat=None,
        n=3,
        max_nbytes=0,
        enabled=False,
    )

    dense = np.asarray([[1.0, 0.5], [0.0, 0.5], [0.0, 0.0]], dtype=np.float64)
    np.testing.assert_allclose(cache.matmat(dense), matrix @ dense)
    assert cache.metadata()["bypass_calls"] == 1
    assert cache.metadata()["stored_columns"] == 0


def test_true_operator_lsq_bundles_reduce_identity_residuals() -> None:
    def true_matvec(x):
        return np.asarray(x, dtype=np.float64)

    z_basis = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)

    window = _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        window_positions=np.asarray([0, 2]),
        a_window=z_basis,
        inv_column_scale=np.ones(2, dtype=np.float64),
        solve_normal=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="window_lsq",
    )
    np.testing.assert_allclose(window.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])

    active_block = _TrueOperatorActiveSubmatrixPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        block_positions=np.asarray([0, 2]),
        a_window=z_basis,
        solve_block=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="active_block",
        damping=True,
    )
    np.testing.assert_allclose(active_block.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])

    coupled = _TrueOperatorCoupledCoarseLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        z_basis=z_basis,
        a_basis=z_basis,
        inv_column_scale=np.ones(2, dtype=np.float64),
        solve_normal=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="coupled_lsq",
        damping=True,
    )
    np.testing.assert_allclose(coupled.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])


def test_rescue_budget_and_sparse_factor_memory_estimate() -> None:
    factor = SimpleNamespace(factor_nbytes_estimate=5_000_000_000)
    assert _rhs1_additive_rescue_nbytes(factor, 512.0) == 5_000_000_000 + 512 * 1024 * 1024
    assert _rhs1_additive_rescue_nbytes(factor, 0.0) == 0

    sparse_factor = SimpleNamespace(
        L=sp.eye(4, format="csr"),
        U=sp.eye(4, format="csr"),
    )
    assert _sparse_factor_nbytes_estimate(sparse_factor) > 0


def test_graph_expansion_and_window_selection_are_layout_aware() -> None:
    graph = sp.csr_matrix(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(
        _expand_sparse_graph_positions(graph, np.asarray([1]), depth=1, max_size=4),
        np.asarray([0, 1, 2]),
    )
    assert _expand_sparse_graph_positions(graph, np.asarray([1]), depth=2, max_size=3) is None

    layout = _layout_with_tail()
    specs = _parse_true_operator_window_specs("0:0:1, invalid, 0/1/2, 2:0:0", layout=layout)
    assert specs == ((0, 0, 1), (0, 1, 2))

    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.asarray([0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 9.0, -1.0]),
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        include_tail=True,
        explicit_specs=specs[:1],
    )
    np.testing.assert_array_equal(positions, np.asarray([1, 6, 7]))
    assert metadata[0]["species"] == 0
    assert metadata[0]["x_center"] == 0
    assert metadata[0]["ell_center"] == 1


def test_residual_coarse_builder_corrects_failed_identity_factor() -> None:
    matrix = sp.eye(4, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit-test",
            backend="cpu",
            shape=(4, 4),
            dense_nbytes=4 * 4 * 8,
            csr_nbytes_estimate=128,
            nnz_estimate=4,
        ),
    )
    rhs = np.asarray([1.0, -2.0, 3.0, -4.0], dtype=np.float64)
    failed_residual = rhs - operator.matvec(_HalfFactor().solve(rhs))

    bundle = _try_build_residual_coarse_host_sparse_preconditioner(
        operator_bundle=operator,
        factor_bundle=_HalfFactor(),
        residual=failed_residual,
        max_rank=2,
        max_nbytes=1024 * 1024,
        regularization=0.0,
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["rank"] == 1
    np.testing.assert_allclose(operator.matvec(bundle.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_residual_window_builder_corrects_targeted_kinetic_window() -> None:
    matrix = sp.eye(4, format="csr")
    operator = SparseOperatorBundle(
        matrix=matrix,
        operator=aslinearoperator(matrix),
        metadata=SparseDecision(
            storage_kind="csr",
            reason="unit-test",
            backend="cpu",
            shape=(4, 4),
            dense_nbytes=4 * 4 * 8,
            csr_nbytes_estimate=128,
            nnz_estimate=4,
        ),
    )
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    rhs = np.asarray([0.0, 2.0, 0.0, 0.0], dtype=np.float64)
    failed_residual = rhs - operator.matvec(_HalfFactor().solve(rhs))

    bundle = _try_build_residual_window_host_sparse_preconditioner(
        operator_bundle=operator,
        factor_bundle=_HalfFactor(),
        residual=failed_residual,
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        coefficient_mode="least_squares",
        combine_mode="independent",
        interface_depth=0,
        max_window_size=16,
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["window_count"] == 1
    np.testing.assert_allclose(operator.matvec(bundle.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_active_reduced_residual_diagnostics_splits_components() -> None:
    layout = _layout_with_tail()
    diagnostics = _rhs1_active_reduced_residual_diagnostics(
        residual=np.asarray([0.0, 3.0, 0.0, 0.0, -4.0, 0.0, 2.0, -1.0]),
        layout=layout,
        active_indices=None,
        top_k=2,
    )

    assert diagnostics["selected"] is True
    assert diagnostics["component_norms"]["kinetic"]["energy_fraction"] > 0.0
    assert diagnostics["component_norms"]["phi1"]["energy_fraction"] > 0.0
    assert diagnostics["component_norms"]["extra"]["energy_fraction"] > 0.0
    assert diagnostics["top_ell"][0]["label"] in {"1", "2"}

    mismatch = _rhs1_active_reduced_residual_diagnostics(
        residual=np.asarray([1.0, 2.0]),
        layout=layout,
        active_indices=np.asarray([0]),
    )
    assert mismatch["selected"] is False
    assert mismatch["reason"] == "shape_mismatch"
