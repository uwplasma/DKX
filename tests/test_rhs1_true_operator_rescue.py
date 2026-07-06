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
    _try_build_true_operator_active_block_lsq_preconditioner,
    _try_build_true_operator_active_residual_block_lsq_preconditioner,
    _try_build_true_operator_active_submatrix_preconditioner,
    _try_build_true_operator_coupled_coarse_lsq_preconditioner,
    _try_build_true_operator_residual_window_lsq_preconditioner,
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


def _small_active_layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=1,
        extra_size=1,
        total_size=6,
        constraint_scheme=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def _identity_true_actions(size: int):
    matrix = np.eye(int(size), dtype=np.float64)
    return (
        lambda x: matrix @ np.asarray(x, dtype=np.float64),
        lambda x: matrix @ np.asarray(x, dtype=np.float64),
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


def test_residual_window_bundle_no_windows_returns_base_solution() -> None:
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
    bundle = _ResidualWindowHostSparsePreconditionerBundle(
        base_factor=_HalfFactor(),
        operator=operator,
        window_positions=(),
        window_factors=(),
        kind="window_empty",
    )

    np.testing.assert_allclose(bundle.solve(np.asarray([2.0, -4.0, 6.0])), [1.0, -2.0, 3.0])


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


def test_reusable_true_action_column_cache_rejects_malformed_one_hot_inputs() -> None:
    cache = _ReusableTrueActionColumnCache(
        true_matvec=lambda x: 2.0 * np.asarray(x, dtype=np.float64),
        true_matmat=None,
        n=3,
        max_nbytes=1024,
        enabled=True,
    )

    assert _ReusableTrueActionColumnCache._one_hot_positions(np.ones(3)) is None
    bad_value = np.zeros((3, 1), dtype=np.float64)
    bad_value[1, 0] = 2.0
    assert _ReusableTrueActionColumnCache._one_hot_positions(bad_value) is None

    wrong_rows = np.eye(2, dtype=np.float64)
    np.testing.assert_allclose(cache.matmat(wrong_rows), 2.0 * wrong_rows)
    assert cache.metadata()["bypass_calls"] == 1


def test_reusable_true_action_column_cache_falls_back_when_batched_shape_is_invalid() -> None:
    matrix = np.asarray(
        [
            [2.0, 0.1, 0.0],
            [0.0, 3.0, -0.2],
            [0.4, 0.0, 4.0],
        ],
        dtype=np.float64,
    )
    calls = 0

    def true_matmat(x):
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros((2, x.shape[1]), dtype=np.float64)
        return matrix @ np.asarray(x, dtype=np.float64)

    cache = _ReusableTrueActionColumnCache(
        true_matvec=lambda x: matrix @ np.asarray(x, dtype=np.float64),
        true_matmat=true_matmat,
        n=3,
        max_nbytes=1024,
        enabled=True,
    )
    basis = np.zeros((3, 2), dtype=np.float64)
    basis[[0, 2], [0, 1]] = 1.0

    np.testing.assert_allclose(cache.matmat(basis), matrix @ basis)
    metadata = cache.metadata()
    assert metadata["bypass_calls"] == 1
    assert metadata["stored_columns"] == 0
    assert calls == 2


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


def test_true_operator_lsq_bundles_clip_residual_step_lengths() -> None:
    def true_matvec(x):
        return np.asarray(x, dtype=np.float64)

    rhs = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    one_column = np.asarray([[1.0], [0.0], [0.0]], dtype=np.float64)

    window = _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        window_positions=np.asarray([0]),
        a_window=one_column,
        inv_column_scale=np.ones(1, dtype=np.float64),
        solve_normal=lambda _rhs: np.asarray([10.0], dtype=np.float64),
        kind="window_lsq",
        damping=True,
        beta_max=0.05,
    )
    np.testing.assert_allclose(window.solve(rhs), [0.5, 0.0, 0.0])

    active = _TrueOperatorActiveSubmatrixPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        block_positions=np.asarray([0]),
        a_window=one_column,
        solve_block=lambda _rhs: np.asarray([10.0], dtype=np.float64),
        kind="active_block",
        damping=True,
        alpha_clip=0.05,
    )
    np.testing.assert_allclose(active.solve(rhs), [0.5, 0.0, 0.0])

    coupled = _TrueOperatorCoupledCoarseLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec,
        z_basis=one_column,
        a_basis=one_column,
        inv_column_scale=np.ones(1, dtype=np.float64),
        solve_normal=lambda _rhs: np.asarray([10.0], dtype=np.float64),
        kind="coupled_lsq",
        damping=True,
        beta_max=0.05,
    )
    np.testing.assert_allclose(coupled.solve(rhs), [0.5, 0.0, 0.0])


def test_true_operator_lsq_bundles_handle_degenerate_damping_and_lsq_fallback(monkeypatch) -> None:
    def true_matvec_zero(x):
        return np.zeros_like(np.asarray(x, dtype=np.float64))

    def fail_solve(*_args, **_kwargs):
        raise np.linalg.LinAlgError("synthetic singular normal equations")

    monkeypatch.setattr(np.linalg, "solve", fail_solve)

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
    least_squares = _ResidualWindowHostSparsePreconditionerBundle(
        base_factor=_ZeroFactor(),
        operator=operator,
        window_positions=(np.asarray([0, 2]),),
        window_factors=(_IdentityFactor(),),
        kind="window_lsq",
        coefficient_mode="normal_equations",
        regularization=0.0,
    )
    np.testing.assert_allclose(least_squares.solve(np.asarray([1.0, -2.0, 3.0])), [1.0, 0.0, 3.0])

    z_basis = np.asarray([[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    window = _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec_zero,
        window_positions=np.asarray([0, 2]),
        a_window=z_basis,
        inv_column_scale=np.ones(2, dtype=np.float64),
        solve_normal=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="window_lsq",
        damping=True,
    )
    active = _TrueOperatorActiveSubmatrixPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec_zero,
        block_positions=np.asarray([0, 2]),
        a_window=np.zeros_like(z_basis),
        solve_block=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="active_block",
        damping=True,
    )
    coupled = _TrueOperatorCoupledCoarseLSQPreconditionerBundle(
        base_factor=_ZeroFactor(),
        true_matvec=true_matvec_zero,
        z_basis=z_basis,
        a_basis=z_basis,
        inv_column_scale=np.ones(2, dtype=np.float64),
        solve_normal=lambda rhs: np.asarray(rhs, dtype=np.float64),
        kind="coupled_lsq",
        damping=True,
    )

    rhs = np.asarray([1.0, -2.0, 3.0])
    np.testing.assert_allclose(window.solve(rhs), np.zeros(3))
    np.testing.assert_allclose(active.solve(rhs), np.zeros(3))
    np.testing.assert_allclose(coupled.solve(rhs), np.zeros(3))


def test_rescue_budget_and_sparse_factor_memory_estimate() -> None:
    factor = SimpleNamespace(factor_nbytes_estimate=5_000_000_000)
    assert _rhs1_additive_rescue_nbytes(factor, 512.0) == 5_000_000_000 + 512 * 1024 * 1024
    assert _rhs1_additive_rescue_nbytes(factor, 0.0) == 0

    sparse_factor = SimpleNamespace(
        L=sp.eye(4, format="csr"),
        U=sp.eye(4, format="csr"),
    )
    assert _sparse_factor_nbytes_estimate(sparse_factor) > 0
    assert _sparse_factor_nbytes_estimate(SimpleNamespace(L=None, U=None)) == 0


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


def test_graph_expansion_and_window_spec_edge_cases() -> None:
    graph = sp.eye(3, format="csr", dtype=np.float64)
    np.testing.assert_array_equal(
        _expand_sparse_graph_positions(graph, np.asarray([2, 1]), depth=0, max_size=3),
        np.asarray([1, 2]),
    )
    np.testing.assert_array_equal(
        _expand_sparse_graph_positions(graph, np.asarray([99]), depth=1, max_size=3),
        np.asarray([99]),
    )
    np.testing.assert_array_equal(
        _expand_sparse_graph_positions(graph, np.asarray([1]), depth=3, max_size=3),
        np.asarray([1]),
    )

    layout = _layout_with_tail()
    specs = _parse_true_operator_window_specs(
        " ; 0:bad:1, 0:0, 0:99:0, 0:0:1, 0/0/1",
        layout=layout,
    )
    assert specs == ((0, 0, 1),)


def test_window_selection_handles_zero_energy_duplicates_and_empty_specs() -> None:
    layout = _layout_with_tail()

    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.zeros((layout.total_size,), dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        include_tail=False,
    )
    assert positions.size == 0
    assert metadata == ()

    active_first_combo = np.asarray([0], dtype=np.int64)
    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.asarray([1.0], dtype=np.float64),
        layout=layout,
        active_indices=active_first_combo,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        include_tail=False,
        explicit_specs=((0, 0, 1),),
    )
    assert positions.size == 0
    assert metadata == ()

    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.asarray([0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64),
        layout=layout,
        active_indices=np.arange(layout.f_size, dtype=np.int64),
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        include_tail=False,
        explicit_specs=((0, 0, 1), (0, 0, 1)),
    )
    np.testing.assert_array_equal(positions, np.asarray([1]))
    assert len(metadata) == 1


def test_window_selection_rejects_shape_mismatch_and_tail_only_residuals() -> None:
    layout = _layout_with_tail()

    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.asarray([1.0, 2.0]),
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        include_tail=True,
    )
    assert positions.size == 0
    assert metadata == ()

    active_tail_only = np.asarray([layout.f_size, layout.f_size + 1], dtype=np.int64)
    positions, metadata = _true_operator_window_positions_from_residual(
        residual=np.asarray([3.0, -4.0]),
        layout=layout,
        active_indices=active_tail_only,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        include_tail=True,
    )
    assert positions.size == 0
    assert metadata == ()


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


def test_true_operator_residual_window_builder_admits_bounded_tail_window() -> None:
    layout = _layout_with_tail()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)
    rhs = np.asarray([0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 2.0, -1.0], dtype=np.float64)
    emitted: list[str] = []

    bundle = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_window_size=4,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        damping=False,
        emit=lambda _level, message: emitted.append(str(message)),
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["window_size"] == 4
    assert bundle.metadata["include_tail"] is True
    assert any("true residual window built" in message for message in emitted)
    np.testing.assert_allclose(true_matvec(bundle.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_true_operator_residual_window_builder_rejects_memory_budget() -> None:
    layout = _layout_with_tail()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)

    bundle = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=np.ones((layout.total_size,), dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=1,
        ell_radius=1,
        max_nbytes=1,
        regularization=0.0,
        max_window_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
    )

    assert bundle is None


def test_true_operator_residual_window_builder_reports_size_and_budget_rejections() -> None:
    layout = _layout_with_tail()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)
    residual = np.asarray([0.0, 4.0, 0.0, 0.0, 0.0, 3.0, 2.0, -1.0], dtype=np.float64)
    emitted: list[str] = []

    too_wide = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=residual,
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_window_size=1,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        emit=lambda _level, message: emitted.append(str(message)),
    )
    assert too_wide is None
    assert any("true residual window skipped" in message for message in emitted)

    emitted.clear()
    too_expensive = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=residual,
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1,
        regularization=0.0,
        max_window_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
        emit=lambda _level, message: emitted.append(str(message)),
    )
    assert too_expensive is None
    assert any("true residual window budget exceeded" in message for message in emitted)


def test_true_operator_residual_window_builder_rejects_invalid_or_dropped_columns() -> None:
    layout = _layout_with_tail()

    invalid_shape = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(x, dtype=np.float64),
        true_matmat=lambda x: np.zeros((layout.total_size - 1, x.shape[1]), dtype=np.float64),
        factor_bundle=_ZeroFactor(),
        residual=np.ones((layout.total_size,), dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_window_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=False,
    )
    assert invalid_shape is None

    dropped = _try_build_true_operator_residual_window_lsq_preconditioner(
        true_matvec=lambda x: np.asarray(x, dtype=np.float64),
        true_matmat=lambda x: np.asarray(x, dtype=np.float64),
        factor_bundle=_ZeroFactor(),
        residual=np.ones((layout.total_size,), dtype=np.float64),
        layout=layout,
        active_indices=None,
        max_windows=1,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_window_size=layout.total_size,
        column_batch=2,
        drop_tol=2.0,
        include_tail=False,
    )
    assert dropped is None


def test_true_operator_active_block_builders_correct_small_identity_operator() -> None:
    layout = _small_active_layout()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)
    rhs = np.asarray([1.0, -2.0, 3.0, -4.0, 0.5, -0.25], dtype=np.float64)

    active_block = _try_build_true_operator_active_block_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        x_count=1,
        ell_count=2,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_block_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        max_tail=2,
        damping=False,
    )
    assert active_block is not None
    assert active_block.metadata is not None
    assert active_block.metadata["kinetic_selected"] == layout.f_size
    assert active_block.metadata["tail_selected"] == 2
    np.testing.assert_allclose(true_matvec(active_block.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)

    active_residual = _try_build_true_operator_active_residual_block_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_block_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        max_tail=2,
        kinetic_only=True,
        damping=False,
    )
    assert active_residual is not None
    assert active_residual.metadata is not None
    assert active_residual.metadata["selection"] == "top_residual_active_positions"
    np.testing.assert_allclose(true_matvec(active_residual.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)

    active_submatrix = _try_build_true_operator_active_submatrix_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=rhs,
        layout=layout,
        active_indices=None,
        x_count=1,
        ell_count=2,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_block_size=layout.total_size,
        column_batch=2,
        drop_tol=0.0,
        include_tail=True,
        max_tail=2,
        damping=False,
    )
    assert active_submatrix is not None
    assert active_submatrix.metadata is not None
    assert active_submatrix.metadata["block_size"] == layout.total_size
    np.testing.assert_allclose(true_matvec(active_submatrix.solve(rhs)), rhs, rtol=1.0e-12, atol=1.0e-12)


def test_true_operator_active_block_builders_fail_closed_on_bad_inputs() -> None:
    layout = _small_active_layout()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)
    rhs = np.arange(1.0, layout.total_size + 1.0, dtype=np.float64)
    emitted: list[str] = []

    assert (
        _try_build_true_operator_active_block_lsq_preconditioner(
            true_matvec=true_matvec,
            true_matmat=true_matmat,
            factor_bundle=_ZeroFactor(),
            residual=rhs,
            layout=layout,
            active_indices=np.arange(layout.total_size - 1, dtype=np.int64),
            x_count=1,
            ell_count=1,
            max_nbytes=1024 * 1024,
            regularization=0.0,
            max_block_size=layout.total_size,
            column_batch=2,
            drop_tol=0.0,
            include_tail=False,
            max_tail=0,
            emit=lambda _level, message: emitted.append(str(message)),
        )
        is None
    )
    assert any("active_shape_mismatch" in message for message in emitted)

    assert (
        _try_build_true_operator_active_residual_block_lsq_preconditioner(
            true_matvec=true_matvec,
            true_matmat=true_matmat,
            factor_bundle=_ZeroFactor(),
            residual=rhs,
            layout=layout,
            active_indices=np.arange(layout.total_size - 1, dtype=np.int64),
            max_nbytes=1024 * 1024,
            regularization=0.0,
            max_block_size=layout.total_size,
            column_batch=2,
            drop_tol=0.0,
            include_tail=False,
            max_tail=0,
            kinetic_only=True,
            emit=lambda _level, message: emitted.append(str(message)),
        )
        is None
    )

    def bad_matmat(x):
        return np.zeros((layout.total_size - 1, x.shape[1]), dtype=np.float64)

    assert (
        _try_build_true_operator_active_block_lsq_preconditioner(
            true_matvec=true_matvec,
            true_matmat=bad_matmat,
            factor_bundle=_ZeroFactor(),
            residual=rhs,
            layout=layout,
            active_indices=None,
            x_count=1,
            ell_count=1,
            max_nbytes=1024 * 1024,
            regularization=0.0,
            max_block_size=layout.total_size,
            column_batch=2,
            drop_tol=0.0,
            include_tail=False,
            max_tail=0,
        )
        is None
    )
    assert (
        _try_build_true_operator_active_residual_block_lsq_preconditioner(
            true_matvec=true_matvec,
            true_matmat=bad_matmat,
            factor_bundle=_ZeroFactor(),
            residual=rhs,
            layout=layout,
            active_indices=None,
            max_nbytes=1024 * 1024,
            regularization=0.0,
            max_block_size=layout.total_size,
            column_batch=2,
            drop_tol=0.0,
            include_tail=False,
            max_tail=0,
            kinetic_only=True,
        )
        is None
    )


def test_true_operator_coupled_coarse_builder_uses_window_and_tail_basis() -> None:
    layout = _small_active_layout()
    true_matvec, true_matmat = _identity_true_actions(layout.total_size)
    rhs = np.asarray([0.0, 2.0, 0.0, -3.0, 0.75, -0.5], dtype=np.float64)
    op = SimpleNamespace(
        constraint_scheme=1,
        fblock=SimpleNamespace(f_shape=layout.f_shape),
        theta_weights=np.ones((layout.n_theta,), dtype=np.float64),
        zeta_weights=np.ones((layout.n_zeta,), dtype=np.float64),
        d_hat=np.ones((layout.n_theta, layout.n_zeta), dtype=np.float64),
    )

    bundle = _try_build_true_operator_coupled_coarse_lsq_preconditioner(
        true_matvec=true_matvec,
        true_matmat=true_matmat,
        factor_bundle=_ZeroFactor(),
        residual=rhs,
        op=op,
        layout=layout,
        active_indices=None,
        max_windows=2,
        x_radius=0,
        ell_radius=0,
        max_nbytes=1024 * 1024,
        regularization=0.0,
        max_coarse_size=8,
        column_batch=2,
        drop_tol=0.0,
        low_lmax=0,
        profile_moment_count=0,
        angular_lmax=0,
        angular_mode_max=0,
        max_tail_units=2,
        include_tail=True,
        include_constraint_sources=False,
        include_fsavg=False,
        include_window_residual=True,
        include_profile_moments=False,
        include_angular_residual=False,
        include_angular_basis=False,
        include_preconditioned_loads=False,
        preconditioned_load_max_columns=0,
        preconditioned_load_max_nnz=0,
        preconditioned_load_drop_tol=0.0,
        damping=False,
    )

    assert bundle is not None
    assert bundle.metadata is not None
    assert bundle.metadata["window_residual_included"] is True
    assert bundle.metadata["tail_included"] is True
    assert bundle.metadata["coarse_size"] >= 2
    np.testing.assert_allclose(true_matvec(bundle.solve(rhs)), rhs, rtol=1.0e-10, atol=1.0e-10)


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
