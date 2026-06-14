from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import aslinearoperator

from sfincs_jax.explicit_sparse import SparseDecision, SparseOperatorBundle
from sfincs_jax.rhs1_block_operator import RHS1BlockLayout
from sfincs_jax.rhs1_true_operator_rescue import (
    _ResidualCoarseHostSparsePreconditionerBundle,
    _ReusableTrueActionColumnCache,
    _expand_sparse_graph_positions,
    _parse_true_operator_window_specs,
    _rhs1_additive_rescue_nbytes,
    _sparse_factor_nbytes_estimate,
    _true_operator_window_positions_from_residual,
)


class _IdentityFactor:
    factor_nbytes_estimate = 128

    def solve(self, rhs):
        return np.asarray(rhs, dtype=np.float64)


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
