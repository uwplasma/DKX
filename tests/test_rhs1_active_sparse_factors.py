from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_layout import RHS1BlockLayout
import sfincs_jax.operators.profile_full_system as rhs1_full_assembly
from sfincs_jax.solvers.preconditioner_symbolic_active import (
    build_active_filtered_sparse_factor_preconditioner,
    build_active_global_sparse_factor_preconditioner,
    build_active_scaled_sparse_factor_preconditioner,
)


def _deterministic_vector(size: int) -> np.ndarray:
    return np.linspace(0.25, 1.25, int(size), dtype=np.float64)


def _small_layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=0,
        extra_size=1,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def test_active_sparse_factor_private_driver_names_are_owner_aliases() -> None:
    assert (
        rhs1_full_assembly._build_active_global_sparse_factor_preconditioner
        is build_active_global_sparse_factor_preconditioner
    )
    assert (
        rhs1_full_assembly._build_active_scaled_sparse_factor_preconditioner
        is build_active_scaled_sparse_factor_preconditioner
    )
    assert (
        rhs1_full_assembly._build_active_projected_filtered_sparse_factor_preconditioner
        is build_active_filtered_sparse_factor_preconditioner
    )


def test_active_global_sparse_factor_solves_small_system(monkeypatch) -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [3.0, -0.2, 0.1],
                [0.3, 2.7, -0.4],
                [0.0, 0.2, 2.4],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FACTOR_KIND", "lu")

    pc = build_active_global_sparse_factor_preconditioner(
        matrix=matrix,
        requested_kind="active_global_lu",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_sparse_factor"
    assert pc.metadata["architecture"] == "global_active_sparse_factor"
    rhs = _deterministic_vector(matrix.shape[0])
    expected = np.linalg.solve(matrix.toarray(), rhs)
    actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_active_scaled_sparse_factor_records_equilibration(monkeypatch) -> None:
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [5.0, 0.4, -0.1],
                [0.2, 4.0, 0.3],
                [-0.2, 0.1, 3.5],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_FACTOR_KIND", "lu")

    pc = build_active_scaled_sparse_factor_preconditioner(
        matrix=matrix,
        requested_kind="active_scaled_lu",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_scaled_lu"
    assert pc.metadata["row_scaling"]["axis"] == "row"
    assert pc.metadata["column_scaling"]["axis"] == "column"
    rhs = _deterministic_vector(matrix.shape[0])
    actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert actual.shape == rhs.shape
    assert np.all(np.isfinite(actual))


def test_active_filtered_sparse_factor_retains_physics_band(monkeypatch) -> None:
    layout = _small_layout()
    rows = list(range(layout.total_size))
    cols = list(range(layout.total_size))
    data = [4.0] * layout.total_size
    near_a = layout.kinetic_flat_index(species=0, x=0, ell=0, theta=0, zeta=0)
    near_b = layout.kinetic_flat_index(species=0, x=1, ell=0, theta=0, zeta=0)
    far_a = layout.kinetic_flat_index(species=0, x=0, ell=0, theta=0, zeta=0)
    far_b = layout.kinetic_flat_index(species=0, x=1, ell=1, theta=1, zeta=0)
    tail = layout.f_size
    rows.extend([near_a, near_b, far_a, far_b, near_a, tail])
    cols.extend([near_b, near_a, far_b, far_a, tail, near_a])
    data.extend([-0.4, 0.3, 0.25, -0.2, 0.1, -0.15])
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(layout.total_size, layout.total_size)).tocsr()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_X_RADIUS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ELL_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_THETA_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ZETA_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_KIND", "splu")

    pc = build_active_filtered_sparse_factor_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        requested_kind="active_filtered_sparse_factor",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
        t0=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_filtered_sparse_factor"
    assert pc.metadata["architecture"] == "active_physics_filtered_sparse_factor"
    assert pc.metadata["filtered_nnz"] < matrix.nnz
    assert pc.metadata["physical_band_nnz"] >= 2
    assert pc.metadata["include_tail_couplings"] is True
    rhs = _deterministic_vector(layout.total_size)
    actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert actual.shape == rhs.shape
    assert np.all(np.isfinite(actual))
