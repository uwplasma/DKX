from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

import sfincs_jax.operators.profile_response.full_system as legacy
from sfincs_jax.operators.profile_response.layout import RHS1BlockLayout
from sfincs_jax.solvers.preconditioners.xblock import active_projected


def _layout() -> RHS1BlockLayout:
    n_species = 1
    n_x = 2
    n_xi = 2
    n_theta = 2
    n_zeta = 1
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    return RHS1BlockLayout(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        extra_size=0,
        total_size=f_size,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def _block_matrix(layout: RHS1BlockLayout) -> sp.csr_matrix:
    block_size = int(layout.n_xi * layout.n_theta * layout.n_zeta)
    blocks = []
    for x in range(int(layout.n_x)):
        dense = np.diag(2.0 + 0.25 * x + 0.1 * np.arange(block_size, dtype=np.float64))
        dense += 0.03 * np.tril(np.ones((block_size, block_size), dtype=np.float64), k=-1)
        dense += 0.02 * np.triu(np.ones((block_size, block_size), dtype=np.float64), k=1)
        blocks.append(sp.csc_matrix(dense))
    return sp.block_diag(blocks, format="csr")


def test_rhs1_full_assembly_keeps_legacy_aliases_to_active_xblock_owner() -> None:
    assert (
        legacy._build_active_projected_xblock_preconditioner
        is active_projected.build_active_projected_xblock_preconditioner
    )
    assert (
        legacy._build_active_projected_overlap_schwarz_preconditioner
        is active_projected.build_active_projected_overlap_schwarz_preconditioner
    )
    assert legacy._active_positions_for_full_indices is active_projected.active_positions_for_full_indices


def test_active_positions_for_full_indices_deduplicates_and_maps_unsorted_active_indices() -> None:
    active = np.asarray([10, 4, 7, 2, 9], dtype=np.int64)
    full = np.asarray([9, 4, 9, 11, 2], dtype=np.int64)

    positions = active_projected.active_positions_for_full_indices(
        active_indices=active,
        full_indices=full,
    )

    np.testing.assert_array_equal(positions, np.asarray([1, 3, 4], dtype=np.int64))


def test_active_projected_xblock_solves_exact_block_system(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    rhs = np.linspace(-0.5, 0.75, int(layout.total_size), dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_xblock_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_xblock",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "active_xblock"
    assert preconditioner.metadata["base_mode"] == "zero"
    assert preconditioner.metadata["block_count"] == layout.n_species * layout.n_x
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_projected_xblock_rejects_too_small_factor_budget(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_xblock_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_xblock",
        regularization=0.0,
        max_factor_nbytes=1,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason.startswith("active_xblock_budget_exceeded:")
    assert preconditioner.metadata["max_factor_nbytes"] == 1


def test_active_projected_overlap_schwarz_solves_exact_radius_zero_blocks(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    rhs = np.cos(np.arange(int(layout.total_size), dtype=np.float64))
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_overlap_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_overlap_schwarz",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "active_overlap_schwarz"
    assert preconditioner.metadata["patch_count"] == layout.n_species * layout.n_x
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_projected_overlap_schwarz_rejects_too_small_factor_budget(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_overlap_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_overlap_schwarz",
        regularization=0.0,
        max_factor_nbytes=1,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason.startswith("active_overlap_schwarz_budget_exceeded:")
    assert preconditioner.metadata["max_factor_nbytes"] == 1
