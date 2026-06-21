from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

import sfincs_jax.rhs1_full_assembly as rfa
from sfincs_jax.rhs1_block_operator import RHS1BlockLayout
from sfincs_jax.solvers.preconditioners import symbolic_sparse
from sfincs_jax.solvers.preconditioners.symbolic_sparse import rhs1_fortran_reduced as rfr


def _small_layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
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


def test_symbolic_sparse_facade_exports_fortran_reduced_family() -> None:
    """Keep the package-level owner importable by future refactor modules."""

    assert (
        symbolic_sparse.build_active_fortran_v3_reduced_sparse_factor_preconditioner
        is rfr.build_active_fortran_v3_reduced_sparse_factor_preconditioner
    )
    assert (
        symbolic_sparse.active_fortran_v3_reduced_preconditioner_matrix
        is rfr.active_fortran_v3_reduced_preconditioner_matrix
    )
    assert (
        symbolic_sparse.select_active_fortran_v3_reduced_support_mode_preconditioner
        is rfr.select_active_fortran_v3_reduced_support_mode_preconditioner
    )


def test_rhs1_full_assembly_keeps_legacy_aliases_pointing_to_new_owner() -> None:
    """Downstream debug scripts can keep old private names during migration."""

    assert (
        rfa._build_active_fortran_v3_reduced_sparse_factor_preconditioner
        is rfr.build_active_fortran_v3_reduced_sparse_factor_preconditioner
    )
    assert (
        rfa._active_fortran_v3_reduced_preconditioner_matrix
        is rfr.active_fortran_v3_reduced_preconditioner_matrix
    )
    assert (
        rfa.select_active_fortran_v3_reduced_support_mode_preconditioner
        is rfr.select_active_fortran_v3_reduced_support_mode_preconditioner
    )


def test_parse_support_mode_candidates_normalizes_tokens_and_rejects_unknown() -> None:
    parsed = rfr.parse_active_fortran_v3_support_mode_candidates(
        candidates="x0,xi0:species0:xmin_l2",
        max_candidates=3,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=0,
    )

    assert parsed[0]["label"] == "current"
    assert parsed[1]["preconditioner_x"] == 0
    assert parsed[2]["preconditioner_xi"] == 0
    assert parsed[2]["preconditioner_species"] == 0
    assert parsed[2]["preconditioner_x_min_l"] == 2

    with pytest.raises(ValueError, match="unsupported Fortran-v3 support-mode"):
        rfr.parse_active_fortran_v3_support_mode_candidates(
            candidates="unsupported-token",
            max_candidates=2,
            preconditioner_x=1,
            preconditioner_xi=1,
            preconditioner_species=1,
            preconditioner_x_min_l=0,
        )


def test_reduced_preconditioner_matrix_drops_default_nonlocal_support(monkeypatch) -> None:
    layout = _small_layout()
    matrix = sp.csr_matrix(np.ones((layout.total_size, layout.total_size), dtype=np.float64))

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")
    reduced_default, default_meta = rfr.active_fortran_v3_reduced_preconditioner_matrix(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        regularization=0.0,
    )
    reduced_full, full_meta = rfr.active_fortran_v3_reduced_preconditioner_matrix(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        regularization=0.0,
        preconditioner_x=0,
        preconditioner_xi=0,
        preconditioner_species=0,
    )

    assert default_meta["fortran_reduced_filter"] == "layout_decoded_supports"
    assert default_meta["dropped_entries"]["x_nonlocal"] > 0
    assert reduced_default.nnz < reduced_full.nnz
    assert full_meta["preconditioner_x"] == 0
    assert full_meta["preconditioner_xi"] == 0
    assert full_meta["preconditioner_species"] == 0


def test_sparse_factor_preconditioner_solves_exact_when_support_is_full(monkeypatch) -> None:
    layout = _small_layout()
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [4.0, 0.2, 0.1, 0.0],
                [0.1, 3.7, 0.3, 0.2],
                [0.0, 0.2, 4.2, 0.1],
                [0.2, 0.0, 0.1, 3.9],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")

    pc = rfr.build_active_fortran_v3_reduced_sparse_factor_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        requested_kind="active_fortran_v3_reduced_lu",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=0.0,
        preconditioner_x=0,
        preconditioner_xi=0,
        preconditioner_species=0,
    )

    assert pc.selected, pc.to_dict()
    x_expected = np.asarray([0.5, -0.25, 0.75, -0.1], dtype=np.float64)
    rhs = np.asarray(matrix @ x_expected, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_expected, rtol=1.0e-12, atol=1.0e-12)
