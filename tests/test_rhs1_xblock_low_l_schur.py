from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

import sfincs_jax.operators.profile_full_system as legacy
from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.solvers.preconditioners.xblock import low_l_schur


def _layout() -> RHS1BlockLayout:
    n_species = 1
    n_x = 2
    n_xi = 3
    n_theta = 2
    n_zeta = 2
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    extra_size = 2
    return RHS1BlockLayout(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        extra_size=extra_size,
        total_size=f_size + extra_size,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def _config() -> dict[str, object]:
    return {
        "lmax": 3,
        "drop_tol": 0.0,
        "fill_factor": 8.0,
        "factor_kind": "splu",
        "coarse_lmax": 2,
        "coarse_include_tail": True,
        "coarse_angular_mmax": 0,
        "coarse_angular_nmax": 0,
        "coarse_helical_mmax": 0,
        "coarse_helical_nmax": 0,
        "coarse_basis": "flux_surface_low_l_plus_tail",
    }


def _block_schur_matrix(layout: RHS1BlockLayout) -> sp.csr_matrix:
    blocks = []
    for block_id in range(int(layout.n_x)):
        size = int(layout.n_xi * layout.n_theta * layout.n_zeta)
        diag = 2.0 + 0.1 * block_id + 0.03 * np.arange(size, dtype=np.float64)
        dense = np.diag(diag)
        dense += 0.015 * np.tril(np.ones((size, size), dtype=np.float64), k=-1)
        dense += 0.01 * np.triu(np.ones((size, size), dtype=np.float64), k=1)
        blocks.append(sp.csc_matrix(dense))
    f_block = sp.block_diag(blocks, format="csr")
    tail_size = int(layout.total_size - layout.f_size)
    u = np.zeros((int(layout.f_size), tail_size), dtype=np.float64)
    v = np.zeros((tail_size, int(layout.f_size)), dtype=np.float64)
    u[0::3, 0] = 0.05
    u[1::4, 1] = -0.03
    v[0, 2::5] = 0.04
    v[1, 1::6] = -0.02
    w = np.array([[3.0, 0.2], [0.1, 2.5]], dtype=np.float64)
    return sp.bmat(
        [[f_block, sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )


def test_rhs1_full_assembly_keeps_legacy_aliases_to_xblock_low_l_owner() -> None:
    assert (
        legacy._build_native_xell_kinetic_preconditioner
        is low_l_schur.build_native_xell_kinetic_preconditioner
    )
    assert (
        legacy._build_native_xell_tail_schur_preconditioner
        is low_l_schur.build_native_xell_tail_schur_preconditioner
    )
    assert (
        legacy._build_xblock_tz_low_l_schur_preconditioner
        is low_l_schur.build_xblock_tz_low_l_schur_preconditioner
    )
    assert (
        legacy._build_xblock_tz_low_l_coarse_residual_preconditioner
        is low_l_schur.build_xblock_tz_low_l_coarse_residual_preconditioner
    )
    assert legacy._xblock_tz_low_l_indices is low_l_schur.xblock_tz_low_l_indices


def test_xblock_tz_low_l_indices_follow_rhs1_v3_flat_order() -> None:
    layout = _layout()

    indices = low_l_schur.xblock_tz_low_l_indices(layout=layout, species=0, x=1, lmax=2)

    expected = np.asarray(
        [
            layout.kinetic_flat_index(species=0, x=1, ell=ell, theta=theta, zeta=zeta)
            for ell in range(2)
            for theta in range(layout.n_theta)
            for zeta in range(layout.n_zeta)
        ],
        dtype=np.int64,
    )
    np.testing.assert_array_equal(indices, expected)


def test_xblock_tz_low_l_schur_solves_exact_support_system() -> None:
    layout = _layout()
    matrix = _block_schur_matrix(layout)
    rhs = np.linspace(-0.4, 0.6, int(layout.total_size), dtype=np.float64)

    preconditioner = low_l_schur.build_xblock_tz_low_l_schur_preconditioner(
        matrix=matrix,
        layout=layout,
        requested_kind="xblock_tz_low_l_schur",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        config=_config(),
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "xblock_tz_low_l_schur"
    assert preconditioner.metadata["selected_blocks"] == layout.n_species * layout.n_x
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)


def test_xblock_tz_low_l_schur_rejects_too_small_factor_budget() -> None:
    layout = _layout()
    matrix = _block_schur_matrix(layout)

    preconditioner = low_l_schur.build_xblock_tz_low_l_schur_preconditioner(
        matrix=matrix,
        layout=layout,
        requested_kind="xblock_tz_low_l_schur",
        regularization=0.0,
        max_factor_nbytes=1,
        config=_config(),
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason.startswith("xblock_tz_low_l_factor_budget_exceeded:")
    assert preconditioner.metadata["max_factor_nbytes"] == 1


def test_xblock_tz_low_l_coarse_schur_preserves_exact_base_solution() -> None:
    layout = _layout()
    matrix = _block_schur_matrix(layout)
    rhs = np.sin(np.arange(int(layout.total_size), dtype=np.float64))

    preconditioner = low_l_schur.build_xblock_tz_low_l_coarse_residual_preconditioner(
        matrix=matrix,
        layout=layout,
        requested_kind="xblock_tz_low_l_coarse_schur",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        config=_config(),
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "xblock_tz_low_l_coarse_schur"
    assert preconditioner.metadata["base_preconditioner"]["kind"] == "xblock_tz_low_l_schur"
    assert preconditioner.metadata["coarse_size"] > int(layout.extra_size)
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)
