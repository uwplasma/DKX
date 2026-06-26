from __future__ import annotations

import numpy as np

from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.solvers.preconditioner_schur_profile import (
    build_active_native_xell_coarse_window_basis_csc,
    build_coarse_residual_basis_csc,
    coarse_residual_config,
    coarse_surface_mode_count,
    coarse_surface_modes,
    estimate_coarse_residual_nbytes,
    estimate_xblock_tz_low_l_factor_nbytes,
    xblock_tz_low_l_config,
)


def _layout() -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=2,
        n_x=2,
        n_xi=4,
        n_theta=4,
        n_zeta=3,
        f_size=2 * 2 * 4 * 4 * 3,
        phi1_size=0,
        extra_size=3,
        total_size=2 * 2 * 4 * 4 * 3 + 3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def test_coarse_residual_config_clamps_modes_and_estimates_storage(monkeypatch) -> None:
    layout = _layout()
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_LMAX", "99")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FACTOR_KIND", "invalid")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_COARSE_LMAX", "3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_MMAX", "99")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_NMAX", "99")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_MMAX", "99")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_NMAX", "99")

    xblock_config = xblock_tz_low_l_config(layout)
    config = coarse_residual_config(layout)

    assert xblock_config["lmax"] == layout.n_xi
    assert xblock_config["factor_kind"] == "splu"
    assert config["coarse_lmax"] == 3
    assert config["coarse_angular_mmax"] == layout.n_theta // 2
    assert config["coarse_angular_nmax"] == layout.n_zeta // 2
    assert config["coarse_helical_mmax"] == layout.n_theta // 2
    assert config["coarse_helical_nmax"] == layout.n_zeta // 2
    assert config["coarse_basis"] == "flux_surface_low_l_angular_plus_tail"
    assert estimate_xblock_tz_low_l_factor_nbytes(layout=layout, config=xblock_config) > 0
    assert estimate_coarse_residual_nbytes(layout=layout, config=config) > 0


def test_coarse_residual_basis_has_expected_columns_and_normalized_modes() -> None:
    layout = _layout()
    config = {
        "coarse_lmax": 2,
        "coarse_include_tail": True,
        "coarse_angular_mmax": 1,
        "coarse_angular_nmax": 1,
        "coarse_helical_mmax": 0,
        "coarse_helical_nmax": 0,
    }

    modes = coarse_surface_modes(layout=layout, config=config)
    basis = build_coarse_residual_basis_csc(layout=layout, config=config)
    expected_kinetic_cols = layout.n_species * layout.n_x * config["coarse_lmax"] * len(modes)
    expected_cols = expected_kinetic_cols + layout.extra_size

    assert coarse_surface_mode_count(layout=layout, config=config) == len(modes)
    assert all(np.isclose(np.linalg.norm(values), 1.0) for _name, values in modes)
    assert basis.shape == (layout.total_size, expected_cols)
    assert basis.nnz > expected_kinetic_cols
    assert np.allclose(np.asarray(basis[layout.f_size :, -layout.extra_size :].toarray()), np.eye(layout.extra_size))


def test_active_native_xell_window_basis_respects_specs_and_column_cap(monkeypatch) -> None:
    layout = _layout()
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS", "bad,all:1:2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_ELL_RADIUS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_X_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_MAX_COLUMNS", "5")

    basis, metadata = build_active_native_xell_coarse_window_basis_csc(layout=layout)

    assert basis.shape == (layout.total_size, 5)
    assert basis.nnz == 5
    assert metadata["window_basis_requested"] is True
    assert metadata["window_basis_columns"] == 5
    assert metadata["window_basis_skipped_specs"] == 1
    assert metadata["window_basis_truncated"] is True
