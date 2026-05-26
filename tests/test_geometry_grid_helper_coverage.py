from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax import jax_geometry_adapters
from sfincs_jax.geometry import boozer_geometry_scheme1, boozer_geometry_scheme2, boozer_geometry_scheme4
from sfincs_jax.geometry import boozer_geometry_from_bc_file
from sfincs_jax.grids import uniform_diff_matrices
from sfincs_jax.io import (
    _conversion_factors_to_from_dpsi_hat,
    _dphi_hat_dpsi_hat_from_er_geometry_scheme4,
    _fortran_logical,
    _scheme4_radial_constants,
    _set_input_radial_coordinate_wish,
)
from sfincs_jax.vmec_geometry import _finite_diff_on_full_mesh_from_half_mesh
from sfincs_jax.vmec_geometry import vmec_geometry_from_wout, vmec_geometry_from_wout_file
from sfincs_jax.vmec_wout import read_vmec_wout


_REPO_ROOT = Path(__file__).resolve().parents[1]
_EQUILIBRIA_DIR = _REPO_ROOT / "sfincs_jax" / "data" / "equilibria"
_W7X_BC = _EQUILIBRIA_DIR / "w7x_standardConfig.bc"
_W7X_WOUT = _EQUILIBRIA_DIR / "wout_w7x_standardConfig.nc"


def test_optional_jax_geometry_backend_status_is_shallow(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_find_spec(name: str) -> object | None:
        calls.append(name)
        if name == "vmec_jax":
            return object()
        if name == "booz_xform_jax":
            return None
        raise AssertionError(f"unexpected optional backend probe: {name}")

    monkeypatch.setattr(jax_geometry_adapters, "find_spec", fake_find_spec)

    status = jax_geometry_adapters.optional_jax_geometry_backend_status()

    assert status == {"vmec_jax": True, "booz_xform_jax": False}
    assert calls == ["vmec_jax", "booz_xform_jax"]


def test_uniform_diff_matrices_invalid_inputs_and_weights() -> None:
    with pytest.raises(ValueError):
        uniform_diff_matrices(n=1, x_min=0.0, x_max=1.0, scheme=2)
    with pytest.raises(ValueError):
        uniform_diff_matrices(n=4, x_min=2.0, x_max=1.0, scheme=2)
    with pytest.raises(ValueError):
        uniform_diff_matrices(n=4, x_min=1.0, x_max=1.0, scheme=2)
    with pytest.raises(NotImplementedError):
        uniform_diff_matrices(n=6, x_min=0.0, x_max=1.0, scheme=122)

    x, w, _, _ = uniform_diff_matrices(n=5, x_min=0.0, x_max=1.0, scheme=2)
    np.testing.assert_allclose(np.asarray(x), np.linspace(0.0, 1.0, 5))
    np.testing.assert_allclose(np.asarray(w), np.asarray([0.125, 0.25, 0.25, 0.25, 0.125]))

    x_per, w_per, _, _ = uniform_diff_matrices(n=4, x_min=0.0, x_max=2.0 * np.pi, scheme=0)
    np.testing.assert_allclose(np.asarray(x_per), np.asarray([0.0, np.pi / 2.0, np.pi, 3.0 * np.pi / 2.0]))
    np.testing.assert_allclose(np.asarray(w_per), np.full((4,), np.pi / 2.0))


def test_uniform_diff_matrices_spectral_derivative_is_exact_for_trig_modes() -> None:
    x, _, ddx, d2dx2 = uniform_diff_matrices(n=16, x_min=0.0, x_max=2.0 * np.pi, scheme=20)
    x_np = np.asarray(x)
    f = np.sin(2.0 * x_np) + 0.5 * np.cos(3.0 * x_np)
    df = 2.0 * np.cos(2.0 * x_np) - 1.5 * np.sin(3.0 * x_np)
    d2f = -4.0 * np.sin(2.0 * x_np) - 4.5 * np.cos(3.0 * x_np)
    np.testing.assert_allclose(np.asarray(ddx) @ f, df, atol=1e-11, rtol=0.0)
    np.testing.assert_allclose(np.asarray(d2dx2) @ f, d2f, atol=1e-11, rtol=0.0)


def test_scheme1_geometry_reduces_to_simple_cosine_model() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    geom = boozer_geometry_scheme1(
        theta=theta,
        zeta=zeta,
        epsilon_t=0.1,
        epsilon_h=0.0,
        epsilon_antisymm=0.0,
        iota=0.4,
        g_hat=5.0,
        i_hat=0.0,
        b0_over_bbar=2.0,
        helicity_l=1,
        helicity_n=3,
        helicity_antisymm_l=0,
        helicity_antisymm_n=0,
    )
    theta2 = theta[:, None]
    expected_b = np.broadcast_to(2.0 + 0.2 * np.cos(theta2), (theta.size, zeta.size))
    expected_db_dtheta = np.broadcast_to(-0.2 * np.sin(theta2), (theta.size, zeta.size))
    expected_d = expected_b**2 / 5.0
    np.testing.assert_allclose(np.asarray(geom.b_hat), expected_b)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dtheta), expected_db_dtheta)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dzeta), 0.0)
    np.testing.assert_allclose(np.asarray(geom.d_hat), expected_d)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_theta), 0.4 * expected_d)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_zeta), expected_d)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_zeta), 5.0)
    assert geom.n_periods == 3


def test_scheme4_zero_harmonics_is_constant_field() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 10, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 5.0, 12, endpoint=False)
    geom = boozer_geometry_scheme4(theta=theta, zeta=zeta, harmonics_amp0=np.zeros(3))
    np.testing.assert_allclose(np.asarray(geom.b_hat), geom.b0_over_bbar)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dtheta), 0.0)
    np.testing.assert_allclose(np.asarray(geom.db_hat_dzeta), 0.0)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_theta), geom.i_hat)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_zeta), geom.g_hat)


def test_scheme2_geometry_shapes_and_scalars() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 10.0, 9, endpoint=False)
    geom = boozer_geometry_scheme2(theta=theta, zeta=zeta)
    assert geom.n_periods == 10
    assert np.asarray(geom.b_hat).shape == (7, 9)
    assert np.asarray(geom.db_hat_dtheta).shape == (7, 9)
    assert np.asarray(geom.db_hat_dzeta).shape == (7, 9)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_theta), 0.0)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sub_zeta), geom.g_hat)


def test_vmec_half_mesh_finite_difference_matches_v3_pattern() -> None:
    arr1 = np.asarray([0.0, 1.0, 4.0, 9.0], dtype=np.float64)
    out1 = _finite_diff_on_full_mesh_from_half_mesh(arr1, 0.5)
    np.testing.assert_allclose(out1, np.asarray([6.0, 6.0, 10.0, 10.0]))

    arr2 = np.asarray([[0.0, 1.0, 4.0, 9.0], [1.0, 2.0, 3.0, 4.0]], dtype=np.float64)
    out2 = _finite_diff_on_full_mesh_from_half_mesh(arr2, 0.5)
    np.testing.assert_allclose(out2[0], np.asarray([6.0, 6.0, 10.0, 10.0]))
    np.testing.assert_allclose(out2[1], np.asarray([2.0, 2.0, 2.0, 2.0]))

    with pytest.raises(ValueError):
        _finite_diff_on_full_mesh_from_half_mesh(np.zeros((2, 2, 2)), 1.0)


def test_boozer_bc_geometry_loader_on_reference_fixture() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 5.0, 6, endpoint=False)
    geom = boozer_geometry_from_bc_file(path=str(_W7X_BC), theta=theta, zeta=zeta, r_n_wish=0.5)

    b_hat = np.asarray(geom.b_hat)
    d_hat = np.asarray(geom.d_hat)
    np.testing.assert_equal(b_hat.shape, (8, 6))
    assert geom.n_periods == 5
    assert np.all(np.isfinite(b_hat))
    assert np.all(b_hat > 0.0)
    np.testing.assert_allclose(d_hat, b_hat**2 / (geom.g_hat + geom.iota * geom.i_hat), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_theta), geom.iota * d_hat, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(geom.b_hat_sup_zeta), d_hat, rtol=1e-12, atol=1e-12)


def test_vmec_geometry_loader_on_reference_fixture_is_consistent_with_bc_range() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 5.0, 6, endpoint=False)
    geom_bc = boozer_geometry_from_bc_file(path=str(_W7X_BC), theta=theta, zeta=zeta, r_n_wish=0.5)
    geom_vmec = vmec_geometry_from_wout_file(path=_W7X_WOUT, theta=theta, zeta=zeta, psi_n_wish=0.25)

    b_bc = np.asarray(geom_bc.b_hat)
    b_vmec = np.asarray(geom_vmec.b_hat)
    np.testing.assert_equal(b_vmec.shape, (8, 6))
    assert geom_vmec.n_periods == 5
    assert np.all(np.isfinite(b_vmec))
    assert np.all(b_vmec > 0.0)
    # The Boozer and VMEC loaders come from different reference files for the same W7-X
    # configuration. Their coarse-grid B ranges should still agree closely.
    assert abs(float(np.min(b_vmec)) - float(np.min(b_bc))) < 0.02
    assert abs(float(np.max(b_vmec)) - float(np.max(b_bc))) < 0.02


def test_vmec_geometry_from_preloaded_wout_matches_file_wrapper() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi / 5.0, 6, endpoint=False)
    from_file = vmec_geometry_from_wout_file(path=_W7X_WOUT, theta=theta, zeta=zeta, psi_n_wish=0.25)
    from_object = vmec_geometry_from_wout(w=read_vmec_wout(_W7X_WOUT), theta=theta, zeta=zeta, psi_n_wish=0.25)

    np.testing.assert_equal(np.asarray(from_object.b_hat).shape, (8, 6))
    np.testing.assert_allclose(np.asarray(from_object.b_hat), np.asarray(from_file.b_hat), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(from_object.db_hat_dtheta), np.asarray(from_file.db_hat_dtheta), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(from_object.db_hat_dzeta), np.asarray(from_file.db_hat_dzeta), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(from_object.d_hat), np.asarray(from_file.d_hat), rtol=0.0, atol=0.0)


def test_io_radial_coordinate_formula_helpers() -> None:
    psi_a_hat, a_hat = _scheme4_radial_constants()
    assert psi_a_hat == pytest.approx(-0.384935)
    assert a_hat == pytest.approx(0.5109)

    wish = _set_input_radial_coordinate_wish(
        input_radial_coordinate=3,
        psi_a_hat=psi_a_hat,
        a_hat=a_hat,
        psi_hat_wish_in=0.0,
        psi_n_wish_in=0.0,
        r_hat_wish_in=0.0,
        r_n_wish_in=0.5,
    )
    psi_hat_wish, psi_n_wish, r_hat_wish, r_n_wish = wish
    assert psi_n_wish == pytest.approx(0.25)
    assert r_n_wish == pytest.approx(0.5)
    assert psi_hat_wish == pytest.approx(psi_a_hat * 0.25)
    assert r_hat_wish == pytest.approx(a_hat * 0.5)

    conv = _conversion_factors_to_from_dpsi_hat(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=0.5)
    assert conv["ddpsiN2ddpsiHat"] == pytest.approx(1.0 / psi_a_hat)
    assert conv["ddpsiHat2ddpsiN"] == pytest.approx(psi_a_hat)
    assert conv["ddrHat2ddpsiHat"] == pytest.approx(a_hat / (2.0 * psi_a_hat * 0.5))
    assert conv["ddpsiHat2ddrHat"] == pytest.approx((2.0 * psi_a_hat * 0.5) / a_hat)

    er = 3.0
    expected = a_hat / (2.0 * psi_a_hat * math.sqrt(0.25)) * (-er)
    assert _dphi_hat_dpsi_hat_from_er_geometry_scheme4(er) == pytest.approx(expected)
    assert int(_fortran_logical(True)) == 1
    assert int(_fortran_logical(False)) == -1
