from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.problems.transport_diagnostics as td


def _minimal_transport_diag_op(*, n_xi: int = 3) -> SimpleNamespace:
    n_species = 1
    n_x = 2
    n_theta = 2
    n_zeta = 2
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        total_size=f_size,
        fblock=SimpleNamespace(
            f_shape=(n_species, n_x, n_xi, n_theta, n_zeta),
            collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([1, n_xi], dtype=jnp.int32)),
        ),
        x=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        x_weights=jnp.asarray([0.4, 0.6], dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        n_hat=jnp.asarray([2.0], dtype=jnp.float64),
        t_hat=jnp.asarray([3.0], dtype=jnp.float64),
        m_hat=jnp.asarray([5.0], dtype=jnp.float64),
        alpha=0.7,
        delta=0.02,
        phi1_hat_base=jnp.zeros((n_theta, n_zeta), dtype=jnp.float64),
        theta_weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        zeta_weights=jnp.asarray([0.6, 0.4], dtype=jnp.float64),
        b_hat=jnp.asarray([[1.0, 1.2], [0.9, 1.1]], dtype=jnp.float64),
        d_hat=jnp.asarray([[2.0, 2.5], [1.5, 3.0]], dtype=jnp.float64),
        b_hat_sub_theta=jnp.asarray([[0.2, 0.3], [0.4, 0.5]], dtype=jnp.float64),
        b_hat_sub_zeta=jnp.asarray([[0.7, 0.8], [0.9, 1.0]], dtype=jnp.float64),
        db_hat_dtheta=jnp.asarray([[0.01, 0.02], [0.03, 0.04]], dtype=jnp.float64),
        db_hat_dzeta=jnp.asarray([[0.05, 0.06], [0.07, 0.08]], dtype=jnp.float64),
        fsab_hat2=jnp.asarray(1.3, dtype=jnp.float64),
        constraint_scheme=0,
    )


def test_transport_diagnostics_dataclasses_are_jax_pytrees() -> None:
    diag = td.V3TransportDiagnostics(
        vprime_hat=jnp.asarray(1.0),
        particle_flux_vm_psi_hat=jnp.asarray([2.0]),
        heat_flux_vm_psi_hat=jnp.asarray([3.0]),
        fsab_flow=jnp.asarray([4.0]),
        particle_flux_before_surface_integral_vm=jnp.ones((1, 2, 2)),
        heat_flux_before_surface_integral_vm=2.0 * jnp.ones((1, 2, 2)),
        particle_flux_before_surface_integral_vm0=3.0 * jnp.ones((1, 2, 2)),
        heat_flux_before_surface_integral_vm0=4.0 * jnp.ones((1, 2, 2)),
        particle_flux_vm_psi_hat_vs_x=jnp.ones((2, 1)),
        heat_flux_vm_psi_hat_vs_x=2.0 * jnp.ones((2, 1)),
        fsab_flow_vs_x=3.0 * jnp.ones((2, 1)),
    )
    pre = td.V3TransportDiagnosticsPrecomputed(
        vprime_hat=jnp.asarray(1.0),
        theta_w=jnp.asarray([0.25, 0.75]),
        zeta_w=jnp.asarray([0.6, 0.4]),
        factor_vm=jnp.ones((2, 2)),
        wpf0=jnp.asarray([1.0, 2.0]),
        wpf2=jnp.asarray([3.0, 4.0]),
        whf0=jnp.asarray([5.0, 6.0]),
        whf2=jnp.asarray([7.0, 8.0]),
        wf1=jnp.asarray([9.0, 10.0]),
        particle_flux_factor_vm=jnp.asarray([11.0]),
        heat_flux_factor_vm=jnp.asarray([12.0]),
        flow_factor=jnp.asarray([13.0]),
        b_over_d=jnp.ones((2, 2)),
    )

    diag_roundtrip = jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(diag)[::-1])
    pre_roundtrip = jax.tree_util.tree_unflatten(*jax.tree_util.tree_flatten(pre)[::-1])

    np.testing.assert_allclose(np.asarray(diag_roundtrip.fsab_flow), np.asarray([4.0]))
    np.testing.assert_allclose(np.asarray(pre_roundtrip.wf1), np.asarray([9.0, 10.0]))


def test_transport_weighted_sums_match_explicit_formulas_and_validate_shapes(monkeypatch) -> None:
    values_sxtz = jnp.arange(2 * 3 * 2 * 2, dtype=jnp.float64).reshape((2, 3, 2, 2))
    w_x = jnp.asarray([0.5, 1.5, 2.0], dtype=jnp.float64)
    expected_x = np.einsum("x,sxtz->stz", np.asarray(w_x), np.asarray(values_sxtz))

    np.testing.assert_allclose(np.asarray(td._weighted_sum_x_fortran(w_x, values_sxtz)), expected_x)
    np.testing.assert_allclose(
        np.asarray(td._weighted_sum_x_fortran(w_x, values_sxtz, strict=True)),
        expected_x,
    )
    with pytest.raises(ValueError, match="expected 3"):
        td._weighted_sum_x_fortran(jnp.ones((2,)), values_sxtz)

    values_stz = jnp.arange(2 * 2 * 2, dtype=jnp.float64).reshape((2, 2, 2))
    w_t = jnp.asarray([0.25, 0.75], dtype=jnp.float64)
    w_z = jnp.asarray([0.6, 0.4], dtype=jnp.float64)
    expected_tz = np.einsum("t,z,stz->s", np.asarray(w_t), np.asarray(w_z), np.asarray(values_stz))
    np.testing.assert_allclose(np.asarray(td._weighted_sum_tz_fortran(w_t, w_z, values_stz)), expected_tz)
    monkeypatch.setattr(td, "_STRICT_SUM_ORDER", True)
    np.testing.assert_allclose(np.asarray(td._weighted_sum_tz_fortran(w_t, w_z, values_stz)), expected_tz)
    with pytest.raises(ValueError, match="Weight shapes"):
        td._weighted_sum_tz_fortran(jnp.ones((3,)), w_z, values_stz)

    expected_sx = np.einsum("t,z,sxtz->sx", np.asarray(w_t), np.asarray(w_z), np.asarray(values_sxtz))
    np.testing.assert_allclose(np.asarray(td._weighted_sum_tz_fortran_sx(w_t, w_z, values_sxtz)), expected_sx)
    np.testing.assert_allclose(
        np.asarray(td._weighted_sum_tz_fortran_sx(w_t, w_z, values_sxtz, strict=True)),
        expected_sx,
    )
    with pytest.raises(ValueError, match="Weight shapes"):
        td._weighted_sum_tz_fortran_sx(w_t, jnp.ones((3,)), values_sxtz)


def test_f0_l0_phi1_override_and_full_f_layout_match_v3_maxwellian() -> None:
    op = _minimal_transport_diag_op(n_xi=3)
    f0 = np.asarray(td.f0_l0_v3_from_operator(op))
    pref = (
        float(op.n_hat[0])
        * float(op.m_hat[0])
        / (np.pi * float(op.t_hat[0]))
        * np.sqrt(float(op.m_hat[0]) / (np.pi * float(op.t_hat[0])))
        * np.exp(-(np.asarray(op.x) ** 2))
    )

    np.testing.assert_allclose(f0[0, :, 0, 0], pref)

    phi1 = jnp.asarray([[0.0, 0.5], [1.0, -0.25]], dtype=jnp.float64)
    f0_phi1 = np.asarray(td.f0_l0_v3_from_operator_phi1(op, phi1))
    factor = np.exp(-float(op.z_s[0]) * float(op.alpha) * np.asarray(phi1) / float(op.t_hat[0]))
    np.testing.assert_allclose(f0_phi1[0, 1, :, :], pref[1] * factor)

    full = np.asarray(td.f0_v3_from_operator(op))
    assert full.shape == op.fblock.f_shape
    np.testing.assert_allclose(full[:, :, 0, :, :], f0)
    np.testing.assert_allclose(full[:, :, 1:, :, :], 0.0)


def test_radial_current_coordinate_conversion_and_validation(monkeypatch) -> None:
    monkeypatch.setattr(
        td,
        "radial_current_vm_psi_hat_from_state",
        lambda _op, *, x_full: jnp.asarray(2.0, dtype=jnp.float64),
    )
    op = SimpleNamespace(total_size=1)
    state = jnp.zeros((1,), dtype=jnp.float64)

    assert float(td.radial_current_vm_from_state(op, x_full=state, radial_coordinate="psiHat")) == pytest.approx(2.0)
    assert float(
        td.radial_current_vm_from_state(
            op,
            x_full=state,
            radial_coordinate="rHat",
            psi_a_hat=4.0,
            a_hat=2.0,
            r_n=0.5,
        )
    ) == pytest.approx(1.0)
    assert float(
        td.radial_current_vm_from_state(
            op,
            x_full=state,
            radial_coordinate="rN",
            psi_a_hat=4.0,
            a_hat=2.0,
            r_n=0.5,
        )
    ) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="required for rHat/rN"):
        td.radial_current_vm_from_state(op, x_full=state, radial_coordinate="rHat")
    with pytest.raises(ValueError, match="radial_coordinate"):
        td.radial_current_vm_from_state(
            op,
            x_full=state,
            radial_coordinate="psiN",
            psi_a_hat=4.0,
            a_hat=2.0,
            r_n=0.5,
        )
