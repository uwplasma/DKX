"""Unit tests for :mod:`dkx.impurity` (roadmap item 4, classical impurity transport).

All cases are algebraic (no kinetic solve) so the whole file runs in well under a
second.  They pin:

* the classical flux against the SFINCS-counterpart :func:`dkx.moments.classical_fluxes`
  to machine precision (the module is the geometry-collapsed form of it);
* the exact ``-Z`` ion-density peaking coefficient and the ``-> 1/2`` collisional
  temperature-screening coefficient, plus the screening sign for peaked/hollow
  profiles;
* the ``vmap``-over-charge-state batched call against a per-element loop;
* the differentiability of the flux w.r.t. the ion temperature gradient
  (AD vs central finite difference) -- the screening-aware gradient.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from dkx import impurity as imp
from dkx.moments import FluxSurface, SpeciesParams, classical_fluxes
from dkx.species import SpeciesSet

DELTA = 0.02
NU_N = 0.5


def _reference_surface() -> tuple[FluxSurface, jnp.ndarray]:
    """A small non-axisymmetric flux surface with a nonzero ``gpsiHatpsiHat`` metric."""
    b_hat = jnp.asarray([[1.0, 1.1], [0.9, 1.2]], dtype=jnp.float64)
    d_hat = jnp.asarray([[1.2, 1.1], [0.9, 1.3]], dtype=jnp.float64)
    theta_weights = jnp.asarray([0.6, 0.4], dtype=jnp.float64)
    zeta_weights = jnp.asarray([0.7, 0.3], dtype=jnp.float64)
    zeros = jnp.zeros_like(b_hat)
    surface = FluxSurface(
        theta_weights=theta_weights,
        zeta_weights=zeta_weights,
        b_hat=b_hat,
        d_hat=d_hat,
        db_hat_dtheta=zeros,
        db_hat_dzeta=zeros,
        b_hat_sub_theta=zeros,
        b_hat_sub_zeta=zeros,
        fsab_hat2=jnp.asarray(1.0, dtype=jnp.float64),
    )
    gpsipsi = jnp.asarray([[0.8, 1.0], [1.4, 0.7]], dtype=jnp.float64)
    return surface, gpsipsi


def _geometry_factor() -> float:
    surface, gpsipsi = _reference_surface()
    return float(
        imp.classical_geometry_factor(
            theta_weights=surface.theta_weights,
            zeta_weights=surface.zeta_weights,
            d_hat=surface.d_hat,
            b_hat=surface.b_hat,
            gpsipsi=gpsipsi,
        )
    )


def _bulk() -> SpeciesSet:
    """Bulk hydrogen: peaked density and temperature (both gradients negative)."""
    return SpeciesSet(
        z=jnp.asarray([1.0], dtype=jnp.float64),
        m_hat=jnp.asarray([1.0], dtype=jnp.float64),
        n_hat=jnp.asarray([1.0], dtype=jnp.float64),
        t_hat=jnp.asarray([1.0], dtype=jnp.float64),
        dn_hat_dpsi_hat=jnp.asarray([-0.8], dtype=jnp.float64),
        dt_hat_dpsi_hat=jnp.asarray([-0.5], dtype=jnp.float64),
    )


# ---------------------------------------------------------------------------
# 1) Classical flux vs the SFINCS-counterpart algebraic reference
# ---------------------------------------------------------------------------


def test_classical_flux_matches_moments_reference_to_machine_precision() -> None:
    """The geometry-collapsed classical flux equals moments.classical_fluxes exactly."""
    surface, gpsipsi = _reference_surface()
    species = SpeciesParams(
        z_s=jnp.asarray([1.0, 6.0], dtype=jnp.float64),
        m_hat=jnp.asarray([1.0, 12.0], dtype=jnp.float64),
        t_hat=jnp.asarray([1.0, 0.9], dtype=jnp.float64),
        n_hat=jnp.asarray([1.0, 1e-3], dtype=jnp.float64),
    )
    dn = jnp.asarray([-0.8, -3e-3], dtype=jnp.float64)
    dt = jnp.asarray([-0.5, -0.4], dtype=jnp.float64)

    pf_ref, hf_ref = classical_fluxes(
        use_phi1=False, surface=surface, species=species, gpsipsi=gpsipsi,
        phi1_hat=jnp.zeros_like(gpsipsi), alpha=0.0, delta=DELTA, nu_n=NU_N,
        dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt,
    )  # fmt: skip

    g = imp.classical_geometry_factor(
        theta_weights=surface.theta_weights, zeta_weights=surface.zeta_weights,
        d_hat=surface.d_hat, b_hat=surface.b_hat, gpsipsi=gpsipsi,
    )  # fmt: skip
    pf_new, hf_new = imp.classical_species_fluxes(
        z=species.z_s, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
        dn_hat_dpsi_hat=dn, dt_hat_dpsi_hat=dt, geometry_factor=g, delta=DELTA, nu_n=NU_N,
    )  # fmt: skip

    np.testing.assert_allclose(np.asarray(pf_new), np.asarray(pf_ref), rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(np.asarray(hf_new), np.asarray(hf_ref), rtol=1e-12, atol=0.0)


def test_zero_drives_give_zero_flux_and_diffusion_is_positive() -> None:
    g = _geometry_factor()
    species = imp.build_impurity_plasma(
        _bulk(), impurity_z=6.0, impurity_m_hat=12.0, impurity_n_hat=1e-4,
        impurity_dn_hat_dpsi_hat=0.0, impurity_dt_hat_dpsi_hat=0.0,
    )  # fmt: skip
    # No drives anywhere -> zero flux.
    flat = SpeciesSet(
        z=species.z, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
        dn_hat_dpsi_hat=jnp.zeros_like(species.z), dt_hat_dpsi_hat=jnp.zeros_like(species.z),
    )  # fmt: skip
    res = imp.classical_impurity_flux(flat, geometry_factor=g, delta=DELTA, nu_n=NU_N)
    assert float(res.particle_flux) == 0.0
    assert float(res.heat_flux) == 0.0
    # The classical particle diffusion coefficient is positive.
    d_cl = imp.classical_diffusion_coefficient(species, geometry_factor=g, delta=DELTA, nu_n=NU_N)
    assert float(d_cl) > 0.0
    # And equals -d(flux)/d(dn_z) by the operational definition.
    d_ad = -float(jax.grad(
        lambda dnz: imp.classical_species_fluxes(
            z=species.z, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
            dn_hat_dpsi_hat=species.dn_hat_dpsi_hat.at[-1].set(dnz),
            dt_hat_dpsi_hat=species.dt_hat_dpsi_hat,
            geometry_factor=g, delta=DELTA, nu_n=NU_N,
        )[0][-1]
    )(float(species.dn_hat_dpsi_hat[-1])))
    np.testing.assert_allclose(float(d_cl), d_ad, rtol=1e-10, atol=0.0)


# ---------------------------------------------------------------------------
# 2) Temperature screening: exact -Z peaking, 1/2 collisional coefficient, sign
# ---------------------------------------------------------------------------


def test_ion_density_peaking_is_exactly_minus_z() -> None:
    g = _geometry_factor()
    for z_imp in (2.0, 6.0, 18.0, 74.0):
        species = imp.build_impurity_plasma(
            _bulk(), impurity_z=z_imp, impurity_m_hat=2.0 * z_imp, impurity_n_hat=1e-6,
            match_bulk_logarithmic_gradients=True,
        )  # fmt: skip
        scr = imp.temperature_screening_diagnostic(species, geometry_factor=g, delta=DELTA, nu_n=NU_N)
        np.testing.assert_allclose(float(scr.ion_density_peaking_coefficient), -z_imp, rtol=1e-10)


def test_screening_coefficient_approaches_one_half_in_heavy_limit() -> None:
    g = _geometry_factor()
    # Equal temperatures; increasing impurity mass -> collisional screening -> 1/2.
    coeffs = []
    for m_imp in (12.0, 50.0, 200.0, 1000.0):
        species = imp.build_impurity_plasma(
            _bulk(), impurity_z=6.0, impurity_m_hat=m_imp, impurity_n_hat=1e-6, impurity_t_hat=1.0,
            match_bulk_logarithmic_gradients=True,
        )  # fmt: skip
        scr = imp.temperature_screening_diagnostic(species, geometry_factor=g, delta=DELTA, nu_n=NU_N)
        coeffs.append(float(scr.screening_coefficient))
    # Monotonically increasing toward 1/2, and close to 1/2 for the heaviest.
    assert all(coeffs[i] < coeffs[i + 1] for i in range(len(coeffs) - 1))
    assert 0.30 < coeffs[0] < 0.5
    np.testing.assert_allclose(coeffs[-1], 0.5, atol=5e-3)


def test_screening_sign_flips_with_ion_temperature_gradient() -> None:
    g = _geometry_factor()
    # Peaked ion temperature (T' < 0): the ion-temperature term opposes the pinch.
    peaked = imp.build_impurity_plasma(
        _bulk(), impurity_z=6.0, impurity_m_hat=12.0, impurity_n_hat=1e-4,
        match_bulk_logarithmic_gradients=True,
    )  # fmt: skip
    scr_peaked = imp.temperature_screening_diagnostic(peaked, geometry_factor=g, delta=DELTA, nu_n=NU_N)
    assert bool(scr_peaked.screens)
    assert float(scr_peaked.density_pinch_flux) < 0.0  # inward accumulation
    assert float(scr_peaked.screening_flux) > 0.0  # outward screening

    # Hollow ion temperature (T' > 0): the ion-temperature term now enhances the pinch.
    hollow = SpeciesSet(
        z=peaked.z, m_hat=peaked.m_hat, n_hat=peaked.n_hat, t_hat=peaked.t_hat,
        dn_hat_dpsi_hat=peaked.dn_hat_dpsi_hat,
        dt_hat_dpsi_hat=peaked.dt_hat_dpsi_hat.at[0].set(+0.5),
    )  # fmt: skip
    scr_hollow = imp.temperature_screening_diagnostic(hollow, geometry_factor=g, delta=DELTA, nu_n=NU_N)
    assert not bool(scr_hollow.screens)
    assert float(scr_hollow.screening_flux) < 0.0  # inward, enhances accumulation


# ---------------------------------------------------------------------------
# 3) vmap over charge states
# ---------------------------------------------------------------------------


def test_vmap_over_charge_states_matches_loop_and_scales_with_z() -> None:
    g = _geometry_factor()
    bulk = _bulk()
    zs = jnp.asarray([2.0, 4.0, 6.0, 10.0, 18.0], dtype=jnp.float64)
    ms = 2.0 * zs
    n_imp = 1e-6

    pf, hf, dcl = imp.classical_impurity_flux_over_charge_states(
        bulk, impurity_charges=zs, impurity_masses=ms, impurity_n_hat=n_imp,
        geometry_factor=g, delta=DELTA, nu_n=NU_N,
    )  # fmt: skip
    assert pf.shape == zs.shape == hf.shape == dcl.shape

    # Element-by-element against the scalar builder + flux.
    for j, z_imp in enumerate(np.asarray(zs)):
        species = imp.build_impurity_plasma(
            bulk, impurity_z=float(z_imp), impurity_m_hat=2.0 * float(z_imp),
            impurity_n_hat=n_imp, match_bulk_logarithmic_gradients=True,
        )  # fmt: skip
        res = imp.classical_impurity_flux(species, geometry_factor=g, delta=DELTA, nu_n=NU_N)
        np.testing.assert_allclose(float(pf[j]), float(res.particle_flux), rtol=1e-12)
        np.testing.assert_allclose(float(hf[j]), float(res.heat_flux), rtol=1e-12)
        np.testing.assert_allclose(float(dcl[j]), float(res.diffusion_coefficient), rtol=1e-12)

    # Higher-Z impurities accumulate more strongly (larger inward flux magnitude).
    assert np.all(np.diff(np.abs(np.asarray(pf))) > 0.0)


# ---------------------------------------------------------------------------
# 4) Differentiability: AD vs FD of the flux w.r.t. the ion temperature gradient
# ---------------------------------------------------------------------------


def test_ion_temperature_gradient_derivative_ad_matches_fd() -> None:
    g = _geometry_factor()
    species = imp.build_impurity_plasma(
        _bulk(), impurity_z=6.0, impurity_m_hat=12.0, impurity_n_hat=1e-4,
        match_bulk_logarithmic_gradients=True,
    )  # fmt: skip

    def gamma_of_dti(dti: jnp.ndarray) -> jnp.ndarray:
        dt = species.dt_hat_dpsi_hat.at[0].set(dti)
        pf, _ = imp.classical_species_fluxes(
            z=species.z, m_hat=species.m_hat, n_hat=species.n_hat, t_hat=species.t_hat,
            dn_hat_dpsi_hat=species.dn_hat_dpsi_hat, dt_hat_dpsi_hat=dt,
            geometry_factor=g, delta=DELTA, nu_n=NU_N,
        )  # fmt: skip
        return pf[-1]

    dti0 = float(species.dt_hat_dpsi_hat[0])
    ad = float(jax.grad(gamma_of_dti)(dti0))
    h = 1e-6 * max(1.0, abs(dti0))
    fd = float((gamma_of_dti(dti0 + h) - gamma_of_dti(dti0 - h)) / (2.0 * h))
    rel = abs(ad - fd) / max(abs(ad), 1e-300)
    assert rel < 1e-6

    # The diagnostic reports exactly this AD-consistent coefficient.
    scr = imp.temperature_screening_diagnostic(species, geometry_factor=g, delta=DELTA, nu_n=NU_N)
    np.testing.assert_allclose(float(scr.ion_temperature_flux_coefficient), ad, rtol=1e-10)
