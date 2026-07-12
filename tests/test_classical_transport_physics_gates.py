from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.moments import FluxSurface, SpeciesParams, classical_fluxes


def _two_species_classical_case() -> dict[str, object]:
    b_hat = jnp.asarray([[1.0, 1.1], [0.9, 1.2]], dtype=jnp.float64)
    d_hat = jnp.asarray([[1.2, 1.1], [0.9, 1.3]], dtype=jnp.float64)
    theta_weights = jnp.asarray([0.6, 0.4], dtype=jnp.float64)
    zeta_weights = jnp.asarray([0.7, 0.3], dtype=jnp.float64)
    w = theta_weights[:, None] * zeta_weights[None, :]
    vprime = jnp.sum(w / d_hat)
    fsab2 = jnp.sum(w * b_hat**2 / d_hat) / vprime
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
        fsab_hat2=fsab2,
    )
    species = SpeciesParams(
        z_s=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        m_hat=jnp.asarray([1.0, 4.0], dtype=jnp.float64),
        t_hat=jnp.asarray([1.0, 0.7], dtype=jnp.float64),
        n_hat=jnp.asarray([1.0, 0.4], dtype=jnp.float64),
    )
    return {
        "use_phi1": False,
        "surface": surface,
        "species": species,
        "gpsipsi": jnp.asarray([[0.8, 1.0], [1.4, 0.7]], dtype=jnp.float64),
        "phi1_hat": zeros,
        "alpha": jnp.asarray(0.3, dtype=jnp.float64),
        "delta": jnp.asarray(0.02, dtype=jnp.float64),
        "nu_n": jnp.asarray(0.5, dtype=jnp.float64),
        "dn_hat_dpsi_hat": jnp.asarray([-0.8, 0.2], dtype=jnp.float64),
        "dt_hat_dpsi_hat": jnp.asarray([-0.3, 0.1], dtype=jnp.float64),
    }


def test_classical_flux_zero_drives_vanish_and_prefactors_scale_exactly() -> None:
    """Classical fluxes must vanish without thermodynamic drives and scale by v3 prefactors."""

    base = _two_species_classical_case()
    pf_base, hf_base = classical_fluxes(**base)
    assert np.all(np.abs(np.asarray(pf_base)) > 0.0)
    assert np.all(np.abs(np.asarray(hf_base)) > 0.0)

    zero_drive = {
        **base,
        "dn_hat_dpsi_hat": jnp.zeros(2, dtype=jnp.float64),
        "dt_hat_dpsi_hat": jnp.zeros(2, dtype=jnp.float64),
    }
    pf_zero, hf_zero = classical_fluxes(**zero_drive)
    np.testing.assert_allclose(pf_zero, 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(hf_zero, 0.0, rtol=0.0, atol=0.0)

    scaled = {
        **base,
        "delta": jnp.asarray(0.04, dtype=jnp.float64),
        "nu_n": jnp.asarray(1.5, dtype=jnp.float64),
        "gpsipsi": base["gpsipsi"] * 3.0,
    }
    pf_scaled, hf_scaled = classical_fluxes(**scaled)

    expected_scale = (0.04 / 0.02) ** 2 * (1.5 / 0.5) * 3.0
    np.testing.assert_allclose(pf_scaled / pf_base, expected_scale, rtol=2.0e-15, atol=2.0e-15)
    np.testing.assert_allclose(hf_scaled / hf_base, expected_scale, rtol=2.0e-15, atol=2.0e-15)
