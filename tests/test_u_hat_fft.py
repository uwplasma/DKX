from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.diagnostics import _u_hat_loop, u_hat, u_hat_np
from sfincs_jax.geometry import boozer_geometry_scheme1, boozer_geometry_scheme4
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist


def test_u_hat_fft_matches_numpy_reference_for_scheme4_fixture() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)

    u_fft = np.asarray(u_hat(grids=grids, geom=geom))
    u_np = u_hat_np(grids=grids, geom=geom)

    np.testing.assert_allclose(u_fft, u_np, rtol=0, atol=1e-11)


def test_u_hat_is_differentiable_wrt_scheme4_harmonics() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    nml = read_sfincs_input(input_path)
    grids = grids_from_namelist(nml)

    amp0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)

    def objective(a: jnp.ndarray) -> jnp.ndarray:
        geom = boozer_geometry_scheme4(theta=grids.theta, zeta=grids.zeta, harmonics_amp0=a)
        u = u_hat(grids=grids, geom=geom)
        return jnp.sum(u * u)

    g = jax.grad(objective)(amp0)
    assert g.shape == amp0.shape
    assert bool(jnp.all(jnp.isfinite(g)))


def test_u_hat_loop_returns_finite_field_on_even_periodic_cosine_geometry() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    geom = boozer_geometry_scheme4(
        theta=theta,
        zeta=zeta,
        harmonics_amp0=np.asarray([0.08, -0.03, 0.02], dtype=np.float64),
        iota=0.4,
        g_hat=5.0,
        i_hat=0.2,
        b0_over_bbar=1.6,
    )
    grids = type("Grids", (), {"theta": theta, "zeta": zeta})()

    u_loop = np.asarray(_u_hat_loop(grids=grids, geom=geom))
    assert u_loop.shape == geom.b_hat.shape
    assert np.all(np.isfinite(u_loop))


def test_u_hat_loop_returns_finite_field_on_odd_periodic_cosine_geometry() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 7, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 5, endpoint=False)
    geom = boozer_geometry_scheme4(
        theta=theta,
        zeta=zeta,
        harmonics_amp0=np.asarray([0.08, -0.03, 0.02], dtype=np.float64),
        iota=0.6,
        g_hat=3.2,
        i_hat=-0.1,
        b0_over_bbar=1.4,
    )
    grids = type("Grids", (), {"theta": theta, "zeta": zeta})()

    u_loop = np.asarray(_u_hat_loop(grids=grids, geom=geom))
    assert u_loop.shape == geom.b_hat.shape
    assert np.all(np.isfinite(u_loop))


def test_u_hat_loop_handles_resonant_mode_denominator_without_nan() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    geom = boozer_geometry_scheme1(
        theta=theta,
        zeta=zeta,
        epsilon_t=0.0,
        epsilon_h=0.08,
        epsilon_antisymm=0.0,
        iota=1.0,
        g_hat=4.0,
        i_hat=0.3,
        b0_over_bbar=1.8,
        helicity_l=1,
        helicity_n=1,
        helicity_antisymm_l=0,
        helicity_antisymm_n=0,
    )
    grids = type("Grids", (), {"theta": theta, "zeta": zeta})()

    u_loop = np.asarray(_u_hat_loop(grids=grids, geom=geom))
    assert np.all(np.isfinite(u_loop))
    assert u_loop.shape == geom.b_hat.shape


def test_u_hat_loop_is_spatially_constant_for_constant_bhat() -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    geom = type(
        "Geom",
        (),
        {
            "b_hat": np.full((8, 6), 3.0),
            "iota": 0.4,
            "g_hat": 1.3,
            "i_hat": -0.7,
            "n_periods": 2,
        },
    )()
    grids = type("Grids", (), {"theta": theta, "zeta": zeta})()
    out = np.asarray(_u_hat_loop(grids=grids, geom=geom))
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, np.full_like(out, out[0, 0]), rtol=0.0, atol=1e-12)
