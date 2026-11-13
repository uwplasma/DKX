"""Small jitted solves meant for a GPU backend (``pytest -m gpu``).

The sizes are the teaching grids of the README, so the tests finish in
seconds on any backend. They run on CPU as well, where the coverage shards pick
them up; no CI job needs a GPU runner, so no required check depends on one.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

import dkx

pytestmark = pytest.mark.gpu


def _tokamak_case(pitch: int = 8) -> "dkx.Case":
    return dkx.Case.from_mapping({
        "name": "tokamak", "run": {"workflow": "profile", "progress": False},
        "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
        "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                     "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
        "physics": {"model": "full_local", "collisions": "pitch_angle_scattering",
                    "magnetic_drifts": "dkes", "phi1": "off"},
        "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
        "resolution": {"theta": 9, "zeta": 1, "pitch": pitch, "speed": 4},
        "solver": {"method": "auto", "relative_tolerance": 1e-8},
    })  # fmt: skip


@pytest.fixture(scope="module")
def tokamak_result():
    return dkx.run(_tokamak_case())


def test_profile_solve_returns_finite_nonzero_flux(tokamak_result):
    flux = np.asarray(tokamak_result.arrays["particle_flux_m2_s"])
    assert np.all(np.isfinite(flux))
    assert np.any(flux != 0.0)


def test_jitted_er_scan_gradient_matches_central_difference():
    import jax.numpy as jnp

    if not jax.config.jax_enable_x64:
        pytest.skip("the central-difference check needs float64")
    problem = dkx.prepare_er_scan(_tokamak_case(pitch=6), surface_index=1)

    def bootstrap_current(er_kv_m):
        scan = dkx.batched_er_scan(problem, er_kv_m, differentiable=True, retain_full_state=True)
        return jnp.sum(scan.moments["FSABjHat"])

    er = jnp.array([-0.2, 0.2])
    value, grad = jax.jit(jax.value_and_grad(bootstrap_current))(er)
    assert np.isfinite(float(value)) and np.all(np.isfinite(np.asarray(grad)))
    step = 1e-3
    unit = jnp.zeros_like(er).at[0].set(step)
    fd = (float(bootstrap_current(er + unit)) - float(bootstrap_current(er - unit))) / (2 * step)
    assert float(grad[0]) == pytest.approx(fd, rel=1e-4, abs=1e-9)


def test_single_point_er_scan_at_zero_field_is_finite():
    import jax.numpy as jnp

    problem = dkx.prepare_er_scan(_tokamak_case(pitch=6), surface_index=1)
    scan = dkx.batched_er_scan(problem, jnp.array([0.0]), differentiable=False)
    assert np.all(np.isfinite(np.asarray(scan.moments["FSABjHat"])))
