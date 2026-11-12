"""The default Fokker-Planck matrix build is traced and matches QUADPACK."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.collisions import make_fokker_planck_v3_operator
from dkx.drift_kinetic import KineticOperator
from dkx.inputs import load_sfincs_input
from dkx.namelist import read_sfincs_input
from dkx.run import _grids_from_input, profile_moments_from_operator
from dkx.solve import solve

REF = "tests/ref/"


def _builder_kwargs(name: str) -> dict:
    inp = load_sfincs_input(REF + name)
    nml = inp.raw
    grids = _grids_from_input(inp, nml)
    species, physics = nml.group("speciesParameters"), nml.group("physicsParameters")
    vec = lambda key: np.atleast_1d(np.asarray(species[key], dtype=np.float64))
    return dict(
        x=np.asarray(grids.x), x_weights=np.asarray(grids.x_weights),
        ddx=np.asarray(grids.ddx), d2dx2=np.asarray(grids.d2dx2), x_grid_k=0.0,
        z_s=vec("ZS"), m_hats=vec("MHATS"), n_hats=vec("NHATS"), t_hats=vec("THATS"),
        nu_n=float(physics["NU_N"]), krook=0.0, n_xi=int(grids.n_xi), nl=int(grids.n_l),
        n_xi_for_x=np.asarray(grids.n_xi_for_x, dtype=np.int32),
    )


@pytest.mark.parametrize("deck", [
    "quick_2species_FPCollisions_noEr.input.namelist",
    "multispecies_HSX_FPCollisions_DKESTrajectories.input.namelist",  # electron-ion
    "fp_1species_FPCollisions_noEr_tiny_cs3.input.namelist",
])
def test_default_matrices_match_quadpack(deck):
    kwargs = _builder_kwargs(deck)
    default = make_fokker_planck_v3_operator(**kwargs)
    quadpack = make_fokker_planck_v3_operator(**kwargs, rosenbluth_method="quadpack")
    reference = np.asarray(quadpack.mat)
    assert np.max(np.abs(np.asarray(default.mat) - reference)) <= 1e-12 * np.max(np.abs(reference))
    np.testing.assert_array_equal(default.mask_xi, quadpack.mask_xi)


def test_default_build_is_cached_and_traceable():
    kwargs = _builder_kwargs("quick_2species_FPCollisions_noEr.input.namelist")
    first = make_fokker_planck_v3_operator(**kwargs)
    again = make_fokker_planck_v3_operator(**{**kwargs, "n_hats": kwargs["n_hats"] * 2.0})
    assert not np.allclose(again.mat, first.mat)
    traced = jax.jit(lambda t: make_fokker_planck_v3_operator(**{**kwargs, "t_hats": t}).mat)
    np.testing.assert_allclose(traced(jnp.asarray(kwargs["t_hats"])), first.mat, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("observable", ["FSABjHat", "heatFlux_vm_psiHat"])
def test_flux_gradient_wrt_temperature_ratio_matches_central_difference(observable):
    kwargs = _builder_kwargs("quick_2species_FPCollisions_noEr.input.namelist")
    base = KineticOperator.from_namelist(read_sfincs_input(
        REF + "quick_2species_FPCollisions_noEr.input.namelist"))

    def flux(ratio):
        # Species 1 temperature scales with `ratio`; the collision matrices
        # must follow, so they are rebuilt from the traced temperatures.
        t = jnp.asarray(kwargs["t_hats"]) * jnp.asarray([1.0, 1.0]).at[1].set(ratio)
        fp = make_fokker_planck_v3_operator(**{**kwargs, "t_hats": t})
        op = replace(base, t_hat=t, fp=fp)
        state = solve(op, op.rhs(), method="gmres", tol=1e-12, differentiable=True).x
        return jnp.ravel(profile_moments_from_operator(op, state)[observable])[0]

    point, h = 1.3, 1e-5
    grad = float(jax.jit(jax.grad(flux))(point))
    fd = float((flux(point + h) - flux(point - h)) / (2 * h))
    assert abs(grad) > 1e-20
    np.testing.assert_allclose(grad, fd, rtol=1e-6)
