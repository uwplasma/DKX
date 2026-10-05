"""Native ``physics.phi1 = "kinetic"``: parity with the SFINCS-namelist Phi1 route."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from dkx.execution import run_case
from dkx.input_compat import case_from_sfincs_namelist
from dkx.phi1 import solve_phi1
from dkx.run import profile_moments_from_operator

_DECK = (
    Path(__file__).resolve().parents[1]
    / "examples/sfincs_examples/geometryScheme4_2species_noEr_withPhi1InDKE"
)


def _deck(tmp_path, **resolution):
    shutil.copytree(_DECK, tmp_path, dirs_exist_ok=True)
    path = tmp_path / "input.namelist"
    text = path.read_text()
    for key, value in resolution.items():
        text = re.sub(rf"(?m)^\s*{key}\s*=.*$", f"  {key} = {value}", text)
    path.write_text(text)
    return path


def test_native_phi1_matches_the_namelist_newton_route(tmp_path):
    deck = _deck(tmp_path, Ntheta=9, Nzeta=9, Nxi=12, Nx=4)
    case = case_from_sfincs_namelist(deck, name="phi1")
    assert case.physics.phi1 == "kinetic"
    native = run_case(case, out=tmp_path / "native.nc")
    # Accepted by the original coupled residual (kinetic + QN + gauge).
    assert np.all(native.primal_residual <= case.solver.relative_tolerance * native.primal_rhs_norm)
    reference = solve_phi1(deck, tol=1.0e-12)
    flux = np.asarray(
        profile_moments_from_operator(reference.operator, reference.x)["particleFlux_vm_psiHat"]
    )
    # The deck's own surface is the middle of the converted three-surface stencil;
    # the native flux is the psiHat flux times one species-independent unit factor.
    ratio = np.asarray(native.particle_flux_m2_s)[1] / flux
    np.testing.assert_allclose(ratio[0], ratio[1], rtol=1.0e-8)


def test_native_phi1_refuses_what_it_does_not_implement(tmp_path):
    deck = _deck(tmp_path)
    text = deck.read_text().replace(
        "includePhi1 = .true.", "includePhi1 = .true.\n  quasineutralityOption = 2"
    )
    deck.write_text(text)
    with pytest.raises(Exception, match="includePhi1"):
        case_from_sfincs_namelist(deck)


def test_native_phi1_adjoint_matches_finite_differences_and_taylor(tmp_path):
    import jax
    import jax.numpy as jnp
    from dataclasses import replace

    from dkx.execution import _make_operator, _prepare_profile
    from dkx.phi1 import phi1_solution

    case = case_from_sfincs_namelist(_deck(tmp_path, Ntheta=7, Nzeta=7, Nxi=8, Nx=3))
    geometry, grids, density, temperature, dn, dt = _prepare_profile(case)
    base, *_ = _make_operator(
        case, surface_index=1, n_hat=density / 1.0e20, t_hat=temperature,
        dn_dr_hat=dn, dt_dr_hat=dt, grids=grids, geometry_state=geometry,
    )
    assert base.active_dof_mask() is not None  # Legendre-truncated layout

    def objective(p):
        op = replace(base, dt_hat_dpsi_hat=base.dt_hat_dpsi_hat * p)
        x = phi1_solution(op, tol=1e-13)
        moments = profile_moments_from_operator(replace(op, phi1_lin_state=x), x)
        return moments["particleFlux_vm_psiHat"][0]

    p0 = jnp.asarray(1.0)
    value, grad = jax.value_and_grad(objective)(p0)
    h = 1.0e-3
    fd = (objective(p0 + h) - objective(p0 - h)) / (2 * h)
    np.testing.assert_allclose(float(grad), float(fd), rtol=1e-5)
    # Taylor test: the first-order remainder falls as h^2.
    rem = [abs(float(objective(p0 + e) - value - e * grad)) for e in (4e-2, 2e-2, 1e-2)]
    assert rem[0] / rem[1] > 3.5 and rem[1] / rem[2] > 3.5
