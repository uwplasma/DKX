"""The DKX bootstrap current along the VMEX -> Boozer -> DKX chain, and its derivative.

``dkx.bootstrap.KineticBootstrapMismatch`` is a residual row of a VMEX
optimization, differentiated by VMEX's implicit Jacobian.  Checks, against
central finite differences at the plan's 1% admission bar where a derivative
is involved:

- the DKX link alone, from the Boozer spectrum to ``<j.B>``, on the committed
  fixture (no optional packages; runs in default CI);
- ``psiAHat`` enters the operator only through the gradient leaves, which is
  what lets the objective convert the Boozer route's handedness inside a trace;
- the objective itself on a VMEX solve of the seed: its sign against Redl and
  its Jacobian row through ``VmecProblem`` on three boundary coefficients.  It
  needs ``vmex`` and ``booz_xform_jax`` and compiles for minutes, so it is
  marked slow.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dkx
from dkx.run import profile_moments_from_operator
from dkx.solve import solve
from dkx.units import PARALLEL_CURRENT
from dkx.bootstrap import KineticBootstrapMismatch
from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import SfincsInput
from dkx.namelist import parse_sfincs_input_text
from dkx.workflows.geometry_adapters import (
    PSI_A_HAT_LEAVES,
    boozer_route_psi_a_hat,
    convert_boozer_route_handedness,
    kinetic_operator_on_boozer_surface,
)

jax.config.update("jax_enable_x64", True)

REF = Path(__file__).resolve().parent / "ref"
FIXTURE = json.loads((REF / "boozer_route_sign_vmex_seed_nfp2.json").read_text())
SEED = Path(__file__).resolve().parents[1] / "examples" / "data" / "input.minimal_seed_nfp2"
N0, T0 = 1.0e19, 2.0e3
GRID = dict(Ntheta=9, Nzeta=9, Nxi=12, NL=4, Nx=4)


def _template(s: float, psi_a_hat: float, grid: dict = GRID):
    n_hat, t_hat = N0 / 1e20 * (1 - s**5), T0 / 1e3 * (1 - s)
    return dkx.run(
        geometryScheme=1, psiAHat=psi_a_hat, aHat=0.1,
        inputRadialCoordinate=1, inputRadialCoordinateForGradients=1, psiN_wish=s,
        Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[n_hat, n_hat], THats=[t_hat, t_hat],
        dNHatdpsiNs=[-5 * N0 / 1e20 * s**4] * 2, dTHatdpsiNs=[-T0 / 1e3] * 2,
        collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0,
        **grid, emit=None,
    ).operator  # fmt: skip


def _jdotb(operator) -> jnp.ndarray:
    rhs = operator.rhs()
    solved = solve(operator, rhs, method="auto", tol=1e-10, differentiable=True,
                   tier1_keep_lowest=operator.n_xi, emit=None)  # fmt: skip
    moments = profile_moments_from_operator(operator, solved.x.reshape(-1))
    return moments["FSABjHat"].reshape(()) * PARALLEL_CURRENT / 1e6


def _relative(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-300)


def test_boozer_spectrum_to_bootstrap_gradient_matches_central_differences() -> None:
    ixm, ixn = np.asarray(FIXTURE["ixm_b"]), np.asarray(FIXTURE["ixn_b"])
    nfp = int(FIXTURE["nfp"])
    psi = boozer_route_psi_a_hat(FIXTURE["phi_edge"], FIXTURE["signgs"], FIXTURE["bvco_b"])
    template = _template(float(FIXTURE["s"]), psi)
    # Parameters: |B| harmonics (1, 0) and (1, nfp), and iota.
    k10 = int(np.flatnonzero((ixm == 1) & (ixn == 0))[0])
    k11 = int(np.flatnonzero((ixm == 1) & (ixn == nfp))[0])
    bmnc0 = jnp.asarray(FIXTURE["bmnc_b"])

    def current(p):
        bmnc = bmnc0.at[k10].add(p[0]).at[k11].add(p[1])
        operator = kinetic_operator_on_boozer_surface(
            template, bmnc_b=bmnc, ixm_b=ixm, ixn_b=ixn, nfp=nfp,
            iota=FIXTURE["iota_b"] + p[2], g_hat=FIXTURE["bvco_b"], i_hat=FIXTURE["buco_b"])
        return _jdotb(operator)

    p0 = jnp.zeros(3)
    gradient = np.asarray(jax.jit(jax.grad(current))(p0))
    value = jax.jit(current)
    for k, step in enumerate((1e-5, 1e-5, 1e-5)):
        e = jnp.zeros(3).at[k].set(step)
        central = (float(value(p0 + e)) - float(value(p0 - e))) / (2 * step)
        assert _relative(gradient[k], central) < 1e-2, (k, gradient[k], central)


def _float_leaves(operator) -> dict[str, np.ndarray]:
    leaves = {}
    for name, value in vars(operator).items():
        array = np.asarray(value) if isinstance(value, (np.ndarray, jnp.ndarray, float)) else None
        if array is not None and array.dtype.kind == "f":
            leaves[name] = array
    return leaves


def _operator(psi_a_hat: float, er: float):
    deck = SfincsInput.from_params(
        geometryScheme=1, psiAHat=psi_a_hat, aHat=0.3, inputRadialCoordinate=1,
        inputRadialCoordinateForGradients=4, psiN_wish=0.5, Zs=[1.0, -1.0],
        mHats=[1.0, 5.446170214e-4], nHats=[0.5, 0.5], THats=[1.0, 1.2],
        dNHatdrHats=[-0.3, -0.3], dTHatdrHats=[-1.0, -1.1], collisionOperator=1,
        Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=er, Ntheta=5, Nzeta=5, Nxi=6, NL=4, Nx=3,
    )  # fmt: skip
    return kinetic_operator_from_namelist(parse_sfincs_input_text(deck.to_namelist()))


def test_psi_a_hat_enters_the_operator_only_through_the_gradient_leaves() -> None:
    """The traced handedness conversion rests on this: nothing else sees ``psiAHat``."""
    unit, flipped, scaled = _operator(1.0, 2.0), _operator(-1.0, 2.0), _operator(0.05, 2.0)
    unit_leaves = _float_leaves(unit)
    changed = set()
    for other, factor in ((flipped, -1.0), (scaled, 1.0 / 0.05)):
        for name, value in _float_leaves(other).items():
            if value.shape == unit_leaves[name].shape and not np.allclose(value, unit_leaves[name]):
                changed.add(name)
                np.testing.assert_allclose(value, factor * unit_leaves[name], rtol=1e-12)
    assert changed == set(PSI_A_HAT_LEAVES)
    converted = convert_boozer_route_handedness(unit, signgs=-1, g_hat=1.4, iota=-0.4, i_hat=0.0)
    for name in PSI_A_HAT_LEAVES:
        np.testing.assert_allclose(np.asarray(getattr(converted, name)),
                                   np.asarray(getattr(flipped, name)), rtol=1e-12)  # fmt: skip


def test_kinetic_surfaces_move_to_the_nearest_half_mesh_rows() -> None:
    class Profiles:
        ne_coeffs, Te_coeffs, Ti_coeffs = [1e19], [1e3], [1e3]

    rows, s = KineticBootstrapMismatch(Profiles(), surfaces=[0.25, 0.46, 0.5, 0.75]).rows(13)
    np.testing.assert_array_equal(rows, [4, 6, 10])  # 0.46 and 0.5 share row 6
    np.testing.assert_allclose(s, (rows - 0.5) / 12)
    with pytest.raises(ValueError):
        KineticBootstrapMismatch(Profiles(), surfaces=[0.0, 0.5])


@pytest.mark.slow
def test_the_objective_agrees_with_redl_in_sign_and_its_jacobian_with_differences() -> None:
    vj = pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax.jax_api", reason="needs booz_xform_jax >= 0.4")
    from vmex import optimize as opt
    from vmex.core.bootstrap import KineticProfiles, j_dot_B_redl, redl_geometry_from_state

    inp = vj.VmecInput.from_file(SEED)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(mpol=4, ntor=4, ntheta=14, nzeta=12)
    inp = replace(inp, ns_array=np.array([13]), ftol_array=np.array([1e-14]),
                  niter_array=np.array([20000]))  # fmt: skip
    profiles = KineticProfiles(N0 * np.array([1, 0, 0, 0, 0, -1.0]), T0 * np.array([1, -1.0]),
                               T0 * np.array([1, -1.0]))  # fmt: skip

    # The value: the fixture's surface and grid, which the Boozer-route sign
    # test pins against the VMEC-file route.  The seed is a vacuum field, so
    # the mismatch row is exactly -1 and the current row carries the physics.
    equilibrium = opt.solve_equilibrium(inp)
    fine = dict(Ntheta=11, Nzeta=11, Nxi=16, NL=4, Nx=4)
    term = KineticBootstrapMismatch(profiles, surfaces=[FIXTURE["s"]], resolution=fine)
    s, j_vmex, j_dkx = term.current_profiles(equilibrium.state, equilibrium.runtime)
    np.testing.assert_allclose(s, [FIXTURE["s"]])
    geometry = redl_geometry_from_state(equilibrium.state, equilibrium.runtime, surfaces=s)
    j_redl = float(j_dot_B_redl(profiles, geometry, 0)[0][0])
    assert j_redl > 0.0 and float(j_dkx[0]) > 0.0, (j_redl, j_dkx)
    # Pitch-angle scattering on a non-quasisymmetric seed: same sign, same
    # order.  Measured: DKX/Redl = 1.018 at this grid.
    assert 0.7 < float(j_dkx[0]) / j_redl < 1.5
    assert abs(float(j_vmex[0])) < 1e-3 * j_redl
    np.testing.assert_allclose(term.residuals_state(equilibrium.state, equilibrium.runtime), [-1.0])

    # The derivative: VMEX's implicit Jacobian of the current row against
    # central differences of the same residual, one re-solve per point.
    row = KineticBootstrapMismatch(profiles, surfaces=[FIXTURE["s"]], mismatch=False, resolution=GRID)
    problem = opt.VmecProblem.from_tuples(inp, [(row, 0.0, 1.0)], max_mode=1)
    jacobian = np.asarray(problem.residual_jac(problem.x0))[0]
    names = list(problem.dof_names)
    for name in ("RBC(1,0)", "RBC(0,1)", "ZBS(-1,1)"):  # (m, n) = (0, 1), (1, 0), (1, -1)
        k = names.index(name)
        e = np.zeros_like(problem.x0)
        e[k] = 1e-4
        central = float((problem.residual(problem.x0 + e) - problem.residual(problem.x0 - e))[0]) / 2e-4
        assert _relative(jacobian[k], central) < 1e-2, (name, jacobian[k], central)
