"""The derivative of the DKX bootstrap current along the VMEX -> Boozer -> DKX chain.

Rung 08 optimizes a VMEX boundary with a DKX ``<j.B>`` row, differentiated by
VMEX's forward implicit Jacobian.  Two checks guard that derivative against
central finite differences at the plan's 1% admission bar:

- the DKX link alone, from the Boozer spectrum to ``<j.B>``, on the committed
  fixture (no optional packages; runs in default CI);
- the whole chain in reverse mode, boundary coefficient -> ``solve_implicit``
  -> ``boozer_input_tables`` -> ``booz_xform_jax`` -> DKX, on three
  coefficients spanning poloidal modes 0 and 1.  It needs ``vmex`` and
  ``booz_xform_jax`` and compiles for minutes, so it is marked slow.
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
from dkx.workflows.geometry_adapters import (
    boozer_route_psi_a_hat,
    kinetic_operator_on_boozer_surface,
)

jax.config.update("jax_enable_x64", True)

REF = Path(__file__).resolve().parent / "ref"
FIXTURE = json.loads((REF / "boozer_route_sign_vmex_seed_nfp2.json").read_text())
SEED = Path(__file__).resolve().parents[1] / "examples" / "08_vmex_optimization" / "input.minimal_seed_nfp2"
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


@pytest.mark.slow
def test_boundary_to_bootstrap_gradient_matches_central_differences() -> None:
    vj = pytest.importorskip("vmex")
    pytest.importorskip("booz_xform_jax")
    from booz_xform_jax.jax_api import (
        BoozerConfig,
        booz_xform_jax_impl,
        prepare_booz_xform_plan,
    )
    from vmex import optimize as opt
    from vmex.core import implicit
    from vmex.core.boozer_tables import boozer_input_tables

    inp = vj.VmecInput.from_file(SEED)
    rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
    rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
    inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(mpol=4, ntor=4, ntheta=14, nzeta=12)
    inp = replace(inp, ns_array=np.array([13]), ftol_array=np.array([1e-14]),
                  niter_array=np.array([20000]))  # fmt: skip
    ns = int(inp.ns_array[-1])
    row, nfp = ns // 2, int(inp.nfp)
    s = (row - 0.5) / (ns - 1)
    cfg = implicit.make_config(inp, adjoint_tol=1e-12, adjoint_maxiter=400)
    params0 = implicit.params_from_input(inp)
    modes = opt._dof_modes(inp, 1)
    names = opt.boundary_dof_names(inp, 1)
    rows = np.asarray([n + int(inp.ntor) for (_, n) in modes])
    cols = np.asarray([m for (m, _) in modes])
    dofs0 = jnp.asarray(opt.pack_boundary(inp, 1))
    state0 = implicit.solve_implicit(params0, cfg)
    tables = boozer_input_tables(state0, implicit.runtime_from_params(params0, cfg), row)
    xm, xn = np.asarray(tables["xm"]), np.asarray(tables["xn"])
    plan = prepare_booz_xform_plan(nfp=nfp, asym=False, xm=xm, xn=xn, xm_nyq=xm, xn_nyq=xn,
                                   config=BoozerConfig.from_env(mboz=6, nboz=6))  # fmt: skip
    modes_kw = dict(xm=jnp.asarray(xm), xn=jnp.asarray(xn), xm_nyq=jnp.asarray(xm),
                    xn_nyq=jnp.asarray(xn), constants=plan.constants, grids=plan.grids, plan=plan)
    keys = ("rmnc", "zmns", "lmns", "bmnc", "bsubumnc", "bsubvmnc", "iota")
    booz0 = booz_xform_jax_impl(**{k: jnp.asarray(tables[k])[None] for k in keys}, **modes_kw)
    ixm, ixn = np.asarray(booz0["ixm_b"]), np.asarray(booz0["ixn_b"])
    psi = boozer_route_psi_a_hat(float(inp.phiedge), -1, float(booz0["bvco_b"][0]))
    template = _template(s, psi, dict(Ntheta=9, Nzeta=9, Nxi=12, NL=4, Nx=4))

    def current(dofs):
        m = len(modes)
        params = replace(params0, rbc=params0.rbc.at[rows, cols].set(dofs[:m]),
                         zbs=params0.zbs.at[rows, cols].set(dofs[m:]))  # fmt: skip
        state = implicit.solve_implicit(params, cfg)
        surface = boozer_input_tables(state, implicit.runtime_from_params(params, cfg), row)
        booz = booz_xform_jax_impl(**{k: jnp.asarray(surface[k])[None] for k in keys}, **modes_kw)
        operator = kinetic_operator_on_boozer_surface(
            template, bmnc_b=booz["bmnc_b"][0], ixm_b=ixm, ixn_b=ixn, nfp=nfp,
            iota=booz["iota_b"][0], g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0])
        return _jdotb(operator)

    gradient = np.asarray(jax.grad(current)(dofs0))
    for name in ("RBC(1,0)", "RBC(0,1)", "ZBS(-1,1)"):  # (m, n) = (0, 1), (1, 0), (1, -1)
        k = names.index(name)
        e = jnp.zeros_like(dofs0).at[k].set(1e-4)
        central = (float(current(dofs0 + e)) - float(current(dofs0 - e))) / 2e-4
        assert _relative(gradient[k], central) < 1e-2, (name, gradient[k], central)
