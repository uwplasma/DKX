"""WIP prototype (not the rung-08 example): boundary coefficient -> VMEX -> Boozer -> DKX <j.B>; jax.grad vs central FD.
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import vmex as vj
from vmex import optimize as opt
from vmex.core import implicit as imp
from vmex.core.boozer_tables import boozer_input_tables
from vmex.core.bootstrap import KineticProfiles, j_dot_B_redl, redl_geometry_from_state
from booz_xform_jax.jax_api import booz_xform_jax

from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import parse_sfincs_input_text
from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.phase_space import make_grids
from dkx.run import profile_moments_from_operator
from dkx.solve import solve
from dkx.units import PARALLEL_CURRENT

COLL = int(os.environ.get("COLL", "1"))
NTH, NZE, NXI, NX = (int(v) for v in os.environ.get("RES", "13,13,16,4").split(","))
MBOZ, NBOZ = 6, 6
FD_STEPS = [float(v) for v in os.environ.get("FD_STEPS", "1e-4,1e-3").split(",")]
NDOF_CHECK = int(os.environ.get("NDOF_CHECK", "3"))
FTOL = float(os.environ.get("FTOL", "1e-14"))

HERE = Path(os.environ.get("DKX_ROOT", ".")) / "examples" / "08_vmex_optimization"
inp = vj.VmecInput.from_file(HERE / "input.minimal_seed_nfp2")
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -0.05, 0.05
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5).change_resolution(mpol=5, ntor=5, ntheta=16, nzeta=14)
inp = replace(inp, ftol_array=np.array([FTOL]), niter_array=np.array([20000]))
NFP = int(inp.nfp)
NS = int(inp.ns_array[-1])
ROW = NS // 2
S = (ROW - 0.5) / (NS - 1)
print(f"ns={NS} row={ROW} s={S:.4f}")

n0, T0 = 1.0e19, 2.0e3
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1.0]), T0 * np.array([1, -1.0]), T0 * np.array([1, -1.0]))
nS = (1 - S**5); dnS = -5 * S**4
TS = 1 - S; dTS = -1.0
psi_a_hat = abs(float(inp.phiedge)) / (2 * np.pi)
deck = f"""&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 1
  inputRadialCoordinateForGradients = 1
  psiN_wish = {S:.12g}
  psiAHat = {psi_a_hat:.12g}
  aHat = 0.1
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = {n0/1e20*nS:.12g} {n0/1e20*nS:.12g}
  THats = {T0/1e3*TS:.12g} {T0/1e3*TS:.12g}
  dNHatdpsiNs = {n0/1e20*dnS:.12g} {n0/1e20*dnS:.12g}
  dTHatdpsiNs = {T0/1e3*dTS:.12g} {T0/1e3*dTS:.12g}
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 8.330d-3
  Er = 0
  collisionOperator = {COLL}
/
&resolutionParameters
  Ntheta = {NTH}
  Nzeta = {NZE}
  Nxi = {NXI}
  NL = 4
  Nx = {NX}
  solverTolerance = 1d-10
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""
template = kinetic_operator_from_namelist(parse_sfincs_input_text(deck))
grids = make_grids(n_theta=template.n_theta, n_zeta=template.n_zeta, n_xi=template.n_xi,
                   n_x=template.n_x, n_l=4, n_periods=NFP)
print("operator size", template.total_size)
METHOD = "auto" if COLL == 1 else "gmres"

cfg = imp.make_config(inp, adjoint_tol=1e-12, adjoint_maxiter=400)
params0 = imp.params_from_input(inp)
modes = opt._dof_modes(inp, 1)
NM = len(modes)
rows = np.asarray([n + int(inp.ntor) for (_, n) in modes]); cols = np.asarray([m for (m, _) in modes])
names = opt.boundary_dof_names(inp, 1)
dofs0 = jnp.asarray(opt.pack_boundary(inp, 1))
print(names)


def chain(dofs):
    params = replace(params0, rbc=params0.rbc.at[rows, cols].set(dofs[:NM]),
                     zbs=params0.zbs.at[rows, cols].set(dofs[NM:]))
    state = imp.solve_implicit(params, cfg)
    rt = imp.runtime_from_params(params, cfg)
    tabs = boozer_input_tables(state, rt, ROW)
    booz = booz_xform_jax(
        rmnc=tabs["rmnc"][None], zmns=tabs["zmns"][None], lmns=tabs["lmns"][None],
        bmnc=tabs["bmnc"][None], bsubumnc=tabs["bsubumnc"][None], bsubvmnc=tabs["bsubvmnc"][None],
        iota=tabs["iota"][None], xm=tabs["xm"], xn=tabs["xn"], xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
        nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False)
    ixm = np.asarray(booz["ixm_b"]); ixn = np.asarray(booz["ixn_b"])
    geom = FluxSurfaceGeometry.from_fourier(
        theta=grids.theta, zeta=grids.zeta, bmnc=booz["bmnc_b"][0], m=jnp.asarray(ixm),
        n=jnp.asarray(ixn // NFP), n_periods=NFP, iota=booz["iota_b"][0],
        g_hat=booz["bvco_b"][0], i_hat=booz["buco_b"][0])
    op = replace(template, b_hat=geom.b_hat, db_hat_dtheta=geom.db_hat_dtheta,
                 db_hat_dzeta=geom.db_hat_dzeta, d_hat=geom.d_hat,
                 b_hat_sup_theta=geom.b_hat_sup_theta, b_hat_sup_zeta=geom.b_hat_sup_zeta,
                 b_hat_sub_theta=geom.b_hat_sub_theta, b_hat_sub_zeta=geom.b_hat_sub_zeta,
                 fsab_hat2=geom.fsab_hat2(theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights))
    rhs = op.rhs()
    res = solve(op, rhs, method=METHOD, tol=1e-11, differentiable=True, emit=None)
    x = res.x.reshape(-1)
    mom = profile_moments_from_operator(op, x)
    j_dkx = jnp.asarray(mom["FSABjHat"]).reshape(()) * PARALLEL_CURRENT
    redl_geom = redl_geometry_from_state(state, rt, surfaces=np.array([S]))
    j_redl = j_dot_B_redl(profiles, redl_geom, 0)[0][0]
    resid = jnp.linalg.norm(op.apply(x) - rhs.reshape(-1)) / jnp.linalg.norm(rhs)
    return j_dkx, (j_redl, resid, booz["bmnc_b"][0], booz["iota_b"][0])


t = time.perf_counter(); j0, aux0 = chain(dofs0); t_p1 = time.perf_counter() - t
t = time.perf_counter(); j0, aux0 = chain(dofs0); t_p2 = time.perf_counter() - t
print(f"primal: DKX <j.B> = {float(j0):.6e} A T/m^2, Redl = {float(aux0[0]):.6e}, ratio {float(j0/aux0[0]):.4f}, "
      f"kinetic residual {float(aux0[1]):.2e}, iota {float(aux0[3]):.4f}; time {t_p1:.1f}s / {t_p2:.1f}s")
vg = jax.value_and_grad(chain, has_aux=True)
t = time.perf_counter(); (j0g, _), g = vg(dofs0); t_g1 = time.perf_counter() - t
t = time.perf_counter(); (j0g, _), g = vg(dofs0); t_g2 = time.perf_counter() - t
print(f"value_and_grad time {t_g1:.1f}s / {t_g2:.1f}s")
g = np.asarray(g)
check = [0, 2, NM + 1][:NDOF_CHECK] if NDOF_CHECK <= 3 else list(range(NDOF_CHECK))
for k in check:
    fds = []
    for h in FD_STEPS:
        e = jnp.zeros_like(dofs0).at[k].set(h)
        jp = float(chain(dofs0 + e)[0]); jm = float(chain(dofs0 - e)[0])
        fds.append((jp - jm) / (2 * h))
    rel = [abs(f - g[k]) / abs(g[k]) for f in fds]
    print(f"{names[k]:>10s}: AD {g[k]:+.8e}  FD " + "  ".join(f"h={h:g}:{f:+.8e} (rel {r:.2e})" for h, f, r in zip(FD_STEPS, fds, rel)))
sys.stdout.flush()
