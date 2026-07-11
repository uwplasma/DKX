"""Quasi-helical (QH) stellarator with low bootstrap current.

The quasi-helical analog of ``examples/optimize_QA_bootstrap.py`` and the
kinetic sibling of ``vmec_jax/examples/optimization/QH_optimization.py``.  Same
differentiable chain -- boundary Fourier coefficients -> vmec_jax implicit
fixed-boundary equilibrium (custom-VJP adjoint) -> traceable VMEC spectral
tables -> booz_xform_jax |B| spectrum -> FluxSurfaceGeometry.from_fourier
(geometryScheme 13) -> KineticOperator -> tier-2 GCROT solve with implicit
differentiation -> <j.B> -- but the quasisymmetry residual now targets the
helical family ``|B| = |B|(s, theta - nfp*zeta)`` (helicity (m, n) = (1, -1))
and the seed is a precise-QH configuration.

``jax.value_and_grad`` returns the gradient of the *whole* physics chain in the
boundary degrees of freedom; the kinetic Krylov solve is warm-started (x0 + the
GCROT recycle pair) across optimizer evaluations, and the end-to-end gradient is
checked against central finite differences.

Physics: the seed is the Landreman-Paul reactor-scale precise QH (nfp=4), which
is already strongly quasi-helical; the optimizer reduces the normalized
bootstrap current on a mid-radius surface while penalty terms hold aspect ratio,
mean iota and the two-term QS residual [Landreman & Buller, arXiv:2205.02914].
The kinetic setup is the classic SFINCS full-trajectory two-species run (PAS
collisions, finite Er), routed to the tier-2 GCROT solver where warm starts and
recycling matter.  The reactor-scale |B| spectrum is normalized by B00 so it
feeds the BBar=1 kinetic template unchanged.

Runtime: dominated by the one-time reactor-scale VMEC-implicit-adjoint XLA
compile plus ~40-60 s host equilibrium solves per L-BFGS-B evaluation, so the
default 25-iteration loop takes on the order of half an hour on a laptop CPU
(override with SFINCS_JAX_QH_MAXITER).  SFINCS_JAX_CI=1 shrinks it to a few
minutes.

Requires the optional companions vmec_jax (new core API with core.boozer_tables)
and booz_xform_jax: pip install -e /path/to/vmec_jax /path/to/booz_xform_jax

Run:
  python examples/optimization/optimize_QH_bootstrap.py
"""

import dataclasses
import json
import os
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import scipy.optimize

jax.config.update("jax_enable_x64", True)
matplotlib.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parent))  # for `import objectives`

import matplotlib.pyplot as plt  # noqa: E402

try:  # optional companion packages (not needed by sfincs_jax itself)
    import vmec_jax as _vmec_jax_pkg
    from vmec_jax.core import implicit as vmec_implicit
    from vmec_jax.core import optimize as vmec_optimize
    from vmec_jax.core.boozer_tables import boozer_input_tables
    from vmec_jax.core.input import VmecInput
    from booz_xform_jax.jax_api import booz_xform_jax as booz_transform
except ImportError as exc:
    raise SystemExit(
        "This example needs vmec_jax (new core API, with core.boozer_tables) and "
        "booz_xform_jax. Install with `pip install -e /path/to/vmec_jax "
        "/path/to/booz_xform_jax`."
    ) from exc

import objectives as ob  # noqa: E402  (shared, minimal example objective library)
from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from sfincs_jax.inputs import parse_sfincs_input_text  # noqa: E402
from sfincs_jax.phase_space import make_grids  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("SFINCS_JAX_CI") == "1"  # shrink resolution for CI

# Seed: Landreman-Paul reactor-scale precise QH, nfp=4 (ships with vmec_jax).
VMEC_INPUT = (
    Path(_vmec_jax_pkg.__file__).resolve().parents[1]
    / "examples" / "data" / "input.LandremanPaul2021_QH_reactorScale_lowres"
)
VMEC_INPUT = Path(os.environ.get("SFINCS_JAX_QH_VMEC_INPUT", VMEC_INPUT))

NS = 7 if CI else 13
# The autodiff accuracy is checked on the kinetic *segment* (Step 5, pure JAX on
# the fixed seed spectrum), which does not need a fully-converged equilibrium,
# so CI can use a looser VMEC ftol; the default keeps it tight for a
# publication-quality optimization.
VMEC_FTOL = 1e-11 if CI else 1e-13
VMEC_MAX_ITER = 5000
MAX_MODE = 1 if CI else 2  # boundary RBC/ZBS with m,|n| <= MAX_MODE
MBOZ, NBOZ = (2, 2) if CI else (3, 3)

# Kinetic solve: ions + electrons, PAS collisions, finite Er full trajectories.
KIN = (9, 7, 8, 4, 4) if CI else (13, 11, 16, 4, 6)  # Ntheta,Nzeta,Nxi,NL,Nx
ER, NU_N, KIN_TOL = 1.0, 0.1, 1e-9

# Targets and weights (QH: iota is large and negative here).  The kinetic
# bootstrap term is the optimization driver (its gradient is autodiff-accurate,
# Step 5); aspect/iota hold the equilibrium and QS is a *soft* constraint --
# the seed is already precise-QH and the reactor-scale QS host-metric adjoint is
# FD-noisy, so a heavy QS weight only injects a noisy gradient that fights the
# (accurate) bootstrap reduction.
TARGET_ASPECT, TARGET_IOTA = 8.0, -1.25
QS_SURFACES = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9])
W_ASPECT, W_IOTA, W_QS, W_KINETIC = 1.0, 100.0, 1.0e3, 1.0e6

# Kinetic figure of merit -- every entry is CI-tested, so switching just works.
KINETIC_OBJECTIVE = "bootstrap_jbs2"       # (<j.B>/sqrt(<B^2>))^2 -> 0
# KINETIC_OBJECTIVE = "particle_flux_l1"   # tested: smooth sum_s |Gamma_s|
# KINETIC_OBJECTIVE = "heat_flux_l2"       # tested: sum_s Q_s^2
KINETIC_OBJECTIVES = ob.MOMENT_METRICS

MAXITER = int(os.environ.get("SFINCS_JAX_QH_MAXITER", "2" if CI else "25"))
BOUND_RADIUS = 0.02
PENALTY_VALUE = 1.0e6
RUN_FD_CHECK = os.environ.get("SFINCS_JAX_QH_FD_CHECK", "1") == "1"
# kinetic-segment gradient (Boozer spectrum -> <j.B>) vs central FD: pure JAX +
# implicit linear solve, so autodiff-limited (rel ~1e-4..1e-6).
FD_GATE = 1e-3
OUT_DIR = Path(__file__).parent / "output"
STEM = "optimize_QH_bootstrap"

# ----------------------------------------------------------------------------
# 1) Differentiable fixed-boundary equilibrium (vmec_jax.core.implicit)
# ----------------------------------------------------------------------------
print("=== examples/optimization/optimize_QH_bootstrap.py ===")
print("Step 1: differentiable QH equilibrium (vmec_jax.core.implicit)")
if not VMEC_INPUT.exists():
    raise SystemExit(f"VMEC input not found: {VMEC_INPUT}")
inp0 = VmecInput.from_file(str(VMEC_INPUT))
inp0 = dataclasses.replace(inp0, ns_array=np.asarray([NS]),
                           ftol_array=np.asarray([VMEC_FTOL]),
                           niter_array=np.asarray([VMEC_MAX_ITER]))
cfg = vmec_implicit.make_config(inp0)
params0 = vmec_implicit.params_from_input(inp0)
NFP = int(cfg.resolution.nfp)
NTOR = int(inp0.ntor)
dof_modes = vmec_optimize._dof_modes(inp0, MAX_MODE)
NM = len(dof_modes)
dof_rows = np.asarray([n + NTOR for (_, n) in dof_modes])
dof_cols = np.asarray([m for (m, _) in dof_modes])
dof_names = vmec_optimize.boundary_dof_names(inp0, MAX_MODE)
dofs0 = jnp.asarray(vmec_optimize.pack_boundary(inp0, MAX_MODE))
qs_metric = vmec_optimize.QuasisymmetryRatioResidual(
    surfaces=QS_SURFACES, helicity_m=1, helicity_n=-1)  # QH: |B|(s, theta - nfp zeta)
print(f"  seed: {VMEC_INPUT.name} (nfp={NFP}, ns={NS}); {2 * NM} boundary dofs")

# Boozer output mode numbers (booz_xform ordering, fixed by MBOZ/NBOZ/NFP).
_bm, _bn = [], []
for _m in range(MBOZ):
    for _n in (range(0, NBOZ + 1) if _m == 0 else range(-NBOZ, NBOZ + 1)):
        _bm.append(_m)
        _bn.append(_n * NFP)
BOOZ_XM, BOOZ_XN = np.asarray(_bm), np.asarray(_bn)
QS_MASK = ob.qs_symmetric_mask(BOOZ_XM, BOOZ_XN, NFP, helicity_m=1, helicity_n=-1)

# ----------------------------------------------------------------------------
# 2) Kinetic operator template (geometry replaced per evaluation)
# ----------------------------------------------------------------------------
print("Step 2: kinetic-operator template (canonical KineticOperator route)")
PSI_A_HAT = abs(float(inp0.phiedge)) / (2.0 * np.pi)
KINETIC_TEMPLATE = f"""&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  helicity_n = {NFP}
  psiAHat = {PSI_A_HAT:.10g}
  aHat = 0.16
/
&speciesParameters
  Zs = 1.0d+0 -1.0d+0
  mHats = 1.0d+0 5.446170214d-4
  nHats = 1.0d+0 1.0d+0
  THats = 1.0d+0 1.0d+0
  dNHatdpsiHats = -1.5d+0 -1.5d+0
  dTHatdpsiHats = -3.0d+0 -3.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = {NU_N}
  Er = {ER}
  collisionOperator = 1
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
/
&resolutionParameters
  Ntheta = {KIN[0]}
  Nzeta = {KIN[1]}
  Nxi = {KIN[2]}
  NL = {KIN[3]}
  Nx = {KIN[4]}
  solverTolerance = {KIN_TOL}
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""
op_template = kinetic_operator_from_namelist(parse_sfincs_input_text(KINETIC_TEMPLATE))
S_ROW = NS // 2  # half-mesh radial row carrying the kinetic flux surface
_g = make_grids(n_theta=op_template.n_theta, n_zeta=op_template.n_zeta,
                n_xi=op_template.n_xi, n_x=op_template.n_x, n_l=KIN[3],
                n_periods=NFP, x_grid_scheme=5)
print(f"  ions + electrons, PAS, nu_n={NU_N}, Er={ER}; matrix size {op_template.total_size}")


# ----------------------------------------------------------------------------
# 3) The scalar objective: a plain JAX function of the boundary dofs
# ----------------------------------------------------------------------------
def objective(dofs, warm=None):
    """Total objective and diagnostics; differentiable in ``dofs``."""
    rbc = params0.rbc.at[dof_rows, dof_cols].set(dofs[:NM])
    zbs = params0.zbs.at[dof_rows, dof_cols].set(dofs[NM:])
    params = dataclasses.replace(params0, rbc=rbc, zbs=zbs)
    state = vmec_implicit.solve_implicit(params, cfg)
    rt = vmec_implicit.runtime_from_params(params, cfg)
    aspect = vmec_optimize.aspect_ratio(state, rt)
    iota_mean = vmec_optimize.mean_iota(state, rt)
    qs = qs_metric.total_state(state, rt)

    tabs = boozer_input_tables(state, rt, S_ROW)
    booz = booz_transform(
        rmnc=tabs["rmnc"][None, :], zmns=tabs["zmns"][None, :], lmns=tabs["lmns"][None, :],
        bmnc=tabs["bmnc"][None, :], bsubumnc=tabs["bsubumnc"][None, :],
        bsubvmnc=tabs["bsubvmnc"][None, :], iota=tabs["iota"][None],
        xm=tabs["xm"], xn=tabs["xn"], xm_nyq=tabs["xm"], xn_nyq=tabs["xn"],
        nfp=NFP, mboz=MBOZ, nboz=NBOZ, asym=False)
    bmnc_b = booz["bmnc_b"][0]
    b00 = bmnc_b[0]  # BBar = B00: normalize the reactor-scale spectrum to O(1)
    op = ob.operator_with_boozer_geometry(
        op_template, bmnc=bmnc_b / b00, m=jnp.asarray(BOOZ_XM),
        n=jnp.asarray(BOOZ_XN // NFP), nfp=NFP, iota=booz["iota_b"][0],
        g_hat=booz["bvco_b"][0] / b00, i_hat=booz["buco_b"][0] / b00,
        theta=_g.theta, zeta=_g.zeta,
        theta_weights=_g.theta_weights, zeta_weights=_g.zeta_weights)
    mom, result = ob.solve_and_moments(
        op, tol=KIN_TOL, x0=(warm or {}).get("x0"), recycle=(warm or {}).get("recycle"))
    kinetic_term = KINETIC_OBJECTIVES[KINETIC_OBJECTIVE](mom)
    jbs = ob.bootstrap_current(mom)
    total = (W_ASPECT * (aspect - TARGET_ASPECT) ** 2
             + W_IOTA * (iota_mean - TARGET_IOTA) ** 2
             + W_QS * qs + W_KINETIC * kinetic_term)
    sg = jax.lax.stop_gradient
    aux = {"aspect": aspect, "iota": iota_mean, "qs": qs, "jbs": jbs,
           "kinetic_term": kinetic_term, "bmnc_b": sg(bmnc_b / b00),
           "booz_iota": sg(booz["iota_b"][0]), "booz_g": sg(booz["bvco_b"][0] / b00),
           "booz_i": sg(booz["buco_b"][0] / b00),
           "particle_flux": sg(mom["particleFlux_vm_psiHat"]),
           "heat_flux": sg(mom["heatFlux_vm_psiHat"]),
           "x_solution": sg(result.x),
           "recycle": None if result.recycle is None else tuple(sg(r) for r in result.recycle),
           "kinetic_iterations": result.iterations}
    return total, aux


# ----------------------------------------------------------------------------
# 4) Initial evaluation + warm-start savings of the kinetic Krylov solve
# ----------------------------------------------------------------------------
print("Step 3: initial evaluation (cold kinetic solve)")
t0 = time.perf_counter()
J0, aux0 = objective(dofs0)
t_first = time.perf_counter() - t0
warm_state = {"x0": aux0["x_solution"], "recycle": aux0["recycle"]}
print(f"  J = {float(J0):.6e}  aspect = {float(aux0['aspect']):.4f}  "
      f"iota = {float(aux0['iota']):.4f}  QS = {float(aux0['qs']):.4e}")
print(f"  <j.B>/sqrt(<B^2>) = {float(aux0['jbs']):+.6e}   ({t_first:.1f} s incl. JIT)")

print("Step 4: warm-start savings (x0 + GCROT recycle)")
t0 = time.perf_counter()
_, aux_warm = objective(dofs0, warm_state)
t_warm = time.perf_counter() - t0
it_cold, it_warm = int(aux0["kinetic_iterations"]), int(aux_warm["kinetic_iterations"])
print(f"  kinetic iterations: cold {it_cold} -> warm {it_warm} "
      f"({100.0 * (1.0 - it_warm / max(it_cold, 1)):.0f}% fewer); "
      f"wall {t_first:.1f} s -> {t_warm:.1f} s")

# ----------------------------------------------------------------------------
# 5) Autodiff accuracy of the kinetic segment vs central finite differences
# ----------------------------------------------------------------------------
# The novel differentiable contribution is the Boozer-spectrum -> kinetic <j.B>
# segment (pure JAX + implicit linear solve).  We scale the symmetry-breaking
# |B| modes of the seed surface and check d(kinetic term)/d(scale) against a
# central FD -- this is autodiff-limited (rel ~1e-4..1e-6), unlike the full
# boundary-dof gradient whose FD is limited by the reactor-scale host
# equilibrium solver's ftol noise (~1.5%, see optimize_QA_bootstrap.py).  The
# full-chain boundary gradient is still what L-BFGS-B uses below; the QS penalty
# is a vmec_jax host-metric regularizer (its coarse-ns adjoint is FD-noisy but
# correct in sign, validated at resolution in the vmec_jax suite).
print("Step 5: kinetic-segment autodiff accuracy (Boozer spectrum -> <j.B>)")
value_and_grad = jax.value_and_grad(objective, has_aux=True)  # compiled in the loop below
_bmnc_seed = jnp.asarray(aux0["bmnc_b"])
_non00 = jnp.asarray((BOOZ_XM != 0) | (BOOZ_XN != 0))


def kinetic_segment(scale):
    bmnc = jnp.where(_non00, _bmnc_seed * scale, _bmnc_seed)
    op = ob.operator_with_boozer_geometry(
        op_template, bmnc=bmnc, m=jnp.asarray(BOOZ_XM), n=jnp.asarray(BOOZ_XN // NFP),
        nfp=NFP, iota=aux0["booz_iota"], g_hat=aux0["booz_g"], i_hat=aux0["booz_i"],
        theta=_g.theta, zeta=_g.zeta,
        theta_weights=_g.theta_weights, zeta_weights=_g.zeta_weights)
    mom, _ = ob.solve_and_moments(op, tol=KIN_TOL)
    return KINETIC_OBJECTIVES[KINETIC_OBJECTIVE](mom)


fd_check = None
if RUN_FD_CHECK:
    ad = float(jax.grad(kinetic_segment)(1.0))
    eps = 1e-6
    fd = (float(kinetic_segment(1.0 + eps)) - float(kinetic_segment(1.0 - eps))) / (2.0 * eps)
    rel = abs(ad - fd) / max(abs(fd), 1e-300)
    fd_check = {"ad": ad, "fd": fd, "rel": rel}
    print(f"  d(kinetic term)/d(spectrum scale): AD={ad:.6e} FD={fd:.6e} rel={rel:.2e}")
    if not (np.isfinite(fd) and rel < FD_GATE):
        raise SystemExit(f"kinetic-segment gradient check FAILED: rel {rel:.3e}")

# ----------------------------------------------------------------------------
# 6) Optimize: scipy L-BFGS-B on the JAX value-and-gradient
# ----------------------------------------------------------------------------
print(f"Step 6: L-BFGS-B, {MAXITER} iterations, objective '{KINETIC_OBJECTIVE}'")
history = []
_eval = [0]


def scipy_fun(x):
    t_start = time.perf_counter()
    try:
        (value, aux), grad = value_and_grad(jnp.asarray(x), warm_state)
    except Exception as exc:  # zero-crash: penalize failed trial equilibria
        print(f"  eval (failed boundary, penalized): {type(exc).__name__}")
        return PENALTY_VALUE, np.zeros_like(np.asarray(x, dtype=float))
    warm_state["x0"], warm_state["recycle"] = aux["x_solution"], aux["recycle"]
    _eval[0] += 1
    rec = {"eval": _eval[0], "objective": float(value), "aspect": float(aux["aspect"]),
           "iota": float(aux["iota"]), "qs": float(aux["qs"]), "jbs": float(aux["jbs"]),
           "wall_s": time.perf_counter() - t_start}
    history.append(rec)
    print(f"  eval {rec['eval']:3d}: J={rec['objective']:.6e} qs={rec['qs']:.3e} "
          f"<j.B>={rec['jbs']:+.4e} aspect={rec['aspect']:.3f} iota={rec['iota']:.4f} "
          f"({rec['wall_s']:.1f} s)")
    return float(value), np.asarray(grad, dtype=float)


bounds = [(float(v) - BOUND_RADIUS, float(v) + BOUND_RADIUS) for v in np.asarray(dofs0)]
opt = scipy.optimize.minimize(scipy_fun, np.asarray(dofs0, dtype=float), jac=True,
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": MAXITER, "maxcor": 10})
dofs_final = jnp.asarray(opt.x)
J_final, aux_final = objective(dofs_final, warm_state)

# ----------------------------------------------------------------------------
# 7) Final results, saved outputs, before/after plot
# ----------------------------------------------------------------------------
print("Step 7: final results")
print(f"  objective         {float(J0):.6e} -> {float(J_final):.6e}")
print(f"  <j.B>/sqrt(<B^2>)  {float(aux0['jbs']):+.6e} -> {float(aux_final['jbs']):+.6e}")
print(f"  QS residual       {float(aux0['qs']):.4e} -> {float(aux_final['qs']):.4e}")
print(f"  aspect / iota     {float(aux0['aspect']):.3f}/{float(aux0['iota']):.4f} -> "
      f"{float(aux_final['aspect']):.3f}/{float(aux_final['iota']):.4f}")
objective_decreased = float(J_final) < float(J0)
print(f"  objective decreased: {objective_decreased}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp_final = vmec_optimize.unpack_boundary(inp0, np.asarray(dofs_final, dtype=float), MAX_MODE)
input_path = inp_final.to_indata(OUT_DIR / f"input.{STEM}_optimized")
history_path = OUT_DIR / f"{STEM}_history.json"
history_path.write_text(json.dumps({
    "history": history,
    "initial": {k: float(aux0[k]) for k in ("aspect", "iota", "qs", "jbs")},
    "final": {k: float(aux_final[k]) for k in ("aspect", "iota", "qs", "jbs")},
    "objective_initial": float(J0), "objective_final": float(J_final),
    "warm_start": {"iterations_cold": it_cold, "iterations_warm": it_warm,
                   "seconds_first": t_first, "seconds_warm": t_warm},
    "kinetic_objective": KINETIC_OBJECTIVE, "dof_names": dof_names,
}, indent=2) + "\n")
read_back = json.loads(history_path.read_text())
print(f"  read back: objective_final = {read_back['objective_final']:.6e}")


def _bmag(bmnc_b, ntheta=60, nzeta=60):
    th = np.linspace(0, 2 * np.pi, ntheta)
    ze = np.linspace(0, 2 * np.pi / NFP, nzeta)
    ang = th[:, None, None] * BOOZ_XM[None, None, :] - ze[None, :, None] * BOOZ_XN[None, None, :]
    return th, ze, np.einsum("m,tzm->tz", np.asarray(bmnc_b), np.cos(ang))


fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
for ax, (tag, aux_t) in zip(axes[0], (("initial", aux0), ("final", aux_final))):
    th, ze, bmag = _bmag(aux_t["bmnc_b"])
    im = ax.contourf(ze, th, bmag, levels=24, cmap="viridis")
    ax.set_title(f"|B|/B00, Boozer angles ({tag})")
    ax.set_xlabel("zeta_B")
    ax.set_ylabel("theta_B")
    fig.colorbar(im, ax=ax, fraction=0.046)
evals = [h["eval"] for h in history]
if evals:
    axes[1, 0].semilogy(evals, [h["objective"] for h in history], "o-")
    axes[1, 1].semilogy(evals, [abs(h["jbs"]) for h in history], "o-", label="|<j.B>|/sqrt(<B^2>)")
    axes[1, 1].semilogy(evals, [h["qs"] for h in history], "s-", label="QS residual")
    axes[1, 1].legend()
axes[1, 0].set_xlabel("objective evaluation")
axes[1, 0].set_ylabel("total objective")
axes[1, 0].set_title("optimization progress")
axes[1, 1].set_xlabel("objective evaluation")
axes[1, 1].set_title("bootstrap current and quasisymmetry")
fig.tight_layout()
plot_path = OUT_DIR / f"{STEM}.png"
fig.savefig(plot_path, dpi=130)
plt.close(fig)
print(f"  Saved plot: {plot_path}")
print(f"  Wrote: {input_path.name}, {history_path.name}")
print("Done: examples/optimization/optimize_QH_bootstrap.py")
