"""Shape the field for impurity temperature screening (multi-species FP).

A trace highly-charged impurity (here fully-ionized carbon, C6+) tends to
accumulate in a stellarator core, pulled inward by the main-ion density
gradient.  *Temperature screening* is the competing neoclassical effect in
which the main-ion temperature gradient drives the impurity radially *outward*
[Helander & Sigmar, Collisional Transport in Magnetized Plasmas (2002); Newton,
Helander et al., J. Plasma Phys. 83 (2017); Mollen et al., PPCF 60, 084001
(2018)].  Whether accumulation or screening wins depends on the magnetic
geometry, so it is an optimization target.

This example optimizes a Boozer ``|B|`` spectrum to push the C6+ radial particle
flux ``Gamma_z`` outward (``Gamma_z > 0`` == screening).  Three species (D+,
electrons, trace C6+) with the linearized Fokker-Planck collision operator make
the impurity flux depend on inter-species momentum exchange (the temperature
screening physics).  The flux comes from the canonical
``KineticOperator -> solve(differentiable=True) -> profile_moments_from_operator``
chain, so ``jax.value_and_grad`` returns the gradient of ``Gamma_z`` in the
shaping coefficients.  The kinetic solve is warm-started (x0 + GCROT recycle
pair) across optimizer evaluations, and the field is held at a fixed
representative ambipolar ``E_r``.

We also report the *temperature-screening coefficient* ``d Gamma_z /
d(dln T_i/dr)`` -- the sensitivity of the impurity flux to the main-ion
temperature gradient -- straight from autodiff.

Run:
  python examples/optimization/optimize_impurity_screening.py
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

import objectives as ob  # noqa: E402  (shared, minimal example objective library)
from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.inputs import parse_sfincs_input_text  # noqa: E402
from dkx.phase_space import make_grids  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("DKX_CI") == "1"

NFP = 4
BMNC_M = np.asarray([0, 0, 1, 1])
BMNC_N = np.asarray([0, 1, 0, 1])
BMNC0 = np.asarray([1.0, 0.05, 0.05, 0.10])  # B00, mirror(0,1), helical(1,0), helical(1,1)
IOTA, GHAT, IHAT = 1.05, 1.2, 0.0
DOF_INDEX = np.asarray([1, 2, 3])
DOFS0 = BMNC0[DOF_INDEX].copy()
DOF_LABELS = ["mirror(0,1)", "helical(1,0)", "helical(1,1)"]
BOUND_LO = np.asarray([0.01, 0.01, 0.02])
BOUND_HI = np.asarray([0.15, 0.15, 0.30])

IMP = 2                             # impurity species index (D+, e-, C6+)
ER = -1.0                           # fixed representative ambipolar (ion-root) field
NU_N = 0.03 if CI else 0.01
KIN = (5, 5, 8, 4, 3) if CI else (7, 7, 10, 4, 4)  # Ntheta,Nzeta,Nxi,NL,Nx
KIN_TOL = 1e-9
# The normalized fluxes are ~1e-8, far below L-BFGS-B's default gradient
# tolerance, so scale the objective to O(1) for the optimizer (scale-invariant
# for the AD-vs-FD check; the raw Gamma_z is reported unscaled).
OBJECTIVE_SCALE = 1.0e8

# Objective on the solved moment table.  Every entry of FLUX_OBJECTIVES is
# CI-tested, so switching the commented line below just works.
FLUX_OBJECTIVE = "impurity_screening"      # maximize outward Gamma_z (screening)
# FLUX_OBJECTIVE = "impurity_flux_zero"    # tested: uncomment to null the impurity flux
# FLUX_OBJECTIVE = "impurity_heat_flux"    # tested: uncomment to minimize impurity heat flux
FLUX_OBJECTIVES = {
    "impurity_screening": lambda mom: -ob.impurity_screening_metric(mom, IMP),
    "impurity_flux_zero": lambda mom: ob.species_particle_flux(mom, IMP) ** 2,
    "impurity_heat_flux": lambda mom: ob.species_heat_flux(mom, IMP) ** 2,
}

MAXITER = int(os.environ.get("DKX_IMP_MAXITER", "3" if CI else "20"))
FD_EPS, FD_GATE = 1e-4, 1e-3
OUT_DIR = Path(__file__).parent / "output"
STEM = "optimize_impurity_screening"

# ----------------------------------------------------------------------------
# 1) Kinetic operator template (3 species, Fokker-Planck)
# ----------------------------------------------------------------------------
print("=== examples/optimization/optimize_impurity_screening.py ===")
DECK = f"""&general
  RHSMode = 1
/
&geometryParameters
  geometryScheme = 1
  inputRadialCoordinate = 3
  rN_wish = 0.5
  psiAHat = 0.045
  aHat = 0.16
  B0OverBBar = 1.0
  GHat = {GHAT}
  IHat = {IHAT}
  iota = {IOTA}
  epsilon_t = 0.06
  epsilon_h = 0.06
  helicity_l = 1
  helicity_n = {NFP}
/
&speciesParameters
  Zs = 1.0 -1.0 6.0
  mHats = 2.0 5.446170214d-4 12.0
  nHats = 1.0 1.06 0.01
  THats = 1.0 1.0 1.0
  dNHatdrHats = -1.0 -1.0 -0.5
  dTHatdrHats = -2.0 -2.0 -2.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = {NU_N}
  Er = {ER}
  collisionOperator = 0
  includeXDotTerm = .true.
  includeElectricFieldTermInXiDot = .true.
  useDKESExBDrift = .false.
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
op_template = kinetic_operator_from_namelist(parse_sfincs_input_text(DECK))
grids = make_grids(n_theta=op_template.n_theta, n_zeta=op_template.n_zeta,
                   n_xi=op_template.n_xi, n_x=op_template.n_x, n_l=KIN[3],
                   n_periods=NFP, x_grid_scheme=5)
print(f"Step 1: 3-species (D+, e-, C6+) Fokker-Planck, nu_n={NU_N}, Er={ER}, "
      f"matrix {op_template.total_size}")
print(f"  shaping dofs: {DOF_LABELS}")


def operator_at(dofs):
    bmnc = jnp.asarray(BMNC0).at[jnp.asarray(DOF_INDEX)].set(jnp.asarray(dofs))
    return ob.operator_with_boozer_geometry(
        op_template, bmnc=bmnc, m=BMNC_M, n=BMNC_N, nfp=NFP, iota=IOTA, g_hat=GHAT,
        i_hat=IHAT, theta=grids.theta, zeta=grids.zeta,
        theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights)


def objective(dofs, warm=None):
    """Screening objective (minimize -Gamma_z) + diagnostics; differentiable."""
    op = operator_at(dofs)
    mom, result = ob.solve_and_moments(
        op, tol=KIN_TOL, x0=(warm or {}).get("x0"), recycle=(warm or {}).get("recycle"))
    value = OBJECTIVE_SCALE * FLUX_OBJECTIVES[FLUX_OBJECTIVE](mom)
    sg = jax.lax.stop_gradient
    aux = {"gamma_z": ob.species_particle_flux(mom, IMP),
           "particle_flux": sg(mom["particleFlux_vm_psiHat"]),
           "heat_flux": sg(mom["heatFlux_vm_psiHat"]),
           "x_solution": sg(result.x),
           "recycle": None if result.recycle is None else tuple(sg(r) for r in result.recycle),
           "kinetic_iterations": result.iterations}
    return value, aux


# ----------------------------------------------------------------------------
# 2) Initial evaluation + warm-start savings (tier-2 FP GCROT recycling)
# ----------------------------------------------------------------------------
print("Step 2: initial evaluation (cold kinetic solve)")
value_and_grad = jax.value_and_grad(objective, has_aux=True)
t0 = time.perf_counter()
J0, aux0 = objective(jnp.asarray(DOFS0))
t_first = time.perf_counter() - t0
warm_state = {"x0": aux0["x_solution"], "recycle": aux0["recycle"]}
gz0 = float(aux0["gamma_z"])
print(f"  Gamma_z (C6+) = {gz0:+.6e}  ({'outward/screening' if gz0 > 0 else 'inward/accumulation'})")
print(f"  objective J = {float(J0):+.6e}   ({t_first:.1f} s incl. JIT)")

print("Step 3: warm-start savings (x0 + GCROT recycle)")
t0 = time.perf_counter()
_, aux_warm = objective(jnp.asarray(DOFS0), warm_state)
t_warm = time.perf_counter() - t0
it_cold, it_warm = int(aux0["kinetic_iterations"]), int(aux_warm["kinetic_iterations"])
print(f"  kinetic iterations: cold {it_cold} -> warm {it_warm} "
      f"({100.0 * (1.0 - it_warm / max(it_cold, 1)):.0f}% fewer); "
      f"wall {t_first:.1f} s -> {t_warm:.1f} s")

# ----------------------------------------------------------------------------
# 3) Gradient vs finite differences + the temperature-screening coefficient
# ----------------------------------------------------------------------------
print("Step 4: gradient (autodiff) vs central finite differences")
(J_check, _), grad0 = value_and_grad(jnp.asarray(DOFS0), warm_state)
grad0 = np.asarray(grad0)
k = int(np.argmax(np.abs(grad0)))
dp, dm = DOFS0.copy(), DOFS0.copy()
dp[k] += FD_EPS
dm[k] -= FD_EPS
vp, _ = objective(jnp.asarray(dp), warm_state)
vm, _ = objective(jnp.asarray(dm), warm_state)
fd = (float(vp) - float(vm)) / (2.0 * FD_EPS)
rel = abs(grad0[k] - fd) / max(abs(fd), 1e-300)
fd_check = {"dof": int(k), "name": DOF_LABELS[k], "ad": float(grad0[k]), "fd": float(fd), "rel": float(rel)}
print(f"  |grad| = {np.linalg.norm(grad0):.4e}; FD check on {DOF_LABELS[k]}: "
      f"AD={grad0[k]:.6e} FD={fd:.6e} rel={rel:.2e}")
if not (np.isfinite(fd) and rel < FD_GATE):
    raise SystemExit(f"impurity-flux gradient check FAILED: rel {rel:.3e}")

# Temperature-screening coefficient d(Gamma_z)/d(scale on ion dT/dr) via autodiff.
base_dt = op_template.dt_hat_dpsi_hat


def gamma_z_vs_ion_dt(scale):
    dt = base_dt.at[0].multiply(scale)  # scale the main-ion temperature gradient
    op = dataclasses.replace(operator_at(jnp.asarray(DOFS0)), dt_hat_dpsi_hat=dt)
    mom, _ = ob.solve_and_moments(op, tol=KIN_TOL)
    return ob.species_particle_flux(mom, IMP)


screening_coeff = float(jax.grad(gamma_z_vs_ion_dt)(1.0))
print(f"  temperature-screening coefficient d(Gamma_z)/d(scale*dTi/dr) = {screening_coeff:+.4e}")
print("    (>0 means a steeper ion temperature gradient screens harder here)")

# ----------------------------------------------------------------------------
# 4) Optimize the shape to maximize outward impurity flux (screening)
# ----------------------------------------------------------------------------
print(f"Step 5: L-BFGS-B, {MAXITER} iterations, objective '{FLUX_OBJECTIVE}'")
history = []


def scipy_fun(x):
    t_start = time.perf_counter()
    try:
        (value, aux), grad = value_and_grad(jnp.asarray(x), warm_state)
    except Exception as exc:
        print(f"  eval (failed, penalized): {type(exc).__name__}")
        return 1.0e6, np.zeros_like(np.asarray(x, dtype=float))
    warm_state["x0"], warm_state["recycle"] = aux["x_solution"], aux["recycle"]
    rec = {"eval": len(history) + 1, "objective": float(value),
           "gamma_z": float(aux["gamma_z"]), "wall_s": time.perf_counter() - t_start}
    history.append(rec)
    print(f"  eval {rec['eval']:3d}: J={rec['objective']:+.6e} Gamma_z={rec['gamma_z']:+.6e} "
          f"({rec['wall_s']:.1f} s)")
    return float(value), np.asarray(grad, dtype=float)


bounds = list(zip([float(v) for v in BOUND_LO], [float(v) for v in BOUND_HI]))
opt = scipy.optimize.minimize(scipy_fun, np.asarray(DOFS0, dtype=float), jac=True,
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": MAXITER, "maxcor": 10})
dofs_final = np.asarray(opt.x)
J_final, aux_final = objective(jnp.asarray(dofs_final), warm_state)
gz_final = float(aux_final["gamma_z"])

# ----------------------------------------------------------------------------
# 5) Results + before/after plot
# ----------------------------------------------------------------------------
print("Step 6: results")
print(f"  Gamma_z (C6+)  {gz0:+.6e} -> {gz_final:+.6e}  "
      f"({'more outward (screening up)' if gz_final > gz0 else 'less outward'})")
print(f"  objective      {float(J0):+.6e} -> {float(J_final):+.6e}")
for lab, a, b in zip(DOF_LABELS, DOFS0, dofs_final):
    print(f"    {lab:>14s}: {a:.4f} -> {b:.4f}")
objective_decreased = float(J_final) < float(J0)
print(f"  objective decreased: {objective_decreased}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
history_path = OUT_DIR / f"{STEM}_history.json"
history_path.write_text(json.dumps({
    "history": history, "gamma_z_initial": gz0, "gamma_z_final": gz_final,
    "objective_initial": float(J0), "objective_final": float(J_final),
    "flux_objective": FLUX_OBJECTIVE, "screening_coefficient": screening_coeff,
    "dofs_initial": [float(v) for v in DOFS0], "dofs_final": [float(v) for v in dofs_final],
    "warm_start": {"iterations_cold": it_cold, "iterations_warm": it_warm,
                   "seconds_first": t_first, "seconds_warm": t_warm},
    "fd_check": fd_check,
}, indent=2) + "\n")
print(f"  read back: gamma_z_final = {json.loads(history_path.read_text())['gamma_z_final']:+.6e}")

fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
species_labels = ["D+", "e-", "C6+"]
x = np.arange(len(species_labels))
axes[0].bar(x - 0.2, np.asarray(aux0["particle_flux"]), 0.4, label="initial")
axes[0].bar(x + 0.2, np.asarray(aux_final["particle_flux"]), 0.4, label="final")
axes[0].axhline(0.0, color="k", lw=0.6)
axes[0].set_xticks(x)
axes[0].set_xticklabels(species_labels)
axes[0].set_ylabel("radial particle flux (+ = outward)")
axes[0].set_title("per-species flux (impurity = screening)")
axes[0].legend()
if history:
    axes[1].plot([h["eval"] for h in history], [h["gamma_z"] for h in history], "o-")
axes[1].axhline(0.0, color="k", lw=0.6)
axes[1].set_xlabel("objective evaluation")
axes[1].set_ylabel("Gamma_z (C6+)")
axes[1].set_title("impurity flux (up = more screening)")
fig.tight_layout()
plot_path = OUT_DIR / f"{STEM}.png"
fig.savefig(plot_path, dpi=130)
plt.close(fig)
print(f"  Saved plot: {plot_path}")
print("Done: examples/optimization/optimize_impurity_screening.py")
