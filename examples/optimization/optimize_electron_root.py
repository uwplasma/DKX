"""Steer the ambipolar radial electric field toward the electron root.

The neoclassical ambipolar condition ``J_r(E_r) = sum_a Z_a Gamma_a(E_r) = 0``
fixes the radial electric field of a stellarator flux surface.  At low
collisionality a stellarator can sit on the *electron root* (``E_r > 0``, strong
outward electron transport balanced by a positive field), which improves
confinement and impurity screening [Turkin et al., Phys. Plasmas 18, 022505
(2011); Maassberg, Beidler & Turkin, Phys. Plasmas 16, 072504 (2009)].

This example optimizes the magnetic-field *shape* (a Boozer ``|B|`` spectrum) so
that the ambipolar ``E_r`` moves toward the electron-root side.  The whole thing
is differentiable: the ambipolar root comes from
:func:`dkx.er.ambipolar_er`, which wraps the root condition with
``solvax.implicit.root_solve`` so ``jax.grad`` flows through ``E_r`` via the
implicit function theorem ``dEr/dp = -(dJr/dEr)^{-1} dJr/dp`` (both Jacobians
from autodiff of the kinetic solve, not finite differences).  The decision
variables are three ``|B|`` amplitudes (a mirror term and two helical ripple
terms); the objective is a plain function of the resulting ``E_r``.

Warm starts: the ambipolar *root finding* (``find_ambipolar_er``) threads the
previous solve's ``x0`` and GCROT recycle pair across its ``E_r`` bracket
evaluations (a tier-2 Fokker-Planck benefit, shown below); the differentiable
``ambipolar_er`` additionally seeds its secant from the previous optimizer
iteration's root.

Run:
  python examples/optimization/optimize_electron_root.py
"""

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
from dkx import er as er_mod  # noqa: E402
from dkx.phase_space import make_grids  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
CI = os.environ.get("DKX_CI") == "1"

NFP = 4
# Model Boozer |B| spectrum (BBar units): B00, mirror(0,1), helical(1,0),
# helical(1,1).  n is WITHOUT the field-period factor.
BMNC_M = np.asarray([0, 0, 1, 1])
BMNC_N = np.asarray([0, 1, 0, 1])
BMNC0 = np.asarray([1.0, 0.05, 0.05, 0.10])
IOTA, GHAT, IHAT = 1.05, 1.2, 0.0
# The three shaping degrees of freedom: the mirror and the two helical ripples.
DOF_INDEX = np.asarray([1, 2, 3])
DOFS0 = BMNC0[DOF_INDEX].copy()
DOF_LABELS = ["mirror(0,1)", "helical(1,0)", "helical(1,1)"]
BOUND_LO = np.asarray([0.01, 0.01, 0.02])
BOUND_HI = np.asarray([0.15, 0.15, 0.30])

NU_N = 0.01 if CI else 0.003       # low collisionality (electron-root regime)
KIN = (5, 5, 8, 4, 3) if CI else (7, 7, 10, 4, 4)  # Ntheta,Nzeta,Nxi,NL,Nx
ER_BRACKET = (-4.0, 4.0)
ER_TARGET = 0.0                    # electron-root threshold; >0 targets a deeper root

# Objective on the scalar ambipolar E_r.  Every entry of ROOT_OBJECTIVES is
# CI-tested, so switching the commented line below just works.
ROOT_OBJECTIVE = "electron_root_offset"    # (E_r - ER_TARGET)^2 -> raise E_r
# ROOT_OBJECTIVE = "ion_root_deepen"       # tested: uncomment to deepen the ion root
# ROOT_OBJECTIVE = "maximize_er"           # tested: uncomment to push E_r up hard
ROOT_OBJECTIVES = {
    "electron_root_offset": lambda er: ob.root_offset_sq(er, ER_TARGET),
    "ion_root_deepen": lambda er: ob.root_offset_sq(er, -2.0),
    "maximize_er": lambda er: -er,
}

MAXITER = int(os.environ.get("DKX_ER_MAXITER", "3" if CI else "20"))
FD_EPS, FD_GATE = 1e-4, 1e-3
OUT_DIR = Path(__file__).parent / "output"
STEM = "optimize_electron_root"

# ----------------------------------------------------------------------------
# 1) Build the ambipolar problem (base operator + Er->dPhi factor + charges)
# ----------------------------------------------------------------------------
print("=== examples/optimization/optimize_electron_root.py ===")
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
  Zs = 1.0 -1.0
  mHats = 1.0 5.446170214d-4
  nHats = 1.0 1.0
  THats = 1.0 1.0
  dNHatdrHats = -1.0 -1.0
  dTHatdrHats = -1.0 -3.0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0
  nu_n = {NU_N}
  Er = 1.0
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
  solverTolerance = 1d-10
/
&otherNumericalParameters
  xGridScheme = 5
  Nxi_for_x_option = 0
/
&preconditionerOptions
/
"""
DECK_PATH = OUT_DIR / f"input.{STEM}"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DECK_PATH.write_text(DECK)
problem = er_mod.prepare(DECK_PATH, er_bracket=ER_BRACKET)
op_base = problem.operator
grids = make_grids(n_theta=op_base.n_theta, n_zeta=op_base.n_zeta, n_xi=op_base.n_xi,
                   n_x=op_base.n_x, n_l=KIN[3], n_periods=NFP, x_grid_scheme=5)
print(f"Step 1: 2-species FP ambipolar problem, nu_n={NU_N}, matrix {op_base.total_size}")
print(f"  shaping dofs: {DOF_LABELS}")


def operator_at(dofs):
    """Kinetic operator with the DOF-controlled |B| spectrum swapped in."""
    bmnc = jnp.asarray(BMNC0).at[jnp.asarray(DOF_INDEX)].set(jnp.asarray(dofs))
    return ob.operator_with_boozer_geometry(
        op_base, bmnc=bmnc, m=BMNC_M, n=BMNC_N, nfp=NFP, iota=IOTA, g_hat=GHAT,
        i_hat=IHAT, theta=grids.theta, zeta=grids.zeta,
        theta_weights=grids.theta_weights, zeta_weights=grids.zeta_weights)


def ambipolar_root(dofs, er0):
    """Differentiable ambipolar E_r for a shaping-dof vector, seeded at ``er0``."""
    return er_mod.ambipolar_er(operator_at(dofs), er0=er0,
                               dphi_per_er=problem.dphi_per_er, z_s=problem.z_s)


# ----------------------------------------------------------------------------
# 2) Locate the seed root (Fortran-parity Brent) + warm-start savings
# ----------------------------------------------------------------------------
print("Step 2: locate seed root and measure warm-start savings (tier-2 FP)")
seed_result = er_mod.find_ambipolar_er(
    er_mod.ErProblem(operator=operator_at(DOFS0), dphi_per_er=problem.dphi_per_er,
                     z_s=problem.z_s, er_initial=0.0, er_min=ER_BRACKET[0], er_max=ER_BRACKET[1]),
    all_roots=False, warm_start=True, emit=None)
er_seed = float(seed_result.er)
print(f"  seed root: E_r = {er_seed:+.5f}  ({seed_result.root_type} root), "
      f"J_r = {seed_result.radial_current:+.3e}")


def krylov_iterations(op0, warm):
    """Sum tier-2 Krylov iterations over a short E_r scan (cold vs warm)."""
    er_seq = np.linspace(er_seed - 0.3, er_seed + 0.3, 5)
    total, state = 0, None
    for e in er_seq:
        _, _, state = er_mod.radial_current(
            op0, float(e), dphi_per_er=problem.dphi_per_er, z_s=problem.z_s,
            x0=(state.x if (warm and state is not None) else None),
            recycle=(state.recycle if (warm and state is not None) else None), tol=1e-9)
        total += int(state.result.iterations or 0)
    return total


_op0 = operator_at(DOFS0)
it_cold = krylov_iterations(_op0, warm=False)
it_warm = krylov_iterations(_op0, warm=True)
print(f"  E_r-scan Krylov iterations: cold {it_cold} -> warm {it_warm} "
      f"({100.0 * (1.0 - it_warm / max(it_cold, 1)):.0f}% fewer; x0 + GCROT recycle pair)")

# ----------------------------------------------------------------------------
# 3) Differentiable root: value + gradient, checked vs finite differences
# ----------------------------------------------------------------------------
print("Step 3: differentiable ambipolar E_r via implicit function theorem")
value_and_grad = jax.value_and_grad(lambda d: ROOT_OBJECTIVES[ROOT_OBJECTIVE](
    ambipolar_root(d, er_seed)))
t0 = time.perf_counter()
J0, grad0 = value_and_grad(jnp.asarray(DOFS0))
er0 = float(ambipolar_root(jnp.asarray(DOFS0), er_seed))
t_grad = time.perf_counter() - t0
grad0 = np.asarray(grad0)
print(f"  E_r = {er0:+.5f}  objective = {float(J0):.6e}  |grad| = {np.linalg.norm(grad0):.4e} "
      f"({t_grad:.1f} s)")
k = int(np.argmax(np.abs(grad0)))


def er_scalar(dofs):
    return float(ambipolar_root(jnp.asarray(dofs), er_seed))


dp, dm = DOFS0.copy(), DOFS0.copy()
dp[k] += FD_EPS
dm[k] -= FD_EPS
fd = (ROOT_OBJECTIVES[ROOT_OBJECTIVE](er_scalar(dp))
      - ROOT_OBJECTIVES[ROOT_OBJECTIVE](er_scalar(dm))) / (2.0 * FD_EPS)
rel = abs(grad0[k] - fd) / max(abs(fd), 1e-300)
fd_check = {"dof": int(k), "name": DOF_LABELS[k], "ad": float(grad0[k]), "fd": float(fd), "rel": float(rel)}
print(f"  FD check on {DOF_LABELS[k]}: AD={grad0[k]:.6e} FD={fd:.6e} rel={rel:.2e}")
if not (np.isfinite(fd) and rel < FD_GATE):
    raise SystemExit(f"ambipolar-root gradient check FAILED: rel {rel:.3e}")

# ----------------------------------------------------------------------------
# 4) Optimize the shape to raise E_r toward the electron root
# ----------------------------------------------------------------------------
print(f"Step 4: L-BFGS-B, {MAXITER} iterations, objective '{ROOT_OBJECTIVE}'")
history = []
er_seed_box = [er_seed]  # thread the previous root as the next secant seed


def _obj_and_er(dofs, seed):
    er = ambipolar_root(dofs, seed)
    return ROOT_OBJECTIVES[ROOT_OBJECTIVE](er), er


def scipy_fun(x):
    t_start = time.perf_counter()
    # one differentiable solve returns objective, gradient, and E_r (aux)
    (value, er_now), grad = jax.value_and_grad(_obj_and_er, has_aux=True)(
        jnp.asarray(x), er_seed_box[0])
    er_seed_box[0] = float(er_now)
    rec = {"eval": len(history) + 1, "objective": float(value), "er": float(er_now),
           "wall_s": time.perf_counter() - t_start}
    history.append(rec)
    print(f"  eval {rec['eval']:3d}: J={rec['objective']:.6e} E_r={rec['er']:+.5f} "
          f"({rec['wall_s']:.1f} s)")
    return float(value), np.asarray(grad, dtype=float)


bounds = list(zip([float(v) for v in BOUND_LO], [float(v) for v in BOUND_HI]))
opt = scipy.optimize.minimize(scipy_fun, np.asarray(DOFS0, dtype=float), jac=True,
                              method="L-BFGS-B", bounds=bounds,
                              options={"maxiter": MAXITER, "maxcor": 10})
dofs_final = np.asarray(opt.x)
er_final = er_scalar(dofs_final)
J_final = float(ROOT_OBJECTIVES[ROOT_OBJECTIVE](er_final))

# ----------------------------------------------------------------------------
# 5) Results + before/after plot
# ----------------------------------------------------------------------------
root_type_final = er_mod._classify(er_final, 1.0) if er_final != 0 else "threshold"
print("Step 5: results")
print(f"  E_r         {er0:+.5f} -> {er_final:+.5f}  (target {ER_TARGET:+.2f}, {root_type_final} side)")
print(f"  objective   {float(J0):.6e} -> {J_final:.6e}")
for lab, a, b in zip(DOF_LABELS, DOFS0, dofs_final):
    print(f"    {lab:>14s}: {a:.4f} -> {b:.4f}")
objective_decreased = J_final < float(J0)
print(f"  objective decreased: {objective_decreased}")

history_path = OUT_DIR / f"{STEM}_history.json"
history_path.write_text(json.dumps({
    "history": history, "er_initial": er0, "er_final": er_final,
    "objective_initial": float(J0), "objective_final": J_final,
    "er_target": ER_TARGET, "root_objective": ROOT_OBJECTIVE,
    "dofs_initial": [float(v) for v in DOFS0], "dofs_final": [float(v) for v in dofs_final],
    "warm_start": {"krylov_iterations_cold": it_cold, "krylov_iterations_warm": it_warm},
    "fd_check": fd_check,
}, indent=2) + "\n")
print(f"  read back: er_final = {json.loads(history_path.read_text())['er_final']:+.5f}")

# Er-scan of the radial current before/after + objective convergence.
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
er_grid = np.linspace(ER_BRACKET[0], ER_BRACKET[1], 21)
for dofs, tag in ((DOFS0, "initial"), (dofs_final, "final")):
    op = operator_at(jnp.asarray(dofs))
    jr = [float(er_mod.radial_current(op, float(e), dphi_per_er=problem.dphi_per_er,
                                      z_s=problem.z_s)[0]) for e in er_grid]
    axes[0].plot(er_grid, jr, "o-", ms=3, label=tag)
axes[0].axhline(0.0, color="k", lw=0.6)
axes[0].axvline(ER_TARGET, color="g", ls="--", lw=0.8, label=f"target {ER_TARGET}")
axes[0].set_xlabel("E_r")
axes[0].set_ylabel("radial current J_r")
axes[0].set_title("ambipolar radial current")
axes[0].legend()
if history:
    axes[1].semilogy([h["eval"] for h in history], [h["objective"] for h in history], "o-")
axes[1].set_xlabel("objective evaluation")
axes[1].set_ylabel("(E_r - target)^2")
axes[1].set_title("optimization progress")
fig.tight_layout()
plot_path = OUT_DIR / f"{STEM}.png"
fig.savefig(plot_path, dpi=130)
plt.close(fig)
print(f"  Saved plot: {plot_path}")
print("Done: examples/optimization/optimize_electron_root.py")
