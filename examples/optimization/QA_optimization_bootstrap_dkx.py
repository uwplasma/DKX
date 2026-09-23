#!/usr/bin/env python
"""Optimize a finite-beta QA boundary against a self-consistent *kinetic* bootstrap current.

VMEX's examples/optimization/QA_optimization_bootstrap.py with one residual row
swapped: the Redl analytic bootstrap current is replaced by the one DKX computes
by solving the drift-kinetic equation on each trial equilibrium.  The chain
boundary -> VMEX -> booz_xform_jax -> DKX is traced end to end, so VMEX's exact
implicit Jacobian carries the DKX row like any other.  BOOTSTRAP_MODEL chooses
"dkx", "redl" (VMEX's example unchanged) or "both".

The kinetic profiles are the Landreman-Buller-Drevlak forms, ne = n0 (1 - s^5)
and Te = Ti = T0 (1 - s), at E_r = 0.  The Picard seed is Redl-consistent; the
optimization then makes the current consistent with DKX.  The default
collision operator is pitch-angle scattering: cheap and exactly
differentiable, but without momentum restoration, so its <j.B> sits above the
Fokker-Planck value (COLLISION_OPERATOR = 0 is the momentum-conserving choice).

Needs vmex and booz_xform_jax >= 0.4.  Runtime: 30 minutes on four laptop CPU
threads (4.6 GB peak), for objective 1.78 -> 0.0061 and DKX mismatch f_boot
1.2e-3 -> 1.2e-4; DKX_EXAMPLES_CI=1 is a 16-minute smoke pass.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,  # noqa: E402
                                 RedlBootstrapMismatch, self_consistent_bootstrap)

from dkx.bootstrap import KineticBootstrapMismatch  # noqa: E402

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Seed boundary: a circular cross-section of this minor radius, plus a
# rotating-ellipse perturbation that gives the optimizer a QA basin:
SEED_MINOR_RADIUS = 0.20
SEED_PERTURBATION = 0.10

# Pressure the profiles are calibrated to, and the weight of the beta residual,
# which is relative because the target is small:
TARGET_BETA = 0.025
BETA_WEIGHT = 1.0 / TARGET_BETA**2

# Flux surfaces the quasisymmetry and Redl residuals are evaluated on:
SURFACES = np.linspace(0.1, 0.9, 8)

# The bootstrap row: "dkx" (kinetic), "redl" (analytic) or "both".  The DKX
# row costs one drift-kinetic solve per kinetic surface per evaluation.
BOOTSTRAP_MODEL = "dkx"
KINETIC_SURFACES = [0.25, 0.5, 0.75]
KINETIC_RESOLUTION = dict(Ntheta=11, Nzeta=11, Nxi=16, NL=4, Nx=4)
COLLISION_OPERATOR = 1            # 1 pitch-angle scattering, 0 Fokker-Planck
BOOTSTRAP_WEIGHT = 1.0

# Mode ladder: highest boundary mode number varied in each stage, the residual
# evaluations each stage may spend, and the optimized I'(s) spline knots:
MAX_MODES = [1, 2]
MAX_NFEV = [10, 10]
N_CURRENT_SPLINE = [6, 6]

# Targets:
ASPECT_TARGET = 6.0
IOTA_FLOOR = 0.42                 # minimum |iota| over the profile

# Stability rows. VMEC's dimensional DMerc/DR values are O(1e2-1e3) for this
# seed, so the weight is small; Mercier coordinates are unreliable near the
# axis, so the weight is zero below STABILITY_MIN_S and rises toward the edge:
STABILITY_WEIGHT = 1.0e-6
EDGE_WEIGHT_FACTOR = 10.0
STABILITY_MIN_S = 0.2

# Picard loop that makes the seed current self-consistent (with Redl):
PICARD_ITERATIONS = 8
PICARD_TOLERANCE = 1e-3
REDL_N_LAMBDA = 32

# Step control. Boundary coefficients move in metres; the current dofs are
# dimensionless, so they carry their own optimizer scale:
PARAMETER_STEP = 0.02
CURRENT_PARAMETER_STEP = 0.05
MAX_PARAMETER_CHANGE = 10.0       # per-stage box guardrail, in scaled step units
VARY_MAJOR_RADIUS = False         # True optimizes RBC(0,0) instead of fixing it

# Equilibrium resolution: poloidal and toroidal mode numbers are max_mode + 2,
# but never below MINIMUM_MPOL:
MINIMUM_MPOL = 5

# Verification solve of the optimized boundary:
FINAL_NS = 101
FINAL_FTOL = 1e-14
FINAL_NITER = 8000

# Every output file name contains this:
OUTPUT_NAME = "QA_bootstrap_dkx_optimized"
CURRENT_FIGURE = "QA_bootstrap_dkx_current.png"

# DKX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("DKX_EXAMPLES_CI") == "1"
if ci_smoke:
    SURFACES, KINETIC_SURFACES = np.linspace(0.2, 0.8, 4), [0.5]
    KINETIC_RESOLUTION = dict(Ntheta=9, Nzeta=9, Nxi=12, NL=4, Nx=4)
    MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE = [1], [2], [4]
    PICARD_ITERATIONS, REDL_N_LAMBDA = 2, 12
    FINAL_NS, FINAL_FTOL = 31, 1e-10

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

# VmecInput is frozen, so copy its arrays before shaping the seed boundary.
inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor, 1] = zbs[inp.ntor, 1] = SEED_MINOR_RADIUS
rbc[inp.ntor + 1, 1], zbs[inp.ntor + 1, 1] = SEED_PERTURBATION, -SEED_PERTURBATION

# The Landreman-Buller-Drevlak profiles. Their product gives p = 2 e ne Te;
# AM below is (1 - s)(1 - s^5), matching that shape.
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21)
am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]
ac = np.zeros(21)
ac[0] = 1.0
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1,
              pcurr_type="power_series", ac=ac, curtor=0.0)

# One seed solve calibrates the profile amplitude to the requested beta.
seed = opt.solve_equilibrium(inp)
profile_scale = TARGET_BETA / float(seed.wout.betatotal)
n0 *= profile_scale ** (1 / 3)
T0 *= profile_scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * profile_scale)

# These polynomials provide ne(s), Te(s) and Ti(s) to both bootstrap models,
# so Redl and DKX describe one plasma.  The Picard loop leaves the pressure and
# boundary unchanged and updates the current profile (I'(s), CURTOR).
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1]),
                           T0 * np.array([1, -1]), T0 * np.array([1, -1]))
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=PICARD_ITERATIONS,
                                   tol=PICARD_TOLERANCE, degree=N_CURRENT_SPLINE[0] - 1,
                                   s_eval=SURFACES, verbose=not ci_smoke)
inp, equilibrium = picard.input, picard.equilibrium
seed_equilibrium = equilibrium

### Set up the objective ######################################################

def iota_floor(equilibrium_state, solver_context):
    """Hinge on the profile minimum of |iota|: a mean target can hide a near-zero surface."""
    return jnp.maximum(
        IOTA_FLOOR - opt.min_abs_iota(equilibrium_state, solver_context), 0.0)


def minimum_dmerc(equilibrium_state, solver_context):
    """Interior minimum of the Mercier criterion, reported not targeted."""
    return opt.d_merc_state(equilibrium_state, solver_context)[2:-1].min()


def maximum_dr(equilibrium_state, solver_context):
    """Interior maximum of the resistive-interchange criterion, reported not targeted."""
    return opt.glasser_d_r_state(equilibrium_state, solver_context)[2:-1].max()


# Zero weight below STABILITY_MIN_S, rising smoothly toward the edge.
stability_s = np.linspace(0.0, 1.0, int(inp.ns_array[-1]))[2:-1]
stability_weights = np.where(stability_s >= STABILITY_MIN_S,
    STABILITY_WEIGHT * (1.0 + (EDGE_WEIGHT_FACTOR - 1.0) * stability_s**4), 0.0)

# Each term is (function, target, weight).  Both bootstrap terms are the
# normalized mismatch between the equilibrium's <j.B> and the model's.
bootstrap = RedlBootstrapMismatch(profiles, helicity_n=0, surfaces=SURFACES,
                                  n_lambda=REDL_N_LAMBDA)
kinetic = KineticBootstrapMismatch(profiles, surfaces=KINETIC_SURFACES,
                                   resolution=KINETIC_RESOLUTION,
                                   collision_operator=COLLISION_OPERATOR)
bootstrap_terms = {"redl": [(bootstrap, 0.0, BOOTSTRAP_WEIGHT)],
                   "dkx": [(kinetic, 0.0, BOOTSTRAP_WEIGHT)]}
bootstrap_terms["both"] = bootstrap_terms["redl"] + bootstrap_terms["dkx"]
qs = opt.QuasisymmetryRatioResidual(SURFACES, helicity_m=1, helicity_n=0)
objective_function_terms = [
    (qs, 0.0, 1.0), *bootstrap_terms[BOOTSTRAP_MODEL],
    (opt.aspect_ratio, ASPECT_TARGET, 1.0),
    (iota_floor, 0.0, 10.0),
    (opt.volume_average_beta, TARGET_BETA, BETA_WEIGHT),
    (opt.mercier_stability_residual, 0.0, stability_weights),
    (opt.glasser_stability_residual, 0.0, stability_weights),
]

report = opt.EquilibriumReporter(
    ("QS", qs.total, ".4e"), ("f_boot Redl", bootstrap.total, ".4e"),
    ("f_boot DKX", kinetic.total, ".4e"),
    ("beta", opt.volume_average_beta, ".3%"), ("aspect", opt.aspect_ratio, ".3f"),
    ("min |iota|", opt.min_abs_iota, ".3f"), ("min DMerc", minimum_dmerc, ".2e"),
    ("max DR", maximum_dr, ".2e"))
monitor = opt.OptimizationMonitor()

### Run the optimization ######################################################

clock = time.perf_counter()
report("self-consistent seed", equilibrium)
costs = []
for max_mode, max_nfev, n_spline in zip(MAX_MODES, MAX_NFEV, N_CURRENT_SPLINE):
    print(f"\n===== QA bootstrap stage ({BOOTSTRAP_MODEL}), max_mode = {max_mode} =====")
    mpol = max(max_mode + 2, MINIMUM_MPOL)
    inp = inp.change_resolution(mpol=mpol, ntor=mpol, ntheta=2 * mpol + 6,
                                nzeta=2 * mpol + 4)
    inp = opt.resample_current_profile(inp, n_spline)
    problem = opt.VmecProblem.from_tuples(
        inp, objective_function_terms, max_mode=max_mode,
        current_dofs=n_spline - 1, vary_major_radius=VARY_MAJOR_RADIUS,
        use_ess=True, restart_from=equilibrium, progress=not ci_smoke)
    print(f"dof_names = {problem.dof_names}")
    monitor.problem = problem
    step = PARAMETER_STEP * problem.scales
    step[-n_spline:] = CURRENT_PARAMETER_STEP  # n-1 spline shapes plus CURTOR
    result = least_squares(
        problem.residual, problem.x0, jac=problem.residual_jac, x_scale=step,
        bounds=(problem.x0 - MAX_PARAMETER_CHANGE * step,
                problem.x0 + MAX_PARAMETER_CHANGE * step),
        max_nfev=max_nfev, ftol=1e-6, xtol=1e-10, verbose=2, callback=monitor)
    costs += [0.5 * float(np.sum(problem.residual(problem.x0) ** 2)), float(result.cost)]
    inp = problem.input_from_x(result.x)
    equilibrium = problem.equilibrium_from_x(result.x)
    report(f"mode {max_mode}", equilibrium)
optimize_seconds = time.perf_counter() - clock

### Check the result ##########################################################

# The optimizer's grid is not the certificate: re-solve the optimized boundary
# on a finer radial grid to a tighter tolerance and quote that.
final_input = replace(inp, ns_array=np.array([FINAL_NS]),
    ftol_array=np.array([FINAL_FTOL]), niter_array=np.array([FINAL_NITER]))
final_equilibrium = opt.solve_equilibrium(final_input, initial_state=equilibrium.solution,
    verbose=not ci_smoke, raise_on_max_iterations=True)

### Print, plot and save ######################################################

report("final", final_equilibrium)
print(f"objective 0.5*|r|^2 per stage (start, end): {np.round(costs, 5).tolist()}; "
      f"optimization wall time {optimize_seconds / 60:.1f} min")

input_path = final_input.to_indata(f"input.{OUTPUT_NAME}")
wout_path = vj.write_wout(f"wout_{OUTPUT_NAME}.nc", final_equilibrium.wout)
print(f"Wrote {input_path}\nWrote {wout_path}")
print(f"Wrote {monitor.save(f'{OUTPUT_NAME}_objectives.csv')}")

# <j.B> of the equilibrium, of Redl and of DKX, before and after, in MA T/m^2.
figure, (left, right) = plt.subplots(1, 2, figsize=(10.5, 3.8), constrained_layout=True)
dkx_label = "DKX (PAS)" if COLLISION_OPERATOR == 1 else "DKX"
for eq, style, label in ((seed_equilibrium, ":", "seed"), (final_equilibrium, "-", "final")):
    s, j_vmex, j_redl = (np.asarray(v) for v in bootstrap.current_profiles(eq))
    s_k, _, j_dkx = (np.asarray(v) for v in kinetic.current_profiles(eq.state, eq.runtime))
    j_vmex, j_redl, j_dkx = j_vmex / 1e6, j_redl / 1e6, j_dkx / 1e6
    left.plot(s, j_vmex, "k" + style, label=f"VMEX, {label}")
    left.plot(s, j_redl, "C0" + style, label=f"Redl, {label}")
    left.plot(s_k, j_dkx, "C3o" + style, label=f"{dkx_label}, {label}")
    print(f"<j.B> [MA T/m^2], {label}, at s = {np.round(s_k, 3).tolist()}: "
          f"DKX {np.round(j_dkx, 4).tolist()}, Redl {np.round(np.interp(s_k, s, j_redl), 4).tolist()}, "
          f"VMEX {np.round(np.interp(s_k, s, j_vmex), 4).tolist()}")
left.set(xlabel=r"$s$", ylabel=r"$\langle j\cdot B\rangle$ [MA T/m$^2$]",
         title=f"bootstrap current, optimized against {BOOTSTRAP_MODEL}")
left.legend(frameon=False, fontsize=8)
right.semilogy([costs[0]] + [record.cost for record in monitor.records], "o-")
right.set(xlabel="accepted iteration", ylabel=r"$\frac{1}{2}|r|^2$", title="objective")
figure.savefig(CURRENT_FIGURE, dpi=120)
print(f"Wrote {CURRENT_FIGURE}")
