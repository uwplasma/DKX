#!/usr/bin/env python
"""Bootstrap current of a VMEX equilibrium at its ambipolar E_r, and the derivative of that row.

``KineticBootstrapMismatch(ambipolar=True)`` is the traced DKX bootstrap row of
``QA_optimization_bootstrap_dkx.py`` with the radial electric field solved for
inside it: on every kinetic surface a damped Newton iteration, warm-started at the
previous root so it stays on the ion branch, finds the E_r where the radial
current vanishes, and <j.B> is read there.  The derivative through the root is
the implicit-function theorem, j_p - j_E J_p / J_E, from one adjoint solve per
surface, so VMEX's implicit Jacobian carries it.

This script solves a finite-beta rotating-ellipse equilibrium and prints, per
kinetic surface, the ambipolar E_r and <j.B> against the E_r = 0 value and
Redl's.  It then compares the Jacobian row of the ambipolar current with
central finite differences of the same row along boundary coefficients, each
difference point a full VMEX re-solve.  Finally it writes a figure and a CSV.

Needs vmex and booz_xform_jax >= 0.4.  DKX_EXAMPLES_CI=1 is the smoke pass.
"""

import os
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import vmex as vj  # noqa: E402
from vmex import optimize as opt  # noqa: E402
from vmex.core.bootstrap import (ELEMENTARY_CHARGE, KineticProfiles,  # noqa: E402
                                 j_dot_B_redl, redl_geometry_from_state, self_consistent_bootstrap)

from dkx.bootstrap import KineticBootstrapMismatch  # noqa: E402

# Number of field periods, and the seed deck the boundary is shaped from:
NFP = 2
INPUT_FILE = Path(__file__).resolve().parents[1] / "data" / f"input.minimal_seed_nfp{NFP}"

# Rotating-ellipse amplitude added to the circular seed, and the beta the
# Landreman-Buller-Drevlak profiles (ne = n0 (1 - s^5), Te = Ti = T0 (1 - s))
# are calibrated to:
SEED_PERTURBATION = 0.05
TARGET_BETA = 0.015

# Kinetic surfaces and grid (None keeps the ambipolar default), the first guess
# of the ion-root E_r [kV/m] and the Newton tolerance on E_r [kV/m]:
KINETIC_SURFACES = (0.3, 0.6)
KINETIC_RESOLUTION = None
ER_INITIAL = -2.0
ER_TOLERANCE = 0.05

# Derivative check: boundary coefficients varied, central step in metres, and
# the relative agreement required of the implicit Jacobian:
CHECK_DOFS = ("RBC(0,1)", "ZBS(-1,1)")
FD_STEP = 5e-4
FD_TOLERANCE = 0.05

# Equilibrium resolution and radial grid, and the Redl Picard loop that gives the
# equilibrium its self-consistent bootstrap current (its rotational transform):
MPOL = 5
NS = 25
PICARD_ITERATIONS = 8

# Every output file name starts with this:
OUTPUT_NAME = "ambipolar_bootstrap_dkx"

# DKX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
ci_smoke = os.environ.get("DKX_EXAMPLES_CI") == "1"
if ci_smoke:
    KINETIC_SURFACES = (0.3,)
    CHECK_DOFS, FD_TOLERANCE, NS, PICARD_ITERATIONS = CHECK_DOFS[:1], 0.2, 13, 2

###############################################################################
# End of input parameters.
###############################################################################

### Set up the equilibrium ####################################################

inp = vj.VmecInput.from_file(INPUT_FILE)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -SEED_PERTURBATION, SEED_PERTURBATION
n0 = 3.0e20 * (TARGET_BETA / 0.05) ** (1 / 3)
T0 = 15.0e3 * (TARGET_BETA / 0.05) ** (2 / 3)
am = np.zeros(21)
am[[0, 1, 5, 6]] = [1.0, -1.0, -1.0, 1.0]       # (1 - s)(1 - s^5)
ac = np.zeros(21)
ac[0] = 1.0
inp = replace(inp, rbc=rbc, zbs=zbs, delt=0.5, pmass_type="power_series", am=am,
              pres_scale=2 * ELEMENTARY_CHARGE * n0 * T0, ncurr=1, pcurr_type="power_series",
              ac=ac, curtor=0.0)
inp = inp.change_resolution(mpol=MPOL, ntor=MPOL, ntheta=2 * MPOL + 6, nzeta=2 * MPOL + 4)
inp = replace(inp, ns_array=np.array([NS]), ftol_array=np.array([1e-14]),
              niter_array=np.array([20000]))
equilibrium = opt.solve_equilibrium(inp)
scale = TARGET_BETA / float(equilibrium.wout.betatotal)
n0, T0 = n0 * scale ** (1 / 3), T0 * scale ** (2 / 3)
inp = replace(inp, pres_scale=inp.pres_scale * scale)
profiles = KineticProfiles(n0 * np.array([1, 0, 0, 0, 0, -1.0]), T0 * np.array([1, -1.0]),
                           T0 * np.array([1, -1.0]))
picard = self_consistent_bootstrap(inp, profiles, 0, n_iter=PICARD_ITERATIONS, tol=1e-3, degree=5,
                                   s_eval=np.linspace(0.1, 0.9, 8), verbose=False)
inp, equilibrium = picard.input, picard.equilibrium
minor_radius = float(equilibrium.wout.Aminor_p)
print(f"beta = {float(equilibrium.wout.betatotal):.3%}, a = {minor_radius:.3f} m, "
      f"n0 = {n0:.2e} m^-3, T0 = {T0:.0f} eV")

### Evaluate the kinetic row at E_r = 0 and at the ambipolar root #############

def kinetic_row(**options):
    return KineticBootstrapMismatch(
        profiles, KINETIC_SURFACES, mismatch=False, resolution=KINETIC_RESOLUTION,
        spectral_angles=True, minor_radius=minor_radius, **options)


zero_er = kinetic_row()
ambipolar = kinetic_row(ambipolar=True, er_initial=ER_INITIAL, er_tol=ER_TOLERANCE)
clock = time.perf_counter()
s, _, j_zero = zero_er.current_profiles(equilibrium.state, equilibrium.runtime)
print(f"E_r = 0 row: {time.perf_counter() - clock:.0f} s (cold)")
clock = time.perf_counter()
_, _, j_amb = ambipolar.current_profiles(equilibrium.state, equilibrium.runtime)
print(f"ambipolar row: {time.perf_counter() - clock:.0f} s (cold)")
clock = time.perf_counter()
ambipolar.current_profiles(equilibrium.state, equilibrium.runtime)
print(f"ambipolar row: {time.perf_counter() - clock:.1f} s (warm)")
geometry = redl_geometry_from_state(equilibrium.state, equilibrium.runtime, surfaces=np.asarray(s))
j_redl = j_dot_B_redl(profiles, geometry, 0)[0]
s, j_zero, j_amb, j_redl = (np.asarray(v) for v in (s, j_zero, j_amb, j_redl))
er = ambipolar.electric_field()
assert ambipolar.unconverged == 0, "the root search did not converge on every surface"
for row in zip(s, er, j_zero / 1e6, j_amb / 1e6, j_redl / 1e6):
    print(f"s = {row[0]:.3f}: ambipolar E_r = {row[1]:+.2f} kV/m; <j.B> [MA T/m^2] "
          f"E_r = 0: {row[2]:+.4f}, ambipolar: {row[3]:+.4f}, Redl: {row[4]:+.4f}")

### Check the derivative against finite differences ###########################

problem = opt.VmecProblem.from_tuples(inp, [(ambipolar, 0.0, 1.0)], max_mode=1)
names = list(problem.dof_names)
clock = time.perf_counter()
jacobian = np.asarray(problem.residual_jac(problem.x0))
print(f"Jacobian row: {time.perf_counter() - clock:.0f} s (cold)")
for name in CHECK_DOFS:
    k = names.index(name)
    step = np.zeros_like(problem.x0)
    step[k] = FD_STEP
    central = (problem.residual(problem.x0 + step) - problem.residual(problem.x0 - step)) / (2 * FD_STEP)
    error = np.linalg.norm(jacobian[:, k] - central) / np.linalg.norm(central)
    print(f"d<j.B>/d{name}: implicit {jacobian[:, k].tolist()}, "
          f"finite differences {central.tolist()} A T/m^2 per m; relative {error:.1%}")
    assert error < FD_TOLERANCE, f"{name}: the implicit Jacobian disagrees with differences"

### Print, plot and save ######################################################

figure, (left, right) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
left.plot(s, j_zero / 1e6, "o:", label=r"DKX, $E_r = 0$")
left.plot(s, j_amb / 1e6, "o-", label=r"DKX, ambipolar $E_r$")
left.plot(s, j_redl / 1e6, "k--", label="Redl")
left.set(xlabel=r"$s$", ylabel=r"$\langle j\cdot B\rangle$ [MA T/m$^2$]", title="bootstrap current")
left.legend(frameon=False, fontsize=8)
right.plot(s, er, "o-")
right.set(xlabel=r"$s$", ylabel=r"ambipolar $E_r$ [kV/m]", title="ion root")
figure_path = f"{OUTPUT_NAME}.png"
figure.savefig(figure_path, dpi=120)
csv_path = f"{OUTPUT_NAME}.csv"
np.savetxt(csv_path, np.column_stack([s, er, j_zero, j_amb, j_redl]), delimiter=",",
           header="s,Er_kV_per_m,jB_Er0,jB_ambipolar,jB_redl", comments="")
print(f"Wrote {figure_path}\nWrote {csv_path}")
