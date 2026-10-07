"""Tutorial 11 -- Phi1: the electrostatic potential varying *within* a flux surface.

Usually the potential is taken constant on a surface, Phi = Phi0(psi).  Its
small in-surface variation Phi1(theta, zeta) is set by quasineutrality,

    sum_s Z_s e n_s1(theta, zeta) = 0,    with n_s1 including -Z_s e Phi1 n_s / T_s,

and it enters the kinetic equation through the E x B drift and the energy
change Z_s e v_d . grad(Phi1).  A highly charged impurity (C6+) feels Phi1
six times as strongly as the bulk ions, which is why Phi1 matters for impurity
accumulation in stellarators.  Including it makes the problem *nonlinear*:
quasineutrality adds a block of rows and DKX switches to a Newton-Krylov
solver -- watch the printed route change.

Both legs below use the same grid and differ only in ``includePhi1``, so the
flux change is physics, not discretization.  Physics: circular tokamak at
r/a = 0.3, hydrogen + carbon + electrons, full Fokker-Planck collisions
(inter-species momentum exchange sets the impurity flux).  SFINCS parameters
are passed as keywords to ``dkx.run``.

Run:              python examples/tutorials/11_phi1.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/11_phi1.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
LABELS = ("hydrogen Z=+1", "carbon Z=+6", "electron Z=-1")
CARBON_FRACTION = 0.0125  # n_C / n_H
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, B0OverBBar=1.0, epsilon_t=-0.07,
    epsilon_h=0.0, iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0, 6.0, -1.0], mHats=[1.0, 12.011, 5.446170214e-4],
    nHats=[1.0, CARBON_FRACTION, 1.0 + 6.0 * CARBON_FRACTION],  # quasineutral
    THats=[1.0, 1.0, 1.0], dNHatdrHats=[-0.5, -0.5, -0.5], dTHatdrHats=[-1.0, -1.0, -1.0],
    collisionOperator=0,  # 0 = linearized Fokker-Planck
    Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
    Ntheta=9 if SMOKE else 15, Nzeta=1, Nxi=8 if SMOKE else 16, NL=4, Nx=4 if SMOKE else 5,
)

# 3. Run
runs = {}
for phi1 in (False, True):
    print(f"\n--- includePhi1 = {phi1} ---", flush=True)
    runs[phi1] = dkx.run(**PARAMETERS, includePhi1=phi1, includePhi1InKineticEquation=phi1, emit=print)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
flux = {k: np.asarray(r.moments["particleFlux_vm_psiHat"], dtype=float).ravel() for k, r in runs.items()}
dn = np.asarray(runs[True].moments["densityPerturbation"], dtype=float)[:, :, 0]  # (species, theta)
theta = np.linspace(0.0, 2.0 * np.pi, dn.shape[1], endpoint=False)
fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
x = np.arange(len(LABELS))
left.bar(x - 0.18, np.abs(flux[False]), 0.36, label=r"$\Phi_1$ off", color="tab:grey")
left.bar(x + 0.18, np.abs(flux[True]), 0.36, label=r"$\Phi_1$ on", color="tab:orange")
left.set_yscale("log")
left.set_xticks(x, LABELS, fontsize=8)
left.set_ylabel(r"$|\Gamma_s|$ (SFINCS units)")
left.legend()
for k, label in enumerate(LABELS):
    right.plot(theta, dn[k], "o-", ms=3, label=label)
right.set(xlabel=r"$\theta$", ylabel=r"$\delta n_s / n_s$", title=r"in-surface density variation, $\Phi_1$ on")
right.legend(fontsize=8)
fig.savefig(OUT / "phi1.png", dpi=120)
np.savez(OUT / "phi1.npz", flux_off=flux[False], flux_on=flux[True], density_perturbation=dn, theta=theta)
print(f"\nsaved {OUT / 'phi1.npz'}\nsaved {OUT / 'phi1.png'}")
print("\n=== Summary ===")
for phi1, run in runs.items():
    print(f"  Phi1 {'on ' if phi1 else 'off'}: solver route {run.solve_result.method},"
          f" converged {bool(run.solve_result.converged)}")
for k, label in enumerate(LABELS):
    print(f"  {label:15s} Gamma off={flux[False][k]:+.4e}  on={flux[True][k]:+.4e}"
          f"  change={flux[True][k] / flux[False][k] - 1:+.1%}  peak |dn/n|={np.abs(dn[k]).max():.2e}")
