"""Tutorial 06 -- collision operators: pitch-angle scattering, Fokker-Planck and Sugama.

The collision operator C(f1) decides how fast the distribution relaxes.  Three
choices, selected by SFINCS's ``collisionOperator``:

* 1, pitch-angle scattering (PAS):  C = nu_D(v) L(f1), with L the Lorentz
  operator (1/2) d/dxi (1-xi^2) d/dxi.  It only rotates velocities, so it
  conserves particle number but *not momentum or energy*;
* 0, linearized Fokker-Planck (FP): test-particle plus field-particle terms
  between all species, conserving number, momentum and energy;
* 3, the improved Sugama model operator: a cheaper model of FP that keeps the
  same conservation laws.

Momentum conservation matters most where friction sets the answer: PAS
overestimates the bootstrap current <j.B>, which comes from friction between
trapped and passing particles and between species.  Hydrogen and electrons on
the analytic tokamak are solved once per operator, identical otherwise; the
summary prints each observable relative to FP.

Run:              python examples/tutorials/06_collision_operators.py
Runtime:          about 102 s on a loaded 36-core CPU node, compilation included.
"""

# 1. Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx
from dkx.units import PARALLEL_CURRENT

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
OPERATORS = {"Fokker-Planck": 0, "pitch-angle scattering": 1, "Sugama": 3}  # SFINCS collisionOperator
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,  # analytic tokamak at r/a = 0.3
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0, iota=0.4542, GHat=3.7481, IHat=0.0,
    psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[1.0, 1.0], THats=[1.0, 1.0],  # hydrogen, electrons
    dNHatdrHats=[-0.5, -0.5], dTHatdrHats=[-1.0, -1.0],  # drive: density and temperature gradients
    Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,  # SFINCS normalization and collisionality
    Ntheta=13, Nzeta=1, Nxi=16, NL=4, Nx=5,  # not a converged grid (tutorial 13)
)

# 3. Run -- one solve per operator, nothing else changes
runs = {}
for name, operator in OPERATORS.items():
    print(f"\n--- {name} (collisionOperator = {operator}) ---", flush=True)
    runs[name] = dkx.run(**PARAMETERS, collisionOperator=operator, emit=print)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
observables = {
    "ion particle flux": lambda r: float(np.asarray(r.moments["particleFlux_vm_psiHat"]).ravel()[0]),
    "ion heat flux": lambda r: float(np.asarray(r.moments["heatFlux_vm_psiHat"]).ravel()[0]),
    "<j.B> [A T m^-2]": lambda r: float(np.asarray(r.moments["FSABjHat"]).reshape(())) * PARALLEL_CURRENT,
}
values = {label: np.array([read(run) for run in runs.values()]) for label, read in observables.items()}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True)
for axis, (label, row) in zip(axes, values.items()):
    axis.bar(list(runs), row / row[0], color=["tab:blue", "tab:orange", "tab:green"])
    axis.set_title(f"{label}, relative to FP", fontsize=9)
    axis.tick_params(axis="x", labelrotation=20, labelsize=7)
fig.savefig(OUT / "operators.png", dpi=120)
np.savez(OUT / "operators.npz", operators=list(runs), **{k: v for k, v in values.items()})
print(f"\nsaved {OUT / 'operators.png'}")
print("\n=== Summary (relative to Fokker-Planck) ===")
print(f"  {'':24s}" + "".join(f"{name:>24s}" for name in runs))
for label, row in values.items():
    print(f"  {label:24s}" + "".join(f"{v / row[0]:24.3f}" for v in row))
