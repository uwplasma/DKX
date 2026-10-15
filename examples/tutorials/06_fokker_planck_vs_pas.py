"""Tutorial 06 -- collision operators: full Fokker-Planck versus pitch-angle scattering.

The collision operator C(f1) decides how fast the distribution relaxes.  Two
choices:

* pitch-angle scattering (PAS):  C = nu_D(v) L(f1), with L the Lorentz operator
  (1/2) d/dxi (1-xi^2) d/dxi.  It only rotates velocities, so it conserves
  particle number but *not momentum or energy*.
* linearized Fokker-Planck (FP): self- and cross-species test-particle plus
  field-particle terms, conserving number, momentum and energy.

Momentum conservation matters most where friction sets the answer: in a
tokamak, like-particle collisions that conserve momentum drive no net ion
particle flux, so PAS overestimates Gamma_i; the bootstrap current <j.B>
comes from friction between trapped and passing particles and between
species.  Deuterium and electrons on the analytic tokamak are solved twice,
identical except for ``physics.collisions``; the summary prints FP/PAS
ratios so you can see which observables move and by how much.

Run:              python examples/tutorials/06_fokker_planck_vs_pas.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/06_fokker_planck_vs_pas.py
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
OPERATORS = ("pitch_angle_scattering", "linearized_fokker_planck")
CASE = {
    "name": "collision_operators",
    "run": {"workflow": "profile", "progress": True},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16, 0.25]},
    "species": [
        {"name": "deuterium", "charge": 1, "mass_amu": 2.014,
         "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]},
        {"name": "electron", "charge": -1, "mass_amu": 5.48579909e-4,
         "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]},
    ],
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "solver": {"method": "auto"},  # "auto" picks the linear-solver route from the operator
    "resolution": {"theta": 13, "zeta": 1, "pitch": 16, "speed": 6},
}
if SMOKE:
    CASE["resolution"] = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}

# 3. Run -- one solve per operator, nothing else changes
results = {}
for operator in OPERATORS:
    print(f"\n--- collisions = {operator} ---", flush=True)
    results[operator] = dkx.run(dkx.Case.from_mapping({**CASE, "physics": {"collisions": operator}}))

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
psi = np.asarray(results[OPERATORS[0]].arrays["surface"], dtype=float)
panels = (("particle_flux_m2_s", 0, "ion particle flux [m$^{-2}$ s$^{-1}$]"),
          ("heat_flux_W_m2", 0, "ion heat flux [W m$^{-2}$]"),
          ("parallel_current_A_T_m2", None, r"$\langle j\cdot B\rangle$ [A T m$^{-2}$]"))
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
for axis, (name, species, label) in zip(axes, panels):
    for operator, marker in zip(OPERATORS, "os"):
        values = np.asarray(results[operator].arrays[name])
        axis.plot(psi, values if species is None else values[:, species], marker=marker, label=operator)
    axis.set_xlabel(r"$\psi_N$")
    axis.set_title(label, fontsize=9)
axes[0].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "operators.png", dpi=120)
for operator, result in results.items():
    result.save(OUT / f"{operator}.nc")
print(f"\nsaved {OUT / 'operators.png'} and one NetCDF result per operator")
print("\n=== Summary (FP / PAS ratio per surface) ===")
for name, species, _ in panels:
    pas, fp = (np.asarray(results[op].arrays[name]) for op in OPERATORS)
    ratio = (fp if species is None else fp[:, species]) / (pas if species is None else pas[:, species])
    print(f"  {name:26s} {np.array2string(ratio, precision=3)}")
