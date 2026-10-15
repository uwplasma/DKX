"""Tutorial 07 -- several species: ions, electrons and a carbon impurity.

Every species s gets its own f1_s, coupled to the others through the
collision operator (cross-species friction and energy exchange).  The
quantities to look at are

* the particle flux Gamma_s of each species, and
* the radial current  J_r = e sum_s Z_s Gamma_s.  In an exactly
  axisymmetric tokamak the fluxes are intrinsically ambipolar (J_r = 0 for
  any E_r).  The built-in ``tokamak`` field carries a small l=2, n=10 helical
  ripple, so here J_r is not zero; tutorial 08 finds the E_r that zeroes it.

A highly charged impurity (C6+, Z = 6) is strongly coupled to the main ions by
friction.  Whether it is pulled in (accumulation) or pushed out (temperature
screening) depends on the sign of its flux, which the summary prints.
Quasineutrality n_e = sum_i Z_i n_i is imposed by the electron density below.

Run:              python examples/tutorials/07_species_and_impurity.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/07_species_and_impurity.py
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
N_D = np.array([8.0e19, 7.0e19, 5.8e19])  # deuterium density per surface [m^-3]
CARBON_FRACTION = 0.01  # n_C / n_D
T_KEV = [1.0, 0.8, 0.6]  # all species share one temperature profile
SPECIES = [
    {"name": "deuterium", "charge": 1, "mass_amu": 2.014, "density_m3": list(N_D)},
    {"name": "carbon", "charge": 6, "mass_amu": 12.011, "density_m3": list(CARBON_FRACTION * N_D)},
    # quasineutrality: n_e = n_D + 6 n_C
    {"name": "electron", "charge": -1, "mass_amu": 5.48579909e-4,
     "density_m3": list((1 + 6 * CARBON_FRACTION) * N_D)},
]
CASE = {
    "name": "species_and_impurity",
    "run": {"workflow": "profile", "progress": True},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16, 0.25]},
    "species": [{**s, "temperature_keV": T_KEV} for s in SPECIES],
    # A momentum-conserving operator: inter-species friction is the physics here.
    "physics": {"collisions": "linearized_fokker_planck"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "solver": {"method": "auto"},  # "auto" picks the linear-solver route from the operator
    "resolution": {"theta": 13, "zeta": 1, "pitch": 16, "speed": 6},
}
if SMOKE:
    CASE["resolution"] = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}

# 3. Run
result = dkx.run(dkx.Case.from_mapping(CASE))

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
result.save(OUT / "result.nc")
psi = np.asarray(result.arrays["surface"], dtype=float)
gamma = np.asarray(result.arrays["particle_flux_m2_s"])  # (surface, species)
charges = np.array([s["charge"] for s in SPECIES], dtype=float)
fig, axis = plt.subplots(figsize=(6, 3.5))
for k, s in enumerate(SPECIES):
    axis.plot(psi, s["charge"] * gamma[:, k], "o-", label=f"{s['name']}  (Z={s['charge']:+d})")
axis.plot(psi, gamma @ charges, "k--", label=r"$\sum_s Z_s\Gamma_s$")
axis.set_xlabel(r"$\psi_N$")
axis.set_ylabel(r"$Z_s\Gamma_s$ [m$^{-2}$ s$^{-1}$]")
axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "species.png", dpi=120)
print(f"\nsaved {OUT / 'result.nc'}\nsaved {OUT / 'species.png'}")
print("\n=== Summary ===")
for i, p in enumerate(psi):
    row = "  ".join(f"{s['name']}={gamma[i, k]:+.3e}" for k, s in enumerate(SPECIES))
    j_r = gamma[i] @ charges
    print(f"  psi_N={p:.2f}  Gamma: {row}  |sum_s Z_s Gamma_s| / Gamma_D = {abs(j_r / gamma[i, 0]):.2e}")
direction = "outward (screened)" if np.all(gamma[:, 1] > 0) else "inward (accumulating)"
print(f"  carbon flux is {direction} on every surface: {np.all(np.sign(gamma[:, 1]) == np.sign(gamma[0, 1]))}")
