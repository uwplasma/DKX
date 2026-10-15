"""Tutorial 09 -- a monoenergetic database: D11*, D31*, D33* versus collisionality.

Stellarator transport codes (DKES, MONKES, ...) and the ICNTS benchmark work
with *monoenergetic* coefficients: freeze the particle speed v, keep only
pitch-angle scattering at rate nu, and solve one pitch-angle problem per
(nu', E*) with nu' = nu R/(iota v) and E* = E_r/(v B).  Three coefficients
follow, normalized ("star") to the plateau/collisional values:

* D11* radial diffusion  (Gamma ~ -D11 dn/dr): the 1/nu, plateau and
  Pfirsch-Schlueter regimes in turn as nu' grows;
* D31* bootstrap coefficient (it can change sign, so it gets a linear axis);
* D33* parallel conductivity, which must approach 1 from below as nu' grows.

Energy convolution of these against a Maxwellian gives the full fluxes,
which is why a database is the cheapest way to characterize a configuration.
Geometry: the circular tokamak from ``09_monoenergetic.namelist``; at E* > 0
ExB precession detraps particles and turns D11* over at low nu'.

Run:              python examples/tutorials/09_monoenergetic.py
CLI:              dkx sfincs monoenergetic-database --input examples/tutorials/09_monoenergetic.namelist
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/09_monoenergetic.py
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
DECK = Path(__file__).with_suffix(".namelist")  # geometry and pitch grid
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
NU_PRIME = np.logspace(-3, 1, 3 if SMOKE else 9)  # collisionality nu'
E_STAR = (0.0, 0.1)  # normalized radial electric field E*

# 3. Run -- one pitch-angle solve per (nu', E*) point, each one printed
OUT.mkdir(parents=True, exist_ok=True)
db = dkx.run_monoenergetic_database(DECK, NU_PRIME, E_STAR, output_path=OUT / "database.npz", emit=print)
coefficients = {name: np.asarray(getattr(db, f"{name.lower()}_star"), dtype=float) for name in ("D11", "D31", "D33")}

# 4. Plot, save and print
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
for axis, (name, grid) in zip(axes, coefficients.items()):
    for j, e_star in enumerate(E_STAR):
        axis.plot(NU_PRIME, grid[:, j] if name == "D31" else np.abs(grid[:, j]), "o-", label=rf"$E^*$={e_star:g}")
    axis.set_xscale("log")
    axis.set_yscale("linear" if name == "D31" else "log")
    axis.set_xlabel(r"$\nu'$")
    axis.set_title(rf"${name[0]}_{{{name[1:]}}}^*$")
    axis.legend(fontsize=8)
fig.savefig(OUT / "monoenergetic.png", dpi=120)
print(f"\nsaved {OUT / 'database.npz'}\nsaved {OUT / 'monoenergetic.png'}")
print("\n=== Summary ===")
print(f"  {'nu_prime':>9} {'E*':>5} {'D11*':>12} {'D31*':>12} {'D33*':>12}")
for i, nu in enumerate(NU_PRIME):
    for j, e_star in enumerate(E_STAR):
        print(f"  {nu:9.3g} {e_star:5.2g}" + "".join(f" {grid[i, j]:12.4e}" for grid in coefficients.values()))
print(f"  D33* at the largest nu' (collisional limit is 1): {coefficients['D33'][-1, 0]:.4f}")
print(f"  D11* monotone in nu' at E*=0: {bool(np.all(np.diff(coefficients['D11'][:, 0]) > 0))}")
