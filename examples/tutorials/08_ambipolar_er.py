"""Tutorial 08 -- the radial electric field from ambipolarity: scan, then solve.

In a stellarator the ion and electron fluxes depend on E_r, and E_r adjusts
until no net charge leaves the surface:

    J_r(E_r) = e sum_s Z_s Gamma_s(E_r) = 0.

Step 1 scans J_r over prescribed E_r values to *see* the curve.  Step 2 runs
``workflow = "ambipolar_profile"`` from ``08_ambipolar_er.toml``, which
brackets every root, refines it, classifies it and records the selection:
an *ion root* (E_r < 0, set by the ions, usual at high collisionality), an
*electron root* (E_r > 0, from strong electron transport) and an unstable
root between them (dJ_r/dE_r < 0 is stable).  In a tokamak J_r is ~0 for any
E_r, so ambipolarity cannot fix the field there (Helander & Simakov 2008).

Geometry: analytic W7-X, deuterium + electrons, full Fokker-Planck.  The grid
teaches root finding; it is not a converged result.  ``08_ambipolar_er_w7x.toml``
is a production-scale case on a real W7-X wout to ``dkx validate``.

Run:              python examples/tutorials/08_ambipolar_er.py
CLI:              dkx run examples/tutorials/08_ambipolar_er.toml --out r.nc && dkx roots r.nc
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/08_ambipolar_er.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
CASE_FILE = Path(__file__).with_suffix(".toml")
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
ER_SCAN_KV_M = np.linspace(-5.0, 5.0, 3 if SMOKE else 9)  # prescribed fields for step 1
E = 1.602176634e-19  # C

# 3. Run
base = dkx.Case.from_file(CASE_FILE)
charges = np.array([s.charge for s in base.species], dtype=float)
j_r = []
for k, er in enumerate(ER_SCAN_KV_M):
    print(f"step 1, Er scan {k + 1}/{ER_SCAN_KV_M.size}: Er = {er:+.2f} kV/m", flush=True)
    case = replace(base, run=replace(base.run, workflow="profile", progress=False),
                   electric_field=type(base.electric_field)(mode="prescribed", value_kV_m=float(er)))
    j_r.append(E * np.asarray(dkx.run(case).arrays["particle_flux_m2_s"]) @ charges)
j_r = np.array(j_r)  # (Er, surface) in A m^-2
print("step 2, ambipolar root solve", flush=True)
if SMOKE:  # fewer bracketing samples
    base = replace(base, electric_field=replace(base.electric_field, search_points=3))
result = dkx.run(base)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
result.save(OUT / "result.nc")
psi = np.asarray(result.arrays["surface"], dtype=float)
roots = np.asarray(result.arrays["ambipolar_root_kV_m"], dtype=float)
kinds = np.asarray(result.arrays["ambipolar_root_type"], dtype=object)
counts = np.asarray(result.arrays["ambipolar_root_count"], dtype=int)
selected = np.asarray(result.arrays["electric_field_kV_m"], dtype=float)
fig, axis = plt.subplots(figsize=(6, 3.8))
for i, p in enumerate(psi):
    line, = axis.plot(ER_SCAN_KV_M, j_r[:, i], "o-", label=rf"$\psi_N$={p:.2f}")
    axis.plot(roots[i, : counts[i]], np.zeros(counts[i]), "x", color=line.get_color(), ms=10, mew=2)
axis.axhline(0.0, color="k", lw=0.5)
axis.set_xlabel(r"$E_r$ [kV/m]")
axis.set_ylabel(r"$J_r=e\sum_s Z_s\Gamma_s$ [A m$^{-2}$]")
axis.set_title("radial current; x = ambipolar roots")
axis.legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT / "ambipolar.png", dpi=120)
print(f"\nsaved {OUT / 'result.nc'}\nsaved {OUT / 'ambipolar.png'}")
print("\n=== Summary ===")
for i, p in enumerate(psi):
    found = ", ".join(f"{roots[i, k]:+.3f} kV/m [{kinds[i, k]}]" for k in range(counts[i]))
    print(f"  psi_N={p:.2f}: {counts[i]} root(s): {found}; selected Er = {selected[i]:+.3f} kV/m")
cert = result.certificate()
print(f"  selection rule: {cert['ambipolar_selection']}")
print(f"  all surfaces bracketed: {cert['ambipolar_all_surfaces_bracketed']}")
