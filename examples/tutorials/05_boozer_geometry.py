"""Tutorial 05 -- Boozer-coordinate geometry from a ``.bc`` file.

In Boozer coordinates (psi, theta_B, zeta_B) field lines are straight and |B|
is a Fourier series B = sum_mn B_mn cos(m theta_B - n N zeta_B) (+ sin terms
without stellarator symmetry).  The ``.bc`` file stores exactly that spectrum
per surface, so nothing is transformed on the way in.  This fixture is not
stellarator-symmetric; DKX detects the column convention itself.

Watch the printed solver route: for an axisymmetric field the operator is
block-tridiagonal in Legendre index and DKX solves it directly; here the
helical coupling breaks that structure and ``method = "auto"`` chooses a
recycled Krylov solver (GCROT) instead, recording the reason in the
certificate.  Nothing in the case asked for it.

Run:              python examples/tutorials/05_boozer_geometry.py
CLI:              dkx run examples/tutorials/05_boozer_geometry.toml --out result.nc
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/05_boozer_geometry.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from dataclasses import replace
from pathlib import Path

import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
CASE_FILE = Path(__file__).with_suffix(".toml")  # every physics input is in here
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
SMOKE_RESOLUTION = {"theta": 5, "zeta": 5, "pitch": 8, "speed": 4}

# 3. Run
case = dkx.Case.from_file(CASE_FILE)
if SMOKE:
    case = replace(case, resolution=replace(case.resolution, **SMOKE_RESOLUTION))
result = dkx.run(case)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
print(f"\nsaved {result.save(OUT / 'result.nc')}")
print(f"saved {result.plot(OUT / 'result.png')}")
cert = result.certificate()
print("\n=== Summary ===")
for i, psi in enumerate(np.asarray(result.arrays["surface"], dtype=float)):
    print(f"  psi_N={psi:.2f}  Gamma={result.arrays['particle_flux_m2_s'][i, 0]:+.3e} m^-2 s^-1"
          f"  <j.B>={result.arrays['parallel_current_A_T_m2'][i]:+.3e} A T m^-2")
print(f"  solver route: {cert['solver_route']}  converged: {cert['converged']}")
print(f"  route reason: {cert['route_reason']}")
