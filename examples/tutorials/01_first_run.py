"""Tutorial 01 -- your first DKX run: a tokamak, one ion species, Python only.

DKX solves the steady drift-kinetic equation for the small non-Maxwellian part
f1 of each species on a flux surface,

    (v_par b + v_E) . grad f1  -  C(f1)  =  - v_d . grad(psi) dfM/dpsi,

driven on the right by the density and temperature gradients through the
magnetic drift v_d; C is the collision operator.  Velocity moments of f1 give
the neoclassical particle flux Gamma, heat flux Q and bootstrap current <j.B>.
Here: the built-in analytic ``tokamak`` field, deuterium, pitch-angle
scattering (PAS) collisions, three surfaces, no radial electric field.

Read the printed table: Gamma and Q are positive (outward, down the gradient)
and <j.B> is the bootstrap current.  The certificate says the *linear solve*
converged; whether the *resolution* is converged is tutorial 13.

Run:              python examples/tutorials/01_first_run.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/01_first_run.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from pathlib import Path

import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"  # tiny grids, for CI
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
CASE = {
    "name": "first_run",
    "run": {"workflow": "profile", "progress": True},  # progress: print every phase
    # Surfaces are normalized toroidal flux psi_N = psi/psi_edge; profiles give
    # one value per surface, and their radial gradients drive the transport.
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16, 0.25]},
    "species": [
        {"name": "deuterium", "charge": 1, "mass_amu": 2.014,
         "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]},
    ],
    # PAS keeps only pitch-angle scattering in C(f1): cheap, but it does not
    # conserve momentum.  Tutorial 06 compares it with full Fokker-Planck.
    "physics": {"collisions": "pitch_angle_scattering"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "solver": {"method": "auto"},  # "auto" picks the linear-solver route from the operator
    # Poloidal/toroidal grid points, Legendre modes in pitch, speed nodes.
    "resolution": {"theta": 13, "zeta": 1, "pitch": 16, "speed": 5},
}
if SMOKE:
    CASE["resolution"] = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}

# 3. Run
result = dkx.run(dkx.Case.from_mapping(CASE))

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
print(f"\nsaved {result.save(OUT / 'result.nc')}")
print(f"saved {result.plot(OUT / 'result.png')}")
cert = result.certificate()
print("\n=== Summary ===")
for i, psi in enumerate(np.asarray(result.arrays["surface"], dtype=float)):
    print(f"  psi_N={psi:.2f}  Gamma={result.arrays['particle_flux_m2_s'][i, 0]:+.3e} m^-2 s^-1"
          f"  Q={result.arrays['heat_flux_W_m2'][i, 0]:+.3e} W m^-2"
          f"  <j.B>={result.arrays['parallel_current_A_T_m2'][i]:+.3e} A T m^-2")
print(f"  converged: {cert['converged']}  route: {cert['solver_route']}"
      f"  residual: {cert['residual_norm']:.1e}")
