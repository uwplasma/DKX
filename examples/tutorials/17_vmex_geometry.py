"""Tutorial 17 -- VMEX to DKX: solve a stellarator equilibrium, then its transport.

VMEX is the JAX VMEC.  It reads a VMEC input deck, solves the fixed-boundary
equilibrium and writes a standard ``wout`` file, which DKX reads exactly as
tutorial 04 reads one from VMEC2000.  Nothing else in the case changes.  The
deck is the two-field-period seed of a quasi-axisymmetric optimization
(``examples/data/input.minimal_seed_nfp2``) with a rotating-ellipse
perturbation added to its boundary, which gives the field its rotational
transform; the radial grid is cut to 13 surfaces so the equilibrium solves
in seconds.

Needs the optional ``vmex`` package (``pip install vmex``).  The advanced
scripts differentiate this chain, boundary -> equilibrium -> Boozer -> DKX.

Run:              python examples/tutorials/17_vmex_geometry.py
Runtime:          about 68 s on a loaded 36-core CPU node, compilation included.
"""

# 1. Imports
from dataclasses import replace
from pathlib import Path

import numpy as np
import vmex as vj

import dkx

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
INPUT = Path(__file__).resolve().parents[1] / "data" / "input.minimal_seed_nfp2"  # VMEC input deck
PERTURBATION = 0.05  # rotating-ellipse amplitude added to the boundary [m]
NS = 13  # radial surfaces of the equilibrium solve
FTOL = 1e-10  # force-residual tolerance of the equilibrium solve
SURFACES = [0.25, 0.36, 0.49]  # normalized toroidal flux where DKX solves
CASE = {
    "name": "vmex_geometry",
    "run": {"workflow": "profile", "progress": True},
    "geometry": {"format": "vmec", "file": str(OUT / "wout_vmex.nc"), "surfaces": SURFACES},
    "species": [
        {"name": "deuterium", "charge": 1, "mass_amu": 2.014,
         "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]},
    ],
    "physics": {"collisions": "pitch_angle_scattering"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "solver": {"method": "auto"},
    "resolution": {"theta": 9, "zeta": 9, "pitch": 8, "speed": 4},  # coarse: seconds, not converged
}

# 3. Run -- the equilibrium first, then the transport on it
OUT.mkdir(parents=True, exist_ok=True)
inp = vj.VmecInput.from_file(INPUT)
rbc, zbs = inp.rbc.copy(), inp.zbs.copy()  # VmecInput is frozen: copy before shaping the boundary
rbc[inp.ntor - 1, 1], zbs[inp.ntor - 1, 1] = -PERTURBATION, PERTURBATION  # the (m, n) = (1, -1) harmonic
inp = replace(inp, rbc=rbc, zbs=zbs, ns_array=np.array([NS]), ftol_array=np.array([FTOL]))
equilibrium = vj.solve_multigrid(inp, verbose=False)
wout = vj.write_wout(OUT / "wout_vmex.nc", vj.wout_from_result(inp, equilibrium))
result = dkx.run(dkx.Case.from_mapping(CASE))

# 4. Plot, save and print
print(f"\nsaved {wout}\nsaved {result.save(OUT / 'result.nc')}\nsaved {result.plot(OUT / 'result.png')}")
cert = result.certificate()
print("\n=== Summary ===")
print(f"  VMEX: converged {equilibrium.converged}  fsqr={equilibrium.fsqr:.1e}  fsqz={equilibrium.fsqz:.1e}"
      f"  fsql={equilibrium.fsql:.1e}")
for i, psi in enumerate(np.asarray(result.arrays["surface"], dtype=float)):
    print(f"  psi_N={psi:.2f}  Gamma={result.arrays['particle_flux_m2_s'][i, 0]:+.3e} m^-2 s^-1"
          f"  <j.B>={result.arrays['parallel_current_A_T_m2'][i]:+.3e} A T m^-2")
print(f"  geometry sha256: {cert['geometry_sha256'][:16]}  converged: {cert['converged']}")
