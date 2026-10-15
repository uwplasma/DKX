"""Tutorial 04 -- real stellarator geometry from a VMEC ``wout`` file.

The only change from tutorials 01-02 is the ``[geometry]`` block of
``04_vmec_geometry.toml``: ``format = "vmec"`` and a file.  DKX reads |B|, the
Jacobian and the covariant field components on each requested surface and
builds the same drift-kinetic operator; the equilibrium is an input, not a
different code path.  |B| now varies along both angles, so the
``zeta`` grid now matters.  The equilibrium is the unoptimized
rotating-ellipse *seed* of a two-field-period QA optimization, so it is far
from quasisymmetric: trapped particles drift out of the helical wells (the
1/nu regime) and Gamma comes out orders of magnitude above tutorial 01.

Inputs live in the TOML file (edit it, or point ``file`` at your own wout);
the script only shrinks the grid in smoke mode.  The certificate pins the
equilibrium's sha256, so a saved result records exactly which file it used.

Run:              python examples/tutorials/04_vmec_geometry.py
CLI:              dkx run examples/tutorials/04_vmec_geometry.toml --out result.nc
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/04_vmec_geometry.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import lzma
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
CASE_FILE = Path(__file__).with_suffix(".toml")  # every physics input is in here
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
SMOKE_RESOLUTION = {"theta": 9, "zeta": 9, "pitch": 8, "speed": 4}

# 3. Run
case = dkx.Case.from_file(CASE_FILE)
wout = case.geometry_path  # shipped as .nc.xz: decompress once, beside the archive
if not wout.exists():
    with lzma.open(wout.with_name(wout.name + ".xz")) as src, open(wout, "wb") as dst:
        shutil.copyfileobj(src, dst)
if SMOKE:
    case = replace(case, resolution=replace(case.resolution, **SMOKE_RESOLUTION))
print(f"case {case.case_id[:12]} on {wout.name}", flush=True)
result = dkx.run(case)

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
print(f"  geometry sha256: {cert['geometry_sha256'][:16]}  converged: {cert['converged']}"
      f"  route: {cert['solver_route']}")
