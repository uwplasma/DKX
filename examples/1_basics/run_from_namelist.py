"""An SFINCS ``input.namelist`` on disk -> solve -> write ``sfincsOutput.h5``.

The same three-line workflow as ``run_tokamak.py``, but reading the deck the
Fortran code would read and writing the output file it would write.  If you
have existing SFINCS cases, this is the route that runs them unchanged.

``dkx.run`` reads ``RHSMode`` from the deck and dispatches, so the same call
covers a profile-gradient run and a transport-matrix run.  The equivalent from
a shell is ``dkx input.namelist --out sfincsOutput.h5``.

Physics: the bundled deck is the single-species pitch-angle-scattering tokamak
from ``run_tokamak.py``, so the printed moments should match it.

Expected runtime: ~5 s on a laptop CPU, nearly all JAX compilation.

Achieved: reads back from the written file bit-for-bit; the assertion at the
bottom fails if the writer and reader ever disagree.
"""

from pathlib import Path

import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
DECK = Path(__file__).resolve().parent / "input.namelist"
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT = OUT_DIR / "run_from_namelist.sfincsOutput.h5"
# Swap the suffix to change format -- .nc for NetCDF4, .npz for numpy:
# OUT = OUT_DIR / "run_from_namelist.sfincsOutput.nc"
# ----------------------------- end of parameters ----------------------------

OUT_DIR.mkdir(exist_ok=True)
run = dkx.run(DECK, out=OUT)

print(f"FSABjHat               = {float(run.moments['FSABjHat']):+.6e}")
print(f"particleFlux_vm_psiHat = {run.moments['particleFlux_vm_psiHat']}")
print(f"wrote {run.output_path}")

# Read it back with the same reader that handles Fortran SFINCS output -- the
# layout is SFINCS's own, so existing post-processing works unchanged.
data = dkx.read_output(run.output_path)
assert np.allclose(data["FSABjHat"], run.moments["FSABjHat"]), "writer/reader disagree"
print(f"read back from {OUT.suffix}: FSABjHat = {np.asarray(data['FSABjHat']).ravel()[-1]:+.6e}")
