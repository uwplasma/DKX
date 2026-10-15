"""Tutorial 03 -- an existing SFINCS ``input.namelist`` deck through the CLI.

DKX reads SFINCS v3 decks unchanged and writes the same ``sfincsOutput``
datasets (HDF5, NetCDF or NPZ by file suffix), so a SFINCS workflow moves over
by changing the executable:

    dkx sfincs write-output --input examples/tutorials/03_sfincs_namelist.namelist --out out.h5
    dkx convert examples/tutorials/03_sfincs_namelist.namelist   # -> native TOML

The deck is W7-X (``geometryScheme = 4``) at r/a = 0.5 with protons and
electrons.  In SFINCS normalization the radial particle flux of species s is
``particleFlux_vm_psiHat`` and the flux-surface-averaged parallel flow is
``FSABFlow``; their charge-weighted sum sum_s Z_s Gamma_s is the radial
current, which is not zero here because Er = 0 is not the ambipolar field
(tutorial 08 finds the Er that zeroes it).  In Python the same deck is one
call: ``dkx.run("03_sfincs_namelist.namelist", out="out.h5", emit=print)``.

Run:              python examples/tutorials/03_sfincs_namelist.py
Smoke mode:       DKX_EXAMPLES_CI=1 (the deck is already seconds-sized)
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dkx.io import read_sfincs_h5

# 2. Input parameters
DECK = Path(__file__).with_suffix(".namelist")
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
H5 = OUT / "sfincsOutput.h5"
SPECIES = ("protons", "electrons")

# 3. Run -- the CLI streams the SFINCS-style progress block
OUT.mkdir(parents=True, exist_ok=True)
command = [sys.executable, "-m", "dkx", "sfincs", "write-output", "--input", str(DECK), "--out", str(H5)]
print("$ " + " ".join(command[2:]), flush=True)
subprocess.run(command, check=True)

# 4. Plot, save and print
data = read_sfincs_h5(H5)
gamma = np.asarray(data["particleFlux_vm_psiHat"]).reshape(-1)
heat = np.asarray(data["heatFlux_vm_psiHat"]).reshape(-1)
flow = np.asarray(data["FSABFlow"]).reshape(-1)
zs = np.asarray(data["Zs"]).reshape(-1)
fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
for axis, values, title in zip(axes, (gamma, heat, flow), ("particle flux", "heat flux", "<B V_par>")):
    axis.bar(SPECIES, values, color=["tab:blue", "tab:orange"])
    axis.set_title(title + " (SFINCS units)")
    axis.axhline(0.0, color="k", lw=0.5)
fig.tight_layout()
fig.savefig(OUT / "fluxes.png", dpi=120)
print(f"\nsaved {H5}\nsaved {OUT / 'fluxes.png'}")
print("\n=== Summary ===")
for name, g, q, u in zip(SPECIES, gamma, heat, flow):
    print(f"  {name:9s}  particleFlux_vm_psiHat={g:+.4e}  heatFlux_vm_psiHat={q:+.4e}  FSABFlow={u:+.4e}")
print(f"  radial current sum_s Z_s Gamma_s = {float(np.sum(zs * gamma)):+.4e} (non-zero: Er=0 is not ambipolar)")
print(f"  FSABjHat (bootstrap current) = {float(np.asarray(data['FSABjHat']).reshape(-1)[0]):+.4e}")
