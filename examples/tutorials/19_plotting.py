"""Tutorial 19 -- plotting a run: the built-in figures and your own.

Three levels, from least to most work:

1. ``dkx.plot(run, "figure.png")``: the six-panel page (``|B|`` on the
   surface, the monoenergetic coefficients, the bootstrap current and the
   species fluxes), the same one ``dkx --plot`` writes;
2. ``dkx.plot(run, "figure.png", style="summary")``: the compact three-panel
   page, and a ``.pdf`` suffix there writes a four-page diagnostics book;
3. ``run.moments`` and matplotlib, for a figure that is going into a paper.

The run is the W7-X deck of tutorial 03.  ``dkx.plot`` reads the output file
the run writes, so pass ``out_path=`` to ``dkx.run`` first; the file is a normal
``sfincsOutput`` one, so the same call plots a Fortran SFINCS output.  Native
``dkx.Case`` results have their own ``result.plot(path)`` (tutorial 01).

Run:              python examples/tutorials/19_plotting.py
Runtime:          about 29 s on a loaded 36-core CPU node, compilation included.
"""

# 1. Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
DECK = Path(__file__).with_name("03_sfincs_namelist.namelist")  # any SFINCS input deck
SPECIES = ("protons", "electrons")  # species order in the deck

# 3. Run -- the output file is what dkx.plot reads
OUT.mkdir(parents=True, exist_ok=True)
run = dkx.run(DECK, out_path=OUT / "sfincsOutput.h5", emit=print)

# 4. Plot, save and print
panels = dkx.plot(run, OUT / "panels.png")
summary = dkx.plot(run, OUT / "summary.png", style="summary")
flux = np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()
heat = np.asarray(run.moments["heatFlux_vm_psiHat"]).ravel()
fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
for axis, values, title in zip(axes, (flux, heat), ("particle flux", "heat flux")):
    axis.bar(SPECIES, values, color=["tab:blue", "tab:orange"])
    axis.set_title(title + " (SFINCS units)", fontsize=9)
    axis.axhline(0.0, color="k", lw=0.5)
fig.savefig(OUT / "custom.png", dpi=120)
print("\n=== Summary ===")
for path in (panels, summary, OUT / "custom.png"):
    print(f"  saved {path}  ({path.stat().st_size / 1e3:.0f} kB)")
for name, g, q in zip(SPECIES, flux, heat):
    print(f"  {name:9s} particleFlux_vm_psiHat={g:+.4e}  heatFlux_vm_psiHat={q:+.4e}")
