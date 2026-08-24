"""Solve -> one call -> a diagnostics figure.  The simple plotting route.

``dkx.plot`` takes whatever you already have -- the object ``dkx.run``
returned, a path to an ``sfincsOutput`` file, or a directory holding one --
and writes one figure.  The suffix picks the kind: ``.png`` for a compact
single-page summary, ``.pdf`` for the multi-page diagnostics panel.

Because the file layout is SFINCS's own, this plots Fortran SFINCS output
too: point it at a ``sfincsOutput.h5`` the Fortran code wrote and it works.

For a figure this does not draw, see ``plot_custom.py`` -- reading the numbers
off ``run.moments`` and using matplotlib directly is the normal way to make a
publication figure, not a fallback.

Physics: the circular tokamak of ``run_tokamak.py``.

Expected runtime: ~6 s on a laptop CPU, nearly all JAX compilation.

Achieved: writes both a .png and a .pdf; the panel shows |B| on the surface
and the per-species fluxes and flows.
"""

from pathlib import Path

import dkx

# --------------------------- parameters -------------------------------------
DECK = Path(__file__).resolve().parent / "input.namelist"
OUT_DIR = Path(__file__).resolve().parent / "output"
# ----------------------------- end of parameters ----------------------------

OUT_DIR.mkdir(exist_ok=True)

# plot() reads the output file, so the run has to write one.
run = dkx.run(DECK, out=OUT_DIR / "plot_summary.sfincsOutput.h5")

print(f"wrote {dkx.plot(run, out=OUT_DIR / 'plot_summary.png')}")
print(f"wrote {dkx.plot(run, out=OUT_DIR / 'plot_summary.pdf')}")

# The compact three-panel page instead of the full figure:
# dkx.plot(run, out=OUT_DIR / "compact.png", style="summary")

# Equivalent, straight from the file -- which is how you plot a run someone
# else did, including one Fortran SFINCS produced:
# print(dkx.plot(OUT_DIR / "plot_summary.sfincsOutput.h5", out=OUT_DIR / "again.png"))
