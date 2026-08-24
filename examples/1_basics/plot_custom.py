"""Build your own figure from ``run.moments``.  The flexible plotting route.

``dkx.plot`` draws the standard panel; for anything else -- a paper figure, a
comparison, other units -- read the numbers off the run and use matplotlib.
That is the normal way, not a fallback, which is why ``run.moments`` is a
plain dict of arrays keyed by the ``sfincsOutput`` names.  :mod:`dkx.units`
holds the SFINCS reference set, so SI is one multiplication away.

Physics: the circular tokamak of ``run_tokamak.py`` across three decades of
``nu_n``.  The bootstrap current is a banana-regime effect, largest at the
collisionless end and collapsing as collisions detrap the particles that
carry it.

Expected runtime: ~20 s on a laptop CPU for the seven points.

Achieved: 177.1 kA/m^2 at nu_n = 1e-3, 166.5 at 1e-2, 3.2 at 1 -- flat below
nu_n ~ 3e-3, then falling by a factor of 56.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx
from dkx.units import CURRENT_DENSITY

# The backend is deliberately not forced.  Run this in Spyder or a notebook and
# the figure appears in the plots pane; run it headless and matplotlib picks a
# file-only backend by itself.  A script that hard-codes Agg can never show you
# anything.

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"
OUT_DIR = Path(__file__).resolve().parent / "output"

BASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=15, Nzeta=1, Nxi=8, NL=4, Nx=6,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0,
)
NU_VALUES = np.logspace(-3, 0, 4 if CI else 7)
# ----------------------------- end of parameters ----------------------------

OUT_DIR.mkdir(exist_ok=True)
case = dkx.SfincsInput.from_params(**BASE, nu_n=1.0e-2)

# One solve per point.  run.moments is a dict, so pulling a series out of a
# scan is a list comprehension.
runs = [dkx.run(case, nu_n=float(nu)) for nu in NU_VALUES]
bootstrap = np.array([float(r.moments["FSABjHatOverRootFSAB2"]) for r in runs])
flux = np.array([float(np.ravel(r.moments["particleFlux_vm_psiHat"])[0]) for r in runs])

# FSABjHatOverRootFSAB2 carries e*nBar*vBar, so this is kA/m^2 (see dkx.units).
bootstrap_kA = bootstrap * CURRENT_DENSITY / 1.0e3

figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.0), constrained_layout=True)
left.plot(NU_VALUES, bootstrap_kA, "o-")
left.set(xscale="log", xlabel=r"$\nu_n$", ylabel=r"$\langle j_\parallel B\rangle/\sqrt{\langle B^2\rangle}$  [kA/m$^2$]")
left.set_title("bootstrap current", fontsize=11)
left.grid(alpha=0.3)

right.plot(NU_VALUES, flux, "s-", color="tab:orange")
right.set(xscale="log", yscale="log", xlabel=r"$\nu_n$",
          ylabel=r"$\langle\Gamma\cdot\nabla\hat\psi\rangle$  [SFINCS units]")  # fmt: skip
right.set_title("radial particle flux", fontsize=11)
right.grid(alpha=0.3)

figure.suptitle("Circular tokamak: collisionality scan")
out = OUT_DIR / "plot_custom.png"
figure.savefig(out, dpi=150)
plt.show(block=False)  # shows in Spyder/Jupyter; never blocks a batch run
print(f"peak bootstrap {bootstrap_kA.max():+.3f} kA/m^2 at nu_n = {NU_VALUES[bootstrap_kA.argmax()]:.3g}")
print(f"wrote {out}")
