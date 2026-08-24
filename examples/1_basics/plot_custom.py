"""Build your own figure from ``run.moments``.  The flexible plotting route.

``dkx.plot`` draws the standard panel.  For anything else -- a paper figure, a
comparison, an axis in units the panel does not use -- read the numbers off
the run and use matplotlib.  That is the normal way to make a figure, not a
fallback, and it is why ``run.moments`` is a plain dict of arrays keyed by the
``sfincsOutput`` names rather than an object you have to interrogate.

This one sweeps collisionality and draws the result in SI units, which is the
thing the standard panel deliberately does not do: :mod:`dkx.units` holds the
SFINCS reference set and the conversions, so a bootstrap current in kA/m^2 is
one multiplication away.

Physics: the circular tokamak of ``run_tokamak.py`` across three decades of
``nu_n``.  The bootstrap current is a banana-regime effect, so it is largest
at the collisionless end and collapses as collisions detrap the particles that
carry it -- flat below nu_n ~ 3e-3, then falling by a factor of 56 up to
nu_n = 1.

Expected runtime: ~20 s on a laptop CPU for the seven points.

Achieved: 177.1 kA/m^2 at nu_n = 1e-3, 166.5 at 1e-2, 3.2 at 1.
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dkx  # noqa: E402
from dkx.units import CURRENT_DENSITY  # noqa: E402

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
print(f"peak bootstrap {bootstrap_kA.max():+.3f} kA/m^2 at nu_n = {NU_VALUES[bootstrap_kA.argmax()]:.3g}")
print(f"wrote {out}")
