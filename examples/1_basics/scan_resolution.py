"""One case, one parameter varied -> a convergence scan.  Six lines of loop.

``dkx.run(case, **overrides)`` returns the case with those SFINCS parameters
replaced, so a scan is a loop over values rather than a directory of decks.
The names are the Fortran namelist spellings; a misspelling raises rather than
silently doing nothing, because a scan that never changed resolution is worse
than one that fails.

Physics: the tokamak of ``run_tokamak.py`` at increasing pitch-angle
resolution.  ``Nxi`` is the axis that binds for the bootstrap current at low
collisionality, so it is the one worth converging first.

Expected runtime: ~30 s on a laptop CPU for the whole scan; each new grid pays
its own JAX compilation.

Achieved, on the grid below: FSABjHat moves 46.3% from Nxi=8 to 16, then 2.65%,
0.35%, and 0.024%, settling at +3.6304e-02 by Nxi=40.  Note that the default
Nxi=8 of ``run_tokamak.py`` is *not* converged for this quantity -- it is 34%
low.  That is the point of running a scan before trusting a number.
"""

import os

import dkx

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"

BASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=15, Nzeta=1, Nxi=8, NL=4, Nx=6,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)

SCAN_PARAMETER = "Nxi"
SCAN_VALUES = (8, 16) if CI else (8, 16, 24, 32, 40, 48)
# Any namelist parameter works the same way. Uncomment to scan the grid instead:
# SCAN_PARAMETER, SCAN_VALUES = "Ntheta", (9, 15, 21, 27)
# SCAN_PARAMETER, SCAN_VALUES = "Nx", (4, 6, 8, 10)
# ----------------------------- end of parameters ----------------------------

case = dkx.SfincsInput.from_params(**BASE)

print(f"{SCAN_PARAMETER:>8}  {'FSABjHat':>14}  {'change':>9}")
previous = None
for value in SCAN_VALUES:
    run = dkx.run(case, **{SCAN_PARAMETER: value})
    bootstrap = float(run.moments["FSABjHat"])
    change = "" if previous is None else f"{abs(bootstrap / previous - 1.0):8.2%}"
    print(f"{value:>8}  {bootstrap:+14.8e}  {change:>9}")
    previous = bootstrap

print("\nConverged when the change column stops moving; that is the resolution to use.")
