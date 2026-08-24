"""Am I converged?  Scan every resolution axis and plot the answer.

``scan_resolution.py`` prints one axis.  This is the figure you make before
trusting a production number: each grid axis varied on its own, everything
else held fixed, plotted against the value at the finest grid.

Read it as a budget.  The axis whose curve is still moving is the one to
spend points on; the axes that have flattened are ones you can *cut* to buy
those points back.  Refining everything at once is how a deck ends up
expensive without being converged, because the cost is the product of the
axes and the error is set by the worst of them.

Physics: the tokamak of ``run_tokamak.py``.  The bootstrap current is the
diagnostic because it is the most resolution-hungry moment here -- it is a
narrow feature of the pitch-angle variable, so Nxi binds long after the
fluxes have settled.

Expected runtime: ~90 s on a laptop CPU; every new grid pays its own JAX
compilation, which dominates at these sizes.

Achieved: Nxi is the binding axis by a wide margin.  At the smallest grid
tried it is 33.7% off, against 2.0% for Ntheta and 0.8% for Nx, and it takes
Nxi = 24 to come inside 1%.  The default Nxi=8 that ``run_tokamak.py`` uses is
34% low -- exactly the kind of error this figure exists to catch before it
reaches a paper.

Nx is worth a second look: its error falls 0.81%, 0.65%, 0.30%, 0.13% but the
values themselves oscillate rather than approach from one side.  That is
normal for the spectral speed grid, and it means a single refinement step is
not evidence of convergence on this axis -- two successive steps that both
shrink are.  Reading one step as a trend is how an oscillating axis gets
declared converged early.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"

BASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=15, Nzeta=1, Nxi=32, NL=4, Nx=6,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)

# Each axis is scanned with the others held at their BASE value.
AXES = {
    "Nxi": (8, 16, 24, 32, 40, 48),
    "Ntheta": (9, 13, 15, 19, 23),
    "Nx": (4, 5, 6, 7, 8),
}
if CI:
    AXES = {"Nxi": (8, 16), "Ntheta": (9, 13), "Nx": (4, 5)}

DIAGNOSTIC = "FSABjHat"
OUT = Path(__file__).resolve().parent / "output" / "plot_convergence_scan.png"
# ----------------------------- end of parameters ----------------------------

case = dkx.SfincsInput.from_params(**BASE)
results = {}

for axis, values in AXES.items():
    series = []
    print(f"\nscanning {axis} (others fixed at the BASE grid)")
    print(f"{axis:>8} {DIAGNOSTIC:>16} {'vs finest':>10}")
    for value in values:
        run = dkx.run(case, **{axis: value})
        series.append(float(np.asarray(run.moments[DIAGNOSTIC]).ravel()[0]))
    reference = series[-1]
    for value, item in zip(values, series):
        print(f"{value:>8} {item:>16.8e} {abs(item / reference - 1.0):>9.2%}")
    results[axis] = (np.array(values), np.array(series), reference)

figure, axes_row = plt.subplots(
    1, len(results), figsize=(4.2 * len(results), 3.8), constrained_layout=True
)
axes_row = np.atleast_1d(axes_row)

for panel, (axis, (values, series, reference)) in zip(axes_row, results.items()):
    error = np.abs(series / reference - 1.0)
    # The finest point is the reference, so its error is identically zero and
    # cannot go on a log axis.  Dropping it silently would hide what everything
    # is measured against, so it becomes a marked vertical line instead of a
    # point pinned at some arbitrary floor -- which would drag the y-axis down
    # and flatten the range that matters.
    panel.semilogy(values[:-1], error[:-1], "o-")
    panel.axvline(values[-1], color="0.6", ls=":", lw=1.0, label=f"reference ({values[-1]})")
    panel.axhline(0.01, color="C3", ls="--", lw=1.0, label="1%")
    panel.set_xlabel(axis)
    panel.set_ylabel("relative change vs finest grid")
    panel.set_title(f"{axis} (others fixed)")
    panel.grid(alpha=0.25, which="both")
    panel.legend(frameon=False, loc="lower left")

figure.suptitle(f"Convergence of {DIAGNOSTIC}: one axis at a time")
OUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUT, dpi=150)
print(f"\nwrote {OUT}")
print("Spend points on the axis still moving; cut the ones that have flattened.")
