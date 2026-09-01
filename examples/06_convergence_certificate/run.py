"""Am I converged?  Scan every resolution axis and plot the answer.

``scan_resolution.py`` prints one axis; this plots each one against the value
at the finest grid.  Read it as a budget: refine the axis still moving, cut
the ones that have flattened.  Cost is the product of the axes while error is
set by the worst of them, so refining everything at once buys expense rather
than accuracy.

Physics: the tokamak of ``run_tokamak.py``.  The bootstrap current is the
diagnostic because it is the most resolution-hungry moment here -- a narrow
feature in pitch angle, so Nxi binds long after the fluxes have settled.

Expected runtime: ~90 s on a laptop CPU, dominated by JAX compilation.

Achieved: Nxi binds by a wide margin -- 33.7% off at its smallest grid against
2.0% for Ntheta and 0.8% for Nx -- and needs Nxi = 24 to come inside 1%.  The
default Nxi=8 in ``run_tokamak.py`` is 34% low.  Watch Nx: its error falls
monotonically but the values oscillate, so one shrinking step is not evidence
of convergence there; two successive ones are.
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
