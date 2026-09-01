"""Monoenergetic coefficients across collisionality -> the benchmark figure.

``D11``, ``D31``, ``D33`` against ``nu'``, one curve per ``E*``, is the figure
neoclassical codes are compared on (the ICNTS set).  It is an ``RHSMode=3``
scan, which ``dkx.run_monoenergetic_database`` does in one call.

``D31*`` gets a linear axis while the others get log: it is the coefficient
that can change sign, and ``|D31|`` on a log axis hides a zero crossing --
the one feature of it a reader looks for.

Physics: the tokamak of ``run_tokamak.py`` over six decades.  These are
*star-normalized*, so the familiar 1/nu branch is divided out and ``D11*``
rises monotonically at ``E* = 0``.  At ``E* = 0.1`` the ExB detrapping
suppresses transport above ``nu' ~ 1`` and turns it over.

Expected runtime: ~90 s on a laptop CPU for the 7x3 grid.

Achieved: D11* spans 0.028 to 1.55e+03 at E* = 0, and at E* = 0.1 turns over
to 0.50 by nu' = 100 -- five decades below the zero-field curve.  D31* falls
0.56 -> ~5e-08; D33* rises 0.83 -> 1.
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"
OUT_DIR = Path(__file__).resolve().parent / "output"
DECK = Path(__file__).resolve().parent / "input.namelist"

NU_PRIME = (1e-3, 1e-1, 1e1) if CI else (1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2)
E_STAR = (0.0,) if CI else (0.0, 1.0e-3, 1.0e-1)
# E* = 1e-3 rather than 1e-2: between zero field and 1e-1 the D11 curves are
# nearly indistinguishable, so a grid of 0 / 0.1 / 0.3 spends two of its three
# curves in the same regime.
# ----------------------------- end of parameters ----------------------------

OUT_DIR.mkdir(exist_ok=True)

# One call scans the whole (nuPrime, EStar) grid.  RHSMode=3 is set by the
# database driver, so the deck's own RHSMode does not matter here.
database = dkx.run_monoenergetic_database(DECK, NU_PRIME, E_STAR)

# The database is a dataclass of arrays: nu_prime (n,), e_star (m,), and each
# coefficient on the (n, m) grid.  Star-normalized, hence the names.
nu = np.asarray(database.nu_prime, dtype=float)
e_star = np.asarray(database.e_star, dtype=float)
coefficients = {
    "D_{11}": np.asarray(database.d11_star, dtype=float),
    "D_{31}": np.asarray(database.d31_star, dtype=float),
    "D_{33}": np.asarray(database.d33_star, dtype=float),
}

print(f"{'nu_prime':>10} {'E_star':>9} {'D11*':>13} {'D31*':>13} {'D33*':>13}")
for i, nu_value in enumerate(nu):
    for j, e_value in enumerate(e_star):
        print(f"{nu_value:>10.3g} {e_value:>9.3g} "
              f"{coefficients['D_{11}'][i, j]:>13.5e} "
              f"{coefficients['D_{31}'][i, j]:>13.5e} "
              f"{coefficients['D_{33}'][i, j]:>13.5e}")

figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
for axis, (label, grid) in zip(axes, coefficients.items()):
    for j, e_value in enumerate(e_star):
        column = grid[:, j]
        axis.plot(nu, column if label == "D_{31}" else np.abs(column), "o-", ms=4,
                  label=rf"$E^*$={e_value:g}")  # fmt: skip
    axis.set_xscale("log")
    axis.set_xlabel(r"$\nu'$")
    axis.grid(alpha=0.3, which="both")
    if label == "D_{31}":
        axis.set_ylabel(rf"${label}^*$")
        axis.axhline(0.0, color="0.7", lw=0.8)
    else:
        axis.set_yscale("log")
        axis.set_ylabel(rf"$|{label}^*|$")
    axis.legend(fontsize=8)

figure.suptitle("Monoenergetic transport coefficients")
out = OUT_DIR / "plot_monoenergetic.png"
figure.savefig(out, dpi=150)
plt.show(block=False)
print(f"wrote {out}")
