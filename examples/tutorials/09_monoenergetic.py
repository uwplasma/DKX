"""Monoenergetic coefficients across collisionality: the benchmark figure.

``D11*``, ``D31*`` and ``D33*`` against ``nu'``, one curve per ``E*``, is the
figure neoclassical codes are compared on (the ICNTS set).  The monoenergetic
problem drops the species and speed physics and solves one pitch-angle problem
per grid point, so a whole database costs about what a single profile solve
costs, and it is the cheapest honest way to benchmark a new configuration.

``D31*`` gets a linear axis while the other two get log: it is the coefficient
that changes sign, and ``|D31*|`` on a log axis hides exactly the zero crossing
a reader is looking for.

Physics: the analytic circular tokamak of rung 01, read from the SFINCS deck
beside this script, swept over five decades of collisionality at zero and
finite normalized radial electric field.  The coefficients are star-normalized,
so the familiar ``1/nu`` branch is divided out and ``D11*`` rises monotonically
at ``E* = 0``; at ``E* = 0.1`` the ExB detrapping turns it over.

Expected runtime: ~5 s on a laptop CPU for the 5 x 2 grid.

There is no ``case.toml`` for this rung: the native case schema accepts
``run.workflow = "monoenergetic"`` but the native executor does not implement
it yet, so the database is built through the stable SFINCS-deck entry point
``dkx.run_monoenergetic_database``.
"""

# 1. Imports
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

import dkx  # noqa: E402

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "04_monoenergetic_scan"
DECK = HERE / "input.namelist"
RESULT_FILE = OUT_DIR / "monoenergetic.nc"
DATABASE_FILE = OUT_DIR / "monoenergetic.npz"
PLOT_FILE = OUT_DIR / "monoenergetic.png"

# Collisionality grid, in the star normalization: nu' = nu R / (iota v).  Widen
# it to (1e-4, ..., 1e2) for the full published sweep; each decade costs one
# solve per E* value.
NU_PRIME = (1.0e-3, 1.0e-2, 1.0e-1, 1.0e0, 1.0e1)
# Normalized radial electric field E* = Er / (v B).  Zero is the reference
# curve; 0.1 is where ExB detrapping visibly suppresses the 1/nu regime.
E_STAR = (0.0, 1.0e-1)
SOLVE_METHOD = "auto"
TOLERANCE = 1.0e-10
# end of parameters

# 3. Geometry and species construction
# The monoenergetic problem has no species: the deck supplies the magnetic
# geometry (geometryScheme=1, a circular tokamak at rN = 0.3) and the pitch
# grid, and the driver overrides the rest.
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"geometry deck: {DECK}")

# 4. Physics and numerical configuration
print(f"nu' grid: {list(NU_PRIME)}")
print(f"E* grid:  {list(E_STAR)}")
print(f"solver:   method={SOLVE_METHOD} tol={TOLERANCE:g}")

# 5. Run
database = dkx.run_monoenergetic_database(
    DECK,
    NU_PRIME,
    E_STAR,
    output_path=DATABASE_FILE,
    solve_method=SOLVE_METHOD,
    tol=TOLERANCE,
)
nu = np.asarray(database.nu_prime, dtype=float)
e_star = np.asarray(database.e_star, dtype=float)
coefficients = {
    "D11": np.asarray(database.d11_star, dtype=float),
    "D31": np.asarray(database.d31_star, dtype=float),
    "D33": np.asarray(database.d33_star, dtype=float),
}

# 6. Print a scientific summary and certificate
print("\n=== Final results ===")
print(f"{'nu_prime':>10} {'E_star':>9} {'D11*':>13} {'D31*':>13} {'D33*':>13}")
for i, nu_value in enumerate(nu):
    for j, e_value in enumerate(e_star):
        print(
            f"{nu_value:>10.3g} {e_value:>9.3g} "
            f"{coefficients['D11'][i, j]:>13.5e} "
            f"{coefficients['D31'][i, j]:>13.5e} "
            f"{coefficients['D33'][i, j]:>13.5e}"
        )
# D33* is normalized by its collision-dominated value, so it must approach 1
# from below as nu' grows and trapping stops mattering.  That limit is a
# property of the operator, not something the solver is told, so how close the
# database gets to it is a free accuracy check.
d33_collisional = float(coefficients["D33"][-1, 0])
print(f"  D33* at the highest nu', E*=0: {d33_collisional:.6f} (collisional limit is 1)")
print(f"  distance from the collisional limit: {abs(d33_collisional - 1.0):.2e}")
assert d33_collisional <= 1.0 + 1e-9, "D33* exceeded its collisional limit"
# D11* must rise monotonically with collisionality at E* = 0; a database that
# does not is under-resolved in pitch angle whatever its residuals say.
d11_zero_field = coefficients["D11"][:, 0]
monotone = bool(np.all(np.diff(d11_zero_field) > 0.0))
print(f"  D11* monotone in nu' at E*=0: {monotone}")
assert monotone, "D11* is not monotone at E*=0: raise Nxi in input.namelist"
print(f"  D11* spans {d11_zero_field.min():.4g} to {d11_zero_field.max():.4g} (dimensionless)")

# 7. Save native result
# The database is not a profile Result, so it carries its own NetCDF layout:
# two grid axes and one array per coefficient.
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("nu_prime", nu.size)
    dataset.createDimension("e_star", e_star.size)
    dataset.createVariable("nu_prime", "f8", ("nu_prime",))[:] = nu
    dataset.createVariable("e_star", "f8", ("e_star",))[:] = e_star
    for name, grid in coefficients.items():
        dataset.createVariable(f"{name}_star", "f8", ("nu_prime", "e_star"))[:] = grid
    dataset.dkx_version = dkx.__version__
    dataset.source_deck = DECK.name
print(f"  Wrote result: {RESULT_FILE}")
print(f"  Wrote database: {DATABASE_FILE}")

# 8. Plot publication-ready outputs
figure, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), constrained_layout=True)
for axis, (label, grid) in zip(axes, coefficients.items()):
    for j, e_value in enumerate(e_star):
        column = grid[:, j]
        axis.plot(nu, column if label == "D31" else np.abs(column), "o-", ms=4,
                  label=rf"$E^*$={e_value:g}")  # fmt: skip
    axis.set_xscale("log")
    axis.set_xlabel(r"$\nu'$")
    axis.grid(alpha=0.3, which="both")
    if label == "D31":
        axis.set_ylabel(r"$D_{31}^*$")
        axis.axhline(0.0, color="0.7", lw=0.8)
    else:
        axis.set_yscale("log")
        axis.set_ylabel(rf"$|D_{{{label[1:]}}}^*|$")
    axis.legend(fontsize=8)
figure.suptitle("Monoenergetic transport coefficients")
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/04_monoenergetic_scan/run.py")
