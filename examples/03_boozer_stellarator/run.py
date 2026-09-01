"""Add helical ripple to the tokamak -> the 1/nu regime appears.

The difference between a tokamak and a stellarator in one figure, and it is
one parameter: ``epsilon_h``, with everything else held fixed.

An axisymmetric field confines the trapped particles it creates; their orbits
close.  A ripple traps particles in local wells with no such protection, and
they drift out between collisions.  Fewer collisions then means *more*
transport: the 1/nu regime, and why stellarator optimization exists.

Physics: the tokamak of ``1_basics/run_tokamak.py`` with and without an l=2,
n=10 ripple.  ``Nzeta`` must resolve the ripple -- ``Nzeta=1`` is valid only
when ``epsilon_h=0``, and getting it wrong silently models a tokamak.

Expected runtime: ~2 min on a laptop CPU for both scans.

Achieved: the two run together at nu_n = 1 (1.15x apart), then diverge as
collisionality drops.  The tokamak flux falls; the rippled one bottoms out
near nu_n ~ 0.2 and climbs to 11700x the tokamak value by nu_n = 1e-4,
scaling there as nu^-0.87 on its way to the asymptotic 1/nu.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
NU_VALUES = np.logspace(-4, 0, 7)
EPSILON_H = 0.05067          # helical ripple amplitude; 0.0 is the tokamak
HELICITY_L, HELICITY_N = 2, 10
N_PERIODS = 10

BASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.5,
    B0OverBBar=1.0, epsilon_t=-0.07053, iota=0.4542,
    GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=13, Nxi=24, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0,
)

# Nzeta=1 is a statement that nothing varies along the field period.  That is
# true only for the axisymmetric case; the rippled case needs the ripple
# resolved, and using 1 there would quietly solve the tokamak again.
CASES = {
    "tokamak (no ripple)": dict(epsilon_h=0.0, Nzeta=1, NPeriods=N_PERIODS),
    f"stellarator (eps_h={EPSILON_H})": dict(
        epsilon_h=EPSILON_H, Nzeta=25, NPeriods=N_PERIODS,
        helicity_l=HELICITY_L, helicity_n=HELICITY_N,
    ),
}
OUT = Path(__file__).resolve().parent / "output" / "run_stellarator.png"
# ----------------------------- end of parameters ----------------------------

results = {}
for label, geometry in CASES.items():
    fluxes = []
    print(f"\n{label}")
    print(f"{'nu_n':>10} {'particle flux':>16}")
    for nu_n in NU_VALUES:
        run = dkx.run(**BASE, **geometry, nu_n=float(nu_n))
        flux = float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[0])
        fluxes.append(flux)
        print(f"{nu_n:>10.2e} {flux:>16.6e}")
    results[label] = np.array(fluxes)

figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)

for label, fluxes in results.items():
    left.loglog(NU_VALUES, np.abs(fluxes), "o-", label=label)
left.set_xlabel(r"$\nu_n$")
left.set_ylabel(r"$|\Gamma|$  [$\hat\psi$ units]")
left.set_title("particle flux vs collisionality")
left.legend(frameon=False, fontsize=9)

ratio = np.abs(results[list(CASES)[1]]) / np.abs(results[list(CASES)[0]])
right.loglog(NU_VALUES, ratio, "s-", color="C3")
right.axhline(1.0, color="0.6", ls="--", lw=1.0)
right.set_xlabel(r"$\nu_n$")
right.set_ylabel("rippled / axisymmetric")
right.set_title("what the ripple costs")

for axis in (left, right):
    axis.grid(alpha=0.25, which="both")

figure.suptitle("Helical ripple and the 1/$\\nu$ regime")
OUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUT, dpi=150)
print(f"\nripple penalty: {ratio.min():.2f}x at nu_n={NU_VALUES[ratio.argmin()]:.1e}, "
      f"{ratio.max():.2f}x at nu_n={NU_VALUES[ratio.argmax()]:.1e}")

# Fit the slope over the two most collisionless points rather than quoting an
# exponent from theory: 1/nu is the asymptote, and what a finite grid at finite
# collisionality actually delivers is a number worth seeing.
rippled = np.abs(results[list(CASES)[1]])
slope = np.log(rippled[0] / rippled[1]) / np.log(NU_VALUES[0] / NU_VALUES[1])
print(f"rippled branch scales as nu^{slope:+.2f} over the last decade "
      f"(1/nu is the asymptote, so -1 is the target)")
print(f"wrote {OUT}")
