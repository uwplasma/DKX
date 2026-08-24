"""Solve the equilibrium and the transport in one script: VMEX -> DKX.

``geometryScheme=5`` reads a VMEC ``wout``, and VMEX produces one in the same
process -- so a boundary shape goes to neoclassical transport without a file
being carried between tools by hand.  The boundary here is a rotating ellipse
written as plain arrays, which is the whole equilibrium: no input deck.

The trap is ``ncurr=0``.  It means the iota profile is *specified*, via ``ai``,
not derived from a current -- so ``ai=[0.0]`` builds a stellarator with no
rotational transform.  VMEC converges happily, DKX runs, and every number is
meaningless.  ``ai=[0.42, 0.15]`` below gives iota rising 0.42 -> 0.57, which
the script prints so you can see it is not zero.

Requires the optional companion ``vmex`` (``pip install vmex``); it skips
cleanly when absent.

Physics: a 3-field-period vacuum rotating ellipse, one hydrogen species,
pitch-angle scattering, no radial electric field.

Expected runtime: ~25 s on a laptop CPU -- about 5 s of VMEX, 20 s of DKX.

Achieved: VMEX converges, then FSABjHat = -4.183e-02 and
particleFlux_vm_psiHat = +5.886e-05 at rN = 0.5.
"""

from pathlib import Path

import numpy as np

import dkx

try:
    import vmex
except ImportError:  # pragma: no cover - optional companion
    raise SystemExit(
        "This example needs vmex:  pip install vmex\n"
        "Everything in 1_basics runs without it."
    )

# --------------------------- parameters -------------------------------------
NFP, MPOL, NTOR = 3, 4, 3
IOTA_PROFILE = [0.42, 0.15]   # iota(s) = 0.42 + 0.15 s.  NOT [0.0] -- see above.
PHIEDGE = 0.08
R_N = 0.5                     # flux surface to solve the kinetics on

OUT_DIR = Path(__file__).resolve().parent / "output"
WOUT = OUT_DIR / "wout_rotating_ellipse.nc"

KINETICS = dict(
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=13, Nzeta=25, Nxi=24, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=0.01,
)
# ----------------------------- end of parameters ----------------------------

# The boundary is (n, m) Fourier coefficients of R and Z.  Four terms make a
# rotating ellipse: the major radius, the circular part, and the m=1 n=1 pair
# whose relative sign is what rotates the cross-section along the torus.
rbc = np.zeros((2 * NTOR + 1, MPOL))
zbs = np.zeros((2 * NTOR + 1, MPOL))


def coefficient(array, n, m, value):
    """Set the (n, m) coefficient; n is offset because n runs negative."""
    array[n + NTOR, m] = value


coefficient(rbc, 0, 0, 1.00)    # major radius
coefficient(rbc, 0, 1, 0.18)    # minor radius
coefficient(zbs, 0, 1, 0.18)
coefficient(rbc, 1, 1, 0.05)    # the rotating pair: opposite signs
coefficient(zbs, 1, 1, -0.05)
coefficient(rbc, 1, 0, 0.08)    # axis excursion
coefficient(zbs, 1, 0, -0.08)

equilibrium = vmex.VmecInput(
    nfp=NFP, mpol=MPOL, ntor=NTOR, lasym=False,
    ns_array=[16, 31], ftol_array=[1e-10, 1e-12], niter_array=[1500, 3000],
    phiedge=PHIEDGE, delt=0.9, nstep=200,
    ncurr=0, am=[0.0], ai=IOTA_PROFILE,   # am=[0.0] is a vacuum field: no pressure
    rbc=rbc, zbs=zbs,
)

print("solving the equilibrium with VMEX ...")
solved = vmex.solve(equilibrium)
print(f"  converged: {bool(solved.converged)}  after {int(solved.iterations)} iterations")
print(f"  iota: {float(solved.iotaf[0]):.3f} on axis -> {float(solved.iotaf[-1]):.3f} at the edge")
if abs(float(solved.iotaf[-1])) < 1e-6:
    raise SystemExit("iota is zero -- check ai; see the note about ncurr=0 above")

OUT_DIR.mkdir(parents=True, exist_ok=True)
vmex.write_wout(
    WOUT,
    vmex.wout_from_state(
        inp=equilibrium, state=solved.state,
        fsqr=solved.fsqr, fsqz=solved.fsqz, fsql=solved.fsql,
        niter=solved.iterations, converged=solved.converged,
    ),
)
print(f"  wrote {WOUT.name} ({WOUT.stat().st_size // 1024} KB)")

print(f"\nsolving the kinetics with DKX at rN = {R_N} ...")
run = dkx.run(
    geometryScheme=5, equilibriumFile=str(WOUT),
    inputRadialCoordinate=3, rN_wish=R_N, **KINETICS,
)
for name in ("FSABjHat", "particleFlux_vm_psiHat", "heatFlux_vm_psiHat"):
    print(f"  {name:<26s} = {np.asarray(run.moments[name]).ravel()[0]:+.6e}")

print("\nSame wout works with dkx.plot, and with any other geometryScheme=5 deck.")
