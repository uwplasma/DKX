"""Define a case in Python -> solve -> read the moments.  Start here.

The whole of DKX's basic workflow is three lines: describe the case, call
``dkx.run``, read ``run.moments``.  Nothing is written to disk and no input
file exists -- the parameters below are the same ones an SFINCS
``input.namelist`` would carry, passed as keywords.  For the file-based route
instead, see ``run_from_namelist.py``; to vary one of these parameters, see
``scan_resolution.py``.

Physics: a concentric circular-cross-section tokamak (``geometryScheme=1``
with zero helical ripple, so ``Nzeta=1``), one hydrogen species, pitch-angle
scattering.  In an axisymmetric field neoclassical theory predicts a small
radial particle flux driven by the density and temperature gradients, and a
parallel bootstrap current; both are printed below.

Expected runtime: ~5 s on a laptop CPU, nearly all of it JAX compilation.  The
second run in the same process is milliseconds.

Achieved: FSABjHat = +2.408e-02, particleFlux_vm_psiHat = +1.268e-06 at the
resolution set below.  That resolution is chosen to run in seconds, not to be
converged: ``scan_resolution.py`` shows FSABjHat still moving to +3.630e-02 by
Nxi=40, so the value here is 34% low.  Run the scan before trusting a number.
"""

import os

import dkx

# --------------------------- parameters -------------------------------------
CI = os.environ.get("DKX_CI") == "1"  # shrink for a fast smoke run

TOKAMAK = dict(
    # Geometry: SFINCS's three-helicity analytic model.  epsilon_h = 0 leaves a
    # pure circular tokamak of inverse aspect ratio 0.07.
    geometryScheme=1,
    inputRadialCoordinate=3,  # 3 = pick the surface by rN = r/a
    rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596,

    # Species: one hydrogen ion.  Add an electron by extending every list:
    # Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[1.0, 1.0], ...
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],

    # Resolution.  Nzeta = 1 because the field is axisymmetric.
    Ntheta=9 if CI else 15, Nzeta=1, Nxi=6 if CI else 8, NL=4, Nx=4 if CI else 6,

    # Collisions: 1 = pitch-angle scattering, 0 = full linearized Fokker-Planck.
    # Use 0 for anything whose headline number is the bootstrap current: pitch
    # angle scattering has no momentum-restoring term.
    collisionOperator=1,
    Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
# ----------------------------- end of parameters ----------------------------

run = dkx.run(**TOKAMAK)

# run.moments is a dict keyed by the sfincsOutput.h5 names, so anything the
# Fortran code reports is here under the same key.
print(f"FSABjHat (bootstrap current)  = {float(run.moments['FSABjHat']):+.6e}")
print(f"particleFlux_vm_psiHat        = {run.moments['particleFlux_vm_psiHat']}")
print(f"heatFlux_vm_psiHat            = {run.moments['heatFlux_vm_psiHat']}")
print(f"FSABFlow                      = {run.moments['FSABFlow']}")

# The linear solve is reported too, so a run that did not converge says so
# rather than returning a plausible number.
print(f"residual                      = {float(run.solve_result.residual_norms[-1]):.3e}")
print(f"converged                     = {bool(run.solve_result.converged)}")
print(f"route                         = {run.solve_result.route}")

# To write an sfincsOutput file, pass out=... and the suffix picks the format:
#   dkx.run(**TOKAMAK, out="tokamak.h5")   # or .nc, or .npz
