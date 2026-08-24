"""Define a case in Python -> solve -> read the moments.  Start here.

The basic workflow is three lines: describe the case, call ``dkx.run``, read
``run.moments``.  Nothing touches disk -- these are the parameters an SFINCS
``input.namelist`` carries, passed as keywords.  For the file route see
``run_from_namelist.py``; to vary a parameter see ``scan_resolution.py``.

Physics: concentric circular tokamak (``geometryScheme=1``, no helical ripple
so ``Nzeta=1``), one hydrogen species, pitch-angle scattering.  Both the
gradient-driven radial particle flux and the bootstrap current are printed.

Expected runtime: ~5 s on a laptop CPU, nearly all JAX compilation; a second
run in the same process is milliseconds.

Achieved: FSABjHat = +2.408e-02, particleFlux_vm_psiHat = +1.268e-06.  That
resolution runs in seconds but is *not* converged -- ``scan_resolution.py``
takes FSABjHat to +3.630e-02, so this is 34% low.  Scan before trusting it.
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
