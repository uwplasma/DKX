"""Full Fokker-Planck ``<j.B>`` against the Redl fit in a quasi-symmetric field.

Redl et al., Phys. Plasmas 28, 022502 (2021) is a fit to tokamak solutions, so a
large-aspect-ratio circular tokamak is the one field where it is an exact
reference.  Saxena et al. (arXiv:2507.05166) find it within 1-2% of SFINCS for
quasi-symmetric fields; this pins DKX to 3%.  Pitch-angle scattering lacks
momentum restoration and overestimates the current by about 15%, so only
Fokker-Planck is a kinetic reference.

Field: ``geometryScheme = 1``, ``B = B0 (1 + 0.1 cos(theta))``, ``iota = 0.4``,
``G = 3.7 T m``, ``I = 0``, ``psi_a = 0.06845 T m^2`` (surface ``rN = 1``,
``a = 0.37 m``), Zeff = 1, ``Te = Ti = 2 keV`` and ``n = 1e19`` or ``1e20 m^-3`` at
the surface, with ``dn/dpsiN = -n/2`` and ``dT/dpsiN = -1 keV``.  The Redl values
are ``vmex.core.bootstrap.j_dot_B_redl`` with ``epsilon = 0.1``,
``f_t = 0.4280`` and ``R = G <1/B> = 3.7186 m`` from that ``B(theta)``,
helicity 0, ``psi_edge = psi_a``.

Ratio to Redl on the shared office host (JAX 0.12.0, 4 pinned cores, load 20-30):
1.021 at this grid and 1.015 at ``Ntheta = 41`` for ``n = 1e19``; 1.018 at
``Ntheta = 41`` for ``n = 1e20``.
"""

import numpy as np
import pytest

import dkx
from dkx.units import PARALLEL_CURRENT

#: (surface density in m^-3, Redl <j.B> in A T m^-2)
CASES = [(1.0e19, 3.02253e5), (1.0e20, 2.33186e6)]


@pytest.mark.parametrize(("density", "redl"), CASES)
def test_full_fokker_planck_matches_redl_in_a_tokamak(density: float, redl: float) -> None:
    n_hat = density / 1e20
    run = dkx.run(
        geometryScheme=1, epsilon_t=0.1, epsilon_h=0.0, epsilon_antisymm=0.0, iota=0.4, GHat=3.7,
        IHat=0.0, B0OverBBar=1.0, psiAHat=0.06845, aHat=0.37, inputRadialCoordinate=3, rN_wish=1.0,
        inputRadialCoordinateForGradients=1, Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4],
        nHats=[n_hat, n_hat], THats=[2.0, 2.0], dNHatdpsiNs=[-n_hat / 2] * 2, dTHatdpsiNs=[-1.0] * 2,
        Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0, collisionOperator=0,
        Ntheta=25, Nzeta=1, Nx=7, Nxi=48, NL=4, emit=None,
    )  # fmt: skip
    j = float(np.asarray(run.moments["FSABjHat"]).reshape(())) * PARALLEL_CURRENT
    assert j / redl == pytest.approx(1.0, abs=0.03), f"<j.B> = {j:.4e}, Redl {redl:.4e}"
