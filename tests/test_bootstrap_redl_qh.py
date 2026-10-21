"""Full Fokker-Planck ``<j.B>`` against the Redl fit in the precise quasi-helical field.

Field: the reactor-scale precise-QH equilibrium of Landreman and Paul, Phys. Rev.
Lett. 128, 035001 (2022) (``nfp = 4``, ``iota = -1.24`` in the file's sign
convention; wout from the simsopt test files, MIT).  Zeff = 1, ``Te = Ti``,
``n = 3.5e19 (1 - s^5)`` m^-3, ``T = 9.45 keV (1 - s)``, surface ``s = 0.5``.
Redl et al., Phys. Plasmas 28, 022502 (2021) with the quasi-helical
substitution ``iota -> iota - N``, ``N = -4`` (``helicity_n = -1``, set by the
dominant ``|B|`` mode ``cos(theta + 4 phi)``), is ``vmex.core.bootstrap.j_dot_B_redl``:
``-2.1027e5 A T m^-2``.

Ratio to Redl on the shared office host (JAX 0.12.0, 4 pinned cores), s = 0.5,
and SFINCS v3 on one rank (MUMPS) at the same grids:

    25x25 angles, Nxi=48, Nx=5   DKX 1.119   SFINCS 1.120   (this test)
    31x31 angles, Nxi=64, Nx=6   DKX 1.069   SFINCS 1.070

Angles beyond 31x31 and Nxi beyond 64 change it by under 1%.  Nxi = 32 gives
0.94 at the converged angles: it is under-resolved, not in agreement.
"""

from pathlib import Path

import numpy as np
import pytest

import dkx
from dkx.units import PARALLEL_CURRENT

WOUT = (
    Path(__file__).parent / "ref" / "wout_LandremanPaul2021_QH_reactorScale_lowres.nc"
)
REDL = -2.10269e5  # A T m^-2
SFINCS_FSABJ = -3.35709e-2  # SFINCS v3 FSABjHat at this grid


def test_full_fokker_planck_matches_sfincs_and_redl_in_a_quasi_helical_field() -> None:
    n_hat, t_hat = 3.5e-1 * (1 - 0.5**5), 9.45 * 0.5
    run = dkx.run(
        equilibriumFile=str(WOUT), geometryScheme=5,
        VMECRadialOption=0, inputRadialCoordinate=1, inputRadialCoordinateForGradients=1, psiN_wish=0.5,
        Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[n_hat] * 2, THats=[t_hat] * 2,
        dNHatdpsiNs=[-5 * 3.5e-1 * 0.5**4] * 2, dTHatdpsiNs=[-9.45] * 2, Delta=4.5694e-3, alpha=1.0,
        nu_n=8.330e-3, Er=0.0, collisionOperator=0, Ntheta=25, Nzeta=25, Nxi=48, NL=4, Nx=5, emit=None,
    )  # fmt: skip
    f = float(np.asarray(run.moments["FSABjHat"]).reshape(()))
    assert f / SFINCS_FSABJ == pytest.approx(1.0, abs=0.01)
    # Redl is a tokamak fit; 7% above it at the converged grid, 12% at this one.
    assert f * PARALLEL_CURRENT / REDL == pytest.approx(1.12, abs=0.02)
