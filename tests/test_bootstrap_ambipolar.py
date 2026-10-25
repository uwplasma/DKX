"""The ambipolar root inside the traced bootstrap row (``KineticBootstrapMismatch(ambipolar=True)``).

The Newton iteration is checked on analytic radial-current curves through the
real ``operator_at_er`` plumbing, so it needs neither VMEX nor a drift-kinetic
solve; the row against VMEX's implicit Jacobian is in ``test_vmex_kinetic_gradient.py``.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from dkx.bootstrap import KineticBootstrapMismatch
from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import SfincsInput
from dkx.namelist import parse_sfincs_input_text

jax.config.update("jax_enable_x64", True)


class Profiles:
    ne_coeffs, Te_coeffs, Ti_coeffs = [1e19], [1e3], [1e3]


@pytest.fixture(scope="module")
def operator():
    deck = SfincsInput.from_params(
        geometryScheme=1, psiAHat=1.0, aHat=0.3, inputRadialCoordinate=1,
        inputRadialCoordinateForGradients=4, psiN_wish=0.5, Zs=[1.0, -1.0],
        mHats=[1.0, 5.446170214e-4], nHats=[0.5, 0.5], THats=[1.0, 1.2],
        dNHatdrHats=[-0.3, -0.3], dTHatdrHats=[-1.0, -1.1], collisionOperator=1,
        Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=1.0, Ntheta=5, Nzeta=5, Nxi=6, NL=4, Nx=3,
    )  # fmt: skip
    return kinetic_operator_from_namelist(parse_sfincs_input_text(deck.to_namelist()))


def _newton(operator, curve, er0, **options):
    term = KineticBootstrapMismatch(Profiles(), ambipolar=True, **options)
    unit = operator.dphi_hat_dpsi_hat_kinetic  # dPhi/dpsi at E_r = 1 kV/m

    def evaluate(op):  # (J_r, <j.B>) as functions of E_r = dPhi/dpsi / unit
        e = op.dphi_hat_dpsi_hat_kinetic / unit
        return curve(e), 2.0 * e

    return [np.asarray(v) for v in jax.jit(lambda e0: term._newton(evaluate, operator, e0))(er0)]


def test_newton_stays_on_the_branch_it_starts_on_and_returns_the_slopes(operator) -> None:
    """Three roots at -3, 1 and 4: the warm start picks the root, and the slopes belong to it."""
    cubic = lambda e: (e + 3.0) * (e - 1.0) * (e - 4.0) / 10.0  # noqa: E731
    for start, root in ((-2.0, -3.0), (0.5, 1.0), (3.0, 4.0)):
        e, d_r, d_b, converged = _newton(operator, cubic, np.float64(start), er_tol=1e-6)
        assert converged and abs(e - root) < 1e-6, (start, e)
        assert abs(d_r - (3 * root**2 - 4 * root - 11) / 10.0) < 1e-6 and abs(d_b - 2.0) < 1e-9


def test_newton_reports_a_missing_root_and_caps_its_step(operator) -> None:
    e, _, _, converged = _newton(operator, lambda e: 1.0 + e**2, np.float64(3.0), er_max_step=0.5)
    assert not converged and abs(e - 3.0) <= 16 * 0.5  # no root: no convergence, bounded drift
    e, *_ = _newton(operator, lambda e: e - 10.0, np.float64(0.0), er_max_step=2.0, er_tol=1e-6)
    assert abs(e - 10.0) < 1e-6  # a linear curve is solved in steps of at most 2 kV/m


def test_the_ambipolar_row_has_its_own_default_grid() -> None:
    """Nx = 4 is 30% off at E_r != 0 (measured on the QI seed); the default keeps Nx = 5."""
    assert KineticBootstrapMismatch(Profiles(), ambipolar=True).resolution["Nx"] == 5
    assert KineticBootstrapMismatch(Profiles()).resolution["Nx"] == 4
    assert KineticBootstrapMismatch(Profiles(), ambipolar=True, resolution={"Nx": 3}).resolution == {"Nx": 3}
