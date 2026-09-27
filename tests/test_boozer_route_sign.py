"""The Boozer route and the VMEC-file route give one bootstrap current, with one sign.

A VMEX equilibrium reaches DKX two ways.  The VMEC-file route
(``geometryScheme = 5``) reads the wout and keeps VMEC's Jacobian, which is
negative for ``signgs = -1``.  The Boozer route, used by optimization because
it is traceable, takes a ``booz_xform_jax`` spectrum of the same surface and
builds ``D = B^2/(G + iota I)``, which is positive there.  Handedness is
converted in one place, :func:`boozer_route_psi_a_hat`; without it every flux
and ``FSABjHat`` on the Boozer route has the opposite sign.

The fixture is a VMEX solve of ``examples/data/input.minimal_seed_nfp2`` (with
the rotating-ellipse seed perturbation 0.05) at ``ns = 13`` and the
``booz_xform_jax`` spectrum of its half-mesh row 6, with the analytic Redl
``<j.B>`` for the same profiles recorded beside it, so this runs without
either optional package.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import dkx
from dkx.solve import solve
from dkx.run import profile_moments_from_operator
from dkx.units import PARALLEL_CURRENT
from dkx.workflows.geometry_adapters import (
    boozer_route_psi_a_hat,
    kinetic_operator_on_boozer_surface,
)

REF = Path(__file__).resolve().parent / "ref"
FIXTURE = json.loads((REF / "boozer_route_sign_vmex_seed_nfp2.json").read_text())
S = float(FIXTURE["s"])
N0, T0 = 1.0e19, 2.0e3  # m^-3, eV: n = N0 (1 - s^5), Te = Ti = T0 (1 - s)
N_HAT, T_HAT = N0 / 1e20 * (1 - S**5), T0 / 1e3 * (1 - S)
DN_DPSIN, DT_DPSIN = -5 * N0 / 1e20 * S**4, -T0 / 1e3
PLASMA = dict(
    inputRadialCoordinate=1, inputRadialCoordinateForGradients=1, psiN_wish=S,
    Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[N_HAT, N_HAT], THats=[T_HAT, T_HAT],
    dNHatdpsiNs=[DN_DPSIN, DN_DPSIN], dTHatdpsiNs=[DT_DPSIN, DT_DPSIN],
    Ntheta=11, Nzeta=11, Nxi=16, NL=4, Nx=4,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0,
)  # fmt: skip


def _jdotb(operator) -> float:
    """``<j.B>`` in A T/m^2 from the exact structured solve of every Legendre block."""
    rhs = operator.rhs()
    result = solve(operator, rhs, method="auto", tol=1e-10,
                   tier1_keep_lowest=operator.n_xi, emit=None)  # fmt: skip
    x = result.x.reshape(-1)
    residual = float(np.linalg.norm(operator.apply(x) - rhs.reshape(-1)) / np.linalg.norm(rhs))
    assert residual < 1e-8, f"original-equation residual {residual:.2e}"
    moments = profile_moments_from_operator(operator, x)
    return float(np.asarray(moments["FSABjHat"]).reshape(())) * PARALLEL_CURRENT


def _boozer_route(psi_a_hat: float) -> float:
    template = dkx.run(geometryScheme=1, psiAHat=psi_a_hat, aHat=0.1, **PLASMA,
                       emit=None).operator  # fmt: skip
    operator = kinetic_operator_on_boozer_surface(
        template, bmnc_b=np.asarray(FIXTURE["bmnc_b"]), ixm_b=FIXTURE["ixm_b"],
        ixn_b=FIXTURE["ixn_b"], nfp=FIXTURE["nfp"], iota=FIXTURE["iota_b"],
        g_hat=FIXTURE["bvco_b"], i_hat=FIXTURE["buco_b"],
    )  # fmt: skip
    return _jdotb(operator)


@pytest.fixture(scope="module")
def vmec_route() -> float:
    wout = REF / FIXTURE["wout"]
    operator = dkx.run(geometryScheme=5, equilibriumFile=str(wout), VMECRadialOption=0,
                       **PLASMA, emit=None).operator  # fmt: skip
    return _jdotb(operator)


def test_the_convention_is_signgs_times_the_sign_of_g_plus_iota_i() -> None:
    phi_edge = FIXTURE["phi_edge"]
    assert FIXTURE["signgs"] == -1
    assert boozer_route_psi_a_hat(phi_edge, -1, 1.4) == pytest.approx(-phi_edge / (2 * np.pi))
    assert boozer_route_psi_a_hat(phi_edge, 1, 1.4) == pytest.approx(phi_edge / (2 * np.pi))
    assert boozer_route_psi_a_hat(-phi_edge, -1, -1.4) == pytest.approx(phi_edge / (2 * np.pi))


def test_boozer_route_matches_the_vmec_route_and_redl_in_sign(vmec_route: float) -> None:
    g_plus_iota_i = FIXTURE["bvco_b"] + FIXTURE["iota_b"] * FIXTURE["buco_b"]
    psi = boozer_route_psi_a_hat(FIXTURE["phi_edge"], FIXTURE["signgs"], g_plus_iota_i)
    boozer = _boozer_route(psi)
    redl = FIXTURE["redl_jdotB_A_T_per_m2"]
    assert redl > 0.0
    assert vmec_route > 0.0, "the VMEC-file route must agree with Redl in sign"
    assert boozer > 0.0, "the Boozer route must agree with the VMEC-file route in sign"
    # Two representations of one surface on an 11x11 angular grid.  Measured:
    # Boozer/VMEC = 1.17 here, 1.08-1.10 at 15x15 and 19x19 on the ns = 31
    # solve of the same seed.  Coarser than Nx = 4, Nxi = 16 the VMEC route
    # itself changes sign (-2.1e6 at Nxi = 10, Nx = 3), so this grid is the floor.
    assert abs(boozer / vmec_route - 1.0) < 0.3


def test_skipping_the_conversion_flips_the_current(vmec_route: float) -> None:
    naive = _boozer_route(abs(FIXTURE["phi_edge"]) / (2 * np.pi))
    assert naive < 0.0 < vmec_route
