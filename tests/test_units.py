"""The SFINCS reference set must reproduce the Fortran normalization defaults."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dkx import units
from dkx.constants import DEFAULT_DELTA, DEFAULT_NU_N, RadialCoordinates


def test_reference_set_reproduces_fortran_delta():
    """``Delta = 4.5694e-3`` (globalVariables.F90:133) pins mBar/BBar/RBar."""
    assert units.reference_delta() == pytest.approx(DEFAULT_DELTA, rel=2e-6)


def test_reference_set_reproduces_fortran_nu_n():
    """``nu_n = 8.330e-3`` (globalVariables.F90:135) pins nBar, TBar and lnLambda.

    Together with the Delta check this is what makes the reference set a fact
    rather than an assumption: no other choice satisfies both.
    """
    assert units.reference_nu_n() == pytest.approx(DEFAULT_NU_N, rel=5e-5)


def test_thermal_speed_and_derived_factors():
    assert units.V_BAR == pytest.approx(math.sqrt(2.0 * units.T_BAR / units.M_BAR))
    assert units.CURRENT_DENSITY == pytest.approx(7.0126e6, rel=1e-4)
    assert units.PARTICLE_FLUX == pytest.approx(4.3769e25, rel=1e-4)
    # heatFlux carries nBar mBar vBar^3, which is 2 nBar TBar vBar exactly.
    assert units.HEAT_FLUX == pytest.approx(2.0 * units.N_BAR * units.T_BAR * units.V_BAR)


def test_flux_coordinate_factor_matches_radial_coordinates():
    """``flux_psi_hat_to_r_hat`` is diagnostics.F90:703's ``ddrHat2ddpsiHat``."""
    psi_a_hat, a_hat, r_n = 0.15596, 0.5585, 0.4
    expected = RadialCoordinates(
        psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n
    ).d_dr_hat_to_d_dpsi_hat
    got = units.flux_psi_hat_to_r_hat(psi_a_hat=psi_a_hat, a_hat=a_hat, r_n=r_n)
    assert got == pytest.approx(expected)
    assert got == pytest.approx(a_hat / (2.0 * psi_a_hat * r_n))


def test_root_fsab2_recovers_the_geometry_ratio():
    """``FSABjHat / FSABjHatOverRootFSAB2`` is sqrt(<B^2>), constant over a scan."""
    from dkx.representative import _root_fsab2

    over_root = np.array([-3.0, -1.0, 0.0, 2.0, 5.0])
    scale = 1.234
    moments = {"FSABjHat": over_root * scale, "FSABjHatOverRootFSAB2": over_root}
    assert _root_fsab2(moments) == pytest.approx(scale)


def test_root_fsab2_returns_none_without_the_pair():
    from dkx.representative import _root_fsab2

    assert _root_fsab2({"FSABjHat": np.array([1.0])}) is None
    assert _root_fsab2({"FSABjHat": np.zeros(3),
                        "FSABjHatOverRootFSAB2": np.zeros(3)}) is None  # fmt: skip
