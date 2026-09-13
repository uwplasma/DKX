"""The full-trajectory ``E_*`` of Landreman et al. (2014) in SFINCS hat units.

``E_* = c G/(iota v_s B0) dPhi0/dpsi`` with ``v_s = sqrt(2 T_s/m_s)`` (Phys. Plasmas
21, 042503, eq. EStar).  The hat-unit form must reproduce the physical definition,
follow the E x B coefficient the operator actually uses, and stay differentiable.
"""

from __future__ import annotations

import math
from pathlib import Path

import jax
import numpy as np
import pytest

from dkx.drift_kinetic import kinetic_operator_build_from_namelist
from dkx.namelist import parse_sfincs_input_text
from dkx.validity import (
    E_STAR_TRAJECTORY_AGREEMENT,
    normalized_radial_electric_field,
    normalized_radial_electric_field_of,
)

REF = Path(__file__).parent / "ref"
ELEMENTARY_CHARGE = 1.602176634e-19  # C
PROTON_MASS = 1.67262192369e-27  # kg


def test_hat_units_reproduce_the_physical_definition() -> None:
    # SI reference scales; in SI the definition carries no factor c and
    # Delta = mBar vBar/(e BBar RBar).
    b_bar, r_bar, t_bar, phi_bar, m_bar = 1.0, 1.0, 1.0e3 * ELEMENTARY_CHARGE, 1.0e3, PROTON_MASS
    v_bar = math.sqrt(2.0 * t_bar / m_bar)
    delta = m_bar * v_bar / (ELEMENTARY_CHARGE * b_bar * r_bar)
    alpha = ELEMENTARY_CHARGE * phi_bar / t_bar
    # A cold-ion, HSX-like surface.
    g_hat, iota, b0_hat, t_hat, m_hat, dphi_hat = 1.2, 1.05, 1.0, 0.059, 1.0, -3.0

    g = g_hat * b_bar * r_bar
    dphi_dpsi = phi_bar * dphi_hat / (b_bar * r_bar**2)
    v_s = math.sqrt(2.0 * t_hat * t_bar / (m_hat * m_bar))
    physical = g * dphi_dpsi / (iota * v_s * b0_hat * b_bar)

    hat = normalized_radial_electric_field(
        alpha=alpha, delta=delta, g_hat=g_hat, iota=iota, b0_over_bbar=b0_hat,
        dphi_hat_dpsi_hat=dphi_hat, t_hat=t_hat, m_hat=m_hat,
    )  # fmt: skip
    assert float(hat) == pytest.approx(physical, rel=1e-12)


def test_species_scale_with_their_thermal_speed() -> None:
    common = dict(alpha=1.0, delta=4.57e-3, g_hat=1.2, iota=1.05, b0_over_bbar=1.0, dphi_hat_dpsi_hat=-3.0)
    ion_electron = normalized_radial_electric_field(
        **common, t_hat=np.array([0.059, 1.37]), m_hat=np.array([1.0, 5.4465e-4])
    )
    ratio = float(ion_electron[1] / ion_electron[0])
    assert ratio == pytest.approx(math.sqrt(5.4465e-4 * 0.059 / 1.37), rel=1e-12)
    assert 0.0 < E_STAR_TRAJECTORY_AGREEMENT == pytest.approx(1.0 / 3.0)


def test_operator_build_uses_the_kinetic_exb_coefficient_and_its_sign() -> None:
    text = (REF / "er_xdot_1species_tiny.input.namelist").read_text()
    assert "Er = 0.5d+0" in text
    builds = [
        kinetic_operator_build_from_namelist(parse_sfincs_input_text(text.replace("Er = 0.5d+0", f"Er = {er}")))
        for er in ("0.5d+0", "-0.5d+0")
    ]
    values = [np.asarray(normalized_radial_electric_field_of(b.operator, b.geometry)) for b in builds]

    op, geometry = builds[0].operator, builds[0].geometry
    expected = (
        0.5 * float(op.alpha) * float(op.delta) * float(geometry.g_hat)
        * float(op.dphi_hat_dpsi_hat_kinetic)
        / (float(geometry.iota) * float(geometry.b0_over_bbar))
        / np.sqrt(np.asarray(op.t_hat) / np.asarray(op.m_hat))
    )  # fmt: skip
    assert np.all(np.isfinite(values[0])) and np.all(values[0] != 0.0)
    np.testing.assert_allclose(values[0], expected, rtol=1e-12)
    np.testing.assert_allclose(values[1], -values[0], rtol=1e-12)


def test_the_field_is_differentiable_in_the_potential_gradient() -> None:
    args = dict(alpha=1.0, delta=4.57e-3, g_hat=1.2, iota=1.05, b0_over_bbar=1.0, t_hat=0.059, m_hat=1.0)
    value = float(normalized_radial_electric_field(**args, dphi_hat_dpsi_hat=-3.0))
    slope = jax.grad(lambda d: normalized_radial_electric_field(**args, dphi_hat_dpsi_hat=d))(-3.0)
    assert float(slope) == pytest.approx(value / -3.0, rel=1e-12)
