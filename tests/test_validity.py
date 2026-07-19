"""Local-validity diagnostic tests (analytic limits + a tiny regime scan).

These are CI-fast: they exercise :mod:`dkx.validity` against closed-form
scalings and the standard neoclassical regime boundaries, and check that the
classifier's predicted ``1/nu -> sqrt(nu)`` crossover lines up with the E x B
parameter on the W7-X standard surface.  The W7-X flux-surface constants used
below are the measured values of the ``geometryScheme = 11``
``w7x_standardConfig.bc`` surface at ``r/a = 0.5`` (delta = 4.5694e-3,
iota = -0.87185, GHat = -16.2, B0OverBBar = 2.7989, eps_t = 0.045306,
nu_star = 0.9589 * nuPrime), the same surface the merged Shaing-Callen
convergence benchmark (``examples/paper_benchmarks/shaing_callen_convergence.py``)
scans; there the small finite ``EStar = 3e-3`` curve is documented to detach
from the ``EStar = 0`` curve below ``nuPrime ~ 1e-3`` as the E x B precession
turns on, which is exactly the marginal band the classifier flags here.  No
drift-kinetic solve or equilibrium file is needed.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import jax

from dkx.validity import (
    DEFAULT_THRESHOLDS,
    LocalValidityReport,
    Regime,
    ValidityFlag,
    banana_orbit_width_hat,
    classify_collisionality_regime,
    drift_resonance_ratio,
    exb_collision_ratio,
    finite_orbit_width_parameter,
    local_validity_report,
    thermal_gyroradius_hat,
)

# Measured W7-X standard-configuration surface (r/a = 0.5), hat units.
W7X = dict(delta=4.5694e-3, iota=-0.87185, g_hat=-16.2, b0_over_bbar=2.798916, eps_t=0.045306)
NU_STAR_PER_NU_PRIME = 0.9589123  # nu_star = 0.9589 * nuPrime on this surface


def _w7x_nu_star(nu_prime: float) -> float:
    return NU_STAR_PER_NU_PRIME * nu_prime


# =============================================================================
# Closed-form scalings of the orbit-width primitives
# =============================================================================


def test_thermal_gyroradius_matches_closed_form() -> None:
    """rho_hat = Delta x sqrt(mHat THat)/(|Z| B0), linear in Delta and 1/|Z|."""
    rho = float(
        thermal_gyroradius_hat(delta=4.5694e-3, z=1.0, m_hat=1.0, t_hat=1.0, b0_over_bbar=2.798916)
    )
    assert rho == pytest.approx(4.5694e-3 / 2.798916, rel=1e-12)
    # A Z = 6 impurity at twice the mass has rho scaled by sqrt(2)/6.
    rho_imp = float(
        thermal_gyroradius_hat(delta=4.5694e-3, z=6.0, m_hat=2.0, t_hat=1.0, b0_over_bbar=2.798916)
    )
    assert rho_imp / rho == pytest.approx(math.sqrt(2.0) / 6.0, rel=1e-12)


def test_banana_width_has_rho_q_over_sqrt_eps_scaling() -> None:
    """w_b = rho/(|iota| sqrt(eps_t)); doubling eps_t shrinks it by sqrt(2)."""
    w_ref = float(banana_orbit_width_hat(rho_hat=1.0e-3, iota=-0.9, eps_t=0.05))
    assert w_ref == pytest.approx(1.0e-3 / (0.9 * math.sqrt(0.05)), rel=1e-12)
    w_eps2 = float(banana_orbit_width_hat(rho_hat=1.0e-3, iota=-0.9, eps_t=0.10))
    assert w_ref / w_eps2 == pytest.approx(math.sqrt(2.0), rel=1e-12)
    # The banana width exceeds the gyroradius by q/sqrt(eps) >> 1.
    assert w_ref > 1.0e-3


def test_finite_orbit_width_parameter_is_ratio() -> None:
    assert float(finite_orbit_width_parameter(orbit_width_hat=0.02, grad_scale_length_hat=0.2)) == (
        pytest.approx(0.1, rel=1e-12)
    )


# =============================================================================
# The E x B / drift ratios and the sqrt(nu) boundary
# =============================================================================


def test_exb_collision_ratio_closed_form_and_boundary() -> None:
    """k_ExB = v_E/(|iota| nu_star); k_ExB = 1 is the nu_star ~ eps_t EStar crossing."""
    v_e, iota, nu_star = 1.2e-4, -0.87, 9.6e-4
    k = float(exb_collision_ratio(v_e=v_e, iota=iota, nu_star=nu_star))
    assert k == pytest.approx(v_e / (0.87 * nu_star), rel=1e-12)

    # v_E = |EStar| |iota| eps_t / x0, so k_ExB = 1 <=> nu_star = eps_t |EStar| / x0.
    eps_t, e_star = 0.045306, 3.0e-3
    v_e_from_estar = abs(e_star) * abs(iota) * eps_t
    nu_star_boundary = eps_t * abs(e_star)  # x0 = 1
    k_boundary = float(exb_collision_ratio(v_e=v_e_from_estar, iota=iota, nu_star=nu_star_boundary))
    assert k_boundary == pytest.approx(1.0, rel=1e-12)


def test_drift_resonance_ratio_closed_form() -> None:
    k = float(drift_resonance_ratio(v_e=1.2e-4, g_hat=-16.2, delta=4.5694e-3))
    assert k == pytest.approx(1.2e-4 * 16.2 / 4.5694e-3, rel=1e-12)


# =============================================================================
# Classifier boundaries (analytic)
# =============================================================================


def test_collisionality_boundaries_at_zero_field() -> None:
    """PS / plateau / long-mfp split exactly at nu_star = 1 and nu_star = eps_t**1.5."""
    eps_t = 0.05
    nu_pb = eps_t**1.5
    # Just inside each band (E_r = 0 -> collisional long-mfp branch is one-over-nu).
    assert classify_collisionality_regime(nu_star=2.0, eps_t=eps_t) is Regime.PFIRSCH_SCHLUETER
    assert classify_collisionality_regime(nu_star=0.5, eps_t=eps_t) is Regime.PLATEAU
    assert classify_collisionality_regime(nu_star=1.5 * nu_pb, eps_t=eps_t) is Regime.PLATEAU
    assert classify_collisionality_regime(nu_star=0.5 * nu_pb, eps_t=eps_t) is Regime.ONE_OVER_NU


def test_banana_vs_one_over_nu_is_selected_by_effective_ripple() -> None:
    """A (quasi)symmetric field (tiny eps_eff) gives banana, not 1/nu."""
    eps_t = 0.15
    low_nu = 0.1 * eps_t**1.5
    assert classify_collisionality_regime(nu_star=low_nu, eps_t=eps_t, epsilon_eff=1e-6) is (
        Regime.BANANA
    )
    assert classify_collisionality_regime(nu_star=low_nu, eps_t=eps_t, epsilon_eff=2e-2) is (
        Regime.ONE_OVER_NU
    )
    # No eps_eff supplied -> assume a general stellarator (1/nu).
    assert classify_collisionality_regime(nu_star=low_nu, eps_t=eps_t) is Regime.ONE_OVER_NU


def test_exb_branches_sqrt_nu_and_superbanana_plateau() -> None:
    """At low collisionality the E x B ratios select sqrt-nu vs the resonance."""
    eps_t = 0.05
    low_nu = 0.1 * eps_t**1.5
    # E x B beats collisions but off the drift resonance -> sqrt-nu.
    assert classify_collisionality_regime(
        nu_star=low_nu, k_exb=5.0, k_res=0.1, eps_t=eps_t
    ) is Regime.SQRT_NU
    # E x B beats collisions and near the drift resonance -> superbanana-plateau.
    assert classify_collisionality_regime(
        nu_star=low_nu, k_exb=5.0, k_res=1.2, eps_t=eps_t
    ) is Regime.SUPERBANANA_PLATEAU
    # Below the sqrt-nu boundary the field is still negligible -> one-over-nu.
    assert classify_collisionality_regime(
        nu_star=low_nu, k_exb=0.2, k_res=1.2, eps_t=eps_t
    ) is Regime.ONE_OVER_NU


# =============================================================================
# A tiny regime scan on the W7-X surface (no solve)
# =============================================================================


def test_w7x_zero_field_scan_transitions_plateau_to_one_over_nu() -> None:
    """EStar = 0: plateau above nuPrime ~ 0.01, one-over-nu below (nu_star = eps_t**1.5)."""
    nu_pb = W7X["eps_t"] ** 1.5
    nu_prime_boundary = nu_pb / NU_STAR_PER_NU_PRIME
    assert nu_prime_boundary == pytest.approx(0.01, abs=3e-3)  # ~1.0e-2

    regimes = {
        nu_prime: classify_collisionality_regime(
            nu_star=_w7x_nu_star(nu_prime), eps_t=W7X["eps_t"]
        )
        for nu_prime in (1.0, 0.1, 3e-2, 3e-3, 3e-4)
    }
    assert regimes[1.0] is Regime.PLATEAU
    assert regimes[3e-2] is Regime.PLATEAU
    assert regimes[3e-3] is Regime.ONE_OVER_NU
    assert regimes[3e-4] is Regime.ONE_OVER_NU


def test_w7x_finite_field_sqrt_nu_onset_matches_exb_parameter() -> None:
    """The EStar = 3e-3 detachment (documented below nuPrime ~ 1e-3) is the k_ExB band.

    The Shaing-Callen convergence benchmark documents that the EStar = 3e-3
    curve detaches from the EStar = 0 curve below nuPrime ~ 1e-3 as the E x B
    precession turns on.  The surrogate verdict must therefore stay PASS in the
    deep 1/nu range and move to MARGINAL/FAIL exactly where k_ExB enters its
    marginal band, and the k_ExB = 1 crossing must sit near nuPrime ~ 1e-4.
    """
    reports = {
        nu_prime: local_validity_report(
            nu_star=_w7x_nu_star(nu_prime), e_star=3e-3, grad_scale_length_hat=0.2622, **W7X
        )
        for nu_prime in (1e-2, 1e-3, 3e-4)
    }
    # Deep 1/nu, E x B still negligible.
    assert reports[1e-2].one_over_nu_surrogate_flag is ValidityFlag.PASS
    assert reports[1e-2].k_exb < DEFAULT_THRESHOLDS.k_exb_marginal
    # By nuPrime = 3e-4 the E x B parameter has entered the marginal band.
    assert reports[3e-4].k_exb >= DEFAULT_THRESHOLDS.k_exb_marginal
    assert reports[3e-4].one_over_nu_surrogate_flag is ValidityFlag.MARGINAL
    # k_ExB grows monotonically as collisionality drops.
    assert reports[1e-2].k_exb < reports[1e-3].k_exb < reports[3e-4].k_exb

    # The hard sqrt-nu crossing (k_ExB = 1) is near nuPrime ~ 1.4e-4.
    v_e = 3e-3 * abs(W7X["iota"]) * W7X["eps_t"]
    nu_star_cross = v_e / abs(W7X["iota"])  # k_ExB = 1
    nu_prime_cross = nu_star_cross / NU_STAR_PER_NU_PRIME
    assert 5e-5 < nu_prime_cross < 5e-4


def test_w7x_stronger_field_reaches_superbanana_plateau() -> None:
    """A larger EStar drives the reference particle to the drift resonance."""
    report = local_validity_report(
        nu_star=_w7x_nu_star(3e-4), e_star=1e-2, grad_scale_length_hat=0.2622, **W7X
    )
    assert report.k_exb > 1.0
    assert (1.0 / DEFAULT_THRESHOLDS.k_res_window) <= report.k_res <= DEFAULT_THRESHOLDS.k_res_window
    assert report.regime is Regime.SUPERBANANA_PLATEAU
    assert report.one_over_nu_surrogate_flag is ValidityFlag.FAIL


# =============================================================================
# Report assembly, flags, and radial locality
# =============================================================================


def test_report_surrogate_flag_by_regime() -> None:
    """The 1/nu surrogate passes only in the collisionless 1/nu (banana) branch."""
    common = dict(grad_scale_length_hat=0.2622, **W7X)
    one_over_nu = local_validity_report(nu_star=_w7x_nu_star(3e-3), e_star=0.0, **common)
    plateau = local_validity_report(nu_star=_w7x_nu_star(0.3), e_star=0.0, **common)
    assert one_over_nu.regime is Regime.ONE_OVER_NU
    assert one_over_nu.one_over_nu_surrogate_flag is ValidityFlag.PASS
    assert plateau.regime is Regime.PLATEAU
    assert plateau.one_over_nu_surrogate_flag is ValidityFlag.FAIL


def test_radial_locality_flag_thresholds() -> None:
    """delta_FOW = w_b/L crosses pass -> marginal -> fail as L shrinks."""
    common = dict(nu_star=_w7x_nu_star(3e-3), e_star=0.0, **W7X)
    big_l = local_validity_report(grad_scale_length_hat=0.2622, **common)
    assert big_l.radial_locality_flag is ValidityFlag.PASS  # delta_FOW ~ 0.034
    # Shrink L so w_b/L lands in each band.
    w_b = big_l.orbit_width_hat
    marginal = local_validity_report(grad_scale_length_hat=w_b / 0.2, **common)
    fail = local_validity_report(grad_scale_length_hat=w_b / 0.5, **common)
    assert marginal.radial_locality_flag is ValidityFlag.MARGINAL
    assert fail.radial_locality_flag is ValidityFlag.FAIL
    # The overall verdict is the worst of the two channels.
    assert fail.overall_flag is ValidityFlag.FAIL


def test_report_is_structured_and_carries_notes() -> None:
    report = local_validity_report(
        nu_star=_w7x_nu_star(3e-3), e_star=3e-3, grad_scale_length_hat=0.2622, **W7X
    )
    assert isinstance(report, LocalValidityReport)
    assert report.notes  # human-readable reasons attached
    assert report.v_e == pytest.approx(3e-3 * abs(W7X["iota"]) * W7X["eps_t"], rel=1e-12)


# =============================================================================
# Differentiability of the scalar ratios
# =============================================================================


def test_ratios_are_differentiable() -> None:
    """jax.grad flows through the E x B ratio and the orbit-width chain."""
    # d k_ExB / d nu_star = -v_E/(|iota| nu_star^2).
    v_e, iota, nu_star = 1.2e-4, -0.87, 9.6e-4
    grad = jax.grad(lambda ns: exb_collision_ratio(v_e=v_e, iota=iota, nu_star=ns))(nu_star)
    analytic = -v_e / (0.87 * nu_star**2)
    assert float(grad) == pytest.approx(analytic, rel=1e-6)
    assert np.isfinite(float(grad))

    # d w_b / d eps_t = -rho/(2 |iota| eps_t^{3/2}) through the width primitive.
    rho, eps_t = 1.0e-3, 0.05
    dwb = jax.grad(lambda e: banana_orbit_width_hat(rho_hat=rho, iota=iota, eps_t=e))(eps_t)
    analytic_wb = -rho / (2.0 * 0.87 * eps_t**1.5)
    assert float(dwb) == pytest.approx(analytic_wb, rel=1e-6)
