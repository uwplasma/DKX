"""The VMEX-facing kinetic bootstrap objective term."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from dkx.bootstrap import KineticBootstrapCurrent, _polynomial
from dkx.units import PARALLEL_CURRENT


class _Profiles:
    """The three attributes the term reads off ``vmex`` ``KineticProfiles``."""

    ne_coeffs = 3.0e20 * np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    Te_coeffs = 12.0e3 * np.array([1.0, -1.0])
    Ti_coeffs = 10.0e3 * np.array([1.0, -1.0])


def test_polynomial_matches_numpy_and_its_derivative():
    coefficients = np.array([2.0, -3.0, 0.5, 1.25])
    for s in (0.0, 0.15, 0.5, 0.97):
        value, dds = _polynomial(coefficients, s)
        assert value == pytest.approx(np.polyval(coefficients[::-1], s))
        assert dds == pytest.approx(np.polyval(np.polyder(coefficients[::-1]), s))


def test_plasma_at_uses_sfincs_normalization_and_the_rhat_chain_rule():
    """nHat is n/1e20, THat is T/1keV, and the gradients are d/drHat.

    Code 4 wants d/drHat while the profiles are polynomials in s, so the deck
    owes ``d/drHat = (2 sqrt(s) / aHat) d/ds``.  Dropping it understates the
    drive by aHat, which is 0.17 on a compact device.
    """
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.4])
    a_hat = 0.5
    plasma = term.plasma_at(0.4, a_hat=a_hat)
    assert plasma["n_hat"] == pytest.approx(3.0 * (1.0 - 0.4**5))
    assert plasma["te_hat"] == pytest.approx(12.0 * (1.0 - 0.4))
    assert plasma["ti_hat"] == pytest.approx(10.0 * (1.0 - 0.4))
    # Both profiles decrease outward, so both gradients are negative.
    assert all(plasma[k] < 0.0 for k in ("dn_drhat", "dte_drhat", "dti_drhat"))
    assert plasma["dte_drhat"] == pytest.approx(-12.0 * 2.0 * np.sqrt(0.4) / a_hat)


def test_namelist_is_a_valid_sfincs_deck(tmp_path):
    """The generated deck must parse as an input, not merely look like one."""
    from dkx.inputs import sfincs_input_from_raw
    from dkx.namelist import read_sfincs_input

    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.36])
    deck = tmp_path / "input.namelist"
    deck.write_text(term.namelist("/nonexistent/wout.nc", 0.36, er=-1.5, a_hat=0.5))
    parsed = sfincs_input_from_raw(read_sfincs_input(deck))
    assert parsed.geometry.geometry_scheme == 5
    # rN_wish is sqrt(s), because inputRadialCoordinate = 3 is rN.
    assert float(parsed.geometry.r_n_wish) == pytest.approx(0.6)
    assert float(parsed.physics.er) == pytest.approx(-1.5)
    # Coordinate 4 (the v3 default) is the ONLY one that drives the potential
    # with Er; anything else raises "Er != 0 with a non-Er
    # inputRadialCoordinateForGradients", which ambipolar=True would hit.
    assert int(parsed.geometry.input_radial_coordinate_for_gradients) == 4
    assert len(parsed.species.d_n_hat_d_r_hats) == 2
    assert all(value < 0.0 for value in parsed.species.d_n_hat_d_r_hats)


def test_residuals_scale_by_the_reference_current(monkeypatch):
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.2, 0.5, 0.8],
                                   reference_current=2.0e5)  # fmt: skip
    values = np.array([1.0e5, -4.0e5, 0.0])
    monkeypatch.setattr(term, "_evaluate", lambda eq: values)
    assert term.residuals(None) == pytest.approx([0.5, -2.0, 0.0])
    assert term.total(None) == pytest.approx(0.25 + 4.0)
    assert term.J(None) == pytest.approx(term.residuals(None))
    assert term(None) == pytest.approx(term.residuals(None))


@pytest.mark.parametrize("failed_current", [np.nan, np.inf, -np.inf])
def test_unsolvable_surface_remains_diagnostic_and_cannot_promote_zero_objective(
    monkeypatch, failed_current
):
    """A failed solve must not read as "this device has no bootstrap current"."""
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.3, 0.6])
    monkeypatch.setattr(term, "_evaluate", lambda eq: np.array([1.0e5, failed_current]))
    _s, profile = term.current_profile(None)
    assert not np.isfinite(profile[1])
    with pytest.raises(RuntimeError, match="nonfinite"):
        term.residuals(None)
    with pytest.raises(RuntimeError, match="nonfinite"):
        term.J(None)
    with pytest.raises(RuntimeError, match="nonfinite"):
        term.profile(None)
    with pytest.raises(RuntimeError, match="nonfinite"):
        term.total(None)


@pytest.mark.parametrize("reference_current", [0.0, -1.0, np.nan, np.inf])
def test_reference_current_must_be_finite_and_positive(reference_current):
    with pytest.raises(ValueError, match="finite and positive"):
        KineticBootstrapCurrent(_Profiles(), reference_current=reference_current)


def test_nonfinite_normalized_residual_cannot_promote_objective(monkeypatch):
    term = KineticBootstrapCurrent(_Profiles(), reference_current=1.0e-320)
    monkeypatch.setattr(term, "_evaluate", lambda eq: np.array([np.finfo(float).max]))
    with pytest.raises(RuntimeError, match="residuals must be finite"):
        term.total(None)


def test_finite_residuals_cannot_overflow_profile_or_total(monkeypatch):
    term = KineticBootstrapCurrent(_Profiles())
    monkeypatch.setattr(term, "_evaluate", lambda eq: np.array([1.0e308]))
    with pytest.raises(RuntimeError, match="profile must be finite"):
        term.profile(None)
    with pytest.raises(RuntimeError, match="profile must be finite"):
        term.total(None)


def test_ambipolar_rejects_finite_current_from_an_unconverged_scan_point(monkeypatch, tmp_path):
    """Interpolation is inadmissible unless every full-state scan point passed."""
    from dkx import api

    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.4], ambipolar=True,
                                   er_values=[-2.0, 0.0, 2.0])  # fmt: skip
    scan = SimpleNamespace(
        algebraic_converged=np.array([True, False, True]),
        radial_current=np.array([-1.0, 1.0, 2.0]),
        moments={"FSABjHat": np.array([2.0, 3.0, 4.0])},
    )
    captured = {}

    def fake_scan(*args, **kwargs):
        captured.update(kwargs)
        return scan

    monkeypatch.setattr(api, "batched_er_scan", fake_scan)
    monkeypatch.setattr(term, "namelist", lambda *args, **kwargs: "&general\n/\n")
    value = term._one_surface(tmp_path / "wout.nc", 0.4, tmp_path)
    assert captured["retain_full_state"] is True
    assert np.isnan(value)
    monkeypatch.setattr(term, "_evaluate", lambda eq: np.array([value]))
    _s, profile = term.current_profile(None)
    assert np.isnan(profile[0])
    with pytest.raises(RuntimeError, match="nonfinite"):
        term.J(None)


@pytest.mark.parametrize("status", [np.array([1, 0, 1]), np.array([True, np.nan, True])])
def test_ambipolar_requires_boolean_scan_admission_status(monkeypatch, tmp_path, status):
    from dkx import api

    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.4], ambipolar=True,
                                   er_values=[-2.0, 0.0, 2.0])  # fmt: skip
    scan = SimpleNamespace(
        algebraic_converged=status,
        radial_current=np.array([-1.0, 1.0, 2.0]),
        moments={"FSABjHat": np.array([2.0, 3.0, 4.0])},
    )
    monkeypatch.setattr(api, "batched_er_scan", lambda *args, **kwargs: scan)
    monkeypatch.setattr(term, "namelist", lambda *args, **kwargs: "&general\n/\n")
    assert np.isnan(term._one_surface(tmp_path / "wout.nc", 0.4, tmp_path))


@pytest.mark.parametrize(
    ("current", "expected"),
    [
        ([-2.0, -1.0, 1.0, 3.0], 0.0),    # bracketed between the middle pair
        ([1.0, 2.0, 3.0, 4.0], None),     # no crossing at all
        ([0.0, 1.0, 2.0, 3.0], -8.0),     # an exact zero at a scanned point
    ],
)
def test_ion_root_bracketing(current, expected):
    er = np.array([-8.0, -2.0, 2.0, 8.0])
    root = KineticBootstrapCurrent._ion_root(er, np.asarray(current, dtype=float))
    if expected is None:
        assert root is None
    else:
        assert root == pytest.approx(expected)


def test_parallel_current_unit_is_the_vmec_jdotb_unit():
    """The term reports A T/m^2, so it can be compared with the wout ``jdotb``."""
    assert PARALLEL_CURRENT == pytest.approx(7.0126e6, rel=1e-4)


def test_evaluate_is_memoized_per_equilibrium_object(monkeypatch, tmp_path):
    """The reporter's ``total`` must not re-run every solve after ``residuals``."""
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.3, 0.7])
    calls = []

    def fake_surface(path, s, work):
        calls.append(s)
        return 1.0e5 * s

    monkeypatch.setattr(term, "_one_surface", fake_surface)
    monkeypatch.setattr(term, "_wout_path", staticmethod(lambda eq, work: tmp_path / "w.nc"))

    class Equilibrium:
        pass

    first = Equilibrium()
    term.residuals(first)
    term.total(first)
    assert calls == [0.3, 0.7]

    second = Equilibrium()
    term.residuals(second)
    assert calls == [0.3, 0.7, 0.3, 0.7]


def test_the_default_collision_operator_is_fokker_planck():
    """PAS has no momentum-restoring term, and <j.B> IS the momentum moment.

    Measured against Redl on a finite-beta precise-QA equilibrium, pitch-angle
    scattering overestimates the bootstrap current by 35-47% while the full
    linearized Fokker-Planck operator agrees to 2-7%, for under twice the cost.
    A cheaper default here would bias every optimization that uses this term.
    """
    from dkx.bootstrap import DEFAULT_COLLISION_OPERATOR

    assert DEFAULT_COLLISION_OPERATOR == 0
    deck = KineticBootstrapCurrent(_Profiles()).namelist("/w.nc", 0.5, er=0.0, a_hat=1.0)
    assert "collisionOperator = 0" in deck
