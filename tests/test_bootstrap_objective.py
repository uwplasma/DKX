"""The VMEX-facing kinetic bootstrap objective term."""

from __future__ import annotations

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


def test_plasma_at_uses_sfincs_normalization():
    """nHat is n/1e20 and THat is T/1keV, and the gradients are d/ds."""
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.4])
    plasma = term.plasma_at(0.4)
    assert plasma["n_hat"] == pytest.approx(3.0 * (1.0 - 0.4**5))
    assert plasma["te_hat"] == pytest.approx(12.0 * (1.0 - 0.4))
    assert plasma["ti_hat"] == pytest.approx(10.0 * (1.0 - 0.4))
    # Both profiles decrease outward, so both gradients are negative.
    assert plasma["dn_ds"] < 0.0 and plasma["dte_ds"] < 0.0 and plasma["dti_ds"] < 0.0
    assert plasma["dte_ds"] == pytest.approx(-12.0)


def test_namelist_is_a_valid_sfincs_deck(tmp_path):
    """The generated deck must parse as an input, not merely look like one."""
    from dkx.inputs import sfincs_input_from_raw
    from dkx.namelist import read_sfincs_input

    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.36])
    deck = tmp_path / "input.namelist"
    deck.write_text(term.namelist("/nonexistent/wout.nc", 0.36, er=-1.5))
    parsed = sfincs_input_from_raw(read_sfincs_input(deck))
    assert parsed.geometry.geometry_scheme == 5
    # rN_wish is sqrt(s), because inputRadialCoordinate = 3 is rN.
    assert float(parsed.geometry.r_n_wish) == pytest.approx(0.6)
    assert float(parsed.physics.er) == pytest.approx(-1.5)
    # Gradients are d/dpsiN, so the deck must select coordinate 1 -- with the
    # default of 4 the dNHatdpsiNs entries are ignored and every moment comes
    # back at ~1e-20 with no error at all.
    assert int(parsed.geometry.input_radial_coordinate_for_gradients) == 1
    assert len(parsed.species.d_n_hat_d_psi_ns) == 2
    assert all(value < 0.0 for value in parsed.species.d_n_hat_d_psi_ns)


def test_residuals_scale_by_the_reference_current(monkeypatch):
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.2, 0.5, 0.8],
                                   reference_current=2.0e5)  # fmt: skip
    values = np.array([1.0e5, -4.0e5, 0.0])
    monkeypatch.setattr(term, "_evaluate", lambda eq: values)
    assert term.residuals(None) == pytest.approx([0.5, -2.0, 0.0])
    assert term.total(None) == pytest.approx(0.25 + 4.0)
    assert term.J(None) == pytest.approx(term.residuals(None))
    assert term(None) == pytest.approx(term.residuals(None))


def test_unsolvable_surface_is_nan_in_the_profile_and_zero_in_the_residual(monkeypatch):
    """A failed solve must not read as "this device has no bootstrap current"."""
    term = KineticBootstrapCurrent(_Profiles(), surfaces=[0.3, 0.6])
    monkeypatch.setattr(term, "_evaluate", lambda eq: np.array([1.0e5, np.nan]))
    _s, profile = term.current_profile(None)
    assert np.isnan(profile[1])
    assert np.all(np.isfinite(term.residuals(None)))


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
