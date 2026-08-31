"""Tests for the phase-space convergence study behind ``dkx converge``.

The arithmetic is tested against a stub solve rather than real ones: a study
costs ``len(axes) + 2`` solves, and what needs pinning here is the refinement
schedule, the relative-change comparison and the verdict, none of which depend
on the physics. One real solve is exercised separately and marked slow.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pytest

from dkx.workflows import converge as cv


@dataclass(frozen=True)
class FakeResolution:
    theta: int
    zeta: int
    pitch: int
    speed: int


@dataclass(frozen=True)
class FakeCase:
    resolution: FakeResolution


class FakeResult:
    def __init__(self, arrays):
        self.arrays = arrays


def study(monkeypatch, response, *, case=None, **kwargs):
    """Run a study where ``response`` maps a resolution to one flux value.

    Returns ``(report, calls)``; ``calls`` is every resolution the study asked
    for, in order, which is what pins the refinement schedule and the count.
    """
    calls: list[FakeResolution] = []

    def fake_run_case(case, **_):
        calls.append(case.resolution)
        return FakeResult({"particle_flux_m2_s": np.array([response(case.resolution)])})

    monkeypatch.setattr("dkx.execution.run_case", fake_run_case)
    report = cv.converge_case(
        case or FakeCase(FakeResolution(theta=10, zeta=4, pitch=10, speed=10)), **kwargs
    )
    return report, calls


# --------------------------------------------------------------------------
# Refinement schedule
# --------------------------------------------------------------------------


def test_each_axis_is_refined_alone_then_all_together(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0)
    assert [r.label for r in report.refinements] == ["theta", "zeta", "pitch", "speed"]
    assert report.joint is not None
    # baseline + one per axis + one joint
    assert len(calls) == 6
    assert report.joint.resolution == {"theta": 15, "zeta": 6, "pitch": 15, "speed": 15}


def test_refining_an_axis_leaves_the_others_alone(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0)
    theta = next(r for r in report.refinements if r.label == "theta")
    assert theta.resolution == {"theta": 15, "zeta": 4, "pitch": 10, "speed": 10}


def test_an_axisymmetric_zeta_is_not_refined(monkeypatch) -> None:
    """``zeta = 1`` says the configuration has no toroidal variation.

    Scaling it would solve a different problem rather than the same one more
    accurately, so the axis is skipped and reported as skipped -- not silently
    refined, and not counted as a converged axis it never tested.
    """
    case = FakeCase(FakeResolution(theta=10, zeta=1, pitch=10, speed=10))
    report, calls = study(monkeypatch, lambda r: 1.0, case=case)
    assert [r.label for r in report.refinements] == ["theta", "pitch", "speed"]
    assert all(r.resolution["zeta"] == 1 for r in report.refinements)
    assert report.joint is not None and report.joint.resolution["zeta"] == 1


def test_refinement_always_advances_even_when_rounding_would_not(monkeypatch) -> None:
    """A small axis with a small factor must still grow by at least one node."""
    case = FakeCase(FakeResolution(theta=2, zeta=2, pitch=2, speed=2))
    report, calls = study(monkeypatch, lambda r: 1.0, case=case, factor=1.01)
    assert all(r.resolution[r.label] == 3 for r in report.refinements)


def test_a_factor_that_does_not_refine_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="factor must exceed 1.0"):
        study(monkeypatch, lambda r: 1.0, factor=1.0)


def test_an_unknown_axis_is_refused(monkeypatch) -> None:
    with pytest.raises(ValueError, match="unknown refinement axes"):
        study(monkeypatch, lambda r: 1.0, axes=("theta", "radius"))


# --------------------------------------------------------------------------
# Verdict
# --------------------------------------------------------------------------


def test_a_solution_independent_of_resolution_is_converged(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 3.0)
    assert report.per_axis_worst == 0.0
    assert report.converged


def test_a_single_unconverged_axis_fails_the_whole_study(monkeypatch) -> None:
    """One axis still moving is enough; the verdict is over the worst axis.

    Averaging would let three settled axes hide a fourth that is not, which is
    the failure this command exists to surface.
    """
    report, calls = study(
        monkeypatch, lambda r: 1.0 + (0.5 if r.speed > 10 else 0.0), tolerance=0.02
    )
    speed = next(r for r in report.refinements if r.label == "speed")
    assert speed.worst == pytest.approx(0.5)
    assert not report.converged


def test_the_joint_run_can_fail_a_study_every_axis_passed(monkeypatch) -> None:
    """The reason the joint run is not redundant.

    This response is flat unless *two* axes move together, so every single-axis
    refinement reports zero change and a per-axis-only study would call the case
    converged. This is not hypothetical: on the shipped analytic tokamak deck,
    theta refinement moves the outputs by 0.2% at pitch=8 and by 74% at
    pitch=40 -- the apparent theta convergence was an artifact of pitch being
    too coarse to expose it.
    """
    def response(r):
        return 2.0 if (r.theta > 10 and r.pitch > 10) else 1.0

    report, calls = study(monkeypatch, response, tolerance=0.02)
    assert report.per_axis_worst == 0.0
    assert report.joint is not None and report.joint.worst == pytest.approx(1.0)
    assert not report.converged
    assert report.axes_understate_the_joint_change


def test_skipping_the_joint_run_is_recorded_not_assumed_converged(monkeypatch) -> None:
    report, calls = study(monkeypatch, lambda r: 1.0, joint=False)
    assert report.joint is None
    assert not report.axes_understate_the_joint_change


def test_a_single_refinable_axis_needs_no_joint_run(monkeypatch) -> None:
    """With one axis there is nothing to refine jointly, so the run is skipped."""
    report, calls = study(monkeypatch, lambda r: 1.0, axes=("theta",))
    assert report.joint is None
    assert len(calls) == 2


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def test_a_quantity_that_is_physically_zero_is_compared_absolutely() -> None:
    """Dividing by an exactly zero reference would report a spurious blow-up.

    An exactly symmetric configuration has zero bootstrap current; a refinement
    that leaves it at 1e-18 has not changed the physics, and a relative measure
    against zero would call that an infinite change.
    """
    changes = cv._relative_changes({"j": 0.0}, {"j": 1e-18})
    assert changes["j"] == pytest.approx(1e-18)


def test_relative_change_is_used_for_ordinary_magnitudes() -> None:
    assert cv._relative_changes({"q": 2.0}, {"q": 3.0})["q"] == pytest.approx(0.5)


def test_an_observable_missing_from_a_run_is_skipped_not_counted_as_zero() -> None:
    assert cv._relative_changes({"a": 1.0, "b": 2.0}, {"a": 1.0}) == {"a": 0.0}


def test_an_observable_is_reduced_by_its_largest_magnitude() -> None:
    """Max absolute value, so a per-species sign cancellation cannot hide drift.

    Summing would let an electron flux rising and an ion flux falling by the
    same amount register as no change at all.
    """
    assert cv._scalarize(np.array([[3.0, -7.0], [1.0, 2.0]])) == pytest.approx(7.0)
    assert cv._scalarize(np.array([])) == 0.0


def test_a_result_without_any_requested_observable_is_an_error(monkeypatch) -> None:
    """Silence here would report a vacuous 'converged' over an empty comparison."""
    monkeypatch.setattr(
        "dkx.execution.run_case", lambda case, **_: FakeResult({"other": np.array([1.0])})
    )
    with pytest.raises(ValueError, match="none of the requested observables"):
        cv.converge_case(FakeCase(FakeResolution(10, 4, 10, 10)))


def test_the_cli_axis_list_matches_the_workflow(monkeypatch) -> None:
    """The CLI mirrors AXES so building the parser does not import the solver."""
    from dkx import cli

    assert cli._CONVERGE_AXES == cv.AXES
