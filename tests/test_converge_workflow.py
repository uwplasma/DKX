"""Tests for the phase-space convergence study behind ``dkx converge``.

The arithmetic is tested against a stub solve rather than real ones: a study
costs ``len(axes) + 2`` solves, and what needs pinning here is the refinement
schedule, the relative-change comparison and the verdict, none of which depend
on the physics. One real solve is exercised separately and marked slow.
"""

from __future__ import annotations

from dataclasses import dataclass

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
        self.metadata = {"converged": True}


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
    kwargs.setdefault("observables", ("particle_flux_m2_s",))
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


def test_a_zero_reference_requires_an_explicit_physical_absolute_budget(monkeypatch):
    def response(r):
        return 0.0 if r.theta == 10 else 1e-18
    report, _ = study(monkeypatch, response, axes=("theta",))
    assert not report.converged
    report, _ = study(monkeypatch, response, axes=("theta",),
                      absolute_tolerances={"particle_flux_m2_s": 2e-18})
    assert report.converged
    report, _ = study(monkeypatch, lambda r: 0.0, axes=("theta",))
    assert report.converged


def test_relative_change_is_used_for_ordinary_magnitudes() -> None:
    assert cv._relative_changes({"q": 2.0}, {"q": 3.0})["q"] == pytest.approx(0.5)


def test_an_observable_missing_from_a_refinement_fails_admission() -> None:
    changes = cv._relative_changes({"a": 1.0, "b": 2.0}, {"a": 1.0})
    assert changes["a"] == 0.0
    assert np.isinf(changes["b"])


def test_a_result_without_any_requested_observable_is_an_error(monkeypatch) -> None:
    """Silence here would report a vacuous 'converged' over an empty comparison."""
    monkeypatch.setattr(
        "dkx.execution.run_case", lambda case, **_: FakeResult({"other": np.array([1.0])})
    )
    with pytest.raises(ValueError, match="requested observables are missing"):
        cv.converge_case(FakeCase(FakeResolution(10, 4, 10, 10)))


def test_the_cli_axis_list_matches_the_workflow(monkeypatch) -> None:
    """The CLI mirrors AXES so building the parser does not import the solver."""
    from dkx import cli

    assert cli._CONVERGE_AXES == cv.AXES

@pytest.mark.parametrize('ref, got', [
    ([1., -1.], [-1., 1.]),
    ([[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]),
    ([1000., 1.], [1000., 2.]),
])
def test_refinement_compares_each_signed_species_and_surface(monkeypatch, ref, got):
    report, _ = study(monkeypatch, lambda r: ref if r.theta == 10 else got,
                      axes=('theta',))
    assert not report.converged
    assert report.refinements[0].worst >= 1.

@pytest.mark.parametrize('got', [[np.nan, 2.], [np.inf, 2.], [], [[1., 2.]]])
def test_invalid_or_misaligned_observables_cannot_pass(monkeypatch, got):
    report, _ = study(monkeypatch, lambda r: [1., 2.] if r.theta == 10 else got,
                      axes=('theta',))
    assert not report.converged


def test_no_refinable_axes_does_not_certify_resolution(monkeypatch):
    report, _ = study(monkeypatch, lambda r: 1., axes=())
    assert not report.converged
    report, _ = study(monkeypatch, lambda r: 1., axes=("zeta",),
                      case=FakeCase(FakeResolution(10, 1, 10, 10)))
    assert not report.converged


def test_a_failed_solve_cannot_certify_resolution(monkeypatch):
    result = FakeResult({"particle_flux_m2_s": np.ones(2)})
    result.metadata["converged"] = False
    monkeypatch.setattr("dkx.execution.run_case", lambda case: result)
    with pytest.raises(ValueError, match="failed solve"):
        cv.converge_case(FakeCase(FakeResolution(10, 4, 10, 10)))


@pytest.mark.parametrize("tolerance", [0., -1., np.inf, np.nan])
def test_invalid_convergence_tolerance_is_rejected(monkeypatch, tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        study(monkeypatch, lambda r: 1., tolerance=tolerance)


@pytest.mark.parametrize("atols", [{"missing": 1.}, {"particle_flux_m2_s": -1.},
                                    {"particle_flux_m2_s": np.nan},
                                    {"particle_flux_m2_s": np.inf}])
def test_invalid_absolute_budgets_are_rejected(monkeypatch, atols):
    with pytest.raises(ValueError, match="absolute"):
        study(monkeypatch, lambda r: 1., absolute_tolerances=atols)


def test_entrywise_change_handles_extreme_finite_values():
    changes = cv._relative_changes({"q": np.array([1e308, 1e-308])},
                                   {"q": np.array([-1e308, 2e-308])})
    assert changes["q"] == pytest.approx(2.)


def test_large_absolute_budget_does_not_overflow_into_false_convergence():
    changes = cv._relative_changes({"q": 0.}, {"q": 1e308}, tolerance=.02,
                                   absolute_tolerances={"q": 5e307})
    assert changes["q"] == pytest.approx(.04)


def test_a_missing_requested_observable_cannot_hide_behind_an_available_one(monkeypatch):
    with pytest.raises(ValueError, match="missing"):
        study(monkeypatch, lambda r: 1., observables=("particle_flux_m2_s", "heat_flux_W_m2"))


@pytest.mark.parametrize('mode', [1, 2, 3])
def test_namelist_refinement_preserves_physics_and_checks_each_rhs(monkeypatch, mode):
    from dataclasses import replace
    from pathlib import Path
    from types import SimpleNamespace
    import importlib
    from dkx.inputs import load_sfincs_input

    inp = load_sfincs_input(Path(__file__).parent / 'ref/pas_1species_PAS_Er_tiny_xgrid4_xdot4.input.namelist')
    inp = replace(inp, general=replace(inp.general, rhs_mode=mode),
                  physics=replace(inp.physics, er=-1., use_dkes_exb_drift=False),
                  resolution=replace(inp.resolution, n_theta=5))
    calls = []
    invalid = False

    def driver(updated, **kwargs):
        calls.append(updated)
        assert updated.physics == inp.physics
        assert updated.geometry == inp.geometry
        assert updated.species == inp.species
        assert kwargs['solver'].tol == inp.resolution.solver_tolerance
        assert kwargs['solver'].keep_lowest == updated.resolution.n_xi
        states = np.ones((mode, 2))
        if invalid:
            states[-1, 0] = 2
        return SimpleNamespace(
            operator=SimpleNamespace(rhs=lambda i: np.ones(2), apply=lambda x: x),
            solve_result=SimpleNamespace(converged=True), state_vector=states[0],
            state_vectors=states, moments={'FSABFlow': np.array([-2.]),
                'particleFlux_vm_psiHat': np.array([3.]), 'heatFlux_vm_psiHat': np.array([4.])},
            transport_matrix=np.eye(mode),
        )

    run_module = importlib.import_module('dkx.run')
    monkeypatch.setattr(run_module, 'run_profile' if mode == 1 else 'run_transport_matrix', driver)
    report = cv.converge_sfincs_input(inp, axes=('theta',), factor=1.2)
    assert report.converged
    assert calls[0].resolution == inp.resolution
    assert calls[1].resolution.n_theta == report.refinements[0].resolution["theta"] == 7
    assert calls[1].resolution.n_xi == inp.resolution.n_xi
    invalid = True
    with pytest.raises(ValueError, match='failed solve'):
        cv.converge_sfincs_input(inp, axes=('theta',))


def test_full_recovery_audits_equations_without_changing_transport():
    from pathlib import Path
    from dkx.api import SolverOptions
    from dkx.run import run_transport_matrix

    deck = Path(__file__).parent / 'ref/monoenergetic_PAS_tiny_scheme1.input.namelist'
    low = run_transport_matrix(deck, solver=SolverOptions(memory_budget_gb=1e-12), emit=None)
    full = run_transport_matrix(deck, solver=SolverOptions(
        memory_budget_gb=1e-12, keep_lowest=low.operator.n_xi), emit=None)
    assert full.solve_result.method == 'block_tridiagonal_truncated'
    assert full.solve_result.converged
    np.testing.assert_allclose(full.transport_matrix, low.transport_matrix, rtol=1e-10, atol=0)
    for i, state in enumerate(full.state_vectors, 1):
        rhs = np.asarray(full.operator.rhs(i))
        assert np.linalg.norm(np.asarray(full.operator.apply(state)) - rhs) / np.linalg.norm(rhs) < 1e-10


# --------------------------------------------------------------------------
# Grid uncertainty: Richardson extrapolation over a three-rung ladder
# --------------------------------------------------------------------------


def manufactured(order, *, exact=3.0, coefficient=1.0):
    """``f(N) = exact + coefficient * N**-order``, a ladder with a known answer."""
    return lambda size: exact + coefficient * float(size) ** -order


def test_a_second_order_ladder_recovers_its_order_and_its_exact_value() -> None:
    f = manufactured(2.0)
    estimate = cv.richardson_uncertainty(f(10), f(20), f(40), sizes=(10, 20, 40))
    assert estimate.status == "asymptotic" and estimate.usable
    assert estimate.order == pytest.approx(2.0, abs=1e-9)
    assert float(np.ravel(estimate.extrapolated)[0]) == pytest.approx(3.0, abs=1e-12)
    # The bar must cover the finest solution's actual error.
    assert estimate.relative >= abs(f(40) - 3.0) / 3.0


def test_a_fourth_order_ladder_recovers_its_order() -> None:
    f = manufactured(4.0)
    estimate = cv.richardson_uncertainty(f(10), f(20), f(40), sizes=(10, 20, 40))
    assert estimate.order == pytest.approx(4.0, abs=1e-9)


def test_the_published_asme_worked_example_is_reproduced() -> None:
    """Celik et al. (2008), ASME J. Fluids Eng. 130, 078001, example 1.

    Three grids with r21 = 1.5 and r32 = 1.333 give a published observed order
    of 1.53, an extrapolated 6.1685 and a fine-grid GCI of 2.2%. Reproducing an
    external worked example checks the procedure itself, which a manufactured
    ladder of our own construction cannot.
    """
    estimate = cv.richardson_uncertainty(
        5.863, 5.972, 6.063, sizes=(3000, 4000, 6000)
    )
    assert estimate.order == pytest.approx(1.53, abs=0.01)
    assert float(np.ravel(estimate.extrapolated)[0]) == pytest.approx(6.1685, abs=1e-4)
    assert estimate.relative == pytest.approx(0.022, abs=0.001)


def test_unequal_refinement_ratios_still_recover_the_order() -> None:
    """Integer resolutions rarely give a constant ratio: 9, 13, 20 is what
    ``factor = 1.5`` actually produces. The ASME V&V 20 iteration handles it."""
    f = manufactured(2.0)
    estimate = cv.richardson_uncertainty(f(9), f(13), f(20), sizes=(9, 13, 20))
    assert estimate.status == "asymptotic"
    assert estimate.order == pytest.approx(2.0, abs=1e-6)
    assert float(np.ravel(estimate.extrapolated)[0]) == pytest.approx(3.0, abs=1e-9)


def test_the_er15_bootstrap_ladder_is_refused_not_extrapolated() -> None:
    """The measured ``Er = 15`` pitch ladder reversed the sign of the current
    between ``Nxi = 40`` and ``60``, and its differences grow under refinement.

    Extrapolating through that invents accuracy the solves do not have, so the
    estimator must refuse it. This is the case the refusal exists for.
    """
    estimate = cv.richardson_uncertainty(
        5.877e-3, 3.000e-3, -3.364e-3, sizes=(30, 40, 60)
    )
    assert not estimate.usable
    assert "asymptotic" in estimate.status and estimate.relative == float("inf")


def test_an_oscillating_ladder_is_refused() -> None:
    estimate = cv.richardson_uncertainty(1.0, 1.1, 1.0, sizes=(10, 20, 40))
    assert not estimate.usable


def test_a_diverging_ladder_is_refused() -> None:
    """Differences that grow under refinement are divergence, not convergence."""
    estimate = cv.richardson_uncertainty(1.0, 1.1, 1.4, sizes=(10, 20, 40))
    assert not estimate.usable


def test_the_worst_entry_sets_the_uncertainty() -> None:
    """One settled surface must not certify a moving one."""
    slow, fast = manufactured(2.0), manufactured(2.0, exact=-2.0, coefficient=50.0)
    ladder = [np.array([slow(n), fast(n)]) for n in (10, 20, 40)]
    estimate = cv.richardson_uncertainty(*ladder, sizes=(10, 20, 40))
    alone = cv.richardson_uncertainty(
        *[np.array([fast(n)]) for n in (10, 20, 40)], sizes=(10, 20, 40)
    )
    assert estimate.relative == pytest.approx(alone.relative)


def test_one_refused_entry_refuses_the_whole_observable() -> None:
    good = manufactured(2.0)
    ladder = [np.array([good(n), v]) for n, v in ((10, 1.0), (20, 1.1), (40, 1.0))]
    assert not cv.richardson_uncertainty(*ladder, sizes=(10, 20, 40)).usable


@pytest.mark.parametrize(
    "ladder, sizes",
    [
        ((1.0, 2.0, float("nan")), (10, 20, 40)),
        ((np.array([1.0]), np.array([1.0, 2.0]), np.array([1.0])), (10, 20, 40)),
        ((1.0, 1.1, 1.2), (40, 20, 10)),
        ((1.0, 1.1, 1.2), (10, 10, 40)),
    ],
)
def test_an_unusable_ladder_is_refused(ladder, sizes) -> None:
    assert not cv.richardson_uncertainty(*ladder, sizes=sizes).usable


def test_a_stalled_coarse_pair_cannot_give_an_order() -> None:
    """Two equal coarse rungs make the difference ratio undefined.

    Refinement produced no change between the coarse and medium grids but did
    between medium and fine, which is not a convergence history any order can
    be read from.
    """
    assert not cv.richardson_uncertainty(1.0, 1.0, 1.1, sizes=(10, 20, 40)).usable


def test_an_implausibly_high_order_is_refused() -> None:
    """An order far above any discretization in DKX means the ladder is not
    measuring convergence, so no bar is reported."""
    # exact=0 keeps the differences representable: with an offset of 3.0 an
    # order-20 ladder underflows to three identical doubles, which is a
    # converged observable rather than a high-order one.
    f = manufactured(20.0, exact=0.0)
    assert not cv.richardson_uncertainty(
        f(10), f(20), f(40), sizes=(10, 20, 40), max_order=12.0
    ).usable


def test_a_nonpositive_safety_factor_is_refused() -> None:
    with pytest.raises(ValueError, match="safety factor"):
        cv.richardson_uncertainty(1.0, 1.1, 1.2, sizes=(10, 20, 40), safety=0.0)


def test_an_exactly_reproduced_observable_has_no_grid_uncertainty() -> None:
    estimate = cv.richardson_uncertainty(2.5, 2.5, 2.5, sizes=(10, 20, 40))
    assert estimate.usable and estimate.relative == 0.0


def test_a_zero_observable_needs_an_explicit_absolute_scale() -> None:
    refused = cv.richardson_uncertainty(0.01, 0.0025, 0.0, sizes=(10, 20, 40))
    assert not refused.usable
    scaled = cv.richardson_uncertainty(
        0.01, 0.0025, 0.0, sizes=(10, 20, 40), absolute_tolerance=1.0
    )
    assert scaled.usable and scaled.relative < 1.0


def test_the_third_rung_gives_every_axis_an_error_bar(monkeypatch) -> None:
    def response(r):
        return 3.0 + sum(getattr(r, a) ** -2.0 for a in ("theta", "zeta", "pitch", "speed"))

    report, calls = study(monkeypatch, response, richardson=True)
    for refinement in report.refinements:
        estimate = refinement.uncertainties["particle_flux_m2_s"]
        assert estimate.usable, (refinement.label, estimate.status)
        assert estimate.order == pytest.approx(2.0, abs=1e-3)
    assert np.isfinite(report.worst_grid_uncertainty)
    # baseline + (refine + third rung) per axis + joint
    assert len(calls) == 1 + 2 * len(report.refinements) + 1


def test_without_the_third_rung_no_uncertainty_is_claimed(monkeypatch) -> None:
    report, _ = study(monkeypatch, lambda r: 1.0 + 1.0 / r.theta)
    assert all(r.uncertainties is None for r in report.refinements)
    assert np.isnan(report.worst_grid_uncertainty)


def test_a_refused_ladder_makes_the_reported_uncertainty_infinite(monkeypatch) -> None:
    """An estimate that could not be made is not a small one."""
    values = {}

    def response(r):
        # Oscillate in theta only; the other axes converge cleanly.
        values.setdefault(r.theta, [1.0, 1.1, 1.0][len(values) % 3])
        return values[r.theta] + sum(getattr(r, a) ** -2.0 for a in ("zeta", "pitch", "speed"))

    report, _ = study(monkeypatch, response, richardson=True, axes=("theta",), joint=False)
    assert report.worst_grid_uncertainty == float("inf")
