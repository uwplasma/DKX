from types import SimpleNamespace

import numpy as np

from dkx.result import Result
from dkx.workflows.ambipolar_native import (
    _brackets,
    _evaluation_budget,
    solve_native_ambipolar_surface,
)


def _fake_batch(current_function):
    def solve(_problem, values, **_kwargs):
        fields = np.asarray(values, dtype=np.float64)
        particle = np.stack((fields + 1.0, current_function(fields)), axis=1)
        return SimpleNamespace(
            moments={
                "particleFlux_vm_psiHat": particle,
                "heatFlux_vm_psiHat": 2.0 * particle,
                "FSABjHat": 3.0 * fields,
            },
            radial_current=current_function(fields),
            residual_norms=np.full(fields.shape, 1.0e-12),
            chunk_size=len(fields),
            n_chunks=1,
        )

    return solve


def _solve(**updates):
    controls = {
        "problem": SimpleNamespace(),
        "electric_field_bounds_kv_m": (-3.0, 3.0),
        "search_points": 7,
        "root_tolerance_kv_m": 1.0e-5,
        "max_root_iterations": 30,
        "find_all_roots": True,
        "previous_root_kv_m": 2.0,
        "radial_factor": 1.0,
        "solve_method": "auto",
        "solve_tolerance": 1.0e-10,
        "memory_budget_gb": 1.0,
    }
    controls.update(updates)
    return solve_native_ambipolar_surface(**controls)


def test_native_ambipolar_refines_all_roots_and_continues_nearest(monkeypatch):
    roots = np.asarray([-1.5, 0.4, 2.2])

    def current(field):
        return np.prod(np.asarray(field)[..., None] - roots, axis=-1)

    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))
    result = _solve()

    np.testing.assert_allclose(
        [root.electric_field_kv_m for root in result.roots],
        roots,
        rtol=0.0,
        atol=1.0e-5,
    )
    assert result.status == "bracketed_root"
    assert result.selected_root == 2
    assert result.roots[2].root_type == "electron"
    current_scale = max(
        abs(evaluation.radial_current_a_m2) for evaluation in result.evaluations
    )
    assert all(
        abs(root.radial_current_a_m2) < 1.0e-5 * current_scale for root in result.roots
    )
    assert len(result.evaluations) > 7
    assert result.batch_chunks > 1


def test_native_ambipolar_no_root_keeps_closest_real_evaluation(monkeypatch):
    current = lambda field: np.asarray(field) ** 2 + 1.0  # noqa: E731
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))
    result = _solve(
        previous_root_kv_m=None,
        convergence_enabled=True,
        max_refinements=1,
    )

    assert result.status == "no_bracketed_root"
    assert result.roots == ()
    assert result.selected_root is None
    assert result.selected.electric_field_kv_m == 0.0
    assert result.selected.radial_current_a_m2 > 0.0
    assert result.refinement_status == "no_bracket_observed"

    from dkx.execution import _ambipolar_result_arrays

    arrays, dimensions = _ambipolar_result_arrays([result], n_species=2)
    assert arrays["ambipolar_root_kV_m"].shape == (1, 1)
    assert np.isnan(arrays["ambipolar_root_kV_m"][0, 0])
    assert arrays["selected_ambipolar_root"][0] == -1
    assert arrays["ambipolar_refinement_status"][0] == "no_bracket_observed"
    assert arrays["ambipolar_evaluation_budget"][0] == result.evaluation_budget
    assert "interval_midpoint" in arrays["evaluation_reason"][0]
    assert dimensions["ambipolar_root_bracket_kV_m"] == (
        "surface",
        "root",
        "bracket_endpoint",
    )


def test_adaptive_midpoints_expose_two_crossings_with_same_endpoint_sign(
    monkeypatch,
):
    roots = np.asarray([-1.5, 0.125, 0.375])

    def current(field):
        return np.prod(np.asarray(field)[..., None] - roots, axis=-1)

    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))

    result = _solve(
        electric_field_bounds_kv_m=(-2.0, 2.0),
        search_points=5,
        previous_root_kv_m=None,
        convergence_enabled=True,
        convergence_observables=("particle_flux", "heat_flux"),
        max_refinements=3,
    )

    np.testing.assert_allclose(
        [root.electric_field_kv_m for root in result.roots],
        roots,
        rtol=0.0,
        atol=1.0e-5,
    )
    assert result.refinement_status == "resolved"
    assert [evidence.root_count for evidence in result.refinement] == [1, 1, 3, 3]
    assert result.refinement[-1].converged is True
    assert result.refinement[-1].root_movement_kv_m == 0.0
    assert result.refinement[-1].observable_relative_movement == 0.0
    midpoint_levels = {
        evaluation.refinement_level
        for evaluation in result.evaluations
        if evaluation.reason == "interval_midpoint"
    }
    assert midpoint_levels == {1, 2, 3}
    assert len(result.evaluations) <= result.evaluation_budget


def test_adaptive_no_bracket_and_exhausted_root_are_explicit(monkeypatch):
    no_root = lambda field: np.asarray(field) ** 2 + 1.0  # noqa: E731
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(no_root))
    missing = _solve(
        previous_root_kv_m=None,
        convergence_enabled=True,
        max_refinements=2,
    )
    assert missing.refinement_status == "no_bracket_observed"
    assert len(missing.refinement) == 3
    assert all(evidence.root_count == 0 for evidence in missing.refinement)

    exact = lambda field: np.asarray(field)  # noqa: E731
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(exact))
    exhausted = _solve(
        previous_root_kv_m=None,
        convergence_enabled=True,
        max_refinements=0,
    )
    assert exhausted.roots[0].electric_field_kv_m == 0.0
    assert exhausted.refinement_status == "refinement_exhausted"
    assert exhausted.refinement[0].max_bracket_width_kv_m == 0.0


def test_adaptive_evaluation_and_memory_preflight_are_bounded(monkeypatch):
    hierarchy, budget = _evaluation_budget(
        search_points=5,
        max_root_iterations=8,
        find_all_roots=True,
        convergence_enabled=True,
        max_refinements=2,
    )
    assert hierarchy == 17
    assert budget == 425
    with np.testing.assert_raises_regex(ValueError, "exceeds 100000"):
        _evaluation_budget(
            search_points=3,
            max_root_iterations=1,
            find_all_roots=False,
            convergence_enabled=True,
            max_refinements=1_000_000,
        )

    current = lambda field: np.asarray(field)  # noqa: E731
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))
    result = _solve(
        search_points=5,
        max_root_iterations=8,
        convergence_enabled=True,
        max_refinements=2,
        previous_root_kv_m=None,
    )
    assert result.evaluation_budget == 425
    assert len(result.evaluations) <= result.evaluation_budget
    assert [item.electric_field_kv_m for item in result.evaluations] == sorted(
        item.electric_field_kv_m for item in result.evaluations
    )

    with np.testing.assert_raises_regex(MemoryError, "memory preflight"):
        _solve(
            convergence_enabled=True,
            max_refinements=2,
            memory_budget_gb=1.0e-7,
        )


def test_exact_grid_zero_is_one_root_and_find_all_false_stops_after_first(
    monkeypatch,
):
    assert _brackets(np.asarray([-1.0, 0.0, 1.0]), np.asarray([-1.0, 0.0, 1.0])) == [
        (1, 1)
    ]
    current = lambda field: np.asarray(field) * (np.asarray(field) - 2.0)  # noqa: E731
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))
    result = _solve(
        electric_field_bounds_kv_m=(-1.0, 3.0),
        search_points=5,
        find_all_roots=False,
        previous_root_kv_m=None,
        convergence_enabled=True,
        max_refinements=1,
    )
    assert len(result.roots) == 1
    assert result.roots[0].electric_field_kv_m == 0.0
    assert result.refinement_status == "resolved"


def test_native_ambipolar_rejects_nonfinite_or_unconverged_batch(monkeypatch):
    def nonfinite(_problem, values, **_kwargs):
        fields = np.asarray(values, dtype=np.float64)
        batch = _fake_batch(lambda field: field)(None, fields)
        batch.residual_norms[0] = np.nan
        return batch

    monkeypatch.setattr("dkx.batch.batched_er_scan", nonfinite)
    with np.testing.assert_raises_regex(RuntimeError, "non-finite residual"):
        _solve()

    class FakeOperator:
        def rhs(self):
            return np.asarray([1.0])

    def unconverged(_problem, values, **_kwargs):
        fields = np.asarray(values, dtype=np.float64)
        batch = _fake_batch(lambda field: field)(None, fields)
        batch.residual_norms[:] = 1.0
        return batch

    monkeypatch.setattr("dkx.batch.batched_er_scan", unconverged)
    monkeypatch.setattr(
        "dkx.er.operator_at_er", lambda *_args, **_kwargs: FakeOperator()
    )
    with np.testing.assert_raises_regex(RuntimeError, "did not converge"):
        _solve(problem=SimpleNamespace(operator=FakeOperator(), dphi_per_er=1.0))


def test_native_result_labels_unbracketed_values_in_summary_and_plot(capsys):
    result = Result(
        case_id="a" * 64,
        case_name="mixed roots",
        workflow="ambipolar_profile",
        arrays={
            "surface": [0.2, 0.4],
            "species": ["ion"],
            "particle_flux_m2_s": [[1.0], [2.0]],
            "electric_field_kV_m": [-1.0, 0.5],
            "ambipolar_status": ["bracketed_root", "no_bracketed_root"],
            "ambipolar_refinement_status": [
                "resolved",
                "no_bracket_observed",
            ],
        },
        dimensions={
            "surface": ("surface",),
            "species": ("species",),
            "particle_flux_m2_s": ("surface", "species"),
            "electric_field_kV_m": ("surface",),
            "ambipolar_status": ("surface",),
            "ambipolar_refinement_status": ("surface",),
        },
        metadata={"converged": True},
    )

    result.print_summary()
    summary = capsys.readouterr().out
    assert "1/2 surfaces bracketed" in summary
    assert "1 resolved" in summary
    assert "1 no bracket observed" in summary
    figure = result.plot()
    assert "no bracketed root" in figure.texts[0].get_text()
    assert "no bracket observed after finite refinement" in figure.texts[0].get_text()
