from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from dkx.result import Result
from dkx.workflows.ambipolar_native import (
    NativeAmbipolarRoot,
    NativeAmbipolarSurface,
    RootEvaluation,
    _brackets,
    _evaluation_budget,
    continue_ambipolar_branches,
    solve_native_ambipolar_surface,
)


def _fake_batch(current_function):
    def solve(_problem, values, **_kwargs):
        fields = np.asarray(values, dtype=np.float64)
        particle = np.stack((fields + 1.0, current_function(fields)), axis=1)
        particle_vs_speed = np.stack((0.25 * particle, 0.75 * particle), axis=1)
        requested = str(_kwargs.get("solve_method", "auto"))
        return SimpleNamespace(
            moments={
                "particleFlux_vm_psiHat": particle,
                "heatFlux_vm_psiHat": 2.0 * particle,
                "particleFlux_vm_psiHat_vs_x": particle_vs_speed,
                "heatFlux_vm_psiHat_vs_x": 2.0 * particle_vs_speed,
                "FSABjHat": 3.0 * fields,
            },
            radial_current=current_function(fields),
            residual_norms=np.full(fields.shape, 1.0e-12),
            chunk_size=len(fields),
            n_chunks=1,
            method=requested,
            executed_method=(
                "block_tridiagonal_truncated" if requested == "auto" else requested
            ),
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


def _branch_surface(fields, root_types):
    roots = []
    evaluations = []
    for field, root_type in zip(fields, root_types):
        evaluation = RootEvaluation(
            electric_field_kv_m=float(field),
            radial_current_a_m2=0.0,
            particle_flux_m2_s=np.asarray([field + 4.0]),
            heat_flux_w_m2=np.asarray([2.0 * (field + 4.0)]),
            parallel_current_a_t_m2=float(field),
            residual_norm=1.0e-12,
            stage="root_refinement",
        )
        evaluations.append(evaluation)
        roots.append(
            NativeAmbipolarRoot(
                electric_field_kv_m=float(field),
                radial_current_a_m2=0.0,
                slope_a_m2_per_kv_m=-1.0 if root_type == "unstable" else 1.0,
                root_type=root_type,
                bracket_kv_m=(float(field), float(field)),
                evaluation=evaluation,
            )
        )
    selected = (
        evaluations[0]
        if evaluations
        else RootEvaluation(
            electric_field_kv_m=0.0,
            radial_current_a_m2=1.0,
            particle_flux_m2_s=np.asarray([0.0]),
            heat_flux_w_m2=np.asarray([0.0]),
            parallel_current_a_t_m2=0.0,
            residual_norm=1.0e-12,
            stage="coarse_scan",
        )
    )
    return NativeAmbipolarSurface(
        evaluations=tuple(evaluations) or (selected,),
        roots=tuple(roots),
        selected_root=0 if roots else None,
        selected=selected,
        status="bracketed_root" if roots else "no_bracketed_root",
        solve_seconds=0.0,
        batch_chunk_size=1,
        batch_chunks=1,
    )


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
    for evaluation in result.evaluations:
        np.testing.assert_allclose(
            np.sum(evaluation.particle_flux_m2_s_vs_speed, axis=0),
            evaluation.particle_flux_m2_s,
        )
        np.testing.assert_allclose(
            np.sum(evaluation.heat_flux_w_m2_vs_speed, axis=0),
            evaluation.heat_flux_w_m2,
        )


def test_seeded_brackets_refine_only_explicit_intervals(monkeypatch):
    all_roots = np.asarray([-2.0, -0.5, 1.0])

    def current(field):
        return np.prod(np.asarray(field)[..., None] - all_roots, axis=-1)

    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(current))
    result = _solve(
        electric_field_bounds_kv_m=(-3.0, 2.0),
        seed_brackets_kv_m=((-2.25, -1.75), (0.75, 1.25)),
        previous_root_kv_m=None,
    )

    np.testing.assert_allclose(
        [root.electric_field_kv_m for root in result.roots],
        [-2.0, 1.0],
        rtol=0.0,
        atol=1.0e-5,
    )
    assert result.status == "bracketed_root"
    assert result.search_strategy == "seeded_brackets"
    assert result.search_scope == "explicit_seeded_intervals_only"
    assert result.evaluation_budget == 64
    assert len(result.evaluations) <= result.evaluation_budget
    assert (
        sum(
            evaluation.reason == "seeded_bracket_endpoint"
            for evaluation in result.evaluations
        )
        == 4
    )

    from dkx.execution import _ambipolar_result_arrays

    arrays, dimensions = _ambipolar_result_arrays([result], n_species=2)
    assert arrays["ambipolar_search_strategy"][0] == "seeded_brackets"
    assert arrays["ambipolar_search_scope"][0] == "explicit_seeded_intervals_only"
    assert dimensions["ambipolar_search_scope"] == ("surface",)


def test_seeded_brackets_report_partial_and_total_failures(monkeypatch):
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(lambda x: x))
    partial = _solve(seed_brackets_kv_m=((-1.0, 1.0), (2.0, 3.0)))
    failed = _solve(seed_brackets_kv_m=((1.0, 2.0),))

    assert len(partial.roots) == 1
    assert partial.status == "seeded_bracket_partial_failure"
    assert failed.roots == ()
    assert failed.status == "seeded_bracket_failed"
    assert failed.selected.electric_field_kv_m == 1.0


def test_native_ambipolar_speed_diagnostics_have_named_axes(monkeypatch):
    monkeypatch.setattr("dkx.batch.batched_er_scan", _fake_batch(lambda x: x))
    result = _solve(search_points=3, find_all_roots=False)
    result = replace(
        result,
        evaluations=tuple(
            replace(
                evaluation,
                legendre_tail_relative_l2=np.full((2, 2), 0.125),
                legendre_tail_relative_l2_upper_bound=np.full((2, 2), 0.25),
            )
            for evaluation in result.evaluations
        ),
    )

    from dkx.execution import _ambipolar_result_arrays

    arrays, dimensions = _ambipolar_result_arrays(
        [result], n_species=2, speed_nodes=np.asarray([0.5, 1.5])
    )
    assert dimensions["evaluation_particle_flux_m2_s_vs_speed"] == (
        "surface",
        "evaluation",
        "speed",
        "species",
    )
    assert dimensions["evaluation_heat_flux_W_m2_vs_speed"] == (
        "surface",
        "evaluation",
        "speed",
        "species",
    )
    assert dimensions["evaluation_legendre_tail_relative_l2"] == (
        "surface",
        "evaluation",
        "speed",
        "species",
    )
    assert dimensions["evaluation_legendre_tail_relative_l2_upper_bound"] == (
        "surface",
        "evaluation",
        "speed",
        "species",
    )
    np.testing.assert_allclose(arrays["speed_v_th"], [0.5, 1.5])
    np.testing.assert_allclose(
        np.sum(arrays["evaluation_particle_flux_m2_s_vs_speed"], axis=2),
        arrays["evaluation_particle_flux_m2_s"],
    )
    np.testing.assert_allclose(
        np.sum(arrays["evaluation_heat_flux_W_m2_vs_speed"], axis=2),
        arrays["evaluation_heat_flux_W_m2"],
    )
    np.testing.assert_allclose(arrays["evaluation_legendre_tail_relative_l2"], 0.125)
    np.testing.assert_allclose(
        arrays["evaluation_legendre_tail_relative_l2_upper_bound"], 0.25
    )


def test_radial_branch_identity_crossing_loss_merger_and_selection_are_retained(
    tmp_path, capsys
):
    tracked = continue_ambipolar_branches(
        [
            _branch_surface([-2.0, 2.0], ["ion", "electron"]),
            _branch_surface([-0.5, 0.0, 0.5], ["ion", "unstable", "electron"]),
            _branch_surface([-1.1, 1.1], ["ion", "electron"]),
        ],
        surfaces=np.asarray([0.1, 0.2, 0.3]),
        electric_field_bounds_kv_m=(-3.0, 3.0),
        continue_selection=True,
    )

    first_ids = [root.branch_id for root in tracked[0].roots]
    assert first_ids == ["ion-000", "electron-000"]
    assert tracked[0].selection_reason == "nearest_zero_initial"
    assert tracked[0].selected_branch_id == "ion-000"
    assert tracked[1].selected_branch_id == "ion-000"
    assert tracked[1].selection_reason == "continued_selected_branch"
    assert tracked[1].roots[1].branch_id == "unstable-000"
    assert {event.kind for event in tracked[1].branch_events} == {"creation"}

    final_by_id = {root.branch_id: root for root in tracked[2].roots}
    assert final_by_id["electron-000"].electric_field_kv_m == -1.1
    assert final_by_id["ion-000"].electric_field_kv_m == 1.1
    assert tracked[2].selected_branch_id == "ion-000"
    assert tracked[2].selected.electric_field_kv_m == 1.1
    final_kinds = [event.kind for event in tracked[2].branch_events]
    assert final_kinds.count("classification_transition") == 2
    assert set(final_kinds) >= {"loss", "merger", "crossing"}
    merger = next(event for event in tracked[2].branch_events if event.kind == "merger")
    assert merger.branch_ids[0] == "unstable-000"
    assert "discrete merger candidate" in merger.detail
    crossing = next(
        event for event in tracked[2].branch_events if event.kind == "crossing"
    )
    assert set(crossing.branch_ids) == {"ion-000", "electron-000"}
    assert all(event.nonsmooth for event in tracked[2].branch_events)

    from dkx.execution import _ambipolar_result_arrays

    arrays, dimensions = _ambipolar_result_arrays(tracked, n_species=1)
    arrays["surface"] = np.asarray([0.1, 0.2, 0.3])
    dimensions["surface"] = ("surface",)
    arrays["electric_field_kV_m"] = np.asarray(
        [result.selected.electric_field_kv_m for result in tracked]
    )
    dimensions["electric_field_kV_m"] = ("surface",)
    np.testing.assert_array_equal(
        arrays["selected_ambipolar_branch"],
        ["ion-000", "ion-000", "ion-000"],
    )
    assert arrays["ambipolar_root_branch_id"][2, 1] == "ion-000"
    assert "crossing" in arrays["ambipolar_branch_event_kind"][2]
    assert arrays["ambipolar_nonsmooth_event"].tolist() == [0, 1, 1]
    result = Result(
        case_id="a" * 64,
        case_name="branch events",
        workflow="ambipolar_profile",
        arrays=arrays,
        dimensions=dimensions,
        metadata={"converged": True},
    )
    path = result.save(tmp_path / "branch-events.nc")
    loaded = Result.load(path)
    np.testing.assert_array_equal(
        loaded.selected_ambipolar_branch,
        result.selected_ambipolar_branch,
    )
    np.testing.assert_array_equal(
        loaded.ambipolar_branch_event_kind,
        result.ambipolar_branch_event_kind,
    )
    result.print_summary()
    assert "3 identities; 6 interior events" in capsys.readouterr().out
    figure = result.plot()
    assert "branch event warning" in figure._suptitle.get_text()
    labels = {
        line.get_label()
        for line in figure.axes[-1].lines
        if not line.get_label().startswith("_")
    }
    assert labels >= {"selected branch", "ion-000", "electron-000", "unstable-000"}


def test_branch_selection_can_be_disabled_without_hiding_identities():
    tracked = continue_ambipolar_branches(
        [
            _branch_surface([-2.0, 2.0], ["ion", "electron"]),
            _branch_surface([-0.5, 0.0, 0.5], ["ion", "unstable", "electron"]),
        ],
        surfaces=np.asarray([0.1, 0.2]),
        electric_field_bounds_kv_m=(-3.0, 3.0),
        continue_selection=False,
    )

    assert tracked[1].selected.electric_field_kv_m == 0.0
    assert tracked[1].selected_branch_id == "unstable-000"
    assert tracked[1].selection_reason == "nearest_zero_continuation_disabled"
    assert all(root.branch_id for result in tracked for root in result.roots)

    with np.testing.assert_raises_regex(ValueError, "strictly increasing surfaces"):
        continue_ambipolar_branches(
            tracked,
            surfaces=np.asarray([0.2, 0.1]),
            electric_field_bounds_kv_m=(-3.0, 3.0),
            continue_selection=True,
        )


def test_branch_loss_to_no_root_keeps_explicit_fallback():
    tracked = continue_ambipolar_branches(
        [
            _branch_surface([-1.0], ["ion"]),
            _branch_surface([], []),
        ],
        surfaces=np.asarray([0.1, 0.2]),
        electric_field_bounds_kv_m=(-3.0, 3.0),
        continue_selection=True,
    )

    assert tracked[1].selected_root is None
    assert tracked[1].selection_reason == "no_bracket_closest_sample"
    assert [event.kind for event in tracked[1].branch_events] == ["loss"]


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


def test_auto_route_recovers_only_failed_points_and_retains_every_attempt(
    monkeypatch, tmp_path, capsys
):
    class FakeOperator:
        def rhs(self):
            return np.asarray([1.0])

    calls = []

    def recover(_problem, values, **kwargs):
        fields = np.asarray(values, dtype=np.float64)
        method = str(kwargs["solve_method"])
        calls.append((method, fields.tolist(), kwargs.get("max_batch")))
        batch = _fake_batch(lambda field: field)(None, fields, solve_method=method)
        if method != "gmres":
            batch.residual_norms[fields == 0.0] = 1.0
        return batch

    monkeypatch.setattr("dkx.batch.batched_er_scan", recover)
    monkeypatch.setattr(
        "dkx.er.operator_at_er", lambda *_args, **_kwargs: FakeOperator()
    )
    result = _solve(
        problem=SimpleNamespace(operator=FakeOperator(), dphi_per_er=1.0),
        previous_root_kv_m=None,
    )

    zero = next(
        evaluation
        for evaluation in result.evaluations
        if evaluation.electric_field_kv_m == 0.0
    )
    assert [attempt.executed_method for attempt in zero.solver_attempts] == [
        "block_tridiagonal_truncated",
        "gmres",
    ]
    assert [attempt.accepted for attempt in zero.solver_attempts] == [False, True]
    assert zero.solver_attempts[1].reason == "automatic_true_residual_recovery"
    assert ("gmres", [0.0], 1) in calls

    from dkx.execution import _ambipolar_result_arrays

    arrays, dimensions = _ambipolar_result_arrays([result], n_species=2)
    assert arrays["evaluation_solver_attempt_count"].max() == 2
    zero_index = np.flatnonzero(arrays["evaluation_electric_field_kV_m"][0] == 0.0)[0]
    np.testing.assert_array_equal(
        arrays["evaluation_solver_attempt_executed_method"][0, zero_index],
        ["block_tridiagonal_truncated", "gmres"],
    )
    assert dimensions["evaluation_solver_attempt_residual"] == (
        "surface",
        "evaluation",
        "solver_attempt",
    )
    native_result = Result(
        case_id="b" * 64,
        case_name="solver attempt evidence",
        workflow="ambipolar_profile",
        arrays=arrays,
        dimensions=dimensions,
        metadata={
            "converged": True,
            "ambipolar_solver_attempts": {
                "attempt_count": sum(
                    len(item.solver_attempts) for item in result.evaluations
                ),
                "executed_route_counts": {
                    "block_tridiagonal_truncated": len(result.evaluations),
                    "gmres": 1,
                },
                "automatic_true_residual_recovery_count": 1,
            },
        },
    )
    loaded = Result.load(native_result.save(tmp_path / "solver-attempts.nc"))
    np.testing.assert_array_equal(
        loaded.evaluation_solver_attempt_executed_method,
        native_result.evaluation_solver_attempt_executed_method,
    )
    np.testing.assert_array_equal(
        loaded.evaluation_solver_attempt_accepted,
        native_result.evaluation_solver_attempt_accepted,
    )
    assert (
        loaded.certificate()["ambipolar_solver_attempts"]
        == (native_result.certificate()["ambipolar_solver_attempts"])
    )
    native_result.print_summary()
    assert "1 automatic true-residual recoveries" in capsys.readouterr().out

    with np.testing.assert_raises_regex(RuntimeError, "did not converge"):
        _solve(
            problem=SimpleNamespace(operator=FakeOperator(), dphi_per_er=1.0),
            solve_method="block_tridiagonal",
            previous_root_kv_m=None,
        )
    assert not any(
        method == "gmres" for method, _fields, _batch in calls[len(calls) - 1 :]
    )


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
