from __future__ import annotations

import pytest

from sfincs_jax.solvers.path_policy import (
    SolverAcceptanceCriteria,
    SolverCandidateMetrics,
    choose_solver_candidate,
    solver_candidate_gate,
)


def test_rejects_candidate_that_is_not_numerically_clean() -> None:
    candidate = SolverCandidateMetrics(
        name="pas_lite",
        residual_norm=1.9e-2,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=4.0,
        peak_rss_mb=1500.0,
    )

    gate = solver_candidate_gate(candidate)

    assert not gate.accepted
    assert "residual_not_clean" in gate.reasons
    assert gate.residual_ratio == 1.9e7


def test_rejects_pathological_auto_promotion_like_nxi_cliff() -> None:
    baseline = SolverCandidateMetrics(
        name="forced_fast_path",
        residual_norm=5.0e-11,
        target=1.0e-9,
        setup_s=2.0,
        solve_s=18.0,
        peak_rss_mb=1400.0,
    )
    candidate = SolverCandidateMetrics(
        name="auto_strong_fallback",
        residual_norm=1.9e-2,
        target=1.0e-9,
        setup_s=30.0,
        solve_s=390.0,
        peak_rss_mb=1900.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert not gate.accepted
    assert "residual_not_clean" in gate.reasons
    assert "runtime_regression" in gate.reasons
    assert "memory_regression" in gate.reasons
    assert gate.runtime_ratio == 21.0
    assert round(gate.memory_ratio or 0.0, 3) == 1.357


def test_accepts_correct_fallback_when_baseline_failed() -> None:
    baseline = SolverCandidateMetrics(
        name="weak_default",
        residual_norm=5.0e-2,
        target=1.0e-9,
        setup_s=0.5,
        solve_s=5.0,
        peak_rss_mb=800.0,
    )
    candidate = SolverCandidateMetrics(
        name="sparse_lu_rescue",
        residual_norm=2.0e-12,
        target=1.0e-9,
        setup_s=10.0,
        solve_s=20.0,
        peak_rss_mb=1200.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert gate.accepted
    assert gate.reasons == ()
    assert gate.runtime_ratio and gate.runtime_ratio > 1.0
    assert gate.memory_ratio and gate.memory_ratio > 1.0


def test_requires_measured_win_when_baseline_is_already_clean() -> None:
    baseline = SolverCandidateMetrics(
        name="current_default",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=9.0,
        peak_rss_mb=1000.0,
    )
    candidate = SolverCandidateMetrics(
        name="different_but_not_better",
        residual_norm=2.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=9.2,
        peak_rss_mb=990.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert not gate.accepted
    assert "no_measured_promotion_win" in gate.reasons


def test_accepts_runtime_or_memory_improvement_against_clean_baseline() -> None:
    baseline = SolverCandidateMetrics(
        name="current_default",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=9.0,
        peak_rss_mb=1000.0,
    )
    faster = SolverCandidateMetrics(
        name="faster",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=7.0,
        peak_rss_mb=1000.0,
    )
    lower_memory = SolverCandidateMetrics(
        name="lower_memory",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=9.0,
        peak_rss_mb=800.0,
    )

    assert solver_candidate_gate(faster, baseline=baseline).accepted
    assert solver_candidate_gate(lower_memory, baseline=baseline).accepted


def test_solver_candidate_gate_compares_paired_active_memory() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-10,
        setup_s=1.0,
        solve_s=1.0,
        peak_rss_mb=900.0,
        active_rss_mb=300.0,
    )
    candidate = SolverCandidateMetrics(
        name="candidate",
        residual_norm=1.0e-12,
        target=1.0e-10,
        setup_s=1.0,
        solve_s=1.0,
        peak_rss_mb=950.0,
        active_rss_mb=200.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert gate.accepted
    assert gate.memory_metric == "active_rss_mb"
    assert gate.memory_ratio == pytest.approx(2.0 / 3.0)


def test_solver_candidate_gate_uses_device_memory_before_rss() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-10,
        setup_s=1.0,
        solve_s=1.0,
        peak_rss_mb=900.0,
        active_rss_mb=300.0,
        device_peak_mb=500.0,
    )
    candidate = SolverCandidateMetrics(
        name="candidate",
        residual_norm=1.0e-12,
        target=1.0e-10,
        setup_s=1.0,
        solve_s=1.0,
        peak_rss_mb=800.0,
        active_rss_mb=200.0,
        device_peak_mb=400.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert gate.accepted
    assert gate.memory_metric == "device_peak_mb"
    assert gate.memory_ratio == pytest.approx(0.8)


def test_choose_solver_candidate_prefers_fastest_accepted_path() -> None:
    baseline = SolverCandidateMetrics(
        name="failed_default",
        residual_norm=1.0e-2,
        target=1.0e-9,
        setup_s=0.5,
        solve_s=2.0,
        peak_rss_mb=600.0,
    )
    candidates = [
        SolverCandidateMetrics(
            name="invalid_fast",
            residual_norm=1.0e-3,
            target=1.0e-9,
            setup_s=0.1,
            solve_s=0.2,
            peak_rss_mb=500.0,
        ),
        SolverCandidateMetrics(
            name="robust_slow",
            residual_norm=1.0e-12,
            target=1.0e-9,
            setup_s=5.0,
            solve_s=30.0,
            peak_rss_mb=1000.0,
        ),
        SolverCandidateMetrics(
            name="robust_fast",
            residual_norm=1.0e-12,
            target=1.0e-9,
            setup_s=2.0,
            solve_s=10.0,
            peak_rss_mb=1200.0,
        ),
    ]

    selected = choose_solver_candidate(candidates, baseline=baseline)

    assert selected is not None
    assert selected.name == "robust_fast"


def test_stricter_criteria_can_block_promotion_regressions() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=9.0,
        peak_rss_mb=1000.0,
    )
    candidate = SolverCandidateMetrics(
        name="lower_memory_but_too_slow",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=20.0,
        peak_rss_mb=700.0,
    )
    criteria = SolverAcceptanceCriteria(max_runtime_factor_vs_baseline=1.5)

    gate = solver_candidate_gate(candidate, baseline=baseline, criteria=criteria)

    assert not gate.accepted
    assert "runtime_regression" in gate.reasons


def test_total_time_and_residual_ratio_ignore_nonfinite_measurements() -> None:
    candidate = SolverCandidateMetrics(
        name="partial_metrics",
        residual_norm=float("nan"),
        target=1.0e-9,
        setup_s=float("inf"),
        solve_s=3.0,
    )

    assert candidate.total_s == 3.0
    assert candidate.residual_ratio is None


def test_total_time_and_residual_ratio_return_none_without_finite_inputs() -> None:
    candidate = SolverCandidateMetrics(
        name="empty_metrics",
        residual_norm=1.0e-12,
        target=0.0,
        setup_s=None,
        solve_s=float("nan"),
    )

    assert candidate.total_s is None
    assert candidate.residual_ratio is None


def test_gate_reports_nonfinite_and_parity_failures_before_promotion() -> None:
    candidate = SolverCandidateMetrics(
        name="bad_candidate",
        residual_norm=1.0e-12,
        target=1.0e-9,
        finite=False,
        parity_failures=2,
    )

    gate = solver_candidate_gate(candidate)

    assert not gate.accepted
    assert "nonfinite_candidate" in gate.reasons
    assert "parity_failures" in gate.reasons


def test_failed_baseline_can_require_runtime_and_memory_measurements() -> None:
    baseline = SolverCandidateMetrics(
        name="failed_baseline",
        residual_norm=1.0e-2,
        target=1.0e-9,
    )
    candidate = SolverCandidateMetrics(
        name="clean_but_unmeasured",
        residual_norm=1.0e-12,
        target=1.0e-9,
    )
    criteria = SolverAcceptanceCriteria(
        allow_unknown_runtime_when_baseline_failed=False,
        allow_unknown_memory_when_baseline_failed=False,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline, criteria=criteria)

    assert not gate.accepted
    assert "missing_runtime" in gate.reasons
    assert "missing_memory" in gate.reasons


def test_clean_baseline_with_zero_runtime_does_not_invent_speedup() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=0.0,
        solve_s=0.0,
        peak_rss_mb=100.0,
    )
    candidate = SolverCandidateMetrics(
        name="candidate",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=0.0,
        solve_s=0.0,
        peak_rss_mb=100.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert not gate.accepted
    assert gate.runtime_ratio is None
    assert gate.memory_ratio == pytest.approx(1.0)
    assert "no_measured_promotion_win" in gate.reasons


def test_memory_gate_falls_back_to_compiled_temp_before_peak_rss() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=1.0,
        compiled_temp_mb=900.0,
        peak_rss_mb=100.0,
    )
    candidate = SolverCandidateMetrics(
        name="candidate",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=1.0,
        compiled_temp_mb=600.0,
        peak_rss_mb=10_000.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert gate.accepted
    assert gate.memory_metric == "compiled_temp_mb"
    assert gate.memory_ratio == pytest.approx(2.0 / 3.0)


def test_memory_gate_reports_no_paired_metric_when_measurements_do_not_overlap() -> None:
    baseline = SolverCandidateMetrics(
        name="baseline",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=1.0,
        device_peak_mb=400.0,
    )
    candidate = SolverCandidateMetrics(
        name="candidate",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=1.0,
        solve_s=1.0,
        peak_rss_mb=300.0,
    )

    gate = solver_candidate_gate(candidate, baseline=baseline)

    assert not gate.accepted
    assert gate.memory_metric is None
    assert gate.memory_ratio is None
    assert "no_measured_promotion_win" in gate.reasons


def test_choose_solver_candidate_tie_breaks_by_memory_then_name() -> None:
    candidates = [
        SolverCandidateMetrics(
            name="zeta",
            residual_norm=1.0e-12,
            target=1.0e-9,
            setup_s=1.0,
            solve_s=1.0,
            active_rss_mb=400.0,
        ),
        SolverCandidateMetrics(
            name="alpha",
            residual_norm=1.0e-12,
            target=1.0e-9,
            setup_s=1.0,
            solve_s=1.0,
            active_rss_mb=300.0,
        ),
    ]

    selected = choose_solver_candidate(candidates)

    assert selected is not None
    assert selected.name == "alpha"


def test_choose_solver_candidate_returns_none_when_all_rejected() -> None:
    selected = choose_solver_candidate(
        [
            SolverCandidateMetrics(
                name="dirty",
                residual_norm=1.0e-3,
                target=1.0e-9,
            )
        ]
    )

    assert selected is None
