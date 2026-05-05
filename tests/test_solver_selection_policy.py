from __future__ import annotations

import pytest

from sfincs_jax.solver_selection_policy import (
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
