from __future__ import annotations

from types import SimpleNamespace

import pytest

from sfincs_jax.rhs1_handoff import (
    RHS1KSPHandoffState,
    RHS1KSPReplayState,
    rhs1_apply_handoff_to_replay_state,
    rhs1_accept_candidate,
    rhs1_accept_candidate_and_update_replay,
    rhs1_accept_measured_candidate,
    rhs1_accept_measured_candidate_and_update_replay,
    rhs1_residual_improves,
    rhs1_solver_candidate_metrics,
)
from sfincs_jax.solver_selection_policy import SolverCandidateMetrics


def _result(residual_norm: float, x: object = None):
    return SimpleNamespace(residual_norm=residual_norm, x=x)


@pytest.mark.parametrize(
    ("current_residual", "candidate_residual", "expected"),
    [
        (1.0, 0.5, True),
        (1.0, 1.0, False),
        (1.0, 2.0, False),
        (float("nan"), 1.0e-9, True),
        (float("inf"), 1.0e-9, True),
        (1.0, float("nan"), False),
        (1.0, float("inf"), False),
    ],
)
def test_rhs1_residual_improves_is_strict_and_finite(
    current_residual: float,
    candidate_residual: float,
    expected: bool,
) -> None:
    assert (
        rhs1_residual_improves(
            current_residual=current_residual,
            candidate_residual=candidate_residual,
        )
        is expected
    )


def test_rhs1_accept_candidate_accepts_improvement_and_emits_handoff_state() -> None:
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")
    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )
    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert handoff is not None
    assert handoff.matvec_fn == "mv"
    assert handoff.b_vec == "rhs"
    assert handoff.precond_fn == "pc"
    assert handoff.x0_vec == "seed"
    assert handoff.restart == 30
    assert handoff.maxiter == 90
    assert handoff.precond_side == "left"
    assert handoff.solver_kind == "gmres"


def test_rhs1_accept_candidate_rejects_non_improving_result() -> None:
    current = _result(1.0, x="x0")
    candidate = _result(1.0, x="x1")
    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )
    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert handoff is None


def test_rhs1_accept_candidate_accepts_finite_rescue_after_nonfinite_current() -> None:
    current = _result(float("nan"), x="x0")
    candidate = _result(1.0e-10, x="x1")

    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert handoff is not None


def test_rhs1_accept_candidate_rejects_nonfinite_candidate_residual() -> None:
    current = _result(1.0e-6, x="x0")
    candidate = _result(float("nan"), x="x1")

    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert handoff is None


def test_rhs1_accept_candidate_keeps_current_residual_vector_when_candidate_residual_is_missing() -> None:
    current = _result(1.0, x="x0")
    candidate = _result(0.5, x="x1")
    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec=None,
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )
    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert handoff is not None


def test_rhs1_accept_candidate_and_update_replay_updates_only_on_acceptance() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old_mv", solver_kind="old")
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")

    result, residual_vec, accepted = rhs1_accept_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert replay.matvec_fn == "mv"
    assert replay.b_vec == "rhs"
    assert replay.precond_fn == "pc"
    assert replay.x0_vec == "seed"
    assert replay.restart == 30
    assert replay.maxiter == 90
    assert replay.precond_side == "left"
    assert replay.solver_kind == "gmres"

    rejected, rejected_residual_vec, accepted = rhs1_accept_candidate_and_update_replay(
        replay_state=replay,
        current_result=candidate,
        candidate_result=_result(0.5, x="x2"),
        current_residual_vec="r1",
        candidate_residual_vec="r2",
        matvec_fn="new_mv",
        b_vec="new_rhs",
        precond_fn="new_pc",
        x0_vec="new_seed",
        restart=10,
        maxiter=20,
        precond_side="right",
        solver_kind="bicgstab",
    )

    assert not accepted
    assert rejected is candidate
    assert rejected_residual_vec == "r1"
    assert replay.matvec_fn == "mv"
    assert replay.solver_kind == "gmres"


def test_rhs1_accept_candidate_rejects_measured_runtime_memory_regression() -> None:
    current = _result(1.0e-12, x="x0")
    candidate = _result(5.0e-13, x="x1")
    baseline_metrics = SolverCandidateMetrics(
        name="current",
        residual_norm=1.0e-12,
        target=1.0e-9,
        setup_s=0.1,
        solve_s=0.9,
        peak_rss_mb=400.0,
    )
    candidate_metrics = SolverCandidateMetrics(
        name="strong_retry",
        residual_norm=5.0e-13,
        target=1.0e-9,
        setup_s=3.0,
        solve_s=6.0,
        peak_rss_mb=900.0,
    )

    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert handoff is None


def test_rhs1_accept_candidate_allows_slower_candidate_when_baseline_failed() -> None:
    current = _result(5.0e-6, x="x0")
    candidate = _result(5.0e-11, x="x1")
    baseline_metrics = SolverCandidateMetrics(
        name="current",
        residual_norm=5.0e-6,
        target=1.0e-9,
        setup_s=0.1,
        solve_s=0.9,
        peak_rss_mb=400.0,
    )
    candidate_metrics = SolverCandidateMetrics(
        name="strong_retry",
        residual_norm=5.0e-11,
        target=1.0e-9,
        setup_s=3.0,
        solve_s=6.0,
        peak_rss_mb=900.0,
    )

    result, residual_vec, handoff, accepted = rhs1_accept_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert handoff is not None


def test_rhs1_solver_candidate_metrics_extracts_finite_result_fields() -> None:
    metrics = rhs1_solver_candidate_metrics(
        name="sparse_full",
        result=_result(2.5e-10),
        target_value=1.0e-9,
        solve_s=0.25,
        setup_s=0.5,
        peak_rss_mb=123.0,
    )

    assert metrics.name == "sparse_full"
    assert metrics.residual_norm == 2.5e-10
    assert metrics.target == 1.0e-9
    assert metrics.solve_s == 0.25
    assert metrics.setup_s == 0.5
    assert metrics.peak_rss_mb == 123.0
    assert metrics.finite


def test_rhs1_solver_candidate_metrics_marks_unreadable_residual_nonfinite() -> None:
    metrics = rhs1_solver_candidate_metrics(
        name="bad",
        result=object(),
        target_value=1.0e-9,
    )

    assert metrics.residual_norm is None
    assert not metrics.finite


def test_rhs1_accept_measured_candidate_uses_standard_metrics_and_handoff() -> None:
    current = _result(5.0e-6, x="x0")
    candidate = _result(5.0e-11, x="x1")

    result, residual_vec, handoff, accepted = rhs1_accept_measured_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        candidate_name="sparse_full",
        baseline_name="current_full",
        target_value=1.0e-9,
        solve_s=0.25,
        peak_rss_mb=456.0,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert handoff is not None
    assert handoff.solver_kind == "gmres"


def test_rhs1_accept_measured_candidate_and_update_replay_uses_standard_metrics() -> None:
    replay = RHS1KSPReplayState()
    current = _result(5.0e-6, x="x0")
    candidate = _result(5.0e-11, x="x1")

    result, residual_vec, accepted = rhs1_accept_measured_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=50,
        maxiter=100,
        precond_side="left",
        solver_kind="gmres",
        candidate_name="strong",
        baseline_name="current",
        target_value=1.0e-9,
        solve_s=0.25,
        peak_rss_mb=300.0,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert replay.matvec_fn == "mv"
    assert replay.restart == 50
    assert replay.solver_kind == "gmres"


def test_rhs1_accept_measured_candidate_rejects_clean_baseline_without_win() -> None:
    current = _result(1.0e-12, x="x0")
    candidate = _result(5.0e-13, x="x1")

    result, residual_vec, handoff, accepted = rhs1_accept_measured_candidate(
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        candidate_name="sparse_full",
        baseline_name="current_full",
        target_value=1.0e-9,
        solve_s=1.0,
        peak_rss_mb=456.0,
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert handoff is None


def test_rhs1_apply_handoff_to_replay_state_is_noop_for_rejected_candidate() -> None:
    replay = RHS1KSPReplayState(
        matvec_fn="mv0",
        b_vec="rhs0",
        precond_fn="pc0",
        x0_vec="x0",
        restart=10,
        maxiter=20,
        precond_side="right",
        solver_kind="gmres",
    )

    applied = rhs1_apply_handoff_to_replay_state(replay, None)

    assert not applied
    assert replay.matvec_fn == "mv0"
    assert replay.b_vec == "rhs0"
    assert replay.precond_fn == "pc0"
    assert replay.x0_vec == "x0"
    assert replay.restart == 10
    assert replay.maxiter == 20
    assert replay.precond_side == "right"
    assert replay.solver_kind == "gmres"


def test_rhs1_apply_handoff_to_replay_state_updates_all_replay_fields() -> None:
    replay = RHS1KSPReplayState()
    handoff = RHS1KSPHandoffState(
        matvec_fn="mv1",
        b_vec="rhs1",
        precond_fn="pc1",
        x0_vec="x1",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="incremental",
    )

    applied = rhs1_apply_handoff_to_replay_state(replay, handoff)

    assert applied
    assert replay.matvec_fn == "mv1"
    assert replay.b_vec == "rhs1"
    assert replay.precond_fn == "pc1"
    assert replay.x0_vec == "x1"
    assert replay.restart == 30
    assert replay.maxiter == 90
    assert replay.precond_side == "left"
    assert replay.solver_kind == "incremental"
