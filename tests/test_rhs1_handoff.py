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
    rhs1_accept_sparse_retry_candidate_and_update_replay,
    rhs1_accept_smoother_candidate_and_update_replay,
    rhs1_residual_improves,
    rhs1_retry_without_preconditioner_if_nonfinite,
    rhs1_run_bicgstab_gmres_fallback_if_allowed,
    rhs1_run_collision_retry_if_allowed,
    rhs1_run_fast_post_xblock_polish,
    rhs1_run_linear_candidate_and_update_replay,
    rhs1_run_measured_linear_candidate_and_update_replay,
    rhs1_run_pas_schur_rescue_if_requested,
    rhs1_run_primary_krylov_and_update_replay,
    rhs1_run_stage2_retry_if_allowed,
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


def test_rhs1_accept_smoother_candidate_updates_replay_and_uses_residual_builder() -> None:
    messages: list[tuple[int, str]] = []
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    smoother = SimpleNamespace(
        x="x1",
        residual_norm=0.25,
        accepted_sweeps=2,
        stop_reason="target",
    )

    result, residual_vec, accepted = rhs1_accept_smoother_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        smoother=smoother,
        result_factory=lambda *, x, residual_norm: _result(residual_norm, x=x),
        candidate_residual_vec=lambda candidate: f"residual:{candidate.x}",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        emit=lambda level, message: messages.append((level, message)),
    )

    assert accepted
    assert result.x == "x1"
    assert result.residual_norm == 0.25
    assert residual_vec == "residual:x1"
    assert replay.matvec_fn == "mv"
    assert replay.x0_vec == "x1"
    assert replay.solver_kind == "gmres"
    assert messages == [
        (
            1,
            "solve_v3_full_system_linear_gmres: PAS adaptive smoother "
            "accepted 2 sweep(s), reason=target, residual=1.000e+00->2.500e-01",
        )
    ]


def test_rhs1_accept_smoother_candidate_rejects_nonimproving_without_emitting() -> None:
    messages: list[tuple[int, str]] = []
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")
    smoother = SimpleNamespace(
        x="x1",
        residual_norm=2.0,
        accepted_sweeps=1,
        stop_reason="worse",
    )

    result, residual_vec, accepted = rhs1_accept_smoother_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        smoother=smoother,
        result_factory=lambda *, x, residual_norm: _result(residual_norm, x=x),
        candidate_residual_vec="r1",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        restart=30,
        maxiter=90,
        precond_side="left",
        solver_kind="gmres",
        emit=lambda level, message: messages.append((level, message)),
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert replay.matvec_fn == "old"
    assert messages == []


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


def test_rhs1_accept_sparse_retry_candidate_updates_replay_with_candidate_seed() -> None:
    replay = RHS1KSPReplayState()
    current = _result(5.0e-6, x="x0")
    candidate = _result(5.0e-11, x="x_sparse")

    result, residual_vec, accepted = rhs1_accept_sparse_retry_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r_sparse",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        restart=40,
        maxiter=80,
        precond_side="left",
        solver_kind="incremental",
        candidate_family="sparse_jax",
        scope="full",
        target_value=1.0e-9,
        solve_s=0.5,
        peak_rss_mb=321.0,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r_sparse"
    assert replay.matvec_fn == "mv"
    assert replay.b_vec == "rhs"
    assert replay.precond_fn == "pc"
    assert replay.x0_vec == "x_sparse"
    assert replay.restart == 40
    assert replay.maxiter == 80
    assert replay.precond_side == "left"
    assert replay.solver_kind == "incremental"


def test_rhs1_accept_sparse_retry_candidate_rejects_without_replay_update() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old_mv", solver_kind="old")
    current = _result(1.0e-12, x="x0")
    candidate = _result(5.0e-13, x="x_sparse")

    result, residual_vec, accepted = rhs1_accept_sparse_retry_candidate_and_update_replay(
        replay_state=replay,
        current_result=current,
        candidate_result=candidate,
        current_residual_vec="r0",
        candidate_residual_vec="r_sparse",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        restart=40,
        maxiter=80,
        precond_side="left",
        solver_kind="incremental",
        candidate_family="sparse",
        scope="reduced",
        target_value=1.0e-9,
        solve_s=5.0,
        peak_rss_mb=900.0,
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert replay.matvec_fn == "old_mv"
    assert replay.solver_kind == "old"


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


def test_rhs1_run_fast_post_xblock_polish_accepts_improved_candidate() -> None:
    messages: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")

    def solve_linear(**kwargs):
        calls.append(kwargs)
        return candidate

    result, accepted = rhs1_run_fast_post_xblock_polish(
        current_result=current,
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=solve_linear,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert accepted
    assert result is candidate
    assert calls == [
        {
            "matvec_fn": "mv",
            "b_vec": "rhs",
            "precond_fn": "pc",
            "x0_vec": "x0",
            "tol_val": 1.0e-10,
            "atol_val": 1.0e-12,
            "restart_val": 17,
            "maxiter_val": 33,
            "solve_method_val": "incremental",
            "precond_side": "left",
        }
    ]
    assert messages == [
        (
            1,
            "solve_v3_full_system_linear_gmres: fast post-xblock polish "
            "(restart=17 maxiter=33 residual=1.000e+00)",
        ),
        (
            1,
            "solve_v3_full_system_linear_gmres: fast post-xblock polish improved residual "
            "1.000e+00 -> 2.500e-01",
        ),
    ]


def test_rhs1_run_fast_post_xblock_polish_rejects_nonimproving_candidate() -> None:
    current = _result(1.0, x="x0")
    candidate = _result(2.0, x="x1")

    result, accepted = rhs1_run_fast_post_xblock_polish(
        current_result=current,
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=None,
        precond_side="right",
        solve_linear=lambda **_kwargs: candidate,
        emit=None,
    )

    assert not accepted
    assert result is current


def test_rhs1_run_measured_linear_candidate_accepts_result_only_solver() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(1.0e-11, x="x1")
    calls: list[dict[str, object]] = []

    def solve_linear(**kwargs):
        calls.append(kwargs)
        return candidate

    result, residual_vec, accepted, elapsed_s = (
        rhs1_run_measured_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="pc",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            solve_method="incremental",
            precond_side="left",
            solve_linear=solve_linear,
            solver_kind="gmres",
            candidate_name="stage2_reduced:incremental",
            baseline_name="current_reduced",
            target_value=1.0e-9,
            peak_rss_mb=123.0,
            returns_residual_vec=False,
        )
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert replay.matvec_fn == "mv"
    assert replay.x0_vec == "x1"
    assert replay.restart == 17
    assert replay.solver_kind == "gmres"
    assert calls[0]["x0_vec"] == "x0"
    assert calls[0]["solve_method_val"] == "incremental"
    assert elapsed_s >= 0.0


def test_rhs1_run_linear_candidate_accepts_result_only_solver() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")
    calls: list[dict[str, object]] = []

    def solve_linear(**kwargs):
        calls.append(kwargs)
        return candidate

    result, residual_vec, accepted, elapsed_s = (
        rhs1_run_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="pc",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            solve_method="incremental",
            precond_side="left",
            solve_linear=solve_linear,
            solver_kind="gmres",
            returns_residual_vec=False,
        )
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert replay.matvec_fn == "mv"
    assert replay.x0_vec == "x1"
    assert replay.restart == 17
    assert replay.solver_kind == "gmres"
    assert calls[0]["x0_vec"] == "x0"
    assert calls[0]["solve_method_val"] == "incremental"
    assert elapsed_s >= 0.0


def test_rhs1_run_linear_candidate_accepts_returned_residual_vector() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")

    def solve_linear(**_kwargs):
        return candidate, "r1"

    result, residual_vec, accepted, _elapsed_s = (
        rhs1_run_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn=None,
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=None,
            solve_method="incremental",
            precond_side="none",
            solve_linear=solve_linear,
            solver_kind="gmres",
            returns_residual_vec=True,
        )
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert replay.precond_fn is None
    assert replay.precond_side == "none"


def test_rhs1_run_linear_candidate_rejects_nonimproving_candidate() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")
    candidate = _result(2.0, x="x1")

    result, residual_vec, accepted, _elapsed_s = (
        rhs1_run_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="pc",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            solve_method="incremental",
            precond_side="left",
            solve_linear=lambda **_kwargs: candidate,
            solver_kind="gmres",
            returns_residual_vec=False,
        )
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert replay.matvec_fn == "old"


def test_rhs1_run_pas_schur_rescue_skips_when_controls_disabled() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")

    result, residual_vec, accepted, elapsed_s = rhs1_run_pas_schur_rescue_if_requested(
        replay_state=replay,
        controls=SimpleNamespace(run=False, ratio=10.0, restart=17, maxiter=33),
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        build_preconditioner=lambda: "pc",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: (_result(0.25, x="x1"), "r1"),
        solver_kind="gmres",
        target=1.0e-9,
    )

    assert result is current
    assert residual_vec == "r0"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_run_pas_schur_rescue_accepts_improving_candidate() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")
    messages: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []

    def solve_linear(**kwargs):
        calls.append(kwargs)
        return candidate, "r1"

    result, residual_vec, accepted, elapsed_s = rhs1_run_pas_schur_rescue_if_requested(
        replay_state=replay,
        controls=SimpleNamespace(run=True, ratio=10.0, restart=17, maxiter=33),
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        build_preconditioner=lambda: "pc",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=solve_linear,
        solver_kind="gmres",
        target=1.0e-9,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert elapsed_s >= 0.0
    assert replay.precond_fn == "pc"
    assert replay.x0_vec == "x1"
    assert calls[0]["precond_fn"] == "pc"
    assert any("PAS Schur rescue" in message for _level, message in messages)


def test_rhs1_run_pas_schur_rescue_keeps_current_on_builder_failure() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")
    messages: list[tuple[int, str]] = []

    def build_preconditioner():
        raise RuntimeError("boom")

    result, residual_vec, accepted, elapsed_s = rhs1_run_pas_schur_rescue_if_requested(
        replay_state=replay,
        controls=SimpleNamespace(run=True, ratio=10.0, restart=17, maxiter=33),
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        build_preconditioner=build_preconditioner,
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: (_result(0.25, x="x1"), "r1"),
        solver_kind="gmres",
        target=1.0e-9,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert result is current
    assert residual_vec == "r0"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"
    assert any("PAS Schur rescue failed" in message for _level, message in messages)


def test_rhs1_run_collision_retry_skips_when_not_allowed() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")

    result, residual_vec, precond, accepted, elapsed_s = rhs1_run_collision_retry_if_allowed(
        allowed=False,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="cached",
        build_preconditioner=lambda: "new",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: (_result(0.25, x="x1"), "r1"),
        solver_kind="gmres",
        target=1.0e-9,
        returns_residual_vec=True,
    )

    assert result is current
    assert residual_vec == "r0"
    assert precond == "cached"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_run_bicgstab_gmres_fallback_skips_when_not_allowed() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")

    result, residual_vec, precond, accepted, elapsed_s = (
        rhs1_run_bicgstab_gmres_fallback_if_allowed(
            allowed=False,
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="cached",
            preconditioner_enabled=True,
            build_preconditioner=lambda: "new",
            x0_vec="seed",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            precond_side="left",
            solve_linear=lambda **_kwargs: _result(0.25, x="x1"),
            target=1.0e-9,
            returns_residual_vec=False,
        )
    )

    assert result is current
    assert residual_vec == "r0"
    assert precond == "cached"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_run_bicgstab_gmres_fallback_replaces_result_and_updates_replay() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(2.0, x="x1")
    messages: list[tuple[int, str]] = []
    calls: list[dict[str, object]] = []

    def solve_linear(**kwargs):
        calls.append(kwargs)
        return candidate, "r1"

    result, residual_vec, precond, accepted, elapsed_s = (
        rhs1_run_bicgstab_gmres_fallback_if_allowed(
            allowed=True,
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn=None,
            preconditioner_enabled=True,
            build_preconditioner=lambda: "built",
            x0_vec="seed",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            precond_side="left",
            solve_linear=solve_linear,
            target=1.0e-9,
            returns_residual_vec=True,
            emit=lambda level, message: messages.append((level, message)),
        )
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert precond == "built"
    assert elapsed_s >= 0.0
    assert replay.matvec_fn == "mv"
    assert replay.precond_fn == "built"
    assert replay.x0_vec == "seed"
    assert replay.solver_kind == "gmres"
    assert calls[0]["solve_method_val"] == "incremental"
    assert calls[0]["x0_vec"] == "seed"
    assert any("BiCGStab fallback to GMRES" in message for _level, message in messages)


def test_rhs1_run_bicgstab_gmres_fallback_reuses_cached_preconditioner_and_ready_hook() -> None:
    replay = RHS1KSPReplayState()
    current = _result(float("nan"), x="x0")
    candidate = _result(0.5, x="x1")
    ready_seen: list[object] = []

    def build_preconditioner():
        raise AssertionError("cached preconditioner should be reused")

    def result_ready(result):
        ready_seen.append(result)
        return _result(result.residual_norm, x="ready")

    result, residual_vec, precond, accepted, _elapsed_s = (
        rhs1_run_bicgstab_gmres_fallback_if_allowed(
            allowed=True,
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="cached",
            preconditioner_enabled=True,
            build_preconditioner=build_preconditioner,
            x0_vec="seed",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            precond_side="left",
            solve_linear=lambda **_kwargs: candidate,
            target=1.0e-9,
            returns_residual_vec=False,
            result_ready=result_ready,
        )
    )

    assert accepted
    assert result.x == "ready"
    assert residual_vec == "r0"
    assert precond == "cached"
    assert ready_seen == [candidate]
    assert replay.precond_fn == "cached"


def test_rhs1_run_primary_krylov_updates_replay_without_resetting_controls() -> None:
    replay = RHS1KSPReplayState(restart=99, maxiter=101, solver_kind="old")
    candidate = _result(0.25, x="x1")
    events: list[str] = []

    def solve_linear(**kwargs):
        events.append("solve")
        assert kwargs["precond_fn"] == "pc"
        assert kwargs["x0_vec"] == "seed"
        assert kwargs["solve_method_val"] == "incremental"
        return candidate

    def result_ready(result):
        events.append("ready")
        return _result(result.residual_norm, x="ready")

    result, residual_vec, elapsed_s = rhs1_run_primary_krylov_and_update_replay(
        replay_state=replay,
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="pc",
        x0_vec="seed",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        solve_method="incremental",
        precond_side="left",
        solve_linear=solve_linear,
        solver_kind="gmres",
        returns_residual_vec=False,
        current_residual_vec="r0",
        result_ready=result_ready,
        progress_start=lambda: events.append("progress"),
        mark=lambda name: events.append(name),
        mark_start="start",
        mark_done="done",
    )

    assert result.x == "ready"
    assert residual_vec == "r0"
    assert elapsed_s >= 0.0
    assert events == ["progress", "start", "solve", "ready", "done"]
    assert replay.matvec_fn == "mv"
    assert replay.b_vec == "rhs"
    assert replay.precond_fn == "pc"
    assert replay.x0_vec == "seed"
    assert replay.precond_side == "left"
    assert replay.solver_kind == "gmres"
    assert replay.restart == 99
    assert replay.maxiter == 101


def test_rhs1_run_primary_krylov_blocks_returned_residual_when_requested() -> None:
    replay = RHS1KSPReplayState()
    candidate = _result(0.25, x="x1")
    block_calls: list[str] = []
    residual = SimpleNamespace(block_until_ready=lambda: block_calls.append("ready"))

    result, residual_vec, _elapsed_s = rhs1_run_primary_krylov_and_update_replay(
        replay_state=replay,
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn=None,
        x0_vec=None,
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=None,
        solve_method="incremental",
        precond_side="none",
        solve_linear=lambda **_kwargs: (candidate, residual),
        solver_kind="gmres",
        returns_residual_vec=True,
        block_residual_until_ready=True,
    )

    assert result is candidate
    assert residual_vec is residual
    assert block_calls == ["ready"]
    assert replay.precond_fn is None


def test_rhs1_retry_without_preconditioner_skips_finite_or_disabled() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(0.25, x="x0")

    def finite(result):
        return result.residual_norm == 0.25

    result, residual_vec, accepted, elapsed_s = rhs1_retry_without_preconditioner_if_nonfinite(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        x0_vec="seed",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        solve_method="incremental",
        precond_side="left",
        solve_linear=lambda **_kwargs: _result(0.1, x="x1"),
        solver_kind="gmres",
        result_is_finite=finite,
        returns_residual_vec=False,
    )

    assert result is current
    assert residual_vec == "r0"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_retry_without_preconditioner_runs_nonfinite_retry() -> None:
    replay = RHS1KSPReplayState(restart=99, maxiter=101)
    current = _result(float("nan"), x="x0")
    candidate = _result(0.1, x="x1")
    messages: list[tuple[int, str]] = []
    marks: list[str] = []

    def solve_linear(**kwargs):
        assert kwargs["precond_fn"] is None
        assert kwargs["x0_vec"] == "seed"
        return candidate, "r1"

    result, residual_vec, accepted, elapsed_s = rhs1_retry_without_preconditioner_if_nonfinite(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        x0_vec="seed",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        solve_method="incremental",
        precond_side="left",
        solve_linear=solve_linear,
        solver_kind="gmres",
        result_is_finite=lambda result: result.residual_norm == result.residual_norm,
        returns_residual_vec=True,
        mark=lambda name: marks.append(name),
        mark_start="retry_start",
        mark_done="retry_done",
        emit=lambda level, message: messages.append((level, message)),
        message="retry message",
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert elapsed_s >= 0.0
    assert replay.precond_fn is None
    assert replay.x0_vec == "seed"
    assert replay.restart == 99
    assert replay.maxiter == 101
    assert marks == ["retry_start", "retry_done"]
    assert messages == [(0, "retry message")]


def test_rhs1_run_collision_retry_reuses_cached_preconditioner() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")
    messages: list[tuple[int, str]] = []

    def build_preconditioner():
        raise AssertionError("cached preconditioner should be reused")

    result, residual_vec, precond, accepted, elapsed_s = rhs1_run_collision_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="cached",
        build_preconditioner=build_preconditioner,
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: (candidate, "r1"),
        solver_kind="gmres",
        target=1.0e-9,
        returns_residual_vec=True,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert precond == "cached"
    assert elapsed_s >= 0.0
    assert replay.precond_fn == "cached"
    assert any("collision preconditioner" in message for _level, message in messages)


def test_rhs1_run_collision_retry_builds_and_returns_preconditioner() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(0.25, x="x1")

    result, residual_vec, precond, accepted, _elapsed_s = rhs1_run_collision_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn=None,
        build_preconditioner=lambda: "built",
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: candidate,
        solver_kind="gmres",
        target=1.0e-9,
        returns_residual_vec=False,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert precond == "built"
    assert replay.precond_fn == "built"


def test_rhs1_run_collision_retry_skips_when_builder_returns_none() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")

    result, residual_vec, precond, accepted, elapsed_s = rhs1_run_collision_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn=None,
        build_preconditioner=lambda: None,
        tol=1.0e-10,
        atol=1.0e-12,
        restart=17,
        maxiter=33,
        precond_side="left",
        solve_linear=lambda **_kwargs: (_result(0.25, x="x1"), "r1"),
        solver_kind="gmres",
        target=1.0e-9,
        returns_residual_vec=True,
    )

    assert result is current
    assert residual_vec == "r0"
    assert precond is None
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_run_stage2_retry_skips_when_not_allowed() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")

    result, residual_vec, precond, accepted, elapsed_s = rhs1_run_stage2_retry_if_allowed(
        allowed=False,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="cached",
        preconditioner_enabled=True,
        build_preconditioner=lambda: "new",
        controls=SimpleNamespace(restart=17, maxiter=33, method="incremental"),
        tol=1.0e-10,
        atol=1.0e-12,
        precond_side="left",
        solve_linear=lambda **_kwargs: (_result(0.25, x="x1"), "r1"),
        solver_kind="gmres",
        candidate_name="stage2_full:incremental",
        baseline_name="current_full",
        target=1.0e-9,
        peak_rss_mb=lambda: 12.0,
        returns_residual_vec=True,
    )

    assert result is current
    assert residual_vec == "r0"
    assert precond == "cached"
    assert not accepted
    assert elapsed_s == 0.0
    assert replay.matvec_fn == "old"


def test_rhs1_run_stage2_retry_builds_preconditioner_and_samples_rss_before_solve() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(1.0e-11, x="x1")
    order: list[str] = []
    messages: list[tuple[int, str]] = []

    def build_preconditioner():
        order.append("build")
        return "built"

    def peak_rss_mb():
        order.append("rss")
        return 42.0

    def solve_linear(**kwargs):
        order.append("solve")
        assert kwargs["precond_fn"] == "built"
        assert kwargs["restart_val"] == 17
        assert kwargs["maxiter_val"] == 33
        return candidate, "r1"

    result, residual_vec, precond, accepted, elapsed_s = rhs1_run_stage2_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn=None,
        preconditioner_enabled=True,
        build_preconditioner=build_preconditioner,
        controls=SimpleNamespace(restart=17, maxiter=33, method="incremental"),
        tol=1.0e-10,
        atol=1.0e-12,
        precond_side="left",
        solve_linear=solve_linear,
        solver_kind="gmres",
        candidate_name="stage2_full:incremental",
        baseline_name="current_full",
        target=1.0e-9,
        peak_rss_mb=peak_rss_mb,
        returns_residual_vec=True,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert precond == "built"
    assert elapsed_s >= 0.0
    assert order == ["build", "rss", "solve"]
    assert replay.precond_fn == "built"
    assert replay.restart == 17
    assert any("stage2 GMRES" in message for _level, message in messages)


def test_rhs1_run_stage2_retry_reuses_cached_preconditioner() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(1.0e-11, x="x1")

    def build_preconditioner():
        raise AssertionError("cached preconditioner should be reused")

    result, residual_vec, precond, accepted, _elapsed_s = rhs1_run_stage2_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn="cached",
        preconditioner_enabled=True,
        build_preconditioner=build_preconditioner,
        controls=SimpleNamespace(restart=17, maxiter=33, method="incremental"),
        tol=1.0e-10,
        atol=1.0e-12,
        precond_side="left",
        solve_linear=lambda **_kwargs: candidate,
        solver_kind="gmres",
        candidate_name="stage2_reduced:incremental",
        baseline_name="current_reduced",
        target=1.0e-9,
        peak_rss_mb=12.0,
        returns_residual_vec=False,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert precond == "cached"
    assert replay.precond_fn == "cached"


def test_rhs1_run_stage2_retry_allows_unpreconditioned_retry_when_disabled() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(1.0e-11, x="x1")

    def build_preconditioner():
        raise AssertionError("preconditioner disabled")

    def solve_linear(**kwargs):
        assert kwargs["precond_fn"] is None
        return candidate

    result, residual_vec, precond, accepted, _elapsed_s = rhs1_run_stage2_retry_if_allowed(
        allowed=True,
        replay_state=replay,
        current_result=current,
        current_residual_vec="r0",
        matvec_fn="mv",
        b_vec="rhs",
        precond_fn=None,
        preconditioner_enabled=False,
        build_preconditioner=build_preconditioner,
        controls=SimpleNamespace(restart=17, maxiter=33, method="incremental"),
        tol=1.0e-10,
        atol=1.0e-12,
        precond_side="none",
        solve_linear=solve_linear,
        solver_kind="gmres",
        candidate_name="stage2_reduced:incremental",
        baseline_name="current_reduced",
        target=1.0e-9,
        peak_rss_mb=12.0,
        returns_residual_vec=False,
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r0"
    assert precond is None
    assert replay.precond_fn is None
    assert replay.precond_side == "none"


def test_rhs1_run_measured_linear_candidate_accepts_returned_residual_vector() -> None:
    replay = RHS1KSPReplayState()
    current = _result(1.0, x="x0")
    candidate = _result(1.0e-11, x="x1")

    def solve_linear(**_kwargs):
        return candidate, "r1"

    result, residual_vec, accepted, _elapsed_s = (
        rhs1_run_measured_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn=None,
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=None,
            solve_method="incremental",
            precond_side="none",
            solve_linear=solve_linear,
            solver_kind="gmres",
            candidate_name="stage2_full:incremental",
            baseline_name="current_full",
            target_value=1.0e-9,
            returns_residual_vec=True,
        )
    )

    assert accepted
    assert result is candidate
    assert residual_vec == "r1"
    assert replay.precond_fn is None
    assert replay.precond_side == "none"


def test_rhs1_run_measured_linear_candidate_rejects_nonimproving_candidate() -> None:
    replay = RHS1KSPReplayState(matvec_fn="old")
    current = _result(1.0, x="x0")
    candidate = _result(2.0, x="x1")

    result, residual_vec, accepted, _elapsed_s = (
        rhs1_run_measured_linear_candidate_and_update_replay(
            replay_state=replay,
            current_result=current,
            current_residual_vec="r0",
            matvec_fn="mv",
            b_vec="rhs",
            precond_fn="pc",
            tol=1.0e-10,
            atol=1.0e-12,
            restart=17,
            maxiter=33,
            solve_method="incremental",
            precond_side="left",
            solve_linear=lambda **_kwargs: candidate,
            solver_kind="gmres",
            candidate_name="stage2_reduced:incremental",
            baseline_name="current_reduced",
            target_value=1.0e-9,
            returns_residual_vec=False,
        )
    )

    assert not accepted
    assert result is current
    assert residual_vec == "r0"
    assert replay.matvec_fn == "old"
