from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_handoff import rhs1_accept_candidate
from sfincs_jax.solver_selection_policy import SolverCandidateMetrics


def _result(residual_norm: float, x: object = None):
    return SimpleNamespace(residual_norm=residual_norm, x=x)


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
