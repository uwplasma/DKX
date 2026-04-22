from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_handoff import rhs1_accept_candidate


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
