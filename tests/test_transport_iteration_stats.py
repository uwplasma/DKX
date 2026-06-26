from __future__ import annotations

import jax.numpy as jnp

import sfincs_jax.problems.transport_matrix.solve as iteration_stats
from sfincs_jax.problems.transport_matrix.solve import emit_transport_ksp_iteration_stats


def test_transport_iteration_stats_skips_when_disabled(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        iteration_stats,
        "gmres_solve_with_history_scipy",
        lambda **_kwargs: calls.append("gmres"),
    )
    emitted: list[str] = []

    emit_transport_ksp_iteration_stats(
        which_rhs=1,
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        precond_fn=None,
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        precond_side="left",
        solver_kind="gmres",
        emit=lambda _level, message: emitted.append(str(message)),
        enabled=False,
        max_size=None,
    )

    assert calls == []
    assert emitted == []


def test_transport_iteration_stats_respects_max_size() -> None:
    emitted: list[str] = []

    emit_transport_ksp_iteration_stats(
        which_rhs=2,
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((3,)),
        precond_fn=None,
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        precond_side="left",
        solver_kind="gmres",
        emit=lambda _level, message: emitted.append(str(message)),
        enabled=True,
        max_size=2,
    )

    assert emitted == ["whichRHS=2 ksp_iterations skipped (size=3 > max=2)"]


def test_transport_iteration_stats_emits_gmres_history(monkeypatch) -> None:
    captured = {}

    def fake_gmres(**kwargs):
        captured.update(kwargs)
        return jnp.zeros((2,)), 0.0, [1.0, 0.5, 0.25]

    monkeypatch.setattr(iteration_stats, "gmres_solve_with_history_scipy", fake_gmres)
    emitted: list[tuple[int, str]] = []

    emit_transport_ksp_iteration_stats(
        which_rhs=3,
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        precond_fn=None,
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=11,
        maxiter_val=22,
        precond_side="right",
        solver_kind="GMRES",
        emit=lambda level, message: emitted.append((int(level), str(message))),
        enabled=True,
        max_size=None,
    )

    assert captured["restart"] == 11
    assert captured["maxiter"] == 22
    assert captured["precondition_side"] == "right"
    assert emitted == [(0, "whichRHS=3 ksp_iterations=3 solver=gmres")]


def test_transport_iteration_stats_reports_solver_failures(monkeypatch) -> None:
    def fake_bicgstab(**_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(iteration_stats, "bicgstab_solve_with_history_scipy", fake_bicgstab)
    emitted: list[str] = []

    emit_transport_ksp_iteration_stats(
        which_rhs=4,
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        precond_fn=None,
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        precond_side="left",
        solver_kind="bicgstab",
        emit=lambda _level, message: emitted.append(str(message)),
        enabled=True,
        max_size=None,
    )

    assert emitted == ["whichRHS=4 ksp_iterations unavailable (RuntimeError: boom)"]
