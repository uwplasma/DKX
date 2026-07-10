from __future__ import annotations

import jax.numpy as jnp

import sfincs_jax.problems.profile_solver_diagnostics as diagnostics


def _identity_matvec(v):
    return v


def _emit_sink(messages):
    def emit(level: int, message: str) -> None:
        messages.append((level, message))

    return emit


def test_newton_krylov_ksp_history_disabled_without_fortran_stdout() -> None:
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_newton_krylov_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(3),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=2,
        precond_side="left",
        emit=_emit_sink(messages),
        fortran_stdout=False,
        max_size=10,
        max_history_iter=100,
    )

    assert history is None
    assert messages == []


def test_newton_krylov_ksp_history_skips_large_system() -> None:
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_newton_krylov_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(4),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=2,
        precond_side="left",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=3,
        max_history_iter=100,
    )

    assert history is None
    assert messages == [(1, "fortran-stdout: KSP history skipped (size=4 > max=3)")]


def test_newton_krylov_ksp_history_skips_iteration_budget() -> None:
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_newton_krylov_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(4),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=4,
        maxiter_val=3,
        precond_side="left",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=10,
        max_history_iter=11,
    )

    assert history is None
    assert messages == [(1, "fortran-stdout: KSP history skipped (estimated_iters=12 > max=11)")]


def test_newton_krylov_ksp_history_emits_trace(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_gmres_solve_with_history_scipy(**kwargs):
        calls.append(kwargs)
        return kwargs["b"], 0.0, [1.0, 0.5, 0.125]

    monkeypatch.setattr(
        diagnostics,
        "gmres_solve_with_history_scipy",
        fake_gmres_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_newton_krylov_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(2),
        precond_fn=None,
        x0_vec=jnp.zeros(2),
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=6,
        maxiter_val=2,
        precond_side="right",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=10,
        max_history_iter=100,
    )

    assert history == [1.0, 0.5, 0.125]
    assert calls[0]["restart"] == 6
    assert calls[0]["precondition_side"] == "right"
    assert messages[:3] == [
        (0, "   0 KSP Residual norm  1.000000000000e+00 "),
        (0, "   1 KSP Residual norm  5.000000000000e-01 "),
        (0, "   2 KSP Residual norm  1.250000000000e-01 "),
    ]
    assert messages[-2] == (0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
    assert messages[-1] == (0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")


def test_newton_krylov_ksp_history_reports_unavailable(monkeypatch) -> None:
    def fail_gmres_solve_with_history_scipy(**_kwargs):
        raise RuntimeError("diagnostic replay failed")

    monkeypatch.setattr(
        diagnostics,
        "gmres_solve_with_history_scipy",
        fail_gmres_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_newton_krylov_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(2),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=4,
        precond_side="left",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=10,
        max_history_iter=100,
    )

    assert history is None
    assert messages == [(1, "fortran-stdout: KSP history unavailable (RuntimeError: diagnostic replay failed)")]
