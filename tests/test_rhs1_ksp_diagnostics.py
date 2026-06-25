from __future__ import annotations

import jax.numpy as jnp

import sfincs_jax.problems.profile_response.ksp_diagnostics as diagnostics


def _identity_matvec(v):
    return v


def _emit_sink(messages):
    def emit(level: int, message: str) -> None:
        messages.append((level, message))

    return emit


def test_rhs1_ksp_diagnostics_controls_preserve_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_FORTRAN_STDOUT", raising=False)
    monkeypatch.delenv("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_KSP_HISTORY_MAX_ITER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_SOLVER_ITER_STATS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", raising=False)

    assert diagnostics.rhs1_fortran_stdout_from_env(emit=None) is False
    assert diagnostics.rhs1_fortran_stdout_from_env(emit=_emit_sink([])) is True
    assert diagnostics.rhs1_ksp_history_limits_from_env() == diagnostics.RHS1KSPHistoryLimits(
        max_size=800,
        max_iter=2000,
    )
    assert diagnostics.rhs1_ksp_iter_stats_controls_from_env() == diagnostics.RHS1KSPIterStatsControls(
        enabled=False,
        max_size=None,
    )
    assert diagnostics.rhs1_ksp_diagnostics_controls_from_env(
        emit=None,
    ) == diagnostics.RHS1KSPDiagnosticsControls(
        fortran_stdout=False,
        history_max_size=800,
        history_max_iter=2000,
        iter_stats_enabled=False,
        iter_stats_max_size=None,
    )


def test_rhs1_ksp_diagnostics_controls_respect_overrides(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "yes")
    monkeypatch.setenv("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "none")
    monkeypatch.setenv("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "77")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "on")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "123")

    assert diagnostics.rhs1_ksp_history_limits_from_env() == diagnostics.RHS1KSPHistoryLimits(
        max_size=None,
        max_iter=77,
    )
    assert diagnostics.rhs1_ksp_iter_stats_controls_from_env() == diagnostics.RHS1KSPIterStatsControls(
        enabled=True,
        max_size=123,
    )
    assert diagnostics.rhs1_ksp_diagnostics_controls_from_env(
        emit=None,
    ) == diagnostics.RHS1KSPDiagnosticsControls(
        fortran_stdout=True,
        history_max_size=None,
        history_max_iter=77,
        iter_stats_enabled=True,
        iter_stats_max_size=123,
    )


def test_rhs1_ksp_diagnostics_controls_handle_disabled_and_invalid(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "off")
    monkeypatch.setenv("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "not-an-int")
    monkeypatch.setenv("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "bad")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "bad")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "bad")

    assert diagnostics.rhs1_ksp_diagnostics_controls_from_env(
        emit=_emit_sink([]),
    ) == diagnostics.RHS1KSPDiagnosticsControls(
        fortran_stdout=False,
        history_max_size=800,
        history_max_iter=2000,
        iter_stats_enabled=False,
        iter_stats_max_size=None,
    )


def test_rhs1_ksp_history_disabled_without_fortran_stdout() -> None:
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_rhs1_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(3),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=2,
        precond_side="left",
        solver_kind="gmres",
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        fortran_stdout=False,
        max_size=10,
        max_history_iter=100,
    )

    assert history is None
    assert messages == []


def test_rhs1_ksp_history_skips_large_system() -> None:
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_rhs1_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(4),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=2,
        precond_side="left",
        solver_kind="gmres",
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=3,
        max_history_iter=100,
    )

    assert history is None
    assert messages == [(1, "fortran-stdout: KSP history skipped (size=4 > max=3)")]


def test_rhs1_ksp_history_emits_gmres_trace(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_gmres_solve_with_history_scipy(**kwargs):
        calls.append(kwargs)
        return kwargs["b"], 0.0, [1.0, 0.25]

    monkeypatch.setattr(
        diagnostics,
        "gmres_solve_with_history_scipy",
        fake_gmres_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    history = diagnostics.emit_rhs1_ksp_history(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(3),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=7,
        maxiter_val=2,
        precond_side="right",
        solver_kind="gmres",
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        fortran_stdout=True,
        max_size=10,
        max_history_iter=100,
    )

    assert history == [1.0, 0.25]
    assert calls[0]["restart"] == 7
    assert calls[0]["precondition_side"] == "right"
    assert messages[0] == (0, "   0 KSP Residual norm  1.000000000000e+00 ")
    assert messages[1] == (0, "   1 KSP Residual norm  2.500000000000e-01 ")
    assert messages[-2] == (0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
    assert messages[-1] == (0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")


def test_rhs1_ksp_iter_stats_reuses_history_without_solver(monkeypatch) -> None:
    def fail_gmres_solve_with_history_scipy(**_kwargs):
        raise AssertionError("history should be reused")

    monkeypatch.setattr(
        diagnostics,
        "gmres_solve_with_history_scipy",
        fail_gmres_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    diagnostics.emit_rhs1_ksp_iter_stats(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(3),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=2,
        precond_side="left",
        solver_kind="gmres",
        history=[1.0, 0.3, 0.01],
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        enabled=True,
        max_size=10,
    )

    assert messages == [(0, "ksp_iterations=3 solver=gmres")]


def test_rhs1_ksp_iter_stats_bicgstab_path(monkeypatch) -> None:
    calls: list[dict] = []

    def fake_bicgstab_solve_with_history_scipy(**kwargs):
        calls.append(kwargs)
        return kwargs["b"], 0.0, [1.0, 0.2, 0.01]

    monkeypatch.setattr(
        diagnostics,
        "bicgstab_solve_with_history_scipy",
        fake_bicgstab_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    diagnostics.emit_rhs1_ksp_iter_stats(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(2),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=4,
        precond_side="left",
        solver_kind="bicgstab",
        history=None,
        solve_method_val="bicgstab",
        emit=_emit_sink(messages),
        enabled=True,
        max_size=10,
    )

    assert len(calls) == 1
    assert calls[0]["maxiter"] == 4
    assert messages == [(0, "ksp_iterations=3 solver=bicgstab")]


def test_rhs1_ksp_iter_stats_reports_unavailable(monkeypatch) -> None:
    def fail_gmres_solve_with_history_scipy(**_kwargs):
        raise RuntimeError("diagnostic replay failed")

    monkeypatch.setattr(
        diagnostics,
        "gmres_solve_with_history_scipy",
        fail_gmres_solve_with_history_scipy,
    )
    messages: list[tuple[int, str]] = []

    diagnostics.emit_rhs1_ksp_iter_stats(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(2),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=4,
        precond_side="left",
        solver_kind="gmres",
        history=None,
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        enabled=True,
        max_size=10,
    )

    assert messages == [(1, "ksp_iterations unavailable (RuntimeError: diagnostic replay failed)")]


def test_rhs1_ksp_iter_stats_skips_estimated_iteration_budget(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER", "3")
    messages: list[tuple[int, str]] = []

    diagnostics.emit_rhs1_ksp_iter_stats(
        matvec_fn=_identity_matvec,
        b_vec=jnp.ones(2),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=2,
        maxiter_val=2,
        precond_side="left",
        solver_kind="gmres",
        history=None,
        solve_method_val="incremental",
        emit=_emit_sink(messages),
        enabled=True,
        max_size=10,
    )

    assert messages == [(1, "ksp_iterations skipped (estimated_iters=4 > max=3)")]
