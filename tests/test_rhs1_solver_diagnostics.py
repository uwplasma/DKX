from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import sfincs_jax.problems.profile_solver_diagnostics as diagnostics
from sfincs_jax.problems.profile_solver_diagnostics import (
    RHS1KSPDiagnosticsContext,
    RHS1PreflightDiagnostics,
    build_profile_response_linear_metadata,
    build_rhs1_xblock_correction_metadata,
    build_rhs1_xblock_correction_metadata_from_solve_state,
    emit_profile_response_ksp_history,
    emit_profile_response_ksp_iter_stats,
    emit_profile_response_ksp_replay_diagnostics,
)


def test_rhs1_xblock_correction_metadata_preserves_historical_keys() -> None:
    metadata = build_rhs1_xblock_correction_metadata(
        preflight=RHS1PreflightDiagnostics(
            min_improvement=1.0e-3,
            required=True,
            residual_norm=3.0,
            improvement=0.2,
            passed=True,
        ),
    )

    assert metadata["xblock_preflight_min_improvement"] == 1.0e-3
    assert metadata["xblock_preflight_required"] is True
    assert metadata["xblock_preflight_residual_norm"] == 3.0
    assert metadata["xblock_preflight_improvement"] == 0.2
    assert metadata["xblock_preflight_passed"] is True


def test_rhs1_xblock_correction_metadata_defaults_are_output_safe() -> None:
    metadata = build_rhs1_xblock_correction_metadata(
        preflight=RHS1PreflightDiagnostics(min_improvement=0.0, required=False),
    )

    assert metadata["xblock_preflight_required"] is False
    assert metadata["xblock_preflight_passed"] is None


def test_profile_response_linear_metadata_merges_parts_without_acceptance() -> None:
    first = {"a": 1, "shared": "first"}
    second = {"b": 2, "shared": "second"}

    metadata = build_profile_response_linear_metadata(
        rhs_mode=2,
        result_residual_norm=1.0e-3,
        rhs=jnp.ones(2),
        tol=1.0e-8,
        atol=0.0,
        metadata_parts=(first, second),
        post_xblock_accept_floor=1.0,
    )

    assert metadata == {"a": 1, "b": 2, "shared": "second"}
    assert first == {"a": 1, "shared": "first"}
    assert second == {"b": 2, "shared": "second"}


def test_profile_response_linear_metadata_marks_post_xblock_acceptance() -> None:
    metadata = build_profile_response_linear_metadata(
        rhs_mode=1,
        result_residual_norm=2.0e-10,
        rhs=jnp.ones(2),
        tol=1.0e-8,
        atol=0.0,
        metadata_parts=({"path": "xblock"},),
        post_xblock_accept_floor=1.0e-9,
    )

    assert metadata["path"] == "xblock"
    assert metadata["accepted_converged"] is True
    assert metadata["acceptance_criterion"] == "post_xblock_abs_floor"
    assert metadata["true_residual_converged"] is True
    assert metadata["accepted_residual_floor"] == 1.0e-9


def test_rhs1_xblock_correction_metadata_from_solve_state_matches_typed_builder() -> None:
    state = {
        "preflight_min_improvement": 1.0e-3,
        "preflight_required": True,
        "preflight_residual_norm": 3.0,
        "preflight_improvement": 0.2,
        "preflight_passed": True,
    }
    expected = build_rhs1_xblock_correction_metadata(
        preflight=RHS1PreflightDiagnostics(
            min_improvement=1.0e-3,
            required=True,
            residual_norm=3.0,
            improvement=0.2,
            passed=True,
        ),
    )

    assert build_rhs1_xblock_correction_metadata_from_solve_state(state) == expected


def test_profile_response_ksp_diagnostics_forward_context(monkeypatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    def fake_history(**kwargs):
        calls["history"] = kwargs
        return [1.0, 0.25]

    def fake_iter_stats(**kwargs):
        calls["iter_stats"] = kwargs

    monkeypatch.setattr(diagnostics, "emit_rhs1_ksp_history", fake_history)
    monkeypatch.setattr(diagnostics, "emit_rhs1_ksp_iter_stats", fake_iter_stats)

    messages: list[tuple[int, str]] = []
    context = RHS1KSPDiagnosticsContext(
        emit=lambda level, msg: messages.append((level, msg)),
        fortran_stdout=True,
        history_max_size=11,
        history_max_iter=13,
        iter_stats_enabled=True,
        iter_stats_max_size=17,
    )
    b_vec = jnp.ones(2)
    history = emit_profile_response_ksp_history(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=b_vec,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=7,
        precond_side="left",
        solver_kind="gmres",
        solve_method_val="incremental",
    )
    emit_profile_response_ksp_iter_stats(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=b_vec,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=0.0,
        restart_val=5,
        maxiter_val=7,
        precond_side="left",
        solver_kind="gmres",
        history=history,
        solve_method_val="incremental",
    )

    assert history == [1.0, 0.25]
    assert calls["history"]["emit"] is context.emit
    assert calls["history"]["fortran_stdout"] is True
    assert calls["history"]["max_size"] == 11
    assert calls["history"]["max_history_iter"] == 13
    assert calls["iter_stats"]["emit"] is context.emit
    assert calls["iter_stats"]["enabled"] is True
    assert calls["iter_stats"]["max_size"] == 17
    assert calls["iter_stats"]["history"] == [1.0, 0.25]


def test_profile_response_ksp_replay_diagnostics_skip_empty_replay(monkeypatch) -> None:
    def fail_history(**_kwargs):
        raise AssertionError("empty replay should not emit history")

    def fail_iter_stats(**_kwargs):
        raise AssertionError("empty replay should not emit iteration stats")

    monkeypatch.setattr(diagnostics, "emit_profile_response_ksp_history", fail_history)
    monkeypatch.setattr(diagnostics, "emit_profile_response_ksp_iter_stats", fail_iter_stats)
    context = RHS1KSPDiagnosticsContext(
        emit=None,
        fortran_stdout=False,
        history_max_size=11,
        history_max_iter=13,
        iter_stats_enabled=False,
        iter_stats_max_size=17,
    )

    assert (
        emit_profile_response_ksp_replay_diagnostics(
            context=context,
            replay_state=SimpleNamespace(matvec_fn=None, b_vec=jnp.ones(2)),
            tol_val=1.0e-8,
            atol_val=0.0,
            solve_method_val="incremental",
        )
        is None
    )


def test_profile_response_ksp_replay_diagnostics_forward_replay_state(monkeypatch) -> None:
    calls: dict[str, dict[str, object]] = {}

    def fake_history(**kwargs):
        calls["history"] = kwargs
        return [1.0, 0.25]

    def fake_iter_stats(**kwargs):
        calls["iter_stats"] = kwargs

    monkeypatch.setattr(diagnostics, "emit_profile_response_ksp_history", fake_history)
    monkeypatch.setattr(diagnostics, "emit_profile_response_ksp_iter_stats", fake_iter_stats)
    context = RHS1KSPDiagnosticsContext(
        emit=None,
        fortran_stdout=True,
        history_max_size=11,
        history_max_iter=13,
        iter_stats_enabled=True,
        iter_stats_max_size=17,
    )
    b_vec = jnp.ones(2)
    x0_vec = jnp.zeros(2)

    def precond(x):
        return x

    replay = SimpleNamespace(
        matvec_fn=lambda x: x,
        b_vec=b_vec,
        precond_fn=precond,
        x0_vec=x0_vec,
        restart="5",
        maxiter=7,
        precond_side="right",
        solver_kind="lgmres",
    )

    history = emit_profile_response_ksp_replay_diagnostics(
        context=context,
        replay_state=replay,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        solve_method_val="incremental",
    )

    assert history == [1.0, 0.25]
    assert calls["history"]["context"] is context
    assert calls["history"]["matvec_fn"] is replay.matvec_fn
    assert calls["history"]["b_vec"] is b_vec
    assert calls["history"]["precond_fn"] is precond
    assert calls["history"]["x0_vec"] is x0_vec
    assert calls["history"]["restart_val"] == 5
    assert calls["history"]["maxiter_val"] == 7
    assert calls["history"]["precond_side"] == "right"
    assert calls["history"]["solver_kind"] == "lgmres"
    assert calls["iter_stats"]["history"] == [1.0, 0.25]
    assert calls["iter_stats"]["tol_val"] == 1.0e-8
    assert calls["iter_stats"]["atol_val"] == 1.0e-12
