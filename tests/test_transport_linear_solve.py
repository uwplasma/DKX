from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
import sfincs_jax.problems.transport_linear_system as transport_linear_system
import sfincs_jax.problems.transport_solve as transport_linear
from sfincs_jax.problems.transport_policies import (
    TransportActiveDOFDecision,
    TransportDensePolicy,
    TransportInitialSolvePolicy,
)
from sfincs_jax.problems.transport_linear_system import (
    TransportLinearSolveCallbacks,
    TransportLinearSolveContext,
    _active_dof_notes,
    _dense_policy_notes,
    solve_transport_linear,
    solve_transport_linear_with_residual,
    transport_restart_for_method,
    transport_solver_kind,
)
from sfincs_jax.problems.transport_solve import (
    TransportDenseBatchContext,
    TransportLoopProgress,
    TransportMatvecCache,
    TransportRecycleState,
    dense_preconditioner_for_matvec,
    dense_solver_for_matvec,
    emit_transport_ksp_iteration_stats,
    resolve_transport_recycle_k,
    solve_transport_dense_batch,
    transport_host_gmres_solve,
)


def _context(**overrides) -> TransportLinearSolveContext:
    values = {
        "rhs_mode": 2,
        "size_hint": 5,
        "use_implicit": False,
        "use_solver_jit": False,
        "distributed_axis": None,
    }
    values.update(overrides)
    return TransportLinearSolveContext(**values)


def test_transport_solver_kind_and_restart_policy() -> None:
    assert transport_solver_kind("auto", rhs_mode=2) == ("bicgstab", "batched")
    assert transport_solver_kind("bicgstab_jax", rhs_mode=3) == ("bicgstab", "batched")
    assert transport_solver_kind("default", rhs_mode=1) == ("bicgstab", "batched")
    assert transport_solver_kind("incremental", rhs_mode=2) == ("gmres", "incremental")

    assert transport_restart_for_method("auto", rhs_mode=2, gmres_restart=30, restart=80) == 80
    assert transport_restart_for_method("incremental", rhs_mode=2, gmres_restart=30, restart=80) == 30


def test_transport_active_dof_notes_explain_disabled_hint_and_enabled_size() -> None:
    op = SimpleNamespace(total_size=128)
    disabled = TransportActiveDOFDecision(
        use_active_dof_mode=False,
        reason=None,
        solve_method_use="incremental",
        emit_disabled_hint=True,
    )
    enabled = TransportActiveDOFDecision(
        use_active_dof_mode=True,
        reason="production-floor active subset",
        solve_method_use="incremental",
        emit_disabled_hint=False,
    )

    disabled_notes = _active_dof_notes(
        op=op,
        active_dof_decision=disabled,
        active_size=128,
    )
    enabled_notes = _active_dof_notes(
        op=op,
        active_dof_decision=enabled,
        active_size=96,
    )

    assert disabled_notes == (
        (
            1,
            "solve_v3_transport_matrix_linear_gmres: active-DOF mode disabled "
            "(set SFINCS_JAX_TRANSPORT_ACTIVE_DOF=1 to enable; "
            "SFINCS_JAX_TRANSPORT_ACTIVE_DOF=0 to force full-size solve)",
        ),
    )
    assert enabled_notes == (
        (
            1,
            "solve_v3_transport_matrix_linear_gmres: active-DOF mode enabled "
            "(size=96/128) (production-floor active subset)",
        ),
    )


def _initial_transport_policy(*, dense_mem_block: bool, dense_use_mixed: bool) -> TransportInitialSolvePolicy:
    return TransportInitialSolvePolicy(
        geometry_scheme=5,
        low_memory_outputs=False,
        stream_diagnostics=False,
        store_state_vectors=False,
        solve_method_use="auto",
        force_krylov=False,
        force_dense=False,
        dense_fallback=True,
        dense_fallback_max=1000,
        dense_retry_max=1000,
        dense_mem_max_mb=32.0,
        dense_mem_est_mb32=16.0,
        dense_mem_est_mb64=64.0,
        dense_mem_block=bool(dense_mem_block),
        dense_use_mixed=bool(dense_use_mixed),
        dense_backend_allowed=True,
        dense_accelerator_auto_allowed=False,
        gmres_restart=40,
        maxiter=80,
    )


def _dense_transport_policy(
    *,
    solve_method_use: str,
    dense_mem_block: bool,
    dense_use_mixed: bool,
    force_dense: bool = False,
    dense_precond_mem_block: bool = False,
) -> TransportDensePolicy:
    return TransportDensePolicy(
        solve_method_use=str(solve_method_use),
        dense_fallback=True,
        dense_retry_max=1000,
        dense_mem_block=bool(dense_mem_block),
        dense_use_mixed=bool(dense_use_mixed),
        force_dense=bool(force_dense),
        dense_precond_enabled=not bool(dense_precond_mem_block),
        dense_precond_mem_block=bool(dense_precond_mem_block),
        dense_precond_est_mb=128.0,
        dense_precond_mem_max_mb=64.0,
        dense_mem_est_active_mb32=48.0,
        dense_mem_est_active_mb64=96.0,
    )


def test_transport_dense_policy_notes_cover_memory_mixed_and_preconditioner_messages() -> None:
    blocked = _dense_policy_notes(
        rhs_mode=2,
        solve_method_before_dense="auto",
        dense_policy=_dense_transport_policy(
            solve_method_use="incremental",
            dense_mem_block=True,
            dense_use_mixed=False,
        ),
        initial_policy=_initial_transport_policy(
            dense_mem_block=False,
            dense_use_mixed=False,
        ),
        active_size=512,
    )
    assert blocked == (
        (
            1,
            "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
            "(active_est_mem32=48.0 MB > 32.0 MB)",
        ),
    )

    mixed_and_precond = _dense_policy_notes(
        rhs_mode=2,
        solve_method_before_dense="auto",
        dense_policy=_dense_transport_policy(
            solve_method_use="dense",
            dense_mem_block=False,
            dense_use_mixed=True,
            dense_precond_mem_block=True,
        ),
        initial_policy=_initial_transport_policy(
            dense_mem_block=False,
            dense_use_mixed=False,
        ),
        active_size=512,
    )
    assert mixed_and_precond == (
        (
            1,
            "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
            "(active_est_mem64=96.0 MB > 32.0 MB)",
        ),
        (
            0,
            "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
            "(n=512)",
        ),
        (
            1,
            "solve_v3_transport_matrix_linear_gmres: dense preconditioner disabled "
            "(est_mem=128.0 MB > 64.0 MB)",
        ),
    )

    forced_dense = _dense_policy_notes(
        rhs_mode=2,
        solve_method_before_dense="auto",
        dense_policy=_dense_transport_policy(
            solve_method_use="dense",
            dense_mem_block=False,
            dense_use_mixed=False,
            force_dense=True,
        ),
        initial_policy=_initial_transport_policy(
            dense_mem_block=False,
            dense_use_mixed=False,
        ),
        active_size=512,
    )
    assert forced_dense == ()


def test_solve_transport_linear_uses_nonjit_or_jit_gmres(monkeypatch) -> None:
    calls: list[str] = []

    def fake_gmres(**kwargs):
        calls.append(f"gmres:{kwargs['solve_method']}")
        return "plain"

    def fake_gmres_jit(**kwargs):
        calls.append(f"jit:{kwargs['solve_method']}")
        return "jit"

    monkeypatch.setattr(transport_linear_system, "gmres_solve", fake_gmres)
    monkeypatch.setattr(transport_linear_system, "gmres_solve_jit", fake_gmres_jit)
    args = dict(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    assert solve_transport_linear(context=_context(use_solver_jit=False), **args) == "plain"
    assert solve_transport_linear(context=_context(use_solver_jit=True), **args) == "jit"
    assert calls == ["gmres:incremental", "jit:incremental"]


def test_transport_linear_solve_callbacks_bind_context(monkeypatch) -> None:
    captured = {}

    def fake_gmres(**kwargs):
        captured.update(kwargs)
        return "bound"

    monkeypatch.setattr(transport_linear_system, "gmres_solve", fake_gmres)
    callbacks = TransportLinearSolveCallbacks(context=_context(use_solver_jit=False))

    result = callbacks.solve(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert result == "bound"
    assert captured["solve_method"] == "incremental"
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_implicit_routes_to_custom_solve(monkeypatch) -> None:
    captured = {}

    def fake_custom(**kwargs):
        captured.update(kwargs)
        return "implicit"

    monkeypatch.setattr(transport_linear_system, "linear_custom_solve", fake_custom)

    assert (
        solve_transport_linear(
            context=_context(use_implicit=True, size_hint=17),
            matvec_fn=lambda x: x,
            b_vec=jnp.ones((2,)),
            x0_vec=None,
            tol_val=1e-8,
            atol_val=1e-12,
            restart_val=10,
            maxiter_val=20,
            solve_method_val="auto",
            preconditioner_val=None,
            precondition_side_val="right",
        )
        == "implicit"
    )
    assert captured["solver"] == "bicgstab"
    assert captured["solve_method"] == "batched"
    assert captured["size_hint"] == 17
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_with_residual_bicgstab_route(monkeypatch) -> None:
    captured = {}

    def fake_bicgstab(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(
        transport_linear_system, "bicgstab_solve_with_residual", fake_bicgstab
    )
    result, residual = solve_transport_linear_with_residual(
        context=_context(use_solver_jit=False),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="left",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["maxiter"] == 20
    assert "restart" not in captured


def test_solve_transport_linear_with_residual_implicit_routes_to_custom_solve(monkeypatch) -> None:
    captured = {}

    def fake_custom(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(
        transport_linear_system, "linear_custom_solve_with_residual", fake_custom
    )

    result, residual = solve_transport_linear_with_residual(
        context=_context(use_implicit=True, size_hint=23),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["solver"] == "bicgstab"
    assert captured["solve_method"] == "batched"
    assert captured["size_hint"] == 23
    assert captured["precondition_side"] == "right"


def test_solve_transport_linear_with_residual_uses_plain_or_jit_gmres(monkeypatch) -> None:
    calls: list[str] = []

    def fake_gmres(**kwargs):
        calls.append(f"gmres:{kwargs['solve_method']}")
        return SimpleNamespace(residual_norm=jnp.asarray(1.0)), jnp.ones((2,))

    def fake_gmres_jit(**kwargs):
        calls.append(f"jit:{kwargs['solve_method']}")
        return SimpleNamespace(residual_norm=jnp.asarray(2.0)), 2.0 * jnp.ones((2,))

    monkeypatch.setattr(
        transport_linear_system, "gmres_solve_with_residual", fake_gmres
    )
    monkeypatch.setattr(
        transport_linear_system, "gmres_solve_with_residual_jit", fake_gmres_jit
    )
    args = dict(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    plain, plain_residual = solve_transport_linear_with_residual(context=_context(use_solver_jit=False), **args)
    jit, jit_residual = solve_transport_linear_with_residual(context=_context(use_solver_jit=True), **args)

    assert float(plain.residual_norm) == 1.0
    assert float(jit.residual_norm) == 2.0
    assert calls == ["gmres:incremental", "jit:incremental"]
    assert plain_residual.tolist() == [1.0, 1.0]
    assert jit_residual.tolist() == [2.0, 2.0]


def test_solve_transport_linear_with_residual_distributed_axis_routes(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    @contextmanager
    def fake_sharding_constraints(enabled: bool):
        calls.append({"context_enabled": enabled})
        yield

    def fake_distributed(**kwargs):
        calls.append(dict(kwargs))
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(
        transport_linear_system, "sharding_constraints", fake_sharding_constraints
    )
    monkeypatch.setattr(
        transport_linear_system,
        "gmres_solve_with_residual_distributed",
        fake_distributed,
    )

    solve_transport_linear_with_residual(
        context=_context(distributed_axis="theta"),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
    )
    solve_transport_linear_with_residual(
        context=_context(distributed_axis="zeta"),
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=11,
        maxiter_val=21,
        solve_method_val="incremental",
        preconditioner_val=None,
    )

    distributed_calls = [call for call in calls if "axis_name" in call]
    assert [call["axis_name"] for call in distributed_calls] == ["theta", "zeta"]
    assert [call["solve_method"] for call in distributed_calls] == ["bicgstab", "incremental"]
    assert sum(1 for call in calls if call == {"context_enabled": True}) == 2


def test_transport_linear_solve_callbacks_bind_residual_context(monkeypatch) -> None:
    captured = {}

    def fake_bicgstab(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(residual_norm=jnp.asarray(0.0)), jnp.zeros((2,))

    monkeypatch.setattr(
        transport_linear_system, "bicgstab_solve_with_residual", fake_bicgstab
    )
    callbacks = TransportLinearSolveCallbacks(context=_context(use_solver_jit=False))

    result, residual = callbacks.solve_with_residual(
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((2,)),
        x0_vec=None,
        tol_val=1e-8,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        solve_method_val="auto",
        preconditioner_val=None,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert residual.shape == (2,)
    assert captured["precondition_side"] == "right"


def test_dense_matvec_lu_helpers_cache_and_solve() -> None:
    matrix = jnp.asarray([[4.0, 1.0], [2.0, 3.0]], dtype=jnp.float64)
    rhs = jnp.asarray([5.0, 8.0], dtype=jnp.float64)
    expected = jnp.linalg.solve(matrix, rhs)

    precond_cache: dict[tuple[object, int], object] = {}
    precond = dense_preconditioner_for_matvec(
        matvec_fn=lambda x: matrix @ x,
        n=2,
        dtype=jnp.float64,
        cache=precond_cache,
        key=("toy", 2),
    )
    assert jnp.allclose(precond(rhs), expected)
    assert (
        dense_preconditioner_for_matvec(
            matvec_fn=lambda x: 2.0 * x,
            n=2,
            dtype=jnp.float64,
            cache=precond_cache,
            key=("toy", 2),
        )
        is precond
    )

    solver_cache: dict[tuple[object, int], object] = {}
    solver = dense_solver_for_matvec(
        matvec_fn=lambda x: matrix @ x,
        n=2,
        dtype=jnp.float64,
        cache=solver_cache,
        key=("toy-solver", 2),
    )
    assert jnp.allclose(solver(rhs), expected)
    assert (
        dense_solver_for_matvec(
            matvec_fn=lambda x: 2.0 * x,
            n=2,
            dtype=jnp.float64,
            cache=solver_cache,
            key=("toy-solver", 2),
        )
        is solver
    )


def test_transport_host_gmres_left_preconditioned_progress(monkeypatch) -> None:
    b_vec = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    progress: list[tuple[int, str]] = []

    def fake_left_gmres(**kwargs):
        kwargs["progress_callback"](1, 0.25)
        return np.asarray([1.0, 2.0]), 10.0, 0.0, [0.25]

    monkeypatch.setattr(transport_linear_system, "explicit_left_preconditioned_gmres_scipy", fake_left_gmres)
    monkeypatch.setattr(
        transport_linear_system,
        "transport_host_gmres_accepts_preconditioned_residual",
        lambda **_kwargs: True,
    )

    result, residual = transport_host_gmres_solve(
        op=SimpleNamespace(rhs_mode=2),
        matvec_fn=lambda x: x,
        b_vec=b_vec,
        x0_vec=None,
        preconditioner_fn=lambda x: x,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precondition_side_val="left",
        emit=lambda level, message: progress.append((level, message)),
        which_rhs=3,
        progress_every=1,
    )

    assert float(result.residual_norm) <= 1.0e-12
    assert jnp.allclose(residual, jnp.zeros_like(b_vec))
    assert any("whichRHS=3 iter=1" in message for _level, message in progress)


def test_transport_host_gmres_plain_path_uses_true_residual(monkeypatch) -> None:
    def fake_gmres(**_kwargs):
        return np.asarray([2.0, -1.0]), 1.5, [1.5]

    monkeypatch.setattr(transport_linear_system, "gmres_solve_with_history_scipy", fake_gmres)
    result, residual = transport_host_gmres_solve(
        op=SimpleNamespace(rhs_mode=3),
        matvec_fn=lambda x: 2.0 * x,
        b_vec=jnp.asarray([4.0, -2.0], dtype=jnp.float64),
        x0_vec=None,
        preconditioner_fn=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert jnp.allclose(residual, jnp.zeros((2,), dtype=jnp.float64))


def test_transport_host_gmres_left_preconditioned_rejects_preconditioned_report(
    monkeypatch,
) -> None:
    b_vec = jnp.asarray([1.0, 2.0], dtype=jnp.float64)
    progress: list[tuple[int, str]] = []

    def fake_left_gmres(**kwargs):
        kwargs["progress_callback"](2, 0.25)
        return np.zeros((2,), dtype=np.float64), 3.0, 0.0, [0.25]

    monkeypatch.setattr(transport_linear_system, "explicit_left_preconditioned_gmres_scipy", fake_left_gmres)
    monkeypatch.setattr(
        transport_linear_system,
        "transport_host_gmres_accepts_preconditioned_residual",
        lambda **_kwargs: False,
    )

    result, residual = transport_host_gmres_solve(
        op=SimpleNamespace(rhs_mode=2),
        matvec_fn=lambda x: x,
        b_vec=b_vec,
        x0_vec=None,
        preconditioner_fn=lambda x: x,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precondition_side_val="left",
        emit=lambda level, message: progress.append((level, message)),
        which_rhs=2,
        progress_every=10,
    )

    assert float(result.residual_norm) == pytest.approx(float(jnp.linalg.norm(b_vec)))
    assert jnp.allclose(residual, b_vec)
    assert progress == []


def test_transport_host_gmres_plain_path_ignores_nonfinite_reported_residual(monkeypatch) -> None:
    def fake_gmres(**_kwargs):
        return np.asarray([2.0, -1.0]), float("nan"), []

    monkeypatch.setattr(transport_linear_system, "gmres_solve_with_history_scipy", fake_gmres)
    result, residual = transport_host_gmres_solve(
        op=SimpleNamespace(rhs_mode=3),
        matvec_fn=lambda x: 2.0 * x,
        b_vec=jnp.asarray([4.0, -2.0], dtype=jnp.float64),
        x0_vec=None,
        preconditioner_fn=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precondition_side_val="right",
    )

    assert float(result.residual_norm) == 0.0
    assert jnp.allclose(residual, jnp.zeros((2,), dtype=jnp.float64))


def test_emit_transport_ksp_iteration_stats_skip_success_and_failure(monkeypatch) -> None:
    emitted: list[tuple[int, str]] = []
    kwargs = dict(
        which_rhs=2,
        matvec_fn=lambda x: x,
        b_vec=jnp.ones((3,), dtype=jnp.float64),
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precond_side="right",
        solver_kind="gmres",
        emit=lambda level, message: emitted.append((level, message)),
    )

    emit_transport_ksp_iteration_stats(enabled=False, max_size=None, **kwargs)
    assert emitted == []

    emit_transport_ksp_iteration_stats(enabled=True, max_size=2, **kwargs)
    assert "skipped" in emitted[-1][1]

    monkeypatch.setattr(transport_linear, "_solve_history", lambda **_kwargs: [0.2, 0.01])
    emit_transport_ksp_iteration_stats(enabled=True, max_size=4, **kwargs)
    assert emitted[-1] == (0, "whichRHS=2 ksp_iterations=2 solver=gmres")

    def raise_history(**_kwargs):
        raise RuntimeError("synthetic history failure")

    monkeypatch.setattr(transport_linear, "_solve_history", raise_history)
    emit_transport_ksp_iteration_stats(enabled=True, max_size=4, **kwargs)
    assert "unavailable" in emitted[-1][1]


def test_transport_dense_batch_solves_full_and_rejects_varying_operators(monkeypatch) -> None:
    matrix = jnp.asarray([[3.0, 1.0], [0.0, 2.0]], dtype=jnp.float64)
    rhs_vectors = [matrix @ jnp.asarray([1.0, -1.0]), matrix @ jnp.asarray([2.0, 0.5])]
    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    solver_kinds: dict[int, str] = {}
    solve_methods: dict[int, str] = {}
    elapsed = np.zeros(2)
    emitted: list[tuple[int, str]] = []

    monkeypatch.setattr(transport_linear_system, "apply_v3_full_system_operator_cached", lambda op, x: op.matrix @ x)
    monkeypatch.setattr(transport_linear_system, "_operator_signature_cached", lambda op: (op.signature,))

    context = TransportDenseBatchContext(
        dense_backend_allowed=True,
        dense_use_mixed=True,
        use_active_dof_mode=False,
        active_size=0,
        op0=SimpleNamespace(total_size=2),
        op_matvec_by_index=[
            SimpleNamespace(signature="same", matrix=matrix),
            SimpleNamespace(signature="same", matrix=matrix),
        ],
        rhs_by_index=rhs_vectors,
        which_rhs_values=[1, 2],
        rhs_norms={1: jnp.linalg.norm(rhs_vectors[0]), 2: jnp.linalg.norm(rhs_vectors[1])},
        residual_norms=residual_norms,
        solver_kinds_by_rhs=solver_kinds,
        solve_methods_by_rhs=solve_methods,
        elapsed_s=elapsed,
        state_vectors=state_vectors,
        store_state_vectors=True,
        stream_diagnostics=False,
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x,
        emit=lambda level, message: emitted.append((level, message)),
    )

    assert solve_transport_dense_batch(context=context, op_probe_ref=context.op_matvec_by_index[0], reason="unit")
    assert jnp.allclose(state_vectors[1], jnp.asarray([1.0, -1.0]))
    assert jnp.allclose(state_vectors[2], jnp.asarray([2.0, 0.5]))
    assert solver_kinds == {1: "dense", 2: "dense"}
    assert solve_methods == {1: "dense", 2: "dense"}
    assert max(float(v) for v in residual_norms.values()) < 1.0e-12
    assert any("dense batched solve" in message for _level, message in emitted)

    varying_context = TransportDenseBatchContext(
        **{
            **context.__dict__,
            "op_matvec_by_index": [
                SimpleNamespace(signature="a", matrix=matrix),
                SimpleNamespace(signature="b", matrix=matrix),
            ],
        }
    )
    assert not solve_transport_dense_batch(
        context=varying_context,
        op_probe_ref=varying_context.op_matvec_by_index[0],
        reason="unit",
    )


def test_transport_dense_batch_active_streaming_requires_collector(monkeypatch) -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)
    monkeypatch.setattr(transport_linear_system, "apply_v3_full_system_operator_cached", lambda op, x: op.matrix @ x)
    monkeypatch.setattr(transport_linear_system, "_operator_signature_cached", lambda op: (op.signature,))
    context = TransportDenseBatchContext(
        dense_backend_allowed=True,
        dense_use_mixed=True,
        use_active_dof_mode=True,
        active_size=1,
        op0=SimpleNamespace(total_size=2),
        op_matvec_by_index=[SimpleNamespace(signature="same", matrix=matrix)],
        rhs_by_index=[jnp.asarray([2.0, 0.0], dtype=jnp.float64)],
        which_rhs_values=[1],
        rhs_norms={1: jnp.asarray(2.0)},
        residual_norms={},
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        elapsed_s=np.zeros(1),
        state_vectors={},
        store_state_vectors=False,
        stream_diagnostics=True,
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x,
        reduce_full=lambda x: x[:1],
        expand_reduced=lambda x: jnp.asarray([x[0], 0.0], dtype=jnp.float64),
        collect_transport_outputs=None,
    )

    with pytest.raises(RuntimeError, match="streaming diagnostics"):
        solve_transport_dense_batch(context=context, op_probe_ref=context.op_matvec_by_index[0], reason="unit")


def test_transport_dense_batch_rejects_disabled_backend_and_rhs3_krylov(monkeypatch) -> None:
    matrix = jnp.eye(2, dtype=jnp.float64)
    monkeypatch.setattr(transport_linear_system, "_operator_signature_cached", lambda op: (op.signature,))
    base = dict(
        dense_use_mixed=False,
        use_active_dof_mode=False,
        active_size=0,
        op0=SimpleNamespace(total_size=2),
        op_matvec_by_index=[SimpleNamespace(signature="same", matrix=matrix)],
        rhs_by_index=[jnp.asarray([1.0, 0.0], dtype=jnp.float64)],
        which_rhs_values=[3],
        rhs_norms={3: jnp.asarray(1.0)},
        residual_norms={},
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        elapsed_s=np.zeros(3),
        state_vectors={},
        store_state_vectors=False,
        stream_diagnostics=False,
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x,
    )

    disabled = TransportDenseBatchContext(
        dense_backend_allowed=False,
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        **base,
    )
    assert not solve_transport_dense_batch(
        context=disabled,
        op_probe_ref=disabled.op_matvec_by_index[0],
        reason="unit",
    )

    rhs3_special = TransportDenseBatchContext(
        dense_backend_allowed=True,
        rhs3_krylov_flags=lambda _which_rhs: (True, False),
        **base,
    )
    assert not solve_transport_dense_batch(
        context=rhs3_special,
        op_probe_ref=rhs3_special.op_matvec_by_index[0],
        reason="unit",
    )


def test_transport_dense_batch_solves_active_streaming_outputs(monkeypatch) -> None:
    matrix = jnp.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 0.0, 7.0],
        ],
        dtype=jnp.float64,
    )
    rhs_vectors = [
        jnp.asarray([2.0, -4.0, 0.0], dtype=jnp.float64),
        jnp.asarray([6.0, 8.0, 0.0], dtype=jnp.float64),
    ]
    collected: dict[int, jnp.ndarray] = {}
    state_vectors: dict[int, jnp.ndarray] = {}
    residual_norms: dict[int, jnp.ndarray] = {}
    emitted: list[tuple[int, str]] = []
    monkeypatch.setattr(transport_linear_system, "apply_v3_full_system_operator_cached", lambda op, x: op.matrix @ x)
    monkeypatch.setattr(transport_linear_system, "_operator_signature_cached", lambda op: (op.signature,))

    context = TransportDenseBatchContext(
        dense_backend_allowed=True,
        dense_use_mixed=True,
        use_active_dof_mode=True,
        active_size=2,
        op0=SimpleNamespace(total_size=3),
        op_matvec_by_index=[SimpleNamespace(signature="same", matrix=matrix)] * 2,
        rhs_by_index=rhs_vectors,
        which_rhs_values=[1, 2],
        rhs_norms={1: jnp.linalg.norm(rhs_vectors[0]), 2: jnp.linalg.norm(rhs_vectors[1])},
        residual_norms=residual_norms,
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        elapsed_s=np.zeros(2),
        state_vectors=state_vectors,
        store_state_vectors=True,
        stream_diagnostics=True,
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x,
        reduce_full=lambda x: x[:2],
        expand_reduced=lambda x: jnp.asarray([x[0], x[1], 0.0], dtype=jnp.float64),
        collect_transport_outputs=lambda which_rhs, x: collected.setdefault(int(which_rhs), x),
        emit=lambda level, message: emitted.append((int(level), str(message))),
    )

    assert solve_transport_dense_batch(
        context=context,
        op_probe_ref=context.op_matvec_by_index[0],
        reason="active-unit",
    )

    np.testing.assert_allclose(np.asarray(state_vectors[1]), np.asarray([1.0, -1.0, 0.0]), atol=1.0e-12)
    np.testing.assert_allclose(np.asarray(state_vectors[2]), np.asarray([3.0, 2.0, 0.0]), atol=1.0e-12)
    assert set(collected) == {1, 2}
    assert max(float(value) for value in residual_norms.values()) < 1.0e-10
    assert any("relative_residual" in message for _level, message in emitted)








def test_transport_matvec_cache_and_recycle_state() -> None:
    calls: list[tuple[str, list[float]]] = []

    def apply_operator(op, x):
        calls.append((op.signature, [float(v) for v in x]))
        return op.scale * x

    cache = TransportMatvecCache(
        use_active_dof_mode=True,
        active_size=1,
        reduce_full=lambda x: x[:1],
        expand_reduced=lambda x: jnp.asarray([x[0], 0.0], dtype=jnp.float64),
        apply_operator=apply_operator,
        operator_signature=lambda op: (op.signature,),
    )
    op = SimpleNamespace(signature="op", scale=2.0)

    full_mv = cache.get_full(op)
    assert cache.get_full(op) is full_mv
    assert jnp.allclose(full_mv(jnp.asarray([1.0, 2.0])), jnp.asarray([2.0, 4.0]))
    reduced_mv = cache.get_reduced(op)
    assert cache.get_reduced(op) is reduced_mv
    assert jnp.allclose(reduced_mv(jnp.asarray([3.0])), jnp.asarray([6.0]))

    state = TransportRecycleState(k=1)
    state.append_full(jnp.asarray([1.0, 0.0]), jnp.asarray([2.0, 0.0]))
    state.append_full(jnp.asarray([0.0, 1.0]), jnp.asarray([0.0, 2.0]))
    assert len(state.full_basis) == 1
    guess = state.candidate_full(jnp.asarray([0.0, 4.0]))
    assert guess is not None
    assert jnp.allclose(guess, jnp.asarray([0.0, 2.0]))

    seeded = TransportRecycleState(k=2)
    seeded.seed_from_state(
        state_x_by_rhs={2: jnp.asarray([4.0, 5.0]), 3: jnp.asarray([7.0])},
        total_size=2,
        active_size=1,
        matvec_cache=cache,
        op_ref=op,
    )
    assert len(seeded.full_basis) == 1
    assert len(seeded.reduced_basis) == 2


def test_transport_loop_progress_reports_and_gates() -> None:
    emitted: list[tuple[int, str]] = []
    progress = TransportLoopProgress(
        which_rhs_values=[1, 2],
        rhs_norms={1: 4.0, 2: 2.0},
        residual_norms={1: 0.04, 2: 1.0},
        elapsed_s=np.zeros(2),
        abort_max_residual=0.0,
        abort_max_relative_residual=0.0,
        emit=lambda level, message: emitted.append((level, message)),
    )
    progress.finish_rhs(which_rhs=1, rhs_elapsed_s=0.5, total_elapsed_s=0.5)
    assert progress.relative_residual(1) == pytest.approx(0.01)
    assert progress.elapsed_s[0] == pytest.approx(0.5)
    assert any("completed 1/2" in message or "whichRHS=1" in message for _level, message in emitted)

    failing = TransportLoopProgress(
        which_rhs_values=[1],
        rhs_norms={1: 2.0},
        residual_norms={1: 1.0},
        elapsed_s=np.zeros(1),
        abort_max_residual=0.1,
        abort_max_relative_residual=0.2,
        emit=lambda level, message: emitted.append((level, message)),
    )
    with pytest.raises(RuntimeError, match="transport residual gate failed"):
        failing.finish_rhs(which_rhs=1, rhs_elapsed_s=0.1, total_elapsed_s=0.1)


def test_resolve_transport_recycle_k_env_disable_and_operator_variation(monkeypatch) -> None:
    emitted: list[tuple[int, str]] = []
    op = SimpleNamespace(rhs_mode=3)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "bad")
    assert (
        resolve_transport_recycle_k(
            op=op,
            use_implicit=False,
            op_matvec_by_index=[],
            disable_auto_recycle=lambda **_kwargs: False,
            emit=lambda level, message: emitted.append((level, message)),
        )
        == 4
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "3")
    assert (
        resolve_transport_recycle_k(
            op=op,
            use_implicit=False,
            op_matvec_by_index=[],
            disable_auto_recycle=lambda **_kwargs: True,
            emit=lambda level, message: emitted.append((level, message)),
        )
        == 0
    )
    assert "auto recycle disabled" in emitted[-1][1]

    assert (
        resolve_transport_recycle_k(
            op=op,
            use_implicit=False,
            op_matvec_by_index=[SimpleNamespace(sig="a"), SimpleNamespace(sig="b")],
            disable_auto_recycle=lambda **_kwargs: False,
            emit=lambda level, message: emitted.append((level, message)),
            operator_signature=lambda probe: (probe.sig,),
        )
        == 0
    )
    assert "operator varies" in emitted[-1][1]








def test_transport_active_dof_krylov_matches_full_tiny_reference(monkeypatch) -> None:
    """Active-DOF compaction should preserve the full transport solution."""
    nml = read_sfincs_input("tests/ref/transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_FORTRAN_STDOUT", "0")
    monkeypatch.setenv("SFINCS_JAX_SOLVER_ITER_STATS", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ACTIVE_DOF", "0")

    full = transport_linear.solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1.0e-10,
        maxiter=30,
        which_rhs_values=[1],
        collect_transport_output_fields=False,
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ACTIVE_DOF", "1")
    active = transport_linear.solve_v3_transport_matrix_linear_gmres(
        nml=nml,
        tol=1.0e-10,
        maxiter=30,
        which_rhs_values=[1],
        collect_transport_output_fields=False,
    )

    assert not full.use_active_dof_mode
    assert active.use_active_dof_mode
    assert int(active.active_size) == int(active.op0.total_size)
    assert full.solver_kinds_by_rhs[1] == "bicgstab"
    assert active.solver_kinds_by_rhs[1] == "bicgstab"
    assert float(np.asarray(full.residual_norms_by_rhs[1])) < 1.0e-8
    assert float(np.asarray(active.residual_norms_by_rhs[1])) < 1.0e-8
    np.testing.assert_allclose(
        np.asarray(active.state_vectors_by_rhs[1]),
        np.asarray(full.state_vectors_by_rhs[1]),
        rtol=1.0e-8,
        atol=1.0e-10,
    )
