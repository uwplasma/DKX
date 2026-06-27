from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
import sfincs_jax.problems.transport_linear_system as transport_linear_system
import sfincs_jax.problems.transport_solve as transport_linear
from sfincs_jax.problems.transport_linear_system import (
    TransportLinearSolveCallbacks,
    TransportLinearSolveContext,
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
    TransportSparseDirectContext,
    dense_preconditioner_for_matvec,
    dense_solver_for_matvec,
    emit_transport_ksp_iteration_stats,
    resolve_transport_recycle_k,
    solve_transport_dense_batch,
    transport_host_gmres_solve,
    transport_sparse_direct_pattern_for_solve,
    transport_sparse_direct_solve,
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
    assert transport_solver_kind("incremental", rhs_mode=2) == ("gmres", "incremental")

    assert transport_restart_for_method("auto", rhs_mode=2, gmres_restart=30, restart=80) == 80
    assert transport_restart_for_method("incremental", rhs_mode=2, gmres_restart=30, restart=80) == 30


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
        dense_use_mixed=False,
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


def _sparse_context(**overrides) -> TransportSparseDirectContext:
    values = dict(
        op=SimpleNamespace(
            rhs_mode=3,
            include_phi1=False,
            fblock=SimpleNamespace(fp=None),
            n_x=1,
            total_size=2,
        ),
        factor_cache={},
        pattern_cache={},
        sparse_drop_tol=0.0,
        sparse_drop_rel=0.0,
        emit=None,
        sparse_factor_cache_key=lambda key, dtype: (*key, np.dtype(dtype).name),
        hash_numpy_array_for_cache=lambda arr: tuple(np.asarray(arr).reshape((-1,)).tolist()),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: (_kwargs["pattern"], SimpleNamespace(operator=SimpleNamespace(matrix="pattern-matrix"), factor="pattern-factor")),
        build_sparse_ilu_from_matvec=lambda **_kwargs: (
            "fallback-matrix",
            None,
            "fallback-factor",
            None,
            None,
            None,
            None,
        ),
        try_build_direct_active_operator_bundle=lambda **_kwargs: None,
        host_sparse_direct_solve_with_refinement=lambda **_kwargs: (np.ones(2), 0.0),
        host_sparse_direct_refine_steps=lambda *_args, **_kwargs: 1,
        host_sparse_direct_polish=lambda **_kwargs: (_kwargs["x0_np"], 0.0),
        sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
        sparse_direct_use_explicit_helper=lambda **_kwargs: False,
        sparse_direct_needs_float64_retry=lambda **_kwargs: False,
    )
    values.update(overrides)
    return TransportSparseDirectContext(**values)


def test_transport_sparse_direct_pattern_policy_cache_and_budget(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=4)
    summary = SimpleNamespace(shape=(2, 2), nnz=4, avg_row_nnz=2.0, max_row_nnz=2)
    emitted: list[tuple[int, str]] = []
    monkeypatch.setattr(transport_linear, "v3_full_system_conservative_sparsity_pattern", lambda op: pattern)
    monkeypatch.setattr(transport_linear, "summarize_v3_sparse_pattern", lambda op, pattern: summary)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB", "1")
    context = _sparse_context(emit=lambda level, message: emitted.append((level, message)))

    assert transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None) is pattern
    assert transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None) is pattern
    assert "transport sparse pattern selected" in emitted[-1][1]

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB", "0")
    forced_context = _sparse_context()
    with pytest.raises(MemoryError, match="CSR budget"):
        transport_sparse_direct_pattern_for_solve(context=forced_context, n=2, active_indices_np=None)


def test_transport_sparse_direct_solve_retries_float64_after_float32_gate(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "off")

    def factor_dtype(**_kwargs):
        return np.dtype(np.float32)

    def build_ilu(**kwargs):
        dtype_name = np.dtype(kwargs["factor_dtype"]).name
        calls.append(dtype_name)
        return (
            f"matrix-{dtype_name}",
            None,
            f"factor-{dtype_name}",
            None,
            None,
            None,
            None,
        )

    def solve_with_refinement(**kwargs):
        if kwargs["factor_dtype"] == np.dtype(np.float32):
            return np.asarray([0.0, 0.0]), 10.0
        return np.asarray([1.0, 2.0]), 0.0

    context = _sparse_context(
        build_sparse_ilu_from_matvec=build_ilu,
        host_sparse_direct_solve_with_refinement=solve_with_refinement,
        sparse_factor_dtype=factor_dtype,
        sparse_direct_needs_float64_retry=lambda **kwargs: kwargs["factor_dtype"] == np.dtype(np.float32),
    )

    result = transport_sparse_direct_solve(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        n=2,
        dtype=jnp.float64,
        cache_key=("toy",),
        active_indices_np=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=5,
        maxiter_val=10,
        precondition_side_val="right",
    )

    assert calls == ["float32", "float64"]
    assert jnp.allclose(result.x, jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    assert float(result.residual_norm) == 0.0


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
