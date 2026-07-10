from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.problems.transport_finalize as transport_finalize
from sfincs_jax.problems.transport_finalize import (
    TransportConstraintNullspaceProjector,
    TransportKSPIterationRequest,
    TransportRHSFinalizationContext,
    compute_transport_postsolve_diagnostics,
    finalize_full_transport_rhs,
    finalize_reduced_transport_rhs,
)
from sfincs_jax.problems.transport_solve import (
    TransportDenseBatchContext,
    TransportLoopProgress,
    TransportMatvecCache,
    TransportRecycleState,
    _dense_dtype,
    _emit_rhs_residual,
    _store_dense_batch_result,
    recycled_transport_initial_guess,
    resolve_transport_recycle_k,
)


def _op(signature: tuple[object, ...], *, scale: float = 2.0):
    return SimpleNamespace(signature=signature, scale=float(scale))


def _signature(op) -> tuple[object, ...]:
    return op.signature


def _apply(op, x):
    return op.scale * x


def test_transport_matvec_cache_reuses_full_and_reduced_closures() -> None:
    def reduce_full(x):
        return x[jnp.asarray([0, 2])]

    def expand_reduced(x):
        out = jnp.zeros((4,), dtype=x.dtype)
        return out.at[jnp.asarray([0, 2])].set(x)

    cache = TransportMatvecCache(
        use_active_dof_mode=True,
        active_size=2,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    op = _op(("same",), scale=3.0)
    full = cache.get_full(op)
    reduced = cache.get_reduced(op)

    assert cache.get_full(op) is full
    assert cache.get_reduced(op) is reduced
    np.testing.assert_allclose(np.asarray(full(jnp.arange(4.0))), np.asarray([0.0, 3.0, 6.0, 9.0]))
    np.testing.assert_allclose(np.asarray(reduced(jnp.asarray([2.0, 5.0]))), np.asarray([6.0, 15.0]))


def test_transport_matvec_cache_returns_full_closure_when_active_mode_disabled() -> None:
    cache = TransportMatvecCache(
        use_active_dof_mode=False,
        active_size=0,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    op = _op(("full",), scale=4.0)
    full = cache.get_full(op)
    reduced = cache.get_reduced(op)

    assert reduced is full
    np.testing.assert_allclose(np.asarray(reduced(jnp.asarray([1.0, 2.0]))), np.asarray([4.0, 8.0]))


def test_recycled_transport_initial_guess_matches_small_subspace() -> None:
    basis = [jnp.asarray([1.0, 0.0]), jnp.asarray([0.0, 1.0])]
    basis_au = [jnp.asarray([2.0, 0.0]), jnp.asarray([0.0, 3.0])]
    x0 = recycled_transport_initial_guess(jnp.asarray([4.0, 9.0]), basis, basis_au)

    assert x0 is not None
    np.testing.assert_allclose(np.asarray(x0), np.asarray([2.0, 3.0]), rtol=1e-10, atol=1e-10)
    assert recycled_transport_initial_guess(jnp.ones((2,)), [], []) is None


def test_transport_recycle_state_trims_and_seeds_full_and_reduced() -> None:
    def reduce_full(x):
        return x[jnp.asarray([0, 2])]

    def expand_reduced(x):
        out = jnp.zeros((4,), dtype=x.dtype)
        return out.at[jnp.asarray([0, 2])].set(x)

    cache = TransportMatvecCache(
        use_active_dof_mode=True,
        active_size=2,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    recycle = TransportRecycleState(k=1)
    recycle.seed_from_state(
        state_x_by_rhs={
            1: jnp.asarray([1.0, 2.0, 3.0, 4.0]),
            2: jnp.asarray([5.0, 7.0]),
        },
        total_size=4,
        active_size=2,
        matvec_cache=cache,
        op_ref=_op(("ref",), scale=2.0),
    )

    assert len(recycle.full_basis) == 1
    assert len(recycle.reduced_basis) == 1
    np.testing.assert_allclose(np.asarray(recycle.reduced_basis[0]), np.asarray([5.0, 7.0]))
    np.testing.assert_allclose(np.asarray(recycle.reduced_basis_au[0]), np.asarray([10.0, 14.0]))

    recycle.append_full(jnp.asarray([1.0, 0.0, 0.0, 0.0]), jnp.asarray([2.0, 0.0, 0.0, 0.0]))
    assert len(recycle.full_basis) == 1
    np.testing.assert_allclose(np.asarray(recycle.full_basis[0]), np.asarray([1.0, 0.0, 0.0, 0.0]))


def test_transport_recycle_state_disabled_is_noop() -> None:
    cache = TransportMatvecCache(
        use_active_dof_mode=False,
        active_size=0,
        apply_operator=_apply,
        operator_signature=_signature,
    )
    recycle = TransportRecycleState(k=0)

    assert not recycle.enabled
    assert recycle.candidate_full(jnp.ones(2)) is None
    assert recycle.candidate_reduced(jnp.ones(2)) is None

    recycle.append_full(jnp.ones(2), jnp.ones(2))
    recycle.append_reduced(jnp.ones(1), jnp.ones(1), x_full=jnp.ones(2), ax_full=jnp.ones(2))
    recycle.seed_from_state(
        state_x_by_rhs={1: jnp.asarray([1.0, 2.0])},
        total_size=2,
        active_size=1,
        matvec_cache=cache,
        op_ref=_op(("ref",), scale=2.0),
    )

    assert recycle.full_basis == []
    assert recycle.reduced_basis == []


def test_transport_recycle_state_candidates_use_trimmed_recent_basis() -> None:
    recycle = TransportRecycleState(k=2)
    recycle.append_full(
        jnp.asarray([100.0, 0.0]),
        jnp.asarray([100.0, 0.0]),
    )
    recycle.append_full(
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([2.0, 0.0]),
    )
    recycle.append_full(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([0.0, 3.0]),
    )

    full_candidate = recycle.candidate_full(jnp.asarray([4.0, 9.0]))

    assert full_candidate is not None
    assert len(recycle.full_basis) == 2
    np.testing.assert_allclose(
        np.asarray(full_candidate),
        np.asarray([2.0, 3.0]),
        rtol=1e-10,
        atol=1e-10,
    )

    recycle.append_reduced(
        jnp.asarray([50.0, 0.0]),
        jnp.asarray([50.0, 0.0]),
    )
    recycle.append_reduced(
        jnp.asarray([1.0, 0.0]),
        jnp.asarray([5.0, 0.0]),
    )
    recycle.append_reduced(
        jnp.asarray([0.0, 1.0]),
        jnp.asarray([0.0, 7.0]),
    )

    reduced_candidate = recycle.candidate_reduced(jnp.asarray([10.0, 21.0]))

    assert reduced_candidate is not None
    assert len(recycle.reduced_basis) == 2
    np.testing.assert_allclose(
        np.asarray(reduced_candidate),
        np.asarray([2.0, 3.0]),
        rtol=1e-10,
        atol=1e-10,
    )


def test_resolve_transport_recycle_k_respects_env_and_operator_variation(monkeypatch) -> None:
    messages: list[tuple[int, str]] = []

    def emit(level: int, message: str) -> None:
        messages.append((int(level), str(message)))

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", raising=False)
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",)), _op(("a",))],
            disable_auto_recycle=lambda **_: False,
            emit=emit,
            operator_signature=_signature,
        )
        == 4
    )

    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",)), _op(("b",))],
            disable_auto_recycle=lambda **_: False,
            emit=emit,
            operator_signature=_signature,
        )
        == 0
    )
    assert any("matvec operator varies" in message for _, message in messages)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "7")
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",))],
            disable_auto_recycle=lambda **_: True,
            emit=emit,
            operator_signature=_signature,
        )
        == 0
    )
    assert any("auto recycle disabled" in message for _, message in messages)


def test_resolve_transport_recycle_k_handles_invalid_and_negative_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "not-an-int")
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",)), _op(("a",))],
            disable_auto_recycle=lambda **_: False,
            emit=None,
            operator_signature=_signature,
        )
        == 4
    )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_RECYCLE_K", "-12")
    assert (
        resolve_transport_recycle_k(
            op=object(),
            use_implicit=False,
            op_matvec_by_index=[_op(("a",))],
            disable_auto_recycle=lambda **_: False,
            emit=None,
            operator_signature=_signature,
        )
        == 0
    )


def test_transport_loop_progress_records_elapsed_and_emits_eta() -> None:
    messages: list[tuple[int, str]] = []
    elapsed_s = np.zeros((2,), dtype=np.float64)
    progress = TransportLoopProgress(
        which_rhs_values=[1, 2],
        rhs_norms={1: jnp.asarray(10.0), 2: jnp.asarray(5.0)},
        residual_norms={1: jnp.asarray(0.25), 2: jnp.asarray(0.1)},
        elapsed_s=elapsed_s,
        abort_max_residual=0.0,
        abort_max_relative_residual=0.0,
        emit=lambda level, message: messages.append((int(level), str(message))),
    )

    progress.finish_rhs(which_rhs=1, rhs_elapsed_s=2.5, total_elapsed_s=3.0)
    progress.finish_rhs(which_rhs=2, rhs_elapsed_s=1.5, total_elapsed_s=4.5)

    np.testing.assert_allclose(elapsed_s, np.asarray([2.5, 1.5]))
    assert progress.elapsed_history == [2.5, 1.5]
    assert any("whichRHS=1: residual_norm=2.500000e-01" in message for _, message in messages)
    assert any("progress 2/2" in message for _, message in messages)


def test_transport_loop_progress_residual_gate_fails_fast() -> None:
    messages: list[tuple[int, str]] = []
    elapsed_s = np.zeros((1,), dtype=np.float64)
    progress = TransportLoopProgress(
        which_rhs_values=[1],
        rhs_norms={1: jnp.asarray(2.0)},
        residual_norms={1: jnp.asarray(1.0)},
        elapsed_s=elapsed_s,
        abort_max_residual=0.5,
        abort_max_relative_residual=0.0,
        emit=lambda level, message: messages.append((int(level), str(message))),
    )

    with pytest.raises(RuntimeError, match="transport residual gate failed"):
        progress.finish_rhs(which_rhs=1, rhs_elapsed_s=7.0, total_elapsed_s=8.0)

    np.testing.assert_allclose(elapsed_s, np.asarray([7.0]))
    assert progress.elapsed_history == []
    assert any("aborting remaining whichRHS solves" in message for _, message in messages)


def test_transport_loop_progress_reports_nan_relative_residual_for_zero_rhs() -> None:
    messages: list[tuple[int, str]] = []
    elapsed_s = np.zeros((1,), dtype=np.float64)
    progress = TransportLoopProgress(
        which_rhs_values=[1],
        rhs_norms={1: jnp.asarray(0.0)},
        residual_norms={1: jnp.asarray(0.25)},
        elapsed_s=elapsed_s,
        abort_max_residual=0.0,
        abort_max_relative_residual=0.0,
        emit=lambda level, message: messages.append((int(level), str(message))),
    )

    assert np.isnan(progress.relative_residual(1))
    assert progress.residual_failure(1) is None

    progress.finish_rhs(which_rhs=1, rhs_elapsed_s=1.25, total_elapsed_s=1.5)

    np.testing.assert_allclose(elapsed_s, np.asarray([1.25]))
    assert any("relative_residual=nan" in message for _, message in messages)


def test_emit_rhs_residual_handles_zero_rhs_and_missing_emit() -> None:
    messages: list[tuple[int, str]] = []

    _emit_rhs_residual(
        emit=None,
        which_rhs=1,
        residual_norm=0.25,
        rhs_norm=0.0,
        elapsed_s=0.5,
    )
    _emit_rhs_residual(
        emit=lambda level, message: messages.append((int(level), str(message))),
        which_rhs=2,
        residual_norm=0.25,
        rhs_norm=0.0,
        elapsed_s=0.5,
    )

    assert len(messages) == 1
    assert "whichRHS=2" in messages[0][1]
    assert "relative_residual=nan" in messages[0][1]


def test_dense_dtype_respects_mixed_precision_flag() -> None:
    assert _dense_dtype(jnp.float64, dense_use_mixed=False) is jnp.float64
    assert _dense_dtype(jnp.float64, dense_use_mixed=True) is jnp.float32


def _dense_batch_context(
    *,
    stream_diagnostics: bool,
    collector=None,
) -> TransportDenseBatchContext:
    return TransportDenseBatchContext(
        dense_backend_allowed=True,
        dense_use_mixed=False,
        use_active_dof_mode=False,
        active_size=2,
        op0=SimpleNamespace(total_size=2),
        op_matvec_by_index=[],
        rhs_by_index=[],
        which_rhs_values=[1],
        rhs_norms={1: jnp.asarray(2.0)},
        residual_norms={},
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        elapsed_s=np.zeros((1,), dtype=np.float64),
        state_vectors={},
        store_state_vectors=True,
        stream_diagnostics=stream_diagnostics,
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x,
        collect_transport_outputs=collector,
        emit=None,
    )


def test_store_dense_batch_result_records_state_solver_metadata_and_elapsed() -> None:
    collected: list[tuple[int, np.ndarray]] = []
    context = _dense_batch_context(
        stream_diagnostics=True,
        collector=lambda which_rhs, x_col: collected.append((which_rhs, np.asarray(x_col))),
    )

    _store_dense_batch_result(
        context=context,
        which_rhs=1,
        x_col=jnp.asarray([1.0, 2.0]),
        residual_norm=jnp.asarray(0.5),
        elapsed_each_s=0.25,
    )

    np.testing.assert_allclose(np.asarray(context.state_vectors[1]), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(context.residual_norms[1]), np.asarray(0.5))
    assert context.solver_kinds_by_rhs[1] == "dense"
    assert context.solve_methods_by_rhs[1] == "dense"
    np.testing.assert_allclose(context.elapsed_s, np.asarray([0.25]))
    assert collected[0][0] == 1
    np.testing.assert_allclose(collected[0][1], np.asarray([1.0, 2.0]))


def test_store_dense_batch_result_requires_collector_for_streaming() -> None:
    context = _dense_batch_context(stream_diagnostics=True, collector=None)

    with pytest.raises(RuntimeError, match="streaming diagnostics"):
        _store_dense_batch_result(
            context=context,
            which_rhs=1,
            x_col=jnp.asarray([1.0, 2.0]),
            residual_norm=jnp.asarray(0.5),
            elapsed_each_s=0.25,
        )


def _finalization_context(*, collector=None, recycle=None, iter_calls=None) -> TransportRHSFinalizationContext:
    def emit_iteration_stats(**kwargs):
        if iter_calls is not None:
            iter_calls.append(kwargs)

    return TransportRHSFinalizationContext(
        state_vectors={},
        residual_norms={},
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        store_state_vectors=True,
        stream_diagnostics=collector is not None,
        collect_transport_outputs=collector,
        recycle_state=recycle,
        apply_operator=lambda op, x: op.scale * x,
        emit_iteration_stats=emit_iteration_stats,
        emit=None,
        iter_stats_enabled=True,
        iter_stats_max_size=64,
        atol=1.0e-12,
        maxiter=17,
        precond_side="left",
    )


class _RecycleRecorder:
    def __init__(self) -> None:
        self.full: list[tuple[np.ndarray, np.ndarray]] = []
        self.reduced: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def append_full(self, x_full, ax_full) -> None:
        self.full.append((np.asarray(x_full), np.asarray(ax_full)))

    def append_reduced(self, x_reduced, ax_reduced, *, x_full, ax_full) -> None:
        self.reduced.append(
            (
                np.asarray(x_reduced),
                np.asarray(ax_reduced),
                np.asarray(x_full),
                np.asarray(ax_full),
            )
        )


def test_transport_constraint_projector_only_projects_policy_candidates() -> None:
    calls: list[dict[str, object]] = []
    policy = SimpleNamespace(projection_candidate=lambda which_rhs: int(which_rhs) == 2)

    def project_solution(**kwargs):
        calls.append(kwargs)
        return kwargs["x_vec"] + 10.0

    projector = TransportConstraintNullspaceProjector(
        op=SimpleNamespace(name="op"),
        policy=policy,
        enabled_env_var="UNIT_TEST_PROJECT",
        project_solution=project_solution,
    )

    x = jnp.asarray([1.0, 2.0])
    np.testing.assert_allclose(np.asarray(projector.project(x, which_rhs=1, op_matvec="a", rhs_vec=x)), np.asarray(x))
    projected = projector.project(x, which_rhs=2, op_matvec="b", rhs_vec=x)

    np.testing.assert_allclose(np.asarray(projected), np.asarray([11.0, 12.0]))
    assert len(calls) == 1
    assert calls[0]["enabled_env_var"] == "UNIT_TEST_PROJECT"
    assert calls[0]["matvec_op"] == "b"


def test_finalize_full_transport_rhs_uses_supplied_residual_and_records_side_effects() -> None:
    collected: list[tuple[int, np.ndarray]] = []
    recycle = _RecycleRecorder()
    iter_calls: list[dict[str, object]] = []
    context = _finalization_context(
        collector=lambda which_rhs, x_full: collected.append((int(which_rhs), np.asarray(x_full))),
        recycle=recycle,
        iter_calls=iter_calls,
    )
    rhs_full = jnp.asarray([3.0, 5.0])
    residual_vec = jnp.asarray([0.25, -0.5])
    result = SimpleNamespace(x=jnp.asarray([1.0, 2.0]), residual_norm=jnp.asarray(0.75))
    request = TransportKSPIterationRequest(
        matvec_fn=lambda x: 2.0 * x,
        b_vec=rhs_full,
        precond_fn=None,
        x0_vec=None,
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=9,
        maxiter_val=13,
        precond_side="left",
        solver_kind="gmres",
    )

    out = finalize_full_transport_rhs(
        context=context,
        which_rhs=3,
        result=result,
        rhs_full=rhs_full,
        op_matvec=SimpleNamespace(scale=99.0),
        solver_kind="gmres",
        solve_method="incremental",
        dense_used=False,
        projection_needed=False,
        residual_vec=residual_vec,
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x + 1000.0,
        ksp_request=request,
    )

    np.testing.assert_allclose(np.asarray(out.x_full), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(out.ax_full), np.asarray([2.75, 5.5]))
    np.testing.assert_allclose(np.asarray(out.residual_norm), np.asarray(0.75))
    assert context.solver_kinds_by_rhs[3] == "gmres"
    assert context.solve_methods_by_rhs[3] == "incremental"
    np.testing.assert_allclose(np.asarray(context.state_vectors[3]), np.asarray([1.0, 2.0]))
    assert collected[0][0] == 3
    np.testing.assert_allclose(collected[0][1], np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(recycle.full[0][1], np.asarray([2.75, 5.5]))
    assert iter_calls[0]["which_rhs"] == 3
    assert iter_calls[0]["solver_kind"] == "gmres"


def test_finalize_full_transport_rhs_projects_and_recomputes_true_residual() -> None:
    context = _finalization_context()
    rhs_full = jnp.asarray([1.0, 1.0])
    result = SimpleNamespace(x=jnp.asarray([2.0, 3.0]), residual_norm=jnp.asarray(123.0))

    out = finalize_full_transport_rhs(
        context=context,
        which_rhs=1,
        result=result,
        rhs_full=rhs_full,
        op_matvec=SimpleNamespace(scale=2.0),
        solver_kind="bicgstab",
        solve_method="batched",
        dense_used=True,
        projection_needed=True,
        residual_vec=jnp.asarray([0.0, 0.0]),
        maybe_project_constraint_nullspace=lambda x, **_kwargs: x - 1.0,
        ksp_request=None,
    )

    np.testing.assert_allclose(np.asarray(out.x_full), np.asarray([1.0, 2.0]))
    np.testing.assert_allclose(np.asarray(out.ax_full), np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(np.asarray(out.residual_norm), np.linalg.norm(np.asarray([1.0, 3.0])))


def test_finalize_reduced_transport_rhs_accepts_precomputed_full_solution_without_recompute() -> None:
    collected: list[tuple[int, np.ndarray]] = []
    recycle = _RecycleRecorder()
    context = _finalization_context(
        collector=lambda which_rhs, x_full: collected.append((int(which_rhs), np.asarray(x_full))),
        recycle=recycle,
    )
    result = SimpleNamespace(x=jnp.asarray([7.0, 8.0]))
    x_full = jnp.asarray([1.0, 2.0, 3.0])
    ax_full = jnp.asarray([2.0, 4.0, 6.0])

    out = finalize_reduced_transport_rhs(
        context=context,
        which_rhs=2,
        result=result,
        rhs_full=jnp.asarray([0.0, 0.0, 0.0]),
        op_matvec=SimpleNamespace(scale=2.0),
        solver_kind="dense",
        solve_method="active_dense",
        dense_used=True,
        expand_reduced=lambda _x: pytest.fail("accepted full solution should skip expansion"),
        reduce_full=lambda value: value[jnp.asarray([0, 2])],
        maybe_project_constraint_nullspace=lambda _x, **_kwargs: pytest.fail("accepted full solution should skip projection"),
        ksp_request=None,
        accepted_x_full=x_full,
        accepted_ax_full=ax_full,
        accepted_residual_norm=jnp.asarray(0.125),
    )

    np.testing.assert_allclose(np.asarray(out.x_full), np.asarray(x_full))
    np.testing.assert_allclose(np.asarray(out.ax_full), np.asarray(ax_full))
    np.testing.assert_allclose(np.asarray(out.residual_norm), np.asarray(0.125))
    assert context.solver_kinds_by_rhs[2] == "dense"
    assert collected[0][0] == 2
    np.testing.assert_allclose(recycle.reduced[0][0], np.asarray([7.0, 8.0]))
    np.testing.assert_allclose(recycle.reduced[0][1], np.asarray([2.0, 6.0]))


def test_compute_transport_postsolve_diagnostics_streaming_requires_accumulator() -> None:
    with pytest.raises(RuntimeError, match="without an accumulator"):
        compute_transport_postsolve_diagnostics(
            op0=SimpleNamespace(n_species=1, total_size=4),
            geom=object(),
            state_vectors={1: jnp.ones(4)},
            which_rhs_values=[1],
            stream_diagnostics=True,
            streaming_outputs=None,
            use_diag_op0=True,
            diag_op_by_index=None,
        )


def test_compute_transport_postsolve_diagnostics_uses_streamed_flux_arrays(monkeypatch) -> None:
    fake_fields = {"FSABFlow": np.asarray([[5.0, 6.0]])}
    streaming = SimpleNamespace(
        diagnostic_flux_arrays=lambda: (
            jnp.asarray([[1.0, 2.0]]),
            jnp.asarray([[3.0, 4.0]]),
            jnp.asarray([[5.0, 6.0]]),
        ),
        output_fields=lambda: fake_fields,
    )
    seen: dict[str, np.ndarray] = {}

    def fake_transport_matrix_from_flux_arrays(**kwargs):
        seen["particle"] = np.asarray(kwargs["particle_flux_vm_psi_hat"])
        seen["heat"] = np.asarray(kwargs["heat_flux_vm_psi_hat"])
        seen["flow"] = np.asarray(kwargs["fsab_flow"])
        return jnp.asarray([[42.0]])

    monkeypatch.setattr(transport_finalize, "v3_transport_matrix_from_flux_arrays", fake_transport_matrix_from_flux_arrays)

    out = compute_transport_postsolve_diagnostics(
        op0=SimpleNamespace(n_species=1, total_size=4),
        geom=object(),
        state_vectors={1: jnp.ones(4), 2: 2.0 * jnp.ones(4)},
        which_rhs_values=[1, 2],
        stream_diagnostics=True,
        streaming_outputs=streaming,
        use_diag_op0=True,
        diag_op_by_index=None,
    )

    np.testing.assert_allclose(np.asarray(out.transport_matrix), np.asarray([[42.0]]))
    assert out.transport_output_fields is fake_fields
    np.testing.assert_allclose(seen["particle"], np.asarray([[1.0, 2.0]]))
    np.testing.assert_allclose(seen["heat"], np.asarray([[3.0, 4.0]]))
    np.testing.assert_allclose(seen["flow"], np.asarray([[5.0, 6.0]]))


def test_compute_transport_postsolve_diagnostics_chunks_batched_nonstreamed_path(monkeypatch) -> None:
    calls: list[np.ndarray] = []

    def fake_diag_batch(*, op0, x_full_stack):
        del op0
        arr = np.asarray(x_full_stack, dtype=np.float64)
        calls.append(arr.copy())
        values = arr[:, 0]
        return SimpleNamespace(
            particle_flux_vm_psi_hat=jnp.asarray(values[:, None]),
            heat_flux_vm_psi_hat=jnp.asarray((10.0 + values)[:, None]),
            fsab_flow=jnp.asarray((20.0 + values)[:, None]),
        )

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DIAG_CHUNK", "2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE", "0")
    monkeypatch.setenv("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS", "0")
    monkeypatch.setattr(transport_finalize, "v3_transport_diagnostics_vm_only_batch_op0_jit", fake_diag_batch)
    monkeypatch.setattr(
        transport_finalize,
        "v3_transport_matrix_from_flux_arrays",
        lambda **kwargs: kwargs["particle_flux_vm_psi_hat"] + kwargs["heat_flux_vm_psi_hat"],
    )

    out = compute_transport_postsolve_diagnostics(
        op0=SimpleNamespace(n_species=1, total_size=10),
        geom=object(),
        state_vectors={
            1: jnp.asarray([1.0, 0.0]),
            2: jnp.asarray([2.0, 0.0]),
            3: jnp.asarray([3.0, 0.0]),
        },
        which_rhs_values=[1, 2, 3],
        stream_diagnostics=False,
        streaming_outputs=None,
        use_diag_op0=True,
        diag_op_by_index=None,
    )

    assert [call.shape[0] for call in calls] == [2, 1]
    np.testing.assert_allclose(np.asarray(out.particle_flux_vm_psi_hat), np.asarray([[1.0, 2.0, 3.0]]))
    np.testing.assert_allclose(np.asarray(out.heat_flux_vm_psi_hat), np.asarray([[11.0, 12.0, 13.0]]))
    np.testing.assert_allclose(np.asarray(out.fsab_flow), np.asarray([[21.0, 22.0, 23.0]]))
    np.testing.assert_allclose(np.asarray(out.transport_matrix), np.asarray([[12.0, 14.0, 16.0]]))
    assert out.transport_output_fields is None
