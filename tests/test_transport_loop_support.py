from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

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
