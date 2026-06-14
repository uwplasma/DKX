from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
from jax import config as jax_config

import sfincs_jax.transport_dense_batch as dense_batch
from sfincs_jax.transport_dense_batch import (
    TransportDenseBatchContext,
    solve_transport_dense_batch,
)

jax_config.update("jax_enable_x64", True)


def _patch_fake_operator(monkeypatch) -> None:
    monkeypatch.setattr(
        dense_batch,
        "apply_v3_full_system_operator_cached",
        lambda op, x: op.matrix @ x,
    )
    monkeypatch.setattr(dense_batch, "_operator_signature_cached", lambda op: op.signature)


def _identity_projection(x_col: jnp.ndarray, **_kwargs) -> jnp.ndarray:
    return x_col


def _context(
    *,
    op,
    rhs_by_index,
    which_rhs_values=(1, 2),
    use_active_dof_mode=False,
    active_size=0,
    reduce_full=None,
    expand_reduced=None,
    store_state_vectors=True,
    stream_diagnostics=False,
    collect_transport_outputs=None,
    emit=None,
    op_matvec_by_index=None,
) -> tuple[TransportDenseBatchContext, dict[int, jnp.ndarray], dict[int, jnp.ndarray]]:
    residual_norms: dict[int, jnp.ndarray] = {}
    state_vectors: dict[int, jnp.ndarray] = {}
    rhs_norms = {
        int(which_rhs): jnp.linalg.norm(rhs_by_index[idx])
        for idx, which_rhs in enumerate(which_rhs_values)
    }
    context = TransportDenseBatchContext(
        dense_backend_allowed=True,
        dense_use_mixed=False,
        use_active_dof_mode=bool(use_active_dof_mode),
        active_size=int(active_size),
        op0=op,
        op_matvec_by_index=op_matvec_by_index or [op for _ in rhs_by_index],
        rhs_by_index=rhs_by_index,
        which_rhs_values=which_rhs_values,
        rhs_norms=rhs_norms,
        residual_norms=residual_norms,
        solver_kinds_by_rhs={},
        solve_methods_by_rhs={},
        elapsed_s=np.zeros((len(which_rhs_values),), dtype=np.float64),
        state_vectors=state_vectors,
        store_state_vectors=bool(store_state_vectors),
        stream_diagnostics=bool(stream_diagnostics),
        rhs3_krylov_flags=lambda _which_rhs: (False, False),
        maybe_project_constraint_nullspace=_identity_projection,
        collect_transport_outputs=collect_transport_outputs,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        emit=emit,
    )
    return context, state_vectors, residual_norms


def test_transport_dense_batch_solves_full_system_and_streams_outputs(monkeypatch) -> None:
    _patch_fake_operator(monkeypatch)
    matrix = jnp.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)
    op = SimpleNamespace(total_size=2, matrix=matrix, signature=("full", 2))
    rhs_by_index = [
        jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        jnp.asarray([4.0, 5.0], dtype=jnp.float64),
    ]
    collected: list[tuple[int, np.ndarray]] = []
    emitted: list[tuple[int, str]] = []
    context, state_vectors, residual_norms = _context(
        op=op,
        rhs_by_index=rhs_by_index,
        stream_diagnostics=True,
        collect_transport_outputs=lambda which_rhs, x: collected.append((int(which_rhs), np.asarray(x))),
        emit=lambda level, message: emitted.append((int(level), str(message))),
    )

    assert solve_transport_dense_batch(context=context, op_probe_ref=op, reason="unit") is True

    for idx, which_rhs in enumerate((1, 2)):
        expected = np.linalg.solve(np.asarray(matrix), np.asarray(rhs_by_index[idx]))
        np.testing.assert_allclose(np.asarray(state_vectors[which_rhs]), expected, rtol=1e-12, atol=1e-12)
        assert float(residual_norms[which_rhs]) < 1e-12
        assert context.solver_kinds_by_rhs[which_rhs] == "dense"
        assert context.solve_methods_by_rhs[which_rhs] == "dense"
    assert [which_rhs for which_rhs, _x in collected] == [1, 2]
    assert any("dense batched solve across all whichRHS" in message for _level, message in emitted)


def test_transport_dense_batch_active_dof_returns_full_state(monkeypatch) -> None:
    _patch_fake_operator(monkeypatch)
    matrix = jnp.asarray(
        [
            [2.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    op = SimpleNamespace(total_size=3, matrix=matrix, signature=("active", 3))
    rhs_by_index = [
        jnp.asarray([2.0, 0.0, 8.0], dtype=jnp.float64),
        jnp.asarray([4.0, 0.0, 12.0], dtype=jnp.float64),
    ]

    def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[jnp.asarray([0, 2])]

    def expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray([v_reduced[0], 0.0, v_reduced[1]], dtype=v_reduced.dtype)

    context, state_vectors, residual_norms = _context(
        op=op,
        rhs_by_index=rhs_by_index,
        use_active_dof_mode=True,
        active_size=2,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )

    assert solve_transport_dense_batch(context=context, op_probe_ref=op, reason="active unit") is True

    np.testing.assert_allclose(np.asarray(state_vectors[1]), np.asarray([1.0, 0.0, 2.0]), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(state_vectors[2]), np.asarray([2.0, 0.0, 3.0]), rtol=1e-12, atol=1e-12)
    assert float(residual_norms[1]) < 1e-12
    assert float(residual_norms[2]) < 1e-12


def test_transport_dense_batch_rejects_varying_operator(monkeypatch) -> None:
    _patch_fake_operator(monkeypatch)
    matrix = jnp.eye(2, dtype=jnp.float64)
    op_a = SimpleNamespace(total_size=2, matrix=matrix, signature=("a",))
    op_b = SimpleNamespace(total_size=2, matrix=matrix, signature=("b",))
    emitted: list[str] = []
    context, state_vectors, _residual_norms = _context(
        op=op_a,
        rhs_by_index=[
            jnp.asarray([1.0, 0.0], dtype=jnp.float64),
            jnp.asarray([0.0, 1.0], dtype=jnp.float64),
        ],
        op_matvec_by_index=[op_a, op_b],
        emit=lambda _level, message: emitted.append(str(message)),
    )

    assert solve_transport_dense_batch(context=context, op_probe_ref=op_a, reason="unit") is False
    assert state_vectors == {}
    assert any("dense batch disabled" in message for message in emitted)
