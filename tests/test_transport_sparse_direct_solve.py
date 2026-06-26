from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.problems.transport_matrix.solve as sparse_direct
from sfincs_jax.problems.transport_matrix.solve import (
    TransportSparseDirectContext,
    transport_sparse_direct_context_from_env,
    transport_sparse_direct_pattern_for_solve,
    transport_sparse_direct_solve,
)


def _op(*, total_size: int = 2, rhs_mode: int = 2):
    return SimpleNamespace(
        total_size=total_size,
        rhs_mode=rhs_mode,
        include_phi1=False,
        n_x=4,
        fblock=SimpleNamespace(fp=object()),
    )


def _context(**overrides) -> TransportSparseDirectContext:
    values = dict(
        op=_op(),
        factor_cache={},
        pattern_cache={},
        sparse_drop_tol=0.0,
        sparse_drop_rel=0.0,
        emit=None,
        sparse_factor_cache_key=lambda cache_key, factor_dtype: (*cache_key, np.dtype(factor_dtype).str),
        hash_numpy_array_for_cache=lambda array: ("hash", tuple(np.asarray(array).reshape((-1,)).tolist())),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: (_fake_operator_bundle(), _fake_factor_bundle()),
        build_sparse_ilu_from_matvec=_fake_build_sparse_ilu,
        try_build_direct_active_operator_bundle=lambda **_kwargs: None,
        host_sparse_direct_solve_with_refinement=_fake_refined_solve,
        host_sparse_direct_refine_steps=lambda *_args, **_kwargs: 2,
        host_sparse_direct_polish=lambda **_kwargs: (np.zeros((2,)), 0.0),
        sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
        sparse_direct_use_explicit_helper=lambda **_kwargs: False,
        sparse_direct_needs_float64_retry=lambda **_kwargs: False,
    )
    values.update(overrides)
    return TransportSparseDirectContext(**values)


def _context_from_env(**overrides) -> TransportSparseDirectContext:
    values = dict(
        op=_op(),
        emit=None,
        sparse_factor_cache_key=lambda cache_key, factor_dtype: (*cache_key, np.dtype(factor_dtype).str),
        hash_numpy_array_for_cache=lambda array: ("hash", tuple(np.asarray(array).reshape((-1,)).tolist())),
        build_host_sparse_direct_factor_from_matvec=lambda **_kwargs: (_fake_operator_bundle(), _fake_factor_bundle()),
        build_sparse_ilu_from_matvec=_fake_build_sparse_ilu,
        try_build_direct_active_operator_bundle=lambda **_kwargs: None,
        host_sparse_direct_solve_with_refinement=_fake_refined_solve,
        host_sparse_direct_refine_steps=lambda *_args, **_kwargs: 2,
        host_sparse_direct_polish=lambda **_kwargs: (np.zeros((2,)), 0.0),
        sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float64),
        sparse_direct_use_explicit_helper=lambda **_kwargs: False,
        sparse_direct_needs_float64_retry=lambda **_kwargs: False,
    )
    values.update(overrides)
    return transport_sparse_direct_context_from_env(**values)


def _fake_operator_bundle():
    return SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="unit"),
        matrix=np.eye(2),
    )


def _fake_factor_bundle():
    return SimpleNamespace(
        operator=SimpleNamespace(matrix=np.eye(2)),
        factor=object(),
        kind="lu",
        factor_s=0.0,
        factor_nbytes_estimate=16,
    )


def _fake_build_sparse_ilu(**_kwargs):
    return np.eye(2), None, object(), None, None, None, None


def _fake_refined_solve(*, rhs_vec, **_kwargs):
    rhs = np.asarray(rhs_vec, dtype=np.float64)
    return rhs.copy(), 0.0


def test_transport_sparse_direct_context_from_env_parses_policy_and_uses_fresh_caches(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", "2.5e-4")

    first = _context_from_env()
    second = _context_from_env()

    assert first.sparse_drop_tol == 0.0
    assert first.sparse_drop_rel == 2.5e-4
    assert first.factor_cache == {}
    assert first.pattern_cache == {}
    assert first.factor_cache is not second.factor_cache
    assert first.pattern_cache is not second.pattern_cache


def test_transport_sparse_direct_pattern_for_solve_uses_cache_and_emits(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=3)
    emitted: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    monkeypatch.setattr(sparse_direct, "v3_full_system_conservative_sparsity_pattern", lambda _op: pattern)
    monkeypatch.setattr(
        sparse_direct,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(shape=(2, 2), nnz=3, avg_row_nnz=1.5, max_row_nnz=2),
    )
    monkeypatch.setattr(sparse_direct, "estimate_csr_nbytes", lambda _shape, _nnz: 24)
    context = _context(emit=lambda _level, message: emitted.append(str(message)))

    first = transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None)
    second = transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None)

    assert first is pattern
    assert second is pattern
    assert len(context.pattern_cache) == 1
    assert any("transport sparse pattern selected" in message for message in emitted)


def test_transport_sparse_direct_context_pattern_method_uses_owner_policy(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=2)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    monkeypatch.setattr(sparse_direct, "v3_full_system_conservative_sparsity_pattern", lambda _op: pattern)
    monkeypatch.setattr(
        sparse_direct,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(shape=(2, 2), nnz=2, avg_row_nnz=1.0, max_row_nnz=1),
    )
    monkeypatch.setattr(sparse_direct, "estimate_csr_nbytes", lambda _shape, _nnz: 16)
    context = _context()

    assert context.pattern_for_solve(n=2, active_indices_np=None) is pattern


def test_transport_sparse_direct_solve_uses_sparse_ilu_refinement_path(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "0")
    context = _context()
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    result = transport_sparse_direct_solve(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=rhs,
        n=2,
        dtype=jnp.float64,
        cache_key=("unit",),
        active_indices_np=None,
        tol_val=1e-10,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        precondition_side_val="left",
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(rhs), rtol=1e-12, atol=1e-12)
    assert float(result.residual_norm) == 0.0


def test_transport_sparse_direct_context_solve_method_uses_true_residual_gate(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "0")
    context = _context()
    rhs = jnp.asarray([0.5, -0.25], dtype=jnp.float64)

    result = context.solve(
        matvec_fn=lambda x: x,
        b_vec=rhs,
        n=2,
        dtype=jnp.float64,
        cache_key=("unit-method",),
        active_indices_np=None,
        tol_val=1e-10,
        atol_val=1e-12,
        restart_val=10,
        maxiter_val=20,
        precondition_side_val="left",
    )

    np.testing.assert_allclose(np.asarray(result.x), np.asarray(rhs), rtol=1e-12, atol=1e-12)
    assert float(result.residual_norm) == 0.0
