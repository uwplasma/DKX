from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.problems.transport_solve as sparse_direct
from sfincs_jax.problems.transport_solve import (
    TransportSparseDirectContext,
    _build_explicit_helper_factor,
    _build_pattern_factor,
    _direct_active_factor_options,
    _maybe_build_direct_active_true_factor,
    _maybe_polish_float32_factor,
    _read_float_env,
    _read_int_env,
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


def test_transport_sparse_direct_pattern_rejects_disabled_and_ineligible_cases(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=1)
    monkeypatch.setattr(sparse_direct, "v3_full_system_conservative_sparsity_pattern", lambda _op: pattern)
    context = _context(op=_op(total_size=3, rhs_mode=2))

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "off")
    assert transport_sparse_direct_pattern_for_solve(context=context, n=3, active_indices_np=None) is None

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "")
    assert transport_sparse_direct_pattern_for_solve(context=context, n=3, active_indices_np=None) is None

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    assert transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None) is None
    assert (
        transport_sparse_direct_pattern_for_solve(
            context=context,
            n=3,
            active_indices_np=np.asarray([0, 2], dtype=np.int64),
        )
        is None
    )


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


def test_transport_sparse_direct_pattern_for_active_indices_uses_distinct_cache(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=4)
    calls: list[tuple[int, ...]] = []
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")

    def fake_pattern_for_indices(_op, active):
        calls.append(tuple(np.asarray(active, dtype=np.int64).tolist()))
        return pattern

    monkeypatch.setattr(sparse_direct, "v3_full_system_conservative_sparsity_pattern_for_indices", fake_pattern_for_indices)
    monkeypatch.setattr(
        sparse_direct,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(shape=(2, 2), nnz=4, avg_row_nnz=2.0, max_row_nnz=2),
    )
    monkeypatch.setattr(sparse_direct, "estimate_csr_nbytes", lambda _shape, _nnz: 32)
    context = _context(op=_op(total_size=4))
    active = np.asarray([0, 3], dtype=np.int64)

    first = transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=active)
    second = transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=active)

    assert first is pattern
    assert second is pattern
    assert calls == [(0, 3)]
    assert len(context.pattern_cache) == 1


def test_transport_sparse_direct_pattern_budget_force_raises_and_auto_falls_back(monkeypatch) -> None:
    pattern = SimpleNamespace(nnz=10_000)
    monkeypatch.setattr(sparse_direct, "v3_full_system_conservative_sparsity_pattern", lambda _op: pattern)
    monkeypatch.setattr(
        sparse_direct,
        "summarize_v3_sparse_pattern",
        lambda _op, _pattern: SimpleNamespace(shape=(10, 10), nnz=10_000, avg_row_nnz=1000.0, max_row_nnz=1000),
    )
    monkeypatch.setattr(sparse_direct, "estimate_csr_nbytes", lambda _shape, _nnz: 10_000_000)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB", "0.001")
    with np.testing.assert_raises(MemoryError):
        transport_sparse_direct_pattern_for_solve(context=_context(), n=2, active_indices_np=None)

    emitted: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "")
    context = _context(
        op=_op(total_size=2, rhs_mode=3),
        emit=lambda _level, message: emitted.append(str(message)),
    )
    context.op.fblock = SimpleNamespace(fp=None)
    context.op.n_x = 1

    assert transport_sparse_direct_pattern_for_solve(context=context, n=2, active_indices_np=None) is None
    assert any("using matvec probing" in message for message in emitted)


def test_direct_active_true_factor_caches_factorized_operator(monkeypatch) -> None:
    calls = {"builder": 0, "factorize": 0}
    operator_bundle = SimpleNamespace(
        metadata=SimpleNamespace(storage_kind="csr", reason="direct-active-unit"),
        matrix=np.eye(2),
    )
    factor = object()
    factor_bundle = SimpleNamespace(
        operator=SimpleNamespace(matrix=np.eye(2)),
        factor=factor,
        kind="lu",
        factor_s=0.0,
        factor_nbytes_estimate=128,
    )

    def fake_builder(**_kwargs):
        calls["builder"] += 1
        return operator_bundle, {"kind": "unit"}

    def fake_factorize(op_bundle, **kwargs):
        calls["factorize"] += 1
        assert op_bundle is operator_bundle
        assert kwargs["kind"] == "lu"
        assert kwargs["permc_spec"] == "MMD_AT_PLUS_A"
        return factor_bundle

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", "lu")
    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize)
    context = _context(
        op=_op(total_size=2, rhs_mode=2),
        try_build_direct_active_operator_bundle=fake_builder,
    )

    first = _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    )
    second = _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    )

    assert first == (True, factor_bundle.operator.matrix, factor)
    assert second == (True, factor_bundle.operator.matrix, factor)
    assert calls == {"builder": 1, "factorize": 1}
    assert len(context.factor_cache) == 1


def test_direct_active_true_factor_rejects_ineligible_shapes(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR", "1")
    context = _context(op=_op(total_size=2, rhs_mode=2))

    assert _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=None,
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    ) == (False, None, None)

    context.op.include_phi1 = True
    assert _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    ) == (False, None, None)

    context.op.include_phi1 = False
    context.op.rhs_mode = 1
    assert _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    ) == (False, None, None)

    context.op.rhs_mode = 2
    context.op.fblock = SimpleNamespace(fp=None)
    assert _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=2,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    ) == (False, None, None)

    context.op.fblock = SimpleNamespace(fp=object())
    assert _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=np.asarray([0, 1], dtype=np.int64),
        n=3,
        cache_key=("direct",),
        factor_dtype_use=np.dtype(np.float64),
    ) == (False, None, None)


def test_transport_sparse_env_readers_and_direct_factor_options_parse_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_UNIT_INT", "bad")
    monkeypatch.setenv("SFINCS_JAX_UNIT_FLOAT", "bad")
    assert _read_int_env("SFINCS_JAX_UNIT_INT", default=7) == 7
    assert _read_float_env("SFINCS_JAX_UNIT_FLOAT", default=2.5) == 2.5

    monkeypatch.setenv("SFINCS_JAX_UNIT_INT", "11")
    monkeypatch.setenv("SFINCS_JAX_UNIT_FLOAT", "1.25")
    assert _read_int_env("SFINCS_JAX_UNIT_INT", default=7) == 11
    assert _read_float_env("SFINCS_JAX_UNIT_FLOAT", default=2.5) == 1.25

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_FILL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_DROP_TOL", raising=False)
    assert _direct_active_factor_options(n=50_000) == ("lu", 6.0, 1.0e-4)
    assert _direct_active_factor_options(n=50_001) == ("ilu", 6.0, 1.0e-4)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", "spilu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_FILL", "bad")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_DROP_TOL", "bad")
    assert _direct_active_factor_options(n=2) == ("ilu", 6.0, 1.0e-4)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", "exact")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_FILL", "3.5")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_DROP_TOL", "2e-3")
    assert _direct_active_factor_options(n=100_000) == ("lu", 3.5, 2.0e-3)


def test_build_pattern_factor_uses_color_batch_and_factor_cache(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        calls.append(dict(kwargs))
        return _fake_operator_bundle(), _fake_factor_bundle()

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_COLOR_BATCH", "bad")
    context = _context(build_host_sparse_direct_factor_from_matvec=fake_builder)
    pattern = SimpleNamespace(nnz=5)

    first_matrix, first_factor = _build_pattern_factor(
        context=context,
        matvec_fn=lambda x: x,
        n=2,
        dtype=jnp.float64,
        cache_key=("pattern",),
        factor_dtype_use=np.dtype(np.float64),
        pattern=pattern,
    )
    second_matrix, second_factor = _build_pattern_factor(
        context=context,
        matvec_fn=lambda x: x,
        n=2,
        dtype=jnp.float64,
        cache_key=("pattern",),
        factor_dtype_use=np.dtype(np.float64),
        pattern=pattern,
    )

    assert calls[0]["default_pattern_color_batch"] == 8
    assert calls[0]["default_factor_kind"] == "lu"
    np.testing.assert_allclose(first_matrix, np.eye(2))
    np.testing.assert_allclose(second_matrix, np.eye(2))
    assert first_factor is second_factor
    assert len(calls) == 1


def test_build_explicit_helper_factor_uses_policy_and_factor_cache(monkeypatch) -> None:
    calls = {"operator": 0, "factor": 0}
    operator_bundle = _fake_operator_bundle()
    factor_bundle = _fake_factor_bundle()

    def fake_build_operator(_matvec_host, **kwargs):
        calls["operator"] += 1
        assert kwargs["block_cols"] == 4
        assert kwargs["dense_max_mb"] == 12.5
        assert kwargs["csr_max_mb"] == 24.5
        assert kwargs["allow_operator_only"] is False
        return operator_bundle

    def fake_factorize(bundle, **kwargs):
        calls["factor"] += 1
        assert bundle is operator_bundle
        assert kwargs["kind"] == "lu"
        return factor_bundle

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_BLOCK_COLS", "4")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_DENSE_MAX_MB", "12.5")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CSR_MAX_MB", "24.5")
    monkeypatch.setattr(sparse_direct, "build_operator_from_matvec", fake_build_operator)
    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fake_factorize)
    context = _context()

    first_matrix, first_factor = _build_explicit_helper_factor(
        context=context,
        matvec_fn=lambda x: x,
        n=2,
        dtype=jnp.float64,
        cache_key=("explicit",),
        factor_dtype_use=np.dtype(np.float64),
    )
    second_matrix, second_factor = _build_explicit_helper_factor(
        context=context,
        matvec_fn=lambda x: x,
        n=2,
        dtype=jnp.float64,
        cache_key=("explicit",),
        factor_dtype_use=np.dtype(np.float64),
    )

    np.testing.assert_allclose(first_matrix, np.eye(2))
    np.testing.assert_allclose(second_matrix, np.eye(2))
    assert first_factor is second_factor
    assert calls == {"operator": 1, "factor": 1}


def test_maybe_polish_float32_factor_respects_policy_and_true_residual(monkeypatch) -> None:
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)
    x0 = np.asarray([0.0, 0.0], dtype=np.float64)
    improved = np.asarray([1.0, -1.0], dtype=np.float64)
    calls: list[dict[str, object]] = []

    def fake_polish(**kwargs):
        calls.append(dict(kwargs))
        return improved, 0.25

    context = _context(host_sparse_direct_polish=fake_polish)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH", "0")
    x_off, residual_off = _maybe_polish_float32_factor(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=rhs,
        x_np=x0,
        residual_norm=10.0,
        ilu_for_polish=object(),
        factor_dtype=np.dtype(np.float32),
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=20,
        maxiter_val=40,
        precondition_side_val="left",
        true_residual_norm=lambda x: float(np.linalg.norm(np.asarray(x) - np.asarray(rhs))),
    )
    np.testing.assert_allclose(x_off, x0)
    assert residual_off == 10.0
    assert calls == []

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH", "1")
    x_on, residual_on = _maybe_polish_float32_factor(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=rhs,
        x_np=x0,
        residual_norm=10.0,
        ilu_for_polish=object(),
        factor_dtype=np.dtype(np.float32),
        tol_val=1.0e-8,
        atol_val=1.0e-12,
        restart_val=20,
        maxiter_val=40,
        precondition_side_val="right",
        true_residual_norm=lambda x: float(np.linalg.norm(np.asarray(x) - np.asarray(rhs))),
    )

    np.testing.assert_allclose(x_on, improved)
    assert residual_on == 0.0
    assert calls[-1]["precondition_side"] == "right"


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


def test_transport_sparse_direct_solve_retries_float64_when_float32_true_residual_fails(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "0")
    factor_dtypes: list[np.dtype] = []

    def fake_sparse_ilu_from_matvec(**kwargs):
        factor_dtype = np.dtype(kwargs["factor_dtype"])
        factor_dtypes.append(factor_dtype)
        return np.eye(2), None, {"factor_dtype": factor_dtype.name}, None, None, None, None

    def fake_refined_solve(*, rhs_vec, factor_dtype, **_kwargs):
        rhs = np.asarray(rhs_vec, dtype=np.float64)
        if np.dtype(factor_dtype) == np.dtype(np.float32):
            return np.zeros_like(rhs), 10.0
        return rhs.copy(), 0.0

    context = _context(
        build_sparse_ilu_from_matvec=fake_sparse_ilu_from_matvec,
        host_sparse_direct_solve_with_refinement=fake_refined_solve,
        sparse_factor_dtype=lambda **_kwargs: np.dtype(np.float32),
        sparse_direct_needs_float64_retry=lambda *, factor_dtype, residual_norm, target_true: (
            np.dtype(factor_dtype) == np.dtype(np.float32) and float(residual_norm) > float(target_true)
        ),
    )
    rhs = jnp.asarray([1.0, -2.0], dtype=jnp.float64)

    result = transport_sparse_direct_solve(
        context=context,
        matvec_fn=lambda x: x,
        b_vec=rhs,
        n=2,
        dtype=jnp.float64,
        cache_key=("retry",),
        active_indices_np=None,
        tol_val=1.0e-10,
        atol_val=1.0e-12,
        restart_val=10,
        maxiter_val=20,
        precondition_side_val="left",
    )

    assert factor_dtypes == [np.dtype(np.float32), np.dtype(np.float64)]
    np.testing.assert_allclose(np.asarray(result.x), np.asarray(rhs), rtol=1.0e-12, atol=1.0e-12)
    assert float(result.residual_norm) == 0.0


def test_transport_sparse_direct_solve_fails_fast_when_sparse_factors_are_unavailable(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "0")

    def no_factors(**_kwargs):
        return np.eye(2), None, None, None, None, None, None

    context = _context(build_sparse_ilu_from_matvec=no_factors)

    with np.testing.assert_raises(RuntimeError):
        transport_sparse_direct_solve(
            context=context,
            matvec_fn=lambda x: x,
            b_vec=jnp.ones((2,), dtype=jnp.float64),
            n=2,
            dtype=jnp.float64,
            cache_key=("missing-factor",),
            active_indices_np=None,
            tol_val=1.0e-10,
            atol_val=1.0e-12,
            restart_val=10,
            maxiter_val=20,
            precondition_side_val="left",
        )


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
