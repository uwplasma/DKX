"""Sparse-direct rescue helpers for RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.explicit_sparse import build_operator_from_matvec, estimate_csr_nbytes, factorize_host_sparse_operator
from sfincs_jax.solver import GMRESSolveResult
from sfincs_jax.v3_sparse_pattern import (
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
)


EmitFn = Callable[[int, str], None]


@dataclass
class TransportSparseDirectContext:
    """Driver-owned state needed by the transport sparse-direct rescue path."""

    op: Any
    factor_cache: MutableMapping[tuple[object, ...], tuple[object, object, str, str]]
    pattern_cache: MutableMapping[tuple[object, ...], object]
    sparse_drop_tol: float
    sparse_drop_rel: float
    emit: EmitFn | None
    sparse_factor_cache_key: Callable[[tuple[object, ...], np.dtype], tuple[object, ...]]
    hash_numpy_array_for_cache: Callable[[np.ndarray], object]
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]]
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]]
    try_build_direct_active_operator_bundle: Callable[..., tuple[Any, Any] | None]
    host_sparse_direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    host_sparse_direct_refine_steps: Callable[..., int]
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]]
    sparse_factor_dtype: Callable[..., np.dtype]
    sparse_direct_use_explicit_helper: Callable[..., bool]
    sparse_direct_needs_float64_retry: Callable[..., bool]


def transport_sparse_direct_context_from_env(
    *,
    op: Any,
    emit: EmitFn | None,
    sparse_factor_cache_key: Callable[[tuple[object, ...], np.dtype], tuple[object, ...]],
    hash_numpy_array_for_cache: Callable[[np.ndarray], object],
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, Any]],
    try_build_direct_active_operator_bundle: Callable[..., tuple[Any, Any] | None],
    host_sparse_direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    host_sparse_direct_refine_steps: Callable[..., int],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
    sparse_factor_dtype: Callable[..., np.dtype],
    sparse_direct_use_explicit_helper: Callable[..., bool],
    sparse_direct_needs_float64_retry: Callable[..., bool],
) -> TransportSparseDirectContext:
    """Create the per-solve sparse-direct context and caches from env policy."""
    return TransportSparseDirectContext(
        op=op,
        factor_cache={},
        pattern_cache={},
        sparse_drop_tol=_read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_DROP_TOL", default=0.0),
        sparse_drop_rel=_read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_DROP_REL", default=0.0),
        emit=emit,
        sparse_factor_cache_key=sparse_factor_cache_key,
        hash_numpy_array_for_cache=hash_numpy_array_for_cache,
        build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
        build_sparse_ilu_from_matvec=build_sparse_ilu_from_matvec,
        try_build_direct_active_operator_bundle=try_build_direct_active_operator_bundle,
        host_sparse_direct_solve_with_refinement=host_sparse_direct_solve_with_refinement,
        host_sparse_direct_refine_steps=host_sparse_direct_refine_steps,
        host_sparse_direct_polish=host_sparse_direct_polish,
        sparse_factor_dtype=sparse_factor_dtype,
        sparse_direct_use_explicit_helper=sparse_direct_use_explicit_helper,
        sparse_direct_needs_float64_retry=sparse_direct_needs_float64_retry,
    )


def transport_sparse_direct_pattern_for_solve(
    *,
    context: TransportSparseDirectContext,
    n: int,
    active_indices_np: np.ndarray | None,
) -> object | None:
    """Return the conservative transport sparse pattern when policy admits it."""
    op = context.op
    raw = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN", "").strip().lower()
    if raw in {"0", "false", "no", "off", "dense", "matvec"}:
        return None
    force_pattern = raw in {"1", "true", "yes", "on", "pattern", "probe", "color_probe"}
    mono_pas_transport = (
        int(op.rhs_mode) == 3
        and not bool(op.include_phi1)
        and getattr(op.fblock, "fp", None) is None
        and int(getattr(op, "n_x", 0) or 0) <= 2
    )
    if not (force_pattern or mono_pas_transport):
        return None
    active_key = "full" if active_indices_np is None else context.hash_numpy_array_for_cache(active_indices_np)
    cache_key = ("transport_sparse_pattern", int(n), active_key)
    cached_pattern = context.pattern_cache.get(cache_key)
    if cached_pattern is not None:
        return cached_pattern
    if active_indices_np is None:
        if int(n) != int(op.total_size):
            return None
        pattern = v3_full_system_conservative_sparsity_pattern(op)
    else:
        active_np = np.asarray(active_indices_np, dtype=np.int32).reshape((-1,))
        if int(n) != int(active_np.size):
            return None
        pattern = v3_full_system_conservative_sparsity_pattern_for_indices(op, active_np)
    summary = summarize_v3_sparse_pattern(op, pattern)
    csr_estimate_mb = float(estimate_csr_nbytes(summary.shape, summary.nnz)) / 1.0e6
    max_mb_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_CSR_MAX_MB", "").strip()
    try:
        max_mb = float(max_mb_env) if max_mb_env else 512.0
    except ValueError:
        max_mb = 512.0
    if csr_estimate_mb > max(0.0, float(max_mb)):
        message = (
            "transport sparse-pattern assembly exceeds CSR budget "
            f"({csr_estimate_mb:.3f} MB > {float(max_mb):.3f} MB, nnz={summary.nnz})"
        )
        if force_pattern:
            raise MemoryError(message)
        if context.emit is not None:
            context.emit(1, f"solve_v3_transport_matrix_linear_gmres: {message}; using matvec probing")
        return None
    context.pattern_cache[cache_key] = pattern
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: transport sparse pattern selected "
            f"shape={summary.shape} nnz={summary.nnz} avg_row_nnz={summary.avg_row_nnz:.3f} "
            f"max_row_nnz={summary.max_row_nnz} csr_estimate_mb={csr_estimate_mb:.3f}",
        )
    return pattern


def transport_sparse_direct_solve(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    active_indices_np: np.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
) -> GMRESSolveResult:
    """Run the RHSMode=2/3 sparse-direct rescue with true-residual checks."""
    factor_dtype = context.sparse_factor_dtype(size=int(n), use_implicit=False)
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: sparse LU factor_dtype="
            f"{np.dtype(factor_dtype).name}",
        )
    target_true = max(float(atol_val), float(tol_val) * float(jnp.linalg.norm(b_vec)))
    x_np, residual_norm, ilu_for_polish = _solve_with_factor_dtype(
        context=context,
        matvec_fn=matvec_fn,
        b_vec=b_vec,
        n=int(n),
        dtype=dtype,
        cache_key=cache_key,
        active_indices_np=active_indices_np,
        factor_dtype_use=np.dtype(factor_dtype),
    )

    def true_residual_norm(x_arr: np.ndarray) -> float:
        ax = matvec_fn(jnp.asarray(x_arr, dtype=dtype))
        residual = np.asarray(ax - b_vec, dtype=np.float64).reshape((-1,))
        return float(np.linalg.norm(residual))

    true_residual = true_residual_norm(x_np)
    if np.isfinite(true_residual) and (
        (not np.isfinite(float(residual_norm))) or float(true_residual) > float(residual_norm)
    ):
        residual_norm = float(true_residual)
    if np.dtype(factor_dtype) == np.dtype(np.float32) and residual_norm > target_true:
        x_np, residual_norm = _maybe_polish_float32_factor(
            context=context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x_np=x_np,
            residual_norm=float(residual_norm),
            ilu_for_polish=ilu_for_polish,
            factor_dtype=np.dtype(factor_dtype),
            tol_val=float(tol_val),
            atol_val=float(atol_val),
            restart_val=int(restart_val),
            maxiter_val=maxiter_val,
            precondition_side_val=str(precondition_side_val),
            true_residual_norm=true_residual_norm,
        )
    if context.sparse_direct_needs_float64_retry(
        factor_dtype=np.dtype(factor_dtype),
        residual_norm=float(residual_norm),
        target_true=float(target_true),
    ):
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: retrying sparse LU with float64 factors "
                f"(residual={float(residual_norm):.6e}, target={float(target_true):.6e})",
            )
        x64_np, residual64, _ilu64 = _solve_with_factor_dtype(
            context=context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            active_indices_np=active_indices_np,
            factor_dtype_use=np.dtype(np.float64),
        )
        if np.isfinite(residual64) and (
            not np.isfinite(float(residual_norm)) or float(residual64) < float(residual_norm)
        ):
            x_np = x64_np
            residual_norm = residual64
    return GMRESSolveResult(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
    )


def _solve_with_factor_dtype(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    active_indices_np: np.ndarray | None,
    factor_dtype_use: np.dtype,
) -> tuple[np.ndarray, float, object]:
    direct_true_attempted, a_csr_full, ilu = _maybe_build_direct_active_true_factor(
        context=context,
        active_indices_np=active_indices_np,
        n=int(n),
        cache_key=cache_key,
        factor_dtype_use=np.dtype(factor_dtype_use),
    )
    if not direct_true_attempted:
        pattern = transport_sparse_direct_pattern_for_solve(
            context=context,
            n=int(n),
            active_indices_np=active_indices_np,
        )
    else:
        pattern = None
    if (not direct_true_attempted) and pattern is not None:
        a_csr_full, ilu = _build_pattern_factor(
            context=context,
            matvec_fn=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            factor_dtype_use=np.dtype(factor_dtype_use),
            pattern=pattern,
        )
    elif (not direct_true_attempted) and context.sparse_direct_use_explicit_helper(size=int(n)):
        a_csr_full, ilu = _build_explicit_helper_factor(
            context=context,
            matvec_fn=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key,
            factor_dtype_use=np.dtype(factor_dtype_use),
        )
    elif not direct_true_attempted:
        cache_key_use = context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use))
        a_csr_full, _a_csr_drop, ilu, _a_dense, _l_dense, _u_dense, _l_unit = context.build_sparse_ilu_from_matvec(
            matvec=matvec_fn,
            n=int(n),
            dtype=dtype,
            cache_key=cache_key_use,
            factor_dtype=np.dtype(factor_dtype_use),
            drop_tol=float(context.sparse_drop_tol),
            drop_rel=float(context.sparse_drop_rel),
            ilu_drop_tol=0.0,
            fill_factor=1.0,
            build_dense_factors=False,
            build_jax_factors=False,
            build_ilu=True,
            store_dense=False,
            factorization="lu",
            emit=context.emit,
        )
        if ilu is None:
            raise RuntimeError("transport sparse_lu: factors unavailable")
    x_local, residual_local = context.host_sparse_direct_solve_with_refinement(
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs_vec=b_vec,
        factor_dtype=np.dtype(factor_dtype_use),
        refine_steps=context.host_sparse_direct_refine_steps(
            "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_REFINE",
            default=2,
        ),
    )
    return x_local, float(residual_local), ilu


def _maybe_build_direct_active_true_factor(
    *,
    context: TransportSparseDirectContext,
    active_indices_np: np.ndarray | None,
    n: int,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
) -> tuple[bool, object | None, object | None]:
    op = context.op
    direct_true_enabled_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR", "").strip().lower()
    direct_true_enabled = direct_true_enabled_env not in {"0", "false", "no", "off"}
    if not (
        bool(direct_true_enabled)
        and int(op.rhs_mode) in {2, 3}
        and getattr(op.fblock, "fp", None) is not None
        and not bool(op.include_phi1)
        and active_indices_np is not None
        and int(n) == int(np.asarray(active_indices_np).size)
    ):
        return False, None, None

    direct_factor_kind, direct_ilu_fill, direct_ilu_drop = _direct_active_factor_options(n=int(n))
    active_hash_direct = context.hash_numpy_array_for_cache(np.asarray(active_indices_np, dtype=np.int64))
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "direct_active_true_fp_operator",
        str(jax.default_backend()),
        str(active_hash_direct),
        str(direct_factor_kind),
        float(direct_ilu_fill),
        float(direct_ilu_drop),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing direct active true FP operator "
                f"storage={storage_kind} reason={reason}",
            )
        return True, a_csr_full, ilu

    direct_result = context.try_build_direct_active_operator_bundle(
        op=op,
        active_indices=np.asarray(active_indices_np, dtype=np.int64),
        factor_dtype=np.dtype(factor_dtype_use),
        emit=context.emit,
    )
    if direct_result is None:
        return False, None, None
    operator_bundle, _direct_metadata = direct_result
    factor_bundle = factorize_host_sparse_operator(
        operator_bundle,
        kind=direct_factor_kind,
        drop_tol=float(direct_ilu_drop) if direct_factor_kind == "ilu" else 0.0,
        fill_factor=float(direct_ilu_fill) if direct_factor_kind == "ilu" else 1.0,
        permc_spec="MMD_AT_PLUS_A",
        diag_pivot_thresh=0.0,
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    context.factor_cache[factor_cache_key] = (
        a_csr_full,
        ilu,
        str(operator_bundle.metadata.storage_kind),
        str(operator_bundle.metadata.reason),
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: direct active true FP operator "
            f"factorization complete factor_kind={factor_bundle.kind} "
            f"factor_s={float(factor_bundle.factor_s or 0.0):.3f} "
            f"factor_mb={float(factor_bundle.factor_nbytes_estimate or 0) / 1.0e6:.3f}",
        )
    return True, a_csr_full, ilu


def _direct_active_factor_options(*, n: int) -> tuple[str, float, float]:
    factor_kind_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_FACTOR", "").strip().lower()
    if factor_kind_env in {"lu", "exact"}:
        direct_factor_kind = "lu"
    elif factor_kind_env in {"ilu", "spilu", "incomplete"}:
        direct_factor_kind = "ilu"
    else:
        direct_factor_kind = "lu" if int(n) <= 50_000 else "ilu"
    fill_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_FILL", "").strip()
    drop_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_OPERATOR_ILU_DROP_TOL", "").strip()
    try:
        direct_ilu_fill = float(fill_env) if fill_env else 6.0
    except ValueError:
        direct_ilu_fill = 6.0
    try:
        direct_ilu_drop = float(drop_env) if drop_env else 1.0e-4
    except ValueError:
        direct_ilu_drop = 1.0e-4
    return direct_factor_kind, direct_ilu_fill, direct_ilu_drop


def _build_pattern_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
    pattern: object,
) -> tuple[object, object]:
    color_batch_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_PATTERN_COLOR_BATCH", "").strip()
    try:
        color_batch = int(color_batch_env) if color_batch_env else 8
    except ValueError:
        color_batch = 8
    color_batch = max(1, int(color_batch))
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "pattern_probe",
        int(getattr(pattern, "nnz", 0)),
        int(color_batch),
        str(jax.default_backend()),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing pattern sparse helper "
                f"storage={storage_kind} reason={reason}",
            )
        return a_csr_full, ilu
    operator_bundle, factor_bundle = context.build_host_sparse_direct_factor_from_matvec(
        matvec=matvec_fn,
        n=int(n),
        dtype=dtype,
        factor_dtype=np.dtype(factor_dtype_use),
        pattern=pattern,
        emit=context.emit,
        default_factor_kind="lu",
        default_pattern_color_batch=int(color_batch),
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    metadata = getattr(operator_bundle, "metadata", None)
    storage_kind = str(getattr(metadata, "storage_kind", "unknown"))
    reason = str(getattr(metadata, "reason", "unknown"))
    context.factor_cache[factor_cache_key] = (a_csr_full, ilu, storage_kind, reason)
    return a_csr_full, ilu


def _build_explicit_helper_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    factor_dtype_use: np.dtype,
) -> tuple[object, object]:
    block_cols = _read_int_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_BLOCK_COLS", default=32)
    dense_max_mb = _read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_DENSE_MAX_MB", default=128.0)
    csr_max_mb = _read_float_env("SFINCS_JAX_TRANSPORT_SPARSE_HELPER_CSR_MAX_MB", default=512.0)

    def matvec_host(x_np: np.ndarray) -> np.ndarray:
        return np.asarray(
            matvec_fn(jnp.asarray(x_np, dtype=dtype)),
            dtype=np.dtype(factor_dtype_use),
            copy=True,
        )

    def matmat_host(cols_np: np.ndarray) -> np.ndarray:
        cols = jnp.asarray(cols_np, dtype=dtype)
        out = jax.vmap(matvec_fn, in_axes=1, out_axes=1)(cols)
        return np.asarray(out, dtype=np.dtype(factor_dtype_use), copy=True)

    force_sparse = bool(jax.default_backend() != "cpu")
    factor_cache_key = (
        *context.sparse_factor_cache_key(cache_key, np.dtype(factor_dtype_use)),
        "explicit_helper",
        str(jax.default_backend()),
        int(max(1, int(block_cols))),
        float(dense_max_mb),
        float(csr_max_mb),
        int(force_sparse),
    )
    cached_factor = context.factor_cache.get(factor_cache_key)
    if cached_factor is not None:
        a_csr_full, ilu, storage_kind, reason = cached_factor
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: reusing explicit sparse helper "
                f"storage={storage_kind} reason={reason}",
            )
        return a_csr_full, ilu
    operator_bundle = build_operator_from_matvec(
        matvec_host,
        n=int(n),
        dtype=np.dtype(factor_dtype_use),
        backend=jax.default_backend(),
        block_cols=max(1, int(block_cols)),
        dense_max_mb=float(dense_max_mb),
        csr_max_mb=float(csr_max_mb),
        prefer_sparse_on_gpu=True,
        force_sparse=force_sparse,
        drop_tol=0.0,
        matmat=matmat_host,
        allow_operator_only=False,
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: explicit sparse helper "
            f"storage={operator_bundle.metadata.storage_kind} "
            f"reason={operator_bundle.metadata.reason}",
        )
    factor_bundle = factorize_host_sparse_operator(
        operator_bundle,
        kind="lu",
        drop_tol=0.0,
        fill_factor=1.0,
    )
    a_csr_full = factor_bundle.operator.matrix
    ilu = factor_bundle.factor
    context.factor_cache[factor_cache_key] = (
        a_csr_full,
        ilu,
        str(operator_bundle.metadata.storage_kind),
        str(operator_bundle.metadata.reason),
    )
    return a_csr_full, ilu


def _maybe_polish_float32_factor(
    *,
    context: TransportSparseDirectContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    x_np: np.ndarray,
    residual_norm: float,
    ilu_for_polish: object,
    factor_dtype: np.dtype,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
    true_residual_norm: Callable[[np.ndarray], float],
) -> tuple[np.ndarray, float]:
    polish_env = os.environ.get("SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH", "").strip().lower()
    if polish_env in {"0", "false", "no", "off"}:
        return x_np, float(residual_norm)
    polish_restart = _read_int_env(
        "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH_RESTART",
        default=min(int(restart_val), 40),
    )
    polish_maxiter = _read_int_env(
        "SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_POLISH_MAXITER",
        default=min(max(40, int(maxiter_val or 120)), 120),
    )
    polish_restart = max(5, int(polish_restart))
    polish_maxiter = max(5, int(polish_maxiter))
    x_polish, residual_norm_polish = context.host_sparse_direct_polish(
        matvec_fn=matvec_fn,
        rhs_vec=b_vec,
        x0_np=x_np,
        ilu=ilu_for_polish,
        factor_dtype=np.dtype(factor_dtype),
        tol=float(tol_val),
        atol=float(atol_val),
        restart=polish_restart,
        maxiter=polish_maxiter,
        precondition_side=precondition_side_val,
    )
    if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm:
        x_np = x_polish
        residual_norm = residual_norm_polish
        true_residual = true_residual_norm(x_np)
        if np.isfinite(true_residual):
            residual_norm = float(true_residual)
    return x_np, float(residual_norm)


def _read_int_env(name: str, *, default: int) -> int:
    env = os.environ.get(name, "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def _read_float_env(name: str, *, default: float) -> float:
    env = os.environ.get(name, "").strip()
    try:
        return float(env) if env else float(default)
    except ValueError:
        return float(default)


__all__ = [
    "TransportSparseDirectContext",
    "transport_sparse_direct_context_from_env",
    "transport_sparse_direct_pattern_for_solve",
    "transport_sparse_direct_solve",
]
