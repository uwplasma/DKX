"""Host sparse factorization helpers for SFINCS-JAX preconditioners.

This module owns the non-differentiable sparse assembly/factor setup used by
CLI-oriented RHSMode=1 preconditioner paths. Keeping it outside ``v3_driver``
makes the solver orchestration easier to read while preserving the JAX
matrix-free path used by differentiable Python workflows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import jax
import jax.numpy as jnp
import numpy as np

from ....preconditioner_caches import _RHSMODE1_SPARSE_ILU_CACHE, _SparseILUCache
from ....preconditioner_context import sparse_structural_tol
from ....solver import assemble_dense_matrix_from_matvec
from ....v3_system import V3FullSystemOperator, apply_v3_full_system_operator_cached
from ....verbose import Timer


@dataclass(frozen=True)
class RHS1FullSystemMatrixFreeOperatorAdapter:
    """Duck-typed full-system operator for Galerkin/coarse corrections."""

    op: V3FullSystemOperator

    @property
    def shape(self) -> tuple[int, int]:
        size = int(self.op.total_size)
        return (size, size)

    @property
    def blocks(self) -> jnp.ndarray:
        return jnp.zeros((1,), dtype=jnp.float64)

    def matmat(self, matrix: jnp.ndarray) -> jnp.ndarray:
        mat = jnp.asarray(matrix, dtype=jnp.float64)
        if mat.ndim != 2 or int(mat.shape[0]) != int(self.op.total_size):
            raise ValueError(f"matrix must have shape ({int(self.op.total_size)}, ncols)")
        return jax.vmap(
            lambda column: apply_v3_full_system_operator_cached(self.op, column),
            in_axes=1,
            out_axes=1,
        )(mat)


def _row_nnz_cap(row_nnz_cap: int | None) -> int:
    if row_nnz_cap is not None:
        return max(0, int(row_nnz_cap))
    row_nnz_cap_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ROW_NNZ_MAX", "").strip()
    try:
        return max(0, int(row_nnz_cap_env) if row_nnz_cap_env else 256)
    except ValueError:
        return 256


def _regularization_settings(max_abs: float) -> tuple[float, float, int]:
    reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", "").strip()
    try:
        reg = float(reg_env) if reg_env else (1.0e-12 * float(max_abs))
    except ValueError:
        reg = 1.0e-12 * float(max_abs)

    singular_reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_SINGULAR_REG_REL", "").strip()
    try:
        singular_reg_rel = float(singular_reg_env) if singular_reg_env else 1.0e-10
    except ValueError:
        singular_reg_rel = 1.0e-10

    attempts_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", "").strip()
    try:
        attempts = int(attempts_env) if attempts_env else 3
    except ValueError:
        attempts = 3

    return max(0.0, float(reg)), max(0.0, float(singular_reg_rel)), max(1, int(attempts))


def _jax_factor_arrays_from_superlu(
    *,
    ilu: object,
    n: int,
    row_nnz_cap: int | None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, int, int, int]:
    cap = _row_nnz_cap(row_nnz_cap)
    perm_r = np.asarray(ilu.perm_r, dtype=np.int32, copy=True)
    perm_c = np.asarray(ilu.perm_c, dtype=np.int32, copy=True)
    inv_perm_c = np.argsort(perm_c).astype(np.int32, copy=False)
    l_csr = ilu.L.tocsr()
    u_csr = ilu.U.tocsr()

    lower_diag = np.asarray(l_csr.diagonal(), dtype=np.float64)
    lower_diag = np.where(lower_diag != 0.0, lower_diag, 1.0)
    upper_diag = np.asarray(u_csr.diagonal(), dtype=np.float64)
    upper_diag = np.where(np.isfinite(upper_diag) & (upper_diag != 0.0), upper_diag, 1.0)

    lower_cols: list[np.ndarray] = []
    lower_vals: list[np.ndarray] = []
    max_lower = 0
    for i in range(int(n)):
        rs = int(l_csr.indptr[i])
        re = int(l_csr.indptr[i + 1])
        cols = l_csr.indices[rs:re]
        vals = l_csr.data[rs:re]
        mask = cols < i
        cols = cols[mask].astype(np.int32, copy=False)
        vals = vals[mask].astype(np.float64, copy=False)
        if cap > 0 and int(cols.size) > cap:
            sel = np.argpartition(np.abs(vals), -cap)[-cap:]
            cols = cols[sel]
            vals = vals[sel]
        if cols.size:
            order = np.argsort(cols)
            cols = cols[order]
            vals = vals[order]
        lower_cols.append(cols)
        lower_vals.append(vals)
        max_lower = max(max_lower, int(cols.size))

    upper_cols: list[np.ndarray] = []
    upper_vals: list[np.ndarray] = []
    max_upper = 0
    for i in range(int(n)):
        rs = int(u_csr.indptr[i])
        re = int(u_csr.indptr[i + 1])
        cols = u_csr.indices[rs:re]
        vals = u_csr.data[rs:re]
        mask = cols > i
        cols_u = cols[mask].astype(np.int32, copy=False)
        vals_u = vals[mask].astype(np.float64, copy=False)
        if cap > 0 and int(cols_u.size) > cap:
            sel = np.argpartition(np.abs(vals_u), -cap)[-cap:]
            cols_u = cols_u[sel]
            vals_u = vals_u[sel]
        if cols_u.size:
            order = np.argsort(cols_u)
            cols_u = cols_u[order]
            vals_u = vals_u[order]
        upper_cols.append(cols_u)
        upper_vals.append(vals_u)
        max_upper = max(max_upper, int(cols_u.size))

    lower_idx = -np.ones((int(n), max_lower), dtype=np.int32)
    lower_val = np.zeros((int(n), max_lower), dtype=np.float64)
    upper_idx = -np.ones((int(n), max_upper), dtype=np.int32)
    upper_val = np.zeros((int(n), max_upper), dtype=np.float64)
    for i in range(int(n)):
        lower_k = int(lower_cols[i].size)
        if lower_k:
            lower_idx[i, :lower_k] = lower_cols[i]
            lower_val[i, :lower_k] = lower_vals[i]
        upper_k = int(upper_cols[i].size)
        if upper_k:
            upper_idx[i, :upper_k] = upper_cols[i]
            upper_val[i, :upper_k] = upper_vals[i]

    return (
        jnp.asarray(perm_r, dtype=jnp.int32),
        jnp.asarray(inv_perm_c, dtype=jnp.int32),
        jnp.asarray(lower_idx, dtype=jnp.int32),
        jnp.asarray(lower_val, dtype=jnp.float64),
        jnp.asarray(lower_diag, dtype=jnp.float64),
        jnp.asarray(upper_idx, dtype=jnp.int32),
        jnp.asarray(upper_val, dtype=jnp.float64),
        jnp.asarray(upper_diag, dtype=jnp.float64),
        max_lower,
        max_upper,
        cap,
    )


def _cache_with_optional_factors(
    *,
    cache_key: tuple[object, ...],
    cached: _SparseILUCache,
    build_dense_factors: bool,
    build_jax_factors: bool,
    row_nnz_cap: int | None,
    emit: Callable[[int, str], None] | None,
) -> _SparseILUCache:
    ilu = cached.ilu
    if ilu is None:
        return cached

    need_dense = bool(build_dense_factors) and (cached.l_dense is None or cached.u_dense is None)
    need_jax = bool(build_jax_factors) and (
        cached.perm_r is None
        or cached.inv_perm_c is None
        or cached.lower_idx is None
        or cached.upper_idx is None
    )
    if not need_dense and not need_jax:
        return cached

    l_dense = cached.l_dense
    u_dense = cached.u_dense
    l_unit_diag = cached.l_unit_diag
    perm_r_jnp = cached.perm_r
    inv_perm_c_jnp = cached.inv_perm_c
    lower_idx_jnp = cached.lower_idx
    lower_val_jnp = cached.lower_val
    lower_diag_jnp = cached.lower_diag
    upper_idx_jnp = cached.upper_idx
    upper_val_jnp = cached.upper_val
    upper_diag_jnp = cached.upper_diag

    if need_dense:
        l_dense = np.asarray(ilu.L.todense(), dtype=np.float64)
        u_dense = np.asarray(ilu.U.todense(), dtype=np.float64)
        l_unit_diag = bool(np.allclose(np.diag(l_dense), 1.0))

    if need_jax:
        (
            perm_r_jnp,
            inv_perm_c_jnp,
            lower_idx_jnp,
            lower_val_jnp,
            lower_diag_jnp,
            upper_idx_jnp,
            upper_val_jnp,
            upper_diag_jnp,
            max_lower,
            max_upper,
            cap,
        ) = _jax_factor_arrays_from_superlu(ilu=ilu, n=int(cached.a_csr_full.shape[0]), row_nnz_cap=row_nnz_cap)
        if emit is not None:
            emit(
                1,
                "sparse_ilu: cached JAX factors "
                f"(max_lower={int(max_lower)} max_upper={int(max_upper)} cap={int(cap)})",
            )

    updated = _SparseILUCache(
        a_csr_full=cached.a_csr_full,
        a_csr_drop=cached.a_csr_drop,
        ilu=ilu,
        a_dense=cached.a_dense,
        l_dense=l_dense,
        u_dense=u_dense,
        l_unit_diag=bool(l_unit_diag),
        perm_r=perm_r_jnp,
        inv_perm_c=inv_perm_c_jnp,
        lower_idx=lower_idx_jnp,
        lower_val=lower_val_jnp,
        lower_diag=lower_diag_jnp,
        upper_idx=upper_idx_jnp,
        upper_val=upper_val_jnp,
        upper_diag=upper_diag_jnp,
    )
    _RHSMODE1_SPARSE_ILU_CACHE[cache_key] = updated
    return updated


def _assemble_sparse_operator_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    factor_dtype: np.dtype,
    store_dense: bool,
    log_tag: str,
    emit: Callable[[int, str], None] | None,
) -> tuple[object, np.ndarray | None, jnp.ndarray | None, float]:
    import scipy.sparse as sp  # noqa: PLC0415

    sparse_block_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK", "").strip()
    sparse_block_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPARSE_ASSEMBLE_BLOCK_MIN", "").strip()
    try:
        sparse_block = int(sparse_block_env) if sparse_block_env else 0
    except ValueError:
        sparse_block = 0
    try:
        sparse_block_min = int(sparse_block_min_env) if sparse_block_min_env else 8000
    except ValueError:
        sparse_block_min = 8000
    if sparse_block == 0 and int(n) >= max(1, int(sparse_block_min)) and (not store_dense):
        sparse_block = 128

    struct_tol = sparse_structural_tol()
    assembly_timer = Timer()
    if sparse_block > 0:
        if emit is not None:
            emit(
                1,
                f"{log_tag}: operator assembly start mode=column_blocks "
                f"n={int(n)} block={int(sparse_block)} factor_dtype={factor_dtype.name}",
            )
        jit_env = os.environ.get("SFINCS_JAX_DENSE_ASSEMBLE_JIT", "").strip().lower()
        use_jit = (jit_env not in {"0", "false", "no", "off"}) if jit_env else int(n) > 800

        def _assemble(block_cols: jnp.ndarray) -> jnp.ndarray:
            return jax.vmap(matvec, in_axes=1, out_axes=1)(block_cols)

        assemble_fn = jax.jit(_assemble) if use_jit else _assemble
        rows_parts: list[np.ndarray] = []
        cols_parts: list[np.ndarray] = []
        data_parts: list[np.ndarray] = []
        max_abs = 0.0
        for start in range(0, int(n), int(sparse_block)):
            width = min(int(sparse_block), int(n) - int(start))
            block_cols_np = np.zeros((int(n), int(width)), dtype=np.float64)
            block_cols_np[np.arange(int(start), int(start) + int(width)), np.arange(int(width))] = 1.0
            chunk = np.array(assemble_fn(jnp.asarray(block_cols_np, dtype=dtype)), dtype=np.float64, copy=True)
            if struct_tol > 0.0 and chunk.size:
                chunk[np.abs(chunk) <= struct_tol] = 0.0
            if chunk.size:
                max_abs = max(max_abs, float(np.max(np.abs(chunk))))
                row_idx, col_local = np.nonzero(chunk)
                if row_idx.size:
                    rows_parts.append(row_idx.astype(np.int32, copy=False))
                    cols_parts.append((col_local + int(start)).astype(np.int32, copy=False))
                    data_parts.append(chunk[row_idx, col_local].astype(factor_dtype, copy=False))
        if rows_parts:
            rows = np.concatenate(rows_parts, axis=0)
            cols = np.concatenate(cols_parts, axis=0)
            data = np.concatenate(data_parts, axis=0)
        else:
            rows = np.zeros((0,), dtype=np.int32)
            cols = np.zeros((0,), dtype=np.int32)
            data = np.zeros((0,), dtype=factor_dtype)
        a_csr_full = sp.csr_matrix((data, (rows, cols)), shape=(int(n), int(n)))
        a_csr_full.eliminate_zeros()
        a_dense_np = None
        a_dense_jnp = None
    else:
        if emit is not None:
            emit(
                1,
                f"{log_tag}: operator assembly start mode=dense n={int(n)} "
                f"factor_dtype={factor_dtype.name}",
            )
        a_dense_jnp = assemble_dense_matrix_from_matvec(matvec=matvec, n=int(n), dtype=dtype)
        a_dense_np = np.array(a_dense_jnp, dtype=factor_dtype, copy=True)
        if struct_tol > 0.0 and a_dense_np.size:
            a_dense_np[np.abs(a_dense_np) <= struct_tol] = 0.0
        max_abs = float(np.max(np.abs(a_dense_np))) if a_dense_np.size else 0.0
        a_csr_full = sp.csr_matrix(a_dense_np)
        a_csr_full.eliminate_zeros()

    if emit is not None:
        nnz_full = int(a_csr_full.nnz)
        emit(
            1,
            f"{log_tag}: operator assembly complete elapsed_s={assembly_timer.elapsed_s():.3f} "
            f"nnz={nnz_full} density={nnz_full / max(1, int(n) * int(n)):.3e}",
        )
    return a_csr_full, a_dense_np, a_dense_jnp, max_abs


def _drop_and_regularize_csr(
    *,
    a_csr_full,
    a_np_full: np.ndarray | None,
    factor_dtype: np.dtype,
    thresh: float,
    reg: float,
):
    if a_np_full is None:
        a_csr_drop = a_csr_full.copy()
        if thresh > 0.0:
            data = a_csr_drop.data
            data[np.abs(data) < thresh] = 0.0
        if int(a_csr_drop.shape[0]) > 0 and reg != 0.0:
            diag_idx = np.arange(int(a_csr_drop.shape[0]), dtype=np.int32)
            a_csr_drop = a_csr_drop.tolil(copy=False)
            diag_vals = (
                np.asarray(a_csr_drop[diag_idx, diag_idx].toarray(), dtype=factor_dtype).reshape((-1,))
                + factor_dtype.type(reg)
            )
            a_csr_drop[diag_idx, diag_idx] = diag_vals
            a_csr_drop = a_csr_drop.tocsr()
        a_csr_drop.eliminate_zeros()
        return a_csr_drop

    if thresh > 0.0:
        a_np_drop = a_np_full.copy()
        a_np_drop[np.abs(a_np_drop) < thresh] = 0.0
    elif reg != 0.0:
        a_np_drop = a_np_full.copy()
    else:
        a_np_drop = a_np_full

    if int(a_np_drop.shape[0]) > 0 and reg != 0.0:
        diag_idx = np.arange(int(a_np_drop.shape[0]), dtype=np.int32)
        a_np_drop[diag_idx, diag_idx] = a_np_full[diag_idx, diag_idx] + factor_dtype.type(reg)

    import scipy.sparse as sp  # noqa: PLC0415

    a_csr_drop = sp.csr_matrix(a_np_drop)
    a_csr_drop.eliminate_zeros()
    return a_csr_drop


def _factorize_with_retries(
    *,
    a_csr_full,
    a_np_full: np.ndarray | None,
    factor_dtype: np.dtype,
    max_abs: float,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    factorization: str,
    log_tag: str,
    emit: Callable[[int, str], None] | None,
) -> tuple[object, object | None]:
    from scipy.sparse.linalg import spilu, splu  # noqa: PLC0415

    exact_lu = str(factorization).strip().lower() == "lu"
    thresh0 = 0.0 if exact_lu else max(float(drop_tol), float(drop_rel) * float(max_abs))
    reg, singular_reg_rel, attempts = _regularization_settings(max_abs)
    a_csr_drop = None
    ilu = None
    ilu_drop_tol_eff = float(ilu_drop_tol)
    fill_factor_eff = float(fill_factor)
    last_exc: Exception | None = None

    for attempt in range(int(attempts)):
        thresh = float(thresh0) * (0.1 ** int(attempt))
        if emit is not None and thresh > 0.0:
            emit(1, f"{log_tag}: dropping entries |a| < {thresh:.3e}")
        a_csr_drop = _drop_and_regularize_csr(
            a_csr_full=a_csr_full,
            a_np_full=a_np_full,
            factor_dtype=factor_dtype,
            thresh=thresh,
            reg=reg,
        )
        if emit is not None:
            nnz = int(a_csr_drop.nnz)
            n = int(a_csr_drop.shape[0])
            emit(1, f"{log_tag}: nnz={nnz} ({nnz / max(1, n * n):.3e} density)")
        attempt_timer = Timer()
        if emit is not None:
            emit(
                1,
                f"{log_tag}: factorization start "
                f"attempt={attempt + 1}/{int(attempts)} shape={a_csr_drop.shape} "
                f"nnz={int(a_csr_drop.nnz)} drop_tol={ilu_drop_tol_eff:.1e} "
                f"fill={fill_factor_eff:.1f}",
            )
        try:
            if exact_lu:
                ilu = splu(a_csr_drop.tocsc(), permc_spec="COLAMD")
            else:
                ilu = spilu(
                    a_csr_drop.tocsc(),
                    drop_tol=float(ilu_drop_tol_eff),
                    fill_factor=float(fill_factor_eff),
                    permc_spec="COLAMD",
                )
            if emit is not None:
                factor_nnz = int(getattr(ilu.L, "nnz", 0) + getattr(ilu.U, "nnz", 0))
                emit(
                    1,
                    f"{log_tag}: factorization complete "
                    f"attempt={attempt + 1}/{int(attempts)} elapsed_s={attempt_timer.elapsed_s():.3f} "
                    f"factor_nnz={int(factor_nnz)}",
                )
            last_exc = None
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            msg = str(exc).lower()
            if emit is not None:
                emit(
                    1,
                    f"{log_tag}: factorization failed "
                    f"(attempt={attempt + 1}/{int(attempts)} "
                    f"elapsed_s={attempt_timer.elapsed_s():.3f} "
                    f"thresh={thresh:.3e} drop_tol={ilu_drop_tol_eff:.1e} fill={fill_factor_eff:.1f}) "
                    f"({type(exc).__name__}: {exc})",
                )
            singular = ("singular" in msg) or ("pivot" in msg) or ("dpivot" in msg) or ("zero" in msg)
            if exact_lu:
                if singular:
                    reg = max(float(reg), max(1.0e-12, 1.0e-10 * float(max_abs)))
                    continue
                raise
            if singular:
                if singular_reg_rel > 0.0:
                    reg_candidate = max(1.0e-12, float(singular_reg_rel) * (10.0 ** int(attempt)) * float(max_abs))
                    if reg_candidate > float(reg):
                        reg = float(reg_candidate)
                        if emit is not None:
                            emit(
                                1,
                                f"{log_tag}: increasing diagonal regularization to {float(reg):.3e} "
                                "after singular local factorization",
                            )
                ilu_drop_tol_eff = max(0.0, float(ilu_drop_tol_eff) * 0.1)
                fill_factor_eff = max(float(fill_factor_eff), float(fill_factor) * 2.0, 20.0)
                continue
            raise

    if ilu is None and last_exc is not None:
        raise RuntimeError(
            f"{log_tag}: factorization failed after {int(attempts)} attempts ({type(last_exc).__name__}: {last_exc})"
        )
    if a_csr_drop is None:
        a_csr_drop = a_csr_full.copy()
        a_csr_drop.eliminate_zeros()
    return a_csr_drop, ilu


def build_sparse_ilu_from_matvec(
    *,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    n: int,
    dtype: jnp.dtype,
    cache_key: tuple[object, ...],
    factor_dtype: np.dtype | None = None,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    build_dense_factors: bool,
    build_jax_factors: bool,
    build_ilu: bool,
    store_dense: bool,
    factorization: str = "ilu",
    row_nnz_cap: int | None = None,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[object, object, object | None, np.ndarray | jnp.ndarray | None, np.ndarray | None, np.ndarray | None, bool]:
    """Assemble a sparse operator from a matvec and optionally factor it on host."""

    exact_lu = str(factorization).strip().lower() == "lu"
    log_tag = "sparse_lu" if exact_lu else "sparse_ilu"

    cached = _RHSMODE1_SPARSE_ILU_CACHE.get(cache_key)
    if cached is not None:
        cached = _cache_with_optional_factors(
            cache_key=cache_key,
            cached=cached,
            build_dense_factors=build_dense_factors,
            build_jax_factors=build_jax_factors,
            row_nnz_cap=row_nnz_cap,
            emit=emit,
        )
        if emit is not None:
            emit(1, f"{log_tag}: factorization cache hit n={int(n)}")
        return (
            cached.a_csr_full,
            cached.a_csr_drop,
            cached.ilu,
            cached.a_dense,
            cached.l_dense,
            cached.u_dense,
            cached.l_unit_diag,
        )

    factor_dtype_use = np.dtype(np.float64 if factor_dtype is None else factor_dtype)
    setup_timer = Timer()
    a_csr_full, a_np_full, a_dense_jnp, max_abs = _assemble_sparse_operator_from_matvec(
        matvec=matvec,
        n=int(n),
        dtype=dtype,
        factor_dtype=factor_dtype_use,
        store_dense=store_dense,
        log_tag=log_tag,
        emit=emit,
    )

    if build_ilu:
        a_csr_drop, ilu = _factorize_with_retries(
            a_csr_full=a_csr_full,
            a_np_full=a_np_full,
            factor_dtype=factor_dtype_use,
            max_abs=max_abs,
            drop_tol=drop_tol,
            drop_rel=drop_rel,
            ilu_drop_tol=ilu_drop_tol,
            fill_factor=fill_factor,
            factorization=factorization,
            log_tag=log_tag,
            emit=emit,
        )
    else:
        reg, _singular_reg_rel, _attempts = _regularization_settings(max_abs)
        a_csr_drop = _drop_and_regularize_csr(
            a_csr_full=a_csr_full,
            a_np_full=a_np_full,
            factor_dtype=factor_dtype_use,
            thresh=0.0,
            reg=reg,
        )
        ilu = None
        if emit is not None:
            nnz = int(a_csr_drop.nnz)
            emit(1, f"{log_tag}: nnz={nnz} ({nnz / max(1, int(n) * int(n)):.3e} density)")

    if store_dense:
        if a_np_full is not None:
            a_dense = a_np_full
        elif int(n) <= 4000:
            a_dense = np.asarray(a_csr_full.toarray(), dtype=np.float64)
        else:
            a_dense = None
    else:
        a_dense = a_dense_jnp if a_np_full is not None and store_dense else None

    l_dense = None
    u_dense = None
    l_unit_diag = True
    factor_arrays = (None, None, None, None, None, None, None, None)
    if build_dense_factors and ilu is not None:
        l_dense = np.asarray(ilu.L.todense(), dtype=np.float64)
        u_dense = np.asarray(ilu.U.todense(), dtype=np.float64)
        l_unit_diag = bool(np.allclose(np.diag(l_dense), 1.0))
    if build_jax_factors and ilu is not None:
        *factor_arrays, max_lower, max_upper, cap = _jax_factor_arrays_from_superlu(
            ilu=ilu,
            n=int(n),
            row_nnz_cap=row_nnz_cap,
        )
        if emit is not None:
            emit(
                1,
                "sparse_ilu: built JAX factors "
                f"(max_lower={int(max_lower)} max_upper={int(max_upper)} cap={int(cap)})",
            )

    (
        perm_r_jnp,
        inv_perm_c_jnp,
        lower_idx_jnp,
        lower_val_jnp,
        lower_diag_jnp,
        upper_idx_jnp,
        upper_val_jnp,
        upper_diag_jnp,
    ) = factor_arrays
    _RHSMODE1_SPARSE_ILU_CACHE[cache_key] = _SparseILUCache(
        a_csr_full=a_csr_full,
        a_csr_drop=a_csr_drop,
        ilu=ilu,
        a_dense=a_dense,
        l_dense=l_dense,
        u_dense=u_dense,
        l_unit_diag=l_unit_diag,
        perm_r=perm_r_jnp,
        inv_perm_c=inv_perm_c_jnp,
        lower_idx=lower_idx_jnp,
        lower_val=lower_val_jnp,
        lower_diag=lower_diag_jnp,
        upper_idx=upper_idx_jnp,
        upper_val=upper_val_jnp,
        upper_diag=upper_diag_jnp,
    )
    if emit is not None:
        factor_nnz = None if ilu is None else int(getattr(ilu.L, "nnz", 0) + getattr(ilu.U, "nnz", 0))
        emit(
            1,
            f"{log_tag}: setup complete elapsed_s={setup_timer.elapsed_s():.3f} "
            f"operator_nnz={int(a_csr_full.nnz)} factor_nnz={factor_nnz}",
        )
    return a_csr_full, a_csr_drop, ilu, a_dense, l_dense, u_dense, l_unit_diag


def factorize_sparse_matrix_csr_host(
    *,
    a_csr_full,
    cache_key: tuple[object, ...],
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    factorization: str = "ilu",
    emit: Callable[[int, str], None] | None = None,
) -> tuple[object, object, object]:
    """Factor an already assembled CSR matrix with the same retry policy."""

    exact_lu = str(factorization).strip().lower() == "lu"
    log_tag = "sparse_lu" if exact_lu else "sparse_ilu"
    cached = _RHSMODE1_SPARSE_ILU_CACHE.get(cache_key)
    if cached is not None and cached.ilu is not None:
        if emit is not None:
            emit(1, f"{log_tag}: factorization cache hit n={int(cached.a_csr_full.shape[0])}")
        return cached.a_csr_full, cached.a_csr_drop, cached.ilu

    setup_timer = Timer()
    a_csr_full = a_csr_full.tocsr()
    a_csr_full.eliminate_zeros()
    factor_dtype = np.dtype(np.float64)
    max_abs = float(np.max(np.abs(a_csr_full.data))) if int(a_csr_full.nnz) > 0 else 0.0
    a_csr_drop, ilu = _factorize_with_retries(
        a_csr_full=a_csr_full,
        a_np_full=None,
        factor_dtype=factor_dtype,
        max_abs=max_abs,
        drop_tol=drop_tol,
        drop_rel=drop_rel,
        ilu_drop_tol=ilu_drop_tol,
        fill_factor=fill_factor,
        factorization=factorization,
        log_tag=log_tag,
        emit=emit,
    )
    if ilu is None:
        raise RuntimeError(f"{log_tag}: factorization failed with no exception detail")

    _RHSMODE1_SPARSE_ILU_CACHE[cache_key] = _SparseILUCache(
        a_csr_full=a_csr_full,
        a_csr_drop=a_csr_drop,
        ilu=ilu,
        a_dense=None,
        l_dense=None,
        u_dense=None,
        l_unit_diag=True,
    )
    cached = _RHSMODE1_SPARSE_ILU_CACHE[cache_key]
    assert cached.ilu is not None
    if emit is not None:
        factor_nnz = int(getattr(cached.ilu.L, "nnz", 0) + getattr(cached.ilu.U, "nnz", 0))
        emit(
            1,
            f"{log_tag}: setup complete elapsed_s={setup_timer.elapsed_s():.3f} "
            f"operator_nnz={int(cached.a_csr_full.nnz)} factor_nnz={int(factor_nnz)}",
        )
    return cached.a_csr_full, cached.a_csr_drop, cached.ilu


__all__ = (
    "RHS1FullSystemMatrixFreeOperatorAdapter",
    "build_sparse_ilu_from_matvec",
    "factorize_sparse_matrix_csr_host",
)
