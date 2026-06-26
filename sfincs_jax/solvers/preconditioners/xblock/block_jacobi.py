"""Dense x-block Jacobi preconditioners for RHSMode=1 solves."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_caches import (
    _RHSMODE1_PRECOND_IDX_CACHE,
    _RHSMODE1_PRECOND_LIST_CACHE,
    _RHSMode1PrecondIdxCache,
    _RHSMode1PrecondListCache,
)
from sfincs_jax.solvers.preconditioner_context import precond_dtype
from sfincs_jax.solvers.preconditioner_setup import (
    matvec_submatrix_v3_unsharded,
    precond_chunk_cols,
    rhs_mode1_precond_cache_key,
)
from ....v3_system import V3FullSystemOperator

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = (
    "build_rhs1_sxblock_tz_preconditioner",
    "build_rhs1_xblock_tz_lmax_preconditioner",
    "build_rhs1_xblock_tz_preconditioner",
)


def _cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=precond_dtype())


def _regularization_from_env() -> np.float64:
    reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
    reg_val = float(reg_env) if reg_env else 1e-10
    return np.float64(reg_val)


def _pas_chunk_cap(*, op: V3FullSystemOperator, n_theta: int, n_zeta: int, lmax: int | None) -> int | None:
    if op.fblock.pas is None:
        return None
    pas_cols_env = os.environ.get("SFINCS_JAX_PRECOND_PAS_MAX_COLS", "").strip()
    try:
        if pas_cols_env:
            return int(pas_cols_env)
        if lmax is not None:
            return 64
        tz_size = int(n_theta) * int(n_zeta)
        return 256 if tz_size <= 256 else 64
    except ValueError:
        if lmax is not None:
            return 64
        tz_size = int(n_theta) * int(n_zeta)
        return 256 if tz_size <= 256 else 64


def _extra_inverse(
    *,
    op: V3FullSystemOperator,
    total_size: int,
    dtype: jnp.dtype,
    reg: np.float64,
) -> tuple[jnp.ndarray, jnp.ndarray | None]:
    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
    extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
    extra_inv_jnp: jnp.ndarray | None = None
    if extra_size > 0:
        chunk_cols = precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
        y_sub = matvec_submatrix_v3_unsharded(
            op,
            col_idx=extra_idx_np,
            row_idx=extra_idx_np,
            total_size=total_size,
            chunk_cols=chunk_cols,
        )
        ee = np.asarray(y_sub.T, dtype=np.float64)
        ee = ee + reg * np.eye(extra_size, dtype=np.float64)
        try:
            ee_inv = np.linalg.inv(ee)
        except np.linalg.LinAlgError:
            ee_inv = np.linalg.pinv(ee, rcond=1e-12)
        if not np.all(np.isfinite(ee_inv)):
            ee_inv = np.linalg.pinv(ee, rcond=1e-12)
        extra_inv_jnp = jnp.asarray(ee_inv, dtype=dtype)
    return extra_idx_jnp, extra_inv_jnp


def _sxblock_indices_for_l(
    *,
    n_species: int,
    n_x: int,
    n_l: int,
    n_theta: int,
    n_zeta: int,
    nxi_for_x: np.ndarray,
    ell: int,
) -> np.ndarray:
    active_x = np.where(nxi_for_x > int(ell))[0]
    if active_x.size == 0:
        return np.zeros((0,), dtype=np.int32)
    idx: list[int] = []
    for s in range(int(n_species)):
        for ix in active_x:
            base = int((((s * n_x + int(ix)) * n_l + int(ell)) * n_theta) * n_zeta)
            for it in range(int(n_theta)):
                for iz in range(int(n_zeta)):
                    idx.append(base + it * int(n_zeta) + iz)
    return np.asarray(idx, dtype=np.int32)


def build_rhs1_sxblock_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build dense species/``x`` blocks over ``(theta,zeta)`` for each pitch mode."""

    cache_key = _cache_key(op, "sxblock_tz")
    cached = _RHSMODE1_PRECOND_IDX_CACHE.get(cache_key)
    if cached is None:
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_block_size = 0
        for ell in range(n_l):
            active_x = np.where(nxi_for_x > ell)[0]
            if active_x.size:
                max_block_size = max(max_block_size, int(active_x.size) * n_species * n_theta * n_zeta)
        dtype = precond_dtype(max_block_size * max_block_size)
        reg = _regularization_from_env()

        block_inv_list: list[jnp.ndarray] = []
        block_idx_list: list[jnp.ndarray] = []
        for ell in range(n_l):
            rep_idx = _sxblock_indices_for_l(
                n_species=n_species,
                n_x=n_x,
                n_l=n_l,
                n_theta=n_theta,
                n_zeta=n_zeta,
                nxi_for_x=nxi_for_x,
                ell=ell,
            )
            if rep_idx.size == 0:
                continue
            chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
            y_sub = matvec_submatrix_v3_unsharded(
                op,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(int(rep_idx.shape[0]), dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv_list.append(jnp.asarray(inv, dtype=dtype))
            block_idx_list.append(jnp.asarray(rep_idx, dtype=jnp.int32))

        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )
        cached = _RHSMode1PrecondIdxCache(
            block_inv_list=tuple(block_inv_list),
            block_idx_list=tuple(block_idx_list),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_IDX_CACHE[cache_key] = cached

    block_inv_list = cached.block_inv_list
    block_idx_list = cached.block_idx_list
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    dtype = block_inv_list[0].dtype if block_inv_list else precond_dtype()

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        z_full = jnp.zeros_like(r_full)
        for inv, idx in zip(block_inv_list, block_idx_list, strict=True):
            r_loc = r_full[idx]
            z_loc = inv @ r_loc
            z_full = z_full.at[idx].set(z_loc, unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def _build_xblock_cache(
    *,
    op: V3FullSystemOperator,
    kind: str,
    lmax: int | None,
) -> tuple[_RHSMode1PrecondListCache, jnp.dtype]:
    cache_key = _cache_key(op, kind)
    cached = _RHSMODE1_PRECOND_LIST_CACHE.get(cache_key)

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    total_size = int(op.total_size)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
    if lmax is not None:
        max_l = min(max_l, int(lmax))
    max_block_size = int(max_l * n_theta * n_zeta)
    dtype = precond_dtype(max_block_size * max_block_size)

    if cached is None:
        reg = _regularization_from_env()
        pas_max_cols = _pas_chunk_cap(op=op, n_theta=n_theta, n_zeta=n_zeta, lmax=lmax)
        block_inv_list: list[jnp.ndarray] = []
        block_slices: list[tuple[int, int]] = []

        for s in range(n_species):
            for ix in range(n_x):
                max_lx = int(nxi_for_x[ix])
                if lmax is not None:
                    max_lx = min(max_lx, int(lmax))
                block_size = int(max_lx * n_theta * n_zeta)
                start = int((((s * n_x + ix) * n_l) * n_theta) * n_zeta)
                if block_size <= 0:
                    continue
                rep_idx = np.arange(start, start + block_size, dtype=np.int32)
                chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
                if pas_max_cols is not None and block_size >= 256:
                    chunk_cols = min(chunk_cols, pas_max_cols)
                y_sub = matvec_submatrix_v3_unsharded(
                    op,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)
                a = a + reg * np.eye(block_size, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                block_inv_list.append(jnp.asarray(inv, dtype=dtype))
                block_slices.append((start, block_size))

        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )
        cached = _RHSMode1PrecondListCache(
            block_inv_list=tuple(block_inv_list),
            block_slices=tuple(block_slices),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_LIST_CACHE[cache_key] = cached

    return cached, dtype


def _apply_xblock_cache(
    *,
    cached: _RHSMode1PrecondListCache,
    dtype: jnp.dtype,
    identity_for_uncovered: bool,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None,
) -> Preconditioner:
    block_inv_list = cached.block_inv_list
    block_slices = cached.block_slices
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    if block_inv_list:
        dtype = block_inv_list[0].dtype

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        z_full = jnp.asarray(r_full, dtype=dtype) if identity_for_uncovered else jnp.zeros_like(r_full)
        for inv, (start, block_size) in zip(block_inv_list, block_slices, strict=True):
            r_loc = r_full[start : start + block_size]
            z_loc = inv @ r_loc
            z_full = z_full.at[start : start + block_size].set(z_loc, unique_indices=True)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_xblock_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build dense per-``(species,x)`` blocks over ``(L,theta,zeta)``."""

    cached, dtype = _build_xblock_cache(op=op, kind="xblock_tz", lmax=None)
    return _apply_xblock_cache(
        cached=cached,
        dtype=dtype,
        identity_for_uncovered=False,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )


def build_rhs1_xblock_tz_lmax_preconditioner(
    *,
    op: V3FullSystemOperator,
    lmax: int,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build dense per-``(species,x)`` blocks truncated to the lowest ``L`` modes."""

    lmax = int(lmax)
    if lmax <= 0:
        return build_rhs1_xblock_tz_preconditioner(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    cached, dtype = _build_xblock_cache(op=op, kind=f"xblock_tz_lmax_{lmax}", lmax=lmax)
    return _apply_xblock_cache(
        cached=cached,
        dtype=dtype,
        identity_for_uncovered=True,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
