"""Species-block preconditioners for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_caches import (
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_PRECOND_GLOBAL_CACHE,
    _RHSMode1PrecondCache,
    _RHSMode1PrecondGlobalCache,
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
    "build_rhs1_species_block_preconditioner",
    "build_rhs1_species_xblock_preconditioner",
)


def _cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=precond_dtype())


def _regularization_from_env() -> np.float64:
    reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
    reg_val = float(reg_env) if reg_env else 1e-10
    return np.float64(reg_val)


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


def build_rhs1_species_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build a preconditioner using one full block per species."""

    cache_key = _cache_key(op, "species_block")
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))
    block_size_hint = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
    dtype = precond_dtype(block_size_hint * block_size_hint)
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)
        block_size = int(local_per_species * n_theta * n_zeta)
        reg = _regularization_from_env()

        idx_map = np.zeros((n_species, block_size), dtype=np.int32)
        block_inv = np.zeros((n_species, block_size, block_size), dtype=np.float64)

        for s in range(n_species):
            k = 0
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    for it in range(n_theta):
                        for iz in range(n_zeta):
                            idx_map[s, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1
            rep_idx = idx_map[s, :]
            chunk_cols = precond_chunk_cols(total_size, int(rep_idx.shape[0]))
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
            block_inv[s, :, :] = inv

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=dtype)
        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_species = int(block_inv_jnp.shape[0])
    block_size = int(block_inv_jnp.shape[-1])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_species, block_size))
        z_loc = jnp.einsum("sij,sj->si", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
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


def build_rhs1_species_xblock_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Preconditioner:
    """Build one species/x/L block shared over each angular grid point."""

    cache_key = _cache_key(op, "sxblock")
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    local_per_species = int(np.sum(nxi_for_x))
    block_size = int(int(op.n_species) * local_per_species)
    dtype = precond_dtype(block_size * block_size)
    cached = _RHSMODE1_PRECOND_GLOBAL_CACHE.get(cache_key)
    if cached is None:
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)
        reg = _regularization_from_env()

        rep_idx: list[int] = []
        for s in range(n_species):
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_theta + 0) * n_zeta + 0)
                    rep_idx.append(int(f_idx))
        rep_idx_np = np.asarray(rep_idx, dtype=np.int32)
        chunk_cols = precond_chunk_cols(total_size, int(rep_idx_np.shape[0]))
        y_sub = matvec_submatrix_v3_unsharded(
            op,
            col_idx=rep_idx_np,
            row_idx=rep_idx_np,
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

        idx_map = np.zeros((n_theta, n_zeta, block_size), dtype=np.int32)
        for it in range(n_theta):
            for iz in range(n_zeta):
                k = 0
                for s in range(n_species):
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[it, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_theta + it) * n_zeta + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(inv, dtype=dtype)
        extra_idx_jnp, extra_inv_jnp = _extra_inverse(
            op=op,
            total_size=total_size,
            dtype=dtype,
            reg=reg,
        )

        cached = _RHSMode1PrecondGlobalCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_GLOBAL_CACHE[cache_key] = cached

    block_inv_jnp = cached.block_inv_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=dtype)
        r_loc = r_full[flat_idx_jnp].reshape((int(op.n_theta), int(op.n_zeta), block_size))
        z_loc = jnp.einsum("ij,tzj->tzi", block_inv_jnp, r_loc)
        z_full = jnp.zeros_like(r_full)
        z_full = z_full.at[flat_idx_jnp].set(z_loc.reshape((-1,)), unique_indices=True)
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
