"""RHSMode=2/3 transport-matrix preconditioner builders.

This module owns the reusable numerical kernels for the common transport
preconditioners. Transport problem owners import these kernels directly, which
keeps preconditioner behavior testable without compatibility driver wrappers.
"""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioning import (
    _RHSMODE23_PRECOND_CACHE,
    _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_PRECOND_CACHE,
    _TRANSPORT_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_PRECOND_CACHE,
    _TRANSPORT_TZFFT_PRECOND_CACHE,
    _TRANSPORT_XMG_PRECOND_CACHE,
    _LowRankXBlockPrecondCache,
    _RHSMode1PrecondCache,
    _TransportFpLocalGeomLinePrecondCache,
    _TransportFpTzFftLinePrecondCache,
    _TransportFpTzFftLineSchurPrecondCache,
    _TransportFpTzFftPrecondCache,
    _TransportPrecondCache,
    _TransportTzFftPrecondCache,
    _TransportXBlockPrecondCache,
    _TransportXmgPrecondCache,
)
from sfincs_jax.solvers.preconditioning import precond_dtype as _precond_dtype
from sfincs_jax.solvers.preconditioning import _build_transport_preconditioner_operator_point
from sfincs_jax.solvers.preconditioning import (
    hash_array as _hash_array,
    matvec_submatrix as _matvec_submatrix_impl,
    precond_chunk_cols as _precond_chunk_cols,
    transport_precond_cache_key as _transport_precond_cache_key_impl,
)
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator,
    _fs_average_factor,
    _ix_min,
    _source_basis_constraint_scheme_1,
    apply_v3_full_system_operator,
    apply_v3_full_system_operator_cached,
)

__all__ = (
    "build_rhsmode23_block_preconditioner",
    "build_rhsmode23_collision_preconditioner",
    "build_rhsmode23_fp_local_geom_line_preconditioner",
    "build_rhsmode23_fp_tzfft_line_preconditioner",
    "build_rhsmode23_fp_tzfft_line_schur_preconditioner",
    "build_rhsmode23_fp_tzfft_preconditioner",
    "build_rhsmode23_sxblock_preconditioner",
    "build_rhsmode23_tzfft_preconditioner",
    "build_rhsmode23_xmg_preconditioner",
)


def _transport_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return _transport_precond_cache_key_impl(op, kind, precond_dtype=_precond_dtype())


def _matvec_submatrix(
    op_pc: V3FullSystemOperator,
    *,
    col_idx: np.ndarray,
    row_idx: np.ndarray,
    total_size: int,
    chunk_cols: int,
) -> np.ndarray:
    # Setup-time host assembly should use the unsharded operator application;
    # cached/pjit matvecs can enter a mesh context that is invalid under vmap.
    return _matvec_submatrix_impl(
        op_pc,
        col_idx=col_idx,
        row_idx=row_idx,
        total_size=total_size,
        chunk_cols=chunk_cols,
        apply_operator_fn=apply_v3_full_system_operator,
    )


def _nxi_for_x_or_all(op: V3FullSystemOperator) -> np.ndarray:
    """Return active pitch counts, defaulting to all modes for collision-only factors."""

    collisionless = getattr(op.fblock, "collisionless", None)
    nxi_for_x_raw = getattr(collisionless, "n_xi_for_x", None)
    if nxi_for_x_raw is None:
        return np.full((int(op.n_x),), int(op.n_xi), dtype=np.int32)
    return np.asarray(nxi_for_x_raw, dtype=np.int32)


def build_rhsmode23_collision_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Cheap diagonal preconditioner for RHSMode=2/3 transport solves.

    Uses analytic diagonal contributions from the collision operator (PAS or FP) plus
    the identity shift, and is diagonal in (theta, zeta).
    """
    cache_key = _transport_precond_cache_key(op, "collision_diag")
    precond_dtype = _precond_dtype()
    cached = _TRANSPORT_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        diag = jnp.zeros(f_shape, dtype=jnp.float64)

        # Identity shift contribution.
        if float(op.fblock.identity_shift) != 0.0:
            diag = diag + jnp.asarray(op.fblock.identity_shift, dtype=jnp.float64)

        # Pitch-angle scattering diagonal term.
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            ell = jnp.arange(n_l, dtype=jnp.float64)
            factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * pas.krook)
            pas_diag = pas.nu_n * pas.nu_d_hat[:, :, None] * factor_l[None, None, :]
            diag = diag + pas_diag[:, :, :, None, None]

        # Fokker-Planck diagonal term (self-species, diagonal in x).
        if op.fblock.fp is not None:
            mat = op.fblock.fp.mat  # (S,S,L,X,X)
            diag_x = jnp.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
            diag_self = jnp.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
            diag_self = jnp.transpose(diag_self, (2, 1, 0))  # (S,X,L)
            diag = diag + diag_self[:, :, :, None, None]

        # Mask out inactive L-modes.
        nxi_for_x = jnp.asarray(_nxi_for_x_or_all(op), dtype=jnp.int32)
        mask = jnp.arange(n_l, dtype=jnp.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        mask = mask[None, :, :, None, None]  # (1,X,L,1,1)
        diag = jnp.where(mask, diag, jnp.asarray(1.0, dtype=jnp.float64))

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        inv_diag_f = 1.0 / (diag + float(reg))
        cached = _TransportPrecondCache(inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype))
        _TRANSPORT_PRECOND_CACHE[cache_key] = cached

    inv_diag_f = cached.inv_diag_f

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)
        z_f = f * inv_diag_f
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_sxblock_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Lightweight block-Jacobi preconditioner for RHSMode=2/3 using species/x blocks.

    Builds per-L blocks across species and x from the collision operator (PAS/FP) plus
    identity shift. This avoids matvec-based assembly while capturing cross-species/x
    coupling in the FP operator.
    """
    if op.fblock.fp is None:
        return build_rhsmode23_collision_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    low_rank_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "").strip()
    if not low_rank_env:
        low_rank_env = os.environ.get("SFINCS_JAX_FP_LOW_RANK_K", "").strip()
    low_rank_env = low_rank_env.strip().lower()
    low_rank_auto = low_rank_env in {"", "auto"}
    if low_rank_env and low_rank_env != "auto":
        try:
            low_rank_k = int(low_rank_env)
        except ValueError:
            low_rank_k = 0
    else:
        low_rank_k = 0

    f_shape = op.fblock.f_shape
    n_species, n_x, n_l, _, _ = f_shape
    n_block = n_species * n_x
    if low_rank_auto and low_rank_k <= 0 and n_block >= 24:
        low_rank_k = min(8, n_block)

    precond_dtype = _precond_dtype()
    if low_rank_k > 0:
        rank_k = min(int(low_rank_k), int(n_block))
        cache_key = _transport_precond_cache_key(op, f"collision_sxblock_lr_{rank_k}")
        cached_lr = _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE.get(cache_key)
        if cached_lr is None:
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-10
            except ValueError:
                reg = 1e-10
            identity_shift = float(op.fblock.identity_shift)
            pas_diag = None
            if op.fblock.pas is not None:
                pas = op.fblock.pas
                l_arr = np.arange(n_l, dtype=np.float64)
                factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
                pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

            nxi_for_x = _nxi_for_x_or_all(op)
            d_inv = np.zeros((n_l, n_block), dtype=np.float64)
            d_inv_u = np.zeros((n_l, n_block, rank_k), dtype=np.float64)
            v_lr = np.zeros((n_l, rank_k, n_block), dtype=np.float64)
            m_inv = np.zeros((n_l, rank_k, rank_k), dtype=np.float64)

            for ell_idx in range(n_l):
                a_fp = np.array(mat[:, :, ell_idx, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                a_fp = a_fp.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                diag = np.full((n_block,), identity_shift + reg, dtype=np.float64)
                if pas_diag is not None:
                    diag += pas_diag[:, :, ell_idx].reshape((n_block,))
                inactive_x = np.where(nxi_for_x <= ell_idx)[0]
                if inactive_x.size:
                    for ix in inactive_x:
                        for s in range(n_species):
                            idx = s * n_x + int(ix)
                            a_fp[idx, :] = 0.0
                            a_fp[:, idx] = 0.0
                            diag[idx] = 1.0
                d_inv_l = 1.0 / diag
                d_inv[ell_idx, :] = d_inv_l
                if rank_k > 0:
                    try:
                        u, svals, vt = np.linalg.svd(a_fp, full_matrices=False)
                    except np.linalg.LinAlgError:
                        u, svals, vt = np.linalg.svd(a_fp + 1e-12 * np.eye(n_block), full_matrices=False)
                    k_use = min(rank_k, int(svals.shape[0]))
                    if k_use > 0:
                        u = u[:, :k_use]
                        svals = svals[:k_use]
                        vt = vt[:k_use, :]
                        s_sqrt = np.sqrt(np.maximum(svals, 0.0))
                        u_lr = u * s_sqrt[None, :]
                        v_lr_l = s_sqrt[:, None] * vt
                        d_inv_u_l = d_inv_l[:, None] * u_lr
                        m = np.eye(k_use, dtype=np.float64) + v_lr_l @ d_inv_u_l
                        try:
                            m_inv_l = np.linalg.inv(m)
                        except np.linalg.LinAlgError:
                            m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                        if not np.all(np.isfinite(m_inv_l)):
                            m_inv_l = np.linalg.pinv(m, rcond=1e-12)
                        d_inv_u[ell_idx, :, :k_use] = d_inv_u_l
                        v_lr[ell_idx, :k_use, :] = v_lr_l
                        m_inv[ell_idx, :k_use, :k_use] = m_inv_l

            cached_lr = _LowRankXBlockPrecondCache(
                d_inv=jnp.asarray(d_inv, dtype=precond_dtype),
                d_inv_u=jnp.asarray(d_inv_u, dtype=precond_dtype),
                v=jnp.asarray(v_lr, dtype=precond_dtype),
                m_inv=jnp.asarray(m_inv, dtype=precond_dtype),
            )
            _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE[cache_key] = cached_lr

        d_inv = cached_lr.d_inv
        d_inv_u = cached_lr.d_inv_u
        v_lr = cached_lr.v
        m_inv = cached_lr.m_inv

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=precond_dtype)
            f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
            f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
            f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
            d_r = d_inv[:, :, None, None] * f_l
            if rank_k > 0:
                tmp = jnp.einsum("lkn,lntz->lktz", v_lr, d_r)
                tmp2 = jnp.einsum("lkm,lmtz->lktz", m_inv, tmp)
                corr = jnp.einsum("lnk,lktz->lntz", d_inv_u, tmp2)
                z_l = d_r - corr
            else:
                z_l = d_r
            z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
            z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
            tail = r_full[op.f_size :]
            z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
            return jnp.asarray(z_full, dtype=jnp.float64)

        if reduce_full is None or expand_reduced is None:
            return _apply_full

        def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
            z_full = _apply_full(expand_reduced(r_reduced))
            return reduce_full(z_full)

        return _apply_reduced

    cache_key = _transport_precond_cache_key(op, "collision_sxblock")
    cached = _TRANSPORT_SXBLOCK_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        n_block = n_species * n_x
        inv_block = np.zeros((n_l, n_block, n_block), dtype=np.float64)
        mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10

        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        nxi_for_x = _nxi_for_x_or_all(op)
        for ell_idx in range(n_l):
            a = np.array(mat[:, :, ell_idx, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
            a = a.transpose(0, 2, 1, 3).reshape((n_block, n_block))
            if identity_shift != 0.0:
                a[np.arange(n_block), np.arange(n_block)] += identity_shift
            if pas_diag is not None:
                diag_add = pas_diag[:, :, ell_idx].reshape((n_block,))
                a[np.arange(n_block), np.arange(n_block)] += diag_add
            if reg != 0.0:
                a[np.arange(n_block), np.arange(n_block)] += reg

            inactive_x = np.where(nxi_for_x <= ell_idx)[0]
            if inactive_x.size:
                for ix in inactive_x:
                    for s in range(n_species):
                        idx = s * n_x + int(ix)
                        a[idx, :] = 0.0
                        a[:, idx] = 0.0
                        a[idx, idx] = 1.0

            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            inv_block[ell_idx, :, :] = inv

        cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_block, dtype=precond_dtype))
        _TRANSPORT_SXBLOCK_PRECOND_CACHE[cache_key] = cached

    inv_block = cached.inv_xblock  # (L, S*X, S*X)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_l = jnp.transpose(f, (2, 0, 1, 3, 4))  # (L,S,X,T,Z)
        f_l = f_l.reshape((int(op.n_xi), int(op.n_species) * int(op.n_x), int(op.n_theta), int(op.n_zeta)))
        z_l = jnp.einsum("lmn,lntz->lmtz", inv_block, f_l)
        z_l = z_l.reshape((int(op.n_xi), int(op.n_species), int(op.n_x), int(op.n_theta), int(op.n_zeta)))
        z_f = jnp.transpose(z_l, (1, 2, 0, 3, 4))  # (S,X,L,T,Z)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_xmg_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Two-level additive x-grid preconditioner for RHSMode=2/3 collision operators.

    Applies a fine-grid diagonal inverse plus a coarse-grid correction on the speed grid.
    The coarse solve is block-diagonal in species and L, ignoring cross-species coupling.
    """
    stride_env = os.environ.get("SFINCS_JAX_XMG_STRIDE", "").strip()
    try:
        stride = int(stride_env) if stride_env else 2
    except ValueError:
        stride = 2
    stride = max(1, stride)
    cache_key = _transport_precond_cache_key(op, f"xmg_{stride}")
    precond_dtype = _precond_dtype()
    cached = _TRANSPORT_XMG_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape
        coarse_idx = np.arange(0, n_x, stride, dtype=np.int32)
        n_coarse = int(coarse_idx.shape[0])
        coarse_map = {int(ix): int(i) for i, ix in enumerate(coarse_idx)}

        diag = np.zeros(f_shape, dtype=np.float64)
        if float(op.fblock.identity_shift) != 0.0:
            diag = diag + float(op.fblock.identity_shift)
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]
            diag = diag + pas_diag[:, :, :, None, None]
        if op.fblock.fp is not None:
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            diag_x = np.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
            diag_self = np.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
            diag_self = np.transpose(diag_self, (2, 1, 0))  # (S,X,L)
            diag = diag + diag_self[:, :, :, None, None]

        nxi_for_x = _nxi_for_x_or_all(op)
        mask = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        mask = mask[None, :, :, None, None]
        diag = np.where(mask, diag, 1.0)

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        inv_diag_f = 1.0 / (diag + float(reg))

        coarse_inv = np.zeros((n_species, n_l, n_coarse, n_coarse), dtype=np.float64)
        mat_fp = None
        if op.fblock.fp is not None:
            mat_fp = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        for s in range(n_species):
            for ell_idx in range(n_l):
                if mat_fp is None:
                    a = np.zeros((n_x, n_x), dtype=np.float64)
                else:
                    a = np.array(mat_fp[s, s, ell_idx, :, :], dtype=np.float64, copy=True)
                a = a[np.ix_(coarse_idx, coarse_idx)]
                diag_vec = np.full((n_coarse,), identity_shift + reg, dtype=np.float64)
                if pas_diag is not None:
                    diag_vec += pas_diag[s, coarse_idx, ell_idx]
                a[np.arange(n_coarse), np.arange(n_coarse)] += diag_vec

                inactive_x = np.where(nxi_for_x <= ell_idx)[0]
                if inactive_x.size:
                    for ix in inactive_x:
                        j = coarse_map.get(int(ix))
                        if j is not None:
                            a[j, :] = 0.0
                            a[:, j] = 0.0
                            a[j, j] = 1.0

                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                coarse_inv[s, ell_idx, :, :] = inv

        cached = _TransportXmgPrecondCache(
            inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype),
            coarse_inv=jnp.asarray(coarse_inv, dtype=precond_dtype),
            coarse_idx=jnp.asarray(coarse_idx, dtype=jnp.int32),
        )
        _TRANSPORT_XMG_PRECOND_CACHE[cache_key] = cached

    inv_diag_f = cached.inv_diag_f
    coarse_inv = cached.coarse_inv
    coarse_idx = cached.coarse_idx

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        z_f = f * inv_diag_f
        f_sl = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
        f_coarse = f_sl[:, :, coarse_idx, :, :]
        z_coarse = jnp.einsum("slij,sljtz->slitz", coarse_inv, f_coarse)
        corr_sl = jnp.zeros_like(f_sl)
        corr_sl = corr_sl.at[:, :, coarse_idx, :, :].set(z_coarse, unique_indices=True)
        corr = jnp.transpose(corr_sl, (0, 2, 1, 3, 4))
        z_f = z_f + corr
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_tzfft_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """FFT-based tridiagonal-in-L preconditioner for collisionless RHSMode=2/3 systems.

    This preconditioner targets the collisionless streaming+mirror operator, which couples
    Legendre modes (L) off-diagonally and is local in x and species. We approximate geometry
    coefficients by a flux-surface average, diagonalize theta/zeta derivative stencils in
    Fourier space (valid for periodic finite-difference matrices), and solve an (L,L) tridiagonal
    system independently for each (k_theta, k_zeta) mode.

    Notes
    -----
    - Designed for collisionless / monoenergetic transport solves, where x-multigrid does not help
      (often Nx=1) and Krylov can stagnate without an angular preconditioner.
    - Fully JAX-native and differentiable; safe to use under `custom_linear_solve`.
    """
    if op.fblock.collisionless is None:
        return build_rhsmode23_collision_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_TZFFT_REG", "").strip()
    try:
        reg = float(reg_env) if reg_env else 1e-8
    except ValueError:
        reg = 1e-8
    # For PAS+Er systems (RHSMode=1 constraintScheme=2 branch), the L=0 diagonal can be
    # exactly zero (krook=0) while the streaming/mirror operator has a constant-mode
    # nullspace in Fourier space. A very small `reg` can therefore produce enormous
    # tridiagonal-solve factors and cause Krylov divergence when this preconditioner is
    # used as part of a PAS hybrid. Use a more conservative L=0 lift in that branch.
    reg_l0 = float(reg)
    if op.fblock.pas is not None and op.fblock.fp is None and (
        op.fblock.er_xdot is not None or op.fblock.er_xidot is not None
    ):
        reg_l0 = max(reg_l0, 1.0)

    precond_dtype = _precond_dtype()
    complex_dtype = jnp.complex64 if precond_dtype == jnp.float32 else jnp.complex128
    cache_key = _transport_precond_cache_key(op, "tzfft") + (float(reg), str(complex_dtype))
    cached = _TRANSPORT_TZFFT_PRECOND_CACHE.get(cache_key)
    if cached is None:
        cl = op.fblock.collisionless
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)

        # Circulant FD derivative matrices: eigenvalues are FFT of first column.
        ddtheta0 = jnp.asarray(cl.ddtheta[:, 0], dtype=complex_dtype)
        ddzeta0 = jnp.asarray(cl.ddzeta[:, 0], dtype=complex_dtype)
        eig_theta = jnp.fft.fft(ddtheta0)  # (T,)
        eig_zeta = jnp.fft.fft(ddzeta0)  # (Z,)

        factor = _fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)  # (T,Z)
        wsum = jnp.sum(factor)
        wsum = jnp.where(wsum != 0, wsum, jnp.asarray(1.0, dtype=jnp.float64))

        v_theta = jnp.sum(factor * (op.b_hat_sup_theta / op.b_hat)) / wsum
        v_zeta = jnp.sum(factor * (op.b_hat_sup_zeta / op.b_hat)) / wsum
        mirror_geom = op.b_hat_sup_theta * op.db_hat_dtheta + op.b_hat_sup_zeta * op.db_hat_dzeta
        mirror_base = jnp.sum(factor * (mirror_geom / (2.0 * (op.b_hat**2)))) / wsum

        sqrt_t_over_m = jnp.sqrt(op.t_hat / op.m_hat)  # (S,)
        v_theta_s = jnp.asarray(sqrt_t_over_m * v_theta, dtype=jnp.float64)  # (S,)
        v_zeta_s = jnp.asarray(sqrt_t_over_m * v_zeta, dtype=jnp.float64)  # (S,)
        mirror_factor_s = jnp.asarray(-sqrt_t_over_m * mirror_base, dtype=jnp.float64)  # (S,)

        d_symbol = (
            v_theta_s[:, None, None] * eig_theta[None, :, None]
            + v_zeta_s[:, None, None] * eig_zeta[None, None, :]
        )  # (S,T,Z) complex

        ell = jnp.arange(n_l, dtype=jnp.float64)
        coef_plus = (ell + 1.0) / (2.0 * ell + 3.0)  # (L,)
        coef_minus = jnp.where(ell > 0, ell / (2.0 * ell - 1.0), 0.0)  # (L,)
        coef_mirror_plus = (ell + 1.0) * (ell + 2.0) / (2.0 * ell + 3.0)  # (L,)
        coef_mirror_minus = jnp.where(ell > 1, -ell * (ell - 1.0) / (2.0 * ell - 1.0), 0.0)  # (L,)

        coef_plus_x = op.x[:, None] * coef_plus[None, :]  # (X,L)
        coef_minus_x = op.x[:, None] * coef_minus[None, :]
        coef_mirror_plus_x = op.x[:, None] * coef_mirror_plus[None, :]
        coef_mirror_minus_x = op.x[:, None] * coef_mirror_minus[None, :]

        # Tridiagonal coefficients over L for each (s,x,theta,zeta).
        sup = d_symbol[:, None, :, :, None] * coef_plus_x[None, :, None, None, :-1]  # (S,X,T,Z,L-1)
        sup = sup + (mirror_factor_s[:, None, None, None, None] * coef_mirror_plus_x[None, :, None, None, :-1]).astype(complex_dtype)
        sub = d_symbol[:, None, :, :, None] * coef_minus_x[None, :, None, None, 1:]  # (S,X,T,Z,L-1)
        sub = sub + (mirror_factor_s[:, None, None, None, None] * coef_mirror_minus_x[None, :, None, None, 1:]).astype(complex_dtype)

        diag_base = jnp.asarray(float(op.fblock.identity_shift) + float(reg), dtype=jnp.float64)
        diag = jnp.full((n_species, n_x, n_theta, n_zeta, n_l), diag_base, dtype=complex_dtype)
        if n_l > 0 and reg_l0 != float(reg):
            diag = diag.at[..., 0].add(jnp.asarray(reg_l0 - float(reg), dtype=complex_dtype))

        # Add diagonal collision/exb contributions that are diagonal in L.
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = jnp.arange(n_l, dtype=jnp.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * pas.krook)
            pas_diag = pas.nu_n * pas.nu_d_hat[:, :, None] * factor_l[None, None, :]  # (S,X,L)
            diag = diag + pas_diag[:, :, None, None, :].astype(complex_dtype)

        if op.fblock.fp is not None:
            mat = op.fblock.fp.mat  # (S,S,L,X,X)
            diag_x = jnp.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
            diag_self = jnp.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
            diag_self = jnp.transpose(diag_self, (2, 1, 0))  # (S,X,L)
            diag = diag + diag_self[:, :, None, None, :].astype(complex_dtype)

        # Approximate ExB drift using a flux-surface-averaged coefficient so it is diagonal in Fourier space.
        if op.fblock.exb_theta is not None or op.fblock.exb_zeta is not None:
            use_dkes_exb = bool(getattr(op.fblock.exb_theta, "use_dkes_exb_drift", False)) if op.fblock.exb_theta is not None else bool(getattr(op.fblock.exb_zeta, "use_dkes_exb_drift", False))
            if use_dkes_exb:
                denom = jnp.asarray(op.fsab_hat2, dtype=jnp.float64)
                denom = jnp.where(denom != 0, denom, jnp.asarray(1.0, dtype=jnp.float64))
                coef_theta = (op.d_hat * op.b_hat_sub_zeta) / denom
                coef_zeta = (op.d_hat * op.b_hat_sub_theta) / denom
            else:
                denom = op.b_hat**2
                coef_theta = (op.d_hat * op.b_hat_sub_zeta) / denom
                coef_zeta = (op.d_hat * op.b_hat_sub_theta) / denom
            coef_theta_avg = jnp.sum(factor * coef_theta) / wsum
            coef_zeta_avg = jnp.sum(factor * coef_zeta) / wsum
            exb_factor = float(op.alpha) * float(op.delta) * 0.5 * float(op.dphi_hat_dpsi_hat)
            exb_theta_symbol = jnp.asarray(exb_factor * coef_theta_avg, dtype=complex_dtype) * eig_theta  # (T,)
            exb_zeta_symbol = jnp.asarray(-exb_factor * coef_zeta_avg, dtype=complex_dtype) * eig_zeta  # (Z,)
            diag_exb = exb_theta_symbol[None, None, :, None, None] + exb_zeta_symbol[None, None, None, :, None]
            diag = diag + diag_exb

        # Mask invalid L modes per x.
        nxi_for_x = jnp.asarray(_nxi_for_x_or_all(op), dtype=jnp.int32)
        active = jnp.arange(n_l, dtype=jnp.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        active_s = active[None, :, None, None, :]  # (1,X,1,1,L)
        diag = jnp.where(active_s, diag, jnp.asarray(1.0 + 0.0j, dtype=complex_dtype))
        link_active = active[:, :-1] & active[:, 1:]  # (X,L-1)
        link_s = link_active[None, :, None, None, :]  # (1,X,1,1,L-1)
        sup = jnp.where(link_s, sup, jnp.asarray(0.0 + 0.0j, dtype=complex_dtype))
        sub = jnp.where(link_s, sub, jnp.asarray(0.0 + 0.0j, dtype=complex_dtype))

        # `lax.linalg.tridiagonal_solve` expects all three diagonals to have the same shape (..., n),
        # with subdiag[..., 0] and superdiag[..., -1] ignored.
        z0 = jnp.zeros((n_species, n_x, n_theta, n_zeta, 1), dtype=complex_dtype)
        sub_full = jnp.concatenate([z0, sub], axis=-1)  # (..., L)
        sup_full = jnp.concatenate([sup, z0], axis=-1)  # (..., L)

        cached = _TransportTzFftPrecondCache(
            subdiag=jnp.asarray(sub_full, dtype=complex_dtype),
            diag=jnp.asarray(diag, dtype=complex_dtype),
            superdiag=jnp.asarray(sup_full, dtype=complex_dtype),
        )
        _TRANSPORT_TZFFT_PRECOND_CACHE[cache_key] = cached

    subdiag = cached.subdiag
    diag = cached.diag
    superdiag = cached.superdiag

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_hat = jnp.fft.fftn(f.astype(complex_dtype), axes=(-2, -1))  # (S,X,L,T,Z)
        rhs = jnp.transpose(f_hat, (0, 1, 3, 4, 2))  # (S,X,T,Z,L)
        rhs = rhs[..., None]  # (S,X,T,Z,L,1)
        sol = jax.lax.linalg.tridiagonal_solve(subdiag, diag, superdiag, rhs)
        sol = sol[..., 0]  # (S,X,T,Z,L)
        sol = jnp.transpose(sol, (0, 1, 4, 2, 3))  # (S,X,L,T,Z)
        z_f = jnp.fft.ifftn(sol, axes=(-2, -1)).real.astype(jnp.float64)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a block-Jacobi preconditioner for RHSMode=2/3 transport solves.

    Uses the same local x/L block structure as RHSMode=1 preconditioning, but
    applies it to the transport operator (RHSMode=2/3) using a simplified
    operator with diagonalized theta/zeta derivatives.
    """
    cache_key = _transport_precond_cache_key(op, "block")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE23_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_transport_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = _nxi_for_x_or_all(op)
        local_per_species = int(np.sum(nxi_for_x))

        rep_indices_by_species: list[np.ndarray] = []
        for s in range(n_s):
            idx: list[int] = []
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0)
                    idx.append(int(f_idx))
            rep_indices_by_species.append(np.asarray(idx, dtype=np.int32))

        reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_REG", "").strip()
        if not reg_env:
            reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10

        block_inv = np.zeros((n_s, local_per_species, local_per_species), dtype=np.float64)
        for s in range(n_s):
            rep_idx = rep_indices_by_species[s]
            chunk_cols = _precond_chunk_cols(total, int(np.asarray(rep_idx).shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=rep_idx,
                row_idx=rep_idx,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(local_per_species, dtype=np.float64)
            try:
                inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(inv)):
                inv = np.linalg.pinv(a, rcond=1e-12)
            block_inv[s, :, :] = inv

        idx_map = np.zeros((n_s, n_t, n_z, local_per_species), dtype=np.int32)
        for s in range(n_s):
            for it in range(n_t):
                for iz in range(n_z):
                    k = 0
                    for ix in range(n_x):
                        max_l = int(nxi_for_x[ix])
                        for il in range(max_l):
                            idx_map[s, it, iz, k] = int(
                                ((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz)
                            )
                            k += 1

        idx_map_jnp = jnp.asarray(idx_map, dtype=jnp.int32)
        flat_idx_jnp = idx_map_jnp.reshape((-1,))
        block_inv_jnp = jnp.asarray(block_inv, dtype=precond_dtype)

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        extra_idx_jnp = jnp.asarray(extra_idx_np, dtype=jnp.int32)
        extra_inv_jnp: jnp.ndarray | None = None
        if extra_size > 0:
            chunk_cols = _precond_chunk_cols(total, int(extra_idx_np.shape[0]))
            y_sub = _matvec_submatrix(
                op_pc,
                col_idx=extra_idx_np,
                row_idx=extra_idx_np,
                total_size=total,
                chunk_cols=chunk_cols,
            )
            a = np.asarray(y_sub.T, dtype=np.float64)
            a = a + reg * np.eye(extra_size, dtype=np.float64)
            try:
                extra_inv = np.linalg.inv(a)
            except np.linalg.LinAlgError:
                extra_inv = np.linalg.pinv(a, rcond=1e-12)
            if not np.all(np.isfinite(extra_inv)):
                extra_inv = np.linalg.pinv(a, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(extra_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE23_PRECOND_CACHE[cache_key] = cached

    idx_map_jnp = cached.idx_map_jnp
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_s = int(op.n_species)
    n_t = int(op.n_theta)
    n_z = int(op.n_zeta)
    local_per_species = int(idx_map_jnp.shape[-1])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        r_loc = r_full[flat_idx_jnp].reshape((n_s, n_t, n_z, local_per_species))
        z_loc = jnp.einsum("sab,stzb->stza", block_inv_jnp, r_loc)
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


def build_rhsmode23_fp_tzfft_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Fourier-space FP transport preconditioner for high-collisionality 3D runs.

    The W7-X high-``nu'`` FP transport lane is stiff because the Fokker-Planck
    collision block is dense in speed/species while the streaming/mirror terms
    dominate the angular Krylov spectrum. This preconditioner keeps the full FP
    dense block in ``(species, x)`` for each Legendre mode and adds a
    flux-surface-averaged streaming/mirror symbol in Fourier space. Each
    ``(k_theta, k_zeta)`` mode then solves one modest dense block over
    ``L * species * x``.

    It is intentionally opt-in through ``SFINCS_JAX_TRANSPORT_PRECOND=fp_tzfft``
    until full W7-X high-``nu'`` GPU residual benchmarks prove it should become
    a default.
    """
    if op.fblock.fp is None:
        return build_rhsmode23_tzfft_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if op.fblock.collisionless is None:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_block = int(n_species * n_x)
    n_mode = int(n_l * n_block)
    if n_mode <= 0 or n_theta <= 0 or n_zeta <= 0:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    precond_dtype = _precond_dtype()
    complex_dtype = jnp.complex64 if precond_dtype == jnp.float32 else jnp.complex128
    bytes_per_complex = 8.0 if complex_dtype == jnp.complex64 else 16.0
    est_mb = float(n_theta * n_zeta * n_mode * n_mode) * bytes_per_complex / 1.0e6
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB", "").strip()
    try:
        max_mb = float(max_env) if max_env else 384.0
    except ValueError:
        max_mb = 384.0
    if max_mb > 0.0 and est_mb > max_mb:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_REG", "").strip()
    try:
        reg = float(reg_env) if reg_env else 1.0e-10
    except ValueError:
        reg = 1.0e-10
    pinv_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_PINV_RCOND", "").strip()
    try:
        pinv_rcond = float(pinv_env) if pinv_env else 1.0e-12
    except ValueError:
        pinv_rcond = 1.0e-12

    cache_key = _transport_precond_cache_key(op, f"fp_tzfft_{complex_dtype}_{float(reg):.3e}")
    cached = _TRANSPORT_FP_TZFFT_PRECOND_CACHE.get(cache_key)
    if cached is None:
        cl = op.fblock.collisionless
        fp = op.fblock.fp
        assert cl is not None
        assert fp is not None

        ddtheta0 = np.asarray(cl.ddtheta[:, 0], dtype=np.complex128)
        ddzeta0 = np.asarray(cl.ddzeta[:, 0], dtype=np.complex128)
        eig_theta = np.fft.fft(ddtheta0)  # (T,)
        eig_zeta = np.fft.fft(ddzeta0)  # (Z,)

        factor = np.asarray(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat), dtype=np.float64)
        wsum = float(np.sum(factor))
        if wsum == 0.0:
            wsum = 1.0

        b_hat = np.asarray(op.b_hat, dtype=np.float64)
        b_sup_theta = np.asarray(op.b_hat_sup_theta, dtype=np.float64)
        b_sup_zeta = np.asarray(op.b_hat_sup_zeta, dtype=np.float64)
        db_dtheta = np.asarray(op.db_hat_dtheta, dtype=np.float64)
        db_dzeta = np.asarray(op.db_hat_dzeta, dtype=np.float64)

        v_theta = float(np.sum(factor * (b_sup_theta / b_hat)) / wsum)
        v_zeta = float(np.sum(factor * (b_sup_zeta / b_hat)) / wsum)
        mirror_geom = b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta
        mirror_base = float(np.sum(factor * (mirror_geom / (2.0 * (b_hat**2)))) / wsum)

        sqrt_t_over_m = np.sqrt(np.asarray(op.t_hat, dtype=np.float64) / np.asarray(op.m_hat, dtype=np.float64))
        v_theta_s = sqrt_t_over_m * v_theta
        v_zeta_s = sqrt_t_over_m * v_zeta
        mirror_factor_s = -sqrt_t_over_m * mirror_base
        d_symbol = (
            v_theta_s[:, None, None] * eig_theta[None, :, None]
            + v_zeta_s[:, None, None] * eig_zeta[None, None, :]
        )  # (S,T,Z)

        exb_symbol = np.zeros((n_theta, n_zeta), dtype=np.complex128)
        if op.fblock.exb_theta is not None or op.fblock.exb_zeta is not None:
            if op.fblock.exb_theta is not None:
                exb_theta = op.fblock.exb_theta
                if getattr(exb_theta, "use_dkes_exb_drift", False):
                    denom = float(np.asarray(exb_theta.fsab_hat2, dtype=np.float64).reshape(()))
                    coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / denom
                else:
                    coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / (
                        np.asarray(exb_theta.b_hat, dtype=np.float64) ** 2
                    )
                coef_avg = float(np.sum(factor * coef) / wsum)
                exb_factor = (
                    float(np.asarray(exb_theta.alpha, dtype=np.float64).reshape(()))
                    * float(np.asarray(exb_theta.delta, dtype=np.float64).reshape(()))
                    * 0.5
                    * float(np.asarray(exb_theta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
                )
                exb_symbol += (exb_factor * coef_avg * eig_theta)[:, None]
            if op.fblock.exb_zeta is not None:
                exb_zeta = op.fblock.exb_zeta
                if getattr(exb_zeta, "use_dkes_exb_drift", False):
                    denom = float(np.asarray(exb_zeta.fsab_hat2, dtype=np.float64).reshape(()))
                    coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / denom
                else:
                    coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / (
                        np.asarray(exb_zeta.b_hat, dtype=np.float64) ** 2
                    )
                coef_avg = float(np.sum(factor * coef) / wsum)
                exb_factor = (
                    -float(np.asarray(exb_zeta.alpha, dtype=np.float64).reshape(()))
                    * float(np.asarray(exb_zeta.delta, dtype=np.float64).reshape(()))
                    * 0.5
                    * float(np.asarray(exb_zeta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
                )
                exb_symbol += (exb_factor * coef_avg * eig_zeta)[None, :]

        mat_fp = np.asarray(fp.mat, dtype=np.float64)  # (S,S,L,X,X)
        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        nxi_for_x = _nxi_for_x_or_all(op)
        active = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        x_arr = np.asarray(cl.x, dtype=np.float64)
        l_arr = np.arange(n_l, dtype=np.float64)
        coef_plus = x_arr[:, None] * (l_arr[None, :] + 1.0) / (2.0 * l_arr[None, :] + 3.0)
        coef_minus = np.where(l_arr[None, :] > 0, x_arr[:, None] * l_arr[None, :] / (2.0 * l_arr[None, :] - 1.0), 0.0)
        coef_mirror_plus = x_arr[:, None] * (l_arr[None, :] + 1.0) * (l_arr[None, :] + 2.0) / (
            2.0 * l_arr[None, :] + 3.0
        )
        coef_mirror_minus = np.where(
            l_arr[None, :] > 1,
            -x_arr[:, None] * l_arr[None, :] * (l_arr[None, :] - 1.0) / (2.0 * l_arr[None, :] - 1.0),
            0.0,
        )

        base_blocks: list[np.ndarray] = []
        for il in range(n_l):
            a_l = np.array(mat_fp[:, :, il, :, :], dtype=np.float64, copy=True)
            a_l = a_l.transpose(0, 2, 1, 3).reshape((n_block, n_block))
            diag_add = np.full((n_block,), identity_shift + float(reg), dtype=np.float64)
            if pas_diag is not None:
                diag_add += pas_diag[:, :, il].reshape((n_block,))
            a_l[np.arange(n_block), np.arange(n_block)] += diag_add
            inactive_x = np.where(~active[:, il])[0]
            for ix in inactive_x:
                for s in range(n_species):
                    p = int(s * n_x + int(ix))
                    a_l[p, :] = 0.0
                    a_l[:, p] = 0.0
                    a_l[p, p] = 1.0
            base_blocks.append(a_l)

        inv_mode = np.zeros((n_theta, n_zeta, n_mode, n_mode), dtype=np.complex128)
        for kt in range(n_theta):
            for kz in range(n_zeta):
                a = np.zeros((n_mode, n_mode), dtype=np.complex128)
                for il, a_l in enumerate(base_blocks):
                    r0 = int(il * n_block)
                    r1 = r0 + n_block
                    a[r0:r1, r0:r1] = a_l.astype(np.complex128)
                    if exb_symbol[kt, kz] != 0.0:
                        for s in range(n_species):
                            for ix in range(n_x):
                                if active[ix, il]:
                                    p = r0 + int(s * n_x + ix)
                                    a[p, p] += exb_symbol[kt, kz]
                for il in range(n_l):
                    for s in range(n_species):
                        symbol = d_symbol[s, kt, kz]
                        mirror_symbol = mirror_factor_s[s]
                        for ix in range(n_x):
                            p = int(il * n_block + s * n_x + ix)
                            if il + 1 < n_l and active[ix, il] and active[ix, il + 1]:
                                q = int((il + 1) * n_block + s * n_x + ix)
                                a[p, q] += coef_plus[ix, il] * symbol + coef_mirror_plus[ix, il] * mirror_symbol
                            if il - 1 >= 0 and active[ix, il] and active[ix, il - 1]:
                                q = int((il - 1) * n_block + s * n_x + ix)
                                a[p, q] += coef_minus[ix, il] * symbol + coef_mirror_minus[ix, il] * mirror_symbol
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=pinv_rcond)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=pinv_rcond)
                inv_mode[kt, kz, :, :] = inv

        cached = _TransportFpTzFftPrecondCache(
            inv_mode=jnp.asarray(inv_mode, dtype=complex_dtype),
            n_block=int(n_block),
        )
        _TRANSPORT_FP_TZFFT_PRECOND_CACHE[cache_key] = cached

    inv_mode = cached.inv_mode
    n_block = int(cached.n_block)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_hat = jnp.fft.fftn(f.astype(complex_dtype), axes=(-2, -1))
        rhs_modes = jnp.transpose(f_hat, (3, 4, 2, 0, 1)).reshape((n_theta, n_zeta, n_l * n_block))
        sol_modes = jnp.einsum("tzij,tzj->tzi", inv_mode, rhs_modes)
        sol = sol_modes.reshape((n_theta, n_zeta, n_l, n_species, n_x))
        sol_f = jnp.transpose(sol, (3, 4, 2, 0, 1))  # (S,X,L,T,Z)
        z_f = jnp.fft.ifftn(sol_f, axes=(-2, -1)).real.astype(jnp.float64)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_fp_tzfft_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Bounded FP transport preconditioner using Fourier block-Thomas factors.

    This is the production-sized replacement for the dense ``fp_tzfft`` inverse
    table.  The approximation keeps the full FP block over ``(species, x)`` for
    each Legendre row and a flux-surface-averaged streaming/mirror symbol in
    Fourier space.  The Legendre coupling is block-tridiagonal, so setup stores
    only effective ``(species*x)`` block inverses instead of one dense inverse
    over all ``L*species*x`` unknowns per Fourier mode.
    """
    if op.fblock.fp is None:
        return build_rhsmode23_tzfft_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if op.fblock.collisionless is None:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_block = int(n_species * n_x)
    n_tz = int(n_theta * n_zeta)
    if n_block <= 0 or n_l <= 0 or n_tz <= 0:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    precond_dtype = _precond_dtype(int(n_tz * n_l * n_block * n_block))
    dtype_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "").strip().lower()
    if dtype_env == "float64":
        precond_dtype = jnp.float64
    elif dtype_env == "float32":
        precond_dtype = jnp.float32
    complex_dtype = jnp.complex64 if precond_dtype == jnp.float32 else jnp.complex128
    complex_np = np.complex64 if complex_dtype == jnp.complex64 else np.complex128
    bytes_per_complex = np.dtype(complex_np).itemsize
    est_mb = float(n_tz * n_l * (n_block * n_block + 2 * n_block)) * float(bytes_per_complex) / 1.0e6
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_MAX_MB", "").strip()
    try:
        max_mb = float(max_env) if max_env else 2048.0
    except ValueError:
        max_mb = 2048.0
    if max_mb > 0.0 and est_mb > float(max_mb):
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_REG", "").strip()
    try:
        reg = float(reg_env) if reg_env else 1.0e-10
    except ValueError:
        reg = 1.0e-10
    pinv_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_PINV_RCOND", "").strip()
    try:
        pinv_rcond = float(pinv_env) if pinv_env else 1.0e-12
    except ValueError:
        pinv_rcond = 1.0e-12

    cache_key = (
        *_transport_precond_cache_key(op, f"fp_tzfft_line_{complex_dtype}_{float(reg):.3e}"),
        _hash_array(op.b_hat_sup_theta),
        _hash_array(op.b_hat_sup_zeta),
        _hash_array(op.db_hat_dtheta),
        _hash_array(op.db_hat_dzeta),
        _hash_array(op.x),
        _hash_array(op.t_hat),
        _hash_array(op.m_hat),
        float(est_mb),
    )
    cached = _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE.get(cache_key)
    if cached is None:
        cl = op.fblock.collisionless
        fp = op.fblock.fp
        assert cl is not None
        assert fp is not None

        ddtheta0 = np.asarray(cl.ddtheta[:, 0], dtype=complex_np)
        ddzeta0 = np.asarray(cl.ddzeta[:, 0], dtype=complex_np)
        eig_theta = np.fft.fft(ddtheta0)
        eig_zeta = np.fft.fft(ddzeta0)

        factor = np.asarray(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat), dtype=np.float64)
        wsum = float(np.sum(factor))
        if wsum == 0.0:
            wsum = 1.0
        b_hat = np.asarray(op.b_hat, dtype=np.float64)
        b_sup_theta = np.asarray(op.b_hat_sup_theta, dtype=np.float64)
        b_sup_zeta = np.asarray(op.b_hat_sup_zeta, dtype=np.float64)
        db_dtheta = np.asarray(op.db_hat_dtheta, dtype=np.float64)
        db_dzeta = np.asarray(op.db_hat_dzeta, dtype=np.float64)
        v_theta = float(np.sum(factor * (b_sup_theta / b_hat)) / wsum)
        v_zeta = float(np.sum(factor * (b_sup_zeta / b_hat)) / wsum)
        mirror_geom = b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta
        mirror_base = float(np.sum(factor * (mirror_geom / (2.0 * (b_hat**2)))) / wsum)

        sqrt_t_over_m = np.sqrt(np.asarray(op.t_hat, dtype=np.float64) / np.asarray(op.m_hat, dtype=np.float64))
        v_theta_s = sqrt_t_over_m * v_theta
        v_zeta_s = sqrt_t_over_m * v_zeta
        mirror_factor_s = -sqrt_t_over_m * mirror_base
        d_symbol = (
            v_theta_s[:, None, None] * eig_theta[None, :, None]
            + v_zeta_s[:, None, None] * eig_zeta[None, None, :]
        ).astype(complex_np, copy=False)  # (S,T,Z)

        exb_symbol = np.zeros((n_theta, n_zeta), dtype=complex_np)
        if op.fblock.exb_theta is not None or op.fblock.exb_zeta is not None:
            if op.fblock.exb_theta is not None:
                exb_theta = op.fblock.exb_theta
                if getattr(exb_theta, "use_dkes_exb_drift", False):
                    denom = float(np.asarray(exb_theta.fsab_hat2, dtype=np.float64).reshape(()))
                    coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / denom
                else:
                    coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / (
                        np.asarray(exb_theta.b_hat, dtype=np.float64) ** 2
                    )
                coef_avg = float(np.sum(factor * coef) / wsum)
                exb_factor = (
                    float(np.asarray(exb_theta.alpha, dtype=np.float64).reshape(()))
                    * float(np.asarray(exb_theta.delta, dtype=np.float64).reshape(()))
                    * 0.5
                    * float(np.asarray(exb_theta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
                )
                exb_symbol += (exb_factor * coef_avg * eig_theta)[:, None]
            if op.fblock.exb_zeta is not None:
                exb_zeta = op.fblock.exb_zeta
                if getattr(exb_zeta, "use_dkes_exb_drift", False):
                    denom = float(np.asarray(exb_zeta.fsab_hat2, dtype=np.float64).reshape(()))
                    coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / denom
                else:
                    coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / (
                        np.asarray(exb_zeta.b_hat, dtype=np.float64) ** 2
                    )
                coef_avg = float(np.sum(factor * coef) / wsum)
                exb_factor = (
                    -float(np.asarray(exb_zeta.alpha, dtype=np.float64).reshape(()))
                    * float(np.asarray(exb_zeta.delta, dtype=np.float64).reshape(()))
                    * 0.5
                    * float(np.asarray(exb_zeta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
                )
                exb_symbol += (exb_factor * coef_avg * eig_zeta)[None, :]

        mat_fp = np.asarray(fp.mat, dtype=np.float64)  # (S,S,L,X,X)
        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr_pas = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr_pas * (l_arr_pas + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        nxi_for_x = _nxi_for_x_or_all(op)
        active = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        x_arr = np.asarray(cl.x, dtype=np.float64)
        l_arr = np.arange(n_l, dtype=np.float64)
        coef_plus = x_arr[:, None] * (l_arr[None, :] + 1.0) / (2.0 * l_arr[None, :] + 3.0)
        coef_minus = np.where(l_arr[None, :] > 0, x_arr[:, None] * l_arr[None, :] / (2.0 * l_arr[None, :] - 1.0), 0.0)
        coef_mirror_plus = x_arr[:, None] * (l_arr[None, :] + 1.0) * (l_arr[None, :] + 2.0) / (
            2.0 * l_arr[None, :] + 3.0
        )
        coef_mirror_minus = np.where(
            l_arr[None, :] > 1,
            -x_arr[:, None] * l_arr[None, :] * (l_arr[None, :] - 1.0) / (2.0 * l_arr[None, :] - 1.0),
            0.0,
        )

        base_blocks = np.zeros((n_l, n_block, n_block), dtype=np.float64)
        for il in range(n_l):
            a_l = np.array(mat_fp[:, :, il, :, :], dtype=np.float64, copy=True)
            a_l = a_l.transpose(0, 2, 1, 3).reshape((n_block, n_block))
            diag_add = np.full((n_block,), identity_shift + float(reg), dtype=np.float64)
            if pas_diag is not None:
                diag_add += pas_diag[:, :, il].reshape((n_block,))
            a_l[np.arange(n_block), np.arange(n_block)] += diag_add
            inactive_x = np.where(~active[:, il])[0]
            for ix in inactive_x:
                for s in range(n_species):
                    p = int(s * n_x + int(ix))
                    a_l[p, :] = 0.0
                    a_l[:, p] = 0.0
                    a_l[p, p] = 1.0
            base_blocks[il, :, :] = a_l

        mode_count = int(n_tz)
        d_symbol_flat = d_symbol.reshape((n_species, mode_count))
        exb_flat = exb_symbol.reshape((mode_count,))
        lower_flat = np.zeros((mode_count, n_l, n_block), dtype=complex_np)
        super_flat = np.zeros((mode_count, n_l, n_block), dtype=complex_np)
        for s in range(n_species):
            symbol_s = d_symbol_flat[s, :]
            mirror_s = complex_np(mirror_factor_s[s])
            for ix in range(n_x):
                p = int(s * n_x + ix)
                if n_l > 1:
                    link_plus = active[ix, :-1] & active[ix, 1:]
                    vals_super = (
                        coef_plus[ix, :-1][None, :] * symbol_s[:, None]
                        + coef_mirror_plus[ix, :-1][None, :] * mirror_s
                    )
                    super_flat[:, :-1, p] = np.where(link_plus[None, :], vals_super, 0.0)
                    vals_lower = (
                        coef_minus[ix, 1:][None, :] * symbol_s[:, None]
                        + coef_mirror_minus[ix, 1:][None, :] * mirror_s
                    )
                    lower_flat[:, 1:, p] = np.where(link_plus[None, :], vals_lower, 0.0)

        diag_idx = np.arange(n_block, dtype=np.intp)
        inv_eff = np.empty((mode_count, n_l, n_block, n_block), dtype=complex_np)
        prev_g = np.zeros((mode_count, n_block, n_block), dtype=complex_np)

        def _invert_stack(a_stack: np.ndarray) -> np.ndarray:
            try:
                return np.linalg.inv(a_stack)
            except np.linalg.LinAlgError:
                out = np.empty_like(a_stack)
                for i_mode in range(int(a_stack.shape[0])):
                    try:
                        out[i_mode, :, :] = np.linalg.inv(a_stack[i_mode, :, :])
                    except np.linalg.LinAlgError:
                        out[i_mode, :, :] = np.linalg.pinv(a_stack[i_mode, :, :], rcond=pinv_rcond)
                return out

        for il in range(n_l):
            d_eff = np.broadcast_to(base_blocks[il, :, :].astype(complex_np), (mode_count, n_block, n_block)).copy()
            if np.any(exb_flat != 0.0):
                active_n = np.zeros((n_block,), dtype=np.float64)
                for s in range(n_species):
                    for ix in range(n_x):
                        active_n[int(s * n_x + ix)] = 1.0 if bool(active[ix, il]) else 0.0
                d_eff[:, diag_idx, diag_idx] += exb_flat[:, None] * active_n[None, :]
            if il > 0:
                d_eff -= lower_flat[:, il, :, None] * prev_g
            inv_l = _invert_stack(d_eff)
            if not np.all(np.isfinite(inv_l)):
                bad = ~np.isfinite(inv_l).reshape((mode_count, -1)).all(axis=1)
                for i_mode in np.where(bad)[0]:
                    inv_l[i_mode, :, :] = np.linalg.pinv(d_eff[i_mode, :, :], rcond=pinv_rcond)
            inv_eff[:, il, :, :] = inv_l.astype(complex_np, copy=False)
            if il + 1 < n_l:
                prev_g = inv_l * super_flat[:, il, None, :]
            else:
                prev_g = np.zeros_like(prev_g)

        cached = _TransportFpTzFftLinePrecondCache(
            inv_eff=jnp.asarray(inv_eff.reshape((n_theta, n_zeta, n_l, n_block, n_block)), dtype=complex_dtype),
            lower_diag=jnp.asarray(lower_flat.reshape((n_theta, n_zeta, n_l, n_block)), dtype=complex_dtype),
            super_diag=jnp.asarray(super_flat.reshape((n_theta, n_zeta, n_l, n_block)), dtype=complex_dtype),
            n_block=int(n_block),
        )
        _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE[cache_key] = cached

    inv_eff = cached.inv_eff
    lower_diag = cached.lower_diag
    super_diag = cached.super_diag
    n_block_cached = int(cached.n_block)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_hat = jnp.fft.fftn(f.astype(complex_dtype), axes=(-2, -1))
        rhs_modes = jnp.transpose(f_hat, (3, 4, 2, 0, 1)).reshape(
            (n_theta, n_zeta, n_l, n_block_cached)
        )
        rhs_ltz = jnp.transpose(rhs_modes, (2, 0, 1, 3))  # (L,T,Z,N)
        inv_ltz = jnp.transpose(inv_eff, (2, 0, 1, 3, 4))  # (L,T,Z,N,N)
        lower_ltz = jnp.transpose(lower_diag, (2, 0, 1, 3))  # (L,T,Z,N)
        super_ltz = jnp.transpose(super_diag, (2, 0, 1, 3))  # (L,T,Z,N)

        def _forward(prev: jnp.ndarray, data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            inv_l, lower_l, rhs_l = data
            rhs_eff = rhs_l - lower_l * prev
            y_l = jnp.einsum("tzij,tzj->tzi", inv_l, rhs_eff)
            return y_l, y_l

        zero_mode = jnp.zeros((n_theta, n_zeta, n_block_cached), dtype=complex_dtype)
        _, y_ltz = jax.lax.scan(_forward, zero_mode, (inv_ltz, lower_ltz, rhs_ltz))

        def _backward(next_x: jnp.ndarray, data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            y_l, inv_l, super_l = data
            corr = jnp.einsum("tzij,tzj->tzi", inv_l, super_l * next_x)
            x_l = y_l - corr
            return x_l, x_l

        if n_l > 1:
            _, x_rev = jax.lax.scan(
                _backward,
                y_ltz[-1],
                (
                    jnp.flip(y_ltz[:-1], axis=0),
                    jnp.flip(inv_ltz[:-1], axis=0),
                    jnp.flip(super_ltz[:-1], axis=0),
                ),
            )
            x_ltz = jnp.concatenate([jnp.flip(x_rev, axis=0), y_ltz[-1][None, ...]], axis=0)
        else:
            x_ltz = y_ltz

        sol_modes = jnp.transpose(x_ltz, (1, 2, 0, 3)).reshape(
            (n_theta, n_zeta, n_l, n_species, n_x)
        )
        sol_f = jnp.transpose(sol_modes, (3, 4, 2, 0, 1))  # (S,X,L,T,Z)
        z_f = jnp.fft.ifftn(sol_f, axes=(-2, -1)).real.astype(jnp.float64)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_fp_tzfft_line_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """FP Fourier line factor plus a small true-action Schur residual equation.

    The line factor handles the stiff local FP/streaming residual equation.  It
    does not by itself invert the global source/constraint tail coupling.  This
    wrapper builds a bounded coarse space containing tail/source response columns
    and low-order source-moment directions, then solves a tiny least-squares
    residual equation using columns of the *true* full operator.
    """
    base_full = build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    if op.fblock.fp is None or bool(op.include_phi1) or int(op.extra_size) <= 0:
        if reduce_full is None or expand_reduced is None:
            return base_full

        def _base_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
            return reduce_full(base_full(expand_reduced(r_reduced)))

        return _base_reduced

    max_cols_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_COLS", "").strip()
    max_mb_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_MB", "").strip()
    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_REG", "").strip()
    damping_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_DAMPING", "").strip()
    corr_rel_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_CORRECTION_REL_MAX", "").strip()
    restriction_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_RESTRICTION", "").strip().lower()
    dtype_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_DTYPE", "").strip().lower()
    try:
        max_cols = int(max_cols_env) if max_cols_env else 32
    except ValueError:
        max_cols = 32
    try:
        max_mb = float(max_mb_env) if max_mb_env else 512.0
    except ValueError:
        max_mb = 512.0
    try:
        reg = float(reg_env) if reg_env else 3.0e-13
    except ValueError:
        reg = 3.0e-13
    try:
        damping = float(damping_env) if damping_env else 1.0
    except ValueError:
        damping = 1.0
    try:
        correction_rel_max = float(corr_rel_env) if corr_rel_env else 10.0
    except ValueError:
        correction_rel_max = 10.0
    coarse_dtype = jnp.float32 if dtype_env == "float32" else jnp.float64
    dtype_np = np.float32 if coarse_dtype == jnp.float32 else np.float64
    restriction_kind = restriction_env if restriction_env in {"tail", "galerkin", "tail_galerkin"} else "tail"
    max_cols = max(0, int(max_cols))
    if max_cols <= 0:
        if reduce_full is None or expand_reduced is None:
            return base_full

        def _base_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
            return reduce_full(base_full(expand_reduced(r_reduced)))

        return _base_reduced

    n_total = int(op.total_size)
    bytes_per = np.dtype(dtype_np).itemsize
    est_mb = float(2 * n_total * max_cols * bytes_per + max_cols * max_cols * bytes_per) / 1.0e6
    if float(max_mb) > 0.0 and est_mb > float(max_mb):
        if reduce_full is None or expand_reduced is None:
            return base_full

        def _base_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
            return reduce_full(base_full(expand_reduced(r_reduced)))

        return _base_reduced

    cache_key = (
        *_transport_precond_cache_key(
            op,
            f"fp_tzfft_line_schur_{coarse_dtype}_{int(max_cols)}_"
            f"{float(reg):.3e}_{float(damping):.3e}_{float(correction_rel_max):.3e}_{restriction_kind}",
        ),
        _hash_array(op.x),
        _hash_array(op.x_weights),
        _hash_array(op.theta_weights),
        _hash_array(op.zeta_weights),
        _hash_array(op.d_hat),
        int(op.extra_size),
    )
    cached = _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE.get(cache_key)
    if cached is None:
        columns: list[np.ndarray] = []
        labels: list[str] = []

        def _add_column(label: str, vec: np.ndarray) -> None:
            if len(columns) >= int(max_cols):
                return
            arr = np.asarray(vec, dtype=np.float64).reshape((-1,))
            if arr.shape != (n_total,):
                return
            finite = np.isfinite(arr)
            if not np.all(finite):
                arr = np.where(finite, arr, 0.0)
            norm = float(np.linalg.norm(arr))
            if not (np.isfinite(norm) and norm > 0.0):
                return
            columns.append((arr / norm).astype(dtype_np, copy=False))
            labels.append(str(label))

        def _true_action(vec: np.ndarray) -> np.ndarray:
            return np.asarray(
                jax.device_get(apply_v3_full_system_operator_cached(op, jnp.asarray(vec, dtype=jnp.float64))),
                dtype=np.float64,
            ).reshape((-1,))

        def _base_response(load: np.ndarray) -> np.ndarray:
            return np.asarray(jax.device_get(base_full(jnp.asarray(load, dtype=jnp.float64))), dtype=np.float64).reshape((-1,))

        tail0 = int(op.f_size + op.phi1_size)
        for i_extra in range(int(op.extra_size)):
            if len(columns) >= int(max_cols):
                break
            unit = np.zeros((n_total,), dtype=np.float64)
            unit[tail0 + int(i_extra)] = 1.0
            source_col = _true_action(unit)
            response = _base_response(source_col)
            _add_column(f"tail_schur_response_{i_extra}", unit - response)
            _add_column(f"tail_unit_{i_extra}", unit)

        factor = np.asarray(jax.device_get(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)), dtype=np.float64)
        factor_norm = float(np.linalg.norm(factor))
        if np.isfinite(factor_norm) and factor_norm > 0.0:
            fs_pattern = factor / factor_norm
        else:
            fs_pattern = np.full((int(op.n_theta), int(op.n_zeta)), 1.0 / np.sqrt(max(1, int(op.n_theta) * int(op.n_zeta))))
        x = np.asarray(jax.device_get(op.x), dtype=np.float64)
        xw = np.asarray(jax.device_get(op.x_weights), dtype=np.float64)
        moment_specs = [
            ("density", 0, (x**2) * xw),
            ("pressure", 0, (x**4) * xw),
            ("flow", min(1, int(op.n_xi) - 1), (x**3) * xw),
            ("heat_flow", min(1, int(op.n_xi) - 1), (x**5) * xw),
        ]
        if int(op.constraint_scheme) == 2:
            for species in range(int(op.n_species)):
                for name, ell, weights in moment_specs:
                    if len(columns) >= int(max_cols):
                        break
                    f_dir = np.zeros(tuple(int(v) for v in op.fblock.f_shape), dtype=np.float64)
                    f_dir[species, :, int(ell), :, :] = weights[:, None, None] * fs_pattern[None, :, :]
                    full = np.concatenate([f_dir.reshape((-1,)), np.zeros((n_total - int(op.f_size),), dtype=np.float64)])
                    _add_column(f"constraint2_{name}_moment_s{species}_l{int(ell)}", full)
        elif int(op.constraint_scheme) == 1:
            xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
            xparts = [
                ("particle_source_shape", np.asarray(jax.device_get(xpart1), dtype=np.float64)),
                ("energy_source_shape", np.asarray(jax.device_get(xpart2), dtype=np.float64)),
            ]
            ix0 = _ix_min(bool(op.point_at_x0))
            for species in range(int(op.n_species)):
                for name, weights in xparts:
                    if len(columns) >= int(max_cols):
                        break
                    f_dir = np.zeros(tuple(int(v) for v in op.fblock.f_shape), dtype=np.float64)
                    f_dir[species, ix0:, 0, :, :] = weights[ix0:, None, None] * fs_pattern[None, :, :]
                    full = np.concatenate([f_dir.reshape((-1,)), np.zeros((n_total - int(op.f_size),), dtype=np.float64)])
                    _add_column(f"constraint1_{name}_s{species}", full)

        if not columns:
            if reduce_full is None or expand_reduced is None:
                return base_full

            def _base_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
                return reduce_full(base_full(expand_reduced(r_reduced)))

            return _base_reduced

        basis_np = np.column_stack(columns).astype(dtype_np, copy=False)
        action_columns = [_true_action(basis_np[:, i]) for i in range(int(basis_np.shape[1]))]
        action_full_np = np.column_stack(action_columns)
        restrict_np: np.ndarray | None
        if restriction_kind == "tail":
            restrict_np = None
            action_np = np.asarray(action_full_np[tail0:, :], dtype=dtype_np)
        else:
            restrict_parts: list[np.ndarray] = []
            if restriction_kind == "tail_galerkin":
                for i_extra in range(int(op.extra_size)):
                    unit = np.zeros((n_total,), dtype=dtype_np)
                    unit[tail0 + int(i_extra)] = 1.0
                    restrict_parts.append(unit)
            restrict_parts.extend([basis_np[:, i].astype(dtype_np, copy=False) for i in range(int(basis_np.shape[1]))])
            restrict_np = np.column_stack(restrict_parts).astype(dtype_np, copy=False)
            action_np = np.asarray(restrict_np.T @ action_full_np, dtype=dtype_np)
        try:
            normal_inv = np.linalg.pinv(action_np, rcond=max(float(abs(reg)), 1.0e-14))
        except np.linalg.LinAlgError:
            normal = np.asarray(action_np.T @ action_np, dtype=np.float64)
            scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
            normal_reg = normal + max(float(abs(reg)), 1.0e-14) * scale * np.eye(int(normal.shape[0]), dtype=np.float64)
            normal_inv = np.linalg.solve(normal_reg, action_np.T)
        cached = _TransportFpTzFftLineSchurPrecondCache(
            basis=jnp.asarray(basis_np, dtype=coarse_dtype),
            action=jnp.asarray(action_np, dtype=coarse_dtype),
            normal_inv=jnp.asarray(normal_inv.astype(dtype_np, copy=False), dtype=coarse_dtype),
            restrict_basis=None if restrict_np is None else jnp.asarray(restrict_np, dtype=coarse_dtype),
            damping=float(damping),
            tail0=int(tail0),
            n_columns=int(basis_np.shape[1]),
            restriction_kind=str(restriction_kind),
            basis_labels=tuple(labels),
        )
        _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE[cache_key] = cached

    basis = cached.basis
    action = cached.action
    normal_inv = cached.normal_inv
    restrict_basis = cached.restrict_basis
    damping_use = float(cached.damping)
    tail0_use = int(cached.tail0)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        y_base = base_full(r_full)
        residual = r_full - apply_v3_full_system_operator_cached(op, y_base)
        if restrict_basis is None:
            residual_c = jnp.asarray(residual[tail0_use:], dtype=action.dtype)
        else:
            residual_c = jnp.asarray(restrict_basis.T @ residual, dtype=action.dtype)
        coeff = normal_inv @ residual_c
        coeff = jnp.where(jnp.isfinite(coeff), coeff, jnp.zeros_like(coeff))
        correction = jnp.asarray(basis @ coeff, dtype=jnp.float64)
        correction = jnp.where(jnp.isfinite(correction), correction, jnp.zeros_like(correction))
        if float(correction_rel_max) > 0.0:
            corr_norm = jnp.linalg.norm(correction)
            ref_norm = jnp.maximum(jnp.maximum(jnp.linalg.norm(y_base), jnp.linalg.norm(r_full)), 1.0)
            limit = jnp.asarray(float(correction_rel_max), dtype=jnp.float64) * ref_norm
            scale = jnp.where(corr_norm > limit, limit / jnp.maximum(corr_norm, jnp.finfo(jnp.float64).tiny), 1.0)
        else:
            scale = jnp.asarray(1.0, dtype=jnp.float64)
        out = y_base + float(damping_use) * scale * correction
        return jnp.where(jnp.isfinite(out), out, y_base)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



def build_rhsmode23_fp_local_geom_line_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """FP line factor retaining local mirror geometry at each angular grid point.

    Unlike ``fp_tzfft_line``, this candidate does not flux-surface-average the
    mirror geometry before setup.  It is block diagonal in ``(theta,zeta)`` and
    block-tridiagonal in Legendre index with dense ``(species,x)`` blocks.
    """
    if op.fblock.fp is None:
        return build_rhsmode23_tzfft_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if op.fblock.collisionless is None:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_block = int(n_species * n_x)
    n_tz = int(n_theta * n_zeta)
    if n_block <= 0 or n_l <= 0 or n_tz <= 0:
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    dtype_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_DTYPE", "").strip().lower()
    precond_dtype = _precond_dtype(int(n_tz * n_l * n_block * n_block))
    if dtype_env == "float64":
        precond_dtype = jnp.float64
    elif dtype_env == "float32":
        precond_dtype = jnp.float32
    dtype_np = np.float32 if precond_dtype == jnp.float32 else np.float64
    bytes_per = np.dtype(dtype_np).itemsize
    est_mb = float(n_tz * n_l * (n_block * n_block + 2 * n_block)) * float(bytes_per) / 1.0e6
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_MAX_MB", "").strip()
    try:
        max_mb = float(max_env) if max_env else 2048.0
    except ValueError:
        max_mb = 2048.0
    if max_mb > 0.0 and est_mb > float(max_mb):
        return build_rhsmode23_sxblock_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    reg_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_REG", "").strip()
    try:
        reg = float(reg_env) if reg_env else 1.0e-10
    except ValueError:
        reg = 1.0e-10
    pinv_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_PINV_RCOND", "").strip()
    try:
        pinv_rcond = float(pinv_env) if pinv_env else 1.0e-12
    except ValueError:
        pinv_rcond = 1.0e-12

    cache_key = (
        *_transport_precond_cache_key(op, f"fp_local_geom_line_{precond_dtype}_{float(reg):.3e}"),
        _hash_array(op.b_hat),
        _hash_array(op.b_hat_sup_theta),
        _hash_array(op.b_hat_sup_zeta),
        _hash_array(op.db_hat_dtheta),
        _hash_array(op.db_hat_dzeta),
        _hash_array(op.x),
        _hash_array(op.t_hat),
        _hash_array(op.m_hat),
        float(est_mb),
    )
    cached = _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE.get(cache_key)
    if cached is None:
        cl = op.fblock.collisionless
        fp = op.fblock.fp
        assert cl is not None
        assert fp is not None

        mat_fp = np.asarray(fp.mat, dtype=np.float64)  # (S,S,L,X,X)
        identity_shift = float(op.fblock.identity_shift)
        pas_diag = None
        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr_pas = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr_pas * (l_arr_pas + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]

        nxi_for_x = _nxi_for_x_or_all(op)
        active = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        x_arr = np.asarray(cl.x, dtype=np.float64)
        l_arr = np.arange(n_l, dtype=np.float64)
        coef_mirror_plus = x_arr[:, None] * (l_arr[None, :] + 1.0) * (l_arr[None, :] + 2.0) / (
            2.0 * l_arr[None, :] + 3.0
        )
        coef_mirror_minus = np.where(
            l_arr[None, :] > 1,
            -x_arr[:, None] * l_arr[None, :] * (l_arr[None, :] - 1.0) / (2.0 * l_arr[None, :] - 1.0),
            0.0,
        )

        b_hat = np.asarray(op.b_hat, dtype=np.float64)
        b_sup_theta = np.asarray(op.b_hat_sup_theta, dtype=np.float64)
        b_sup_zeta = np.asarray(op.b_hat_sup_zeta, dtype=np.float64)
        db_dtheta = np.asarray(op.db_hat_dtheta, dtype=np.float64)
        db_dzeta = np.asarray(op.db_hat_dzeta, dtype=np.float64)
        sqrt_t_over_m = np.sqrt(np.asarray(op.t_hat, dtype=np.float64) / np.asarray(op.m_hat, dtype=np.float64))
        mirror_geom = (b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta) / (2.0 * (b_hat**2))
        mirror_factor = (-sqrt_t_over_m[:, None, None] * mirror_geom[None, :, :]).reshape((n_species, n_tz))

        base_blocks = np.zeros((n_l, n_block, n_block), dtype=np.float64)
        for il in range(n_l):
            a_l = np.array(mat_fp[:, :, il, :, :], dtype=np.float64, copy=True)
            a_l = a_l.transpose(0, 2, 1, 3).reshape((n_block, n_block))
            diag_add = np.full((n_block,), identity_shift + float(reg), dtype=np.float64)
            if pas_diag is not None:
                diag_add += pas_diag[:, :, il].reshape((n_block,))
            a_l[np.arange(n_block), np.arange(n_block)] += diag_add
            inactive_x = np.where(~active[:, il])[0]
            for ix in inactive_x:
                for s in range(n_species):
                    p = int(s * n_x + int(ix))
                    a_l[p, :] = 0.0
                    a_l[:, p] = 0.0
                    a_l[p, p] = 1.0
            base_blocks[il, :, :] = a_l

        lower_flat = np.zeros((n_tz, n_l, n_block), dtype=np.float64)
        super_flat = np.zeros((n_tz, n_l, n_block), dtype=np.float64)
        for s in range(n_species):
            mirror_s = mirror_factor[s, :]
            for ix in range(n_x):
                p = int(s * n_x + ix)
                if n_l > 1:
                    link_plus = active[ix, :-1] & active[ix, 1:]
                    vals_super = coef_mirror_plus[ix, :-1][None, :] * mirror_s[:, None]
                    vals_lower = coef_mirror_minus[ix, 1:][None, :] * mirror_s[:, None]
                    super_flat[:, :-1, p] = np.where(link_plus[None, :], vals_super, 0.0)
                    lower_flat[:, 1:, p] = np.where(link_plus[None, :], vals_lower, 0.0)

        inv_eff = np.empty((n_tz, n_l, n_block, n_block), dtype=dtype_np)
        prev_g = np.zeros((n_tz, n_block, n_block), dtype=np.float64)

        def _invert_stack(a_stack: np.ndarray) -> np.ndarray:
            try:
                return np.linalg.inv(a_stack)
            except np.linalg.LinAlgError:
                out = np.empty_like(a_stack)
                for i_mode in range(int(a_stack.shape[0])):
                    try:
                        out[i_mode, :, :] = np.linalg.inv(a_stack[i_mode, :, :])
                    except np.linalg.LinAlgError:
                        out[i_mode, :, :] = np.linalg.pinv(a_stack[i_mode, :, :], rcond=pinv_rcond)
                return out

        for il in range(n_l):
            d_eff = np.broadcast_to(base_blocks[il, :, :], (n_tz, n_block, n_block)).copy()
            if il > 0:
                d_eff -= lower_flat[:, il, :, None] * prev_g
            inv_l = _invert_stack(d_eff)
            if not np.all(np.isfinite(inv_l)):
                bad = ~np.isfinite(inv_l).reshape((n_tz, -1)).all(axis=1)
                for i_mode in np.where(bad)[0]:
                    inv_l[i_mode, :, :] = np.linalg.pinv(d_eff[i_mode, :, :], rcond=pinv_rcond)
            inv_eff[:, il, :, :] = inv_l.astype(dtype_np, copy=False)
            if il + 1 < n_l:
                prev_g = inv_l * super_flat[:, il, None, :]
            else:
                prev_g = np.zeros_like(prev_g)

        cached = _TransportFpLocalGeomLinePrecondCache(
            inv_eff=jnp.asarray(inv_eff.reshape((n_theta, n_zeta, n_l, n_block, n_block)), dtype=precond_dtype),
            lower_diag=jnp.asarray(lower_flat.reshape((n_theta, n_zeta, n_l, n_block)), dtype=precond_dtype),
            super_diag=jnp.asarray(super_flat.reshape((n_theta, n_zeta, n_l, n_block)), dtype=precond_dtype),
            n_block=int(n_block),
        )
        _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE[cache_key] = cached

    inv_eff = cached.inv_eff
    lower_diag = cached.lower_diag
    super_diag = cached.super_diag
    n_block_cached = int(cached.n_block)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        rhs_tzl = jnp.transpose(f.astype(precond_dtype), (3, 4, 2, 0, 1)).reshape(
            (n_theta, n_zeta, n_l, n_block_cached)
        )
        rhs_ltz = jnp.transpose(rhs_tzl, (2, 0, 1, 3))
        inv_ltz = jnp.transpose(inv_eff, (2, 0, 1, 3, 4))
        lower_ltz = jnp.transpose(lower_diag, (2, 0, 1, 3))
        super_ltz = jnp.transpose(super_diag, (2, 0, 1, 3))

        def _forward(prev: jnp.ndarray, data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            inv_l, lower_l, rhs_l = data
            rhs_eff = rhs_l - lower_l * prev
            y_l = jnp.einsum("tzij,tzj->tzi", inv_l, rhs_eff)
            return y_l, y_l

        zero_mode = jnp.zeros((n_theta, n_zeta, n_block_cached), dtype=precond_dtype)
        _, y_ltz = jax.lax.scan(_forward, zero_mode, (inv_ltz, lower_ltz, rhs_ltz))

        def _backward(next_x: jnp.ndarray, data: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]):
            y_l, inv_l, super_l = data
            corr = jnp.einsum("tzij,tzj->tzi", inv_l, super_l * next_x)
            x_l = y_l - corr
            return x_l, x_l

        if n_l > 1:
            _, x_rev = jax.lax.scan(
                _backward,
                y_ltz[-1],
                (
                    jnp.flip(y_ltz[:-1], axis=0),
                    jnp.flip(inv_ltz[:-1], axis=0),
                    jnp.flip(super_ltz[:-1], axis=0),
                ),
            )
            x_ltz = jnp.concatenate([jnp.flip(x_rev, axis=0), y_ltz[-1][None, ...]], axis=0)
        else:
            x_ltz = y_ltz
        sol_tzl = jnp.transpose(x_ltz, (1, 2, 0, 3)).reshape((n_theta, n_zeta, n_l, n_species, n_x))
        sol_f = jnp.transpose(sol_tzl, (3, 4, 2, 0, 1))
        z_f = sol_f.astype(jnp.float64)
        tail = r_full[op.f_size :]
        return jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced



