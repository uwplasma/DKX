"""Kinetic collision and point-block RHSMode=1 preconditioners."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioning import (
    _LowRankXBlockPrecondCache,
    _RHSMODE1_DIAG_PRECOND_CACHE,
    _RHSMODE1_PRECOND_CACHE,
    _RHSMODE1_PRECOND_DIAGX_CACHE,
    _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE,
    _RHSMODE1_SXBLOCK_PRECOND_CACHE,
    _RHSMODE1_XBLOCK_PRECOND_CACHE,
    _RHSMode1PrecondCache,
    _RHSMode1PrecondDiagXCache,
    _TransportPrecondCache,
    _TransportXBlockPrecondCache,
)
from sfincs_jax.solvers.preconditioning import precond_dtype as _precond_dtype
from sfincs_jax.solvers.preconditioning import _build_rhsmode1_preconditioner_operator_point
from sfincs_jax.solvers.preconditioning import (
    matvec_submatrix_v3_unsharded as _matvec_submatrix,
    precond_chunk_cols as _precond_chunk_cols,
    rhs_mode1_precond_cache_key,
    transport_precond_cache_key,
)
from sfincs_jax.operators.profile_response.system import V3FullSystemOperator
from .species_blocks import build_rhs1_species_xblock_preconditioner

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = (
    "build_rhs1_block_preconditioner",
    "build_rhs1_block_preconditioner_xdiag",
    "build_rhs1_collision_preconditioner",
)


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=_precond_dtype())


def _transport_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return transport_precond_cache_key(op, kind, precond_dtype=_precond_dtype())


def build_rhs1_collision_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Cheap collision-based preconditioner for RHSMode=1 solves (BiCGStab-friendly)."""
    use_xblock = False
    use_sxblock = False
    precond_dtype = _precond_dtype()
    kind_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "").strip().lower()
    if kind_env in {"xblock", "block_x", "x"}:
        use_xblock = True
    elif kind_env in {"sxblock", "species_block", "block"}:
        use_sxblock = True
    elif kind_env in {"", "auto"} and op.fblock.fp is not None:
        f_shape = op.fblock.f_shape
        n_species, n_x, _, _, _ = f_shape
        n_block = int(n_species) * int(n_x)
        sxblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX", "").strip()
        try:
            sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
        except ValueError:
            sxblock_max = 64
        xblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX", "").strip()
        try:
            xblock_max = int(xblock_max_env) if xblock_max_env else 256
        except ValueError:
            xblock_max = 256
        if sxblock_max >= 0 and n_block <= sxblock_max:
            use_sxblock = True
        elif xblock_max >= 0 and int(n_x) <= xblock_max:
            use_xblock = True

    if use_sxblock and op.fblock.fp is not None:
        low_rank_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K", "").strip()
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

        if low_rank_k > 0:
            rank_k = min(int(low_rank_k), int(n_block))
            cache_key = _transport_precond_cache_key(op, f"rhs1_collision_sxblock_lr_{rank_k}")
            cached_lr = _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE.get(cache_key)
            if cached_lr is None:
                mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
                reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
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

                nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                d_inv = np.zeros((n_l, n_block), dtype=np.float64)
                d_inv_u = np.zeros((n_l, n_block, rank_k), dtype=np.float64)
                v_lr = np.zeros((n_l, rank_k, n_block), dtype=np.float64)
                m_inv = np.zeros((n_l, rank_k, rank_k), dtype=np.float64)

                for ell in range(n_l):
                    a_fp = np.array(mat[:, :, ell, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                    a_fp = a_fp.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                    diag = np.full((n_block,), identity_shift + reg, dtype=np.float64)
                    if pas_diag is not None:
                        diag += pas_diag[:, :, ell].reshape((n_block,))
                    inactive_x = np.where(nxi_for_x <= ell)[0]
                    if inactive_x.size:
                        for ix in inactive_x:
                            for s in range(n_species):
                                idx = s * n_x + int(ix)
                                a_fp[idx, :] = 0.0
                                a_fp[:, idx] = 0.0
                                diag[idx] = 1.0
                    d_inv_l = 1.0 / diag
                    d_inv[ell, :] = d_inv_l
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
                            d_inv_u[ell, :, :k_use] = d_inv_u_l
                            v_lr[ell, :k_use, :] = v_lr_l
                            m_inv[ell, :k_use, :k_use] = m_inv_l

                cached_lr = _LowRankXBlockPrecondCache(
                    d_inv=jnp.asarray(d_inv, dtype=precond_dtype),
                    d_inv_u=jnp.asarray(d_inv_u, dtype=precond_dtype),
                    v=jnp.asarray(v_lr, dtype=precond_dtype),
                    m_inv=jnp.asarray(m_inv, dtype=precond_dtype),
                )
                _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE[cache_key] = cached_lr

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
        else:
            cache_key = _transport_precond_cache_key(op, "rhs1_collision_sxblock")
            cached = _RHSMODE1_SXBLOCK_PRECOND_CACHE.get(cache_key)
            if cached is None:
                f_shape = op.fblock.f_shape
                n_species, n_x, n_l, _, _ = f_shape
                n_block = n_species * n_x
                inv_block = np.zeros((n_l, n_block, n_block), dtype=np.float64)
                mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
                reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
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

                for ell in range(n_l):
                    a = np.array(mat[:, :, ell, :, :], dtype=np.float64, copy=True)  # (S,S,X,X)
                    a = a.transpose(0, 2, 1, 3).reshape((n_block, n_block))
                    if identity_shift != 0.0:
                        a[np.arange(n_block), np.arange(n_block)] += identity_shift
                    if pas_diag is not None:
                        diag_add = pas_diag[:, :, ell].reshape((n_block,))
                        a[np.arange(n_block), np.arange(n_block)] += diag_add
                    if reg != 0.0:
                        a[np.arange(n_block), np.arange(n_block)] += reg
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_block[ell, :, :] = inv

                cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_block, dtype=precond_dtype))
                _RHSMODE1_SXBLOCK_PRECOND_CACHE[cache_key] = cached

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

    elif use_xblock and op.fblock.fp is not None:
        cache_key = _transport_precond_cache_key(op, "rhs1_collision_xblock")
        cached = _RHSMODE1_XBLOCK_PRECOND_CACHE.get(cache_key)
        if cached is None:
            f_shape = op.fblock.f_shape
            n_species, n_x, n_l, _, _ = f_shape
            inv_xblock = np.zeros((n_species, n_l, n_x, n_x), dtype=np.float64)
            mat = np.asarray(op.fblock.fp.mat, dtype=np.float64)  # (S,S,L,X,X)
            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
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

            for s in range(n_species):
                for ell in range(n_l):
                    a = np.array(mat[s, s, ell, :, :], dtype=np.float64, copy=True)
                    if identity_shift != 0.0:
                        a[np.arange(n_x), np.arange(n_x)] += identity_shift
                    if pas_diag is not None:
                        a[np.arange(n_x), np.arange(n_x)] += pas_diag[s, :, ell]
                    if reg != 0.0:
                        a[np.arange(n_x), np.arange(n_x)] += reg
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a, rcond=1e-12)
                    inv_xblock[s, ell, :, :] = inv

            cached = _TransportXBlockPrecondCache(inv_xblock=jnp.asarray(inv_xblock, dtype=precond_dtype))
            _RHSMODE1_XBLOCK_PRECOND_CACHE[cache_key] = cached

        inv_xblock = cached.inv_xblock

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=precond_dtype)
            f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
            f_sl = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
            z_sl = jnp.einsum("slij,sljtz->slitz", inv_xblock, f_sl)
            z_f = jnp.transpose(z_sl, (0, 2, 1, 3, 4))  # (S,X,L,T,Z)
            tail = r_full[op.f_size :]
            z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
            return jnp.asarray(z_full, dtype=jnp.float64)
    else:
        cache_key = _transport_precond_cache_key(op, "rhs1_collision_diag")
        cached = _RHSMODE1_DIAG_PRECOND_CACHE.get(cache_key)
        if cached is None:
            f_shape = op.fblock.f_shape
            n_species, n_x, n_l, _, _ = f_shape
            diag = jnp.zeros(f_shape, dtype=jnp.float64)

            if float(op.fblock.identity_shift) != 0.0:
                diag = diag + jnp.asarray(op.fblock.identity_shift, dtype=jnp.float64)

            if op.fblock.pas is not None:
                pas = op.fblock.pas
                ell = jnp.arange(n_l, dtype=jnp.float64)
                factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * pas.krook)
                pas_diag = pas.nu_n * pas.nu_d_hat[:, :, None] * factor_l[None, None, :]
                diag = diag + pas_diag[:, :, :, None, None]

            if op.fblock.fp is not None:
                mat = op.fblock.fp.mat  # (S,S,L,X,X)
                diag_x = jnp.diagonal(mat, axis1=3, axis2=4)  # (S,S,L,X)
                diag_self = jnp.diagonal(diag_x, axis1=0, axis2=1)  # (L,X,S)
                diag_self = jnp.transpose(diag_self, (2, 1, 0))  # (S,X,L)
                diag = diag + diag_self[:, :, :, None, None]

            nxi_for_x = op.fblock.collisionless.n_xi_for_x.astype(jnp.int32)
            mask = jnp.arange(n_l, dtype=jnp.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
            mask = mask[None, :, :, None, None]
            diag = jnp.where(mask, diag, jnp.asarray(1.0, dtype=jnp.float64))

            reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "").strip()
            try:
                reg = float(reg_env) if reg_env else 1e-10
            except ValueError:
                reg = 1e-10
            reg_l0 = float(reg)
            if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
                # PAS+Er runs can have an exactly-zero L=0 diagonal (krook=0), so a tiny
                # regularization produces enormous inverse diagonals and can destabilize
                # left-preconditioned Krylov residual norms.
                reg_l0 = max(reg_l0, 1.0)
            reg_by_l = jnp.full((n_l,), jnp.asarray(float(reg), dtype=jnp.float64))
            if n_l > 0:
                reg_by_l = reg_by_l.at[0].set(jnp.asarray(reg_l0, dtype=jnp.float64))
            inv_diag_f = 1.0 / (diag + reg_by_l[None, None, :, None, None])
            cached = _TransportPrecondCache(inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype))
            _RHSMODE1_DIAG_PRECOND_CACHE[cache_key] = cached

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


def build_rhs1_block_preconditioner_xdiag(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    preconditioner_xi: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Block preconditioner with x-diagonal blocks (per species, per x).

    This matches v3's `preconditioner_x=1` behavior by dropping x-couplings while
    retaining the requested xi coupling.
    """
    cache_key = _rhsmode1_precond_cache_key(op, f"point_xdiag_xi{int(preconditioner_xi)}")
    cached = _RHSMODE1_PRECOND_DIAGX_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
        precond_dtype = _precond_dtype(max_l * max_l)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

        block_inv_list: list[list[jnp.ndarray]] = []
        idx_map_list: list[list[jnp.ndarray]] = []
        for s in range(n_s):
            inv_row: list[jnp.ndarray] = []
            idx_row: list[jnp.ndarray] = []
            for ix in range(n_x):
                max_lx = int(nxi_for_x[ix])
                if max_lx <= 0:
                    inv_row.append(jnp.zeros((0, 0), dtype=precond_dtype))
                    idx_row.append(jnp.zeros((n_t, n_z, 0), dtype=jnp.int32))
                    continue
                rep_idx = np.zeros((max_lx,), dtype=np.int32)
                for il in range(max_lx):
                    rep_idx[il] = int(((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0))
                chunk_cols = _precond_chunk_cols(total, int(rep_idx.shape[0]))
                y_sub = _matvec_submatrix(
                    op_pc,
                    col_idx=rep_idx,
                    row_idx=rep_idx,
                    total_size=total,
                    chunk_cols=chunk_cols,
                )
                a = np.asarray(y_sub.T, dtype=np.float64)
                if preconditioner_xi != 0:
                    a = np.diag(np.diag(a))
                a = a + reg * np.eye(max_lx, dtype=np.float64)
                try:
                    inv = np.linalg.inv(a)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(a, rcond=1e-12)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=1e-12)
                inv_row.append(jnp.asarray(inv, dtype=precond_dtype))
                idx_map = np.zeros((n_t, n_z, max_lx), dtype=np.int32)
                for it in range(n_t):
                    for iz in range(n_z):
                        for il in range(max_lx):
                            idx_map[it, iz, il] = int(
                                ((((s * n_x + ix) * n_l + il) * n_t + it) * n_z + iz)
                            )
                idx_row.append(jnp.asarray(idx_map, dtype=jnp.int32))
            block_inv_list.append(inv_row)
            idx_map_list.append(idx_row)

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
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondDiagXCache(
            block_inv_list=tuple(tuple(row) for row in block_inv_list),
            idx_map_list=tuple(tuple(row) for row in idx_map_list),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_DIAGX_CACHE[cache_key] = cached

    block_inv_list = cached.block_inv_list
    idx_map_list = cached.idx_map_list
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp
    n_s = int(op.n_species)
    n_x = int(op.n_x)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=block_inv_list[0][0].dtype if n_s > 0 and n_x > 0 else _precond_dtype())
        z_full = jnp.zeros_like(r_full)
        for s in range(n_s):
            for ix in range(n_x):
                inv = block_inv_list[s][ix]
                idx_map = idx_map_list[s][ix]
                if idx_map.size == 0:
                    continue
                r_loc = r_full[idx_map].reshape((int(op.n_theta), int(op.n_zeta), int(inv.shape[0])))
                z_loc = jnp.einsum("ij,tzj->tzi", inv, r_loc)
                z_full = z_full.at[idx_map].set(z_loc, unique_indices=True)
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


def build_rhs1_block_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    preconditioner_species: int = 1,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build a PETSc-like block preconditioner for RHSMode=1 solves.

    Structure:
    - x/L local block solve per species at each (theta,zeta), using a representative
      per-species block matrix from a simplified operator.
    - explicit extra/source-row solve via a dense small block.
    """
    if int(preconditioner_species) == 0:
        return build_rhs1_species_xblock_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if int(preconditioner_x) == 1:
        return build_rhs1_block_preconditioner_xdiag(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced, preconditioner_xi=preconditioner_xi
        )
    cache_key = _rhsmode1_precond_cache_key(op, "point")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PRECOND_CACHE.get(cache_key)
    if cached is None:
        op_pc = _build_rhsmode1_preconditioner_operator_point(op)
        n_s = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_t = int(op.n_theta)
        n_z = int(op.n_zeta)
        total = int(op.total_size)

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        local_per_species = int(np.sum(nxi_for_x))

        # Representative local x/L blocks at (theta,zeta)=(0,0), one per species.
        rep_indices_by_species: list[np.ndarray] = []
        for s in range(n_s):
            idx: list[int] = []
            for ix in range(n_x):
                max_l = int(nxi_for_x[ix])
                for il in range(max_l):
                    f_idx = ((((s * n_x + ix) * n_l + il) * n_t + 0) * n_z + 0)
                    idx.append(int(f_idx))
            rep_indices_by_species.append(np.asarray(idx, dtype=np.int32))

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        reg = np.float64(reg_val)

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

        # Build per-(s,theta,zeta) gather map for active x/L rows.
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
            ee = np.asarray(y_sub.T, dtype=np.float64)
            ee = ee + reg * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=precond_dtype)

        cached = _RHSMode1PrecondCache(
            idx_map_jnp=idx_map_jnp,
            flat_idx_jnp=flat_idx_jnp,
            block_inv_jnp=block_inv_jnp,
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_CACHE[cache_key] = cached

    n_s = int(op.n_species)
    n_t = int(op.n_theta)
    n_z = int(op.n_zeta)
    local_per_species = int(cached.block_inv_jnp.shape[-1])
    flat_idx_jnp = cached.flat_idx_jnp
    block_inv_jnp = cached.block_inv_jnp
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

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
