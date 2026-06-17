"""Radial x-grid RHSMode=1 preconditioners."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from ....preconditioner_caches import (
    _RHSMODE1_XMG_PRECOND_CACHE,
    _RHSMODE1_XUPWIND_PRECOND_CACHE,
    _TransportXmgPrecondCache,
    _XUpwindPrecondCache,
)
from ....preconditioner_context import precond_dtype as _precond_dtype
from ....preconditioner_setup import rhs_mode1_precond_cache_key
from ....v3_system import V3FullSystemOperator

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = (
    "build_rhs1_xmg_preconditioner",
    "build_rhs1_xupwind_preconditioner",
)


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=_precond_dtype())


def build_rhs1_xmg_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    stride_override: int | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Two-level additive x-grid preconditioner for RHSMode=1 collision operators.

    Uses the same coarse-x correction as the transport x-multigrid preconditioner,
    but applies it to the RHSMode=1 operator (including PAS/FP diagonals).
    """
    # For PAS+Er systems, inverting dense x-blocks derived from ddx matrices can be
    # extremely ill-conditioned. Use a stable x-upwind solve instead.
    if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
        return build_rhs1_xupwind_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    stride_env = os.environ.get("SFINCS_JAX_RHSMODE1_XMG_STRIDE", "").strip()
    if not stride_env:
        stride_env = os.environ.get("SFINCS_JAX_XMG_STRIDE", "").strip()
    try:
        if stride_override is not None:
            stride = int(stride_override)
        elif stride_env:
            stride = int(stride_env)
        else:
            # Large FP systems benefit from a slightly coarser x-grid in the default
            # xmg preconditioner: it improves conditioning of the coarse correction and
            # lowers setup/apply cost enough to keep large CPU runs within practical
            # wall-clock limits.
            if (
                op.fblock.fp is not None
                and op.fblock.pas is None
                and int(op.total_size) >= 120000
            ):
                stride = 4
            else:
                stride = 2
    except ValueError:
        stride = 2
    stride = max(1, stride)
    cache_key = _rhsmode1_precond_cache_key(op, f"xmg_{stride}")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_XMG_PRECOND_CACHE.get(cache_key)
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

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        mask = np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]  # (X,L)
        mask = mask[None, :, :, None, None]
        diag = np.where(mask, diag, 1.0)

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        reg_l0 = float(reg)
        if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
            # The PAS operator can have an exactly-zero L=0 diagonal (krook=0). With Er xDot enabled,
            # the x-coupled block can be near-singular if we only add an extremely small regularization,
            # leading to huge preconditioner entries and GMRES blow-up. Use a modest L=0 shift here to
            # keep the xmg inverse numerically stable.
            reg_l0 = max(reg_l0, 1.0)
        reg_by_l = np.full((n_l,), float(reg), dtype=np.float64)
        if n_l > 0:
            reg_by_l[0] = reg_l0
        inv_diag_f = 1.0 / (diag + reg_by_l[None, None, :, None, None])

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

        # Optional dense-x coupling from the v3 Er xDot term.
        #
        # For PAS-only systems, the collision operator is diagonal in x, so the x-multigrid
        # preconditioner is otherwise ineffective. The Er xDot term introduces dense ddx
        # coupling in x (see `apply_er_xdot_v3`), so we include a flux-surface-averaged
        # diagonal-in-L approximation here to recover useful x-coupling at low cost.
        xdot_x_part: np.ndarray | None = None  # (Xc,Xc)
        xdot_coef_l: np.ndarray | None = None  # (L,)
        xdot_rms = 0.0
        if op.fblock.er_xdot is not None:
            try:
                er = op.fblock.er_xdot
                alpha = float(np.asarray(er.alpha, dtype=np.float64).reshape(()))
                delta = float(np.asarray(er.delta, dtype=np.float64).reshape(()))
                dphi = float(np.asarray(er.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
                d_hat = np.asarray(er.d_hat, dtype=np.float64)  # (T,Z)
                b_hat = np.asarray(er.b_hat, dtype=np.float64)  # (T,Z)
                b_sub_theta = np.asarray(er.b_hat_sub_theta, dtype=np.float64)  # (T,Z)
                b_sub_zeta = np.asarray(er.b_hat_sub_zeta, dtype=np.float64)  # (T,Z)
                db_dtheta = np.asarray(er.db_hat_dtheta, dtype=np.float64)  # (T,Z)
                db_dzeta = np.asarray(er.db_hat_dzeta, dtype=np.float64)  # (T,Z)

                factor0 = -(alpha * delta * dphi) / 4.0  # adjointFactor=1
                xdot_factor = factor0 * d_hat / (b_hat**3) * (b_sub_theta * db_dzeta - b_sub_zeta * db_dtheta)  # (T,Z)

                theta_w = np.asarray(op.theta_weights, dtype=np.float64)
                zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
                d_hat_full = np.asarray(op.d_hat, dtype=np.float64)
                fs_factor = (theta_w[:, None] * zeta_w[None, :]) / d_hat_full  # (T,Z), un-normalized
                # Signed flux-surface averages of xdot_factor can cancel (common for tokamak-like
                # geometries), which would incorrectly eliminate dense-x coupling from the
                # preconditioner. Use an RMS magnitude instead to capture the typical scale.
                fs_sum = float(np.sum(fs_factor))
                if fs_sum > 0.0:
                    xdot_rms = float(np.sqrt(np.sum(fs_factor * (xdot_factor * xdot_factor)) / fs_sum))
                else:
                    xdot_rms = 0.0

                l_vec = np.arange(n_l, dtype=np.float64)
                denom = (2.0 * l_vec + 3.0) * (2.0 * l_vec - 1.0)
                diag_coef = 2.0 * (3.0 * l_vec * l_vec + 3.0 * l_vec - 2.0) / denom
                # force0RadialCurrentInEquilibrium=.true. (v3 default) -> xdotFactor2 = 0.
                xdot_coef_l = diag_coef * xdot_rms

                x_arr = np.asarray(er.x, dtype=np.float64)
                ddx = np.asarray(er.ddx_plus, dtype=np.float64)
                x_part = x_arr[:, None] * ddx  # (X,X)
                xdot_x_part = x_part[np.ix_(coarse_idx, coarse_idx)]
            except Exception:  # noqa: BLE001
                xdot_x_part = None
                xdot_coef_l = None
                xdot_rms = 0.0

        # If PAS+Er includes ΔL=±2 xDot coupling, build a small coupled (L,x) coarse inverse
        # for low Legendre modes. This is much stronger than per-L coarse inverses, while still
        # cheap: we only couple a few low-L modes and only on the coarse x grid.
        coarse_inv_lblock: np.ndarray | None = None  # (S, Lb*Xc, Lb*Xc)
        lblock = 0
        pinv_env = os.environ.get("SFINCS_JAX_RHSMODE1_XMG_PINV_RCOND", "").strip()
        try:
            pinv_rcond = float(pinv_env) if pinv_env else 1e-12
        except ValueError:
            pinv_rcond = 1e-12
        if (
            xdot_x_part is not None
            and xdot_coef_l is not None
            and float(xdot_rms) != 0.0
            and op.fblock.pas is not None
            and mat_fp is None
            and n_l >= 3
        ):
            # Default to a small block that captures the first few ΔL=±2 couplings.
            # L is small (Nxi is usually O(10)), so a block of size 6 is still cheap.
            lblock = int(min(int(n_l), 6))
            n_block = int(lblock * n_coarse)
            coarse_inv_lblock = np.zeros((n_species, n_block, n_block), dtype=np.float64)

            # Coefficients for ΔL=±2 couplings in `apply_er_xdot_v3`:
            #   row L gets col L+2 with coef sup2[L]
            #   row L gets col L-2 with coef sub2[L]
            l_vec = np.arange(n_l, dtype=np.float64)
            sup2 = np.zeros((n_l,), dtype=np.float64)
            sub2 = np.zeros((n_l,), dtype=np.float64)
            if n_l >= 3:
                l0 = l_vec[:-2]
                sup2[:-2] = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
                l2 = l_vec[2:]
                sub2[2:] = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))

            for s in range(n_species):
                a = np.zeros((n_block, n_block), dtype=np.float64)
                # Base diagonal blocks (PAS + identity shift + regularization).
                for ell in range(lblock):
                    i0 = int(ell * n_coarse)
                    i1 = i0 + n_coarse
                    block = np.zeros((n_coarse, n_coarse), dtype=np.float64)
                    reg_eff = reg_l0 if ell == 0 else float(reg)
                    diag_vec = np.full((n_coarse,), identity_shift + reg_eff, dtype=np.float64)
                    if pas_diag is not None:
                        diag_vec = diag_vec + pas_diag[s, coarse_idx, ell]
                    block[np.arange(n_coarse), np.arange(n_coarse)] += diag_vec
                    a[i0:i1, i0:i1] += block

                # Add xDot dense-x coupling (diagonal + ΔL=±2 couplings) using an RMS magnitude.
                for ell in range(lblock):
                    i0 = int(ell * n_coarse)
                    i1 = i0 + n_coarse
                    a[i0:i1, i0:i1] += float(xdot_coef_l[ell]) * xdot_x_part
                    if ell + 2 < lblock:
                        j0 = int((ell + 2) * n_coarse)
                        j1 = j0 + n_coarse
                        a[i0:i1, j0:j1] += float(sup2[ell] * xdot_rms) * xdot_x_part
                    if ell - 2 >= 0:
                        j0 = int((ell - 2) * n_coarse)
                        j1 = j0 + n_coarse
                        a[i0:i1, j0:j1] += float(sub2[ell] * xdot_rms) * xdot_x_part

                # Identity rows/cols for inactive (x,L) combinations.
                for ell in range(lblock):
                    inactive_x = np.where(nxi_for_x <= ell)[0]
                    if not inactive_x.size:
                        continue
                    for ix in inactive_x:
                        idx = coarse_map.get(int(ix))
                        if idx is None:
                            continue
                        p = int(ell * n_coarse + idx)
                        a[p, :] = 0.0
                        a[:, p] = 0.0
                        a[p, p] = 1.0

                inv = np.linalg.pinv(a, rcond=pinv_rcond)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=pinv_rcond)
                coarse_inv_lblock[s, :, :] = inv

        for s in range(n_species):
            for ell in range(n_l):
                if mat_fp is None:
                    a = np.zeros((n_x, n_x), dtype=np.float64)
                else:
                    a = np.array(mat_fp[s, s, ell, :, :], dtype=np.float64, copy=True)
                a = a[np.ix_(coarse_idx, coarse_idx)]
                reg_eff = reg_l0 if ell == 0 else float(reg)
                diag_vec = np.full((n_coarse,), identity_shift + reg_eff, dtype=np.float64)
                if pas_diag is not None:
                    diag_vec += pas_diag[s, coarse_idx, ell]
                a[np.arange(n_coarse), np.arange(n_coarse)] += diag_vec
                if xdot_x_part is not None and xdot_coef_l is not None:
                    a = a + float(xdot_coef_l[ell]) * xdot_x_part

                inactive_x = np.where(nxi_for_x <= ell)[0]
                if inactive_x.size:
                    for ix in inactive_x:
                        idx = coarse_map.get(int(ix))
                        if idx is None:
                            continue
                        a[idx, :] = 0.0
                        a[:, idx] = 0.0
                        a[idx, idx] = 1.0

                use_pinv = bool(xdot_x_part is not None and xdot_coef_l is not None and ell == 0 and mat_fp is None)
                if use_pinv:
                    inv = np.linalg.pinv(a, rcond=pinv_rcond)
                else:
                    try:
                        inv = np.linalg.inv(a)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a, rcond=pinv_rcond)
                if not np.all(np.isfinite(inv)):
                    inv = np.linalg.pinv(a, rcond=pinv_rcond)
                coarse_inv[s, ell, :, :] = inv

        cached = _TransportXmgPrecondCache(
            inv_diag_f=jnp.asarray(inv_diag_f, dtype=precond_dtype),
            coarse_inv=jnp.asarray(coarse_inv, dtype=precond_dtype),
            coarse_idx=jnp.asarray(coarse_idx, dtype=jnp.int32),
            coarse_inv_lblock=(
                None if coarse_inv_lblock is None else jnp.asarray(coarse_inv_lblock, dtype=precond_dtype)
            ),
            lblock=int(lblock),
        )
        _RHSMODE1_XMG_PRECOND_CACHE[cache_key] = cached

    inv_diag_f = cached.inv_diag_f
    coarse_inv = cached.coarse_inv
    coarse_idx = cached.coarse_idx
    coarse_inv_lblock = cached.coarse_inv_lblock
    lblock = int(cached.lblock)

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        z_f = f * inv_diag_f
        f_sl = jnp.transpose(f, (0, 2, 1, 3, 4))  # (S,L,X,T,Z)
        corr_sl = jnp.zeros_like(f_sl)
        if coarse_inv_lblock is not None and lblock > 0:
            n_coarse = int(coarse_idx.shape[0])
            lblock_use = int(min(lblock, int(f_sl.shape[1])))
            f_block = f_sl[:, :lblock_use, coarse_idx, :, :].reshape((f_sl.shape[0], lblock_use * n_coarse) + f_sl.shape[3:])
            z_block = jnp.einsum("sij,sjtz->sitz", coarse_inv_lblock, f_block)
            z_block = z_block.reshape((f_sl.shape[0], lblock_use, n_coarse) + f_sl.shape[3:])
            corr_sl = corr_sl.at[:, :lblock_use, coarse_idx, :, :].set(z_block, unique_indices=True)
            if lblock_use < int(f_sl.shape[1]):
                f_high = f_sl[:, lblock_use:, coarse_idx, :, :]
                z_high = jnp.einsum("slij,sljtz->slitz", coarse_inv[:, lblock_use:, :, :], f_high)
                corr_sl = corr_sl.at[:, lblock_use:, coarse_idx, :, :].set(z_high, unique_indices=True)
        else:
            f_coarse = f_sl[:, :, coarse_idx, :, :]
            z_coarse = jnp.einsum("slij,sljtz->slitz", coarse_inv, f_coarse)
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


def build_rhs1_xupwind_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """x-upwind preconditioner for PAS+Er RHSMode=1 systems.

    Motivation
    ----------
    The v3 Er xDot operator applies a dense ddx matvec that is effectively a first-order
    advection operator in x. On realistic x grids, explicitly inverting dense x blocks
    (as done by the xmg coarse correction) can be extremely ill-conditioned and produce
    enormous preconditioner factors.

    This preconditioner instead approximates the xDot coupling using a *bidiagonal*
    first-order upwind operator on the x grid, yielding a stable forward-substitution
    solve along x for each (species, L, theta, zeta) slice.

    Notes
    -----
    - Designed for PAS-only systems with Er xDot enabled (FP absent).
    - Uses a flux-surface RMS magnitude of xdotFactor to obtain a single scalar strength,
      and ignores the θ/ζ sign variation and ΔL=±2 couplings (handled elsewhere).
    - Fully JAX-native and differentiable; safe under `custom_linear_solve`.
    """
    if op.fblock.fp is not None or op.fblock.er_xdot is None:
        return build_rhs1_xmg_preconditioner(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    precond_dtype = _precond_dtype()
    lblock_env = os.environ.get("SFINCS_JAX_RHSMODE1_XUPWIND_LBLOCK", "").strip()
    try:
        lblock_req = int(lblock_env) if lblock_env else 8
    except ValueError:
        lblock_req = 8
    lblock_req = max(0, int(lblock_req))
    cache_key = _rhsmode1_precond_cache_key(op, f"xupwind_lb{lblock_req}")
    cached = _RHSMODE1_XUPWIND_PRECOND_CACHE.get(cache_key)
    if cached is None:
        f_shape = op.fblock.f_shape
        n_species, n_x, n_l, _, _ = f_shape

        diag_base = np.zeros((n_species, n_x, n_l), dtype=np.float64)
        if float(op.fblock.identity_shift) != 0.0:
            diag_base = diag_base + float(op.fblock.identity_shift)

        if op.fblock.pas is not None:
            pas = op.fblock.pas
            l_arr = np.arange(n_l, dtype=np.float64)
            factor_l = 0.5 * (l_arr * (l_arr + 1.0) + 2.0 * float(pas.krook))
            pas_diag = float(pas.nu_n) * np.asarray(pas.nu_d_hat, dtype=np.float64)[:, :, None] * factor_l[None, None, :]
            diag_base = diag_base + pas_diag

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-10
        except ValueError:
            reg = 1e-10
        reg_l0 = float(reg)
        if op.fblock.pas is not None and op.fblock.fp is None and op.fblock.er_xdot is not None:
            reg_l0 = max(reg_l0, 1.0)
        reg_by_l = np.full((n_l,), float(reg), dtype=np.float64)
        if n_l > 0:
            reg_by_l[0] = reg_l0
        diag_base = diag_base + reg_by_l[None, None, :]

        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        active = (np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None])  # (X,L)
        for ix in range(n_x):
            for ell in range(n_l):
                if not bool(active[ix, ell]):
                    diag_base[:, ix, ell] = 1.0

        # Flux-surface RMS magnitude of xdotFactor.
        xdot_rms = 0.0
        try:
            er = op.fblock.er_xdot
            alpha = float(np.asarray(er.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(er.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(er.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            d_hat = np.asarray(er.d_hat, dtype=np.float64)
            b_hat = np.asarray(er.b_hat, dtype=np.float64)
            b_sub_theta = np.asarray(er.b_hat_sub_theta, dtype=np.float64)
            b_sub_zeta = np.asarray(er.b_hat_sub_zeta, dtype=np.float64)
            db_dtheta = np.asarray(er.db_hat_dtheta, dtype=np.float64)
            db_dzeta = np.asarray(er.db_hat_dzeta, dtype=np.float64)
            factor0 = -(alpha * delta * dphi) / 4.0
            xdot_factor = factor0 * d_hat / (b_hat**3) * (b_sub_theta * db_dzeta - b_sub_zeta * db_dtheta)
            theta_w = np.asarray(op.theta_weights, dtype=np.float64)
            zeta_w = np.asarray(op.zeta_weights, dtype=np.float64)
            d_hat_full = np.asarray(op.d_hat, dtype=np.float64)
            fs_factor = (theta_w[:, None] * zeta_w[None, :]) / d_hat_full
            fs_sum = float(np.sum(fs_factor))
            if fs_sum > 0.0:
                xdot_rms = float(np.sqrt(np.sum(fs_factor * (xdot_factor * xdot_factor)) / fs_sum))
        except Exception:  # noqa: BLE001
            xdot_rms = 0.0

        l_vec = np.arange(n_l, dtype=np.float64)
        denom = (2.0 * l_vec + 3.0) * (2.0 * l_vec - 1.0)
        diag_coef = 2.0 * (3.0 * l_vec * l_vec + 3.0 * l_vec - 2.0) / denom
        c_l = diag_coef * float(xdot_rms)  # (L,)
        sup2 = np.zeros((n_l,), dtype=np.float64)
        sub2 = np.zeros((n_l,), dtype=np.float64)
        if n_l >= 3:
            l0 = l_vec[:-2]
            sup2[:-2] = (l0 + 1.0) * (l0 + 2.0) / ((2.0 * l0 + 5.0) * (2.0 * l0 + 3.0))
            l2 = l_vec[2:]
            sub2[2:] = l2 * (l2 - 1.0) / ((2.0 * l2 - 3.0) * (2.0 * l2 - 1.0))

        x_arr = np.asarray(op.fblock.er_xdot.x, dtype=np.float64)
        dx = np.diff(x_arr)
        x_over_dx = np.zeros((n_x,), dtype=np.float64)
        if n_x >= 2:
            x_over_dx[1:] = x_arr[1:] / np.where(dx != 0, dx, 1.0)

        # Lower-bidiagonal coefficients for (diag + c_l * x * d/dx_upwind).
        sub = -(x_over_dx[:, None] * c_l[None, :])  # (X,L), sub[0]=0
        diag = diag_base + (x_over_dx[:, None] * c_l[None, :])  # (S,X,L)
        sub = np.broadcast_to(sub[None, :, :], (n_species, n_x, n_l)).copy()

        # Inactive (x,L): identity.
        for ix in range(n_x):
            for ell in range(n_l):
                if not bool(active[ix, ell]):
                    sub[:, ix, ell] = 0.0

        lblock = int(min(max(0, lblock_req), n_l))
        block_inv: np.ndarray | None = None
        block_sub: np.ndarray | None = None
        if lblock >= 3 and float(xdot_rms) != 0.0:
            block_inv = np.zeros((n_species, n_x, lblock, lblock), dtype=np.float64)
            block_sub = np.zeros((n_species, n_x, lblock, lblock), dtype=np.float64)
            pinv_env = os.environ.get("SFINCS_JAX_RHSMODE1_XUPWIND_PINV_RCOND", "").strip()
            try:
                pinv_rcond = float(pinv_env) if pinv_env else 1e-12
            except ValueError:
                pinv_rcond = 1e-12
            for s in range(n_species):
                for ix in range(n_x):
                    c_x = float(x_over_dx[ix])
                    a0 = np.zeros((lblock, lblock), dtype=np.float64)
                    a1 = np.zeros((lblock, lblock), dtype=np.float64)
                    for ell in range(lblock):
                        a0[ell, ell] = float(diag_base[s, ix, ell]) + c_x * float(c_l[ell])
                        a1[ell, ell] = -c_x * float(c_l[ell])
                        if ell + 2 < lblock:
                            v = c_x * float(sup2[ell] * xdot_rms)
                            a0[ell, ell + 2] += v
                            a1[ell, ell + 2] -= v
                        if ell - 2 >= 0:
                            v = c_x * float(sub2[ell] * xdot_rms)
                            a0[ell, ell - 2] += v
                            a1[ell, ell - 2] -= v
                    n_active = int(min(max(0, int(nxi_for_x[ix])), lblock))
                    if n_active < lblock:
                        for ell in range(n_active, lblock):
                            a0[ell, :] = 0.0
                            a0[:, ell] = 0.0
                            a0[ell, ell] = 1.0
                            a1[ell, :] = 0.0
                            a1[:, ell] = 0.0
                    try:
                        inv = np.linalg.inv(a0)
                    except np.linalg.LinAlgError:
                        inv = np.linalg.pinv(a0, rcond=pinv_rcond)
                    if not np.all(np.isfinite(inv)):
                        inv = np.linalg.pinv(a0, rcond=pinv_rcond)
                    block_inv[s, ix, :, :] = inv
                    block_sub[s, ix, :, :] = a1

        cached = _XUpwindPrecondCache(
            diag=jnp.asarray(diag, dtype=precond_dtype),
            sub=jnp.asarray(sub, dtype=precond_dtype),
            lblock=int(lblock),
            block_inv=(None if block_inv is None else jnp.asarray(block_inv, dtype=precond_dtype)),
            block_sub=(None if block_sub is None else jnp.asarray(block_sub, dtype=precond_dtype)),
        )
        _RHSMODE1_XUPWIND_PRECOND_CACHE[cache_key] = cached
        alias_key = _rhsmode1_precond_cache_key(op, "xupwind")
        if alias_key not in _RHSMODE1_XUPWIND_PRECOND_CACHE:
            _RHSMODE1_XUPWIND_PRECOND_CACHE[alias_key] = cached

    diag = cached.diag  # (S,X,L)
    sub = cached.sub  # (S,X,L)
    lblock = int(cached.lblock)
    block_inv = cached.block_inv
    block_sub = cached.block_sub

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=precond_dtype)
        f = r_full[: op.f_size].reshape(op.fblock.f_shape)  # (S,X,L,T,Z)
        f_x = jnp.transpose(f, (1, 0, 2, 3, 4))  # (X,S,L,T,Z)
        diag_x = jnp.transpose(diag, (1, 0, 2))  # (X,S,L)
        sub_x = jnp.transpose(sub, (1, 0, 2))  # (X,S,L)
        use_block = bool(block_inv is not None and block_sub is not None and lblock > 0)
        if use_block:
            block_inv_x = jnp.transpose(block_inv, (1, 0, 2, 3))  # (X,S,Lb,Lb)
            block_sub_x = jnp.transpose(block_sub, (1, 0, 2, 3))  # (X,S,Lb,Lb)
        else:
            block_inv_x = jnp.zeros((f_x.shape[0], f_x.shape[1], 0, 0), dtype=precond_dtype)
            block_sub_x = jnp.zeros((f_x.shape[0], f_x.shape[1], 0, 0), dtype=precond_dtype)

        def _step(z_prev, inp):
            rhs_i, diag_i, sub_i, binv_i, bsub_i = inp
            diag_b = diag_i[..., None, None]
            sub_b = sub_i[..., None, None]
            if (not use_block) or lblock <= 0:
                z_i = (rhs_i - sub_b * z_prev) / diag_b
                return z_i, z_i

            rhs_lo = rhs_i[:, :lblock, :, :]
            zprev_lo = z_prev[:, :lblock, :, :]
            rhs_lo = rhs_lo - jnp.einsum("sij,sjtz->sitz", bsub_i, zprev_lo)
            z_lo = jnp.einsum("sij,sjtz->sitz", binv_i, rhs_lo)
            if rhs_i.shape[1] > lblock:
                rhs_hi = rhs_i[:, lblock:, :, :]
                zprev_hi = z_prev[:, lblock:, :, :]
                z_hi = (rhs_hi - sub_b[:, lblock:, :, :] * zprev_hi) / diag_b[:, lblock:, :, :]
                z_i = jnp.concatenate([z_lo, z_hi], axis=1)
            else:
                z_i = z_lo
            return z_i, z_i

        z0 = jnp.zeros_like(f_x[0])
        _, z_x = jax.lax.scan(_step, z0, (f_x, diag_x, sub_x, block_inv_x, block_sub_x))
        z_f = jnp.transpose(z_x, (1, 0, 2, 3, 4))  # (S,X,L,T,Z)
        tail = r_full[op.f_size :]
        z_full = jnp.concatenate([z_f.reshape((-1,)), tail], axis=0)
        return jnp.asarray(z_full, dtype=jnp.float64)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced
