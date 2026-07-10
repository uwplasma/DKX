"""PAS angular RHSMode=1 preconditioners."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioning import (
    _PasTokamakThetaPrecondCache,
    _PasTzPrecondCache,
    _RHSMODE1_PAS_TOKAMAK_THETA_CACHE,
    _RHSMODE1_PAS_TZ_CACHE,
)
from sfincs_jax.solvers.preconditioning import precond_dtype as _precond_dtype
from sfincs_jax.solvers.preconditioning import rhs_mode1_precond_cache_key
from sfincs_jax.solvers.preconditioner_pas_policy import build_pas_tz_memory_fallback
from sfincs_jax.discretization.structured_velocity import factor_block_tridiagonal
from sfincs_jax.operators.profile_system import V3FullSystemOperator

Preconditioner = Callable[[jnp.ndarray], jnp.ndarray]

__all__ = (
    "build_rhs1_pas_tokamak_theta_preconditioner",
    "build_rhs1_pas_tz_preconditioner",
)


def _rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=_precond_dtype())


def build_rhs1_pas_tokamak_theta_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    block_preconditioner_builder: Callable[..., Preconditioner],
    pas_tokamak_theta_applicable: Callable[[V3FullSystemOperator], bool],
) -> Preconditioner:
    """PAS tokamak (Nzeta=1) theta/L block-tridiagonal preconditioner for RHSMode=1.

    This preconditioner targets the stiff PAS-only tokamak branch (no drifts) where:
    - x-coupling is absent (block-diagonal in x),
    - the dominant couplings are along theta (streaming) and in Legendre index L,
    - full theta-line blocks over *all* x are far too large to factor.

    Approach:
    - Build a block-tridiagonal-in-L approximation for each x with blocks of size Ntheta.
    - Precompute block-Thomas factors (approximate inverses of effective diagonal blocks + G factors).
    - Apply in O(Nx * Nxi * Ntheta^2) per Krylov iteration.
    """
    if not pas_tokamak_theta_applicable(op):
        return block_preconditioner_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    n_x = int(op.n_x)
    n_l_total = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_l_build = n_l_total
    lmax_env = os.environ.get("SFINCS_JAX_PAS_TOKAMAK_LMAX", "").strip()
    if lmax_env:
        try:
            n_l_build = max(2, int(lmax_env))
        except ValueError:
            n_l_build = n_l_total
    n_l_build = max(2, min(n_l_total, n_l_build))

    cache_key = _rhsmode1_precond_cache_key(op, f"pas_tokamak_theta_l{int(n_l_build)}")
    precond_dtype = _precond_dtype()
    cached = _RHSMODE1_PAS_TOKAMAK_THETA_CACHE.get(cache_key)
    if cached is None:
        cl = op.fblock.collisionless
        pas = op.fblock.pas
        assert cl is not None
        assert pas is not None

        n_species = int(op.n_species)

        if n_theta <= 1:
            return block_preconditioner_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        if n_l_total < 2:
            # The (L,theta) block-tridiagonal structure requires at least L=0,1.
            return block_preconditioner_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

        x = np.asarray(cl.x, dtype=np.float64)  # (X,)
        ddtheta = np.asarray(cl.ddtheta, dtype=np.float64)  # (T,T)
        b_hat = np.asarray(cl.b_hat, dtype=np.float64)[:, 0]  # (T,)
        b_sup_theta = np.asarray(cl.b_hat_sup_theta, dtype=np.float64)[:, 0]  # (T,)
        b_sup_zeta = np.asarray(cl.b_hat_sup_zeta, dtype=np.float64)[:, 0]  # (T,)
        db_dtheta = np.asarray(cl.db_hat_dtheta, dtype=np.float64)[:, 0]  # (T,)
        db_dzeta = np.asarray(cl.db_hat_dzeta, dtype=np.float64)[:, 0]  # (T,)
        t_hats = np.asarray(cl.t_hats, dtype=np.float64)  # (S,)
        m_hats = np.asarray(cl.m_hats, dtype=np.float64)  # (S,)
        nxi_for_x = np.asarray(cl.n_xi_for_x, dtype=np.int32)  # (X,)

        l_arr = np.arange(n_l_build, dtype=np.float64)
        coef_plus = (l_arr + 1.0) / (2.0 * l_arr + 3.0)
        coef_minus = np.where(l_arr > 0, l_arr / (2.0 * l_arr - 1.0), 0.0)
        coef_mirror_plus = (l_arr + 1.0) * (l_arr + 2.0) / (2.0 * l_arr + 3.0)
        coef_mirror_minus = np.where(l_arr > 1, -l_arr * (l_arr - 1.0) / (2.0 * l_arr - 1.0), 0.0)

        coef_plus_x = x[:, None] * coef_plus[None, :]  # (X,L)
        coef_minus_x = x[:, None] * coef_minus[None, :]  # (X,L)
        coef_mirror_plus_x = x[:, None] * coef_mirror_plus[None, :]  # (X,L)
        coef_mirror_minus_x = x[:, None] * coef_mirror_minus[None, :]  # (X,L)

        mask_active_full = (np.arange(n_l_total, dtype=np.int32)[None, :] < nxi_for_x[:, None]).astype(np.float64)
        mask_active = mask_active_full[:, :n_l_build]  # (X,L_build)
        mask_plus = (mask_active[:, :-1] * mask_active[:, 1:]).astype(np.float64)  # (X,L_build-1)
        mask_minus = mask_plus  # same condition, just shifted in index

        b_stream = np.zeros((n_x, n_l_build), dtype=np.float64)
        b_mirror = np.zeros((n_x, n_l_build), dtype=np.float64)
        c_stream = np.zeros((n_x, n_l_build), dtype=np.float64)
        c_mirror = np.zeros((n_x, n_l_build), dtype=np.float64)
        b_stream[:, :-1] = coef_plus_x[:, :-1] * mask_plus
        b_mirror[:, :-1] = coef_mirror_plus_x[:, :-1] * mask_plus
        c_stream[:, 1:] = coef_minus_x[:, 1:] * mask_minus
        c_mirror[:, 1:] = coef_mirror_minus_x[:, 1:] * mask_minus

        # Preconditioner regularization:
        # For PAS tokamak-like systems, the L=0 diagonal can be exactly 0 (krook=0), while
        # the theta derivative operator has a constant-mode nullspace. A naive block-Thomas
        # factorization that inverts the L=0 diagonal block can therefore be unstable.
        #
        # We keep a *tiny* diagonal regularization to protect against true singularities,
        # and we avoid the large L=0 "lift" heuristic by using a combined (L=0,1) block
        # as the first elimination step (below). This yields a much tighter approximation
        # and typically reduces Krylov iterations dramatically.
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-12
        except ValueError:
            reg = 1e-12
        reg = max(float(reg), 0.0)

        pas_coef = np.asarray(pas.coef, dtype=np.float64)[:, :, :n_l_build]  # (S,X,L_build)
        identity_shift = float(op.fblock.identity_shift)
        diag_vals = pas_coef + identity_shift + reg  # (S,X,L_build)
        # For inactive (x,L), decouple by setting diag=1 and couplings=0 (handled above).
        diag_vals = diag_vals * mask_active[None, :, :] + (1.0 - mask_active)[None, :, :]

        # Geometry-dependent base matrices (per species).
        v_theta = b_sup_theta / b_hat  # (T,)
        sqrt_t_over_m = np.sqrt(t_hats / m_hats)  # (S,)
        v_theta_s = sqrt_t_over_m[:, None] * v_theta[None, :]  # (S,T)
        m_theta = v_theta_s[:, :, None] * ddtheta[None, :, :]  # (S,T,T) row-scaled ddtheta

        mirror_geom = b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta  # (T,)
        mirror_factor = -sqrt_t_over_m[:, None] * mirror_geom[None, :] / (2.0 * (b_hat * b_hat)[None, :])  # (S,T)
        mirror_diag = np.zeros((n_species, n_theta, n_theta), dtype=np.float64)
        for s in range(n_species):
            mirror_diag[s, :, :] = np.diag(mirror_factor[s, :])

        eye_t = np.eye(n_theta, dtype=np.float64)
        # Factorization with a combined (L=0,1) block (size 2*Ntheta) followed by standard
        # block-Thomas elimination for L>=2 (size Ntheta).
        twot = 2 * n_theta
        inv_a01 = np.zeros((n_species, n_x, twot, twot), dtype=np.float64)
        g01 = np.zeros((n_species, n_x, twot, n_theta), dtype=np.float64)
        inv_a = np.zeros((n_species, n_x, max(0, n_l_build - 2), n_theta, n_theta), dtype=np.float64)
        g = np.zeros((n_species, n_x, max(0, n_l_build - 3), n_theta, n_theta), dtype=np.float64)
        structured_tail_env = os.environ.get("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", "").strip().lower()
        # Keep the structured tail available for experiments, but do not enable it by default:
        # on the current shipped tokamak PAS examples, the legacy tail is both faster and lower-RSS.
        structured_tail_enabled = structured_tail_env in {"1", "true", "yes", "on"}
        tail_factors_build: list[list[object | None]] | None = (
            [[None for _ in range(n_x)] for _ in range(n_species)] if structured_tail_enabled and n_l_build > 2 else None
        )

        for s in range(n_species):
            m_s = m_theta[s, :, :]
            mir_s = mirror_diag[s, :, :]
            for ix in range(n_x):
                # Build and factor the combined (L=0,1) block.
                a0 = float(diag_vals[s, ix, 0])
                a1 = float(diag_vals[s, ix, 1])
                a0_blk = a0 * eye_t
                a1_blk = a1 * eye_t
                b0 = float(b_stream[ix, 0]) * m_s + float(b_mirror[ix, 0]) * mir_s
                c1 = float(c_stream[ix, 1]) * m_s + float(c_mirror[ix, 1]) * mir_s
                a01 = np.zeros((twot, twot), dtype=np.float64)
                a01[:n_theta, :n_theta] = a0_blk
                a01[:n_theta, n_theta:] = b0
                a01[n_theta:, :n_theta] = c1
                a01[n_theta:, n_theta:] = a1_blk
                try:
                    inv01 = np.linalg.inv(a01)
                except np.linalg.LinAlgError:
                    inv01 = np.linalg.pinv(a01, rcond=1e-12)
                if not np.all(np.isfinite(inv01)):
                    inv01 = np.linalg.pinv(a01, rcond=1e-12)
                inv_a01[s, ix, :, :] = inv01

                if n_l_build == 2:
                    continue

                # G01 = inv(A01) @ B_tilde, where B_tilde couples f2 into the L=1 equation only.
                b1 = float(b_stream[ix, 1]) * m_s + float(b_mirror[ix, 1]) * mir_s
                b_tilde = np.zeros((twot, n_theta), dtype=np.float64)
                b_tilde[n_theta:, :] = b1
                g01_loc = inv01 @ b_tilde  # (2T,T)
                g01[s, ix, :, :] = g01_loc

                # Eliminate L>=2 using standard block-Thomas with previous G having shape (2T,T) for l=2.
                g_prev = g01_loc
                for ell in range(2, n_l_build):
                    a_diag = float(diag_vals[s, ix, ell])
                    if ell == 2:
                        c2 = float(c_stream[ix, ell]) * m_s + float(c_mirror[ix, ell]) * mir_s
                        c_tilde = np.zeros((n_theta, twot), dtype=np.float64)
                        c_tilde[:, n_theta:] = c2
                        a_eff = a_diag * eye_t - c_tilde @ g_prev
                    else:
                        c_ell = float(c_stream[ix, ell]) * m_s + float(c_mirror[ix, ell]) * mir_s
                        a_eff = a_diag * eye_t - c_ell @ g_prev
                    try:
                        a_inv = np.linalg.inv(a_eff)
                    except np.linalg.LinAlgError:
                        a_inv = np.linalg.pinv(a_eff, rcond=1e-12)
                    if not np.all(np.isfinite(a_inv)):
                        a_inv = np.linalg.pinv(a_eff, rcond=1e-12)
                    inv_a[s, ix, ell - 2, :, :] = a_inv
                    if ell < n_l_build - 1:
                        b_ell = float(b_stream[ix, ell]) * m_s + float(b_mirror[ix, ell]) * mir_s
                        g_ell = a_inv @ b_ell
                        g[s, ix, ell - 2, :, :] = g_ell
                        g_prev = g_ell

                if tail_factors_build is not None:
                    n_tail = n_l_build - 2
                    diag_tail = np.zeros((n_tail, n_theta, n_theta), dtype=np.float64)
                    lower_tail = np.zeros((max(0, n_tail - 1), n_theta, n_theta), dtype=np.float64)
                    upper_tail = np.zeros((max(0, n_tail - 1), n_theta, n_theta), dtype=np.float64)
                    for ell in range(2, n_l_build):
                        diag_tail[ell - 2, :, :] = float(diag_vals[s, ix, ell]) * eye_t
                        if ell < n_l_build - 1:
                            upper_tail[ell - 2, :, :] = float(b_stream[ix, ell]) * m_s + float(b_mirror[ix, ell]) * mir_s
                        if ell > 2:
                            lower_tail[ell - 3, :, :] = float(c_stream[ix, ell]) * m_s + float(c_mirror[ix, ell]) * mir_s
                    tail_factors_build[s][ix] = factor_block_tridiagonal(
                        jnp.asarray(diag_tail, dtype=precond_dtype),
                        jnp.asarray(lower_tail, dtype=precond_dtype),
                        jnp.asarray(upper_tail, dtype=precond_dtype),
                    )

        cached = _PasTokamakThetaPrecondCache(
            inv_a01=jnp.asarray(inv_a01, dtype=precond_dtype),
            g01=jnp.asarray(g01, dtype=precond_dtype),
            inv_a=jnp.asarray(inv_a, dtype=precond_dtype),
            g=jnp.asarray(g, dtype=precond_dtype),
            c_stream=jnp.asarray(c_stream, dtype=precond_dtype),
            c_mirror=jnp.asarray(c_mirror, dtype=precond_dtype),
            m_theta=jnp.asarray(m_theta, dtype=precond_dtype),
            mirror_factor=jnp.asarray(mirror_factor, dtype=precond_dtype),
            mask_active=jnp.asarray(mask_active_full, dtype=precond_dtype),
            n_l_build=int(n_l_build),
            tail_factors=None if tail_factors_build is None else tuple(tuple(row) for row in tail_factors_build),
        )
        _RHSMODE1_PAS_TOKAMAK_THETA_CACHE[cache_key] = cached

    inv_a01 = cached.inv_a01
    g01 = cached.g01
    inv_a = cached.inv_a
    g = cached.g
    c_stream = cached.c_stream
    c_mirror = cached.c_mirror
    m_theta = cached.m_theta
    mirror_factor = cached.mirror_factor
    mask_active = cached.mask_active
    n_l_build = int(cached.n_l_build)
    tail_factors = cached.tail_factors

    f_shape = op.fblock.f_shape  # (S,X,L,T,Z)
    n_l_total = int(f_shape[2])
    n_theta = int(f_shape[3])
    n_zeta = int(f_shape[4])

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f_rhs_full = r_full[: op.f_size].reshape(f_shape)  # (S,X,L,T,Z)
        f_rhs_full = jnp.asarray(f_rhs_full, dtype=precond_dtype)
        f_rhs_full = f_rhs_full * mask_active[None, :, :, None, None]

        if n_zeta == 1:
            f_rhs_full = f_rhs_full[:, :, :, :, 0]  # (S,X,L,T)
            f_rhs = f_rhs_full[:, :, :n_l_build, :]
            f_rhs_high = f_rhs_full[:, :, n_l_build:, :]
            rhs0 = f_rhs[:, :, 0, :]  # (S,X,T)
            rhs1 = f_rhs[:, :, 1, :]  # (S,X,T)
            rhs01 = jnp.concatenate([rhs0, rhs1], axis=-1)  # (S,X,2T)
            d01 = jnp.einsum("sxij,sxj->sxi", inv_a01, rhs01)  # (S,X,2T)

            if n_l_build == 2:
                f0 = d01[:, :, :n_theta]
                f1 = d01[:, :, n_theta:]
                f_all = jnp.concatenate([f0[:, :, None, :], f1[:, :, None, :]], axis=2)  # (S,X,2,T)
            elif tail_factors is not None:
                f_rest_rows: list[list[jnp.ndarray]] = []
                for s in range(int(op.n_species)):
                    species_rows: list[jnp.ndarray] = []
                    for ix in range(int(op.n_x)):
                        factor = tail_factors[s][ix]
                        assert factor is not None
                        rhs_tail = jnp.asarray(f_rhs[s, ix, 2:n_l_build, :], dtype=precond_dtype)
                        d_prev = d01[s, ix, n_theta:]
                        corr2 = c_stream[ix, 2] * (m_theta[s] @ d_prev)
                        corr2 = corr2 + c_mirror[ix, 2] * (mirror_factor[s] * d_prev)
                        rhs_tail = rhs_tail.at[0, :].add(-corr2)
                        sol_tail = factor.solve(rhs_tail)
                        species_rows.append(jnp.asarray(sol_tail, dtype=precond_dtype))
                    f_rest_rows.append(jnp.stack(species_rows, axis=0))
                f_rest_all = jnp.stack(f_rest_rows, axis=0)  # (S,X,L-2,T)
                f2 = f_rest_all[:, :, 0, :]
                u01 = d01 - jnp.einsum("sxij,sxj->sxi", g01, f2)
                f0 = u01[:, :, :n_theta]
                f1 = u01[:, :, n_theta:]
                f_all = jnp.concatenate(
                    [f0[:, :, None, :], f1[:, :, None, :], f_rest_all],
                    axis=2,
                )
            else:
                d_prev = d01[:, :, n_theta:]  # (S,X,T)

                inv_a_l = jnp.transpose(inv_a, (2, 0, 1, 3, 4))  # (L-2,S,X,T,T)
                rhs_l = jnp.transpose(f_rhs[:, :, 2:, :], (2, 0, 1, 3))  # (L-2,S,X,T)
                c_stream_l = jnp.transpose(c_stream[:, 2:], (1, 0))  # (L-2,X)
                c_mirror_l = jnp.transpose(c_mirror[:, 2:], (1, 0))  # (L-2,X)

                def _fwd_step(d_prev: jnp.ndarray, inputs):
                    inv_block, rhs_block, cs, cm = inputs
                    corr_stream = jnp.einsum("sij,sxj->sxi", m_theta, d_prev) * cs[None, :, None]
                    corr_mirror = (mirror_factor[:, None, :] * d_prev) * cm[None, :, None]
                    rhs_eff = rhs_block - (corr_stream + corr_mirror)
                    d = jnp.einsum("sxij,sxj->sxi", inv_block, rhs_eff)
                    return d, d

                _, d_out = jax.lax.scan(_fwd_step, d_prev, (inv_a_l, rhs_l, c_stream_l, c_mirror_l))
                d_rest = jnp.transpose(d_out, (1, 2, 0, 3))  # (S,X,L-2,T)

                f_last = d_rest[:, :, -1, :]  # (S,X,T)
                if n_l_build > 3:
                    g_l = jnp.transpose(g, (2, 0, 1, 3, 4))  # (L-3,S,X,T,T)
                    d_rev = jnp.transpose(d_rest[:, :, :-1, :], (2, 0, 1, 3))[::-1]  # (L-3,S,X,T)
                    g_rev = g_l[::-1]

                    def _bwd_step(f_next: jnp.ndarray, inputs):
                        g_block, d_block = inputs
                        f_l = d_block - jnp.einsum("sxij,sxj->sxi", g_block, f_next)
                        return f_l, f_l

                    _, f_rev = jax.lax.scan(_bwd_step, f_last, (g_rev, d_rev))
                    f_prefix = f_rev[::-1]  # (L-3,S,X,T)
                    f_mid = jnp.transpose(f_prefix, (1, 2, 0, 3))  # (S,X,L-3,T)
                    f2 = f_mid[:, :, 0, :]  # L=2
                    f_rest_all = jnp.concatenate([f_mid, f_last[:, :, None, :]], axis=2)  # (S,X,L-2,T)
                else:
                    f2 = f_last
                    f_rest_all = f_last[:, :, None, :]  # (S,X,1,T)

                u01 = d01 - jnp.einsum("sxij,sxj->sxi", g01, f2)  # (S,X,2T)
                f0 = u01[:, :, :n_theta]
                f1 = u01[:, :, n_theta:]
                f_all = jnp.concatenate(
                    [f0[:, :, None, :], f1[:, :, None, :], f_rest_all],
                    axis=2,
                )  # (S,X,L,T)

            if n_l_build < n_l_total:
                f_all = jnp.concatenate([f_all, f_rhs_high], axis=2)
            f_all = f_all * mask_active[None, :, :, None]
            f_all = jnp.asarray(f_all, dtype=jnp.float64)
            z_f = f_all[:, :, :, :, None].reshape((-1,))
        else:
            # Apply the same theta/L preconditioner independently to each zeta plane.
            f_rhs_z_full = jnp.transpose(f_rhs_full, (4, 0, 1, 2, 3))  # (Z,S,X,L,T)
            f_rhs_z = f_rhs_z_full[:, :, :, :n_l_build, :]
            f_rhs_z_high = f_rhs_z_full[:, :, :, n_l_build:, :]
            rhs0 = f_rhs_z[:, :, :, 0, :]  # (Z,S,X,T)
            rhs1 = f_rhs_z[:, :, :, 1, :]  # (Z,S,X,T)
            rhs01 = jnp.concatenate([rhs0, rhs1], axis=-1)  # (Z,S,X,2T)
            d01 = jnp.einsum("sxij,zsxj->zsxi", inv_a01, rhs01)  # (Z,S,X,2T)

            if n_l_build == 2:
                f0 = d01[:, :, :, :n_theta]
                f1 = d01[:, :, :, n_theta:]
                f_all = jnp.concatenate([f0[:, :, :, None, :], f1[:, :, :, None, :]], axis=3)  # (Z,S,X,2,T)
            elif tail_factors is not None:
                z_rows: list[list[list[jnp.ndarray]]] = []
                for iz in range(n_zeta):
                    z_species: list[list[jnp.ndarray]] = []
                    for s in range(int(op.n_species)):
                        species_rows: list[jnp.ndarray] = []
                        for ix in range(int(op.n_x)):
                            factor = tail_factors[s][ix]
                            assert factor is not None
                            rhs_tail = jnp.asarray(f_rhs_z[iz, s, ix, 2:n_l_build, :], dtype=precond_dtype)
                            d_prev = d01[iz, s, ix, n_theta:]
                            corr2 = c_stream[ix, 2] * (m_theta[s] @ d_prev)
                            corr2 = corr2 + c_mirror[ix, 2] * (mirror_factor[s] * d_prev)
                            rhs_tail = rhs_tail.at[0, :].add(-corr2)
                            sol_tail = factor.solve(rhs_tail)
                            species_rows.append(jnp.asarray(sol_tail, dtype=precond_dtype))
                        z_species.append(species_rows)
                    z_rows.append(z_species)
                f_rest_all = jnp.asarray(z_rows, dtype=precond_dtype)  # (Z,S,X,L-2,T)
                f2 = f_rest_all[:, :, :, 0, :]
                u01 = d01 - jnp.einsum("sxij,zsxj->zsxi", g01, f2)
                f0 = u01[:, :, :, :n_theta]
                f1 = u01[:, :, :, n_theta:]
                f_all = jnp.concatenate(
                    [f0[:, :, :, None, :], f1[:, :, :, None, :], f_rest_all],
                    axis=3,
                )
            else:
                d_prev = d01[:, :, :, n_theta:]  # (Z,S,X,T)

                inv_a_l = jnp.transpose(inv_a, (2, 0, 1, 3, 4))  # (L-2,S,X,T,T)
                rhs_l = jnp.transpose(f_rhs_z[:, :, :, 2:, :], (3, 0, 1, 2, 4))  # (L-2,Z,S,X,T)
                c_stream_l = jnp.transpose(c_stream[:, 2:], (1, 0))  # (L-2,X)
                c_mirror_l = jnp.transpose(c_mirror[:, 2:], (1, 0))  # (L-2,X)

                def _fwd_step(d_prev: jnp.ndarray, inputs):
                    inv_block, rhs_block, cs, cm = inputs  # inv: (S,X,T,T), rhs: (Z,S,X,T)
                    corr_stream = jnp.einsum("sij,zsxj->zsxi", m_theta, d_prev) * cs[None, None, :, None]
                    corr_mirror = (mirror_factor[None, :, None, :] * d_prev) * cm[None, None, :, None]
                    rhs_eff = rhs_block - (corr_stream + corr_mirror)
                    d = jnp.einsum("sxij,zsxj->zsxi", inv_block, rhs_eff)
                    return d, d

                _, d_out = jax.lax.scan(_fwd_step, d_prev, (inv_a_l, rhs_l, c_stream_l, c_mirror_l))
                d_rest = jnp.transpose(d_out, (1, 2, 3, 0, 4))  # (Z,S,X,L-2,T)

                f_last = d_rest[:, :, :, -1, :]  # (Z,S,X,T)
                if n_l_build > 3:
                    g_l = jnp.transpose(g, (2, 0, 1, 3, 4))  # (L-3,S,X,T,T)
                    d_rev = jnp.transpose(d_rest[:, :, :, :-1, :], (3, 0, 1, 2, 4))[::-1]  # (L-3,Z,S,X,T)
                    g_rev = g_l[::-1]

                    def _bwd_step(f_next: jnp.ndarray, inputs):
                        g_block, d_block = inputs
                        f_l = d_block - jnp.einsum("sxij,zsxj->zsxi", g_block, f_next)
                        return f_l, f_l

                    _, f_rev = jax.lax.scan(_bwd_step, f_last, (g_rev, d_rev))
                    f_prefix = f_rev[::-1]  # (L-3,Z,S,X,T)
                    f_mid = jnp.transpose(f_prefix, (1, 2, 3, 0, 4))  # (Z,S,X,L-3,T)
                    f2 = f_mid[:, :, :, 0, :]  # L=2
                    f_rest_all = jnp.concatenate([f_mid, f_last[:, :, :, None, :]], axis=3)  # (Z,S,X,L-2,T)
                else:
                    f2 = f_last
                    f_rest_all = f_last[:, :, :, None, :]  # (Z,S,X,1,T)

                u01 = d01 - jnp.einsum("sxij,zsxj->zsxi", g01, f2)  # (Z,S,X,2T)
                f0 = u01[:, :, :, :n_theta]
                f1 = u01[:, :, :, n_theta:]
                f_all = jnp.concatenate(
                    [f0[:, :, :, None, :], f1[:, :, :, None, :], f_rest_all],
                    axis=3,
                )  # (Z,S,X,L,T)

            if n_l_build < n_l_total:
                f_all = jnp.concatenate([f_all, f_rhs_z_high], axis=3)
            f_all = f_all * mask_active[None, None, :, :, None]
            f_all = jnp.asarray(f_all, dtype=jnp.float64)
            f_all = jnp.transpose(f_all, (1, 2, 3, 4, 0))  # (S,X,L,T,Z)
            z_f = f_all.reshape((-1,))

        tail = r_full[op.f_size :]
        return jnp.concatenate([z_f, tail], axis=0)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def build_rhs1_pas_tz_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    pas_tz_applicable: Callable[[V3FullSystemOperator], bool],
    pas_tz_memory_safe: Callable[[V3FullSystemOperator], bool],
    matvec_shard_axis: Callable[[object], str | None],
    device_count: Callable[[], int],
    theta_schwarz_builder: Callable[..., Preconditioner],
    zeta_schwarz_builder: Callable[..., Preconditioner],
    pas_hybrid_builder: Callable[..., Preconditioner],
    collision_builder: Callable[..., Preconditioner],
    tzfft_builder: Callable[..., Preconditioner],
) -> Preconditioner:
    """PAS 3D (theta,zeta)/L block-tridiagonal preconditioner for RHSMode=1.

    This preconditioner targets PAS-only 3D cases for which building dense per-x
    (theta,zeta,L) blocks (e.g. `xblock_tz`) is extremely expensive in both time and
    memory. We exploit the collisionless streaming+mirror structure, which couples
    only L<->L±1, to build a block-tridiagonal-in-L approximation with blocks of size
    Ntheta*Nzeta.

    The block-Thomas factors are precomputed once per operator signature and applied
    in O(Nx * Nxi * (Ntheta*Nzeta)^2) per Krylov iteration.
    """
    if not pas_tz_applicable(op):
        return pas_hybrid_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if not pas_tz_memory_safe(op):
        return build_pas_tz_memory_fallback(
            op=op,
            matvec_shard_axis=matvec_shard_axis,
            device_count=device_count,
            theta_schwarz_builder=theta_schwarz_builder,
            zeta_schwarz_builder=zeta_schwarz_builder,
            hybrid_builder=pas_hybrid_builder,
            collision_builder=collision_builder,
            tzfft_builder=tzfft_builder,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    precond_dtype = _precond_dtype()
    cl = op.fblock.collisionless
    pas = op.fblock.pas
    assert cl is not None
    assert pas is not None

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l_full = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_tz = int(n_theta * n_zeta)

    if n_tz <= 1 or n_l_full < 2:
        return pas_hybrid_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    pas_tz_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", "").strip()
    try:
        pas_tz_lmax = int(pas_tz_lmax_env) if pas_tz_lmax_env else 0
    except ValueError:
        pas_tz_lmax = 0
    if pas_tz_lmax <= 0:
        # Keep the PAS TZ preconditioner inexpensive for large angular grids, but for
        # moderate (theta,zeta) sizes we can afford the full L coupling to improve convergence.
        if n_tz <= 192:
            pas_tz_lmax = n_l_full
        elif n_tz >= 256:
            pas_tz_lmax = 6
        elif n_tz >= 128:
            pas_tz_lmax = 8
        else:
            pas_tz_lmax = 12
        if op.fblock.exb_theta is None and op.fblock.exb_zeta is None and op.fblock.magdrift_theta is None and op.fblock.magdrift_zeta is None and op.fblock.magdrift_xidot is None and op.fblock.er_xdot is None and op.fblock.er_xidot is None:
            # Pure PAS (no drift/Er) systems are block-diagonal in x; use full-L coupling
            # to reduce Krylov iterations even on moderately sized angular grids.
            if n_tz <= 256:
                pas_tz_lmax = n_l_full
    if n_l_full >= 2:
        n_l_use = min(n_l_full, max(2, int(pas_tz_lmax)))
    else:
        n_l_use = n_l_full

    cache_key = _rhsmode1_precond_cache_key(op, f"pas_tz_l{n_l_use}")
    cached = _RHSMODE1_PAS_TZ_CACHE.get(cache_key)
    if cached is None:
        n_l = n_l_full

        x = np.asarray(cl.x, dtype=np.float64)  # (X,)
        ddtheta = np.asarray(cl.ddtheta, dtype=np.float64)  # (T,T)
        ddzeta = np.asarray(cl.ddzeta, dtype=np.float64)  # (Z,Z)
        b_hat = np.asarray(cl.b_hat, dtype=np.float64)  # (T,Z)
        b_sup_theta = np.asarray(cl.b_hat_sup_theta, dtype=np.float64)  # (T,Z)
        b_sup_zeta = np.asarray(cl.b_hat_sup_zeta, dtype=np.float64)  # (T,Z)
        db_dtheta = np.asarray(cl.db_hat_dtheta, dtype=np.float64)  # (T,Z)
        db_dzeta = np.asarray(cl.db_hat_dzeta, dtype=np.float64)  # (T,Z)
        t_hats = np.asarray(cl.t_hats, dtype=np.float64)  # (S,)
        m_hats = np.asarray(cl.m_hats, dtype=np.float64)  # (S,)
        nxi_for_x = np.asarray(cl.n_xi_for_x, dtype=np.int32)  # (X,)

        l_arr = np.arange(n_l, dtype=np.float64)
        coef_plus = (l_arr + 1.0) / (2.0 * l_arr + 3.0)
        coef_minus = np.where(l_arr > 0, l_arr / (2.0 * l_arr - 1.0), 0.0)
        coef_mirror_plus = (l_arr + 1.0) * (l_arr + 2.0) / (2.0 * l_arr + 3.0)
        coef_mirror_minus = np.where(l_arr > 1, -l_arr * (l_arr - 1.0) / (2.0 * l_arr - 1.0), 0.0)

        coef_plus_x = x[:, None] * coef_plus[None, :]  # (X,L)
        coef_minus_x = x[:, None] * coef_minus[None, :]  # (X,L)
        coef_mirror_plus_x = x[:, None] * coef_mirror_plus[None, :]  # (X,L)
        coef_mirror_minus_x = x[:, None] * coef_mirror_minus[None, :]  # (X,L)

        mask_active = (np.arange(n_l, dtype=np.int32)[None, :] < nxi_for_x[:, None]).astype(np.float64)  # (X,L)
        mask_plus = (mask_active[:, :-1] * mask_active[:, 1:]).astype(np.float64)  # (X,L-1)
        mask_minus = mask_plus

        b_stream = np.zeros((n_x, n_l), dtype=np.float64)
        b_mirror = np.zeros((n_x, n_l), dtype=np.float64)
        c_stream = np.zeros((n_x, n_l), dtype=np.float64)
        c_mirror = np.zeros((n_x, n_l), dtype=np.float64)
        b_stream[:, :-1] = coef_plus_x[:, :-1] * mask_plus
        b_mirror[:, :-1] = coef_mirror_plus_x[:, :-1] * mask_plus
        c_stream[:, 1:] = coef_minus_x[:, 1:] * mask_minus
        c_mirror[:, 1:] = coef_mirror_minus_x[:, 1:] * mask_minus

        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_REG", "").strip()
        try:
            reg = float(reg_env) if reg_env else 1e-12
        except ValueError:
            reg = 1e-12
        reg = max(float(reg), 0.0)

        pas_coef = np.asarray(pas.coef, dtype=np.float64)  # (S,X,L)
        identity_shift = float(op.fblock.identity_shift)
        diag_vals = pas_coef + identity_shift + reg  # (S,X,L)
        # Inactive (x,L) decouple to identity.
        diag_vals = diag_vals * mask_active[None, :, :] + (1.0 - mask_active)[None, :, :]
        diag_inv = 1.0 / diag_vals

        i_zeta = np.eye(n_zeta, dtype=np.float64)
        i_theta = np.eye(n_theta, dtype=np.float64)
        dtheta_tz = np.kron(ddtheta, i_zeta)  # (TZ,TZ) zeta-fast layout
        dzeta_tz = np.kron(i_theta, ddzeta)  # (TZ,TZ)

        v_theta = b_sup_theta / b_hat  # (T,Z)
        v_zeta = b_sup_zeta / b_hat  # (T,Z)
        sqrt_t_over_m = np.sqrt(t_hats / m_hats)  # (S,)
        mirror_geom = b_sup_theta * db_dtheta + b_sup_zeta * db_dzeta  # (T,Z)
        mirror_base = mirror_geom / (2.0 * (b_hat * b_hat))  # (T,Z)

        m_tz = np.zeros((n_species, n_tz, n_tz), dtype=np.float64)
        mirror_factor = np.zeros((n_species, n_tz), dtype=np.float64)
        for s in range(n_species):
            v_theta_s = (sqrt_t_over_m[s] * v_theta).reshape((-1,))  # (TZ,)
            v_zeta_s = (sqrt_t_over_m[s] * v_zeta).reshape((-1,))  # (TZ,)
            m_tz[s, :, :] = v_theta_s[:, None] * dtheta_tz + v_zeta_s[:, None] * dzeta_tz
            mirror_factor[s, :] = (-sqrt_t_over_m[s] * mirror_base).reshape((-1,))

        exb_op_tz = np.zeros((n_tz, n_tz), dtype=np.float64)
        if op.fblock.exb_theta is not None:
            exb_theta = op.fblock.exb_theta
            alpha = float(np.asarray(exb_theta.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(exb_theta.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(exb_theta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            if getattr(exb_theta, "use_dkes_exb_drift", False):
                denom = float(np.asarray(exb_theta.fsab_hat2, dtype=np.float64).reshape(()))
                coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / denom
            else:
                b2 = np.asarray(exb_theta.b_hat, dtype=np.float64) ** 2
                coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64) / b2
            factor = alpha * delta * 0.5 * dphi
            coef_flat = (factor * coef).reshape((-1,))
            exb_op_tz = exb_op_tz + coef_flat[:, None] * dtheta_tz
        if op.fblock.exb_zeta is not None:
            exb_zeta = op.fblock.exb_zeta
            alpha = float(np.asarray(exb_zeta.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(exb_zeta.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(exb_zeta.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            if getattr(exb_zeta, "use_dkes_exb_drift", False):
                denom = float(np.asarray(exb_zeta.fsab_hat2, dtype=np.float64).reshape(()))
                coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / denom
            else:
                b2 = np.asarray(exb_zeta.b_hat, dtype=np.float64) ** 2
                coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64) / b2
            factor = -alpha * delta * 0.5 * dphi
            coef_flat = (factor * coef).reshape((-1,))
            exb_op_tz = exb_op_tz + coef_flat[:, None] * dzeta_tz

        tz = int(n_tz)
        twotz = int(2 * tz)
        eye_tz = np.eye(tz, dtype=np.float64)

        n_l_build = int(n_l_use)
        inv_a01 = np.zeros((n_species, n_x, twotz, twotz), dtype=np.float64)
        g01 = np.zeros((n_species, n_x, twotz, tz), dtype=np.float64)
        inv_a = np.zeros((n_species, n_x, max(n_l_build - 2, 0), tz, tz), dtype=np.float64)
        g = np.zeros((n_species, n_x, max(n_l_build - 3, 0), tz, tz), dtype=np.float64)
        b_stream_use = b_stream[:, :n_l_build]
        b_mirror_use = b_mirror[:, :n_l_build]
        c_stream_use = c_stream[:, :n_l_build]
        c_mirror_use = c_mirror[:, :n_l_build]
        mask_active_use = mask_active[:, :n_l_build]

        for s in range(n_species):
            m_s = m_tz[s, :, :]  # (TZ,TZ)
            mir_vec = mirror_factor[s, :]  # (TZ,)
            mir_s = mir_vec[:, None] * eye_tz  # diagonal matrix
            for ix in range(n_x):
                # Combined (L=0,1) block to avoid L=0 singularities.
                active0 = float(mask_active_use[ix, 0])
                active1 = float(mask_active_use[ix, 1])
                a0 = float(diag_vals[s, ix, 0])
                a1 = float(diag_vals[s, ix, 1])
                a0_blk = a0 * eye_tz + active0 * exb_op_tz
                a1_blk = a1 * eye_tz + active1 * exb_op_tz
                b0 = float(b_stream_use[ix, 0]) * m_s + float(b_mirror_use[ix, 0]) * mir_s
                c1 = float(c_stream_use[ix, 1]) * m_s + float(c_mirror_use[ix, 1]) * mir_s
                a01 = np.zeros((twotz, twotz), dtype=np.float64)
                a01[:tz, :tz] = a0_blk
                a01[:tz, tz:] = b0
                a01[tz:, :tz] = c1
                a01[tz:, tz:] = a1_blk
                try:
                    inv01 = np.linalg.inv(a01)
                except np.linalg.LinAlgError:
                    inv01 = np.linalg.pinv(a01, rcond=1e-12)
                if not np.all(np.isfinite(inv01)):
                    inv01 = np.linalg.pinv(a01, rcond=1e-12)
                inv_a01[s, ix, :, :] = inv01

                if n_l == 2:
                    continue

                # G01 = inv(A01) @ B_tilde, where B_tilde couples f2 into the L=1 equation only.
                b1 = float(b_stream_use[ix, 1]) * m_s + float(b_mirror_use[ix, 1]) * mir_s
                b_tilde = np.zeros((twotz, tz), dtype=np.float64)
                b_tilde[tz:, :] = b1
                g01_loc = inv01 @ b_tilde
                g01[s, ix, :, :] = g01_loc

                g_prev = g01_loc
                for ell in range(2, n_l_build):
                    active_ell = float(mask_active_use[ix, ell])
                    a_diag = float(diag_vals[s, ix, ell])
                    a_blk = a_diag * eye_tz + active_ell * exb_op_tz
                    if ell == 2:
                        c2 = float(c_stream_use[ix, ell]) * m_s + float(c_mirror_use[ix, ell]) * mir_s
                        c_tilde = np.zeros((tz, twotz), dtype=np.float64)
                        c_tilde[:, tz:] = c2
                        a_eff = a_blk - c_tilde @ g_prev
                    else:
                        c_ell = float(c_stream_use[ix, ell]) * m_s + float(c_mirror_use[ix, ell]) * mir_s
                        a_eff = a_blk - c_ell @ g_prev
                    try:
                        a_inv = np.linalg.inv(a_eff)
                    except np.linalg.LinAlgError:
                        a_inv = np.linalg.pinv(a_eff, rcond=1e-12)
                    if not np.all(np.isfinite(a_inv)):
                        a_inv = np.linalg.pinv(a_eff, rcond=1e-12)
                    inv_a[s, ix, ell - 2, :, :] = a_inv
                    if ell < n_l_build - 1:
                        b_ell = float(b_stream_use[ix, ell]) * m_s + float(b_mirror_use[ix, ell]) * mir_s
                        g_ell = a_inv @ b_ell
                        g[s, ix, ell - 2, :, :] = g_ell
                        g_prev = g_ell

        cached = _PasTzPrecondCache(
            inv_a01=jnp.asarray(inv_a01, dtype=precond_dtype),
            g01=jnp.asarray(g01, dtype=precond_dtype),
            inv_a=jnp.asarray(inv_a, dtype=precond_dtype),
            g=jnp.asarray(g, dtype=precond_dtype),
            c_stream=jnp.asarray(c_stream_use, dtype=precond_dtype),
            c_mirror=jnp.asarray(c_mirror_use, dtype=precond_dtype),
            m_tz=jnp.asarray(m_tz, dtype=precond_dtype),
            mirror_factor=jnp.asarray(mirror_factor, dtype=precond_dtype),
            mask_active=jnp.asarray(mask_active, dtype=precond_dtype),
            diag_inv=jnp.asarray(diag_inv, dtype=precond_dtype),
            n_l_use=int(n_l_use),
        )
        _RHSMODE1_PAS_TZ_CACHE[cache_key] = cached

    inv_a01 = cached.inv_a01
    g01 = cached.g01
    inv_a = cached.inv_a
    g = cached.g
    c_stream = cached.c_stream
    c_mirror = cached.c_mirror
    m_tz = cached.m_tz
    mirror_factor = cached.mirror_factor
    mask_active = cached.mask_active
    diag_inv = cached.diag_inv
    n_l_use = int(cached.n_l_use)

    f_shape = op.fblock.f_shape  # (S,X,L,T,Z)
    n_species = int(f_shape[0])
    n_x = int(f_shape[1])
    n_l_full = int(f_shape[2])
    n_theta = int(f_shape[3])
    n_zeta = int(f_shape[4])
    tz = int(n_theta * n_zeta)
    n_l_use = max(0, min(int(n_l_use), n_l_full))

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        f_rhs = r_full[: op.f_size].reshape(f_shape).reshape((n_species, n_x, n_l_full, tz))  # (S,X,L,TZ)
        f_rhs = jnp.asarray(f_rhs, dtype=precond_dtype)
        f_rhs = f_rhs * mask_active[None, :, :, None]
        if n_l_use <= 1:
            f_use = f_rhs[:, :, :n_l_use, :]
            f_high = f_rhs[:, :, n_l_use:, :]
            f_use = f_use * diag_inv[:, :, :n_l_use, None]
            f_high = f_high * diag_inv[:, :, n_l_use:, None]
            f_all = jnp.concatenate([f_use, f_high], axis=2)
            f_all = f_all * mask_active[None, :, :, None]
            f_all = jnp.asarray(f_all, dtype=jnp.float64).reshape((n_species, n_x, n_l_full, n_theta, n_zeta))
            z_f = f_all.reshape((-1,))
            tail = r_full[op.f_size :]
            return jnp.concatenate([z_f, tail], axis=0)

        f_rhs_use = f_rhs[:, :, :n_l_use, :]

        rhs0 = f_rhs_use[:, :, 0, :]  # (S,X,TZ)
        rhs1 = f_rhs_use[:, :, 1, :]  # (S,X,TZ)
        rhs01 = jnp.concatenate([rhs0, rhs1], axis=-1)  # (S,X,2TZ)
        d01 = jnp.einsum("sxij,sxj->sxi", inv_a01, rhs01)  # (S,X,2TZ)

        if n_l_use == 2:
            f0 = d01[:, :, :tz]
            f1 = d01[:, :, tz:]
            f_all = jnp.concatenate([f0[:, :, None, :], f1[:, :, None, :]], axis=2)  # (S,X,2,TZ)
        else:
            d_prev = d01[:, :, tz:]  # (S,X,TZ)

            inv_a_l = jnp.transpose(inv_a, (2, 0, 1, 3, 4))  # (L-2,S,X,TZ,TZ) for L>=2
            rhs_l = jnp.transpose(f_rhs_use[:, :, 2:, :], (2, 0, 1, 3))  # (L-2,S,X,TZ)
            c_stream_l = jnp.transpose(c_stream[:, 2:], (1, 0))  # (L-2,X)
            c_mirror_l = jnp.transpose(c_mirror[:, 2:], (1, 0))  # (L-2,X)

            def _fwd_step(d_prev: jnp.ndarray, inputs):
                inv_block, rhs_block, cs, cm = inputs
                corr_stream = jnp.einsum("sij,sxj->sxi", m_tz, d_prev) * cs[None, :, None]
                corr_mirror = (mirror_factor[:, None, :] * d_prev) * cm[None, :, None]
                rhs_eff = rhs_block - (corr_stream + corr_mirror)
                d = jnp.einsum("sxij,sxj->sxi", inv_block, rhs_eff)
                return d, d

            _, d_out = jax.lax.scan(_fwd_step, d_prev, (inv_a_l, rhs_l, c_stream_l, c_mirror_l))
            d_rest = jnp.transpose(d_out, (1, 2, 0, 3))  # (S,X,L-2,TZ)

            f_last = d_rest[:, :, -1, :]  # (S,X,TZ) corresponds to L=n_l-1
            if n_l_use > 3:
                g_l = jnp.transpose(g, (2, 0, 1, 3, 4))  # (L-3,S,X,TZ,TZ) for L=2..n_l-2
                d_rev = jnp.transpose(d_rest[:, :, :-1, :], (2, 0, 1, 3))[::-1]  # (L-3,S,X,TZ)
                g_rev = g_l[::-1]

                def _bwd_step(f_next: jnp.ndarray, inputs):
                    g_block, d_block = inputs
                    f_l = d_block - jnp.einsum("sxij,sxj->sxi", g_block, f_next)
                    return f_l, f_l

                _, f_rev = jax.lax.scan(_bwd_step, f_last, (g_rev, d_rev))
                f_prefix = f_rev[::-1]  # (L-3,S,X,TZ)
                f_mid = jnp.transpose(f_prefix, (1, 2, 0, 3))  # (S,X,L-3,TZ) for L=2..n_l-2
                f2 = f_mid[:, :, 0, :]  # L=2
                f_rest_all = jnp.concatenate([f_mid, f_last[:, :, None, :]], axis=2)  # (S,X,L-2,TZ)
            else:
                f2 = f_last
                f_rest_all = f_last[:, :, None, :]  # (S,X,1,TZ)

            u01 = d01 - jnp.einsum("sxij,sxj->sxi", g01, f2)  # (S,X,2TZ)
            f0 = u01[:, :, :tz]
            f1 = u01[:, :, tz:]
            f_all = jnp.concatenate([f0[:, :, None, :], f1[:, :, None, :], f_rest_all], axis=2)  # (S,X,L_use,TZ)

        if n_l_use < n_l_full:
            f_high = f_rhs[:, :, n_l_use:, :] * diag_inv[:, :, n_l_use:, None]
            f_all = jnp.concatenate([f_all, f_high], axis=2)
        f_all = f_all * mask_active[None, :, :, None]
        f_all = jnp.asarray(f_all, dtype=jnp.float64).reshape((n_species, n_x, n_l_full, n_theta, n_zeta))
        z_f = f_all.reshape((-1,))
        tail = r_full[op.f_size :]
        return jnp.concatenate([z_f, tail], axis=0)

    if reduce_full is None or expand_reduced is None:
        return _apply_full

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = _apply_full(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced

