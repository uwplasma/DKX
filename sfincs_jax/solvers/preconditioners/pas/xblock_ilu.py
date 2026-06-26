"""PAS x-block ILU preconditioner for RHSMode=1 solves."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_caches import (
    _RHSMODE1_PRECOND_ILU_CACHE,
    _RHSMode1ILUBlockPrecondCache,
)
from sfincs_jax.solvers.preconditioner_context import precond_dtype
from sfincs_jax.solvers.preconditioner_setup import (
    hash_array,
    matvec_submatrix_v3_unsharded,
    precond_chunk_cols,
    rhs_mode1_precond_cache_key,
)
from ....problems.profile_response.residual import safe_preconditioner
from sfincs_jax.solvers.sparse_triangular import (
    inverse_permutation,
    triangular_solve_lower_padded,
    triangular_solve_upper_padded,
)
from ....v3_system import V3FullSystemOperator

__all__ = [
    "build_rhs1_pas_xblock_ilu_preconditioner",
    "rhsmode1_pas_xblock_precond_cache_key",
]


def rhsmode1_pas_xblock_precond_cache_key(
    op: V3FullSystemOperator,
    kind: str = "pas_xblock_ilu",
) -> tuple[object, ...]:
    """Return the RHSMode=1 PAS x-block preconditioner cache key."""

    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=precond_dtype())


def build_rhs1_pas_xblock_ilu_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    pas_hybrid_preconditioner: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Sparse block-Jacobi ILU preconditioner for PAS-like RHSMode=1 operators.

    This preconditioner targets PAS DKES / PAS collisionless runs that are *dense* in the
    existing (theta,zeta,L) block inverses, making preconditioner build/application very
    expensive.

    We assemble a sparse approximation to the per-(species,x) (L,theta,zeta) block matrix
    using the same stencil-like operator structure used in the matrix-free apply:
    - collisionless streaming+mirror (ΔL=±1, spatial derivatives)
    - ExB drifts (spatial derivatives, diagonal in L)
    - pitch-angle scattering collisions (diagonal in L and in theta/zeta)

    Then we compute an ILU factorization with SciPy (`spilu`), and convert the L/U factors
    to a padded row format so the apply path is pure JAX (triangular solves) and can run
    inside JITted GMRES iterations.

    Notes
    -----
    - The ILU factorization is *not* differentiated through; gradients for solves come from
      the implicit-diff VJP of the linear solve, not backpropagating through GMRES.
    - This is intended for PAS-only (no FP) operators. For FP, the x-coupling makes a
      per-x block ILU far less effective.
    """
    if op.fblock.pas is None or op.fblock.fp is not None:
        # Not applicable: fall back to existing defaults.
        return pas_hybrid_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    try:
        import scipy.sparse as sp  # noqa: PLC0415
        from scipy.sparse.linalg import spilu, splu  # noqa: PLC0415
    except Exception:
        # SciPy missing: fall back to existing defaults.
        return pas_hybrid_preconditioner(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

    # Detect the DKES-trajectory branch (strong ExB drift). These PAS systems are often
    # much stiffer / less diagonally-dominant than tokamak-like cases, and the default
    # ILU drop settings can be too aggressive, leading to slow or stalled Krylov solves.
    exb_theta = op.fblock.exb_theta
    exb_zeta = op.fblock.exb_zeta
    use_dkes_exb = bool(getattr(exb_theta, "use_dkes_exb_drift", False)) or bool(
        getattr(exb_zeta, "use_dkes_exb_drift", False)
    )

    # ILU parameters (PETSc-like PCILU defaults). Keep these conservative to
    # avoid unstable factors on ill-conditioned blocks. In particular, we cap
    # the stored nnz-per-row for L/U factors so the JAX-side padded-row format
    # does not explode memory usage on hard PAS blocks.
    ilu_drop_tol_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_ILU_DROP_TOL", "").strip()
    ilu_fill_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_ILU_FILL_FACTOR", "").strip()
    row_nnz_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_ILU_ROW_NNZ_MAX", "").strip()
    reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_ILU_REG", "").strip()
    try:
        # For DKES-like PAS cases, default to a stronger (less dropping) ILU since the
        # operator can be poorly conditioned.
        default_drop = 1e-4 if use_dkes_exb else 1e-3
        ilu_drop_tol = float(ilu_drop_tol_env) if ilu_drop_tol_env else default_drop
    except ValueError:
        ilu_drop_tol = 1e-4 if use_dkes_exb else 1e-3
    try:
        default_fill = 10.0 if use_dkes_exb else 5.0
        ilu_fill_factor = float(ilu_fill_env) if ilu_fill_env else default_fill
    except ValueError:
        ilu_fill_factor = 10.0 if use_dkes_exb else 5.0
    try:
        row_nnz_max = int(row_nnz_env) if row_nnz_env else 64
    except ValueError:
        row_nnz_max = 64
    row_nnz_max = max(0, int(row_nnz_max))
    try:
        reg = float(reg_env) if reg_env else 1e-12
    except ValueError:
        reg = 1e-12
    reg = float(max(reg, 0.0))

    # Cache key: include the physics pieces that affect the assembled sparse block.
    # For small per-x blocks, an exact sparse LU can be both robust and faster overall
    # than a weak ILU (fewer Krylov iterations). We only enable this for small blocks
    # to avoid dense factors in the padded-row JAX format.
    lu_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_LU_MAX", "").strip()
    try:
        # For DKES-like PAS systems, the sparse blocks can be much less diagonally-dominant.
        # Default to a larger LU cutoff there for robustness. For non-DKES PAS systems,
        # prefer ILU for medium-sized blocks to reduce factorization time and memory.
        default_lu_max = 5000 if use_dkes_exb else 1000
        lu_max = int(lu_max_env) if lu_max_env else default_lu_max
    except ValueError:
        lu_max = 5000 if use_dkes_exb else 1000
    lu_max = max(0, int(lu_max))
    # Row cap for *exact LU* factors. Exact sparse LU can have substantially more fill than
    # ILU; we allow a higher cap for LU factors (to keep the preconditioner strong) but
    # still cap aggressively to avoid allocating dense padded-row arrays.
    lu_full_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_LU_ROW_NNZ_MAX", "").strip()
    try:
        # Default to a lower LU row cap than earlier revisions to reduce peak RSS.
        # If LU is needed for robustness in a particularly stiff DKES case, this can
        # be increased via env var without touching code.
        lu_full_row_nnz_max = int(lu_full_env) if lu_full_env else 256
    except ValueError:
        lu_full_row_nnz_max = 256
    lu_full_row_nnz_max = max(0, int(lu_full_row_nnz_max))

    colless = op.fblock.collisionless
    pas = op.fblock.pas

    cache_key = (
        *rhsmode1_pas_xblock_precond_cache_key(op),
        float(op.fblock.identity_shift),
        float(pas.nu_n),
        float(pas.krook),
        hash_array(pas.nu_d_hat),
        hash_array(colless.ddtheta),
        hash_array(colless.ddzeta),
        hash_array(colless.b_hat_sup_theta),
        hash_array(colless.b_hat_sup_zeta),
        hash_array(colless.db_hat_dtheta),
        hash_array(colless.db_hat_dzeta),
        float(ilu_drop_tol),
        float(ilu_fill_factor),
        int(row_nnz_max),
        float(reg),
        int(lu_max),
        int(lu_full_row_nnz_max),
        bool(exb_theta is not None),
        bool(exb_zeta is not None),
        bool(getattr(exb_theta, "use_dkes_exb_drift", False)) if exb_theta is not None else False,
        bool(getattr(exb_zeta, "use_dkes_exb_drift", False)) if exb_zeta is not None else False,
    )
    cached = _RHSMODE1_PRECOND_ILU_CACHE.get(cache_key)
    if cached is None:
        n_species = int(op.n_species)
        n_x = int(op.n_x)
        n_l = int(op.n_xi)
        n_theta = int(op.n_theta)
        n_zeta = int(op.n_zeta)
        total_size = int(op.total_size)
        nxi_for_x = np.asarray(colless.n_xi_for_x, dtype=np.int32)

        # Build sparse derivative matrices on the (theta,zeta) grid. (zeta is fastest.)
        ddtheta = np.asarray(colless.ddtheta, dtype=np.float64)
        ddzeta = np.asarray(colless.ddzeta, dtype=np.float64)
        ddtheta_sp = sp.csr_matrix(ddtheta)
        ddzeta_sp = sp.csr_matrix(ddzeta)
        i_theta = sp.eye(n_theta, format="csr", dtype=np.float64)
        i_zeta = sp.eye(n_zeta, format="csr", dtype=np.float64)
        dtheta_tz = sp.kron(ddtheta_sp, i_zeta, format="csr")
        dzeta_tz = sp.kron(i_theta, ddzeta_sp, format="csr")
        n_tz = int(n_theta * n_zeta)
        i_tz = sp.eye(n_tz, format="csr", dtype=np.float64)

        # Precompute ExB drift diagonal-in-L spatial operator (shared across species/x).
        exb_op_tz: sp.csr_matrix | None = None
        if exb_theta is not None:
            exb_theta_np = exb_theta
            alpha = float(np.asarray(exb_theta_np.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(exb_theta_np.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(exb_theta_np.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            if getattr(exb_theta_np, "use_dkes_exb_drift", False):
                denom = float(np.asarray(exb_theta_np.fsab_hat2, dtype=np.float64).reshape(()))
                coef = np.asarray(exb_theta_np.d_hat * exb_theta_np.b_hat_sub_zeta, dtype=np.float64) / denom
            else:
                b2 = np.asarray(exb_theta_np.b_hat, dtype=np.float64) ** 2
                coef = np.asarray(exb_theta_np.d_hat * exb_theta_np.b_hat_sub_zeta, dtype=np.float64) / b2
            factor = alpha * delta * 0.5 * dphi
            coef_flat = (factor * coef).reshape((-1,))
            exb_op_tz = sp.diags(coef_flat, 0, format="csr") @ dtheta_tz
        if exb_zeta is not None:
            exb_zeta_np = exb_zeta
            alpha = float(np.asarray(exb_zeta_np.alpha, dtype=np.float64).reshape(()))
            delta = float(np.asarray(exb_zeta_np.delta, dtype=np.float64).reshape(()))
            dphi = float(np.asarray(exb_zeta_np.dphi_hat_dpsi_hat, dtype=np.float64).reshape(()))
            if getattr(exb_zeta_np, "use_dkes_exb_drift", False):
                denom = float(np.asarray(exb_zeta_np.fsab_hat2, dtype=np.float64).reshape(()))
                coef = np.asarray(exb_zeta_np.d_hat * exb_zeta_np.b_hat_sub_theta, dtype=np.float64) / denom
            else:
                b2 = np.asarray(exb_zeta_np.b_hat, dtype=np.float64) ** 2
                coef = np.asarray(exb_zeta_np.d_hat * exb_zeta_np.b_hat_sub_theta, dtype=np.float64) / b2
            factor = -alpha * delta * 0.5 * dphi
            coef_flat = (factor * coef).reshape((-1,))
            term = sp.diags(coef_flat, 0, format="csr") @ dzeta_tz
            exb_op_tz = term if exb_op_tz is None else (exb_op_tz + term)

        # Geometry pieces for collisionless streaming/mirror.
        b_hat = np.asarray(colless.b_hat, dtype=np.float64)
        v_theta = np.asarray(colless.b_hat_sup_theta, dtype=np.float64) / b_hat
        v_zeta = np.asarray(colless.b_hat_sup_zeta, dtype=np.float64) / b_hat
        mirror_geom = (
            np.asarray(colless.b_hat_sup_theta, dtype=np.float64) * np.asarray(colless.db_hat_dtheta, dtype=np.float64)
            + np.asarray(colless.b_hat_sup_zeta, dtype=np.float64) * np.asarray(colless.db_hat_dzeta, dtype=np.float64)
        )
        mirror_geom = np.asarray(mirror_geom, dtype=np.float64)

        t_hat = np.asarray(op.t_hat, dtype=np.float64).reshape((n_species,))
        m_hat = np.asarray(op.m_hat, dtype=np.float64).reshape((n_species,))
        x_arr = np.asarray(colless.x, dtype=np.float64).reshape((n_x,))
        pas_coef = np.asarray(pas.coef, dtype=np.float64)  # (S,X,Lmax)
        identity_shift = float(np.asarray(op.fblock.identity_shift, dtype=np.float64).reshape(()))

        # Precompute per-species spatial operators.
        stream_op_by_s: list[sp.csr_matrix] = []
        mirror_diag_by_s: list[sp.csr_matrix] = []
        sqrt_t_over_m = np.sqrt(t_hat / m_hat)  # (S,)
        for s in range(n_species):
            v_theta_s_flat = (sqrt_t_over_m[s] * v_theta).reshape((-1,))
            v_zeta_s_flat = (sqrt_t_over_m[s] * v_zeta).reshape((-1,))
            stream = sp.diags(v_theta_s_flat, 0, format="csr") @ dtheta_tz
            stream = stream + (sp.diags(v_zeta_s_flat, 0, format="csr") @ dzeta_tz)
            stream_op_by_s.append(stream)

            mirror_factor_flat = (-sqrt_t_over_m[s] * mirror_geom / (2.0 * (b_hat**2))).reshape((-1,))
            mirror_diag_by_s.append(sp.diags(mirror_factor_flat, 0, format="csr"))

        block_size_max = int(n_l * n_tz)
        # Store one block per (species,x), matching F-block layout.
        n_blocks = int(n_species) * int(n_x)
        inv_perm_r_blocks: list[np.ndarray | None] = [None] * n_blocks
        perm_c_blocks: list[np.ndarray | None] = [None] * n_blocks
        lower_idx_blocks: list[np.ndarray | None] = [None] * n_blocks
        lower_val_blocks: list[np.ndarray | None] = [None] * n_blocks
        upper_idx_blocks: list[np.ndarray | None] = [None] * n_blocks
        upper_val_blocks: list[np.ndarray | None] = [None] * n_blocks
        upper_diag_blocks: list[np.ndarray | None] = [None] * n_blocks
        max_lower_global = 0
        max_upper_global = 0

        def _factor_one(args: tuple[int, int]) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            s, ix = args
            stream_tz = stream_op_by_s[s]
            mirror_diag = mirror_diag_by_s[s]

            n_lx = int(nxi_for_x[ix])
            active_n = int(n_lx * n_tz)

            # Default: identity for the whole block (used when active_n==0 or ILU fails).
            perm_r_full = np.arange(block_size_max, dtype=np.int32)
            perm_c_full = np.arange(block_size_max, dtype=np.int32)
            inv_perm_r_full = perm_r_full.copy()
            lower_idx_full = np.zeros((block_size_max, 0), dtype=np.int32)
            lower_val_full = np.zeros((block_size_max, 0), dtype=np.float64)
            upper_idx_full = np.zeros((block_size_max, 0), dtype=np.int32)
            upper_val_full = np.zeros((block_size_max, 0), dtype=np.float64)
            upper_diag_full = np.ones((block_size_max,), dtype=np.float64)

            if active_n > 0:
                # L-coupling coefficients for this x.
                ell = np.arange(n_lx, dtype=np.float64)
                coef_plus = (ell + 1.0) / (2.0 * ell + 3.0)
                coef_minus = np.where(ell > 0, ell / (2.0 * ell - 1.0), 0.0)
                coef_m_plus = (ell + 1.0) * (ell + 2.0) / (2.0 * ell + 3.0)
                coef_m_minus = np.where(ell > 1, -ell * (ell - 1.0) / (2.0 * ell - 1.0), 0.0)
                coef_plus *= x_arr[ix]
                coef_minus *= x_arr[ix]
                coef_m_plus *= x_arr[ix]
                coef_m_minus *= x_arr[ix]

                # Assemble sparse block matrix for the active L-modes.
                #
                # Previous implementations used a Python nested list + `sp.bmat`, which
                # can be surprisingly expensive. The structure here is block-tridiagonal
                # in L, so build it with Kronecker products.
                diag_vals = identity_shift + np.asarray(pas_coef[s, ix, :n_lx], dtype=np.float64) + float(reg)
                a = sp.kron(sp.diags(diag_vals, 0, format="csr"), i_tz, format="csc")
                if exb_op_tz is not None:
                    a = a + sp.kron(sp.eye(n_lx, format="csr", dtype=np.float64), exb_op_tz, format="csc")
                if n_lx > 1:
                    c_stream = sp.diags(
                        [coef_minus[1:], coef_plus[:-1]],
                        offsets=[-1, 1],
                        shape=(n_lx, n_lx),
                        format="csr",
                        dtype=np.float64,
                    )
                    c_mirror = sp.diags(
                        [coef_m_minus[1:], coef_m_plus[:-1]],
                        offsets=[-1, 1],
                        shape=(n_lx, n_lx),
                        format="csr",
                        dtype=np.float64,
                    )
                    a = a + sp.kron(c_stream, stream_tz, format="csc") + sp.kron(c_mirror, mirror_diag, format="csc")
                a = a.tocsc()

                # Prefer ILU for most PAS blocks to avoid SuperLU exact-LU peak allocations
                # and the resulting resident-set spikes. Fall back to exact LU only if ILU
                # fails to factor (e.g. due to numerical issues) and the block is small enough.
                fac = None
                used_lu = False
                try:
                    # ILU (PETSc PCILU analogue).
                    fac = spilu(
                        a,
                        drop_tol=ilu_drop_tol,
                        fill_factor=ilu_fill_factor,
                        permc_spec="COLAMD",
                    )
                except Exception:  # noqa: BLE001
                    fac = None
                if fac is None and lu_max > 0 and int(active_n) <= int(lu_max):
                    # Exact sparse LU (PETSc PCLU analogue) fallback for robustness.
                    try:
                        fac = splu(a, permc_spec="COLAMD")
                        used_lu = True
                    except Exception:  # noqa: BLE001
                        fac = None

                if fac is not None:
                    perm_r = np.asarray(fac.perm_r, dtype=np.int32)
                    perm_c = np.asarray(fac.perm_c, dtype=np.int32)

                    l_csr = fac.L.tocsr()
                    u_csr = fac.U.tocsr()
                    n = int(active_n)
                    # Truncating exact LU factors can severely degrade preconditioner quality.
                    # For LU factors, allow more fill than the ILU cap, but still cap to
                    # avoid dense padded-row allocations.
                    row_nnz_eff = int(row_nnz_max)
                    if used_lu and lu_full_row_nnz_max > 0:
                        row_nnz_eff = max(int(row_nnz_eff), min(int(n), int(lu_full_row_nnz_max)))

                    # Strict lower-triangular part (unit diagonal assumed).
                    lower_cols: list[np.ndarray] = []
                    lower_vals: list[np.ndarray] = []
                    max_lower = 0
                    for i in range(n):
                        rs = int(l_csr.indptr[i])
                        re = int(l_csr.indptr[i + 1])
                        cols = l_csr.indices[rs:re]
                        vals = l_csr.data[rs:re]
                        mask = cols < i
                        cols = cols[mask].astype(np.int32, copy=False)
                        vals = vals[mask].astype(np.float64, copy=False)
                        if row_nnz_eff > 0 and int(cols.size) > row_nnz_eff:
                            sel = np.argpartition(np.abs(vals), -row_nnz_eff)[-row_nnz_eff:]
                            cols = cols[sel]
                            vals = vals[sel]
                        if cols.size:
                            order = np.argsort(cols)
                            cols = cols[order]
                            vals = vals[order]
                        lower_cols.append(cols)
                        lower_vals.append(vals)
                        max_lower = max(max_lower, int(cols.size))
                    lower_idx = -np.ones((n, max_lower), dtype=np.int32)
                    lower_val = np.zeros((n, max_lower), dtype=np.float64)
                    for i in range(n):
                        k = int(lower_cols[i].size)
                        if k:
                            lower_idx[i, :k] = lower_cols[i]
                            lower_val[i, :k] = lower_vals[i]

                    # Strict upper + diagonal.
                    upper_cols: list[np.ndarray] = []
                    upper_vals: list[np.ndarray] = []
                    upper_diag = np.ones((n,), dtype=np.float64)
                    max_upper = 0
                    for i in range(n):
                        rs = int(u_csr.indptr[i])
                        re = int(u_csr.indptr[i + 1])
                        cols = u_csr.indices[rs:re]
                        vals = u_csr.data[rs:re]
                        diag_mask = cols == i
                        if np.any(diag_mask):
                            upper_diag[i] = float(vals[diag_mask][0])
                        mask = cols > i
                        cols_u = cols[mask].astype(np.int32, copy=False)
                        vals_u = vals[mask].astype(np.float64, copy=False)
                        if row_nnz_eff > 0 and int(cols_u.size) > row_nnz_eff:
                            sel = np.argpartition(np.abs(vals_u), -row_nnz_eff)[-row_nnz_eff:]
                            cols_u = cols_u[sel]
                            vals_u = vals_u[sel]
                        if cols_u.size:
                            order = np.argsort(cols_u)
                            cols_u = cols_u[order]
                            vals_u = vals_u[order]
                        upper_cols.append(cols_u)
                        upper_vals.append(vals_u)
                        max_upper = max(max_upper, int(cols_u.size))
                    upper_idx = -np.ones((n, max_upper), dtype=np.int32)
                    upper_val = np.zeros((n, max_upper), dtype=np.float64)
                    for i in range(n):
                        k = int(upper_cols[i].size)
                        if k:
                            upper_idx[i, :k] = upper_cols[i]
                            upper_val[i, :k] = upper_vals[i]

                    bad = ~np.isfinite(upper_diag) | (upper_diag == 0.0)
                    if np.any(bad):
                        upper_diag[bad] = 1.0

                    # Pad to the full x-block size by appending identity rows/cols.
                    perm_r_full[:n] = perm_r
                    perm_c_full[:n] = perm_c
                    inv_perm_r_full = inverse_permutation(perm_r_full)

                    lower_idx_full = -np.ones((block_size_max, max_lower), dtype=np.int32)
                    lower_val_full = np.zeros((block_size_max, max_lower), dtype=np.float64)
                    lower_idx_full[:n, :] = lower_idx
                    lower_val_full[:n, :] = lower_val

                    upper_idx_full = -np.ones((block_size_max, max_upper), dtype=np.int32)
                    upper_val_full = np.zeros((block_size_max, max_upper), dtype=np.float64)
                    upper_idx_full[:n, :] = upper_idx
                    upper_val_full[:n, :] = upper_val

                    upper_diag_full = np.ones((block_size_max,), dtype=np.float64)
                    upper_diag_full[:n] = upper_diag

            block_idx = int(s) * int(n_x) + int(ix)
            return (
                block_idx,
                inv_perm_r_full,
                perm_c_full,
                lower_idx_full,
                lower_val_full,
                upper_idx_full,
                upper_val_full,
                upper_diag_full,
            )

        args_list = [(s, ix) for s in range(n_species) for ix in range(n_x)]
        threads_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_ILU_THREADS", "").strip()
        try:
            threads_req = int(threads_env) if threads_env else 0
        except ValueError:
            threads_req = 0
        if threads_req > 0:
            n_workers = min(int(threads_req), int(n_blocks))
        else:
            # Auto: ILU build is often the dominant cost in PAS DKES cases. Parallelize across
            # independent (species,x) blocks, but cap workers to avoid oversubscribing CPU.
            n_cpu = os.cpu_count() or 1
            n_workers = min(int(n_blocks), max(1, int(n_cpu)), 8)

        if n_workers > 1 and n_blocks > 1:
            from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

            with ThreadPoolExecutor(max_workers=int(n_workers)) as ex:
                for (
                    block_idx,
                    inv_perm_r_full,
                    perm_c_full,
                    lower_idx_full,
                    lower_val_full,
                    upper_idx_full,
                    upper_val_full,
                    upper_diag_full,
                ) in ex.map(_factor_one, args_list):
                    inv_perm_r_blocks[block_idx] = inv_perm_r_full
                    perm_c_blocks[block_idx] = perm_c_full
                    lower_idx_blocks[block_idx] = lower_idx_full
                    lower_val_blocks[block_idx] = lower_val_full
                    upper_idx_blocks[block_idx] = upper_idx_full
                    upper_val_blocks[block_idx] = upper_val_full
                    upper_diag_blocks[block_idx] = upper_diag_full
                    max_lower_global = max(max_lower_global, int(lower_idx_full.shape[1]))
                    max_upper_global = max(max_upper_global, int(upper_idx_full.shape[1]))
        else:
            for out in map(_factor_one, args_list):
                (
                    block_idx,
                    inv_perm_r_full,
                    perm_c_full,
                    lower_idx_full,
                    lower_val_full,
                    upper_idx_full,
                    upper_val_full,
                    upper_diag_full,
                ) = out
                inv_perm_r_blocks[block_idx] = inv_perm_r_full
                perm_c_blocks[block_idx] = perm_c_full
                lower_idx_blocks[block_idx] = lower_idx_full
                lower_val_blocks[block_idx] = lower_val_full
                upper_idx_blocks[block_idx] = upper_idx_full
                upper_val_blocks[block_idx] = upper_val_full
                upper_diag_blocks[block_idx] = upper_diag_full
                max_lower_global = max(max_lower_global, int(lower_idx_full.shape[1]))
                max_upper_global = max(max_upper_global, int(upper_idx_full.shape[1]))

        assert all(v is not None for v in inv_perm_r_blocks)
        assert all(v is not None for v in perm_c_blocks)
        assert all(v is not None for v in lower_idx_blocks)
        assert all(v is not None for v in lower_val_blocks)
        assert all(v is not None for v in upper_idx_blocks)
        assert all(v is not None for v in upper_val_blocks)
        assert all(v is not None for v in upper_diag_blocks)

        # Finalize uniform (Klower,Kupper) padding.
        for i in range(len(lower_idx_blocks)):
            lower_idx_block = lower_idx_blocks[i]
            lower_val_block = lower_val_blocks[i]
            upper_idx_block = upper_idx_blocks[i]
            upper_val_block = upper_val_blocks[i]
            assert lower_idx_block is not None
            assert lower_val_block is not None
            assert upper_idx_block is not None
            assert upper_val_block is not None

            if int(lower_idx_block.shape[1]) < max_lower_global:
                pad = int(max_lower_global - int(lower_idx_block.shape[1]))
                lower_idx_block = np.pad(lower_idx_block, ((0, 0), (0, pad)), constant_values=-1)
                lower_val_block = np.pad(lower_val_block, ((0, 0), (0, pad)), constant_values=0.0)
            if int(upper_idx_block.shape[1]) < max_upper_global:
                pad = int(max_upper_global - int(upper_idx_block.shape[1]))
                upper_idx_block = np.pad(upper_idx_block, ((0, 0), (0, pad)), constant_values=-1)
                upper_val_block = np.pad(upper_val_block, ((0, 0), (0, pad)), constant_values=0.0)

            lower_idx_blocks[i] = lower_idx_block
            lower_val_blocks[i] = lower_val_block
            upper_idx_blocks[i] = upper_idx_block
            upper_val_blocks[i] = upper_val_block

        inv_perm_r_np = np.stack(inv_perm_r_blocks, axis=0).reshape((n_species, n_x, block_size_max))  # type: ignore[arg-type]
        perm_c_np = np.stack(perm_c_blocks, axis=0).reshape((n_species, n_x, block_size_max))  # type: ignore[arg-type]
        lower_idx_np = np.stack(lower_idx_blocks, axis=0).reshape(  # type: ignore[arg-type]
            (n_species, n_x, block_size_max, max_lower_global)
        )
        lower_val_np = np.stack(lower_val_blocks, axis=0).reshape(  # type: ignore[arg-type]
            (n_species, n_x, block_size_max, max_lower_global)
        )
        upper_idx_np = np.stack(upper_idx_blocks, axis=0).reshape(  # type: ignore[arg-type]
            (n_species, n_x, block_size_max, max_upper_global)
        )
        upper_val_np = np.stack(upper_val_blocks, axis=0).reshape(  # type: ignore[arg-type]
            (n_species, n_x, block_size_max, max_upper_global)
        )
        upper_diag_np = np.stack(upper_diag_blocks, axis=0).reshape((n_species, n_x, block_size_max))  # type: ignore[arg-type]

        # Extra (constraint) block: keep the existing dense inverse on the small extra subsystem.
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
            ee = ee + float(reg) * np.eye(extra_size, dtype=np.float64)
            try:
                ee_inv = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(ee_inv)):
                ee_inv = np.linalg.pinv(ee, rcond=1e-12)
            extra_inv_jnp = jnp.asarray(ee_inv, dtype=jnp.float64)

        cached = _RHSMode1ILUBlockPrecondCache(
            inv_perm_r_sx=jnp.asarray(inv_perm_r_np, dtype=jnp.int32),
            perm_c_sx=jnp.asarray(perm_c_np, dtype=jnp.int32),
            lower_idx_sx=jnp.asarray(lower_idx_np, dtype=jnp.int32),
            lower_val_sx=jnp.asarray(lower_val_np, dtype=jnp.float64),
            upper_idx_sx=jnp.asarray(upper_idx_np, dtype=jnp.int32),
            upper_val_sx=jnp.asarray(upper_val_np, dtype=jnp.float64),
            upper_diag_sx=jnp.asarray(upper_diag_np, dtype=jnp.float64),
            extra_idx_jnp=extra_idx_jnp,
            extra_inv_jnp=extra_inv_jnp,
        )
        _RHSMODE1_PRECOND_ILU_CACHE[cache_key] = cached

    inv_perm_r_sx = cached.inv_perm_r_sx
    perm_c_sx = cached.perm_c_sx
    lower_idx_sx = cached.lower_idx_sx
    lower_val_sx = cached.lower_val_sx
    upper_idx_sx = cached.upper_idx_sx
    upper_val_sx = cached.upper_val_sx
    upper_diag_sx = cached.upper_diag_sx
    extra_idx_jnp = cached.extra_idx_jnp
    extra_inv_jnp = cached.extra_inv_jnp

    f_size = int(op.f_size)
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    block_size_max = int(inv_perm_r_sx.shape[-1])

    def _solve_block(
        r: jnp.ndarray,
        inv_perm_r: jnp.ndarray,
        perm_c: jnp.ndarray,
        lower_idx: jnp.ndarray,
        lower_val: jnp.ndarray,
        upper_idx: jnp.ndarray,
        upper_val: jnp.ndarray,
        upper_diag: jnp.ndarray,
    ) -> jnp.ndarray:
        r_perm = r[inv_perm_r]
        y = triangular_solve_lower_padded(lower_idx=lower_idx, lower_val=lower_val, b=r_perm)
        z = triangular_solve_upper_padded(
            upper_idx=upper_idx, upper_val=upper_val, upper_diag=upper_diag, b=y
        )
        return z[perm_c]

    _solve_over_x = jax.vmap(
        _solve_block,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )
    _solve_over_sx = jax.vmap(
        _solve_over_x,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0),
        out_axes=0,
    )

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_full = jnp.asarray(r_full, dtype=jnp.float64)
        r_f = r_full[:f_size].reshape((n_species, n_x, block_size_max))
        z_f = _solve_over_sx(
            r_f,
            inv_perm_r_sx,
            perm_c_sx,
            lower_idx_sx,
            lower_val_sx,
            upper_idx_sx,
            upper_val_sx,
            upper_diag_sx,
        )
        z_full = jnp.concatenate([z_f.reshape((-1,)), r_full[f_size:]], axis=0)
        if extra_inv_jnp is not None:
            r_extra = r_full[extra_idx_jnp]
            z_extra = extra_inv_jnp @ r_extra
            z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
        return jnp.asarray(z_full, dtype=jnp.float64)

    apply_full_safe = safe_preconditioner(_apply_full)
    # Expose ILU factor data for higher-level preconditioners (e.g. Schur) that can
    # exploit the per-(species,x) block structure without repeatedly applying the
    # full block-Jacobi operator during preconditioner setup.
    try:  # pragma: no cover - best-effort metadata for performance
        setattr(
            apply_full_safe,
            "_sfincs_pas_ilu_factors",
            (
                inv_perm_r_sx,
                perm_c_sx,
                lower_idx_sx,
                lower_val_sx,
                upper_idx_sx,
                upper_val_sx,
                upper_diag_sx,
            ),
        )
        setattr(apply_full_safe, "_sfincs_pas_ilu_block_size_max", int(block_size_max))
    except Exception:
        pass

    if reduce_full is None or expand_reduced is None:
        return apply_full_safe

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full_safe(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced
