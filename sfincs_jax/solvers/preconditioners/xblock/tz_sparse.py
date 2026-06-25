"""Sparse x-block RHSMode=1 preconditioner setup."""

from __future__ import annotations

from collections.abc import Callable
import os

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.xblock import policy as _rhs1_xblock_policy
from sfincs_jax.solvers.preconditioners.xblock import policy as _rhs1_xblock_sparse_host_policy
from ....preconditioner_context import precond_dtype, sparse_structural_tol
from ....preconditioner_caches import (
    _RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE,
    _RHSMODE1_SPARSE_ILU_CACHE,
    _RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE,
    _RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE,
    _RHSMode1FPXBlockAssembledHostCache,
    _RHSMode1SparseSXBlockHostPrecondCache,
    _RHSMode1SparseXBlockCSRPrecondCache,
    _RHSMode1SparseXBlockHostPrecondCache,
    _RHSMode1SparseXBlockPrecondCache,
)
from ....preconditioner_setup import (
    matvec_submatrix_v3_unsharded,
    precond_chunk_cols,
    rhs_mode1_precond_cache_key,
)
from ....problems.profile_response.residual import safe_preconditioner
from sfincs_jax.problems.profile_response.large_cpu_policy import (
    rhs1_fp_xblock_assembled_host_allowed as _rhs1_fp_xblock_assembled_host_allowed,
)
from ....problems.profile_response.policies import (
    rhs1_host_factor_probe_ok as _rhs1_host_factor_probe_ok,
)
from sfincs_jax.problems.profile_response.solver_policy import read_bool_env as _rhs1_bool_env
from sfincs_jax.problems.profile_response.solver_policy import read_float_env as _rhs1_float_env
from ....sparse_triangular import (
    triangular_solve_lower_csr_rows as _triangular_solve_lower_csr_rows,
    triangular_solve_lower_padded as _triangular_solve_lower_padded,
    triangular_solve_upper_csr_rows as _triangular_solve_upper_csr_rows,
    triangular_solve_upper_padded as _triangular_solve_upper_padded,
)
from ..symbolic_sparse import build_sparse_ilu_from_matvec, factorize_sparse_matrix_csr_host
from ....v3_system import V3FullSystemOperator, apply_v3_full_system_operator_cached

__all__ = [
    "assemble_rhsmode1_fp_xblock_tz_sparse_matrix",
    "assemble_selected_theta_tz_operator",
    "assemble_selected_zeta_tz_operator",
    "build_rhs1_sxblock_tz_sparse_host_preconditioner",
    "compute_rhs1_sxblock_tz_sparse_host_seed",
    "build_rhs1_xblock_tz_sparse_preconditioner",
    "get_rhsmode1_fp_xblock_assembled_host_cache",
    "rhsmode1_fp_xblock_assembled_host_allowed",
    "rhsmode1_fp_xblock_species_decoupled_for_host_assembly",
    "rhsmode1_fp_xblock_tz_sparse_diagonal",
    "rhsmode1_host_factor_probe_ok",
    "rhsmode1_precond_cache_key",
    "rhsmode1_xblock_sparse_lu_default_max",
    "safe_inverse_diagonal_np",
]


def assemble_selected_theta_tz_operator(
    *, dd_plus: np.ndarray, dd_minus: np.ndarray, use_plus: np.ndarray
):
    """Assemble a theta-upwind derivative over flattened ``(theta, zeta)`` rows."""

    import scipy.sparse as sp  # noqa: PLC0415

    n_theta, n_zeta = use_plus.shape
    n_tz = int(n_theta * n_zeta)
    struct_tol = sparse_structural_tol()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for it in range(n_theta):
        row_plus = np.asarray(dd_plus[it, :], dtype=np.float64)
        row_minus = np.asarray(dd_minus[it, :], dtype=np.float64)
        if struct_tol > 0.0:
            row_plus = row_plus.copy()
            row_minus = row_minus.copy()
            row_plus[np.abs(row_plus) <= struct_tol] = 0.0
            row_minus[np.abs(row_minus) <= struct_tol] = 0.0
        nz_plus = np.flatnonzero(row_plus)
        nz_minus = np.flatnonzero(row_minus)
        for iz in range(n_zeta):
            row = int(it * n_zeta + iz)
            nz = nz_plus if bool(use_plus[it, iz]) else nz_minus
            row_vals = row_plus if bool(use_plus[it, iz]) else row_minus
            if int(nz.size) == 0:
                continue
            rows.append(np.full((int(nz.size),), row, dtype=np.int32))
            cols.append((nz.astype(np.int32, copy=False) * int(n_zeta)) + int(iz))
            data.append(np.asarray(row_vals[nz], dtype=np.float64))
    if not data:
        return sp.csr_matrix((n_tz, n_tz), dtype=np.float64)
    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(data)
    return sp.csr_matrix((values, (row_idx, col_idx)), shape=(n_tz, n_tz), dtype=np.float64)


def assemble_selected_zeta_tz_operator(
    *, dd_plus: np.ndarray, dd_minus: np.ndarray, use_plus: np.ndarray
):
    """Assemble a zeta-upwind derivative over flattened ``(theta, zeta)`` rows."""

    import scipy.sparse as sp  # noqa: PLC0415

    n_theta, n_zeta = use_plus.shape
    n_tz = int(n_theta * n_zeta)
    struct_tol = sparse_structural_tol()
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    for it in range(n_theta):
        for iz in range(n_zeta):
            row = int(it * n_zeta + iz)
            row_vals = np.asarray(
                dd_plus[iz, :] if bool(use_plus[it, iz]) else dd_minus[iz, :],
                dtype=np.float64,
            )
            if struct_tol > 0.0:
                row_vals = row_vals.copy()
                row_vals[np.abs(row_vals) <= struct_tol] = 0.0
            nz = np.flatnonzero(row_vals)
            if int(nz.size) == 0:
                continue
            rows.append(np.full((int(nz.size),), row, dtype=np.int32))
            cols.append((int(it) * int(n_zeta)) + nz.astype(np.int32, copy=False))
            data.append(np.asarray(row_vals[nz], dtype=np.float64))
    if not data:
        return sp.csr_matrix((n_tz, n_tz), dtype=np.float64)
    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(data)
    return sp.csr_matrix((values, (row_idx, col_idx)), shape=(n_tz, n_tz), dtype=np.float64)


def safe_inverse_diagonal_np(diagonal: np.ndarray, *, floor: float) -> np.ndarray | None:
    """Return a finite inverse diagonal or ``None`` if the diagonal is unusable."""

    diag = np.asarray(diagonal, dtype=np.float64).reshape((-1,))
    if diag.size == 0 or not np.all(np.isfinite(diag)):
        return None
    floor_use = max(0.0, float(floor))
    if floor_use > 0.0:
        sign = np.where(diag < 0.0, -1.0, 1.0)
        diag = np.where(np.abs(diag) > floor_use, diag, sign * floor_use)
    elif np.any(diag == 0.0):
        return None
    inv = 1.0 / diag
    if not np.all(np.isfinite(inv)):
        return None
    return np.asarray(inv, dtype=np.float64)


def get_rhsmode1_fp_xblock_assembled_host_cache(*, op: V3FullSystemOperator) -> _RHSMode1FPXBlockAssembledHostCache:
    """Return cached host-side pieces for explicit FP x-block sparse assembly."""

    import scipy.sparse as sp  # noqa: PLC0415

    cache_key = rhs_mode1_precond_cache_key(
        op,
        "fp_xblock_assembled_host",
        precond_dtype=precond_dtype(),
    )
    cached = _RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    colless = op.fblock.collisionless
    fp = op.fblock.fp
    if colless is None or fp is None:
        raise ValueError("assembled FP x-block host cache requires collisionless and FP operators")

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_tz = int(n_theta * n_zeta)

    x = np.asarray(colless.x, dtype=np.float64).reshape((n_x,))
    z_s = np.asarray(op.z_s, dtype=np.float64).reshape((n_species,))
    identity_shift = float(np.asarray(op.fblock.identity_shift, dtype=np.float64).reshape(()))
    fp_diag_sxl = np.zeros((n_species, n_x, n_l), dtype=np.float64)
    for s in range(n_species):
        for ix in range(n_x):
            fp_diag_sxl[s, ix, :] = np.asarray(fp.mat[s, s, :, ix, ix], dtype=np.float64)

    struct_tol = sparse_structural_tol()
    ddtheta = np.asarray(colless.ddtheta, dtype=np.float64)
    ddzeta = np.asarray(colless.ddzeta, dtype=np.float64)
    if struct_tol > 0.0:
        ddtheta = ddtheta.copy()
        ddzeta = ddzeta.copy()
        ddtheta[np.abs(ddtheta) <= struct_tol] = 0.0
        ddzeta[np.abs(ddzeta) <= struct_tol] = 0.0
    n_tz_eye = sp.eye(n_tz, format="csr", dtype=np.float64)
    dtheta_tz = sp.kron(sp.csr_matrix(ddtheta), sp.eye(n_zeta, format="csr", dtype=np.float64), format="csr")
    dzeta_tz = sp.kron(sp.eye(n_theta, format="csr", dtype=np.float64), sp.csr_matrix(ddzeta), format="csr")

    b_hat = np.asarray(colless.b_hat, dtype=np.float64)
    b_hat_sup_theta = np.asarray(colless.b_hat_sup_theta, dtype=np.float64)
    b_hat_sup_zeta = np.asarray(colless.b_hat_sup_zeta, dtype=np.float64)
    db_hat_dtheta = np.asarray(colless.db_hat_dtheta, dtype=np.float64)
    db_hat_dzeta = np.asarray(colless.db_hat_dzeta, dtype=np.float64)
    t_hats = np.asarray(colless.t_hats, dtype=np.float64).reshape((n_species,))
    m_hats = np.asarray(colless.m_hats, dtype=np.float64).reshape((n_species,))
    sqrt_t_over_m = np.sqrt(t_hats / m_hats)

    stream_tz_by_species: list[object] = []
    mirror_diag_by_species: list[object] = []
    v_theta = b_hat_sup_theta / b_hat
    v_zeta = b_hat_sup_zeta / b_hat
    mirror_geom = b_hat_sup_theta * db_hat_dtheta + b_hat_sup_zeta * db_hat_dzeta
    for s in range(n_species):
        stream_tz = sp.diags((sqrt_t_over_m[s] * v_theta).reshape((-1,)), 0, format="csr") @ dtheta_tz
        stream_tz = stream_tz + (sp.diags((sqrt_t_over_m[s] * v_zeta).reshape((-1,)), 0, format="csr") @ dzeta_tz)
        mirror_factor = (-sqrt_t_over_m[s] * mirror_geom / (2.0 * (b_hat**2))).reshape((-1,))
        stream_tz_by_species.append(stream_tz.tocsr())
        mirror_diag_by_species.append(sp.diags(mirror_factor, 0, format="csr"))

    exb_op_tz = None
    exb_theta = op.fblock.exb_theta
    exb_zeta = op.fblock.exb_zeta
    if exb_theta is not None:
        denom = (
            float(np.asarray(exb_theta.fsab_hat2, dtype=np.float64).reshape(()))
            if bool(getattr(exb_theta, "use_dkes_exb_drift", False))
            else None
        )
        coef = np.asarray(exb_theta.d_hat * exb_theta.b_hat_sub_zeta, dtype=np.float64)
        coef = coef / denom if denom is not None else coef / (np.asarray(exb_theta.b_hat, dtype=np.float64) ** 2)
        factor = float(
            np.asarray(
                exb_theta.alpha * exb_theta.delta * 0.5 * exb_theta.dphi_hat_dpsi_hat,
                dtype=np.float64,
            ).reshape(())
        )
        exb_op_tz = sp.diags((factor * coef).reshape((-1,)), 0, format="csr") @ dtheta_tz
    if exb_zeta is not None:
        denom = (
            float(np.asarray(exb_zeta.fsab_hat2, dtype=np.float64).reshape(()))
            if bool(getattr(exb_zeta, "use_dkes_exb_drift", False))
            else None
        )
        coef = np.asarray(exb_zeta.d_hat * exb_zeta.b_hat_sub_theta, dtype=np.float64)
        coef = coef / denom if denom is not None else coef / (np.asarray(exb_zeta.b_hat, dtype=np.float64) ** 2)
        factor = float(
            np.asarray(
                -exb_zeta.alpha * exb_zeta.delta * 0.5 * exb_zeta.dphi_hat_dpsi_hat,
                dtype=np.float64,
            ).reshape(())
        )
        exb_term = sp.diags((factor * coef).reshape((-1,)), 0, format="csr") @ dzeta_tz
        exb_op_tz = exb_term if exb_op_tz is None else (exb_op_tz + exb_term)
    if exb_op_tz is not None:
        exb_op_tz = exb_op_tz.tocsr()

    mag_theta_m1_tz_by_species = None
    mag_theta_m2_tz_by_species = None
    mag_theta = op.fblock.magdrift_theta
    if mag_theta is not None:
        gf1 = np.asarray(
            mag_theta.b_hat_sub_zeta * mag_theta.db_hat_dpsi_hat - mag_theta.b_hat_sub_psi * mag_theta.db_hat_dzeta,
            dtype=np.float64,
        )
        gf2 = np.asarray(
            2.0 * mag_theta.b_hat * (mag_theta.db_hat_sub_psi_dzeta - mag_theta.db_hat_sub_zeta_dpsi_hat),
            dtype=np.float64,
        )
        base = np.asarray(
            mag_theta.delta * mag_theta.t_hat * mag_theta.d_hat / (2.0 * mag_theta.z * (mag_theta.b_hat**3)),
            dtype=np.float64,
        )
        d_hat_ref = float(np.asarray(mag_theta.d_hat, dtype=np.float64).reshape((-1,))[0])
        mag_theta_m1_tz_by_species = []
        mag_theta_m2_tz_by_species = []
        for s in range(n_species):
            use_plus = (gf1 * d_hat_ref / float(z_s[s])) > 0.0
            dtheta_used_tz = assemble_selected_theta_tz_operator(
                dd_plus=np.asarray(mag_theta.ddtheta_plus, dtype=np.float64),
                dd_minus=np.asarray(mag_theta.ddtheta_minus, dtype=np.float64),
                use_plus=np.asarray(use_plus, dtype=bool),
            )
            mag_theta_m1_tz_by_species.append(
                (sp.diags((base * gf1).reshape((-1,)), 0, format="csr") @ dtheta_used_tz).tocsr()
            )
            mag_theta_m2_tz_by_species.append(
                (sp.diags((base * gf2).reshape((-1,)), 0, format="csr") @ dtheta_used_tz).tocsr()
            )
        mag_theta_m1_tz_by_species = tuple(mag_theta_m1_tz_by_species)
        mag_theta_m2_tz_by_species = tuple(mag_theta_m2_tz_by_species)

    mag_zeta_m1_tz_by_species = None
    mag_zeta_m2_tz_by_species = None
    mag_zeta = op.fblock.magdrift_zeta
    if mag_zeta is not None:
        gf1 = np.asarray(
            mag_zeta.b_hat_sub_psi * mag_zeta.db_hat_dtheta - mag_zeta.b_hat_sub_theta * mag_zeta.db_hat_dpsi_hat,
            dtype=np.float64,
        )
        gf2 = np.asarray(
            2.0 * mag_zeta.b_hat * (mag_zeta.db_hat_sub_theta_dpsi_hat - mag_zeta.db_hat_sub_psi_dtheta),
            dtype=np.float64,
        )
        base = np.asarray(
            mag_zeta.delta * mag_zeta.t_hat * mag_zeta.d_hat / (2.0 * mag_zeta.z * (mag_zeta.b_hat**3)),
            dtype=np.float64,
        )
        d_hat_ref = float(np.asarray(mag_zeta.d_hat, dtype=np.float64).reshape((-1,))[0])
        mag_zeta_m1_tz_by_species = []
        mag_zeta_m2_tz_by_species = []
        for s in range(n_species):
            use_plus = (gf1 * d_hat_ref / float(z_s[s])) > 0.0
            dzeta_used_tz = assemble_selected_zeta_tz_operator(
                dd_plus=np.asarray(mag_zeta.ddzeta_plus, dtype=np.float64),
                dd_minus=np.asarray(mag_zeta.ddzeta_minus, dtype=np.float64),
                use_plus=np.asarray(use_plus, dtype=bool),
            )
            mag_zeta_m1_tz_by_species.append(
                (sp.diags((base * gf1).reshape((-1,)), 0, format="csr") @ dzeta_used_tz).tocsr()
            )
            mag_zeta_m2_tz_by_species.append(
                (sp.diags((base * gf2).reshape((-1,)), 0, format="csr") @ dzeta_used_tz).tocsr()
            )
        mag_zeta_m1_tz_by_species = tuple(mag_zeta_m1_tz_by_species)
        mag_zeta_m2_tz_by_species = tuple(mag_zeta_m2_tz_by_species)

    mag_xidot_factor_flat = None
    mag_xidot = op.fblock.magdrift_xidot
    if mag_xidot is not None:
        temp = np.asarray(
            (mag_xidot.db_hat_sub_psi_dzeta - mag_xidot.db_hat_sub_zeta_dpsi_hat) * mag_xidot.db_hat_dtheta
            + (mag_xidot.db_hat_sub_theta_dpsi_hat - mag_xidot.db_hat_sub_psi_dtheta) * mag_xidot.db_hat_dzeta,
            dtype=np.float64,
        )
        mag_xidot_factor_flat = np.asarray(
            (-(mag_xidot.delta * mag_xidot.t_hat) * mag_xidot.d_hat / (2.0 * mag_xidot.z * (mag_xidot.b_hat**3)) * temp),
            dtype=np.float64,
        ).reshape((-1,))

    er_xidot_factor_flat = None
    er_xidot = op.fblock.er_xidot
    if er_xidot is not None:
        temp = np.asarray(
            er_xidot.b_hat_sub_zeta * er_xidot.db_hat_dtheta - er_xidot.b_hat_sub_theta * er_xidot.db_hat_dzeta,
            dtype=np.float64,
        )
        er_xidot_factor_flat = np.asarray(
            er_xidot.alpha
            * er_xidot.delta
            * er_xidot.dphi_hat_dpsi_hat
            / (4.0 * (er_xidot.b_hat**3))
            * er_xidot.d_hat
            * temp,
            dtype=np.float64,
        ).reshape((-1,))

    er_xdot_factor_flat = None
    ddx_plus_diag = None
    ddx_minus_diag = None
    er_xdot = op.fblock.er_xdot
    if er_xdot is not None:
        factor0 = float(np.asarray(-(er_xdot.alpha * er_xdot.delta * er_xdot.dphi_hat_dpsi_hat) / 4.0, dtype=np.float64).reshape(()))
        er_xdot_factor_flat = np.asarray(
            factor0
            * er_xdot.d_hat
            / (er_xdot.b_hat**3)
            * (er_xdot.b_hat_sub_theta * er_xdot.db_hat_dzeta - er_xdot.b_hat_sub_zeta * er_xdot.db_hat_dtheta),
            dtype=np.float64,
        ).reshape((-1,))
        ddx_plus_diag = np.asarray(np.diag(np.asarray(er_xdot.ddx_plus, dtype=np.float64)), dtype=np.float64)
        ddx_minus_diag = np.asarray(np.diag(np.asarray(er_xdot.ddx_minus, dtype=np.float64)), dtype=np.float64)

    cached = _RHSMode1FPXBlockAssembledHostCache(
        x=x,
        z_s=z_s,
        fp_diag_sxl=fp_diag_sxl,
        n_tz_eye=n_tz_eye,
        stream_tz_by_species=tuple(stream_tz_by_species),
        mirror_diag_by_species=tuple(mirror_diag_by_species),
        exb_op_tz=exb_op_tz,
        mag_theta_m1_tz_by_species=mag_theta_m1_tz_by_species,
        mag_theta_m2_tz_by_species=mag_theta_m2_tz_by_species,
        mag_zeta_m1_tz_by_species=mag_zeta_m1_tz_by_species,
        mag_zeta_m2_tz_by_species=mag_zeta_m2_tz_by_species,
        mag_xidot_factor_flat=mag_xidot_factor_flat,
        er_xidot_factor_flat=er_xidot_factor_flat,
        er_xdot_factor_flat=er_xdot_factor_flat,
        ddx_plus_diag=ddx_plus_diag,
        ddx_minus_diag=ddx_minus_diag,
        identity_shift=identity_shift,
    )
    _RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE[cache_key] = cached
    return cached


def assemble_rhsmode1_fp_xblock_tz_sparse_matrix(
    *,
    op: V3FullSystemOperator,
    species: int,
    ix: int,
    preconditioner_xi: int,
    host_cache: _RHSMode1FPXBlockAssembledHostCache | None = None,
):
    """Assemble one explicit FP ``(theta,zeta,L)`` sparse x-block on the host."""

    import scipy.sparse as sp  # noqa: PLC0415

    if op.fblock.fp is None:
        raise ValueError("assembled FP x-block matrix requires an FP operator")
    if int(preconditioner_xi) != 1:
        raise ValueError("assembled FP x-block matrix currently requires preconditioner_xi=1")
    if bool(op.point_at_x0):
        raise ValueError("assembled FP x-block matrix currently requires pointAtX0=false")

    colless = op.fblock.collisionless
    host = host_cache if host_cache is not None else get_rhsmode1_fp_xblock_assembled_host_cache(op=op)
    mag_theta = op.fblock.magdrift_theta
    mag_zeta = op.fblock.magdrift_zeta

    n_lx = int(np.asarray(colless.n_xi_for_x, dtype=np.int32)[int(ix)])
    if n_lx <= 0:
        return sp.csc_matrix((0, 0), dtype=np.float64)

    x_val = float(host.x[int(ix)])

    diag_vals = float(host.identity_shift) + host.fp_diag_sxl[int(species), int(ix), :n_lx]
    a = sp.kron(sp.diags(diag_vals, 0, format="csr"), host.n_tz_eye, format="csc")
    stream_tz = host.stream_tz_by_species[int(species)]
    mirror_diag = host.mirror_diag_by_species[int(species)]

    ell = np.arange(n_lx, dtype=np.float64)
    coef_plus = x_val * (ell + 1.0) / (2.0 * ell + 3.0)
    coef_minus = np.where(ell > 0, x_val * ell / (2.0 * ell - 1.0), 0.0)
    coef_mirror_plus = x_val * (ell + 1.0) * (ell + 2.0) / (2.0 * ell + 3.0)
    coef_mirror_minus = np.where(ell > 1, -x_val * ell * (ell - 1.0) / (2.0 * ell - 1.0), 0.0)
    if n_lx > 1:
        c_stream = sp.diags(
            [coef_minus[1:], coef_plus[:-1]],
            offsets=[-1, 1],
            shape=(n_lx, n_lx),
            format="csr",
            dtype=np.float64,
        )
        c_mirror = sp.diags(
            [coef_mirror_minus[1:], coef_mirror_plus[:-1]],
            offsets=[-1, 1],
            shape=(n_lx, n_lx),
            format="csr",
            dtype=np.float64,
        )
        a = a + sp.kron(c_stream, stream_tz, format="csc") + sp.kron(c_mirror, mirror_diag, format="csc")

    if host.exb_op_tz is not None:
        a = a + sp.kron(sp.eye(n_lx, format="csr", dtype=np.float64), host.exb_op_tz, format="csc")

    if mag_theta is not None:
        assert host.mag_theta_m1_tz_by_species is not None
        assert host.mag_theta_m2_tz_by_species is not None
        m1 = (x_val * x_val) * host.mag_theta_m1_tz_by_species[int(species)]
        m2 = (x_val * x_val) * host.mag_theta_m2_tz_by_species[int(species)]
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom
        a = a + sp.kron(sp.diags(c1, 0, format="csr"), m1, format="csc")
        a = a + sp.kron(sp.diags(c2, 0, format="csr"), m2, format="csc")

    if mag_zeta is not None:
        assert host.mag_zeta_m1_tz_by_species is not None
        assert host.mag_zeta_m2_tz_by_species is not None
        m1 = (x_val * x_val) * host.mag_zeta_m1_tz_by_species[int(species)]
        m2 = (x_val * x_val) * host.mag_zeta_m2_tz_by_species[int(species)]
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom
        a = a + sp.kron(sp.diags(c1, 0, format="csr"), m1, format="csc")
        a = a + sp.kron(sp.diags(c2, 0, format="csr"), m2, format="csc")

    if host.mag_xidot_factor_flat is not None:
        diag_c = np.where(ell > 0, (ell + 1.0) * ell / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0)), 0.0)
        a = a + sp.kron(
            sp.diags(diag_c, 0, format="csr"),
            sp.diags((x_val * x_val * host.mag_xidot_factor_flat).reshape((-1,)), 0, format="csr"),
            format="csc",
        )

    if host.er_xidot_factor_flat is not None:
        diag_c = np.where(ell > 0, (ell + 1.0) * ell / ((2.0 * ell - 1.0) * (2.0 * ell + 3.0)), 0.0)
        a = a + sp.kron(
            sp.diags(diag_c, 0, format="csr"),
            sp.diags((x_val * x_val * host.er_xidot_factor_flat).reshape((-1,)), 0, format="csr"),
            format="csc",
        )

    if host.er_xdot_factor_flat is not None and host.ddx_plus_diag is not None and host.ddx_minus_diag is not None:
        use_plus = host.er_xdot_factor_flat > 0.0
        xdiag = x_val * np.where(use_plus, host.ddx_plus_diag[int(ix)], host.ddx_minus_diag[int(ix)])
        denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
        diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        xdot_diag = sp.diags((xdiag * host.er_xdot_factor_flat).reshape((-1,)), 0, format="csr")
        a = a + sp.kron(sp.diags(diag_coef, 0, format="csr"), xdot_diag, format="csc")

    a = a.tocsc()
    a.eliminate_zeros()
    return a


def rhsmode1_fp_xblock_tz_sparse_diagonal(
    *,
    op: V3FullSystemOperator,
    species: int,
    ix: int,
    preconditioner_xi: int,
    host_cache: _RHSMode1FPXBlockAssembledHostCache | None = None,
) -> np.ndarray:
    """Assemble only the diagonal of an explicit FP x-block preconditioner."""

    if op.fblock.fp is None:
        raise ValueError("assembled FP x-block diagonal requires an FP operator")
    if int(preconditioner_xi) != 1:
        raise ValueError("assembled FP x-block diagonal currently requires preconditioner_xi=1")
    if bool(op.point_at_x0):
        raise ValueError("assembled FP x-block diagonal currently requires pointAtX0=false")

    colless = op.fblock.collisionless
    host = host_cache if host_cache is not None else get_rhsmode1_fp_xblock_assembled_host_cache(op=op)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    n_tz = int(n_theta * n_zeta)
    n_lx = int(np.asarray(colless.n_xi_for_x, dtype=np.int32)[int(ix)])
    if n_lx <= 0:
        return np.zeros((0,), dtype=np.float64)

    def _matrix_diag(matrix: object) -> np.ndarray:
        if hasattr(matrix, "diagonal"):
            return np.asarray(matrix.diagonal(), dtype=np.float64).reshape((-1,))
        return np.asarray(np.diag(np.asarray(matrix, dtype=np.float64)), dtype=np.float64).reshape((-1,))

    x_val = float(host.x[int(ix)])
    ell = np.arange(n_lx, dtype=np.float64)
    diag = np.repeat(
        float(host.identity_shift) + host.fp_diag_sxl[int(species), int(ix), :n_lx],
        n_tz,
    ).astype(np.float64, copy=False)

    if host.exb_op_tz is not None:
        diag += np.tile(_matrix_diag(host.exb_op_tz), n_lx)

    denom = (2.0 * ell + 3.0) * (2.0 * ell - 1.0)
    if host.mag_theta_m1_tz_by_species is not None and host.mag_theta_m2_tz_by_species is not None:
        m1_diag = _matrix_diag(host.mag_theta_m1_tz_by_species[int(species)])
        m2_diag = _matrix_diag(host.mag_theta_m2_tz_by_species[int(species)])
        c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom
        diag += np.repeat((x_val * x_val) * c1, n_tz) * np.tile(m1_diag, n_lx)
        diag += np.repeat((x_val * x_val) * c2, n_tz) * np.tile(m2_diag, n_lx)

    if host.mag_zeta_m1_tz_by_species is not None and host.mag_zeta_m2_tz_by_species is not None:
        m1_diag = _matrix_diag(host.mag_zeta_m1_tz_by_species[int(species)])
        m2_diag = _matrix_diag(host.mag_zeta_m2_tz_by_species[int(species)])
        c1 = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        c2 = (2.0 * ell * ell + 2.0 * ell - 1.0) / denom
        diag += np.repeat((x_val * x_val) * c1, n_tz) * np.tile(m1_diag, n_lx)
        diag += np.repeat((x_val * x_val) * c2, n_tz) * np.tile(m2_diag, n_lx)

    diag_c = np.where(ell > 0, (ell + 1.0) * ell / denom, 0.0)
    if host.mag_xidot_factor_flat is not None:
        diag += np.repeat(x_val * x_val * diag_c, n_tz) * np.tile(host.mag_xidot_factor_flat, n_lx)
    if host.er_xidot_factor_flat is not None:
        diag += np.repeat(x_val * x_val * diag_c, n_tz) * np.tile(host.er_xidot_factor_flat, n_lx)

    if host.er_xdot_factor_flat is not None and host.ddx_plus_diag is not None and host.ddx_minus_diag is not None:
        use_plus = host.er_xdot_factor_flat > 0.0
        xdiag = x_val * np.where(use_plus, host.ddx_plus_diag[int(ix)], host.ddx_minus_diag[int(ix)])
        diag_coef = 2.0 * (3.0 * ell * ell + 3.0 * ell - 2.0) / denom
        diag += np.repeat(diag_coef, n_tz) * np.tile(xdiag * host.er_xdot_factor_flat, n_lx)

    return np.asarray(diag, dtype=np.float64)


def rhsmode1_fp_xblock_assembled_host_allowed(
    *,
    op: V3FullSystemOperator,
    preconditioner_species: int,
    preconditioner_xi: int,
    use_implicit: bool,
    active_size: int | None = None,
) -> bool:
    """Return whether this RHSMode=1 FP x-block can use host sparse assembly."""

    return _rhs1_fp_xblock_assembled_host_allowed(
        op=op,
        preconditioner_species=int(preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
        active_size=None if active_size is None else int(active_size),
    )


def rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
    *, op: V3FullSystemOperator, preconditioner_species: int
) -> bool:
    """Return whether x-block host assembly preserves the requested species coupling."""

    return _rhs1_xblock_sparse_host_policy.rhs1_fp_xblock_species_decoupled_for_host_assembly(
        n_species=int(getattr(op, "n_species", 0) or 0),
        preconditioner_species=int(preconditioner_species),
    )


def rhsmode1_xblock_sparse_lu_default_max(op: object, *, build_jax_factors: bool) -> int:
    """Return the default exact-LU cap for RHSMode=1 x-block sparse factors."""

    fblock = getattr(op, "fblock", None)
    return _rhs1_xblock_sparse_host_policy.rhs1_xblock_sparse_lu_default_max(
        has_fp=getattr(fblock, "fp", None) is not None,
        has_pas=getattr(fblock, "pas", None) is not None,
        build_jax_factors=bool(build_jax_factors),
    )


def rhsmode1_host_factor_probe_ok(*, factor: object | None, block_size: int) -> bool:
    """Return whether a host x-block factor solve passes the bounded probe."""

    return _rhs1_host_factor_probe_ok(factor=factor, block_size=int(block_size))


def rhsmode1_precond_cache_key(op: V3FullSystemOperator, kind: str) -> tuple[object, ...]:
    """Return the RHSMode=1 x-block preconditioner cache key."""

    return rhs_mode1_precond_cache_key(op, kind, precond_dtype=precond_dtype())


def _sxblock_active_indices_for_l(
    *,
    n_species: int,
    n_x: int,
    n_l: int,
    n_theta: int,
    n_zeta: int,
    nxi_for_x: np.ndarray,
    ell: int,
) -> np.ndarray:
    """Return flattened active f-block indices for one pitch mode across species/x."""

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


def build_rhs1_sxblock_tz_sparse_host_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the explicit sparse per-L species/x RHSMode=1 rescue preconditioner.

    The block layout keeps all species and active speed-grid points coupled for
    a single pitch mode over the full ``(theta,zeta)`` surface.  This is a host
    preconditioner used by explicit/non-autodiff rescue paths, so it can keep
    SciPy sparse factors in the cache while still returning a JAX-compatible
    callable to the driver.
    """

    cache_key = (
        *rhsmode1_precond_cache_key(op, "sxblock_tz_sparse_host"),
        float(drop_tol),
        float(drop_rel),
        float(ilu_drop_tol),
        float(fill_factor),
    )
    cached = _RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE.get(cache_key)
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    total_size = int(op.total_size)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    lu_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_SPARSE_LU_MAX", "").strip()
    try:
        lu_max = int(lu_max_env) if lu_max_env else 4000
    except ValueError:
        lu_max = 4000
    lu_max = max(0, int(lu_max))

    if cached is None:
        import scipy.sparse as sp  # noqa: PLC0415

        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        struct_tol = sparse_structural_tol()
        block_indices: list[np.ndarray] = []
        block_factors: list[object | None] = []
        for ell in range(n_l):
            rep_idx_np = _sxblock_active_indices_for_l(
                n_species=n_species,
                n_x=n_x,
                n_l=n_l,
                n_theta=n_theta,
                n_zeta=n_zeta,
                nxi_for_x=nxi_for_x,
                ell=ell,
            )
            if rep_idx_np.size == 0:
                continue
            block_cache_key = (cache_key, int(ell), int(rep_idx_np.size))
            exact_lu = int(rep_idx_np.size) <= int(lu_max)
            try:
                if emit is not None:
                    emit(
                        1,
                        "sxblock_sparse_host: assembling per-L species/x block "
                        f"(L={int(ell)} size={int(rep_idx_np.size)})",
                    )
                chunk_cols = precond_chunk_cols(total_size, int(rep_idx_np.shape[0]))
                y_sub = matvec_submatrix_v3_unsharded(
                    op,
                    col_idx=rep_idx_np,
                    row_idx=rep_idx_np,
                    total_size=total_size,
                    chunk_cols=chunk_cols,
                )
                a_block = np.asarray(y_sub.T, dtype=np.float64)
                if struct_tol > 0.0 and a_block.size:
                    a_block[np.abs(a_block) <= struct_tol] = 0.0
                a_csr = sp.csr_matrix(a_block)
                a_csr.eliminate_zeros()
                factorize_sparse_matrix_csr_host(
                    a_csr_full=a_csr,
                    cache_key=block_cache_key,
                    drop_tol=drop_tol,
                    drop_rel=drop_rel,
                    ilu_drop_tol=ilu_drop_tol,
                    fill_factor=fill_factor,
                    factorization="lu" if exact_lu else "ilu",
                    emit=emit,
                )
                fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(block_cache_key)
            except Exception as exc:  # noqa: BLE001
                fac_cache = None
                if emit is not None:
                    emit(
                        1,
                        "sxblock_sparse_host: factorization failed for block "
                        f"(L={int(ell)} size={int(rep_idx_np.size)}) "
                        f"({type(exc).__name__}: {exc})",
                    )
            host_factor = None if fac_cache is None else fac_cache.ilu
            if host_factor is not None and (
                not rhsmode1_host_factor_probe_ok(factor=host_factor, block_size=int(rep_idx_np.size))
            ):
                if emit is not None:
                    emit(
                        1,
                        "sxblock_sparse_host: rejecting unstable block factor "
                        f"(L={int(ell)} size={int(rep_idx_np.size)})",
                    )
                host_factor = None
            block_indices.append(rep_idx_np)
            block_factors.append(host_factor)

        extra_inv_np: np.ndarray | None = None
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
            ee = ee + np.float64(reg_val) * np.eye(extra_size, dtype=np.float64)
            try:
                extra_inv_np = np.linalg.inv(ee)
            except np.linalg.LinAlgError:
                extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)
            if not np.all(np.isfinite(extra_inv_np)):
                extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)

        cached = _RHSMode1SparseSXBlockHostPrecondCache(
            block_indices=tuple(block_indices),
            block_factors=tuple(block_factors),
            extra_idx_np=extra_idx_np,
            extra_inv_np=extra_inv_np,
        )
        _RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE[cache_key] = cached

    def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
        r_np = np.asarray(r_full, dtype=np.float64).reshape((-1,))
        z_np = np.array(r_np, copy=True)
        for idx_np, fac in zip(cached.block_indices, cached.block_factors, strict=True):
            if fac is None or idx_np.size == 0:
                continue
            z_np[idx_np] = np.asarray(fac.solve(r_np[idx_np]), dtype=np.float64)
        if cached.extra_inv_np is not None and cached.extra_idx_np.size:
            z_np[cached.extra_idx_np] = cached.extra_inv_np @ r_np[cached.extra_idx_np]
        return jnp.asarray(z_np, dtype=jnp.float64)

    apply_full_safe = safe_preconditioner(_apply_full)

    if reduce_full is None or expand_reduced is None:
        return apply_full_safe

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full_safe(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced


def compute_rhs1_sxblock_tz_sparse_host_seed(
    *,
    op: V3FullSystemOperator,
    rhs_reduced: jnp.ndarray,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    emit: Callable[[int, str], None] | None = None,
) -> jnp.ndarray:
    """Compute an explicit sparse per-L species/x seed without retaining factors."""

    import scipy.sparse as sp  # noqa: PLC0415

    rhs_full = expand_reduced(rhs_reduced) if expand_reduced is not None else rhs_reduced
    rhs_full_np = np.asarray(rhs_full, dtype=np.float64).reshape((-1,))
    seed_full_np = np.array(rhs_full_np, copy=True)

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    total_size = int(op.total_size)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    lu_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_SPARSE_LU_MAX", "").strip()
    chunk_cols_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_CHUNK_COLS", "").strip()
    try:
        lu_max = int(lu_max_env) if lu_max_env else 4000
    except ValueError:
        lu_max = 4000
    lu_max = max(0, int(lu_max))
    try:
        chunk_cols_cap = int(chunk_cols_env) if chunk_cols_env else 32
    except ValueError:
        chunk_cols_cap = 32
    chunk_cols_cap = max(1, int(chunk_cols_cap))
    struct_tol = sparse_structural_tol()

    for ell in range(n_l):
        rep_idx_np = _sxblock_active_indices_for_l(
            n_species=n_species,
            n_x=n_x,
            n_l=n_l,
            n_theta=n_theta,
            n_zeta=n_zeta,
            nxi_for_x=nxi_for_x,
            ell=ell,
        )
        if rep_idx_np.size == 0:
            continue
        exact_lu = int(rep_idx_np.size) <= int(lu_max)
        block_cache_key = (
            *rhsmode1_precond_cache_key(op, "sxblock_tz_sparse_seed"),
            int(ell),
            int(rep_idx_np.size),
            float(drop_tol),
            float(drop_rel),
            float(ilu_drop_tol),
            float(fill_factor),
            int(exact_lu),
        )
        try:
            if emit is not None:
                emit(
                    1,
                    "sxblock_sparse_seed: assembling per-L species/x block "
                    f"(L={int(ell)} size={int(rep_idx_np.size)})",
                )
            chunk_cols = min(precond_chunk_cols(total_size, int(rep_idx_np.shape[0])), int(chunk_cols_cap))
            y_sub = matvec_submatrix_v3_unsharded(
                op,
                col_idx=rep_idx_np,
                row_idx=rep_idx_np,
                total_size=total_size,
                chunk_cols=chunk_cols,
            )
            a_block = np.asarray(y_sub.T, dtype=np.float64)
            if struct_tol > 0.0 and a_block.size:
                a_block[np.abs(a_block) <= struct_tol] = 0.0
            a_csr = sp.csr_matrix(a_block)
            a_csr.eliminate_zeros()
            factorize_sparse_matrix_csr_host(
                a_csr_full=a_csr,
                cache_key=block_cache_key,
                drop_tol=drop_tol,
                drop_rel=drop_rel,
                ilu_drop_tol=ilu_drop_tol,
                fill_factor=fill_factor,
                factorization="lu" if exact_lu else "ilu",
                emit=emit,
            )
            fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.pop(block_cache_key, None)
            fac = None if fac_cache is None else fac_cache.ilu
            if fac is None:
                continue
            sol = np.asarray(fac.solve(rhs_full_np[rep_idx_np]), dtype=np.float64)
            if np.all(np.isfinite(sol)):
                seed_full_np[rep_idx_np] = sol
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(
                    1,
                    "sxblock_sparse_seed: block solve failed "
                    f"(L={int(ell)} size={int(rep_idx_np.size)}) "
                    f"({type(exc).__name__}: {exc})",
                )

    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    if extra_size > 0:
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        chunk_cols = precond_chunk_cols(total_size, int(extra_idx_np.shape[0]))
        y_sub = matvec_submatrix_v3_unsharded(
            op,
            col_idx=extra_idx_np,
            row_idx=extra_idx_np,
            total_size=total_size,
            chunk_cols=chunk_cols,
        )
        ee = np.asarray(y_sub.T, dtype=np.float64)
        ee = ee + np.float64(reg_val) * np.eye(extra_size, dtype=np.float64)
        try:
            extra_inv_np = np.linalg.inv(ee)
        except np.linalg.LinAlgError:
            extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)
        if not np.all(np.isfinite(extra_inv_np)):
            extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)
        seed_full_np[extra_idx_np] = extra_inv_np @ rhs_full_np[extra_idx_np]

    seed_full = jnp.asarray(seed_full_np, dtype=jnp.float64)
    return reduce_full(seed_full) if reduce_full is not None else seed_full


def build_rhs1_xblock_tz_sparse_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    build_jax_factors: bool,
    preconditioner_species: int,
    preconditioner_xi: int,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    force_assembled_host_fp: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Sparse per-x preconditioner for large FP RHSMode=1 systems.

    This is a closer analogue to v3's default matrix preconditioner when
    ``preconditioner_x > 0``: retain the full local (theta,zeta,L) coupling
    inside each x-block, but drop inter-x coupling by factorizing each
    per-(species,x) block independently with sparse ILU/LU.
    """
    row_cap_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_ROW_NNZ_MAX", "").strip()
    try:
        row_cap = int(row_cap_env) if row_cap_env else 64
    except ValueError:
        row_cap = 64
    row_cap = max(0, int(row_cap))
    jax_factor_format_env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "padded")
        .strip()
        .lower()
        .replace("-", "_")
    )
    xblock_jax_factor_format = (
        "csr"
        if jax_factor_format_env in {"csr", "compact", "compact_csr", "ragged_csr"}
        else "padded"
    )
    jax_factor_apply_env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY", "exact")
        .strip()
        .lower()
        .replace("-", "_")
    )
    xblock_jax_factor_apply = (
        "diagonal"
        if jax_factor_apply_env in {"diag", "diagonal", "jacobi", "factor_diag", "factor_diagonal"}
        else "identity"
        if jax_factor_apply_env in {"identity", "none", "skip"}
        else "upper"
        if jax_factor_apply_env in {"upper", "upper_only", "u", "u_only"}
        else "lower"
        if jax_factor_apply_env in {"lower", "lower_only", "l", "l_only"}
        else "exact"
    )
    compact_row_cap_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_COMPACT_ROW_NNZ_MAX",
        "",
    ).strip()
    try:
        compact_row_cap = int(compact_row_cap_env) if compact_row_cap_env else 0
    except ValueError:
        compact_row_cap = 0
    compact_row_cap = max(0, int(compact_row_cap))
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    total_size = int(op.total_size)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    block_size_max = int(n_l * n_theta * n_zeta)
    default_lu_max = rhsmode1_xblock_sparse_lu_default_max(op, build_jax_factors=build_jax_factors)
    lu_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "").strip()
    try:
        lu_max = int(lu_max_env) if lu_max_env else default_lu_max
    except ValueError:
        lu_max = default_lu_max
    lu_max = max(0, int(lu_max))
    lower_fill_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL", "").strip()
    lower_fill_drop_tol_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_DROP_TOL",
        "",
    ).strip()
    lower_fill_drop_rel_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_DROP_REL",
        "",
    ).strip()
    lower_fill_ilu_drop_tol_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ILU_DROP_TOL",
        "",
    ).strip()
    lower_fill_factor_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_FACTOR",
        "",
    ).strip()
    lower_fill_row_cap_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ROW_NNZ_MAX",
        "",
    ).strip()
    lower_fill_compact_row_cap_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_COMPACT_ROW_NNZ_MAX",
        "",
    ).strip()
    lower_fill_max_block_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_MAX_BLOCK_SIZE",
        "",
    ).strip()
    host_block_max_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX",
        "",
    ).strip()
    skipped_diag_fallback = _rhs1_bool_env(
        "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_SKIPPED_DIAG_FALLBACK",
        default=False,
    )
    skipped_diag_floor = _rhs1_float_env(
        "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_SKIPPED_DIAG_FLOOR",
        default=1.0e-10,
        minimum=0.0,
    )
    lower_fill_mode, lower_fill_ignored_env = _rhs1_xblock_policy.rhs1_xblock_lower_fill_mode(lower_fill_env)

    def _local_factor_candidate(block_size: int) -> _rhs1_xblock_policy.RHS1XBlockLocalSolveCandidate:
        return _rhs1_xblock_policy.rhs1_xblock_local_solve_candidate(
            block_size=int(block_size),
            lu_max=int(lu_max),
            lower_fill_env_value=lower_fill_env,
            drop_tol_env_value=str(drop_tol),
            drop_rel_env_value=str(drop_rel),
            ilu_drop_tol_env_value=str(ilu_drop_tol),
            fill_factor_env_value=str(fill_factor),
            row_nnz_cap_env_value=str(row_cap),
            compact_row_nnz_cap_env_value=str(compact_row_cap),
            lower_fill_drop_tol_env_value=lower_fill_drop_tol_env,
            lower_fill_drop_rel_env_value=lower_fill_drop_rel_env,
            lower_fill_ilu_drop_tol_env_value=lower_fill_ilu_drop_tol_env,
            lower_fill_factor_env_value=lower_fill_factor_env,
            lower_fill_row_nnz_cap_env_value=lower_fill_row_cap_env,
            lower_fill_compact_row_nnz_cap_env_value=lower_fill_compact_row_cap_env,
            lower_fill_max_block_size_env_value=lower_fill_max_block_env,
        )

    cache_key = (
        *rhsmode1_precond_cache_key(op, "xblock_tz_sparse"),
        bool(build_jax_factors),
        float(drop_tol),
        float(drop_rel),
        float(ilu_drop_tol),
        float(fill_factor),
        int(row_cap),
        str(xblock_jax_factor_format) if build_jax_factors else "host",
        int(compact_row_cap) if build_jax_factors and xblock_jax_factor_format == "csr" else 0,
        str(xblock_jax_factor_apply) if build_jax_factors else "host",
        int(lu_max),
        bool(force_assembled_host_fp),
        str(lower_fill_mode),
        bool(lower_fill_ignored_env),
        lower_fill_drop_tol_env,
        lower_fill_drop_rel_env,
        lower_fill_ilu_drop_tol_env,
        lower_fill_factor_env,
        lower_fill_row_cap_env,
        lower_fill_compact_row_cap_env,
        lower_fill_max_block_env,
        host_block_max_env if not build_jax_factors else "",
        bool(skipped_diag_fallback) if not build_jax_factors else False,
        float(skipped_diag_floor) if not build_jax_factors else 0.0,
    )

    if build_jax_factors:
        if xblock_jax_factor_format == "csr":
            cached = _RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE.get(cache_key)
        else:
            cached = _RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE.get(cache_key)
    else:
        cached = _RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE.get(cache_key)

    if cached is None:
        extra_start = int(op.f_size + op.phi1_size)
        extra_size = int(op.extra_size)
        extra_idx_np = np.arange(extra_start, extra_start + extra_size, dtype=np.int32)
        reg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PRECOND_REG", "").strip()
        reg_val = float(reg_env) if reg_env else 1e-10
        assembled_host_fp = bool(
            (
                bool(force_assembled_host_fp)
                and (not bool(build_jax_factors))
                and int(op.rhs_mode) == 1
                and (not bool(op.include_phi1))
                and op.fblock.fp is not None
                and op.fblock.pas is None
                and rhsmode1_fp_xblock_species_decoupled_for_host_assembly(
                    op=op,
                    preconditioner_species=preconditioner_species,
                )
                and int(preconditioner_xi) == 1
                and (not bool(op.point_at_x0))
            )
            or rhsmode1_fp_xblock_assembled_host_allowed(
                op=op,
                preconditioner_species=preconditioner_species,
                preconditioner_xi=preconditioner_xi,
                use_implicit=build_jax_factors,
            )
        )
        assembled_host_cache = get_rhsmode1_fp_xblock_assembled_host_cache(op=op) if assembled_host_fp else None

        if build_jax_factors and xblock_jax_factor_format == "csr":
            perm_r_blocks: list[np.ndarray] = []
            inv_perm_c_blocks: list[np.ndarray] = []
            upper_diag_blocks: list[np.ndarray] = []
            lower_indptr_parts: list[np.ndarray] = []
            lower_indices_parts: list[np.ndarray] = []
            lower_val_parts: list[np.ndarray] = []
            upper_indptr_parts: list[np.ndarray] = []
            upper_indices_parts: list[np.ndarray] = []
            upper_val_parts: list[np.ndarray] = []
            lower_nnz_total = 0
            upper_nnz_total = 0
            csr_index_limit = int(np.iinfo(np.int32).max)

            for s in range(n_species):
                for ix in range(n_x):
                    n_lx = int(nxi_for_x[ix])
                    block_size = int(n_lx * n_theta * n_zeta)
                    start = int((((s * n_x + ix) * n_l) * n_theta) * n_zeta)

                    perm_r_full = np.arange(block_size_max, dtype=np.int32)
                    inv_perm_c_full = np.arange(block_size_max, dtype=np.int32)
                    upper_diag_full = np.ones((block_size_max,), dtype=np.float64)
                    lower_indptr_full = np.empty((block_size_max + 1,), dtype=np.int32)
                    upper_indptr_full = np.empty((block_size_max + 1,), dtype=np.int32)
                    lower_indptr_full[0] = int(lower_nnz_total)
                    upper_indptr_full[0] = int(upper_nnz_total)

                    fac_cache = None
                    if block_size > 0:
                        block_slice = slice(start, start + block_size)

                        def _mv_block(v_block: jnp.ndarray, *, _block_slice=block_slice) -> jnp.ndarray:
                            x_full = jnp.zeros((total_size,), dtype=v_block.dtype)
                            x_full = x_full.at[_block_slice].set(v_block)
                            y_full = apply_v3_full_system_operator_cached(op, x_full)
                            return y_full[_block_slice]

                        block_cache_key = (cache_key, "csr", int(s), int(ix), int(block_size))
                        local_candidate = _local_factor_candidate(block_size)
                        block_compact_row_cap = int(local_candidate.tuning.compact_row_nnz_cap)
                        if local_candidate.lower_fill and emit is not None:
                            emit(
                                1,
                                "xblock_sparse_csr: lower-fill local factor "
                                f"(species={int(s)} x={int(ix)} size={int(block_size)} "
                                f"drop_tol={local_candidate.tuning.ilu_drop_tol:.1e} "
                                f"fill={local_candidate.tuning.fill_factor:.1f} "
                                f"row_cap={int(local_candidate.tuning.row_nnz_cap)} "
                                f"compact_row_cap={int(block_compact_row_cap)})",
                            )
                        try:
                            build_sparse_ilu_from_matvec(
                                matvec=_mv_block,
                                n=int(block_size),
                                dtype=jnp.float64,
                                cache_key=block_cache_key,
                                drop_tol=local_candidate.tuning.drop_tol,
                                drop_rel=local_candidate.tuning.drop_rel,
                                ilu_drop_tol=local_candidate.tuning.ilu_drop_tol,
                                fill_factor=local_candidate.tuning.fill_factor,
                                build_dense_factors=False,
                                build_jax_factors=False,
                                build_ilu=True,
                                store_dense=False,
                                factorization=local_candidate.factorization,
                                row_nnz_cap=0,
                                emit=emit,
                            )
                            fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(block_cache_key)
                        except Exception as exc:  # noqa: BLE001
                            fac_cache = None
                            if emit is not None:
                                emit(
                                    1,
                                    "xblock_sparse_csr: factorization failed for block "
                                    f"(species={int(s)} x={int(ix)} size={int(block_size)}) "
                                    f"({type(exc).__name__}: {exc})",
                                )

                    fac = None if fac_cache is None else fac_cache.ilu
                    if fac is not None:
                        perm_r = np.asarray(fac.perm_r, dtype=np.int32)
                        perm_c = np.asarray(fac.perm_c, dtype=np.int32)
                        inv_perm_c = np.argsort(perm_c).astype(np.int32, copy=False)
                        perm_r_full[:block_size] = perm_r
                        inv_perm_c_full[:block_size] = inv_perm_c

                        l_csr = fac.L.tocsr()
                        u_csr = fac.U.tocsr()
                        for i in range(int(block_size)):
                            rs = int(l_csr.indptr[i])
                            re = int(l_csr.indptr[i + 1])
                            cols = l_csr.indices[rs:re]
                            vals = l_csr.data[rs:re]
                            mask = cols < i
                            cols_l = cols[mask].astype(np.int32, copy=False)
                            vals_l = vals[mask].astype(np.float64, copy=False)
                            if block_compact_row_cap > 0 and int(cols_l.size) > int(block_compact_row_cap):
                                sel = np.argpartition(np.abs(vals_l), -int(block_compact_row_cap))[
                                    -int(block_compact_row_cap) :
                                ]
                                cols_l = cols_l[sel]
                                vals_l = vals_l[sel]
                            if cols_l.size:
                                order = np.argsort(cols_l)
                                cols_l = cols_l[order]
                                vals_l = vals_l[order]
                                lower_indices_parts.append(cols_l)
                                lower_val_parts.append(vals_l)
                                lower_nnz_total += int(cols_l.size)
                                if lower_nnz_total > csr_index_limit:
                                    raise MemoryError("compact x-block lower factor exceeds int32 CSR index capacity")
                            lower_indptr_full[i + 1] = int(lower_nnz_total)

                            rs_u = int(u_csr.indptr[i])
                            re_u = int(u_csr.indptr[i + 1])
                            cols_u_all = u_csr.indices[rs_u:re_u]
                            vals_u_all = u_csr.data[rs_u:re_u]
                            diag_mask = cols_u_all == i
                            if np.any(diag_mask):
                                upper_diag_full[i] = float(vals_u_all[diag_mask][0])
                            cols_u = cols_u_all[cols_u_all > i].astype(np.int32, copy=False)
                            vals_u = vals_u_all[cols_u_all > i].astype(np.float64, copy=False)
                            if block_compact_row_cap > 0 and int(cols_u.size) > int(block_compact_row_cap):
                                sel = np.argpartition(np.abs(vals_u), -int(block_compact_row_cap))[
                                    -int(block_compact_row_cap) :
                                ]
                                cols_u = cols_u[sel]
                                vals_u = vals_u[sel]
                            if cols_u.size:
                                order = np.argsort(cols_u)
                                cols_u = cols_u[order]
                                vals_u = vals_u[order]
                                upper_indices_parts.append(cols_u)
                                upper_val_parts.append(vals_u)
                                upper_nnz_total += int(cols_u.size)
                                if upper_nnz_total > csr_index_limit:
                                    raise MemoryError("compact x-block upper factor exceeds int32 CSR index capacity")
                            upper_indptr_full[i + 1] = int(upper_nnz_total)

                        bad_diag = ~np.isfinite(upper_diag_full[:block_size]) | (upper_diag_full[:block_size] == 0.0)
                        if np.any(bad_diag):
                            upper_diag_full[np.flatnonzero(bad_diag)] = 1.0
                    elif emit is not None and block_size > 0:
                        emit(
                            1,
                            "xblock_sparse_csr: missing factor for block "
                            f"(species={int(s)} x={int(ix)} size={int(block_size)}); using identity",
                        )

                    if fac is None:
                        lower_indptr_full[1:] = int(lower_nnz_total)
                        upper_indptr_full[1:] = int(upper_nnz_total)
                    elif block_size < block_size_max:
                        lower_indptr_full[block_size + 1 :] = int(lower_nnz_total)
                        upper_indptr_full[block_size + 1 :] = int(upper_nnz_total)

                    perm_r_blocks.append(perm_r_full)
                    inv_perm_c_blocks.append(inv_perm_c_full)
                    upper_diag_blocks.append(upper_diag_full)
                    lower_indptr_parts.append(lower_indptr_full)
                    upper_indptr_parts.append(upper_indptr_full)

            lower_indices_np = (
                np.concatenate(lower_indices_parts).astype(np.int32, copy=False)
                if lower_indices_parts
                else np.zeros((0,), dtype=np.int32)
            )
            lower_val_np = (
                np.concatenate(lower_val_parts).astype(np.float64, copy=False)
                if lower_val_parts
                else np.zeros((0,), dtype=np.float64)
            )
            upper_indices_np = (
                np.concatenate(upper_indices_parts).astype(np.int32, copy=False)
                if upper_indices_parts
                else np.zeros((0,), dtype=np.int32)
            )
            upper_val_np = (
                np.concatenate(upper_val_parts).astype(np.float64, copy=False)
                if upper_val_parts
                else np.zeros((0,), dtype=np.float64)
            )
            lower_indptr_np = np.concatenate(lower_indptr_parts).astype(np.int32, copy=False)
            upper_indptr_np = np.concatenate(upper_indptr_parts).astype(np.int32, copy=False)
            perm_r_np = np.stack(perm_r_blocks, axis=0).reshape((n_species, n_x, block_size_max))
            inv_perm_c_np = np.stack(inv_perm_c_blocks, axis=0).reshape((n_species, n_x, block_size_max))
            upper_diag_np = np.stack(upper_diag_blocks, axis=0).reshape((n_species, n_x, block_size_max))
            factor_nbytes = int(
                lower_indptr_np.nbytes
                + upper_indptr_np.nbytes
                + lower_indices_np.nbytes
                + upper_indices_np.nbytes
                + lower_val_np.nbytes
                + upper_val_np.nbytes
                + perm_r_np.nbytes
                + inv_perm_c_np.nbytes
                + upper_diag_np.nbytes
            )

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
                ee = ee + np.float64(reg_val) * np.eye(extra_size, dtype=np.float64)
                try:
                    ee_inv = np.linalg.inv(ee)
                except np.linalg.LinAlgError:
                    ee_inv = np.linalg.pinv(ee, rcond=1e-12)
                if not np.all(np.isfinite(ee_inv)):
                    ee_inv = np.linalg.pinv(ee, rcond=1e-12)
                extra_inv_jnp = jnp.asarray(ee_inv, dtype=jnp.float64)

            cached = _RHSMode1SparseXBlockCSRPrecondCache(
                perm_r_sx=jnp.asarray(perm_r_np, dtype=jnp.int32),
                inv_perm_c_sx=jnp.asarray(inv_perm_c_np, dtype=jnp.int32),
                lower_indptr=jnp.asarray(lower_indptr_np, dtype=jnp.int32),
                lower_indices=jnp.asarray(lower_indices_np, dtype=jnp.int32),
                lower_val=jnp.asarray(lower_val_np, dtype=jnp.float64),
                upper_indptr=jnp.asarray(upper_indptr_np, dtype=jnp.int32),
                upper_indices=jnp.asarray(upper_indices_np, dtype=jnp.int32),
                upper_val=jnp.asarray(upper_val_np, dtype=jnp.float64),
                upper_diag_sx=jnp.asarray(upper_diag_np, dtype=jnp.float64),
                extra_idx_jnp=extra_idx_jnp,
                extra_inv_jnp=extra_inv_jnp,
                lower_nnz=int(lower_indices_np.size),
                upper_nnz=int(upper_indices_np.size),
                nbytes_estimate=int(factor_nbytes),
            )
            _RHSMODE1_SPARSE_XBLOCK_CSR_PRECOND_CACHE[cache_key] = cached
            if emit is not None:
                emit(
                    1,
                    "xblock_sparse_csr: built compact JAX factors "
                    f"(lower_nnz={int(lower_indices_np.size)} upper_nnz={int(upper_indices_np.size)} "
                    f"nbytes={int(factor_nbytes)})",
                )
        elif build_jax_factors:
            perm_r_blocks: list[np.ndarray] = []
            inv_perm_c_blocks: list[np.ndarray] = []
            lower_idx_blocks: list[np.ndarray] = []
            lower_val_blocks: list[np.ndarray] = []
            upper_idx_blocks: list[np.ndarray] = []
            upper_val_blocks: list[np.ndarray] = []
            upper_diag_blocks: list[np.ndarray] = []
            max_lower_global = 0
            max_upper_global = 0

            for s in range(n_species):
                for ix in range(n_x):
                    n_lx = int(nxi_for_x[ix])
                    block_size = int(n_lx * n_theta * n_zeta)
                    start = int((((s * n_x + ix) * n_l) * n_theta) * n_zeta)

                    perm_r_full = np.arange(block_size_max, dtype=np.int32)
                    inv_perm_c_full = np.arange(block_size_max, dtype=np.int32)
                    lower_idx_full = np.zeros((block_size_max, 0), dtype=np.int32)
                    lower_val_full = np.zeros((block_size_max, 0), dtype=np.float64)
                    upper_idx_full = np.zeros((block_size_max, 0), dtype=np.int32)
                    upper_val_full = np.zeros((block_size_max, 0), dtype=np.float64)
                    upper_diag_full = np.ones((block_size_max,), dtype=np.float64)

                    if block_size > 0:
                        block_slice = slice(start, start + block_size)

                        def _mv_block(v_block: jnp.ndarray, *, _block_slice=block_slice) -> jnp.ndarray:
                            x_full = jnp.zeros((total_size,), dtype=v_block.dtype)
                            x_full = x_full.at[_block_slice].set(v_block)
                            y_full = apply_v3_full_system_operator_cached(op, x_full)
                            return y_full[_block_slice]

                        block_cache_key = (cache_key, int(s), int(ix), int(block_size))
                        local_candidate = _local_factor_candidate(block_size)
                        if local_candidate.lower_fill and emit is not None:
                            emit(
                                1,
                                "xblock_sparse: lower-fill local factor "
                                f"(species={int(s)} x={int(ix)} size={int(block_size)} "
                                f"drop_tol={local_candidate.tuning.ilu_drop_tol:.1e} "
                                f"fill={local_candidate.tuning.fill_factor:.1f} "
                                f"row_cap={int(local_candidate.tuning.row_nnz_cap)})",
                            )
                        try:
                            build_sparse_ilu_from_matvec(
                                matvec=_mv_block,
                                n=int(block_size),
                                dtype=jnp.float64,
                                cache_key=block_cache_key,
                                drop_tol=local_candidate.tuning.drop_tol,
                                drop_rel=local_candidate.tuning.drop_rel,
                                ilu_drop_tol=local_candidate.tuning.ilu_drop_tol,
                                fill_factor=local_candidate.tuning.fill_factor,
                                build_dense_factors=False,
                                build_jax_factors=True,
                                build_ilu=True,
                                store_dense=False,
                                factorization=local_candidate.factorization,
                                row_nnz_cap=int(local_candidate.tuning.row_nnz_cap),
                                emit=emit,
                            )
                            fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(block_cache_key)
                        except Exception as exc:  # noqa: BLE001
                            fac_cache = None
                            if emit is not None:
                                emit(
                                    1,
                                    "xblock_sparse: factorization failed for block "
                                    f"(species={int(s)} x={int(ix)} size={int(block_size)}) "
                                    f"({type(exc).__name__}: {exc})",
                                )
                        if fac_cache is not None:
                            perm_r = fac_cache.perm_r
                            inv_perm_c = fac_cache.inv_perm_c
                            lower_idx = fac_cache.lower_idx
                            lower_val = fac_cache.lower_val
                            upper_idx = fac_cache.upper_idx
                            upper_val = fac_cache.upper_val
                            upper_diag = fac_cache.upper_diag
                            if (
                                perm_r is not None
                                and inv_perm_c is not None
                                and lower_idx is not None
                                and lower_val is not None
                                and upper_idx is not None
                                and upper_val is not None
                                and upper_diag is not None
                            ):
                                perm_r_np = np.asarray(perm_r, dtype=np.int32)
                                inv_perm_c_np = np.asarray(inv_perm_c, dtype=np.int32)
                                lower_idx_np = np.asarray(lower_idx, dtype=np.int32)
                                lower_val_np = np.asarray(lower_val, dtype=np.float64)
                                upper_idx_np = np.asarray(upper_idx, dtype=np.int32)
                                upper_val_np = np.asarray(upper_val, dtype=np.float64)
                                upper_diag_np = np.asarray(upper_diag, dtype=np.float64)
                                max_lower = int(lower_idx_np.shape[1])
                                max_upper = int(upper_idx_np.shape[1])
                                perm_r_full[:block_size] = perm_r_np
                                inv_perm_c_full[:block_size] = inv_perm_c_np
                                lower_idx_full = -np.ones((block_size_max, max_lower), dtype=np.int32)
                                lower_val_full = np.zeros((block_size_max, max_lower), dtype=np.float64)
                                lower_idx_full[:block_size, :] = lower_idx_np
                                lower_val_full[:block_size, :] = lower_val_np
                                upper_idx_full = -np.ones((block_size_max, max_upper), dtype=np.int32)
                                upper_val_full = np.zeros((block_size_max, max_upper), dtype=np.float64)
                                upper_idx_full[:block_size, :] = upper_idx_np
                                upper_val_full[:block_size, :] = upper_val_np
                                upper_diag_full[:block_size] = upper_diag_np
                            elif emit is not None:
                                emit(
                                    1,
                                    "xblock_sparse: missing JAX factors for block "
                                    f"(species={int(s)} x={int(ix)} size={int(block_size)}); using identity",
                                )

                    perm_r_blocks.append(perm_r_full)
                    inv_perm_c_blocks.append(inv_perm_c_full)
                    lower_idx_blocks.append(lower_idx_full)
                    lower_val_blocks.append(lower_val_full)
                    upper_idx_blocks.append(upper_idx_full)
                    upper_val_blocks.append(upper_val_full)
                    upper_diag_blocks.append(upper_diag_full)
                    max_lower_global = max(max_lower_global, int(lower_idx_full.shape[1]))
                    max_upper_global = max(max_upper_global, int(upper_idx_full.shape[1]))

            for i in range(len(lower_idx_blocks)):
                lower_idx_block = lower_idx_blocks[i]
                lower_val_block = lower_val_blocks[i]
                upper_idx_block = upper_idx_blocks[i]
                upper_val_block = upper_val_blocks[i]
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

            perm_r_np = np.stack(perm_r_blocks, axis=0).reshape((n_species, n_x, block_size_max))
            inv_perm_c_np = np.stack(inv_perm_c_blocks, axis=0).reshape((n_species, n_x, block_size_max))
            lower_idx_np = np.stack(lower_idx_blocks, axis=0).reshape(
                (n_species, n_x, block_size_max, max_lower_global)
            )
            lower_val_np = np.stack(lower_val_blocks, axis=0).reshape(
                (n_species, n_x, block_size_max, max_lower_global)
            )
            upper_idx_np = np.stack(upper_idx_blocks, axis=0).reshape(
                (n_species, n_x, block_size_max, max_upper_global)
            )
            upper_val_np = np.stack(upper_val_blocks, axis=0).reshape(
                (n_species, n_x, block_size_max, max_upper_global)
            )
            upper_diag_np = np.stack(upper_diag_blocks, axis=0).reshape((n_species, n_x, block_size_max))

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
                ee = ee + np.float64(reg_val) * np.eye(extra_size, dtype=np.float64)
                try:
                    ee_inv = np.linalg.inv(ee)
                except np.linalg.LinAlgError:
                    ee_inv = np.linalg.pinv(ee, rcond=1e-12)
                if not np.all(np.isfinite(ee_inv)):
                    ee_inv = np.linalg.pinv(ee, rcond=1e-12)
                extra_inv_jnp = jnp.asarray(ee_inv, dtype=jnp.float64)

            cached = _RHSMode1SparseXBlockPrecondCache(
                perm_r_sx=jnp.asarray(perm_r_np, dtype=jnp.int32),
                inv_perm_c_sx=jnp.asarray(inv_perm_c_np, dtype=jnp.int32),
                lower_idx_sx=jnp.asarray(lower_idx_np, dtype=jnp.int32),
                lower_val_sx=jnp.asarray(lower_val_np, dtype=jnp.float64),
                upper_idx_sx=jnp.asarray(upper_idx_np, dtype=jnp.int32),
                upper_val_sx=jnp.asarray(upper_val_np, dtype=jnp.float64),
                upper_diag_sx=jnp.asarray(upper_diag_np, dtype=jnp.float64),
                extra_idx_jnp=extra_idx_jnp,
                extra_inv_jnp=extra_inv_jnp,
            )
            _RHSMODE1_SPARSE_XBLOCK_PRECOND_CACHE[cache_key] = cached
        else:
            block_slices: list[tuple[int, int]] = []
            block_factors: list[object | None] = []
            block_diag_inv: list[np.ndarray | None] = []

            def _maybe_skipped_diag_inv(*, species: int, ix: int, block_size: int) -> np.ndarray | None:
                if (not bool(skipped_diag_fallback)) or (not bool(assembled_host_fp)) or block_size <= 0:
                    return None
                try:
                    diag = rhsmode1_fp_xblock_tz_sparse_diagonal(
                        op=op,
                        species=int(species),
                        ix=int(ix),
                        preconditioner_xi=preconditioner_xi,
                        host_cache=assembled_host_cache,
                    )
                    diag_inv = safe_inverse_diagonal_np(diag, floor=float(skipped_diag_floor))
                    if diag_inv is not None and int(diag_inv.size) == int(block_size):
                        if emit is not None:
                            emit(
                                1,
                                "xblock_sparse_host: using diagonal fallback for skipped/rejected block "
                                f"(species={int(species)} x={int(ix)} size={int(block_size)})",
                            )
                        return diag_inv
                except Exception as exc:  # noqa: BLE001
                    if emit is not None:
                        emit(
                            1,
                            "xblock_sparse_host: diagonal fallback unavailable for block "
                            f"(species={int(species)} x={int(ix)} size={int(block_size)}) "
                            f"({type(exc).__name__}: {exc})",
                        )
                return None

            for s in range(n_species):
                for ix in range(n_x):
                    n_lx = int(nxi_for_x[ix])
                    block_size = int(n_lx * n_theta * n_zeta)
                    start = int((((s * n_x + ix) * n_l) * n_theta) * n_zeta)
                    if block_size <= 0:
                        continue
                    block_slice = slice(start, start + block_size)
                    if (
                        not build_jax_factors
                        and not _rhs1_xblock_sparse_host_policy.rhs1_xblock_sparse_host_block_factor_allowed(
                            block_size=int(block_size),
                            max_block_size_env_value=host_block_max_env,
                        )
                    ):
                        if emit is not None:
                            emit(
                                1,
                                "xblock_sparse_host: skipping local factor "
                                f"(species={int(s)} x={int(ix)} size={int(block_size)} "
                                "exceeds host block cap; set "
                                "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX=0 "
                                "to force the historical uncapped path)",
                            )
                        block_slices.append((start, block_size))
                        block_factors.append(None)
                        block_diag_inv.append(
                            _maybe_skipped_diag_inv(species=int(s), ix=int(ix), block_size=int(block_size))
                        )
                        continue

                    def _mv_block(v_block: jnp.ndarray, *, _block_slice=block_slice) -> jnp.ndarray:
                        x_full = jnp.zeros((total_size,), dtype=v_block.dtype)
                        x_full = x_full.at[_block_slice].set(v_block)
                        y_full = apply_v3_full_system_operator_cached(op, x_full)
                        return y_full[_block_slice]

                    block_cache_key = (cache_key, int(s), int(ix), int(block_size))
                    local_candidate = _local_factor_candidate(block_size)
                    if local_candidate.lower_fill and emit is not None:
                        emit(
                            1,
                            "xblock_sparse_host: lower-fill local factor "
                            f"(species={int(s)} x={int(ix)} size={int(block_size)} "
                            f"drop_tol={local_candidate.tuning.ilu_drop_tol:.1e} "
                            f"fill={local_candidate.tuning.fill_factor:.1f} "
                            f"row_cap={int(local_candidate.tuning.row_nnz_cap)} "
                            f"reason={local_candidate.selection_reason})",
                        )
                    try:
                        fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(block_cache_key)
                        if fac_cache is None or fac_cache.ilu is None:
                            if assembled_host_fp:
                                if emit is not None:
                                    emit(
                                        1,
                                        "xblock_sparse_host: assembling explicit FP x-block "
                                        f"(species={int(s)} x={int(ix)} size={int(block_size)})",
                                    )
                                a_block = assemble_rhsmode1_fp_xblock_tz_sparse_matrix(
                                    op=op,
                                    species=int(s),
                                    ix=int(ix),
                                    preconditioner_xi=preconditioner_xi,
                                    host_cache=assembled_host_cache,
                                )
                                factorize_sparse_matrix_csr_host(
                                    a_csr_full=a_block.tocsr(),
                                    cache_key=block_cache_key,
                                    drop_tol=local_candidate.tuning.drop_tol,
                                    drop_rel=local_candidate.tuning.drop_rel,
                                    ilu_drop_tol=local_candidate.tuning.ilu_drop_tol,
                                    fill_factor=local_candidate.tuning.fill_factor,
                                    factorization=local_candidate.factorization,
                                    emit=emit,
                                )
                            else:
                                build_sparse_ilu_from_matvec(
                                    matvec=_mv_block,
                                    n=int(block_size),
                                    dtype=jnp.float64,
                                    cache_key=block_cache_key,
                                    drop_tol=local_candidate.tuning.drop_tol,
                                    drop_rel=local_candidate.tuning.drop_rel,
                                    ilu_drop_tol=local_candidate.tuning.ilu_drop_tol,
                                    fill_factor=local_candidate.tuning.fill_factor,
                                    build_dense_factors=False,
                                    build_jax_factors=False,
                                    build_ilu=True,
                                    store_dense=False,
                                    factorization=local_candidate.factorization,
                                    row_nnz_cap=int(local_candidate.tuning.row_nnz_cap),
                                    emit=emit,
                                )
                            fac_cache = _RHSMODE1_SPARSE_ILU_CACHE.get(block_cache_key)
                    except Exception as exc:  # noqa: BLE001
                        fac_cache = None
                        if emit is not None:
                            emit(
                                1,
                                "xblock_sparse_host: factorization failed for block "
                                f"(species={int(s)} x={int(ix)} size={int(block_size)}) "
                                f"({type(exc).__name__}: {exc})",
                            )
                    host_factor = None if fac_cache is None else fac_cache.ilu
                    if assembled_host_fp and host_factor is not None:
                        if not rhsmode1_host_factor_probe_ok(factor=host_factor, block_size=int(block_size)):
                            if emit is not None:
                                emit(
                                    1,
                                    "xblock_sparse_host: rejecting unstable block factor "
                                    f"(species={int(s)} x={int(ix)} size={int(block_size)})",
                            )
                            host_factor = None
                    block_slices.append((start, block_size))
                    block_factors.append(host_factor)
                    block_diag_inv.append(
                        None
                        if host_factor is not None
                        else _maybe_skipped_diag_inv(species=int(s), ix=int(ix), block_size=int(block_size))
                    )

            extra_inv_np: np.ndarray | None = None
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
                ee = ee + np.float64(reg_val) * np.eye(extra_size, dtype=np.float64)
                try:
                    extra_inv_np = np.linalg.inv(ee)
                except np.linalg.LinAlgError:
                    extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)
                if not np.all(np.isfinite(extra_inv_np)):
                    extra_inv_np = np.linalg.pinv(ee, rcond=1e-12)

            cached = _RHSMode1SparseXBlockHostPrecondCache(
                block_slices=tuple(block_slices),
                block_factors=tuple(block_factors),
                block_diag_inv=tuple(block_diag_inv),
                extra_idx_np=extra_idx_np,
                extra_inv_np=extra_inv_np,
            )
            _RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE[cache_key] = cached

    if build_jax_factors and xblock_jax_factor_format == "csr":
        perm_r_sx = cached.perm_r_sx
        inv_perm_c_sx = cached.inv_perm_c_sx
        lower_indptr = cached.lower_indptr
        lower_indices = cached.lower_indices
        lower_val = cached.lower_val
        upper_indptr = cached.upper_indptr
        upper_indices = cached.upper_indices
        upper_val = cached.upper_val
        upper_diag_sx = cached.upper_diag_sx
        extra_idx_jnp = cached.extra_idx_jnp
        extra_inv_jnp = cached.extra_inv_jnp

        f_size = int(op.f_size)
        n_blocks = int(n_species * n_x)
        block_ids = jnp.arange(n_blocks, dtype=jnp.int32)
        row_stride = jnp.asarray(block_size_max + 1, dtype=jnp.int32)
        if emit is not None and xblock_jax_factor_apply != "exact":
            emit(
                1,
                "xblock_sparse_csr: using approximate compact JAX factor apply "
                f"mode={xblock_jax_factor_apply}",
            )

        def _solve_block_csr(inputs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
            r, block_id, perm_r, inv_perm_c, upper_diag = inputs
            if xblock_jax_factor_apply == "identity":
                return r
            r_perm = r[perm_r]
            if xblock_jax_factor_apply == "diagonal":
                diag_abs = jnp.abs(upper_diag)
                sign = jnp.where(upper_diag < 0.0, -1.0, 1.0)
                denom = jnp.where(diag_abs > 1.0e-14, upper_diag, sign * 1.0e-14)
                return (r_perm / denom)[inv_perm_c]
            row_base = block_id * row_stride
            if xblock_jax_factor_apply == "lower":
                y = _triangular_solve_lower_csr_rows(
                    indptr=lower_indptr,
                    indices=lower_indices,
                    data=lower_val,
                    b=r_perm,
                    row_base=row_base,
                )
                return y[inv_perm_c]
            if xblock_jax_factor_apply == "upper":
                z = _triangular_solve_upper_csr_rows(
                    indptr=upper_indptr,
                    indices=upper_indices,
                    data=upper_val,
                    upper_diag=upper_diag,
                    b=r_perm,
                    row_base=row_base,
                )
                return z[inv_perm_c]
            y = _triangular_solve_lower_csr_rows(
                indptr=lower_indptr,
                indices=lower_indices,
                data=lower_val,
                b=r_perm,
                row_base=row_base,
            )
            z = _triangular_solve_upper_csr_rows(
                indptr=upper_indptr,
                indices=upper_indices,
                data=upper_val,
                upper_diag=upper_diag,
                b=y,
                row_base=row_base,
            )
            return z[inv_perm_c]

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=jnp.float64)
            r_blocks = r_full[:f_size].reshape((n_blocks, block_size_max))
            z_blocks = jax.lax.map(
                _solve_block_csr,
                (
                    r_blocks,
                    block_ids,
                    perm_r_sx.reshape((n_blocks, block_size_max)),
                    inv_perm_c_sx.reshape((n_blocks, block_size_max)),
                    upper_diag_sx.reshape((n_blocks, block_size_max)),
                ),
            )
            z_full = jnp.concatenate([z_blocks.reshape((-1,)), r_full[f_size:]], axis=0)
            if extra_inv_jnp is not None:
                r_extra = r_full[extra_idx_jnp]
                z_extra = extra_inv_jnp @ r_extra
                z_full = z_full.at[extra_idx_jnp].set(z_extra, unique_indices=True)
            return jnp.asarray(z_full, dtype=jnp.float64)
    elif build_jax_factors:
        perm_r_sx = cached.perm_r_sx
        inv_perm_c_sx = cached.inv_perm_c_sx
        lower_idx_sx = cached.lower_idx_sx
        lower_val_sx = cached.lower_val_sx
        upper_idx_sx = cached.upper_idx_sx
        upper_val_sx = cached.upper_val_sx
        upper_diag_sx = cached.upper_diag_sx
        extra_idx_jnp = cached.extra_idx_jnp
        extra_inv_jnp = cached.extra_inv_jnp

        f_size = int(op.f_size)

        def _solve_block(
            r: jnp.ndarray,
            perm_r: jnp.ndarray,
            inv_perm_c: jnp.ndarray,
            lower_idx: jnp.ndarray,
            lower_val: jnp.ndarray,
            upper_idx: jnp.ndarray,
            upper_val: jnp.ndarray,
            upper_diag: jnp.ndarray,
        ) -> jnp.ndarray:
            r_perm = r[perm_r]
            y = _triangular_solve_lower_padded(lower_idx=lower_idx, lower_val=lower_val, b=r_perm)
            z = _triangular_solve_upper_padded(
                upper_idx=upper_idx,
                upper_val=upper_val,
                upper_diag=upper_diag,
                b=y,
            )
            return z[inv_perm_c]

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
        if emit is not None and xblock_jax_factor_apply != "exact":
            emit(
                1,
                "xblock_sparse: using approximate JAX factor apply "
                f"mode={xblock_jax_factor_apply}",
            )

        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_full = jnp.asarray(r_full, dtype=jnp.float64)
            r_f = r_full[:f_size].reshape((n_species, n_x, block_size_max))
            if xblock_jax_factor_apply == "identity":
                z_f = r_f
            elif xblock_jax_factor_apply == "diagonal":
                diag_abs = jnp.abs(upper_diag_sx)
                sign = jnp.where(upper_diag_sx < 0.0, -1.0, 1.0)
                denom = jnp.where(diag_abs > 1.0e-14, upper_diag_sx, sign * 1.0e-14)
                r_perm = jnp.take_along_axis(r_f, perm_r_sx, axis=2)
                z_perm = r_perm / denom
                z_f = jnp.take_along_axis(z_perm, inv_perm_c_sx, axis=2)
            elif xblock_jax_factor_apply == "lower":
                r_perm = jnp.take_along_axis(r_f, perm_r_sx, axis=2)
                y = jax.vmap(
                    jax.vmap(
                        lambda r, lower_idx, lower_val: _triangular_solve_lower_padded(
                            lower_idx=lower_idx,
                            lower_val=lower_val,
                            b=r,
                        ),
                        in_axes=(0, 0, 0),
                        out_axes=0,
                    ),
                    in_axes=(0, 0, 0),
                    out_axes=0,
                )(r_perm, lower_idx_sx, lower_val_sx)
                z_f = jnp.take_along_axis(y, inv_perm_c_sx, axis=2)
            elif xblock_jax_factor_apply == "upper":
                r_perm = jnp.take_along_axis(r_f, perm_r_sx, axis=2)
                z_perm = jax.vmap(
                    jax.vmap(
                        lambda r, upper_idx, upper_val, upper_diag: _triangular_solve_upper_padded(
                            upper_idx=upper_idx,
                            upper_val=upper_val,
                            upper_diag=upper_diag,
                            b=r,
                        ),
                        in_axes=(0, 0, 0, 0),
                        out_axes=0,
                    ),
                    in_axes=(0, 0, 0, 0),
                    out_axes=0,
                )(r_perm, upper_idx_sx, upper_val_sx, upper_diag_sx)
                z_f = jnp.take_along_axis(z_perm, inv_perm_c_sx, axis=2)
            else:
                z_f = _solve_over_sx(
                    r_f,
                    perm_r_sx,
                    inv_perm_c_sx,
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
    else:
        def _apply_full(r_full: jnp.ndarray) -> jnp.ndarray:
            r_np = np.asarray(r_full, dtype=np.float64).reshape((-1,))
            z_np = np.array(r_np, copy=True)
            for (start, block_size), fac, diag_inv in zip(
                cached.block_slices,
                cached.block_factors,
                cached.block_diag_inv,
                strict=True,
            ):
                if block_size <= 0:
                    continue
                sl = slice(int(start), int(start + block_size))
                if fac is not None:
                    z_np[sl] = np.asarray(fac.solve(r_np[sl]), dtype=np.float64)
                elif diag_inv is not None:
                    z_np[sl] = np.asarray(diag_inv, dtype=np.float64) * r_np[sl]
            if cached.extra_inv_np is not None and cached.extra_idx_np.size:
                z_np[cached.extra_idx_np] = cached.extra_inv_np @ r_np[cached.extra_idx_np]
            return jnp.asarray(z_np, dtype=jnp.float64)

    apply_full_safe = safe_preconditioner(_apply_full)

    if reduce_full is None or expand_reduced is None:
        return apply_full_safe

    def _apply_reduced(r_reduced: jnp.ndarray) -> jnp.ndarray:
        z_full = apply_full_safe(expand_reduced(r_reduced))
        return reduce_full(z_full)

    return _apply_reduced
