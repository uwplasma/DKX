"""Active sparse-factor preconditioners for RHSMode=1 CSR solves.

These builders operate on active-projected RHSMode=1 sparse systems. They are
host-side, non-differentiable setup routines for CLI and ``differentiable=False``
Python solves; JAX-native differentiable paths should use matrix-free operators
and implicit-linear-solve contracts instead.
"""

from __future__ import annotations

from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_response.layout import RHS1BlockLayout
from ..schur.profile_response import RHS1StructuredFullCSRPreconditioner
from .rhs1_fortran_reduced import (
    estimate_spilu_factor_nbytes,
    sparse_equilibration_scale,
    sparse_lu_factor_nbytes,
)

__all__ = (
    "build_active_filtered_sparse_factor_preconditioner",
    "build_active_global_sparse_factor_preconditioner",
    "build_active_scaled_sparse_factor_preconditioner",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _scipy_csr_nbytes(matrix: Any) -> int:
    csr = matrix.tocsr()
    return int(csr.data.nbytes + csr.indices.nbytes + csr.indptr.nbytes)


_estimate_spilu_factor_nbytes = estimate_spilu_factor_nbytes
_sparse_equilibration_scale = sparse_equilibration_scale
_sparse_lu_factor_nbytes = sparse_lu_factor_nbytes


def build_active_global_sparse_factor_preconditioner(
    *,
    matrix: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a reusable global sparse factor for the active reduced operator.

    This is the fail-closed Python-native replacement for the monolithic
    Fortran/PETSc factor role: the active Fortran-reduced CSR is assembled once,
    a sparse factor is memory-gated, and the resulting object can be cached by
    the direct-tail driver.  Exact LU is useful for bounded validation and
    small systems; ILU is the production candidate for memory-limited CPU/GPU
    workflows where the Krylov solve still owns final accuracy.
    """

    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    matrix_csc = matrix.tocsc().astype(np.float64)
    if matrix_csc.shape[0] != matrix_csc.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_sparse_factor",
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix_csc.shape)},
        )

    n = int(matrix_csc.shape[0])
    max_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FACTOR_MAX_SIZE", 1_000_000)
    if int(max_size) > 0 and n > int(max_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_sparse_factor",
            reason=f"active_global_sparse_factor_size_exceeded:{n}>{int(max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "matrix_shape": tuple(int(v) for v in matrix_csc.shape),
                "matrix_nnz": int(matrix_csc.nnz),
                "max_size": int(max_size),
            },
        )

    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FACTOR_KIND", "").strip().lower()
    if factor_kind_env in {"lu", "splu", "exact", "direct"}:
        factor_kind = "lu"
    elif factor_kind_env in {"ilu", "spilu", "incomplete_lu"}:
        factor_kind = "ilu"
    elif "ilu" in requested:
        factor_kind = "ilu"
    elif "lu" in requested or "direct" in requested:
        factor_kind = "lu"
    else:
        factor_kind = "ilu"

    fill_factor_default = 4.0 if factor_kind == "ilu" else 12.0
    fill_factor = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FILL_FACTOR", fill_factor_default))
    fill_factor = max(1.0, float(fill_factor))
    drop_tol_default = 1.0e-3 if factor_kind == "ilu" else 0.0
    drop_tol = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_DROP_TOL", drop_tol_default))
    diag_pivot_default = 1.0 if factor_kind == "lu" else 0.0
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_DIAG_PIVOT_THRESH", diag_pivot_default))
    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"

    matrix_scale = max(float(np.max(np.abs(matrix_csc.data))) if matrix_csc.nnz else 0.0, 1.0)
    diagonal_shift = max(float(abs(regularization)), 0.0) * matrix_scale
    if diagonal_shift > 0.0:
        matrix_csc = matrix_csc + diagonal_shift * sp.eye(n, dtype=np.float64, format="csc")

    estimate = _estimate_spilu_factor_nbytes(matrix=matrix_csc, fill_factor=fill_factor)
    if estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_sparse_factor",
            reason=f"active_global_sparse_factor_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "factor_kind": str(factor_kind),
                "matrix_shape": tuple(int(v) for v in matrix_csc.shape),
                "matrix_nnz": int(matrix_csc.nnz),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "permc_spec": str(permc_spec),
            },
        )

    try:
        if factor_kind == "lu":
            factor = splu(
                matrix_csc,
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
        else:
            factor = spilu(
                matrix_csc,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_sparse_factor",
            reason=f"active_global_sparse_factor_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "factor_kind": str(factor_kind),
                "error": str(exc),
                "factor_nbytes_estimate": int(estimate),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "permc_spec": str(permc_spec),
            },
        )

    factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_sparse_factor",
            reason=f"active_global_sparse_factor_budget_exceeded_after_factor:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "factor_kind": str(factor_kind),
                "matrix_shape": tuple(int(v) for v in matrix_csc.shape),
                "matrix_nnz": int(matrix_csc.nnz),
                "factor_nnz": int(factor.L.nnz + factor.U.nnz),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "permc_spec": str(permc_spec),
            },
        )

    operator = LinearOperator(
        matrix_csc.shape,
        matvec=lambda x: factor.solve(np.asarray(x, dtype=np.float64).reshape((-1,))),
        dtype=np.float64,
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_global_sparse_factor",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "global_active_sparse_factor",
            "factor_kind": str(factor_kind),
            "matrix_shape": tuple(int(v) for v in matrix_csc.shape),
            "matrix_nnz": int(matrix_csc.nnz),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "diag_pivot_thresh": float(diag_pivot),
            "permc_spec": str(permc_spec),
            "diagonal_shift": float(diagonal_shift),
        },
    )


def build_active_scaled_sparse_factor_preconditioner(
    *,
    matrix: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a row/column-equilibrated active sparse factor preconditioner."""

    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    matrix_csr = matrix.tocsr().astype(np.float64)
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_scaled_ilu",
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix_csr.shape)},
        )
    n = int(matrix_csr.shape[0])
    max_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_MAX_SIZE", 200_000)
    if int(max_size) > 0 and n > int(max_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_scaled_ilu",
            reason=f"active_scaled_ilu_size_exceeded:{n}>{int(max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
                "matrix_nnz": int(matrix_csr.nnz),
                "max_size": int(max_size),
            },
        )
    fill_factor = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_FILL_FACTOR", 4.0))
    drop_tol = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_DROP_TOL", 1.0e-3))
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_DIAG_PIVOT_THRESH", 0.0))
    max_scale = max(1.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_MAX_SCALE", 1.0e6)))
    scale_norm = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_SCALE_NORM", "l1").strip().lower()
    if scale_norm not in {"l1", "l2", "max"}:
        scale_norm = "l1"
    factor_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_FACTOR_KIND", "").strip().lower()
    if str(requested_kind).endswith("_lu") and not factor_kind:
        factor_kind = "lu"
    if factor_kind not in {"ilu", "spilu", "lu", "splu"}:
        factor_kind = "ilu"
    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    diagonal_shift = float(
        _env_float(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_DIAGONAL_SHIFT",
            max(float(abs(regularization)), 1.0e-14),
        )
    )

    estimate = _estimate_spilu_factor_nbytes(matrix=matrix_csr, fill_factor=fill_factor) + 2 * n * np.dtype(np.float64).itemsize
    if estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_scaled_ilu",
            reason=f"active_scaled_ilu_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
                "matrix_nnz": int(matrix_csr.nnz),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "drop_tol": float(drop_tol),
                "fill_factor": float(fill_factor),
                "factor_kind": str(factor_kind),
                "permc_spec": str(permc_spec),
            },
        )

    row_scale, row_meta = _sparse_equilibration_scale(
        matrix_csr,
        axis=1,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = matrix_csr.multiply(row_scale[:, None]).tocsc()
    col_scale, col_meta = _sparse_equilibration_scale(
        scaled,
        axis=0,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = scaled.multiply(col_scale[None, :]).tocsc()
    scaled_scale = max(float(np.max(np.abs(scaled.data))) if scaled.nnz else 0.0, 1.0)
    shift = max(0.0, float(diagonal_shift)) * scaled_scale
    if shift > 0.0:
        scaled = scaled + shift * sp.eye(n, dtype=np.float64, format="csc")
    try:
        if factor_kind in {"lu", "splu"}:
            factor = splu(scaled, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot))
            selected_kind = "active_scaled_lu"
        else:
            factor = spilu(
                scaled,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
            selected_kind = "active_scaled_ilu"
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_scaled_ilu",
            reason=f"active_scaled_ilu_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "error": str(exc),
                "factor_nbytes_estimate": int(estimate),
                "drop_tol": float(drop_tol),
                "fill_factor": float(fill_factor),
                "factor_kind": str(factor_kind),
                "permc_spec": str(permc_spec),
                "diagonal_shift": float(shift),
                "row_scaling": row_meta,
                "column_scaling": col_meta,
            },
        )
    factor_nbytes = int(_sparse_lu_factor_nbytes(factor) + row_scale.nbytes + col_scale.nbytes)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_scaled_ilu",
            reason=f"active_scaled_ilu_factor_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        scaled_rhs = row_scale * arr
        scaled_solution = np.asarray(factor.solve(scaled_rhs), dtype=np.float64).reshape((-1,))
        return col_scale * scaled_solution

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=selected_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            "matrix_nnz": int(matrix_csr.nnz),
            "scaled_matrix_nnz": int(scaled.nnz),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "drop_tol": float(drop_tol),
            "fill_factor": float(fill_factor),
            "diag_pivot_thresh": float(diag_pivot),
            "factor_kind": str(factor_kind),
            "permc_spec": str(permc_spec),
            "diagonal_shift": float(shift),
            "scale_norm": str(scale_norm),
            "max_scale": float(max_scale),
            "row_scaling": row_meta,
            "column_scaling": col_meta,
        },
    )


def build_active_filtered_sparse_factor_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Factor a physics-filtered active sparse operator.

    This is the first replacement path that keeps selected off-diagonal
    kinetic couplings throughout the full active system instead of solving a
    small Schur rescue subspace.  The retained matrix is built from the true
    active CSR operator by keeping:

    - all diagonal entries;
    - all rows/columns touching non-kinetic tail variables;
    - kinetic-kinetic entries inside a bounded physical neighborhood in
      ``(species, x, ell, theta, zeta)``.

    The result is still an active-only host sparse factor, but its sparsity is
    derived from the true RHSMode=1 operator and all use is guarded by the
    driver's true residual preflight before GMRES.
    """

    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    n_active = int(matrix_csr.shape[0])
    active_kinetic = active_np < int(layout.f_size)
    species = np.full((n_active,), -1, dtype=np.int64)
    x_index = np.full((n_active,), -1, dtype=np.int64)
    ell = np.full((n_active,), -1, dtype=np.int64)
    theta = np.full((n_active,), -1, dtype=np.int64)
    zeta = np.full((n_active,), -1, dtype=np.int64)
    if np.any(active_kinetic):
        decoded = layout.decode_kinetic_indices(active_np[active_kinetic])
        species[active_kinetic] = decoded.species
        x_index[active_kinetic] = decoded.x
        ell[active_kinetic] = decoded.ell
        theta[active_kinetic] = decoded.theta
        zeta[active_kinetic] = decoded.zeta

    x_radius = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_X_RADIUS", 1)))
    ell_radius = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ELL_RADIUS", 1)))
    theta_radius = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_THETA_RADIUS", 0))
    zeta_radius = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ZETA_RADIUS", 0))
    include_tail_couplings = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_INCLUDE_TAIL",
        True,
    )
    include_same_angle_all_pitch = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_SAME_ANGLE_ALL_PITCH",
        False,
    )
    include_same_x_all_angle = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_SAME_X_ALL_ANGLE",
        False,
    )

    coo = matrix_csr.tocoo(copy=False)
    row = np.asarray(coo.row, dtype=np.int64)
    col = np.asarray(coo.col, dtype=np.int64)
    row_kinetic = active_kinetic[row]
    col_kinetic = active_kinetic[col]
    diagonal_mask = row == col
    tail_mask = (~row_kinetic | ~col_kinetic) if bool(include_tail_couplings) else np.zeros_like(diagonal_mask)
    kinetic_pair = row_kinetic & col_kinetic

    same_species = np.zeros_like(diagonal_mask)
    same_x = np.zeros_like(diagonal_mask)
    same_angle = np.zeros_like(diagonal_mask)
    physical_band = np.zeros_like(diagonal_mask)
    if np.any(kinetic_pair):
        kp = kinetic_pair
        same_species[kp] = species[row[kp]] == species[col[kp]]
        dx = np.abs(x_index[row[kp]] - x_index[col[kp]])
        dell = np.abs(ell[row[kp]] - ell[col[kp]])
        dtheta = np.abs(theta[row[kp]] - theta[col[kp]])
        dzeta = np.abs(zeta[row[kp]] - zeta[col[kp]])
        if int(layout.n_theta) > 0:
            dtheta = np.minimum(dtheta, int(layout.n_theta) - dtheta)
        if int(layout.n_zeta) > 0:
            dzeta = np.minimum(dzeta, int(layout.n_zeta) - dzeta)
        theta_ok = np.ones_like(dtheta, dtype=bool) if int(theta_radius) < 0 else dtheta <= int(theta_radius)
        zeta_ok = np.ones_like(dzeta, dtype=bool) if int(zeta_radius) < 0 else dzeta <= int(zeta_radius)
        same_x[kp] = dx == 0
        same_angle[kp] = (dtheta == 0) & (dzeta == 0)
        band = (dx <= int(x_radius)) & (dell <= int(ell_radius)) & theta_ok & zeta_ok
        if bool(include_same_angle_all_pitch):
            band = band | ((dx <= int(x_radius)) & same_angle[kp])
        if bool(include_same_x_all_angle):
            band = band | ((dx == 0) & (dell <= int(ell_radius)))
        physical_band[kp] = same_species[kp] & band

    keep = diagonal_mask | tail_mask | physical_band
    if not np.any(keep):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason="active_filtered_sparse_factor_empty",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "matrix_nnz": int(matrix_csr.nnz),
                "x_radius": int(x_radius),
                "ell_radius": int(ell_radius),
                "theta_radius": int(theta_radius),
                "zeta_radius": int(zeta_radius),
            },
        )

    filtered = sp.coo_matrix(
        (np.asarray(coo.data, dtype=np.float64)[keep], (row[keep], col[keep])),
        shape=matrix_csr.shape,
    ).tocsc()
    filtered.sum_duplicates()
    filtered.eliminate_zeros()

    scale = max(float(np.max(np.abs(filtered.data))) if filtered.nnz else 0.0, 1.0)
    diagonal_shift_override = _env_float(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_DIAGONAL_SHIFT",
        float("nan"),
    )
    if np.isfinite(float(diagonal_shift_override)):
        diagonal_shift = max(0.0, float(diagonal_shift_override)) * scale
    else:
        diagonal_shift = max(float(abs(regularization)), 1.0e-14) * scale
    if diagonal_shift > 0.0:
        filtered = filtered + diagonal_shift * sp.eye(n_active, dtype=np.float64, format="csc")

    factor_kind_env = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_KIND", "spilu")
        .strip()
        .lower()
    )
    factor_kind = "splu" if factor_kind_env in {"lu", "splu", "exact", "direct"} else "spilu"
    fill_factor = max(
        1.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_FILL_FACTOR", 4.0)),
    )
    drop_tol = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_DROP_TOL", 1.0e-3)),
    )
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_DIAG_PIVOT_THRESH", 0.0))
    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    damping = max(
        0.0,
        min(2.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_DAMPING", 1.0))),
    )

    filtered_nbytes = int(_scipy_csr_nbytes(filtered.tocsr()))
    sparse_estimate = int(_estimate_spilu_factor_nbytes(matrix=filtered, fill_factor=fill_factor))
    dense_estimate = int(n_active * n_active * np.dtype(np.float64).itemsize)
    factor_estimate = int(filtered_nbytes + (sparse_estimate if factor_kind == "spilu" else max(sparse_estimate, dense_estimate)))
    large_size = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_LARGE_SIZE", 300_000))
    large_matrix = int(large_size) > 0 and int(n_active) >= int(large_size)
    prefill_safety_default = 4.0 if bool(large_matrix) else 1.0
    prefill_safety = max(
        1.0,
        float(
            _env_float(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_PREFILL_SAFETY_FACTOR",
                prefill_safety_default,
            )
        ),
    )
    prefill_estimate = int(np.ceil(float(factor_estimate) * float(prefill_safety)))
    progress = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_PROGRESS", bool(large_matrix))
    if bool(progress):
        print(
            "active_filtered_sparse_factor: setup "
            f"n={int(n_active)} matrix_nnz={int(matrix_csr.nnz)} filtered_nnz={int(filtered.nnz)} "
            f"factor_kind={factor_kind} fill_factor={float(fill_factor):.3g} "
            f"drop_tol={float(drop_tol):.3g} estimate_mb={float(factor_estimate) / (1024.0 * 1024.0):.1f} "
            f"prefill_estimate_mb={float(prefill_estimate) / (1024.0 * 1024.0):.1f} "
            f"budget_mb={float(max_factor_nbytes) / (1024.0 * 1024.0):.1f}",
            flush=True,
        )
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason=f"active_filtered_sparse_factor_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(n_active),
                "matrix_nnz": int(matrix_csr.nnz),
                "filtered_nnz": int(filtered.nnz),
                "retained_fraction": float(filtered.nnz / max(int(matrix_csr.nnz), 1)),
                "filtered_nbytes_actual": int(filtered_nbytes),
                "factor_nbytes_estimate": int(factor_estimate),
                "sparse_factor_nbytes_estimate": int(sparse_estimate),
                "dense_factor_nbytes_estimate": int(dense_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "x_radius": int(x_radius),
                "ell_radius": int(ell_radius),
                "theta_radius": int(theta_radius),
                "zeta_radius": int(zeta_radius),
                "large_matrix_defaults": bool(large_matrix),
                "prefill_safety_factor": float(prefill_safety),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
            },
        )
    if prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason=f"active_filtered_sparse_factor_prefill_budget_exceeded:{prefill_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(n_active),
                "matrix_nnz": int(matrix_csr.nnz),
                "filtered_nnz": int(filtered.nnz),
                "retained_fraction": float(filtered.nnz / max(int(matrix_csr.nnz), 1)),
                "filtered_nbytes_actual": int(filtered_nbytes),
                "factor_nbytes_estimate": int(factor_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "sparse_factor_nbytes_estimate": int(sparse_estimate),
                "dense_factor_nbytes_estimate": int(dense_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "x_radius": int(x_radius),
                "ell_radius": int(ell_radius),
                "theta_radius": int(theta_radius),
                "zeta_radius": int(zeta_radius),
                "large_matrix_defaults": bool(large_matrix),
                "prefill_safety_factor": float(prefill_safety),
            },
        )

    try:
        if factor_kind == "splu":
            factor = splu(filtered, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot))
        else:
            factor = spilu(
                filtered,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason=f"active_filtered_sparse_factor_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "error": str(exc),
                "active_size": int(n_active),
                "matrix_nnz": int(matrix_csr.nnz),
                "filtered_nnz": int(filtered.nnz),
                "factor_nbytes_estimate": int(factor_estimate),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
            },
        )

    factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
    actual_total = int(filtered_nbytes + factor_nbytes)
    if actual_total > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_filtered_sparse_factor",
            reason=f"active_filtered_sparse_factor_budget_exceeded_actual:{actual_total}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(n_active),
                "matrix_nnz": int(matrix_csr.nnz),
                "filtered_nnz": int(filtered.nnz),
                "filtered_nbytes_actual": int(filtered_nbytes),
                "factor_nbytes_actual": int(actual_total),
                "local_factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        solved = np.asarray(factor.solve(arr), dtype=np.float64).reshape((-1,))
        return float(damping) * solved

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_filtered_sparse_factor",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_physics_filtered_sparse_factor",
            "active_size": int(n_active),
            "matrix_nnz": int(matrix_csr.nnz),
            "filtered_nnz": int(filtered.nnz),
            "retained_fraction": float(filtered.nnz / max(int(matrix_csr.nnz), 1)),
            "kinetic_pair_nnz": int(np.count_nonzero(kinetic_pair)),
            "physical_band_nnz": int(np.count_nonzero(physical_band)),
            "tail_or_diagonal_nnz": int(np.count_nonzero(diagonal_mask | tail_mask)),
            "filtered_nbytes_actual": int(filtered_nbytes),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_prefill_estimate": int(prefill_estimate),
            "factor_nbytes_actual": int(actual_total),
            "local_factor_nbytes_actual": int(factor_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "factor_kind": str(factor_kind),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "prefill_safety_factor": float(prefill_safety),
            "large_matrix_defaults": bool(large_matrix),
            "diag_pivot_thresh": float(diag_pivot),
            "permc_spec": str(permc_spec),
            "diagonal_shift": float(diagonal_shift),
            "damping": float(damping),
            "x_radius": int(x_radius),
            "ell_radius": int(ell_radius),
            "theta_radius": int(theta_radius),
            "zeta_radius": int(zeta_radius),
            "include_tail_couplings": bool(include_tail_couplings),
            "include_same_angle_all_pitch": bool(include_same_angle_all_pitch),
            "include_same_x_all_angle": bool(include_same_x_all_angle),
            "requires_preflight": True,
            "note": "experimental_full_active_filtered_operator_factor",
        },
    )
