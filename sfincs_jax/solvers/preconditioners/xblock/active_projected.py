"""Active-projected x-block preconditioners for RHSMode=1 full-CSR solves.

These builders operate on an already active-projected CSR matrix while using
the original RHSMode=1 layout to recover physical ``(species, x, ell)`` blocks.
They are host-side, non-autodiff preconditioner setup utilities for the explicit
CSR solve lane; differentiable paths should stay on JAX-native operators.
"""

from __future__ import annotations

from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from ....rhs1_block_operator import RHS1BlockLayout
from ..schur.rhs1_full_csr import (
    RHS1StructuredFullCSRPreconditioner,
    build_jacobi_preconditioner,
)
from ..symbolic_sparse.rhs1_fortran_reduced import (
    estimate_spilu_factor_nbytes,
    sparse_equilibration_scale,
    sparse_lu_factor_nbytes,
)
from .low_l_schur import xblock_tz_low_l_indices

__all__ = (
    "active_positions_for_full_indices",
    "build_active_projected_overlap_schwarz_preconditioner",
    "build_active_projected_xblock_preconditioner",
)


def build_active_projected_xblock_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build sparse active block factors at fixed species and speed index."""

    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = (
        np.arange(int(matrix_csr.shape[0]), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xblock",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", int(layout.n_xi))
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FACTOR_KIND", "").strip().lower()
    if factor_kind_env in {"ilu", "spilu", "incomplete_lu"}:
        factor_kind = "spilu"
    elif factor_kind_env in {"lu", "splu", "exact", "direct"}:
        factor_kind = "splu"
    elif "ilu" in requested or "spilu" in requested or "block_asm" in requested:
        factor_kind = "spilu"
    else:
        factor_kind = "splu"
    fill_estimate = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FILL_ESTIMATE", 8.0))
    fill_factor_default = min(float(fill_estimate), 4.0) if factor_kind == "spilu" else float(fill_estimate)
    fill_factor = max(1.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FILL_FACTOR", fill_factor_default)))
    drop_tol_default = 1.0e-2 if factor_kind == "spilu" else 0.0
    drop_tol = max(0.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DROP_TOL", drop_tol_default)))
    diag_pivot_default = 0.0
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DIAG_PIVOT_THRESH", diag_pivot_default))
    damping = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DAMPING", 1.0))
    damping = max(0.0, min(float(damping), 2.0))
    allow_block_fallback = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_ALLOW_SINGULAR_FALLBACK",
        factor_kind == "spilu" or "fallback" in requested or "block_asm" in requested,
    )
    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    scale_blocks = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_SCALE", False)
    scale_norm = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_SCALE_NORM", "l1").strip().lower()
    max_scale = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_MAX_SCALE", 1.0e6))
    diagonal_shift_override = _env_float(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DIAGONAL_SHIFT",
        float("nan"),
    )
    base_mode = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_BASE", "jacobi").strip().lower()
    if base_mode not in {"jacobi", "diagonal", "zero", "none"}:
        base_mode = "jacobi"

    if base_mode in {"zero", "none"}:
        base = RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=True,
            kind="zero",
            reason="active_xblock_base_zero",
            setup_s=0.0,
            metadata={"base_mode": str(base_mode)},
        )
    else:
        base = build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=str(requested_kind),
            regularization=regularization,
            t0=t0,
            reason="active_xblock_base_jacobi",
        )
        if not bool(base.selected) or base.operator is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_xblock",
                reason=f"base_preconditioner_not_selected:{base.reason}",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={"base_preconditioner": base.to_dict(), "lmax": int(lmax), "base_mode": str(base_mode)},
            )

    block_positions: list[np.ndarray] = []
    block_factor_estimates: list[int] = []
    block_dense_estimates: list[int] = []
    block_sparse_estimates: list[int] = []
    block_matrix_nnz: list[int] = []
    block_matrix_nbytes: list[int] = []
    block_row_scales: list[np.ndarray] = []
    block_col_scales: list[np.ndarray] = []
    row_scale_min: list[float] = []
    row_scale_max: list[float] = []
    col_scale_min: list[float] = []
    col_scale_max: list[float] = []
    factors: list[Any] = []
    skipped_block_count = 0
    skipped_factor_count = 0
    skipped_budget_count = 0
    skipped_size = 0
    skipped_errors: list[str] = []
    estimated_total = 0
    actual_total = 0
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            full_idx = xblock_tz_low_l_indices(layout=layout, species=species, x=x, lmax=lmax)
            positions = active_positions_for_full_indices(active_indices=active_np, full_indices=full_idx)
            if positions.size == 0:
                continue
            block_raw = matrix_csr[positions[:, None], positions].tocsc()
            block_scale = max(float(np.max(np.abs(block_raw.data))) if block_raw.nnz else 0.0, 1.0)
            if np.isfinite(float(diagonal_shift_override)):
                diagonal_shift = max(0.0, float(diagonal_shift_override)) * block_scale
            else:
                diagonal_shift = max(float(abs(regularization)), 1.0e-14) * block_scale
            scale_nbytes = 0
            row_scale = np.ones((int(block_raw.shape[0]),), dtype=np.float64)
            col_scale = np.ones((int(block_raw.shape[1]),), dtype=np.float64)
            if scale_blocks:
                row_scale, row_meta = sparse_equilibration_scale(
                    block_raw,
                    axis=1,
                    norm=scale_norm,
                    max_scale=max_scale,
                )
                block_row_scaled = block_raw.multiply(row_scale[:, None]).tocsc()
                col_scale, col_meta = sparse_equilibration_scale(
                    block_row_scaled,
                    axis=0,
                    norm=scale_norm,
                    max_scale=max_scale,
                )
                block = block_row_scaled.multiply(col_scale[None, :]).tocsc()
                scale_nbytes = int(row_scale.nbytes + col_scale.nbytes)
                row_scale_min.append(float(row_meta["scale_min"]))
                row_scale_max.append(float(row_meta["scale_max"]))
                col_scale_min.append(float(col_meta["scale_min"]))
                col_scale_max.append(float(col_meta["scale_max"]))
            else:
                block = block_raw
            if diagonal_shift > 0.0:
                block = block + diagonal_shift * sp.eye(block.shape[0], dtype=np.float64, format="csc")
            sparse_estimate = estimate_spilu_factor_nbytes(matrix=block, fill_factor=fill_factor)
            dense_estimate = int(block.shape[0] * block.shape[1] * np.dtype(np.float64).itemsize)
            factor_estimate_core = (
                int(sparse_estimate)
                if factor_kind == "spilu"
                else max(int(sparse_estimate), int(dense_estimate))
            )
            factor_estimate = int(factor_estimate_core) + int(scale_nbytes)
            if estimated_total + factor_estimate > int(max_factor_nbytes):
                if allow_block_fallback:
                    skipped_block_count += 1
                    skipped_budget_count += 1
                    skipped_size += int(positions.size)
                    if len(skipped_errors) < 3:
                        skipped_errors.append(
                            f"budget:{estimated_total + factor_estimate}>{int(max_factor_nbytes)}"
                        )
                    continue
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_xblock",
                    reason=f"active_xblock_budget_exceeded:{estimated_total + factor_estimate}>{int(max_factor_nbytes)}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "active_size": int(matrix_csr.shape[0]),
                        "block_count": int(len(block_positions) + 1),
                        "covered_size": int(sum(int(pos.size) for pos in block_positions) + int(positions.size)),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "max_factor_nbytes": int(max_factor_nbytes),
                        "lmax": int(lmax),
                        "fill_estimate": float(fill_estimate),
                        "fill_factor": float(fill_factor),
                        "drop_tol": float(drop_tol),
                        "factor_kind": str(factor_kind),
                        "allow_block_fallback": bool(allow_block_fallback),
                        "block_scaling_enabled": bool(scale_blocks),
                        "block_scale_nbytes_estimate": int(scale_nbytes),
                    },
                )
            try:
                if factor_kind == "spilu":
                    factor = spilu(
                        block,
                        drop_tol=float(drop_tol),
                        fill_factor=float(fill_factor),
                        permc_spec=str(permc_spec),
                        diag_pivot_thresh=float(diag_pivot),
                    )
                else:
                    factor = splu(block, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot))
            except Exception as exc:  # noqa: BLE001
                if allow_block_fallback:
                    skipped_block_count += 1
                    skipped_factor_count += 1
                    skipped_size += int(positions.size)
                    if len(skipped_errors) < 3:
                        skipped_errors.append(f"{type(exc).__name__}:{str(exc)[:160]}")
                    continue
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_xblock",
                    reason=f"active_xblock_factor_failed:{type(exc).__name__}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "error": str(exc),
                        "block_shape": tuple(int(v) for v in block.shape),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "lmax": int(lmax),
                        "factor_kind": str(factor_kind),
                        "fill_factor": float(fill_factor),
                        "drop_tol": float(drop_tol),
                        "diag_pivot_thresh": float(diag_pivot),
                        "allow_block_fallback": bool(allow_block_fallback),
                        "block_scaling_enabled": bool(scale_blocks),
                    },
                )
            factor_nbytes = int(sparse_lu_factor_nbytes(factor))
            factor_nbytes_with_scale = int(factor_nbytes + scale_nbytes)
            if actual_total + factor_nbytes_with_scale > int(max_factor_nbytes):
                if allow_block_fallback:
                    skipped_block_count += 1
                    skipped_budget_count += 1
                    skipped_size += int(positions.size)
                    if len(skipped_errors) < 3:
                        skipped_errors.append(
                            f"factor_budget:{actual_total + factor_nbytes_with_scale}>{int(max_factor_nbytes)}"
                        )
                    continue
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_xblock",
                    reason=f"active_xblock_factor_budget_exceeded:{actual_total + factor_nbytes_with_scale}>{int(max_factor_nbytes)}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "active_size": int(matrix_csr.shape[0]),
                        "block_count": int(len(block_positions) + 1),
                        "covered_size": int(sum(int(pos.size) for pos in block_positions) + int(positions.size)),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "factor_nbytes_actual": int(actual_total + factor_nbytes_with_scale),
                        "max_factor_nbytes": int(max_factor_nbytes),
                        "lmax": int(lmax),
                        "fill_estimate": float(fill_estimate),
                        "fill_factor": float(fill_factor),
                        "drop_tol": float(drop_tol),
                        "factor_kind": str(factor_kind),
                        "allow_block_fallback": bool(allow_block_fallback),
                        "block_scaling_enabled": bool(scale_blocks),
                        "block_scale_nbytes_estimate": int(scale_nbytes),
                    },
                )
            block_positions.append(positions)
            factors.append(factor)
            if scale_blocks:
                block_row_scales.append(row_scale)
                block_col_scales.append(col_scale)
            estimated_total += int(factor_estimate)
            actual_total += int(factor_nbytes_with_scale)
            block_factor_estimates.append(int(factor_estimate))
            block_dense_estimates.append(int(dense_estimate))
            block_sparse_estimates.append(int(sparse_estimate))
            block_matrix_nnz.append(int(block.nnz))
            block_matrix_nbytes.append(int(_scipy_csr_nbytes(block.tocsr())))

    if not factors:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xblock",
            reason="empty_active_xblock_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "lmax": int(lmax),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "allow_block_fallback": bool(allow_block_fallback),
                "skipped_block_count": int(skipped_block_count),
                "skipped_factor_count": int(skipped_factor_count),
                "skipped_budget_count": int(skipped_budget_count),
                "skipped_size": int(skipped_size),
                "skipped_errors": tuple(skipped_errors),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        if base_mode in {"zero", "none"}:
            y_base = np.zeros_like(arr)
        else:
            y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        out = y_base.copy()
        if scale_blocks:
            for positions, factor, row_scale, col_scale in zip(
                block_positions,
                factors,
                block_row_scales,
                block_col_scales,
                strict=True,
            ):
                scaled_residual = row_scale * residual[positions]
                correction = col_scale * np.asarray(factor.solve(scaled_residual), dtype=np.float64).reshape((-1,))
                if np.all(np.isfinite(correction)):
                    out[positions] += float(damping) * correction
        else:
            for positions, factor in zip(block_positions, factors, strict=True):
                correction = np.asarray(factor.solve(residual[positions]), dtype=np.float64).reshape((-1,))
                if np.all(np.isfinite(correction)):
                    out[positions] += float(damping) * correction
        return out

    block_sizes = [int(pos.size) for pos in block_positions]
    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_xblock",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "base_mode": str(base_mode),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "block_count": int(len(factors)),
            "covered_size": int(sum(block_sizes)),
            "block_size_min": int(min(block_sizes)),
            "block_size_max": int(max(block_sizes)),
            "block_matrix_nnz": int(sum(block_matrix_nnz)),
            "block_matrix_nbytes_actual": int(sum(block_matrix_nbytes)),
            "factor_nbytes_estimate": int(estimated_total),
            "sparse_factor_nbytes_estimate": int(sum(block_sparse_estimates)),
            "dense_factor_nbytes_estimate": int(sum(block_dense_estimates)),
            "factor_nbytes_actual": int(actual_total),
            "block_factor_nbytes_estimate_max": int(max(block_factor_estimates)),
            "max_factor_nbytes": int(max_factor_nbytes),
            "lmax": int(lmax),
            "fill_estimate": float(fill_estimate),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "diag_pivot_thresh": float(diag_pivot),
            "factor_kind": str(factor_kind),
            "damping": float(damping),
            "permc_spec": str(permc_spec),
            "allow_block_fallback": bool(allow_block_fallback),
            "skipped_block_count": int(skipped_block_count),
            "skipped_factor_count": int(skipped_factor_count),
            "skipped_budget_count": int(skipped_budget_count),
            "skipped_size": int(skipped_size),
            "skipped_errors": tuple(skipped_errors),
            "covered_fraction": float(sum(block_sizes) / max(int(matrix_csr.shape[0]), 1)),
            "block_scaling_enabled": bool(scale_blocks),
            "block_scale_norm": str(scale_norm),
            "block_scale_max": float(max_scale),
            "block_scale_nbytes_actual": int(
                sum(int(scale.nbytes) for scale in block_row_scales)
                + sum(int(scale.nbytes) for scale in block_col_scales)
            ),
            "row_scale_min": float(min(row_scale_min)) if row_scale_min else 1.0,
            "row_scale_max": float(max(row_scale_max)) if row_scale_max else 1.0,
            "col_scale_min": float(min(col_scale_min)) if col_scale_min else 1.0,
            "col_scale_max": float(max(col_scale_max)) if col_scale_max else 1.0,
        },
    )


def build_active_projected_overlap_schwarz_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a restricted additive-Schwarz residual correction."""

    from scipy.sparse.linalg import LinearOperator, splu  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = (
        np.arange(int(matrix_csr.shape[0]), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_overlap_schwarz",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    radius = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", 1)
    radius = max(0, min(int(layout.n_x) - 1, int(radius)))
    lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", min(8, int(layout.n_xi)))
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    fill_estimate = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_FILL_ESTIMATE", 8.0))
    damping = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_DAMPING", 1.0))
    damping = max(0.0, min(float(damping), 2.0))
    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"

    base = build_jacobi_preconditioner(
        matrix=matrix_csr,
        requested_kind=str(requested_kind),
        regularization=regularization,
        t0=t0,
        reason="active_overlap_schwarz_base_jacobi",
    )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_overlap_schwarz",
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), "lmax": int(lmax), "radius": int(radius)},
        )

    patch_positions: list[np.ndarray] = []
    patch_core_local: list[np.ndarray] = []
    patch_core_global: list[np.ndarray] = []
    factors: list[Any] = []
    patch_factor_estimates: list[int] = []
    patch_sparse_estimates: list[int] = []
    patch_dense_estimates: list[int] = []
    patch_matrix_nnz: list[int] = []
    patch_matrix_nbytes: list[int] = []
    estimated_total = 0
    actual_total = 0
    corrected_total = 0
    covered_total = 0

    for species in range(int(layout.n_species)):
        for x_center in range(int(layout.n_x)):
            core_full = xblock_tz_low_l_indices(layout=layout, species=species, x=x_center, lmax=lmax)
            core_positions = active_positions_for_full_indices(active_indices=active_np, full_indices=core_full)
            if core_positions.size == 0:
                continue
            x_min = max(0, int(x_center) - int(radius))
            x_max = min(int(layout.n_x) - 1, int(x_center) + int(radius))
            patch_full_parts = [
                xblock_tz_low_l_indices(layout=layout, species=species, x=x_patch, lmax=lmax)
                for x_patch in range(x_min, x_max + 1)
            ]
            patch_full = np.unique(np.concatenate(patch_full_parts).astype(np.int64, copy=False))
            positions = active_positions_for_full_indices(active_indices=active_np, full_indices=patch_full)
            if positions.size == 0:
                continue
            core_mask = np.isin(positions, core_positions, assume_unique=True)
            core_local = np.flatnonzero(core_mask).astype(np.int64, copy=False)
            if core_local.size == 0:
                continue
            block = matrix_csr[positions[:, None], positions].tocsc()
            block_scale = max(float(np.max(np.abs(block.data))) if block.nnz else 0.0, 1.0)
            diagonal_shift = max(float(abs(regularization)), 1.0e-14) * block_scale
            if diagonal_shift > 0.0:
                block = block + diagonal_shift * sp.eye(block.shape[0], dtype=np.float64, format="csc")
            sparse_estimate = estimate_spilu_factor_nbytes(matrix=block, fill_factor=fill_estimate)
            dense_estimate = int(block.shape[0] * block.shape[1] * np.dtype(np.float64).itemsize)
            factor_estimate = max(int(sparse_estimate), int(dense_estimate))
            if estimated_total + factor_estimate > int(max_factor_nbytes):
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_overlap_schwarz",
                    reason=f"active_overlap_schwarz_budget_exceeded:{estimated_total + factor_estimate}>{int(max_factor_nbytes)}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "active_size": int(matrix_csr.shape[0]),
                        "patch_count": int(len(patch_positions) + 1),
                        "covered_size": int(covered_total + positions.size),
                        "corrected_size": int(corrected_total + core_local.size),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "max_factor_nbytes": int(max_factor_nbytes),
                        "lmax": int(lmax),
                        "radius": int(radius),
                        "fill_estimate": float(fill_estimate),
                    },
                )
            try:
                factor = splu(block, permc_spec=str(permc_spec), diag_pivot_thresh=0.0)
            except Exception as exc:  # noqa: BLE001
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_overlap_schwarz",
                    reason=f"active_overlap_schwarz_factor_failed:{type(exc).__name__}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "error": str(exc),
                        "block_shape": tuple(int(v) for v in block.shape),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "lmax": int(lmax),
                        "radius": int(radius),
                    },
                )
            factor_nbytes = int(sparse_lu_factor_nbytes(factor))
            if actual_total + factor_nbytes > int(max_factor_nbytes):
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_overlap_schwarz",
                    reason=f"active_overlap_schwarz_factor_budget_exceeded:{actual_total + factor_nbytes}>{int(max_factor_nbytes)}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "active_size": int(matrix_csr.shape[0]),
                        "patch_count": int(len(patch_positions) + 1),
                        "covered_size": int(covered_total + positions.size),
                        "corrected_size": int(corrected_total + core_local.size),
                        "factor_nbytes_estimate": int(estimated_total + factor_estimate),
                        "factor_nbytes_actual": int(actual_total + factor_nbytes),
                        "max_factor_nbytes": int(max_factor_nbytes),
                        "lmax": int(lmax),
                        "radius": int(radius),
                        "fill_estimate": float(fill_estimate),
                    },
                )

            patch_positions.append(positions)
            patch_core_local.append(core_local)
            patch_core_global.append(positions[core_local])
            factors.append(factor)
            estimated_total += int(factor_estimate)
            actual_total += int(factor_nbytes)
            corrected_total += int(core_local.size)
            covered_total += int(positions.size)
            patch_factor_estimates.append(int(factor_estimate))
            patch_sparse_estimates.append(int(sparse_estimate))
            patch_dense_estimates.append(int(dense_estimate))
            patch_matrix_nnz.append(int(block.nnz))
            patch_matrix_nbytes.append(int(_scipy_csr_nbytes(block.tocsr())))

    if not factors:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_overlap_schwarz",
            reason="empty_active_overlap_schwarz_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"lmax": int(lmax), "radius": int(radius)},
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        out = y_base.copy()
        for positions, core_local, core_global, factor in zip(
            patch_positions,
            patch_core_local,
            patch_core_global,
            factors,
            strict=True,
        ):
            correction = np.asarray(factor.solve(residual[positions]), dtype=np.float64).reshape((-1,))
            out[core_global] += damping * correction[core_local]
        return out

    patch_sizes = [int(pos.size) for pos in patch_positions]
    core_sizes = [int(core.size) for core in patch_core_local]
    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_overlap_schwarz",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "patch_count": int(len(factors)),
            "covered_size": int(covered_total),
            "corrected_size": int(corrected_total),
            "patch_size_min": int(min(patch_sizes)),
            "patch_size_max": int(max(patch_sizes)),
            "core_size_min": int(min(core_sizes)),
            "core_size_max": int(max(core_sizes)),
            "patch_matrix_nnz": int(sum(patch_matrix_nnz)),
            "patch_matrix_nbytes_actual": int(sum(patch_matrix_nbytes)),
            "factor_nbytes_estimate": int(estimated_total),
            "sparse_factor_nbytes_estimate": int(sum(patch_sparse_estimates)),
            "dense_factor_nbytes_estimate": int(sum(patch_dense_estimates)),
            "factor_nbytes_actual": int(actual_total),
            "patch_factor_nbytes_estimate_max": int(max(patch_factor_estimates)),
            "max_factor_nbytes": int(max_factor_nbytes),
            "lmax": int(lmax),
            "radius": int(radius),
            "damping": float(damping),
            "fill_estimate": float(fill_estimate),
            "permc_spec": str(permc_spec),
        },
    )


def active_positions_for_full_indices(*, active_indices: Any, full_indices: Any) -> np.ndarray:
    """Map original full-system indices into active-projected matrix positions."""

    active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    full = np.asarray(full_indices, dtype=np.int64).reshape((-1,))
    if active.size == 0 or full.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(active)
    sorted_active = active[order]
    loc = np.searchsorted(sorted_active, full)
    valid = loc < sorted_active.size
    valid[valid] &= sorted_active[loc[valid]] == full[valid]
    if not np.any(valid):
        return np.zeros((0,), dtype=np.int64)
    return np.unique(order[loc[valid]].astype(np.int64, copy=False))


def _scipy_csr_nbytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return bool(default)
    return raw in {"1", "true", "yes", "on"}
