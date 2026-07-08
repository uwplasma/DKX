"""Active-projected x-block preconditioners for RHSMode=1 full-CSR solves.

These builders operate on an already active-projected CSR matrix while using
the original RHSMode=1 layout to recover physical ``(species, x, ell)`` blocks.
They are host-side, non-autodiff preconditioner setup utilities for the explicit
CSR solve lane; differentiable paths should stay on JAX-native operators.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_layout import RHS1ActiveBlockLayout, RHS1BlockLayout
from .preconditioner_schur_profile import (
    RHS1StructuredFullCSRPreconditioner,
    build_jacobi_preconditioner,
    safe_inverse_diagonal,
)
from .preconditioner_schur_profile import (
    append_adaptive_residual_basis_csc,
    build_active_native_xell_coarse_window_basis_csc,
    build_coarse_residual_basis_csc,
    coarse_residual_config,
)
from .preconditioner_schur_profile import resolve_active_native_stack_policy
from .preconditioner_reduced_pmat import (
    estimate_spilu_factor_nbytes,
    sparse_equilibration_scale,
    sparse_lu_factor_nbytes,
)
from .preconditioner_xblock_low_l_schur import build_native_xell_kinetic_preconditioner, xblock_tz_low_l_indices

__all__ = (
    "active_positions_for_full_indices",
    "build_active_fortran_v3_reduced_native_stack_preconditioner",
    "build_active_projected_bounded_native_stack_preconditioner",
    "build_active_projected_global_field_split_schur_preconditioner",
    "build_active_projected_multiline_field_split_base_preconditioner",
    "build_active_projected_angular_line_preconditioner",
    "build_active_projected_diagonal_schur_preconditioner",
    "build_active_projected_native_indexed_schwarz_preconditioner",
    "build_active_projected_overlap_schwarz_preconditioner",
    "build_active_projected_xell_kinetic_line_preconditioner",
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


def build_active_projected_native_indexed_schwarz_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded JAX-native overlapping line-block Schwarz factor.

    This combines fixed ``(species, theta, zeta)`` speed/pitch lines with
    fixed ``(species, x, ell)`` angular lines.  Setup happens on the host, while
    application uses a JAX padded-indexed block factor so the preconditioner can
    be called from device Krylov paths.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    from sfincs_jax.solvers.native_block_factor import (  # noqa: PLC0415
        apply_native_padded_indexed_block_factor,
        build_native_padded_indexed_block_factor,
    )

    matrix_csr = matrix.tocsr()
    active_full = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix_csr.shape)},
        )
    if active_full.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_full.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    if active_full.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="empty_active_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )
    if np.any(active_full < 0) or np.any(active_full >= int(layout.total_size)):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="active_indices_outside_full_vector",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "layout_total_size": int(layout.total_size),
                "active_min": int(np.min(active_full)),
                "active_max": int(np.max(active_full)),
            },
        )

    include_xell = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_INCLUDE_XELL", True)
    include_angular = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_INCLUDE_ANGULAR", True)
    include_tail_jacobi = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_INCLUDE_TAIL_JACOBI", True)
    normalize_overlap = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_NORMALIZE_OVERLAP", True)
    damping = max(0.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_DAMPING", 1.0)))
    min_block_size = max(1, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MIN_BLOCK_SIZE", 2)))
    max_block_size = max(1, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MAX_BLOCK_SIZE", 512)))
    max_blocks = max(1, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MAX_BLOCKS", 200000)))

    kinetic_mask = active_full < int(layout.f_size)
    kinetic_positions = np.flatnonzero(kinetic_mask).astype(np.int64, copy=False)
    tail_positions = np.flatnonzero(~kinetic_mask).astype(np.int64, copy=False)
    if kinetic_positions.size == 0:
        return build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="active_native_indexed_no_kinetic_rows",
        )

    active_kinetic_full = active_full[kinetic_positions]
    decoded = layout.decode_kinetic_indices(active_kinetic_full)
    positions_by_block: list[np.ndarray] = []
    block_families: list[str] = []
    skipped_too_small = 0
    skipped_too_large = 0

    def append_groups(line_ids: np.ndarray, family: str) -> None:
        nonlocal skipped_too_large, skipped_too_small
        for line_id in np.unique(line_ids):
            positions = kinetic_positions[np.flatnonzero(line_ids == line_id)].astype(np.int64, copy=False)
            if int(positions.size) < int(min_block_size):
                skipped_too_small += 1
                continue
            if int(positions.size) > int(max_block_size):
                skipped_too_large += 1
                continue
            positions_by_block.append(positions)
            block_families.append(str(family))

    if bool(include_xell):
        xell_ids = (
            (decoded.species.astype(np.int64, copy=False) * int(layout.n_theta) + decoded.theta)
            * int(layout.n_zeta)
            + decoded.zeta
        )
        append_groups(xell_ids, "xell")
    if bool(include_angular):
        angular_ids = (
            (decoded.species.astype(np.int64, copy=False) * int(layout.n_x) + decoded.x)
            * int(layout.n_xi)
            + decoded.ell
        )
        append_groups(angular_ids, "angular")

    if not positions_by_block:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="empty_active_native_indexed_block_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "include_xell": bool(include_xell),
                "include_angular": bool(include_angular),
                "skipped_too_small": int(skipped_too_small),
                "skipped_too_large": int(skipped_too_large),
                "min_block_size": int(min_block_size),
                "max_block_size": int(max_block_size),
            },
        )
    if len(positions_by_block) > int(max_blocks):
        positions_by_block = positions_by_block[: int(max_blocks)]
        block_families = block_families[: int(max_blocks)]
        truncated_blocks = True
    else:
        truncated_blocks = False

    block_sizes = np.asarray([int(pos.size) for pos in positions_by_block], dtype=np.int64)
    padded_block_size = int(np.max(block_sizes))
    n_blocks = int(len(positions_by_block))
    inverse_nbytes_estimate = int(n_blocks * padded_block_size * padded_block_size * np.dtype(np.float64).itemsize)
    index_nbytes_estimate = int(n_blocks * padded_block_size * np.dtype(np.int32).itemsize)
    mask_nbytes_estimate = int(n_blocks * padded_block_size * np.dtype(np.bool_).itemsize)
    weights_nbytes_estimate = int(matrix_csr.shape[0] * np.dtype(np.float64).itemsize)
    tail_nbytes_estimate = int(tail_positions.size * np.dtype(np.float64).itemsize) if include_tail_jacobi else 0
    factor_estimate = int(
        inverse_nbytes_estimate
        + index_nbytes_estimate
        + mask_nbytes_estimate
        + weights_nbytes_estimate
        + tail_nbytes_estimate
    )
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason=f"active_native_indexed_schwarz_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "inverse_nbytes_estimate": int(inverse_nbytes_estimate),
                "index_nbytes_estimate": int(index_nbytes_estimate),
                "mask_nbytes_estimate": int(mask_nbytes_estimate),
                "weights_nbytes_estimate": int(weights_nbytes_estimate),
                "tail_nbytes_estimate": int(tail_nbytes_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(n_blocks),
                "padded_block_size": int(padded_block_size),
                "block_size_min": int(np.min(block_sizes)),
                "block_size_max": int(np.max(block_sizes)),
            },
        )

    padded_inverses = np.zeros((n_blocks, padded_block_size, padded_block_size), dtype=np.float64)
    padded_indices = np.zeros((n_blocks, padded_block_size), dtype=np.int32)
    padded_mask = np.zeros((n_blocks, padded_block_size), dtype=bool)
    regularized_count = 0
    singular_count = 0
    nonfinite_count = 0
    condition_nonfinite_count = 0
    max_condition_estimate: float | None = 0.0 if padded_block_size <= 64 else None
    max_block_scale = 0.0
    block_matrix_nnz = 0
    for block_index, positions in enumerate(positions_by_block):
        block = np.asarray(matrix_csr[positions[:, None], positions].toarray(), dtype=np.float64)
        block_matrix_nnz += int(np.count_nonzero(block))
        block_scale = max(float(np.linalg.norm(block, ord=np.inf)) if block.size else 0.0, 1.0)
        max_block_scale = max(float(max_block_scale), float(block_scale))
        regularization_abs = float(abs(regularization)) * float(block_scale)
        if regularization_abs > 0.0:
            block = block + regularization_abs * np.eye(int(block.shape[0]), dtype=np.float64)
            regularized_count += 1
        if max_condition_estimate is not None:
            condition_estimate = float(np.linalg.cond(block))
            if np.isfinite(condition_estimate):
                max_condition_estimate = max(float(max_condition_estimate), condition_estimate)
            else:
                condition_nonfinite_count += 1
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse = np.linalg.pinv(block, rcond=max(float(abs(regularization)), 1.0e-14))
        if not np.all(np.isfinite(inverse)):
            nonfinite_count += 1
        block_size = int(positions.size)
        padded_inverses[block_index, :block_size, :block_size] = np.asarray(inverse, dtype=np.float64)
        padded_indices[block_index, :block_size] = positions.astype(np.int32, copy=False)
        padded_mask[block_index, :block_size] = True

    if int(nonfinite_count) > 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason="active_native_indexed_schwarz_inverse_nonfinite",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"block_inverse_nonfinite_count": int(nonfinite_count), "n_blocks": int(n_blocks)},
        )

    inv_tail = np.empty((0,), dtype=np.float64)
    tail_metadata: dict[str, object] = {"tail_size": int(tail_positions.size)}
    if bool(include_tail_jacobi) and tail_positions.size:
        inv_tail, tail_metadata = safe_inverse_diagonal(
            matrix_csr.diagonal()[tail_positions],
            regularization=regularization,
        )
    actual_nbytes = int(
        padded_inverses.nbytes
        + padded_indices.nbytes
        + padded_mask.nbytes
        + matrix_csr.shape[0] * np.dtype(np.float64).itemsize
        + inv_tail.nbytes
    )
    if actual_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_indexed_schwarz",
            reason=f"active_native_indexed_schwarz_budget_exceeded_actual:{actual_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "factor_nbytes_actual": int(actual_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(n_blocks),
                "padded_block_size": int(padded_block_size),
            },
        )

    native_factor = build_native_padded_indexed_block_factor(
        block_inverses=jnp.asarray(padded_inverses),
        block_indices=jnp.asarray(padded_indices),
        block_mask=jnp.asarray(padded_mask),
        total_size=int(matrix_csr.shape[0]),
        normalize_overlap=bool(normalize_overlap),
        damping=float(damping),
    )

    def apply(rhs: Any) -> np.ndarray:
        rhs_vec = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if rhs_vec.shape != (int(matrix_csr.shape[0]),):
            raise ValueError(f"rhs must have shape {(int(matrix_csr.shape[0]),)}, got {rhs_vec.shape}")
        out = np.array(apply_native_padded_indexed_block_factor(native_factor, jnp.asarray(rhs_vec)), dtype=np.float64)
        if bool(include_tail_jacobi) and tail_positions.size:
            out[tail_positions] = inv_tail * rhs_vec[tail_positions]
        return out

    family_counts = {family: int(block_families.count(family)) for family in sorted(set(block_families))}
    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_native_indexed_schwarz",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "jax_native_overlapping_indexed_line_blocks",
            "backend": "jax_native_padded_indexed_block_factor",
            "active_size": int(matrix_csr.shape[0]),
            "kinetic_size": int(kinetic_positions.size),
            "tail_size": int(tail_positions.size),
            "n_blocks": int(n_blocks),
            "block_family_counts": family_counts,
            "block_size_min": int(np.min(block_sizes)),
            "block_size_max": int(np.max(block_sizes)),
            "block_size_mean": float(np.mean(block_sizes)),
            "padded_block_size": int(padded_block_size),
            "block_matrix_nnz": int(block_matrix_nnz),
            "include_xell": bool(include_xell),
            "include_angular": bool(include_angular),
            "include_tail_jacobi": bool(include_tail_jacobi),
            "normalize_overlap": bool(normalize_overlap),
            "damping": float(damping),
            "truncated_blocks": bool(truncated_blocks),
            "max_blocks": int(max_blocks),
            "skipped_too_small": int(skipped_too_small),
            "skipped_too_large": int(skipped_too_large),
            "min_block_size": int(min_block_size),
            "max_block_size": int(max_block_size),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_actual": int(actual_nbytes),
            "inverse_nbytes_actual": int(padded_inverses.nbytes),
            "index_nbytes_actual": int(padded_indices.nbytes),
            "mask_nbytes_actual": int(padded_mask.nbytes),
            "overlap_weight_nbytes_actual": int(matrix_csr.shape[0] * np.dtype(np.float64).itemsize),
            "tail_inverse_nbytes_actual": int(inv_tail.nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "regularization": float(regularization),
            "block_inverse_regularized_count": int(regularized_count),
            "block_inverse_singular_count": int(singular_count),
            "block_inverse_nonfinite_count": int(nonfinite_count),
            "block_inverse_condition_estimate_max": max_condition_estimate,
            "block_inverse_condition_nonfinite_count": int(condition_nonfinite_count),
            "block_inverse_scale_max": float(max_block_scale),
            "layout": layout.to_dict(),
            **{f"tail_{key}": value for key, value in tail_metadata.items()},
        },
    )


def build_active_projected_global_field_split_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    base_preconditioner_factory: Callable[..., RHS1StructuredFullCSRPreconditioner] | None = None,
    base_kind_override: str | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a coupled active kinetic/global Schur preconditioner.

    This opt-in path is the active-system counterpart of a PETSc-style
    field-split block factorization.  It uses an existing local active kinetic
    preconditioner as an approximate ``K^{-1}``, explicitly forms the compact
    global Schur residual equation ``S = W - V K^{-1} U``, and applies the
    corresponding block inverse.  It is intentionally fail-closed and is not an
    automatic default until QA/QH production gates demonstrate net benefit.
    """

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    try:
        active_layout = RHS1ActiveBlockLayout.from_layout(
            layout,
            None if active_indices is None else active_np,
        )
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="invalid_active_layout",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    if int(active_layout.active_size) != int(matrix_csr.shape[0]):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_layout": active_layout.to_dict(),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    if int(active_layout.phi1_count) != 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="active_phi1_tail_split_unsupported",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"active_layout": active_layout.to_dict()},
        )
    if int(active_layout.extra_count) <= 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="active_no_global_tail",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"active_layout": active_layout.to_dict()},
        )
    if not bool(active_layout.has_contiguous_extra_tail):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="active_tail_not_contiguous",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"active_layout": active_layout.to_dict()},
        )

    n_k = int(active_layout.kinetic_count)
    tail_size = int(active_layout.extra_count)
    if n_k + tail_size != int(matrix_csr.shape[0]):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason="active_kinetic_tail_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "kinetic_size": int(n_k),
                "tail_size": int(tail_size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
                "active_layout": active_layout.to_dict(),
            },
        )

    max_schur_size = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_MAX_TAIL", 256))
    max_schur_size = max(1, int(max_schur_size))
    if tail_size > max_schur_size:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason=f"active_global_field_split_tail_size_exceeded:{tail_size}>{max_schur_size}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "kinetic_size": int(n_k),
                "tail_size": int(tail_size),
                "max_tail_size": int(max_schur_size),
                "active_layout": active_layout.to_dict(),
            },
        )

    base_kind = (
        str(base_kind_override)
        if base_kind_override is not None
        else os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_BASE", "active_xblock")
    )
    base_kind = base_kind.strip().lower().replace("-", "_")
    if base_kind in {
        "",
        "auto",
        "active_auto",
        "active_global_field_split_schur",
        "active_field_split_schur",
        "active_xblock_global_schur",
    }:
        base_kind = "active_xblock"

    kinetic_matrix = matrix_csr[:n_k, :n_k].tocsc()
    kinetic_active = active_np[:n_k].astype(np.int64, copy=False)
    base_budget = max(1, int(max_factor_nbytes))
    if base_kind in {
        "active_native_xell",
        "active_native_x_ell",
        "native_xell",
        "native_x_ell",
        "jax_native_xell",
        "jax_xell",
    }:
        full_kinetic_active = (
            int(n_k) == int(layout.f_size)
            and np.array_equal(kinetic_active, np.arange(int(layout.f_size), dtype=np.int64))
        )
        if bool(full_kinetic_active):
            kinetic_layout = RHS1BlockLayout(
                n_species=int(layout.n_species),
                n_x=int(layout.n_x),
                n_xi=int(layout.n_xi),
                n_theta=int(layout.n_theta),
                n_zeta=int(layout.n_zeta),
                f_size=int(layout.f_size),
                phi1_size=0,
                extra_size=0,
                total_size=int(layout.f_size),
                constraint_scheme=int(layout.constraint_scheme),
                include_phi1=False,
                include_phi1_in_kinetic=False,
                rhs_mode=int(layout.rhs_mode),
            )
            base = build_native_xell_kinetic_preconditioner(
                matrix=kinetic_matrix,
                layout=kinetic_layout,
                requested_kind=base_kind,
                regularization=regularization,
                max_factor_nbytes=base_budget,
                t0=t0,
            )
        else:
            base = build_active_projected_xell_kinetic_line_preconditioner(
                matrix=kinetic_matrix,
                layout=layout,
                active_kinetic_indices=kinetic_active,
                requested_kind=base_kind,
                regularization=regularization,
                max_factor_nbytes=base_budget,
                t0=t0,
            )
    else:
        base = _build_active_projected_local_base_preconditioner(
            matrix=kinetic_matrix,
            layout=layout,
            active_indices=kinetic_active,
            kind=base_kind,
            max_factor_nbytes=base_budget,
            regularization=regularization,
            t0=t0,
        )
        if (not bool(base.selected) or base.operator is None) and base.reason == "unsupported_local_base_kind":
            base = (
                None
                if base_preconditioner_factory is None
                else base_preconditioner_factory(
                    matrix=kinetic_matrix,
                    layout=layout,
                    active_indices=kinetic_active,
                    kind=base_kind,
                    max_factor_nbytes=base_budget,
                    regularization=regularization,
                )
            )
        if base is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_global_field_split_schur",
                reason="active_global_field_split_missing_base_factory",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "requested_kind": str(requested_kind),
                    "requested_base_kind": str(base_kind),
                    "kinetic_size": int(n_k),
                    "tail_size": int(tail_size),
                    "active_layout": active_layout.to_dict(),
                },
            )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason=f"active_global_field_split_base_failed:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "kinetic_size": int(n_k),
                "tail_size": int(tail_size),
                "active_layout": active_layout.to_dict(),
            },
        )

    u = matrix_csr[:n_k, n_k:].tocsc()
    v = matrix_csr[n_k:, :n_k].tocsr()
    w = matrix_csr[n_k:, n_k:].tocsr()
    base_factor_nbytes = int(
        base.metadata.get("factor_nbytes_actual", base.metadata.get("factor_nbytes_estimate", 0)) or 0
    )
    kinetic_response_nbytes = int(n_k * tail_size * np.dtype(np.float64).itemsize)
    schur_nbytes = int(tail_size * tail_size * np.dtype(np.float64).itemsize)
    partition_nbytes = int(_scipy_csr_nbytes(u.tocsr()) + _scipy_csr_nbytes(v) + _scipy_csr_nbytes(w))
    factor_estimate = int(base_factor_nbytes + kinetic_response_nbytes + schur_nbytes + partition_nbytes)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_global_field_split_schur",
            reason=f"active_global_field_split_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "base_kind": str(base.kind),
                "requested_base_kind": str(base_kind),
                "kinetic_size": int(n_k),
                "tail_size": int(tail_size),
                "kinetic_response_nbytes": int(kinetic_response_nbytes),
                "schur_nbytes": int(schur_nbytes),
                "partition_nbytes": int(partition_nbytes),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "factor_nbytes_estimate": int(factor_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "active_layout": active_layout.to_dict(),
                "base_preconditioner": base.to_dict(),
            },
        )

    def solve_kinetic(rhs_kinetic: Any) -> np.ndarray:
        return np.asarray(base.operator.matvec(rhs_kinetic), dtype=np.float64).reshape((-1,))

    kinetic_response = np.zeros((n_k, tail_size), dtype=np.float64)
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u.indptr[col_index])
        stop = int(u.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_k,), dtype=np.float64)
        column[u.indices[start:stop]] = u.data[start:stop]
        kinetic_response[:, col_index] = solve_kinetic(column)

    schur = np.asarray(w.toarray(), dtype=np.float64) - np.asarray(v @ kinetic_response, dtype=np.float64)
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    schur_solver_kind = "lu"
    try:
        lu, piv = lu_factor(schur, check_finite=False)

        def solve_tail(rhs_tail: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs_tail, check_finite=False), dtype=np.float64).reshape((-1,))

    except Exception:  # noqa: BLE001
        pinv = np.linalg.pinv(schur, rcond=max(float(abs(regularization)), 1.0e-14))
        schur_solver_kind = "pinv"

        def solve_tail(rhs_tail: np.ndarray) -> np.ndarray:
            return np.asarray(pinv @ rhs_tail, dtype=np.float64).reshape((-1,))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_k = solve_kinetic(arr[:n_k])
        rhs_tail = arr[n_k:] - np.asarray(v @ y_k, dtype=np.float64).reshape((-1,))
        y_tail = solve_tail(rhs_tail)
        y_k = y_k - np.asarray(kinetic_response @ y_tail, dtype=np.float64).reshape((-1,))
        return np.concatenate((y_k, y_tail))

    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_global_field_split_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_kinetic_global_field_split_schur",
            "base_kind": str(base.kind),
            "requested_base_kind": str(base_kind),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "kinetic_size": int(n_k),
            "tail_size": int(tail_size),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "kinetic_response_nbytes": int(kinetic_response.nbytes),
            "schur_nbytes": int(schur.nbytes),
            "partition_nbytes": int(partition_nbytes),
            "base_factor_nbytes_actual": int(base_factor_nbytes),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_actual": int(factor_estimate),
            "max_factor_nbytes": int(max_factor_nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            "schur_solver_kind": str(schur_solver_kind),
            "active_layout": active_layout.to_dict(),
            "note": "opt_in_probe_not_auto_default",
        },
    )


def _build_active_projected_local_base_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    kind: str,
    max_factor_nbytes: int,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Resolve active x-block base preconditioners owned by this module."""

    kind_l = str(kind).strip().lower().replace("-", "_")
    if kind_l in {"active_xblock", "active_x_block", "active_block_x", "active_xblock_spilu", "active_xblock_lu"}:
        return build_active_projected_xblock_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_angular_line", "active_tz_line", "active_theta_zeta_line"}:
        return build_active_projected_angular_line_preconditioner(
            matrix=matrix,
            layout=layout,
            active_kinetic_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_diagonal_schur", "active_diag_schur"}:
        return build_active_projected_diagonal_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_overlap_schwarz", "active_additive_schwarz", "active_restricted_schwarz", "active_ras"}:
        return build_active_projected_overlap_schwarz_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_native_indexed_schwarz", "active_native_indexed_asm", "active_native_indexed_lines"}:
        return build_active_projected_native_indexed_schwarz_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    return RHS1StructuredFullCSRPreconditioner(
        operator=None,
        selected=False,
        kind=str(kind_l or "active_unknown"),
        reason="unsupported_local_base_kind",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={"requested_kind": str(kind)},
    )


def build_active_projected_multiline_field_split_base_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    base_preconditioner_factory: Callable[..., RHS1StructuredFullCSRPreconditioner] | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Compose x-ell and angular-line field splits through a true residual equation.

    The default multiplicative form applies the native ``(x, ell)`` field-split
    inverse, computes the actual active-system residual, then applies the
    angular-line field-split inverse to that residual.  This gives the sparse
    coarse layer a stronger base than either line factor alone while preserving
    the same fail-closed memory accounting.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    xell = build_active_projected_global_field_split_schur_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_global_field_split_schur",
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
        base_preconditioner_factory=base_preconditioner_factory,
        base_kind_override="active_native_xell",
    )
    angular = build_active_projected_global_field_split_schur_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_global_field_split_schur",
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
        base_preconditioner_factory=base_preconditioner_factory,
        base_kind_override="active_angular_line",
    )
    if not bool(xell.selected) or xell.operator is None or not bool(angular.selected) or angular.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_multiline_field_split_base",
            reason=f"active_multiline_base_failed:xell={xell.reason};angular={angular.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "xell_preconditioner": xell.to_dict(),
                "angular_preconditioner": angular.to_dict(),
            },
        )

    xell_nbytes = int(xell.metadata.get("factor_nbytes_actual", xell.metadata.get("factor_nbytes_estimate", 0)) or 0)
    angular_nbytes = int(
        angular.metadata.get("factor_nbytes_actual", angular.metadata.get("factor_nbytes_estimate", 0)) or 0
    )
    factor_nbytes = int(xell_nbytes + angular_nbytes)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_multiline_field_split_base",
            reason=f"active_multiline_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "xell_preconditioner": xell.to_dict(),
                "angular_preconditioner": angular.to_dict(),
                "xell_factor_nbytes_actual": int(xell_nbytes),
                "angular_factor_nbytes_actual": int(angular_nbytes),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_MULTILINE_MODE", "multiplicative")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if mode not in {"multiplicative", "xell_then_angular", "angular_then_xell", "additive"}:
        mode = "multiplicative"
    xell_weight = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_MULTILINE_XELL_WEIGHT", 1.0))
    angular_weight = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_MULTILINE_ANGULAR_WEIGHT", 1.0))

    def _matvec_xell(arr: np.ndarray) -> np.ndarray:
        return np.asarray(xell.operator.matvec(arr), dtype=np.float64).reshape((-1,))

    def _matvec_angular(arr: np.ndarray) -> np.ndarray:
        return np.asarray(angular.operator.matvec(arr), dtype=np.float64).reshape((-1,))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        if mode == "additive":
            denom = max(abs(xell_weight) + abs(angular_weight), 1.0e-300)
            return (xell_weight * _matvec_xell(arr) + angular_weight * _matvec_angular(arr)) / denom
        if mode == "angular_then_xell":
            first = _matvec_angular(arr)
            residual = arr - np.asarray(matrix_csr @ first, dtype=np.float64).reshape((-1,))
            return first + xell_weight * _matvec_xell(residual)
        first = _matvec_xell(arr)
        residual = arr - np.asarray(matrix_csr @ first, dtype=np.float64).reshape((-1,))
        return first + angular_weight * _matvec_angular(residual)

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_multiline_field_split_base",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": "active_multiline_field_split_base",
            "architecture": "active_multiline_xell_angular_field_split_residual",
            "base_kind": "active_multiline_xell_angular",
            "requested_base_kind": "active_multiline_xell_angular",
            "mode": str(mode),
            "xell_weight": float(xell_weight),
            "angular_weight": float(angular_weight),
            "xell_preconditioner": xell.to_dict(),
            "angular_preconditioner": angular.to_dict(),
            "xell_factor_nbytes_actual": int(xell_nbytes),
            "angular_factor_nbytes_actual": int(angular_nbytes),
            "factor_nbytes_actual": int(factor_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "note": "opt_in_probe_not_auto_default",
        },
    )


def build_active_projected_bounded_native_stack_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    base_preconditioner_factory: Callable[..., RHS1StructuredFullCSRPreconditioner] | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded native line/patch/coarse stack for RHSMode=1.

    This is the production-facing replacement for monolithic serial sparse
    factor setup.  The stack uses only bounded components:

    - native ``(x, ell)`` line solves at fixed angular grid point,
    - native angular-line solves at fixed ``(species, x, ell)``,
    - an optional local additive-Schwarz patch layer under a setup-size gate,
    - and a compact coupled coarse residual equation over physics/profile
      moments plus optional targeted windows.

    The final true residual is still owned by Krylov.  This object is a fixed
    linear preconditioner and is therefore safe to use in GMRES/FGMRES policy
    probes without any right-hand-side-dependent tuning.
    """

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

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
            kind="active_bounded_native_stack",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    policy = resolve_active_native_stack_policy(max_factor_nbytes=int(max_factor_nbytes))
    base_budget = int(policy.base_budget_nbytes)
    base = build_active_projected_multiline_field_split_base_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        regularization=regularization,
        max_factor_nbytes=base_budget,
        t0=t0,
        base_preconditioner_factory=base_preconditioner_factory,
    )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_bounded_native_stack",
            reason=f"line_factor_base_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"line_base_preconditioner": base.to_dict(), "base_budget_nbytes": int(base_budget)},
        )

    base_nbytes = int(base.metadata.get("factor_nbytes_actual", base.metadata.get("factor_nbytes_estimate", 0)) or 0)
    schwarz_requested = bool(policy.schwarz_requested)
    schwarz_max_size = int(policy.schwarz_max_size)
    schwarz: RHS1StructuredFullCSRPreconditioner | None = None
    schwarz_nbytes = 0
    schwarz_reason = "disabled"
    if bool(schwarz_requested):
        if int(schwarz_max_size) > 0 and int(matrix_csr.shape[0]) > int(schwarz_max_size):
            schwarz_reason = f"size_exceeded:{int(matrix_csr.shape[0])}>{int(schwarz_max_size)}"
        else:
            remaining_budget = max(1, int(max_factor_nbytes) - int(base_nbytes))
            schwarz = build_active_projected_overlap_schwarz_preconditioner(
                matrix=matrix_csr,
                layout=layout,
                active_indices=active_np,
                requested_kind="active_overlap_schwarz",
                regularization=regularization,
                max_factor_nbytes=remaining_budget,
                t0=t0,
            )
            if bool(schwarz.selected) and schwarz.operator is not None:
                schwarz_reason = "selected"
                schwarz_nbytes = int(
                    schwarz.metadata.get("factor_nbytes_actual", schwarz.metadata.get("factor_nbytes_estimate", 0))
                    or 0
                )
            else:
                schwarz_reason = f"not_selected:{getattr(schwarz, 'reason', 'unknown')}"
                schwarz = None

    config = coarse_residual_config(layout)
    max_coarse_size = int(policy.max_coarse_size)
    window_basis, window_metadata = build_active_native_xell_coarse_window_basis_csc(layout=layout)
    physics_basis = build_coarse_residual_basis_csc(layout=layout, config=config)
    full_basis = (
        sp.hstack([window_basis, physics_basis], format="csc")
        if int(window_basis.shape[1]) > 0
        else physics_basis
    )
    basis = full_basis[active_np, :].tocsc()
    if basis.shape[1] == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_bounded_native_stack",
            reason="empty_projected_native_stack_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "line_base_preconditioner": base.to_dict(),
                "schwarz_reason": str(schwarz_reason),
                **window_metadata,
                **config,
            },
        )

    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    keep = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if keep.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_bounded_native_stack",
            reason="zero_projected_native_stack_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "line_base_preconditioner": base.to_dict(),
                "schwarz_reason": str(schwarz_reason),
                **window_metadata,
                **config,
            },
        )
    if keep.size > int(max_coarse_size):
        keep = keep[: int(max_coarse_size)]
    basis = basis[:, keep].tocsc()
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    valid = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if valid.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_bounded_native_stack",
            reason="invalid_projected_native_stack_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "line_base_preconditioner": base.to_dict(),
                "schwarz_reason": str(schwarz_reason),
                **window_metadata,
                **config,
            },
        )
    basis = basis[:, valid].tocsc()
    col_norm = col_norm[valid]
    basis = (basis @ sp.diags(1.0 / col_norm, format="csc")).tocsc()
    basis.eliminate_zeros()
    adaptive_base_operator = base.operator
    adaptive_residual_operator = "line_base"
    if schwarz is not None and schwarz.operator is not None:
        adaptive_residual_operator = "line_base_plus_schwarz"

        def _apply_base_plus_schwarz_for_adaptive_basis(x: Any) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float64).reshape((-1,))
            y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
            residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
            return y_base + np.asarray(schwarz.operator.matvec(residual), dtype=np.float64).reshape((-1,))

        adaptive_base_operator = LinearOperator(
            matrix_csr.shape,
            matvec=_apply_base_plus_schwarz_for_adaptive_basis,
            dtype=np.float64,
        )
    basis, adaptive_metadata = append_adaptive_residual_basis_csc(
        matrix=matrix_csr,
        base_operator=adaptive_base_operator,
        basis=basis,
        max_total_columns=int(max_coarse_size),
    )

    az_basis = (matrix_csr @ basis).tocsc()
    coarse_solver_mode = str(policy.coarse_solver_mode)
    if coarse_solver_mode == "galerkin":
        coarse = np.asarray((basis.T @ az_basis).toarray(), dtype=np.float64)
    else:
        coarse = np.asarray((az_basis.T @ az_basis).toarray(), dtype=np.float64)
    coarse_size = int(coarse.shape[0])
    coarse_scale = max(float(np.linalg.norm(coarse, ord=np.inf)) if coarse.size else 0.0, 1.0)
    coarse_regularization = max(float(abs(regularization)), 1.0e-14) * coarse_scale
    coarse_reg = coarse + coarse_regularization * np.eye(coarse_size, dtype=np.float64)
    basis_nbytes = int(_scipy_csr_nbytes(basis.tocsr()))
    az_basis_nbytes = int(_scipy_csr_nbytes(az_basis.tocsr()))
    coarse_nbytes = int(coarse_reg.nbytes)
    total_nbytes = int(base_nbytes + schwarz_nbytes + basis_nbytes + az_basis_nbytes + coarse_nbytes)
    if total_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_bounded_native_stack",
            reason=f"active_bounded_native_stack_budget_exceeded:{total_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "line_base_preconditioner": base.to_dict(),
                "schwarz_preconditioner": None if schwarz is None else schwarz.to_dict(),
                "schwarz_reason": str(schwarz_reason),
                "coarse_size": int(coarse_size),
                "basis_nbytes_actual": int(basis_nbytes),
                "az_basis_nbytes_actual": int(az_basis_nbytes),
                "coarse_nbytes_actual": int(coarse_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
                "schwarz_factor_nbytes_actual": int(schwarz_nbytes),
                "factor_nbytes_actual": int(total_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                **adaptive_metadata,
                **window_metadata,
                **config,
            },
        )

    solver_kind = "lu"
    try:
        lu, piv = lu_factor(coarse_reg, check_finite=False)

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs, check_finite=False), dtype=np.float64).reshape((-1,))

        coarse_condition = float(np.linalg.cond(coarse_reg)) if coarse_size <= 512 else None
    except Exception:  # noqa: BLE001
        solver_kind = "pinv"
        pinv = np.linalg.pinv(coarse_reg, rcond=max(float(abs(regularization)), 1.0e-14))

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(pinv @ rhs, dtype=np.float64).reshape((-1,))

        coarse_condition = None

    def _apply_line_base(arr: np.ndarray) -> np.ndarray:
        return np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))

    def _apply_schwarz(residual: np.ndarray) -> np.ndarray:
        if schwarz is None or schwarz.operator is None:
            return np.zeros_like(residual)
        return np.asarray(schwarz.operator.matvec(residual), dtype=np.float64).reshape((-1,))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y = _apply_line_base(arr)
        if schwarz is not None:
            residual = arr - np.asarray(matrix_csr @ y, dtype=np.float64).reshape((-1,))
            y = y + _apply_schwarz(residual)
        residual = arr - np.asarray(matrix_csr @ y, dtype=np.float64).reshape((-1,))
        coarse_rhs = (
            np.asarray(basis.T @ residual, dtype=np.float64).reshape((-1,))
            if coarse_solver_mode == "galerkin"
            else np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        )
        coeff = solve_coarse(coarse_rhs)
        return y + np.asarray(basis @ coeff, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_bounded_native_stack",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_bounded_native_line_schwarz_coupled_coarse",
            "base_kind": str(base.kind),
            "line_base_preconditioner": base.to_dict(),
            "schwarz_requested": bool(schwarz_requested),
            "schwarz_selected": bool(schwarz is not None),
            "schwarz_reason": str(schwarz_reason),
            "schwarz_preconditioner": None if schwarz is None else schwarz.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_size),
            "max_coarse_size": int(max_coarse_size),
            "projected_basis_nnz": int(basis.nnz),
            "az_basis_nnz": int(az_basis.nnz),
            "coarse_solver": str(solver_kind),
            "coarse_equation": str(coarse_solver_mode),
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": coarse_condition,
            "adaptive_residual_operator": str(adaptive_residual_operator),
            "basis_nbytes_actual": int(basis_nbytes),
            "az_basis_nbytes_actual": int(az_basis_nbytes),
            "coarse_nbytes_actual": int(coarse_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "schwarz_factor_nbytes_actual": int(schwarz_nbytes),
            "factor_nbytes_actual": int(total_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "note": "no_global_serial_sparse_factor",
            "requires_preflight": True,
            **adaptive_metadata,
            **window_metadata,
            **config,
        },
    )


def build_active_fortran_v3_reduced_native_stack_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    base_preconditioner_factory: Callable[..., RHS1StructuredFullCSRPreconditioner] | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build the production-named bounded native stack for direct-tail solves.

    ``active_fortran_v3_reduced_lu`` is the robust full-grid incumbent, but it
    stores a large monolithic sparse factor.  This production candidate keeps
    the same active direct-tail operator and residual gate while replacing the
    global serial factor by bounded native line factors, optional local
    Schwarz, and a compact coupled coarse residual equation.
    """

    base = build_active_projected_bounded_native_stack_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind=str(requested_kind),
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
        base_preconditioner_factory=base_preconditioner_factory,
    )
    metadata = dict(base.metadata)
    metadata.update(
        {
            "requested_kind": str(requested_kind),
            "architecture": "fortran_v3_reduced_active_native_line_schwarz_coupled_coarse",
            "base_preconditioner_kind": str(base.kind),
            "base_architecture": str(base.metadata.get("architecture", "")),
            "no_global_serial_sparse_factor": True,
            "exact_serial_lu_factor": False,
            "production_candidate": True,
            "requires_preflight": True,
            "note": "bounded_direct_tail_native_stack",
        }
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=base.operator,
        selected=bool(base.selected),
        kind="active_fortran_v3_reduced_native_stack",
        reason=str(base.reason),
        setup_s=float(base.setup_s),
        metadata=metadata,
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


def build_active_projected_diagonal_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a cheap active kinetic-diagonal plus global-tail Schur split."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    try:
        active_layout = RHS1ActiveBlockLayout.from_layout(layout, active_indices)
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason="invalid_active_layout",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    if int(active_layout.active_size) != int(matrix_csr.shape[0]):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_layout": active_layout.to_dict(),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    if int(active_layout.phi1_count) != 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason="active_phi1_tail_split_unsupported",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"active_layout": active_layout.to_dict()},
        )
    if int(active_layout.extra_count) <= 0:
        return build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="active_no_global_tail",
        )
    if not bool(active_layout.has_contiguous_extra_tail):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason="active_tail_not_contiguous",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"active_layout": active_layout.to_dict()},
        )

    n_f = int(active_layout.kinetic_count)
    tail_size = int(active_layout.extra_count)
    if n_f + tail_size != int(matrix_csr.shape[0]):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason="active_kinetic_tail_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "kinetic_size": int(n_f),
                "tail_size": int(tail_size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
                "active_layout": active_layout.to_dict(),
            },
        )

    schur_nbytes_estimate = int(tail_size) * int(tail_size) * np.dtype(np.float64).itemsize
    if schur_nbytes_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_diagonal_schur",
            reason=f"active_diagonal_schur_budget_exceeded:{schur_nbytes_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "kinetic_size": int(n_f),
                "tail_size": int(tail_size),
                "schur_nbytes_estimate": int(schur_nbytes_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "active_layout": active_layout.to_dict(),
            },
        )

    diag = matrix_csr.diagonal()
    inv_f, diag_meta = safe_inverse_diagonal(diag[:n_f], regularization=regularization)
    u = matrix_csr[:n_f, n_f:].tocsr()
    v = matrix_csr[n_f:, :n_f].tocsr()
    w = matrix_csr[n_f:, n_f:].tocsr()
    scaled_u = u.multiply(inv_f[:, None])
    schur = np.asarray((w - v @ scaled_u).toarray(), dtype=np.float64)
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    try:
        lu, piv = lu_factor(schur)
        schur_solver_kind = "lu"

        def solve_tail(rhs_tail: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs_tail), dtype=np.float64).reshape((-1,))

    except Exception:  # noqa: BLE001
        pinv = np.linalg.pinv(schur, rcond=max(float(abs(regularization)), 1.0e-14))
        schur_solver_kind = "pinv"

        def solve_tail(rhs_tail: np.ndarray) -> np.ndarray:
            return np.asarray(pinv @ rhs_tail, dtype=np.float64).reshape((-1,))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = inv_f * arr[:n_f]
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = solve_tail(rhs_tail)
        y_f = y_f - inv_f * np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,))
        return np.concatenate((y_f, y_tail))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_diagonal_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            "matrix_nnz": int(matrix_csr.nnz),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "factor_nbytes_actual": int(schur.nbytes),
            "factor_nbytes_estimate": int(schur_nbytes_estimate),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            "schur_solver_kind": str(schur_solver_kind),
            "active_layout": active_layout.to_dict(),
            **diag_meta,
        },
    )


def build_active_projected_xell_kinetic_line_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_kinetic_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build dense active ``(x, ell)`` line inverses for a projected kinetic block."""

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_full = np.asarray(active_kinetic_indices, dtype=np.int64).reshape((-1,))
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix_csr.shape)},
        )
    if active_full.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason="active_kinetic_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_kinetic_index_size": int(active_full.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    if active_full.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason="empty_active_kinetic_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )
    if np.any(active_full < 0) or np.any(active_full >= int(layout.f_size)):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason="active_kinetic_indices_outside_f_block",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "layout_f_size": int(layout.f_size),
                "active_min": int(np.min(active_full)),
                "active_max": int(np.max(active_full)),
            },
        )

    decoded = layout.decode_kinetic_indices(active_full)
    line_ids = (
        (decoded.species.astype(np.int64, copy=False) * int(layout.n_theta) + decoded.theta)
        * int(layout.n_zeta)
        + decoded.zeta
    )
    unique_lines = np.unique(line_ids)
    positions_by_line = [np.flatnonzero(line_ids == line_id).astype(np.int64, copy=False) for line_id in unique_lines]
    block_sizes = np.asarray([int(pos.size) for pos in positions_by_line], dtype=np.int64)
    max_block_size = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_MAX_BLOCK_SIZE", 512))
    max_block_size = max(1, int(max_block_size))
    block_size_max = int(np.max(block_sizes)) if block_sizes.size else 0
    if block_size_max > int(max_block_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason=f"active_native_xell_block_size_exceeded:{block_size_max}>{int(max_block_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_size_max": int(block_size_max),
                "max_block_size": int(max_block_size),
                "n_blocks": int(len(positions_by_line)),
            },
        )

    inverse_nbytes_estimate = int(sum(int(size) * int(size) * np.dtype(np.float64).itemsize for size in block_sizes))
    index_nbytes_estimate = int(sum(int(size) * np.dtype(np.int64).itemsize for size in block_sizes))
    factor_estimate = int(inverse_nbytes_estimate + index_nbytes_estimate)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason=f"active_native_xell_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "inverse_nbytes_estimate": int(inverse_nbytes_estimate),
                "index_nbytes_estimate": int(index_nbytes_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(len(positions_by_line)),
                "block_size_min": int(np.min(block_sizes)) if block_sizes.size else 0,
                "block_size_max": int(block_size_max),
            },
        )

    inverse_blocks: list[np.ndarray] = []
    regularized_count = 0
    singular_count = 0
    nonfinite_count = 0
    condition_nonfinite_count = 0
    max_condition_estimate: float | None = 0.0 if block_size_max <= 64 else None
    max_block_scale = 0.0
    for positions in positions_by_line:
        block = np.asarray(matrix_csr[positions[:, None], positions].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(block, ord=np.inf)) if block.size else 0.0, 1.0)
        max_block_scale = max(float(max_block_scale), float(block_scale))
        regularization_abs = float(abs(regularization)) * float(block_scale)
        if regularization_abs > 0.0:
            block = block + regularization_abs * np.eye(int(block.shape[0]), dtype=np.float64)
            regularized_count += 1
        if max_condition_estimate is not None:
            condition_estimate = float(np.linalg.cond(block))
            if np.isfinite(condition_estimate):
                max_condition_estimate = max(float(max_condition_estimate), condition_estimate)
            else:
                condition_nonfinite_count += 1
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse = np.linalg.pinv(block, rcond=max(float(abs(regularization)), 1.0e-14))
        if not np.all(np.isfinite(inverse)):
            nonfinite_count += 1
        inverse_blocks.append(np.asarray(inverse, dtype=np.float64))
    if int(nonfinite_count) > 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason="active_native_xell_inverse_nonfinite",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_inverse_nonfinite_count": int(nonfinite_count),
                "n_blocks": int(len(positions_by_line)),
            },
        )

    actual_inverse_nbytes = int(sum(int(inv.nbytes) for inv in inverse_blocks))
    actual_index_nbytes = int(sum(int(pos.nbytes) for pos in positions_by_line))
    actual_nbytes = int(actual_inverse_nbytes + actual_index_nbytes)
    if actual_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_native_xell",
            reason=f"active_native_xell_budget_exceeded_actual:{actual_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "factor_nbytes_actual": int(actual_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(len(positions_by_line)),
            },
        )

    def apply(rhs: Any) -> np.ndarray:
        rhs_vec = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if rhs_vec.shape != (int(matrix_csr.shape[0]),):
            raise ValueError(f"rhs must have shape {(int(matrix_csr.shape[0]),)}, got {rhs_vec.shape}")
        out = np.empty_like(rhs_vec)
        for positions, inverse in zip(positions_by_line, inverse_blocks, strict=True):
            out[positions] = np.asarray(inverse @ rhs_vec[positions], dtype=np.float64)
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_native_xell",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "backend": "python_native_active_x_ell_line_inverse",
            "active_size": int(matrix_csr.shape[0]),
            "n_blocks": int(len(positions_by_line)),
            "block_size_min": int(np.min(block_sizes)) if block_sizes.size else 0,
            "block_size_max": int(block_size_max),
            "block_size_unique": tuple(int(v) for v in np.unique(block_sizes)),
            "fixed_axes": ("species", "theta", "zeta"),
            "line_axes": ("active_x", "active_ell"),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_actual": int(actual_nbytes),
            "inverse_nbytes_actual": int(actual_inverse_nbytes),
            "index_nbytes_actual": int(actual_index_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "max_block_size": int(max_block_size),
            "regularization": float(regularization),
            "block_inverse_regularized_count": int(regularized_count),
            "block_inverse_singular_count": int(singular_count),
            "block_inverse_nonfinite_count": int(nonfinite_count),
            "block_inverse_condition_estimate_max": max_condition_estimate,
            "block_inverse_condition_nonfinite_count": int(condition_nonfinite_count),
            "block_inverse_scale_max": float(max_block_scale),
            "layout": layout.to_dict(),
        },
    )


def build_active_projected_angular_line_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_kinetic_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build active angular-line inverses grouped by ``(species, x, ell)``."""

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_full = np.asarray(active_kinetic_indices, dtype=np.int64).reshape((-1,))
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix_csr.shape)},
        )
    if active_full.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_full.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    if active_full.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason="empty_active_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )

    kinetic_mask = active_full < int(layout.f_size)
    kinetic_positions = np.flatnonzero(kinetic_mask).astype(np.int64, copy=False)
    if kinetic_positions.size == 0:
        return build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="active_angular_line_no_kinetic_rows",
        )
    active_kinetic_full = active_full[kinetic_positions]
    if np.any(active_kinetic_full < 0) or np.any(active_kinetic_full >= int(layout.f_size)):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason="active_kinetic_indices_outside_f_block",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "layout_f_size": int(layout.f_size),
                "active_min": int(np.min(active_kinetic_full)),
                "active_max": int(np.max(active_kinetic_full)),
            },
        )

    decoded = layout.decode_kinetic_indices(active_kinetic_full)
    line_ids = (
        (decoded.species.astype(np.int64, copy=False) * int(layout.n_x) + decoded.x)
        * int(layout.n_xi)
        + decoded.ell
    )
    unique_lines = np.unique(line_ids)
    positions_by_line = [
        kinetic_positions[np.flatnonzero(line_ids == line_id)].astype(np.int64, copy=False)
        for line_id in unique_lines
    ]
    block_sizes = np.asarray([int(pos.size) for pos in positions_by_line], dtype=np.int64)
    max_block_size = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ANGULAR_LINE_MAX_BLOCK_SIZE", 4096))
    max_block_size = max(1, int(max_block_size))
    block_size_max = int(np.max(block_sizes)) if block_sizes.size else 0
    if block_size_max > int(max_block_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason=f"active_angular_line_block_size_exceeded:{block_size_max}>{int(max_block_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_size_max": int(block_size_max),
                "max_block_size": int(max_block_size),
                "n_blocks": int(len(positions_by_line)),
            },
        )

    inverse_nbytes_estimate = int(sum(int(size) * int(size) * np.dtype(np.float64).itemsize for size in block_sizes))
    index_nbytes_estimate = int(sum(int(size) * np.dtype(np.int64).itemsize for size in block_sizes))
    tail_positions = np.flatnonzero(~kinetic_mask).astype(np.int64, copy=False)
    tail_nbytes_estimate = int(tail_positions.size * np.dtype(np.float64).itemsize)
    factor_estimate = int(inverse_nbytes_estimate + index_nbytes_estimate + tail_nbytes_estimate)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason=f"active_angular_line_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "inverse_nbytes_estimate": int(inverse_nbytes_estimate),
                "index_nbytes_estimate": int(index_nbytes_estimate),
                "tail_nbytes_estimate": int(tail_nbytes_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(len(positions_by_line)),
                "block_size_min": int(np.min(block_sizes)) if block_sizes.size else 0,
                "block_size_max": int(block_size_max),
            },
        )

    inverse_blocks: list[np.ndarray] = []
    regularized_count = 0
    singular_count = 0
    nonfinite_count = 0
    condition_nonfinite_count = 0
    max_condition_estimate: float | None = 0.0 if block_size_max <= 64 else None
    max_block_scale = 0.0
    for positions in positions_by_line:
        block = np.asarray(matrix_csr[positions[:, None], positions].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(block, ord=np.inf)) if block.size else 0.0, 1.0)
        max_block_scale = max(float(max_block_scale), float(block_scale))
        regularization_abs = float(abs(regularization)) * float(block_scale)
        if regularization_abs > 0.0:
            block = block + regularization_abs * np.eye(int(block.shape[0]), dtype=np.float64)
            regularized_count += 1
        if max_condition_estimate is not None:
            condition_estimate = float(np.linalg.cond(block))
            if np.isfinite(condition_estimate):
                max_condition_estimate = max(float(max_condition_estimate), condition_estimate)
            else:
                condition_nonfinite_count += 1
        try:
            inverse = np.linalg.inv(block)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse = np.linalg.pinv(block, rcond=max(float(abs(regularization)), 1.0e-14))
        if not np.all(np.isfinite(inverse)):
            nonfinite_count += 1
        inverse_blocks.append(np.asarray(inverse, dtype=np.float64))
    if int(nonfinite_count) > 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason="active_angular_line_inverse_nonfinite",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"block_inverse_nonfinite_count": int(nonfinite_count), "n_blocks": int(len(positions_by_line))},
        )

    inv_tail = np.empty((0,), dtype=np.float64)
    tail_metadata: dict[str, object] = {"tail_size": int(tail_positions.size)}
    if tail_positions.size:
        inv_tail, tail_metadata = safe_inverse_diagonal(
            matrix_csr.diagonal()[tail_positions],
            regularization=regularization,
        )
    actual_inverse_nbytes = int(sum(int(inv.nbytes) for inv in inverse_blocks))
    actual_index_nbytes = int(sum(int(pos.nbytes) for pos in positions_by_line))
    actual_nbytes = int(actual_inverse_nbytes + actual_index_nbytes + inv_tail.nbytes)
    if actual_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_angular_line",
            reason=f"active_angular_line_budget_exceeded_actual:{actual_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "factor_nbytes_estimate": int(factor_estimate),
                "factor_nbytes_actual": int(actual_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "n_blocks": int(len(positions_by_line)),
            },
        )

    def apply(rhs: Any) -> np.ndarray:
        rhs_vec = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if rhs_vec.shape != (int(matrix_csr.shape[0]),):
            raise ValueError(f"rhs must have shape {(int(matrix_csr.shape[0]),)}, got {rhs_vec.shape}")
        out = np.zeros_like(rhs_vec)
        for positions, inverse in zip(positions_by_line, inverse_blocks, strict=True):
            out[positions] = np.asarray(inverse @ rhs_vec[positions], dtype=np.float64)
        if tail_positions.size:
            out[tail_positions] = inv_tail * rhs_vec[tail_positions]
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_angular_line",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "backend": "python_native_active_angular_line_inverse",
            "active_size": int(matrix_csr.shape[0]),
            "kinetic_size": int(kinetic_positions.size),
            "tail_size": int(tail_positions.size),
            "n_blocks": int(len(positions_by_line)),
            "block_size_min": int(np.min(block_sizes)) if block_sizes.size else 0,
            "block_size_max": int(block_size_max),
            "block_size_unique": tuple(int(v) for v in np.unique(block_sizes)),
            "fixed_axes": ("species", "x", "ell"),
            "line_axes": ("active_theta", "active_zeta"),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_actual": int(actual_nbytes),
            "inverse_nbytes_actual": int(actual_inverse_nbytes),
            "index_nbytes_actual": int(actual_index_nbytes),
            "tail_inverse_nbytes_actual": int(inv_tail.nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "max_block_size": int(max_block_size),
            "regularization": float(regularization),
            "block_inverse_regularized_count": int(regularized_count),
            "block_inverse_singular_count": int(singular_count),
            "block_inverse_nonfinite_count": int(nonfinite_count),
            "block_inverse_condition_estimate_max": max_condition_estimate,
            "block_inverse_condition_nonfinite_count": int(condition_nonfinite_count),
            "block_inverse_scale_max": float(max_block_scale),
            "layout": layout.to_dict(),
            **{f"tail_{key}": value for key, value in tail_metadata.items()},
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
