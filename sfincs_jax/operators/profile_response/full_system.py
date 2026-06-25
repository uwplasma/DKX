"""Structured host-CSR assembly for RHSMode=1 full-system operators.

The f-block stencil assembly lives in :mod:`sfincs_jax.operators.profile_response.kinetic`.
This module adds the small global couplings around that block: Phi1/QN rows
when they are linear, and the source/constraint rows used by SFINCS v3.  The
result is an exact sparse matrix for supported RHSMode=1 cases without dense
column probing.

This file is intentionally large during the v3-driver consolidation because it
keeps the term-level full-system CSR formulas, admission gates, and current
preconditioner aliases in one behavior-preserving owner. The safe future split
is by responsibility: sparse assembly, preconditioner construction, cache
metadata, and solve wrappers.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_response.layout import RHS1ActiveFieldSplitOrdering, RHS1BlockLayout
from sfincs_jax.operators.profile_response.compressed_layout import infer_rhs1_compressed_pitch_layout_from_active_indices
from sfincs_jax.operators.profile_response.kinetic import (
    RHS1StructuredFBlockCSRSelection,
    clear_structured_rhs1_fblock_csr_cache,
    select_structured_rhs1_fblock_csr_operator,
)
from sfincs_jax.problems.profile_response.active_preconditioner_policy import resolve_active_projected_preconditioner_auto_policy
from sfincs_jax.solvers.preconditioners.symbolic_sparse.policy import (
    active_fortran_v3_reduced_permc_candidates,
)
from sfincs_jax.solvers.preconditioners.symbolic_sparse.policy import resolve_active_symbolic_frontal_policy
from sfincs_jax.solvers.preconditioners.symbolic_sparse.policy import (
    resolve_active_symbolic_block_schur_policy,
    resolve_active_symbolic_superblock_policy,
)
from sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_policy import (
    resolve_active_native_field_split_sparse_coarse_policy,
    resolve_active_sparse_coarse_residual_policy,
)
from sfincs_jax.solvers.preconditioners.schur.rhs1_coarse_basis import (
    append_adaptive_residual_basis_csc as _append_adaptive_residual_basis_csc,
    build_active_native_xell_coarse_window_basis_csc as _build_active_native_xell_coarse_window_basis_csc,
    build_coarse_residual_basis_csc as _build_coarse_residual_basis_csc,
    coarse_residual_config as _coarse_residual_config,
    estimate_coarse_residual_nbytes as _estimate_coarse_residual_nbytes,
    estimate_xblock_tz_low_l_factor_nbytes as _estimate_xblock_tz_low_l_factor_nbytes,
    xblock_tz_low_l_config as _xblock_tz_low_l_config,
)
from sfincs_jax.solvers.preconditioners.schur.rhs1_full_csr import (
    RHS1StructuredFullCSRPreconditioner,
    build_block_schur_preconditioner as _build_block_schur_preconditioner,
    build_diagonal_schur_preconditioner as _build_diagonal_schur_preconditioner,
    build_jacobi_preconditioner as _build_jacobi_preconditioner,
    build_x_xi_block_schur_preconditioner as _build_x_xi_block_schur_preconditioner,
    build_xi_block_schur_preconditioner as _build_xi_block_schur_preconditioner,
    estimate_x_xi_block_inverse_nbytes as _estimate_x_xi_block_inverse_nbytes,
    estimate_xi_block_inverse_nbytes as _estimate_xi_block_inverse_nbytes,
    estimate_zeta_block_inverse_nbytes as _estimate_zeta_block_inverse_nbytes,
)
from sfincs_jax.solvers.preconditioners.symbolic_sparse import rhs1_fortran_reduced as _rhs1_fortran_reduced_pc
from sfincs_jax.solvers.preconditioners.symbolic_sparse.active_factors import (
    build_active_filtered_sparse_factor_preconditioner as _build_active_projected_filtered_sparse_factor_preconditioner,
    build_active_global_sparse_factor_preconditioner as _build_active_global_sparse_factor_preconditioner,
    build_active_scaled_sparse_factor_preconditioner as _build_active_scaled_sparse_factor_preconditioner,
)
from sfincs_jax.solvers.preconditioners.xblock.active_projected import (
    active_positions_for_full_indices as _active_positions_for_full_indices,
    build_active_fortran_v3_reduced_native_stack_preconditioner as _build_active_fortran_v3_reduced_native_stack_preconditioner,
    build_active_projected_bounded_native_stack_preconditioner as _build_active_projected_bounded_native_stack_preconditioner,
    build_active_projected_global_field_split_schur_preconditioner as _build_active_projected_global_field_split_schur_preconditioner,
    build_active_projected_multiline_field_split_base_preconditioner as _build_active_projected_multiline_field_split_base_preconditioner,
    build_active_projected_angular_line_preconditioner as _build_active_projected_angular_line_preconditioner,
    build_active_projected_diagonal_schur_preconditioner as _build_active_projected_diagonal_schur_preconditioner,
    build_active_projected_native_indexed_schwarz_preconditioner as _build_active_projected_native_indexed_schwarz_preconditioner,
    build_active_projected_overlap_schwarz_preconditioner as _build_active_projected_overlap_schwarz_preconditioner,
    build_active_projected_xblock_preconditioner as _build_active_projected_xblock_preconditioner,
)
from sfincs_jax.solvers.preconditioners.xblock.low_l_schur import (
    build_native_xell_kinetic_preconditioner as _build_native_xell_kinetic_preconditioner,
    build_native_xell_tail_schur_preconditioner as _build_native_xell_tail_schur_preconditioner,
    build_xblock_tz_low_l_coarse_residual_preconditioner as _build_xblock_tz_low_l_coarse_residual_preconditioner,
    build_xblock_tz_low_l_schur_preconditioner as _build_xblock_tz_low_l_schur_preconditioner,
    xblock_tz_low_l_indices as _xblock_tz_low_l_indices,
)
from sfincs_jax.v3_sparse_pattern import estimate_v3_full_system_conservative_sparsity_summary

_STRUCTURED_FULL_CSR_OBJECT_CACHE: dict[tuple[object, ...], tuple[Any, dict[str, object]]] = {}
_active_fortran_v3_reduced_permc_candidates = active_fortran_v3_reduced_permc_candidates
_active_fortran_v3_reduced_preconditioner_matrix = (
    _rhs1_fortran_reduced_pc.active_fortran_v3_reduced_preconditioner_matrix
)
_active_fortran_v3_support_mode_dropped_no_entries = (
    _rhs1_fortran_reduced_pc.active_fortran_v3_support_mode_dropped_no_entries
)
_apply_active_fortran_v3_support_mode_token = _rhs1_fortran_reduced_pc.apply_active_fortran_v3_support_mode_token
_build_active_fortran_v3_reduced_sparse_factor_preconditioner = (
    _rhs1_fortran_reduced_pc.build_active_fortran_v3_reduced_sparse_factor_preconditioner
)
_estimate_spilu_factor_nbytes = _rhs1_fortran_reduced_pc.estimate_spilu_factor_nbytes
_parse_active_fortran_v3_support_mode_candidates = (
    _rhs1_fortran_reduced_pc.parse_active_fortran_v3_support_mode_candidates
)
select_active_fortran_v3_reduced_support_mode_preconditioner = (
    _rhs1_fortran_reduced_pc.select_active_fortran_v3_reduced_support_mode_preconditioner
)
_sparse_equilibration_scale = _rhs1_fortran_reduced_pc.sparse_equilibration_scale
_sparse_lu_factor_nbytes = _rhs1_fortran_reduced_pc.sparse_lu_factor_nbytes


def _estimate_csr_nbytes_from_nnz(*, shape: tuple[int, int], nnz: int, dtype: Any = np.float64) -> int:
    dtype_np = np.dtype(dtype)
    nnz_use = max(0, int(nnz))
    n_rows = max(0, int(shape[0]))
    data_nbytes = nnz_use * int(dtype_np.itemsize)
    index_nbytes = nnz_use * np.dtype(np.int32).itemsize
    indptr_nbytes = (n_rows + 1) * np.dtype(np.int32).itemsize
    return int(data_nbytes + index_nbytes + indptr_nbytes)


@dataclass(frozen=True)
class RHS1StructuredFullCSRSelection:
    """Fail-closed host CSR materialization for a supported RHSMode=1 system."""

    fblock_selection: RHS1StructuredFBlockCSRSelection
    matrix: Any | None
    selected: bool
    reason: str
    cache_hit: bool
    build_s: float
    metadata: dict[str, object]

    def matvec(self, x: Any) -> np.ndarray:
        """Apply the materialized full-system CSR matrix."""

        if self.matrix is None:
            raise RuntimeError(f"structured full CSR operator was not selected: {self.reason}")
        return np.asarray(self.matrix @ np.asarray(x, dtype=np.float64).reshape((-1,)))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly sparse-storage and coverage metadata."""

        return {
            "selected": bool(self.selected),
            "reason": str(self.reason),
            "cache_hit": bool(self.cache_hit),
            "build_s": float(self.build_s),
            "metadata": dict(self.metadata),
            "fblock_selection": self.fblock_selection.to_dict(),
        }


@dataclass(frozen=True)
class RHS1StructuredFullCSRSolveResult:
    """Result from the explicit host-CSR RHSMode=1 solve lane."""

    selection: RHS1StructuredFullCSRSelection
    x: np.ndarray
    residual_norm: float
    residual_history: tuple[float, ...]
    info: int
    converged: bool
    solve_s: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly solver metadata without the solution vector."""

        return {
            "selected": bool(self.selection.selected),
            "residual_norm": float(self.residual_norm),
            "residual_history": tuple(float(v) for v in self.residual_history),
            "info": int(self.info),
            "converged": bool(self.converged),
            "solve_s": float(self.solve_s),
            "metadata": dict(self.metadata),
            "selection": self.selection.to_dict(),
        }


def clear_structured_rhs1_full_csr_cache(*, clear_fblock_cache: bool = False) -> None:
    """Clear full-operator CSR caches used by tests and bounded benchmarks."""

    _STRUCTURED_FULL_CSR_OBJECT_CACHE.clear()
    if bool(clear_fblock_cache):
        clear_structured_rhs1_fblock_csr_cache()


def select_structured_rhs1_full_csr_operator(
    op: Any,
    *,
    include_identity_shift: bool = True,
    phi1_hat_base: Any | None = None,
    drop_tol: float = 0.0,
    require_complete: bool = True,
    max_csr_nbytes: int | None = None,
    use_cache: bool = True,
    include_jacobian_terms: bool = True,
) -> RHS1StructuredFullCSRSelection:
    """Build or reuse an exact host CSR matrix for supported full systems.

    Supported cases are intentionally narrow and fail closed:

    - ``constraintScheme`` 1 and 2;
    - no nonlinear Phi1-in-kinetic coupling;
    - linear Phi1/QN/lambda rows are allowed when ``includePhi1`` is true.

    The assembled matrix is a host-side, non-autodiff artifact intended for
    CLI/runtime sparse-solve and preconditioner development. JAX matrix-free
    and block-COO paths remain the differentiable route.
    """

    t0 = time.perf_counter()
    layout = RHS1BlockLayout.from_operator(op)
    reason = _unsupported_reason(op=op, include_jacobian_terms=include_jacobian_terms)
    if reason is not None:
        f_sel = select_structured_rhs1_fblock_csr_operator(
            op.fblock,
            include_identity_shift=include_identity_shift,
            phi1_hat_base=phi1_hat_base,
            drop_tol=drop_tol,
            require_complete=False,
            max_csr_nbytes=None,
            use_cache=use_cache,
        )
        return RHS1StructuredFullCSRSelection(
            fblock_selection=f_sel,
            matrix=None,
            selected=False,
            reason=reason,
            cache_hit=False,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata={"selected": False, "reason": reason, "layout": layout.to_dict()},
        )

    object_cache_key = _structured_full_csr_object_cache_key(
        op=op,
        include_identity_shift=include_identity_shift,
        phi1_hat_base=phi1_hat_base,
        drop_tol=drop_tol,
        require_complete=require_complete,
        include_jacobian_terms=include_jacobian_terms,
    )
    if bool(use_cache) and object_cache_key in _STRUCTURED_FULL_CSR_OBJECT_CACHE:
        matrix, cached_metadata = _STRUCTURED_FULL_CSR_OBJECT_CACHE[object_cache_key]
        actual_nbytes = int(cached_metadata.get("csr_nbytes_actual", 0) or 0)
        if max_csr_nbytes is not None and actual_nbytes > int(max_csr_nbytes):
            reason = f"csr_budget_exceeded:{actual_nbytes}>{int(max_csr_nbytes)}"
            return RHS1StructuredFullCSRSelection(
                fblock_selection=cached_metadata["fblock_selection"],
                matrix=None,
                selected=False,
                reason=reason,
                cache_hit=True,
                build_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "selected": False,
                    "reason": reason,
                    "csr_nbytes_actual": actual_nbytes,
                    "max_csr_nbytes": int(max_csr_nbytes),
                    "object_cache_hit": True,
                },
            )
        metadata = dict(cached_metadata)
        f_sel = metadata.pop("fblock_selection")
        metadata["cache_hit"] = True
        metadata["object_cache_hit"] = True
        return RHS1StructuredFullCSRSelection(
            fblock_selection=f_sel,
            matrix=matrix,
            selected=True,
            reason="complete",
            cache_hit=True,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata=metadata,
        )

    if max_csr_nbytes is not None:
        summary = estimate_v3_full_system_conservative_sparsity_summary(op)
        estimated_nbytes = _estimate_csr_nbytes_from_nnz(
            shape=summary.shape,
            nnz=int(summary.nnz),
            dtype=np.float64,
        )
        if estimated_nbytes > int(max_csr_nbytes):
            reason = f"csr_budget_preflight_exceeded:{int(estimated_nbytes)}>{int(max_csr_nbytes)}"
            fblock_stub = RHS1StructuredFBlockCSRSelection(
                selection=SimpleNamespace(
                    to_dict=lambda: {
                        "selected": False,
                        "reason": "not_built_due_to_full_csr_preflight",
                    },
                ),
                matrix=None,
                selected=False,
                reason="not_built_due_to_full_csr_preflight",
                cache_hit=False,
                build_s=0.0,
                metadata={
                    "selected": False,
                    "reason": "not_built_due_to_full_csr_preflight",
                },
            )
            return RHS1StructuredFullCSRSelection(
                fblock_selection=fblock_stub,
                matrix=None,
                selected=False,
                reason=reason,
                cache_hit=False,
                build_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "selected": False,
                    "reason": reason,
                    "csr_nbytes_estimate": int(estimated_nbytes),
                    "max_csr_nbytes": int(max_csr_nbytes),
                    "sparsity_summary": summary.to_dict(),
                    "layout": layout.to_dict(),
                },
            )

    f_sel = select_structured_rhs1_fblock_csr_operator(
        op.fblock,
        include_identity_shift=include_identity_shift,
        phi1_hat_base=op.phi1_hat_base if phi1_hat_base is None else phi1_hat_base,
        drop_tol=drop_tol,
        require_complete=require_complete,
        max_csr_nbytes=None,
        use_cache=use_cache,
    )
    if not bool(f_sel.selected) or f_sel.matrix is None:
        return RHS1StructuredFullCSRSelection(
            fblock_selection=f_sel,
            matrix=None,
            selected=False,
            reason=str(f_sel.reason),
            cache_hit=False,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata={"selected": False, "reason": str(f_sel.reason), "layout": layout.to_dict()},
        )

    tail = _assemble_full_tail_csr(op=op, layout=layout, include_jacobian_terms=include_jacobian_terms)
    zero_tail = sp.csr_matrix((int(layout.total_size) - int(layout.f_size), int(layout.total_size) - int(layout.f_size)))
    base = sp.block_diag((f_sel.matrix, zero_tail), format="csr")
    matrix = (base + tail).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    actual_nbytes = _scipy_csr_nbytes(matrix)
    if max_csr_nbytes is not None and actual_nbytes > int(max_csr_nbytes):
        reason = f"csr_budget_exceeded:{actual_nbytes}>{int(max_csr_nbytes)}"
        return RHS1StructuredFullCSRSelection(
            fblock_selection=f_sel,
            matrix=None,
            selected=False,
            reason=reason,
            cache_hit=False,
            build_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "selected": False,
                "reason": reason,
                "csr_nbytes_actual": int(actual_nbytes),
                "max_csr_nbytes": int(max_csr_nbytes),
                "layout": layout.to_dict(),
            },
        )

    metadata: dict[str, object] = {
        "selected": True,
        "reason": "complete",
        "storage_kind": "csr",
        "shape": tuple(int(v) for v in matrix.shape),
        "nnz": int(matrix.nnz),
        "csr_nbytes_actual": int(actual_nbytes),
        "tail_nnz": int(tail.nnz),
        "tail_nbytes_actual": int(_scipy_csr_nbytes(tail)),
        "fblock_csr_nbytes_actual": int(f_sel.metadata.get("csr_nbytes_actual", 0) or 0),
        "max_csr_nbytes": None if max_csr_nbytes is None else int(max_csr_nbytes),
        "drop_tol": float(drop_tol),
        "cache_hit": False,
        "object_cache_hit": False,
        "layout": layout.to_dict(),
        "fblock": f_sel.to_dict(),
    }
    if bool(use_cache):
        cache_metadata = dict(metadata)
        cache_metadata["fblock_selection"] = f_sel
        _STRUCTURED_FULL_CSR_OBJECT_CACHE[object_cache_key] = (matrix, cache_metadata)
    return RHS1StructuredFullCSRSelection(
        fblock_selection=f_sel,
        matrix=matrix,
        selected=True,
        reason="complete",
        cache_hit=False,
        build_s=max(0.0, time.perf_counter() - t0),
        metadata=metadata,
        )


def solve_structured_rhs1_full_csr(
    op: Any,
    rhs: Any,
    *,
    x0: Any | None = None,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    method: str = "gmres",
    preconditioner: str | None = "auto",
    preconditioner_max_schur_size: int = 2048,
    preconditioner_max_block_inverse_nbytes: int = 64 * 1024 * 1024,
    preconditioner_regularization: float = 1.0e-12,
    max_csr_nbytes: int | None = None,
    drop_tol: float = 0.0,
    active_indices: Any | None = None,
    direct_permc_spec: str = "COLAMD",
) -> RHS1StructuredFullCSRSolveResult:
    """Solve a supported RHSMode=1 system with explicit host CSR Krylov.

    This is a deliberately non-autodiff lane for CLI/runtime experiments. It
    never calls a JAX-traced matvec: SciPy operates directly on the assembled CSR
    matrix, and the returned residual is the true host residual ``||b - A x||``.
    """

    selection = select_structured_rhs1_full_csr_operator(
        op,
        drop_tol=float(drop_tol),
        max_csr_nbytes=max_csr_nbytes,
    )
    if not bool(selection.selected) or selection.matrix is None:
        raise RuntimeError(f"structured full CSR solve was not selected: {selection.reason}")

    from scipy.sparse.linalg import LinearOperator, gmres, lgmres, splu  # noqa: PLC0415

    matrix_full = selection.matrix.tocsr()
    layout = RHS1BlockLayout.from_operator(op)
    rhs_np = np.asarray(rhs, dtype=np.float64).reshape((-1,))
    if rhs_np.shape != (int(matrix_full.shape[0]),):
        raise ValueError(f"rhs must have shape {(int(matrix_full.shape[0]),)}, got {rhs_np.shape}")
    active_np: np.ndarray | None = None
    if active_indices is not None:
        active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
        if active_np.size == 0:
            raise ValueError("active_indices must not be empty")
        if np.any(active_np < 0) or np.any(active_np >= rhs_np.size):
            raise ValueError("active_indices contains entries outside the full system")
        active_np = np.unique(active_np)
    matrix = matrix_full if active_np is None else matrix_full[active_np[:, None], active_np].tocsr()
    rhs_solve = rhs_np if active_np is None else rhs_np[active_np]
    x0_np_full = None if x0 is None else np.asarray(x0, dtype=np.float64).reshape((-1,))
    if x0_np_full is not None and x0_np_full.shape != rhs_np.shape:
        raise ValueError(f"x0 must have shape {rhs_np.shape}, got {x0_np_full.shape}")
    x0_np = x0_np_full if active_np is None else (None if x0_np_full is None else x0_np_full[active_np])

    operator = LinearOperator(
        matrix.shape,
        matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64).reshape((-1,))),
        dtype=np.float64,
    )
    history: list[float] = []

    def callback(value: Any) -> None:
        residual = float(value) if np.isscalar(value) else float(np.linalg.norm(np.asarray(value)))
        history.append(residual)

    method_l = str(method).strip().lower()
    t0 = time.perf_counter()
    direct_metadata: dict[str, object] = {}
    if method_l in {"direct", "splu", "sparse_direct"}:
        factor_start = time.perf_counter()
        factor = splu(matrix.tocsc(), permc_spec=str(direct_permc_spec))
        factor_s = max(0.0, time.perf_counter() - factor_start)
        x_np = np.asarray(factor.solve(rhs_solve), dtype=np.float64).reshape((-1,))
        info = 0
        direct_metadata = {
            "factor_kind": "splu",
            "permc_spec": str(direct_permc_spec),
            "factor_s": float(factor_s),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_actual": int(_sparse_lu_factor_nbytes(factor)),
        }
        preconditioner_result = RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="none",
            reason="direct_solve",
            setup_s=0.0,
            metadata={},
        )
    else:
        if active_np is not None:
            preconditioner_result = build_active_projected_rhs1_full_csr_preconditioner(
                matrix=matrix,
                layout=layout,
                active_indices=active_np,
                kind=preconditioner,
                max_factor_nbytes=preconditioner_max_block_inverse_nbytes,
                regularization=preconditioner_regularization,
            )
        else:
            preconditioner_result = build_structured_rhs1_full_csr_preconditioner(
                matrix=matrix,
                layout=layout,
                kind=preconditioner,
                max_schur_size=preconditioner_max_schur_size,
                max_block_inverse_nbytes=preconditioner_max_block_inverse_nbytes,
                regularization=preconditioner_regularization,
            )
        scipy_preconditioner = preconditioner_result.operator if preconditioner_result.selected else None
    if method_l == "lgmres":
        x_np, info = lgmres(
            operator,
            rhs_solve,
            x0=x0_np,
            rtol=float(tol),
            atol=float(atol),
            maxiter=int(maxiter) if maxiter is not None else None,
            M=scipy_preconditioner,
            callback=callback,
        )
    elif method_l == "gmres":
        x_np, info = gmres(
            operator,
            rhs_solve,
            x0=x0_np,
            rtol=float(tol),
            atol=float(atol),
            restart=max(1, int(restart)),
            maxiter=int(maxiter) if maxiter is not None else None,
            M=scipy_preconditioner,
            callback=callback,
            callback_type="pr_norm",
        )
    elif method_l not in {"direct", "splu", "sparse_direct"}:
        raise ValueError("method must be 'gmres', 'lgmres', or 'direct'")
    solve_s = max(0.0, time.perf_counter() - t0)
    x_full = np.zeros_like(rhs_np)
    if active_np is None:
        x_full = np.asarray(x_np, dtype=np.float64).reshape((-1,))
    else:
        x_full[active_np] = np.asarray(x_np, dtype=np.float64).reshape((-1,))
    residual = rhs_np - np.asarray(matrix_full @ x_full, dtype=np.float64).reshape((-1,))
    residual_norm = float(np.linalg.norm(residual))
    target = max(float(atol), float(tol) * float(np.linalg.norm(rhs_np)))
    converged = bool(int(info) == 0 and np.isfinite(residual_norm) and residual_norm <= max(target, 1.0e-300))
    metadata = {
        "method": method_l,
        "restart": int(restart),
        "maxiter": None if maxiter is None else int(maxiter),
        "tol": float(tol),
        "atol": float(atol),
        "target": float(target),
        "rhs_norm": float(np.linalg.norm(rhs_np)),
        "matrix_nnz": int(matrix.nnz),
        "matrix_shape": tuple(int(v) for v in matrix.shape),
        "full_matrix_nnz": int(matrix_full.nnz),
        "full_matrix_shape": tuple(int(v) for v in matrix_full.shape),
        "active_dof": active_np is not None,
        "active_size": int(matrix.shape[0]),
        "full_size": int(matrix_full.shape[0]),
        "preconditioner": preconditioner_result.to_dict(),
        **direct_metadata,
    }
    return RHS1StructuredFullCSRSolveResult(
        selection=selection,
        x=np.asarray(x_full, dtype=np.float64),
        residual_norm=float(residual_norm),
        residual_history=tuple(float(v) for v in history),
        info=int(info),
        converged=bool(converged),
        solve_s=float(solve_s),
        metadata=metadata,
    )


def build_structured_rhs1_full_csr_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    kind: str | None = "auto",
    max_schur_size: int = 2048,
    max_block_inverse_nbytes: int = 64 * 1024 * 1024,
    regularization: float = 1.0e-12,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a memory-bounded host preconditioner for the full CSR operator.

    ``diagonal_schur`` is the current evidence-based automatic choice: it keeps
    the kinetic block cheap with scalar diagonal inversion and solves the small
    global Phi1/constraint tail Schur complement. ``block_schur``,
    ``xi_block_schur``, ``x_xi_block_schur``, and the coarse residual
    correction are explicit kinetic-block candidates protected by memory gates;
    they are not promoted unless benchmarks justify their setup cost.
    """

    t0 = time.perf_counter()
    matrix = matrix.tocsr()
    n_total = int(matrix.shape[0])
    if matrix.shape[0] != matrix.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=str(kind or "none"),
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix.shape)},
        )
    if n_total != int(layout.total_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=str(kind or "none"),
            reason="layout_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "matrix_shape": tuple(int(v) for v in matrix.shape),
                "layout_total_size": int(layout.total_size),
            },
        )

    kind_l = "none" if kind is None else str(kind).strip().lower().replace("-", "_")
    if kind_l in {"", "none", "false", "off", "disabled"}:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="none",
            reason="disabled",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )

    if kind_l == "auto":
        tail_size = int(layout.total_size) - int(layout.f_size)
        auto_min_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_AUTO_MIN_SIZE", 10_000)
        if int(layout.total_size) >= int(auto_min_size) and tail_size > 0 and tail_size <= int(max_schur_size):
            config = _xblock_tz_low_l_config(layout)
            estimate = _estimate_xblock_tz_low_l_factor_nbytes(layout=layout, config=config)
            if estimate <= int(max_block_inverse_nbytes):
                xblock = _build_xblock_tz_low_l_schur_preconditioner(
                    matrix=matrix,
                    layout=layout,
                    requested_kind=kind_l,
                    regularization=regularization,
                    max_factor_nbytes=int(max_block_inverse_nbytes),
                    config=config,
                    t0=t0,
                )
                if bool(xblock.selected):
                    return xblock
        if tail_size > 0 and tail_size <= int(max_schur_size):
            return _build_diagonal_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                t0=t0,
            )
        return _build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            t0=t0,
            reason="auto_fallback_jacobi",
        )

    if kind_l in {"native_xell", "native_x_ell", "jax_native_xell", "jax_xell"}:
        return _build_native_xell_kinetic_preconditioner(
            matrix=matrix,
            layout=layout,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=int(max_block_inverse_nbytes),
            t0=t0,
        )

    if kind_l in {
        "native_xell_tail_schur",
        "native_xell_schur",
        "native_x_ell_tail_schur",
        "jax_native_xell_tail_schur",
        "jax_xell_tail_schur",
    }:
        return _build_native_xell_tail_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            requested_kind=kind_l,
            regularization=regularization,
            max_schur_size=int(max_schur_size),
            max_factor_nbytes=int(max_block_inverse_nbytes),
            t0=t0,
        )

    if kind_l in {"xi_block_schur", "pitch_block_schur", "pitch_angle_block_schur"}:
        tail_size = int(layout.total_size) - int(layout.f_size)
        estimate = _estimate_xi_block_inverse_nbytes(layout)
        if tail_size > 0 and tail_size <= int(max_schur_size) and estimate <= int(max_block_inverse_nbytes):
            return _build_xi_block_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                t0=t0,
            )
        reason = (
            f"xi_block_schur_budget_exceeded:{estimate}>{int(max_block_inverse_nbytes)}"
            if estimate > int(max_block_inverse_nbytes)
            else f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=kind_l,
            reason=reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "tail_size": int(tail_size),
                "max_schur_size": int(max_schur_size),
                "block_inverse_nbytes_estimate": int(estimate),
                "max_block_inverse_nbytes": int(max_block_inverse_nbytes),
            },
        )

    if kind_l in {"x_xi_block_schur", "radial_velocity_block_schur", "velocity_radial_block_schur"}:
        tail_size = int(layout.total_size) - int(layout.f_size)
        estimate = _estimate_x_xi_block_inverse_nbytes(layout)
        if tail_size > 0 and tail_size <= int(max_schur_size) and estimate <= int(max_block_inverse_nbytes):
            return _build_x_xi_block_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                t0=t0,
            )
        reason = (
            f"x_xi_block_schur_budget_exceeded:{estimate}>{int(max_block_inverse_nbytes)}"
            if estimate > int(max_block_inverse_nbytes)
            else f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=kind_l,
            reason=reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "tail_size": int(tail_size),
                "max_schur_size": int(max_schur_size),
                "block_inverse_nbytes_estimate": int(estimate),
                "max_block_inverse_nbytes": int(max_block_inverse_nbytes),
            },
        )

    if kind_l in {"xblock_tz_low_l_schur", "xblock_tz_lowl_schur", "fortran_xblock_tz_schur"}:
        tail_size = int(layout.total_size) - int(layout.f_size)
        config = _xblock_tz_low_l_config(layout)
        estimate = _estimate_xblock_tz_low_l_factor_nbytes(layout=layout, config=config)
        if tail_size > 0 and tail_size <= int(max_schur_size) and estimate <= int(max_block_inverse_nbytes):
            return _build_xblock_tz_low_l_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                max_factor_nbytes=int(max_block_inverse_nbytes),
                config=config,
                t0=t0,
            )
        reason = (
            f"xblock_tz_low_l_schur_budget_exceeded:{estimate}>{int(max_block_inverse_nbytes)}"
            if estimate > int(max_block_inverse_nbytes)
            else f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=kind_l,
            reason=reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "tail_size": int(tail_size),
                "max_schur_size": int(max_schur_size),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_block_inverse_nbytes),
                **config,
            },
        )

    if kind_l in {
        "xblock_tz_low_l_coarse_schur",
        "xblock_tz_lowl_coarse_schur",
        "coarse_residual_schur",
        "field_split_coarse_residual",
    }:
        tail_size = int(layout.total_size) - int(layout.f_size)
        config = _coarse_residual_config(layout)
        estimate = _estimate_xblock_tz_low_l_factor_nbytes(layout=layout, config=config)
        estimate += _estimate_coarse_residual_nbytes(layout=layout, config=config)
        if tail_size > 0 and tail_size <= int(max_schur_size) and estimate <= int(max_block_inverse_nbytes):
            return _build_xblock_tz_low_l_coarse_residual_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                max_factor_nbytes=int(max_block_inverse_nbytes),
                config=config,
                t0=t0,
            )
        reason = (
            f"xblock_tz_low_l_coarse_schur_budget_exceeded:{estimate}>{int(max_block_inverse_nbytes)}"
            if estimate > int(max_block_inverse_nbytes)
            else f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=kind_l,
            reason=reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "tail_size": int(tail_size),
                "max_schur_size": int(max_schur_size),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_block_inverse_nbytes),
                **config,
            },
        )

    if kind_l in {"block_schur", "block_diagonal_schur", "zeta_block_schur"}:
        tail_size = int(layout.total_size) - int(layout.f_size)
        estimate = _estimate_zeta_block_inverse_nbytes(layout)
        if tail_size > 0 and tail_size <= int(max_schur_size) and estimate <= int(max_block_inverse_nbytes):
            return _build_block_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                t0=t0,
            )
        if kind_l != "auto":
            reason = (
                f"block_schur_budget_exceeded:{estimate}>{int(max_block_inverse_nbytes)}"
                if estimate > int(max_block_inverse_nbytes)
                else f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}"
            )
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind=kind_l,
                reason=reason,
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "tail_size": int(tail_size),
                    "max_schur_size": int(max_schur_size),
                    "block_inverse_nbytes_estimate": int(estimate),
                    "max_block_inverse_nbytes": int(max_block_inverse_nbytes),
                },
            )

    if kind_l in {"diagonal_schur", "diag_schur", "field_split"}:
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0 and tail_size <= int(max_schur_size):
            return _build_diagonal_schur_preconditioner(
                matrix=matrix,
                layout=layout,
                requested_kind=kind_l,
                regularization=regularization,
                t0=t0,
            )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=kind_l,
            reason=f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"tail_size": int(tail_size), "max_schur_size": int(max_schur_size)},
        )

    if kind_l in {"jacobi", "diagonal"}:
        return _build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            t0=t0,
            reason="complete",
        )

    return RHS1StructuredFullCSRPreconditioner(
        operator=None,
        selected=False,
        kind=kind_l,
        reason="unsupported_preconditioner",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={},
        )


def build_active_projected_rhs1_full_csr_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout | None = None,
    active_indices: Any | None = None,
    kind: str | None = "auto",
    max_factor_nbytes: int = 64 * 1024 * 1024,
    regularization: float = 1.0e-12,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a host preconditioner for the active projected RHSMode=1 matrix.

    The full-layout Schur preconditioners need the original ``(species, x, xi,
    theta, zeta)`` indexing. Once the inactive pitch-angle rows are projected
    out, that layout is no longer rectangular. This helper therefore uses
    generic sparse active-system preconditioners with explicit memory gates.
    """

    from scipy.sparse.linalg import LinearOperator, spilu  # noqa: PLC0415

    t0 = time.perf_counter()
    matrix = matrix.tocsc()
    if matrix.shape[0] != matrix.shape[1]:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=str(kind or "none"),
            reason="matrix_not_square",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"shape": tuple(int(v) for v in matrix.shape)},
        )
    kind_l = "none" if kind is None else str(kind).strip().lower().replace("-", "_")
    if kind_l in {"", "none", "false", "off", "disabled"}:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="none",
            reason="disabled",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )
    if kind_l in {"auto", "active_auto", "structured_auto"}:
        auto_policy = resolve_active_projected_preconditioner_auto_policy(
            matrix_size=int(matrix.shape[0])
        )
        large_fallback_size = int(auto_policy.large_fallback_size)
        candidates = list(auto_policy.candidates)
        auto_candidates_requested = list(auto_policy.candidates_requested)
        skipped_large_fallbacks = list(auto_policy.skipped_large_fallbacks)
        large_default_used = bool(auto_policy.large_default_used)
        rejected: list[dict[str, object]] = []
        log_large_auto = bool(auto_policy.log_progress)
        for candidate in candidates:
            if candidate in {"auto", "active_auto", "structured_auto"}:
                continue
            candidate_start_s = time.perf_counter()
            if bool(log_large_auto):
                print(
                    "active_projected_rhs1_full_csr_preconditioner: auto candidate start "
                    f"kind={candidate} size={int(matrix.shape[0])} "
                    f"max_factor_mb={float(max_factor_nbytes) / (1024.0 * 1024.0):.1f}",
                    flush=True,
                )
            pc = build_active_projected_rhs1_full_csr_preconditioner(
                matrix=matrix,
                layout=layout,
                active_indices=active_indices,
                kind=candidate,
                max_factor_nbytes=max_factor_nbytes,
                regularization=regularization,
                preconditioner_x=preconditioner_x,
                preconditioner_xi=preconditioner_xi,
                preconditioner_species=preconditioner_species,
                preconditioner_x_min_l=preconditioner_x_min_l,
            )
            candidate_setup_s = max(0.0, time.perf_counter() - candidate_start_s)
            entry = {
                "kind": str(candidate),
                "selected": bool(pc.selected),
                "reason": str(pc.reason),
                "setup_s": float(pc.setup_s),
                "metadata": dict(pc.metadata),
            }
            if bool(log_large_auto):
                print(
                    "active_projected_rhs1_full_csr_preconditioner: auto candidate done "
                    f"kind={candidate} selected={bool(pc.selected)} reason={pc.reason} "
                    f"candidate_elapsed_s={candidate_setup_s:.3f} setup_s={float(pc.setup_s):.3f}",
                    flush=True,
                )
            if bool(pc.selected) and pc.operator is not None:
                metadata = dict(pc.metadata)
                metadata["auto_requested_kind"] = str(kind_l)
                metadata["auto_selected_kind"] = str(pc.kind)
                metadata["auto_candidates"] = list(candidates)
                metadata["auto_candidates_requested"] = list(auto_candidates_requested)
                metadata["auto_skipped_large_fallbacks"] = tuple(skipped_large_fallbacks)
                metadata["auto_large_fallback_size"] = int(large_fallback_size)
                metadata["auto_large_default_used"] = bool(large_default_used)
                metadata["auto_rejected_candidates"] = tuple(rejected)
                return RHS1StructuredFullCSRPreconditioner(
                    operator=pc.operator,
                    selected=True,
                    kind=str(pc.kind),
                    reason=f"auto_selected:{pc.reason}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata=metadata,
                )
            rejected.append(entry)
        if skipped_large_fallbacks or large_default_used:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind=str(kind_l),
                reason="active_auto_no_safe_large_candidate_selected",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "auto_candidates": list(candidates),
                    "auto_candidates_requested": list(auto_candidates_requested),
                    "auto_skipped_large_fallbacks": tuple(skipped_large_fallbacks),
                    "auto_large_fallback_size": int(large_fallback_size),
                    "auto_large_default_used": bool(large_default_used),
                    "auto_rejected_candidates": tuple(rejected),
                },
            )
        fallback = _build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            t0=t0,
            reason="active_auto_no_candidate_selected",
        )
        metadata = dict(fallback.metadata)
        metadata["auto_candidates"] = list(candidates)
        metadata["auto_candidates_requested"] = list(auto_candidates_requested)
        metadata["auto_skipped_large_fallbacks"] = tuple(skipped_large_fallbacks)
        metadata["auto_large_fallback_size"] = int(large_fallback_size)
        metadata["auto_large_default_used"] = bool(large_default_used)
        metadata["auto_rejected_candidates"] = tuple(rejected)
        return RHS1StructuredFullCSRPreconditioner(
            operator=fallback.operator,
            selected=bool(fallback.selected),
            kind=str(fallback.kind),
            reason=str(fallback.reason),
            setup_s=float(fallback.setup_s),
            metadata=metadata,
        )
    if kind_l in {"jacobi", "diagonal"}:
        return _build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            t0=t0,
            reason="active_projected_jacobi",
        )
    if kind_l in {
        "active_diagonal_schur",
        "active_diag_schur",
        "active_tail_schur",
        "active_constraint_tail_schur",
        "active_field_split",
        "active_field_split_tail",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_diagonal_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_diagonal_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_schwarz_sparse_coarse",
        "active_overlap_schwarz_sparse_coarse",
        "active_additive_schwarz_sparse_coarse",
        "active_ras_sparse_coarse",
        "active_tail_sparse_coarse",
        "active_sparse_tail_coarse",
        "active_tail_coarse_residual",
        "active_diagonal_schur_sparse_coarse",
        "active_diagonal_schur_coarse",
        "active_coupled_tail_coarse",
        "active_filtered_sparse_coarse",
        "active_filtered_sparse_factor_sparse_coarse",
        "active_filtered_factor_sparse_coarse",
        "active_physics_filtered_sparse_coarse",
        "active_xblock_sparse_coarse",
        "active_xblock_tail_sparse_coarse",
        "active_xblock_coupled_tail_coarse",
        "active_scaled_ilu_sparse_coarse",
        "active_equilibrated_ilu_sparse_coarse",
        "active_rowcol_ilu_sparse_coarse",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_tail_sparse_coarse",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_sparse_coarse_residual_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_fortran_v3_pc_matrix",
        "active_fortran_v3_reduced_lu",
        "active_fortran_v3_reduced_ilu",
        "active_fortran_v3_reduced_planned_lu",
        "active_fortran_v3_reduced_planned_ilu",
        "active_fortran_v3_planned_reduced_lu",
        "active_fortran_v3_planned_reduced_ilu",
        "active_v3_pc_matrix",
        "active_v3_reduced_lu",
        "active_v3_reduced_ilu",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_fortran_v3_pc_matrix",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_fortran_v3_reduced_sparse_factor_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
            preconditioner_x=preconditioner_x,
            preconditioner_xi=preconditioner_xi,
            preconditioner_species=preconditioner_species,
            preconditioner_x_min_l=preconditioner_x_min_l,
        )
    if kind_l in {
        "active_coarse",
        "active_jacobi_coarse",
        "active_field_split_coarse",
        "active_coarse_residual",
        "active_coarse_ls",
        "active_least_squares_coarse",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_coarse",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_coarse_residual_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            base_kind="jacobi",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_low_l_schur",
        "active_lowl_schur",
        "active_low_pitch_schur",
        "active_full_angle_low_l_schur",
        "active_xblock_low_l_schur",
        "active_xblock_lowl_schur",
        "active_xblock_ilu_low_l_schur",
        "active_xblock_ilu_lowl_schur",
        "active_block_asm_low_l_schur",
        "active_schwarz_low_l_schur",
        "active_overlap_low_l_schur",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_low_l_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_low_l_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            base_kind=(
                "active_xblock_ilu"
                if "xblock_ilu" in kind_l or "block_asm" in kind_l
                else
                "xblock"
                if "xblock" in kind_l
                else "overlap_schwarz"
                if "schwarz" in kind_l or "overlap" in kind_l
                else None
            ),
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_ell_band_schur",
        "active_l_band_schur",
        "active_pitch_band_schur",
        "active_xblock_ell_band_schur",
    }:
        if layout is None or active_indices is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_ell_band_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_ell_band_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            base_kind="xblock" if "xblock" in kind_l else None,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_xell_window_lsq_schur",
        "active_x_ell_window_lsq_schur",
        "active_xell_lsq_schur",
        "active_window_lsq_schur",
    }:
        if layout is None or active_indices is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_xell_window_lsq_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_xell_window_lsq_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_coupled_kinetic_block",
        "active_dominant_kinetic_block",
        "active_coupled_native_block",
        "active_coupled_kinetic",
        "active_native_coupled_kinetic",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_coupled_kinetic_block",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_coupled_kinetic_block_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_filtered_sparse_factor",
        "active_physics_filtered_sparse_factor",
        "active_offdiag_sparse_factor",
        "active_selected_offdiag_sparse_factor",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_filtered_sparse_factor",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_filtered_sparse_factor_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_symbolic_frontal_schur_lu",
        "active_frontal_schur_lu",
        "active_reduced_pmat_frontal_schur",
        "active_native_frontal_schur",
        "active_symbolic_nd_frontal_schur_lu",
        "active_nd_frontal_schur_lu",
        "active_nested_dissection_frontal_schur_lu",
        "active_multilevel_frontal_schur_lu",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_symbolic_frontal_schur_lu",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_symbolic_frontal_schur_lu_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_symbolic_superblock_lu",
        "active_superblock_lu",
        "active_symbolic_grouped_block_lu",
        "active_reduced_pmat_superblock_lu",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_symbolic_superblock_lu",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_symbolic_superblock_lu_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_symbolic_block_schur_lu",
        "active_symbolic_separator_schur_lu",
        "active_separator_schur_lu",
        "active_native_symbolic_block_schur",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_symbolic_block_schur_lu",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_symbolic_block_schur_lu_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_symbolic_coupled_schur",
        "active_coupled_symbolic_schur",
        "active_symbolic_field_split_schur",
        "active_true_schur_residual",
        "active_symbolic_kinetic_schur",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_symbolic_coupled_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_symbolic_coupled_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_bounded_native_stack",
        "active_native_stack",
        "active_block_native_stack",
        "active_multiline_native_stack",
        "active_bounded_block_coarse",
        "active_fortran_v3_reduced_native_stack",
        "active_v3_reduced_native_stack",
        "fortran_v3_reduced_native_stack",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind=(
                    "active_fortran_v3_reduced_native_stack"
                    if "fortran_v3_reduced_native_stack" in kind_l or "v3_reduced_native_stack" in kind_l
                    else "active_bounded_native_stack"
                ),
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        if kind_l in {
            "active_fortran_v3_reduced_native_stack",
            "active_v3_reduced_native_stack",
            "fortran_v3_reduced_native_stack",
        }:
            return _build_active_fortran_v3_reduced_native_stack_preconditioner(
                matrix=matrix,
                layout=layout,
                active_indices=active_indices,
                requested_kind=kind_l,
                regularization=regularization,
                max_factor_nbytes=max_factor_nbytes,
                t0=t0,
                base_preconditioner_factory=build_active_projected_rhs1_full_csr_preconditioner,
            )
        return _build_active_projected_bounded_native_stack_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_native_xell_field_split_sparse_coarse",
        "active_native_xell_global_schur_sparse_coarse",
        "active_native_xell_coupled_coarse_schur",
        "active_global_field_split_native_xell_sparse_coarse",
        "active_angular_line_field_split_sparse_coarse",
        "active_angular_line_global_schur_sparse_coarse",
        "active_angular_line_coupled_coarse_schur",
        "active_global_field_split_angular_line_sparse_coarse",
        "active_multiline_field_split_sparse_coarse",
        "active_xell_angular_field_split_sparse_coarse",
        "active_coupled_multiline_field_split_sparse_coarse",
        "active_global_field_split_multiline_sparse_coarse",
        "active_coupled_kinetic_field_split_sparse_coarse",
        "active_coupled_kinetic_sparse_coarse",
        "active_dominant_kinetic_sparse_coarse",
        "active_true_coupled_kinetic_sparse_coarse",
    }:
        if layout is None:
            missing_kind = "active_native_xell_field_split_sparse_coarse"
            if "coupled_kinetic" in kind_l or "dominant_kinetic" in kind_l:
                missing_kind = "active_coupled_kinetic_field_split_sparse_coarse"
            if "angular" in kind_l and "xell_angular" not in kind_l:
                missing_kind = "active_angular_line_field_split_sparse_coarse"
            if "multiline" in kind_l or "xell_angular" in kind_l:
                missing_kind = "active_multiline_field_split_sparse_coarse"
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind=missing_kind,
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_native_xell_field_split_sparse_coarse_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_global_field_split_schur",
        "active_field_split_schur",
        "active_xblock_global_schur",
        "active_kinetic_tail_schur",
        "active_kinetic_global_schur",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_global_field_split_schur",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_global_field_split_schur_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
            base_preconditioner_factory=build_active_projected_rhs1_full_csr_preconditioner,
        )
    if kind_l in {
        "active_overlap_schwarz",
        "active_additive_schwarz",
        "active_restricted_schwarz",
        "active_ras",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_overlap_schwarz",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_overlap_schwarz_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_angular_line", "active_tz_line", "active_theta_zeta_line"}:
        if layout is None or active_indices is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_angular_line",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_angular_line_preconditioner(
            matrix=matrix,
            layout=layout,
            active_kinetic_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_native_indexed_schwarz",
        "active_indexed_schwarz",
        "active_multiline_indexed_schwarz",
        "active_xell_angular_native_schwarz",
        "active_native_multiline_schwarz",
    }:
        if layout is None or active_indices is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_native_indexed_schwarz",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_native_indexed_schwarz_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_xblock",
        "active_x_block",
        "active_species_xblock",
        "active_x_species_block",
        "active_xblock_ilu",
        "active_xblock_spilu",
        "active_block_asm",
        "active_block_asm_ilu",
    }:
        if layout is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_xblock",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_xblock_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {"active_ilu_coarse", "active_spilu_coarse"}:
        if layout is None or active_indices is None:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_ilu_coarse",
                reason="missing_active_layout",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={},
            )
        return _build_active_projected_coarse_residual_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            requested_kind=kind_l,
            base_kind="active_ilu",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_global_sparse_factor",
        "active_global_sparse_lu",
        "active_global_sparse_ilu",
        "active_global_lu",
        "active_global_ilu",
        "active_sparse_lu",
        "active_direct_lu",
        "active_direct_sparse_lu",
        "active_direct_sparse_factor",
    }:
        return _build_active_global_sparse_factor_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if kind_l in {
        "active_scaled_ilu",
        "active_equilibrated_ilu",
        "active_rowcol_ilu",
        "active_scaled_spilu",
        "active_scaled_lu",
    }:
        return _build_active_scaled_sparse_factor_preconditioner(
            matrix=matrix,
            requested_kind=kind_l,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    explicit_ilu = kind_l in {"active_ilu", "active_spilu", "ilu", "spilu", "incomplete_lu"}
    if kind_l == "auto" or explicit_ilu:
        drop_tol = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_DROP_TOL", 1.0e-3))
        fill_factor = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_FILL_FACTOR", 4.0))
        diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_DIAG_PIVOT_THRESH", 0.0))
        permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ILU_PERMC_SPEC", "COLAMD").strip().upper()
        if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
            permc_spec = "COLAMD"
        estimate = _estimate_spilu_factor_nbytes(matrix=matrix, fill_factor=fill_factor)
        if estimate > int(max_factor_nbytes):
            if kind_l == "auto":
                return _build_jacobi_preconditioner(
                    matrix=matrix,
                    requested_kind=kind_l,
                    regularization=regularization,
                    t0=t0,
                    reason=f"active_spilu_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
                )
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_spilu",
                reason=f"active_spilu_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "matrix_shape": tuple(int(v) for v in matrix.shape),
                    "matrix_nnz": int(matrix.nnz),
                    "factor_nbytes_estimate": int(estimate),
                    "max_factor_nbytes": int(max_factor_nbytes),
                    "drop_tol": float(drop_tol),
                    "fill_factor": float(fill_factor),
                    "permc_spec": str(permc_spec),
                },
            )
        try:
            factor = spilu(
                matrix,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
        except Exception as exc:  # noqa: BLE001
            if kind_l == "auto":
                return _build_jacobi_preconditioner(
                    matrix=matrix,
                    requested_kind=kind_l,
                    regularization=regularization,
                    t0=t0,
                    reason=f"active_spilu_failed:{type(exc).__name__}",
                )
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_spilu",
                reason=f"active_spilu_failed:{type(exc).__name__}",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={"error": str(exc), "factor_nbytes_estimate": int(estimate)},
            )
        operator = LinearOperator(matrix.shape, matvec=lambda x: factor.solve(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
        return RHS1StructuredFullCSRPreconditioner(
            operator=operator,
            selected=True,
            kind="active_spilu",
            reason="complete",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(kind_l),
                "matrix_shape": tuple(int(v) for v in matrix.shape),
                "matrix_nnz": int(matrix.nnz),
                "factor_nnz": int(factor.L.nnz + factor.U.nnz),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "drop_tol": float(drop_tol),
                "fill_factor": float(fill_factor),
                "diag_pivot_thresh": float(diag_pivot),
                "permc_spec": str(permc_spec),
            },
        )
    return RHS1StructuredFullCSRPreconditioner(
        operator=None,
        selected=False,
        kind=kind_l,
        reason="unsupported_active_projected_preconditioner",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={},
    )


def _direct_active_fortran_v3_reduced_pmat_input_matrix(
    *,
    op: Any,
    active_indices: Any | None,
    include_identity_shift: bool = True,
    include_jacobian_terms: bool = True,
    drop_tol: float = 0.0,
    max_csr_nbytes: int | None = None,
) -> tuple[Any, RHS1BlockLayout, np.ndarray, dict[str, object]]:
    """Emit an active reduced-Pmat input matrix without full active CSR assembly."""

    layout = RHS1BlockLayout.from_operator(op)
    reason = _unsupported_reason(op=op, include_jacobian_terms=include_jacobian_terms)
    if reason is not None:
        raise ValueError(reason)

    compressed_layout = infer_rhs1_compressed_pitch_layout_from_active_indices(
        layout,
        active_indices,
    )
    active = compressed_layout.active_full_indices.astype(np.int64, copy=False)

    phi1_base = getattr(op, "phi1_hat_base", None)
    fblock_selection = select_structured_rhs1_fblock_csr_operator(
        op.fblock,
        include_identity_shift=bool(include_identity_shift),
        phi1_hat_base=phi1_base,
        drop_tol=float(drop_tol),
        require_complete=True,
        max_csr_nbytes=max_csr_nbytes,
        use_cache=True,
    )
    if not bool(fblock_selection.selected) or fblock_selection.matrix is None:
        raise ValueError(f"fblock_not_selected:{fblock_selection.reason}")

    kinetic_full = compressed_layout.kinetic_active_full_indices.astype(np.int64, copy=False)
    kinetic = fblock_selection.matrix.tocsr()[kinetic_full[:, None], kinetic_full].tocsr()
    tail = _assemble_full_tail_csr(op=op, layout=layout, include_jacobian_terms=include_jacobian_terms)
    tail_active = tail[active[:, None], active].tocsr()
    if int(compressed_layout.tail_size) > 0:
        zero_tail = sp.csr_matrix((int(compressed_layout.tail_size), int(compressed_layout.tail_size)), dtype=np.float64)
        base = sp.block_diag((kinetic, zero_tail), format="csr")
    else:
        base = kinetic
    direct = (base + tail_active).tocsr()
    direct.sum_duplicates()
    direct.eliminate_zeros()

    direct_nbytes = _scipy_csr_nbytes(direct)
    if max_csr_nbytes is not None and int(direct_nbytes) > int(max_csr_nbytes):
        raise ValueError(f"direct_reduced_pmat_budget_exceeded:{int(direct_nbytes)}>{int(max_csr_nbytes)}")

    metadata = {
        "direct_reduced_pmat_emission": True,
        "direct_reduced_pmat_source": "structured_fblock_csr_plus_direct_tail",
        "direct_reduced_pmat_shape": tuple(int(v) for v in direct.shape),
        "direct_reduced_pmat_nnz": int(direct.nnz),
        "direct_reduced_pmat_nbytes": int(direct_nbytes),
        "direct_reduced_pmat_max_csr_nbytes": None if max_csr_nbytes is None else int(max_csr_nbytes),
        "direct_reduced_pmat_kinetic_nnz": int(kinetic.nnz),
        "direct_reduced_pmat_tail_nnz": int(tail_active.nnz),
        "direct_reduced_pmat_avoids_full_active_true_csr": True,
        "compressed_layout": {
            "nxi_for_x": tuple(int(v) for v in compressed_layout.nxi_for_x),
            "first_index_for_x": tuple(int(v) for v in compressed_layout.first_index_for_x),
            "kinetic_active_size": int(compressed_layout.kinetic_active_size),
            "tail_size": int(compressed_layout.tail_size),
            "reduced_size": int(compressed_layout.reduced_size),
        },
        "fblock_selection": fblock_selection.to_dict(),
    }
    return direct, layout, active, metadata


def build_direct_active_fortran_v3_reduced_pmat_preconditioner(
    *,
    op: Any,
    active_indices: Any | None,
    requested_kind: str = "active_fortran_v3_reduced_direct_pmat_lu",
    regularization: float = 1.0e-12,
    max_factor_nbytes: int = 512 * 1024 * 1024,
    max_csr_nbytes: int | None = None,
    include_identity_shift: bool = True,
    include_jacobian_terms: bool = True,
    drop_tol: float = 0.0,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a reduced-Pmat preconditioner without materializing true active CSR."""

    t0 = time.perf_counter()
    active_size_estimate = (
        int(getattr(op, "total_size", 0))
        if active_indices is None
        else int(np.asarray(active_indices, dtype=np.int64).reshape((-1,)).size)
    )
    emission_max_size = int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_DIRECT_REDUCED_PMAT_EMISSION_MAX_SIZE", 350_000))
    if int(emission_max_size) > 0 and int(active_size_estimate) > int(emission_max_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_reduced_direct_pmat",
            reason=f"direct_reduced_pmat_emission_size_exceeded:{active_size_estimate}>{int(emission_max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "direct_reduced_pmat_emission": False,
                "direct_reduced_pmat_emission_active_size_estimate": int(active_size_estimate),
                "direct_reduced_pmat_emission_max_size": int(emission_max_size),
                "note": (
                    "large direct-Pmat emission is fail-closed by default; raise "
                    "SFINCS_JAX_RHS1_FULL_CSR_DIRECT_REDUCED_PMAT_EMISSION_MAX_SIZE "
                    "only for explicit diagnostics"
                ),
            },
        )
    try:
        pmat_input, layout, active, emission_metadata = _direct_active_fortran_v3_reduced_pmat_input_matrix(
            op=op,
            active_indices=active_indices,
            include_identity_shift=include_identity_shift,
            include_jacobian_terms=include_jacobian_terms,
            drop_tol=drop_tol,
            max_csr_nbytes=max_csr_nbytes,
        )
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_reduced_direct_pmat",
            reason="direct_reduced_pmat_emission_failed",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc), "direct_reduced_pmat_emission": False},
        )

    pc = _build_active_fortran_v3_reduced_sparse_factor_preconditioner(
        matrix=pmat_input,
        layout=layout,
        active_indices=active,
        requested_kind=requested_kind,
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
        preconditioner_x=preconditioner_x,
        preconditioner_xi=preconditioner_xi,
        preconditioner_species=preconditioner_species,
        preconditioner_x_min_l=preconditioner_x_min_l,
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=pc.operator,
        selected=bool(pc.selected),
        kind=str(pc.kind),
        reason=str(pc.reason),
        setup_s=float(pc.setup_s),
        metadata={
            **dict(pc.metadata),
            **emission_metadata,
        },
    )







def _build_active_projected_coupled_kinetic_block_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Factor a dominant active kinetic subspace with true off-diagonal couplings.

    Previous bounded native candidates inverted separate ``(x, ell)`` or
    angular lines and then relied on a Schur/coarse residual equation to repair
    the missing cross-line physics.  This path instead keeps the dominant
    low-speed/low-pitch kinetic angular block as one sparse active-only factor,
    optionally including the Phi1/source/constraint tail.  The coarse layer can
    then act as a correction to a coupled local inverse rather than as the main
    rescue mechanism.
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
            kind="active_coupled_kinetic_block",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    try:
        ordering = RHS1ActiveFieldSplitOrdering.cached_from_layout(layout, active_np)
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason="invalid_active_ordering",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    max_block_size = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", 32768)),
    )
    max_positions = max(
        0,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS",
                int(max_block_size),
            )
        ),
    )
    x_count = max(
        0,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT",
                min(1, int(layout.n_x)),
            )
        ),
    )
    ell_count = max(
        0,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT",
                min(6, int(layout.n_xi)),
            )
        ),
    )
    species_count_env = os.environ.get(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_SPECIES_COUNT",
        "",
    ).strip()
    species_count = None if species_count_env == "" else int(species_count_env)
    theta_stride = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_THETA_STRIDE", 1)),
    )
    zeta_stride = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ZETA_STRIDE", 1)),
    )
    kinetic_positions = ordering.dominant_kinetic_positions(
        x_count=int(x_count),
        ell_count=int(ell_count),
        species_count=species_count,
        theta_stride=int(theta_stride),
        zeta_stride=int(zeta_stride),
        max_positions=int(max_positions),
    )
    kinetic_selected = int(kinetic_positions.size)
    include_tail = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_INCLUDE_TAIL", True)
    max_tail = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_TAIL", 512)),
    )
    positions = kinetic_positions.astype(np.int64, copy=False)
    tail_selected = 0
    if bool(include_tail) and int(positions.size) < int(max_block_size) and int(max_tail) > 0:
        tail_candidates = np.concatenate(
            (
                ordering.phi1_positions.astype(np.int64, copy=False),
                ordering.extra_positions.astype(np.int64, copy=False),
            )
        )
        if int(tail_candidates.size) > 0:
            remaining = min(int(max_tail), int(max_block_size) - int(positions.size))
            tail_candidates = tail_candidates[:remaining]
            tail_selected = int(tail_candidates.size)
            positions = np.concatenate((positions, tail_candidates)).astype(np.int64, copy=False)
    if int(positions.size) > 0:
        _, first = np.unique(positions, return_index=True)
        positions = positions[np.sort(first)].astype(np.int64, copy=False)
    block_size = int(positions.size)
    if block_size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason="empty_coupled_kinetic_block",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "x_count": int(x_count),
                "ell_count": int(ell_count),
                "max_positions": int(max_positions),
                "symbolic_ordering": ordering.to_dict(),
            },
        )
    if block_size > int(max_block_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason=f"active_coupled_kinetic_block_size_exceeded:{block_size}>{int(max_block_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_size": int(block_size),
                "max_block_size": int(max_block_size),
                "kinetic_selected": int(kinetic_selected),
                "tail_selected": int(tail_selected),
                "symbolic_ordering": ordering.to_dict(),
            },
        )

    base_kind = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "jacobi")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if base_kind in {"", "default"}:
        base_kind = "jacobi"
    if base_kind in {"zero", "none", "off", "disabled"}:
        def zero_apply(x: Any) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float64).reshape((-1,))
            return np.zeros_like(arr, dtype=np.float64)

        base = RHS1StructuredFullCSRPreconditioner(
            operator=LinearOperator(matrix_csr.shape, matvec=zero_apply, dtype=np.float64),
            selected=True,
            kind="zero_coupled_kinetic_base",
            reason="complete",
            setup_s=0.0,
            metadata={"factor_nbytes_actual": 0, "base_mode": "zero"},
        )
    elif base_kind in {"jacobi", "diagonal"}:
        base = _build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind="active_coupled_kinetic_base_jacobi",
            regularization=regularization,
            t0=t0,
            reason="active_coupled_kinetic_base_jacobi",
        )
    else:
        if base_kind in {
            "active_coupled_kinetic_block",
            "active_dominant_kinetic_block",
            "active_coupled_native_block",
            "active_coupled_kinetic",
            "active_native_coupled_kinetic",
        }:
            base_kind = "jacobi"
        base = build_active_projected_rhs1_full_csr_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            kind=base_kind,
            max_factor_nbytes=max(1, int(max_factor_nbytes) // 4),
            regularization=regularization,
        )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "block_size": int(block_size),
                "symbolic_ordering": ordering.to_dict(),
            },
        )

    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind_env = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FACTOR_KIND", "auto")
        .strip()
        .lower()
    )
    exact_max_size = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_EXACT_MAX_SIZE", 4096)),
    )
    if factor_kind_env in {"auto", ""}:
        factor_kind = "splu" if block_size <= int(exact_max_size) else "spilu"
    elif factor_kind_env in {"lu", "splu", "exact", "direct"}:
        factor_kind = "splu"
    elif factor_kind_env in {"ilu", "spilu", "incomplete_lu"}:
        factor_kind = "spilu"
    elif "ilu" in requested:
        factor_kind = "spilu"
    else:
        factor_kind = "splu"
    fill_default = 6.0 if factor_kind == "spilu" else 24.0
    fill_factor = max(
        1.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FILL_FACTOR", fill_default)),
    )
    drop_default = 1.0e-4 if factor_kind == "spilu" else 0.0
    drop_tol = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_DROP_TOL", drop_default)),
    )
    diag_pivot = float(
        _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_DIAG_PIVOT_THRESH", 0.0)
    )
    permc_spec = os.environ.get(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_PERMC_SPEC",
        "COLAMD",
    ).strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    allow_ilu_fallback = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ALLOW_ILU_FALLBACK",
        True,
    )
    scale_block = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_SCALE", True)
    scale_norm = os.environ.get(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_SCALE_NORM",
        "l1",
    ).strip().lower()
    max_scale = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_SCALE", 1.0e6))
    damping = max(
        0.0,
        min(2.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_DAMPING", 1.0))),
    )
    diagonal_shift_override = _env_float(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_DIAGONAL_SHIFT",
        float("nan"),
    )

    block_raw = matrix_csr[positions[:, None], positions].tocsc()
    block_scale = max(float(np.max(np.abs(block_raw.data))) if block_raw.nnz else 0.0, 1.0)
    if np.isfinite(float(diagonal_shift_override)):
        diagonal_shift = max(0.0, float(diagonal_shift_override)) * block_scale
    else:
        diagonal_shift = max(float(abs(regularization)), 1.0e-14) * block_scale
    row_scale = np.ones((block_size,), dtype=np.float64)
    col_scale = np.ones((block_size,), dtype=np.float64)
    scale_nbytes = 0
    row_metadata: dict[str, object] = {}
    col_metadata: dict[str, object] = {}
    if bool(scale_block):
        row_scale, row_metadata = _sparse_equilibration_scale(
            block_raw,
            axis=1,
            norm=scale_norm,
            max_scale=max_scale,
        )
        block_row_scaled = block_raw.multiply(row_scale[:, None]).tocsc()
        col_scale, col_metadata = _sparse_equilibration_scale(
            block_row_scaled,
            axis=0,
            norm=scale_norm,
            max_scale=max_scale,
        )
        block = block_row_scaled.multiply(col_scale[None, :]).tocsc()
        scale_nbytes = int(row_scale.nbytes + col_scale.nbytes)
    else:
        block = block_raw
    if diagonal_shift > 0.0:
        block = block + diagonal_shift * sp.eye(block_size, dtype=np.float64, format="csc")
    block_nbytes = int(_scipy_csr_nbytes(block.tocsr()))
    sparse_estimate = int(_estimate_spilu_factor_nbytes(matrix=block, fill_factor=fill_factor))
    dense_estimate = int(block_size * block_size * np.dtype(np.float64).itemsize)
    factor_estimate = int(
        (sparse_estimate if factor_kind == "spilu" else max(sparse_estimate, dense_estimate))
        + block_nbytes
        + scale_nbytes
    )
    base_nbytes = int(base.metadata.get("factor_nbytes_actual", base.metadata.get("factor_nbytes_estimate", 0)) or 0)
    total_estimate = int(base_nbytes + factor_estimate)
    if total_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason=f"active_coupled_kinetic_budget_exceeded:{total_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_size": int(block_size),
                "kinetic_selected": int(kinetic_selected),
                "tail_selected": int(tail_selected),
                "block_nnz": int(block.nnz),
                "block_nbytes_actual": int(block_nbytes),
                "factor_nbytes_estimate": int(total_estimate),
                "local_factor_nbytes_estimate": int(factor_estimate),
                "base_factor_nbytes_actual": int(base_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "block_scaling_enabled": bool(scale_block),
                "symbolic_ordering": ordering.to_dict(),
            },
        )

    factor_kind_actual = str(factor_kind)
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
        if factor_kind == "splu" and bool(allow_ilu_fallback):
            try:
                factor = spilu(
                    block,
                    drop_tol=max(float(drop_tol), 1.0e-4),
                    fill_factor=max(float(fill_factor), 6.0),
                    permc_spec=str(permc_spec),
                    diag_pivot_thresh=float(diag_pivot),
                )
                factor_kind_actual = "spilu_fallback"
            except Exception as fallback_exc:  # noqa: BLE001
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="active_coupled_kinetic_block",
                    reason=f"active_coupled_kinetic_factor_failed:{type(fallback_exc).__name__}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "error": str(fallback_exc),
                        "first_error": str(exc),
                        "block_size": int(block_size),
                        "block_nnz": int(block.nnz),
                        "factor_kind": str(factor_kind),
                        "allow_ilu_fallback": bool(allow_ilu_fallback),
                    },
                )
        else:
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind="active_coupled_kinetic_block",
                reason=f"active_coupled_kinetic_factor_failed:{type(exc).__name__}",
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "error": str(exc),
                    "block_size": int(block_size),
                    "block_nnz": int(block.nnz),
                    "factor_kind": str(factor_kind),
                    "allow_ilu_fallback": bool(allow_ilu_fallback),
                },
            )
    factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
    actual_total = int(base_nbytes + block_nbytes + scale_nbytes + factor_nbytes)
    if actual_total > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coupled_kinetic_block",
            reason=f"active_coupled_kinetic_budget_exceeded_actual:{actual_total}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "block_size": int(block_size),
                "kinetic_selected": int(kinetic_selected),
                "tail_selected": int(tail_selected),
                "block_nnz": int(block.nnz),
                "factor_nbytes_actual": int(actual_total),
                "local_factor_nbytes_actual": int(factor_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "factor_kind": str(factor_kind_actual),
                "symbolic_ordering": ordering.to_dict(),
            },
        )

    def solve_block(residual: np.ndarray) -> np.ndarray:
        rhs_block = np.asarray(residual[positions], dtype=np.float64).reshape((-1,))
        if bool(scale_block):
            solved = np.asarray(factor.solve(row_scale * rhs_block), dtype=np.float64).reshape((-1,))
            return col_scale * solved
        return np.asarray(factor.solve(rhs_block), dtype=np.float64).reshape((-1,))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        if arr.shape != (int(matrix_csr.shape[0]),):
            raise ValueError(f"rhs must have shape {(int(matrix_csr.shape[0]),)}, got {arr.shape}")
        out = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ out, dtype=np.float64).reshape((-1,))
        out = out.copy()
        out[positions] += float(damping) * solve_block(residual)
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_coupled_kinetic_block",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_dominant_kinetic_sparse_coupled_factor",
            "active_size": int(matrix_csr.shape[0]),
            "block_size": int(block_size),
            "kinetic_selected": int(kinetic_selected),
            "tail_selected": int(tail_selected),
            "block_covers_active": bool(block_size == int(matrix_csr.shape[0])),
            "x_count": int(x_count),
            "ell_count": int(ell_count),
            "species_count": None if species_count is None else int(species_count),
            "theta_stride": int(theta_stride),
            "zeta_stride": int(zeta_stride),
            "requested_base_kind": str(base_kind),
            "base_kind": str(base.kind),
            "base_preconditioner": base.to_dict(),
            "factor_kind": str(factor_kind_actual),
            "factor_requested_kind": str(factor_kind),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "block_nnz": int(block.nnz),
            "block_nbytes_actual": int(block_nbytes),
            "local_factor_nbytes_estimate": int(factor_estimate),
            "local_factor_nbytes_actual": int(factor_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "factor_nbytes_estimate": int(total_estimate),
            "factor_nbytes_actual": int(actual_total),
            "max_factor_nbytes": int(max_factor_nbytes),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "diag_pivot_thresh": float(diag_pivot),
            "permc_spec": str(permc_spec),
            "diagonal_shift": float(diagonal_shift),
            "damping": float(damping),
            "block_scaling_enabled": bool(scale_block),
            "block_scale_norm": str(scale_norm),
            "block_scale_nbytes_actual": int(scale_nbytes),
            "row_scale_metadata": dict(row_metadata),
            "col_scale_metadata": dict(col_metadata),
            "symbolic_ordering": ordering.to_dict(),
            "note": "active_only_coupled_factor_with_schur_as_correction_base",
            "requires_preflight": True,
        },
    )




def _build_active_symbolic_dominant_kinetic_basis_csc(
    *,
    ordering: RHS1ActiveFieldSplitOrdering,
    max_columns: int,
) -> tuple[Any, dict[str, object]]:
    """Build sparse identity columns for bounded symbolic active subspaces.

    The dominant kinetic modes are the primary target for production grids, but
    small active systems also need the source/constraint/profile tail to be
    represented exactly.  The tail identity columns are cheap when the active
    set is small and make the residual equation a true active-space solve
    instead of a kinetic-only probe.
    """

    x_count = _env_int(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_X_COUNT",
        min(1, int(ordering.layout.n_x)),
    )
    ell_count = _env_int(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ELL_COUNT",
        min(6, int(ordering.layout.n_xi)),
    )
    species_count_env = os.environ.get(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_SPECIES_COUNT",
        "",
    ).strip()
    species_count = None if species_count_env == "" else int(species_count_env)
    theta_stride = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_THETA_STRIDE", 1)
    zeta_stride = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ZETA_STRIDE", 1)
    max_use = max(0, int(max_columns))
    positions = ordering.dominant_kinetic_positions(
        x_count=max(0, int(x_count)),
        ell_count=max(0, int(ell_count)),
        species_count=species_count,
        theta_stride=max(1, int(theta_stride)),
        zeta_stride=max(1, int(zeta_stride)),
        max_positions=max_use,
    )
    kinetic_columns = int(positions.size)
    include_tail_identity = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_INCLUDE_TAIL_IDENTITY",
        True,
    )
    tail_columns = 0
    if bool(include_tail_identity) and int(positions.size) < int(max_use):
        tail_candidates = np.concatenate(
            (
                ordering.phi1_positions.astype(np.int64, copy=False),
                ordering.extra_positions.astype(np.int64, copy=False),
            )
        )
        if int(tail_candidates.size) > 0:
            remaining = int(max_use) - int(positions.size)
            tail_candidates = tail_candidates[:remaining]
            tail_columns = int(tail_candidates.size)
            positions = np.concatenate((positions.astype(np.int64, copy=False), tail_candidates))
            if int(positions.size) > 0:
                _, first = np.unique(positions, return_index=True)
                positions = positions[np.sort(first)].astype(np.int64, copy=False)
                if int(positions.size) > int(max_use):
                    positions = positions[:max_use]
                    tail_columns = max(0, int(positions.size) - kinetic_columns)
    unique_positions = np.unique(positions) if int(positions.size) > 0 else positions
    covers_active = int(unique_positions.size) == int(ordering.active_size)
    metadata = {
        "symbolic_kinetic_basis_columns": int(kinetic_columns),
        "symbolic_tail_identity_columns": int(tail_columns),
        "symbolic_identity_basis_columns": int(positions.size),
        "symbolic_kinetic_basis_max_columns": int(max_use),
        "symbolic_kinetic_basis_x_count": int(max(0, int(x_count))),
        "symbolic_kinetic_basis_ell_count": int(max(0, int(ell_count))),
        "symbolic_kinetic_basis_species_count": (
            None if species_count is None else int(species_count)
        ),
        "symbolic_kinetic_basis_theta_stride": int(max(1, int(theta_stride))),
        "symbolic_kinetic_basis_zeta_stride": int(max(1, int(zeta_stride))),
        "symbolic_tail_identity_requested": bool(include_tail_identity),
        "symbolic_identity_basis_unique_columns": int(unique_positions.size),
        "symbolic_identity_basis_covers_active": bool(covers_active),
    }
    if positions.size == 0:
        return sp.csc_matrix((int(ordering.active_size), 0), dtype=np.float64), metadata
    cols = np.arange(int(positions.size), dtype=np.int64)
    basis = sp.coo_matrix(
        (np.ones((int(positions.size),), dtype=np.float64), (positions, cols)),
        shape=(int(ordering.active_size), int(positions.size)),
    ).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    metadata["symbolic_kinetic_basis_nnz"] = int(basis.nnz)
    return basis, metadata


def _active_nonkinetic_tail_size(
    *,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray,
) -> tuple[int, bool]:
    """Return the active-vector suffix size occupied by non-kinetic rows."""

    active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active.size == 0:
        return 0, False
    nonkinetic = active >= int(layout.f_size)
    tail_size = int(np.count_nonzero(nonkinetic))
    if tail_size == 0:
        return 0, True
    return tail_size, bool(np.all(nonkinetic[-tail_size:]))


def _admit_linear_operator_against_matrix(
    *,
    matrix: Any,
    operator: Any,
    probe_count: int,
    max_relative_residual: float,
    min_improvement_vs_identity: float,
) -> tuple[bool, dict[str, object]]:
    """Gate a preconditioner with deterministic true residual probes."""

    from sfincs_jax.explicit_sparse import deterministic_sparse_probe_matrix  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    n_rows, n_cols = int(matrix_csr.shape[0]), int(matrix_csr.shape[1])
    if n_rows != n_cols:
        metadata = {
            "accepted": False,
            "reason": "operator_not_square",
            "max_relative_residual": float("inf"),
            "median_relative_residual": float("inf"),
            "min_improvement_vs_identity": 0.0,
            "probe_count": 0,
        }
        return False, metadata
    probes = deterministic_sparse_probe_matrix(
        n_rows,
        count=max(1, int(probe_count)),
        dtype=matrix_csr.dtype,
    )
    tiny = np.finfo(np.float64).tiny
    residuals: list[float] = []
    improvements: list[float] = []
    for col in range(int(probes.shape[1])):
        rhs = np.asarray(probes[:, col], dtype=matrix_csr.dtype).reshape((n_rows,))
        rhs_norm = max(tiny, float(np.linalg.norm(rhs.astype(np.float64, copy=False))))
        try:
            y = np.asarray(operator.matvec(rhs), dtype=matrix_csr.dtype).reshape((n_rows,))
            residual = np.asarray(matrix_csr @ y - rhs, dtype=np.float64).reshape((-1,))
            identity_residual = np.asarray(matrix_csr @ rhs - rhs, dtype=np.float64).reshape((-1,))
            rel = float(np.linalg.norm(residual) / rhs_norm)
            identity_rel = float(np.linalg.norm(identity_residual) / rhs_norm)
        except Exception:  # noqa: BLE001
            rel = float("inf")
            identity_rel = 0.0
        if not np.isfinite(rel):
            rel = float("inf")
        residuals.append(float(rel))
        if rel <= tiny:
            improvement = float("inf") if identity_rel > tiny else 1.0
        else:
            improvement = float(identity_rel / rel) if np.isfinite(identity_rel) else 0.0
        if not np.isfinite(improvement) and improvement != float("inf"):
            improvement = 0.0
        improvements.append(float(improvement))
    residuals_np = np.asarray(residuals, dtype=np.float64)
    improvements_np = np.asarray(improvements, dtype=np.float64)
    max_rel = float(np.max(residuals_np)) if residuals_np.size else float("inf")
    median_rel = float(np.median(residuals_np)) if residuals_np.size else float("inf")
    min_improvement = float(np.min(improvements_np)) if improvements_np.size else 0.0
    accepted = bool(
        np.isfinite(max_rel)
        and max_rel <= float(max_relative_residual)
        and min_improvement >= float(min_improvement_vs_identity)
    )
    metadata = {
        "accepted": bool(accepted),
        "reason": "accepted" if accepted else "residual_or_improvement_gate_failed",
        "max_relative_residual": float(max_rel),
        "median_relative_residual": float(median_rel),
        "min_improvement_vs_identity": float(min_improvement),
        "probe_count": int(probes.shape[1]),
        "max_relative_residual_gate": float(max_relative_residual),
        "min_improvement_vs_identity_gate": float(min_improvement_vs_identity),
    }
    return accepted, metadata


def _build_active_projected_symbolic_frontal_schur_lu_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded frontal/Schur factor over the active reduced-Pmat.

    The separator is selected from source/constraint tail rows, high-degree
    rows, block boundaries, and endpoints of unresolved inter-frontal couplings.
    This is the first native path that keeps a Schur equation for global
    coupling instead of relying on local block factors plus a post-hoc coarse
    correction.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    from sfincs_jax.explicit_sparse import (  # noqa: PLC0415
        admit_sparse_factor_against_operator,
        analyze_sparse_symbolic_structure,
        factorize_host_sparse_operator,
    )

    matrix_csr = matrix.tocsr()
    active_size = int(matrix_csr.shape[0])
    policy = resolve_active_symbolic_frontal_policy(
        requested_kind=requested_kind,
        active_size=int(active_size),
        regularization=float(regularization),
    )
    use_nd_frontal = bool(policy.use_nd_frontal)
    active_symbolic_kind = str(policy.active_symbolic_kind)
    active_architecture = str(policy.active_architecture)
    max_active_size = int(policy.max_active_size)
    if int(max_active_size) > 0 and int(active_size) > int(max_active_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason=f"{active_symbolic_kind}_size_exceeded:{active_size}>{int(max_active_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(active_size),
                "max_active_size": int(max_active_size),
                "requested_kind": str(requested_kind),
            },
        )

    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (active_size,):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    tail_size, nonkinetic_tail_is_suffix = _active_nonkinetic_tail_size(
        layout=layout,
        active_indices=active_np,
    )

    ordering_kind = str(policy.ordering_kind)
    block_size = int(policy.block_size)
    max_permutation_size = int(policy.max_permutation_size)
    separator_cols = int(policy.separator_cols)
    max_superblock_size = int(policy.max_superblock_size)
    max_superblock_blocks = int(policy.max_superblock_blocks)
    boundary_width = int(policy.boundary_width)
    high_degree_cols = int(policy.high_degree_cols)
    min_cross_nnz = int(policy.min_cross_nnz)
    max_dense_rhs_entries = int(policy.max_dense_rhs_entries)
    max_dense_rhs_cols_per_block = int(policy.max_dense_rhs_cols_per_block)
    min_cross_separator_fraction = float(policy.min_cross_separator_fraction)
    regularization_rel = float(policy.regularization_rel)
    nd_max_leaf_size = int(policy.nd_max_leaf_size)
    nd_max_terminal_factor_size = int(policy.nd_max_terminal_factor_size)
    nd_max_depth = int(policy.nd_max_depth)
    nd_separator_width = int(policy.nd_separator_width)
    nd_max_separator_cols = int(policy.nd_max_separator_cols)
    nd_high_degree_cols = int(policy.nd_high_degree_cols)
    nd_max_dense_rhs_entries = int(policy.nd_max_dense_rhs_entries)
    nd_max_dense_rhs_entries_per_child = int(policy.nd_max_dense_rhs_entries_per_child)
    nd_max_dense_rhs_cols_per_child = int(policy.nd_max_dense_rhs_cols_per_child)
    nd_max_setup_s = float(policy.nd_max_setup_s)
    nd_residual_polish_steps = int(policy.nd_residual_polish_steps)
    nd_residual_polish_damping = float(policy.nd_residual_polish_damping)
    analysis = analyze_sparse_symbolic_structure(
        matrix_csr,
        ordering_kind=ordering_kind,
        block_size_target=block_size,
        max_permutation_size=max_permutation_size,
    )
    csr_nbytes = int(_scipy_csr_nbytes(matrix_csr))
    group_factor = max(1.0, float(max_superblock_size) / float(max(1, block_size)))
    separator_dense_estimate = int(separator_cols) * int(separator_cols) * np.dtype(np.float64).itemsize
    raw_estimate = int(np.ceil(float(csr_nbytes) * (1.0 + np.sqrt(group_factor)) + 4.0 * separator_dense_estimate))
    prefill_safety = float(policy.prefill_safety_factor)
    prefill_estimate = int(np.ceil(float(raw_estimate) * float(prefill_safety)))
    dense_rhs_entries_estimate = int(max(0, active_size - separator_cols)) * int(separator_cols)
    if prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason=f"{active_symbolic_kind}_prefill_budget_exceeded:{prefill_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": active_architecture,
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "prefill_safety_factor": float(prefill_safety),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    try:
        factor = factorize_host_sparse_operator(
            matrix_csr,
            kind="symbolic_nd_frontal_schur_lu" if bool(use_nd_frontal) else "symbolic_frontal_schur_lu",
            symbolic_analysis=analysis,
            symbolic_block_size=block_size,
            symbolic_frontal_max_separator_cols=separator_cols,
            symbolic_frontal_tail_size=int(tail_size if nonkinetic_tail_is_suffix else 0),
            symbolic_frontal_boundary_width=boundary_width,
            symbolic_frontal_high_degree_cols=high_degree_cols,
            symbolic_frontal_max_superblock_size=max_superblock_size,
            symbolic_frontal_max_superblock_blocks=max_superblock_blocks,
            symbolic_frontal_min_cross_nnz=min_cross_nnz,
            symbolic_frontal_min_cross_separator_fraction=min_cross_separator_fraction,
            symbolic_frontal_regularization_rel=regularization_rel,
            symbolic_frontal_max_dense_rhs_entries=max_dense_rhs_entries,
            symbolic_frontal_max_dense_rhs_cols_per_block=max_dense_rhs_cols_per_block,
            symbolic_nd_max_leaf_size=nd_max_leaf_size,
            symbolic_nd_max_terminal_factor_size=nd_max_terminal_factor_size,
            symbolic_nd_max_depth=nd_max_depth,
            symbolic_nd_separator_width=nd_separator_width,
            symbolic_nd_max_separator_cols=nd_max_separator_cols,
            symbolic_nd_high_degree_cols=nd_high_degree_cols,
            symbolic_nd_regularization_rel=regularization_rel,
            symbolic_nd_max_dense_rhs_entries=nd_max_dense_rhs_entries,
            symbolic_nd_max_dense_rhs_entries_per_child=nd_max_dense_rhs_entries_per_child,
            symbolic_nd_max_dense_rhs_cols_per_child=nd_max_dense_rhs_cols_per_child,
            symbolic_nd_max_setup_s=nd_max_setup_s,
            symbolic_nd_residual_polish_steps=nd_residual_polish_steps,
            symbolic_nd_residual_polish_damping=nd_residual_polish_damping,
        )
    except Exception as exc:  # noqa: BLE001
        root_exc: BaseException = exc
        while getattr(root_exc, "__cause__", None) is not None:
            root_exc = root_exc.__cause__  # type: ignore[assignment]
        error_detail = str(root_exc).strip().replace("\n", " ")[:320]
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason=f"{active_symbolic_kind}_factor_failed:{type(exc).__name__}:{error_detail}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": active_architecture,
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "error": str(exc),
                "symbolic_analysis": analysis.to_dict(),
                "min_cross_separator_fraction": float(min_cross_separator_fraction),
                "dense_rhs_entries_estimate": int(dense_rhs_entries_estimate),
                "max_dense_rhs_entries": int(max_dense_rhs_entries),
                "max_dense_rhs_cols_per_block": int(max_dense_rhs_cols_per_block),
            },
        )
    factor_nbytes = int(factor.factor_nbytes_estimate or 0)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason=f"{active_symbolic_kind}_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": active_architecture,
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        probe_count=int(policy.admission_probes),
        max_relative_residual=float(policy.admission_max_relative_residual),
        min_improvement_vs_identity=float(policy.admission_min_improvement),
    )
    if not bool(admission.accepted):
        admission_reason = (
            f"{active_symbolic_kind}_admission_failed:"
            f"{admission.reason}:"
            f"max_rel={float(admission.max_relative_residual):.3e}:"
            f"median_rel={float(admission.median_relative_residual):.3e}:"
            f"min_improvement={float(admission.min_improvement_vs_identity):.3e}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=active_symbolic_kind,
            reason=admission_reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": active_architecture,
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
                "dense_rhs_entries_estimate": int(dense_rhs_entries_estimate),
                "max_dense_rhs_entries": int(max_dense_rhs_entries),
                "max_dense_rhs_cols_per_block": int(max_dense_rhs_cols_per_block),
                "separator_count": int(getattr(factor.factor, "separator_count", 0)),
                "frontal_block_count": int(getattr(factor.factor, "frontal_block_count", 0)),
                "total_cross_nnz": int(getattr(factor.factor, "total_cross_nnz", 0)),
                "selected_cross_nnz": int(getattr(factor.factor, "selected_cross_nnz", 0)),
                "cross_separator_fraction": float(getattr(factor.factor, "cross_separator_fraction", 0.0)),
                "dense_rhs_entries_actual": int(getattr(factor.factor, "dense_rhs_entries", 0)),
                "peak_dense_rhs_entries_actual": int(getattr(factor.factor, "peak_dense_rhs_entries", 0)),
                "separator_update_columns": int(getattr(factor.factor, "separator_update_columns", 0)),
                "admission": admission.to_dict(),
                "requires_preflight": True,
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        return np.asarray(factor.solve(arr), dtype=np.float64).reshape((-1,))

    inner_factor_metadata = getattr(getattr(factor, "factor", None), "metadata", None)
    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=active_symbolic_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": active_architecture,
            "active_size": int(active_size),
            "matrix_nnz": int(matrix_csr.nnz),
            "tail_size": int(tail_size),
            "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
            "symbolic_analysis": analysis.to_dict(),
            "symbolic_factor_kind": str(factor.kind),
            "separator_count": int(getattr(factor.factor, "separator_count", 0)),
            "frontal_block_count": int(getattr(factor.factor, "frontal_block_count", 0)),
            "total_cross_nnz": int(getattr(factor.factor, "total_cross_nnz", 0)),
            "selected_cross_nnz": int(getattr(factor.factor, "selected_cross_nnz", 0)),
            "cross_separator_fraction": float(getattr(factor.factor, "cross_separator_fraction", 0.0)),
            "dense_rhs_entries_actual": int(getattr(factor.factor, "dense_rhs_entries", 0)),
            "peak_dense_rhs_entries_actual": int(getattr(factor.factor, "peak_dense_rhs_entries", 0)),
            "separator_update_columns": int(getattr(factor.factor, "separator_update_columns", 0)),
            "factor_failures": int(getattr(factor.factor, "factor_failures", 0)),
            "factor_nbytes_estimate": int(raw_estimate),
            "factor_nbytes_prefill_estimate": int(prefill_estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
            "max_factor_nbytes": int(max_factor_nbytes),
            "dense_rhs_entries_estimate": int(dense_rhs_entries_estimate),
            "max_dense_rhs_entries": int(max_dense_rhs_entries),
            "max_dense_rhs_cols_per_block": int(max_dense_rhs_cols_per_block),
            "block_size": int(block_size),
            "ordering_kind": str(ordering_kind),
            "max_separator_cols": int(separator_cols),
            "max_superblock_size": int(max_superblock_size),
            "max_superblock_blocks": int(max_superblock_blocks),
            "min_cross_nnz": int(min_cross_nnz),
            "min_cross_separator_fraction": float(min_cross_separator_fraction),
            "regularization_rel": float(regularization_rel),
            "symbolic_nd_max_leaf_size": int(nd_max_leaf_size),
            "symbolic_nd_max_terminal_factor_size": int(nd_max_terminal_factor_size),
            "symbolic_nd_max_depth": int(nd_max_depth),
            "symbolic_nd_separator_width": int(nd_separator_width),
            "symbolic_nd_max_separator_cols": int(nd_max_separator_cols),
            "symbolic_nd_high_degree_cols": int(nd_high_degree_cols),
            "symbolic_nd_max_dense_rhs_entries": int(nd_max_dense_rhs_entries),
            "symbolic_nd_max_dense_rhs_entries_per_child": int(nd_max_dense_rhs_entries_per_child),
            "symbolic_nd_max_dense_rhs_cols_per_child": int(nd_max_dense_rhs_cols_per_child),
            "symbolic_nd_max_setup_s": float(nd_max_setup_s),
            "symbolic_nd_residual_polish_steps": int(nd_residual_polish_steps),
            "symbolic_nd_residual_polish_damping": float(nd_residual_polish_damping),
            "symbolic_factor_metadata": dict(inner_factor_metadata) if isinstance(inner_factor_metadata, dict) else {},
            "admission": admission.to_dict(),
            "requires_preflight": True,
            "note": "bounded_nd_frontal_schur_reduced_pmat_candidate"
            if bool(use_nd_frontal)
            else "bounded_frontal_schur_reduced_pmat_candidate",
        },
    )


def _build_active_projected_symbolic_superblock_lu_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded grouped-block sparse factor over the active true matrix.

    This candidate keeps the direct reduced-Pmat path native and bounded: it
    reuses a symbolic ordering, merges strongly coupled blocks into capped
    superblocks, factors those submatrices, then requires true-residual probe
    admission before the factor can be used by GMRES.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    from sfincs_jax.explicit_sparse import (  # noqa: PLC0415
        admit_sparse_factor_against_operator,
        analyze_sparse_symbolic_structure,
        factorize_host_sparse_operator,
    )

    matrix_csr = matrix.tocsr()
    active_size = int(matrix_csr.shape[0])
    policy = resolve_active_symbolic_superblock_policy(
        active_size=int(active_size),
        regularization=float(regularization),
    )
    max_active_size = int(policy.max_active_size)
    if int(max_active_size) > 0 and int(active_size) > int(max_active_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason=f"active_symbolic_superblock_lu_size_exceeded:{active_size}>{int(max_active_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(active_size),
                "max_active_size": int(max_active_size),
                "requested_kind": str(requested_kind),
            },
        )

    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (active_size,):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    tail_size, nonkinetic_tail_is_suffix = _active_nonkinetic_tail_size(
        layout=layout,
        active_indices=active_np,
    )

    ordering_kind = str(policy.ordering_kind)
    block_size = int(policy.block_size)
    max_permutation_size = int(policy.max_permutation_size)
    max_superblock_size = int(policy.max_superblock_size)
    max_superblock_blocks = int(policy.max_superblock_blocks)
    min_cross_nnz = int(policy.min_cross_nnz)
    min_retained_cross_fraction = float(policy.min_retained_cross_fraction)
    regularization_rel = float(policy.regularization_rel)
    analysis = analyze_sparse_symbolic_structure(
        matrix_csr,
        ordering_kind=ordering_kind,
        block_size_target=block_size,
        max_permutation_size=max_permutation_size,
    )
    csr_nbytes = int(_scipy_csr_nbytes(matrix_csr))
    group_factor = max(1.0, float(max_superblock_size) / float(max(1, block_size)))
    prefill_safety = float(policy.prefill_safety_factor)
    raw_estimate = int(np.ceil(float(csr_nbytes) * (1.0 + np.sqrt(group_factor))))
    prefill_estimate = int(np.ceil(float(raw_estimate) * float(prefill_safety)))
    if prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason=f"active_symbolic_superblock_lu_prefill_budget_exceeded:{prefill_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_superblock_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "prefill_safety_factor": float(prefill_safety),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    try:
        factor = factorize_host_sparse_operator(
            matrix_csr,
            kind="symbolic_superblock_lu",
            symbolic_analysis=analysis,
            symbolic_block_size=block_size,
            symbolic_superblock_max_size=max_superblock_size,
            symbolic_superblock_max_blocks=max_superblock_blocks,
            symbolic_superblock_min_cross_nnz=min_cross_nnz,
            symbolic_superblock_min_retained_cross_fraction=min_retained_cross_fraction,
            symbolic_superblock_regularization_rel=regularization_rel,
        )
    except Exception as exc:  # noqa: BLE001
        root_exc: BaseException = exc
        while getattr(root_exc, "__cause__", None) is not None:
            root_exc = root_exc.__cause__  # type: ignore[assignment]
        error_detail = str(root_exc).strip().replace("\n", " ")[:320]
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason=f"active_symbolic_superblock_lu_factor_failed:{type(exc).__name__}:{error_detail}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_superblock_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "error": str(exc),
                "symbolic_analysis": analysis.to_dict(),
                "min_retained_cross_fraction": float(min_retained_cross_fraction),
            },
        )
    factor_nbytes = int(factor.factor_nbytes_estimate or 0)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason=f"active_symbolic_superblock_lu_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_superblock_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        probe_count=int(policy.admission_probes),
        max_relative_residual=float(policy.admission_max_relative_residual),
        min_improvement_vs_identity=float(policy.admission_min_improvement),
    )
    if not bool(admission.accepted):
        admission_reason = (
            "active_symbolic_superblock_lu_admission_failed:"
            f"{admission.reason}:"
            f"max_rel={float(admission.max_relative_residual):.3e}:"
            f"median_rel={float(admission.median_relative_residual):.3e}:"
            f"min_improvement={float(admission.min_improvement_vs_identity):.3e}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_superblock_lu",
            reason=admission_reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_superblock_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
                "superblock_count": int(getattr(factor.factor, "superblock_count", 0)),
                "retained_cross_nnz": int(getattr(factor.factor, "retained_cross_nnz", 0)),
                "dropped_cross_nnz": int(getattr(factor.factor, "dropped_cross_nnz", 0)),
                "retained_cross_fraction": float(getattr(factor.factor, "retained_cross_fraction", 0.0)),
                "admission": admission.to_dict(),
                "requires_preflight": True,
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        return np.asarray(factor.solve(arr), dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_symbolic_superblock_lu",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_true_operator_symbolic_superblock_lu",
            "active_size": int(active_size),
            "matrix_nnz": int(matrix_csr.nnz),
            "tail_size": int(tail_size),
            "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
            "symbolic_analysis": analysis.to_dict(),
            "symbolic_factor_kind": str(factor.kind),
            "superblock_count": int(getattr(factor.factor, "superblock_count", 0)),
            "base_block_count": int(getattr(factor.factor, "base_block_count", 0)),
            "max_superblock_size": int(max_superblock_size),
            "max_superblock_blocks": int(max_superblock_blocks),
            "retained_cross_nnz": int(getattr(factor.factor, "retained_cross_nnz", 0)),
            "dropped_cross_nnz": int(getattr(factor.factor, "dropped_cross_nnz", 0)),
            "retained_cross_fraction": float(getattr(factor.factor, "retained_cross_fraction", 0.0)),
            "factor_failures": int(getattr(factor.factor, "factor_failures", 0)),
            "factor_nbytes_estimate": int(raw_estimate),
            "factor_nbytes_prefill_estimate": int(prefill_estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
            "max_factor_nbytes": int(max_factor_nbytes),
            "block_size": int(block_size),
            "ordering_kind": str(ordering_kind),
            "min_cross_nnz": int(min_cross_nnz),
            "min_retained_cross_fraction": float(min_retained_cross_fraction),
            "regularization_rel": float(regularization_rel),
            "admission": admission.to_dict(),
            "requires_preflight": True,
            "note": "bounded_grouped_block_sparse_direct_candidate",
        },
    )


def _build_active_projected_symbolic_block_schur_lu_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded separator-Schur factor over the active true operator.

    This candidate is closer to sparse-direct PETSc/MUMPS/SuperLU behavior than
    the local smoothers: it factors interior blocks, forms an explicit separator
    Schur complement, and admits the factor only when deterministic true-action
    probes pass.  It remains size-gated by default so production-grid auto
    selection can fail closed instead of entering an unbounded host factor.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    from sfincs_jax.explicit_sparse import (  # noqa: PLC0415
        admit_sparse_factor_against_operator,
        analyze_sparse_symbolic_structure,
        factorize_host_sparse_operator,
    )

    matrix_csr = matrix.tocsr()
    active_size = int(matrix_csr.shape[0])
    policy = resolve_active_symbolic_block_schur_policy(regularization=float(regularization))
    max_active_size = int(policy.max_active_size)
    if int(max_active_size) > 0 and int(active_size) > int(max_active_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason=f"active_symbolic_block_schur_lu_size_exceeded:{active_size}>{int(max_active_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(active_size),
                "max_active_size": int(max_active_size),
                "requested_kind": str(requested_kind),
                "note": (
                    "increase SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_ACTIVE_SIZE "
                    "for explicit large separator-Schur probes"
                ),
            },
        )

    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (active_size,):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    tail_size, nonkinetic_tail_is_suffix = _active_nonkinetic_tail_size(
        layout=layout,
        active_indices=active_np,
    )

    ordering_kind = str(policy.ordering_kind)
    block_size = int(policy.block_size)
    max_permutation_size = int(policy.max_permutation_size)
    separator_cols = int(policy.separator_cols)
    boundary_width = int(policy.boundary_width)
    high_degree_cols = int(policy.high_degree_cols)
    regularization_rel = float(policy.regularization_rel)
    analysis = analyze_sparse_symbolic_structure(
        matrix_csr,
        ordering_kind=ordering_kind,
        block_size_target=block_size,
        max_permutation_size=max_permutation_size,
    )
    separator_use = min(int(separator_cols), int(active_size))
    local_dense_estimate = (
        int(analysis.block_count)
        * int(max(1, analysis.block_size_max))
        * int(max(1, analysis.block_size_max))
        * np.dtype(np.float64).itemsize
    )
    separator_dense_estimate = int(separator_use * separator_use * np.dtype(np.float64).itemsize)
    csr_nbytes = int(_scipy_csr_nbytes(matrix_csr))
    raw_estimate = int(csr_nbytes + 2 * local_dense_estimate + 3 * separator_dense_estimate)
    prefill_safety = float(policy.prefill_safety_factor)
    prefill_estimate = int(np.ceil(float(raw_estimate) * float(prefill_safety)))
    if prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason=f"active_symbolic_block_schur_lu_prefill_budget_exceeded:{prefill_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_separator_schur_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "prefill_safety_factor": float(prefill_safety),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    try:
        factor = factorize_host_sparse_operator(
            matrix_csr,
            kind="symbolic_block_schur_lu",
            symbolic_analysis=analysis,
            symbolic_block_size=block_size,
            symbolic_schur_tail_size=int(tail_size if nonkinetic_tail_is_suffix else 0),
            symbolic_schur_max_separator_cols=separator_cols,
            symbolic_schur_boundary_width=boundary_width,
            symbolic_schur_high_degree_cols=high_degree_cols,
            symbolic_schur_regularization_rel=regularization_rel,
        )
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason=f"active_symbolic_block_schur_lu_factor_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_separator_schur_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "error": str(exc),
                "symbolic_analysis": analysis.to_dict(),
            },
        )
    factor_nbytes = int(factor.factor_nbytes_estimate or 0)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason=f"active_symbolic_block_schur_lu_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_separator_schur_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    admission = admit_sparse_factor_against_operator(
        factor.operator,
        factor,
        probe_count=int(policy.admission_probes),
        max_relative_residual=float(policy.admission_max_relative_residual),
        min_improvement_vs_identity=float(policy.admission_min_improvement),
    )
    if not bool(admission.accepted):
        admission_reason = (
            "active_symbolic_block_schur_lu_admission_failed:"
            f"{admission.reason}:"
            f"max_rel={float(admission.max_relative_residual):.3e}:"
            f"median_rel={float(admission.median_relative_residual):.3e}:"
            f"min_improvement={float(admission.min_improvement_vs_identity):.3e}"
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_block_schur_lu",
            reason=admission_reason,
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "architecture": "active_true_operator_symbolic_separator_schur_lu",
                "active_size": int(active_size),
                "matrix_nnz": int(matrix_csr.nnz),
                "tail_size": int(tail_size),
                "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
                "symbolic_analysis": analysis.to_dict(),
                "factor_nbytes_estimate": int(raw_estimate),
                "factor_nbytes_prefill_estimate": int(prefill_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
                "max_factor_nbytes": int(max_factor_nbytes),
                "admission": admission.to_dict(),
                "requires_preflight": True,
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        return np.asarray(factor.solve(arr), dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_symbolic_block_schur_lu",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_true_operator_symbolic_separator_schur_lu",
            "active_size": int(active_size),
            "matrix_nnz": int(matrix_csr.nnz),
            "tail_size": int(tail_size),
            "nonkinetic_tail_is_suffix": bool(nonkinetic_tail_is_suffix),
            "symbolic_analysis": analysis.to_dict(),
            "symbolic_factor_kind": str(factor.kind),
            "separator_size": int(getattr(factor.factor, "coarse_size", 0)),
            "factor_nbytes_estimate": int(raw_estimate),
            "factor_nbytes_prefill_estimate": int(prefill_estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "factor_nnz_actual": int(factor.factor_nnz_estimate or 0),
            "max_factor_nbytes": int(max_factor_nbytes),
            "block_size": int(block_size),
            "ordering_kind": str(ordering_kind),
            "max_separator_cols": int(separator_cols),
            "boundary_width": int(boundary_width),
            "high_degree_cols": int(high_degree_cols),
            "regularization_rel": float(regularization_rel),
            "admission": admission.to_dict(),
            "requires_preflight": True,
            "note": "explicit_separator_schur_probe_not_production_default_until_full_grid_gate_passes",
        },
    )


def _build_active_projected_symbolic_coupled_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a bounded active symbolic field-split Schur residual preconditioner.

    This explicit candidate combines a native multiline kinetic base with a
    true least-squares Schur residual equation over three spaces:

    - dominant active kinetic identity columns selected by symbolic ordering,
    - flux-surface/profile/current moment columns from the existing coarse
      basis builder,
    - and optional adaptive residual columns derived from the composed base.

    It is intentionally marked as requiring preflight before GMRES.  The
    production QA/QH gate is the true residual after one application, not the
    mere ability to build the coarse equation.
    """

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_size = int(matrix_csr.shape[0])
    max_active_size = int(
        _env_int(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_ACTIVE_SIZE",
            300_000,
        )
    )
    if int(max_active_size) > 0 and int(active_size) > int(max_active_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason=f"active_symbolic_coupled_schur_size_exceeded:{active_size}>{int(max_active_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(active_size),
                "max_active_size": int(max_active_size),
                "requested_kind": str(requested_kind),
                "note": (
                    "increase SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_ACTIVE_SIZE "
                    "for explicit large probes"
                ),
            },
        )
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )
    try:
        ordering = RHS1ActiveFieldSplitOrdering.cached_from_layout(layout, active_np)
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason="invalid_symbolic_ordering",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    base_budget_fraction = float(
        _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_BASE_BUDGET_FRACTION", 0.55)
    )
    base_budget_fraction = min(max(float(base_budget_fraction), 0.05), 1.0)
    base_budget = max(1, int(float(max_factor_nbytes) * base_budget_fraction))
    base_kind = (
        os.environ.get(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_BASE",
            "active_multiline_field_split_base",
        )
        .strip()
        .lower()
        .replace("-", "_")
    )
    if base_kind in {"", "default", "auto"}:
        base_kind = "active_multiline_field_split_base"
    if base_kind in {"zero", "none", "coarse_only", "symbolic_coarse_only"}:
        def zero_apply(x: Any) -> np.ndarray:
            arr = np.asarray(x, dtype=np.float64).reshape((-1,))
            return np.zeros_like(arr, dtype=np.float64)

        base = RHS1StructuredFullCSRPreconditioner(
            operator=LinearOperator(matrix_csr.shape, matvec=zero_apply, dtype=np.float64),
            selected=True,
            kind="zero_symbolic_schur_base",
            reason="complete",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "kind": "zero_symbolic_schur_base",
                "factor_nbytes_actual": 0,
                "factor_nbytes_estimate": 0,
                "note": "coarse_residual_equation_applied_without_native_base",
            },
        )
    elif base_kind == "active_multiline_field_split_base":
        base = _build_active_projected_multiline_field_split_base_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            regularization=regularization,
            max_factor_nbytes=base_budget,
            t0=t0,
            base_preconditioner_factory=build_active_projected_rhs1_full_csr_preconditioner,
        )
    else:
        base = build_active_projected_rhs1_full_csr_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            kind=base_kind,
            max_factor_nbytes=base_budget,
            regularization=regularization,
        )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason=f"symbolic_coupled_schur_base_failed:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "symbolic_ordering": ordering.to_dict(),
            },
        )

    max_coarse_size = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_COARSE_SIZE", 2048)),
    )
    max_identity_columns = max(
        0,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_IDENTITY_COLUMNS",
                min(2048, int(max_coarse_size)),
            )
        ),
    )
    identity_basis, identity_metadata = _build_active_symbolic_dominant_kinetic_basis_csc(
        ordering=ordering,
        max_columns=min(int(max_identity_columns), int(max_coarse_size)),
    )
    config = _coarse_residual_config(layout)
    remaining_after_identity = max(0, int(max_coarse_size) - int(identity_basis.shape[1]))
    if bool(identity_metadata.get("symbolic_identity_basis_covers_active", False)):
        moment_basis = sp.csc_matrix((int(matrix_csr.shape[0]), 0), dtype=np.float64)
    else:
        moment_basis_full = _build_coarse_residual_basis_csc(layout=layout, config=config)
        moment_basis = moment_basis_full[active_np, :].tocsc()
        if int(moment_basis.shape[1]) > int(remaining_after_identity):
            moment_basis = moment_basis[:, : int(remaining_after_identity)].tocsc()
    basis_parts = [part for part in (identity_basis, moment_basis) if int(part.shape[1]) > 0]
    if not basis_parts:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason="empty_symbolic_coupled_schur_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "symbolic_ordering": ordering.to_dict(),
                **identity_metadata,
                **config,
            },
        )
    basis = sp.hstack(basis_parts, format="csc")
    basis.sum_duplicates()
    basis.eliminate_zeros()
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    keep = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if keep.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason="zero_symbolic_coupled_schur_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "symbolic_ordering": ordering.to_dict(),
                **identity_metadata,
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
            kind="active_symbolic_coupled_schur",
            reason="invalid_symbolic_coupled_schur_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "symbolic_ordering": ordering.to_dict(),
                **identity_metadata,
                **config,
            },
        )
    basis = (basis[:, valid].tocsc() @ sp.diags(1.0 / col_norm[valid], format="csc")).tocsc()
    basis.eliminate_zeros()
    basis, adaptive_metadata = _append_adaptive_residual_basis_csc(
        matrix=matrix_csr,
        base_operator=base.operator,
        basis=basis,
        max_total_columns=int(max_coarse_size),
    )

    az_basis = (matrix_csr @ basis).tocsc()
    coarse = np.asarray((az_basis.T @ az_basis).toarray(), dtype=np.float64)
    coarse_size = int(coarse.shape[0])
    coarse_scale = max(float(np.linalg.norm(coarse, ord=np.inf)) if coarse.size else 0.0, 1.0)
    coarse_regularization = max(float(abs(regularization)), 1.0e-14) * coarse_scale
    coarse_reg = coarse + coarse_regularization * np.eye(coarse_size, dtype=np.float64)
    base_nbytes = int(base.metadata.get("factor_nbytes_actual", base.metadata.get("factor_nbytes_estimate", 0)) or 0)
    basis_nbytes = int(_scipy_csr_nbytes(basis.tocsr()))
    az_basis_nbytes = int(_scipy_csr_nbytes(az_basis.tocsr()))
    coarse_nbytes = int(coarse_reg.nbytes)
    total_nbytes = int(base_nbytes + basis_nbytes + az_basis_nbytes + coarse_nbytes)
    if total_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_symbolic_coupled_schur",
            reason=f"active_symbolic_coupled_schur_budget_exceeded:{total_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_base_kind": str(base_kind),
                "base_preconditioner": base.to_dict(),
                "symbolic_ordering": ordering.to_dict(),
                "coarse_size": int(coarse_size),
                "basis_nbytes_actual": int(basis_nbytes),
                "az_basis_nbytes_actual": int(az_basis_nbytes),
                "coarse_nbytes_actual": int(coarse_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
                "factor_nbytes_actual": int(total_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                **identity_metadata,
                **adaptive_metadata,
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

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y, dtype=np.float64).reshape((-1,))
        coarse_rhs = np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        coeff = solve_coarse(coarse_rhs)
        return y + np.asarray(basis @ coeff, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_symbolic_coupled_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": "active_symbolic_native_base_true_lsq_schur",
            "requested_base_kind": str(base_kind),
            "base_kind": str(base.kind),
            "base_preconditioner": base.to_dict(),
            "symbolic_ordering": ordering.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_size),
            "max_coarse_size": int(max_coarse_size),
            "projected_basis_nnz": int(basis.nnz),
            "az_basis_nnz": int(az_basis.nnz),
            "coarse_solver": str(solver_kind),
            "coarse_equation": "least_squares_true_action",
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": coarse_condition,
            "basis_nbytes_actual": int(basis_nbytes),
            "az_basis_nbytes_actual": int(az_basis_nbytes),
            "coarse_nbytes_actual": int(coarse_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "factor_nbytes_actual": int(total_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "requires_preflight": True,
            "note": "explicit_symbolic_coupled_schur_probe_not_auto_default",
            **identity_metadata,
            **adaptive_metadata,
            **config,
        },
    )


def _build_active_projected_native_xell_field_split_sparse_coarse_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a native line-factor/field-split preconditioner with true coarse correction.

    The base preconditioner is a two-level field split:

    ``K^{-1}`` is the memory-bounded active native ``(x, ell)`` line inverse,
    and the compact current/constraint tail is handled through the exact Schur
    residual equation ``S = W - V K^{-1} U``.  This helper adds one more
    physically structured residual equation on top of that base using the true
    active CSR operator.  It is the first production-oriented path that combines
    low-memory line factors with a global moment/low-mode coarse correction.
    """

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    policy = resolve_active_native_field_split_sparse_coarse_policy(requested_kind=str(requested_kind))
    is_multiline = bool(policy.is_multiline)
    is_coupled_kinetic = bool(policy.is_coupled_kinetic)
    output_kind = str(policy.output_kind)
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    requested_base_kind = str(policy.requested_base_kind)
    if bool(is_coupled_kinetic):
        base = _build_active_projected_coupled_kinetic_block_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            requested_kind="active_coupled_kinetic_block",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    elif is_multiline:
        base = _build_active_projected_multiline_field_split_base_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_indices,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    else:
        base = _build_active_projected_global_field_split_schur_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_indices,
            requested_kind="active_global_field_split_schur",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
            base_preconditioner_factory=build_active_projected_rhs1_full_csr_preconditioner,
            base_kind_override=requested_base_kind,
        )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict()},
        )

    config = _coarse_residual_config(layout)
    max_coarse_size = int(policy.max_coarse_size)
    window_basis, window_metadata = _build_active_native_xell_coarse_window_basis_csc(layout=layout)
    physics_basis = _build_coarse_residual_basis_csc(layout=layout, config=config)
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
            kind=output_kind,
            reason="empty_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **window_metadata, **config},
        )
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    keep = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if keep.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="zero_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **window_metadata, **config},
        )
    if keep.size > max_coarse_size:
        keep = keep[:max_coarse_size]
    basis = basis[:, keep].tocsc()
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    valid = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if valid.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="invalid_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **window_metadata, **config},
        )
    basis = basis[:, valid].tocsc()
    col_norm = col_norm[valid]
    basis = (basis @ sp.diags(1.0 / col_norm, format="csc")).tocsc()
    basis.eliminate_zeros()
    basis, adaptive_metadata = _append_adaptive_residual_basis_csc(
        matrix=matrix_csr,
        base_operator=base.operator,
        basis=basis,
        max_total_columns=max_coarse_size,
    )
    basis, probe_residual_metadata = _append_probe_residual_basis_csc(
        matrix=matrix_csr,
        base_operator=base.operator,
        basis=basis,
        max_total_columns=max_coarse_size,
        enabled_default=bool(is_coupled_kinetic),
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
    base_nbytes = int(base.metadata.get("factor_nbytes_actual", 0) or 0)
    total_nbytes = int(base_nbytes + basis_nbytes + az_basis_nbytes + coarse_nbytes)
    if total_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason=f"{output_kind}_budget_exceeded:{total_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "base_preconditioner": base.to_dict(),
                "coarse_size": int(coarse_size),
                "basis_nbytes_actual": int(basis_nbytes),
                "az_basis_nbytes_actual": int(az_basis_nbytes),
                "coarse_nbytes_actual": int(coarse_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
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

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        coarse_rhs = (
            np.asarray(basis.T @ residual, dtype=np.float64).reshape((-1,))
            if coarse_solver_mode == "galerkin"
            else np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        )
        coeff = solve_coarse(coarse_rhs)
        return y_base + np.asarray(basis @ coeff, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    admission_metadata: dict[str, object] = {"admission_enabled": bool(is_coupled_kinetic), "accepted": True}
    if bool(is_coupled_kinetic):
        accepted, admission_metadata = _admit_linear_operator_against_matrix(
            matrix=matrix_csr,
            operator=operator,
            probe_count=int(policy.admission_probes),
            max_relative_residual=float(policy.admission_max_relative_residual),
            min_improvement_vs_identity=float(policy.admission_min_improvement),
        )
        admission_metadata["admission_enabled"] = True
        if not bool(accepted):
            reason = (
                "active_coupled_kinetic_sparse_coarse_admission_failed:"
                f"{admission_metadata.get('reason', 'unknown')}:"
                f"max_rel={float(admission_metadata.get('max_relative_residual', float('inf'))):.3e}:"
                f"median_rel={float(admission_metadata.get('median_relative_residual', float('inf'))):.3e}:"
                f"min_improvement={float(admission_metadata.get('min_improvement_vs_identity', 0.0)):.3e}"
            )
            return RHS1StructuredFullCSRPreconditioner(
                operator=None,
                selected=False,
                kind=output_kind,
                reason=reason,
                setup_s=max(0.0, time.perf_counter() - t0),
                metadata={
                    "requested_kind": str(requested_kind),
                    "architecture": "active_coupled_kinetic_true_action_sparse_coarse",
                    "base_kind": str(base.kind),
                    "requested_base_kind": str(requested_base_kind),
                    "base_preconditioner": base.to_dict(),
                    "active_size": int(matrix_csr.shape[0]),
                    "coarse_size": int(coarse_size),
                    "max_coarse_size": int(max_coarse_size),
                    "projected_basis_nnz": int(basis.nnz),
                    "az_basis_nnz": int(az_basis.nnz),
                    "coarse_solver": str(solver_kind),
                    "coarse_equation": str(coarse_solver_mode),
                    "admission": dict(admission_metadata),
                    "factor_nbytes_actual": int(total_nbytes),
                    "max_factor_nbytes": int(max_factor_nbytes),
                    "requires_preflight": True,
                    **adaptive_metadata,
                    **probe_residual_metadata,
                    **window_metadata,
                    **config,
                },
            )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=output_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "architecture": (
                "active_coupled_kinetic_true_action_sparse_coarse"
                if bool(is_coupled_kinetic)
                else (
                    "active_multiline_xell_angular_global_field_split_sparse_coarse"
                    if is_multiline
                    else (
                        "active_angular_line_global_field_split_sparse_coarse"
                        if str(requested_base_kind) == "active_angular_line"
                        else "active_native_xell_global_field_split_sparse_coarse"
                    )
                )
            ),
            "base_kind": str(base.kind),
            "requested_base_kind": str(requested_base_kind),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_size),
            "max_coarse_size": int(max_coarse_size),
            "projected_basis_nnz": int(basis.nnz),
            "az_basis_nnz": int(az_basis.nnz),
            "coarse_solver": str(solver_kind),
            "coarse_equation": str(coarse_solver_mode),
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": coarse_condition,
            "basis_nbytes_actual": int(basis_nbytes),
            "az_basis_nbytes_actual": int(az_basis_nbytes),
            "coarse_nbytes_actual": int(coarse_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "factor_nbytes_actual": int(total_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "admission": dict(admission_metadata),
            "requires_preflight": bool(is_coupled_kinetic),
            "note": "opt_in_probe_not_auto_default",
            **adaptive_metadata,
            **probe_residual_metadata,
            **window_metadata,
            **config,
        },
    )


def _build_active_projected_sparse_coarse_residual_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a sparse physics coarse correction on top of active tail Schur.

    This keeps the projected basis and ``A @ basis`` sparse. Only the coarse
    matrix itself is dense, so production-size active systems can be rejected or
    tested by memory budget before any long Krylov iteration.
    """

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    if active_indices is None:
        active_np = np.arange(int(matrix_csr.shape[0]), dtype=np.int64)
    else:
        active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_tail_sparse_coarse",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    policy = resolve_active_sparse_coarse_residual_policy(requested_kind=str(requested_kind))
    base_kind = str(policy.base_kind)
    output_kind = str(policy.output_kind)
    base = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_np,
        kind=base_kind,
        max_factor_nbytes=max_factor_nbytes,
        regularization=regularization,
    )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict()},
        )

    config = _coarse_residual_config(layout)
    max_coarse_size = int(policy.max_coarse_size)
    full_basis = _build_coarse_residual_basis_csc(layout=layout, config=config)
    basis = full_basis[active_np, :].tocsc()
    if basis.shape[1] == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="empty_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    keep = np.flatnonzero(col_norm > 0.0)
    if keep.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="zero_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    if keep.size > max_coarse_size:
        keep = keep[:max_coarse_size]
    basis = basis[:, keep].tocsc()
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    valid = np.flatnonzero(np.isfinite(col_norm) & (col_norm > 0.0))
    if valid.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason="invalid_projected_sparse_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    basis = basis[:, valid].tocsc()
    col_norm = col_norm[valid]
    basis = (basis @ sp.diags(1.0 / col_norm, format="csc")).tocsc()
    basis.eliminate_zeros()

    az_basis = (matrix_csr @ basis).tocsc()
    coarse_solver_mode = str(policy.coarse_solver_mode)
    if coarse_solver_mode == "least_squares":
        coarse = np.asarray((az_basis.T @ az_basis).toarray(), dtype=np.float64)
    else:
        coarse = np.asarray((basis.T @ az_basis).toarray(), dtype=np.float64)
    coarse_size = int(coarse.shape[0])
    coarse_scale = max(float(np.linalg.norm(coarse, ord=np.inf)) if coarse.size else 0.0, 1.0)
    coarse_regularization = max(float(abs(regularization)), 1.0e-14) * coarse_scale
    coarse_reg = coarse + coarse_regularization * np.eye(coarse_size, dtype=np.float64)
    basis_nbytes = int(_scipy_csr_nbytes(basis.tocsr()))
    az_basis_nbytes = int(_scipy_csr_nbytes(az_basis.tocsr()))
    coarse_nbytes = int(coarse_reg.nbytes)
    base_nbytes = int(base.metadata.get("factor_nbytes_actual", 0) or 0)
    total_nbytes = int(base_nbytes + basis_nbytes + az_basis_nbytes + coarse_nbytes)
    if total_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=output_kind,
            reason=f"active_sparse_coarse_budget_exceeded:{total_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "base_preconditioner": base.to_dict(),
                "coarse_size": int(coarse_size),
                "basis_nbytes_actual": int(basis_nbytes),
                "az_basis_nbytes_actual": int(az_basis_nbytes),
                "coarse_nbytes_actual": int(coarse_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
                "factor_nbytes_actual": int(total_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                **config,
            },
        )

    solver_kind = "lu"
    try:
        lu, piv = lu_factor(coarse_reg)

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs), dtype=np.float64).reshape((-1,))

        coarse_condition = float(np.linalg.cond(coarse_reg)) if coarse_size <= 512 else None
    except Exception:  # noqa: BLE001
        solver_kind = "pinv"
        pinv = np.linalg.pinv(coarse_reg, rcond=max(float(abs(regularization)), 1.0e-14))

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(pinv @ rhs, dtype=np.float64).reshape((-1,))

        coarse_condition = None

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        if coarse_solver_mode == "least_squares":
            coarse_rhs = np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        else:
            coarse_rhs = np.asarray(basis.T @ residual, dtype=np.float64).reshape((-1,))
        coeff = solve_coarse(coarse_rhs)
        return y_base + np.asarray(basis @ coeff, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=output_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "requested_base_kind": str(base_kind),
            "architecture": (
                "scaled_ilu_global_sparse_coarse"
                if str(base_kind) == "active_scaled_ilu"
                else (
                    "additive_schwarz_global_sparse_coarse"
                    if str(base_kind) == "active_overlap_schwarz"
                    else (
                        "filtered_sparse_factor_global_sparse_coarse"
                        if str(base_kind) == "active_filtered_sparse_factor"
                        else "field_split_global_sparse_coarse"
                    )
                )
            ),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_size),
            "max_coarse_size": int(max_coarse_size),
            "projected_basis_nnz": int(basis.nnz),
            "az_basis_nnz": int(az_basis.nnz),
            "coarse_solver": str(solver_kind),
            "coarse_equation": str(coarse_solver_mode),
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": coarse_condition,
            "basis_nbytes_actual": int(basis_nbytes),
            "az_basis_nbytes_actual": int(az_basis_nbytes),
            "coarse_nbytes_actual": int(coarse_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "factor_nbytes_actual": int(total_nbytes),
            "requires_preflight": bool(str(base_kind) == "active_filtered_sparse_factor"),
            **config,
        },
    )


def _build_active_projected_coarse_residual_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    base_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a physics coarse residual correction for an active projected matrix."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

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
            kind="active_coarse",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    if str(base_kind) == "active_ilu":
        base = build_active_projected_rhs1_full_csr_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            kind="active_ilu",
            max_factor_nbytes=max_factor_nbytes,
            regularization=regularization,
        )
    else:
        base = _build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=str(requested_kind),
            regularization=regularization,
            t0=t0,
            reason="active_coarse_base_jacobi",
        )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coarse",
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict()},
        )

    config = _coarse_residual_config(layout)
    max_coarse_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COARSE_MAX_SIZE", 640)
    max_coarse_size = max(1, int(max_coarse_size))
    full_basis = _build_coarse_residual_basis_csc(layout=layout, config=config)
    basis = full_basis[active_np, :].tocsc()
    if basis.shape[1] == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coarse",
            reason="empty_projected_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    col_norm = np.sqrt(np.asarray(basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    keep = np.flatnonzero(col_norm > 0.0)
    if keep.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coarse",
            reason="zero_projected_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    if keep.size > max_coarse_size:
        keep = keep[:max_coarse_size]
    basis = basis[:, keep].tocsc()
    z_basis = np.asarray(basis.toarray(), dtype=np.float64)
    z_norm = np.linalg.norm(z_basis, axis=0)
    valid = np.flatnonzero(np.isfinite(z_norm) & (z_norm > 0.0))
    if valid.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coarse",
            reason="invalid_projected_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    z_basis = z_basis[:, valid] / z_norm[valid][None, :]
    az_basis = np.asarray(matrix_csr @ z_basis, dtype=np.float64)
    default_coarse_solver = (
        "least_squares"
        if str(requested_kind) in {"active_coarse_ls", "active_least_squares_coarse"}
        else "galerkin"
    )
    coarse_solver_mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COARSE_SOLVER", default_coarse_solver)
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver_mode in {"galerkin", "petrov_galerkin", "ztaz"}:
        coarse_solver_mode = "galerkin"
        coarse = np.asarray(z_basis.T @ az_basis, dtype=np.float64)
    else:
        coarse_solver_mode = "least_squares"
        coarse = np.asarray(az_basis.T @ az_basis, dtype=np.float64)
    coarse_size = int(coarse.shape[0])
    coarse_scale = max(float(np.linalg.norm(coarse, ord=np.inf)) if coarse.size else 0.0, 1.0)
    coarse_regularization = max(float(abs(regularization)), 1.0e-14) * coarse_scale
    coarse_reg = coarse + coarse_regularization * np.eye(coarse_size, dtype=np.float64)
    coarse_nbytes = int(z_basis.nbytes + az_basis.nbytes + coarse_reg.nbytes)
    base_nbytes = int(base.metadata.get("factor_nbytes_actual", 0) or 0)
    total_nbytes = int(base_nbytes + coarse_nbytes)
    if total_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_coarse",
            reason=f"active_coarse_budget_exceeded:{total_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "base_preconditioner": base.to_dict(),
                "coarse_size": int(coarse_size),
                "coarse_nbytes_actual": int(coarse_nbytes),
                "base_factor_nbytes_actual": int(base_nbytes),
                "factor_nbytes_actual": int(total_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                **config,
            },
        )
    solver_kind = "lu"
    try:
        lu, piv = lu_factor(coarse_reg)

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs), dtype=np.float64).reshape((-1,))

        coarse_condition = float(np.linalg.cond(coarse_reg)) if coarse_size <= 512 else None
    except Exception:  # noqa: BLE001
        solver_kind = "pinv"
        pinv = np.linalg.pinv(coarse_reg, rcond=max(float(abs(regularization)), 1.0e-14))

        def solve_coarse(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(pinv @ rhs, dtype=np.float64).reshape((-1,))

        coarse_condition = None

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        coarse_rhs = (
            np.asarray(z_basis.T @ residual, dtype=np.float64).reshape((-1,))
            if coarse_solver_mode == "galerkin"
            else np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        )
        coeff = solve_coarse(coarse_rhs)
        return y_base + np.asarray(z_basis @ coeff, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_coarse",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "requested_base_kind": str(base_kind),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_size),
            "max_coarse_size": int(max_coarse_size),
            "projected_basis_nnz": int(basis.nnz),
            "coarse_solver": str(solver_kind),
            "coarse_equation": str(coarse_solver_mode),
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": coarse_condition,
            "coarse_nbytes_actual": int(coarse_nbytes),
            "base_factor_nbytes_actual": int(base_nbytes),
            "factor_nbytes_actual": int(total_nbytes),
            **config,
        },
    )


def _build_active_projected_low_l_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    base_kind: str | None,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a sparse Schur residual correction over low-pitch active variables.

    This is a host-side field split for singular/near-singular physical
    RHSMode=1 systems: Jacobi handles the full active system cheaply, then a
    sparse exact residual solve over all theta/zeta points for the first
    Legendre modes plus the global tail closes the slow current/constraint
    error.  Unlike the modal coarse basis, this uses a selected sparse submatrix
    instead of a dense Galerkin matrix, so it can be memory-gated before use.
    """

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
            kind="active_low_l_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_LMAX", min(4, int(layout.n_xi)))
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    coarse_full = _active_low_l_schur_full_indices(layout=layout, lmax=lmax)
    coarse_positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=coarse_full)
    if coarse_positions.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_low_l_schur",
            reason="empty_active_low_l_schur_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"lmax": int(lmax)},
        )

    requested_base = (
        str(base_kind)
        if base_kind is not None
        else os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_BASE", "jacobi")
    )
    requested_base = requested_base.strip().lower().replace("-", "_")
    if requested_base in {
        "xblock",
        "x_block",
        "active_xblock",
        "active_x_block",
        "active_xblock_ilu",
        "active_xblock_spilu",
        "active_block_asm",
        "active_block_asm_ilu",
    }:
        base_requested_kind = (
            "active_xblock_ilu"
            if "ilu" in requested_base or "spilu" in requested_base or "block_asm" in requested_base
            else "active_xblock"
        )
        base = _build_active_projected_xblock_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            requested_kind=base_requested_kind,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
        base_kind_used = str(base_requested_kind)
    elif requested_base in {
        "overlap_schwarz",
        "active_overlap_schwarz",
        "additive_schwarz",
        "restricted_schwarz",
        "ras",
        "active_ras",
    }:
        base = _build_active_projected_overlap_schwarz_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            requested_kind="active_overlap_schwarz",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
        base_kind_used = "active_overlap_schwarz"
    else:
        base = _build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=str(requested_kind),
            regularization=regularization,
            t0=t0,
            reason="active_low_l_schur_base_jacobi",
        )
        base_kind_used = "jacobi"
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_low_l_schur",
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), "lmax": int(lmax)},
        )
    base_factor_nbytes = int(base.metadata.get("factor_nbytes_actual", 0) or 0)

    coarse_matrix = matrix_csr[coarse_positions[:, None], coarse_positions].tocsc()
    coarse_scale = max(float(np.max(np.abs(coarse_matrix.data))) if coarse_matrix.nnz else 0.0, 1.0)
    diagonal_shift = max(float(abs(regularization)), 1.0e-14) * coarse_scale
    if diagonal_shift > 0.0:
        coarse_matrix = coarse_matrix + diagonal_shift * sp.eye(coarse_matrix.shape[0], dtype=np.float64, format="csc")
    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FACTOR_KIND", "").strip().lower()
    if factor_kind_env in {"ilu", "spilu", "incomplete_lu"}:
        factor_kind = "spilu"
    elif factor_kind_env in {"lu", "splu", "exact", "direct"}:
        factor_kind = "splu"
    elif "ilu" in requested or "spilu" in requested:
        factor_kind = "spilu"
    else:
        factor_kind = "splu"
    fill_estimate = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FILL_ESTIMATE", 8.0))
    fill_factor_default = min(float(fill_estimate), 4.0) if factor_kind == "spilu" else float(fill_estimate)
    fill_factor = max(
        1.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FILL_FACTOR", fill_factor_default)),
    )
    drop_tol_default = 3.0e-3 if factor_kind == "spilu" else 0.0
    drop_tol = max(0.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_DROP_TOL", drop_tol_default)))
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_DIAG_PIVOT_THRESH", 0.0))
    damping = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_DAMPING", 1.0))
    damping = max(0.0, min(float(damping), 2.0))
    scale_coarse = _env_bool(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_SCALE",
        factor_kind == "spilu" or "ilu" in requested,
    )
    scale_norm = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_SCALE_NORM", "l1").strip().lower()
    if scale_norm not in {"l1", "l2", "max"}:
        scale_norm = "l1"
    max_scale = max(1.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_MAX_SCALE", 1.0e6)))
    row_scale = np.ones((int(coarse_matrix.shape[0]),), dtype=np.float64)
    col_scale = np.ones((int(coarse_matrix.shape[1]),), dtype=np.float64)
    row_scale_meta: dict[str, object] = {"enabled": False, "scale_min": 1.0, "scale_max": 1.0}
    col_scale_meta: dict[str, object] = {"enabled": False, "scale_min": 1.0, "scale_max": 1.0}
    scale_nbytes = 0
    if bool(scale_coarse):
        row_scale, row_scale_meta = _sparse_equilibration_scale(
            coarse_matrix,
            axis=1,
            norm=scale_norm,
            max_scale=max_scale,
        )
        coarse_row_scaled = coarse_matrix.multiply(row_scale[:, None]).tocsc()
        col_scale, col_scale_meta = _sparse_equilibration_scale(
            coarse_row_scaled,
            axis=0,
            norm=scale_norm,
            max_scale=max_scale,
        )
        coarse_matrix = coarse_row_scaled.multiply(col_scale[None, :]).tocsc()
        scale_nbytes = int(row_scale.nbytes + col_scale.nbytes)
    sparse_factor_estimate = _estimate_spilu_factor_nbytes(matrix=coarse_matrix, fill_factor=fill_factor)
    dense_factor_estimate = int(coarse_matrix.shape[0] * coarse_matrix.shape[1] * np.dtype(np.float64).itemsize)
    low_l_factor_estimate = (
        int(sparse_factor_estimate)
        if factor_kind == "spilu"
        else max(int(sparse_factor_estimate), int(dense_factor_estimate))
    )
    factor_estimate = int(base_factor_nbytes + low_l_factor_estimate + scale_nbytes)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_low_l_schur",
            reason=f"active_low_l_schur_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(matrix_csr.shape[0]),
                "coarse_size": int(coarse_matrix.shape[0]),
                "coarse_matrix_nnz": int(coarse_matrix.nnz),
                "coarse_matrix_nbytes_actual": int(_scipy_csr_nbytes(coarse_matrix.tocsr())),
                "factor_nbytes_estimate": int(factor_estimate),
                "low_l_factor_nbytes_estimate": int(low_l_factor_estimate),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
                "dense_factor_nbytes_estimate": int(dense_factor_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "lmax": int(lmax),
                "fill_estimate": float(fill_estimate),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "factor_kind": str(factor_kind),
                "damping": float(damping),
                "coarse_scaling_enabled": bool(scale_coarse),
                "coarse_scale_nbytes_estimate": int(scale_nbytes),
                "row_scaling": row_scale_meta,
                "column_scaling": col_scale_meta,
            },
        )

    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    try:
        if factor_kind == "spilu":
            factor = spilu(
                coarse_matrix,
                drop_tol=float(drop_tol),
                fill_factor=float(fill_factor),
                permc_spec=str(permc_spec),
                diag_pivot_thresh=float(diag_pivot),
            )
        else:
            factor = splu(coarse_matrix, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot))
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_low_l_schur",
            reason=f"active_low_l_schur_factor_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "error": str(exc),
                "factor_nbytes_estimate": int(factor_estimate),
                "low_l_factor_nbytes_estimate": int(low_l_factor_estimate),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "lmax": int(lmax),
                "factor_kind": str(factor_kind),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "diag_pivot_thresh": float(diag_pivot),
                "damping": float(damping),
                "coarse_scaling_enabled": bool(scale_coarse),
                "row_scaling": row_scale_meta,
                "column_scaling": col_scale_meta,
            },
        )
    low_l_factor_nbytes = int(_sparse_lu_factor_nbytes(factor) + scale_nbytes)
    factor_nbytes = int(base_factor_nbytes + low_l_factor_nbytes)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_low_l_schur",
            reason=f"active_low_l_schur_factor_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(matrix_csr.shape[0]),
                "coarse_size": int(coarse_matrix.shape[0]),
                "coarse_matrix_nnz": int(coarse_matrix.nnz),
                "coarse_matrix_nbytes_actual": int(_scipy_csr_nbytes(coarse_matrix.tocsr())),
                "factor_nbytes_estimate": int(factor_estimate),
                "low_l_factor_nbytes_estimate": int(low_l_factor_estimate),
                "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
                "dense_factor_nbytes_estimate": int(dense_factor_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "low_l_factor_nbytes_actual": int(low_l_factor_nbytes),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "lmax": int(lmax),
                "fill_estimate": float(fill_estimate),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "factor_kind": str(factor_kind),
                "damping": float(damping),
                "coarse_scaling_enabled": bool(scale_coarse),
                "coarse_scale_nbytes_actual": int(scale_nbytes),
                "row_scaling": row_scale_meta,
                "column_scaling": col_scale_meta,
                "permc_spec": str(permc_spec),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        coarse_rhs = row_scale * residual[coarse_positions] if bool(scale_coarse) else residual[coarse_positions]
        correction = np.asarray(factor.solve(coarse_rhs), dtype=np.float64).reshape((-1,))
        if bool(scale_coarse):
            correction = col_scale * correction
        out = y_base.copy()
        if np.all(np.isfinite(correction)):
            out[coarse_positions] += float(damping) * correction
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    tail_size = int(layout.total_size) - int(layout.f_size)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_low_l_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "requested_base_kind": str(base_kind_used),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "coarse_size": int(coarse_matrix.shape[0]),
            "coarse_matrix_nnz": int(coarse_matrix.nnz),
            "coarse_matrix_nbytes_actual": int(_scipy_csr_nbytes(coarse_matrix.tocsr())),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(factor_estimate),
            "low_l_factor_nbytes_estimate": int(low_l_factor_estimate),
            "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
            "dense_factor_nbytes_estimate": int(dense_factor_estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "low_l_factor_nbytes_actual": int(low_l_factor_nbytes),
            "base_factor_nbytes_actual": int(base_factor_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "diagonal_shift": float(diagonal_shift),
            "lmax": int(lmax),
            "tail_size": int(tail_size),
            "tail_included": bool(tail_size > 0),
            "fill_estimate": float(fill_estimate),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "factor_kind": str(factor_kind),
            "diag_pivot_thresh": float(diag_pivot),
            "damping": float(damping),
            "coarse_scaling_enabled": bool(scale_coarse),
            "coarse_scale_norm": str(scale_norm),
            "coarse_scale_max": float(max_scale),
            "coarse_scale_nbytes_actual": int(scale_nbytes),
            "row_scaling": row_scale_meta,
            "column_scaling": col_scale_meta,
            "permc_spec": str(permc_spec),
        },
    )


def _build_active_projected_ell_band_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    base_kind: str | None,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build an exact sparse residual correction on a selected pitch band.

    The production QH audits show residual energy concentrated in a narrow
    Legendre band while full low-L Schur factors are too large. This factor
    solves the residual equation on a bounded ``ell`` interval across all
    species/speeds/angles, optionally including the global tail, and leaves the
    rest to a cheap base preconditioner.
    """

    from scipy.sparse.linalg import LinearOperator, splu  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    default_center = min(4, max(0, int(layout.n_xi) - 1))
    ell_center = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_CENTER", default_center)
    half_width = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_HALF_WIDTH", 1)
    ell_min = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_MIN", int(ell_center) - int(half_width))
    ell_max = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_MAX", int(ell_center) + int(half_width))
    ell_min = max(0, min(int(layout.n_xi) - 1, int(ell_min)))
    ell_max = max(0, min(int(layout.n_xi) - 1, int(ell_max)))
    if ell_min > ell_max:
        ell_min, ell_max = ell_max, ell_min
    include_tail = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_INCLUDE_TAIL", True)

    band_full = _active_ell_band_full_indices(
        layout=layout,
        ell_min=int(ell_min),
        ell_max=int(ell_max),
        include_tail=bool(include_tail),
    )
    band_positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=band_full)
    if band_positions.size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason="empty_active_ell_band_schur_space",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
                "include_tail": bool(include_tail),
            },
        )

    requested_base = (
        str(base_kind)
        if base_kind is not None
        else os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_BASE", "active_xblock")
    )
    requested_base = requested_base.strip().lower().replace("-", "_")
    if requested_base in {"none", "jacobi", "diagonal"}:
        base = _build_jacobi_preconditioner(
            matrix=matrix_csr,
            requested_kind=str(requested_kind),
            regularization=regularization,
            t0=t0,
            reason="active_ell_band_schur_base_jacobi",
        )
        base_kind_used = "jacobi"
    elif requested_base in {
        "overlap_schwarz",
        "active_overlap_schwarz",
        "additive_schwarz",
        "restricted_schwarz",
        "ras",
        "active_ras",
    }:
        base = _build_active_projected_overlap_schwarz_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            requested_kind="active_overlap_schwarz",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
        base_kind_used = "active_overlap_schwarz"
    else:
        base = _build_active_projected_xblock_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            requested_kind="active_xblock",
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
        base_kind_used = "active_xblock"
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason=f"base_preconditioner_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "base_preconditioner": base.to_dict(),
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
            },
        )
    base_factor_nbytes = int(base.metadata.get("factor_nbytes_actual", 0) or 0)

    band_matrix_raw = matrix_csr[band_positions[:, None], band_positions].tocsc()
    scale_band = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_SCALE", True)
    scale_norm = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_SCALE_NORM", "l1").strip().lower()
    max_scale = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_MAX_SCALE", 1.0e6))
    row_scale = np.ones((int(band_matrix_raw.shape[0]),), dtype=np.float64)
    col_scale = np.ones((int(band_matrix_raw.shape[1]),), dtype=np.float64)
    row_meta: dict[str, object] = {"scale_min": 1.0, "scale_max": 1.0}
    col_meta: dict[str, object] = {"scale_min": 1.0, "scale_max": 1.0}
    if scale_band:
        row_scale, row_meta = _sparse_equilibration_scale(
            band_matrix_raw,
            axis=1,
            norm=scale_norm,
            max_scale=max_scale,
        )
        row_scaled = band_matrix_raw.multiply(row_scale[:, None]).tocsc()
        col_scale, col_meta = _sparse_equilibration_scale(
            row_scaled,
            axis=0,
            norm=scale_norm,
            max_scale=max_scale,
        )
        band_matrix = row_scaled.multiply(col_scale[None, :]).tocsc()
    else:
        band_matrix = band_matrix_raw

    band_scale = max(float(np.max(np.abs(band_matrix.data))) if band_matrix.nnz else 0.0, 1.0)
    diagonal_shift_override = _env_float(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_DIAGONAL_SHIFT",
        float("nan"),
    )
    if np.isfinite(float(diagonal_shift_override)):
        diagonal_shift = max(0.0, float(diagonal_shift_override)) * band_scale
    else:
        diagonal_shift = max(float(abs(regularization)), 1.0e-14) * band_scale
    if diagonal_shift > 0.0:
        band_matrix = band_matrix + diagonal_shift * sp.eye(
            band_matrix.shape[0],
            dtype=np.float64,
            format="csc",
        )

    fill_estimate = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_FILL_ESTIMATE", 8.0))
    sparse_factor_estimate = _estimate_spilu_factor_nbytes(matrix=band_matrix, fill_factor=fill_estimate)
    dense_factor_estimate = int(band_matrix.shape[0] * band_matrix.shape[1] * np.dtype(np.float64).itemsize)
    include_dense_estimate = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_DENSE_ESTIMATE", False)
    scale_nbytes = int(row_scale.nbytes + col_scale.nbytes) if scale_band else 0
    band_factor_estimate = (
        max(int(sparse_factor_estimate), int(dense_factor_estimate))
        if include_dense_estimate
        else int(sparse_factor_estimate)
    )
    factor_estimate = int(base_factor_nbytes + band_factor_estimate + scale_nbytes)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason=f"active_ell_band_schur_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(matrix_csr.shape[0]),
                "band_size": int(band_matrix.shape[0]),
                "band_matrix_nnz": int(band_matrix.nnz),
                "band_matrix_nbytes_actual": int(_scipy_csr_nbytes(band_matrix.tocsr())),
                "factor_nbytes_estimate": int(factor_estimate),
                "band_factor_nbytes_estimate": int(band_factor_estimate),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
                "dense_factor_nbytes_estimate": int(dense_factor_estimate),
                "scale_nbytes_actual": int(scale_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
                "include_tail": bool(include_tail),
                "include_dense_estimate": bool(include_dense_estimate),
                "fill_estimate": float(fill_estimate),
            },
        )

    permc_spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_PERMC_SPEC", "COLAMD").strip().upper()
    if permc_spec not in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}:
        permc_spec = "COLAMD"
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_DIAG_PIVOT_THRESH", 0.0))
    coefficient_mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_COEFFICIENTS", "least_squares")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coefficient_mode not in {"additive", "least_squares", "normal", "normal_equations"}:
        coefficient_mode = "least_squares"
    if coefficient_mode in {"normal", "normal_equations"}:
        coefficient_mode = "least_squares"
    alpha_max = max(0.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_ALPHA_MAX", 10.0)))
    try:
        factor = splu(band_matrix, permc_spec=str(permc_spec), diag_pivot_thresh=float(diag_pivot))
    except Exception as exc:  # noqa: BLE001
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason=f"active_ell_band_schur_factor_failed:{type(exc).__name__}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "error": str(exc),
                "factor_nbytes_estimate": int(factor_estimate),
                "band_factor_nbytes_estimate": int(band_factor_estimate),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
            },
        )
    band_factor_nbytes = int(_sparse_lu_factor_nbytes(factor) + scale_nbytes)
    factor_nbytes = int(base_factor_nbytes + band_factor_nbytes)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_ell_band_schur",
            reason=f"active_ell_band_schur_factor_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_size": int(matrix_csr.shape[0]),
                "band_size": int(band_matrix.shape[0]),
                "band_matrix_nnz": int(band_matrix.nnz),
                "band_matrix_nbytes_actual": int(_scipy_csr_nbytes(band_matrix.tocsr())),
                "factor_nbytes_estimate": int(factor_estimate),
                "band_factor_nbytes_estimate": int(band_factor_estimate),
                "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
                "dense_factor_nbytes_estimate": int(dense_factor_estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "band_factor_nbytes_actual": int(band_factor_nbytes),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "scale_nbytes_actual": int(scale_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "ell_min": int(ell_min),
                "ell_max": int(ell_max),
                "include_tail": bool(include_tail),
                "fill_estimate": float(fill_estimate),
                "permc_spec": str(permc_spec),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        band_rhs = residual[band_positions]
        if scale_band:
            correction = col_scale * np.asarray(factor.solve(row_scale * band_rhs), dtype=np.float64).reshape((-1,))
        else:
            correction = np.asarray(factor.solve(band_rhs), dtype=np.float64).reshape((-1,))
        out = y_base.copy()
        alpha = 1.0
        if coefficient_mode == "least_squares":
            trial = np.zeros_like(out)
            trial[band_positions] = correction
            a_trial = np.asarray(matrix_csr @ trial, dtype=np.float64).reshape((-1,))
            denom = float(np.dot(a_trial, a_trial))
            if np.isfinite(denom) and denom > 0.0:
                alpha = float(np.dot(a_trial, residual) / denom)
                if alpha_max > 0.0:
                    alpha = float(np.clip(alpha, -float(alpha_max), float(alpha_max)))
            else:
                alpha = 0.0
        out[band_positions] += float(alpha) * correction
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_ell_band_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base.kind),
            "requested_base_kind": str(base_kind_used),
            "base_preconditioner": base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "band_size": int(band_matrix.shape[0]),
            "band_matrix_nnz": int(band_matrix.nnz),
            "band_matrix_nbytes_actual": int(_scipy_csr_nbytes(band_matrix.tocsr())),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(factor_estimate),
            "band_factor_nbytes_estimate": int(band_factor_estimate),
            "sparse_factor_nbytes_estimate": int(sparse_factor_estimate),
            "dense_factor_nbytes_estimate": int(dense_factor_estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "band_factor_nbytes_actual": int(band_factor_nbytes),
            "base_factor_nbytes_actual": int(base_factor_nbytes),
            "scale_nbytes_actual": int(scale_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "diagonal_shift": float(diagonal_shift),
            "ell_center": int(ell_center),
            "ell_min": int(ell_min),
            "ell_max": int(ell_max),
            "ell_count": int(ell_max - ell_min + 1),
            "include_tail": bool(include_tail),
            "tail_size": int(layout.total_size) - int(layout.f_size),
            "tail_included": bool(include_tail),
            "fill_estimate": float(fill_estimate),
            "permc_spec": str(permc_spec),
            "diag_pivot_thresh": float(diag_pivot),
            "coefficient_mode": str(coefficient_mode),
            "alpha_max": float(alpha_max),
            "band_scaling_enabled": bool(scale_band),
            "band_scale_norm": str(scale_norm),
            "band_scale_max": float(max_scale),
            "row_scaling": row_meta,
            "column_scaling": col_meta,
            "include_dense_estimate": bool(include_dense_estimate),
        },
    )


def _build_active_projected_xell_window_lsq_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a residual-equation correction on a small ``(x, ell)`` window.

    Unlike a principal submatrix Schur correction, this preconditioner forms
    ``A[:, W]`` and solves ``min_delta ||r - A[:, W] delta||``. This lets
    off-window rows veto corrections that would locally solve the selected
    pitch/radial window but worsen the global active residual.
    """

    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    matrix_csr = matrix.tocsr()
    active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active_np.shape != (int(matrix_csr.shape[0]),):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason="active_index_size_mismatch",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "active_index_size": int(active_np.size),
                "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
            },
        )

    spec_env = os.environ.get(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_SPEC",
        f"0:0:{min(4, max(0, int(layout.n_xi) - 1))}",
    )
    specs = _parse_active_xell_window_specs(spec_env, layout=layout)
    x_radius = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_X_RADIUS", 0)))
    ell_radius = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_ELL_RADIUS", 1)))
    include_tail = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_INCLUDE_TAIL", True)
    max_window_size = max(1, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_MAX_SIZE", 4096)))

    full_pieces: list[np.ndarray] = []
    theta_range = range(int(layout.n_theta))
    zeta_range = range(int(layout.n_zeta))
    for species, x_center, ell_center in specs:
        x_min = max(0, int(x_center) - int(x_radius))
        x_max = min(int(layout.n_x) - 1, int(x_center) + int(x_radius))
        ell_min = max(0, int(ell_center) - int(ell_radius))
        ell_max = min(int(layout.n_xi) - 1, int(ell_center) + int(ell_radius))
        indices = [
            layout.kinetic_flat_index(
                species=int(species),
                x=int(x_index),
                ell=int(ell),
                theta=int(theta),
                zeta=int(zeta),
            )
            for x_index in range(int(x_min), int(x_max) + 1)
            for ell in range(int(ell_min), int(ell_max) + 1)
            for theta in theta_range
            for zeta in zeta_range
        ]
        if indices:
            full_pieces.append(np.asarray(indices, dtype=np.int64))
    if bool(include_tail):
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0:
            full_pieces.append(int(layout.f_size) + np.arange(tail_size, dtype=np.int64))
    if not full_pieces:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason="empty_active_xell_window",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"spec": str(spec_env)},
        )
    window_full = np.unique(np.concatenate(full_pieces).astype(np.int64, copy=False))
    window_positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=window_full)
    window_size = int(window_positions.size)
    if window_size == 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason="empty_active_xell_window",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"spec": str(spec_env), "window_full_size": int(window_full.size)},
        )
    if window_size > int(max_window_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason=f"active_xell_window_lsq_schur_window_size_exceeded:{window_size}>{int(max_window_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "spec": str(spec_env),
                "window_size": int(window_size),
                "max_window_size": int(max_window_size),
                "x_radius": int(x_radius),
                "ell_radius": int(ell_radius),
                "include_tail": bool(include_tail),
            },
        )

    base_kind = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_BASE", "active_xblock")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if base_kind in {"", "none", "false", "off", "zero"}:
        base = None
        base_kind_used = "zero"
        base_factor_nbytes = 0
    else:
        base = build_active_projected_rhs1_full_csr_preconditioner(
            matrix=matrix_csr,
            layout=layout,
            active_indices=active_np,
            kind=base_kind,
            max_factor_nbytes=max_factor_nbytes,
            regularization=regularization,
        )
        base_kind_used = str(base.kind)
        if not bool(base.selected) or base.operator is None:
            base = _build_jacobi_preconditioner(
                matrix=matrix_csr,
                requested_kind="active_xell_window_lsq_schur_base_fallback",
                regularization=regularization,
                t0=t0,
                reason=f"active_xell_window_base_failed:{base.reason}",
            )
            base_kind_used = "jacobi"
        base_factor_nbytes = int(
            base.metadata.get("factor_nbytes_actual", base.metadata.get("factor_nbytes_estimate", 0)) or 0
        )

    a_window = matrix_csr[:, window_positions].tocsc()
    a_window_nbytes = int(_scipy_csr_nbytes(a_window.tocsr()))
    col_norms = np.sqrt(np.asarray(a_window.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    inv_col_scale = np.zeros_like(col_norms)
    good = np.isfinite(col_norms) & (col_norms > norm_floor)
    inv_col_scale[good] = 1.0 / col_norms[good]
    if not np.any(good):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason="active_xell_window_lsq_schur_zero_columns",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"window_size": int(window_size), "spec": str(spec_env)},
        )
    a_scaled = a_window @ sp.diags(inv_col_scale, format="csc")
    normal_nbytes = int(window_size * window_size * np.dtype(np.float64).itemsize)
    factor_estimate = int(base_factor_nbytes + a_window_nbytes + 2 * normal_nbytes + inv_col_scale.nbytes)
    if factor_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_xell_window_lsq_schur",
            reason=f"active_xell_window_lsq_schur_budget_exceeded:{factor_estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "spec": str(spec_env),
                "window_size": int(window_size),
                "a_window_nbytes_actual": int(a_window_nbytes),
                "normal_nbytes": int(normal_nbytes),
                "base_factor_nbytes_actual": int(base_factor_nbytes),
                "factor_nbytes_estimate": int(factor_estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    normal = np.asarray((a_scaled.T @ a_scaled).toarray(), dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    normal_regularization = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + normal_regularization * np.eye(window_size, dtype=np.float64)
    solver_kind = "cholesky"
    try:
        factor = cho_factor(normal, lower=True, check_finite=False)

        def solve_normal(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(cho_solve(factor, rhs, check_finite=False), dtype=np.float64).reshape((-1,))

    except Exception:  # noqa: BLE001
        solver_kind = "lu"
        lu, piv = lu_factor(normal, check_finite=False)

        def solve_normal(rhs: np.ndarray) -> np.ndarray:
            return np.asarray(lu_solve((lu, piv), rhs, check_finite=False), dtype=np.float64).reshape((-1,))

    condition_estimate = None
    if window_size <= 256:
        condition_estimate = float(np.linalg.cond(normal))

    global_damping = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_GLOBAL_DAMPING", True)
    beta_max = max(0.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_BETA_MAX", 10.0)))

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        if base is None:
            y_base = np.zeros_like(arr)
        else:
            y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix_csr @ y_base, dtype=np.float64).reshape((-1,))
        rhs_window = np.asarray(a_scaled.T @ residual, dtype=np.float64).reshape((-1,))
        scaled_delta = solve_normal(rhs_window)
        delta = inv_col_scale * scaled_delta
        out = y_base.copy()
        out[window_positions] += delta
        if bool(global_damping):
            a_out = np.asarray(matrix_csr @ out, dtype=np.float64).reshape((-1,))
            denom = float(np.dot(a_out, a_out))
            if np.isfinite(denom) and denom > 0.0:
                beta = float(np.dot(a_out, arr) / denom)
                if beta_max > 0.0:
                    beta = float(np.clip(beta, -float(beta_max), float(beta_max)))
            else:
                beta = 0.0
            out = float(beta) * out
        return out

    operator = LinearOperator(matrix_csr.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="active_xell_window_lsq_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "base_kind": str(base_kind_used),
            "requested_base_kind": str(base_kind),
            "base_preconditioner": None if base is None else base.to_dict(),
            "active_size": int(matrix_csr.shape[0]),
            "spec": str(spec_env),
            "parsed_specs": tuple(tuple(int(v) for v in spec) for spec in specs),
            "x_radius": int(x_radius),
            "ell_radius": int(ell_radius),
            "include_tail": bool(include_tail),
            "window_size": int(window_size),
            "window_kinetic_size": int(np.count_nonzero(active_np[window_positions] < int(layout.f_size))),
            "window_tail_size": int(np.count_nonzero(active_np[window_positions] >= int(layout.f_size))),
            "a_window_nnz": int(a_window.nnz),
            "a_window_nbytes_actual": int(a_window_nbytes),
            "normal_nbytes": int(normal_nbytes),
            "normal_regularization": float(normal_regularization),
            "normal_solver": str(solver_kind),
            "normal_condition_estimate": condition_estimate,
            "global_damping": bool(global_damping),
            "global_damping_beta_max": float(beta_max),
            "base_factor_nbytes_actual": int(base_factor_nbytes),
            "factor_nbytes_estimate": int(factor_estimate),
            "factor_nbytes_actual": int(factor_estimate),
            "max_factor_nbytes": int(max_factor_nbytes),
            "nonzero_column_count": int(np.count_nonzero(good)),
            "zero_or_invalid_column_count": int(window_size - np.count_nonzero(good)),
        },
    )


def _parse_active_xell_window_specs(spec: str, *, layout: RHS1BlockLayout) -> tuple[tuple[int, int, int], ...]:
    """Parse ``species:x:ell`` target triples for active residual windows."""

    triples: list[tuple[int, int, int]] = []
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.replace("/", ":").split(":")
        if len(parts) != 3:
            continue
        try:
            species, x_index, ell = (int(part.strip()) for part in parts)
        except ValueError:
            continue
        species = max(0, min(int(layout.n_species) - 1, int(species)))
        x_index = max(0, min(int(layout.n_x) - 1, int(x_index)))
        ell = max(0, min(int(layout.n_xi) - 1, int(ell)))
        triples.append((int(species), int(x_index), int(ell)))
    if not triples:
        triples.append((0, 0, min(4, max(0, int(layout.n_xi) - 1))))
    return tuple(dict.fromkeys(triples))


def _active_low_l_schur_full_indices(*, layout: RHS1BlockLayout, lmax: int) -> np.ndarray:
    """Return full-system indices used by the active low-L Schur correction."""

    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    pieces: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            pieces.append(_xblock_tz_low_l_indices(layout=layout, species=species, x=x, lmax=lmax))
    tail_size = int(layout.total_size) - int(layout.f_size)
    if tail_size > 0:
        pieces.append(int(layout.f_size) + np.arange(tail_size, dtype=np.int64))
    if not pieces:
        return np.zeros((0,), dtype=np.int64)
    indices = np.concatenate(pieces).astype(np.int64, copy=False)
    return np.unique(indices)


def _active_ell_band_full_indices(
    *,
    layout: RHS1BlockLayout,
    ell_min: int,
    ell_max: int,
    include_tail: bool,
) -> np.ndarray:
    """Return full-system indices for a selected active pitch band."""

    ell_min_use = max(0, min(int(layout.n_xi) - 1, int(ell_min)))
    ell_max_use = max(0, min(int(layout.n_xi) - 1, int(ell_max)))
    if ell_min_use > ell_max_use:
        ell_min_use, ell_max_use = ell_max_use, ell_min_use
    pieces: list[np.ndarray] = []
    theta = range(int(layout.n_theta))
    zeta = range(int(layout.n_zeta))
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            indices = [
                layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta_i, zeta=zeta_i)
                for ell in range(int(ell_min_use), int(ell_max_use) + 1)
                for theta_i in theta
                for zeta_i in zeta
            ]
            if indices:
                pieces.append(np.asarray(indices, dtype=np.int64))
    if bool(include_tail):
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0:
            pieces.append(int(layout.f_size) + np.arange(tail_size, dtype=np.int64))
    if not pieces:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(pieces).astype(np.int64, copy=False))


def _append_probe_residual_basis_csc(
    *,
    matrix: Any,
    base_operator: Any,
    basis: Any,
    max_total_columns: int,
    enabled_default: bool = False,
) -> tuple[Any, dict[str, object]]:
    """Append bounded residual modes from deterministic setup probes."""

    from sfincs_jax.explicit_sparse import deterministic_sparse_probe_matrix  # noqa: PLC0415

    enabled = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_BASIS", bool(enabled_default))
    max_columns = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_MAX_COLUMNS", 32)))
    probe_count = max(1, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_PROBES", 8)))
    max_nnz_per_column = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_MAX_NNZ_PER_COLUMN", 8192)),
    )
    drop_rel = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_DROP_REL", 1.0e-3)),
    )
    min_rel_norm = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_MIN_REL_NORM", 1.0e-8)),
    )
    metadata = {
        "probe_residual_basis_enabled": bool(enabled),
        "probe_residual_basis_columns": 0,
        "probe_residual_basis_probe_count": int(probe_count),
        "probe_residual_basis_max_columns": int(max_columns),
        "probe_residual_basis_max_nnz_per_column": int(max_nnz_per_column),
        "probe_residual_basis_drop_rel": float(drop_rel),
        "probe_residual_basis_min_rel_norm": float(min_rel_norm),
    }
    basis_csc = basis.tocsc()
    if not bool(enabled) or int(max_columns) <= 0 or int(max_total_columns) <= int(basis_csc.shape[1]):
        metadata["probe_residual_basis_truncated_by_total_cap"] = bool(
            int(max_total_columns) <= int(basis_csc.shape[1])
        )
        return basis_csc, metadata

    matrix_csr = matrix.tocsr()
    remaining = max(0, int(max_total_columns) - int(basis_csc.shape[1]))
    max_columns_use = min(int(max_columns), int(remaining))
    probes = deterministic_sparse_probe_matrix(
        int(matrix_csr.shape[0]),
        count=int(probe_count),
        dtype=matrix_csr.dtype,
    )
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    residual_norms: list[float] = []
    skipped_small = 0
    skipped_zero = 0
    for probe_col in range(int(probes.shape[1])):
        if len(rows) >= max_columns_use:
            break
        probe = np.asarray(probes[:, probe_col], dtype=np.float64).reshape((-1,))
        probe_norm = max(float(np.linalg.norm(probe)), np.finfo(np.float64).tiny)
        mz = np.asarray(base_operator.matvec(probe), dtype=np.float64).reshape((-1,))
        residual = probe - np.asarray(matrix_csr @ mz, dtype=np.float64).reshape((-1,))
        if int(basis_csc.shape[1]) > 0:
            projection = basis_csc @ np.asarray(basis_csc.T @ residual, dtype=np.float64).reshape((-1,))
            residual = residual - np.asarray(projection, dtype=np.float64).reshape((-1,))
        for previous_rows, previous_data in zip(rows, data, strict=False):
            previous = np.zeros_like(residual)
            previous[previous_rows] = previous_data
            residual = residual - previous * float(np.dot(previous, residual))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or residual_norm <= 0.0:
            skipped_zero += 1
            continue
        if residual_norm / probe_norm < float(min_rel_norm):
            skipped_small += 1
            continue
        abs_residual = np.abs(residual)
        threshold = float(drop_rel) * max(float(np.max(abs_residual)), np.finfo(np.float64).tiny)
        keep = np.flatnonzero(abs_residual >= threshold)
        if keep.size > int(max_nnz_per_column):
            order = np.argpartition(abs_residual[keep], -int(max_nnz_per_column))[-int(max_nnz_per_column) :]
            keep = keep[order]
            keep.sort()
        if keep.size == 0:
            skipped_zero += 1
            continue
        values = residual[keep]
        value_norm = float(np.linalg.norm(values))
        if not np.isfinite(value_norm) or value_norm <= 0.0:
            skipped_zero += 1
            continue
        rows.append(keep.astype(np.int64, copy=False))
        cols.append(np.full((int(keep.size),), len(rows) - 1, dtype=np.int64))
        data.append((values / value_norm).astype(np.float64, copy=False))
        residual_norms.append(float(residual_norm))

    if not rows:
        metadata.update(
            {
                "probe_residual_basis_skipped_small": int(skipped_small),
                "probe_residual_basis_skipped_zero": int(skipped_zero),
                "probe_residual_basis_truncated_by_total_cap": False,
            }
        )
        return basis_csc, metadata

    residual_basis = sp.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(int(matrix_csr.shape[0]), int(len(rows))),
    ).tocsc()
    residual_basis.sum_duplicates()
    residual_basis.eliminate_zeros()
    combined = sp.hstack([basis_csc, residual_basis], format="csc")
    metadata.update(
        {
            "probe_residual_basis_columns": int(residual_basis.shape[1]),
            "probe_residual_basis_nnz": int(residual_basis.nnz),
            "probe_residual_basis_skipped_small": int(skipped_small),
            "probe_residual_basis_skipped_zero": int(skipped_zero),
            "probe_residual_basis_residual_norm_max": float(max(residual_norms)),
            "probe_residual_basis_residual_norm_min": float(min(residual_norms)),
            "probe_residual_basis_truncated_by_total_cap": bool(len(rows) >= max_columns_use),
        }
    )
    return combined, metadata


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or int(default))
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "").strip() or float(default))
    except ValueError:
        return float(default)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return bool(default)
    return value in {"1", "true", "yes", "on"}


def _unsupported_reason(*, op: Any, include_jacobian_terms: bool) -> str | None:
    if int(op.constraint_scheme) not in {1, 2}:
        return f"unsupported_constraint_scheme:{int(op.constraint_scheme)}"
    if bool(op.include_phi1_in_kinetic):
        return "unsupported_phi1_in_kinetic"
    if bool(op.include_phi1) and not bool(include_jacobian_terms):
        return "unsupported_phi1_nonlinear_residual"
    return None


def _assemble_full_tail_csr(*, op: Any, layout: RHS1BlockLayout, include_jacobian_terms: bool) -> sp.csr_matrix:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []

    def append(row_values: Any, col_values: Any, values: Any) -> None:
        row_arr = np.asarray(row_values, dtype=np.int64).reshape((-1,))
        col_arr = np.asarray(col_values, dtype=np.int64).reshape((-1,))
        val_arr = np.asarray(values, dtype=np.float64).reshape((-1,))
        if row_arr.size == 0:
            return
        if row_arr.shape != col_arr.shape or row_arr.shape != val_arr.shape:
            raise ValueError("tail COO row, column, and data arrays must have matching lengths")
        keep = np.abs(val_arr) > 0.0
        if not np.any(keep):
            return
        rows.append(row_arr[keep])
        cols.append(col_arr[keep])
        data.append(val_arr[keep])

    factor = _fs_average_factor_np(op)
    if bool(op.include_phi1):
        _append_phi1_tail(
            op=op,
            layout=layout,
            factor=factor,
            append=append,
            include_jacobian_terms=include_jacobian_terms,
        )
    if int(op.constraint_scheme) == 1:
        _append_constraint_scheme1_tail(op=op, layout=layout, factor=factor, append=append)
    elif int(op.constraint_scheme) == 2:
        _append_constraint_scheme2_tail(op=op, layout=layout, factor=factor, append=append)

    if rows:
        row = np.concatenate(rows)
        col = np.concatenate(cols)
        val = np.concatenate(data)
    else:
        row = np.zeros((0,), dtype=np.int64)
        col = np.zeros((0,), dtype=np.int64)
        val = np.zeros((0,), dtype=np.float64)
    matrix = sp.coo_matrix((val, (row, col)), shape=(int(layout.total_size), int(layout.total_size))).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    return matrix


def _append_phi1_tail(
    *,
    op: Any,
    layout: RHS1BlockLayout,
    factor: np.ndarray,
    append: Any,
    include_jacobian_terms: bool,
) -> None:
    n_t = int(layout.n_theta)
    n_z = int(layout.n_zeta)
    n_x = int(layout.n_x)
    qn_rows = int(layout.f_size) + np.arange(n_t * n_z, dtype=np.int64)
    theta = np.arange(n_t, dtype=np.int64)
    zeta = np.arange(n_z, dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    qn_grid = qn_rows.reshape((n_t, n_z))

    x = np.asarray(op.x, dtype=np.float64)
    x2w = x * x * np.asarray(op.x_weights, dtype=np.float64)
    species_factor = (
        4.0
        * np.pi
        * np.asarray(op.z_s, dtype=np.float64)
        * np.asarray(op.t_hat, dtype=np.float64)
        / np.asarray(op.m_hat, dtype=np.float64)
        * np.sqrt(np.asarray(op.t_hat, dtype=np.float64) / np.asarray(op.m_hat, dtype=np.float64))
    )
    species = [0] if int(op.quasineutrality_option) == 2 else range(int(layout.n_species))
    for s in species:
        for ix in range(n_x):
            cols = _f_indices_for_l0_surface(layout=layout, species=int(s), x=ix, theta=theta_grid, zeta=zeta_grid)
            append(qn_grid, cols, species_factor[int(s)] * x2w[ix] * np.ones((n_t, n_z), dtype=np.float64))

    phi1_cols = int(layout.f_size) + np.arange(n_t * n_z, dtype=np.int64)
    diag = _phi1_qn_diag_np(op=op, include_jacobian_terms=include_jacobian_terms)
    if np.ndim(diag) == 0:
        diag_values = float(diag) * np.ones((n_t * n_z,), dtype=np.float64)
    else:
        diag_values = np.asarray(diag, dtype=np.float64).reshape((-1,))
    append(qn_rows, phi1_cols, diag_values)

    lambda_col = int(layout.f_size) + n_t * n_z
    append(qn_rows, np.full((n_t * n_z,), lambda_col, dtype=np.int64), np.ones((n_t * n_z,), dtype=np.float64))

    lambda_row = int(layout.f_size) + n_t * n_z
    append(
        np.full((n_t * n_z,), lambda_row, dtype=np.int64),
        phi1_cols,
        factor.reshape((-1,)),
    )


def _append_constraint_scheme1_tail(*, op: Any, layout: RHS1BlockLayout, factor: np.ndarray, append: Any) -> None:
    ix0 = 1 if bool(op.point_at_x0) else 0
    n_t = int(layout.n_theta)
    n_z = int(layout.n_zeta)
    theta = np.arange(n_t, dtype=np.int64)
    zeta = np.arange(n_z, dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    extra_start = int(layout.f_size + layout.phi1_size)
    xpart1, xpart2 = _source_basis_constraint_scheme1_np(op.x)

    for s in range(int(layout.n_species)):
        particle_col = extra_start + 2 * s
        energy_col = extra_start + 2 * s + 1
        for ix in range(ix0, int(layout.n_x)):
            rows = _f_indices_for_l0_surface(layout=layout, species=s, x=ix, theta=theta_grid, zeta=zeta_grid)
            append(rows, np.full((n_t, n_z), particle_col, dtype=np.int64), xpart1[ix] * np.ones((n_t, n_z)))
            append(rows, np.full((n_t, n_z), energy_col, dtype=np.int64), xpart2[ix] * np.ones((n_t, n_z)))

        x = np.asarray(op.x, dtype=np.float64)
        x2 = x * x
        x4 = x2 * x2
        w2 = x2 * np.asarray(op.x_weights, dtype=np.float64)
        w4 = x4 * np.asarray(op.x_weights, dtype=np.float64)
        density_row = extra_start + 2 * s
        pressure_row = extra_start + 2 * s + 1
        for ix in range(int(layout.n_x)):
            cols = _f_indices_for_l0_surface(layout=layout, species=s, x=ix, theta=theta_grid, zeta=zeta_grid)
            append(np.full((n_t, n_z), density_row, dtype=np.int64), cols, w2[ix] * factor)
            append(np.full((n_t, n_z), pressure_row, dtype=np.int64), cols, w4[ix] * factor)


def _append_constraint_scheme2_tail(*, op: Any, layout: RHS1BlockLayout, factor: np.ndarray, append: Any) -> None:
    ix0 = 1 if bool(op.point_at_x0) else 0
    n_t = int(layout.n_theta)
    n_z = int(layout.n_zeta)
    theta = np.arange(n_t, dtype=np.int64)
    zeta = np.arange(n_z, dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    extra_start = int(layout.f_size + layout.phi1_size)

    for s in range(int(layout.n_species)):
        for ix in range(int(layout.n_x)):
            extra_col = extra_start + s * int(layout.n_x) + ix
            if ix >= ix0:
                rows = _f_indices_for_l0_surface(layout=layout, species=s, x=ix, theta=theta_grid, zeta=zeta_grid)
                append(rows, np.full((n_t, n_z), extra_col, dtype=np.int64), np.ones((n_t, n_z), dtype=np.float64))

            row = extra_start + s * int(layout.n_x) + ix
            if bool(op.point_at_x0) and ix == 0:
                append(np.asarray([row], dtype=np.int64), np.asarray([extra_col], dtype=np.int64), np.asarray([1.0]))
            else:
                cols = _f_indices_for_l0_surface(layout=layout, species=s, x=ix, theta=theta_grid, zeta=zeta_grid)
                append(np.full((n_t, n_z), row, dtype=np.int64), cols, factor)


def _f_indices_for_l0_surface(
    *,
    layout: RHS1BlockLayout,
    species: int,
    x: int,
    theta: np.ndarray,
    zeta: np.ndarray,
) -> np.ndarray:
    return (
        (((int(species) * int(layout.n_x) + int(x)) * int(layout.n_xi)) * int(layout.n_theta) + theta)
        * int(layout.n_zeta)
        + zeta
    ).astype(np.int64, copy=False)


def _fs_average_factor_np(op: Any) -> np.ndarray:
    theta_weights = np.asarray(op.theta_weights, dtype=np.float64)
    zeta_weights = np.asarray(op.zeta_weights, dtype=np.float64)
    d_hat = np.asarray(op.d_hat, dtype=np.float64)
    return (theta_weights[:, None] * zeta_weights[None, :]) / d_hat


def _source_basis_constraint_scheme1_np(x_in: Any) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_in, dtype=np.float64)
    x2 = x * x
    sqrt_pi = np.sqrt(np.pi)
    coef = np.exp(-x2) / (np.pi * sqrt_pi)
    s1 = (-x2 + 2.5) * coef
    s2 = ((2.0 / 3.0) * x2 - 1.0) * coef
    return s1, s2


def _phi1_qn_diag_np(*, op: Any, include_jacobian_terms: bool) -> np.ndarray | float:
    if not bool(include_jacobian_terms) and int(op.quasineutrality_option) == 1:
        return 0.0
    if int(op.quasineutrality_option) == 1:
        z = np.asarray(op.z_s, dtype=np.float64)
        alpha = float(np.asarray(op.alpha, dtype=np.float64))
        n_hat = np.asarray(op.n_hat, dtype=np.float64)
        t_hat = np.asarray(op.t_hat, dtype=np.float64)
        phi_base = np.asarray(op.phi1_hat_base, dtype=np.float64)
        exp_phi = np.exp(-(z[:, None, None] * alpha / t_hat[:, None, None]) * phi_base[None, :, :])
        diag = -np.sum((z * z * alpha * n_hat / t_hat)[:, None, None] * exp_phi, axis=0)
        if bool(op.with_adiabatic):
            adiabatic_z = float(np.asarray(op.adiabatic_z, dtype=np.float64))
            adiabatic_nhat = float(np.asarray(op.adiabatic_nhat, dtype=np.float64))
            adiabatic_that = float(np.asarray(op.adiabatic_that, dtype=np.float64))
            diag = diag - (
                (adiabatic_z * adiabatic_z * alpha * adiabatic_nhat / adiabatic_that)
                * np.exp(-(adiabatic_z * alpha / adiabatic_that) * phi_base)
            )
        return diag
    if int(op.quasineutrality_option) == 2 and bool(op.with_adiabatic) and int(op.n_species) > 0:
        z = np.asarray(op.z_s, dtype=np.float64)
        n_hat = np.asarray(op.n_hat, dtype=np.float64)
        t_hat = np.asarray(op.t_hat, dtype=np.float64)
        alpha = float(np.asarray(op.alpha, dtype=np.float64))
        adiabatic_z = float(np.asarray(op.adiabatic_z, dtype=np.float64))
        adiabatic_nhat = float(np.asarray(op.adiabatic_nhat, dtype=np.float64))
        adiabatic_that = float(np.asarray(op.adiabatic_that, dtype=np.float64))
        return -alpha * (z[0] * z[0] * n_hat[0] / t_hat[0] + adiabatic_z * adiabatic_z * adiabatic_nhat / adiabatic_that)
    return 0.0


def _scipy_csr_nbytes(matrix: Any) -> int:
    return int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)


def _structured_full_csr_object_cache_key(
    *,
    op: Any,
    include_identity_shift: bool,
    phi1_hat_base: Any | None,
    drop_tol: float,
    require_complete: bool,
    include_jacobian_terms: bool,
) -> tuple[object, ...]:
    return (
        "structured_rhs1_full_csr_object",
        id(op),
        bool(include_identity_shift),
        id(phi1_hat_base) if phi1_hat_base is not None else None,
        float(drop_tol),
        bool(require_complete),
        bool(include_jacobian_terms),
    )


__all__ = [
    "RHS1StructuredFullCSRPreconditioner",
    "RHS1StructuredFullCSRSelection",
    "RHS1StructuredFullCSRSolveResult",
    "build_active_projected_rhs1_full_csr_preconditioner",
    "build_direct_active_fortran_v3_reduced_pmat_preconditioner",
    "build_structured_rhs1_full_csr_preconditioner",
    "clear_structured_rhs1_full_csr_cache",
    "select_active_fortran_v3_reduced_support_mode_preconditioner",
    "select_structured_rhs1_full_csr_operator",
    "solve_structured_rhs1_full_csr",
]
