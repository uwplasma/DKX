"""Structured host-CSR assembly for RHSMode=1 full-system operators.

The f-block stencil assembly lives in :mod:`sfincs_jax.rhs1_fblock_assembly`.
This module adds the small global couplings around that block: Phi1/QN rows
when they are linear, and the source/constraint rows used by SFINCS v3.  The
result is an exact sparse matrix for supported RHSMode=1 cases without dense
column probing.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from collections.abc import Callable, Sequence
from typing import Any
import os
import time

import numpy as np
import scipy.sparse as sp

from .rhs1_block_operator import RHS1ActiveBlockLayout, RHS1ActiveFieldSplitOrdering, RHS1BlockLayout
from .rhs1_fblock_assembly import (
    RHS1StructuredFBlockCSRSelection,
    clear_structured_rhs1_fblock_csr_cache,
    select_structured_rhs1_fblock_csr_operator,
)
from .v3_sparse_pattern import estimate_v3_full_system_conservative_sparsity_summary

_STRUCTURED_FULL_CSR_OBJECT_CACHE: dict[tuple[object, ...], tuple[Any, dict[str, object]]] = {}


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


@dataclass(frozen=True)
class RHS1StructuredFullCSRPreconditioner:
    """Host-side inverse preconditioner used by the explicit CSR solve lane."""

    operator: Any | None
    selected: bool
    kind: str
    reason: str
    setup_s: float
    metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly preconditioner metadata."""

        return {
            "selected": bool(self.selected),
            "kind": str(self.kind),
            "reason": str(self.reason),
            "setup_s": float(self.setup_s),
            "metadata": dict(self.metadata),
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
        large_fallback_size = max(
            1,
            int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", 300000)),
        )
        candidate_env_override = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES")
        candidate_env = (
            candidate_env_override
            if candidate_env_override is not None
            else (
                "active_fortran_v3_reduced_lu,"
                "active_fortran_v3_reduced_native_stack,"
                "active_symbolic_block_schur_lu,"
                "active_schwarz_sparse_coarse,"
                "active_global_field_split_schur,"
                "active_xblock_ell_band_schur,"
                "active_ell_band_schur,"
                "active_bounded_native_stack,"
                "active_xblock,"
                "active_diagonal_schur,"
                "active_spilu,"
                "jacobi"
            )
        )
        large_default_used = False
        if candidate_env_override is None and int(matrix.shape[0]) >= int(large_fallback_size):
            candidate_env = os.environ.get(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES",
                (
                    "active_fortran_v3_reduced_native_stack,"
                    "active_coupled_kinetic_field_split_sparse_coarse,"
                    "active_symbolic_block_schur_lu,"
                    "active_fortran_v3_reduced_lu"
                ),
            )
            large_default_used = True
        candidates = [
            item.strip().lower().replace("-", "_")
            for item in candidate_env.split(",")
            if item.strip()
        ]
        if not candidates:
            candidates = ["active_diagonal_schur", "jacobi"]
        auto_candidates_requested = list(candidates)
        large_fallbacks = {
            "active_diagonal_schur",
            "active_diag_schur",
            "active_tail_schur",
            "active_constraint_tail_schur",
            "active_field_split",
            "active_field_split_tail",
            "jacobi",
            "diagonal",
        }
        allow_large_fallback = _env_bool(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_ALLOW_LARGE_DIAGONAL_FALLBACK",
            False,
        )
        skipped_large_fallbacks: list[str] = []
        if int(matrix.shape[0]) >= int(large_fallback_size) and not bool(allow_large_fallback):
            skipped_large_fallbacks = [candidate for candidate in candidates if candidate in large_fallbacks]
            candidates = [candidate for candidate in candidates if candidate not in large_fallbacks]
        rejected: list[dict[str, object]] = []
        log_large_auto = _env_bool(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_PROGRESS",
            bool(large_default_used) or int(matrix.shape[0]) >= int(large_fallback_size),
        )
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


def _build_active_fortran_v3_reduced_sparse_factor_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> RHS1StructuredFullCSRPreconditioner:
    """Factor a Fortran-v3-inspired reduced active preconditioner matrix.

    SFINCS Fortran v3 does not factor the exact Jacobian in its iterative
    branch. It builds ``whichMatrix=0`` with simplified derivative stencils and
    pitch-angle couplings, then uses PETSc GMRES with ``PC LU`` on that reduced
    matrix. This Python-native path mirrors that idea for bounded validation:
    build a reduced active CSR from the exact active operator by dropping the
    couplings controlled by the v3 preconditioner defaults, equilibrate it, and
    factor the reduced matrix under a hard memory gate.
    """

    from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    try:
        reduced, reduction_metadata = _active_fortran_v3_reduced_preconditioner_matrix(
            matrix=matrix,
            layout=layout,
            active_indices=active_indices,
            regularization=regularization,
            preconditioner_x=preconditioner_x,
            preconditioner_xi=preconditioner_xi,
            preconditioner_species=preconditioner_species,
            preconditioner_x_min_l=preconditioner_x_min_l,
        )
    except ValueError as exc:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason="active_fortran_v3_pc_matrix_invalid_layout",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"error": str(exc)},
        )

    max_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_MAX_SIZE", 1_000_000)
    n = int(reduced.shape[0])
    if int(max_size) > 0 and n > int(max_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_size_exceeded:{n}>{int(max_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "max_size": int(max_size),
            },
        )

    requested = str(requested_kind).strip().lower().replace("-", "_")
    factor_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "").strip().lower()
    if not factor_kind:
        factor_kind = "ilu" if "ilu" in requested or requested.endswith("pc_matrix") else "lu"
    if factor_kind not in {"ilu", "spilu", "lu", "splu"}:
        factor_kind = "ilu"
    factor_kind = "lu" if factor_kind in {"lu", "splu"} else "ilu"
    large_matrix = n >= int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_SIZE", 300_000))
    fill_factor_default = 3.0 if factor_kind == "ilu" else 12.0
    drop_tol_default = 3.0e-3 if factor_kind == "ilu" else 0.0
    if bool(large_matrix) and factor_kind == "ilu":
        fill_factor_default = float(
            _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_FILL_FACTOR", 1.2)
        )
        drop_tol_default = float(
            _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_DROP_TOL", 5.0e-2)
        )
    fill_factor = max(
        1.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FILL_FACTOR", fill_factor_default)),
    )
    drop_tol = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DROP_TOL", drop_tol_default))
    diag_pivot = float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAG_PIVOT_THRESH", 0.0))
    permc_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", "").strip().upper()
    permc_candidates = _active_fortran_v3_reduced_permc_candidates(
        requested=permc_env,
        factor_kind=factor_kind,
    )
    permc_spec = str(permc_candidates[0])
    scale_norm = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_SCALE_NORM", "l1").strip().lower()
    if scale_norm not in {"l1", "l2", "max"}:
        scale_norm = "l1"
    max_scale = max(1.0, float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_MAX_SCALE", 1.0e6)))

    estimate = _estimate_spilu_factor_nbytes(matrix=reduced, fill_factor=fill_factor) + 2 * n * np.dtype(np.float64).itemsize
    if _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PROGRESS", bool(large_matrix)):
        print(
            "active_fortran_v3_pc_matrix: factor setup "
            f"n={n} nnz={int(reduced.nnz)} factor_kind={factor_kind} "
            f"fill_factor={float(fill_factor):.3g} drop_tol={float(drop_tol):.3g} "
            f"estimate_mb={float(estimate) / (1024.0 * 1024.0):.1f} "
            f"budget_mb={float(max_factor_nbytes) / (1024.0 * 1024.0):.1f}",
            flush=True,
        )
    if estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_budget_exceeded:{estimate}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            },
        )
    lu_large_prefill_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_SIZE", 300_000)
    lu_prefill_default = 4.5
    if factor_kind == "lu" and n >= int(lu_large_prefill_size):
        lu_prefill_default = float(
            _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_PREFILL_SAFETY_FACTOR", 32.0)
        )
    lu_prefill_safety_factor = max(
        1.0,
        float(
            _env_float(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_PREFILL_SAFETY_FACTOR",
                lu_prefill_default,
            )
        ),
    )
    lu_prefill_estimate = int(np.ceil(float(estimate) * float(lu_prefill_safety_factor)))
    if factor_kind == "lu" and lu_prefill_estimate > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=(
                "active_fortran_v3_pc_matrix_lu_prefill_budget_exceeded:"
                f"{lu_prefill_estimate}>{int(max_factor_nbytes)}"
            ),
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_prefill_estimate": int(lu_prefill_estimate),
                "lu_prefill_safety_factor": float(lu_prefill_safety_factor),
                "lu_large_prefill_size": int(lu_large_prefill_size),
                "max_factor_nbytes": int(max_factor_nbytes),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            },
        )

    row_scale, row_meta = _sparse_equilibration_scale(
        reduced,
        axis=1,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = reduced.multiply(row_scale[:, None]).tocsc()
    col_scale, col_meta = _sparse_equilibration_scale(
        scaled,
        axis=0,
        norm=scale_norm,
        max_scale=max_scale,
    )
    scaled = scaled.multiply(col_scale[None, :]).tocsc()
    factor = None
    selected_kind = "active_fortran_v3_reduced_lu" if factor_kind == "lu" else "active_fortran_v3_reduced_ilu"
    permc_failures: list[dict[str, object]] = []
    selected_permutation: np.ndarray | None = None
    selected_superlu_permc = str(permc_spec)
    for candidate_permc in permc_candidates:
        candidate_permc_use = str(candidate_permc).upper()
        factor_matrix = scaled
        factor_permutation: np.ndarray | None = None
        superlu_permc = candidate_permc_use
        if candidate_permc_use == "RCM":
            # PETSc's single-rank fallback in SFINCS v3 uses MATORDERINGRCM
            # before the sparse direct factorization. SuperLU does not expose
            # RCM through permc_spec, so apply the symmetric permutation
            # explicitly and let SuperLU preserve it.
            factor_permutation = np.asarray(
                reverse_cuthill_mckee(scaled.tocsr(), symmetric_mode=False),
                dtype=np.int64,
            )
            factor_matrix = scaled[factor_permutation, :][:, factor_permutation].tocsc()
            superlu_permc = "NATURAL"
        try:
            if factor_kind == "lu":
                factor = splu(factor_matrix, permc_spec=str(superlu_permc), diag_pivot_thresh=float(diag_pivot))
            else:
                factor = spilu(
                    factor_matrix,
                    drop_tol=float(drop_tol),
                    fill_factor=float(fill_factor),
                    permc_spec=str(superlu_permc),
                    diag_pivot_thresh=float(diag_pivot),
                )
            permc_spec = str(candidate_permc_use)
            selected_permutation = factor_permutation
            selected_superlu_permc = str(superlu_permc)
            break
        except Exception as exc:  # noqa: BLE001
            permc_failures.append(
                {
                    "permc_spec": str(candidate_permc_use),
                    "superlu_permc_spec": str(superlu_permc),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    if factor is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason="active_fortran_v3_pc_matrix_failed:all_permc_specs",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "error": str(permc_failures[-1]["error"]) if permc_failures else "",
                "permc_failures": tuple(dict(entry) for entry in permc_failures),
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "fill_factor": float(fill_factor),
                "drop_tol": float(drop_tol),
                "large_matrix_defaults": bool(large_matrix),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
                "row_scaling": row_meta,
                "column_scaling": col_meta,
            },
        )

    factor_nbytes = int(_sparse_lu_factor_nbytes(factor) + row_scale.nbytes + col_scale.nbytes)
    if factor_nbytes > int(max_factor_nbytes):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="active_fortran_v3_pc_matrix",
            reason=f"active_fortran_v3_pc_matrix_factor_budget_exceeded:{factor_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                **reduction_metadata,
                "factor_kind": str(factor_kind),
                "factor_nbytes_estimate": int(estimate),
                "factor_nbytes_actual": int(factor_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "permc_spec": str(permc_spec),
                "permc_spec_requested": str(permc_env or "AUTO"),
                "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
                "permc_failures": tuple(dict(entry) for entry in permc_failures),
            },
        )

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        scaled_rhs = row_scale * arr
        if selected_permutation is None:
            scaled_solution = np.asarray(factor.solve(scaled_rhs), dtype=np.float64).reshape((-1,))
        else:
            permuted_solution = np.asarray(
                factor.solve(scaled_rhs[selected_permutation]),
                dtype=np.float64,
            ).reshape((-1,))
            scaled_solution = np.empty_like(permuted_solution)
            scaled_solution[selected_permutation] = permuted_solution
        return col_scale * scaled_solution

    operator = LinearOperator(reduced.shape, matvec=apply, dtype=np.float64)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind=selected_kind,
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            **reduction_metadata,
            "requested_kind": str(requested_kind),
            "architecture": "fortran_v3_reduced_active_pc_matrix",
            "factor_kind": str(factor_kind),
            "factor_nnz": int(factor.L.nnz + factor.U.nnz),
            "factor_nbytes_estimate": int(estimate),
            "factor_nbytes_prefill_estimate": int(lu_prefill_estimate) if factor_kind == "lu" else int(estimate),
            "factor_nbytes_actual": int(factor_nbytes),
            "lu_prefill_safety_factor": float(lu_prefill_safety_factor) if factor_kind == "lu" else None,
            "lu_large_prefill_size": int(lu_large_prefill_size),
            "max_factor_nbytes": int(max_factor_nbytes),
            "fill_factor": float(fill_factor),
            "drop_tol": float(drop_tol),
            "large_matrix_defaults": bool(large_matrix),
            "diag_pivot_thresh": float(diag_pivot),
            "permc_spec": str(permc_spec),
            "superlu_permc_spec": str(selected_superlu_permc),
            "explicit_symmetric_ordering": bool(selected_permutation is not None),
            "permc_spec_requested": str(permc_env or "AUTO"),
            "permc_spec_candidates": tuple(str(candidate) for candidate in permc_candidates),
            "permc_failures": tuple(dict(entry) for entry in permc_failures),
            "scale_norm": str(scale_norm),
            "max_scale": float(max_scale),
            "row_scaling": row_meta,
            "column_scaling": col_meta,
            "requires_preflight": bool(factor_kind == "ilu"),
        },
    )


def _active_fortran_v3_reduced_permc_candidates(*, requested: str, factor_kind: str) -> tuple[str, ...]:
    """Return SuperLU ordering candidates for the active Fortran-v3 factor.

    ``RCM`` is implemented by an explicit symmetric permutation before calling
    SuperLU with ``NATURAL`` ordering. This mirrors SFINCS Fortran v3's PETSc
    serial sparse-direct fallback, where ``MATORDERINGRCM`` is requested for
    the preconditioner factor. The unset default keeps the previously measured
    NATURAL-first behavior for compatibility until the RCM path is promoted by
    full-grid evidence.
    """

    valid = ("RCM", "NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD")
    requested_use = str(requested or "").strip().upper()
    if requested_use in valid:
        return (requested_use,)
    if requested_use and requested_use not in {"AUTO", "DEFAULT"}:
        return ("COLAMD",)
    if str(factor_kind).strip().lower() == "lu":
        return ("NATURAL", "COLAMD")
    return ("COLAMD",)


def _active_fortran_v3_reduced_preconditioner_matrix(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    regularization: float,
    preconditioner_x: int | None = None,
    preconditioner_xi: int | None = None,
    preconditioner_species: int | None = None,
    preconditioner_x_min_l: int | None = None,
) -> tuple[Any, dict[str, object]]:
    """Return a CSR matrix with Fortran-v3-style preconditioner sparsening.

    This works from the already assembled active CSR so it is conservative and
    testable. It should be replaced by direct term-level assembly once the
    Fortran ``whichMatrix=0`` stencil is fully ported.
    """

    matrix_csr = matrix.tocsr().astype(np.float64)
    active = (
        np.arange(int(matrix_csr.shape[0]), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    active_layout = RHS1ActiveBlockLayout.from_layout(layout, active)
    if active.size != int(matrix_csr.shape[0]):
        raise ValueError("active_indices size must match active matrix shape")
    if matrix_csr.shape[0] != matrix_csr.shape[1]:
        raise ValueError("active matrix must be square")

    matrix_coo = matrix_csr.tocoo(copy=False)
    row_full = active[np.asarray(matrix_coo.row, dtype=np.int64)]
    col_full = active[np.asarray(matrix_coo.col, dtype=np.int64)]
    kinetic_entry = (row_full < int(layout.f_size)) & (col_full < int(layout.f_size))
    keep = np.ones(matrix_coo.nnz, dtype=bool)
    preconditioner_x_use = (
        int(preconditioner_x)
        if preconditioner_x is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X", 1))
    )
    preconditioner_xi_use = (
        int(preconditioner_xi)
        if preconditioner_xi is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_XI", 1))
    )
    preconditioner_species_use = (
        int(preconditioner_species)
        if preconditioner_species is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_SPECIES", 1))
    )
    preconditioner_x_min_l_use = max(
        0,
        int(preconditioner_x_min_l)
        if preconditioner_x_min_l is not None
        else int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X_MIN_L", 0)),
    )
    dropped: dict[str, int] = {
        "x_nonlocal": 0,
        "x_unsupported": 0,
        "ell_two": 0,
        "ell_outside_support": 0,
        "species_cross": 0,
    }

    if np.any(kinetic_entry):
        kinetic_positions = np.flatnonzero(kinetic_entry)
        row_decoded = layout.decode_kinetic_indices(row_full[kinetic_positions])
        col_decoded = layout.decode_kinetic_indices(col_full[kinetic_positions])

        if int(preconditioner_species_use) > 0:
            mask = row_decoded.species != col_decoded.species
            keep[kinetic_positions[mask]] = False
            dropped["species_cross"] = int(np.count_nonzero(mask))

        if int(preconditioner_x_use) > 0:
            row_x = row_decoded.x.astype(np.int64, copy=False)
            col_x = col_decoded.x.astype(np.int64, copy=False)
            if int(preconditioner_x_use) == 1:
                x_allowed = row_x == col_x
            elif int(preconditioner_x_use) == 2:
                x_allowed = col_x >= row_x
            elif int(preconditioner_x_use) in {3, 5}:
                x_allowed = np.abs(row_x - col_x) <= 1
            elif int(preconditioner_x_use) == 4:
                x_allowed = (col_x == row_x) | (col_x == row_x + 1)
            else:
                x_allowed = row_x == col_x
            if int(preconditioner_x_min_l_use) > 0:
                x_gate = row_decoded.ell >= int(preconditioner_x_min_l_use)
                x_allowed = np.where(x_gate, x_allowed, True)
            mask = ~x_allowed
            keep[kinetic_positions[mask]] = False
            dropped["x_unsupported"] = int(np.count_nonzero(mask))
            dropped["x_nonlocal"] = int(np.count_nonzero(mask & (row_decoded.x != col_decoded.x)))

        ell_distance = np.abs(row_decoded.ell - col_decoded.ell)
        ell_radius = 1 if int(preconditioner_xi_use) > 0 else 2
        mask = ell_distance > int(ell_radius)
        if int(preconditioner_xi_use) > 0:
            dropped["ell_two"] = int(np.count_nonzero(ell_distance == 2))
        if np.any(mask):
            keep[kinetic_positions[mask]] = False
            dropped["ell_outside_support"] = int(np.count_nonzero(mask))

    reduced = sp.coo_matrix(
        (matrix_coo.data[keep], (matrix_coo.row[keep], matrix_coo.col[keep])),
        shape=matrix_csr.shape,
        dtype=np.float64,
    ).tocsr()
    reduced.sum_duplicates()
    reduced.eliminate_zeros()
    diagonal_shift = float(
        _env_float(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT",
            max(float(abs(regularization)), 1.0e-14),
        )
    )
    if diagonal_shift > 0.0:
        scale = max(float(np.max(np.abs(reduced.data))) if reduced.nnz else 0.0, 1.0)
        reduced = reduced + float(diagonal_shift * scale) * sp.eye(reduced.shape[0], dtype=np.float64, format="csr")
    metadata = {
        "matrix_shape": tuple(int(v) for v in matrix_csr.shape),
        "matrix_nnz": int(matrix_csr.nnz),
        "reduced_matrix_nnz": int(reduced.nnz),
        "reduced_nnz_ratio": float(reduced.nnz / max(int(matrix_csr.nnz), 1)),
        "dropped_entries": {str(k): int(v) for k, v in dropped.items()},
        "diagonal_shift": float(diagonal_shift),
        "preconditioner_x": int(preconditioner_x_use),
        "preconditioner_xi": int(preconditioner_xi_use),
        "preconditioner_species": int(preconditioner_species_use),
        "preconditioner_x_min_l": int(preconditioner_x_min_l_use),
        "fortran_reduced_filter": "layout_decoded_supports",
        "active_layout": active_layout.to_dict(),
        "fortran_v3_source": (
            "solver.F90 uses GMRES+PCLU on whichMatrix=0; populateMatrix.F90 "
            "drops off-by-2 ell terms when preconditioner_xi=1 and createGrids.F90 "
            "defaults preconditioner_x=1 to diagonal x derivative stencils."
        ),
        "implementation_note": (
            "This is an active-CSR reduction of the exact operator. The next "
            "production step is direct term-level assembly of the v3 whichMatrix=0 "
            "operator to avoid first materializing the full true CSR."
        ),
    }
    return reduced, metadata


def select_active_fortran_v3_reduced_support_mode_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    rhs: Any,
    true_matvec: Callable[[np.ndarray], Any],
    candidates: str | Sequence[str] = "current,x0,xmin_l2,species0",
    max_candidates: int = 4,
    min_improvement_ratio: float = 1.05,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
) -> tuple[RHS1StructuredFullCSRPreconditioner, dict[str, object]]:
    """Select a Fortran-v3 reduced support mode using the true residual.

    The Fortran-v3-style active matrix has several support knobs controlling
    radial, pitch-angle, and species couplings.  Large RHSMode=1 cases can be
    very sensitive to those choices, so this bounded selector evaluates a small
    candidate set against the actual operator used by the outer Krylov solve.
    A denser support is promoted only when its one-application true residual is
    better than the current support by ``min_improvement_ratio`` and still fits
    under ``max_factor_nbytes``.
    """

    t0 = time.perf_counter()
    rhs_np = np.asarray(rhs, dtype=np.float64).reshape((-1,))
    rhs_norm = float(np.linalg.norm(rhs_np))
    candidate_specs = _parse_active_fortran_v3_support_mode_candidates(
        candidates=candidates,
        max_candidates=max_candidates,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )
    candidate_entries: list[dict[str, object]] = []
    first_selected_pc: RHS1StructuredFullCSRPreconditioner | None = None
    baseline_pc: RHS1StructuredFullCSRPreconditioner | None = None
    baseline_residual: float | None = None
    best_pc: RHS1StructuredFullCSRPreconditioner | None = None
    best_residual = float("inf")
    best_label: str | None = None
    early_stop_reason: str | None = None

    for spec_index, spec in enumerate(candidate_specs):
        candidate_start = time.perf_counter()
        try:
            pc = _build_active_fortran_v3_reduced_sparse_factor_preconditioner(
                matrix=matrix,
                layout=layout,
                active_indices=active_indices,
                requested_kind=requested_kind,
                regularization=float(regularization),
                max_factor_nbytes=int(max_factor_nbytes),
                t0=candidate_start,
                preconditioner_x=int(spec["preconditioner_x"]),
                preconditioner_xi=int(spec["preconditioner_xi"]),
                preconditioner_species=int(spec["preconditioner_species"]),
                preconditioner_x_min_l=int(spec["preconditioner_x_min_l"]),
            )
            residual_after: float | None = None
            improvement_ratio: float | None = None
            if bool(pc.selected) and pc.operator is not None:
                if first_selected_pc is None:
                    first_selected_pc = pc
                y_np = np.asarray(pc.operator.matvec(rhs_np), dtype=np.float64).reshape((-1,))
                true_y = np.asarray(true_matvec(y_np), dtype=np.float64).reshape((-1,))
                residual_after = float(np.linalg.norm(rhs_np - true_y))
                if rhs_norm > 0.0 and np.isfinite(float(residual_after)):
                    improvement_ratio = float(rhs_norm / max(float(residual_after), 1.0e-300))
                if bool(spec["is_current"]):
                    baseline_pc = pc
                    baseline_residual = float(residual_after)
                if np.isfinite(float(residual_after)) and float(residual_after) < float(best_residual):
                    best_pc = pc
                    best_residual = float(residual_after)
                    best_label = str(spec["label"])
            entry = {
                "label": str(spec["label"]),
                "is_current": bool(spec["is_current"]),
                "selected": bool(pc.selected),
                "kind": str(pc.kind),
                "reason": str(pc.reason),
                "setup_s": float(pc.setup_s),
                "residual_after": None if residual_after is None else float(residual_after),
                "improvement_ratio": None if improvement_ratio is None else float(improvement_ratio),
                "preconditioner_x": int(spec["preconditioner_x"]),
                "preconditioner_xi": int(spec["preconditioner_xi"]),
                "preconditioner_species": int(spec["preconditioner_species"]),
                "preconditioner_x_min_l": int(spec["preconditioner_x_min_l"]),
                "factor_nbytes_actual": pc.metadata.get("factor_nbytes_actual"),
                "factor_nbytes_estimate": pc.metadata.get("factor_nbytes_estimate"),
                "reduced_matrix_nnz": pc.metadata.get("reduced_matrix_nnz"),
            }
            if (
                bool(spec["is_current"])
                and bool(pc.selected)
                and _active_fortran_v3_support_mode_dropped_no_entries(pc.metadata)
                and spec_index + 1 < len(candidate_specs)
            ):
                early_stop_reason = "current_support_dropped_no_entries"
                entry["early_stop_reason"] = early_stop_reason
                candidate_entries.append(entry)
                break
        except Exception as exc:  # noqa: BLE001
            pc = None
            entry = {
                "label": str(spec["label"]),
                "is_current": bool(spec["is_current"]),
                "selected": False,
                "reason": f"support_mode_candidate_failed:{type(exc).__name__}",
                "error": str(exc),
                "setup_s": max(0.0, time.perf_counter() - candidate_start),
                "residual_after": None,
                "improvement_ratio": None,
                "preconditioner_x": int(spec["preconditioner_x"]),
                "preconditioner_xi": int(spec["preconditioner_xi"]),
                "preconditioner_species": int(spec["preconditioner_species"]),
                "preconditioner_x_min_l": int(spec["preconditioner_x_min_l"]),
            }
        candidate_entries.append(entry)

    min_ratio = max(1.0, float(min_improvement_ratio))
    accepted_nonbaseline = False
    selected_pc = baseline_pc if baseline_pc is not None else first_selected_pc
    selected_label = "current" if baseline_pc is not None else None
    if best_pc is not None and np.isfinite(float(best_residual)):
        if baseline_residual is None:
            selected_pc = best_pc
            selected_label = best_label
            accepted_nonbaseline = bool(best_label != "current")
        elif float(best_residual) <= float(baseline_residual) / float(min_ratio):
            selected_pc = best_pc
            selected_label = best_label
            accepted_nonbaseline = bool(best_label != "current")
    if selected_pc is None:
        selected_pc = RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind=str(requested_kind),
            reason="support_mode_preflight_no_candidate_selected",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={},
        )
    metadata = {
        "selected": bool(selected_pc.selected),
        "selection_kind": "active_fortran_v3_support_mode_preflight",
        "requested_kind": str(requested_kind),
        "rhs_norm": float(rhs_norm),
        "max_candidates": int(max_candidates),
        "min_improvement_ratio": float(min_ratio),
        "candidate_specs": tuple(str(spec["label"]) for spec in candidate_specs),
        "candidates": tuple(candidate_entries),
        "baseline_residual_after": baseline_residual,
        "best_residual_after": None if best_pc is None else float(best_residual),
        "selected_candidate": selected_label,
        "accepted_nonbaseline": bool(accepted_nonbaseline),
        "early_stop_reason": early_stop_reason,
        "setup_s": max(0.0, time.perf_counter() - t0),
    }
    selected_metadata = dict(selected_pc.metadata)
    selected_metadata["support_mode_preflight"] = metadata
    return (
        RHS1StructuredFullCSRPreconditioner(
            operator=selected_pc.operator,
            selected=bool(selected_pc.selected),
            kind=str(selected_pc.kind),
            reason=str(selected_pc.reason),
            setup_s=float(selected_pc.setup_s),
            metadata=selected_metadata,
        ),
        metadata,
    )


def _parse_active_fortran_v3_support_mode_candidates(
    *,
    candidates: str | Sequence[str],
    max_candidates: int,
    preconditioner_x: int,
    preconditioner_xi: int,
    preconditioner_species: int,
    preconditioner_x_min_l: int,
) -> tuple[dict[str, object], ...]:
    if isinstance(candidates, str):
        raw_specs = [item.strip() for item in candidates.split(",") if item.strip()]
    else:
        raw_specs = [str(item).strip() for item in candidates if str(item).strip()]
    if not raw_specs:
        raw_specs = ["current"]
    if raw_specs[0].strip().lower().replace("-", "_") not in {"current", "base", "default"}:
        raw_specs.insert(0, "current")

    current_modes = {
        "preconditioner_x": int(preconditioner_x),
        "preconditioner_xi": int(preconditioner_xi),
        "preconditioner_species": int(preconditioner_species),
        "preconditioner_x_min_l": int(preconditioner_x_min_l),
    }
    parsed: list[dict[str, object]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for raw_spec in raw_specs:
        spec_l = str(raw_spec).strip().lower().replace("-", "_")
        modes = dict(current_modes)
        is_current = spec_l in {"current", "base", "default"}
        if not is_current:
            for token in spec_l.replace("+", ":").replace(";", ":").split(":"):
                token = token.strip()
                if not token:
                    continue
                _apply_active_fortran_v3_support_mode_token(modes, token)
        key = (
            int(modes["preconditioner_x"]),
            int(modes["preconditioner_xi"]),
            int(modes["preconditioner_species"]),
            int(modes["preconditioner_x_min_l"]),
        )
        if key in seen:
            continue
        seen.add(key)
        parsed.append(
            {
                "label": "current" if is_current else spec_l,
                "is_current": bool(is_current),
                **modes,
            }
        )
        if len(parsed) >= max(1, int(max_candidates)):
            break
    return tuple(parsed)


def _active_fortran_v3_support_mode_dropped_no_entries(metadata: dict[str, object]) -> bool:
    dropped = metadata.get("dropped_entries")
    if not isinstance(dropped, dict):
        return False
    try:
        return sum(int(value) for value in dropped.values()) == 0
    except (TypeError, ValueError):
        return False


def _apply_active_fortran_v3_support_mode_token(modes: dict[str, int], token: str) -> None:
    token_l = token.strip().lower().replace("-", "_")
    if token_l.startswith("x_min_l="):
        modes["preconditioner_x_min_l"] = max(0, int(token_l.split("=", 1)[1]))
    elif token_l.startswith("xmin_l="):
        modes["preconditioner_x_min_l"] = max(0, int(token_l.split("=", 1)[1]))
    elif token_l.startswith("xmin_l") and token_l[6:].isdigit():
        modes["preconditioner_x_min_l"] = max(0, int(token_l[6:]))
    elif token_l.startswith("x_min_l") and token_l[7:].isdigit():
        modes["preconditioner_x_min_l"] = max(0, int(token_l[7:]))
    elif token_l.startswith("xi="):
        modes["preconditioner_xi"] = int(token_l.split("=", 1)[1])
    elif token_l.startswith("xi") and token_l[2:].isdigit():
        modes["preconditioner_xi"] = int(token_l[2:])
    elif token_l.startswith("species="):
        modes["preconditioner_species"] = int(token_l.split("=", 1)[1])
    elif token_l.startswith("species") and token_l[7:].isdigit():
        modes["preconditioner_species"] = int(token_l[7:])
    elif token_l.startswith("x="):
        modes["preconditioner_x"] = int(token_l.split("=", 1)[1])
    elif token_l.startswith("x") and token_l[1:].isdigit():
        modes["preconditioner_x"] = int(token_l[1:])
    else:
        raise ValueError(f"unsupported Fortran-v3 support-mode candidate token: {token!r}")


def _build_active_global_sparse_factor_preconditioner(
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


def _build_active_scaled_sparse_factor_preconditioner(
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


def _build_active_projected_diagonal_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a cheap active kinetic-diagonal plus global-tail Schur split.

    The direct-tail Fortran-reduced RHSMode=1 matrix is ordered as active
    kinetic unknowns followed by the retained global constraint/source rows.
    This preconditioner mirrors the full-system ``diagonal_schur`` split
    without materializing or factoring the large active kinetic block.
    """

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
        return _build_jacobi_preconditioner(
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
    inv_f, diag_meta = _safe_inverse_diagonal(diag[:n_f], regularization=regularization)
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


def _build_active_projected_xell_kinetic_line_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_kinetic_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build dense active ``(x, ell)`` line inverses for a projected kinetic block.

    The full-layout native ``x_ell`` factor cannot be used after active-DOF
    projection because each fixed ``(species, theta, zeta)`` line may contain a
    reduced pitch set.  This host/native bridge keeps the same physics split but
    groups the active kinetic rows by their original line identity and factors
    only those compact line blocks.  It is memory-gated before any dense block
    allocation and is currently used as an opt-in kinetic base for the global
    field-split Schur preconditioner.
    """

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


def _build_active_projected_angular_line_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_kinetic_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build active angular-line inverses grouped by ``(species, x, ell)``.

    This is the complementary line factor to the native ``x_ell`` preconditioner.
    Instead of solving dense speed/pitch blocks at fixed angular location, it
    solves dense ``(theta, zeta)`` blocks at fixed species, speed, and pitch.
    Geometry-rich QA/QH residuals often live in angular coupling, so this path
    gives the field-split/coarse solvers a stronger local inverse without
    materializing global sparse LU factors.
    """

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
        return _build_jacobi_preconditioner(
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
        inv_tail, tail_metadata = _safe_inverse_diagonal(
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


def _build_active_projected_native_indexed_schwarz_preconditioner(
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

    This combines complementary local physics blocks: fixed
    ``(species, theta, zeta)`` speed/pitch lines and fixed
    ``(species, x, ell)`` angular lines.  The block inverses are emitted as a
    padded indexed PyTree so the apply step is a device-compatible gather,
    dense batched solve, and scatter-add with overlap normalization.
    """

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    import jax.numpy as jnp  # noqa: PLC0415

    from .native_block_factor import (  # noqa: PLC0415
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
        return _build_jacobi_preconditioner(
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
        inv_tail, tail_metadata = _safe_inverse_diagonal(
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


def _build_active_projected_global_field_split_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
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
            base = _build_native_xell_kinetic_preconditioner(
                matrix=kinetic_matrix,
                layout=kinetic_layout,
                requested_kind=base_kind,
                regularization=regularization,
                max_factor_nbytes=base_budget,
                t0=t0,
            )
        else:
            base = _build_active_projected_xell_kinetic_line_preconditioner(
                matrix=kinetic_matrix,
                layout=layout,
                active_kinetic_indices=kinetic_active,
                requested_kind=base_kind,
                regularization=regularization,
                max_factor_nbytes=base_budget,
                t0=t0,
            )
    else:
        base = build_active_projected_rhs1_full_csr_preconditioner(
            matrix=kinetic_matrix,
            layout=layout,
            active_indices=kinetic_active,
            kind=base_kind,
            max_factor_nbytes=base_budget,
            regularization=regularization,
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


def _build_active_projected_multiline_field_split_base_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
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
    xell = _build_active_projected_global_field_split_schur_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_global_field_split_schur",
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
        base_kind_override="active_native_xell",
    )
    angular = _build_active_projected_global_field_split_schur_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_global_field_split_schur",
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
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


def _build_active_projected_bounded_native_stack_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
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

    base_budget_fraction = float(
        _env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_BASE_BUDGET_FRACTION", 0.75)
    )
    base_budget_fraction = min(max(float(base_budget_fraction), 0.05), 1.0)
    base_budget = max(1, int(float(max_factor_nbytes) * base_budget_fraction))
    base = _build_active_projected_multiline_field_split_base_preconditioner(
        matrix=matrix_csr,
        layout=layout,
        active_indices=active_indices,
        regularization=regularization,
        max_factor_nbytes=base_budget,
        t0=t0,
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
    schwarz_requested = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ", False)
    schwarz_max_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ_MAX_SIZE", 100_000)
    schwarz: RHS1StructuredFullCSRPreconditioner | None = None
    schwarz_nbytes = 0
    schwarz_reason = "disabled"
    if bool(schwarz_requested):
        if int(schwarz_max_size) > 0 and int(matrix_csr.shape[0]) > int(schwarz_max_size):
            schwarz_reason = f"size_exceeded:{int(matrix_csr.shape[0])}>{int(schwarz_max_size)}"
        else:
            remaining_budget = max(1, int(max_factor_nbytes) - int(base_nbytes))
            schwarz = _build_active_projected_overlap_schwarz_preconditioner(
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

    config = _coarse_residual_config(layout)
    max_coarse_size = _env_int(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE",
        _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", 640),
    )
    max_coarse_size = max(1, int(max_coarse_size))
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
    basis, adaptive_metadata = _append_adaptive_residual_basis_csc(
        matrix=matrix_csr,
        base_operator=adaptive_base_operator,
        basis=basis,
        max_total_columns=int(max_coarse_size),
    )

    az_basis = (matrix_csr @ basis).tocsc()
    coarse_solver_mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_SOLVER", "least_squares")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver_mode in {"galerkin", "petrov_galerkin", "ztaz"}:
        coarse_solver_mode = "galerkin"
        coarse = np.asarray((basis.T @ az_basis).toarray(), dtype=np.float64)
    else:
        coarse_solver_mode = "least_squares"
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


def _build_active_fortran_v3_reduced_native_stack_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any | None,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build the production-named bounded native stack for direct-tail solves.

    ``active_fortran_v3_reduced_lu`` is the robust full-grid incumbent, but it
    stores a large monolithic sparse factor.  This production candidate keeps
    the same active direct-tail operator and residual gate while replacing the
    global serial factor by bounded native line factors, optional local
    Schwarz, and a compact coupled coarse residual equation.
    """

    base = _build_active_projected_bounded_native_stack_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind=str(requested_kind),
        regularization=regularization,
        max_factor_nbytes=max_factor_nbytes,
        t0=t0,
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
        ordering = RHS1ActiveFieldSplitOrdering.from_layout(layout, active_np)
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


def _build_active_projected_filtered_sparse_factor_preconditioner(
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

    from .explicit_sparse import deterministic_sparse_probe_matrix  # noqa: PLC0415

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

    from .explicit_sparse import (  # noqa: PLC0415
        admit_sparse_factor_against_operator,
        analyze_sparse_symbolic_structure,
        factorize_host_sparse_operator,
    )

    matrix_csr = matrix.tocsr()
    active_size = int(matrix_csr.shape[0])
    max_active_size = int(
        _env_int(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_ACTIVE_SIZE",
            300_000,
        )
    )
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

    ordering_kind = (
        os.environ.get(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ORDERING",
            "rcm",
        )
        .strip()
        .lower()
    )
    block_size = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BLOCK_SIZE", 2048)),
    )
    max_permutation_size = max(
        1,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_PERMUTATION_SIZE",
                300_000,
            )
        ),
    )
    separator_cols = max(
        0,
        int(
            _env_int(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_SEPARATOR_COLS",
                512,
            )
        ),
    )
    boundary_width = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BOUNDARY_WIDTH", 1)),
    )
    high_degree_cols = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_HIGH_DEGREE_COLS", 128)),
    )
    regularization_rel = max(
        0.0,
        float(
            _env_float(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_REGULARIZATION_REL",
                max(float(abs(regularization)), 1.0e-12),
            )
        ),
    )
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
    prefill_safety = max(
        1.0,
        float(
            _env_float(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_PREFILL_SAFETY_FACTOR",
                4.0,
            )
        ),
    )
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
        probe_count=max(
            1,
            int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_PROBES", 4)),
        ),
        max_relative_residual=max(
            0.0,
            float(
                _env_float(
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MAX_RELATIVE_RESIDUAL",
                    1.0e-2,
                )
            ),
        ),
        min_improvement_vs_identity=max(
            0.0,
            float(
                _env_float(
                    "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MIN_IMPROVEMENT",
                    1.0,
                )
            ),
        ),
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
        ordering = RHS1ActiveFieldSplitOrdering.from_layout(layout, active_np)
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
    requested_kind_l = str(requested_kind).strip().lower().replace("-", "_")
    is_multiline = "multiline" in requested_kind_l or "xell_angular" in requested_kind_l
    is_angular_only = "angular" in requested_kind_l and not is_multiline
    is_coupled_kinetic = "coupled_kinetic" in requested_kind_l or "dominant_kinetic" in requested_kind_l
    output_kind = "active_native_xell_field_split_sparse_coarse"
    if is_coupled_kinetic:
        output_kind = "active_coupled_kinetic_field_split_sparse_coarse"
    if is_angular_only:
        output_kind = "active_angular_line_field_split_sparse_coarse"
    if is_multiline:
        output_kind = "active_multiline_field_split_sparse_coarse"
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

    requested_base_kind = "active_multiline_xell_angular"
    if bool(is_coupled_kinetic):
        requested_base_kind = "active_coupled_kinetic_block"
    elif not is_multiline:
        requested_base_kind = (
            "active_angular_line"
            if is_angular_only
            else os.environ.get(
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LINE_FIELD_SPLIT_BASE",
                "active_native_xell",
            )
        )
        requested_base_kind = str(requested_base_kind).strip().lower().replace("-", "_") or "active_native_xell"
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
    max_coarse_size = _env_int(
        (
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE"
            if bool(is_coupled_kinetic)
            else "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE"
        ),
        _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640),
    )
    max_coarse_size = max(1, int(max_coarse_size))
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

    az_basis = (matrix_csr @ basis).tocsc()
    coarse_solver_mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER", "least_squares")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver_mode in {"galerkin", "petrov_galerkin", "ztaz"}:
        coarse_solver_mode = "galerkin"
        coarse = np.asarray((basis.T @ az_basis).toarray(), dtype=np.float64)
    else:
        coarse_solver_mode = "least_squares"
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
            probe_count=max(
                1,
                int(
                    _env_int(
                        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_PROBES",
                        4,
                    )
                ),
            ),
            max_relative_residual=max(
                0.0,
                float(
                    _env_float(
                        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL",
                        1.0e-2,
                    )
                ),
            ),
            min_improvement_vs_identity=max(
                0.0,
                float(
                    _env_float(
                        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MIN_IMPROVEMENT",
                        1.0,
                    )
                ),
            ),
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

    requested_norm = str(requested_kind).strip().lower().replace("-", "_")
    if "scaled_ilu" in requested_norm or "equilibrated_ilu" in requested_norm or "rowcol_ilu" in requested_norm:
        base_kind = "active_scaled_ilu"
    elif "schwarz" in requested_norm or "ras" in requested_norm:
        base_kind = "active_overlap_schwarz"
    elif "filtered" in requested_norm:
        base_kind = "active_filtered_sparse_factor"
    elif "xblock" in requested_norm:
        base_kind = "active_xblock"
    else:
        base_kind = "active_diagonal_schur"
    output_kind = (
        "active_filtered_sparse_coarse"
        if str(base_kind) == "active_filtered_sparse_factor"
        else "active_tail_sparse_coarse"
    )
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
    max_coarse_size = _env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", 640)
    max_coarse_size = max(1, int(max_coarse_size))
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
    default_coarse_solver = "least_squares" if str(base_kind) == "active_filtered_sparse_factor" else "galerkin"
    coarse_solver_mode = (
        os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_SOLVER", default_coarse_solver)
        .strip()
        .lower()
        .replace("-", "_")
    )
    if coarse_solver_mode in {"least_squares", "normal", "normal_equations"}:
        coarse_solver_mode = "least_squares"
        coarse = np.asarray((az_basis.T @ az_basis).toarray(), dtype=np.float64)
    else:
        coarse_solver_mode = "galerkin"
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


def _build_active_projected_xblock_preconditioner(
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
        base = _build_jacobi_preconditioner(
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
            full_idx = _xblock_tz_low_l_indices(layout=layout, species=species, x=x, lmax=lmax)
            positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=full_idx)
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
                row_scale, row_meta = _sparse_equilibration_scale(
                    block_raw,
                    axis=1,
                    norm=scale_norm,
                    max_scale=max_scale,
                )
                block_row_scaled = block_raw.multiply(row_scale[:, None]).tocsc()
                col_scale, col_meta = _sparse_equilibration_scale(
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
            sparse_estimate = _estimate_spilu_factor_nbytes(matrix=block, fill_factor=fill_factor)
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
            factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
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


def _build_active_projected_overlap_schwarz_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    active_indices: Any,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a restricted additive-Schwarz residual correction.

    The active projected RHSMode=1 system is not rectangular in the native
    ``(species, x, xi, theta, zeta)`` coordinates because inactive pitch rows
    have been removed.  This host-only preconditioner reconstructs overlapping
    speed-space patches in those original coordinates, solves the residual
    equation on each patch, and scatters only the center speed block.  That is a
    restricted additive-Schwarz (RAS) update, so neighboring speed blocks overlap
    without double-counting the correction.
    """

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

    base = _build_jacobi_preconditioner(
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
            core_full = _xblock_tz_low_l_indices(layout=layout, species=species, x=x_center, lmax=lmax)
            core_positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=core_full)
            if core_positions.size == 0:
                continue
            x_min = max(0, int(x_center) - int(radius))
            x_max = min(int(layout.n_x) - 1, int(x_center) + int(radius))
            patch_full_parts = [
                _xblock_tz_low_l_indices(layout=layout, species=species, x=x_patch, lmax=lmax)
                for x_patch in range(x_min, x_max + 1)
            ]
            patch_full = np.unique(np.concatenate(patch_full_parts).astype(np.int64, copy=False))
            positions = _active_positions_for_full_indices(active_indices=active_np, full_indices=patch_full)
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
            sparse_estimate = _estimate_spilu_factor_nbytes(matrix=block, fill_factor=fill_estimate)
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
            factor_nbytes = int(_sparse_lu_factor_nbytes(factor))
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


def _build_jacobi_preconditioner(
    *,
    matrix: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
    reason: str,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    inv_diag, diag_meta = _safe_inverse_diagonal(matrix.diagonal(), regularization=regularization)
    operator = LinearOperator(
        matrix.shape,
        matvec=lambda x: inv_diag * np.asarray(x, dtype=np.float64).reshape((-1,)),
        dtype=np.float64,
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="jacobi",
        reason=reason,
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={"requested_kind": str(requested_kind), **diag_meta},
    )


def _build_native_xell_kinetic_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build an opt-in JAX-native ``x_ell`` kinetic-line preconditioner."""

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    from .rhs1_full_csr_kinetic_pc import build_rhs1_full_csr_kinetic_preconditioner  # noqa: PLC0415

    candidate = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="x_ell",
        max_candidate_nbytes=int(max_factor_nbytes),
        regularization=float(regularization),
        tail_policy="jacobi",
        build_native_factor=True,
    )
    if not bool(candidate.selected) or candidate.native_factor is None:
        metadata = dict(candidate.metadata)
        metadata["requested_kind"] = str(requested_kind)
        metadata["native_factor_available"] = False
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="native_xell",
            reason=str(candidate.reason),
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata=metadata,
        )

    def apply(rhs: Any) -> np.ndarray:
        rhs_vec = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if rhs_vec.shape != (int(layout.total_size),):
            raise ValueError(f"rhs must have shape {(int(layout.total_size),)}, got {rhs_vec.shape}")
        return np.array(candidate.apply_native(rhs_vec), dtype=np.float64, copy=True)

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "requested_kind": str(requested_kind),
            "native_factor_available": True,
            "backend": "jax_native_x_ell",
            "note": "opt_in_probe_not_auto_default",
        }
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="native_xell",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata=metadata,
    )


def _build_native_xell_tail_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_schur_size: int,
    max_factor_nbytes: int,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a native ``x_ell`` kinetic inverse plus dense tail Schur factor."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    from .rhs1_full_csr_kinetic_pc import build_rhs1_full_csr_kinetic_preconditioner  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    if tail_size <= 0:
        return _build_native_xell_kinetic_preconditioner(
            matrix=matrix,
            layout=layout,
            requested_kind=requested_kind,
            regularization=regularization,
            max_factor_nbytes=max_factor_nbytes,
            t0=t0,
        )
    if tail_size > int(max_schur_size):
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="native_xell_tail_schur",
            reason=f"schur_tail_size_exceeded:{tail_size}>{int(max_schur_size)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "tail_size": int(tail_size),
                "max_schur_size": int(max_schur_size),
            },
        )

    schur_nbytes = int(tail_size * tail_size * np.dtype(np.float64).itemsize)
    work_nbytes = int(2 * n_f * np.dtype(np.float64).itemsize)
    kinetic_budget = int(max_factor_nbytes) - int(schur_nbytes + work_nbytes)
    if kinetic_budget <= 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="native_xell_tail_schur",
            reason=f"native_xell_tail_schur_budget_exceeded:{schur_nbytes + work_nbytes}>{int(max_factor_nbytes)}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={
                "requested_kind": str(requested_kind),
                "tail_size": int(tail_size),
                "schur_nbytes": int(schur_nbytes),
                "work_vector_nbytes": int(work_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
            },
        )

    candidate = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="x_ell",
        max_candidate_nbytes=int(kinetic_budget),
        regularization=float(regularization),
        tail_policy="identity",
        build_native_factor=True,
    )
    if not bool(candidate.selected) or candidate.native_factor is None:
        metadata = dict(candidate.metadata)
        metadata.update(
            {
                "requested_kind": str(requested_kind),
                "native_factor_available": False,
                "schur_nbytes": int(schur_nbytes),
                "work_vector_nbytes": int(work_nbytes),
                "max_factor_nbytes": int(max_factor_nbytes),
                "kinetic_factor_budget_nbytes": int(kinetic_budget),
            }
        )
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="native_xell_tail_schur",
            reason=str(candidate.reason),
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata=metadata,
        )

    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        full_rhs = np.zeros((n_total,), dtype=np.float64)
        full_rhs[:n_f] = np.asarray(vec, dtype=np.float64).reshape((-1,))
        return np.array(candidate.apply_native(full_rhs)[:n_f], dtype=np.float64, copy=True)

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(rhs: Any) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        if arr.shape != (n_total,):
            raise ValueError(f"rhs must have shape {(n_total,)}, got {arr.shape}")
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    metadata = dict(candidate.metadata)
    metadata.update(
        {
            "requested_kind": str(requested_kind),
            "native_factor_available": True,
            "backend": "jax_native_x_ell_tail_schur",
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "schur_nbytes": int(schur.nbytes),
            "work_vector_nbytes": int(work_nbytes),
            "factor_nbytes_actual": int(metadata.get("candidate_nbytes_actual", 0) or 0)
            + int(schur.nbytes)
            + int(work_nbytes),
            "max_factor_nbytes": int(max_factor_nbytes),
            "kinetic_factor_budget_nbytes": int(kinetic_budget),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            "note": "opt_in_probe_not_auto_default",
        }
    )
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="native_xell_tail_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata=metadata,
    )


def _build_diagonal_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    if tail_size <= 0:
        return _build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="no_global_tail",
        )

    diag = matrix.diagonal()
    inv_f, diag_meta = _safe_inverse_diagonal(diag[:n_f], regularization=regularization)
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()
    scaled_u = u.multiply(inv_f[:, None])
    schur = (w - v @ scaled_u).toarray()
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = inv_f * arr[:n_f]
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - inv_f * np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="diagonal_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **diag_meta,
        },
    )


def _build_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    block_size = int(layout.n_zeta)
    n_blocks = n_f // block_size
    inverse_blocks, block_meta = _build_zeta_diagonal_inverse_blocks(
        matrix=matrix,
        n_f=n_f,
        block_size=block_size,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((n_blocks, block_size))
        out = np.einsum("bij,bj->bi", inverse_blocks, flat, optimize=True)
        return out.reshape((-1,))

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(block_size),
            "n_blocks": int(n_blocks),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def _build_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = _build_xi_diagonal_inverse_blocks(
        matrix=matrix,
        layout=layout,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((-1,))
        gathered = flat[block_indices]
        block_values = np.einsum("bij,bj->bi", inverse_blocks, gathered, optimize=True)
        out = np.zeros((n_f,), dtype=np.float64)
        out[block_indices] = block_values
        return out

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="xi_block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(layout.n_xi),
            "n_blocks": int(block_indices.shape[0]),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def _build_x_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = _build_x_xi_diagonal_inverse_blocks(
        matrix=matrix,
        layout=layout,
        regularization=regularization,
    )
    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((-1,))
        gathered = flat[block_indices]
        block_values = np.einsum("bij,bj->bi", inverse_blocks, gathered, optimize=True)
        out = np.zeros((n_f,), dtype=np.float64)
        out[block_indices] = block_values
        return out

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="x_xi_block_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(layout.n_x * layout.n_xi),
            "n_blocks": int(block_indices.shape[0]),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **block_meta,
        },
    )


def _build_xblock_tz_low_l_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    config: dict[str, object],
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inv_diag, diag_meta = _safe_inverse_diagonal(matrix.diagonal()[:n_f], regularization=regularization)
    lmax = int(config["lmax"])
    block_factors: list[Any] = []
    block_indices: list[np.ndarray] = []
    factor_nbytes = 0
    factor_failures = 0
    factor_kind = str(config["factor_kind"])
    drop_tol = float(config["drop_tol"])
    fill_factor = float(config["fill_factor"])
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            indices = _xblock_tz_low_l_indices(layout=layout, species=species, x=x, lmax=lmax)
            block = matrix[indices[:, None], indices].tocsc()
            scale = max(float(np.linalg.norm(np.asarray(block.sum(axis=1)).reshape((-1,)), ord=np.inf)), 1.0)
            reg_abs = float(abs(regularization)) * scale
            if reg_abs > 0.0:
                block = (block + reg_abs * sp.eye(block.shape[0], format="csc", dtype=np.float64)).tocsc()
            if drop_tol > 0.0 and block.nnz:
                block = block.copy()
                block.data[np.abs(block.data) <= drop_tol] = 0.0
                block.eliminate_zeros()
                block = block.tocsc()
            try:
                factor = (
                    spilu(block, drop_tol=drop_tol, fill_factor=fill_factor, permc_spec="COLAMD")
                    if factor_kind == "spilu"
                    else splu(block, permc_spec="COLAMD", diag_pivot_thresh=0.0)
                )
            except RuntimeError:
                factor_failures += 1
                continue
            current_nbytes = _sparse_lu_factor_nbytes(factor)
            if factor_nbytes + current_nbytes > int(max_factor_nbytes):
                return RHS1StructuredFullCSRPreconditioner(
                    operator=None,
                    selected=False,
                    kind="xblock_tz_low_l_schur",
                    reason=f"xblock_tz_low_l_factor_budget_exceeded:{factor_nbytes + current_nbytes}>{int(max_factor_nbytes)}",
                    setup_s=max(0.0, time.perf_counter() - t0),
                    metadata={
                        "factor_nbytes_actual": int(factor_nbytes + current_nbytes),
                        "max_factor_nbytes": int(max_factor_nbytes),
                        "selected_blocks": int(len(block_factors)),
                        "factor_failures": int(factor_failures),
                        **config,
                    },
                )
            factor_nbytes += current_nbytes
            block_factors.append(factor)
            block_indices.append(indices)

    if not block_factors:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="xblock_tz_low_l_schur",
            reason="no_xblock_factors_selected",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"factor_failures": int(factor_failures), **config},
        )

    u = matrix[:n_f, n_f:].tocsr()
    v = matrix[n_f:, :n_f].tocsr()
    w = matrix[n_f:, n_f:].tocsr()

    def apply_f_inverse(vec: Any) -> np.ndarray:
        flat = np.asarray(vec, dtype=np.float64).reshape((-1,))
        out = inv_diag * flat
        for factor, indices in zip(block_factors, block_indices, strict=True):
            out[indices] = factor.solve(flat[indices])
        return out

    schur = np.asarray(w.toarray(), dtype=np.float64)
    u_csc = u.tocsc()
    active_u_columns = 0
    for col_index in range(tail_size):
        start = int(u_csc.indptr[col_index])
        stop = int(u_csc.indptr[col_index + 1])
        if start == stop:
            continue
        active_u_columns += 1
        column = np.zeros((n_f,), dtype=np.float64)
        column[u_csc.indices[start:stop]] = u_csc.data[start:stop]
        schur[:, col_index] -= np.asarray(v @ apply_f_inverse(column), dtype=np.float64).reshape((-1,))
    schur_scale = max(float(np.linalg.norm(schur, ord=np.inf)) if schur.size else 0.0, 1.0)
    schur_regularization = float(abs(regularization)) * schur_scale
    if schur_regularization > 0.0:
        schur = schur + schur_regularization * np.eye(tail_size, dtype=np.float64)
    lu, piv = lu_factor(schur)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_f = apply_f_inverse(arr[:n_f])
        rhs_tail = arr[n_f:] - np.asarray(v @ y_f, dtype=np.float64).reshape((-1,))
        y_tail = lu_solve((lu, piv), rhs_tail)
        y_f = y_f - apply_f_inverse(np.asarray(u @ y_tail, dtype=np.float64).reshape((-1,)))
        return np.concatenate((y_f, np.asarray(y_tail, dtype=np.float64).reshape((-1,))))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if tail_size <= 128:
        cond_estimate = float(np.linalg.cond(schur))
    block_nnz = int(sum(int(factor.L.nnz + factor.U.nnz) for factor in block_factors))
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="xblock_tz_low_l_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(n_f),
            "tail_size": int(tail_size),
            "block_size": int(lmax * layout.n_theta * layout.n_zeta),
            "n_blocks": int(layout.n_species * layout.n_x),
            "selected_blocks": int(len(block_factors)),
            "factor_failures": int(factor_failures),
            "factor_nnz": int(block_nnz),
            "factor_nbytes_actual": int(factor_nbytes),
            "u_nnz": int(u.nnz),
            "v_nnz": int(v.nnz),
            "w_nnz": int(w.nnz),
            "active_u_columns": int(active_u_columns),
            "work_vector_nbytes": int(n_f * np.dtype(np.float64).itemsize),
            "schur_nbytes": int(schur.nbytes),
            "schur_regularization": float(schur_regularization),
            "schur_condition_estimate": cond_estimate,
            **diag_meta,
            **config,
        },
    )


def _build_xblock_tz_low_l_coarse_residual_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    config: dict[str, object],
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    base = _build_xblock_tz_low_l_schur_preconditioner(
        matrix=matrix,
        layout=layout,
        requested_kind=requested_kind,
        regularization=regularization,
        max_factor_nbytes=int(max_factor_nbytes),
        config=config,
        t0=t0,
    )
    if not bool(base.selected) or base.operator is None:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="xblock_tz_low_l_coarse_schur",
            reason=f"base_not_selected:{base.reason}",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )

    basis = _build_coarse_residual_basis_csc(layout=layout, config=config)
    if basis.shape[1] <= 0:
        return RHS1StructuredFullCSRPreconditioner(
            operator=None,
            selected=False,
            kind="xblock_tz_low_l_coarse_schur",
            reason="empty_coarse_basis",
            setup_s=max(0.0, time.perf_counter() - t0),
            metadata={"base_preconditioner": base.to_dict(), **config},
        )
    matrix = matrix.tocsr()
    a_basis = matrix @ basis
    coarse = np.asarray((basis.T @ a_basis).toarray(), dtype=np.float64)
    coarse_scale = max(float(np.linalg.norm(coarse, ord=np.inf)) if coarse.size else 0.0, 1.0)
    coarse_regularization = float(abs(regularization)) * coarse_scale
    if coarse_regularization > 0.0:
        coarse = coarse + coarse_regularization * np.eye(coarse.shape[0], dtype=np.float64)
    lu, piv = lu_factor(coarse)

    def apply(x: Any) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(base.operator.matvec(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(matrix @ y_base, dtype=np.float64).reshape((-1,))
        coarse_rhs = np.asarray(basis.T @ residual, dtype=np.float64).reshape((-1,))
        alpha = lu_solve((lu, piv), coarse_rhs)
        return y_base + np.asarray(basis @ alpha, dtype=np.float64).reshape((-1,))

    operator = LinearOperator(matrix.shape, matvec=apply, dtype=np.float64)
    cond_estimate = None
    if coarse.shape[0] <= 256:
        cond_estimate = float(np.linalg.cond(coarse))
    base_metadata = dict(base.metadata)
    base_nbytes = int(base_metadata.get("factor_nbytes_actual", 0) or 0)
    coarse_nbytes = int(basis.data.nbytes + basis.indices.nbytes + basis.indptr.nbytes + coarse.nbytes)
    surface_modes = _coarse_surface_modes(layout=layout, config=config)
    return RHS1StructuredFullCSRPreconditioner(
        operator=operator,
        selected=True,
        kind="xblock_tz_low_l_coarse_schur",
        reason="complete",
        setup_s=max(0.0, time.perf_counter() - t0),
        metadata={
            "requested_kind": str(requested_kind),
            "kinetic_size": int(layout.f_size),
            "tail_size": int(layout.total_size - layout.f_size),
            "coarse_size": int(basis.shape[1]),
            "coarse_basis_nnz": int(basis.nnz),
            "coarse_basis_nbytes": int(basis.data.nbytes + basis.indices.nbytes + basis.indptr.nbytes),
            "coarse_matrix_nbytes": int(coarse.nbytes),
            "coarse_surface_mode_count": int(len(surface_modes)),
            "coarse_surface_modes": tuple(str(name) for name, _values in surface_modes),
            "coarse_regularization": float(coarse_regularization),
            "coarse_condition_estimate": cond_estimate,
            "base_factor_nbytes_actual": int(base_nbytes),
            "coarse_total_nbytes_actual": int(coarse_nbytes),
            "factor_nbytes_actual": int(base_nbytes + coarse_nbytes),
            "operator_matvecs_per_apply": 1,
            "base_preconditioner": base.to_dict(),
            **config,
        },
    )


def _build_zeta_diagonal_inverse_blocks(
    *,
    matrix: Any,
    n_f: int,
    block_size: int,
    regularization: float,
) -> tuple[np.ndarray, dict[str, object]]:
    n_blocks = int(n_f) // int(block_size)
    inverse_blocks = np.empty((n_blocks, int(block_size), int(block_size)), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block in range(n_blocks):
        start = block * int(block_size)
        stop = start + int(block_size)
        dense = np.asarray(matrix[start:stop, start:stop].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(int(block_size), dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, metadata


def _build_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    block_size = int(layout.n_xi)
    index_rows: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            for theta in range(int(layout.n_theta)):
                for zeta in range(int(layout.n_zeta)):
                    indices = [
                        layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
                        for ell in range(int(layout.n_xi))
                    ]
                    index_rows.append(np.asarray(indices, dtype=np.int64))
    block_indices = np.asarray(index_rows, dtype=np.int64)
    inverse_blocks = np.empty((block_indices.shape[0], block_size, block_size), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block, indices in enumerate(block_indices):
        dense = np.asarray(matrix[indices[:, None], indices].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(block_size, dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_index_nbytes_actual": int(block_indices.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, block_indices, metadata


def _build_x_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    block_size = int(layout.n_x) * int(layout.n_xi)
    index_rows: list[np.ndarray] = []
    for species in range(int(layout.n_species)):
        for theta in range(int(layout.n_theta)):
            for zeta in range(int(layout.n_zeta)):
                indices = [
                    layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
                    for x in range(int(layout.n_x))
                    for ell in range(int(layout.n_xi))
                ]
                index_rows.append(np.asarray(indices, dtype=np.int64))
    block_indices = np.asarray(index_rows, dtype=np.int64)
    inverse_blocks = np.empty((block_indices.shape[0], block_size, block_size), dtype=np.float64)
    regularized_count = 0
    singular_count = 0
    max_block_scale = 0.0
    for block, indices in enumerate(block_indices):
        dense = np.asarray(matrix[indices[:, None], indices].toarray(), dtype=np.float64)
        block_scale = max(float(np.linalg.norm(dense, ord=np.inf)) if dense.size else 0.0, 1.0)
        max_block_scale = max(max_block_scale, block_scale)
        regularization_abs = float(abs(regularization)) * block_scale
        if regularization_abs > 0.0:
            dense = dense + regularization_abs * np.eye(block_size, dtype=np.float64)
            regularized_count += 1
        try:
            inverse_blocks[block] = np.linalg.inv(dense)
        except np.linalg.LinAlgError:
            singular_count += 1
            inverse_blocks[block] = np.linalg.pinv(dense, rcond=max(float(abs(regularization)), 1.0e-14))
    metadata = {
        "block_inverse_nbytes_actual": int(inverse_blocks.nbytes),
        "block_index_nbytes_actual": int(block_indices.nbytes),
        "block_inverse_regularized_count": int(regularized_count),
        "block_inverse_singular_count": int(singular_count),
        "block_inverse_scale_max": float(max_block_scale),
    }
    return inverse_blocks, block_indices, metadata


def _estimate_zeta_block_inverse_nbytes(layout: RHS1BlockLayout) -> int:
    block_size = int(layout.n_zeta)
    n_blocks = int(layout.f_size) // block_size
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def _estimate_xi_block_inverse_nbytes(layout: RHS1BlockLayout) -> int:
    block_size = int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_x) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def _estimate_x_xi_block_inverse_nbytes(layout: RHS1BlockLayout) -> int:
    block_size = int(layout.n_x) * int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def _estimate_spilu_factor_nbytes(*, matrix: Any, fill_factor: float) -> int:
    """Conservative storage estimate for a SuperLU ILU factorization."""

    matrix = matrix.tocsr()
    nnz_estimate = int(np.ceil(max(1.0, float(fill_factor)) * max(1, int(matrix.nnz))))
    data_nbytes = nnz_estimate * np.dtype(np.float64).itemsize
    index_nbytes = nnz_estimate * np.dtype(np.int32).itemsize
    indptr_nbytes = 2 * (int(matrix.shape[0]) + 1) * np.dtype(np.int32).itemsize
    return int(data_nbytes + index_nbytes + indptr_nbytes)


def _sparse_equilibration_scale(
    matrix: Any,
    *,
    axis: int,
    norm: str,
    max_scale: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return clipped inverse sparse row/column norms for equilibration."""

    matrix_csr = matrix.tocsr() if int(axis) == 1 else matrix.tocsc()
    abs_matrix = matrix_csr.copy()
    abs_matrix.data = np.abs(np.asarray(abs_matrix.data, dtype=np.float64))
    norm_l = str(norm).strip().lower()
    if norm_l == "l2":
        squared = matrix_csr.copy()
        squared.data = np.square(np.asarray(squared.data, dtype=np.float64))
        values = np.sqrt(np.asarray(squared.sum(axis=1 if int(axis) == 1 else 0), dtype=np.float64).reshape((-1,)))
    elif norm_l == "max":
        values = np.asarray(abs_matrix.max(axis=1 if int(axis) == 1 else 0).toarray(), dtype=np.float64).reshape((-1,))
    else:
        norm_l = "l1"
        values = np.asarray(abs_matrix.sum(axis=1 if int(axis) == 1 else 0), dtype=np.float64).reshape((-1,))
    scale = np.ones_like(values, dtype=np.float64)
    good = np.isfinite(values) & (values > 0.0)
    scale[good] = 1.0 / values[good]
    max_scale_use = max(1.0, float(max_scale))
    scale = np.clip(scale, 1.0 / max_scale_use, max_scale_use)
    metadata = {
        "axis": "row" if int(axis) == 1 else "column",
        "norm": str(norm_l),
        "size": int(scale.size),
        "norm_min": float(np.min(values)) if values.size else 0.0,
        "norm_max": float(np.max(values)) if values.size else 0.0,
        "scale_min": float(np.min(scale)) if scale.size else 0.0,
        "scale_max": float(np.max(scale)) if scale.size else 0.0,
        "zero_or_invalid_norm_count": int(np.count_nonzero(~good)),
    }
    return scale, metadata


def _xblock_tz_low_l_config(layout: RHS1BlockLayout) -> dict[str, object]:
    lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_LMAX", 8)
    lmax = max(1, min(int(layout.n_xi), int(lmax)))
    factor_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FACTOR_KIND", "splu").strip().lower()
    if factor_kind not in {"splu", "spilu"}:
        factor_kind = "splu"
    return {
        "lmax": int(lmax),
        "drop_tol": float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_DROP_TOL", 0.0)),
        "fill_factor": float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_FILL_FACTOR", 8.0)),
        "factor_kind": factor_kind,
    }


def _coarse_residual_config(layout: RHS1BlockLayout) -> dict[str, object]:
    config = _xblock_tz_low_l_config(layout)
    coarse_lmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_LMAX", min(4, int(layout.n_xi)))
    coarse_lmax = max(1, min(int(layout.n_xi), int(coarse_lmax)))
    angular_mmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_MMAX", 1)
    angular_nmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_ANGULAR_NMAX", 1)
    helical_mmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_MMAX", 1)
    helical_nmax = _env_int("SFINCS_JAX_RHS1_FULL_CSR_COARSE_HELICAL_NMAX", min(4, int(layout.n_zeta) // 2))
    angular_mmax = max(0, min(int(layout.n_theta) // 2, int(angular_mmax)))
    angular_nmax = max(0, min(int(layout.n_zeta) // 2, int(angular_nmax)))
    helical_mmax = max(0, min(int(layout.n_theta) // 2, int(helical_mmax)))
    helical_nmax = max(0, min(int(layout.n_zeta) // 2, int(helical_nmax)))
    has_angular_modes = any((angular_mmax, angular_nmax, helical_mmax and helical_nmax))
    config.update(
        {
            "coarse_lmax": int(coarse_lmax),
            "coarse_include_tail": True,
            "coarse_angular_mmax": int(angular_mmax),
            "coarse_angular_nmax": int(angular_nmax),
            "coarse_helical_mmax": int(helical_mmax),
            "coarse_helical_nmax": int(helical_nmax),
            "coarse_basis": (
                "flux_surface_low_l_angular_plus_tail"
                if has_angular_modes
                else "flux_surface_low_l_plus_tail"
            ),
        }
    )
    return config


def _xblock_tz_low_l_indices(*, layout: RHS1BlockLayout, species: int, x: int, lmax: int) -> np.ndarray:
    indices = [
        layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
        for ell in range(int(lmax))
        for theta in range(int(layout.n_theta))
        for zeta in range(int(layout.n_zeta))
    ]
    return np.asarray(indices, dtype=np.int64)


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


def _active_positions_for_full_indices(*, active_indices: Any, full_indices: Any) -> np.ndarray:
    """Map full-system indices into active-system row/column positions."""

    active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    full_np = np.asarray(full_indices, dtype=np.int64).reshape((-1,))
    if active_np.size == 0 or full_np.size == 0:
        return np.zeros((0,), dtype=np.int64)
    order = np.argsort(active_np, kind="mergesort")
    active_sorted = active_np[order]
    hits = np.searchsorted(active_sorted, full_np)
    keep = (hits < active_sorted.size) & (active_sorted[hits.clip(max=max(0, active_sorted.size - 1))] == full_np)
    if not np.any(keep):
        return np.zeros((0,), dtype=np.int64)
    positions = order[hits[keep]]
    return np.unique(positions.astype(np.int64, copy=False))


def _estimate_xblock_tz_low_l_factor_nbytes(*, layout: RHS1BlockLayout, config: dict[str, object]) -> int:
    block_size = int(config["lmax"]) * int(layout.n_theta) * int(layout.n_zeta)
    n_blocks = int(layout.n_species) * int(layout.n_x)
    # Sparse factors should be much smaller than dense inverse blocks. This
    # conservative cap estimate prevents accidental full-resolution promotion.
    return int(n_blocks * block_size * min(block_size, 64) * np.dtype(np.float64).itemsize)


def _estimate_coarse_residual_nbytes(*, layout: RHS1BlockLayout, config: dict[str, object]) -> int:
    coarse_lmax = int(config["coarse_lmax"])
    surface_mode_count = _coarse_surface_mode_count(layout=layout, config=config)
    coarse_kinetic = int(layout.n_species) * int(layout.n_x) * int(coarse_lmax) * int(surface_mode_count)
    coarse_tail = int(layout.total_size) - int(layout.f_size)
    coarse_size = int(coarse_kinetic + coarse_tail)
    basis_nnz = int(coarse_kinetic * layout.n_theta * layout.n_zeta + coarse_tail)
    sparse_bytes = int(basis_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
    sparse_bytes += int((coarse_size + 1) * np.dtype(np.int32).itemsize)
    dense_bytes = int(coarse_size * coarse_size * np.dtype(np.float64).itemsize)
    return int(sparse_bytes + dense_bytes)


def _build_coarse_residual_basis_csc(*, layout: RHS1BlockLayout, config: dict[str, object]) -> Any:
    coarse_lmax = int(config["coarse_lmax"])
    surface_modes = _coarse_surface_modes(layout=layout, config=config)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    col = 0
    ntz = int(layout.n_theta) * int(layout.n_zeta)
    theta = np.arange(int(layout.n_theta), dtype=np.int64)
    zeta = np.arange(int(layout.n_zeta), dtype=np.int64)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    for species in range(int(layout.n_species)):
        for x in range(int(layout.n_x)):
            for ell in range(coarse_lmax):
                idx = (
                    (((int(species) * int(layout.n_x) + int(x)) * int(layout.n_xi) + int(ell)) * int(layout.n_theta) + theta_grid)
                    * int(layout.n_zeta)
                    + zeta_grid
                ).astype(np.int64, copy=False).reshape((-1,))
                for _mode_name, surface_values in surface_modes:
                    rows.append(idx)
                    cols.append(np.full((ntz,), col, dtype=np.int64))
                    data.append(surface_values)
                    col += 1
    if bool(config.get("coarse_include_tail", True)):
        tail_size = int(layout.total_size) - int(layout.f_size)
        if tail_size > 0:
            tail_rows = int(layout.f_size) + np.arange(tail_size, dtype=np.int64)
            rows.append(tail_rows)
            cols.append(np.arange(col, col + tail_size, dtype=np.int64))
            data.append(np.ones((tail_size,), dtype=np.float64))
            col += tail_size
    if not rows:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64)
    row = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    values = np.concatenate(data)
    basis = sp.coo_matrix((values, (row, col_idx)), shape=(int(layout.total_size), int(col))).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    return basis


def _build_active_native_xell_coarse_window_basis_csc(
    *,
    layout: RHS1BlockLayout,
) -> tuple[Any, dict[str, object]]:
    """Return optional identity columns for targeted active-native coarse windows.

    ``SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS`` accepts
    comma-separated ``species:x:ell`` triples. ``*`` or ``all`` may be used for
    species or x. The selected ell is expanded by
    ``..._WINDOW_ELL_RADIUS`` and x by ``..._WINDOW_X_RADIUS``. The generated
    columns are identity columns in the full kinetic block, so after projection
    they target the actual residual entries that the low-mode coarse basis may
    miss. The final active coarse builder still applies its own column and
    memory caps.
    """

    spec = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS", "").strip()
    max_columns = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_MAX_COLUMNS", 8192)),
    )
    metadata = {
        "window_basis_requested": bool(spec),
        "window_basis_specs": str(spec),
        "window_basis_columns": 0,
        "window_basis_max_columns": int(max_columns),
    }
    if not spec or int(max_columns) <= 0:
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    ell_radius = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_ELL_RADIUS", 1)),
    )
    x_radius = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_X_RADIUS", 0)),
    )

    def parse_axis(raw: str, stop: int) -> tuple[int, ...]:
        item = str(raw).strip().lower()
        if item in {"*", "all"}:
            return tuple(range(int(stop)))
        try:
            value = int(item)
        except ValueError:
            return ()
        if value < 0 or value >= int(stop):
            return ()
        return (int(value),)

    selected: list[int] = []
    skipped_specs = 0
    for raw_item in str(spec).replace(";", ",").split(","):
        item = raw_item.strip()
        if not item:
            continue
        parts = item.replace("/", ":").split(":")
        if len(parts) != 3:
            skipped_specs += 1
            continue
        species_values = parse_axis(parts[0], int(layout.n_species))
        x_centers = parse_axis(parts[1], int(layout.n_x))
        ell_centers = parse_axis(parts[2], int(layout.n_xi))
        if not species_values or not x_centers or not ell_centers:
            skipped_specs += 1
            continue
        for species in species_values:
            for x_center in x_centers:
                x_min = max(0, int(x_center) - int(x_radius))
                x_max = min(int(layout.n_x) - 1, int(x_center) + int(x_radius))
                for ell_center in ell_centers:
                    ell_min = max(0, int(ell_center) - int(ell_radius))
                    ell_max = min(int(layout.n_xi) - 1, int(ell_center) + int(ell_radius))
                    for x_index in range(x_min, x_max + 1):
                        for ell in range(ell_min, ell_max + 1):
                            for theta in range(int(layout.n_theta)):
                                for zeta in range(int(layout.n_zeta)):
                                    selected.append(
                                        layout.kinetic_flat_index(
                                            species=species,
                                            x=x_index,
                                            ell=ell,
                                            theta=theta,
                                            zeta=zeta,
                                        )
                                    )
                                    if len(selected) >= int(max_columns):
                                        break
                                if len(selected) >= int(max_columns):
                                    break
                            if len(selected) >= int(max_columns):
                                break
                        if len(selected) >= int(max_columns):
                            break
                    if len(selected) >= int(max_columns):
                        break
                if len(selected) >= int(max_columns):
                    break
            if len(selected) >= int(max_columns):
                break
        if len(selected) >= int(max_columns):
            break

    if not selected:
        metadata.update(
            {
                "window_basis_skipped_specs": int(skipped_specs),
                "window_basis_truncated": False,
            }
        )
        return sp.csc_matrix((int(layout.total_size), 0), dtype=np.float64), metadata

    rows = np.unique(np.asarray(selected, dtype=np.int64))
    if int(rows.size) > int(max_columns):
        rows = rows[: int(max_columns)]
    cols = np.arange(int(rows.size), dtype=np.int64)
    basis = sp.coo_matrix(
        (np.ones((int(rows.size),), dtype=np.float64), (rows, cols)),
        shape=(int(layout.total_size), int(rows.size)),
    ).tocsc()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    metadata.update(
        {
            "window_basis_columns": int(basis.shape[1]),
            "window_basis_nnz": int(basis.nnz),
            "window_basis_ell_radius": int(ell_radius),
            "window_basis_x_radius": int(x_radius),
            "window_basis_skipped_specs": int(skipped_specs),
            "window_basis_truncated": bool(len(selected) >= int(max_columns)),
        }
    )
    return basis, metadata


def _append_adaptive_residual_basis_csc(
    *,
    matrix: Any,
    base_operator: Any,
    basis: Any,
    max_total_columns: int,
) -> tuple[Any, dict[str, object]]:
    """Append bounded residual-derived coarse columns ``z - A M z``.

    The generated vectors are construction-time snapshots of the mismatch
    between the true active operator ``A`` and the selected base preconditioner
    ``M``. They are independent of the Krylov right-hand side, so the resulting
    preconditioner remains linear. Columns are sparsified by relative magnitude
    and capped by both column count and per-column nonzeros.
    """

    enabled = _env_bool("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", False)
    max_columns = max(0, int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_COLUMNS", 32)))
    max_seed_columns = max(
        0,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_SEED_COLUMNS", 64)),
    )
    max_nnz_per_column = max(
        1,
        int(_env_int("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_NNZ_PER_COLUMN", 4096)),
    )
    drop_rel = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_DROP_REL", 1.0e-3)),
    )
    min_rel_norm = max(
        0.0,
        float(_env_float("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MIN_REL_NORM", 1.0e-8)),
    )
    metadata = {
        "adaptive_residual_basis_enabled": bool(enabled),
        "adaptive_residual_basis_columns": 0,
        "adaptive_residual_basis_seed_columns": 0,
        "adaptive_residual_basis_max_columns": int(max_columns),
        "adaptive_residual_basis_max_seed_columns": int(max_seed_columns),
        "adaptive_residual_basis_max_nnz_per_column": int(max_nnz_per_column),
        "adaptive_residual_basis_drop_rel": float(drop_rel),
        "adaptive_residual_basis_min_rel_norm": float(min_rel_norm),
    }
    if not bool(enabled) or int(max_columns) <= 0 or int(max_seed_columns) <= 0 or int(basis.shape[1]) <= 0:
        return basis, metadata

    matrix_csr = matrix.tocsr()
    basis_csc = basis.tocsc()
    remaining = max(0, int(max_total_columns) - int(basis_csc.shape[1]))
    max_columns_use = min(int(max_columns), int(remaining))
    if max_columns_use <= 0:
        metadata["adaptive_residual_basis_truncated_by_total_cap"] = True
        return basis_csc, metadata

    seed_count = min(int(basis_csc.shape[1]), int(max_seed_columns))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    residual_norms: list[float] = []
    skipped_small = 0
    skipped_zero = 0
    for seed_col in range(seed_count):
        if len(rows) >= max_columns_use:
            break
        z = np.asarray(basis_csc[:, seed_col].toarray(), dtype=np.float64).reshape((-1,))
        z_norm = max(float(np.linalg.norm(z)), np.finfo(np.float64).tiny)
        mz = np.asarray(base_operator.matvec(z), dtype=np.float64).reshape((-1,))
        residual = z - np.asarray(matrix_csr @ mz, dtype=np.float64).reshape((-1,))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or residual_norm <= 0.0:
            skipped_zero += 1
            continue
        if residual_norm / z_norm < float(min_rel_norm):
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
        values_norm = float(np.linalg.norm(values))
        if not np.isfinite(values_norm) or values_norm <= 0.0:
            skipped_zero += 1
            continue
        rows.append(keep.astype(np.int64, copy=False))
        cols.append(np.full((int(keep.size),), len(rows) - 1, dtype=np.int64))
        data.append((values / values_norm).astype(np.float64, copy=False))
        residual_norms.append(float(residual_norm))

    if not rows:
        metadata.update(
            {
                "adaptive_residual_basis_seed_columns": int(seed_count),
                "adaptive_residual_basis_skipped_small": int(skipped_small),
                "adaptive_residual_basis_skipped_zero": int(skipped_zero),
                "adaptive_residual_basis_truncated_by_total_cap": False,
            }
        )
        return basis_csc, metadata

    adaptive = sp.coo_matrix(
        (
            np.concatenate(data),
            (np.concatenate(rows), np.concatenate(cols)),
        ),
        shape=(int(matrix_csr.shape[0]), int(len(rows))),
    ).tocsc()
    adaptive.sum_duplicates()
    adaptive.eliminate_zeros()
    combined = sp.hstack([basis_csc, adaptive], format="csc")
    metadata.update(
        {
            "adaptive_residual_basis_columns": int(adaptive.shape[1]),
            "adaptive_residual_basis_seed_columns": int(seed_count),
            "adaptive_residual_basis_nnz": int(adaptive.nnz),
            "adaptive_residual_basis_skipped_small": int(skipped_small),
            "adaptive_residual_basis_skipped_zero": int(skipped_zero),
            "adaptive_residual_basis_residual_norm_max": float(max(residual_norms)),
            "adaptive_residual_basis_residual_norm_min": float(min(residual_norms)),
            "adaptive_residual_basis_truncated_by_total_cap": bool(len(rows) >= max_columns_use),
        }
    )
    return combined, metadata


def _coarse_surface_mode_count(*, layout: RHS1BlockLayout, config: dict[str, object]) -> int:
    return int(len(_coarse_surface_modes(layout=layout, config=config)))


def _coarse_surface_modes(*, layout: RHS1BlockLayout, config: dict[str, object]) -> tuple[tuple[str, np.ndarray], ...]:
    """Return normalized low-angle modes for the RHSMode=1 coarse residual space."""

    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    theta = 2.0 * np.pi * np.arange(n_theta, dtype=np.float64) / max(1, n_theta)
    zeta = 2.0 * np.pi * np.arange(n_zeta, dtype=np.float64) / max(1, n_zeta)
    theta_grid, zeta_grid = np.meshgrid(theta, zeta, indexing="ij")
    modes: list[tuple[str, np.ndarray]] = []

    def add_mode(name: str, values: np.ndarray) -> None:
        flat = np.asarray(values, dtype=np.float64).reshape((-1,))
        norm = float(np.linalg.norm(flat))
        if not np.isfinite(norm) or norm <= 0.0:
            return
        flat = flat / norm
        for _existing_name, existing in modes:
            # Avoid exact duplicate modes on tiny grids; the coarse solve still
            # has regularization, but removing duplicates keeps conditioning sane.
            if flat.shape == existing.shape and float(abs(np.dot(flat, existing))) > 1.0 - 1.0e-12:
                return
        modes.append((name, flat))

    add_mode("constant", np.ones((n_theta, n_zeta), dtype=np.float64))
    angular_mmax = int(config.get("coarse_angular_mmax", 0) or 0)
    angular_nmax = int(config.get("coarse_angular_nmax", 0) or 0)
    helical_mmax = int(config.get("coarse_helical_mmax", 0) or 0)
    helical_nmax = int(config.get("coarse_helical_nmax", 0) or 0)

    for m in range(1, max(0, angular_mmax) + 1):
        add_mode(f"cos_theta_{m}", np.cos(float(m) * theta_grid))
        add_mode(f"sin_theta_{m}", np.sin(float(m) * theta_grid))
    for n in range(1, max(0, angular_nmax) + 1):
        add_mode(f"cos_zeta_{n}", np.cos(float(n) * zeta_grid))
        add_mode(f"sin_zeta_{n}", np.sin(float(n) * zeta_grid))
    for m in range(1, max(0, helical_mmax) + 1):
        for n in range(1, max(0, helical_nmax) + 1):
            phase = float(m) * theta_grid - float(n) * zeta_grid
            add_mode(f"cos_helical_{m}_{n}", np.cos(phase))
            add_mode(f"sin_helical_{m}_{n}", np.sin(phase))
    return tuple(modes)


def _sparse_lu_factor_nbytes(factor: Any) -> int:
    return int(_scipy_csr_nbytes(factor.L.tocsr()) + _scipy_csr_nbytes(factor.U.tocsr()))


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


def _safe_inverse_diagonal(diagonal: Any, *, regularization: float) -> tuple[np.ndarray, dict[str, object]]:
    diag = np.asarray(diagonal, dtype=np.float64).reshape((-1,))
    abs_diag = np.abs(diag)
    scale = max(float(np.max(abs_diag)) if abs_diag.size else 0.0, 1.0)
    floor = float(abs(regularization)) * scale
    if floor == 0.0:
        floor = np.finfo(np.float64).tiny
    safe = diag.copy()
    small = abs_diag <= floor
    signs = np.where(safe < 0.0, -1.0, 1.0)
    safe[small] = signs[small] * floor
    inv = 1.0 / safe
    metadata = {
        "diagonal_size": int(diag.size),
        "diagonal_abs_max": float(np.max(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_abs_min": float(np.min(abs_diag)) if abs_diag.size else 0.0,
        "diagonal_floor": float(floor),
        "diagonal_regularized_count": int(np.count_nonzero(small)),
    }
    return inv, metadata


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
    "build_structured_rhs1_full_csr_preconditioner",
    "clear_structured_rhs1_full_csr_cache",
    "select_active_fortran_v3_reduced_support_mode_preconditioner",
    "select_structured_rhs1_full_csr_operator",
    "solve_structured_rhs1_full_csr",
]
