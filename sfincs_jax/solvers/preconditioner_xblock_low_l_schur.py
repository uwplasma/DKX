"""Low-pitch x-block Schur preconditioners for full RHSMode=1 CSR systems.

These builders are host-side setup utilities for the explicit full-CSR
RHSMode=1 solve lane. They are intentionally non-autodiff preconditioners:
setup may use SciPy sparse factors, while the installed action is a fixed
linear operator checked by the caller against the true physical residual.
"""

from __future__ import annotations

from typing import Any
import time

import numpy as np
import scipy.sparse as sp

from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.solvers.preconditioner_full_fp_csr import build_rhs1_full_csr_kinetic_preconditioner
from .preconditioner_schur_profile import RHS1StructuredFullCSRPreconditioner, safe_inverse_diagonal
from .preconditioner_schur_profile import build_coarse_residual_basis_csc, coarse_surface_modes
from .preconditioner_reduced_pmat import sparse_lu_factor_nbytes

__all__ = (
    "build_native_xell_kinetic_preconditioner",
    "build_native_xell_tail_schur_preconditioner",
    "build_xblock_tz_low_l_coarse_residual_preconditioner",
    "build_xblock_tz_low_l_schur_preconditioner",
    "xblock_tz_low_l_indices",
)


def build_native_xell_kinetic_preconditioner(
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


def build_native_xell_tail_schur_preconditioner(
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

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    if tail_size <= 0:
        return build_native_xell_kinetic_preconditioner(
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


def build_xblock_tz_low_l_schur_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    config: dict[str, object],
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build sparse low-``ell`` ``(theta,zeta)`` x-block factors plus tail Schur."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator, spilu, splu  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inv_diag, diag_meta = safe_inverse_diagonal(matrix.diagonal()[:n_f], regularization=regularization)
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
            indices = xblock_tz_low_l_indices(layout=layout, species=species, x=x, lmax=lmax)
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
            current_nbytes = sparse_lu_factor_nbytes(factor)
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


def build_xblock_tz_low_l_coarse_residual_preconditioner(
    *,
    matrix: Any,
    layout: RHS1BlockLayout,
    requested_kind: str,
    regularization: float,
    max_factor_nbytes: int,
    config: dict[str, object],
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Add a physics low-mode coarse residual equation to the x-block Schur base."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    base = build_xblock_tz_low_l_schur_preconditioner(
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

    basis = build_coarse_residual_basis_csc(layout=layout, config=config)
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
    surface_modes = coarse_surface_modes(layout=layout, config=config)
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


def xblock_tz_low_l_indices(*, layout: RHS1BlockLayout, species: int, x: int, lmax: int) -> np.ndarray:
    """Return full-system kinetic indices for one low-``ell`` species/x block."""

    indices = [
        layout.kinetic_flat_index(species=species, x=x, ell=ell, theta=theta, zeta=zeta)
        for ell in range(int(lmax))
        for theta in range(int(layout.n_theta))
        for zeta in range(int(layout.n_zeta))
    ]
    return np.asarray(indices, dtype=np.int64)
