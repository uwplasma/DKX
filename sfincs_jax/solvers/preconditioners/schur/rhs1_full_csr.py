"""Full-CSR Schur preconditioners for RHSMode=1 profile-response solves.

The builders in this module are host-side setup utilities for the explicit
RHSMode=1 CSR solve lane. They install fixed linear preconditioner actions for
non-autodiff production solves; differentiable JAX-native lanes should not route
through these SciPy factorization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np

__all__ = (
    "RHS1StructuredFullCSRPreconditioner",
    "build_block_schur_preconditioner",
    "build_diagonal_schur_preconditioner",
    "build_jacobi_preconditioner",
    "build_x_xi_block_schur_preconditioner",
    "build_xi_block_schur_preconditioner",
    "estimate_x_xi_block_inverse_nbytes",
    "estimate_xi_block_inverse_nbytes",
    "estimate_zeta_block_inverse_nbytes",
    "safe_inverse_diagonal",
)


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


def build_jacobi_preconditioner(
    *,
    matrix: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
    reason: str,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a diagonal inverse fallback with regularized zero pivots."""

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    inv_diag, diag_meta = safe_inverse_diagonal(matrix.diagonal(), regularization=regularization)
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


def build_diagonal_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a diagonal kinetic inverse plus exact dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    if tail_size <= 0:
        return build_jacobi_preconditioner(
            matrix=matrix,
            requested_kind=requested_kind,
            regularization=regularization,
            t0=t0,
            reason="no_global_tail",
        )

    diag = matrix.diagonal()
    inv_f, diag_meta = safe_inverse_diagonal(diag[:n_f], regularization=regularization)
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


def build_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a zeta-line kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    block_size = int(layout.n_zeta)
    n_blocks = n_f // block_size
    inverse_blocks, block_meta = build_zeta_diagonal_inverse_blocks(
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


def build_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build a pitch-line kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = build_xi_diagonal_inverse_blocks(
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


def build_x_xi_block_schur_preconditioner(
    *,
    matrix: Any,
    layout: Any,
    requested_kind: str,
    regularization: float,
    t0: float,
) -> RHS1StructuredFullCSRPreconditioner:
    """Build an x-pitch kinetic inverse plus dense tail Schur solve."""

    from scipy.linalg import lu_factor, lu_solve  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    n_f = int(layout.f_size)
    n_total = int(layout.total_size)
    tail_size = n_total - n_f
    inverse_blocks, block_indices, block_meta = build_x_xi_diagonal_inverse_blocks(
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


def build_zeta_diagonal_inverse_blocks(
    *,
    matrix: Any,
    n_f: int,
    block_size: int,
    regularization: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Build dense inverses for contiguous zeta-line kinetic blocks."""

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


def build_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: Any,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dense inverses for active pitch-line kinetic blocks."""

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


def build_x_xi_diagonal_inverse_blocks(
    *,
    matrix: Any,
    layout: Any,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build dense inverses for combined radial-pitch kinetic blocks."""

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


def estimate_zeta_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for zeta-line dense inverse blocks."""

    block_size = int(layout.n_zeta)
    n_blocks = int(layout.f_size) // block_size
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def estimate_xi_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for pitch-line dense inverse blocks."""

    block_size = int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_x) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def estimate_x_xi_block_inverse_nbytes(layout: Any) -> int:
    """Estimate memory for combined radial-pitch dense inverse blocks."""

    block_size = int(layout.n_x) * int(layout.n_xi)
    n_blocks = int(layout.n_species) * int(layout.n_theta) * int(layout.n_zeta)
    return int(n_blocks * block_size * block_size * np.dtype(np.float64).itemsize)


def safe_inverse_diagonal(diagonal: Any, *, regularization: float) -> tuple[np.ndarray, dict[str, object]]:
    """Invert a diagonal with a scale-aware floor and return pivot metadata."""

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
