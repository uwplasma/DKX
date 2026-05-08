"""Memory models for SFINCS_JAX linear solves.

The estimates here are intentionally simple and conservative. They are used for
policy tests, solver-route explanations, and preflight benchmark manifests; the
measured profiler data remains the final authority for production decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


def dtype_nbytes(dtype: Any = np.float64) -> int:
    """Return bytes per scalar for ``dtype``."""

    return int(np.dtype(dtype).itemsize)


def dense_matrix_nbytes(n_rows: int, n_cols: int | None = None, *, dtype: Any = np.float64) -> int:
    """Return storage for a dense matrix with the requested dtype."""

    n_rows = max(0, int(n_rows))
    n_cols = n_rows if n_cols is None else max(0, int(n_cols))
    return int(n_rows) * int(n_cols) * dtype_nbytes(dtype)


def csr_matrix_nbytes(
    n_rows: int,
    nnz: int,
    *,
    data_dtype: Any = np.float64,
    index_dtype: Any = np.int32,
) -> int:
    """Return storage for CSR ``data``/``indices``/``indptr`` arrays."""

    n_rows = max(0, int(n_rows))
    nnz = max(0, int(nnz))
    data_bytes = nnz * dtype_nbytes(data_dtype)
    index_bytes = nnz * dtype_nbytes(index_dtype)
    indptr_bytes = (n_rows + 1) * dtype_nbytes(index_dtype)
    return int(data_bytes + index_bytes + indptr_bytes)


def gmres_basis_nbytes(
    n: int,
    restart: int,
    *,
    dtype: Any = np.float64,
    extra_vectors: int = 4,
) -> int:
    """Estimate GMRES vector storage.

    Restarted GMRES stores an Arnoldi basis of roughly ``restart + 1`` vectors.
    The ``extra_vectors`` term accounts for residual/work/output vectors and
    keeps the model conservative enough for route selection.
    """

    n = max(0, int(n))
    restart = max(0, int(restart))
    extra_vectors = max(0, int(extra_vectors))
    return int(n * (restart + 1 + extra_vectors) * dtype_nbytes(dtype))


def bicgstab_work_nbytes(n: int, *, dtype: Any = np.float64, work_vectors: int = 8) -> int:
    """Estimate short-recurrence BiCGStab vector storage."""

    n = max(0, int(n))
    work_vectors = max(0, int(work_vectors))
    return int(n * work_vectors * dtype_nbytes(dtype))


def gmres_restart_for_budget(
    n: int,
    requested_restart: int,
    *,
    dtype: Any = np.float64,
    max_bytes: int | float | None,
    extra_vectors: int = 4,
) -> int:
    """Cap GMRES restart so the basis estimate fits in ``max_bytes``."""

    n = int(n)
    requested_restart = int(requested_restart)
    if n <= 0 or requested_restart <= 1 or max_bytes is None:
        return requested_restart
    max_bytes_f = float(max_bytes)
    if not math.isfinite(max_bytes_f) or max_bytes_f <= 0.0:
        return requested_restart
    denom = n * dtype_nbytes(dtype)
    if denom <= 0:
        return requested_restart
    max_vectors = int(max_bytes_f // denom)
    max_restart = max(1, max_vectors - 1 - max(0, int(extra_vectors)))
    return min(requested_restart, max_restart)


@dataclass(frozen=True)
class LinearSolveMemoryEstimate:
    """Dominant memory terms for one linear solve route."""

    unknowns: int
    dtype: str
    dense_operator_nbytes: int
    csr_operator_nbytes: int | None
    gmres_basis_nbytes: int
    bicgstab_work_nbytes: int
    preconditioner_nbytes: int | None = None
    compiled_temp_nbytes: int | None = None
    device_count: int = 1

    @property
    def dense_total_nbytes(self) -> int:
        """Dense operator plus GMRES basis and known preconditioner/temp costs."""

        total = int(self.dense_operator_nbytes) + int(self.gmres_basis_nbytes)
        if self.preconditioner_nbytes is not None:
            total += int(self.preconditioner_nbytes)
        if self.compiled_temp_nbytes is not None:
            total += int(self.compiled_temp_nbytes)
        return total

    @property
    def csr_total_nbytes(self) -> int | None:
        """CSR operator plus GMRES basis and known preconditioner/temp costs."""

        if self.csr_operator_nbytes is None:
            return None
        total = int(self.csr_operator_nbytes) + int(self.gmres_basis_nbytes)
        if self.preconditioner_nbytes is not None:
            total += int(self.preconditioner_nbytes)
        if self.compiled_temp_nbytes is not None:
            total += int(self.compiled_temp_nbytes)
        return total

    @property
    def dense_per_device_nbytes(self) -> int:
        """Dense total divided over a sharded device mesh."""

        return int(math.ceil(self.dense_total_nbytes / max(1, int(self.device_count))))

    @property
    def csr_per_device_nbytes(self) -> int | None:
        """CSR total divided over a sharded device mesh."""

        if self.csr_total_nbytes is None:
            return None
        return int(math.ceil(self.csr_total_nbytes / max(1, int(self.device_count))))

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable fields for traces and manifests."""

        return {
            "unknowns": int(self.unknowns),
            "dtype": self.dtype,
            "dense_operator_nbytes": int(self.dense_operator_nbytes),
            "csr_operator_nbytes": None if self.csr_operator_nbytes is None else int(self.csr_operator_nbytes),
            "gmres_basis_nbytes": int(self.gmres_basis_nbytes),
            "bicgstab_work_nbytes": int(self.bicgstab_work_nbytes),
            "preconditioner_nbytes": (
                None if self.preconditioner_nbytes is None else int(self.preconditioner_nbytes)
            ),
            "compiled_temp_nbytes": None if self.compiled_temp_nbytes is None else int(self.compiled_temp_nbytes),
            "device_count": int(self.device_count),
            "dense_total_nbytes": int(self.dense_total_nbytes),
            "csr_total_nbytes": None if self.csr_total_nbytes is None else int(self.csr_total_nbytes),
            "dense_per_device_nbytes": int(self.dense_per_device_nbytes),
            "csr_per_device_nbytes": (
                None if self.csr_per_device_nbytes is None else int(self.csr_per_device_nbytes)
            ),
        }


def estimate_linear_solve_memory(
    *,
    unknowns: int,
    gmres_restart: int,
    dtype: Any = np.float64,
    csr_nnz: int | None = None,
    preconditioner_nbytes: int | None = None,
    compiled_temp_nbytes: int | None = None,
    device_count: int = 1,
) -> LinearSolveMemoryEstimate:
    """Build a conservative memory estimate for a linear solve route."""

    dtype_np = np.dtype(dtype)
    n = max(0, int(unknowns))
    csr_bytes = None
    if csr_nnz is not None:
        csr_bytes = csr_matrix_nbytes(n, int(csr_nnz), data_dtype=dtype_np)
    return LinearSolveMemoryEstimate(
        unknowns=n,
        dtype=str(dtype_np),
        dense_operator_nbytes=dense_matrix_nbytes(n, dtype=dtype_np),
        csr_operator_nbytes=csr_bytes,
        gmres_basis_nbytes=gmres_basis_nbytes(n, int(gmres_restart), dtype=dtype_np),
        bicgstab_work_nbytes=bicgstab_work_nbytes(n, dtype=dtype_np),
        preconditioner_nbytes=None if preconditioner_nbytes is None else int(preconditioner_nbytes),
        compiled_temp_nbytes=None if compiled_temp_nbytes is None else int(compiled_temp_nbytes),
        device_count=max(1, int(device_count)),
    )


def estimate_sparse_pc_memory(
    *,
    unknowns: int,
    gmres_restart: int,
    csr_nnz: int,
    dtype: Any = np.float64,
    factor_fill_estimate: float = 8.0,
    device_count: int = 1,
) -> LinearSolveMemoryEstimate:
    """Estimate sparse-PC storage before factorizing the preconditioner.

    ``factor_fill_estimate`` is a multiplicative estimate for SuperLU/ILU
    factor storage relative to the input CSR operator. It is intentionally
    conservative and is used only for opt-in memory-budget preflight checks; the
    measured ``L``/``U`` factor storage in solver traces remains authoritative.
    """

    base = estimate_linear_solve_memory(
        unknowns=unknowns,
        gmres_restart=gmres_restart,
        dtype=dtype,
        csr_nnz=csr_nnz,
        device_count=device_count,
    )
    csr_bytes = int(base.csr_operator_nbytes or 0)
    factor_fill = max(0.0, float(factor_fill_estimate))
    factor_bytes = int(math.ceil(csr_bytes * factor_fill)) if csr_bytes > 0 and factor_fill > 0.0 else None
    return estimate_linear_solve_memory(
        unknowns=unknowns,
        gmres_restart=gmres_restart,
        dtype=dtype,
        csr_nnz=csr_nnz,
        preconditioner_nbytes=factor_bytes,
        device_count=device_count,
    )


__all__ = [
    "LinearSolveMemoryEstimate",
    "bicgstab_work_nbytes",
    "csr_matrix_nbytes",
    "dense_matrix_nbytes",
    "dtype_nbytes",
    "estimate_linear_solve_memory",
    "estimate_sparse_pc_memory",
    "gmres_basis_nbytes",
    "gmres_restart_for_budget",
]
