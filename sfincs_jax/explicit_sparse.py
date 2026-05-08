from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu, splu

import jax

StorageKind = Literal["dense", "csr", "linear_operator"]
FactorKind = Literal["lu", "ilu"]


def _backend_name(backend: str | None = None) -> str:
    if backend is not None and str(backend).strip():
        return str(backend).strip().lower()
    try:
        return jax.default_backend().strip().lower()
    except Exception:  # pragma: no cover - defensive fallback
        return "cpu"


def _host_array(value, *, dtype=None) -> np.ndarray:
    arr = np.asarray(jax.device_get(value), dtype=dtype)
    return np.array(arr, copy=True)


def estimate_dense_nbytes(shape: tuple[int, int], dtype=np.float64) -> int:
    dtype_np = np.dtype(dtype)
    return int(shape[0]) * int(shape[1]) * int(dtype_np.itemsize)


def estimate_csr_nbytes(
    shape: tuple[int, int],
    nnz: int,
    *,
    data_dtype=np.float64,
    index_dtype=np.int32,
) -> int:
    data_itemsize = np.dtype(data_dtype).itemsize
    index_itemsize = np.dtype(index_dtype).itemsize
    return int(nnz) * (data_itemsize + index_itemsize) + (int(shape[0]) + 1) * index_itemsize


@dataclass(frozen=True)
class SparseDecision:
    storage_kind: StorageKind
    reason: str
    backend: str
    shape: tuple[int, int]
    dense_nbytes: int
    csr_nbytes_estimate: int
    nnz_estimate: int | None
    block_cols: int | None = None
    drop_tol: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "storage_kind": self.storage_kind,
            "reason": self.reason,
            "backend": self.backend,
            "shape": self.shape,
            "dense_nbytes": self.dense_nbytes,
            "csr_nbytes_estimate": self.csr_nbytes_estimate,
            "nnz_estimate": self.nnz_estimate,
            "block_cols": self.block_cols,
            "drop_tol": self.drop_tol,
        }


@dataclass(frozen=True)
class SparseOperatorBundle:
    matrix: np.ndarray | sp.spmatrix | None
    operator: LinearOperator
    metadata: SparseDecision

    def matvec(self, x) -> np.ndarray:
        return np.asarray(self.operator.matvec(np.asarray(x)))


@dataclass(frozen=True)
class SparseFactorBundle:
    factor: object
    operator: SparseOperatorBundle
    metadata: SparseDecision
    kind: FactorKind
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None

    def solve(self, rhs) -> np.ndarray:
        rhs_host = _host_array(rhs, dtype=self.operator.matrix.dtype if self.operator.matrix is not None else None)
        sol = self.factor.solve(rhs_host)
        return np.asarray(sol)


def estimate_superlu_factor_storage(factor: object) -> tuple[int | None, int | None]:
    """Estimate SuperLU/ILU storage from materialized ``L`` and ``U`` factors."""

    total_nbytes = 0
    total_nnz = 0
    saw_factor = False
    for name in ("L", "U"):
        matrix = getattr(factor, name, None)
        if matrix is None:
            continue
        if sp.issparse(matrix):
            matrix_csr = matrix.tocsr()
            total_nbytes += estimate_csr_nbytes(
                tuple(matrix_csr.shape),
                int(matrix_csr.nnz),
                data_dtype=matrix_csr.dtype,
                index_dtype=matrix_csr.indices.dtype,
            )
            total_nnz += int(matrix_csr.nnz)
            saw_factor = True
            continue
        arr = np.asarray(matrix)
        if arr.size:
            total_nbytes += int(arr.nbytes)
            total_nnz += int(np.count_nonzero(arr))
            saw_factor = True
    if not saw_factor:
        return None, None
    return int(total_nbytes), int(total_nnz)


def choose_storage_kind(
    *,
    shape: tuple[int, int],
    nnz_estimate: int | None,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    block_cols: int | None = None,
    drop_tol: float = 0.0,
) -> SparseDecision:
    backend_norm = _backend_name(backend)
    dense_nbytes = estimate_dense_nbytes(shape, np.float64)
    csr_nnz = int(nnz_estimate) if nnz_estimate is not None else int(shape[0] * shape[1])
    csr_nbytes = estimate_csr_nbytes(shape, csr_nnz)
    dense_cap = int(max(0.0, float(dense_max_mb)) * 1e6)
    csr_cap = int(max(0.0, float(csr_max_mb)) * 1e6)

    dense_fits = dense_nbytes <= dense_cap if dense_cap > 0 else False
    csr_fits = csr_nbytes <= csr_cap if csr_cap > 0 else False
    sparse_smaller = csr_nbytes <= dense_nbytes

    if force_dense and force_sparse:
        raise ValueError("force_dense and force_sparse cannot both be true")

    if force_dense:
        return SparseDecision(
            storage_kind="dense",
            reason="forced dense materialization",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if force_sparse:
        if csr_fits or not dense_fits:
            return SparseDecision(
                storage_kind="csr",
                reason="forced sparse materialization",
                backend=backend_norm,
                shape=shape,
                dense_nbytes=dense_nbytes,
                csr_nbytes_estimate=csr_nbytes,
                nnz_estimate=nnz_estimate,
                block_cols=block_cols,
                drop_tol=drop_tol,
            )
        return SparseDecision(
            storage_kind="linear_operator",
            reason="forced sparse but CSR budget unavailable",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if backend_norm == "gpu" and prefer_sparse_on_gpu and csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="preferred sparse host materialization on GPU",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if sparse_smaller and csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="CSR smaller than dense and within budget",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if dense_fits:
        return SparseDecision(
            storage_kind="dense",
            reason="dense within budget",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    if csr_fits:
        return SparseDecision(
            storage_kind="csr",
            reason="dense budget exceeded but CSR fits",
            backend=backend_norm,
            shape=shape,
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes,
            nnz_estimate=nnz_estimate,
            block_cols=block_cols,
            drop_tol=drop_tol,
        )

    return SparseDecision(
        storage_kind="linear_operator",
        reason="dense and CSR budgets exceeded; keep operator-only fallback",
        backend=backend_norm,
        shape=shape,
        dense_nbytes=dense_nbytes,
        csr_nbytes_estimate=csr_nbytes,
        nnz_estimate=nnz_estimate,
        block_cols=block_cols,
        drop_tol=drop_tol,
    )


def _drop_tol_dense(a: np.ndarray, drop_tol: float) -> np.ndarray:
    if float(drop_tol) <= 0.0:
        return np.array(a, copy=True)
    out = np.array(a, copy=True)
    out[np.abs(out) <= float(drop_tol)] = 0.0
    return out


def _operator_from_matrix(matrix: np.ndarray | sp.spmatrix) -> LinearOperator:
    n_rows, n_cols = matrix.shape

    def _matvec(x):
        return np.asarray(matrix @ np.asarray(x))

    dtype = matrix.dtype if sp.issparse(matrix) else np.asarray(matrix).dtype
    return LinearOperator((n_rows, n_cols), matvec=_matvec, dtype=dtype)


def _normalize_dense_input(a, *, dtype=None) -> np.ndarray:
    arr = _host_array(a, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D matrix, got shape {arr.shape}")
    return arr


def build_operator_from_dense(
    a,
    *,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    drop_tol: float = 0.0,
) -> SparseOperatorBundle:
    dense = _normalize_dense_input(a)
    nnz = int(np.count_nonzero(np.abs(dense) > float(drop_tol))) if float(drop_tol) > 0.0 else int(np.count_nonzero(dense))
    decision = choose_storage_kind(
        shape=dense.shape,
        nnz_estimate=nnz,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=force_sparse,
        force_dense=force_dense,
        drop_tol=drop_tol,
    )
    if decision.storage_kind == "csr":
        matrix = sp.csr_matrix(_drop_tol_dense(dense, drop_tol))
    else:
        matrix = np.array(dense, copy=True)
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def _coerce_block(block, *, dtype=None):
    if block is None:
        return None
    arr = _host_array(block, dtype=dtype)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D block, got shape {arr.shape}")
    return arr


def build_operator_from_blocks(
    blocks,
    *,
    backend: str | None = None,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    drop_tol: float = 0.0,
) -> SparseOperatorBundle:
    block_rows = [[_coerce_block(block, dtype=None) for block in row] for row in blocks]
    matrix = sp.bmat(block_rows, format="csr")
    if float(drop_tol) > 0.0:
        matrix = matrix.copy()
        matrix.data[np.abs(matrix.data) <= float(drop_tol)] = 0.0
        matrix.eliminate_zeros()
    nnz = int(matrix.nnz)
    decision = choose_storage_kind(
        shape=matrix.shape,
        nnz_estimate=nnz,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=True,
        drop_tol=drop_tol,
    )
    if decision.storage_kind != "csr":
        decision = SparseDecision(
            storage_kind="csr",
            reason="block assembly is explicitly sparse",
            backend=decision.backend,
            shape=matrix.shape,
            dense_nbytes=decision.dense_nbytes,
            csr_nbytes_estimate=decision.csr_nbytes_estimate,
            nnz_estimate=nnz,
            block_cols=decision.block_cols,
            drop_tol=drop_tol,
        )
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def _matvec_to_dense(
    matvec: Callable[[np.ndarray], np.ndarray],
    n: int,
    *,
    dtype=np.float64,
    block_cols: int = 32,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    n = int(n)
    block_cols = max(1, int(block_cols))
    dtype_np = np.dtype(dtype)
    dense = np.empty((n, n), dtype=dtype_np)
    for start in range(0, n, block_cols):
        width = min(block_cols, n - start)
        cols = np.zeros((n, width), dtype=dtype_np)
        cols[np.arange(start, start + width), np.arange(width)] = 1
        if matmat is not None:
            out = _host_array(matmat(cols), dtype=dtype)
        else:
            out_cols = [_host_array(matvec(cols[:, j]), dtype=dtype) for j in range(cols.shape[1])]
            out = np.column_stack(out_cols)
        dense[:, start : start + width] = out
    return dense


def build_operator_from_matvec(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    n: int,
    dtype=np.float64,
    backend: str | None = None,
    block_cols: int = 32,
    dense_max_mb: float = 128.0,
    csr_max_mb: float = 512.0,
    prefer_sparse_on_gpu: bool = True,
    force_sparse: bool = False,
    force_dense: bool = False,
    drop_tol: float = 0.0,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
    allow_operator_only: bool = True,
) -> SparseOperatorBundle:
    dense_nbytes = estimate_dense_nbytes((int(n), int(n)), dtype)
    backend_norm = _backend_name(backend)
    if allow_operator_only and dense_nbytes > int(max(0.0, float(dense_max_mb)) * 1e6):
        decision = SparseDecision(
            storage_kind="linear_operator",
            reason="dense assembly would exceed budget; keep operator-only fallback",
            backend=backend_norm,
            shape=(int(n), int(n)),
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=estimate_csr_nbytes((int(n), int(n)), int(n) * int(n), data_dtype=dtype),
            nnz_estimate=None,
            block_cols=int(block_cols),
            drop_tol=float(drop_tol),
        )

        def _op_matvec(x):
            return np.asarray(matvec(np.asarray(x)))

        operator = LinearOperator((int(n), int(n)), matvec=_op_matvec, dtype=np.dtype(dtype))
        return SparseOperatorBundle(matrix=None, operator=operator, metadata=decision)

    dense = _matvec_to_dense(matvec, int(n), dtype=dtype, block_cols=block_cols, matmat=matmat)
    return build_operator_from_dense(
        dense,
        backend=backend,
        dense_max_mb=dense_max_mb,
        csr_max_mb=csr_max_mb,
        prefer_sparse_on_gpu=prefer_sparse_on_gpu,
        force_sparse=force_sparse,
        force_dense=force_dense,
        drop_tol=drop_tol,
    )


def _normalize_pattern(pattern, *, shape: tuple[int, int] | None = None) -> sp.csr_matrix:
    if sp.issparse(pattern):
        pattern_csr = pattern.tocsr(copy=True)
    else:
        pattern_csr = sp.csr_matrix(np.asarray(pattern))
    if shape is not None and tuple(pattern_csr.shape) != tuple(shape):
        raise ValueError(f"pattern shape {pattern_csr.shape} does not match expected shape {shape}")
    if pattern_csr.ndim != 2:
        raise ValueError(f"expected a 2D sparsity pattern, got shape {pattern_csr.shape}")
    pattern_csr.sum_duplicates()
    if pattern_csr.nnz:
        pattern_csr.data = np.ones_like(pattern_csr.data, dtype=bool)
    pattern_csr = pattern_csr.astype(bool)
    pattern_csr.eliminate_zeros()
    return pattern_csr


def color_pattern_columns(pattern) -> list[list[int]]:
    """Greedily group columns whose declared row supports do not overlap.

    A single matvec with ones in all columns of a color recovers every value in
    that color when their row supports are disjoint. This is the sparse analogue
    of column-by-column probing, but it avoids materializing a dense identity or
    dense operator before converting to CSR.
    """

    pattern_csc = _normalize_pattern(pattern).tocsc()
    color_rows: list[set[int]] = []
    color_cols: list[list[int]] = []
    for col in range(pattern_csc.shape[1]):
        start, end = pattern_csc.indptr[col], pattern_csc.indptr[col + 1]
        rows = set(int(row) for row in pattern_csc.indices[start:end])
        if not rows:
            continue
        for color_index, used_rows in enumerate(color_rows):
            if rows.isdisjoint(used_rows):
                used_rows.update(rows)
                color_cols[color_index].append(col)
                break
        else:
            color_rows.append(set(rows))
            color_cols.append([col])
    return color_cols


def build_operator_from_pattern(
    matvec: Callable[[np.ndarray], np.ndarray],
    *,
    pattern,
    dtype=np.float64,
    backend: str | None = None,
    csr_max_mb: float = 512.0,
    drop_tol: float = 0.0,
    allow_operator_only: bool = False,
    max_colors: int | None = None,
) -> SparseOperatorBundle:
    """Materialize a sparse operator by probing a known sparsity pattern.

    The pattern must be a conservative structural superset of all nonzeros. Each
    color is evaluated with one combined seed vector; entries are then unpacked
    using the declared row supports. If the pattern misses a true nonzero, that
    value cannot be recovered, so callers should validate this path against the
    matrix-free operator before using it as a production backend.
    """

    dtype_np = np.dtype(dtype)
    pattern_csr = _normalize_pattern(pattern)
    n_rows, n_cols = pattern_csr.shape
    backend_norm = _backend_name(backend)
    dense_nbytes = estimate_dense_nbytes((n_rows, n_cols), dtype_np)
    csr_nbytes_estimate = estimate_csr_nbytes((n_rows, n_cols), int(pattern_csr.nnz), data_dtype=dtype_np)
    csr_cap = int(max(0.0, float(csr_max_mb)) * 1e6)
    if allow_operator_only and (csr_cap <= 0 or csr_nbytes_estimate > csr_cap):
        decision = SparseDecision(
            storage_kind="linear_operator",
            reason="pattern CSR estimate would exceed budget; keep operator-only fallback",
            backend=backend_norm,
            shape=(n_rows, n_cols),
            dense_nbytes=dense_nbytes,
            csr_nbytes_estimate=csr_nbytes_estimate,
            nnz_estimate=int(pattern_csr.nnz),
            drop_tol=float(drop_tol),
        )

        def _op_matvec(x):
            return np.asarray(matvec(np.asarray(x, dtype=dtype_np)))

        operator = LinearOperator((n_rows, n_cols), matvec=_op_matvec, dtype=dtype_np)
        return SparseOperatorBundle(matrix=None, operator=operator, metadata=decision)

    colors = color_pattern_columns(pattern_csr)
    if max_colors is not None and len(colors) > int(max_colors):
        raise ValueError(f"pattern probing would require {len(colors)} colors, exceeding max_colors={max_colors}")

    pattern_csc = pattern_csr.tocsc()
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []
    for cols in colors:
        seed = np.zeros(n_cols, dtype=dtype_np)
        seed[np.asarray(cols, dtype=np.intp)] = 1
        out = _host_array(matvec(seed), dtype=dtype_np)
        if out.shape != (n_rows,):
            raise ValueError(f"matvec returned shape {out.shape}; expected {(n_rows,)}")
        for col in cols:
            start, end = pattern_csc.indptr[col], pattern_csc.indptr[col + 1]
            rows = pattern_csc.indices[start:end]
            values = np.asarray(out[rows], dtype=dtype_np)
            if float(drop_tol) > 0.0:
                keep = np.abs(values) > float(drop_tol)
            else:
                keep = values != 0
            if not np.any(keep):
                continue
            rows_kept = np.asarray(rows[keep], dtype=np.int32)
            values_kept = np.asarray(values[keep], dtype=dtype_np)
            row_parts.append(rows_kept)
            col_parts.append(np.full(rows_kept.shape, int(col), dtype=np.int32))
            data_parts.append(values_kept)

    if data_parts:
        data = np.concatenate(data_parts)
        row_indices = np.concatenate(row_parts)
        col_indices = np.concatenate(col_parts)
    else:
        data = np.asarray([], dtype=dtype_np)
        row_indices = np.asarray([], dtype=np.int32)
        col_indices = np.asarray([], dtype=np.int32)
    matrix = sp.coo_matrix((data, (row_indices, col_indices)), shape=(n_rows, n_cols), dtype=dtype_np).tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    decision = SparseDecision(
        storage_kind="csr",
        reason=f"pattern-probed sparse materialization ({len(colors)} colors for {n_cols} columns)",
        backend=backend_norm,
        shape=(n_rows, n_cols),
        dense_nbytes=dense_nbytes,
        csr_nbytes_estimate=estimate_csr_nbytes((n_rows, n_cols), int(matrix.nnz), data_dtype=dtype_np),
        nnz_estimate=int(matrix.nnz),
        block_cols=len(colors),
        drop_tol=float(drop_tol),
    )
    return SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=decision)


def factorize_host_sparse_operator(
    operator: SparseOperatorBundle | np.ndarray | sp.spmatrix,
    *,
    kind: FactorKind = "lu",
    fill_factor: float = 10.0,
    drop_tol: float = 1.0e-4,
    permc_spec: str = "COLAMD",
    diag_pivot_thresh: float = 1.0,
) -> SparseFactorBundle:
    if isinstance(operator, SparseOperatorBundle):
        matrix = operator.matrix
        metadata = operator.metadata
    else:
        matrix = operator
        if sp.issparse(matrix):
            nnz = int(matrix.nnz)
            dense_dtype = matrix.dtype
            shape = tuple(matrix.shape)
        else:
            matrix = np.asarray(matrix)
            nnz = int(np.count_nonzero(matrix))
            dense_dtype = matrix.dtype
            shape = tuple(matrix.shape)
        metadata = SparseDecision(
            storage_kind="csr" if sp.issparse(matrix) else "dense",
            reason="factorized direct matrix input",
            backend="cpu",
            shape=shape,
            dense_nbytes=estimate_dense_nbytes(shape, dense_dtype),
            csr_nbytes_estimate=estimate_csr_nbytes(shape, nnz, data_dtype=dense_dtype),
            nnz_estimate=nnz,
        )

    if matrix is None:
        raise ValueError("factorize_host_sparse_operator requires a materialized matrix")

    if not sp.issparse(matrix):
        matrix = sp.csr_matrix(np.asarray(matrix))

    csc = matrix.tocsc()
    try:
        if kind == "lu":
            factor = splu(csc, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh)
        elif kind == "ilu":
            factor = spilu(csc, fill_factor=fill_factor, drop_tol=drop_tol, permc_spec=permc_spec)
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown factorization kind: {kind}")
    except RuntimeError as exc:
        raise RuntimeError(
            "Host sparse factorization failed. The assembled RHSMode=1 operator may be singular, "
            "ill-conditioned, or missing a pinned gauge/nullspace constraint for this solver branch. "
            "Use the default solver for parity runs, try solve_method='sparse_lsmr' only for diagnostic "
            "minimum-norm probes, or adjust SFINCS_JAX_EXPLICIT_SPARSE_* factorization controls."
        ) from exc
    factor_nbytes, factor_nnz = estimate_superlu_factor_storage(factor)
    return SparseFactorBundle(
        factor=factor,
        operator=SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=metadata),
        metadata=metadata,
        kind=kind,
        factor_nbytes_estimate=factor_nbytes,
        factor_nnz_estimate=factor_nnz,
    )
