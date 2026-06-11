from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
import time
from typing import Callable, Literal

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, spilu, splu

import jax

StorageKind = Literal["dense", "csr", "linear_operator"]
FactorKind = Literal[
    "lu",
    "ilu",
    "jacobi",
    "symbolic_block_lu",
    "symbolic_block_lu_coarse",
    "symbolic_block_schur_lu",
    "symbolic_superblock_lu",
    "symbolic_frontal_schur_lu",
]


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
    factor_s: float | None = None

    def solve(self, rhs) -> np.ndarray:
        rhs_host = _host_array(rhs, dtype=self.operator.matrix.dtype if self.operator.matrix is not None else None)
        sol = self.factor.solve(rhs_host)
        return np.asarray(sol)


@dataclass(frozen=True)
class SparseSymbolicAnalysis:
    """Reusable structural metadata for sparse operator/factor plans.

    This deliberately stores the symbolic facts that matter for production
    preconditioner decisions without owning a PETSc/MUMPS-style symbolic object:
    pattern fingerprint, row/column density, diagonal coverage, bandwidth,
    profile, and a bounded block plan over an optional structural permutation.
    Numeric factorization can then be gated and compared across runs using a
    stable structural key before a stronger native sparse factor is introduced.
    """

    shape: tuple[int, int]
    nnz: int
    pattern_hash: str
    ordering_kind: str
    ordering_hash: str
    bandwidth: int
    profile: int
    permuted_bandwidth: int
    permuted_profile: int
    diagonal_present: int
    diagonal_missing: int
    row_nnz_min: int
    row_nnz_max: int
    row_nnz_mean: float
    row_nnz_p95: float
    col_nnz_min: int
    col_nnz_max: int
    col_nnz_mean: float
    col_nnz_p95: float
    block_size_target: int
    block_count: int
    block_size_max: int
    block_nnz_max: int
    permutation: np.ndarray | None = None
    inverse_permutation: np.ndarray | None = None

    def cache_key(self) -> tuple[object, ...]:
        return (
            tuple(int(v) for v in self.shape),
            int(self.nnz),
            str(self.pattern_hash),
            str(self.ordering_kind),
            str(self.ordering_hash),
            int(self.block_size_target),
        )

    def to_dict(self, *, include_permutation: bool = False) -> dict[str, object]:
        out: dict[str, object] = {
            "shape": tuple(int(v) for v in self.shape),
            "nnz": int(self.nnz),
            "pattern_hash": str(self.pattern_hash),
            "ordering_kind": str(self.ordering_kind),
            "ordering_hash": str(self.ordering_hash),
            "bandwidth": int(self.bandwidth),
            "profile": int(self.profile),
            "permuted_bandwidth": int(self.permuted_bandwidth),
            "permuted_profile": int(self.permuted_profile),
            "diagonal_present": int(self.diagonal_present),
            "diagonal_missing": int(self.diagonal_missing),
            "row_nnz_min": int(self.row_nnz_min),
            "row_nnz_max": int(self.row_nnz_max),
            "row_nnz_mean": float(self.row_nnz_mean),
            "row_nnz_p95": float(self.row_nnz_p95),
            "col_nnz_min": int(self.col_nnz_min),
            "col_nnz_max": int(self.col_nnz_max),
            "col_nnz_mean": float(self.col_nnz_mean),
            "col_nnz_p95": float(self.col_nnz_p95),
            "block_size_target": int(self.block_size_target),
            "block_count": int(self.block_count),
            "block_size_max": int(self.block_size_max),
            "block_nnz_max": int(self.block_nnz_max),
        }
        if include_permutation and self.permutation is not None and self.inverse_permutation is not None:
            out["permutation"] = np.asarray(self.permutation, dtype=np.int64).tolist()
            out["inverse_permutation"] = np.asarray(self.inverse_permutation, dtype=np.int64).tolist()
        return out


@dataclass(frozen=True)
class SparseFactorAdmission:
    """Setup-time quality gate for an approximate sparse factor."""

    accepted: bool
    max_relative_residual: float
    median_relative_residual: float
    min_improvement_vs_identity: float
    probe_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "max_relative_residual": float(self.max_relative_residual),
            "median_relative_residual": float(self.median_relative_residual),
            "min_improvement_vs_identity": float(self.min_improvement_vs_identity),
            "probe_count": int(self.probe_count),
            "reason": str(self.reason),
        }


@dataclass(frozen=True)
class _JacobiFactor:
    inverse_diagonal: np.ndarray

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.inverse_diagonal.dtype)
        if rhs_np.ndim == 2:
            return rhs_np * self.inverse_diagonal[:, None]
        return rhs_np * self.inverse_diagonal


@dataclass(frozen=True)
class _DenseInverseFactor:
    inverse: np.ndarray

    def solve(self, rhs) -> np.ndarray:
        return np.asarray(self.inverse @ np.asarray(rhs, dtype=self.inverse.dtype), dtype=self.inverse.dtype)


@dataclass(frozen=True)
class _SymbolicBlockFactor:
    """Block-diagonal sparse factor in a reusable symbolic ordering."""

    blocks: tuple[tuple[int, int, int, int, object], ...]
    analysis: SparseSymbolicAnalysis
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    dtype: np.dtype
    overlap_size: int = 0

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.permutation.size),))
        rhs_perm = rhs_np[np.asarray(self.permutation, dtype=np.int64)]
        sol_perm = np.array(rhs_perm, copy=True)
        for center_start, center_stop, local_start, local_stop, factor in self.blocks:
            center_sl = slice(int(center_start), int(center_stop))
            local_sl = slice(int(local_start), int(local_stop))
            offset0 = int(center_start) - int(local_start)
            offset1 = int(center_stop) - int(local_start)
            try:
                local_sol = np.asarray(factor.solve(rhs_perm[local_sl]), dtype=self.dtype)
                sol_perm[center_sl] = local_sol[offset0:offset1]
            except Exception:
                # Fail soft: this is a preconditioner, not the residual gate.
                sol_perm[center_sl] = rhs_perm[center_sl]
        out = np.empty_like(sol_perm)
        out[np.asarray(self.permutation, dtype=np.int64)] = sol_perm
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicBlockCoarseFactor:
    """Block factor plus sparse Galerkin residual coarse correction."""

    local_factor: _SymbolicBlockFactor
    matrix: sp.csr_matrix
    coarse_basis: sp.csr_matrix
    coarse_factor: object
    coarse_matrix_nnz: int
    dtype: np.dtype
    coarse_damping: float = 1.0

    @property
    def analysis(self) -> SparseSymbolicAnalysis:
        return self.local_factor.analysis

    @property
    def overlap_size(self) -> int:
        return int(self.local_factor.overlap_size)

    @property
    def coarse_size(self) -> int:
        return int(self.coarse_basis.shape[1])

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.matrix.shape[0]),))
        z0 = np.asarray(self.local_factor.solve(rhs_np), dtype=self.dtype).reshape(rhs_np.shape)
        if self.coarse_basis.shape[1] == 0:
            return z0
        residual = rhs_np - np.asarray(self.matrix @ z0, dtype=self.dtype)
        coarse_rhs = np.asarray(self.coarse_basis.T @ residual, dtype=self.dtype).reshape((self.coarse_size,))
        try:
            coarse_delta = np.asarray(self.coarse_factor.solve(coarse_rhs), dtype=self.dtype).reshape((self.coarse_size,))
        except Exception:
            coarse_delta = np.zeros((self.coarse_size,), dtype=self.dtype)
        correction = np.asarray(self.coarse_basis @ coarse_delta, dtype=self.dtype).reshape(rhs_np.shape)
        out = z0 + float(self.coarse_damping) * correction
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, z0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SparseCoarseCorrectionFactor:
    """Wrap a sparse factor with a caller-provided residual coarse basis."""

    base_factor: object
    matrix: sp.csr_matrix
    coarse_basis: sp.csr_matrix
    coarse_factor: object
    dtype: np.dtype
    damping: float = 1.0

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.matrix.shape[0]),))
        try:
            z0 = np.asarray(self.base_factor.solve(rhs_np), dtype=self.dtype).reshape(rhs_np.shape)
        except Exception:
            z0 = np.array(rhs_np, copy=True)
        if self.coarse_basis.shape[1] == 0:
            return z0
        residual = rhs_np - np.asarray(self.matrix @ z0, dtype=self.dtype)
        coarse_rhs = np.asarray(self.coarse_basis.T @ residual, dtype=self.dtype).reshape((self.coarse_basis.shape[1],))
        try:
            coarse_delta = np.asarray(self.coarse_factor.solve(coarse_rhs), dtype=self.dtype).reshape((self.coarse_basis.shape[1],))
        except Exception:
            coarse_delta = np.zeros((self.coarse_basis.shape[1],), dtype=self.dtype)
        correction = np.asarray(self.coarse_basis @ coarse_delta, dtype=self.dtype).reshape(rhs_np.shape)
        out = z0 + float(self.damping) * correction
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, z0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicSchurBlock:
    indices: np.ndarray
    factor: object
    b_to_separator: sp.csr_matrix
    c_from_separator: sp.csr_matrix


@dataclass(frozen=True)
class _SymbolicBlockSchurFactor:
    """Block sparse factor with an explicit separator Schur complement.

    This mirrors the analysis/factor/solve split used by sparse direct solvers
    at a bounded scale: local block factors eliminate interior unknowns, while a
    compact separator system retains source/constraint and high-connectivity
    couplings that simple block Jacobi drops.
    """

    blocks: tuple[_SymbolicSchurBlock, ...]
    separator_indices: np.ndarray
    schur_factor: object
    dtype: np.dtype
    n: int
    analysis: SparseSymbolicAnalysis
    separator_count: int
    frontal_block_count: int = 0
    total_cross_nnz: int = 0
    selected_cross_nnz: int = 0
    cross_separator_fraction: float = 1.0
    factor_failures: int = 0

    @property
    def overlap_size(self) -> int:
        return 0

    @property
    def coarse_size(self) -> int:
        return int(self.separator_count)

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.n),))
        sep = np.asarray(self.separator_indices, dtype=np.int64)
        sep_count = int(sep.size)
        out = np.zeros((int(self.n),), dtype=self.dtype)
        if sep_count == 0:
            for block in self.blocks:
                idx = np.asarray(block.indices, dtype=np.int64)
                try:
                    out[idx] = np.asarray(block.factor.solve(rhs_np[idx]), dtype=self.dtype)
                except Exception:
                    out[idx] = rhs_np[idx]
            return out
        schur_rhs = np.array(rhs_np[sep], dtype=self.dtype, copy=True)
        block_solutions: list[tuple[np.ndarray, np.ndarray]] = []
        for block in self.blocks:
            idx = np.asarray(block.indices, dtype=np.int64)
            rhs_i = rhs_np[idx]
            try:
                y_i = np.asarray(block.factor.solve(rhs_i), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                y_i = np.array(rhs_i, dtype=self.dtype, copy=True)
            block_solutions.append((idx, y_i))
            if block.c_from_separator.shape[0] == sep_count:
                schur_rhs -= np.asarray(block.c_from_separator @ y_i, dtype=self.dtype).reshape((sep_count,))
        try:
            y_sep = np.asarray(self.schur_factor.solve(schur_rhs), dtype=self.dtype).reshape((sep_count,))
        except Exception:
            y_sep = np.zeros((sep_count,), dtype=self.dtype)
        out[sep] = y_sep
        for block, (idx, y_i) in zip(self.blocks, block_solutions, strict=True):
            rhs_corr = np.asarray(block.b_to_separator @ y_sep, dtype=self.dtype).reshape((idx.size,))
            try:
                delta = np.asarray(block.factor.solve(rhs_corr), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                delta = np.zeros((idx.size,), dtype=self.dtype)
            out[idx] = y_i - delta
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


@dataclass(frozen=True)
class _SymbolicSuperblock:
    indices: np.ndarray
    base_blocks: tuple[int, ...]
    factor: object


@dataclass(frozen=True)
class _SymbolicSuperblockFactor:
    """Bounded sparse-direct factor over grouped symbolic blocks.

    This is a native analogue of the first useful layer in multifrontal/sparse
    direct solvers: reuse a symbolic ordering, merge strongly coupled base
    blocks subject to a hard size cap, and factor each merged block exactly.
    Cross-superblock couplings are intentionally dropped, then measured by the
    setup residual admission gate before the factor is allowed into Krylov.
    """

    blocks: tuple[_SymbolicSuperblock, ...]
    analysis: SparseSymbolicAnalysis
    dtype: np.dtype
    n: int
    max_superblock_size: int
    max_superblock_blocks: int
    retained_cross_nnz: int
    dropped_cross_nnz: int
    retained_cross_fraction: float
    factor_failures: int

    @property
    def superblock_count(self) -> int:
        return int(len(self.blocks))

    @property
    def base_block_count(self) -> int:
        return int(self.analysis.block_count)

    def solve(self, rhs) -> np.ndarray:
        rhs_np = np.asarray(rhs, dtype=self.dtype).reshape((int(self.n),))
        out = np.array(rhs_np, dtype=self.dtype, copy=True)
        for block in self.blocks:
            idx = np.asarray(block.indices, dtype=np.int64)
            if idx.size == 0:
                continue
            try:
                out[idx] = np.asarray(block.factor.solve(rhs_np[idx]), dtype=self.dtype).reshape((idx.size,))
            except Exception:
                out[idx] = rhs_np[idx]
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=self.dtype)


class _DisjointSet:
    def __init__(self, sizes: np.ndarray) -> None:
        self.parent = np.arange(int(sizes.size), dtype=np.int64)
        self.rows = np.asarray(sizes, dtype=np.int64).copy()
        self.blocks = np.ones((int(sizes.size),), dtype=np.int64)

    def find(self, item: int) -> int:
        idx = int(item)
        parent = self.parent
        while int(parent[idx]) != idx:
            parent[idx] = parent[int(parent[idx])]
            idx = int(parent[idx])
        return idx

    def union_if_fits(
        self,
        a: int,
        b: int,
        *,
        max_rows: int,
        max_blocks: int,
    ) -> bool:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False
        rows = int(self.rows[root_a] + self.rows[root_b])
        blocks = int(self.blocks[root_a] + self.blocks[root_b])
        if rows > int(max_rows) or blocks > int(max_blocks):
            return False
        if self.rows[root_a] < self.rows[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.rows[root_a] = rows
        self.blocks[root_a] = blocks
        return True


def _build_symbolic_superblock_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_superblock_size: int = 32768,
    max_superblock_blocks: int = 8,
    min_cross_nnz: int = 1,
    min_retained_cross_fraction: float = 0.0,
    regularization_rel: float = 1.0e-12,
) -> tuple[_SymbolicSuperblockFactor, int, int]:
    """Build a bounded grouped-block sparse factor retaining dominant couplings."""

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    if n != int(matrix_csr.shape[1]):
        raise ValueError("symbolic superblock factor requires a square matrix")
    dtype = np.dtype(matrix_csr.dtype)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    base_block_count = int(analysis.block_count)
    if base_block_count <= 0:
        return (
            _SymbolicSuperblockFactor(
                blocks=tuple(),
                analysis=analysis,
                dtype=dtype,
                n=n,
                max_superblock_size=max(1, int(max_superblock_size)),
                max_superblock_blocks=max(1, int(max_superblock_blocks)),
                retained_cross_nnz=0,
                dropped_cross_nnz=0,
                retained_cross_fraction=1.0,
                factor_failures=0,
            ),
            0,
            0,
        )

    block_sizes = np.asarray(
        [
            min(block_size, max(0, n - int(block) * block_size))
            for block in range(base_block_count)
        ],
        dtype=np.int64,
    )
    dsu = _DisjointSet(block_sizes)
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    coo = matrix_perm.tocoo()
    row_block = np.asarray(coo.row // block_size, dtype=np.int64)
    col_block = np.asarray(coo.col // block_size, dtype=np.int64)
    cross = row_block != col_block
    edge_counts: list[tuple[int, int, int]] = []
    if np.any(cross):
        lo = np.minimum(row_block[cross], col_block[cross])
        hi = np.maximum(row_block[cross], col_block[cross])
        valid = (lo >= 0) & (hi < base_block_count)
        if np.any(valid):
            pair_id = lo[valid] * int(base_block_count) + hi[valid]
            unique, counts = np.unique(pair_id, return_counts=True)
            for pair, count in zip(unique, counts, strict=True):
                c = int(count)
                if c < int(min_cross_nnz):
                    continue
                a = int(pair // int(base_block_count))
                b = int(pair % int(base_block_count))
                edge_counts.append((c, a, b))
    edge_counts.sort(key=lambda item: (-item[0], item[1], item[2]))
    max_rows = max(1, int(max_superblock_size))
    max_blocks = max(1, int(max_superblock_blocks))
    for _, a, b in edge_counts:
        dsu.union_if_fits(a, b, max_rows=max_rows, max_blocks=max_blocks)

    groups: dict[int, list[int]] = {}
    for block in range(base_block_count):
        groups.setdefault(dsu.find(block), []).append(int(block))
    ordered_groups = sorted(groups.values(), key=lambda values: (min(values), len(values)))

    retained_cross = 0
    dropped_cross = 0
    if np.any(cross):
        for rb, cb in zip(row_block[cross], col_block[cross], strict=True):
            if dsu.find(int(rb)) == dsu.find(int(cb)):
                retained_cross += 1
            else:
                dropped_cross += 1
    cross_total = int(retained_cross + dropped_cross)
    retained_fraction = 1.0 if cross_total == 0 else float(retained_cross) / float(cross_total)
    min_retained = max(0.0, min(1.0, float(min_retained_cross_fraction)))
    if retained_fraction < min_retained:
        raise RuntimeError(
            "symbolic_superblock_lu retained insufficient cross-block coupling "
            f"({retained_fraction:.6g} < {min_retained:.6g}; "
            f"retained={int(retained_cross)} dropped={int(dropped_cross)} total={int(cross_total)})"
        )

    blocks: list[_SymbolicSuperblock] = []
    total_nbytes = 0
    total_nnz = 0
    factor_failures = 0
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))
    for group in ordered_groups:
        perm_positions: list[np.ndarray] = []
        for block in group:
            start = int(block) * block_size
            stop = min(n, start + block_size)
            if stop > start:
                perm_positions.append(np.arange(start, stop, dtype=np.int64))
        if not perm_positions:
            continue
        positions = np.concatenate(perm_positions).astype(np.int64, copy=False)
        indices = np.asarray(permutation[positions], dtype=np.int64)
        local = matrix_csr[indices, :][:, indices].tocsc()
        try:
            factor = splu(local, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            factor_failures += 1
            local_reg = (local + reg * sp.eye(local.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(local_reg, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                diagonal = np.asarray(local.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += int(nbytes or 0)
        total_nnz += int(nnz or 0)
        blocks.append(
            _SymbolicSuperblock(
                indices=indices,
                base_blocks=tuple(int(v) for v in group),
                factor=factor,
            )
        )

    factor = _SymbolicSuperblockFactor(
        blocks=tuple(blocks),
        analysis=analysis,
        dtype=dtype,
        n=n,
        max_superblock_size=int(max_rows),
        max_superblock_blocks=int(max_blocks),
        retained_cross_nnz=int(retained_cross),
        dropped_cross_nnz=int(dropped_cross),
        retained_cross_fraction=float(retained_fraction),
        factor_failures=int(factor_failures),
    )
    return factor, int(total_nbytes), int(total_nnz)


def _build_symbolic_frontal_schur_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_separator_cols: int = 1024,
    tail_size: int = 0,
    boundary_width: int = 1,
    high_degree_cols: int = 128,
    max_superblock_size: int = 8192,
    max_superblock_blocks: int = 8,
    min_cross_nnz: int = 1,
    min_cross_separator_fraction: float = 0.0,
    regularization_rel: float = 1.0e-12,
    max_dense_rhs_entries: int = 0,
) -> tuple[_SymbolicBlockSchurFactor, int, int]:
    """Build a bounded frontal/Schur elimination over symbolic block groups.

    Unlike ``symbolic_superblock_lu``, cross-group couplings are not simply
    dropped.  Their endpoints are promoted into a bounded separator, local
    interiors are eliminated, and the separator Schur complement retains the
    global coupling that local block factors miss.
    """

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    if n != int(matrix_csr.shape[1]):
        raise ValueError("symbolic frontal Schur factor requires a square matrix")
    dtype = np.dtype(matrix_csr.dtype)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    base_block_count = int(analysis.block_count)
    max_cols = max(0, min(int(max_separator_cols), n))
    if n == 0 or base_block_count <= 0:
        empty = _SymbolicBlockSchurFactor(
            blocks=tuple(),
            separator_indices=np.asarray([], dtype=np.int64),
            schur_factor=_DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype)),
            dtype=dtype,
            n=n,
            analysis=analysis,
            separator_count=0,
        )
        return empty, 0, 0

    block_sizes = np.asarray(
        [
            min(block_size, max(0, n - int(block) * block_size))
            for block in range(base_block_count)
        ],
        dtype=np.int64,
    )
    dsu = _DisjointSet(block_sizes)
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    coo = matrix_perm.tocoo()
    row_block = np.asarray(coo.row // block_size, dtype=np.int64)
    col_block = np.asarray(coo.col // block_size, dtype=np.int64)
    cross = row_block != col_block
    edge_counts: list[tuple[int, int, int]] = []
    if np.any(cross):
        lo = np.minimum(row_block[cross], col_block[cross])
        hi = np.maximum(row_block[cross], col_block[cross])
        valid = (lo >= 0) & (hi < base_block_count)
        if np.any(valid):
            pair_id = lo[valid] * int(base_block_count) + hi[valid]
            unique, counts = np.unique(pair_id, return_counts=True)
            for pair, count in zip(unique, counts, strict=True):
                c = int(count)
                if c < int(min_cross_nnz):
                    continue
                a = int(pair // int(base_block_count))
                b = int(pair % int(base_block_count))
                edge_counts.append((c, a, b))
    edge_counts.sort(key=lambda item: (-item[0], item[1], item[2]))
    max_rows = max(1, int(max_superblock_size))
    max_blocks = max(1, int(max_superblock_blocks))
    for _, a, b in edge_counts:
        dsu.union_if_fits(a, b, max_rows=max_rows, max_blocks=max_blocks)

    unresolved_cross = np.zeros((0,), dtype=bool)
    if np.any(cross):
        unresolved_cross = np.asarray(
            [dsu.find(int(rb)) != dsu.find(int(cb)) for rb, cb in zip(row_block[cross], col_block[cross], strict=True)],
            dtype=bool,
        )
    total_cross_nnz = int(np.count_nonzero(unresolved_cross))

    candidates: dict[int, tuple[int, int]] = {}

    def _add_candidate(index: int, priority: int, score: int = 0) -> None:
        idx = int(index)
        if idx < 0 or idx >= n or len(candidates) >= max(8 * max(1, max_cols), n):
            return
        value = (int(priority), int(score))
        old = candidates.get(idx)
        if old is None or value > old:
            candidates[idx] = value

    tail = max(0, min(int(tail_size), n))
    if tail:
        for idx in range(n - tail, n):
            _add_candidate(idx, 1000, n - idx)

    if total_cross_nnz:
        cross_rows = np.asarray(coo.row[cross], dtype=np.int64)[unresolved_cross]
        cross_cols = np.asarray(coo.col[cross], dtype=np.int64)[unresolved_cross]
        endpoints = np.concatenate((permutation[cross_rows], permutation[cross_cols])).astype(np.int64, copy=False)
        counts = np.bincount(endpoints, minlength=n)
        for idx in np.flatnonzero(counts):
            _add_candidate(int(idx), 900, int(counts[int(idx)]))

    high_degree = max(0, min(int(high_degree_cols), n))
    if high_degree:
        row_degree = np.diff(matrix_csr.indptr).astype(np.int64, copy=False)
        col_degree = np.diff(matrix_csr.tocsc().indptr).astype(np.int64, copy=False)
        degree = row_degree + col_degree
        top = np.argpartition(-degree, kth=high_degree - 1)[:high_degree]
        for idx in top:
            _add_candidate(int(idx), 700, int(degree[int(idx)]))

    boundary = max(0, int(boundary_width))
    if boundary:
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            edge = min(boundary, stop - start)
            for local in range(start, start + edge):
                _add_candidate(int(permutation[local]), 500, edge - (local - start))
            for local in range(max(start, stop - edge), stop):
                _add_candidate(int(permutation[local]), 500, local - max(start, stop - edge) + 1)

    if candidates and max_cols:
        ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
        separator = np.asarray(sorted(idx for idx, _ in ordered[:max_cols]), dtype=np.int64)
    else:
        separator = np.asarray([], dtype=np.int64)
    separator_set = set(int(v) for v in separator)
    sep_count = int(separator.size)
    selected_cross = 0
    if total_cross_nnz and sep_count:
        cross_rows_orig = np.asarray(coo.row[cross], dtype=np.int64)[unresolved_cross]
        cross_cols_orig = np.asarray(coo.col[cross], dtype=np.int64)[unresolved_cross]
        row_orig = permutation[cross_rows_orig]
        col_orig = permutation[cross_cols_orig]
        selected_cross = int(
            np.count_nonzero(
                np.fromiter(
                    ((int(r) in separator_set) or (int(c) in separator_set) for r, c in zip(row_orig, col_orig, strict=True)),
                    dtype=bool,
                    count=int(total_cross_nnz),
                )
            )
        )
    cross_fraction = 1.0 if total_cross_nnz == 0 else float(selected_cross) / float(total_cross_nnz)
    min_fraction = max(0.0, min(1.0, float(min_cross_separator_fraction)))
    if cross_fraction < min_fraction:
        raise RuntimeError(
            "symbolic_frontal_schur_lu selected insufficient cross-block separator coverage "
            f"({cross_fraction:.6g} < {min_fraction:.6g}; "
            f"selected={int(selected_cross)} total={int(total_cross_nnz)} separator={int(sep_count)})"
        )

    groups: dict[int, list[int]] = {}
    for block in range(base_block_count):
        groups.setdefault(dsu.find(block), []).append(int(block))
    ordered_groups = sorted(groups.values(), key=lambda values: (min(values), len(values)))

    if sep_count and int(max_dense_rhs_entries) > 0:
        separator_mask = np.zeros((n,), dtype=bool)
        separator_mask[separator] = True
        dense_rhs_entries = 0
        peak_dense_rhs_entries = 0
        for group in ordered_groups:
            group_rows = 0
            for block in group:
                start = int(block) * block_size
                stop = min(n, start + block_size)
                if stop <= start:
                    continue
                local_positions = np.arange(start, stop, dtype=np.int64)
                local_indices = np.asarray(permutation[local_positions], dtype=np.int64)
                group_rows += int(np.count_nonzero(~separator_mask[local_indices]))
            entries = int(group_rows) * int(sep_count)
            dense_rhs_entries += entries
            peak_dense_rhs_entries = max(peak_dense_rhs_entries, entries)
        if dense_rhs_entries > int(max_dense_rhs_entries):
            raise RuntimeError(
                "symbolic_frontal_schur_lu dense separator RHS work budget exceeded "
                f"({int(dense_rhs_entries)}>{int(max_dense_rhs_entries)}; "
                f"peak_block_entries={int(peak_dense_rhs_entries)} separator={int(sep_count)} "
                f"groups={int(len(ordered_groups))})"
            )

    schur = (
        matrix_csr[separator, :][:, separator].toarray().astype(dtype, copy=False)
        if sep_count
        else np.zeros((0, 0), dtype=dtype)
    )
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))
    blocks: list[_SymbolicSchurBlock] = []
    total_nbytes = 0
    total_nnz = 0
    factor_failures = 0

    for group in ordered_groups:
        positions: list[np.ndarray] = []
        for block in group:
            start = int(block) * block_size
            stop = min(n, start + block_size)
            if stop > start:
                positions.append(np.arange(start, stop, dtype=np.int64))
        if not positions:
            continue
        idx_all = np.asarray(permutation[np.concatenate(positions)], dtype=np.int64)
        if sep_count:
            idx = np.asarray([int(v) for v in idx_all if int(v) not in separator_set], dtype=np.int64)
        else:
            idx = idx_all
        if idx.size == 0:
            continue
        local = matrix_csr[idx, :][:, idx].tocsc()
        try:
            factor = splu(local, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            factor_failures += 1
            local_reg = (local + reg * sp.eye(local.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(local_reg, permc_spec="COLAMD", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                diagonal = np.asarray(local.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += int(nbytes or 0)
        total_nnz += int(nnz or 0)
        if sep_count:
            b_mat = matrix_csr[idx, :][:, separator].tocsr().astype(dtype, copy=False)
            c_mat = matrix_csr[separator, :][:, idx].tocsr().astype(dtype, copy=False)
            if b_mat.nnz and c_mat.nnz:
                b_dense = b_mat.toarray().astype(dtype, copy=False)
                try:
                    eliminated = np.asarray(factor.solve(b_dense), dtype=dtype)
                except Exception:
                    eliminated = np.zeros((idx.size, sep_count), dtype=dtype)
                schur -= np.asarray(c_mat @ eliminated, dtype=dtype)
            total_nbytes += estimate_csr_nbytes(b_mat.shape, int(b_mat.nnz), data_dtype=b_mat.dtype, index_dtype=b_mat.indices.dtype)
            total_nbytes += estimate_csr_nbytes(c_mat.shape, int(c_mat.nnz), data_dtype=c_mat.dtype, index_dtype=c_mat.indices.dtype)
            total_nnz += int(b_mat.nnz) + int(c_mat.nnz)
        else:
            b_mat = sp.csr_matrix((idx.size, 0), dtype=dtype)
            c_mat = sp.csr_matrix((0, idx.size), dtype=dtype)
        blocks.append(_SymbolicSchurBlock(indices=idx, factor=factor, b_to_separator=b_mat, c_from_separator=c_mat))

    if sep_count:
        schur_csc = sp.csc_matrix(schur)
        schur_csc.sum_duplicates()
        schur_max = float(np.max(np.abs(schur_csc.data))) if schur_csc.nnz else 0.0
        schur_reg = max(1.0e-14, float(regularization_rel) * max(1.0, schur_max))
        schur_reg_csc = (schur_csc + schur_reg * sp.eye(sep_count, dtype=dtype, format="csc")).tocsc()
        try:
            schur_factor = splu(schur_reg_csc, permc_spec="COLAMD", diag_pivot_thresh=1.0)
        except RuntimeError:
            schur_factor = _DenseInverseFactor(inverse=np.linalg.pinv(schur_reg_csc.toarray()))
        schur_nbytes, schur_nnz = estimate_superlu_factor_storage(schur_factor)
        if schur_nbytes is None and isinstance(schur_factor, _DenseInverseFactor):
            schur_nbytes = int(schur_factor.inverse.nbytes)
            schur_nnz = int(schur_factor.inverse.size)
        total_nbytes += estimate_csr_nbytes(schur_csc.shape, int(schur_csc.nnz), data_dtype=schur_csc.dtype, index_dtype=schur_csc.indices.dtype)
        total_nbytes += int(schur_nbytes or 0)
        total_nnz += int(schur_csc.nnz) + int(schur_nnz or 0)
    else:
        schur_factor = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype))

    factor = _SymbolicBlockSchurFactor(
        blocks=tuple(blocks),
        separator_indices=separator,
        schur_factor=schur_factor,
        dtype=dtype,
        n=n,
        analysis=analysis,
        separator_count=sep_count,
        frontal_block_count=len(blocks),
        total_cross_nnz=int(total_cross_nnz),
        selected_cross_nnz=int(selected_cross),
        cross_separator_fraction=float(cross_fraction),
        factor_failures=int(factor_failures),
    )
    return factor, int(total_nbytes), int(total_nnz)


def wrap_sparse_factor_with_coarse_correction(
    factor_bundle: SparseFactorBundle,
    coarse_basis: sp.spmatrix,
    *,
    damping: float = 1.0,
    regularization_rel: float = 1.0e-10,
) -> SparseFactorBundle:
    """Return a factor bundle with a supplied sparse Galerkin residual correction."""

    matrix = factor_bundle.operator.matrix
    if matrix is None:
        return factor_bundle
    matrix_csr = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))
    basis = coarse_basis.tocsr() if sp.issparse(coarse_basis) else sp.csr_matrix(np.asarray(coarse_basis))
    if basis.shape[0] != matrix_csr.shape[0] or basis.shape[1] == 0:
        return factor_bundle
    basis.sum_duplicates()
    basis.eliminate_zeros()
    coarse_matrix = (basis.T @ matrix_csr @ basis).tocsc()
    coarse_matrix.sum_duplicates()
    max_abs = float(np.max(np.abs(coarse_matrix.data))) if coarse_matrix.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))
    coarse_matrix_reg = (coarse_matrix + reg * sp.eye(coarse_matrix.shape[0], dtype=matrix_csr.dtype, format="csc")).tocsc()
    try:
        coarse_factor = splu(coarse_matrix_reg, permc_spec="COLAMD", diag_pivot_thresh=1.0)
    except RuntimeError:
        coarse_factor = _DenseInverseFactor(inverse=np.linalg.pinv(coarse_matrix_reg.toarray()))
    coarse_nbytes, coarse_nnz = estimate_superlu_factor_storage(coarse_factor)
    if coarse_nbytes is None and isinstance(coarse_factor, _DenseInverseFactor):
        coarse_nbytes = int(coarse_factor.inverse.nbytes)
        coarse_nnz = int(coarse_factor.inverse.size)
    basis_nbytes = estimate_csr_nbytes(
        basis.shape,
        int(basis.nnz),
        data_dtype=basis.dtype,
        index_dtype=basis.indices.dtype,
    )
    coarse_matrix_nbytes = estimate_csr_nbytes(
        coarse_matrix.shape,
        int(coarse_matrix.nnz),
        data_dtype=coarse_matrix.dtype,
        index_dtype=coarse_matrix.indices.dtype,
    )
    wrapper = _SparseCoarseCorrectionFactor(
        base_factor=factor_bundle.factor,
        matrix=matrix_csr,
        coarse_basis=basis,
        coarse_factor=coarse_factor,
        dtype=np.dtype(matrix_csr.dtype),
        damping=float(damping),
    )
    nbytes = None if factor_bundle.factor_nbytes_estimate is None else int(factor_bundle.factor_nbytes_estimate)
    nnz = None if factor_bundle.factor_nnz_estimate is None else int(factor_bundle.factor_nnz_estimate)
    total_nbytes = None if nbytes is None else int(nbytes + basis_nbytes + coarse_matrix_nbytes + int(coarse_nbytes or 0))
    total_nnz = None if nnz is None else int(nnz + basis.nnz + coarse_matrix.nnz + int(coarse_nnz or 0))
    return SparseFactorBundle(
        factor=wrapper,
        operator=factor_bundle.operator,
        metadata=factor_bundle.metadata,
        kind=factor_bundle.kind,
        factor_nbytes_estimate=total_nbytes,
        factor_nnz_estimate=total_nnz,
        factor_s=factor_bundle.factor_s,
    )


def _build_symbolic_block_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    overlap_size: int = 0,
    regularization_rel: float = 1.0e-12,
) -> tuple[_SymbolicBlockFactor, int, int]:
    matrix_csr = matrix.tocsr()
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    inverse_permutation = np.asarray(analysis.inverse_permutation, dtype=np.int64)
    if permutation.size != matrix_csr.shape[0]:
        raise ValueError("symbolic permutation size does not match matrix")
    matrix_perm = matrix_csr[permutation, :][:, permutation].tocsr()
    blocks: list[tuple[int, int, int, int, object]] = []
    total_nbytes = 0
    total_nnz = 0
    dtype = np.dtype(matrix_perm.dtype)
    block_size = max(1, int(analysis.block_size_target))
    overlap = max(0, int(overlap_size))
    n = int(matrix_perm.shape[0])
    for start in range(0, int(matrix_perm.shape[0]), block_size):
        stop = min(int(matrix_perm.shape[0]), start + block_size)
        local_start = max(0, int(start) - overlap)
        local_stop = min(n, int(stop) + overlap)
        block = matrix_perm[local_start:local_stop, local_start:local_stop].tocsc()
        if block.shape[0] == 0:
            continue
        try:
            factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            diagonal = np.asarray(block.diagonal(), dtype=np.float64)
            scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
            floor = max(1.0e-14, float(regularization_rel) * scale)
            sign = np.where(diagonal < 0.0, -1.0, 1.0)
            diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
            factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += 0 if nbytes is None else int(nbytes)
        total_nnz += 0 if nnz is None else int(nnz)
        blocks.append((int(start), int(stop), int(local_start), int(local_stop), factor))
    factor = _SymbolicBlockFactor(
        blocks=tuple(blocks),
        analysis=analysis,
        permutation=permutation,
        inverse_permutation=inverse_permutation,
        dtype=dtype,
        overlap_size=overlap,
    )
    return factor, int(total_nbytes), int(total_nnz)


def _build_symbolic_block_coarse_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    overlap_size: int = 0,
    coarse_max_cols: int = 256,
    coarse_probe_cols: int = 4,
    coarse_damping: float = 1.0,
    coarse_regularization_rel: float = 1.0e-10,
) -> tuple[_SymbolicBlockCoarseFactor, int, int]:
    """Build a bounded local block factor with block-indicator Galerkin coarse solve."""

    matrix_csr = matrix.tocsr()
    local_factor, local_nbytes, local_nnz = _build_symbolic_block_factor(
        matrix_csr,
        analysis=analysis,
        diag_pivot_thresh=float(diag_pivot_thresh),
        overlap_size=int(overlap_size),
    )
    n = int(matrix_csr.shape[0])
    if n == 0 or not local_factor.blocks:
        basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)
        coarse_factor: object = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=matrix_csr.dtype))
        return (
            _SymbolicBlockCoarseFactor(
                local_factor=local_factor,
                matrix=matrix_csr,
                coarse_basis=basis,
                coarse_factor=coarse_factor,
                coarse_matrix_nnz=0,
                dtype=np.dtype(matrix_csr.dtype),
                coarse_damping=float(coarse_damping),
            ),
            int(local_nbytes),
            int(local_nnz),
        )

    max_cols = max(1, int(coarse_max_cols))
    block_count = len(local_factor.blocks)
    coarse_cols = min(block_count, max_cols)
    rows_parts: list[np.ndarray] = []
    cols_parts: list[np.ndarray] = []
    data_parts: list[np.ndarray] = []
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    group_edges = np.linspace(0, block_count, coarse_cols + 1, dtype=np.int64)
    for coarse_col in range(coarse_cols):
        block_start = int(group_edges[coarse_col])
        block_stop = int(group_edges[coarse_col + 1])
        if block_stop <= block_start:
            continue
        owned_indices: list[np.ndarray] = []
        for block_index in range(block_start, block_stop):
            center_start, center_stop, _, _, _ = local_factor.blocks[block_index]
            owned_indices.append(permutation[int(center_start) : int(center_stop)])
        if not owned_indices:
            continue
        rows = np.concatenate(owned_indices).astype(np.int64, copy=False)
        if rows.size == 0:
            continue
        rows_parts.append(rows)
        cols_parts.append(np.full(rows.shape, int(coarse_col), dtype=np.int64))
        data_parts.append(np.full(rows.shape, 1.0 / np.sqrt(float(rows.size)), dtype=matrix_csr.dtype))
    if rows_parts:
        block_basis = sp.coo_matrix(
            (np.concatenate(data_parts), (np.concatenate(rows_parts), np.concatenate(cols_parts))),
            shape=(n, coarse_cols),
            dtype=matrix_csr.dtype,
        ).tocsr()
    else:
        block_basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)

    basis_parts: list[sp.csr_matrix] = [block_basis] if block_basis.shape[1] else []
    probe_cols = max(0, int(coarse_probe_cols))
    if probe_cols:
        probes = deterministic_sparse_probe_matrix(n, count=probe_cols, dtype=matrix_csr.dtype)
        residual_cols: list[np.ndarray] = []
        for col in range(probes.shape[1]):
            probe = np.asarray(probes[:, col], dtype=matrix_csr.dtype)
            local_sol = np.asarray(local_factor.solve(probe), dtype=matrix_csr.dtype)
            residual = probe - np.asarray(matrix_csr @ local_sol, dtype=matrix_csr.dtype)
            if basis_parts:
                # Remove the already represented block-average component so
                # residual modes carry only the missing fine/global coupling.
                block_projection = block_basis @ np.asarray(block_basis.T @ residual, dtype=matrix_csr.dtype)
                residual = residual - np.asarray(block_projection, dtype=matrix_csr.dtype)
            for prev in residual_cols:
                residual = residual - prev * float(np.dot(prev, residual))
            norm = float(np.linalg.norm(residual.astype(np.float64, copy=False)))
            if np.isfinite(norm) and norm > 1.0e-14:
                residual_cols.append(np.asarray(residual / norm, dtype=matrix_csr.dtype))
        if residual_cols:
            residual_basis = sp.csr_matrix(np.stack(residual_cols, axis=1))
            basis_parts.append(residual_basis)

    if basis_parts:
        basis = sp.hstack(basis_parts, format="csr")
    else:
        basis = sp.csr_matrix((n, 0), dtype=matrix_csr.dtype)

    coarse_matrix = (basis.T @ matrix_csr @ basis).tocsc()
    coarse_matrix.sum_duplicates()
    max_abs = float(np.max(np.abs(coarse_matrix.data))) if coarse_matrix.nnz else 0.0
    reg = max(1.0e-14, float(coarse_regularization_rel) * max(1.0, max_abs))
    coarse_matrix_reg = (coarse_matrix + reg * sp.eye(coarse_matrix.shape[0], dtype=matrix_csr.dtype, format="csc")).tocsc()
    try:
        coarse_factor = splu(coarse_matrix_reg, permc_spec="COLAMD", diag_pivot_thresh=1.0)
    except RuntimeError:
        coarse_factor = _DenseInverseFactor(inverse=np.linalg.pinv(coarse_matrix_reg.toarray()))
    coarse_nbytes, coarse_nnz = estimate_superlu_factor_storage(coarse_factor)
    if coarse_nbytes is None and isinstance(coarse_factor, _DenseInverseFactor):
        coarse_nbytes = int(coarse_factor.inverse.nbytes)
        coarse_nnz = int(coarse_factor.inverse.size)
    basis_nbytes = estimate_csr_nbytes(
        basis.shape,
        int(basis.nnz),
        data_dtype=basis.dtype,
        index_dtype=basis.indices.dtype,
    )
    coarse_matrix_nbytes = estimate_csr_nbytes(
        coarse_matrix.shape,
        int(coarse_matrix.nnz),
        data_dtype=coarse_matrix.dtype,
        index_dtype=coarse_matrix.indices.dtype,
    )
    factor = _SymbolicBlockCoarseFactor(
        local_factor=local_factor,
        matrix=matrix_csr,
        coarse_basis=basis,
        coarse_factor=coarse_factor,
        coarse_matrix_nnz=int(coarse_matrix.nnz),
        dtype=np.dtype(matrix_csr.dtype),
        coarse_damping=float(coarse_damping),
    )
    total_nbytes = int(local_nbytes) + int(basis_nbytes) + int(coarse_matrix_nbytes) + int(coarse_nbytes or 0)
    total_nnz = int(local_nnz) + int(basis.nnz) + int(coarse_matrix.nnz) + int(coarse_nnz or 0)
    return factor, int(total_nbytes), int(total_nnz)


def _select_symbolic_schur_separator(
    matrix: sp.csr_matrix,
    *,
    analysis: SparseSymbolicAnalysis,
    max_separator_cols: int,
    tail_size: int,
    boundary_width: int,
    high_degree_cols: int,
) -> np.ndarray:
    n = int(matrix.shape[0])
    max_cols = max(0, min(int(max_separator_cols), n))
    if n == 0 or max_cols == 0:
        return np.asarray([], dtype=np.int64)
    candidates: dict[int, tuple[int, int]] = {}

    def _add(index: int, priority: int, score: int = 0) -> None:
        idx = int(index)
        if idx < 0 or idx >= n:
            return
        old = candidates.get(idx)
        value = (int(priority), int(score))
        if old is None or value > old:
            candidates[idx] = value

    tail = max(0, min(int(tail_size), n))
    if tail:
        for idx in range(n - tail, n):
            _add(idx, 100, n - idx)

    boundary = max(0, int(boundary_width))
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    inverse_permutation = np.asarray(analysis.inverse_permutation, dtype=np.int64)
    block_size = max(1, int(analysis.block_size_target))
    if permutation.size == n and boundary:
        for start in range(0, n, block_size):
            stop = min(n, start + block_size)
            edge = min(boundary, stop - start)
            for local in range(start, start + edge):
                _add(int(permutation[local]), 40, edge - (local - start))
            for local in range(max(start, stop - edge), stop):
                _add(int(permutation[local]), 40, local - max(start, stop - edge) + 1)

    if permutation.size == n and inverse_permutation.size == n and block_size < n:
        coo = matrix.tocoo()
        row_pos = inverse_permutation[np.asarray(coo.row, dtype=np.int64)]
        col_pos = inverse_permutation[np.asarray(coo.col, dtype=np.int64)]
        row_block = row_pos // block_size
        col_block = col_pos // block_size
        cross = row_block != col_block
        if np.any(cross):
            endpoints = np.concatenate(
                [
                    np.asarray(coo.row[cross], dtype=np.int64),
                    np.asarray(coo.col[cross], dtype=np.int64),
                ]
            )
            counts = np.bincount(endpoints, minlength=n)
            cross_indices = np.flatnonzero(counts)
            for idx in cross_indices:
                _add(int(idx), 80, int(counts[int(idx)]))

    high_degree = max(0, int(high_degree_cols))
    if high_degree:
        row_degree = np.diff(matrix.indptr).astype(np.int64, copy=False)
        col_degree = np.diff(matrix.tocsc().indptr).astype(np.int64, copy=False)
        degree = row_degree + col_degree
        count = min(high_degree, n)
        if count:
            top = np.argpartition(-degree, kth=count - 1)[:count]
            for idx in top:
                _add(int(idx), 60, int(degree[int(idx)]))

    if not candidates:
        return np.asarray([], dtype=np.int64)
    ordered = sorted(candidates.items(), key=lambda item: (-item[1][0], -item[1][1], item[0]))
    selected = [idx for idx, _ in ordered[:max_cols]]
    selected.sort()
    return np.asarray(selected, dtype=np.int64)


def _build_symbolic_block_schur_factor(
    matrix: sp.spmatrix,
    *,
    analysis: SparseSymbolicAnalysis,
    diag_pivot_thresh: float,
    max_separator_cols: int = 256,
    tail_size: int = 0,
    boundary_width: int = 1,
    high_degree_cols: int = 64,
    regularization_rel: float = 1.0e-12,
) -> tuple[_SymbolicBlockSchurFactor, int, int]:
    """Build a bounded block-elimination factor with an explicit separator Schur solve."""

    matrix_csr = matrix.tocsr()
    n = int(matrix_csr.shape[0])
    dtype = np.dtype(matrix_csr.dtype)
    separator = _select_symbolic_schur_separator(
        matrix_csr,
        analysis=analysis,
        max_separator_cols=int(max_separator_cols),
        tail_size=int(tail_size),
        boundary_width=int(boundary_width),
        high_degree_cols=int(high_degree_cols),
    )
    separator_set = set(int(v) for v in separator)
    permutation = np.asarray(analysis.permutation, dtype=np.int64)
    if permutation.size != n:
        permutation = np.arange(n, dtype=np.int64)
    interior_order = np.asarray([int(v) for v in permutation if int(v) not in separator_set], dtype=np.int64)
    sep_count = int(separator.size)
    block_size = max(1, int(analysis.block_size_target))
    blocks: list[_SymbolicSchurBlock] = []
    total_nbytes = 0
    total_nnz = 0
    schur = matrix_csr[separator, :][:, separator].toarray().astype(dtype, copy=False) if sep_count else np.zeros((0, 0), dtype=dtype)
    max_abs = float(np.max(np.abs(matrix_csr.data))) if matrix_csr.nnz else 0.0
    reg = max(1.0e-14, float(regularization_rel) * max(1.0, max_abs))

    for start in range(0, int(interior_order.size), block_size):
        idx = np.asarray(interior_order[start : start + block_size], dtype=np.int64)
        if idx.size == 0:
            continue
        block = matrix_csr[idx, :][:, idx].tocsc()
        if block.shape[0] == 0:
            continue
        try:
            factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
        except RuntimeError:
            block = (block + reg * sp.eye(block.shape[0], dtype=dtype, format="csc")).tocsc()
            try:
                factor = splu(block, permc_spec="NATURAL", diag_pivot_thresh=float(diag_pivot_thresh))
            except RuntimeError:
                diagonal = np.asarray(block.diagonal(), dtype=np.float64)
                scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
                floor = max(1.0e-14, float(regularization_rel) * scale)
                sign = np.where(diagonal < 0.0, -1.0, 1.0)
                diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
                factor = _JacobiFactor(inverse_diagonal=np.asarray(1.0 / diagonal_safe, dtype=dtype))
        nbytes, nnz = estimate_superlu_factor_storage(factor)
        if nbytes is None and isinstance(factor, _JacobiFactor):
            nbytes = int(factor.inverse_diagonal.nbytes)
            nnz = int(factor.inverse_diagonal.size)
        total_nbytes += int(nbytes or 0)
        total_nnz += int(nnz or 0)
        if sep_count:
            b_mat = matrix_csr[idx, :][:, separator].tocsr().astype(dtype, copy=False)
            c_mat = matrix_csr[separator, :][:, idx].tocsr().astype(dtype, copy=False)
            if b_mat.nnz:
                b_dense = b_mat.toarray().astype(dtype, copy=False)
                try:
                    eliminated = np.asarray(factor.solve(b_dense), dtype=dtype)
                except Exception:
                    eliminated = np.zeros((idx.size, sep_count), dtype=dtype)
                if c_mat.nnz:
                    schur -= np.asarray(c_mat @ eliminated, dtype=dtype)
            total_nbytes += estimate_csr_nbytes(b_mat.shape, int(b_mat.nnz), data_dtype=b_mat.dtype, index_dtype=b_mat.indices.dtype)
            total_nbytes += estimate_csr_nbytes(c_mat.shape, int(c_mat.nnz), data_dtype=c_mat.dtype, index_dtype=c_mat.indices.dtype)
            total_nnz += int(b_mat.nnz) + int(c_mat.nnz)
        else:
            b_mat = sp.csr_matrix((idx.size, 0), dtype=dtype)
            c_mat = sp.csr_matrix((0, idx.size), dtype=dtype)
        blocks.append(_SymbolicSchurBlock(indices=idx, factor=factor, b_to_separator=b_mat, c_from_separator=c_mat))

    if sep_count:
        schur_csc = sp.csc_matrix(schur)
        schur_csc.sum_duplicates()
        schur_max = float(np.max(np.abs(schur_csc.data))) if schur_csc.nnz else 0.0
        schur_reg = max(1.0e-14, float(regularization_rel) * max(1.0, schur_max))
        schur_reg_csc = (schur_csc + schur_reg * sp.eye(sep_count, dtype=dtype, format="csc")).tocsc()
        try:
            schur_factor = splu(schur_reg_csc, permc_spec="COLAMD", diag_pivot_thresh=1.0)
        except RuntimeError:
            schur_factor = _DenseInverseFactor(inverse=np.linalg.pinv(schur_reg_csc.toarray()))
        schur_nbytes, schur_nnz = estimate_superlu_factor_storage(schur_factor)
        if schur_nbytes is None and isinstance(schur_factor, _DenseInverseFactor):
            schur_nbytes = int(schur_factor.inverse.nbytes)
            schur_nnz = int(schur_factor.inverse.size)
        total_nbytes += estimate_csr_nbytes(schur_csc.shape, int(schur_csc.nnz), data_dtype=schur_csc.dtype, index_dtype=schur_csc.indices.dtype)
        total_nbytes += int(schur_nbytes or 0)
        total_nnz += int(schur_csc.nnz) + int(schur_nnz or 0)
    else:
        schur_factor = _DenseInverseFactor(inverse=np.zeros((0, 0), dtype=dtype))

    factor = _SymbolicBlockSchurFactor(
        blocks=tuple(blocks),
        separator_indices=separator,
        schur_factor=schur_factor,
        dtype=dtype,
        n=n,
        analysis=analysis,
        separator_count=sep_count,
    )
    return factor, int(total_nbytes), int(total_nnz)


def _hash_int_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        arr_np = np.ascontiguousarray(arr)
        digest.update(str(arr_np.dtype).encode("ascii"))
        digest.update(np.asarray(arr_np.shape, dtype=np.int64).tobytes())
        digest.update(arr_np.tobytes())
    return digest.hexdigest()


def _safe_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return 0.0
    return float(np.percentile(values.astype(np.float64, copy=False), float(percentile)))


def deterministic_sparse_probe_matrix(
    size: int,
    *,
    count: int = 4,
    dtype=np.float64,
) -> np.ndarray:
    """Return deterministic probe RHS columns for sparse factor admission."""

    n = max(0, int(size))
    probe_count = max(1, int(count))
    dtype_np = np.dtype(dtype)
    if n == 0:
        return np.zeros((0, probe_count), dtype=dtype_np)
    probes: list[np.ndarray] = []
    probes.append(np.ones((n,), dtype=np.float64))
    probes.append(np.where(np.arange(n) % 2 == 0, 1.0, -1.0).astype(np.float64))
    if n == 1:
        probes.append(np.ones((1,), dtype=np.float64))
    else:
        probes.append(np.linspace(-1.0, 1.0, n, dtype=np.float64))
    rng = np.random.default_rng(8675309)
    while len(probes) < probe_count:
        probes.append(rng.standard_normal(n).astype(np.float64))
    normalized: list[np.ndarray] = []
    for probe in probes[:probe_count]:
        norm = float(np.linalg.norm(probe))
        if norm > 0.0 and np.isfinite(norm):
            probe = probe / norm
        normalized.append(np.asarray(probe, dtype=dtype_np))
    return np.stack(normalized, axis=1)


def admit_sparse_factor_against_operator(
    operator: SparseOperatorBundle | np.ndarray | sp.spmatrix,
    factor: SparseFactorBundle | object,
    *,
    probes: np.ndarray | None = None,
    probe_count: int = 4,
    max_relative_residual: float = 1.0e-2,
    min_improvement_vs_identity: float = 10.0,
) -> SparseFactorAdmission:
    """Accept an approximate factor only if it reduces the true setup residual.

    This is a cheap PETSc-style setup safeguard for native Python/JAX sparse
    preconditioners: before a candidate enters Krylov, apply it to deterministic
    RHS probes and measure ``||P M^{-1} b - b|| / ||b||`` against the actual
    materialized preconditioner matrix ``P``.  Weak block factors are rejected
    before they can amplify the nonlinear solve residual.
    """

    if isinstance(operator, SparseOperatorBundle):
        matrix = operator.matrix
    else:
        matrix = operator
    if matrix is None:
        return SparseFactorAdmission(
            accepted=False,
            max_relative_residual=float("inf"),
            median_relative_residual=float("inf"),
            min_improvement_vs_identity=0.0,
            probe_count=0,
            reason="missing_operator_matrix",
        )
    if not sp.issparse(matrix):
        matrix_csr = sp.csr_matrix(np.asarray(matrix))
    else:
        matrix_csr = matrix.tocsr()
    n_rows, n_cols = int(matrix_csr.shape[0]), int(matrix_csr.shape[1])
    if n_rows != n_cols:
        return SparseFactorAdmission(
            accepted=False,
            max_relative_residual=float("inf"),
            median_relative_residual=float("inf"),
            min_improvement_vs_identity=0.0,
            probe_count=0,
            reason="operator_not_square",
        )
    if probes is None:
        probes_use = deterministic_sparse_probe_matrix(n_rows, count=int(probe_count), dtype=matrix_csr.dtype)
    else:
        probes_use = np.asarray(probes, dtype=matrix_csr.dtype)
        if probes_use.ndim == 1:
            probes_use = probes_use.reshape((n_rows, 1))
    if probes_use.shape[0] != n_rows:
        raise ValueError(f"probe rows {probes_use.shape[0]} do not match operator size {n_rows}")

    if isinstance(factor, SparseFactorBundle):
        solve = factor.solve
    else:
        solve = factor.solve

    tiny = np.finfo(np.float64).tiny
    rel_residuals: list[float] = []
    improvements: list[float] = []
    for col in range(int(probes_use.shape[1])):
        rhs = np.asarray(probes_use[:, col], dtype=matrix_csr.dtype).reshape((n_rows,))
        rhs_norm = max(tiny, float(np.linalg.norm(rhs.astype(np.float64, copy=False))))
        try:
            y = np.asarray(solve(rhs), dtype=matrix_csr.dtype).reshape((n_rows,))
            residual = np.asarray(matrix_csr @ y - rhs, dtype=np.float64)
            identity_residual = np.asarray(matrix_csr @ rhs - rhs, dtype=np.float64)
            rel = float(np.linalg.norm(residual) / rhs_norm)
            identity_rel = float(np.linalg.norm(identity_residual) / rhs_norm)
        except Exception:
            rel = float("inf")
            identity_rel = 0.0
        if not np.isfinite(rel):
            rel = float("inf")
        rel_residuals.append(float(rel))
        if rel <= tiny:
            improvement = float("inf") if identity_rel > tiny else 1.0
        else:
            improvement = float(identity_rel / rel) if np.isfinite(identity_rel) else 0.0
        if not np.isfinite(improvement) and improvement != float("inf"):
            improvement = 0.0
        improvements.append(float(improvement))

    residuals_np = np.asarray(rel_residuals, dtype=np.float64)
    improvements_np = np.asarray(improvements, dtype=np.float64)
    max_rel = float(np.max(residuals_np)) if residuals_np.size else float("inf")
    median_rel = float(np.median(residuals_np)) if residuals_np.size else float("inf")
    min_improvement = float(np.min(improvements_np)) if improvements_np.size else 0.0
    accepted = bool(
        np.isfinite(max_rel)
        and max_rel <= float(max_relative_residual)
        and min_improvement >= float(min_improvement_vs_identity)
    )
    reason = "accepted" if accepted else "residual_or_improvement_gate_failed"
    return SparseFactorAdmission(
        accepted=accepted,
        max_relative_residual=max_rel,
        median_relative_residual=median_rel,
        min_improvement_vs_identity=min_improvement,
        probe_count=int(probes_use.shape[1]),
        reason=reason,
    )


def _bandwidth_and_profile(pattern_csr: sp.csr_matrix) -> tuple[int, int]:
    if pattern_csr.nnz == 0:
        return 0, 0
    rows = np.repeat(np.arange(pattern_csr.shape[0], dtype=np.int64), np.diff(pattern_csr.indptr))
    cols = pattern_csr.indices.astype(np.int64, copy=False)
    bandwidth = int(np.max(np.abs(rows - cols))) if rows.size else 0
    profile = 0
    for row in range(pattern_csr.shape[0]):
        start, end = int(pattern_csr.indptr[row]), int(pattern_csr.indptr[row + 1])
        if start == end:
            continue
        min_col = int(np.min(pattern_csr.indices[start:end]))
        if min_col < row:
            profile += int(row - min_col)
    return bandwidth, int(profile)


def analyze_sparse_symbolic_structure(
    matrix: np.ndarray | sp.spmatrix,
    *,
    ordering_kind: str = "rcm",
    block_size_target: int = 4096,
    max_permutation_size: int = 250_000,
) -> SparseSymbolicAnalysis:
    """Analyze a sparse pattern for reusable symbolic factor planning.

    The returned metadata is intentionally independent of any numerical factor.
    It can be used as a cache/admission key for Python/JAX-native sparse
    preconditioners and for comparing how close SFINCS-JAX is to
    PETSc/MUMPS/SuperLU-style symbolic reuse.
    """

    if sp.issparse(matrix):
        pattern = matrix.tocsr(copy=True)
    else:
        pattern = sp.csr_matrix(np.asarray(matrix))
    pattern.sum_duplicates()
    if pattern.nnz:
        pattern.data = np.ones_like(pattern.data, dtype=np.int8)
    pattern = pattern.astype(np.int8, copy=False)
    pattern.eliminate_zeros()
    n_rows, n_cols = int(pattern.shape[0]), int(pattern.shape[1])
    if n_rows != n_cols:
        raise ValueError(f"symbolic analysis requires a square matrix, got {pattern.shape}")

    row_counts = np.diff(pattern.indptr).astype(np.int64, copy=False)
    col_counts = np.diff(pattern.tocsc().indptr).astype(np.int64, copy=False)
    diagonal = pattern.diagonal()
    diagonal_present = int(np.count_nonzero(diagonal))
    diagonal_missing = int(n_rows - diagonal_present)
    bandwidth, profile = _bandwidth_and_profile(pattern)
    pattern_hash = _hash_int_arrays(
        np.asarray(pattern.shape, dtype=np.int64),
        pattern.indptr.astype(np.int64, copy=False),
        pattern.indices.astype(np.int64, copy=False),
    )

    ordering_norm = str(ordering_kind).strip().lower()
    permutation: np.ndarray | None
    if ordering_norm in {"natural", "none", "identity"} or n_rows == 0:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)
    elif ordering_norm in {"rcm", "reverse_cuthill_mckee", "reverse-cuthill-mckee"} and n_rows <= int(max_permutation_size):
        try:
            from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415

            permutation = np.asarray(reverse_cuthill_mckee(pattern, symmetric_mode=False), dtype=np.int64)
            ordering_norm = "rcm"
        except Exception:
            ordering_norm = "natural"
            permutation = np.arange(n_rows, dtype=np.int64)
    else:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)

    if permutation.size != n_rows or np.unique(permutation).size != n_rows:
        ordering_norm = "natural"
        permutation = np.arange(n_rows, dtype=np.int64)
    inverse_permutation = np.empty_like(permutation)
    inverse_permutation[permutation] = np.arange(n_rows, dtype=np.int64)
    ordering_hash = _hash_int_arrays(permutation.astype(np.int64, copy=False))
    pattern_perm = pattern[permutation, :][:, permutation].tocsr()
    permuted_bandwidth, permuted_profile = _bandwidth_and_profile(pattern_perm)

    block_size_target = max(1, int(block_size_target))
    block_count = int((n_rows + block_size_target - 1) // block_size_target) if n_rows else 0
    block_size_max = 0
    block_nnz_max = 0
    for start in range(0, n_rows, block_size_target):
        stop = min(n_rows, start + block_size_target)
        block_size_max = max(block_size_max, int(stop - start))
        block_nnz_max = max(block_nnz_max, int(pattern_perm[start:stop, start:stop].nnz))

    return SparseSymbolicAnalysis(
        shape=(n_rows, n_cols),
        nnz=int(pattern.nnz),
        pattern_hash=pattern_hash,
        ordering_kind=ordering_norm,
        ordering_hash=ordering_hash,
        bandwidth=int(bandwidth),
        profile=int(profile),
        permuted_bandwidth=int(permuted_bandwidth),
        permuted_profile=int(permuted_profile),
        diagonal_present=int(diagonal_present),
        diagonal_missing=int(diagonal_missing),
        row_nnz_min=int(np.min(row_counts)) if row_counts.size else 0,
        row_nnz_max=int(np.max(row_counts)) if row_counts.size else 0,
        row_nnz_mean=float(np.mean(row_counts)) if row_counts.size else 0.0,
        row_nnz_p95=_safe_percentile(row_counts, 95.0),
        col_nnz_min=int(np.min(col_counts)) if col_counts.size else 0,
        col_nnz_max=int(np.max(col_counts)) if col_counts.size else 0,
        col_nnz_mean=float(np.mean(col_counts)) if col_counts.size else 0.0,
        col_nnz_p95=_safe_percentile(col_counts, 95.0),
        block_size_target=int(block_size_target),
        block_count=int(block_count),
        block_size_max=int(block_size_max),
        block_nnz_max=int(block_nnz_max),
        permutation=permutation,
        inverse_permutation=inverse_permutation,
    )


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


def color_pattern_columns(pattern, *, max_colors: int | None = None) -> list[list[int]]:
    """Greedily group columns whose declared row supports do not overlap.

    A single matvec with ones in all columns of a color recovers every value in
    that color when their row supports are disjoint. This is the sparse analogue
    of column-by-column probing, but it avoids materializing a dense identity or
    dense operator before converting to CSR.
    """

    pattern_csc = _normalize_pattern(pattern).tocsc()
    max_colors_use = None if max_colors is None else max(1, int(max_colors))
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
            if max_colors_use is not None and len(color_cols) > max_colors_use:
                raise ValueError(
                    f"pattern probing would require more than max_colors={max_colors_use} colors"
                )
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
    color_batch: int = 1,
    matmat: Callable[[np.ndarray], np.ndarray] | None = None,
    progress_callback: Callable[[str], None] | None = None,
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
    if (not allow_operator_only) and (csr_cap <= 0 or csr_nbytes_estimate > csr_cap):
        raise MemoryError(
            "pattern CSR estimate would exceed budget "
            f"({csr_nbytes_estimate / 1.0e6:.3g} MB > {float(csr_max_mb):.3g} MB)"
        )
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

    if progress_callback is not None:
        progress_callback(
            "pattern-probe preflight "
            f"shape={n_rows}x{n_cols} pattern_nnz={int(pattern_csr.nnz)} "
            f"csr_estimate_mb={csr_nbytes_estimate / 1.0e6:.3g}"
        )
    colors = color_pattern_columns(pattern_csr, max_colors=max_colors)
    color_batch_use = max(1, int(color_batch))
    if matmat is None:
        color_batch_use = 1
    if progress_callback is not None:
        progress_callback(
            f"pattern-probe coloring complete colors={len(colors)} columns={n_cols} "
            f"color_batch={color_batch_use}"
        )

    pattern_csc = pattern_csr.tocsc()
    data_parts: list[np.ndarray] = []
    row_parts: list[np.ndarray] = []
    col_parts: list[np.ndarray] = []

    def _append_color_values(cols: list[int], out: np.ndarray) -> None:
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

    color_index = 0
    for batch_start in range(0, len(colors), color_batch_use):
        batch_colors = colors[batch_start : batch_start + color_batch_use]
        if color_batch_use == 1:
            cols = batch_colors[0]
            seed = np.zeros(n_cols, dtype=dtype_np)
            seed[np.asarray(cols, dtype=np.intp)] = 1
            out = _host_array(matvec(seed), dtype=dtype_np)
            _append_color_values(cols, np.asarray(out, dtype=dtype_np))
            color_index += 1
        else:
            seeds = np.zeros((n_cols, len(batch_colors)), dtype=dtype_np)
            for batch_col, cols in enumerate(batch_colors):
                seeds[np.asarray(cols, dtype=np.intp), batch_col] = 1
            assert matmat is not None
            out_batch = _host_array(matmat(seeds), dtype=dtype_np)
            expected_shape = (n_rows, len(batch_colors))
            if out_batch.shape != expected_shape:
                raise ValueError(f"matmat returned shape {out_batch.shape}; expected {expected_shape}")
            for batch_col, cols in enumerate(batch_colors):
                _append_color_values(cols, np.asarray(out_batch[:, batch_col], dtype=dtype_np))
            color_index += len(batch_colors)
        if progress_callback is not None and (color_index == len(colors) or color_index % 10 == 0):
            progress_callback(f"pattern-probe colors_done={color_index}/{len(colors)}")

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
    if progress_callback is not None:
        progress_callback(f"pattern-probe csr built nnz={int(matrix.nnz)}")
    decision = SparseDecision(
        storage_kind="csr",
        reason=(
            f"pattern-probed sparse materialization ({len(colors)} colors for {n_cols} columns; "
            f"color_batch={color_batch_use})"
        ),
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
    symbolic_analysis: SparseSymbolicAnalysis | None = None,
    symbolic_ordering_kind: str = "rcm",
    symbolic_block_size: int = 4096,
    symbolic_block_overlap: int = 0,
    symbolic_coarse_max_cols: int = 256,
    symbolic_coarse_probe_cols: int = 4,
    symbolic_coarse_damping: float = 1.0,
    symbolic_coarse_regularization_rel: float = 1.0e-10,
    symbolic_schur_max_separator_cols: int = 256,
    symbolic_schur_tail_size: int = 0,
    symbolic_schur_boundary_width: int = 1,
    symbolic_schur_high_degree_cols: int = 64,
    symbolic_schur_regularization_rel: float = 1.0e-12,
    symbolic_frontal_max_separator_cols: int = 1024,
    symbolic_frontal_tail_size: int = 0,
    symbolic_frontal_boundary_width: int = 1,
    symbolic_frontal_high_degree_cols: int = 128,
    symbolic_frontal_max_superblock_size: int = 8192,
    symbolic_frontal_max_superblock_blocks: int = 8,
    symbolic_frontal_min_cross_nnz: int = 1,
    symbolic_frontal_min_cross_separator_fraction: float = 0.0,
    symbolic_frontal_regularization_rel: float = 1.0e-12,
    symbolic_frontal_max_dense_rhs_entries: int = 0,
    symbolic_superblock_max_size: int = 32768,
    symbolic_superblock_max_blocks: int = 8,
    symbolic_superblock_min_cross_nnz: int = 1,
    symbolic_superblock_min_retained_cross_fraction: float = 0.0,
    symbolic_superblock_regularization_rel: float = 1.0e-12,
    symbolic_max_permutation_size: int = 250_000,
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
    factor_start_s = time.perf_counter()
    max_abs = float(np.max(np.abs(csc.data))) if csc.nnz else 0.0
    attempts_env = os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_ILU_ATTEMPTS", "").strip()
    try:
        ilu_attempts = int(attempts_env) if attempts_env else 1
    except ValueError:
        ilu_attempts = 1
    ilu_attempts = max(1, int(ilu_attempts))
    singular_reg_env = os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_ILU_SINGULAR_REG_REL", "").strip()
    try:
        singular_reg_rel = float(singular_reg_env) if singular_reg_env else 1.0e-10
    except ValueError:
        singular_reg_rel = 1.0e-10
    singular_reg_rel = max(0.0, float(singular_reg_rel))
    ilu_drop_tol_eff = float(drop_tol)
    fill_factor_eff = float(fill_factor)
    last_factor_error: RuntimeError | None = None

    def _regularized_csc(attempt: int) -> sp.csc_matrix:
        if attempt <= 0 or singular_reg_rel <= 0.0:
            return csc
        reg = max(1.0e-12, float(singular_reg_rel) * (10.0 ** (attempt - 1)) * max(1.0, max_abs))
        return (csc + reg * sp.eye(csc.shape[0], csc.shape[1], dtype=csc.dtype, format="csc")).tocsc()

    try:
        if kind == "lu":
            factor = splu(csc, permc_spec=permc_spec, diag_pivot_thresh=diag_pivot_thresh)
        elif kind == "ilu":
            factor = None
            for attempt in range(int(ilu_attempts)):
                try:
                    factor = spilu(
                        _regularized_csc(attempt),
                        fill_factor=fill_factor_eff,
                        drop_tol=ilu_drop_tol_eff,
                        permc_spec=permc_spec,
                        diag_pivot_thresh=diag_pivot_thresh,
                    )
                    break
                except RuntimeError as exc:
                    last_factor_error = exc
                    msg = str(exc).lower()
                    if not (
                        attempt + 1 < int(ilu_attempts)
                        and (("singular" in msg) or ("pivot" in msg) or ("dpivot" in msg) or ("zero" in msg))
                    ):
                        raise
                    ilu_drop_tol_eff = max(0.0, float(ilu_drop_tol_eff) * 0.1)
                    fill_factor_eff = max(float(fill_factor_eff), float(fill_factor) * 2.0, 20.0)
            if factor is None:
                assert last_factor_error is not None
                raise last_factor_error
        elif kind == "jacobi":
            diagonal = np.asarray(matrix.diagonal(), dtype=np.float64)
            if diagonal.size != int(matrix.shape[0]):
                raise RuntimeError("Jacobi preconditioner requires a square matrix diagonal")
            if np.any(~np.isfinite(diagonal)):
                raise RuntimeError("Jacobi preconditioner diagonal is non-finite")
            scale = max(1.0, float(np.max(np.abs(diagonal))) if diagonal.size else 1.0)
            floor = 1.0e-12 * scale
            sign = np.where(diagonal < 0.0, -1.0, 1.0)
            diagonal_safe = np.where(np.abs(diagonal) > floor, diagonal, sign * floor)
            factor = _JacobiFactor(inverse_diagonal=1.0 / diagonal_safe)
        elif kind in {
            "symbolic_block_lu",
            "symbolic_block_lu_coarse",
            "symbolic_block_schur_lu",
            "symbolic_superblock_lu",
            "symbolic_frontal_schur_lu",
        }:
            analysis = symbolic_analysis
            if analysis is None:
                analysis = analyze_sparse_symbolic_structure(
                    matrix,
                    ordering_kind=str(symbolic_ordering_kind),
                    block_size_target=int(symbolic_block_size),
                    max_permutation_size=int(symbolic_max_permutation_size),
                )
            if kind == "symbolic_block_schur_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_schur_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_separator_cols=int(symbolic_schur_max_separator_cols),
                    tail_size=int(symbolic_schur_tail_size),
                    boundary_width=int(symbolic_schur_boundary_width),
                    high_degree_cols=int(symbolic_schur_high_degree_cols),
                    regularization_rel=float(symbolic_schur_regularization_rel),
                )
            elif kind == "symbolic_frontal_schur_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_frontal_schur_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_separator_cols=int(symbolic_frontal_max_separator_cols),
                    tail_size=int(symbolic_frontal_tail_size),
                    boundary_width=int(symbolic_frontal_boundary_width),
                    high_degree_cols=int(symbolic_frontal_high_degree_cols),
                    max_superblock_size=int(symbolic_frontal_max_superblock_size),
                    max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                    min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                    min_cross_separator_fraction=float(symbolic_frontal_min_cross_separator_fraction),
                    regularization_rel=float(symbolic_frontal_regularization_rel),
                    max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                )
            elif kind == "symbolic_superblock_lu":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_superblock_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    max_superblock_size=int(symbolic_superblock_max_size),
                    max_superblock_blocks=int(symbolic_superblock_max_blocks),
                    min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                    min_retained_cross_fraction=float(symbolic_superblock_min_retained_cross_fraction),
                    regularization_rel=float(symbolic_superblock_regularization_rel),
                )
            elif kind == "symbolic_block_lu_coarse":
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_coarse_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    overlap_size=int(symbolic_block_overlap),
                    coarse_max_cols=int(symbolic_coarse_max_cols),
                    coarse_probe_cols=int(symbolic_coarse_probe_cols),
                    coarse_damping=float(symbolic_coarse_damping),
                    coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                )
            else:
                factor, symbolic_nbytes, symbolic_nnz = _build_symbolic_block_factor(
                    matrix,
                    analysis=analysis,
                    diag_pivot_thresh=float(diag_pivot_thresh),
                    overlap_size=int(symbolic_block_overlap),
                )
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown factorization kind: {kind}")
    except RuntimeError as exc:
        detail = str(exc).strip()
        raise RuntimeError(
            "Host sparse factorization failed. The assembled RHSMode=1 operator may be singular, "
            "ill-conditioned, or missing a pinned gauge/nullspace constraint for this solver branch. "
            "Use the default solver for parity runs, try solve_method='sparse_lsmr' only for diagnostic "
            "minimum-norm probes, or adjust SFINCS_JAX_EXPLICIT_SPARSE_* factorization controls. "
            f"Underlying factorization error: {detail}"
        ) from exc
    factor_s = time.perf_counter() - factor_start_s
    if kind == "jacobi":
        factor_nbytes, factor_nnz = int(factor.inverse_diagonal.nbytes), int(factor.inverse_diagonal.size)
    elif kind in {
        "symbolic_block_lu",
        "symbolic_block_lu_coarse",
        "symbolic_block_schur_lu",
        "symbolic_superblock_lu",
        "symbolic_frontal_schur_lu",
    }:
        factor_nbytes, factor_nnz = int(symbolic_nbytes), int(symbolic_nnz)
    else:
        factor_nbytes, factor_nnz = estimate_superlu_factor_storage(factor)
    return SparseFactorBundle(
        factor=factor,
        operator=SparseOperatorBundle(matrix=matrix, operator=_operator_from_matrix(matrix), metadata=metadata),
        metadata=metadata,
        kind=kind,
        factor_nbytes_estimate=factor_nbytes,
        factor_nnz_estimate=factor_nnz,
        factor_s=float(factor_s),
    )
