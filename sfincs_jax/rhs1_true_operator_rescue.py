from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from .explicit_sparse import SparseOperatorBundle
from .rhs1_block_operator import RHS1BlockLayout

__all__ = [
    "_ResidualCoarseHostSparsePreconditionerBundle",
    "_ResidualWindowHostSparsePreconditionerBundle",
    "_ReusableTrueActionColumnCache",
    "_TrueOperatorActiveSubmatrixPreconditionerBundle",
    "_TrueOperatorCoupledCoarseLSQPreconditionerBundle",
    "_TrueOperatorWindowLSQPreconditionerBundle",
    "_expand_sparse_graph_positions",
    "_parse_true_operator_window_specs",
    "_rhs1_additive_rescue_nbytes",
    "_sparse_factor_nbytes_estimate",
    "_true_operator_window_positions_from_residual",
]


@dataclass(frozen=True)
class _ResidualCoarseHostSparsePreconditionerBundle:
    """Low-rank residual coarse correction around an existing host factor."""

    base_factor: object
    operator: SparseOperatorBundle
    z_basis: np.ndarray
    az_basis: np.ndarray
    coarse_inverse: np.ndarray
    kind: str
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(self.base_factor.solve(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(self.operator.matvec(y_base), dtype=np.float64).reshape((-1,))
        coarse_rhs = np.asarray(self.az_basis.T @ residual, dtype=np.float64).reshape((-1,))
        coeff = np.asarray(self.coarse_inverse @ coarse_rhs, dtype=np.float64).reshape((-1,))
        return y_base + np.asarray(self.z_basis @ coeff, dtype=np.float64).reshape((-1,))


@dataclass(frozen=True)
class _ResidualWindowHostSparsePreconditionerBundle:
    """Residual-localized active kinetic window correction around a base factor."""

    base_factor: object
    operator: SparseOperatorBundle
    window_positions: tuple[np.ndarray, ...]
    window_factors: tuple[object, ...]
    kind: str
    coefficient_mode: str = "additive"
    regularization: float = 1.0e-12
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        out = np.asarray(self.base_factor.solve(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(self.operator.matvec(out), dtype=np.float64).reshape((-1,))
        correction_columns: list[np.ndarray] = []
        for positions, factor in zip(self.window_positions, self.window_factors, strict=True):
            correction = np.asarray(factor.solve(residual[positions]), dtype=np.float64).reshape((-1,))
            column = np.zeros_like(out)
            column[positions] = correction
            correction_columns.append(column)
        if not correction_columns:
            return out
        if str(self.coefficient_mode).strip().lower().replace("-", "_") in {
            "least_squares",
            "normal",
            "normal_equations",
        }:
            z_basis = np.column_stack(correction_columns)
            az_basis = np.column_stack(
                [np.asarray(self.operator.matvec(z_basis[:, i]), dtype=np.float64).reshape((-1,)) for i in range(z_basis.shape[1])]
            )
            normal = np.asarray(az_basis.T @ az_basis, dtype=np.float64)
            rhs = np.asarray(az_basis.T @ residual, dtype=np.float64).reshape((-1,))
            normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
            reg = max(float(abs(self.regularization)), 1.0e-14) * normal_scale
            try:
                coeff = np.linalg.solve(normal + reg * np.eye(normal.shape[0], dtype=np.float64), rhs)
            except Exception:  # noqa: BLE001
                coeff = np.linalg.pinv(normal, rcond=max(float(abs(self.regularization)), 1.0e-14)) @ rhs
            return out + np.asarray(z_basis @ coeff, dtype=np.float64).reshape((-1,))
        for column in correction_columns:
            out += column
        return out


@dataclass(frozen=True)
class _TrueOperatorWindowLSQPreconditionerBundle:
    """Residual-window correction using columns of the true active operator."""

    base_factor: object
    true_matvec: Callable[[np.ndarray], np.ndarray]
    window_positions: np.ndarray
    a_window: object
    inv_column_scale: np.ndarray
    solve_normal: Callable[[np.ndarray], np.ndarray]
    kind: str
    regularization: float = 1.0e-12
    damping: bool = False
    beta_max: float = 10.0
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(self.base_factor.solve(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(self.true_matvec(y_base), dtype=np.float64).reshape((-1,))
        a_scaled_t_residual = np.asarray(self.a_window.T @ residual, dtype=np.float64).reshape((-1,))
        scaled_delta = np.asarray(self.solve_normal(a_scaled_t_residual), dtype=np.float64).reshape((-1,))
        delta = np.asarray(self.inv_column_scale * scaled_delta, dtype=np.float64).reshape((-1,))
        out = y_base.copy()
        out[np.asarray(self.window_positions, dtype=np.int64)] += delta
        if bool(self.damping):
            a_out = np.asarray(self.true_matvec(out), dtype=np.float64).reshape((-1,))
            denom = float(np.dot(a_out, a_out))
            if np.isfinite(denom) and denom > 0.0:
                beta = float(np.dot(a_out, arr) / denom)
                if float(self.beta_max) > 0.0:
                    beta = float(np.clip(beta, -float(self.beta_max), float(self.beta_max)))
            else:
                beta = 0.0
            out = float(beta) * out
        return out


@dataclass(frozen=True)
class _TrueOperatorActiveSubmatrixPreconditionerBundle:
    """Additive active-block correction using the true local operator block."""

    base_factor: object
    true_matvec: Callable[[np.ndarray], np.ndarray]
    block_positions: np.ndarray
    a_window: object
    solve_block: Callable[[np.ndarray], np.ndarray]
    kind: str
    damping: bool = True
    alpha_clip: float = 10.0
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(self.base_factor.solve(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(self.true_matvec(y_base), dtype=np.float64).reshape((-1,))
        positions = np.asarray(self.block_positions, dtype=np.int64)
        delta = np.asarray(self.solve_block(residual[positions]), dtype=np.float64).reshape((-1,))
        alpha = 1.0
        if bool(self.damping):
            a_delta = np.asarray(self.a_window @ delta, dtype=np.float64).reshape((-1,))
            denom = float(np.dot(a_delta, a_delta))
            if np.isfinite(denom) and denom > 1.0e-300:
                alpha = float(np.dot(a_delta, residual) / denom)
                clip = max(0.0, float(self.alpha_clip))
                if clip > 0.0:
                    alpha = float(np.clip(alpha, -clip, clip))
            else:
                alpha = 0.0
        out = y_base.copy()
        out[positions] += float(alpha) * delta
        return out


class _ReusableTrueActionColumnCache:
    """Bounded one-hot true-operator action cache for active-block builders."""

    def __init__(
        self,
        *,
        true_matvec: Callable[[np.ndarray], np.ndarray],
        true_matmat: Callable[[np.ndarray], np.ndarray] | None,
        n: int,
        max_nbytes: int,
        enabled: bool = True,
    ) -> None:
        self._true_matvec = true_matvec
        self._true_matmat = true_matmat
        self._n = int(n)
        self._max_nbytes = max(0, int(max_nbytes))
        self._enabled = bool(enabled)
        self._columns: dict[int, np.ndarray] = {}
        self.hits = 0
        self.misses = 0
        self.batches = 0
        self.bypass_calls = 0
        self.stored_nbytes = 0

    def matvec(self, vec: np.ndarray) -> np.ndarray:
        return np.asarray(self._true_matvec(np.asarray(vec, dtype=np.float64)), dtype=np.float64).reshape((-1,))

    def _uncached_matmat(self, mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float64)
        if self._true_matmat is not None and arr.ndim == 2 and arr.shape[1] > 1:
            return np.asarray(self._true_matmat(arr), dtype=np.float64)
        return np.column_stack([self.matvec(arr[:, j]) for j in range(arr.shape[1])])

    @staticmethod
    def _one_hot_positions(arr: np.ndarray) -> tuple[int, ...] | None:
        if arr.ndim != 2:
            return None
        positions: list[int] = []
        for col in range(int(arr.shape[1])):
            nz = np.flatnonzero(arr[:, col] != 0.0)
            if nz.size != 1:
                return None
            pos = int(nz[0])
            if float(arr[pos, col]) != 1.0:
                return None
            positions.append(pos)
        return tuple(positions)

    def matmat(self, mat: np.ndarray) -> np.ndarray:
        arr = np.asarray(mat, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] != self._n:
            self.bypass_calls += 1
            return self._uncached_matmat(arr)
        positions = self._one_hot_positions(arr)
        if not self._enabled or positions is None:
            self.bypass_calls += 1
            return self._uncached_matmat(arr)

        out = np.empty((self._n, int(arr.shape[1])), dtype=np.float64)
        missing_positions: list[int] = []
        missing_columns: list[int] = []
        for col, pos in enumerate(positions):
            cached = self._columns.get(int(pos))
            if cached is None:
                missing_positions.append(int(pos))
                missing_columns.append(int(col))
                continue
            out[:, col] = cached
            self.hits += 1
        if missing_positions:
            basis = np.zeros((self._n, len(missing_positions)), dtype=np.float64)
            basis[np.asarray(missing_positions, dtype=np.int64), np.arange(len(missing_positions))] = 1.0
            y_missing = np.asarray(self._uncached_matmat(basis), dtype=np.float64)
            if y_missing.shape != (self._n, len(missing_positions)):
                self.bypass_calls += 1
                return self._uncached_matmat(arr)
            self.batches += 1
            for local_col, (global_col, pos) in enumerate(zip(missing_columns, missing_positions, strict=True)):
                column = np.asarray(y_missing[:, local_col], dtype=np.float64).reshape((-1,))
                out[:, global_col] = column
                self.misses += 1
                next_nbytes = int(self.stored_nbytes + column.nbytes)
                if self._max_nbytes <= 0 or next_nbytes <= self._max_nbytes:
                    self._columns[int(pos)] = column.copy()
                    self.stored_nbytes = next_nbytes
        return out

    def metadata(self) -> dict[str, object]:
        return {
            "enabled": bool(self._enabled),
            "max_nbytes": int(self._max_nbytes),
            "stored_columns": int(len(self._columns)),
            "stored_nbytes": int(self.stored_nbytes),
            "hits": int(self.hits),
            "misses": int(self.misses),
            "batches": int(self.batches),
            "bypass_calls": int(self.bypass_calls),
        }


@dataclass(frozen=True)
class _TrueOperatorCoupledCoarseLSQPreconditionerBundle:
    """Coupled true-operator coarse correction around an existing host factor."""

    base_factor: object
    true_matvec: Callable[[np.ndarray], np.ndarray]
    z_basis: object
    a_basis: object
    inv_column_scale: np.ndarray
    solve_normal: Callable[[np.ndarray], np.ndarray]
    kind: str
    regularization: float = 1.0e-12
    damping: bool = False
    beta_max: float = 10.0
    factor_nbytes_estimate: int | None = None
    factor_nnz_estimate: int | None = None
    factor_s: float | None = None
    metadata: dict[str, object] | None = None

    def solve(self, rhs) -> np.ndarray:
        arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
        y_base = np.asarray(self.base_factor.solve(arr), dtype=np.float64).reshape((-1,))
        residual = arr - np.asarray(self.true_matvec(y_base), dtype=np.float64).reshape((-1,))
        a_scaled_t_residual = np.asarray(self.a_basis.T @ residual, dtype=np.float64).reshape((-1,))
        scaled_coeff = np.asarray(self.solve_normal(a_scaled_t_residual), dtype=np.float64).reshape((-1,))
        coeff = np.asarray(self.inv_column_scale * scaled_coeff, dtype=np.float64).reshape((-1,))
        out = y_base + np.asarray(self.z_basis @ coeff, dtype=np.float64).reshape((-1,))
        if bool(self.damping):
            a_out = np.asarray(self.true_matvec(out), dtype=np.float64).reshape((-1,))
            denom = float(np.dot(a_out, a_out))
            if np.isfinite(denom) and denom > 0.0:
                beta = float(np.dot(a_out, arr) / denom)
                if float(self.beta_max) > 0.0:
                    beta = float(np.clip(beta, -float(self.beta_max), float(self.beta_max)))
            else:
                beta = 0.0
            out = float(beta) * out
        return out


def _sparse_factor_nbytes_estimate(factor: object) -> int:
    total = 0
    for attr in ("L", "U"):
        matrix = getattr(factor, attr, None)
        if matrix is None:
            continue
        total += int(getattr(matrix, "data", np.asarray([])).nbytes)
        total += int(getattr(matrix, "indices", np.asarray([])).nbytes)
        total += int(getattr(matrix, "indptr", np.asarray([])).nbytes)
    return int(total)


def _rhs1_additive_rescue_nbytes(factor_bundle: object, max_additional_mb: float) -> int:
    """Return a total preconditioner budget from an additive rescue cap.

    RHSMode=1 true-operator rescue builders record total storage, including
    the already-built base factor. Environment caps for rescue layers are
    interpreted as additional storage budgets so a small coarse correction can
    still be evaluated on top of a multi-GB base preconditioner.
    """

    add_nbytes = int(max(0.0, float(max_additional_mb)) * 1024.0 * 1024.0)
    if add_nbytes <= 0:
        return 0
    base_nbytes = int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    return int(base_nbytes + add_nbytes)


def _expand_sparse_graph_positions(
    matrix_csr: Any,
    positions: np.ndarray,
    *,
    depth: int,
    max_size: int,
) -> np.ndarray | None:
    """Expand active positions by sparse row/column adjacency with a hard cap."""

    if int(depth) <= 0:
        return np.unique(np.asarray(positions, dtype=np.int64))
    selected = set(int(v) for v in np.asarray(positions, dtype=np.int64).reshape((-1,)))
    frontier = np.asarray(sorted(selected), dtype=np.int64)
    matrix_csc = matrix_csr.tocsc()
    n = int(matrix_csr.shape[0])
    for _ in range(int(depth)):
        if frontier.size == 0:
            break
        neighbors: set[int] = set()
        for row in frontier:
            if int(row) < 0 or int(row) >= n:
                continue
            row_start = int(matrix_csr.indptr[int(row)])
            row_stop = int(matrix_csr.indptr[int(row) + 1])
            neighbors.update(int(v) for v in matrix_csr.indices[row_start:row_stop])
            col_start = int(matrix_csc.indptr[int(row)])
            col_stop = int(matrix_csc.indptr[int(row) + 1])
            neighbors.update(int(v) for v in matrix_csc.indices[col_start:col_stop])
        neighbors.difference_update(selected)
        if not neighbors:
            break
        selected.update(neighbors)
        if len(selected) > int(max_size):
            return None
        frontier = np.asarray(sorted(neighbors), dtype=np.int64)
    return np.asarray(sorted(selected), dtype=np.int64)


def _parse_true_operator_window_specs(spec: str, *, layout: RHS1BlockLayout) -> tuple[tuple[int, int, int], ...]:
    """Parse comma-separated ``species:x:ell`` residual-window targets."""

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
        if not (
            0 <= int(species) < int(layout.n_species)
            and 0 <= int(x_index) < int(layout.n_x)
            and 0 <= int(ell) < int(layout.n_xi)
        ):
            continue
        triples.append((int(species), int(x_index), int(ell)))
    return tuple(dict.fromkeys(triples))


def _true_operator_window_positions_from_residual(
    *,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    max_windows: int,
    x_radius: int,
    ell_radius: int,
    include_tail: bool,
    explicit_specs: tuple[tuple[int, int, int], ...] = (),
) -> tuple[np.ndarray, tuple[dict[str, object], ...]]:
    """Select active positions for true-operator residual-window probing."""

    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    if active_indices is None:
        active_np = np.arange(int(layout.total_size), dtype=np.int64)
    else:
        active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if residual_np.shape != active_np.shape:
        return np.zeros((0,), dtype=np.int64), ()
    kinetic_mask = active_np < int(layout.f_size)
    if not bool(np.any(kinetic_mask)):
        return np.zeros((0,), dtype=np.int64), ()
    kinetic_full = active_np[kinetic_mask]
    kinetic_positions = np.flatnonzero(kinetic_mask)
    decoded = layout.decode_kinetic_indices(kinetic_full)
    specs: list[tuple[int, int, int]] = list(explicit_specs)
    if not specs:
        kinetic_square = np.square(residual_np[kinetic_positions])
        combo_index = (decoded.species * int(layout.n_x) + decoded.x) * int(layout.n_xi) + decoded.ell
        combo_energy = np.bincount(
            combo_index,
            weights=kinetic_square,
            minlength=int(layout.n_species * layout.n_x * layout.n_xi),
        )
        top_combo = np.argsort(combo_energy)[::-1]
        top_combo = top_combo[combo_energy[top_combo] > 0.0][: max(1, int(max_windows))]
        for combo in top_combo:
            ell = int(combo % int(layout.n_xi))
            sx = int(combo // int(layout.n_xi))
            x_index = int(sx % int(layout.n_x))
            species = int(sx // int(layout.n_x))
            specs.append((species, x_index, ell))
    if not specs:
        return np.zeros((0,), dtype=np.int64), ()

    selected_positions: list[np.ndarray] = []
    metadata: list[dict[str, object]] = []
    used: set[int] = set()
    total_energy = float(np.sum(np.square(residual_np)))
    for species, x_center, ell_center in specs[: max(1, int(max_windows))]:
        x_lo = max(0, int(x_center) - max(0, int(x_radius)))
        x_hi = min(int(layout.n_x) - 1, int(x_center) + max(0, int(x_radius)))
        ell_lo = max(0, int(ell_center) - max(0, int(ell_radius)))
        ell_hi = min(int(layout.n_xi) - 1, int(ell_center) + max(0, int(ell_radius)))
        mask = (
            (decoded.species == int(species))
            & (decoded.x >= int(x_lo))
            & (decoded.x <= int(x_hi))
            & (decoded.ell >= int(ell_lo))
            & (decoded.ell <= int(ell_hi))
        )
        positions = kinetic_positions[mask]
        if positions.size == 0:
            continue
        positions = np.asarray([int(pos) for pos in positions if int(pos) not in used], dtype=np.int64)
        if positions.size == 0:
            continue
        used.update(int(pos) for pos in positions)
        selected_positions.append(positions)
        energy = float(np.sum(np.square(residual_np[positions])))
        metadata.append(
            {
                "species": int(species),
                "x_center": int(x_center),
                "ell_center": int(ell_center),
                "x_range": (int(x_lo), int(x_hi)),
                "ell_range": (int(ell_lo), int(ell_hi)),
                "size": int(positions.size),
                "residual_energy_fraction": float(energy / total_energy) if total_energy > 0.0 else 0.0,
            }
        )
    if not selected_positions:
        return np.zeros((0,), dtype=np.int64), tuple(metadata)
    positions_out = np.unique(np.concatenate(selected_positions).astype(np.int64, copy=False))
    if bool(include_tail):
        tail_positions = np.flatnonzero(active_np >= int(layout.f_size)).astype(np.int64, copy=False)
        if tail_positions.size:
            positions_out = np.unique(np.concatenate((positions_out, tail_positions)).astype(np.int64, copy=False))
    return positions_out, tuple(metadata)
