"""True-operator RHSMode-1 residual rescue preconditioners.

This file is intentionally large during consolidation because each rescue
candidate shares the same true-action column cache, residual diagnostics, and
admission gates. The safe future split is by candidate family once Iteration 3
has moved all solver/preconditioner ownership under ``solvers.preconditioners``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.explicit_sparse import SparseOperatorBundle
from sfincs_jax.operators.profile_sources import (
    constraint_scheme2_inject_source as _constraint_scheme2_inject_source,
    constraint_scheme2_source_from_f as _constraint_scheme2_source_from_f,
)
from sfincs_jax.operators.profile_system import _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1
from sfincs_jax.operators.profile_layout import RHS1ActiveFieldSplitOrdering, RHS1BlockLayout

if TYPE_CHECKING:
    from sfincs_jax.operators.profile_system import V3FullSystemOperator

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
    "_rhs1_active_reduced_residual_diagnostics",
    "_true_operator_window_positions_from_residual",
    "_try_build_true_operator_active_block_lsq_preconditioner",
    "_try_build_true_operator_active_residual_block_lsq_preconditioner",
    "_try_build_true_operator_active_submatrix_preconditioner",
    "_try_build_true_operator_coupled_coarse_lsq_preconditioner",
    "_try_build_true_operator_residual_window_lsq_preconditioner",
    "_try_build_residual_coarse_host_sparse_preconditioner",
    "_try_build_residual_window_host_sparse_preconditioner",
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


def _try_build_true_operator_residual_window_lsq_preconditioner(
    *,
    true_matvec: Callable[[np.ndarray], np.ndarray],
    true_matmat: Callable[[np.ndarray], np.ndarray] | None,
    factor_bundle: object,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    max_windows: int,
    x_radius: int,
    ell_radius: int,
    max_nbytes: int,
    regularization: float,
    max_window_size: int,
    column_batch: int,
    drop_tol: float,
    include_tail: bool,
    explicit_specs: tuple[tuple[int, int, int], ...] = (),
    damping: bool = False,
    beta_max: float = 10.0,
    emit: Callable[[int, str], None] | None = None,
) -> _TrueOperatorWindowLSQPreconditionerBundle | None:
    """Build a bounded LSQ correction from columns of the true active operator."""

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve  # noqa: PLC0415

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    positions, window_metadata = _true_operator_window_positions_from_residual(
        residual=residual_np,
        layout=layout,
        active_indices=active_indices,
        max_windows=int(max_windows),
        x_radius=int(x_radius),
        ell_radius=int(ell_radius),
        include_tail=bool(include_tail),
        explicit_specs=tuple(explicit_specs),
    )
    if positions.size == 0:
        return None
    if positions.size > int(max_window_size):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true residual window skipped "
                f"window_size={int(positions.size)} max_window_size={int(max_window_size)}",
            )
        return None

    batch = max(1, int(column_batch))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    nnz_total = 0
    max_nbytes_use = max(0, int(max_nbytes))
    drop = max(0.0, float(drop_tol))
    for start in range(0, int(positions.size), int(batch)):
        stop = min(int(positions.size), int(start) + int(batch))
        pos_batch = np.asarray(positions[start:stop], dtype=np.int64)
        matmat_batch = int(batch) if true_matmat is not None and int(batch) > 1 else int(pos_batch.size)
        transient_estimated = int(2 * n * max(1, int(matmat_batch)) * np.dtype(np.float64).itemsize)
        estimated_before_batch = int(nnz_total * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated_before_batch += int((positions.size + 1) * np.dtype(np.int32).itemsize)
        estimated_before_batch += int(2 * positions.size * positions.size * np.dtype(np.float64).itemsize)
        estimated_before_batch += int(n * np.dtype(np.float64).itemsize)
        estimated_before_batch += int(transient_estimated)
        if max_nbytes_use > 0 and estimated_before_batch > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true residual window budget exceeded "
                    f"estimated_bytes={int(estimated_before_batch)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(start)}/{int(positions.size)}",
                )
            return None
        basis = np.zeros((n, int(matmat_batch)), dtype=np.float64)
        basis[pos_batch, np.arange(int(pos_batch.size), dtype=np.int64)] = 1.0
        if true_matmat is not None and int(matmat_batch) > 1:
            y_batch = np.asarray(true_matmat(basis), dtype=np.float64)[:, : int(pos_batch.size)]
        else:
            y_batch = np.column_stack(
                [
                    np.asarray(true_matvec(basis[:, j]), dtype=np.float64).reshape((-1,))
                    for j in range(int(pos_batch.size))
                ]
            )
        if y_batch.shape != (n, int(pos_batch.size)):
            return None
        for local_col in range(int(pos_batch.size)):
            y_col = y_batch[:, local_col]
            keep = np.flatnonzero(np.isfinite(y_col) & (np.abs(y_col) > drop))
            if keep.size == 0:
                continue
            rows.append(keep.astype(np.int32, copy=False))
            cols.append(np.full((int(keep.size),), int(start + local_col), dtype=np.int32))
            data.append(np.asarray(y_col[keep], dtype=np.float64))
            nnz_total += int(keep.size)
        estimated = int(nnz_total * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated += int((positions.size + 1) * np.dtype(np.int32).itemsize)
        estimated += int(2 * positions.size * positions.size * np.dtype(np.float64).itemsize)
        estimated += int(n * np.dtype(np.float64).itemsize)
        estimated += int(transient_estimated)
        if max_nbytes_use > 0 and estimated > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true residual window budget exceeded "
                    f"estimated_bytes={int(estimated)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(stop)}/{int(positions.size)}",
                )
            return None
    if not data:
        return None
    a_window = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(positions.size)),
        dtype=np.float64,
    )
    a_window.sum_duplicates()
    a_window.eliminate_zeros()
    col_norms = np.sqrt(np.asarray(a_window.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    inv_col_scale = np.zeros_like(col_norms)
    good = np.isfinite(col_norms) & (col_norms > norm_floor)
    inv_col_scale[good] = 1.0 / col_norms[good]
    if not np.any(good):
        return None
    a_scaled = a_window @ sp.diags(inv_col_scale, format="csc")
    normal = np.asarray((a_scaled.T @ a_scaled).toarray(), dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    normal_regularization = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + normal_regularization * np.eye(int(positions.size), dtype=np.float64)
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

    a_window_nbytes = int(a_window.data.nbytes + a_window.indices.nbytes + a_window.indptr.nbytes)
    normal_nbytes = int(normal.nbytes)
    factor_nbytes = int(
        a_window_nbytes
        + 2 * normal_nbytes
        + inv_col_scale.nbytes
        + int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    )
    if max_nbytes_use > 0 and factor_nbytes > max_nbytes_use:
        return None
    condition_estimate = None
    if int(positions.size) <= 256:
        condition_estimate = float(np.linalg.cond(normal))
    metadata = {
        "window_size": int(positions.size),
        "windows": tuple(window_metadata),
        "x_radius": int(x_radius),
        "ell_radius": int(ell_radius),
        "include_tail": bool(include_tail),
        "column_batch": int(batch),
        "drop_tol": float(drop),
        "a_window_nnz": int(a_window.nnz),
        "a_window_nbytes_actual": int(a_window_nbytes),
        "normal_nbytes": int(normal_nbytes),
        "normal_regularization": float(normal_regularization),
        "normal_solver": str(solver_kind),
        "normal_condition_estimate": condition_estimate,
        "factor_nbytes_estimate": int(factor_nbytes),
        "nonzero_column_count": int(np.count_nonzero(good)),
        "zero_or_invalid_column_count": int(positions.size - np.count_nonzero(good)),
        "damping": bool(damping),
        "beta_max": float(beta_max),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        first = window_metadata[0] if window_metadata else {}
        emit(
            1,
            "solve_v3_full_system_linear_gmres: true residual window built "
            f"window_size={int(positions.size)} nnz={int(a_window.nnz)} "
            f"setup_s={metadata['setup_s']:.3f} bytes={int(factor_nbytes)} "
            f"first=s{first.get('species', 'na')}/x{first.get('x_center', 'na')}/ell{first.get('ell_center', 'na')}",
        )
    return _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=factor_bundle,
        true_matvec=lambda x: np.asarray(true_matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        window_positions=np.asarray(positions, dtype=np.int64),
        a_window=a_scaled,
        inv_column_scale=np.asarray(inv_col_scale, dtype=np.float64),
        solve_normal=solve_normal,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_true_residual_window_lsq",
        regularization=float(regularization),
        damping=bool(damping),
        beta_max=float(beta_max),
        factor_nbytes_estimate=int(factor_nbytes),
        factor_nnz_estimate=int(a_window.nnz),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _try_build_true_operator_active_block_lsq_preconditioner(
    *,
    true_matvec: Callable[[np.ndarray], np.ndarray],
    true_matmat: Callable[[np.ndarray], np.ndarray] | None,
    factor_bundle: object,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    x_count: int,
    ell_count: int,
    max_nbytes: int,
    regularization: float,
    max_block_size: int,
    column_batch: int,
    drop_tol: float,
    include_tail: bool,
    max_tail: int,
    species_count: int | None = None,
    theta_stride: int = 1,
    zeta_stride: int = 1,
    damping: bool = False,
    beta_max: float = 10.0,
    emit: Callable[[int, str], None] | None = None,
) -> _TrueOperatorWindowLSQPreconditionerBundle | None:
    """Build a deterministic true-operator LSQ correction over an active kinetic block.

    Unlike the residual-window builder, this selects the same symbolic
    low-speed/low-pitch block used by the native active coupled factor and then
    forms columns of the *true* active operator.  It directly targets the
    remaining mismatch between the fast ``whichMatrix=0`` preconditioner and
    the true RHSMode=1 operator while preserving bounded setup and memory.
    """

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve  # noqa: PLC0415

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (n,):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active block skipped "
                f"active_shape_mismatch active={tuple(active_np.shape)} residual={(n,)}",
            )
        return None
    try:
        ordering = RHS1ActiveFieldSplitOrdering.cached_from_layout(layout, active_np)
    except ValueError as exc:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active block skipped "
                f"invalid_ordering={type(exc).__name__}: {exc}",
            )
        return None

    max_block_size_use = max(1, int(max_block_size))
    kinetic_positions = ordering.dominant_kinetic_positions(
        x_count=max(0, int(x_count)),
        ell_count=max(0, int(ell_count)),
        species_count=species_count,
        theta_stride=max(1, int(theta_stride)),
        zeta_stride=max(1, int(zeta_stride)),
        max_positions=max_block_size_use,
    )
    positions = kinetic_positions.astype(np.int64, copy=False)
    tail_selected = 0
    if bool(include_tail) and positions.size < max_block_size_use and int(max_tail) > 0:
        tail_candidates = np.concatenate(
            (
                ordering.phi1_positions.astype(np.int64, copy=False),
                ordering.extra_positions.astype(np.int64, copy=False),
            )
        )
        if tail_candidates.size:
            take = min(int(max_tail), max_block_size_use - int(positions.size))
            tail_selected = int(min(int(tail_candidates.size), int(take)))
            positions = np.concatenate((positions, tail_candidates[:tail_selected])).astype(np.int64, copy=False)
    if positions.size:
        _, first = np.unique(positions, return_index=True)
        positions = positions[np.sort(first)].astype(np.int64, copy=False)
    if positions.size == 0:
        return None
    if positions.size > max_block_size_use:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active block skipped "
                f"block_size={int(positions.size)} max_block_size={int(max_block_size_use)}",
            )
        return None

    batch = max(1, int(column_batch))
    max_nbytes_use = max(0, int(max_nbytes))
    drop = max(0.0, float(drop_tol))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    nnz_total = 0
    for start in range(0, int(positions.size), int(batch)):
        stop = min(int(positions.size), int(start) + int(batch))
        pos_batch = np.asarray(positions[start:stop], dtype=np.int64)
        ncols = int(pos_batch.size)
        transient_estimated = int(2 * n * ncols * np.dtype(np.float64).itemsize)
        estimated_before = int(nnz_total * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated_before += int((positions.size + 1) * np.dtype(np.int32).itemsize)
        estimated_before += int(2 * positions.size * positions.size * np.dtype(np.float64).itemsize)
        estimated_before += int(transient_estimated)
        if max_nbytes_use > 0 and estimated_before > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active block budget exceeded "
                    f"estimated_bytes={int(estimated_before)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(start)}/{int(positions.size)}",
                )
            return None
        basis = np.zeros((n, ncols), dtype=np.float64)
        basis[pos_batch, np.arange(ncols, dtype=np.int64)] = 1.0
        if true_matmat is not None and ncols > 1:
            y_batch = np.asarray(true_matmat(basis), dtype=np.float64)
        else:
            y_batch = np.column_stack(
                [np.asarray(true_matvec(basis[:, j]), dtype=np.float64).reshape((-1,)) for j in range(ncols)]
            )
        if y_batch.shape != (n, ncols):
            return None
        for local_col in range(ncols):
            y_col = y_batch[:, local_col]
            keep = np.flatnonzero(np.isfinite(y_col) & (np.abs(y_col) > drop))
            if keep.size == 0:
                continue
            rows.append(keep.astype(np.int32, copy=False))
            cols.append(np.full((int(keep.size),), int(start + local_col), dtype=np.int32))
            data.append(np.asarray(y_col[keep], dtype=np.float64))
            nnz_total += int(keep.size)
    if not data:
        return None

    a_window = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(positions.size)),
        dtype=np.float64,
    )
    a_window.sum_duplicates()
    a_window.eliminate_zeros()
    col_norms = np.sqrt(np.asarray(a_window.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    good = np.isfinite(col_norms) & (col_norms > norm_floor)
    if not np.any(good):
        return None
    inv_col_scale = np.zeros_like(col_norms)
    inv_col_scale[good] = 1.0 / col_norms[good]
    a_scaled = a_window @ sp.diags(inv_col_scale, format="csc")
    normal = np.asarray((a_scaled.T @ a_scaled).toarray(), dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    normal_regularization = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + normal_regularization * np.eye(int(positions.size), dtype=np.float64)
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

    a_window_nbytes = int(a_window.data.nbytes + a_window.indices.nbytes + a_window.indptr.nbytes)
    normal_nbytes = int(normal.nbytes)
    factor_nbytes = int(
        a_window_nbytes
        + 2 * normal_nbytes
        + inv_col_scale.nbytes
        + int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    )
    if max_nbytes_use > 0 and factor_nbytes > max_nbytes_use:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active block skipped final_budget "
                f"bytes={int(factor_nbytes)} max_bytes={int(max_nbytes_use)} block_size={int(positions.size)}",
            )
        return None
    metadata = {
        "block_size": int(positions.size),
        "kinetic_selected": int(kinetic_positions.size),
        "tail_selected": int(tail_selected),
        "x_count": int(x_count),
        "ell_count": int(ell_count),
        "species_count": None if species_count is None else int(species_count),
        "theta_stride": int(max(1, int(theta_stride))),
        "zeta_stride": int(max(1, int(zeta_stride))),
        "include_tail": bool(include_tail),
        "max_tail": int(max_tail),
        "column_batch": int(batch),
        "drop_tol": float(drop),
        "a_window_nnz": int(a_window.nnz),
        "a_window_nbytes_actual": int(a_window_nbytes),
        "normal_nbytes": int(normal_nbytes),
        "normal_regularization": float(normal_regularization),
        "normal_solver": str(solver_kind),
        "factor_nbytes_estimate": int(factor_nbytes),
        "nonzero_column_count": int(np.count_nonzero(good)),
        "zero_or_invalid_column_count": int(positions.size - np.count_nonzero(good)),
        "damping": bool(damping),
        "beta_max": float(beta_max),
        "symbolic_ordering": ordering.to_dict(),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: true active block built "
            f"block_size={int(positions.size)} kinetic={int(kinetic_positions.size)} "
            f"tail={int(tail_selected)} nnz={int(a_window.nnz)} "
            f"setup_s={metadata['setup_s']:.3f} bytes={int(factor_nbytes)}",
        )
    return _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=factor_bundle,
        true_matvec=lambda x: np.asarray(true_matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        window_positions=np.asarray(positions, dtype=np.int64),
        a_window=a_scaled,
        inv_column_scale=np.asarray(inv_col_scale, dtype=np.float64),
        solve_normal=solve_normal,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_true_active_block_lsq",
        regularization=float(regularization),
        damping=bool(damping),
        beta_max=float(beta_max),
        factor_nbytes_estimate=int(factor_nbytes),
        factor_nnz_estimate=int(a_window.nnz),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _try_build_true_operator_active_residual_block_lsq_preconditioner(
    *,
    true_matvec: Callable[[np.ndarray], np.ndarray],
    true_matmat: Callable[[np.ndarray], np.ndarray] | None,
    factor_bundle: object,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    max_nbytes: int,
    regularization: float,
    max_block_size: int,
    column_batch: int,
    drop_tol: float,
    include_tail: bool,
    max_tail: int,
    kinetic_only: bool = True,
    damping: bool = False,
    beta_max: float = 10.0,
    emit: Callable[[int, str], None] | None = None,
) -> _TrueOperatorWindowLSQPreconditionerBundle | None:
    """Build a true-operator LSQ correction on dominant residual components.

    This is the residual-adaptive counterpart to the deterministic low-x/low-ell
    active block.  It selects the largest remaining residual entries in the
    active vector, forms the exact true-operator columns ``A[:, W]``, and solves
    a bounded normal equation for the correction on ``W``.  The caller still
    performs the measured true-residual acceptance test.
    """

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve  # noqa: PLC0415

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (n,):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active residual block skipped "
                f"active_shape_mismatch active={tuple(active_np.shape)} residual={(n,)}",
            )
        return None

    max_block_size_use = max(1, int(max_block_size))
    residual_abs = np.abs(residual_np)
    finite = np.isfinite(residual_abs)
    kinetic_mask = active_np < int(layout.f_size)
    base_mask = finite & kinetic_mask if bool(kinetic_only) else finite
    if bool(include_tail) and bool(kinetic_only):
        tail_mask = finite & ~kinetic_mask
    else:
        tail_mask = np.zeros_like(finite, dtype=bool)

    tail_selected = 0
    tail_positions = np.zeros((0,), dtype=np.int64)
    if bool(include_tail) and int(max_tail) > 0 and np.any(tail_mask):
        tail_candidates = np.flatnonzero(tail_mask).astype(np.int64, copy=False)
        tail_order = np.argsort(residual_abs[tail_candidates])[::-1]
        tail_take = min(int(max_tail), max_block_size_use, int(tail_candidates.size))
        tail_positions = tail_candidates[tail_order[:tail_take]].astype(np.int64, copy=False)
        tail_selected = int(tail_positions.size)

    kinetic_capacity = max(0, max_block_size_use - int(tail_positions.size))
    if not np.any(base_mask) or kinetic_capacity == 0:
        selected = tail_positions
    else:
        candidates = np.flatnonzero(base_mask).astype(np.int64, copy=False)
        order = np.argsort(residual_abs[candidates])[::-1]
        selected = candidates[order[:kinetic_capacity]].astype(np.int64, copy=False)
        if tail_positions.size:
            selected = np.concatenate((selected, tail_positions)).astype(np.int64, copy=False)
    if selected.size:
        _, first = np.unique(selected, return_index=True)
        positions = selected[np.sort(first)].astype(np.int64, copy=False)
    else:
        positions = np.zeros((0,), dtype=np.int64)
    if positions.size == 0:
        return None
    if positions.size > max_block_size_use:
        return None

    batch = max(1, int(column_batch))
    max_nbytes_use = max(0, int(max_nbytes))
    drop = max(0.0, float(drop_tol))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    nnz_total = 0
    for start in range(0, int(positions.size), int(batch)):
        stop = min(int(positions.size), int(start) + int(batch))
        pos_batch = np.asarray(positions[start:stop], dtype=np.int64)
        ncols = int(pos_batch.size)
        transient_estimated = int(2 * n * ncols * np.dtype(np.float64).itemsize)
        estimated_before = int(nnz_total * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated_before += int((positions.size + 1) * np.dtype(np.int32).itemsize)
        estimated_before += int(2 * positions.size * positions.size * np.dtype(np.float64).itemsize)
        estimated_before += int(transient_estimated)
        if max_nbytes_use > 0 and estimated_before > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active residual block budget exceeded "
                    f"estimated_bytes={int(estimated_before)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(start)}/{int(positions.size)}",
                )
            return None
        basis = np.zeros((n, ncols), dtype=np.float64)
        basis[pos_batch, np.arange(ncols, dtype=np.int64)] = 1.0
        if true_matmat is not None and ncols > 1:
            y_batch = np.asarray(true_matmat(basis), dtype=np.float64)
        else:
            y_batch = np.column_stack(
                [np.asarray(true_matvec(basis[:, j]), dtype=np.float64).reshape((-1,)) for j in range(ncols)]
            )
        if y_batch.shape != (n, ncols):
            return None
        for local_col in range(ncols):
            y_col = y_batch[:, local_col]
            keep = np.flatnonzero(np.isfinite(y_col) & (np.abs(y_col) > drop))
            if keep.size == 0:
                continue
            rows.append(keep.astype(np.int32, copy=False))
            cols.append(np.full((int(keep.size),), int(start + local_col), dtype=np.int32))
            data.append(np.asarray(y_col[keep], dtype=np.float64))
            nnz_total += int(keep.size)
    if not data:
        return None

    a_window = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(positions.size)),
        dtype=np.float64,
    )
    a_window.sum_duplicates()
    a_window.eliminate_zeros()
    col_norms = np.sqrt(np.asarray(a_window.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    good = np.isfinite(col_norms) & (col_norms > norm_floor)
    if not np.any(good):
        return None
    inv_col_scale = np.zeros_like(col_norms)
    inv_col_scale[good] = 1.0 / col_norms[good]
    a_scaled = a_window @ sp.diags(inv_col_scale, format="csc")
    normal = np.asarray((a_scaled.T @ a_scaled).toarray(), dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    normal_regularization = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + normal_regularization * np.eye(int(positions.size), dtype=np.float64)
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

    a_window_nbytes = int(a_window.data.nbytes + a_window.indices.nbytes + a_window.indptr.nbytes)
    normal_nbytes = int(normal.nbytes)
    factor_nbytes = int(
        a_window_nbytes
        + 2 * normal_nbytes
        + inv_col_scale.nbytes
        + int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    )
    if max_nbytes_use > 0 and factor_nbytes > max_nbytes_use:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active residual block skipped final_budget "
                f"bytes={int(factor_nbytes)} max_bytes={int(max_nbytes_use)} block_size={int(positions.size)}",
            )
        return None

    total_energy = float(np.sum(np.square(residual_np[np.isfinite(residual_np)])))
    selected_energy = float(np.sum(np.square(residual_np[positions])))
    metadata = {
        "block_size": int(positions.size),
        "kinetic_selected": int(np.count_nonzero(active_np[positions] < int(layout.f_size))),
        "tail_selected": int(tail_selected),
        "kinetic_only": bool(kinetic_only),
        "include_tail": bool(include_tail),
        "max_tail": int(max_tail),
        "column_batch": int(batch),
        "drop_tol": float(drop),
        "a_window_nnz": int(a_window.nnz),
        "a_window_nbytes_actual": int(a_window_nbytes),
        "normal_nbytes": int(normal_nbytes),
        "normal_regularization": float(normal_regularization),
        "normal_solver": str(solver_kind),
        "factor_nbytes_estimate": int(factor_nbytes),
        "nonzero_column_count": int(np.count_nonzero(good)),
        "zero_or_invalid_column_count": int(positions.size - np.count_nonzero(good)),
        "residual_energy_fraction": float(selected_energy / total_energy) if total_energy > 0.0 else 0.0,
        "max_residual_abs_selected": float(np.max(residual_abs[positions])) if positions.size else 0.0,
        "damping": bool(damping),
        "beta_max": float(beta_max),
        "selection": "top_residual_active_positions",
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: true active residual block built "
            f"block_size={int(positions.size)} kinetic={metadata['kinetic_selected']} "
            f"tail={int(tail_selected)} energy={metadata['residual_energy_fraction']:.3e} "
            f"nnz={int(a_window.nnz)} setup_s={metadata['setup_s']:.3f} bytes={int(factor_nbytes)}",
        )
    return _TrueOperatorWindowLSQPreconditionerBundle(
        base_factor=factor_bundle,
        true_matvec=lambda x: np.asarray(true_matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        window_positions=np.asarray(positions, dtype=np.int64),
        a_window=a_scaled,
        inv_column_scale=np.asarray(inv_col_scale, dtype=np.float64),
        solve_normal=solve_normal,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_true_active_residual_block_lsq",
        regularization=float(regularization),
        damping=bool(damping),
        beta_max=float(beta_max),
        factor_nbytes_estimate=int(factor_nbytes),
        factor_nnz_estimate=int(a_window.nnz),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _try_build_true_operator_active_submatrix_preconditioner(
    *,
    true_matvec: Callable[[np.ndarray], np.ndarray],
    true_matmat: Callable[[np.ndarray], np.ndarray] | None,
    factor_bundle: object,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    x_count: int,
    ell_count: int,
    max_nbytes: int,
    regularization: float,
    max_block_size: int,
    column_batch: int,
    drop_tol: float,
    include_tail: bool,
    max_tail: int,
    species_count: int | None = None,
    theta_stride: int = 1,
    zeta_stride: int = 1,
    damping: bool = True,
    alpha_clip: float = 10.0,
    emit: Callable[[int, str], None] | None = None,
) -> _TrueOperatorActiveSubmatrixPreconditionerBundle | None:
    """Build a true local active-block factor ``A[W,W]``.

    The LSQ active-block path uses full columns ``A[:, W]`` as a coarse
    correction.  This builder instead factors the local block rows and columns
    directly, which is closer to an additive-Schwarz block solve and retains
    the true finite-beta/field-split couplings inside the selected active
    kinetic block.
    """

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import splu, spilu  # noqa: PLC0415

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    active_np = (
        np.arange(int(layout.total_size), dtype=np.int64)
        if active_indices is None
        else np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    )
    if active_np.shape != (n,):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active submatrix skipped "
                f"active_shape_mismatch active={tuple(active_np.shape)} residual={(n,)}",
            )
        return None
    try:
        ordering = RHS1ActiveFieldSplitOrdering.cached_from_layout(layout, active_np)
    except ValueError as exc:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active submatrix skipped "
                f"invalid_ordering={type(exc).__name__}: {exc}",
            )
        return None

    max_block_size_use = max(1, int(max_block_size))
    kinetic_positions = ordering.dominant_kinetic_positions(
        x_count=max(0, int(x_count)),
        ell_count=max(0, int(ell_count)),
        species_count=species_count,
        theta_stride=max(1, int(theta_stride)),
        zeta_stride=max(1, int(zeta_stride)),
        max_positions=max_block_size_use,
    )
    positions = kinetic_positions.astype(np.int64, copy=False)
    tail_selected = 0
    if bool(include_tail) and positions.size < max_block_size_use and int(max_tail) > 0:
        tail_candidates = np.concatenate(
            (
                ordering.phi1_positions.astype(np.int64, copy=False),
                ordering.extra_positions.astype(np.int64, copy=False),
            )
        )
        if tail_candidates.size:
            take = min(int(max_tail), max_block_size_use - int(positions.size))
            tail_selected = int(min(int(tail_candidates.size), int(take)))
            positions = np.concatenate((positions, tail_candidates[:tail_selected])).astype(np.int64, copy=False)
    if positions.size:
        _, first = np.unique(positions, return_index=True)
        positions = positions[np.sort(first)].astype(np.int64, copy=False)
    if positions.size == 0 or positions.size > max_block_size_use:
        return None

    batch = max(1, int(column_batch))
    max_nbytes_use = max(0, int(max_nbytes))
    drop = max(0.0, float(drop_tol))
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    data: list[np.ndarray] = []
    nnz_total = 0
    for start in range(0, int(positions.size), int(batch)):
        stop = min(int(positions.size), int(start) + int(batch))
        pos_batch = np.asarray(positions[start:stop], dtype=np.int64)
        ncols = int(pos_batch.size)
        transient_estimated = int(2 * n * ncols * np.dtype(np.float64).itemsize)
        estimated_before = int(nnz_total * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated_before += int((positions.size + 1) * np.dtype(np.int32).itemsize)
        estimated_before += int(positions.size * positions.size * np.dtype(np.float64).itemsize)
        estimated_before += int(transient_estimated)
        if max_nbytes_use > 0 and estimated_before > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true active submatrix budget exceeded "
                    f"estimated_bytes={int(estimated_before)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(start)}/{int(positions.size)}",
                )
            return None
        basis = np.zeros((n, ncols), dtype=np.float64)
        basis[pos_batch, np.arange(ncols, dtype=np.int64)] = 1.0
        if true_matmat is not None and ncols > 1:
            y_batch = np.asarray(true_matmat(basis), dtype=np.float64)
        else:
            y_batch = np.column_stack(
                [np.asarray(true_matvec(basis[:, j]), dtype=np.float64).reshape((-1,)) for j in range(ncols)]
            )
        if y_batch.shape != (n, ncols):
            return None
        for local_col in range(ncols):
            y_col = y_batch[:, local_col]
            keep = np.flatnonzero(np.isfinite(y_col) & (np.abs(y_col) > drop))
            if keep.size == 0:
                continue
            rows.append(keep.astype(np.int32, copy=False))
            cols.append(np.full((int(keep.size),), int(start + local_col), dtype=np.int32))
            data.append(np.asarray(y_col[keep], dtype=np.float64))
            nnz_total += int(keep.size)
    if not data:
        return None

    a_window = sp.csc_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n, int(positions.size)),
        dtype=np.float64,
    )
    a_window.sum_duplicates()
    a_window.eliminate_zeros()
    a_block = a_window[positions, :].tocsc()
    a_block.sum_duplicates()
    a_block.eliminate_zeros()
    if a_block.shape != (int(positions.size), int(positions.size)) or a_block.nnz == 0:
        return None

    row_norms = np.sqrt(np.asarray(a_block.power(2).sum(axis=1), dtype=np.float64).reshape((-1,)))
    col_norms = np.sqrt(np.asarray(a_block.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    inv_row_scale = np.ones_like(row_norms)
    inv_col_scale = np.ones_like(col_norms)
    good_rows = np.isfinite(row_norms) & (row_norms > norm_floor)
    good_cols = np.isfinite(col_norms) & (col_norms > norm_floor)
    inv_row_scale[good_rows] = 1.0 / row_norms[good_rows]
    inv_col_scale[good_cols] = 1.0 / col_norms[good_cols]
    a_scaled = sp.diags(inv_row_scale, format="csc") @ a_block @ sp.diags(inv_col_scale, format="csc")
    scale = max(float(np.max(np.abs(a_scaled.data))) if a_scaled.nnz else 0.0, 1.0)
    reg = max(float(abs(regularization)), 1.0e-14) * scale
    if reg > 0.0:
        a_scaled = (a_scaled + reg * sp.eye(int(positions.size), format="csc", dtype=np.float64)).tocsc()

    solver_kind = "splu"
    try:
        lu = splu(a_scaled, permc_spec="COLAMD", diag_pivot_thresh=0.0)
    except Exception:  # noqa: BLE001
        solver_kind = "spilu"
        lu = spilu(a_scaled, drop_tol=max(float(regularization), 1.0e-12), fill_factor=20.0)

    def solve_block(rhs_block: np.ndarray) -> np.ndarray:
        rhs_np = np.asarray(rhs_block, dtype=np.float64).reshape((-1,))
        y_scaled = np.asarray(lu.solve(inv_row_scale * rhs_np), dtype=np.float64).reshape((-1,))
        return np.asarray(inv_col_scale * y_scaled, dtype=np.float64).reshape((-1,))

    a_window_nbytes = int(a_window.data.nbytes + a_window.indices.nbytes + a_window.indptr.nbytes)
    a_block_nbytes = int(a_block.data.nbytes + a_block.indices.nbytes + a_block.indptr.nbytes)
    lu_nnz = int(getattr(lu, "L").nnz + getattr(lu, "U").nnz)
    factor_nbytes = int(
        a_window_nbytes
        + a_block_nbytes
        + lu_nnz * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
        + inv_row_scale.nbytes
        + inv_col_scale.nbytes
        + int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    )
    if max_nbytes_use > 0 and factor_nbytes > max_nbytes_use:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true active submatrix skipped final_budget "
                f"bytes={int(factor_nbytes)} max_bytes={int(max_nbytes_use)} block_size={int(positions.size)}",
            )
        return None

    metadata = {
        "block_size": int(positions.size),
        "kinetic_selected": int(kinetic_positions.size),
        "tail_selected": int(tail_selected),
        "x_count": int(x_count),
        "ell_count": int(ell_count),
        "species_count": None if species_count is None else int(species_count),
        "theta_stride": int(max(1, int(theta_stride))),
        "zeta_stride": int(max(1, int(zeta_stride))),
        "include_tail": bool(include_tail),
        "max_tail": int(max_tail),
        "column_batch": int(batch),
        "drop_tol": float(drop),
        "a_window_nnz": int(a_window.nnz),
        "a_block_nnz": int(a_block.nnz),
        "lu_nnz": int(lu_nnz),
        "a_window_nbytes_actual": int(a_window_nbytes),
        "a_block_nbytes_actual": int(a_block_nbytes),
        "local_regularization": float(reg),
        "local_solver": str(solver_kind),
        "factor_nbytes_estimate": int(factor_nbytes),
        "damping": bool(damping),
        "alpha_clip": float(alpha_clip),
        "symbolic_ordering": ordering.to_dict(),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: true active submatrix built "
            f"block_size={int(positions.size)} kinetic={int(kinetic_positions.size)} "
            f"tail={int(tail_selected)} block_nnz={int(a_block.nnz)} lu_nnz={int(lu_nnz)} "
            f"setup_s={metadata['setup_s']:.3f} bytes={int(factor_nbytes)}",
        )
    return _TrueOperatorActiveSubmatrixPreconditionerBundle(
        base_factor=factor_bundle,
        true_matvec=lambda x: np.asarray(true_matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        block_positions=np.asarray(positions, dtype=np.int64),
        a_window=a_window,
        solve_block=solve_block,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_true_active_submatrix",
        damping=bool(damping),
        alpha_clip=float(alpha_clip),
        factor_nbytes_estimate=int(factor_nbytes),
        factor_nnz_estimate=int(a_block.nnz),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _try_build_true_operator_coupled_coarse_lsq_preconditioner(
    *,
    true_matvec: Callable[[np.ndarray], np.ndarray],
    true_matmat: Callable[[np.ndarray], np.ndarray] | None,
    factor_bundle: object,
    residual: np.ndarray,
    op: V3FullSystemOperator,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    max_windows: int,
    x_radius: int,
    ell_radius: int,
    max_nbytes: int,
    regularization: float,
    max_coarse_size: int,
    column_batch: int,
    drop_tol: float,
    low_lmax: int,
    profile_moment_count: int,
    angular_lmax: int,
    angular_mode_max: int,
    max_tail_units: int,
    include_tail: bool,
    include_constraint_sources: bool,
    include_fsavg: bool,
    include_window_residual: bool,
    include_profile_moments: bool,
    include_angular_residual: bool,
    include_angular_basis: bool,
    include_preconditioned_loads: bool,
    preconditioned_load_max_columns: int,
    preconditioned_load_max_nnz: int,
    preconditioned_load_drop_tol: float,
    damping: bool = False,
    beta_max: float = 10.0,
    emit: Callable[[int, str], None] | None = None,
) -> _TrueOperatorCoupledCoarseLSQPreconditionerBundle | None:
    """Build a coupled true-operator coarse residual correction.

    The basis is deliberately small and physics-structured: it combines the
    global tail/source unknowns, source-moment directions used by
    constraintScheme=1/2, velocity/profile moments, low-L flux-surface-averaged
    residual moments, low Fourier angular residual projections, and the dominant
    kinetic residual window.  Optionally it also applies the existing active
    preconditioner to these physics/load directions, yielding solution-space
    response columns closer to a field-split Schur correction.  The coarse
    equation uses columns of the *true* active operator, not the reduced
    preconditioner matrix.
    """

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.linalg import cho_factor, cho_solve, lu_factor, lu_solve  # noqa: PLC0415

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    if active_indices is None:
        active_np = np.arange(int(layout.total_size), dtype=np.int64)
    else:
        active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active_np.shape != (n,):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true coupled coarse skipped "
                f"active_shape_mismatch active={tuple(active_np.shape)} residual={(n,)}",
            )
        return None
    max_cols = max(1, int(max_coarse_size))
    max_nbytes_use = max(0, int(max_nbytes))
    drop = max(0.0, float(drop_tol))

    col_rows: list[np.ndarray] = []
    col_data: list[np.ndarray] = []
    names: list[str] = []

    def _add_active_column(name: str, positions: Any, values: Any) -> None:
        if len(names) >= max_cols:
            return
        pos = np.asarray(positions, dtype=np.int64).reshape((-1,))
        val = np.asarray(values, dtype=np.float64).reshape((-1,))
        if pos.shape != val.shape or pos.size == 0:
            return
        valid = (pos >= 0) & (pos < n) & np.isfinite(val) & (np.abs(val) > 0.0)
        if not np.any(valid):
            return
        pos = pos[valid]
        val = val[valid]
        order = np.argsort(pos)
        pos = pos[order]
        val = val[order]
        unique, inverse = np.unique(pos, return_inverse=True)
        if unique.size != pos.size:
            summed = np.zeros((int(unique.size),), dtype=np.float64)
            np.add.at(summed, inverse, val)
            pos = unique
            val = summed
        norm = float(np.linalg.norm(val))
        if not (np.isfinite(norm) and norm > 0.0):
            return
        col_rows.append(pos.astype(np.int32, copy=False))
        col_data.append(val.astype(np.float64, copy=False))
        names.append(str(name))

    def _add_full_column(name: str, full: Any) -> None:
        if len(names) >= max_cols:
            return
        vec = np.asarray(full, dtype=np.float64).reshape((-1,))
        if vec.shape != (int(layout.total_size),):
            return
        active_values = vec[active_np]
        keep = np.flatnonzero(np.isfinite(active_values) & (np.abs(active_values) > 0.0))
        if keep.size == 0:
            return
        _add_active_column(name, keep, active_values[keep])

    def _apply_base_preconditioner_response(load: np.ndarray) -> np.ndarray | None:
        load_np = np.asarray(load, dtype=np.float64).reshape((-1,))
        if load_np.shape != (n,):
            return None
        try:
            solve = getattr(factor_bundle, "solve", None)
            if solve is not None:
                response = np.asarray(solve(load_np), dtype=np.float64).reshape((-1,))
                if response.shape == (n,):
                    return response
        except Exception:
            pass
        try:
            operator = getattr(factor_bundle, "operator", None)
            matvec = getattr(operator, "matvec", None)
            if matvec is not None:
                response = np.asarray(matvec(load_np), dtype=np.float64).reshape((-1,))
                if response.shape == (n,):
                    return response
            if callable(operator):
                response = np.asarray(operator(load_np), dtype=np.float64).reshape((-1,))
                if response.shape == (n,):
                    return response
        except Exception:
            pass
        return None

    kinetic_mask = active_np < int(layout.f_size)
    kinetic_positions = np.flatnonzero(kinetic_mask).astype(np.int64, copy=False)
    decoded = None
    if kinetic_positions.size:
        decoded = layout.decode_kinetic_indices(active_np[kinetic_mask])

    def _add_decoded_kinetic_column(name: str, mask: np.ndarray, values: np.ndarray) -> None:
        if decoded is None or kinetic_positions.size == 0:
            return
        mask_np = np.asarray(mask, dtype=bool).reshape((-1,))
        if mask_np.shape != kinetic_positions.shape:
            return
        local = kinetic_positions[mask_np]
        if local.size == 0:
            return
        values_np = np.asarray(values, dtype=np.float64).reshape((-1,))
        if values_np.shape != kinetic_positions.shape:
            return
        _add_active_column(name, local, values_np[mask_np])

    positions, window_metadata = _true_operator_window_positions_from_residual(
        residual=residual_np,
        layout=layout,
        active_indices=active_np,
        max_windows=int(max_windows),
        x_radius=int(x_radius),
        ell_radius=int(ell_radius),
        include_tail=False,
        explicit_specs=(),
    )
    if bool(include_window_residual) and positions.size:
        _add_active_column("dominant_kinetic_residual_window", positions, residual_np[positions])
        if decoded is not None:
            for i_window, meta in enumerate(window_metadata):
                if len(names) >= max_cols:
                    break
                try:
                    species = int(meta.get("species", -1))
                    x_lo, x_hi = tuple(int(v) for v in meta.get("x_range", (-1, -1)))
                    ell_lo, ell_hi = tuple(int(v) for v in meta.get("ell_range", (-1, -1)))
                    x_center = int(meta.get("x_center", x_lo))
                    ell_center = int(meta.get("ell_center", ell_lo))
                except Exception:
                    continue
                mask = (
                    (decoded.species == int(species))
                    & (decoded.x >= int(x_lo))
                    & (decoded.x <= int(x_hi))
                    & (decoded.ell >= int(ell_lo))
                    & (decoded.ell <= int(ell_hi))
                )
                values = residual_np[kinetic_positions]
                _add_decoded_kinetic_column(
                    f"dominant_kinetic_residual_window_{i_window}_s{species}_x{x_center}_l{ell_center}",
                    mask,
                    values,
                )

    tail_positions = np.flatnonzero(active_np >= int(layout.f_size)).astype(np.int64, copy=False)
    if bool(include_tail) and tail_positions.size:
        _add_active_column("tail_residual", tail_positions, residual_np[tail_positions])
        if int(tail_positions.size) <= int(max_tail_units):
            for local_pos in tail_positions:
                _add_active_column(f"tail_unit_{int(active_np[int(local_pos)] - int(layout.f_size))}", [local_pos], [1.0])

    if bool(include_constraint_sources):
        if int(op.constraint_scheme) == 1:
            ix0 = _ix_min(bool(op.point_at_x0))
            source_basis = _source_basis_constraint_scheme_1(op.x)
            for species in range(int(op.n_species)):
                for basis_index, basis in enumerate(source_basis):
                    if len(names) >= max_cols:
                        break
                    f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                    f_dir = f_dir.at[species, ix0:, 0, :, :].set(basis[ix0:, None, None])
                    full = jnp.concatenate(
                        [f_dir.reshape((-1,)), jnp.zeros((int(layout.total_size) - int(layout.f_size),), dtype=jnp.float64)]
                    )
                    _add_full_column(f"constraint1_source_s{species}_{basis_index}", full)
        elif int(op.constraint_scheme) == 2:
            try:
                f_res = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                kinetic_mask = active_np < int(layout.f_size)
                if np.any(kinetic_mask):
                    f_res = f_res.reshape((-1,)).at[jnp.asarray(active_np[kinetic_mask], dtype=jnp.int32)].set(
                        jnp.asarray(residual_np[np.flatnonzero(kinetic_mask)], dtype=jnp.float64)
                    )
                    f_res = f_res.reshape(op.fblock.f_shape)
                    src = _constraint_scheme2_source_from_f(op, f_res)
                    full = jnp.concatenate(
                        [
                            _constraint_scheme2_inject_source(op, src),
                            jnp.zeros((int(layout.total_size) - int(layout.f_size),), dtype=jnp.float64),
                        ]
                    )
                    _add_full_column("constraint2_source_residual", full)
            except Exception:
                pass

    factor_np = None
    f_res_np = None
    if kinetic_positions.size:
        f_res_np = np.zeros(tuple(int(v) for v in op.fblock.f_shape), dtype=np.float64)
        f_res_np.reshape((-1,))[active_np[kinetic_mask]] = residual_np[np.flatnonzero(kinetic_mask)]
        factor_np = np.asarray(
            jax.device_get(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)),
            dtype=np.float64,
        )

    if bool(include_profile_moments) and kinetic_positions.size and decoded is not None:
        x_np = np.asarray(jax.device_get(op.x), dtype=np.float64).reshape((-1,))
        xw_np = np.asarray(jax.device_get(op.x_weights), dtype=np.float64).reshape((-1,))
        if x_np.size == int(layout.n_x) and xw_np.size == int(layout.n_x):
            moment_specs: list[tuple[str, int, np.ndarray]] = [
                ("density_moment", 0, (x_np**2) * xw_np),
                ("pressure_moment", 0, (x_np**4) * xw_np),
                ("flow_current_moment", min(1, int(layout.n_xi) - 1), (x_np**3) * xw_np),
                ("heat_flow_moment", min(1, int(layout.n_xi) - 1), (x_np**5) * xw_np),
            ]
            if factor_np is not None:
                angular_values = np.asarray(factor_np[decoded.theta, decoded.zeta], dtype=np.float64)
                angular_norm = float(np.linalg.norm(angular_values))
                if np.isfinite(angular_norm) and angular_norm > 0.0:
                    angular_values = angular_values / float(angular_norm)
                else:
                    angular_values = np.full(
                        kinetic_positions.shape,
                        float(max(1, int(layout.n_theta) * int(layout.n_zeta))) ** -0.5,
                        dtype=np.float64,
                    )
            else:
                angular_values = np.full(
                    kinetic_positions.shape,
                    float(max(1, int(layout.n_theta) * int(layout.n_zeta))) ** -0.5,
                    dtype=np.float64,
                )
            for moment_name, ell, weights in moment_specs[: max(0, int(profile_moment_count))]:
                if len(names) >= max_cols:
                    break
                if int(ell) < 0 or int(ell) >= int(layout.n_xi):
                    continue
                for species in range(int(layout.n_species)):
                    if len(names) >= max_cols:
                        break
                    mask = (decoded.species == int(species)) & (decoded.ell == int(ell))
                    values = np.asarray(weights[decoded.x], dtype=np.float64) * angular_values
                    _add_decoded_kinetic_column(f"profile_{moment_name}_s{species}_l{int(ell)}", mask, values)

    if bool(include_fsavg):
        if f_res_np is not None and factor_np is not None:
            f_res_np = np.zeros(tuple(int(v) for v in op.fblock.f_shape), dtype=np.float64)
            f_res_np.reshape((-1,))[active_np[kinetic_mask]] = residual_np[np.flatnonzero(kinetic_mask)]
            lmax_use = max(0, min(int(low_lmax), int(layout.n_xi) - 1))
            for ell in range(lmax_use + 1):
                for species in range(int(layout.n_species)):
                    if len(names) >= max_cols:
                        break
                    avg = np.einsum("tz,xtz->x", factor_np, f_res_np[species, :, ell, :, :])
                    f_dir_np = np.zeros_like(f_res_np)
                    f_dir_np[species, :, ell, :, :] = avg[:, None, None]
                    full_np = np.concatenate(
                        [
                            f_dir_np.reshape((-1,)),
                            np.zeros((int(layout.total_size) - int(layout.f_size),), dtype=np.float64),
                        ]
                    )
                    _add_full_column(f"fsavg_residual_s{species}_l{ell}", full_np)

    if (
        (bool(include_angular_residual) or bool(include_angular_basis))
        and f_res_np is not None
        and factor_np is not None
        and decoded is not None
        and int(layout.n_theta) > 1
        and int(layout.n_zeta) > 1
    ):
        theta = np.arange(int(layout.n_theta), dtype=np.float64)
        zeta = np.arange(int(layout.n_zeta), dtype=np.float64)
        two_pi = float(2.0 * np.pi)
        all_mode_pairs = (
            (1, 0),
            (0, 1),
            (1, 1),
            (1, -1),
            (2, 0),
            (0, 2),
            (2, 1),
            (1, 2),
            (2, -1),
            (1, -2),
            (2, 2),
            (2, -2),
            (3, 0),
            (0, 3),
            (3, 1),
            (1, 3),
            (3, -1),
            (1, -3),
        )
        max_mode = max(0, int(angular_mode_max))
        mode_pairs = tuple(
            pair for pair in all_mode_pairs if max(abs(int(pair[0])), abs(int(pair[1]))) <= int(max_mode)
        )
        lmax_use = max(0, min(int(angular_lmax), int(layout.n_xi) - 1))
        for ell in range(lmax_use + 1):
            for m_mode, n_mode in mode_pairs:
                if len(names) >= max_cols:
                    break
                phase = two_pi * (
                    float(m_mode) * theta[:, None] / float(max(1, int(layout.n_theta)))
                    + float(n_mode) * zeta[None, :] / float(max(1, int(layout.n_zeta)))
                )
                for parity, pattern in (("cos", np.cos(phase)), ("sin", np.sin(phase))):
                    if len(names) >= max_cols:
                        break
                    weighted_pattern = factor_np * pattern
                    denom = float(np.sum(weighted_pattern * pattern))
                    if not (np.isfinite(denom) and abs(denom) > 0.0):
                        denom = float(np.sum(pattern * pattern))
                    if not (np.isfinite(denom) and abs(denom) > 0.0):
                        continue
                    pattern_values = pattern[decoded.theta, decoded.zeta]
                    if bool(include_angular_residual):
                        for species in range(int(layout.n_species)):
                            if len(names) >= max_cols:
                                break
                            coeff = np.einsum(
                                "tz,xtz->x",
                                weighted_pattern,
                                f_res_np[species, :, ell, :, :],
                            ) / float(denom)
                            mask = (decoded.species == int(species)) & (decoded.ell == int(ell))
                            values = np.asarray(coeff[decoded.x], dtype=np.float64) * pattern_values
                            _add_decoded_kinetic_column(
                                f"angular_residual_s{species}_l{ell}_m{m_mode}_n{n_mode}_{parity}",
                                mask,
                                values,
                            )
                    if bool(include_angular_basis):
                        pattern_norm = float(np.linalg.norm(pattern))
                        if not (np.isfinite(pattern_norm) and pattern_norm > 0.0):
                            continue
                        unit_pattern_values = pattern_values / float(pattern_norm)
                        for species in range(int(layout.n_species)):
                            if len(names) >= max_cols:
                                break
                            mask = (decoded.species == int(species)) & (decoded.ell == int(ell))
                            _add_decoded_kinetic_column(
                                f"angular_basis_s{species}_l{ell}_m{m_mode}_n{n_mode}_{parity}",
                                mask,
                                unit_pattern_values,
                            )

    if not names:
        if emit is not None:
            emit(1, "solve_v3_full_system_linear_gmres: true coupled coarse skipped empty_basis")
        return None
    rows = np.concatenate(col_rows)
    cols = np.concatenate(
        [np.full((int(row.size),), int(i), dtype=np.int32) for i, row in enumerate(col_rows)]
    )
    data = np.concatenate(col_data)
    z_basis = sp.csc_matrix((data, (rows, cols)), shape=(n, int(len(names))), dtype=np.float64)
    z_basis.sum_duplicates()
    z_basis.eliminate_zeros()
    preconditioned_load_column_count = 0
    preconditioned_load_nnz = 0
    if bool(include_preconditioned_loads) and int(z_basis.shape[1]) > 0 and len(names) < max_cols:
        max_pre_cols = max(0, min(int(preconditioned_load_max_columns), int(z_basis.shape[1])))
        max_pre_nnz = max(0, int(preconditioned_load_max_nnz))
        pre_drop = max(0.0, float(preconditioned_load_drop_tol))
        extra_rows: list[np.ndarray] = []
        extra_cols: list[np.ndarray] = []
        extra_data: list[np.ndarray] = []
        extra_names: list[str] = []
        for source_col in range(max_pre_cols):
            if len(names) + len(extra_names) >= max_cols:
                break
            try:
                load = np.asarray(z_basis[:, source_col].toarray(), dtype=np.float64).reshape((-1,))
                response = _apply_base_preconditioner_response(load)
            except Exception:
                continue
            if response is None:
                continue
            if response.shape != (n,):
                continue
            keep = np.flatnonzero(np.isfinite(response) & (np.abs(response) > float(pre_drop)))
            if keep.size == 0:
                continue
            if max_pre_nnz > 0 and keep.size > max_pre_nnz:
                local_order = np.argpartition(np.abs(response[keep]), -int(max_pre_nnz))[-int(max_pre_nnz) :]
                keep = np.sort(keep[local_order])
            values = response[keep]
            norm = float(np.linalg.norm(values))
            if not (np.isfinite(norm) and norm > 0.0):
                continue
            col_index = int(len(extra_names))
            extra_rows.append(keep.astype(np.int32, copy=False))
            extra_cols.append(np.full((int(keep.size),), col_index, dtype=np.int32))
            extra_data.append(np.asarray(values, dtype=np.float64))
            extra_names.append(f"preconditioned_{names[source_col]}")
            preconditioned_load_nnz += int(keep.size)
        if extra_data:
            extra_basis = sp.csc_matrix(
                (np.concatenate(extra_data), (np.concatenate(extra_rows), np.concatenate(extra_cols))),
                shape=(n, int(len(extra_names))),
                dtype=np.float64,
            )
            extra_basis.sum_duplicates()
            extra_basis.eliminate_zeros()
            z_basis = sp.hstack([z_basis, extra_basis], format="csc")
            names.extend(extra_names)
            preconditioned_load_column_count = int(extra_basis.shape[1])
    z_col_norms = np.sqrt(np.asarray(z_basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    valid_z = np.flatnonzero(np.isfinite(z_col_norms) & (z_col_norms > 0.0))
    if valid_z.size == 0:
        return None
    if valid_z.size != z_basis.shape[1]:
        z_basis = z_basis[:, valid_z].tocsc()
        names = [names[int(i)] for i in valid_z]

    batch = max(1, int(column_batch))
    az_rows: list[np.ndarray] = []
    az_cols: list[np.ndarray] = []
    az_data: list[np.ndarray] = []
    nnz_total = 0
    for start in range(0, int(z_basis.shape[1]), int(batch)):
        stop = min(int(z_basis.shape[1]), int(start) + int(batch))
        z_batch = np.asarray(z_basis[:, start:stop].toarray(), dtype=np.float64)
        transient_estimated = int(2 * n * int(z_batch.shape[1]) * np.dtype(np.float64).itemsize)
        estimated_before = int((z_basis.nnz + nnz_total) * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize))
        estimated_before += int((z_basis.shape[1] + 1) * np.dtype(np.int32).itemsize)
        estimated_before += int(2 * z_basis.shape[1] * z_basis.shape[1] * np.dtype(np.float64).itemsize)
        estimated_before += int(transient_estimated)
        if max_nbytes_use > 0 and estimated_before > max_nbytes_use:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true coupled coarse skipped budget_pre_matvec "
                    f"estimated_bytes={int(estimated_before)} max_bytes={int(max_nbytes_use)} "
                    f"columns_done={int(start)}/{int(z_basis.shape[1])}",
                )
            return None
        if true_matmat is not None and z_batch.shape[1] > 1:
            y_batch = np.asarray(true_matmat(z_batch), dtype=np.float64)
        else:
            y_batch = np.column_stack(
                [
                    np.asarray(true_matvec(z_batch[:, j]), dtype=np.float64).reshape((-1,))
                    for j in range(int(z_batch.shape[1]))
                ]
            )
        if y_batch.shape != z_batch.shape:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: true coupled coarse skipped matmat_shape_mismatch "
                    f"got={tuple(int(v) for v in y_batch.shape)} expected={tuple(int(v) for v in z_batch.shape)}",
                )
            return None
        for local_col in range(int(y_batch.shape[1])):
            y_col = y_batch[:, local_col]
            keep = np.flatnonzero(np.isfinite(y_col) & (np.abs(y_col) > drop))
            if keep.size == 0:
                continue
            az_rows.append(keep.astype(np.int32, copy=False))
            az_cols.append(np.full((int(keep.size),), int(start + local_col), dtype=np.int32))
            az_data.append(np.asarray(y_col[keep], dtype=np.float64))
            nnz_total += int(keep.size)
    if not az_data:
        if emit is not None:
            emit(1, "solve_v3_full_system_linear_gmres: true coupled coarse skipped empty_operator_columns")
        return None
    a_basis = sp.csc_matrix(
        (np.concatenate(az_data), (np.concatenate(az_rows), np.concatenate(az_cols))),
        shape=(n, int(z_basis.shape[1])),
        dtype=np.float64,
    )
    a_basis.sum_duplicates()
    a_basis.eliminate_zeros()
    col_norms = np.sqrt(np.asarray(a_basis.power(2).sum(axis=0), dtype=np.float64).reshape((-1,)))
    norm_floor = max(float(abs(regularization)), np.finfo(np.float64).eps)
    good = np.isfinite(col_norms) & (col_norms > norm_floor)
    if not np.any(good):
        if emit is not None:
            emit(1, "solve_v3_full_system_linear_gmres: true coupled coarse skipped zero_operator_columns")
        return None
    if np.count_nonzero(good) != a_basis.shape[1]:
        keep = np.flatnonzero(good)
        a_basis = a_basis[:, keep].tocsc()
        z_basis = z_basis[:, keep].tocsc()
        col_norms = col_norms[keep]
        names = [names[int(i)] for i in keep]
    inv_col_scale = np.zeros_like(col_norms)
    inv_col_scale[:] = 1.0 / col_norms
    a_scaled = a_basis @ sp.diags(inv_col_scale, format="csc")
    normal = np.asarray((a_scaled.T @ a_scaled).toarray(), dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    normal_regularization = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + normal_regularization * np.eye(int(a_scaled.shape[1]), dtype=np.float64)
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

    z_nbytes = int(z_basis.data.nbytes + z_basis.indices.nbytes + z_basis.indptr.nbytes)
    a_nbytes = int(a_basis.data.nbytes + a_basis.indices.nbytes + a_basis.indptr.nbytes)
    normal_nbytes = int(normal.nbytes)
    factor_nbytes = int(
        z_nbytes
        + a_nbytes
        + 2 * normal_nbytes
        + inv_col_scale.nbytes
        + int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    )
    if max_nbytes_use > 0 and factor_nbytes > max_nbytes_use:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: true coupled coarse skipped final_budget "
                f"bytes={int(factor_nbytes)} max_bytes={int(max_nbytes_use)} coarse_size={int(z_basis.shape[1])}",
            )
        return None
    condition_estimate = None
    if int(normal.shape[0]) <= 256:
        condition_estimate = float(np.linalg.cond(normal))
    metadata = {
        "coarse_size": int(z_basis.shape[1]),
        "basis_names": tuple(str(name) for name in names),
        "dominant_windows": tuple(window_metadata),
        "window_count": int(len(window_metadata)),
        "x_radius": int(x_radius),
        "ell_radius": int(ell_radius),
        "low_lmax": int(low_lmax),
        "profile_moment_count": int(profile_moment_count),
        "angular_lmax": int(angular_lmax),
        "angular_mode_max": int(angular_mode_max),
        "tail_included": bool(include_tail),
        "constraint_sources_included": bool(include_constraint_sources),
        "fsavg_included": bool(include_fsavg),
        "window_residual_included": bool(include_window_residual),
        "profile_moments_included": bool(include_profile_moments),
        "angular_residual_included": bool(include_angular_residual),
        "angular_basis_included": bool(include_angular_basis),
        "preconditioned_loads_included": bool(include_preconditioned_loads),
        "preconditioned_load_column_count": int(preconditioned_load_column_count),
        "preconditioned_load_nnz": int(preconditioned_load_nnz),
        "preconditioned_load_max_columns": int(preconditioned_load_max_columns),
        "preconditioned_load_max_nnz": int(preconditioned_load_max_nnz),
        "preconditioned_load_drop_tol": float(preconditioned_load_drop_tol),
        "column_batch": int(batch),
        "drop_tol": float(drop),
        "z_basis_nnz": int(z_basis.nnz),
        "a_basis_nnz": int(a_basis.nnz),
        "z_basis_nbytes_actual": int(z_nbytes),
        "a_basis_nbytes_actual": int(a_nbytes),
        "normal_nbytes": int(normal_nbytes),
        "normal_regularization": float(normal_regularization),
        "normal_solver": str(solver_kind),
        "normal_condition_estimate": condition_estimate,
        "factor_nbytes_estimate": int(factor_nbytes),
        "damping": bool(damping),
        "beta_max": float(beta_max),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: true coupled coarse built "
            f"coarse_size={int(z_basis.shape[1])} z_nnz={int(z_basis.nnz)} "
            f"a_nnz={int(a_basis.nnz)} setup_s={metadata['setup_s']:.3f} "
            f"bytes={int(factor_nbytes)}",
        )
    return _TrueOperatorCoupledCoarseLSQPreconditionerBundle(
        base_factor=factor_bundle,
        true_matvec=lambda x: np.asarray(true_matvec(np.asarray(x, dtype=np.float64)), dtype=np.float64).reshape((-1,)),
        z_basis=z_basis,
        a_basis=a_scaled,
        inv_column_scale=np.asarray(inv_col_scale, dtype=np.float64),
        solve_normal=solve_normal,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_true_coupled_coarse_lsq",
        regularization=float(regularization),
        damping=bool(damping),
        beta_max=float(beta_max),
        factor_nbytes_estimate=int(factor_nbytes),
        factor_nnz_estimate=int(a_basis.nnz + z_basis.nnz),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )



def _try_build_residual_window_host_sparse_preconditioner(
    *,
    operator_bundle: SparseOperatorBundle,
    factor_bundle: object,
    residual: np.ndarray,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    max_windows: int,
    x_radius: int,
    ell_radius: int,
    max_nbytes: int,
    regularization: float,
    coefficient_mode: str,
    combine_mode: str,
    interface_depth: int,
    max_window_size: int,
    emit: Callable[[int, str], None] | None = None,
) -> _ResidualWindowHostSparsePreconditionerBundle | None:
    """Build a small kinetic-window Schur correction from failed residual energy."""

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import splu  # noqa: PLC0415

    t0 = time.perf_counter()
    matrix = operator_bundle.matrix
    if matrix is None or not hasattr(matrix, "tocsr"):
        return None
    matrix_csr = matrix.tocsr()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    if active_indices is None:
        active_np = np.arange(int(layout.total_size), dtype=np.int64)
    else:
        active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if residual_np.shape != active_np.shape or matrix_csr.shape[0] != residual_np.size:
        return None
    kinetic_mask = active_np < int(layout.f_size)
    if not bool(np.any(kinetic_mask)):
        return None
    kinetic_full = active_np[kinetic_mask]
    kinetic_positions = np.flatnonzero(kinetic_mask)
    kinetic_square = np.square(residual_np[kinetic_positions])
    decoded = layout.decode_kinetic_indices(kinetic_full)
    combo_index = (decoded.species * int(layout.n_x) + decoded.x) * int(layout.n_xi) + decoded.ell
    combo_energy = np.bincount(
        combo_index,
        weights=kinetic_square,
        minlength=int(layout.n_species * layout.n_x * layout.n_xi),
    )
    top_combo = np.argsort(combo_energy)[::-1]
    top_combo = top_combo[combo_energy[top_combo] > 0.0][: max(1, int(max_windows))]
    if top_combo.size == 0:
        return None

    window_positions: list[np.ndarray] = []
    window_metadata: list[dict[str, object]] = []
    used_positions: set[int] = set()
    for combo in top_combo:
        ell = int(combo % int(layout.n_xi))
        sx = int(combo // int(layout.n_xi))
        x = int(sx % int(layout.n_x))
        species = int(sx // int(layout.n_x))
        x_lo = max(0, x - max(0, int(x_radius)))
        x_hi = min(int(layout.n_x) - 1, x + max(0, int(x_radius)))
        ell_lo = max(0, ell - max(0, int(ell_radius)))
        ell_hi = min(int(layout.n_xi) - 1, ell + max(0, int(ell_radius)))
        window_mask = (
            (decoded.species == species)
            & (decoded.x >= x_lo)
            & (decoded.x <= x_hi)
            & (decoded.ell >= ell_lo)
            & (decoded.ell <= ell_hi)
        )
        positions = kinetic_positions[window_mask]
        if positions.size == 0:
            continue
        positions = np.asarray([int(pos) for pos in positions if int(pos) not in used_positions], dtype=np.int64)
        if positions.size == 0:
            continue
        used_positions.update(int(pos) for pos in positions)
        window_positions.append(positions)
        window_energy = float(np.sum(np.square(residual_np[positions])))
        window_metadata.append(
            {
                "species": int(species),
                "x_center": int(x),
                "ell_center": int(ell),
                "x_range": (int(x_lo), int(x_hi)),
                "ell_range": (int(ell_lo), int(ell_hi)),
                "size": int(positions.size),
                "residual_energy_fraction": (
                    float(window_energy / float(np.sum(np.square(residual_np))))
                    if float(np.sum(np.square(residual_np))) > 0.0
                    else 0.0
                ),
            }
        )
    if not window_positions:
        return None
    combine_mode_norm = str(combine_mode).strip().lower().replace("-", "_")
    if combine_mode_norm in {"union", "coupled", "interface", "graph_interface"}:
        union_positions = np.unique(np.concatenate(window_positions).astype(np.int64, copy=False))
        if int(max_window_size) > 0 and union_positions.size > int(max_window_size):
            return None
        if combine_mode_norm in {"interface", "graph_interface"} or int(interface_depth) > 0:
            expanded = _expand_sparse_graph_positions(
                matrix_csr,
                union_positions,
                depth=max(0, int(interface_depth)),
                max_size=int(max_window_size) if int(max_window_size) > 0 else int(matrix_csr.shape[0]),
            )
            if expanded is None:
                return None
            union_positions = expanded
        if int(max_window_size) > 0 and union_positions.size > int(max_window_size):
            return None
        window_positions = [union_positions]

    base_nbytes = int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    actual_total = int(base_nbytes)
    factors: list[object] = []
    factor_nnz_total = 0
    for positions in window_positions:
        submatrix = matrix_csr[positions[:, None], positions].tocsc()
        sub_scale = max(float(np.max(np.abs(submatrix.data))) if submatrix.nnz else 0.0, 1.0)
        diagonal_shift = max(float(abs(regularization)), 1.0e-14) * sub_scale
        if diagonal_shift > 0.0:
            submatrix = submatrix + diagonal_shift * sp.eye(submatrix.shape[0], dtype=np.float64, format="csc")
        matrix_nbytes = int(submatrix.data.nbytes + submatrix.indices.nbytes + submatrix.indptr.nbytes)
        if actual_total + matrix_nbytes > int(max_nbytes):
            return None
        try:
            factor = splu(submatrix, permc_spec="COLAMD", diag_pivot_thresh=0.0)
        except Exception:  # noqa: BLE001
            return None
        factor_nbytes = _sparse_factor_nbytes_estimate(factor)
        actual_total += int(factor_nbytes)
        factor_nnz_total += int(factor.L.nnz + factor.U.nnz)
        if actual_total > int(max_nbytes):
            return None
        factors.append(factor)

    metadata = {
        "window_count": int(len(window_positions)),
        "windows": tuple(window_metadata),
        "x_radius": int(x_radius),
        "ell_radius": int(ell_radius),
        "coefficient_mode": str(coefficient_mode),
        "combine_mode": str(combine_mode_norm),
        "interface_depth": int(interface_depth),
        "max_window_size": int(max_window_size),
        "factor_nbytes_estimate": int(actual_total),
        "base_factor_nbytes_estimate": int(base_nbytes),
        "factor_nnz_estimate": int(factor_nnz_total),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        first = window_metadata[0]
        emit(
            1,
            "solve_v3_full_system_linear_gmres: residual window built "
            f"windows={int(len(window_positions))} setup_s={metadata['setup_s']:.3f} "
            f"bytes={int(actual_total)} first=s{first['species']}/x{first['x_center']}/ell{first['ell_center']}",
        )
    return _ResidualWindowHostSparsePreconditionerBundle(
        base_factor=factor_bundle,
        operator=operator_bundle,
        window_positions=tuple(np.asarray(pos, dtype=np.int64) for pos in window_positions),
        window_factors=tuple(factors),
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_residual_window",
        coefficient_mode=str(coefficient_mode),
        regularization=float(regularization),
        factor_nbytes_estimate=int(actual_total),
        factor_nnz_estimate=int(factor_nnz_total),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _try_build_residual_coarse_host_sparse_preconditioner(
    *,
    operator_bundle: SparseOperatorBundle,
    factor_bundle: object,
    residual: np.ndarray,
    max_rank: int,
    max_nbytes: int,
    regularization: float,
    emit: Callable[[int, str], None] | None = None,
) -> _ResidualCoarseHostSparsePreconditionerBundle | None:
    """Build a small adaptive coarse residual equation from a failed preflight."""

    t0 = time.perf_counter()
    residual_np = np.asarray(residual, dtype=np.float64).reshape((-1,))
    n = int(residual_np.size)
    rank_limit = max(1, int(max_rank))
    z_cols: list[np.ndarray] = []
    az_cols: list[np.ndarray] = []
    work = residual_np.copy()
    for _ in range(rank_limit):
        try:
            z = np.asarray(factor_bundle.solve(work), dtype=np.float64).reshape((-1,))
        except Exception:  # noqa: BLE001
            return None
        if z.shape != (n,) or not np.all(np.isfinite(z)):
            return None
        for prev in z_cols:
            z = z - float(np.dot(prev, z)) * prev
        z_norm = float(np.linalg.norm(z))
        if (not np.isfinite(z_norm)) or z_norm <= 1.0e-14:
            break
        z = z / z_norm
        az = np.asarray(operator_bundle.matvec(z), dtype=np.float64).reshape((-1,))
        if az.shape != (n,) or not np.all(np.isfinite(az)):
            return None
        az_norm_sq = float(np.dot(az, az))
        if (not np.isfinite(az_norm_sq)) or az_norm_sq <= 1.0e-28:
            break
        z_cols.append(z)
        az_cols.append(az)
        alpha = float(np.dot(az, work)) / az_norm_sq
        work = work - alpha * az
        if float(np.linalg.norm(work)) >= float(np.linalg.norm(residual_np)):
            # Keep the basis already collected, but stop adding directions that
            # do not make the local residual equation easier.
            break

    if not z_cols:
        return None
    z_basis = np.column_stack(z_cols).astype(np.float64, copy=False)
    az_basis = np.column_stack(az_cols).astype(np.float64, copy=False)
    rank = int(z_basis.shape[1])
    normal = np.asarray(az_basis.T @ az_basis, dtype=np.float64)
    normal_scale = max(float(np.linalg.norm(normal, ord=np.inf)) if normal.size else 0.0, 1.0)
    reg = max(float(abs(regularization)), 1.0e-14) * normal_scale
    normal = normal + reg * np.eye(rank, dtype=np.float64)
    try:
        coarse_inverse = np.linalg.inv(normal)
        solver_kind = "inverse"
    except Exception:  # noqa: BLE001
        coarse_inverse = np.linalg.pinv(normal, rcond=max(float(abs(regularization)), 1.0e-14))
        solver_kind = "pinv"
    base_nbytes = int(getattr(factor_bundle, "factor_nbytes_estimate", 0) or 0)
    nbytes = int(base_nbytes + z_basis.nbytes + az_basis.nbytes + coarse_inverse.nbytes)
    if nbytes > int(max_nbytes):
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: residual coarse rejected by budget "
                f"bytes={int(nbytes)}>{int(max_nbytes)} rank={int(rank)}",
            )
        return None
    metadata = {
        "rank": int(rank),
        "regularization": float(reg),
        "coarse_solver": str(solver_kind),
        "factor_nbytes_estimate": int(nbytes),
        "base_factor_nbytes_estimate": int(base_nbytes),
        "z_basis_nbytes": int(z_basis.nbytes),
        "az_basis_nbytes": int(az_basis.nbytes),
        "setup_s": float(max(0.0, time.perf_counter() - t0)),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: residual coarse built "
            f"rank={int(rank)} setup_s={metadata['setup_s']:.3f} bytes={int(nbytes)}",
        )
    return _ResidualCoarseHostSparsePreconditionerBundle(
        base_factor=factor_bundle,
        operator=operator_bundle,
        z_basis=z_basis,
        az_basis=az_basis,
        coarse_inverse=coarse_inverse,
        kind=f"{getattr(factor_bundle, 'kind', 'host')}_residual_coarse",
        factor_nbytes_estimate=int(nbytes),
        factor_nnz_estimate=getattr(factor_bundle, "factor_nnz_estimate", None),
        factor_s=float(max(0.0, time.perf_counter() - t0)),
        metadata=metadata,
    )


def _rhs1_active_reduced_residual_diagnostics(
    *,
    residual: Any,
    layout: RHS1BlockLayout,
    active_indices: np.ndarray | None,
    top_k: int = 6,
) -> dict[str, object]:
    """Summarize where an active reduced RHSMode=1 residual lives."""

    residual_np = np.asarray(jax.device_get(residual), dtype=np.float64).reshape((-1,))
    if active_indices is None:
        full_indices = np.arange(int(layout.total_size), dtype=np.int64)
    else:
        full_indices = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if full_indices.shape != residual_np.shape:
        return {
            "selected": False,
            "reason": "shape_mismatch",
            "residual_shape": tuple(int(v) for v in residual_np.shape),
            "active_index_shape": tuple(int(v) for v in full_indices.shape),
        }
    square = np.square(residual_np)
    total_energy = float(np.sum(square))
    total_norm = float(np.sqrt(max(total_energy, 0.0)))

    def _energy_summary(mask: np.ndarray) -> dict[str, float]:
        energy = float(np.sum(square[mask])) if bool(np.any(mask)) else 0.0
        return {
            "norm": float(np.sqrt(max(energy, 0.0))),
            "energy_fraction": float(energy / total_energy) if total_energy > 0.0 else 0.0,
        }

    def _top_entries(values: np.ndarray, *, labels: list[str] | None = None) -> list[dict[str, object]]:
        values_np = np.asarray(values, dtype=np.float64).reshape((-1,))
        if values_np.size == 0:
            return []
        order = np.argsort(values_np)[::-1][: max(1, int(top_k))]
        out: list[dict[str, object]] = []
        for idx in order:
            energy = float(values_np[int(idx)])
            if energy <= 0.0:
                continue
            out.append(
                {
                    "index": int(idx),
                    "label": str(labels[int(idx)]) if labels is not None and int(idx) < len(labels) else str(idx),
                    "norm": float(np.sqrt(max(energy, 0.0))),
                    "energy_fraction": float(energy / total_energy) if total_energy > 0.0 else 0.0,
                }
            )
        return out

    kinetic_mask = full_indices < int(layout.f_size)
    phi1_start = int(layout.f_size)
    extra_start = int(layout.f_size + layout.phi1_size)
    phi1_mask = (full_indices >= phi1_start) & (full_indices < extra_start)
    extra_mask = full_indices >= extra_start
    diagnostics: dict[str, object] = {
        "selected": True,
        "total_norm": float(total_norm),
        "component_norms": {
            "kinetic": _energy_summary(kinetic_mask),
            "phi1": _energy_summary(phi1_mask),
            "extra": _energy_summary(extra_mask),
        },
        "max_abs": float(np.max(np.abs(residual_np))) if residual_np.size else 0.0,
    }
    if bool(np.any(kinetic_mask)):
        kinetic_full = full_indices[kinetic_mask]
        kinetic_square = square[kinetic_mask]
        decoded = layout.decode_kinetic_indices(kinetic_full)
        species_energy = np.bincount(decoded.species, weights=kinetic_square, minlength=int(layout.n_species))
        x_energy = np.bincount(decoded.x, weights=kinetic_square, minlength=int(layout.n_x))
        ell_energy = np.bincount(decoded.ell, weights=kinetic_square, minlength=int(layout.n_xi))
        sx_index = decoded.species * int(layout.n_x) + decoded.x
        sx_energy = np.bincount(
            sx_index,
            weights=kinetic_square,
            minlength=int(layout.n_species * layout.n_x),
        )
        sx_labels = [
            f"s={int(species)},x={int(x)}"
            for species in range(int(layout.n_species))
            for x in range(int(layout.n_x))
        ]
        diagnostics["top_species"] = _top_entries(species_energy)
        diagnostics["top_x"] = _top_entries(x_energy)
        diagnostics["top_ell"] = _top_entries(ell_energy)
        diagnostics["top_species_x"] = _top_entries(sx_energy, labels=sx_labels)
    if bool(np.any(extra_mask)):
        diagnostics["extra_values"] = [
            float(v) for v in residual_np[extra_mask][: max(0, min(int(top_k), int(np.count_nonzero(extra_mask))))]
        ]
    return diagnostics
