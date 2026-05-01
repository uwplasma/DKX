from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class V3SparsePatternSummary:
    """Summary of a conservative full-system sparsity pattern."""

    shape: tuple[int, int]
    nnz: int
    avg_row_nnz: float
    max_row_nnz: int
    include_phi1: bool
    constraint_scheme: int
    has_fp: bool
    has_pas: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "shape": self.shape,
            "nnz": int(self.nnz),
            "avg_row_nnz": float(self.avg_row_nnz),
            "max_row_nnz": int(self.max_row_nnz),
            "include_phi1": bool(self.include_phi1),
            "constraint_scheme": int(self.constraint_scheme),
            "has_fp": bool(self.has_fp),
            "has_pas": bool(self.has_pas),
        }


def _f_index(op: Any, s: int, x: int, ell: int, theta: int, zeta: int) -> int:
    return (((int(s) * int(op.n_x) + int(x)) * int(op.n_xi) + int(ell)) * int(op.n_theta) + int(theta)) * int(op.n_zeta) + int(zeta)


def _phi1_index(op: Any, theta: int, zeta: int) -> int:
    return int(op.f_size) + int(theta) * int(op.n_zeta) + int(zeta)


def _lambda_index(op: Any) -> int:
    return int(op.f_size) + int(op.n_theta) * int(op.n_zeta)


def _extra_index(op: Any, offset: int) -> int:
    return int(op.f_size) + int(op.phi1_size) + int(offset)


def _matrix_row_supports(matrix: Any, *, threshold: float = 1.0e-14) -> list[np.ndarray]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected a 2D derivative matrix, got shape {arr.shape}")
    supports: list[np.ndarray] = []
    for row in range(arr.shape[0]):
        cols = np.flatnonzero(np.abs(arr[row, :]) > float(threshold)).astype(np.int32)
        supports.append(cols)
    return supports


def _ddtheta_supports(op: Any) -> list[np.ndarray]:
    return _matrix_row_supports(op.fblock.collisionless.ddtheta)


def _ddzeta_supports(op: Any) -> list[np.ndarray]:
    return _matrix_row_supports(op.fblock.collisionless.ddzeta)


def _nearby_l(ell: int, n_xi: int, *, radius: int = 2) -> range:
    lo = max(0, int(ell) - int(radius))
    hi = min(int(n_xi), int(ell) + int(radius) + 1)
    return range(lo, hi)


def _x_supports(op: Any, *, dense: bool) -> list[range]:
    if dense:
        all_x = range(int(op.n_x))
        return [all_x for _ in range(int(op.n_x))]
    return [range(ix, ix + 1) for ix in range(int(op.n_x))]


def _append_fblock_candidates(
    *,
    op: Any,
    rows: list[int],
    cols: list[int],
    dense_velocity_block: bool,
) -> None:
    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    has_fp = op.fblock.fp is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    x_supports = _x_supports(op, dense=bool(dense_velocity_block or has_xdot))

    for s in range(int(op.n_species)):
        species_cols = range(int(op.n_species)) if has_fp else range(s, s + 1)
        for ix in range(int(op.n_x)):
            x_cols = x_supports[ix]
            for ell in range(int(op.n_xi)):
                l_cols = range(int(op.n_xi)) if dense_velocity_block else _nearby_l(ell, int(op.n_xi), radius=2)
                for theta in range(int(op.n_theta)):
                    theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, ell, theta, zeta)
                        zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                        for s_col in species_cols:
                            for x_col in x_cols:
                                for ell_col in l_cols:
                                    for theta_col in theta_cols:
                                        rows.append(row)
                                        cols.append(_f_index(op, s_col, x_col, ell_col, int(theta_col), zeta))
                                    for zeta_col in zeta_cols:
                                        rows.append(row)
                                        cols.append(_f_index(op, s_col, x_col, ell_col, theta, int(zeta_col)))


def _append_phi1_candidates(*, op: Any, rows: list[int], cols: list[int]) -> None:
    if not bool(op.include_phi1):
        return

    qn_start = int(op.f_size)
    lambda_row = int(op.f_size) + int(op.n_theta) * int(op.n_zeta)
    lambda_col = _lambda_index(op)

    for theta in range(int(op.n_theta)):
        for zeta in range(int(op.n_zeta)):
            row = qn_start + theta * int(op.n_zeta) + zeta
            for s in range(int(op.n_species)):
                for ix in range(int(op.n_x)):
                    rows.append(row)
                    cols.append(_f_index(op, s, ix, 0, theta, zeta))
            rows.append(row)
            cols.append(_phi1_index(op, theta, zeta))
            rows.append(row)
            cols.append(lambda_col)

    for theta in range(int(op.n_theta)):
        for zeta in range(int(op.n_zeta)):
            rows.append(lambda_row)
            cols.append(_phi1_index(op, theta, zeta))

    if bool(op.include_phi1_in_kinetic):
        theta_supports = _ddtheta_supports(op)
        zeta_supports = _ddzeta_supports(op)
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                    for zeta in range(int(op.n_zeta)):
                        zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        for theta_col in theta_cols:
                            rows.append(row)
                            cols.append(_phi1_index(op, int(theta_col), zeta))
                        for zeta_col in zeta_cols:
                            rows.append(row)
                            cols.append(_phi1_index(op, theta, int(zeta_col)))


def _append_constraint_candidates(*, op: Any, rows: list[int], cols: list[int]) -> None:
    ix0 = 1 if bool(op.point_at_x0) else 0
    constraint_scheme = int(op.constraint_scheme)
    if constraint_scheme == 0:
        return

    if constraint_scheme == 2:
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                extra_col = _extra_index(op, s * int(op.n_x) + ix)
                if ix >= ix0:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            rows.append(_f_index(op, s, ix, 0, theta, zeta))
                            cols.append(extra_col)

                row = _extra_index(op, s * int(op.n_x) + ix)
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        rows.append(row)
                        cols.append(_f_index(op, s, ix, 0, theta, zeta))
                if bool(op.point_at_x0) and ix == 0:
                    rows.append(row)
                    cols.append(extra_col)
        return

    if constraint_scheme in {1, 3, 4}:
        for s in range(int(op.n_species)):
            src_particle = _extra_index(op, 2 * s)
            src_energy = _extra_index(op, 2 * s + 1)
            for ix in range(ix0, int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        rows.append(row)
                        cols.append(src_particle)
                        rows.append(row)
                        cols.append(src_energy)

            density_row = _extra_index(op, 2 * s)
            pressure_row = _extra_index(op, 2 * s + 1)
            for ix in range(int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        col = _f_index(op, s, ix, 0, theta, zeta)
                        rows.append(density_row)
                        cols.append(col)
                        rows.append(pressure_row)
                        cols.append(col)
        return

    raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported by the sparse pattern builder.")


def v3_full_system_conservative_sparsity_pattern(
    op: Any,
    *,
    fp_dense_velocity_block: bool | None = None,
) -> sp.csr_matrix:
    """Return a conservative CSR sparsity pattern for the v3 full-system operator.

    The pattern is intentionally a superset. It encodes local spatial stencil
    structure, dense same-``(theta,zeta)`` velocity/species blocks for
    Fokker-Planck collisions, Phi1/quasineutrality couplings, and constraint
    rows/columns. A later value-probing pass drops structural zeros.
    """

    if int(op.total_size) <= 0:
        return sp.csr_matrix((0, 0), dtype=bool)

    if fp_dense_velocity_block is None:
        fp_dense_velocity_block = op.fblock.fp is not None

    rows: list[int] = []
    cols: list[int] = []
    _append_fblock_candidates(
        op=op,
        rows=rows,
        cols=cols,
        dense_velocity_block=bool(fp_dense_velocity_block),
    )
    _append_phi1_candidates(op=op, rows=rows, cols=cols)
    _append_constraint_candidates(op=op, rows=rows, cols=cols)

    data = np.ones(len(rows), dtype=bool)
    pattern = sp.coo_matrix((data, (np.asarray(rows), np.asarray(cols))), shape=(int(op.total_size), int(op.total_size)), dtype=bool).tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def summarize_v3_sparse_pattern(op: Any, pattern: sp.spmatrix) -> V3SparsePatternSummary:
    pattern_csr = pattern.tocsr()
    row_nnz = np.diff(pattern_csr.indptr)
    max_row_nnz = int(row_nnz.max()) if row_nnz.size else 0
    avg_row_nnz = float(row_nnz.mean()) if row_nnz.size else 0.0
    return V3SparsePatternSummary(
        shape=tuple(int(v) for v in pattern_csr.shape),
        nnz=int(pattern_csr.nnz),
        avg_row_nnz=avg_row_nnz,
        max_row_nnz=max_row_nnz,
        include_phi1=bool(op.include_phi1),
        constraint_scheme=int(op.constraint_scheme),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
    )
