"""Sparse structural patterns for profile-response full-system operators."""

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


def _fortran_reduced_x_supports(op: Any, *, preconditioner_x: int) -> list[range]:
    n_x = int(op.n_x)
    mode = int(preconditioner_x)
    if mode == 0:
        all_x = range(n_x)
        return [all_x for _ in range(n_x)]
    if mode == 1:
        return [range(ix, ix + 1) for ix in range(n_x)]
    if mode == 2:
        return [range(ix, n_x) for ix in range(n_x)]
    if mode in {3, 5}:
        return [range(max(0, ix - 1), min(n_x, ix + 2)) for ix in range(n_x)]
    if mode == 4:
        return [range(ix, min(n_x, ix + 2)) for ix in range(n_x)]
    return [range(ix, ix + 1) for ix in range(n_x)]


def _fortran_reduced_x_supports_for_l(
    op: Any,
    *,
    ell: int,
    preconditioner_x: int,
    preconditioner_x_min_l: int,
) -> list[range]:
    if int(ell) < max(0, int(preconditioner_x_min_l)):
        all_x = range(int(op.n_x))
        return [all_x for _ in range(int(op.n_x))]
    return _fortran_reduced_x_supports(op, preconditioner_x=int(preconditioner_x))


def _fortran_reduced_l_supports(ell: int, n_xi: int, *, preconditioner_xi: int) -> range:
    # Fortran whichMatrix=0 drops selected +/-2 xi couplings when
    # preconditioner_xi=1, while preserving streaming/mirror +/-1 couplings.
    # `build_operator_from_pattern` then materializes the pattern-projected
    # reduced preconditioner, which is exactly what this support describes.
    radius = 1 if int(preconditioner_xi) > 0 else 2
    return _nearby_l(int(ell), int(n_xi), radius=radius)


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


def estimate_v3_full_system_conservative_sparsity_summary(
    op: Any,
    *,
    fp_dense_velocity_block: bool | None = None,
) -> V3SparsePatternSummary:
    """Return a cheap upper-bound summary before materializing the pattern.

    The estimate intentionally overcounts duplicate row/column entries. It is
    used as a memory-safety preflight for experimental assembled-operator paths,
    where constructing Python row/column lists for a clearly infeasible pattern
    would itself be the failure mode.
    """

    n_total = int(op.total_size)
    if n_total <= 0:
        return V3SparsePatternSummary(
            shape=(0, 0),
            nnz=0,
            avg_row_nnz=0.0,
            max_row_nnz=0,
            include_phi1=bool(op.include_phi1),
            constraint_scheme=int(op.constraint_scheme),
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
        )

    if fp_dense_velocity_block is None:
        fp_dense_velocity_block = op.fblock.fp is not None

    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    theta_max = max((len(support) + 1 for support in theta_supports), default=1)
    zeta_max = max((len(support) + 1 for support in zeta_supports), default=1)
    has_fp = op.fblock.fp is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    species_cols = int(op.n_species) if has_fp else 1
    x_cols = int(op.n_x) if bool(fp_dense_velocity_block or has_xdot) else 1
    l_cols = int(op.n_xi) if bool(fp_dense_velocity_block) else min(int(op.n_xi), 5)
    f_rows = int(op.f_size)
    f_row_nnz = max(1, species_cols * x_cols * l_cols * (theta_max + zeta_max))
    nnz_estimate = int(f_rows * f_row_nnz)
    max_row_nnz = int(f_row_nnz)

    if bool(op.include_phi1):
        qn_rows = int(op.n_theta) * int(op.n_zeta)
        qn_row_nnz = int(op.n_species) * int(op.n_x) + 2
        nnz_estimate += int(qn_rows * qn_row_nnz)
        nnz_estimate += int(op.n_theta) * int(op.n_zeta)
        max_row_nnz = max(max_row_nnz, qn_row_nnz, int(op.n_theta) * int(op.n_zeta))
        if bool(op.include_phi1_in_kinetic):
            nnz_estimate += int(op.n_species) * int(op.n_x) * int(op.n_theta) * int(op.n_zeta) * (
                theta_max + zeta_max
            )

    ix0 = 1 if bool(op.point_at_x0) else 0
    if int(op.constraint_scheme) == 2:
        active_x = max(0, int(op.n_x) - ix0)
        nnz_estimate += int(op.n_species) * active_x * int(op.n_theta) * int(op.n_zeta)
        extra_row_nnz = int(op.n_theta) * int(op.n_zeta) + (1 if bool(op.point_at_x0) else 0)
        nnz_estimate += int(op.n_species) * int(op.n_x) * extra_row_nnz
        max_row_nnz = max(max_row_nnz, extra_row_nnz)
    elif int(op.constraint_scheme) in {1, 3, 4}:
        active_x = max(0, int(op.n_x) - ix0)
        nnz_estimate += int(op.n_species) * active_x * int(op.n_theta) * int(op.n_zeta) * 2
        moment_row_nnz = int(op.n_x) * int(op.n_theta) * int(op.n_zeta)
        nnz_estimate += int(op.n_species) * 2 * moment_row_nnz
        max_row_nnz = max(max_row_nnz, moment_row_nnz)

    avg_row_nnz = float(nnz_estimate) / float(max(n_total, 1))
    return V3SparsePatternSummary(
        shape=(n_total, n_total),
        nnz=int(nnz_estimate),
        avg_row_nnz=float(avg_row_nnz),
        max_row_nnz=int(max_row_nnz),
        include_phi1=bool(op.include_phi1),
        constraint_scheme=int(op.constraint_scheme),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
    )


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


def _active_transport_structured_pattern(
    op: Any,
    active_indices: np.ndarray,
    *,
    active_mask: np.ndarray,
    full_to_active: np.ndarray,
    fp_dense_velocity_block: bool,
) -> sp.csr_matrix | None:
    """Build the common active transport pattern as ``velocity ⊗ angular``.

    The active-DOF transport ordering used by ``_transport_active_dof_indices``
    keeps complete ``(theta,zeta)`` blocks for each active ``(species,x,L)``
    velocity mode, followed by all Phi1/constraint tail variables. In that case
    the kinetic block is a Kronecker product of the conservative velocity graph
    and the conservative angular stencil, which avoids Python appends over every
    ``(velocity,theta,zeta)`` candidate in large 3D full-FP grids.
    """

    n_tz = int(op.n_theta) * int(op.n_zeta)
    n_vel = int(op.n_species) * int(op.n_x) * int(op.n_xi)
    if n_tz <= 0 or n_vel <= 0:
        return None

    f_active = np.asarray(active_indices[active_indices < int(op.f_size)], dtype=np.int64)
    tail_active = np.asarray(active_indices[active_indices >= int(op.f_size)], dtype=np.int64)
    if int(f_active.size) == 0 or int(f_active.size) % n_tz != 0:
        return None
    expected_tail = np.arange(int(op.f_size), int(op.total_size), dtype=np.int64)
    if not np.array_equal(tail_active, expected_tail):
        return None

    active_vel_all = f_active // n_tz
    active_tz = f_active % n_tz
    counts = np.bincount(active_vel_all, minlength=n_vel)
    active_vel = np.flatnonzero(counts > 0).astype(np.int32)
    if not np.all(counts[active_vel] == n_tz):
        return None
    expected_f = (active_vel[:, None].astype(np.int64) * n_tz + np.arange(n_tz, dtype=np.int64)[None, :]).reshape((-1,))
    if not np.array_equal(f_active, expected_f) or not np.array_equal(active_tz.reshape((-1, n_tz))[0], np.arange(n_tz)):
        return None

    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    angular_rows: list[int] = []
    angular_cols: list[int] = []
    for theta in range(int(op.n_theta)):
        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
        for zeta in range(int(op.n_zeta)):
            row = theta * int(op.n_zeta) + zeta
            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
            for theta_col in theta_cols:
                angular_rows.append(row)
                angular_cols.append(int(theta_col) * int(op.n_zeta) + zeta)
            for zeta_col in zeta_cols:
                angular_rows.append(row)
                angular_cols.append(theta * int(op.n_zeta) + int(zeta_col))
    angular = sp.coo_matrix(
        (np.ones(len(angular_rows), dtype=bool), (np.asarray(angular_rows), np.asarray(angular_cols))),
        shape=(n_tz, n_tz),
        dtype=bool,
    ).tocsr()
    angular.sum_duplicates()
    angular.eliminate_zeros()

    vel_to_reduced = np.full((n_vel,), -1, dtype=np.int32)
    vel_to_reduced[active_vel] = np.arange(int(active_vel.size), dtype=np.int32)
    velocity_rows: list[int] = []
    velocity_cols: list[int] = []
    has_fp = op.fblock.fp is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    x_supports = _x_supports(op, dense=bool(fp_dense_velocity_block or has_xdot))
    for row_reduced, vel in enumerate(active_vel):
        s = int(vel // (int(op.n_x) * int(op.n_xi)))
        rem = int(vel % (int(op.n_x) * int(op.n_xi)))
        ix = int(rem // int(op.n_xi))
        ell = int(rem % int(op.n_xi))
        species_cols = range(int(op.n_species)) if has_fp else range(s, s + 1)
        x_cols = x_supports[ix]
        l_cols = range(int(op.n_xi)) if bool(fp_dense_velocity_block) else _nearby_l(ell, int(op.n_xi), radius=2)
        for s_col in species_cols:
            for x_col in x_cols:
                for ell_col in l_cols:
                    col_vel = ((int(s_col) * int(op.n_x) + int(x_col)) * int(op.n_xi)) + int(ell_col)
                    col_reduced = int(vel_to_reduced[col_vel])
                    if col_reduced >= 0:
                        velocity_rows.append(int(row_reduced))
                        velocity_cols.append(col_reduced)
    velocity = sp.coo_matrix(
        (np.ones(len(velocity_rows), dtype=bool), (np.asarray(velocity_rows), np.asarray(velocity_cols))),
        shape=(int(active_vel.size), int(active_vel.size)),
        dtype=bool,
    ).tocsr()
    velocity.sum_duplicates()
    velocity.eliminate_zeros()

    f_pattern = sp.kron(velocity, angular, format="coo")
    rows_parts = [np.asarray(f_pattern.row, dtype=np.int32)]
    cols_parts = [np.asarray(f_pattern.col, dtype=np.int32)]

    tail_rows: list[int] = []
    tail_cols: list[int] = []

    def append_if_active(row_full: int, col_full: int) -> None:
        if active_mask[int(row_full)] and active_mask[int(col_full)]:
            tail_rows.append(int(full_to_active[int(row_full)]))
            tail_cols.append(int(full_to_active[int(col_full)]))

    if bool(op.include_phi1):
        qn_start = int(op.f_size)
        lambda_row = int(op.f_size) + int(op.n_theta) * int(op.n_zeta)
        lambda_col = _lambda_index(op)
        for theta in range(int(op.n_theta)):
            for zeta in range(int(op.n_zeta)):
                row = qn_start + theta * int(op.n_zeta) + zeta
                for s in range(int(op.n_species)):
                    for ix in range(int(op.n_x)):
                        append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                append_if_active(row, _phi1_index(op, theta, zeta))
                append_if_active(row, lambda_col)

        for theta in range(int(op.n_theta)):
            for zeta in range(int(op.n_zeta)):
                append_if_active(lambda_row, _phi1_index(op, theta, zeta))

        if bool(op.include_phi1_in_kinetic):
            for s in range(int(op.n_species)):
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                        for zeta in range(int(op.n_zeta)):
                            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                            row = _f_index(op, s, ix, 0, theta, zeta)
                            for theta_col in theta_cols:
                                append_if_active(row, _phi1_index(op, int(theta_col), zeta))
                            for zeta_col in zeta_cols:
                                append_if_active(row, _phi1_index(op, theta, int(zeta_col)))

    ix0 = 1 if bool(op.point_at_x0) else 0
    constraint_scheme = int(op.constraint_scheme)
    if constraint_scheme == 2:
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                extra_col = _extra_index(op, s * int(op.n_x) + ix)
                if ix >= ix0:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(_f_index(op, s, ix, 0, theta, zeta), extra_col)

                row = _extra_index(op, s * int(op.n_x) + ix)
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                if bool(op.point_at_x0) and ix == 0:
                    append_if_active(row, extra_col)
    elif constraint_scheme in {1, 3, 4}:
        for s in range(int(op.n_species)):
            src_particle = _extra_index(op, 2 * s)
            src_energy = _extra_index(op, 2 * s + 1)
            for ix in range(ix0, int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        append_if_active(row, src_particle)
                        append_if_active(row, src_energy)

            density_row = _extra_index(op, 2 * s)
            pressure_row = _extra_index(op, 2 * s + 1)
            for ix in range(int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        col = _f_index(op, s, ix, 0, theta, zeta)
                        append_if_active(density_row, col)
                        append_if_active(pressure_row, col)
    elif constraint_scheme != 0:
        return None

    if tail_rows:
        rows_parts.append(np.asarray(tail_rows, dtype=np.int32))
        cols_parts.append(np.asarray(tail_cols, dtype=np.int32))
    rows = np.concatenate(rows_parts) if len(rows_parts) > 1 else rows_parts[0]
    cols = np.concatenate(cols_parts) if len(cols_parts) > 1 else cols_parts[0]
    data = np.ones(int(rows.size), dtype=bool)
    pattern = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(int(active_indices.size), int(active_indices.size)),
        dtype=bool,
    ).tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def v3_full_system_conservative_sparsity_pattern_for_indices(
    op: Any,
    active_indices: np.ndarray,
    *,
    fp_dense_velocity_block: bool | None = None,
) -> sp.csr_matrix:
    """Return a conservative pattern restricted to selected full-vector indices.

    This is the memory-safe companion to
    :func:`v3_full_system_conservative_sparsity_pattern`. It maps the requested
    full-system row/column indices to a reduced active coordinate system while
    generating candidates, so large ``Nxi_for_x``-truncated systems do not first
    materialize the full inactive pattern only to slice it afterwards.
    """

    n_total = int(op.total_size)
    active_indices = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
    if n_total <= 0 or int(active_indices.size) == 0:
        return sp.csr_matrix((0, 0), dtype=bool)
    if np.any(active_indices < 0) or np.any(active_indices >= n_total):
        raise ValueError("active_indices contains entries outside the full-system vector")

    if fp_dense_velocity_block is None:
        fp_dense_velocity_block = op.fblock.fp is not None

    active_mask = np.zeros((n_total,), dtype=bool)
    active_mask[active_indices] = True
    full_to_active = np.full((n_total,), -1, dtype=np.int32)
    full_to_active[active_indices] = np.arange(int(active_indices.size), dtype=np.int32)

    structured = _active_transport_structured_pattern(
        op,
        active_indices,
        active_mask=active_mask,
        full_to_active=full_to_active,
        fp_dense_velocity_block=bool(fp_dense_velocity_block),
    )
    if structured is not None:
        return structured

    rows: list[int] = []
    cols: list[int] = []

    def append_if_active(row_full: int, col_full: int) -> None:
        if active_mask[int(col_full)]:
            rows.append(int(full_to_active[int(row_full)]))
            cols.append(int(full_to_active[int(col_full)]))

    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    has_fp = op.fblock.fp is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    x_supports = _x_supports(op, dense=bool(fp_dense_velocity_block or has_xdot))

    for s in range(int(op.n_species)):
        species_cols = range(int(op.n_species)) if has_fp else range(s, s + 1)
        for ix in range(int(op.n_x)):
            x_cols = x_supports[ix]
            for ell in range(int(op.n_xi)):
                l_cols = range(int(op.n_xi)) if bool(fp_dense_velocity_block) else _nearby_l(ell, int(op.n_xi), radius=2)
                for theta in range(int(op.n_theta)):
                    theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, ell, theta, zeta)
                        if not active_mask[row]:
                            continue
                        zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                        for s_col in species_cols:
                            for x_col in x_cols:
                                for ell_col in l_cols:
                                    for theta_col in theta_cols:
                                        append_if_active(row, _f_index(op, s_col, x_col, ell_col, int(theta_col), zeta))
                                    for zeta_col in zeta_cols:
                                        append_if_active(row, _f_index(op, s_col, x_col, ell_col, theta, int(zeta_col)))

    if bool(op.include_phi1):
        qn_start = int(op.f_size)
        lambda_row = int(op.f_size) + int(op.n_theta) * int(op.n_zeta)
        lambda_col = _lambda_index(op)
        for theta in range(int(op.n_theta)):
            for zeta in range(int(op.n_zeta)):
                row = qn_start + theta * int(op.n_zeta) + zeta
                if active_mask[row]:
                    for s in range(int(op.n_species)):
                        for ix in range(int(op.n_x)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    append_if_active(row, _phi1_index(op, theta, zeta))
                    append_if_active(row, lambda_col)

        if active_mask[lambda_row]:
            for theta in range(int(op.n_theta)):
                for zeta in range(int(op.n_zeta)):
                    append_if_active(lambda_row, _phi1_index(op, theta, zeta))

        if bool(op.include_phi1_in_kinetic):
            for s in range(int(op.n_species)):
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                        for zeta in range(int(op.n_zeta)):
                            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                            row = _f_index(op, s, ix, 0, theta, zeta)
                            if not active_mask[row]:
                                continue
                            for theta_col in theta_cols:
                                append_if_active(row, _phi1_index(op, int(theta_col), zeta))
                            for zeta_col in zeta_cols:
                                append_if_active(row, _phi1_index(op, theta, int(zeta_col)))

    ix0 = 1 if bool(op.point_at_x0) else 0
    constraint_scheme = int(op.constraint_scheme)
    if constraint_scheme == 2:
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                extra_col = _extra_index(op, s * int(op.n_x) + ix)
                if ix >= ix0:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            row = _f_index(op, s, ix, 0, theta, zeta)
                            if active_mask[row]:
                                append_if_active(row, extra_col)

                row = _extra_index(op, s * int(op.n_x) + ix)
                if active_mask[row]:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    if bool(op.point_at_x0) and ix == 0:
                        append_if_active(row, extra_col)
    elif constraint_scheme in {1, 3, 4}:
        for s in range(int(op.n_species)):
            src_particle = _extra_index(op, 2 * s)
            src_energy = _extra_index(op, 2 * s + 1)
            for ix in range(ix0, int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        if active_mask[row]:
                            append_if_active(row, src_particle)
                            append_if_active(row, src_energy)

            density_row = _extra_index(op, 2 * s)
            pressure_row = _extra_index(op, 2 * s + 1)
            if active_mask[density_row] or active_mask[pressure_row]:
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            col = _f_index(op, s, ix, 0, theta, zeta)
                            if active_mask[density_row]:
                                append_if_active(density_row, col)
                            if active_mask[pressure_row]:
                                append_if_active(pressure_row, col)
    elif constraint_scheme != 0:
        raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported by the sparse pattern builder.")

    if not rows:
        return sp.csr_matrix((int(active_indices.size), int(active_indices.size)), dtype=bool)
    data = np.ones(len(rows), dtype=bool)
    pattern = sp.coo_matrix(
        (data, (np.asarray(rows), np.asarray(cols))),
        shape=(int(active_indices.size), int(active_indices.size)),
        dtype=bool,
    ).tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def _fortran_reduced_transport_structured_pattern(
    op: Any,
    active_indices: np.ndarray,
    *,
    active_mask: np.ndarray,
    full_to_active: np.ndarray,
    preconditioner_x: int,
    preconditioner_xi: int,
    preconditioner_species: int,
    preconditioner_x_min_l: int,
) -> sp.csr_matrix | None:
    """Build the Fortran-reduced pattern as ``velocity_graph x angular_graph``.

    The Fortran ``whichMatrix=0`` preconditioner keeps the same angular
    streaming stencils as the full operator while reducing selected velocity
    couplings. For active sets that retain complete angular blocks, this product
    structure avoids Python row appends over every phase-space cell.
    """

    n_tz = int(op.n_theta) * int(op.n_zeta)
    n_vel = int(op.n_species) * int(op.n_x) * int(op.n_xi)
    if n_tz <= 0 or n_vel <= 0:
        return None

    active_indices = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
    f_active = np.asarray(active_indices[active_indices < int(op.f_size)], dtype=np.int64)
    tail_active = np.asarray(active_indices[active_indices >= int(op.f_size)], dtype=np.int64)
    expected_tail = np.arange(int(op.f_size), int(op.total_size), dtype=np.int64)
    if not np.array_equal(tail_active, expected_tail):
        return None
    if int(f_active.size) == 0 or int(f_active.size) % n_tz != 0:
        return None

    active_vel_all = f_active // n_tz
    active_tz = f_active % n_tz
    counts = np.bincount(active_vel_all, minlength=n_vel)
    active_vel = np.flatnonzero(counts > 0).astype(np.int32)
    if not np.all(counts[active_vel] == n_tz):
        return None
    expected_f = (active_vel[:, None].astype(np.int64) * n_tz + np.arange(n_tz, dtype=np.int64)[None, :]).reshape(
        (-1,)
    )
    if not np.array_equal(f_active, expected_f):
        return None
    active_tz_blocks = active_tz.reshape((-1, n_tz))
    if active_tz_blocks.size and not np.array_equal(active_tz_blocks[0], np.arange(n_tz)):
        return None

    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    angular_rows: list[int] = []
    angular_cols: list[int] = []
    for theta in range(int(op.n_theta)):
        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
        for zeta in range(int(op.n_zeta)):
            row = theta * int(op.n_zeta) + zeta
            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
            for theta_col in theta_cols:
                angular_rows.append(row)
                angular_cols.append(int(theta_col) * int(op.n_zeta) + zeta)
            for zeta_col in zeta_cols:
                angular_rows.append(row)
                angular_cols.append(theta * int(op.n_zeta) + int(zeta_col))
    angular = sp.coo_matrix(
        (np.ones(len(angular_rows), dtype=bool), (np.asarray(angular_rows), np.asarray(angular_cols))),
        shape=(n_tz, n_tz),
        dtype=bool,
    ).tocsr()
    angular.sum_duplicates()
    angular.eliminate_zeros()

    vel_to_reduced = np.full((n_vel,), -1, dtype=np.int32)
    vel_to_reduced[active_vel] = np.arange(int(active_vel.size), dtype=np.int32)
    velocity_rows: list[int] = []
    velocity_cols: list[int] = []
    has_fp = op.fblock.fp is not None or getattr(op.fblock, "fp_phi1", None) is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    x_supports_default = _x_supports(op, dense=False)
    for row_reduced, vel in enumerate(active_vel):
        s = int(vel // (int(op.n_x) * int(op.n_xi)))
        rem = int(vel % (int(op.n_x) * int(op.n_xi)))
        ix = int(rem // int(op.n_xi))
        ell = int(rem % int(op.n_xi))
        species_cols = (
            range(int(op.n_species))
            if (has_fp and int(preconditioner_species) == 0)
            else range(s, s + 1)
        )
        if has_xdot and int(preconditioner_x_min_l) > 0:
            x_supports = _x_supports(op, dense=True)
        elif has_fp or has_xdot:
            x_supports = _fortran_reduced_x_supports_for_l(
                op,
                ell=int(ell),
                preconditioner_x=int(preconditioner_x),
                preconditioner_x_min_l=int(preconditioner_x_min_l),
            )
        else:
            x_supports = x_supports_default
        l_cols = _fortran_reduced_l_supports(
            ell,
            int(op.n_xi),
            preconditioner_xi=int(preconditioner_xi),
        )
        for s_col in species_cols:
            for x_col in x_supports[ix]:
                for ell_col in l_cols:
                    col_vel = ((int(s_col) * int(op.n_x) + int(x_col)) * int(op.n_xi)) + int(ell_col)
                    col_reduced = int(vel_to_reduced[col_vel])
                    if col_reduced >= 0:
                        velocity_rows.append(int(row_reduced))
                        velocity_cols.append(col_reduced)
    velocity = sp.coo_matrix(
        (np.ones(len(velocity_rows), dtype=bool), (np.asarray(velocity_rows), np.asarray(velocity_cols))),
        shape=(int(active_vel.size), int(active_vel.size)),
        dtype=bool,
    ).tocsr()
    velocity.sum_duplicates()
    velocity.eliminate_zeros()

    f_pattern = sp.kron(velocity, angular, format="coo")
    rows_parts = [np.asarray(f_pattern.row, dtype=np.int32)]
    cols_parts = [np.asarray(f_pattern.col, dtype=np.int32)]
    tail_rows: list[int] = []
    tail_cols: list[int] = []

    def append_if_active(row_full: int, col_full: int) -> None:
        if active_mask[int(row_full)] and active_mask[int(col_full)]:
            tail_rows.append(int(full_to_active[int(row_full)]))
            tail_cols.append(int(full_to_active[int(col_full)]))

    if bool(op.include_phi1):
        qn_start = int(op.f_size)
        lambda_row = int(op.f_size) + int(op.n_theta) * int(op.n_zeta)
        lambda_col = _lambda_index(op)
        for theta in range(int(op.n_theta)):
            for zeta in range(int(op.n_zeta)):
                row = qn_start + theta * int(op.n_zeta) + zeta
                if active_mask[row]:
                    for s in range(int(op.n_species)):
                        for ix in range(int(op.n_x)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    append_if_active(row, _phi1_index(op, theta, zeta))
                    append_if_active(row, lambda_col)

        if active_mask[lambda_row]:
            for theta in range(int(op.n_theta)):
                for zeta in range(int(op.n_zeta)):
                    append_if_active(lambda_row, _phi1_index(op, theta, zeta))

        if bool(op.include_phi1_in_kinetic):
            for s in range(int(op.n_species)):
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                        for zeta in range(int(op.n_zeta)):
                            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                            row = _f_index(op, s, ix, 0, theta, zeta)
                            if not active_mask[row]:
                                continue
                            for theta_col in theta_cols:
                                append_if_active(row, _phi1_index(op, int(theta_col), zeta))
                            for zeta_col in zeta_cols:
                                append_if_active(row, _phi1_index(op, theta, int(zeta_col)))

    ix0 = 1 if bool(op.point_at_x0) else 0
    constraint_scheme = int(op.constraint_scheme)
    if constraint_scheme == 2:
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                extra_col = _extra_index(op, s * int(op.n_x) + ix)
                if ix >= ix0:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(_f_index(op, s, ix, 0, theta, zeta), extra_col)

                row = _extra_index(op, s * int(op.n_x) + ix)
                if active_mask[row]:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    if bool(op.point_at_x0) and ix == 0:
                        append_if_active(row, extra_col)
    elif constraint_scheme in {1, 3, 4}:
        for s in range(int(op.n_species)):
            src_particle = _extra_index(op, 2 * s)
            src_energy = _extra_index(op, 2 * s + 1)
            for ix in range(ix0, int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        append_if_active(row, src_particle)
                        append_if_active(row, src_energy)

            density_row = _extra_index(op, 2 * s)
            pressure_row = _extra_index(op, 2 * s + 1)
            if active_mask[density_row] or active_mask[pressure_row]:
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            col = _f_index(op, s, ix, 0, theta, zeta)
                            append_if_active(density_row, col)
                            append_if_active(pressure_row, col)
    elif constraint_scheme != 0:
        return None

    if tail_rows:
        rows_parts.append(np.asarray(tail_rows, dtype=np.int32))
        cols_parts.append(np.asarray(tail_cols, dtype=np.int32))
    rows = np.concatenate(rows_parts) if len(rows_parts) > 1 else rows_parts[0]
    cols = np.concatenate(cols_parts) if len(cols_parts) > 1 else cols_parts[0]
    pattern = sp.coo_matrix(
        (np.ones(int(rows.size), dtype=bool), (rows, cols)),
        shape=(int(active_indices.size), int(active_indices.size)),
        dtype=bool,
    ).tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def _v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_impl(
    op: Any,
    *,
    active_indices: np.ndarray | None,
    preconditioner_x: int,
    preconditioner_xi: int,
    preconditioner_species: int,
    preconditioner_x_min_l: int,
) -> sp.csr_matrix:
    n_total = int(op.total_size)
    if active_indices is None:
        active_mask = np.ones((n_total,), dtype=bool)
        full_to_active = np.arange(n_total, dtype=np.int32)
        n_rows = n_total
    else:
        active_indices = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
        if active_indices.size == 0:
            return sp.csr_matrix((0, 0), dtype=bool)
        if np.any(active_indices < 0) or np.any(active_indices >= n_total):
            raise ValueError("active_indices contains entries outside the full-system vector")
        active_mask = np.zeros((n_total,), dtype=bool)
        active_mask[active_indices] = True
        full_to_active = np.full((n_total,), -1, dtype=np.int32)
        full_to_active[active_indices] = np.arange(int(active_indices.size), dtype=np.int32)
        n_rows = int(active_indices.size)

    structured = _fortran_reduced_transport_structured_pattern(
        op,
        active_indices=np.arange(n_total, dtype=np.int32) if active_indices is None else active_indices,
        active_mask=active_mask,
        full_to_active=full_to_active,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )
    if structured is not None:
        return structured

    rows: list[int] = []
    cols: list[int] = []

    def append_if_active(row_full: int, col_full: int) -> None:
        if active_mask[int(row_full)] and active_mask[int(col_full)]:
            rows.append(int(full_to_active[int(row_full)]))
            cols.append(int(full_to_active[int(col_full)]))

    theta_supports = _ddtheta_supports(op)
    zeta_supports = _ddzeta_supports(op)
    has_fp = op.fblock.fp is not None or op.fblock.fp_phi1 is not None
    has_xdot = op.fblock.er_xdot is not None or op.fblock.magdrift_xidot is not None
    x_supports_default = _x_supports(op, dense=False)

    for s in range(int(op.n_species)):
        species_cols = (
            range(int(op.n_species))
            if (has_fp and int(preconditioner_species) == 0)
            else range(s, s + 1)
        )
        for ix in range(int(op.n_x)):
            for ell in range(int(op.n_xi)):
                if has_xdot and int(preconditioner_x_min_l) > 0:
                    x_supports = _x_supports(op, dense=True)
                elif has_fp or has_xdot:
                    x_supports = _fortran_reduced_x_supports_for_l(
                        op,
                        ell=int(ell),
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                    )
                else:
                    x_supports = x_supports_default
                x_cols = x_supports[ix]
                l_cols = _fortran_reduced_l_supports(
                    ell,
                    int(op.n_xi),
                    preconditioner_xi=int(preconditioner_xi),
                )
                for theta in range(int(op.n_theta)):
                    theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, ell, theta, zeta)
                        if not active_mask[row]:
                            continue
                        zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                        for s_col in species_cols:
                            for x_col in x_cols:
                                for ell_col in l_cols:
                                    for theta_col in theta_cols:
                                        append_if_active(row, _f_index(op, s_col, x_col, ell_col, int(theta_col), zeta))
                                    for zeta_col in zeta_cols:
                                        append_if_active(row, _f_index(op, s_col, x_col, ell_col, theta, int(zeta_col)))

    if bool(op.include_phi1):
        qn_start = int(op.f_size)
        lambda_row = int(op.f_size) + int(op.n_theta) * int(op.n_zeta)
        lambda_col = _lambda_index(op)
        for theta in range(int(op.n_theta)):
            for zeta in range(int(op.n_zeta)):
                row = qn_start + theta * int(op.n_zeta) + zeta
                if active_mask[row]:
                    for s in range(int(op.n_species)):
                        for ix in range(int(op.n_x)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    append_if_active(row, _phi1_index(op, theta, zeta))
                    append_if_active(row, lambda_col)

        if active_mask[lambda_row]:
            for theta in range(int(op.n_theta)):
                for zeta in range(int(op.n_zeta)):
                    append_if_active(lambda_row, _phi1_index(op, theta, zeta))

        if bool(op.include_phi1_in_kinetic):
            for s in range(int(op.n_species)):
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        theta_cols = np.union1d(np.asarray([theta], dtype=np.int32), theta_supports[theta])
                        for zeta in range(int(op.n_zeta)):
                            zeta_cols = np.union1d(np.asarray([zeta], dtype=np.int32), zeta_supports[zeta])
                            row = _f_index(op, s, ix, 0, theta, zeta)
                            if not active_mask[row]:
                                continue
                            for theta_col in theta_cols:
                                append_if_active(row, _phi1_index(op, int(theta_col), zeta))
                            for zeta_col in zeta_cols:
                                append_if_active(row, _phi1_index(op, theta, int(zeta_col)))

    ix0 = 1 if bool(op.point_at_x0) else 0
    constraint_scheme = int(op.constraint_scheme)
    if constraint_scheme == 2:
        for s in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                extra_col = _extra_index(op, s * int(op.n_x) + ix)
                if ix >= ix0:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(_f_index(op, s, ix, 0, theta, zeta), extra_col)

                row = _extra_index(op, s * int(op.n_x) + ix)
                if active_mask[row]:
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            append_if_active(row, _f_index(op, s, ix, 0, theta, zeta))
                    if bool(op.point_at_x0) and ix == 0:
                        append_if_active(row, extra_col)
    elif constraint_scheme in {1, 3, 4}:
        for s in range(int(op.n_species)):
            src_particle = _extra_index(op, 2 * s)
            src_energy = _extra_index(op, 2 * s + 1)
            for ix in range(ix0, int(op.n_x)):
                for theta in range(int(op.n_theta)):
                    for zeta in range(int(op.n_zeta)):
                        row = _f_index(op, s, ix, 0, theta, zeta)
                        append_if_active(row, src_particle)
                        append_if_active(row, src_energy)

            density_row = _extra_index(op, 2 * s)
            pressure_row = _extra_index(op, 2 * s + 1)
            if active_mask[density_row] or active_mask[pressure_row]:
                for ix in range(int(op.n_x)):
                    for theta in range(int(op.n_theta)):
                        for zeta in range(int(op.n_zeta)):
                            col = _f_index(op, s, ix, 0, theta, zeta)
                            append_if_active(density_row, col)
                            append_if_active(pressure_row, col)
    elif constraint_scheme != 0:
        raise NotImplementedError(f"constraintScheme={constraint_scheme} is not supported by the sparse pattern builder.")

    if not rows:
        return sp.csr_matrix((n_rows, n_rows), dtype=bool)
    pattern = sp.coo_matrix(
        (np.ones(len(rows), dtype=bool), (np.asarray(rows), np.asarray(cols))),
        shape=(n_rows, n_rows),
        dtype=bool,
    ).tocsr()
    pattern.sum_duplicates()
    pattern.eliminate_zeros()
    return pattern


def v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
    op: Any,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
) -> sp.csr_matrix:
    """Return a compact structural pattern for the Fortran-v3 reduced preconditioner."""

    return _v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_impl(
        op,
        active_indices=None,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )


def v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
    op: Any,
    active_indices: np.ndarray,
    *,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    preconditioner_species: int = 1,
    preconditioner_x_min_l: int = 0,
) -> sp.csr_matrix:
    """Return the Fortran-reduced structural pattern restricted to active DOFs."""

    return _v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_impl(
        op,
        active_indices=active_indices,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
    )


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
