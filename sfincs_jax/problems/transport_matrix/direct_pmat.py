"""Direct reduced-Pmat assembly for RHSMode=2/3 transport solves.

This module owns the Fortran-v3-style term-level reduced ``Pmat`` emission used
by production transport preconditioners.  It is intentionally narrower than the
solver policy in :mod:`sfincs_jax.v3_driver`: the functions here build sparse
operators and physics coarse bases, while admission, factor selection, and
fallback ordering stay in the driver/preconditioner policy layers.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import numpy as np

from sfincs_jax.explicit_sparse import SparseDecision, SparseOperatorBundle, estimate_csr_nbytes, estimate_dense_nbytes
from sfincs_jax.rhs1_fblock_assembly import select_structured_rhs1_fblock_operator
from sfincs_jax.v3_system import V3FullSystemOperator, _fs_average_factor, _ix_min, _source_basis_constraint_scheme_1
from sfincs_jax.verbose import Timer

__all__ = [
    "_build_rhsmode23_direct_pmat_physics_coarse_basis",
    "_try_build_rhsmode23_fp_direct_active_operator_bundle",
    "_try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle",
]


def _build_rhsmode23_direct_pmat_physics_coarse_basis(
    *,
    op: V3FullSystemOperator,
    active_indices: np.ndarray,
    max_cols: int,
    base_factor_bundle: object | None = None,
) -> tuple[object | None, tuple[str, ...]]:
    """Build physics moment/source columns in active direct-Pmat coordinates."""

    try:
        import scipy.sparse as sp  # noqa: PLC0415
    except Exception:
        return None, ()

    active_np = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active_np.size == 0:
        return None, ()
    n_active = int(active_np.size)
    n_total = int(op.total_size)
    full_to_active = np.full((n_total,), -1, dtype=np.int64)
    full_to_active[active_np] = np.arange(n_active, dtype=np.int64)
    max_cols_use = max(1, int(max_cols))
    col_rows: list[np.ndarray] = []
    col_data: list[np.ndarray] = []
    names: list[str] = []

    f_active_mask = active_np < int(op.f_size)
    f_active_pos = np.flatnonzero(f_active_mask).astype(np.int64, copy=False)
    f_linear = active_np[f_active_mask]
    if f_linear.size:
        zeta_idx = (f_linear % int(op.n_zeta)).astype(np.int64, copy=False)
        tmp = f_linear // int(op.n_zeta)
        theta_idx = (tmp % int(op.n_theta)).astype(np.int64, copy=False)
        tmp = tmp // int(op.n_theta)
        ell_idx = (tmp % int(op.n_xi)).astype(np.int64, copy=False)
        tmp = tmp // int(op.n_xi)
        x_idx = (tmp % int(op.n_x)).astype(np.int64, copy=False)
        species_idx = (tmp // int(op.n_x)).astype(np.int64, copy=False)
    else:
        zeta_idx = np.asarray([], dtype=np.int64)
        theta_idx = np.asarray([], dtype=np.int64)
        ell_idx = np.asarray([], dtype=np.int64)
        x_idx = np.asarray([], dtype=np.int64)
        species_idx = np.asarray([], dtype=np.int64)

    def _add_active_column(name: str, active_values: np.ndarray) -> None:
        if len(names) >= max_cols_use:
            return
        values = np.asarray(active_values, dtype=np.float64).reshape((n_active,))
        keep = np.flatnonzero(np.isfinite(values) & (np.abs(values) > 0.0))
        if keep.size == 0:
            return
        vals = values[keep]
        norm = float(np.linalg.norm(vals))
        if not (np.isfinite(norm) and norm > 0.0):
            return
        col_rows.append(keep.astype(np.int64, copy=False))
        col_data.append((vals / norm).astype(np.float64, copy=False))
        names.append(str(name))

    def _add_f_fsavg_column(
        name: str,
        *,
        species: int,
        ell: int,
        x_weights: np.ndarray,
        fs_pattern: np.ndarray,
    ) -> None:
        if len(names) >= max_cols_use or f_active_pos.size == 0:
            return
        weights = np.asarray(x_weights, dtype=np.float64).reshape((int(op.n_x),))
        mask = (species_idx == int(species)) & (ell_idx == int(ell))
        if not np.any(mask):
            return
        values = np.zeros((n_active,), dtype=np.float64)
        local_pos = f_active_pos[mask]
        values[local_pos] = weights[x_idx[mask]] * fs_pattern[theta_idx[mask], zeta_idx[mask]]
        _add_active_column(name, values)

    def _add_tail_unit(name: str, full_index: int) -> None:
        if len(names) >= max_cols_use:
            return
        pos = full_to_active[int(full_index)] if 0 <= int(full_index) < n_total else -1
        if pos < 0:
            return
        col_rows.append(np.asarray([int(pos)], dtype=np.int64))
        col_data.append(np.asarray([1.0], dtype=np.float64))
        names.append(str(name))

    tail0 = int(op.f_size + op.phi1_size)
    for i_extra in range(int(op.extra_size)):
        _add_tail_unit(f"direct_pmat_tail_unit_{i_extra}", tail0 + int(i_extra))

    factor = np.asarray(
        jax.device_get(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat)),
        dtype=np.float64,
    )
    factor_norm = float(np.linalg.norm(factor))
    if np.isfinite(factor_norm) and factor_norm > 0.0:
        fs_pattern = factor / factor_norm
    else:
        fs_pattern = np.full(
            (int(op.n_theta), int(op.n_zeta)),
            1.0 / np.sqrt(max(1, int(op.n_theta) * int(op.n_zeta))),
            dtype=np.float64,
        )

    if int(op.constraint_scheme) == 1:
        try:
            xpart1, xpart2 = _source_basis_constraint_scheme_1(op.x)
            x = np.asarray(jax.device_get(op.x), dtype=np.float64)
            xw = np.asarray(jax.device_get(op.x_weights), dtype=np.float64)
            xparts = (
                ("particle_source_shape", np.asarray(jax.device_get(xpart1), dtype=np.float64)),
                ("energy_source_shape", np.asarray(jax.device_get(xpart2), dtype=np.float64)),
            )
            ix0 = _ix_min(bool(op.point_at_x0))
            for species in range(int(op.n_species)):
                for name, weights in xparts:
                    if len(names) >= max_cols_use:
                        break
                    _add_f_fsavg_column(
                        f"direct_pmat_constraint1_{name}_s{species}",
                        species=species,
                        ell=0,
                        x_weights=weights,
                        fs_pattern=fs_pattern,
                    )
                # The exact Fortran Pmat couples source amplitudes to the L=0
                # kinetic equations at each retained speed.  Per-speed columns
                # give the coarse equation a bounded representation of that
                # Schur complement instead of relying only on global shapes.
                for ix in range(ix0, int(op.n_x)):
                    if len(names) >= max_cols_use:
                        break
                    unit_x = np.zeros((int(op.n_x),), dtype=np.float64)
                    unit_x[ix] = 1.0
                    _add_f_fsavg_column(
                        f"direct_pmat_constraint1_l0_fsavg_s{species}_x{ix}",
                        species=species,
                        ell=0,
                        x_weights=unit_x,
                        fs_pattern=fs_pattern,
                    )
                moment_specs = [
                    ("density_moment", 0, (x**2) * xw),
                    ("pressure_moment", 0, (x**4) * xw),
                ]
                if int(op.n_xi) > 1:
                    moment_specs.extend(
                        [
                            ("flow_moment", 1, (x**3) * xw),
                            ("heat_flow_moment", 1, (x**5) * xw),
                        ]
                    )
                for name, ell, weights in moment_specs:
                    if len(names) >= max_cols_use:
                        break
                    _add_f_fsavg_column(
                        f"direct_pmat_constraint1_{name}_s{species}_l{int(ell)}",
                        species=species,
                        ell=int(ell),
                        x_weights=weights,
                        fs_pattern=fs_pattern,
                    )
        except Exception:
            pass
    elif int(op.constraint_scheme) == 2:
        x = np.asarray(jax.device_get(op.x), dtype=np.float64)
        xw = np.asarray(jax.device_get(op.x_weights), dtype=np.float64)
        moment_specs = [
            ("density", 0, (x**2) * xw),
            ("pressure", 0, (x**4) * xw),
            ("flow", min(1, int(op.n_xi) - 1), (x**3) * xw),
            ("heat_flow", min(1, int(op.n_xi) - 1), (x**5) * xw),
        ]
        for species in range(int(op.n_species)):
            ix0 = _ix_min(bool(op.point_at_x0))
            for ix in range(ix0, int(op.n_x)):
                if len(names) >= max_cols_use:
                    break
                unit_x = np.zeros((int(op.n_x),), dtype=np.float64)
                unit_x[ix] = 1.0
                _add_f_fsavg_column(
                    f"direct_pmat_constraint2_l0_source_s{species}_x{ix}",
                    species=species,
                    ell=0,
                    x_weights=unit_x,
                    fs_pattern=fs_pattern,
                )
            for name, ell, weights in moment_specs:
                if len(names) >= max_cols_use:
                    break
                _add_f_fsavg_column(
                    f"direct_pmat_constraint2_{name}_moment_s{species}_l{int(ell)}",
                    species=species,
                    ell=int(ell),
                    x_weights=weights,
                    fs_pattern=fs_pattern,
                )

    matrix = None
    if base_factor_bundle is not None:
        try:
            matrix = getattr(getattr(base_factor_bundle, "operator", None), "matrix", None)
        except Exception:
            matrix = None
    if matrix is not None and len(names) < max_cols_use:
        try:
            matrix_csr = matrix.tocsr() if sp.issparse(matrix) else sp.csr_matrix(np.asarray(matrix))
            tail_positions = np.asarray(
                [
                    int(full_to_active[full_idx])
                    for full_idx in range(tail0, n_total)
                    if int(full_to_active[full_idx]) >= 0
                ],
                dtype=np.int64,
            )
            for local_tail, tail_pos in enumerate(tail_positions):
                if len(names) >= max_cols_use:
                    break
                source_rhs = np.asarray(matrix_csr[:, int(tail_pos)].toarray()).reshape((n_active,))
                if tail_positions.size:
                    source_rhs[tail_positions] = 0.0
                try:
                    response = np.asarray(base_factor_bundle.solve(source_rhs), dtype=np.float64).reshape((n_active,))
                except Exception:
                    response = np.zeros((n_active,), dtype=np.float64)
                mode = -response
                mode[int(tail_pos)] += 1.0
                _add_active_column(f"direct_pmat_tail_schur_response_{int(local_tail)}", mode)
        except Exception:
            pass

    if not names:
        return None, ()
    rows = np.concatenate(col_rows)
    cols = np.concatenate(
        [np.full((int(row.size),), int(i), dtype=np.int64) for i, row in enumerate(col_rows)]
    )
    data = np.concatenate(col_data)
    basis = sp.coo_matrix((data, (rows, cols)), shape=(n_active, len(names)), dtype=np.float64).tocsr()
    basis.sum_duplicates()
    basis.eliminate_zeros()
    return basis, tuple(names)


def _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
    *,
    op_pc: V3FullSystemOperator,
    active_indices: np.ndarray | None,
    factor_dtype: np.dtype,
    pc_shift: float,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[SparseOperatorBundle, dict[str, object]] | None:
    """Build a reduced Fortran-style transport ``Pmat`` directly from terms.

    This avoids the pattern-color probe path for production FP transport
    preconditioners.  It is deliberately fail-closed and only handles non-Phi1
    RHSMode=2/3 systems whose active kinetic set preserves complete zeta
    blocks plus the complete source/constraint tail.
    """

    if int(op_pc.rhs_mode) not in {2, 3} or op_pc.fblock.fp is None:
        return None
    if bool(op_pc.include_phi1) or bool(op_pc.include_phi1_in_kinetic):
        return None
    if int(op_pc.constraint_scheme) not in {0, 1, 2}:
        return None

    try:
        import scipy.sparse as sp  # noqa: PLC0415
        from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415
    except Exception:
        return None

    build_timer = Timer()
    dtype_np = np.dtype(factor_dtype)
    total_size = int(op_pc.total_size)
    f_size = int(op_pc.f_size)
    phi1_size = int(op_pc.phi1_size)
    extra_start = f_size + phi1_size
    n_zeta = int(op_pc.n_zeta)
    n_theta = int(op_pc.n_theta)
    n_xi = int(op_pc.n_xi)
    n_x = int(op_pc.n_x)
    n_species = int(op_pc.n_species)

    if active_indices is None:
        active = np.arange(total_size, dtype=np.int64)
    else:
        active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if active.size == 0 or np.any(active < 0) or np.any(active >= total_size):
        return None
    if np.unique(active).size != active.size:
        return None

    f_active = active[active < f_size]
    tail_active = active[active >= f_size]
    expected_tail = np.arange(extra_start, total_size, dtype=np.int64)
    if phi1_size != 0 or not np.array_equal(tail_active, expected_tail):
        return None
    if f_active.size == 0 or int(f_active.size) % n_zeta != 0:
        return None

    f_blocks = f_active.reshape((-1, n_zeta))
    first = f_blocks[:, 0]
    if np.any(first % n_zeta != 0):
        return None
    expected_blocks = first[:, None] + np.arange(n_zeta, dtype=np.int64)[None, :]
    if not np.array_equal(f_blocks, expected_blocks):
        return None
    active_blocks = (first // n_zeta).astype(np.int64, copy=False)

    try:
        fblock_selection = select_structured_rhs1_fblock_operator(
            op_pc.fblock,
            include_identity_shift=True,
            require_complete=True,
        )
        if not bool(fblock_selection.selected):
            return None
        projected_fblock = fblock_selection.assembly.operator.project_block_indices(active_blocks)
        k_ff = projected_fblock.to_scipy_csr_matrix().astype(dtype_np, copy=False)
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat unavailable "
                f"({type(exc).__name__}: {exc})",
            )
        return None

    active_size = int(active.size)
    kinetic_size = int(f_active.size)
    tail_size = int(tail_active.size)
    full_to_active = np.full((total_size,), -1, dtype=np.int64)
    full_to_active[active] = np.arange(active_size, dtype=np.int64)

    def _f_index(species: int, ix: int, ell: int, theta: int, zeta: int) -> int:
        return int(((((species * n_x) + ix) * n_xi + ell) * n_theta + theta) * n_zeta + zeta)

    def _extra_index(offset: int) -> int:
        return int(extra_start + offset)

    def _active_position(full_index: int) -> int:
        if full_index < 0 or full_index >= total_size:
            return -1
        return int(full_to_active[int(full_index)])

    b_rows: list[int] = []
    b_cols: list[int] = []
    b_data: list[float] = []
    c_rows: list[int] = []
    c_cols: list[int] = []
    c_data: list[float] = []
    d_rows: list[int] = []
    d_cols: list[int] = []
    d_data: list[float] = []

    def _append_b(row_full: int, extra_offset: int, value: float) -> None:
        row = _active_position(row_full)
        col = _active_position(_extra_index(extra_offset))
        if row >= 0 and 0 <= col - kinetic_size < tail_size:
            b_rows.append(row)
            b_cols.append(col - kinetic_size)
            b_data.append(float(value))

    def _append_c(extra_offset: int, col_full: int, value: float) -> None:
        row = _active_position(_extra_index(extra_offset))
        col = _active_position(col_full)
        if col >= 0 and 0 <= row - kinetic_size < tail_size:
            c_rows.append(row - kinetic_size)
            c_cols.append(col)
            c_data.append(float(value))

    def _append_d(row_extra_offset: int, col_extra_offset: int, value: float) -> None:
        row = _active_position(_extra_index(row_extra_offset))
        col = _active_position(_extra_index(col_extra_offset))
        if 0 <= row - kinetic_size < tail_size and 0 <= col - kinetic_size < tail_size:
            d_rows.append(row - kinetic_size)
            d_cols.append(col - kinetic_size)
            d_data.append(float(value))

    ix0 = _ix_min(bool(op_pc.point_at_x0))
    factor = np.asarray(
        jax.device_get(_fs_average_factor(op_pc.theta_weights, op_pc.zeta_weights, op_pc.d_hat)),
        dtype=np.float64,
    )

    if int(op_pc.constraint_scheme) == 2:
        for species in range(n_species):
            for ix in range(ix0, n_x):
                extra_offset = species * n_x + ix
                for theta in range(n_theta):
                    for zeta in range(n_zeta):
                        _append_b(_f_index(species, ix, 0, theta, zeta), extra_offset, 1.0)
            for ix in range(n_x):
                extra_offset = species * n_x + ix
                if bool(op_pc.point_at_x0) and ix == 0:
                    _append_d(extra_offset, extra_offset, 1.0)
                    continue
                for theta in range(n_theta):
                    for zeta in range(n_zeta):
                        _append_c(extra_offset, _f_index(species, ix, 0, theta, zeta), factor[theta, zeta])
    elif int(op_pc.constraint_scheme) == 1:
        xpart1_j, xpart2_j = _source_basis_constraint_scheme_1(op_pc.x)
        xpart1 = np.asarray(jax.device_get(xpart1_j), dtype=np.float64)
        xpart2 = np.asarray(jax.device_get(xpart2_j), dtype=np.float64)
        x = np.asarray(jax.device_get(op_pc.x), dtype=np.float64)
        x_weights = np.asarray(jax.device_get(op_pc.x_weights), dtype=np.float64)
        w2 = (x * x) * x_weights
        w4 = (x * x * x * x) * x_weights
        for species in range(n_species):
            dens_offset = 2 * species
            pres_offset = dens_offset + 1
            for ix in range(ix0, n_x):
                for theta in range(n_theta):
                    for zeta in range(n_zeta):
                        row = _f_index(species, ix, 0, theta, zeta)
                        _append_b(row, dens_offset, xpart1[ix])
                        _append_b(row, pres_offset, xpart2[ix])
            for ix in range(n_x):
                for theta in range(n_theta):
                    for zeta in range(n_zeta):
                        col = _f_index(species, ix, 0, theta, zeta)
                        avg = factor[theta, zeta]
                        _append_c(dens_offset, col, w2[ix] * avg)
                        _append_c(pres_offset, col, w4[ix] * avg)

    def _coo(
        rows: list[int],
        cols: list[int],
        data: list[float],
        shape: tuple[int, int],
    ):
        if not data:
            return sp.csr_matrix(shape, dtype=dtype_np)
        matrix = sp.coo_matrix(
            (
                np.asarray(data, dtype=dtype_np),
                (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)),
            ),
            shape=shape,
            dtype=dtype_np,
        )
        matrix.sum_duplicates()
        return matrix.tocsr()

    if tail_size:
        b_mat = _coo(b_rows, b_cols, b_data, (kinetic_size, tail_size))
        c_mat = _coo(c_rows, c_cols, c_data, (tail_size, kinetic_size))
        d_mat = _coo(d_rows, d_cols, d_data, (tail_size, tail_size))
        matrix = sp.bmat([[k_ff, b_mat], [c_mat, d_mat]], format="csr", dtype=dtype_np)
    else:
        b_mat = sp.csr_matrix((kinetic_size, 0), dtype=dtype_np)
        c_mat = sp.csr_matrix((0, kinetic_size), dtype=dtype_np)
        d_mat = sp.csr_matrix((0, 0), dtype=dtype_np)
        matrix = k_ff.tocsr()
    if float(pc_shift) != 0.0:
        matrix = matrix + float(pc_shift) * sp.eye(active_size, format="csr", dtype=dtype_np)
    matrix.sum_duplicates()
    matrix.eliminate_zeros()

    decision = SparseDecision(
        storage_kind="csr",
        reason="direct term-level reduced Fortran Pmat emission",
        backend=jax.default_backend(),
        shape=(active_size, active_size),
        dense_nbytes=estimate_dense_nbytes((active_size, active_size), dtype_np),
        csr_nbytes_estimate=estimate_csr_nbytes((active_size, active_size), int(matrix.nnz), data_dtype=dtype_np),
        nnz_estimate=int(matrix.nnz),
        block_cols=None,
        drop_tol=0.0,
    )

    def _matvec(x_vec: np.ndarray) -> np.ndarray:
        return np.asarray(matrix @ np.asarray(x_vec, dtype=dtype_np).reshape((active_size,)), dtype=dtype_np)

    bundle = SparseOperatorBundle(
        matrix=matrix,
        operator=LinearOperator((active_size, active_size), matvec=_matvec, dtype=dtype_np),
        metadata=decision,
    )
    metadata = {
        "direct_pmat": True,
        "direct_pmat_reason": "term_level_reduced_fortran_pmat",
        "direct_pmat_build_s": float(build_timer.elapsed_s()),
        "direct_pmat_active_size": int(active_size),
        "direct_pmat_kinetic_size": int(kinetic_size),
        "direct_pmat_tail_size": int(tail_size),
        "direct_pmat_nnz": int(matrix.nnz),
        "direct_pmat_csr_nbytes_estimate": int(decision.csr_nbytes_estimate),
        "direct_pmat_kinetic_nnz": int(k_ff.nnz),
        "direct_pmat_source_nnz": int(b_mat.nnz),
        "direct_pmat_constraint_nnz": int(c_mat.nnz),
        "direct_pmat_tail_nnz": int(d_mat.nnz),
        "direct_pmat_included_terms": tuple(str(v) for v in fblock_selection.assembly.included_terms),
    }
    if emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat selected "
            f"active={active_size} nnz={int(matrix.nnz)} "
            f"csr_mb={float(decision.csr_nbytes_estimate) / 1.0e6:.3f} "
            f"build_s={float(metadata['direct_pmat_build_s']):.3f}",
        )
    return bundle, metadata


def _try_build_rhsmode23_fp_direct_active_operator_bundle(
    *,
    op: V3FullSystemOperator,
    active_indices: np.ndarray | None,
    factor_dtype: np.dtype,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[SparseOperatorBundle, dict[str, object]] | None:
    """Build the active true RHSMode=2/3 FP operator directly from terms.

    This is the exact-operator counterpart to the reduced Fortran ``Pmat``
    emitter above.  It targets non-differentiable production transport solves:
    the Krylov/factor path applies the same active operator used by the
    matrix-free residual gate, but avoids generic sparse pattern coloring.
    """

    result = _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op,
        active_indices=active_indices,
        factor_dtype=factor_dtype,
        pc_shift=0.0,
        emit=None,
    )
    if result is None:
        return None
    bundle, metadata = result
    decision = bundle.metadata
    bundle = SparseOperatorBundle(
        matrix=bundle.matrix,
        operator=bundle.operator,
        metadata=SparseDecision(
            storage_kind=decision.storage_kind,
            reason="direct term-level active true FP operator emission",
            backend=decision.backend,
            shape=decision.shape,
            dense_nbytes=decision.dense_nbytes,
            csr_nbytes_estimate=decision.csr_nbytes_estimate,
            nnz_estimate=decision.nnz_estimate,
            block_cols=decision.block_cols,
            drop_tol=decision.drop_tol,
        ),
    )
    metadata = dict(metadata)
    metadata.update(
        {
            "direct_true_operator": True,
            "direct_true_operator_reason": "term_level_active_fp_operator",
            "direct_true_operator_build_s": float(metadata.get("direct_pmat_build_s", 0.0)),
            "direct_true_operator_active_size": int(
                metadata.get(
                    "direct_pmat_active_size",
                    0 if bundle.matrix is None else int(bundle.matrix.shape[0]),
                )
            ),
            "direct_true_operator_nnz": int(metadata.get("direct_pmat_nnz", 0)),
            "direct_true_operator_csr_nbytes_estimate": int(metadata.get("direct_pmat_csr_nbytes_estimate", 0)),
        }
    )
    if emit is not None:
        emit(
            1,
            "solve_v3_transport_matrix_linear_gmres: direct active true FP operator selected "
            f"active={int(metadata['direct_true_operator_active_size'])} "
            f"nnz={int(metadata['direct_true_operator_nnz'])} "
            f"csr_mb={float(metadata['direct_true_operator_csr_nbytes_estimate']) / 1.0e6:.3f} "
            f"build_s={float(metadata['direct_true_operator_build_s']):.3f}",
        )
    return bundle, metadata
