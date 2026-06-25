"""Fortran-reduced RHSMode=1 direct-tail operator assembly.

The direct-tail materializer replaces expensive color probing of dense
constraint rows with term-level source/moment formulas matching the v3
matrix-free operator. It is kept outside ``v3_driver`` so solver orchestration
can depend on a focused assembly helper instead of owning the sparse block
construction details.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.explicit_sparse import (
    SparseDecision,
    SparseOperatorBundle,
    build_operator_from_pattern,
    estimate_csr_nbytes,
    estimate_dense_nbytes,
)
from sfincs_jax.operators.profile_response.layout import RHS1ActiveBlockLayout, RHS1BlockLayout
from sfincs_jax.operators.profile_response.sources import constraint_scheme1_inject_source as _constraint_scheme1_inject_source
from sfincs_jax.operators.profile_response.kinetic import select_structured_rhs1_fblock_operator
from sfincs_jax.problems.profile_response.solver_policy import read_bool_env, read_int_env
from sfincs_jax.v3_system import _fs_average_factor, apply_v3_full_system_operator_cached

if TYPE_CHECKING:
    from sfincs_jax.v3_system import V3FullSystemOperator

__all__ = ["_try_build_fortran_reduced_constraint1_direct_tail_bundle"]


def _try_build_fortran_reduced_constraint1_direct_tail_bundle(
    *,
    op: V3FullSystemOperator,
    op_pc: V3FullSystemOperator,
    pattern,
    active_indices: np.ndarray | None,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray],
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray],
    pc_shift: float,
    dtype: jnp.dtype,
    factor_dtype: np.dtype,
    csr_max_mb: float,
    drop_tol: float,
    color_batch: int,
    emit: Callable[[int, str], None] | None = None,
    build_structured_rhs1_full_csr_operator_bundle_callback: Callable[..., SparseOperatorBundle | None],
) -> SparseOperatorBundle | None:
    """Materialize RHSMode=1 constraintScheme=1 tails without probing dense rows.

    Constraint rows couple to every active ``L=0`` angular point, so including
    them in the coloring pattern forces thousands of probe colors. The kinetic
    block is still probed, but the source columns and moment rows are inserted
    from the same formulas used by the matrix-free operator.
    """

    if int(op.rhs_mode) != 1 or int(op.constraint_scheme) != 1 or int(op.phi1_size) != 0:
        return None
    extra_size = int(op.extra_size)
    if extra_size != 2 * int(op.n_species) or extra_size <= 0:
        return None

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    pattern_csr = pattern.tocsr()
    n_reduced = int(pattern_csr.shape[0])
    if pattern_csr.shape[0] != pattern_csr.shape[1] or n_reduced <= extra_size:
        return None
    layout = RHS1BlockLayout.from_operator(op)
    if active_indices is None:
        active_idx_np = np.arange(int(layout.total_size), dtype=np.int32)
    else:
        active_idx_np = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
    if int(active_idx_np.size) != n_reduced:
        return None
    active_layout = RHS1ActiveBlockLayout.from_layout(layout, active_idx_np)
    if (
        int(active_layout.extra_count) != int(extra_size)
        or int(active_layout.phi1_count) != 0
        or not bool(active_layout.has_contiguous_extra_tail)
    ):
        return None
    f_count = int(active_layout.kinetic_count)
    f_active = np.asarray(active_layout.active_kinetic_indices(), dtype=np.int64)
    if np.any(f_active >= int(op.f_size)):
        return None
    color_batch_requested = max(1, int(color_batch))
    assembly_mode = (
        os.environ.get("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "auto")
        .strip()
        .lower()
        .replace("-", "_")
    )
    if assembly_mode == "":
        assembly_mode = "auto"
    whichmatrix0_modes = {
        "whichmatrix0",
        "which_matrix0",
        "direct_whichmatrix0",
        "active_whichmatrix0",
        "term",
        "term_level",
        "term_level_whichmatrix0",
        "fortran_v3",
        "fortran_v3_whichmatrix0",
    }
    pattern_only_modes = {"pattern", "probe", "pattern_probe", "color_probe"}

    structured_first_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR",
        "1",
    ).strip().lower()
    if structured_first_env not in {"0", "false", "no", "off"} and assembly_mode not in (
        whichmatrix0_modes | pattern_only_modes
    ):
        structured_bundle = build_structured_rhs1_full_csr_operator_bundle_callback(
            op=op_pc,
            active_indices=active_idx_np,
            csr_max_mb=float(csr_max_mb),
            drop_tol=float(drop_tol),
            emit=emit,
        )
        if structured_bundle is not None and structured_bundle.matrix is not None:
            matrix = structured_bundle.matrix.tocsr().astype(np.dtype(factor_dtype), copy=False)
            if float(pc_shift) != 0.0:
                matrix = (
                    matrix
                    + sp.eye(int(matrix.shape[0]), format="csr", dtype=np.dtype(factor_dtype))
                    * np.asarray(float(pc_shift), dtype=np.dtype(factor_dtype))
                ).tocsr()
                matrix.sum_duplicates()
                matrix.eliminate_zeros()
            decision = SparseDecision(
                storage_kind="csr",
                reason=(
                    "fortran-reduced constraintScheme=1 structured direct-tail CSR "
                    "(term-separated f-block; no kinetic probing)"
                ),
                backend=jax.default_backend(),
                shape=tuple(int(v) for v in matrix.shape),
                dense_nbytes=estimate_dense_nbytes(tuple(int(v) for v in matrix.shape), matrix.dtype),
                csr_nbytes_estimate=estimate_csr_nbytes(
                    tuple(int(v) for v in matrix.shape),
                    int(matrix.nnz),
                    data_dtype=matrix.dtype,
                    index_dtype=matrix.indices.dtype,
                ),
                nnz_estimate=int(matrix.nnz),
                block_cols=0,
                drop_tol=float(drop_tol),
            )
            operator = LinearOperator(
                matrix.shape,
                matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=matrix.dtype)),
                dtype=matrix.dtype,
            )
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    f"structured csr built nnz={int(matrix.nnz)} "
                    f"csr_mb={float(decision.csr_nbytes_estimate) / 1.0e6:.3f}",
            )
            return SparseOperatorBundle(matrix=matrix, operator=operator, metadata=decision)

    k_ff = None
    kinetic_block_cols = 0
    kinetic_reason = ""
    active_term_auto_max_size = read_int_env(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_WHICHMATRIX0_ACTIVE_TERM_AUTO_MAX_SIZE",
        default=100_000,
        minimum=1,
    )
    active_term_allowed = bool(
        assembly_mode in whichmatrix0_modes
        or (
            assembly_mode == "auto"
            and active_indices is not None
            and int(n_reduced) <= int(active_term_auto_max_size)
        )
    )
    if bool(active_term_allowed) and assembly_mode not in pattern_only_modes:
        fblock_max_mb_env = os.environ.get(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_WHICHMATRIX0_FBLOCK_MAX_MB",
            "",
        ).strip()
        try:
            fblock_max_mb = float(fblock_max_mb_env) if fblock_max_mb_env else 0.0
        except ValueError:
            fblock_max_mb = 0.0
        fblock_max_nbytes = None
        if float(fblock_max_mb) > 0.0:
            fblock_max_nbytes = int(float(fblock_max_mb) * 1.0e6)
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                "whichMatrix=0 active term assembly start "
                f"f_active={int(f_count)} full_f={int(op.f_size)} "
                f"max_fblock_mb={'none' if fblock_max_nbytes is None else f'{float(fblock_max_mb):.3g}'}",
            )
        fblock_selection = select_structured_rhs1_fblock_operator(
            op_pc.fblock,
            include_identity_shift=True,
            phi1_hat_base=getattr(op_pc, "phi1_hat_base", None),
            drop_tol=float(drop_tol),
            require_complete=True,
        )
        if bool(fblock_selection.selected):
            fblock_operator = fblock_selection.assembly.operator
            fblock_block_size = int(fblock_operator.block_size)
            active_blocks: np.ndarray | None = None
            if fblock_block_size > 0 and f_count % fblock_block_size == 0:
                f_active_blocks = f_active.reshape((-1, fblock_block_size))
                first_indices = f_active_blocks[:, 0]
                expected = first_indices[:, None] + np.arange(fblock_block_size, dtype=np.int64)[None, :]
                if np.array_equal(f_active_blocks, expected) and np.all(first_indices % fblock_block_size == 0):
                    active_blocks = (first_indices // fblock_block_size).astype(np.int64, copy=False)
            if active_blocks is None:
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                        "whichMatrix=0 active term assembly not selected "
                        "reason=active_indices_do_not_form_complete_fblock_blocks",
                    )
            else:
                projected_operator = fblock_operator.project_block_indices(active_blocks)
                projected_nnz_bound = int(projected_operator.nnz_blocks) * fblock_block_size * fblock_block_size
                projected_csr_bound = estimate_csr_nbytes(
                    tuple(int(v) for v in projected_operator.shape),
                    int(projected_nnz_bound),
                    data_dtype=np.dtype(factor_dtype),
                )
                if fblock_max_nbytes is not None and int(projected_csr_bound) > int(fblock_max_nbytes):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                            "whichMatrix=0 active term assembly not selected "
                            f"reason=projected_csr_budget_exceeded:{int(projected_csr_bound)}>{int(fblock_max_nbytes)}",
                        )
                else:
                    k_ff = projected_operator.to_scipy_csr_matrix().astype(np.dtype(factor_dtype), copy=False)
                    if float(pc_shift) != 0.0:
                        k_ff = (
                            k_ff
                            + sp.eye(int(k_ff.shape[0]), format="csr", dtype=np.dtype(factor_dtype))
                            * np.asarray(float(pc_shift), dtype=np.dtype(factor_dtype))
                        ).tocsr()
                        k_ff.sum_duplicates()
                        k_ff.eliminate_zeros()
                    kinetic_reason = (
                        "fortran-reduced constraintScheme=1 whichMatrix=0 active term-level "
                        "direct-tail CSR (active block-projected structured f-block; no kinetic probing)"
                    )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                            "whichMatrix=0 active term CSR built "
                            f"kinetic_nnz={int(k_ff.nnz)} active_blocks={int(active_blocks.size)} "
                            f"projected_block_nnz={int(projected_operator.nnz_blocks)}",
                        )
        elif emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                "whichMatrix=0 active term assembly not selected "
                f"reason={fblock_selection.reason}",
            )

    if k_ff is None:
        pattern_ff = pattern_csr[:f_count, :f_count].tocsr()
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail materialization "
                f"kinetic_pattern_nnz={int(pattern_ff.nnz)} f_size={int(f_count)} extra={int(extra_size)}",
            )

        def _matvec_ff(x_np: np.ndarray) -> np.ndarray:
            x_red = np.zeros((n_reduced,), dtype=np.float64)
            x_red[:f_count] = np.asarray(x_np, dtype=np.float64).reshape((f_count,))
            x_full = expand_reduced(jnp.asarray(x_red, dtype=dtype))
            y_full = apply_v3_full_system_operator_cached(op_pc, x_full)
            if float(pc_shift) != 0.0:
                y_full = y_full + jnp.asarray(float(pc_shift), dtype=dtype) * x_full
            y_red = reduce_full(y_full)
            return np.asarray(y_red[:f_count], dtype=np.float64)

        def _matmat_ff(x_np: np.ndarray) -> np.ndarray:
            seeds = np.asarray(x_np, dtype=np.float64)
            if seeds.ndim != 2 or seeds.shape[0] != f_count:
                raise ValueError(f"matmat seed shape {seeds.shape}; expected ({f_count}, n_batch)")
            batch = int(seeds.shape[1])
            if batch == 0:
                return np.zeros((f_count, 0), dtype=np.float64)
            try:
                x_red = jnp.zeros((n_reduced, batch), dtype=dtype)
                x_red = x_red.at[:f_count, :].set(jnp.asarray(seeds, dtype=dtype))

                def _mv_col(col_red: jnp.ndarray) -> jnp.ndarray:
                    x_full = expand_reduced(col_red)
                    y_full = apply_v3_full_system_operator_cached(op_pc, x_full)
                    if float(pc_shift) != 0.0:
                        y_full = y_full + jnp.asarray(float(pc_shift), dtype=dtype) * x_full
                    return reduce_full(y_full)[:f_count]

                return np.asarray(jax.device_get(jax.vmap(_mv_col, in_axes=1, out_axes=1)(x_red)), dtype=np.float64)
            except Exception:  # noqa: BLE001
                return np.column_stack([_matvec_ff(seeds[:, j]) for j in range(batch)])

        pattern_matmat_enabled = read_bool_env(
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PATTERN_MATMAT",
            default=bool(assembly_mode in whichmatrix0_modes and color_batch_requested > 1),
        )
        ff_bundle = build_operator_from_pattern(
            _matvec_ff,
            pattern=pattern_ff,
            dtype=np.dtype(factor_dtype),
            backend=jax.default_backend(),
            csr_max_mb=float(csr_max_mb),
            drop_tol=float(drop_tol),
            allow_operator_only=False,
            color_batch=int(color_batch_requested) if bool(pattern_matmat_enabled) else 1,
            matmat=_matmat_ff if bool(pattern_matmat_enabled) else None,
            progress_callback=(
                None
                if emit is None
                else lambda message: emit(1, f"explicit_sparse direct-tail kinetic: {message}")
            ),
        )
        if ff_bundle.matrix is None:
            return None
        k_ff = ff_bundle.matrix.tocsr()
        kinetic_block_cols = int(ff_bundle.metadata.block_cols)
        kinetic_reason = (
            "fortran-reduced constraintScheme=1 direct-tail materialization "
            f"({kinetic_block_cols} kinetic colors; requested_color_batch={color_batch_requested}; "
            f"batched_matmat={bool(pattern_matmat_enabled)})"
        )

    b_rows: list[np.ndarray] = []
    b_cols: list[np.ndarray] = []
    b_data: list[np.ndarray] = []
    for j in range(extra_size):
        src = np.zeros((int(op.n_species), 2), dtype=np.float64)
        src.reshape((-1,))[j] = 1.0
        f_src = np.asarray(
            _constraint_scheme1_inject_source(op, jnp.asarray(src, dtype=jnp.float64)),
            dtype=np.float64,
        ).reshape((-1,))
        vals = f_src[f_active]
        keep = np.abs(vals) > max(float(drop_tol), 0.0) if float(drop_tol) > 0.0 else vals != 0.0
        if np.any(keep):
            row_idx = np.flatnonzero(keep).astype(np.int32)
            b_rows.append(row_idx)
            b_cols.append(np.full((int(row_idx.size),), int(j), dtype=np.int32))
            b_data.append(np.asarray(vals[keep], dtype=np.dtype(factor_dtype)))
    if b_data:
        b_mat = sp.coo_matrix(
            (np.concatenate(b_data), (np.concatenate(b_rows), np.concatenate(b_cols))),
            shape=(f_count, extra_size),
            dtype=np.dtype(factor_dtype),
        ).tocsr()
    else:
        b_mat = sp.csr_matrix((f_count, extra_size), dtype=np.dtype(factor_dtype))

    kinetic_indices = layout.decode_kinetic_indices(f_active)
    zeta = kinetic_indices.zeta
    theta = kinetic_indices.theta
    ell = kinetic_indices.ell
    ix = kinetic_indices.x
    species = kinetic_indices.species
    l0 = ell == 0
    fs_factor = np.asarray(_fs_average_factor(op.theta_weights, op.zeta_weights, op.d_hat), dtype=np.float64)
    x_np = np.asarray(op.x, dtype=np.float64)
    xw_np = np.asarray(op.x_weights, dtype=np.float64)
    w2 = x_np * x_np * xw_np
    w4 = x_np * x_np * x_np * x_np * xw_np
    c_rows: list[np.ndarray] = []
    c_cols: list[np.ndarray] = []
    c_data: list[np.ndarray] = []
    reduced_cols = np.arange(f_count, dtype=np.int32)
    for s in range(int(op.n_species)):
        mask = l0 & (species == int(s))
        if not np.any(mask):
            continue
        cols_use = reduced_cols[mask]
        vals_density = w2[ix[mask]] * fs_factor[theta[mask], zeta[mask]]
        vals_pressure = w4[ix[mask]] * fs_factor[theta[mask], zeta[mask]]
        c_rows.append(np.full((int(cols_use.size),), 2 * int(s), dtype=np.int32))
        c_cols.append(cols_use)
        c_data.append(np.asarray(vals_density, dtype=np.dtype(factor_dtype)))
        c_rows.append(np.full((int(cols_use.size),), 2 * int(s) + 1, dtype=np.int32))
        c_cols.append(cols_use)
        c_data.append(np.asarray(vals_pressure, dtype=np.dtype(factor_dtype)))
    if c_data:
        c_mat = sp.coo_matrix(
            (np.concatenate(c_data), (np.concatenate(c_rows), np.concatenate(c_cols))),
            shape=(extra_size, f_count),
            dtype=np.dtype(factor_dtype),
        ).tocsr()
    else:
        c_mat = sp.csr_matrix((extra_size, f_count), dtype=np.dtype(factor_dtype))

    if float(pc_shift) != 0.0:
        d_mat = sp.eye(extra_size, format="csr", dtype=np.dtype(factor_dtype)) * np.asarray(
            float(pc_shift),
            dtype=np.dtype(factor_dtype),
        )
    else:
        d_mat = sp.csr_matrix((extra_size, extra_size), dtype=np.dtype(factor_dtype))

    matrix = sp.bmat([[k_ff, b_mat], [c_mat, d_mat]], format="csr", dtype=np.dtype(factor_dtype))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    decision = SparseDecision(
        storage_kind="csr",
        reason=kinetic_reason,
        backend=jax.default_backend(),
        shape=(n_reduced, n_reduced),
        dense_nbytes=estimate_dense_nbytes((n_reduced, n_reduced), np.dtype(factor_dtype)),
        csr_nbytes_estimate=estimate_csr_nbytes(
            (n_reduced, n_reduced),
            int(matrix.nnz),
            data_dtype=np.dtype(factor_dtype),
        ),
        nnz_estimate=int(matrix.nnz),
        block_cols=int(kinetic_block_cols),
        drop_tol=float(drop_tol),
    )
    operator = LinearOperator(
        matrix.shape,
        matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.dtype(factor_dtype))),
        dtype=np.dtype(factor_dtype),
    )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail materialization "
            f"csr built nnz={int(matrix.nnz)} kinetic_nnz={int(k_ff.nnz)} "
            f"source_nnz={int(b_mat.nnz)} moment_nnz={int(c_mat.nnz)}",
        )
    return SparseOperatorBundle(matrix=matrix, operator=operator, metadata=decision)


