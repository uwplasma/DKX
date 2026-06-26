"""Structured RHSMode=1 full-CSR operator bundle construction.

This module wraps the analytic RHSMode=1 full-system CSR assembly in the
``SparseOperatorBundle`` contract used by sparse-PC solver paths. It is a
runtime/non-autodiff path for supported systems and intentionally returns
``None`` for unsupported cases so callers can fall back to matrix-free or
pattern-probed sparse assembly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import jax
import numpy as np

from sfincs_jax.solvers.explicit_sparse import SparseDecision, SparseOperatorBundle, estimate_csr_nbytes, estimate_dense_nbytes
from sfincs_jax.operators.profile_response.full_system import select_structured_rhs1_full_csr_operator
from sfincs_jax.problems.profile_response.policies import read_bool_env, read_int_env
from sfincs_jax.profiling import Timer

if TYPE_CHECKING:
    from sfincs_jax.v3_system import V3FullSystemOperator

__all__ = ["_try_build_structured_rhs1_full_csr_operator_bundle"]


def _try_build_structured_rhs1_full_csr_operator_bundle(
    *,
    op: V3FullSystemOperator,
    active_indices: np.ndarray | None,
    csr_max_mb: float,
    drop_tol: float,
    emit: Callable[[int, str], None] | None = None,
) -> SparseOperatorBundle | None:
    """Build a no-probe RHSMode=1 CSR operator for supported full systems.

    This is a runtime/non-autodiff path. It replaces expensive full-column or
    pattern-color probing with analytic f-block assembly plus analytic global
    constraint/Phi1 couplings. Unsupported cases return ``None`` so the caller
    can keep the established matrix-free/probed sparse path.
    """

    if int(op.rhs_mode) != 1:
        return None

    from scipy.sparse.linalg import LinearOperator  # noqa: PLC0415

    max_csr_nbytes = int(max(0.0, float(csr_max_mb)) * 1.0e6)
    if active_indices is not None:
        active = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
        projects_active_subset = active.size != int(op.total_size) or not np.array_equal(
            active,
            np.arange(int(op.total_size), dtype=np.int32),
        )
        if bool(projects_active_subset) and not read_bool_env(
            "SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_ALLOW_PROJECT_AFTER_BUILD",
            default=False,
        ):
            max_project_full_size = read_int_env(
                "SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_PROJECT_AFTER_BUILD_MAX_SIZE",
                default=200_000,
                minimum=1,
            )
            if int(op.total_size) > int(max_project_full_size):
                if emit is not None:
                    emit(
                        1,
                        "structured_full_csr: skipped full build before active projection "
                        f"full_size={int(op.total_size)} active_size={int(active.size)} "
                        f"max_full_size={int(max_project_full_size)}",
                    )
                return None
    # The direct-tail path exists to avoid the conservative product-pattern
    # estimate used by sparse probing.  That estimate can reject compact
    # term-separated RHSMode=1 matrices before assembly, so enforce the memory
    # budget below on the actual active CSR matrix instead.
    assembly_timer = Timer()
    if emit is not None:
        active_text = (
            "full"
            if active_indices is None
            else f"active_size={int(np.asarray(active_indices).size)}/{int(op.total_size)}"
        )
        emit(
            1,
            "structured_full_csr: assembly start "
            f"total_size={int(op.total_size)} {active_text} drop_tol={float(drop_tol):.3e}",
        )
    selected = select_structured_rhs1_full_csr_operator(
        op,
        drop_tol=float(drop_tol),
        max_csr_nbytes=None,
    )
    if not bool(selected.selected) or selected.matrix is None:
        if emit is not None:
            emit(
                1,
                "structured_full_csr: assembly not selected "
                f"elapsed_s={assembly_timer.elapsed_s():.3f} reason={selected.reason}",
            )
        return None

    matrix = selected.matrix.tocsr()
    if active_indices is not None:
        active = np.asarray(active_indices, dtype=np.int32).reshape((-1,))
        if active.size != int(op.total_size) or not np.array_equal(active, np.arange(int(op.total_size), dtype=np.int32)):
            matrix = matrix[active][:, active].tocsr()
    if float(drop_tol) > 0.0 and matrix.nnz:
        matrix = matrix.copy()
        matrix.data[np.abs(matrix.data) <= float(drop_tol)] = 0.0
        matrix.eliminate_zeros()

    actual_csr_nbytes = estimate_csr_nbytes(
        tuple(int(v) for v in matrix.shape),
        int(matrix.nnz),
        data_dtype=matrix.dtype,
        index_dtype=matrix.indices.dtype,
    )
    if max_csr_nbytes > 0 and int(actual_csr_nbytes) > int(max_csr_nbytes):
        if emit is not None:
            emit(
                1,
                "structured_full_csr: rejected actual CSR budget "
                f"shape={tuple(int(v) for v in matrix.shape)} nnz={int(matrix.nnz)} "
                f"elapsed_s={assembly_timer.elapsed_s():.3f} "
                f"csr_mb={float(actual_csr_nbytes) / 1.0e6:.3f} "
                f"max_mb={float(max_csr_nbytes) / 1.0e6:.3f}",
            )
        return None

    decision = SparseDecision(
        storage_kind="csr",
        reason="structured RHSMode=1 full CSR assembly (no matrix probing)",
        backend=jax.default_backend(),
        shape=tuple(int(v) for v in matrix.shape),
        dense_nbytes=estimate_dense_nbytes(tuple(int(v) for v in matrix.shape), matrix.dtype),
        csr_nbytes_estimate=int(actual_csr_nbytes),
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
        meta = selected.metadata
        emit(
            1,
            "structured_full_csr: assembly complete "
            f"elapsed_s={assembly_timer.elapsed_s():.3f} "
            f"shape={tuple(int(v) for v in matrix.shape)} nnz={int(matrix.nnz)}",
        )
        emit(
            1,
            "structured_full_csr: selected "
            f"shape={tuple(int(v) for v in matrix.shape)} nnz={int(matrix.nnz)} "
            f"csr_mb={float(decision.csr_nbytes_estimate) / 1.0e6:.3f} "
            f"tail_nnz={int(meta.get('tail_nnz', 0) or 0)} "
            f"fblock_mb={float(meta.get('fblock_csr_nbytes_actual', 0) or 0) / 1.0e6:.3f}",
        )
    return SparseOperatorBundle(matrix=matrix, operator=operator, metadata=decision)
