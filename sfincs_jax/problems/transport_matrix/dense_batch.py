"""Batched dense RHSMode=2/3 transport solve helper.

Small transport systems are sometimes best handled by assembling one dense
operator and solving all transport drives as a block.  Keeping this path outside
``v3_driver.py`` makes the driver responsible only for policy and orchestration,
while this module owns the dense matrix assembly, active-DOF projection, streamed
diagnostics, and residual bookkeeping for the bounded dense branch.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solver import assemble_dense_matrix_from_matvec, dense_solve_from_matrix
from sfincs_jax.v3_system import _operator_signature_cached, apply_v3_full_system_operator_cached
from sfincs_jax.verbose import Timer


EmitFn = Callable[[int, str], None]


@dataclass
class TransportDenseBatchContext:
    """Mutable state needed to solve all transport RHSs through one dense matrix."""

    dense_backend_allowed: bool
    dense_use_mixed: bool
    use_active_dof_mode: bool
    active_size: int
    op0: Any
    op_matvec_by_index: Sequence[Any]
    rhs_by_index: Sequence[jnp.ndarray]
    which_rhs_values: Sequence[int]
    rhs_norms: MutableMapping[int, jnp.ndarray]
    residual_norms: MutableMapping[int, jnp.ndarray]
    solver_kinds_by_rhs: MutableMapping[int, str]
    solve_methods_by_rhs: MutableMapping[int, str]
    elapsed_s: np.ndarray
    state_vectors: MutableMapping[int, jnp.ndarray]
    store_state_vectors: bool
    stream_diagnostics: bool
    rhs3_krylov_flags: Callable[[int], tuple[bool, bool]]
    maybe_project_constraint_nullspace: Callable[..., jnp.ndarray]
    collect_transport_outputs: Callable[[int, jnp.ndarray], None] | None = None
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    emit: EmitFn | None = None

    @property
    def n_rhs(self) -> int:
        """Number of transport drives solved by this context."""
        return int(len(self.which_rhs_values))


def _dense_dtype(dtype_in: jnp.dtype, *, dense_use_mixed: bool) -> jnp.dtype:
    return jnp.float32 if dense_use_mixed else dtype_in


def _emit_rhs_residual(
    *,
    emit: EmitFn | None,
    which_rhs: int,
    residual_norm: float,
    rhs_norm: float,
    elapsed_s: float,
) -> None:
    if emit is None:
        return
    relative_residual = (
        float(residual_norm) / float(rhs_norm)
        if np.isfinite(float(rhs_norm)) and float(rhs_norm) > 0.0
        else float("nan")
    )
    emit(
        0,
        f"whichRHS={which_rhs}: residual_norm={float(residual_norm):.6e} "
        f"rhs_norm={float(rhs_norm):.6e} relative_residual={relative_residual:.6e} "
        f"elapsed_s={float(elapsed_s):.3f}",
    )


def solve_transport_dense_batch(
    *,
    context: TransportDenseBatchContext,
    op_probe_ref: Any,
    reason: str,
) -> bool:
    """Solve all transport RHS vectors using one dense matrix, if admissible.

    Returns ``True`` only when the dense batched branch was actually used.
    Operator variation across RHSs or requested special Krylov treatment for
    the E_parallel RHS leaves the caller on the incremental solve path.
    """
    if not context.dense_backend_allowed:
        return False
    requested_epar_krylov = any(
        (context.rhs3_krylov_flags(int(which_rhs))[0] or context.rhs3_krylov_flags(int(which_rhs))[1])
        for which_rhs in context.which_rhs_values
    )
    if requested_epar_krylov:
        return False
    sig_ref = _operator_signature_cached(op_probe_ref)
    for op_probe in context.op_matvec_by_index[1:]:
        if _operator_signature_cached(op_probe) != sig_ref:
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: dense batch disabled (matvec operator varies)",
                )
            return False
    if context.emit is not None:
        context.emit(1, "solve_v3_transport_matrix_linear_gmres: evaluateJacobian called (matrix-free)")
        context.emit(1, f"solve_v3_transport_matrix_linear_gmres: dense batched solve across all whichRHS ({reason})")

    timer = Timer()
    if context.use_active_dof_mode:
        _solve_active_dense_batch(context=context, op_probe_ref=op_probe_ref, timer=timer)
    else:
        _solve_full_dense_batch(context=context, op_probe_ref=op_probe_ref, timer=timer)
    return True


def _solve_active_dense_batch(
    *,
    context: TransportDenseBatchContext,
    op_probe_ref: Any,
    timer: Timer,
) -> None:
    assert context.reduce_full is not None
    assert context.expand_reduced is not None

    def matvec_reduced(x: jnp.ndarray) -> jnp.ndarray:
        y_full = apply_v3_full_system_operator_cached(op_probe_ref, context.expand_reduced(x))
        return context.reduce_full(y_full)

    dense_dtype = _dense_dtype(jnp.float64, dense_use_mixed=bool(context.dense_use_mixed))
    rhs_mat = jnp.stack([context.reduce_full(rhs) for rhs in context.rhs_by_index], axis=1)
    a_dense = assemble_dense_matrix_from_matvec(
        matvec=matvec_reduced,
        n=int(context.active_size),
        dtype=dense_dtype,
    )
    rhs_mat = jnp.asarray(rhs_mat, dtype=dense_dtype)
    x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)
    if context.dense_use_mixed:
        r_mat = rhs_mat - a_dense @ x_mat
        dx_mat, _ = dense_solve_from_matrix(a=a_dense, b=r_mat)
        x_mat = x_mat + dx_mat
    x_mat = jnp.asarray(x_mat, dtype=jnp.float64)
    res_mat = a_dense @ x_mat - rhs_mat
    res_norms = jnp.linalg.norm(res_mat, axis=0)

    for idx, which_rhs in enumerate(context.which_rhs_values):
        which_rhs_int = int(which_rhs)
        x_col = context.expand_reduced(x_mat[:, idx])
        rhs_vec = context.rhs_by_index[idx]
        x_col = context.maybe_project_constraint_nullspace(
            x_col,
            which_rhs=which_rhs_int,
            op_matvec=op_probe_ref,
            rhs_vec=rhs_vec,
        )
        _store_dense_batch_result(
            context=context,
            which_rhs=which_rhs_int,
            x_col=x_col,
            residual_norm=res_norms[idx],
            elapsed_each_s=float(timer.elapsed_s() / float(context.n_rhs)),
        )


def _solve_full_dense_batch(
    *,
    context: TransportDenseBatchContext,
    op_probe_ref: Any,
    timer: Timer,
) -> None:
    def matvec_full(x: jnp.ndarray) -> jnp.ndarray:
        return apply_v3_full_system_operator_cached(op_probe_ref, x)

    a_dense = assemble_dense_matrix_from_matvec(
        matvec=matvec_full,
        n=int(context.op0.total_size),
        dtype=_dense_dtype(jnp.float64, dense_use_mixed=bool(context.dense_use_mixed)),
    )
    rhs_mat = jnp.stack(context.rhs_by_index, axis=1)
    rhs_mat = jnp.asarray(rhs_mat, dtype=a_dense.dtype)
    x_mat, _ = dense_solve_from_matrix(a=a_dense, b=rhs_mat)
    if context.dense_use_mixed:
        r_mat = rhs_mat - a_dense @ x_mat
        dx_mat, _ = dense_solve_from_matrix(a=a_dense, b=r_mat)
        x_mat = x_mat + dx_mat
    x_mat = jnp.asarray(x_mat, dtype=jnp.float64)

    x_cols: list[jnp.ndarray] = []
    for idx, which_rhs in enumerate(context.which_rhs_values):
        x_col = context.maybe_project_constraint_nullspace(
            x_mat[:, idx],
            which_rhs=int(which_rhs),
            op_matvec=op_probe_ref,
            rhs_vec=context.rhs_by_index[idx],
        )
        x_cols.append(x_col)

    x_mat_projected = jnp.stack(x_cols, axis=1)
    res_mat = a_dense @ x_mat_projected - rhs_mat
    res_norms = jnp.linalg.norm(res_mat, axis=0)

    for idx, which_rhs in enumerate(context.which_rhs_values):
        _store_dense_batch_result(
            context=context,
            which_rhs=int(which_rhs),
            x_col=x_mat_projected[:, idx],
            residual_norm=res_norms[idx],
            elapsed_each_s=float(timer.elapsed_s() / float(context.n_rhs)),
        )


def _store_dense_batch_result(
    *,
    context: TransportDenseBatchContext,
    which_rhs: int,
    x_col: jnp.ndarray,
    residual_norm: jnp.ndarray,
    elapsed_each_s: float,
) -> None:
    if context.store_state_vectors:
        context.state_vectors[int(which_rhs)] = x_col
    if context.stream_diagnostics:
        if context.collect_transport_outputs is None:
            raise RuntimeError("dense batch streaming diagnostics requested without an output collector")
        context.collect_transport_outputs(int(which_rhs), x_col)
    context.residual_norms[int(which_rhs)] = residual_norm
    context.solver_kinds_by_rhs[int(which_rhs)] = "dense"
    context.solve_methods_by_rhs[int(which_rhs)] = "dense"
    context.elapsed_s[int(which_rhs) - 1] = float(elapsed_each_s)
    _emit_rhs_residual(
        emit=context.emit,
        which_rhs=int(which_rhs),
        residual_norm=float(residual_norm),
        rhs_norm=float(context.rhs_norms[int(which_rhs)]),
        elapsed_s=float(elapsed_each_s),
    )


__all__ = [
    "TransportDenseBatchContext",
    "solve_transport_dense_batch",
]
