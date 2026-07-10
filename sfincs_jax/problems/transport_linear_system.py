"""Transport linear-system setup, direct factors, and reduced-Pmat builders.

This owner consolidates the active-DOF setup, active block factors,
direct reduced-Pmat emission, direct block-Schur preconditioner, and
Fortran-reduced LU preconditioner used by RHSMode=2/3 transport solves.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping, Sequence
from dataclasses import dataclass
import os
import time
from typing import Any
import jax.numpy as jnp
import jax.scipy.linalg as jla
import numpy as np
from sfincs_jax.operators.profile_layout import build_rhs1_compressed_pitch_layout
from sfincs_jax.problems.transport_policies import (
    TransportActiveDOFDecision,
    TransportActiveDOFState,
    TransportDensePolicy,
    TransportInitialSolvePolicy,
    build_transport_active_dof_state,
    resolve_transport_active_dof_mode,
    resolve_transport_dense_policy,
    resolve_transport_initial_solve_policy,
    transport_host_gmres_accepts_preconditioned_residual,
)
from sfincs_jax.solvers.implicit import (
    linear_custom_solve,
    linear_custom_solve_with_residual,
)
from sfincs_jax.solvers.krylov import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    dense_solve_from_matrix,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve,
    gmres_solve_jit,
    gmres_solve_with_history_scipy,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
)
from sfincs_jax.operators.profile_system import (
    _operator_signature_cached,
    sharding_constraints,
)
from sfincs_jax.profiling import Timer
from sfincs_jax.operators.profile_system import apply_v3_full_system_operator_cached

__all__ = (
    "TransportLinearSolveCallbacks",
    "TransportLinearSolveContext",
    "TransportDenseBatchContext",
    "TransportActiveDenseSetup",
    "resolve_transport_active_dense_setup",
    "dense_preconditioner_for_matvec",
    "dense_solver_for_matvec",
    "solve_transport_linear",
    "solve_transport_linear_with_residual",
    "solve_transport_dense_batch",
    "transport_host_gmres_solve",
    "transport_restart_for_method",
    "transport_solver_kind",
)


@dataclass(frozen=True)
class TransportLinearSolveContext:
    """Routing state shared by transport linear solves."""

    rhs_mode: int
    size_hint: int
    use_implicit: bool
    use_solver_jit: bool
    distributed_axis: str | None


@dataclass(frozen=True)
class TransportLinearSolveCallbacks:
    """Bound solve callbacks used by the transport per-RHS loop."""

    context: TransportLinearSolveContext

    def solve(
        self,
        *,
        matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        precondition_side_val: str = "left",
    ):
        return solve_transport_linear(
            context=self.context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            preconditioner_val=preconditioner_val,
            precondition_side_val=precondition_side_val,
        )

    def solve_with_residual(
        self,
        *,
        matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
        b_vec: jnp.ndarray,
        x0_vec: jnp.ndarray | None,
        tol_val: float,
        atol_val: float,
        restart_val: int,
        maxiter_val: int | None,
        solve_method_val: str,
        preconditioner_val: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
        precondition_side_val: str = "left",
    ) -> tuple[GMRESSolveResult, jnp.ndarray]:
        return solve_transport_linear_with_residual(
            context=self.context,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            solve_method_val=solve_method_val,
            preconditioner_val=preconditioner_val,
            precondition_side_val=precondition_side_val,
        )


def transport_solver_kind(method: str, *, rhs_mode: int) -> tuple[str, str]:
    """Map transport solve-method tokens to a concrete Krylov solver."""
    method_l = str(method).strip().lower()
    if method_l in {"auto", "default"}:
        if int(rhs_mode) in {2, 3}:
            # Favor short-recurrence Krylov for transport; later retries can fall back to GMRES.
            return "bicgstab", "batched"
        return "bicgstab", "batched"
    if method_l in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab", "batched"
    return "gmres", method_l


def transport_restart_for_method(
    method: str,
    *,
    rhs_mode: int,
    gmres_restart: int,
    restart: int,
) -> int:
    """Return the restart budget relevant for a transport solve method."""
    solver_kind, _ = transport_solver_kind(method, rhs_mode=int(rhs_mode))
    return int(gmres_restart) if solver_kind == "gmres" else int(restart)


def solve_transport_linear(
    *,
    context: TransportLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    preconditioner_val: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    precondition_side_val: str = "left",
):
    """Solve a transport linear system without returning an explicit residual."""
    if context.use_implicit:
        solver_kind, gmres_method = transport_solver_kind(
            solve_method_val, rhs_mode=int(context.rhs_mode)
        )
        return linear_custom_solve(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_val,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precondition_side_val,
            size_hint=int(context.size_hint),
        )
    solver_fn = gmres_solve_jit if context.use_solver_jit else gmres_solve
    return solver_fn(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=preconditioner_val,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=solve_method_val,
        precondition_side=precondition_side_val,
    )


def solve_transport_linear_with_residual(
    *,
    context: TransportLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    preconditioner_val: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    precondition_side_val: str = "left",
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve a transport linear system and return the solver residual vector."""
    solver_kind, gmres_method = transport_solver_kind(
        solve_method_val, rhs_mode=int(context.rhs_mode)
    )
    if context.use_implicit:
        return linear_custom_solve_with_residual(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_val,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precondition_side_val,
            size_hint=int(context.size_hint),
        )
    if solver_kind == "bicgstab":
        if context.distributed_axis is not None:
            with sharding_constraints(True):
                return gmres_solve_with_residual_distributed(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=preconditioner_val,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    restart=restart_val,
                    maxiter=maxiter_val,
                    solve_method="bicgstab",
                    precondition_side=precondition_side_val,
                    axis_name=context.distributed_axis,
                )
        solver_fn = (
            bicgstab_solve_with_residual_jit
            if context.use_solver_jit
            else bicgstab_solve_with_residual
        )
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_val,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            maxiter=maxiter_val,
            precondition_side=precondition_side_val,
        )
    if context.distributed_axis is not None:
        with sharding_constraints(True):
            return gmres_solve_with_residual_distributed(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=preconditioner_val,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method,
                precondition_side=precondition_side_val,
                axis_name=context.distributed_axis,
            )
    solver_fn = (
        gmres_solve_with_residual_jit
        if context.use_solver_jit
        else gmres_solve_with_residual
    )
    return solver_fn(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=preconditioner_val,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=gmres_method,
        precondition_side=precondition_side_val,
    )


EmitFn = Callable[[int, str], None]


def dense_preconditioner_for_matvec(
    *,
    matvec_fn,
    n: int,
    dtype: jnp.dtype,
    cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
    key: tuple[Any, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build or reuse a dense-LU preconditioner for a matrix-free operator."""
    if key in cache:
        return cache[key]
    a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=int(n), dtype=dtype)
    a_dense = jnp.asarray(a_dense, dtype=dtype)
    lu, piv = jla.lu_factor(a_dense)

    def precond(v: jnp.ndarray) -> jnp.ndarray:
        return jla.lu_solve((lu, piv), v)

    cache[key] = precond
    return precond


def dense_solver_for_matvec(
    *,
    matvec_fn,
    n: int,
    dtype: jnp.dtype,
    cache: dict[tuple[object, int], Callable[[jnp.ndarray], jnp.ndarray]],
    key: tuple[Any, ...],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build or reuse a dense-LU direct solver for a matrix-free operator."""
    if key in cache:
        return cache[key]
    a_dense = assemble_dense_matrix_from_matvec(matvec=matvec_fn, n=int(n), dtype=dtype)
    a_dense = jnp.asarray(a_dense, dtype=dtype)
    lu, piv = jla.lu_factor(a_dense)

    def solve(v: jnp.ndarray) -> jnp.ndarray:
        return jla.lu_solve((lu, piv), v)

    cache[key] = solve
    return solve


def transport_host_gmres_solve(
    *,
    op: Any,
    matvec_fn,
    b_vec: jnp.ndarray,
    x0_vec: jnp.ndarray | None,
    preconditioner_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precondition_side_val: str,
    emit: Callable[[int, str], None] | None = None,
    which_rhs: int | None = None,
    progress_every: int = 10,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Run host SciPy GMRES and return a JAX result plus true residual vector."""
    side = str(precondition_side_val).strip().lower()
    b_norm = float(jnp.linalg.norm(b_vec))
    target_true = max(float(atol_val), float(tol_val) * b_norm)
    reported_residual_norm: float | None = None
    started = time.perf_counter()
    progress_stride = max(0, int(progress_every))

    def _progress(iteration: int, residual: float) -> None:
        if emit is None or progress_stride <= 0:
            return
        iteration_int = int(iteration)
        if iteration_int != 1 and iteration_int % progress_stride != 0:
            return
        rhs_label = "unknown" if which_rhs is None else str(int(which_rhs))
        emit(
            1,
            "transport host SciPy GMRES progress "
            f"whichRHS={rhs_label} iter={iteration_int} "
            f"reported_residual={float(residual):.6e} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
        )

    if preconditioner_fn is not None and side == "left":
        x_np, rn_true, rn_pc, _history = explicit_left_preconditioned_gmres_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            progress_callback=_progress,
        )
        rhs_pc_norm = float(jnp.linalg.norm(preconditioner_fn(b_vec)))
        target_pc = max(float(atol_val), float(tol_val) * rhs_pc_norm)
        if (
            np.isfinite(float(rn_pc))
            and float(rn_pc) <= float(target_pc)
            and transport_host_gmres_accepts_preconditioned_residual(
                op=op,
                true_residual_norm=float(rn_true),
                target_true=float(target_true),
            )
        ):
            # Mirror the PETSc-style transport lane, which may accept convergence
            # on the preconditioned KSP residual for singular/near-singular systems.
            reported_residual_norm = min(float(rn_true), float(target_true))
    else:
        x_np, rn_true, _history = gmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=preconditioner_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precondition_side_val,
            progress_callback=_progress,
        )
    x_jnp = jnp.asarray(x_np, dtype=jnp.float64)
    residual_vec = b_vec - matvec_fn(x_jnp)
    residual_norm = float(jnp.linalg.norm(residual_vec))
    if np.isfinite(float(rn_true)):
        residual_norm = min(residual_norm, float(rn_true))
    if reported_residual_norm is not None:
        residual_norm = min(residual_norm, float(reported_residual_norm))
    return (
        GMRESSolveResult(
            x=x_jnp,
            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        ),
        residual_vec,
    )


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


# --- Active-DOF and dense-path setup ---




def transport_active_dof_indices(op: Any) -> np.ndarray:
    """Return full-vector indices for active RHSMode=2/3 transport unknowns."""
    return build_rhs1_compressed_pitch_layout(op).active_full_indices.astype(np.int32, copy=False)


@dataclass(frozen=True)
class TransportActiveDenseSetup:
    """Resolved active-DOF and dense-solve setup plus ordered emit notes."""

    initial_policy: TransportInitialSolvePolicy
    active_dof_decision: TransportActiveDOFDecision
    active_dof_state: TransportActiveDOFState
    dense_policy: TransportDensePolicy
    initial_notes: tuple[tuple[int, str], ...]
    active_notes: tuple[tuple[int, str], ...]
    dense_notes: tuple[tuple[int, str], ...]
    low_memory_outputs: bool
    stream_diagnostics: bool
    store_state_vectors: bool
    solve_method_use: str
    force_krylov: bool
    force_dense: bool
    dense_fallback: bool
    dense_retry_max: int
    dense_mem_max_mb: float
    dense_mem_block: bool
    dense_use_mixed: bool
    dense_backend_allowed: bool
    gmres_restart: int
    maxiter: int | None
    use_active_dof_mode: bool
    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int
    dense_precond_enabled: bool


def resolve_transport_active_dense_setup(
    *,
    op: Any,
    rhs_mode: int,
    n_rhs: int,
    solve_method: str,
    restart: int,
    maxiter: int | None,
    backend: str,
    geometry_scheme: int,
    dense_accelerator_auto_allowed: bool,
    dense_backend_policy_allowed: bool,
    state_out_requested: bool,
    force_stream_diagnostics: bool | None,
    force_store_state: bool | None,
    subset_mode: bool,
    active_dof_indices: Callable[[Any], np.ndarray],
    active_dof_env: str | None = None,
) -> TransportActiveDenseSetup:
    """Resolve RHSMode=2/3 active-index compaction and dense-path policy."""
    initial = resolve_transport_initial_solve_policy(
        op=op,
        rhs_mode=int(rhs_mode),
        n_rhs=int(n_rhs),
        solve_method=str(solve_method),
        restart=int(restart),
        maxiter=maxiter,
        backend=str(backend),
        geometry_scheme=int(geometry_scheme),
        dense_accelerator_auto_allowed=bool(dense_accelerator_auto_allowed),
        dense_backend_policy_allowed=bool(dense_backend_policy_allowed),
        state_out_requested=bool(state_out_requested),
        force_stream_diagnostics=force_stream_diagnostics,
        force_store_state=force_store_state,
        subset_mode=bool(subset_mode),
    )

    active_env = (
        os.environ.get("SFINCS_JAX_TRANSPORT_ACTIVE_DOF", "").strip().lower()
        if active_dof_env is None
        else str(active_dof_env).strip().lower()
    )
    active_decision = resolve_transport_active_dof_mode(
        op=op,
        rhs_mode=int(rhs_mode),
        solve_method_use=str(initial.solve_method_use),
        solve_method=str(solve_method),
        active_dof_env=active_env,
    )
    active_state = build_transport_active_dof_state(
        op=op,
        use_active_dof_mode=bool(active_decision.use_active_dof_mode),
        active_dof_indices=active_dof_indices,
    )
    active_notes = _active_dof_notes(
        op=op,
        active_dof_decision=active_decision,
        active_size=int(active_state.active_size),
    )

    dense_policy = resolve_transport_dense_policy(
        rhs_mode=int(rhs_mode),
        n_rhs=int(n_rhs),
        total_size=int(op.total_size),
        active_size=int(active_state.active_size),
        solve_method_use=str(active_decision.solve_method_use),
        force_krylov=bool(initial.force_krylov),
        force_dense=bool(initial.force_dense),
        dense_fallback=bool(initial.dense_fallback),
        dense_retry_max=int(initial.dense_retry_max),
        dense_mem_max_mb=float(initial.dense_mem_max_mb),
        dense_mem_block=bool(initial.dense_mem_block),
        dense_use_mixed=bool(initial.dense_use_mixed),
        low_memory_outputs=bool(initial.low_memory_outputs),
        dense_backend_allowed=bool(initial.dense_backend_allowed),
        dense_precond_default=bool(not initial.dense_mem_block),
    )
    dense_notes = _dense_policy_notes(
        rhs_mode=int(rhs_mode),
        solve_method_before_dense=str(active_decision.solve_method_use),
        dense_policy=dense_policy,
        initial_policy=initial,
        active_size=int(active_state.active_size),
    )

    return TransportActiveDenseSetup(
        initial_policy=initial,
        active_dof_decision=active_decision,
        active_dof_state=active_state,
        dense_policy=dense_policy,
        initial_notes=tuple(initial.notes),
        active_notes=tuple(active_notes),
        dense_notes=tuple(dense_notes),
        low_memory_outputs=bool(initial.low_memory_outputs),
        stream_diagnostics=bool(initial.stream_diagnostics),
        store_state_vectors=bool(initial.store_state_vectors),
        solve_method_use=str(dense_policy.solve_method_use),
        force_krylov=bool(initial.force_krylov),
        force_dense=bool(dense_policy.force_dense),
        dense_fallback=bool(dense_policy.dense_fallback),
        dense_retry_max=int(dense_policy.dense_retry_max),
        dense_mem_max_mb=float(initial.dense_mem_max_mb),
        dense_mem_block=bool(dense_policy.dense_mem_block),
        dense_use_mixed=bool(dense_policy.dense_use_mixed),
        dense_backend_allowed=bool(initial.dense_backend_allowed),
        gmres_restart=int(initial.gmres_restart),
        maxiter=initial.maxiter,
        use_active_dof_mode=bool(active_decision.use_active_dof_mode),
        active_idx_np=active_state.active_idx_np,
        active_idx_jnp=active_state.active_idx_jnp,
        full_to_active_jnp=active_state.full_to_active_jnp,
        active_size=int(active_state.active_size),
        dense_precond_enabled=bool(dense_policy.dense_precond_enabled),
    )


def _active_dof_notes(
    *,
    op: Any,
    active_dof_decision: TransportActiveDOFDecision,
    active_size: int,
) -> tuple[tuple[int, str], ...]:
    notes: list[tuple[int, str]] = []
    if active_dof_decision.emit_disabled_hint:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: active-DOF mode disabled "
                "(set SFINCS_JAX_TRANSPORT_ACTIVE_DOF=1 to enable; "
                "SFINCS_JAX_TRANSPORT_ACTIVE_DOF=0 to force full-size solve)",
            )
        )
    if active_dof_decision.use_active_dof_mode:
        reason = f" ({active_dof_decision.reason})" if active_dof_decision.reason else ""
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: active-DOF mode enabled "
                f"(size={int(active_size)}/{int(op.total_size)}){reason}",
            )
        )
    return tuple(notes)


def _dense_policy_notes(
    *,
    rhs_mode: int,
    solve_method_before_dense: str,
    dense_policy: TransportDensePolicy,
    initial_policy: TransportInitialSolvePolicy,
    active_size: int,
) -> tuple[tuple[int, str], ...]:
    notes: list[tuple[int, str]] = []
    if dense_policy.dense_mem_block and not initial_policy.dense_mem_block:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
                f"(active_est_mem32={dense_policy.dense_mem_est_active_mb32:.1f} MB > "
                f"{initial_policy.dense_mem_max_mb:.1f} MB)",
            )
        )
    elif dense_policy.dense_use_mixed and not initial_policy.dense_use_mixed:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
                f"(active_est_mem64={dense_policy.dense_mem_est_active_mb64:.1f} MB > "
                f"{initial_policy.dense_mem_max_mb:.1f} MB)",
            )
        )
    if str(dense_policy.solve_method_use).lower() == "dense" and str(solve_method_before_dense).lower() != "dense":
        if int(rhs_mode) == 2 and not dense_policy.force_dense:
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
                    f"(n={int(active_size)})",
                )
            )
    if dense_policy.dense_precond_mem_block:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense preconditioner disabled "
                f"(est_mem={dense_policy.dense_precond_est_mb:.1f} MB > "
                f"{dense_policy.dense_precond_mem_max_mb:.1f} MB)",
            )
        )
    return tuple(notes)


# --- Active block factors and residual admission ---



