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
import jax
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
from sfincs_jax.solvers.explicit_sparse import SparseDecision, SparseOperatorBundle, estimate_csr_nbytes, estimate_dense_nbytes
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
from sfincs_jax.operators.profile_kinetic import select_structured_rhs1_fblock_operator
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator,
    _fs_average_factor,
    _ix_min,
    _operator_signature_cached,
    _source_basis_constraint_scheme_1,
    sharding_constraints,
)
from sfincs_jax.profiling import Timer
from sfincs_jax.solvers.preconditioning import (
    _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE,
    _TransportFpDirectActiveBlockSchurPrecondCache,
)
from sfincs_jax.problems.profile_policies import _hash_numpy_array_for_cache
from sfincs_jax.solvers.explicit_sparse import (
    admit_sparse_factor_against_operator,
    analyze_sparse_symbolic_structure,
    estimate_multifrontal_direct_lu_nbytes,
    factorize_host_sparse_operator,
    wrap_sparse_factor_with_coarse_correction,
)
from sfincs_jax.solvers.preconditioning import (
    _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE,
    _TransportFpFortranReducedLuPrecondCache,
)
from sfincs_jax.solvers.preconditioning import _build_transport_preconditioner_operator_fortran_reduced
from sfincs_jax.operators.profile_sparse_pattern import (
    summarize_v3_sparse_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices,
)
from sfincs_jax.operators.profile_system import apply_v3_full_system_operator_cached

__all__ = (
    "TransportLinearSolveCallbacks",
    "TransportLinearSolveContext",
    "TransportDenseBatchContext",
    "TransportActiveDenseSetup",
    "resolve_transport_active_dense_setup",
    "ActiveBlockAdmission",
    "ActiveBlockOrdering",
    "ActiveBlockSchurFactor",
    "ActiveBlockSchurResidualCoarseFactor",
    "admit_active_block_schur_factor",
    "build_active_block_ordering",
    "build_active_block_schur_factor",
    "build_active_block_schur_residual_coarse_factor",
    "deterministic_probe_matrix",
    "_build_rhsmode23_direct_pmat_physics_coarse_basis",
    "_try_build_rhsmode23_fp_direct_active_operator_bundle",
    "_try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle",
    "build_transport_fp_direct_active_block_schur_preconditioner",
    "build_transport_fp_fortran_reduced_lu_preconditioner",
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



@dataclass(frozen=True)
class ActiveBlockOrdering:
    """Symbolic block layout for an active transport operator.

    The active operator is assumed to use the SFINCS-JAX active ordering:
    kinetic unknowns first, then the source/constraint tail.  ``blocks`` stores
    kinetic reduced indices only; tail indices are handled by the Schur layer.
    """

    blocks: tuple[np.ndarray, ...]
    block_kind: str
    kinetic_size: int
    tail_size: int
    active_size: int
    block_size_max: int
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ActiveBlockSchurFactor:
    """Numerical block inverse plus optional exact tail Schur closure."""

    ordering: ActiveBlockOrdering
    block_inverse: tuple[np.ndarray, ...]
    c_tail: Any | None
    mb_tail: np.ndarray | None
    schur_inverse: np.ndarray | None
    dtype: np.dtype
    metadata: dict[str, Any]

    def apply(self, rhs: np.ndarray) -> np.ndarray:
        """Apply the block-Schur inverse approximation to one reduced RHS."""

        dtype = np.dtype(self.dtype)
        ordering = self.ordering
        active_size = int(ordering.active_size)
        kinetic_size = int(ordering.kinetic_size)
        tail_size = int(ordering.tail_size)
        rhs_np = np.asarray(rhs, dtype=dtype).reshape((active_size,))
        y_k = np.zeros((kinetic_size,), dtype=dtype)
        for indices, inverse in zip(ordering.blocks, self.block_inverse, strict=True):
            idx = np.asarray(indices, dtype=np.int64)
            y_k[idx] = np.asarray(inverse @ rhs_np[idx], dtype=dtype)
        if tail_size <= 0 or self.c_tail is None or self.mb_tail is None or self.schur_inverse is None:
            return np.concatenate([y_k, rhs_np[kinetic_size:]], axis=0).astype(np.float64, copy=False)
        rhs_t = rhs_np[kinetic_size:]
        tail_residual = np.asarray(rhs_t - self.c_tail @ y_k, dtype=dtype).reshape((tail_size,))
        y_t = np.asarray(self.schur_inverse @ tail_residual, dtype=dtype).reshape((tail_size,))
        y_k = np.asarray(y_k - self.mb_tail @ y_t, dtype=dtype).reshape((kinetic_size,))
        out = np.concatenate([y_k, y_t], axis=0)
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class ActiveBlockSchurResidualCoarseFactor:
    """Block-Schur factor plus a true-operator residual coarse correction.

    The base block factor intentionally ignores most off-block kinetic
    couplings.  This wrapper adds a bounded least-squares correction in a small
    solution subspace derived from setup residual probes:

    ``y = M0 r + Z (A Z)^+ (r - A M0 r)``.

    ``Z`` and ``A Z`` are precomputed, so each application uses one true sparse
    residual and a small dense normal-equation solve.  Admission still uses the
    true operator before this factor can be promoted by the caller.
    """

    base: ActiveBlockSchurFactor
    matrix: Any
    coarse_basis: np.ndarray
    action_basis: np.ndarray
    normal_inverse: np.ndarray
    damping: float
    ordering: ActiveBlockOrdering
    dtype: np.dtype
    metadata: dict[str, Any]

    def apply(self, rhs: np.ndarray) -> np.ndarray:
        """Apply one base solve plus the residual-derived coarse correction."""

        dtype = np.dtype(self.dtype)
        rhs_np = np.asarray(rhs, dtype=dtype).reshape((int(self.ordering.active_size),))
        y0 = np.asarray(self.base.apply(rhs_np), dtype=dtype).reshape(rhs_np.shape)
        residual = np.asarray(rhs_np - self.matrix @ y0, dtype=dtype)
        alpha_rhs = np.asarray(self.action_basis.T @ residual, dtype=dtype)
        alpha = np.asarray(self.normal_inverse @ alpha_rhs, dtype=dtype)
        out = np.asarray(y0 + float(self.damping) * (self.coarse_basis @ alpha), dtype=dtype)
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class ActiveBlockAdmission:
    """Result of setup-time residual admission."""

    accepted: bool
    max_relative_residual: float
    median_relative_residual: float
    min_improvement_vs_identity: float
    probe_count: int
    reason: str


def build_active_block_ordering(
    *,
    kinetic_size: int,
    tail_size: int,
    n_theta: int,
    n_zeta: int,
    block_kind: str = "zeta_line",
    ell_block: int = 1,
    max_block_size: int = 4096,
) -> ActiveBlockOrdering:
    """Build a reusable symbolic ordering over active kinetic unknowns.

    Parameters
    ----------
    kinetic_size:
        Number of retained kinetic unknowns in the active reduced system.
    tail_size:
        Number of retained source/constraint unknowns.
    n_theta, n_zeta:
        Angular grid shape in SFINCS storage order ``(..., theta, zeta)``.
    block_kind:
        ``"zeta_line"`` keeps contiguous zeta lines, ``"theta_line"`` keeps
        one theta line at fixed zeta inside each angular plane, and
        ``"angular_plane"``/``"ell_band"`` keeps one or more complete
        ``(theta,zeta)`` planes.
    ell_block:
        Number of complete angular planes per block for ``"ell_band"``.
    max_block_size:
        Hard memory/speed safety cap.  Oversized symbolic blocks are rejected.
    """

    kinetic_size = int(kinetic_size)
    tail_size = int(tail_size)
    n_theta = int(n_theta)
    n_zeta = int(n_zeta)
    block_kind = str(block_kind).strip().lower()
    max_block_size = max(1, int(max_block_size))
    ell_block = max(1, int(ell_block))
    if kinetic_size <= 0:
        raise ValueError("kinetic_size must be positive")
    if n_theta <= 0 or n_zeta <= 0:
        raise ValueError("n_theta and n_zeta must be positive")
    plane_size = int(n_theta * n_zeta)
    blocks: list[np.ndarray] = []
    if block_kind in {"zeta", "zeta_line", "zeta-line"}:
        if kinetic_size % n_zeta != 0:
            raise ValueError("zeta-line ordering requires kinetic_size divisible by n_zeta")
        for start in range(0, kinetic_size, n_zeta):
            blocks.append(np.arange(start, start + n_zeta, dtype=np.int64))
        canonical = "zeta_line"
    elif block_kind in {"theta", "theta_line", "theta-line"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("theta-line ordering requires complete angular planes")
        for base in range(0, kinetic_size, plane_size):
            for izeta in range(n_zeta):
                blocks.append(base + np.arange(n_theta, dtype=np.int64) * n_zeta + int(izeta))
        canonical = "theta_line"
    elif block_kind in {"plane", "angular_plane", "ell_plane"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("angular-plane ordering requires complete angular planes")
        for start in range(0, kinetic_size, plane_size):
            blocks.append(np.arange(start, start + plane_size, dtype=np.int64))
        canonical = "angular_plane"
    elif block_kind in {"ell_band", "pitch_band", "plane_band"}:
        if kinetic_size % plane_size != 0:
            raise ValueError("ell-band ordering requires complete angular planes")
        block_size = int(ell_block * plane_size)
        for start in range(0, kinetic_size, block_size):
            stop = min(kinetic_size, start + block_size)
            blocks.append(np.arange(start, stop, dtype=np.int64))
        canonical = "ell_band"
    else:
        raise ValueError(f"unsupported active block ordering {block_kind!r}")
    if not blocks:
        raise ValueError("active block ordering produced no blocks")
    block_size_max = max(int(block.size) for block in blocks)
    if block_size_max > max_block_size:
        raise MemoryError(f"active block size {block_size_max} exceeds max_block_size={max_block_size}")
    return ActiveBlockOrdering(
        blocks=tuple(blocks),
        block_kind=canonical,
        kinetic_size=kinetic_size,
        tail_size=tail_size,
        active_size=int(kinetic_size + tail_size),
        block_size_max=int(block_size_max),
        metadata={
            "block_kind": canonical,
            "block_count": int(len(blocks)),
            "block_size_max": int(block_size_max),
            "n_theta": int(n_theta),
            "n_zeta": int(n_zeta),
            "ell_block": int(ell_block),
        },
    )


def _inverse_dense_block(block: np.ndarray, *, reg: float, dtype: np.dtype) -> np.ndarray:
    block_np = np.asarray(block, dtype=dtype)
    if float(reg) > 0.0:
        scale = max(float(np.linalg.norm(block_np, ord=np.inf)), 1.0)
        block_np = block_np + np.asarray(float(reg) * scale, dtype=dtype) * np.eye(block_np.shape[0], dtype=dtype)
    try:
        return np.linalg.inv(block_np)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(block_np, rcond=max(float(abs(reg)), 1.0e-14))


def build_active_block_schur_factor(
    matrix: Any,
    ordering: ActiveBlockOrdering,
    *,
    dtype: np.dtype = np.dtype(np.float64),
    reg: float = 1.0e-12,
    max_mb: float = 2048.0,
) -> ActiveBlockSchurFactor:
    """Build a block inverse and dense tail Schur complement from a CSR matrix."""

    try:
        import scipy.sparse as sp  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - scipy is required by this path
        raise RuntimeError("scipy is required for active block-Schur factors") from exc

    dtype = np.dtype(dtype)
    matrix_csr = matrix.tocsr().astype(dtype, copy=False)
    kinetic_size = int(ordering.kinetic_size)
    tail_size = int(ordering.tail_size)
    active_size = int(ordering.active_size)
    if matrix_csr.shape != (active_size, active_size):
        raise ValueError(f"matrix shape {matrix_csr.shape} does not match active size {active_size}")
    block_inverse: list[np.ndarray] = []
    inverse_nbytes = 0
    for indices in ordering.blocks:
        idx = np.asarray(indices, dtype=np.int64)
        block = np.asarray(matrix_csr[idx[:, None], idx].toarray(), dtype=dtype)
        inverse = _inverse_dense_block(block, reg=float(reg), dtype=dtype)
        inverse_nbytes += int(inverse.nbytes)
        block_inverse.append(np.asarray(inverse, dtype=dtype))
    c_tail = None
    mb_tail = None
    schur_inverse = None
    tail_nbytes = 0
    if tail_size > 0:
        b_tail = np.asarray(matrix_csr[:kinetic_size, kinetic_size:].toarray(), dtype=dtype)
        c_tail = matrix_csr[kinetic_size:, :kinetic_size].tocsr().astype(dtype, copy=False)
        d_tail = np.asarray(matrix_csr[kinetic_size:, kinetic_size:].toarray(), dtype=dtype)
        mb_tail = np.zeros((kinetic_size, tail_size), dtype=dtype)
        for indices, inverse in zip(ordering.blocks, block_inverse, strict=True):
            idx = np.asarray(indices, dtype=np.int64)
            mb_tail[idx, :] = inverse @ b_tail[idx, :]
        schur = np.asarray(d_tail - c_tail @ mb_tail, dtype=dtype)
        schur_inverse = _inverse_dense_block(schur, reg=float(reg), dtype=dtype)
        tail_nbytes = int(b_tail.nbytes + mb_tail.nbytes + schur_inverse.nbytes)
        if not sp.issparse(c_tail):
            c_tail = sp.csr_matrix(c_tail, dtype=dtype)
    matrix_nbytes = int(matrix_csr.data.nbytes + matrix_csr.indices.nbytes + matrix_csr.indptr.nbytes)
    total_nbytes = int(matrix_nbytes + inverse_nbytes + tail_nbytes)
    if float(max_mb) > 0.0 and total_nbytes > int(float(max_mb) * 1.0e6):
        raise MemoryError(f"active block-Schur factor estimate {total_nbytes / 1.0e6:.3f} MB exceeds {max_mb:.3f} MB")
    return ActiveBlockSchurFactor(
        ordering=ordering,
        block_inverse=tuple(block_inverse),
        c_tail=c_tail,
        mb_tail=None if mb_tail is None else np.asarray(mb_tail, dtype=dtype),
        schur_inverse=None if schur_inverse is None else np.asarray(schur_inverse, dtype=dtype),
        dtype=dtype,
        metadata={
            **ordering.metadata,
            "factor_dtype": dtype.name,
            "inverse_nbytes_estimate": int(inverse_nbytes),
            "tail_nbytes_estimate": int(tail_nbytes),
            "matrix_nbytes_estimate": int(matrix_nbytes),
            "total_nbytes_estimate": int(total_nbytes),
            "reg": float(reg),
        },
    )


def build_active_block_schur_residual_coarse_factor(
    matrix: Any,
    factor: ActiveBlockSchurFactor,
    probes: np.ndarray | None = None,
    *,
    max_cols: int = 8,
    regularization_rel: float = 1.0e-10,
    damping: float = 1.0,
    max_mb: float = 512.0,
) -> ActiveBlockSchurResidualCoarseFactor:
    """Build a small true-residual coarse correction for a block factor.

    The candidate columns are ``M0`` applied to setup residuals from deterministic
    probes.  A thin QR keeps the correction numerically independent and bounded.
    The caller should still run :func:`admit_active_block_schur_factor` against
    the returned factor before using it in production solves.
    """

    matrix_csr = matrix.tocsr().astype(np.dtype(factor.dtype), copy=False)
    ordering = factor.ordering
    active_size = int(ordering.active_size)
    kinetic_size = int(ordering.kinetic_size)
    tail_size = int(ordering.tail_size)
    max_cols = max(1, int(max_cols))
    dtype = np.dtype(factor.dtype)
    if probes is None:
        probes = deterministic_probe_matrix(
            active_size=active_size,
            kinetic_size=kinetic_size,
            tail_size=tail_size,
            count=max_cols,
        )
    probes_np = np.asarray(probes, dtype=dtype)
    if probes_np.ndim == 1:
        probes_np = probes_np.reshape((-1, 1))
    if int(probes_np.shape[0]) != active_size:
        raise ValueError(f"probe length {int(probes_np.shape[0])} does not match active size {active_size}")

    candidates: list[np.ndarray] = []
    residual_norms: list[float] = []
    for icol in range(min(int(probes_np.shape[1]), max_cols)):
        rhs = np.asarray(probes_np[:, icol], dtype=dtype).reshape((active_size,))
        y0 = np.asarray(factor.apply(rhs), dtype=dtype).reshape((active_size,))
        residual = np.asarray(rhs - matrix_csr @ y0, dtype=dtype).reshape((active_size,))
        residual_norm = float(np.linalg.norm(residual))
        if not np.isfinite(residual_norm) or residual_norm <= 0.0:
            continue
        correction = np.asarray(factor.apply(residual), dtype=dtype).reshape((active_size,))
        correction_norm = float(np.linalg.norm(correction))
        if not np.isfinite(correction_norm) or correction_norm <= 0.0:
            continue
        candidates.append(correction / correction_norm)
        residual_norms.append(residual_norm)
    if not candidates:
        raise ValueError("residual coarse correction produced no finite candidate columns")

    candidate_matrix = np.column_stack(candidates).astype(dtype, copy=False)
    q, r = np.linalg.qr(candidate_matrix, mode="reduced")
    diag = np.abs(np.diag(r)) if r.ndim == 2 else np.asarray([], dtype=dtype)
    tol = max(float(np.finfo(dtype).eps) * max(candidate_matrix.shape) * float(np.linalg.norm(candidate_matrix)), 1.0e-14)
    keep = np.nonzero(diag > tol)[0]
    if keep.size == 0:
        raise ValueError("residual coarse correction columns are rank deficient")
    keep = keep[:max_cols]
    coarse_basis = np.asarray(q[:, keep], dtype=dtype)
    action_basis = np.asarray(matrix_csr @ coarse_basis, dtype=dtype)
    coarse_cols = int(coarse_basis.shape[1])
    gram = np.asarray(action_basis.T @ action_basis, dtype=dtype)
    gram_scale = max(float(np.linalg.norm(gram, ord=np.inf)), 1.0)
    regularization = max(float(regularization_rel), 0.0) * gram_scale
    normal = gram + np.asarray(regularization, dtype=dtype) * np.eye(coarse_cols, dtype=dtype)
    try:
        normal_inverse = np.linalg.inv(normal)
    except np.linalg.LinAlgError:
        normal_inverse = np.linalg.pinv(normal, rcond=max(float(regularization_rel), 1.0e-14))

    coarse_nbytes = int(coarse_basis.nbytes + action_basis.nbytes + normal_inverse.nbytes)
    if float(max_mb) > 0.0 and coarse_nbytes > int(float(max_mb) * 1.0e6):
        raise MemoryError(f"residual coarse estimate {coarse_nbytes / 1.0e6:.3f} MB exceeds {max_mb:.3f} MB")

    metadata = {
        **dict(factor.metadata),
        "residual_coarse": True,
        "residual_coarse_cols": int(coarse_cols),
        "residual_coarse_candidate_cols": int(len(candidates)),
        "residual_coarse_nbytes_estimate": int(coarse_nbytes),
        "residual_coarse_regularization_rel": float(regularization_rel),
        "residual_coarse_regularization": float(regularization),
        "residual_coarse_damping": float(damping),
        "residual_coarse_probe_residual_norm_max": float(np.max(residual_norms)),
        "residual_coarse_probe_residual_norm_median": float(np.median(residual_norms)),
    }
    return ActiveBlockSchurResidualCoarseFactor(
        base=factor,
        matrix=matrix_csr,
        coarse_basis=coarse_basis,
        action_basis=action_basis,
        normal_inverse=np.asarray(normal_inverse, dtype=dtype),
        damping=float(damping),
        ordering=ordering,
        dtype=dtype,
        metadata=metadata,
    )


def deterministic_probe_matrix(
    *,
    active_size: int,
    kinetic_size: int,
    tail_size: int,
    count: int = 4,
) -> np.ndarray:
    """Return deterministic setup probes for preconditioner admission."""

    active_size = int(active_size)
    kinetic_size = int(kinetic_size)
    tail_size = int(tail_size)
    count = max(1, int(count))
    probes: list[np.ndarray] = []
    if kinetic_size > 0:
        probes.append(np.r_[np.ones((kinetic_size,), dtype=np.float64), np.zeros((tail_size,), dtype=np.float64)])
        ramp = np.linspace(-1.0, 1.0, kinetic_size, dtype=np.float64)
        probes.append(np.r_[ramp, np.zeros((tail_size,), dtype=np.float64)])
    for itail in range(min(tail_size, max(0, count - len(probes)))):
        vec = np.zeros((active_size,), dtype=np.float64)
        vec[kinetic_size + int(itail)] = 1.0
        probes.append(vec)
    rng = np.random.default_rng(20260609)
    while len(probes) < count:
        probes.append(rng.normal(size=active_size))
    out = np.column_stack(probes[:count])
    norms = np.linalg.norm(out, axis=0)
    norms = np.where(norms > 0.0, norms, 1.0)
    return out / norms[None, :]


def admit_active_block_schur_factor(
    matrix: Any,
    factor: ActiveBlockSchurFactor,
    probes: np.ndarray | None = None,
    *,
    max_relative_residual: float = 1.0e-2,
    min_improvement_vs_identity: float = 10.0,
) -> ActiveBlockAdmission:
    """Evaluate true residual quality before admitting a block factor."""

    matrix_csr = matrix.tocsr()
    ordering = factor.ordering
    if probes is None:
        probes = deterministic_probe_matrix(
            active_size=int(ordering.active_size),
            kinetic_size=int(ordering.kinetic_size),
            tail_size=int(ordering.tail_size),
            count=4,
        )
    probes_np = np.asarray(probes, dtype=np.float64)
    if probes_np.ndim == 1:
        probes_np = probes_np.reshape((-1, 1))
    rels: list[float] = []
    improvements: list[float] = []
    for icol in range(int(probes_np.shape[1])):
        rhs = probes_np[:, icol]
        rhs_norm = max(float(np.linalg.norm(rhs)), 1.0e-300)
        y = factor.apply(rhs)
        residual = np.asarray(matrix_csr @ y - rhs, dtype=np.float64)
        rel = float(np.linalg.norm(residual) / rhs_norm)
        identity_residual = np.asarray(matrix_csr @ rhs - rhs, dtype=np.float64)
        identity_rel = float(np.linalg.norm(identity_residual) / rhs_norm)
        improvement = identity_rel / max(rel, 1.0e-300)
        rels.append(rel)
        improvements.append(improvement)
    max_rel = float(np.max(rels)) if rels else float("inf")
    med_rel = float(np.median(rels)) if rels else float("inf")
    min_improvement = float(np.min(improvements)) if improvements else 0.0
    accepted = bool(max_rel <= float(max_relative_residual) and min_improvement >= float(min_improvement_vs_identity))
    if accepted:
        reason = "accepted"
    elif max_rel > float(max_relative_residual):
        reason = "relative_residual_gate"
    else:
        reason = "improvement_gate"
    return ActiveBlockAdmission(
        accepted=accepted,
        max_relative_residual=max_rel,
        median_relative_residual=med_rel,
        min_improvement_vs_identity=min_improvement,
        probe_count=int(probes_np.shape[1]),
        reason=reason,
    )


# --- Direct reduced-Pmat and active-operator emission ---





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


# --- Direct active block-Schur transport preconditioner ---





def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def _int_env(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return int(default)
    try:
        return int(value)
    except ValueError:
        return int(default)


def build_transport_fp_direct_active_block_schur_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    fallback_builder: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]],
    transport_precond_cache_key: Callable[[V3FullSystemOperator, str], tuple[object, ...]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Build the bounded direct active block-Schur FP transport preconditioner.

    The module owns setup, cache, admission, residual-coarse rescue, and host
    callback application.  The caller injects the fallback preconditioner and
    cache-key builder because those still live in the driver orchestration
    layer during the active refactor branch.
    """

    if int(op.rhs_mode) not in {2, 3} or op.fblock.fp is None:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    if reduce_full is None or expand_reduced is None or active_indices_np is None:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    prefix = "SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR"
    dtype_env = os.environ.get(f"{prefix}_FACTOR_DTYPE", "").strip().lower()
    block_kind_env = os.environ.get(f"{prefix}_BLOCK_KIND", "").strip().lower()
    admission_env = os.environ.get(f"{prefix}_ADMISSION", "").strip().lower()
    coarse_env = os.environ.get(f"{prefix}_RESIDUAL_COARSE", "").strip().lower()
    max_mb = _float_env(f"{prefix}_MAX_MB", 2048.0)
    max_block = _int_env(f"{prefix}_MAX_BLOCK", 64)
    reg = _float_env(f"{prefix}_REG", 1.0e-12)
    tail_max = _int_env(f"{prefix}_TAIL_MAX", 256)
    ell_block = _int_env(f"{prefix}_ELL_BLOCK", 1)
    admission_max_rel = _float_env(f"{prefix}_ADMISSION_MAX_REL", 1.0e-2)
    admission_min_improvement = _float_env(f"{prefix}_ADMISSION_MIN_IMPROVEMENT", 10.0)
    admission_probe_count = _int_env(f"{prefix}_ADMISSION_PROBES", 4)
    coarse_max_cols = _int_env(f"{prefix}_RESIDUAL_COARSE_MAX_COLS", 8)
    coarse_max_mb = _float_env(f"{prefix}_RESIDUAL_COARSE_MAX_MB", 512.0)
    coarse_regularization_rel = _float_env(f"{prefix}_RESIDUAL_COARSE_REGULARIZATION_REL", 1.0e-10)
    coarse_damping = _float_env(f"{prefix}_RESIDUAL_COARSE_DAMPING", 1.0)
    block_kind = block_kind_env if block_kind_env else "zeta_line"
    admission_enabled = admission_env not in {"0", "false", "no", "off"}
    coarse_enabled = coarse_env not in {"0", "false", "no", "off"}
    factor_dtype = np.dtype(np.float32) if dtype_env in {"float32", "fp32", "32"} else np.dtype(np.float64)
    active_indices_use = np.asarray(active_indices_np, dtype=np.int64).reshape((-1,))
    if active_indices_use.size <= 0:
        return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
    active_hash = _hash_numpy_array_for_cache(active_indices_use)
    cache_key = (
        *transport_precond_cache_key(
            op,
            "fp_direct_active_block_schur_"
            f"{block_kind}_{int(ell_block)}_{factor_dtype.name}_{float(reg):.3e}_"
            f"{int(max_block)}_{int(tail_max)}_{int(admission_enabled)}_"
            f"{float(admission_max_rel):.3e}_{float(admission_min_improvement):.3e}_"
            f"{int(admission_probe_count)}_{int(coarse_enabled)}_{int(coarse_max_cols)}_"
            f"{float(coarse_max_mb):.3e}_{float(coarse_regularization_rel):.3e}_"
            f"{float(coarse_damping):.3e}",
        ),
        str(active_hash),
        int(active_indices_use.size),
        int(float(max_mb) * 1.0e6) if float(max_mb) > 0.0 else 0,
    )
    cached = _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE.get(cache_key)
    if cached is None:
        try:
            direct_result = _try_build_rhsmode23_fp_direct_active_operator_bundle(
                op=op,
                active_indices=active_indices_use,
                factor_dtype=factor_dtype,
                emit=emit,
            )
            if direct_result is None:
                raise RuntimeError("direct active true operator unavailable")
            operator_bundle, direct_metadata = direct_result
            matrix = operator_bundle.matrix
            if matrix is None:
                raise RuntimeError("direct active true operator has no materialized CSR matrix")
            matrix = matrix.tocsr().astype(factor_dtype, copy=False)
            kinetic_size = int(direct_metadata.get("direct_pmat_kinetic_size", 0))
            tail_size = int(direct_metadata.get("direct_pmat_tail_size", 0))
            active_size = int(matrix.shape[0])
            if kinetic_size <= 0 or kinetic_size > active_size:
                raise RuntimeError("invalid direct active kinetic size")
            if tail_size < 0 or kinetic_size + tail_size != active_size:
                raise RuntimeError("invalid direct active tail size")
            if tail_size > int(tail_max):
                raise RuntimeError(f"tail size {tail_size} exceeds tail_max={int(tail_max)}")

            build_timer = Timer()
            ordering = build_active_block_ordering(
                kinetic_size=int(kinetic_size),
                tail_size=int(tail_size),
                n_theta=int(op.n_theta),
                n_zeta=int(op.n_zeta),
                block_kind=str(block_kind),
                ell_block=int(ell_block),
                max_block_size=int(max_block),
            )
            factor = build_active_block_schur_factor(
                matrix,
                ordering,
                dtype=factor_dtype,
                reg=float(reg),
                max_mb=float(max_mb),
            )
            admission = None
            coarse_metadata: dict[str, object] = {
                "residual_coarse_enabled": bool(coarse_enabled),
                "residual_coarse_accepted": None,
            }
            if bool(admission_enabled):
                probes = deterministic_probe_matrix(
                    active_size=int(active_size),
                    kinetic_size=int(kinetic_size),
                    tail_size=int(tail_size),
                    count=max(1, int(admission_probe_count)),
                )
                admission = admit_active_block_schur_factor(
                    matrix,
                    factor,
                    probes,
                    max_relative_residual=float(admission_max_rel),
                    min_improvement_vs_identity=float(admission_min_improvement),
                )
                if not bool(admission.accepted):
                    if bool(coarse_enabled):
                        try:
                            coarse_factor = build_active_block_schur_residual_coarse_factor(
                                matrix,
                                factor,
                                probes,
                                max_cols=int(coarse_max_cols),
                                regularization_rel=float(coarse_regularization_rel),
                                damping=float(coarse_damping),
                                max_mb=float(coarse_max_mb),
                            )
                            coarse_admission = admit_active_block_schur_factor(
                                matrix,
                                coarse_factor,
                                probes,
                                max_relative_residual=float(admission_max_rel),
                                min_improvement_vs_identity=float(admission_min_improvement),
                            )
                            coarse_metadata = {
                                "residual_coarse_enabled": True,
                                "residual_coarse_accepted": bool(coarse_admission.accepted),
                                "residual_coarse_reason": str(coarse_admission.reason),
                                "residual_coarse_max_relative_residual": float(
                                    coarse_admission.max_relative_residual
                                ),
                                "residual_coarse_median_relative_residual": float(
                                    coarse_admission.median_relative_residual
                                ),
                                "residual_coarse_min_improvement_vs_identity": float(
                                    coarse_admission.min_improvement_vs_identity
                                ),
                                "residual_coarse_probe_count": int(coarse_admission.probe_count),
                                "residual_coarse_max_cols": int(coarse_max_cols),
                                "residual_coarse_max_mb": float(coarse_max_mb),
                                "residual_coarse_regularization_rel": float(coarse_regularization_rel),
                                "residual_coarse_damping": float(coarse_damping),
                            }
                            if bool(coarse_admission.accepted):
                                factor = coarse_factor
                                admission = coarse_admission
                            elif emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: "
                                    "fp_direct_active_block_schur residual coarse rejected "
                                    f"max_rel={float(coarse_admission.max_relative_residual):.3e} "
                                    f"reason={coarse_admission.reason}",
                                )
                        except Exception as coarse_exc:  # noqa: BLE001
                            coarse_metadata = {
                                "residual_coarse_enabled": True,
                                "residual_coarse_accepted": False,
                                "residual_coarse_error": f"{type(coarse_exc).__name__}: {coarse_exc}",
                                "residual_coarse_max_cols": int(coarse_max_cols),
                                "residual_coarse_max_mb": float(coarse_max_mb),
                                "residual_coarse_regularization_rel": float(coarse_regularization_rel),
                                "residual_coarse_damping": float(coarse_damping),
                            }
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: "
                                    "fp_direct_active_block_schur residual coarse unavailable "
                                    f"({type(coarse_exc).__name__}: {coarse_exc})",
                                )
                    if not bool(admission.accepted):
                        coarse_suffix = ""
                        if coarse_metadata.get("residual_coarse_accepted") is False:
                            coarse_suffix = (
                                ", residual_coarse="
                                f"{coarse_metadata.get('residual_coarse_reason', coarse_metadata.get('residual_coarse_error', 'rejected'))}"
                            )
                        raise RuntimeError(
                            "admission rejected direct active block-Schur "
                            f"(block_kind={factor.ordering.block_kind}, blocks={len(factor.ordering.blocks)}, "
                            f"reason={admission.reason}, max_rel={admission.max_relative_residual:.3e}, "
                            f"min_improvement={admission.min_improvement_vs_identity:.3e}{coarse_suffix})"
                        )
            else:
                coarse_metadata = {
                    "residual_coarse_enabled": bool(coarse_enabled),
                    "residual_coarse_accepted": None,
                    "residual_coarse_reason": "admission_disabled",
                }

            factor_metadata = dict(factor.metadata)
            metadata = {
                "kind": "fp_direct_active_block_schur",
                "factor_dtype": str(factor_dtype.name),
                "axis": str(factor_metadata.get("block_kind", block_kind)),
                "block_kind": str(factor_metadata.get("block_kind", block_kind)),
                "block_size": int(factor_metadata.get("block_size_max", 0)),
                "block_count": int(factor_metadata.get("block_count", 0)),
                "kinetic_size": int(kinetic_size),
                "tail_size": int(tail_size),
                "reg": float(reg),
                "matrix_nbytes_estimate": int(factor_metadata.get("matrix_nbytes_estimate", 0)),
                "block_inverse_nbytes_estimate": int(factor_metadata.get("inverse_nbytes_estimate", 0)),
                "tail_dense_nbytes_estimate": int(factor_metadata.get("tail_nbytes_estimate", 0)),
                "total_nbytes_estimate": int(factor_metadata.get("total_nbytes_estimate", 0)),
                "setup_s": float(build_timer.elapsed_s()),
                "schur_reason": "dense_schur" if int(tail_size) > 0 else "none",
                "admission_enabled": bool(admission_enabled),
                "admission_accepted": None if admission is None else bool(admission.accepted),
                "admission_reason": None if admission is None else str(admission.reason),
                "admission_max_relative_residual": None
                if admission is None
                else float(admission.max_relative_residual),
                "admission_median_relative_residual": None
                if admission is None
                else float(admission.median_relative_residual),
                "admission_min_improvement_vs_identity": None
                if admission is None
                else float(admission.min_improvement_vs_identity),
                "admission_probe_count": None if admission is None else int(admission.probe_count),
                "residual_coarse_enabled": bool(coarse_enabled),
            }
            metadata.update(direct_metadata)
            metadata.update(coarse_metadata)
            packed_factor = getattr(factor, "base", factor)
            cached = _TransportFpDirectActiveBlockSchurPrecondCache(
                block_inverse=(),
                block_size=int(factor.ordering.block_size_max),
                kinetic_size=int(kinetic_size),
                tail_size=int(tail_size),
                c_tail=getattr(packed_factor, "c_tail", None),
                mb_tail=None
                if getattr(packed_factor, "mb_tail", None) is None
                else np.asarray(packed_factor.mb_tail, dtype=factor_dtype),
                schur_inverse=None
                if getattr(packed_factor, "schur_inverse", None) is None
                else np.asarray(packed_factor.schur_inverse, dtype=factor_dtype),
                metadata=metadata,
                factor=factor,
            )
            _TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_PRECOND_CACHE[cache_key] = cached
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_direct_active_block_schur selected "
                    f"active={active_size} kinetic={kinetic_size} tail={tail_size} "
                    f"blocks={int(metadata['block_count'])}x<= {int(metadata['block_size'])} "
                    f"setup_s={float(metadata['setup_s']):.3f} "
                    f"est_mb={float(metadata['total_nbytes_estimate']) / 1.0e6:.3f}",
                )
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_direct_active_block_schur unavailable; "
                    f"using sxblock ({type(exc).__name__}: {exc})",
                )
            return fallback_builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)

    block_inverse = cached.block_inverse
    block_size = int(cached.block_size)
    kinetic_size = int(cached.kinetic_size)
    tail_size = int(cached.tail_size)
    n_blocks = int(kinetic_size // block_size)
    c_tail = cached.c_tail
    mb_tail = cached.mb_tail
    schur_inverse = cached.schur_inverse
    factor_obj = getattr(cached, "factor", None)
    if factor_obj is not None:
        factor_dtype_use = np.dtype(getattr(factor_obj, "dtype", np.float64))
    else:
        factor_dtype_use = np.dtype(getattr(block_inverse, "dtype", np.float64))
    active_size_use = int(kinetic_size + tail_size)

    def _apply_host(rhs_host: np.ndarray) -> np.ndarray:
        if factor_obj is not None:
            return np.asarray(factor_obj.apply(rhs_host), dtype=np.float64)
        rhs_np = np.asarray(rhs_host, dtype=factor_dtype_use).reshape((active_size_use,))
        rhs_k = rhs_np[:kinetic_size]
        rhs_blocks = rhs_k.reshape((n_blocks, block_size, 1))
        y_k = np.einsum("bij,bjk->bik", block_inverse, rhs_blocks, optimize=True).reshape((kinetic_size,))
        if tail_size > 0 and c_tail is not None and mb_tail is not None and schur_inverse is not None:
            rhs_t = rhs_np[kinetic_size:]
            tail_residual = np.asarray(rhs_t - c_tail @ y_k, dtype=factor_dtype_use).reshape((tail_size,))
            y_t = np.asarray(schur_inverse @ tail_residual, dtype=factor_dtype_use).reshape((tail_size,))
            y_k = np.asarray(y_k - mb_tail @ y_t, dtype=factor_dtype_use).reshape((kinetic_size,))
            out = np.concatenate([y_k, y_t], axis=0)
        else:
            out = np.concatenate([y_k, rhs_np[kinetic_size:]], axis=0)
        finite = np.isfinite(out)
        if not np.all(finite):
            out = np.where(finite, out, 0.0)
        return np.asarray(out, dtype=np.float64)

    def _apply_reduced(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64)
        return jax.pure_callback(
            _apply_host,
            jax.ShapeDtypeStruct((active_size_use,), jnp.float64),
            v,
        )

    try:
        setattr(_apply_reduced, "_sfincs_jax_transport_fp_direct_active_block_schur_metadata", dict(cached.metadata))
    except Exception:
        pass
    return _apply_reduced


# --- Fortran-reduced transport LU preconditioner ---





def build_transport_fp_fortran_reduced_lu_preconditioner(
    *,
    op: V3FullSystemOperator,
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
    active_indices_np: np.ndarray | None = None,
    emit: Callable[[int, str], None] | None = None,
    fallback_builder: Callable[..., Callable[[jnp.ndarray], jnp.ndarray]],
    transport_precond_cache_key: Callable[[V3FullSystemOperator, str], tuple[object, ...]],
    build_host_sparse_direct_factor_from_matvec: Callable[..., object],
    host_physical_memory_mb: Callable[[], float | None],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Global reduced sparse-factor transport preconditioner.

    This is the closest SFINCS-JAX transport analogue to the Fortran v3
    PETSc setup: GMRES still applies the true operator, while this builder
    materializes and factors a separate reduced ``Pmat``.  It is intentionally
    opt-in until production-size residual gates prove that the setup cost and
    memory footprint are justified.
    """

    if int(op.rhs_mode) not in {2, 3} or op.fblock.fp is None:
        return fallback_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    use_reduced = reduce_full is not None and expand_reduced is not None
    if use_reduced and active_indices_np is None:
        return fallback_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )

    def _int_env(name: str, default: int, *, minimum: int = 0) -> int:
        value = os.environ.get(name, "").strip()
        try:
            parsed = int(value) if value else int(default)
        except ValueError:
            parsed = int(default)
        return max(int(minimum), int(parsed))

    def _float_env(name: str, default: float, *, minimum: float = 0.0) -> float:
        value = os.environ.get(name, "").strip()
        try:
            parsed = float(value) if value else float(default)
        except ValueError:
            parsed = float(default)
        return max(float(minimum), float(parsed))

    def _bool_env(name: str, default: bool) -> bool:
        value = os.environ.get(name, "").strip().lower()
        if value in {"1", "true", "yes", "on"}:
            return True
        if value in {"0", "false", "no", "off"}:
            return False
        return bool(default)

    # Fortran v3 defaults reduce x and xi couplings, but keep the full
    # theta/zeta derivative matrices (preconditioner_theta/zeta=0).  That exact
    # Pmat is available via env overrides, but this opt-in transport candidate
    # defaults to the stronger x/xi-coupled variant because the default-reduced
    # Pmat is too slow for the current FP geometry-rich residual gates.
    preconditioner_x = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_X", 0)
    preconditioner_xi = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_XI", 0)
    preconditioner_species = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_SPECIES", 1)
    preconditioner_x_min_l = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECONDITIONER_X_MIN_L", 0)
    keep_theta_zeta = _bool_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_KEEPS_THETA_ZETA", True)
    pc_shift = _float_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SHIFT", 1.0e-10)
    max_factor_mb = _float_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", 4096.0)
    direct_pmat_enabled = _bool_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", True)
    symbolic_ordering = (
        os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "").strip().lower()
        or "mumps_like"
    )
    symbolic_block_size = _int_env("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", 4096, minimum=1)
    symbolic_block_overlap = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_OVERLAP",
        0,
        minimum=0,
    )
    symbolic_coarse_max_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_MAX_COLS",
        256,
        minimum=1,
    )
    symbolic_coarse_probe_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_PROBE_COLS",
        4,
        minimum=0,
    )
    symbolic_coarse_damping = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_DAMPING",
        1.0,
        minimum=0.0,
    )
    symbolic_coarse_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_COARSE_REG_REL",
        1.0e-10,
        minimum=0.0,
    )
    symbolic_physics_coarse_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PHYSICS_COARSE",
        True,
    )
    symbolic_physics_coarse_max_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PHYSICS_COARSE_MAX_COLS",
        32,
        minimum=1,
    )
    symbolic_schur_max_separator_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS",
        256,
        minimum=0,
    )
    symbolic_schur_boundary_width = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_BOUNDARY_WIDTH",
        1,
        minimum=0,
    )
    symbolic_schur_high_degree_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_HIGH_DEGREE_COLS",
        64,
        minimum=0,
    )
    symbolic_schur_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_frontal_max_separator_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS",
        1024,
        minimum=0,
    )
    symbolic_frontal_boundary_width = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH",
        1,
        minimum=0,
    )
    symbolic_frontal_high_degree_cols = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS",
        128,
        minimum=0,
    )
    symbolic_frontal_max_superblock_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE",
        8192,
        minimum=1,
    )
    symbolic_frontal_max_superblock_blocks = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS",
        8,
        minimum=1,
    )
    symbolic_frontal_min_cross_nnz = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MIN_CROSS_NNZ",
        1,
        minimum=1,
    )
    symbolic_frontal_min_cross_separator_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION",
        0.0,
        minimum=0.0,
    )
    symbolic_frontal_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_frontal_max_dense_rhs_entries = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_ENTRIES",
        0,
        minimum=0,
    )
    symbolic_frontal_max_dense_rhs_cols_per_block = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_COLS_PER_BLOCK",
        0,
        minimum=0,
    )
    symbolic_superblock_max_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MAX_SIZE",
        32768,
        minimum=1,
    )
    symbolic_superblock_max_blocks = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS",
        8,
        minimum=1,
    )
    symbolic_superblock_min_cross_nnz = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MIN_CROSS_NNZ",
        1,
        minimum=1,
    )
    symbolic_superblock_min_retained_cross_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_MIN_RETAINED_CROSS_FRACTION",
        0.0,
        minimum=0.0,
    )
    symbolic_superblock_regularization_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SUPERBLOCK_REG_REL",
        1.0e-12,
        minimum=0.0,
    )
    symbolic_numeric_parallel_workers_default = min(4, max(1, int(os.cpu_count() or 1)))
    symbolic_numeric_parallel_workers = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_NUMERIC_PARALLEL_WORKERS",
        symbolic_numeric_parallel_workers_default,
        minimum=1,
    )
    symbolic_max_permutation_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_MAX_PERMUTATION_SIZE",
        250_000,
        minimum=0,
    )
    symbolic_admission_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION",
        True,
    )
    symbolic_admission_max_rel = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL",
        1.0e-2,
        minimum=0.0,
    )
    symbolic_admission_min_improvement = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT",
        10.0,
        minimum=0.0,
    )
    symbolic_admission_probes = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_PROBES",
        4,
        minimum=1,
    )
    symbolic_admission_rescue_lu = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU",
        True,
    )
    symbolic_admission_rescue_lu_max_mb = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU_MAX_MB",
        max_factor_mb,
        minimum=0.0,
    )
    auto_exact_rescue_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE",
        True,
    )
    auto_exact_rescue_ram_fraction = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_RAM_FRACTION",
        0.45,
        minimum=0.0,
    )
    host_memory_mb = host_physical_memory_mb()
    auto_exact_rescue_default_max_mb = (
        0.0
        if host_memory_mb is None
        else max(0.0, float(host_memory_mb) * float(auto_exact_rescue_ram_fraction))
    )
    auto_exact_rescue_max_mb = _float_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_MB",
        auto_exact_rescue_default_max_mb,
        minimum=0.0,
    )
    auto_exact_rescue_max_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_SIZE",
        250_000,
        minimum=0,
    )
    auto_exact_rescue_max_factor_entries = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_FACTOR_ENTRIES",
        250_000_000,
        minimum=0,
    )
    direct_admission_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT_ADMISSION",
        True,
    )
    direct_admission_explicit_enabled = _bool_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT_ADMISSION_EXPLICIT",
        False,
    )
    factor_dtype_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_DTYPE", "").strip().lower()
    factor_dtype = np.dtype(np.float32) if factor_dtype_env in {"float32", "fp32", "32"} else np.dtype(np.float64)
    factor_kind_env = os.environ.get("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "").strip().lower()

    active_hash = "full"
    active_indices_use: np.ndarray | None = None
    if use_reduced:
        active_indices_use = np.asarray(active_indices_np, dtype=np.int32).reshape((-1,))
        active_hash = _hash_numpy_array_for_cache(active_indices_use)
        linear_size = int(active_indices_use.size)
    else:
        linear_size = int(op.total_size)
    default_factor_kind = (
        factor_kind_env
        if factor_kind_env
        in {
            "lu",
            "ilu",
            "jacobi",
            "spilu",
            "diag",
            "diagonal",
            "none",
            "symbolic_block_lu",
            "block_lu",
            "native_block_lu",
            "symbolic_lu",
            "symbolic_block_schur_lu",
            "block_schur_lu",
            "native_block_schur_lu",
            "symbolic_schur_lu",
            "symbolic_block_lu_coarse",
            "block_lu_coarse",
            "native_block_lu_coarse",
            "symbolic_lu_coarse",
            "symbolic_frontal_schur_lu",
            "frontal_schur_lu",
            "native_frontal_schur_lu",
            "multifrontal_schur_lu",
            "symbolic_superblock_lu",
            "superblock_lu",
            "native_superblock_lu",
            "block_edge_lu",
        }
        else "lu"
    )
    if default_factor_kind in {"spilu"}:
        default_factor_kind = "ilu"
    elif default_factor_kind in {"diag", "diagonal", "none"}:
        default_factor_kind = "jacobi"
    elif default_factor_kind in {"block_schur_lu", "native_block_schur_lu", "symbolic_schur_lu"}:
        default_factor_kind = "symbolic_block_schur_lu"
    elif default_factor_kind in {"frontal_schur_lu", "native_frontal_schur_lu", "multifrontal_schur_lu"}:
        default_factor_kind = "symbolic_frontal_schur_lu"
    elif default_factor_kind in {"superblock_lu", "native_superblock_lu", "block_edge_lu"}:
        default_factor_kind = "symbolic_superblock_lu"
    elif default_factor_kind in {"block_lu_coarse", "native_block_lu_coarse", "symbolic_lu_coarse"}:
        default_factor_kind = "symbolic_block_lu_coarse"
    elif default_factor_kind in {"block_lu", "native_block_lu", "symbolic_lu"}:
        default_factor_kind = "symbolic_block_lu"
    explicit_factor_requested = bool(factor_kind_env) or bool(
        os.environ.get("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "").strip()
    )
    monolithic_auto_guard_size = _int_env(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MONOLITHIC_AUTO_MAX_SIZE",
        250_000,
        minimum=0,
    )
    if (
        bool(direct_pmat_enabled)
        and not bool(explicit_factor_requested)
        and default_factor_kind in {"lu", "ilu"}
        and int(monolithic_auto_guard_size) > 0
        and int(linear_size) > int(monolithic_auto_guard_size)
    ):
        default_factor_kind = "symbolic_block_lu_coarse"
        if emit is not None:
            emit(
                1,
                "solve_v3_transport_matrix_linear_gmres: large direct-Pmat auto factor switched "
                "from monolithic LU/ILU to symbolic_block_lu_coarse "
                f"(linear_size={int(linear_size)} max_size={int(monolithic_auto_guard_size)})",
            )

    op_pc = _build_transport_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x_min_l=int(preconditioner_x_min_l),
        keep_theta_zeta=bool(keep_theta_zeta),
    )
    cache_key = (
        *transport_precond_cache_key(
            op,
            "fp_fortran_reduced_lu_"
            f"{default_factor_kind}_{factor_dtype.name}_{float(pc_shift):.3e}_"
            f"{int(preconditioner_x)}_{int(preconditioner_xi)}_{int(preconditioner_species)}_"
            f"{int(preconditioner_x_min_l)}_{int(keep_theta_zeta)}_direct{int(direct_pmat_enabled)}_"
            f"symbolic{str(symbolic_ordering)}_{int(symbolic_block_size)}_{int(symbolic_block_overlap)}_"
            f"coarse{int(symbolic_coarse_max_cols)}_probes{int(symbolic_coarse_probe_cols)}_"
            f"damp{float(symbolic_coarse_damping):.3e}_{float(symbolic_coarse_regularization_rel):.3e}_"
            f"phys{int(symbolic_physics_coarse_enabled)}_{int(symbolic_physics_coarse_max_cols)}_"
            f"schur{int(symbolic_schur_max_separator_cols)}_{int(symbolic_schur_boundary_width)}_"
            f"{int(symbolic_schur_high_degree_cols)}_{float(symbolic_schur_regularization_rel):.3e}_"
            f"frontal{int(symbolic_frontal_max_separator_cols)}_{int(symbolic_frontal_boundary_width)}_"
            f"{int(symbolic_frontal_high_degree_cols)}_{int(symbolic_frontal_max_superblock_size)}_"
            f"{int(symbolic_frontal_max_superblock_blocks)}_{int(symbolic_frontal_min_cross_nnz)}_"
            f"{float(symbolic_frontal_min_cross_separator_fraction):.3e}_"
            f"{float(symbolic_frontal_regularization_rel):.3e}_"
            f"{int(symbolic_frontal_max_dense_rhs_entries)}_"
            f"{int(symbolic_frontal_max_dense_rhs_cols_per_block)}_"
            f"super{int(symbolic_superblock_max_size)}_{int(symbolic_superblock_max_blocks)}_"
            f"{int(symbolic_superblock_min_cross_nnz)}_"
            f"{float(symbolic_superblock_min_retained_cross_fraction):.3e}_"
            f"{float(symbolic_superblock_regularization_rel):.3e}_"
            f"symworkers{int(symbolic_numeric_parallel_workers)}_"
            f"{int(symbolic_max_permutation_size)}_"
            f"adm{int(symbolic_admission_enabled)}_{float(symbolic_admission_max_rel):.3e}_"
            f"{float(symbolic_admission_min_improvement):.3e}_{int(symbolic_admission_probes)}_"
            f"rescue{int(symbolic_admission_rescue_lu)}_{float(symbolic_admission_rescue_lu_max_mb):.3e}_"
            f"autoexact{int(auto_exact_rescue_enabled)}_{float(auto_exact_rescue_max_mb):.3e}_"
            f"{float(auto_exact_rescue_ram_fraction):.3e}_{int(auto_exact_rescue_max_size)}_"
            f"{int(auto_exact_rescue_max_factor_entries)}_"
            f"directadm{int(direct_admission_enabled)}_{int(direct_admission_explicit_enabled)}_"
            f"maxfactor{float(max_factor_mb):.3e}",
        ),
        str(active_hash),
        int(linear_size),
    )
    cached = _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE.get(cache_key)
    if cached is None:
        try:
            direct_operator_bundle: SparseOperatorBundle | None = None
            direct_metadata: dict[str, object] = {}
            if bool(direct_pmat_enabled):
                direct_result = _try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
                    op_pc=op_pc,
                    active_indices=active_indices_use,
                    factor_dtype=factor_dtype,
                    pc_shift=float(pc_shift),
                    emit=emit,
                )
                if direct_result is not None:
                    direct_operator_bundle, direct_metadata = direct_result

            def _expand(x: jnp.ndarray) -> jnp.ndarray:
                if expand_reduced is None:
                    return x
                return expand_reduced(x)

            def _reduce(y: jnp.ndarray) -> jnp.ndarray:
                if reduce_full is None:
                    return y
                return reduce_full(y)

            def _pc_matvec(x_vec: jnp.ndarray) -> jnp.ndarray:
                x_full = _expand(jnp.asarray(x_vec, dtype=jnp.float64))
                y_full = apply_v3_full_system_operator_cached(op_pc, x_full)
                if float(pc_shift) != 0.0:
                    y_full = y_full + jnp.asarray(float(pc_shift), dtype=jnp.float64) * x_full
                return _reduce(y_full)

            _operator_bundle = None
            factor_bundle = None
            factor_kind_for_build = str(default_factor_kind)
            effective_factor_max_mb = float(max_factor_mb)
            auto_exact_rescue_selected = False
            if direct_operator_bundle is not None:
                direct_csr_nbytes = int(direct_metadata.get("direct_pmat_csr_nbytes_estimate", 0) or 0)
                direct_symbolic_prefill_safety = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR",
                    64.0,
                    minimum=1.0,
                )
                direct_symbolic_prefill_estimate = (
                    int(np.ceil(float(direct_csr_nbytes) * float(direct_symbolic_prefill_safety)))
                    if direct_csr_nbytes > 0
                    and factor_kind_for_build
                    in {
                        "symbolic_block_lu",
                        "symbolic_block_lu_coarse",
                        "symbolic_block_schur_lu",
                        "symbolic_frontal_schur_lu",
                        "symbolic_superblock_lu",
                    }
                    else 0
                )
                direct_pmat_nnz = int(direct_metadata.get("direct_pmat_nnz", 0) or 0)
                direct_mf_fill_ratio = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MULTIFRONTAL_FILL_RATIO",
                    104.0,
                    minimum=1.0,
                )
                direct_mf_overhead = _float_env(
                    "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MULTIFRONTAL_OVERHEAD",
                    1.15,
                    minimum=1.0,
                )
                direct_multifrontal_entries_estimate = (
                    int(np.ceil(float(direct_pmat_nnz) * float(direct_mf_fill_ratio)))
                    if direct_pmat_nnz > 0
                    else 0
                )
                direct_multifrontal_nbytes_estimate = (
                    estimate_multifrontal_direct_lu_nbytes(
                        direct_pmat_nnz,
                        fill_ratio=float(direct_mf_fill_ratio),
                        data_dtype=factor_dtype,
                        overhead=float(direct_mf_overhead),
                    )
                    if direct_pmat_nnz > 0
                    else 0
                )
                direct_metadata.update(
                    {
                        "direct_pmat_multifrontal_fill_ratio_estimate": float(direct_mf_fill_ratio),
                        "direct_pmat_multifrontal_overhead_estimate": float(direct_mf_overhead),
                        "direct_pmat_multifrontal_factor_entries_estimate": int(
                            direct_multifrontal_entries_estimate
                        ),
                        "direct_pmat_multifrontal_factor_nbytes_estimate": int(
                            direct_multifrontal_nbytes_estimate
                        ),
                    }
                )
                max_factor_nbytes = int(float(effective_factor_max_mb) * 1.0e6)
                if emit is not None and direct_multifrontal_nbytes_estimate > 0:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat "
                        "MUMPS-like fill estimate "
                        f"nnz={int(direct_pmat_nnz)} "
                        f"fill_ratio={float(direct_mf_fill_ratio):.3g} "
                        f"factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                        f"max_mb={float(effective_factor_max_mb):.3f}",
                    )
                if (
                    factor_kind_for_build in {"lu", "ilu"}
                    and direct_multifrontal_nbytes_estimate > 0
                    and max_factor_nbytes > 0
                    and direct_multifrontal_nbytes_estimate > max_factor_nbytes
                ):
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat exact factor "
                            "rejected by MUMPS-like fill guard "
                            f"factor_kind={factor_kind_for_build} "
                            f"factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                            f"max_mb={float(effective_factor_max_mb):.3f}",
                        )
                    return fallback_builder(
                        op=op,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                    )
                if (
                    direct_symbolic_prefill_estimate > 0
                    and max_factor_nbytes > 0
                    and direct_symbolic_prefill_estimate > max_factor_nbytes
                ):
                    auto_exact_cap_nbytes = int(float(auto_exact_rescue_max_mb) * 1.0e6)
                    auto_exact_candidate_ok = (
                        bool(auto_exact_rescue_enabled)
                        and not bool(explicit_factor_requested)
                        and factor_kind_for_build == "symbolic_block_lu_coarse"
                        and (
                            int(auto_exact_rescue_max_size) <= 0
                            or int(linear_size) <= int(auto_exact_rescue_max_size)
                        )
                        and (
                            int(auto_exact_rescue_max_factor_entries) <= 0
                            or int(direct_multifrontal_entries_estimate)
                            <= int(auto_exact_rescue_max_factor_entries)
                        )
                        and direct_multifrontal_nbytes_estimate > 0
                        and auto_exact_cap_nbytes > 0
                        and direct_multifrontal_nbytes_estimate <= auto_exact_cap_nbytes
                    )
                    if auto_exact_candidate_ok:
                        factor_kind_for_build = "lu"
                        effective_factor_max_mb = max(float(max_factor_mb), float(auto_exact_rescue_max_mb))
                        max_factor_nbytes = int(float(effective_factor_max_mb) * 1.0e6)
                        auto_exact_rescue_selected = True
                        direct_metadata.update(
                            {
                                "direct_pmat_auto_exact_rescue_selected": True,
                                "direct_pmat_auto_exact_rescue_reason": "symbolic_prefill_guard",
                                "direct_pmat_auto_exact_rescue_max_mb": float(auto_exact_rescue_max_mb),
                            }
                        )
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat symbolic factor "
                                "prefill exceeds default budget; trying exact LU rescue "
                                f"prefill_mb={float(direct_symbolic_prefill_estimate) / 1.0e6:.3f} "
                                f"exact_factor_mb={float(direct_multifrontal_nbytes_estimate) / 1.0e6:.3f} "
                                f"rescue_max_mb={float(auto_exact_rescue_max_mb):.3f}",
                            )
                    else:
                        if emit is not None:
                            emit(
                                1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat symbolic factor "
                            "rejected by prefill guard "
                            f"factor_kind={factor_kind_for_build} "
                            f"prefill_mb={float(direct_symbolic_prefill_estimate) / 1.0e6:.3f} "
                            f"max_mb={float(effective_factor_max_mb):.3f} "
                            f"exact_entries={int(direct_multifrontal_entries_estimate)} "
                            f"exact_entries_cap={int(auto_exact_rescue_max_factor_entries)} "
                            f"safety={float(direct_symbolic_prefill_safety):.3g}",
                        )
                        return fallback_builder(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                try:
                    _operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
                        matvec=_pc_matvec,
                        n=int(linear_size),
                        dtype=jnp.float64,
                        factor_dtype=factor_dtype,
                        pattern=None,
                        operator_bundle_override=direct_operator_bundle,
                        emit=emit,
                        default_factor_kind=str(factor_kind_for_build),
                        default_ilu_fill_factor=4.0,
                        default_ilu_drop_tol=1.0e-4,
                        default_permc_spec="MMD_AT_PLUS_A",
                        default_diag_pivot_thresh=0.0,
                        default_pattern_color_batch=8,
                        default_symbolic_ordering_kind=str(symbolic_ordering),
                        default_symbolic_block_size=int(symbolic_block_size),
                        default_symbolic_block_overlap=int(symbolic_block_overlap),
                        default_symbolic_coarse_max_cols=int(symbolic_coarse_max_cols),
                        default_symbolic_coarse_probe_cols=int(symbolic_coarse_probe_cols),
                        default_symbolic_coarse_damping=float(symbolic_coarse_damping),
                        default_symbolic_coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                        default_symbolic_schur_max_separator_cols=int(symbolic_schur_max_separator_cols),
                        default_symbolic_schur_tail_size=int(direct_metadata.get("direct_pmat_tail_size", 0)),
                        default_symbolic_schur_boundary_width=int(symbolic_schur_boundary_width),
                        default_symbolic_schur_high_degree_cols=int(symbolic_schur_high_degree_cols),
                        default_symbolic_schur_regularization_rel=float(symbolic_schur_regularization_rel),
                        default_symbolic_frontal_max_separator_cols=int(symbolic_frontal_max_separator_cols),
                        default_symbolic_frontal_tail_size=int(direct_metadata.get("direct_pmat_tail_size", 0)),
                        default_symbolic_frontal_boundary_width=int(symbolic_frontal_boundary_width),
                        default_symbolic_frontal_high_degree_cols=int(symbolic_frontal_high_degree_cols),
                        default_symbolic_frontal_max_superblock_size=int(symbolic_frontal_max_superblock_size),
                        default_symbolic_frontal_max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                        default_symbolic_frontal_min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                        default_symbolic_frontal_min_cross_separator_fraction=float(
                            symbolic_frontal_min_cross_separator_fraction
                        ),
                        default_symbolic_frontal_regularization_rel=float(symbolic_frontal_regularization_rel),
                        default_symbolic_frontal_max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                        default_symbolic_frontal_max_dense_rhs_cols_per_block=int(
                            symbolic_frontal_max_dense_rhs_cols_per_block
                        ),
                        default_symbolic_superblock_max_size=int(symbolic_superblock_max_size),
                        default_symbolic_superblock_max_blocks=int(symbolic_superblock_max_blocks),
                        default_symbolic_superblock_min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                        default_symbolic_superblock_min_retained_cross_fraction=float(
                            symbolic_superblock_min_retained_cross_fraction
                        ),
                        default_symbolic_superblock_regularization_rel=float(symbolic_superblock_regularization_rel),
                        default_symbolic_numeric_parallel_workers=int(symbolic_numeric_parallel_workers),
                        default_symbolic_max_permutation_size=int(symbolic_max_permutation_size),
                        default_monolithic_guard_enabled=not bool(auto_exact_rescue_selected),
                    )
                except Exception as exc:  # noqa: BLE001
                    if bool(auto_exact_rescue_selected):
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat exact LU rescue "
                                f"failed; skipping pattern-probe fallback ({type(exc).__name__}: {exc})",
                            )
                        return fallback_builder(
                            op=op,
                            reduce_full=reduce_full,
                            expand_reduced=expand_reduced,
                        )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: direct reduced Pmat factorization failed; "
                            f"falling back to pattern probe ({type(exc).__name__}: {exc})",
                        )
                    direct_operator_bundle = None
                    direct_metadata = {}

            if factor_bundle is None:
                if active_indices_use is None:
                    pattern = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
                        op_pc,
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                    )
                else:
                    pattern = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
                        op_pc,
                        active_indices_use,
                        preconditioner_x=int(preconditioner_x),
                        preconditioner_xi=int(preconditioner_xi),
                        preconditioner_species=int(preconditioner_species),
                        preconditioner_x_min_l=int(preconditioner_x_min_l),
                    )
                if emit is not None:
                    summary = summarize_v3_sparse_pattern(op_pc, pattern)
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu Pmat pattern "
                        f"scope={'active' if active_indices_use is not None else 'full'} "
                        f"shape={summary.shape} nnz={summary.nnz} avg_row_nnz={summary.avg_row_nnz:.3g}",
                    )

                _operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
                    matvec=_pc_matvec,
                    n=int(linear_size),
                    dtype=jnp.float64,
                    factor_dtype=factor_dtype,
                    pattern=pattern,
                    operator_bundle_override=None,
                    emit=emit,
                    default_factor_kind=str(default_factor_kind),
                    default_ilu_fill_factor=4.0,
                    default_ilu_drop_tol=1.0e-4,
                    default_permc_spec="MMD_AT_PLUS_A",
                    default_diag_pivot_thresh=0.0,
                    default_pattern_color_batch=8,
                    default_symbolic_ordering_kind=str(symbolic_ordering),
                    default_symbolic_block_size=int(symbolic_block_size),
                    default_symbolic_block_overlap=int(symbolic_block_overlap),
                    default_symbolic_coarse_max_cols=int(symbolic_coarse_max_cols),
                    default_symbolic_coarse_probe_cols=int(symbolic_coarse_probe_cols),
                    default_symbolic_coarse_damping=float(symbolic_coarse_damping),
                    default_symbolic_coarse_regularization_rel=float(symbolic_coarse_regularization_rel),
                    default_symbolic_schur_max_separator_cols=int(symbolic_schur_max_separator_cols),
                    default_symbolic_schur_tail_size=0,
                    default_symbolic_schur_boundary_width=int(symbolic_schur_boundary_width),
                    default_symbolic_schur_high_degree_cols=int(symbolic_schur_high_degree_cols),
                    default_symbolic_schur_regularization_rel=float(symbolic_schur_regularization_rel),
                    default_symbolic_frontal_max_separator_cols=int(symbolic_frontal_max_separator_cols),
                    default_symbolic_frontal_tail_size=0,
                    default_symbolic_frontal_boundary_width=int(symbolic_frontal_boundary_width),
                    default_symbolic_frontal_high_degree_cols=int(symbolic_frontal_high_degree_cols),
                    default_symbolic_frontal_max_superblock_size=int(symbolic_frontal_max_superblock_size),
                    default_symbolic_frontal_max_superblock_blocks=int(symbolic_frontal_max_superblock_blocks),
                    default_symbolic_frontal_min_cross_nnz=int(symbolic_frontal_min_cross_nnz),
                    default_symbolic_frontal_min_cross_separator_fraction=float(
                        symbolic_frontal_min_cross_separator_fraction
                    ),
                    default_symbolic_frontal_regularization_rel=float(symbolic_frontal_regularization_rel),
                    default_symbolic_frontal_max_dense_rhs_entries=int(symbolic_frontal_max_dense_rhs_entries),
                    default_symbolic_frontal_max_dense_rhs_cols_per_block=int(
                        symbolic_frontal_max_dense_rhs_cols_per_block
                    ),
                    default_symbolic_superblock_max_size=int(symbolic_superblock_max_size),
                    default_symbolic_superblock_max_blocks=int(symbolic_superblock_max_blocks),
                    default_symbolic_superblock_min_cross_nnz=int(symbolic_superblock_min_cross_nnz),
                    default_symbolic_superblock_min_retained_cross_fraction=float(
                        symbolic_superblock_min_retained_cross_fraction
                    ),
                    default_symbolic_superblock_regularization_rel=float(symbolic_superblock_regularization_rel),
                    default_symbolic_numeric_parallel_workers=int(symbolic_numeric_parallel_workers),
                    default_symbolic_max_permutation_size=int(symbolic_max_permutation_size),
                )
        except Exception as exc:  # noqa: BLE001
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu disabled after "
                    f"{type(exc).__name__}: {exc}",
                )
            return fallback_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )

        physics_coarse_metadata: dict[str, object] = {}
        if (
            str(getattr(factor_bundle, "kind", "")) == "symbolic_block_lu_coarse"
            and bool(symbolic_physics_coarse_enabled)
            and active_indices_use is not None
            and direct_operator_bundle is not None
        ):
            try:
                physics_basis, physics_names = _build_rhsmode23_direct_pmat_physics_coarse_basis(
                    op=op_pc,
                    active_indices=active_indices_use,
                    max_cols=int(symbolic_physics_coarse_max_cols),
                    base_factor_bundle=factor_bundle,
                )
                if physics_basis is not None and int(getattr(physics_basis, "shape", (0, 0))[1]) > 0:
                    factor_bundle = wrap_sparse_factor_with_coarse_correction(
                        factor_bundle,
                        physics_basis,
                        damping=float(symbolic_coarse_damping),
                        regularization_rel=float(symbolic_coarse_regularization_rel),
                    )
                    physics_coarse_metadata = {
                        "symbolic_physics_coarse": True,
                        "symbolic_physics_coarse_cols": int(physics_basis.shape[1]),
                        "symbolic_physics_coarse_nnz": int(physics_basis.nnz),
                        "symbolic_physics_coarse_labels": tuple(str(v) for v in physics_names),
                    }
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                            f"physics coarse basis cols={int(physics_basis.shape[1])} "
                            f"nnz={int(physics_basis.nnz)}",
                        )
            except Exception as exc:  # noqa: BLE001
                physics_coarse_metadata = {
                    "symbolic_physics_coarse": False,
                    "symbolic_physics_coarse_error": f"{type(exc).__name__}: {exc}",
                }
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                        f"physics coarse disabled ({type(exc).__name__}: {exc})",
                    )

        factor_nbytes = getattr(factor_bundle, "factor_nbytes_estimate", None)
        if (
            float(effective_factor_max_mb) > 0.0
            and factor_nbytes is not None
            and int(factor_nbytes) > int(float(effective_factor_max_mb) * 1.0e6)
        ):
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu factor rejected by budget "
                    f"factor_mb={float(factor_nbytes) / 1.0e6:.3f} max_mb={float(effective_factor_max_mb):.3f}",
                )
            return fallback_builder(
                op=op,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        symbolic_metadata: dict[str, object] = {}
        factor_operator = getattr(factor_bundle, "operator", None)
        factor_matrix = None if factor_operator is None else getattr(factor_operator, "matrix", None)
        factor_kind_for_admission = str(getattr(factor_bundle, "kind", ""))
        direct_admission_required = (
            bool(direct_admission_enabled)
            and factor_kind_for_admission in {"lu", "ilu"}
            and (bool(auto_exact_rescue_selected) or bool(direct_admission_explicit_enabled))
        )
        if bool(direct_admission_required):
            direct_admission = admit_sparse_factor_against_operator(
                factor_operator if factor_operator is not None else factor_matrix,
                factor_bundle,
                probe_count=int(symbolic_admission_probes),
                max_relative_residual=float(symbolic_admission_max_rel),
                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
            )
            symbolic_metadata["direct_admission"] = direct_admission.to_dict()
            if emit is not None:
                emit(
                    1,
                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu exact factor admission "
                    f"{'accepted' if direct_admission.accepted else 'rejected'} "
                    f"max_rel={float(direct_admission.max_relative_residual):.3e} "
                    f"median_rel={float(direct_admission.median_relative_residual):.3e} "
                    f"min_improvement={float(direct_admission.min_improvement_vs_identity):.3e} "
                    f"probes={int(direct_admission.probe_count)}",
                )
            if not bool(direct_admission.accepted):
                return fallback_builder(
                    op=op,
                    reduce_full=reduce_full,
                    expand_reduced=expand_reduced,
                )
        if factor_kind_for_admission in {
            "symbolic_block_lu",
            "symbolic_block_lu_coarse",
            "symbolic_block_schur_lu",
            "symbolic_frontal_schur_lu",
            "symbolic_superblock_lu",
        } and bool(symbolic_admission_enabled):
            admission = admit_sparse_factor_against_operator(
                factor_operator if factor_operator is not None else factor_matrix,
                factor_bundle,
                probe_count=int(symbolic_admission_probes),
                max_relative_residual=float(symbolic_admission_max_rel),
                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
            )
            admission_metadata = admission.to_dict()
            symbolic_metadata["symbolic_admission"] = admission_metadata
            if emit is not None:
                admission_label = factor_kind_for_admission
                emit(
                    1,
                    f"solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu {admission_label} admission "
                    f"{'accepted' if admission.accepted else 'rejected'} "
                    f"max_rel={float(admission.max_relative_residual):.3e} "
                    f"median_rel={float(admission.median_relative_residual):.3e} "
                    f"min_improvement={float(admission.min_improvement_vs_identity):.3e} "
                    f"probes={int(admission.probe_count)}",
                )
            if not bool(admission.accepted):
                rescue_bundle = None
                rescue_metadata: dict[str, object] = {}
                if (
                    bool(symbolic_admission_rescue_lu)
                    and factor_operator is not None
                    and getattr(factor_operator, "matrix", None) is not None
                ):
                    try:
                        rescue_candidate = factorize_host_sparse_operator(
                            factor_operator,
                            kind="lu",
                            permc_spec="MMD_AT_PLUS_A",
                            diag_pivot_thresh=0.0,
                        )
                        rescue_nbytes = getattr(rescue_candidate, "factor_nbytes_estimate", None)
                        rescue_budget_ok = (
                            float(symbolic_admission_rescue_lu_max_mb) <= 0.0
                            or rescue_nbytes is None
                            or int(rescue_nbytes) <= int(float(symbolic_admission_rescue_lu_max_mb) * 1.0e6)
                        )
                        if rescue_budget_ok:
                            rescue_admission = admit_sparse_factor_against_operator(
                                factor_operator,
                                rescue_candidate,
                                probe_count=int(symbolic_admission_probes),
                                max_relative_residual=float(symbolic_admission_max_rel),
                                min_improvement_vs_identity=float(symbolic_admission_min_improvement),
                            )
                            rescue_metadata = {
                                "symbolic_admission_rescue_lu": True,
                                "symbolic_admission_rescue_lu_factor_nbytes_estimate": (
                                    None if rescue_nbytes is None else int(rescue_nbytes)
                                ),
                                "symbolic_admission_rescue_lu_factor_nnz_estimate": (
                                    None
                                    if getattr(rescue_candidate, "factor_nnz_estimate", None) is None
                                    else int(rescue_candidate.factor_nnz_estimate)
                                ),
                                "symbolic_admission_rescue_lu_factor_s": (
                                    None
                                    if getattr(rescue_candidate, "factor_s", None) is None
                                    else float(rescue_candidate.factor_s)
                                ),
                                "symbolic_admission_rescue_lu_admission": rescue_admission.to_dict(),
                            }
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                                    "symbolic admission rescue lu "
                                    f"{'accepted' if rescue_admission.accepted else 'rejected'} "
                                    f"max_rel={float(rescue_admission.max_relative_residual):.3e} "
                                    f"factor_mb={float(rescue_nbytes or 0) / 1.0e6:.3f}",
                                )
                            if bool(rescue_admission.accepted):
                                rescue_bundle = rescue_candidate
                        else:
                            rescue_metadata = {
                                "symbolic_admission_rescue_lu": False,
                                "symbolic_admission_rescue_lu_reason": "factor_budget",
                                "symbolic_admission_rescue_lu_factor_nbytes_estimate": (
                                    None if rescue_nbytes is None else int(rescue_nbytes)
                                ),
                                "symbolic_admission_rescue_lu_max_mb": float(symbolic_admission_rescue_lu_max_mb),
                            }
                    except Exception as exc:  # noqa: BLE001
                        rescue_metadata = {
                            "symbolic_admission_rescue_lu": False,
                            "symbolic_admission_rescue_lu_error": f"{type(exc).__name__}: {exc}",
                        }
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu "
                                f"symbolic admission rescue lu failed ({type(exc).__name__}: {exc})",
                            )
                symbolic_metadata.update(rescue_metadata)
                if rescue_bundle is None:
                    return fallback_builder(
                        op=op,
                        reduce_full=reduce_full,
                        expand_reduced=expand_reduced,
                    )
                factor_bundle = rescue_bundle
                factor_nbytes = getattr(factor_bundle, "factor_nbytes_estimate", None)
                factor_kind_for_admission = str(getattr(factor_bundle, "kind", ""))
        inner_factor_metadata = getattr(getattr(factor_bundle, "factor", None), "metadata", None)
        if isinstance(inner_factor_metadata, dict):
            symbolic_metadata["symbolic_factor_metadata"] = dict(inner_factor_metadata)
        if factor_matrix is not None:
            try:
                symbolic_analysis = getattr(getattr(factor_bundle, "factor", None), "analysis", None)
                if symbolic_analysis is None:
                    symbolic_analysis = analyze_sparse_symbolic_structure(
                        factor_matrix,
                        ordering_kind=str(symbolic_ordering),
                        block_size_target=int(symbolic_block_size),
                        max_permutation_size=int(symbolic_max_permutation_size),
                    )
                symbolic_metadata.update({
                    "symbolic": symbolic_analysis.to_dict(),
                    "symbolic_cache_key": symbolic_analysis.cache_key(),
                    "symbolic_factor_coarse_size": int(getattr(getattr(factor_bundle, "factor", None), "coarse_size", 0)),
                    "symbolic_factor_overlap_size": int(getattr(getattr(factor_bundle, "factor", None), "overlap_size", 0)),
                })
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_transport_matrix_linear_gmres: fp_fortran_reduced_lu symbolic analysis "
                        f"ordering={symbolic_analysis.ordering_kind} "
                        f"pattern_hash={symbolic_analysis.pattern_hash[:12]} "
                        f"nnz={int(symbolic_analysis.nnz)} "
                        f"bandwidth={int(symbolic_analysis.bandwidth)}->{int(symbolic_analysis.permuted_bandwidth)} "
                        f"profile={int(symbolic_analysis.profile)}->{int(symbolic_analysis.permuted_profile)} "
                        f"blocks={int(symbolic_analysis.block_count)}x<= {int(symbolic_analysis.block_size_max)}",
                    )
            except Exception as exc:  # noqa: BLE001
                symbolic_metadata = {
                    "symbolic_error": f"{type(exc).__name__}: {exc}",
                    "symbolic_ordering": str(symbolic_ordering),
                    "symbolic_block_size": int(symbolic_block_size),
                    "symbolic_max_permutation_size": int(symbolic_max_permutation_size),
                    "symbolic_block_overlap": int(symbolic_block_overlap),
                    "symbolic_coarse_max_cols": int(symbolic_coarse_max_cols),
                    "symbolic_coarse_probe_cols": int(symbolic_coarse_probe_cols),
                    "symbolic_coarse_damping": float(symbolic_coarse_damping),
                    "symbolic_coarse_regularization_rel": float(symbolic_coarse_regularization_rel),
                }

        metadata = {
            "factor_kind": str(factor_bundle.kind),
            "factor_dtype": str(factor_dtype.name),
            "factor_nbytes_estimate": None if factor_nbytes is None else int(factor_nbytes),
            "factor_nnz_estimate": None
            if getattr(factor_bundle, "factor_nnz_estimate", None) is None
            else int(factor_bundle.factor_nnz_estimate),
            "factor_s": None if getattr(factor_bundle, "factor_s", None) is None else float(factor_bundle.factor_s),
            "linear_size": int(linear_size),
            "active_dof": bool(active_indices_use is not None),
            "preconditioner_x": int(preconditioner_x),
            "preconditioner_xi": int(preconditioner_xi),
            "preconditioner_species": int(preconditioner_species),
            "preconditioner_x_min_l": int(preconditioner_x_min_l),
            "keeps_theta_zeta": bool(keep_theta_zeta),
            "shift": float(pc_shift),
            "direct_pmat_enabled": bool(direct_pmat_enabled),
            "factor_max_mb": float(max_factor_mb),
            "effective_factor_max_mb": float(effective_factor_max_mb),
            "host_memory_mb": None if host_memory_mb is None else float(host_memory_mb),
            "auto_exact_rescue_enabled": bool(auto_exact_rescue_enabled),
            "auto_exact_rescue_ram_fraction": float(auto_exact_rescue_ram_fraction),
            "auto_exact_rescue_max_mb": float(auto_exact_rescue_max_mb),
            "auto_exact_rescue_max_size": int(auto_exact_rescue_max_size),
            "auto_exact_rescue_max_factor_entries": int(auto_exact_rescue_max_factor_entries),
            "auto_exact_rescue_selected": bool(auto_exact_rescue_selected),
            "direct_admission_enabled": bool(direct_admission_enabled),
            "direct_admission_explicit_enabled": bool(direct_admission_explicit_enabled),
            "direct_admission_required": bool(direct_admission_required),
            "symbolic_ordering": str(symbolic_ordering),
            "symbolic_block_size": int(symbolic_block_size),
            "symbolic_block_overlap": int(symbolic_block_overlap),
            "symbolic_coarse_max_cols": int(symbolic_coarse_max_cols),
            "symbolic_coarse_probe_cols": int(symbolic_coarse_probe_cols),
            "symbolic_coarse_damping": float(symbolic_coarse_damping),
            "symbolic_coarse_regularization_rel": float(symbolic_coarse_regularization_rel),
            "symbolic_physics_coarse_enabled": bool(symbolic_physics_coarse_enabled),
            "symbolic_physics_coarse_max_cols": int(symbolic_physics_coarse_max_cols),
            "symbolic_schur_max_separator_cols": int(symbolic_schur_max_separator_cols),
            "symbolic_schur_boundary_width": int(symbolic_schur_boundary_width),
            "symbolic_schur_high_degree_cols": int(symbolic_schur_high_degree_cols),
            "symbolic_schur_regularization_rel": float(symbolic_schur_regularization_rel),
            "symbolic_frontal_max_separator_cols": int(symbolic_frontal_max_separator_cols),
            "symbolic_frontal_boundary_width": int(symbolic_frontal_boundary_width),
            "symbolic_frontal_high_degree_cols": int(symbolic_frontal_high_degree_cols),
            "symbolic_frontal_max_superblock_size": int(symbolic_frontal_max_superblock_size),
            "symbolic_frontal_max_superblock_blocks": int(symbolic_frontal_max_superblock_blocks),
            "symbolic_frontal_min_cross_nnz": int(symbolic_frontal_min_cross_nnz),
            "symbolic_frontal_min_cross_separator_fraction": float(symbolic_frontal_min_cross_separator_fraction),
            "symbolic_frontal_regularization_rel": float(symbolic_frontal_regularization_rel),
            "symbolic_frontal_max_dense_rhs_entries": int(symbolic_frontal_max_dense_rhs_entries),
            "symbolic_frontal_max_dense_rhs_cols_per_block": int(symbolic_frontal_max_dense_rhs_cols_per_block),
            "symbolic_superblock_max_size": int(symbolic_superblock_max_size),
            "symbolic_superblock_max_blocks": int(symbolic_superblock_max_blocks),
            "symbolic_superblock_min_cross_nnz": int(symbolic_superblock_min_cross_nnz),
            "symbolic_superblock_min_retained_cross_fraction": float(symbolic_superblock_min_retained_cross_fraction),
            "symbolic_superblock_regularization_rel": float(symbolic_superblock_regularization_rel),
            "symbolic_numeric_parallel_workers": int(symbolic_numeric_parallel_workers),
            "symbolic_max_permutation_size": int(symbolic_max_permutation_size),
            "symbolic_admission_enabled": bool(symbolic_admission_enabled),
            "symbolic_admission_max_rel": float(symbolic_admission_max_rel),
            "symbolic_admission_min_improvement": float(symbolic_admission_min_improvement),
            "symbolic_admission_probes": int(symbolic_admission_probes),
            "symbolic_admission_rescue_lu_enabled": bool(symbolic_admission_rescue_lu),
            "symbolic_admission_rescue_lu_max_mb": float(symbolic_admission_rescue_lu_max_mb),
        }
        metadata.update(direct_metadata)
        metadata.update(symbolic_metadata)
        metadata.update(physics_coarse_metadata)
        cached = _TransportFpFortranReducedLuPrecondCache(
            factor_bundle=factor_bundle,
            linear_size=int(linear_size),
            metadata=metadata,
        )
        _TRANSPORT_FP_FORTRAN_REDUCED_LU_PRECOND_CACHE[cache_key] = cached

    factor_bundle = cached.factor_bundle
    linear_size_use = int(cached.linear_size)

    def _solve_host(rhs_host: np.ndarray) -> np.ndarray:
        rhs_np = np.asarray(rhs_host, dtype=np.float64).reshape((linear_size_use,))
        try:
            sol = np.asarray(factor_bundle.solve(rhs_np), dtype=np.float64).reshape((linear_size_use,))
        except Exception:
            sol = rhs_np
        finite = np.isfinite(sol)
        if not np.all(finite):
            sol = np.where(finite, sol, 0.0)
        return sol.astype(np.float64, copy=False)

    def _apply(v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.asarray(v, dtype=jnp.float64)
        return jax.pure_callback(
            _solve_host,
            jax.ShapeDtypeStruct((linear_size_use,), jnp.float64),
            v,
        )

    try:
        setattr(_apply, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", dict(cached.metadata))
    except Exception:
        pass
    return _apply
