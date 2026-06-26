"""Finalize one RHSMode=2/3 transport solve result.

The transport driver has several solver branches, but after a branch accepts a
candidate it must always perform the same bookkeeping: recover the full state,
apply optional constraint projection, compute the true residual when needed,
store diagnostics, update recycle bases, and optionally emit KSP iteration
statistics.  Keeping this logic here makes solver-policy changes less risky.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any
import os

from jax import tree_util as jtu
import jax.numpy as jnp
import numpy as np

from sfincs_jax.operators.profile_response.system import V3FullSystemOperator
from sfincs_jax.problems.transport_matrix.diagnostics import (
    v3_transport_diagnostics_vm_only_batch_jit,
    v3_transport_diagnostics_vm_only_batch_op0_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit,
    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit,
    v3_transport_diagnostics_vm_only_batch_op0_remat_jit,
    v3_transport_diagnostics_vm_only_batch_remat_jit,
    v3_transport_diagnostics_vm_only_precompute,
    v3_transport_matrix_from_flux_arrays,
)
from sfincs_jax.outputs.transport import TransportStreamingOutputAccumulator
from sfincs_jax.solvers.preconditioning import project_constraint_scheme1_nullspace_solution


EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class TransportKSPIterationRequest:
    """Inputs needed to optionally replay a small KSP solve for diagnostics."""

    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray]
    b_vec: jnp.ndarray
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None
    x0_vec: jnp.ndarray | None
    tol_val: float
    atol_val: float
    restart_val: int
    maxiter_val: int | None
    precond_side: str
    solver_kind: str


@dataclass(frozen=True)
class V3TransportMatrixSolveResult:
    """Result of a RHSMode=2/3 transport-matrix or monoenergetic solve."""

    op0: Any
    transport_matrix: jnp.ndarray
    state_vectors_by_rhs: dict[int, jnp.ndarray]
    residual_norms_by_rhs: dict[int, jnp.ndarray]
    fsab_flow: jnp.ndarray
    particle_flux_vm_psi_hat: jnp.ndarray
    heat_flux_vm_psi_hat: jnp.ndarray
    elapsed_time_s: jnp.ndarray
    transport_output_fields: dict[str, object] | None = None
    rhs_norms_by_rhs: dict[int, jnp.ndarray] | None = None
    active_size: int | None = None
    use_active_dof_mode: bool | None = None
    solver_kinds_by_rhs: dict[int, str] | None = None
    solve_methods_by_rhs: dict[int, str] | None = None
    preconditioner_kind: str | None = None
    strong_preconditioner_kind: str | None = None


@dataclass(frozen=True)
class TransportPostsolveDiagnostics:
    """Flux arrays, optional output fields, and matrix assembled after transport solves."""

    transport_matrix: jnp.ndarray
    particle_flux_vm_psi_hat: jnp.ndarray
    heat_flux_vm_psi_hat: jnp.ndarray
    fsab_flow: jnp.ndarray
    transport_output_fields: dict[str, np.ndarray] | None


@dataclass
class TransportRHSFinalizationContext:
    """Mutable solve-loop state shared by transport RHS finalization helpers."""

    state_vectors: MutableMapping[int, jnp.ndarray]
    residual_norms: MutableMapping[int, jnp.ndarray]
    solver_kinds_by_rhs: MutableMapping[int, str]
    solve_methods_by_rhs: MutableMapping[int, str]
    store_state_vectors: bool
    stream_diagnostics: bool
    collect_transport_outputs: Callable[[int, jnp.ndarray], None] | None
    recycle_state: Any | None
    apply_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    emit_iteration_stats: Callable[..., None]
    emit: EmitFn | None
    iter_stats_enabled: bool
    iter_stats_max_size: int | None
    atol: float
    maxiter: int | None
    precond_side: str

    def ksp_request(
        self,
        matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
        b_vec: jnp.ndarray,
        precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
        x0_vec: jnp.ndarray | None,
        *,
        tol_val: float,
        restart_val: int,
        solver_kind: str,
    ) -> TransportKSPIterationRequest:
        """Build a KSP diagnostic request using solve-loop defaults."""
        return TransportKSPIterationRequest(
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=float(tol_val),
            atol_val=float(self.atol),
            restart_val=int(restart_val),
            maxiter_val=self.maxiter,
            precond_side=str(self.precond_side),
            solver_kind=str(solver_kind),
        )


@dataclass(frozen=True)
class TransportConstraintNullspaceProjector:
    """Apply constraintScheme=1 nullspace projection only for policy-selected RHSs."""

    op: Any
    policy: Any
    enabled_env_var: str = "SFINCS_JAX_TRANSPORT_PROJECT_NULLSPACE"
    project_solution: Callable[..., jnp.ndarray] = project_constraint_scheme1_nullspace_solution

    def project(
        self,
        x_vec: jnp.ndarray,
        *,
        which_rhs: int,
        op_matvec: Any,
        rhs_vec: jnp.ndarray,
    ) -> jnp.ndarray:
        if not self.policy.projection_candidate(int(which_rhs)):
            return x_vec
        return self.project_solution(
            op=self.op,
            x_vec=x_vec,
            rhs_vec=rhs_vec,
            matvec_op=op_matvec,
            enabled_env_var=self.enabled_env_var,
        )


@dataclass(frozen=True)
class TransportRHSFinalizationResult:
    """Full-space result values after final transport RHS bookkeeping."""

    x_full: jnp.ndarray
    ax_full: jnp.ndarray
    residual_norm: jnp.ndarray


def compute_transport_postsolve_diagnostics(
    *,
    op0: V3FullSystemOperator,
    geom: Any,
    state_vectors: Mapping[int, jnp.ndarray],
    which_rhs_values: Sequence[int],
    stream_diagnostics: bool,
    streaming_outputs: TransportStreamingOutputAccumulator | None,
    use_diag_op0: bool,
    diag_op_by_index: Sequence[V3FullSystemOperator] | None,
    emit: Callable[[int, str], None] | None = None,
) -> TransportPostsolveDiagnostics:
    """Compute batched or streamed transport diagnostics after all whichRHS solves.

    The solve loop owns state-vector generation. This helper owns only the
    memory policy for post-solve diagnostics: streamed buffers when available,
    otherwise batched or chunked JAX diagnostics with optional rematerialization
    and precomputed fixed-operator geometry factors.
    """

    if emit is not None:
        emit(0, "solve_v3_transport_matrix_linear_gmres: computing whichRHS diagnostics (batched)")

    n_rhs = int(len(which_rhs_values))
    if stream_diagnostics:
        if streaming_outputs is None:
            raise RuntimeError("streaming transport diagnostics requested without an accumulator")
        diag_pf_jnp, diag_hf_jnp, diag_flow_jnp = streaming_outputs.diagnostic_flux_arrays()
        transport_output_fields = streaming_outputs.output_fields()
    else:
        remat_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS", "").strip().lower()
        if remat_env in {"1", "true", "yes", "on"}:
            use_remat_diag = True
        elif remat_env in {"0", "false", "no", "off"}:
            use_remat_diag = False
        else:
            remat_min_env = os.environ.get("SFINCS_JAX_REMAT_TRANSPORT_DIAGNOSTICS_MIN", "").strip()
            try:
                remat_min = int(remat_min_env) if remat_min_env else 20000
            except ValueError:
                remat_min = 20000
            use_remat_diag = int(op0.total_size) * int(n_rhs) >= remat_min

        diag_chunk_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_CHUNK", "").strip()
        try:
            diag_chunk = int(diag_chunk_env) if diag_chunk_env else None
        except ValueError:
            diag_chunk = None
        if diag_chunk is None or int(diag_chunk) <= 0:
            diag_chunk = 0
        if diag_chunk == 0 and int(op0.total_size) * int(n_rhs) >= 200_000:
            diag_chunk = 4

        if use_diag_op0:
            precompute_env = os.environ.get("SFINCS_JAX_TRANSPORT_DIAG_PRECOMPUTE", "").strip().lower()
            use_precompute = precompute_env not in {"0", "false", "no", "off"}
            if use_precompute:
                precomputed = v3_transport_diagnostics_vm_only_precompute(op0)
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_precomputed_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_precomputed_jit
                )
            else:
                precomputed = None
                diag_fn = (
                    v3_transport_diagnostics_vm_only_batch_op0_remat_jit
                    if use_remat_diag
                    else v3_transport_diagnostics_vm_only_batch_op0_jit
                )
        else:
            if diag_op_by_index is None:
                raise RuntimeError("transport diagnostics with RHS operators require diag_op_by_index")
            diag_op_stack = jtu.tree_map(lambda *xs: jnp.stack(xs, axis=0), *diag_op_by_index)
            diag_fn = (
                v3_transport_diagnostics_vm_only_batch_remat_jit
                if use_remat_diag
                else v3_transport_diagnostics_vm_only_batch_jit
            )

        if diag_chunk <= 0 or int(diag_chunk) >= int(n_rhs):
            x_stack = jnp.stack([state_vectors[int(which_rhs)] for which_rhs in which_rhs_values], axis=0)
            if use_diag_op0:
                if use_precompute:
                    diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack)
                else:
                    diag_stack = diag_fn(op0=op0, x_full_stack=x_stack)
            else:
                diag_stack = diag_fn(op_stack=diag_op_stack, x_full_stack=x_stack)
            diag_pf_jnp = jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))
            diag_hf_jnp = jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0))
            diag_flow_jnp = jnp.transpose(diag_stack.fsab_flow, (1, 0))
        else:
            n_species = int(op0.n_species)
            diag_pf_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            diag_hf_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            diag_flow_arr = np.zeros((n_species, n_rhs), dtype=np.float64)
            for start in range(0, n_rhs, int(diag_chunk)):
                end = min(n_rhs, start + int(diag_chunk))
                rhs_chunk = which_rhs_values[start:end]
                x_stack_chunk = jnp.stack([state_vectors[int(which_rhs)] for which_rhs in rhs_chunk], axis=0)
                if use_diag_op0:
                    if use_precompute:
                        diag_stack = diag_fn(op0=op0, precomputed=precomputed, x_full_stack=x_stack_chunk)
                    else:
                        diag_stack = diag_fn(op0=op0, x_full_stack=x_stack_chunk)
                else:
                    op_chunk = jtu.tree_map(lambda arr: arr[start:end], diag_op_stack)
                    diag_stack = diag_fn(op_stack=op_chunk, x_full_stack=x_stack_chunk)
                diag_pf_arr[:, start:end] = np.asarray(
                    jnp.transpose(diag_stack.particle_flux_vm_psi_hat, (1, 0))
                )
                diag_hf_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.heat_flux_vm_psi_hat, (1, 0)))
                diag_flow_arr[:, start:end] = np.asarray(jnp.transpose(diag_stack.fsab_flow, (1, 0)))
            diag_pf_jnp = jnp.asarray(diag_pf_arr, dtype=jnp.float64)
            diag_hf_jnp = jnp.asarray(diag_hf_arr, dtype=jnp.float64)
            diag_flow_jnp = jnp.asarray(diag_flow_arr, dtype=jnp.float64)
        transport_output_fields = None

    tm = v3_transport_matrix_from_flux_arrays(
        op=op0,
        geom=geom,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        fsab_flow=diag_flow_jnp,
    )
    return TransportPostsolveDiagnostics(
        transport_matrix=tm,
        particle_flux_vm_psi_hat=diag_pf_jnp,
        heat_flux_vm_psi_hat=diag_hf_jnp,
        fsab_flow=diag_flow_jnp,
        transport_output_fields=transport_output_fields,
    )


def finalize_reduced_transport_rhs(
    *,
    context: TransportRHSFinalizationContext,
    which_rhs: int,
    result: Any,
    rhs_full: jnp.ndarray,
    op_matvec: Any,
    solver_kind: str,
    solve_method: str,
    dense_used: bool,
    expand_reduced: Callable[[jnp.ndarray], jnp.ndarray],
    reduce_full: Callable[[jnp.ndarray], jnp.ndarray],
    maybe_project_constraint_nullspace: Callable[..., jnp.ndarray],
    ksp_request: TransportKSPIterationRequest | None,
    accepted_x_full: jnp.ndarray | None = None,
    accepted_ax_full: jnp.ndarray | None = None,
    accepted_residual_norm: jnp.ndarray | None = None,
) -> TransportRHSFinalizationResult:
    """Finalize an active-DOF transport solve and update loop bookkeeping."""
    which_rhs_i = int(which_rhs)
    if accepted_x_full is not None and accepted_ax_full is not None and accepted_residual_norm is not None:
        x_full = accepted_x_full
        ax_full = accepted_ax_full
        residual_norm = accepted_residual_norm
    else:
        x_full = expand_reduced(result.x)
        x_full = maybe_project_constraint_nullspace(
            x_full,
            which_rhs=which_rhs_i,
            op_matvec=op_matvec,
            rhs_vec=rhs_full,
        )
        ax_full = context.apply_operator(op_matvec, x_full)
        residual_norm = jnp.linalg.norm(ax_full - rhs_full)

    if context.store_state_vectors:
        context.state_vectors[which_rhs_i] = x_full
    context.residual_norms[which_rhs_i] = residual_norm
    context.solver_kinds_by_rhs[which_rhs_i] = str(solver_kind)
    context.solve_methods_by_rhs[which_rhs_i] = str(solve_method)
    if context.stream_diagnostics and context.collect_transport_outputs is not None:
        context.collect_transport_outputs(which_rhs_i, x_full)
    if context.recycle_state is not None:
        context.recycle_state.append_reduced(
            result.x,
            reduce_full(ax_full),
            x_full=x_full,
            ax_full=ax_full,
        )
    _maybe_emit_ksp_iteration_stats(
        context=context,
        which_rhs=which_rhs_i,
        dense_used=bool(dense_used),
        request=ksp_request,
    )
    return TransportRHSFinalizationResult(
        x_full=x_full,
        ax_full=ax_full,
        residual_norm=residual_norm,
    )


def finalize_full_transport_rhs(
    *,
    context: TransportRHSFinalizationContext,
    which_rhs: int,
    result: Any,
    rhs_full: jnp.ndarray,
    op_matvec: Any,
    solver_kind: str,
    solve_method: str,
    dense_used: bool,
    projection_needed: bool,
    residual_vec: jnp.ndarray | None,
    maybe_project_constraint_nullspace: Callable[..., jnp.ndarray],
    ksp_request: TransportKSPIterationRequest | None,
) -> TransportRHSFinalizationResult:
    """Finalize a full-space transport solve and update loop bookkeeping."""
    which_rhs_i = int(which_rhs)
    x_full = result.x
    if projection_needed:
        x_full = maybe_project_constraint_nullspace(
            x_full,
            which_rhs=which_rhs_i,
            op_matvec=op_matvec,
            rhs_vec=rhs_full,
        )
    if context.store_state_vectors:
        context.state_vectors[which_rhs_i] = x_full
    if (not projection_needed) and residual_vec is not None and residual_vec.shape == rhs_full.shape:
        ax_full = rhs_full - residual_vec
        residual_norm = result.residual_norm
    else:
        ax_full = context.apply_operator(op_matvec, x_full)
        residual_vec = ax_full - rhs_full
        residual_norm = jnp.linalg.norm(residual_vec)
    context.residual_norms[which_rhs_i] = residual_norm
    context.solver_kinds_by_rhs[which_rhs_i] = str(solver_kind)
    context.solve_methods_by_rhs[which_rhs_i] = str(solve_method)
    if context.stream_diagnostics and context.collect_transport_outputs is not None:
        context.collect_transport_outputs(which_rhs_i, x_full)
    if context.recycle_state is not None:
        context.recycle_state.append_full(x_full, ax_full)
    _maybe_emit_ksp_iteration_stats(
        context=context,
        which_rhs=which_rhs_i,
        dense_used=bool(dense_used),
        request=ksp_request,
    )
    return TransportRHSFinalizationResult(
        x_full=x_full,
        ax_full=ax_full,
        residual_norm=residual_norm,
    )


def _maybe_emit_ksp_iteration_stats(
    *,
    context: TransportRHSFinalizationContext,
    which_rhs: int,
    dense_used: bool,
    request: TransportKSPIterationRequest | None,
) -> None:
    if dense_used or request is None:
        return
    context.emit_iteration_stats(
        which_rhs=int(which_rhs),
        matvec_fn=request.matvec_fn,
        b_vec=request.b_vec,
        precond_fn=request.precond_fn,
        x0_vec=request.x0_vec,
        tol_val=float(request.tol_val),
        atol_val=float(request.atol_val),
        restart_val=int(request.restart_val),
        maxiter_val=request.maxiter_val,
        precond_side=str(request.precond_side),
        solver_kind=str(request.solver_kind),
        emit=context.emit,
        enabled=bool(context.iter_stats_enabled),
        max_size=context.iter_stats_max_size,
    )


__all__ = [
    "TransportConstraintNullspaceProjector",
    "TransportKSPIterationRequest",
    "TransportPostsolveDiagnostics",
    "TransportRHSFinalizationContext",
    "TransportRHSFinalizationResult",
    "V3TransportMatrixSolveResult",
    "compute_transport_postsolve_diagnostics",
    "finalize_full_transport_rhs",
    "finalize_reduced_transport_rhs",
]
