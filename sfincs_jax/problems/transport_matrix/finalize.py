"""Finalize one RHSMode=2/3 transport solve result.

The transport driver has several solver branches, but after a branch accepts a
candidate it must always perform the same bookkeeping: recover the full state,
apply optional constraint projection, compute the true residual when needed,
store diagnostics, update recycle bases, and optionally emit KSP iteration
statistics.  Keeping this logic here makes solver-policy changes less risky.
"""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from sfincs_jax.constraint_projection import project_constraint_scheme1_nullspace_solution


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
    "TransportRHSFinalizationContext",
    "TransportRHSFinalizationResult",
    "finalize_full_transport_rhs",
    "finalize_reduced_transport_rhs",
]
