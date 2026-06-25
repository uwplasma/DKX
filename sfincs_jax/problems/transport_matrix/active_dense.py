"""Combined active-DOF and dense-path setup for RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.operators.profile_response.compressed_layout import build_rhs1_compressed_pitch_layout
from sfincs_jax.problems.transport_matrix.solve_policy import (
    TransportActiveDOFDecision,
    TransportActiveDOFState,
    TransportDensePolicy,
    TransportInitialSolvePolicy,
    build_transport_active_dof_state,
    resolve_transport_active_dof_mode,
    resolve_transport_dense_policy,
    resolve_transport_initial_solve_policy,
)


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


__all__ = [
    "TransportActiveDenseSetup",
    "resolve_transport_active_dense_setup",
]
