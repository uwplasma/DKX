"""Shared transport active-DOF and dense-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TransportActiveDOFDecision:
    use_active_dof_mode: bool
    reason: str | None
    solve_method_use: str
    emit_disabled_hint: bool


@dataclass(frozen=True)
class TransportActiveDOFState:
    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int


@dataclass(frozen=True)
class TransportDensePolicy:
    solve_method_use: str
    dense_fallback: bool
    dense_retry_max: int
    dense_mem_block: bool
    dense_use_mixed: bool
    force_dense: bool
    dense_precond_enabled: bool
    dense_precond_mem_block: bool
    dense_precond_est_mb: float
    dense_precond_mem_max_mb: float
    dense_mem_est_active_mb32: float
    dense_mem_est_active_mb64: float


def resolve_transport_active_dof_mode(
    *,
    op: Any,
    rhs_mode: int,
    solve_method_use: str,
    solve_method: str,
    active_dof_env: str,
) -> TransportActiveDOFDecision:
    env = str(active_dof_env).strip().lower()
    reason: str | None = None
    if env in {"0", "false", "no", "off"}:
        use_active_dof_mode = False
    elif env in {"1", "true", "yes", "on"}:
        use_active_dof_mode = True
        reason = "env"
    elif int(rhs_mode) in {2, 3}:
        nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
        use_active_dof_mode = bool(np.any(nxi_for_x < int(op.n_xi)))
        if use_active_dof_mode:
            reason = "auto"
    else:
        use_active_dof_mode = False
    solve_method_out = str(solve_method_use)
    if use_active_dof_mode and str(solve_method_out).lower() == "dense":
        solve_method_out = str(solve_method)
    emit_disabled_hint = (
        (not use_active_dof_mode)
        and int(rhs_mode) in {2, 3}
        and env not in {"0", "false", "no", "off"}
    )
    return TransportActiveDOFDecision(
        use_active_dof_mode=bool(use_active_dof_mode),
        reason=reason,
        solve_method_use=solve_method_out,
        emit_disabled_hint=bool(emit_disabled_hint),
    )


def build_transport_active_dof_state(
    *,
    op: Any,
    use_active_dof_mode: bool,
    active_dof_indices,
) -> TransportActiveDOFState:
    if not use_active_dof_mode:
        return TransportActiveDOFState(
            active_idx_np=None,
            active_idx_jnp=None,
            full_to_active_jnp=None,
            active_size=int(op.total_size),
        )
    active_idx_np = np.asarray(active_dof_indices(op), dtype=np.int32)
    active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
    full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
    full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(1, int(active_idx_np.shape[0]) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
    return TransportActiveDOFState(
        active_idx_np=active_idx_np,
        active_idx_jnp=active_idx_jnp,
        full_to_active_jnp=full_to_active_jnp,
        active_size=int(active_idx_np.shape[0]),
    )


def resolve_transport_dense_policy(
    *,
    rhs_mode: int,
    n_rhs: int,
    total_size: int,
    active_size: int,
    solve_method_use: str,
    force_krylov: bool,
    force_dense: bool,
    dense_fallback: bool,
    dense_retry_max: int,
    dense_mem_max_mb: float,
    dense_mem_block: bool,
    dense_use_mixed: bool,
    low_memory_outputs: bool,
    dense_backend_allowed: bool,
    dense_precond_default: bool,
) -> TransportDensePolicy:
    dense_mem_est_active_mb64 = (int(active_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_active_mb32 = (int(active_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block_active32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb32 > dense_mem_max_mb)
    dense_mem_block_active64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_active_mb64 > dense_mem_max_mb)

    solve_method_out = str(solve_method_use)
    dense_fallback_out = bool(dense_fallback)
    dense_retry_max_out = int(dense_retry_max)
    dense_mem_block_out = bool(dense_mem_block)
    dense_use_mixed_out = bool(dense_use_mixed)
    force_dense_out = bool(force_dense)

    if dense_mem_block_active32 and not dense_mem_block_out:
        dense_mem_block_out = True
        dense_use_mixed_out = False
        dense_fallback_out = False
        dense_retry_max_out = 0
        force_dense_out = False
        if str(solve_method_out).lower() == "dense":
            solve_method_out = "incremental"
    elif dense_mem_block_active64 and not dense_mem_block_out and not dense_use_mixed_out:
        dense_use_mixed_out = True

    if (
        int(rhs_mode) == 2
        and (not force_krylov)
        and (not force_dense_out)
        and str(solve_method_out).lower() in {"auto", "default", "batched", "incremental"}
    ):
        auto_dense_limit = 1500
        if int(n_rhs) > 1 and dense_retry_max_out > 0:
            auto_dense_limit = max(auto_dense_limit, min(3000, int(dense_retry_max_out)))
        if int(active_size) <= auto_dense_limit and (not dense_mem_block_out):
            solve_method_out = "dense"

    dense_precond_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", "").strip()
    dense_precond_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB", "").strip()
    try:
        dense_precond_max = int(dense_precond_max_env) if dense_precond_max_env else (1600 if int(rhs_mode) == 2 else 600)
    except ValueError:
        dense_precond_max = 1600 if int(rhs_mode) == 2 else 600
    try:
        dense_precond_mem_max_mb = float(dense_precond_mem_env) if dense_precond_mem_env else min(32.0, dense_mem_max_mb or 32.0)
    except ValueError:
        dense_precond_mem_max_mb = min(32.0, dense_mem_max_mb or 32.0)
    dense_precond_size = int(active_size)
    dense_precond_bytes = 4.0 if dense_use_mixed_out else 8.0
    dense_precond_est_mb = (dense_precond_size**2) * dense_precond_bytes / 1.0e6
    dense_precond_mem_block = bool(dense_precond_mem_max_mb > 0.0 and dense_precond_est_mb > dense_precond_mem_max_mb)
    dense_precond_enabled = (
        bool(dense_precond_default)
        and dense_precond_max > 0
        and int(rhs_mode) in {2, 3}
        and int(dense_precond_size) <= dense_precond_max
        and str(solve_method_out).lower() != "dense"
        and (not low_memory_outputs)
        and (not dense_mem_block_out)
        and (not dense_precond_mem_block)
        and dense_backend_allowed
    )
    return TransportDensePolicy(
        solve_method_use=solve_method_out,
        dense_fallback=dense_fallback_out,
        dense_retry_max=int(dense_retry_max_out),
        dense_mem_block=bool(dense_mem_block_out),
        dense_use_mixed=bool(dense_use_mixed_out),
        force_dense=bool(force_dense_out),
        dense_precond_enabled=bool(dense_precond_enabled),
        dense_precond_mem_block=bool(dense_precond_mem_block),
        dense_precond_est_mb=float(dense_precond_est_mb),
        dense_precond_mem_max_mb=float(dense_precond_mem_max_mb),
        dense_mem_est_active_mb32=float(dense_mem_est_active_mb32),
        dense_mem_est_active_mb64=float(dense_mem_est_active_mb64),
    )


__all__ = [
    "TransportActiveDOFDecision",
    "TransportActiveDOFState",
    "TransportDensePolicy",
    "build_transport_active_dof_state",
    "resolve_transport_active_dof_mode",
    "resolve_transport_dense_policy",
]
