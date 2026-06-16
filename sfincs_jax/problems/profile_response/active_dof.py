"""RHSMode=1 active-degree-of-freedom routing helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class RHS1ActiveDOFDecision:
    """Resolved active-DOF routing decision for the RHSMode=1 solve."""

    use_active_dof_mode: bool
    reason: str | None = None


@dataclass(frozen=True)
class RHS1ActiveDOFState:
    """Index maps used to reduce a full RHSMode=1 system to active pitch modes."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int


def _env_on(raw: str) -> bool:
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _env_off(raw: str) -> bool:
    return str(raw).strip().lower() in {"0", "false", "no", "off"}


def resolve_rhs1_active_dof_mode(
    *,
    active_dof_env: str,
    dkes_active_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_reduced_modes: bool,
    sparse_host_like_requested: bool,
    xblock_active_dof_requested: bool,
    has_pas: bool,
    use_dkes: bool,
) -> RHS1ActiveDOFDecision:
    """Resolve the default RHSMode=1/transport active-DOF compaction policy.

    The policy mirrors the historical in-driver logic: explicit
    ``SFINCS_JAX_ACTIVE_DOF`` settings win, otherwise reduced pitch grids are
    compacted automatically except for sparse-host-like solves that have not
    opted into x-block active-DOF handling. PAS DKES can still opt out through
    ``SFINCS_JAX_ACTIVE_DOF_DKES=0``.
    """

    env = str(active_dof_env).strip().lower()
    if _env_on(env):
        return RHS1ActiveDOFDecision(use_active_dof_mode=True, reason="env")
    if _env_off(env):
        return RHS1ActiveDOFDecision(use_active_dof_mode=False, reason="env")

    use_active_dof_mode = bool(
        has_reduced_modes
        and (
            int(rhs_mode) in {2, 3}
            or (int(rhs_mode) == 1 and not bool(include_phi1))
        )
    )
    reason = "auto" if use_active_dof_mode else None
    if sparse_host_like_requested and not bool(xblock_active_dof_requested):
        use_active_dof_mode = False
        reason = "sparse_host"
    if (
        _env_off(dkes_active_env)
        and use_active_dof_mode
        and int(rhs_mode) == 1
        and bool(has_pas)
        and bool(use_dkes)
    ):
        use_active_dof_mode = False
        reason = "dkes_env"
    return RHS1ActiveDOFDecision(
        use_active_dof_mode=bool(use_active_dof_mode),
        reason=reason,
    )


def build_rhs1_active_dof_state(
    *,
    op: Any,
    use_active_dof_mode: bool,
    use_pas_projection: bool,
    active_dof_indices: Callable[[Any], np.ndarray],
) -> RHS1ActiveDOFState:
    """Build active-DOF index maps for full-system or PAS-projected solves."""

    if not bool(use_active_dof_mode):
        return RHS1ActiveDOFState(
            active_idx_np=None,
            active_idx_jnp=None,
            full_to_active_jnp=None,
            active_size=int(op.total_size),
        )

    active_idx_np = np.asarray(active_dof_indices(op), dtype=np.int32)
    map_size = int(op.total_size)
    if bool(use_pas_projection):
        active_idx_np = active_idx_np[active_idx_np < int(op.f_size)]
        map_size = int(op.f_size)

    active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
    full_to_active_np = np.zeros((map_size,), dtype=np.int32)
    full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
        1,
        int(active_idx_np.shape[0]) + 1,
        dtype=np.int32,
    )
    full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)
    return RHS1ActiveDOFState(
        active_idx_np=active_idx_np,
        active_idx_jnp=active_idx_jnp,
        full_to_active_jnp=full_to_active_jnp,
        active_size=int(active_idx_np.shape[0]),
    )


__all__ = [
    "RHS1ActiveDOFDecision",
    "RHS1ActiveDOFState",
    "build_rhs1_active_dof_state",
    "resolve_rhs1_active_dof_mode",
]
