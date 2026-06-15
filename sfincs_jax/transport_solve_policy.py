"""Shared transport active-DOF and dense-policy helpers."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class TransportActiveDOFDecision:
    """Resolved active-DOF routing decision for one transport solve."""

    use_active_dof_mode: bool
    reason: str | None
    solve_method_use: str
    emit_disabled_hint: bool


@dataclass(frozen=True)
class TransportActiveDOFState:
    """Active-index arrays and size metadata used by reduced transport solves."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    active_size: int


@dataclass(frozen=True)
class TransportDensePolicy:
    """Resolved dense fallback and dense-preconditioner policy."""

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


@dataclass(frozen=True)
class TransportInitialSolvePolicy:
    """Initial RHSMode=2/3 output, dense fallback, and restart policy."""

    geometry_scheme: int
    low_memory_outputs: bool
    stream_diagnostics: bool
    store_state_vectors: bool
    solve_method_use: str
    force_krylov: bool
    force_dense: bool
    dense_fallback: bool
    dense_fallback_max: int
    dense_retry_max: int
    dense_mem_max_mb: float
    dense_mem_est_mb32: float
    dense_mem_est_mb64: float
    dense_mem_block: bool
    dense_use_mixed: bool
    dense_backend_allowed: bool
    dense_accelerator_auto_allowed: bool
    gmres_restart: int
    maxiter: int | None
    notes: tuple[tuple[int, str], ...] = ()


def transport_geometry_scheme_from_namelist(nml: Any) -> int:
    """Return ``geometryScheme`` from a namelist-like object, or ``-1``."""
    geom_params = nml.group("geometryParameters")
    try:
        return int(geom_params.get("GEOMETRYSCHEME", geom_params.get("geometryScheme", -1)) or -1)
    except (TypeError, ValueError):
        return -1


def _transport_bool_env(name: str) -> bool | None:
    value = os.environ.get(name, "").strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return None


def resolve_transport_initial_solve_policy(
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
) -> TransportInitialSolvePolicy:
    """Resolve initial transport solve policy before active-DOF setup."""
    notes: list[tuple[int, str]] = []

    low_memory_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_LOW_MEMORY")
    if low_memory_env is not None:
        low_memory_outputs = bool(low_memory_env)
    elif transport_geometry5_mono_low_memory_preferred(
        rhs_mode=int(rhs_mode),
        geometry_scheme=int(geometry_scheme),
        backend=str(backend),
        has_fp=op.fblock.fp is not None,
        n_x=int(op.n_x),
        total_size=int(op.total_size),
    ):
        low_memory_outputs = True
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: geometryScheme=5 RHSMode=3 "
                "auto -> low-memory Krylov transport path",
            )
        )
    else:
        low_memory_outputs = int(op.total_size) * int(n_rhs) >= 200_000

    stream_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS")
    stream_diagnostics = bool(low_memory_outputs) if stream_env is None else bool(stream_env)
    store_state_env = _transport_bool_env("SFINCS_JAX_TRANSPORT_STORE_STATE")
    store_state_vectors = (not stream_diagnostics) if store_state_env is None else bool(store_state_env)
    if state_out_requested:
        store_state_vectors = True
    if (not stream_diagnostics) and (not store_state_vectors):
        store_state_vectors = True
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: forcing state storage (streaming disabled)"))
    if force_stream_diagnostics is not None:
        stream_diagnostics = bool(force_stream_diagnostics)
    if force_store_state is not None:
        store_state_vectors = bool(force_store_state)
    if subset_mode and not stream_diagnostics:
        stream_diagnostics = True
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: streaming diagnostics forced for subset whichRHS"))

    solve_method_use = str(solve_method)
    force_krylov = _transport_bool_env("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV") is True
    force_dense = _transport_bool_env("SFINCS_JAX_TRANSPORT_FORCE_DENSE") is True
    if low_memory_outputs:
        force_krylov = True
        force_dense = False

    dense_fallback_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK", "").strip().lower()
    dense_fallback_max_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX", "").strip()
    try:
        dense_fallback_max = int(dense_fallback_max_env) if dense_fallback_max_env else 0
    except ValueError:
        dense_fallback_max = 0
    dense_retry_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX", "").strip()
    try:
        if dense_retry_env:
            dense_retry_max = int(dense_retry_env)
        else:
            dense_retry_max = 6000 if int(rhs_mode) in {2, 3} else 0
    except ValueError:
        dense_retry_max = 3000 if int(rhs_mode) in {2, 3} else 0
    if low_memory_outputs:
        dense_retry_max = 0
    dense_mem_env = os.environ.get("SFINCS_JAX_TRANSPORT_DENSE_MAX_MB", "").strip()
    try:
        dense_mem_max_mb = float(dense_mem_env) if dense_mem_env else 128.0
    except ValueError:
        dense_mem_max_mb = 128.0
    dense_mem_est_mb64 = (int(op.total_size) ** 2) * 8.0 / 1.0e6
    dense_mem_est_mb32 = (int(op.total_size) ** 2) * 4.0 / 1.0e6
    dense_mem_block64 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb64 > dense_mem_max_mb)
    dense_mem_block32 = bool(dense_mem_max_mb > 0.0 and dense_mem_est_mb32 > dense_mem_max_mb)
    dense_mem_block = dense_mem_block32
    dense_use_mixed = dense_mem_block64 and not dense_mem_block32
    dense_fallback_enabled_env = dense_fallback_env in {"1", "true", "yes", "on"}
    dense_fallback_disabled_env = dense_fallback_env in {"0", "false", "no", "off"}
    if dense_fallback_enabled_env:
        dense_fallback = True
        if not dense_fallback_max_env:
            dense_fallback_max = 1600
    elif dense_fallback_disabled_env:
        dense_fallback = False
    else:
        dense_fallback = int(rhs_mode) == 3
        if dense_fallback and not dense_fallback_max_env:
            dense_fallback_max = 6000

    dense_backend_allowed = bool(dense_backend_policy_allowed) or bool(dense_accelerator_auto_allowed)
    if dense_accelerator_auto_allowed:
        notes.append((1, "solve_v3_transport_matrix_linear_gmres: bounded accelerator dense transport auto enabled"))
    if not dense_backend_allowed:
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        if str(solve_method_use).lower() == "dense":
            solve_method_use = "incremental"
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense transport path disabled "
                f"on backend={backend}",
            )
        )
    if dense_mem_block:
        dense_fallback = False
        dense_retry_max = 0
        force_dense = False
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback disabled "
                f"(est_mem32={dense_mem_est_mb32:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        )
        if str(solve_method_use).lower() in {"auto", "default", "batched"}:
            solve_method_use = "incremental"
    elif dense_use_mixed:
        notes.append(
            (
                1,
                "solve_v3_transport_matrix_linear_gmres: dense fallback using float32 "
                f"(est_mem64={dense_mem_est_mb64:.1f} MB > {dense_mem_max_mb:.1f} MB)",
            )
        )
    if low_memory_outputs:
        dense_fallback = False

    if int(rhs_mode) in {2, 3}:
        if force_dense:
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: forced dense solve "
                    f"for RHSMode={rhs_mode} (n={int(op.total_size)})",
                )
            )
        elif (
            int(rhs_mode) == 2
            and (not force_krylov)
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and int(op.total_size) <= 1500
            and (not dense_mem_block)
        ):
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: auto dense solve for RHSMode=2 "
                    f"(n={int(op.total_size)})",
                )
            )
        elif (
            dense_fallback
            and (not force_krylov)
            and int(op.total_size) <= int(dense_fallback_max)
            and str(solve_method_use).lower() in {"auto", "default", "batched", "incremental"}
            and (not dense_mem_block)
        ):
            solve_method_use = "dense"
            notes.append(
                (
                    0,
                    "solve_v3_transport_matrix_linear_gmres: dense fallback enabled "
                    f"for RHSMode={rhs_mode} (n={int(op.total_size)})",
                )
            )

    gmres_restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_GMRES_RESTART", "").strip()
    try:
        gmres_restart = int(gmres_restart_env) if gmres_restart_env else min(int(restart), 40)
    except ValueError:
        gmres_restart = min(int(restart), 40)
    if dense_mem_block and gmres_restart < 80:
        gmres_restart = 80

    maxiter_out = maxiter
    if dense_mem_block:
        if maxiter_out is None:
            maxiter_out = 800
        else:
            maxiter_out = max(int(maxiter_out), 800)

    return TransportInitialSolvePolicy(
        geometry_scheme=int(geometry_scheme),
        low_memory_outputs=bool(low_memory_outputs),
        stream_diagnostics=bool(stream_diagnostics),
        store_state_vectors=bool(store_state_vectors),
        solve_method_use=str(solve_method_use),
        force_krylov=bool(force_krylov),
        force_dense=bool(force_dense),
        dense_fallback=bool(dense_fallback),
        dense_fallback_max=int(dense_fallback_max),
        dense_retry_max=int(dense_retry_max),
        dense_mem_max_mb=float(dense_mem_max_mb),
        dense_mem_est_mb32=float(dense_mem_est_mb32),
        dense_mem_est_mb64=float(dense_mem_est_mb64),
        dense_mem_block=bool(dense_mem_block),
        dense_use_mixed=bool(dense_use_mixed),
        dense_backend_allowed=bool(dense_backend_allowed),
        dense_accelerator_auto_allowed=bool(dense_accelerator_auto_allowed),
        gmres_restart=int(gmres_restart),
        maxiter=maxiter_out,
        notes=tuple(notes),
    )


def transport_geometry5_mono_low_memory_preferred(
    *,
    rhs_mode: int,
    geometry_scheme: int,
    backend: str,
    has_fp: bool,
    n_x: int,
    total_size: int,
) -> bool:
    """Return whether VMEC monoenergetic transport should avoid dense fallback.

    The CPU VMEC RHSMode=3 examples are small enough that dense batched solves are
    numerically safe, but the CLI/XLA dense path can transiently retain multi-GB
    allocations. The existing Krylov + ``tzfft`` path is parity-clean on the
    geometryScheme=5 monoenergetic examples and has much lower peak RSS.
    """
    mode = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY", "").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if int(rhs_mode) != 3 or int(geometry_scheme) != 5:
        return False
    if bool(has_fp) or int(n_x) > 2:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True
    if str(backend).strip().lower() != "cpu":
        return False
    min_env = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MIN", "").strip()
    max_env = os.environ.get("SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MAX", "").strip()
    try:
        min_size = int(min_env) if min_env else 1000
    except ValueError:
        min_size = 1000
    try:
        max_size = int(max_env) if max_env else 20000
    except ValueError:
        max_size = 20000
    return max(1, int(min_size)) <= int(total_size) <= max(1, int(max_size))


def resolve_transport_active_dof_mode(
    *,
    op: Any,
    rhs_mode: int,
    solve_method_use: str,
    solve_method: str,
    active_dof_env: str,
) -> TransportActiveDOFDecision:
    """Resolve whether transport should compact to the active pitch-angle DOFs."""
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
    """Build active-DOF indexing state or a full-size no-op state."""
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
    """Resolve dense fallback/preconditioner admission under memory caps."""
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
    "TransportInitialSolvePolicy",
    "build_transport_active_dof_state",
    "resolve_transport_active_dof_mode",
    "resolve_transport_dense_policy",
    "resolve_transport_initial_solve_policy",
    "transport_geometry5_mono_low_memory_preferred",
    "transport_geometry_scheme_from_namelist",
]
