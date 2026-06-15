"""Setup policy for RHSMode=2/3 transport solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import os
from typing import Any

import jax.numpy as jnp

from .transport_matrix import transport_matrix_size_from_rhs_mode


StateLoader = Callable[..., dict[str, Any] | None]


@dataclass(frozen=True)
class TransportMaxiterSetup:
    """Resolved transport max-iteration value and user-facing notes."""

    maxiter: int | None
    notes: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class TransportStateSetup:
    """Resolved transport Krylov state input/output settings."""

    state_in_path: str
    state_out_path: str
    x0: jnp.ndarray | None
    x0_by_rhs: dict[int, jnp.ndarray] | None
    state_x_by_rhs: dict[int, jnp.ndarray] | None


@dataclass(frozen=True)
class TransportWhichRHSSetup:
    """Normalized RHSMode=2/3 transport drive selection."""

    rhs_mode: int
    n_rhs: int
    which_rhs_values: list[int]
    subset_mode: bool


@dataclass(frozen=True)
class TransportParallelRequest:
    """Resolved process/GPU parallel request before parent-side dispatch."""

    parallel_child: bool
    parallel_workers: int
    parallel_backend: str


def resolve_transport_maxiter_setup(
    maxiter: int | None,
    *,
    maxiter_env: str | None = None,
) -> TransportMaxiterSetup:
    """Resolve ``SFINCS_JAX_TRANSPORT_MAXITER`` without side effects."""
    env = (
        os.environ.get("SFINCS_JAX_TRANSPORT_MAXITER", "").strip()
        if maxiter_env is None
        else str(maxiter_env).strip()
    )
    if not env:
        return TransportMaxiterSetup(maxiter=maxiter)
    try:
        maxiter_use = max(1, int(env))
    except ValueError:
        return TransportMaxiterSetup(
            maxiter=maxiter,
            notes=(
                (
                    1,
                    "solve_v3_transport_matrix_linear_gmres: ignoring invalid "
                    f"SFINCS_JAX_TRANSPORT_MAXITER={env!r}",
                ),
            ),
        )
    return TransportMaxiterSetup(
        maxiter=maxiter_use,
        notes=(
            (
                1,
                f"solve_v3_transport_matrix_linear_gmres: maxiter override={int(maxiter_use)}",
            ),
        ),
    )


def resolve_transport_state_setup(
    *,
    op: Any,
    x0: jnp.ndarray | None,
    x0_by_rhs: dict[int, jnp.ndarray] | None,
    state_in_env: str | None = None,
    state_out_env: str | None = None,
    load_state: StateLoader | None = None,
) -> TransportStateSetup:
    """Load optional Krylov state and merge it with explicit initial guesses."""
    state_in_path = (
        os.environ.get("SFINCS_JAX_STATE_IN", "").strip()
        if state_in_env is None
        else str(state_in_env).strip()
    )
    state_out_path = (
        os.environ.get("SFINCS_JAX_STATE_OUT", "").strip()
        if state_out_env is None
        else str(state_out_env).strip()
    )
    state_x_by_rhs: dict[int, jnp.ndarray] | None = None
    x0_use = x0
    x0_by_rhs_use = x0_by_rhs
    if state_in_path:
        loader = _load_transport_krylov_state if load_state is None else load_state
        try:
            state = loader(path=state_in_path, op=op)
        except Exception:  # noqa: BLE001 - state files must never make solves fail.
            state = None
        if state:
            state_x_by_rhs = state.get("x_by_rhs")
            if x0_by_rhs_use is None:
                x0_by_rhs_use = state_x_by_rhs
            if x0_use is None:
                x0_use = state.get("x_full")
    return TransportStateSetup(
        state_in_path=state_in_path,
        state_out_path=state_out_path,
        x0=x0_use,
        x0_by_rhs=x0_by_rhs_use,
        state_x_by_rhs=state_x_by_rhs,
    )


def resolve_transport_which_rhs_setup(
    *,
    rhs_mode: int,
    which_rhs_values: Sequence[int] | None,
) -> TransportWhichRHSSetup:
    """Normalize a requested subset of transport ``whichRHS`` drives."""
    n_rhs = transport_matrix_size_from_rhs_mode(int(rhs_mode))
    if which_rhs_values is None:
        values = list(range(1, int(n_rhs) + 1))
    else:
        values = sorted({int(v) for v in which_rhs_values if 1 <= int(v) <= int(n_rhs)})
    return TransportWhichRHSSetup(
        rhs_mode=int(rhs_mode),
        n_rhs=int(n_rhs),
        which_rhs_values=values,
        subset_mode=len(values) < int(n_rhs),
    )


def resolve_transport_parallel_request(
    *,
    which_rhs_count: int,
    n_rhs: int,
    parallel_workers: int | None,
    parallel_backend: str,
    visible_gpu_ids: Callable[[int], Sequence[str]],
    parallel_child_env: str | None = None,
    parallel_env: str | None = None,
    workers_env: str | None = None,
    cpu_count: int | None = None,
) -> TransportParallelRequest:
    """Resolve parent/worker parallel settings before launching workers."""
    parallel_child_raw = (
        os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_CHILD", "").strip().lower()
        if parallel_child_env is None
        else str(parallel_child_env).strip().lower()
    )
    parallel_child = parallel_child_raw in {"1", "true", "yes", "on"}
    parallel_raw = (
        os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL", "").strip().lower()
        if parallel_env is None
        else str(parallel_env).strip().lower()
    )
    if parallel_workers is None:
        workers_raw = (
            os.environ.get("SFINCS_JAX_TRANSPORT_PARALLEL_WORKERS", "").strip()
            if workers_env is None
            else str(workers_env).strip()
        )
        try:
            workers_val = int(workers_raw) if workers_raw else 0
        except ValueError:
            workers_val = 0
        if parallel_raw in {"", "0", "false", "no", "off"}:
            workers = 1
        elif parallel_raw in {"process", "auto", "1", "true", "yes", "on"}:
            if workers_val > 0:
                workers = workers_val
            else:
                count = int(cpu_count if cpu_count is not None else (os.cpu_count() or 1))
                workers = min(count, int(n_rhs)) if int(n_rhs) > 1 else 1
        else:
            workers = 1
    else:
        workers = max(1, int(parallel_workers))
    if workers > 1:
        workers = min(int(workers), int(which_rhs_count))
    backend = str(parallel_backend)
    if backend == "gpu" and workers > 1:
        gpu_ids = list(visible_gpu_ids(int(workers)))
        workers = min(int(workers), len(gpu_ids))
        if workers <= 1:
            backend = "cpu"
    return TransportParallelRequest(
        parallel_child=bool(parallel_child),
        parallel_workers=max(1, int(workers)),
        parallel_backend=backend,
    )


def _load_transport_krylov_state(*, path: str, op: Any) -> dict[str, Any] | None:
    from .solver_state import load_krylov_state  # noqa: PLC0415

    return load_krylov_state(path=path, op=op)


__all__ = [
    "TransportMaxiterSetup",
    "TransportParallelRequest",
    "TransportStateSetup",
    "TransportWhichRHSSetup",
    "resolve_transport_maxiter_setup",
    "resolve_transport_parallel_request",
    "resolve_transport_state_setup",
    "resolve_transport_which_rhs_setup",
]
