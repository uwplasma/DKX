"""Krylov-solver dispatch helpers for v3-compatible solve paths.

This module translates user-facing solve-method names into the concrete JAX,
host, or distributed GMRES implementations. It intentionally stays small: the
physics operator and residual gates remain with the RHSMode-specific solve
orchestration, while this module owns backend/routing compatibility rules.
"""

from __future__ import annotations

from collections.abc import Callable
import os
from typing import Any

import jax

from .preconditioning import use_solver_jit
from ..solver import (
    distributed_gmres_enabled,
    gmres_solve,
    gmres_solve_distributed,
    gmres_solve_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
)
from sfincs_jax.operators.profile_system import _matvec_shard_axis, sharding_constraints


HOST_SCIPY_KRYLOV_METHODS = frozenset({"lgmres", "lgmres_scipy"})


def host_scipy_krylov_requested(solve_method: str | None) -> bool:
    """Return whether ``solve_method`` requests a host-only SciPy Krylov method."""

    return str(solve_method or "").strip().lower() in HOST_SCIPY_KRYLOV_METHODS


def gmres_solve_dispatch(
    *,
    distributed_axis: str | None = None,
    size_hint: int | None = None,
    gmres_solve_fn: Callable[..., Any] = gmres_solve,
    gmres_solve_jit_fn: Callable[..., Any] = gmres_solve_jit,
    gmres_solve_distributed_fn: Callable[..., Any] = gmres_solve_distributed,
    distributed_gmres_enabled_fn: Callable[[], bool] = distributed_gmres_enabled,
    use_solver_jit_fn: Callable[[int | None], bool] = use_solver_jit,
    **kwargs: Any,
):
    """Dispatch GMRES to host, JIT, or distributed implementations."""

    solve_method = kwargs.get("solve_method")
    if host_scipy_krylov_requested(solve_method):
        if distributed_axis is not None or distributed_gmres_enabled_fn():
            raise ValueError(f"solve_method={solve_method} is host-only and incompatible with distributed GMRES.")
        return gmres_solve_fn(**kwargs)
    if distributed_axis is not None:
        with sharding_constraints(True):
            return gmres_solve_distributed_fn(axis_name=distributed_axis, **kwargs)
    if distributed_gmres_enabled_fn():
        with sharding_constraints(True):
            return gmres_solve_distributed_fn(**kwargs)
    solver_fn = gmres_solve_jit_fn if use_solver_jit_fn(size_hint) else gmres_solve_fn
    return solver_fn(**kwargs)


def gmres_solve_with_residual_dispatch(
    *,
    distributed_axis: str | None = None,
    size_hint: int | None = None,
    gmres_solve_with_residual_fn: Callable[..., Any] = gmres_solve_with_residual,
    gmres_solve_with_residual_jit_fn: Callable[..., Any] = gmres_solve_with_residual_jit,
    gmres_solve_with_residual_distributed_fn: Callable[..., Any] = gmres_solve_with_residual_distributed,
    distributed_gmres_enabled_fn: Callable[[], bool] = distributed_gmres_enabled,
    use_solver_jit_fn: Callable[[int | None], bool] = use_solver_jit,
    **kwargs: Any,
):
    """Dispatch GMRES-with-residual to host, JIT, or distributed implementations."""

    solve_method = kwargs.get("solve_method")
    if host_scipy_krylov_requested(solve_method):
        if distributed_axis is not None or distributed_gmres_enabled_fn():
            raise ValueError(f"solve_method={solve_method} is host-only and incompatible with distributed GMRES.")
        return gmres_solve_with_residual_fn(**kwargs)
    if distributed_axis is not None:
        with sharding_constraints(True):
            return gmres_solve_with_residual_distributed_fn(axis_name=distributed_axis, **kwargs)
    if distributed_gmres_enabled_fn():
        with sharding_constraints(True):
            return gmres_solve_with_residual_distributed_fn(**kwargs)
    solver_fn = gmres_solve_with_residual_jit_fn if use_solver_jit_fn(size_hint) else gmres_solve_with_residual_fn
    return solver_fn(**kwargs)


def rhs_krylov_method_for_context(
    *,
    gmres_method: str,
    use_implicit: bool,
    distributed_axis: str | None,
    solver_jit: bool,
) -> str:
    """Downgrade host-only methods when the active context cannot run them."""

    method = str(gmres_method).strip().lower()
    if method in {"lgmres", "lgmres_scipy"} and (bool(use_implicit) or distributed_axis is not None or bool(solver_jit)):
        return "incremental"
    return method


def ksp_iteration_solver_label(*, solver_kind: str, solve_method: str) -> str:
    """Return the concrete Krylov label used for iteration-history replay."""

    solver_kind_l = str(solver_kind or "").strip().lower()
    if solver_kind_l == "gmres":
        _kind, gmres_method = solver_kind_for_label(str(solve_method))
        method_l = str(gmres_method).strip().lower()
        if method_l in HOST_SCIPY_KRYLOV_METHODS:
            return "lgmres"
        if method_l in {"incremental", "batched", "auto", "default", ""}:
            return "gmres"
        return method_l
    return solver_kind_l


def solver_kind_for_label(method: str) -> tuple[str, str]:
    """Map solve-method tokens to diagnostic solver-kind labels."""

    method_l = str(method).strip().lower()
    if method_l in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab", "batched"
    if method_l in {"auto", "default", ""}:
        return "gmres", "incremental"
    return "gmres", method_l


def resolve_distributed_gmres_axis(
    *,
    op: Any | None,
    emit: Callable[[int, str], None] | None = None,
    matvec_shard_axis_fn: Callable[[Any], str | None] = _matvec_shard_axis,
) -> str | None:
    """Resolve the distributed-GMRES sharding axis from environment and operator metadata."""

    env = os.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED", "").strip().lower()
    if env in {"", "0", "false", "no", "off"}:
        return None
    if env in {"theta", "zeta"}:
        axis = env
    elif env in {"1", "true", "yes", "on", "auto"}:
        axis = matvec_shard_axis_fn(op) if op is not None else None
        if axis not in {"theta", "zeta"}:
            axis = None
    else:
        axis = None
    if axis not in {"theta", "zeta"}:
        if env not in {"theta", "zeta", "1", "true", "yes", "on", "auto"} and emit is not None:
            emit(1, f"SFINCS_JAX_GMRES_DISTRIBUTED={env!r} not recognized; distributed GMRES disabled.")
        return None
    if env in {"1", "true", "yes", "on", "auto"} and jax.default_backend() != "cpu":
        allow_accel = os.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", "").strip().lower()
        if allow_accel not in {"1", "true", "yes", "on"}:
            if emit is not None:
                emit(
                    1,
                    "solve_v3_*: distributed GMRES auto mode disabled on "
                    f"backend={jax.default_backend()}",
                )
            return None
    n_devices = jax.local_device_count()
    if n_devices <= 1:
        return None
    # Padding/masking in the sharded matvec path handles irregular partitions.
    return axis


__all__ = [
    "HOST_SCIPY_KRYLOV_METHODS",
    "gmres_solve_dispatch",
    "gmres_solve_with_residual_dispatch",
    "host_scipy_krylov_requested",
    "ksp_iteration_solver_label",
    "resolve_distributed_gmres_axis",
    "rhs_krylov_method_for_context",
    "solver_kind_for_label",
]
