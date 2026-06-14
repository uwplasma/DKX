"""Transport linear-solver dispatch for RHSMode=2/3 solve loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from .implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from .solver import (
    GMRESSolveResult,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    gmres_solve,
    gmres_solve_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
)
from .v3_system import sharding_constraints


@dataclass(frozen=True)
class TransportLinearSolveContext:
    """Routing state shared by transport linear solves."""

    rhs_mode: int
    size_hint: int
    use_implicit: bool
    use_solver_jit: bool
    distributed_axis: str | None


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
    return int(gmres_restart) if transport_solver_kind(method, rhs_mode=int(rhs_mode))[0] == "gmres" else int(restart)


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
        solver_kind, gmres_method = transport_solver_kind(solve_method_val, rhs_mode=int(context.rhs_mode))
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
    solver_kind, gmres_method = transport_solver_kind(solve_method_val, rhs_mode=int(context.rhs_mode))
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
        solver_fn = bicgstab_solve_with_residual_jit if context.use_solver_jit else bicgstab_solve_with_residual
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
    solver_fn = gmres_solve_with_residual_jit if context.use_solver_jit else gmres_solve_with_residual
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


__all__ = [
    "TransportLinearSolveContext",
    "solve_transport_linear",
    "solve_transport_linear_with_residual",
    "transport_restart_for_method",
    "transport_solver_kind",
]
