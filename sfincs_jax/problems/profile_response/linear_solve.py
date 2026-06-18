"""Linear-solver dispatch for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from sfincs_jax.implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from sfincs_jax.krylov_dispatch import gmres_solve_dispatch, rhs_krylov_method_for_context
from sfincs_jax.solver import (
    GMRESSolveResult,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
)
from sfincs_jax.v3_system import sharding_constraints


@dataclass(frozen=True)
class ProfileLinearSolveContext:
    """Routing state shared by RHSMode=1 linear-solve attempts."""

    rhs_mode: int
    total_size: int
    use_implicit: bool
    use_solver_jit: bool
    distributed_axis: str | None
    distributed_auto_solver: str
    small_gmres_max: int


def profile_solver_kind(method: str, *, context: ProfileLinearSolveContext) -> tuple[str, str]:
    """Map RHSMode=1 solve-method tokens to the concrete Krylov family."""

    method_l = str(method).strip().lower()
    if method_l in {"auto", "default"}:
        if (
            context.distributed_axis is not None
            and int(context.rhs_mode) == 1
            and context.distributed_auto_solver == "bicgstab"
        ):
            return "bicgstab", "batched"
        if int(context.rhs_mode) in {2, 3}:
            return "gmres", "incremental"
        if int(context.small_gmres_max) > 0 and int(context.total_size) <= int(context.small_gmres_max):
            return "gmres", "incremental"
        return "gmres", "incremental"
    if method_l in {"bicgstab", "bicgstab_jax"}:
        return "bicgstab", "batched"
    return "gmres", method_l


def solve_profile_linear(
    *,
    context: ProfileLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    precond_side: str,
) -> GMRESSolveResult:
    """Solve an RHSMode=1 linear system without returning an explicit residual."""

    solver_kind, gmres_method = profile_solver_kind(solve_method_val, context=context)
    if context.use_implicit:
        return linear_custom_solve(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precond_side,
            size_hint=int(b_vec.shape[0]),
        )
    solve_method_dispatch = "bicgstab" if solver_kind == "bicgstab" else rhs_krylov_method_for_context(
        gmres_method=gmres_method,
        use_implicit=bool(context.use_implicit),
        distributed_axis=context.distributed_axis,
        solver_jit=bool(context.use_solver_jit),
    )
    return gmres_solve_dispatch(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=solve_method_dispatch,
        distributed_axis=context.distributed_axis,
        precondition_side=precond_side,
        use_solver_jit_fn=lambda _size_hint: bool(context.use_solver_jit),
    )


def solve_profile_linear_with_residual(
    *,
    context: ProfileLinearSolveContext,
    matvec_fn: Callable[[jnp.ndarray], jnp.ndarray],
    b_vec: jnp.ndarray,
    precond_fn: Callable[[jnp.ndarray], jnp.ndarray] | None,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    solve_method_val: str,
    precond_side: str,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve an RHSMode=1 linear system and return the explicit residual."""

    solver_kind, gmres_method = profile_solver_kind(solve_method_val, context=context)
    if context.use_implicit:
        return linear_custom_solve_with_residual(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            solve_method=gmres_method,
            solver=solver_kind,
            precondition_side=precond_side,
            size_hint=int(b_vec.shape[0]),
        )
    if solver_kind == "bicgstab":
        if context.distributed_axis is not None:
            with sharding_constraints(True):
                return gmres_solve_with_residual_distributed(
                    matvec=matvec_fn,
                    b=b_vec,
                    preconditioner=precond_fn,
                    x0=x0_vec,
                    tol=tol_val,
                    atol=atol_val,
                    restart=restart_val,
                    maxiter=maxiter_val,
                    solve_method="bicgstab",
                    precondition_side=precond_side,
                    axis_name=context.distributed_axis,
                )
        solver_fn = bicgstab_solve_with_residual_jit if context.use_solver_jit else bicgstab_solve_with_residual
        return solver_fn(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
    gmres_method_dispatch = rhs_krylov_method_for_context(
        gmres_method=gmres_method,
        use_implicit=bool(context.use_implicit),
        distributed_axis=context.distributed_axis,
        solver_jit=bool(context.use_solver_jit),
    )
    if context.distributed_axis is not None:
        with sharding_constraints(True):
            return gmres_solve_with_residual_distributed(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                restart=restart_val,
                maxiter=maxiter_val,
                solve_method=gmres_method_dispatch,
                precondition_side=precond_side,
                axis_name=context.distributed_axis,
            )
    solver_fn = gmres_solve_with_residual_jit if context.use_solver_jit else gmres_solve_with_residual
    return solver_fn(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        solve_method=gmres_method_dispatch,
        precondition_side=precond_side,
    )


__all__ = [
    "ProfileLinearSolveContext",
    "profile_solver_kind",
    "solve_profile_linear",
    "solve_profile_linear_with_residual",
]
