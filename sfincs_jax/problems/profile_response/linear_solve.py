"""Linear-solver dispatch for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from sfincs_jax.implicit_solve import linear_custom_solve, linear_custom_solve_with_residual
from sfincs_jax.krylov_dispatch import gmres_solve_dispatch, rhs_krylov_method_for_context
from sfincs_jax.solver import (
    GMRESSolveResult,
    bicgstab_solve_with_history_scipy,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
    gmres_solve_with_history_scipy,
)
from sfincs_jax.v3_system import sharding_constraints

from .residual import result_with_true_residual


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


@dataclass(frozen=True)
class RHS1ScipyRescueContext:
    """Host-only SciPy rescue solve inputs for stalled RHSMode=1 systems."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    method: str
    tol: float
    atol: float
    restart: int
    maxiter: int
    precond_side: str


@dataclass(frozen=True)
class RHS1ScipyRescueOutcome:
    """Result payload and measured diagnostics from a SciPy rescue attempt."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray
    reported_residual: float
    history_len: int
    preconditioned_residual: float | None = None


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


def run_rhs1_scipy_rescue(
    *,
    context: RHS1ScipyRescueContext,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1ScipyRescueOutcome:
    """Run the host-only SciPy rescue and recompute its true residual.

    This is intentionally non-differentiable and should only be called by
    CLI/host production lanes. The driver owns the size, timeout, and residual
    admission policy; this helper only executes the selected SciPy Krylov
    method and returns a true-residual payload.
    """

    method = str(context.method).strip().lower()
    if method not in {"gmres", "bicgstab"}:
        method = "gmres"
    side = str(context.precond_side).strip().lower()
    if method == "bicgstab":
        x_np, reported_residual, history = bicgstab_solve_with_history_scipy(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=int(context.maxiter),
            precondition_side=context.precond_side,
        )
        preconditioned_residual = None
    elif context.preconditioner is not None and side == "left":
        x_np, reported_residual, preconditioned_residual, history = (
            explicit_left_preconditioned_gmres_scipy(
                matvec=context.matvec,
                b=context.rhs,
                preconditioner=context.preconditioner,
                x0=context.x0,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=int(context.maxiter),
            )
        )
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: SciPy rescue residuals "
                f"true={float(reported_residual):.3e} "
                f"preconditioned={float(preconditioned_residual):.3e}",
            )
    else:
        x_np, reported_residual, history = gmres_solve_with_history_scipy(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precond_side,
        )
        preconditioned_residual = None
    x_scipy = jnp.asarray(x_np, dtype=jnp.float64)
    result, residual_vec = result_with_true_residual(
        x=x_scipy,
        rhs=context.rhs,
        matvec=context.matvec,
    )
    return RHS1ScipyRescueOutcome(
        result=result,
        residual_vec=residual_vec,
        reported_residual=float(reported_residual),
        history_len=len(history or []),
        preconditioned_residual=(
            None
            if preconditioned_residual is None
            else float(preconditioned_residual)
        ),
    )


__all__ = [
    "ProfileLinearSolveContext",
    "RHS1ScipyRescueContext",
    "RHS1ScipyRescueOutcome",
    "profile_solver_kind",
    "run_rhs1_scipy_rescue",
    "solve_profile_linear",
    "solve_profile_linear_with_residual",
]
