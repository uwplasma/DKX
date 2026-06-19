"""RHSMode=1 optional KSP history and iteration diagnostics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import os

import jax.numpy as jnp

from .krylov_dispatch import ksp_iteration_solver_label
from .solver import (
    bicgstab_solve_with_history_scipy,
    gmres_solve_with_history_scipy,
    lgmres_solve_with_history_scipy,
)


EmitFn = Callable[[int, str], None]

_FALSE_TOKENS = {"0", "false", "no", "off"}
_TRUE_TOKENS = {"1", "true", "yes", "on"}
_UNLIMITED_TOKENS = {"none", "inf", "infinite", "unlimited"}


@dataclass(frozen=True)
class RHS1KSPHistoryLimits:
    """Size and iteration caps for optional PETSc-like KSP replay."""

    max_size: int | None
    max_iter: int


@dataclass(frozen=True)
class RHS1KSPIterStatsControls:
    """Controls for optional bounded KSP iteration-count replay."""

    enabled: bool
    max_size: int | None


@dataclass(frozen=True)
class RHS1KSPDiagnosticsControls:
    """Environment-normalized controls shared by RHSMode=1 diagnostics."""

    fortran_stdout: bool
    history_max_size: int | None
    history_max_iter: int
    iter_stats_enabled: bool
    iter_stats_max_size: int | None


def rhs1_fortran_stdout_from_env(*, emit: EmitFn | None) -> bool:
    """Resolve Fortran-style solver stdout from the environment and emit state."""

    env = os.environ.get("SFINCS_JAX_FORTRAN_STDOUT", "").strip().lower()
    if env in _FALSE_TOKENS:
        return False
    if env in _TRUE_TOKENS:
        return True
    return emit is not None


def rhs1_ksp_history_limits_from_env() -> RHS1KSPHistoryLimits:
    """Return bounded replay limits for optional PETSc-like KSP history."""

    max_size_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_SIZE", "").strip().lower()
    if max_size_env in _UNLIMITED_TOKENS:
        max_size = None
    else:
        try:
            max_size = int(max_size_env) if max_size_env else 800
        except ValueError:
            max_size = 800

    max_iter_env = os.environ.get("SFINCS_JAX_KSP_HISTORY_MAX_ITER", "").strip()
    try:
        max_iter = int(max_iter_env) if max_iter_env else 2000
    except ValueError:
        max_iter = 2000
    return RHS1KSPHistoryLimits(max_size=max_size, max_iter=max_iter)


def rhs1_ksp_iter_stats_controls_from_env() -> RHS1KSPIterStatsControls:
    """Return opt-in iteration replay controls for solver diagnostics."""

    enabled_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS", "").strip().lower()
    max_size_env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_SIZE", "").strip()
    try:
        max_size = int(max_size_env) if max_size_env else None
    except ValueError:
        max_size = None
    return RHS1KSPIterStatsControls(enabled=enabled_env in _TRUE_TOKENS, max_size=max_size)


def rhs1_ksp_diagnostics_controls_from_env(*, emit: EmitFn | None) -> RHS1KSPDiagnosticsControls:
    """Parse all shared RHSMode=1 diagnostic replay controls."""

    history = rhs1_ksp_history_limits_from_env()
    iter_stats = rhs1_ksp_iter_stats_controls_from_env()
    return RHS1KSPDiagnosticsControls(
        fortran_stdout=rhs1_fortran_stdout_from_env(emit=emit),
        history_max_size=history.max_size,
        history_max_iter=history.max_iter,
        iter_stats_enabled=iter_stats.enabled,
        iter_stats_max_size=iter_stats.max_size,
    )


def emit_rhs1_ksp_history(
    *,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    solve_method_val: str,
    emit: EmitFn | None,
    fortran_stdout: bool,
    max_size: int | None,
    max_history_iter: int | None,
) -> list[float] | None:
    """Emit PETSc-like KSP residual history for bounded RHSMode=1 diagnostics."""
    if emit is None or not fortran_stdout:
        return None
    solver_label = ksp_iteration_solver_label(solver_kind=solver_kind, solve_method=solve_method_val)
    if solver_label not in {"gmres", "lgmres"}:
        return None
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"fortran-stdout: KSP history skipped (size={size} > max={int(max_size)})")
        return None
    if maxiter_val is not None and max_history_iter is not None:
        est_iters = int(maxiter_val)
        if solver_label == "gmres":
            est_iters *= max(1, int(restart_val))
        if est_iters > int(max_history_iter):
            emit(
                1,
                "fortran-stdout: KSP history skipped "
                f"(estimated_iters={est_iters} > max={int(max_history_iter)})",
            )
            return None
    try:
        history = _solve_history(
            solver_label=solver_label,
            matvec_fn=matvec_fn,
            b_vec=b_vec,
            precond_fn=precond_fn,
            x0_vec=x0_vec,
            tol_val=tol_val,
            atol_val=atol_val,
            restart_val=restart_val,
            maxiter_val=maxiter_val,
            precond_side=precond_side,
        )
    except Exception as exc:  # noqa: BLE001
        emit(1, f"fortran-stdout: KSP history unavailable ({type(exc).__name__}: {exc})")
        return None
    for k, rn in enumerate(history):
        emit(0, f"{k:4d} KSP Residual norm {rn: .12e} ")
    if history:
        emit(0, " Linear iteration (KSP) converged.  KSPConvergedReason =            2")
        emit(0, "   KSP_CONVERGED_RTOL: Norm decreased by rtol.")
    return history


def emit_rhs1_ksp_iter_stats(
    *,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
    solver_kind: str,
    history: list[float] | None,
    solve_method_val: str,
    emit: EmitFn | None,
    enabled: bool,
    max_size: int | None,
) -> None:
    """Emit bounded RHSMode=1 KSP iteration-count diagnostics."""
    if emit is None or not enabled:
        return
    size = int(b_vec.size)
    if max_size is not None and size > int(max_size):
        emit(1, f"ksp_iterations skipped (size={size} > max={int(max_size)})")
        return
    solver_kind_l = str(solver_kind).strip().lower()
    solver_label = ksp_iteration_solver_label(solver_kind=solver_kind_l, solve_method=solve_method_val)
    iter_stats_max_iter = _read_iter_stats_max_iter()
    if maxiter_val is not None and iter_stats_max_iter is not None:
        est_iters = int(maxiter_val)
        if solver_label == "gmres":
            est_iters *= max(1, int(restart_val))
        if est_iters > int(iter_stats_max_iter):
            emit(
                1,
                "ksp_iterations skipped "
                f"(estimated_iters={est_iters} > max={int(iter_stats_max_iter)})",
            )
            return
    try:
        if solver_label in {"gmres", "lgmres"}:
            if history is None:
                history = _solve_history(
                    solver_label=solver_label,
                    matvec_fn=matvec_fn,
                    b_vec=b_vec,
                    precond_fn=precond_fn,
                    x0_vec=x0_vec,
                    tol_val=tol_val,
                    atol_val=atol_val,
                    restart_val=restart_val,
                    maxiter_val=maxiter_val,
                    precond_side=precond_side,
                )
            iters = len(history or [])
        elif solver_kind_l == "bicgstab":
            _x_hist, _rn, history = bicgstab_solve_with_history_scipy(
                matvec=matvec_fn,
                b=b_vec,
                preconditioner=precond_fn,
                x0=x0_vec,
                tol=tol_val,
                atol=atol_val,
                maxiter=maxiter_val,
                precondition_side=precond_side,
            )
            iters = len(history or [])
        else:
            return
    except Exception as exc:  # noqa: BLE001
        emit(1, f"ksp_iterations unavailable ({type(exc).__name__}: {exc})")
        return
    emit(0, f"ksp_iterations={iters} solver={solver_label}")


def _solve_history(
    *,
    solver_label: str,
    matvec_fn,
    b_vec: jnp.ndarray,
    precond_fn,
    x0_vec: jnp.ndarray | None,
    tol_val: float,
    atol_val: float,
    restart_val: int,
    maxiter_val: int | None,
    precond_side: str,
) -> list[float]:
    if solver_label == "lgmres":
        _x_hist, _rn, history = lgmres_solve_with_history_scipy(
            matvec=matvec_fn,
            b=b_vec,
            preconditioner=precond_fn,
            x0=x0_vec,
            tol=tol_val,
            atol=atol_val,
            restart=restart_val,
            maxiter=maxiter_val,
            precondition_side=precond_side,
        )
        return history
    _x_hist, _rn, history = gmres_solve_with_history_scipy(
        matvec=matvec_fn,
        b=b_vec,
        preconditioner=precond_fn,
        x0=x0_vec,
        tol=tol_val,
        atol=atol_val,
        restart=restart_val,
        maxiter=maxiter_val,
        precondition_side=precond_side,
    )
    return history


def _read_iter_stats_max_iter() -> int:
    env = os.environ.get("SFINCS_JAX_SOLVER_ITER_STATS_MAX_ITER", "").strip()
    try:
        return int(env) if env else 2000
    except ValueError:
        return 2000


__all__ = [
    "RHS1KSPDiagnosticsControls",
    "RHS1KSPHistoryLimits",
    "RHS1KSPIterStatsControls",
    "emit_rhs1_ksp_history",
    "emit_rhs1_ksp_iter_stats",
    "rhs1_fortran_stdout_from_env",
    "rhs1_ksp_diagnostics_controls_from_env",
    "rhs1_ksp_history_limits_from_env",
    "rhs1_ksp_iter_stats_controls_from_env",
]
