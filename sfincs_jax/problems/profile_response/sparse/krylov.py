"""Generic sparse-PC Krylov execution helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .finalization import SparsePCGMRESResult


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


@dataclass(frozen=True)
class SparsePCGMRESContext:
    """Solve-local dependencies for one sparse-PC GMRES attempt."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    restart: int
    tol: float
    atol: float
    precondition_side: str
    factor_dtype: np.dtype
    progress_every: int
    stagnation_abort: bool
    stagnation_min_iter: int
    stagnation_window: int
    stagnation_rel_improvement: float
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]

def run_sparse_pc_gmres_once(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> SparsePCGMRESResult:
    """Run one host sparse-PC GMRES attempt and recompute the true residual."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres solve start "
            f"form={context.pc_form} restart={int(context.restart)} maxiter={int(maxiter)} "
            f"precondition_side={context.precondition_side} "
            f"factor_dtype={np.dtype(context.factor_dtype).name}",
        )

    solve_start_s = float(context.elapsed_s())
    stagnation_best = float("inf")
    stagnation_best_iter = 0

    def _progress_callback(iteration: int, residual_norm: float) -> None:
        nonlocal stagnation_best, stagnation_best_iter
        iteration_i = int(iteration)
        residual_f = float(residual_norm)
        if np.isfinite(residual_f) and (
            not np.isfinite(stagnation_best)
            or residual_f < stagnation_best * (1.0 - float(context.stagnation_rel_improvement))
        ):
            stagnation_best = float(residual_f)
            stagnation_best_iter = int(iteration_i)
        if (
            bool(context.stagnation_abort)
            and iteration_i >= int(context.stagnation_min_iter)
            and iteration_i - int(stagnation_best_iter) >= int(context.stagnation_window)
        ):
            raise RuntimeError(
                "sparse_pc_gmres stagnation detected: "
                f"iters={iteration_i} best_iter={int(stagnation_best_iter)} "
                f"best_ksp_residual={float(stagnation_best):.6e} "
                f"current_ksp_residual={residual_f:.6e} "
                f"window={int(context.stagnation_window)} "
                f"rel_improvement={float(context.stagnation_rel_improvement):.3e}"
            )
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if iteration_i % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres "
            f"iters={iteration_i} ksp_residual={residual_f:.6e} "
            f"elapsed_s={float(context.elapsed_s()):.3f}",
        )

    preconditioned_residual_norm = float("nan")
    if context.pc_form in {"explicit_left", "petsc_left"}:
        x_np, residual_norm, preconditioned_residual_norm, history = context.explicit_left_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            progress_callback=_progress_callback,
        )
    else:
        x_np, residual_norm, history = context.gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(maxiter),
            precondition_side=context.precondition_side,
            progress_callback=_progress_callback,
        )

    solve_s = float(context.elapsed_s()) - solve_start_s
    try:
        residual_true = np.asarray(context.rhs, dtype=np.float64) - np.asarray(
            jax.device_get(context.matvec(jnp.asarray(x_np, dtype=jnp.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(residual_norm)

    return SparsePCGMRESResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(preconditioned_residual_norm),
        history=tuple(float(v) for v in (history or ())),
        solve_s=float(solve_s),
    )

def run_sparse_pc_gmres_once_for_retry(
    *,
    context: SparsePCGMRESContext,
    x0: jnp.ndarray | np.ndarray | None,
    maxiter: int,
) -> tuple[np.ndarray, float, float, tuple[float, ...], float]:
    """Run sparse-PC GMRES and return the tuple contract used by dtype retry."""

    result = run_sparse_pc_gmres_once(
        context=context,
        x0=x0,
        maxiter=int(maxiter),
    )
    return (
        result.x,
        float(result.residual_norm),
        float(result.preconditioned_residual_norm),
        tuple(float(value) for value in result.history),
        float(result.solve_s),
    )


__all__ = (
    "SparsePCGMRESContext",
    "run_sparse_pc_gmres_once",
    "run_sparse_pc_gmres_once_for_retry",
)
