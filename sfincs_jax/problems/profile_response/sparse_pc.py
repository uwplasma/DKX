"""Host sparse-PC Krylov helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .setup import (
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
)


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


@dataclass(frozen=True)
class SparsePCGMRESResult:
    """Measured result from one sparse-PC GMRES attempt."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    solve_s: float


@dataclass(frozen=True)
class SparsePCPostMinresContext:
    """Solve-local dependencies for the optional sparse-PC residual polish."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    steps: int
    alpha_clip: float
    min_improvement: float
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]]


@dataclass(frozen=True)
class SparsePCPostMinresResult:
    """Result of the optional sparse-PC post-minres polish."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    alphas: tuple[float, ...]
    residual_before: float
    residual_after: float | None
    error: str | None
    solve_s: float


@dataclass(frozen=True)
class SparsePCEntryPolicySetup:
    """Physics classification and GMRES budget for RHSMode=1 sparse-PC paths."""

    constrained_pas_pc: bool
    tokamak_pas_noer_pc: bool
    tokamak_pas_er_pc: bool
    tokamak_fp_er_pc: bool
    tokamak_fp_noer_pc: bool
    tokamak_fp_pc: bool
    xblock_sparse_pc: bool
    fortran_reduced_sparse_pc: bool
    sparse_pc_use_active_dof: bool
    xblock_use_active_dof: bool
    sparse_pc_fp_dense_velocity_block: bool | None
    pc_restart_env: str
    pc_restart: int
    pc_maxiter: int


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    source = env if env is not None else {}
    return str(source.get(key, "")).strip()


def resolve_sparse_pc_entry_policy(
    *,
    op: object,
    solve_method_kind: str,
    has_reduced_modes: bool,
    use_active_dof_mode: bool,
    xblock_active_dof_requested: bool,
    active_maps_available: bool,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    er_abs_sparse_pc: float,
    restart: int,
    maxiter: int | None,
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    sparse_pc_default_restart: Callable[..., int],
    env: Mapping[str, str] | None = None,
) -> SparsePCEntryPolicySetup:
    """Resolve the entry policy for host sparse-PC GMRES RHSMode=1 solves."""

    constrained_pas_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 2
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and op.fblock.fp is None
    )
    tokamak_pas_noer_pc = bool(
        constrained_pas_pc
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) == 0.0
    )
    tokamak_pas_er_pc = bool(
        constrained_pas_pc
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) > 0.0
        and (bool(use_dkes) or bool(include_xdot_sparse_pc) or bool(include_electric_field_xi_sparse_pc))
    )
    tokamak_fp_er_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) > 0.0
        and (bool(use_dkes) or bool(include_xdot_sparse_pc) or bool(include_electric_field_xi_sparse_pc))
    )
    tokamak_fp_noer_pc = bool(
        int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 0
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) == 1
        and float(er_abs_sparse_pc) == 0.0
    )
    tokamak_fp_pc = bool(tokamak_fp_er_pc or tokamak_fp_noer_pc)
    xblock_sparse_pc = solve_method_kind in SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS
    fortran_reduced_sparse_pc = solve_method_kind in SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS

    sparse_pc_active_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF").lower()
    sparse_pc_active_forced_on = sparse_pc_active_env in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    sparse_pc_active_forced_off = sparse_pc_active_env in {"0", "false", "f", "no", "off", ".false.", ".f."}
    sparse_pc_active_auto = sparse_pc_active_env in {"", "auto"}
    sparse_pc_use_active_dof = bool(
        (not xblock_sparse_pc)
        and bool(has_reduced_modes)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and (
            sparse_pc_active_forced_on
            or (
                sparse_pc_active_auto
                and (tokamak_pas_er_pc or tokamak_pas_noer_pc or fortran_reduced_sparse_pc)
            )
        )
        and (not sparse_pc_active_forced_off)
    )
    xblock_use_active_dof = bool(
        xblock_sparse_pc
        and bool(use_active_dof_mode)
        and bool(xblock_active_dof_requested)
        and bool(active_maps_available)
    )
    if bool(use_active_dof_mode) and not (sparse_pc_use_active_dof or xblock_use_active_dof):
        raise NotImplementedError(
            "solve_method='sparse_pc_gmres'/'xblock_sparse_pc_gmres' active-DOF mode is only implemented "
            "for the generic sparse_pc_gmres branch or opt-in x-block branch. Set "
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF=1, "
            "SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF=1, or SFINCS_JAX_ACTIVE_DOF=0."
        )

    fp_dense_velocity_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK").lower()
    if fp_dense_velocity_env in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        sparse_pc_fp_dense_velocity_block: bool | None = False
    elif fp_dense_velocity_env in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        sparse_pc_fp_dense_velocity_block = True
    else:
        sparse_pc_fp_dense_velocity_block = None

    pc_restart_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART")
    pc_restart, pc_maxiter = parse_polish_gmres_config(
        restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART",
        maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER",
        default_restart=max(20, int(restart)),
        default_maxiter=max(100, int(maxiter) if maxiter is not None else 400),
        min_restart=2,
        min_maxiter=1,
    )
    pc_restart = sparse_pc_default_restart(
        requested_restart=int(pc_restart),
        restart_env_value=pc_restart_env,
        tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
        n_species=int(op.n_species),
    )

    return SparsePCEntryPolicySetup(
        constrained_pas_pc=bool(constrained_pas_pc),
        tokamak_pas_noer_pc=bool(tokamak_pas_noer_pc),
        tokamak_pas_er_pc=bool(tokamak_pas_er_pc),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        tokamak_fp_noer_pc=bool(tokamak_fp_noer_pc),
        tokamak_fp_pc=bool(tokamak_fp_pc),
        xblock_sparse_pc=bool(xblock_sparse_pc),
        fortran_reduced_sparse_pc=bool(fortran_reduced_sparse_pc),
        sparse_pc_use_active_dof=bool(sparse_pc_use_active_dof),
        xblock_use_active_dof=bool(xblock_use_active_dof),
        sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        pc_restart_env=str(pc_restart_env),
        pc_restart=int(pc_restart),
        pc_maxiter=int(pc_maxiter),
    )


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


def apply_sparse_pc_post_minres(
    *,
    context: SparsePCPostMinresContext,
    x: np.ndarray,
    residual_norm: float,
    preconditioned_residual_norm: float,
) -> SparsePCPostMinresResult:
    """Apply the optional sparse-PC minimum-residual polish and gate acceptance."""

    residual_before = float(residual_norm)
    post_minres_start_s = float(context.elapsed_s())
    history: tuple[float, ...] = ()
    alphas: tuple[float, ...] = ()
    residual_after: float | None = None
    error: str | None = None
    x_out = np.asarray(x, dtype=np.float64)
    rn_out = float(residual_norm)
    rn_pc_out = float(preconditioned_residual_norm)

    try:
        x_post_minres, residual_post_minres, post_history, post_alphas = context.minres_correction(
            matvec=context.matvec,
            rhs=context.rhs,
            x0=jnp.asarray(x_out, dtype=jnp.float64),
            preconditioner=context.preconditioner,
            steps=int(context.steps),
            alpha_clip=float(context.alpha_clip),
            min_improvement=float(context.min_improvement),
        )
        history = tuple(float(v) for v in post_history)
        alphas = tuple(float(v) for v in post_alphas)
        residual_after = float(jnp.linalg.norm(residual_post_minres))
        if np.isfinite(float(residual_after)) and float(residual_after) < float(rn_out):
            x_out = np.asarray(x_post_minres, dtype=np.float64)
            rn_out = float(residual_after)
            if context.pc_form in {"explicit_left", "petsc_left"}:
                try:
                    residual_pc = context.preconditioner(
                        context.rhs - context.matvec(jnp.asarray(x_out, dtype=jnp.float64))
                    )
                    rn_pc_out = float(jnp.linalg.norm(residual_pc))
                except Exception:
                    pass
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres "
                    f"improved residual {residual_before:.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(accepted_steps={len(alphas)})",
                )
        elif context.emit is not None:
            after = float(residual_after) if residual_after is not None else float("nan")
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres "
                f"rejected residual {residual_before:.6e} -> {after:.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres post-minres failed "
                f"({error})",
            )

    return SparsePCPostMinresResult(
        x=x_out,
        residual_norm=float(rn_out),
        preconditioned_residual_norm=float(rn_pc_out),
        history=history,
        alphas=alphas,
        residual_before=float(residual_before),
        residual_after=residual_after,
        error=error,
        solve_s=float(context.elapsed_s()) - post_minres_start_s,
    )


__all__ = [
    "SparsePCGMRESContext",
    "SparsePCGMRESResult",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "apply_sparse_pc_post_minres",
    "run_sparse_pc_gmres_once",
]
