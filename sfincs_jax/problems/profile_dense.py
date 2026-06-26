"""Host dense reduced-system helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
import os
import time
from typing import Any

import jax
import jax.scipy.linalg as jla
import jax.numpy as jnp
import numpy as np

from ..namelist import Namelist
from ..operators.profile_full_system import solve_structured_rhs1_full_csr
from ..solvers.implicit import linear_custom_solve, linear_custom_solve_with_residual
from ..solvers.krylov_dispatch import gmres_solve_dispatch, rhs_krylov_method_for_context
from ..solver import (
    GMRESSolveResult,
    assemble_dense_matrix_from_matvec,
    bicgstab_solve_with_history_scipy,
    bicgstab_solve_with_residual,
    bicgstab_solve_with_residual_jit,
    dense_krylov_solve_from_matrix_with_residual,
    dense_solve_from_matrix,
    dense_solve_from_matrix_row_scaled,
    explicit_left_preconditioned_gmres_scipy,
    gmres_solve_with_history_scipy,
    gmres_solve_with_residual,
    gmres_solve_with_residual_distributed,
    gmres_solve_with_residual_jit,
)
from sfincs_jax.operators.profile_system import (
    V3FullSystemOperator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
    sharding_constraints,
    with_transport_rhs_settings,
)
from .transport_linear_system import transport_active_dof_indices
from .profile_policies import (
    rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend,
    rhsmode1_scipy_rescue_active_size_allowed_current_backend,
    rhs1_scipy_rescue_controls_from_env,
)
from .profile_residual import result_with_true_residual, true_residual_norm_or_inf
from .profile_solver_diagnostics import V3LinearSolveResult


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


@dataclass(frozen=True)
class RHS1ScipyRescueStageContext:
    """Driver-independent policy and state for the CPU SciPy rescue phase."""

    op: Any
    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    strong_preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    preconditioner_name: str | None
    strong_preconditioner_name: str | None
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precond_side: str
    active_size: int
    used_large_cpu_xblock_shortcut: bool
    used_explicit_fp_xblock_seed: bool
    use_implicit: bool
    skip_global_sparse_after_xblock: bool
    elapsed_s: Callable[[], float]
    emit: Callable[[int, str], None] | None
    mark: Callable[[str], None]


@dataclass(frozen=True)
class RHS1ScipyRescueStageResult:
    """Updated result and metadata produced by the SciPy rescue phase."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    metadata: dict[str, object]


@dataclass(frozen=True)
class RHS1Constraint0PETScCompatSolveContext:
    """Inputs for the constraintScheme=0 PETSc-compatible sparse-ILU solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    active_size: int
    tol: float
    atol: float
    sparse_drop_tol: float
    sparse_drop_rel: float
    config: Any
    regularization: Callable[[float], float]


@dataclass(frozen=True)
class RHS1Constraint0PETScCompatSolveOutcome:
    """Result and replay system for the constraintScheme=0 PETSc-compatible solve."""

    result: GMRESSolveResult
    replay_matvec: Callable[[jnp.ndarray], jnp.ndarray]
    replay_rhs: jnp.ndarray
    true_residual: float
    preconditioned_residual: float
    rhs_pc_norm: float
    drop_threshold: float
    regularization: float
    nnz: int


@dataclass(frozen=True)
class RHS1DenseKSPFullSolveContext:
    """Inputs for the full-system RHSMode=1 dense-KSP solve path."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    total_size: int
    phi1_size: int
    n_species: int
    n_theta: int
    n_zeta: int
    nxi_for_x: object
    extra_size: int
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_linear: Callable[..., GMRESSolveResult]


@dataclass(frozen=True)
class RHS1DenseKSPReducedSolveContext:
    """Inputs for the reduced active-DOF RHSMode=1 dense-KSP solve path."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    active_size: int
    phi1_size: int
    n_species: int
    n_theta: int
    n_zeta: int
    nxi_for_x: object
    extra_size: int
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_linear: Callable[..., GMRESSolveResult]
    result_ready: Callable[[GMRESSolveResult], GMRESSolveResult]


@dataclass(frozen=True)
class RHS1DenseKSPFullSolveOutcome:
    """Physical result plus the preconditioned replay system."""

    result: GMRESSolveResult
    replay_matvec: Callable[[jnp.ndarray], jnp.ndarray]
    replay_rhs: jnp.ndarray


@dataclass(frozen=True)
class RHS1DenseKSPReducedSolveOutcome:
    """Reduced dense-KSP result plus the preconditioned replay system."""

    result: GMRESSolveResult
    replay_matvec: Callable[[jnp.ndarray], jnp.ndarray]
    replay_rhs: jnp.ndarray


def rhs1_small_gmres_max_from_env(*, default: int = 600) -> int:
    """Return the size cutoff for small-system GMRES auto routing."""

    env = os.environ.get("SFINCS_JAX_RHSMODE1_GMRES_SMALL_MAX", "").strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


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


def run_rhs1_scipy_rescue_stage(
    context: RHS1ScipyRescueStageContext,
) -> RHS1ScipyRescueStageResult:
    """Apply the bounded CPU SciPy rescue policy and return metadata updates.

    The solve entry point owns the surrounding active-DOF orchestration. This
    stage owns the last-resort host SciPy rescue decision, progress messages,
    and metadata contract, keeping the non-differentiable CLI robustness path
    explicit and testable.
    """

    result = context.result
    metadata: dict[str, object] = {}
    if (
        bool(context.use_implicit)
        or jax.default_backend() != "cpu"
        or int(context.op.rhs_mode) != 1
        or bool(context.op.include_phi1)
        or float(result.residual_norm) <= float(context.target)
    ):
        return RHS1ScipyRescueStageResult(
            result=result,
            residual_vec=context.residual_vec,
            metadata=metadata,
        )
    if bool(context.skip_global_sparse_after_xblock):
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: skipping SciPy rescue after "
                "bounded x-block seed "
                f"(residual={float(result.residual_norm):.3e}; not accepted as converged)",
            )
        return RHS1ScipyRescueStageResult(
            result=result,
            residual_vec=context.residual_vec,
            metadata=metadata,
        )

    controls = rhs1_scipy_rescue_controls_from_env(
        restart=int(context.restart),
        maxiter=context.maxiter,
    )
    if not controls.enabled:
        return RHS1ScipyRescueStageResult(
            result=result,
            residual_vec=context.residual_vec,
            metadata=metadata,
        )

    rescue_abs_floor = rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend(
        op=context.op,
        active_size=int(context.active_size),
        used_large_cpu_xblock_shortcut=bool(context.used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(context.used_explicit_fp_xblock_seed),
        use_implicit=bool(context.use_implicit),
    )
    rescue_threshold = max(
        float(context.target) * float(controls.ratio),
        float(rescue_abs_floor),
    )
    size_allowed = rhsmode1_scipy_rescue_active_size_allowed_current_backend(
        op=context.op,
        active_size=int(context.active_size),
        used_large_cpu_xblock_shortcut=bool(context.used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(context.used_explicit_fp_xblock_seed),
        use_implicit=bool(context.use_implicit),
    )
    if (not bool(size_allowed)) and float(result.residual_norm) > float(rescue_threshold):
        context.mark("rhs1_scipy_rescue_skipped")
        metadata.update(
            {
                "scipy_rescue_attempted": False,
                "scipy_rescue_skipped": True,
                "scipy_rescue_skip_reason": "active_size_cap",
                "scipy_rescue_initial_residual": float(result.residual_norm),
                "scipy_rescue_target": float(context.target),
                "scipy_rescue_threshold": float(rescue_threshold),
                "scipy_rescue_active_size": int(context.active_size),
                "scipy_rescue_used_large_cpu_xblock_shortcut": bool(
                    context.used_large_cpu_xblock_shortcut
                ),
                "scipy_rescue_used_explicit_fp_xblock_seed": bool(
                    context.used_explicit_fp_xblock_seed
                ),
            }
        )
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: skipping SciPy rescue "
                f"(active_size={int(context.active_size)} exceeds default rescue cap; "
                "set SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE=0 to force)",
            )
        return RHS1ScipyRescueStageResult(
            result=result,
            residual_vec=context.residual_vec,
            metadata=metadata,
        )

    if float(result.residual_norm) <= float(rescue_threshold):
        if context.emit is not None and float(rescue_abs_floor) > 0.0:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: skipping SciPy rescue after "
                "x-block seed "
                f"(residual={float(result.residual_norm):.3e} <= "
                f"floor={float(rescue_abs_floor):.1e})",
            )
        return RHS1ScipyRescueStageResult(
            result=result,
            residual_vec=context.residual_vec,
            metadata=metadata,
        )

    rescue_preconditioner = context.preconditioner
    rescue_preconditioner_name = context.preconditioner_name or "none"
    using_strong_preconditioner = (
        bool(controls.use_strong) and context.strong_preconditioner is not None
    )
    if using_strong_preconditioner:
        rescue_preconditioner = context.strong_preconditioner
        rescue_preconditioner_name = context.strong_preconditioner_name or "strong"
    rescue_method = str(controls.method)
    if rescue_method == "auto":
        rescue_method = "bicgstab" if using_strong_preconditioner else "gmres"

    try:
        rescue_start_s = float(context.elapsed_s())
        rescue_initial_residual = float(result.residual_norm)
        context.mark("rhs1_scipy_rescue_start")
        metadata.update(
            {
                "scipy_rescue_attempted": True,
                "scipy_rescue_method": str(rescue_method),
                "scipy_rescue_preconditioner": str(rescue_preconditioner_name),
                "scipy_rescue_restart": int(controls.restart),
                "scipy_rescue_maxiter": int(controls.maxiter),
                "scipy_rescue_initial_residual": float(rescue_initial_residual),
                "scipy_rescue_target": float(context.target),
                "scipy_rescue_threshold": float(rescue_threshold),
                "scipy_rescue_start_s": float(rescue_start_s),
            }
        )
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: SciPy rescue "
                f"(residual={float(result.residual_norm):.3e} > "
                f"{float(controls.ratio):.1e}x target={float(context.target):.3e} "
                f"method={rescue_method} restart={int(controls.restart)} "
                f"maxiter={int(controls.maxiter)} "
                f"preconditioner={rescue_preconditioner_name})",
            )
        outcome = run_rhs1_scipy_rescue(
            context=RHS1ScipyRescueContext(
                matvec=context.matvec,
                rhs=context.rhs,
                x0=result.x,
                preconditioner=rescue_preconditioner,
                method=rescue_method,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(controls.restart),
                maxiter=int(controls.maxiter),
                precond_side=context.precond_side,
            ),
            emit=context.emit,
        )
        rescue_elapsed_s = float(context.elapsed_s() - rescue_start_s)
        rescue_final_residual = float(outcome.result.residual_norm)
        context.mark("rhs1_scipy_rescue_done")
        metadata.update(
            {
                "scipy_rescue_elapsed_s": float(rescue_elapsed_s),
                "scipy_rescue_final_residual": float(rescue_final_residual),
                "scipy_rescue_reported_residual": float(outcome.reported_residual),
                "scipy_rescue_history_len": int(outcome.history_len),
                "scipy_rescue_improved": bool(
                    rescue_final_residual < rescue_initial_residual
                ),
            }
        )
        if float(outcome.result.residual_norm) < float(result.residual_norm):
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: SciPy rescue improved "
                    "residual "
                    f"{float(result.residual_norm):.3e} -> "
                    f"{float(outcome.result.residual_norm):.3e}",
                )
            result = outcome.result
    except Exception as exc:  # noqa: BLE001
        context.mark("rhs1_scipy_rescue_failed")
        metadata.update(
            {
                "scipy_rescue_failed": True,
                "scipy_rescue_error": f"{type(exc).__name__}: {exc}",
            }
        )
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: SciPy rescue failed "
                f"({type(exc).__name__}: {exc})",
            )

    return RHS1ScipyRescueStageResult(
        result=result,
        residual_vec=context.residual_vec,
        metadata=metadata,
    )


def solve_rhs1_constraint0_petsc_compat(
    context: RHS1Constraint0PETScCompatSolveContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1Constraint0PETScCompatSolveOutcome:
    """Run the host sparse-ILU PETSc-compatibility lane for constraintScheme=0."""

    import scipy.sparse as sp  # noqa: PLC0415
    from scipy.sparse.csgraph import reverse_cuthill_mckee  # noqa: PLC0415
    from scipy.sparse.linalg import spilu  # noqa: PLC0415

    config = context.config
    drop_tol = float(config.drop_tol)
    fill = float(config.fill)
    diag_pivot = float(config.diag_pivot)
    restart = int(config.restart)
    maxiter = int(config.maxiter)
    active_size = int(context.active_size)

    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat sparse ILU solve "
            f"(size={active_size} drop_tol={drop_tol:.1e} fill={fill:.1f})",
        )

    a_dense = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=active_size,
        dtype=context.rhs.dtype,
    )
    a_np = np.asarray(a_dense, dtype=np.float64)
    max_abs = float(np.max(np.abs(a_np))) if a_np.size else 0.0
    drop_threshold = max(float(context.sparse_drop_tol), float(context.sparse_drop_rel) * max_abs)
    if drop_threshold > 0.0:
        a_np = a_np.copy()
        a_np[np.abs(a_np) < drop_threshold] = 0.0
    a_csr = sp.csr_matrix(a_np)
    a_csr.eliminate_zeros()
    max_abs = float(np.max(np.abs(a_csr.data))) if int(a_csr.nnz) > 0 else 0.0
    regularization = float(context.regularization(max_abs))
    perm = np.asarray(
        reverse_cuthill_mckee(a_csr, symmetric_mode=False),
        dtype=np.int32,
    )
    inv_perm = np.argsort(perm).astype(np.int32, copy=False)
    a_perm = a_csr[perm][:, perm].tocsc()
    if regularization != 0.0:
        diag_idx = np.arange(active_size, dtype=np.int32)
        a_perm = a_perm.copy()
        a_perm[diag_idx, diag_idx] = a_perm[diag_idx, diag_idx] + regularization
    ilu = spilu(
        a_perm,
        drop_tol=drop_tol,
        fill_factor=fill,
        permc_spec="NATURAL",
        diag_pivot_thresh=diag_pivot,
    )
    rhs_perm = jnp.asarray(
        np.asarray(context.rhs, dtype=np.float64)[perm],
        dtype=jnp.float64,
    )

    def mv_perm(v: jnp.ndarray) -> jnp.ndarray:
        x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
        return jnp.asarray(a_perm @ x_np, dtype=jnp.float64)

    def precond_perm(v: jnp.ndarray) -> jnp.ndarray:
        x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
        return jnp.asarray(ilu.solve(x_np), dtype=jnp.float64)

    rhs_pc_perm_np = np.asarray(precond_perm(rhs_perm), dtype=np.float64)
    rhs_pc_norm = float(np.linalg.norm(rhs_pc_perm_np))
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat rhs_pc "
            f"norm={rhs_pc_norm:.3e} finite={bool(np.all(np.isfinite(rhs_pc_perm_np)))} "
            f"drop={drop_threshold:.3e} reg={regularization:.3e} nnz={int(a_csr.nnz)}",
        )
    rhs_perm_norm = float(np.linalg.norm(np.asarray(rhs_perm, dtype=np.float64)))
    rhs_pc_zero_tol = max(float(context.atol), max(1.0, rhs_perm_norm) * float(context.tol))
    if np.isfinite(rhs_pc_norm) and rhs_pc_norm <= rhs_pc_zero_tol:
        x_perm_np = np.zeros((active_size,), dtype=np.float64)
        true_residual = rhs_perm_norm
        preconditioned_residual = rhs_pc_norm
    else:
        x_perm_np, true_residual, preconditioned_residual, _history = (
            explicit_left_preconditioned_gmres_scipy(
                matvec=mv_perm,
                b=rhs_perm,
                preconditioner=precond_perm,
                x0=None,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=min(active_size, max(1, restart)),
                maxiter=max(1, maxiter),
            )
        )
    x_np = np.asarray(x_perm_np, dtype=np.float64)[inv_perm]
    rhs_pc_np = rhs_pc_perm_np[inv_perm]

    def mv_pc_full(v: jnp.ndarray) -> jnp.ndarray:
        x_np_local = np.asarray(v, dtype=np.float64).reshape((-1,))
        y_perm = np.asarray(a_perm @ x_np_local[perm], dtype=np.float64)
        z_perm = ilu.solve(y_perm)
        return jnp.asarray(z_perm[inv_perm], dtype=jnp.float64)

    result = GMRESSolveResult(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(preconditioned_residual, dtype=jnp.float64),
    )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat residuals "
            f"preconditioned={preconditioned_residual:.3e} true={true_residual:.3e}",
        )
    return RHS1Constraint0PETScCompatSolveOutcome(
        result=result,
        replay_matvec=mv_pc_full,
        replay_rhs=jnp.asarray(rhs_pc_np, dtype=jnp.float64),
        true_residual=float(true_residual),
        preconditioned_residual=float(preconditioned_residual),
        rhs_pc_norm=float(rhs_pc_norm),
        drop_threshold=float(drop_threshold),
        regularization=float(regularization),
        nnz=int(a_csr.nnz),
    )


def solve_rhs1_dense_ksp_full(
    context: RHS1DenseKSPFullSolveContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1DenseKSPFullSolveOutcome:
    """Run the host dense-KSP branch for full RHSMode=1 systems.

    This path assembles the full operator, builds PETSc-like species blocks,
    solves the left-preconditioned dense system, and reports the physical
    residual. Replay-state mutation remains in the driver.
    """

    if int(context.phi1_size) != 0:
        raise NotImplementedError(
            "dense_ksp is only supported for includePhi1=false RHSMode=1 solves."
        )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: assembling dense full matrix for dense_ksp",
        )
    a_dense = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.total_size),
        dtype=context.rhs.dtype,
    )

    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: building PETSc-like species-block "
            "preconditioner (dense_ksp)",
        )

    n_species = int(context.n_species)
    n_theta = int(context.n_theta)
    n_zeta = int(context.n_zeta)
    local_per_species = int(np.sum(np.asarray(context.nxi_for_x, dtype=np.int64)))
    dke_size = int(local_per_species * n_theta * n_zeta)
    extra_size = int(context.extra_size)
    extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
    if extra_size and (extra_per_species * n_species != extra_size):
        extra_per_species = 0

    f_size = int(n_species * dke_size)
    expected_size = int(f_size + int(context.phi1_size) + extra_size)
    if int(context.total_size) != expected_size:
        raise RuntimeError(
            f"dense_ksp expects total_size={expected_size}, got {int(context.total_size)}"
        )

    lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    idx_blocks: list[jnp.ndarray] = []
    for species_index in range(n_species):
        f_idx = np.arange(
            species_index * dke_size,
            (species_index + 1) * dke_size,
            dtype=np.int32,
        )
        extra_idx = np.arange(
            f_size + species_index * extra_per_species,
            f_size + (species_index + 1) * extra_per_species,
            dtype=np.int32,
        )
        block_idx_np = (
            np.concatenate([f_idx, extra_idx], axis=0)
            if extra_per_species
            else f_idx
        )
        block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
        a_block = a_dense[jnp.ix_(block_idx, block_idx)]
        lu, piv = jla.lu_factor(a_block)
        lu_factors.append((lu, piv))
        idx_blocks.append(block_idx)

    def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(v)
        for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
            rhs_block = v[block_idx]
            sol_block = jla.lu_solve((lu, piv), rhs_block)
            out = out.at[block_idx].set(sol_block, unique_indices=True)
        return out

    def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
        return a_dense @ x

    rhs_pc = preconditioner_dense(context.rhs)

    def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
        return preconditioner_dense(mv_dense(x))

    res_pc = context.solve_linear(
        matvec_fn=mv_pc,
        b_vec=rhs_pc,
        precond_fn=None,
        x0_vec=context.x0,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        restart_val=int(context.restart),
        maxiter_val=context.maxiter,
        solve_method_val="incremental",
        precond_side="none",
    )
    residual_norm_full = jnp.linalg.norm(context.matvec(res_pc.x) - context.rhs)
    result = GMRESSolveResult(x=res_pc.x, residual_norm=residual_norm_full)
    return RHS1DenseKSPFullSolveOutcome(
        result=result,
        replay_matvec=mv_pc,
        replay_rhs=rhs_pc,
    )


def solve_rhs1_dense_ksp_reduced(
    context: RHS1DenseKSPReducedSolveContext,
    *,
    emit: Callable[[int, str], None] | None = None,
) -> RHS1DenseKSPReducedSolveOutcome:
    """Run the reduced active-DOF dense-KSP branch.

    This mirrors the full-system dense-KSP path but preserves the reduced
    branch's historical residual semantics: the returned result is the
    left-preconditioned solve result after the driver's result-ready hook.
    """

    if int(context.phi1_size) != 0:
        raise NotImplementedError(
            "dense_ksp is only supported for includePhi1=false RHSMode=1 solves."
        )
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: assembling dense reduced matrix for dense_ksp",
        )
    a_dense = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.active_size),
        dtype=context.rhs.dtype,
    )

    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: building PETSc-like species-block "
            "preconditioner (dense_ksp)",
        )

    n_species = int(context.n_species)
    n_theta = int(context.n_theta)
    n_zeta = int(context.n_zeta)
    local_per_species = int(np.sum(np.asarray(context.nxi_for_x, dtype=np.int64)))
    dke_size = int(local_per_species * n_theta * n_zeta)
    extra_size = int(context.extra_size)
    extra_per_species = int(extra_size // max(1, n_species)) if extra_size else 0
    if extra_size and (extra_per_species * n_species != extra_size):
        extra_per_species = 0

    f_size = int(n_species * dke_size)
    expected_active = int(f_size + int(context.phi1_size) + extra_size)
    if int(context.active_size) != expected_active:
        raise RuntimeError(
            f"dense_ksp expects active_size={expected_active}, got {int(context.active_size)}"
        )

    lu_factors: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    idx_blocks: list[jnp.ndarray] = []
    for species_index in range(n_species):
        f_idx = np.arange(
            species_index * dke_size,
            (species_index + 1) * dke_size,
            dtype=np.int32,
        )
        extra_idx = np.arange(
            f_size + species_index * extra_per_species,
            f_size + (species_index + 1) * extra_per_species,
            dtype=np.int32,
        )
        block_idx_np = (
            np.concatenate([f_idx, extra_idx], axis=0)
            if extra_per_species
            else f_idx
        )
        block_idx = jnp.asarray(block_idx_np, dtype=jnp.int32)
        a_block = a_dense[jnp.ix_(block_idx, block_idx)]
        lu, piv = jla.lu_factor(a_block)
        lu_factors.append((lu, piv))
        idx_blocks.append(block_idx)

    def preconditioner_dense(v: jnp.ndarray) -> jnp.ndarray:
        out = jnp.zeros_like(v)
        for block_idx, (lu, piv) in zip(idx_blocks, lu_factors, strict=True):
            rhs_block = v[block_idx]
            sol_block = jla.lu_solve((lu, piv), rhs_block)
            out = out.at[block_idx].set(sol_block, unique_indices=True)
        return out

    def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
        return a_dense @ x

    rhs_pc = preconditioner_dense(context.rhs)

    def mv_pc(x: jnp.ndarray) -> jnp.ndarray:
        return preconditioner_dense(mv_dense(x))

    result = context.solve_linear(
        matvec_fn=mv_pc,
        b_vec=rhs_pc,
        precond_fn=None,
        x0_vec=context.x0,
        tol_val=float(context.tol),
        atol_val=float(context.atol),
        restart_val=int(context.restart),
        maxiter_val=context.maxiter,
        solve_method_val="incremental",
        precond_side="none",
    )
    result = context.result_ready(result)
    return RHS1DenseKSPReducedSolveOutcome(
        result=result,
        replay_matvec=mv_pc,
        replay_rhs=rhs_pc,
    )

@dataclass(frozen=True)
class HostDenseReducedSolveContext:
    """Solve-local inputs for a host dense reduced RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    active_size: int
    constraint_scheme: int
    has_fp: bool
    dense_matrix_cache: np.ndarray | None = None


@dataclass(frozen=True)
class HostDenseFullSolveContext:
    """Solve-local inputs for a host dense full-system RHSMode=1 solve."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    total_size: int


@dataclass(frozen=True)
class RHS1ReducedHostDenseShortcutContext:
    """Inputs for reduced-system host dense shortcut execution."""

    enabled: bool
    solve_context: HostDenseReducedSolveContext
    current_result: GMRESSolveResult | None
    x0: jnp.ndarray | None
    active_size: int
    early_dense_shortcut: bool
    probe_shortcut: bool


@dataclass(frozen=True)
class RHS1ReducedHostDenseShortcutResult:
    """Outputs from reduced-system host dense shortcut execution."""

    result: GMRESSolveResult
    early_dense_shortcut: bool
    probe_shortcut: bool


@dataclass(frozen=True)
class RHS1FullHostDenseShortcutContext:
    """Inputs for full-system host dense shortcut execution."""

    enabled: bool
    solve_context: HostDenseFullSolveContext
    current_result: GMRESSolveResult | None
    current_residual_vec: jnp.ndarray | None
    x0: jnp.ndarray | None
    total_size: int


@dataclass(frozen=True)
class RHS1FullHostDenseShortcutResult:
    """Outputs from full-system host dense shortcut execution."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None


@dataclass(frozen=True)
class RHS1ReducedDenseFallbackCandidateContext:
    """Inputs for the reduced RHSMode=1 dense fallback candidate.

    This is the richer post-primary fallback used by the v3 driver after a
    matrix-free reduced solve stalls. It intentionally supports both
    host/non-autodiff LU and JAX-visible dense Krylov lanes so the CLI can use a
    fast host path while implicit-differentiation callers still have a JAX
    custom-linear-solve contract.
    """

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    x0: jnp.ndarray
    active_size: int
    constraint_scheme: int
    has_fp: bool
    has_pas: bool
    dense_matrix_cache: np.ndarray | jnp.ndarray | None
    dense_backend_allowed: bool
    use_implicit: bool
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    gmres_precond_side: str
    backend: str | None = None


@dataclass(frozen=True)
class RHS1ReducedDenseFallbackStageContext:
    """Inputs for the reduced-system dense fallback execution/acceptance stage."""

    candidate_context: RHS1ReducedDenseFallbackCandidateContext
    current_result: GMRESSolveResult
    current_residual_vec: jnp.ndarray | None
    target: float


@dataclass(frozen=True)
class RHS1ReducedDenseFallbackAdmissionStageContext:
    """Inputs for reduced dense fallback admission plus execution handoff."""

    stage_context: RHS1ReducedDenseFallbackStageContext
    dense_fallback_max: int
    residual_norm_true: float
    reported_residual_norm: float
    active_size: int
    rhs_mode: int
    include_phi1: bool
    has_fp: bool
    disable_dense_pas: bool
    any_dense_path_allowed: bool
    host_sparse_direct_used: bool
    backend: str
    host_sparse_skip_ratio: float
    cs0_dense_fallback_allowed: bool
    cs0_sparse_first: bool
    cs0_petsc_compat: bool


@dataclass(frozen=True)
class RHS1FullDenseFallbackContext:
    """Inputs for the final full-system RHSMode=1 dense fallback candidate."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    current_result: GMRESSolveResult
    current_residual_vec: jnp.ndarray | None
    total_size: int
    constraint_scheme: int
    dense_matrix_cache: np.ndarray | jnp.ndarray | None
    dense_backend_allowed: bool
    residual_norm_check: float
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    backend: str | None = None


@dataclass(frozen=True)
class RHS1FullDenseFallbackStageContext:
    """Inputs for the full-system dense fallback admission and execution stage."""

    candidate_context: RHS1FullDenseFallbackContext
    dense_fallback_max: int
    residual_norm_true: float
    active_size: int
    rhs_mode: int
    include_phi1: bool
    has_fp: bool
    any_dense_path_allowed: bool
    host_sparse_direct_used: bool
    host_sparse_skip_ratio: float
    cs0_sparse_first: bool


@dataclass(frozen=True)
class RHS1DenseProbeAdmission:
    """Whether the reduced-system dense probe should run."""

    enabled: bool


@dataclass(frozen=True)
class RHS1DenseProbeShortcutDecision:
    """Dense-probe shortcut decision after the probe residual is known."""

    accept_shortcut: bool
    seed_x0_if_missing: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1DenseProbeStageContext:
    """Inputs for the reduced dense-probe shortcut/seed stage."""

    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    rhs: jnp.ndarray
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None
    current_result: GMRESSolveResult | None
    x0_reduced: jnp.ndarray | None
    target: float
    active_size: int
    constraint_scheme: int
    probe_shortcut: bool
    cs0_petsc_compat: bool
    cs0_sparse_first: bool
    cs0_dense_fallback_allowed: bool
    solve_method_kind: str
    solve_method: str
    dense_shortcut_ratio: float
    dense_fallback_max: int
    sparse_prefer_over_dense_shortcut: bool
    gmres_precond_side: str


@dataclass(frozen=True)
class RHS1DenseProbeStageResult:
    """Outputs from the reduced dense-probe shortcut/seed stage."""

    result: GMRESSolveResult | None
    x0_reduced: jnp.ndarray | None
    early_dense_shortcut: bool
    probe_shortcut: bool


@dataclass(frozen=True)
class RHS1DenseShortcutSetup:
    """Dense shortcut/fallback controls after env and backend gates."""

    dense_shortcut_ratio: float
    dense_fallback_max: int
    disable_dense_pas: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1DenseFallbackThresholds:
    """Residual-ratio dense fallback limits resolved from environment controls."""

    dense_fallback_max_huge: int
    dense_fallback_ratio: float
    dense_fallback_limit: int
    dense_fallback_trigger: bool


@dataclass(frozen=True)
class RHS1DenseFallbackAdmission:
    """Resolved dense-fallback admission decision and derived limits."""

    should_run: bool
    dense_fallback_max: int
    dense_fallback_limit: int
    dense_fallback_trigger: bool
    force_dense_cs0: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1EarlyDenseShortcutDecision:
    """Early dense-shortcut state after residual-ratio admission checks."""

    early_dense_shortcut: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1PostKrylovDenseShortcutDecision:
    """Dense-shortcut state after true-residual admission before sparse rescue."""

    dense_shortcut: bool
    messages: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class RHS1PostKrylovDenseShortcutEvaluationContext:
    """Inputs for late dense-shortcut evaluation after primary Krylov retries."""

    dense_shortcut: bool
    dense_shortcut_ratio: float
    current_result: GMRESSolveResult
    rhs: jnp.ndarray
    matvec: Callable[[jnp.ndarray], jnp.ndarray]
    target: float
    dense_fallback_max: int
    active_size: int
    constraint_scheme: int
    cs0_sparse_first: bool
    sparse_prefer_over_dense_shortcut: bool
    sparse_exact_direct: bool


@dataclass(frozen=True)
class RHS1PostKrylovDenseShortcutEvaluation:
    """Late dense-shortcut result with optional true-residual diagnostics."""

    dense_shortcut: bool
    residual_norm_true: float | None = None
    residual_ratio: float | None = None
    messages: tuple[tuple[int, str], ...] = ()


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def rhs1_dense_shortcut_setup_from_env(
    *,
    has_pas: bool,
    include_phi1: bool,
    constraint_scheme: int,
    active_size: int,
    dense_fallback_max: int,
    dense_backend_allowed: bool,
    host_dense_fallback_allowed: bool,
    dense_krylov_allowed: bool,
    backend: str,
) -> RHS1DenseShortcutSetup:
    """Resolve dense shortcut/fallback controls with legacy PAS/backend guards."""

    dense_shortcut_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_DENSE_SHORTCUT_RATIO",
        1.0e6,
    )
    disable_dense_pas = (
        bool(has_pas) and (not bool(include_phi1)) and int(constraint_scheme) != 0
    )
    pas_dense_allow_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_DENSE_ALLOW_MAX", 4000)
    if disable_dense_pas and int(active_size) <= max(0, int(pas_dense_allow_max)):
        disable_dense_pas = False
    if disable_dense_pas or bool(has_pas):
        dense_shortcut_ratio = 0.0

    dense_fallback_max_use = int(dense_fallback_max)
    if disable_dense_pas:
        dense_fallback_max_use = 0

    messages: list[tuple[int, str]] = []
    if not bool(dense_backend_allowed):
        dense_shortcut_ratio = 0.0
        if not bool(host_dense_fallback_allowed) and not bool(dense_krylov_allowed):
            dense_fallback_max_use = 0
        dense_note = "dense shortcut/fallback"
        if bool(host_dense_fallback_allowed):
            dense_note = "dense shortcut (host dense fallback kept)"
        elif bool(dense_krylov_allowed):
            dense_note = "dense shortcut disabled (dense Krylov fallback kept)"
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: disabling RHSMode=1 "
                f"{dense_note} on backend={backend}",
            )
        )

    return RHS1DenseShortcutSetup(
        dense_shortcut_ratio=float(dense_shortcut_ratio),
        dense_fallback_max=int(dense_fallback_max_use),
        disable_dense_pas=bool(disable_dense_pas),
        messages=tuple(messages),
    )


def rhs1_dense_fallback_thresholds_from_env(
    *,
    dense_fallback_max: int,
    residual_ratio: float,
    allow_huge_limit: bool = True,
) -> RHS1DenseFallbackThresholds:
    """Resolve dense-fallback residual-ratio gates with legacy defaults."""

    fallback_max = int(dense_fallback_max)
    dense_fallback_ratio = 1.0e2
    dense_fallback_max_huge = 0
    if fallback_max > 0:
        if bool(allow_huge_limit):
            dense_fallback_max_huge = _env_int(
                "SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX_HUGE",
                fallback_max,
            )
        else:
            dense_fallback_max_huge = fallback_max
        dense_fallback_ratio = _env_float(
            "SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_RATIO",
            1.0e2,
        )
    trigger = (
        bool(float(residual_ratio) > float(dense_fallback_ratio))
        if float(dense_fallback_ratio) > 0.0
        else True
    )
    limit = dense_fallback_max_huge if trigger and bool(allow_huge_limit) else fallback_max
    return RHS1DenseFallbackThresholds(
        dense_fallback_max_huge=int(dense_fallback_max_huge),
        dense_fallback_ratio=float(dense_fallback_ratio),
        dense_fallback_limit=int(limit),
        dense_fallback_trigger=bool(trigger),
    )


def rhs1_early_dense_shortcut_decision(
    *,
    early_dense_shortcut: bool,
    cs0_sparse_first: bool,
    cs0_dense_fallback_allowed: bool,
    constraint_scheme: int,
    dense_shortcut_ratio: float,
    residual_ratio: float,
    sparse_prefer_over_dense_shortcut: bool,
    dense_fallback_max: int,
    active_size: int,
) -> RHS1EarlyDenseShortcutDecision:
    """Resolve the cheap early dense-shortcut gate from residual-ratio scalars."""

    messages: list[tuple[int, str]] = []
    shortcut = bool(early_dense_shortcut)
    if not (
        (not shortcut)
        and (not bool(cs0_sparse_first))
        and (bool(cs0_dense_fallback_allowed) or int(constraint_scheme) != 0)
        and float(dense_shortcut_ratio) > 0.0
        and float(residual_ratio) >= float(dense_shortcut_ratio)
        and (not bool(sparse_prefer_over_dense_shortcut))
    ):
        return RHS1EarlyDenseShortcutDecision(
            early_dense_shortcut=shortcut,
            messages=(),
        )

    thresholds = rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=int(dense_fallback_max),
        residual_ratio=float(residual_ratio),
    )
    limit = int(thresholds.dense_fallback_limit)
    if limit > 0 and int(active_size) <= int(limit):
        shortcut = True
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: dense fallback shortcut (early) "
                f"(ratio={float(residual_ratio):.3e} >= {float(dense_shortcut_ratio):.1e})",
            )
        )
    else:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: dense fallback shortcut skipped "
                f"(size={int(active_size)} > dense_max={int(limit)})",
            )
        )

    return RHS1EarlyDenseShortcutDecision(
        early_dense_shortcut=bool(shortcut),
        messages=tuple(messages),
    )


def rhs1_post_krylov_dense_shortcut_decision(
    *,
    dense_shortcut: bool,
    dense_shortcut_ratio: float,
    residual_norm_true: float,
    residual_ratio: float,
    target: float,
    dense_fallback_max: int,
    active_size: int,
    constraint_scheme: int,
    cs0_sparse_first: bool,
    sparse_prefer_over_dense_shortcut: bool,
    sparse_exact_direct: bool,
) -> RHS1PostKrylovDenseShortcutDecision:
    """Resolve late dense-shortcut admission before sparse rescue setup."""

    shortcut = bool(dense_shortcut)
    if shortcut or float(dense_shortcut_ratio) <= 0.0:
        return RHS1PostKrylovDenseShortcutDecision(dense_shortcut=shortcut)

    thresholds = rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=int(dense_fallback_max),
        residual_ratio=float(residual_ratio),
    )
    limit = int(thresholds.dense_fallback_limit)
    force_dense_cs0 = bool(int(constraint_scheme) == 0 and not bool(cs0_sparse_first))
    if force_dense_cs0:
        limit = max(limit, int(dense_fallback_max))

    admitted = (
        limit > 0
        and int(active_size) <= int(limit)
        and bool(thresholds.dense_fallback_trigger)
        and (float(residual_norm_true) > float(target) or force_dense_cs0)
        and float(residual_ratio) >= float(dense_shortcut_ratio)
    )
    if not admitted:
        return RHS1PostKrylovDenseShortcutDecision(dense_shortcut=False)

    if bool(sparse_prefer_over_dense_shortcut) and not bool(sparse_exact_direct):
        return RHS1PostKrylovDenseShortcutDecision(
            dense_shortcut=False,
            messages=(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: dense shortcut skipped "
                    "(preferring sparse rescue over dense shortcut)",
                ),
            ),
        )

    return RHS1PostKrylovDenseShortcutDecision(
        dense_shortcut=True,
        messages=(
            (
                0,
                "solve_v3_full_system_linear_gmres: dense fallback shortcut "
                f"(ratio={float(residual_ratio):.3e} >= {float(dense_shortcut_ratio):.1e})",
            ),
        ),
    )


def rhs1_evaluate_post_krylov_dense_shortcut(
    context: RHS1PostKrylovDenseShortcutEvaluationContext,
) -> RHS1PostKrylovDenseShortcutEvaluation:
    """Evaluate the late dense shortcut and compute true residual only if needed."""

    dense_shortcut = bool(context.dense_shortcut)
    if dense_shortcut or float(context.dense_shortcut_ratio) <= 0.0:
        return RHS1PostKrylovDenseShortcutEvaluation(
            dense_shortcut=dense_shortcut
        )

    quick_ratio = float(context.current_result.residual_norm) / max(
        float(context.target),
        1.0e-300,
    )
    if quick_ratio < float(context.dense_shortcut_ratio):
        return RHS1PostKrylovDenseShortcutEvaluation(dense_shortcut=False)

    residual_norm_true = true_residual_norm_or_inf(
        rhs=context.rhs,
        matvec=context.matvec,
        x=context.current_result.x,
    )
    residual_ratio = float(residual_norm_true) / max(float(context.target), 1.0e-300)
    decision = rhs1_post_krylov_dense_shortcut_decision(
        dense_shortcut=False,
        dense_shortcut_ratio=float(context.dense_shortcut_ratio),
        residual_norm_true=float(residual_norm_true),
        residual_ratio=float(residual_ratio),
        target=float(context.target),
        dense_fallback_max=int(context.dense_fallback_max),
        active_size=int(context.active_size),
        constraint_scheme=int(context.constraint_scheme),
        cs0_sparse_first=bool(context.cs0_sparse_first),
        sparse_prefer_over_dense_shortcut=bool(
            context.sparse_prefer_over_dense_shortcut
        ),
        sparse_exact_direct=bool(context.sparse_exact_direct),
    )
    return RHS1PostKrylovDenseShortcutEvaluation(
        dense_shortcut=bool(decision.dense_shortcut),
        residual_norm_true=float(residual_norm_true),
        residual_ratio=float(residual_ratio),
        messages=decision.messages,
    )


def resolve_rhs1_reduced_dense_fallback_admission(
    *,
    dense_fallback_max: int,
    residual_norm_true: float,
    reported_residual_norm: float,
    target: float,
    active_size: int,
    rhs_mode: int,
    include_phi1: bool,
    constraint_scheme: int,
    has_fp: bool,
    disable_dense_pas: bool,
    any_dense_path_allowed: bool,
    host_sparse_direct_used: bool,
    backend: str,
    host_sparse_skip_ratio: float,
    cs0_dense_fallback_allowed: bool,
    cs0_sparse_first: bool,
    cs0_petsc_compat: bool,
) -> RHS1DenseFallbackAdmission:
    """Resolve reduced active-DOF dense-fallback admission from policy scalars."""

    max_use = int(dense_fallback_max) if bool(any_dense_path_allowed) else 0
    residual_ratio = float(residual_norm_true) / max(float(target), 1e-300)
    thresholds = rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=max_use,
        residual_ratio=residual_ratio,
    )
    limit = int(thresholds.dense_fallback_limit)
    trigger = bool(thresholds.dense_fallback_trigger)
    messages: list[tuple[int, str]] = []

    if bool(host_sparse_direct_used) and str(backend) != "cpu":
        skip_ratio = float(host_sparse_skip_ratio)
        if skip_ratio > 0.0 and residual_ratio <= skip_ratio:
            trigger = False
            max_use = 0
            limit = 0
            messages.append(
                (
                    0,
                    "solve_v3_full_system_linear_gmres: skipping dense fallback after host sparse LU "
                    f"(ratio={residual_ratio:.3e} <= {skip_ratio:.1e})",
                )
            )

    pas_force_dense = (
        (not bool(disable_dense_pas))
        and (not bool(has_fp))
        and int(constraint_scheme) == 2
        and limit > 0
        and int(active_size) <= limit
        and float(reported_residual_norm) > float(target)
    )
    if pas_force_dense:
        trigger = True

    fp_force_dense = (
        bool(has_fp)
        and max_use > 0
        and int(active_size) <= max_use
        and float(residual_norm_true) > float(target)
    )
    if fp_force_dense:
        trigger = True
        limit = max(limit, max_use)

    force_dense_cs0 = bool(
        int(constraint_scheme) == 0
        and bool(cs0_dense_fallback_allowed)
        and (not bool(cs0_sparse_first))
        and (not bool(cs0_petsc_compat))
    )
    if force_dense_cs0:
        limit = max(limit, max_use)
        trigger = True

    if int(constraint_scheme) == 0 and not bool(cs0_dense_fallback_allowed):
        limit = 0
        trigger = False

    should_run = (
        limit > 0
        and int(rhs_mode) == 1
        and not bool(include_phi1)
        and int(active_size) <= limit
        and bool(trigger)
        and (float(residual_norm_true) > float(target) or force_dense_cs0)
    )
    return RHS1DenseFallbackAdmission(
        should_run=bool(should_run),
        dense_fallback_max=int(max_use),
        dense_fallback_limit=int(limit),
        dense_fallback_trigger=bool(trigger),
        force_dense_cs0=bool(force_dense_cs0),
        messages=tuple(messages),
    )


def resolve_rhs1_full_dense_fallback_admission(
    *,
    dense_fallback_max: int,
    residual_norm_true: float,
    target: float,
    active_size: int,
    total_size: int,
    rhs_mode: int,
    include_phi1: bool,
    constraint_scheme: int,
    has_fp: bool,
    any_dense_path_allowed: bool,
    host_sparse_direct_used: bool,
    backend: str,
    host_sparse_skip_ratio: float,
    cs0_sparse_first: bool,
) -> RHS1DenseFallbackAdmission:
    """Resolve full-system dense-fallback admission from policy scalars."""

    max_use = int(dense_fallback_max) if bool(any_dense_path_allowed) else 0
    residual_ratio = float(residual_norm_true) / max(float(target), 1e-300)
    thresholds = rhs1_dense_fallback_thresholds_from_env(
        dense_fallback_max=max_use,
        residual_ratio=residual_ratio,
        allow_huge_limit=False,
    )
    trigger = bool(thresholds.dense_fallback_trigger)
    messages: list[tuple[int, str]] = []

    if bool(host_sparse_direct_used) and str(backend) != "cpu":
        skip_ratio = float(host_sparse_skip_ratio)
        if skip_ratio > 0.0 and residual_ratio <= skip_ratio:
            trigger = False
            max_use = 0
            messages.append(
                (
                    0,
                    "solve_v3_full_system_linear_gmres: skipping dense fallback after host sparse LU "
                    f"(ratio={residual_ratio:.3e} <= {skip_ratio:.1e})",
                )
            )

    if (
        bool(has_fp)
        and max_use > 0
        and int(active_size) <= max_use
        and float(residual_norm_true) > float(target)
    ):
        trigger = True

    force_dense_cs0 = bool(int(constraint_scheme) == 0 and not bool(cs0_sparse_first))
    if force_dense_cs0:
        trigger = True

    should_run = (
        max_use > 0
        and int(rhs_mode) == 1
        and not bool(include_phi1)
        and int(total_size) <= max_use
        and bool(trigger)
        and float(residual_norm_true) > float(target)
    )
    return RHS1DenseFallbackAdmission(
        should_run=bool(should_run),
        dense_fallback_max=int(max_use),
        dense_fallback_limit=int(max_use),
        dense_fallback_trigger=bool(trigger),
        force_dense_cs0=bool(force_dense_cs0),
        messages=tuple(messages),
    )


_RHS1_FP_PROBE_HEAVY_PRECONDITIONERS = frozenset(
    {
        "point",
        "theta_line",
        "theta_schwarz",
        "zeta_line",
        "zeta_schwarz",
        "theta_zeta",
        "adi",
        "xblock_tz",
        "sxblock_tz",
        "species_block",
        "schur",
        "pas_hybrid",
    }
)


def rhs1_fp_preconditioner_probe_kind_from_env(
    *,
    rhs1_precond_kind: str | None,
    rhs1_precond_env: str,
    has_fp: bool,
    use_dkes: bool,
    include_phi1: bool,
    dense_fallback_max: int,
    active_size: int,
    rhs1_precond_enabled: bool,
    solve_method_kind: str,
) -> str | None:
    """Downgrade heavy FP preconditioners to collision for dense-probe setup."""

    fp_probe_env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE", "").strip().lower()
    )
    fp_probe_enabled = fp_probe_env not in {"0", "false", "no", "off"}
    if bool(has_fp) and (not bool(use_dkes)):
        fp_probe_enabled = False
    fp_probe_min = _env_int("SFINCS_JAX_RHSMODE1_FP_PRECOND_PROBE_MIN", 2500)
    if (
        fp_probe_enabled
        and (not rhs1_precond_env)
        and bool(has_fp)
        and (not bool(include_phi1))
        and int(dense_fallback_max) > 0
        and int(active_size) >= int(fp_probe_min)
        and int(active_size) <= int(dense_fallback_max)
        and bool(rhs1_precond_enabled)
        and str(solve_method_kind) not in {"dense", "dense_ksp"}
        and rhs1_precond_kind in _RHS1_FP_PROBE_HEAVY_PRECONDITIONERS
    ):
        return "collision"
    return rhs1_precond_kind


def rhs1_dense_probe_enabled_from_env() -> bool:
    """Return whether the reduced dense probe is globally enabled."""

    probe_env = os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PROBE", "").strip().lower()
    return probe_env not in {"0", "false", "no", "off"}


def rhs1_dense_probe_admission(
    *,
    probe_enabled: bool,
    probe_shortcut: bool,
    cs0_petsc_compat: bool,
    cs0_sparse_first: bool,
    cs0_dense_fallback_allowed: bool,
    constraint_scheme: int,
    has_preconditioner: bool,
    solve_method_kind: str,
) -> RHS1DenseProbeAdmission:
    """Apply cheap guards before evaluating a reduced dense fallback probe."""

    enabled = (
        bool(probe_enabled)
        and (not bool(probe_shortcut))
        and (not bool(cs0_petsc_compat))
        and (not bool(cs0_sparse_first))
        and (bool(cs0_dense_fallback_allowed) or int(constraint_scheme) != 0)
        and bool(has_preconditioner)
        and str(solve_method_kind) not in {"dense", "dense_ksp"}
    )
    return RHS1DenseProbeAdmission(enabled=bool(enabled))


def rhs1_dense_probe_shortcut_decision(
    *,
    dense_shortcut_ratio: float,
    probe_ratio: float,
    dense_fallback_max: int,
    active_size: int,
    sparse_prefer_over_dense_shortcut: bool,
) -> RHS1DenseProbeShortcutDecision:
    """Resolve whether a dense probe should become an early dense shortcut."""

    if float(dense_shortcut_ratio) <= 0.0 or float(probe_ratio) < float(
        dense_shortcut_ratio
    ):
        return RHS1DenseProbeShortcutDecision(
            accept_shortcut=False,
            seed_x0_if_missing=True,
        )

    allow_probe_shortcut = int(dense_fallback_max) > 0 and int(active_size) <= int(
        dense_fallback_max
    )
    if allow_probe_shortcut and (not bool(sparse_prefer_over_dense_shortcut)):
        return RHS1DenseProbeShortcutDecision(
            accept_shortcut=True,
            seed_x0_if_missing=False,
            messages=(
                (
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback shortcut (probe) "
                    f"(ratio={float(probe_ratio):.3e} >= {float(dense_shortcut_ratio):.1e})",
                ),
            ),
        )

    if bool(sparse_prefer_over_dense_shortcut) and allow_probe_shortcut:
        message = (
            "solve_v3_full_system_linear_gmres: probe shortcut skipped "
            "(preferring sparse rescue over dense shortcut)"
        )
    else:
        message = (
            "solve_v3_full_system_linear_gmres: probe shortcut skipped "
            f"(size={int(active_size)} > dense_max={int(dense_fallback_max)})"
        )
    return RHS1DenseProbeShortcutDecision(
        accept_shortcut=False,
        seed_x0_if_missing=True,
        messages=((1, message),),
    )


def run_rhs1_dense_probe_stage(
    *,
    context: RHS1DenseProbeStageContext,
    replay_state,
    record_replay_problem: Callable[..., None],
    solver_kind: Callable[[str], tuple[str, str]],
    emit: Callable[[int, str], None] | None = None,
) -> RHS1DenseProbeStageResult:
    """Run the reduced dense-probe shortcut/seed stage with replay handoff."""

    result = context.current_result
    x0_reduced = context.x0_reduced
    probe_shortcut = bool(context.probe_shortcut)
    early_dense_shortcut = False
    admission = rhs1_dense_probe_admission(
        probe_enabled=rhs1_dense_probe_enabled_from_env(),
        probe_shortcut=probe_shortcut,
        cs0_petsc_compat=bool(context.cs0_petsc_compat),
        cs0_sparse_first=bool(context.cs0_sparse_first),
        cs0_dense_fallback_allowed=bool(context.cs0_dense_fallback_allowed),
        constraint_scheme=int(context.constraint_scheme),
        has_preconditioner=context.preconditioner is not None,
        solve_method_kind=str(context.solve_method_kind),
    )
    if not bool(admission.enabled):
        return RHS1DenseProbeStageResult(
            result=result,
            x0_reduced=x0_reduced,
            early_dense_shortcut=False,
            probe_shortcut=probe_shortcut,
        )

    try:
        if context.preconditioner is None:
            return RHS1DenseProbeStageResult(
                result=result,
                x0_reduced=x0_reduced,
                early_dense_shortcut=False,
                probe_shortcut=probe_shortcut,
            )
        probe_x0 = context.preconditioner(context.rhs)
        probe_r = context.rhs - context.matvec(probe_x0)
        probe_norm = float(jnp.linalg.norm(probe_r))
        probe_ratio = probe_norm / max(float(context.target), 1e-300)
        decision = rhs1_dense_probe_shortcut_decision(
            dense_shortcut_ratio=float(context.dense_shortcut_ratio),
            probe_ratio=float(probe_ratio),
            dense_fallback_max=int(context.dense_fallback_max),
            active_size=int(context.active_size),
            sparse_prefer_over_dense_shortcut=bool(
                context.sparse_prefer_over_dense_shortcut
            ),
        )
        if bool(decision.accept_shortcut):
            early_dense_shortcut = True
            probe_shortcut = True
            result = GMRESSolveResult(
                x=probe_x0,
                residual_norm=jnp.asarray(probe_norm),
            )
            record_replay_problem(
                replay_state,
                matvec_fn=context.matvec,
                b_vec=context.rhs,
                precond_fn=context.preconditioner,
                x0_vec=probe_x0,
                precond_side=str(context.gmres_precond_side),
                solver_kind=solver_kind(str(context.solve_method))[0],
            )
        elif bool(decision.seed_x0_if_missing) and x0_reduced is None:
            x0_reduced = probe_x0
        if emit is not None:
            for level, message in decision.messages:
                emit(level, message)
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: probe failed "
                f"({type(exc).__name__}: {exc})",
            )

    return RHS1DenseProbeStageResult(
        result=result,
        x0_reduced=x0_reduced,
        early_dense_shortcut=bool(early_dense_shortcut),
        probe_shortcut=bool(probe_shortcut),
    )


def solve_host_dense_reduced(
    *,
    context: HostDenseReducedSolveContext,
    x0: jnp.ndarray | None = None,
) -> GMRESSolveResult:
    """Solve the reduced system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    use_row_scaled = bool(int(context.constraint_scheme) == 0 or (int(context.constraint_scheme) == 1 and context.has_fp))
    if context.dense_matrix_cache is not None:
        a_np = np.asarray(context.dense_matrix_cache, dtype=np.float64)
    else:
        a_dense_jnp = assemble_dense_matrix_from_matvec(
            matvec=context.matvec,
            n=int(context.active_size),
            dtype=context.rhs.dtype,
        )
        a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)

    matvec_residual = context.matvec
    b_dense = jnp.asarray(context.rhs, dtype=jnp.float64)
    if use_row_scaled:
        diag_floor = 1e-12
        diag = np.diag(a_np).astype(np.float64, copy=False)
        diag_abs = np.abs(diag)
        diag_safe = np.where(diag_abs > diag_floor, diag, np.sign(diag) * diag_floor)
        diag_safe = np.where(diag_safe != 0.0, diag_safe, diag_floor)
        scale = (1.0 / diag_safe).astype(np.float64, copy=False)
        a_np = a_np * scale[:, None]
        scale_jnp = jnp.asarray(scale, dtype=jnp.float64)
        b_dense = b_dense * scale_jnp

        def matvec_residual(x_vec: jnp.ndarray) -> jnp.ndarray:
            return scale_jnp * context.matvec(x_vec)

    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(b_dense, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(b_dense, dtype=np.float64)), dtype=np.float64)
        if x0 is not None and x0.shape == context.rhs.shape:
            x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)

    r_dense = b_dense - matvec_residual(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(r_dense))


def solve_host_dense_full(
    *,
    context: HostDenseFullSolveContext,
    x0: jnp.ndarray | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray]:
    """Solve the full system on the host using LU or least squares."""

    import scipy.linalg as sla  # noqa: PLC0415

    a_dense_jnp = assemble_dense_matrix_from_matvec(
        matvec=context.matvec,
        n=int(context.total_size),
        dtype=context.rhs.dtype,
    )
    a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)
    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        x_np = np.asarray(
            np.linalg.lstsq(a_np, np.asarray(context.rhs, dtype=np.float64), rcond=None)[0],
            dtype=np.float64,
        )
    else:
        lu, piv = sla.lu_factor(a_np)
        x_np = np.asarray(sla.lu_solve((lu, piv), np.asarray(context.rhs, dtype=np.float64)), dtype=np.float64)
    if x0 is not None and x0.shape == context.rhs.shape:
        x_np = x_np + 0.0 * np.asarray(x0, dtype=np.float64)
    x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    residual_vec = context.rhs - context.matvec(x_dense)
    return GMRESSolveResult(x=x_dense, residual_norm=jnp.linalg.norm(residual_vec)), residual_vec


def run_rhs1_reduced_host_dense_shortcut_stage(
    *,
    context: RHS1ReducedHostDenseShortcutContext,
    replay_state,
    record_replay_problem: Callable[..., None],
    solver_kind: Callable[[str], tuple[str, str]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
) -> RHS1ReducedHostDenseShortcutResult:
    """Run the reduced host dense shortcut and record replay metadata."""

    if not bool(context.enabled):
        if context.current_result is None:
            raise ValueError("disabled reduced host dense shortcut needs current_result")
        return RHS1ReducedHostDenseShortcutResult(
            result=context.current_result,
            early_dense_shortcut=bool(context.early_dense_shortcut),
            probe_shortcut=bool(context.probe_shortcut),
        )

    if mark is not None:
        mark("rhs1_host_dense_shortcut_start")
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: accelerator FP small system -> "
            f"using host dense shortcut (size={int(context.active_size)})",
        )
    result = solve_host_dense_reduced(
        context=context.solve_context,
        x0=context.x0,
    )
    if mark is not None:
        mark("rhs1_host_dense_shortcut_done")
    record_replay_problem(
        replay_state,
        matvec_fn=context.solve_context.matvec,
        b_vec=context.solve_context.rhs,
        precond_fn=None,
        x0_vec=context.x0,
        precond_side="none",
        solver_kind=solver_kind("incremental")[0],
    )
    return RHS1ReducedHostDenseShortcutResult(
        result=result,
        early_dense_shortcut=True,
        probe_shortcut=True,
    )


def run_rhs1_full_host_dense_shortcut_stage(
    *,
    context: RHS1FullHostDenseShortcutContext,
    replay_state,
    record_replay_problem: Callable[..., None],
    solver_kind: Callable[[str], tuple[str, str]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
) -> RHS1FullHostDenseShortcutResult:
    """Run the full-system host dense shortcut and record replay metadata."""

    if not bool(context.enabled):
        if context.current_result is None:
            raise ValueError("disabled full host dense shortcut needs current_result")
        return RHS1FullHostDenseShortcutResult(
            result=context.current_result,
            residual_vec=context.current_residual_vec,
        )

    if mark is not None:
        mark("rhs1_host_dense_shortcut_start")
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: accelerator FP small system -> "
            f"using host dense shortcut (size={int(context.total_size)})",
        )
    result, residual_vec = solve_host_dense_full(
        context=context.solve_context,
        x0=context.x0,
    )
    if mark is not None:
        mark("rhs1_host_dense_shortcut_done")
    record_replay_problem(
        replay_state,
        matvec_fn=context.solve_context.matvec,
        b_vec=context.solve_context.rhs,
        precond_fn=None,
        x0_vec=context.x0,
        precond_side="none",
        solver_kind=solver_kind("incremental")[0],
    )
    return RHS1FullHostDenseShortcutResult(
        result=result,
        residual_vec=residual_vec,
    )


def solve_rhs1_reduced_dense_fallback_candidate(
    *,
    context: RHS1ReducedDenseFallbackCandidateContext,
    emit: Callable[[int, str], None] | None = None,
) -> tuple[GMRESSolveResult, float]:
    """Run one dense fallback candidate for a reduced RHSMode=1 system.

    The caller remains responsible for residual/runtime/memory admission. This
    keeps the policy gate in the driver while moving the dense solve mechanics
    out of the monolithic solve function.
    """

    started = time.perf_counter()
    use_row_scaled = bool(
        int(context.constraint_scheme) == 0
        or (int(context.constraint_scheme) == 1 and bool(context.has_fp))
    )
    host_dense_env = os.environ.get(
        "SFINCS_JAX_RHSMODE1_DENSE_HOST_LU", ""
    ).strip().lower()
    backend = context.backend or jax.default_backend()
    if host_dense_env in {"0", "false", "no", "off"}:
        use_host_dense = False
    elif host_dense_env in {"1", "true", "yes", "on"}:
        use_host_dense = True
    else:
        # Default: avoid backend LAPACK/SVD paths on accelerators, and avoid
        # XLA dense-solve scratch allocations for medium/large CPU systems.
        use_host_dense = backend != "cpu" or (
            bool(context.use_implicit) and int(context.active_size) >= 2000
        )
    if bool(context.has_pas) and int(context.active_size) <= 2000:
        use_host_dense = True

    if use_host_dense:
        result = _solve_rhs1_reduced_dense_fallback_host_candidate(
            context=context,
            backend=backend,
            host_dense_env=host_dense_env,
            use_row_scaled=use_row_scaled,
            emit=emit,
        )
    elif context.dense_backend_allowed and context.dense_matrix_cache is not None:
        a_dense_jnp = jnp.asarray(context.dense_matrix_cache, dtype=context.rhs.dtype)
        if use_row_scaled:
            x_dense, _rn = dense_solve_from_matrix_row_scaled(
                a=a_dense_jnp,
                b=context.rhs,
            )
        else:
            x_dense, _rn = dense_solve_from_matrix(a=a_dense_jnp, b=context.rhs)
        result, _residual = result_with_true_residual(
            x=x_dense,
            rhs=context.rhs,
            matvec=context.matvec,
        )
    else:
        if context.dense_matrix_cache is not None:
            a_dense_jnp = jnp.asarray(context.dense_matrix_cache, dtype=context.rhs.dtype)
        else:
            a_dense_jnp = assemble_dense_matrix_from_matvec(
                matvec=context.matvec,
                n=int(context.active_size),
                dtype=context.rhs.dtype,
            )
        if emit is not None and jax.default_backend() != "cpu":
            emit(
                0,
                "solve_v3_full_system_linear_gmres: dense fallback using explicit dense Krylov "
                f"on backend={jax.default_backend()}",
            )
        result, _residual = dense_krylov_solve_from_matrix_with_residual(
            a=a_dense_jnp,
            b=context.rhs,
            x0=context.x0,
            preconditioner=None,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            solve_method="incremental",
            precondition_side=(
                "none" if use_row_scaled else str(context.gmres_precond_side)
            ),
            row_scaled=use_row_scaled,
        )

    return result, time.perf_counter() - started


def run_rhs1_reduced_dense_fallback_stage(
    *,
    context: RHS1ReducedDenseFallbackStageContext,
    replay_state,
    accept_candidate: Callable[..., tuple[GMRESSolveResult, jnp.ndarray | None, bool]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
    peak_rss_mb: Callable[[], float] | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray | None, bool]:
    """Run a reduced dense fallback candidate and measured replay handoff."""

    candidate_context = context.candidate_context
    if mark is not None:
        mark("rhs1_dense_fallback_start")
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: dense fallback "
            f"(size={int(candidate_context.active_size)} "
            f"residual={float(context.current_result.residual_norm):.3e} "
            f"> target={float(context.target):.3e})",
        )

    accepted = False
    result = context.current_result
    residual_vec = context.current_residual_vec
    try:
        res_dense, elapsed_s = solve_rhs1_reduced_dense_fallback_candidate(
            context=candidate_context,
            emit=emit,
        )
        result, residual_vec, accepted = accept_candidate(
            replay_state=replay_state,
            current_result=context.current_result,
            candidate_result=res_dense,
            current_residual_vec=context.current_residual_vec,
            candidate_residual_vec=None,
            matvec_fn=candidate_context.matvec,
            b_vec=candidate_context.rhs,
            precond_fn=None,
            x0_vec=res_dense.x,
            restart=int(candidate_context.restart),
            maxiter=candidate_context.maxiter,
            precond_side="none",
            solver_kind="dense",
            candidate_name="dense_reduced",
            baseline_name="current_reduced",
            target_value=float(context.target),
            solve_s=float(elapsed_s),
            peak_rss_mb=peak_rss_mb() if peak_rss_mb is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: dense fallback failed "
                f"({type(exc).__name__}: {exc})",
            )
    finally:
        if mark is not None:
            mark("rhs1_dense_fallback_done")
    return result, residual_vec, bool(accepted)


def run_rhs1_reduced_dense_fallback_admission_stage(
    *,
    context: RHS1ReducedDenseFallbackAdmissionStageContext,
    replay_state,
    accept_candidate: Callable[..., tuple[GMRESSolveResult, jnp.ndarray | None, bool]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
    peak_rss_mb: Callable[[], float] | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray | None, bool]:
    """Resolve admission and run reduced dense fallback if policy allows it."""

    stage_context = context.stage_context
    admission = resolve_rhs1_reduced_dense_fallback_admission(
        dense_fallback_max=int(context.dense_fallback_max),
        residual_norm_true=float(context.residual_norm_true),
        reported_residual_norm=float(context.reported_residual_norm),
        target=float(stage_context.target),
        active_size=int(context.active_size),
        rhs_mode=int(context.rhs_mode),
        include_phi1=bool(context.include_phi1),
        constraint_scheme=int(stage_context.candidate_context.constraint_scheme),
        has_fp=bool(context.has_fp),
        disable_dense_pas=bool(context.disable_dense_pas),
        any_dense_path_allowed=bool(context.any_dense_path_allowed),
        host_sparse_direct_used=bool(context.host_sparse_direct_used),
        backend=str(context.backend),
        host_sparse_skip_ratio=float(context.host_sparse_skip_ratio),
        cs0_dense_fallback_allowed=bool(context.cs0_dense_fallback_allowed),
        cs0_sparse_first=bool(context.cs0_sparse_first),
        cs0_petsc_compat=bool(context.cs0_petsc_compat),
    )
    if emit is not None:
        for level, message in admission.messages:
            emit(level, message)
    if not bool(admission.should_run):
        return (
            stage_context.current_result,
            stage_context.current_residual_vec,
            False,
        )

    return run_rhs1_reduced_dense_fallback_stage(
        context=stage_context,
        replay_state=replay_state,
        accept_candidate=accept_candidate,
        emit=emit,
        mark=mark,
        peak_rss_mb=peak_rss_mb,
    )


def run_rhs1_full_dense_fallback_candidate(
    *,
    context: RHS1FullDenseFallbackContext,
    replay_state,
    accept_candidate: Callable[..., tuple[GMRESSolveResult, jnp.ndarray | None, bool]],
    solve_linear_with_residual: Callable[..., tuple[GMRESSolveResult, jnp.ndarray]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
    peak_rss_mb: Callable[[], float] | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray | None, bool]:
    """Run the final full-system dense fallback and measured acceptance handoff."""

    if mark is not None:
        mark("rhs1_dense_fallback_start")
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: dense fallback "
            f"(size={int(context.total_size)} "
            f"residual={float(context.residual_norm_check):.3e} "
            f"> target={float(context.target):.3e})",
        )

    accepted = False
    result = context.current_result
    residual_vec = context.current_residual_vec
    try:
        started = time.perf_counter()
        use_row_scaled = int(context.constraint_scheme) == 0
        if context.dense_backend_allowed:
            dense_method = "dense_row_scaled" if use_row_scaled else "dense"
            res_dense, residual_vec_dense = solve_linear_with_residual(
                matvec_fn=context.matvec,
                b_vec=context.rhs,
                precond_fn=None,
                x0_vec=None,
                tol_val=float(context.tol),
                atol_val=float(context.atol),
                restart_val=int(context.restart),
                maxiter_val=context.maxiter,
                solve_method_val=dense_method,
                precond_side="none",
            )
        else:
            backend = context.backend or jax.default_backend()
            if emit is not None and backend != "cpu":
                emit(
                    0,
                    "solve_v3_full_system_linear_gmres: dense fallback using explicit dense Krylov "
                    f"on backend={backend}",
                )
            if context.dense_matrix_cache is not None:
                a_dense = jnp.asarray(context.dense_matrix_cache, dtype=context.rhs.dtype)
            else:
                a_dense = assemble_dense_matrix_from_matvec(
                    matvec=context.matvec,
                    n=int(context.total_size),
                    dtype=context.rhs.dtype,
                )
            res_dense, residual_vec_dense = dense_krylov_solve_from_matrix_with_residual(
                a=a_dense,
                b=context.rhs,
                x0=context.current_result.x,
                preconditioner=None,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                solve_method="incremental",
                precondition_side="none",
                row_scaled=use_row_scaled,
            )
        elapsed_s = time.perf_counter() - started
        result, residual_vec, accepted = accept_candidate(
            replay_state=replay_state,
            current_result=context.current_result,
            candidate_result=res_dense,
            current_residual_vec=context.current_residual_vec,
            candidate_residual_vec=residual_vec_dense,
            matvec_fn=context.matvec,
            b_vec=context.rhs,
            precond_fn=None,
            x0_vec=res_dense.x,
            restart=int(context.restart),
            maxiter=context.maxiter,
            precond_side="none",
            solver_kind="dense",
            candidate_name="dense_full",
            baseline_name="current_full",
            target_value=float(context.target),
            solve_s=elapsed_s,
            peak_rss_mb=peak_rss_mb() if peak_rss_mb is not None else None,
        )
    except Exception as exc:  # noqa: BLE001
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: dense fallback failed "
                f"({type(exc).__name__}: {exc})",
            )
    finally:
        if mark is not None:
            mark("rhs1_dense_fallback_done")
    return result, residual_vec, bool(accepted)


def run_rhs1_full_dense_fallback_stage(
    *,
    context: RHS1FullDenseFallbackStageContext,
    replay_state,
    accept_candidate: Callable[..., tuple[GMRESSolveResult, jnp.ndarray | None, bool]],
    solve_linear_with_residual: Callable[..., tuple[GMRESSolveResult, jnp.ndarray]],
    emit: Callable[[int, str], None] | None = None,
    mark: Callable[[str], None] | None = None,
    peak_rss_mb: Callable[[], float] | None = None,
) -> tuple[GMRESSolveResult, jnp.ndarray | None, bool]:
    """Resolve admission and run the final full-system dense fallback if allowed."""

    candidate_context = context.candidate_context
    admission = resolve_rhs1_full_dense_fallback_admission(
        dense_fallback_max=int(context.dense_fallback_max),
        residual_norm_true=float(context.residual_norm_true),
        target=float(candidate_context.target),
        active_size=int(context.active_size),
        total_size=int(candidate_context.total_size),
        rhs_mode=int(context.rhs_mode),
        include_phi1=bool(context.include_phi1),
        constraint_scheme=int(candidate_context.constraint_scheme),
        has_fp=bool(context.has_fp),
        any_dense_path_allowed=bool(context.any_dense_path_allowed),
        host_sparse_direct_used=bool(context.host_sparse_direct_used),
        backend=candidate_context.backend or jax.default_backend(),
        host_sparse_skip_ratio=float(context.host_sparse_skip_ratio),
        cs0_sparse_first=bool(context.cs0_sparse_first),
    )
    if emit is not None:
        for level, message in admission.messages:
            emit(level, message)
    if not bool(admission.should_run):
        return (
            candidate_context.current_result,
            candidate_context.current_residual_vec,
            False,
        )

    return run_rhs1_full_dense_fallback_candidate(
        context=candidate_context,
        replay_state=replay_state,
        accept_candidate=accept_candidate,
        solve_linear_with_residual=solve_linear_with_residual,
        emit=emit,
        mark=mark,
        peak_rss_mb=peak_rss_mb,
    )


def _solve_rhs1_reduced_dense_fallback_host_candidate(
    *,
    context: RHS1ReducedDenseFallbackCandidateContext,
    backend: str,
    host_dense_env: str,
    use_row_scaled: bool,
    emit: Callable[[int, str], None] | None,
) -> GMRESSolveResult:
    """Host LU/least-squares branch for the reduced dense fallback."""

    import scipy.linalg as sla  # noqa: PLC0415

    if emit is not None and backend != "cpu" and host_dense_env in {"", "auto"}:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: dense fallback using host LU "
            f"on backend={backend}",
        )

    if context.dense_matrix_cache is not None:
        a_np = np.asarray(context.dense_matrix_cache, dtype=np.float64)
    else:
        a_dense_jnp = assemble_dense_matrix_from_matvec(
            matvec=context.matvec,
            n=int(context.active_size),
            dtype=context.rhs.dtype,
        )
        a_np = np.asarray(a_dense_jnp, dtype=np.float64)
    a_np = np.array(a_np, dtype=np.float64, copy=True)
    if a_np.ndim != 2:
        a_np = np.squeeze(a_np)

    mv_dense = context.matvec
    b_dense = jnp.asarray(context.rhs, dtype=jnp.float64)
    if use_row_scaled:
        diag_floor = 1e-12
        diag = np.diag(a_np).astype(np.float64, copy=False)
        diag_abs = np.abs(diag)
        diag_safe = np.where(
            diag_abs > diag_floor,
            diag,
            np.sign(diag) * diag_floor,
        )
        diag_safe = np.where(diag_safe != 0.0, diag_safe, diag_floor)
        scale = (1.0 / diag_safe).astype(np.float64, copy=False)
        a_np = a_np * scale[:, None]
        scale_jnp = jnp.asarray(scale, dtype=jnp.float64)
        b_dense = b_dense * scale_jnp

        def mv_dense(x: jnp.ndarray) -> jnp.ndarray:
            return scale_jnp * context.matvec(x)

    if a_np.ndim != 2 or a_np.shape[0] != a_np.shape[1]:
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: dense fallback "
                f"non-square matrix shape={a_np.shape}; using least-squares host solve",
            )
        if not context.use_implicit:
            x_np = np.asarray(
                np.linalg.lstsq(
                    a_np,
                    np.asarray(b_dense, dtype=np.float64),
                    rcond=None,
                )[0],
                dtype=np.float64,
            )
            x_dense = jnp.asarray(x_np, dtype=jnp.float64)
        else:

            def _solve_cb(rhs_np: np.ndarray) -> np.ndarray:
                rhs_np = np.asarray(rhs_np, dtype=np.float64)
                return np.asarray(
                    np.linalg.lstsq(a_np, rhs_np, rcond=None)[0],
                    dtype=np.float64,
                )

            out_spec = jax.ShapeDtypeStruct(b_dense.shape, jnp.float64)
            x_dense = jax.pure_callback(_solve_cb, out_spec, b_dense)
        result, _residual = result_with_true_residual(
            x=x_dense,
            rhs=context.rhs,
            matvec=context.matvec,
        )
        return result

    lu, piv = sla.lu_factor(a_np)
    refine_steps = 0
    if bool(context.has_pas) and int(context.active_size) <= 2000:
        refine_steps = 2
    if not context.use_implicit:
        rhs_np = np.asarray(b_dense, dtype=np.float64)
        x_np = np.asarray(sla.lu_solve((lu, piv), rhs_np), dtype=np.float64)
        for _ in range(int(refine_steps)):
            r_np = rhs_np - a_np @ x_np
            dx_np = np.asarray(sla.lu_solve((lu, piv), r_np), dtype=np.float64)
            x_np = x_np + dx_np
        x_dense = jnp.asarray(x_np, dtype=jnp.float64)
    else:
        out_spec = jax.ShapeDtypeStruct(b_dense.shape, jnp.float64)

        def _solve_cb(rhs_np: np.ndarray) -> np.ndarray:
            rhs_np = np.asarray(rhs_np, dtype=np.float64)
            x_np = np.asarray(sla.lu_solve((lu, piv), rhs_np), dtype=np.float64)
            for _ in range(int(refine_steps)):
                r_np = rhs_np - a_np @ x_np
                dx_np = np.asarray(sla.lu_solve((lu, piv), r_np), dtype=np.float64)
                x_np = x_np + dx_np
            return x_np

        def _solveT_cb(rhs_np: np.ndarray) -> np.ndarray:
            rhs_np = np.asarray(rhs_np, dtype=np.float64)
            x_np = np.asarray(
                sla.lu_solve((lu, piv), rhs_np, trans=1),
                dtype=np.float64,
            )
            for _ in range(int(refine_steps)):
                r_np = rhs_np - a_np.T @ x_np
                dx_np = np.asarray(
                    sla.lu_solve((lu, piv), r_np, trans=1),
                    dtype=np.float64,
                )
                x_np = x_np + dx_np
            return x_np

        def _solve_host(_mv, rhs: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(_solve_cb, out_spec, rhs)

        def _transpose_solve_host(_mv_t, rhs: jnp.ndarray) -> jnp.ndarray:
            return jax.pure_callback(_solveT_cb, out_spec, rhs)

        x_dense = jax.lax.custom_linear_solve(
            mv_dense,
            b_dense,
            solve=_solve_host,
            transpose_solve=_transpose_solve_host,
            symmetric=False,
        )
    result, _residual = result_with_true_residual(
        x=x_dense,
        rhs=context.rhs,
        matvec=context.matvec,
    )
    return result


# Host-only automatic RHSMode=1 solve routing.
_FALSE_TOKENS = {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


@dataclass(frozen=True)
class RHS1AutoHostSolveContext:
    """Inputs needed to try non-autodiff RHSMode=1 host solver shortcuts."""

    nml: Any
    which_rhs: int | None
    op: Any
    x0: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_method: str
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    recycle_basis: Sequence[Any] | None
    solve_driver: Callable[..., Any]
    solve_method_kind_requested: str
    structured_full_csr_explicit_requested: bool
    use_implicit: bool
    structured_auto_allowed: bool
    structured_sharded_multidevice: bool


@dataclass(frozen=True)
class RHS1StructuredCSRSolveContext:
    """Inputs for the explicit structured full-CSR host solve route."""

    nml: Any
    op: Any
    x0: Any
    rhs_norm: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    solve_method: str
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    structured_solver: Callable[..., Any]


@dataclass(frozen=True)
class RHS1SparseHostSafeSolveContext:
    """Inputs for the host sparse solve with constrained-PAS safe fallback."""

    nml: Any
    which_rhs: int | None
    op: Any
    x0: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    identity_shift: float
    phi1_hat_base: Any
    differentiable: bool | None
    emit: Callable[[int, str], None] | None
    recycle_basis: Sequence[Any] | None
    solve_driver: Callable[..., Any]
    solve_method_kind_explicit: str
    requested: bool


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _annotate_auto_result(result: Any, metadata_updates: dict[str, Any]) -> Any:
    metadata = dict(getattr(result, "metadata", None) or {})
    metadata.update(metadata_updates)
    return replace(result, metadata=metadata)


def _try_fortran_reduced_sparse_pc_auto(context: RHS1AutoHostSolveContext) -> Any | None:
    op = context.op
    auto_enabled = os.environ.get("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO", "").strip().lower()
    if auto_enabled in _FALSE_TOKENS:
        return None

    min_size = max(1, _env_int("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", 10_000))
    system_size = int(op.total_size)
    requested = bool(
        context.solve_method_kind_requested in {"auto", "default"}
        and not bool(context.use_implicit)
        and int(op.rhs_mode) == 1
        and not bool(op.include_phi1)
        and int(op.constraint_scheme) == 1
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and abs(float(context.identity_shift)) == 0.0
        and system_size >= min_size
    )
    if not requested:
        return None

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: auto selecting Fortran-reduced "
            "sparse-PC GMRES for large RHSMode=1 full-FP solve "
            f"(system_size={system_size} >= {min_size})",
        )
    result = context.solve_driver(
        nml=context.nml,
        which_rhs=context.which_rhs,
        op=op,
        x0=context.x0,
        tol=context.tol,
        atol=context.atol,
        restart=context.restart,
        maxiter=context.maxiter,
        solve_method="fortran_reduced_pc_gmres",
        identity_shift=context.identity_shift,
        phi1_hat_base=context.phi1_hat_base,
        differentiable=False,
        emit=context.emit,
        recycle_basis=context.recycle_basis,
    )
    return _annotate_auto_result(
        result,
        {
            "solve_method_requested": str(context.solve_method),
            "requested_solve_method": str(context.solve_method),
            "auto_solver_selected": True,
            "auto_solver_policy": "fortran_reduced_pc_gmres",
            "auto_solver_size": system_size,
            "auto_solver_min_size": min_size,
        },
    )


def _try_structured_full_csr_auto(context: RHS1AutoHostSolveContext) -> Any | None:
    if context.structured_full_csr_explicit_requested:
        return None
    requested = bool(
        context.solve_method_kind_requested in {"auto", "default"}
        and not context.structured_sharded_multidevice
        and context.structured_auto_allowed
    )
    if not requested:
        return None

    if context.emit is not None:
        context.emit(0, "solve_v3_full_system_linear_gmres: auto trying structured full CSR host solve")
    try:
        result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=context.op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="structured_full_csr",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=False,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
    except RuntimeError as exc:
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: auto structured full CSR skipped "
                f"({exc}); falling back to matrix-free policy",
            )
        return None

    metadata = dict(getattr(result, "metadata", None) or {})
    if bool(metadata.get("accepted_converged", False)):
        return _annotate_auto_result(
            result,
            {
                "solve_method_requested": str(context.solve_method),
                "requested_solve_method": str(context.solve_method),
                "auto_solver_selected": True,
                "auto_solver_policy": "structured_full_csr",
            },
        )
    if context.emit is not None:
        residual = metadata.get("reported_residual_norm", getattr(result.gmres, "residual_norm", 0.0))
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: auto structured full CSR did not converge "
            f"(residual={float(residual):.3e}); falling back to matrix-free policy",
        )
    return None


def try_rhs1_auto_host_solve(context: RHS1AutoHostSolveContext) -> Any | None:
    """Try early non-autodiff RHSMode=1 host solver routes in priority order."""

    result = _try_fortran_reduced_sparse_pc_auto(context)
    if result is not None:
        return result
    return _try_structured_full_csr_auto(context)


def solve_v3_full_system_structured_csr(
    *,
    nml: Namelist,
    which_rhs: int | None = None,
    op: V3FullSystemOperator | None = None,
    x0: jnp.ndarray | None = None,
    tol: float = 1.0e-10,
    atol: float = 0.0,
    restart: int = 80,
    maxiter: int | None = 400,
    identity_shift: float = 0.0,
    phi1_hat_base: jnp.ndarray | None = None,
    max_csr_nbytes: int | None = None,
    method: str = "gmres",
    preconditioner: str | None = "auto",
    preconditioner_max_schur_size: int = 2048,
    preconditioner_max_block_inverse_nbytes: int = 64 * 1024 * 1024,
    active_dof: bool = False,
    emit: Callable[[int, str], None] | None = None,
) -> V3LinearSolveResult:
    """Solve a supported RHSMode=1 system with explicit host CSR Krylov."""

    if op is None:
        op = full_system_operator_from_namelist(
            nml=nml,
            identity_shift=identity_shift,
            phi1_hat_base=phi1_hat_base,
        )
    if which_rhs is not None:
        op = with_transport_rhs_settings(op, which_rhs=int(which_rhs))
    rhs = rhs_v3_full_system(op)
    active_indices = transport_active_dof_indices(op) if bool(active_dof) else None
    if emit is not None:
        active_msg = (
            f" active_size={int(active_indices.size)}/{int(op.total_size)}"
            if active_indices is not None
            else " full_size"
        )
        emit(
            0,
            "solve_v3_full_system_structured_csr: assembling no-probe host CSR "
            f"(size={int(op.total_size)}{active_msg} method={method} preconditioner={preconditioner})",
        )
    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        x0=x0,
        tol=tol,
        atol=atol,
        restart=restart,
        maxiter=maxiter,
        method=method,
        preconditioner=preconditioner,
        preconditioner_max_schur_size=preconditioner_max_schur_size,
        preconditioner_max_block_inverse_nbytes=preconditioner_max_block_inverse_nbytes,
        max_csr_nbytes=max_csr_nbytes,
        active_indices=active_indices,
    )
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_structured_csr: "
            f"converged={bool(result.converged)} residual={float(result.residual_norm):.3e} "
            f"solve_s={float(result.solve_s):.3f}",
        )
        pc_summary = dict(result.metadata.get("preconditioner", {}) or {})
        pc_metadata = dict(pc_summary.get("metadata", {}) or {})
        factor_nbytes = pc_metadata.get("factor_nbytes_actual")
        if factor_nbytes is None:
            factor_nbytes = pc_metadata.get("factor_nbytes_estimate")
        if pc_summary:
            emit(
                0,
                "solve_v3_full_system_structured_csr: "
                f"pc_kind={pc_summary.get('kind', 'unknown')} "
                f"pc_selected={bool(pc_summary.get('selected', False))} "
                f"pc_reason={pc_summary.get('reason', 'unknown')} "
                f"pc_setup_s={float(pc_summary.get('setup_s', 0.0) or 0.0):.3f} "
                f"pc_factor_nbytes={factor_nbytes if factor_nbytes is not None else 'na'} "
                f"pc_permc={pc_metadata.get('permc_spec', 'na')} "
                f"pc_superlu_permc={pc_metadata.get('superlu_permc_spec', 'na')}",
            )
    return V3LinearSolveResult(
        op=op,
        rhs=rhs,
        gmres=GMRESSolveResult(
            x=jnp.asarray(result.x, dtype=jnp.float64),
            residual_norm=jnp.asarray(result.residual_norm, dtype=jnp.float64),
        ),
        metadata={
            "solver_path": "structured_full_csr_host_gmres",
            "structured_full_csr": result.to_dict(),
            "active_dof": bool(active_dof),
        },
    )


def solve_rhs1_structured_full_csr_explicit(context: RHS1StructuredCSRSolveContext) -> Any:
    """Run the explicit host-only structured full-CSR path and normalize metadata."""

    op = context.op
    if context.differentiable is True:
        raise ValueError(
            "solve_method='structured_csr' is host-only/non-differentiable; "
            "use differentiable=False or choose a JAX-native solve method."
        )
    if int(op.rhs_mode) != 1:
        raise ValueError("solve_method='structured_csr' is only implemented for RHSMode=1 full-system solves.")

    csr_max_mb = _env_float("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", 1024.0)
    pc_max_mb = _env_float("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", 128.0)
    pc_kind = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER", "auto").strip() or "auto"
    pc_schur_max = _env_int("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_SCHUR_SIZE", 2048)
    structured_krylov_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "").strip().lower()
    structured_krylov_default = "direct" if abs(float(context.identity_shift)) <= 0.0 else "gmres"
    structured_krylov = structured_krylov_env or structured_krylov_default
    active_dof_env = os.environ.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "").strip().lower()
    if active_dof_env in {"1", "true", "yes", "on", "active"}:
        structured_active_dof = True
    elif active_dof_env in {"0", "false", "no", "off", "full"}:
        structured_active_dof = False
    else:
        structured_active_dof = structured_krylov in {"direct", "splu", "sparse_direct"}

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: using structured full CSR host solve "
            f"(preconditioner={pc_kind} csr_max_mb={csr_max_mb:.3g} pc_max_mb={pc_max_mb:.3g} "
            f"active_dof={structured_active_dof})",
        )
    structured_result = context.structured_solver(
        nml=context.nml,
        which_rhs=None,
        op=op,
        x0=context.x0,
        tol=context.tol,
        atol=context.atol,
        restart=context.restart,
        maxiter=context.maxiter,
        identity_shift=context.identity_shift,
        phi1_hat_base=context.phi1_hat_base,
        max_csr_nbytes=int(max(0.0, float(csr_max_mb)) * 1024.0 * 1024.0),
        method=structured_krylov,
        preconditioner=pc_kind,
        preconditioner_max_schur_size=max(1, int(pc_schur_max)),
        preconditioner_max_block_inverse_nbytes=int(max(0.0, float(pc_max_mb)) * 1024.0 * 1024.0),
        active_dof=bool(structured_active_dof),
        emit=context.emit,
    )

    structured_metadata = dict(structured_result.metadata or {})
    structured_csr_metadata = structured_metadata.get("structured_full_csr", {})
    if not isinstance(structured_csr_metadata, dict):
        structured_csr_metadata = {}
    structured_solve_metadata = structured_csr_metadata.get("metadata", {})
    if not isinstance(structured_solve_metadata, dict):
        structured_solve_metadata = {}
    structured_selection = structured_csr_metadata.get("selection", {})
    if not isinstance(structured_selection, dict):
        structured_selection = {}
    structured_selection_metadata = structured_selection.get("metadata", {})
    if not isinstance(structured_selection_metadata, dict):
        structured_selection_metadata = {}
    structured_preconditioner = structured_solve_metadata.get("preconditioner", {})
    if not isinstance(structured_preconditioner, dict):
        structured_preconditioner = {}
    structured_preconditioner_metadata = structured_preconditioner.get("metadata", {})
    if not isinstance(structured_preconditioner_metadata, dict):
        structured_preconditioner_metadata = {}

    residual_norm = float(structured_csr_metadata.get("residual_norm", structured_result.gmres.residual_norm))
    target = float(structured_solve_metadata.get("target", max(float(context.atol), float(context.tol) * float(context.rhs_norm))))
    converged = bool(structured_csr_metadata.get("converged", residual_norm <= target))
    setup_s = float(structured_preconditioner.get("setup_s", 0.0) or 0.0)
    solve_s = float(structured_csr_metadata.get("solve_s", 0.0) or 0.0)
    direct_factor_s = structured_solve_metadata.get("factor_s", None)
    direct_factor_nbytes = structured_solve_metadata.get("factor_nbytes_actual", None)
    factor_nbytes = structured_preconditioner_metadata.get(
        "factor_nbytes_actual",
        structured_preconditioner_metadata.get("block_inverse_nbytes_actual", direct_factor_nbytes),
    )
    structured_metadata.update(
        {
            "solver_path": "structured_full_csr_host_gmres",
            "solver_kind": "structured_full_csr",
            "solve_method_requested": str(context.solve_method),
            "requested_solve_method": str(context.solve_method),
            "differentiable": False,
            "residual_kind": "true_residual",
            "accepted_converged": bool(converged),
            "acceptance_criterion": "true_residual",
            "reported_residual_norm": float(residual_norm),
            "iterations": len(tuple(structured_csr_metadata.get("residual_history", ()) or ())),
            "info_code": int(structured_csr_metadata.get("info", 0)),
            "setup_s": setup_s,
            "solve_s": solve_s,
            "elapsed_s": setup_s + solve_s,
            "csr_nnz": int(structured_selection_metadata.get("nnz", structured_solve_metadata.get("matrix_nnz", 0)) or 0),
            "csr_operator_nbytes": int(structured_selection_metadata.get("csr_nbytes_actual", 0) or 0),
            "preconditioner_kind": str(structured_preconditioner.get("kind", pc_kind)),
            "sparse_pc_factor_nbytes_estimate": None if factor_nbytes is None else int(factor_nbytes),
            "direct_factor_s": None if direct_factor_s is None else float(direct_factor_s),
            "direct_factor_nbytes_actual": None if direct_factor_nbytes is None else int(direct_factor_nbytes),
            "structured_active_dof": bool(structured_solve_metadata.get("active_dof", False)),
            "structured_active_size": int(structured_solve_metadata.get("active_size", 0) or 0),
            "structured_full_size": int(structured_solve_metadata.get("full_size", 0) or 0),
            "structured_full_csr_env": {
                "csr_max_mb": float(csr_max_mb),
                "preconditioner": str(pc_kind),
                "preconditioner_max_mb": float(pc_max_mb),
                "preconditioner_max_schur_size": int(pc_schur_max),
                "krylov": str(structured_krylov),
                "active_dof": bool(structured_active_dof),
            },
        }
    )
    return replace(structured_result, metadata=structured_metadata)


def try_rhs1_sparse_host_safe_solve(context: RHS1SparseHostSafeSolveContext) -> Any | None:
    """Run ``sparse_host_safe`` or return ``None`` when it was not requested."""

    if not bool(context.requested):
        return None
    try:
        direct_result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=context.op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="sparse_host",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=context.differentiable,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
    except RuntimeError as exc:
        if "Host sparse factorization failed" not in str(exc):
            raise
        op = context.op
        constrained_pas = bool(
            int(op.rhs_mode) == 1
            and int(op.constraint_scheme) == 2
            and (not bool(op.include_phi1))
            and op.fblock.pas is not None
        )
        if not constrained_pas:
            raise
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: sparse_host_safe falling back to "
                "PETSc-compatible minimum-norm constrained-PAS branch after sparse LU failure",
            )
        compat_result = context.solve_driver(
            nml=context.nml,
            which_rhs=context.which_rhs,
            op=op,
            x0=context.x0,
            tol=context.tol,
            atol=context.atol,
            restart=context.restart,
            maxiter=context.maxiter,
            solve_method="petsc_compat",
            identity_shift=context.identity_shift,
            phi1_hat_base=context.phi1_hat_base,
            differentiable=context.differentiable,
            emit=context.emit,
            recycle_basis=context.recycle_basis,
        )
        return _annotate_auto_result(
            compat_result,
            {
                "requested_solve_method": str(context.solve_method_kind_explicit),
                "safe_sparse_host_fallback_used": True,
                "sparse_host_failure": str(exc),
            },
        )

    metadata = dict(getattr(direct_result, "metadata", None) or {})
    metadata.update(
        {
            "requested_solve_method": str(context.solve_method_kind_explicit),
            "safe_sparse_host_fallback_used": False,
            "accepted_converged": bool(metadata.get("accepted_converged", True)),
            "acceptance_criterion": metadata.get("acceptance_criterion", "true_residual"),
        }
    )
    return replace(direct_result, metadata=metadata)



__all__ = [
    "try_rhs1_auto_host_solve",
    "try_rhs1_sparse_host_safe_solve",
    "solve_v3_full_system_structured_csr",
    "solve_rhs1_structured_full_csr_explicit",
    "RHS1StructuredCSRSolveContext",
    "RHS1SparseHostSafeSolveContext",
    "RHS1AutoHostSolveContext",
    "RHS1DenseProbeAdmission",
    "RHS1DenseProbeStageContext",
    "RHS1DenseProbeStageResult",
    "RHS1DenseProbeShortcutDecision",
    "RHS1DenseFallbackThresholds",
    "RHS1DenseFallbackAdmission",
    "RHS1EarlyDenseShortcutDecision",
    "RHS1PostKrylovDenseShortcutDecision",
    "RHS1PostKrylovDenseShortcutEvaluation",
    "RHS1PostKrylovDenseShortcutEvaluationContext",
    "RHS1DenseShortcutSetup",
    "HostDenseFullSolveContext",
    "HostDenseReducedSolveContext",
    "RHS1FullHostDenseShortcutContext",
    "RHS1FullHostDenseShortcutResult",
    "RHS1FullDenseFallbackContext",
    "RHS1FullDenseFallbackStageContext",
    "RHS1ReducedHostDenseShortcutContext",
    "RHS1ReducedHostDenseShortcutResult",
    "RHS1ScipyRescueContext",
    "RHS1ScipyRescueOutcome",
    "RHS1ScipyRescueStageContext",
    "RHS1ScipyRescueStageResult",
    "RHS1ReducedDenseFallbackAdmissionStageContext",
    "RHS1ReducedDenseFallbackCandidateContext",
    "RHS1ReducedDenseFallbackStageContext",
    "rhs1_dense_probe_admission",
    "rhs1_dense_probe_enabled_from_env",
    "rhs1_dense_probe_shortcut_decision",
    "rhs1_early_dense_shortcut_decision",
    "rhs1_evaluate_post_krylov_dense_shortcut",
    "rhs1_post_krylov_dense_shortcut_decision",
    "rhs1_dense_fallback_thresholds_from_env",
    "rhs1_dense_shortcut_setup_from_env",
    "rhs1_fp_preconditioner_probe_kind_from_env",
    "resolve_rhs1_full_dense_fallback_admission",
    "resolve_rhs1_reduced_dense_fallback_admission",
    "run_rhs1_full_dense_fallback_candidate",
    "run_rhs1_full_dense_fallback_stage",
    "run_rhs1_full_host_dense_shortcut_stage",
    "run_rhs1_dense_probe_stage",
    "run_rhs1_reduced_dense_fallback_admission_stage",
    "run_rhs1_reduced_dense_fallback_stage",
    "run_rhs1_reduced_host_dense_shortcut_stage",
    "run_rhs1_scipy_rescue",
    "run_rhs1_scipy_rescue_stage",
    "solve_rhs1_reduced_dense_fallback_candidate",
    "solve_host_dense_full",
    "solve_host_dense_reduced",
]
