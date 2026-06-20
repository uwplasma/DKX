"""X-block sparse rescue stages for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..policies import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
)
from ....solver import GMRESSolveResult


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


def _env_value(env: object, key: str) -> str:
    if not isinstance(env, dict):
        try:
            return str(env.get(key, "")).strip()  # type: ignore[union-attr]
        except AttributeError:
            return ""
    return str(env.get(key, "")).strip()


@dataclass(frozen=True)
class SparseXBlockRescueBuildContext:
    """Dependencies for the generic sparse x-block rescue preconditioner build."""

    op: object
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    active_size: int
    preconditioner_species: int
    preconditioner_x: int
    preconditioner_xi: int
    use_implicit: bool
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    emit: EmitFn | None
    mark: Callable[[str], None]
    assembled_host_allowed: Callable[..., bool]
    builder: Callable[..., ArrayFn]


@dataclass(frozen=True)
class SparseXBlockRescueBuildResult:
    """Result from building the generic sparse x-block rescue preconditioner."""

    preconditioner: ArrayFn
    preconditioner_xi: int
    force_assembled_host_fp: bool


@dataclass(frozen=True)
class SparseXBlockExplicitSeedContext:
    """Inputs for the explicit FP x-block seed/refine/polish path."""

    preconditioner: ArrayFn
    rhs: jnp.ndarray
    matvec: ArrayFn
    current_result: GMRESSolveResult
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    active_size: int
    emit: EmitFn | None
    polish_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


@dataclass(frozen=True)
class SparseXBlockExplicitSeedResult:
    """Explicit FP x-block seed outcome and diagnostics."""

    result: GMRESSolveResult | None
    seed_residual: float
    seed_improvement_ratio: float
    seed_accept_ratio: float
    refine_steps: int
    refines_performed: int
    reason: str


@dataclass(frozen=True)
class SparseXBlockRescueSolveContext:
    """Inputs for one generic sparse x-block rescue solve candidate."""

    preconditioner: ArrayFn
    rhs: jnp.ndarray
    matvec: ArrayFn
    current_result: GMRESSolveResult
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    active_size: int
    use_implicit: bool
    assembled_host_fp: bool
    emit: EmitFn | None
    mark: Callable[[str], None]
    solve_linear: Callable[..., GMRESSolveResult]
    host_gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


@dataclass(frozen=True)
class SparseXBlockRescueSolveResult:
    """Solve candidate and diagnostics for generic sparse x-block rescue."""

    result: GMRESSolveResult | None
    reason: str
    candidate_residual: float | None = None
    seed_residual: float | None = None
    seed_improvement_ratio: float | None = None
    seed_accept_ratio: float | None = None
    seed_refine_steps: int | None = None
    seed_refines_performed: int | None = None


@dataclass(frozen=True)
class SparseXBlockRescueAcceptanceContext:
    """Inputs for accepting a sparse x-block rescue candidate."""

    current_result: GMRESSolveResult
    candidate_result: GMRESSolveResult | None
    reason: str
    assembled_host_fp: bool
    use_implicit: bool
    replay_state: Any
    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    precondition_side: str
    solver_kind: str
    restart: int
    maxiter: int | None
    record_replay_problem: Callable[..., None]


@dataclass(frozen=True)
class SparseXBlockRescueAcceptanceResult:
    """Accepted sparse x-block rescue state and replay diagnostics."""

    result: GMRESSolveResult
    accepted: bool
    reason: str
    candidate_residual: float | None = None
    explicit_seed_used: bool = False


@dataclass(frozen=True)
class SparseSXBlockRescueContext:
    """Dependencies for the sparse sxblock_tz seed and optional polish stage."""

    op: Any
    current_result: GMRESSolveResult
    matvec: ArrayFn
    rhs: jnp.ndarray
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    preconditioner: ArrayFn | None
    replay_state: Any
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    target: float
    precondition_side: str
    solver_kind: str
    emit: EmitFn | None
    mark: Callable[[str], None]
    seed_builder: Callable[..., jnp.ndarray]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    parse_polish_gmres_config: Callable[..., tuple[int, int]]
    record_replay_problem: Callable[..., None]


@dataclass(frozen=True)
class SparseSXBlockRescueResult:
    """Updated state and diagnostics from the sparse sxblock_tz rescue stage."""

    result: GMRESSolveResult
    accepted: bool
    polished: bool
    error: str | None
    seed_residual: float | None
    polish_residual: float | None
    polish_restart: int | None
    polish_maxiter: int | None


@dataclass(frozen=True)
class FPXBlockGlobalCorrectionContext:
    """Dependencies for the optional FP x-block global correction stage."""

    current_result: GMRESSolveResult
    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn | None
    preconditioner_label: str | None
    steps: int
    alpha_clip: float
    min_improvement: float
    preconditioner_clip: float
    replay_state: Any
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    mark: Callable[[str], None]
    safe_preconditioner: Callable[..., ArrayFn]
    correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]]


@dataclass(frozen=True)
class FPXBlockGlobalCorrectionResult:
    """Updated state and diagnostics from the FP x-block global correction."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    accepted: bool
    reason: str
    error: str | None
    preconditioner_label: str | None
    steps: int | None
    accepted_steps: int | None
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    elapsed_s: float | None


@dataclass(frozen=True)
class FPXBlockHighXCorrectionContext:
    """Dependencies for FP high-x residual-equation correction."""

    current_result: GMRESSolveResult
    matvec: ArrayFn
    rhs: jnp.ndarray
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    total_size: int
    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    n_xi_for_x: Sequence[int]
    host_block_max_env_value: str
    include_factored_blocks: bool
    max_blocks: int
    steps: int
    max_directions: int
    alpha_clip: float
    rcond: float
    min_improvement: float
    include_all: bool
    include_raw: bool
    replay_state: Any
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    mark: Callable[[str], None]
    block_factor_allowed: Callable[..., bool]
    correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[int], Sequence[str]]]


@dataclass(frozen=True)
class FPXBlockHighXCorrectionResult:
    """Updated state and diagnostics from the FP high-x correction."""

    result: GMRESSolveResult
    residual_vec: jnp.ndarray | None
    accepted: bool
    reason: str
    error: str | None
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    elapsed_s: float | None
    direction_count: int | None
    direction_names: tuple[str, ...]


def build_sparse_xblock_rescue_preconditioner(
    *,
    context: SparseXBlockRescueBuildContext,
) -> SparseXBlockRescueBuildResult:
    """Build the generic sparse x-block rescue preconditioner."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: v3-like sparse x-block rescue "
            f"(size={int(context.active_size)} preconditioner_x={int(context.preconditioner_x)})",
        )

    preconditioner_xi = int(context.preconditioner_xi)
    fblock = getattr(context.op, "fblock", None)
    if (
        preconditioner_xi == 0
        and not bool(context.use_implicit)
        and getattr(fblock, "fp", None) is not None
        and getattr(fblock, "pas", None) is None
    ):
        preconditioner_xi = 1
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: promoting sparse x-block rescue "
                "preconditioner_xi 0 -> 1 for stronger host FP factorization",
            )

    force_assembled_host_fp = bool(
        context.assembled_host_allowed(
            op=context.op,
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_xi=int(preconditioner_xi),
            use_implicit=bool(context.use_implicit),
            active_size=int(context.active_size),
        )
    )
    context.mark("rhs1_sparse_precond_build_start")
    preconditioner = context.builder(
        op=context.op,
        reduce_full=context.reduce_full,
        expand_reduced=context.expand_reduced,
        build_jax_factors=bool(context.use_implicit),
        preconditioner_species=int(context.preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        drop_tol=float(context.drop_tol),
        drop_rel=float(context.drop_rel),
        ilu_drop_tol=float(context.ilu_drop_tol),
        fill_factor=float(context.fill_factor),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        emit=context.emit,
    )
    context.mark("rhs1_sparse_precond_build_done")
    return SparseXBlockRescueBuildResult(
        preconditioner=preconditioner,
        preconditioner_xi=int(preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
    )


def apply_sparse_xblock_explicit_seed(
    *,
    context: SparseXBlockExplicitSeedContext,
) -> SparseXBlockExplicitSeedResult:
    """Apply, refine, and optionally polish the explicit FP x-block seed."""

    refine_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XBLOCK_REFINES", "").strip()
    try:
        refine_steps = int(refine_env) if refine_env else 2
    except ValueError:
        refine_steps = 2
    refine_steps = max(0, int(refine_steps))
    accept_ratio = rhs1_parse_accept_ratio(
        env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_ACCEPT_RATIO",
        default=10.0,
    )
    polish_enabled = rhs1_polish_enabled(
        env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH",
    )
    polish_restart, polish_maxiter = rhs1_parse_polish_gmres_config(
        restart_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_RESTART",
        maxiter_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_MAXITER",
        default_restart=min(int(context.restart), 40),
        default_maxiter=min(int(context.maxiter or 80), 80),
        active_size=int(context.active_size),
        large_active_min_env_name="SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_MIN",
        large_default_restart_env_name=(
            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_RESTART_DEFAULT"
        ),
        large_default_maxiter_env_name=(
            "SFINCS_JAX_RHSMODE1_FP_XBLOCK_POLISH_LARGE_MAXITER_DEFAULT"
        ),
        default_large_restart=10,
        default_large_maxiter=1,
        min_maxiter=1,
    )
    base_residual_norm = float(context.current_result.residual_norm)
    x_trial = jnp.asarray(context.preconditioner(context.rhs), dtype=jnp.float64)
    residual_vec = context.rhs - context.matvec(x_trial)
    residual_norm = float(jnp.linalg.norm(residual_vec))
    seed_residual_initial = float(residual_norm)
    improvement_ratio = 1.0
    if np.isfinite(residual_norm) and residual_norm > 0.0:
        improvement_ratio = float(base_residual_norm) / float(residual_norm)
    elif np.isfinite(residual_norm):
        improvement_ratio = float("inf")

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: explicit FP x-block seed "
            f"(residual={residual_norm:.6e} current={base_residual_norm:.6e})",
        )

    performed_refines = 0
    for refine_index in range(int(refine_steps)):
        if not np.isfinite(residual_norm) or residual_norm == 0.0:
            break
        dx_trial = jnp.asarray(context.preconditioner(residual_vec), dtype=jnp.float64)
        x_next = x_trial + dx_trial
        residual_vec_next = context.rhs - context.matvec(x_next)
        residual_norm_next = float(jnp.linalg.norm(residual_vec_next))
        if not np.isfinite(residual_norm_next) or residual_norm_next >= residual_norm:
            break
        x_trial = x_next
        residual_vec = residual_vec_next
        residual_norm = residual_norm_next
        performed_refines = int(refine_index) + 1

    if context.emit is not None and int(refine_steps) > 0:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: explicit FP x-block refinement "
            f"steps={int(performed_refines)}/{int(refine_steps)} "
            f"residual={float(residual_norm):.6e}",
        )

    reason = "seed_rejected_accept_gate"
    result: GMRESSolveResult | None = None
    if (
        np.isfinite(residual_norm)
        and residual_norm <= max(float(context.target), base_residual_norm * accept_ratio)
    ):
        reason = "seed_accepted"
        if bool(polish_enabled) and residual_norm > float(context.target):
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: explicit FP x-block polish "
                    f"start residual={float(residual_norm):.6e} "
                    f"target={float(context.target):.3e} restart={int(polish_restart)} "
                    f"maxiter={int(polish_maxiter)}",
                )
            x_np, _rn, _history = context.polish_solver(
                matvec=context.matvec,
                b=context.rhs,
                preconditioner=context.preconditioner,
                x0=x_trial,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(polish_restart),
                maxiter=int(polish_maxiter),
                precondition_side=context.precondition_side,
            )
            x_polish = jnp.asarray(x_np, dtype=jnp.float64)
            residual_vec_polish = context.rhs - context.matvec(x_polish)
            residual_norm_polish = float(jnp.linalg.norm(residual_vec_polish))
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: explicit FP x-block polish "
                    f"done residual={float(residual_norm_polish):.6e}",
                )
            if np.isfinite(residual_norm_polish) and residual_norm_polish < residual_norm:
                x_trial = x_polish
                residual_norm = residual_norm_polish
        result = GMRESSolveResult(
            x=x_trial,
            residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        )
    elif context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: explicit FP x-block seed rejected "
            f"(residual={residual_norm:.6e}, base={base_residual_norm:.6e}, "
            f"accept_ratio={accept_ratio:.1e})",
        )

    return SparseXBlockExplicitSeedResult(
        result=result,
        seed_residual=float(seed_residual_initial),
        seed_improvement_ratio=float(improvement_ratio),
        seed_accept_ratio=float(accept_ratio),
        refine_steps=int(refine_steps),
        refines_performed=int(performed_refines),
        reason=reason,
    )


def run_sparse_xblock_rescue_solve_stage(
    *,
    context: SparseXBlockRescueSolveContext,
) -> SparseXBlockRescueSolveResult:
    """Run one sparse x-block rescue solve candidate without accepting it."""

    context.mark("rhs1_sparse_precond_solve_start")
    try:
        if bool(context.use_implicit):
            result = context.solve_linear(
                matvec_fn=context.matvec,
                b_vec=context.rhs,
                precond_fn=context.preconditioner,
                x0_vec=context.current_result.x,
                tol_val=float(context.tol),
                atol_val=float(context.atol),
                restart_val=int(context.restart),
                maxiter_val=context.maxiter,
                solve_method_val="incremental",
                precond_side=context.precondition_side,
            )
            return SparseXBlockRescueSolveResult(
                result=result,
                reason="started",
            )

        if bool(context.assembled_host_fp):
            seed = apply_sparse_xblock_explicit_seed(
                context=SparseXBlockExplicitSeedContext(
                    preconditioner=context.preconditioner,
                    rhs=context.rhs,
                    matvec=context.matvec,
                    current_result=context.current_result,
                    target=float(context.target),
                    tol=float(context.tol),
                    atol=float(context.atol),
                    restart=int(context.restart),
                    maxiter=context.maxiter,
                    precondition_side=context.precondition_side,
                    active_size=int(context.active_size),
                    emit=context.emit,
                    polish_solver=context.host_gmres_solver,
                )
            )
            return SparseXBlockRescueSolveResult(
                result=seed.result,
                reason=seed.reason,
                seed_residual=float(seed.seed_residual),
                seed_improvement_ratio=float(seed.seed_improvement_ratio),
                seed_accept_ratio=float(seed.seed_accept_ratio),
                seed_refine_steps=int(seed.refine_steps),
                seed_refines_performed=int(seed.refines_performed),
            )

        x_np, _rn, _history = context.host_gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.current_result.x,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            precondition_side=context.precondition_side,
        )
        x_sparse_xblock = jnp.asarray(x_np, dtype=jnp.float64)
        residual_vec = context.rhs - context.matvec(x_sparse_xblock)
        result = GMRESSolveResult(
            x=x_sparse_xblock,
            residual_norm=jnp.asarray(jnp.linalg.norm(residual_vec), dtype=jnp.float64),
        )
        return SparseXBlockRescueSolveResult(
            result=result,
            reason="gmres_candidate",
            candidate_residual=float(result.residual_norm),
        )
    finally:
        context.mark("rhs1_sparse_precond_solve_done")


def accept_sparse_xblock_rescue_candidate(
    *,
    context: SparseXBlockRescueAcceptanceContext,
) -> SparseXBlockRescueAcceptanceResult:
    """Accept an improving sparse x-block candidate and update replay state."""

    candidate = context.candidate_result
    if candidate is None or not (
        float(candidate.residual_norm) < float(context.current_result.residual_norm)
    ):
        return SparseXBlockRescueAcceptanceResult(
            result=context.current_result,
            accepted=False,
            reason=str(context.reason),
        )

    reason = str(context.reason)
    if reason == "gmres_candidate":
        reason = "gmres_candidate_improved"
    explicit_seed_used = bool(context.assembled_host_fp and (not bool(context.use_implicit)))
    if bool(context.assembled_host_fp):
        context.replay_state.x0_vec = candidate.x
    else:
        context.record_replay_problem(
            context.replay_state,
            matvec_fn=context.matvec,
            b_vec=context.rhs,
            precond_fn=context.preconditioner,
            x0_vec=candidate.x,
            precond_side=context.precondition_side,
            solver_kind=context.solver_kind,
            restart=int(context.restart),
            maxiter=context.maxiter,
        )
    return SparseXBlockRescueAcceptanceResult(
        result=candidate,
        accepted=True,
        reason=reason,
        candidate_residual=float(candidate.residual_norm),
        explicit_seed_used=bool(explicit_seed_used),
    )


def run_sparse_sxblock_rescue_stage(
    *,
    context: SparseSXBlockRescueContext,
) -> SparseSXBlockRescueResult:
    """Run sparse sxblock_tz seed rescue and optional GMRES polish."""

    try:
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: sparse sxblock_tz rescue "
                f"(size={int(context.current_result.x.size)} "
                f"n_species={int(context.op.n_species)})",
            )
        context.mark("rhs1_sparse_precond_build_start")
        x_sparse = context.seed_builder(
            op=context.op,
            rhs_reduced=context.rhs,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            drop_tol=float(context.drop_tol),
            drop_rel=float(context.drop_rel),
            ilu_drop_tol=float(context.ilu_drop_tol),
            fill_factor=float(context.fill_factor),
            emit=context.emit,
        )
        context.mark("rhs1_sparse_precond_build_done")
        context.mark("rhs1_sparse_precond_solve_start")
        residual_vec_sparse = context.rhs - context.matvec(x_sparse)
        seed_result = GMRESSolveResult(
            x=x_sparse,
            residual_norm=jnp.asarray(
                jnp.linalg.norm(residual_vec_sparse),
                dtype=jnp.float64,
            ),
        )
        seed_residual = float(seed_result.residual_norm)
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: explicit sxblock seed "
                f"(residual={seed_residual:.6e})",
            )
        context.mark("rhs1_sparse_precond_solve_done")
        if seed_residual >= float(context.current_result.residual_norm):
            return SparseSXBlockRescueResult(
                result=context.current_result,
                accepted=False,
                polished=False,
                error=None,
                seed_residual=seed_residual,
                polish_residual=None,
                polish_restart=None,
                polish_maxiter=None,
            )

        result = seed_result
        context.replay_state.x0_vec = result.x
        polish_residual: float | None = None
        polish_restart: int | None = None
        polish_maxiter: int | None = None
        polished = False
        if float(result.residual_norm) > float(context.target):
            polish_precond = context.preconditioner
            if polish_precond is not None:
                polish_restart, polish_maxiter = context.parse_polish_gmres_config(
                    restart_env_name="SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_RESTART",
                    maxiter_env_name="SFINCS_JAX_RHSMODE1_SXBLOCK_POLISH_MAXITER",
                    default_restart=min(int(context.restart), 40),
                    default_maxiter=min(
                        max(40, int(context.maxiter or 120)),
                        120,
                    ),
                )
                if context.emit is not None:
                    context.emit(
                        0,
                        "solve_v3_full_system_linear_gmres: sxblock seed polish "
                        f"restart={polish_restart} maxiter={polish_maxiter}",
                    )
                x_np, _rn_polish, _history = context.gmres_solver(
                    matvec=context.matvec,
                    b=context.rhs,
                    preconditioner=polish_precond,
                    x0=result.x,
                    tol=float(context.tol),
                    atol=float(context.atol),
                    restart=int(polish_restart),
                    maxiter=int(polish_maxiter),
                    precondition_side=context.precondition_side,
                )
                x_polish = jnp.asarray(x_np, dtype=jnp.float64)
                residual_vec_polish = context.rhs - context.matvec(x_polish)
                polish_candidate = GMRESSolveResult(
                    x=x_polish,
                    residual_norm=jnp.asarray(
                        jnp.linalg.norm(residual_vec_polish),
                        dtype=jnp.float64,
                    ),
                )
                polish_residual = float(polish_candidate.residual_norm)
                if polish_residual < float(result.residual_norm):
                    result = polish_candidate
                    polished = True
                    context.record_replay_problem(
                        context.replay_state,
                        matvec_fn=context.matvec,
                        b_vec=context.rhs,
                        precond_fn=polish_precond,
                        x0_vec=result.x,
                        precond_side=context.precondition_side,
                        solver_kind=context.solver_kind,
                        restart=int(polish_restart),
                        maxiter=int(polish_maxiter),
                    )

        return SparseSXBlockRescueResult(
            result=result,
            accepted=True,
            polished=bool(polished),
            error=None,
            seed_residual=seed_residual,
            polish_residual=polish_residual,
            polish_restart=polish_restart,
            polish_maxiter=polish_maxiter,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(1, f"sxblock_sparse: failed ({error})")
        return SparseSXBlockRescueResult(
            result=context.current_result,
            accepted=False,
            polished=False,
            error=error,
            seed_residual=None,
            polish_residual=None,
            polish_restart=None,
            polish_maxiter=None,
        )


def run_fp_xblock_global_correction_stage(
    *,
    context: FPXBlockGlobalCorrectionContext,
) -> FPXBlockGlobalCorrectionResult:
    """Run the optional FP x-block global correction and accept improvement."""

    if context.preconditioner is None:
        return FPXBlockGlobalCorrectionResult(
            result=context.current_result,
            residual_vec=None,
            accepted=False,
            reason="missing_preconditioner",
            error=None,
            preconditioner_label=context.preconditioner_label,
            steps=None,
            accepted_steps=None,
            residual_before=None,
            residual_after=None,
            improvement_ratio=None,
            elapsed_s=None,
        )

    steps = int(context.steps)
    residual_before = float(context.current_result.residual_norm)
    start_s = float(context.elapsed_s())
    context.mark("rhs1_fp_xblock_global_correction_start")
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: FP x-block global correction "
            f"(steps={steps} preconditioner={context.preconditioner_label} "
            f"residual={residual_before:.6e})",
        )

    try:
        x_corr, residual_corr, correction_history, correction_alphas = (
            context.correction(
                matvec=context.matvec,
                rhs=context.rhs,
                x0=context.current_result.x,
                preconditioner=context.safe_preconditioner(
                    context.preconditioner,
                    clip=float(context.preconditioner_clip),
                ),
                steps=steps,
                alpha_clip=float(context.alpha_clip),
                min_improvement=float(context.min_improvement),
            )
        )
        elapsed_s = float(context.elapsed_s() - start_s)
        accepted_steps = int(len(correction_alphas))
        residual_after = (
            float(correction_history[-1]) if correction_history else None
        )
        if (
            residual_after is not None
            and np.isfinite(float(residual_after))
            and float(residual_after) < residual_before
        ):
            improvement_ratio = residual_before / max(float(residual_after), 1.0e-300)
            accepted_result = GMRESSolveResult(
                x=jnp.asarray(x_corr, dtype=jnp.float64),
                residual_norm=jnp.asarray(float(residual_after), dtype=jnp.float64),
            )
            context.replay_state.x0_vec = accepted_result.x
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: FP x-block global "
                    f"correction accepted {residual_before:.3e}->{float(residual_after):.3e} "
                    f"steps={accepted_steps}",
                )
            context.mark("rhs1_fp_xblock_global_correction_done")
            return FPXBlockGlobalCorrectionResult(
                result=accepted_result,
                residual_vec=jnp.asarray(residual_corr, dtype=jnp.float64),
                accepted=True,
                reason="accepted",
                error=None,
                preconditioner_label=context.preconditioner_label,
                steps=steps,
                accepted_steps=accepted_steps,
                residual_before=residual_before,
                residual_after=float(residual_after),
                improvement_ratio=float(improvement_ratio),
                elapsed_s=elapsed_s,
            )

        context.mark("rhs1_fp_xblock_global_correction_done")
        return FPXBlockGlobalCorrectionResult(
            result=context.current_result,
            residual_vec=None,
            accepted=False,
            reason="no_improvement",
            error=None,
            preconditioner_label=context.preconditioner_label,
            steps=steps,
            accepted_steps=accepted_steps,
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=None,
            elapsed_s=elapsed_s,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        elapsed_s = float(context.elapsed_s() - start_s)
        context.mark("rhs1_fp_xblock_global_correction_failed")
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: FP x-block global correction "
                f"failed ({error})",
            )
        return FPXBlockGlobalCorrectionResult(
            result=context.current_result,
            residual_vec=None,
            accepted=False,
            reason="exception",
            error=error,
            preconditioner_label=context.preconditioner_label,
            steps=steps,
            accepted_steps=None,
            residual_before=residual_before,
            residual_after=None,
            improvement_ratio=None,
            elapsed_s=elapsed_s,
        )


def run_fp_xblock_highx_residual_correction_stage(
    *,
    context: FPXBlockHighXCorrectionContext,
) -> FPXBlockHighXCorrectionResult:
    """Run the optional FP high-x residual-equation correction stage."""

    start_s = float(context.elapsed_s())
    residual_before: float | None = None
    context.mark("rhs1_fp_xblock_highx_residual_correction_start")
    try:
        highx_slices: list[tuple[str, int, int]] = []
        nxi_for_x = tuple(int(v) for v in context.n_xi_for_x)
        for species in range(int(context.n_species)):
            for ix in range(int(context.n_x)):
                n_lx = int(nxi_for_x[int(ix)])
                block_size = int(n_lx * int(context.n_theta) * int(context.n_zeta))
                if block_size <= 0:
                    continue
                block_factor_allowed = bool(
                    context.block_factor_allowed(
                        block_size=block_size,
                        max_block_size_env_value=context.host_block_max_env_value,
                    )
                )
                if block_factor_allowed and not bool(context.include_factored_blocks):
                    continue
                start = int(
                    (int(species) * int(context.n_x) + int(ix))
                    * int(context.n_xi)
                    * int(context.n_theta)
                    * int(context.n_zeta)
                )
                highx_slices.append((f"s{int(species)}_x{int(ix)}", start, block_size))

        highx_slices = highx_slices[: int(context.max_blocks)]
        if not highx_slices:
            context.mark("rhs1_fp_xblock_highx_residual_correction_done")
            return FPXBlockHighXCorrectionResult(
                result=context.current_result,
                residual_vec=None,
                accepted=False,
                reason="no_skipped_blocks",
                error=None,
                residual_before=None,
                residual_after=None,
                improvement_ratio=None,
                elapsed_s=None,
                direction_count=None,
                direction_names=(),
            )

        residual_before = float(context.current_result.residual_norm)
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: FP high-x "
                "residual-equation correction "
                f"(blocks={len(highx_slices)} directions<={int(context.max_directions)} "
                f"residual={residual_before:.6e})",
            )

        def _direction_builder(
            residual_reduced: jnp.ndarray,
        ) -> tuple[tuple[str, jnp.ndarray], ...]:
            residual_full_np = np.asarray(
                jax.device_get(
                    context.expand_reduced(
                        jnp.asarray(residual_reduced, dtype=jnp.float64)
                    )
                ),
                dtype=np.float64,
            ).reshape((-1,))
            directions: list[tuple[str, jnp.ndarray]] = []
            if bool(context.include_raw):
                directions.append(
                    ("raw_residual", jnp.asarray(residual_reduced, dtype=jnp.float64))
                )

            def _direction_for(
                blocks: Sequence[tuple[str, int, int]],
            ) -> jnp.ndarray | None:
                full_np = np.zeros((int(context.total_size),), dtype=np.float64)
                for _label, block_start, block_size in blocks:
                    sl = slice(int(block_start), int(block_start + block_size))
                    full_np[sl] = residual_full_np[sl]
                if not np.any(np.isfinite(full_np) & (full_np != 0.0)):
                    return None
                return context.reduce_full(jnp.asarray(full_np, dtype=jnp.float64))

            if bool(context.include_all):
                all_direction = _direction_for(highx_slices)
                if all_direction is not None:
                    directions.append(("highx_all", all_direction))
            for label, block_start, block_size in highx_slices:
                direction = _direction_for(((label, block_start, block_size),))
                if direction is not None:
                    directions.append((f"highx_{label}", direction))
            return tuple(directions)

        x_highx, residual_highx, history, counts, names = context.correction(
            matvec=context.matvec,
            rhs=context.rhs,
            x0=context.current_result.x,
            direction_builder=_direction_builder,
            steps=int(context.steps),
            max_directions=int(context.max_directions),
            alpha_clip=float(context.alpha_clip),
            rcond=float(context.rcond),
            min_improvement=float(context.min_improvement),
        )
        elapsed_s = float(context.elapsed_s() - start_s)
        direction_count = int(sum(int(v) for v in counts))
        direction_names = tuple(str(v) for v in names)
        residual_after = float(history[-1]) if history else None
        if (
            residual_after is not None
            and np.isfinite(float(residual_after))
            and float(residual_after) < residual_before
        ):
            improvement_ratio = residual_before / max(float(residual_after), 1.0e-300)
            accepted_result = GMRESSolveResult(
                x=jnp.asarray(x_highx, dtype=jnp.float64),
                residual_norm=jnp.asarray(float(residual_after), dtype=jnp.float64),
            )
            context.replay_state.x0_vec = accepted_result.x
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: FP high-x "
                    f"residual-equation correction accepted {residual_before:.3e}"
                    f"->{float(residual_after):.3e} directions={direction_count}",
                )
            context.mark("rhs1_fp_xblock_highx_residual_correction_done")
            return FPXBlockHighXCorrectionResult(
                result=accepted_result,
                residual_vec=jnp.asarray(residual_highx, dtype=jnp.float64),
                accepted=True,
                reason="accepted",
                error=None,
                residual_before=residual_before,
                residual_after=float(residual_after),
                improvement_ratio=float(improvement_ratio),
                elapsed_s=elapsed_s,
                direction_count=direction_count,
                direction_names=direction_names,
            )

        context.mark("rhs1_fp_xblock_highx_residual_correction_done")
        return FPXBlockHighXCorrectionResult(
            result=context.current_result,
            residual_vec=None,
            accepted=False,
            reason="no_improvement",
            error=None,
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=None,
            elapsed_s=elapsed_s,
            direction_count=direction_count,
            direction_names=direction_names,
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        elapsed_s = float(context.elapsed_s() - start_s)
        context.mark("rhs1_fp_xblock_highx_residual_correction_failed")
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: FP high-x "
                f"residual-equation correction failed ({error})",
            )
        return FPXBlockHighXCorrectionResult(
            result=context.current_result,
            residual_vec=None,
            accepted=False,
            reason="exception",
            error=error,
            residual_before=residual_before,
            residual_after=None,
            improvement_ratio=None,
            elapsed_s=elapsed_s,
            direction_count=None,
            direction_names=(),
        )



class MatvecCounter:
    """Mutable matvec counter that preserves ``int(counter)`` call sites."""

    def __init__(self, value: int = 0) -> None:
        self.value = int(value)

    def increment(self) -> None:
        self.value += 1

    def __iadd__(self, increment: int) -> "MatvecCounter":
        self.value += int(increment)
        return self

    def __int__(self) -> int:
        return int(self.value)

    def __mod__(self, divisor: int) -> int:
        return int(self.value) % int(divisor)

@dataclass(frozen=True)
class XBlockKrylovMatvecSetup:
    """Active-DOF reduction and true-matvec context for x-block Krylov solves."""

    progress_every: int
    mv_count: MatvecCounter
    xblock_linear_size: int
    xblock_active_idx_np: np.ndarray | None
    xblock_rhs: jnp.ndarray
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    matvec_no_count: ArrayFn
    matvec: ArrayFn
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class XBlockInitialGuessSetup:
    """Accepted initial guess for an x-block Krylov solve."""

    x0_full: jnp.ndarray | None
    messages: tuple[tuple[int, str], ...]

def build_xblock_krylov_matvec_setup(
    *,
    op: object,
    rhs: jnp.ndarray,
    xblock_use_active_dof: bool,
    active_idx: jnp.ndarray | None,
    full_to_active: jnp.ndarray | None,
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    operator_matvec: ArrayFn,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    env: Mapping[str, str] | None = None,
    progress_every: int | None = None,
    mv_count: MatvecCounter | None = None,
    progress_label: str = "xblock_sparse_pc_gmres",
    emit_active_message: bool = True,
) -> XBlockKrylovMatvecSetup:
    """Build reduced/full matvec closures and progress accounting."""

    if progress_every is None:
        progress_every_env = _env_value(env, "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY")
        try:
            progress_every = int(progress_every_env) if progress_every_env else 25
        except ValueError:
            progress_every = 25
    progress_every = max(0, int(progress_every))
    counter = mv_count if mv_count is not None else MatvecCounter(0)

    linear_size = int(op.total_size)
    active_idx_np: np.ndarray | None = None
    xblock_rhs = rhs
    messages: list[tuple[int, str]] = []
    if bool(xblock_use_active_dof):
        if active_idx is None or full_to_active is None:
            raise ValueError("x-block active-DOF matvec setup requires active_idx and full_to_active maps.")
        active_idx_np = np.asarray(jax.device_get(active_idx), dtype=np.int32)
        linear_size = int(active_idx_np.shape[0])
        xblock_rhs = rhs[active_idx]
        if bool(emit_active_message):
            messages.append(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: "
                    f"{progress_label} active-DOF reduction enabled "
                    f"(size={int(linear_size)}/{int(op.total_size)})",
                )
            )

    def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        if not bool(xblock_use_active_dof):
            return v_full
        assert active_idx is not None
        return reduce_full_with_indices(v_full, active_idx)

    def expand_reduced(v_vec: jnp.ndarray) -> jnp.ndarray:
        if not bool(xblock_use_active_dof):
            return v_vec
        assert full_to_active is not None
        return expand_reduced_with_map(v_vec, full_to_active)

    def matvec_no_count(v: jnp.ndarray) -> jnp.ndarray:
        x_full = expand_reduced(jnp.asarray(v, dtype=rhs.dtype))
        y_full = operator_matvec(x_full)
        return reduce_full(y_full)

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        counter.increment()
        if emit is not None and progress_every > 0 and int(counter) % progress_every == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: "
                f"{progress_label} matvecs={int(counter)} elapsed_s={float(elapsed_s()):.3f}",
            )
        return matvec_no_count(v)

    return XBlockKrylovMatvecSetup(
        progress_every=int(progress_every),
        mv_count=counter,
        xblock_linear_size=int(linear_size),
        xblock_active_idx_np=active_idx_np,
        xblock_rhs=jnp.asarray(xblock_rhs, dtype=rhs.dtype),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        matvec_no_count=matvec_no_count,
        matvec=matvec,
        messages=tuple(messages),
    )

def prepare_xblock_initial_guess(
    *,
    x0: object | None,
    xblock_rhs: jnp.ndarray,
    full_rhs: jnp.ndarray,
    xblock_use_active_dof: bool,
    reduce_full: ArrayFn,
) -> XBlockInitialGuessSetup:
    """Accept a user-provided initial guess if its shape matches the active x-block solve."""

    if x0 is None:
        return XBlockInitialGuessSetup(x0_full=None, messages=())
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    xblock_shape = tuple(xblock_rhs.shape)
    full_shape = tuple(full_rhs.shape)
    if x0_arr.shape == xblock_rhs.shape:
        return XBlockInitialGuessSetup(x0_full=x0_arr, messages=())
    if bool(xblock_use_active_dof) and x0_arr.shape == full_rhs.shape:
        return XBlockInitialGuessSetup(
            x0_full=jnp.asarray(reduce_full(x0_arr), dtype=jnp.float64),
            messages=(),
        )
    expected = f"expected={xblock_shape}" + (f" or {full_shape}" if bool(xblock_use_active_dof) else "")
    return XBlockInitialGuessSetup(
        x0_full=None,
        messages=(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"ignoring incompatible x0 shape={tuple(x0_arr.shape)} {expected}",
            ),
        ),
    )



@dataclass(frozen=True)
class XBlockMomentSchurPolicySetup:
    """Admission and probe policy for x-block constraint moment-Schur correction."""

    default_candidate: bool
    default_blocked_by_compact_factors: bool
    enabled: bool
    rcond: float
    probe_enabled: bool
    probe_min_improvement: float
    messages: tuple[tuple[int, str], ...]

@dataclass(frozen=True)
class XBlockGlobalCouplingPolicySetup:
    """Admission and build parameters for x-block global-coupling correction."""

    enabled: bool
    should_build: bool
    use_device_builder: bool
    mode: str
    max_directions: int
    fsavg_lmax: int
    angular_lmax: int
    max_extra_units: int
    rcond: float
    include_rhs: bool
    setup_max_s: float


__all__ = (
    "XBlockMomentSchurPolicySetup",
    "XBlockGlobalCouplingPolicySetup",
    "MatvecCounter",
    "XBlockKrylovMatvecSetup",
    "XBlockInitialGuessSetup",
    "build_xblock_krylov_matvec_setup",
    "prepare_xblock_initial_guess",
    "FPXBlockGlobalCorrectionContext",
    "FPXBlockGlobalCorrectionResult",
    "FPXBlockHighXCorrectionContext",
    "FPXBlockHighXCorrectionResult",
    "SparseSXBlockRescueContext",
    "SparseSXBlockRescueResult",
    "SparseXBlockExplicitSeedContext",
    "SparseXBlockExplicitSeedResult",
    "SparseXBlockRescueAcceptanceContext",
    "SparseXBlockRescueAcceptanceResult",
    "SparseXBlockRescueBuildContext",
    "SparseXBlockRescueBuildResult",
    "SparseXBlockRescueSolveContext",
    "SparseXBlockRescueSolveResult",
    "accept_sparse_xblock_rescue_candidate",
    "apply_sparse_xblock_explicit_seed",
    "build_sparse_xblock_rescue_preconditioner",
    "run_fp_xblock_global_correction_stage",
    "run_fp_xblock_highx_residual_correction_stage",
    "run_sparse_sxblock_rescue_stage",
    "run_sparse_xblock_rescue_solve_stage",
)
