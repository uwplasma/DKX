"""X-block sparse rescue stages for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, fields
import os
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..diagnostics import (
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_qi_deflated_preconditioner_diagnostics,
    xblock_qi_device_preconditioner_diagnostics,
    xblock_qi_seed_preconditioner_diagnostics,
    xblock_sparse_pc_result_diagnostics_from_driver_state,
    xblock_side_probe_diagnostics,
)
from ..policies import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
)
from ..solver_diagnostics import (
    build_rhs1_xblock_correction_metadata_from_driver_state,
    prepare_cached_qi_correction_basis,
)
from .finalization import (
    SparsePCGMRESFinalPayload,
    SparsePCPostMinresUpdateContext,
    apply_sparse_pc_post_minres_if_needed,
)
from .policy import _env_bool, _env_float, _env_int, _env_value
from .qi import (
    build_xblock_qi_stage_pipeline_context,
    resolve_xblock_qi_device_operator_reuse_setup,
    run_xblock_qi_preconditioner_pipeline,
)
from ..residual import (
    l2_norm_float as profile_l2_norm_float,
    residual_converged as profile_residual_converged,
    safe_ratio as profile_safe_ratio,
)
from sfincs_jax.solvers.memory_model import (
    bicgstab_work_nbytes,
    gmres_basis_nbytes,
    tfqmr_work_nbytes,
)
from ....solver import GMRESSolveResult


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


def _unique_state_keys(*groups: Sequence[str]) -> tuple[str, ...]:
    """Return keys in first-seen order for diagnostic state contracts."""

    seen: set[str] = set()
    ordered: list[str] = []
    for group in groups:
        for key in group:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return tuple(ordered)


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


@dataclass(frozen=True)
class XBlockSparsePCSetup:
    """Setup controls for RHSMode=1 x-block sparse-PC solves."""

    xblock_drop_tol: float
    xblock_drop_rel: float
    xblock_ilu_drop_tol: float
    xblock_fill_factor: float
    xblock_lower_fill_mode: str
    xblock_lower_fill_ignored_env: bool
    xblock_preconditioner_xi: int
    force_assembled_host_fp: bool
    xblock_assembled_host_fp: bool
    xblock_krylov_env_requested: str
    xblock_krylov_env: str
    xblock_krylov_requested: str
    xblock_device_fgmres_requested: bool
    xblock_device_gmres_requested: bool
    xblock_device_bicgstab_requested: bool
    xblock_device_tfqmr_requested: bool
    xblock_device_krylov_requested: bool
    xblock_device_host_fallback_decision: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: bool
    qi_device_preconditioner_requested_for_fallback: bool
    qi_device_matrix_free_requested_for_fallback: bool
    qi_device_use_in_krylov_requested_for_fallback: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockSparsePCSidePolicySetup:
    """JAX-factor and side-preconditioner policy for x-block sparse-PC solves."""

    xblock_jax_factors_env: str
    xblock_jax_factors_requested: bool
    xblock_jax_factors: bool
    xblock_jax_factor_format: str
    xblock_jax_factor_apply: str
    xblock_device_krylov_forced_jax_factors: bool
    full_fp_3d_pc: bool
    side_env: str
    precondition_side: str
    xblock_default_right_pc: bool
    xblock_krylov_method: str
    xblock_device_fgmres_forced_right_pc: bool
    pc_restart: int
    xblock_default_restart_capped: bool
    messages: tuple[tuple[int, str], ...]




@dataclass(frozen=True)
class XBlockSparsePCBranchSetup:
    """Combined x-block sparse-PC branch setup before factor construction."""

    xblock_drop_tol: float
    xblock_drop_rel: float
    xblock_ilu_drop_tol: float
    xblock_fill_factor: float
    xblock_lower_fill_mode: str
    xblock_lower_fill_ignored_env: bool
    xblock_preconditioner_xi: int
    force_assembled_host_fp: bool
    xblock_assembled_host_fp: bool
    xblock_krylov_env_requested: str
    xblock_krylov_env: str
    xblock_krylov_requested: str
    xblock_device_fgmres_requested: bool
    xblock_device_gmres_requested: bool
    xblock_device_bicgstab_requested: bool
    xblock_device_tfqmr_requested: bool
    xblock_device_krylov_requested: bool
    xblock_device_host_fallback_decision: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: bool
    qi_device_preconditioner_requested_for_fallback: bool
    qi_device_matrix_free_requested_for_fallback: bool
    qi_device_use_in_krylov_requested_for_fallback: bool
    xblock_jax_factors: bool
    xblock_jax_factor_format: str
    xblock_jax_factor_apply: str
    xblock_device_krylov_forced_jax_factors: bool
    full_fp_3d_pc: bool
    side_env: str
    precondition_side: str
    xblock_default_right_pc: bool
    xblock_krylov_method: str
    xblock_device_fgmres_forced_right_pc: bool
    pc_restart: int
    xblock_default_restart_capped: bool
    xblock_qi_device_operator_reuse_decision: object
    xblock_qi_device_operator_reuse_skip_factors: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockLocalPreconditionerBuildResult:
    """Local x-block preconditioner and timing metadata."""

    preconditioner: ArrayFn
    factor_s: float
    built: bool






@dataclass(frozen=True)
class XBlockAssembledEquilibrationSetup:
    """Row/column equilibration state for an assembled x-block operator."""

    row_enabled: bool
    row_built: bool
    row_metadata: dict[str, object]
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_enabled: bool
    col_built: bool
    col_metadata: dict[str, object]
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None
    messages: tuple[tuple[int, str], ...]


class XBlockAssembledPreflightMemoryError(MemoryError):
    """Preflight rejection that carries metadata for solver diagnostics."""

    def __init__(self, message: str, metadata: Mapping[str, object]) -> None:
        super().__init__(message)
        self.metadata = dict(metadata)


XBlockAssembledPreflightError = XBlockAssembledPreflightMemoryError


@dataclass(frozen=True)
class XBlockAssembledOperatorPreflightSetup:
    """Memory-budget and structural-pattern preflight for assembled x-block operators."""

    csr_max_mb: float
    drop_tol: float
    device_enabled: bool
    device_required: bool
    max_colors: int
    csr_cap_nbytes: int
    pattern: object
    summary: object
    metadata: dict[str, object]


@dataclass(frozen=True)
class XBlockAssembledDeviceSetup:
    """Optional device-resident CSR operator setup for assembled x-block matvecs."""

    device_operator: object | None
    device_resident: bool
    validation_errors: tuple[float, ...]
    error: str | None
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockAssembledMatvecSetup:
    """Matvec closure for an assembled x-block operator."""

    matvec: ArrayFn
    location: str


@dataclass(frozen=True)
class XBlockAssembledOperatorBuildResult:
    """Optional assembled x-block operator build state."""

    matvec: ArrayFn
    built: bool
    device_resident: bool
    metadata: dict[str, object]
    device_operator: object | None
    pc_factor_increment_s: float
    row_enabled: bool
    row_built: bool
    row_metadata: dict[str, object]
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_enabled: bool
    col_built: bool
    col_metadata: dict[str, object]
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None




@dataclass(frozen=True)
class XBlockMomentSchurProbeResult:
    """Decision from probing a moment-Schur seed against the true residual."""

    used: bool
    reason: str
    residual_before: float
    residual_after: float
    improvement_ratio: float
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockTwoLevelPolicySetup:
    """Admission and build parameters for x-block two-level correction."""

    enabled: bool
    should_build: bool
    mode: str
    max_directions: int
    fsavg_lmax: int
    max_extra_units: int
    rcond: float
    include_rhs: bool




@dataclass(frozen=True)
class XBlockMomentSchurStageContext:
    """Dependencies for optional primary x-block moment-Schur setup."""

    op: object
    base_preconditioner: ArrayFn
    reduce_full: ArrayFn | None
    expand_reduced: ArrayFn | None
    policy: XBlockMomentSchurPolicySetup
    precondition_side: str
    rhs: jnp.ndarray
    matvec_no_count: ArrayFn
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]


@dataclass(frozen=True)
class XBlockMomentSchurStageResult:
    """Result from optional primary x-block moment-Schur setup."""

    preconditioner: ArrayFn
    built: bool
    used: bool
    reason: str | None
    metadata: dict[str, object]
    stats: dict[str, int]
    probe_residual_before: float | None
    probe_residual_after: float | None
    probe_improvement_ratio: float | None
    setup_s: float


@dataclass(frozen=True)
class XBlockTwoLevelStageContext:
    """Dependencies for optional primary x-block two-level setup."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockTwoLevelPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None


@dataclass(frozen=True)
class XBlockTwoLevelStageResult:
    """Result from optional primary x-block two-level setup."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float


@dataclass(frozen=True)
class XBlockGlobalCouplingStageContext:
    """Dependencies for optional primary x-block global-coupling setup."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockGlobalCouplingPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    host_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None
    device_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]] | None = None


@dataclass(frozen=True)
class XBlockGlobalCouplingStageResult:
    """Result from optional primary x-block global-coupling setup."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float






























@dataclass(frozen=True)
class XBlockSeedPolicySetup:
    """Initial preconditioner seed controls for x-block Krylov solves."""

    initial_seed_enabled: bool
    moment_schur_seed_enabled: bool








@dataclass(frozen=True)
class XBlockSparsePCBranchContext:
    """Solve-local state and callbacks for the x-block sparse-PC branch."""

    _apply_device_subspace_residual_equation_correction: object
    _apply_preconditioned_minres_correction: object
    _apply_subspace_minres_correction: object
    _rhs1_xblock_post_coarse_directions: object
    _build_rhs1_xblock_constraint1_moment_schur_preconditioner: object
    _build_rhsmode1_xblock_tz_sparse_preconditioner: object
    _read_rhs1_post_solve_correction_policy: object
    _read_rhs1_probe_coarse_policy: object
    _rhs1_bool_env: object
    _rhs1_float_env: object
    _rhs1_xblock_fallback_initial_guess: object
    _rhs1_xblock_policy: object
    _rhsmode1_fp_xblock_assembled_host_allowed: object
    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly: object
    active_idx_jnp: object
    active_size: object
    apply_v3_full_system_operator_cached: object
    atol: object
    bicgstab_solve_with_history_scipy: object
    bicgstab_solve_with_residual: object
    build_operator_from_pattern: object
    device_csr_from_matrix: object
    emit: object
    estimate_v3_full_system_conservative_sparsity_summary: object
    expand_reduced_with_map: object
    fgmres_cycle_jit_solve_with_residual: object
    fgmres_solve_with_residual: object
    fgmres_solve_with_residual_jit: object
    full_to_active_jnp: object
    gcrotmk_solve_with_history_scipy: object
    gmres_solve_with_history_scipy: object
    include_electric_field_xi_sparse_pc: object
    include_xdot_sparse_pc: object
    lgmres_solve_with_history_scipy: object
    op: object
    pc_maxiter: object
    pc_restart: object
    pc_restart_env: object
    preconditioner_species: object
    preconditioner_xi: object
    reduce_full_with_indices: object
    resolve_rhs1_xblock_sparse_pc_policy: object
    rhs: object
    rhs1_l2_norm_float: object
    rhs1_residual_target: object
    rhs1_safe_ratio: object
    sparse_pc_fp_dense_velocity_block: object
    sparse_timer: object
    summarize_v3_sparse_pattern: object
    tfqmr_solve_with_residual: object
    tokamak_fp_er_pc: object
    tol: object
    use_dkes: object
    v3_full_system_conservative_sparsity_pattern: object
    v3_full_system_conservative_sparsity_pattern_for_indices: object
    v3_linear_solve_result_from_payload: object
    validate_device_csr_matvec: object
    x0: object
    xblock_sparse_pc: object
    xblock_use_active_dof: object


def run_xblock_sparse_pc_branch(context: XBlockSparsePCBranchContext):
    """Run the RHSMode=1 x-block sparse-PC GMRES branch outside solve.py."""

    _apply_device_subspace_residual_equation_correction = context._apply_device_subspace_residual_equation_correction
    _apply_preconditioned_minres_correction = context._apply_preconditioned_minres_correction
    _apply_subspace_minres_correction = context._apply_subspace_minres_correction
    _rhs1_xblock_post_coarse_directions = context._rhs1_xblock_post_coarse_directions
    _build_rhs1_xblock_constraint1_moment_schur_preconditioner = context._build_rhs1_xblock_constraint1_moment_schur_preconditioner
    _build_rhsmode1_xblock_tz_sparse_preconditioner = context._build_rhsmode1_xblock_tz_sparse_preconditioner
    _read_rhs1_post_solve_correction_policy = context._read_rhs1_post_solve_correction_policy
    _read_rhs1_probe_coarse_policy = context._read_rhs1_probe_coarse_policy
    _rhs1_bool_env = context._rhs1_bool_env
    _rhs1_float_env = context._rhs1_float_env
    _rhs1_xblock_fallback_initial_guess = context._rhs1_xblock_fallback_initial_guess
    _rhs1_xblock_policy = context._rhs1_xblock_policy
    _rhsmode1_fp_xblock_assembled_host_allowed = context._rhsmode1_fp_xblock_assembled_host_allowed
    _rhsmode1_fp_xblock_species_decoupled_for_host_assembly = context._rhsmode1_fp_xblock_species_decoupled_for_host_assembly
    active_idx_jnp = context.active_idx_jnp
    active_size = context.active_size
    apply_v3_full_system_operator_cached = context.apply_v3_full_system_operator_cached
    atol = context.atol
    bicgstab_solve_with_history_scipy = context.bicgstab_solve_with_history_scipy
    bicgstab_solve_with_residual = context.bicgstab_solve_with_residual
    build_operator_from_pattern = context.build_operator_from_pattern
    device_csr_from_matrix = context.device_csr_from_matrix
    emit = context.emit
    estimate_v3_full_system_conservative_sparsity_summary = context.estimate_v3_full_system_conservative_sparsity_summary
    expand_reduced_with_map = context.expand_reduced_with_map
    fgmres_cycle_jit_solve_with_residual = context.fgmres_cycle_jit_solve_with_residual
    fgmres_solve_with_residual = context.fgmres_solve_with_residual
    fgmres_solve_with_residual_jit = context.fgmres_solve_with_residual_jit
    full_to_active_jnp = context.full_to_active_jnp
    gcrotmk_solve_with_history_scipy = context.gcrotmk_solve_with_history_scipy
    gmres_solve_with_history_scipy = context.gmres_solve_with_history_scipy
    include_electric_field_xi_sparse_pc = context.include_electric_field_xi_sparse_pc
    include_xdot_sparse_pc = context.include_xdot_sparse_pc
    lgmres_solve_with_history_scipy = context.lgmres_solve_with_history_scipy
    op = context.op
    pc_maxiter = context.pc_maxiter
    pc_restart = context.pc_restart
    pc_restart_env = context.pc_restart_env
    preconditioner_species = context.preconditioner_species
    preconditioner_xi = context.preconditioner_xi
    reduce_full_with_indices = context.reduce_full_with_indices
    resolve_rhs1_xblock_sparse_pc_policy = context.resolve_rhs1_xblock_sparse_pc_policy
    rhs = context.rhs
    rhs1_l2_norm_float = context.rhs1_l2_norm_float
    rhs1_residual_target = context.rhs1_residual_target
    rhs1_safe_ratio = context.rhs1_safe_ratio
    sparse_pc_fp_dense_velocity_block = context.sparse_pc_fp_dense_velocity_block
    sparse_timer = context.sparse_timer
    summarize_v3_sparse_pattern = context.summarize_v3_sparse_pattern
    tfqmr_solve_with_residual = context.tfqmr_solve_with_residual
    tokamak_fp_er_pc = context.tokamak_fp_er_pc
    tol = context.tol
    use_dkes = context.use_dkes
    v3_full_system_conservative_sparsity_pattern = context.v3_full_system_conservative_sparsity_pattern
    v3_full_system_conservative_sparsity_pattern_for_indices = context.v3_full_system_conservative_sparsity_pattern_for_indices
    v3_linear_solve_result_from_payload = context.v3_linear_solve_result_from_payload
    validate_device_csr_matvec = context.validate_device_csr_matvec
    x0 = context.x0
    xblock_sparse_pc = context.xblock_sparse_pc
    xblock_use_active_dof = context.xblock_use_active_dof
    if xblock_sparse_pc:
        xblock_branch_setup = resolve_xblock_sparse_pc_branch_setup(
            op=op,
            preconditioner_species=int(preconditioner_species),
            preconditioner_xi=int(preconditioner_xi),
            active_size=int(active_size),
            pc_restart=int(pc_restart),
            pc_restart_env=str(pc_restart_env),
            tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
            use_dkes=bool(use_dkes),
            include_xdot_sparse_pc=bool(include_xdot_sparse_pc),
            include_electric_field_xi_sparse_pc=bool(include_electric_field_xi_sparse_pc),
            lower_fill_mode=_rhs1_xblock_policy.rhs1_xblock_lower_fill_mode,
            species_decoupled_for_host_assembly=_rhsmode1_fp_xblock_species_decoupled_for_host_assembly,
            assembled_host_allowed=_rhsmode1_fp_xblock_assembled_host_allowed,
            krylov_method=_rhs1_xblock_policy.rhs1_xblock_krylov_method,
            device_host_fallback_decision=_rhs1_xblock_policy.rhs1_xblock_device_host_fallback_decision,
            resolve_xblock_policy=resolve_rhs1_xblock_sparse_pc_policy,
            reuse_decision=_rhs1_xblock_policy.rhs1_xblock_qi_device_operator_reuse_decision,
            env=os.environ,
        )
        xblock_drop_tol = float(xblock_branch_setup.xblock_drop_tol)
        xblock_drop_rel = float(xblock_branch_setup.xblock_drop_rel)
        xblock_ilu_drop_tol = float(xblock_branch_setup.xblock_ilu_drop_tol)
        xblock_fill_factor = float(xblock_branch_setup.xblock_fill_factor)
        xblock_lower_fill_mode = str(xblock_branch_setup.xblock_lower_fill_mode)
        xblock_lower_fill_ignored_env = bool(xblock_branch_setup.xblock_lower_fill_ignored_env)
        xblock_preconditioner_xi = int(xblock_branch_setup.xblock_preconditioner_xi)
        force_assembled_host_fp = bool(xblock_branch_setup.force_assembled_host_fp)
        xblock_assembled_host_fp = bool(xblock_branch_setup.xblock_assembled_host_fp)
        xblock_krylov_env_requested = str(xblock_branch_setup.xblock_krylov_env_requested)
        xblock_krylov_env = str(xblock_branch_setup.xblock_krylov_env)
        xblock_krylov_requested = str(xblock_branch_setup.xblock_krylov_requested)
        xblock_device_fgmres_requested = bool(xblock_branch_setup.xblock_device_fgmres_requested)
        xblock_device_gmres_requested = bool(xblock_branch_setup.xblock_device_gmres_requested)
        xblock_device_bicgstab_requested = bool(xblock_branch_setup.xblock_device_bicgstab_requested)
        xblock_device_tfqmr_requested = bool(xblock_branch_setup.xblock_device_tfqmr_requested)
        xblock_device_krylov_requested = bool(xblock_branch_setup.xblock_device_krylov_requested)
        xblock_device_host_fallback_decision = xblock_branch_setup.xblock_device_host_fallback_decision
        xblock_device_host_fallback_auto_disabled_by_qi_device = bool(
            xblock_branch_setup.xblock_device_host_fallback_auto_disabled_by_qi_device
        )
        qi_device_preconditioner_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_preconditioner_requested_for_fallback
        )
        qi_device_matrix_free_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_matrix_free_requested_for_fallback
        )
        qi_device_use_in_krylov_requested_for_fallback = bool(
            xblock_branch_setup.qi_device_use_in_krylov_requested_for_fallback
        )
        xblock_jax_factors = bool(xblock_branch_setup.xblock_jax_factors)
        xblock_jax_factor_format = str(xblock_branch_setup.xblock_jax_factor_format)
        xblock_jax_factor_apply = str(xblock_branch_setup.xblock_jax_factor_apply)
        xblock_device_krylov_forced_jax_factors = bool(
            xblock_branch_setup.xblock_device_krylov_forced_jax_factors
        )
        full_fp_3d_pc = bool(xblock_branch_setup.full_fp_3d_pc)
        side_env = str(xblock_branch_setup.side_env)
        precondition_side = str(xblock_branch_setup.precondition_side)
        xblock_default_right_pc = bool(xblock_branch_setup.xblock_default_right_pc)
        xblock_krylov_method = str(xblock_branch_setup.xblock_krylov_method)
        xblock_device_fgmres_forced_right_pc = bool(
            xblock_branch_setup.xblock_device_fgmres_forced_right_pc
        )
        pc_restart = int(xblock_branch_setup.pc_restart)
        xblock_default_restart_capped = bool(xblock_branch_setup.xblock_default_restart_capped)
        xblock_qi_device_operator_reuse_decision = xblock_branch_setup.xblock_qi_device_operator_reuse_decision
        xblock_qi_device_operator_reuse_skip_factors = bool(
            xblock_branch_setup.xblock_qi_device_operator_reuse_skip_factors
        )
        if emit is not None:
            for level, message in xblock_branch_setup.messages:
                emit(int(level), str(message))
        xblock_local_preconditioner = build_xblock_local_preconditioner(
            skip_factors=bool(xblock_qi_device_operator_reuse_skip_factors),
            elapsed_s=sparse_timer.elapsed_s,
            build_preconditioner=_build_rhsmode1_xblock_tz_sparse_preconditioner,
            op=op,
            build_jax_factors=bool(xblock_jax_factors),
            preconditioner_species=preconditioner_species,
            preconditioner_xi=xblock_preconditioner_xi,
            drop_tol=xblock_drop_tol,
            drop_rel=xblock_drop_rel,
            ilu_drop_tol=xblock_ilu_drop_tol,
            fill_factor=xblock_fill_factor,
            force_assembled_host_fp=bool(force_assembled_host_fp),
            emit=emit,
        )
        precond_xblock = xblock_local_preconditioner.preconditioner
        pc_factor_s = float(xblock_local_preconditioner.factor_s)
        xblock_preconditioner_built = bool(xblock_local_preconditioner.built)
        setup_s = sparse_timer.elapsed_s()
        xblock_matvec_setup = build_xblock_krylov_matvec_setup(
            op=op,
            rhs=rhs,
            xblock_use_active_dof=bool(xblock_use_active_dof),
            active_idx=active_idx_jnp,
            full_to_active=full_to_active_jnp,
            reduce_full_with_indices=reduce_full_with_indices,
            expand_reduced_with_map=expand_reduced_with_map,
            operator_matvec=lambda x_full: apply_v3_full_system_operator_cached(op, x_full),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            env=os.environ,
        )
        progress_every = int(xblock_matvec_setup.progress_every)
        mv_count = xblock_matvec_setup.mv_count
        xblock_linear_size = int(xblock_matvec_setup.xblock_linear_size)
        xblock_active_idx_np = xblock_matvec_setup.xblock_active_idx_np
        xblock_rhs = xblock_matvec_setup.xblock_rhs
        _xblock_reduce_full = xblock_matvec_setup.reduce_full
        _xblock_expand_reduced = xblock_matvec_setup.expand_reduced
        _mv_true_no_count = xblock_matvec_setup.matvec_no_count
        _mv_true = xblock_matvec_setup.matvec
        if emit is not None:
            for level, message in xblock_matvec_setup.messages:
                emit(int(level), str(message))

        _mv_xblock_krylov = _mv_true

        def _precond_xblock_krylov_base(v: jnp.ndarray) -> jnp.ndarray:
            if not xblock_use_active_dof:
                return precond_xblock(v)
            z_full = precond_xblock(_xblock_expand_reduced(jnp.asarray(v, dtype=rhs.dtype)))
            return _xblock_reduce_full(z_full)

        assembled_operator_enabled = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR",
            default=False,
        )
        assembled_operator = build_xblock_assembled_operator_if_requested(
            enabled=bool(assembled_operator_enabled),
            op=op,
            rhs_dtype=rhs.dtype,
            xblock_active_idx_np=xblock_active_idx_np,
            sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
            xblock_krylov_method=str(xblock_krylov_method),
            xblock_linear_size=int(xblock_linear_size),
            true_matvec_no_count=_mv_true_no_count,
            default_matvec=_mv_xblock_krylov,
            mv_count=mv_count,
            progress_every=int(progress_every),
            elapsed_s=sparse_timer.elapsed_s,
            emit=emit,
            estimate_summary=estimate_v3_full_system_conservative_sparsity_summary,
            full_pattern=v3_full_system_conservative_sparsity_pattern,
            active_pattern=v3_full_system_conservative_sparsity_pattern_for_indices,
            summarize_pattern=summarize_v3_sparse_pattern,
            build_operator_from_pattern=build_operator_from_pattern,
            device_csr_from_matrix=device_csr_from_matrix,
            validate_device_csr_matvec=validate_device_csr_matvec,
            finalize_metadata=finalize_xblock_assembled_operator_metadata,
            backend=str(jax.default_backend()),
            env=os.environ,
        )
        _mv_xblock_krylov = assembled_operator.matvec
        assembled_operator_built = bool(assembled_operator.built)
        assembled_operator_device_resident = bool(assembled_operator.device_resident)
        assembled_operator_metadata = dict(assembled_operator.metadata)
        assembled_device_operator = assembled_operator.device_operator
        pc_factor_s += float(assembled_operator.pc_factor_increment_s)
        xblock_row_equilibration_enabled = bool(assembled_operator.row_enabled)
        xblock_row_equilibration_built = bool(assembled_operator.row_built)
        xblock_row_equilibration_metadata = dict(assembled_operator.row_metadata)
        xblock_row_scale_jnp = assembled_operator.row_scale
        xblock_inv_row_scale_jnp = assembled_operator.inv_row_scale
        xblock_col_equilibration_enabled = bool(assembled_operator.col_enabled)
        xblock_col_equilibration_built = bool(assembled_operator.col_built)
        xblock_col_equilibration_metadata = dict(assembled_operator.col_metadata)
        xblock_col_scale_jnp = assembled_operator.col_scale
        xblock_inv_col_scale_jnp = assembled_operator.inv_col_scale

        precond_xblock_krylov = _precond_xblock_krylov_base
        moment_schur_policy = resolve_xblock_moment_schur_policy_setup(
            op=op,
            xblock_krylov_method=str(xblock_krylov_method),
            xblock_jax_factors=bool(xblock_jax_factors),
            xblock_jax_factor_format=str(xblock_jax_factor_format),
            precondition_side=str(precondition_side),
            env=os.environ,
        )
        moment_schur_default_candidate = bool(moment_schur_policy.default_candidate)
        moment_schur_default_blocked_by_compact_factors = bool(
            moment_schur_policy.default_blocked_by_compact_factors
        )
        moment_schur_enabled = bool(moment_schur_policy.enabled)
        moment_schur_stage = apply_xblock_moment_schur_stage(
            context=XBlockMomentSchurStageContext(
                op=op,
                base_preconditioner=precond_xblock_krylov,
                reduce_full=_xblock_reduce_full if xblock_use_active_dof else None,
                expand_reduced=_xblock_expand_reduced if xblock_use_active_dof else None,
                policy=moment_schur_policy,
                precondition_side=str(precondition_side),
                rhs=xblock_rhs,
                matvec_no_count=_mv_true_no_count,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                builder=_build_rhs1_xblock_constraint1_moment_schur_preconditioner,
            )
        )
        precond_xblock_krylov = moment_schur_stage.preconditioner
        moment_schur_built = bool(moment_schur_stage.built)
        moment_schur_used = bool(moment_schur_stage.used)
        moment_schur_reason = moment_schur_stage.reason
        moment_schur_probe_residual_before = moment_schur_stage.probe_residual_before
        moment_schur_probe_residual_after = moment_schur_stage.probe_residual_after
        moment_schur_probe_improvement_ratio = moment_schur_stage.probe_improvement_ratio
        moment_schur_metadata = moment_schur_stage.metadata
        moment_schur_stats = moment_schur_stage.stats
        pc_factor_s += float(moment_schur_stage.setup_s)

        two_level_policy = resolve_xblock_two_level_policy_setup(
            precondition_side=str(precondition_side),
            env=os.environ,
        )
        two_level_enabled = bool(two_level_policy.enabled)
        two_level_stage = apply_xblock_two_level_stage(
            context=XBlockTwoLevelStageContext(
                op=op,
                rhs=rhs,
                matvec=_mv_xblock_krylov,
                base_preconditioner=precond_xblock_krylov,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                policy=two_level_policy,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        precond_xblock_krylov = two_level_stage.preconditioner
        two_level_built = bool(two_level_stage.built)
        two_level_metadata = two_level_stage.metadata
        two_level_stats = two_level_stage.stats
        pc_factor_s += float(two_level_stage.setup_s)

        global_coupling_policy = resolve_xblock_global_coupling_policy_setup(
            precondition_side=str(precondition_side),
            xblock_krylov_method=str(xblock_krylov_method),
            env=os.environ,
        )
        global_coupling_enabled = bool(global_coupling_policy.enabled)
        global_coupling_stage = apply_xblock_global_coupling_stage(
            context=XBlockGlobalCouplingStageContext(
                op=op,
                rhs=rhs,
                matvec=_mv_xblock_krylov,
                base_preconditioner=precond_xblock_krylov,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                policy=global_coupling_policy,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        precond_xblock_krylov = global_coupling_stage.preconditioner
        global_coupling_built = bool(global_coupling_stage.built)
        global_coupling_metadata = global_coupling_stage.metadata
        global_coupling_stats = global_coupling_stage.stats
        pc_factor_s += float(global_coupling_stage.setup_s)

        setup_s = sparse_timer.elapsed_s()
        x0_setup = prepare_xblock_initial_guess(
            x0=x0,
            xblock_rhs=xblock_rhs,
            full_rhs=rhs,
            xblock_use_active_dof=bool(xblock_use_active_dof),
            reduce_full=_xblock_reduce_full,
        )
        x0_full = x0_setup.x0_full
        for level, message in x0_setup.messages:
            if emit is not None:
                emit(level, message)
        xblock_initial_seed_used = False
        xblock_initial_seed_residual_norm: float | None = None
        xblock_initial_seed_residual_ratio: float | None = None
        seed_policy = resolve_xblock_seed_policy_setup(
            moment_schur_used=bool(moment_schur_used),
            env=os.environ,
        )
        seed_enabled = bool(seed_policy.initial_seed_enabled)
        if x0_full is None and seed_enabled:
            try:
                seed_vec = jnp.asarray(precond_xblock_krylov(xblock_rhs), dtype=jnp.float64)
                if seed_vec.shape == xblock_rhs.shape and bool(jnp.all(jnp.isfinite(seed_vec))):
                    seed_residual = xblock_rhs - _mv_true(seed_vec)
                    seed_residual_norm = rhs1_l2_norm_float(seed_residual)
                    rhs_norm_float = rhs1_l2_norm_float(xblock_rhs)
                    xblock_initial_seed_residual_norm = float(seed_residual_norm)
                    xblock_initial_seed_residual_ratio = rhs1_safe_ratio(
                        seed_residual_norm,
                        rhs_norm_float,
                    )
                    if np.isfinite(seed_residual_norm) and seed_residual_norm < rhs_norm_float:
                        x0_full = seed_vec
                        xblock_initial_seed_used = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                f"initial x-block seed residual={seed_residual_norm:.6e} "
                                f"rhs_norm={rhs_norm_float:.6e}",
                            )
                    elif emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"initial x-block seed rejected residual={seed_residual_norm:.6e} "
                            f"rhs_norm={rhs_norm_float:.6e}",
                        )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        f"initial x-block seed failed ({type(exc).__name__}: {exc})",
                    )

        xblock_rhs_norm = rhs1_l2_norm_float(xblock_rhs)
        target_xblock = rhs1_residual_target(
            atol=float(atol),
            tol=float(tol),
            rhs_norm=float(xblock_rhs_norm),
        )
        moment_schur_seed_enabled = bool(seed_policy.moment_schur_seed_enabled)
        moment_schur_seed_used = False
        moment_schur_seed_residual_norm: float | None = None
        moment_schur_seed_residual_ratio: float | None = None
        if moment_schur_seed_enabled and moment_schur_built:
            try:
                seed_vec = jnp.asarray(precond_xblock_krylov(xblock_rhs), dtype=jnp.float64)
                if seed_vec.shape == xblock_rhs.shape and bool(jnp.all(jnp.isfinite(seed_vec))):
                    seed_residual = xblock_rhs - jnp.asarray(_mv_true_no_count(seed_vec), dtype=jnp.float64)
                    seed_residual_norm = rhs1_l2_norm_float(seed_residual)
                    moment_schur_seed_residual_norm = float(seed_residual_norm)
                    moment_schur_seed_residual_ratio = rhs1_safe_ratio(
                        seed_residual_norm,
                        target_xblock,
                    )
                    incumbent_norm = float(xblock_rhs_norm)
                    if x0_full is not None:
                        incumbent_residual = xblock_rhs - jnp.asarray(
                            _mv_true_no_count(jnp.asarray(x0_full, dtype=jnp.float64)),
                            dtype=jnp.float64,
                        )
                        incumbent_norm = rhs1_l2_norm_float(incumbent_residual)
                    if np.isfinite(seed_residual_norm) and float(seed_residual_norm) < float(incumbent_norm):
                        x0_full = seed_vec
                        moment_schur_seed_used = True
                        if emit is not None:
                            emit(
                                0,
                                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                                f"constraint1 moment-Schur seed residual={seed_residual_norm:.6e} "
                                f"rhs_norm={float(xblock_rhs_norm):.6e}",
                            )
                    elif emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"constraint1 moment-Schur seed rejected residual={seed_residual_norm:.6e} "
                            f"incumbent={float(incumbent_norm):.6e}",
                        )
            except Exception as exc:  # noqa: BLE001
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        f"constraint1 moment-Schur seed failed ({type(exc).__name__}: {exc})",
                    )
        qi_pipeline = run_xblock_qi_preconditioner_pipeline(
            build_xblock_qi_stage_pipeline_context(
                op=op,
                rhs=rhs,
                x0_full=x0_full,
                xblock_rhs=xblock_rhs,
                xblock_rhs_norm=float(xblock_rhs_norm),
                base_preconditioner=precond_xblock_krylov,
                matvec=_mv_xblock_krylov,
                true_matvec_no_count=_mv_true_no_count,
                direction_projector=(
                    _xblock_reduce_full if xblock_use_active_dof else None
                ),
                active_dof=bool(xblock_use_active_dof),
                linear_size=int(xblock_linear_size),
                host_fallback_used=bool(xblock_device_host_fallback_decision.used),
                precondition_side=str(precondition_side),
                assembled_device_operator=assembled_device_operator,
                assembled_operator_metadata=assembled_operator_metadata,
                assembled_operator_enabled=bool(assembled_operator_enabled),
                assembled_operator_built=bool(assembled_operator_built),
                assembled_operator_device_resident=bool(
                    assembled_operator_device_resident
                ),
                assembled_operator_device_error=assembled_operator_metadata.get(
                    "device_error"
                ),
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
                env=os.environ,
                reduce_full=_xblock_reduce_full,
            )
        )
        precond_xblock_krylov = qi_pipeline.preconditioner
        x0_full = qi_pipeline.x0_full
        qi_device_state_for_augmented_krylov = (
            qi_pipeline.qi_device_state_for_augmented_krylov
        )
        qi_device_augmented_seed_basis_for_krylov = (
            qi_pipeline.qi_device_augmented_seed_basis_for_krylov
        )
        qi_device_augmented_seed_action_for_krylov = (
            qi_pipeline.qi_device_augmented_seed_action_for_krylov
        )
        qi_device_augmented_seed_available = (
            qi_pipeline.qi_device_augmented_seed_available
        )
        qi_device_augmented_seed_used = qi_pipeline.qi_device_augmented_seed_used
        qi_device_augmented_seed_rank = qi_pipeline.qi_device_augmented_seed_rank
        qi_device_preconditioner_metadata = (
            qi_pipeline.qi_device_preconditioner_metadata
        )
        pc_factor_s += float(qi_pipeline.pc_factor_s)
        xblock_side_probe_controls = _rhs1_xblock_policy.rhs1_xblock_side_probe_controls_from_env(
            env=os.environ,
            explicit_side_env_value=side_env,
            full_fp_3d_pc=bool(full_fp_3d_pc),
            active_size=int(active_size),
            krylov_method=str(xblock_krylov_method),
            precondition_side=str(precondition_side),
            pc_restart=int(pc_restart),
            pc_maxiter=int(pc_maxiter),
            backend=str(jax.default_backend()),
            krylov_env_value=xblock_krylov_env,
            device_host_fallback_used=bool(xblock_device_host_fallback_decision.used),
        )
        xblock_side_probe_stage = apply_xblock_side_probe_stage(
            XBlockSideProbeStageContext(
                controls=xblock_side_probe_controls,
                precondition_side=str(precondition_side),
                krylov_method=str(xblock_krylov_method),
                pc_maxiter=pc_maxiter,
                side_env=str(side_env),
                global_coupling_built=bool(global_coupling_built),
                matvec=_mv_xblock_krylov,
                true_matvec_no_count=_mv_true_no_count,
                rhs=xblock_rhs,
                rhs_norm=float(xblock_rhs_norm),
                target=float(target_xblock),
                preconditioner=precond_xblock_krylov,
                x0=x0_full,
                tol=float(tol),
                atol=float(atol),
                elapsed_s=sparse_timer.elapsed_s,
                matvec_count=lambda: int(mv_count),
                emit=emit,
                gmres_solver=gmres_solve_with_history_scipy,
            )
        )
        x0_full = xblock_side_probe_stage.x0
        precondition_side = xblock_side_probe_stage.precondition_side
        xblock_krylov_method = xblock_side_probe_stage.krylov_method
        pc_maxiter = xblock_side_probe_stage.pc_maxiter
        xblock_side_probe_enabled = bool(xblock_side_probe_stage.enabled)
        xblock_side_probe_used = bool(xblock_side_probe_stage.used)
        xblock_side_probe_switched = bool(xblock_side_probe_stage.switched)
        xblock_side_probe_initial_side = xblock_side_probe_stage.initial_side
        xblock_side_probe_selected_side = xblock_side_probe_stage.selected_side
        xblock_side_probe_initial_method = xblock_side_probe_stage.initial_method
        xblock_side_probe_selected_method = xblock_side_probe_stage.selected_method
        xblock_side_probe_lgmres_rescue = bool(xblock_side_probe_stage.lgmres_rescue)
        xblock_lgmres_rescue_maxiter_capped = bool(
            xblock_side_probe_stage.lgmres_rescue_maxiter_capped
        )
        xblock_lgmres_rescue_outer_k = xblock_side_probe_stage.lgmres_rescue_outer_k
        xblock_side_probe_residual_norm = xblock_side_probe_stage.residual_norm
        xblock_side_probe_residual_ratio = xblock_side_probe_stage.residual_ratio
        xblock_side_probe_iterations = int(xblock_side_probe_stage.iterations)
        xblock_side_probe_matvecs = int(xblock_side_probe_stage.matvecs)
        xblock_side_probe_s = float(xblock_side_probe_stage.elapsed_s)
        xblock_side_probe_switch_suppressed_by_global_coupling = bool(
            xblock_side_probe_stage.switch_suppressed_by_global_coupling
        )
        xblock_side_probe_switch_suppressed_by_explicit_side = bool(
            xblock_side_probe_stage.switch_suppressed_by_explicit_side
        )
        xblock_side_probe_physical_seed_preserved_after_switch = bool(
            xblock_side_probe_stage.physical_seed_preserved_after_switch
        )
        xblock_side_probe_seed_used = bool(xblock_side_probe_stage.seed_used)
        xblock_side_probe_seed_residual_norm = (
            xblock_side_probe_stage.seed_residual_norm
        )

        if precondition_side != "none":
            if xblock_use_active_dof:

                def _coarse_preconditioner_for_basis(v_full: jnp.ndarray) -> jnp.ndarray:
                    reduced = _xblock_reduce_full(jnp.asarray(v_full, dtype=jnp.float64))
                    return _xblock_expand_reduced(precond_xblock_krylov(reduced))

            else:
                _coarse_preconditioner_for_basis = precond_xblock_krylov
        else:

            def _coarse_preconditioner_for_basis(v_full: jnp.ndarray) -> jnp.ndarray:
                return jnp.asarray(v_full, dtype=jnp.float64)

        def _xblock_coarse_direction_builder(
            residual_vec: jnp.ndarray,
            *,
            include_raw: bool,
            fsavg_lmax: int,
            angular_lmax: int,
            max_extra_units: int,
            max_directions: int,
            include_angular_residual: bool,
        ) -> tuple[tuple[str, jnp.ndarray], ...]:
            residual_for_basis = (
                _xblock_expand_reduced(jnp.asarray(residual_vec, dtype=jnp.float64))
                if xblock_use_active_dof
                else jnp.asarray(residual_vec, dtype=jnp.float64)
            )
            return _rhs1_xblock_post_coarse_directions(
                op=op,
                residual=residual_for_basis,
                preconditioner=_coarse_preconditioner_for_basis,
                direction_projector=_xblock_reduce_full if xblock_use_active_dof else None,
                expected_size=int(xblock_linear_size),
                include_raw=bool(include_raw),
                fsavg_lmax=int(fsavg_lmax),
                angular_lmax=int(angular_lmax),
                max_extra_units=int(max_extra_units),
                max_directions=int(max_directions),
                include_angular_residual=bool(include_angular_residual),
            )

        probe_coarse_policy = _read_rhs1_probe_coarse_policy()
        probe_coarse_stage = apply_xblock_probe_coarse_stage(
            XBlockProbeCoarseStageContext(
                policy=probe_coarse_policy,
                rhs=xblock_rhs,
                x0=x0_full,
                matvec=_mv_true,
                target=float(target_xblock),
                direction_builder=_xblock_coarse_direction_builder,
                correction=_apply_subspace_minres_correction,
                elapsed_s=sparse_timer.elapsed_s,
                emit=emit,
            )
        )
        x0_full = probe_coarse_stage.x0
        probe_coarse_steps_requested = int(probe_coarse_stage.steps_requested)
        probe_coarse_max_directions = int(probe_coarse_stage.max_directions)
        probe_coarse_max_extra_units = int(probe_coarse_stage.max_extra_units)
        probe_coarse_fsavg_lmax = int(probe_coarse_stage.fsavg_lmax)
        probe_coarse_angular_lmax = int(probe_coarse_stage.angular_lmax)
        probe_coarse_include_angular_residual = bool(
            probe_coarse_stage.include_angular_residual
        )
        probe_coarse_include_raw = bool(probe_coarse_stage.include_raw)
        probe_coarse_alpha_clip = float(probe_coarse_stage.alpha_clip)
        probe_coarse_rcond = float(probe_coarse_stage.rcond)
        probe_coarse_min_improvement = float(probe_coarse_stage.min_improvement)
        probe_coarse_s = float(probe_coarse_stage.elapsed_s)
        probe_coarse_history = probe_coarse_stage.history
        probe_coarse_direction_counts = probe_coarse_stage.direction_counts
        probe_coarse_direction_names = probe_coarse_stage.direction_names
        probe_coarse_residual_before = probe_coarse_stage.residual_before
        probe_coarse_residual_after = probe_coarse_stage.residual_after
        probe_coarse_seed_initialized = bool(probe_coarse_stage.seed_initialized)

        preflight_min_improvement = _rhs1_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_MIN_IMPROVEMENT",
            default=0.0,
            minimum=0.0,
        )
        preflight_required = _rhs1_bool_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_REQUIRED",
            default=False,
        )
        preflight_gate = evaluate_xblock_preflight_gate(
            XBlockPreflightGateContext(
                min_improvement=float(preflight_min_improvement),
                required=bool(preflight_required),
                rhs=xblock_rhs,
                rhs_norm=float(xblock_rhs_norm),
                x0=x0_full,
                matvec=_mv_true_no_count,
                target=float(target_xblock),
                emit=emit,
            )
        )
        preflight_residual_norm = preflight_gate.residual_norm
        preflight_improvement = preflight_gate.improvement
        preflight_passed = preflight_gate.passed

        xblock_krylov_controls = resolve_xblock_krylov_control_setup(
            XBlockKrylovControlSetupContext(
                env=os.environ,
                krylov_method=str(xblock_krylov_method),
                pc_restart=int(pc_restart),
                pc_maxiter=pc_maxiter,
                precondition_side=str(precondition_side),
                emit=emit,
            )
        )
        fgmres_block_between_cycles = bool(
            xblock_krylov_controls.fgmres_block_between_cycles
        )
        tfqmr_replacement_interval = int(
            xblock_krylov_controls.tfqmr_replacement_interval
        )
        xblock_device_fgmres_jit = bool(
            xblock_krylov_controls.device_fgmres_jit
        )
        xblock_device_fgmres_jit_mode = (
            xblock_krylov_controls.device_fgmres_jit_mode
        )
        xblock_device_fgmres_jit_outer_k = int(
            xblock_krylov_controls.device_fgmres_jit_outer_k
        )
        qi_device_augmented_krylov_requested = bool(
            xblock_krylov_controls.qi_device_augmented_krylov_requested
        )
        qi_device_augmented_krylov_mode = (
            xblock_krylov_controls.qi_device_augmented_krylov_mode
        )
        solve_matvec = _mv_xblock_krylov
        solve_rhs = xblock_rhs
        solve_preconditioner = precond_xblock_krylov if precondition_side != "none" else None
        solve_x0 = x0_full
        solve_space = prepare_xblock_krylov_solve_space(
            XBlockKrylovSolveSpaceContext(
                matvec=solve_matvec,
                rhs=solve_rhs,
                preconditioner=solve_preconditioner,
                x0=solve_x0,
                precondition_side=str(precondition_side),
                row_equilibration_built=bool(xblock_row_equilibration_built),
                col_equilibration_built=bool(xblock_col_equilibration_built),
                row_scale=xblock_row_scale_jnp,
                inv_row_scale=xblock_inv_row_scale_jnp,
                col_scale=xblock_col_scale_jnp,
                inv_col_scale=xblock_inv_col_scale_jnp,
            )
        )
        solve_matvec = solve_space.matvec
        solve_rhs = solve_space.rhs
        solve_preconditioner = solve_space.preconditioner
        solve_x0 = solve_space.x0
        solve_solution_to_physical = solve_space.solution_to_physical
        if emit is not None and solve_space.transform_label is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"using {solve_space.transform_label}-equilibrated assembled operator for Krylov solve",
            )
        augmentation_basis_for_solve = None
        operator_on_augmentation_for_solve = None
        augmented_krylov_stage = apply_xblock_augmented_krylov_stage(
            XBlockAugmentedKrylovStageContext(
                requested=bool(qi_device_augmented_krylov_requested),
                krylov_method=str(xblock_krylov_method),
                qi_device_state=qi_device_state_for_augmented_krylov,
                seed_available=bool(qi_device_augmented_seed_available),
                seed_rank=int(qi_device_augmented_seed_rank),
                seed_basis=qi_device_augmented_seed_basis_for_krylov,
                seed_operator_on_basis=qi_device_augmented_seed_action_for_krylov,
                seed_used=bool(qi_device_augmented_seed_used),
                row_equilibration_built=bool(xblock_row_equilibration_built),
                col_equilibration_built=bool(xblock_col_equilibration_built),
                row_scale=xblock_row_scale_jnp,
                inv_col_scale=xblock_inv_col_scale_jnp,
                precondition_side=str(precondition_side),
                solve_preconditioner=solve_preconditioner,
                mode=str(qi_device_augmented_krylov_mode),
                metadata=qi_device_preconditioner_metadata,
                emit=emit,
                basis_builder=prepare_xblock_augmented_krylov_basis,
            )
        )
        augmentation_basis_for_solve = augmented_krylov_stage.basis
        operator_on_augmentation_for_solve = (
            augmented_krylov_stage.operator_on_basis
        )
        qi_device_augmented_krylov_used = bool(augmented_krylov_stage.used)
        qi_device_augmented_krylov_rank = int(augmented_krylov_stage.rank)
        qi_device_augmented_krylov_reason = augmented_krylov_stage.reason
        qi_device_augmented_seed_used = bool(augmented_krylov_stage.seed_used)
        qi_device_preconditioner_metadata = augmented_krylov_stage.metadata
        solve_start_s = sparse_timer.elapsed_s()
        progress_callbacks = build_xblock_krylov_progress_callbacks(
            XBlockKrylovProgressCallbacksContext(
                emit=emit,
                elapsed_s=sparse_timer.elapsed_s,
                progress_every=int(progress_every),
            )
        )

        fallback_to_gmres = _rhs1_xblock_policy.rhs1_xblock_fallback_to_gmres_enabled(
            env_value=os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_PC_FALLBACK_GMRES", ""),
            xblock_side_probe_lgmres_rescue=bool(xblock_side_probe_lgmres_rescue),
            xblock_krylov_method=str(xblock_krylov_method),
        )
        krylov_stage = run_xblock_krylov_solve_stage(
            XBlockKrylovSolveStageContext(
                first_attempt=XBlockFirstKrylovAttemptContext(
                    krylov_method=str(xblock_krylov_method),
                    matvec=solve_matvec,
                    rhs=solve_rhs,
                    preconditioner=solve_preconditioner,
                    x0=solve_x0,
                    tol=float(tol),
                    atol=float(atol),
                    restart=int(pc_restart),
                    maxiter=pc_maxiter,
                    precondition_side=str(precondition_side),
                    lgmres_outer_k=xblock_lgmres_rescue_outer_k,
                    fgmres_block_between_cycles=bool(fgmres_block_between_cycles),
                    skip_inactive_work=not bool(two_level_built),
                    device_fgmres_jit=bool(xblock_device_fgmres_jit),
                    device_fgmres_jit_mode=str(xblock_device_fgmres_jit_mode),
                    device_fgmres_jit_outer_k=int(xblock_device_fgmres_jit_outer_k),
                    augmented_krylov_used=bool(qi_device_augmented_krylov_used),
                    augmentation_basis=augmentation_basis_for_solve,
                    operator_on_augmentation=operator_on_augmentation_for_solve,
                    augmentation_mode=str(qi_device_augmented_krylov_mode),
                    tfqmr_replacement_interval=int(tfqmr_replacement_interval),
                    mv_count=int(mv_count),
                    host_progress_callback=progress_callbacks.host_progress_callback,
                    device_cycle_progress_callback=(
                        progress_callbacks.device_cycle_progress_callback
                    ),
                    gmres_solver=gmres_solve_with_history_scipy,
                    lgmres_solver=lgmres_solve_with_history_scipy,
                    gcrotmk_solver=gcrotmk_solve_with_history_scipy,
                    bicgstab_solver=bicgstab_solve_with_history_scipy,
                    fgmres_solver=fgmres_solve_with_residual,
                    fgmres_jit_solver=fgmres_solve_with_residual_jit,
                    fgmres_cycle_jit_solver=fgmres_cycle_jit_solve_with_residual,
                    bicgstab_jax_solver=bicgstab_solve_with_residual,
                    tfqmr_jax_solver=tfqmr_solve_with_residual,
                ),
                solve_start_s=float(solve_start_s),
                side_probe_s=float(xblock_side_probe_s),
                probe_coarse_s=float(probe_coarse_s),
                elapsed_s=sparse_timer.elapsed_s,
                solution_to_physical=solve_solution_to_physical,
                physical_rhs=xblock_rhs,
                physical_matvec=_mv_true,
                target=float(target_xblock),
                rhs_norm=float(xblock_rhs_norm),
                fallback_enabled=bool(fallback_to_gmres),
                progress_callback=progress_callbacks.host_progress_callback,
                emit=emit,
                initial_guess_builder=_rhs1_xblock_fallback_initial_guess,
            )
        )
        candidate_state = krylov_stage.candidate_state
        candidate_krylov_method = str(candidate_state.krylov_method)
        candidate_residual_norm = float(candidate_state.residual_norm)
        candidate_iterations = int(candidate_state.reported_iterations)
        candidate_matvecs = int(candidate_state.reported_matvecs)
        solve_state = krylov_stage.final_state
        xblock_krylov_method = str(solve_state.krylov_method)
        x_solution_np = solve_state.x_solution
        x_physical_np = solve_state.x_physical
        residual_norm_xblock_pc = float(solve_state.residual_norm)
        history = solve_state.history
        solve_s = float(solve_state.solve_s)
        device_krylov_iterations = solve_state.device_iterations
        device_krylov_estimated_matvecs = solve_state.device_estimated_matvecs
        fallback_started_from_candidate = solve_state.fallback_started_from_candidate
        fallback_candidate_improved_rhs = solve_state.fallback_candidate_improved_rhs
        reported_iterations = int(solve_state.reported_iterations)
        reported_matvecs = int(solve_state.reported_matvecs)
        x_np = solve_state.x_physical
        post_completion = complete_xblock_post_krylov_stage(
            XBlockPostKrylovCompletionContext(
                corrections=XBlockPostSolveCorrectionContext(
                    matvec=_mv_true,
                    rhs=xblock_rhs,
                    x=np.asarray(x_np, dtype=np.float64),
                    residual_norm=float(residual_norm_xblock_pc),
                    target=float(target_xblock),
                    solve_s=float(solve_s),
                    preconditioner=precond_xblock_krylov,
                    precondition_side=str(precondition_side),
                    post_solve_policy=_read_rhs1_post_solve_correction_policy(),
                    qi_device_state=qi_device_state_for_augmented_krylov,
                    coarse_direction_builder=_xblock_coarse_direction_builder,
                    emit=emit,
                    elapsed_s=sparse_timer.elapsed_s,
                    minres_correction=_apply_preconditioned_minres_correction,
                    residual_equation_correction=(
                        _apply_device_subspace_residual_equation_correction
                    ),
                    coarse_correction=_apply_subspace_minres_correction,
                ),
                krylov_method=str(xblock_krylov_method),
                elapsed_s=sparse_timer.elapsed_s,
                iterations=int(reported_iterations),
                matvecs=int(reported_matvecs),
                target=float(target_xblock),
                history=history,
            )
        )
        post_corrections = post_completion.corrections
        x_np = np.asarray(post_completion.x, dtype=np.float64)
        residual_norm_xblock_pc = float(post_completion.residual_norm)
        solve_s = float(post_completion.solve_s)
        xblock_final_driver_state = {**qi_pipeline.diagnostic_scope(), **locals()}
        xblock_final_metadata_state = (
            xblock_sparse_pc_final_metadata_state_from_driver_scope(
                xblock_final_driver_state
            )
        )
        xblock_sparse_pc_final_payload = (
            xblock_sparse_pc_final_payload_from_driver_state(
                {
                    **xblock_final_metadata_state,
                    "op": op,
                    "x_np": np.asarray(x_np, dtype=np.float64),
                    "residual_norm_xblock_pc": float(residual_norm_xblock_pc),
                    "target_xblock": float(target_xblock),
                    "xblock_krylov_method": str(xblock_krylov_method),
                    "xblock_linear_size": int(xblock_linear_size),
                    "pc_restart": int(pc_restart),
                },
                expand_reduced=_xblock_expand_reduced,
                post_corrections=post_corrections,
            )
        )
        return v3_linear_solve_result_from_payload(
            op=op,
            rhs=rhs,
            payload=xblock_sparse_pc_final_payload,
        )
    return None


def _xblock_device_flags(method: str) -> tuple[bool, bool, bool, bool, bool]:
    method_s = str(method)
    fgmres = method_s == "fgmres_jax"
    gmres = method_s == "gmres_jax"
    bicgstab = method_s == "bicgstab_jax"
    tfqmr = method_s == "tfqmr_jax"
    return fgmres, gmres, bicgstab, tfqmr, bool(fgmres or gmres or bicgstab or tfqmr)


def resolve_xblock_sparse_pc_setup(
    *,
    op: object,
    preconditioner_species: int,
    preconditioner_xi: int,
    active_size: int,
    lower_fill_mode: Callable[[str], tuple[str, bool]],
    species_decoupled_for_host_assembly: Callable[..., bool],
    assembled_host_allowed: Callable[..., bool],
    krylov_method: Callable[[str], tuple[str, bool]],
    device_host_fallback_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSetup:
    """Resolve x-block sparse-PC setup controls before factor construction."""

    if op.fblock.fp is None or op.fblock.pas is not None:
        raise NotImplementedError("solve_method='xblock_sparse_pc_gmres' currently targets full-FP RHSMode=1 systems.")

    drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL", 0.0)
    drop_rel = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_REL", 1.0e-8)
    ilu_drop_tol = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ILU_DROP_TOL", 1.0e-4)
    fill_factor = _env_float(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR", 10.0)
    lower_fill_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL")
    lower_fill_mode_value, lower_fill_ignored_env = lower_fill_mode(lower_fill_env)

    xblock_preconditioner_xi = int(preconditioner_xi)
    if xblock_preconditioner_xi == 0:
        xblock_preconditioner_xi = 1

    force_assembled_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ASSEMBLED_HOST").lower()
    force_assembled_host_fp = force_assembled_env not in {"0", "false", "f", "no", "off", ".false.", ".f."}
    xblock_assembled_host_fp = bool(
        (
            bool(force_assembled_host_fp)
            and int(op.rhs_mode) == 1
            and (not bool(op.include_phi1))
            and op.fblock.fp is not None
            and op.fblock.pas is None
            and species_decoupled_for_host_assembly(
                op=op,
                preconditioner_species=int(preconditioner_species),
            )
            and int(xblock_preconditioner_xi) == 1
            and (not bool(op.point_at_x0))
        )
        or assembled_host_allowed(
            op=op,
            preconditioner_species=int(preconditioner_species),
            preconditioner_xi=int(xblock_preconditioner_xi),
            use_implicit=False,
        )
    )

    krylov_env_requested = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV").lower()
    krylov_env = str(krylov_env_requested)
    krylov_requested, _unknown = krylov_method(krylov_env)
    (
        device_fgmres,
        device_gmres,
        device_bicgstab,
        device_tfqmr,
        device_krylov,
    ) = _xblock_device_flags(str(krylov_requested))

    fallback_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK")
    fallback_auto_disabled_by_qi_device = False
    qi_device_preconditioner = _env_bool(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER", default=False)
    qi_device_matrix_free = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
        default=False,
    )
    qi_device_use_in_krylov = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV",
        default=False,
    )
    precondition_side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    fallback_env_token = fallback_env.strip().lower().replace("-", "_")
    if (
        bool(device_krylov)
        and bool(qi_device_preconditioner)
        and bool(qi_device_matrix_free)
        and bool(qi_device_use_in_krylov)
        and precondition_side_env != "none"
        and fallback_env_token in {"", "auto", "default"}
    ):
        fallback_env = "off"
        fallback_auto_disabled_by_qi_device = True

    fallback_decision = device_host_fallback_decision(
        env_value=fallback_env,
        requested_krylov_method=str(krylov_requested),
        active_size=int(active_size),
        min_active_size_env_value=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK_MIN_ACTIVE"),
        rhs_mode=int(op.rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
        n_zeta=int(getattr(op, "n_zeta", 1)),
    )
    messages: list[tuple[int, str]] = []
    if bool(fallback_decision.used):
        krylov_env = str(fallback_decision.effective_krylov_env_value)
        krylov_requested, _unknown = krylov_method(krylov_env)
        (
            device_fgmres,
            device_gmres,
            device_bicgstab,
            device_tfqmr,
            _device_krylov_after_fallback,
        ) = _xblock_device_flags(str(krylov_requested))
        device_krylov = False
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "using non-autodiff host x-block fallback for requested device Krylov "
                f"method={fallback_decision.requested_method} "
                f"reason={fallback_decision.reason} "
                f"active_size={int(active_size)}",
            )
        )
    elif bool(fallback_decision.ignored_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK value; "
                f"using auto policy reason={fallback_decision.reason}",
            )
        )
    elif bool(fallback_auto_disabled_by_qi_device):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "automatic non-autodiff host fallback disabled by explicit matrix-free "
                "QI-device Krylov preconditioner request",
            )
        )

    return XBlockSparsePCSetup(
        xblock_drop_tol=float(drop_tol),
        xblock_drop_rel=float(drop_rel),
        xblock_ilu_drop_tol=float(ilu_drop_tol),
        xblock_fill_factor=float(fill_factor),
        xblock_lower_fill_mode=str(lower_fill_mode_value),
        xblock_lower_fill_ignored_env=bool(lower_fill_ignored_env),
        xblock_preconditioner_xi=int(xblock_preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        xblock_assembled_host_fp=bool(xblock_assembled_host_fp),
        xblock_krylov_env_requested=str(krylov_env_requested),
        xblock_krylov_env=str(krylov_env),
        xblock_krylov_requested=str(krylov_requested),
        xblock_device_fgmres_requested=bool(device_fgmres),
        xblock_device_gmres_requested=bool(device_gmres),
        xblock_device_bicgstab_requested=bool(device_bicgstab),
        xblock_device_tfqmr_requested=bool(device_tfqmr),
        xblock_device_krylov_requested=bool(device_krylov),
        xblock_device_host_fallback_decision=fallback_decision,
        xblock_device_host_fallback_auto_disabled_by_qi_device=bool(fallback_auto_disabled_by_qi_device),
        qi_device_preconditioner_requested_for_fallback=bool(qi_device_preconditioner),
        qi_device_matrix_free_requested_for_fallback=bool(qi_device_matrix_free),
        qi_device_use_in_krylov_requested_for_fallback=bool(qi_device_use_in_krylov),
        messages=tuple(messages),
    )


def _normalize_jax_factor_format(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"csr", "compact", "compact_csr", "ragged_csr"}:
        return "csr"
    return "padded"


def _normalize_jax_factor_apply(value: str) -> str:
    token = str(value).strip().lower().replace("-", "_")
    if token in {"diag", "diagonal", "jacobi", "factor_diag", "factor_diagonal"}:
        return "diagonal"
    if token in {"identity", "none", "skip"}:
        return "identity"
    if token in {"upper", "upper_only", "u", "u_only"}:
        return "upper"
    if token in {"lower", "lower_only", "l", "l_only"}:
        return "lower"
    return "exact"


def resolve_xblock_sparse_pc_side_policy_setup(
    *,
    op: object,
    xblock_device_krylov_requested: bool,
    xblock_device_host_fallback_decision: object,
    xblock_krylov_env: str,
    pc_restart: int,
    pc_restart_env: str,
    tokamak_fp_er_pc: bool,
    active_size: int,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    resolve_xblock_policy: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCSidePolicySetup:
    """Resolve x-block factor format and preconditioner-side policy."""

    jax_factors_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS").lower()
    jax_factors_requested = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS",
        default=False,
    )
    fallback_used = bool(getattr(xblock_device_host_fallback_decision, "used", False))
    jax_factors = bool(jax_factors_requested or bool(xblock_device_krylov_requested)) and not fallback_used

    messages: list[tuple[int, str]] = []
    if fallback_used and bool(jax_factors_requested):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "ignoring SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS=1 because "
                "the non-autodiff host fallback requires host sparse factors",
            )
        )

    jax_factor_format = _normalize_jax_factor_format(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT") or "padded"
    )
    jax_factor_apply = _normalize_jax_factor_apply(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY") or "exact"
    )
    device_krylov_forced_jax_factors = bool(
        xblock_device_krylov_requested
        and jax_factors_env not in {"1", "true", "t", "yes", "on", ".true.", ".t."}
    )

    side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    full_fp_3d_right_pc_max_env = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_RIGHT_PC_MAX")
    full_fp_3d_pc = bool(
        op.fblock.fp is not None
        and op.fblock.pas is None
        and int(getattr(op, "n_zeta", 1)) > 1
    )
    xblock_policy = resolve_xblock_policy(
        precondition_side_env_value=side_env,
        krylov_env_value=str(xblock_krylov_env),
        requested_restart=int(pc_restart),
        restart_env_value=str(pc_restart_env),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        active_size=int(active_size),
        full_fp_3d_right_pc_max_env_value=str(full_fp_3d_right_pc_max_env),
        use_dkes=bool(use_dkes),
        include_xdot=bool(include_xdot_sparse_pc),
        include_electric_field_xi=bool(include_electric_field_xi_sparse_pc),
    )
    precondition_side = str(xblock_policy.precondition_side)
    xblock_default_right_pc = bool(xblock_policy.default_right_preconditioned)
    xblock_krylov_method = str(xblock_policy.krylov_method)
    device_fgmres_forced_right_pc = False
    if xblock_krylov_method == "fgmres_jax" and precondition_side == "left":
        precondition_side = "right"
        device_fgmres_forced_right_pc = True
    if bool(xblock_policy.ignored_krylov_env):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"ignoring unknown SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV={xblock_krylov_env!r}",
            )
        )

    return XBlockSparsePCSidePolicySetup(
        xblock_jax_factors_env=str(jax_factors_env),
        xblock_jax_factors_requested=bool(jax_factors_requested),
        xblock_jax_factors=bool(jax_factors),
        xblock_jax_factor_format=str(jax_factor_format),
        xblock_jax_factor_apply=str(jax_factor_apply),
        xblock_device_krylov_forced_jax_factors=bool(device_krylov_forced_jax_factors),
        full_fp_3d_pc=bool(full_fp_3d_pc),
        side_env=str(side_env),
        precondition_side=str(precondition_side),
        xblock_default_right_pc=bool(xblock_default_right_pc),
        xblock_krylov_method=str(xblock_krylov_method),
        xblock_device_fgmres_forced_right_pc=bool(device_fgmres_forced_right_pc),
        pc_restart=int(xblock_policy.gmres_restart),
        xblock_default_restart_capped=bool(xblock_policy.restart_capped),
        messages=tuple(messages),
    )




def resolve_xblock_sparse_pc_branch_setup(
    *,
    op: object,
    preconditioner_species: int,
    preconditioner_xi: int,
    active_size: int,
    pc_restart: int,
    pc_restart_env: str,
    tokamak_fp_er_pc: bool,
    use_dkes: bool,
    include_xdot_sparse_pc: bool,
    include_electric_field_xi_sparse_pc: bool,
    lower_fill_mode: Callable[[str], tuple[str, bool]],
    species_decoupled_for_host_assembly: Callable[..., bool],
    assembled_host_allowed: Callable[..., bool],
    krylov_method: Callable[[str], tuple[str, bool]],
    device_host_fallback_decision: Callable[..., object],
    resolve_xblock_policy: Callable[..., object],
    reuse_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockSparsePCBranchSetup:
    """Resolve x-block sparse-PC branch policy as one typed setup contract."""

    setup = resolve_xblock_sparse_pc_setup(
        op=op,
        preconditioner_species=int(preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        active_size=int(active_size),
        lower_fill_mode=lower_fill_mode,
        species_decoupled_for_host_assembly=species_decoupled_for_host_assembly,
        assembled_host_allowed=assembled_host_allowed,
        krylov_method=krylov_method,
        device_host_fallback_decision=device_host_fallback_decision,
        env=env,
    )
    side = resolve_xblock_sparse_pc_side_policy_setup(
        op=op,
        xblock_device_krylov_requested=bool(setup.xblock_device_krylov_requested),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        xblock_krylov_env=str(setup.xblock_krylov_env),
        pc_restart=int(pc_restart),
        pc_restart_env=str(pc_restart_env),
        tokamak_fp_er_pc=bool(tokamak_fp_er_pc),
        active_size=int(active_size),
        use_dkes=bool(use_dkes),
        include_xdot_sparse_pc=bool(include_xdot_sparse_pc),
        include_electric_field_xi_sparse_pc=bool(include_electric_field_xi_sparse_pc),
        resolve_xblock_policy=resolve_xblock_policy,
        env=env,
    )
    reuse = resolve_xblock_qi_device_operator_reuse_setup(
        op=op,
        xblock_krylov_method=str(side.xblock_krylov_method),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        qi_device_preconditioner_requested=bool(setup.qi_device_preconditioner_requested_for_fallback),
        qi_device_matrix_free_requested=bool(setup.qi_device_matrix_free_requested_for_fallback),
        qi_device_use_in_krylov_requested=bool(setup.qi_device_use_in_krylov_requested_for_fallback),
        precondition_side=str(side.precondition_side),
        xblock_jax_factors=bool(side.xblock_jax_factors),
        xblock_device_krylov_forced_jax_factors=bool(side.xblock_device_krylov_forced_jax_factors),
        xblock_preconditioner_xi=int(setup.xblock_preconditioner_xi),
        reuse_decision=reuse_decision,
        env=env,
    )
    return XBlockSparsePCBranchSetup(
        xblock_drop_tol=float(setup.xblock_drop_tol),
        xblock_drop_rel=float(setup.xblock_drop_rel),
        xblock_ilu_drop_tol=float(setup.xblock_ilu_drop_tol),
        xblock_fill_factor=float(setup.xblock_fill_factor),
        xblock_lower_fill_mode=str(setup.xblock_lower_fill_mode),
        xblock_lower_fill_ignored_env=bool(setup.xblock_lower_fill_ignored_env),
        xblock_preconditioner_xi=int(setup.xblock_preconditioner_xi),
        force_assembled_host_fp=bool(setup.force_assembled_host_fp),
        xblock_assembled_host_fp=bool(setup.xblock_assembled_host_fp),
        xblock_krylov_env_requested=str(setup.xblock_krylov_env_requested),
        xblock_krylov_env=str(setup.xblock_krylov_env),
        xblock_krylov_requested=str(setup.xblock_krylov_requested),
        xblock_device_fgmres_requested=bool(setup.xblock_device_fgmres_requested),
        xblock_device_gmres_requested=bool(setup.xblock_device_gmres_requested),
        xblock_device_bicgstab_requested=bool(setup.xblock_device_bicgstab_requested),
        xblock_device_tfqmr_requested=bool(setup.xblock_device_tfqmr_requested),
        xblock_device_krylov_requested=bool(setup.xblock_device_krylov_requested),
        xblock_device_host_fallback_decision=setup.xblock_device_host_fallback_decision,
        xblock_device_host_fallback_auto_disabled_by_qi_device=bool(
            setup.xblock_device_host_fallback_auto_disabled_by_qi_device
        ),
        qi_device_preconditioner_requested_for_fallback=bool(
            setup.qi_device_preconditioner_requested_for_fallback
        ),
        qi_device_matrix_free_requested_for_fallback=bool(
            setup.qi_device_matrix_free_requested_for_fallback
        ),
        qi_device_use_in_krylov_requested_for_fallback=bool(
            setup.qi_device_use_in_krylov_requested_for_fallback
        ),
        xblock_jax_factors=bool(reuse.xblock_jax_factors),
        xblock_jax_factor_format=str(side.xblock_jax_factor_format),
        xblock_jax_factor_apply=str(side.xblock_jax_factor_apply),
        xblock_device_krylov_forced_jax_factors=bool(
            reuse.xblock_device_krylov_forced_jax_factors
        ),
        full_fp_3d_pc=bool(side.full_fp_3d_pc),
        side_env=str(side.side_env),
        precondition_side=str(side.precondition_side),
        xblock_default_right_pc=bool(side.xblock_default_right_pc),
        xblock_krylov_method=str(side.xblock_krylov_method),
        xblock_device_fgmres_forced_right_pc=bool(side.xblock_device_fgmres_forced_right_pc),
        pc_restart=int(side.pc_restart),
        xblock_default_restart_capped=bool(side.xblock_default_restart_capped),
        xblock_qi_device_operator_reuse_decision=reuse.decision,
        xblock_qi_device_operator_reuse_skip_factors=bool(reuse.skip_xblock_factors),
        messages=tuple((*setup.messages, *side.messages, *reuse.messages)),
    )


def build_xblock_local_preconditioner(
    *,
    skip_factors: bool,
    elapsed_s: Callable[[], float],
    build_preconditioner: Callable[..., ArrayFn],
    op: object,
    build_jax_factors: bool,
    preconditioner_species: int,
    preconditioner_xi: int,
    drop_tol: float,
    drop_rel: float,
    ilu_drop_tol: float,
    fill_factor: float,
    force_assembled_host_fp: bool,
    emit: EmitFn | None = None,
) -> XBlockLocalPreconditionerBuildResult:
    """Build or skip the local x-block factor preconditioner with timing."""

    factor_start_s = float(elapsed_s())
    if bool(skip_factors):

        def identity_preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(v, dtype=jnp.float64)

        return XBlockLocalPreconditionerBuildResult(
            preconditioner=identity_preconditioner,
            factor_s=float(elapsed_s()) - factor_start_s,
            built=False,
        )

    preconditioner = build_preconditioner(
        op=op,
        build_jax_factors=bool(build_jax_factors),
        preconditioner_species=int(preconditioner_species),
        preconditioner_xi=int(preconditioner_xi),
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        ilu_drop_tol=float(ilu_drop_tol),
        fill_factor=float(fill_factor),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        emit=emit,
    )
    return XBlockLocalPreconditionerBuildResult(
        preconditioner=preconditioner,
        factor_s=float(elapsed_s()) - factor_start_s,
        built=True,
    )




def _normalized_equilibration_norm(value: str) -> str:
    norm = str(value).strip().lower().replace("-", "_")
    if norm in {"inf", "max", "maximum"}:
        return "linf"
    if norm in {"linf", "l1", "l2"}:
        return norm
    return "linf"


def build_xblock_assembled_equilibration_setup(
    *,
    assembled_matrix: object,
    xblock_linear_size: int,
    elapsed_s: Callable[[], float],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledEquilibrationSetup:
    """Build optional row/column scaling for assembled x-block Krylov operators."""

    col_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE",
        default=False,
    )
    row_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE",
        default=bool(col_enabled),
    )
    row_metadata: dict[str, object] = {}
    col_metadata: dict[str, object] = {}
    row_scale_jnp: jnp.ndarray | None = None
    inv_row_scale_jnp: jnp.ndarray | None = None
    col_scale_jnp: jnp.ndarray | None = None
    inv_col_scale_jnp: jnp.ndarray | None = None
    messages: list[tuple[int, str]] = []
    row_built = False
    col_built = False
    if not bool(row_enabled):
        return XBlockAssembledEquilibrationSetup(
            row_enabled=bool(row_enabled),
            row_built=False,
            row_metadata=row_metadata,
            row_scale=None,
            inv_row_scale=None,
            col_enabled=bool(col_enabled),
            col_built=False,
            col_metadata=col_metadata,
            col_scale=None,
            inv_col_scale=None,
            messages=(),
        )

    row_start_s = float(elapsed_s())
    norm = _normalized_equilibration_norm(
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM") or "linf"
    )
    floor = _env_float(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_FLOOR",
        default=1.0e-14,
    )
    floor = max(0.0, float(floor))
    max_scale = max(
        1.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_MAX_SCALE",
            default=1.0e8,
        ),
    )
    assembled_csr = assembled_matrix.tocsr()
    abs_csr = abs(assembled_csr)
    if norm == "l1":
        row_norm = np.asarray(abs_csr.sum(axis=1), dtype=np.float64).reshape((-1,))
    elif norm == "l2":
        squared_csr = assembled_csr.copy()
        squared_csr.data = np.asarray(np.abs(squared_csr.data) ** 2, dtype=np.float64)
        row_norm = np.sqrt(np.asarray(squared_csr.sum(axis=1), dtype=np.float64).reshape((-1,)))
    else:
        row_norm = np.asarray(abs_csr.max(axis=1).toarray(), dtype=np.float64).reshape((-1,))
    row_norm = np.asarray(row_norm, dtype=np.float64)
    finite_positive = np.isfinite(row_norm) & (row_norm > float(floor))
    raw_scale = np.ones_like(row_norm, dtype=np.float64)
    raw_scale[finite_positive] = 1.0 / row_norm[finite_positive]
    row_scale_np = np.clip(raw_scale, 1.0 / float(max_scale), float(max_scale))
    inv_row_scale_np = 1.0 / row_scale_np
    expected_shape = (int(xblock_linear_size),)
    if (
        row_scale_np.shape != expected_shape
        or not np.all(np.isfinite(row_scale_np))
        or not np.all(np.isfinite(inv_row_scale_np))
    ):
        raise RuntimeError("assembled x-block row equilibration produced invalid row scales")
    row_scale_jnp = jnp.asarray(row_scale_np, dtype=jnp.float64)
    inv_row_scale_jnp = jnp.asarray(inv_row_scale_np, dtype=jnp.float64)
    row_built = True

    if bool(col_enabled):
        col_start_s = float(elapsed_s())
        row_scaled_abs = abs_csr.multiply(row_scale_np[:, None])
        if norm == "l1":
            col_norm = np.asarray(row_scaled_abs.sum(axis=0), dtype=np.float64).reshape((-1,))
        elif norm == "l2":
            row_scaled_squared = assembled_csr.copy()
            row_scaled_squared.data = np.asarray(row_scaled_squared.data, dtype=np.float64) ** 2
            row_scaled_squared = row_scaled_squared.multiply((row_scale_np**2)[:, None])
            col_norm = np.sqrt(np.asarray(row_scaled_squared.sum(axis=0), dtype=np.float64).reshape((-1,)))
        else:
            col_norm = np.asarray(row_scaled_abs.max(axis=0).toarray(), dtype=np.float64).reshape((-1,))
        col_norm = np.asarray(col_norm, dtype=np.float64)
        col_finite_positive = np.isfinite(col_norm) & (col_norm > float(floor))
        raw_col_scale = np.ones_like(col_norm, dtype=np.float64)
        raw_col_scale[col_finite_positive] = 1.0 / col_norm[col_finite_positive]
        col_scale_np = np.clip(raw_col_scale, 1.0 / float(max_scale), float(max_scale))
        inv_col_scale_np = 1.0 / col_scale_np
        if (
            col_scale_np.shape != expected_shape
            or not np.all(np.isfinite(col_scale_np))
            or not np.all(np.isfinite(inv_col_scale_np))
        ):
            raise RuntimeError("assembled x-block column equilibration produced invalid column scales")
        col_scale_jnp = jnp.asarray(col_scale_np, dtype=jnp.float64)
        inv_col_scale_jnp = jnp.asarray(inv_col_scale_np, dtype=jnp.float64)
        col_built = True
        col_norm_positive = col_norm[col_finite_positive]
        col_metadata = {
            "enabled": True,
            "built": True,
            "norm": norm,
            "floor": float(floor),
            "max_scale": float(max_scale),
            "setup_s": float(elapsed_s()) - col_start_s,
            "zero_or_tiny_columns": int(col_norm.size - np.count_nonzero(col_finite_positive)),
            "col_norm_min": float(np.min(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_norm_max": float(np.max(col_norm_positive)) if col_norm_positive.size else 0.0,
            "col_scale_min": float(np.min(col_scale_np)) if col_scale_np.size else 0.0,
            "col_scale_max": float(np.max(col_scale_np)) if col_scale_np.size else 0.0,
        }

    row_norm_positive = row_norm[finite_positive]
    row_metadata = {
        "enabled": True,
        "built": True,
        "norm": norm,
        "floor": float(floor),
        "max_scale": float(max_scale),
        "setup_s": float(elapsed_s()) - row_start_s,
        "zero_or_tiny_rows": int(row_norm.size - np.count_nonzero(finite_positive)),
        "row_norm_min": float(np.min(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_norm_max": float(np.max(row_norm_positive)) if row_norm_positive.size else 0.0,
        "row_scale_min": float(np.min(row_scale_np)) if row_scale_np.size else 0.0,
        "row_scale_max": float(np.max(row_scale_np)) if row_scale_np.size else 0.0,
        "column_equilibration": bool(col_built),
    }
    messages.append(
        (
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "assembled row equilibration built "
            f"norm={norm} "
            f"scale_range=[{float(np.min(row_scale_np)):.3e}, {float(np.max(row_scale_np)):.3e}]",
        )
    )
    if bool(col_built):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled column equilibration built "
                f"norm={norm} "
                f"scale_range=[{col_metadata['col_scale_min']:.3e}, {col_metadata['col_scale_max']:.3e}]",
            )
        )

    return XBlockAssembledEquilibrationSetup(
        row_enabled=bool(row_enabled),
        row_built=bool(row_built),
        row_metadata=row_metadata,
        row_scale=row_scale_jnp,
        inv_row_scale=inv_row_scale_jnp,
        col_enabled=bool(col_enabled),
        col_built=bool(col_built),
        col_metadata=col_metadata,
        col_scale=col_scale_jnp,
        inv_col_scale=inv_col_scale_jnp,
        messages=tuple(messages),
    )


def _csr_storage_nbytes(*, nnz: int, n_rows: int) -> int:
    return int(
        int(nnz) * (np.dtype(np.float64).itemsize + np.dtype(np.int32).itemsize)
        + (int(n_rows) + 1) * np.dtype(np.int32).itemsize
    )


def build_xblock_assembled_operator_preflight_setup(
    *,
    op: object,
    xblock_active_idx_np: np.ndarray | None,
    sparse_pc_fp_dense_velocity_block: bool | None,
    xblock_krylov_method: str,
    estimate_summary: Callable[..., object],
    full_pattern: Callable[..., object],
    active_pattern: Callable[..., object],
    summarize_pattern: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledOperatorPreflightSetup:
    """Resolve assembled-operator memory budget and structural pattern."""

    csr_max_mb = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB",
            default=2048.0,
        ),
    )
    drop_tol = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DROP_TOL",
            default=0.0,
        ),
    )
    device_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE",
        default=str(xblock_krylov_method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"},
    )
    device_required = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED",
        default=False,
    )
    max_colors = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS",
        default=512,
        minimum=1,
    )
    full_preflight = estimate_summary(
        op,
        fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
    )
    full_csr_nbytes = _csr_storage_nbytes(
        nnz=int(full_preflight.nnz),
        n_rows=int(full_preflight.shape[0]),
    )
    preflight_csr_nbytes = int(full_csr_nbytes)
    preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
    csr_cap_nbytes = int(float(csr_max_mb) * 1.0e6)
    pattern = None
    preflight_scope = "full"
    metadata: dict[str, object] = {
        "active_dof": bool(xblock_active_idx_np is not None),
        "preflight_scope": preflight_scope,
        "preflight_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_pattern_max_row_nnz_estimate": int(full_preflight.max_row_nnz),
        "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
        "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
        "preflight_full_pattern_nnz_estimate": int(full_preflight.nnz),
        "preflight_full_csr_nbytes_estimate": int(full_csr_nbytes),
        "preflight_csr_max_mb": float(csr_max_mb),
        "preflight_rejected": False,
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": False,
    }
    if int(csr_cap_nbytes) <= 0:
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected non-positive CSR memory budget "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if xblock_active_idx_np is not None:
        pattern = active_pattern(
            op,
            xblock_active_idx_np,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
        active_preflight = summarize_pattern(op, pattern)
        preflight_scope = "active_dof"
        preflight_csr_nbytes = _csr_storage_nbytes(
            nnz=int(active_preflight.nnz),
            n_rows=int(active_preflight.shape[0]),
        )
        preflight_peak_nbytes = int(3 * preflight_csr_nbytes)
        metadata.update(
            {
                "preflight_scope": preflight_scope,
                "preflight_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_pattern_max_row_nnz_estimate": int(active_preflight.max_row_nnz),
                "preflight_csr_nbytes_estimate": int(preflight_csr_nbytes),
                "preflight_peak_nbytes_estimate": int(preflight_peak_nbytes),
                "preflight_active_pattern_nnz_estimate": int(active_preflight.nnz),
                "preflight_active_csr_nbytes_estimate": int(preflight_csr_nbytes),
            }
        )
    if int(preflight_csr_nbytes) > int(csr_cap_nbytes):
        metadata["preflight_rejected"] = True
        raise XBlockAssembledPreflightError(
            "assembled x-block operator preflight rejected "
            f"{preflight_scope} CSR estimate "
            f"{int(preflight_csr_nbytes) / 1.0e6:.3g} MB > "
            f"{float(csr_max_mb):.3g} MB",
            metadata,
        )
    if pattern is None:
        pattern = full_pattern(
            op,
            fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
        )
    summary = summarize_pattern(op, pattern)
    return XBlockAssembledOperatorPreflightSetup(
        csr_max_mb=float(csr_max_mb),
        drop_tol=float(drop_tol),
        device_enabled=bool(device_enabled),
        device_required=bool(device_required),
        max_colors=int(max_colors),
        csr_cap_nbytes=int(csr_cap_nbytes),
        pattern=pattern,
        summary=summary,
        metadata=metadata,
    )


def build_xblock_assembled_device_setup(
    *,
    assembled_matrix: object,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    csr_cap_nbytes: int,
    device_enabled: bool,
    device_required: bool,
    validation_samples: int,
    validation_tol: float,
    device_csr_from_matrix: Callable[..., object],
    validate_device_csr_matvec: Callable[..., Sequence[float]],
) -> XBlockAssembledDeviceSetup:
    """Optionally build and validate a device CSR matvec for an assembled operator."""

    if not bool(device_enabled):
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=None,
            messages=(),
        )
    messages: list[tuple[int, str]] = []
    try:
        device_operator = device_csr_from_matrix(
            assembled_matrix,
            dtype=np.float64,
            max_nbytes=int(csr_cap_nbytes),
        )
        validation_errors = validate_device_csr_matvec(
            device_operator,
            assembled_matvec,
            samples=int(validation_samples),
            rtol=float(validation_tol),
            seed=1730,
        )
        return XBlockAssembledDeviceSetup(
            device_operator=device_operator,
            device_resident=True,
            validation_errors=tuple(float(v) for v in validation_errors),
            error=None,
            messages=(),
        )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if bool(device_required):
            raise RuntimeError(f"assembled x-block device CSR operator failed ({error})") from exc
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "assembled device operator disabled after build failure "
                f"({error})",
            )
        )
        return XBlockAssembledDeviceSetup(
            device_operator=None,
            device_resident=False,
            validation_errors=(),
            error=error,
            messages=tuple(messages),
    )


def build_xblock_assembled_matvec_setup(
    *,
    assembled_matvec: Callable[[np.ndarray], np.ndarray],
    device_operator: object | None,
    mv_count: MatvecCounter,
    progress_every: int,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
) -> XBlockAssembledMatvecSetup:
    """Select host or device matvec closure for assembled x-block Krylov solves."""

    if device_operator is not None:
        device_matvec = device_operator.jitted_matvec()

        def matvec(v: jnp.ndarray) -> jnp.ndarray:
            mv_count.increment()
            if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"assembled_device_matvecs={int(mv_count)} "
                    f"elapsed_s={float(elapsed_s()):.3f}",
                )
            return device_matvec(jnp.asarray(v, dtype=jnp.float64))

        return XBlockAssembledMatvecSetup(matvec=matvec, location="device")

    def matvec(v: jnp.ndarray) -> jnp.ndarray:
        mv_count.increment()
        if emit is not None and int(progress_every) > 0 and int(mv_count) % int(progress_every) == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"assembled_host_matvecs={int(mv_count)} "
                f"elapsed_s={float(elapsed_s()):.3f}",
            )
        v_np = np.asarray(jax.device_get(v), dtype=np.float64).reshape((-1,))
        return jnp.asarray(assembled_matvec(v_np), dtype=jnp.float64)

    return XBlockAssembledMatvecSetup(matvec=matvec, location="host")


def build_xblock_assembled_operator_if_requested(
    *,
    enabled: bool,
    op: object,
    rhs_dtype: object,
    xblock_active_idx_np: np.ndarray | None,
    sparse_pc_fp_dense_velocity_block: bool | None,
    xblock_krylov_method: str,
    xblock_linear_size: int,
    true_matvec_no_count: ArrayFn,
    default_matvec: ArrayFn,
    mv_count: MatvecCounter,
    progress_every: int,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    estimate_summary: Callable[..., object],
    full_pattern: Callable[..., object],
    active_pattern: Callable[..., object],
    summarize_pattern: Callable[..., object],
    build_operator_from_pattern: Callable[..., object],
    device_csr_from_matrix: Callable[..., object],
    validate_device_csr_matvec: Callable[..., object],
    finalize_metadata: Callable[..., dict[str, object]],
    backend: str,
    env: Mapping[str, str] | None = None,
) -> XBlockAssembledOperatorBuildResult:
    """Optionally assemble an x-block Krylov operator and return replacement matvec state."""

    if not bool(enabled):
        return XBlockAssembledOperatorBuildResult(
            matvec=default_matvec,
            built=False,
            device_resident=False,
            metadata={},
            device_operator=None,
            pc_factor_increment_s=0.0,
            row_enabled=False,
            row_built=False,
            row_metadata={},
            row_scale=None,
            inv_row_scale=None,
            col_enabled=False,
            col_built=False,
            col_metadata={},
            col_scale=None,
            inv_col_scale=None,
        )

    start_s = float(elapsed_s())
    metadata: dict[str, object] = {}
    if emit is not None:
        emit(
            1,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "building assembled operator for Krylov matvec reuse",
        )
    try:
        try:
            preflight = build_xblock_assembled_operator_preflight_setup(
                op=op,
                xblock_active_idx_np=xblock_active_idx_np,
                sparse_pc_fp_dense_velocity_block=sparse_pc_fp_dense_velocity_block,
                xblock_krylov_method=str(xblock_krylov_method),
                estimate_summary=estimate_summary,
                full_pattern=full_pattern,
                active_pattern=active_pattern,
                summarize_pattern=summarize_pattern,
                env=env,
            )
        except XBlockAssembledPreflightError as preflight_exc:
            metadata.update(preflight_exc.metadata)
            raise
        metadata.update(preflight.metadata)

        def matvec_np_no_count(x_np: np.ndarray) -> np.ndarray:
            return np.asarray(
                jax.device_get(
                    true_matvec_no_count(
                        jnp.asarray(np.asarray(x_np, dtype=np.float64), dtype=rhs_dtype)
                    )
                ),
                dtype=np.float64,
            ).reshape((-1,))

        bundle = build_operator_from_pattern(
            matvec_np_no_count,
            pattern=preflight.pattern,
            dtype=np.float64,
            backend=str(backend),
            csr_max_mb=float(preflight.csr_max_mb),
            drop_tol=float(preflight.drop_tol),
            allow_operator_only=False,
            max_colors=int(preflight.max_colors),
        )
        matrix = bundle.matrix
        if matrix is None:
            raise RuntimeError("assembled x-block operator materialization returned no matrix")

        validation_samples = _env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE",
            default=1,
            minimum=0,
        )
        validation_tol = max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_VALIDATE_RTOL",
                default=1.0e-8,
            ),
        )
        validation_errors: list[float] = []
        rng = np.random.default_rng(1729)
        for _ in range(int(validation_samples)):
            probe = rng.standard_normal(int(xblock_linear_size)).astype(np.float64)
            probe_norm = float(np.linalg.norm(probe))
            if np.isfinite(probe_norm) and probe_norm > 0.0:
                probe /= probe_norm
            ref = matvec_np_no_count(probe)
            got = np.asarray(bundle.matvec(probe), dtype=np.float64).reshape((-1,))
            denom = max(float(np.linalg.norm(ref)), 1.0e-300)
            validation_errors.append(float(np.linalg.norm(got - ref) / denom))
        max_validation_error = max(validation_errors, default=0.0)
        if max_validation_error > float(validation_tol):
            raise RuntimeError(
                "assembled x-block operator validation failed "
                f"max_rel_error={max_validation_error:.3e} > {float(validation_tol):.3e}"
            )

        equilibration = build_xblock_assembled_equilibration_setup(
            assembled_matrix=matrix,
            xblock_linear_size=int(xblock_linear_size),
            elapsed_s=elapsed_s,
            env=env,
        )
        if emit is not None:
            for level, message in equilibration.messages:
                emit(int(level), str(message))

        device = build_xblock_assembled_device_setup(
            assembled_matrix=matrix,
            assembled_matvec=bundle.matvec,
            csr_cap_nbytes=int(preflight.csr_cap_nbytes),
            device_enabled=bool(preflight.device_enabled),
            device_required=bool(preflight.device_required),
            validation_samples=int(validation_samples),
            validation_tol=float(validation_tol),
            device_csr_from_matrix=device_csr_from_matrix,
            validate_device_csr_matvec=validate_device_csr_matvec,
        )
        if emit is not None:
            for level, message in device.messages:
                emit(int(level), str(message))

        matvec_setup = build_xblock_assembled_matvec_setup(
            assembled_matvec=bundle.matvec,
            device_operator=device.device_operator,
            mv_count=mv_count,
            progress_every=int(progress_every),
            elapsed_s=elapsed_s,
            emit=emit,
        )
        metadata = finalize_metadata(
            metadata=metadata,
            setup_s=float(elapsed_s()) - start_s,
            assembled_matrix=matrix,
            assembled_summary=preflight.summary,
            assembled_bundle_metadata=bundle.metadata,
            max_colors=int(preflight.max_colors),
            validation_errors=validation_errors,
            device_enabled=bool(preflight.device_enabled),
            device_required=bool(preflight.device_required),
            device_resident=bool(device.device_resident),
            device_operator=device.device_operator,
            device_validation_errors=tuple(device.validation_errors),
            device_error=device.error,
        )
        if emit is not None:
            emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres assembled operator "
                f"built location={matvec_setup.location} nnz={metadata['matrix_nnz']} "
                f"setup_s={metadata['setup_s']:.3f}",
            )
        return XBlockAssembledOperatorBuildResult(
            matvec=matvec_setup.matvec,
            built=True,
            device_resident=bool(device.device_resident),
            metadata=metadata,
            device_operator=device.device_operator,
            pc_factor_increment_s=float(metadata["setup_s"]),
            row_enabled=bool(equilibration.row_enabled),
            row_built=bool(equilibration.row_built),
            row_metadata=dict(equilibration.row_metadata),
            row_scale=equilibration.row_scale,
            inv_row_scale=equilibration.inv_row_scale,
            col_enabled=bool(equilibration.col_enabled),
            col_built=bool(equilibration.col_built),
            col_metadata=dict(equilibration.col_metadata),
            col_scale=equilibration.col_scale,
            inv_col_scale=equilibration.inv_col_scale,
        )
    except Exception as exc:  # noqa: BLE001
        metadata = {
            **metadata,
            "error": f"{type(exc).__name__}: {exc}",
            "setup_s": float(elapsed_s()) - start_s,
        }
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"assembled operator disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockAssembledOperatorBuildResult(
            matvec=default_matvec,
            built=False,
            device_resident=False,
            metadata=metadata,
            device_operator=None,
            pc_factor_increment_s=0.0,
            row_enabled=False,
            row_built=False,
            row_metadata={},
            row_scale=None,
            inv_row_scale=None,
            col_enabled=False,
            col_built=False,
            col_metadata={},
            col_scale=None,
            inv_col_scale=None,
        )


def finalize_xblock_assembled_operator_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
    assembled_matrix: object,
    assembled_summary: object,
    assembled_bundle_metadata: object,
    max_colors: int,
    validation_errors: Sequence[float],
    device_enabled: bool,
    device_required: bool,
    device_resident: bool,
    device_operator: object | None,
    device_validation_errors: Sequence[float],
    device_error: str | None,
) -> dict[str, object]:
    """Return normalized metadata after assembled x-block operator construction."""

    if hasattr(assembled_matrix, "nnz"):
        matrix_nnz = int(assembled_matrix.nnz)
    else:
        matrix_nnz = int(np.count_nonzero(np.asarray(assembled_matrix)))
    return {
        **dict(metadata),
        "setup_s": float(setup_s),
        "pattern_nnz": int(assembled_summary.nnz),
        "pattern_avg_row_nnz": float(assembled_summary.avg_row_nnz),
        "pattern_max_row_nnz": int(assembled_summary.max_row_nnz),
        "storage_kind": assembled_bundle_metadata.storage_kind,
        "reason": assembled_bundle_metadata.reason,
        "matrix_nnz": int(matrix_nnz),
        "csr_nbytes_estimate": int(assembled_bundle_metadata.csr_nbytes_estimate),
        "max_colors": int(max_colors),
        "validation_rel_errors": tuple(float(v) for v in validation_errors),
        "device_enabled": bool(device_enabled),
        "device_required": bool(device_required),
        "device_resident": bool(device_resident),
        "device_nnz": int(device_operator.nnz) if device_operator is not None else None,
        "device_csr_nbytes_estimate": (
            int(device_operator.nbytes_estimate) if device_operator is not None else None
        ),
        "device_validation_rel_errors": tuple(float(v) for v in device_validation_errors),
        "device_error": device_error,
    }


def resolve_xblock_moment_schur_policy_setup(
    *,
    op: object,
    xblock_krylov_method: str,
    xblock_jax_factors: bool,
    xblock_jax_factor_format: str,
    precondition_side: str,
    env: Mapping[str, str] | None = None,
) -> XBlockMomentSchurPolicySetup:
    """Resolve x-block moment-Schur default, force, and probe settings."""

    default_candidate = bool(
        str(xblock_krylov_method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
        and int(op.rhs_mode) == 1
        and int(op.constraint_scheme) == 1
        and int(op.extra_size) > 0
        and int(op.phi1_size) == 0
    )
    env_raw = _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR").lower()
    default_blocked_by_compact_factors = bool(
        default_candidate
        and env_raw in {"", "auto", "default"}
        and bool(xblock_jax_factors)
        and str(xblock_jax_factor_format).strip().lower() == "csr"
    )
    default_enabled = bool(default_candidate and not default_blocked_by_compact_factors)
    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR",
        default=default_enabled,
    )
    rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND",
            default=1.0e-12,
        ),
    )
    probe_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE",
        default=False,
    )
    probe_min_improvement = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT",
            default=0.0,
        ),
    )
    messages: list[tuple[int, str]] = []
    if bool(default_blocked_by_compact_factors) and not bool(enabled):
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "constraint1 moment-Schur default disabled for compact JAX factors "
                "(set SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR=1 to force)",
            )
        )
    if bool(enabled) and str(precondition_side) != "none":
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "constraint1 moment-Schur build start",
            )
        )
    return XBlockMomentSchurPolicySetup(
        default_candidate=bool(default_candidate),
        default_blocked_by_compact_factors=bool(default_blocked_by_compact_factors),
        enabled=bool(enabled),
        rcond=float(rcond),
        probe_enabled=bool(probe_enabled),
        probe_min_improvement=float(probe_min_improvement),
        messages=tuple(messages),
    )


def evaluate_xblock_moment_schur_probe_result(
    *,
    residual_before: float,
    residual_after: float,
    min_improvement: float,
) -> XBlockMomentSchurProbeResult:
    """Gate moment-Schur use from before/after residual norms."""

    before = float(residual_before)
    after = float(residual_after)
    if before > 0.0:
        ratio = float(after / before)
        required = before * max(0.0, 1.0 - float(min_improvement))
        used = bool(np.isfinite(after) and after < float(required))
    else:
        ratio = 0.0 if after == 0.0 else float("inf")
        used = bool(np.isfinite(after) and after <= 0.0)
    reason = "probe_reduced" if bool(used) else "probe_not_reduced"
    messages = (
        (
            0 if bool(used) else 1,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "constraint1 moment-Schur "
            f"{'accepted' if bool(used) else 'rejected'} "
            f"seed residual {before:.6e} -> {after:.6e} "
            f"(ratio={float(ratio):.6e})",
        ),
    )
    return XBlockMomentSchurProbeResult(
        used=bool(used),
        reason=str(reason),
        residual_before=float(before),
        residual_after=float(after),
        improvement_ratio=float(ratio),
        messages=messages,
    )


def finalize_xblock_moment_schur_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return moment-Schur metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_moment_schur_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized moment-Schur failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def resolve_xblock_two_level_policy_setup(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None = None,
) -> XBlockTwoLevelPolicySetup:
    """Resolve x-block two-level correction admission and build parameters."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL",
        default=False,
    )
    return XBlockTwoLevelPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        mode=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE") or "additive",
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS",
            default=48,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX",
            default=8,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_INCLUDE_RHS",
            default=True,
        ),
    )


def finalize_xblock_two_level_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return two-level metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_two_level_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized two-level failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def _xblock_device_krylov_method(method: str) -> bool:
    return str(method) in {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}


def resolve_xblock_global_coupling_policy_setup(
    *,
    precondition_side: str,
    xblock_krylov_method: str,
    env: Mapping[str, str] | None = None,
) -> XBlockGlobalCouplingPolicySetup:
    """Resolve x-block global-coupling admission and build parameters."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING",
        default=False,
    )
    use_device_builder = _xblock_device_krylov_method(str(xblock_krylov_method))
    return XBlockGlobalCouplingPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        use_device_builder=bool(use_device_builder),
        mode=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE") or "additive",
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS",
            default=96,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX",
            default=12,
            minimum=0,
        ),
        angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX",
            default=2,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_INCLUDE_RHS",
            default=True,
        ),
        setup_max_s=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S",
                default=180.0 if bool(use_device_builder) else 0.0,
            ),
        ),
    )


def finalize_xblock_global_coupling_metadata(
    *,
    metadata: Mapping[str, object],
    setup_s: float,
) -> dict[str, object]:
    """Return global-coupling metadata with normalized setup timing."""

    out = dict(metadata)
    out["setup_s"] = float(setup_s)
    return out


def failed_xblock_global_coupling_metadata(
    *,
    exc: BaseException,
    setup_s: float,
) -> dict[str, object]:
    """Return normalized global-coupling failure metadata."""

    return {
        "error": f"{type(exc).__name__}: {exc}",
        "setup_s": float(setup_s),
    }


def apply_xblock_moment_schur_stage(
    *,
    context: XBlockMomentSchurStageContext,
) -> XBlockMomentSchurStageResult:
    """Build and optionally probe the primary x-block moment-Schur stage."""

    if context.emit is not None:
        for level, message in context.policy.messages:
            context.emit(int(level), str(message))
    if (not bool(context.policy.enabled)) or str(context.precondition_side) == "none":
        return XBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=None,
            metadata={},
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        candidate, metadata, stats = context.builder(
            op=context.op,
            base_preconditioner=context.base_preconditioner,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            rcond=float(context.policy.rcond),
            emit=context.emit,
        )
        used = True
        reason: str | None = "built"
        probe_residual_before: float | None = None
        probe_residual_after: float | None = None
        probe_improvement_ratio: float | None = None
        if bool(context.policy.probe_enabled):
            seed_candidate = jnp.asarray(candidate(context.rhs), dtype=jnp.float64)
            seed_residual = context.rhs - jnp.asarray(
                context.matvec_no_count(seed_candidate),
                dtype=jnp.float64,
            )
            probe_result = evaluate_xblock_moment_schur_probe_result(
                residual_before=float(jnp.linalg.norm(context.rhs)),
                residual_after=float(jnp.linalg.norm(seed_residual)),
                min_improvement=float(context.policy.probe_min_improvement),
            )
            used = bool(probe_result.used)
            reason = str(probe_result.reason)
            probe_residual_before = float(probe_result.residual_before)
            probe_residual_after = float(probe_result.residual_after)
            probe_improvement_ratio = float(probe_result.improvement_ratio)
            if context.emit is not None:
                for level, message in probe_result.messages:
                    context.emit(int(level), str(message))
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockMomentSchurStageResult(
            preconditioner=candidate if bool(used) else context.base_preconditioner,
            built=True,
            used=bool(used),
            reason=reason,
            metadata=finalize_xblock_moment_schur_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            probe_residual_before=probe_residual_before,
            probe_residual_after=probe_residual_after,
            probe_improvement_ratio=probe_improvement_ratio,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"constraint1 moment-Schur disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=reason,
            metadata=failed_xblock_moment_schur_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=float(setup_s),
        )


def apply_xblock_two_level_stage(
    *,
    context: XBlockTwoLevelStageContext,
) -> XBlockTwoLevelStageResult:
    """Build the optional primary x-block two-level stage."""

    if not bool(context.policy.should_build):
        return XBlockTwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        if context.builder is None:
            from sfincs_jax.solvers.preconditioners.qi.corrections import (
                build_rhs1_xblock_two_level_preconditioner,
            )

            builder = build_rhs1_xblock_two_level_preconditioner
        else:
            builder = context.builder
        preconditioner, metadata, stats = builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=int(context.policy.fsavg_lmax),
            max_extra_units=int(context.policy.max_extra_units),
            max_directions=int(context.policy.max_directions),
            rcond=float(context.policy.rcond),
            include_rhs=bool(context.policy.include_rhs),
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockTwoLevelStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=finalize_xblock_two_level_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"two-level coarse disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockTwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata=failed_xblock_two_level_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )


def apply_xblock_global_coupling_stage(
    *,
    context: XBlockGlobalCouplingStageContext,
) -> XBlockGlobalCouplingStageResult:
    """Build the optional primary x-block global-coupling stage."""

    if not bool(context.policy.should_build):
        return XBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    try:
        if context.host_builder is None or context.device_builder is None:
            from sfincs_jax.solvers.preconditioners.qi.corrections import (
                build_rhs1_xblock_device_global_coupling_preconditioner,
                build_rhs1_xblock_smoothed_global_coupling_preconditioner,
            )

            host_builder = (
                build_rhs1_xblock_smoothed_global_coupling_preconditioner
                if context.host_builder is None
                else context.host_builder
            )
            device_builder = (
                build_rhs1_xblock_device_global_coupling_preconditioner
                if context.device_builder is None
                else context.device_builder
            )
        else:
            host_builder = context.host_builder
            device_builder = context.device_builder
        builder = device_builder if bool(context.policy.use_device_builder) else host_builder
        preconditioner, metadata, stats = builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=int(context.policy.fsavg_lmax),
            angular_lmax=int(context.policy.angular_lmax),
            max_extra_units=int(context.policy.max_extra_units),
            max_directions=int(context.policy.max_directions),
            rcond=float(context.policy.rcond),
            include_rhs=bool(context.policy.include_rhs),
            max_setup_s=float(context.policy.setup_max_s),
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        return XBlockGlobalCouplingStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=finalize_xblock_global_coupling_metadata(
                metadata=metadata,
                setup_s=float(setup_s),
            ),
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"global-coupling disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata=failed_xblock_global_coupling_metadata(
                exc=exc,
                setup_s=float(setup_s),
            ),
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )






def resolve_xblock_seed_policy_setup(
    *,
    moment_schur_used: bool,
    env: Mapping[str, str] | None = None,
) -> XBlockSeedPolicySetup:
    """Resolve initial and moment-Schur x-block seed controls."""

    return XBlockSeedPolicySetup(
        initial_seed_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED",
            default=False,
        ),
        moment_schur_seed_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED",
            default=bool(moment_schur_used),
        ),
    )




































def _elapsed_since_now() -> Callable[[], float]:
    """Return a cheap elapsed-time callback for explicit host sparse branches."""

    start_s = perf_counter()
    return lambda: perf_counter() - start_s


_XBLOCK_SPARSE_PC_FINAL_METADATA_CORE_STATE_KEYS = (
    "assembled_operator_built",
    "assembled_operator_enabled",
    "assembled_operator_metadata",
    "candidate_iterations",
    "candidate_krylov_method",
    "candidate_matvecs",
    "candidate_residual_norm",
    "device_krylov_estimated_matvecs",
    "fallback_candidate_improved_rhs",
    "fallback_started_from_candidate",
    "mv_count",
    "pc_factor_s",
    "pc_maxiter",
    "pc_restart",
    "precondition_side",
    "reported_iterations",
    "reported_matvecs",
    "setup_s",
    "solve_s",
    "sparse_timer",
    "xblock_assembled_host_fp",
    "xblock_col_equilibration_built",
    "xblock_col_equilibration_enabled",
    "xblock_col_equilibration_metadata",
    "xblock_default_restart_capped",
    "xblock_default_right_pc",
    "xblock_jax_factor_apply",
    "xblock_jax_factor_format",
    "xblock_jax_factors",
    "xblock_krylov_method",
    "xblock_lgmres_rescue_maxiter_capped",
    "xblock_lgmres_rescue_outer_k",
    "xblock_linear_size",
    "xblock_lower_fill_ignored_env",
    "xblock_lower_fill_mode",
    "xblock_preconditioner_built",
    "xblock_preconditioner_xi",
    "xblock_row_equilibration_built",
    "xblock_row_equilibration_enabled",
    "xblock_row_equilibration_metadata",
    "xblock_side_probe_enabled",
    "xblock_side_probe_initial_method",
    "xblock_side_probe_initial_side",
    "xblock_side_probe_iterations",
    "xblock_side_probe_lgmres_rescue",
    "xblock_side_probe_matvecs",
    "xblock_side_probe_physical_seed_preserved_after_switch",
    "xblock_side_probe_residual_norm",
    "xblock_side_probe_residual_ratio",
    "xblock_side_probe_s",
    "xblock_side_probe_seed_residual_norm",
    "xblock_side_probe_seed_used",
    "xblock_side_probe_selected_method",
    "xblock_side_probe_selected_side",
    "xblock_side_probe_switch_suppressed_by_explicit_side",
    "xblock_side_probe_switch_suppressed_by_global_coupling",
    "xblock_side_probe_switched",
    "xblock_side_probe_used",
    "xblock_use_active_dof",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_NESTED_STATE_KEYS = (
    "global_coupling_built",
    "global_coupling_enabled",
    "global_coupling_metadata",
    "global_coupling_stats",
    "moment_schur_built",
    "moment_schur_default_blocked_by_compact_factors",
    "moment_schur_enabled",
    "moment_schur_metadata",
    "moment_schur_probe_improvement_ratio",
    "moment_schur_probe_residual_after",
    "moment_schur_probe_residual_before",
    "moment_schur_reason",
    "moment_schur_stats",
    "moment_schur_used",
    "two_level_built",
    "two_level_enabled",
    "two_level_metadata",
    "two_level_stats",
    "moment_schur_seed_enabled",
    "moment_schur_seed_residual_norm",
    "moment_schur_seed_residual_ratio",
    "moment_schur_seed_used",
    "qi_coarse_seed_candidate_count",
    "qi_coarse_seed_enabled",
    "qi_coarse_seed_improvement_ratio",
    "qi_coarse_seed_labels",
    "qi_coarse_seed_rank",
    "qi_coarse_seed_reason",
    "qi_coarse_seed_residual_after",
    "qi_coarse_seed_residual_before",
    "qi_coarse_seed_s",
    "qi_coarse_seed_used",
    "qi_galerkin_preconditioner_basis_reused_from_seed",
    "qi_galerkin_preconditioner_built",
    "qi_galerkin_preconditioner_candidate_count",
    "qi_galerkin_preconditioner_coarse_norm",
    "qi_galerkin_preconditioner_coarse_shape",
    "qi_galerkin_preconditioner_damping",
    "qi_galerkin_preconditioner_enabled",
    "qi_galerkin_preconditioner_improvement_ratio",
    "qi_galerkin_preconditioner_mode",
    "qi_galerkin_preconditioner_probe_candidates",
    "qi_galerkin_preconditioner_probe_reduced",
    "qi_galerkin_preconditioner_rank",
    "qi_galerkin_preconditioner_rcond",
    "qi_galerkin_preconditioner_reason",
    "qi_galerkin_preconditioner_residual_after",
    "qi_galerkin_preconditioner_residual_before",
    "qi_galerkin_preconditioner_selected_index",
    "qi_galerkin_preconditioner_setup_s",
    "qi_galerkin_preconditioner_used",
    "qi_galerkin_stats",
    "qi_seed_basis_kind",
    "qi_seed_max_angular_mode",
    "qi_seed_max_candidates",
    "qi_two_level_preconditioner_augmentation_labels",
    "qi_two_level_preconditioner_basis_reused_from_seed",
    "qi_two_level_preconditioner_built",
    "qi_two_level_preconditioner_candidate_count",
    "qi_two_level_preconditioner_coarse_norm",
    "qi_two_level_preconditioner_coarse_shape",
    "qi_two_level_preconditioner_coarse_solver",
    "qi_two_level_preconditioner_damping",
    "qi_two_level_preconditioner_enabled",
    "qi_two_level_preconditioner_improvement_ratio",
    "qi_two_level_preconditioner_operator_on_basis_norm",
    "qi_two_level_preconditioner_operator_on_basis_shape",
    "qi_two_level_preconditioner_probe_candidates",
    "qi_two_level_preconditioner_rank",
    "qi_two_level_preconditioner_rank_before_augmentation",
    "qi_two_level_preconditioner_rcond",
    "qi_two_level_preconditioner_reason",
    "qi_two_level_preconditioner_residual_after",
    "qi_two_level_preconditioner_residual_augment_include_residuals",
    "qi_two_level_preconditioner_residual_augment_max_extra",
    "qi_two_level_preconditioner_residual_augment_steps",
    "qi_two_level_preconditioner_residual_augmented",
    "qi_two_level_preconditioner_residual_before",
    "qi_two_level_preconditioner_selected_index",
    "qi_two_level_preconditioner_setup_s",
    "qi_two_level_preconditioner_smoothed_load_basis",
    "qi_two_level_preconditioner_smoothed_load_metadata",
    "qi_two_level_preconditioner_used",
    "qi_two_level_stats",
    "xblock_initial_seed_residual_norm",
    "xblock_initial_seed_residual_ratio",
    "xblock_initial_seed_used",
    "qi_device_augmented_krylov_mode",
    "qi_device_augmented_krylov_rank",
    "qi_device_augmented_krylov_reason",
    "qi_device_augmented_krylov_requested",
    "qi_device_augmented_krylov_used",
    "qi_device_augmented_seed_available",
    "qi_device_augmented_seed_labels",
    "qi_device_augmented_seed_max_rank",
    "qi_device_augmented_seed_projection_residual",
    "qi_device_augmented_seed_rank",
    "qi_device_augmented_seed_reason",
    "qi_device_augmented_seed_requested",
    "qi_device_augmented_seed_used",
    "qi_device_preconditioner_built",
    "qi_device_preconditioner_candidate_count",
    "qi_device_preconditioner_coarse_norm",
    "qi_device_preconditioner_coarse_shape",
    "qi_device_preconditioner_enabled",
    "qi_device_preconditioner_improvement_ratio",
    "qi_device_preconditioner_metadata",
    "qi_device_preconditioner_min_improvement",
    "qi_device_preconditioner_operator_on_basis_norm",
    "qi_device_preconditioner_operator_on_basis_shape",
    "qi_device_preconditioner_rank",
    "qi_device_preconditioner_reason",
    "qi_device_preconditioner_residual_after",
    "qi_device_preconditioner_residual_before",
    "qi_device_preconditioner_setup_s",
    "qi_device_preconditioner_use_in_krylov",
    "qi_device_preconditioner_used",
    "qi_device_preconditioner_used_in_krylov",
    "qi_device_stats",
    "qi_deflated_preconditioner_built",
    "qi_deflated_preconditioner_candidate_count",
    "qi_deflated_preconditioner_enabled",
    "qi_deflated_preconditioner_improvement_ratio",
    "qi_deflated_preconditioner_metadata",
    "qi_deflated_preconditioner_rank",
    "qi_deflated_preconditioner_reason",
    "qi_deflated_preconditioner_residual_after",
    "qi_deflated_preconditioner_residual_before",
    "qi_deflated_preconditioner_setup_s",
    "qi_deflated_preconditioner_used",
    "qi_deflated_preconditioner_used_in_krylov",
    "qi_deflated_stats",
    "assembled_operator_device_resident",
    "fgmres_block_between_cycles",
    "tfqmr_replacement_interval",
    "xblock_device_fgmres_forced_right_pc",
    "xblock_device_fgmres_jit",
    "xblock_device_fgmres_jit_mode",
    "xblock_device_fgmres_jit_outer_k",
    "xblock_device_host_fallback_auto_disabled_by_qi_device",
    "xblock_device_host_fallback_decision",
    "xblock_device_krylov_forced_jax_factors",
    "xblock_krylov_env_requested",
    "xblock_qi_device_operator_reuse_decision",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_PREFLIGHT_STATE_KEYS = (
    "preflight_improvement",
    "preflight_min_improvement",
    "preflight_passed",
    "preflight_required",
    "preflight_residual_norm",
    "probe_coarse_angular_lmax",
    "probe_coarse_direction_counts",
    "probe_coarse_direction_names",
    "probe_coarse_fsavg_lmax",
    "probe_coarse_history",
    "probe_coarse_include_angular_residual",
    "probe_coarse_residual_after",
    "probe_coarse_residual_before",
    "probe_coarse_s",
    "probe_coarse_seed_initialized",
    "probe_coarse_steps_requested",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_COMPACT_CORE_STATE_KEYS = (
    "candidate_iterations",
    "candidate_krylov_method",
    "candidate_matvecs",
    "candidate_residual_norm",
    "device_krylov_estimated_matvecs",
    "fallback_candidate_improved_rhs",
    "fallback_started_from_candidate",
    "mv_count",
    "pc_factor_s",
    "pc_maxiter",
    "pc_restart",
    "precondition_side",
    "reported_iterations",
    "reported_matvecs",
    "setup_s",
    "solve_s",
    "sparse_timer",
    "xblock_assembled_host_fp",
    "xblock_default_restart_capped",
    "xblock_default_right_pc",
    "xblock_jax_factor_apply",
    "xblock_jax_factor_format",
    "xblock_jax_factors",
    "xblock_krylov_method",
    "xblock_linear_size",
    "xblock_lower_fill_ignored_env",
    "xblock_lower_fill_mode",
    "xblock_preconditioner_built",
    "xblock_preconditioner_xi",
    "xblock_use_active_dof",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_DEVICE_STATE_KEYS = (
    "assembled_operator_built",
    "assembled_operator_device_resident",
    "fgmres_block_between_cycles",
    "global_coupling_built",
    "global_coupling_metadata",
    "qi_device_augmented_krylov_mode",
    "qi_device_augmented_krylov_rank",
    "qi_device_augmented_krylov_reason",
    "qi_device_augmented_krylov_requested",
    "qi_device_augmented_krylov_used",
    "qi_device_augmented_seed_available",
    "qi_device_augmented_seed_labels",
    "qi_device_augmented_seed_max_rank",
    "qi_device_augmented_seed_projection_residual",
    "qi_device_augmented_seed_rank",
    "qi_device_augmented_seed_reason",
    "qi_device_augmented_seed_requested",
    "qi_device_augmented_seed_used",
    "tfqmr_replacement_interval",
    "two_level_built",
    "xblock_device_fgmres_forced_right_pc",
    "xblock_device_fgmres_jit",
    "xblock_device_fgmres_jit_mode",
    "xblock_device_fgmres_jit_outer_k",
    "xblock_device_host_fallback_auto_disabled_by_qi_device",
    "xblock_device_host_fallback_decision",
    "xblock_device_krylov_forced_jax_factors",
    "xblock_krylov_env_requested",
    "xblock_qi_device_operator_reuse_decision",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_PRECOMPUTED_KEYS = (
    "xblock_assembled_operator_result_metadata",
    "xblock_coarse_correction_metadata",
    "xblock_qi_seed_preconditioner_metadata",
    "xblock_qi_device_preconditioner_metadata",
    "xblock_qi_deflated_preconditioner_metadata",
    "xblock_side_probe_metadata",
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_STATE_KEYS = _unique_state_keys(
    _XBLOCK_SPARSE_PC_FINAL_METADATA_COMPACT_CORE_STATE_KEYS,
    _XBLOCK_SPARSE_PC_FINAL_METADATA_DEVICE_STATE_KEYS,
    _XBLOCK_SPARSE_PC_FINAL_METADATA_PREFLIGHT_STATE_KEYS,
)

_XBLOCK_SPARSE_PC_FINAL_METADATA_SCOPE_KEYS = _unique_state_keys(
    _XBLOCK_SPARSE_PC_FINAL_METADATA_CORE_STATE_KEYS,
    _XBLOCK_SPARSE_PC_FINAL_METADATA_NESTED_STATE_KEYS,
    _XBLOCK_SPARSE_PC_FINAL_METADATA_PREFLIGHT_STATE_KEYS,
)


@dataclass(frozen=True)
class XBlockSparsePCFinalCoreState:
    """Core x-block solve counters and user-facing solver controls."""

    candidate_iterations: object
    candidate_krylov_method: object
    candidate_matvecs: object
    candidate_residual_norm: object
    device_krylov_estimated_matvecs: object
    fallback_candidate_improved_rhs: object
    fallback_started_from_candidate: object
    mv_count: object
    pc_factor_s: object
    pc_maxiter: object
    pc_restart: object
    precondition_side: object
    reported_iterations: object
    reported_matvecs: object
    setup_s: object
    solve_s: object
    sparse_timer: object
    xblock_assembled_host_fp: object
    xblock_default_restart_capped: object
    xblock_default_right_pc: object
    xblock_jax_factor_apply: object
    xblock_jax_factor_format: object
    xblock_jax_factors: object
    xblock_krylov_method: object
    xblock_linear_size: object
    xblock_lower_fill_ignored_env: object
    xblock_lower_fill_mode: object
    xblock_preconditioner_built: object
    xblock_preconditioner_xi: object
    xblock_use_active_dof: object


@dataclass(frozen=True)
class XBlockSparsePCFinalDeviceState:
    """Device, QI, and global-coupling state for x-block diagnostics."""

    assembled_operator_built: object
    assembled_operator_device_resident: object
    fgmres_block_between_cycles: object
    global_coupling_built: object
    global_coupling_metadata: object
    qi_device_augmented_krylov_mode: object
    qi_device_augmented_krylov_rank: object
    qi_device_augmented_krylov_reason: object
    qi_device_augmented_krylov_requested: object
    qi_device_augmented_krylov_used: object
    qi_device_augmented_seed_available: object
    qi_device_augmented_seed_labels: object
    qi_device_augmented_seed_max_rank: object
    qi_device_augmented_seed_projection_residual: object
    qi_device_augmented_seed_rank: object
    qi_device_augmented_seed_reason: object
    qi_device_augmented_seed_requested: object
    qi_device_augmented_seed_used: object
    tfqmr_replacement_interval: object
    two_level_built: object
    xblock_device_fgmres_forced_right_pc: object
    xblock_device_fgmres_jit: object
    xblock_device_fgmres_jit_mode: object
    xblock_device_fgmres_jit_outer_k: object
    xblock_device_host_fallback_auto_disabled_by_qi_device: object
    xblock_device_host_fallback_decision: object
    xblock_device_krylov_forced_jax_factors: object
    xblock_krylov_env_requested: object
    xblock_qi_device_operator_reuse_decision: object


@dataclass(frozen=True)
class XBlockSparsePCFinalPreflightState:
    """Pre-Krylov probe and residual-gate state for x-block diagnostics."""

    preflight_improvement: object
    preflight_min_improvement: object
    preflight_passed: object
    preflight_required: object
    preflight_residual_norm: object
    probe_coarse_angular_lmax: object
    probe_coarse_direction_counts: object
    probe_coarse_direction_names: object
    probe_coarse_fsavg_lmax: object
    probe_coarse_history: object
    probe_coarse_include_angular_residual: object
    probe_coarse_residual_after: object
    probe_coarse_residual_before: object
    probe_coarse_s: object
    probe_coarse_seed_initialized: object
    probe_coarse_steps_requested: object


@dataclass(frozen=True)
class XBlockSparsePCFinalNestedMetadata:
    """Precomputed nested x-block diagnostic groups."""

    xblock_assembled_operator_result_metadata: object
    xblock_coarse_correction_metadata: object
    xblock_qi_seed_preconditioner_metadata: object
    xblock_qi_device_preconditioner_metadata: object
    xblock_qi_deflated_preconditioner_metadata: object
    xblock_side_probe_metadata: object


@dataclass(frozen=True)
class XBlockSparsePCFinalMetadataStateContext:
    """Grouped state used to build final x-block sparse-PC metadata."""

    core: XBlockSparsePCFinalCoreState
    device: XBlockSparsePCFinalDeviceState
    preflight: XBlockSparsePCFinalPreflightState
    nested: XBlockSparsePCFinalNestedMetadata


def _dataclass_field_mapping(value: object) -> dict[str, object]:
    return {field.name: getattr(value, field.name) for field in fields(value)}


def xblock_sparse_pc_final_metadata_driver_state_keys() -> tuple[str, ...]:
    """Return driver-scope keys copied into x-block final metadata."""

    return _XBLOCK_SPARSE_PC_FINAL_METADATA_STATE_KEYS


def xblock_sparse_pc_final_metadata_driver_scope_keys() -> tuple[str, ...]:
    """Return raw driver-scope keys needed to derive x-block final metadata."""

    return _XBLOCK_SPARSE_PC_FINAL_METADATA_SCOPE_KEYS


def _xblock_metadata_or_compute(
    scope: Mapping[str, object],
    key: str,
    builder: Callable[[Mapping[str, object]], dict[str, object]],
) -> object:
    if key in scope:
        return scope[key]
    return builder(scope)


def xblock_sparse_pc_final_metadata_state_from_context(
    context: XBlockSparsePCFinalMetadataStateContext,
) -> dict[str, object]:
    """Return the compact final x-block diagnostic state from typed groups."""

    raw = {
        **_dataclass_field_mapping(context.core),
        **_dataclass_field_mapping(context.device),
        **_dataclass_field_mapping(context.preflight),
        **_dataclass_field_mapping(context.nested),
    }
    missing = tuple(
        key
        for key in (
            *_XBLOCK_SPARSE_PC_FINAL_METADATA_STATE_KEYS,
            *_XBLOCK_SPARSE_PC_FINAL_METADATA_PRECOMPUTED_KEYS,
        )
        if key not in raw
    )
    if missing:
        joined = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise KeyError(f"x-block sparse-PC final metadata missing: {joined}{suffix}")
    return {
        **{key: raw[key] for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_STATE_KEYS},
        **{
            key: raw[key]
            for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_PRECOMPUTED_KEYS
        },
    }


def xblock_sparse_pc_final_metadata_state_from_driver_scope(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Copy compact x-block final state and precompute nested diagnostics."""

    missing = tuple(
        key for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_STATE_KEYS if key not in scope
    )
    if missing:
        joined = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise KeyError(f"x-block sparse-PC final metadata missing: {joined}{suffix}")
    nested = XBlockSparsePCFinalNestedMetadata(
        xblock_assembled_operator_result_metadata=_xblock_metadata_or_compute(
        scope,
        "xblock_assembled_operator_result_metadata",
        lambda raw: xblock_assembled_operator_diagnostics(
            XBlockAssembledOperatorDiagnosticsContext(
                enabled=raw["assembled_operator_enabled"],
                built=raw["assembled_operator_built"],
                metadata=raw["assembled_operator_metadata"],
                row_equilibration_enabled=raw["xblock_row_equilibration_enabled"],
                row_equilibration_built=raw["xblock_row_equilibration_built"],
                row_equilibration_metadata=raw["xblock_row_equilibration_metadata"],
                col_equilibration_enabled=raw["xblock_col_equilibration_enabled"],
                col_equilibration_built=raw["xblock_col_equilibration_built"],
                col_equilibration_metadata=raw["xblock_col_equilibration_metadata"],
            )
        ),
        ),
        xblock_coarse_correction_metadata=_xblock_metadata_or_compute(
            scope,
            "xblock_coarse_correction_metadata",
            xblock_coarse_correction_diagnostics,
        ),
        xblock_qi_seed_preconditioner_metadata=_xblock_metadata_or_compute(
            scope,
            "xblock_qi_seed_preconditioner_metadata",
            xblock_qi_seed_preconditioner_diagnostics,
        ),
        xblock_qi_device_preconditioner_metadata=_xblock_metadata_or_compute(
            scope,
            "xblock_qi_device_preconditioner_metadata",
            xblock_qi_device_preconditioner_diagnostics,
        ),
        xblock_qi_deflated_preconditioner_metadata=_xblock_metadata_or_compute(
            scope,
            "xblock_qi_deflated_preconditioner_metadata",
            xblock_qi_deflated_preconditioner_diagnostics,
        ),
        xblock_side_probe_metadata=_xblock_metadata_or_compute(
            scope,
            "xblock_side_probe_metadata",
            lambda raw: xblock_side_probe_diagnostics(
                XBlockSideProbeDiagnosticsContext(
                    enabled=raw["xblock_side_probe_enabled"],
                    used=raw["xblock_side_probe_used"],
                    switched=raw["xblock_side_probe_switched"],
                    switch_suppressed_by_global_coupling=raw[
                        "xblock_side_probe_switch_suppressed_by_global_coupling"
                    ],
                    switch_suppressed_by_explicit_side=raw[
                        "xblock_side_probe_switch_suppressed_by_explicit_side"
                    ],
                    physical_seed_preserved_after_switch=raw[
                        "xblock_side_probe_physical_seed_preserved_after_switch"
                    ],
                    seed_used=raw["xblock_side_probe_seed_used"],
                    seed_residual_norm=raw["xblock_side_probe_seed_residual_norm"],
                    initial_side=raw["xblock_side_probe_initial_side"],
                    selected_side=raw["xblock_side_probe_selected_side"],
                    initial_method=raw["xblock_side_probe_initial_method"],
                    selected_method=raw["xblock_side_probe_selected_method"],
                    lgmres_rescue=raw["xblock_side_probe_lgmres_rescue"],
                    lgmres_rescue_maxiter_capped=raw[
                        "xblock_lgmres_rescue_maxiter_capped"
                    ],
                    lgmres_rescue_outer_k=raw["xblock_lgmres_rescue_outer_k"],
                    residual_norm=raw["xblock_side_probe_residual_norm"],
                    residual_ratio=raw["xblock_side_probe_residual_ratio"],
                    iterations=raw["xblock_side_probe_iterations"],
                    matvecs=raw["xblock_side_probe_matvecs"],
                    elapsed_s=raw["xblock_side_probe_s"],
                )
            ),
        ),
    )
    return xblock_sparse_pc_final_metadata_state_from_context(
        XBlockSparsePCFinalMetadataStateContext(
            core=XBlockSparsePCFinalCoreState(
                **{
                    key: scope[key]
                    for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_COMPACT_CORE_STATE_KEYS
                }
            ),
            device=XBlockSparsePCFinalDeviceState(
                **{
                    key: scope[key]
                    for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_DEVICE_STATE_KEYS
                }
            ),
            preflight=XBlockSparsePCFinalPreflightState(
                **{
                    key: scope[key]
                    for key in _XBLOCK_SPARSE_PC_FINAL_METADATA_PREFLIGHT_STATE_KEYS
                }
            ),
            nested=nested,
        )
    )



@dataclass(frozen=True)
class XBlockKrylovReport:
    """Reported xblock Krylov work counters after optional device execution."""

    iterations: int
    matvecs: int


@dataclass(frozen=True)
class XBlockSparsePCCompletionContext:
    """Explicit inputs for the final xblock sparse-PC progress line."""

    emit: EmitFn | None
    krylov_method: str
    elapsed_s: float
    iterations: int
    matvecs: int
    residual_norm: float
    target: float
    history: Sequence[float] | None


@dataclass(frozen=True)
class XBlockSparsePCFinalPayloadContext:
    """Explicit inputs for finalizing the xblock sparse-PC payload."""

    op: object
    x: np.ndarray
    residual_norm: float
    target: float
    krylov_method: str
    linear_size: int | None
    restart: int | None
    diagnostic_state: Mapping[str, object]
    post_corrections: object | None = None


def xblock_sparse_pc_final_metadata_from_driver_state(
    state: Mapping[str, object],
    *,
    full_size: object,
) -> dict[str, object]:
    """Build final x-block sparse-PC metadata from one driver-state handoff."""

    return {
        **xblock_sparse_pc_result_diagnostics_from_driver_state(
            state,
            full_size=full_size,
        ),
        **build_rhs1_xblock_correction_metadata_from_driver_state(state),
    }


def xblock_sparse_pc_final_payload_from_driver_state(
    state: Mapping[str, object],
    *,
    expand_reduced: ArrayFn,
    post_corrections: object | None = None,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the x-block sparse-PC branch from driver state."""

    return xblock_sparse_pc_final_payload(
        XBlockSparsePCFinalPayloadContext(
            op=state["op"],
            x=np.asarray(state["x_np"], dtype=np.float64),
            residual_norm=float(state["residual_norm_xblock_pc"]),
            target=float(state["target_xblock"]),
            krylov_method=str(state["xblock_krylov_method"]),
            linear_size=(
                int(state["xblock_linear_size"])
                if "xblock_linear_size" in state
                else None
            ),
            restart=int(state["pc_restart"]) if "pc_restart" in state else None,
            diagnostic_state=state,
            post_corrections=post_corrections,
        ),
        expand_reduced=expand_reduced,
    )


def xblock_sparse_pc_final_payload(
    context: XBlockSparsePCFinalPayloadContext,
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the x-block sparse-PC branch."""

    residual_norm = float(context.residual_norm)
    metadata_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    metadata_state.update(
        {
            "op": context.op,
            "x_np": np.asarray(context.x, dtype=np.float64),
            "residual_norm_xblock_pc": residual_norm,
            "target_xblock": float(context.target),
            "xblock_krylov_method": str(context.krylov_method),
        }
    )
    if context.linear_size is not None:
        metadata_state["xblock_linear_size"] = int(context.linear_size)
    if context.restart is not None:
        metadata_state["pc_restart"] = int(context.restart)
    if context.post_corrections is not None:
        metadata_state.update(context.post_corrections.driver_state())
    if (
        "xblock_solver_kind" not in metadata_state
        and context.linear_size is not None
        and context.restart is not None
    ):
        work_estimates = xblock_sparse_pc_work_estimates(
            krylov_method=str(context.krylov_method),
            linear_size=int(context.linear_size),
            restart=int(context.restart),
            dtype=np.float64,
        )
        metadata_state.update(
            {
                "xblock_solver_kind": work_estimates.solver_kind,
                "xblock_device_krylov_methods": set(work_estimates.device_krylov_methods),
                "xblock_estimated_gmres_basis_nbytes": work_estimates.gmres_basis_nbytes,
                "xblock_estimated_bicgstab_work_nbytes": work_estimates.bicgstab_work_nbytes,
                "xblock_estimated_tfqmr_work_nbytes": work_estimates.tfqmr_work_nbytes,
            }
        )
    metadata_state["accepted_converged_xblock"] = profile_residual_converged(
        residual_norm,
        float(context.target),
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(context.x, dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=xblock_sparse_pc_final_metadata_from_driver_state(
            metadata_state,
            full_size=getattr(context.op, "total_size"),
        ),
    )


@dataclass(frozen=True)
class XBlockGMRESFallbackDecision:
    """Admission result for a non-GMRES xblock solve retrying with GMRES."""

    run: bool


@dataclass(frozen=True)
class XBlockGMRESFallbackContext:
    """Inputs for retrying a failed non-GMRES xblock solve with GMRES."""

    krylov_method: str
    fallback_enabled: bool
    x_solution: np.ndarray
    x_physical: np.ndarray
    residual_norm: float
    history: Sequence[float] | None
    solve_s: float
    target: float
    rhs_norm: float
    original_x0: jnp.ndarray | None
    solve_rhs: jnp.ndarray
    solve_matvec: ArrayFn
    solve_preconditioner: ArrayFn | None
    precondition_side: str
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    progress_callback: Callable[[int, float], None] | None
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    initial_guess_builder: Callable[..., tuple[jnp.ndarray | None, bool, bool]]
    solution_to_physical: Callable[[jnp.ndarray], jnp.ndarray]
    physical_rhs: jnp.ndarray
    physical_matvec: ArrayFn
    device_iterations: int | None = None
    device_estimated_matvecs: int | None = None


@dataclass(frozen=True)
class XBlockGMRESFallbackResult:
    """Updated xblock solve state after optional GMRES fallback."""

    krylov_method: str
    x_solution: np.ndarray
    x_physical: np.ndarray
    residual_norm: float
    history: tuple[float, ...]
    solve_s: float
    device_iterations: int | None
    device_estimated_matvecs: int | None
    fallback_started_from_candidate: bool
    fallback_candidate_improved_rhs: bool


@dataclass(frozen=True)
class XBlockDeviceKrylovState:
    """Host-side arrays and counters from a device xblock Krylov solve."""

    x: np.ndarray
    residual_norm: float
    history: tuple[float, ...]
    n_iterations: int
    estimated_matvecs: int | None


@dataclass(frozen=True)
class XBlockFirstKrylovAttemptContext:
    """Inputs for the first xblock sparse-PC Krylov attempt."""

    krylov_method: str
    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn | None
    x0: jnp.ndarray | None
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    lgmres_outer_k: int | None
    fgmres_block_between_cycles: bool
    skip_inactive_work: bool
    device_fgmres_jit: bool
    device_fgmres_jit_mode: str
    device_fgmres_jit_outer_k: int
    augmented_krylov_used: bool
    augmentation_basis: jnp.ndarray | None
    operator_on_augmentation: jnp.ndarray | None
    augmentation_mode: str
    tfqmr_replacement_interval: int
    mv_count: int
    host_progress_callback: Callable[[int, float], None] | None
    device_cycle_progress_callback: Callable[..., None] | None
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    lgmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    gcrotmk_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    bicgstab_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    fgmres_solver: Callable[..., tuple[object, object]]
    fgmres_jit_solver: Callable[..., tuple[object, object]]
    fgmres_cycle_jit_solver: Callable[..., tuple[object, object]]
    bicgstab_jax_solver: Callable[..., tuple[object, object]]
    tfqmr_jax_solver: Callable[..., tuple[object, object]]


@dataclass(frozen=True)
class XBlockFirstKrylovAttemptResult:
    """Result from the first xblock sparse-PC Krylov attempt."""

    x: np.ndarray
    residual_norm: float
    history: tuple[float, ...]
    device_iterations: int | None
    device_estimated_matvecs: int | None


@dataclass(frozen=True)
class XBlockSideProbeStageContext:
    """Inputs for the bounded precondition-side probe before the main x-block solve."""

    controls: object
    precondition_side: str
    krylov_method: str
    pc_maxiter: int | None
    side_env: str
    global_coupling_built: bool
    matvec: ArrayFn
    true_matvec_no_count: ArrayFn
    rhs: jnp.ndarray
    rhs_norm: float
    target: float
    preconditioner: ArrayFn
    x0: jnp.ndarray | None
    tol: float
    atol: float
    elapsed_s: Callable[[], float]
    matvec_count: Callable[[], int]
    emit: EmitFn | None
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


@dataclass(frozen=True)
class XBlockSideProbeStageResult:
    """Updated solve state and diagnostics from the bounded side probe."""

    x0: jnp.ndarray | None
    precondition_side: str
    krylov_method: str
    pc_maxiter: int | None
    enabled: bool
    used: bool
    switched: bool
    initial_side: str | None
    selected_side: str | None
    initial_method: str | None
    selected_method: str | None
    lgmres_rescue: bool
    lgmres_rescue_maxiter_capped: bool
    lgmres_rescue_outer_k: int | None
    residual_norm: float | None
    residual_ratio: float | None
    iterations: int
    matvecs: int
    elapsed_s: float
    switch_suppressed_by_global_coupling: bool
    switch_suppressed_by_explicit_side: bool
    physical_seed_preserved_after_switch: bool
    seed_used: bool
    seed_residual_norm: float | None
    failed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class XBlockProbeCoarseStageContext:
    """Inputs for the optional pre-Krylov projected coarse seed correction."""

    policy: object
    rhs: jnp.ndarray
    x0: jnp.ndarray | None
    matvec: ArrayFn
    target: float
    direction_builder: Callable[..., tuple[tuple[str, jnp.ndarray], ...]]
    correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[int], Sequence[str]]]
    elapsed_s: Callable[[], float]
    emit: EmitFn | None


@dataclass(frozen=True)
class XBlockProbeCoarseStageResult:
    """Updated seed and diagnostics from the optional probe-coarse stage."""

    x0: jnp.ndarray | None
    steps_requested: int
    max_directions: int
    max_extra_units: int
    fsavg_lmax: int
    angular_lmax: int
    include_angular_residual: bool
    include_raw: bool
    alpha_clip: float
    rcond: float
    min_improvement: float
    elapsed_s: float
    history: tuple[float, ...]
    direction_counts: tuple[int, ...]
    direction_names: tuple[str, ...]
    residual_before: float | None
    residual_after: float | None
    seed_initialized: bool
    improved: bool
    failed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class XBlockPreflightGateContext:
    """Inputs for the optional x-block seed residual preflight gate."""

    min_improvement: float
    required: bool
    rhs: jnp.ndarray
    rhs_norm: float
    x0: jnp.ndarray | None
    matvec: ArrayFn
    target: float
    emit: EmitFn | None


@dataclass(frozen=True)
class XBlockPreflightGateResult:
    """Diagnostics from the optional x-block seed residual preflight gate."""

    residual_norm: float | None
    improvement: float | None
    passed: bool | None
    evaluated: bool
    failed: bool
    failure_reason: str | None


@dataclass(frozen=True)
class XBlockKrylovControlSetupContext:
    """Inputs for resolving x-block Krylov runtime controls and messages."""

    env: Mapping[str, str] | None
    krylov_method: str
    pc_restart: int
    pc_maxiter: int | None
    precondition_side: str
    emit: EmitFn | None


@dataclass(frozen=True)
class XBlockKrylovControlSetup:
    """Resolved x-block Krylov controls for the first solve attempt."""

    fgmres_block_between_cycles: bool
    tfqmr_replacement_interval: int
    device_fgmres_jit: bool
    device_fgmres_jit_mode: str
    device_fgmres_jit_outer_k: int
    qi_device_augmented_krylov_requested: bool
    qi_device_augmented_krylov_mode: str


@dataclass(frozen=True)
class XBlockKrylovProgressCallbacksContext:
    """Inputs for x-block Krylov host/device progress callbacks."""

    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    progress_every: int


@dataclass(frozen=True)
class XBlockKrylovProgressCallbacks:
    """Host and device progress callbacks passed to the first Krylov attempt."""

    host_progress_callback: Callable[[int, float], None]
    device_cycle_progress_callback: Callable[..., None]


@dataclass(frozen=True)
class XBlockKrylovSolveState:
    """Physical-space xblock Krylov solve state used by downstream metadata."""

    krylov_method: str
    x_solution: np.ndarray
    x_physical: np.ndarray
    residual_norm: float
    history: tuple[float, ...]
    solve_s: float
    device_iterations: int | None
    device_estimated_matvecs: int | None
    reported_iterations: int
    reported_matvecs: int
    fallback_started_from_candidate: bool = False
    fallback_candidate_improved_rhs: bool = False


@dataclass(frozen=True)
class XBlockFirstKrylovSolveStateContext:
    """Inputs for converting a first xblock Krylov attempt to physical state."""

    krylov_method: str
    first_attempt: XBlockFirstKrylovAttemptResult
    solve_s: float
    solution_to_physical: ArrayFn
    physical_rhs: jnp.ndarray
    physical_matvec: ArrayFn
    mv_count: int


@dataclass(frozen=True)
class XBlockKrylovSolveStageContext:
    """Inputs for first x-block Krylov attempt plus optional GMRES fallback."""

    first_attempt: XBlockFirstKrylovAttemptContext
    solve_start_s: float
    side_probe_s: float
    probe_coarse_s: float
    elapsed_s: Callable[[], float]
    solution_to_physical: ArrayFn
    physical_rhs: jnp.ndarray
    physical_matvec: ArrayFn
    target: float
    rhs_norm: float
    fallback_enabled: bool
    progress_callback: Callable[[int, float], None] | None
    emit: EmitFn | None
    initial_guess_builder: Callable[..., tuple[jnp.ndarray | None, bool, bool]]


@dataclass(frozen=True)
class XBlockKrylovSolveStageResult:
    """Candidate and final x-block Krylov state after optional GMRES fallback."""

    first_attempt: XBlockFirstKrylovAttemptResult
    fallback: XBlockGMRESFallbackResult
    candidate_state: XBlockKrylovSolveState
    final_state: XBlockKrylovSolveState


@dataclass(frozen=True)
class XBlockKrylovSolveSpaceContext:
    """Prepared physical/equilibrated xblock Krylov solve-space inputs."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn | None
    x0: jnp.ndarray | None
    precondition_side: str
    row_equilibration_built: bool
    col_equilibration_built: bool
    row_scale: jnp.ndarray | None
    inv_row_scale: jnp.ndarray | None
    col_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None


@dataclass(frozen=True)
class XBlockKrylovSolveSpace:
    """Krylov solve-space callbacks after optional row/column equilibration."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn | None
    x0: jnp.ndarray | None
    solution_to_physical: ArrayFn
    transform_label: str | None


@dataclass(frozen=True)
class XBlockAugmentedKrylovBasisContext:
    """Inputs for preparing a QI augmented Krylov basis in solve coordinates."""

    krylov_method: str
    qi_device_state: object | None
    seed_available: bool
    seed_rank: int
    seed_basis: jnp.ndarray | None
    seed_operator_on_basis: jnp.ndarray | None
    row_equilibration_built: bool
    col_equilibration_built: bool
    row_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None
    precondition_side: str
    solve_preconditioner: ArrayFn | None


@dataclass(frozen=True)
class XBlockAugmentedKrylovBasisResult:
    """Prepared QI augmented Krylov basis and diagnostic state."""

    basis: jnp.ndarray | None
    operator_on_basis: jnp.ndarray | None
    used: bool
    rank: int
    reason: str
    seed_used: bool


@dataclass(frozen=True)
class XBlockAugmentedKrylovStageContext:
    """Inputs for optional QI augmented-Krylov solve setup and diagnostics."""

    requested: bool
    krylov_method: str
    qi_device_state: object | None
    seed_available: bool
    seed_rank: int
    seed_basis: jnp.ndarray | None
    seed_operator_on_basis: jnp.ndarray | None
    seed_used: bool
    row_equilibration_built: bool
    col_equilibration_built: bool
    row_scale: jnp.ndarray | None
    inv_col_scale: jnp.ndarray | None
    precondition_side: str
    solve_preconditioner: ArrayFn | None
    mode: str
    metadata: Mapping[str, object]
    emit: EmitFn | None
    basis_builder: Callable[[XBlockAugmentedKrylovBasisContext], XBlockAugmentedKrylovBasisResult]


@dataclass(frozen=True)
class XBlockAugmentedKrylovStageResult:
    """Optional QI augmented-Krylov basis and updated diagnostic metadata."""

    basis: jnp.ndarray | None
    operator_on_basis: jnp.ndarray | None
    used: bool
    rank: int
    reason: str | None
    seed_used: bool
    metadata: dict[str, object]


@dataclass(frozen=True)
class XBlockSparsePCWorkEstimates:
    """User-facing solver-kind and Krylov work-memory estimates."""

    solver_kind: str
    device_krylov_methods: frozenset[str]
    gmres_basis_nbytes: int
    bicgstab_work_nbytes: int
    tfqmr_work_nbytes: int


@dataclass(frozen=True)
class XBlockPhysicalResidual:
    """Physical-space xblock solution and true residual norm."""

    x_physical: np.ndarray
    residual_norm: float


def xblock_krylov_report(
    *,
    device_iterations: int | None,
    device_estimated_matvecs: int | None,
    history: Sequence[float] | None,
    mv_count: int,
) -> XBlockKrylovReport:
    """Return the xblock Krylov iteration/matvec counters reported to users."""

    iterations = int(device_iterations) if device_iterations is not None else int(len(history or ()))
    matvecs = int(device_estimated_matvecs) if device_estimated_matvecs is not None else int(mv_count)
    return XBlockKrylovReport(iterations=int(iterations), matvecs=int(matvecs))


def apply_xblock_side_probe_stage(
    context: XBlockSideProbeStageContext,
) -> XBlockSideProbeStageResult:
    """Run the bounded x-block precondition-side probe and return updated state."""

    controls = context.controls
    enabled = bool(getattr(controls, "enabled", False))
    x0 = context.x0
    precondition_side = str(context.precondition_side)
    krylov_method = str(context.krylov_method)
    pc_maxiter = context.pc_maxiter
    used = False
    switched = False
    initial_side: str | None = None
    selected_side: str | None = None
    initial_method: str | None = None
    selected_method: str | None = None
    lgmres_rescue = False
    lgmres_rescue_maxiter_capped = False
    lgmres_rescue_outer_k: int | None = None
    residual_norm: float | None = None
    residual_ratio: float | None = None
    iterations = 0
    matvecs = 0
    elapsed_s = 0.0
    switch_suppressed_by_global_coupling = False
    switch_suppressed_by_explicit_side = False
    physical_seed_preserved_after_switch = False
    seed_used = False
    seed_residual_norm: float | None = None
    failed = False
    failure_reason: str | None = None

    if not enabled:
        return XBlockSideProbeStageResult(
            x0=x0,
            precondition_side=precondition_side,
            krylov_method=krylov_method,
            pc_maxiter=pc_maxiter,
            enabled=False,
            used=False,
            switched=False,
            initial_side=None,
            selected_side=None,
            initial_method=None,
            selected_method=None,
            lgmres_rescue=False,
            lgmres_rescue_maxiter_capped=False,
            lgmres_rescue_outer_k=None,
            residual_norm=None,
            residual_ratio=None,
            iterations=0,
            matvecs=0,
            elapsed_s=0.0,
            switch_suppressed_by_global_coupling=False,
            switch_suppressed_by_explicit_side=False,
            physical_seed_preserved_after_switch=False,
            seed_used=False,
            seed_residual_norm=None,
            failed=False,
            failure_reason=None,
        )

    used = True
    initial_side = precondition_side
    initial_method = krylov_method
    probe_restart = int(getattr(controls, "restart"))
    probe_maxiter = int(getattr(controls, "maxiter"))
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe start "
            f"side={precondition_side} restart={int(probe_restart)} maxiter={int(probe_maxiter)}",
        )
    probe_start_s = float(context.elapsed_s())
    probe_start_mv = int(context.matvec_count())
    try:
        x_probe, residual_probe, history_probe = context.gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=probe_restart,
            maxiter=probe_maxiter,
            precondition_side=precondition_side,
        )
        elapsed_s = float(context.elapsed_s()) - probe_start_s
        matvecs = int(context.matvec_count()) - int(probe_start_mv)
        iterations = int(len(history_probe or []))
        residual_norm = float(residual_probe)
        residual_ratio = profile_safe_ratio(residual_norm, context.target)
        incumbent_seed_norm = float(context.rhs_norm)
        if x0 is not None:
            try:
                incumbent_residual = context.rhs - jnp.asarray(
                    context.true_matvec_no_count(jnp.asarray(x0, dtype=jnp.float64)),
                    dtype=jnp.float64,
                )
                incumbent_seed_norm = profile_l2_norm_float(incumbent_residual)
            except Exception:
                incumbent_seed_norm = float(context.rhs_norm)
        if str(precondition_side) == "left" and np.isfinite(float(residual_probe)):
            # The left-preconditioned side probe returns a physical-space state,
            # so it can seed a later side switch.
            x0 = jnp.asarray(x_probe, dtype=jnp.float64)
            seed_used = True
            seed_residual_norm = float(residual_probe)
        elif (
            np.isfinite(float(residual_probe))
            and float(residual_probe) < float(incumbent_seed_norm)
        ):
            x0 = jnp.asarray(x_probe, dtype=jnp.float64)
            seed_used = True
            seed_residual_norm = float(residual_probe)

        should_switch_side = bool(controls.should_switch(residual_ratio))
        if should_switch_side and context.side_env in {"left", "right", "none"}:
            should_switch_side = False
            switch_suppressed_by_explicit_side = True
        lgmres_rescue_enabled = bool(getattr(controls, "lgmres_rescue_enabled"))
        if (
            should_switch_side
            and bool(context.global_coupling_built)
            and (not bool(lgmres_rescue_enabled))
            and str(precondition_side) == "left"
        ):
            keep_left_ratio = float(getattr(controls, "global_coupling_keep_left_ratio"))
            if (
                residual_ratio is not None
                and np.isfinite(float(residual_ratio))
                and float(residual_ratio) <= float(keep_left_ratio)
            ):
                should_switch_side = False
                switch_suppressed_by_global_coupling = True
        if should_switch_side and lgmres_rescue_enabled and str(precondition_side) == "left":
            krylov_method = "lgmres"
            lgmres_rescue = True
            pc_maxiter = int(getattr(controls, "lgmres_rescue_maxiter"))
            lgmres_rescue_maxiter_capped = bool(
                getattr(controls, "lgmres_rescue_maxiter_capped")
            )
            lgmres_rescue_outer_k = int(getattr(controls, "lgmres_rescue_outer_k"))
        elif should_switch_side:
            precondition_side = "right" if str(precondition_side) == "left" else "left"
            switched = True
            if str(precondition_side) == "right" and x0 is not None:
                physical_seed_preserved_after_switch = True
        selected_side = str(precondition_side)
        selected_method = str(krylov_method)
        if context.emit is not None:
            if lgmres_rescue:
                action = "method_rescue"
            elif switch_suppressed_by_explicit_side:
                action = "keep_explicit_side"
            elif switch_suppressed_by_global_coupling:
                action = "keep_global_coupling"
            else:
                action = "switch" if switched else "keep"
            ratio_for_message = (
                float(residual_ratio) if residual_ratio is not None else float("nan")
            )
            residual_for_message = (
                float(residual_norm) if residual_norm is not None else float("nan")
            )
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres side probe "
                f"{action} side={initial_side}->{selected_side} "
                f"method={initial_method}->{selected_method} "
                f"iters={iterations} matvecs={matvecs} "
                f"residual={residual_for_message:.6e} "
                f"ratio={ratio_for_message:.6e}"
                + (" seed_used=1" if seed_used else "")
                + (
                    " preserved_physical_seed=1"
                    if physical_seed_preserved_after_switch
                    else ""
                ),
            )
    except Exception as exc:  # noqa: BLE001
        elapsed_s = float(context.elapsed_s()) - probe_start_s
        selected_side = str(precondition_side)
        selected_method = str(krylov_method)
        failed = True
        failure_reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"side probe failed ({type(exc).__name__}: {exc}); keeping side={precondition_side}",
            )

    return XBlockSideProbeStageResult(
        x0=x0,
        precondition_side=precondition_side,
        krylov_method=krylov_method,
        pc_maxiter=pc_maxiter,
        enabled=True,
        used=used,
        switched=switched,
        initial_side=initial_side,
        selected_side=selected_side,
        initial_method=initial_method,
        selected_method=selected_method,
        lgmres_rescue=lgmres_rescue,
        lgmres_rescue_maxiter_capped=lgmres_rescue_maxiter_capped,
        lgmres_rescue_outer_k=lgmres_rescue_outer_k,
        residual_norm=residual_norm,
        residual_ratio=residual_ratio,
        iterations=iterations,
        matvecs=matvecs,
        elapsed_s=float(elapsed_s),
        switch_suppressed_by_global_coupling=switch_suppressed_by_global_coupling,
        switch_suppressed_by_explicit_side=switch_suppressed_by_explicit_side,
        physical_seed_preserved_after_switch=physical_seed_preserved_after_switch,
        seed_used=seed_used,
        seed_residual_norm=seed_residual_norm,
        failed=failed,
        failure_reason=failure_reason,
    )


def apply_xblock_probe_coarse_stage(
    context: XBlockProbeCoarseStageContext,
) -> XBlockProbeCoarseStageResult:
    """Apply the optional projected coarse correction before x-block Krylov."""

    policy = context.policy
    steps_requested = int(getattr(policy, "steps_requested"))
    max_directions = int(getattr(policy, "max_directions"))
    max_extra_units = int(getattr(policy, "max_extra_units"))
    fsavg_lmax = int(getattr(policy, "fsavg_lmax"))
    angular_lmax = int(getattr(policy, "angular_lmax"))
    include_angular_residual = bool(getattr(policy, "include_angular_residual"))
    include_raw = bool(getattr(policy, "include_raw"))
    alpha_clip = float(getattr(policy, "alpha_clip"))
    rcond = float(getattr(policy, "rcond"))
    min_improvement = float(getattr(policy, "min_improvement"))
    x0 = context.x0
    elapsed_s = 0.0
    history: tuple[float, ...] = ()
    direction_counts: tuple[int, ...] = ()
    direction_names: tuple[str, ...] = ()
    residual_before: float | None = None
    residual_after: float | None = None
    seed_initialized = False
    improved = False
    failed = False
    failure_reason: str | None = None

    if steps_requested > 0 and x0 is None:
        # Let this opt-in stage act as a true pre-Krylov projected solve even
        # without an unrelated seed from an earlier stage.
        x0 = jnp.zeros_like(context.rhs)
        seed_initialized = True

    if steps_requested > 0 and x0 is not None:
        start_s = float(context.elapsed_s())

        def coarse_direction_builder(
            residual_vec: jnp.ndarray,
        ) -> tuple[tuple[str, jnp.ndarray], ...]:
            return context.direction_builder(
                residual_vec,
                include_raw=bool(include_raw),
                fsavg_lmax=int(fsavg_lmax),
                angular_lmax=int(angular_lmax),
                max_extra_units=int(max_extra_units),
                max_directions=int(max_directions),
                include_angular_residual=bool(include_angular_residual),
            )

        try:
            seed_residual = context.rhs - jnp.asarray(
                context.matvec(jnp.asarray(x0, dtype=jnp.float64)),
                dtype=jnp.float64,
            )
            residual_before = profile_l2_norm_float(seed_residual)
            if (
                np.isfinite(float(residual_before))
                and float(residual_before) > float(context.target)
            ):
                (
                    x_probe,
                    residual_probe,
                    history_raw,
                    direction_counts_raw,
                    direction_names_raw,
                ) = context.correction(
                    matvec=context.matvec,
                    rhs=context.rhs,
                    x0=jnp.asarray(x0, dtype=jnp.float64),
                    direction_builder=coarse_direction_builder,
                    steps=int(steps_requested),
                    max_directions=int(max_directions),
                    alpha_clip=float(alpha_clip),
                    rcond=float(rcond),
                    min_improvement=float(min_improvement),
                )
                history = tuple(float(v) for v in history_raw)
                direction_counts = tuple(int(v) for v in direction_counts_raw)
                direction_names = tuple(str(v) for v in direction_names_raw)
                residual_after = profile_l2_norm_float(residual_probe)
                if (
                    np.isfinite(float(residual_after))
                    and float(residual_after) < float(residual_before)
                ):
                    x0 = jnp.asarray(x_probe, dtype=jnp.float64)
                    improved = True
                    if context.emit is not None:
                        context.emit(
                            0,
                            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                            f"probe-coarse improved seed residual {residual_before:.6e} "
                            f"-> {residual_after:.6e} "
                            f"(steps={len(direction_counts)} "
                            f"directions={sum(direction_counts)})",
                        )
                elif context.emit is not None:
                    after = (
                        float(residual_after)
                        if residual_after is not None
                        else float("nan")
                    )
                    context.emit(
                        1,
                        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                        f"probe-coarse rejected seed residual {residual_before:.6e} "
                        f"-> {after:.6e}",
                    )
        except Exception as exc:  # noqa: BLE001
            failed = True
            failure_reason = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"probe-coarse failed ({type(exc).__name__}: {exc})",
                )
        elapsed_s = float(context.elapsed_s()) - start_s

    return XBlockProbeCoarseStageResult(
        x0=x0,
        steps_requested=int(steps_requested),
        max_directions=int(max_directions),
        max_extra_units=int(max_extra_units),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        include_angular_residual=bool(include_angular_residual),
        include_raw=bool(include_raw),
        alpha_clip=float(alpha_clip),
        rcond=float(rcond),
        min_improvement=float(min_improvement),
        elapsed_s=float(elapsed_s),
        history=history,
        direction_counts=direction_counts,
        direction_names=direction_names,
        residual_before=residual_before,
        residual_after=residual_after,
        seed_initialized=bool(seed_initialized),
        improved=bool(improved),
        failed=bool(failed),
        failure_reason=failure_reason,
    )


def evaluate_xblock_preflight_gate(
    context: XBlockPreflightGateContext,
) -> XBlockPreflightGateResult:
    """Evaluate the optional x-block seed residual preflight gate."""

    min_improvement = float(context.min_improvement)
    required = bool(context.required)
    active = bool(min_improvement > 0.0 or required)
    if not active:
        return XBlockPreflightGateResult(
            residual_norm=None,
            improvement=None,
            passed=None,
            evaluated=False,
            failed=False,
            failure_reason=None,
        )

    if context.x0 is None:
        if required:
            raise RuntimeError(
                "xblock_sparse_pc_gmres preflight gate required an initial seed"
            )
        return XBlockPreflightGateResult(
            residual_norm=None,
            improvement=0.0,
            passed=False,
            evaluated=False,
            failed=False,
            failure_reason=None,
        )

    try:
        residual = context.rhs - jnp.asarray(
            context.matvec(jnp.asarray(context.x0, dtype=jnp.float64)),
            dtype=jnp.float64,
        )
        residual_norm = profile_l2_norm_float(residual)
        ratio = profile_safe_ratio(residual_norm, context.rhs_norm)
        improvement = 1.0 - float(ratio) if ratio is not None else 1.0
        passed = bool(
            profile_residual_converged(residual_norm, context.target)
            or float(improvement) >= min_improvement
        )
        if context.emit is not None:
            context.emit(
                0 if passed else 1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"preflight residual={float(residual_norm):.6e} "
                f"improvement={float(improvement):.6e} "
                f"required={float(min_improvement):.6e} passed={int(passed)}",
            )
        if required and not passed:
            raise RuntimeError(
                "xblock_sparse_pc_gmres preflight gate failed "
                f"improvement={float(improvement):.6e} "
                f"< required={float(min_improvement):.6e}"
            )
        return XBlockPreflightGateResult(
            residual_norm=float(residual_norm),
            improvement=float(improvement),
            passed=bool(passed),
            evaluated=True,
            failed=False,
            failure_reason=None,
        )
    except Exception as exc:  # noqa: BLE001
        if required:
            raise
        failure_reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"preflight failed ({type(exc).__name__}: {exc})",
            )
        return XBlockPreflightGateResult(
            residual_norm=None,
            improvement=None,
            passed=None,
            evaluated=True,
            failed=True,
            failure_reason=failure_reason,
        )


def resolve_xblock_krylov_control_setup(
    context: XBlockKrylovControlSetupContext,
) -> XBlockKrylovControlSetup:
    """Resolve x-block Krylov runtime controls and emit user-facing setup lines."""

    env = context.env
    method = str(context.krylov_method)
    fgmres_block_between_cycles = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES",
        default=False,
    )
    tfqmr_replacement_interval = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TFQMR_REPLACE_INTERVAL",
        default=0,
        minimum=0,
    )
    if context.emit is not None:
        tfqmr_note = (
            f" tfqmr_replacement_interval={int(tfqmr_replacement_interval)}"
            if method == "tfqmr_jax"
            else ""
        )
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres solve start "
            f"method={method} restart={int(context.pc_restart)} "
            f"maxiter={int(context.pc_maxiter)} "
            f"precondition_side={context.precondition_side}{tfqmr_note}",
        )

    device_fgmres_jit = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT",
        default=False,
    )
    device_fgmres_jit_mode = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE")
        or "cycle"
    ).lower().replace("-", "_")
    if device_fgmres_jit_mode not in {"cycle", "full"}:
        device_fgmres_jit_mode = "cycle"
    device_fgmres_jit_outer_k = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K",
        default=0,
        minimum=0,
    )
    qi_device_augmented_krylov_requested = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV",
        default=False,
    )
    qi_device_augmented_krylov_mode = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_KRYLOV_MODE",
        )
        or "combined"
    ).lower().replace("-", "_")
    if qi_device_augmented_krylov_mode not in {"projected", "combined"}:
        qi_device_augmented_krylov_mode = "combined"
    if (
        context.emit is not None
        and method in {"fgmres_jax", "gmres_jax"}
        and bool(fgmres_block_between_cycles)
    ):
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            "FGMRES cycle-boundary synchronization enabled",
        )
    if (
        context.emit is not None
        and method in {"fgmres_jax", "gmres_jax"}
        and bool(device_fgmres_jit)
    ):
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            f"JIT-compiled device FGMRES enabled mode={device_fgmres_jit_mode}",
        )

    return XBlockKrylovControlSetup(
        fgmres_block_between_cycles=bool(fgmres_block_between_cycles),
        tfqmr_replacement_interval=int(tfqmr_replacement_interval),
        device_fgmres_jit=bool(device_fgmres_jit),
        device_fgmres_jit_mode=str(device_fgmres_jit_mode),
        device_fgmres_jit_outer_k=int(device_fgmres_jit_outer_k),
        qi_device_augmented_krylov_requested=bool(
            qi_device_augmented_krylov_requested
        ),
        qi_device_augmented_krylov_mode=str(qi_device_augmented_krylov_mode),
    )


def xblock_krylov_state_from_first_attempt(
    context: XBlockFirstKrylovSolveStateContext,
) -> XBlockKrylovSolveState:
    """Convert a first xblock Krylov attempt to physical-space solve state."""

    x_solution = np.asarray(context.first_attempt.x, dtype=np.float64)
    physical_residual = xblock_physical_solution_and_residual(
        x=x_solution,
        solution_to_physical=context.solution_to_physical,
        rhs=context.physical_rhs,
        matvec=context.physical_matvec,
        fallback_residual_norm=float(context.first_attempt.residual_norm),
    )
    report = xblock_krylov_report(
        device_iterations=context.first_attempt.device_iterations,
        device_estimated_matvecs=context.first_attempt.device_estimated_matvecs,
        history=context.first_attempt.history,
        mv_count=int(context.mv_count),
    )
    return XBlockKrylovSolveState(
        krylov_method=str(context.krylov_method),
        x_solution=x_solution,
        x_physical=physical_residual.x_physical,
        residual_norm=float(physical_residual.residual_norm),
        history=tuple(float(v) for v in context.first_attempt.history),
        solve_s=float(context.solve_s),
        device_iterations=context.first_attempt.device_iterations,
        device_estimated_matvecs=context.first_attempt.device_estimated_matvecs,
        reported_iterations=int(report.iterations),
        reported_matvecs=int(report.matvecs),
    )


def xblock_krylov_state_from_gmres_fallback(
    *,
    fallback: XBlockGMRESFallbackResult,
    mv_count: int,
) -> XBlockKrylovSolveState:
    """Convert an optional GMRES fallback result to physical-space solve state."""

    report = xblock_krylov_report(
        device_iterations=fallback.device_iterations,
        device_estimated_matvecs=fallback.device_estimated_matvecs,
        history=fallback.history,
        mv_count=int(mv_count),
    )
    return XBlockKrylovSolveState(
        krylov_method=str(fallback.krylov_method),
        x_solution=np.asarray(fallback.x_solution, dtype=np.float64),
        x_physical=np.asarray(fallback.x_physical, dtype=np.float64),
        residual_norm=float(fallback.residual_norm),
        history=tuple(float(v) for v in fallback.history),
        solve_s=float(fallback.solve_s),
        device_iterations=fallback.device_iterations,
        device_estimated_matvecs=fallback.device_estimated_matvecs,
        reported_iterations=int(report.iterations),
        reported_matvecs=int(report.matvecs),
        fallback_started_from_candidate=bool(fallback.fallback_started_from_candidate),
        fallback_candidate_improved_rhs=bool(fallback.fallback_candidate_improved_rhs),
    )


def run_xblock_krylov_solve_stage(
    context: XBlockKrylovSolveStageContext,
) -> XBlockKrylovSolveStageResult:
    """Run the x-block Krylov attempt and optional GMRES fallback as one stage."""

    first_attempt = run_xblock_first_krylov_attempt(context.first_attempt)
    solve_s = (
        float(context.elapsed_s())
        - float(context.solve_start_s)
        + float(context.side_probe_s)
        + float(context.probe_coarse_s)
    )
    candidate_state = xblock_krylov_state_from_first_attempt(
        XBlockFirstKrylovSolveStateContext(
            krylov_method=str(context.first_attempt.krylov_method),
            first_attempt=first_attempt,
            solve_s=float(solve_s),
            solution_to_physical=context.solution_to_physical,
            physical_rhs=context.physical_rhs,
            physical_matvec=context.physical_matvec,
            mv_count=int(context.first_attempt.mv_count),
        )
    )
    fallback = run_xblock_gmres_fallback_if_needed(
        XBlockGMRESFallbackContext(
            krylov_method=str(context.first_attempt.krylov_method),
            fallback_enabled=bool(context.fallback_enabled),
            x_solution=candidate_state.x_solution,
            x_physical=candidate_state.x_physical,
            residual_norm=float(candidate_state.residual_norm),
            history=candidate_state.history,
            solve_s=float(candidate_state.solve_s),
            target=float(context.target),
            rhs_norm=float(context.rhs_norm),
            original_x0=context.first_attempt.x0,
            solve_rhs=context.first_attempt.rhs,
            solve_matvec=context.first_attempt.matvec,
            solve_preconditioner=context.first_attempt.preconditioner,
            precondition_side=str(context.first_attempt.precondition_side),
            tol=float(context.first_attempt.tol),
            atol=float(context.first_attempt.atol),
            restart=int(context.first_attempt.restart),
            maxiter=context.first_attempt.maxiter,
            progress_callback=context.progress_callback,
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            gmres_solver=context.first_attempt.gmres_solver,
            initial_guess_builder=context.initial_guess_builder,
            solution_to_physical=context.solution_to_physical,
            physical_rhs=context.physical_rhs,
            physical_matvec=context.physical_matvec,
            device_iterations=candidate_state.device_iterations,
            device_estimated_matvecs=candidate_state.device_estimated_matvecs,
        )
    )
    final_state = xblock_krylov_state_from_gmres_fallback(
        fallback=fallback,
        mv_count=int(context.first_attempt.mv_count),
    )
    return XBlockKrylovSolveStageResult(
        first_attempt=first_attempt,
        fallback=fallback,
        candidate_state=candidate_state,
        final_state=final_state,
    )


def xblock_device_cycle_progress_message(
    *,
    cycle: int,
    iterations: int,
    residual_norm: float,
    target: float,
    elapsed_s: float,
) -> str:
    """Return the user-facing xblock device-cycle progress line."""

    ratio = float(residual_norm) / float(target) if float(target) > 0.0 else float("nan")
    return (
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
        f"device-cycle cycle={int(cycle)} iterations={int(iterations)} "
        f"residual={float(residual_norm):.6e} target={float(target):.6e} "
        f"ratio={float(ratio):.6e} elapsed_s={float(elapsed_s):.3f}"
    )


def xblock_host_krylov_progress_message(
    *,
    iteration: int,
    residual_norm: float,
    elapsed_s: float,
) -> str:
    """Return the user-facing host xblock Krylov progress line."""

    return (
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
        f"iters={int(iteration)} ksp_residual={float(residual_norm):.6e} "
        f"elapsed_s={float(elapsed_s):.3f}"
    )


def build_xblock_krylov_progress_callbacks(
    context: XBlockKrylovProgressCallbacksContext,
) -> XBlockKrylovProgressCallbacks:
    """Return host/device progress callbacks for the x-block Krylov solve."""

    def device_cycle_progress_callback(
        *,
        cycle: int,
        iterations: int,
        residual_norm: float,
        target: float,
    ) -> None:
        if context.emit is None:
            return
        context.emit(
            0,
            xblock_device_cycle_progress_message(
                cycle=int(cycle),
                iterations=int(iterations),
                residual_norm=float(residual_norm),
                target=float(target),
                elapsed_s=float(context.elapsed_s()),
            ),
        )

    def host_progress_callback(iteration: int, residual_norm: float) -> None:
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if int(iteration) % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            xblock_host_krylov_progress_message(
                iteration=int(iteration),
                residual_norm=float(residual_norm),
                elapsed_s=float(context.elapsed_s()),
            ),
        )

    return XBlockKrylovProgressCallbacks(
        host_progress_callback=host_progress_callback,
        device_cycle_progress_callback=device_cycle_progress_callback,
    )


def xblock_device_krylov_state(
    result: object,
    *,
    estimated_matvecs_floor: int | None = None,
) -> XBlockDeviceKrylovState:
    """Transfer a device xblock Krylov result to host arrays and counters."""

    x = np.asarray(jax.device_get(result.x), dtype=np.float64)
    residual_norm = float(jax.device_get(result.residual_norm))
    history_arr = np.asarray(jax.device_get(result.residual_history), dtype=np.float64)
    n_iterations = int(jax.device_get(result.n_iterations))
    history = tuple(
        float(v)
        for v in history_arr[: n_iterations + 1]
        if np.isfinite(float(v))
    )
    estimated_matvecs = None
    if estimated_matvecs_floor is not None:
        estimated_matvecs = max(int(estimated_matvecs_floor), int(n_iterations) + 2)
    return XBlockDeviceKrylovState(
        x=x,
        residual_norm=float(residual_norm),
        history=history,
        n_iterations=int(n_iterations),
        estimated_matvecs=estimated_matvecs,
    )


def prepare_xblock_krylov_solve_space(
    context: XBlockKrylovSolveSpaceContext,
) -> XBlockKrylovSolveSpace:
    """Apply xblock row/column equilibration to the Krylov solve callbacks."""

    def _identity_solution(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(v, dtype=jnp.float64)

    if not bool(context.row_equilibration_built):
        return XBlockKrylovSolveSpace(
            matvec=context.matvec,
            rhs=context.rhs,
            preconditioner=context.preconditioner if str(context.precondition_side) != "none" else None,
            x0=context.x0,
            solution_to_physical=_identity_solution,
            transform_label=None,
        )

    if context.row_scale is None or context.inv_row_scale is None:
        raise ValueError("row equilibration requires row_scale and inv_row_scale")
    if bool(context.col_equilibration_built) and (
        context.col_scale is None or context.inv_col_scale is None
    ):
        raise ValueError("column equilibration requires col_scale and inv_col_scale")

    row_scale = jnp.asarray(context.row_scale, dtype=jnp.float64)
    inv_row_scale = jnp.asarray(context.inv_row_scale, dtype=jnp.float64)
    col_scale = (
        jnp.asarray(context.col_scale, dtype=jnp.float64)
        if bool(context.col_equilibration_built)
        else None
    )
    inv_col_scale = (
        jnp.asarray(context.inv_col_scale, dtype=jnp.float64)
        if bool(context.col_equilibration_built)
        else None
    )
    base_matvec = context.matvec
    base_preconditioner = context.preconditioner

    def _mv_equilibrated(v: jnp.ndarray) -> jnp.ndarray:
        v_j = jnp.asarray(v, dtype=jnp.float64)
        physical_v = col_scale * v_j if col_scale is not None else v_j
        return row_scale * jnp.asarray(base_matvec(physical_v), dtype=jnp.float64)

    def _precond_equilibrated(v: jnp.ndarray) -> jnp.ndarray:
        physical_residual = inv_row_scale * jnp.asarray(v, dtype=jnp.float64)
        if base_preconditioner is None:
            physical_update = physical_residual
        else:
            physical_update = jnp.asarray(base_preconditioner(physical_residual), dtype=jnp.float64)
        if inv_col_scale is not None:
            return inv_col_scale * physical_update
        return physical_update

    rhs = row_scale * jnp.asarray(context.rhs, dtype=jnp.float64)
    x0 = context.x0
    if col_scale is not None and inv_col_scale is not None:
        x0 = None if x0 is None else inv_col_scale * jnp.asarray(x0, dtype=jnp.float64)

        def _solution_to_physical(v: jnp.ndarray) -> jnp.ndarray:
            return col_scale * jnp.asarray(v, dtype=jnp.float64)

        solution_to_physical = _solution_to_physical
        transform_label = "row/column"
    else:
        solution_to_physical = _identity_solution
        transform_label = "row"

    return XBlockKrylovSolveSpace(
        matvec=_mv_equilibrated,
        rhs=rhs,
        preconditioner=_precond_equilibrated if str(context.precondition_side) != "none" else None,
        x0=x0,
        solution_to_physical=solution_to_physical,
        transform_label=transform_label,
    )


def prepare_xblock_augmented_krylov_basis(
    context: XBlockAugmentedKrylovBasisContext,
) -> XBlockAugmentedKrylovBasisResult:
    """Prepare the optional QI augmented Krylov basis for the solve-space operator."""

    seed_available = bool(
        context.seed_available
        and context.seed_basis is not None
        and context.seed_operator_on_basis is not None
        and int(context.seed_rank) > 0
    )
    if context.qi_device_state is None:
        return XBlockAugmentedKrylovBasisResult(
            basis=None,
            operator_on_basis=None,
            used=False,
            rank=0,
            reason="disabled_missing_qi_device_state",
            seed_used=False,
        )
    if str(context.krylov_method) not in {"fgmres_jax", "gmres_jax"}:
        return XBlockAugmentedKrylovBasisResult(
            basis=None,
            operator_on_basis=None,
            used=False,
            rank=0,
            reason="disabled_non_jax_fgmres_method",
            seed_used=False,
        )
    if int(context.qi_device_state.metadata.rank) <= 0 and not seed_available:
        return XBlockAugmentedKrylovBasisResult(
            basis=None,
            operator_on_basis=None,
            used=False,
            rank=0,
            reason="disabled_empty_qi_device_basis",
            seed_used=False,
        )

    try:
        if seed_available:
            basis = jnp.asarray(context.seed_basis, dtype=jnp.float64)
            operator_on_basis = jnp.asarray(context.seed_operator_on_basis, dtype=jnp.float64)
            reason = "enabled_from_augmented_seed"
            seed_used = True
        else:
            basis = jnp.asarray(context.qi_device_state.basis.vectors, dtype=jnp.float64)
            operator_on_basis = jnp.asarray(context.qi_device_state.operator_on_basis, dtype=jnp.float64)
            reason = "enabled"
            seed_used = False

        if bool(context.col_equilibration_built) and context.inv_col_scale is not None:
            basis = jnp.asarray(context.inv_col_scale, dtype=jnp.float64).reshape((-1, 1)) * basis
        if bool(context.row_equilibration_built) and context.row_scale is not None:
            operator_on_basis = (
                jnp.asarray(context.row_scale, dtype=jnp.float64).reshape((-1, 1))
                * operator_on_basis
            )
        if str(context.precondition_side) == "left" and context.solve_preconditioner is not None:
            operator_on_basis = jnp.stack(
                [
                    jnp.asarray(
                        context.solve_preconditioner(operator_on_basis[:, idx]),
                        dtype=jnp.float64,
                    )
                    for idx in range(int(operator_on_basis.shape[1]))
                ],
                axis=1,
            )
        return XBlockAugmentedKrylovBasisResult(
            basis=basis,
            operator_on_basis=operator_on_basis,
            used=True,
            rank=int(basis.shape[1]),
            reason=reason,
            seed_used=seed_used,
        )
    except Exception as exc:  # noqa: BLE001
        return XBlockAugmentedKrylovBasisResult(
            basis=None,
            operator_on_basis=None,
            used=False,
            rank=0,
            reason=f"{type(exc).__name__}: {exc}",
            seed_used=False,
        )


def apply_xblock_augmented_krylov_stage(
    context: XBlockAugmentedKrylovStageContext,
) -> XBlockAugmentedKrylovStageResult:
    """Prepare optional QI augmented-Krylov inputs and update metadata."""

    metadata = dict(context.metadata)
    if not bool(context.requested):
        return XBlockAugmentedKrylovStageResult(
            basis=None,
            operator_on_basis=None,
            used=False,
            rank=0,
            reason=None,
            seed_used=bool(context.seed_used),
            metadata=metadata,
        )

    augmented_krylov = context.basis_builder(
        XBlockAugmentedKrylovBasisContext(
            krylov_method=str(context.krylov_method),
            qi_device_state=context.qi_device_state,
            seed_available=bool(context.seed_available),
            seed_rank=int(context.seed_rank),
            seed_basis=context.seed_basis,
            seed_operator_on_basis=context.seed_operator_on_basis,
            row_equilibration_built=bool(context.row_equilibration_built),
            col_equilibration_built=bool(context.col_equilibration_built),
            row_scale=context.row_scale,
            inv_col_scale=context.inv_col_scale,
            precondition_side=str(context.precondition_side),
            solve_preconditioner=context.solve_preconditioner,
        )
    )
    seed_used = bool(context.seed_used or augmented_krylov.seed_used)
    metadata["augmented_seed_used"] = bool(seed_used)
    metadata["augmented_seed_available"] = bool(context.seed_available)
    if context.emit is not None:
        context.emit(
            0 if bool(augmented_krylov.used) else 1,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            f"QI augmented Krylov {augmented_krylov.reason} "
            f"rank={int(augmented_krylov.rank)} "
            f"mode={context.mode}",
        )
    return XBlockAugmentedKrylovStageResult(
        basis=augmented_krylov.basis,
        operator_on_basis=augmented_krylov.operator_on_basis,
        used=bool(augmented_krylov.used),
        rank=int(augmented_krylov.rank),
        reason=str(augmented_krylov.reason),
        seed_used=bool(seed_used),
        metadata=metadata,
    )


def run_xblock_first_krylov_attempt(
    context: XBlockFirstKrylovAttemptContext,
) -> XBlockFirstKrylovAttemptResult:
    """Run the selected first xblock sparse-PC Krylov method."""

    method = str(context.krylov_method)
    device_iterations: int | None = None
    device_estimated_matvecs: int | None = None

    if method == "lgmres":
        x_np, residual_norm, history = context.lgmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            outer_k=context.lgmres_outer_k,
            precondition_side=str(context.precondition_side),
        )
    elif method in {"gmres_jax", "fgmres_jax"}:
        fgmres_solver = (
            (
                context.fgmres_cycle_jit_solver
                if str(context.device_fgmres_jit_mode) == "cycle"
                else context.fgmres_jit_solver
            )
            if bool(context.device_fgmres_jit)
            else context.fgmres_solver
        )
        fgmres_kwargs: dict[str, Any] = {
            "matvec": context.matvec,
            "b": context.rhs,
            "preconditioner": context.preconditioner,
            "x0": context.x0,
            "tol": float(context.tol),
            "atol": float(context.atol),
            "restart": int(context.restart),
            "maxiter": context.maxiter,
            "precondition_side": str(context.precondition_side),
            "skip_inactive_work": bool(context.skip_inactive_work),
            "block_between_cycles": bool(context.fgmres_block_between_cycles),
        }
        if bool(context.device_fgmres_jit) and str(context.device_fgmres_jit_mode) == "cycle":
            fgmres_kwargs["outer_k"] = int(context.device_fgmres_jit_outer_k)
            fgmres_kwargs["augmentation_mode"] = str(context.augmentation_mode)
            fgmres_kwargs["progress_callback"] = context.device_cycle_progress_callback
        if bool(context.augmented_krylov_used):
            fgmres_kwargs["augmentation_basis"] = context.augmentation_basis
            fgmres_kwargs["operator_on_augmentation"] = context.operator_on_augmentation
        fgmres_result, _fgmres_residual = fgmres_solver(**fgmres_kwargs)
        device_state = xblock_device_krylov_state(
            fgmres_result,
            estimated_matvecs_floor=(
                int(context.mv_count)
                if bool(context.device_fgmres_jit)
                and str(context.device_fgmres_jit_mode) == "cycle"
                else None
            ),
        )
        x_np = device_state.x
        residual_norm = float(device_state.residual_norm)
        history = device_state.history
        device_iterations = int(device_state.n_iterations)
        device_estimated_matvecs = device_state.estimated_matvecs
    elif method == "bicgstab_jax":
        bicgstab_result, _bicgstab_residual = context.bicgstab_jax_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=context.maxiter,
            precondition_side=str(context.precondition_side),
        )
        device_state = xblock_device_krylov_state(bicgstab_result)
        x_np = device_state.x
        residual_norm = float(device_state.residual_norm)
        history = device_state.history
        device_iterations = int(device_state.n_iterations)
    elif method == "tfqmr_jax":
        tfqmr_result, _tfqmr_residual = context.tfqmr_jax_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=context.maxiter,
            precondition_side=str(context.precondition_side),
            residual_replacement_interval=int(context.tfqmr_replacement_interval),
        )
        device_state = xblock_device_krylov_state(tfqmr_result)
        x_np = device_state.x
        residual_norm = float(device_state.residual_norm)
        history = device_state.history
        device_iterations = int(device_state.n_iterations)
    elif method == "gcrotmk":
        x_np, residual_norm, history = context.gcrotmk_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            precondition_side=str(context.precondition_side),
        )
    elif method == "bicgstab":
        x_np, residual_norm, history = context.bicgstab_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=context.maxiter,
            precondition_side=str(context.precondition_side),
        )
    else:
        x_np, residual_norm, history = context.gmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=context.x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=context.maxiter,
            precondition_side=str(context.precondition_side),
            progress_callback=context.host_progress_callback,
        )

    return XBlockFirstKrylovAttemptResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        history=tuple(float(v) for v in (history or ())),
        device_iterations=device_iterations,
        device_estimated_matvecs=device_estimated_matvecs,
    )


def xblock_gmres_fallback_decision(
    *,
    krylov_method: str,
    fallback_enabled: bool,
    residual_norm: float,
    target: float,
) -> XBlockGMRESFallbackDecision:
    """Decide whether a non-GMRES xblock solve needs a GMRES fallback."""

    residual = float(residual_norm)
    should_retry = (
        str(krylov_method) != "gmres"
        and bool(fallback_enabled)
        and ((not np.isfinite(residual)) or residual > float(target))
    )
    return XBlockGMRESFallbackDecision(run=bool(should_retry))


def run_xblock_gmres_fallback_if_needed(
    context: XBlockGMRESFallbackContext,
) -> XBlockGMRESFallbackResult:
    """Retry a failed non-GMRES xblock solve with GMRES when policy permits."""

    x_solution = np.asarray(context.x_solution, dtype=np.float64)
    x_physical = np.asarray(context.x_physical, dtype=np.float64)
    residual_norm = float(context.residual_norm)
    history = tuple(float(v) for v in (context.history or ()))
    krylov_method = str(context.krylov_method)
    device_iterations = context.device_iterations
    device_estimated_matvecs = context.device_estimated_matvecs
    fallback_started_from_candidate = False
    fallback_candidate_improved_rhs = False

    fallback_decision = xblock_gmres_fallback_decision(
        krylov_method=krylov_method,
        fallback_enabled=bool(context.fallback_enabled),
        residual_norm=float(residual_norm),
        target=float(context.target),
    )
    if not fallback_decision.run:
        return XBlockGMRESFallbackResult(
            krylov_method=krylov_method,
            x_solution=x_solution,
            x_physical=x_physical,
            residual_norm=float(residual_norm),
            history=history,
            solve_s=float(context.solve_s),
            device_iterations=device_iterations,
            device_estimated_matvecs=device_estimated_matvecs,
            fallback_started_from_candidate=False,
            fallback_candidate_improved_rhs=False,
        )

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
            f"{krylov_method} residual={float(residual_norm):.6e} "
            f"> target={float(context.target):.6e}; falling back to gmres",
        )

    (
        fallback_x0,
        fallback_started_from_candidate,
        fallback_candidate_improved_rhs,
    ) = context.initial_guess_builder(
        candidate=x_solution,
        original_x0=context.original_x0,
        rhs_shape=tuple(context.solve_rhs.shape),
        candidate_residual_norm=float(residual_norm),
        rhs_norm=float(context.rhs_norm),
        precondition_side=str(context.precondition_side),
    )
    fallback_start_s = float(context.elapsed_s())
    x_np, residual_fallback, history_fallback = context.gmres_solver(
        matvec=context.solve_matvec,
        b=context.solve_rhs,
        preconditioner=context.solve_preconditioner,
        x0=fallback_x0,
        tol=float(context.tol),
        atol=float(context.atol),
        restart=int(context.restart),
        maxiter=context.maxiter,
        precondition_side=str(context.precondition_side),
        progress_callback=context.progress_callback,
    )
    solve_s = float(context.solve_s) + (float(context.elapsed_s()) - fallback_start_s)
    x_solution = np.asarray(x_np, dtype=np.float64)
    physical_residual = xblock_physical_solution_and_residual(
        x=x_solution,
        solution_to_physical=context.solution_to_physical,
        rhs=context.physical_rhs,
        matvec=context.physical_matvec,
        fallback_residual_norm=float(residual_fallback),
    )
    return XBlockGMRESFallbackResult(
        krylov_method="gmres",
        x_solution=x_solution,
        x_physical=physical_residual.x_physical,
        residual_norm=float(physical_residual.residual_norm),
        history=tuple(float(v) for v in (history_fallback or ())),
        solve_s=float(solve_s),
        device_iterations=None,
        device_estimated_matvecs=None,
        fallback_started_from_candidate=bool(fallback_started_from_candidate),
        fallback_candidate_improved_rhs=bool(fallback_candidate_improved_rhs),
    )


def xblock_sparse_pc_work_estimates(
    *,
    krylov_method: str,
    linear_size: int,
    restart: int,
    dtype: Any = np.float64,
) -> XBlockSparsePCWorkEstimates:
    """Return xblock sparse-PC method labels and Krylov work estimates."""

    method = str(krylov_method)
    return XBlockSparsePCWorkEstimates(
        solver_kind=(
            "xblock_sparse_pc_gmres"
            if method == "gmres"
            else f"xblock_sparse_pc_{method}"
        ),
        device_krylov_methods=frozenset(
            {"fgmres_jax", "gmres_jax", "bicgstab_jax", "tfqmr_jax"}
        ),
        gmres_basis_nbytes=gmres_basis_nbytes(
            int(linear_size),
            int(restart),
            dtype=dtype,
        ),
        bicgstab_work_nbytes=bicgstab_work_nbytes(int(linear_size), dtype=dtype),
        tfqmr_work_nbytes=tfqmr_work_nbytes(int(linear_size), dtype=dtype),
    )


def xblock_sparse_pc_completion_message(
    *,
    krylov_method: str,
    elapsed_s: float,
    iterations: int,
    matvecs: int,
    residual_norm: float,
    target: float,
    history: Sequence[float] | None,
) -> str:
    """Format the final xblock sparse-PC progress line shown to users."""

    ksp_suffix = (
        f" ksp_residual={float(history[-1]):.6e}" if history else ""
    )
    return (
        "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres complete "
        f"method={krylov_method} elapsed_s={float(elapsed_s):.3f} "
        f"iters={int(iterations)} "
        f"matvecs={int(matvecs)} residual={float(residual_norm):.6e} "
        f"target={float(target):.6e}{ksp_suffix}"
    )


def emit_xblock_sparse_pc_completion(
    context: XBlockSparsePCCompletionContext,
) -> None:
    """Emit the final xblock sparse-PC progress line from explicit inputs."""

    if context.emit is None:
        return
    context.emit(
        0,
        xblock_sparse_pc_completion_message(
            krylov_method=str(context.krylov_method),
            elapsed_s=float(context.elapsed_s),
            iterations=int(context.iterations),
            matvecs=int(context.matvecs),
            residual_norm=float(context.residual_norm),
            target=float(context.target),
            history=context.history,
        ),
    )


def emit_xblock_sparse_pc_completion_from_driver_state(
    state: Mapping[str, object],
) -> None:
    """Emit the final xblock sparse-PC progress line from driver state."""

    if state["emit"] is None:
        return
    emit_xblock_sparse_pc_completion(
        XBlockSparsePCCompletionContext(
            emit=state["emit"],
            krylov_method=str(state["xblock_krylov_method"]),
            elapsed_s=state["sparse_timer"].elapsed_s(),
            iterations=int(state["reported_iterations"]),
            matvecs=int(state["reported_matvecs"]),
            residual_norm=float(state["residual_norm_xblock_pc"]),
            target=float(state["target_xblock"]),
            history=state["history"],
        ),
    )


def xblock_physical_solution_and_residual(
    *,
    x: np.ndarray,
    solution_to_physical: Callable[[jnp.ndarray], jnp.ndarray],
    rhs: jnp.ndarray,
    matvec: Callable[[jnp.ndarray], jnp.ndarray],
    fallback_residual_norm: float,
) -> XBlockPhysicalResidual:
    """Map a Krylov solution to physical coordinates and measure true residual."""

    x_solution = np.asarray(x, dtype=np.float64)
    x_physical = np.asarray(
        jax.device_get(solution_to_physical(jnp.asarray(x_solution, dtype=jnp.float64))),
        dtype=np.float64,
    )
    try:
        residual_true = np.asarray(rhs, dtype=np.float64) - np.asarray(
            jax.device_get(matvec(jnp.asarray(x_physical, dtype=jnp.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(fallback_residual_norm)
    return XBlockPhysicalResidual(
        x_physical=np.asarray(x_physical, dtype=np.float64),
        residual_norm=float(residual_norm),
    )



@dataclass(frozen=True)
class XBlockSubspaceCorrectionContext:
    """Dependencies for an x-block sparse-PC subspace correction."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    x: np.ndarray
    residual_norm: float
    target: float
    direction_builder: Callable[[jnp.ndarray], tuple[tuple[str, jnp.ndarray], ...]]
    steps: int
    max_directions: int
    alpha_clip: float
    rcond: float
    min_improvement: float
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[int], Sequence[str]]]
    correction_kwargs: Mapping[str, Any] | None = None
    solver_label: str = "xblock_sparse_pc_gmres"
    correction_label: str = "post-coarse"
    diagnostic_suffix: str = ""


@dataclass(frozen=True)
class XBlockSubspaceCorrectionResult:
    """Accepted x-block subspace correction state and diagnostics."""

    x: np.ndarray
    residual_norm: float
    history: tuple[float, ...]
    direction_counts: tuple[int, ...]
    direction_names: tuple[str, ...]
    residual_before: float | None
    residual_after: float | None
    error: str | None
    solve_s: float


@dataclass(frozen=True)
class XBlockPostSolveCorrectionContext:
    """Inputs for x-block sparse-PC post-solve correction orchestration."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    x: np.ndarray
    residual_norm: float
    target: float
    solve_s: float
    preconditioner: ArrayFn
    precondition_side: str
    post_solve_policy: object
    qi_device_state: object | None
    coarse_direction_builder: Callable[..., tuple[tuple[str, jnp.ndarray], ...]]
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]]
    residual_equation_correction: Callable[
        ...,
        tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[int], Sequence[str]],
    ]
    coarse_correction: Callable[
        ...,
        tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[int], Sequence[str]],
    ]


@dataclass(frozen=True)
class XBlockPostSolveCorrectionResult:
    """Updated x-block sparse-PC solve state and correction diagnostics."""

    x: np.ndarray
    residual_norm: float
    solve_s: float
    post_minres_steps_requested: int
    post_minres_alpha_clip: float
    post_minres_min_improvement: float
    post_minres_history: tuple[float, ...]
    post_minres_alphas: tuple[float, ...]
    post_minres_residual_before: float | None
    post_minres_residual_after: float | None
    post_coarse_steps_requested: int
    post_coarse_max_directions: int
    post_coarse_max_extra_units: int
    post_coarse_fsavg_lmax: int
    post_coarse_angular_lmax: int
    post_coarse_include_angular_residual: bool
    post_coarse_include_raw: bool
    post_coarse_alpha_clip: float
    post_coarse_rcond: float
    post_coarse_min_improvement: float
    post_coarse_history: tuple[float, ...]
    post_coarse_direction_counts: tuple[int, ...]
    post_coarse_direction_names: tuple[str, ...]
    post_coarse_residual_before: float | None
    post_coarse_residual_after: float | None
    post_residual_equation_steps_requested: int
    post_residual_equation_max_directions: int
    post_residual_equation_max_extra_units: int
    post_residual_equation_fsavg_lmax: int
    post_residual_equation_angular_lmax: int
    post_residual_equation_include_angular_residual: bool
    post_residual_equation_include_raw: bool
    post_residual_equation_include_post_coarse: bool
    post_residual_equation_include_qi_basis: bool
    post_residual_equation_alpha_clip: float
    post_residual_equation_rcond: float
    post_residual_equation_min_improvement: float
    post_residual_equation_history: tuple[float, ...]
    post_residual_equation_direction_counts: tuple[int, ...]
    post_residual_equation_direction_names: tuple[str, ...]
    post_residual_equation_residual_before: float | None
    post_residual_equation_residual_after: float | None

    def driver_state(self) -> dict[str, object]:
        """Return historical driver-state keys consumed by final metadata."""

        return {
            "post_minres_steps_requested": self.post_minres_steps_requested,
            "post_minres_alpha_clip": self.post_minres_alpha_clip,
            "post_minres_min_improvement": self.post_minres_min_improvement,
            "post_minres_history": self.post_minres_history,
            "post_minres_alphas": self.post_minres_alphas,
            "post_minres_residual_before": self.post_minres_residual_before,
            "post_minres_residual_after": self.post_minres_residual_after,
            "post_coarse_steps_requested": self.post_coarse_steps_requested,
            "post_coarse_max_directions": self.post_coarse_max_directions,
            "post_coarse_max_extra_units": self.post_coarse_max_extra_units,
            "post_coarse_fsavg_lmax": self.post_coarse_fsavg_lmax,
            "post_coarse_angular_lmax": self.post_coarse_angular_lmax,
            "post_coarse_include_angular_residual": self.post_coarse_include_angular_residual,
            "post_coarse_include_raw": self.post_coarse_include_raw,
            "post_coarse_alpha_clip": self.post_coarse_alpha_clip,
            "post_coarse_rcond": self.post_coarse_rcond,
            "post_coarse_min_improvement": self.post_coarse_min_improvement,
            "post_coarse_history": self.post_coarse_history,
            "post_coarse_direction_counts": self.post_coarse_direction_counts,
            "post_coarse_direction_names": self.post_coarse_direction_names,
            "post_coarse_residual_before": self.post_coarse_residual_before,
            "post_coarse_residual_after": self.post_coarse_residual_after,
            "post_residual_equation_steps_requested": self.post_residual_equation_steps_requested,
            "post_residual_equation_max_directions": self.post_residual_equation_max_directions,
            "post_residual_equation_max_extra_units": self.post_residual_equation_max_extra_units,
            "post_residual_equation_fsavg_lmax": self.post_residual_equation_fsavg_lmax,
            "post_residual_equation_angular_lmax": self.post_residual_equation_angular_lmax,
            "post_residual_equation_include_angular_residual": self.post_residual_equation_include_angular_residual,
            "post_residual_equation_include_raw": self.post_residual_equation_include_raw,
            "post_residual_equation_include_post_coarse": self.post_residual_equation_include_post_coarse,
            "post_residual_equation_include_qi_basis": self.post_residual_equation_include_qi_basis,
            "post_residual_equation_alpha_clip": self.post_residual_equation_alpha_clip,
            "post_residual_equation_rcond": self.post_residual_equation_rcond,
            "post_residual_equation_min_improvement": self.post_residual_equation_min_improvement,
            "post_residual_equation_history": self.post_residual_equation_history,
            "post_residual_equation_direction_counts": self.post_residual_equation_direction_counts,
            "post_residual_equation_direction_names": self.post_residual_equation_direction_names,
            "post_residual_equation_residual_before": self.post_residual_equation_residual_before,
            "post_residual_equation_residual_after": self.post_residual_equation_residual_after,
        }


@dataclass(frozen=True)
class XBlockPostKrylovCompletionContext:
    """Inputs for post-Krylov correction followed by completion emission."""

    corrections: XBlockPostSolveCorrectionContext
    krylov_method: str
    elapsed_s: Callable[[], float]
    iterations: int
    matvecs: int
    target: float
    history: Sequence[float] | None


@dataclass(frozen=True)
class XBlockPostKrylovCompletionResult:
    """Final x-block state after post-solve corrections and completion emission."""

    corrections: XBlockPostSolveCorrectionResult
    x: np.ndarray
    residual_norm: float
    solve_s: float


def apply_xblock_subspace_correction_if_needed(
    context: XBlockSubspaceCorrectionContext,
) -> XBlockSubspaceCorrectionResult:
    """Apply one x-block subspace correction and accept only residual improvement."""

    x_out = np.asarray(context.x, dtype=np.float64)
    residual_current = float(context.residual_norm)
    if (
        int(context.steps) <= 0
        or not np.isfinite(residual_current)
        or residual_current <= float(context.target)
    ):
        return XBlockSubspaceCorrectionResult(
            x=x_out,
            residual_norm=residual_current,
            history=(),
            direction_counts=(),
            direction_names=(),
            residual_before=None,
            residual_after=None,
            error=None,
            solve_s=0.0,
        )

    residual_before = residual_current
    start_s = float(context.elapsed_s())
    history: tuple[float, ...] = ()
    direction_counts: tuple[int, ...] = ()
    direction_names: tuple[str, ...] = ()
    residual_after: float | None = None
    error: str | None = None
    try:
        correction_kwargs = dict(context.correction_kwargs or {})
        (
            x_candidate,
            residual_candidate,
            correction_history,
            correction_direction_counts,
            correction_direction_names,
        ) = context.correction(
            matvec=context.matvec,
            rhs=context.rhs,
            x0=jnp.asarray(x_out, dtype=jnp.float64),
            direction_builder=context.direction_builder,
            steps=int(context.steps),
            max_directions=int(context.max_directions),
            alpha_clip=float(context.alpha_clip),
            rcond=float(context.rcond),
            min_improvement=float(context.min_improvement),
            **correction_kwargs,
        )
        history = tuple(float(v) for v in correction_history)
        direction_counts = tuple(int(v) for v in correction_direction_counts)
        direction_names = tuple(str(v) for v in correction_direction_names)
        residual_after = float(jnp.linalg.norm(residual_candidate))
        if np.isfinite(float(residual_after)) and float(residual_after) < residual_current:
            x_out = np.asarray(x_candidate, dtype=np.float64)
            residual_current = float(residual_after)
            if context.emit is not None:
                context.emit(
                    0,
                    f"solve_v3_full_system_linear_gmres: {context.solver_label} "
                    f"{context.correction_label} improved residual {residual_before:.6e} "
                    f"-> {residual_after:.6e} "
                    f"(steps={len(direction_counts)} directions={sum(direction_counts)}"
                    f"{context.diagnostic_suffix})",
                )
        elif context.emit is not None:
            after = float(residual_after) if residual_after is not None else float("nan")
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} "
                f"{context.correction_label} rejected residual {residual_before:.6e} -> {after:.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} "
                f"{context.correction_label} failed ({error})",
            )

    return XBlockSubspaceCorrectionResult(
        x=x_out,
        residual_norm=float(residual_current),
        history=history,
        direction_counts=direction_counts,
        direction_names=direction_names,
        residual_before=float(residual_before),
        residual_after=residual_after,
        error=error,
        solve_s=float(context.elapsed_s()) - start_s,
    )


def run_xblock_post_solve_corrections(
    context: XBlockPostSolveCorrectionContext,
) -> XBlockPostSolveCorrectionResult:
    """Run x-block post-residual, minres, and coarse correction hooks."""

    post_minres_policy = context.post_solve_policy.post_minres
    post_minres_steps_requested = int(post_minres_policy.steps_requested)
    post_minres_alpha_clip = float(post_minres_policy.alpha_clip)
    post_minres_min_improvement = float(post_minres_policy.min_improvement)

    post_coarse_policy = context.post_solve_policy.post_coarse
    post_coarse_steps_requested = int(post_coarse_policy.steps_requested)
    post_coarse_max_directions = int(post_coarse_policy.max_directions)
    post_coarse_max_extra_units = int(post_coarse_policy.max_extra_units)
    post_coarse_fsavg_lmax = int(post_coarse_policy.fsavg_lmax)
    post_coarse_angular_lmax = int(post_coarse_policy.angular_lmax)
    post_coarse_include_angular_residual = bool(
        post_coarse_policy.include_angular_residual
    )
    post_coarse_include_raw = bool(post_coarse_policy.include_raw)
    post_coarse_alpha_clip = float(post_coarse_policy.alpha_clip)
    post_coarse_rcond = float(post_coarse_policy.rcond)
    post_coarse_min_improvement = float(post_coarse_policy.min_improvement)

    post_residual_policy = context.post_solve_policy.post_residual_equation
    post_residual_equation_steps_requested = int(post_residual_policy.steps_requested)
    post_residual_equation_max_directions = int(post_residual_policy.max_directions)
    post_residual_equation_max_extra_units = int(post_residual_policy.max_extra_units)
    post_residual_equation_fsavg_lmax = int(post_residual_policy.fsavg_lmax)
    post_residual_equation_angular_lmax = int(post_residual_policy.angular_lmax)
    post_residual_equation_include_angular_residual = bool(
        post_residual_policy.include_angular_residual
    )
    post_residual_equation_include_raw = bool(post_residual_policy.include_raw)
    post_residual_equation_include_post_coarse = bool(
        post_residual_policy.include_post_coarse
    )
    post_residual_equation_include_qi_basis = bool(
        post_residual_policy.include_qi_basis
    )
    post_residual_equation_alpha_clip = float(post_residual_policy.alpha_clip)
    post_residual_equation_rcond = float(post_residual_policy.rcond)
    post_residual_equation_min_improvement = float(
        post_residual_policy.min_improvement
    )

    x_np = np.asarray(context.x, dtype=np.float64)
    residual_norm = float(context.residual_norm)
    solve_s = float(context.solve_s)

    def _post_residual_equation_direction_builder(
        residual_vec: jnp.ndarray,
    ) -> tuple[tuple[str, jnp.ndarray], ...]:
        if not bool(post_residual_equation_include_post_coarse):
            return ()
        return context.coarse_direction_builder(
            residual_vec,
            include_raw=bool(post_residual_equation_include_raw),
            fsavg_lmax=int(post_residual_equation_fsavg_lmax),
            angular_lmax=int(post_residual_equation_angular_lmax),
            max_extra_units=int(post_residual_equation_max_extra_units),
            max_directions=int(post_residual_equation_max_directions),
            include_angular_residual=bool(
                post_residual_equation_include_angular_residual
            ),
        )

    post_residual_equation_active = (
        post_residual_equation_steps_requested > 0
        and np.isfinite(float(residual_norm))
        and float(residual_norm) > float(context.target)
    )
    cached_qi_basis = prepare_cached_qi_correction_basis(
        active=bool(post_residual_equation_active),
        include_qi_basis=bool(post_residual_equation_include_qi_basis),
        qi_device_state=context.qi_device_state,
    )
    post_residual_equation = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=context.matvec,
            rhs=context.rhs,
            x=np.asarray(x_np, dtype=np.float64),
            residual_norm=float(residual_norm),
            target=float(context.target),
            direction_builder=_post_residual_equation_direction_builder,
            steps=int(post_residual_equation_steps_requested),
            max_directions=int(post_residual_equation_max_directions),
            alpha_clip=float(post_residual_equation_alpha_clip),
            rcond=float(post_residual_equation_rcond),
            min_improvement=float(post_residual_equation_min_improvement),
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            correction=context.residual_equation_correction,
            correction_kwargs={
                "cached_basis": cached_qi_basis.vectors,
                "cached_operator_on_basis": cached_qi_basis.operator_on_basis,
                "cached_labels": cached_qi_basis.labels,
            },
            solver_label="xblock_sparse_pc_gmres",
            correction_label="post-residual-equation",
            diagnostic_suffix=f" cached_qi={int(cached_qi_basis.vectors is not None)}",
        )
    )
    x_np = np.asarray(post_residual_equation.x, dtype=np.float64)
    residual_norm = float(post_residual_equation.residual_norm)
    solve_s += float(post_residual_equation.solve_s)

    post_minres = apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=context.matvec,
            rhs=context.rhs,
            preconditioner=(
                context.preconditioner
                if str(context.precondition_side) != "none"
                else (lambda v: v)
            ),
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            pc_form="none",
            steps=int(post_minres_steps_requested),
            alpha_clip=float(post_minres_alpha_clip),
            min_improvement=float(post_minres_min_improvement),
            minres_correction=context.minres_correction,
            x=np.asarray(x_np, dtype=np.float64),
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(residual_norm),
            solve_s=float(solve_s),
            target=float(context.target),
            solver_label="xblock_sparse_pc_gmres",
        )
    )
    x_np = np.asarray(post_minres.x, dtype=np.float64)
    residual_norm = float(post_minres.residual_norm)
    solve_s = float(post_minres.solve_s)

    def _post_coarse_direction_builder(
        residual_vec: jnp.ndarray,
    ) -> tuple[tuple[str, jnp.ndarray], ...]:
        return context.coarse_direction_builder(
            residual_vec,
            include_raw=bool(post_coarse_include_raw),
            fsavg_lmax=int(post_coarse_fsavg_lmax),
            angular_lmax=int(post_coarse_angular_lmax),
            max_extra_units=int(post_coarse_max_extra_units),
            max_directions=int(post_coarse_max_directions),
            include_angular_residual=bool(post_coarse_include_angular_residual),
        )

    post_coarse = apply_xblock_subspace_correction_if_needed(
        XBlockSubspaceCorrectionContext(
            matvec=context.matvec,
            rhs=context.rhs,
            x=np.asarray(x_np, dtype=np.float64),
            residual_norm=float(residual_norm),
            target=float(context.target),
            direction_builder=_post_coarse_direction_builder,
            steps=int(post_coarse_steps_requested),
            max_directions=int(post_coarse_max_directions),
            alpha_clip=float(post_coarse_alpha_clip),
            rcond=float(post_coarse_rcond),
            min_improvement=float(post_coarse_min_improvement),
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            correction=context.coarse_correction,
            solver_label="xblock_sparse_pc_gmres",
            correction_label="post-coarse",
        )
    )
    x_np = np.asarray(post_coarse.x, dtype=np.float64)
    residual_norm = float(post_coarse.residual_norm)
    solve_s += float(post_coarse.solve_s)

    return XBlockPostSolveCorrectionResult(
        x=x_np,
        residual_norm=float(residual_norm),
        solve_s=float(solve_s),
        post_minres_steps_requested=int(post_minres_steps_requested),
        post_minres_alpha_clip=float(post_minres_alpha_clip),
        post_minres_min_improvement=float(post_minres_min_improvement),
        post_minres_history=post_minres.history,
        post_minres_alphas=post_minres.alphas,
        post_minres_residual_before=post_minres.residual_before,
        post_minres_residual_after=post_minres.residual_after,
        post_coarse_steps_requested=int(post_coarse_steps_requested),
        post_coarse_max_directions=int(post_coarse_max_directions),
        post_coarse_max_extra_units=int(post_coarse_max_extra_units),
        post_coarse_fsavg_lmax=int(post_coarse_fsavg_lmax),
        post_coarse_angular_lmax=int(post_coarse_angular_lmax),
        post_coarse_include_angular_residual=bool(
            post_coarse_include_angular_residual
        ),
        post_coarse_include_raw=bool(post_coarse_include_raw),
        post_coarse_alpha_clip=float(post_coarse_alpha_clip),
        post_coarse_rcond=float(post_coarse_rcond),
        post_coarse_min_improvement=float(post_coarse_min_improvement),
        post_coarse_history=post_coarse.history,
        post_coarse_direction_counts=post_coarse.direction_counts,
        post_coarse_direction_names=post_coarse.direction_names,
        post_coarse_residual_before=post_coarse.residual_before,
        post_coarse_residual_after=post_coarse.residual_after,
        post_residual_equation_steps_requested=int(
            post_residual_equation_steps_requested
        ),
        post_residual_equation_max_directions=int(
            post_residual_equation_max_directions
        ),
        post_residual_equation_max_extra_units=int(
            post_residual_equation_max_extra_units
        ),
        post_residual_equation_fsavg_lmax=int(post_residual_equation_fsavg_lmax),
        post_residual_equation_angular_lmax=int(
            post_residual_equation_angular_lmax
        ),
        post_residual_equation_include_angular_residual=bool(
            post_residual_equation_include_angular_residual
        ),
        post_residual_equation_include_raw=bool(
            post_residual_equation_include_raw
        ),
        post_residual_equation_include_post_coarse=bool(
            post_residual_equation_include_post_coarse
        ),
        post_residual_equation_include_qi_basis=bool(
            post_residual_equation_include_qi_basis
        ),
        post_residual_equation_alpha_clip=float(post_residual_equation_alpha_clip),
        post_residual_equation_rcond=float(post_residual_equation_rcond),
        post_residual_equation_min_improvement=float(
            post_residual_equation_min_improvement
        ),
        post_residual_equation_history=post_residual_equation.history,
        post_residual_equation_direction_counts=(
            post_residual_equation.direction_counts
        ),
        post_residual_equation_direction_names=post_residual_equation.direction_names,
        post_residual_equation_residual_before=(
            post_residual_equation.residual_before
        ),
        post_residual_equation_residual_after=post_residual_equation.residual_after,
    )


def complete_xblock_post_krylov_stage(
    context: XBlockPostKrylovCompletionContext,
) -> XBlockPostKrylovCompletionResult:
    """Apply x-block post-solve corrections and emit the completion line."""

    corrections = run_xblock_post_solve_corrections(context.corrections)
    emit_xblock_sparse_pc_completion(
        XBlockSparsePCCompletionContext(
            emit=context.corrections.emit,
            krylov_method=str(context.krylov_method),
            elapsed_s=float(context.elapsed_s()),
            iterations=int(context.iterations),
            matvecs=int(context.matvecs),
            residual_norm=float(corrections.residual_norm),
            target=float(context.target),
            history=context.history,
        )
    )
    return XBlockPostKrylovCompletionResult(
        corrections=corrections,
        x=np.asarray(corrections.x, dtype=np.float64),
        residual_norm=float(corrections.residual_norm),
        solve_s=float(corrections.solve_s),
    )



__all__ = (
    "XBlockSparsePCFinalCoreState",
    "XBlockSparsePCFinalDeviceState",
    "XBlockSparsePCFinalPreflightState",
    "XBlockSparsePCFinalNestedMetadata",
    "XBlockSparsePCFinalMetadataStateContext",
    "xblock_sparse_pc_final_metadata_driver_state_keys",
    "xblock_sparse_pc_final_metadata_driver_scope_keys",
    "xblock_sparse_pc_final_metadata_state_from_context",
    "xblock_sparse_pc_final_metadata_state_from_driver_scope",
    "XBlockSubspaceCorrectionContext",
    "XBlockSubspaceCorrectionResult",
    "XBlockPostSolveCorrectionContext",
    "XBlockPostSolveCorrectionResult",
    "XBlockPostKrylovCompletionContext",
    "XBlockPostKrylovCompletionResult",
    "apply_xblock_subspace_correction_if_needed",
    "run_xblock_post_solve_corrections",
    "complete_xblock_post_krylov_stage",
    "XBlockKrylovReport",
    "XBlockSparsePCCompletionContext",
    "XBlockSparsePCFinalPayloadContext",
    "xblock_sparse_pc_final_metadata_from_driver_state",
    "xblock_sparse_pc_final_payload_from_driver_state",
    "xblock_sparse_pc_final_payload",
    "XBlockGMRESFallbackDecision",
    "XBlockGMRESFallbackContext",
    "XBlockGMRESFallbackResult",
    "XBlockDeviceKrylovState",
    "XBlockFirstKrylovAttemptContext",
    "XBlockFirstKrylovAttemptResult",
    "XBlockSideProbeStageContext",
    "XBlockSideProbeStageResult",
    "XBlockProbeCoarseStageContext",
    "XBlockProbeCoarseStageResult",
    "XBlockPreflightGateContext",
    "XBlockPreflightGateResult",
    "XBlockKrylovControlSetupContext",
    "XBlockKrylovControlSetup",
    "XBlockKrylovProgressCallbacksContext",
    "XBlockKrylovProgressCallbacks",
    "XBlockKrylovSolveState",
    "XBlockFirstKrylovSolveStateContext",
    "XBlockKrylovSolveStageContext",
    "XBlockKrylovSolveStageResult",
    "XBlockKrylovSolveSpaceContext",
    "XBlockKrylovSolveSpace",
    "XBlockAugmentedKrylovBasisContext",
    "XBlockAugmentedKrylovBasisResult",
    "XBlockAugmentedKrylovStageContext",
    "XBlockAugmentedKrylovStageResult",
    "XBlockSparsePCWorkEstimates",
    "XBlockPhysicalResidual",
    "xblock_krylov_report",
    "apply_xblock_side_probe_stage",
    "apply_xblock_probe_coarse_stage",
    "evaluate_xblock_preflight_gate",
    "resolve_xblock_krylov_control_setup",
    "xblock_krylov_state_from_first_attempt",
    "xblock_krylov_state_from_gmres_fallback",
    "run_xblock_krylov_solve_stage",
    "xblock_device_cycle_progress_message",
    "xblock_host_krylov_progress_message",
    "build_xblock_krylov_progress_callbacks",
    "xblock_device_krylov_state",
    "prepare_xblock_krylov_solve_space",
    "prepare_xblock_augmented_krylov_basis",
    "apply_xblock_augmented_krylov_stage",
    "run_xblock_first_krylov_attempt",
    "xblock_gmres_fallback_decision",
    "run_xblock_gmres_fallback_if_needed",
    "xblock_sparse_pc_work_estimates",
    "xblock_sparse_pc_completion_message",
    "emit_xblock_sparse_pc_completion",
    "emit_xblock_sparse_pc_completion_from_driver_state",
    "xblock_physical_solution_and_residual",
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
