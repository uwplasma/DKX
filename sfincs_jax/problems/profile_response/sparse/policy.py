"""Sparse profile-response policy and admission helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..setup import (
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
)

ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]

__all__ = (
    "SparsePCActiveDOFSetup",
    "SparsePCEntryPolicySetup",
    "SparsePCFactorPolicySetup",
    "SparsePCPatternSetupContext",
    "SparsePCPatternSetupResult",
    "SparsePCMemoryBudgetPreflightContext",
    "SparsePCFactorPreflightPolicyContext",
    "SparsePCFactorPreflightPolicy",
    "SparsePCFactorPreflightEvaluationContext",
    "SparsePCFactorPreflightEvaluationResult",
    "SparsePCResidualCandidateAcceptanceContext",
    "SparsePCResidualCandidateAcceptanceResult",
    "SparsePCAutoPreflightRetrySelectionContext",
    "SparsePCAutoPreflightRetrySelectionResult",
    "SparsePCAutoPreflightRetryEvaluationContext",
    "SparsePCAutoPreflightRetryEvaluationResult",
    "SparsePCGMRESControlPolicy",
    "build_sparse_pc_active_dof_setup",
    "build_sparse_pc_pattern_setup",
    "resolve_sparse_pc_entry_policy",
    "resolve_sparse_pc_factor_policy",
    "enforce_sparse_pc_memory_budget",
    "resolve_sparse_pc_factor_preflight_policy",
    "evaluate_sparse_pc_factor_preflight",
    "evaluate_sparse_pc_residual_candidate_acceptance",
    "select_sparse_pc_auto_preflight_retry_candidates",
    "evaluate_sparse_pc_auto_preflight_retry",
    "resolve_sparse_pc_gmres_control_policy",
    "_env_value",
    "_env_float",
    "_env_int",
    "_env_bool",
)

@dataclass(frozen=True)
class SparsePCActiveDOFSetup:
    """Active-DOF maps and vector routing for the generic sparse-PC path."""

    active_idx_np: np.ndarray | None
    active_idx_jnp: jnp.ndarray | None
    full_to_active_jnp: jnp.ndarray | None
    rhs: jnp.ndarray
    linear_size: int
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    messages: tuple[tuple[int, str], ...]


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


@dataclass(frozen=True)
class SparsePCFactorPolicySetup:
    """Host sparse-PC factor policy resolved before materializing the matrix."""

    pc_shift: float
    factorization: str
    default_factor_kind: str
    default_ilu_fill_factor: float
    default_ilu_drop_tol: float
    default_pattern_color_batch: int
    factor_dtype_initial: np.dtype
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    default_permc_spec: str
    permc_spec: str
    fp32_probe_maxiter: int
    first_attempt_maxiter: int


@dataclass(frozen=True)
class SparsePCPatternSetupContext:
    """Inputs for selecting and summarizing a generic sparse-PC pattern."""

    op: object
    pattern_source_op: object
    fortran_reduced_sparse_pc: bool
    sparse_pc_use_active_dof: bool
    active_idx_np: np.ndarray | None
    preconditioner_x: int
    preconditioner_xi: int
    preconditioner_species: int
    preconditioner_x_min_l: int
    fp_dense_velocity_block: bool | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    fortran_reduced_pattern_for_indices: Callable[..., object]
    fortran_reduced_pattern: Callable[..., object]
    conservative_pattern_for_indices: Callable[..., object]
    conservative_pattern: Callable[..., object]
    summarize_pattern: Callable[[object, object], object]


@dataclass(frozen=True)
class SparsePCPatternSetupResult:
    """Selected sparse-PC pattern, scope, timing, and summary."""

    pattern: object
    scope: str
    build_s: float
    summary: object


@dataclass(frozen=True)
class SparsePCMemoryBudgetPreflightContext:
    """Inputs for the optional generic sparse-PC memory-budget guard."""

    env: Mapping[str, str] | None
    unknowns: int
    gmres_restart: int
    csr_nnz: int
    dtype: object
    device_count: int
    estimate_sparse_pc_memory: Callable[..., object]


@dataclass(frozen=True)
class SparsePCFactorPreflightPolicyContext:
    """Inputs for sparse-PC factor residual-preflight policy parsing."""

    env: Mapping[str, str] | None
    fortran_reduced_sparse_pc: bool
    structured_pc_ready: bool
    structured_pc_metadata: dict[str, object] | None
    sparse_pc_linear_size: int


@dataclass(frozen=True)
class SparsePCFactorPreflightPolicy:
    """Resolved sparse-PC factor residual-preflight controls."""

    factor_preflight_enabled: bool
    factor_preflight_required: bool
    factor_preflight_seed_enabled: bool
    structured_pc_preflight_required_min_size: int
    direct_tail_structured_pc_requires_preflight: bool
    direct_tail_structured_pc_kind_for_preflight: str
    direct_tail_structured_pc_size_requires_preflight: bool
    structured_pc_preflight_required: bool
    factor_preflight_max_target_ratio: float


@dataclass(frozen=True)
class SparsePCFactorPreflightEvaluationContext:
    """Callbacks and controls for one sparse-PC factor residual preflight."""

    rhs: jnp.ndarray
    rhs_norm: float
    target: float
    preconditioner: ArrayFn
    matvec: ArrayFn
    diagnostics: Callable[..., dict[str, object]]
    layout: Any
    active_indices: np.ndarray | None
    seed_enabled: bool
    max_target_ratio: float


@dataclass(frozen=True)
class SparsePCFactorPreflightEvaluationResult:
    """Residual-preflight result for sparse-PC factor admission."""

    residual_before: float
    residual_after: float
    improvement_ratio: float | None
    target_ratio: float | None
    diagnostics: dict[str, object] | None
    seed_used: bool
    passed: bool
    x_seed: jnp.ndarray
    residual_vec: jnp.ndarray
    x0_seed: jnp.ndarray | None


@dataclass(frozen=True)
class SparsePCResidualCandidateAcceptanceContext:
    """Residual-admission controls for one sparse-PC rescue candidate."""

    candidate_residual_after: float
    current_residual_after: float | None
    original_residual_before: float | None
    target: float
    max_target_ratio: float
    seed_enabled: bool
    require_original_improvement: bool = True
    current_min_improvement: float = 0.0
    accept_base_improvement: bool = False
    base_improvement_requires_original_miss: bool = True
    base_improvement_sets_passed: bool = False
    missing_original_improves: bool = False


@dataclass(frozen=True)
class SparsePCResidualCandidateAcceptanceResult:
    """Admission and post-admission residual metrics for a rescue candidate."""

    finite_candidate: bool
    improves_current_residual: bool
    improves_original_residual: bool
    strict_accept: bool
    base_improvement_accept: bool
    accepted: bool
    base_improvement_override_used: bool
    residual_after: float
    improvement_ratio: float | None
    target_ratio: float | None
    passed: bool
    seed_used: bool


@dataclass(frozen=True)
class SparsePCAutoPreflightRetrySelectionContext:
    """Metadata and policy controls used to choose auto-preflight retry kinds."""

    metadata: Mapping[str, object] | None
    current_kind: str
    sparse_pc_linear_size: int
    preflight_required_min_size: int
    skip_large_kinds_raw: str
    max_candidates: int


@dataclass(frozen=True)
class SparsePCAutoPreflightRetrySelectionResult:
    """Normalized retry candidates derived from an auto structured-PC attempt."""

    selected_kind: str
    auto_candidates: tuple[str, ...]
    rejected_kinds: frozenset[str]
    retry_candidates: tuple[str, ...]


@dataclass(frozen=True)
class SparsePCAutoPreflightRetryEvaluationContext:
    """Scalar preflight policy inputs for one auto-retry candidate."""

    residual_after: float
    target: float
    max_target_ratio: float
    residual_before: float | None
    sparse_pc_linear_size: int
    preflight_required_min_size: int
    retry_kind: str
    retry_metadata: Mapping[str, object] | None


@dataclass(frozen=True)
class SparsePCAutoPreflightRetryEvaluationResult:
    """Preflight decision for one auto-retry candidate."""

    target_ratio: float
    requires_metadata: bool
    requires_size: bool
    required: bool
    preflight_passed: bool
    policy_passed: bool


@dataclass(frozen=True)
class SparsePCGMRESControlPolicy:
    """Resolved sparse-PC GMRES progress, stagnation, and polish controls."""

    stagnation_abort: bool
    stagnation_min_iter: int
    stagnation_window: int
    stagnation_rel_improvement: float
    post_minres_steps: int
    post_minres_alpha_clip: float
    post_minres_min_improvement: float


def build_sparse_pc_active_dof_setup(
    *,
    op: object,
    rhs: jnp.ndarray,
    sparse_pc_use_active_dof: bool,
    active_dof_indices: Callable[[object], np.ndarray],
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
) -> SparsePCActiveDOFSetup:
    """Build active-DOF reduction maps for the generic sparse-PC solve."""

    if not bool(sparse_pc_use_active_dof):
        return SparsePCActiveDOFSetup(
            active_idx_np=None,
            active_idx_jnp=None,
            full_to_active_jnp=None,
            rhs=rhs,
            linear_size=int(op.total_size),
            reduce_full=lambda v_full: v_full,
            expand_reduced=lambda v_vec: v_vec,
            messages=(),
        )

    active_idx_np = np.asarray(active_dof_indices(op))
    active_idx_jnp = jnp.asarray(active_idx_np, dtype=jnp.int32)
    full_to_active_np = np.zeros((int(op.total_size),), dtype=np.int32)
    full_to_active_np[np.asarray(active_idx_np, dtype=np.int32)] = np.arange(
        1,
        int(active_idx_np.shape[0]) + 1,
        dtype=np.int32,
    )
    full_to_active_jnp = jnp.asarray(full_to_active_np, dtype=jnp.int32)

    def reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return reduce_full_with_indices(v_full, active_idx_jnp)

    def expand_reduced(v_vec: jnp.ndarray) -> jnp.ndarray:
        return expand_reduced_with_map(v_vec, full_to_active_jnp)

    linear_size = int(active_idx_np.shape[0])
    return SparsePCActiveDOFSetup(
        active_idx_np=active_idx_np,
        active_idx_jnp=active_idx_jnp,
        full_to_active_jnp=full_to_active_jnp,
        rhs=rhs[active_idx_jnp],
        linear_size=linear_size,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        messages=(
            (
                1,
                "solve_v3_full_system_linear_gmres: sparse_pc_gmres active-DOF reduction "
                f"enabled (size={int(linear_size)}/{int(op.total_size)})",
            ),
        ),
    )


def build_sparse_pc_pattern_setup(
    context: SparsePCPatternSetupContext,
) -> SparsePCPatternSetupResult:
    """Build the conservative sparse-PC pattern and its diagnostic summary."""

    pattern_start_s = float(context.elapsed_s())
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres building conservative pattern",
        )

    if bool(context.fortran_reduced_sparse_pc):
        if bool(context.sparse_pc_use_active_dof):
            if context.active_idx_np is None:
                raise AssertionError("sparse_pc active indices are required")
            pattern = context.fortran_reduced_pattern_for_indices(
                context.pattern_source_op,
                np.asarray(context.active_idx_np, dtype=np.int32),
                preconditioner_x=int(context.preconditioner_x),
                preconditioner_xi=int(context.preconditioner_xi),
                preconditioner_species=int(context.preconditioner_species),
                preconditioner_x_min_l=int(context.preconditioner_x_min_l),
            )
            scope = "fortran_reduced_active_dof"
        else:
            pattern = context.fortran_reduced_pattern(
                context.pattern_source_op,
                preconditioner_x=int(context.preconditioner_x),
                preconditioner_xi=int(context.preconditioner_xi),
                preconditioner_species=int(context.preconditioner_species),
                preconditioner_x_min_l=int(context.preconditioner_x_min_l),
            )
            scope = "fortran_reduced_full"
    elif bool(context.sparse_pc_use_active_dof):
        if context.active_idx_np is None:
            raise AssertionError("sparse_pc active indices are required")
        pattern = context.conservative_pattern_for_indices(
            context.pattern_source_op,
            np.asarray(context.active_idx_np, dtype=np.int32),
            fp_dense_velocity_block=context.fp_dense_velocity_block,
        )
        scope = "active_dof"
    else:
        pattern = context.conservative_pattern(
            context.pattern_source_op,
            fp_dense_velocity_block=context.fp_dense_velocity_block,
        )
        scope = "full"

    build_s = float(context.elapsed_s()) - pattern_start_s
    summary = context.summarize_pattern(context.op, pattern)
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres pattern "
            f"scope={scope} nnz={summary.nnz} "
            f"avg_row_nnz={summary.avg_row_nnz:.3g} max_row_nnz={summary.max_row_nnz}",
        )

    return SparsePCPatternSetupResult(
        pattern=pattern,
        scope=str(scope),
        build_s=float(build_s),
        summary=summary,
    )


def _env_value(env: object, key: str) -> str:
    if env is None:
        return ""
    try:
        return str(env.get(key, "")).strip()  # type: ignore[union-attr]
    except AttributeError:
        return ""


def _env_float(env: object, key: str, default: float) -> float:
    raw = _env_value(env, key)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(env: object, key: str, default: int, minimum: int | None = None) -> int:
    raw = _env_value(env, key)
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), int(value))
    return int(value)


def _env_bool(env: object, key: str, default: bool = False) -> bool:
    raw = _env_value(env, key).lower()
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    return bool(default)


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


def resolve_sparse_pc_factor_policy(
    *,
    env: Mapping[str, str] | None,
    constrained_pas_pc: bool,
    tokamak_fp_pc: bool,
    fortran_reduced_sparse_pc: bool,
    sparse_pc_linear_size: int,
    pc_maxiter: int,
    default_permc_spec: str,
    host_sparse_factor_dtype: Callable[..., np.dtype],
) -> SparsePCFactorPolicySetup:
    """Resolve host sparse-PC factor controls before matrix materialization."""

    shift_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_SHIFT")
    default_shift = (
        1.0e-8
        if (bool(constrained_pas_pc) or bool(tokamak_fp_pc) or bool(fortran_reduced_sparse_pc))
        else 0.0
    )
    try:
        pc_shift = float(shift_env) if shift_env else float(default_shift)
    except ValueError:
        pc_shift = float(default_shift)

    factor_kind_env = _env_value(env, "SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND").lower()
    default_factor_kind = (
        "ilu"
        if bool(fortran_reduced_sparse_pc) and int(sparse_pc_linear_size) >= 100000
        else "lu"
    )
    if factor_kind_env in {"jacobi", "diagonal", "diag", "none"}:
        factorization = "jacobi"
    elif factor_kind_env in {"ilu", "spilu"}:
        factorization = "ilu"
    elif factor_kind_env in {"lu", "splu"}:
        factorization = "lu"
    else:
        factorization = str(default_factor_kind)

    default_ilu_fill_factor = (
        2.0
        if bool(fortran_reduced_sparse_pc) and int(sparse_pc_linear_size) >= 100000
        else 10.0
    )
    default_ilu_drop_tol = (
        1.0e-3
        if bool(fortran_reduced_sparse_pc) and int(sparse_pc_linear_size) >= 100000
        else 1.0e-4
    )
    default_pattern_color_batch = (
        16
        if bool(fortran_reduced_sparse_pc) and int(sparse_pc_linear_size) >= 100000
        else 1
    )

    dtype_env = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_DTYPE").lower()
    if dtype_env in {"float32", "fp32", "32"}:
        factor_dtype_initial = np.dtype(np.float32)
    elif dtype_env in {"float64", "fp64", "64"}:
        factor_dtype_initial = np.dtype(np.float64)
    elif _env_value(env, "SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE"):
        factor_dtype_initial = np.dtype(
            host_sparse_factor_dtype(
                size=int(sparse_pc_linear_size),
                factorization=str(factorization),
                use_implicit=False,
            )
        )
    else:
        # Sparse-PC GMRES is more sensitive to factor quality than direct
        # fallback solves, so default to FP64 unless a memory experiment opts in.
        factor_dtype_initial = np.dtype(np.float64)

    permc_env = _env_value(env, "SFINCS_JAX_EXPLICIT_SPARSE_PERMC_SPEC").upper()
    permc_spec = (
        permc_env
        if permc_env in {"NATURAL", "MMD_ATA", "MMD_AT_PLUS_A", "COLAMD"}
        else str(default_permc_spec)
    )
    fp32_probe_maxiter = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_FP32_PROBE_MAXITER",
        2,
        minimum=1,
    )
    first_attempt_maxiter = (
        min(int(pc_maxiter), int(fp32_probe_maxiter))
        if factor_dtype_initial == np.dtype(np.float32)
        else int(pc_maxiter)
    )

    return SparsePCFactorPolicySetup(
        pc_shift=float(pc_shift),
        factorization=str(factorization),
        default_factor_kind=str(default_factor_kind),
        default_ilu_fill_factor=float(default_ilu_fill_factor),
        default_ilu_drop_tol=float(default_ilu_drop_tol),
        default_pattern_color_batch=int(default_pattern_color_batch),
        factor_dtype_initial=np.dtype(factor_dtype_initial),
        factor_dtype_used=np.dtype(factor_dtype_initial),
        factor_dtype_retry=None,
        default_permc_spec=str(default_permc_spec),
        permc_spec=str(permc_spec),
        fp32_probe_maxiter=int(fp32_probe_maxiter),
        first_attempt_maxiter=int(first_attempt_maxiter),
    )


def enforce_sparse_pc_memory_budget(
    context: SparsePCMemoryBudgetPreflightContext,
) -> None:
    """Apply the optional sparse-PC memory budget guard before factor setup."""

    budget_env = _env_value(
        context.env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB",
    )
    if not budget_env:
        return
    try:
        budget_mb = float(budget_env)
    except ValueError:
        budget_mb = 0.0
    if budget_mb <= 0.0:
        return

    fill_env = _env_value(
        context.env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_FILL_ESTIMATE",
    )
    try:
        fill_estimate = float(fill_env) if fill_env else 8.0
    except ValueError:
        fill_estimate = 8.0
    memory_estimate = context.estimate_sparse_pc_memory(
        unknowns=int(context.unknowns),
        gmres_restart=int(context.gmres_restart),
        csr_nnz=int(context.csr_nnz),
        dtype=context.dtype,
        factor_fill_estimate=float(fill_estimate),
        device_count=max(1, int(context.device_count)),
    )
    estimated_mb = (
        float(memory_estimate.csr_total_nbytes or memory_estimate.dense_total_nbytes)
        / 1.0e6
    )
    if estimated_mb > float(budget_mb):
        raise MemoryError(
            "sparse_pc_gmres memory preflight exceeds "
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_MAX_ESTIMATED_MB: "
            f"estimated={estimated_mb:.3f} MB budget={float(budget_mb):.3f} MB "
            f"unknowns={int(context.unknowns)} csr_nnz={int(context.csr_nnz)} "
            f"restart={int(context.gmres_restart)} factor_fill_estimate={float(fill_estimate):.3g}. "
            "Raise the budget, lower the resolution, or use a lower-memory matrix-free route."
        )


def resolve_sparse_pc_factor_preflight_policy(
    context: SparsePCFactorPreflightPolicyContext,
) -> SparsePCFactorPreflightPolicy:
    """Resolve factor-preflight gates for sparse-PC production solves."""

    factor_preflight_enabled = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT",
        default=bool(context.fortran_reduced_sparse_pc),
    )
    factor_preflight_required = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_REQUIRED",
        default=False,
    )
    factor_preflight_seed_enabled = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_SEED",
        default=True,
    )
    structured_pc_preflight_required_min_size = _env_int(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        100_000,
        minimum=1,
    )
    direct_tail_structured_pc_requires_preflight = False
    direct_tail_structured_pc_kind_for_preflight = ""
    if bool(context.structured_pc_ready) and isinstance(context.structured_pc_metadata, dict):
        direct_tail_structured_pc_kind_for_preflight = (
            str(context.structured_pc_metadata.get("kind", "")).strip().lower().replace("-", "_")
        )
        structured_pc_metadata_inner = context.structured_pc_metadata.get("metadata")
        if isinstance(structured_pc_metadata_inner, dict):
            direct_tail_structured_pc_requires_preflight = bool(
                structured_pc_metadata_inner.get("requires_preflight", False)
            )
            if not direct_tail_structured_pc_kind_for_preflight:
                direct_tail_structured_pc_kind_for_preflight = (
                    str(structured_pc_metadata_inner.get("requested_kind", "")).strip().lower().replace("-", "_")
                )
    direct_tail_structured_pc_size_requires_preflight = bool(
        context.structured_pc_ready
        and int(context.sparse_pc_linear_size) >= int(structured_pc_preflight_required_min_size)
        and direct_tail_structured_pc_kind_for_preflight != "active_fortran_v3_reduced_lu"
    )
    structured_pc_preflight_required = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED",
        default=bool(
            context.structured_pc_ready
            and (
                bool(direct_tail_structured_pc_requires_preflight)
                or bool(direct_tail_structured_pc_size_requires_preflight)
            )
        ),
    )
    factor_preflight_max_target_ratio = max(
        1.0,
        _env_float(
            context.env,
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_FACTOR_PREFLIGHT_MAX_TARGET_RATIO",
            1.0e6,
        ),
    )
    return SparsePCFactorPreflightPolicy(
        factor_preflight_enabled=bool(factor_preflight_enabled),
        factor_preflight_required=bool(factor_preflight_required),
        factor_preflight_seed_enabled=bool(factor_preflight_seed_enabled),
        structured_pc_preflight_required_min_size=int(structured_pc_preflight_required_min_size),
        direct_tail_structured_pc_requires_preflight=bool(direct_tail_structured_pc_requires_preflight),
        direct_tail_structured_pc_kind_for_preflight=str(direct_tail_structured_pc_kind_for_preflight),
        direct_tail_structured_pc_size_requires_preflight=bool(
            direct_tail_structured_pc_size_requires_preflight
        ),
        structured_pc_preflight_required=bool(structured_pc_preflight_required),
        factor_preflight_max_target_ratio=float(factor_preflight_max_target_ratio),
    )


def evaluate_sparse_pc_factor_preflight(
    context: SparsePCFactorPreflightEvaluationContext,
) -> SparsePCFactorPreflightEvaluationResult:
    """Evaluate whether the current sparse factor is a useful seed."""

    residual_before = float(context.rhs_norm)
    rhs = jnp.asarray(context.rhs, dtype=jnp.float64)
    x_seed = jnp.asarray(context.preconditioner(rhs), dtype=jnp.float64)
    residual_vec = rhs - jnp.asarray(context.matvec(x_seed), dtype=jnp.float64)
    residual_after = float(jnp.linalg.norm(residual_vec))
    residual_diagnostics = context.diagnostics(
        residual=residual_vec,
        layout=context.layout,
        active_indices=context.active_indices,
    )
    improvement_ratio: float | None = None
    if residual_before > 0.0 and np.isfinite(float(residual_after)):
        improvement_ratio = float(residual_before) / max(float(residual_after), 1.0e-300)
    target_ratio: float | None = None
    if float(context.target) > 0.0:
        target_ratio = (
            float(residual_after) / float(context.target)
            if np.isfinite(float(residual_after))
            else float("inf")
        )
    passed = bool(
        np.isfinite(float(residual_after))
        and float(residual_after) < float(residual_before)
        and (
            target_ratio is None
            or float(target_ratio) <= float(context.max_target_ratio)
        )
    )
    seed_used = bool(
        context.seed_enabled
        and np.isfinite(float(residual_after))
        and float(residual_after) < float(residual_before)
    )
    return SparsePCFactorPreflightEvaluationResult(
        residual_before=float(residual_before),
        residual_after=float(residual_after),
        improvement_ratio=improvement_ratio,
        target_ratio=target_ratio,
        diagnostics=residual_diagnostics,
        seed_used=bool(seed_used),
        passed=bool(passed),
        x_seed=x_seed,
        residual_vec=residual_vec,
        x0_seed=x_seed if bool(seed_used) else None,
    )


def evaluate_sparse_pc_residual_candidate_acceptance(
    context: SparsePCResidualCandidateAcceptanceContext,
) -> SparsePCResidualCandidateAcceptanceResult:
    """Evaluate residual admission for one sparse-PC rescue candidate.

    The driver owns candidate construction and state updates; this helper owns
    only the scalar residual bookkeeping that every rescue path repeats.
    """

    candidate = float(context.candidate_residual_after)
    current = context.current_residual_after
    original = context.original_residual_before
    finite_candidate = bool(np.isfinite(candidate))

    improves_current = False
    if current is not None and np.isfinite(float(current)):
        threshold = float(current) * (1.0 - float(context.current_min_improvement))
        improves_current = bool(finite_candidate and candidate < threshold)

    if original is None:
        improves_original = bool(context.missing_original_improves)
    else:
        improves_original = bool(
            finite_candidate
            and np.isfinite(float(original))
            and candidate < float(original)
        )

    strict_accept = bool(
        improves_current
        and (
            not bool(context.require_original_improvement)
            or bool(improves_original)
        )
    )
    base_improvement_accept = bool(
        context.accept_base_improvement
        and improves_current
        and (
            not bool(context.base_improvement_requires_original_miss)
            or not bool(improves_original)
        )
    )
    accepted = bool(strict_accept or base_improvement_accept)
    base_improvement_override_used = bool(base_improvement_accept)

    improvement_ratio: float | None = None
    if original is not None and float(original) > 0.0 and finite_candidate:
        improvement_ratio = float(original) / max(candidate, 1.0e-300)

    target_ratio: float | None = None
    if float(context.target) > 0.0:
        target_ratio = candidate / float(context.target) if finite_candidate else float("inf")

    passed = bool(
        finite_candidate
        and original is not None
        and candidate < float(original)
        and (
            target_ratio is None
            or float(target_ratio) <= float(context.max_target_ratio)
        )
    )
    if bool(base_improvement_override_used) and bool(context.base_improvement_sets_passed):
        passed = True

    seed_used = bool(context.seed_enabled and original is not None and finite_candidate and candidate < float(original))

    return SparsePCResidualCandidateAcceptanceResult(
        finite_candidate=bool(finite_candidate),
        improves_current_residual=bool(improves_current),
        improves_original_residual=bool(improves_original),
        strict_accept=bool(strict_accept),
        base_improvement_accept=bool(base_improvement_accept),
        accepted=bool(accepted),
        base_improvement_override_used=bool(base_improvement_override_used),
        residual_after=float(candidate),
        improvement_ratio=improvement_ratio,
        target_ratio=target_ratio,
        passed=bool(passed),
        seed_used=bool(seed_used),
    )


def _normalize_sparse_pc_kind(value: object) -> str:
    return str(value).strip().lower().replace("-", "_")


def select_sparse_pc_auto_preflight_retry_candidates(
    context: SparsePCAutoPreflightRetrySelectionContext,
) -> SparsePCAutoPreflightRetrySelectionResult:
    """Select bounded auto-retry candidates after a failed preflight."""

    metadata = context.metadata if isinstance(context.metadata, Mapping) else {}
    auto_candidates_raw = metadata.get("auto_candidates", ())
    rejected_raw = metadata.get("auto_rejected_candidates", ())
    selected_kind = _normalize_sparse_pc_kind(
        metadata.get("auto_selected_kind", context.current_kind)
    )
    auto_candidates = tuple(
        candidate
        for candidate in (_normalize_sparse_pc_kind(candidate) for candidate in auto_candidates_raw)
        if candidate
    )

    rejected_kinds: set[str] = set()
    if isinstance(rejected_raw, (tuple, list)):
        for entry in rejected_raw:
            if isinstance(entry, Mapping):
                rejected_kind = _normalize_sparse_pc_kind(entry.get("kind", ""))
                if rejected_kind:
                    rejected_kinds.add(rejected_kind)

    try:
        selected_index = auto_candidates.index(selected_kind)
        retry_candidates = auto_candidates[selected_index + 1 :]
    except ValueError:
        retry_candidates = auto_candidates

    retry_candidates = tuple(
        candidate
        for candidate in retry_candidates
        if candidate
        and candidate not in rejected_kinds
        and candidate not in {"auto", "active_auto", "structured", "structured_auto"}
    )
    if int(context.sparse_pc_linear_size) >= int(context.preflight_required_min_size):
        skip_large_kinds = {
            item
            for item in (
                _normalize_sparse_pc_kind(raw_item)
                for raw_item in str(context.skip_large_kinds_raw).split(",")
            )
            if item
        }
        retry_candidates = tuple(
            candidate for candidate in retry_candidates if candidate not in skip_large_kinds
        )

    max_candidates = max(1, int(context.max_candidates))
    return SparsePCAutoPreflightRetrySelectionResult(
        selected_kind=str(selected_kind),
        auto_candidates=tuple(auto_candidates),
        rejected_kinds=frozenset(rejected_kinds),
        retry_candidates=tuple(retry_candidates[:max_candidates]),
    )


def evaluate_sparse_pc_auto_preflight_retry(
    context: SparsePCAutoPreflightRetryEvaluationContext,
) -> SparsePCAutoPreflightRetryEvaluationResult:
    """Evaluate one auto-preflight retry residual against its policy gate."""

    residual = float(context.residual_after)
    target_ratio = (
        residual / float(context.target)
        if float(context.target) > 0.0 and np.isfinite(residual)
        else float("inf")
    )
    retry_kind = _normalize_sparse_pc_kind(context.retry_kind)
    retry_metadata = context.retry_metadata if isinstance(context.retry_metadata, Mapping) else {}
    requires_metadata = bool(retry_metadata.get("requires_preflight", False))
    requires_size = bool(
        int(context.sparse_pc_linear_size) >= int(context.preflight_required_min_size)
        and retry_kind != "active_fortran_v3_reduced_lu"
    )
    required = bool(requires_metadata or requires_size)
    preflight_passed = bool(
        np.isfinite(residual)
        and context.residual_before is not None
        and residual < float(context.residual_before)
        and float(target_ratio) <= float(context.max_target_ratio)
    )
    policy_passed = bool((not required) or preflight_passed)
    return SparsePCAutoPreflightRetryEvaluationResult(
        target_ratio=float(target_ratio),
        requires_metadata=bool(requires_metadata),
        requires_size=bool(requires_size),
        required=bool(required),
        preflight_passed=bool(preflight_passed),
        policy_passed=bool(policy_passed),
    )


def resolve_sparse_pc_gmres_control_policy(
    env: Mapping[str, str] | None,
) -> SparsePCGMRESControlPolicy:
    """Resolve sparse-PC GMRES stagnation and post-minres controls."""

    return SparsePCGMRESControlPolicy(
        stagnation_abort=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_ABORT",
            default=False,
        ),
        stagnation_min_iter=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_MIN_ITER",
            default=500,
            minimum=1,
        ),
        stagnation_window=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_WINDOW",
            default=500,
            minimum=1,
        ),
        stagnation_rel_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_REL_IMPROVEMENT",
                default=1.0e-3,
            ),
        ),
        post_minres_steps=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_STEPS",
            default=0,
            minimum=0,
        ),
        post_minres_alpha_clip=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_ALPHA_CLIP",
                default=10.0,
            ),
        ),
        post_minres_min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_MIN_IMPROVEMENT",
                default=0.0,
            ),
        ),
    )
