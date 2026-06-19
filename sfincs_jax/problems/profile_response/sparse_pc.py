"""Host sparse-PC Krylov helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .diagnostics import (
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockSparsePCCoreDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    fortran_reduced_xblock_result_metadata,
    fp_xblock_global_correction_metadata,
    fp_xblock_highx_residual_correction_metadata,
    sparse_pc_gmres_result_metadata,
    sparse_rescue_tail_metadata,
    sparse_xblock_rescue_metadata,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_device_krylov_diagnostics,
    xblock_qi_deflated_preconditioner_diagnostics,
    xblock_qi_device_preconditioner_diagnostics,
    xblock_qi_seed_preconditioner_diagnostics,
    xblock_sparse_pc_core_diagnostics,
    xblock_sparse_pc_result_diagnostics_from_driver_state,
    xblock_side_probe_diagnostics,
)
from .residual import (
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
)
from .setup import (
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
)
from .solver_diagnostics import build_rhs1_xblock_correction_metadata_from_driver_state


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
class SparsePCGMRESFinalPayload:
    """Driver-independent payload for constructing the final sparse-PC result."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class SparseMinimumNormPolicy:
    """Host LSQR/LSMR controls for sparse minimum-norm solves."""

    solver_name: str
    atol: float
    btol: float
    conlim: float
    damp: float
    maxiter: int
    show: bool
    petsc_compat_requested: bool


@dataclass(frozen=True)
class SparseMinimumNormPayload:
    """Driver-independent payload for a sparse minimum-norm solve."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]
    start_message: str
    completion_message: str


@dataclass(frozen=True)
class SparseHostDirectPayload:
    """Driver-independent payload for an explicit host sparse direct solve."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]
    completion_message: str


@dataclass(frozen=True)
class SparseHostDirectFactorSolvePayload:
    """Host direct-solve result from an explicit factor or fallback ILU factor."""

    x: np.ndarray
    residual_norm: float
    used_explicit_factor: bool


@dataclass(frozen=True)
class SparseHostDirectPolishPayload:
    """Post-direct-solve polish result for host sparse direct fallback solves."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    attempted: bool
    accepted: bool
    restart: int | None
    maxiter: int | None


@dataclass(frozen=True)
class ExplicitSparseOperatorBuildPolicy:
    """Materialization controls shared by explicit host sparse solve paths."""

    csr_max_mb: float
    drop_tol: float


@dataclass(frozen=True)
class ExplicitSparseOperatorBuildResult:
    """Materialized explicit sparse operator and stable progress messages."""

    operator_bundle: object
    policy: ExplicitSparseOperatorBuildPolicy
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class SparsePCGMRESCompletionMessageContext:
    """Fields used to format the sparse-PC GMRES completion progress line."""

    elapsed_s: float
    iterations: int
    matvecs: int
    residual_norm: float
    target: float
    preconditioned_residual_norm: float
    history: Sequence[float]


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
class SparsePCPostMinresUpdateContext:
    """Current sparse-PC solve state for optional post-minres polishing."""

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
    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    solve_s: float
    target: float


@dataclass(frozen=True)
class SparsePCPostMinresUpdateResult:
    """Updated sparse-PC state and diagnostics after optional post-minres."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    alphas: tuple[float, ...]
    residual_before: float | None
    residual_after: float | None
    error: str | None
    solve_s: float


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
class FortranReducedSparsePCBackendSetup:
    """Backend and direct-tail policy for fortran-reduced sparse-PC solves."""

    backend_raw: str
    xblock_min_size: int
    backend_ignored_env: bool
    direct_tail_pc_env: str
    direct_tail_pc_explicit: bool
    direct_tail_structured_pc_required: bool
    direct_tail_default: bool
    direct_tail_enabled: bool
    direct_tail_structured_pc_forces_global: bool
    direct_tail_auto_forces_global: bool
    direct_tail_auto_structured_pc_forces_global: bool
    backend: str
    reason: str
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FortranReducedXBlockFactorPolicySetup:
    """Local x-block factor controls for fortran-reduced sparse-PC solves."""

    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    preconditioner_xi: int
    promote_xi: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FortranReducedXBlockKrylovPolicySetup:
    """Krylov-side, method, progress, and matvec-counter controls."""

    side_env: str
    precondition_side: str
    pc_form: str
    krylov_method: str
    progress_every: int
    mv_count: "MatvecCounter"
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FortranReducedXBlockInitialSeedPolicySetup:
    """Initial-seed refinement controls for fortran-reduced x-block solves."""

    enabled: bool
    refine_steps: int
    accept_ratio: float


@dataclass(frozen=True)
class FortranReducedXBlockInitialSeedResult:
    """Accepted initial seed and diagnostics for fortran-reduced x-block solves."""

    x0: jnp.ndarray | None
    used: bool
    residual_norm: float | None
    improvement_ratio: float | None
    refines_performed: int
    elapsed_s: float
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class FortranReducedXBlockKrylovSetupContext:
    """Dependencies for fortran-reduced x-block Krylov policy and matvec setup."""

    op: object
    rhs: jnp.ndarray
    xblock_use_active_dof: bool
    active_idx: jnp.ndarray | None
    full_to_active: jnp.ndarray | None
    reduce_full_with_indices: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    expand_reduced_with_map: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
    operator_matvec: ArrayFn
    base_preconditioner: ArrayFn
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None


@dataclass(frozen=True)
class FortranReducedXBlockKrylovSetupResult:
    """Policy values and closures for fortran-reduced x-block Krylov solves."""

    side_env: str
    precondition_side: str
    pc_form: str
    krylov_method: str
    progress_every: int
    mv_count: MatvecCounter
    matvec_no_count: ArrayFn
    matvec: ArrayFn
    preconditioner: ArrayFn


@dataclass(frozen=True)
class FortranReducedXBlockKrylovSolveContext:
    """Solve-local Krylov dispatch dependencies for fortran-reduced x-block solves."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    method: str
    pc_form: str
    restart: int
    maxiter: int
    tol: float
    atol: float
    target: float
    precondition_side: str
    progress_every: int
    mv_count: MatvecCounter
    explicit_left_solver: Callable[..., tuple[np.ndarray, float, float, Sequence[float]]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    lgmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    gcrotmk_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    bicgstab_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]


@dataclass(frozen=True)
class FortranReducedXBlockFactorBuildContext:
    """Dependencies for the fortran-reduced x-block factor build stage."""

    op_pc: object
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    preconditioner_species: int
    preconditioner_xi: int
    sparse_pc_linear_size: int
    backend_reason: str
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None
    assembled_host_allowed: Callable[..., bool]
    builder: Callable[..., ArrayFn]


@dataclass(frozen=True)
class FortranReducedXBlockFactorBuildResult:
    """Result from building the local fortran-reduced x-block preconditioner."""

    preconditioner: ArrayFn
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    preconditioner_xi: int
    force_assembled_host_fp: bool
    factor_s: float


@dataclass(frozen=True)
class FortranReducedXBlockMomentSchurStageContext:
    """Dependencies for the optional fortran-reduced moment-Schur stage."""

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
class FortranReducedXBlockMomentSchurStageResult:
    """Result from the optional fortran-reduced moment-Schur stage."""

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
class FortranReducedXBlockGlobalCouplingStageContext:
    """Dependencies for the optional fortran-reduced global-coupling stage."""

    op: object
    rhs: jnp.ndarray
    matvec: ArrayFn
    base_preconditioner: ArrayFn
    direction_projector: ArrayFn | None
    expected_size: int
    policy: XBlockGlobalCouplingPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]


@dataclass(frozen=True)
class FortranReducedXBlockGlobalCouplingStageResult:
    """Result from the optional fortran-reduced global-coupling stage."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float


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
class SparsePCFactorDtypeRetryDecision:
    """Decision for retrying a sparse-PC factor with higher precision."""

    retry: bool
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None


@dataclass(frozen=True)
class SparsePCFactorDtypeRetryContext:
    """Callbacks and state for retrying a sparse-PC factor in higher precision."""

    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    residual_norm: float
    preconditioned_residual_norm: float
    history: Sequence[float]
    target: float
    x: np.ndarray
    x0_fallback: jnp.ndarray
    solve_s: float
    pc_maxiter: int
    operator_bundle: Any
    factor_bundle: Any
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    build_factor: Callable[[np.dtype], tuple[Any, Any]]
    run_gmres_once: Callable[[jnp.ndarray, int], tuple[np.ndarray, float, float, Sequence[float], float]]


@dataclass(frozen=True)
class SparsePCFactorDtypeRetryResult:
    """Sparse-PC factor dtype retry result and updated solve state."""

    retried: bool
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    factor_s_increment: float
    setup_s: float | None
    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: tuple[float, ...]
    solve_s: float


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
class DirectTailMaterializationContext:
    """Inputs for optional Fortran-reduced direct-tail matrix materialization."""

    env: Mapping[str, str] | None
    op: object
    op_pc: object
    pattern: object
    active_indices: np.ndarray | None
    sparse_pc_use_active_dof: bool
    reduce_full: ArrayFn
    expand_reduced: ArrayFn
    pc_shift: float
    dtype: object
    factor_dtype: object
    sparse_pc_linear_size: int
    default_pattern_color_batch: int
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    is_direct_reduced_pmat_pc_kind: Callable[[str], bool]
    build_direct_tail_bundle: Callable[..., object]
    build_structured_rhs1_full_csr_operator_bundle_callback: Callable[..., object]


@dataclass(frozen=True)
class DirectTailMaterializationResult:
    """Result from optional direct-tail operator materialization."""

    direct_tail_default: bool
    enabled: bool
    built: bool
    error: str | None
    operator_bundle: object | None
    pc_env: str
    direct_reduced_pmat_requested: bool


@dataclass(frozen=True)
class DirectTailStructuredAdmissionContext:
    """Inputs for direct-tail structured preconditioner admission policy."""

    env: Mapping[str, str] | None
    pc_env: str
    operator_bundle: object | None
    direct_reduced_pmat_requested: bool
    sparse_pc_linear_size: int
    default_max_mb: Callable[..., float]


@dataclass(frozen=True)
class DirectTailStructuredAdmissionResult:
    """Resolved structured direct-tail preconditioner admission controls."""

    pc_env: str
    requested: str | None
    auto_default: bool
    fail_closed_size: int
    auto_large_fail_closed: bool
    required: bool
    setup_allowed: bool
    max_mb_auto: bool
    max_mb: float
    regularization: float


@dataclass(frozen=True)
class DirectTailStructuredBuildContext:
    """Inputs for one direct-tail structured preconditioner construction attempt."""

    env: Mapping[str, str] | None
    op: Any
    operator_bundle: Any | None
    active_indices: np.ndarray | None
    requested_kind: str | None
    direct_reduced_pmat_requested: bool
    sparse_pc_linear_size: int
    max_mb: float
    regularization: float
    preconditioner_x: int
    preconditioner_xi: int
    preconditioner_species: int
    preconditioner_x_min_l: int
    layout_from_operator: Callable[[Any], Any]
    build_direct_active_preconditioner: Callable[..., Any]
    build_active_projected_preconditioner: Callable[..., Any]
    cache: MutableMapping[tuple[object, ...], Any]
    cache_key: Callable[..., tuple[object, ...]]
    with_cache_metadata: Callable[..., Any]
    factor_bundle: Callable[..., Any]


@dataclass(frozen=True)
class DirectTailStructuredBuildResult:
    """Result of direct-tail structured preconditioner construction."""

    layout: Any | None
    active_indices: np.ndarray | None
    max_nbytes: int | None
    preconditioner: Any | None
    factor_bundle: Any | None
    operator_bundle_pc: Any | None
    ready: bool
    selected: bool
    reason: str | None
    metadata: dict[str, object] | None
    error: str | None
    cache_hit: bool
    cache_key: tuple[object, ...] | None


@dataclass(frozen=True)
class DirectTailSupportModePreflightContext:
    """Inputs for optional direct-tail support-mode preflight."""

    env: Mapping[str, str] | None
    factor_kind: str
    structured_pc_ready: bool
    operator_bundle: Any | None
    layout: Any | None
    active_indices: np.ndarray | None
    max_nbytes: int | None
    regularization: float
    rhs: np.ndarray
    true_matvec: Callable[[np.ndarray], np.ndarray]
    preconditioner_x: int
    preconditioner_xi: int
    preconditioner_species: int
    preconditioner_x_min_l: int
    selector: Callable[..., tuple[Any, dict[str, object]]]
    factor_bundle: Callable[..., Any]


@dataclass(frozen=True)
class DirectTailSupportModePreflightResult:
    """Result of optional direct-tail support-mode preflight."""

    requested: bool
    applicable: bool
    selected: bool
    preconditioner: Any | None
    factor_bundle: Any | None
    metadata: dict[str, object] | None
    error: str | None
    factor_kind: str


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


@dataclass(frozen=True)
class DirectTailResidualRescuePolicy:
    """Resolved direct-tail residual rescue controls."""

    residual_coarse_requested: bool
    residual_coarse_rank: int
    residual_coarse_max_mb: float
    residual_coarse_regularization: float
    residual_window_requested: bool
    residual_window_max_windows: int
    residual_window_x_radius: int
    residual_window_ell_radius: int
    residual_window_max_mb: float
    residual_window_regularization: float
    residual_window_coefficient_mode: str
    residual_window_combine_mode: str
    residual_window_interface_depth: int
    residual_window_max_size: int
    true_window_requested: bool
    true_window_max_windows: int
    true_window_x_radius: int
    true_window_ell_radius: int
    true_window_max_mb: float
    true_window_regularization: float
    true_window_max_size: int
    true_window_column_batch: int
    true_window_drop_tol: float
    true_window_include_tail: bool
    true_window_damping: bool
    true_window_beta_max: float
    true_coupled_coarse_explicit_requested: bool
    true_coupled_coarse_auto_enabled: bool
    true_coupled_coarse_auto_native_enabled: bool
    true_coupled_coarse_auto_target_ratio: float
    true_coupled_coarse_auto_min_size: int


@dataclass(frozen=True)
class DirectTailTrueActiveRescuePolicy:
    """Resolved direct-tail true-operator active rescue controls."""

    active_block_requested: bool
    active_residual_block_requested: bool
    active_submatrix_requested: bool
    active_column_cache_requested: bool
    active_column_cache_max_mb: float
    active_block_x_count: int
    active_block_ell_count: int
    active_block_species_count: int | None
    active_block_theta_stride: int
    active_block_zeta_stride: int
    active_block_max_mb: float
    active_block_regularization: float
    active_block_max_size: int
    active_block_column_batch: int
    active_block_drop_tol: float
    active_block_include_tail: bool
    active_block_max_tail: int
    active_block_damping: bool
    active_block_beta_max: float
    active_residual_block_max_mb: float
    active_residual_block_regularization: float
    active_residual_block_max_size: int
    active_residual_block_column_batch: int
    active_residual_block_drop_tol: float
    active_residual_block_include_tail: bool
    active_residual_block_max_tail: int
    active_residual_block_kinetic_only: bool
    active_residual_block_damping: bool
    active_residual_block_beta_max: float
    active_residual_block_min_improvement: float
    active_residual_block_accept_base_improvement: bool
    active_submatrix_damping: bool
    active_submatrix_alpha_clip: float
    active_submatrix_min_improvement: float


@dataclass(frozen=True)
class DirectTailCoupledCoarseRescuePolicy:
    """Resolved direct-tail true-operator coupled-coarse rescue controls."""

    max_windows: int
    x_radius: int
    ell_radius: int
    max_mb: float
    regularization: float
    max_size: int
    column_batch: int
    drop_tol: float
    low_lmax: int
    profile_moment_count: int
    angular_lmax: int
    angular_mode_max: int
    max_tail_units: int
    include_tail: bool
    include_constraint_sources: bool
    include_fsavg: bool
    include_window_residual: bool
    include_profile_moments: bool
    include_angular_residual: bool
    include_angular_basis: bool
    include_preconditioned_loads: bool
    preconditioned_load_max_columns: int
    preconditioned_load_max_nnz: int
    preconditioned_load_drop_tol: float
    damping: bool
    beta_max: float
    accept_base_improvement: bool


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
class XBlockQIDeviceOperatorReuseSetup:
    """QI-device operator-reuse admission and x-block factor-build routing."""

    decision: object
    skip_xblock_factors: bool
    xblock_jax_factors: bool
    xblock_device_krylov_forced_jax_factors: bool
    factor_backend: str
    factor_reason: str
    messages: tuple[tuple[int, str], ...]


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
class XBlockQISeedPolicySetup:
    """Shared QI coarse-basis admission and seed/preconditioner settings."""

    coarse_seed_enabled: bool
    galerkin_preconditioner_enabled: bool
    two_level_preconditioner_enabled: bool
    device_preconditioner_enabled: bool
    deflated_preconditioner_enabled: bool
    shared_basis_required: bool
    max_rank: int
    max_candidates: int
    max_angular_mode: int
    rank_rtol: float
    min_improvement: float
    rcond: float
    include_angular: bool
    include_blocks: bool
    include_radial: bool
    include_radial_angular: bool
    include_constraint_moments: bool
    include_schur: bool
    basis_kind: str | None


@dataclass(frozen=True)
class XBlockInitialGuessSetup:
    """Accepted initial guess for an x-block Krylov solve."""

    x0_full: jnp.ndarray | None
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockSeedPolicySetup:
    """Initial preconditioner seed controls for x-block Krylov solves."""

    initial_seed_enabled: bool
    moment_schur_seed_enabled: bool


@dataclass(frozen=True)
class XBlockQIGalerkinPolicySetup:
    """Admission and build controls for the QI Galerkin x-block preconditioner."""

    enabled: bool
    should_build: bool
    reason: str | None
    mode_raw: str
    candidate_modes: tuple[str, ...]
    preconditioner_mode: str | None
    rcond: float
    damping: float
    candidate_dampings: tuple[float, ...]
    probe_enabled: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockQITwoLevelPolicySetup:
    """Admission and build controls for the QI two-level x-block preconditioner."""

    enabled: bool
    should_build: bool
    reason: str | None
    rcond: float
    damping: float
    candidate_dampings: tuple[float, ...]
    min_improvement: float
    coarse_solver: str | None
    residual_augment: bool
    residual_augment_max_extra: int
    residual_augment_steps: int
    residual_augment_include_residuals: bool
    smoothed_load_basis: bool
    smoothed_load_basis_combine: bool
    smoothed_load_max_directions: int
    smoothed_load_max_rank: int
    smoothed_load_fsavg_lmax: int
    smoothed_load_angular_lmax: int
    smoothed_load_max_extra_units: int
    smoothed_load_include_rhs: bool
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockQIDeviceAdmissionSetup:
    """Admission decision for the QI device/matrix-free x-block preconditioner."""

    enabled: bool
    should_build: bool
    reason: str | None
    matrix_free_enabled: bool
    metadata: dict[str, object]
    messages: tuple[tuple[int, str], ...]


@dataclass(frozen=True)
class XBlockQIDeviceBaseConfigSetup:
    """Base QI-device smoother, solve, and Krylov-composition settings."""

    rcond: float
    damping: float
    jacobi_damping: float
    jacobi_sweeps: int
    jacobi_floor: float
    jacobi_require_all_diagonal: bool
    local_smoother_kind: str
    matrix_free_smoother_sweeps: int
    matrix_free_smoother_damping: float
    matrix_free_smoother_step_policy: str
    matrix_free_smoother_alpha_clip: float
    matrix_free_block_smoother_max_groups: int
    matrix_free_block_smoother_include_tail: bool
    matrix_free_block_smoother_rcond: float
    matrix_free_block_smoother_grouping: str
    jacobi_step_policy: str
    coarse_solver: str
    min_improvement: float
    cycles: int
    augmented_seed_requested: bool
    augmented_seed_max_rank: int
    minres_step: bool
    alpha_clip: float
    use_in_krylov_requested: bool
    use_in_krylov: bool
    compose_with_base: bool
    compose_mode: str


@dataclass(frozen=True)
class XBlockQIDeviceEnrichmentConfigSetup:
    """QI-device residual/recycle/operator enrichment settings."""

    residual_enrichment: bool
    residual_enrichment_depth: int
    residual_enrichment_include_residual: bool
    recycle_enrichment: bool
    recycle_cycles: int
    operator_krylov_enrichment: bool
    operator_krylov_depth: int
    adjoint_krylov_enrichment: bool
    adjoint_krylov_depth: int
    adjoint_krylov_transpose_source: str
    operator_action_enrichment: bool
    operator_action_depth: int


@dataclass(frozen=True)
class XBlockQIDeviceMultilevelConfigSetup:
    """QI-device multilevel coarse-space and staged residual-equation controls."""

    multilevel_coarse: bool
    multilevel_max_levels: int
    multilevel_aggregate_factor: int
    multilevel_max_angular_mode: int
    multilevel_max_radial_degree: int
    multilevel_max_pitch_degree: int
    multilevel_current_moments: bool
    multilevel_species_current_moments: bool
    multilevel_radial_current_moments: bool
    multilevel_tail_constraint_moments: bool
    multilevel_current_max_pitch_degree: int
    multilevel_residual_equation: bool
    multilevel_residual_equation_max_level_rank: int
    multilevel_residual_equation_order: str
    multilevel_residual_equation_solver: str
    multilevel_residual_equation_include_global: bool


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    source = env if env is not None else {}
    return str(source.get(key, "")).strip()


def _env_float(env: Mapping[str, str] | None, key: str, default: float) -> float:
    raw = _env_value(env, key)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(env: Mapping[str, str] | None, key: str, default: int, minimum: int | None = None) -> int:
    raw = _env_value(env, key)
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    if minimum is not None:
        value = max(int(minimum), int(value))
    return int(value)


def _env_bool(env: Mapping[str, str] | None, key: str, default: bool = False) -> bool:
    raw = _env_value(env, key).lower()
    if raw in {"1", "true", "t", "yes", "on", ".true.", ".t."}:
        return True
    if raw in {"0", "false", "f", "no", "off", ".false.", ".f."}:
        return False
    return bool(default)


def resolve_fortran_reduced_sparse_pc_backend(
    *,
    op: object,
    env: Mapping[str, str] | None,
    fortran_reduced_sparse_pc: bool,
    sparse_pc_linear_size: int,
) -> FortranReducedSparsePCBackendSetup:
    """Resolve backend routing for fortran-reduced sparse-PC solves."""

    backend_raw = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND")
        .lower()
        .replace("-", "_")
    )
    xblock_min_size = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE",
        100000,
        minimum=1,
    )
    direct_tail_pc_env = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        )
        .lower()
        .replace("-", "_")
    )
    direct_tail_pc_explicit = direct_tail_pc_env not in {
        "",
        "auto",
        "active_auto",
        "structured",
        "factor",
        "host_factor",
        "legacy",
        "default",
    }
    direct_tail_structured_pc_required = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED",
        default=bool(direct_tail_pc_explicit),
    )
    direct_tail_default = bool(
        fortran_reduced_sparse_pc
        and int(sparse_pc_linear_size) >= 100000
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )
    direct_tail_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL",
        default=direct_tail_default,
    )
    direct_tail_structured_pc_forces_global = bool(
        fortran_reduced_sparse_pc
        and direct_tail_enabled
        and direct_tail_structured_pc_required
        and (
            direct_tail_pc_explicit
            or direct_tail_pc_env in {"auto", "active_auto", "structured"}
            or direct_tail_default
        )
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )
    direct_tail_auto_forces_global = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_AUTO_FORCES_GLOBAL",
        default=True,
    )
    direct_tail_auto_structured_pc_forces_global = bool(
        fortran_reduced_sparse_pc
        and direct_tail_enabled
        and bool(direct_tail_auto_forces_global)
        and (
            direct_tail_pc_env in {"", "auto", "active_auto", "structured"}
            or direct_tail_default
        )
        and int(getattr(op, "rhs_mode", 0)) == 1
        and int(getattr(op, "constraint_scheme", 0)) == 1
        and int(getattr(op, "phi1_size", 0)) == 0
    )

    backend_ignored_env = False
    if backend_raw in {"xblock", "x_block", "local_xblock", "block", "blocked"}:
        backend = "xblock"
        reason = "env"
    elif backend_raw in {"global", "monolithic", "csr", "full"}:
        backend = "global"
        reason = "env"
    else:
        backend_ignored_env = bool(backend_raw not in {"", "auto"})
        if direct_tail_structured_pc_forces_global:
            backend = "global"
            reason = "required_direct_tail_structured_pc"
        elif direct_tail_auto_structured_pc_forces_global:
            backend = "global"
            reason = "auto_direct_tail_structured_pc"
        else:
            fblock = getattr(op, "fblock", None)
            auto_xblock_backend = bool(
                fortran_reduced_sparse_pc
                and int(sparse_pc_linear_size) >= int(xblock_min_size)
                and int(getattr(op, "rhs_mode", 0)) == 1
                and (not bool(getattr(op, "include_phi1", False)))
                and getattr(fblock, "fp", None) is not None
                and getattr(fblock, "pas", None) is None
            )
            backend = "xblock" if auto_xblock_backend else "global"
            reason = (
                f"auto_large_full_fp_size>={int(xblock_min_size)}"
                if auto_xblock_backend
                else "auto_global"
            )

    messages: tuple[tuple[int, str], ...] = ()
    if backend_ignored_env:
        messages = (
            (
                1,
                "solve_v3_full_system_linear_gmres: ignoring unknown "
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND="
                f"{backend_raw!r}; using {backend}",
            ),
        )

    return FortranReducedSparsePCBackendSetup(
        backend_raw=backend_raw,
        xblock_min_size=int(xblock_min_size),
        backend_ignored_env=bool(backend_ignored_env),
        direct_tail_pc_env=direct_tail_pc_env,
        direct_tail_pc_explicit=bool(direct_tail_pc_explicit),
        direct_tail_structured_pc_required=bool(
            direct_tail_structured_pc_required
        ),
        direct_tail_default=bool(direct_tail_default),
        direct_tail_enabled=bool(direct_tail_enabled),
        direct_tail_structured_pc_forces_global=bool(
            direct_tail_structured_pc_forces_global
        ),
        direct_tail_auto_forces_global=bool(direct_tail_auto_forces_global),
        direct_tail_auto_structured_pc_forces_global=bool(
            direct_tail_auto_structured_pc_forces_global
        ),
        backend=backend,
        reason=reason,
        messages=messages,
    )


def _env_float_first(
    env: Mapping[str, str] | None,
    names: Sequence[str],
    default: float,
) -> float:
    for name in names:
        raw = _env_value(env, name)
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            return float(default)
    return float(default)


def resolve_fortran_reduced_xblock_factor_policy(
    *,
    env: Mapping[str, str] | None,
    preconditioner_xi: int,
) -> FortranReducedXBlockFactorPolicySetup:
    """Resolve x-block factor tolerances for fortran-reduced sparse-PC solves."""

    drop_tol = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_TOL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL",
        ),
        0.0,
    )
    drop_rel = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_DROP_REL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_REL",
        ),
        1.0e-8,
    )
    ilu_drop_tol = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_ILU_DROP_TOL",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_ILU_DROP_TOL",
        ),
        1.0e-4,
    )
    fill_factor = _env_float_first(
        env,
        (
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PC_FILL_FACTOR",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_FILL_FACTOR",
        ),
        10.0,
    )
    xblock_preconditioner_xi = int(preconditioner_xi)
    promote_xi_raw = _env_value(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_PROMOTE_XI",
    ).lower()
    promote_xi = promote_xi_raw not in {
        "0",
        "false",
        "f",
        "no",
        "off",
        ".false.",
        ".f.",
    }

    messages: tuple[tuple[int, str], ...] = ()
    if xblock_preconditioner_xi == 0 and bool(promote_xi):
        xblock_preconditioner_xi = 1
        messages = (
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
                "promoting x-block backend preconditioner_xi 0 -> 1 for stronger FP block factors",
            ),
        )

    return FortranReducedXBlockFactorPolicySetup(
        drop_tol=float(drop_tol),
        drop_rel=float(drop_rel),
        ilu_drop_tol=float(ilu_drop_tol),
        fill_factor=float(fill_factor),
        preconditioner_xi=int(xblock_preconditioner_xi),
        promote_xi=bool(promote_xi),
        messages=messages,
    )


def resolve_fortran_reduced_xblock_krylov_policy(
    *,
    env: Mapping[str, str] | None,
) -> FortranReducedXBlockKrylovPolicySetup:
    """Resolve fortran-reduced x-block Krylov method and progress controls."""

    side_env = _env_value(env, "SFINCS_JAX_GMRES_PRECONDITION_SIDE").lower()
    precondition_side = side_env if side_env in {"left", "right", "none"} else "left"

    pc_form = _env_value(env, "SFINCS_JAX_RHSMODE1_SPARSE_PC_FORM").lower()
    if pc_form not in {"", "scipy_left", "scipy", "explicit_left", "petsc_left"}:
        pc_form = ""
    pc_form = pc_form or "scipy_left"

    krylov_method = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV")
        or "gmres"
    )
    krylov_method = krylov_method.lower().replace("-", "_")
    messages: list[tuple[int, str]] = []
    if krylov_method in {"lgmres_scipy"}:
        krylov_method = "lgmres"
    elif krylov_method in {"gcrot", "gcrotmk_scipy"}:
        krylov_method = "gcrotmk"
    elif krylov_method in {"bicgstab_scipy", "bi_cgstab"}:
        krylov_method = "bicgstab"
    elif krylov_method not in {"gmres", "lgmres", "gcrotmk", "bicgstab"}:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                "ignoring unknown SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV="
                f"{krylov_method!r}; using gmres",
            )
        )
        krylov_method = "gmres"

    progress_every_env = _env_value(env, "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY")
    try:
        progress_every = int(progress_every_env) if progress_every_env else 25
    except ValueError:
        progress_every = 25
    progress_every = max(0, int(progress_every))

    return FortranReducedXBlockKrylovPolicySetup(
        side_env=side_env,
        precondition_side=precondition_side,
        pc_form=pc_form,
        krylov_method=krylov_method,
        progress_every=int(progress_every),
        mv_count=MatvecCounter(0),
        messages=tuple(messages),
    )


def resolve_fortran_reduced_xblock_initial_seed_policy(
    *,
    env: Mapping[str, str] | None,
) -> FortranReducedXBlockInitialSeedPolicySetup:
    """Resolve initial-seed controls for fortran-reduced x-block solves."""

    seed_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_INITIAL_SEED",
        default=True,
    )
    refine_steps = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_REFINES",
        2,
        minimum=0,
    )
    accept_ratio = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_SEED_ACCEPT_RATIO",
            default=1.0,
        ),
    )
    return FortranReducedXBlockInitialSeedPolicySetup(
        enabled=bool(seed_enabled),
        refine_steps=int(refine_steps),
        accept_ratio=float(accept_ratio),
    )


def apply_fortran_reduced_xblock_initial_seed(
    *,
    policy: FortranReducedXBlockInitialSeedPolicySetup,
    rhs: jnp.ndarray,
    rhs_norm: float,
    x0: jnp.ndarray | None,
    preconditioner: ArrayFn,
    matvec_no_count: ArrayFn,
    elapsed_s: Callable[[], float],
) -> FortranReducedXBlockInitialSeedResult:
    """Apply and refine the fortran-reduced x-block initial preconditioner seed."""

    if (not bool(policy.enabled)) or x0 is not None:
        return FortranReducedXBlockInitialSeedResult(
            x0=x0,
            used=False,
            residual_norm=None,
            improvement_ratio=None,
            refines_performed=0,
            elapsed_s=0.0,
            messages=(),
        )

    seed_start_s = float(elapsed_s())
    x_seed = jnp.asarray(preconditioner(rhs), dtype=jnp.float64)
    residual_vec_seed = rhs - matvec_no_count(x_seed)
    seed_residual_norm = float(jnp.linalg.norm(residual_vec_seed))
    seed_improvement_ratio: float | None = None
    if np.isfinite(seed_residual_norm) and seed_residual_norm > 0.0:
        seed_improvement_ratio = float(rhs_norm) / float(seed_residual_norm)
    elif np.isfinite(seed_residual_norm):
        seed_improvement_ratio = float("inf")

    refines_performed = 0
    for refine_index in range(int(policy.refine_steps)):
        if not np.isfinite(seed_residual_norm) or seed_residual_norm == 0.0:
            break
        dx_seed = jnp.asarray(preconditioner(residual_vec_seed), dtype=jnp.float64)
        x_next = x_seed + dx_seed
        residual_vec_next = rhs - matvec_no_count(x_next)
        residual_norm_next = float(jnp.linalg.norm(residual_vec_next))
        if not np.isfinite(residual_norm_next) or residual_norm_next >= float(seed_residual_norm):
            break
        x_seed = x_next
        residual_vec_seed = residual_vec_next
        seed_residual_norm = float(residual_norm_next)
        refines_performed = int(refine_index) + 1
        if seed_residual_norm > 0.0:
            seed_improvement_ratio = float(rhs_norm) / float(seed_residual_norm)
        else:
            seed_improvement_ratio = float("inf")

    seed_used = bool(
        np.isfinite(seed_residual_norm)
        and seed_residual_norm
        <= (float(rhs_norm) * max(float(policy.accept_ratio), 1.0e-300))
    )
    elapsed = float(elapsed_s()) - seed_start_s
    messages = (
        (
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "initial seed "
            f"residual={float(seed_residual_norm):.6e} "
            f"rhs_norm={float(rhs_norm):.6e} "
            f"improvement={float(seed_improvement_ratio or 0.0):.6e} "
            f"refines={int(refines_performed)}/{int(policy.refine_steps)} "
            f"accepted={bool(seed_used)} elapsed_s={elapsed:.3f}",
        ),
    )
    return FortranReducedXBlockInitialSeedResult(
        x0=x_seed if bool(seed_used) else x0,
        used=bool(seed_used),
        residual_norm=float(seed_residual_norm),
        improvement_ratio=seed_improvement_ratio,
        refines_performed=int(refines_performed),
        elapsed_s=float(elapsed),
        messages=messages,
    )


def build_fortran_reduced_xblock_krylov_setup(
    *,
    context: FortranReducedXBlockKrylovSetupContext,
) -> FortranReducedXBlockKrylovSetupResult:
    """Resolve Krylov controls and closures for fortran-reduced x-block solves."""

    policy = resolve_fortran_reduced_xblock_krylov_policy(env=context.env)
    if context.emit is not None:
        for level, message in policy.messages:
            context.emit(level, message)

    matvec_setup = build_xblock_krylov_matvec_setup(
        op=context.op,
        rhs=context.rhs,
        xblock_use_active_dof=bool(context.xblock_use_active_dof),
        active_idx=context.active_idx,
        full_to_active=context.full_to_active,
        reduce_full_with_indices=context.reduce_full_with_indices,
        expand_reduced_with_map=context.expand_reduced_with_map,
        operator_matvec=context.operator_matvec,
        elapsed_s=context.elapsed_s,
        emit=context.emit,
        env=context.env,
        progress_every=int(policy.progress_every),
        mv_count=policy.mv_count,
        progress_label="fortran_reduced_pc_gmres xblock",
        emit_active_message=False,
    )

    def preconditioner(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(
            context.base_preconditioner(jnp.asarray(v, dtype=context.rhs.dtype)),
            dtype=jnp.float64,
        )

    return FortranReducedXBlockKrylovSetupResult(
        side_env=policy.side_env,
        precondition_side=policy.precondition_side,
        pc_form=policy.pc_form,
        krylov_method=policy.krylov_method,
        progress_every=int(policy.progress_every),
        mv_count=policy.mv_count,
        matvec_no_count=matvec_setup.matvec_no_count,
        matvec=matvec_setup.matvec,
        preconditioner=preconditioner,
    )


def run_fortran_reduced_xblock_krylov_solve(
    *,
    context: FortranReducedXBlockKrylovSolveContext,
    x0: jnp.ndarray | np.ndarray | None,
) -> SparsePCGMRESResult:
    """Run the host Krylov method for a fortran-reduced x-block solve."""

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock solve start "
            f"method={context.method} form={context.pc_form} "
            f"restart={int(context.restart)} maxiter={int(context.maxiter)} "
            f"precondition_side={context.precondition_side}",
        )
    solve_start_s = float(context.elapsed_s())

    def _progress_callback(iteration: int, residual_norm: float) -> None:
        if context.emit is None or int(context.progress_every) <= 0:
            return
        if int(iteration) % int(context.progress_every) != 0:
            return
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            f"iters={int(iteration)} ksp_residual={float(residual_norm):.6e} "
            f"elapsed_s={float(context.elapsed_s()):.3f}",
        )

    rn_pc = float("nan")
    if context.method == "lgmres":
        x_np, residual_norm, history = context.lgmres_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.method == "gcrotmk":
        x_np, residual_norm, history = context.gcrotmk_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.method == "bicgstab":
        x_np, residual_norm, history = context.bicgstab_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner if context.precondition_side != "none" else None,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            maxiter=int(context.maxiter),
            precondition_side=context.precondition_side,
        )
    elif context.pc_form in {"explicit_left", "petsc_left"}:
        x_np, residual_norm, rn_pc, history = context.explicit_left_solver(
            matvec=context.matvec,
            b=context.rhs,
            preconditioner=context.preconditioner,
            x0=x0,
            tol=float(context.tol),
            atol=float(context.atol),
            restart=int(context.restart),
            maxiter=int(context.maxiter),
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
            maxiter=int(context.maxiter),
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

    history_tuple = tuple(float(value) for value in (history or ()))
    if context.emit is not None:
        pc_suffix = (
            f" preconditioned_residual={float(rn_pc):.6e}"
            if np.isfinite(rn_pc)
            else ""
        )
        if history_tuple:
            pc_suffix = f"{pc_suffix} ksp_residual={float(history_tuple[-1]):.6e}"
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock complete "
            f"elapsed_s={float(context.elapsed_s()):.3f} iters={len(history_tuple)} "
            f"matvecs={int(context.mv_count)} residual={float(residual_norm):.6e} "
            f"target={float(context.target):.6e}{pc_suffix}",
        )

    return SparsePCGMRESResult(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(rn_pc),
        history=history_tuple,
        solve_s=float(solve_s),
    )


def build_fortran_reduced_xblock_factor_stage(
    *,
    context: FortranReducedXBlockFactorBuildContext,
) -> FortranReducedXBlockFactorBuildResult:
    """Resolve and build the fortran-reduced x-block local preconditioner."""

    policy = resolve_fortran_reduced_xblock_factor_policy(
        env=context.env,
        preconditioner_xi=int(context.preconditioner_xi),
    )
    if context.emit is not None:
        for level, message in policy.messages:
            context.emit(level, message)
    force_assembled_host_fp = bool(
        context.assembled_host_allowed(
            op=context.op_pc,
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_xi=int(policy.preconditioner_xi),
            use_implicit=False,
            active_size=int(context.sparse_pc_linear_size),
        )
    )
    if context.emit is not None:
        context.emit(
            1,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres "
            "using x-block backend instead of monolithic CSR factor "
            f"(reason={context.backend_reason} "
            f"size={int(context.sparse_pc_linear_size)} "
            f"preconditioner_xi={int(policy.preconditioner_xi)} "
            f"assembled_host_fp={bool(force_assembled_host_fp)})",
        )
    factor_start_s = float(context.elapsed_s())
    preconditioner = context.builder(
        op=context.op_pc,
        reduce_full=context.reduce_full,
        expand_reduced=context.expand_reduced,
        build_jax_factors=False,
        preconditioner_species=int(context.preconditioner_species),
        preconditioner_xi=int(policy.preconditioner_xi),
        drop_tol=float(policy.drop_tol),
        drop_rel=float(policy.drop_rel),
        ilu_drop_tol=float(policy.ilu_drop_tol),
        fill_factor=float(policy.fill_factor),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        emit=context.emit,
    )
    return FortranReducedXBlockFactorBuildResult(
        preconditioner=preconditioner,
        drop_tol=float(policy.drop_tol),
        drop_rel=float(policy.drop_rel),
        ilu_drop_tol=float(policy.ilu_drop_tol),
        fill_factor=float(policy.fill_factor),
        preconditioner_xi=int(policy.preconditioner_xi),
        force_assembled_host_fp=bool(force_assembled_host_fp),
        factor_s=float(context.elapsed_s()) - factor_start_s,
    )


def apply_fortran_reduced_xblock_moment_schur_stage(
    *,
    context: FortranReducedXBlockMomentSchurStageContext,
) -> FortranReducedXBlockMomentSchurStageResult:
    """Build and optionally probe the fortran-reduced moment-Schur stage."""

    if (not bool(context.policy.enabled)) or str(context.precondition_side) == "none":
        return FortranReducedXBlockMomentSchurStageResult(
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
    if context.emit is not None:
        for level, message in context.policy.messages:
            context.emit(level, message)
    try:
        candidate, metadata, stats = context.builder(
            op=context.op,
            base_preconditioner=context.base_preconditioner,
            reduce_full=context.reduce_full,
            expand_reduced=context.expand_reduced,
            rcond=context.policy.rcond,
            emit=context.emit,
        )
        built = True
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
            probe_residual_after = float(jnp.linalg.norm(seed_residual))
            probe_residual_before = float(jnp.linalg.norm(context.rhs))
            if probe_residual_before > 0.0:
                probe_improvement_ratio = probe_residual_after / probe_residual_before
                required = float(probe_residual_before) * max(
                    0.0,
                    1.0 - float(context.policy.probe_min_improvement),
                )
                used = bool(
                    np.isfinite(float(probe_residual_after))
                    and float(probe_residual_after) < float(required)
                )
            else:
                probe_improvement_ratio = (
                    0.0 if probe_residual_after == 0.0 else float("inf")
                )
                used = bool(
                    np.isfinite(float(probe_residual_after))
                    and float(probe_residual_after) <= 0.0
                )
            reason = "probe_reduced" if bool(used) else "probe_not_reduced"
            if context.emit is not None:
                context.emit(
                    0 if bool(used) else 1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                    "constraint1 moment-Schur "
                    f"{'accepted' if bool(used) else 'rejected'} "
                    f"seed residual {float(probe_residual_before):.6e} "
                    f"-> {float(probe_residual_after):.6e} "
                    f"(ratio={float(probe_improvement_ratio):.6e})",
                )
        preconditioner = candidate if bool(used) else context.base_preconditioner
        setup_s = float(context.elapsed_s()) - start_s
        metadata = dict(metadata)
        metadata["setup_s"] = float(setup_s)
        return FortranReducedXBlockMomentSchurStageResult(
            preconditioner=preconditioner,
            built=bool(built),
            used=bool(used),
            reason=reason,
            metadata=metadata,
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
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"constraint1 moment-Schur disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return FortranReducedXBlockMomentSchurStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            used=False,
            reason=reason,
            metadata={"error": reason, "setup_s": float(setup_s)},
            stats={"applies": 0, "base_applies": 0},
            probe_residual_before=None,
            probe_residual_after=None,
            probe_improvement_ratio=None,
            setup_s=float(setup_s),
        )


def apply_fortran_reduced_xblock_global_coupling_stage(
    *,
    context: FortranReducedXBlockGlobalCouplingStageContext,
) -> FortranReducedXBlockGlobalCouplingStageResult:
    """Build the optional fortran-reduced global-coupling stage."""

    if not bool(context.policy.should_build):
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
            "global-coupling build start",
        )
    try:
        preconditioner, metadata, stats = context.builder(
            op=context.op,
            rhs=context.rhs,
            matvec=context.matvec,
            base_preconditioner=context.base_preconditioner,
            direction_projector=context.direction_projector,
            expected_size=int(context.expected_size),
            mode=context.policy.mode,
            fsavg_lmax=context.policy.fsavg_lmax,
            angular_lmax=context.policy.angular_lmax,
            max_extra_units=context.policy.max_extra_units,
            max_directions=context.policy.max_directions,
            rcond=context.policy.rcond,
            include_rhs=context.policy.include_rhs,
            max_setup_s=context.policy.setup_max_s,
            emit=context.emit,
        )
        setup_s = float(context.elapsed_s()) - start_s
        metadata = dict(metadata)
        metadata["setup_s"] = float(setup_s)
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=preconditioner,
            built=True,
            metadata=metadata,
            stats=stats,
            setup_s=float(setup_s),
        )
    except Exception as exc:  # noqa: BLE001
        setup_s = float(context.elapsed_s()) - start_s
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"global-coupling disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return FortranReducedXBlockGlobalCouplingStageResult(
            preconditioner=context.base_preconditioner,
            built=False,
            metadata={"error": error, "setup_s": float(setup_s)},
            stats={"applies": 0, "coarse_applies": 0},
            setup_s=float(setup_s),
        )


def resolve_fortran_reduced_xblock_moment_schur_policy(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None,
) -> XBlockMomentSchurPolicySetup:
    """Resolve moment-Schur controls for fortran-reduced x-block solves."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR",
        default=False,
    )
    generic_rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND",
            default=1.0e-12,
        ),
    )
    rcond = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_RCOND",
            default=generic_rcond,
        ),
    )
    probe_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_PROBE",
        default=False,
    )
    probe_min_improvement = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR_MIN_IMPROVEMENT",
            default=0.0,
        ),
    )

    messages: tuple[tuple[int, str], ...] = ()
    if bool(enabled) and str(precondition_side) != "none":
        messages = (
            (
                0,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                "constraint1 moment-Schur build start",
            ),
        )

    return XBlockMomentSchurPolicySetup(
        default_candidate=False,
        default_blocked_by_compact_factors=False,
        enabled=bool(enabled),
        rcond=float(rcond),
        probe_enabled=bool(probe_enabled),
        probe_min_improvement=float(probe_min_improvement),
        messages=messages,
    )


def resolve_fortran_reduced_xblock_global_coupling_policy(
    *,
    precondition_side: str,
    env: Mapping[str, str] | None,
) -> XBlockGlobalCouplingPolicySetup:
    """Resolve global-coupling controls for fortran-reduced x-block solves."""

    enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING",
        default=False,
    )
    mode = _env_value(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MODE",
    ) or (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE")
        or "additive"
    )
    return XBlockGlobalCouplingPolicySetup(
        enabled=bool(enabled),
        should_build=bool(enabled and str(precondition_side) != "none"),
        use_device_builder=False,
        mode=mode,
        max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_DIRECTIONS",
            default=96,
            minimum=1,
        ),
        fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_FSAVG_LMAX",
            default=12,
            minimum=0,
        ),
        angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_ANGULAR_LMAX",
            default=2,
            minimum=0,
        ),
        max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_RCOND",
                default=1.0e-11,
            ),
        ),
        include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_INCLUDE_RHS",
            default=True,
        ),
        setup_max_s=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_GLOBAL_COUPLING_SETUP_MAX_S",
                default=0.0,
            ),
        ),
    )


def _normalize_qi_device_residual_equation_solver(
    value: str,
    *,
    default: str,
    fallback: str,
    allow_schur_alias: bool = False,
) -> str:
    solver = (str(value).strip() or str(default)).lower().replace("-", "_")
    if solver in {"action", "action_ls", "least_squares", "lstsq", "staged"}:
        return "action_lstsq"
    galerkin_aliases = {"galerkin", "projected", "qtaq", "coarse_grid"}
    if bool(allow_schur_alias):
        galerkin_aliases.add("schur")
    if solver in galerkin_aliases:
        return "galerkin"
    return str(fallback)


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


def evaluate_sparse_pc_factor_dtype_retry(
    *,
    factor_dtype_used: np.dtype,
    residual_norm: float,
    target: float,
) -> SparsePCFactorDtypeRetryDecision:
    """Decide whether an FP32 sparse-PC factor should retry in FP64."""

    dtype_used = np.dtype(factor_dtype_used)
    should_retry = bool(
        dtype_used == np.dtype(np.float32)
        and (
            not np.isfinite(float(residual_norm))
            or float(residual_norm) > float(target)
        )
    )
    if not should_retry:
        return SparsePCFactorDtypeRetryDecision(
            retry=False,
            factor_dtype_used=dtype_used,
            factor_dtype_retry=None,
        )
    return SparsePCFactorDtypeRetryDecision(
        retry=True,
        factor_dtype_used=np.dtype(np.float64),
        factor_dtype_retry="float64",
    )


def sparse_pc_factor_dtype_retry_initial_guess(
    x_candidate: np.ndarray,
    fallback: jnp.ndarray,
) -> jnp.ndarray:
    """Use the first solve as the retry seed only if it is finite."""

    x_np = np.asarray(x_candidate)
    if np.all(np.isfinite(x_np)):
        return jnp.asarray(x_np, dtype=jnp.float64)
    return fallback


def retry_sparse_pc_factor_dtype_if_needed(
    context: SparsePCFactorDtypeRetryContext,
) -> SparsePCFactorDtypeRetryResult:
    """Retry an FP32 sparse-PC factor in FP64 when the probe residual fails."""

    decision = evaluate_sparse_pc_factor_dtype_retry(
        factor_dtype_used=context.factor_dtype_used,
        residual_norm=float(context.residual_norm),
        target=float(context.target),
    )
    if not bool(decision.retry):
        return SparsePCFactorDtypeRetryResult(
            retried=False,
            factor_dtype_used=np.dtype(context.factor_dtype_used),
            factor_dtype_retry=context.factor_dtype_retry,
            operator_bundle=context.operator_bundle,
            factor_bundle=context.factor_bundle,
            factor_s_increment=0.0,
            setup_s=None,
            x=np.asarray(context.x, dtype=np.float64),
            residual_norm=float(context.residual_norm),
            preconditioned_residual_norm=float(context.preconditioned_residual_norm),
            history=tuple(float(v) for v in (context.history or ())),
            solve_s=float(context.solve_s),
        )

    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse_pc_gmres retrying preconditioner "
            f"with factor_dtype={decision.factor_dtype_used.name} "
            f"after residual={float(context.residual_norm):.6e} target={float(context.target):.6e}",
        )
    retry_factor_start_s = float(context.elapsed_s())
    operator_bundle, factor_bundle = context.build_factor(decision.factor_dtype_used)
    factor_s_increment = float(context.elapsed_s()) - retry_factor_start_s
    setup_s = float(context.elapsed_s())
    x0_retry = sparse_pc_factor_dtype_retry_initial_guess(context.x, context.x0_fallback)
    x, residual_norm, rn_pc, history, solve_s_retry = context.run_gmres_once(
        x0_retry,
        int(context.pc_maxiter),
    )
    return SparsePCFactorDtypeRetryResult(
        retried=True,
        factor_dtype_used=np.dtype(decision.factor_dtype_used),
        factor_dtype_retry=decision.factor_dtype_retry,
        operator_bundle=operator_bundle,
        factor_bundle=factor_bundle,
        factor_s_increment=float(factor_s_increment),
        setup_s=float(setup_s),
        x=np.asarray(x, dtype=np.float64),
        residual_norm=float(residual_norm),
        preconditioned_residual_norm=float(rn_pc),
        history=tuple(float(v) for v in (history or ())),
        solve_s=float(context.solve_s) + float(solve_s_retry),
    )


def retry_sparse_pc_factor_dtype_from_driver_state(
    state: Mapping[str, object],
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
) -> SparsePCFactorDtypeRetryResult:
    """Retry sparse-PC factor precision using the historical driver state names."""

    def build_factor(factor_dtype_arg: np.dtype) -> tuple[Any, Any]:
        return build_host_sparse_direct_factor_from_matvec(
            matvec=state["_sparse_pc_factor_mv"],
            n=int(state["sparse_pc_linear_size"]),
            dtype=state["rhs"].dtype,
            factor_dtype=np.dtype(factor_dtype_arg),
            pattern=state["pattern"],
            emit=state["emit"],
            default_diag_pivot_thresh=(
                0.0
                if (
                    bool(state["constrained_pas_pc"])
                    or bool(state["tokamak_fp_pc"])
                    or bool(state["fortran_reduced_sparse_pc"])
                )
                else 1.0
            ),
            default_permc_spec=state["sparse_pc_default_permc_spec"],
            default_factor_kind=state["sparse_pc_default_factor_kind"],
            default_ilu_fill_factor=float(state["sparse_pc_default_ilu_fill_factor"]),
            default_ilu_drop_tol=float(state["sparse_pc_default_ilu_drop_tol"]),
            default_pattern_color_batch=int(state["sparse_pc_default_pattern_color_batch"]),
        )

    return retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(state["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=state["sparse_pc_factor_dtype_retry"],
            residual_norm=float(state["residual_norm_sparse_pc"]),
            preconditioned_residual_norm=float(state["rn_pc"]),
            history=state["history"],
            target=float(state["target"]),
            x=np.asarray(state["x_np"], dtype=np.float64),
            x0_fallback=state["x0_sparse"],
            solve_s=float(state["solve_s"]),
            pc_maxiter=int(state["pc_maxiter"]),
            operator_bundle=state["_operator_bundle_pc"],
            factor_bundle=state["factor_bundle_pc"],
            elapsed_s=state["sparse_timer"].elapsed_s,
            emit=state["emit"],
            build_factor=build_factor,
            run_gmres_once=lambda x0, maxiter: run_sparse_pc_gmres_once_callback(
                x0,
                maxiter_arg=int(maxiter),
            ),
        )
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


def build_direct_tail_materialization_setup(
    context: DirectTailMaterializationContext,
) -> DirectTailMaterializationResult:
    """Optionally materialize the Fortran-reduced direct-tail operator bundle."""

    direct_tail_default = bool(
        int(context.sparse_pc_linear_size) >= 100000
        and int(getattr(context.op, "rhs_mode", 0)) == 1
        and int(getattr(context.op, "constraint_scheme", 0)) == 1
        and int(getattr(context.op, "phi1_size", 0)) == 0
    )
    direct_tail_enabled = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL",
        default=direct_tail_default,
    )
    direct_tail_pc_env = (
        _env_value(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        )
        .lower()
        .replace("-", "_")
    )
    direct_reduced_pmat_requested = bool(
        context.is_direct_reduced_pmat_pc_kind(direct_tail_pc_env)
    )

    if bool(direct_tail_enabled) and bool(direct_reduced_pmat_requested):
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                "materialization skipped; direct reduced-Pmat preconditioner requested "
                f"kind={direct_tail_pc_env}",
            )
        return DirectTailMaterializationResult(
            direct_tail_default=bool(direct_tail_default),
            enabled=bool(direct_tail_enabled),
            built=False,
            error=None,
            operator_bundle=None,
            pc_env=str(direct_tail_pc_env),
            direct_reduced_pmat_requested=True,
        )

    direct_tail_built = False
    direct_tail_error: str | None = None
    direct_tail_operator_bundle: object | None = None

    if bool(direct_tail_enabled):
        direct_tail_start_s = float(context.elapsed_s())
        csr_max_env = _env_value(context.env, "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB")
        drop_tol_env = _env_value(context.env, "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL")
        color_batch_env = _env_value(
            context.env,
            "SFINCS_JAX_EXPLICIT_SPARSE_PATTERN_COLOR_BATCH",
        )
        try:
            csr_max_mb = float(csr_max_env) if csr_max_env else 512.0
        except ValueError:
            csr_max_mb = 512.0
        try:
            drop_tol = float(drop_tol_env) if drop_tol_env else 0.0
        except ValueError:
            drop_tol = 0.0
        try:
            color_batch = (
                int(color_batch_env)
                if color_batch_env
                else int(context.default_pattern_color_batch)
            )
        except ValueError:
            color_batch = int(context.default_pattern_color_batch)

        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                "materialization start "
                f"size={int(context.sparse_pc_linear_size)} "
                f"csr_max_mb={float(csr_max_mb):.3g} "
                f"drop_tol={float(drop_tol):.3e} "
                f"color_batch={int(color_batch)}",
            )
        try:
            direct_tail_operator_bundle = context.build_direct_tail_bundle(
                op=context.op,
                op_pc=context.op_pc,
                pattern=context.pattern,
                active_indices=(
                    context.active_indices
                    if bool(context.sparse_pc_use_active_dof)
                    else None
                ),
                reduce_full=context.reduce_full,
                expand_reduced=context.expand_reduced,
                pc_shift=float(context.pc_shift),
                dtype=context.dtype,
                factor_dtype=context.factor_dtype,
                csr_max_mb=float(csr_max_mb),
                drop_tol=float(drop_tol),
                color_batch=int(color_batch),
                emit=context.emit,
                build_structured_rhs1_full_csr_operator_bundle_callback=(
                    context.build_structured_rhs1_full_csr_operator_bundle_callback
                ),
            )
            direct_tail_built = direct_tail_operator_bundle is not None
            if context.emit is not None and direct_tail_built:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    f"materialization complete elapsed_s={float(context.elapsed_s()) - direct_tail_start_s:.3f}",
                )
            elif context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    f"materialization not selected elapsed_s={float(context.elapsed_s()) - direct_tail_start_s:.3f}",
                )
        except Exception as exc:  # noqa: BLE001
            direct_tail_operator_bundle = None
            direct_tail_error = f"{type(exc).__name__}: {exc}"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: fortran_reduced direct-tail "
                    "materialization disabled after failure "
                    f"elapsed_s={float(context.elapsed_s()) - direct_tail_start_s:.3f} "
                    f"({direct_tail_error})",
                )

    return DirectTailMaterializationResult(
        direct_tail_default=bool(direct_tail_default),
        enabled=bool(direct_tail_enabled),
        built=bool(direct_tail_built),
        error=direct_tail_error,
        operator_bundle=direct_tail_operator_bundle,
        pc_env=str(direct_tail_pc_env),
        direct_reduced_pmat_requested=bool(direct_reduced_pmat_requested),
    )


def resolve_direct_tail_structured_admission(
    context: DirectTailStructuredAdmissionContext,
) -> DirectTailStructuredAdmissionResult:
    """Resolve structured direct-tail preconditioner admission controls."""

    pc_env = str(context.pc_env).strip().lower().replace("-", "_")
    auto_default = bool(
        pc_env == ""
        and context.operator_bundle is not None
        and int(context.sparse_pc_linear_size) >= 100_000
    )
    if pc_env in {"", "factor", "host_factor", "legacy", "default"}:
        requested = "auto" if bool(auto_default) else None
    elif pc_env in {"auto", "active_auto", "structured"}:
        requested = "auto"
    else:
        requested = pc_env

    fail_closed_size = _env_int(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_FAIL_CLOSED_SIZE",
        300_000,
    )
    auto_large_fail_closed = bool(
        requested is not None
        and pc_env in {"", "auto", "active_auto", "structured"}
        and int(context.sparse_pc_linear_size) >= int(fail_closed_size)
    )
    required = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED",
        default=bool(
            auto_large_fail_closed
            or (
                requested is not None
                and pc_env not in {"auto", "active_auto", "structured"}
            )
        ),
    )
    setup_allowed = bool(
        requested is not None
        and (context.operator_bundle is not None or bool(context.direct_reduced_pmat_requested))
    )

    max_mb_auto = False
    max_mb = 0.0
    regularization = 1.0e-12
    if bool(setup_allowed):
        max_mb_env = _env_value(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB",
        )
        if max_mb_env:
            max_mb = _env_float(
                context.env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB",
                512.0,
            )
        else:
            max_mb_auto = True
            max_mb = float(
                context.default_max_mb(
                    requested_kind=requested,
                    active_size=int(context.sparse_pc_linear_size),
                )
            )
        regularization = _env_float(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_REGULARIZATION",
            1.0e-12,
        )

    return DirectTailStructuredAdmissionResult(
        pc_env=str(pc_env),
        requested=requested,
        auto_default=bool(auto_default),
        fail_closed_size=int(fail_closed_size),
        auto_large_fail_closed=bool(auto_large_fail_closed),
        required=bool(required),
        setup_allowed=bool(setup_allowed),
        max_mb_auto=bool(max_mb_auto),
        max_mb=float(max_mb),
        regularization=float(regularization),
    )


def build_direct_tail_structured_preconditioner_setup(
    context: DirectTailStructuredBuildContext,
) -> DirectTailStructuredBuildResult:
    """Build or retrieve the direct-tail structured preconditioner.

    This helper owns only setup and cache plumbing. The driver keeps the
    residual-admission and retry logic that depend on the live true operator.
    """

    layout = None
    active_indices = context.active_indices
    max_nbytes: int | None = None
    preconditioner = None
    factor_bundle = None
    operator_bundle_pc = None
    cache_hit = False
    cache_key: tuple[object, ...] | None = None
    selected = False
    ready = False
    reason: str | None = None
    metadata: dict[str, object] | None = None
    error: str | None = None

    try:
        layout = context.layout_from_operator(context.op)
        max_nbytes = int(max(0.0, float(context.max_mb)) * 1024.0 * 1024.0)
        support_modes = (
            int(context.preconditioner_x),
            int(context.preconditioner_xi),
            int(context.preconditioner_species),
            int(context.preconditioner_x_min_l),
        )
        requested_kind = str(context.requested_kind)
        if bool(context.direct_reduced_pmat_requested):
            preconditioner = context.build_direct_active_preconditioner(
                op=context.op,
                active_indices=active_indices,
                requested_kind=requested_kind,
                regularization=float(context.regularization),
                max_factor_nbytes=int(max_nbytes),
                max_csr_nbytes=int(max_nbytes),
                include_identity_shift=True,
                include_jacobian_terms=True,
                drop_tol=0.0,
                preconditioner_x=int(context.preconditioner_x),
                preconditioner_xi=int(context.preconditioner_xi),
                preconditioner_species=int(context.preconditioner_species),
                preconditioner_x_min_l=int(context.preconditioner_x_min_l),
            )
            cache_key = (
                "direct_reduced_pmat_pc_cache_disabled",
                requested_kind,
                int(context.sparse_pc_linear_size),
                support_modes,
            )
            preconditioner = context.with_cache_metadata(
                preconditioner,
                cache_hit=False,
                cache_key=cache_key,
            )
        else:
            cache_enabled = _env_bool(
                context.env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_CACHE",
                True,
            )
            if context.operator_bundle is None:
                if bool(cache_enabled):
                    raise RuntimeError("direct-tail structured cache requested without a direct-tail matrix")
                raise RuntimeError("direct-tail structured preconditioner requested without a direct-tail matrix")

            if bool(cache_enabled):
                cache_key = context.cache_key(
                    matrix=context.operator_bundle.matrix,
                    layout=layout,
                    active_indices=active_indices,
                    kind=requested_kind,
                    max_factor_nbytes=int(max_nbytes),
                    regularization=float(context.regularization),
                    support_modes=support_modes,
                )
                cached_preconditioner = context.cache.get(cache_key)
                if cached_preconditioner is not None:
                    cache_hit = True
                    preconditioner = context.with_cache_metadata(
                        cached_preconditioner,
                        cache_hit=True,
                        cache_key=cache_key,
                    )
                else:
                    preconditioner = context.build_active_projected_preconditioner(
                        matrix=context.operator_bundle.matrix,
                        layout=layout,
                        active_indices=active_indices,
                        kind=requested_kind,
                        max_factor_nbytes=int(max_nbytes),
                        regularization=float(context.regularization),
                        preconditioner_x=int(context.preconditioner_x),
                        preconditioner_xi=int(context.preconditioner_xi),
                        preconditioner_species=int(context.preconditioner_species),
                        preconditioner_x_min_l=int(context.preconditioner_x_min_l),
                    )
                    preconditioner = context.with_cache_metadata(
                        preconditioner,
                        cache_hit=False,
                        cache_key=cache_key,
                    )
                    context.cache[cache_key] = preconditioner
            else:
                preconditioner = context.build_active_projected_preconditioner(
                    matrix=context.operator_bundle.matrix,
                    layout=layout,
                    active_indices=active_indices,
                    kind=requested_kind,
                    max_factor_nbytes=int(max_nbytes),
                    regularization=float(context.regularization),
                    preconditioner_x=int(context.preconditioner_x),
                    preconditioner_xi=int(context.preconditioner_xi),
                    preconditioner_species=int(context.preconditioner_species),
                    preconditioner_x_min_l=int(context.preconditioner_x_min_l),
                )
                cache_key = (
                    "direct_tail_structured_pc_cache_disabled",
                    requested_kind,
                    support_modes,
                )
                preconditioner = context.with_cache_metadata(
                    preconditioner,
                    cache_hit=False,
                    cache_key=cache_key,
                )

        selected = bool(getattr(preconditioner, "selected", False))
        reason = str(getattr(preconditioner, "reason", None))
        if hasattr(preconditioner, "to_dict"):
            metadata = dict(preconditioner.to_dict())
        else:
            metadata = {
                "selected": bool(selected),
                "kind": str(getattr(preconditioner, "kind", requested_kind)),
                "reason": str(reason),
                "setup_s": float(getattr(preconditioner, "setup_s", 0.0) or 0.0),
                "metadata": dict(getattr(preconditioner, "metadata", None) or {}),
            }
        preconditioner_operator = getattr(preconditioner, "operator", None)
        if bool(selected) and preconditioner_operator is not None:
            factor_nbytes = dict(getattr(preconditioner, "metadata", None) or {}).get("factor_nbytes_actual")
            if factor_nbytes is None:
                factor_nbytes = dict(getattr(preconditioner, "metadata", None) or {}).get(
                    "factor_nbytes_estimate"
                )
            factor_bundle = context.factor_bundle(
                preconditioner=preconditioner,
                operator=context.operator_bundle,
                kind=str(getattr(preconditioner, "kind", requested_kind)),
                factor_nbytes_estimate=None if factor_nbytes is None else int(factor_nbytes),
                factor_nnz_estimate=None,
                factor_s=float(getattr(preconditioner, "setup_s", 0.0) or 0.0),
            )
            operator_bundle_pc = context.operator_bundle
            ready = True
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        selected = False
        ready = False
        reason = "structured_pc_exception"

    return DirectTailStructuredBuildResult(
        layout=layout,
        active_indices=active_indices,
        max_nbytes=max_nbytes,
        preconditioner=preconditioner,
        factor_bundle=factor_bundle,
        operator_bundle_pc=operator_bundle_pc,
        ready=bool(ready),
        selected=bool(selected),
        reason=reason,
        metadata=metadata,
        error=error,
        cache_hit=bool(cache_hit),
        cache_key=cache_key,
    )


def run_direct_tail_support_mode_preflight(
    context: DirectTailSupportModePreflightContext,
) -> DirectTailSupportModePreflightResult:
    """Try the optional support-mode rescue for active direct-tail factors."""

    requested = _env_bool(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_PREFLIGHT",
        False,
    )
    factor_kind = str(context.factor_kind).strip().lower().replace("-", "_")
    if not bool(requested):
        return DirectTailSupportModePreflightResult(
            requested=False,
            applicable=False,
            selected=False,
            preconditioner=None,
            factor_bundle=None,
            metadata=None,
            error=None,
            factor_kind=factor_kind,
        )

    applicable = bool(
        context.structured_pc_ready
        and factor_kind in {"active_fortran_v3_reduced_lu", "active_fortran_v3_reduced_ilu"}
        and context.operator_bundle is not None
        and context.layout is not None
        and context.max_nbytes is not None
    )
    if not bool(applicable):
        return DirectTailSupportModePreflightResult(
            requested=True,
            applicable=False,
            selected=False,
            preconditioner=None,
            factor_bundle=None,
            metadata={
                "selected": False,
                "reason": "support_mode_preflight_not_applicable",
                "structured_pc_ready": bool(context.structured_pc_ready),
                "factor_kind": str(factor_kind),
            },
            error=None,
            factor_kind=factor_kind,
        )

    support_candidates = _env_value(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_CANDIDATES",
    )
    if not support_candidates:
        support_candidates = "current,x0,xmin_l2,species0"
    support_candidates = str(support_candidates).strip()
    support_max_candidates = _env_int(
        context.env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MAX_CANDIDATES",
        4,
        minimum=1,
    )
    support_min_improvement = max(
        1.0,
        _env_float(
            context.env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_SUPPORT_MODE_MIN_IMPROVEMENT",
            1.05,
        ),
    )

    try:
        support_pc, support_metadata = context.selector(
            matrix=context.operator_bundle.matrix,
            layout=context.layout,
            active_indices=context.active_indices,
            requested_kind=factor_kind,
            regularization=float(context.regularization),
            max_factor_nbytes=int(context.max_nbytes),
            rhs=np.asarray(context.rhs, dtype=np.float64),
            true_matvec=context.true_matvec,
            candidates=support_candidates or "current",
            max_candidates=int(support_max_candidates),
            min_improvement_ratio=float(support_min_improvement),
            preconditioner_x=int(context.preconditioner_x),
            preconditioner_xi=int(context.preconditioner_xi),
            preconditioner_species=int(context.preconditioner_species),
            preconditioner_x_min_l=int(context.preconditioner_x_min_l),
        )
        selected = bool(getattr(support_pc, "selected", False)) and getattr(support_pc, "operator", None) is not None
        factor_bundle = None
        if bool(selected):
            support_pc_metadata = dict(getattr(support_pc, "metadata", None) or {})
            support_factor_nbytes = support_pc_metadata.get("factor_nbytes_actual")
            if support_factor_nbytes is None:
                support_factor_nbytes = support_pc_metadata.get("factor_nbytes_estimate")
            factor_bundle = context.factor_bundle(
                preconditioner=support_pc,
                operator=context.operator_bundle,
                kind=str(getattr(support_pc, "kind", factor_kind)),
                factor_nbytes_estimate=None if support_factor_nbytes is None else int(support_factor_nbytes),
                factor_nnz_estimate=None,
                factor_s=float(getattr(support_pc, "setup_s", 0.0) or 0.0),
            )
        return DirectTailSupportModePreflightResult(
            requested=True,
            applicable=True,
            selected=bool(selected),
            preconditioner=support_pc,
            factor_bundle=factor_bundle,
            metadata=support_metadata,
            error=None,
            factor_kind=factor_kind,
        )
    except Exception as exc:  # noqa: BLE001
        return DirectTailSupportModePreflightResult(
            requested=True,
            applicable=True,
            selected=False,
            preconditioner=None,
            factor_bundle=None,
            metadata=None,
            error=f"{type(exc).__name__}: {exc}",
            factor_kind=factor_kind,
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


def resolve_direct_tail_residual_rescue_policy(
    env: Mapping[str, str] | None,
) -> DirectTailResidualRescuePolicy:
    """Resolve direct-tail residual rescue controls without building rescues."""

    residual_window_coefficient_mode = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COEFFICIENTS",
        )
        or "additive"
    ).lower().replace("-", "_")
    if residual_window_coefficient_mode not in {
        "additive",
        "least_squares",
        "normal",
        "normal_equations",
    }:
        residual_window_coefficient_mode = "additive"
    residual_window_combine_mode = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_COMBINE",
        )
        or "independent"
    ).lower().replace("-", "_")
    if residual_window_combine_mode not in {
        "independent",
        "union",
        "coupled",
        "interface",
        "graph_interface",
    }:
        residual_window_combine_mode = "independent"

    return DirectTailResidualRescuePolicy(
        residual_coarse_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE",
            default=False,
        ),
        residual_coarse_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_RANK",
            4,
            minimum=1,
        ),
        residual_coarse_max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_MAX_MB",
                512.0,
            ),
        ),
        residual_coarse_regularization=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_COARSE_REGULARIZATION",
                1.0e-12,
            ),
        ),
        residual_window_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW",
            default=False,
        ),
        residual_window_max_windows=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_MAX_WINDOWS",
            2,
            minimum=1,
        ),
        residual_window_x_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_X_RADIUS",
            0,
            minimum=0,
        ),
        residual_window_ell_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_ELL_RADIUS",
            1,
            minimum=0,
        ),
        residual_window_max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_MAX_MB",
                512.0,
            ),
        ),
        residual_window_regularization=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_REGULARIZATION",
                1.0e-12,
            ),
        ),
        residual_window_coefficient_mode=str(residual_window_coefficient_mode),
        residual_window_combine_mode=str(residual_window_combine_mode),
        residual_window_interface_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_INTERFACE_DEPTH",
            0,
            minimum=0,
        ),
        residual_window_max_size=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_RESIDUAL_WINDOW_MAX_SIZE",
            100_000,
            minimum=1,
        ),
        true_window_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW",
            default=False,
        ),
        true_window_max_windows=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_MAX_WINDOWS",
            1,
            minimum=1,
        ),
        true_window_x_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_X_RADIUS",
            0,
            minimum=0,
        ),
        true_window_ell_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_ELL_RADIUS",
            1,
            minimum=0,
        ),
        true_window_max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_MAX_MB",
                512.0,
            ),
        ),
        true_window_regularization=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_REGULARIZATION",
                1.0e-12,
            ),
        ),
        true_window_max_size=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_MAX_SIZE",
            4096,
            minimum=1,
        ),
        true_window_column_batch=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_COLUMN_BATCH",
            4,
            minimum=1,
        ),
        true_window_drop_tol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DROP_TOL",
                1.0e-14,
            ),
        ),
        true_window_include_tail=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_INCLUDE_TAIL",
            default=True,
        ),
        true_window_damping=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_DAMPING",
            default=False,
        ),
        true_window_beta_max=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_WINDOW_BETA_MAX",
                10.0,
            ),
        ),
        true_coupled_coarse_explicit_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE",
            default=False,
        ),
        true_coupled_coarse_auto_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO",
            default=True,
        ),
        true_coupled_coarse_auto_native_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_NATIVE",
            default=False,
        ),
        true_coupled_coarse_auto_target_ratio=max(
            1.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO",
                10.0,
            ),
        ),
        true_coupled_coarse_auto_min_size=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE",
            300_000,
            minimum=1,
        ),
    )


def resolve_direct_tail_true_active_rescue_policy(
    env: Mapping[str, str] | None,
) -> DirectTailTrueActiveRescuePolicy:
    """Resolve true-operator active-block rescue controls."""

    species_count_raw = _env_value(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_SPECIES_COUNT",
    )
    active_block_species_count: int | None = None
    if species_count_raw:
        try:
            active_block_species_count = max(0, int(species_count_raw))
        except ValueError:
            active_block_species_count = None

    active_block_max_mb = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_MB",
            1024.0,
        ),
    )
    active_block_regularization = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_REGULARIZATION",
            1.0e-12,
        ),
    )
    active_block_max_size = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_SIZE",
        4096,
        minimum=1,
    )
    active_block_column_batch = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_COLUMN_BATCH",
        8,
        minimum=1,
    )
    active_block_drop_tol = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DROP_TOL",
            1.0e-14,
        ),
    )
    active_block_include_tail = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_INCLUDE_TAIL",
        default=True,
    )
    active_block_max_tail = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_MAX_TAIL",
        512,
        minimum=0,
    )
    active_block_damping = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_DAMPING",
        default=False,
    )
    active_block_beta_max = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_BETA_MAX",
            10.0,
        ),
    )

    return DirectTailTrueActiveRescuePolicy(
        active_block_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK",
            default=False,
        ),
        active_residual_block_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK",
            default=False,
        ),
        active_submatrix_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX",
            default=False,
        ),
        active_column_cache_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE",
            default=True,
        ),
        active_column_cache_max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_COLUMN_CACHE_MAX_MB",
                512.0,
            ),
        ),
        active_block_x_count=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_X_COUNT",
            1,
            minimum=0,
        ),
        active_block_ell_count=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ELL_COUNT",
            8,
            minimum=0,
        ),
        active_block_species_count=active_block_species_count,
        active_block_theta_stride=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_THETA_STRIDE",
            1,
            minimum=1,
        ),
        active_block_zeta_stride=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_BLOCK_ZETA_STRIDE",
            1,
            minimum=1,
        ),
        active_block_max_mb=float(active_block_max_mb),
        active_block_regularization=float(active_block_regularization),
        active_block_max_size=int(active_block_max_size),
        active_block_column_batch=int(active_block_column_batch),
        active_block_drop_tol=float(active_block_drop_tol),
        active_block_include_tail=bool(active_block_include_tail),
        active_block_max_tail=int(active_block_max_tail),
        active_block_damping=bool(active_block_damping),
        active_block_beta_max=float(active_block_beta_max),
        active_residual_block_max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_MB",
                float(active_block_max_mb),
            ),
        ),
        active_residual_block_regularization=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_REGULARIZATION",
                float(active_block_regularization),
            ),
        ),
        active_residual_block_max_size=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_SIZE",
            int(active_block_max_size),
            minimum=1,
        ),
        active_residual_block_column_batch=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_COLUMN_BATCH",
            int(active_block_column_batch),
            minimum=1,
        ),
        active_residual_block_drop_tol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DROP_TOL",
                float(active_block_drop_tol),
            ),
        ),
        active_residual_block_include_tail=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_INCLUDE_TAIL",
            default=bool(active_block_include_tail),
        ),
        active_residual_block_max_tail=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MAX_TAIL",
            int(active_block_max_tail),
            minimum=0,
        ),
        active_residual_block_kinetic_only=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_KINETIC_ONLY",
            default=True,
        ),
        active_residual_block_damping=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_DAMPING",
            default=bool(active_block_damping),
        ),
        active_residual_block_beta_max=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_BETA_MAX",
                float(active_block_beta_max),
            ),
        ),
        active_residual_block_min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_MIN_IMPROVEMENT",
                1.0e-6,
            ),
        ),
        active_residual_block_accept_base_improvement=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_RESIDUAL_BLOCK_ACCEPT_BASE_IMPROVEMENT",
            default=False,
        ),
        active_submatrix_damping=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_DAMPING",
            default=True,
        ),
        active_submatrix_alpha_clip=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_ALPHA_CLIP",
                10.0,
            ),
        ),
        active_submatrix_min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_ACTIVE_SUBMATRIX_MIN_IMPROVEMENT",
                1.0e-6,
            ),
        ),
    )


def resolve_direct_tail_coupled_coarse_rescue_policy(
    env: Mapping[str, str] | None,
) -> DirectTailCoupledCoarseRescuePolicy:
    """Resolve true-operator coupled-coarse rescue controls."""

    return DirectTailCoupledCoarseRescuePolicy(
        max_windows=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_WINDOWS",
            2,
            minimum=1,
        ),
        x_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_X_RADIUS",
            0,
            minimum=0,
        ),
        ell_radius=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ELL_RADIUS",
            1,
            minimum=0,
        ),
        max_mb=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_MB",
                512.0,
            ),
        ),
        regularization=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_REGULARIZATION",
                1.0e-12,
            ),
        ),
        max_size=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_SIZE",
            64,
            minimum=1,
        ),
        column_batch=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COLUMN_BATCH",
            4,
            minimum=1,
        ),
        drop_tol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DROP_TOL",
                1.0e-14,
            ),
        ),
        low_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_LOW_LMAX",
            3,
            minimum=0,
        ),
        profile_moment_count=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PROFILE_MOMENT_COUNT",
            4,
            minimum=0,
        ),
        angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_LMAX",
            2,
            minimum=0,
        ),
        angular_mode_max=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ANGULAR_MODE_MAX",
            1,
            minimum=0,
        ),
        max_tail_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_MAX_TAIL_UNITS",
            16,
            minimum=0,
        ),
        include_tail=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_TAIL",
            default=True,
        ),
        include_constraint_sources=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_CONSTRAINT_SOURCES",
            default=True,
        ),
        include_fsavg=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_FSAVG",
            default=True,
        ),
        include_window_residual=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_WINDOW_RESIDUAL",
            default=True,
        ),
        include_profile_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PROFILE_MOMENTS",
            default=True,
        ),
        include_angular_residual=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_RESIDUAL",
            default=True,
        ),
        include_angular_basis=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_ANGULAR_BASIS",
            default=False,
        ),
        include_preconditioned_loads=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_INCLUDE_PRECONDITIONED_LOADS",
            default=False,
        ),
        preconditioned_load_max_columns=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_COLUMNS",
            16,
            minimum=0,
        ),
        preconditioned_load_max_nnz=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_MAX_NNZ",
            50_000,
            minimum=0,
        ),
        preconditioned_load_drop_tol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_PRECONDITIONED_LOAD_DROP_TOL",
                1.0e-12,
            ),
        ),
        damping=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_DAMPING",
            default=False,
        ),
        beta_max=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_BETA_MAX",
                10.0,
            ),
        ),
        accept_base_improvement=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_ACCEPT_BASE_IMPROVEMENT",
            default=False,
        ),
    )


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


def resolve_xblock_qi_device_operator_reuse_setup(
    *,
    op: object,
    xblock_krylov_method: str,
    xblock_device_host_fallback_decision: object,
    qi_device_preconditioner_requested: bool,
    qi_device_matrix_free_requested: bool,
    qi_device_use_in_krylov_requested: bool,
    precondition_side: str,
    xblock_jax_factors: bool,
    xblock_device_krylov_forced_jax_factors: bool,
    xblock_preconditioner_xi: int,
    reuse_decision: Callable[..., object],
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceOperatorReuseSetup:
    """Resolve QI-device reuse admission before local x-block factor setup."""

    decision = reuse_decision(
        env_value=_env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_QI_DEVICE_OPERATOR_REUSE"),
        requested_krylov_method=str(xblock_krylov_method),
        host_fallback_used=bool(getattr(xblock_device_host_fallback_decision, "used", False)),
        rhs_mode=int(op.rhs_mode),
        constraint_scheme=int(op.constraint_scheme),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        has_pas=op.fblock.pas is not None,
        n_zeta=int(getattr(op, "n_zeta", 1)),
        qi_device_preconditioner_requested=bool(qi_device_preconditioner_requested),
        qi_device_matrix_free_requested=bool(qi_device_matrix_free_requested),
        qi_device_use_in_krylov_requested=bool(qi_device_use_in_krylov_requested),
        precondition_side=str(precondition_side),
    )
    skip_factors = bool(getattr(decision, "skip_xblock_factors", False))
    jax_factors = bool(xblock_jax_factors)
    forced_jax_factors = bool(xblock_device_krylov_forced_jax_factors)
    messages: list[tuple[int, str]] = []
    if skip_factors:
        jax_factors = False
        forced_jax_factors = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "using matrix-free QI-device operator reuse; skipping local x-block factors",
            )
        )
    else:
        factor_backend = "jax" if bool(jax_factors) else "host"
        factor_reason = " device-krylov" if bool(forced_jax_factors) else ""
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres building "
                f"{factor_backend} x-block preconditioner preconditioner_xi={int(xblock_preconditioner_xi)}"
                f"{factor_reason}",
            )
        )

    factor_backend = "jax" if bool(jax_factors) else "host"
    factor_reason = " device-krylov" if bool(forced_jax_factors) else ""
    return XBlockQIDeviceOperatorReuseSetup(
        decision=decision,
        skip_xblock_factors=bool(skip_factors),
        xblock_jax_factors=bool(jax_factors),
        xblock_device_krylov_forced_jax_factors=bool(forced_jax_factors),
        factor_backend=str(factor_backend),
        factor_reason=str(factor_reason),
        messages=tuple(messages),
    )


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


def prepare_fortran_reduced_xblock_initial_guess(
    *,
    x0: object | None,
    sparse_pc_rhs: jnp.ndarray,
    full_rhs: jnp.ndarray,
    reduce_full: ArrayFn,
) -> XBlockInitialGuessSetup:
    """Route user-provided x0 into the fortran-reduced x-block solve space."""

    if x0 is None:
        return XBlockInitialGuessSetup(x0_full=None, messages=())
    x0_arr = jnp.asarray(x0, dtype=jnp.float64)
    if x0_arr.shape == sparse_pc_rhs.shape:
        return XBlockInitialGuessSetup(x0_full=x0_arr, messages=())
    if x0_arr.shape == full_rhs.shape:
        return XBlockInitialGuessSetup(
            x0_full=jnp.asarray(reduce_full(x0_arr), dtype=jnp.float64),
            messages=(),
        )
    return XBlockInitialGuessSetup(
        x0_full=None,
        messages=(
            (
                1,
                "solve_v3_full_system_linear_gmres: fortran_reduced_pc_gmres xblock "
                f"ignoring incompatible x0 shape={tuple(x0_arr.shape)} "
                f"expected={tuple(sparse_pc_rhs.shape)} or {tuple(full_rhs.shape)}",
            ),
        ),
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


def resolve_xblock_qi_seed_policy_setup(env: Mapping[str, str] | None = None) -> XBlockQISeedPolicySetup:
    """Resolve QI seed and coarse-basis controls shared by RHSMode=1 x-block policies."""

    coarse_seed_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED",
        default=False,
    )
    galerkin_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER",
        default=False,
    )
    two_level_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER",
        default=False,
    )
    device_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER",
        default=False,
    )
    deflated_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER",
        default=False,
    )
    shared_basis_required = bool(
        coarse_seed_enabled
        or galerkin_enabled
        or two_level_enabled
        or device_enabled
    )
    if not bool(shared_basis_required):
        return XBlockQISeedPolicySetup(
            coarse_seed_enabled=bool(coarse_seed_enabled),
            galerkin_preconditioner_enabled=bool(galerkin_enabled),
            two_level_preconditioner_enabled=bool(two_level_enabled),
            device_preconditioner_enabled=bool(device_enabled),
            deflated_preconditioner_enabled=bool(deflated_enabled),
            shared_basis_required=False,
            max_rank=0,
            max_candidates=0,
            max_angular_mode=0,
            rank_rtol=0.0,
            min_improvement=0.0,
            rcond=0.0,
            include_angular=False,
            include_blocks=False,
            include_radial=False,
            include_radial_angular=False,
            include_constraint_moments=False,
            include_schur=False,
            basis_kind=None,
        )

    basis_kind = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS") or "legacy"
    ).lower().replace("-", "_")
    return XBlockQISeedPolicySetup(
        coarse_seed_enabled=bool(coarse_seed_enabled),
        galerkin_preconditioner_enabled=bool(galerkin_enabled),
        two_level_preconditioner_enabled=bool(two_level_enabled),
        device_preconditioner_enabled=bool(device_enabled),
        deflated_preconditioner_enabled=bool(deflated_enabled),
        shared_basis_required=True,
        max_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK",
            default=24,
            minimum=1,
        ),
        max_candidates=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES",
            default=96,
            minimum=1,
        ),
        max_angular_mode=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_ANGULAR_MODE",
            default=2,
            minimum=0,
        ),
        rank_rtol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RANK_RTOL",
                default=1.0e-10,
            ),
        ),
        min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MIN_IMPROVEMENT",
                default=0.0,
            ),
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RCOND",
                default=1.0e-12,
            ),
        ),
        include_angular=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_ANGULAR",
            default=True,
        ),
        include_blocks=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_BLOCKS",
            default=True,
        ),
        include_radial=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL",
            default=True,
        ),
        include_radial_angular=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL_ANGULAR",
            default=True,
        ),
        include_constraint_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_CONSTRAINT_MOMENTS",
            default=True,
        ),
        include_schur=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_SCHUR",
            default=True,
        ),
        basis_kind=str(basis_kind),
    )


def resolve_xblock_qi_galerkin_policy_setup(
    *,
    enabled: bool,
    host_fallback_used: bool,
    precondition_side: str,
    parse_modes: Callable[..., tuple[str, ...]],
    parse_dampings: Callable[..., tuple[float, ...]],
    env: Mapping[str, str] | None = None,
) -> XBlockQIGalerkinPolicySetup:
    """Resolve QI Galerkin admission and build parameters."""

    messages: list[tuple[int, str]] = []
    if not bool(enabled):
        return XBlockQIGalerkinPolicySetup(
            enabled=False,
            should_build=False,
            reason=None,
            mode_raw="auto",
            candidate_modes=("auto",),
            preconditioner_mode=None,
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            probe_enabled=False,
            messages=(),
        )
    if bool(host_fallback_used):
        reason = "disabled_by_device_host_fallback"
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI Galerkin preconditioner disabled because device-host fallback is active",
            )
        )
        return XBlockQIGalerkinPolicySetup(
            enabled=True,
            should_build=False,
            reason=reason,
            mode_raw="auto",
            candidate_modes=("auto",),
            preconditioner_mode=None,
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            probe_enabled=False,
            messages=tuple(messages),
        )
    if str(precondition_side) == "none":
        return XBlockQIGalerkinPolicySetup(
            enabled=True,
            should_build=False,
            reason="disabled_by_precondition_side_none",
            mode_raw="auto",
            candidate_modes=("auto",),
            preconditioner_mode=None,
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            probe_enabled=False,
            messages=(),
        )

    mode_raw = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_MODE") or "auto"
    ).lower().replace("-", "_")
    damping = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPING",
            default=1.0,
        ),
    )
    return XBlockQIGalerkinPolicySetup(
        enabled=True,
        should_build=True,
        reason=None,
        mode_raw=str(mode_raw),
        candidate_modes=tuple(str(mode) for mode in parse_modes(str(mode_raw), default="auto")),
        preconditioner_mode=str(mode_raw) if str(mode_raw) in {"additive", "multiplicative"} else "auto",
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_RCOND",
                default=1.0e-12,
            ),
        ),
        damping=float(damping),
        candidate_dampings=tuple(
            float(value)
            for value in parse_dampings(
                _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPINGS"),
                default=float(damping),
            )
        ),
        probe_enabled=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_PROBE",
            default=True,
        ),
        messages=(),
    )


def resolve_xblock_qi_two_level_policy_setup(
    *,
    enabled: bool,
    host_fallback_used: bool,
    precondition_side: str,
    seed_max_rank: int,
    parse_dampings: Callable[..., tuple[float, ...]],
    env: Mapping[str, str] | None = None,
) -> XBlockQITwoLevelPolicySetup:
    """Resolve QI two-level admission and build parameters."""

    messages: list[tuple[int, str]] = []
    if not bool(enabled):
        return XBlockQITwoLevelPolicySetup(
            enabled=False,
            should_build=False,
            reason=None,
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            min_improvement=0.0,
            coarse_solver=None,
            residual_augment=False,
            residual_augment_max_extra=0,
            residual_augment_steps=1,
            residual_augment_include_residuals=False,
            smoothed_load_basis=False,
            smoothed_load_basis_combine=True,
            smoothed_load_max_directions=48,
            smoothed_load_max_rank=max(1, int(seed_max_rank)),
            smoothed_load_fsavg_lmax=8,
            smoothed_load_angular_lmax=1,
            smoothed_load_max_extra_units=8,
            smoothed_load_include_rhs=True,
            messages=(),
        )
    if bool(host_fallback_used):
        reason = "disabled_by_device_host_fallback"
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI two-level preconditioner disabled because device-host fallback is active",
            )
        )
        return XBlockQITwoLevelPolicySetup(
            enabled=True,
            should_build=False,
            reason=reason,
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            min_improvement=0.0,
            coarse_solver=None,
            residual_augment=False,
            residual_augment_max_extra=0,
            residual_augment_steps=1,
            residual_augment_include_residuals=False,
            smoothed_load_basis=False,
            smoothed_load_basis_combine=True,
            smoothed_load_max_directions=48,
            smoothed_load_max_rank=max(1, int(seed_max_rank)),
            smoothed_load_fsavg_lmax=8,
            smoothed_load_angular_lmax=1,
            smoothed_load_max_extra_units=8,
            smoothed_load_include_rhs=True,
            messages=tuple(messages),
        )
    if str(precondition_side) == "none":
        return XBlockQITwoLevelPolicySetup(
            enabled=True,
            should_build=False,
            reason="disabled_by_precondition_side_none",
            rcond=0.0,
            damping=1.0,
            candidate_dampings=(1.0,),
            min_improvement=0.0,
            coarse_solver=None,
            residual_augment=False,
            residual_augment_max_extra=0,
            residual_augment_steps=1,
            residual_augment_include_residuals=False,
            smoothed_load_basis=False,
            smoothed_load_basis_combine=True,
            smoothed_load_max_directions=48,
            smoothed_load_max_rank=max(1, int(seed_max_rank)),
            smoothed_load_fsavg_lmax=8,
            smoothed_load_angular_lmax=1,
            smoothed_load_max_extra_units=8,
            smoothed_load_include_rhs=True,
            messages=(),
        )

    damping = max(
        0.0,
        _env_float(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPING",
            default=1.0,
        ),
    )
    coarse_solver = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_COARSE_SOLVER")
        or "action_lstsq"
    ).lower().replace("-", "_")
    return XBlockQITwoLevelPolicySetup(
        enabled=True,
        should_build=True,
        reason=None,
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RCOND",
                default=1.0e-12,
            ),
        ),
        damping=float(damping),
        candidate_dampings=tuple(
            float(value)
            for value in parse_dampings(
                _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS"),
                default=float(damping),
                auto_defaults=(float(damping),),
            )
        ),
        min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_MIN_IMPROVEMENT",
                default=0.05,
            ),
        ),
        coarse_solver=str(coarse_solver),
        residual_augment=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT",
            default=False,
        ),
        residual_augment_max_extra=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_MAX_EXTRA",
            default=3,
            minimum=0,
        ),
        residual_augment_steps=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_STEPS",
            default=1,
            minimum=1,
        ),
        residual_augment_include_residuals=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_INCLUDE_RESIDUALS",
            default=True,
        ),
        smoothed_load_basis=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS",
            default=False,
        ),
        smoothed_load_basis_combine=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS_COMBINE",
            default=True,
        ),
        smoothed_load_max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_DIRECTIONS",
            default=48,
            minimum=1,
        ),
        smoothed_load_max_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_RANK",
            default=max(1, int(seed_max_rank)),
            minimum=1,
        ),
        smoothed_load_fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_FSAVG_LMAX",
            default=8,
            minimum=0,
        ),
        smoothed_load_angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_ANGULAR_LMAX",
            default=1,
            minimum=0,
        ),
        smoothed_load_max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        smoothed_load_include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_INCLUDE_RHS",
            default=True,
        ),
        messages=(),
    )


def resolve_xblock_qi_device_admission_setup(
    *,
    enabled: bool,
    host_fallback_used: bool,
    assembled_device_operator_available: bool,
    assembled_operator_enabled: bool,
    assembled_operator_built: bool,
    assembled_operator_device_resident: bool,
    assembled_operator_device_error: object | None,
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceAdmissionSetup:
    """Resolve whether the QI device preconditioner can build."""

    matrix_free_enabled = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE",
        default=False,
    )
    if not bool(enabled):
        return XBlockQIDeviceAdmissionSetup(
            enabled=False,
            should_build=False,
            reason=None,
            matrix_free_enabled=bool(matrix_free_enabled),
            metadata={},
            messages=(),
        )
    if bool(host_fallback_used):
        reason = "disabled_by_device_host_fallback"
        return XBlockQIDeviceAdmissionSetup(
            enabled=True,
            should_build=False,
            reason=reason,
            matrix_free_enabled=bool(matrix_free_enabled),
            metadata={},
            messages=(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI device preconditioner disabled because device-host fallback is active",
                ),
            ),
        )
    if not bool(assembled_device_operator_available) and not bool(matrix_free_enabled):
        reason = "disabled_missing_assembled_device_operator"
        return XBlockQIDeviceAdmissionSetup(
            enabled=True,
            should_build=False,
            reason=reason,
            matrix_free_enabled=False,
            metadata={
                "reason": reason,
                "requires": (
                    "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR=1 and device CSR success, "
                    "or SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE=1"
                ),
                "assembled_operator_enabled": bool(assembled_operator_enabled),
                "assembled_operator_built": bool(assembled_operator_built),
                "assembled_operator_device_resident": bool(assembled_operator_device_resident),
                "assembled_operator_device_error": assembled_operator_device_error,
            },
            messages=(
                (
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI device preconditioner disabled because no assembled device CSR operator is available",
                ),
            ),
        )
    return XBlockQIDeviceAdmissionSetup(
        enabled=True,
        should_build=True,
        reason=None,
        matrix_free_enabled=bool(matrix_free_enabled),
        metadata={},
        messages=(),
    )


def resolve_xblock_qi_device_base_config_setup(
    *,
    matrix_free_enabled: bool,
    assembled_device_operator_available: bool,
    precondition_side: str,
    probe_uses_minres_step: Callable[[], bool],
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceBaseConfigSetup:
    """Resolve base QI-device preconditioner settings before enrichment setup."""

    local_smoother_kind_default = "none" if not bool(assembled_device_operator_available) else "auto"
    local_smoother_kind = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER")
        or local_smoother_kind_default
    ).lower().replace("-", "_")
    compose_mode = (
        _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_MODE")
        or "multiplicative"
    ).lower().replace("-", "_")
    if compose_mode not in {"additive", "multiplicative"}:
        compose_mode = "multiplicative"

    cycles = _env_int(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES",
        default=1,
        minimum=1,
    )
    use_in_krylov = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV",
        default=bool(assembled_device_operator_available),
    )
    use_in_krylov_requested = bool(use_in_krylov)
    if str(precondition_side) == "none":
        use_in_krylov = False

    return XBlockQIDeviceBaseConfigSetup(
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RCOND",
                default=1.0e-12,
            ),
        ),
        damping=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_DAMPING",
                default=1.0,
            ),
        ),
        jacobi_damping=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DAMPING",
                default=0.7,
            ),
        ),
        jacobi_sweeps=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_SWEEPS",
            default=1,
            minimum=1,
        ),
        jacobi_floor=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DIAGONAL_FLOOR",
                default=1.0e-14,
            ),
        ),
        jacobi_require_all_diagonal=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_REQUIRE_ALL_DIAGONAL",
            default=True,
        ),
        local_smoother_kind=str(local_smoother_kind),
        matrix_free_smoother_sweeps=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS",
            default=1,
            minimum=1,
        ),
        matrix_free_smoother_damping=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_DAMPING",
                default=1.0,
            ),
        ),
        matrix_free_smoother_step_policy=(
            _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_STEP_POLICY")
            or "residual_minimizing"
        ).lower().replace("-", "_"),
        matrix_free_smoother_alpha_clip=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_ALPHA_CLIP",
                default=10.0,
            ),
        ),
        matrix_free_block_smoother_max_groups=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS",
            default=32,
            minimum=1,
        ),
        matrix_free_block_smoother_include_tail=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_INCLUDE_TAIL",
            default=True,
        ),
        matrix_free_block_smoother_rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_RCOND",
                default=1.0e-12,
            ),
        ),
        matrix_free_block_smoother_grouping=(
            _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING")
            or "contiguous"
        ).lower().replace("-", "_"),
        jacobi_step_policy=(
            _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_STEP_POLICY")
            or "stationary"
        ).lower().replace("-", "_"),
        coarse_solver=(
            _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COARSE_SOLVER")
            or "action_lstsq"
        ).lower().replace("-", "_"),
        min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT",
                default=0.05,
            ),
        ),
        cycles=int(cycles),
        augmented_seed_requested=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED",
            default=False,
        ),
        augmented_seed_max_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED_MAX_RANK",
            default=max(1, min(8, int(cycles))),
            minimum=1,
        ),
        minres_step=bool(probe_uses_minres_step()),
        alpha_clip=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ALPHA_CLIP",
                default=10.0,
            ),
        ),
        use_in_krylov_requested=bool(use_in_krylov_requested),
        use_in_krylov=bool(use_in_krylov),
        compose_with_base=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_WITH_BASE",
            default=False,
        ),
        compose_mode=str(compose_mode),
    )


def resolve_xblock_qi_device_enrichment_config_setup(
    *,
    matrix_free_enabled: bool,
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceEnrichmentConfigSetup:
    """Resolve QI-device residual, recycle, and operator-enrichment controls."""

    residual_enrichment = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT",
        default=bool(matrix_free_enabled),
    )
    recycle_enrichment = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT",
        default=False,
    )
    operator_krylov_enrichment = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT",
        default=False,
    )
    adjoint_krylov_enrichment = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT",
        default=False,
    )
    operator_action_enrichment = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT",
        default=False,
    )
    return XBlockQIDeviceEnrichmentConfigSetup(
        residual_enrichment=bool(residual_enrichment),
        residual_enrichment_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_DEPTH",
            default=2 if bool(residual_enrichment) else 0,
            minimum=0,
        ),
        residual_enrichment_include_residual=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_INCLUDE_RESIDUAL",
            default=True,
        ),
        recycle_enrichment=bool(recycle_enrichment),
        recycle_cycles=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_CYCLES",
            default=1 if bool(recycle_enrichment) else 0,
            minimum=0,
        ),
        operator_krylov_enrichment=bool(operator_krylov_enrichment),
        operator_krylov_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH",
            default=4 if bool(operator_krylov_enrichment) else 0,
            minimum=0,
        ),
        adjoint_krylov_enrichment=bool(adjoint_krylov_enrichment),
        adjoint_krylov_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH",
            default=4 if bool(adjoint_krylov_enrichment) else 0,
            minimum=0,
        ),
        adjoint_krylov_transpose_source=(
            _env_value(env, "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE")
            or "autodiff"
        ).lower().replace("-", "_"),
        operator_action_enrichment=bool(operator_action_enrichment),
        operator_action_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH",
            default=1 if bool(operator_action_enrichment) else 0,
            minimum=0,
        ),
    )


def resolve_xblock_qi_device_multilevel_config_setup(
    *,
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeviceMultilevelConfigSetup:
    """Resolve QI-device multilevel coarse-space and residual-equation controls."""

    multilevel_coarse = _env_bool(
        env,
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE",
        default=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR",
            default=False,
        ),
    )
    residual_order = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER",
        )
        or "coarse_to_fine"
    ).lower().replace("-", "_")
    if residual_order not in {"coarse_to_fine", "fine_to_coarse"}:
        residual_order = "coarse_to_fine"

    return XBlockQIDeviceMultilevelConfigSetup(
        multilevel_coarse=bool(multilevel_coarse),
        multilevel_max_levels=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS",
            default=3 if bool(multilevel_coarse) else 1,
            minimum=1,
        ),
        multilevel_aggregate_factor=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_AGGREGATE_FACTOR",
            default=2,
            minimum=2,
        ),
        multilevel_max_angular_mode=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_ANGULAR_MODE",
            default=1,
            minimum=0,
        ),
        multilevel_max_radial_degree=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RADIAL_DEGREE",
            default=2,
            minimum=0,
        ),
        multilevel_max_pitch_degree=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE",
            default=0,
            minimum=0,
        ),
        multilevel_current_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS",
            default=False,
        ),
        multilevel_species_current_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_SPECIES_CURRENT_MOMENTS",
            default=True,
        ),
        multilevel_radial_current_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RADIAL_CURRENT_MOMENTS",
            default=True,
        ),
        multilevel_tail_constraint_moments=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_TAIL_CONSTRAINT_MOMENTS",
            default=True,
        ),
        multilevel_current_max_pitch_degree=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE",
            default=1,
            minimum=0,
        ),
        multilevel_residual_equation=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION",
            default=False,
        ),
        multilevel_residual_equation_max_level_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK",
            default=16,
            minimum=1,
        ),
        multilevel_residual_equation_order=str(residual_order),
        multilevel_residual_equation_solver=_normalize_qi_device_residual_equation_solver(
            _env_value(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER",
            ),
            default="action_lstsq",
            fallback="action_lstsq",
        ),
        multilevel_residual_equation_include_global=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL",
            default=True,
        ),
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


def sparse_pc_gmres_completion_message(
    context: SparsePCGMRESCompletionMessageContext,
) -> str:
    """Format the final sparse-PC GMRES progress message."""

    pc_suffix = (
        f" preconditioned_residual={float(context.preconditioned_residual_norm):.6e}"
        if np.isfinite(float(context.preconditioned_residual_norm))
        else ""
    )
    history = tuple(float(v) for v in (context.history or ()))
    if history:
        pc_suffix = f"{pc_suffix} ksp_residual={float(history[-1]):.6e}"
    return (
        "solve_v3_full_system_linear_gmres: sparse_pc_gmres complete "
        f"elapsed_s={float(context.elapsed_s):.3f} iters={int(context.iterations)} "
        f"matvecs={int(context.matvecs)} residual={float(context.residual_norm):.6e} "
        f"target={float(context.target):.6e}{pc_suffix}"
    )


def emit_sparse_pc_gmres_completion_from_driver_state(
    state: Mapping[str, object],
) -> None:
    """Emit the sparse-PC GMRES completion line from historical driver names."""

    emit = state["emit"]
    if emit is None:
        return
    emit(
        0,
        sparse_pc_gmres_completion_message(
            SparsePCGMRESCompletionMessageContext(
                elapsed_s=float(state["sparse_timer"].elapsed_s()),
                iterations=int(len(state["history"] or ())),
                matvecs=int(state["mv_count"]),
                residual_norm=float(state["residual_norm_sparse_pc"]),
                target=float(state["target"]),
                preconditioned_residual_norm=float(state["rn_pc"]),
                history=state["history"],
            )
        ),
    )


def sparse_pc_gmres_final_payload_from_driver_state(
    state: Mapping[str, object],
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final sparse-PC solve payload from historical driver names."""

    residual_norm = float(state["residual_norm_sparse_pc"])
    metadata_state = state if isinstance(state, MutableMapping) else dict(state)
    metadata_state["sparse_pc_accepted_converged"] = profile_residual_converged(
        residual_norm,
        profile_residual_target(
            atol=float(state["atol"]),
            tol=float(state["tol"]),
            rhs_norm=float(state["rhs_norm"]),
        ),
    )
    metadata_state["sparse_pc_factor_quality_rejected"] = not profile_residual_converged(
        residual_norm,
        float(state["target"]),
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(state["x_np"], dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=sparse_pc_gmres_result_metadata(metadata_state),
    )


def finalize_sparse_pc_gmres_from_driver_state(
    state: Mapping[str, object],
    *,
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Apply final sparse-PC polish, emit completion, and build solve payload.

    This helper keeps the driver from manually copying the post-minres result
    back into its local variables before constructing the final metadata. The
    broad metadata schema is still mapping-backed for compatibility, but the
    mutation is isolated to a copied state map instead of scattered through the
    solve loop.
    """
    post_minres = apply_sparse_pc_post_minres_from_driver_state(
        state,
        minres_correction=minres_correction,
    )
    final_state = state.__class__(state) if isinstance(state, MutableMapping) else dict(state)
    final_state.update(
        {
            "x_np": post_minres.x,
            "residual_norm_sparse_pc": float(post_minres.residual_norm),
            "rn_pc": float(post_minres.preconditioned_residual_norm),
            "sparse_pc_post_minres_history": post_minres.history,
            "sparse_pc_post_minres_alphas": post_minres.alphas,
            "sparse_pc_post_minres_residual_before": post_minres.residual_before,
            "sparse_pc_post_minres_residual_after": post_minres.residual_after,
            "sparse_pc_post_minres_error": post_minres.error,
            "solve_s": float(post_minres.solve_s),
        }
    )
    emit_sparse_pc_gmres_completion_from_driver_state(final_state)
    return sparse_pc_gmres_final_payload_from_driver_state(
        final_state,
        expand_reduced=expand_reduced,
    )


def fortran_reduced_xblock_final_payload_from_driver_state(
    state: Mapping[str, object],
    *,
    result: SparsePCGMRESResult,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the fortran-reduced x-block sparse-PC branch.

    The x-block branch has its own metadata schema, but its final convergence
    gates are the same true-residual gates used by the generic sparse-PC path.
    Keeping that acceptance bookkeeping here avoids duplicating target logic in
    the driver while preserving the historical metadata keys.
    """

    residual_norm = float(result.residual_norm)
    metadata_state = state.__class__(state) if isinstance(state, MutableMapping) else dict(state)
    metadata_state.update(
        {
            "x_np": np.asarray(result.x, dtype=np.float64),
            "residual_norm_sparse_pc": residual_norm,
            "history": tuple(result.history),
            "solve_s": float(result.solve_s),
            "fortran_reduced_xblock_accepted_converged": profile_residual_converged(
                residual_norm,
                profile_residual_target(
                    atol=float(state["atol"]),
                    tol=float(state["tol"]),
                    rhs_norm=float(state["rhs_norm"]),
                ),
            ),
            "fortran_reduced_xblock_factor_quality_rejected": not profile_residual_converged(
                residual_norm,
                float(state["target"]),
            ),
        }
    )
    return SparsePCGMRESFinalPayload(
        x=expand_reduced(jnp.asarray(result.x, dtype=jnp.float64)),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata=fortran_reduced_xblock_result_metadata(metadata_state),
    )


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


def explicit_sparse_pattern_progress_messages(
    *,
    solver_label: str,
    summary: object,
) -> tuple[tuple[int, str], ...]:
    """Return stable progress lines for conservative sparse-pattern setup."""

    return (
        (
            1,
            f"solve_v3_full_system_linear_gmres: {solver_label} building conservative pattern",
        ),
        (
            1,
            f"solve_v3_full_system_linear_gmres: {solver_label} pattern "
            f"nnz={int(summary.nnz)} avg_row_nnz={float(summary.avg_row_nnz):.3g} "
            f"max_row_nnz={int(summary.max_row_nnz)}",
        ),
    )


def resolve_explicit_sparse_operator_build_policy(
    env: Mapping[str, str] | None,
) -> ExplicitSparseOperatorBuildPolicy:
    """Resolve explicit sparse operator materialization controls."""

    return ExplicitSparseOperatorBuildPolicy(
        csr_max_mb=_env_float(env, "SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", 512.0),
        drop_tol=_env_float(env, "SFINCS_JAX_EXPLICIT_SPARSE_DROP_TOL", 0.0),
    )


def build_explicit_sparse_operator_from_pattern(
    *,
    matvec_np: Callable[[np.ndarray], np.ndarray],
    pattern: object,
    dtype: object,
    backend: str,
    env: Mapping[str, str] | None,
    build_operator_from_pattern: Callable[..., object],
    allow_operator_only: bool = False,
) -> ExplicitSparseOperatorBuildResult:
    """Materialize an explicit sparse operator using shared host controls."""

    policy = resolve_explicit_sparse_operator_build_policy(env)
    operator_bundle = build_operator_from_pattern(
        matvec_np,
        pattern=pattern,
        dtype=dtype,
        backend=backend,
        csr_max_mb=float(policy.csr_max_mb),
        drop_tol=float(policy.drop_tol),
        allow_operator_only=bool(allow_operator_only),
    )
    return ExplicitSparseOperatorBuildResult(
        operator_bundle=operator_bundle,
        policy=policy,
        messages=(
            (
                1,
                "explicit_sparse: "
                f"storage={operator_bundle.metadata.storage_kind} "
                f"reason={operator_bundle.metadata.reason}",
            ),
        ),
    )


def validate_explicit_sparse_host_request(
    *,
    solve_method_label: str,
    differentiable: bool | None,
    rhs_mode: int,
    use_active_dof: bool,
    path_description: str,
) -> None:
    """Validate that an explicit host sparse solve is on the non-autodiff lane."""

    if differentiable is True:
        raise ValueError(
            f"solve_method='{solve_method_label}' is a non-differentiable {path_description}."
        )
    if int(rhs_mode) != 1:
        raise NotImplementedError(
            f"solve_method='{solve_method_label}' is currently implemented for RHSMode=1 only."
        )
    if bool(use_active_dof):
        raise NotImplementedError(
            f"solve_method='{solve_method_label}' currently targets the full system; "
            "set SFINCS_JAX_ACTIVE_DOF=0 or use the default matrix-free solver for "
            "active-DOF runs."
        )


def resolve_sparse_minimum_norm_policy(
    env: Mapping[str, str],
    *,
    solve_method_kind: str,
    tol: float,
    maxiter: int | None,
    emit_enabled: bool,
) -> SparseMinimumNormPolicy:
    """Parse host sparse minimum-norm controls from environment values."""

    maxiter_default = max(1000, int(maxiter or 400))
    kind = str(solve_method_kind)
    return SparseMinimumNormPolicy(
        solver_name="lsqr" if kind in {"sparse_lsqr", "sparse_host_lsqr"} else "lsmr",
        atol=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_ATOL", float(tol)),
        btol=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_BTOL", float(tol)),
        conlim=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_CONLIM", 1.0e8),
        damp=_env_float(env, "SFINCS_JAX_SPARSE_LSMR_DAMP", 0.0),
        maxiter=max(1, _env_int(env, "SFINCS_JAX_SPARSE_LSMR_MAXITER", maxiter_default)),
        show=bool(emit_enabled and _env_bool(env, "SFINCS_JAX_SPARSE_LSMR_SHOW")),
        petsc_compat_requested=kind in SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS,
    )


def sparse_minimum_norm_start_message(policy: SparseMinimumNormPolicy) -> str:
    """Return the stable progress line emitted before the LSQR/LSMR solve."""

    return (
        "solve_v3_full_system_linear_gmres: sparse_lsmr solve start "
        f"solver={policy.solver_name} atol={policy.atol:.1e} btol={policy.btol:.1e} "
        f"damp={policy.damp:.1e} conlim={policy.conlim:.1e} maxiter={int(policy.maxiter)}"
    )


def sparse_minimum_norm_solve_payload(
    *,
    matrix: Any,
    rhs: jnp.ndarray,
    policy: SparseMinimumNormPolicy,
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
) -> SparseMinimumNormPayload:
    """Solve a materialized host sparse system with LSQR/LSMR and gate residuals."""

    import scipy.sparse.linalg as _spla  # noqa: PLC0415

    rhs_np = np.asarray(rhs, dtype=np.float64).reshape((-1,))
    if policy.solver_name == "lsqr":
        ls_result = _spla.lsqr(
            matrix,
            rhs_np,
            damp=float(policy.damp),
            atol=float(policy.atol),
            btol=float(policy.btol),
            conlim=float(policy.conlim),
            iter_lim=int(policy.maxiter),
            show=bool(policy.show),
        )
    else:
        ls_result = _spla.lsmr(
            matrix,
            rhs_np,
            damp=float(policy.damp),
            atol=float(policy.atol),
            btol=float(policy.btol),
            conlim=float(policy.conlim),
            maxiter=int(policy.maxiter),
            show=bool(policy.show),
        )

    x_np = np.asarray(ls_result[0], dtype=np.float64)
    istop = int(ls_result[1])
    iters = int(ls_result[2])
    solver_reported_residual = float(ls_result[3])
    residual_true = rhs_np - np.asarray(matrix @ x_np, dtype=np.float64)
    residual_norm = float(np.linalg.norm(residual_true))
    target = profile_residual_target(
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
    )
    true_residual_converged = profile_residual_converged(residual_norm, target)
    compatibility_converged = bool(istop in {1, 2})
    accepted_converged = bool(
        true_residual_converged
        or (policy.petsc_compat_requested and compatibility_converged)
    )
    acceptance_criterion = (
        "true_residual"
        if true_residual_converged
        else "petsc_compatible_minimum_norm"
        if policy.petsc_compat_requested and compatibility_converged
        else "not_converged"
    )
    completion_message = (
        "solve_v3_full_system_linear_gmres: sparse_lsmr complete "
        f"elapsed_s={float(elapsed_s()):.3f} iters={iters} istop={istop} "
        f"reported_residual={solver_reported_residual:.6e} "
        f"residual={residual_norm:.6e} target={float(target):.6e} "
        f"accepted={accepted_converged} criterion={acceptance_criterion}"
    )
    return SparseMinimumNormPayload(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata={
            "solver_kind": "sparse_lsmr",
            "residual_kind": "least_squares_true_residual",
            "reported_residual_norm": float(solver_reported_residual),
            "iterations": int(iters),
            "info_code": int(istop),
            "least_squares_converged": bool(compatibility_converged),
            "true_residual_converged": bool(true_residual_converged),
            "accepted_converged": bool(accepted_converged),
            "acceptance_criterion": str(acceptance_criterion),
            "petsc_compat_requested": bool(policy.petsc_compat_requested),
        },
        start_message=sparse_minimum_norm_start_message(policy),
        completion_message=completion_message,
    )


def sparse_host_direct_solve_payload(
    *,
    factor_solve: Callable[[Any], Any],
    operator_matrix: Any,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    matvec: Callable[[np.ndarray], jnp.ndarray],
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPayload:
    """Solve with a host sparse direct factor and return stable result metadata."""

    x_np, residual_norm = direct_solve_with_refinement(
        factor_solve=factor_solve,
        operator_matrix=operator_matrix,
        rhs_vec=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
    )
    try:
        residual_true = np.asarray(rhs, dtype=np.float64) - np.asarray(
            jax.device_get(matvec(np.asarray(x_np, dtype=np.float64))),
            dtype=np.float64,
        )
        residual_norm = float(np.linalg.norm(residual_true))
    except Exception:
        residual_norm = float(residual_norm)

    target = profile_residual_target(
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
    )
    accepted_converged = profile_residual_converged(float(residual_norm), target)
    completion_message = (
        "solve_v3_full_system_linear_gmres: sparse_host complete "
        f"elapsed_s={float(elapsed_s()):.3f} residual={float(residual_norm):.6e}"
    )
    return SparseHostDirectPayload(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
        metadata={
            "solver_kind": "sparse_host",
            "residual_kind": "true_residual",
            "accepted_converged": bool(accepted_converged),
            "acceptance_criterion": "true_residual",
        },
        completion_message=completion_message,
    )


def solve_sparse_host_direct_from_available_factor(
    *,
    explicit_sparse_factor: object | None,
    explicit_sparse_operator: object | None,
    ilu: object,
    a_csr_full: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectFactorSolvePayload:
    """Solve with an explicit host factor when present, otherwise with ILU/CSR."""

    if explicit_sparse_factor is not None and explicit_sparse_operator is not None:
        x_np, residual_norm = direct_solve_with_refinement(
            factor_solve=explicit_sparse_factor.solve,
            operator_matrix=explicit_sparse_operator.matrix,
            rhs_vec=rhs,
            factor_dtype=factor_dtype,
            refine_steps=int(refine_steps),
        )
        return SparseHostDirectFactorSolvePayload(
            x=np.asarray(x_np, dtype=np.float64),
            residual_norm=float(residual_norm),
            used_explicit_factor=True,
        )

    x_np, residual_norm = ilu_solve_with_refinement(
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs_vec=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
    )
    return SparseHostDirectFactorSolvePayload(
        x=np.asarray(x_np, dtype=np.float64),
        residual_norm=float(residual_norm),
        used_explicit_factor=False,
    )


def apply_sparse_host_direct_polish_if_needed(
    *,
    x: np.ndarray,
    residual_norm: float,
    factor_dtype: np.dtype,
    target: float,
    matvec: ArrayFn,
    rhs: jnp.ndarray,
    ilu: object,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    emit: EmitFn | None,
    polish_enabled: Callable[..., bool],
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPolishPayload:
    """Optionally polish a float32 host sparse direct solve with GMRES."""

    x_current = np.asarray(x, dtype=np.float64)
    residual_current = float(residual_norm)
    if np.dtype(factor_dtype) != np.dtype(np.float32) or residual_current <= float(target):
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_current, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
            attempted=False,
            accepted=False,
            restart=None,
            maxiter=None,
        )
    if not polish_enabled(env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH"):
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_current, dtype=jnp.float64),
            residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
            attempted=False,
            accepted=False,
            restart=None,
            maxiter=None,
        )

    polish_restart, polish_maxiter = parse_polish_gmres_config(
        restart_env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH_RESTART",
        maxiter_env_name="SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_POLISH_MAXITER",
        default_restart=min(int(restart), 40),
        default_maxiter=min(max(40, int(maxiter or 120)), 120),
    )
    if emit is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: host sparse direct polish "
            f"restart={polish_restart} maxiter={polish_maxiter}",
        )
    x_polish, residual_norm_polish = host_sparse_direct_polish(
        matvec_fn=matvec,
        rhs_vec=rhs,
        x0_np=x_current,
        ilu=ilu,
        factor_dtype=factor_dtype,
        tol=tol,
        atol=atol,
        restart=polish_restart,
        maxiter=polish_maxiter,
        precondition_side=precondition_side,
    )
    if np.isfinite(residual_norm_polish) and float(residual_norm_polish) < residual_current:
        return SparseHostDirectPolishPayload(
            x=jnp.asarray(x_polish, dtype=jnp.float64),
            residual_norm=jnp.asarray(float(residual_norm_polish), dtype=jnp.float64),
            attempted=True,
            accepted=True,
            restart=int(polish_restart),
            maxiter=int(polish_maxiter),
        )
    return SparseHostDirectPolishPayload(
        x=jnp.asarray(x_current, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_current, dtype=jnp.float64),
        attempted=True,
        accepted=False,
        restart=int(polish_restart),
        maxiter=int(polish_maxiter),
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


def apply_sparse_pc_post_minres_if_needed(
    context: SparsePCPostMinresUpdateContext,
) -> SparsePCPostMinresUpdateResult:
    """Apply sparse-PC post-minres only when requested and still above target."""

    if (
        int(context.steps) <= 0
        or not np.isfinite(float(context.residual_norm))
        or float(context.residual_norm) <= float(context.target)
    ):
        return SparsePCPostMinresUpdateResult(
            x=np.asarray(context.x, dtype=np.float64),
            residual_norm=float(context.residual_norm),
            preconditioned_residual_norm=float(context.preconditioned_residual_norm),
            history=(),
            alphas=(),
            residual_before=None,
            residual_after=None,
            error=None,
            solve_s=float(context.solve_s),
        )

    post_minres = apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=context.matvec,
            rhs=context.rhs,
            preconditioner=context.preconditioner,
            emit=context.emit,
            elapsed_s=context.elapsed_s,
            pc_form=context.pc_form,
            steps=int(context.steps),
            alpha_clip=float(context.alpha_clip),
            min_improvement=float(context.min_improvement),
            minres_correction=context.minres_correction,
        ),
        x=np.asarray(context.x, dtype=np.float64),
        residual_norm=float(context.residual_norm),
        preconditioned_residual_norm=float(context.preconditioned_residual_norm),
    )
    return SparsePCPostMinresUpdateResult(
        x=post_minres.x,
        residual_norm=float(post_minres.residual_norm),
        preconditioned_residual_norm=float(post_minres.preconditioned_residual_norm),
        history=post_minres.history,
        alphas=post_minres.alphas,
        residual_before=post_minres.residual_before,
        residual_after=post_minres.residual_after,
        error=post_minres.error,
        solve_s=float(context.solve_s) + float(post_minres.solve_s),
    )


def apply_sparse_pc_post_minres_from_driver_state(
    state: Mapping[str, object],
    *,
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
) -> SparsePCPostMinresUpdateResult:
    """Apply sparse-PC post-minres using the historical driver state names."""

    return apply_sparse_pc_post_minres_if_needed(
        SparsePCPostMinresUpdateContext(
            matvec=state["_mv_true"],
            rhs=state["sparse_pc_rhs"],
            preconditioner=state["_precond_sparse"],
            emit=state["emit"],
            elapsed_s=state["sparse_timer"].elapsed_s,
            pc_form=str(state["pc_form"]),
            steps=int(state["sparse_pc_post_minres_steps"]),
            alpha_clip=float(state["sparse_pc_post_minres_alpha_clip"]),
            min_improvement=float(state["sparse_pc_post_minres_min_improvement"]),
            minres_correction=minres_correction,
            x=np.asarray(state["x_np"], dtype=np.float64),
            residual_norm=float(state["residual_norm_sparse_pc"]),
            preconditioned_residual_norm=float(state["rn_pc"]),
            solve_s=float(state["solve_s"]),
            target=float(state["target"]),
        )
    )


__all__ = [
    "FortranReducedSparsePCBackendSetup",
    "FortranReducedXBlockFactorPolicySetup",
    "FortranReducedXBlockFactorBuildContext",
    "FortranReducedXBlockFactorBuildResult",
    "FortranReducedXBlockInitialSeedPolicySetup",
    "FortranReducedXBlockInitialSeedResult",
    "FortranReducedXBlockGlobalCouplingStageContext",
    "FortranReducedXBlockGlobalCouplingStageResult",
    "FortranReducedXBlockKrylovSolveContext",
    "FortranReducedXBlockKrylovPolicySetup",
    "FortranReducedXBlockKrylovSetupContext",
    "FortranReducedXBlockKrylovSetupResult",
    "FortranReducedXBlockMomentSchurStageContext",
    "FortranReducedXBlockMomentSchurStageResult",
    "DirectTailMaterializationContext",
    "DirectTailMaterializationResult",
    "DirectTailStructuredAdmissionContext",
    "DirectTailStructuredAdmissionResult",
    "DirectTailStructuredBuildContext",
    "DirectTailStructuredBuildResult",
    "DirectTailSupportModePreflightContext",
    "DirectTailSupportModePreflightResult",
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
    "DirectTailResidualRescuePolicy",
    "DirectTailTrueActiveRescuePolicy",
    "DirectTailCoupledCoarseRescuePolicy",
    "MatvecCounter",
    "XBlockAssembledOperatorDiagnosticsContext",
    "XBlockSparsePCCoreDiagnosticsContext",
    "XBlockSideProbeDiagnosticsContext",
    "SparsePCActiveDOFSetup",
    "SparsePCFactorPolicySetup",
    "SparsePCFactorDtypeRetryDecision",
    "SparsePCFactorDtypeRetryContext",
    "SparsePCFactorDtypeRetryResult",
    "SparsePCMemoryBudgetPreflightContext",
    "SparsePCPatternSetupContext",
    "SparsePCPatternSetupResult",
    "SparsePCGMRESContext",
    "SparsePCGMRESResult",
    "SparsePCGMRESFinalPayload",
    "SparseMinimumNormPolicy",
    "SparseMinimumNormPayload",
    "SparseHostDirectPayload",
    "SparseHostDirectFactorSolvePayload",
    "SparseHostDirectPolishPayload",
    "ExplicitSparseOperatorBuildPolicy",
    "ExplicitSparseOperatorBuildResult",
    "SparsePCGMRESCompletionMessageContext",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "SparsePCPostMinresUpdateContext",
    "SparsePCPostMinresUpdateResult",
    "apply_fortran_reduced_xblock_global_coupling_stage",
    "apply_fortran_reduced_xblock_initial_seed",
    "apply_fortran_reduced_xblock_moment_schur_stage",
    "apply_sparse_pc_post_minres",
    "apply_sparse_pc_post_minres_if_needed",
    "apply_sparse_pc_post_minres_from_driver_state",
    "build_fortran_reduced_xblock_factor_stage",
    "build_fortran_reduced_xblock_krylov_setup",
    "build_sparse_pc_active_dof_setup",
    "build_sparse_pc_pattern_setup",
    "build_direct_tail_materialization_setup",
    "build_direct_tail_structured_preconditioner_setup",
    "enforce_sparse_pc_memory_budget",
    "evaluate_sparse_pc_factor_preflight",
    "evaluate_sparse_pc_residual_candidate_acceptance",
    "select_sparse_pc_auto_preflight_retry_candidates",
    "evaluate_sparse_pc_auto_preflight_retry",
    "resolve_sparse_pc_gmres_control_policy",
    "resolve_sparse_pc_factor_preflight_policy",
    "resolve_direct_tail_residual_rescue_policy",
    "resolve_direct_tail_true_active_rescue_policy",
    "resolve_direct_tail_coupled_coarse_rescue_policy",
    "run_direct_tail_support_mode_preflight",
    "resolve_direct_tail_structured_admission",
    "fp_xblock_global_correction_metadata",
    "fp_xblock_highx_residual_correction_metadata",
    "prepare_fortran_reduced_xblock_initial_guess",
    "resolve_fortran_reduced_sparse_pc_backend",
    "resolve_fortran_reduced_xblock_factor_policy",
    "resolve_fortran_reduced_xblock_global_coupling_policy",
    "resolve_fortran_reduced_xblock_initial_seed_policy",
    "resolve_fortran_reduced_xblock_krylov_policy",
    "resolve_fortran_reduced_xblock_moment_schur_policy",
    "resolve_sparse_pc_factor_policy",
    "evaluate_sparse_pc_factor_dtype_retry",
    "sparse_pc_factor_dtype_retry_initial_guess",
    "retry_sparse_pc_factor_dtype_if_needed",
    "retry_sparse_pc_factor_dtype_from_driver_state",
    "run_fortran_reduced_xblock_krylov_solve",
    "run_sparse_pc_gmres_once",
    "sparse_pc_gmres_completion_message",
    "emit_sparse_pc_gmres_completion_from_driver_state",
    "sparse_pc_gmres_final_payload_from_driver_state",
    "finalize_sparse_pc_gmres_from_driver_state",
    "fortran_reduced_xblock_final_payload_from_driver_state",
    "resolve_sparse_minimum_norm_policy",
    "sparse_minimum_norm_solve_payload",
    "sparse_minimum_norm_start_message",
    "sparse_host_direct_solve_payload",
    "solve_sparse_host_direct_from_available_factor",
    "apply_sparse_host_direct_polish_if_needed",
    "build_explicit_sparse_operator_from_pattern",
    "explicit_sparse_pattern_progress_messages",
    "resolve_explicit_sparse_operator_build_policy",
    "validate_explicit_sparse_host_request",
    "sparse_rescue_tail_metadata",
    "sparse_xblock_rescue_metadata",
    "xblock_assembled_operator_diagnostics",
    "xblock_coarse_correction_diagnostics",
    "xblock_device_krylov_diagnostics",
    "xblock_qi_deflated_preconditioner_diagnostics",
    "xblock_qi_device_preconditioner_diagnostics",
    "xblock_qi_seed_preconditioner_diagnostics",
    "xblock_sparse_pc_core_diagnostics",
    "xblock_sparse_pc_result_diagnostics_from_driver_state",
    "xblock_sparse_pc_final_metadata_from_driver_state",
    "xblock_side_probe_diagnostics",
]
