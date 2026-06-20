"""Host sparse-PC Krylov helpers for RHSMode=1 profile-response solves."""

from __future__ import annotations

from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, fields
import os
from time import perf_counter
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .diagnostics import (
    SparsePCDirectTailMetadataContext,
    SparsePCFactorPreflightMetadataContext,
    SparsePCGMRESStaticMetadataContext,
    SparsePCPatternMetadataContext,
    XBlockAssembledOperatorDiagnosticsContext,
    XBlockSparsePCCoreDiagnosticsContext,
    XBlockSideProbeDiagnosticsContext,
    fortran_reduced_xblock_result_metadata,
    fp_xblock_global_correction_metadata,
    fp_xblock_highx_residual_correction_metadata,
    sparse_pc_factor_preflight_result_metadata,
    sparse_pc_factor_preflight_result_metadata_from_context,
    sparse_pc_gmres_static_metadata,
    sparse_pc_gmres_static_metadata_from_context,
    sparse_pc_direct_tail_result_metadata,
    sparse_pc_direct_tail_result_metadata_from_context,
    sparse_pc_gmres_result_metadata,
    sparse_pc_pattern_result_metadata_from_context,
    sparse_pc_pattern_result_metadata,
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
    l2_norm_float as profile_l2_norm_float,
    residual_converged as profile_residual_converged,
    residual_target as profile_residual_target,
    safe_ratio as profile_safe_ratio,
)
from .setup import (
    SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS,
    SPARSE_HOST_PETSC_COMPAT_SOLVE_METHODS,
    SPARSE_HOST_XBLOCK_PC_GMRES_SOLVE_METHODS,
)
from .solver_diagnostics import (
    build_rhs1_xblock_correction_metadata_from_driver_state,
    prepare_cached_qi_correction_basis,
)
from ...memory_model import (
    bicgstab_work_nbytes,
    gmres_basis_nbytes,
    tfqmr_work_nbytes,
)
from ...sparse_triangular import (
    triangular_solve_lower_padded,
    triangular_solve_upper_padded,
)
from ...solver import GMRESSolveResult
from ...rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    rhs1_xblock_qi_block_geometry_metadata,
)
from ...rhs1_qi_device_preconditioner import RHS1QIDevicePreconditionerConfig
from ...rhs1_qi_galerkin_policy import (
    RHS1QIGalerkinProbeCandidate,
    select_rhs1_qi_galerkin_probe_candidate,
)
from .policies import (
    rhs1_parse_accept_ratio,
    rhs1_parse_polish_gmres_config,
    rhs1_polish_enabled,
    rhs1_qi_device_tail_block_required,
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


@dataclass(frozen=True)
class SparsePCGMRESFinalPayload:
    """Driver-independent payload for constructing the final sparse-PC result."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    metadata: dict[str, object]


@dataclass(frozen=True)
class SparsePCPostMinresFinalizationContext:
    """Dependencies for final optional sparse-PC post-MinRes polishing."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    emit: EmitFn | None
    elapsed_s: Callable[[], float]
    pc_form: str
    steps: int
    alpha_clip: float
    min_improvement: float
    target: float


@dataclass(frozen=True)
class SparsePCFactorDtypeRetryFinalizationContext:
    """Dependencies for optional sparse-PC factor dtype retry."""

    factor_matvec: ArrayFn
    linear_size: int
    rhs_dtype: np.dtype
    pattern: object
    emit: EmitFn | None
    constrained_pas_pc: bool
    tokamak_fp_pc: bool
    fortran_reduced_sparse_pc: bool
    default_permc_spec: str
    default_factor_kind: str
    default_ilu_fill_factor: float
    default_ilu_drop_tol: float
    default_pattern_color_batch: int
    x0_fallback: jnp.ndarray
    pc_maxiter: int
    elapsed_s: Callable[[], float]


@dataclass(frozen=True)
class SparsePCGMRESFinalizationContext:
    """Explicit inputs for final sparse-PC GMRES retry, polish, and payload."""

    diagnostic_state: Mapping[str, object]
    result: SparsePCGMRESResult
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    pc_factor_s: float
    setup_s: float | None
    post_minres: SparsePCPostMinresFinalizationContext | None = None
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext | None = None


@dataclass(frozen=True)
class SparsePCGMRESFinalResultContext:
    """Result and setup timing from the first sparse-PC GMRES attempt."""

    x: np.ndarray
    residual_norm: float
    preconditioned_residual_norm: float
    history: Sequence[float] | None
    solve_s: float
    factor_dtype_used: np.dtype
    factor_dtype_retry: str | None
    operator_bundle: Any
    factor_bundle: Any
    pc_factor_s: float
    setup_s: float


@dataclass(frozen=True)
class SparsePCGMRESFinalizationBundleContext:
    """Typed sparse-PC finalization inputs that the driver passes as one bundle."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    direct_tail: "SparsePCDirectTailFinalMetadataContext"
    factor_preflight: SparsePCFactorPreflightMetadataContext
    pattern: SparsePCPatternMetadataContext
    static: SparsePCGMRESStaticMetadataContext
    result: SparsePCGMRESFinalResultContext
    post_minres: SparsePCPostMinresFinalizationContext
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext


def _unique_state_keys(*groups: Sequence[str]) -> tuple[str, ...]:
    """Return state keys in first-seen order without duplicate diagnostics."""

    keys: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for key in group:
            if key in seen:
                continue
            seen.add(key)
            keys.append(key)
    return tuple(keys)


_SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS = (
    "atol",
    "mv_count",
    "rhs_norm",
    "target",
    "tol",
)

_SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS = (
    "fortran_reduced_sparse_pc",
    "fortran_reduced_sparse_pc_backend",
    "fortran_reduced_sparse_pc_backend_reason",
    "fortran_reduced_xblock_min_size",
    "op",
    "pc_maxiter",
    "pc_restart",
    "pc_shift",
    "preconditioner_species",
    "preconditioner_x",
    "preconditioner_x_min_l",
    "preconditioner_xi",
    "sparse_pc_default_factor_kind",
    "sparse_pc_default_ilu_drop_tol",
    "sparse_pc_default_ilu_fill_factor",
    "sparse_pc_default_pattern_color_batch",
    "sparse_pc_default_permc_spec",
    "sparse_pc_factor_dtype_initial",
    "sparse_pc_factorization",
    "sparse_pc_first_attempt_maxiter",
    "sparse_pc_fp_dense_velocity_block",
    "sparse_pc_linear_size",
    "sparse_pc_permc_spec",
    "sparse_pc_preconditioner_operator",
    "sparse_pc_use_active_dof",
)

_SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
)

_SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS = _unique_state_keys(
    _SPARSE_PC_GMRES_FINALIZATION_CORE_STATE_KEYS,
    _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
)


def sparse_pc_gmres_finalization_driver_state_keys() -> tuple[str, ...]:
    """Return finalizer keys copied from driver scope before metadata injection."""

    return _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS


def sparse_pc_gmres_finalization_driver_scope_keys() -> tuple[str, ...]:
    """Return raw driver-scope keys needed to build sparse-PC finalization state."""

    return _SPARSE_PC_GMRES_FINALIZATION_SCOPE_KEYS


@dataclass(frozen=True)
class SparsePCGMRESFinalizationStateContext:
    """Explicit sparse-PC final metadata state inputs."""

    atol: object
    mv_count: object
    rhs_norm: object
    target: object
    tol: object
    sparse_pc_direct_tail_metadata: object
    sparse_pc_factor_preflight_metadata: object
    sparse_pc_pattern_metadata: object
    sparse_pc_static_metadata: object


def sparse_pc_gmres_finalization_state_from_context(
    context: SparsePCGMRESFinalizationStateContext,
) -> dict[str, object]:
    """Build sparse-PC finalization metadata state from typed inputs."""

    return {
        "atol": context.atol,
        "mv_count": context.mv_count,
        "rhs_norm": context.rhs_norm,
        "target": context.target,
        "tol": context.tol,
        "sparse_pc_direct_tail_metadata": context.sparse_pc_direct_tail_metadata,
        "sparse_pc_factor_preflight_metadata": (
            context.sparse_pc_factor_preflight_metadata
        ),
        "sparse_pc_pattern_metadata": context.sparse_pc_pattern_metadata,
        "sparse_pc_static_metadata": context.sparse_pc_static_metadata,
    }


def sparse_pc_gmres_finalization_state_from_driver_scope(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Copy only sparse-PC finalizer state and precompute direct-tail metadata."""

    required_keys = _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS
    if "sparse_pc_static_metadata" not in scope:
        required_keys = _unique_state_keys(
            required_keys,
            _SPARSE_PC_GMRES_FINALIZATION_STATIC_METADATA_SCOPE_KEYS,
        )
    missing = tuple(key for key in required_keys if key not in scope)
    if missing:
        joined = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise KeyError(f"sparse-PC GMRES finalization state missing: {joined}{suffix}")
    state = {key: scope[key] for key in _SPARSE_PC_GMRES_FINALIZATION_STATE_KEYS}
    if "sparse_pc_direct_tail_metadata" in scope:
        direct_tail_metadata = scope["sparse_pc_direct_tail_metadata"]
    else:
        direct_tail_metadata = sparse_pc_direct_tail_result_metadata(scope)
    if "sparse_pc_factor_preflight_metadata" in scope:
        factor_preflight_metadata = scope["sparse_pc_factor_preflight_metadata"]
    else:
        factor_preflight_metadata = sparse_pc_factor_preflight_result_metadata(scope)
    if "sparse_pc_pattern_metadata" in scope:
        pattern_metadata = scope["sparse_pc_pattern_metadata"]
    else:
        pattern_metadata = sparse_pc_pattern_result_metadata(scope)
    if "sparse_pc_static_metadata" in scope:
        static_metadata = scope["sparse_pc_static_metadata"]
    else:
        static_metadata = sparse_pc_gmres_static_metadata(scope)
    return sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=state["atol"],
            mv_count=state["mv_count"],
            rhs_norm=state["rhs_norm"],
            target=state["target"],
            tol=state["tol"],
            sparse_pc_direct_tail_metadata=direct_tail_metadata,
            sparse_pc_factor_preflight_metadata=factor_preflight_metadata,
            sparse_pc_pattern_metadata=pattern_metadata,
            sparse_pc_static_metadata=static_metadata,
        )
    )


def sparse_pc_gmres_finalization_bundle_from_driver_scope(
    scope: Mapping[str, object],
    *,
    result: SparsePCGMRESFinalResultContext,
    post_minres: SparsePCPostMinresFinalizationContext,
    dtype_retry: SparsePCFactorDtypeRetryFinalizationContext,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the typed sparse-PC finalization bundle from driver-local names."""

    return SparsePCGMRESFinalizationBundleContext(
        atol=scope["atol"],
        mv_count=scope["mv_count"],
        rhs_norm=scope["rhs_norm"],
        target=scope["target"],
        tol=scope["tol"],
        direct_tail=SparsePCDirectTailFinalMetadataContext(
            structured_pc_preflight_required=bool(
                scope["structured_pc_preflight_required"]
            ),
            structured_pc_preflight_required_min_size=int(
                scope["structured_pc_preflight_required_min_size"]
            ),
            materialization=scope["direct_tail_materialization"],
            structured_admission=scope["direct_tail_structured_admission"],
            residual_policy=scope["direct_tail_residual_rescue_policy"],
            true_active_policy=scope["direct_tail_true_active_rescue_policy"],
            coupled_coarse_policy=scope["direct_tail_true_coupled_coarse_policy"],
            true_window_specs=tuple(
                tuple(int(value) for value in spec)
                for spec in scope["direct_tail_true_window_specs"]
            ),
            true_active_block_species_count=scope[
                "direct_tail_true_active_block_species_count"
            ],
            structured_max_nbytes=scope["direct_tail_structured_max_nbytes"],
            structured_pc_selected=bool(scope["direct_tail_structured_pc_selected"]),
            structured_pc_reason=scope["direct_tail_structured_pc_reason"],
            structured_pc_error=scope["direct_tail_structured_pc_error"],
            structured_pc_metadata=scope["direct_tail_structured_pc_metadata"],
            support_mode_preflight_requested=bool(
                scope["direct_tail_support_mode_preflight_requested"]
            ),
            support_mode_preflight_selected=bool(
                scope["direct_tail_support_mode_preflight_selected"]
            ),
            support_mode_preflight_error=scope[
                "direct_tail_support_mode_preflight_error"
            ],
            support_mode_preflight_metadata=scope[
                "direct_tail_support_mode_preflight_metadata"
            ],
            residual_coarse_selected=bool(scope["direct_tail_residual_coarse_selected"]),
            residual_coarse_residual_after=scope[
                "direct_tail_residual_coarse_residual_after"
            ],
            residual_coarse_error=scope["direct_tail_residual_coarse_error"],
            residual_coarse_metadata=scope["direct_tail_residual_coarse_metadata"],
            true_coupled_coarse_requested=bool(
                scope["direct_tail_true_coupled_coarse_requested"]
            ),
            true_coupled_coarse_auto_selected=bool(
                scope["direct_tail_true_coupled_coarse_auto_selected"]
            ),
            true_coupled_coarse_selected=bool(
                scope["direct_tail_true_coupled_coarse_selected"]
            ),
            true_coupled_coarse_residual_after=scope[
                "direct_tail_true_coupled_coarse_residual_after"
            ],
            true_coupled_coarse_error=scope["direct_tail_true_coupled_coarse_error"],
            true_coupled_coarse_metadata=scope[
                "direct_tail_true_coupled_coarse_metadata"
            ],
            true_coupled_coarse_base_improvement_override_used=bool(
                scope[
                    "direct_tail_true_coupled_coarse_base_improvement_override_used"
                ]
            ),
            true_active_submatrix_selected=bool(
                scope["direct_tail_true_active_submatrix_selected"]
            ),
            true_active_submatrix_residual_after=scope[
                "direct_tail_true_active_submatrix_residual_after"
            ],
            true_active_submatrix_error=scope[
                "direct_tail_true_active_submatrix_error"
            ],
            true_active_submatrix_metadata=scope[
                "direct_tail_true_active_submatrix_metadata"
            ],
            true_active_column_cache_metadata=scope[
                "direct_tail_true_active_column_cache_metadata"
            ],
            true_active_block_selected=bool(
                scope["direct_tail_true_active_block_selected"]
            ),
            true_active_block_residual_after=scope[
                "direct_tail_true_active_block_residual_after"
            ],
            true_active_block_error=scope["direct_tail_true_active_block_error"],
            true_active_block_metadata=scope["direct_tail_true_active_block_metadata"],
            true_active_residual_block_selected=bool(
                scope["direct_tail_true_active_residual_block_selected"]
            ),
            true_active_residual_block_residual_after=scope[
                "direct_tail_true_active_residual_block_residual_after"
            ],
            true_active_residual_block_error=scope[
                "direct_tail_true_active_residual_block_error"
            ],
            true_active_residual_block_metadata=scope[
                "direct_tail_true_active_residual_block_metadata"
            ],
            true_active_residual_block_base_improvement_override_used=bool(
                scope[
                    "direct_tail_true_active_residual_block_base_improvement_override_used"
                ]
            ),
            true_window_selected=bool(scope["direct_tail_true_window_selected"]),
            true_window_residual_after=scope["direct_tail_true_window_residual_after"],
            true_window_error=scope["direct_tail_true_window_error"],
            true_window_metadata=scope["direct_tail_true_window_metadata"],
            residual_window_selected=bool(scope["direct_tail_residual_window_selected"]),
            residual_window_residual_after=scope[
                "direct_tail_residual_window_residual_after"
            ],
            residual_window_error=scope["direct_tail_residual_window_error"],
            residual_window_metadata=scope["direct_tail_residual_window_metadata"],
        ),
        factor_preflight=SparsePCFactorPreflightMetadataContext(
            enabled=bool(scope["factor_preflight_enabled"]),
            required=bool(scope["factor_preflight_required"]),
            seed_enabled=bool(scope["factor_preflight_seed_enabled"]),
            seed_used=bool(scope["factor_preflight_seed_used"]),
            passed=scope["factor_preflight_passed"],
            error=scope["factor_preflight_error"],
            residual_before=scope["factor_preflight_residual_before"],
            residual_after=scope["factor_preflight_residual_after"],
            improvement_ratio=scope["factor_preflight_improvement_ratio"],
            target_ratio=scope["factor_preflight_target_ratio"],
            max_target_ratio=float(scope["factor_preflight_max_target_ratio"]),
            residual_diagnostics=scope["factor_preflight_residual_diagnostics"],
        ),
        pattern=SparsePCPatternMetadataContext(
            summary=scope["summary"],
            scope=scope["sparse_pattern_scope"],
            build_s=float(scope["pattern_build_s"]),
        ),
        static=SparsePCGMRESStaticMetadataContext(
            op=scope["op"],
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            fortran_reduced_sparse_pc_backend=scope[
                "fortran_reduced_sparse_pc_backend"
            ],
            fortran_reduced_sparse_pc_backend_reason=scope[
                "fortran_reduced_sparse_pc_backend_reason"
            ],
            fortran_reduced_xblock_min_size=scope["fortran_reduced_xblock_min_size"],
            pc_restart=int(scope["pc_restart"]),
            pc_maxiter=int(scope["pc_maxiter"]),
            sparse_pc_first_attempt_maxiter=int(
                scope["sparse_pc_first_attempt_maxiter"]
            ),
            pc_shift=float(scope["pc_shift"]),
            sparse_pc_factor_dtype_initial=scope["sparse_pc_factor_dtype_initial"],
            sparse_pc_preconditioner_operator=scope[
                "sparse_pc_preconditioner_operator"
            ],
            sparse_pc_factorization=scope["sparse_pc_factorization"],
            sparse_pc_default_factor_kind=scope["sparse_pc_default_factor_kind"],
            sparse_pc_default_ilu_fill_factor=float(
                scope["sparse_pc_default_ilu_fill_factor"]
            ),
            sparse_pc_default_ilu_drop_tol=float(
                scope["sparse_pc_default_ilu_drop_tol"]
            ),
            sparse_pc_default_pattern_color_batch=int(
                scope["sparse_pc_default_pattern_color_batch"]
            ),
            preconditioner_x=int(scope["preconditioner_x"]),
            preconditioner_x_min_l=int(scope["preconditioner_x_min_l"]),
            preconditioner_xi=int(scope["preconditioner_xi"]),
            preconditioner_species=int(scope["preconditioner_species"]),
            sparse_pc_permc_spec=scope["sparse_pc_permc_spec"],
            sparse_pc_default_permc_spec=scope["sparse_pc_default_permc_spec"],
            sparse_pc_use_active_dof=bool(scope["sparse_pc_use_active_dof"]),
            sparse_pc_linear_size=int(scope["sparse_pc_linear_size"]),
            sparse_pc_fp_dense_velocity_block=scope[
                "sparse_pc_fp_dense_velocity_block"
            ],
        ),
        result=result,
        post_minres=post_minres,
        dtype_retry=dtype_retry,
    )


def sparse_pc_gmres_finalization_bundle_from_driver_result(
    scope: Mapping[str, object],
    *,
    x: np.ndarray,
    residual_norm: float,
    preconditioned_residual_norm: float,
    history: Sequence[float] | None,
    solve_s: float,
) -> SparsePCGMRESFinalizationBundleContext:
    """Build the full sparse-PC finalization bundle from the first GMRES result."""

    return sparse_pc_gmres_finalization_bundle_from_driver_scope(
        scope,
        result=SparsePCGMRESFinalResultContext(
            x=np.asarray(x, dtype=np.float64),
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(preconditioned_residual_norm),
            history=tuple(float(v) for v in (history or ())),
            solve_s=float(solve_s),
            factor_dtype_used=np.dtype(scope["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=scope["sparse_pc_factor_dtype_retry"],
            operator_bundle=scope["_operator_bundle_pc"],
            factor_bundle=scope["factor_bundle_pc"],
            pc_factor_s=float(scope["pc_factor_s"]),
            setup_s=float(scope["setup_s"]),
        ),
        post_minres=SparsePCPostMinresFinalizationContext(
            matvec=scope["_mv_true"],
            rhs=scope["sparse_pc_rhs"],
            preconditioner=scope["_precond_sparse"],
            emit=scope["emit"],
            elapsed_s=scope["sparse_timer"].elapsed_s,
            pc_form=scope["pc_form"],
            steps=int(scope["sparse_pc_post_minres_steps"]),
            alpha_clip=float(scope["sparse_pc_post_minres_alpha_clip"]),
            min_improvement=float(scope["sparse_pc_post_minres_min_improvement"]),
            target=float(scope["target"]),
        ),
        dtype_retry=SparsePCFactorDtypeRetryFinalizationContext(
            factor_matvec=scope["_sparse_pc_factor_mv"],
            linear_size=int(scope["sparse_pc_linear_size"]),
            rhs_dtype=np.dtype(scope["rhs"].dtype),
            pattern=scope["pattern"],
            emit=scope["emit"],
            constrained_pas_pc=bool(scope["constrained_pas_pc"]),
            tokamak_fp_pc=bool(scope["tokamak_fp_pc"]),
            fortran_reduced_sparse_pc=bool(scope["fortran_reduced_sparse_pc"]),
            default_permc_spec=scope["sparse_pc_default_permc_spec"],
            default_factor_kind=scope["sparse_pc_default_factor_kind"],
            default_ilu_fill_factor=float(scope["sparse_pc_default_ilu_fill_factor"]),
            default_ilu_drop_tol=float(scope["sparse_pc_default_ilu_drop_tol"]),
            default_pattern_color_batch=int(
                scope["sparse_pc_default_pattern_color_batch"]
            ),
            x0_fallback=scope["x0_sparse"],
            pc_maxiter=int(scope["pc_maxiter"]),
            elapsed_s=scope["sparse_timer"].elapsed_s,
        ),
    )


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
class SparseHostDirectFallbackPayload:
    """Complete host sparse direct fallback result with its true residual."""

    x: jnp.ndarray
    residual_norm: jnp.ndarray
    residual_vec: jnp.ndarray
    used_explicit_factor: bool
    polish_attempted: bool
    polish_accepted: bool
    polish_restart: int | None
    polish_maxiter: int | None


@dataclass(frozen=True)
class ExplicitSparseMinimumNormBranchContext:
    """Driver callbacks and controls for the explicit sparse LSQR/LSMR branch."""

    op: Any
    rhs: jnp.ndarray
    solve_method_kind: str
    differentiable: bool | None
    use_active_dof: bool
    tol: float
    atol: float
    maxiter: int | None
    rhs_norm: float
    backend: str
    env: Mapping[str, str]
    emit: EmitFn | None
    build_pattern: Callable[[Any], object]
    summarize_pattern: Callable[[Any, object], object]
    apply_cached_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    build_operator_from_pattern: Callable[..., object]


@dataclass(frozen=True)
class ExplicitSparseHostDirectBranchContext:
    """Driver callbacks and controls for the explicit sparse host-LU branch."""

    op: Any
    rhs: jnp.ndarray
    differentiable: bool | None
    use_active_dof: bool
    tol: float
    atol: float
    rhs_norm: float
    refine_steps: int
    emit: EmitFn | None
    build_pattern: Callable[[Any], object]
    summarize_pattern: Callable[[Any, object], object]
    apply_operator: Callable[[Any, jnp.ndarray], jnp.ndarray]
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]]
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]


@dataclass(frozen=True)
class SparseHostOrILUFactorBuildContext:
    """Inputs for choosing explicit host sparse direct factorization or ILU."""

    matvec: ArrayFn
    n: int
    dtype: object
    cache_key: object
    factor_dtype: np.dtype
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    fill_factor: float
    build_dense_factors: bool
    build_jax_factors: bool
    store_dense: bool
    factorization: str
    emit: EmitFn | None
    host_sparse_direct_wanted: bool
    explicit_sparse_allowed: bool
    explicit_sparse_pattern: object | None = None
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]] | None = None
    build_sparse_ilu_from_matvec: Callable[..., tuple[Any, Any, Any, Any, Any, Any, bool]] | None = None


@dataclass(frozen=True)
class SparseHostOrILUFactorControls:
    """Resolved routing controls for a host sparse direct/ILU factor build."""

    host_sparse_direct_wanted: bool
    factor_dtype: np.dtype
    cache_key_use: object
    build_dense_factors: bool
    build_jax_factors: bool
    store_dense: bool
    explicit_sparse_allowed: bool


@dataclass(frozen=True)
class SparseHostOrILUFactorBuildResult:
    """Factor objects and matrix caches returned by sparse host/ILU setup."""

    explicit_sparse_operator: object | None
    explicit_sparse_factor: object | None
    a_csr_full: object
    a_csr_drop: object
    ilu: object
    a_dense_cache: object | None
    l_dense: object | None
    u_dense: object | None
    l_unit_diag: bool
    used_explicit_sparse: bool


@dataclass(frozen=True)
class SparseILUPreconditionerBuildContext:
    """Cached ILU factors needed to build a JAX-side sparse preconditioner."""

    cache_entry: object | None
    l_dense: object | None
    u_dense: object | None
    l_unit_diag: bool
    require_lower_diag: bool = False


@dataclass(frozen=True)
class SparseILUPreconditionerBuildResult:
    """JAX preconditioner selected from cached dense or padded ILU factors."""

    preconditioner: ArrayFn | None
    used_dense_triangular: bool
    used_padded_triangular: bool


@dataclass(frozen=True)
class SparseHostScipyPreconditionerBuildContext:
    """Host ILU factor and optional explicit matrix used by SciPy Krylov."""

    ilu: object | None
    a_csr_full: object
    base_matvec: ArrayFn
    sparse_use_matvec: bool
    unavailable_message: str = "sparse_ilu: ILU factors unavailable"


@dataclass(frozen=True)
class SparseHostScipyPreconditionerBuildResult:
    """Host preconditioner and matrix-vector product for SciPy Krylov fallback."""

    preconditioner: ArrayFn
    matvec: ArrayFn


@dataclass(frozen=True)
class SparseHostScipyGMRESContext:
    """Inputs for one host SciPy GMRES sparse fallback solve."""

    matvec: ArrayFn
    rhs: jnp.ndarray
    preconditioner: ArrayFn
    x0: jnp.ndarray
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    residual_matvec: ArrayFn | None = None


@dataclass(frozen=True)
class SparseHostRetryCandidateContext:
    """Inputs for choosing one sparse-host retry candidate after factor setup."""

    factor_build: SparseHostOrILUFactorBuildResult
    host_sparse_direct: bool
    host_direct_operator_pc: bool
    use_implicit: bool
    matvec: ArrayFn
    rhs: jnp.ndarray
    x0: jnp.ndarray
    factor_dtype: np.dtype
    refine_steps: int
    target: float
    tol: float
    atol: float
    restart: int
    maxiter: int | None
    precondition_side: str
    emit: EmitFn | None
    backend_name: str
    sparse_use_matvec: bool
    sparse_exact_lu: bool
    cache_entry: object | None
    require_lower_diag: bool
    polish_enabled: Callable[..., bool]
    parse_polish_gmres_config: Callable[..., tuple[int, int]]
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]]
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]]
    gmres_solver: Callable[..., tuple[np.ndarray, float, Sequence[float]]]
    implicit_solver: Callable[[ArrayFn], tuple[GMRESSolveResult | None, jnp.ndarray | None]]
    operator_pc_restart: int | None = None
    operator_pc_maxiter: int | None = None
    compute_scipy_residual_vec: bool = True


@dataclass(frozen=True)
class SparseHostRetryCandidateResult:
    """Sparse retry candidate plus callbacks needed by the replay accept gate."""

    result: GMRESSolveResult | None
    residual_vec: jnp.ndarray | None
    matvec: ArrayFn
    preconditioner: ArrayFn | None
    solve_s: float
    host_sparse_direct_used: bool


@dataclass(frozen=True)
class SparseJAXRetryPreconditionerBuildContext:
    """Inputs for building the sparse-JAX retry preconditioner."""

    matvec: ArrayFn
    n: int
    dtype: object
    cache_key: object
    drop_tol: float
    drop_rel: float
    reg: float
    omega: float
    sweeps: int
    emit: EmitFn | None
    builder: Callable[..., ArrayFn]


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
    solver_label: str = "sparse_pc_gmres"


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
    solver_label: str = "sparse_pc_gmres"


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
class FortranReducedXBlockFinalPayloadContext:
    """Explicit inputs for final fortran-reduced xblock sparse-PC payloads."""

    diagnostic_state: Mapping[str, object]
    result: SparsePCGMRESResult
    atol: float
    tol: float
    rhs_norm: float
    target: float


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
class SparsePCDirectTailFinalMetadataContext:
    """Semantic direct-tail state used by final sparse-PC diagnostics.

    Most direct-tail metadata is policy state, not solver scratch. Keeping that
    mapping here avoids exposing dozens of historical report keys to the driver.
    """

    structured_pc_preflight_required: bool
    structured_pc_preflight_required_min_size: int
    materialization: DirectTailMaterializationResult
    structured_admission: DirectTailStructuredAdmissionResult
    residual_policy: "DirectTailResidualRescuePolicy"
    true_active_policy: "DirectTailTrueActiveRescuePolicy"
    coupled_coarse_policy: "DirectTailCoupledCoarseRescuePolicy"
    true_window_specs: tuple[tuple[int, ...], ...]
    true_active_block_species_count: int | None
    structured_max_nbytes: int | None
    structured_pc_selected: bool
    structured_pc_reason: str | None
    structured_pc_error: str | None
    structured_pc_metadata: dict[str, object] | None
    support_mode_preflight_requested: bool
    support_mode_preflight_selected: bool
    support_mode_preflight_error: str | None
    support_mode_preflight_metadata: dict[str, object] | None
    residual_coarse_selected: bool
    residual_coarse_residual_after: float | None
    residual_coarse_error: str | None
    residual_coarse_metadata: dict[str, object] | None
    true_coupled_coarse_requested: bool
    true_coupled_coarse_auto_selected: bool
    true_coupled_coarse_selected: bool
    true_coupled_coarse_residual_after: float | None
    true_coupled_coarse_error: str | None
    true_coupled_coarse_metadata: dict[str, object] | None
    true_coupled_coarse_base_improvement_override_used: bool
    true_active_submatrix_selected: bool
    true_active_submatrix_residual_after: float | None
    true_active_submatrix_error: str | None
    true_active_submatrix_metadata: dict[str, object] | None
    true_active_column_cache_metadata: dict[str, object] | None
    true_active_block_selected: bool
    true_active_block_residual_after: float | None
    true_active_block_error: str | None
    true_active_block_metadata: dict[str, object] | None
    true_active_residual_block_selected: bool
    true_active_residual_block_residual_after: float | None
    true_active_residual_block_error: str | None
    true_active_residual_block_metadata: dict[str, object] | None
    true_active_residual_block_base_improvement_override_used: bool
    true_window_selected: bool
    true_window_residual_after: float | None
    true_window_error: str | None
    true_window_metadata: dict[str, object] | None
    residual_window_selected: bool
    residual_window_residual_after: float | None
    residual_window_error: str | None
    residual_window_metadata: dict[str, object] | None


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


def _direct_tail_final_suffix_values(
    context: SparsePCDirectTailFinalMetadataContext,
) -> dict[str, object]:
    residual = context.residual_policy
    active = context.true_active_policy
    coupled = context.coupled_coarse_policy
    return {
        "residual_coarse_requested": residual.residual_coarse_requested,
        "residual_coarse_selected": context.residual_coarse_selected,
        "residual_coarse_rank": residual.residual_coarse_rank,
        "residual_coarse_max_mb": residual.residual_coarse_max_mb,
        "residual_coarse_regularization": residual.residual_coarse_regularization,
        "residual_coarse_residual_after": context.residual_coarse_residual_after,
        "residual_coarse_error": context.residual_coarse_error,
        "residual_coarse_metadata": context.residual_coarse_metadata,
        "true_coupled_coarse_requested": context.true_coupled_coarse_requested,
        "true_coupled_coarse_explicit_requested": (
            residual.true_coupled_coarse_explicit_requested
        ),
        "true_coupled_coarse_auto_enabled": residual.true_coupled_coarse_auto_enabled,
        "true_coupled_coarse_auto_native_enabled": (
            residual.true_coupled_coarse_auto_native_enabled
        ),
        "true_coupled_coarse_auto_target_ratio": (
            residual.true_coupled_coarse_auto_target_ratio
        ),
        "true_coupled_coarse_auto_min_size": (
            residual.true_coupled_coarse_auto_min_size
        ),
        "true_coupled_coarse_auto_selected": (
            context.true_coupled_coarse_auto_selected
        ),
        "true_coupled_coarse_selected": context.true_coupled_coarse_selected,
        "true_coupled_coarse_max_windows": coupled.max_windows,
        "true_coupled_coarse_x_radius": coupled.x_radius,
        "true_coupled_coarse_ell_radius": coupled.ell_radius,
        "true_coupled_coarse_max_mb": coupled.max_mb,
        "true_coupled_coarse_regularization": coupled.regularization,
        "true_coupled_coarse_max_size": coupled.max_size,
        "true_coupled_coarse_column_batch": coupled.column_batch,
        "true_coupled_coarse_drop_tol": coupled.drop_tol,
        "true_coupled_coarse_low_lmax": coupled.low_lmax,
        "true_coupled_coarse_profile_moment_count": (
            coupled.profile_moment_count
        ),
        "true_coupled_coarse_angular_lmax": coupled.angular_lmax,
        "true_coupled_coarse_angular_mode_max": coupled.angular_mode_max,
        "true_coupled_coarse_max_tail_units": coupled.max_tail_units,
        "true_coupled_coarse_include_tail": coupled.include_tail,
        "true_coupled_coarse_include_constraint_sources": (
            coupled.include_constraint_sources
        ),
        "true_coupled_coarse_include_fsavg": coupled.include_fsavg,
        "true_coupled_coarse_include_window_residual": (
            coupled.include_window_residual
        ),
        "true_coupled_coarse_include_profile_moments": (
            coupled.include_profile_moments
        ),
        "true_coupled_coarse_include_angular_residual": (
            coupled.include_angular_residual
        ),
        "true_coupled_coarse_include_angular_basis": (
            coupled.include_angular_basis
        ),
        "true_coupled_coarse_include_preconditioned_loads": (
            coupled.include_preconditioned_loads
        ),
        "true_coupled_coarse_preconditioned_load_max_columns": (
            coupled.preconditioned_load_max_columns
        ),
        "true_coupled_coarse_preconditioned_load_max_nnz": (
            coupled.preconditioned_load_max_nnz
        ),
        "true_coupled_coarse_preconditioned_load_drop_tol": (
            coupled.preconditioned_load_drop_tol
        ),
        "true_coupled_coarse_damping": coupled.damping,
        "true_coupled_coarse_beta_max": coupled.beta_max,
        "true_coupled_coarse_accept_base_improvement": (
            coupled.accept_base_improvement
        ),
        "true_coupled_coarse_base_improvement_override_used": (
            context.true_coupled_coarse_base_improvement_override_used
        ),
        "true_coupled_coarse_residual_after": (
            context.true_coupled_coarse_residual_after
        ),
        "true_coupled_coarse_error": context.true_coupled_coarse_error,
        "true_coupled_coarse_metadata": context.true_coupled_coarse_metadata,
        "true_active_submatrix_requested": active.active_submatrix_requested,
        "true_active_submatrix_selected": context.true_active_submatrix_selected,
        "true_active_submatrix_damping": active.active_submatrix_damping,
        "true_active_submatrix_alpha_clip": active.active_submatrix_alpha_clip,
        "true_active_submatrix_min_improvement": (
            active.active_submatrix_min_improvement
        ),
        "true_active_submatrix_residual_after": (
            context.true_active_submatrix_residual_after
        ),
        "true_active_submatrix_error": context.true_active_submatrix_error,
        "true_active_submatrix_metadata": context.true_active_submatrix_metadata,
        "true_active_column_cache_requested": active.active_column_cache_requested,
        "true_active_column_cache_max_mb": active.active_column_cache_max_mb,
        "true_active_column_cache_metadata": (
            context.true_active_column_cache_metadata
        ),
        "true_active_block_requested": active.active_block_requested,
        "true_active_block_selected": context.true_active_block_selected,
        "true_active_block_x_count": active.active_block_x_count,
        "true_active_block_ell_count": active.active_block_ell_count,
        "true_active_block_theta_stride": active.active_block_theta_stride,
        "true_active_block_zeta_stride": active.active_block_zeta_stride,
        "true_active_block_max_mb": active.active_block_max_mb,
        "true_active_block_regularization": active.active_block_regularization,
        "true_active_block_max_size": active.active_block_max_size,
        "true_active_block_column_batch": active.active_block_column_batch,
        "true_active_block_drop_tol": active.active_block_drop_tol,
        "true_active_block_include_tail": active.active_block_include_tail,
        "true_active_block_max_tail": active.active_block_max_tail,
        "true_active_block_damping": active.active_block_damping,
        "true_active_block_beta_max": active.active_block_beta_max,
        "true_active_block_residual_after": context.true_active_block_residual_after,
        "true_active_block_error": context.true_active_block_error,
        "true_active_block_metadata": context.true_active_block_metadata,
        "true_active_residual_block_requested": (
            active.active_residual_block_requested
        ),
        "true_active_residual_block_selected": (
            context.true_active_residual_block_selected
        ),
        "true_active_residual_block_max_mb": active.active_residual_block_max_mb,
        "true_active_residual_block_regularization": (
            active.active_residual_block_regularization
        ),
        "true_active_residual_block_max_size": (
            active.active_residual_block_max_size
        ),
        "true_active_residual_block_column_batch": (
            active.active_residual_block_column_batch
        ),
        "true_active_residual_block_drop_tol": (
            active.active_residual_block_drop_tol
        ),
        "true_active_residual_block_include_tail": (
            active.active_residual_block_include_tail
        ),
        "true_active_residual_block_max_tail": (
            active.active_residual_block_max_tail
        ),
        "true_active_residual_block_kinetic_only": (
            active.active_residual_block_kinetic_only
        ),
        "true_active_residual_block_damping": (
            active.active_residual_block_damping
        ),
        "true_active_residual_block_beta_max": (
            active.active_residual_block_beta_max
        ),
        "true_active_residual_block_min_improvement": (
            active.active_residual_block_min_improvement
        ),
        "true_active_residual_block_accept_base_improvement": (
            active.active_residual_block_accept_base_improvement
        ),
        "true_active_residual_block_base_improvement_override_used": (
            context.true_active_residual_block_base_improvement_override_used
        ),
        "true_active_residual_block_residual_after": (
            context.true_active_residual_block_residual_after
        ),
        "true_active_residual_block_error": (
            context.true_active_residual_block_error
        ),
        "true_active_residual_block_metadata": (
            context.true_active_residual_block_metadata
        ),
        "true_window_requested": residual.true_window_requested,
        "true_window_selected": context.true_window_selected,
        "true_window_max_windows": residual.true_window_max_windows,
        "true_window_x_radius": residual.true_window_x_radius,
        "true_window_ell_radius": residual.true_window_ell_radius,
        "true_window_max_mb": residual.true_window_max_mb,
        "true_window_regularization": residual.true_window_regularization,
        "true_window_max_size": residual.true_window_max_size,
        "true_window_column_batch": residual.true_window_column_batch,
        "true_window_drop_tol": residual.true_window_drop_tol,
        "true_window_include_tail": residual.true_window_include_tail,
        "true_window_damping": residual.true_window_damping,
        "true_window_beta_max": residual.true_window_beta_max,
        "true_window_residual_after": context.true_window_residual_after,
        "true_window_error": context.true_window_error,
        "true_window_metadata": context.true_window_metadata,
        "residual_window_requested": residual.residual_window_requested,
        "residual_window_selected": context.residual_window_selected,
        "residual_window_max_windows": residual.residual_window_max_windows,
        "residual_window_x_radius": residual.residual_window_x_radius,
        "residual_window_ell_radius": residual.residual_window_ell_radius,
        "residual_window_max_mb": residual.residual_window_max_mb,
        "residual_window_regularization": residual.residual_window_regularization,
        "residual_window_coefficient_mode": (
            residual.residual_window_coefficient_mode
        ),
        "residual_window_combine_mode": residual.residual_window_combine_mode,
        "residual_window_interface_depth": residual.residual_window_interface_depth,
        "residual_window_max_size": residual.residual_window_max_size,
        "residual_window_residual_after": context.residual_window_residual_after,
        "residual_window_error": context.residual_window_error,
        "residual_window_metadata": context.residual_window_metadata,
    }


def sparse_pc_direct_tail_final_metadata(
    context: SparsePCDirectTailFinalMetadataContext,
) -> dict[str, object]:
    """Build final sparse-PC direct-tail metadata from grouped solver state."""

    return sparse_pc_direct_tail_result_metadata_from_context(
        SparsePCDirectTailMetadataContext(
            structured_pc_preflight_required=(
                context.structured_pc_preflight_required
            ),
            structured_pc_preflight_required_min_size=(
                context.structured_pc_preflight_required_min_size
            ),
            suffix_values=_direct_tail_final_suffix_values(context),
            true_active_block_species_count=(
                context.true_active_block_species_count
            ),
            true_window_specs=context.true_window_specs,
            operator_bundle=context.materialization.operator_bundle,
            structured_max_nbytes=context.structured_max_nbytes,
            enabled=context.materialization.enabled,
            direct_reduced_pmat_requested=(
                context.materialization.direct_reduced_pmat_requested
            ),
            built=context.materialization.built,
            error=context.materialization.error,
            structured_pc_requested=context.structured_admission.requested,
            structured_pc_required=context.structured_admission.required,
            structured_pc_selected=context.structured_pc_selected,
            structured_pc_reason=context.structured_pc_reason,
            structured_pc_error=context.structured_pc_error,
            structured_pc_max_mb_auto=context.structured_admission.max_mb_auto,
            structured_pc_metadata=context.structured_pc_metadata,
            support_mode_preflight_requested=(
                context.support_mode_preflight_requested
            ),
            support_mode_preflight_selected=(
                context.support_mode_preflight_selected
            ),
            support_mode_preflight_error=context.support_mode_preflight_error,
            support_mode_preflight_metadata=(
                context.support_mode_preflight_metadata
            ),
        )
    )


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
    builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]


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
    host_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]
    device_builder: Callable[..., tuple[ArrayFn, dict[str, object], dict[str, int]]]


@dataclass(frozen=True)
class XBlockGlobalCouplingStageResult:
    """Result from optional primary x-block global-coupling setup."""

    preconditioner: ArrayFn
    built: bool
    metadata: dict[str, object]
    stats: dict[str, int]
    setup_s: float


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
class XBlockQICoarseSeedStageContext:
    """Dependencies for optional QI coarse residual-seed setup."""

    op: object
    x0_full: jnp.ndarray | None
    xblock_rhs: jnp.ndarray
    matvec_no_count: ArrayFn
    active_dof: bool
    linear_size: int
    policy: XBlockQISeedPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    basis_builder: Callable[..., RHS1QICoarseBasis]
    correction_builder: Callable[..., object]


@dataclass(frozen=True)
class XBlockQICoarseSeedStageResult:
    """Result from optional QI coarse residual-seed setup."""

    x0_full: jnp.ndarray | None
    basis_for_galerkin: RHS1QICoarseBasis | None
    used: bool
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    rank: int
    candidate_count: int
    reason: str | None
    labels: tuple[str, ...]
    setup_s: float


@dataclass(frozen=True)
class XBlockQIGalerkinStageContext:
    """Dependencies for optional QI Galerkin preconditioner setup."""

    op: object
    base_preconditioner: ArrayFn
    matvec: ArrayFn
    true_matvec_no_count: ArrayFn
    xblock_rhs: jnp.ndarray
    xblock_rhs_norm: float
    active_dof: bool
    linear_size: int
    basis_for_galerkin: RHS1QICoarseBasis | None
    seed_policy: XBlockQISeedPolicySetup
    galerkin_policy: XBlockQIGalerkinPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    basis_builder: Callable[..., RHS1QICoarseBasis]
    preconditioner_builder: Callable[..., object]


@dataclass(frozen=True)
class XBlockQIGalerkinStageResult:
    """Result from optional QI Galerkin preconditioner setup."""

    preconditioner: ArrayFn
    basis_for_galerkin: RHS1QICoarseBasis | None
    built: bool
    used: bool
    reason: str | None
    mode: str | None
    rank: int
    candidate_count: int
    coarse_shape: tuple[int, int]
    coarse_norm: float
    setup_s: float
    rcond: float
    damping: float
    basis_reused_from_seed: bool
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    probe_reduced: bool
    probe_candidates: list[dict[str, object]]
    selected_index: int | None
    stats: dict[str, int]


@dataclass(frozen=True)
class XBlockQITwoLevelStageContext:
    """Dependencies for optional QI two-level preconditioner setup."""

    op: object
    rhs: jnp.ndarray
    x0_full: jnp.ndarray | None
    xblock_rhs: jnp.ndarray
    base_preconditioner: ArrayFn
    matvec: ArrayFn
    true_matvec_no_count: ArrayFn
    direction_projector: ArrayFn | None
    active_dof: bool
    linear_size: int
    basis_for_galerkin: RHS1QICoarseBasis | None
    seed_policy: XBlockQISeedPolicySetup
    two_level_policy: XBlockQITwoLevelPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    basis_builder: Callable[..., RHS1QICoarseBasis]
    smoothed_load_basis_builder: Callable[..., tuple[RHS1QICoarseBasis, dict[str, object]]]
    orthonormalizer: Callable[..., RHS1QICoarseBasis]
    preconditioner_builder: Callable[..., object]


@dataclass(frozen=True)
class XBlockQITwoLevelStageResult:
    """Result from optional QI two-level preconditioner setup."""

    preconditioner: ArrayFn
    x0_full: jnp.ndarray | None
    basis_for_galerkin: RHS1QICoarseBasis | None
    built: bool
    used: bool
    reason: str | None
    rank: int
    candidate_count: int
    coarse_shape: tuple[int, int]
    coarse_norm: float
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    coarse_solver: str | None
    residual_augmented: bool
    rank_before_augmentation: int
    augmentation_labels: tuple[str, ...]
    residual_augment_max_extra: int
    residual_augment_steps: int
    residual_augment_include_residuals: bool
    smoothed_load_basis: bool
    smoothed_load_metadata: dict[str, object]
    setup_s: float
    rcond: float
    damping: float
    basis_reused_from_seed: bool
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    probe_candidates: list[dict[str, object]]
    selected_index: int | None
    stats: dict[str, int]


@dataclass(frozen=True)
class XBlockQIDeviceMetadataContext:
    """Explicit inputs for QI device preconditioner diagnostic metadata."""

    probe: object
    state: object
    basis_reused_from_seed: bool
    min_improvement: float
    cycles_requested: int
    minres_step: bool
    alpha_clip: float
    augmented_seed_requested: bool
    augmented_seed_available: bool
    augmented_seed_used: bool
    augmented_seed_rank: int
    augmented_seed_max_rank: int
    augmented_seed_reason: str | None
    augmented_seed_projection_residual: float | None
    augmented_seed_labels: Sequence[str]
    use_in_krylov: bool
    use_in_krylov_requested: bool
    precondition_side: str
    compose_with_base: bool
    compose_mode: str
    matrix_free_enabled: bool
    local_smoother_kind: str
    enrichment_config: object
    multilevel_config: object
    multilevel_max_rank: int | None
    extra_coarse_metadata: Mapping[str, object]
    residual_correction_metadata: Mapping[str, object]
    max_rank_requested: int | None


@dataclass(frozen=True)
class XBlockQIDeviceSetupConfigContext:
    """Inputs for building the QI device preconditioner setup contract."""

    op: object
    active_dof: bool
    linear_size: int
    base_config: object
    enrichment_config: object
    multilevel_config: object
    multilevel_max_rank: int | None
    max_rank: int | None
    extra_coarse_controls: Mapping[str, object]
    extra_coarse_setup_kwargs: Mapping[str, object]
    residual_correction_setup_kwargs: Mapping[str, object]


@dataclass(frozen=True)
class XBlockQIDeviceSetupConfig:
    """Geometry metadata and config object for device preconditioner setup."""

    geometry_metadata: dict[str, object]
    config: RHS1QIDevicePreconditionerConfig


@dataclass(frozen=True)
class XBlockQIDeflatedPolicySetup:
    """Environment controls for the QI residual-deflated preconditioner."""

    krylov_depth: int
    max_rank: int
    rcond: float
    basis_rtol: float
    min_improvement: float
    damping: float
    correction_cycles: int
    use_in_krylov: bool
    seed_solver: str
    composition: str
    include_raw_residual: bool
    extra_global_loads: bool
    extra_smooth_loads: bool
    extra_max_directions: int
    extra_fsavg_lmax: int
    extra_angular_lmax: int
    extra_max_extra_units: int
    extra_include_rhs: bool


@dataclass(frozen=True)
class XBlockQIDeflatedStageContext:
    """Dependencies for optional QI residual-deflated preconditioner setup."""

    op: object
    rhs: jnp.ndarray
    x0_full: jnp.ndarray | None
    xblock_rhs: jnp.ndarray
    base_preconditioner: ArrayFn
    matvec: ArrayFn
    true_matvec_no_count: ArrayFn
    active_dof: bool
    reduce_full: ArrayFn | None
    policy: XBlockQIDeflatedPolicySetup
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    global_load_basis_builder: Callable[..., Sequence[tuple[str, jnp.ndarray]]]
    preconditioner_builder: Callable[..., object]
    minres_seed_probe: Callable[..., tuple[jnp.ndarray, object]]
    linear_probe: Callable[..., tuple[jnp.ndarray, object]]


@dataclass(frozen=True)
class XBlockQIDeflatedStageResult:
    """Result from optional QI residual-deflated setup and seed probe."""

    preconditioner: ArrayFn
    x0_full: jnp.ndarray | None
    built: bool
    used: bool
    used_in_krylov: bool
    reason: str | None
    rank: int
    candidate_count: int
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    metadata: dict[str, object]
    setup_s: float
    stats: dict[str, int]
    correction_cycles: int
    use_in_krylov: bool
    seed_solver: str


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


def retry_sparse_pc_factor_dtype_from_finalization_context(
    context: SparsePCFactorDtypeRetryFinalizationContext,
    *,
    factor_dtype_used: np.dtype,
    factor_dtype_retry: str | None,
    residual_norm: float,
    preconditioned_residual_norm: float,
    history: Sequence[float],
    target: float,
    x: np.ndarray,
    solve_s: float,
    operator_bundle: Any,
    factor_bundle: Any,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
) -> SparsePCFactorDtypeRetryResult:
    """Retry sparse-PC factor precision from explicit finalization inputs."""

    def build_factor(factor_dtype_arg: np.dtype) -> tuple[Any, Any]:
        return build_host_sparse_direct_factor_from_matvec(
            matvec=context.factor_matvec,
            n=int(context.linear_size),
            dtype=np.dtype(context.rhs_dtype),
            factor_dtype=np.dtype(factor_dtype_arg),
            pattern=context.pattern,
            emit=context.emit,
            default_diag_pivot_thresh=(
                0.0
                if (
                    bool(context.constrained_pas_pc)
                    or bool(context.tokamak_fp_pc)
                    or bool(context.fortran_reduced_sparse_pc)
                )
                else 1.0
            ),
            default_permc_spec=context.default_permc_spec,
            default_factor_kind=context.default_factor_kind,
            default_ilu_fill_factor=float(context.default_ilu_fill_factor),
            default_ilu_drop_tol=float(context.default_ilu_drop_tol),
            default_pattern_color_batch=int(context.default_pattern_color_batch),
        )

    return retry_sparse_pc_factor_dtype_if_needed(
        SparsePCFactorDtypeRetryContext(
            factor_dtype_used=np.dtype(factor_dtype_used),
            factor_dtype_retry=factor_dtype_retry,
            residual_norm=float(residual_norm),
            preconditioned_residual_norm=float(preconditioned_residual_norm),
            history=tuple(float(v) for v in (history or ())),
            target=float(target),
            x=np.asarray(x, dtype=np.float64),
            x0_fallback=context.x0_fallback,
            solve_s=float(solve_s),
            pc_maxiter=int(context.pc_maxiter),
            operator_bundle=operator_bundle,
            factor_bundle=factor_bundle,
            elapsed_s=context.elapsed_s,
            emit=context.emit,
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
        preconditioner, metadata, stats = context.builder(
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
        builder = (
            context.device_builder
            if bool(context.policy.use_device_builder)
            else context.host_builder
        )
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


def _object_metadata_dict(metadata: object) -> dict[str, object]:
    """Return a plain metadata dictionary from dataclass-like solver metadata."""

    if hasattr(metadata, "to_dict"):
        return dict(metadata.to_dict())
    if isinstance(metadata, Mapping):
        return dict(metadata)
    return {}


def build_xblock_qi_device_preconditioner_metadata(
    context: XBlockQIDeviceMetadataContext,
) -> dict[str, object]:
    """Build stable diagnostics for the QI device preconditioner probe."""

    probe = context.probe
    state = context.state
    probe_metadata = _object_metadata_dict(getattr(probe, "metadata", {}))
    probe_cycles = int(
        getattr(
            probe,
            "cycles",
            1 if bool(getattr(probe, "accepted", False)) else 0,
        )
    )
    residual_history = tuple(
        float(value)
        for value in getattr(
            probe,
            "residual_history",
            (
                float(getattr(probe, "residual_before_norm", float("nan"))),
                float(getattr(probe, "residual_after_norm", float("nan"))),
            ),
        )
    )
    step_history = tuple(float(value) for value in getattr(probe, "step_history", ()))
    local_smoother = getattr(state, "local_smoother", None)
    local_smoother_metadata = None
    if local_smoother is not None:
        local_metadata = getattr(local_smoother, "metadata", None)
        if hasattr(local_metadata, "to_dict"):
            local_smoother_metadata = dict(local_metadata.to_dict())

    enrichment = context.enrichment_config
    multilevel = context.multilevel_config
    return {
        **probe_metadata,
        "basis_reused_from_seed": bool(context.basis_reused_from_seed),
        "min_improvement": float(context.min_improvement),
        "cycles_requested": int(context.cycles_requested),
        "cycles": int(probe_cycles),
        "residual_history": residual_history,
        "step_policy": "residual_minimizing" if bool(context.minres_step) else "fixed",
        "alpha_clip": float(context.alpha_clip),
        "step_history": step_history,
        "augmented_seed_requested": bool(context.augmented_seed_requested),
        "augmented_seed_available": bool(context.augmented_seed_available),
        "augmented_seed_used": bool(context.augmented_seed_used),
        "augmented_seed_rank": int(context.augmented_seed_rank),
        "augmented_seed_max_rank": int(context.augmented_seed_max_rank),
        "augmented_seed_reason": context.augmented_seed_reason,
        "augmented_seed_projection_residual_norm": (
            None
            if context.augmented_seed_projection_residual is None
            else float(context.augmented_seed_projection_residual)
        ),
        "augmented_seed_labels": tuple(
            str(label) for label in context.augmented_seed_labels
        ),
        "use_in_krylov": bool(context.use_in_krylov),
        "use_in_krylov_requested": bool(context.use_in_krylov_requested),
        "precondition_side": str(context.precondition_side),
        "compose_with_base": bool(context.compose_with_base),
        "compose_mode": str(context.compose_mode),
        "use_in_krylov_blocked_by_precondition_side_none": bool(
            context.use_in_krylov_requested and str(context.precondition_side) == "none"
        ),
        "matrix_free_enabled": bool(context.matrix_free_enabled),
        "local_smoother_kind_requested": str(context.local_smoother_kind),
        "local_smoother_metadata": local_smoother_metadata,
        "residual_enrichment_requested": bool(
            getattr(enrichment, "residual_enrichment", False)
        ),
        "residual_enrichment_depth_requested": int(
            getattr(enrichment, "residual_enrichment_depth", 0)
        ),
        "residual_enrichment_include_residual": bool(
            getattr(enrichment, "residual_enrichment_include_residual", False)
        ),
        "recycle_enrichment_requested": bool(
            getattr(enrichment, "recycle_enrichment", False)
        ),
        "recycle_enrichment_cycles_requested": int(
            getattr(enrichment, "recycle_cycles", 0)
        ),
        "operator_krylov_enrichment_requested": bool(
            getattr(enrichment, "operator_krylov_enrichment", False)
        ),
        "operator_krylov_depth_requested": int(
            getattr(enrichment, "operator_krylov_depth", 0)
        ),
        "adjoint_krylov_enrichment_requested": bool(
            getattr(enrichment, "adjoint_krylov_enrichment", False)
        ),
        "adjoint_krylov_depth_requested": int(
            getattr(enrichment, "adjoint_krylov_depth", 0)
        ),
        "adjoint_krylov_transpose_requested": getattr(
            enrichment,
            "adjoint_krylov_transpose_source",
            None,
        ),
        "operator_action_enrichment_requested": bool(
            getattr(enrichment, "operator_action_enrichment", False)
        ),
        "operator_action_depth_requested": int(
            getattr(enrichment, "operator_action_depth", 0)
        ),
        "multilevel_coarse_requested": bool(
            getattr(multilevel, "multilevel_coarse", False)
        ),
        "multilevel_max_levels_requested": int(
            getattr(multilevel, "multilevel_max_levels", 1)
        ),
        "multilevel_aggregate_factor_requested": int(
            getattr(multilevel, "multilevel_aggregate_factor", 2)
        ),
        "multilevel_max_rank_requested": (
            None
            if context.multilevel_max_rank is None
            else int(context.multilevel_max_rank)
        ),
        "multilevel_max_angular_mode_requested": int(
            getattr(multilevel, "multilevel_max_angular_mode", 0)
        ),
        "multilevel_max_radial_degree_requested": int(
            getattr(multilevel, "multilevel_max_radial_degree", 0)
        ),
        "multilevel_max_pitch_degree_requested": int(
            getattr(multilevel, "multilevel_max_pitch_degree", 0)
        ),
        "multilevel_current_moments_requested": bool(
            getattr(multilevel, "multilevel_current_moments", False)
        ),
        "multilevel_species_current_moments_requested": bool(
            getattr(multilevel, "multilevel_species_current_moments", False)
        ),
        "multilevel_radial_current_moments_requested": bool(
            getattr(multilevel, "multilevel_radial_current_moments", False)
        ),
        "multilevel_tail_constraint_moments_requested": bool(
            getattr(multilevel, "multilevel_tail_constraint_moments", False)
        ),
        "multilevel_current_max_pitch_degree_requested": int(
            getattr(multilevel, "multilevel_current_max_pitch_degree", 0)
        ),
        "multilevel_residual_equation_requested": bool(
            getattr(multilevel, "multilevel_residual_equation", False)
        ),
        "multilevel_residual_equation_max_level_rank_requested": int(
            getattr(multilevel, "multilevel_residual_equation_max_level_rank", 0)
        ),
        "multilevel_residual_equation_order_requested": getattr(
            multilevel,
            "multilevel_residual_equation_order",
            None,
        ),
        "multilevel_residual_equation_solver_requested": getattr(
            multilevel,
            "multilevel_residual_equation_solver",
            None,
        ),
        "multilevel_residual_equation_include_global_requested": bool(
            getattr(multilevel, "multilevel_residual_equation_include_global", False)
        ),
        **dict(context.extra_coarse_metadata),
        **dict(context.residual_correction_metadata),
        "max_rank_requested": (
            None
            if context.max_rank_requested is None
            else int(context.max_rank_requested)
        ),
    }


def build_xblock_qi_device_setup_config(
    context: XBlockQIDeviceSetupConfigContext,
) -> XBlockQIDeviceSetupConfig:
    """Build geometry metadata and config for the QI device preconditioner."""

    base = context.base_config
    enrichment = context.enrichment_config
    multilevel = context.multilevel_config
    active_dof = bool(context.active_dof)
    linear_size = int(context.linear_size)
    extra_coarse_controls = dict(context.extra_coarse_controls)
    include_tail_block = rhs1_qi_device_tail_block_required(
        multilevel_coarse=bool(getattr(multilevel, "multilevel_coarse", False)),
        extra_coarse_controls=extra_coarse_controls,
    )
    geometry_metadata: dict[str, object] = {
        "rhs_mode": int(getattr(context.op, "rhs_mode")),
        "n_theta": int(getattr(context.op, "n_theta", 1)),
        "n_zeta": int(getattr(context.op, "n_zeta", 1)),
        "n_x": int(getattr(context.op, "n_x", 1)),
        "n_species": int(getattr(context.op, "n_species", 1)),
        "active_dof": active_dof,
        **rhs1_xblock_qi_block_geometry_metadata(
            op=context.op,
            active_dof=active_dof,
            linear_size=linear_size,
            include_tail_block=bool(include_tail_block),
        ),
    }
    config = RHS1QIDevicePreconditionerConfig(
        regularization_rcond=float(getattr(base, "rcond")),
        damping=float(getattr(base, "damping")),
        coarse_solver=getattr(base, "coarse_solver"),
        jacobi_damping=float(getattr(base, "jacobi_damping")),
        jacobi_sweeps=int(getattr(base, "jacobi_sweeps")),
        jacobi_step_policy=getattr(base, "jacobi_step_policy"),
        jacobi_diagonal_floor=float(getattr(base, "jacobi_floor")),
        jacobi_require_all_diagonal=bool(
            getattr(base, "jacobi_require_all_diagonal")
        ),
        local_smoother_kind=getattr(base, "local_smoother_kind"),
        matrix_free_smoother_sweeps=int(
            getattr(base, "matrix_free_smoother_sweeps")
        ),
        matrix_free_smoother_damping=float(
            getattr(base, "matrix_free_smoother_damping")
        ),
        matrix_free_smoother_step_policy=getattr(
            base,
            "matrix_free_smoother_step_policy",
        ),
        matrix_free_smoother_alpha_clip=float(
            getattr(base, "matrix_free_smoother_alpha_clip")
        ),
        matrix_free_block_smoother_max_groups=int(
            getattr(base, "matrix_free_block_smoother_max_groups")
        ),
        matrix_free_block_smoother_include_tail=bool(
            getattr(base, "matrix_free_block_smoother_include_tail")
        ),
        matrix_free_block_smoother_rcond=float(
            getattr(base, "matrix_free_block_smoother_rcond")
        ),
        matrix_free_block_smoother_grouping=getattr(
            base,
            "matrix_free_block_smoother_grouping",
        ),
        max_rank=context.max_rank,
        residual_enrichment=bool(getattr(enrichment, "residual_enrichment")),
        residual_enrichment_depth=int(
            getattr(enrichment, "residual_enrichment_depth")
        ),
        residual_enrichment_include_residual=bool(
            getattr(enrichment, "residual_enrichment_include_residual")
        ),
        recycle_enrichment=bool(getattr(enrichment, "recycle_enrichment")),
        recycle_enrichment_cycles=int(getattr(enrichment, "recycle_cycles")),
        operator_krylov_enrichment=bool(
            getattr(enrichment, "operator_krylov_enrichment")
        ),
        operator_krylov_depth=int(getattr(enrichment, "operator_krylov_depth")),
        adjoint_krylov_enrichment=bool(
            getattr(enrichment, "adjoint_krylov_enrichment")
        ),
        adjoint_krylov_depth=int(getattr(enrichment, "adjoint_krylov_depth")),
        adjoint_krylov_transpose_source=getattr(
            enrichment,
            "adjoint_krylov_transpose_source",
        ),
        operator_action_enrichment=bool(
            getattr(enrichment, "operator_action_enrichment")
        ),
        operator_action_enrichment_depth=int(
            getattr(enrichment, "operator_action_depth")
        ),
        multilevel_coarse=bool(getattr(multilevel, "multilevel_coarse")),
        multilevel_max_levels=int(getattr(multilevel, "multilevel_max_levels")),
        multilevel_aggregate_factor=int(
            getattr(multilevel, "multilevel_aggregate_factor")
        ),
        multilevel_max_rank=context.multilevel_max_rank,
        multilevel_max_angular_mode=int(
            getattr(multilevel, "multilevel_max_angular_mode")
        ),
        multilevel_max_radial_degree=int(
            getattr(multilevel, "multilevel_max_radial_degree")
        ),
        multilevel_max_pitch_degree=int(
            getattr(multilevel, "multilevel_max_pitch_degree")
        ),
        multilevel_current_moments=bool(
            getattr(multilevel, "multilevel_current_moments")
        ),
        multilevel_species_current_moments=bool(
            getattr(multilevel, "multilevel_species_current_moments")
        ),
        multilevel_radial_current_moments=bool(
            getattr(multilevel, "multilevel_radial_current_moments")
        ),
        multilevel_tail_constraint_moments=bool(
            getattr(multilevel, "multilevel_tail_constraint_moments")
        ),
        multilevel_current_max_pitch_degree=int(
            getattr(multilevel, "multilevel_current_max_pitch_degree")
        ),
        multilevel_residual_equation=bool(
            getattr(multilevel, "multilevel_residual_equation")
        ),
        multilevel_residual_equation_max_level_rank=int(
            getattr(multilevel, "multilevel_residual_equation_max_level_rank")
        ),
        multilevel_residual_equation_order=getattr(
            multilevel,
            "multilevel_residual_equation_order",
        ),
        multilevel_residual_equation_solver=getattr(
            multilevel,
            "multilevel_residual_equation_solver",
        ),
        multilevel_residual_equation_include_global=bool(
            getattr(multilevel, "multilevel_residual_equation_include_global")
        ),
        **dict(context.extra_coarse_setup_kwargs),
        **dict(context.residual_correction_setup_kwargs),
    )
    return XBlockQIDeviceSetupConfig(
        geometry_metadata=geometry_metadata,
        config=config,
    )


def resolve_xblock_qi_deflated_policy_setup(
    env: Mapping[str, str] | None = None,
) -> XBlockQIDeflatedPolicySetup:
    """Resolve QI residual-deflated preconditioner controls."""

    seed_solver = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_SEED_SOLVER",
        )
        or "cycle_minres"
    ).lower().replace("-", "_")
    if seed_solver in {"minres", "cycle_minres", "cycle_lstsq", "gcro_seed"}:
        seed_solver = "cycle_minres"
    else:
        seed_solver = "linear_apply"
    composition = (
        _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_COMPOSITION",
        )
        or "multiplicative"
    ).lower().replace("-", "_")
    return XBlockQIDeflatedPolicySetup(
        krylov_depth=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_KRYLOV_DEPTH",
            default=4,
            minimum=0,
        ),
        max_rank=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_MAX_RANK",
            default=16,
            minimum=1,
        ),
        rcond=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_RCOND",
                default=1.0e-12,
            ),
        ),
        basis_rtol=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_BASIS_RTOL",
                default=1.0e-10,
            ),
        ),
        min_improvement=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_MIN_IMPROVEMENT",
                default=0.05,
            ),
        ),
        damping=max(
            0.0,
            _env_float(
                env,
                "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_DAMPING",
                default=1.0,
            ),
        ),
        correction_cycles=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_CYCLES",
            default=8,
            minimum=1,
        ),
        use_in_krylov=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_USE_IN_KRYLOV",
            default=False,
        ),
        seed_solver=seed_solver,
        composition=composition,
        include_raw_residual=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_INCLUDE_RAW_RESIDUAL",
            default=False,
        ),
        extra_global_loads=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_GLOBAL_LOADS",
            default=True,
        ),
        extra_smooth_loads=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_SMOOTH_LOADS",
            default=True,
        ),
        extra_max_directions=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_MAX_DIRECTIONS",
            default=16,
            minimum=0,
        ),
        extra_fsavg_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_FSAVG_LMAX",
            default=4,
            minimum=0,
        ),
        extra_angular_lmax=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_ANGULAR_LMAX",
            default=1,
            minimum=0,
        ),
        extra_max_extra_units=_env_int(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_MAX_EXTRA_UNITS",
            default=8,
            minimum=0,
        ),
        extra_include_rhs=_env_bool(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER_EXTRA_INCLUDE_RHS",
            default=True,
        ),
    )


def apply_xblock_qi_deflated_stage(
    *,
    context: XBlockQIDeflatedStageContext,
) -> XBlockQIDeflatedStageResult:
    """Build, probe, and optionally install a QI residual-deflated preconditioner."""

    start_s = float(context.elapsed_s())
    policy = context.policy
    stats = {"applies": 0, "local_applies": 0}
    x0_full = context.x0_full
    built = False
    used = False
    used_in_krylov = False
    reason: str | None = None
    rank = 0
    candidate_count = 0
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_ratio: float | None = None
    metadata: dict[str, object] = {}
    preconditioner = context.base_preconditioner

    try:
        def local_smoother(v: jnp.ndarray) -> jnp.ndarray:
            stats["local_applies"] += 1
            return jnp.asarray(
                context.base_preconditioner(jnp.asarray(v, dtype=jnp.float64)),
                dtype=jnp.float64,
            )

        current = (
            jnp.zeros_like(context.xblock_rhs)
            if x0_full is None
            else jnp.asarray(x0_full, dtype=jnp.float64)
        )
        residual_seed = context.xblock_rhs - jnp.asarray(
            context.true_matvec_no_count(current),
            dtype=jnp.float64,
        )
        extra_directions: list[tuple[str, jnp.ndarray]] = []
        if bool(policy.extra_global_loads) and int(policy.extra_max_directions) > 0:
            raw_loads = context.global_load_basis_builder(
                op=context.op,
                rhs=context.rhs,
                include_rhs=bool(policy.extra_include_rhs),
                fsavg_lmax=int(policy.extra_fsavg_lmax),
                angular_lmax=int(policy.extra_angular_lmax),
                max_extra_units=int(policy.extra_max_extra_units),
                max_directions=int(policy.extra_max_directions),
            )
            for load_name, load_values in raw_loads[: int(policy.extra_max_directions)]:
                load_vec = jnp.asarray(load_values, dtype=jnp.float64).reshape((-1,))
                if bool(context.active_dof):
                    if context.reduce_full is None:
                        raise RuntimeError("QI deflated active-DOF stage requires reduce_full")
                    load_vec = context.reduce_full(load_vec)
                load_norm = float(jnp.linalg.norm(load_vec))
                if not np.isfinite(load_norm) or load_norm <= 0.0:
                    continue
                load_vec = load_vec / jnp.asarray(load_norm, dtype=load_vec.dtype)
                if bool(policy.extra_smooth_loads):
                    load_vec = local_smoother(load_vec)
                extra_directions.append((f"global_load:{load_name}", load_vec))

        qi_deflated = context.preconditioner_builder(
            operator=context.matvec,
            local_smoother=local_smoother,
            residual_seed=residual_seed,
            extra_directions=tuple(extra_directions),
            krylov_depth=int(policy.krylov_depth),
            max_rank=int(policy.max_rank),
            regularization_rcond=float(policy.rcond),
            basis_rtol=float(policy.basis_rtol),
            damping=float(policy.damping),
            correction_cycles=int(policy.correction_cycles),
            composition=policy.composition,
            include_raw_residual=bool(policy.include_raw_residual),
        )
        built = True
        if policy.seed_solver == "cycle_minres":
            x_candidate, probe = context.minres_seed_probe(
                operator=context.true_matvec_no_count,
                rhs=context.xblock_rhs,
                x0=current,
                preconditioner=qi_deflated,
                cycles=int(policy.correction_cycles),
                min_relative_improvement=float(policy.min_improvement),
                regularization_rcond=float(policy.rcond),
            )
        else:
            x_candidate, probe = context.linear_probe(
                operator=context.true_matvec_no_count,
                rhs=context.xblock_rhs,
                x0=current,
                preconditioner=qi_deflated,
                min_relative_improvement=float(policy.min_improvement),
            )

        metadata = {
            **_object_metadata_dict(getattr(probe, "metadata", {})),
            "seed_solver": getattr(probe, "seed_solver", policy.seed_solver),
            "cycle_residual_history": getattr(probe, "cycle_residual_history", ()),
            "cycle_coefficients": getattr(probe, "cycle_coefficients", ()),
        }
        probe_metadata = getattr(probe, "metadata", None)
        rank = int(getattr(probe_metadata, "rank", metadata.get("rank", 0)) or 0)
        candidate_count = int(
            getattr(
                probe_metadata,
                "candidate_count",
                metadata.get("candidate_count", 0),
            )
            or 0
        )
        residual_before = float(getattr(probe, "residual_before_norm"))
        residual_after = float(getattr(probe, "residual_after_norm"))
        improvement_ratio = getattr(probe, "improvement_ratio", None)
        reason = str(getattr(probe, "reason"))

        def deflated_preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            stats["applies"] += 1
            return jnp.asarray(
                qi_deflated.apply(jnp.asarray(v, dtype=jnp.float64)),
                dtype=jnp.float64,
            )

        if bool(getattr(probe, "accepted", False)):
            x0_full = jnp.asarray(x_candidate, dtype=jnp.float64)
            used = True
            if bool(policy.use_in_krylov):
                preconditioner = deflated_preconditioner
                used_in_krylov = True
            ratio_for_message = (
                float(improvement_ratio)
                if improvement_ratio is not None
                else float("nan")
            )
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI residual-deflated preconditioner accepted "
                    f"residual {float(residual_before):.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(rank={int(rank)} "
                    f"seed_solver={metadata['seed_solver']} "
                    f"cycles={int(policy.correction_cycles)} "
                    f"use_in_krylov={int(policy.use_in_krylov)} "
                    f"ratio={ratio_for_message:.6e})",
                )
        elif context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI residual-deflated preconditioner rejected "
                f"reason={reason} residual={float(residual_before):.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        metadata = {"error": reason}
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI residual-deflated preconditioner disabled after build failure "
                f"({type(exc).__name__}: {exc})",
            )

    return XBlockQIDeflatedStageResult(
        preconditioner=preconditioner,
        x0_full=x0_full,
        built=bool(built),
        used=bool(used),
        used_in_krylov=bool(used_in_krylov),
        reason=reason,
        rank=int(rank),
        candidate_count=int(candidate_count),
        residual_before=residual_before,
        residual_after=residual_after,
        improvement_ratio=improvement_ratio,
        metadata=metadata,
        setup_s=float(context.elapsed_s()) - start_s,
        stats=stats,
        correction_cycles=int(policy.correction_cycles),
        use_in_krylov=bool(policy.use_in_krylov),
        seed_solver=str(policy.seed_solver),
    )


def apply_xblock_qi_coarse_seed_stage(
    *,
    context: XBlockQICoarseSeedStageContext,
) -> XBlockQICoarseSeedStageResult:
    """Build a QI coarse basis and optionally use it as the initial x-block seed."""

    if not bool(context.policy.coarse_seed_enabled):
        return XBlockQICoarseSeedStageResult(
            x0_full=context.x0_full,
            basis_for_galerkin=None,
            used=False,
            residual_before=None,
            residual_after=None,
            improvement_ratio=None,
            rank=0,
            candidate_count=0,
            reason=None,
            labels=(),
            setup_s=0.0,
        )

    start_s = float(context.elapsed_s())
    basis_for_galerkin: RHS1QICoarseBasis | None = None
    used = False
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_ratio: float | None = None
    rank = 0
    candidate_count = 0
    reason: str | None = None
    labels: tuple[str, ...] = ()
    x0_full = context.x0_full
    try:
        basis_for_galerkin = context.basis_builder(
            op=context.op,
            active_dof=bool(context.active_dof),
            linear_size=int(context.linear_size),
            max_rank=int(context.policy.max_rank),
            rank_rtol=float(context.policy.rank_rtol),
            include_angular=bool(context.policy.include_angular),
            include_blocks=bool(context.policy.include_blocks),
            basis_kind=context.policy.basis_kind,
            max_candidates=int(context.policy.max_candidates),
            max_angular_mode=int(context.policy.max_angular_mode),
            include_radial=bool(context.policy.include_radial),
            include_radial_angular=bool(context.policy.include_radial_angular),
            include_constraint_moments=bool(context.policy.include_constraint_moments),
            include_schur=bool(context.policy.include_schur),
        )
        current = (
            jnp.zeros_like(context.xblock_rhs)
            if x0_full is None
            else jnp.asarray(x0_full, dtype=jnp.float64)
        )
        qi_result = context.correction_builder(
            context.matvec_no_count,
            context.xblock_rhs,
            current=current,
            basis=basis_for_galerkin,
            min_relative_improvement=float(context.policy.min_improvement),
            rcond=float(context.policy.rcond) if float(context.policy.rcond) > 0.0 else None,
        )
        residual_before = float(qi_result.residual_before_norm)
        residual_after = float(qi_result.residual_after_norm)
        improvement_ratio = float(qi_result.improvement_ratio)
        rank = int(qi_result.basis_metadata.rank)
        candidate_count = int(qi_result.basis_metadata.candidate_count)
        reason = str(qi_result.reason)
        labels = tuple(qi_result.basis_metadata.accepted_labels)
        if bool(qi_result.applied):
            x0_full = jnp.asarray(qi_result.solution, dtype=jnp.float64)
            used = True
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"QI coarse seed improved residual {float(residual_before):.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(rank={int(rank)} ratio={float(improvement_ratio):.6e})",
                )
        elif context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI coarse seed rejected reason={reason} "
                f"residual={float(residual_before):.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI coarse seed failed ({type(exc).__name__}: {exc})",
            )
    return XBlockQICoarseSeedStageResult(
        x0_full=x0_full,
        basis_for_galerkin=basis_for_galerkin,
        used=bool(used),
        residual_before=residual_before,
        residual_after=residual_after,
        improvement_ratio=improvement_ratio,
        rank=int(rank),
        candidate_count=int(candidate_count),
        reason=reason,
        labels=labels,
        setup_s=float(context.elapsed_s()) - start_s,
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


def apply_xblock_qi_galerkin_stage(
    *,
    context: XBlockQIGalerkinStageContext,
) -> XBlockQIGalerkinStageResult:
    """Build, probe, and optionally install a QI Galerkin preconditioner."""

    reason = (
        context.galerkin_policy.reason
        if context.galerkin_policy.reason is not None and not context.galerkin_policy.should_build
        else None
    )
    for level, message in context.galerkin_policy.messages:
        if context.emit is not None:
            context.emit(int(level), str(message))
    if not bool(context.galerkin_policy.should_build):
        return XBlockQIGalerkinStageResult(
            preconditioner=context.base_preconditioner,
            basis_for_galerkin=context.basis_for_galerkin,
            built=False,
            used=False,
            reason=reason,
            mode=None,
            rank=0,
            candidate_count=0,
            coarse_shape=(0, 0),
            coarse_norm=0.0,
            setup_s=0.0,
            rcond=0.0,
            damping=1.0,
            basis_reused_from_seed=False,
            residual_before=None,
            residual_after=None,
            improvement_ratio=None,
            probe_reduced=False,
            probe_candidates=[],
            selected_index=None,
            stats={"applies": 0, "coarse_applies": 0, "base_applies": 0},
        )

    start_s = float(context.elapsed_s())
    stats = {"applies": 0, "coarse_applies": 0, "base_applies": 0}
    basis_for_galerkin = context.basis_for_galerkin
    basis_reused_from_seed = basis_for_galerkin is not None
    mode = context.galerkin_policy.preconditioner_mode
    rcond = float(context.galerkin_policy.rcond)
    damping = float(context.galerkin_policy.damping)
    rank = 0
    candidate_count = 0
    coarse_shape = (0, 0)
    coarse_norm = 0.0
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_ratio: float | None = None
    probe_reduced = False
    probe_candidates: list[dict[str, object]] = []
    selected_index: int | None = None
    built = False
    used = False
    try:
        if basis_for_galerkin is None:
            basis_for_galerkin = context.basis_builder(
                op=context.op,
                active_dof=bool(context.active_dof),
                linear_size=int(context.linear_size),
                max_rank=int(context.seed_policy.max_rank),
                rank_rtol=float(context.seed_policy.rank_rtol),
                include_angular=bool(context.seed_policy.include_angular),
                include_blocks=bool(context.seed_policy.include_blocks),
                basis_kind=context.seed_policy.basis_kind,
                max_candidates=int(context.seed_policy.max_candidates),
                max_angular_mode=int(context.seed_policy.max_angular_mode),
                include_radial=bool(context.seed_policy.include_radial),
                include_radial_angular=bool(context.seed_policy.include_radial_angular),
                include_constraint_moments=bool(context.seed_policy.include_constraint_moments),
                include_schur=bool(context.seed_policy.include_schur),
            )
        qi_galerkin = context.preconditioner_builder(
            context.matvec,
            basis=basis_for_galerkin,
            rcond=float(rcond) if float(rcond) > 0.0 else None,
        )
        rank = int(qi_galerkin.metadata.rank)
        candidate_count = int(qi_galerkin.metadata.basis_metadata.candidate_count)
        coarse_shape = tuple(int(value) for value in qi_galerkin.metadata.coarse_operator_shape)
        coarse_norm = float(qi_galerkin.metadata.coarse_operator_norm)
        qi_galerkin_apply = qi_galerkin.as_preconditioner()
        mode_use = str(context.galerkin_policy.candidate_modes[0])
        damping_use = float(context.galerkin_policy.candidate_dampings[0])

        def preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            stats["applies"] += 1
            v_j = jnp.asarray(v, dtype=jnp.float64)
            base = jnp.asarray(context.base_preconditioner(v_j), dtype=jnp.float64)
            stats["base_applies"] += 1
            if mode_use == "multiplicative":
                coarse_input = v_j - jnp.asarray(context.matvec(base), dtype=jnp.float64)
            else:
                coarse_input = v_j
            coarse = jnp.asarray(qi_galerkin_apply(coarse_input), dtype=jnp.float64)
            stats["coarse_applies"] += 1
            return base + damping_use * coarse

        built = True
        reason = "built"
        if bool(context.galerkin_policy.probe_enabled):
            candidates: list[RHS1QIGalerkinProbeCandidate] = []
            v_probe = jnp.asarray(context.xblock_rhs, dtype=jnp.float64)
            base_probe = jnp.asarray(context.base_preconditioner(v_probe), dtype=jnp.float64)
            for candidate_mode in context.galerkin_policy.candidate_modes:
                if str(candidate_mode) == "multiplicative":
                    coarse_input = v_probe - jnp.asarray(context.matvec(base_probe), dtype=jnp.float64)
                else:
                    coarse_input = v_probe
                coarse_probe = jnp.asarray(qi_galerkin_apply(coarse_input), dtype=jnp.float64)
                for candidate_damping in context.galerkin_policy.candidate_dampings:
                    probe_solution = base_probe + float(candidate_damping) * coarse_probe
                    probe_residual = context.xblock_rhs - jnp.asarray(
                        context.true_matvec_no_count(probe_solution),
                        dtype=jnp.float64,
                    )
                    residual_norm = profile_l2_norm_float(probe_residual)
                    ratio_after = profile_safe_ratio(residual_norm, float(context.xblock_rhs_norm))
                    candidates.append(
                        RHS1QIGalerkinProbeCandidate(
                            mode=str(candidate_mode),
                            damping=float(candidate_damping),
                            residual_norm=float(residual_norm),
                            improvement_ratio=ratio_after,
                            reduced=bool(residual_norm < float(context.xblock_rhs_norm)),
                        )
                    )
            probe_selection = select_rhs1_qi_galerkin_probe_candidate(
                float(context.xblock_rhs_norm),
                candidates,
            )
            probe_candidates = [candidate.to_dict() for candidate in probe_selection.candidates]
            selected_index = probe_selection.selected_index
            residual_before = float(probe_selection.residual_before_norm)
            residual_after = probe_selection.residual_after_norm
            improvement_ratio = probe_selection.improvement_ratio
            probe_reduced = bool(probe_selection.accepted)
            if probe_selection.accepted:
                mode_use = str(probe_selection.selected_mode)
                damping_use = float(probe_selection.selected_damping)
                mode = mode_use
                damping = damping_use
                used = True
                reason = "probe_reduced"
            else:
                used = False
                reason = str(probe_selection.reason)
        else:
            used = True
            reason = "probe_disabled"
        if context.emit is not None:
            ratio = (
                f" probe_ratio={float(improvement_ratio):.6e}"
                if improvement_ratio is not None
                else ""
            )
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI Galerkin preconditioner built "
                f"mode={mode} rank={int(rank)} used={bool(used)} "
                f"reason={reason}{ratio}",
            )
        return XBlockQIGalerkinStageResult(
            preconditioner=preconditioner if bool(used) else context.base_preconditioner,
            basis_for_galerkin=basis_for_galerkin,
            built=bool(built),
            used=bool(used),
            reason=reason,
            mode=mode,
            rank=int(rank),
            candidate_count=int(candidate_count),
            coarse_shape=coarse_shape,
            coarse_norm=float(coarse_norm),
            setup_s=float(context.elapsed_s()) - start_s,
            rcond=float(rcond),
            damping=float(damping),
            basis_reused_from_seed=bool(basis_reused_from_seed),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=improvement_ratio,
            probe_reduced=bool(probe_reduced),
            probe_candidates=probe_candidates,
            selected_index=selected_index,
            stats=stats,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI Galerkin preconditioner disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockQIGalerkinStageResult(
            preconditioner=context.base_preconditioner,
            basis_for_galerkin=basis_for_galerkin,
            built=False,
            used=False,
            reason=reason,
            mode=mode,
            rank=int(rank),
            candidate_count=int(candidate_count),
            coarse_shape=coarse_shape,
            coarse_norm=float(coarse_norm),
            setup_s=float(context.elapsed_s()) - start_s,
            rcond=float(rcond),
            damping=float(damping),
            basis_reused_from_seed=bool(basis_reused_from_seed),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=improvement_ratio,
            probe_reduced=False,
            probe_candidates=probe_candidates,
            selected_index=selected_index,
            stats=stats,
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


def apply_xblock_qi_two_level_stage(
    *,
    context: XBlockQITwoLevelStageContext,
) -> XBlockQITwoLevelStageResult:
    """Build, probe, and optionally install a QI two-level preconditioner."""

    reason = (
        context.two_level_policy.reason
        if context.two_level_policy.reason is not None and not context.two_level_policy.should_build
        else None
    )
    for level, message in context.two_level_policy.messages:
        if context.emit is not None:
            context.emit(int(level), str(message))
    if not bool(context.two_level_policy.should_build):
        return XBlockQITwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            x0_full=context.x0_full,
            basis_for_galerkin=context.basis_for_galerkin,
            built=False,
            used=False,
            reason=reason,
            rank=0,
            candidate_count=0,
            coarse_shape=(0, 0),
            coarse_norm=0.0,
            operator_on_basis_shape=(0, 0),
            operator_on_basis_norm=0.0,
            coarse_solver=None,
            residual_augmented=False,
            rank_before_augmentation=0,
            augmentation_labels=(),
            residual_augment_max_extra=0,
            residual_augment_steps=0,
            residual_augment_include_residuals=False,
            smoothed_load_basis=False,
            smoothed_load_metadata={},
            setup_s=0.0,
            rcond=0.0,
            damping=1.0,
            basis_reused_from_seed=False,
            residual_before=None,
            residual_after=None,
            improvement_ratio=None,
            probe_candidates=[],
            selected_index=None,
            stats={"applies": 0, "local_applies": 0},
        )

    start_s = float(context.elapsed_s())
    policy = context.two_level_policy
    rcond = float(policy.rcond)
    damping = float(policy.damping)
    coarse_solver = policy.coarse_solver
    residual_augment_max_extra = int(policy.residual_augment_max_extra)
    residual_augment_steps = int(policy.residual_augment_steps)
    residual_augment_include_residuals = bool(policy.residual_augment_include_residuals)
    stats = {"applies": 0, "local_applies": 0}
    basis_for_galerkin = context.basis_for_galerkin
    basis_reused_from_seed = basis_for_galerkin is not None
    x0_full = context.x0_full
    built = False
    used = False
    rank = 0
    candidate_count = 0
    coarse_shape = (0, 0)
    coarse_norm = 0.0
    operator_on_basis_shape = (0, 0)
    operator_on_basis_norm = 0.0
    residual_augmented = False
    rank_before_augmentation = 0
    augmentation_labels: tuple[str, ...] = ()
    smoothed_load_basis_used = False
    smoothed_load_metadata: dict[str, object] = {}
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_ratio: float | None = None
    probe_candidates: list[dict[str, object]] = []
    selected_index: int | None = None
    try:
        if basis_for_galerkin is None:
            basis_for_galerkin = context.basis_builder(
                op=context.op,
                active_dof=bool(context.active_dof),
                linear_size=int(context.linear_size),
                max_rank=int(context.seed_policy.max_rank),
                rank_rtol=float(context.seed_policy.rank_rtol),
                include_angular=bool(context.seed_policy.include_angular),
                include_blocks=bool(context.seed_policy.include_blocks),
                basis_kind=context.seed_policy.basis_kind,
                max_candidates=int(context.seed_policy.max_candidates),
                max_angular_mode=int(context.seed_policy.max_angular_mode),
                include_radial=bool(context.seed_policy.include_radial),
                include_radial_angular=bool(context.seed_policy.include_radial_angular),
                include_constraint_moments=bool(context.seed_policy.include_constraint_moments),
                include_schur=bool(context.seed_policy.include_schur),
            )

        def local_smoother(v: jnp.ndarray) -> jnp.ndarray:
            stats["local_applies"] += 1
            return jnp.asarray(
                context.base_preconditioner(jnp.asarray(v, dtype=jnp.float64)),
                dtype=jnp.float64,
            )

        current = (
            jnp.zeros_like(context.xblock_rhs)
            if x0_full is None
            else jnp.asarray(x0_full, dtype=jnp.float64)
        )
        residual_before_vec = context.xblock_rhs - jnp.asarray(
            context.true_matvec_no_count(current),
            dtype=jnp.float64,
        )
        residual_before = float(jnp.linalg.norm(residual_before_vec))
        two_level_basis = basis_for_galerkin
        if bool(policy.smoothed_load_basis):
            smoothed_basis, smoothed_metadata = context.smoothed_load_basis_builder(
                op=context.op,
                rhs=context.rhs,
                base_preconditioner=local_smoother,
                direction_projector=context.direction_projector,
                expected_size=int(context.linear_size),
                include_rhs=bool(policy.smoothed_load_include_rhs),
                fsavg_lmax=int(policy.smoothed_load_fsavg_lmax),
                angular_lmax=int(policy.smoothed_load_angular_lmax),
                max_extra_units=int(policy.smoothed_load_max_extra_units),
                max_directions=int(policy.smoothed_load_max_directions),
                rank_rtol=float(context.seed_policy.rank_rtol),
                max_rank=int(policy.smoothed_load_max_rank),
            )
            smoothed_load_basis_used = True
            smoothed_load_metadata = dict(smoothed_metadata)
            if bool(policy.smoothed_load_basis_combine):
                combined_candidates = jnp.concatenate(
                    [
                        jnp.asarray(smoothed_basis.vectors, dtype=jnp.float64),
                        jnp.asarray(two_level_basis.vectors, dtype=jnp.float64),
                    ],
                    axis=1,
                )
                combined_labels = tuple(smoothed_basis.metadata.accepted_labels) + tuple(
                    two_level_basis.metadata.accepted_labels
                )
                two_level_basis = context.orthonormalizer(
                    combined_candidates,
                    labels=combined_labels,
                    rtol=float(context.seed_policy.rank_rtol),
                    max_rank=int(policy.smoothed_load_max_rank) + int(context.seed_policy.max_rank),
                )
            else:
                two_level_basis = smoothed_basis

        if bool(policy.residual_augment) and int(residual_augment_max_extra) > 0:
            rank_before_augmentation = int(two_level_basis.metadata.rank)
            extra_vectors: list[jnp.ndarray] = []
            extra_labels: list[str] = []

            def add_adaptive_vector(label: str, values: jnp.ndarray) -> None:
                if len(extra_vectors) >= int(residual_augment_max_extra):
                    return
                vec = jnp.asarray(values, dtype=jnp.float64).reshape((-1,))
                if int(vec.shape[0]) != int(two_level_basis.vectors.shape[0]):
                    return
                norm = float(jnp.linalg.norm(vec))
                if not np.isfinite(norm) or norm <= 0.0:
                    return
                extra_vectors.append(vec / jnp.asarray(norm, dtype=vec.dtype))
                extra_labels.append(label)

            adaptive_residual = residual_before_vec
            for adaptive_step in range(int(residual_augment_steps)):
                if len(extra_vectors) >= int(residual_augment_max_extra):
                    break
                adaptive_correction = local_smoother(adaptive_residual)
                add_adaptive_vector(
                    f"adaptive:krylov_local_step_{adaptive_step}",
                    adaptive_correction,
                )
                adaptive_residual = adaptive_residual - jnp.asarray(
                    context.matvec(adaptive_correction),
                    dtype=jnp.float64,
                )
                if bool(residual_augment_include_residuals):
                    add_adaptive_vector(
                        f"adaptive:krylov_remaining_step_{adaptive_step}",
                        adaptive_residual,
                    )
            if len(extra_vectors) < int(residual_augment_max_extra):
                final_local = local_smoother(adaptive_residual)
                add_adaptive_vector(
                    f"adaptive:krylov_local_step_{int(residual_augment_steps)}",
                    final_local,
                )
            if extra_vectors:
                residual_augmented = True
                augmentation_labels = tuple(extra_labels)
                augmented_candidates = jnp.concatenate(
                    [jnp.stack(tuple(extra_vectors), axis=1), jnp.asarray(two_level_basis.vectors)],
                    axis=1,
                )
                augmented_labels = tuple(extra_labels) + tuple(two_level_basis.metadata.accepted_labels)
                two_level_basis = context.orthonormalizer(
                    augmented_candidates,
                    labels=augmented_labels,
                    rtol=float(context.seed_policy.rank_rtol),
                    max_rank=int(context.seed_policy.max_rank) + int(residual_augment_max_extra),
                )

        qi_two_level = context.preconditioner_builder(
            operator=context.matvec,
            local_smoother=local_smoother,
            basis=two_level_basis,
            regularization_rcond=float(rcond) if float(rcond) > 0.0 else 0.0,
            damping=1.0,
            coarse_solver=coarse_solver,
        )
        built = True
        rank = int(qi_two_level.metadata.rank)
        candidate_count = int(two_level_basis.metadata.candidate_count)
        coarse_shape = tuple(int(value) for value in qi_two_level.metadata.coarse_operator_shape)
        coarse_norm = float(qi_two_level.metadata.coarse_operator_norm)
        coarse_solver = str(qi_two_level.metadata.coarse_solver)
        operator_on_basis_shape = tuple(
            int(value) for value in qi_two_level.metadata.operator_on_basis_shape
        )
        operator_on_basis_norm = float(qi_two_level.metadata.operator_on_basis_norm)
        correction = jnp.asarray(qi_two_level.apply(residual_before_vec), dtype=jnp.float64)
        required = float(residual_before) * max(0.0, 1.0 - float(policy.min_improvement))
        best_index: int | None = None
        best_damping: float | None = None
        best_residual = float("inf")
        best_solution = current
        for candidate_index, candidate_damping in enumerate(policy.candidate_dampings):
            probe_solution = current + float(candidate_damping) * correction
            probe_residual = context.xblock_rhs - jnp.asarray(
                context.true_matvec_no_count(probe_solution),
                dtype=jnp.float64,
            )
            candidate_residual = float(jnp.linalg.norm(probe_residual))
            ratio_after = (
                candidate_residual / float(residual_before)
                if float(residual_before) > 0.0
                else None
            )
            reduced = bool(np.isfinite(candidate_residual) and candidate_residual < float(residual_before))
            probe_candidates.append(
                {
                    "damping": float(candidate_damping),
                    "residual_norm": float(candidate_residual),
                    "improvement_ratio": ratio_after,
                    "reduced": reduced,
                }
            )
            if np.isfinite(candidate_residual) and candidate_residual < best_residual:
                best_index = int(candidate_index)
                best_damping = float(candidate_damping)
                best_residual = float(candidate_residual)
                best_solution = probe_solution
        selected_index = best_index
        residual_after = float(best_residual)
        improvement_ratio = (
            float(best_residual) / float(residual_before)
            if float(residual_before) > 0.0
            else None
        )
        reason = (
            "residual_reduced"
            if np.isfinite(float(best_residual)) and float(best_residual) < float(required)
            else "residual_not_reduced"
        )

        if reason == "residual_reduced":
            damping = float(best_damping)
            x0_full = jnp.asarray(best_solution, dtype=jnp.float64)

            def preconditioner(v: jnp.ndarray) -> jnp.ndarray:
                stats["applies"] += 1
                return float(damping) * jnp.asarray(
                    qi_two_level.apply(jnp.asarray(v, dtype=jnp.float64)),
                    dtype=jnp.float64,
                )

            used = True
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI two-level preconditioner accepted "
                    f"residual {float(residual_before):.6e} -> {float(residual_after):.6e} "
                    f"(rank={int(rank)} damping={float(damping):.3e} "
                    f"ratio={float(improvement_ratio):.6e})",
                )
            selected_preconditioner = preconditioner
        else:
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    f"QI two-level preconditioner rejected reason={reason} "
                    f"residual={float(residual_before):.6e}",
                )
            selected_preconditioner = context.base_preconditioner
        return XBlockQITwoLevelStageResult(
            preconditioner=selected_preconditioner,
            x0_full=x0_full,
            basis_for_galerkin=basis_for_galerkin,
            built=bool(built),
            used=bool(used),
            reason=reason,
            rank=int(rank),
            candidate_count=int(candidate_count),
            coarse_shape=coarse_shape,
            coarse_norm=float(coarse_norm),
            operator_on_basis_shape=operator_on_basis_shape,
            operator_on_basis_norm=float(operator_on_basis_norm),
            coarse_solver=coarse_solver,
            residual_augmented=bool(residual_augmented),
            rank_before_augmentation=int(rank_before_augmentation),
            augmentation_labels=augmentation_labels,
            residual_augment_max_extra=int(residual_augment_max_extra),
            residual_augment_steps=int(residual_augment_steps),
            residual_augment_include_residuals=bool(residual_augment_include_residuals),
            smoothed_load_basis=bool(smoothed_load_basis_used),
            smoothed_load_metadata=smoothed_load_metadata,
            setup_s=float(context.elapsed_s()) - start_s,
            rcond=float(rcond),
            damping=float(damping),
            basis_reused_from_seed=bool(basis_reused_from_seed),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=improvement_ratio,
            probe_candidates=probe_candidates,
            selected_index=selected_index,
            stats=stats,
        )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI two-level preconditioner disabled after build failure ({type(exc).__name__}: {exc})",
            )
        return XBlockQITwoLevelStageResult(
            preconditioner=context.base_preconditioner,
            x0_full=x0_full,
            basis_for_galerkin=basis_for_galerkin,
            built=False,
            used=False,
            reason=reason,
            rank=int(rank),
            candidate_count=int(candidate_count),
            coarse_shape=coarse_shape,
            coarse_norm=float(coarse_norm),
            operator_on_basis_shape=operator_on_basis_shape,
            operator_on_basis_norm=float(operator_on_basis_norm),
            coarse_solver=coarse_solver,
            residual_augmented=bool(residual_augmented),
            rank_before_augmentation=int(rank_before_augmentation),
            augmentation_labels=augmentation_labels,
            residual_augment_max_extra=int(residual_augment_max_extra),
            residual_augment_steps=int(residual_augment_steps),
            residual_augment_include_residuals=bool(residual_augment_include_residuals),
            smoothed_load_basis=bool(smoothed_load_basis_used),
            smoothed_load_metadata=smoothed_load_metadata,
            setup_s=float(context.elapsed_s()) - start_s,
            rcond=float(rcond),
            damping=float(damping),
            basis_reused_from_seed=bool(basis_reused_from_seed),
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=improvement_ratio,
            probe_candidates=probe_candidates,
            selected_index=selected_index,
            stats=stats,
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


def finalize_sparse_pc_gmres_bundle(
    context: SparsePCGMRESFinalizationBundleContext,
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build typed sparse-PC final metadata, apply retry/polish, and return payload."""

    diagnostic_state = sparse_pc_gmres_finalization_state_from_context(
        SparsePCGMRESFinalizationStateContext(
            atol=context.atol,
            mv_count=context.mv_count,
            rhs_norm=context.rhs_norm,
            target=context.target,
            tol=context.tol,
            sparse_pc_direct_tail_metadata=sparse_pc_direct_tail_final_metadata(
                context.direct_tail
            ),
            sparse_pc_factor_preflight_metadata=(
                sparse_pc_factor_preflight_result_metadata_from_context(
                    context.factor_preflight
                )
            ),
            sparse_pc_pattern_metadata=sparse_pc_pattern_result_metadata_from_context(
                context.pattern
            ),
            sparse_pc_static_metadata=sparse_pc_gmres_static_metadata_from_context(
                context.static
            ),
        )
    )
    result = context.result
    return finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=diagnostic_state,
            result=SparsePCGMRESResult(
                x=np.asarray(result.x, dtype=np.float64),
                residual_norm=float(result.residual_norm),
                preconditioned_residual_norm=float(
                    result.preconditioned_residual_norm
                ),
                history=tuple(float(v) for v in (result.history or ())),
                solve_s=float(result.solve_s),
            ),
            factor_dtype_used=np.dtype(result.factor_dtype_used),
            factor_dtype_retry=result.factor_dtype_retry,
            operator_bundle=result.operator_bundle,
            factor_bundle=result.factor_bundle,
            pc_factor_s=float(result.pc_factor_s),
            setup_s=float(result.setup_s),
            post_minres=context.post_minres,
            dtype_retry=context.dtype_retry,
        ),
        build_host_sparse_direct_factor_from_matvec=(
            build_host_sparse_direct_factor_from_matvec
        ),
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
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


def finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state(
    state: Mapping[str, object],
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Retry factor dtype if needed, then build the final sparse-PC payload."""

    return finalize_sparse_pc_gmres_with_dtype_retry(
        SparsePCGMRESFinalizationContext(
            diagnostic_state=state,
            result=SparsePCGMRESResult(
                x=np.asarray(state["x_np"], dtype=np.float64),
                residual_norm=float(state["residual_norm_sparse_pc"]),
                preconditioned_residual_norm=float(state["rn_pc"]),
                history=tuple(float(v) for v in (state["history"] or ())),
                solve_s=float(state["solve_s"]),
            ),
            factor_dtype_used=np.dtype(state["sparse_pc_factor_dtype_used"]),
            factor_dtype_retry=state["sparse_pc_factor_dtype_retry"],
            operator_bundle=state["_operator_bundle_pc"],
            factor_bundle=state["factor_bundle_pc"],
            pc_factor_s=float(state["pc_factor_s"]),
            setup_s=float(state["setup_s"]) if "setup_s" in state else None,
        ),
        build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
        run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
    )


def finalize_sparse_pc_gmres_with_dtype_retry(
    context: SparsePCGMRESFinalizationContext,
    *,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[Any, Any]],
    run_sparse_pc_gmres_once_callback: Callable[..., tuple[np.ndarray, float, float, Sequence[float], float]],
    minres_correction: Callable[..., tuple[jnp.ndarray, jnp.ndarray, Sequence[float], Sequence[float]]],
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Retry factor dtype if needed from explicit solve state, then finalize."""

    initial_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    initial_state.update(
        {
            "sparse_pc_factor_dtype_used": np.dtype(context.factor_dtype_used),
            "sparse_pc_factor_dtype_retry": context.factor_dtype_retry,
            "_operator_bundle_pc": context.operator_bundle,
            "factor_bundle_pc": context.factor_bundle,
            "pc_factor_s": float(context.pc_factor_s),
            "x_np": np.asarray(context.result.x, dtype=np.float64),
            "residual_norm_sparse_pc": float(context.result.residual_norm),
            "rn_pc": float(context.result.preconditioned_residual_norm),
            "history": tuple(float(v) for v in (context.result.history or ())),
            "solve_s": float(context.result.solve_s),
        }
    )
    if context.setup_s is not None:
        initial_state["setup_s"] = float(context.setup_s)
    if context.dtype_retry is None:
        retry_result = retry_sparse_pc_factor_dtype_from_driver_state(
            initial_state,
            build_host_sparse_direct_factor_from_matvec=build_host_sparse_direct_factor_from_matvec,
            run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        )
    else:
        retry_result = retry_sparse_pc_factor_dtype_from_finalization_context(
            context.dtype_retry,
            factor_dtype_used=np.dtype(context.factor_dtype_used),
            factor_dtype_retry=context.factor_dtype_retry,
            residual_norm=float(context.result.residual_norm),
            preconditioned_residual_norm=float(
                context.result.preconditioned_residual_norm
            ),
            history=context.result.history,
            target=float(initial_state["target"]),
            x=np.asarray(context.result.x, dtype=np.float64),
            solve_s=float(context.result.solve_s),
            operator_bundle=context.operator_bundle,
            factor_bundle=context.factor_bundle,
            build_host_sparse_direct_factor_from_matvec=(
                build_host_sparse_direct_factor_from_matvec
            ),
            run_sparse_pc_gmres_once_callback=run_sparse_pc_gmres_once_callback,
        )
    final_state = (
        initial_state.__class__(initial_state)
        if isinstance(initial_state, MutableMapping)
        else dict(initial_state)
    )
    final_state.update(
        {
            "sparse_pc_factor_dtype_used": retry_result.factor_dtype_used,
            "sparse_pc_factor_dtype_retry": retry_result.factor_dtype_retry,
            "_operator_bundle_pc": retry_result.operator_bundle,
            "factor_bundle_pc": retry_result.factor_bundle,
            "pc_factor_s": float(context.pc_factor_s) + float(retry_result.factor_s_increment),
            "x_np": retry_result.x,
            "residual_norm_sparse_pc": float(retry_result.residual_norm),
            "rn_pc": float(retry_result.preconditioned_residual_norm),
            "history": retry_result.history,
            "solve_s": float(retry_result.solve_s),
        }
    )
    if retry_result.setup_s is not None:
        final_state["setup_s"] = float(retry_result.setup_s)
    if context.post_minres is not None:
        post_context = context.post_minres
        post_minres = apply_sparse_pc_post_minres_if_needed(
            SparsePCPostMinresUpdateContext(
                matvec=post_context.matvec,
                rhs=post_context.rhs,
                preconditioner=post_context.preconditioner,
                emit=post_context.emit,
                elapsed_s=post_context.elapsed_s,
                pc_form=str(post_context.pc_form),
                steps=int(post_context.steps),
                alpha_clip=float(post_context.alpha_clip),
                min_improvement=float(post_context.min_improvement),
                minres_correction=minres_correction,
                x=np.asarray(retry_result.x, dtype=np.float64),
                residual_norm=float(retry_result.residual_norm),
                preconditioned_residual_norm=float(
                    retry_result.preconditioned_residual_norm
                ),
                solve_s=float(retry_result.solve_s),
                target=float(post_context.target),
            )
        )
        final_state.update(
            {
                "x_np": post_minres.x,
                "residual_norm_sparse_pc": float(post_minres.residual_norm),
                "rn_pc": float(post_minres.preconditioned_residual_norm),
                "sparse_pc_post_minres_steps": int(post_context.steps),
                "sparse_pc_post_minres_alpha_clip": float(post_context.alpha_clip),
                "sparse_pc_post_minres_min_improvement": float(
                    post_context.min_improvement
                ),
                "sparse_pc_post_minres_history": post_minres.history,
                "sparse_pc_post_minres_alphas": post_minres.alphas,
                "sparse_pc_post_minres_residual_before": (
                    post_minres.residual_before
                ),
                "sparse_pc_post_minres_residual_after": post_minres.residual_after,
                "sparse_pc_post_minres_error": post_minres.error,
                "solve_s": float(post_minres.solve_s),
                "sparse_pc_elapsed_s": float(post_context.elapsed_s()),
            }
        )
        if post_context.emit is not None:
            post_context.emit(
                0,
                sparse_pc_gmres_completion_message(
                    SparsePCGMRESCompletionMessageContext(
                        elapsed_s=float(final_state["sparse_pc_elapsed_s"]),
                        iterations=int(len(final_state["history"] or ())),
                        matvecs=int(final_state["mv_count"]),
                        residual_norm=float(final_state["residual_norm_sparse_pc"]),
                        target=float(final_state["target"]),
                        preconditioned_residual_norm=float(final_state["rn_pc"]),
                        history=final_state["history"],
                    )
                ),
            )
        return sparse_pc_gmres_final_payload_from_driver_state(
            final_state,
            expand_reduced=expand_reduced,
        )
    return finalize_sparse_pc_gmres_from_driver_state(
        final_state,
        minres_correction=minres_correction,
        expand_reduced=expand_reduced,
    )


def fortran_reduced_xblock_final_payload_from_driver_state(
    state: Mapping[str, object],
    *,
    result: SparsePCGMRESResult,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the fortran-reduced x-block branch from state."""

    return fortran_reduced_xblock_final_payload(
        FortranReducedXBlockFinalPayloadContext(
            diagnostic_state=state,
            result=result,
            atol=float(state["atol"]),
            tol=float(state["tol"]),
            rhs_norm=float(state["rhs_norm"]),
            target=float(state["target"]),
        ),
        expand_reduced=expand_reduced,
    )


def fortran_reduced_xblock_final_payload(
    context: FortranReducedXBlockFinalPayloadContext,
    *,
    expand_reduced: ArrayFn,
) -> SparsePCGMRESFinalPayload:
    """Build the final payload for the fortran-reduced x-block sparse-PC branch.

    The x-block branch has its own metadata schema, but its final convergence
    gates are the same true-residual gates used by the generic sparse-PC path.
    Keeping that acceptance bookkeeping here avoids duplicating target logic in
    the driver while preserving the historical metadata keys.
    """

    result = context.result
    residual_norm = float(result.residual_norm)
    metadata_state = (
        context.diagnostic_state.__class__(context.diagnostic_state)
        if isinstance(context.diagnostic_state, MutableMapping)
        else dict(context.diagnostic_state)
    )
    metadata_state.update(
        {
            "x_np": np.asarray(result.x, dtype=np.float64),
            "residual_norm_sparse_pc": residual_norm,
            "history": tuple(result.history),
            "solve_s": float(result.solve_s),
            "fortran_reduced_xblock_accepted_converged": profile_residual_converged(
                residual_norm,
                profile_residual_target(
                    atol=float(context.atol),
                    tol=float(context.tol),
                    rhs_norm=float(context.rhs_norm),
                ),
            ),
            "fortran_reduced_xblock_factor_quality_rejected": not profile_residual_converged(
                residual_norm,
                float(context.target),
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


def sparse_minimum_norm_solve_from_pattern(
    *,
    matvec_np: Callable[[np.ndarray], np.ndarray],
    pattern: object,
    summary: object,
    rhs: jnp.ndarray,
    solve_method_kind: str,
    tol: float,
    atol: float,
    maxiter: int | None,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    backend: str,
    env: Mapping[str, str],
    emit: EmitFn | None,
    build_operator_from_pattern: Callable[..., object],
) -> SparseMinimumNormPayload:
    """Materialize the explicit sparse matrix and run the host minimum-norm solve."""

    if emit is not None:
        for level, message in explicit_sparse_pattern_progress_messages(
            solver_label="sparse_lsmr",
            summary=summary,
        ):
            emit(level, message)
    sparse_operator_build = build_explicit_sparse_operator_from_pattern(
        matvec_np=matvec_np,
        pattern=pattern,
        dtype=np.float64,
        backend=backend,
        env=env,
        build_operator_from_pattern=build_operator_from_pattern,
        allow_operator_only=False,
    )
    if emit is not None:
        for level, message in sparse_operator_build.messages:
            emit(level, message)
    matrix = sparse_operator_build.operator_bundle.matrix
    if matrix is None:
        raise RuntimeError("sparse_lsmr requires a materialized sparse matrix.")

    policy = resolve_sparse_minimum_norm_policy(
        env,
        solve_method_kind=solve_method_kind,
        tol=float(tol),
        maxiter=maxiter,
        emit_enabled=emit is not None,
    )
    if emit is not None:
        emit(0, sparse_minimum_norm_start_message(policy))
    payload = sparse_minimum_norm_solve_payload(
        matrix=matrix,
        rhs=rhs,
        policy=policy,
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
        elapsed_s=elapsed_s,
    )
    if emit is not None:
        emit(0, payload.completion_message)
    return payload


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


def sparse_host_direct_solve_from_pattern(
    *,
    matvec: Callable[[np.ndarray], jnp.ndarray],
    pattern: object,
    summary: object,
    n: int,
    dtype: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    atol: float,
    tol: float,
    rhs_norm: float,
    elapsed_s: Callable[[], float],
    emit: EmitFn | None,
    build_host_sparse_direct_factor_from_matvec: Callable[..., tuple[object, object]],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectPayload:
    """Build an explicit host sparse factor and solve the full RHSMode=1 system."""

    if emit is not None:
        for level, message in explicit_sparse_pattern_progress_messages(
            solver_label="sparse_host",
            summary=summary,
        ):
            emit(level, message)
    operator_bundle, factor_bundle = build_host_sparse_direct_factor_from_matvec(
        matvec=matvec,
        n=int(n),
        dtype=dtype,
        factor_dtype=factor_dtype,
        pattern=pattern,
        emit=emit,
    )
    payload = sparse_host_direct_solve_payload(
        factor_solve=factor_bundle.solve,
        operator_matrix=operator_bundle.matrix,
        rhs=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
        matvec=matvec,
        atol=float(atol),
        tol=float(tol),
        rhs_norm=float(rhs_norm),
        elapsed_s=elapsed_s,
        direct_solve_with_refinement=direct_solve_with_refinement,
    )
    if emit is not None:
        emit(0, payload.completion_message)
    return payload


def _elapsed_since_now() -> Callable[[], float]:
    """Return a cheap elapsed-time callback for explicit host sparse branches."""

    start_s = perf_counter()
    return lambda: perf_counter() - start_s


def solve_explicit_sparse_minimum_norm_branch(
    context: ExplicitSparseMinimumNormBranchContext,
) -> SparseMinimumNormPayload:
    """Run the explicit sparse LSQR/LSMR branch from driver-provided callbacks."""

    validate_explicit_sparse_host_request(
        solve_method_label="sparse_lsmr",
        differentiable=context.differentiable,
        rhs_mode=int(context.op.rhs_mode),
        use_active_dof=bool(context.use_active_dof),
        path_description="host sparse minimum-norm path",
    )
    pattern = context.build_pattern(context.op)
    summary = context.summarize_pattern(context.op, pattern)
    rhs_dtype = context.rhs.dtype

    def matvec_np(x_np: np.ndarray) -> np.ndarray:
        x_device = jnp.asarray(np.asarray(x_np, dtype=np.float64), dtype=rhs_dtype)
        return np.asarray(
            context.apply_cached_operator(context.op, x_device),
            dtype=np.float64,
        )

    return sparse_minimum_norm_solve_from_pattern(
        matvec_np=matvec_np,
        pattern=pattern,
        summary=summary,
        rhs=context.rhs,
        solve_method_kind=context.solve_method_kind,
        tol=float(context.tol),
        atol=float(context.atol),
        maxiter=context.maxiter,
        rhs_norm=float(context.rhs_norm),
        elapsed_s=_elapsed_since_now(),
        backend=str(context.backend),
        env=context.env,
        emit=context.emit,
        build_operator_from_pattern=context.build_operator_from_pattern,
    )


def solve_explicit_sparse_host_direct_branch(
    context: ExplicitSparseHostDirectBranchContext,
) -> SparseHostDirectPayload:
    """Run the explicit sparse host-LU branch from driver-provided callbacks."""

    validate_explicit_sparse_host_request(
        solve_method_label="sparse_host",
        differentiable=context.differentiable,
        rhs_mode=int(context.op.rhs_mode),
        use_active_dof=bool(context.use_active_dof),
        path_description="host sparse LU path",
    )
    pattern = context.build_pattern(context.op)
    summary = context.summarize_pattern(context.op, pattern)
    rhs_dtype = context.rhs.dtype

    def matvec(x_np: np.ndarray) -> jnp.ndarray:
        x_device = jnp.asarray(x_np, dtype=rhs_dtype)
        return context.apply_operator(context.op, x_device)

    return sparse_host_direct_solve_from_pattern(
        matvec=matvec,
        pattern=pattern,
        summary=summary,
        n=int(context.op.total_size),
        dtype=rhs_dtype,
        factor_dtype=np.dtype(np.float64),
        rhs=context.rhs,
        refine_steps=int(context.refine_steps),
        atol=float(context.atol),
        tol=float(context.tol),
        rhs_norm=float(context.rhs_norm),
        elapsed_s=_elapsed_since_now(),
        emit=context.emit,
        build_host_sparse_direct_factor_from_matvec=(
            context.build_host_sparse_direct_factor_from_matvec
        ),
        direct_solve_with_refinement=context.direct_solve_with_refinement,
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


def sparse_host_direct_fallback_payload(
    *,
    explicit_sparse_factor: object | None,
    explicit_sparse_operator: object | None,
    ilu: object,
    a_csr_full: object,
    rhs: jnp.ndarray,
    factor_dtype: np.dtype,
    refine_steps: int,
    matvec: ArrayFn,
    target: float,
    tol: float,
    atol: float,
    restart: int,
    maxiter: int | None,
    precondition_side: str,
    emit: EmitFn | None,
    backend_name: str | None = None,
    polish_enabled: Callable[..., bool],
    parse_polish_gmres_config: Callable[..., tuple[int, int]],
    direct_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    ilu_solve_with_refinement: Callable[..., tuple[np.ndarray, float]],
    host_sparse_direct_polish: Callable[..., tuple[np.ndarray, float]],
) -> SparseHostDirectFallbackPayload:
    """Run a host sparse direct fallback, optional polish, and true residual check."""

    if emit is not None and backend_name is not None:
        emit(
            0,
            "solve_v3_full_system_linear_gmres: host sparse LU direct fallback "
            f"on backend={backend_name}",
        )
    factor_payload = solve_sparse_host_direct_from_available_factor(
        explicit_sparse_factor=explicit_sparse_factor,
        explicit_sparse_operator=explicit_sparse_operator,
        ilu=ilu,
        a_csr_full=a_csr_full,
        rhs=rhs,
        factor_dtype=factor_dtype,
        refine_steps=int(refine_steps),
        direct_solve_with_refinement=direct_solve_with_refinement,
        ilu_solve_with_refinement=ilu_solve_with_refinement,
    )
    polish_payload = apply_sparse_host_direct_polish_if_needed(
        x=factor_payload.x,
        residual_norm=float(factor_payload.residual_norm),
        factor_dtype=factor_dtype,
        target=float(target),
        matvec=matvec,
        rhs=rhs,
        ilu=ilu,
        tol=float(tol),
        atol=float(atol),
        restart=int(restart),
        maxiter=maxiter,
        precondition_side=precondition_side,
        emit=emit,
        polish_enabled=polish_enabled,
        parse_polish_gmres_config=parse_polish_gmres_config,
        host_sparse_direct_polish=host_sparse_direct_polish,
    )
    residual_vec = jnp.asarray(rhs, dtype=jnp.float64) - matvec(polish_payload.x)
    return SparseHostDirectFallbackPayload(
        x=polish_payload.x,
        residual_norm=polish_payload.residual_norm,
        residual_vec=residual_vec,
        used_explicit_factor=bool(factor_payload.used_explicit_factor),
        polish_attempted=bool(polish_payload.attempted),
        polish_accepted=bool(polish_payload.accepted),
        polish_restart=polish_payload.restart,
        polish_maxiter=polish_payload.maxiter,
    )


def build_sparse_host_or_ilu_factor(
    context: SparseHostOrILUFactorBuildContext,
) -> SparseHostOrILUFactorBuildResult:
    """Build either an explicit host sparse direct factor or the ILU fallback."""

    if bool(context.host_sparse_direct_wanted) and bool(context.explicit_sparse_allowed):
        if context.build_host_sparse_direct_factor_from_matvec is None:
            raise ValueError("explicit sparse host factor requested without a build callback")
        explicit_sparse_operator, explicit_sparse_factor = (
            context.build_host_sparse_direct_factor_from_matvec(
                matvec=context.matvec,
                n=int(context.n),
                dtype=context.dtype,
                factor_dtype=context.factor_dtype,
                pattern=context.explicit_sparse_pattern,
                emit=context.emit,
            )
        )
        return SparseHostOrILUFactorBuildResult(
            explicit_sparse_operator=explicit_sparse_operator,
            explicit_sparse_factor=explicit_sparse_factor,
            a_csr_full=explicit_sparse_operator.matrix,
            a_csr_drop=explicit_sparse_operator.matrix,
            ilu=explicit_sparse_factor.factor,
            a_dense_cache=None,
            l_dense=None,
            u_dense=None,
            l_unit_diag=False,
            used_explicit_sparse=True,
        )

    if context.build_sparse_ilu_from_matvec is None:
        raise ValueError("ILU factor requested without a build callback")
    a_csr_full, a_csr_drop, ilu, a_dense_cache, l_dense, u_dense, l_unit_diag = (
        context.build_sparse_ilu_from_matvec(
            matvec=context.matvec,
            n=int(context.n),
            dtype=context.dtype,
            cache_key=context.cache_key,
            factor_dtype=context.factor_dtype,
            drop_tol=float(context.drop_tol),
            drop_rel=float(context.drop_rel),
            ilu_drop_tol=float(context.ilu_drop_tol),
            fill_factor=float(context.fill_factor),
            build_dense_factors=bool(context.build_dense_factors),
            build_jax_factors=bool(context.build_jax_factors),
            build_ilu=True,
            store_dense=bool(context.store_dense),
            factorization=str(context.factorization),
            emit=context.emit,
        )
    )
    return SparseHostOrILUFactorBuildResult(
        explicit_sparse_operator=None,
        explicit_sparse_factor=None,
        a_csr_full=a_csr_full,
        a_csr_drop=a_csr_drop,
        ilu=ilu,
        a_dense_cache=a_dense_cache,
        l_dense=l_dense,
        u_dense=u_dense,
        l_unit_diag=bool(l_unit_diag),
        used_explicit_sparse=False,
    )


def resolve_sparse_host_or_ilu_factor_controls(
    *,
    n: int,
    cache_key: object,
    sparse_exact_lu: bool,
    use_implicit: bool,
    force_host_sparse_direct: bool,
    sparse_ilu_dense_max: int,
    sparse_dense_cache_max: int,
    host_sparse_direct_wanted: bool | None = None,
    host_sparse_direct_allowed: Callable[..., bool],
    host_sparse_factor_dtype: Callable[..., np.dtype],
    sparse_factor_cache_key: Callable[..., object],
    explicit_sparse_host_direct_allowed: Callable[..., bool],
) -> SparseHostOrILUFactorControls:
    """Resolve host sparse direct/ILU build controls shared by reduced/full paths."""

    direct_wanted = (
        bool(host_sparse_direct_wanted)
        if host_sparse_direct_wanted is not None
        else bool(
            host_sparse_direct_allowed(
                sparse_exact_lu=bool(sparse_exact_lu),
                use_implicit=bool(use_implicit),
            )
        )
    )
    if bool(force_host_sparse_direct) and bool(sparse_exact_lu):
        direct_wanted = True
    factorization = "lu" if bool(sparse_exact_lu) else "ilu"
    factor_dtype = (
        host_sparse_factor_dtype(
            size=int(n),
            factorization=factorization,
            use_implicit=bool(use_implicit),
        )
        if direct_wanted
        else np.dtype(np.float64)
    )
    cache_key_use = sparse_factor_cache_key(cache_key, factor_dtype) if direct_wanted else cache_key
    build_dense_factors = bool(use_implicit) and (not direct_wanted) and int(n) <= int(sparse_ilu_dense_max)
    build_jax_factors = bool(use_implicit) and (not direct_wanted)
    store_dense = int(n) <= int(sparse_dense_cache_max)
    explicit_sparse_allowed = direct_wanted and bool(
        explicit_sparse_host_direct_allowed(
            sparse_exact_lu=bool(sparse_exact_lu),
            use_implicit=bool(use_implicit),
            active_size=int(n),
        )
    )
    return SparseHostOrILUFactorControls(
        host_sparse_direct_wanted=bool(direct_wanted),
        factor_dtype=np.dtype(factor_dtype),
        cache_key_use=cache_key_use,
        build_dense_factors=bool(build_dense_factors),
        build_jax_factors=bool(build_jax_factors),
        store_dense=bool(store_dense),
        explicit_sparse_allowed=bool(explicit_sparse_allowed),
    )


def build_sparse_ilu_preconditioner_from_cache(
    context: SparseILUPreconditionerBuildContext,
) -> SparseILUPreconditionerBuildResult:
    """Build a JAX ILU preconditioner from cached permutations and factors."""

    cache_entry = context.cache_entry
    perm_r = None if cache_entry is None else getattr(cache_entry, "perm_r", None)
    inv_perm_c = (
        None if cache_entry is None else getattr(cache_entry, "inv_perm_c", None)
    )
    lower_idx = None if cache_entry is None else getattr(cache_entry, "lower_idx", None)
    lower_val = None if cache_entry is None else getattr(cache_entry, "lower_val", None)
    lower_diag = None if cache_entry is None else getattr(cache_entry, "lower_diag", None)
    upper_idx = None if cache_entry is None else getattr(cache_entry, "upper_idx", None)
    upper_val = None if cache_entry is None else getattr(cache_entry, "upper_val", None)
    upper_diag = None if cache_entry is None else getattr(cache_entry, "upper_diag", None)

    if (
        context.l_dense is not None
        and context.u_dense is not None
        and perm_r is not None
        and inv_perm_c is not None
    ):
        import jax.scipy.linalg as jla  # noqa: PLC0415

        l_jnp = jnp.asarray(context.l_dense, dtype=jnp.float64)
        u_jnp = jnp.asarray(context.u_dense, dtype=jnp.float64)

        def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            v = jnp.asarray(v, dtype=jnp.float64)
            v_perm = v[perm_r]
            y = jla.solve_triangular(
                l_jnp,
                v_perm,
                lower=True,
                unit_diagonal=bool(context.l_unit_diag),
            )
            z = jla.solve_triangular(u_jnp, y, lower=False)
            return z[inv_perm_c]

        return SparseILUPreconditionerBuildResult(
            preconditioner=_preconditioner,
            used_dense_triangular=True,
            used_padded_triangular=False,
        )

    if (
        perm_r is not None
        and inv_perm_c is not None
        and lower_idx is not None
        and lower_val is not None
        and (lower_diag is not None or not bool(context.require_lower_diag))
        and upper_idx is not None
        and upper_val is not None
        and upper_diag is not None
    ):

        def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
            v = jnp.asarray(v, dtype=jnp.float64)
            v_perm = v[perm_r]
            y = triangular_solve_lower_padded(
                lower_idx=lower_idx,
                lower_val=lower_val,
                b=v_perm,
            )
            z = triangular_solve_upper_padded(
                upper_idx=upper_idx,
                upper_val=upper_val,
                upper_diag=upper_diag,
                b=y,
            )
            return z[inv_perm_c]

        return SparseILUPreconditionerBuildResult(
            preconditioner=_preconditioner,
            used_dense_triangular=False,
            used_padded_triangular=True,
        )

    return SparseILUPreconditionerBuildResult(
        preconditioner=None,
        used_dense_triangular=False,
        used_padded_triangular=False,
    )


def build_sparse_host_scipy_preconditioner(
    context: SparseHostScipyPreconditionerBuildContext,
) -> SparseHostScipyPreconditionerBuildResult:
    """Build host callbacks for SciPy Krylov sparse fallback solves."""

    if context.ilu is None:
        raise RuntimeError(str(context.unavailable_message))

    def _preconditioner(v: jnp.ndarray) -> jnp.ndarray:
        x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
        y_np = context.ilu.solve(x_np)
        return jnp.asarray(y_np, dtype=jnp.float64)

    if bool(context.sparse_use_matvec):

        def _matvec(v: jnp.ndarray) -> jnp.ndarray:
            x_np = np.asarray(v, dtype=np.float64).reshape((-1,))
            y_np = context.a_csr_full @ x_np
            return jnp.asarray(y_np, dtype=jnp.float64)

    else:
        _matvec = context.base_matvec

    return SparseHostScipyPreconditionerBuildResult(
        preconditioner=_preconditioner,
        matvec=_matvec,
    )


def run_sparse_host_scipy_gmres(
    context: SparseHostScipyGMRESContext,
) -> tuple[GMRESSolveResult, jnp.ndarray | None]:
    """Run host SciPy GMRES and wrap the result for RHSMode=1 retry gates."""

    x_np, residual_norm, _history = context.gmres_solver(
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
    result = GMRESSolveResult(
        x=jnp.asarray(x_np, dtype=jnp.float64),
        residual_norm=jnp.asarray(residual_norm, dtype=jnp.float64),
    )
    residual_vec = None
    if context.residual_matvec is not None:
        residual_vec = jnp.asarray(context.rhs, dtype=jnp.float64) - jnp.asarray(
            context.residual_matvec(result.x),
            dtype=jnp.float64,
        )
    return result, residual_vec


def run_sparse_host_retry_candidate(
    context: SparseHostRetryCandidateContext,
) -> SparseHostRetryCandidateResult:
    """Run one sparse-host retry candidate from already-built factors."""

    start_s = perf_counter()
    factor_build = context.factor_build
    matvec_for_accept = context.matvec
    preconditioner_for_accept: ArrayFn | None = None
    result: GMRESSolveResult | None = None
    residual_vec: jnp.ndarray | None = None
    host_sparse_direct_used = False
    label = "sparse LU" if bool(context.sparse_exact_lu) else "sparse ILU"

    if bool(context.host_sparse_direct) and factor_build.ilu is not None:
        if bool(context.host_direct_operator_pc):
            scipy_sparse_build = build_sparse_host_scipy_preconditioner(
                SparseHostScipyPreconditionerBuildContext(
                    ilu=factor_build.ilu,
                    a_csr_full=factor_build.a_csr_full,
                    base_matvec=context.matvec,
                    sparse_use_matvec=False,
                )
            )
            preconditioner_for_accept = scipy_sparse_build.preconditioner
            restart = (
                int(context.operator_pc_restart)
                if context.operator_pc_restart is not None
                else int(context.restart)
            )
            maxiter = (
                int(context.operator_pc_maxiter)
                if context.operator_pc_maxiter is not None
                else context.maxiter
            )
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: sparse LU operator-preconditioned "
                    f"GMRES fallback restart={int(restart)} maxiter={int(maxiter or 0)}",
                )
            result, residual_vec = run_sparse_host_scipy_gmres(
                SparseHostScipyGMRESContext(
                    matvec=context.matvec,
                    rhs=context.rhs,
                    preconditioner=preconditioner_for_accept,
                    x0=context.x0,
                    tol=float(context.tol),
                    atol=float(context.atol),
                    restart=int(restart),
                    maxiter=maxiter,
                    precondition_side=context.precondition_side,
                    gmres_solver=context.gmres_solver,
                    residual_matvec=context.matvec,
                )
            )
        else:
            host_sparse_direct_used = True
            direct_payload = sparse_host_direct_fallback_payload(
                explicit_sparse_factor=factor_build.explicit_sparse_factor,
                explicit_sparse_operator=factor_build.explicit_sparse_operator,
                ilu=factor_build.ilu,
                a_csr_full=factor_build.a_csr_full,
                rhs=context.rhs,
                factor_dtype=context.factor_dtype,
                refine_steps=int(context.refine_steps),
                matvec=context.matvec,
                target=float(context.target),
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                precondition_side=context.precondition_side,
                emit=context.emit,
                backend_name=context.backend_name,
                polish_enabled=context.polish_enabled,
                parse_polish_gmres_config=context.parse_polish_gmres_config,
                direct_solve_with_refinement=context.direct_solve_with_refinement,
                ilu_solve_with_refinement=context.ilu_solve_with_refinement,
                host_sparse_direct_polish=context.host_sparse_direct_polish,
            )
            result = GMRESSolveResult(
                x=direct_payload.x,
                residual_norm=direct_payload.residual_norm,
            )
            residual_vec = direct_payload.residual_vec
    elif bool(context.use_implicit):
        precond_build = build_sparse_ilu_preconditioner_from_cache(
            SparseILUPreconditionerBuildContext(
                cache_entry=context.cache_entry,
                l_dense=factor_build.l_dense,
                u_dense=factor_build.u_dense,
                l_unit_diag=factor_build.l_unit_diag,
                require_lower_diag=bool(context.require_lower_diag),
            )
        )
        preconditioner_for_accept = precond_build.preconditioner
        if preconditioner_for_accept is None:
            if context.emit is not None:
                context.emit(
                    1,
                    f"{'sparse_lu' if context.sparse_exact_lu else 'sparse_ilu'}: "
                    "implicit preconditioner factors unavailable; skipping",
                )
        else:
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: "
                    f"{label} (implicit) fallback",
                )
            result, residual_vec = context.implicit_solver(preconditioner_for_accept)
    else:
        scipy_sparse_build = build_sparse_host_scipy_preconditioner(
            SparseHostScipyPreconditionerBuildContext(
                ilu=factor_build.ilu,
                a_csr_full=factor_build.a_csr_full,
                base_matvec=context.matvec,
                sparse_use_matvec=bool(context.sparse_use_matvec),
            )
        )
        preconditioner_for_accept = scipy_sparse_build.preconditioner
        matvec_for_accept = scipy_sparse_build.matvec
        if context.emit is not None:
            context.emit(
                0,
                "solve_v3_full_system_linear_gmres: "
                f"{label} GMRES fallback",
            )
        result, residual_vec = run_sparse_host_scipy_gmres(
            SparseHostScipyGMRESContext(
                matvec=matvec_for_accept,
                rhs=context.rhs,
                preconditioner=preconditioner_for_accept,
                x0=context.x0,
                tol=float(context.tol),
                atol=float(context.atol),
                restart=int(context.restart),
                maxiter=context.maxiter,
                precondition_side=context.precondition_side,
                gmres_solver=context.gmres_solver,
                residual_matvec=(
                    matvec_for_accept
                    if bool(context.compute_scipy_residual_vec)
                    else None
                ),
            )
        )

    return SparseHostRetryCandidateResult(
        result=result,
        residual_vec=residual_vec,
        matvec=matvec_for_accept,
        preconditioner=preconditioner_for_accept,
        solve_s=perf_counter() - start_s,
        host_sparse_direct_used=bool(host_sparse_direct_used),
    )


def build_sparse_jax_retry_preconditioner(
    context: SparseJAXRetryPreconditionerBuildContext,
) -> ArrayFn:
    """Build the sparse-JAX retry preconditioner and emit its progress line."""

    preconditioner = context.builder(
        matvec=context.matvec,
        n=int(context.n),
        dtype=context.dtype,
        cache_key=context.cache_key,
        drop_tol=float(context.drop_tol),
        drop_rel=float(context.drop_rel),
        reg=float(context.reg),
        omega=float(context.omega),
        sweeps=int(context.sweeps),
        emit=context.emit,
    )
    if context.emit is not None:
        context.emit(
            0,
            "solve_v3_full_system_linear_gmres: sparse JAX Jacobi fallback "
            f"(sweeps={int(context.sweeps)} omega={float(context.omega):.2f})",
        )
    return preconditioner


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
                    f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres "
                    f"improved residual {residual_before:.6e} "
                    f"-> {float(residual_after):.6e} "
                    f"(accepted_steps={len(alphas)})",
                )
        elif context.emit is not None:
            after = float(residual_after) if residual_after is not None else float("nan")
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres "
                f"rejected residual {residual_before:.6e} -> {after:.6e}",
            )
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"
        if context.emit is not None:
            context.emit(
                1,
                f"solve_v3_full_system_linear_gmres: {context.solver_label} post-minres failed "
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
            solver_label=str(context.solver_label),
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


__all__ = [
    "FortranReducedSparsePCBackendSetup",
    "FortranReducedXBlockFactorPolicySetup",
    "FortranReducedXBlockFactorBuildContext",
    "FortranReducedXBlockFactorBuildResult",
    "FortranReducedXBlockFinalPayloadContext",
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
    "FPXBlockGlobalCorrectionContext",
    "FPXBlockGlobalCorrectionResult",
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
    "SparsePCGMRESFinalizationContext",
    "SparsePCGMRESFinalizationBundleContext",
    "SparsePCGMRESFinalResultContext",
    "SparsePCGMRESFinalizationStateContext",
    "SparsePCFactorDtypeRetryFinalizationContext",
    "SparsePCPostMinresFinalizationContext",
    "sparse_pc_gmres_finalization_driver_scope_keys",
    "sparse_pc_gmres_finalization_driver_state_keys",
    "sparse_pc_gmres_finalization_bundle_from_driver_scope",
    "sparse_pc_gmres_finalization_bundle_from_driver_result",
    "sparse_pc_gmres_finalization_state_from_context",
    "sparse_pc_gmres_finalization_state_from_driver_scope",
    "XBlockKrylovReport",
    "XBlockSparsePCCompletionContext",
    "XBlockSparsePCFinalPayloadContext",
    "XBlockGMRESFallbackDecision",
    "XBlockGMRESFallbackContext",
    "XBlockGMRESFallbackResult",
    "XBlockDeviceKrylovState",
    "XBlockFirstKrylovAttemptContext",
    "XBlockFirstKrylovAttemptResult",
    "XBlockFirstKrylovSolveStateContext",
    "XBlockKrylovSolveStageContext",
    "XBlockKrylovSolveStageResult",
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
    "XBlockAugmentedKrylovBasisContext",
    "XBlockAugmentedKrylovBasisResult",
    "XBlockAugmentedKrylovStageContext",
    "XBlockAugmentedKrylovStageResult",
    "XBlockKrylovSolveSpace",
    "XBlockKrylovSolveSpaceContext",
    "XBlockSparsePCFinalCoreState",
    "XBlockSparsePCFinalDeviceState",
    "XBlockSparsePCFinalPreflightState",
    "XBlockSparsePCFinalNestedMetadata",
    "XBlockSparsePCFinalMetadataStateContext",
    "XBlockSparsePCWorkEstimates",
    "XBlockSparsePCBranchSetup",
    "XBlockLocalPreconditionerBuildResult",
    "XBlockAssembledOperatorBuildResult",
    "XBlockMomentSchurStageContext",
    "XBlockMomentSchurStageResult",
    "XBlockTwoLevelStageContext",
    "XBlockTwoLevelStageResult",
    "XBlockGlobalCouplingStageContext",
    "XBlockGlobalCouplingStageResult",
    "XBlockQICoarseSeedStageContext",
    "XBlockQICoarseSeedStageResult",
    "XBlockQIDeviceMetadataContext",
    "XBlockQIDeviceSetupConfig",
    "XBlockQIDeviceSetupConfigContext",
    "XBlockQIDeflatedPolicySetup",
    "XBlockQIDeflatedStageContext",
    "XBlockQIDeflatedStageResult",
    "XBlockQIGalerkinStageContext",
    "XBlockQIGalerkinStageResult",
    "XBlockQITwoLevelStageContext",
    "XBlockQITwoLevelStageResult",
    "XBlockPhysicalResidual",
    "SparsePCGMRESFinalPayload",
    "SparseMinimumNormPolicy",
    "SparseMinimumNormPayload",
    "SparseHostDirectPayload",
    "SparseHostDirectFactorSolvePayload",
    "SparseHostDirectPolishPayload",
    "SparseHostDirectFallbackPayload",
    "ExplicitSparseMinimumNormBranchContext",
    "ExplicitSparseHostDirectBranchContext",
    "SparseHostOrILUFactorBuildContext",
    "SparseHostOrILUFactorBuildResult",
    "SparseHostOrILUFactorControls",
    "SparseILUPreconditionerBuildContext",
    "SparseILUPreconditionerBuildResult",
    "SparseHostScipyPreconditionerBuildContext",
    "SparseHostScipyPreconditionerBuildResult",
    "SparseHostScipyGMRESContext",
    "SparseHostRetryCandidateContext",
    "SparseHostRetryCandidateResult",
    "SparseJAXRetryPreconditionerBuildContext",
    "SparseXBlockExplicitSeedContext",
    "SparseXBlockExplicitSeedResult",
    "SparseXBlockRescueAcceptanceContext",
    "SparseXBlockRescueAcceptanceResult",
    "SparseXBlockRescueBuildContext",
    "SparseXBlockRescueBuildResult",
    "SparseXBlockRescueSolveContext",
    "SparseXBlockRescueSolveResult",
    "SparsePCDirectTailFinalMetadataContext",
    "ExplicitSparseOperatorBuildPolicy",
    "ExplicitSparseOperatorBuildResult",
    "SparsePCGMRESCompletionMessageContext",
    "SparsePCPostMinresContext",
    "SparsePCPostMinresResult",
    "SparsePCPostMinresUpdateContext",
    "SparsePCPostMinresUpdateResult",
    "XBlockSubspaceCorrectionContext",
    "XBlockSubspaceCorrectionResult",
    "XBlockPostSolveCorrectionContext",
    "XBlockPostSolveCorrectionResult",
    "XBlockPostKrylovCompletionContext",
    "XBlockPostKrylovCompletionResult",
    "accept_sparse_xblock_rescue_candidate",
    "apply_fortran_reduced_xblock_global_coupling_stage",
    "apply_fortran_reduced_xblock_initial_seed",
    "apply_fortran_reduced_xblock_moment_schur_stage",
    "apply_xblock_global_coupling_stage",
    "apply_xblock_moment_schur_stage",
    "apply_xblock_qi_coarse_seed_stage",
    "apply_xblock_qi_deflated_stage",
    "apply_xblock_qi_galerkin_stage",
    "apply_xblock_qi_two_level_stage",
    "apply_xblock_side_probe_stage",
    "apply_xblock_probe_coarse_stage",
    "apply_xblock_augmented_krylov_stage",
    "apply_xblock_two_level_stage",
    "apply_sparse_pc_post_minres",
    "apply_sparse_pc_post_minres_if_needed",
    "apply_sparse_pc_post_minres_from_driver_state",
    "apply_sparse_xblock_explicit_seed",
    "apply_xblock_subspace_correction_if_needed",
    "build_fortran_reduced_xblock_factor_stage",
    "build_sparse_xblock_rescue_preconditioner",
    "build_fortran_reduced_xblock_krylov_setup",
    "build_xblock_local_preconditioner",
    "build_xblock_assembled_operator_if_requested",
    "build_xblock_qi_device_preconditioner_metadata",
    "build_xblock_qi_device_setup_config",
    "build_xblock_krylov_progress_callbacks",
    "resolve_xblock_qi_deflated_policy_setup",
    "resolve_xblock_krylov_control_setup",
    "build_sparse_pc_active_dof_setup",
    "build_sparse_pc_pattern_setup",
    "build_direct_tail_materialization_setup",
    "build_direct_tail_structured_preconditioner_setup",
    "enforce_sparse_pc_memory_budget",
    "emit_xblock_sparse_pc_completion",
    "emit_xblock_sparse_pc_completion_from_driver_state",
    "evaluate_sparse_pc_factor_preflight",
    "evaluate_sparse_pc_residual_candidate_acceptance",
    "evaluate_xblock_preflight_gate",
    "select_sparse_pc_auto_preflight_retry_candidates",
    "evaluate_sparse_pc_auto_preflight_retry",
    "resolve_sparse_pc_gmres_control_policy",
    "resolve_sparse_pc_factor_preflight_policy",
    "resolve_direct_tail_residual_rescue_policy",
    "resolve_direct_tail_true_active_rescue_policy",
    "resolve_direct_tail_coupled_coarse_rescue_policy",
    "run_direct_tail_support_mode_preflight",
    "sparse_pc_direct_tail_final_metadata",
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
    "resolve_xblock_sparse_pc_branch_setup",
    "resolve_sparse_pc_factor_policy",
    "evaluate_sparse_pc_factor_dtype_retry",
    "sparse_pc_factor_dtype_retry_initial_guess",
    "retry_sparse_pc_factor_dtype_if_needed",
    "retry_sparse_pc_factor_dtype_from_driver_state",
    "retry_sparse_pc_factor_dtype_from_finalization_context",
    "run_fortran_reduced_xblock_krylov_solve",
    "run_fp_xblock_global_correction_stage",
    "run_sparse_xblock_rescue_solve_stage",
    "run_sparse_pc_gmres_once",
    "run_sparse_pc_gmres_once_for_retry",
    "finalize_sparse_pc_gmres_bundle",
    "prepare_xblock_augmented_krylov_basis",
    "prepare_xblock_krylov_solve_space",
    "run_xblock_first_krylov_attempt",
    "run_xblock_krylov_solve_stage",
    "run_xblock_gmres_fallback_if_needed",
    "run_xblock_post_solve_corrections",
    "complete_xblock_post_krylov_stage",
    "xblock_device_krylov_state",
    "xblock_device_cycle_progress_message",
    "xblock_host_krylov_progress_message",
    "xblock_sparse_pc_final_metadata_driver_scope_keys",
    "xblock_sparse_pc_final_metadata_driver_state_keys",
    "xblock_sparse_pc_final_metadata_state_from_context",
    "xblock_sparse_pc_final_metadata_state_from_driver_scope",
    "xblock_krylov_state_from_first_attempt",
    "xblock_krylov_state_from_gmres_fallback",
    "xblock_sparse_pc_completion_message",
    "xblock_gmres_fallback_decision",
    "xblock_krylov_report",
    "xblock_physical_solution_and_residual",
    "xblock_sparse_pc_work_estimates",
    "sparse_pc_gmres_completion_message",
    "emit_sparse_pc_gmres_completion_from_driver_state",
    "sparse_pc_gmres_final_payload_from_driver_state",
    "finalize_sparse_pc_gmres_from_driver_state",
    "finalize_sparse_pc_gmres_with_dtype_retry",
    "finalize_sparse_pc_gmres_with_dtype_retry_from_driver_state",
    "fortran_reduced_xblock_final_payload",
    "fortran_reduced_xblock_final_payload_from_driver_state",
    "xblock_sparse_pc_final_payload",
    "xblock_sparse_pc_final_payload_from_driver_state",
    "resolve_sparse_minimum_norm_policy",
    "sparse_minimum_norm_solve_payload",
    "sparse_minimum_norm_solve_from_pattern",
    "sparse_minimum_norm_start_message",
    "solve_explicit_sparse_minimum_norm_branch",
    "sparse_host_direct_solve_payload",
    "sparse_host_direct_solve_from_pattern",
    "solve_explicit_sparse_host_direct_branch",
    "solve_sparse_host_direct_from_available_factor",
    "apply_sparse_host_direct_polish_if_needed",
    "sparse_host_direct_fallback_payload",
    "build_sparse_host_or_ilu_factor",
    "resolve_sparse_host_or_ilu_factor_controls",
    "build_sparse_ilu_preconditioner_from_cache",
    "build_sparse_host_scipy_preconditioner",
    "run_sparse_host_scipy_gmres",
    "run_sparse_host_retry_candidate",
    "build_sparse_jax_retry_preconditioner",
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
