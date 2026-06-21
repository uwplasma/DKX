"""QI-specific x-block sparse-PC policy and stage helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..residual import (
    l2_norm_float as profile_l2_norm_float,
    safe_ratio as profile_safe_ratio,
)
from ..policies import (
    rhs1_qi_device_coupled_install_on_reject_requested,
    rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_extra_coarse_metadata,
    rhs1_qi_device_extra_coarse_setup_kwargs,
    rhs1_qi_device_probe_uses_minres_step,
    rhs1_qi_device_residual_correction_controls,
    rhs1_qi_device_residual_correction_metadata,
    rhs1_qi_device_residual_correction_setup_kwargs,
    rhs1_qi_device_setup_summary,
    rhs1_qi_device_status_fields,
    rhs1_qi_device_tail_block_required,
)
from ....rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    rhs1_xblock_qi_block_geometry_metadata,
)
from ....rhs1_qi_device_preconditioner import RHS1QIDevicePreconditionerConfig
from ....rhs1_qi_galerkin_policy import (
    RHS1QIGalerkinProbeCandidate,
    select_rhs1_qi_galerkin_probe_candidate,
)


ArrayFn = Callable[[jnp.ndarray], jnp.ndarray]
EmitFn = Callable[[int, str], None]


def _env_value(env: Mapping[str, str] | None, key: str) -> str:
    if env is None:
        return ""
    return str(env.get(key, "")).strip()


def _env_float(env: Mapping[str, str] | None, key: str, default: float) -> float:
    raw = _env_value(env, key)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def _env_int(
    env: Mapping[str, str] | None,
    key: str,
    default: int,
    minimum: int | None = None,
) -> int:
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
class XBlockQIDeviceStageContext:
    """Dependencies for optional QI-device preconditioner setup and admission."""

    op: object
    x0_full: jnp.ndarray | None
    xblock_rhs: jnp.ndarray
    base_preconditioner: ArrayFn
    basis_for_galerkin: RHS1QICoarseBasis | None
    seed_policy: XBlockQISeedPolicySetup
    active_dof: bool
    linear_size: int
    precondition_side: str
    true_matvec_no_count: ArrayFn
    assembled_device_operator: object | None
    assembled_operator_metadata: Mapping[str, object]
    assembled_operator_enabled: bool
    assembled_operator_built: bool
    assembled_operator_device_resident: bool
    assembled_operator_device_error: object | None
    host_fallback_used: bool
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None
    basis_builder: Callable[..., RHS1QICoarseBasis]
    setup_preconditioner: Callable[..., object]
    probe_preconditioner: Callable[..., tuple[jnp.ndarray, object]]
    probe_augmented_seed: Callable[..., object]


@dataclass(frozen=True)
class XBlockQIDeviceStageResult:
    """Result from optional QI-device preconditioner setup and seed probing."""

    preconditioner: ArrayFn
    x0_full: jnp.ndarray | None
    basis_for_galerkin: RHS1QICoarseBasis | None
    state_for_augmented_krylov: object | None
    augmented_seed_basis_for_krylov: jnp.ndarray | None
    augmented_seed_action_for_krylov: jnp.ndarray | None
    enabled: bool
    built: bool
    used: bool
    used_in_krylov: bool
    reason: str | None
    rank: int
    candidate_count: int
    coarse_shape: tuple[int, int]
    operator_on_basis_shape: tuple[int, int]
    coarse_norm: float
    operator_on_basis_norm: float
    residual_before: float | None
    residual_after: float | None
    improvement_ratio: float | None
    metadata: dict[str, object]
    setup_s: float
    min_improvement: float
    use_in_krylov: bool
    stats: dict[str, int]
    augmented_seed_requested: bool
    augmented_seed_available: bool
    augmented_seed_used: bool
    augmented_seed_rank: int
    augmented_seed_max_rank: int
    augmented_seed_reason: str | None
    augmented_seed_projection_residual: float | None
    augmented_seed_labels: tuple[str, ...]


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
class XBlockQIStagePipelineContext:
    """Dependencies for the complete optional QI x-block preconditioner lane."""

    op: object
    rhs: jnp.ndarray
    x0_full: jnp.ndarray | None
    xblock_rhs: jnp.ndarray
    xblock_rhs_norm: float
    base_preconditioner: ArrayFn
    matvec: ArrayFn
    true_matvec_no_count: ArrayFn
    direction_projector: ArrayFn | None
    active_dof: bool
    linear_size: int
    host_fallback_used: bool
    precondition_side: str
    assembled_device_operator: object | None
    assembled_operator_metadata: Mapping[str, object]
    assembled_operator_enabled: bool
    assembled_operator_built: bool
    assembled_operator_device_resident: bool
    assembled_operator_device_error: object | None
    elapsed_s: Callable[[], float]
    emit: EmitFn | None
    env: Mapping[str, str] | None
    basis_builder: Callable[..., RHS1QICoarseBasis]
    smoothed_load_basis_builder: Callable[..., RHS1QICoarseBasis]
    global_load_basis_builder: Callable[..., Sequence[tuple[str, jnp.ndarray]]]
    correction_builder: Callable[..., object]
    galerkin_preconditioner_builder: Callable[..., object]
    two_level_preconditioner_builder: Callable[..., object]
    orthonormalizer: Callable[..., RHS1QICoarseBasis]
    device_setup_preconditioner: Callable[..., object]
    device_probe_preconditioner: Callable[..., tuple[jnp.ndarray, object]]
    device_probe_augmented_seed: Callable[..., object]
    deflated_preconditioner_builder: Callable[..., object]
    deflated_minres_seed_probe: Callable[..., tuple[jnp.ndarray, object]]
    deflated_linear_probe: Callable[..., tuple[jnp.ndarray, object]]
    parse_galerkin_modes: Callable[..., tuple[str, ...]]
    parse_galerkin_dampings: Callable[..., tuple[float, ...]]
    reduce_full: ArrayFn | None


@dataclass(frozen=True)
class XBlockQIStagePipelineResult:
    """State produced by the complete optional QI x-block preconditioner lane."""

    preconditioner: ArrayFn
    x0_full: jnp.ndarray | None
    basis_for_galerkin: RHS1QICoarseBasis | None
    pc_factor_s: float
    qi_device_state_for_augmented_krylov: object | None
    qi_device_augmented_seed_basis_for_krylov: jnp.ndarray | None
    qi_device_augmented_seed_action_for_krylov: jnp.ndarray | None
    qi_coarse_seed_enabled: bool
    qi_coarse_seed_used: bool
    qi_coarse_seed_residual_before: float | None
    qi_coarse_seed_residual_after: float | None
    qi_coarse_seed_improvement_ratio: float | None
    qi_coarse_seed_rank: int
    qi_coarse_seed_candidate_count: int
    qi_coarse_seed_reason: str | None
    qi_coarse_seed_labels: tuple[str, ...]
    qi_coarse_seed_s: float
    qi_seed_max_rank: int
    qi_seed_max_candidates: int
    qi_seed_max_angular_mode: int
    qi_seed_basis_kind: str | None
    qi_galerkin_preconditioner_enabled: bool
    qi_galerkin_preconditioner_built: bool
    qi_galerkin_preconditioner_used: bool
    qi_galerkin_preconditioner_reason: str | None
    qi_galerkin_preconditioner_mode: str | None
    qi_galerkin_preconditioner_rank: int
    qi_galerkin_preconditioner_candidate_count: int
    qi_galerkin_preconditioner_coarse_shape: tuple[int, int]
    qi_galerkin_preconditioner_coarse_norm: float
    qi_galerkin_preconditioner_setup_s: float
    qi_galerkin_preconditioner_rcond: float
    qi_galerkin_preconditioner_damping: float
    qi_galerkin_preconditioner_basis_reused_from_seed: bool
    qi_galerkin_preconditioner_residual_before: float | None
    qi_galerkin_preconditioner_residual_after: float | None
    qi_galerkin_preconditioner_improvement_ratio: float | None
    qi_galerkin_preconditioner_probe_reduced: bool
    qi_galerkin_preconditioner_probe_candidates: list[dict[str, object]]
    qi_galerkin_preconditioner_selected_index: int | None
    qi_galerkin_stats: dict[str, int]
    qi_two_level_preconditioner_enabled: bool
    qi_two_level_preconditioner_built: bool
    qi_two_level_preconditioner_used: bool
    qi_two_level_preconditioner_reason: str | None
    qi_two_level_preconditioner_rank: int
    qi_two_level_preconditioner_candidate_count: int
    qi_two_level_preconditioner_coarse_shape: tuple[int, int]
    qi_two_level_preconditioner_coarse_norm: float
    qi_two_level_preconditioner_operator_on_basis_shape: tuple[int, int]
    qi_two_level_preconditioner_operator_on_basis_norm: float
    qi_two_level_preconditioner_coarse_solver: str | None
    qi_two_level_preconditioner_residual_augmented: bool
    qi_two_level_preconditioner_rank_before_augmentation: int
    qi_two_level_preconditioner_augmentation_labels: tuple[str, ...]
    qi_two_level_preconditioner_residual_augment_max_extra: int
    qi_two_level_preconditioner_residual_augment_steps: int
    qi_two_level_preconditioner_residual_augment_include_residuals: bool
    qi_two_level_preconditioner_smoothed_load_basis: bool
    qi_two_level_preconditioner_smoothed_load_metadata: dict[str, object]
    qi_two_level_preconditioner_setup_s: float
    qi_two_level_preconditioner_rcond: float
    qi_two_level_preconditioner_damping: float
    qi_two_level_preconditioner_basis_reused_from_seed: bool
    qi_two_level_preconditioner_residual_before: float | None
    qi_two_level_preconditioner_residual_after: float | None
    qi_two_level_preconditioner_improvement_ratio: float | None
    qi_two_level_preconditioner_probe_candidates: list[dict[str, object]]
    qi_two_level_preconditioner_selected_index: int | None
    qi_two_level_stats: dict[str, int]
    qi_device_preconditioner_enabled: bool
    qi_device_preconditioner_built: bool
    qi_device_preconditioner_used: bool
    qi_device_preconditioner_used_in_krylov: bool
    qi_device_preconditioner_reason: str | None
    qi_device_preconditioner_rank: int
    qi_device_preconditioner_candidate_count: int
    qi_device_preconditioner_coarse_shape: tuple[int, int]
    qi_device_preconditioner_operator_on_basis_shape: tuple[int, int]
    qi_device_preconditioner_coarse_norm: float
    qi_device_preconditioner_operator_on_basis_norm: float
    qi_device_preconditioner_residual_before: float | None
    qi_device_preconditioner_residual_after: float | None
    qi_device_preconditioner_improvement_ratio: float | None
    qi_device_preconditioner_metadata: dict[str, object]
    qi_device_preconditioner_setup_s: float
    qi_device_preconditioner_min_improvement: float
    qi_device_preconditioner_use_in_krylov: bool
    qi_device_stats: dict[str, int]
    qi_device_augmented_seed_requested: bool
    qi_device_augmented_seed_available: bool
    qi_device_augmented_seed_used: bool
    qi_device_augmented_seed_rank: int
    qi_device_augmented_seed_max_rank: int
    qi_device_augmented_seed_reason: str | None
    qi_device_augmented_seed_projection_residual: float | None
    qi_device_augmented_seed_labels: tuple[str, ...]
    qi_deflated_preconditioner_enabled: bool
    qi_deflated_preconditioner_built: bool
    qi_deflated_preconditioner_used: bool
    qi_deflated_preconditioner_used_in_krylov: bool
    qi_deflated_preconditioner_reason: str | None
    qi_deflated_preconditioner_rank: int
    qi_deflated_preconditioner_candidate_count: int
    qi_deflated_preconditioner_residual_before: float | None
    qi_deflated_preconditioner_residual_after: float | None
    qi_deflated_preconditioner_improvement_ratio: float | None
    qi_deflated_preconditioner_metadata: dict[str, object]
    qi_deflated_preconditioner_setup_s: float
    qi_deflated_stats: dict[str, int]

    def diagnostic_scope(self) -> dict[str, object]:
        """Return historical QI diagnostic names for final metadata builders."""

        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name
            not in {
                "preconditioner",
                "x0_full",
                "basis_for_galerkin",
                "pc_factor_s",
                "qi_device_state_for_augmented_krylov",
                "qi_device_augmented_seed_basis_for_krylov",
                "qi_device_augmented_seed_action_for_krylov",
            }
        }

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


def apply_xblock_qi_device_stage(
    context: XBlockQIDeviceStageContext,
) -> XBlockQIDeviceStageResult:
    """Build, probe, and optionally install the QI-device x-block preconditioner."""

    env = context.env
    stats = {"applies": 0}
    enabled = bool(context.seed_policy.device_preconditioner_enabled)
    reason: str | None = None
    metadata: dict[str, object] = {}
    setup_s = 0.0
    min_improvement = 0.0
    use_in_krylov = False
    state_for_augmented_krylov = None
    augmented_seed_basis_for_krylov = None
    augmented_seed_action_for_krylov = None
    augmented_seed_requested = False
    augmented_seed_available = False
    augmented_seed_used = False
    augmented_seed_rank = 0
    augmented_seed_max_rank = 0
    augmented_seed_reason: str | None = None
    augmented_seed_projection_residual: float | None = None
    augmented_seed_labels: tuple[str, ...] = ()
    rank = 0
    candidate_count = 0
    coarse_shape = (0, 0)
    operator_on_basis_shape = (0, 0)
    coarse_norm = 0.0
    operator_on_basis_norm = 0.0
    residual_before: float | None = None
    residual_after: float | None = None
    improvement_ratio: float | None = None

    admission = resolve_xblock_qi_device_admission_setup(
        enabled=enabled,
        host_fallback_used=bool(context.host_fallback_used),
        assembled_device_operator_available=context.assembled_device_operator is not None,
        assembled_operator_enabled=bool(context.assembled_operator_enabled),
        assembled_operator_built=bool(context.assembled_operator_built),
        assembled_operator_device_resident=bool(context.assembled_operator_device_resident),
        assembled_operator_device_error=context.assembled_operator_device_error,
        env=env,
    )
    if admission.reason is not None and not admission.should_build:
        reason = admission.reason
        metadata = dict(admission.metadata)
    for level, message in admission.messages:
        if context.emit is not None:
            context.emit(level, message)
    if not bool(admission.should_build):
        return XBlockQIDeviceStageResult(
            preconditioner=context.base_preconditioner,
            x0_full=context.x0_full,
            basis_for_galerkin=context.basis_for_galerkin,
            state_for_augmented_krylov=None,
            augmented_seed_basis_for_krylov=None,
            augmented_seed_action_for_krylov=None,
            enabled=enabled,
            built=False,
            used=False,
            used_in_krylov=False,
            reason=reason,
            rank=rank,
            candidate_count=candidate_count,
            coarse_shape=coarse_shape,
            operator_on_basis_shape=operator_on_basis_shape,
            coarse_norm=coarse_norm,
            operator_on_basis_norm=operator_on_basis_norm,
            residual_before=residual_before,
            residual_after=residual_after,
            improvement_ratio=improvement_ratio,
            metadata=metadata,
            setup_s=setup_s,
            min_improvement=min_improvement,
            use_in_krylov=use_in_krylov,
            stats=stats,
            augmented_seed_requested=augmented_seed_requested,
            augmented_seed_available=augmented_seed_available,
            augmented_seed_used=augmented_seed_used,
            augmented_seed_rank=augmented_seed_rank,
            augmented_seed_max_rank=augmented_seed_max_rank,
            augmented_seed_reason=augmented_seed_reason,
            augmented_seed_projection_residual=augmented_seed_projection_residual,
            augmented_seed_labels=augmented_seed_labels,
        )

    start_s = context.elapsed_s()
    preconditioner = context.base_preconditioner
    x0_full = context.x0_full
    basis_for_galerkin = context.basis_for_galerkin
    used = False
    used_in_krylov = False
    try:
        matrix_free_enabled = bool(admission.matrix_free_enabled)
        base_config = resolve_xblock_qi_device_base_config_setup(
            matrix_free_enabled=matrix_free_enabled,
            assembled_device_operator_available=context.assembled_device_operator is not None,
            precondition_side=str(context.precondition_side),
            probe_uses_minres_step=rhs1_qi_device_probe_uses_minres_step,
            env=env,
        )
        local_smoother_kind = base_config.local_smoother_kind
        min_improvement = float(base_config.min_improvement)
        cycles = int(base_config.cycles)
        augmented_seed_requested = bool(base_config.augmented_seed_requested)
        augmented_seed_max_rank = int(base_config.augmented_seed_max_rank)
        minres_step = bool(base_config.minres_step)
        alpha_clip = float(base_config.alpha_clip)
        use_in_krylov_requested = bool(base_config.use_in_krylov_requested)
        use_in_krylov = bool(base_config.use_in_krylov)
        compose_with_base = bool(base_config.compose_with_base)
        compose_mode = base_config.compose_mode
        enrichment_config = resolve_xblock_qi_device_enrichment_config_setup(
            matrix_free_enabled=matrix_free_enabled,
            env=env,
        )
        operator_krylov_enrichment = bool(
            enrichment_config.operator_krylov_enrichment
        )
        multilevel_config = resolve_xblock_qi_device_multilevel_config_setup(env=env)
        multilevel_coarse = bool(multilevel_config.multilevel_coarse)
        residual_correction_controls = rhs1_qi_device_residual_correction_controls()
        residual_correction_setup_kwargs = (
            rhs1_qi_device_residual_correction_setup_kwargs(
                residual_correction_controls
            )
        )
        residual_correction_metadata = rhs1_qi_device_residual_correction_metadata(
            residual_correction_controls
        )
        multilevel_max_rank_env = _env_value(
            env,
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RANK",
        )
        multilevel_max_rank: int | None = None
        if multilevel_max_rank_env:
            try:
                multilevel_max_rank = max(1, int(multilevel_max_rank_env))
            except ValueError:
                multilevel_max_rank = None
        extra_coarse_controls = rhs1_qi_device_extra_coarse_controls()
        extra_coarse_setup_kwargs = rhs1_qi_device_extra_coarse_setup_kwargs(
            extra_coarse_controls
        )
        extra_coarse_metadata = rhs1_qi_device_extra_coarse_metadata(
            extra_coarse_controls
        )
        setup_summary = rhs1_qi_device_setup_summary(
            seed_max_rank=int(context.seed_policy.max_rank),
            n_species=int(getattr(context.op, "n_species", 1)),
            assembled_device_operator_available=(
                context.assembled_device_operator is not None
            ),
            enrichment_config=enrichment_config,
            multilevel_config=multilevel_config,
            multilevel_max_rank=multilevel_max_rank,
            extra_coarse_controls=extra_coarse_controls,
            residual_correction_controls=residual_correction_controls,
        )
        max_rank = setup_summary.max_rank
        if (
            context.assembled_device_operator is None
            and not bool(matrix_free_enabled)
        ):
            raise RuntimeError(
                "missing assembled device CSR operator and matrix-free fallback disabled"
            )
        basis_reused_from_seed = basis_for_galerkin is not None
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
                include_radial_angular=bool(
                    context.seed_policy.include_radial_angular
                ),
                include_constraint_moments=bool(
                    context.seed_policy.include_constraint_moments
                ),
                include_schur=bool(context.seed_policy.include_schur),
            )
        current = (
            jnp.zeros_like(context.xblock_rhs)
            if x0_full is None
            else jnp.asarray(x0_full, dtype=jnp.float64)
        )
        residual_seed = None
        if bool(setup_summary.residual_seed_required):
            residual_seed = context.xblock_rhs - context.true_matvec_no_count(current)
        operator_for_setup = (
            context.assembled_device_operator
            if context.assembled_device_operator is not None
            else context.true_matvec_no_count
        )
        if context.emit is not None:
            for message in setup_summary.progress_messages:
                context.emit(1, message)
        setup_config = build_xblock_qi_device_setup_config(
            XBlockQIDeviceSetupConfigContext(
                op=context.op,
                active_dof=bool(context.active_dof),
                linear_size=int(context.linear_size),
                base_config=base_config,
                enrichment_config=enrichment_config,
                multilevel_config=multilevel_config,
                multilevel_max_rank=multilevel_max_rank,
                max_rank=max_rank,
                extra_coarse_controls=extra_coarse_controls,
                extra_coarse_setup_kwargs=extra_coarse_setup_kwargs,
                residual_correction_setup_kwargs=residual_correction_setup_kwargs,
            )
        )
        state = context.setup_preconditioner(
            operator=operator_for_setup,
            coarse_basis=basis_for_galerkin,
            residual_seed=residual_seed,
            total_size=int(context.linear_size),
            dtype=jnp.float64,
            operator_metadata=context.assembled_operator_metadata,
            geometry_metadata=setup_config.geometry_metadata,
            config=setup_config.config,
        )
        state_for_augmented_krylov = state
        rank = int(state.metadata.rank)
        candidate_count = int(state.basis.metadata.candidate_count)
        coarse_shape = tuple(
            int(value) for value in state.metadata.coarse_operator_shape
        )
        operator_on_basis_shape = tuple(
            int(value) for value in state.metadata.operator_on_basis_shape
        )
        coarse_norm = float(state.metadata.coarse_operator_norm)
        operator_on_basis_norm = float(state.metadata.operator_on_basis_norm)
        if bool(augmented_seed_requested):
            augmented_seed = context.probe_augmented_seed(
                rhs=context.xblock_rhs,
                x0=current,
                state=state,
                operator=context.true_matvec_no_count,
                min_relative_improvement=float(min_improvement),
                max_cycles=int(cycles),
                residual_minimizing_step=bool(minres_step),
                alpha_clip=float(alpha_clip),
                max_rank=int(augmented_seed_max_rank),
            )
            candidate = augmented_seed.solution
            probe = augmented_seed.probe
            augmented_seed_rank = int(augmented_seed.rank)
            augmented_seed_available = bool(
                probe.accepted and augmented_seed_rank > 0
            )
            augmented_seed_reason = str(augmented_seed.reason)
            augmented_seed_projection_residual = (
                None
                if augmented_seed.projection_residual_norm is None
                else float(augmented_seed.projection_residual_norm)
            )
            augmented_seed_labels = tuple(
                str(label) for label in augmented_seed.accepted_labels
            )
            if bool(augmented_seed_available):
                augmented_seed_basis_for_krylov = jnp.asarray(
                    augmented_seed.augmentation_basis,
                    dtype=jnp.float64,
                )
                augmented_seed_action_for_krylov = jnp.asarray(
                    augmented_seed.operator_on_augmentation,
                    dtype=jnp.float64,
                )
        else:
            candidate, probe = context.probe_preconditioner(
                rhs=context.xblock_rhs,
                x0=current,
                state=state,
                operator=context.true_matvec_no_count,
                min_relative_improvement=float(min_improvement),
                max_cycles=int(cycles),
                residual_minimizing_step=bool(minres_step),
                alpha_clip=float(alpha_clip),
            )
        residual_before = float(probe.residual_before_norm)
        residual_after = float(probe.residual_after_norm)
        improvement_ratio = (
            None if probe.improvement_ratio is None else float(probe.improvement_ratio)
        )
        reason = str(probe.reason)
        metadata = build_xblock_qi_device_preconditioner_metadata(
            XBlockQIDeviceMetadataContext(
                probe=probe,
                state=state,
                basis_reused_from_seed=basis_reused_from_seed,
                min_improvement=min_improvement,
                cycles_requested=cycles,
                minres_step=minres_step,
                alpha_clip=alpha_clip,
                augmented_seed_requested=augmented_seed_requested,
                augmented_seed_available=augmented_seed_available,
                augmented_seed_used=augmented_seed_used,
                augmented_seed_rank=augmented_seed_rank,
                augmented_seed_max_rank=augmented_seed_max_rank,
                augmented_seed_reason=augmented_seed_reason,
                augmented_seed_projection_residual=augmented_seed_projection_residual,
                augmented_seed_labels=augmented_seed_labels,
                use_in_krylov=use_in_krylov,
                use_in_krylov_requested=use_in_krylov_requested,
                precondition_side=context.precondition_side,
                compose_with_base=compose_with_base,
                compose_mode=compose_mode,
                matrix_free_enabled=matrix_free_enabled,
                local_smoother_kind=local_smoother_kind,
                enrichment_config=enrichment_config,
                multilevel_config=multilevel_config,
                multilevel_max_rank=multilevel_max_rank,
                extra_coarse_metadata=extra_coarse_metadata,
                residual_correction_metadata=residual_correction_metadata,
                max_rank_requested=max_rank,
            )
        )
        probe_cycles = int(metadata.get("cycles", 1 if bool(probe.accepted) else 0))
        base_preconditioner = context.base_preconditioner

        def _precond_qi_device(v: jnp.ndarray) -> jnp.ndarray:
            stats["applies"] += 1
            v_j = jnp.asarray(v, dtype=jnp.float64)
            if bool(compose_with_base):
                base = jnp.asarray(base_preconditioner(v_j), dtype=jnp.float64)
                if compose_mode == "multiplicative":
                    coarse_input = v_j - jnp.asarray(
                        context.true_matvec_no_count(base),
                        dtype=jnp.float64,
                    )
                else:
                    coarse_input = v_j
                coarse = jnp.asarray(
                    state.apply(jnp.asarray(coarse_input, dtype=jnp.float64)),
                    dtype=jnp.float64,
                )
                return base + coarse
            return jnp.asarray(state.apply(v_j), dtype=jnp.float64)

        coupled_stage_accepted_for_krylov = (
            bool(metadata.get("coupled_residual_equation_accepted", False))
            and int(metadata.get("coupled_residual_equation_rank", 0) or 0) > 0
        )
        install_after_seed_reject = bool(
            rhs1_qi_device_coupled_install_on_reject_requested(
                residual_correction_controls
            )
            and use_in_krylov
            and coupled_stage_accepted_for_krylov
            and not bool(probe.accepted)
        )
        metadata["seed_probe_accepted"] = bool(probe.accepted)
        metadata["installed_in_krylov_after_seed_reject"] = bool(
            install_after_seed_reject
        )
        status_fields = rhs1_qi_device_status_fields(
            extra_coarse_controls=extra_coarse_controls,
            residual_correction_controls=residual_correction_controls,
            metadata=metadata,
        )
        if bool(probe.accepted):
            x0_full = jnp.asarray(candidate, dtype=jnp.float64)
            used = True
            if bool(use_in_krylov):
                preconditioner = _precond_qi_device
                used_in_krylov = True
            if context.emit is not None:
                context.emit(
                    0,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI device preconditioner accepted "
                    f"residual {residual_before:.6e} -> {residual_after:.6e} "
                    f"(rank={int(rank)} cycles={int(probe_cycles)} "
                    f"ratio={float(improvement_ratio):.6e} "
                    f"use_in_krylov={int(bool(use_in_krylov))} "
                    f"augmented_seed_requested={int(bool(augmented_seed_requested))} "
                    f"augmented_seed_available={int(bool(augmented_seed_available))} "
                    f"augmented_seed_used={int(bool(augmented_seed_used))} "
                    f"augmented_seed_rank={int(augmented_seed_rank)} "
                    f"augmented_seed_max_rank={int(augmented_seed_max_rank)} "
                    f"augmented_seed_reason={augmented_seed_reason or 'none'} "
                    "augmented_seed_projection_residual="
                    f"{float(augmented_seed_projection_residual) if augmented_seed_projection_residual is not None else float('nan'):.6e} "
                    f"operator_krylov={int(bool(operator_krylov_enrichment))} "
                    f"coarse_reuse={int(bool(multilevel_coarse))} "
                    f"{status_fields} "
                    f"compose_base={int(bool(compose_with_base))})",
                )
        elif bool(install_after_seed_reject):
            preconditioner = _precond_qi_device
            used = True
            used_in_krylov = True
            reason = "krylov_installed_after_seed_probe_reject"
            if context.emit is not None:
                context.emit(
                    1,
                    "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                    "QI device preconditioner installed in Krylov after seed "
                    f"probe reject (rank={int(rank)} "
                    f"coupled_rank={int(metadata.get('coupled_residual_equation_rank', 0))} "
                    f"coupled_candidates={int(metadata.get('coupled_residual_equation_candidate_count', 0))} "
                    f"residual {float(residual_before):.6e} -> {float(residual_after):.6e})",
                )
        elif context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI device preconditioner rejected "
                f"reason={reason} "
                f"residual {float(residual_before):.6e} -> {float(residual_after):.6e} "
                f"(rank={int(rank)} cycles={int(probe_cycles)} "
                f"ratio={float(improvement_ratio) if improvement_ratio is not None else float('nan'):.6e} "
                f"step_policy={metadata.get('step_policy', 'fixed')} "
                f"use_in_krylov={int(bool(use_in_krylov))} "
                f"augmented_seed_requested={int(bool(augmented_seed_requested))} "
                f"augmented_seed_available={int(bool(augmented_seed_available))} "
                f"augmented_seed_used={int(bool(augmented_seed_used))} "
                f"augmented_seed_rank={int(augmented_seed_rank)} "
                f"augmented_seed_max_rank={int(augmented_seed_max_rank)} "
                f"augmented_seed_reason={augmented_seed_reason or 'none'} "
                "augmented_seed_projection_residual="
                f"{float(augmented_seed_projection_residual) if augmented_seed_projection_residual is not None else float('nan'):.6e} "
                f"operator_krylov={int(bool(operator_krylov_enrichment))} "
                f"coarse_reuse={int(bool(multilevel_coarse))} "
                f"{status_fields} "
                f"compose_base={int(bool(compose_with_base))})",
            )
    except Exception as exc:  # noqa: BLE001
        reason = f"{type(exc).__name__}: {exc}"
        metadata = {"error": reason}
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"QI device preconditioner disabled after build failure ({type(exc).__name__}: {exc})",
            )
    setup_s = float(context.elapsed_s() - start_s)
    return XBlockQIDeviceStageResult(
        preconditioner=preconditioner,
        x0_full=x0_full,
        basis_for_galerkin=basis_for_galerkin,
        state_for_augmented_krylov=state_for_augmented_krylov,
        augmented_seed_basis_for_krylov=augmented_seed_basis_for_krylov,
        augmented_seed_action_for_krylov=augmented_seed_action_for_krylov,
        enabled=enabled,
        built=state_for_augmented_krylov is not None,
        used=used,
        used_in_krylov=used_in_krylov,
        reason=reason,
        rank=rank,
        candidate_count=candidate_count,
        coarse_shape=coarse_shape,
        operator_on_basis_shape=operator_on_basis_shape,
        coarse_norm=coarse_norm,
        operator_on_basis_norm=operator_on_basis_norm,
        residual_before=residual_before,
        residual_after=residual_after,
        improvement_ratio=improvement_ratio,
        metadata=metadata,
        setup_s=setup_s,
        min_improvement=min_improvement,
        use_in_krylov=use_in_krylov,
        stats=stats,
        augmented_seed_requested=augmented_seed_requested,
        augmented_seed_available=augmented_seed_available,
        augmented_seed_used=augmented_seed_used,
        augmented_seed_rank=augmented_seed_rank,
        augmented_seed_max_rank=augmented_seed_max_rank,
        augmented_seed_reason=augmented_seed_reason,
        augmented_seed_projection_residual=augmented_seed_projection_residual,
        augmented_seed_labels=augmented_seed_labels,
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


def run_xblock_qi_preconditioner_pipeline(
    context: XBlockQIStagePipelineContext,
) -> XBlockQIStagePipelineResult:
    """Run all optional QI x-block stages and return one diagnostic state."""

    seed_policy = resolve_xblock_qi_seed_policy_setup(env=context.env)
    preconditioner = context.base_preconditioner
    x0_full = context.x0_full
    basis_for_galerkin: RHS1QICoarseBasis | None = None
    pc_factor_s = 0.0

    coarse_seed_stage = apply_xblock_qi_coarse_seed_stage(
        context=XBlockQICoarseSeedStageContext(
            op=context.op,
            x0_full=x0_full,
            xblock_rhs=context.xblock_rhs,
            matvec_no_count=context.true_matvec_no_count,
            active_dof=bool(context.active_dof),
            linear_size=int(context.linear_size),
            policy=seed_policy,
            elapsed_s=context.elapsed_s,
            emit=context.emit,
            basis_builder=context.basis_builder,
            correction_builder=context.correction_builder,
        )
    )
    x0_full = coarse_seed_stage.x0_full
    basis_for_galerkin = coarse_seed_stage.basis_for_galerkin

    galerkin_policy = resolve_xblock_qi_galerkin_policy_setup(
        enabled=bool(seed_policy.galerkin_preconditioner_enabled),
        host_fallback_used=bool(context.host_fallback_used),
        precondition_side=str(context.precondition_side),
        parse_modes=context.parse_galerkin_modes,
        parse_dampings=context.parse_galerkin_dampings,
        env=context.env,
    )
    galerkin_stage = apply_xblock_qi_galerkin_stage(
        context=XBlockQIGalerkinStageContext(
            op=context.op,
            base_preconditioner=preconditioner,
            matvec=context.matvec,
            true_matvec_no_count=context.true_matvec_no_count,
            xblock_rhs=context.xblock_rhs,
            xblock_rhs_norm=float(context.xblock_rhs_norm),
            active_dof=bool(context.active_dof),
            linear_size=int(context.linear_size),
            basis_for_galerkin=basis_for_galerkin,
            seed_policy=seed_policy,
            galerkin_policy=galerkin_policy,
            elapsed_s=context.elapsed_s,
            emit=context.emit,
            basis_builder=context.basis_builder,
            preconditioner_builder=context.galerkin_preconditioner_builder,
        )
    )
    preconditioner = galerkin_stage.preconditioner
    basis_for_galerkin = galerkin_stage.basis_for_galerkin
    pc_factor_s += float(galerkin_stage.setup_s)

    two_level_policy = resolve_xblock_qi_two_level_policy_setup(
        enabled=bool(seed_policy.two_level_preconditioner_enabled),
        host_fallback_used=bool(context.host_fallback_used),
        precondition_side=str(context.precondition_side),
        seed_max_rank=int(seed_policy.max_rank),
        parse_dampings=context.parse_galerkin_dampings,
        env=context.env,
    )
    two_level_stage = apply_xblock_qi_two_level_stage(
        context=XBlockQITwoLevelStageContext(
            op=context.op,
            rhs=context.rhs,
            x0_full=x0_full,
            xblock_rhs=context.xblock_rhs,
            base_preconditioner=preconditioner,
            matvec=context.matvec,
            true_matvec_no_count=context.true_matvec_no_count,
            direction_projector=context.direction_projector,
            active_dof=bool(context.active_dof),
            linear_size=int(context.linear_size),
            basis_for_galerkin=basis_for_galerkin,
            seed_policy=seed_policy,
            two_level_policy=two_level_policy,
            elapsed_s=context.elapsed_s,
            emit=context.emit,
            basis_builder=context.basis_builder,
            smoothed_load_basis_builder=context.smoothed_load_basis_builder,
            orthonormalizer=context.orthonormalizer,
            preconditioner_builder=context.two_level_preconditioner_builder,
        )
    )
    preconditioner = two_level_stage.preconditioner
    x0_full = two_level_stage.x0_full
    basis_for_galerkin = two_level_stage.basis_for_galerkin
    pc_factor_s += float(two_level_stage.setup_s)

    device_stage = apply_xblock_qi_device_stage(
        XBlockQIDeviceStageContext(
            op=context.op,
            x0_full=x0_full,
            xblock_rhs=context.xblock_rhs,
            base_preconditioner=preconditioner,
            basis_for_galerkin=basis_for_galerkin,
            seed_policy=seed_policy,
            active_dof=bool(context.active_dof),
            linear_size=int(context.linear_size),
            precondition_side=str(context.precondition_side),
            true_matvec_no_count=context.true_matvec_no_count,
            assembled_device_operator=context.assembled_device_operator,
            assembled_operator_metadata=context.assembled_operator_metadata,
            assembled_operator_enabled=bool(context.assembled_operator_enabled),
            assembled_operator_built=bool(context.assembled_operator_built),
            assembled_operator_device_resident=bool(
                context.assembled_operator_device_resident
            ),
            assembled_operator_device_error=context.assembled_operator_device_error,
            host_fallback_used=bool(context.host_fallback_used),
            elapsed_s=context.elapsed_s,
            emit=context.emit,
            env=context.env,
            basis_builder=context.basis_builder,
            setup_preconditioner=context.device_setup_preconditioner,
            probe_preconditioner=context.device_probe_preconditioner,
            probe_augmented_seed=context.device_probe_augmented_seed,
        )
    )
    preconditioner = device_stage.preconditioner
    x0_full = device_stage.x0_full
    basis_for_galerkin = device_stage.basis_for_galerkin
    pc_factor_s += float(device_stage.setup_s)

    deflated_enabled = bool(seed_policy.deflated_preconditioner_enabled)
    deflated_stage: XBlockQIDeflatedStageResult | None = None
    if deflated_enabled and bool(context.host_fallback_used):
        deflated_reason = "disabled_by_device_host_fallback"
        if context.emit is not None:
            context.emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                "QI residual-deflated preconditioner disabled because "
                "device-host fallback is active",
            )
    elif deflated_enabled and str(context.precondition_side) == "none":
        deflated_reason = "disabled_by_precondition_side_none"
    elif deflated_enabled:
        deflated_policy = resolve_xblock_qi_deflated_policy_setup(context.env)
        deflated_stage = apply_xblock_qi_deflated_stage(
            context=XBlockQIDeflatedStageContext(
                op=context.op,
                rhs=context.rhs,
                x0_full=x0_full,
                xblock_rhs=context.xblock_rhs,
                base_preconditioner=preconditioner,
                matvec=context.matvec,
                true_matvec_no_count=context.true_matvec_no_count,
                active_dof=bool(context.active_dof),
                reduce_full=context.reduce_full if bool(context.active_dof) else None,
                policy=deflated_policy,
                elapsed_s=context.elapsed_s,
                emit=context.emit,
                global_load_basis_builder=context.global_load_basis_builder,
                preconditioner_builder=context.deflated_preconditioner_builder,
                minres_seed_probe=context.deflated_minres_seed_probe,
                linear_probe=context.deflated_linear_probe,
            )
        )
        preconditioner = deflated_stage.preconditioner
        x0_full = deflated_stage.x0_full
        pc_factor_s += float(deflated_stage.setup_s)
        deflated_reason = deflated_stage.reason
    else:
        deflated_reason = None

    return XBlockQIStagePipelineResult(
        preconditioner=preconditioner,
        x0_full=x0_full,
        basis_for_galerkin=basis_for_galerkin,
        pc_factor_s=float(pc_factor_s),
        qi_device_state_for_augmented_krylov=device_stage.state_for_augmented_krylov,
        qi_device_augmented_seed_basis_for_krylov=(
            device_stage.augmented_seed_basis_for_krylov
        ),
        qi_device_augmented_seed_action_for_krylov=(
            device_stage.augmented_seed_action_for_krylov
        ),
        qi_coarse_seed_enabled=bool(seed_policy.coarse_seed_enabled),
        qi_coarse_seed_used=bool(coarse_seed_stage.used),
        qi_coarse_seed_residual_before=coarse_seed_stage.residual_before,
        qi_coarse_seed_residual_after=coarse_seed_stage.residual_after,
        qi_coarse_seed_improvement_ratio=coarse_seed_stage.improvement_ratio,
        qi_coarse_seed_rank=int(coarse_seed_stage.rank),
        qi_coarse_seed_candidate_count=int(coarse_seed_stage.candidate_count),
        qi_coarse_seed_reason=coarse_seed_stage.reason,
        qi_coarse_seed_labels=coarse_seed_stage.labels,
        qi_coarse_seed_s=float(coarse_seed_stage.setup_s),
        qi_seed_max_rank=int(seed_policy.max_rank),
        qi_seed_max_candidates=int(seed_policy.max_candidates),
        qi_seed_max_angular_mode=int(seed_policy.max_angular_mode),
        qi_seed_basis_kind=seed_policy.basis_kind,
        qi_galerkin_preconditioner_enabled=bool(
            seed_policy.galerkin_preconditioner_enabled
        ),
        qi_galerkin_preconditioner_built=bool(galerkin_stage.built),
        qi_galerkin_preconditioner_used=bool(galerkin_stage.used),
        qi_galerkin_preconditioner_reason=galerkin_stage.reason,
        qi_galerkin_preconditioner_mode=galerkin_stage.mode,
        qi_galerkin_preconditioner_rank=int(galerkin_stage.rank),
        qi_galerkin_preconditioner_candidate_count=int(
            galerkin_stage.candidate_count
        ),
        qi_galerkin_preconditioner_coarse_shape=galerkin_stage.coarse_shape,
        qi_galerkin_preconditioner_coarse_norm=float(galerkin_stage.coarse_norm),
        qi_galerkin_preconditioner_setup_s=float(galerkin_stage.setup_s),
        qi_galerkin_preconditioner_rcond=float(galerkin_stage.rcond),
        qi_galerkin_preconditioner_damping=float(galerkin_stage.damping),
        qi_galerkin_preconditioner_basis_reused_from_seed=bool(
            galerkin_stage.basis_reused_from_seed
        ),
        qi_galerkin_preconditioner_residual_before=(
            galerkin_stage.residual_before
        ),
        qi_galerkin_preconditioner_residual_after=galerkin_stage.residual_after,
        qi_galerkin_preconditioner_improvement_ratio=(
            galerkin_stage.improvement_ratio
        ),
        qi_galerkin_preconditioner_probe_reduced=bool(galerkin_stage.probe_reduced),
        qi_galerkin_preconditioner_probe_candidates=(
            galerkin_stage.probe_candidates
        ),
        qi_galerkin_preconditioner_selected_index=galerkin_stage.selected_index,
        qi_galerkin_stats=galerkin_stage.stats,
        qi_two_level_preconditioner_enabled=bool(
            seed_policy.two_level_preconditioner_enabled
        ),
        qi_two_level_preconditioner_built=bool(two_level_stage.built),
        qi_two_level_preconditioner_used=bool(two_level_stage.used),
        qi_two_level_preconditioner_reason=two_level_stage.reason,
        qi_two_level_preconditioner_rank=int(two_level_stage.rank),
        qi_two_level_preconditioner_candidate_count=int(
            two_level_stage.candidate_count
        ),
        qi_two_level_preconditioner_coarse_shape=two_level_stage.coarse_shape,
        qi_two_level_preconditioner_coarse_norm=float(two_level_stage.coarse_norm),
        qi_two_level_preconditioner_operator_on_basis_shape=(
            two_level_stage.operator_on_basis_shape
        ),
        qi_two_level_preconditioner_operator_on_basis_norm=float(
            two_level_stage.operator_on_basis_norm
        ),
        qi_two_level_preconditioner_coarse_solver=two_level_stage.coarse_solver,
        qi_two_level_preconditioner_residual_augmented=bool(
            two_level_stage.residual_augmented
        ),
        qi_two_level_preconditioner_rank_before_augmentation=int(
            two_level_stage.rank_before_augmentation
        ),
        qi_two_level_preconditioner_augmentation_labels=(
            two_level_stage.augmentation_labels
        ),
        qi_two_level_preconditioner_residual_augment_max_extra=int(
            two_level_stage.residual_augment_max_extra
        ),
        qi_two_level_preconditioner_residual_augment_steps=int(
            two_level_stage.residual_augment_steps
        ),
        qi_two_level_preconditioner_residual_augment_include_residuals=bool(
            two_level_stage.residual_augment_include_residuals
        ),
        qi_two_level_preconditioner_smoothed_load_basis=bool(
            two_level_stage.smoothed_load_basis
        ),
        qi_two_level_preconditioner_smoothed_load_metadata=(
            two_level_stage.smoothed_load_metadata
        ),
        qi_two_level_preconditioner_setup_s=float(two_level_stage.setup_s),
        qi_two_level_preconditioner_rcond=float(two_level_stage.rcond),
        qi_two_level_preconditioner_damping=float(two_level_stage.damping),
        qi_two_level_preconditioner_basis_reused_from_seed=bool(
            two_level_stage.basis_reused_from_seed
        ),
        qi_two_level_preconditioner_residual_before=(
            two_level_stage.residual_before
        ),
        qi_two_level_preconditioner_residual_after=two_level_stage.residual_after,
        qi_two_level_preconditioner_improvement_ratio=(
            two_level_stage.improvement_ratio
        ),
        qi_two_level_preconditioner_probe_candidates=(
            two_level_stage.probe_candidates
        ),
        qi_two_level_preconditioner_selected_index=two_level_stage.selected_index,
        qi_two_level_stats=two_level_stage.stats,
        qi_device_preconditioner_enabled=bool(device_stage.enabled),
        qi_device_preconditioner_built=bool(device_stage.built),
        qi_device_preconditioner_used=bool(device_stage.used),
        qi_device_preconditioner_used_in_krylov=bool(device_stage.used_in_krylov),
        qi_device_preconditioner_reason=device_stage.reason,
        qi_device_preconditioner_rank=int(device_stage.rank),
        qi_device_preconditioner_candidate_count=int(device_stage.candidate_count),
        qi_device_preconditioner_coarse_shape=device_stage.coarse_shape,
        qi_device_preconditioner_operator_on_basis_shape=(
            device_stage.operator_on_basis_shape
        ),
        qi_device_preconditioner_coarse_norm=float(device_stage.coarse_norm),
        qi_device_preconditioner_operator_on_basis_norm=float(
            device_stage.operator_on_basis_norm
        ),
        qi_device_preconditioner_residual_before=device_stage.residual_before,
        qi_device_preconditioner_residual_after=device_stage.residual_after,
        qi_device_preconditioner_improvement_ratio=device_stage.improvement_ratio,
        qi_device_preconditioner_metadata=device_stage.metadata,
        qi_device_preconditioner_setup_s=float(device_stage.setup_s),
        qi_device_preconditioner_min_improvement=float(
            device_stage.min_improvement
        ),
        qi_device_preconditioner_use_in_krylov=bool(device_stage.use_in_krylov),
        qi_device_stats=device_stage.stats,
        qi_device_augmented_seed_requested=bool(
            device_stage.augmented_seed_requested
        ),
        qi_device_augmented_seed_available=bool(
            device_stage.augmented_seed_available
        ),
        qi_device_augmented_seed_used=bool(device_stage.augmented_seed_used),
        qi_device_augmented_seed_rank=int(device_stage.augmented_seed_rank),
        qi_device_augmented_seed_max_rank=int(device_stage.augmented_seed_max_rank),
        qi_device_augmented_seed_reason=device_stage.augmented_seed_reason,
        qi_device_augmented_seed_projection_residual=(
            device_stage.augmented_seed_projection_residual
        ),
        qi_device_augmented_seed_labels=device_stage.augmented_seed_labels,
        qi_deflated_preconditioner_enabled=bool(deflated_enabled),
        qi_deflated_preconditioner_built=(
            bool(deflated_stage.built) if deflated_stage is not None else False
        ),
        qi_deflated_preconditioner_used=(
            bool(deflated_stage.used) if deflated_stage is not None else False
        ),
        qi_deflated_preconditioner_used_in_krylov=(
            bool(deflated_stage.used_in_krylov)
            if deflated_stage is not None
            else False
        ),
        qi_deflated_preconditioner_reason=deflated_reason,
        qi_deflated_preconditioner_rank=(
            int(deflated_stage.rank) if deflated_stage is not None else 0
        ),
        qi_deflated_preconditioner_candidate_count=(
            int(deflated_stage.candidate_count) if deflated_stage is not None else 0
        ),
        qi_deflated_preconditioner_residual_before=(
            deflated_stage.residual_before if deflated_stage is not None else None
        ),
        qi_deflated_preconditioner_residual_after=(
            deflated_stage.residual_after if deflated_stage is not None else None
        ),
        qi_deflated_preconditioner_improvement_ratio=(
            deflated_stage.improvement_ratio if deflated_stage is not None else None
        ),
        qi_deflated_preconditioner_metadata=(
            deflated_stage.metadata if deflated_stage is not None else {}
        ),
        qi_deflated_preconditioner_setup_s=(
            float(deflated_stage.setup_s) if deflated_stage is not None else 0.0
        ),
        qi_deflated_stats=(
            deflated_stage.stats
            if deflated_stage is not None
            else {"applies": 0, "local_applies": 0}
        ),
    )


__all__ = (
    "XBlockQICoarseSeedStageContext",
    "XBlockQICoarseSeedStageResult",
    "XBlockQIDeflatedPolicySetup",
    "XBlockQIDeflatedStageContext",
    "XBlockQIDeflatedStageResult",
    "XBlockQIDeviceAdmissionSetup",
    "XBlockQIDeviceBaseConfigSetup",
    "XBlockQIDeviceEnrichmentConfigSetup",
    "XBlockQIDeviceMetadataContext",
    "XBlockQIDeviceMultilevelConfigSetup",
    "XBlockQIDeviceOperatorReuseSetup",
    "XBlockQIDeviceStageContext",
    "XBlockQIDeviceStageResult",
    "XBlockQIDeviceSetupConfig",
    "XBlockQIDeviceSetupConfigContext",
    "XBlockQIGalerkinPolicySetup",
    "XBlockQIGalerkinStageContext",
    "XBlockQIGalerkinStageResult",
    "XBlockQIStagePipelineContext",
    "XBlockQIStagePipelineResult",
    "XBlockQISeedPolicySetup",
    "XBlockQITwoLevelPolicySetup",
    "XBlockQITwoLevelStageContext",
    "XBlockQITwoLevelStageResult",
    "apply_xblock_qi_coarse_seed_stage",
    "apply_xblock_qi_deflated_stage",
    "apply_xblock_qi_device_stage",
    "apply_xblock_qi_galerkin_stage",
    "apply_xblock_qi_two_level_stage",
    "build_xblock_qi_device_preconditioner_metadata",
    "build_xblock_qi_device_setup_config",
    "resolve_xblock_qi_deflated_policy_setup",
    "resolve_xblock_qi_device_admission_setup",
    "resolve_xblock_qi_device_base_config_setup",
    "resolve_xblock_qi_device_enrichment_config_setup",
    "resolve_xblock_qi_device_multilevel_config_setup",
    "resolve_xblock_qi_device_operator_reuse_setup",
    "resolve_xblock_qi_galerkin_policy_setup",
    "resolve_xblock_qi_seed_policy_setup",
    "resolve_xblock_qi_two_level_policy_setup",
    "run_xblock_qi_preconditioner_pipeline",
)
