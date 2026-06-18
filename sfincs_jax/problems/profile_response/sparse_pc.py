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
class XBlockQIDeviceOperatorReuseSetup:
    """QI-device operator-reuse admission and x-block factor-build routing."""

    decision: object
    skip_xblock_factors: bool
    xblock_jax_factors: bool
    xblock_device_krylov_forced_jax_factors: bool
    factor_backend: str
    factor_reason: str
    messages: tuple[tuple[int, str], ...]


def sparse_xblock_rescue_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return stable diagnostics for the sparse x-block rescue tail."""

    return {
        "sparse_xblock_rescue_active": bool(
            scope["sparse_xblock_rescue_active"]
        ),
        "sparse_xblock_rescue_attempted": bool(
            scope["sparse_xblock_rescue_attempted"]
        ),
        "sparse_xblock_rescue_built": bool(scope["sparse_xblock_rescue_built"]),
        "sparse_xblock_rescue_error": scope["sparse_xblock_rescue_error"],
        "sparse_xblock_rescue_reason": str(scope["sparse_xblock_rescue_reason"]),
        "sparse_xblock_rescue_assembled_host_fp": bool(
            scope["sparse_xblock_rescue_assembled_host_fp"]
        ),
        "sparse_xblock_rescue_preconditioner_xi": scope[
            "sparse_xblock_rescue_preconditioner_xi"
        ],
        "sparse_xblock_rescue_seed_residual": scope[
            "sparse_xblock_rescue_seed_residual"
        ],
        "sparse_xblock_rescue_seed_improvement_ratio": scope[
            "sparse_xblock_rescue_seed_improvement_ratio"
        ],
        "sparse_xblock_rescue_seed_accept_ratio": scope[
            "sparse_xblock_rescue_seed_accept_ratio"
        ],
        "sparse_xblock_rescue_seed_refine_steps": scope[
            "sparse_xblock_rescue_seed_refine_steps"
        ],
        "sparse_xblock_rescue_seed_refines_performed": scope[
            "sparse_xblock_rescue_seed_refines_performed"
        ],
        "sparse_xblock_rescue_candidate_residual": scope[
            "sparse_xblock_rescue_candidate_residual"
        ],
        "sparse_xblock_rescue_candidate_accepted": bool(
            scope["sparse_xblock_rescue_candidate_accepted"]
        ),
    }


def fp_xblock_global_correction_metadata(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return stable diagnostics for FP x-block global correction."""

    return {
        "fp_xblock_global_correction_allowed": bool(
            scope["fp_xblock_global_correction_allowed"]
        ),
        "fp_xblock_global_correction_attempted": bool(
            scope["fp_xblock_global_correction_attempted"]
        ),
        "fp_xblock_global_correction_accepted": bool(
            scope["fp_xblock_global_correction_accepted"]
        ),
        "fp_xblock_global_correction_reason": str(
            scope["fp_xblock_global_correction_reason"]
        ),
        "fp_xblock_global_correction_error": scope[
            "fp_xblock_global_correction_error"
        ],
        "fp_xblock_global_correction_preconditioner": scope[
            "fp_xblock_global_correction_preconditioner"
        ],
        "fp_xblock_global_correction_steps": scope[
            "fp_xblock_global_correction_steps"
        ],
        "fp_xblock_global_correction_accepted_steps": scope[
            "fp_xblock_global_correction_accepted_steps"
        ],
        "fp_xblock_global_correction_residual_before": scope[
            "fp_xblock_global_correction_residual_before"
        ],
        "fp_xblock_global_correction_residual_after": scope[
            "fp_xblock_global_correction_residual_after"
        ],
        "fp_xblock_global_correction_improvement_ratio": scope[
            "fp_xblock_global_correction_improvement_ratio"
        ],
        "fp_xblock_global_correction_elapsed_s": scope[
            "fp_xblock_global_correction_elapsed_s"
        ],
    }


def fp_xblock_highx_residual_correction_metadata(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return stable diagnostics for FP high-x residual correction."""

    return {
        "fp_xblock_highx_residual_correction_allowed": bool(
            scope["fp_xblock_highx_residual_correction_allowed"]
        ),
        "fp_xblock_highx_residual_correction_attempted": bool(
            scope["fp_xblock_highx_residual_correction_attempted"]
        ),
        "fp_xblock_highx_residual_correction_accepted": bool(
            scope["fp_xblock_highx_residual_correction_accepted"]
        ),
        "fp_xblock_highx_residual_correction_reason": str(
            scope["fp_xblock_highx_residual_correction_reason"]
        ),
        "fp_xblock_highx_residual_correction_error": scope[
            "fp_xblock_highx_residual_correction_error"
        ],
        "fp_xblock_highx_residual_correction_residual_before": scope[
            "fp_xblock_highx_residual_correction_residual_before"
        ],
        "fp_xblock_highx_residual_correction_residual_after": scope[
            "fp_xblock_highx_residual_correction_residual_after"
        ],
        "fp_xblock_highx_residual_correction_improvement_ratio": scope[
            "fp_xblock_highx_residual_correction_improvement_ratio"
        ],
        "fp_xblock_highx_residual_correction_elapsed_s": scope[
            "fp_xblock_highx_residual_correction_elapsed_s"
        ],
        "fp_xblock_highx_residual_correction_direction_count": scope[
            "fp_xblock_highx_residual_correction_direction_count"
        ],
        "fp_xblock_highx_residual_correction_direction_names": tuple(
            scope["fp_xblock_highx_residual_correction_direction_names"]
        ),
    }


def sparse_rescue_tail_metadata(scope: Mapping[str, object]) -> dict[str, object]:
    """Return the combined sparse-rescue tail diagnostics for final metadata."""

    return {
        **sparse_xblock_rescue_metadata(scope),
        **fp_xblock_global_correction_metadata(scope),
        **fp_xblock_highx_residual_correction_metadata(scope),
    }


def xblock_qi_device_preconditioner_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return the x-block QI-device preconditioner diagnostics payload."""

    metadata = scope["qi_device_preconditioner_metadata"]
    stats = scope["qi_device_stats"]
    if not isinstance(metadata, Mapping):
        raise TypeError("qi_device_preconditioner_metadata must be a mapping")
    if not isinstance(stats, Mapping):
        raise TypeError("qi_device_stats must be a mapping")

    out: dict[str, object] = {
        "xblock_qi_device_preconditioner_enabled": bool(
            scope["qi_device_preconditioner_enabled"]
        ),
        "xblock_qi_device_preconditioner_built": bool(
            scope["qi_device_preconditioner_built"]
        ),
        "xblock_qi_device_preconditioner_used": bool(
            scope["qi_device_preconditioner_used"]
        ),
        "xblock_qi_device_preconditioner_used_in_krylov": bool(
            scope["qi_device_preconditioner_used_in_krylov"]
        ),
        "xblock_qi_device_preconditioner_reason": scope[
            "qi_device_preconditioner_reason"
        ],
        "xblock_qi_device_preconditioner_rank": int(
            scope["qi_device_preconditioner_rank"]
        ),
        "xblock_qi_device_preconditioner_candidate_count": int(
            scope["qi_device_preconditioner_candidate_count"]
        ),
        "xblock_qi_device_preconditioner_coarse_operator_shape": scope[
            "qi_device_preconditioner_coarse_shape"
        ],
        "xblock_qi_device_preconditioner_operator_on_basis_shape": scope[
            "qi_device_preconditioner_operator_on_basis_shape"
        ],
        "xblock_qi_device_preconditioner_coarse_operator_norm": float(
            scope["qi_device_preconditioner_coarse_norm"]
        ),
        "xblock_qi_device_preconditioner_operator_on_basis_norm": float(
            scope["qi_device_preconditioner_operator_on_basis_norm"]
        ),
        "xblock_qi_device_preconditioner_residual_before": scope[
            "qi_device_preconditioner_residual_before"
        ],
        "xblock_qi_device_preconditioner_residual_after": scope[
            "qi_device_preconditioner_residual_after"
        ],
        "xblock_qi_device_preconditioner_improvement_ratio": scope[
            "qi_device_preconditioner_improvement_ratio"
        ],
        "xblock_qi_device_preconditioner_metadata": metadata,
        "xblock_qi_device_preconditioner_setup_s": float(
            scope["qi_device_preconditioner_setup_s"]
        ),
        "xblock_qi_device_preconditioner_min_improvement": float(
            scope["qi_device_preconditioner_min_improvement"]
        ),
        "xblock_qi_device_preconditioner_use_in_krylov": bool(
            scope["qi_device_preconditioner_use_in_krylov"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_requested": bool(
            scope["qi_device_augmented_krylov_requested"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_used": bool(
            scope["qi_device_augmented_krylov_used"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_rank": int(
            scope["qi_device_augmented_krylov_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_krylov_reason": scope[
            "qi_device_augmented_krylov_reason"
        ],
        "xblock_qi_device_preconditioner_augmented_krylov_mode": scope[
            "qi_device_augmented_krylov_mode"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_requested": bool(
            scope["qi_device_augmented_seed_requested"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_available": bool(
            scope["qi_device_augmented_seed_available"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_used": bool(
            scope["qi_device_augmented_seed_used"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_rank": int(
            scope["qi_device_augmented_seed_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_max_rank": int(
            scope["qi_device_augmented_seed_max_rank"]
        ),
        "xblock_qi_device_preconditioner_augmented_seed_reason": scope[
            "qi_device_augmented_seed_reason"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_projection_residual_norm": scope[
            "qi_device_augmented_seed_projection_residual"
        ],
        "xblock_qi_device_preconditioner_augmented_seed_labels": scope[
            "qi_device_augmented_seed_labels"
        ],
        "xblock_qi_device_preconditioner_applies": int(stats.get("applies", 0)),
        "xblock_qi_device_preconditioner_operator_krylov_enrichment": bool(
            metadata.get("operator_krylov_enrichment_enabled", False)
        ),
        "xblock_qi_device_preconditioner_coarse_reuse": bool(
            metadata.get("multilevel_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_enrichment": bool(
            metadata.get("residual_snapshot_enrichment_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation": bool(
            metadata.get("residual_snapshot_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_rank": int(
            metadata.get("residual_snapshot_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_candidate_count": int(
            metadata.get("residual_snapshot_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_snapshot_residual_equation_group_count": int(
            metadata.get("residual_snapshot_residual_equation_group_count", 0)
        ),
        "xblock_qi_device_preconditioner_multilevel_residual_equation": bool(
            metadata.get("multilevel_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_multilevel_residual_equation_solver": metadata.get(
            "multilevel_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation": bool(
            metadata.get("global_moment_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_solver": metadata.get(
            "global_moment_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_rank": int(
            metadata.get("global_moment_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_candidate_count": int(
            metadata.get("global_moment_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_global_moment_residual_equation_condition_estimate": float(
            metadata.get("global_moment_residual_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation": bool(
            metadata.get("residual_galerkin_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_solver": metadata.get(
            "residual_galerkin_equation_solver"
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_rank": int(
            metadata.get("residual_galerkin_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_candidate_count": int(
            metadata.get("residual_galerkin_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_stage_count": int(
            metadata.get("residual_galerkin_equation_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_galerkin_equation_condition_estimate": float(
            metadata.get("residual_galerkin_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation": bool(
            metadata.get("phase_space_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_max_rank": int(
            metadata.get(
                "phase_space_residual_equation_max_rank_requested",
                metadata.get("phase_space_residual_equation_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_solver": metadata.get(
            "phase_space_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_rank": int(
            metadata.get("phase_space_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_candidate_count": int(
            metadata.get("phase_space_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_stage_count": int(
            metadata.get("phase_space_residual_equation_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_condition_estimate": float(
            metadata.get("phase_space_residual_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_residual_before": float(
            metadata.get("phase_space_residual_equation_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_residual_after": float(
            metadata.get("phase_space_residual_equation_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_global": bool(
            metadata.get("phase_space_residual_equation_include_global", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_trapped_boundary_fraction": float(
            metadata.get(
                "phase_space_residual_equation_trapped_boundary_fraction",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_radial": bool(
            metadata.get("phase_space_residual_equation_include_radial", False)
        ),
        "xblock_qi_device_preconditioner_phase_space_residual_equation_include_species": bool(
            metadata.get("phase_space_residual_equation_include_species", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse": bool(
            metadata.get("residual_region_bounce_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank": int(
            metadata.get(
                "residual_region_bounce_coarse_max_rank_requested",
                metadata.get("residual_region_bounce_coarse_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_solver": metadata.get(
            "residual_region_bounce_coarse_solver"
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_rank": int(
            metadata.get("residual_region_bounce_coarse_rank", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_candidate_count": int(
            metadata.get("residual_region_bounce_coarse_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_stage_count": int(
            metadata.get("residual_region_bounce_coarse_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_condition_estimate": float(
            metadata.get("residual_region_bounce_coarse_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_before": float(
            metadata.get("residual_region_bounce_coarse_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_residual_after": float(
            metadata.get("residual_region_bounce_coarse_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_global": bool(
            metadata.get("residual_region_bounce_coarse_include_global", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_radial": bool(
            metadata.get("residual_region_bounce_coarse_include_radial", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_include_species": bool(
            metadata.get("residual_region_bounce_coarse_include_species", False)
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_bounce_boundary": float(
            metadata.get("residual_region_bounce_coarse_bounce_boundary", float("nan"))
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_min_region_energy_fraction": float(
            metadata.get(
                "residual_region_bounce_coarse_min_region_energy_fraction",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_residual_region_bounce_coarse_region_bands": metadata.get(
            "residual_region_bounce_coarse_region_bands"
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse": bool(
            metadata.get("active_pattern_coarse_enabled", False)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_max_rank": int(
            metadata.get(
                "active_pattern_coarse_max_rank_requested",
                metadata.get("active_pattern_coarse_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_max_candidates": int(
            metadata.get("active_pattern_coarse_max_candidates_requested", 0) or 0
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_solver": metadata.get(
            "active_pattern_coarse_solver"
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_rank": int(
            metadata.get("active_pattern_coarse_rank", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_candidate_count": int(
            metadata.get("active_pattern_coarse_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_stage_count": int(
            metadata.get("active_pattern_coarse_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_condition_estimate": float(
            metadata.get("active_pattern_coarse_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_residual_before": float(
            metadata.get("active_pattern_coarse_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_residual_after": float(
            metadata.get("active_pattern_coarse_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_include_global": bool(
            metadata.get("active_pattern_coarse_include_global", False)
        ),
        "xblock_qi_device_preconditioner_active_pattern_coarse_min_chunk_energy_fraction": float(
            metadata.get("active_pattern_coarse_min_chunk_energy_fraction", float("nan"))
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation": bool(
            metadata.get("block_schur_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_rank": int(
            metadata.get("block_schur_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_candidate_count": int(
            metadata.get("block_schur_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_equation_group_count": int(
            metadata.get("block_schur_residual_equation_group_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation": bool(
            metadata.get("coupled_residual_equation_enabled", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_max_rank": int(
            metadata.get(
                "coupled_residual_equation_max_rank_requested",
                metadata.get("coupled_residual_equation_max_rank", 0),
            )
            or 0
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_rank": int(
            metadata.get("coupled_residual_equation_rank", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_candidate_count": int(
            metadata.get("coupled_residual_equation_candidate_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_source_stage_count": int(
            metadata.get("coupled_residual_equation_source_stage_count", 0)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_source_stage_ranks": metadata.get(
            "coupled_residual_equation_source_stage_ranks"
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_solver": metadata.get(
            "coupled_residual_equation_solver"
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_include_flat": bool(
            metadata.get("coupled_residual_equation_include_flat", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_min_relative_improvement": float(
            metadata.get(
                "coupled_residual_equation_min_relative_improvement_requested",
                float("nan"),
            )
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_install_in_krylov_on_reject": bool(
            metadata.get(
                "coupled_residual_equation_install_in_krylov_on_reject_requested",
                False,
            )
        ),
        "xblock_qi_device_preconditioner_seed_probe_accepted": bool(
            metadata.get("seed_probe_accepted", False)
        ),
        "xblock_qi_device_preconditioner_installed_in_krylov_after_seed_reject": bool(
            metadata.get("installed_in_krylov_after_seed_reject", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_condition_estimate": float(
            metadata.get("coupled_residual_equation_condition_estimate", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_residual_before": float(
            metadata.get("coupled_residual_equation_residual_before", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_residual_after": float(
            metadata.get("coupled_residual_equation_residual_after", float("inf"))
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_accepted": bool(
            metadata.get("coupled_residual_equation_accepted", False)
        ),
        "xblock_qi_device_preconditioner_coupled_residual_equation_reason": metadata.get(
            "coupled_residual_equation_reason"
        ),
        "xblock_qi_device_preconditioner_block_schur_residual_enrichment": bool(
            metadata.get("block_schur_residual_enrichment_enabled", False)
        ),
    }
    return out


def xblock_qi_deflated_preconditioner_diagnostics(
    scope: Mapping[str, object],
) -> dict[str, object]:
    """Return the x-block QI residual-deflation preconditioner diagnostics."""

    metadata = scope["qi_deflated_preconditioner_metadata"]
    stats = scope["qi_deflated_stats"]
    if not isinstance(metadata, Mapping):
        raise TypeError("qi_deflated_preconditioner_metadata must be a mapping")
    if not isinstance(stats, Mapping):
        raise TypeError("qi_deflated_stats must be a mapping")

    return {
        "xblock_qi_deflated_preconditioner_enabled": bool(
            scope["qi_deflated_preconditioner_enabled"]
        ),
        "xblock_qi_deflated_preconditioner_built": bool(
            scope["qi_deflated_preconditioner_built"]
        ),
        "xblock_qi_deflated_preconditioner_used": bool(
            scope["qi_deflated_preconditioner_used"]
        ),
        "xblock_qi_deflated_preconditioner_reason": scope[
            "qi_deflated_preconditioner_reason"
        ],
        "xblock_qi_deflated_preconditioner_rank": int(
            scope["qi_deflated_preconditioner_rank"]
        ),
        "xblock_qi_deflated_preconditioner_candidate_count": int(
            scope["qi_deflated_preconditioner_candidate_count"]
        ),
        "xblock_qi_deflated_preconditioner_residual_before": scope[
            "qi_deflated_preconditioner_residual_before"
        ],
        "xblock_qi_deflated_preconditioner_residual_after": scope[
            "qi_deflated_preconditioner_residual_after"
        ],
        "xblock_qi_deflated_preconditioner_improvement_ratio": scope[
            "qi_deflated_preconditioner_improvement_ratio"
        ],
        "xblock_qi_deflated_preconditioner_metadata": metadata,
        "xblock_qi_deflated_preconditioner_setup_s": float(
            scope["qi_deflated_preconditioner_setup_s"]
        ),
        "xblock_qi_deflated_preconditioner_applies": int(stats.get("applies", 0)),
        "xblock_qi_deflated_preconditioner_local_applies": int(
            stats.get("local_applies", 0)
        ),
        "xblock_qi_deflated_preconditioner_cycles": int(
            metadata.get("correction_cycles", 0)
        ),
        "xblock_qi_deflated_preconditioner_seed_solver": metadata.get("seed_solver"),
        "xblock_qi_deflated_preconditioner_cycle_residual_history": metadata.get(
            "cycle_residual_history",
            (),
        ),
        "xblock_qi_deflated_preconditioner_cycle_coefficients": metadata.get(
            "cycle_coefficients",
            (),
        ),
        "xblock_qi_deflated_preconditioner_use_in_krylov": bool(
            scope["qi_deflated_preconditioner_used_in_krylov"]
        ),
    }


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
) -> XBlockKrylovMatvecSetup:
    """Build reduced/full matvec closures and progress accounting."""

    progress_every_env = _env_value(env, "SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY")
    try:
        progress_every = int(progress_every_env) if progress_every_env else 25
    except ValueError:
        progress_every = 25
    progress_every = max(0, int(progress_every))
    mv_count = MatvecCounter(0)

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
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres active-DOF reduction "
                f"enabled (size={int(linear_size)}/{int(op.total_size)})",
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
        mv_count.increment()
        if emit is not None and progress_every > 0 and int(mv_count) % progress_every == 0:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: xblock_sparse_pc_gmres "
                f"matvecs={int(mv_count)} elapsed_s={float(elapsed_s()):.3f}",
            )
        return matvec_no_count(v)

    return XBlockKrylovMatvecSetup(
        progress_every=int(progress_every),
        mv_count=mv_count,
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
