"""Device-resident two-level preconditioner for RHSMode=1 QI lanes.

This module is the first production-shaped primitive for the true device-QI
research lane.  It keeps the preconditioner state as JAX arrays plus static
metadata:

``M^{-1} r = S_local^{-1} r + P_c A_c^{-1} R_c (r - A S_local^{-1} r)``.

``S_local`` is a CSR-backed device Jacobi/stationary smoother when a device CSR
operator is available.  For larger QI seeds the module can also run a coarse-only
matrix-free path: it builds only ``A Q`` by applying a JAX matvec to rank-gated
coarse vectors, avoiding full CSR materialization.  The opt-in operator-Krylov
and operator-action coarse architectures expand the correction space with
rank-gated ``orth([Q, r, A r, ...])`` or ``{Q, A Q, A^2 Q, ...}`` columns and
then reuse the final ``A Q_aug`` action for all subsequent coarse solves.  Both
variants keep the timed apply path free of SciPy, Python callbacks, and host
factors.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any

import jax.numpy as jnp
import jax
import numpy as np

from .rhs1_device_operator import DeviceCSR
from .rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    RHS1QICoarseBasisMetadata,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)
from .rhs1_qi_device_smoother import (
    RHS1QIDeviceJacobiSmoother,
    build_rhs1_qi_device_jacobi_smoother,
)
from .rhs1_qi_multilevel_coarse import (
    RHS1QIMultilevelCoarseConfig,
    build_rhs1_qi_multilevel_coarse_basis,
    build_rhs1_qi_multilevel_coarse_candidates,
    build_rhs1_qi_multilevel_residual_level_bases,
)
from .rhs1_qi_active_pattern_coarse import (
    RHS1QIActivePatternCoarseConfig,
    build_rhs1_qi_active_pattern_coarse_basis,
)
from .rhs1_qi_phase_space_coarse import (
    RHS1QIPhaseSpaceCoarseConfig,
    build_rhs1_qi_phase_space_coarse_basis,
)
from .rhs1_qi_residual_galerkin import (
    RHS1QIResidualGalerkinConfig,
    setup_rhs1_qi_residual_galerkin,
)
from .rhs1_qi_residual_region_coarse import (
    RHS1QIResidualRegionCoarseConfig,
    build_rhs1_qi_residual_region_coarse_basis,
)

ArrayLike = Any


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerConfig:
    """Static controls for a device QI two-level preconditioner."""

    regularization_rcond: float = 1.0e-12
    damping: float = 1.0
    coarse_solver: str = "action_lstsq"
    composition: str = "multiplicative"
    basis_rtol: float = 1.0e-10
    basis_atol: float = 0.0
    max_rank: int | None = None
    jacobi_damping: float = 0.7
    jacobi_sweeps: int = 1
    jacobi_step_policy: str = "stationary"
    jacobi_diagonal_floor: float = 1.0e-14
    jacobi_require_all_diagonal: bool = True
    local_smoother_kind: str = "auto"
    matrix_free_smoother_sweeps: int = 1
    matrix_free_smoother_damping: float = 1.0
    matrix_free_smoother_step_policy: str = "residual_minimizing"
    matrix_free_smoother_alpha_clip: float = 10.0
    matrix_free_smoother_min_denominator: float = 1.0e-300
    matrix_free_block_smoother_max_groups: int = 32
    matrix_free_block_smoother_include_tail: bool = True
    matrix_free_block_smoother_rcond: float = 1.0e-12
    matrix_free_block_smoother_grouping: str = "contiguous"
    residual_enrichment: bool = False
    residual_enrichment_depth: int = 0
    residual_enrichment_include_residual: bool = True
    recycle_enrichment: bool = False
    recycle_enrichment_cycles: int = 0
    operator_krylov_enrichment: bool = False
    operator_krylov_depth: int = 0
    adjoint_krylov_enrichment: bool = False
    adjoint_krylov_depth: int = 0
    adjoint_krylov_transpose_source: str = "autodiff"
    operator_action_enrichment: bool = False
    operator_action_enrichment_depth: int = 0
    multilevel_coarse: bool = False
    multilevel_max_levels: int = 3
    multilevel_aggregate_factor: int = 2
    multilevel_max_rank: int | None = None
    multilevel_max_angular_mode: int = 1
    multilevel_max_radial_degree: int = 2
    multilevel_max_pitch_degree: int = 0
    multilevel_current_moments: bool = False
    multilevel_species_current_moments: bool = True
    multilevel_radial_current_moments: bool = True
    multilevel_tail_constraint_moments: bool = True
    multilevel_current_max_pitch_degree: int = 1
    multilevel_residual_equation: bool = False
    multilevel_residual_equation_max_level_rank: int = 16
    multilevel_residual_equation_order: str = "coarse_to_fine"
    multilevel_residual_equation_solver: str = "action_lstsq"
    multilevel_residual_equation_include_global: bool = True
    global_moment_residual_equation: bool = False
    global_moment_residual_equation_max_rank: int = 16
    global_moment_residual_equation_solver: str = "galerkin"
    global_moment_residual_equation_include_profile: bool = True
    global_moment_residual_equation_include_current: bool = True
    global_moment_residual_equation_include_tail: bool = True
    residual_galerkin_equation: bool = False
    residual_galerkin_equation_max_stages: int = 3
    residual_galerkin_equation_max_stage_rank: int = 4
    residual_galerkin_equation_max_rank: int = 24
    residual_galerkin_equation_solver: str = "action_lstsq"
    residual_galerkin_equation_include_global_residual: bool = True
    residual_galerkin_equation_include_block_residuals: bool = True
    residual_galerkin_equation_include_operator_images: bool = False
    phase_space_residual_equation: bool = False
    phase_space_residual_equation_max_rank: int = 24
    phase_space_residual_equation_solver: str = "action_lstsq"
    phase_space_residual_equation_include_global: bool = False
    phase_space_residual_equation_trapped_boundary_fraction: float = 0.35
    phase_space_residual_equation_include_trapped: bool = True
    phase_space_residual_equation_include_passing: bool = True
    phase_space_residual_equation_include_boundary: bool = True
    phase_space_residual_equation_include_even: bool = True
    phase_space_residual_equation_include_odd: bool = True
    phase_space_residual_equation_include_radial: bool = True
    phase_space_residual_equation_include_species: bool = True
    residual_region_bounce_coarse: bool = False
    residual_region_bounce_coarse_max_rank: int = 32
    residual_region_bounce_coarse_max_candidates: int = 48
    residual_region_bounce_coarse_solver: str = "action_lstsq"
    residual_region_bounce_coarse_include_global: bool = True
    residual_region_bounce_coarse_include_radial: bool = True
    residual_region_bounce_coarse_include_species: bool = True
    residual_region_bounce_coarse_trapped_boundary_fraction: float = 0.35
    residual_region_bounce_coarse_min_region_energy_fraction: float = 1.0e-2
    residual_region_bounce_coarse_region_bands: str = "bounce,trapped,passing"
    active_pattern_coarse: bool = False
    active_pattern_coarse_max_rank: int = 32
    active_pattern_coarse_max_candidates: int = 64
    active_pattern_coarse_solver: str = "action_lstsq"
    active_pattern_coarse_min_chunk_energy_fraction: float = 1.0e-2
    active_pattern_coarse_include_global: bool = True
    active_pattern_coarse_include_block_pitch: bool = True
    active_pattern_coarse_include_block_angular: bool = True
    active_pattern_coarse_include_radial_pitch: bool = True
    active_pattern_coarse_include_radial_angular: bool = True
    active_pattern_coarse_include_block: bool = True
    active_pattern_coarse_include_radial: bool = True
    active_pattern_coarse_include_species: bool = True
    block_schur_residual_equation: bool = False
    block_schur_residual_equation_max_rank: int = 24
    block_schur_residual_equation_include_global: bool = False
    block_schur_residual_equation_include_blocks: bool = True
    block_schur_residual_equation_include_aggregates: bool = True
    residual_snapshot_enrichment: bool = False
    residual_snapshot_max_rank: int = 24
    residual_snapshot_include_primal: bool = True
    residual_snapshot_use_adjoint: bool = False
    residual_snapshot_include_global: bool = False
    residual_snapshot_include_blocks: bool = True
    residual_snapshot_include_aggregates: bool = True
    residual_snapshot_residual_equation: bool = False
    residual_snapshot_residual_equation_max_rank: int = 24
    residual_snapshot_residual_equation_solver: str = "action_lstsq"
    residual_snapshot_residual_equation_include_global: bool = False
    block_schur_residual_enrichment: bool = False
    block_schur_residual_max_rank: int = 24
    block_schur_residual_include_global: bool = False
    block_schur_residual_include_blocks: bool = True
    block_schur_residual_include_aggregates: bool = True


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerMetadata:
    """JSON-friendly diagnostics for the device-QI preconditioner state."""

    shape: tuple[int, int]
    nnz: int
    rank: int
    operator_source: str
    coarse_operator_shape: tuple[int, int]
    operator_on_basis_shape: tuple[int, int]
    coarse_operator_norm: float
    operator_on_basis_norm: float
    regularization_rcond: float
    damping: float
    coarse_solver: str
    composition: str
    local_smoother_kind: str
    local_smoother_reason: str
    device_resident: bool
    host_fallback_used: bool
    host_callback_free: bool
    operator_metadata_keys: tuple[str, ...]
    geometry_metadata_keys: tuple[str, ...]
    accepted_basis_labels: tuple[str, ...]
    residual_enrichment_enabled: bool
    residual_enrichment_depth: int
    residual_enrichment_candidate_count: int
    recycle_enrichment_enabled: bool
    recycle_enrichment_cycles: int
    recycle_enrichment_candidate_count: int
    operator_krylov_enrichment_enabled: bool
    operator_krylov_depth: int
    operator_krylov_candidate_count: int
    adjoint_krylov_enrichment_enabled: bool
    adjoint_krylov_depth: int
    adjoint_krylov_candidate_count: int
    adjoint_krylov_transpose_source: str
    operator_action_enrichment_enabled: bool
    operator_action_enrichment_depth: int
    operator_action_enrichment_candidate_count: int
    multilevel_coarse_enabled: bool
    multilevel_coarse_level_count: int
    multilevel_coarse_candidate_count: int
    multilevel_coarse_rank: int
    multilevel_residual_equation_enabled: bool
    multilevel_residual_equation_stage_count: int
    multilevel_residual_equation_rank: int
    multilevel_residual_equation_stage_ranks: tuple[int, ...]
    multilevel_residual_equation_order: str
    multilevel_residual_equation_solver: str
    multilevel_residual_equation_include_global: bool
    global_moment_residual_equation_enabled: bool
    global_moment_residual_equation_candidate_count: int
    global_moment_residual_equation_rank: int
    global_moment_residual_equation_stage_ranks: tuple[int, ...]
    global_moment_residual_equation_solver: str
    global_moment_residual_equation_include_profile: bool
    global_moment_residual_equation_include_current: bool
    global_moment_residual_equation_include_tail: bool
    global_moment_residual_equation_condition_estimate: float
    residual_galerkin_equation_enabled: bool
    residual_galerkin_equation_candidate_count: int
    residual_galerkin_equation_rank: int
    residual_galerkin_equation_stage_count: int
    residual_galerkin_equation_stage_ranks: tuple[int, ...]
    residual_galerkin_equation_solver: str
    residual_galerkin_equation_condition_estimate: float
    residual_galerkin_equation_residual_before: float
    residual_galerkin_equation_residual_after: float
    phase_space_residual_equation_enabled: bool
    phase_space_residual_equation_candidate_count: int
    phase_space_residual_equation_rank: int
    phase_space_residual_equation_stage_count: int
    phase_space_residual_equation_stage_ranks: tuple[int, ...]
    phase_space_residual_equation_max_rank: int
    phase_space_residual_equation_solver: str
    phase_space_residual_equation_include_global: bool
    phase_space_residual_equation_trapped_boundary_fraction: float
    phase_space_residual_equation_include_trapped: bool
    phase_space_residual_equation_include_passing: bool
    phase_space_residual_equation_include_boundary: bool
    phase_space_residual_equation_include_even: bool
    phase_space_residual_equation_include_odd: bool
    phase_space_residual_equation_include_radial: bool
    phase_space_residual_equation_include_species: bool
    phase_space_residual_equation_condition_estimate: float
    phase_space_residual_equation_residual_before: float
    phase_space_residual_equation_residual_after: float
    residual_region_bounce_coarse_enabled: bool
    residual_region_bounce_coarse_candidate_count: int
    residual_region_bounce_coarse_rank: int
    residual_region_bounce_coarse_stage_count: int
    residual_region_bounce_coarse_stage_ranks: tuple[int, ...]
    residual_region_bounce_coarse_max_rank_requested: int
    residual_region_bounce_coarse_solver: str
    residual_region_bounce_coarse_condition_estimate: float
    residual_region_bounce_coarse_residual_before: float
    residual_region_bounce_coarse_residual_after: float
    residual_region_bounce_coarse_include_global: bool
    residual_region_bounce_coarse_include_radial: bool
    residual_region_bounce_coarse_include_species: bool
    residual_region_bounce_coarse_bounce_boundary: float
    residual_region_bounce_coarse_min_region_energy_fraction: float
    residual_region_bounce_coarse_region_bands: str
    active_pattern_coarse_enabled: bool
    active_pattern_coarse_candidate_count: int
    active_pattern_coarse_rank: int
    active_pattern_coarse_stage_count: int
    active_pattern_coarse_stage_ranks: tuple[int, ...]
    active_pattern_coarse_max_rank_requested: int
    active_pattern_coarse_max_candidates_requested: int
    active_pattern_coarse_solver: str
    active_pattern_coarse_condition_estimate: float
    active_pattern_coarse_residual_before: float
    active_pattern_coarse_residual_after: float
    active_pattern_coarse_min_chunk_energy_fraction: float
    active_pattern_coarse_include_global: bool
    active_pattern_coarse_include_block_pitch: bool
    active_pattern_coarse_include_block_angular: bool
    active_pattern_coarse_include_radial_pitch: bool
    active_pattern_coarse_include_radial_angular: bool
    active_pattern_coarse_include_block: bool
    active_pattern_coarse_include_radial: bool
    active_pattern_coarse_include_species: bool
    block_schur_residual_equation_enabled: bool
    block_schur_residual_equation_group_count: int
    block_schur_residual_equation_candidate_count: int
    block_schur_residual_equation_rank: int
    block_schur_residual_equation_stage_ranks: tuple[int, ...]
    block_schur_residual_equation_include_global: bool
    block_schur_residual_equation_include_blocks: bool
    block_schur_residual_equation_include_aggregates: bool
    residual_snapshot_enrichment_enabled: bool
    residual_snapshot_candidate_count: int
    residual_snapshot_rank: int
    residual_snapshot_group_count: int
    residual_snapshot_include_primal: bool
    residual_snapshot_use_adjoint: bool
    residual_snapshot_residual_equation_enabled: bool
    residual_snapshot_residual_equation_group_count: int
    residual_snapshot_residual_equation_candidate_count: int
    residual_snapshot_residual_equation_rank: int
    residual_snapshot_residual_equation_stage_ranks: tuple[int, ...]
    residual_snapshot_residual_equation_solver: str
    residual_snapshot_residual_equation_include_global: bool
    block_schur_residual_enrichment_enabled: bool
    block_schur_residual_candidate_count: int
    block_schur_residual_rank: int
    block_schur_residual_group_count: int
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return a plain mapping suitable for solver traces and JSON."""

        payload = asdict(self)
        payload["shape"] = tuple(int(v) for v in self.shape)
        payload["coarse_operator_shape"] = tuple(int(v) for v in self.coarse_operator_shape)
        payload["operator_on_basis_shape"] = tuple(int(v) for v in self.operator_on_basis_shape)
        return payload


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerState:
    """Reusable pure-JAX local-plus-coarse QI preconditioner action."""

    operator: DeviceCSR | None
    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    shape: tuple[int, int]
    local_smoother: (
        RHS1QIDeviceJacobiSmoother
        | "RHS1QIMatrixFreeResidualSmoother"
        | "RHS1QIMatrixFreeProjectedResidualSmoother"
        | None
    )
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    residual_equation_bases: tuple[RHS1QICoarseBasis, ...]
    residual_equation_operator_on_bases: tuple[ArrayLike, ...]
    residual_equation_coarse_operators: tuple[ArrayLike, ...]
    residual_equation_stage_solvers: tuple[str, ...]
    metadata: RHS1QIDevicePreconditionerMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small coarse problem for the current residual."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        if self.metadata.coarse_solver == "action_lstsq":
            return _regularized_least_squares(
                self.operator_on_basis,
                residual_vec,
                rcond=float(self.metadata.regularization_rcond),
            )
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _regularized_least_squares(
            self.coarse_operator,
            projected,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one field-split/two-level correction to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match operator rows {self.shape[0]}"
            )
        if self.local_smoother is None:
            local = jnp.zeros_like(residual_vec)
        else:
            local = jnp.asarray(self.local_smoother.apply(residual_vec), dtype=residual_vec.dtype).reshape((-1,))
        residual_equation_rank = (
            int(self.metadata.multilevel_residual_equation_rank)
            + int(self.metadata.global_moment_residual_equation_rank)
            + int(self.metadata.residual_galerkin_equation_rank)
            + int(self.metadata.phase_space_residual_equation_rank)
            + int(self.metadata.residual_region_bounce_coarse_rank)
            + int(self.metadata.active_pattern_coarse_rank)
            + int(self.metadata.block_schur_residual_equation_rank)
            + int(self.metadata.residual_snapshot_residual_equation_rank)
        )
        if int(self.metadata.rank) <= 0 and residual_equation_rank <= 0:
            return float(self.metadata.damping) * local

        if self.metadata.composition == "multiplicative" and self.local_smoother is not None:
            coarse_input = residual_vec - jnp.asarray(self.operator_matvec(local), dtype=residual_vec.dtype).reshape((-1,))
        else:
            coarse_input = residual_vec
        if (
            bool(self.metadata.multilevel_residual_equation_enabled)
            or bool(self.metadata.global_moment_residual_equation_enabled)
            or bool(self.metadata.residual_galerkin_equation_enabled)
            or bool(self.metadata.phase_space_residual_equation_enabled)
            or bool(self.metadata.residual_region_bounce_coarse_enabled)
            or bool(self.metadata.active_pattern_coarse_enabled)
            or bool(self.metadata.block_schur_residual_equation_enabled)
            or bool(self.metadata.residual_snapshot_residual_equation_enabled)
        ):
            coarse = self.solve_coarse_residual_equation(coarse_input)
        else:
            coefficients = self.solve_coarse(coarse_input)
            coarse = jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype) @ coefficients
        return float(self.metadata.damping) * (local + coarse)

    def solve_coarse_residual_equation(self, residual: ArrayLike) -> ArrayLike:
        """Apply a bounded multilevel residual-equation cascade.

        Each stage solves the configured residual equation for the residual
        left by previous stages.  The Galerkin variant uses cached
        ``Q_l^T A Q_l`` built from setup-time ``A Q_l`` and keeps apply-time
        work as pure JAX array operations.
        """

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        for basis, action, coarse_operator, solver in zip(
            self.residual_equation_bases,
            self.residual_equation_operator_on_bases,
            self.residual_equation_coarse_operators,
            self.residual_equation_stage_solvers,
            strict=True,
        ):
            if int(basis.metadata.rank) <= 0:
                continue
            if solver == "galerkin":
                coefficients = _regularized_projected_solve(
                    coarse_operator,
                    jnp.conjugate(jnp.asarray(basis.vectors, dtype=residual_vec.dtype)).T @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = _regularized_least_squares(
                    action,
                    remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            stage_update = jnp.asarray(basis.vectors, dtype=residual_vec.dtype) @ coefficients
            correction = correction + stage_update
            remaining = remaining - jnp.asarray(action, dtype=residual_vec.dtype) @ coefficients
        include_flat_coarse = (
            bool(self.metadata.multilevel_residual_equation_enabled)
            and bool(self.metadata.multilevel_residual_equation_include_global)
        ) or (
            bool(self.metadata.global_moment_residual_equation_enabled)
        ) or (
            bool(self.metadata.residual_galerkin_equation_enabled)
        ) or (
            bool(self.metadata.phase_space_residual_equation_enabled)
            and bool(self.metadata.phase_space_residual_equation_include_global)
        ) or (
            bool(self.metadata.residual_region_bounce_coarse_enabled)
            and bool(self.metadata.residual_region_bounce_coarse_include_global)
        ) or (
            bool(self.metadata.active_pattern_coarse_enabled)
            and bool(self.metadata.active_pattern_coarse_include_global)
        ) or (
            bool(self.metadata.block_schur_residual_equation_enabled)
            and bool(self.metadata.block_schur_residual_equation_include_global)
        ) or (
            bool(self.metadata.residual_snapshot_residual_equation_enabled)
            and bool(self.metadata.residual_snapshot_residual_equation_include_global)
        )
        if include_flat_coarse and int(self.metadata.rank) > 0:
            if (
                (
                    bool(self.metadata.multilevel_residual_equation_enabled)
                    and self.metadata.multilevel_residual_equation_solver == "galerkin"
                )
                or (
                    bool(self.metadata.global_moment_residual_equation_enabled)
                    and self.metadata.global_moment_residual_equation_solver == "galerkin"
                )
                or (
                    bool(self.metadata.residual_snapshot_residual_equation_enabled)
                    and self.metadata.residual_snapshot_residual_equation_solver == "galerkin"
                )
                or (
                    bool(self.metadata.phase_space_residual_equation_enabled)
                    and self.metadata.phase_space_residual_equation_solver == "galerkin"
                )
                or (
                    bool(self.metadata.residual_region_bounce_coarse_enabled)
                    and self.metadata.residual_region_bounce_coarse_solver == "galerkin"
                )
                or (
                    bool(self.metadata.active_pattern_coarse_enabled)
                    and self.metadata.active_pattern_coarse_solver == "galerkin"
                )
            ):
                coefficients = _regularized_projected_solve(
                    self.coarse_operator,
                    jnp.conjugate(jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype)).T @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = self.solve_coarse(remaining)
            correction = correction + jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype) @ coefficients
        return correction

    def as_preconditioner(self):
        """Return a callable for Krylov hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIDevicePreconditionerProbe:
    """Fail-closed true-residual probe for a device-QI candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIDevicePreconditionerMetadata
    cycles: int = 0
    residual_history: tuple[float, ...] = ()
    step_history: tuple[float, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
            "cycles": int(self.cycles),
            "residual_history": tuple(float(value) for value in self.residual_history),
            "step_history": tuple(float(value) for value in self.step_history),
        }


@dataclass(frozen=True)
class RHS1QIDeviceAugmentedSeedProbe:
    """Accepted device-QI seed correction space for augmented Krylov solves."""

    solution: ArrayLike
    probe: RHS1QIDevicePreconditionerProbe
    augmentation_basis: ArrayLike
    operator_on_augmentation: ArrayLike
    rank: int
    reason: str
    accepted_labels: tuple[str, ...]
    projection_residual_norm: float | None = None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly augmentation diagnostics without array payloads."""

        return {
            "rank": int(self.rank),
            "reason": self.reason,
            "accepted_labels": tuple(str(label) for label in self.accepted_labels),
            "projection_residual_norm": (
                None if self.projection_residual_norm is None else float(self.projection_residual_norm)
            ),
            "probe": self.probe.to_dict(),
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeResidualSmootherMetadata:
    """Diagnostics for a bounded matrix-free local smoother."""

    shape: tuple[int, int]
    sweeps: int
    damping: float
    step_policy: str
    alpha_clip: float
    min_denominator: float
    device_resident: bool
    source: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly matrix-free smoother diagnostics."""

        return {
            "shape": tuple(int(v) for v in self.shape),
            "sweeps": int(self.sweeps),
            "damping": float(self.damping),
            "step_policy": self.step_policy,
            "alpha_clip": float(self.alpha_clip),
            "min_denominator": float(self.min_denominator),
            "device_resident": bool(self.device_resident),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeResidualSmoother:
    """Pure-JAX matrix-free local smoother for large QI seeds.

    Each sweep applies a Richardson/minimal-residual step using the current
    residual as the search direction.  This is deliberately bounded: it is a
    local preconditioner component, not an unbounded Krylov solve.
    """

    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    metadata: RHS1QIMatrixFreeResidualSmootherMetadata

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply fixed-count residual-polynomial smoothing to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.metadata.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.metadata.shape[0]}"
            )
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        min_denominator = jnp.asarray(max(0.0, float(self.metadata.min_denominator)), dtype=residual_vec.dtype)
        alpha_clip = jnp.asarray(max(0.0, float(self.metadata.alpha_clip)), dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            direction = remaining
            action = jnp.asarray(self.operator_matvec(direction), dtype=residual_vec.dtype).reshape((-1,))
            if self.metadata.step_policy == "residual_minimizing":
                numerator = jnp.real(jnp.vdot(action, remaining))
                denominator = jnp.real(jnp.vdot(action, action))
                valid = (
                    jnp.isfinite(numerator)
                    & jnp.isfinite(denominator)
                    & (denominator > min_denominator)
                )
                raw_alpha = numerator / jnp.where(valid, denominator, jnp.asarray(1.0, dtype=denominator.dtype))
                if float(self.metadata.alpha_clip) > 0.0:
                    raw_alpha = jnp.clip(raw_alpha, -alpha_clip, alpha_clip)
                alpha = jnp.where(valid & jnp.isfinite(raw_alpha), raw_alpha, jnp.asarray(0.0, dtype=raw_alpha.dtype))
            elif self.metadata.step_policy == "stationary":
                alpha = jnp.asarray(1.0, dtype=residual_vec.dtype)
            else:
                raise ValueError(f"unsupported matrix-free smoother step policy {self.metadata.step_policy!r}")
            step_scale = damping * alpha
            correction = correction + step_scale * direction
            remaining = remaining - step_scale * action
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for local-smoother hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIMatrixFreeProjectedResidualSmootherMetadata:
    """Diagnostics for projected block residual smoothing."""

    shape: tuple[int, int]
    group_slices: tuple[tuple[int, int], ...]
    group_partitions: tuple[tuple[tuple[int, int], ...], ...]
    sweeps: int
    damping: float
    regularization_rcond: float
    block_count: int
    group_count: int
    max_groups: int
    include_tail: bool
    grouping: str
    device_resident: bool
    source: str
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly projected smoother diagnostics."""

        return {
            "shape": tuple(int(v) for v in self.shape),
            "group_slices": tuple((int(start), int(stop)) for start, stop in self.group_slices),
            "group_partitions": tuple(
                tuple((int(start), int(stop)) for start, stop in partition)
                for partition in self.group_partitions
            ),
            "sweeps": int(self.sweeps),
            "damping": float(self.damping),
            "regularization_rcond": float(self.regularization_rcond),
            "block_count": int(self.block_count),
            "group_count": int(self.group_count),
            "max_groups": int(self.max_groups),
            "include_tail": bool(self.include_tail),
            "grouping": self.grouping,
            "device_resident": bool(self.device_resident),
            "source": self.source,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RHS1QIMatrixFreeProjectedResidualSmoother:
    """Pure-JAX projected block/angular/radial residual smoother.

    Each sweep splits the current residual into structural x/species block
    pieces, optional x/species aggregate pieces, applies the matrix-free
    operator to those pieces, and solves the small problem
    ``min_c ||r - A D c||_2``.  The lifted correction ``D c`` is a bounded
    additive-Schwarz-like local action that keeps all angular content inside
    each selected partition while avoiding full CSR materialization.
    """

    operator_matvec: Callable[[ArrayLike], ArrayLike]
    dtype: Any
    metadata: RHS1QIMatrixFreeProjectedResidualSmootherMetadata

    def _project_partition(
        self,
        residual: ArrayLike,
        partition: Sequence[tuple[int, int]],
    ) -> ArrayLike:
        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        indices = jnp.arange(int(residual_vec.shape[0]))
        mask = jnp.zeros_like(residual_vec, dtype=bool)
        for start, stop in partition:
            mask = mask | ((indices >= int(start)) & (indices < int(stop)))
        return jnp.where(mask, residual_vec, jnp.zeros_like(residual_vec))

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply fixed-count projected residual smoothing to ``residual``."""

        residual_vec = jnp.asarray(residual, dtype=self.dtype).reshape((-1,))
        if int(residual_vec.shape[0]) != int(self.metadata.shape[0]):
            raise ValueError(
                f"residual length {residual_vec.shape[0]} does not match "
                f"operator rows {self.metadata.shape[0]}"
            )
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        damping = jnp.asarray(float(self.metadata.damping), dtype=residual_vec.dtype)
        for _ in range(max(1, int(self.metadata.sweeps))):
            directions = tuple(
                self._project_partition(remaining, partition)
                for partition in self.metadata.group_partitions
            )
            direction_matrix = jnp.stack(directions, axis=1)
            action_matrix = _operator_on_basis(
                self.operator_matvec,
                direction_matrix,
                shape=self.metadata.shape,
                dtype=residual_vec.dtype,
            )
            coefficients = _regularized_least_squares(
                action_matrix,
                remaining,
                rcond=float(self.metadata.regularization_rcond),
            )
            step = direction_matrix @ coefficients
            action = action_matrix @ coefficients
            correction = correction + damping * step
            remaining = remaining - damping * action
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for local-smoother hooks."""

        return self.apply


def _normalize_coarse_solver(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action_lstsq": "action_lstsq",
        "action_ls": "action_lstsq",
        "lstsq": "action_lstsq",
        "least_squares": "action_lstsq",
        "projected": "projected",
        "galerkin": "projected",
        "qtaq": "projected",
    }
    if normalized not in aliases:
        raise ValueError("coarse_solver must be 'action_lstsq' or 'projected'")
    return aliases[normalized]


def _normalize_composition(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "add": "additive",
        "additive": "additive",
        "mult": "multiplicative",
        "multiplicative": "multiplicative",
        "field_split": "multiplicative",
        "schur": "multiplicative",
    }
    if normalized not in aliases:
        raise ValueError("composition must be 'multiplicative' or 'additive'")
    return aliases[normalized]


def _normalize_matrix_free_smoother_step_policy(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "fixed": "stationary",
        "richardson": "stationary",
        "stationary": "stationary",
        "minres": "residual_minimizing",
        "minimum_residual": "residual_minimizing",
        "residual_minimizing": "residual_minimizing",
        "residual_reducing": "residual_minimizing",
    }
    if normalized not in aliases:
        raise ValueError("matrix_free_smoother_step_policy must be 'stationary' or 'residual_minimizing'")
    return aliases[normalized]


def _normalize_matrix_free_block_smoother_grouping(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "contiguous": "contiguous",
        "block": "contiguous",
        "blocks": "contiguous",
        "block_contiguous": "contiguous",
        "hierarchy": "block_hierarchy",
        "hierarchical": "block_hierarchy",
        "multilevel": "block_hierarchy",
        "block_hierarchy": "block_hierarchy",
        "residual_hierarchy": "block_hierarchy",
        "adaptive": "block_hierarchy",
        "adaptive_residual": "block_hierarchy",
        "adaptive_residual_equation": "block_hierarchy",
        "hybrid": "block_x_species",
        "aggregate": "block_x_species",
        "aggregates": "block_x_species",
        "block_x": "block_x_species",
        "block_species": "block_x_species",
        "x_species": "block_x_species",
        "block_x_species": "block_x_species",
        "radial_species": "block_x_species",
        "block_radial_species": "block_x_species",
    }
    if normalized not in aliases:
        raise ValueError(
            "matrix_free_block_smoother_grouping must be 'contiguous', "
            "'block_x_species', or 'block_hierarchy'"
        )
    return aliases[normalized]


def _normalize_adjoint_krylov_transpose_source(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    aliases = {
        "auto": "autodiff",
        "autodiff": "autodiff",
        "jax": "autodiff",
        "vjp": "autodiff",
        "linear_transpose": "autodiff",
        "off": "off",
        "none": "off",
        "disabled": "off",
    }
    if normalized not in aliases:
        raise ValueError("adjoint_krylov_transpose_source must be 'autodiff' or 'off'")
    return aliases[normalized]


def _metadata_int_tuple(metadata: Mapping[str, object] | None, key: str) -> tuple[int, ...]:
    if metadata is None or key not in metadata:
        return ()
    value = metadata[key]
    if isinstance(value, str):
        raw_values: Sequence[object] = tuple(part for part in value.replace(",", " ").split() if part)
    elif isinstance(value, Sequence):
        raw_values = value
    else:
        return ()
    result: list[int] = []
    for item in raw_values:
        try:
            result.append(int(item))
        except (TypeError, ValueError):
            return ()
    return tuple(result)


def _metadata_int_value(metadata: Mapping[str, object] | None, key: str, default: int) -> int:
    if metadata is None or key not in metadata:
        return int(default)
    value = metadata[key]
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _multilevel_layout_from_metadata(
    *,
    geometry_metadata: Mapping[str, object] | None,
    total_size: int,
) -> RHS1QICoarseBlockLayout:
    block_sizes = list(_metadata_int_tuple(geometry_metadata, "qi_block_sizes"))
    if not block_sizes:
        raise ValueError("multilevel_coarse requires geometry_metadata['qi_block_sizes']")
    block_x = list(_metadata_int_tuple(geometry_metadata, "qi_block_x"))
    block_species = list(_metadata_int_tuple(geometry_metadata, "qi_block_species"))
    covered_size = int(sum(block_sizes))
    tail_size = _metadata_int_value(
        geometry_metadata,
        "qi_block_tail_size",
        max(0, int(total_size) - int(covered_size)),
    )
    if covered_size < int(total_size) and int(tail_size) == int(total_size) - covered_size:
        block_sizes.append(int(tail_size))
        if block_x:
            block_x.append(-1)
        if block_species:
            block_species.append(-1)
        covered_size = int(sum(block_sizes))
    if covered_size != int(total_size):
        raise ValueError(
            "multilevel_coarse requires qi_block_sizes to cover the full operator size "
            f"({covered_size} != {int(total_size)})"
        )
    return RHS1QICoarseBlockLayout(
        block_sizes=tuple(block_sizes),
        n_theta=_metadata_int_value(geometry_metadata, "n_theta", 1),
        n_zeta=_metadata_int_value(geometry_metadata, "n_zeta", 1),
        block_x=tuple(block_x) if block_x else None,
        block_species=tuple(block_species) if block_species else None,
    )


def _build_multilevel_coarse_basis_from_metadata(
    *,
    geometry_metadata: Mapping[str, object] | None,
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int, int]:
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    max_rank = config.multilevel_max_rank
    if max_rank is None:
        max_rank = config.max_rank if config.max_rank is not None else 48
    multilevel_config = RHS1QIMultilevelCoarseConfig(
        max_levels=max(1, int(config.multilevel_max_levels)),
        aggregate_factor=max(2, int(config.multilevel_aggregate_factor)),
        max_rank=max(0, int(max_rank)),
        max_angular_mode=max(0, int(config.multilevel_max_angular_mode)),
        max_radial_degree=max(0, int(config.multilevel_max_radial_degree)),
        max_pitch_degree=max(0, int(config.multilevel_max_pitch_degree)),
        include_current_moments=bool(config.multilevel_current_moments),
        include_species_current_moments=bool(config.multilevel_species_current_moments),
        include_radial_current_moments=bool(config.multilevel_radial_current_moments),
        include_tail_constraint_moments=bool(config.multilevel_tail_constraint_moments),
        max_current_pitch_degree=max(0, int(config.multilevel_current_max_pitch_degree)),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        regularization_rcond=float(config.regularization_rcond),
        damping=float(config.damping),
        dtype=dtype,
    )
    basis, levels = build_rhs1_qi_multilevel_coarse_basis(layout, config=multilevel_config)
    return basis, len(levels), int(basis.metadata.candidate_count)


def _normalize_multilevel_residual_equation_order(value: str) -> str:
    order = str(value).strip().lower().replace("-", "_")
    if order in {"coarse_to_fine", "coarse", "coarse_first"}:
        return "coarse_to_fine"
    if order in {"fine_to_coarse", "fine", "fine_first"}:
        return "fine_to_coarse"
    raise ValueError("multilevel_residual_equation_order must be 'coarse_to_fine' or 'fine_to_coarse'")


def _normalize_multilevel_residual_equation_solver(value: str) -> str:
    solver = str(value).strip().lower().replace("-", "_")
    aliases = {
        "action": "action_lstsq",
        "action_ls": "action_lstsq",
        "action_lstsq": "action_lstsq",
        "least_squares": "action_lstsq",
        "lstsq": "action_lstsq",
        "staged": "action_lstsq",
        "galerkin": "galerkin",
        "projected": "galerkin",
        "qtaq": "galerkin",
        "coarse_grid": "galerkin",
    }
    if solver not in aliases:
        raise ValueError("multilevel_residual_equation_solver must be 'action_lstsq' or 'galerkin'")
    return aliases[solver]


def _build_multilevel_residual_equation_bases_from_metadata(
    *,
    geometry_metadata: Mapping[str, object] | None,
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, int, tuple[int, ...]]:
    """Build separate coarse-grid bases for the nested residual equation."""

    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    max_rank = max(1, int(config.multilevel_residual_equation_max_level_rank))
    multilevel_config = RHS1QIMultilevelCoarseConfig(
        max_levels=max(1, int(config.multilevel_max_levels)),
        aggregate_factor=max(2, int(config.multilevel_aggregate_factor)),
        max_rank=max_rank,
        max_angular_mode=max(0, int(config.multilevel_max_angular_mode)),
        max_radial_degree=max(0, int(config.multilevel_max_radial_degree)),
        max_pitch_degree=max(0, int(config.multilevel_max_pitch_degree)),
        include_current_moments=bool(config.multilevel_current_moments),
        include_species_current_moments=bool(config.multilevel_species_current_moments),
        include_radial_current_moments=bool(config.multilevel_radial_current_moments),
        include_tail_constraint_moments=bool(config.multilevel_tail_constraint_moments),
        max_current_pitch_degree=max(0, int(config.multilevel_current_max_pitch_degree)),
        nested_residual_correction=True,
        nested_level_max_rank=max_rank,
        nested_order=_normalize_multilevel_residual_equation_order(
            config.multilevel_residual_equation_order
        ),
        nested_include_global=bool(config.multilevel_residual_equation_include_global),
        dtype=dtype,
    )
    bases, _levels = build_rhs1_qi_multilevel_residual_level_bases(layout, config=multilevel_config)
    stage_ranks = tuple(int(basis.metadata.rank) for basis in bases)
    return bases, len(bases), sum(stage_ranks), stage_ranks


def _build_global_moment_residual_equation_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, tuple[int, ...], int, float]:
    """Build a fail-closed global moment Schur/Galerkin residual equation.

    This path differs from static multilevel moments: setup first constructs a
    compact space of current, tail-constraint, and radial-profile moments, then
    accepts it only if the reduced residual equation lowers the actual hard-seed
    operator residual.  Apply-time still uses the existing cached ``A Q`` JAX
    cascade, so the installed preconditioner remains device-compatible.
    """

    if int(shape[0]) != int(shape[1]):
        raise ValueError("global moment residual equation requires a square operator")
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    solver = _normalize_multilevel_residual_equation_solver(
        config.global_moment_residual_equation_solver
    )
    max_rank = max(1, int(config.global_moment_residual_equation_max_rank))
    moment_config = RHS1QIMultilevelCoarseConfig(
        max_levels=max(1, int(config.multilevel_max_levels)),
        aggregate_factor=max(2, int(config.multilevel_aggregate_factor)),
        max_rank=max_rank,
        max_angular_mode=0,
        max_radial_degree=max(0, int(config.multilevel_max_radial_degree)),
        max_pitch_degree=0,
        include_level_aggregates=bool(config.global_moment_residual_equation_include_profile),
        include_angular=False,
        include_radial=bool(config.global_moment_residual_equation_include_profile),
        include_radial_angular=False,
        include_pitch=False,
        include_radial_pitch=False,
        include_current_moments=bool(config.global_moment_residual_equation_include_current),
        include_species_current_moments=bool(config.multilevel_species_current_moments),
        include_radial_current_moments=bool(config.multilevel_radial_current_moments),
        include_tail_constraint_moments=bool(config.global_moment_residual_equation_include_tail),
        include_finest_blocks=False,
        max_current_pitch_degree=max(1, int(config.multilevel_current_max_pitch_degree)),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        regularization_rcond=float(config.regularization_rcond),
        damping=float(config.damping),
        dtype=dtype,
    )
    candidates, labels, _levels = build_rhs1_qi_multilevel_coarse_candidates(
        layout,
        config=moment_config,
    )
    selected_columns: list[ArrayLike] = []
    selected_labels: list[str] = []
    for index, label in enumerate(labels):
        label_text = str(label)
        include = (
            (bool(config.global_moment_residual_equation_include_current) and label_text.startswith("current:"))
            or (
                bool(config.global_moment_residual_equation_include_tail)
                and label_text.startswith("constraint_tail:")
            )
            or (
                bool(config.global_moment_residual_equation_include_profile)
                and label_text.startswith("level:")
            )
        )
        if include:
            selected_columns.append(jnp.asarray(candidates[:, int(index)], dtype=dtype).reshape((-1,)))
            selected_labels.append(f"global_moment:{label_text}")
    if not selected_columns:
        return (), 0, (), 0, float("inf")

    basis = orthonormalize_rhs1_qi_coarse_basis(
        jnp.stack(tuple(selected_columns), axis=1),
        labels=tuple(selected_labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=max_rank,
    )
    rank = int(basis.metadata.rank)
    if rank <= 0:
        return (), int(basis.metadata.candidate_count), (), 0, float("inf")

    q = jnp.asarray(basis.vectors, dtype=dtype)
    action = _operator_on_basis(operator_matvec, q, shape=shape, dtype=dtype)
    coarse_operator = jnp.conjugate(q).T @ action
    if solver == "galerkin":
        coefficients = _regularized_projected_solve(
            coarse_operator,
            jnp.conjugate(q).T @ residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = coarse_operator
    else:
        coefficients = _regularized_least_squares(
            action,
            residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = action
    next_residual = residual - action @ coefficients
    residual_norm = float(jnp.linalg.norm(residual))
    next_norm = float(jnp.linalg.norm(next_residual))
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_norm))
    singular_values = np.asarray(jnp.linalg.svd(condition_matrix, compute_uv=False), dtype=np.float64)
    positive = singular_values[singular_values > threshold]
    if positive.size == 0:
        condition_estimate = float("inf")
    else:
        condition_estimate = float(np.max(singular_values) / np.min(positive))
    if (not np.isfinite(next_norm)) or next_norm >= residual_norm - threshold:
        return (), int(basis.metadata.candidate_count), (), 0, condition_estimate
    return (basis,), int(basis.metadata.candidate_count), (rank,), rank, condition_estimate


def _basis_from_residual_galerkin_state(
    vectors: ArrayLike,
    *,
    labels: Sequence[str],
    candidate_count: int,
    candidate_labels: Sequence[str],
    dtype: Any,
) -> RHS1QICoarseBasis:
    q = jnp.asarray(vectors, dtype=dtype)
    if q.ndim != 2:
        raise ValueError("residual Galerkin basis must be two-dimensional")
    accepted_labels = tuple(f"residual_galerkin:{label}" for label in labels)
    norms = tuple(float(jnp.linalg.norm(q[:, index])) for index in range(int(q.shape[1])))
    return RHS1QICoarseBasis(
        vectors=q,
        metadata=RHS1QICoarseBasisMetadata(
            total_size=int(q.shape[0]),
            candidate_count=int(candidate_count),
            rank=int(q.shape[1]),
            discarded_count=max(0, int(candidate_count) - int(q.shape[1])),
            candidate_labels=tuple(f"residual_galerkin:{label}" for label in candidate_labels),
            accepted_labels=accepted_labels,
            candidate_norms=norms,
            accepted_norms=norms,
            rank_rtol=0.0,
            rank_atol=0.0,
        ),
    )


def _build_residual_galerkin_equation_basis_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, int, tuple[int, ...], int, float, float, float]:
    """Build a residual-derived Galerkin stage from actual block residuals."""

    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")
    solver = _normalize_multilevel_residual_equation_solver(
        config.residual_galerkin_equation_solver
    )
    galerkin_state = setup_rhs1_qi_residual_galerkin(
        operator=operator_matvec,
        residual=residual,
        block_sizes=tuple(int(value) for value in layout.block_sizes),
        config=RHS1QIResidualGalerkinConfig(
            max_stages=max(1, int(config.residual_galerkin_equation_max_stages)),
            max_stage_rank=max(1, int(config.residual_galerkin_equation_max_stage_rank)),
            max_rank=max(1, int(config.residual_galerkin_equation_max_rank)),
            rank_rtol=float(config.basis_rtol),
            rank_atol=float(config.basis_atol),
            regularization_rcond=float(config.regularization_rcond),
            damping=float(config.damping),
            solver=solver,
            min_relative_improvement=0.0,
            include_global_residual=bool(
                config.residual_galerkin_equation_include_global_residual
            ),
            include_block_residuals=bool(
                config.residual_galerkin_equation_include_block_residuals
            ),
            include_operator_images=bool(
                config.residual_galerkin_equation_include_operator_images
            ),
            include_operator_preimages=False,
            sort_blocks_by_residual_norm=True,
        ),
    )
    metadata = galerkin_state.metadata
    rank = int(metadata.rank)
    if not bool(metadata.accepted) or rank <= 0:
        return (
            (),
            int(metadata.candidate_count),
            int(metadata.stage_count),
            tuple(int(value) for value in metadata.stage_ranks),
            0,
            float(metadata.condition_estimate),
            float(metadata.residual_before),
            float(metadata.residual_after),
        )
    basis = _basis_from_residual_galerkin_state(
        galerkin_state.basis,
        labels=metadata.labels,
        candidate_count=int(metadata.candidate_count),
        candidate_labels=metadata.candidate_labels,
        dtype=dtype,
    )
    return (
        (basis,),
        int(metadata.candidate_count),
        int(metadata.stage_count),
        tuple(int(value) for value in metadata.stage_ranks),
        rank,
        float(metadata.condition_estimate),
        float(metadata.residual_before),
        float(metadata.residual_after),
    )


def _condition_estimate(matrix: ArrayLike, *, threshold: float) -> float:
    singular_values = np.asarray(jnp.linalg.svd(matrix, compute_uv=False), dtype=np.float64)
    positive = singular_values[singular_values > float(threshold)]
    if positive.size == 0:
        return float("inf")
    return float(np.max(singular_values) / np.min(positive))


def _build_phase_space_residual_equation_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, tuple[int, ...], int, float, float, float]:
    """Build a fail-closed trapped/passing phase-space residual equation."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("phase-space residual equation requires a square operator")
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    solver = _normalize_multilevel_residual_equation_solver(
        config.phase_space_residual_equation_solver
    )
    phase_config = RHS1QIPhaseSpaceCoarseConfig(
        max_rank=max(1, int(config.phase_space_residual_equation_max_rank)),
        trapped_boundary_fraction=float(config.phase_space_residual_equation_trapped_boundary_fraction),
        include_trapped=bool(config.phase_space_residual_equation_include_trapped),
        include_passing=bool(config.phase_space_residual_equation_include_passing),
        include_boundary=bool(config.phase_space_residual_equation_include_boundary),
        include_even=bool(config.phase_space_residual_equation_include_even),
        include_odd=bool(config.phase_space_residual_equation_include_odd),
        include_radial=bool(config.phase_space_residual_equation_include_radial),
        include_species=bool(config.phase_space_residual_equation_include_species),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        dtype=dtype,
    )
    basis = build_rhs1_qi_phase_space_coarse_basis(layout, config=phase_config)
    rank = int(basis.metadata.rank)
    candidate_count = int(basis.metadata.candidate_count)
    residual_before = float(jnp.linalg.norm(residual))
    if rank <= 0:
        return (), candidate_count, (), 0, float("inf"), residual_before, float("inf")

    q = jnp.asarray(basis.vectors, dtype=dtype)
    action = _operator_on_basis(operator_matvec, q, shape=shape, dtype=dtype)
    coarse_operator = jnp.conjugate(q).T @ action
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_before))
    if solver == "galerkin":
        coefficients = _regularized_projected_solve(
            coarse_operator,
            jnp.conjugate(q).T @ residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = coarse_operator
    else:
        coefficients = _regularized_least_squares(
            action,
            residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = action
    next_residual = residual - action @ coefficients
    residual_after = float(jnp.linalg.norm(next_residual))
    condition_estimate = _condition_estimate(condition_matrix, threshold=threshold)
    if (not np.isfinite(residual_after)) or residual_after >= residual_before - threshold:
        return (), candidate_count, (), 0, condition_estimate, residual_before, residual_after
    return (basis,), candidate_count, (rank,), rank, condition_estimate, residual_before, residual_after


def _build_residual_region_bounce_coarse_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, tuple[int, ...], int, float, float, float]:
    """Build a fail-closed residual-localized bounce-region coarse equation."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("residual-region bounce coarse requires a square operator")
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    solver = _normalize_multilevel_residual_equation_solver(
        config.residual_region_bounce_coarse_solver
    )
    region_config = RHS1QIResidualRegionCoarseConfig(
        max_rank=max(1, int(config.residual_region_bounce_coarse_max_rank)),
        max_candidates=max(1, int(config.residual_region_bounce_coarse_max_candidates)),
        trapped_boundary_fraction=float(config.residual_region_bounce_coarse_trapped_boundary_fraction),
        min_region_energy_fraction=float(config.residual_region_bounce_coarse_min_region_energy_fraction),
        include_global_active_region=bool(config.residual_region_bounce_coarse_include_global),
        include_block_regions=True,
        include_block_bounce_regions=True,
        include_pitch_regions=True,
        include_radial_regions=bool(config.residual_region_bounce_coarse_include_radial),
        include_radial_bounce_regions=bool(config.residual_region_bounce_coarse_include_radial),
        include_species_regions=bool(config.residual_region_bounce_coarse_include_species),
        include_species_bounce_regions=bool(config.residual_region_bounce_coarse_include_species),
        region_bands=str(config.residual_region_bounce_coarse_region_bands),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        dtype=dtype,
    )
    basis = build_rhs1_qi_residual_region_coarse_basis(
        layout,
        residual,
        config=region_config,
    )
    rank = int(basis.metadata.rank)
    candidate_count = int(basis.metadata.candidate_count)
    residual_before = float(jnp.linalg.norm(residual))
    if rank <= 0:
        return (), candidate_count, (), 0, float("inf"), residual_before, float("inf")

    q = jnp.asarray(basis.vectors, dtype=dtype)
    action = _operator_on_basis(operator_matvec, q, shape=shape, dtype=dtype)
    coarse_operator = jnp.conjugate(q).T @ action
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_before))
    if solver == "galerkin":
        coefficients = _regularized_projected_solve(
            coarse_operator,
            jnp.conjugate(q).T @ residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = coarse_operator
    else:
        coefficients = _regularized_least_squares(
            action,
            residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = action
    next_residual = residual - action @ coefficients
    residual_after = float(jnp.linalg.norm(next_residual))
    condition_estimate = _condition_estimate(condition_matrix, threshold=threshold)
    if (not np.isfinite(residual_after)) or residual_after >= residual_before - threshold:
        return (), candidate_count, (), 0, condition_estimate, residual_before, residual_after
    return (basis,), candidate_count, (rank,), rank, condition_estimate, residual_before, residual_after


def _build_active_pattern_coarse_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, tuple[int, ...], int, float, float, float]:
    """Build a residual active-pattern coarse equation.

    The candidate directions are selected from actual high-energy residual
    pitch/angular/radial/species chunks, then accepted only if the cached
    ``A Q`` least-squares residual equation reduces the setup residual.  This
    gives hard QI seeds a stronger residual-derived global closure without
    adding apply-time host callbacks or unbounded smoother work.
    """

    if int(shape[0]) != int(shape[1]):
        raise ValueError("active-pattern coarse requires a square operator")
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    solver = _normalize_multilevel_residual_equation_solver(config.active_pattern_coarse_solver)
    active_config = RHS1QIActivePatternCoarseConfig(
        max_rank=max(1, int(config.active_pattern_coarse_max_rank)),
        max_candidates=max(1, int(config.active_pattern_coarse_max_candidates)),
        min_chunk_energy_fraction=float(config.active_pattern_coarse_min_chunk_energy_fraction),
        include_block_pitch_chunks=bool(config.active_pattern_coarse_include_block_pitch),
        include_block_angular_chunks=bool(config.active_pattern_coarse_include_block_angular),
        include_radial_pitch_chunks=bool(config.active_pattern_coarse_include_radial_pitch),
        include_radial_angular_chunks=bool(config.active_pattern_coarse_include_radial_angular),
        include_block_chunks=bool(config.active_pattern_coarse_include_block),
        include_radial_chunks=bool(config.active_pattern_coarse_include_radial),
        include_species_chunks=bool(config.active_pattern_coarse_include_species),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        dtype=dtype,
    )
    basis = build_rhs1_qi_active_pattern_coarse_basis(
        layout,
        residual,
        config=active_config,
    )
    rank = int(basis.metadata.rank)
    candidate_count = int(basis.metadata.candidate_count)
    residual_before = float(jnp.linalg.norm(residual))
    if rank <= 0:
        return (), candidate_count, (), 0, float("inf"), residual_before, float("inf")

    q = jnp.asarray(basis.vectors, dtype=dtype)
    action = _operator_on_basis(operator_matvec, q, shape=shape, dtype=dtype)
    coarse_operator = jnp.conjugate(q).T @ action
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_before))
    if solver == "galerkin":
        coefficients = _regularized_projected_solve(
            coarse_operator,
            jnp.conjugate(q).T @ residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = coarse_operator
    else:
        coefficients = _regularized_least_squares(
            action,
            residual,
            rcond=float(config.regularization_rcond),
        )
        condition_matrix = action
    next_residual = residual - action @ coefficients
    residual_after = float(jnp.linalg.norm(next_residual))
    condition_estimate = _condition_estimate(condition_matrix, threshold=threshold)
    if (not np.isfinite(residual_after)) or residual_after >= residual_before - threshold:
        return (), candidate_count, (), 0, condition_estimate, residual_before, residual_after
    return (basis,), candidate_count, (rank,), rank, condition_estimate, residual_before, residual_after


def _unique_snapshot_groups(groups: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    unique: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    for group in groups:
        item = tuple(int(index) for index in group)
        if not item or item in seen:
            continue
        unique.append(item)
        seen.add(item)
    return tuple(unique)


def _residual_snapshot_groups(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[int, ...], ...]:
    """Return deterministic block groups used to sample the current residual."""

    block_count = len(tuple(layout.block_sizes))
    groups: list[tuple[int, ...]] = []
    if bool(config.residual_snapshot_include_global):
        groups.append(tuple(range(block_count)))
    if bool(config.residual_snapshot_include_blocks):
        groups.extend((index,) for index in range(block_count))
    if bool(config.residual_snapshot_include_aggregates):
        factor = max(2, int(config.multilevel_aggregate_factor))
        for level in range(1, max(1, int(config.multilevel_max_levels))):
            width = factor**level
            for start in range(0, block_count, width):
                groups.append(tuple(range(start, min(block_count, start + width))))
    return _unique_snapshot_groups(groups)


def _block_schur_residual_groups(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[int, ...], ...]:
    """Return block groups used for off-block Schur residual candidates."""

    snapshot_config = RHS1QIDevicePreconditionerConfig(
        multilevel_max_levels=int(config.multilevel_max_levels),
        multilevel_aggregate_factor=int(config.multilevel_aggregate_factor),
        residual_snapshot_include_global=bool(config.block_schur_residual_include_global),
        residual_snapshot_include_blocks=bool(config.block_schur_residual_include_blocks),
        residual_snapshot_include_aggregates=bool(config.block_schur_residual_include_aggregates),
    )
    return _residual_snapshot_groups(layout, config=snapshot_config)


def _restrict_to_block_group(
    values: ArrayLike,
    *,
    group: Sequence[int],
    offsets: Sequence[int],
    total_size: int,
    dtype: Any,
) -> ArrayLike:
    vector = jnp.asarray(values, dtype=dtype).reshape((-1,))
    restricted = jnp.zeros((int(total_size),), dtype=dtype)
    for block_index in group:
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        restricted = restricted.at[start:stop].set(vector[start:stop])
    return restricted


def _remove_block_group(
    values: ArrayLike,
    *,
    group: Sequence[int],
    offsets: Sequence[int],
    dtype: Any,
) -> ArrayLike:
    vector = jnp.asarray(values, dtype=dtype).reshape((-1,))
    outside = vector
    for block_index in group:
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        outside = outside.at[start:stop].set(jnp.zeros((stop - start,), dtype=dtype))
    return outside


def _build_residual_snapshot_basis_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int, int]:
    """Build coarse directions by restricting the current residual to QI blocks.

    The multilevel analytic basis can miss seed-specific residual modes on hard
    QI cases.  This builder adds only rank-gated snapshots of the actual current
    residual on block and aggregate supports, so the timed preconditioner still
    applies a cached ``A Q`` coarse solve rather than doing additional smoothing.
    """

    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    offsets = tuple(int(value) for value in layout.block_offsets)
    columns: list[ArrayLike] = []
    labels: list[str] = []
    groups = _residual_snapshot_groups(layout, config=config)
    rank_limit = max(1, int(config.residual_snapshot_max_rank))
    for group in groups:
        masked_residual = jnp.zeros((int(total_size),), dtype=dtype)
        for block_index in group:
            start = offsets[int(block_index)]
            stop = offsets[int(block_index) + 1]
            masked_residual = masked_residual.at[start:stop].set(residual[start:stop])
        group_label = ",".join(str(index) for index in group)
        if bool(config.residual_snapshot_include_primal):
            _append_normalized_candidate(
                columns,
                labels,
                masked_residual,
                f"residual_snapshot_primal:{group_label}",
                total_size=int(total_size),
                dtype=dtype,
            )
        if len(columns) >= rank_limit:
            break
        if bool(config.residual_snapshot_use_adjoint):
            adjoint_candidate = _autodiff_transpose_matvec(
                operator_matvec,
                masked_residual,
                shape=shape,
                dtype=dtype,
            )
            _append_normalized_candidate(
                columns,
                labels,
                adjoint_candidate,
                f"residual_snapshot_adjoint:{group_label}",
                total_size=int(total_size),
                dtype=dtype,
            )
        if len(columns) >= rank_limit:
            break

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((int(total_size), 0), dtype=dtype)
    snapshot_basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=rank_limit,
    )
    return snapshot_basis, int(len(columns)), int(len(groups))


def _build_residual_snapshot_residual_equation_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, int, tuple[int, ...], int]:
    """Build staged residual equations from residual snapshots.

    The flat residual-snapshot enrichment uses all accepted snapshot columns in
    one rank-gated coarse solve.  This staged variant instead gives each
    block/aggregate snapshot group its own residual equation, accepts the stage
    only if it reduces the remaining setup residual, and caches the resulting
    ``A Q_l`` action for pure-JAX apply-time reuse.
    """

    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    offsets = tuple(int(value) for value in layout.block_offsets)
    groups = _residual_snapshot_groups(layout, config=config)
    if not groups:
        raise ValueError("residual-snapshot residual equation requires at least one QI block group")

    max_rank = max(1, int(config.residual_snapshot_residual_equation_max_rank))
    residual_norm = float(jnp.linalg.norm(residual))
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_norm))
    solver = _normalize_multilevel_residual_equation_solver(
        config.residual_snapshot_residual_equation_solver
    )
    accepted_columns: list[ArrayLike] = []
    stage_bases: list[RHS1QICoarseBasis] = []
    remaining = residual
    candidate_count = 0
    used_rank = 0

    for group in groups:
        if used_rank >= max_rank:
            break
        masked_residual = _restrict_to_block_group(
            residual,
            group=group,
            offsets=offsets,
            total_size=int(total_size),
            dtype=dtype,
        )
        group_columns: list[ArrayLike] = []
        group_labels: list[str] = []
        group_label = ",".join(str(index) for index in group)
        if bool(config.residual_snapshot_include_primal):
            primal = _orthogonalized_candidate(
                masked_residual,
                (*accepted_columns, *group_columns),
                threshold=threshold,
                total_size=int(total_size),
                dtype=dtype,
            )
            if primal is not None:
                group_columns.append(primal)
                group_labels.append(f"residual_snapshot_equation_primal:{group_label}")
        if bool(config.residual_snapshot_use_adjoint) and used_rank + len(group_columns) < max_rank:
            adjoint_candidate = _autodiff_transpose_matvec(
                operator_matvec,
                masked_residual,
                shape=shape,
                dtype=dtype,
            )
            adjoint = _orthogonalized_candidate(
                adjoint_candidate,
                (*accepted_columns, *group_columns),
                threshold=threshold,
                total_size=int(total_size),
                dtype=dtype,
            )
            if adjoint is not None:
                group_columns.append(adjoint)
                group_labels.append(f"residual_snapshot_equation_adjoint:{group_label}")
        if not group_columns:
            continue

        stage_basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.stack(tuple(group_columns), axis=1),
            labels=tuple(group_labels),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=max(1, int(max_rank - used_rank)),
        )
        if int(stage_basis.metadata.rank) <= 0:
            continue
        stage_q = jnp.asarray(stage_basis.vectors, dtype=dtype)
        stage_action = _operator_on_basis(operator_matvec, stage_q, shape=shape, dtype=dtype)
        if solver == "galerkin":
            coefficients = _regularized_projected_solve(
                jnp.conjugate(stage_q).T @ stage_action,
                jnp.conjugate(stage_q).T @ remaining,
                rcond=float(config.regularization_rcond),
            )
        else:
            coefficients = _regularized_least_squares(
                stage_action,
                remaining,
                rcond=float(config.regularization_rcond),
            )
        next_remaining = remaining - stage_action @ coefficients
        if float(jnp.linalg.norm(next_remaining)) >= float(jnp.linalg.norm(remaining)) - threshold:
            continue

        stage_bases.append(stage_basis)
        for column_index in range(int(stage_q.shape[1])):
            accepted_columns.append(stage_q[:, column_index])
        candidate_count += int(stage_basis.metadata.candidate_count)
        used_rank += int(stage_basis.metadata.rank)
        remaining = next_remaining

    stage_ranks = tuple(int(basis.metadata.rank) for basis in stage_bases)
    return tuple(stage_bases), int(len(groups)), int(candidate_count), stage_ranks, int(sum(stage_ranks))


def _build_block_schur_residual_basis_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int, int]:
    """Build off-block Schur residual directions from the current residual.

    For a block-supported trial correction ``d_g = P_g r``, the off-block
    response ``P_not_g A d_g`` identifies coupling that the local block cannot
    cancel by itself.  Pulling that response back with ``A.T`` and restricting
    it to the source group gives a setup-time Schur-like candidate
    ``P_g A.T P_not_g A P_g r``.  The reusable preconditioner still stores only
    rank-gated ``Q`` and ``A Q`` columns, so apply-time remains device-only.
    """

    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")
    if int(shape[0]) != int(shape[1]):
        raise ValueError("block-Schur residual enrichment requires a square operator")

    offsets = tuple(int(value) for value in layout.block_offsets)
    columns: list[ArrayLike] = []
    labels: list[str] = []
    groups = _block_schur_residual_groups(layout, config=config)
    rank_limit = max(1, int(config.block_schur_residual_max_rank))
    for group in groups:
        block_residual = _restrict_to_block_group(
            residual,
            group=group,
            offsets=offsets,
            total_size=int(total_size),
            dtype=dtype,
        )
        action = jnp.asarray(operator_matvec(block_residual), dtype=dtype).reshape((-1,))
        offblock_action = _remove_block_group(action, group=group, offsets=offsets, dtype=dtype)
        pulled_back = _autodiff_transpose_matvec(
            operator_matvec,
            offblock_action,
            shape=shape,
            dtype=dtype,
        )
        schur_candidate = _restrict_to_block_group(
            pulled_back,
            group=group,
            offsets=offsets,
            total_size=int(total_size),
            dtype=dtype,
        )
        group_label = ",".join(str(index) for index in group)
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            schur_candidate,
            f"block_schur_residual:{group_label}",
            total_size=int(total_size),
            dtype=dtype,
        )
        if len(columns) - before >= 1 and len(columns) >= rank_limit:
            break

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((int(total_size), 0), dtype=dtype)
    schur_basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=rank_limit,
    )
    return schur_basis, int(len(columns)), int(len(groups))


def _combine_coarse_bases(
    base: RHS1QICoarseBasis,
    extra: RHS1QICoarseBasis,
    *,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QICoarseBasis:
    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(base.vectors, dtype=dtype)
    for index, label in enumerate(base.metadata.accepted_labels):
        columns.append(base_vectors[:, int(index)])
        labels.append(str(label))
    extra_vectors = jnp.asarray(extra.vectors, dtype=dtype)
    for index, label in enumerate(extra.metadata.accepted_labels):
        columns.append(extra_vectors[:, int(index)])
        labels.append(f"multilevel:{label}")
    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((int(base_vectors.shape[0]), 0), dtype=dtype)
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )


def _merge_group_slices(
    slices: Sequence[tuple[int, int]],
    *,
    max_groups: int,
) -> tuple[tuple[int, int], ...]:
    valid = tuple((int(start), int(stop)) for start, stop in slices if int(stop) > int(start))
    if not valid:
        return ()
    group_limit = max(1, int(max_groups))
    if len(valid) <= group_limit:
        return valid
    merged: list[tuple[int, int]] = []
    n_slices = len(valid)
    for group_index in range(group_limit):
        first = int(group_index * n_slices // group_limit)
        last = int((group_index + 1) * n_slices // group_limit)
        if last <= first:
            continue
        merged.append((valid[first][0], valid[last - 1][1]))
    return tuple(merged)


def _partition_bounds(partition: Sequence[tuple[int, int]]) -> tuple[int, int]:
    starts = [int(start) for start, _ in partition]
    stops = [int(stop) for _, stop in partition]
    return (min(starts), max(stops))


def _matrix_free_block_group_partitions(
    *,
    shape: tuple[int, int],
    geometry_metadata: Mapping[str, object] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[
    tuple[tuple[tuple[int, int], ...], ...],
    tuple[tuple[int, int], ...],
    int,
    str,
]:
    block_sizes = _metadata_int_tuple(geometry_metadata, "qi_block_sizes")
    if not block_sizes:
        raise ValueError(
            "matrix_free_block_minres local smoother requires geometry_metadata['qi_block_sizes']"
        )
    if any(size <= 0 for size in block_sizes):
        raise ValueError("qi_block_sizes entries must be positive")
    n_rows = int(shape[0])
    offsets = [0]
    for size in block_sizes:
        offsets.append(offsets[-1] + int(size))
    if offsets[-1] > n_rows:
        raise ValueError(f"qi_block_sizes sum {offsets[-1]} exceeds operator rows {n_rows}")
    block_slices = tuple((offsets[index], offsets[index + 1]) for index in range(len(block_sizes)))
    tail_slice = (offsets[-1], n_rows) if offsets[-1] < n_rows else None
    grouping = _normalize_matrix_free_block_smoother_grouping(config.matrix_free_block_smoother_grouping)
    max_groups = max(1, int(config.matrix_free_block_smoother_max_groups))
    if grouping == "block_hierarchy":
        layout = _multilevel_layout_from_metadata(
            geometry_metadata=geometry_metadata,
            total_size=int(shape[0]),
        )
        layout_offsets = tuple(int(value) for value in layout.block_offsets)
        layout_block_count = len(tuple(layout.block_sizes))
        layout_slices = tuple(
            (layout_offsets[index], layout_offsets[index + 1])
            for index in range(layout_block_count)
        )
        factor = max(2, int(config.multilevel_aggregate_factor))
        levels = max(1, int(config.multilevel_max_levels))
        hierarchy_groups: list[tuple[int, ...]] = [tuple(range(layout_block_count))]
        for level in range(levels - 1, 0, -1):
            width = factor**level
            for start in range(0, layout_block_count, width):
                group = tuple(range(start, min(layout_block_count, start + width)))
                if len(group) > 1:
                    hierarchy_groups.append(group)
        hierarchy_groups.extend((index,) for index in range(layout_block_count))
        seen: set[tuple[int, ...]] = set()
        group_partitions_list: list[tuple[tuple[int, int], ...]] = []
        for group in hierarchy_groups:
            if len(group_partitions_list) >= max_groups:
                break
            if not group or group in seen:
                continue
            seen.add(group)
            partition = tuple(layout_slices[int(index)] for index in group)
            if partition:
                group_partitions_list.append(partition)
        group_partitions = tuple(group_partitions_list)
        group_slices = tuple(_partition_bounds(partition) for partition in group_partitions)
        block_slices = layout_slices
    elif grouping == "contiguous":
        slices = list(block_slices)
        if bool(config.matrix_free_block_smoother_include_tail) and tail_slice is not None:
            slices.append(tail_slice)
        group_slices = _merge_group_slices(slices, max_groups=max_groups)
        group_partitions = tuple(((int(start), int(stop)),) for start, stop in group_slices)
    else:
        block_x = _metadata_int_tuple(geometry_metadata, "qi_block_x")
        block_species = _metadata_int_tuple(geometry_metadata, "qi_block_species")
        aggregate_partitions: list[tuple[tuple[int, int], ...]] = []
        if len(block_x) == len(block_slices):
            for x_index in sorted(set(int(value) for value in block_x)):
                aggregate_partitions.append(
                    tuple(
                        block_slices[index]
                        for index, value in enumerate(block_x)
                        if int(value) == int(x_index)
                    )
                )
        if len(block_species) == len(block_slices):
            for species_index in sorted(set(int(value) for value in block_species)):
                aggregate_partitions.append(
                    tuple(
                        block_slices[index]
                        for index, value in enumerate(block_species)
                        if int(value) == int(species_index)
                    )
                )
        tail_partitions: list[tuple[tuple[int, int], ...]] = []
        if bool(config.matrix_free_block_smoother_include_tail) and tail_slice is not None:
            tail_partitions.append((tail_slice,))
        reserved_groups = len(aggregate_partitions) + len(tail_partitions)
        block_group_limit = max(1, max_groups - reserved_groups) if max_groups > reserved_groups else max_groups
        block_partitions = tuple(
            ((int(start), int(stop)),)
            for start, stop in _merge_group_slices(block_slices, max_groups=block_group_limit)
        )
        group_partitions = tuple(block_partitions + tuple(aggregate_partitions) + tuple(tail_partitions))
        if len(group_partitions) > max_groups:
            group_partitions = group_partitions[:max_groups]
        group_slices = tuple(_partition_bounds(partition) for partition in group_partitions)
    if not group_slices:
        raise ValueError("matrix_free_block_minres local smoother found no non-empty groups")
    return group_partitions, group_slices, len(block_slices), grouping


def _build_matrix_free_residual_smoother(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QIMatrixFreeResidualSmoother:
    """Build a bounded device-compatible matrix-free local smoother."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("matrix-free residual smoother requires a square operator")
    sweeps = max(1, int(config.matrix_free_smoother_sweeps))
    damping = float(config.matrix_free_smoother_damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("matrix_free_smoother_damping must be finite and positive")
    alpha_clip = max(0.0, float(config.matrix_free_smoother_alpha_clip))
    min_denominator = max(0.0, float(config.matrix_free_smoother_min_denominator))
    metadata = RHS1QIMatrixFreeResidualSmootherMetadata(
        shape=tuple(int(value) for value in shape),
        sweeps=sweeps,
        damping=damping,
        step_policy=_normalize_matrix_free_smoother_step_policy(config.matrix_free_smoother_step_policy),
        alpha_clip=alpha_clip,
        min_denominator=min_denominator,
        device_resident=True,
        source="matrix_free_matvec",
        reason="built",
    )
    return RHS1QIMatrixFreeResidualSmoother(
        operator_matvec=operator_matvec,
        dtype=dtype,
        metadata=metadata,
    )


def _build_matrix_free_projected_residual_smoother(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    shape: tuple[int, int],
    dtype: Any,
    geometry_metadata: Mapping[str, object] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QIMatrixFreeProjectedResidualSmoother:
    """Build a bounded projected block residual smoother."""

    if int(shape[0]) != int(shape[1]):
        raise ValueError("matrix-free block smoother requires a square operator")
    sweeps = max(1, int(config.matrix_free_smoother_sweeps))
    damping = float(config.matrix_free_smoother_damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("matrix_free_smoother_damping must be finite and positive")
    rcond = float(config.matrix_free_block_smoother_rcond)
    if not np.isfinite(rcond) or rcond <= 0.0:
        raise ValueError("matrix_free_block_smoother_rcond must be finite and positive")
    group_partitions, group_slices, block_count, grouping = _matrix_free_block_group_partitions(
        shape=shape,
        geometry_metadata=geometry_metadata,
        config=config,
    )
    metadata = RHS1QIMatrixFreeProjectedResidualSmootherMetadata(
        shape=tuple(int(value) for value in shape),
        group_slices=group_slices,
        group_partitions=group_partitions,
        sweeps=sweeps,
        damping=damping,
        regularization_rcond=rcond,
        block_count=int(block_count),
        group_count=int(len(group_slices)),
        max_groups=max(1, int(config.matrix_free_block_smoother_max_groups)),
        include_tail=bool(config.matrix_free_block_smoother_include_tail),
        grouping=grouping,
        device_resident=True,
        source="matrix_free_block_projections",
        reason="built",
    )
    return RHS1QIMatrixFreeProjectedResidualSmoother(
        operator_matvec=operator_matvec,
        dtype=dtype,
        metadata=metadata,
    )


def _regularized_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    b = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("coarse least-squares matrix must be two-dimensional")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=b.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ b
    scale = jnp.maximum(jnp.linalg.norm(gram), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(float(rcond), 0.0), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _regularized_projected_solve(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    """Solve a square Galerkin residual equation with a scale-relative ridge."""

    a = jnp.asarray(matrix)
    b = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("projected coarse matrix must be two-dimensional")
    if int(a.shape[0]) != int(a.shape[1]):
        raise ValueError("projected coarse matrix must be square")
    if int(a.shape[0]) != int(b.shape[0]):
        raise ValueError("rhs length must match projected coarse matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=b.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(float(rcond), 0.0), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, b)


def _operator_on_basis(
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis_vectors: ArrayLike,
    *,
    shape: tuple[int, int],
    dtype: Any,
) -> ArrayLike:
    q = jnp.asarray(basis_vectors, dtype=dtype)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a matrix")
    if int(q.shape[0]) != int(shape[1]):
        raise ValueError(f"basis row count {q.shape[0]} does not match operator columns {shape[1]}")
    if int(q.shape[1]) == 0:
        return jnp.zeros((int(shape[0]), 0), dtype=dtype)
    columns = [jnp.asarray(operator_matvec(q[:, idx]), dtype=dtype).reshape((-1,)) for idx in range(int(q.shape[1]))]
    return jnp.stack(columns, axis=1)


def _autodiff_transpose_matvec(
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    vector: ArrayLike,
    *,
    shape: tuple[int, int],
    dtype: Any,
) -> ArrayLike:
    """Apply ``A.T`` to ``vector`` using JAX's linear VJP machinery.

    This is setup-time only for the adjoint-normal coarse space; the timed
    Krylov preconditioner still uses the cached ``A Q`` action and does not
    call a transpose.
    """

    vector_arr = jnp.asarray(vector, dtype=dtype).reshape((-1,))
    if int(vector_arr.shape[0]) != int(shape[0]):
        raise ValueError(f"transpose vector length {vector_arr.shape[0]} does not match operator rows {shape[0]}")

    def _mv(x):
        return jnp.asarray(operator_matvec(jnp.asarray(x, dtype=dtype)), dtype=dtype).reshape((-1,))

    x0 = jnp.zeros((int(shape[1]),), dtype=dtype)
    _, pullback = jax.vjp(_mv, x0)
    return jnp.asarray(pullback(vector_arr)[0], dtype=dtype).reshape((-1,))


def _append_normalized_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike,
    label: str,
    *,
    total_size: int,
    dtype: Any,
) -> None:
    vector = jnp.asarray(values, dtype=dtype).reshape((-1,))
    if int(vector.shape[0]) != int(total_size):
        raise ValueError(f"candidate {label!r} has length {vector.shape[0]}, expected {total_size}")
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector / norm)
    labels.append(str(label))


def _enrich_basis_with_residual(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Add residual-generated matrix-free directions to a physics coarse basis.

    For large QI seeds the full CSR operator can be too expensive to keep on
    device.  This enrichment builds a bounded correction-space Krylov basis
    ``{r, A r, A^2 r, ...}`` using only matrix-vector products.  The resulting
    basis still goes through the same rank gate and true-residual acceptance
    probe, so weak or harmful enrichments fail closed.
    """

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    total_size = int(shape[1])
    if int(residual.shape[0]) != total_size:
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(basis.vectors, dtype=dtype)
    for index, label in enumerate(basis.metadata.accepted_labels):
        _append_normalized_candidate(
            columns,
            labels,
            base_vectors[:, int(index)],
            f"base:{label}",
            total_size=total_size,
            dtype=dtype,
        )

    residual_candidate_count = 0
    current = residual
    if bool(config.residual_enrichment_include_residual):
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current,
            "residual:0",
            total_size=total_size,
            dtype=dtype,
        )
        residual_candidate_count += len(columns) - before

    for depth in range(max(0, int(config.residual_enrichment_depth))):
        current = jnp.asarray(operator_matvec(current), dtype=dtype).reshape((-1,))
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current,
            f"operator_power:{depth + 1}",
            total_size=total_size,
            dtype=dtype,
        )
        residual_candidate_count += len(columns) - before

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=dtype)
    enriched = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )
    return enriched, residual_candidate_count


def _coarse_action_residual(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    rcond: float,
) -> ArrayLike:
    """Return the true residual after the best current coarse correction."""

    residual_vec = jnp.asarray(residual, dtype=dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return residual_vec
    aq = _operator_on_basis(operator_matvec, basis.vectors, shape=shape, dtype=dtype)
    coefficients = _regularized_least_squares(aq, residual_vec, rcond=float(rcond))
    correction = jnp.asarray(basis.vectors, dtype=dtype) @ coefficients
    return residual_vec - jnp.asarray(operator_matvec(correction), dtype=dtype).reshape((-1,))


def _enrich_basis_with_recycle_residuals(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Append residuals left by bounded coarse corrections.

    This is a device-compatible GCRO-style seed: the current coarse space first
    removes what it can from the true residual, then the remaining slow residual
    is appended as a new candidate direction.  Repeating this for a small number
    of cycles builds a recycle space targeted at the actual hard seed without
    host factors, dense full operators, or unbounded Krylov work.
    """

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    total_size = int(shape[1])
    if int(residual.shape[0]) != total_size:
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    current_basis = basis
    current_residual = residual
    candidate_count = 0
    for cycle in range(max(0, int(config.recycle_enrichment_cycles))):
        current_residual = _coarse_action_residual(
            operator_matvec=operator_matvec,
            basis=current_basis,
            residual=current_residual,
            shape=shape,
            dtype=dtype,
            rcond=float(config.regularization_rcond),
        )
        columns = [jnp.asarray(current_basis.vectors, dtype=dtype)[:, idx] for idx in range(int(current_basis.metadata.rank))]
        labels = [str(label) for label in current_basis.metadata.accepted_labels]
        before = len(columns)
        _append_normalized_candidate(
            columns,
            labels,
            current_residual,
            f"recycle_residual:{cycle}",
            total_size=total_size,
            dtype=dtype,
        )
        candidate_count += len(columns) - before
        if not columns:
            break
        current_basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.stack(tuple(columns), axis=1),
            labels=tuple(labels),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    return current_basis, candidate_count


def _orthogonalized_candidate(
    values: ArrayLike,
    columns: Sequence[ArrayLike],
    *,
    threshold: float,
    total_size: int,
    dtype: Any,
) -> ArrayLike | None:
    vector = jnp.asarray(values, dtype=dtype).reshape((-1,))
    if int(vector.shape[0]) != int(total_size):
        raise ValueError(f"candidate has length {vector.shape[0]}, expected {total_size}")
    residual = vector
    for _ in range(2):
        for column in columns:
            q = jnp.asarray(column, dtype=dtype).reshape((-1,))
            residual = residual - q * jnp.vdot(q, residual)
    norm = float(jnp.linalg.norm(residual))
    if (not np.isfinite(norm)) or norm <= float(threshold):
        return None
    return residual / norm


def _block_schur_residual_equation_groups(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[str, tuple[int, ...]], ...]:
    """Return source QI block groups for setup-time Schur-like residual solves."""

    block_count = len(tuple(layout.block_sizes))
    specs: list[tuple[str, tuple[int, ...]]] = []
    if bool(config.block_schur_residual_equation_include_global):
        specs.append(("global", tuple(range(block_count))))
    if bool(config.block_schur_residual_equation_include_blocks):
        specs.extend(("block", (index,)) for index in range(block_count))
    if bool(config.block_schur_residual_equation_include_aggregates):
        factor = max(2, int(config.multilevel_aggregate_factor))
        for level in range(1, max(1, int(config.multilevel_max_levels))):
            width = factor**level
            for start in range(0, block_count, width):
                group = tuple(range(start, min(block_count, start + width)))
                if len(group) > 1:
                    specs.append(("aggregate", group))

    unique: list[tuple[str, tuple[int, ...]]] = []
    seen: set[tuple[int, ...]] = set()
    for kind, group in specs:
        if not group or group in seen:
            continue
        unique.append((kind, group))
        seen.add(group)
    return tuple(unique)


def _block_indicator_vector(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    *,
    total_size: int,
    dtype: Any,
) -> ArrayLike:
    offsets = tuple(int(value) for value in layout.block_offsets)
    start = offsets[int(block_index)]
    stop = offsets[int(block_index) + 1]
    size = max(1, int(stop) - int(start))
    values = jnp.zeros((int(total_size),), dtype=dtype)
    return values.at[start:stop].set(jnp.asarray(1.0 / np.sqrt(float(size)), dtype=dtype))


def _masked_group_residual(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    group: Sequence[int],
    *,
    total_size: int,
    dtype: Any,
) -> ArrayLike:
    offsets = tuple(int(value) for value in layout.block_offsets)
    residual_vec = jnp.asarray(residual, dtype=dtype).reshape((-1,))
    values = jnp.zeros((int(total_size),), dtype=dtype)
    for block_index in group:
        start = offsets[int(block_index)]
        stop = offsets[int(block_index) + 1]
        values = values.at[start:stop].set(residual_vec[start:stop])
    return values


def _block_schur_trial_vectors(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    group: Sequence[int],
    *,
    total_size: int,
    dtype: Any,
) -> ArrayLike:
    """Build tiny source-space trial columns for one QI block aggregate."""

    columns: list[ArrayLike] = []
    labels: list[str] = []
    for block_index in group:
        _append_normalized_candidate(
            columns,
            labels,
            _block_indicator_vector(layout, int(block_index), total_size=total_size, dtype=dtype),
            f"block_indicator:{int(block_index)}",
            total_size=total_size,
            dtype=dtype,
        )
    _append_normalized_candidate(
        columns,
        labels,
        _masked_group_residual(layout, residual, group, total_size=total_size, dtype=dtype),
        "masked_group_residual",
        total_size=total_size,
        dtype=dtype,
    )
    if not columns:
        return jnp.zeros((int(total_size), 0), dtype=dtype)
    return jnp.stack(tuple(columns), axis=1)


def _block_schur_trial_columns(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    group: Sequence[int],
    *,
    total_size: int,
    dtype: Any,
    include_indicators: bool,
    label_prefix: str,
) -> tuple[tuple[ArrayLike, str], ...]:
    """Return labeled source-space columns for one block-Schur group.

    The coupled residual equation uses these columns as a bounded source space
    ``D`` and caches ``A D``.  Keeping labels here avoids relying on column
    order when the coupled space is rank-gated.
    """

    columns: list[tuple[ArrayLike, str]] = []
    labels: list[str] = []
    raw_columns: list[ArrayLike] = []
    if bool(include_indicators):
        for block_index in group:
            _append_normalized_candidate(
                raw_columns,
                labels,
                _block_indicator_vector(layout, int(block_index), total_size=total_size, dtype=dtype),
                f"{label_prefix}:block_indicator:{int(block_index)}",
                total_size=total_size,
                dtype=dtype,
            )
    _append_normalized_candidate(
        raw_columns,
        labels,
        _masked_group_residual(layout, residual, group, total_size=total_size, dtype=dtype),
        f"{label_prefix}:masked_group_residual",
        total_size=total_size,
        dtype=dtype,
    )
    columns.extend((column, label) for column, label in zip(raw_columns, labels, strict=True))
    return tuple(columns)


def _build_coupled_block_schur_residual_basis(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    groups: Sequence[tuple[str, tuple[int, ...]]],
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
    max_rank: int,
    threshold: float,
) -> tuple[RHS1QICoarseBasis | None, int, float]:
    """Build one coupled block/aggregate Schur residual-equation basis.

    Earlier QI probes built one Schur-like direction per group in sequence. That
    is robust but misses coupled block interactions and forces the Krylov solve
    to rediscover them. This builder instead forms a bounded source space from
    the largest residual-carrying block and aggregate groups, solves the coupled
    reduced residual equation ``min ||r - A D c||``, and accepts the whole
    rank-gated ``D`` space only when the true setup residual is reduced.
    """

    residual_vec = jnp.asarray(residual, dtype=dtype).reshape((-1,))
    residual_norm = float(jnp.linalg.norm(residual_vec))
    if residual_norm <= 0.0:
        return None, 0, float("inf")

    def group_norm(item: tuple[str, tuple[int, ...]]) -> tuple[int, float, int]:
        kind, group = item
        masked = _masked_group_residual(layout, residual_vec, group, total_size=total_size, dtype=dtype)
        # Keep the global group first when requested, then prioritize blocks and
        # aggregates that actually carry the remaining residual.
        global_priority = 0 if kind == "global" else 1
        return (global_priority, -float(jnp.linalg.norm(masked)), int(group[0]) if group else 0)

    ordered_groups = tuple(sorted(groups, key=group_norm))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    max_columns = max(1, int(max_rank))
    for kind, group in ordered_groups:
        if len(columns) >= max_columns:
            break
        group_label = ",".join(str(index) for index in group)
        # The all-block global group can create hundreds of nearly redundant
        # indicator columns. For the coupled coarse solve, the masked global
        # residual is the useful source direction; block indicators are kept for
        # block/aggregate groups where they expose inter-block Schur coupling.
        include_indicators = kind != "global"
        for column, label in _block_schur_trial_columns(
            layout,
            residual_vec,
            group,
            total_size=int(total_size),
            dtype=dtype,
            include_indicators=include_indicators,
            label_prefix=f"block_schur_coupled:{kind}:{group_label}",
        ):
            if len(columns) >= max_columns:
                break
            columns.append(column)
            labels.append(label)

    if not columns:
        return None, 0, float("inf")
    candidate_count = len(columns)
    candidate_matrix = jnp.stack(tuple(columns), axis=1)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidate_matrix,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=max_rank,
    )
    if int(basis.metadata.rank) <= 0:
        return None, candidate_count, float("inf")
    action = _operator_on_basis(operator_matvec, basis.vectors, shape=shape, dtype=dtype)
    coefficients = _regularized_least_squares(
        action,
        residual_vec,
        rcond=float(config.regularization_rcond),
    )
    next_residual = residual_vec - action @ coefficients
    next_norm = float(jnp.linalg.norm(next_residual))
    if next_norm >= residual_norm - float(threshold):
        return None, candidate_count, next_norm
    return basis, candidate_count, next_norm


def _build_block_schur_residual_equation_bases_from_metadata(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    geometry_metadata: Mapping[str, object] | None,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    total_size: int,
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[tuple[RHS1QICoarseBasis, ...], int, int, tuple[int, ...], int]:
    """Build residual-equation stages from QI block/aggregate Schur probes.

    Each setup stage solves a small source-space least-squares problem
    ``min ||r_seed - A D_g c||`` over QI block indicator and masked-residual
    columns for one block group ``g``.  The resulting correction direction is
    cached as a basis column; apply-time only reuses the cached ``A Q`` action.
    """

    if int(shape[0]) != int(shape[1]):
        raise ValueError("block-Schur residual equation requires a square operator")
    layout = _multilevel_layout_from_metadata(geometry_metadata=geometry_metadata, total_size=total_size)
    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(total_size):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")

    groups = _block_schur_residual_equation_groups(layout, config=config)
    if not groups:
        raise ValueError("block-Schur residual equation requires at least one QI block group")

    max_rank = max(1, int(config.block_schur_residual_equation_max_rank))
    residual_norm = float(jnp.linalg.norm(residual))
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_norm))

    coupled_basis, coupled_candidate_count, coupled_residual_norm = _build_coupled_block_schur_residual_basis(
        operator_matvec=operator_matvec,
        layout=layout,
        residual=residual,
        groups=groups,
        shape=shape,
        total_size=int(total_size),
        dtype=dtype,
        config=config,
        max_rank=max_rank,
        threshold=threshold,
    )
    accepted_columns: list[ArrayLike] = []
    stage_bases: list[RHS1QICoarseBasis] = []
    remaining = residual
    candidate_count = 0

    for kind, group in groups:
        if len(stage_bases) >= max_rank:
            break
        trial_vectors = _block_schur_trial_vectors(
            layout,
            residual,
            group,
            total_size=int(total_size),
            dtype=dtype,
        )
        if int(trial_vectors.shape[1]) <= 0:
            continue
        trial_action = _operator_on_basis(operator_matvec, trial_vectors, shape=shape, dtype=dtype)
        coefficients = _regularized_least_squares(
            trial_action,
            remaining,
            rcond=float(config.regularization_rcond),
        )
        candidate = _orthogonalized_candidate(
            trial_vectors @ coefficients,
            accepted_columns,
            threshold=threshold,
            total_size=int(total_size),
            dtype=dtype,
        )
        if candidate is None:
            continue
        action = jnp.asarray(operator_matvec(candidate), dtype=dtype).reshape((-1,))
        action_norm = float(jnp.linalg.norm(action))
        if (not np.isfinite(action_norm)) or action_norm <= threshold:
            continue
        stage_coefficients = _regularized_least_squares(
            action.reshape((-1, 1)),
            remaining,
            rcond=float(config.regularization_rcond),
        )
        next_remaining = remaining - action * stage_coefficients[0]
        if float(jnp.linalg.norm(next_remaining)) >= float(jnp.linalg.norm(remaining)) - threshold:
            continue

        group_label = ",".join(str(index) for index in group)
        stage_basis = orthonormalize_rhs1_qi_coarse_basis(
            candidate.reshape((int(total_size), 1)),
            labels=(f"block_schur_residual:{kind}:{group_label}",),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=1,
        )
        if int(stage_basis.metadata.rank) <= 0:
            continue
        stage_vector = jnp.asarray(stage_basis.vectors, dtype=dtype)[:, 0]
        accepted_columns.append(stage_vector)
        stage_bases.append(stage_basis)
        remaining = next_remaining
        candidate_count += 1

    stage_ranks = tuple(int(basis.metadata.rank) for basis in stage_bases)
    sequential_rank = int(sum(stage_ranks))
    sequential_residual_norm = float(jnp.linalg.norm(remaining)) if sequential_rank > 0 else float("inf")
    if coupled_basis is not None and int(coupled_basis.metadata.rank) > 0:
        # Keep the coupled Schur space only when it is at least as good as the
        # sequential fail-closed construction on the measured setup residual.
        if sequential_rank <= 0 or float(coupled_residual_norm) <= sequential_residual_norm + float(threshold):
            rank = int(coupled_basis.metadata.rank)
            return (coupled_basis,), int(len(groups)), int(coupled_candidate_count), (rank,), rank
    return tuple(stage_bases), int(len(groups)), int(candidate_count), stage_ranks, sequential_rank


def _enrich_basis_with_operator_krylov(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Append a bounded Arnoldi-like residual Krylov coarse space.

    This is an operator-reuse coarse construction, not a smoother.  It adds
    rank-gated directions generated from the actual current residual and then
    stores the final ``A V`` action for all subsequent least-squares coarse
    solves.  It is intended for hard QI seeds where physics moments plus block
    smoothers reduce the residual but do not expose enough of the slow operator
    subspace.
    """

    depth_count = max(0, int(config.operator_krylov_depth))

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    total_size = int(shape[1])
    if int(residual.shape[0]) != total_size:
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator size {total_size}")
    if int(shape[0]) != total_size:
        raise ValueError("operator-Krylov enrichment requires a square operator")

    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(basis.vectors, dtype=dtype)
    for index, label in enumerate(basis.metadata.accepted_labels):
        _append_normalized_candidate(
            columns,
            labels,
            base_vectors[:, int(index)],
            f"base:{label}",
            total_size=total_size,
            dtype=dtype,
        )

    residual_norm = float(jnp.linalg.norm(residual))
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_norm))
    rank_limit = config.max_rank if config.max_rank is not None else total_size
    candidate_count = 0
    current = _orthogonalized_candidate(
        residual,
        columns,
        threshold=threshold,
        total_size=total_size,
        dtype=dtype,
    )
    if current is None:
        return basis, 0
    if len(columns) < int(rank_limit):
        columns.append(current)
        labels.append("operator_krylov:0")
        candidate_count += 1
    else:
        return basis, 0

    for depth in range(depth_count):
        if len(columns) >= int(rank_limit):
            break
        action = jnp.asarray(operator_matvec(current), dtype=dtype).reshape((-1,))
        current = _orthogonalized_candidate(
            action,
            columns,
            threshold=threshold,
            total_size=total_size,
            dtype=dtype,
        )
        if current is None:
            break
        columns.append(current)
        labels.append(f"operator_krylov:{depth + 1}")
        candidate_count += 1

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=dtype)
    enriched = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )
    return enriched, candidate_count


def _enrich_basis_with_adjoint_krylov(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    residual_seed: ArrayLike,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Append a bounded adjoint-normal residual coarse space.

    The existing residual/operator-Krylov path spans ``{r, A r, ...}``.  That
    can miss non-normal left-error modes.  This enrichment instead starts from
    ``A.T r`` and then builds ``(A.T A)^k A.T r`` directions.  Setup uses JAX
    VJP once per bounded candidate; the reusable preconditioner still only
    stores ``Q`` and ``A Q`` and remains a forward-operator apply.
    """

    transpose_source = _normalize_adjoint_krylov_transpose_source(config.adjoint_krylov_transpose_source)
    if transpose_source == "off":
        return basis, 0
    depth_count = max(0, int(config.adjoint_krylov_depth))
    total_size = int(shape[1])
    if int(shape[0]) != total_size:
        raise ValueError("adjoint-Krylov enrichment requires a square operator")

    residual = jnp.asarray(residual_seed, dtype=dtype).reshape((-1,))
    if int(residual.shape[0]) != int(shape[0]):
        raise ValueError(f"residual_seed length {residual.shape[0]} does not match operator rows {shape[0]}")

    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(basis.vectors, dtype=dtype)
    for index, label in enumerate(basis.metadata.accepted_labels):
        _append_normalized_candidate(
            columns,
            labels,
            base_vectors[:, int(index)],
            f"base:{label}",
            total_size=total_size,
            dtype=dtype,
        )

    residual_norm = float(jnp.linalg.norm(residual))
    threshold = max(float(config.basis_atol), float(config.basis_rtol) * max(1.0, residual_norm))
    rank_limit = config.max_rank if config.max_rank is not None else total_size
    candidate_count = 0
    current = _autodiff_transpose_matvec(
        operator_matvec,
        residual,
        shape=shape,
        dtype=dtype,
    )
    for depth in range(depth_count + 1):
        if len(columns) >= int(rank_limit):
            break
        candidate = _orthogonalized_candidate(
            current,
            columns,
            threshold=threshold,
            total_size=total_size,
            dtype=dtype,
        )
        if candidate is None:
            break
        columns.append(candidate)
        labels.append(f"adjoint_krylov:{depth}")
        candidate_count += 1
        if depth >= depth_count:
            break
        action = jnp.asarray(operator_matvec(candidate), dtype=dtype).reshape((-1,))
        current = _autodiff_transpose_matvec(
            operator_matvec,
            action,
            shape=shape,
            dtype=dtype,
        )

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=dtype)
    enriched = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )
    return enriched, candidate_count


def _enrich_basis_with_operator_actions(
    *,
    operator_matvec: Callable[[ArrayLike], ArrayLike],
    basis: RHS1QICoarseBasis,
    shape: tuple[int, int],
    dtype: Any,
    config: RHS1QIDevicePreconditionerConfig,
) -> tuple[RHS1QICoarseBasis, int]:
    """Append rank-gated operator-image directions to the coarse space.

    QI hard seeds have shown slow modes that are not well represented by
    constants or block moments alone.  This enrichment builds a small
    device-resident polynomial correction space ``{Q, A Q, A^2 Q, ...}``.  The
    final state still stores only the rank-gated basis and its reusable
    ``A Q_aug`` action, so apply-time cost stays independent of the enrichment
    work done during setup.
    """

    depth_count = max(0, int(config.operator_action_enrichment_depth))
    if depth_count <= 0 or int(basis.metadata.rank) <= 0:
        return basis, 0

    total_size = int(shape[1])
    if int(shape[0]) != total_size:
        raise ValueError("operator-action enrichment requires a square operator")

    columns: list[ArrayLike] = []
    labels: list[str] = []
    base_vectors = jnp.asarray(basis.vectors, dtype=dtype)
    base_labels = tuple(str(label) for label in basis.metadata.accepted_labels)
    for index, label in enumerate(base_labels):
        _append_normalized_candidate(
            columns,
            labels,
            base_vectors[:, int(index)],
            f"base:{label}",
            total_size=total_size,
            dtype=dtype,
        )

    frontier = base_vectors
    candidate_count = 0
    frontier_labels = base_labels
    for depth in range(depth_count):
        if int(frontier.shape[1]) <= 0:
            break
        action_vectors = _operator_on_basis(operator_matvec, frontier, shape=shape, dtype=dtype)
        next_labels: list[str] = []
        for index, label in enumerate(frontier_labels):
            before = len(columns)
            action_label = f"operator_action:{depth + 1}:{label}"
            _append_normalized_candidate(
                columns,
                labels,
                action_vectors[:, int(index)],
                action_label,
                total_size=total_size,
                dtype=dtype,
            )
            candidate_count += len(columns) - before
            next_labels.append(action_label)
        frontier = action_vectors
        frontier_labels = tuple(next_labels)

    if columns:
        candidates = jnp.stack(tuple(columns), axis=1)
    else:
        candidates = jnp.zeros((total_size, 0), dtype=dtype)
    enriched = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(config.basis_rtol),
        atol=float(config.basis_atol),
        max_rank=config.max_rank,
    )
    return enriched, candidate_count


def _basis_from_value(
    coarse_basis: RHS1QICoarseBasis | ArrayLike | None,
    *,
    total_size: int,
    dtype: Any,
    labels: Sequence[str] | None,
    config: RHS1QIDevicePreconditionerConfig,
) -> RHS1QICoarseBasis:
    if isinstance(coarse_basis, RHS1QICoarseBasis):
        basis = coarse_basis
    elif coarse_basis is None:
        basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.zeros((int(total_size), 0), dtype=dtype),
            labels=(),
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    else:
        basis = orthonormalize_rhs1_qi_coarse_basis(
            jnp.asarray(coarse_basis, dtype=dtype),
            labels=labels,
            rtol=float(config.basis_rtol),
            atol=float(config.basis_atol),
            max_rank=config.max_rank,
        )
    vectors = jnp.asarray(basis.vectors, dtype=dtype)
    if vectors.ndim != 2 or int(vectors.shape[0]) != int(total_size):
        raise ValueError(f"coarse basis must have shape ({total_size}, rank)")
    if vectors.dtype != jnp.dtype(dtype):
        basis = RHS1QICoarseBasis(vectors=vectors, metadata=basis.metadata)
    return basis


def _metadata_keys(value: Mapping[str, object] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(sorted(str(key) for key in value.keys()))


def setup_rhs1_qi_device_preconditioner(
    *,
    operator: DeviceCSR | Callable[[ArrayLike], ArrayLike],
    coarse_basis: RHS1QICoarseBasis | ArrayLike | None = None,
    local_smoother: RHS1QIDeviceJacobiSmoother | None = None,
    coarse_labels: Sequence[str] | None = None,
    residual_seed: ArrayLike | None = None,
    total_size: int | None = None,
    dtype: Any = jnp.float64,
    operator_metadata: Mapping[str, object] | None = None,
    geometry_metadata: Mapping[str, object] | None = None,
    config: RHS1QIDevicePreconditionerConfig | None = None,
) -> RHS1QIDevicePreconditionerState:
    """Build a device-resident QI preconditioner state.

    The returned state is intentionally standalone and does not change any
    production solver defaults.  It can be closed over by JAX transforms and later
    wired into ``v3_driver.py`` behind an explicit opt-in.
    """

    config_use = RHS1QIDevicePreconditionerConfig() if config is None else config
    coarse_solver = _normalize_coarse_solver(config_use.coarse_solver)
    composition = _normalize_composition(config_use.composition)
    local_smoother_kind_requested = str(config_use.local_smoother_kind).strip().lower().replace("-", "_")
    matrix_free_residual_smoother_tokens = {
        "matrix_free",
        "matrix_free_residual",
        "matrix_free_minres",
        "matrix_free_richardson",
        "residual_polynomial",
    }
    matrix_free_block_smoother_tokens = {
        "adaptive_residual_equation",
        "adaptive_residual_galerkin",
        "block_minres",
        "matrix_free_block",
        "matrix_free_block_minres",
        "matrix_free_block_minres_hybrid",
        "block_angular_radial",
        "projected_block_minres",
    }
    matrix_free_smoother_tokens = matrix_free_residual_smoother_tokens | matrix_free_block_smoother_tokens
    if local_smoother_kind_requested not in {
        "auto",
        "device_jacobi",
        "jacobi",
        "none",
        "coarse_only",
        *matrix_free_smoother_tokens,
    }:
        raise ValueError("local_smoother_kind must be 'auto', 'device_jacobi', 'matrix_free_minres', or 'none'")
    damping = float(config_use.damping)
    if not np.isfinite(damping) or damping <= 0.0:
        raise ValueError("damping must be finite and positive")
    rcond = float(config_use.regularization_rcond)
    if not np.isfinite(rcond) or rcond < 0.0:
        raise ValueError("regularization_rcond must be finite and non-negative")

    if isinstance(operator, DeviceCSR):
        operator_csr: DeviceCSR | None = operator
        operator_matvec: Callable[[ArrayLike], ArrayLike] = operator.matvec
        shape = tuple(int(v) for v in operator.shape)
        dtype_use = operator.data.dtype
        nnz = int(operator.nnz)
        operator_source = "device_csr"
    else:
        if total_size is None:
            if isinstance(coarse_basis, RHS1QICoarseBasis):
                total_size = int(coarse_basis.vectors.shape[0])
            elif coarse_basis is not None:
                total_size = int(jnp.asarray(coarse_basis).shape[0])
        if total_size is None or int(total_size) <= 0:
            raise ValueError("total_size is required for a matrix-free QI device preconditioner")
        operator_csr = None
        operator_matvec = operator
        shape = (int(total_size), int(total_size))
        dtype_use = jnp.dtype(dtype)
        nnz = 0
        operator_source = "matrix_free"

    if int(shape[0]) != int(shape[1]):
        raise ValueError("device QI preconditioner requires a square operator")

    smoother = local_smoother
    if smoother is None and operator_csr is not None and local_smoother_kind_requested not in {"none", "coarse_only"}:
        smoother = build_rhs1_qi_device_jacobi_smoother(
            operator_csr,
            damping=float(config_use.jacobi_damping),
            sweeps=int(config_use.jacobi_sweeps),
            step_policy=str(config_use.jacobi_step_policy),
            diagonal_floor=float(config_use.jacobi_diagonal_floor),
            require_all_diagonal=bool(config_use.jacobi_require_all_diagonal),
        )
    elif (
        smoother is not None
        and isinstance(smoother, RHS1QIDeviceJacobiSmoother)
        and smoother.operator.shape != shape
    ):
        raise ValueError("local_smoother operator shape must match operator shape")
    elif smoother is None and operator_csr is None and local_smoother_kind_requested in matrix_free_block_smoother_tokens:
        projected_config = config_use
        if local_smoother_kind_requested in {"adaptive_residual_equation", "adaptive_residual_galerkin"}:
            projected_config = replace(
                config_use,
                matrix_free_block_smoother_grouping="block_hierarchy",
            )
        smoother = _build_matrix_free_projected_residual_smoother(
            operator_matvec=operator_matvec,
            shape=shape,
            dtype=dtype_use,
            geometry_metadata=geometry_metadata,
            config=projected_config,
        )
    elif (
        smoother is None
        and operator_csr is None
        and local_smoother_kind_requested in matrix_free_residual_smoother_tokens
    ):
        smoother = _build_matrix_free_residual_smoother(
            operator_matvec=operator_matvec,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    if smoother is None and local_smoother_kind_requested in {"device_jacobi", "jacobi"}:
        raise ValueError("device_jacobi local smoother requires a DeviceCSR operator")

    basis = _basis_from_value(
        coarse_basis,
        total_size=int(shape[1]),
        dtype=dtype_use,
        labels=coarse_labels,
        config=config_use,
    )
    multilevel_coarse_level_count = 0
    multilevel_coarse_candidate_count = 0
    multilevel_coarse_rank = 0
    if bool(config_use.multilevel_coarse):
        multilevel_basis, multilevel_coarse_level_count, multilevel_coarse_candidate_count = (
            _build_multilevel_coarse_basis_from_metadata(
                geometry_metadata=geometry_metadata,
                total_size=int(shape[1]),
                dtype=dtype_use,
                config=config_use,
            )
        )
        multilevel_coarse_rank = int(multilevel_basis.metadata.rank)
        basis = _combine_coarse_bases(
            basis,
            multilevel_basis,
            dtype=dtype_use,
            config=config_use,
        )
    residual_snapshot_candidate_count = 0
    residual_snapshot_rank = 0
    residual_snapshot_group_count = 0
    if bool(config_use.residual_snapshot_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_snapshot_enrichment=True")
        snapshot_basis, residual_snapshot_candidate_count, residual_snapshot_group_count = (
            _build_residual_snapshot_basis_from_metadata(
                operator_matvec=operator_matvec,
                geometry_metadata=geometry_metadata,
                residual_seed=residual_seed,
                shape=shape,
                total_size=int(shape[1]),
                dtype=dtype_use,
                config=config_use,
            )
        )
        residual_snapshot_rank = int(snapshot_basis.metadata.rank)
        basis = _combine_coarse_bases(
            basis,
            snapshot_basis,
            dtype=dtype_use,
            config=config_use,
        )
    block_schur_residual_candidate_count = 0
    block_schur_residual_rank = 0
    block_schur_residual_group_count = 0
    if bool(config_use.block_schur_residual_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when block_schur_residual_enrichment=True")
        schur_basis, block_schur_residual_candidate_count, block_schur_residual_group_count = (
            _build_block_schur_residual_basis_from_metadata(
                operator_matvec=operator_matvec,
                geometry_metadata=geometry_metadata,
                residual_seed=residual_seed,
                shape=shape,
                total_size=int(shape[1]),
                dtype=dtype_use,
                config=config_use,
            )
        )
        block_schur_residual_rank = int(schur_basis.metadata.rank)
        basis = _combine_coarse_bases(
            basis,
            schur_basis,
            dtype=dtype_use,
            config=config_use,
        )
    residual_enrichment_candidate_count = 0
    if bool(config_use.residual_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_enrichment=True")
        basis, residual_enrichment_candidate_count = _enrich_basis_with_residual(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    recycle_enrichment_candidate_count = 0
    if bool(config_use.recycle_enrichment) and int(config_use.recycle_enrichment_cycles) > 0:
        if residual_seed is None:
            raise ValueError("residual_seed is required when recycle_enrichment=True")
        basis, recycle_enrichment_candidate_count = _enrich_basis_with_recycle_residuals(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    operator_krylov_candidate_count = 0
    if bool(config_use.operator_krylov_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when operator_krylov_enrichment=True")
        basis, operator_krylov_candidate_count = _enrich_basis_with_operator_krylov(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    adjoint_krylov_transpose_source = _normalize_adjoint_krylov_transpose_source(
        config_use.adjoint_krylov_transpose_source
    )
    adjoint_krylov_candidate_count = 0
    if bool(config_use.adjoint_krylov_enrichment):
        if residual_seed is None:
            raise ValueError("residual_seed is required when adjoint_krylov_enrichment=True")
        basis, adjoint_krylov_candidate_count = _enrich_basis_with_adjoint_krylov(
            operator_matvec=operator_matvec,
            basis=basis,
            residual_seed=residual_seed,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    operator_action_enrichment_candidate_count = 0
    if bool(config_use.operator_action_enrichment):
        basis, operator_action_enrichment_candidate_count = _enrich_basis_with_operator_actions(
            operator_matvec=operator_matvec,
            basis=basis,
            shape=shape,
            dtype=dtype_use,
            config=config_use,
        )
    residual_equation_bases: tuple[RHS1QICoarseBasis, ...] = ()
    residual_equation_actions: tuple[ArrayLike, ...] = ()
    residual_equation_coarse_operators: tuple[ArrayLike, ...] = ()
    residual_equation_stage_solvers: tuple[str, ...] = ()
    residual_equation_stage_count = 0
    residual_equation_rank = 0
    residual_equation_stage_ranks: tuple[int, ...] = ()
    residual_equation_order = _normalize_multilevel_residual_equation_order(
        config_use.multilevel_residual_equation_order
    )
    residual_equation_solver = _normalize_multilevel_residual_equation_solver(
        config_use.multilevel_residual_equation_solver
    )
    global_moment_residual_equation_solver = _normalize_multilevel_residual_equation_solver(
        config_use.global_moment_residual_equation_solver
    )
    global_moment_residual_equation_candidate_count = 0
    global_moment_residual_equation_rank = 0
    global_moment_residual_equation_stage_ranks: tuple[int, ...] = ()
    global_moment_residual_equation_condition_estimate = float("inf")
    if bool(config_use.global_moment_residual_equation):
        if residual_seed is None:
            raise ValueError("residual_seed is required when global_moment_residual_equation=True")
        (
            global_moment_bases,
            global_moment_residual_equation_candidate_count,
            global_moment_residual_equation_stage_ranks,
            global_moment_residual_equation_rank,
            global_moment_residual_equation_condition_estimate,
        ) = _build_global_moment_residual_equation_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + global_moment_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            global_moment_residual_equation_solver for _ in global_moment_bases
        )
    residual_galerkin_equation_solver = _normalize_multilevel_residual_equation_solver(
        config_use.residual_galerkin_equation_solver
    )
    residual_galerkin_equation_candidate_count = 0
    residual_galerkin_equation_rank = 0
    residual_galerkin_equation_stage_count = 0
    residual_galerkin_equation_stage_ranks: tuple[int, ...] = ()
    residual_galerkin_equation_condition_estimate = float("inf")
    residual_galerkin_equation_residual_before = float("inf")
    residual_galerkin_equation_residual_after = float("inf")
    if bool(config_use.residual_galerkin_equation):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_galerkin_equation=True")
        (
            residual_galerkin_bases,
            residual_galerkin_equation_candidate_count,
            residual_galerkin_equation_stage_count,
            residual_galerkin_equation_stage_ranks,
            residual_galerkin_equation_rank,
            residual_galerkin_equation_condition_estimate,
            residual_galerkin_equation_residual_before,
            residual_galerkin_equation_residual_after,
        ) = _build_residual_galerkin_equation_basis_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + residual_galerkin_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            residual_galerkin_equation_solver for _ in residual_galerkin_bases
        )
    residual_region_bounce_coarse_solver = _normalize_multilevel_residual_equation_solver(
        config_use.residual_region_bounce_coarse_solver
    )
    residual_region_bounce_coarse_candidate_count = 0
    residual_region_bounce_coarse_rank = 0
    residual_region_bounce_coarse_stage_ranks: tuple[int, ...] = ()
    residual_region_bounce_coarse_condition_estimate = float("inf")
    residual_region_bounce_coarse_residual_before = float("inf")
    residual_region_bounce_coarse_residual_after = float("inf")
    if bool(config_use.residual_region_bounce_coarse):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_region_bounce_coarse=True")
        (
            residual_region_bounce_bases,
            residual_region_bounce_coarse_candidate_count,
            residual_region_bounce_coarse_stage_ranks,
            residual_region_bounce_coarse_rank,
            residual_region_bounce_coarse_condition_estimate,
            residual_region_bounce_coarse_residual_before,
            residual_region_bounce_coarse_residual_after,
        ) = _build_residual_region_bounce_coarse_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + residual_region_bounce_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            residual_region_bounce_coarse_solver for _ in residual_region_bounce_bases
        )
    active_pattern_coarse_solver = _normalize_multilevel_residual_equation_solver(
        config_use.active_pattern_coarse_solver
    )
    active_pattern_coarse_candidate_count = 0
    active_pattern_coarse_rank = 0
    active_pattern_coarse_stage_ranks: tuple[int, ...] = ()
    active_pattern_coarse_condition_estimate = float("inf")
    active_pattern_coarse_residual_before = float("inf")
    active_pattern_coarse_residual_after = float("inf")
    if bool(config_use.active_pattern_coarse):
        if residual_seed is None:
            raise ValueError("residual_seed is required when active_pattern_coarse=True")
        (
            active_pattern_bases,
            active_pattern_coarse_candidate_count,
            active_pattern_coarse_stage_ranks,
            active_pattern_coarse_rank,
            active_pattern_coarse_condition_estimate,
            active_pattern_coarse_residual_before,
            active_pattern_coarse_residual_after,
        ) = _build_active_pattern_coarse_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + active_pattern_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            active_pattern_coarse_solver for _ in active_pattern_bases
        )
    phase_space_residual_equation_solver = _normalize_multilevel_residual_equation_solver(
        config_use.phase_space_residual_equation_solver
    )
    phase_space_residual_equation_candidate_count = 0
    phase_space_residual_equation_rank = 0
    phase_space_residual_equation_stage_ranks: tuple[int, ...] = ()
    phase_space_residual_equation_condition_estimate = float("inf")
    phase_space_residual_equation_residual_before = float("inf")
    phase_space_residual_equation_residual_after = float("inf")
    if bool(config_use.phase_space_residual_equation):
        if residual_seed is None:
            raise ValueError("residual_seed is required when phase_space_residual_equation=True")
        (
            phase_space_bases,
            phase_space_residual_equation_candidate_count,
            phase_space_residual_equation_stage_ranks,
            phase_space_residual_equation_rank,
            phase_space_residual_equation_condition_estimate,
            phase_space_residual_equation_residual_before,
            phase_space_residual_equation_residual_after,
        ) = _build_phase_space_residual_equation_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + phase_space_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            phase_space_residual_equation_solver for _ in phase_space_bases
        )
    residual_snapshot_residual_equation_solver = _normalize_multilevel_residual_equation_solver(
        config_use.residual_snapshot_residual_equation_solver
    )
    residual_snapshot_residual_equation_group_count = 0
    residual_snapshot_residual_equation_candidate_count = 0
    residual_snapshot_residual_equation_rank = 0
    residual_snapshot_residual_equation_stage_ranks: tuple[int, ...] = ()
    if bool(config_use.residual_snapshot_residual_equation):
        if residual_seed is None:
            raise ValueError("residual_seed is required when residual_snapshot_residual_equation=True")
        (
            residual_snapshot_bases,
            residual_snapshot_residual_equation_group_count,
            residual_snapshot_residual_equation_candidate_count,
            residual_snapshot_residual_equation_stage_ranks,
            residual_snapshot_residual_equation_rank,
        ) = _build_residual_snapshot_residual_equation_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + residual_snapshot_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            residual_snapshot_residual_equation_solver for _ in residual_snapshot_bases
        )
    block_schur_residual_equation_group_count = 0
    block_schur_residual_equation_candidate_count = 0
    block_schur_residual_equation_rank = 0
    block_schur_residual_equation_stage_ranks: tuple[int, ...] = ()
    if bool(config_use.block_schur_residual_equation):
        if residual_seed is None:
            raise ValueError("residual_seed is required when block_schur_residual_equation=True")
        (
            block_schur_bases,
            block_schur_residual_equation_group_count,
            block_schur_residual_equation_candidate_count,
            block_schur_residual_equation_stage_ranks,
            block_schur_residual_equation_rank,
        ) = _build_block_schur_residual_equation_bases_from_metadata(
            operator_matvec=operator_matvec,
            geometry_metadata=geometry_metadata,
            residual_seed=residual_seed,
            shape=shape,
            total_size=int(shape[1]),
            dtype=dtype_use,
            config=config_use,
        )
        residual_equation_bases = residual_equation_bases + block_schur_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            "action_lstsq" for _ in block_schur_bases
        )
    if bool(config_use.multilevel_residual_equation):
        multilevel_residual_equation_bases: tuple[RHS1QICoarseBasis, ...]
        multilevel_residual_equation_bases, residual_equation_stage_count, residual_equation_rank, residual_equation_stage_ranks = (
            _build_multilevel_residual_equation_bases_from_metadata(
                geometry_metadata=geometry_metadata,
                total_size=int(shape[1]),
                dtype=dtype_use,
                config=config_use,
            )
        )
        residual_equation_bases = residual_equation_bases + multilevel_residual_equation_bases
        residual_equation_stage_solvers = residual_equation_stage_solvers + tuple(
            residual_equation_solver for _ in multilevel_residual_equation_bases
        )
    residual_equation_action_list: list[ArrayLike] = []
    residual_equation_coarse_operator_list: list[ArrayLike] = []
    for level_basis in residual_equation_bases:
        level_q = jnp.asarray(level_basis.vectors, dtype=dtype_use)
        level_action = _operator_on_basis(operator_matvec, level_q, shape=shape, dtype=dtype_use)
        residual_equation_action_list.append(level_action)
        residual_equation_coarse_operator_list.append(jnp.conjugate(level_q).T @ level_action)
    residual_equation_actions = tuple(residual_equation_action_list)
    residual_equation_coarse_operators = tuple(residual_equation_coarse_operator_list)
    rank = int(basis.metadata.rank)
    if rank > 0:
        aq = _operator_on_basis(operator_matvec, basis.vectors, shape=shape, dtype=dtype_use)
        coarse_operator = jnp.conjugate(jnp.asarray(basis.vectors, dtype=dtype_use)).T @ aq
    else:
        aq = jnp.zeros((int(shape[0]), 0), dtype=dtype_use)
        coarse_operator = jnp.zeros((0, 0), dtype=dtype_use)

    if smoother is None:
        local_smoother_kind = "none"
        local_smoother_reason = "matrix_free_coarse_only"
    elif isinstance(smoother, RHS1QIDeviceJacobiSmoother):
        local_smoother_kind = "device_jacobi"
        local_smoother_reason = str(smoother.metadata.reason)
    elif isinstance(smoother, RHS1QIMatrixFreeProjectedResidualSmoother):
        local_smoother_kind = (
            "adaptive_residual_equation"
            if str(smoother.metadata.grouping) == "block_hierarchy"
            else "matrix_free_block_minres"
        )
        local_smoother_reason = str(smoother.metadata.reason)
    else:
        local_smoother_kind = "matrix_free_residual"
        local_smoother_reason = str(smoother.metadata.reason)
    if (
        bool(config_use.residual_snapshot_residual_equation)
        and residual_snapshot_residual_equation_rank > 0
    ):
        reason = "built_with_residual_snapshot_residual_equation"
    elif bool(config_use.global_moment_residual_equation) and global_moment_residual_equation_rank > 0:
        reason = "built_with_global_moment_residual_equation"
    elif bool(config_use.residual_galerkin_equation) and residual_galerkin_equation_rank > 0:
        reason = "built_with_residual_galerkin_equation"
    elif bool(config_use.residual_region_bounce_coarse) and residual_region_bounce_coarse_rank > 0:
        reason = "built_with_residual_region_bounce_coarse"
    elif bool(config_use.active_pattern_coarse) and active_pattern_coarse_rank > 0:
        reason = "built_with_active_pattern_coarse"
    elif bool(config_use.phase_space_residual_equation) and phase_space_residual_equation_rank > 0:
        reason = "built_with_phase_space_residual_equation"
    elif bool(config_use.block_schur_residual_equation) and block_schur_residual_equation_rank > 0:
        reason = "built_with_block_schur_residual_equation"
    elif (
        bool(config_use.multilevel_residual_equation)
        and residual_equation_rank > 0
        and residual_equation_solver == "galerkin"
    ):
        reason = "built_with_multilevel_galerkin_residual_equation"
    elif bool(config_use.multilevel_residual_equation) and residual_equation_rank > 0:
        reason = "built_with_multilevel_residual_equation"
    elif rank > 0 and block_schur_residual_rank > 0:
        reason = "built_with_block_schur_residual"
    elif rank > 0 and residual_snapshot_rank > 0:
        reason = "built_with_residual_snapshot"
    elif rank > 0 and multilevel_coarse_rank > 0:
        reason = "built_with_multilevel_coarse"
    elif rank > 0 and smoother is None:
        reason = "built_matrix_free_coarse_only"
    elif rank > 0:
        reason = "built_with_coarse"
    else:
        reason = "built_local_only" if smoother is not None else "built_empty"
    metadata = RHS1QIDevicePreconditionerMetadata(
        shape=shape,
        nnz=nnz,
        rank=rank,
        operator_source=operator_source,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        operator_on_basis_shape=tuple(int(v) for v in aq.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0,
        operator_on_basis_norm=float(jnp.linalg.norm(aq)) if rank > 0 else 0.0,
        regularization_rcond=rcond,
        damping=damping,
        coarse_solver=coarse_solver,
        composition=composition,
        local_smoother_kind=local_smoother_kind,
        local_smoother_reason=local_smoother_reason,
        device_resident=True,
        host_fallback_used=False,
        host_callback_free=True,
        operator_metadata_keys=_metadata_keys(operator_metadata),
        geometry_metadata_keys=_metadata_keys(geometry_metadata),
        accepted_basis_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
        residual_enrichment_enabled=bool(config_use.residual_enrichment),
        residual_enrichment_depth=max(0, int(config_use.residual_enrichment_depth)),
        residual_enrichment_candidate_count=int(residual_enrichment_candidate_count),
        recycle_enrichment_enabled=bool(config_use.recycle_enrichment),
        recycle_enrichment_cycles=max(0, int(config_use.recycle_enrichment_cycles)),
        recycle_enrichment_candidate_count=int(recycle_enrichment_candidate_count),
        operator_krylov_enrichment_enabled=bool(config_use.operator_krylov_enrichment),
        operator_krylov_depth=max(0, int(config_use.operator_krylov_depth)),
        operator_krylov_candidate_count=int(operator_krylov_candidate_count),
        adjoint_krylov_enrichment_enabled=bool(config_use.adjoint_krylov_enrichment),
        adjoint_krylov_depth=max(0, int(config_use.adjoint_krylov_depth)),
        adjoint_krylov_candidate_count=int(adjoint_krylov_candidate_count),
        adjoint_krylov_transpose_source=adjoint_krylov_transpose_source,
        operator_action_enrichment_enabled=bool(config_use.operator_action_enrichment),
        operator_action_enrichment_depth=max(0, int(config_use.operator_action_enrichment_depth)),
        operator_action_enrichment_candidate_count=int(operator_action_enrichment_candidate_count),
        multilevel_coarse_enabled=bool(config_use.multilevel_coarse),
        multilevel_coarse_level_count=int(multilevel_coarse_level_count),
        multilevel_coarse_candidate_count=int(multilevel_coarse_candidate_count),
        multilevel_coarse_rank=int(multilevel_coarse_rank),
        multilevel_residual_equation_enabled=bool(
            config_use.multilevel_residual_equation and residual_equation_rank > 0
        ),
        multilevel_residual_equation_stage_count=int(residual_equation_stage_count),
        multilevel_residual_equation_rank=int(residual_equation_rank),
        multilevel_residual_equation_stage_ranks=tuple(int(v) for v in residual_equation_stage_ranks),
        multilevel_residual_equation_order=residual_equation_order,
        multilevel_residual_equation_solver=residual_equation_solver,
        multilevel_residual_equation_include_global=bool(
            config_use.multilevel_residual_equation_include_global
        ),
        global_moment_residual_equation_enabled=bool(
            config_use.global_moment_residual_equation and global_moment_residual_equation_rank > 0
        ),
        global_moment_residual_equation_candidate_count=int(
            global_moment_residual_equation_candidate_count
        ),
        global_moment_residual_equation_rank=int(global_moment_residual_equation_rank),
        global_moment_residual_equation_stage_ranks=tuple(
            int(v) for v in global_moment_residual_equation_stage_ranks
        ),
        global_moment_residual_equation_solver=global_moment_residual_equation_solver,
        global_moment_residual_equation_include_profile=bool(
            config_use.global_moment_residual_equation_include_profile
        ),
        global_moment_residual_equation_include_current=bool(
            config_use.global_moment_residual_equation_include_current
        ),
        global_moment_residual_equation_include_tail=bool(
            config_use.global_moment_residual_equation_include_tail
        ),
        global_moment_residual_equation_condition_estimate=float(
            global_moment_residual_equation_condition_estimate
        ),
        residual_galerkin_equation_enabled=bool(
            config_use.residual_galerkin_equation and residual_galerkin_equation_rank > 0
        ),
        residual_galerkin_equation_candidate_count=int(
            residual_galerkin_equation_candidate_count
        ),
        residual_galerkin_equation_rank=int(residual_galerkin_equation_rank),
        residual_galerkin_equation_stage_count=int(residual_galerkin_equation_stage_count),
        residual_galerkin_equation_stage_ranks=tuple(
            int(v) for v in residual_galerkin_equation_stage_ranks
        ),
        residual_galerkin_equation_solver=residual_galerkin_equation_solver,
        residual_galerkin_equation_condition_estimate=float(
            residual_galerkin_equation_condition_estimate
        ),
        residual_galerkin_equation_residual_before=float(
            residual_galerkin_equation_residual_before
        ),
        residual_galerkin_equation_residual_after=float(
            residual_galerkin_equation_residual_after
        ),
        phase_space_residual_equation_enabled=bool(
            config_use.phase_space_residual_equation and phase_space_residual_equation_rank > 0
        ),
        phase_space_residual_equation_candidate_count=int(
            phase_space_residual_equation_candidate_count
        ),
        phase_space_residual_equation_rank=int(phase_space_residual_equation_rank),
        phase_space_residual_equation_stage_count=len(phase_space_residual_equation_stage_ranks),
        phase_space_residual_equation_stage_ranks=tuple(
            int(v) for v in phase_space_residual_equation_stage_ranks
        ),
        phase_space_residual_equation_max_rank=int(
            config_use.phase_space_residual_equation_max_rank
        ),
        phase_space_residual_equation_solver=phase_space_residual_equation_solver,
        phase_space_residual_equation_include_global=bool(
            config_use.phase_space_residual_equation_include_global
        ),
        phase_space_residual_equation_trapped_boundary_fraction=float(
            config_use.phase_space_residual_equation_trapped_boundary_fraction
        ),
        phase_space_residual_equation_include_trapped=bool(
            config_use.phase_space_residual_equation_include_trapped
        ),
        phase_space_residual_equation_include_passing=bool(
            config_use.phase_space_residual_equation_include_passing
        ),
        phase_space_residual_equation_include_boundary=bool(
            config_use.phase_space_residual_equation_include_boundary
        ),
        phase_space_residual_equation_include_even=bool(
            config_use.phase_space_residual_equation_include_even
        ),
        phase_space_residual_equation_include_odd=bool(
            config_use.phase_space_residual_equation_include_odd
        ),
        phase_space_residual_equation_include_radial=bool(
            config_use.phase_space_residual_equation_include_radial
        ),
        phase_space_residual_equation_include_species=bool(
            config_use.phase_space_residual_equation_include_species
        ),
        phase_space_residual_equation_condition_estimate=float(
            phase_space_residual_equation_condition_estimate
        ),
        phase_space_residual_equation_residual_before=float(
            phase_space_residual_equation_residual_before
        ),
        phase_space_residual_equation_residual_after=float(
            phase_space_residual_equation_residual_after
        ),
        residual_region_bounce_coarse_enabled=bool(
            config_use.residual_region_bounce_coarse and residual_region_bounce_coarse_rank > 0
        ),
        residual_region_bounce_coarse_candidate_count=int(
            residual_region_bounce_coarse_candidate_count
        ),
        residual_region_bounce_coarse_rank=int(residual_region_bounce_coarse_rank),
        residual_region_bounce_coarse_stage_count=len(residual_region_bounce_coarse_stage_ranks),
        residual_region_bounce_coarse_stage_ranks=tuple(
            int(v) for v in residual_region_bounce_coarse_stage_ranks
        ),
        residual_region_bounce_coarse_max_rank_requested=int(
            config_use.residual_region_bounce_coarse_max_rank
        ),
        residual_region_bounce_coarse_solver=residual_region_bounce_coarse_solver,
        residual_region_bounce_coarse_condition_estimate=float(
            residual_region_bounce_coarse_condition_estimate
        ),
        residual_region_bounce_coarse_residual_before=float(
            residual_region_bounce_coarse_residual_before
        ),
        residual_region_bounce_coarse_residual_after=float(
            residual_region_bounce_coarse_residual_after
        ),
        residual_region_bounce_coarse_include_global=bool(
            config_use.residual_region_bounce_coarse_include_global
        ),
        residual_region_bounce_coarse_include_radial=bool(
            config_use.residual_region_bounce_coarse_include_radial
        ),
        residual_region_bounce_coarse_include_species=bool(
            config_use.residual_region_bounce_coarse_include_species
        ),
        residual_region_bounce_coarse_bounce_boundary=float(
            config_use.residual_region_bounce_coarse_trapped_boundary_fraction
        ),
        residual_region_bounce_coarse_min_region_energy_fraction=float(
            config_use.residual_region_bounce_coarse_min_region_energy_fraction
        ),
        residual_region_bounce_coarse_region_bands=str(
            config_use.residual_region_bounce_coarse_region_bands
        ),
        active_pattern_coarse_enabled=bool(
            config_use.active_pattern_coarse and active_pattern_coarse_rank > 0
        ),
        active_pattern_coarse_candidate_count=int(active_pattern_coarse_candidate_count),
        active_pattern_coarse_rank=int(active_pattern_coarse_rank),
        active_pattern_coarse_stage_count=len(active_pattern_coarse_stage_ranks),
        active_pattern_coarse_stage_ranks=tuple(int(v) for v in active_pattern_coarse_stage_ranks),
        active_pattern_coarse_max_rank_requested=int(config_use.active_pattern_coarse_max_rank),
        active_pattern_coarse_max_candidates_requested=int(
            config_use.active_pattern_coarse_max_candidates
        ),
        active_pattern_coarse_solver=active_pattern_coarse_solver,
        active_pattern_coarse_condition_estimate=float(active_pattern_coarse_condition_estimate),
        active_pattern_coarse_residual_before=float(active_pattern_coarse_residual_before),
        active_pattern_coarse_residual_after=float(active_pattern_coarse_residual_after),
        active_pattern_coarse_min_chunk_energy_fraction=float(
            config_use.active_pattern_coarse_min_chunk_energy_fraction
        ),
        active_pattern_coarse_include_global=bool(config_use.active_pattern_coarse_include_global),
        active_pattern_coarse_include_block_pitch=bool(
            config_use.active_pattern_coarse_include_block_pitch
        ),
        active_pattern_coarse_include_block_angular=bool(
            config_use.active_pattern_coarse_include_block_angular
        ),
        active_pattern_coarse_include_radial_pitch=bool(
            config_use.active_pattern_coarse_include_radial_pitch
        ),
        active_pattern_coarse_include_radial_angular=bool(
            config_use.active_pattern_coarse_include_radial_angular
        ),
        active_pattern_coarse_include_block=bool(config_use.active_pattern_coarse_include_block),
        active_pattern_coarse_include_radial=bool(config_use.active_pattern_coarse_include_radial),
        active_pattern_coarse_include_species=bool(
            config_use.active_pattern_coarse_include_species
        ),
        block_schur_residual_equation_enabled=bool(
            config_use.block_schur_residual_equation and block_schur_residual_equation_rank > 0
        ),
        block_schur_residual_equation_group_count=int(block_schur_residual_equation_group_count),
        block_schur_residual_equation_candidate_count=int(block_schur_residual_equation_candidate_count),
        block_schur_residual_equation_rank=int(block_schur_residual_equation_rank),
        block_schur_residual_equation_stage_ranks=tuple(
            int(v) for v in block_schur_residual_equation_stage_ranks
        ),
        block_schur_residual_equation_include_global=bool(
            config_use.block_schur_residual_equation_include_global
        ),
        block_schur_residual_equation_include_blocks=bool(
            config_use.block_schur_residual_equation_include_blocks
        ),
        block_schur_residual_equation_include_aggregates=bool(
            config_use.block_schur_residual_equation_include_aggregates
        ),
        residual_snapshot_enrichment_enabled=bool(
            config_use.residual_snapshot_enrichment and residual_snapshot_rank > 0
        ),
        residual_snapshot_candidate_count=int(residual_snapshot_candidate_count),
        residual_snapshot_rank=int(residual_snapshot_rank),
        residual_snapshot_group_count=int(residual_snapshot_group_count),
        residual_snapshot_include_primal=bool(config_use.residual_snapshot_include_primal),
        residual_snapshot_use_adjoint=bool(config_use.residual_snapshot_use_adjoint),
        residual_snapshot_residual_equation_enabled=bool(
            config_use.residual_snapshot_residual_equation
            and residual_snapshot_residual_equation_rank > 0
        ),
        residual_snapshot_residual_equation_group_count=int(
            residual_snapshot_residual_equation_group_count
        ),
        residual_snapshot_residual_equation_candidate_count=int(
            residual_snapshot_residual_equation_candidate_count
        ),
        residual_snapshot_residual_equation_rank=int(residual_snapshot_residual_equation_rank),
        residual_snapshot_residual_equation_stage_ranks=tuple(
            int(v) for v in residual_snapshot_residual_equation_stage_ranks
        ),
        residual_snapshot_residual_equation_solver=residual_snapshot_residual_equation_solver,
        residual_snapshot_residual_equation_include_global=bool(
            config_use.residual_snapshot_residual_equation_include_global
        ),
        block_schur_residual_enrichment_enabled=bool(
            config_use.block_schur_residual_enrichment and block_schur_residual_rank > 0
        ),
        block_schur_residual_candidate_count=int(block_schur_residual_candidate_count),
        block_schur_residual_rank=int(block_schur_residual_rank),
        block_schur_residual_group_count=int(block_schur_residual_group_count),
        reason=reason,
    )
    return RHS1QIDevicePreconditionerState(
        operator=operator_csr,
        operator_matvec=operator_matvec,
        dtype=dtype_use,
        shape=shape,
        local_smoother=smoother,
        basis=basis,
        operator_on_basis=aq,
        coarse_operator=coarse_operator,
        residual_equation_bases=residual_equation_bases,
        residual_equation_operator_on_bases=residual_equation_actions,
        residual_equation_coarse_operators=residual_equation_coarse_operators,
        residual_equation_stage_solvers=residual_equation_stage_solvers,
        metadata=metadata,
    )


def probe_rhs1_qi_device_preconditioner(
    *,
    rhs: ArrayLike,
    x0: ArrayLike,
    state: RHS1QIDevicePreconditionerState,
    operator: Callable[[ArrayLike], ArrayLike] | None = None,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    max_cycles: int = 1,
    residual_minimizing_step: bool = False,
    alpha_clip: float = 10.0,
) -> tuple[ArrayLike, RHS1QIDevicePreconditionerProbe]:
    """Apply bounded preconditioner corrections accepted only by true residual.

    ``max_cycles`` is intentionally small and fail-closed.  Each cycle applies
    the reusable device-QI action to the current true residual, accepts only a
    material residual drop, and stops as soon as a candidate is non-finite or no
    longer improves.  When ``residual_minimizing_step`` is enabled, each
    correction direction is scaled by the scalar that minimizes
    ``||r - alpha A d||_2`` before the true-residual gate is evaluated.  This
    gives the GPU hard-seed lane a real residual-reducing sequence without
    installing the coarse action as an unbounded Krylov preconditioner.
    """

    matvec = state.operator_matvec if operator is None else operator
    rhs_vec = jnp.asarray(rhs, dtype=state.dtype).reshape((-1,))
    x_initial = jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")
    residual_before = rhs_vec - jnp.asarray(matvec(x_initial), dtype=rhs_vec.dtype).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIDevicePreconditionerProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=state.metadata,
            cycles=0,
            residual_history=(0.0,),
            step_history=(),
        )
        return x_initial, probe

    x_best = x_initial
    residual_current = residual_before
    residual_current_norm = residual_before_norm
    history: list[float] = [float(residual_before_norm)]
    accepted_cycles = 0
    last_finite = True
    last_candidate_norm = residual_before_norm
    max_cycles_use = max(1, int(max_cycles))
    step_history: list[float] = []
    alpha_clip_use = max(0.0, float(alpha_clip))
    for _ in range(max_cycles_use):
        dx = jnp.asarray(state.apply(residual_current), dtype=rhs_vec.dtype).reshape((-1,))
        alpha = 1.0
        if bool(residual_minimizing_step):
            a_dx = jnp.asarray(matvec(dx), dtype=rhs_vec.dtype).reshape((-1,))
            denom = float(jnp.real(jnp.vdot(a_dx, a_dx)))
            if (not np.isfinite(denom)) or denom <= 1.0e-300:
                last_finite = False
                break
            numer = float(jnp.real(jnp.vdot(a_dx, residual_current)))
            alpha = numer / denom
            if alpha_clip_use > 0.0:
                alpha = max(-alpha_clip_use, min(alpha_clip_use, float(alpha)))
            if (not np.isfinite(alpha)) or alpha == 0.0:
                last_finite = False
                break
            x_candidate = x_best + float(alpha) * dx
            residual_after = residual_current - float(alpha) * a_dx
        else:
            x_candidate = x_best + dx
            residual_after = rhs_vec - jnp.asarray(matvec(x_candidate), dtype=rhs_vec.dtype).reshape((-1,))
        residual_after_norm_measured = float(jnp.linalg.norm(residual_after))
        finite = bool(np.isfinite(residual_after_norm_measured))
        last_finite = finite
        if finite:
            last_candidate_norm = residual_after_norm_measured
        required_drop = max(
            float(acceptance_atol),
            residual_current_norm * max(0.0, float(min_relative_improvement)),
        )
        if not (finite and residual_after_norm_measured < residual_current_norm - required_drop):
            break
        x_best = x_candidate
        residual_current = residual_after
        residual_current_norm = residual_after_norm_measured
        history.append(float(residual_current_norm))
        step_history.append(float(alpha))
        accepted_cycles += 1

    accepted = accepted_cycles > 0
    if accepted:
        reason = "residual_reduced"
        residual_after_norm = residual_current_norm
    elif not last_finite:
        reason = "nonfinite_candidate"
        residual_after_norm = residual_before_norm
    else:
        reason = "residual_not_reduced"
        residual_after_norm = last_candidate_norm
    improvement_ratio = residual_after_norm / residual_before_norm if last_finite else None
    probe = RHS1QIDevicePreconditionerProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=state.metadata,
        cycles=int(accepted_cycles),
        residual_history=tuple(float(value) for value in history),
        step_history=tuple(float(value) for value in step_history),
    )
    return x_best if accepted else x_initial, probe


def probe_rhs1_qi_device_augmented_seed(
    *,
    rhs: ArrayLike,
    x0: ArrayLike,
    state: RHS1QIDevicePreconditionerState,
    operator: Callable[[ArrayLike], ArrayLike] | None = None,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    max_cycles: int = 1,
    residual_minimizing_step: bool = False,
    alpha_clip: float = 10.0,
    max_rank: int | None = None,
    basis_rtol: float | None = None,
    basis_atol: float | None = None,
) -> RHS1QIDeviceAugmentedSeedProbe:
    """Probe device-QI corrections and retain accepted directions for Krylov.

    The regular probe accepts or rejects a bounded sequence of device-QI
    corrections, then discards the actual correction subspace.  This helper
    keeps the accepted update directions, rank-gates them, recomputes ``A Q``
    after orthonormalization, and returns a paired ``(Q, A Q)`` augmentation
    that can be passed to the FGMRES/GMRES augmented Krylov hooks.  If no true
    residual improvement is accepted, the augmentation is empty.
    """

    matvec = state.operator_matvec if operator is None else operator
    rhs_vec = jnp.asarray(rhs, dtype=state.dtype).reshape((-1,))
    x_initial = jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    empty_basis = jnp.zeros((int(state.shape[1]), 0), dtype=rhs_vec.dtype)
    empty_action = jnp.zeros((int(state.shape[0]), 0), dtype=rhs_vec.dtype)
    residual_before = rhs_vec - jnp.asarray(matvec(x_initial), dtype=rhs_vec.dtype).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIDevicePreconditionerProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=state.metadata,
            cycles=0,
            residual_history=(0.0,),
            step_history=(),
        )
        return RHS1QIDeviceAugmentedSeedProbe(
            solution=x_initial,
            probe=probe,
            augmentation_basis=empty_basis,
            operator_on_augmentation=empty_action,
            rank=0,
            reason="zero_residual",
            accepted_labels=(),
            projection_residual_norm=None,
        )

    x_best = x_initial
    residual_current = residual_before
    residual_current_norm = residual_before_norm
    history: list[float] = [float(residual_before_norm)]
    step_history: list[float] = []
    accepted_directions: list[ArrayLike] = []
    accepted_cycles = 0
    last_finite = True
    last_candidate_norm = residual_before_norm
    max_cycles_use = max(1, int(max_cycles))
    alpha_clip_use = max(0.0, float(alpha_clip))

    for _ in range(max_cycles_use):
        dx = jnp.asarray(state.apply(residual_current), dtype=rhs_vec.dtype).reshape((-1,))
        a_dx = jnp.asarray(matvec(dx), dtype=rhs_vec.dtype).reshape((-1,))
        alpha = 1.0
        if bool(residual_minimizing_step):
            denom = float(jnp.real(jnp.vdot(a_dx, a_dx)))
            if (not np.isfinite(denom)) or denom <= 1.0e-300:
                last_finite = False
                break
            numer = float(jnp.real(jnp.vdot(a_dx, residual_current)))
            alpha = numer / denom
            if alpha_clip_use > 0.0:
                alpha = max(-alpha_clip_use, min(alpha_clip_use, float(alpha)))
            if (not np.isfinite(alpha)) or alpha == 0.0:
                last_finite = False
                break
        direction = float(alpha) * dx
        action = float(alpha) * a_dx
        x_candidate = x_best + direction
        residual_after = residual_current - action
        residual_after_norm_measured = float(jnp.linalg.norm(residual_after))
        finite = bool(np.isfinite(residual_after_norm_measured))
        last_finite = finite
        if finite:
            last_candidate_norm = residual_after_norm_measured
        required_drop = max(
            float(acceptance_atol),
            residual_current_norm * max(0.0, float(min_relative_improvement)),
        )
        if not (finite and residual_after_norm_measured < residual_current_norm - required_drop):
            break
        accepted_directions.append(direction)
        x_best = x_candidate
        residual_current = residual_after
        residual_current_norm = residual_after_norm_measured
        history.append(float(residual_current_norm))
        step_history.append(float(alpha))
        accepted_cycles += 1

    projection_residual_norm: float | None = None
    augmentation_basis = empty_basis
    operator_on_augmentation = empty_action
    accepted_labels: tuple[str, ...] = ()
    rank = 0
    invalid_augmentation = False
    if accepted_directions:
        raw_basis = jnp.stack(tuple(accepted_directions), axis=1)
        labels = tuple(f"augmented_seed:{idx}" for idx in range(int(raw_basis.shape[1])))
        max_rank_use = int(raw_basis.shape[1]) if max_rank is None else max(0, int(max_rank))
        basis = orthonormalize_rhs1_qi_coarse_basis(
            raw_basis,
            labels=labels,
            rtol=float(state.metadata.regularization_rcond if basis_rtol is None else basis_rtol),
            atol=0.0 if basis_atol is None else float(basis_atol),
            max_rank=max_rank_use,
        )
        rank = int(basis.metadata.rank)
        if rank > 0:
            candidate_basis = jnp.asarray(basis.vectors, dtype=rhs_vec.dtype)
            candidate_action = _operator_on_basis(
                matvec,
                candidate_basis,
                shape=state.shape,
                dtype=rhs_vec.dtype,
            )
            valid_shape = (
                int(candidate_basis.ndim) == 2
                and int(candidate_action.ndim) == 2
                and int(candidate_basis.shape[0]) == int(state.shape[1])
                and int(candidate_action.shape[0]) == int(state.shape[0])
                and int(candidate_basis.shape[1]) == int(candidate_action.shape[1])
                and int(candidate_basis.shape[1]) == int(rank)
            )
            valid_values = bool(jnp.all(jnp.isfinite(candidate_basis))) and bool(
                jnp.all(jnp.isfinite(candidate_action))
            )
            if valid_shape and valid_values:
                augmentation_basis = candidate_basis
                operator_on_augmentation = candidate_action
                accepted_labels = tuple(str(label) for label in basis.metadata.accepted_labels)
                coefficients = _regularized_least_squares(
                    operator_on_augmentation,
                    residual_before,
                    rcond=float(state.metadata.regularization_rcond),
                )
                projected_update = augmentation_basis @ coefficients
                projected_residual = residual_before - operator_on_augmentation @ coefficients
                projected_norm = float(jnp.linalg.norm(projected_residual))
                projection_residual_norm = projected_norm
                if np.isfinite(projected_norm) and projected_norm < residual_current_norm:
                    x_best = x_initial + projected_update
                    residual_current_norm = projected_norm
                    history.append(float(projected_norm))
            else:
                invalid_augmentation = True
                rank = 0

    accepted = accepted_cycles > 0
    if accepted:
        if invalid_augmentation:
            reason = "residual_reduced_invalid_augmentation"
        else:
            reason = "augmented_residual_reduced" if rank > 0 else "residual_reduced"
        residual_after_norm = residual_current_norm
    elif not last_finite:
        reason = "nonfinite_candidate"
        residual_after_norm = residual_before_norm
    else:
        reason = "residual_not_reduced"
        residual_after_norm = last_candidate_norm
    improvement_ratio = residual_after_norm / residual_before_norm if last_finite else None
    probe = RHS1QIDevicePreconditionerProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm,
        improvement_ratio=improvement_ratio,
        metadata=state.metadata,
        cycles=int(accepted_cycles),
        residual_history=tuple(float(value) for value in history),
        step_history=tuple(float(value) for value in step_history),
    )
    return RHS1QIDeviceAugmentedSeedProbe(
        solution=x_best if accepted else x_initial,
        probe=probe,
        augmentation_basis=augmentation_basis,
        operator_on_augmentation=operator_on_augmentation,
        rank=int(rank),
        reason=reason,
        accepted_labels=accepted_labels,
        projection_residual_norm=projection_residual_norm,
    )


__all__ = [
    "RHS1QIMatrixFreeResidualSmoother",
    "RHS1QIMatrixFreeResidualSmootherMetadata",
    "RHS1QIMatrixFreeProjectedResidualSmoother",
    "RHS1QIMatrixFreeProjectedResidualSmootherMetadata",
    "RHS1QIDeviceAugmentedSeedProbe",
    "RHS1QIDevicePreconditionerConfig",
    "RHS1QIDevicePreconditionerMetadata",
    "RHS1QIDevicePreconditionerProbe",
    "RHS1QIDevicePreconditionerState",
    "probe_rhs1_qi_device_augmented_seed",
    "probe_rhs1_qi_device_preconditioner",
    "setup_rhs1_qi_device_preconditioner",
]
