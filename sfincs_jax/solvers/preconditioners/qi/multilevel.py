"""Multilevel angular-radial coarse correction for RHSMode=1 QI seeds.

This module is a standalone architecture prototype for hard RHSMode=1 QI
seeds.  It builds a small hierarchy of angular-radial prolongation spaces over
radial block aggregates, rank-gates the combined space, and applies a pure-JAX
coarse correction:

``M^{-1} r = S_local^{-1} r + Q c``,
``c = argmin ||A Q c - (r - A S_local^{-1} r)||``.

The builder is intentionally independent of ``v3_driver.py``.  It is suitable
for device use because the reusable apply path is composed of JAX array
operations and closed-over JAX callables only; no SciPy factors, host callbacks,
or Python-side solves are needed after construction.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import jax.numpy as jnp

from sfincs_jax.solvers.preconditioners.qi.coarse import (
    RHS1QICoarseBasis,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)

ArrayLike = Any
LinearOperator = Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseConfig:
    """Static controls for angular-radial multilevel coarse construction."""

    max_levels: int = 3
    aggregate_factor: int = 2
    max_rank: int = 48
    max_angular_mode: int = 1
    max_radial_degree: int = 2
    include_level_aggregates: bool = True
    include_angular: bool = True
    include_radial: bool = True
    include_radial_angular: bool = True
    include_pitch: bool = True
    include_radial_pitch: bool = True
    include_current_moments: bool = False
    include_species_current_moments: bool = True
    include_radial_current_moments: bool = True
    include_tail_constraint_moments: bool = True
    include_finest_blocks: bool = False
    rtol: float = 1.0e-10
    atol: float = 0.0
    regularization_rcond: float = 1.0e-12
    damping: float = 1.0
    max_pitch_degree: int = 0
    max_current_pitch_degree: int = 1
    nested_residual_correction: bool = False
    nested_level_max_rank: int = 16
    nested_order: str = "coarse_to_fine"
    nested_solver: str = "action_lstsq"
    nested_include_global: bool = True
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseLevelMetadata:
    """Diagnostics for one radial aggregation level in the hierarchy."""

    level_index: int
    aggregate_size: int
    aggregate_count: int
    block_groups: tuple[tuple[int, ...], ...]
    candidate_count: int
    rank: int
    discarded_count: int
    candidate_labels: tuple[str, ...]
    accepted_labels: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly per-level diagnostics."""

        return {
            "level_index": int(self.level_index),
            "aggregate_size": int(self.aggregate_size),
            "aggregate_count": int(self.aggregate_count),
            "block_groups": tuple(tuple(int(v) for v in group) for group in self.block_groups),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "discarded_count": int(self.discarded_count),
            "candidate_labels": self.candidate_labels,
            "accepted_labels": self.accepted_labels,
        }


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseMetadata:
    """Diagnostics for a reusable multilevel angular-radial coarse action."""

    total_size: int
    level_count: int
    levels: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]
    candidate_count: int
    rank: int
    nested_residual_correction_enabled: bool
    nested_level_count: int
    nested_rank: int
    nested_level_ranks: tuple[int, ...]
    nested_order: str
    nested_solver: str
    nested_include_global: bool
    nested_coarse_operator_shapes: tuple[tuple[int, int], ...]
    nested_coarse_operator_norms: tuple[float, ...]
    discarded_count: int
    operator_on_basis_shape: tuple[int, int]
    operator_on_basis_norm: float
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    regularization_rcond: float
    damping: float
    accepted_labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    device_resident: bool
    host_callback_free: bool
    reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for traces and integration probes."""

        payload = asdict(self)
        payload["levels"] = tuple(level.to_dict() for level in self.levels)
        payload["operator_on_basis_shape"] = tuple(int(v) for v in self.operator_on_basis_shape)
        payload["coarse_operator_shape"] = tuple(int(v) for v in self.coarse_operator_shape)
        payload["nested_coarse_operator_shapes"] = tuple(
            tuple(int(v) for v in shape) for shape in self.nested_coarse_operator_shapes
        )
        return payload


@dataclass(frozen=True)
class RHS1QIMultilevelCoarsePreconditioner:
    """Reusable pure-JAX local-plus-multilevel-coarse correction."""

    operator: LinearOperator
    local_smoother: LinearOperator
    basis: RHS1QICoarseBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    nested_bases: tuple[RHS1QICoarseBasis, ...]
    nested_operator_on_bases: tuple[ArrayLike, ...]
    nested_coarse_operators: tuple[ArrayLike, ...]
    metadata: RHS1QIMultilevelCoarseMetadata

    def solve_coarse(self, residual: ArrayLike) -> ArrayLike:
        """Return action least-squares coarse coefficients for ``residual``."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if rank <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        return _regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Apply one multiplicative local plus coarse correction."""

        residual_vec = jnp.asarray(residual).reshape((-1,))
        local = jnp.asarray(self.local_smoother(residual_vec)).reshape((-1,))
        if int(self.metadata.rank) <= 0 and int(self.metadata.nested_rank) <= 0:
            return float(self.metadata.damping) * local
        remaining = residual_vec - jnp.asarray(self.operator(local)).reshape((-1,))
        if bool(self.metadata.nested_residual_correction_enabled):
            coarse = self.solve_nested_residual_equation(remaining)
        else:
            coarse = self.basis.vectors @ self.solve_coarse(remaining)
        return float(self.metadata.damping) * (local + coarse)

    def solve_nested_residual_equation(self, residual: ArrayLike) -> ArrayLike:
        """Solve the coarse residual equation level by level.

        Each level receives a separate rank budget and acts on the residual left
        by previous levels. This is a bounded multilevel residual equation, not
        another smoother sweep: every stage minimizes ``||r - A Q_l c||`` over
        its own coarse action, updates by ``Q_l c``, and passes the remaining
        residual to the next coarser/finer level.
        """

        residual_vec = jnp.asarray(residual).reshape((-1,))
        correction = jnp.zeros_like(residual_vec)
        remaining = residual_vec
        solver = str(self.metadata.nested_solver)
        for basis, action, coarse_operator in zip(
            self.nested_bases,
            self.nested_operator_on_bases,
            self.nested_coarse_operators,
            strict=True,
        ):
            if int(basis.metadata.rank) <= 0:
                continue
            if solver == "galerkin":
                coefficients = _regularized_galerkin_solve(
                    coarse_operator,
                    jnp.conjugate(jnp.asarray(basis.vectors, dtype=residual_vec.dtype)).T @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = _regularized_action_least_squares(
                    action,
                    remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            level_update = jnp.asarray(basis.vectors, dtype=residual_vec.dtype) @ coefficients
            correction = correction + level_update
            remaining = remaining - jnp.asarray(action, dtype=residual_vec.dtype) @ coefficients
        if bool(self.metadata.nested_include_global) and int(self.metadata.rank) > 0:
            if solver == "galerkin":
                coefficients = _regularized_galerkin_solve(
                    self.coarse_operator,
                    jnp.conjugate(jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype)).T @ remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            else:
                coefficients = _regularized_action_least_squares(
                    self.operator_on_basis,
                    remaining,
                    rcond=float(self.metadata.regularization_rcond),
                )
            correction = correction + jnp.asarray(self.basis.vectors, dtype=residual_vec.dtype) @ coefficients
        return correction

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov preconditioner hooks."""

        return self.apply


@dataclass(frozen=True)
class RHS1QIMultilevelCoarseProbe:
    """Fail-closed true-residual probe for the multilevel coarse candidate."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    metadata: RHS1QIMultilevelCoarseMetadata

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly probe diagnostics."""

        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "metadata": self.metadata.to_dict(),
        }


def _empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _block_offsets(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    offsets = [0]
    for size in layout.block_sizes:
        offsets.append(offsets[-1] + int(size))
    return tuple(offsets)


def _radial_groups(n_blocks: int, aggregate_size: int) -> tuple[tuple[int, ...], ...]:
    size = max(1, int(aggregate_size))
    groups: list[tuple[int, ...]] = []
    for start in range(0, int(n_blocks), size):
        stop = min(int(n_blocks), start + size)
        groups.append(tuple(range(start, stop)))
    return tuple(groups)


def _hierarchy_groups(n_blocks: int, config: RHS1QIMultilevelCoarseConfig) -> tuple[tuple[int, tuple[tuple[int, ...], ...]], ...]:
    max_levels = max(1, int(config.max_levels))
    factor = max(2, int(config.aggregate_factor))
    levels: list[tuple[int, tuple[tuple[int, ...], ...]]] = []
    aggregate_size = 1
    previous_groups: tuple[tuple[int, ...], ...] | None = None
    for _ in range(max_levels):
        groups = _radial_groups(n_blocks, aggregate_size)
        if groups == previous_groups:
            break
        levels.append((aggregate_size, groups))
        if len(groups) <= 1:
            break
        previous_groups = groups
        aggregate_size *= factor
    return tuple(levels)


def _centered_unit_weights(values: Sequence[float]) -> tuple[float, ...] | None:
    raw = tuple(float(value) for value in values)
    if not raw:
        return None
    mean = sum(raw) / float(len(raw))
    centered = tuple(value - mean for value in raw)
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple(value / scale for value in centered)


def _centered_power_weights(values: Sequence[float], degree: int) -> tuple[float, ...] | None:
    linear = _centered_unit_weights(values)
    if linear is None:
        return None
    if int(degree) == 1:
        return linear
    return _centered_unit_weights(tuple(value ** int(degree) for value in linear))


def _angular_specs(
    layout: RHS1QICoarseBlockLayout,
    *,
    max_angular_mode: int,
    dtype: Any,
) -> tuple[tuple[str, ArrayLike], ...]:
    n_theta = int(layout.n_theta)
    n_zeta = int(layout.n_zeta)
    max_mode = max(0, int(max_angular_mode))
    if max_mode <= 0 or n_theta * n_zeta <= 1:
        return ()

    theta = jnp.arange(n_theta, dtype=dtype)
    zeta = jnp.arange(n_zeta, dtype=dtype)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    specs: list[tuple[str, ArrayLike]] = []
    for mode in range(1, max_mode + 1):
        if n_theta > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(n_theta)
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if n_zeta > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if n_theta > 1 and n_zeta > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(n_theta)
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(n_zeta)
            specs.append((f"mixed_cos_plus{mode}", jnp.cos(theta_phase + zeta_phase)))
            specs.append((f"mixed_sin_plus{mode}", jnp.sin(theta_phase + zeta_phase)))
            specs.append((f"mixed_cos_minus{mode}", jnp.cos(theta_phase - zeta_phase)))
            specs.append((f"mixed_sin_minus{mode}", jnp.sin(theta_phase - zeta_phase)))
    return tuple(specs)


def _group_weighted_constant(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any(float(weight) != 0.0 for weight in group_weights):
        return None

    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(float(weight))
    return values


def _block_weighted_constant(
    layout: RHS1QICoarseBlockLayout,
    block_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    """Return a block-weighted constant over arbitrary structural blocks."""

    if len(block_weights) != len(layout.block_sizes):
        raise ValueError("block_weights must match block count")
    if not any(float(weight) != 0.0 for weight in block_weights):
        return None

    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for block_index, weight in enumerate(block_weights):
        if float(weight) == 0.0:
            continue
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _group_weighted_angular(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    angular_values: ArrayLike,
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any(float(weight) != 0.0 for weight in group_weights):
        return None

    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 1:
        return None
    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            block_size = int(layout.block_sizes[int(block_index)])
            if block_size % n_angular != 0:
                return None
            repeats = block_size // n_angular
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(float(weight) * jnp.tile(angular, repeats))
    return values


def _pitch_weights(
    layout: RHS1QICoarseBlockLayout,
    *,
    degree: int,
) -> tuple[float, ...] | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    pitch_count: int | None = None
    for block_size in layout.block_sizes:
        if int(block_size) % n_angular != 0:
            return None
        block_pitch_count = int(block_size) // n_angular
        if block_pitch_count <= 1:
            return None
        if pitch_count is None:
            pitch_count = block_pitch_count
        elif pitch_count != block_pitch_count:
            return None
    if pitch_count is None:
        return None
    return _centered_power_weights(tuple(range(pitch_count)), int(degree))


def _pitch_weights_for_count(pitch_count: int, *, degree: int) -> tuple[float, ...] | None:
    if int(pitch_count) <= 1:
        return None
    return _centered_power_weights(tuple(range(int(pitch_count))), int(degree))


def _group_weighted_pitch(
    layout: RHS1QICoarseBlockLayout,
    groups: Sequence[Sequence[int]],
    pitch_weights: Sequence[float],
    group_weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    """Return a group-weighted pitch moment candidate.

    SFINCS-JAX stores each block as repeated angular planes over pitch/xi.  A
    pitch moment is therefore the low-order xi polynomial repeated over each
    theta-zeta plane.  This adds a real coarse-space direction for PAS/QI slow
    modes that radial/angular aggregate spaces cannot represent.
    """

    if len(groups) != len(group_weights):
        raise ValueError("group_weights must match groups")
    if not any(float(weight) != 0.0 for weight in group_weights):
        return None

    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    pitch = jnp.asarray(tuple(float(value) for value in pitch_weights), dtype=dtype).reshape((-1,))
    if int(pitch.shape[0]) <= 1:
        return None
    block_values = jnp.repeat(pitch, n_angular)
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    for group, weight in zip(groups, group_weights, strict=True):
        for block_index in group:
            block_size = int(layout.block_sizes[int(block_index)])
            if block_size != int(block_values.shape[0]):
                return None
            start = int(offsets[int(block_index)])
            stop = int(offsets[int(block_index) + 1])
            values = values.at[start:stop].set(float(weight) * block_values)
    return values


def _block_weighted_pitch_moment(
    layout: RHS1QICoarseBlockLayout,
    block_weights: Sequence[float],
    *,
    degree: int,
    dtype: Any,
) -> ArrayLike | None:
    """Return a variable-pitch-count current/flow moment candidate.

    The production active-DOF layout can have a different number of retained xi
    points at different x locations.  Generic multilevel pitch candidates require
    identical block sizes, but current and bootstrap-current slow modes need a
    low-order xi moment even when the x=0 block is truncated.  This helper builds
    the centered pitch polynomial independently inside each supported block.
    """

    if len(block_weights) != len(layout.block_sizes):
        raise ValueError("block_weights must match block count")
    if not any(float(weight) != 0.0 for weight in block_weights):
        return None

    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = _block_offsets(layout)
    any_supported = False
    for block_index, weight in enumerate(block_weights):
        if float(weight) == 0.0:
            continue
        block_size = int(layout.block_sizes[int(block_index)])
        if block_size % n_angular != 0:
            continue
        pitch_count = block_size // n_angular
        pitch_weights = _pitch_weights_for_count(pitch_count, degree=int(degree))
        if pitch_weights is None:
            continue
        block_values = jnp.repeat(jnp.asarray(pitch_weights, dtype=dtype), n_angular)
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(float(weight) * block_values)
        any_supported = True
    return values if any_supported else None


def _non_tail_block_indices(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    result: list[int] = []
    for block_index, (x_value, species_value) in enumerate(
        zip(layout.block_x or (), layout.block_species or (), strict=True)
    ):
        if int(x_value) >= 0 and int(species_value) >= 0:
            result.append(int(block_index))
    return tuple(result)


def _tail_block_indices(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    result: list[int] = []
    for block_index, (x_value, species_value) in enumerate(
        zip(layout.block_x or (), layout.block_species or (), strict=True)
    ):
        if int(x_value) < 0 or int(species_value) < 0:
            result.append(int(block_index))
    return tuple(result)


def _weights_for_blocks(layout: RHS1QICoarseBlockLayout, weighted_blocks: Sequence[tuple[int, float]]) -> tuple[float, ...]:
    weights = [0.0 for _ in layout.block_sizes]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _species_block_groups(layout: RHS1QICoarseBlockLayout, block_indices: Sequence[int]) -> dict[int, tuple[int, ...]]:
    groups: dict[int, list[int]] = {}
    for block_index in block_indices:
        species = int((layout.block_species or ())[int(block_index)])
        groups.setdefault(species, []).append(int(block_index))
    return {species: tuple(indices) for species, indices in groups.items()}


def _centered_weights_for_block_indices(
    layout: RHS1QICoarseBlockLayout,
    block_indices: Sequence[int],
) -> tuple[tuple[int, float], ...] | None:
    if len(block_indices) <= 1:
        return None
    x_values = tuple(float((layout.block_x or ())[int(index)]) for index in block_indices)
    weights = _centered_unit_weights(x_values)
    if weights is None:
        return None
    return tuple((int(block_index), float(weight)) for block_index, weight in zip(block_indices, weights, strict=True))


def _build_current_constraint_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build high-priority current and constraint/nullspace candidates.

    These columns target the slow RHSMode=1 channels that are easy to truncate
    when generic angular/radial candidates fill the coarse rank budget first:
    global/species bootstrap-current moments, radial variation of those current
    moments, and the reduced tail/source block.  They are structural and
    deterministic, so they remain compatible with device-side operator reuse.
    """

    dtype = config.dtype
    columns: list[ArrayLike] = []
    labels: list[str] = []
    f_blocks = _non_tail_block_indices(layout)
    species_groups = _species_block_groups(layout, f_blocks)
    max_degree = max(0, int(config.max_current_pitch_degree))

    if bool(config.include_current_moments) and f_blocks and max_degree > 0:
        all_f_weights = _weights_for_blocks(layout, tuple((block_index, 1.0) for block_index in f_blocks))
        radial_weights = _centered_weights_for_block_indices(layout, f_blocks)
        for degree in range(1, max_degree + 1):
            _append_candidate(
                columns,
                labels,
                _block_weighted_pitch_moment(layout, all_f_weights, degree=degree, dtype=dtype),
                f"current:global:p{degree}",
            )
            if bool(config.include_radial_current_moments) and radial_weights is not None:
                _append_candidate(
                    columns,
                    labels,
                    _block_weighted_pitch_moment(
                        layout,
                        _weights_for_blocks(layout, radial_weights),
                        degree=degree,
                        dtype=dtype,
                    ),
                    f"current:radial:p1:pitch:p{degree}",
                )
            if bool(config.include_species_current_moments):
                for species in sorted(species_groups):
                    species_blocks = species_groups[species]
                    species_weights = _weights_for_blocks(
                        layout,
                        tuple((block_index, 1.0) for block_index in species_blocks),
                    )
                    _append_candidate(
                        columns,
                        labels,
                        _block_weighted_pitch_moment(layout, species_weights, degree=degree, dtype=dtype),
                        f"current:species:{species}:p{degree}",
                    )
                    if bool(config.include_radial_current_moments):
                        species_radial_weights = _centered_weights_for_block_indices(layout, species_blocks)
                        if species_radial_weights is not None:
                            _append_candidate(
                                columns,
                                labels,
                                _block_weighted_pitch_moment(
                                    layout,
                                    _weights_for_blocks(layout, species_radial_weights),
                                    degree=degree,
                                    dtype=dtype,
                                ),
                                f"current:species:{species}:radial:p1:pitch:p{degree}",
                            )

    if bool(config.include_tail_constraint_moments):
        tail_blocks = _tail_block_indices(layout)
        if tail_blocks:
            tail_weights = _weights_for_blocks(layout, tuple((block_index, 1.0) for block_index in tail_blocks))
            _append_candidate(
                columns,
                labels,
                _block_weighted_constant(layout, tail_weights, dtype),
                "constraint_tail:aggregate",
            )

    if not columns:
        return _empty_matrix(layout.total_size, dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def _group_centers(layout: RHS1QICoarseBlockLayout, groups: Sequence[Sequence[int]]) -> tuple[float, ...]:
    block_x = tuple(float(value) for value in (layout.block_x or tuple(range(len(layout.block_sizes)))))
    centers: list[float] = []
    for group in groups:
        group_values = tuple(block_x[int(block_index)] for block_index in group)
        centers.append(sum(group_values) / float(len(group_values)))
    return tuple(centers)


def _append_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
) -> None:
    if values is not None:
        columns.append(jnp.asarray(values).reshape((-1,)))
        labels.append(str(label))


def _is_finest_raw_aggregate_label(label: str) -> bool:
    prefix = "level:0:aggregate:"
    if not label.startswith(prefix):
        return False
    return ":" not in label[len(prefix) :]


def _build_level_candidates(
    layout: RHS1QICoarseBlockLayout,
    groups: tuple[tuple[int, ...], ...],
    *,
    level_index: int,
    config: RHS1QIMultilevelCoarseConfig,
) -> tuple[ArrayLike, tuple[str, ...]]:
    dtype = config.dtype
    columns: list[ArrayLike] = []
    labels: list[str] = []
    prefix = f"level:{int(level_index)}"
    angular_specs = _angular_specs(layout, max_angular_mode=config.max_angular_mode, dtype=dtype)
    pitch_specs: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(config.max_pitch_degree)) + 1):
        weights = _pitch_weights(layout, degree=degree)
        if weights is not None:
            pitch_specs.append((f"pitch:p{degree}", weights))

    if config.include_level_aggregates:
        for group_index, group in enumerate(groups):
            weights = tuple(1.0 if index == group_index else 0.0 for index in range(len(groups)))
            label = f"{prefix}:aggregate:{group_index}"
            _append_candidate(columns, labels, _group_weighted_constant(layout, groups, weights, dtype), label)
            if config.include_angular:
                for angular_label, angular_values in angular_specs:
                    _append_candidate(
                        columns,
                        labels,
                        _group_weighted_angular(layout, groups, angular_values, weights, dtype),
                        f"{label}:angular:{angular_label}",
                    )
            if config.include_pitch:
                for pitch_label, pitch_weights in pitch_specs:
                    _append_candidate(
                        columns,
                        labels,
                        _group_weighted_pitch(layout, groups, pitch_weights, weights, dtype),
                        f"{label}:{pitch_label}",
                    )

    centers = _group_centers(layout, groups)
    radial_weights: list[tuple[str, tuple[float, ...]]] = []
    for degree in range(1, max(0, int(config.max_radial_degree)) + 1):
        weights = _centered_power_weights(centers, degree)
        if weights is not None:
            radial_weights.append((f"radial:p{degree}", weights))

    if config.include_radial:
        for radial_label, weights in radial_weights:
            _append_candidate(
                columns,
                labels,
                _group_weighted_constant(layout, groups, weights, dtype),
                f"{prefix}:{radial_label}",
            )

    if config.include_radial_angular:
        for radial_label, weights in radial_weights:
            for angular_label, angular_values in angular_specs:
                _append_candidate(
                    columns,
                    labels,
                    _group_weighted_angular(layout, groups, angular_values, weights, dtype),
                    f"{prefix}:{radial_label}:angular:{angular_label}",
                )

    if config.include_radial_pitch:
        for radial_label, weights in radial_weights:
            for pitch_label, pitch_values in pitch_specs:
                _append_candidate(
                    columns,
                    labels,
                    _group_weighted_pitch(layout, groups, pitch_values, weights, dtype),
                    f"{prefix}:{radial_label}:{pitch_label}",
                )

    if not columns:
        return _empty_matrix(layout.total_size, dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def _level_metadata(
    candidates: ArrayLike,
    labels: tuple[str, ...],
    groups: tuple[tuple[int, ...], ...],
    *,
    level_index: int,
    aggregate_size: int,
    config: RHS1QIMultilevelCoarseConfig,
) -> RHS1QIMultilevelCoarseLevelMetadata:
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(config.rtol),
        atol=float(config.atol),
        max_rank=max(0, int(config.max_rank)),
    )
    return RHS1QIMultilevelCoarseLevelMetadata(
        level_index=int(level_index),
        aggregate_size=int(aggregate_size),
        aggregate_count=len(groups),
        block_groups=groups,
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        discarded_count=int(basis.metadata.discarded_count),
        candidate_labels=tuple(str(label) for label in labels),
        accepted_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
    )


def build_rhs1_qi_multilevel_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[ArrayLike, tuple[str, ...], tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]]:
    """Build deterministic multilevel angular-radial coarse candidates.

    Level 0 works on individual radial blocks.  Coarser levels aggregate
    contiguous radial blocks by ``aggregate_factor`` and add aggregate constants,
    angular harmonics, radial moments, and radial-angular tensor-product modes.
    The final basis builder rank-gates the union, so redundant level content is
    safe and useful for deterministic diagnostics.
    """

    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    columns: list[ArrayLike] = []
    labels: list[str] = []
    levels: list[RHS1QIMultilevelCoarseLevelMetadata] = []
    hierarchy = _hierarchy_groups(len(layout.block_sizes), cfg)

    structural_candidates, structural_labels = _build_current_constraint_candidates(layout, config=cfg)
    for column_index, label in enumerate(structural_labels):
        columns.append(structural_candidates[:, column_index])
        labels.append(label)

    for level_index, (aggregate_size, groups) in enumerate(hierarchy):
        level_candidates, level_labels = _build_level_candidates(
            layout,
            groups,
            level_index=level_index,
            config=cfg,
        )
        levels.append(
            _level_metadata(
                level_candidates,
                level_labels,
                groups,
                level_index=level_index,
                aggregate_size=aggregate_size,
                config=cfg,
            )
        )
        for column_index, label in enumerate(level_labels):
            if cfg.include_finest_blocks or not _is_finest_raw_aggregate_label(label):
                columns.append(level_candidates[:, column_index])
                labels.append(label)

    if not columns:
        return _empty_matrix(layout.total_size, cfg.dtype), (), tuple(levels)
    return jnp.stack(tuple(columns), axis=1), tuple(labels), tuple(levels)


def build_rhs1_qi_multilevel_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[RHS1QICoarseBasis, tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]]:
    """Build and rank-gate the multilevel angular-radial coarse basis."""

    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    candidates, labels, levels = build_rhs1_qi_multilevel_coarse_candidates(layout, config=cfg)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )
    return basis, levels


def _normalize_nested_order(value: str) -> str:
    order = str(value).strip().lower().replace("-", "_")
    if order in {"coarse_to_fine", "coarse", "coarse_first"}:
        return "coarse_to_fine"
    if order in {"fine_to_coarse", "fine", "fine_first"}:
        return "fine_to_coarse"
    raise ValueError("nested_order must be 'coarse_to_fine' or 'fine_to_coarse'")


def _normalize_nested_solver(value: str) -> str:
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
        raise ValueError("nested_solver must be 'action_lstsq' or 'galerkin'")
    return aliases[solver]


def build_rhs1_qi_multilevel_residual_level_bases(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> tuple[tuple[RHS1QICoarseBasis, ...], tuple[RHS1QIMultilevelCoarseLevelMetadata, ...]]:
    """Build per-level bases for the nested coarse residual equation.

    Unlike the global multilevel basis, these bases are rank-gated separately by
    level. This preserves coarse-grid residual directions that can be discarded
    by a single flat rank budget, while keeping every level bounded and
    deterministic.
    """

    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    order = _normalize_nested_order(cfg.nested_order)
    hierarchy = tuple(enumerate(_hierarchy_groups(len(layout.block_sizes), cfg)))
    if order == "coarse_to_fine":
        hierarchy = tuple(reversed(hierarchy))
    level_bases: list[RHS1QICoarseBasis] = []
    level_metadata: list[RHS1QIMultilevelCoarseLevelMetadata] = []
    for level_index, (aggregate_size, groups) in hierarchy:
        candidates, labels = _build_level_candidates(
            layout,
            groups,
            level_index=int(level_index),
            config=cfg,
        )
        max_rank = max(0, int(cfg.nested_level_max_rank))
        basis = orthonormalize_rhs1_qi_coarse_basis(
            candidates,
            labels=labels,
            rtol=float(cfg.rtol),
            atol=float(cfg.atol),
            max_rank=max_rank,
        )
        level_metadata.append(
            RHS1QIMultilevelCoarseLevelMetadata(
                level_index=int(level_index),
                aggregate_size=int(aggregate_size),
                aggregate_count=len(groups),
                block_groups=groups,
                candidate_count=int(basis.metadata.candidate_count),
                rank=int(basis.metadata.rank),
                discarded_count=int(basis.metadata.discarded_count),
                candidate_labels=tuple(str(label) for label in labels),
                accepted_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
            )
        )
        if int(basis.metadata.rank) > 0:
            level_bases.append(basis)
    return tuple(level_bases), tuple(level_metadata)


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    if int(q.shape[1]) == 0:
        return _empty_matrix(int(q.shape[0]), q.dtype)
    return jnp.stack(tuple(jnp.asarray(operator(q[:, index])).reshape((-1,)) for index in range(int(q.shape[1]))), axis=1)


def _regularized_action_least_squares(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    gram = jnp.conjugate(a).T @ a
    normal_rhs = jnp.conjugate(a).T @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _regularized_galerkin_solve(matrix: ArrayLike, rhs: ArrayLike, *, rcond: float) -> ArrayLike:
    """Solve a square projected residual equation with a scale-relative ridge."""

    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("Galerkin matrix must be 2D")
    if int(a.shape[0]) != int(a.shape[1]):
        raise ValueError("Galerkin matrix must be square")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match Galerkin matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def build_rhs1_qi_multilevel_coarse_preconditioner(
    *,
    operator: LinearOperator,
    layout: RHS1QICoarseBlockLayout | None = None,
    basis: RHS1QICoarseBasis | None = None,
    level_metadata: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...] = (),
    local_smoother: LinearOperator | None = None,
    config: RHS1QIMultilevelCoarseConfig | None = None,
) -> RHS1QIMultilevelCoarsePreconditioner:
    """Build a reusable local-plus-multilevel coarse preconditioner."""

    cfg = RHS1QIMultilevelCoarseConfig() if config is None else config
    nested_solver = _normalize_nested_solver(cfg.nested_solver)
    if basis is None:
        if layout is None:
            raise ValueError("either basis or layout must be provided")
        basis, level_metadata = build_rhs1_qi_multilevel_coarse_basis(layout, config=cfg)

    q = jnp.asarray(basis.vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    smoother = (lambda residual: jnp.zeros_like(jnp.asarray(residual).reshape((-1,)))) if local_smoother is None else local_smoother
    rank = int(q.shape[1])
    if rank <= 0:
        operator_on_basis = _empty_matrix(int(q.shape[0]), q.dtype)
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        operator_on_basis = _apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ operator_on_basis

    nested_bases: tuple[RHS1QICoarseBasis, ...] = ()
    nested_operator_on_bases: tuple[ArrayLike, ...] = ()
    nested_coarse_operators: tuple[ArrayLike, ...] = ()
    nested_level_metadata: tuple[RHS1QIMultilevelCoarseLevelMetadata, ...] = ()
    if bool(cfg.nested_residual_correction) and layout is not None:
        nested_bases, nested_level_metadata = build_rhs1_qi_multilevel_residual_level_bases(
            layout,
            config=cfg,
        )
        nested_actions: list[ArrayLike] = []
        nested_projected_operators: list[ArrayLike] = []
        for nested_basis in nested_bases:
            nested_q = jnp.asarray(nested_basis.vectors)
            nested_action = _apply_operator_to_basis(operator, nested_q)
            nested_actions.append(nested_action)
            nested_projected_operators.append(jnp.conjugate(nested_q).T @ nested_action)
        nested_operator_on_bases = tuple(nested_actions)
        nested_coarse_operators = tuple(nested_projected_operators)
    nested_level_ranks = tuple(int(nested_basis.metadata.rank) for nested_basis in nested_bases)
    nested_rank = sum(nested_level_ranks)
    nested_enabled = bool(cfg.nested_residual_correction) and nested_rank > 0
    nested_coarse_operator_shapes = tuple(
        tuple(int(v) for v in nested_coarse_operator.shape)
        for nested_coarse_operator in nested_coarse_operators
    )
    nested_coarse_operator_norms = tuple(
        float(jnp.linalg.norm(nested_coarse_operator))
        for nested_coarse_operator in nested_coarse_operators
    )

    if nested_enabled and nested_solver == "galerkin":
        reason = "built_with_nested_galerkin_residual_equation"
    elif nested_enabled:
        reason = "built_with_nested_residual_equation"
    elif rank > 0:
        reason = "built_with_multilevel_coarse"
    else:
        reason = "empty_basis"
    metadata = RHS1QIMultilevelCoarseMetadata(
        total_size=int(q.shape[0]),
        level_count=len(level_metadata),
        levels=tuple(level_metadata),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        nested_residual_correction_enabled=nested_enabled,
        nested_level_count=len(nested_bases),
        nested_rank=int(nested_rank),
        nested_level_ranks=nested_level_ranks,
        nested_order=_normalize_nested_order(cfg.nested_order),
        nested_solver=nested_solver,
        nested_include_global=bool(cfg.nested_include_global),
        nested_coarse_operator_shapes=nested_coarse_operator_shapes,
        nested_coarse_operator_norms=nested_coarse_operator_norms,
        discarded_count=int(basis.metadata.discarded_count),
        operator_on_basis_shape=tuple(int(v) for v in operator_on_basis.shape),
        operator_on_basis_norm=float(jnp.linalg.norm(operator_on_basis)) if rank > 0 else 0.0,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        coarse_operator_norm=float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0,
        regularization_rcond=float(cfg.regularization_rcond),
        damping=float(cfg.damping),
        accepted_labels=tuple(str(label) for label in basis.metadata.accepted_labels),
        candidate_labels=tuple(str(label) for label in basis.metadata.candidate_labels),
        device_resident=True,
        host_callback_free=True,
        reason=reason,
    )
    return RHS1QIMultilevelCoarsePreconditioner(
        operator=operator,
        local_smoother=smoother,
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        nested_bases=nested_bases,
        nested_operator_on_bases=nested_operator_on_bases,
        nested_coarse_operators=nested_coarse_operators,
        metadata=metadata,
    )


def probe_rhs1_qi_multilevel_coarse_correction(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike,
    preconditioner: RHS1QIMultilevelCoarsePreconditioner,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> tuple[ArrayLike, RHS1QIMultilevelCoarseProbe]:
    """Apply one correction and accept it only if true residual decreases."""

    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = jnp.asarray(x0).reshape((-1,))
    if rhs_vec.shape != x_initial.shape:
        raise ValueError("rhs and x0 must have the same shape")

    residual_before = rhs_vec - jnp.asarray(operator(x_initial)).reshape((-1,))
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    if residual_before_norm == 0.0:
        probe = RHS1QIMultilevelCoarseProbe(
            accepted=False,
            reason="zero_residual",
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=None,
            metadata=preconditioner.metadata,
        )
        return x_initial, probe
    if int(preconditioner.metadata.rank) <= 0 and int(preconditioner.metadata.nested_rank) <= 0:
        probe = RHS1QIMultilevelCoarseProbe(
            accepted=False,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            metadata=preconditioner.metadata,
        )
        return x_initial, probe

    dx = jnp.asarray(preconditioner.apply(residual_before)).reshape((-1,))
    candidate = x_initial + dx
    residual_after = rhs_vec - jnp.asarray(operator(candidate)).reshape((-1,))
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(float(acceptance_atol), residual_before_norm * max(0.0, float(min_relative_improvement)))
    finite = bool(jnp.isfinite(jnp.asarray(residual_after_norm)))
    accepted = finite and residual_after_norm < residual_before_norm - required_drop
    if accepted:
        reason = "residual_reduced"
    elif not finite:
        reason = "nonfinite_candidate"
    else:
        reason = "not_reduced"
    probe = RHS1QIMultilevelCoarseProbe(
        accepted=bool(accepted),
        reason=reason,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm if finite else residual_before_norm,
        improvement_ratio=float(improvement_ratio) if finite else None,
        metadata=preconditioner.metadata,
    )
    return candidate if accepted else x_initial, probe


__all__ = [
    "RHS1QIMultilevelCoarseConfig",
    "RHS1QIMultilevelCoarseLevelMetadata",
    "RHS1QIMultilevelCoarseMetadata",
    "RHS1QIMultilevelCoarsePreconditioner",
    "RHS1QIMultilevelCoarseProbe",
    "build_rhs1_qi_multilevel_coarse_basis",
    "build_rhs1_qi_multilevel_coarse_candidates",
    "build_rhs1_qi_multilevel_coarse_preconditioner",
    "build_rhs1_qi_multilevel_residual_level_bases",
    "probe_rhs1_qi_multilevel_coarse_correction",
]
