"""QI basis, coarse-space, and residual-region utilities."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any
import jax.numpy as jnp
import numpy as np
from sfincs_jax.operators.profile_system import (
    _ix_min,
    _source_basis_constraint_scheme_1,
)


ArrayLike = Any
LinearOperator = ArrayLike | Callable[[ArrayLike], ArrayLike]


@dataclass(frozen=True)
class RHS1QICoarseBlockLayout:
    """Block layout used to construct deterministic QI coarse candidates.

    ``block_sizes`` partitions the reduced RHSMode=1 vector.  ``block_x`` and
    ``block_species`` are optional structural labels used for x-ramp and
    species-average candidates; when absent the builder treats all blocks as one
    species and orders blocks monotonically in x.
    """

    block_sizes: Sequence[int]
    n_theta: int = 1
    n_zeta: int = 1
    block_x: Sequence[int] | None = None
    block_species: Sequence[int] | None = None

    def __post_init__(self) -> None:
        block_sizes = tuple((int(size) for size in self.block_sizes))
        if not block_sizes:
            raise ValueError("block_sizes must contain at least one block")
        if any((size <= 0 for size in block_sizes)):
            raise ValueError("block_sizes must be positive")
        object.__setattr__(self, "block_sizes", block_sizes)
        object.__setattr__(self, "n_theta", max(1, int(self.n_theta)))
        object.__setattr__(self, "n_zeta", max(1, int(self.n_zeta)))
        n_blocks = len(block_sizes)
        if self.block_x is None:
            block_x = tuple(range(n_blocks))
        else:
            block_x = tuple((int(value) for value in self.block_x))
        if len(block_x) != n_blocks:
            raise ValueError("block_x must have the same length as block_sizes")
        object.__setattr__(self, "block_x", block_x)
        if self.block_species is None:
            block_species = tuple((0 for _ in range(n_blocks)))
        else:
            block_species = tuple((int(value) for value in self.block_species))
        if len(block_species) != n_blocks:
            raise ValueError("block_species must have the same length as block_sizes")
        object.__setattr__(self, "block_species", block_species)

    @property
    def total_size(self) -> int:
        """Return the full vector length represented by this layout."""
        return int(sum(self.block_sizes))

    @property
    def block_offsets(self) -> tuple[int, ...]:
        """Return block starts including the final sentinel offset."""
        offsets = [0]
        for size in self.block_sizes:
            offsets.append(offsets[-1] + int(size))
        return tuple(offsets)


@dataclass(frozen=True)
class RHS1QICoarseBasisMetadata:
    """Diagnostics for coarse-basis construction and rank gating."""

    total_size: int
    candidate_count: int
    rank: int
    discarded_count: int
    candidate_labels: tuple[str, ...]
    accepted_labels: tuple[str, ...]
    candidate_norms: tuple[float, ...]
    accepted_norms: tuple[float, ...]
    rank_rtol: float
    rank_atol: float


@dataclass(frozen=True)
class RHS1QICoarseBasis:
    """Rank-gated orthonormal coarse basis."""

    vectors: ArrayLike
    metadata: RHS1QICoarseBasisMetadata


@dataclass(frozen=True)
class RHS1QICoarseCorrection:
    """Result from applying a guarded coarse least-squares correction."""

    solution: ArrayLike
    correction: ArrayLike
    coefficients: ArrayLike
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float
    applied: bool
    reason: str
    basis_metadata: RHS1QICoarseBasisMetadata


@dataclass(frozen=True)
class RHS1QIGalerkinPreconditionerMetadata:
    """Diagnostics for a reusable QI Galerkin coarse preconditioner."""

    rank: int
    coarse_operator_shape: tuple[int, int]
    coarse_operator_norm: float
    regularization_rcond: float
    basis_metadata: RHS1QICoarseBasisMetadata


@dataclass(frozen=True)
class RHS1QIGalerkinPreconditioner:
    """Reusable JAX-compatible Galerkin coarse preconditioner.

    The basis ``Q`` is orthonormal and the stored coarse operator is
    ``Q.T @ A @ Q``.  Applying the preconditioner solves the small projected
    problem in regularized least-squares form and lifts the result back with
    ``Q``.  This is intentionally device-compatible and avoids host callbacks
    or SciPy solvers, so it can be closed over by JIT-compiled Krylov code.
    """

    basis: RHS1QICoarseBasis
    coarse_operator: ArrayLike
    metadata: RHS1QIGalerkinPreconditionerMetadata

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        residual_vec = jnp.asarray(residual).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros((0,), dtype=residual_vec.dtype)
        projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
        return _coarse_small_regularized_least_squares(
            self.coarse_operator,
            projected,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        residual_vec = jnp.asarray(residual).reshape((-1,))
        if int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        return self.basis.vectors @ self.solve_coefficients(residual_vec)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        return self.apply


def _coarse_empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _coarse_append_candidate(
    columns: list[ArrayLike], labels: list[str], values: ArrayLike, label: str
) -> None:
    columns.append(jnp.asarray(values).reshape((-1,)))
    labels.append(str(label))


def _coarse_append_candidate_if_room(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
    max_candidates: int,
) -> bool:
    if values is None or len(columns) >= int(max_candidates):
        return False
    _coarse_append_candidate(columns, labels, values, label)
    return True


def _coarse_stack_candidates(
    total_size: int, dtype: Any, columns: Sequence[ArrayLike], labels: Sequence[str]
) -> tuple[ArrayLike, tuple[str, ...]]:
    if not columns:
        return (_coarse_empty_matrix(total_size, dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple((str(label) for label in labels)))


def _coarse_block_constant(
    layout: RHS1QICoarseBlockLayout, block_index: int, dtype: Any
) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    start = int(offsets[block_index])
    stop = int(offsets[block_index + 1])
    return values.at[start:stop].set(1.0)


def _coarse_group_constant(
    layout: RHS1QICoarseBlockLayout, block_ids: Sequence[int], dtype: Any
) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index in block_ids:
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(1.0)
    return values


def _coarse_block_weighted_constant(
    layout: RHS1QICoarseBlockLayout, weights: Sequence[float], dtype: Any
) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _coarse_species_to_blocks(layout: RHS1QICoarseBlockLayout) -> dict[int, list[int]]:
    species_to_blocks: dict[int, list[int]] = {}
    for block_index, species in enumerate(layout.block_species or ()):
        species_to_blocks.setdefault(int(species), []).append(block_index)
    return species_to_blocks


def _coarse_centered_unit_weights(values: Sequence[float]) -> tuple[float, ...] | None:
    weights = tuple((float(value) for value in values))
    if not weights:
        return None
    mean = sum(weights) / float(len(weights))
    centered = tuple((value - mean for value in weights))
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple((value / scale for value in centered))


def _coarse_centered_power_weights(
    values: Sequence[float], power: int
) -> tuple[float, ...] | None:
    linear = _coarse_centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple((value ** int(power) for value in linear))
    return _coarse_centered_unit_weights(powered)


def _coarse_weights_for_blocks(
    n_blocks: int, weighted_blocks: Sequence[tuple[int, float]]
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _coarse_angular_candidate(
    layout: RHS1QICoarseBlockLayout, angular_values: ArrayLike, dtype: Any
) -> ArrayLike | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 1:
        return None
    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index, block_size in enumerate(layout.block_sizes):
        if int(block_size) % n_angular != 0:
            return None
        repeats = int(block_size) // n_angular
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(jnp.tile(angular, repeats))
    return values


def _coarse_block_weighted_angular_candidate(
    layout: RHS1QICoarseBlockLayout,
    angular_values: ArrayLike,
    weights: Sequence[float],
    dtype: Any,
) -> ArrayLike | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 1:
        return None
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match the number of blocks")
    angular = jnp.asarray(angular_values, dtype=dtype).reshape((n_angular,))
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    any_nonzero = False
    for block_index, block_size in enumerate(layout.block_sizes):
        if int(block_size) % n_angular != 0:
            return None
        weight = float(weights[block_index])
        any_nonzero = any_nonzero or weight != 0.0
        repeats = int(block_size) // n_angular
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(weight * jnp.tile(angular, repeats))
    if not any_nonzero:
        return None
    return values


def _coarse_centered_index_moment(
    repeats: int, power: int, dtype: Any
) -> ArrayLike | None:
    repeats = int(repeats)
    if repeats <= 1:
        return None
    denominator = float(max(1, repeats - 1))
    base = tuple((2.0 * float(index) / denominator - 1.0 for index in range(repeats)))
    if int(power) == 1:
        values = base
    else:
        centered = _coarse_centered_unit_weights(
            tuple((value ** int(power) for value in base))
        )
        if centered is None:
            return None
        values = centered
    return jnp.asarray(values, dtype=dtype)


def _coarse_intra_block_moment_candidate(
    layout: RHS1QICoarseBlockLayout, weights: Sequence[float], *, power: int, dtype: Any
) -> ArrayLike | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return None
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match the number of blocks")
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    any_supported = False
    any_nonzero = False
    for block_index, block_size in enumerate(layout.block_sizes):
        if int(block_size) % n_angular != 0:
            return None
        weight = float(weights[block_index])
        if weight == 0.0:
            continue
        repeats = int(block_size) // n_angular
        moment = _coarse_centered_index_moment(repeats, int(power), dtype)
        if moment is None:
            continue
        any_supported = True
        any_nonzero = True
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(weight * jnp.repeat(moment, n_angular))
    if not any_supported or not any_nonzero:
        return None
    return values


def _coarse_angular_harmonic_specs(
    layout: RHS1QICoarseBlockLayout,
    *,
    max_angular_mode: int,
    include_mixed: bool,
    dtype: Any,
) -> tuple[tuple[str, ArrayLike], ...]:
    max_mode = max(0, int(max_angular_mode))
    if max_mode <= 0:
        return ()
    theta = jnp.arange(int(layout.n_theta), dtype=dtype)
    zeta = jnp.arange(int(layout.n_zeta), dtype=dtype)
    theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
    specs: list[tuple[str, ArrayLike]] = []
    for mode in range(1, max_mode + 1):
        if int(layout.n_theta) > 1:
            theta_phase = (
                2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            )
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if int(layout.n_zeta) > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if bool(include_mixed) and int(layout.n_theta) > 1 and (int(layout.n_zeta) > 1):
            theta_phase = (
                2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            )
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"mixed_cos_plus{mode}", jnp.cos(theta_phase + zeta_phase)))
            specs.append((f"mixed_sin_plus{mode}", jnp.sin(theta_phase + zeta_phase)))
            specs.append((f"mixed_cos_minus{mode}", jnp.cos(theta_phase - zeta_phase)))
            specs.append((f"mixed_sin_minus{mode}", jnp.sin(theta_phase - zeta_phase)))
    return tuple(specs)


def build_rhs1_qi_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    include_global: bool = True,
    include_species: bool = True,
    include_x_ramp: bool = True,
    include_angular: bool = True,
    include_blocks: bool = True,
    dtype: Any = jnp.float64,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build deterministic structure-informed coarse-basis candidates.

    The candidates are deliberately redundant.  ``orthonormalize_rhs1_qi_coarse``
    removes dependent or tiny columns, leaving a compact coarse space while
    preserving deterministic ordering for metadata.
    """
    columns: list[ArrayLike] = []
    labels: list[str] = []
    if include_global:
        _coarse_append_candidate(
            columns, labels, jnp.ones((layout.total_size,), dtype=dtype), "global"
        )
    if include_species and layout.block_species is not None:
        species_to_blocks: dict[int, list[int]] = {}
        for block_index, species in enumerate(layout.block_species):
            species_to_blocks.setdefault(int(species), []).append(block_index)
        if len(species_to_blocks) > 1:
            for species in sorted(species_to_blocks):
                _coarse_append_candidate(
                    columns,
                    labels,
                    _coarse_group_constant(layout, species_to_blocks[species], dtype),
                    f"species:{species}",
                )
    if include_x_ramp and layout.block_x is not None:
        block_x = jnp.asarray(layout.block_x, dtype=dtype)
        centered = block_x - jnp.mean(block_x)
        if bool(jnp.max(jnp.abs(centered)) > 0):
            _coarse_append_candidate(
                columns,
                labels,
                _coarse_block_weighted_constant(
                    layout, tuple((float(v) for v in centered)), dtype
                ),
                "x_ramp",
            )
    if include_angular:
        theta = jnp.arange(int(layout.n_theta), dtype=dtype)
        zeta = jnp.arange(int(layout.n_zeta), dtype=dtype)
        theta_grid, zeta_grid = jnp.meshgrid(theta, zeta, indexing="ij")
        angular_specs = (
            ("theta_cos1", jnp.cos(2.0 * jnp.pi * theta_grid / float(layout.n_theta))),
            ("theta_sin1", jnp.sin(2.0 * jnp.pi * theta_grid / float(layout.n_theta))),
            ("zeta_cos1", jnp.cos(2.0 * jnp.pi * zeta_grid / float(layout.n_zeta))),
            ("zeta_sin1", jnp.sin(2.0 * jnp.pi * zeta_grid / float(layout.n_zeta))),
        )
        for label, angular_values in angular_specs:
            candidate = _coarse_angular_candidate(layout, angular_values, dtype)
            if candidate is not None:
                _coarse_append_candidate(columns, labels, candidate, label)
    if include_blocks:
        for block_index in range(len(layout.block_sizes)):
            _coarse_append_candidate(
                columns,
                labels,
                _coarse_block_constant(layout, block_index, dtype),
                f"block:{block_index}",
            )
    if not columns:
        return (_coarse_empty_matrix(layout.total_size, dtype), ())
    return (jnp.stack(columns, axis=1), tuple(labels))


def build_rhs1_qi_xblock_hard_seed_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    max_candidates: int = 96,
    max_angular_mode: int = 2,
    include_global: bool = True,
    include_species: bool = True,
    include_radial: bool = True,
    include_angular: bool = True,
    include_radial_angular: bool = True,
    include_constraint_moments: bool = True,
    include_schur: bool = True,
    include_blocks: bool = True,
    dtype: Any = jnp.float64,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build bounded enriched RHSMode=1 x-block hard-seed candidates.

    The candidate order favors low-dimensional moments before raw block
    directions: global and species constants, radial x ramps/curvature,
    constraint-like intra-block moments, angular harmonics, radial-angular
    products, block-Schur-like neighbor/species contrasts, then block constants.
    ``max_candidates`` bounds the pre-orthogonalization work deterministically.
    """
    candidate_limit = max(0, int(max_candidates))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    n_blocks = len(layout.block_sizes)

    def has_room() -> bool:
        return len(columns) < candidate_limit

    def finish() -> tuple[ArrayLike, tuple[str, ...]]:
        return _coarse_stack_candidates(layout.total_size, dtype, columns, labels)

    def add(values: ArrayLike | None, label: str) -> None:
        _coarse_append_candidate_if_room(
            columns, labels, values, label, candidate_limit
        )

    if candidate_limit <= 0:
        return (_coarse_empty_matrix(layout.total_size, dtype), ())
    species_to_blocks = _coarse_species_to_blocks(layout)
    block_x_values = tuple((float(value) for value in layout.block_x or ()))
    x_ramp_weights = _coarse_centered_unit_weights(block_x_values)
    x_quad_weights = _coarse_centered_power_weights(block_x_values, 2)
    if include_global:
        add(jnp.ones((layout.total_size,), dtype=dtype), "global")
    if not has_room():
        return finish()
    if include_species and len(species_to_blocks) > 1:
        for species in sorted(species_to_blocks):
            if not has_room():
                return finish()
            add(
                _coarse_group_constant(layout, species_to_blocks[species], dtype),
                f"species:{species}",
            )
    if include_radial:
        if x_ramp_weights is not None:
            if not has_room():
                return finish()
            add(
                _coarse_block_weighted_constant(layout, x_ramp_weights, dtype),
                "radial:x_ramp",
            )
        if x_quad_weights is not None:
            if not has_room():
                return finish()
            add(
                _coarse_block_weighted_constant(layout, x_quad_weights, dtype),
                "radial:x_quad",
            )
        for species in sorted(species_to_blocks):
            block_ids = species_to_blocks[species]
            local_x = tuple(
                (float(layout.block_x[block_index]) for block_index in block_ids)
            )
            local_ramp = _coarse_centered_unit_weights(local_x)
            local_quad = _coarse_centered_power_weights(local_x, 2)
            if local_ramp is not None:
                if not has_room():
                    return finish()
                weights = _coarse_weights_for_blocks(
                    n_blocks, tuple(zip(block_ids, local_ramp, strict=True))
                )
                add(
                    _coarse_block_weighted_constant(layout, weights, dtype),
                    f"radial:species:{species}:x_ramp",
                )
            if local_quad is not None:
                if not has_room():
                    return finish()
                weights = _coarse_weights_for_blocks(
                    n_blocks, tuple(zip(block_ids, local_quad, strict=True))
                )
                add(
                    _coarse_block_weighted_constant(layout, weights, dtype),
                    f"radial:species:{species}:x_quad",
                )
    if include_constraint_moments:
        ones = tuple((1.0 for _ in range(n_blocks)))
        if not has_room():
            return finish()
        add(
            _coarse_intra_block_moment_candidate(layout, ones, power=1, dtype=dtype),
            "constraint:xi_ramp",
        )
        if not has_room():
            return finish()
        add(
            _coarse_intra_block_moment_candidate(layout, ones, power=2, dtype=dtype),
            "constraint:xi_quad",
        )
        if x_ramp_weights is not None:
            if not has_room():
                return finish()
            add(
                _coarse_intra_block_moment_candidate(
                    layout, x_ramp_weights, power=1, dtype=dtype
                ),
                "constraint:radial_x_ramp*xi_ramp",
            )
        for species in sorted(species_to_blocks):
            if not has_room():
                return finish()
            weights = _coarse_weights_for_blocks(
                n_blocks,
                tuple(
                    ((block_index, 1.0) for block_index in species_to_blocks[species])
                ),
            )
            add(
                _coarse_intra_block_moment_candidate(
                    layout, weights, power=1, dtype=dtype
                ),
                f"constraint:species:{species}:xi_ramp",
            )
    angular_specs: tuple[tuple[str, ArrayLike], ...] = ()
    if has_room() and (include_angular or include_radial_angular):
        angular_specs = _coarse_angular_harmonic_specs(
            layout, max_angular_mode=max_angular_mode, include_mixed=True, dtype=dtype
        )
    if include_angular:
        for label, angular_values in angular_specs:
            if not has_room():
                return finish()
            add(_coarse_angular_candidate(layout, angular_values, dtype), label)
    if include_radial_angular and x_ramp_weights is not None:
        for label, angular_values in angular_specs:
            if not has_room():
                return finish()
            add(
                _coarse_block_weighted_angular_candidate(
                    layout, angular_values, x_ramp_weights, dtype
                ),
                f"radial:x_ramp*{label}",
            )
        if x_quad_weights is not None:
            for label, angular_values in angular_specs:
                if not has_room():
                    return finish()
                add(
                    _coarse_block_weighted_angular_candidate(
                        layout, angular_values, x_quad_weights, dtype
                    ),
                    f"radial:x_quad*{label}",
                )
    if include_schur:
        for species in sorted(species_to_blocks):
            x_to_blocks: dict[int, list[int]] = {}
            for block_index in species_to_blocks[species]:
                x_to_blocks.setdefault(int(layout.block_x[block_index]), []).append(
                    block_index
                )
            x_values = sorted(x_to_blocks)
            for left_x, right_x in zip(x_values, x_values[1:], strict=False):
                if not has_room():
                    return finish()
                weighted_blocks = tuple(
                    ((block_index, -1.0) for block_index in x_to_blocks[left_x])
                ) + tuple(((block_index, 1.0) for block_index in x_to_blocks[right_x]))
                add(
                    _coarse_block_weighted_constant(
                        layout,
                        _coarse_weights_for_blocks(n_blocks, weighted_blocks),
                        dtype,
                    ),
                    f"schur:x_diff:s{species}:{left_x}->{right_x}",
                )
            for left_x, center_x, right_x in zip(
                x_values, x_values[1:], x_values[2:], strict=False
            ):
                if not has_room():
                    return finish()
                weighted_blocks = (
                    tuple(((block_index, 1.0) for block_index in x_to_blocks[left_x]))
                    + tuple(
                        ((block_index, -2.0) for block_index in x_to_blocks[center_x])
                    )
                    + tuple(
                        ((block_index, 1.0) for block_index in x_to_blocks[right_x])
                    )
                )
                add(
                    _coarse_block_weighted_constant(
                        layout,
                        _coarse_weights_for_blocks(n_blocks, weighted_blocks),
                        dtype,
                    ),
                    f"schur:x_curve:s{species}:{left_x},{center_x},{right_x}",
                )
        x_species_blocks: dict[int, dict[int, list[int]]] = {}
        for block_index, (x_value, species) in enumerate(
            zip(layout.block_x, layout.block_species, strict=True)
        ):
            x_species_blocks.setdefault(int(x_value), {}).setdefault(
                int(species), []
            ).append(block_index)
        for x_value in sorted(x_species_blocks):
            species_values = sorted(x_species_blocks[x_value])
            for left_species, right_species in zip(
                species_values, species_values[1:], strict=False
            ):
                if not has_room():
                    return finish()
                weighted_blocks = tuple(
                    (
                        (block_index, -1.0)
                        for block_index in x_species_blocks[x_value][left_species]
                    )
                ) + tuple(
                    (
                        (block_index, 1.0)
                        for block_index in x_species_blocks[x_value][right_species]
                    )
                )
                add(
                    _coarse_block_weighted_constant(
                        layout,
                        _coarse_weights_for_blocks(n_blocks, weighted_blocks),
                        dtype,
                    ),
                    f"schur:species_diff:x{x_value}:s{left_species}->{right_species}",
                )
    if include_blocks:
        for block_index in range(n_blocks):
            if not has_room():
                return finish()
            add(
                _coarse_block_constant(layout, block_index, dtype),
                f"block:{block_index}",
            )
    return finish()


def orthonormalize_rhs1_qi_coarse_basis(
    candidates: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int | None = None,
) -> RHS1QICoarseBasis:
    """Return a deterministic modified-Gram-Schmidt basis with rank gating."""
    matrix = jnp.asarray(candidates)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2:
        raise ValueError("candidates must be a vector or a matrix")
    n_rows = int(matrix.shape[0])
    n_cols = int(matrix.shape[1])
    candidate_labels = tuple(
        (
            str(label)
            for label in labels or tuple((f"candidate:{i}" for i in range(n_cols)))
        )
    )
    if len(candidate_labels) != n_cols:
        raise ValueError("labels must match the number of candidate columns")
    candidate_norms = tuple(
        (float(jnp.linalg.norm(matrix[:, i])) for i in range(n_cols))
    )
    reference_norm = max(candidate_norms, default=0.0)
    threshold = max(float(atol), float(rtol) * float(reference_norm))
    rank_limit = n_cols if max_rank is None else max(0, min(int(max_rank), n_cols))
    q_columns: list[ArrayLike] = []
    accepted_labels: list[str] = []
    accepted_norms: list[float] = []
    for i in range(n_cols):
        if len(q_columns) >= rank_limit:
            break
        vector = matrix[:, i]
        norm = float(jnp.linalg.norm(vector))
        if norm <= threshold:
            continue
        residual = vector
        for _ in range(2):
            for q_col in q_columns:
                residual = residual - q_col * jnp.vdot(q_col, residual)
        residual_norm = float(jnp.linalg.norm(residual))
        if residual_norm <= threshold:
            continue
        q_columns.append(residual / residual_norm)
        accepted_labels.append(candidate_labels[i])
        accepted_norms.append(residual_norm)
    if q_columns:
        vectors = jnp.stack(q_columns, axis=1)
    else:
        vectors = _coarse_empty_matrix(n_rows, matrix.dtype)
    metadata = RHS1QICoarseBasisMetadata(
        total_size=n_rows,
        candidate_count=n_cols,
        rank=int(vectors.shape[1]),
        discarded_count=n_cols - int(vectors.shape[1]),
        candidate_labels=candidate_labels,
        accepted_labels=tuple(accepted_labels),
        candidate_norms=candidate_norms,
        accepted_norms=tuple(accepted_norms),
        rank_rtol=float(rtol),
        rank_atol=float(atol),
    )
    return RHS1QICoarseBasis(vectors=vectors, metadata=metadata)


def build_rhs1_qi_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int | None = None,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QICoarseBasis:
    """Build and rank-gate a QI coarse basis from a block layout."""
    candidates, labels = build_rhs1_qi_coarse_candidates(
        layout, dtype=dtype, **candidate_options
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates, labels=labels, rtol=rtol, atol=atol, max_rank=max_rank
    )


def build_rhs1_qi_xblock_hard_seed_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int = 32,
    max_candidates: int = 96,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QICoarseBasis:
    """Build the enriched bounded QI x-block hard-seed coarse basis.

    ``max_candidates`` limits the deterministic candidate list before rank
    gating; ``max_rank`` limits the accepted orthonormal columns.  The basis
    combines block constants, radial ramps, angular harmonics, species/global
    coupling moments, constraint-like intra-block moments, and block-Schur-like
    x/species contrasts.
    """
    candidates, labels = build_rhs1_qi_xblock_hard_seed_candidates(
        layout, max_candidates=max_candidates, dtype=dtype, **candidate_options
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates, labels=labels, rtol=rtol, atol=atol, max_rank=max(0, int(max_rank))
    )


def _coarse_rhs1_qi_block_layout_from_operator(
    op: Any, *, active_dof: bool
) -> tuple[RHS1QICoarseBlockLayout, int]:
    """Return the x/species QI block layout represented by a full operator."""
    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    block_sizes: list[int] = []
    block_x: list[int] = []
    block_species: list[int] = []
    for species in range(n_species):
        for ix in range(n_x):
            n_lx = int(nxi_for_x[ix]) if bool(active_dof) else n_l
            size = int(max(0, n_lx) * n_theta * n_zeta)
            if size <= 0:
                continue
            block_sizes.append(size)
            block_x.append(ix)
            block_species.append(species)
    if not block_sizes:
        raise RuntimeError("QI coarse seed found no active f blocks")
    layout = RHS1QICoarseBlockLayout(
        block_sizes=tuple(block_sizes),
        n_theta=n_theta,
        n_zeta=n_zeta,
        block_x=tuple(block_x),
        block_species=tuple(block_species),
    )
    return (layout, int(sum(block_sizes)))


def build_rhs1_xblock_qi_coarse_basis(
    *,
    op: Any,
    active_dof: bool,
    linear_size: int,
    max_rank: int,
    rank_rtol: float,
    include_angular: bool,
    include_blocks: bool,
    basis_kind: str = "legacy",
    max_candidates: int = 96,
    max_angular_mode: int = 2,
    include_radial: bool = True,
    include_radial_angular: bool = True,
    include_constraint_moments: bool = True,
    include_schur: bool = True,
) -> RHS1QICoarseBasis:
    """Build a padded QI coarse basis in the current x-block Krylov space.

    ``basis_kind='enriched'`` adds radial moments, angular harmonics,
    constraint-like moments, and local block-Schur contrast vectors before rank
    truncation. The legacy basis remains available for A/B tests and
    reproducibility. The returned basis is padded to ``linear_size`` so it can
    be passed directly to x-block Krylov/coarse hooks that include tail
    variables after the kinetic block.
    """
    layout, _f_block_size = _coarse_rhs1_qi_block_layout_from_operator(
        op, active_dof=bool(active_dof)
    )
    basis_kind_norm = str(basis_kind).strip().lower().replace("-", "_")
    if basis_kind_norm in {"enriched", "hard_seed", "xblock_hard_seed", "schur"}:
        basis = build_rhs1_qi_xblock_hard_seed_basis(
            layout,
            max_candidates=max(1, int(max_candidates)),
            max_rank=max(1, int(max_rank)),
            max_angular_mode=max(0, int(max_angular_mode)),
            rtol=float(rank_rtol),
            include_radial=bool(include_radial),
            include_angular=bool(include_angular),
            include_radial_angular=bool(include_radial_angular),
            include_constraint_moments=bool(include_constraint_moments),
            include_schur=bool(include_schur),
            include_blocks=bool(include_blocks),
        )
    elif basis_kind_norm in {"legacy", "basic", "coarse"}:
        basis = build_rhs1_qi_coarse_basis(
            layout,
            max_rank=max(1, int(max_rank)),
            rtol=float(rank_rtol),
            include_angular=bool(include_angular),
            include_blocks=bool(include_blocks),
        )
    else:
        raise ValueError(f"Unknown QI coarse seed basis kind: {basis_kind!r}")
    basis_vectors = jnp.asarray(basis.vectors, dtype=jnp.float64)
    tail_size = int(linear_size) - int(basis_vectors.shape[0])
    if tail_size < 0:
        raise RuntimeError(
            f"QI coarse seed basis is larger than the active x-block space ({basis_vectors.shape[0]} > {int(linear_size)})"
        )
    if tail_size > 0:
        basis_vectors = jnp.concatenate(
            [
                basis_vectors,
                jnp.zeros((tail_size, int(basis_vectors.shape[1])), dtype=jnp.float64),
            ],
            axis=0,
        )
    return RHS1QICoarseBasis(vectors=basis_vectors, metadata=basis.metadata)


def rhs1_xblock_qi_block_geometry_metadata(
    *, op: Any, active_dof: bool, linear_size: int, include_tail_block: bool = False
) -> dict[str, object]:
    """Return x/species block metadata for matrix-free QI device helpers."""
    layout, f_block_size = _coarse_rhs1_qi_block_layout_from_operator(
        op, active_dof=bool(active_dof)
    )
    block_sizes = tuple((int(value) for value in layout.block_sizes))
    block_x = tuple((int(value) for value in layout.block_x or ()))
    block_species = tuple((int(value) for value in layout.block_species or ()))
    tail_size = max(0, int(linear_size) - int(f_block_size))
    if bool(include_tail_block) and tail_size > 0:
        block_sizes = (*block_sizes, int(tail_size))
        block_x = (*block_x, -1)
        block_species = (*block_species, -1)
    return {
        "qi_block_sizes": block_sizes,
        "qi_block_x": block_x,
        "qi_block_species": block_species,
        "qi_block_f_size": int(f_block_size),
        "qi_block_tail_size": int(tail_size),
        "qi_block_tail_included": bool(include_tail_block and tail_size > 0),
    }


def _coarse_append_checked_direction(
    directions: list[tuple[str, ArrayLike]],
    *,
    name: str,
    direction: ArrayLike,
    total_size: int,
    max_directions: int,
) -> None:
    if len(directions) >= int(max_directions):
        return
    vec = jnp.asarray(direction, dtype=jnp.float64).reshape((-1,))
    if vec.shape != (int(total_size),):
        return
    try:
        norm = float(jnp.linalg.norm(vec))
    except Exception:
        return
    if np.isfinite(norm) and norm > 0.0:
        directions.append((str(name), vec))


def build_rhs1_xblock_global_coarse_basis(
    *,
    op: Any,
    rhs: ArrayLike,
    preconditioner: Callable[[ArrayLike], ArrayLike],
    include_rhs: bool,
    fsavg_lmax: int,
    max_extra_units: int,
    max_directions: int,
) -> tuple[tuple[str, ArrayLike], ...]:
    """Build fixed global directions missed by local x-block preconditioners."""
    rhs = jnp.asarray(rhs, dtype=jnp.float64).reshape((-1,))
    total = int(op.total_size)
    directions: list[tuple[str, ArrayLike]] = []

    def add(name: str, direction: ArrayLike) -> None:
        _coarse_append_checked_direction(
            directions,
            name=name,
            direction=direction,
            total_size=total,
            max_directions=int(max_directions),
        )

    if include_rhs:
        try:
            add("preconditioned_rhs", preconditioner(rhs))
        except Exception:
            pass
        add("raw_rhs", rhs)
    _coarse_append_tail_loads(
        op=op,
        rhs=rhs,
        directions=directions,
        max_extra_units=int(max_extra_units),
        max_directions=int(max_directions),
    )
    _coarse_append_constraint_source_loads(
        op=op, directions=directions, max_directions=int(max_directions)
    )
    _coarse_append_fsavg_loads(
        op=op,
        directions=directions,
        fsavg_lmax=int(fsavg_lmax),
        max_directions=int(max_directions),
    )
    return tuple(directions)


def _coarse_append_tail_loads(
    *,
    op: Any,
    rhs: ArrayLike,
    directions: list[tuple[str, ArrayLike]],
    max_extra_units: int,
    max_directions: int,
) -> None:
    total = int(op.total_size)
    extra_start = int(op.f_size + op.phi1_size)
    extra_size = int(op.extra_size)
    if extra_size <= 0:
        return
    rhs_extra = jnp.asarray(rhs, dtype=jnp.float64).reshape((-1,))[
        extra_start : extra_start + extra_size
    ]
    extra_dir = (
        jnp.zeros((total,), dtype=jnp.float64)
        .at[extra_start : extra_start + extra_size]
        .set(rhs_extra)
    )
    _coarse_append_checked_direction(
        directions,
        name="extra_rhs",
        direction=extra_dir,
        total_size=total,
        max_directions=int(max_directions),
    )
    if extra_size <= int(max_extra_units):
        for ie in range(extra_size):
            if len(directions) >= int(max_directions):
                break
            unit = jnp.zeros((total,), dtype=jnp.float64).at[extra_start + ie].set(1.0)
            _coarse_append_checked_direction(
                directions,
                name=f"extra_unit_{ie}",
                direction=unit,
                total_size=total,
                max_directions=int(max_directions),
            )


def _coarse_append_constraint_source_loads(
    *, op: Any, directions: list[tuple[str, ArrayLike]], max_directions: int
) -> None:
    if int(op.constraint_scheme) != 1:
        return
    total = int(op.total_size)
    ix0 = _ix_min(bool(op.point_at_x0))
    source_basis = _source_basis_constraint_scheme_1(op.x)
    for species in range(int(op.n_species)):
        for ibasis, basis in enumerate(source_basis):
            if len(directions) >= int(max_directions):
                break
            f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
            f_dir = f_dir.at[species, ix0:, 0, :, :].set(basis[ix0:, None, None])
            tail = jnp.zeros((total - int(op.f_size),), dtype=jnp.float64)
            _coarse_append_checked_direction(
                directions,
                name=f"constraint1_source_s{species}_{ibasis}",
                direction=jnp.concatenate([f_dir.reshape((-1,)), tail]),
                total_size=total,
                max_directions=int(max_directions),
            )


def _coarse_append_fsavg_loads(
    *,
    op: Any,
    directions: list[tuple[str, ArrayLike]],
    fsavg_lmax: int,
    max_directions: int,
) -> None:
    total = int(op.total_size)
    lmax_use = min(max(0, int(fsavg_lmax)), max(0, int(op.n_xi) - 1))
    angular_norm = float(max(1, int(op.n_theta) * int(op.n_zeta))) ** (-0.5)
    for il in range(lmax_use + 1):
        for species in range(int(op.n_species)):
            for ix in range(int(op.n_x)):
                if len(directions) >= int(max_directions):
                    return
                f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                f_dir = f_dir.at[species, ix, il, :, :].set(angular_norm)
                tail = jnp.zeros((total - int(op.f_size),), dtype=jnp.float64)
                _coarse_append_checked_direction(
                    directions,
                    name=f"fsavg_s{species}_x{ix}_l{il}",
                    direction=jnp.concatenate([f_dir.reshape((-1,)), tail]),
                    total_size=total,
                    max_directions=int(max_directions),
                )


def _coarse_append_low_angular_loads(
    *,
    op: Any,
    directions: list[tuple[str, ArrayLike]],
    angular_lmax: int,
    max_directions: int,
) -> None:
    angular_l_use = min(max(0, int(angular_lmax)), max(0, int(op.n_xi) - 1))
    if angular_l_use < 0 or int(op.n_theta) <= 1 or int(op.n_zeta) <= 1:
        return
    total = int(op.total_size)
    theta = jnp.arange(int(op.n_theta), dtype=jnp.float64)
    zeta = jnp.arange(int(op.n_zeta), dtype=jnp.float64)
    two_pi = float(2.0 * np.pi)
    mode_pairs = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (1, 2))
    for il in range(angular_l_use + 1):
        for m_mode, n_mode in mode_pairs:
            phase = two_pi * (
                float(m_mode) * theta[:, None] / float(max(1, int(op.n_theta)))
                + float(n_mode) * zeta[None, :] / float(max(1, int(op.n_zeta)))
            )
            for parity, pattern in (("cos", jnp.cos(phase)), ("sin", jnp.sin(phase))):
                pattern_norm = float(jnp.linalg.norm(pattern))
                if not np.isfinite(pattern_norm) or pattern_norm <= 0.0:
                    continue
                pattern = pattern / pattern_norm
                for species in range(int(op.n_species)):
                    if len(directions) >= int(max_directions):
                        return
                    f_dir = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
                    f_dir = f_dir.at[species, :, il, :, :].set(pattern[None, :, :])
                    tail = jnp.zeros((total - int(op.f_size),), dtype=jnp.float64)
                    _coarse_append_checked_direction(
                        directions,
                        name=f"angular_s{species}_allx_l{il}_m{m_mode}_n{n_mode}_{parity}",
                        direction=jnp.concatenate([f_dir.reshape((-1,)), tail]),
                        total_size=total,
                        max_directions=int(max_directions),
                    )


def build_rhs1_xblock_global_coupling_load_basis(
    *,
    op: Any,
    rhs: ArrayLike,
    include_rhs: bool,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
) -> tuple[tuple[str, ArrayLike], ...]:
    """Build low-rank source, moment, and angular load vectors."""
    rhs = jnp.asarray(rhs, dtype=jnp.float64).reshape((-1,))
    directions: list[tuple[str, ArrayLike]] = []
    if include_rhs:
        _coarse_append_checked_direction(
            directions,
            name="raw_rhs",
            direction=rhs,
            total_size=int(op.total_size),
            max_directions=int(max_directions),
        )
    _coarse_append_tail_loads(
        op=op,
        rhs=rhs,
        directions=directions,
        max_extra_units=int(max_extra_units),
        max_directions=int(max_directions),
    )
    _coarse_append_constraint_source_loads(
        op=op, directions=directions, max_directions=int(max_directions)
    )
    _coarse_append_fsavg_loads(
        op=op,
        directions=directions,
        fsavg_lmax=int(fsavg_lmax),
        max_directions=int(max_directions),
    )
    _coarse_append_low_angular_loads(
        op=op,
        directions=directions,
        angular_lmax=int(angular_lmax),
        max_directions=int(max_directions),
    )
    return tuple(directions)


def build_rhs1_xblock_smoothed_load_qi_basis(
    *,
    op: Any,
    rhs: ArrayLike,
    base_preconditioner: Callable[[ArrayLike], ArrayLike],
    direction_projector: Callable[[ArrayLike], ArrayLike] | None = None,
    expected_size: int | None = None,
    include_rhs: bool,
    fsavg_lmax: int,
    angular_lmax: int,
    max_extra_units: int,
    max_directions: int,
    rank_rtol: float,
    max_rank: int,
) -> tuple[RHS1QICoarseBasis, dict[str, object]]:
    """Build a QI coarse basis from smoothed global-coupling load vectors."""
    expected_size_use = (
        int(op.total_size) if expected_size is None else int(expected_size)
    )
    max_dirs_use = max(1, int(max_directions))
    raw_loads = build_rhs1_xblock_global_coupling_load_basis(
        op=op,
        rhs=rhs,
        include_rhs=bool(include_rhs),
        fsavg_lmax=int(fsavg_lmax),
        angular_lmax=int(angular_lmax),
        max_extra_units=int(max_extra_units),
        max_directions=max_dirs_use,
    )
    columns: list[ArrayLike] = []
    labels: list[str] = []
    for name, load in raw_loads[:max_dirs_use]:
        load_vec = jnp.asarray(load, dtype=jnp.float64).reshape((-1,))
        if direction_projector is not None:
            load_vec = jnp.asarray(
                direction_projector(load_vec), dtype=jnp.float64
            ).reshape((-1,))
        if int(load_vec.shape[0]) != expected_size_use:
            continue
        try:
            load_norm = float(jnp.linalg.norm(load_vec))
        except Exception:
            continue
        if not np.isfinite(load_norm) or load_norm <= 0.0:
            continue
        try:
            smoothed = jnp.asarray(
                base_preconditioner(
                    load_vec / jnp.asarray(load_norm, dtype=load_vec.dtype)
                ),
                dtype=jnp.float64,
            ).reshape((-1,))
        except Exception:
            continue
        if int(smoothed.shape[0]) != expected_size_use:
            continue
        try:
            smooth_norm = float(jnp.linalg.norm(smoothed))
        except Exception:
            continue
        if not np.isfinite(smooth_norm) or smooth_norm <= 0.0:
            continue
        columns.append(smoothed / jnp.asarray(smooth_norm, dtype=smoothed.dtype))
        labels.append(f"smoothed_load:{name}")
    if not columns:
        raise RuntimeError(
            "smoothed-load QI basis found no valid preconditioned load directions"
        )
    candidates = jnp.stack(tuple(columns), axis=1)
    basis = orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=tuple(labels),
        rtol=float(rank_rtol),
        max_rank=max(1, int(max_rank)),
    )
    metadata = {
        "load_basis_size": int(len(raw_loads)),
        "smoothed_candidate_count": int(len(columns)),
        "rank": int(basis.metadata.rank),
        "max_directions": int(max_dirs_use),
        "max_rank": int(max_rank),
        "fsavg_lmax": int(fsavg_lmax),
        "angular_lmax": int(angular_lmax),
        "include_rhs": bool(include_rhs),
        "accepted_labels": tuple(basis.metadata.accepted_labels),
    }
    return (basis, metadata)


def _coarse_apply_operator(operator: LinearOperator, vector: ArrayLike) -> ArrayLike:
    if callable(operator):
        return jnp.asarray(operator(vector))
    return jnp.asarray(operator) @ vector


def _coarse_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    if not callable(operator):
        return jnp.asarray(operator) @ basis_vectors
    columns = [
        _coarse_apply_operator(operator, basis_vectors[:, i])
        for i in range(int(basis_vectors.shape[1]))
    ]
    if not columns:
        return _coarse_empty_matrix(int(basis_vectors.shape[0]), basis_vectors.dtype)
    return jnp.stack(columns, axis=1)


def _coarse_small_regularized_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float | None = None
) -> ArrayLike:
    """Solve a small least-squares problem with JAX-only normal equations.

    QI coarse spaces are deliberately tiny.  Forming the Gram matrix avoids the
    shape-polymorphic paths in ``jnp.linalg.lstsq`` and gives the device Krylov
    lane a deterministic primitive that can be used inside compiled code.  A
    small scale-relative ridge keeps singular projected systems bounded.
    """
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    n_rows = int(a.shape[0])
    n_cols = int(a.shape[1])
    if int(rhs_vec.shape[0]) != n_rows:
        raise ValueError("rhs length must match matrix rows")
    if n_cols == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    a_h = jnp.conjugate(a).T
    gram = a_h @ a
    coarse_rhs = a_h @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=gram.dtype))
    rcond_value = 1e-14 if rcond is None else max(0.0, float(rcond))
    ridge = jnp.asarray(rcond_value, dtype=gram.dtype) * scale
    eye = jnp.eye(n_cols, dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, coarse_rhs)


def build_rhs1_qi_galerkin_preconditioner(
    operator: LinearOperator,
    *,
    basis: RHS1QICoarseBasis | None = None,
    layout: RHS1QICoarseBlockLayout | None = None,
    rcond: float | None = None,
) -> RHS1QIGalerkinPreconditioner:
    """Build a reusable projected QI coarse preconditioner.

    The returned object stores ``Q.T @ A @ Q`` and can be repeatedly applied to
    residual vectors without rebuilding ``A @ Q``.  This is the primitive needed
    by device-side hard-seed experiments: the expensive operator applications
    happen once, while each Krylov iteration only solves a small dense system.
    """
    if basis is None:
        if layout is None:
            raise ValueError("either basis or layout must be provided")
        basis = build_rhs1_qi_coarse_basis(layout)
    q = jnp.asarray(basis.vectors)
    rank = int(basis.metadata.rank)
    if rank <= 0:
        coarse_operator = jnp.zeros((0, 0), dtype=q.dtype)
    else:
        aq = _coarse_apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ aq
    coarse_norm = float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0
    regularization_rcond = 1e-14 if rcond is None else max(0.0, float(rcond))
    metadata = RHS1QIGalerkinPreconditionerMetadata(
        rank=rank,
        coarse_operator_shape=tuple((int(v) for v in coarse_operator.shape)),
        coarse_operator_norm=coarse_norm,
        regularization_rcond=float(regularization_rcond),
        basis_metadata=basis.metadata,
    )
    return RHS1QIGalerkinPreconditioner(
        basis=basis, coarse_operator=coarse_operator, metadata=metadata
    )


def apply_rhs1_qi_galerkin_correction(
    operator: LinearOperator,
    rhs: ArrayLike,
    *,
    current: ArrayLike | None = None,
    preconditioner: RHS1QIGalerkinPreconditioner | None = None,
    basis: RHS1QICoarseBasis | None = None,
    layout: RHS1QICoarseBlockLayout | None = None,
    damping: float = 1.0,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    rcond: float | None = None,
) -> RHS1QICoarseCorrection:
    """Apply a guarded Galerkin coarse correction using a reusable preconditioner."""
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    if current is None:
        current_vec = jnp.zeros_like(rhs_vec)
    else:
        current_vec = jnp.asarray(current).reshape((-1,))
    if current_vec.shape != rhs_vec.shape:
        raise ValueError("current and rhs must have the same shape")
    if preconditioner is None:
        preconditioner = build_rhs1_qi_galerkin_preconditioner(
            operator, basis=basis, layout=layout, rcond=rcond
        )
    residual_before = rhs_vec - _coarse_apply_operator(operator, current_vec)
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    rank = int(preconditioner.metadata.rank)
    zero_correction = jnp.zeros_like(rhs_vec)
    zero_coefficients = jnp.zeros((rank,), dtype=rhs_vec.dtype)
    if residual_before_norm == 0.0:
        return RHS1QICoarseCorrection(
            solution=current_vec,
            correction=zero_correction,
            coefficients=zero_coefficients,
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=1.0,
            applied=False,
            reason="zero_residual",
            basis_metadata=preconditioner.basis.metadata,
        )
    if rank <= 0:
        return RHS1QICoarseCorrection(
            solution=current_vec,
            correction=zero_correction,
            coefficients=zero_coefficients,
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            applied=False,
            reason="empty_basis",
            basis_metadata=preconditioner.basis.metadata,
        )
    coefficients = preconditioner.solve_coefficients(residual_before)
    correction = float(damping) * (preconditioner.basis.vectors @ coefficients)
    candidate_solution = current_vec + correction
    residual_after = rhs_vec - _coarse_apply_operator(operator, candidate_solution)
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
    if residual_after_norm < residual_before_norm - required_drop:
        return RHS1QICoarseCorrection(
            solution=candidate_solution,
            correction=correction,
            coefficients=coefficients,
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_after_norm,
            improvement_ratio=float(improvement_ratio),
            applied=True,
            reason="galerkin_residual_reduced",
            basis_metadata=preconditioner.basis.metadata,
        )
    return RHS1QICoarseCorrection(
        solution=current_vec,
        correction=zero_correction,
        coefficients=zero_coefficients,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_before_norm,
        improvement_ratio=1.0,
        applied=False,
        reason="not_reduced",
        basis_metadata=preconditioner.basis.metadata,
    )


def apply_rhs1_qi_coarse_correction(
    operator: LinearOperator,
    rhs: ArrayLike,
    *,
    current: ArrayLike | None = None,
    basis: RHS1QICoarseBasis | None = None,
    layout: RHS1QICoarseBlockLayout | None = None,
    damping: float = 1.0,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    rcond: float | None = None,
) -> RHS1QICoarseCorrection:
    """Apply a small guarded least-squares coarse correction.

    The correction solves ``min_c ||r - A Q c||`` where ``Q`` is the coarse basis
    and ``r = rhs - A current``.  The returned solution is updated only when the
    measured residual norm decreases by the requested acceptance margin.
    """
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    if current is None:
        current_vec = jnp.zeros_like(rhs_vec)
    else:
        current_vec = jnp.asarray(current).reshape((-1,))
    if current_vec.shape != rhs_vec.shape:
        raise ValueError("current and rhs must have the same shape")
    if basis is None:
        if layout is None:
            raise ValueError("either basis or layout must be provided")
        basis = build_rhs1_qi_coarse_basis(layout, dtype=rhs_vec.dtype)
    residual_before = rhs_vec - _coarse_apply_operator(operator, current_vec)
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    zero_correction = jnp.zeros_like(rhs_vec)
    zero_coefficients = jnp.zeros((int(basis.metadata.rank),), dtype=rhs_vec.dtype)
    if residual_before_norm == 0.0:
        return RHS1QICoarseCorrection(
            solution=current_vec,
            correction=zero_correction,
            coefficients=zero_coefficients,
            residual_before_norm=0.0,
            residual_after_norm=0.0,
            improvement_ratio=1.0,
            applied=False,
            reason="zero_residual",
            basis_metadata=basis.metadata,
        )
    if int(basis.metadata.rank) <= 0:
        return RHS1QICoarseCorrection(
            solution=current_vec,
            correction=zero_correction,
            coefficients=zero_coefficients,
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_before_norm,
            improvement_ratio=1.0,
            applied=False,
            reason="empty_basis",
            basis_metadata=basis.metadata,
        )
    coarse_operator = _coarse_apply_operator_to_basis(operator, basis.vectors)
    coefficients = _coarse_small_regularized_least_squares(
        coarse_operator, residual_before, rcond=rcond
    )
    correction = float(damping) * (basis.vectors @ coefficients)
    candidate_solution = current_vec + correction
    residual_after = rhs_vec - _coarse_apply_operator(operator, candidate_solution)
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
    if residual_after_norm < residual_before_norm - required_drop:
        return RHS1QICoarseCorrection(
            solution=candidate_solution,
            correction=correction,
            coefficients=coefficients,
            residual_before_norm=residual_before_norm,
            residual_after_norm=residual_after_norm,
            improvement_ratio=float(improvement_ratio),
            applied=True,
            reason="residual_reduced",
            basis_metadata=basis.metadata,
        )
    return RHS1QICoarseCorrection(
        solution=current_vec,
        correction=zero_correction,
        coefficients=zero_coefficients,
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_before_norm,
        improvement_ratio=1.0,
        applied=False,
        reason="not_reduced",
        basis_metadata=basis.metadata,
    )


@dataclass(frozen=True)
class RHS1QIActivePatternCoarseConfig:
    """Controls for residual active-pattern coarse directions."""

    max_rank: int = 16
    max_candidates: int = 48
    min_chunk_energy_fraction: float = 0.01
    include_block_pitch_chunks: bool = True
    include_block_angular_chunks: bool = True
    include_radial_pitch_chunks: bool = True
    include_radial_angular_chunks: bool = True
    include_block_chunks: bool = True
    include_radial_chunks: bool = True
    include_species_chunks: bool = True
    rtol: float = 1e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class _activepattern_ActivePatternCandidate:
    label: str
    indices: np.ndarray
    energy: float
    order: int


def _activepattern_empty_candidates(
    layout: RHS1QICoarseBlockLayout, dtype: Any
) -> tuple[ArrayLike, tuple[str, ...]]:
    return (jnp.zeros((layout.total_size, 0), dtype=dtype), ())


def _activepattern_is_physical_block(
    layout: RHS1QICoarseBlockLayout, block_index: int
) -> bool:
    block_x = tuple((int(value) for value in layout.block_x or ()))
    block_species = tuple((int(value) for value in layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and int(block_x[int(block_index)]) < 0:
        return False
    if (
        len(block_species) == len(layout.block_sizes)
        and int(block_species[int(block_index)]) < 0
    ):
        return False
    return True


def _activepattern_active_blocks(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    return tuple(
        (
            index
            for index in range(len(layout.block_sizes))
            if _activepattern_is_physical_block(layout, index)
        )
    )


def _activepattern_layout_is_supported(
    layout: RHS1QICoarseBlockLayout, active: tuple[int, ...]
) -> bool:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return False
    return all((int(layout.block_sizes[index]) % n_angular == 0 for index in active))


def _activepattern_block_indices(
    layout: RHS1QICoarseBlockLayout, block_index: int
) -> np.ndarray:
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    return np.arange(start, stop, dtype=np.int64)


def _activepattern_block_pitch_indices(
    layout: RHS1QICoarseBlockLayout, block_index: int, pitch_index: int
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    start = int(layout.block_offsets[int(block_index)]) + int(pitch_index) * n_angular
    return np.arange(start, start + n_angular, dtype=np.int64)


def _activepattern_block_angular_indices(
    layout: RHS1QICoarseBlockLayout, block_index: int, angular_index: int
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    block_size = int(layout.block_sizes[int(block_index)])
    n_pitch = block_size // n_angular
    start = int(layout.block_offsets[int(block_index)]) + int(angular_index)
    return start + n_angular * np.arange(n_pitch, dtype=np.int64)


def _activepattern_concat_indices(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    nonempty = tuple(
        (np.asarray(part, dtype=np.int64).reshape((-1,)) for part in parts if part.size)
    )
    if not nonempty:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(nonempty).astype(np.int64, copy=False)


def _activepattern_group_blocks_by_value(
    layout: RHS1QICoarseBlockLayout, active: tuple[int, ...], values: tuple[int, ...]
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if len(values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, tuple[int, ...]]] = []
    for value in sorted(
        {int(values[index]) for index in active if int(values[index]) >= 0}
    ):
        blocks = tuple((index for index in active if int(values[index]) == value))
        if blocks:
            result.append((value, blocks))
    return tuple(result)


def _activepattern_angular_label(
    layout: RHS1QICoarseBlockLayout, angular_index: int
) -> str:
    n_zeta = int(layout.n_zeta)
    theta = int(angular_index) // n_zeta
    zeta = int(angular_index) % n_zeta
    if int(layout.n_theta) > 1 and int(layout.n_zeta) > 1:
        return f"theta:{theta}:zeta:{zeta}"
    if int(layout.n_theta) > 1:
        return f"theta:{theta}"
    if int(layout.n_zeta) > 1:
        return f"zeta:{zeta}"
    return "all"


def _activepattern_energy(residual: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    values = residual[np.asarray(indices, dtype=np.int64)]
    return float(np.vdot(values, values).real)


def _activepattern_append_candidate(
    candidates: list[_activepattern_ActivePatternCandidate],
    seen_indices: set[bytes],
    residual: np.ndarray,
    indices: np.ndarray,
    label: str,
    *,
    min_energy: float,
) -> None:
    chunk_indices = np.asarray(indices, dtype=np.int64).reshape((-1,))
    if chunk_indices.size == 0:
        return
    key = chunk_indices.tobytes()
    if key in seen_indices:
        return
    seen_indices.add(key)
    energy = _activepattern_energy(residual, chunk_indices)
    if not np.isfinite(energy) or energy <= min_energy:
        return
    candidates.append(
        _activepattern_ActivePatternCandidate(
            label=f"active_pattern:{label}",
            indices=chunk_indices,
            energy=energy,
            order=len(seen_indices) - 1,
        )
    )


def _activepattern_build_active_patterns(
    layout: RHS1QICoarseBlockLayout,
    residual: np.ndarray,
    *,
    cfg: RHS1QIActivePatternCoarseConfig,
) -> tuple[_activepattern_ActivePatternCandidate, ...]:
    active = _activepattern_active_blocks(layout)
    if not active or not _activepattern_layout_is_supported(layout, active):
        return ()
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    block_n_pitch = {
        block_index: int(layout.block_sizes[block_index]) // n_angular
        for block_index in active
    }
    active_indices = _activepattern_concat_indices(
        tuple((_activepattern_block_indices(layout, index) for index in active))
    )
    active_energy = _activepattern_energy(residual, active_indices)
    if not np.isfinite(active_energy) or active_energy <= 0.0:
        return ()
    min_energy = active_energy * max(0.0, float(cfg.min_chunk_energy_fraction))
    candidates: list[_activepattern_ActivePatternCandidate] = []
    seen_indices: set[bytes] = set()

    def add(indices: np.ndarray, label: str) -> None:
        _activepattern_append_candidate(
            candidates, seen_indices, residual, indices, label, min_energy=min_energy
        )

    if cfg.include_block_pitch_chunks:
        for block_index in active:
            for pitch_index in range(block_n_pitch[block_index]):
                add(
                    _activepattern_block_pitch_indices(
                        layout, block_index, pitch_index
                    ),
                    f"block:{block_index}*pitch:{pitch_index}",
                )
    if cfg.include_block_angular_chunks:
        for block_index in active:
            for angular_index in range(n_angular):
                add(
                    _activepattern_block_angular_indices(
                        layout, block_index, angular_index
                    ),
                    f"block:{block_index}*angular:{_activepattern_angular_label(layout, angular_index)}",
                )
    block_x = tuple((int(value) for value in layout.block_x or ()))
    radial_groups = _activepattern_group_blocks_by_value(layout, active, block_x)
    if cfg.include_radial_pitch_chunks:
        for radial_value, blocks in radial_groups:
            max_pitch = max((block_n_pitch[block_index] for block_index in blocks))
            for pitch_index in range(max_pitch):
                add(
                    _activepattern_concat_indices(
                        tuple(
                            (
                                _activepattern_block_pitch_indices(
                                    layout, block_index, pitch_index
                                )
                                for block_index in blocks
                                if pitch_index < block_n_pitch[block_index]
                            )
                        )
                    ),
                    f"radial:{radial_value}*pitch:{pitch_index}",
                )
    if cfg.include_radial_angular_chunks:
        for radial_value, blocks in radial_groups:
            for angular_index in range(n_angular):
                add(
                    _activepattern_concat_indices(
                        tuple(
                            (
                                _activepattern_block_angular_indices(
                                    layout, block_index, angular_index
                                )
                                for block_index in blocks
                            )
                        )
                    ),
                    f"radial:{radial_value}*angular:{_activepattern_angular_label(layout, angular_index)}",
                )
    if cfg.include_block_chunks:
        for block_index in active:
            add(
                _activepattern_block_indices(layout, block_index),
                f"block:{block_index}",
            )
    if cfg.include_radial_chunks:
        for radial_value, blocks in radial_groups:
            add(
                _activepattern_concat_indices(
                    tuple(
                        (
                            _activepattern_block_indices(layout, block_index)
                            for block_index in blocks
                        )
                    )
                ),
                f"radial:{radial_value}",
            )
    if cfg.include_species_chunks:
        block_species = tuple((int(value) for value in layout.block_species or ()))
        for species, blocks in _activepattern_group_blocks_by_value(
            layout, active, block_species
        ):
            add(
                _activepattern_concat_indices(
                    tuple(
                        (
                            _activepattern_block_indices(layout, block_index)
                            for block_index in blocks
                        )
                    )
                ),
                f"species:{species}",
            )
    candidates.sort(key=lambda candidate: (-candidate.energy, candidate.order))
    return tuple(candidates[: max(0, int(cfg.max_candidates))])


def _activepattern_candidate_from_indices(
    residual: ArrayLike, indices: np.ndarray
) -> ArrayLike:
    residual_vec = jnp.asarray(residual).reshape((-1,))
    index_array = jnp.asarray(np.asarray(indices, dtype=np.int32), dtype=jnp.int32)
    return jnp.zeros_like(residual_vec).at[index_array].set(residual_vec[index_array])


def build_rhs1_qi_active_pattern_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    residual_seed: ArrayLike,
    *,
    config: RHS1QIActivePatternCoarseConfig | None = None,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build residual-seed active-pattern candidate columns.

    Unsupported active block shapes and residuals with no finite physical-block
    energy fail closed by returning an empty candidate matrix.  Explicit tail
    blocks, marked with negative ``block_x`` or ``block_species`` labels, are
    ignored rather than used to infer pitch/angular chunking.
    """
    cfg = RHS1QIActivePatternCoarseConfig() if config is None else config
    residual_vec = jnp.asarray(residual_seed, dtype=cfg.dtype).reshape((-1,))
    if int(residual_vec.shape[0]) != int(layout.total_size):
        raise ValueError("residual_seed length must match layout.total_size")
    residual_host = np.asarray(residual_vec)
    if not np.all(np.isfinite(residual_host)):
        return _activepattern_empty_candidates(layout, cfg.dtype)
    patterns = _activepattern_build_active_patterns(layout, residual_host, cfg=cfg)
    if not patterns:
        return _activepattern_empty_candidates(layout, cfg.dtype)
    columns = [
        _activepattern_candidate_from_indices(residual_vec, pattern.indices)
        for pattern in patterns
    ]
    labels = tuple((pattern.label for pattern in patterns))
    return (jnp.stack(tuple(columns), axis=1), labels)


def build_rhs1_qi_active_pattern_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    residual_seed: ArrayLike,
    *,
    config: RHS1QIActivePatternCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated residual active-pattern QI coarse basis."""
    cfg = RHS1QIActivePatternCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_active_pattern_coarse_candidates(
        layout, residual_seed, config=cfg
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


def project_rhs1_qi_active_pattern_correction(
    residual: ArrayLike, basis: RHS1QICoarseBasis
) -> ArrayLike:
    """Project ``residual`` into an active-pattern coarse basis."""
    residual_vec = jnp.asarray(residual, dtype=basis.vectors.dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return jnp.zeros_like(residual_vec)
    return basis.vectors @ (jnp.conjugate(basis.vectors).T @ residual_vec)


@dataclass(frozen=True)
class RHS1QIPhaseSpaceCoarseConfig:
    """Controls for physics-derived phase-space coarse directions."""

    max_rank: int = 24
    trapped_boundary_fraction: float = 0.35
    include_trapped: bool = True
    include_passing: bool = True
    include_boundary: bool = True
    include_even: bool = True
    include_odd: bool = True
    include_radial: bool = True
    include_species: bool = True
    rtol: float = 1e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


def _phasespace_empty_candidates(
    layout: RHS1QICoarseBlockLayout, dtype: Any
) -> tuple[ArrayLike, tuple[str, ...]]:
    return (jnp.zeros((layout.total_size, 0), dtype=dtype), ())


def _phasespace_centered_unit_weights(
    values: tuple[float, ...],
) -> tuple[float, ...] | None:
    if not values:
        return None
    mean = sum(values) / float(len(values))
    centered = tuple((value - mean for value in values))
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple((value / scale for value in centered))


def _phasespace_pitch_grid(size: int) -> np.ndarray:
    if int(size) <= 1:
        return np.zeros((int(size),), dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(size), dtype=np.float64)


def _phasespace_boundary_mask(pitch: np.ndarray, boundary: float) -> np.ndarray:
    if pitch.size <= 1:
        return np.ones_like(pitch)
    spacing = 2.0 / float(max(1, pitch.size - 1))
    width = max(0.5 * spacing, spacing * 0.25)
    mask = np.abs(np.abs(pitch) - boundary) <= width
    if not bool(np.any(mask)):
        mask[int(np.argmin(np.abs(np.abs(pitch) - boundary)))] = True
    return mask.astype(np.float64)


def _phasespace_candidate_from_pitch_weights(
    layout: RHS1QICoarseBlockLayout,
    *,
    block_weights: tuple[float, ...],
    pitch_weight: Any,
    dtype: Any,
) -> ArrayLike | None:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0 or len(block_weights) != len(layout.block_sizes):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    any_nonzero = False
    for block_index, block_size in enumerate(layout.block_sizes):
        block_weight = float(block_weights[block_index])
        if block_weight == 0.0:
            continue
        if int(block_size) % n_angular != 0:
            return None
        n_pitch = int(block_size) // n_angular
        pitch = _phasespace_pitch_grid(n_pitch)
        weights = np.asarray(pitch_weight(pitch), dtype=np.float64).reshape((-1,))
        if int(weights.size) != n_pitch:
            raise ValueError(
                "pitch_weight must return one value per inferred pitch index"
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("phase-space pitch weights must be finite")
        if not bool(np.any(weights != 0.0)):
            continue
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(
            block_weight * jnp.repeat(jnp.asarray(weights, dtype=dtype), n_angular)
        )
        any_nonzero = True
    if not any_nonzero:
        return None
    return values


def _phasespace_append_candidate(
    columns: list[ArrayLike], labels: list[str], values: ArrayLike | None, label: str
) -> None:
    if values is None:
        return
    vector = jnp.asarray(values).reshape((-1,))
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector)
    labels.append(f"phase_space:{label}")


def _phasespace_all_block_weights(layout: RHS1QICoarseBlockLayout) -> tuple[float, ...]:
    return tuple(
        (
            1.0 if _phasespace_is_physical_block(layout, index) else 0.0
            for index in range(len(layout.block_sizes))
        )
    )


def _phasespace_is_physical_block(
    layout: RHS1QICoarseBlockLayout, block_index: int
) -> bool:
    block_x = tuple((int(value) for value in layout.block_x or ()))
    block_species = tuple((int(value) for value in layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and int(block_x[int(block_index)]) < 0:
        return False
    if (
        len(block_species) == len(layout.block_sizes)
        and int(block_species[int(block_index)]) < 0
    ):
        return False
    return True


def _phasespace_species_weights(
    layout: RHS1QICoarseBlockLayout,
) -> tuple[tuple[int, tuple[float, ...]], ...]:
    species_values = tuple((int(value) for value in layout.block_species or ()))
    if len(species_values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, tuple[float, ...]]] = []
    for species in sorted(set(species_values)):
        if species < 0:
            continue
        weights = tuple((1.0 if value == species else 0.0 for value in species_values))
        result.append((species, weights))
    return tuple(result)


def _phasespace_block_x_ramp(
    layout: RHS1QICoarseBlockLayout,
) -> tuple[float, ...] | None:
    block_x = tuple((float(value) for value in layout.block_x or ()))
    if len(block_x) != len(layout.block_sizes):
        return None
    active = tuple(
        (
            index
            for index in range(len(layout.block_sizes))
            if _phasespace_is_physical_block(layout, index)
        )
    )
    if len(active) <= 1:
        return None
    active_weights = _phasespace_centered_unit_weights(
        tuple((block_x[index] for index in active))
    )
    if active_weights is None:
        return None
    result = [0.0 for _ in layout.block_sizes]
    for block_index, weight in zip(active, active_weights, strict=True):
        result[int(block_index)] = float(weight)
    return tuple(result)


def build_rhs1_qi_phase_space_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIPhaseSpaceCoarseConfig | None = None,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build deterministic phase-space candidate columns.

    Pitch coordinates are inferred within each block after the angular
    ``n_theta * n_zeta`` stride.  If a block size is not divisible by that
    angular stride, unsupported pitch-dependent candidates are skipped by
    returning an empty candidate set rather than guessing a layout.
    """
    cfg = RHS1QIPhaseSpaceCoarseConfig() if config is None else config
    dtype = cfg.dtype
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return _phasespace_empty_candidates(layout, dtype)
    active = tuple(
        (
            index
            for index in range(len(layout.block_sizes))
            if _phasespace_is_physical_block(layout, index)
        )
    )
    if not active or any(
        (int(layout.block_sizes[index]) % n_angular != 0 for index in active)
    ):
        return _phasespace_empty_candidates(layout, dtype)
    boundary = max(0.0, min(1.0, float(cfg.trapped_boundary_fraction)))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    all_blocks = _phasespace_all_block_weights(layout)

    def trapped(pitch: np.ndarray) -> np.ndarray:
        mask = np.abs(pitch) <= boundary
        if pitch.size and (not bool(np.any(mask))):
            mask[int(np.argmin(np.abs(pitch)))] = True
        return mask.astype(np.float64)

    def passing(pitch: np.ndarray) -> np.ndarray:
        return (np.abs(pitch) > boundary).astype(np.float64)

    def passing_sign(pitch: np.ndarray) -> np.ndarray:
        return passing(pitch) * np.sign(pitch)

    def boundary_band(pitch: np.ndarray) -> np.ndarray:
        return _phasespace_boundary_mask(pitch, boundary)

    def even_abs(pitch: np.ndarray) -> np.ndarray:
        values = np.abs(pitch)
        centered = values - float(np.mean(values))
        scale = float(np.max(np.abs(centered))) if centered.size else 0.0
        if scale <= 0.0:
            return np.zeros_like(values)
        return centered / scale

    def odd_pitch(pitch: np.ndarray) -> np.ndarray:
        return pitch

    def odd_sign(pitch: np.ndarray) -> np.ndarray:
        return np.sign(pitch)

    if cfg.include_trapped:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=trapped, dtype=dtype
            ),
            "pitch_band:trapped",
        )
    if cfg.include_passing:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=passing, dtype=dtype
            ),
            "pitch_band:passing",
        )
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=passing_sign, dtype=dtype
            ),
            "pitch_band:passing*sign",
        )
    if cfg.include_boundary:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=boundary_band,
                dtype=dtype,
            ),
            "pitch_band:boundary",
        )
    if cfg.include_even:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=even_abs, dtype=dtype
            ),
            "pitch:even_abs",
        )
    if cfg.include_odd:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=odd_pitch, dtype=dtype
            ),
            "pitch:odd",
        )
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout, block_weights=all_blocks, pitch_weight=odd_sign, dtype=dtype
            ),
            "pitch:sign",
        )
    radial = _phasespace_block_x_ramp(layout) if cfg.include_radial else None
    if radial is not None:
        _phasespace_append_candidate(
            columns,
            labels,
            _phasespace_candidate_from_pitch_weights(
                layout,
                block_weights=radial,
                pitch_weight=lambda pitch: np.ones_like(pitch),
                dtype=dtype,
            ),
            "radial:ramp",
        )
        if cfg.include_trapped:
            _phasespace_append_candidate(
                columns,
                labels,
                _phasespace_candidate_from_pitch_weights(
                    layout, block_weights=radial, pitch_weight=trapped, dtype=dtype
                ),
                "radial:ramp*pitch_band:trapped",
            )
        if cfg.include_passing:
            _phasespace_append_candidate(
                columns,
                labels,
                _phasespace_candidate_from_pitch_weights(
                    layout, block_weights=radial, pitch_weight=passing, dtype=dtype
                ),
                "radial:ramp*pitch_band:passing",
            )
    if cfg.include_species:
        for species, species_block_weights in _phasespace_species_weights(layout):
            _phasespace_append_candidate(
                columns,
                labels,
                _phasespace_candidate_from_pitch_weights(
                    layout,
                    block_weights=species_block_weights,
                    pitch_weight=lambda pitch: np.ones_like(pitch),
                    dtype=dtype,
                ),
                f"species:{species}",
            )
            if cfg.include_trapped:
                _phasespace_append_candidate(
                    columns,
                    labels,
                    _phasespace_candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=trapped,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch_band:trapped",
                )
            if cfg.include_passing:
                _phasespace_append_candidate(
                    columns,
                    labels,
                    _phasespace_candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=passing,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch_band:passing",
                )
            if cfg.include_odd:
                _phasespace_append_candidate(
                    columns,
                    labels,
                    _phasespace_candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=odd_sign,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch:sign",
                )
    if not columns:
        return _phasespace_empty_candidates(layout, dtype)
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def build_rhs1_qi_phase_space_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIPhaseSpaceCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated QI coarse basis from phase-space directions."""
    cfg = RHS1QIPhaseSpaceCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_phase_space_coarse_candidates(layout, config=cfg)
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


@dataclass(frozen=True)
class RHS1QIResidualRegionCoarseConfig:
    """Controls for residual-localized region coarse directions."""

    max_rank: int = 16
    max_candidates: int = 48
    trapped_boundary_fraction: float = 0.35
    min_region_energy_fraction: float = 0.01
    include_global_active_region: bool = False
    include_block_regions: bool = True
    include_block_bounce_regions: bool = True
    include_pitch_regions: bool = True
    include_radial_regions: bool = True
    include_radial_bounce_regions: bool = True
    include_species_regions: bool = True
    include_species_bounce_regions: bool = True
    region_bands: str = "bounce,trapped,passing"
    rtol: float = 1e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class _residualregions_RegionCandidate:
    label: str
    mask: np.ndarray
    energy: float
    order: int


def _residualregions_empty_candidates(
    layout: RHS1QICoarseBlockLayout, dtype: Any
) -> tuple[ArrayLike, tuple[str, ...]]:
    return (jnp.zeros((layout.total_size, 0), dtype=dtype), ())


def _residualregions_is_physical_block(
    layout: RHS1QICoarseBlockLayout, block_index: int
) -> bool:
    block_x = tuple((int(value) for value in layout.block_x or ()))
    block_species = tuple((int(value) for value in layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and block_x[int(block_index)] < 0:
        return False
    if (
        len(block_species) == len(layout.block_sizes)
        and block_species[int(block_index)] < 0
    ):
        return False
    return True


def _residualregions_active_blocks(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    return tuple(
        (
            index
            for index in range(len(layout.block_sizes))
            if _residualregions_is_physical_block(layout, index)
        )
    )


def _residualregions_layout_is_supported(
    layout: RHS1QICoarseBlockLayout, active: tuple[int, ...]
) -> bool:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return False
    return all((int(layout.block_sizes[index]) % n_angular == 0 for index in active))


def _residualregions_pitch_grid(size: int) -> np.ndarray:
    if int(size) <= 1:
        return np.zeros((int(size),), dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(size), dtype=np.float64)


def _residualregions_boundary_mask(pitch: np.ndarray, boundary: float) -> np.ndarray:
    if pitch.size <= 1:
        return np.ones_like(pitch, dtype=bool)
    spacing = 2.0 / float(max(1, pitch.size - 1))
    width = max(0.5 * spacing, 0.25 * spacing)
    mask = np.abs(np.abs(pitch) - boundary) <= width
    if not bool(np.any(mask)):
        mask[int(np.argmin(np.abs(np.abs(pitch) - boundary)))] = True
    return mask.astype(bool)


def _residualregions_bounce_band_masks(
    n_pitch: int, boundary: float
) -> tuple[tuple[str, np.ndarray], ...]:
    pitch = _residualregions_pitch_grid(n_pitch)
    trapped = np.abs(pitch) <= boundary
    if pitch.size and (not bool(np.any(trapped))):
        trapped[int(np.argmin(np.abs(pitch)))] = True
    passing_negative = pitch < -boundary
    passing_positive = pitch > boundary
    return (
        ("trapped", trapped.astype(bool)),
        ("boundary", _residualregions_boundary_mask(pitch, boundary)),
        ("passing_negative", passing_negative.astype(bool)),
        ("passing_positive", passing_positive.astype(bool)),
    )


def _residualregions_selected_bounce_labels(region_bands: object) -> tuple[str, ...]:
    """Normalize public band controls to internal trapped/boundary/passing masks."""
    tokens = tuple(
        (
            token.strip().lower().replace("-", "_")
            for token in str(region_bands or "").split(",")
            if token.strip()
        )
    )
    if not tokens:
        tokens = ("bounce", "trapped", "passing")
    selected: list[str] = []

    def add(label: str) -> None:
        if label not in selected:
            selected.append(label)

    for token in tokens:
        if token in {"all", "default", "pitch"}:
            for label in (
                "trapped",
                "boundary",
                "passing_negative",
                "passing_positive",
            ):
                add(label)
        elif token in {"bounce", "boundary", "trapped_boundary"}:
            add("boundary")
        elif token == "trapped":
            add("trapped")
        elif token in {"passing", "passing_both"}:
            add("passing_negative")
            add("passing_positive")
        elif token in {"passing_negative", "passing_minus", "negative_passing"}:
            add("passing_negative")
        elif token in {"passing_positive", "passing_plus", "positive_passing"}:
            add("passing_positive")
    return tuple(selected) or (
        "trapped",
        "boundary",
        "passing_negative",
        "passing_positive",
    )


def _residualregions_block_mask(
    layout: RHS1QICoarseBlockLayout, block_index: int
) -> np.ndarray:
    mask = np.zeros((layout.total_size,), dtype=bool)
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    mask[start:stop] = True
    return mask


def _residualregions_block_bounce_mask(
    layout: RHS1QICoarseBlockLayout, block_index: int, pitch_mask: np.ndarray
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    mask = np.zeros((layout.total_size,), dtype=bool)
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    mask[start:stop] = np.repeat(np.asarray(pitch_mask, dtype=bool), n_angular)
    return mask


def _residualregions_union_masks(masks: tuple[np.ndarray, ...]) -> np.ndarray:
    if not masks:
        return np.zeros((0,), dtype=bool)
    result = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        result = np.logical_or(result, mask)
    return result


def _residualregions_energy(residual: np.ndarray, mask: np.ndarray) -> float:
    values = residual[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return 0.0
    return float(np.vdot(values, values).real)


def _residualregions_append_region(
    regions: list[_residualregions_RegionCandidate],
    seen_masks: set[bytes],
    residual: np.ndarray,
    mask: np.ndarray,
    label: str,
    *,
    min_energy: float,
) -> None:
    if mask.shape != residual.shape or not bool(np.any(mask)):
        return
    key = np.asarray(mask, dtype=np.uint8).tobytes()
    if key in seen_masks:
        return
    seen_masks.add(key)
    energy = _residualregions_energy(residual, mask)
    if not np.isfinite(energy) or energy <= min_energy:
        return
    regions.append(
        _residualregions_RegionCandidate(
            label=f"residual_region:{label}",
            mask=np.asarray(mask, dtype=bool),
            energy=energy,
            order=len(seen_masks) - 1,
        )
    )


def _residualregions_group_masks_by_value(
    layout: RHS1QICoarseBlockLayout, active: tuple[int, ...], values: tuple[int, ...]
) -> tuple[tuple[int, np.ndarray], ...]:
    if len(values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, np.ndarray]] = []
    for value in sorted(
        {int(values[index]) for index in active if int(values[index]) >= 0}
    ):
        masks = tuple(
            (
                _residualregions_block_mask(layout, index)
                for index in active
                if int(values[index]) == value
            )
        )
        if masks:
            result.append((value, _residualregions_union_masks(masks)))
    return tuple(result)


def _residualregions_build_regions(
    layout: RHS1QICoarseBlockLayout,
    residual: np.ndarray,
    *,
    cfg: RHS1QIResidualRegionCoarseConfig,
) -> tuple[_residualregions_RegionCandidate, ...]:
    active = _residualregions_active_blocks(layout)
    if not active or not _residualregions_layout_is_supported(layout, active):
        return ()
    active_mask = _residualregions_union_masks(
        tuple((_residualregions_block_mask(layout, index) for index in active))
    )
    active_energy = _residualregions_energy(residual, active_mask)
    if not np.isfinite(active_energy) or active_energy <= 0.0:
        return ()
    min_fraction = max(0.0, float(cfg.min_region_energy_fraction))
    min_energy = active_energy * min_fraction
    boundary = max(0.0, min(1.0, float(cfg.trapped_boundary_fraction)))
    regions: list[_residualregions_RegionCandidate] = []
    seen_masks: set[bytes] = set()
    selected_bounce_labels = _residualregions_selected_bounce_labels(cfg.region_bands)

    def add(mask: np.ndarray, label: str) -> None:
        _residualregions_append_region(
            regions, seen_masks, residual, mask, label, min_energy=min_energy
        )

    block_bounce_masks: dict[tuple[int, str], np.ndarray] = {}
    for block_index in active:
        n_angular = int(layout.n_theta) * int(layout.n_zeta)
        n_pitch = int(layout.block_sizes[block_index]) // n_angular
        for bounce_label, pitch_mask in _residualregions_bounce_band_masks(
            n_pitch, boundary
        ):
            block_bounce_masks[block_index, bounce_label] = (
                _residualregions_block_bounce_mask(layout, block_index, pitch_mask)
            )
    if cfg.include_block_bounce_regions:
        for block_index in active:
            for bounce_label in selected_bounce_labels:
                add(
                    block_bounce_masks[block_index, bounce_label],
                    f"block:{block_index}*bounce:{bounce_label}",
                )
    if cfg.include_radial_bounce_regions:
        block_x = tuple((int(value) for value in layout.block_x or ()))
        for radial_value, radial_mask in _residualregions_group_masks_by_value(
            layout, active, block_x
        ):
            for bounce_label in selected_bounce_labels:
                masks = tuple(
                    (
                        block_bounce_masks[block_index, bounce_label]
                        for block_index in active
                        if int(block_x[block_index]) == radial_value
                    )
                )
                if masks:
                    add(
                        np.logical_and(
                            radial_mask, _residualregions_union_masks(masks)
                        ),
                        f"radial:{radial_value}*bounce:{bounce_label}",
                    )
    if cfg.include_species_bounce_regions:
        block_species = tuple((int(value) for value in layout.block_species or ()))
        for species, species_mask in _residualregions_group_masks_by_value(
            layout, active, block_species
        ):
            for bounce_label in selected_bounce_labels:
                masks = tuple(
                    (
                        block_bounce_masks[block_index, bounce_label]
                        for block_index in active
                        if int(block_species[block_index]) == species
                    )
                )
                if masks:
                    add(
                        np.logical_and(
                            species_mask, _residualregions_union_masks(masks)
                        ),
                        f"species:{species}*bounce:{bounce_label}",
                    )
    if cfg.include_pitch_regions:
        for bounce_label in selected_bounce_labels:
            masks = tuple(
                (
                    block_bounce_masks[block_index, bounce_label]
                    for block_index in active
                )
            )
            add(_residualregions_union_masks(masks), f"bounce:{bounce_label}")
    if cfg.include_block_regions:
        for block_index in active:
            add(
                _residualregions_block_mask(layout, block_index), f"block:{block_index}"
            )
    if cfg.include_radial_regions:
        block_x = tuple((int(value) for value in layout.block_x or ()))
        for radial_value, radial_mask in _residualregions_group_masks_by_value(
            layout, active, block_x
        ):
            add(radial_mask, f"radial:{radial_value}")
    if cfg.include_species_regions:
        block_species = tuple((int(value) for value in layout.block_species or ()))
        for species, species_mask in _residualregions_group_masks_by_value(
            layout, active, block_species
        ):
            add(species_mask, f"species:{species}")
    if cfg.include_global_active_region:
        add(active_mask, "active")
    regions.sort(key=lambda region: (-region.energy, region.order))
    return tuple(regions[: max(0, int(cfg.max_candidates))])


def build_rhs1_qi_residual_region_coarse_candidates(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    *,
    config: RHS1QIResidualRegionCoarseConfig | None = None,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build residual-restricted region candidate columns.

    Unsupported active block shapes fail closed by returning an empty candidate
    matrix. Explicit nonphysical tail blocks, identified by negative
    ``block_x`` or ``block_species`` labels, are ignored.
    """
    cfg = RHS1QIResidualRegionCoarseConfig() if config is None else config
    residual_vec = jnp.asarray(residual, dtype=cfg.dtype).reshape((-1,))
    if int(residual_vec.shape[0]) != int(layout.total_size):
        raise ValueError("residual length must match layout.total_size")
    residual_host = np.asarray(residual_vec)
    if not np.all(np.isfinite(residual_host)):
        return _residualregions_empty_candidates(layout, cfg.dtype)
    regions = _residualregions_build_regions(layout, residual_host, cfg=cfg)
    if not regions:
        return _residualregions_empty_candidates(layout, cfg.dtype)
    columns = [
        residual_vec * jnp.asarray(region.mask, dtype=residual_vec.dtype)
        for region in regions
    ]
    labels = tuple((region.label for region in regions))
    return (jnp.stack(tuple(columns), axis=1), labels)


def build_rhs1_qi_residual_region_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    *,
    config: RHS1QIResidualRegionCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated residual-region QI coarse basis."""
    cfg = RHS1QIResidualRegionCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_residual_region_coarse_candidates(
        layout, residual, config=cfg
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


def project_rhs1_qi_residual_region_correction(
    residual: ArrayLike, basis: RHS1QICoarseBasis
) -> ArrayLike:
    """Project ``residual`` into a residual-region coarse basis."""
    residual_vec = jnp.asarray(residual, dtype=basis.vectors.dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return jnp.zeros_like(residual_vec)
    return basis.vectors @ (jnp.conjugate(basis.vectors).T @ residual_vec)


@dataclass(frozen=True)
class RHS1QIGlobalMomentLayout:
    """Simple block layout for low-dimensional QI moment candidates."""

    block_sizes: Sequence[int]
    block_x: Sequence[float] | None = None
    block_species: Sequence[int] | None = None

    def __post_init__(self) -> None:
        block_sizes = tuple((int(size) for size in self.block_sizes))
        if not block_sizes:
            raise ValueError("block_sizes must contain at least one block")
        if any((size <= 0 for size in block_sizes)):
            raise ValueError("block_sizes must be positive")
        object.__setattr__(self, "block_sizes", block_sizes)
        n_blocks = len(block_sizes)
        if self.block_x is None:
            block_x = tuple((float(index) for index in range(n_blocks)))
        else:
            block_x = tuple((float(value) for value in self.block_x))
        if len(block_x) != n_blocks:
            raise ValueError("block_x must have the same length as block_sizes")
        object.__setattr__(self, "block_x", block_x)
        if self.block_species is None:
            block_species = tuple((0 for _ in range(n_blocks)))
        else:
            block_species = tuple((int(value) for value in self.block_species))
        if len(block_species) != n_blocks:
            raise ValueError("block_species must have the same length as block_sizes")
        object.__setattr__(self, "block_species", block_species)

    @property
    def total_size(self) -> int:
        """Return the full vector length represented by this layout."""
        return int(sum(self.block_sizes))

    @property
    def block_offsets(self) -> tuple[int, ...]:
        """Return block starts including the final sentinel offset."""
        offsets = [0]
        for size in self.block_sizes:
            offsets.append(offsets[-1] + int(size))
        return tuple(offsets)


@dataclass(frozen=True)
class RHS1QIGlobalMomentBasisMetadata:
    """Rank-gating diagnostics for the global moment basis."""

    total_size: int
    candidate_count: int
    rank: int
    discarded_count: int
    candidate_labels: tuple[str, ...]
    accepted_labels: tuple[str, ...]
    candidate_norms: tuple[float, ...]
    accepted_norms: tuple[float, ...]
    rank_rtol: float
    rank_atol: float


@dataclass(frozen=True)
class RHS1QIGlobalMomentBasis:
    """Orthonormal global moment basis."""

    vectors: ArrayLike
    metadata: RHS1QIGlobalMomentBasisMetadata


@dataclass(frozen=True)
class RHS1QIGlobalMomentClosureMetadata:
    """Acceptance and conditioning metadata for a global moment closure."""

    accepted: bool
    reason: str
    total_size: int
    candidate_count: int
    rank: int
    numerical_rank: int
    discarded_count: int
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    labels: tuple[str, ...]
    candidate_labels: tuple[str, ...]
    solver: str
    condition_estimate: float
    min_singular_value: float
    max_singular_value: float
    singular_threshold: float
    regularization_rcond: float
    damping: float

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics for solver traces."""
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "total_size": int(self.total_size),
            "candidate_count": int(self.candidate_count),
            "rank": int(self.rank),
            "numerical_rank": int(self.numerical_rank),
            "discarded_count": int(self.discarded_count),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None
            if self.improvement_ratio is None
            else float(self.improvement_ratio),
            "labels": self.labels,
            "candidate_labels": self.candidate_labels,
            "solver": self.solver,
            "condition_estimate": float(self.condition_estimate),
            "min_singular_value": float(self.min_singular_value),
            "max_singular_value": float(self.max_singular_value),
            "singular_threshold": float(self.singular_threshold),
            "regularization_rcond": float(self.regularization_rcond),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1QIGlobalMomentClosure:
    """Reusable JAX-compatible closure action over global QI moments."""

    basis: RHS1QIGlobalMomentBasis
    operator_on_basis: ArrayLike
    coarse_operator: ArrayLike
    metadata: RHS1QIGlobalMomentClosureMetadata

    def solve_coefficients(self, residual: ArrayLike) -> ArrayLike:
        """Solve the small moment residual equation for a residual vector."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        rank = int(self.metadata.rank)
        if not bool(self.metadata.accepted) or rank <= 0:
            return jnp.zeros((rank,), dtype=residual_vec.dtype)
        if self.metadata.solver == "galerkin":
            projected = jnp.conjugate(self.basis.vectors).T @ residual_vec
            return _globalmoments_regularized_square_solve(
                self.coarse_operator,
                projected,
                rcond=float(self.metadata.regularization_rcond),
            )
        return _globalmoments_regularized_action_least_squares(
            self.operator_on_basis,
            residual_vec,
            rcond=float(self.metadata.regularization_rcond),
        )

    def apply(self, residual: ArrayLike) -> ArrayLike:
        """Return the lifted moment correction for a residual vector."""
        residual_vec = jnp.asarray(residual).reshape((-1,))
        if not bool(self.metadata.accepted) or int(self.metadata.rank) <= 0:
            return jnp.zeros_like(residual_vec)
        coefficients = self.solve_coefficients(residual_vec)
        return float(self.metadata.damping) * (self.basis.vectors @ coefficients)

    def as_preconditioner(self) -> Callable[[ArrayLike], ArrayLike]:
        """Return a callable suitable for Krylov/preconditioner hooks."""
        return self.apply


def _globalmoments_empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _globalmoments_append(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
    limit: int,
) -> None:
    if values is None or len(columns) >= int(limit):
        return
    vector = jnp.asarray(values).reshape((-1,))
    if int(vector.shape[0]) == 0:
        return
    columns.append(vector)
    labels.append(str(label))


def _globalmoments_centered_unit_weights(
    values: Sequence[float],
) -> tuple[float, ...] | None:
    raw = tuple((float(value) for value in values))
    if not raw:
        return None
    mean = sum(raw) / float(len(raw))
    centered = tuple((value - mean for value in raw))
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple((value / scale for value in centered))


def _globalmoments_centered_power_weights(
    values: Sequence[float], power: int
) -> tuple[float, ...] | None:
    linear = _globalmoments_centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple((value ** int(power) for value in linear))
    return _globalmoments_centered_unit_weights(powered)


def _globalmoments_weights_for_blocks(
    n_blocks: int, weighted_blocks: Sequence[tuple[int, float]]
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _globalmoments_block_weighted_constant(
    layout: RHS1QIGlobalMomentLayout, weights: Sequence[float], dtype: Any
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    if not any((float(value) != 0.0 for value in weights)):
        return None
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _globalmoments_species_to_blocks(
    layout: RHS1QIGlobalMomentLayout,
) -> dict[int, list[int]]:
    species_to_blocks: dict[int, list[int]] = {}
    for block_index, species in enumerate(layout.block_species or ()):
        species_to_blocks.setdefault(int(species), []).append(block_index)
    return species_to_blocks


def _globalmoments_centered_index_moment(
    size: int, power: int, dtype: Any
) -> ArrayLike | None:
    size = int(size)
    if size <= 1:
        return None
    denominator = float(max(1, size - 1))
    base = tuple((2.0 * float(index) / denominator - 1.0 for index in range(size)))
    if int(power) == 1:
        values = base
    else:
        centered = _globalmoments_centered_unit_weights(
            tuple((value ** int(power) for value in base))
        )
        if centered is None:
            return None
        values = centered
    return jnp.asarray(values, dtype=dtype)


def _globalmoments_block_weighted_constraint_moment(
    layout: RHS1QIGlobalMomentLayout,
    weights: Sequence[float],
    *,
    power: int,
    dtype: Any,
) -> ArrayLike | None:
    if len(weights) != len(layout.block_sizes):
        raise ValueError("weights must match block_sizes")
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    any_supported = False
    any_nonzero = False
    for block_index, block_size in enumerate(layout.block_sizes):
        weight = float(weights[block_index])
        if weight == 0.0:
            continue
        moment = _globalmoments_centered_index_moment(
            int(block_size), int(power), dtype
        )
        if moment is None:
            continue
        any_supported = True
        any_nonzero = True
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(weight * moment)
    if not any_supported or not any_nonzero:
        return None
    return values


def build_rhs1_qi_global_moment_candidates(
    layout: RHS1QIGlobalMomentLayout,
    *,
    max_candidates: int = 32,
    include_current: bool = True,
    include_profile: bool = True,
    include_constraint: bool = True,
    include_cross_moments: bool = True,
    include_blocks: bool = False,
    dtype: Any = jnp.float64,
) -> tuple[ArrayLike, tuple[str, ...]]:
    """Build low-dimensional current/constraint/profile moment candidates."""
    limit = max(0, int(max_candidates))
    if limit <= 0:
        return (_globalmoments_empty_matrix(layout.total_size, dtype), ())
    columns: list[ArrayLike] = []
    labels: list[str] = []
    n_blocks = len(layout.block_sizes)
    species_to_blocks = _globalmoments_species_to_blocks(layout)
    block_x = tuple(
        (float(value) for value in layout.block_x or tuple(range(n_blocks)))
    )
    x_ramp = _globalmoments_centered_unit_weights(block_x)
    x_quad = _globalmoments_centered_power_weights(block_x, 2)

    def add(values: ArrayLike | None, label: str) -> None:
        _globalmoments_append(columns, labels, values, label, limit)

    if include_current:
        add(jnp.ones((layout.total_size,), dtype=dtype), "current:global")
        species_values = sorted(species_to_blocks)
        if len(species_values) > 1:
            reference = species_values[0]
            for species in species_values[1:]:
                weights = _globalmoments_weights_for_blocks(
                    n_blocks,
                    tuple(((block, -1.0) for block in species_to_blocks[reference]))
                    + tuple(((block, 1.0) for block in species_to_blocks[species])),
                )
                add(
                    _globalmoments_block_weighted_constant(layout, weights, dtype),
                    f"current:species_contrast:{reference}->{species}",
                )
    profile_weights: list[tuple[str, tuple[float, ...]]] = []
    if include_profile:
        if x_ramp is not None:
            profile_weights.append(("profile:x_ramp", x_ramp))
        if x_quad is not None:
            profile_weights.append(("profile:x_quad", x_quad))
        for label, weights in profile_weights:
            add(_globalmoments_block_weighted_constant(layout, weights, dtype), label)
    constraint_weights: list[tuple[str, tuple[float, ...], int]] = []
    if include_constraint:
        ones = tuple((1.0 for _ in range(n_blocks)))
        constraint_weights.append(("constraint:m1", ones, 1))
        constraint_weights.append(("constraint:m2", ones, 2))
        for label, weights, power in constraint_weights:
            add(
                _globalmoments_block_weighted_constraint_moment(
                    layout, weights, power=power, dtype=dtype
                ),
                label,
            )
    if include_cross_moments and include_constraint:
        if include_profile:
            for profile_label, profile in profile_weights:
                add(
                    _globalmoments_block_weighted_constraint_moment(
                        layout, profile, power=1, dtype=dtype
                    ),
                    f"{profile_label}*constraint:m1",
                )
                add(
                    _globalmoments_block_weighted_constraint_moment(
                        layout, profile, power=2, dtype=dtype
                    ),
                    f"{profile_label}*constraint:m2",
                )
        species_values = sorted(species_to_blocks)
        if len(species_values) > 1:
            reference = species_values[0]
            for species in species_values[1:]:
                weights = _globalmoments_weights_for_blocks(
                    n_blocks,
                    tuple(((block, -1.0) for block in species_to_blocks[reference]))
                    + tuple(((block, 1.0) for block in species_to_blocks[species])),
                )
                add(
                    _globalmoments_block_weighted_constraint_moment(
                        layout, weights, power=1, dtype=dtype
                    ),
                    f"current:species_contrast:{reference}->{species}*constraint:m1",
                )
    if include_blocks:
        for block_index in range(n_blocks):
            weights = _globalmoments_weights_for_blocks(n_blocks, ((block_index, 1.0),))
            add(
                _globalmoments_block_weighted_constant(layout, weights, dtype),
                f"block:{block_index}",
            )
    if not columns:
        return (_globalmoments_empty_matrix(layout.total_size, dtype), ())
    return (jnp.stack(tuple(columns), axis=1), tuple(labels))


def orthonormalize_rhs1_qi_global_moment_basis(
    candidates: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int | None = None,
) -> RHS1QIGlobalMomentBasis:
    """Return a deterministic modified-Gram-Schmidt moment basis."""
    matrix = jnp.asarray(candidates)
    if matrix.ndim == 1:
        matrix = matrix.reshape((-1, 1))
    if matrix.ndim != 2:
        raise ValueError("candidates must be a vector or a matrix")
    n_rows = int(matrix.shape[0])
    n_cols = int(matrix.shape[1])
    candidate_labels = tuple(
        (
            str(label)
            for label in labels or tuple((f"candidate:{i}" for i in range(n_cols)))
        )
    )
    if len(candidate_labels) != n_cols:
        raise ValueError("labels must match the number of candidate columns")
    candidate_norms = tuple(
        (float(jnp.linalg.norm(matrix[:, i])) for i in range(n_cols))
    )
    reference_norm = max(candidate_norms, default=0.0)
    threshold = max(float(atol), float(rtol) * float(reference_norm))
    rank_limit = n_cols if max_rank is None else max(0, min(int(max_rank), n_cols))
    q_columns: list[ArrayLike] = []
    accepted_labels: list[str] = []
    accepted_norms: list[float] = []
    for column_index in range(n_cols):
        if len(q_columns) >= rank_limit:
            break
        vector = matrix[:, column_index]
        norm = float(jnp.linalg.norm(vector))
        if norm <= threshold:
            continue
        residual = vector
        for _ in range(2):
            for q_col in q_columns:
                residual = residual - q_col * jnp.vdot(q_col, residual)
        residual_norm = float(jnp.linalg.norm(residual))
        if residual_norm <= threshold:
            continue
        q_columns.append(residual / residual_norm)
        accepted_labels.append(candidate_labels[column_index])
        accepted_norms.append(residual_norm)
    vectors = (
        jnp.stack(tuple(q_columns), axis=1)
        if q_columns
        else _globalmoments_empty_matrix(n_rows, matrix.dtype)
    )
    metadata = RHS1QIGlobalMomentBasisMetadata(
        total_size=n_rows,
        candidate_count=n_cols,
        rank=int(vectors.shape[1]),
        discarded_count=n_cols - int(vectors.shape[1]),
        candidate_labels=candidate_labels,
        accepted_labels=tuple(accepted_labels),
        candidate_norms=candidate_norms,
        accepted_norms=tuple(accepted_norms),
        rank_rtol=float(rtol),
        rank_atol=float(atol),
    )
    return RHS1QIGlobalMomentBasis(vectors=vectors, metadata=metadata)


def build_rhs1_qi_global_moment_basis(
    layout: RHS1QIGlobalMomentLayout,
    *,
    rtol: float = 1e-10,
    atol: float = 0.0,
    max_rank: int | None = 16,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QIGlobalMomentBasis:
    """Build and rank-gate the global moment basis from a block layout."""
    candidates, labels = build_rhs1_qi_global_moment_candidates(
        layout, dtype=dtype, **candidate_options
    )
    return orthonormalize_rhs1_qi_global_moment_basis(
        candidates, labels=labels, rtol=rtol, atol=atol, max_rank=max_rank
    )


def _globalmoments_apply_operator(
    operator: LinearOperator, vector: ArrayLike
) -> ArrayLike:
    vector_arr = jnp.asarray(vector).reshape((-1,))
    if callable(operator):
        return jnp.asarray(operator(vector_arr)).reshape((-1,))
    return (jnp.asarray(operator) @ vector_arr).reshape((-1,))


def _globalmoments_apply_operator_to_basis(
    operator: LinearOperator, basis_vectors: ArrayLike
) -> ArrayLike:
    q = jnp.asarray(basis_vectors)
    if q.ndim != 2:
        raise ValueError("basis vectors must be a 2D matrix")
    if int(q.shape[1]) == 0:
        return _globalmoments_empty_matrix(int(q.shape[0]), q.dtype)
    if not callable(operator):
        return jnp.asarray(operator) @ q
    columns = tuple(
        (
            _globalmoments_apply_operator(operator, q[:, i])
            for i in range(int(q.shape[1]))
        )
    )
    return jnp.stack(columns, axis=1)


def _globalmoments_regularized_square_solve(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2 or int(a.shape[0]) != int(a.shape[1]):
        raise ValueError("matrix must be square")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    row_sums = jnp.sum(jnp.abs(a), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=a.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=a.dtype) * scale
    eye = jnp.eye(int(a.shape[0]), dtype=a.dtype)
    return jnp.linalg.solve(a + ridge * eye, rhs_vec)


def _globalmoments_regularized_action_least_squares(
    matrix: ArrayLike, rhs: ArrayLike, *, rcond: float
) -> ArrayLike:
    a = jnp.asarray(matrix)
    rhs_vec = jnp.asarray(rhs, dtype=a.dtype).reshape((-1,))
    if a.ndim != 2:
        raise ValueError("matrix must be 2D")
    if int(a.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("rhs length must match matrix rows")
    if int(a.shape[1]) == 0:
        return jnp.zeros((0,), dtype=a.dtype)
    a_h = jnp.conjugate(a).T
    gram = a_h @ a
    normal_rhs = a_h @ rhs_vec
    row_sums = jnp.sum(jnp.abs(gram), axis=1)
    scale = jnp.maximum(jnp.max(row_sums), jnp.asarray(1.0, dtype=gram.dtype))
    ridge = jnp.asarray(max(0.0, float(rcond)), dtype=gram.dtype) * scale
    eye = jnp.eye(int(gram.shape[0]), dtype=gram.dtype)
    return jnp.linalg.solve(gram + ridge * eye, normal_rhs)


def _globalmoments_normalize_solver(solver: str) -> str:
    normalized = str(solver).strip().lower().replace("-", "_")
    if normalized in {"galerkin", "projected", "block_schur_galerkin"}:
        return "galerkin"
    if normalized in {
        "action_lstsq",
        "action_ls",
        "least_squares",
        "minres",
        "residual_lstsq",
    }:
        return "action_lstsq"
    raise ValueError("solver must be 'galerkin' or 'action_lstsq'")


def _globalmoments_conditioning_metadata(
    matrix: ArrayLike, *, rcond: float, atol: float
) -> tuple[int, float, float, float, float]:
    a = jnp.asarray(matrix)
    if a.ndim != 2 or int(a.shape[0]) == 0 or int(a.shape[1]) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    singular_values = jnp.linalg.svd(a, compute_uv=False)
    if int(singular_values.shape[0]) == 0:
        return (0, 0.0, 0.0, 0.0, 0.0)
    max_sv = float(jnp.max(singular_values))
    min_sv = float(jnp.min(singular_values))
    threshold = max(float(atol), max_sv * max(0.0, float(rcond)))
    numerical_rank = int(jnp.sum(singular_values > threshold))
    condition = 0.0 if max_sv <= 0.0 else max_sv / max(min_sv, threshold, 1e-300)
    return (numerical_rank, condition, min_sv, max_sv, threshold)


def _globalmoments_zero_closure(
    *,
    basis: RHS1QIGlobalMomentBasis,
    operator_on_basis: ArrayLike,
    coarse_operator: ArrayLike,
    solver: str,
    reason: str,
    residual_before_norm: float,
    regularization_rcond: float,
    damping: float,
    numerical_rank: int = 0,
    condition_estimate: float = 0.0,
    min_singular_value: float = 0.0,
    max_singular_value: float = 0.0,
    singular_threshold: float = 0.0,
) -> RHS1QIGlobalMomentClosure:
    metadata = RHS1QIGlobalMomentClosureMetadata(
        accepted=False,
        reason=reason,
        total_size=int(basis.metadata.total_size),
        candidate_count=int(basis.metadata.candidate_count),
        rank=int(basis.metadata.rank),
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        residual_before_norm=float(residual_before_norm),
        residual_after_norm=float(residual_before_norm),
        improvement_ratio=1.0 if residual_before_norm > 0.0 else None,
        labels=tuple(basis.metadata.accepted_labels),
        candidate_labels=tuple(basis.metadata.candidate_labels),
        solver=solver,
        condition_estimate=float(condition_estimate),
        min_singular_value=float(min_singular_value),
        max_singular_value=float(max_singular_value),
        singular_threshold=float(singular_threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
    )
    return RHS1QIGlobalMomentClosure(
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )


def build_rhs1_qi_global_moment_closure(
    *,
    operator: LinearOperator,
    rhs: ArrayLike,
    x0: ArrayLike | None = None,
    layout: RHS1QIGlobalMomentLayout | None = None,
    candidates: ArrayLike | None = None,
    labels: Sequence[str] | None = None,
    solver: str = "action_lstsq",
    max_candidates: int = 32,
    max_rank: int | None = 16,
    basis_rtol: float = 1e-10,
    basis_atol: float = 0.0,
    min_rank: int = 1,
    require_independent_candidates: bool = False,
    regularization_rcond: float = 1e-12,
    conditioning_atol: float = 0.0,
    damping: float = 1.0,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
    **candidate_options: Any,
) -> tuple[ArrayLike, RHS1QIGlobalMomentClosure]:
    """Build, probe, and fail-close a global moment closure correction.

    The setup residual is ``r = rhs - A x0``.  The closure forms ``A Q`` with the
    supplied operator matvec and solves either the Galerkin equation
    ``Q* A Q c = Q* r`` or the action least-squares equation
    ``min ||A Q c - r||``.  The returned solution is updated only when the true
    residual norm strictly decreases by the requested acceptance margin.
    """
    solver_use = _globalmoments_normalize_solver(solver)
    rhs_vec = jnp.asarray(rhs).reshape((-1,))
    x_initial = (
        jnp.zeros_like(rhs_vec)
        if x0 is None
        else jnp.asarray(x0, dtype=rhs_vec.dtype).reshape((-1,))
    )
    if int(x_initial.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("x0 and rhs must have the same length")
    if candidates is None:
        if layout is None:
            raise ValueError("either layout or candidates must be provided")
        candidates, labels = build_rhs1_qi_global_moment_candidates(
            layout,
            max_candidates=max_candidates,
            dtype=rhs_vec.dtype,
            **candidate_options,
        )
    basis = orthonormalize_rhs1_qi_global_moment_basis(
        candidates,
        labels=labels,
        rtol=float(basis_rtol),
        atol=float(basis_atol),
        max_rank=max_rank,
    )
    q = jnp.asarray(basis.vectors, dtype=rhs_vec.dtype)
    if int(q.shape[0]) != int(rhs_vec.shape[0]):
        raise ValueError("basis length must match rhs length")
    rank = int(basis.metadata.rank)
    operator_on_basis = (
        _globalmoments_apply_operator_to_basis(operator, q)
        if rank > 0
        else _globalmoments_empty_matrix(int(rhs_vec.shape[0]), rhs_vec.dtype)
    )
    coarse_operator = (
        jnp.conjugate(q).T @ operator_on_basis
        if rank > 0
        else jnp.zeros((0, 0), dtype=rhs_vec.dtype)
    )
    conditioning_matrix = (
        coarse_operator if solver_use == "galerkin" else operator_on_basis
    )
    numerical_rank, condition, min_sv, max_sv, threshold = (
        _globalmoments_conditioning_metadata(
            conditioning_matrix,
            rcond=float(regularization_rcond),
            atol=float(conditioning_atol),
        )
    )
    residual_before = rhs_vec - _globalmoments_apply_operator(operator, x_initial)
    residual_before_norm = float(jnp.linalg.norm(residual_before))
    rank_limit = (
        int(basis.metadata.candidate_count)
        if max_rank is None
        else min(int(basis.metadata.candidate_count), int(max_rank))
    )
    candidate_rank_deficient = bool(int(basis.metadata.rank) < rank_limit)
    if residual_before_norm == 0.0:
        closure = _globalmoments_zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="zero_residual",
            residual_before_norm=0.0,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return (x_initial, closure)
    if rank <= 0:
        closure = _globalmoments_zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="empty_basis",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
        )
        return (x_initial, closure)
    if rank < max(1, int(min_rank)):
        closure = _globalmoments_zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="rank_below_min",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return (x_initial, closure)
    if bool(require_independent_candidates) and candidate_rank_deficient:
        closure = _globalmoments_zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="candidate_rank_deficient",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return (x_initial, closure)
    if int(numerical_rank) < rank:
        closure = _globalmoments_zero_closure(
            basis=basis,
            operator_on_basis=operator_on_basis,
            coarse_operator=coarse_operator,
            solver=solver_use,
            reason="coarse_operator_rank_deficient",
            residual_before_norm=residual_before_norm,
            regularization_rcond=float(regularization_rcond),
            damping=float(damping),
            numerical_rank=numerical_rank,
            condition_estimate=condition,
            min_singular_value=min_sv,
            max_singular_value=max_sv,
            singular_threshold=threshold,
        )
        return (x_initial, closure)
    if solver_use == "galerkin":
        projected = jnp.conjugate(q).T @ residual_before
        coefficients = _globalmoments_regularized_square_solve(
            coarse_operator, projected, rcond=float(regularization_rcond)
        )
    else:
        coefficients = _globalmoments_regularized_action_least_squares(
            operator_on_basis, residual_before, rcond=float(regularization_rcond)
        )
    correction = float(damping) * (q @ coefficients)
    candidate_solution = x_initial + correction
    residual_after = rhs_vec - _globalmoments_apply_operator(
        operator, candidate_solution
    )
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm
    finite = bool(jnp.isfinite(jnp.asarray(residual_after_norm)))
    required_drop = max(
        float(acceptance_atol),
        residual_before_norm * max(0.0, float(min_relative_improvement)),
    )
    accepted = bool(
        finite and residual_after_norm < residual_before_norm - required_drop
    )
    if accepted:
        reason = "residual_reduced"
    elif not finite:
        reason = "nonfinite_candidate"
    else:
        reason = "not_reduced"
    metadata = RHS1QIGlobalMomentClosureMetadata(
        accepted=accepted,
        reason=reason,
        total_size=int(basis.metadata.total_size),
        candidate_count=int(basis.metadata.candidate_count),
        rank=rank,
        numerical_rank=int(numerical_rank),
        discarded_count=int(basis.metadata.discarded_count),
        residual_before_norm=residual_before_norm,
        residual_after_norm=residual_after_norm if finite else residual_before_norm,
        improvement_ratio=float(improvement_ratio) if finite else None,
        labels=tuple(basis.metadata.accepted_labels),
        candidate_labels=tuple(basis.metadata.candidate_labels),
        solver=solver_use,
        condition_estimate=float(condition),
        min_singular_value=float(min_sv),
        max_singular_value=float(max_sv),
        singular_threshold=float(threshold),
        regularization_rcond=float(regularization_rcond),
        damping=float(damping),
    )
    closure = RHS1QIGlobalMomentClosure(
        basis=basis,
        operator_on_basis=operator_on_basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
    )
    return (candidate_solution if accepted else x_initial, closure)


__all__ = (
    "RHS1QICoarseBasis",
    "RHS1QICoarseBasisMetadata",
    "RHS1QICoarseBlockLayout",
    "RHS1QICoarseCorrection",
    "RHS1QIGalerkinPreconditioner",
    "RHS1QIGalerkinPreconditionerMetadata",
    "apply_rhs1_qi_galerkin_correction",
    "apply_rhs1_qi_coarse_correction",
    "build_rhs1_xblock_global_coarse_basis",
    "build_rhs1_xblock_global_coupling_load_basis",
    "build_rhs1_xblock_qi_coarse_basis",
    "build_rhs1_xblock_smoothed_load_qi_basis",
    "build_rhs1_qi_galerkin_preconditioner",
    "build_rhs1_qi_coarse_basis",
    "build_rhs1_qi_coarse_candidates",
    "build_rhs1_qi_xblock_hard_seed_basis",
    "build_rhs1_qi_xblock_hard_seed_candidates",
    "orthonormalize_rhs1_qi_coarse_basis",
    "rhs1_xblock_qi_block_geometry_metadata",
    "RHS1QIActivePatternCoarseConfig",
    "build_rhs1_qi_active_pattern_coarse_basis",
    "build_rhs1_qi_active_pattern_coarse_candidates",
    "project_rhs1_qi_active_pattern_correction",
    "RHS1QIPhaseSpaceCoarseConfig",
    "build_rhs1_qi_phase_space_coarse_basis",
    "build_rhs1_qi_phase_space_coarse_candidates",
    "RHS1QIResidualRegionCoarseConfig",
    "build_rhs1_qi_residual_region_coarse_basis",
    "build_rhs1_qi_residual_region_coarse_candidates",
    "project_rhs1_qi_residual_region_correction",
    "RHS1QIGlobalMomentBasis",
    "RHS1QIGlobalMomentBasisMetadata",
    "RHS1QIGlobalMomentClosure",
    "RHS1QIGlobalMomentClosureMetadata",
    "RHS1QIGlobalMomentLayout",
    "build_rhs1_qi_global_moment_basis",
    "build_rhs1_qi_global_moment_candidates",
    "build_rhs1_qi_global_moment_closure",
    "orthonormalize_rhs1_qi_global_moment_basis",
)
