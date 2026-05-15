"""RHSMode=1 QI coarse-basis correction helpers.

These utilities are intentionally independent of ``v3_driver.py`` so the hard
QI seed path can build and validate a small residual-reducing coarse correction
without coupling tests to the full solver orchestration.  The basic builder uses
only layout information: block constants, species/group constants, a radial
block ramp, and first angular harmonics when the block shape permits them.

The x-block hard-seed helper adds a bounded enriched basis: block constants,
radial ramps and curvature moments, angular harmonics, species/global coupling
moments, constraint-like intra-block moments, and block-Schur-like x/species
contrasts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


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
        block_sizes = tuple(int(size) for size in self.block_sizes)
        if not block_sizes:
            raise ValueError("block_sizes must contain at least one block")
        if any(size <= 0 for size in block_sizes):
            raise ValueError("block_sizes must be positive")
        object.__setattr__(self, "block_sizes", block_sizes)
        object.__setattr__(self, "n_theta", max(1, int(self.n_theta)))
        object.__setattr__(self, "n_zeta", max(1, int(self.n_zeta)))

        n_blocks = len(block_sizes)
        if self.block_x is None:
            block_x = tuple(range(n_blocks))
        else:
            block_x = tuple(int(value) for value in self.block_x)
        if len(block_x) != n_blocks:
            raise ValueError("block_x must have the same length as block_sizes")
        object.__setattr__(self, "block_x", block_x)

        if self.block_species is None:
            block_species = tuple(0 for _ in range(n_blocks))
        else:
            block_species = tuple(int(value) for value in self.block_species)
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
        return _small_regularized_least_squares(
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


def _empty_matrix(n_rows: int, dtype: Any) -> ArrayLike:
    return jnp.zeros((int(n_rows), 0), dtype=dtype)


def _append_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike,
    label: str,
) -> None:
    columns.append(jnp.asarray(values).reshape((-1,)))
    labels.append(str(label))


def _append_candidate_if_room(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
    max_candidates: int,
) -> bool:
    if values is None or len(columns) >= int(max_candidates):
        return False
    _append_candidate(columns, labels, values, label)
    return True


def _stack_candidates(
    total_size: int,
    dtype: Any,
    columns: Sequence[ArrayLike],
    labels: Sequence[str],
) -> tuple[ArrayLike, tuple[str, ...]]:
    if not columns:
        return _empty_matrix(total_size, dtype), ()
    return jnp.stack(tuple(columns), axis=1), tuple(str(label) for label in labels)


def _block_constant(layout: RHS1QICoarseBlockLayout, block_index: int, dtype: Any) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    start = int(offsets[block_index])
    stop = int(offsets[block_index + 1])
    return values.at[start:stop].set(1.0)


def _group_constant(
    layout: RHS1QICoarseBlockLayout,
    block_ids: Sequence[int],
    dtype: Any,
) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index in block_ids:
        start = int(offsets[int(block_index)])
        stop = int(offsets[int(block_index) + 1])
        values = values.at[start:stop].set(1.0)
    return values


def _block_weighted_constant(
    layout: RHS1QICoarseBlockLayout,
    weights: Sequence[float],
    dtype: Any,
) -> ArrayLike:
    values = jnp.zeros((layout.total_size,), dtype=dtype)
    offsets = layout.block_offsets
    for block_index, weight in enumerate(weights):
        start = int(offsets[block_index])
        stop = int(offsets[block_index + 1])
        values = values.at[start:stop].set(float(weight))
    return values


def _species_to_blocks(layout: RHS1QICoarseBlockLayout) -> dict[int, list[int]]:
    species_to_blocks: dict[int, list[int]] = {}
    for block_index, species in enumerate(layout.block_species or ()):
        species_to_blocks.setdefault(int(species), []).append(block_index)
    return species_to_blocks


def _centered_unit_weights(values: Sequence[float]) -> tuple[float, ...] | None:
    weights = tuple(float(value) for value in values)
    if not weights:
        return None
    mean = sum(weights) / float(len(weights))
    centered = tuple(value - mean for value in weights)
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple(value / scale for value in centered)


def _centered_power_weights(values: Sequence[float], power: int) -> tuple[float, ...] | None:
    linear = _centered_unit_weights(values)
    if linear is None:
        return None
    powered = tuple(value ** int(power) for value in linear)
    return _centered_unit_weights(powered)


def _weights_for_blocks(
    n_blocks: int,
    weighted_blocks: Sequence[tuple[int, float]],
) -> tuple[float, ...]:
    weights = [0.0 for _ in range(int(n_blocks))]
    for block_index, value in weighted_blocks:
        weights[int(block_index)] = float(value)
    return tuple(weights)


def _angular_candidate(
    layout: RHS1QICoarseBlockLayout,
    angular_values: ArrayLike,
    dtype: Any,
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


def _block_weighted_angular_candidate(
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


def _centered_index_moment(repeats: int, power: int, dtype: Any) -> ArrayLike | None:
    repeats = int(repeats)
    if repeats <= 1:
        return None
    denominator = float(max(1, repeats - 1))
    base = tuple((2.0 * float(index) / denominator) - 1.0 for index in range(repeats))
    if int(power) == 1:
        values = base
    else:
        centered = _centered_unit_weights(tuple(value ** int(power) for value in base))
        if centered is None:
            return None
        values = centered
    return jnp.asarray(values, dtype=dtype)


def _intra_block_moment_candidate(
    layout: RHS1QICoarseBlockLayout,
    weights: Sequence[float],
    *,
    power: int,
    dtype: Any,
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
        moment = _centered_index_moment(repeats, int(power), dtype)
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


def _angular_harmonic_specs(
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
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
            specs.append((f"theta_cos{mode}", jnp.cos(theta_phase)))
            specs.append((f"theta_sin{mode}", jnp.sin(theta_phase)))
        if int(layout.n_zeta) > 1:
            zeta_phase = 2.0 * jnp.pi * float(mode) * zeta_grid / float(layout.n_zeta)
            specs.append((f"zeta_cos{mode}", jnp.cos(zeta_phase)))
            specs.append((f"zeta_sin{mode}", jnp.sin(zeta_phase)))
        if bool(include_mixed) and int(layout.n_theta) > 1 and int(layout.n_zeta) > 1:
            theta_phase = 2.0 * jnp.pi * float(mode) * theta_grid / float(layout.n_theta)
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
        _append_candidate(columns, labels, jnp.ones((layout.total_size,), dtype=dtype), "global")

    if include_species and layout.block_species is not None:
        species_to_blocks: dict[int, list[int]] = {}
        for block_index, species in enumerate(layout.block_species):
            species_to_blocks.setdefault(int(species), []).append(block_index)
        if len(species_to_blocks) > 1:
            for species in sorted(species_to_blocks):
                _append_candidate(
                    columns,
                    labels,
                    _group_constant(layout, species_to_blocks[species], dtype),
                    f"species:{species}",
                )

    if include_x_ramp and layout.block_x is not None:
        block_x = jnp.asarray(layout.block_x, dtype=dtype)
        centered = block_x - jnp.mean(block_x)
        if bool(jnp.max(jnp.abs(centered)) > 0):
            _append_candidate(
                columns,
                labels,
                _block_weighted_constant(layout, tuple(float(v) for v in centered), dtype),
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
            candidate = _angular_candidate(layout, angular_values, dtype)
            if candidate is not None:
                _append_candidate(columns, labels, candidate, label)

    if include_blocks:
        for block_index in range(len(layout.block_sizes)):
            _append_candidate(
                columns,
                labels,
                _block_constant(layout, block_index, dtype),
                f"block:{block_index}",
            )

    if not columns:
        return _empty_matrix(layout.total_size, dtype), ()
    return jnp.stack(columns, axis=1), tuple(labels)


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
        return _stack_candidates(layout.total_size, dtype, columns, labels)

    def add(values: ArrayLike | None, label: str) -> None:
        _append_candidate_if_room(columns, labels, values, label, candidate_limit)

    if candidate_limit <= 0:
        return _empty_matrix(layout.total_size, dtype), ()

    species_to_blocks = _species_to_blocks(layout)
    block_x_values = tuple(float(value) for value in (layout.block_x or ()))
    x_ramp_weights = _centered_unit_weights(block_x_values)
    x_quad_weights = _centered_power_weights(block_x_values, 2)

    if include_global:
        add(jnp.ones((layout.total_size,), dtype=dtype), "global")
    if not has_room():
        return finish()

    if include_species and len(species_to_blocks) > 1:
        for species in sorted(species_to_blocks):
            if not has_room():
                return finish()
            add(_group_constant(layout, species_to_blocks[species], dtype), f"species:{species}")

    if include_radial:
        if x_ramp_weights is not None:
            if not has_room():
                return finish()
            add(_block_weighted_constant(layout, x_ramp_weights, dtype), "radial:x_ramp")
        if x_quad_weights is not None:
            if not has_room():
                return finish()
            add(_block_weighted_constant(layout, x_quad_weights, dtype), "radial:x_quad")
        for species in sorted(species_to_blocks):
            block_ids = species_to_blocks[species]
            local_x = tuple(float(layout.block_x[block_index]) for block_index in block_ids)
            local_ramp = _centered_unit_weights(local_x)
            local_quad = _centered_power_weights(local_x, 2)
            if local_ramp is not None:
                if not has_room():
                    return finish()
                weights = _weights_for_blocks(n_blocks, tuple(zip(block_ids, local_ramp, strict=True)))
                add(_block_weighted_constant(layout, weights, dtype), f"radial:species:{species}:x_ramp")
            if local_quad is not None:
                if not has_room():
                    return finish()
                weights = _weights_for_blocks(n_blocks, tuple(zip(block_ids, local_quad, strict=True)))
                add(_block_weighted_constant(layout, weights, dtype), f"radial:species:{species}:x_quad")

    if include_constraint_moments:
        ones = tuple(1.0 for _ in range(n_blocks))
        if not has_room():
            return finish()
        add(_intra_block_moment_candidate(layout, ones, power=1, dtype=dtype), "constraint:xi_ramp")
        if not has_room():
            return finish()
        add(_intra_block_moment_candidate(layout, ones, power=2, dtype=dtype), "constraint:xi_quad")
        if x_ramp_weights is not None:
            if not has_room():
                return finish()
            add(
                _intra_block_moment_candidate(layout, x_ramp_weights, power=1, dtype=dtype),
                "constraint:radial_x_ramp*xi_ramp",
            )
        for species in sorted(species_to_blocks):
            if not has_room():
                return finish()
            weights = _weights_for_blocks(
                n_blocks,
                tuple((block_index, 1.0) for block_index in species_to_blocks[species]),
            )
            add(
                _intra_block_moment_candidate(layout, weights, power=1, dtype=dtype),
                f"constraint:species:{species}:xi_ramp",
            )

    angular_specs: tuple[tuple[str, ArrayLike], ...] = ()
    if has_room() and (include_angular or include_radial_angular):
        angular_specs = _angular_harmonic_specs(
            layout,
            max_angular_mode=max_angular_mode,
            include_mixed=True,
            dtype=dtype,
        )
    if include_angular:
        for label, angular_values in angular_specs:
            if not has_room():
                return finish()
            add(_angular_candidate(layout, angular_values, dtype), label)

    if include_radial_angular and x_ramp_weights is not None:
        for label, angular_values in angular_specs:
            if not has_room():
                return finish()
            add(
                _block_weighted_angular_candidate(layout, angular_values, x_ramp_weights, dtype),
                f"radial:x_ramp*{label}",
            )
        if x_quad_weights is not None:
            for label, angular_values in angular_specs:
                if not has_room():
                    return finish()
                add(
                    _block_weighted_angular_candidate(layout, angular_values, x_quad_weights, dtype),
                    f"radial:x_quad*{label}",
                )

    if include_schur:
        for species in sorted(species_to_blocks):
            x_to_blocks: dict[int, list[int]] = {}
            for block_index in species_to_blocks[species]:
                x_to_blocks.setdefault(int(layout.block_x[block_index]), []).append(block_index)
            x_values = sorted(x_to_blocks)
            for left_x, right_x in zip(x_values, x_values[1:], strict=False):
                if not has_room():
                    return finish()
                weighted_blocks = tuple((block_index, -1.0) for block_index in x_to_blocks[left_x]) + tuple(
                    (block_index, 1.0) for block_index in x_to_blocks[right_x]
                )
                add(
                    _block_weighted_constant(layout, _weights_for_blocks(n_blocks, weighted_blocks), dtype),
                    f"schur:x_diff:s{species}:{left_x}->{right_x}",
                )
            for left_x, center_x, right_x in zip(x_values, x_values[1:], x_values[2:], strict=False):
                if not has_room():
                    return finish()
                weighted_blocks = (
                    tuple((block_index, 1.0) for block_index in x_to_blocks[left_x])
                    + tuple((block_index, -2.0) for block_index in x_to_blocks[center_x])
                    + tuple((block_index, 1.0) for block_index in x_to_blocks[right_x])
                )
                add(
                    _block_weighted_constant(layout, _weights_for_blocks(n_blocks, weighted_blocks), dtype),
                    f"schur:x_curve:s{species}:{left_x},{center_x},{right_x}",
                )

        x_species_blocks: dict[int, dict[int, list[int]]] = {}
        for block_index, (x_value, species) in enumerate(zip(layout.block_x, layout.block_species, strict=True)):
            x_species_blocks.setdefault(int(x_value), {}).setdefault(int(species), []).append(block_index)
        for x_value in sorted(x_species_blocks):
            species_values = sorted(x_species_blocks[x_value])
            for left_species, right_species in zip(species_values, species_values[1:], strict=False):
                if not has_room():
                    return finish()
                weighted_blocks = tuple(
                    (block_index, -1.0) for block_index in x_species_blocks[x_value][left_species]
                ) + tuple((block_index, 1.0) for block_index in x_species_blocks[x_value][right_species])
                add(
                    _block_weighted_constant(layout, _weights_for_blocks(n_blocks, weighted_blocks), dtype),
                    f"schur:species_diff:x{x_value}:s{left_species}->{right_species}",
                )

    if include_blocks:
        for block_index in range(n_blocks):
            if not has_room():
                return finish()
            add(_block_constant(layout, block_index, dtype), f"block:{block_index}")

    return finish()


def orthonormalize_rhs1_qi_coarse_basis(
    candidates: ArrayLike,
    *,
    labels: Sequence[str] | None = None,
    rtol: float = 1.0e-10,
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
    candidate_labels = tuple(str(label) for label in (labels or tuple(f"candidate:{i}" for i in range(n_cols))))
    if len(candidate_labels) != n_cols:
        raise ValueError("labels must match the number of candidate columns")

    candidate_norms = tuple(float(jnp.linalg.norm(matrix[:, i])) for i in range(n_cols))
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
        vectors = _empty_matrix(n_rows, matrix.dtype)

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
    rtol: float = 1.0e-10,
    atol: float = 0.0,
    max_rank: int | None = None,
    dtype: Any = jnp.float64,
    **candidate_options: Any,
) -> RHS1QICoarseBasis:
    """Build and rank-gate a QI coarse basis from a block layout."""

    candidates, labels = build_rhs1_qi_coarse_candidates(layout, dtype=dtype, **candidate_options)
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=rtol,
        atol=atol,
        max_rank=max_rank,
    )


def build_rhs1_qi_xblock_hard_seed_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    rtol: float = 1.0e-10,
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
        layout,
        max_candidates=max_candidates,
        dtype=dtype,
        **candidate_options,
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=rtol,
        atol=atol,
        max_rank=max(0, int(max_rank)),
    )


def _apply_operator(operator: LinearOperator, vector: ArrayLike) -> ArrayLike:
    if callable(operator):
        return jnp.asarray(operator(vector))
    return jnp.asarray(operator) @ vector


def _apply_operator_to_basis(operator: LinearOperator, basis_vectors: ArrayLike) -> ArrayLike:
    if not callable(operator):
        return jnp.asarray(operator) @ basis_vectors
    columns = [_apply_operator(operator, basis_vectors[:, i]) for i in range(int(basis_vectors.shape[1]))]
    if not columns:
        return _empty_matrix(int(basis_vectors.shape[0]), basis_vectors.dtype)
    return jnp.stack(columns, axis=1)


def _small_regularized_least_squares(
    matrix: ArrayLike,
    rhs: ArrayLike,
    *,
    rcond: float | None = None,
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
    rcond_value = 1.0e-14 if rcond is None else max(0.0, float(rcond))
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
        aq = _apply_operator_to_basis(operator, q)
        coarse_operator = jnp.conjugate(q).T @ aq

    coarse_norm = float(jnp.linalg.norm(coarse_operator)) if rank > 0 else 0.0
    regularization_rcond = 1.0e-14 if rcond is None else max(0.0, float(rcond))
    metadata = RHS1QIGalerkinPreconditionerMetadata(
        rank=rank,
        coarse_operator_shape=tuple(int(v) for v in coarse_operator.shape),
        coarse_operator_norm=coarse_norm,
        regularization_rcond=float(regularization_rcond),
        basis_metadata=basis.metadata,
    )
    return RHS1QIGalerkinPreconditioner(
        basis=basis,
        coarse_operator=coarse_operator,
        metadata=metadata,
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
            operator,
            basis=basis,
            layout=layout,
            rcond=rcond,
        )

    residual_before = rhs_vec - _apply_operator(operator, current_vec)
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
    residual_after = rhs_vec - _apply_operator(operator, candidate_solution)
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm

    required_drop = max(float(acceptance_atol), residual_before_norm * max(0.0, float(min_relative_improvement)))
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

    residual_before = rhs_vec - _apply_operator(operator, current_vec)
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

    coarse_operator = _apply_operator_to_basis(operator, basis.vectors)
    coefficients = _small_regularized_least_squares(coarse_operator, residual_before, rcond=rcond)
    correction = float(damping) * (basis.vectors @ coefficients)
    candidate_solution = current_vec + correction
    residual_after = rhs_vec - _apply_operator(operator, candidate_solution)
    residual_after_norm = float(jnp.linalg.norm(residual_after))
    improvement_ratio = residual_after_norm / residual_before_norm

    required_drop = max(float(acceptance_atol), residual_before_norm * max(0.0, float(min_relative_improvement)))
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


__all__ = [
    "RHS1QICoarseBasis",
    "RHS1QICoarseBasisMetadata",
    "RHS1QICoarseBlockLayout",
    "RHS1QICoarseCorrection",
    "RHS1QIGalerkinPreconditioner",
    "RHS1QIGalerkinPreconditionerMetadata",
    "apply_rhs1_qi_galerkin_correction",
    "apply_rhs1_qi_coarse_correction",
    "build_rhs1_qi_galerkin_preconditioner",
    "build_rhs1_qi_coarse_basis",
    "build_rhs1_qi_coarse_candidates",
    "build_rhs1_qi_xblock_hard_seed_basis",
    "build_rhs1_qi_xblock_hard_seed_candidates",
    "orthonormalize_rhs1_qi_coarse_basis",
]
