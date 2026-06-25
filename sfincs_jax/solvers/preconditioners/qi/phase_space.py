"""Phase-space coarse basis directions for RHSMode=1 QI residuals.

This module is intentionally standalone.  It derives a small deterministic
candidate space from ``RHS1QICoarseBlockLayout`` metadata only, then returns the
standard ``RHS1QICoarseBasis`` used by the existing QI coarse helpers.  The
directions are pitch-space moments inside each block: trapped/passing-like
bands, boundary bands, even/odd parity components, and optional radial/species
aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.qi.coarse import (
    RHS1QICoarseBasis,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)


ArrayLike = Any


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
    rtol: float = 1.0e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


def _empty_candidates(layout: RHS1QICoarseBlockLayout, dtype: Any) -> tuple[ArrayLike, tuple[str, ...]]:
    return jnp.zeros((layout.total_size, 0), dtype=dtype), ()


def _centered_unit_weights(values: tuple[float, ...]) -> tuple[float, ...] | None:
    if not values:
        return None
    mean = sum(values) / float(len(values))
    centered = tuple(value - mean for value in values)
    scale = max((abs(value) for value in centered), default=0.0)
    if scale <= 0.0:
        return None
    return tuple(value / scale for value in centered)


def _pitch_grid(size: int) -> np.ndarray:
    if int(size) <= 1:
        return np.zeros((int(size),), dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(size), dtype=np.float64)


def _boundary_mask(pitch: np.ndarray, boundary: float) -> np.ndarray:
    if pitch.size <= 1:
        return np.ones_like(pitch)
    spacing = 2.0 / float(max(1, pitch.size - 1))
    width = max(0.5 * spacing, spacing * 0.25)
    mask = np.abs(np.abs(pitch) - boundary) <= width
    if not bool(np.any(mask)):
        mask[int(np.argmin(np.abs(np.abs(pitch) - boundary)))] = True
    return mask.astype(np.float64)


def _candidate_from_pitch_weights(
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
        pitch = _pitch_grid(n_pitch)
        weights = np.asarray(pitch_weight(pitch), dtype=np.float64).reshape((-1,))
        if int(weights.size) != n_pitch:
            raise ValueError("pitch_weight must return one value per inferred pitch index")
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


def _append_candidate(
    columns: list[ArrayLike],
    labels: list[str],
    values: ArrayLike | None,
    label: str,
) -> None:
    if values is None:
        return
    vector = jnp.asarray(values).reshape((-1,))
    norm = float(jnp.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 0.0:
        return
    columns.append(vector)
    labels.append(f"phase_space:{label}")


def _all_block_weights(layout: RHS1QICoarseBlockLayout) -> tuple[float, ...]:
    return tuple(1.0 if _is_physical_block(layout, index) else 0.0 for index in range(len(layout.block_sizes)))


def _is_physical_block(layout: RHS1QICoarseBlockLayout, block_index: int) -> bool:
    block_x = tuple(int(value) for value in (layout.block_x or ()))
    block_species = tuple(int(value) for value in (layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and int(block_x[int(block_index)]) < 0:
        return False
    if len(block_species) == len(layout.block_sizes) and int(block_species[int(block_index)]) < 0:
        return False
    return True


def _species_weights(layout: RHS1QICoarseBlockLayout) -> tuple[tuple[int, tuple[float, ...]], ...]:
    species_values = tuple(int(value) for value in (layout.block_species or ()))
    if len(species_values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, tuple[float, ...]]] = []
    for species in sorted(set(species_values)):
        if species < 0:
            continue
        weights = tuple(1.0 if value == species else 0.0 for value in species_values)
        result.append((species, weights))
    return tuple(result)


def _block_x_ramp(layout: RHS1QICoarseBlockLayout) -> tuple[float, ...] | None:
    block_x = tuple(float(value) for value in (layout.block_x or ()))
    if len(block_x) != len(layout.block_sizes):
        return None
    active = tuple(index for index in range(len(layout.block_sizes)) if _is_physical_block(layout, index))
    if len(active) <= 1:
        return None
    active_weights = _centered_unit_weights(tuple(block_x[index] for index in active))
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
        return _empty_candidates(layout, dtype)
    active = tuple(index for index in range(len(layout.block_sizes)) if _is_physical_block(layout, index))
    if not active or any(int(layout.block_sizes[index]) % n_angular != 0 for index in active):
        return _empty_candidates(layout, dtype)

    boundary = max(0.0, min(1.0, float(cfg.trapped_boundary_fraction)))
    columns: list[ArrayLike] = []
    labels: list[str] = []
    all_blocks = _all_block_weights(layout)

    def trapped(pitch: np.ndarray) -> np.ndarray:
        mask = np.abs(pitch) <= boundary
        if pitch.size and not bool(np.any(mask)):
            mask[int(np.argmin(np.abs(pitch)))] = True
        return mask.astype(np.float64)

    def passing(pitch: np.ndarray) -> np.ndarray:
        return (np.abs(pitch) > boundary).astype(np.float64)

    def passing_sign(pitch: np.ndarray) -> np.ndarray:
        return passing(pitch) * np.sign(pitch)

    def boundary_band(pitch: np.ndarray) -> np.ndarray:
        return _boundary_mask(pitch, boundary)

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
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=trapped,
                dtype=dtype,
            ),
            "pitch_band:trapped",
        )
    if cfg.include_passing:
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=passing,
                dtype=dtype,
            ),
            "pitch_band:passing",
        )
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=passing_sign,
                dtype=dtype,
            ),
            "pitch_band:passing*sign",
        )
    if cfg.include_boundary:
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=boundary_band,
                dtype=dtype,
            ),
            "pitch_band:boundary",
        )
    if cfg.include_even:
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=even_abs,
                dtype=dtype,
            ),
            "pitch:even_abs",
        )
    if cfg.include_odd:
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=odd_pitch,
                dtype=dtype,
            ),
            "pitch:odd",
        )
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=all_blocks,
                pitch_weight=odd_sign,
                dtype=dtype,
            ),
            "pitch:sign",
        )

    radial = _block_x_ramp(layout) if cfg.include_radial else None
    if radial is not None:
        _append_candidate(
            columns,
            labels,
            _candidate_from_pitch_weights(
                layout,
                block_weights=radial,
                pitch_weight=lambda pitch: np.ones_like(pitch),
                dtype=dtype,
            ),
            "radial:ramp",
        )
        if cfg.include_trapped:
            _append_candidate(
                columns,
                labels,
                _candidate_from_pitch_weights(
                    layout,
                    block_weights=radial,
                    pitch_weight=trapped,
                    dtype=dtype,
                ),
                "radial:ramp*pitch_band:trapped",
            )
        if cfg.include_passing:
            _append_candidate(
                columns,
                labels,
                _candidate_from_pitch_weights(
                    layout,
                    block_weights=radial,
                    pitch_weight=passing,
                    dtype=dtype,
                ),
                "radial:ramp*pitch_band:passing",
            )

    if cfg.include_species:
        for species, species_block_weights in _species_weights(layout):
            _append_candidate(
                columns,
                labels,
                _candidate_from_pitch_weights(
                    layout,
                    block_weights=species_block_weights,
                    pitch_weight=lambda pitch: np.ones_like(pitch),
                    dtype=dtype,
                ),
                f"species:{species}",
            )
            if cfg.include_trapped:
                _append_candidate(
                    columns,
                    labels,
                    _candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=trapped,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch_band:trapped",
                )
            if cfg.include_passing:
                _append_candidate(
                    columns,
                    labels,
                    _candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=passing,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch_band:passing",
                )
            if cfg.include_odd:
                _append_candidate(
                    columns,
                    labels,
                    _candidate_from_pitch_weights(
                        layout,
                        block_weights=species_block_weights,
                        pitch_weight=odd_sign,
                        dtype=dtype,
                    ),
                    f"species:{species}*pitch:sign",
                )

    if not columns:
        return _empty_candidates(layout, dtype)
    return jnp.stack(tuple(columns), axis=1), tuple(labels)


def build_rhs1_qi_phase_space_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    *,
    config: RHS1QIPhaseSpaceCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated QI coarse basis from phase-space directions."""

    cfg = RHS1QIPhaseSpaceCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_phase_space_coarse_candidates(
        layout,
        config=cfg,
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


__all__ = [
    "RHS1QIPhaseSpaceCoarseConfig",
    "build_rhs1_qi_phase_space_coarse_basis",
    "build_rhs1_qi_phase_space_coarse_candidates",
]
