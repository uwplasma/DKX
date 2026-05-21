"""Residual active-pattern coarse directions for RHSMode=1 QI lanes.

This primitive builds deterministic chunk-local coarse directions from an
``RHS1QICoarseBlockLayout`` and a residual seed.  It infers pitch/angular
chunks from the block size and ``n_theta * n_zeta`` metadata, uses ``block_x``
and ``block_species`` labels for radial/species aggregate chunks, selects the
highest-energy active patterns, then rank-gates them with the existing QI
coarse-basis orthonormalizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp
import numpy as np

from .rhs1_qi_coarse import (
    RHS1QICoarseBasis,
    RHS1QICoarseBlockLayout,
    orthonormalize_rhs1_qi_coarse_basis,
)


ArrayLike = Any


@dataclass(frozen=True)
class RHS1QIActivePatternCoarseConfig:
    """Controls for residual active-pattern coarse directions."""

    max_rank: int = 16
    max_candidates: int = 48
    min_chunk_energy_fraction: float = 1.0e-2
    include_block_pitch_chunks: bool = True
    include_block_angular_chunks: bool = True
    include_radial_pitch_chunks: bool = True
    include_radial_angular_chunks: bool = True
    include_block_chunks: bool = True
    include_radial_chunks: bool = True
    include_species_chunks: bool = True
    rtol: float = 1.0e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class _ActivePatternCandidate:
    label: str
    indices: np.ndarray
    energy: float
    order: int


def _empty_candidates(layout: RHS1QICoarseBlockLayout, dtype: Any) -> tuple[ArrayLike, tuple[str, ...]]:
    return jnp.zeros((layout.total_size, 0), dtype=dtype), ()


def _is_physical_block(layout: RHS1QICoarseBlockLayout, block_index: int) -> bool:
    block_x = tuple(int(value) for value in (layout.block_x or ()))
    block_species = tuple(int(value) for value in (layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and int(block_x[int(block_index)]) < 0:
        return False
    if len(block_species) == len(layout.block_sizes) and int(block_species[int(block_index)]) < 0:
        return False
    return True


def _active_blocks(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    return tuple(index for index in range(len(layout.block_sizes)) if _is_physical_block(layout, index))


def _layout_is_supported(layout: RHS1QICoarseBlockLayout, active: tuple[int, ...]) -> bool:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return False
    return all(int(layout.block_sizes[index]) % n_angular == 0 for index in active)


def _block_indices(layout: RHS1QICoarseBlockLayout, block_index: int) -> np.ndarray:
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    return np.arange(start, stop, dtype=np.int64)


def _block_pitch_indices(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    pitch_index: int,
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    start = int(layout.block_offsets[int(block_index)]) + int(pitch_index) * n_angular
    return np.arange(start, start + n_angular, dtype=np.int64)


def _block_angular_indices(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    angular_index: int,
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    block_size = int(layout.block_sizes[int(block_index)])
    n_pitch = block_size // n_angular
    start = int(layout.block_offsets[int(block_index)]) + int(angular_index)
    return start + n_angular * np.arange(n_pitch, dtype=np.int64)


def _concat_indices(parts: tuple[np.ndarray, ...]) -> np.ndarray:
    nonempty = tuple(np.asarray(part, dtype=np.int64).reshape((-1,)) for part in parts if part.size)
    if not nonempty:
        return np.zeros((0,), dtype=np.int64)
    return np.concatenate(nonempty).astype(np.int64, copy=False)


def _group_blocks_by_value(
    layout: RHS1QICoarseBlockLayout,
    active: tuple[int, ...],
    values: tuple[int, ...],
) -> tuple[tuple[int, tuple[int, ...]], ...]:
    if len(values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, tuple[int, ...]]] = []
    for value in sorted({int(values[index]) for index in active if int(values[index]) >= 0}):
        blocks = tuple(index for index in active if int(values[index]) == value)
        if blocks:
            result.append((value, blocks))
    return tuple(result)


def _angular_label(layout: RHS1QICoarseBlockLayout, angular_index: int) -> str:
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


def _energy(residual: np.ndarray, indices: np.ndarray) -> float:
    if indices.size == 0:
        return 0.0
    values = residual[np.asarray(indices, dtype=np.int64)]
    return float(np.vdot(values, values).real)


def _append_candidate(
    candidates: list[_ActivePatternCandidate],
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
    energy = _energy(residual, chunk_indices)
    if not np.isfinite(energy) or energy <= min_energy:
        return
    candidates.append(
        _ActivePatternCandidate(
            label=f"active_pattern:{label}",
            indices=chunk_indices,
            energy=energy,
            order=len(seen_indices) - 1,
        )
    )


def _build_active_patterns(
    layout: RHS1QICoarseBlockLayout,
    residual: np.ndarray,
    *,
    cfg: RHS1QIActivePatternCoarseConfig,
) -> tuple[_ActivePatternCandidate, ...]:
    active = _active_blocks(layout)
    if not active or not _layout_is_supported(layout, active):
        return ()

    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    block_n_pitch = {
        block_index: int(layout.block_sizes[block_index]) // n_angular for block_index in active
    }
    active_indices = _concat_indices(tuple(_block_indices(layout, index) for index in active))
    active_energy = _energy(residual, active_indices)
    if not np.isfinite(active_energy) or active_energy <= 0.0:
        return ()

    min_energy = active_energy * max(0.0, float(cfg.min_chunk_energy_fraction))
    candidates: list[_ActivePatternCandidate] = []
    seen_indices: set[bytes] = set()

    def add(indices: np.ndarray, label: str) -> None:
        _append_candidate(candidates, seen_indices, residual, indices, label, min_energy=min_energy)

    if cfg.include_block_pitch_chunks:
        for block_index in active:
            for pitch_index in range(block_n_pitch[block_index]):
                add(
                    _block_pitch_indices(layout, block_index, pitch_index),
                    f"block:{block_index}*pitch:{pitch_index}",
                )

    if cfg.include_block_angular_chunks:
        for block_index in active:
            for angular_index in range(n_angular):
                add(
                    _block_angular_indices(layout, block_index, angular_index),
                    f"block:{block_index}*angular:{_angular_label(layout, angular_index)}",
                )

    block_x = tuple(int(value) for value in (layout.block_x or ()))
    radial_groups = _group_blocks_by_value(layout, active, block_x)

    if cfg.include_radial_pitch_chunks:
        for radial_value, blocks in radial_groups:
            max_pitch = max(block_n_pitch[block_index] for block_index in blocks)
            for pitch_index in range(max_pitch):
                add(
                    _concat_indices(
                        tuple(
                            _block_pitch_indices(layout, block_index, pitch_index)
                            for block_index in blocks
                            if pitch_index < block_n_pitch[block_index]
                        )
                    ),
                    f"radial:{radial_value}*pitch:{pitch_index}",
                )

    if cfg.include_radial_angular_chunks:
        for radial_value, blocks in radial_groups:
            for angular_index in range(n_angular):
                add(
                    _concat_indices(
                        tuple(
                            _block_angular_indices(layout, block_index, angular_index)
                            for block_index in blocks
                        )
                    ),
                    f"radial:{radial_value}*angular:{_angular_label(layout, angular_index)}",
                )

    if cfg.include_block_chunks:
        for block_index in active:
            add(_block_indices(layout, block_index), f"block:{block_index}")

    if cfg.include_radial_chunks:
        for radial_value, blocks in radial_groups:
            add(
                _concat_indices(tuple(_block_indices(layout, block_index) for block_index in blocks)),
                f"radial:{radial_value}",
            )

    if cfg.include_species_chunks:
        block_species = tuple(int(value) for value in (layout.block_species or ()))
        for species, blocks in _group_blocks_by_value(layout, active, block_species):
            add(
                _concat_indices(tuple(_block_indices(layout, block_index) for block_index in blocks)),
                f"species:{species}",
            )

    candidates.sort(key=lambda candidate: (-candidate.energy, candidate.order))
    return tuple(candidates[: max(0, int(cfg.max_candidates))])


def _candidate_from_indices(residual: ArrayLike, indices: np.ndarray) -> ArrayLike:
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
        return _empty_candidates(layout, cfg.dtype)

    patterns = _build_active_patterns(layout, residual_host, cfg=cfg)
    if not patterns:
        return _empty_candidates(layout, cfg.dtype)

    columns = [_candidate_from_indices(residual_vec, pattern.indices) for pattern in patterns]
    labels = tuple(pattern.label for pattern in patterns)
    return jnp.stack(tuple(columns), axis=1), labels


def build_rhs1_qi_active_pattern_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    residual_seed: ArrayLike,
    *,
    config: RHS1QIActivePatternCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated residual active-pattern QI coarse basis."""

    cfg = RHS1QIActivePatternCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_active_pattern_coarse_candidates(
        layout,
        residual_seed,
        config=cfg,
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


def project_rhs1_qi_active_pattern_correction(
    residual: ArrayLike,
    basis: RHS1QICoarseBasis,
) -> ArrayLike:
    """Project ``residual`` into an active-pattern coarse basis."""

    residual_vec = jnp.asarray(residual, dtype=basis.vectors.dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return jnp.zeros_like(residual_vec)
    return basis.vectors @ (jnp.conjugate(basis.vectors).T @ residual_vec)


__all__ = [
    "RHS1QIActivePatternCoarseConfig",
    "build_rhs1_qi_active_pattern_coarse_basis",
    "build_rhs1_qi_active_pattern_coarse_candidates",
    "project_rhs1_qi_active_pattern_correction",
]
