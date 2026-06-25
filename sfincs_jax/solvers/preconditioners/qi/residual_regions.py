"""Residual-region coarse directions for RHSMode=1 QI hard seeds.

This standalone primitive builds deterministic coarse vectors from an
``RHS1QICoarseBlockLayout`` and the remaining operator residual. It selects
active bounce/pitch, radial, species, and block regions by residual energy,
then rank-gates the residual-restricted candidate vectors with the existing QI
coarse-basis orthonormalizer.
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
class RHS1QIResidualRegionCoarseConfig:
    """Controls for residual-localized region coarse directions."""

    max_rank: int = 16
    max_candidates: int = 48
    trapped_boundary_fraction: float = 0.35
    min_region_energy_fraction: float = 1.0e-2
    include_global_active_region: bool = False
    include_block_regions: bool = True
    include_block_bounce_regions: bool = True
    include_pitch_regions: bool = True
    include_radial_regions: bool = True
    include_radial_bounce_regions: bool = True
    include_species_regions: bool = True
    include_species_bounce_regions: bool = True
    region_bands: str = "bounce,trapped,passing"
    rtol: float = 1.0e-10
    atol: float = 0.0
    dtype: Any = jnp.float64


@dataclass(frozen=True)
class _RegionCandidate:
    label: str
    mask: np.ndarray
    energy: float
    order: int


def _empty_candidates(layout: RHS1QICoarseBlockLayout, dtype: Any) -> tuple[ArrayLike, tuple[str, ...]]:
    return jnp.zeros((layout.total_size, 0), dtype=dtype), ()


def _is_physical_block(layout: RHS1QICoarseBlockLayout, block_index: int) -> bool:
    block_x = tuple(int(value) for value in (layout.block_x or ()))
    block_species = tuple(int(value) for value in (layout.block_species or ()))
    if len(block_x) == len(layout.block_sizes) and block_x[int(block_index)] < 0:
        return False
    if len(block_species) == len(layout.block_sizes) and block_species[int(block_index)] < 0:
        return False
    return True


def _active_blocks(layout: RHS1QICoarseBlockLayout) -> tuple[int, ...]:
    return tuple(index for index in range(len(layout.block_sizes)) if _is_physical_block(layout, index))


def _layout_is_supported(layout: RHS1QICoarseBlockLayout, active: tuple[int, ...]) -> bool:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    if n_angular <= 0:
        return False
    return all(int(layout.block_sizes[index]) % n_angular == 0 for index in active)


def _pitch_grid(size: int) -> np.ndarray:
    if int(size) <= 1:
        return np.zeros((int(size),), dtype=np.float64)
    return np.linspace(-1.0, 1.0, int(size), dtype=np.float64)


def _boundary_mask(pitch: np.ndarray, boundary: float) -> np.ndarray:
    if pitch.size <= 1:
        return np.ones_like(pitch, dtype=bool)
    spacing = 2.0 / float(max(1, pitch.size - 1))
    width = max(0.5 * spacing, 0.25 * spacing)
    mask = np.abs(np.abs(pitch) - boundary) <= width
    if not bool(np.any(mask)):
        mask[int(np.argmin(np.abs(np.abs(pitch) - boundary)))] = True
    return mask.astype(bool)


def _bounce_band_masks(n_pitch: int, boundary: float) -> tuple[tuple[str, np.ndarray], ...]:
    pitch = _pitch_grid(n_pitch)
    trapped = np.abs(pitch) <= boundary
    if pitch.size and not bool(np.any(trapped)):
        trapped[int(np.argmin(np.abs(pitch)))] = True
    passing_negative = pitch < -boundary
    passing_positive = pitch > boundary
    return (
        ("trapped", trapped.astype(bool)),
        ("boundary", _boundary_mask(pitch, boundary)),
        ("passing_negative", passing_negative.astype(bool)),
        ("passing_positive", passing_positive.astype(bool)),
    )


def _selected_bounce_labels(region_bands: object) -> tuple[str, ...]:
    """Normalize public band controls to internal trapped/boundary/passing masks."""

    tokens = tuple(
        token.strip().lower().replace("-", "_")
        for token in str(region_bands or "").split(",")
        if token.strip()
    )
    if not tokens:
        tokens = ("bounce", "trapped", "passing")

    selected: list[str] = []

    def add(label: str) -> None:
        if label not in selected:
            selected.append(label)

    for token in tokens:
        if token in {"all", "default", "pitch"}:
            for label in ("trapped", "boundary", "passing_negative", "passing_positive"):
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

    return tuple(selected) or ("trapped", "boundary", "passing_negative", "passing_positive")


def _block_mask(layout: RHS1QICoarseBlockLayout, block_index: int) -> np.ndarray:
    mask = np.zeros((layout.total_size,), dtype=bool)
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    mask[start:stop] = True
    return mask


def _block_bounce_mask(
    layout: RHS1QICoarseBlockLayout,
    block_index: int,
    pitch_mask: np.ndarray,
) -> np.ndarray:
    n_angular = int(layout.n_theta) * int(layout.n_zeta)
    mask = np.zeros((layout.total_size,), dtype=bool)
    offsets = layout.block_offsets
    start = int(offsets[int(block_index)])
    stop = int(offsets[int(block_index) + 1])
    mask[start:stop] = np.repeat(np.asarray(pitch_mask, dtype=bool), n_angular)
    return mask


def _union_masks(masks: tuple[np.ndarray, ...]) -> np.ndarray:
    if not masks:
        return np.zeros((0,), dtype=bool)
    result = np.zeros_like(masks[0], dtype=bool)
    for mask in masks:
        result = np.logical_or(result, mask)
    return result


def _energy(residual: np.ndarray, mask: np.ndarray) -> float:
    values = residual[np.asarray(mask, dtype=bool)]
    if values.size == 0:
        return 0.0
    return float(np.vdot(values, values).real)


def _append_region(
    regions: list[_RegionCandidate],
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
    energy = _energy(residual, mask)
    if not np.isfinite(energy) or energy <= min_energy:
        return
    regions.append(
        _RegionCandidate(
            label=f"residual_region:{label}",
            mask=np.asarray(mask, dtype=bool),
            energy=energy,
            order=len(seen_masks) - 1,
        )
    )


def _group_masks_by_value(
    layout: RHS1QICoarseBlockLayout,
    active: tuple[int, ...],
    values: tuple[int, ...],
) -> tuple[tuple[int, np.ndarray], ...]:
    if len(values) != len(layout.block_sizes):
        return ()
    result: list[tuple[int, np.ndarray]] = []
    for value in sorted({int(values[index]) for index in active if int(values[index]) >= 0}):
        masks = tuple(_block_mask(layout, index) for index in active if int(values[index]) == value)
        if masks:
            result.append((value, _union_masks(masks)))
    return tuple(result)


def _build_regions(
    layout: RHS1QICoarseBlockLayout,
    residual: np.ndarray,
    *,
    cfg: RHS1QIResidualRegionCoarseConfig,
) -> tuple[_RegionCandidate, ...]:
    active = _active_blocks(layout)
    if not active or not _layout_is_supported(layout, active):
        return ()

    active_mask = _union_masks(tuple(_block_mask(layout, index) for index in active))
    active_energy = _energy(residual, active_mask)
    if not np.isfinite(active_energy) or active_energy <= 0.0:
        return ()

    min_fraction = max(0.0, float(cfg.min_region_energy_fraction))
    min_energy = active_energy * min_fraction
    boundary = max(0.0, min(1.0, float(cfg.trapped_boundary_fraction)))
    regions: list[_RegionCandidate] = []
    seen_masks: set[bytes] = set()
    selected_bounce_labels = _selected_bounce_labels(cfg.region_bands)

    def add(mask: np.ndarray, label: str) -> None:
        _append_region(regions, seen_masks, residual, mask, label, min_energy=min_energy)

    block_bounce_masks: dict[tuple[int, str], np.ndarray] = {}
    for block_index in active:
        n_angular = int(layout.n_theta) * int(layout.n_zeta)
        n_pitch = int(layout.block_sizes[block_index]) // n_angular
        for bounce_label, pitch_mask in _bounce_band_masks(n_pitch, boundary):
            block_bounce_masks[(block_index, bounce_label)] = _block_bounce_mask(
                layout,
                block_index,
                pitch_mask,
            )

    if cfg.include_block_bounce_regions:
        for block_index in active:
            for bounce_label in selected_bounce_labels:
                add(block_bounce_masks[(block_index, bounce_label)], f"block:{block_index}*bounce:{bounce_label}")

    if cfg.include_radial_bounce_regions:
        block_x = tuple(int(value) for value in (layout.block_x or ()))
        for radial_value, radial_mask in _group_masks_by_value(layout, active, block_x):
            for bounce_label in selected_bounce_labels:
                masks = tuple(
                    block_bounce_masks[(block_index, bounce_label)]
                    for block_index in active
                    if int(block_x[block_index]) == radial_value
                )
                if masks:
                    add(
                        np.logical_and(radial_mask, _union_masks(masks)),
                        f"radial:{radial_value}*bounce:{bounce_label}",
                    )

    if cfg.include_species_bounce_regions:
        block_species = tuple(int(value) for value in (layout.block_species or ()))
        for species, species_mask in _group_masks_by_value(layout, active, block_species):
            for bounce_label in selected_bounce_labels:
                masks = tuple(
                    block_bounce_masks[(block_index, bounce_label)]
                    for block_index in active
                    if int(block_species[block_index]) == species
                )
                if masks:
                    add(
                        np.logical_and(species_mask, _union_masks(masks)),
                        f"species:{species}*bounce:{bounce_label}",
                    )

    if cfg.include_pitch_regions:
        for bounce_label in selected_bounce_labels:
            masks = tuple(block_bounce_masks[(block_index, bounce_label)] for block_index in active)
            add(_union_masks(masks), f"bounce:{bounce_label}")

    if cfg.include_block_regions:
        for block_index in active:
            add(_block_mask(layout, block_index), f"block:{block_index}")

    if cfg.include_radial_regions:
        block_x = tuple(int(value) for value in (layout.block_x or ()))
        for radial_value, radial_mask in _group_masks_by_value(layout, active, block_x):
            add(radial_mask, f"radial:{radial_value}")

    if cfg.include_species_regions:
        block_species = tuple(int(value) for value in (layout.block_species or ()))
        for species, species_mask in _group_masks_by_value(layout, active, block_species):
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
        return _empty_candidates(layout, cfg.dtype)

    regions = _build_regions(layout, residual_host, cfg=cfg)
    if not regions:
        return _empty_candidates(layout, cfg.dtype)

    columns = [residual_vec * jnp.asarray(region.mask, dtype=residual_vec.dtype) for region in regions]
    labels = tuple(region.label for region in regions)
    return jnp.stack(tuple(columns), axis=1), labels


def build_rhs1_qi_residual_region_coarse_basis(
    layout: RHS1QICoarseBlockLayout,
    residual: ArrayLike,
    *,
    config: RHS1QIResidualRegionCoarseConfig | None = None,
) -> RHS1QICoarseBasis:
    """Return a rank-gated residual-region QI coarse basis."""

    cfg = RHS1QIResidualRegionCoarseConfig() if config is None else config
    candidates, labels = build_rhs1_qi_residual_region_coarse_candidates(
        layout,
        residual,
        config=cfg,
    )
    return orthonormalize_rhs1_qi_coarse_basis(
        candidates,
        labels=labels,
        rtol=float(cfg.rtol),
        atol=float(cfg.atol),
        max_rank=max(0, int(cfg.max_rank)),
    )


def project_rhs1_qi_residual_region_correction(
    residual: ArrayLike,
    basis: RHS1QICoarseBasis,
) -> ArrayLike:
    """Project ``residual`` into a residual-region coarse basis."""

    residual_vec = jnp.asarray(residual, dtype=basis.vectors.dtype).reshape((-1,))
    if int(basis.metadata.rank) <= 0:
        return jnp.zeros_like(residual_vec)
    return basis.vectors @ (jnp.conjugate(basis.vectors).T @ residual_vec)


__all__ = [
    "RHS1QIResidualRegionCoarseConfig",
    "build_rhs1_qi_residual_region_coarse_basis",
    "build_rhs1_qi_residual_region_coarse_candidates",
    "project_rhs1_qi_residual_region_correction",
]
