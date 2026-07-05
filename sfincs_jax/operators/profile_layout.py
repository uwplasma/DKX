"""JAX-native block-operator primitives for RHSMode=1 production solves.

The current RHSMode=1 driver still has several matrix-free and diagnostic
sparse paths.  This module is the neutral foundation for the replacement
architecture: it records the physical block layout, wraps matvecs with explicit
metadata, and provides the first small JAX-native block factor kernel.  The
objects here deliberately do not choose solver policy.

This file is intentionally larger than the usual module target during the
consolidation pass because layout metadata, block-COO storage, and reusable
symbolic active-ordering caches must evolve together. A safe future split is to
separate pure data layouts from executable block-operator kernels after
``v3_driver.py`` no longer owns solve orchestration.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
from types import SimpleNamespace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


_ACTIVE_FIELD_SPLIT_ORDERING_CACHE_MAX_SIZE = 64
_ACTIVE_FIELD_SPLIT_ORDERING_CACHE: dict[tuple[object, ...], "RHS1ActiveFieldSplitOrdering"] = {}


def clear_rhs1_active_field_split_ordering_cache() -> None:
    """Clear cached RHSMode=1 symbolic active field-split orderings."""

    _ACTIVE_FIELD_SPLIT_ORDERING_CACHE.clear()


@dataclass(frozen=True)
class RHS1KineticIndices:
    """Decoded kinetic ``f``-block indices in SFINCS v3 flat ordering."""

    species: np.ndarray
    x: np.ndarray
    ell: np.ndarray
    theta: np.ndarray
    zeta: np.ndarray


@dataclass(frozen=True)
class RHS1BlockLayout:
    """Physical block layout for an RHSMode=1 full-system vector."""

    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    f_size: int
    phi1_size: int
    extra_size: int
    total_size: int
    constraint_scheme: int
    include_phi1: bool
    include_phi1_in_kinetic: bool
    rhs_mode: int

    @classmethod
    def from_operator(cls, op: Any) -> "RHS1BlockLayout":
        """Build layout metadata from a ``V3FullSystemOperator``-like object."""

        return cls(
            n_species=int(op.n_species),
            n_x=int(op.n_x),
            n_xi=int(op.n_xi),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            f_size=int(op.f_size),
            phi1_size=int(op.phi1_size),
            extra_size=int(op.extra_size),
            total_size=int(op.total_size),
            constraint_scheme=int(op.constraint_scheme),
            include_phi1=bool(op.include_phi1),
            include_phi1_in_kinetic=bool(op.include_phi1_in_kinetic),
            rhs_mode=int(op.rhs_mode),
        )

    @property
    def f_shape(self) -> tuple[int, int, int, int, int]:
        return (
            int(self.n_species),
            int(self.n_x),
            int(self.n_xi),
            int(self.n_theta),
            int(self.n_zeta),
        )

    @property
    def f_slice(self) -> slice:
        return slice(0, int(self.f_size))

    @property
    def phi1_slice(self) -> slice:
        start = int(self.f_size)
        return slice(start, start + int(self.phi1_size))

    @property
    def phi1_field_slice(self) -> slice:
        start = int(self.f_size)
        field_size = int(self.n_theta * self.n_zeta) if bool(self.include_phi1) else 0
        return slice(start, start + field_size)

    @property
    def phi1_lambda_slice(self) -> slice:
        if not bool(self.include_phi1):
            return slice(int(self.f_size), int(self.f_size))
        start = int(self.f_size + self.n_theta * self.n_zeta)
        return slice(start, start + 1)

    @property
    def extra_slice(self) -> slice:
        start = int(self.f_size + self.phi1_size)
        return slice(start, start + int(self.extra_size))

    def component_sizes(self) -> dict[str, int]:
        """Return vector component sizes in full-system order."""

        return {
            "kinetic": int(self.f_size),
            "phi1": int(self.phi1_size),
            "extra": int(self.extra_size),
            "total": int(self.total_size),
        }

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly layout metadata for solver traces."""

        return {
            "n_species": int(self.n_species),
            "n_x": int(self.n_x),
            "n_xi": int(self.n_xi),
            "n_theta": int(self.n_theta),
            "n_zeta": int(self.n_zeta),
            "f_size": int(self.f_size),
            "phi1_size": int(self.phi1_size),
            "extra_size": int(self.extra_size),
            "total_size": int(self.total_size),
            "constraint_scheme": int(self.constraint_scheme),
            "include_phi1": bool(self.include_phi1),
            "include_phi1_in_kinetic": bool(self.include_phi1_in_kinetic),
            "rhs_mode": int(self.rhs_mode),
        }

    def kinetic_flat_index(
        self,
        species: int,
        x: int,
        ell: int,
        theta: int,
        zeta: int,
    ) -> int:
        """Return the flat ``f`` index for SFINCS v3 kinetic ordering."""

        self._check_kinetic_index_bounds(species, x, ell, theta, zeta)
        index = ((int(species) * int(self.n_x) + int(x)) * int(self.n_xi) + int(ell)) * int(self.n_theta)
        index += int(theta)
        return int(index * int(self.n_zeta) + int(zeta))

    def decode_kinetic_indices(self, flat_indices: Any) -> RHS1KineticIndices:
        """Decode flat ``f`` indices into physical kinetic axes."""

        flat = np.asarray(flat_indices, dtype=np.int64).reshape((-1,))
        if np.any(flat < 0) or np.any(flat >= int(self.f_size)):
            raise ValueError("kinetic flat indices must lie inside the f block")
        n_zeta = int(self.n_zeta)
        n_theta = int(self.n_theta)
        n_xi = int(self.n_xi)
        n_x = int(self.n_x)
        zeta = flat % n_zeta
        theta = (flat // n_zeta) % n_theta
        ell = (flat // (n_zeta * n_theta)) % n_xi
        x = (flat // (n_zeta * n_theta * n_xi)) % n_x
        species = flat // (n_zeta * n_theta * n_xi * n_x)
        return RHS1KineticIndices(
            species=species.astype(np.int64, copy=False),
            x=x.astype(np.int64, copy=False),
            ell=ell.astype(np.int64, copy=False),
            theta=theta.astype(np.int64, copy=False),
            zeta=zeta.astype(np.int64, copy=False),
        )

    def _check_kinetic_index_bounds(
        self,
        species: int,
        x: int,
        ell: int,
        theta: int,
        zeta: int,
    ) -> None:
        bounds = (
            ("species", int(species), int(self.n_species)),
            ("x", int(x), int(self.n_x)),
            ("ell", int(ell), int(self.n_xi)),
            ("theta", int(theta), int(self.n_theta)),
            ("zeta", int(zeta), int(self.n_zeta)),
        )
        for name, value, stop in bounds:
            if value < 0 or value >= stop:
                raise IndexError(f"{name} index {value} outside [0, {stop})")


@dataclass(frozen=True)
class RHS1CompressedPitchLayout:
    """Fortran-style compressed kinetic layout plus explicit tail blocks."""

    n_species: int
    n_x: int
    n_xi: int
    n_theta: int
    n_zeta: int
    f_size: int
    total_size: int
    nxi_for_x: np.ndarray
    first_index_for_x: np.ndarray
    active_full_indices: np.ndarray
    kinetic_active_full_indices: np.ndarray
    tail_full_indices: np.ndarray
    full_to_reduced_index: np.ndarray

    @property
    def active_pitch_count_per_species(self) -> int:
        return int(np.sum(self.nxi_for_x))

    @property
    def kinetic_active_size_per_species(self) -> int:
        return int(self.active_pitch_count_per_species * self.n_theta * self.n_zeta)

    @property
    def kinetic_active_size(self) -> int:
        return int(self.n_species * self.kinetic_active_size_per_species)

    @property
    def tail_size(self) -> int:
        return int(self.total_size - self.f_size)

    @property
    def reduced_size(self) -> int:
        return int(self.kinetic_active_size + self.tail_size)

    @property
    def tail_reduced_start(self) -> int:
        return int(self.kinetic_active_size)

    def full_kinetic_index(
        self, species: int, x_index: int, ell: int, theta: int, zeta: int
    ) -> int:
        """Return the rectangular full-system f-block index for one kinetic DOF."""

        return int(
            (((int(species) * self.n_x + int(x_index)) * self.n_xi + int(ell)) * self.n_theta + int(theta))
            * self.n_zeta
            + int(zeta)
        )

    def reduced_kinetic_index(
        self, species: int, x_index: int, ell: int, theta: int, zeta: int
    ) -> int:
        """Return the compressed reduced index for an active kinetic DOF."""

        species = int(species)
        x_index = int(x_index)
        ell = int(ell)
        theta = int(theta)
        zeta = int(zeta)
        if ell < 0 or ell >= int(self.nxi_for_x[x_index]):
            raise IndexError(f"ell={ell} is inactive for x index {x_index}")
        pitch_offset = int(self.first_index_for_x[x_index] + ell)
        return int(
            species * self.kinetic_active_size_per_species
            + pitch_offset * self.n_theta * self.n_zeta
            + theta * self.n_zeta
            + zeta
        )

    def species_x_reduced_slice(self, species: int, x_index: int) -> slice:
        """Return the contiguous reduced kinetic range for one species and speed node."""

        species_offset = int(species) * self.kinetic_active_size_per_species
        x_start = species_offset + int(self.first_index_for_x[int(x_index)]) * self.n_theta * self.n_zeta
        x_stop = x_start + int(self.nxi_for_x[int(x_index)]) * self.n_theta * self.n_zeta
        return slice(x_start, x_stop)


def _nxi_for_x_from_operator(op: Any) -> np.ndarray:
    collisionless = getattr(getattr(op, "fblock", None), "collisionless", None)
    nxi_for_x = getattr(collisionless, "n_xi_for_x", None)
    if nxi_for_x is None:
        nxi_for_x = np.full((int(op.n_x),), int(op.n_xi), dtype=np.int64)
    return np.asarray(nxi_for_x, dtype=np.int64)


def build_rhs1_compressed_pitch_layout(op: Any) -> RHS1CompressedPitchLayout:
    """Build the reduced active-pitch layout matching Fortran v3 ordering."""

    n_species = int(op.n_species)
    n_x = int(op.n_x)
    n_xi = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    f_size = int(op.f_size)
    total_size = int(op.total_size)

    nxi_for_x = _nxi_for_x_from_operator(op)
    if nxi_for_x.shape != (n_x,):
        raise ValueError(f"n_xi_for_x must have shape {(n_x,)}, got {nxi_for_x.shape}")
    if np.any(nxi_for_x < 0) or np.any(nxi_for_x > n_xi):
        raise ValueError("n_xi_for_x entries must satisfy 0 <= Nxi_for_x <= n_xi")

    expected_f_size = n_species * n_x * n_xi * n_theta * n_zeta
    if f_size != expected_f_size:
        raise ValueError(
            f"f_size={f_size} does not match rectangular kinetic size {expected_f_size}"
        )
    if total_size < f_size:
        raise ValueError(f"total_size={total_size} must be >= f_size={f_size}")

    first_index_for_x = np.zeros((n_x,), dtype=np.int64)
    if n_x > 1:
        first_index_for_x[1:] = np.cumsum(nxi_for_x[:-1], dtype=np.int64)

    kinetic_full_indices: list[int] = []
    for species in range(n_species):
        for x_index in range(n_x):
            for ell in range(int(nxi_for_x[x_index])):
                for theta in range(n_theta):
                    base = (((species * n_x + x_index) * n_xi + ell) * n_theta + theta) * n_zeta
                    kinetic_full_indices.extend(range(base, base + n_zeta))

    kinetic_active_full_indices = np.asarray(kinetic_full_indices, dtype=np.int64)
    tail_full_indices = np.arange(f_size, total_size, dtype=np.int64)
    active_full_indices = np.concatenate([kinetic_active_full_indices, tail_full_indices])

    full_to_reduced = np.full((total_size,), -1, dtype=np.int64)
    full_to_reduced[active_full_indices] = np.arange(active_full_indices.size, dtype=np.int64)

    return RHS1CompressedPitchLayout(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=total_size,
        nxi_for_x=nxi_for_x,
        first_index_for_x=first_index_for_x,
        active_full_indices=active_full_indices,
        kinetic_active_full_indices=kinetic_active_full_indices,
        tail_full_indices=tail_full_indices,
        full_to_reduced_index=full_to_reduced,
    )


def infer_rhs1_compressed_pitch_layout_from_active_indices(
    op_or_layout: Any,
    active_indices: Any | None,
) -> RHS1CompressedPitchLayout:
    """Infer a compressed active-pitch layout from an active full-index vector."""

    n_species = int(op_or_layout.n_species)
    n_x = int(op_or_layout.n_x)
    n_xi = int(op_or_layout.n_xi)
    n_theta = int(op_or_layout.n_theta)
    n_zeta = int(op_or_layout.n_zeta)
    f_size = int(op_or_layout.f_size)
    total_size = int(op_or_layout.total_size)

    if active_indices is None:
        active = np.arange(total_size, dtype=np.int64)
    else:
        active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
    if np.any(active < 0) or np.any(active >= total_size):
        raise ValueError("active_indices contains entries outside the full vector")
    if np.unique(active).size != active.size:
        raise ValueError("active_indices must not contain duplicates")

    kinetic_active = active[active < f_size]
    tail_active = active[active >= f_size]
    if tail_active.size:
        expected_tail = np.arange(f_size, total_size, dtype=np.int64)
        if not np.array_equal(tail_active, expected_tail):
            raise ValueError("active tail indices must be the explicit contiguous tail block")
        inferred_total_size = total_size
    else:
        inferred_total_size = f_size

    active_mask = np.zeros((n_species, n_x, n_xi, n_theta, n_zeta), dtype=bool)
    if kinetic_active.size:
        flat = kinetic_active.astype(np.int64, copy=False)
        zeta = flat % n_zeta
        theta = (flat // n_zeta) % n_theta
        ell = (flat // (n_zeta * n_theta)) % n_xi
        x_index = (flat // (n_zeta * n_theta * n_xi)) % n_x
        species = flat // (n_zeta * n_theta * n_xi * n_x)
        active_mask[species, x_index, ell, theta, zeta] = True

    nxi_for_x = np.zeros((n_x,), dtype=np.int64)
    for x_index in range(n_x):
        ell_active = np.any(active_mask[:, x_index, :, :, :], axis=(0, 2, 3))
        active_ells = np.flatnonzero(ell_active)
        if active_ells.size == 0:
            nxi = 0
        else:
            nxi = int(active_ells[-1]) + 1
            if not np.array_equal(active_ells, np.arange(nxi, dtype=np.int64)):
                raise ValueError("active kinetic pitch modes must be a contiguous ell prefix for each x")
        expected = np.zeros((n_species, n_xi, n_theta, n_zeta), dtype=bool)
        expected[:, :nxi, :, :] = True
        if not np.array_equal(active_mask[:, x_index, :, :, :], expected):
            raise ValueError("active kinetic indices do not form complete species/theta/zeta planes")
        nxi_for_x[x_index] = nxi

    op = SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=inferred_total_size,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=nxi_for_x),
        ),
    )
    layout = build_rhs1_compressed_pitch_layout(op)
    if not np.array_equal(layout.active_full_indices, active):
        raise ValueError("active_indices ordering does not match compressed Fortran active-pitch ordering")
    return layout


@dataclass(frozen=True)
class RHS1ActiveBlockLayout:
    """Active-DOF view of an RHSMode=1 layout."""

    layout: RHS1BlockLayout
    active_indices: np.ndarray | None
    active_size: int
    kinetic_count: int
    phi1_count: int
    extra_count: int
    has_contiguous_extra_tail: bool

    @classmethod
    def from_layout(
        cls,
        layout: RHS1BlockLayout,
        active_indices: Any | None = None,
    ) -> "RHS1ActiveBlockLayout":
        """Build active-block metadata, validating index bounds and duplicates."""

        if active_indices is None:
            return cls(
                layout=layout,
                active_indices=None,
                active_size=int(layout.total_size),
                kinetic_count=int(layout.f_size),
                phi1_count=int(layout.phi1_size),
                extra_count=int(layout.extra_size),
                has_contiguous_extra_tail=bool(layout.extra_size > 0),
            )

        active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
        if np.any(active < 0) or np.any(active >= int(layout.total_size)):
            raise ValueError("active_indices contains entries outside the full vector")
        if np.unique(active).size != active.size:
            raise ValueError("active_indices must not contain duplicates")

        phi1_start = int(layout.f_size)
        extra_start = int(layout.f_size + layout.phi1_size)
        kinetic_count = int(np.count_nonzero(active < phi1_start))
        phi1_count = int(np.count_nonzero((active >= phi1_start) & (active < extra_start)))
        extra_count = int(np.count_nonzero(active >= extra_start))
        expected_tail = np.arange(extra_start, int(layout.total_size), dtype=np.int64)
        has_tail = bool(
            int(layout.extra_size) > 0
            and int(active.size) >= int(layout.extra_size)
            and np.array_equal(active[-int(layout.extra_size) :], expected_tail)
        )
        return cls(
            layout=layout,
            active_indices=active.astype(np.int32, copy=False),
            active_size=int(active.size),
            kinetic_count=kinetic_count,
            phi1_count=phi1_count,
            extra_count=extra_count,
            has_contiguous_extra_tail=has_tail,
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly active-layout metadata."""

        return {
            "active_size": int(self.active_size),
            "kinetic_count": int(self.kinetic_count),
            "phi1_count": int(self.phi1_count),
            "extra_count": int(self.extra_count),
            "has_contiguous_extra_tail": bool(self.has_contiguous_extra_tail),
        }

    def active_kinetic_indices(self) -> np.ndarray:
        """Return active flat indices restricted to the kinetic block."""

        if self.active_indices is None:
            return np.arange(int(self.layout.f_size), dtype=np.int32)
        active = np.asarray(self.active_indices, dtype=np.int32)
        return active[active < int(self.layout.f_size)]


@dataclass(frozen=True)
class RHS1ActiveFieldSplitOrdering:
    """Reusable symbolic ordering for active RHSMode=1 field splits.

    The direct-tail RHSMode=1 path works on an active-projected vector, but the
    important physics blocks are still defined in full SFINCS ordering:
    kinetic ``f`` rows, optional ``Phi1`` rows, and global
    source/constraint/profile rows.  This class stores the active-to-full
    mapping once and exposes deterministic subsets for block factors and coarse
    residual equations.
    """

    layout: RHS1BlockLayout
    active_indices: np.ndarray
    position_by_full_index: np.ndarray
    kinetic_positions: np.ndarray
    phi1_positions: np.ndarray
    extra_positions: np.ndarray

    @staticmethod
    def cache_key(
        layout: RHS1BlockLayout,
        active_indices: Any | None = None,
    ) -> tuple[object, ...]:
        """Return a semantic fixed-shape key for active field-split ordering.

        The key intentionally contains only layout and active-set information.
        It must not depend on matrix values, electric field, profiles, or
        geometry amplitudes because this cache stores symbolic index maps only.
        """

        if active_indices is None:
            active_size = int(layout.total_size)
            active_digest = "all"
        else:
            active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
            active_size = int(active.size)
            active_digest = hashlib.sha256(active.tobytes()).hexdigest()
        return (
            "rhs1_active_field_split_ordering_v1",
            int(layout.n_species),
            int(layout.n_x),
            int(layout.n_xi),
            int(layout.n_theta),
            int(layout.n_zeta),
            int(layout.phi1_size),
            int(layout.extra_size),
            int(layout.total_size),
            int(layout.constraint_scheme),
            int(bool(layout.include_phi1)),
            int(bool(layout.include_phi1_in_kinetic)),
            int(layout.rhs_mode),
            active_size,
            active_digest,
        )

    @classmethod
    def cached_from_layout(
        cls,
        layout: RHS1BlockLayout,
        active_indices: Any | None = None,
    ) -> "RHS1ActiveFieldSplitOrdering":
        """Build or reuse symbolic active-position maps for the same shape."""

        key = cls.cache_key(layout, active_indices)
        cached = _ACTIVE_FIELD_SPLIT_ORDERING_CACHE.get(key)
        if cached is not None:
            return cached
        ordering = cls.from_layout(layout, active_indices)
        if len(_ACTIVE_FIELD_SPLIT_ORDERING_CACHE) >= _ACTIVE_FIELD_SPLIT_ORDERING_CACHE_MAX_SIZE:
            _ACTIVE_FIELD_SPLIT_ORDERING_CACHE.clear()
        _ACTIVE_FIELD_SPLIT_ORDERING_CACHE[key] = ordering
        return ordering

    @classmethod
    def from_layout(
        cls,
        layout: RHS1BlockLayout,
        active_indices: Any | None = None,
    ) -> "RHS1ActiveFieldSplitOrdering":
        """Build symbolic active-position maps from a layout and active set."""

        if active_indices is None:
            active = np.arange(int(layout.total_size), dtype=np.int64)
        else:
            active = np.asarray(active_indices, dtype=np.int64).reshape((-1,))
        if active.size == 0:
            raise ValueError("active_indices must not be empty")
        if np.any(active < 0) or np.any(active >= int(layout.total_size)):
            raise ValueError("active_indices contains entries outside the full vector")
        if np.unique(active).size != active.size:
            raise ValueError("active_indices must not contain duplicates")

        position_by_full = np.full((int(layout.total_size),), -1, dtype=np.int64)
        position_by_full[active] = np.arange(int(active.size), dtype=np.int64)
        phi1_start = int(layout.f_size)
        extra_start = int(layout.f_size + layout.phi1_size)
        positions = np.arange(int(active.size), dtype=np.int64)
        kinetic = positions[active < phi1_start]
        phi1 = positions[(active >= phi1_start) & (active < extra_start)]
        extra = positions[active >= extra_start]
        return cls(
            layout=layout,
            active_indices=active.astype(np.int64, copy=False),
            position_by_full_index=position_by_full,
            kinetic_positions=kinetic.astype(np.int64, copy=False),
            phi1_positions=phi1.astype(np.int64, copy=False),
            extra_positions=extra.astype(np.int64, copy=False),
        )

    @property
    def active_size(self) -> int:
        return int(self.active_indices.size)

    @property
    def kinetic_size(self) -> int:
        return int(self.kinetic_positions.size)

    @property
    def phi1_size(self) -> int:
        return int(self.phi1_positions.size)

    @property
    def extra_size(self) -> int:
        return int(self.extra_positions.size)

    def active_positions_for_full_indices(self, full_indices: Any) -> np.ndarray:
        """Map full-system indices into active-vector positions."""

        full = np.asarray(full_indices, dtype=np.int64).reshape((-1,))
        if full.size == 0:
            return np.zeros((0,), dtype=np.int64)
        if np.any(full < 0) or np.any(full >= int(self.layout.total_size)):
            raise ValueError("full_indices contains entries outside the layout")
        positions = self.position_by_full_index[full]
        keep = positions >= 0
        if not np.any(keep):
            return np.zeros((0,), dtype=np.int64)
        return np.unique(positions[keep].astype(np.int64, copy=False))

    def dominant_kinetic_positions(
        self,
        *,
        x_count: int,
        ell_count: int,
        max_positions: int,
        species_count: int | None = None,
        theta_stride: int = 1,
        zeta_stride: int = 1,
    ) -> np.ndarray:
        """Return a deterministic low-``x``/low-``ell`` active kinetic subset.

        The subset is intended for bounded true-Schur residual equations.  It
        keeps the angular structure for selected speed/pitch modes while
        allowing strides and a hard cap for production-size systems.
        """

        x_stop = max(0, min(int(self.layout.n_x), int(x_count)))
        ell_stop = max(0, min(int(self.layout.n_xi), int(ell_count)))
        species_stop = int(self.layout.n_species) if species_count is None else int(species_count)
        species_stop = max(0, min(int(self.layout.n_species), int(species_stop)))
        max_use = max(0, int(max_positions))
        theta_step = max(1, int(theta_stride))
        zeta_step = max(1, int(zeta_stride))
        if x_stop == 0 or ell_stop == 0 or species_stop == 0 or max_use == 0:
            return np.zeros((0,), dtype=np.int64)

        positions: list[int] = []
        for species in range(species_stop):
            for x_index in range(x_stop):
                for ell in range(ell_stop):
                    for theta in range(0, int(self.layout.n_theta), theta_step):
                        for zeta in range(0, int(self.layout.n_zeta), zeta_step):
                            full = self.layout.kinetic_flat_index(
                                species=species,
                                x=x_index,
                                ell=ell,
                                theta=theta,
                                zeta=zeta,
                            )
                            position = int(self.position_by_full_index[full])
                            if position >= 0:
                                positions.append(position)
                                if len(positions) >= max_use:
                                    return np.asarray(positions, dtype=np.int64)
        return np.asarray(positions, dtype=np.int64)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly symbolic split metadata."""

        return {
            "kind": "active_field_split_symbolic_ordering",
            "active_size": int(self.active_size),
            "kinetic_size": int(self.kinetic_size),
            "phi1_size": int(self.phi1_size),
            "extra_size": int(self.extra_size),
            "has_phi1": bool(self.phi1_size > 0),
            "has_extra_tail": bool(self.extra_size > 0),
        }


@dataclass(frozen=True)
class RHS1BlockLinearOperator:
    """JAX-compatible matvec wrapper carrying RHSMode=1 block metadata."""

    layout: RHS1BlockLayout
    matvec_fn: Callable[[Any], Any]
    dtype: Any = jnp.float64
    name: str = "rhs1_block_operator"
    active_layout: RHS1ActiveBlockLayout | None = None

    @property
    def shape(self) -> tuple[int, int]:
        n = int(self.active_layout.active_size if self.active_layout is not None else self.layout.total_size)
        return (n, n)

    def matvec(self, x: Any) -> jax.Array:
        """Apply the operator to a vector, checking only the public shape."""

        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 1 or int(x_arr.shape[0]) != int(self.shape[1]):
            raise ValueError(f"matvec expects shape {(self.shape[1],)}, got {tuple(x_arr.shape)}")
        out = jnp.asarray(self.matvec_fn(x_arr), dtype=self.dtype)
        if out.ndim != 1 or int(out.shape[0]) != int(self.shape[0]):
            raise ValueError(f"matvec returned shape {tuple(out.shape)}, expected {(self.shape[0],)}")
        return out

    def matmat(self, x: Any) -> jax.Array:
        """Apply the operator to a dense column batch."""

        x_arr = jnp.asarray(x, dtype=self.dtype)
        if x_arr.ndim != 2 or int(x_arr.shape[0]) != int(self.shape[1]):
            raise ValueError(f"matmat expects shape ({self.shape[1]}, ncols), got {tuple(x_arr.shape)}")
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(x_arr)

    def __call__(self, x: Any) -> jax.Array:
        return self.matvec(x)

    def jitted_matvec(self) -> Callable[[Any], jax.Array]:
        """Return a JIT-compiled matvec closure."""

        return jax.jit(self.matvec)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly operator metadata."""

        return {
            "name": str(self.name),
            "shape": tuple(int(v) for v in self.shape),
            "dtype": str(jnp.dtype(self.dtype)),
            "layout": self.layout.to_dict(),
            "active_layout": None if self.active_layout is None else self.active_layout.to_dict(),
        }


@dataclass(frozen=True)
class RHS1BlockPreconditionerProbe:
    """True-residual probe for a JAX-native block preconditioner."""

    accepted: bool
    reason: str
    residual_before_norm: float
    residual_after_norm: float
    improvement_ratio: float | None
    target_residual_norm: float | None
    target_ratio: float
    x_candidate: jax.Array
    factor_metadata: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly diagnostics without the candidate vector."""

        return {
            "accepted": bool(self.accepted),
            "reason": str(self.reason),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "target_residual_norm": (
                None if self.target_residual_norm is None else float(self.target_residual_norm)
            ),
            "target_ratio": float(self.target_ratio),
            "factor_metadata": dict(self.factor_metadata),
        }


class RHS1BlockCOOBuilder:
    """Incremental builder for uniform block-COO operators.

    Physics terms can add scalar stencil entries or whole dense blocks.  The
    builder deterministically sums duplicate block entries and emits a compact
    ``RHS1BlockCOOOperator`` without requiring a dense matrix or SciPy CSR.
    """

    def __init__(self, *, shape: tuple[int, int], block_size: int, dtype: Any = np.float64) -> None:
        n_rows, n_cols = int(shape[0]), int(shape[1])
        block = int(block_size)
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("shape entries must be positive")
        if block <= 0 or n_rows % block != 0 or n_cols % block != 0:
            raise ValueError("block_size must divide both shape dimensions")
        self.shape = (n_rows, n_cols)
        self.block_size = block
        self.dtype = np.dtype(dtype)
        self.n_block_rows = n_rows // block
        self.n_block_cols = n_cols // block
        self._blocks: dict[tuple[int, int], np.ndarray] = {}

    @property
    def nnz_blocks(self) -> int:
        return len(self._blocks)

    @property
    def data_nbytes_estimate(self) -> int:
        return int(self.nnz_blocks * self.block_size * self.block_size * self.dtype.itemsize)

    def add_dense_block(self, row_block: int, col_block: int, block: Any) -> None:
        """Add a dense block at ``(row_block, col_block)``."""

        row = int(row_block)
        col = int(col_block)
        if row < 0 or row >= int(self.n_block_rows):
            raise ValueError("row_block outside builder shape")
        if col < 0 or col >= int(self.n_block_cols):
            raise ValueError("col_block outside builder shape")
        value = np.asarray(block, dtype=self.dtype)
        expected = (int(self.block_size), int(self.block_size))
        if value.shape != expected:
            raise ValueError(f"dense block must have shape {expected}, got {value.shape}")
        key = (row, col)
        if key not in self._blocks:
            self._blocks[key] = np.zeros(expected, dtype=self.dtype)
        self._blocks[key] += value

    def add_scalar_entries(
        self,
        *,
        row_indices: Any,
        col_indices: Any,
        values: Any,
        drop_tol: float = 0.0,
    ) -> None:
        """Add scalar COO entries, grouped into the builder's block map."""

        rows = np.asarray(row_indices, dtype=np.int64).reshape((-1,))
        cols = np.asarray(col_indices, dtype=np.int64).reshape((-1,))
        vals = np.asarray(values, dtype=self.dtype).reshape((-1,))
        if rows.shape != cols.shape or rows.shape != vals.shape:
            raise ValueError("row_indices, col_indices, and values must have matching lengths")
        if rows.size and (
            np.any(rows < 0)
            or np.any(rows >= int(self.shape[0]))
            or np.any(cols < 0)
            or np.any(cols >= int(self.shape[1]))
        ):
            raise ValueError("COO entries contain row/column indices outside shape")
        threshold = float(drop_tol)
        block = int(self.block_size)
        for row, col, value in zip(rows, cols, vals, strict=True):
            if threshold > 0.0 and abs(value) <= threshold:
                continue
            key = (int(row // block), int(col // block))
            if key not in self._blocks:
                self._blocks[key] = np.zeros((block, block), dtype=self.dtype)
            self._blocks[key][int(row % block), int(col % block)] += value

    def add_tridiagonal_block_line(
        self,
        *,
        block_indices: Any,
        diagonal_blocks: Any,
        lower_blocks: Any | None = None,
        upper_blocks: Any | None = None,
    ) -> None:
        """Add a nearest-neighbor dense-block line stencil.

        ``block_indices`` are global block-row/block-column ids along a physical
        line, for example fixed ``(species, x, theta, zeta)`` with neighboring
        pitch blocks, or fixed ``(species, xi, theta, zeta)`` with neighboring
        radial blocks.  The method is intentionally small and deterministic:
        diagonal blocks are always added, and lower/upper couplings add
        row-position ``i`` to column-position ``i-1``/``i+1`` entries.
        """

        indices = np.asarray(block_indices, dtype=np.int64).reshape((-1,))
        if indices.size == 0:
            raise ValueError("block_indices must contain at least one block")
        if np.any(indices < 0) or np.any(indices >= int(self.n_block_rows)) or np.any(indices >= int(self.n_block_cols)):
            raise ValueError("block_indices contains entries outside builder block shape")
        if np.unique(indices).size != indices.size:
            raise ValueError("block_indices must not contain duplicates")

        diagonal = self._coerce_block_sequence(diagonal_blocks, count=int(indices.size), name="diagonal_blocks")
        lower = self._coerce_block_sequence(
            lower_blocks,
            count=max(0, int(indices.size) - 1),
            name="lower_blocks",
            allow_none=True,
        )
        upper = self._coerce_block_sequence(
            upper_blocks,
            count=max(0, int(indices.size) - 1),
            name="upper_blocks",
            allow_none=True,
        )
        for position, block_id in enumerate(indices):
            self.add_dense_block(int(block_id), int(block_id), diagonal[int(position)])
        for position in range(max(0, int(indices.size) - 1)):
            row_next = int(indices[position + 1])
            row_this = int(indices[position])
            if lower is not None:
                self.add_dense_block(row_next, row_this, lower[position])
            if upper is not None:
                self.add_dense_block(row_this, row_next, upper[position])

    def _coerce_block_sequence(
        self,
        values: Any | None,
        *,
        count: int,
        name: str,
        allow_none: bool = False,
    ) -> np.ndarray | None:
        if values is None:
            if bool(allow_none):
                return None
            raise ValueError(f"{name} is required")
        block = int(self.block_size)
        arr = np.asarray(values, dtype=self.dtype)
        if int(count) == 0:
            if arr.size == 0:
                return np.zeros((0, block, block), dtype=self.dtype)
            raise ValueError(f"{name} must be empty for a one-block line")
        if arr.shape == (block, block):
            return np.broadcast_to(arr, (int(count), block, block)).copy()
        expected = (int(count), block, block)
        if arr.shape != expected:
            raise ValueError(f"{name} must have shape {(block, block)} or {expected}, got {arr.shape}")
        return arr

    def build(self, *, drop_tol: float = 0.0) -> "RHS1BlockCOOOperator":
        """Build the final immutable block-COO operator."""

        block_rows: list[int] = []
        block_cols: list[int] = []
        block_values: list[jax.Array] = []
        threshold = float(drop_tol)
        for key in sorted(self._blocks):
            block_value = self._blocks[key]
            if threshold > 0.0 and not np.any(np.abs(block_value) > threshold):
                continue
            block_rows.append(int(key[0]))
            block_cols.append(int(key[1]))
            block_values.append(jnp.asarray(block_value))
        if block_values:
            blocks = jnp.stack(block_values)
        else:
            blocks = jnp.zeros((0, int(self.block_size), int(self.block_size)), dtype=jnp.dtype(self.dtype))
        return RHS1BlockCOOOperator.from_blocks(
            row_blocks=jnp.asarray(block_rows, dtype=jnp.int32),
            col_blocks=jnp.asarray(block_cols, dtype=jnp.int32),
            blocks=blocks,
            n_block_rows=int(self.n_block_rows),
            n_block_cols=int(self.n_block_cols),
        )


@dataclass(frozen=True)
class RHS1BlockCOOOperator:
    """Uniform-block COO operator with a pure JAX matvec.

    This is intentionally small: it represents nonzero blocks by
    ``(row_block, col_block, dense_block)`` triples and accumulates with
    ``jax.Array.at[...].add``.  It is enough to prototype BSR/ELL-style local
    operators without building a monolithic scalar CSR matrix.
    """

    row_blocks: jax.Array
    col_blocks: jax.Array
    blocks: jax.Array
    n_block_rows: int
    n_block_cols: int

    @classmethod
    def from_blocks(
        cls,
        *,
        row_blocks: Any,
        col_blocks: Any,
        blocks: Any,
        n_block_rows: int,
        n_block_cols: int,
    ) -> "RHS1BlockCOOOperator":
        """Create a block-COO operator, validating static dimensions."""

        rows = jnp.asarray(row_blocks, dtype=jnp.int32).reshape((-1,))
        cols = jnp.asarray(col_blocks, dtype=jnp.int32).reshape((-1,))
        vals = jnp.asarray(blocks)
        if vals.ndim != 3 or int(vals.shape[1]) != int(vals.shape[2]):
            raise ValueError("blocks must have shape (nnz_blocks, block_size, block_size)")
        if int(rows.shape[0]) != int(cols.shape[0]) or int(rows.shape[0]) != int(vals.shape[0]):
            raise ValueError("row_blocks, col_blocks, and blocks must have the same leading dimension")
        if int(n_block_rows) <= 0 or int(n_block_cols) <= 0:
            raise ValueError("block dimensions must be positive")
        if int(rows.shape[0]) > 0:
            row_np = np.asarray(rows)
            col_np = np.asarray(cols)
            if np.any(row_np < 0) or np.any(row_np >= int(n_block_rows)):
                raise ValueError("row_blocks contains entries outside n_block_rows")
            if np.any(col_np < 0) or np.any(col_np >= int(n_block_cols)):
                raise ValueError("col_blocks contains entries outside n_block_cols")
        return cls(
            row_blocks=rows,
            col_blocks=cols,
            blocks=vals,
            n_block_rows=int(n_block_rows),
            n_block_cols=int(n_block_cols),
        )

    @classmethod
    def from_dense_matrix(cls, matrix: Any, *, block_size: int, drop_tol: float = 0.0) -> "RHS1BlockCOOOperator":
        """Extract nonzero uniform blocks from a dense matrix."""

        mat = jnp.asarray(matrix)
        block = int(block_size)
        if mat.ndim != 2:
            raise ValueError("matrix must be 2D")
        if block <= 0 or int(mat.shape[0]) % block != 0 or int(mat.shape[1]) % block != 0:
            raise ValueError("block_size must divide both matrix dimensions")
        n_row_blocks = int(mat.shape[0]) // block
        n_col_blocks = int(mat.shape[1]) // block
        rows: list[int] = []
        cols: list[int] = []
        vals: list[jax.Array] = []
        mat_np = np.asarray(mat)
        for row in range(n_row_blocks):
            for col in range(n_col_blocks):
                block_value = mat_np[row * block : (row + 1) * block, col * block : (col + 1) * block]
                if np.any(np.abs(block_value) > float(drop_tol)):
                    rows.append(row)
                    cols.append(col)
                    vals.append(jnp.asarray(block_value, dtype=mat.dtype))
        if vals:
            blocks = jnp.stack(vals)
        else:
            blocks = jnp.zeros((0, block, block), dtype=mat.dtype)
        return cls.from_blocks(
            row_blocks=jnp.asarray(rows, dtype=jnp.int32),
            col_blocks=jnp.asarray(cols, dtype=jnp.int32),
            blocks=blocks,
            n_block_rows=n_row_blocks,
            n_block_cols=n_col_blocks,
        )

    @classmethod
    def from_scalar_coo_entries(
        cls,
        *,
        row_indices: Any,
        col_indices: Any,
        values: Any,
        shape: tuple[int, int],
        block_size: int,
        drop_tol: float = 0.0,
    ) -> "RHS1BlockCOOOperator":
        """Group scalar COO entries into uniform dense block entries.

        This is the bridge between physics-term stencil generation and the
        JAX-native block operator.  It avoids constructing a dense matrix while
        still producing deterministic dense blocks for each occupied
        ``(row_block, col_block)`` pair.
        """

        rows = np.asarray(row_indices, dtype=np.int64).reshape((-1,))
        cols = np.asarray(col_indices, dtype=np.int64).reshape((-1,))
        vals = np.asarray(values).reshape((-1,))
        if rows.shape != cols.shape or rows.shape != vals.shape:
            raise ValueError("row_indices, col_indices, and values must have matching lengths")
        n_rows, n_cols = (int(shape[0]), int(shape[1]))
        block = int(block_size)
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("shape entries must be positive")
        if block <= 0 or n_rows % block != 0 or n_cols % block != 0:
            raise ValueError("block_size must divide both shape dimensions")
        if rows.size and (np.any(rows < 0) or np.any(rows >= n_rows) or np.any(cols < 0) or np.any(cols >= n_cols)):
            raise ValueError("COO entries contain row/column indices outside shape")

        dtype = vals.dtype if vals.size else np.dtype(np.float64)
        builder = RHS1BlockCOOBuilder(shape=(n_rows, n_cols), block_size=block, dtype=dtype)
        builder.add_scalar_entries(row_indices=rows, col_indices=cols, values=vals, drop_tol=float(drop_tol))
        return builder.build(drop_tol=float(drop_tol))

    @property
    def block_size(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def nnz_blocks(self) -> int:
        return int(self.blocks.shape[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (
            int(self.n_block_rows * self.block_size),
            int(self.n_block_cols * self.block_size),
        )

    @property
    def data_nbytes(self) -> int:
        return int(self.blocks.size * self.blocks.dtype.itemsize)

    def matvec(self, x: Any) -> jax.Array:
        """Apply the block-COO operator to a dense vector."""

        x_arr = jnp.asarray(x, dtype=self.blocks.dtype)
        if x_arr.ndim != 1 or int(x_arr.shape[0]) != int(self.shape[1]):
            raise ValueError(f"matvec expects shape {(self.shape[1],)}, got {tuple(x_arr.shape)}")
        block = int(self.block_size)
        x_blocks = x_arr.reshape((int(self.n_block_cols), block))
        gathered = x_blocks[self.col_blocks]
        contrib = jnp.einsum("nij,nj->ni", self.blocks, gathered)
        out = jnp.zeros((int(self.n_block_rows), block), dtype=self.blocks.dtype)
        out = out.at[self.row_blocks].add(contrib)
        return out.reshape((-1,))

    def matmat(self, x: Any) -> jax.Array:
        """Apply the block-COO operator to a dense column batch."""

        x_arr = jnp.asarray(x, dtype=self.blocks.dtype)
        if x_arr.ndim != 2 or int(x_arr.shape[0]) != int(self.shape[1]):
            raise ValueError(f"matmat expects shape ({self.shape[1]}, ncols), got {tuple(x_arr.shape)}")
        return jax.vmap(self.matvec, in_axes=1, out_axes=1)(x_arr)

    def to_dense_matrix(self) -> jax.Array:
        """Materialize a dense matrix for tiny tests and audits."""

        block = int(self.block_size)
        out = jnp.zeros(self.shape, dtype=self.blocks.dtype)
        for i in range(int(self.nnz_blocks)):
            row = int(self.row_blocks[i])
            col = int(self.col_blocks[i])
            out = out.at[row * block : (row + 1) * block, col * block : (col + 1) * block].add(
                self.blocks[i]
            )
        return out

    def to_scipy_csr_matrix(self):
        """Materialize the block operator as a host SciPy CSR matrix.

        This is the non-autodiff bridge for CLI/runtime sparse solves. It uses
        the already assembled block stencils directly, so it does not probe
        matrix-free columns or build an intermediate dense matrix.
        """

        import scipy.sparse as sp

        block = int(self.block_size)
        blocks = np.asarray(jax.device_get(self.blocks))
        row_blocks = np.asarray(jax.device_get(self.row_blocks), dtype=np.int32)
        col_blocks = np.asarray(jax.device_get(self.col_blocks), dtype=np.int32)
        local_rows, local_cols = np.indices((block, block), dtype=np.int32)
        rows = (row_blocks[:, None, None] * block + local_rows[None, :, :]).reshape((-1,))
        cols = (col_blocks[:, None, None] * block + local_cols[None, :, :]).reshape((-1,))
        data = blocks.reshape((-1,))
        keep = data != 0
        matrix = sp.coo_matrix((data[keep], (rows[keep], cols[keep])), shape=self.shape).tocsr()
        matrix.sum_duplicates()
        return matrix

    def project_block_indices(
        self,
        row_block_indices: Any,
        col_block_indices: Any | None = None,
    ) -> "RHS1BlockCOOOperator":
        """Return a block-COO operator restricted to selected block rows/columns.

        The order of ``row_block_indices`` and ``col_block_indices`` defines the
        order in the projected operator.  This is useful for RHSMode=1 active-DOF
        systems, where inactive pitch rows remove whole ``Nzeta`` blocks before
        scalar CSR materialization.
        """

        row_keep = np.asarray(row_block_indices, dtype=np.int64).reshape((-1,))
        col_keep = row_keep if col_block_indices is None else np.asarray(col_block_indices, dtype=np.int64).reshape((-1,))
        if row_keep.size == 0 or col_keep.size == 0:
            raise ValueError("projected block indices must not be empty")
        if np.any(row_keep < 0) or np.any(row_keep >= int(self.n_block_rows)):
            raise ValueError("row_block_indices contains entries outside n_block_rows")
        if np.any(col_keep < 0) or np.any(col_keep >= int(self.n_block_cols)):
            raise ValueError("col_block_indices contains entries outside n_block_cols")
        if np.unique(row_keep).size != row_keep.size:
            raise ValueError("row_block_indices must not contain duplicates")
        if np.unique(col_keep).size != col_keep.size:
            raise ValueError("col_block_indices must not contain duplicates")

        row_map = np.full((int(self.n_block_rows),), -1, dtype=np.int64)
        col_map = np.full((int(self.n_block_cols),), -1, dtype=np.int64)
        row_map[row_keep] = np.arange(int(row_keep.size), dtype=np.int64)
        col_map[col_keep] = np.arange(int(col_keep.size), dtype=np.int64)

        rows_np = np.asarray(jax.device_get(self.row_blocks), dtype=np.int64)
        cols_np = np.asarray(jax.device_get(self.col_blocks), dtype=np.int64)
        projected_rows = row_map[rows_np]
        projected_cols = col_map[cols_np]
        keep = (projected_rows >= 0) & (projected_cols >= 0)
        keep_idx = np.flatnonzero(keep).astype(np.int32, copy=False)
        projected_blocks = self.blocks[jnp.asarray(keep_idx, dtype=jnp.int32)]
        return RHS1BlockCOOOperator.from_blocks(
            row_blocks=jnp.asarray(projected_rows[keep], dtype=jnp.int32),
            col_blocks=jnp.asarray(projected_cols[keep], dtype=jnp.int32),
            blocks=projected_blocks,
            n_block_rows=int(row_keep.size),
            n_block_cols=int(col_keep.size),
        )

    def to_scipy_bsr_matrix(self):
        """Materialize the block operator as a host SciPy BSR matrix."""

        block = int(self.block_size)
        matrix = self.to_scipy_csr_matrix().tobsr(blocksize=(block, block))
        matrix.sum_duplicates()
        return matrix

    def as_block_linear_operator(
        self,
        layout: RHS1BlockLayout,
        *,
        name: str = "rhs1_block_coo_operator",
        active_layout: RHS1ActiveBlockLayout | None = None,
    ) -> RHS1BlockLinearOperator:
        """Wrap this kernel in the common block-linear-operator interface."""

        expected = int(active_layout.active_size if active_layout is not None else layout.total_size)
        if self.shape != (expected, expected):
            raise ValueError(f"block COO shape {self.shape} does not match layout size {(expected, expected)}")
        return RHS1BlockLinearOperator(
            layout=layout,
            matvec_fn=self.matvec,
            dtype=self.blocks.dtype,
            name=name,
            active_layout=active_layout,
        )

    def block_jacobi_factor(
        self,
        *,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1UniformBlockDiagonalFactor":
        """Extract a JAX block-Jacobi factor from diagonal block entries."""

        diag_blocks = min(int(self.n_block_rows), int(self.n_block_cols))
        block = int(self.block_size)
        diagonal = jnp.zeros((diag_blocks, block, block), dtype=self.blocks.dtype)
        mask = self.row_blocks == self.col_blocks
        diag_rows = self.row_blocks[mask]
        diag_vals = self.blocks[mask]
        diagonal = diagonal.at[diag_rows].add(diag_vals)
        return RHS1UniformBlockDiagonalFactor.from_dense_blocks(
            diagonal,
            regularization=float(regularization),
            damping=float(damping),
        )

    def line_jacobi_factor(
        self,
        *,
        blocks_per_line: int,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1UniformBlockDiagonalFactor":
        """Extract a grouped line-Jacobi factor from same-line block entries."""

        group = int(blocks_per_line)
        if group <= 0:
            raise ValueError("blocks_per_line must be positive")
        if int(self.n_block_rows) != int(self.n_block_cols):
            raise ValueError("line_jacobi_factor requires a square block operator")
        if int(self.n_block_rows) % group != 0:
            raise ValueError("blocks_per_line must divide the number of block rows")
        n_lines = int(self.n_block_rows) // group
        scalar_block = int(self.block_size)
        line_size = group * scalar_block
        line_blocks = jnp.zeros((n_lines, line_size, line_size), dtype=self.blocks.dtype)
        row_blocks_np = np.asarray(self.row_blocks, dtype=np.int64)
        col_blocks_np = np.asarray(self.col_blocks, dtype=np.int64)
        row_lines = row_blocks_np // group
        col_lines = col_blocks_np // group
        for i in np.flatnonzero(row_lines == col_lines):
            line = int(row_lines[int(i)])
            row_in_line = int(row_blocks_np[int(i)] % group) * scalar_block
            col_in_line = int(col_blocks_np[int(i)] % group) * scalar_block
            line_blocks = line_blocks.at[
                line,
                row_in_line : row_in_line + scalar_block,
                col_in_line : col_in_line + scalar_block,
            ].add(self.blocks[int(i)])
        return RHS1UniformBlockDiagonalFactor.from_dense_blocks(
            line_blocks,
            regularization=float(regularization),
            damping=float(damping),
        )

    def grouped_jacobi_factor(
        self,
        *,
        block_groups: Any,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1GroupedBlockDiagonalFactor":
        """Extract an indexed grouped-Jacobi factor from arbitrary block groups.

        Unlike ``line_jacobi_factor``, groups do not need to be contiguous in
        the flattened operator ordering. This supports physics groups such as
        fixed-``(L, theta)`` species/radial FP collision blocks.
        """

        groups = np.asarray(block_groups, dtype=np.int64)
        if groups.ndim != 2 or groups.shape[0] == 0 or groups.shape[1] == 0:
            raise ValueError("block_groups must have shape (n_groups, blocks_per_group)")
        if int(self.n_block_rows) != int(self.n_block_cols):
            raise ValueError("grouped_jacobi_factor requires a square block operator")
        if np.any(groups < 0) or np.any(groups >= int(self.n_block_rows)):
            raise ValueError("block_groups contains block indices outside the operator")
        if np.unique(groups.reshape((-1,))).size != groups.size:
            raise ValueError("block_groups must not contain duplicate block indices")

        n_groups, group_size = (int(groups.shape[0]), int(groups.shape[1]))
        scalar_block = int(self.block_size)
        grouped_size = group_size * scalar_block
        grouped_blocks = jnp.zeros((n_groups, grouped_size, grouped_size), dtype=self.blocks.dtype)

        block_to_group = np.full((int(self.n_block_rows),), -1, dtype=np.int64)
        block_to_offset = np.full((int(self.n_block_rows),), -1, dtype=np.int64)
        for group_id in range(n_groups):
            for offset, block_id in enumerate(groups[group_id]):
                block_to_group[int(block_id)] = int(group_id)
                block_to_offset[int(block_id)] = int(offset)

        row_blocks_np = np.asarray(self.row_blocks, dtype=np.int64)
        col_blocks_np = np.asarray(self.col_blocks, dtype=np.int64)
        for i in range(int(self.nnz_blocks)):
            row_block = int(row_blocks_np[i])
            col_block = int(col_blocks_np[i])
            group_id = int(block_to_group[row_block])
            if group_id < 0 or group_id != int(block_to_group[col_block]):
                continue
            row_offset = int(block_to_offset[row_block]) * scalar_block
            col_offset = int(block_to_offset[col_block]) * scalar_block
            grouped_blocks = grouped_blocks.at[
                group_id,
                row_offset : row_offset + scalar_block,
                col_offset : col_offset + scalar_block,
            ].add(self.blocks[i])

        return RHS1GroupedBlockDiagonalFactor.from_dense_blocks(
            grouped_blocks,
            block_groups=groups,
            n_operator_blocks=int(self.n_block_rows),
            scalar_block_size=scalar_block,
            regularization=float(regularization),
            damping=float(damping),
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly block-COO metadata."""

        return {
            "kind": "uniform_block_coo",
            "shape": tuple(int(v) for v in self.shape),
            "n_block_rows": int(self.n_block_rows),
            "n_block_cols": int(self.n_block_cols),
            "block_size": int(self.block_size),
            "nnz_blocks": int(self.nnz_blocks),
            "dtype": str(self.blocks.dtype),
            "data_nbytes": int(self.data_nbytes),
        }


@dataclass(frozen=True)
class RHS1UniformBlockDiagonalFactor:
    """Exact JAX solve for repeated small diagonal blocks."""

    blocks: jax.Array
    regularization: float = 0.0
    damping: float = 1.0

    @classmethod
    def from_dense_blocks(
        cls,
        blocks: Any,
        *,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1UniformBlockDiagonalFactor":
        """Create a factor from ``(n_blocks, block_size, block_size)`` blocks."""

        arr = jnp.asarray(blocks)
        if arr.ndim != 3 or int(arr.shape[1]) != int(arr.shape[2]):
            raise ValueError("blocks must have shape (n_blocks, block_size, block_size)")
        return cls(blocks=arr, regularization=float(regularization), damping=float(damping))

    @classmethod
    def from_dense_matrix(
        cls,
        matrix: Any,
        *,
        block_size: int,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1UniformBlockDiagonalFactor":
        """Extract diagonal blocks from a dense square matrix."""

        mat = jnp.asarray(matrix)
        block = int(block_size)
        if mat.ndim != 2 or int(mat.shape[0]) != int(mat.shape[1]):
            raise ValueError("matrix must be square")
        if block <= 0 or int(mat.shape[0]) % block != 0:
            raise ValueError("block_size must divide the matrix size")
        n_blocks = int(mat.shape[0]) // block
        blocks = jnp.stack([mat[i * block : (i + 1) * block, i * block : (i + 1) * block] for i in range(n_blocks)])
        return cls.from_dense_blocks(blocks, regularization=regularization, damping=damping)

    @property
    def n_blocks(self) -> int:
        return int(self.blocks.shape[0])

    @property
    def block_size(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def size(self) -> int:
        return int(self.n_blocks * self.block_size)

    def apply(self, rhs: Any) -> jax.Array:
        """Apply the exact block-diagonal inverse to ``rhs``."""

        rhs_vec = jnp.asarray(rhs, dtype=self.blocks.dtype)
        if rhs_vec.ndim != 1 or int(rhs_vec.shape[0]) != int(self.size):
            raise ValueError(f"rhs must have shape {(self.size,)}, got {tuple(rhs_vec.shape)}")
        blocks = self.blocks
        if float(self.regularization) != 0.0:
            eye = jnp.eye(int(self.block_size), dtype=blocks.dtype)
            blocks = blocks + jnp.asarray(float(self.regularization), dtype=blocks.dtype) * eye[None, :, :]
        rhs_blocks = rhs_vec.reshape((int(self.n_blocks), int(self.block_size)))
        solved = jax.vmap(jnp.linalg.solve)(blocks, rhs_blocks)
        return jnp.asarray(float(self.damping), dtype=blocks.dtype) * solved.reshape((-1,))

    def as_preconditioner(self) -> Callable[[Any], jax.Array]:
        """Return the factor application for Krylov preconditioner hooks."""

        return self.apply

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly factor metadata."""

        return {
            "kind": "uniform_block_diagonal",
            "n_blocks": int(self.n_blocks),
            "block_size": int(self.block_size),
            "size": int(self.size),
            "dtype": str(self.blocks.dtype),
            "regularization": float(self.regularization),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1GroupedBlockDiagonalFactor:
    """Exact JAX solves for non-contiguous grouped diagonal blocks."""

    blocks: jax.Array
    block_groups: jax.Array
    n_operator_blocks: int
    scalar_block_size: int
    regularization: float = 0.0
    damping: float = 1.0

    @classmethod
    def from_dense_blocks(
        cls,
        blocks: Any,
        *,
        block_groups: Any,
        n_operator_blocks: int,
        scalar_block_size: int,
        regularization: float = 0.0,
        damping: float = 1.0,
    ) -> "RHS1GroupedBlockDiagonalFactor":
        """Create a grouped factor from dense per-group matrices."""

        arr = jnp.asarray(blocks)
        groups = np.asarray(block_groups, dtype=np.int32)
        if groups.ndim != 2 or groups.shape[0] == 0 or groups.shape[1] == 0:
            raise ValueError("block_groups must have shape (n_groups, blocks_per_group)")
        scalar = int(scalar_block_size)
        if scalar <= 0:
            raise ValueError("scalar_block_size must be positive")
        expected_block = int(groups.shape[1]) * scalar
        if arr.ndim != 3 or int(arr.shape[0]) != int(groups.shape[0]):
            raise ValueError("blocks must have shape (n_groups, grouped_block_size, grouped_block_size)")
        if int(arr.shape[1]) != expected_block or int(arr.shape[2]) != expected_block:
            raise ValueError("blocks grouped size is inconsistent with block_groups and scalar_block_size")
        n_blocks = int(n_operator_blocks)
        if n_blocks <= 0:
            raise ValueError("n_operator_blocks must be positive")
        if np.any(groups < 0) or np.any(groups >= n_blocks):
            raise ValueError("block_groups contains block indices outside the operator")
        if np.unique(groups.reshape((-1,))).size != groups.size:
            raise ValueError("block_groups must not contain duplicate block indices")
        return cls(
            blocks=arr,
            block_groups=jnp.asarray(groups, dtype=jnp.int32),
            n_operator_blocks=n_blocks,
            scalar_block_size=scalar,
            regularization=float(regularization),
            damping=float(damping),
        )

    @property
    def n_groups(self) -> int:
        return int(self.blocks.shape[0])

    @property
    def blocks_per_group(self) -> int:
        return int(self.block_groups.shape[1])

    @property
    def block_size(self) -> int:
        return int(self.blocks.shape[1])

    @property
    def size(self) -> int:
        return int(self.n_operator_blocks * self.scalar_block_size)

    def apply(self, rhs: Any) -> jax.Array:
        """Apply the grouped block inverse, leaving ungrouped blocks unchanged."""

        rhs_vec = jnp.asarray(rhs, dtype=self.blocks.dtype)
        if rhs_vec.ndim != 1 or int(rhs_vec.shape[0]) != int(self.size):
            raise ValueError(f"rhs must have shape {(self.size,)}, got {tuple(rhs_vec.shape)}")
        blocks = self.blocks
        if float(self.regularization) != 0.0:
            eye = jnp.eye(int(self.block_size), dtype=blocks.dtype)
            blocks = blocks + jnp.asarray(float(self.regularization), dtype=blocks.dtype) * eye[None, :, :]
        rhs_blocks = rhs_vec.reshape((int(self.n_operator_blocks), int(self.scalar_block_size)))
        grouped_rhs = rhs_blocks[self.block_groups].reshape((int(self.n_groups), int(self.block_size)))
        solved = jax.vmap(jnp.linalg.solve)(blocks, grouped_rhs)
        solved_blocks = (
            jnp.asarray(float(self.damping), dtype=blocks.dtype)
            * solved.reshape((int(self.n_groups), int(self.blocks_per_group), int(self.scalar_block_size)))
        )
        out_blocks = rhs_blocks.at[self.block_groups].set(solved_blocks, unique_indices=True)
        return out_blocks.reshape((-1,))

    def as_preconditioner(self) -> Callable[[Any], jax.Array]:
        """Return the grouped factor application for Krylov hooks."""

        return self.apply

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly grouped-factor metadata."""

        return {
            "kind": "grouped_block_diagonal",
            "n_groups": int(self.n_groups),
            "blocks_per_group": int(self.blocks_per_group),
            "scalar_block_size": int(self.scalar_block_size),
            "block_size": int(self.block_size),
            "size": int(self.size),
            "dtype": str(self.blocks.dtype),
            "data_nbytes": int(self.blocks.size * self.blocks.dtype.itemsize + self.block_groups.size * self.block_groups.dtype.itemsize),
            "regularization": float(self.regularization),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1GalerkinResidualCorrection:
    """Small Galerkin residual equation for JAX-native block preconditioners."""

    basis: jax.Array
    coarse_matrix: jax.Array
    regularization: float = 0.0
    damping: float = 1.0

    @classmethod
    def from_basis(
        cls,
        *,
        operator: RHS1BlockCOOOperator,
        basis: Any,
        regularization: float = 0.0,
        damping: float = 1.0,
        max_basis_nbytes: int | None = None,
        max_coarse_size: int | None = None,
    ) -> "RHS1GalerkinResidualCorrection":
        """Build ``Z (Z^T A Z)^{-1} Z^T`` for a fixed coarse basis ``Z``."""

        basis_arr = jnp.asarray(basis)
        if basis_arr.ndim != 2 or int(basis_arr.shape[0]) != int(operator.shape[0]) or int(basis_arr.shape[1]) <= 0:
            raise ValueError(
                f"basis must have shape ({int(operator.shape[0])}, n_coarse>0), got {tuple(basis_arr.shape)}"
            )
        if max_basis_nbytes is not None and int(basis_arr.size * basis_arr.dtype.itemsize) > int(max_basis_nbytes):
            raise MemoryError("Galerkin basis exceeds max_basis_nbytes")
        n_coarse = int(basis_arr.shape[1])
        if max_coarse_size is not None and n_coarse > int(max_coarse_size):
            raise MemoryError("Galerkin coarse size exceeds max_coarse_size")
        coarse = basis_arr.T @ operator.matmat(basis_arr)
        return cls(
            basis=basis_arr,
            coarse_matrix=coarse,
            regularization=float(regularization),
            damping=float(damping),
        )

    @property
    def size(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_coarse(self) -> int:
        return int(self.basis.shape[1])

    def apply(self, residual: Any) -> jax.Array:
        """Solve the coarse residual equation and prolong the correction."""

        residual_vec = jnp.asarray(residual, dtype=self.basis.dtype)
        if residual_vec.ndim != 1 or int(residual_vec.shape[0]) != int(self.size):
            raise ValueError(f"residual must have shape {(self.size,)}, got {tuple(residual_vec.shape)}")
        coarse = self.coarse_matrix
        if float(self.regularization) != 0.0:
            eye = jnp.eye(int(self.n_coarse), dtype=coarse.dtype)
            coarse = coarse + jnp.asarray(float(self.regularization), dtype=coarse.dtype) * eye
        coefficients = jnp.linalg.solve(coarse, self.basis.T @ residual_vec)
        correction = self.basis @ coefficients
        return jnp.asarray(float(self.damping), dtype=self.basis.dtype) * correction

    def as_preconditioner(self) -> Callable[[Any], jax.Array]:
        """Return the coarse residual correction for Krylov hooks."""

        return self.apply

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly Galerkin correction metadata."""

        return {
            "kind": "galerkin_residual_correction",
            "size": int(self.size),
            "n_coarse": int(self.n_coarse),
            "dtype": str(self.basis.dtype),
            "basis_nbytes": int(self.basis.size * self.basis.dtype.itemsize),
            "coarse_nbytes": int(self.coarse_matrix.size * self.coarse_matrix.dtype.itemsize),
            "regularization": float(self.regularization),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1MatrixFreeGalerkinResidualCorrection:
    """Galerkin residual equation using callback restriction/prolongation."""

    coarse_matrix: jax.Array
    coarse_inverse: jax.Array
    restrict_fn: Callable[[Any], jax.Array]
    prolong_fn: Callable[[Any], jax.Array]
    size: int
    regularization: float = 0.0
    damping: float = 1.0
    basis_batch_size: int = 0
    basis_storage_nbytes: int = 0

    @classmethod
    def from_callbacks(
        cls,
        *,
        operator: RHS1BlockCOOOperator,
        restrict_fn: Callable[[Any], jax.Array],
        prolong_fn: Callable[[Any], jax.Array],
        n_coarse: int,
        regularization: float = 0.0,
        damping: float = 1.0,
        basis_batch_size: int = 0,
        max_coarse_size: int | None = None,
        max_basis_batch_nbytes: int | None = None,
    ) -> "RHS1MatrixFreeGalerkinResidualCorrection":
        """Build ``Z^T A Z`` without storing the dense basis matrix ``Z``."""

        coarse_size = int(n_coarse)
        if coarse_size <= 0:
            raise ValueError("n_coarse must be positive")
        if max_coarse_size is not None and coarse_size > int(max_coarse_size):
            raise MemoryError("Galerkin coarse size exceeds max_coarse_size")
        size = int(operator.shape[0])
        batch = int(basis_batch_size) if int(basis_batch_size or 0) > 0 else coarse_size
        batch = max(1, min(coarse_size, batch))

        columns: list[jax.Array] = []
        dtype = operator.blocks.dtype
        eye = jnp.eye(coarse_size, dtype=dtype)
        for start in range(0, coarse_size, batch):
            stop = min(coarse_size, start + batch)
            coarse_basis_batch = eye[:, start:stop]
            basis_batch = jnp.asarray(prolong_fn(coarse_basis_batch), dtype=dtype)
            if basis_batch.ndim != 2 or int(basis_batch.shape[0]) != size:
                raise ValueError(
                    f"prolong_fn must return shape ({size}, batch), got {tuple(basis_batch.shape)}"
                )
            if max_basis_batch_nbytes is not None:
                batch_nbytes = int(basis_batch.size * basis_batch.dtype.itemsize)
                if batch_nbytes > int(max_basis_batch_nbytes):
                    raise MemoryError("Galerkin basis batch exceeds max_basis_batch_nbytes")
            az_batch = operator.matmat(basis_batch)
            coarse_batch = jnp.asarray(restrict_fn(az_batch), dtype=dtype)
            if coarse_batch.shape != (coarse_size, stop - start):
                raise ValueError(
                    "restrict_fn must return shape "
                    f"{(coarse_size, stop - start)}, got {tuple(coarse_batch.shape)}"
                )
            columns.append(coarse_batch)
        coarse = jnp.concatenate(columns, axis=1)
        coarse_for_solve = coarse
        if float(regularization) != 0.0:
            eye = jnp.eye(coarse_size, dtype=coarse.dtype)
            coarse_for_solve = coarse_for_solve + jnp.asarray(float(regularization), dtype=coarse.dtype) * eye
        coarse_inverse = jnp.linalg.solve(coarse_for_solve, jnp.eye(coarse_size, dtype=coarse.dtype))
        return cls(
            coarse_matrix=coarse,
            coarse_inverse=coarse_inverse,
            restrict_fn=restrict_fn,
            prolong_fn=prolong_fn,
            size=size,
            regularization=float(regularization),
            damping=float(damping),
            basis_batch_size=int(batch),
            basis_storage_nbytes=0,
        )

    @property
    def n_coarse(self) -> int:
        return int(self.coarse_matrix.shape[0])

    def apply(self, residual: Any) -> jax.Array:
        """Apply the matrix-free Galerkin residual correction."""

        residual_vec = jnp.asarray(residual, dtype=self.coarse_matrix.dtype)
        if residual_vec.ndim != 1 or int(residual_vec.shape[0]) != int(self.size):
            raise ValueError(f"residual must have shape {(self.size,)}, got {tuple(residual_vec.shape)}")
        coarse_rhs = jnp.asarray(self.restrict_fn(residual_vec), dtype=self.coarse_matrix.dtype)
        if coarse_rhs.shape != (int(self.n_coarse),):
            raise ValueError(f"restrict_fn returned shape {tuple(coarse_rhs.shape)}, expected {(int(self.n_coarse),)}")
        coefficients = self.coarse_inverse @ coarse_rhs
        correction = jnp.asarray(self.prolong_fn(coefficients), dtype=self.coarse_matrix.dtype)
        if correction.shape != (int(self.size),):
            raise ValueError(f"prolong_fn returned shape {tuple(correction.shape)}, expected {(int(self.size),)}")
        return jnp.asarray(float(self.damping), dtype=self.coarse_matrix.dtype) * correction

    def as_preconditioner(self) -> Callable[[Any], jax.Array]:
        """Return the matrix-free coarse residual correction for Krylov hooks."""

        return self.apply

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly matrix-free Galerkin metadata."""

        return {
            "kind": "matrix_free_galerkin_residual_correction",
            "size": int(self.size),
            "n_coarse": int(self.n_coarse),
            "dtype": str(self.coarse_matrix.dtype),
            "basis_storage_nbytes": int(self.basis_storage_nbytes),
            "basis_batch_size": int(self.basis_batch_size),
            "coarse_nbytes": int(self.coarse_matrix.size * self.coarse_matrix.dtype.itemsize),
            "coarse_inverse_nbytes": int(self.coarse_inverse.size * self.coarse_inverse.dtype.itemsize),
            "solver_kind": "precomputed_dense_inverse",
            "regularization": float(self.regularization),
            "damping": float(self.damping),
        }


@dataclass(frozen=True)
class RHS1MatrixFreeLeastSquaresResidualCorrection:
    """Matrix-free minimum-residual coarse correction.

    The Galerkin equation ``Z^T A Z c = Z^T r`` can be unstable for small
    nonsymmetric tail blocks.  This variant builds the compact action
    ``B = A Z`` and applies the Tikhonov-regularized normal equation
    ``(B^T B + lambda I)c = B^T r``.  It is intended for small global/tail
    spaces where storing ``A Z`` is cheaper and safer than assembling a dense
    full operator.
    """

    action_matrix: jax.Array
    coarse_inverse: jax.Array
    prolong_fn: Callable[[Any], jax.Array]
    regularization: float = 0.0
    damping: float = 1.0
    basis_batch_size: int = 0
    basis_storage_nbytes: int = 0

    @classmethod
    def from_callbacks(
        cls,
        *,
        operator: RHS1BlockCOOOperator,
        prolong_fn: Callable[[Any], jax.Array],
        n_coarse: int,
        regularization: float = 0.0,
        damping: float = 1.0,
        basis_batch_size: int = 0,
        max_coarse_size: int | None = None,
        max_basis_batch_nbytes: int | None = None,
        max_action_nbytes: int | None = None,
    ) -> "RHS1MatrixFreeLeastSquaresResidualCorrection":
        """Build a bounded ``A Z`` action matrix without storing the basis."""

        coarse_size = int(n_coarse)
        if coarse_size <= 0:
            raise ValueError("n_coarse must be positive")
        if max_coarse_size is not None and coarse_size > int(max_coarse_size):
            raise MemoryError("least-squares coarse size exceeds max_coarse_size")
        size = int(operator.shape[0])
        batch = int(basis_batch_size) if int(basis_batch_size or 0) > 0 else coarse_size
        batch = max(1, min(coarse_size, batch))

        columns: list[jax.Array] = []
        dtype = operator.blocks.dtype
        eye = jnp.eye(coarse_size, dtype=dtype)
        action_nbytes = 0
        for start in range(0, coarse_size, batch):
            stop = min(coarse_size, start + batch)
            coarse_basis_batch = eye[:, start:stop]
            basis_batch = jnp.asarray(prolong_fn(coarse_basis_batch), dtype=dtype)
            if basis_batch.ndim != 2 or int(basis_batch.shape[0]) != size:
                raise ValueError(
                    f"prolong_fn must return shape ({size}, batch), got {tuple(basis_batch.shape)}"
                )
            if max_basis_batch_nbytes is not None:
                batch_nbytes = int(basis_batch.size * basis_batch.dtype.itemsize)
                if batch_nbytes > int(max_basis_batch_nbytes):
                    raise MemoryError("least-squares basis batch exceeds max_basis_batch_nbytes")
            action_batch = jnp.asarray(operator.matmat(basis_batch), dtype=dtype)
            if action_batch.shape != (size, stop - start):
                raise ValueError(
                    f"operator.matmat returned shape {tuple(action_batch.shape)}, expected {(size, stop - start)}"
                )
            action_nbytes += int(action_batch.size * action_batch.dtype.itemsize)
            if max_action_nbytes is not None and action_nbytes > int(max_action_nbytes):
                raise MemoryError("least-squares action matrix exceeds max_action_nbytes")
            columns.append(action_batch)
        action = jnp.concatenate(columns, axis=1)
        normal = action.T @ action
        if float(regularization) != 0.0:
            eye = jnp.eye(coarse_size, dtype=normal.dtype)
            normal = normal + jnp.asarray(float(regularization), dtype=normal.dtype) * eye
        coarse_inverse = jnp.linalg.solve(normal, jnp.eye(coarse_size, dtype=normal.dtype))
        return cls(
            action_matrix=action,
            coarse_inverse=coarse_inverse,
            prolong_fn=prolong_fn,
            regularization=float(regularization),
            damping=float(damping),
            basis_batch_size=int(batch),
            basis_storage_nbytes=0,
        )

    @property
    def size(self) -> int:
        return int(self.action_matrix.shape[0])

    @property
    def n_coarse(self) -> int:
        return int(self.action_matrix.shape[1])

    def apply(self, residual: Any) -> jax.Array:
        """Apply the minimum-residual coarse correction."""

        residual_vec = jnp.asarray(residual, dtype=self.action_matrix.dtype)
        if residual_vec.ndim != 1 or int(residual_vec.shape[0]) != int(self.size):
            raise ValueError(f"residual must have shape {(self.size,)}, got {tuple(residual_vec.shape)}")
        coefficients = self.coarse_inverse @ (self.action_matrix.T @ residual_vec)
        correction = jnp.asarray(self.prolong_fn(coefficients), dtype=self.action_matrix.dtype)
        if correction.shape != (int(self.size),):
            raise ValueError(f"prolong_fn returned shape {tuple(correction.shape)}, expected {(int(self.size),)}")
        return jnp.asarray(float(self.damping), dtype=self.action_matrix.dtype) * correction

    def as_preconditioner(self) -> Callable[[Any], jax.Array]:
        """Return the least-squares residual correction for Krylov hooks."""

        return self.apply

    def to_dict(self) -> dict[str, object]:
        """Return JSON-friendly least-squares correction metadata."""

        return {
            "kind": "matrix_free_least_squares_residual_correction",
            "size": int(self.size),
            "n_coarse": int(self.n_coarse),
            "dtype": str(self.action_matrix.dtype),
            "basis_storage_nbytes": int(self.basis_storage_nbytes),
            "basis_batch_size": int(self.basis_batch_size),
            "action_nbytes": int(self.action_matrix.size * self.action_matrix.dtype.itemsize),
            "coarse_inverse_nbytes": int(self.coarse_inverse.size * self.coarse_inverse.dtype.itemsize),
            "solver_kind": "precomputed_normal_inverse",
            "regularization": float(self.regularization),
            "damping": float(self.damping),
        }


def probe_rhs1_block_preconditioner(
    *,
    matvec: Callable[[Any], Any],
    rhs: Any,
    preconditioner: Callable[[Any], Any],
    x0: Any | None = None,
    target_residual_norm: float | None = None,
    target_ratio: float = 0.5,
    max_correction_ratio: float | None = None,
    factor_metadata: dict[str, object] | None = None,
) -> RHS1BlockPreconditionerProbe:
    """Apply one preconditioner step and gate it by true residual reduction."""

    rhs_vec = jnp.asarray(rhs)
    guess = jnp.zeros_like(rhs_vec) if x0 is None else jnp.asarray(x0, dtype=rhs_vec.dtype)
    if guess.shape != rhs_vec.shape:
        raise ValueError(f"x0 shape {tuple(guess.shape)} does not match rhs shape {tuple(rhs_vec.shape)}")
    residual_before = rhs_vec - jnp.asarray(matvec(guess), dtype=rhs_vec.dtype)
    correction = jnp.asarray(preconditioner(residual_before), dtype=rhs_vec.dtype)
    if correction.shape != rhs_vec.shape:
        raise ValueError(
            f"preconditioner returned shape {tuple(correction.shape)}, expected {tuple(rhs_vec.shape)}"
        )
    before = float(jnp.linalg.norm(residual_before))
    correction_norm = float(jnp.linalg.norm(correction))
    correction_ratio = (
        0.0
        if before == 0.0 and correction_norm == 0.0
        else (float("inf") if before == 0.0 else correction_norm / before)
    )
    if (
        not np.isfinite(correction_norm)
        or not np.isfinite(correction_ratio)
        or (max_correction_ratio is not None and correction_ratio > float(max_correction_ratio))
    ):
        return RHS1BlockPreconditionerProbe(
            accepted=False,
            reason="correction_amplification_exceeded",
            residual_before_norm=before,
            residual_after_norm=float("inf"),
            improvement_ratio=None,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            x_candidate=guess,
            factor_metadata={} if factor_metadata is None else dict(factor_metadata),
        )
    candidate = guess + correction
    residual_after = rhs_vec - jnp.asarray(matvec(candidate), dtype=rhs_vec.dtype)
    after = float(jnp.linalg.norm(residual_after))
    if not np.isfinite(before) or not np.isfinite(after):
        return RHS1BlockPreconditionerProbe(
            accepted=False,
            reason="nonfinite_residual",
            residual_before_norm=before,
            residual_after_norm=after,
            improvement_ratio=None,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            x_candidate=guess,
            factor_metadata={} if factor_metadata is None else dict(factor_metadata),
        )
    ratio = 0.0 if before == 0.0 and after == 0.0 else (float("inf") if before == 0.0 else after / before)
    target_ok = target_residual_norm is not None and after <= float(target_residual_norm)
    ratio_ok = ratio <= float(target_ratio)
    accepted = bool(target_ok or ratio_ok)
    if target_ok:
        reason = "target_residual_met"
    elif ratio_ok:
        reason = "residual_ratio_met"
    else:
        reason = "insufficient_residual_reduction"
    return RHS1BlockPreconditionerProbe(
        accepted=accepted,
        reason=reason,
        residual_before_norm=before,
        residual_after_norm=after,
        improvement_ratio=float(ratio),
        target_residual_norm=target_residual_norm,
        target_ratio=float(target_ratio),
        x_candidate=candidate if accepted else guess,
        factor_metadata={} if factor_metadata is None else dict(factor_metadata),
    )


def probe_rhs1_block_jacobi_preconditioner(
    *,
    operator: RHS1BlockCOOOperator,
    rhs: Any,
    x0: Any | None = None,
    regularization: float = 0.0,
    damping: float = 1.0,
    target_residual_norm: float | None = None,
    target_ratio: float = 0.5,
    max_correction_ratio: float | None = 1.0e8,
) -> RHS1BlockPreconditionerProbe:
    """Build and probe a block-Jacobi preconditioner from a block-COO operator."""

    factor = operator.block_jacobi_factor(regularization=float(regularization), damping=float(damping))
    return probe_rhs1_block_preconditioner(
        matvec=operator.matvec,
        rhs=rhs,
        preconditioner=factor.apply,
        x0=x0,
        target_residual_norm=target_residual_norm,
        target_ratio=float(target_ratio),
        max_correction_ratio=max_correction_ratio,
        factor_metadata=factor.to_dict(),
    )


def _rejected_block_probe(
    *,
    reason: str,
    rhs: Any,
    x0: Any | None,
    target_residual_norm: float | None,
    target_ratio: float,
    factor_metadata: dict[str, object] | None = None,
) -> RHS1BlockPreconditionerProbe:
    rhs_vec = jnp.asarray(rhs)
    guess = jnp.zeros_like(rhs_vec) if x0 is None else jnp.asarray(x0, dtype=rhs_vec.dtype)
    return RHS1BlockPreconditionerProbe(
        accepted=False,
        reason=str(reason),
        residual_before_norm=float("inf"),
        residual_after_norm=float("inf"),
        improvement_ratio=None,
        target_residual_norm=target_residual_norm,
        target_ratio=float(target_ratio),
        x_candidate=guess,
        factor_metadata={} if factor_metadata is None else dict(factor_metadata),
    )


def preflight_rhs1_block_jacobi_candidate(
    *,
    operator: RHS1BlockCOOOperator,
    rhs: Any,
    x0: Any | None = None,
    regularization: float = 0.0,
    damping: float = 1.0,
    target_residual_norm: float | None = None,
    target_ratio: float = 0.5,
    max_correction_ratio: float = 1.0e8,
    max_data_nbytes: int | None = None,
    require_jit_apply: bool = True,
) -> RHS1BlockPreconditionerProbe:
    """Fail-closed preflight for a JAX-native block-Jacobi candidate."""

    rhs_vec = jnp.asarray(rhs)
    if rhs_vec.ndim != 1 or int(rhs_vec.shape[0]) != int(operator.shape[0]):
        return _rejected_block_probe(
            reason="shape_mismatch",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
        )
    if max_data_nbytes is not None and int(operator.data_nbytes) > int(max_data_nbytes):
        return _rejected_block_probe(
            reason="data_budget_exceeded",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            factor_metadata=operator.to_dict(),
        )

    try:
        factor = operator.block_jacobi_factor(regularization=float(regularization), damping=float(damping))
        metadata = {**operator.to_dict(), "factor": factor.to_dict()}
        if bool(require_jit_apply):
            zero = jnp.zeros_like(rhs_vec)
            jax.block_until_ready(jax.jit(operator.matvec)(zero))
            jax.block_until_ready(jax.jit(factor.apply)(zero))
        return probe_rhs1_block_preconditioner(
            matvec=operator.matvec,
            rhs=rhs_vec,
            preconditioner=factor.apply,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            max_correction_ratio=float(max_correction_ratio),
            factor_metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed and preserve diagnostics.
        return _rejected_block_probe(
            reason=f"{type(exc).__name__}: {exc}",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            factor_metadata=operator.to_dict(),
        )


def preflight_rhs1_line_jacobi_candidate(
    *,
    operator: RHS1BlockCOOOperator,
    rhs: Any,
    blocks_per_line: int,
    x0: Any | None = None,
    regularization: float = 0.0,
    damping: float = 1.0,
    target_residual_norm: float | None = None,
    target_ratio: float = 0.5,
    max_correction_ratio: float = 1.0e8,
    max_data_nbytes: int | None = None,
    require_jit_apply: bool = True,
) -> RHS1BlockPreconditionerProbe:
    """Fail-closed preflight for a grouped line-Jacobi candidate."""

    rhs_vec = jnp.asarray(rhs)
    if rhs_vec.ndim != 1 or int(rhs_vec.shape[0]) != int(operator.shape[0]):
        return _rejected_block_probe(
            reason="shape_mismatch",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
        )
    if max_data_nbytes is not None and int(operator.data_nbytes) > int(max_data_nbytes):
        return _rejected_block_probe(
            reason="data_budget_exceeded",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            factor_metadata=operator.to_dict(),
        )

    try:
        factor = operator.line_jacobi_factor(
            blocks_per_line=int(blocks_per_line),
            regularization=float(regularization),
            damping=float(damping),
        )
        metadata = {
            **operator.to_dict(),
            "factor": factor.to_dict(),
            "blocks_per_line": int(blocks_per_line),
        }
        if bool(require_jit_apply):
            zero = jnp.zeros_like(rhs_vec)
            jax.block_until_ready(jax.jit(operator.matvec)(zero))
            jax.block_until_ready(jax.jit(factor.apply)(zero))
        return probe_rhs1_block_preconditioner(
            matvec=operator.matvec,
            rhs=rhs_vec,
            preconditioner=factor.apply,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            max_correction_ratio=float(max_correction_ratio),
            factor_metadata=metadata,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed and preserve diagnostics.
        return _rejected_block_probe(
            reason=f"{type(exc).__name__}: {exc}",
            rhs=rhs_vec,
            x0=x0,
            target_residual_norm=target_residual_norm,
            target_ratio=float(target_ratio),
            factor_metadata=operator.to_dict(),
        )


__all__ = [
    "RHS1ActiveBlockLayout",
    "RHS1ActiveFieldSplitOrdering",
    "RHS1BlockCOOBuilder",
    "RHS1BlockLayout",
    "RHS1BlockCOOOperator",
    "RHS1BlockLinearOperator",
    "RHS1BlockPreconditionerProbe",
    "RHS1GalerkinResidualCorrection",
    "RHS1MatrixFreeGalerkinResidualCorrection",
    "RHS1MatrixFreeLeastSquaresResidualCorrection",
    "RHS1GroupedBlockDiagonalFactor",
    "RHS1KineticIndices",
    "RHS1UniformBlockDiagonalFactor",
    "clear_rhs1_active_field_split_ordering_cache",
    "preflight_rhs1_block_jacobi_candidate",
    "preflight_rhs1_line_jacobi_candidate",
    "probe_rhs1_block_jacobi_preconditioner",
    "probe_rhs1_block_preconditioner",
]
