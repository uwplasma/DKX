"""Compressed RHSMode=1 pitch-space layout helpers.

SFINCS Fortran v3 does not store inactive Legendre modes in the transport
unknown vector.  For each speed node ``x``, only ``Nxi_for_x[x]`` pitch modes
are retained, then explicit source/constraint/Phi1 tail blocks are appended.
This module makes that compressed layout first-class so reduced-Pmat assembly
and factor planning do not need to start from a rectangular full ``(x, L)``
CSR and filter it after the fact.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np


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

    def full_kinetic_index(self, species: int, x_index: int, ell: int, theta: int, zeta: int) -> int:
        """Return the rectangular full-system f-block index for one kinetic DOF."""

        return int(
            (((int(species) * self.n_x + int(x_index)) * self.n_xi + int(ell)) * self.n_theta + int(theta))
            * self.n_zeta
            + int(zeta)
        )

    def reduced_kinetic_index(self, species: int, x_index: int, ell: int, theta: int, zeta: int) -> int:
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
        raise ValueError(f"f_size={f_size} does not match rectangular kinetic size {expected_f_size}")
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
    """Infer a compressed active-pitch layout from an active full-index vector.

    This is the bridge from legacy active-index helpers to the native
    reduced-Pmat stack.  It validates that active kinetic indices form the
    Fortran v3 prefix structure ``ell < Nxi_for_x[x]`` for every species and
    angle, and that any retained tail is the explicit contiguous tail block.
    """

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


__all__ = [
    "RHS1CompressedPitchLayout",
    "build_rhs1_compressed_pitch_layout",
    "infer_rhs1_compressed_pitch_layout_from_active_indices",
]
