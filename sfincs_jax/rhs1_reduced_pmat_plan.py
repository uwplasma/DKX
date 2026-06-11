"""Symbolic reduced-Pmat elimination plans for RHSMode=1.

The production RHSMode=1 preconditioner needs a native analogue of the
Fortran/PETSc ``whichMatrix=0`` sparse-direct path.  This module provides the
symbolic half of that stack: a bounded ordering over the compressed active
pitch layout, with kinetic interiors eliminated before selected kinetic
separators and explicit source/constraint/Phi1 tail variables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .rhs1_compressed_layout import RHS1CompressedPitchLayout, build_rhs1_compressed_pitch_layout


@dataclass(frozen=True)
class RHS1ReducedPmatGroup:
    """One contiguous symbolic group in reduced active-pitch ordering."""

    name: str
    kind: str
    indices: np.ndarray

    @property
    def size(self) -> int:
        return int(self.indices.size)

    @property
    def dense_lu_nbytes_estimate(self) -> int:
        return int(self.size * self.size * 8)


@dataclass(frozen=True)
class RHS1ReducedPmatEliminationPlan:
    """Bounded symbolic ordering for direct reduced-Pmat assembly/factorization."""

    layout: RHS1CompressedPitchLayout
    interior_groups: tuple[RHS1ReducedPmatGroup, ...]
    separator_group: RHS1ReducedPmatGroup
    tail_group: RHS1ReducedPmatGroup
    root_group: RHS1ReducedPmatGroup
    permutation: np.ndarray
    inverse_permutation: np.ndarray
    selected_separator_ells: tuple[int, ...]
    selected_separator_x_indices: tuple[int, ...]
    max_interior_group_size: int
    max_separator_size: int

    @property
    def interior_size(self) -> int:
        return int(sum(group.size for group in self.interior_groups))

    @property
    def separator_size(self) -> int:
        return int(self.separator_group.size)

    @property
    def tail_size(self) -> int:
        return int(self.tail_group.size)

    @property
    def root_size(self) -> int:
        return int(self.root_group.size)

    @property
    def root_dense_nbytes_estimate(self) -> int:
        return int(self.root_group.dense_lu_nbytes_estimate)

    @property
    def max_interior_dense_nbytes_estimate(self) -> int:
        if not self.interior_groups:
            return 0
        return int(max(group.dense_lu_nbytes_estimate for group in self.interior_groups))

    def metadata(self) -> dict[str, object]:
        return {
            "reduced_size": int(self.layout.reduced_size),
            "interior_group_count": int(len(self.interior_groups)),
            "interior_size": int(self.interior_size),
            "separator_size": int(self.separator_size),
            "tail_size": int(self.tail_size),
            "root_size": int(self.root_size),
            "selected_separator_ells": tuple(int(v) for v in self.selected_separator_ells),
            "selected_separator_x_indices": tuple(int(v) for v in self.selected_separator_x_indices),
            "max_interior_group_size": int(self.max_interior_group_size),
            "max_separator_size": int(self.max_separator_size),
            "max_interior_dense_nbytes_estimate": int(self.max_interior_dense_nbytes_estimate),
            "root_dense_nbytes_estimate": int(self.root_dense_nbytes_estimate),
        }


def _normalize_separator_ells(values: Iterable[int], *, n_xi: int) -> tuple[int, ...]:
    out = sorted({int(v) for v in values if 0 <= int(v) < int(n_xi)})
    return tuple(out)


def _selected_x_indices(n_x: int, *, n_theta: int, n_zeta: int, ell_count: int, max_separator_size: int) -> tuple[int, ...]:
    if n_x <= 0 or ell_count <= 0 or max_separator_size <= 0:
        return tuple()
    per_x_size = max(1, int(ell_count) * int(n_theta) * int(n_zeta))
    max_x = max(1, int(max_separator_size) // per_x_size)
    if max_x >= int(n_x):
        return tuple(range(int(n_x)))
    # Always keep the endpoints, then add approximately uniform interior x
    # separators.  This mirrors the purpose of nested dissection separators
    # without committing to a numeric factor implementation here.
    raw = np.linspace(0, int(n_x) - 1, num=max_x, dtype=np.int64)
    return tuple(int(v) for v in np.unique(raw))


def _split_group_indices(indices: np.ndarray, *, name_prefix: str, kind: str, max_size: int) -> tuple[RHS1ReducedPmatGroup, ...]:
    indices = np.asarray(indices, dtype=np.int64)
    if indices.size == 0:
        return tuple()
    max_size = max(1, int(max_size))
    groups: list[RHS1ReducedPmatGroup] = []
    for start in range(0, int(indices.size), max_size):
        chunk = indices[start : start + max_size]
        groups.append(
            RHS1ReducedPmatGroup(
                name=f"{name_prefix}_{len(groups)}",
                kind=kind,
                indices=np.asarray(chunk, dtype=np.int64),
            )
        )
    return tuple(groups)


def build_rhs1_reduced_pmat_elimination_plan(
    op_or_layout: object,
    *,
    separator_ells: Iterable[int] = (0,),
    max_interior_group_size: int = 32768,
    max_separator_size: int = 8192,
) -> RHS1ReducedPmatEliminationPlan:
    """Build a bounded symbolic plan over compressed RHSMode=1 active DOFs.

    Parameters
    ----------
    op_or_layout:
      Either a :class:`RHS1CompressedPitchLayout` or an operator accepted by
      :func:`build_rhs1_compressed_pitch_layout`.
    separator_ells:
      Pitch modes retained in the Schur root candidate.  ``ell=0`` is the
      default because density/source/profile moments couple most directly to
      that mode.
    max_interior_group_size:
      Maximum rows in one kinetic interior group before symbolic splitting.
    max_separator_size:
      Maximum selected kinetic separator rows.  Tail rows are appended to the
      root even if this bound is saturated.
    """

    layout = (
        op_or_layout
        if isinstance(op_or_layout, RHS1CompressedPitchLayout)
        else build_rhs1_compressed_pitch_layout(op_or_layout)
    )
    separator_ells_norm = _normalize_separator_ells(separator_ells, n_xi=layout.n_xi)
    selected_x = _selected_x_indices(
        layout.n_x,
        n_theta=layout.n_theta,
        n_zeta=layout.n_zeta,
        ell_count=len(separator_ells_norm),
        max_separator_size=max_separator_size,
    )

    separator_reduced: list[int] = []
    selected_x_set = set(selected_x)
    selected_ell_set = set(separator_ells_norm)
    for species in range(layout.n_species):
        for x_index in selected_x:
            n_active_l = int(layout.nxi_for_x[x_index])
            for ell in separator_ells_norm:
                if ell >= n_active_l:
                    continue
                for theta in range(layout.n_theta):
                    for zeta in range(layout.n_zeta):
                        separator_reduced.append(layout.reduced_kinetic_index(species, x_index, ell, theta, zeta))

    separator_indices = np.asarray(sorted(set(separator_reduced)), dtype=np.int64)
    all_kinetic = np.arange(layout.kinetic_active_size, dtype=np.int64)
    if separator_indices.size:
        interior_mask = np.ones((layout.kinetic_active_size,), dtype=bool)
        interior_mask[separator_indices] = False
        interior_reduced = all_kinetic[interior_mask]
    else:
        interior_reduced = all_kinetic

    interior_groups: list[RHS1ReducedPmatGroup] = []
    for species in range(layout.n_species):
        for x_index in range(layout.n_x):
            block = np.arange(
                layout.species_x_reduced_slice(species, x_index).start,
                layout.species_x_reduced_slice(species, x_index).stop,
                dtype=np.int64,
            )
            if x_index in selected_x_set and separator_ells_norm:
                keep = np.ones((block.size,), dtype=bool)
                for ell in selected_ell_set:
                    if ell >= int(layout.nxi_for_x[x_index]):
                        continue
                    ell_start = (
                        layout.reduced_kinetic_index(species, x_index, ell, 0, 0)
                        - layout.species_x_reduced_slice(species, x_index).start
                    )
                    keep[ell_start : ell_start + layout.n_theta * layout.n_zeta] = False
                block = block[keep]
            interior_groups.extend(
                _split_group_indices(
                    block,
                    name_prefix=f"s{species}_x{x_index}",
                    kind="kinetic_interior",
                    max_size=max_interior_group_size,
                )
            )

    interior_concat = np.concatenate([group.indices for group in interior_groups]) if interior_groups else np.asarray([], dtype=np.int64)
    if not np.array_equal(np.sort(interior_concat), np.sort(interior_reduced)):
        raise ValueError("reduced-Pmat interior groups do not cover the expected kinetic interior")

    tail_indices = np.arange(layout.tail_reduced_start, layout.reduced_size, dtype=np.int64)
    separator_group = RHS1ReducedPmatGroup("kinetic_separator", "kinetic_separator", separator_indices)
    tail_group = RHS1ReducedPmatGroup("tail", "tail", tail_indices)
    root_indices = np.concatenate([separator_indices, tail_indices])
    root_group = RHS1ReducedPmatGroup("schur_root", "schur_root", root_indices)

    permutation = np.concatenate([interior_concat, root_indices]).astype(np.int64, copy=False)
    if permutation.size != layout.reduced_size or np.unique(permutation).size != layout.reduced_size:
        raise ValueError("reduced-Pmat symbolic permutation is not a complete permutation")
    inverse = np.empty_like(permutation)
    inverse[permutation] = np.arange(permutation.size, dtype=np.int64)

    return RHS1ReducedPmatEliminationPlan(
        layout=layout,
        interior_groups=tuple(interior_groups),
        separator_group=separator_group,
        tail_group=tail_group,
        root_group=root_group,
        permutation=permutation,
        inverse_permutation=inverse,
        selected_separator_ells=separator_ells_norm,
        selected_separator_x_indices=selected_x,
        max_interior_group_size=max(1, int(max_interior_group_size)),
        max_separator_size=max(1, int(max_separator_size)),
    )


__all__ = [
    "RHS1ReducedPmatEliminationPlan",
    "RHS1ReducedPmatGroup",
    "build_rhs1_reduced_pmat_elimination_plan",
]
