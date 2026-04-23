"""Shared RHSMode=1 preconditioner dispatch helpers.

This module isolates the decision ladder that maps a resolved RHSMode=1
preconditioner kind string to the corresponding builder. The goal is to keep
the orchestration in ``v3_driver.py`` thin while preserving the exact runtime
behavior and builder surface already validated by the existing regression
tests.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


Preconditioner = Callable[[Any], Any]
Builder = Callable[..., Preconditioner]


@dataclass(frozen=True)
class RHS1PreconditionerDispatchBuilders:
    """Builder bundle used by the shared RHSMode=1 dispatch helper."""

    theta_line_builder: Builder
    theta_dd_builder: Builder
    theta_schwarz_builder: Builder
    theta_line_xdiag_builder: Builder
    block_xdiag_builder: Builder
    species_block_builder: Builder
    sxblock_builder: Builder
    sxblock_tz_builder: Builder
    xblock_tz_builder: Builder
    xblock_tz_lmax_builder: Builder
    theta_zeta_builder: Builder
    xmg_builder: Builder
    pas_lite_builder: Builder
    pas_hybrid_builder: Builder
    pas_schur_builder: Builder
    pas_tz_builder: Builder
    pas_tokamak_theta_builder: Builder
    pas_ilu_builder: Builder
    zeta_line_builder: Builder
    zeta_dd_builder: Builder
    zeta_schwarz_builder: Builder
    schur_builder: Builder
    collision_builder: Builder
    block_builder: Builder
    compose_preconditioners: Callable[[Preconditioner, Preconditioner], Preconditioner]


def build_rhs1_preconditioner_from_kind(
    *,
    op,
    rhs1_precond_kind: str | None,
    builders: RHS1PreconditionerDispatchBuilders,
    reduce_full: Callable[[Any], Any] | None = None,
    expand_reduced: Callable[[Any], Any] | None = None,
    preconditioner_species: int = 1,
    preconditioner_x: int = 1,
    preconditioner_xi: int = 1,
    rhs1_xblock_tz_lmax: int | None = None,
    dd_block_theta: int = 8,
    dd_overlap_theta: int = 1,
    dd_block_zeta: int = 8,
    dd_overlap_zeta: int = 1,
    adi_sweeps: int = 2,
    emit: Callable[[int, str], None] | None = None,
) -> Preconditioner:
    """Dispatch a resolved RHSMode=1 preconditioner kind to its builder."""
    if rhs1_precond_kind == "theta_line":
        return builders.theta_line_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "theta_dd":
        if dd_overlap_theta > 0:
            return builders.theta_schwarz_builder(
                op=op,
                block=dd_block_theta,
                overlap=dd_overlap_theta,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        return builders.theta_dd_builder(
            op=op,
            block=dd_block_theta,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if rhs1_precond_kind == "theta_schwarz":
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: theta_schwarz "
                f"(block={int(dd_block_theta)}, overlap={int(dd_overlap_theta)})",
            )
        return builders.theta_schwarz_builder(
            op=op,
            block=dd_block_theta,
            overlap=dd_overlap_theta,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if rhs1_precond_kind == "theta_line_xdiag":
        precond = builders.theta_line_xdiag_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
        if op.fblock.fp is not None or op.fblock.pas is not None:
            collision_precond = builders.collision_builder(
                op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
            )
            precond = builders.compose_preconditioners(collision_precond, precond)
        return precond
    if rhs1_precond_kind == "point_xdiag":
        return builders.block_xdiag_builder(
            op=op,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
            preconditioner_xi=preconditioner_xi,
        )
    if rhs1_precond_kind == "species_block":
        return builders.species_block_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "sxblock":
        return builders.sxblock_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "sxblock_tz":
        return builders.sxblock_tz_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "xblock_tz":
        return builders.xblock_tz_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "xblock_tz_lmax":
        return builders.xblock_tz_lmax_builder(
            op=op,
            lmax=int(rhs1_xblock_tz_lmax or 0),
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if rhs1_precond_kind == "theta_zeta":
        return builders.theta_zeta_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "xmg":
        return builders.xmg_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_lite":
        return builders.pas_lite_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_hybrid":
        return builders.pas_hybrid_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_schur":
        return builders.pas_schur_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_tz":
        return builders.pas_tz_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_tokamak_theta":
        return builders.pas_tokamak_theta_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "pas_ilu":
        return builders.pas_ilu_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "zeta_line":
        return builders.zeta_line_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "zeta_dd":
        if dd_overlap_zeta > 0:
            return builders.zeta_schwarz_builder(
                op=op,
                block=dd_block_zeta,
                overlap=dd_overlap_zeta,
                reduce_full=reduce_full,
                expand_reduced=expand_reduced,
            )
        return builders.zeta_dd_builder(
            op=op,
            block=dd_block_zeta,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if rhs1_precond_kind == "zeta_schwarz":
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: zeta_schwarz "
                f"(block={int(dd_block_zeta)}, overlap={int(dd_overlap_zeta)})",
            )
        return builders.zeta_schwarz_builder(
            op=op,
            block=dd_block_zeta,
            overlap=dd_overlap_zeta,
            reduce_full=reduce_full,
            expand_reduced=expand_reduced,
        )
    if rhs1_precond_kind == "schur":
        return builders.schur_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "collision":
        return builders.collision_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
    if rhs1_precond_kind == "adi":
        pre_theta = builders.theta_line_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )
        pre_zeta = builders.zeta_line_builder(
            op=op, reduce_full=reduce_full, expand_reduced=expand_reduced
        )

        def preconditioner(v):
            out = v
            for _ in range(max(1, int(adi_sweeps))):
                out = pre_zeta(pre_theta(out))
            return out

        return preconditioner
    return builders.block_builder(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        preconditioner_species=preconditioner_species,
        preconditioner_x=preconditioner_x,
        preconditioner_xi=preconditioner_xi,
    )


__all__ = [
    "RHS1PreconditionerDispatchBuilders",
    "build_rhs1_preconditioner_from_kind",
]
