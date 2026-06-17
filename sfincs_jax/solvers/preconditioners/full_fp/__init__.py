"""Full Fokker-Planck preconditioner package."""

from __future__ import annotations

from .species_blocks import (
    build_rhs1_species_block_preconditioner,
    build_rhs1_species_xblock_preconditioner,
)
from .structured_fblock import (
    build_rhs1_structured_fblock_angular_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner,
    build_rhs1_structured_fblock_fp_moment_schur_preconditioner,
    build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner,
    build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner,
    build_rhs1_structured_fblock_jacobi_preconditioner,
    build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner,
)

__all__ = (
    "build_rhs1_species_block_preconditioner",
    "build_rhs1_species_xblock_preconditioner",
    "build_rhs1_structured_fblock_angular_jacobi_preconditioner",
    "build_rhs1_structured_fblock_fp_coupled_moment_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_lowmode_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_moment_schur_preconditioner",
    "build_rhs1_structured_fblock_fp_radial_jacobi_preconditioner",
    "build_rhs1_structured_fblock_fp_tail_coupled_schur_preconditioner",
    "build_rhs1_structured_fblock_jacobi_preconditioner",
    "build_rhs1_structured_fblock_xi_angular_jacobi_preconditioner",
)
