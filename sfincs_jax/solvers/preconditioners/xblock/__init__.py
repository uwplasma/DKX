"""Speed-line and x-block preconditioner package."""

from __future__ import annotations

from .active_projected import (
    active_positions_for_full_indices,
    build_active_fortran_v3_reduced_native_stack_preconditioner,
    build_active_projected_bounded_native_stack_preconditioner,
    build_active_projected_global_field_split_schur_preconditioner,
    build_active_projected_multiline_field_split_base_preconditioner,
    build_active_projected_angular_line_preconditioner,
    build_active_projected_diagonal_schur_preconditioner,
    build_active_projected_native_indexed_schwarz_preconditioner,
    build_active_projected_overlap_schwarz_preconditioner,
    build_active_projected_xell_kinetic_line_preconditioner,
    build_active_projected_xblock_preconditioner,
)
from .block_jacobi import (
    build_rhs1_sxblock_tz_preconditioner,
    build_rhs1_xblock_tz_lmax_preconditioner,
    build_rhs1_xblock_tz_preconditioner,
)
from .low_l_schur import (
    build_native_xell_kinetic_preconditioner,
    build_native_xell_tail_schur_preconditioner,
    build_xblock_tz_low_l_coarse_residual_preconditioner,
    build_xblock_tz_low_l_schur_preconditioner,
    xblock_tz_low_l_indices,
)
from .radial import (
    build_rhs1_xmg_preconditioner,
    build_rhs1_xupwind_preconditioner,
)
from .tz_sparse import (
    assemble_rhsmode1_fp_xblock_tz_sparse_matrix,
    assemble_selected_theta_tz_operator,
    assemble_selected_zeta_tz_operator,
    build_rhs1_sxblock_tz_sparse_host_preconditioner,
    build_rhs1_xblock_tz_sparse_preconditioner,
    compute_rhs1_sxblock_tz_sparse_host_seed,
    get_rhsmode1_fp_xblock_assembled_host_cache,
    rhsmode1_fp_xblock_assembled_host_allowed,
    rhsmode1_fp_xblock_species_decoupled_for_host_assembly,
    rhsmode1_fp_xblock_tz_sparse_diagonal,
    rhsmode1_host_factor_probe_ok,
    rhsmode1_precond_cache_key,
    rhsmode1_xblock_sparse_lu_default_max,
    safe_inverse_diagonal_np,
)

__all__ = (
    "active_positions_for_full_indices",
    "assemble_rhsmode1_fp_xblock_tz_sparse_matrix",
    "assemble_selected_theta_tz_operator",
    "assemble_selected_zeta_tz_operator",
    "build_active_fortran_v3_reduced_native_stack_preconditioner",
    "build_active_projected_bounded_native_stack_preconditioner",
    "build_active_projected_global_field_split_schur_preconditioner",
    "build_active_projected_multiline_field_split_base_preconditioner",
    "build_active_projected_angular_line_preconditioner",
    "build_active_projected_diagonal_schur_preconditioner",
    "build_active_projected_native_indexed_schwarz_preconditioner",
    "build_active_projected_overlap_schwarz_preconditioner",
    "build_active_projected_xell_kinetic_line_preconditioner",
    "build_active_projected_xblock_preconditioner",
    "build_rhs1_sxblock_tz_preconditioner",
    "build_rhs1_sxblock_tz_sparse_host_preconditioner",
    "build_rhs1_xmg_preconditioner",
    "build_rhs1_xupwind_preconditioner",
    "build_rhs1_xblock_tz_lmax_preconditioner",
    "build_rhs1_xblock_tz_preconditioner",
    "build_rhs1_xblock_tz_sparse_preconditioner",
    "build_native_xell_kinetic_preconditioner",
    "build_native_xell_tail_schur_preconditioner",
    "build_xblock_tz_low_l_coarse_residual_preconditioner",
    "build_xblock_tz_low_l_schur_preconditioner",
    "compute_rhs1_sxblock_tz_sparse_host_seed",
    "get_rhsmode1_fp_xblock_assembled_host_cache",
    "rhsmode1_fp_xblock_assembled_host_allowed",
    "rhsmode1_fp_xblock_species_decoupled_for_host_assembly",
    "rhsmode1_fp_xblock_tz_sparse_diagonal",
    "rhsmode1_host_factor_probe_ok",
    "rhsmode1_precond_cache_key",
    "rhsmode1_xblock_sparse_lu_default_max",
    "safe_inverse_diagonal_np",
    "xblock_tz_low_l_indices",
)
