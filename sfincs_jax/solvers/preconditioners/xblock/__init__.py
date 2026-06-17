"""Speed-line and x-block preconditioner package."""

from __future__ import annotations

from .block_jacobi import (
    build_rhs1_sxblock_tz_preconditioner,
    build_rhs1_xblock_tz_lmax_preconditioner,
    build_rhs1_xblock_tz_preconditioner,
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
    "assemble_rhsmode1_fp_xblock_tz_sparse_matrix",
    "assemble_selected_theta_tz_operator",
    "assemble_selected_zeta_tz_operator",
    "build_rhs1_sxblock_tz_preconditioner",
    "build_rhs1_sxblock_tz_sparse_host_preconditioner",
    "build_rhs1_xmg_preconditioner",
    "build_rhs1_xupwind_preconditioner",
    "build_rhs1_xblock_tz_lmax_preconditioner",
    "build_rhs1_xblock_tz_preconditioner",
    "build_rhs1_xblock_tz_sparse_preconditioner",
    "compute_rhs1_sxblock_tz_sparse_host_seed",
    "get_rhsmode1_fp_xblock_assembled_host_cache",
    "rhsmode1_fp_xblock_assembled_host_allowed",
    "rhsmode1_fp_xblock_species_decoupled_for_host_assembly",
    "rhsmode1_fp_xblock_tz_sparse_diagonal",
    "rhsmode1_host_factor_probe_ok",
    "rhsmode1_precond_cache_key",
    "rhsmode1_xblock_sparse_lu_default_max",
    "safe_inverse_diagonal_np",
)
