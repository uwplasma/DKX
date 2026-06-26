"""Symbolic sparse ordering and native factorization preconditioner package."""

from __future__ import annotations

from .host_factor import (
    RHS1FullSystemMatrixFreeOperatorAdapter,
    build_sparse_ilu_from_matvec,
    factorize_sparse_matrix_csr_host,
)
from .active_factors import (
    build_active_filtered_sparse_factor_preconditioner,
    build_active_global_sparse_factor_preconditioner,
    build_active_scaled_sparse_factor_preconditioner,
)
from .profile_response import (
    active_fortran_v3_reduced_preconditioner_matrix,
    build_active_fortran_v3_reduced_sparse_factor_preconditioner,
    estimate_spilu_factor_nbytes,
    parse_active_fortran_v3_support_mode_candidates,
    select_active_fortran_v3_reduced_support_mode_preconditioner,
    sparse_equilibration_scale,
    sparse_lu_factor_nbytes,
)

__all__ = (
    "RHS1FullSystemMatrixFreeOperatorAdapter",
    "active_fortran_v3_reduced_preconditioner_matrix",
    "build_active_filtered_sparse_factor_preconditioner",
    "build_active_global_sparse_factor_preconditioner",
    "build_active_scaled_sparse_factor_preconditioner",
    "build_sparse_ilu_from_matvec",
    "build_active_fortran_v3_reduced_sparse_factor_preconditioner",
    "estimate_spilu_factor_nbytes",
    "factorize_sparse_matrix_csr_host",
    "parse_active_fortran_v3_support_mode_candidates",
    "select_active_fortran_v3_reduced_support_mode_preconditioner",
    "sparse_equilibration_scale",
    "sparse_lu_factor_nbytes",
)
