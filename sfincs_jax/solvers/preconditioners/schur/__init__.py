"""Schur-complement and moment-coupling preconditioner package."""

from __future__ import annotations

from .rhs1 import (
    RHS1SchurPreconditionerBuilders,
    build_rhs1_schur_preconditioner,
)
from .rhs1_coarse_policy import (
    ActiveNativeFieldSplitSparseCoarsePolicy,
    ActiveNativeStackPolicy,
    ActiveSparseCoarseResidualPolicy,
    resolve_active_native_field_split_sparse_coarse_policy,
    resolve_active_native_stack_policy,
    resolve_active_sparse_coarse_residual_policy,
)
from .rhs1_coarse_basis import (
    build_active_native_xell_coarse_window_basis_csc,
    build_coarse_residual_basis_csc,
    coarse_residual_config,
    coarse_surface_mode_count,
    coarse_surface_modes,
    estimate_coarse_residual_nbytes,
    estimate_xblock_tz_low_l_factor_nbytes,
    xblock_tz_low_l_config,
)
from .rhs1_full_csr import (
    RHS1StructuredFullCSRPreconditioner,
    build_block_schur_preconditioner,
    build_diagonal_schur_preconditioner,
    build_jacobi_preconditioner,
    build_x_xi_block_schur_preconditioner,
    build_xi_block_schur_preconditioner,
    estimate_x_xi_block_inverse_nbytes,
    estimate_xi_block_inverse_nbytes,
    estimate_zeta_block_inverse_nbytes,
    safe_inverse_diagonal,
)

__all__ = (
    "ActiveNativeFieldSplitSparseCoarsePolicy",
    "ActiveNativeStackPolicy",
    "ActiveSparseCoarseResidualPolicy",
    "RHS1SchurPreconditionerBuilders",
    "RHS1StructuredFullCSRPreconditioner",
    "build_active_native_xell_coarse_window_basis_csc",
    "build_block_schur_preconditioner",
    "build_coarse_residual_basis_csc",
    "build_diagonal_schur_preconditioner",
    "build_jacobi_preconditioner",
    "build_rhs1_schur_preconditioner",
    "build_x_xi_block_schur_preconditioner",
    "build_xi_block_schur_preconditioner",
    "coarse_residual_config",
    "coarse_surface_mode_count",
    "coarse_surface_modes",
    "estimate_coarse_residual_nbytes",
    "estimate_x_xi_block_inverse_nbytes",
    "estimate_xblock_tz_low_l_factor_nbytes",
    "estimate_xi_block_inverse_nbytes",
    "estimate_zeta_block_inverse_nbytes",
    "resolve_active_native_field_split_sparse_coarse_policy",
    "resolve_active_native_stack_policy",
    "resolve_active_sparse_coarse_residual_policy",
    "safe_inverse_diagonal",
    "xblock_tz_low_l_config",
)
