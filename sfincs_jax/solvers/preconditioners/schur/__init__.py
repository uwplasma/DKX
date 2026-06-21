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

__all__ = (
    "ActiveNativeFieldSplitSparseCoarsePolicy",
    "ActiveNativeStackPolicy",
    "ActiveSparseCoarseResidualPolicy",
    "RHS1SchurPreconditionerBuilders",
    "build_active_native_xell_coarse_window_basis_csc",
    "build_coarse_residual_basis_csc",
    "build_rhs1_schur_preconditioner",
    "coarse_residual_config",
    "coarse_surface_mode_count",
    "coarse_surface_modes",
    "estimate_coarse_residual_nbytes",
    "estimate_xblock_tz_low_l_factor_nbytes",
    "resolve_active_native_field_split_sparse_coarse_policy",
    "resolve_active_native_stack_policy",
    "resolve_active_sparse_coarse_residual_policy",
    "xblock_tz_low_l_config",
)
