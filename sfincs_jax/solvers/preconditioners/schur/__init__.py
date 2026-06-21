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

__all__ = (
    "ActiveNativeFieldSplitSparseCoarsePolicy",
    "ActiveNativeStackPolicy",
    "ActiveSparseCoarseResidualPolicy",
    "RHS1SchurPreconditionerBuilders",
    "build_rhs1_schur_preconditioner",
    "resolve_active_native_field_split_sparse_coarse_policy",
    "resolve_active_native_stack_policy",
    "resolve_active_sparse_coarse_residual_policy",
)
