"""Compatibility facade for RHSMode=1 strong-preconditioner fallback builds."""

from __future__ import annotations

from .problems.profile_response.preconditioner_build import (
    RHS1StrongPreconditionerFamilyBuilders,
    build_rhs1_strong_preconditioner_full_from_kind,
    build_rhs1_strong_preconditioner_reduced_from_kind,
    resolve_rhs1_strong_preconditioner_kind_for_build,
)

__all__ = [
    "RHS1StrongPreconditionerFamilyBuilders",
    "build_rhs1_strong_preconditioner_full_from_kind",
    "build_rhs1_strong_preconditioner_reduced_from_kind",
    "resolve_rhs1_strong_preconditioner_kind_for_build",
]
