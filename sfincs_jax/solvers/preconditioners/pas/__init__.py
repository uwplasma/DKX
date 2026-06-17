"""Pitch-angle-scattering preconditioner package."""

from __future__ import annotations

from .composite import (
    RHS1PasCompositeBuilders,
    build_rhs1_pas_hybrid_preconditioner,
    build_rhs1_pas_lite_preconditioner,
    build_rhs1_pas_schur_preconditioner,
    compose_preconditioners,
)
from .xblock_ilu import (
    build_rhs1_pas_xblock_ilu_preconditioner,
    rhsmode1_pas_xblock_precond_cache_key,
)

__all__ = (
    "RHS1PasCompositeBuilders",
    "build_rhs1_pas_hybrid_preconditioner",
    "build_rhs1_pas_lite_preconditioner",
    "build_rhs1_pas_schur_preconditioner",
    "build_rhs1_pas_xblock_ilu_preconditioner",
    "compose_preconditioners",
    "rhsmode1_pas_xblock_precond_cache_key",
)
