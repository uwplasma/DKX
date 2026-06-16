"""Pitch-angle-scattering preconditioner package."""

from __future__ import annotations

from .xblock_ilu import build_rhs1_pas_xblock_ilu_preconditioner

__all__ = ("build_rhs1_pas_xblock_ilu_preconditioner",)
