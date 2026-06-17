"""Domain-decomposition and additive-Schwarz preconditioner package."""

from __future__ import annotations

from .line_blocks import build_rhs1_zeta_line_preconditioner

__all__ = ("build_rhs1_zeta_line_preconditioner",)
