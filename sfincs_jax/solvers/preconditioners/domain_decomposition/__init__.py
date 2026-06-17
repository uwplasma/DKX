"""Domain-decomposition and additive-Schwarz preconditioner package."""

from __future__ import annotations

from .line_blocks import (
    build_rhs1_theta_dd_preconditioner,
    build_rhs1_theta_line_preconditioner,
    build_rhs1_theta_line_xdiag_preconditioner,
    build_rhs1_theta_schwarz_preconditioner,
    build_rhs1_theta_zeta_preconditioner,
    build_rhs1_zeta_dd_preconditioner,
    build_rhs1_zeta_line_preconditioner,
    build_rhs1_zeta_schwarz_preconditioner,
)

__all__ = (
    "build_rhs1_theta_dd_preconditioner",
    "build_rhs1_theta_line_preconditioner",
    "build_rhs1_theta_line_xdiag_preconditioner",
    "build_rhs1_theta_schwarz_preconditioner",
    "build_rhs1_theta_zeta_preconditioner",
    "build_rhs1_zeta_dd_preconditioner",
    "build_rhs1_zeta_line_preconditioner",
    "build_rhs1_zeta_schwarz_preconditioner",
)
