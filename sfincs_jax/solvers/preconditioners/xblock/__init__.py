"""Speed-line and x-block preconditioner package."""

from __future__ import annotations

from .tz_sparse import (
    assemble_selected_theta_tz_operator,
    assemble_selected_zeta_tz_operator,
    build_rhs1_xblock_tz_sparse_preconditioner,
    safe_inverse_diagonal_np,
)

__all__ = (
    "assemble_selected_theta_tz_operator",
    "assemble_selected_zeta_tz_operator",
    "build_rhs1_xblock_tz_sparse_preconditioner",
    "safe_inverse_diagonal_np",
)
