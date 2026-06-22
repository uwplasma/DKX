"""RHSMode=2/3 transport output-schema helpers."""

from __future__ import annotations

from typing import Any

import numpy as np


def transport_solver_diagnostic_arrays(
    result: Any,
    n_rhs: int,
) -> dict[str, np.ndarray]:
    """Return absolute and relative transport residual diagnostics.

    The writer stores one residual, one RHS norm, and one relative residual per
    ``whichRHS`` solve, plus max summaries. Missing RHS entries are represented
    by ``NaN`` so partial debug artifacts remain explicit instead of silently
    looking converged.
    """

    residuals_by_rhs = getattr(result, "residual_norms_by_rhs", None) or {}
    rhs_norms_by_rhs = getattr(result, "rhs_norms_by_rhs", None) or {}
    residuals = np.asarray(
        [
            float(np.asarray(residuals_by_rhs.get(i, np.nan), dtype=np.float64))
            for i in range(1, int(n_rhs) + 1)
        ],
        dtype=np.float64,
    )
    rhs_norms = np.asarray(
        [
            float(np.asarray(rhs_norms_by_rhs.get(i, np.nan), dtype=np.float64))
            for i in range(1, int(n_rhs) + 1)
        ],
        dtype=np.float64,
    )
    rel = np.full_like(residuals, np.nan, dtype=np.float64)
    valid = np.isfinite(residuals) & np.isfinite(rhs_norms) & (rhs_norms > 0.0)
    rel[valid] = residuals[valid] / rhs_norms[valid]
    finite_residuals = residuals[np.isfinite(residuals)]
    finite_rel = rel[np.isfinite(rel)]
    return {
        "transportResidualNorms": residuals,
        "transportRhsNorms": rhs_norms,
        "transportRelativeResidualNorms": rel,
        "transportMaxResidualNorm": np.asarray(
            float(np.max(finite_residuals)) if finite_residuals.size else float("nan"),
            dtype=np.float64,
        ),
        "transportMaxRelativeResidualNorm": np.asarray(
            float(np.max(finite_rel)) if finite_rel.size else float("nan"),
            dtype=np.float64,
        ),
    }


__all__ = ["transport_solver_diagnostic_arrays"]
