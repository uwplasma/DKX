from __future__ import annotations

"""Transport solve handoff and retry policy helpers.

The transport solve loop has separate reduced/full execution branches, but the
retry thresholds and RHSMode=3 polish settings should remain identical across
both. This module centralizes those small policy decisions without owning the
actual solver calls.
"""

from dataclasses import dataclass
import os
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TransportPolishConfig:
    """Resolved RHSMode=3 polish settings."""

    enabled: bool
    threshold: float
    ratio: float
    abs_tol: float
    restart: int
    maxiter: int


def transport_residual_value(result: Any) -> float:
    """Return a finite-comparable residual value for a solver result."""
    val = float(result.residual_norm)
    return val if np.isfinite(val) else float("inf")


def transport_result_needs_retry(
    result: Any,
    target: float,
    *,
    result_is_finite,
) -> bool:
    """Shared retry gate for transport solve results."""
    return (not bool(result_is_finite(result))) or (transport_residual_value(result) > float(target))


def transport_candidate_is_better(*, candidate: Any, current: Any) -> bool:
    """Compare candidate and current results using the transport residual metric."""
    return transport_residual_value(candidate) < transport_residual_value(current)


def transport_polish_config_from_env(
    *,
    rhs_mode: int,
    residual_norm: float,
    target: float,
    gmres_restart: int,
    maxiter: int | None,
) -> TransportPolishConfig:
    """Resolve the RHSMode=3 GMRES polish trigger and iteration budget."""
    polish_ratio_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_RATIO", "").strip()
    polish_abs_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_ABS", "").strip()
    polish_restart_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_RESTART", "").strip()
    polish_maxiter_env = os.environ.get("SFINCS_JAX_TRANSPORT_POLISH_MAXITER", "").strip()
    try:
        polish_ratio = float(polish_ratio_env) if polish_ratio_env else 2.0
    except ValueError:
        polish_ratio = 2.0
    try:
        polish_abs = float(polish_abs_env) if polish_abs_env else 1e-8
    except ValueError:
        polish_abs = 1e-8
    polish_thresh = max(float(target) * float(polish_ratio), float(polish_abs))
    base_restart = max(int(gmres_restart), 40)
    base_maxiter = int(maxiter) if maxiter is not None else 800
    try:
        polish_restart = int(polish_restart_env) if polish_restart_env else max(base_restart * 2, 80)
    except ValueError:
        polish_restart = max(base_restart * 2, 80)
    try:
        polish_maxiter = int(polish_maxiter_env) if polish_maxiter_env else max(base_maxiter * 2, 1200)
    except ValueError:
        polish_maxiter = max(base_maxiter * 2, 1200)
    enabled = int(rhs_mode) == 3 and float(residual_norm) > float(polish_thresh)
    return TransportPolishConfig(
        enabled=bool(enabled),
        threshold=float(polish_thresh),
        ratio=float(polish_ratio),
        abs_tol=float(polish_abs),
        restart=int(polish_restart),
        maxiter=int(polish_maxiter),
    )


__all__ = [
    "TransportPolishConfig",
    "transport_candidate_is_better",
    "transport_polish_config_from_env",
    "transport_residual_value",
    "transport_result_needs_retry",
]
