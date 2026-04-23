"""RHSMode=1 acceptance and factor-probe policy helpers.

This module holds small solve-path gates that are not matrix assembly or Krylov
logic: accepting a large explicit PAS solution after a bounded residual, and
probing host x-block factors for obviously unsafe amplification.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from .pas_smoother import pas_fast_accept as _pas_fast_accept_metric


_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_token(name: str) -> str:
    return str(os.environ.get(name, "")).strip().lower()


def _env_int(name: str, default: int) -> int:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    raw = str(os.environ.get(name, "")).strip()
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def rhs1_pas_fast_accept(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a large explicit CPU PAS solve may be accepted quickly."""
    env = _env_token("SFINCS_JAX_PAS_FAST_ACCEPT")
    if env in _FALSE_VALUES:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.pas is None:
        return False
    return _pas_fast_accept_metric(
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        min_size=_env_int("SFINCS_JAX_PAS_FAST_ACCEPT_MIN", 20000),
        ratio=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", 1.0e2),
        abs_floor=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", 1.0e-7),
    )


def rhs1_host_factor_probe_ok(*, factor: object | None, block_size: int) -> bool:
    """Return whether a host factor solve passes a bounded unit-vector probe."""
    if factor is None or int(block_size) <= 0:
        return False
    probe_max = max(_env_float("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", 1.0e8), 1.0)
    probe = np.ones((int(block_size),), dtype=np.float64)
    try:
        solved = np.asarray(factor.solve(probe), dtype=np.float64).reshape((-1,))
    except Exception:
        return False
    if solved.shape != probe.shape or not np.all(np.isfinite(solved)):
        return False
    ratio = float(np.linalg.norm(solved)) / max(float(np.linalg.norm(probe)), 1.0e-300)
    return np.isfinite(ratio) and ratio <= probe_max


__all__ = [
    "rhs1_host_factor_probe_ok",
    "rhs1_pas_fast_accept",
]
