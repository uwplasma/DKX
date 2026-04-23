"""RHSMode=1 post-x-block polish and skip policy helpers.

These predicates decide whether a large explicit full-FP CPU solve should run a
post-x-block polish, run targeted FP polish, or skip global sparse rescue after a
good x-block seed.  Keeping them outside the driver makes the runtime-offender
handoff logic directly testable without launching large solves.
"""

from __future__ import annotations

import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
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


def _is_explicit_cpu_rhs1_fp_only(*, op: Any, use_implicit: bool, backend: str) -> bool:
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    return op.fblock.fp is not None and getattr(op.fblock, "pas", None) is None


def rhs1_fast_post_xblock_polish_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a bad large-CPU x-block seed should receive fast polish."""
    env = _env_token("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH")
    if env in _FALSE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(op=op, use_implicit=use_implicit, backend=backend):
        return False

    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_ratio = _env_float("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RATIO", 1.0e3)
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_ABS", 1.0e-6)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)


def rhs1_fp_targeted_polish_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    rhs1_precond_kind: str,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a medium/large explicit FP xmg solve should be polished."""
    env = _env_token("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH")
    if env in _FALSE_VALUES:
        return False
    if not _is_explicit_cpu_rhs1_fp_only(op=op, use_implicit=use_implicit, backend=backend):
        return False
    if str(rhs1_precond_kind) != "xmg":
        return False

    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_ratio = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_RATIO", 10.0)
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_ABS", 1.0e-9)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)


def rhs1_skip_global_sparse_after_xblock_allowed(
    *,
    op: Any,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a good x-block seed may skip global sparse rescue."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK")
    if env not in _TRUE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut) or not bool(used_explicit_fp_xblock_seed):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(op=op, use_implicit=use_implicit, backend=backend):
        return False

    skip_min = _env_int("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_MIN", 12000)
    if int(active_size) < max(1, int(skip_min)):
        return False
    skip_ratio = _env_float("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_RATIO", 5.0e4)
    skip_abs = _env_float("SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_ABS", 5.0e-4)
    threshold = max(float(skip_abs), float(target) * max(1.0, float(skip_ratio)))
    return float(residual_norm) <= float(threshold)


__all__ = [
    "rhs1_fast_post_xblock_polish_allowed",
    "rhs1_fp_targeted_polish_allowed",
    "rhs1_skip_global_sparse_after_xblock_allowed",
]
