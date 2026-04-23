"""RHSMode=1 sparse exact-LU and sparse-prefer policy helpers.

These helpers decide when a RHSMode=1 solve may use exact sparse LU, when a
moderate full-FP solve should prefer sparse work over the small dense shortcut,
and whether that sparse-prefer decision may skip the stage-2 fallback.  The logic
is intentionally side-effect free apart from environment-variable reads so it can
be validated without building the kinetic matrix.
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


def rhs1_sparse_exact_lu_requested(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    full_precond_requested: bool = False,
    preconditioner_x: int,
    use_dkes: bool,
    backend: str,
) -> bool:
    """Return whether the RHSMode=1 sparse exact-LU lane should be attempted.

    ``sparse_max_size`` is accepted to keep the policy signature aligned with the
    driver wrapper.  The exact-LU lane has its own environment-controlled cap
    because it can intentionally exceed the ILU/sparse-preconditioner size cap on
    accelerator DKES or full-x cases.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU")
    if env in _FALSE_VALUES:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False

    has_fp = op.fblock.fp is not None
    has_pas = getattr(op.fblock, "pas", None) is not None
    allow_pas_full = bool(has_pas) and (bool(full_precond_requested) or env in _TRUE_VALUES)
    if (not has_fp) and (not allow_pas_full):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False

    backend_name = str(backend).strip().lower()
    exact_default = 6000 if backend_name == "cpu" else 12000
    exact_max = max(0, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", exact_default))
    if int(active_size) > int(exact_max):
        return False
    if env in _TRUE_VALUES:
        return True

    accel_small_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", 4000)
    accel_small_case = backend_name != "cpu" and int(active_size) <= max(0, int(accel_small_max))
    return int(preconditioner_x) == 0 or (
        backend_name != "cpu" and (bool(use_dkes) or bool(accel_small_case))
    )


def rhs1_prefer_sparse_over_dense_shortcut(
    *,
    op: Any,
    active_size: int,
    sparse_max_size: int,
    use_implicit: bool,
) -> bool:
    """Return whether a moderate explicit FP solve should keep the sparse path."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT")
    if env in _FALSE_VALUES:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.fp is None or bool(use_implicit):
        return False
    if int(active_size) > int(sparse_max_size):
        return False

    min_size = max(1, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", 2000))
    return int(active_size) >= int(min_size)


def rhs1_sparse_prefer_skips_stage2(
    *,
    sparse_prefer_over_dense_shortcut: bool,
    sparse_precond_mode: str,
) -> bool:
    """Return whether sparse-prefer routing should skip the stage-2 fallback."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2")
    if env in _FALSE_VALUES:
        return False
    return bool(sparse_prefer_over_dense_shortcut) and (
        str(sparse_precond_mode).strip().lower() != "off"
    )


__all__ = [
    "rhs1_prefer_sparse_over_dense_shortcut",
    "rhs1_sparse_exact_lu_requested",
    "rhs1_sparse_prefer_skips_stage2",
]
