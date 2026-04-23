"""RHSMode=1 host dense/sparse-direct policy helpers.

The functions in this module decide when the driver may leave the default JAX
Krylov path for host dense or host sparse direct work.  They intentionally depend
only on environment variables, backend strings, and small operator metadata so
they can be tested without assembling a kinetic operator.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_bool(name: str) -> bool | None:
    env = str(os.environ.get(name, "")).strip().lower()
    if env in _TRUE_VALUES:
        return True
    if env in _FALSE_VALUES:
        return False
    return None


def _env_int(name: str, default: int) -> int:
    env = str(os.environ.get(name, "")).strip()
    try:
        return int(env) if env else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float) -> float:
    env = str(os.environ.get(name, "")).strip()
    try:
        return float(env) if env else float(default)
    except ValueError:
        return float(default)


def rhs1_dense_backend_allowed(*, backend: str) -> bool:
    """Return whether RHSMode=1 dense linear algebra may run on the active backend."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR")
    if env is not None:
        return bool(env)
    return str(backend).strip().lower() == "cpu"


def rhs1_host_dense_fallback_allowed(*, backend: str) -> bool:
    """Return whether host dense LU fallback is allowed for RHSMode=1."""
    if str(backend).strip().lower() == "cpu":
        return True
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU")
    return bool(env)


def rhs1_host_dense_shortcut_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    dense_fallback_max: int,
) -> bool:
    """Allow the small accelerator FP branch to use host dense LU directly."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT")
    if env is False:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() == "cpu":
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if op.fblock.fp is None:
        return False
    host_dense_env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_HOST_LU")
    if host_dense_env is False:
        return False
    shortcut_max = _env_int("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX", 900)
    dense_cap = min(max(0, int(shortcut_max)), max(0, int(dense_fallback_max)))
    if dense_cap <= 0:
        return False
    return int(active_size) <= dense_cap


def rhs1_dense_krylov_allowed() -> bool:
    """Return whether dense Krylov fallback is enabled."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_DENSE_KRYLOV")
    if env is not None:
        return bool(env)
    return True


def rhs1_host_sparse_direct_allowed(*, sparse_exact_lu: bool, use_implicit: bool = False) -> bool:
    """Return whether exact sparse LU may be built and solved on the host."""
    if not bool(sparse_exact_lu):
        return False
    if bool(use_implicit):
        return False
    env = _env_bool("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_HOST")
    if env is not None:
        return bool(env)
    return True


def rhs1_sparse_operator_preconditioned_rescue_allowed(
    *,
    op: Any,
    sparse_exact_lu: bool,
    host_sparse_direct_wanted: bool,
    backend: str,
) -> bool:
    """Allow sparse-preconditioned GMRES before exact sparse LU.

    This branch is kept narrow because it is a parity-preserving rescue for CPU
    full-FP constraint-scheme-1 systems, not a general sparse solve replacement.
    """
    if not bool(sparse_exact_lu) or not bool(host_sparse_direct_wanted):
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    env = _env_bool("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES")
    if env is False:
        return False
    return True


def host_sparse_factor_dtype(
    *,
    size: int,
    factorization: str,
    use_implicit: bool,
    backend: str,
) -> np.dtype:
    """Resolve the dtype used for host sparse factorization."""
    env = str(os.environ.get("SFINCS_JAX_HOST_SPARSE_FACTOR_DTYPE", "")).strip().lower()
    if env in {"float64", "fp64", "64"}:
        return np.dtype(np.float64)
    if env in {"float32", "fp32", "32"}:
        return np.dtype(np.float32)
    if bool(use_implicit):
        return np.dtype(np.float64)
    if str(backend).strip().lower() != "cpu":
        return np.dtype(np.float64)
    if str(factorization).strip().lower() != "lu":
        return np.dtype(np.float64)
    min_size = _env_int("SFINCS_JAX_HOST_SPARSE_FACTOR_FLOAT32_MIN", 12000)
    if int(size) >= max(1, int(min_size)):
        return np.dtype(np.float32)
    return np.dtype(np.float64)


def host_sparse_direct_refine_steps(env_name: str, default: int = 2) -> int:
    """Parse nonnegative iterative-refinement step count for host direct solves."""
    return max(0, _env_int(env_name, int(default)))


def rhs1_host_sparse_skip_dense_ratio() -> float:
    """Residual ratio above which sparse direct paths may skip dense fallback."""
    return _env_float("SFINCS_JAX_RHSMODE1_SPARSE_DIRECT_SKIP_DENSE_RATIO", 1.0e4)


def rhs1_explicit_sparse_host_direct_allowed(
    *,
    sparse_exact_lu: bool,
    use_implicit: bool,
    active_size: int,
) -> bool:
    """Return whether the explicit sparse helper may build a host sparse operator."""
    env = _env_bool("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER")
    if env is False:
        return False
    if bool(use_implicit) or (not bool(sparse_exact_lu)):
        return False
    max_size = _env_int("SFINCS_JAX_RHSMODE1_EXPLICIT_SPARSE_HELPER_MAX", 20000)
    return int(active_size) <= max(1, int(max_size))


__all__ = [
    "host_sparse_direct_refine_steps",
    "host_sparse_factor_dtype",
    "rhs1_dense_backend_allowed",
    "rhs1_dense_krylov_allowed",
    "rhs1_explicit_sparse_host_direct_allowed",
    "rhs1_host_dense_fallback_allowed",
    "rhs1_host_dense_shortcut_allowed",
    "rhs1_host_sparse_direct_allowed",
    "rhs1_host_sparse_skip_dense_ratio",
    "rhs1_sparse_operator_preconditioned_rescue_allowed",
]
