"""RHSMode=1 large host-sparse and x-block rescue policy helpers.

Large explicit full-FP cases need a careful ordering between x-block seeds,
global sparse rescue, exact sparse LU, and species-x-block rescue. CPU solves
use these host sparse branches directly. Accelerator solves may also use them
for bounded non-implicit CLI/output lanes when device Krylov is slower or less
robust than a host sparse factorization. This module keeps those
branch-selection rules outside the driver so runtime-offender lanes can be
tested without constructing the full kinetic operator.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


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


def _is_explicit_rhs1_fp(op: Any) -> bool:
    return int(op.rhs_mode) == 1 and (not bool(op.include_phi1)) and op.fblock.fp is not None


def _is_explicit_rhs1_fp_only(op: Any) -> bool:
    return _is_explicit_rhs1_fp(op) and getattr(op.fblock, "pas", None) is None


def _host_sparse_rescue_backend_allowed(*, backend: str, active_size: int | None = None) -> bool:
    """Return whether this backend may use non-differentiable host sparse rescue."""

    backend_name = str(backend).strip().lower()
    if backend_name == "cpu":
        return True

    env = _env_token("SFINCS_JAX_RHSMODE1_ACCELERATOR_HOST_SPARSE_RESCUE")
    if env in _FALSE_VALUES:
        return False

    # If the caller cannot provide a size, require an explicit opt-in on
    # accelerators. Size-aware solver-policy calls get a conservative default
    # cap so moderate GPU CLI/output cases can avoid fragile device Krylov tails.
    if active_size is None:
        return env in _TRUE_VALUES

    rescue_max = _env_int("SFINCS_JAX_RHSMODE1_ACCELERATOR_HOST_SPARSE_RESCUE_MAX", 30000)
    return int(active_size) <= max(1, int(rescue_max))


def rhs1_large_cpu_sparse_exact_lu_allowed(*, active_size: int) -> bool:
    """Return whether the large-CPU sparse rescue may use exact sparse LU."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU")
    if env in _FALSE_VALUES:
        return False
    exact_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_MAX", 30000)
    return int(active_size) <= max(0, int(exact_max))


def rhs1_large_cpu_sparse_rescue_allowed(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    residual_norm: float,
    target: float,
    backend: str,
) -> bool:
    """Return whether a large CPU FP solve should try global sparse rescue."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE")
    if env in _FALSE_VALUES:
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=int(active_size)):
        return False
    if not _is_explicit_rhs1_fp(op):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False

    fullx_min = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FULLX_MIN", 50000)
    if int(preconditioner_x) != 0 and int(active_size) < max(0, int(fullx_min)):
        if not rhs1_large_cpu_sparse_exact_lu_allowed(active_size=int(active_size)):
            return False
    if int(active_size) <= int(sparse_max_size):
        return False

    rescue_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_MAX", 80000)
    if int(active_size) > max(1, int(rescue_max)):
        return False
    if float(target) <= 0.0:
        return True
    rescue_ratio = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_RATIO", 1.0e3)
    return float(residual_norm) > float(target) * float(rescue_ratio)


def rhs1_large_cpu_sparse_rescue_first(
    *,
    large_cpu_sparse_rescue: bool,
    strong_precond_env: str,
) -> bool:
    """Return whether large-CPU sparse rescue should run before strong preconditioning."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_FIRST")
    if env in _FALSE_VALUES:
        return False
    return bool(large_cpu_sparse_rescue) and str(strong_precond_env).strip().lower() in {"", "auto"}


def rhs1_large_cpu_sparse_skip_primary_allowed(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether mid-size full-FP solves may jump directly to sparse LU.

    This is the early-entry form of the existing non-differentiable host sparse
    rescue for systems just above the dense cutoff where the measured default
    Krylov path only serves as a slow gateway to exact active sparse LU. It is
    deliberately bounded by the same exact-LU cap used by the rescue itself.
    """

    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY")
    if env in _FALSE_VALUES:
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=int(active_size)):
        return False
    if not _is_explicit_rhs1_fp(op):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(active_size) <= int(sparse_max_size):
        return False
    skip_min_default = max(int(sparse_max_size) + 1, 8000)
    skip_min = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MIN", skip_min_default)
    if int(active_size) < max(1, int(skip_min)):
        return False
    skip_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MAX", 30000)
    if int(active_size) > max(1, int(skip_max)):
        return False
    return rhs1_large_cpu_sparse_exact_lu_allowed(active_size=int(active_size))


def rhs1_large_cpu_sparse_exact_lu_xblock_allowed(
    *,
    op: Any,
    active_size: int,
    preconditioner_x: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    xblock_seed_residual: float,
    xblock_seed_improvement_ratio: float,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether a good x-block seed should promote exact sparse LU."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK")
    if env in _FALSE_VALUES:
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=int(active_size)):
        return False
    if bool(use_implicit):
        return False
    if not bool(used_large_cpu_xblock_shortcut) or not bool(used_explicit_fp_xblock_seed):
        return False
    if not _is_explicit_rhs1_fp_only(op):
        return False
    if int(preconditioner_x) == 0:
        return False

    exact_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_MAX", 70000)
    if int(active_size) > max(0, int(exact_max)):
        return False
    residual_abs = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_ABS", 5.0e-4)
    if not np.isfinite(float(xblock_seed_residual)) or float(xblock_seed_residual) > float(residual_abs):
        return False
    improvement_ratio = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_RESCUE_EXACT_LU_XBLOCK_RATIO", 100.0)
    return float(xblock_seed_improvement_ratio) >= max(1.0, float(improvement_ratio))


def rhs1_sparse_xblock_rescue_allowed(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    pre_theta: int,
    pre_zeta: int,
    residual_norm: float,
    target: float,
    backend: str,
) -> bool:
    """Return whether the CPU FP x-block sparse rescue path is eligible."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE")
    if env in _FALSE_VALUES:
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=int(active_size)):
        return False
    if not _is_explicit_rhs1_fp(op):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(preconditioner_x) == 0:
        return False
    if int(pre_theta) != 0 or int(pre_zeta) != 0:
        return False

    rescue_min_default = max(int(sparse_max_size) + 1, 12000)
    rescue_min = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MIN", rescue_min_default)
    rescue_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_MAX", 120000)
    if int(active_size) < max(1, int(rescue_min)):
        return False
    if int(active_size) > max(1, int(rescue_max)):
        return False
    if float(target) <= 0.0:
        return True
    rescue_ratio = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_XBLOCK_RESCUE_RATIO", 1.0e2)
    return float(residual_norm) > float(target) * float(rescue_ratio)


def rhs1_fp_xblock_assembled_host_allowed(
    *,
    op: Any,
    preconditioner_species: int,
    preconditioner_xi: int,
    use_implicit: bool,
    backend: str,
    active_size: int | None = None,
) -> bool:
    """Return whether an explicit CPU FP x-block seed may use host assembly."""
    env = _env_token("SFINCS_JAX_RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST")
    if env in _FALSE_VALUES:
        return False
    if bool(use_implicit):
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=active_size):
        return False
    if not _is_explicit_rhs1_fp_only(op):
        return False
    # In Fortran decks, preconditioner_species=0 means "keep species coupling".
    # For a one-species full-FP system this is algebraically identical to the
    # per-species x-block used by the host-assembled sparse path, and it avoids
    # expensive dense matvec probing.  Multi-species systems must keep the old
    # guard because dropping inter-species coupling changes the preconditioner.
    n_species = int(getattr(op, "n_species", 0) or 0)
    if int(preconditioner_species) == 0 and n_species != 1:
        return False
    if int(preconditioner_xi) != 1:
        return False
    if bool(op.point_at_x0):
        return False
    return True


def rhs1_large_cpu_xblock_skip_primary_allowed(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_species: int,
    preconditioner_x: int,
    preconditioner_xi: int,
    pre_theta: int,
    pre_zeta: int,
    use_implicit: bool,
    rhs1_precond_env: str,
    backend: str,
) -> bool:
    """Return whether a large CPU FP solve should seed with x-block first."""
    env = _env_token("SFINCS_JAX_RHSMODE1_LARGE_CPU_XBLOCK_SKIP_PRIMARY")
    if env in _FALSE_VALUES:
        return False
    if not _host_sparse_rescue_backend_allowed(backend=backend, active_size=int(active_size)):
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(active_size) <= int(sparse_max_size):
        return False
    if int(preconditioner_x) == 0 or int(pre_theta) != 0 or int(pre_zeta) != 0:
        return False
    if rhs1_precond_env not in {"", "auto", "default"}:
        return False
    return rhs1_fp_xblock_assembled_host_allowed(
        op=op,
        preconditioner_species=preconditioner_species,
        preconditioner_xi=preconditioner_xi,
        use_implicit=bool(use_implicit),
        backend=backend,
        active_size=int(active_size),
    )


def rhs1_sparse_sxblock_rescue_allowed(
    *,
    op: Any,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    pre_theta: int,
    pre_zeta: int,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether species-x-block sparse rescue is eligible."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE")
    if env not in _TRUE_VALUES:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if int(active_size) <= int(sparse_max_size):
        return False
    if int(preconditioner_x) == 0 or int(pre_theta) != 0 or int(pre_zeta) != 0:
        return False
    if int(getattr(op, "n_species", 1)) <= 1:
        return False
    if not _is_explicit_rhs1_fp_only(op):
        return False

    rescue_min_default = max(int(sparse_max_size) + 1, 12000)
    rescue_min = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE_MIN", rescue_min_default)
    rescue_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_SXBLOCK_RESCUE_MAX", 120000)
    return max(1, int(rescue_min)) <= int(active_size) <= max(1, int(rescue_max))


__all__ = [
    "rhs1_fp_xblock_assembled_host_allowed",
    "rhs1_large_cpu_sparse_exact_lu_allowed",
    "rhs1_large_cpu_sparse_exact_lu_xblock_allowed",
    "rhs1_large_cpu_sparse_rescue_allowed",
    "rhs1_large_cpu_sparse_rescue_first",
    "rhs1_large_cpu_sparse_skip_primary_allowed",
    "rhs1_large_cpu_xblock_skip_primary_allowed",
    "rhs1_sparse_sxblock_rescue_allowed",
    "rhs1_sparse_xblock_rescue_allowed",
]
