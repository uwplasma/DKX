"""RHSMode=1 constraint-scheme-0 sparse-first policy helpers.

Constraint scheme 0 is the PETSc-compatible path for several RHSMode=1 full-FP
examples.  These helpers keep the sparse-first, PETSc-compatibility, and dense
fallback switches separate from the driver so their guard conditions can be
tested without assembling the kinetic operator.
"""

from __future__ import annotations

import os
from typing import Any


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


def _env_token(name: str) -> str:
    return str(os.environ.get(name, "")).strip().lower()


def _has_constraint0_fp_rhs1(op: Any) -> bool:
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 0:
        return False
    return op.fblock.fp is not None


def _sparse_method_allowed(
    *,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    if str(sparse_precond_mode).strip().lower() == "off":
        return False
    return int(active_size) <= int(sparse_max_size)


def rhs1_constraint0_sparse_first(
    *,
    op: Any,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
    backend: str,
) -> bool:
    """Return whether constraint-scheme-0 RHSMode=1 should try sparse first.

    The default is accelerator-only because this lane was introduced to avoid
    small/medium GPU dense-LU regressions while retaining CPU dense fallback
    behavior unless the user explicitly opts into sparse-first CPU behavior.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_SPARSE_FIRST")
    if env in _FALSE_VALUES:
        return False
    if env not in _TRUE_VALUES and str(backend).strip().lower() == "cpu":
        return False
    if not _has_constraint0_fp_rhs1(op):
        return False
    return _sparse_method_allowed(
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=active_size,
        sparse_max_size=sparse_max_size,
    )


def rhs1_constraint0_petsc_compat(
    *,
    op: Any,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    """Return whether explicit PETSc-compatible sparse behavior is requested."""
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT")
    if env in {"", *_FALSE_VALUES}:
        return False
    if not _has_constraint0_fp_rhs1(op):
        return False
    if not _sparse_method_allowed(
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=active_size,
        sparse_max_size=sparse_max_size,
    ):
        return False
    return env in _TRUE_VALUES


def rhs1_constraint0_dense_fallback_allowed(op: Any) -> bool:
    """Return whether dense fallback is allowed for constraint-scheme-0 solves."""
    if int(op.constraint_scheme) != 0:
        return True
    env = _env_token("SFINCS_JAX_RHSMODE1_CS0_DENSE_FALLBACK")
    return env in _TRUE_VALUES


__all__ = [
    "rhs1_constraint0_dense_fallback_allowed",
    "rhs1_constraint0_petsc_compat",
    "rhs1_constraint0_sparse_first",
]
