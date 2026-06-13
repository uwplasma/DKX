"""Mutable preconditioner context used by v3 solve auto-selection.

The numerical policy decisions live in :mod:`sfincs_jax.solver_path_policy`.
This module owns the small amount of runtime state that the v3 driver updates
while it is building a solve: current operator size, geometry family, collision
model, RHS mode, and electric-field magnitude. Keeping this state outside the
driver makes policy tests independent of the full v3 orchestration module.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from . import solver_path_policy
from .v3_system import _THRESHOLD_FOR_INCLUSION


_PRECOND_SIZE_HINT: int | None = None
_PRECOND_GEOM_SCHEME_HINT: int | None = None
_PRECOND_USE_DKES_HINT: bool | None = None
_PRECOND_RHS1_PRECOND_KIND_HINT: str | None = None
_PRECOND_HAS_PAS_HINT: bool | None = None
_PRECOND_HAS_FP_HINT: bool | None = None
_PRECOND_INCLUDE_PHI1_HINT: bool | None = None
_PRECOND_RHS_MODE_HINT: int | None = None
_PRECOND_ER_ABS_HINT: float | None = None


def set_precond_size_hint(n: int | None) -> None:
    """Cache the current operator size for automatic preconditioner policy."""

    global _PRECOND_SIZE_HINT
    if n is None:
        _PRECOND_SIZE_HINT = None
    else:
        _PRECOND_SIZE_HINT = int(n)


def set_precond_policy_hints(
    *,
    geom_scheme: int | None = None,
    use_dkes: bool | None = None,
    rhs1_precond_kind: str | None = None,
    has_pas: bool | None = None,
    has_fp: bool | None = None,
    include_phi1: bool | None = None,
    rhs_mode: int | None = None,
    er_abs: float | None = None,
) -> None:
    """Cache operator metadata used by stability-first dtype and path policy."""

    global _PRECOND_GEOM_SCHEME_HINT
    global _PRECOND_USE_DKES_HINT
    global _PRECOND_RHS1_PRECOND_KIND_HINT
    global _PRECOND_HAS_PAS_HINT
    global _PRECOND_HAS_FP_HINT
    global _PRECOND_INCLUDE_PHI1_HINT
    global _PRECOND_RHS_MODE_HINT
    global _PRECOND_ER_ABS_HINT
    _PRECOND_GEOM_SCHEME_HINT = None if geom_scheme is None else int(geom_scheme)
    _PRECOND_USE_DKES_HINT = None if use_dkes is None else bool(use_dkes)
    _PRECOND_RHS1_PRECOND_KIND_HINT = None if rhs1_precond_kind is None else str(rhs1_precond_kind)
    _PRECOND_HAS_PAS_HINT = None if has_pas is None else bool(has_pas)
    _PRECOND_HAS_FP_HINT = None if has_fp is None else bool(has_fp)
    _PRECOND_INCLUDE_PHI1_HINT = None if include_phi1 is None else bool(include_phi1)
    _PRECOND_RHS_MODE_HINT = None if rhs_mode is None else int(rhs_mode)
    _PRECOND_ER_ABS_HINT = None if er_abs is None else float(er_abs)


def precond_policy_hints() -> solver_path_policy.PreconditionerPolicyHints:
    """Return the current preconditioner metadata as an immutable policy object."""

    return solver_path_policy.PreconditionerPolicyHints(
        size_hint=_PRECOND_SIZE_HINT,
        geom_scheme=_PRECOND_GEOM_SCHEME_HINT,
        use_dkes=_PRECOND_USE_DKES_HINT,
        rhs1_precond_kind=_PRECOND_RHS1_PRECOND_KIND_HINT,
        has_pas=_PRECOND_HAS_PAS_HINT,
        has_fp=_PRECOND_HAS_FP_HINT,
        include_phi1=_PRECOND_INCLUDE_PHI1_HINT,
        rhs_mode=_PRECOND_RHS_MODE_HINT,
        er_abs=_PRECOND_ER_ABS_HINT,
    )


def use_solver_jit(size_hint: int | None = None) -> bool:
    """Return whether the active solve should use the JIT Krylov wrapper."""

    return solver_path_policy.use_solver_jit(
        size_hint=size_hint,
        precond_size_hint=_PRECOND_SIZE_HINT,
    )


def auto_pas_geom4_fp32_precond_allowed(*, size_hint: int) -> bool:
    """Return whether the narrow PAS geometry-4 fp32 preconditioner path applies."""

    return solver_path_policy.auto_pas_geom4_fp32_precond_allowed(
        size_hint=int(size_hint),
        hints=precond_policy_hints(),
        backend=jax.default_backend(),
    )


def sparse_structural_tol() -> float:
    """Return the sparse structural drop tolerance for pattern extraction."""

    return solver_path_policy.sparse_structural_tol(default_tol=float(_THRESHOLD_FOR_INCLUSION))


def precond_dtype(size_hint: int | None = None) -> jnp.dtype:
    """Return the JAX dtype used for preconditioner factors in this context."""

    dtype_name = solver_path_policy.precond_dtype_name(
        size_hint=size_hint,
        hints=precond_policy_hints(),
        backend=jax.default_backend(),
    )
    return jnp.float32 if dtype_name == "float32" else jnp.float64


__all__ = [
    "auto_pas_geom4_fp32_precond_allowed",
    "precond_dtype",
    "precond_policy_hints",
    "set_precond_policy_hints",
    "set_precond_size_hint",
    "sparse_structural_tol",
    "use_solver_jit",
]
