"""Pure solver/preconditioner path policy helpers for the v3 driver.

The v3 driver keeps runtime state such as the active JAX backend and cached
operator-size hints.  This module holds the small policy decisions underneath
that state so they can be tested without importing the full driver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os


_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}


@dataclass(frozen=True)
class PreconditionerPolicyHints:
    """Cached operator metadata used by preconditioner auto-selection."""

    size_hint: int | None = None
    geom_scheme: int | None = None
    use_dkes: bool | None = None
    rhs1_precond_kind: str | None = None
    has_pas: bool | None = None
    has_fp: bool | None = None
    include_phi1: bool | None = None
    rhs_mode: int | None = None
    er_abs: float | None = None


def _env_value(name: str, env: Mapping[str, str] | None) -> str:
    source = os.environ if env is None else env
    return str(source.get(name, "")).strip()


def _env_token(name: str, env: Mapping[str, str] | None) -> str:
    return _env_value(name, env).lower()


def _env_int(name: str, default: int, env: Mapping[str, str] | None) -> int:
    raw = _env_value(name, env)
    try:
        return int(raw) if raw else int(default)
    except ValueError:
        return int(default)


def _env_float(name: str, default: float, env: Mapping[str, str] | None) -> float:
    raw = _env_value(name, env)
    try:
        return float(raw) if raw else float(default)
    except ValueError:
        return float(default)


def rhs1_dkes_gmres_budget(
    *,
    restart: int,
    maxiter: int | None,
    restart_forced: bool,
    maxiter_forced: bool,
    restart_cap_env: str,
) -> tuple[int, int | None, bool, bool]:
    """Apply PAS/FP DKES GMRES defaults without overriding explicit budgets."""
    restart_use = max(1, int(restart))
    maxiter_use = None if maxiter is None else max(1, int(maxiter))
    restart_defaulted = False
    maxiter_defaulted = False

    if not bool(restart_forced):
        restart_use = max(int(restart_use), 80)
        try:
            restart_cap = int(str(restart_cap_env).strip()) if str(restart_cap_env).strip() else 100
        except ValueError:
            restart_cap = 100
        if restart_cap > 0:
            restart_use = min(int(restart_use), int(restart_cap))
        restart_defaulted = True

    if not bool(maxiter_forced):
        if maxiter_use is None:
            maxiter_use = 600
        else:
            maxiter_use = max(int(maxiter_use), 600)
        maxiter_defaulted = True

    return int(restart_use), maxiter_use, bool(restart_defaulted), bool(maxiter_defaulted)


def use_solver_jit(
    *,
    size_hint: int | None = None,
    precond_size_hint: int | None = None,
    env: Mapping[str, str] | None = None,
    default_max_size: int = 100000,
) -> bool:
    """Return whether the v3 driver should JIT the Krylov solve loop."""
    solver_jit = _env_token("SFINCS_JAX_SOLVER_JIT", env)
    if solver_jit in _TRUE_VALUES:
        return True
    if solver_jit in _FALSE_VALUES:
        return False
    if size_hint is None:
        size_hint = precond_size_hint or 0
    thresh = _env_int("SFINCS_JAX_SOLVER_JIT_MAX_SIZE", int(default_max_size), env)
    return int(size_hint) <= thresh


def rhs1_residual_needs_rescue(
    residual_norm: float,
    target: float,
    *,
    force: bool = False,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether a second-stage RHSMode=1 rescue is worth launching."""
    if bool(force):
        return True
    residual = float(residual_norm)
    target_value = float(target)
    if not math.isfinite(residual):
        return True
    if target_value <= 0.0 or not math.isfinite(target_value):
        return residual > target_value
    slack = _env_float("SFINCS_JAX_RHSMODE1_RESCUE_TARGET_SLACK", 1.0e-2, env)
    slack = max(0.0, float(slack))
    return residual > target_value * (1.0 + slack)


def is_resource_exhausted_error(exc: Exception) -> bool:
    """Return whether an exception message looks like backend memory exhaustion."""
    text = f"{type(exc).__name__}: {exc}"
    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        text += f" | cause={type(cause).__name__}: {cause}"
    text = text.lower()
    markers = (
        "resource_exhausted",
        "out of memory",
        "allocator",
        "memory allocation",
        "cudaerrormemoryallocation",
    )
    return any(marker in text for marker in markers)


def auto_pas_geom4_fp32_precond_allowed(
    *,
    size_hint: int,
    hints: PreconditionerPolicyHints,
    backend: str,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Return whether the narrow PAS geometry-4 fp32 preconditioner auto path applies."""
    if _env_token("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4", env) in _FALSE_VALUES:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if hints.geom_scheme != 4:
        return False
    if hints.rhs_mode != 1:
        return False
    if not bool(hints.has_pas) or bool(hints.has_fp):
        return False
    if bool(hints.include_phi1) or bool(hints.use_dkes):
        return False
    if str(hints.rhs1_precond_kind or "").strip().lower() not in {"schur", "pas_schur"}:
        return False

    er_abs = abs(float(hints.er_abs or 0.0))
    er_max = _env_float("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_ER_MAX", 1.0e-12, env)
    if er_abs > max(0.0, float(er_max)):
        return False

    policy_size = int(hints.size_hint or size_hint)
    min_size = _env_int("SFINCS_JAX_PRECOND_FP32_PAS_GEOM4_MIN_SIZE", 15000, env)
    return policy_size >= max(1, int(min_size))


def sparse_structural_tol(
    *,
    default_tol: float,
    env: Mapping[str, str] | None = None,
) -> float:
    """Resolve the explicit sparse structural drop tolerance."""
    tol = _env_float("SFINCS_JAX_SPARSE_STRUCTURAL_TOL", float(default_tol), env)
    return max(0.0, float(tol))


def precond_dtype_name(
    *,
    size_hint: int | None = None,
    hints: PreconditionerPolicyHints,
    backend: str,
    env: Mapping[str, str] | None = None,
) -> str:
    """Resolve the preconditioner factor dtype name, preserving v3 defaults."""
    dtype_env = _env_token("SFINCS_JAX_PRECOND_DTYPE", env)
    if dtype_env in {"", "auto", "mixed"}:
        if size_hint is None:
            effective_size = hints.size_hint or 0
            if auto_pas_geom4_fp32_precond_allowed(
                size_hint=int(effective_size),
                hints=hints,
                backend=backend,
                env=env,
            ):
                return "float32"
            thresh_name = "SFINCS_JAX_PRECOND_FP32_MIN_SIZE"
            thresh_default = 5_000_000
        else:
            effective_size = int(size_hint)
            if auto_pas_geom4_fp32_precond_allowed(
                size_hint=int(effective_size),
                hints=hints,
                backend=backend,
                env=env,
            ):
                return "float32"
            thresh_name = "SFINCS_JAX_PRECOND_FP32_MIN_BLOCK"
            thresh_default = 5_000_000
        thresh = _env_int(thresh_name, thresh_default, env)
        return "float32" if int(effective_size) >= thresh else "float64"

    if dtype_env in {"float32", "fp32", "f32", "32"}:
        return "float32"
    if dtype_env in {"float64", "fp64", "f64", "64"}:
        return "float64"
    return "float64"


def rhsmode1_sparse_pc_default_permc_spec(
    *,
    constrained_pas_pc: bool,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> str:
    """Return the measured SuperLU column-ordering default for sparse-PC RHSMode=1."""
    del n_species
    if bool(tokamak_pas_er_pc):
        return "MMD_AT_PLUS_A"
    if bool(constrained_pas_pc):
        return "MMD_ATA"
    return "COLAMD"


def rhsmode1_sparse_pc_default_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> int:
    """Return the sparse-PC GMRES restart after scoped production caps."""
    requested = max(1, int(requested_restart))
    if str(restart_env_value).strip():
        return requested
    if bool(tokamak_pas_er_pc) and int(n_species) == 1:
        return min(requested, 40)
    return requested
