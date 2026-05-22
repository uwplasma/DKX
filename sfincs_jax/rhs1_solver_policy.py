"""RHSMode=1 solver-policy parsing helpers.

This module keeps environment-variable parsing and small solver-policy records
out of the large v3 driver. The functions are deliberately conservative:
invalid values fall back to documented defaults, and opt-in correction hooks
report zero requested steps unless their enabling flag is truthy.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping
import os


TRUE_ENV_VALUES = {"1", "true", "t", "yes", "on", ".true.", ".t."}
FALSE_ENV_VALUES = {"0", "false", "f", "no", "off", ".false.", ".f."}


def _env_get(env: Mapping[str, str] | None, name: str) -> str:
    source = os.environ if env is None else env
    return str(source.get(name, "")).strip()


def read_bool_env(name: str, *, default: bool = False, env: Mapping[str, str] | None = None) -> bool:
    """Parse a Fortran/Python-style boolean environment value."""

    raw = _env_get(env, name).lower()
    if not raw:
        return bool(default)
    if raw in TRUE_ENV_VALUES:
        return True
    if raw in FALSE_ENV_VALUES:
        return False
    return bool(default)


def read_int_env(
    name: str,
    *,
    default: int,
    minimum: int = 0,
    env: Mapping[str, str] | None = None,
) -> int:
    """Parse an integer environment value with a lower bound."""

    raw = _env_get(env, name)
    try:
        value = int(raw) if raw else int(default)
    except ValueError:
        value = int(default)
    return max(int(minimum), int(value))


def read_float_env(
    name: str,
    *,
    default: float,
    minimum: float = 0.0,
    env: Mapping[str, str] | None = None,
) -> float:
    """Parse a floating-point environment value with a lower bound."""

    raw = _env_get(env, name)
    try:
        value = float(raw) if raw else float(default)
    except ValueError:
        value = float(default)
    return max(float(minimum), float(value))


@dataclass(frozen=True)
class RHS1PostMinresPolicy:
    """Opt-in post-Krylov minres cleanup policy."""

    steps_requested: int
    alpha_clip: float
    min_improvement: float


@dataclass(frozen=True)
class RHS1SubspaceCorrectionPolicy:
    """Policy for small residual-equation or coarse correction spaces."""

    steps_requested: int
    max_directions: int
    max_extra_units: int
    fsavg_lmax: int
    angular_lmax: int
    include_angular_residual: bool
    include_raw: bool
    alpha_clip: float
    rcond: float
    min_improvement: float
    include_post_coarse: bool = True
    include_qi_basis: bool = True


@dataclass(frozen=True)
class RHS1PostSolveCorrectionPolicy:
    """All post-Krylov x-block correction policies used by the v3 driver."""

    post_minres: RHS1PostMinresPolicy
    post_coarse: RHS1SubspaceCorrectionPolicy
    post_residual_equation: RHS1SubspaceCorrectionPolicy


def read_post_minres_policy(*, env: Mapping[str, str] | None = None) -> RHS1PostMinresPolicy:
    """Read the post-minres cleanup policy."""

    return RHS1PostMinresPolicy(
        steps_requested=read_int_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS",
            default=0,
            minimum=0,
            env=env,
        ),
        alpha_clip=read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_ALPHA_CLIP",
            default=10.0,
            minimum=0.0,
            env=env,
        ),
        min_improvement=read_float_env(
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_MIN_IMPROVEMENT",
            default=0.0,
            minimum=0.0,
            env=env,
        ),
    )


def read_subspace_correction_policy(
    prefix: str,
    *,
    enabled_default: bool = False,
    steps_default: int = 1,
    max_directions_default: int = 16,
    max_extra_units_default: int = 8,
    fsavg_lmax_default: int = 2,
    angular_lmax_default: int = -1,
    include_angular_residual_default: bool = False,
    include_raw_default: bool = True,
    alpha_clip_default: float = 0.0,
    rcond_default: float = 1.0e-12,
    min_improvement_default: float = 0.0,
    include_post_coarse_default: bool = True,
    include_qi_basis_default: bool = True,
    env: Mapping[str, str] | None = None,
) -> RHS1SubspaceCorrectionPolicy:
    """Read a named small-subspace correction policy.

    ``prefix`` is the environment-variable stem, for example
    ``SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE``.
    """

    enabled = read_bool_env(prefix, default=enabled_default, env=env)
    steps_requested = (
        read_int_env(f"{prefix}_STEPS", default=steps_default, minimum=1, env=env)
        if enabled
        else 0
    )
    return RHS1SubspaceCorrectionPolicy(
        steps_requested=steps_requested,
        max_directions=read_int_env(
            f"{prefix}_MAX_DIRECTIONS",
            default=max_directions_default,
            minimum=1,
            env=env,
        ),
        max_extra_units=read_int_env(
            f"{prefix}_MAX_EXTRA_UNITS",
            default=max_extra_units_default,
            minimum=0,
            env=env,
        ),
        fsavg_lmax=read_int_env(
            f"{prefix}_FSAVG_LMAX",
            default=fsavg_lmax_default,
            minimum=0,
            env=env,
        ),
        angular_lmax=read_int_env(
            f"{prefix}_ANGULAR_LMAX",
            default=angular_lmax_default,
            minimum=-1,
            env=env,
        ),
        include_angular_residual=read_bool_env(
            f"{prefix}_ANGULAR_RESIDUAL",
            default=include_angular_residual_default,
            env=env,
        ),
        include_raw=read_bool_env(
            f"{prefix}_INCLUDE_RAW",
            default=include_raw_default,
            env=env,
        ),
        alpha_clip=read_float_env(
            f"{prefix}_ALPHA_CLIP",
            default=alpha_clip_default,
            minimum=0.0,
            env=env,
        ),
        rcond=read_float_env(
            f"{prefix}_RCOND",
            default=rcond_default,
            minimum=0.0,
            env=env,
        ),
        min_improvement=read_float_env(
            f"{prefix}_MIN_IMPROVEMENT",
            default=min_improvement_default,
            minimum=0.0,
            env=env,
        ),
        include_post_coarse=read_bool_env(
            f"{prefix}_INCLUDE_POST_COARSE",
            default=include_post_coarse_default,
            env=env,
        ),
        include_qi_basis=read_bool_env(
            f"{prefix}_INCLUDE_QI_BASIS",
            default=include_qi_basis_default,
            env=env,
        ),
    )


def read_probe_coarse_policy(*, env: Mapping[str, str] | None = None) -> RHS1SubspaceCorrectionPolicy:
    """Read the pre-Krylov probe-coarse policy."""

    return read_subspace_correction_policy(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE",
        max_directions_default=16,
        max_extra_units_default=8,
        fsavg_lmax_default=2,
        angular_lmax_default=-1,
        include_angular_residual_default=False,
        env=env,
    )


def read_post_coarse_policy(*, env: Mapping[str, str] | None = None) -> RHS1SubspaceCorrectionPolicy:
    """Read the post-Krylov coarse correction policy."""

    return read_subspace_correction_policy(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE",
        max_directions_default=16,
        max_extra_units_default=8,
        fsavg_lmax_default=2,
        angular_lmax_default=-1,
        include_angular_residual_default=False,
        env=env,
    )


def read_post_residual_equation_policy(
    *,
    env: Mapping[str, str] | None = None,
) -> RHS1SubspaceCorrectionPolicy:
    """Read the post-Krylov final-residual equation policy."""

    return read_subspace_correction_policy(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION",
        max_directions_default=64,
        max_extra_units_default=8,
        fsavg_lmax_default=4,
        angular_lmax_default=1,
        include_angular_residual_default=True,
        include_raw_default=True,
        include_post_coarse_default=True,
        include_qi_basis_default=True,
        env=env,
    )


def read_post_solve_correction_policy(
    *,
    env: Mapping[str, str] | None = None,
) -> RHS1PostSolveCorrectionPolicy:
    """Read all post-Krylov correction policies for the x-block solve."""

    return RHS1PostSolveCorrectionPolicy(
        post_minres=read_post_minres_policy(env=env),
        post_coarse=read_post_coarse_policy(env=env),
        post_residual_equation=read_post_residual_equation_policy(env=env),
    )
