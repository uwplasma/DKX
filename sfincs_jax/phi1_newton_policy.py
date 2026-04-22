from __future__ import annotations

from dataclasses import dataclass
import os


def phi1_use_active_dof_mode(
    *,
    rhs_mode: int,
    include_phi1: bool,
    has_reduced_modes: bool,
    env_value: str,
) -> bool:
    env = str(env_value).strip().lower()
    if env in {"1", "true", "yes", "on"}:
        return True
    if env in {"0", "false", "no", "off"}:
        return False
    return bool(int(rhs_mode) == 1 and bool(include_phi1) and bool(has_reduced_modes))


def phi1_gmres_restart(active_size: int, gmres_restart: int) -> int:
    restart = int(gmres_restart)
    if int(active_size) <= 1000:
        restart = min(restart, 200)
    return max(1, restart)


@dataclass(frozen=True)
class Phi1FrozenJacobianPolicy:
    mode: str
    use_cache: bool
    every: int


def phi1_frozen_jacobian_policy(
    *,
    include_phi1: bool,
) -> Phi1FrozenJacobianPolicy:
    jac_mode = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_MODE", "").strip().lower()
    if jac_mode not in {"frozen", "frozen_rhs", "frozen_op"}:
        jac_mode = "frozen" if bool(include_phi1) else "frozen_rhs"

    cache_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE", "").strip().lower()
    if cache_env in {"0", "false", "no", "off"}:
        use_cache = False
    elif cache_env in {"1", "true", "yes", "on"}:
        use_cache = True
    else:
        use_cache = True

    every_env = os.environ.get("SFINCS_JAX_PHI1_FROZEN_JAC_CACHE_EVERY", "").strip()
    try:
        every = int(every_env) if every_env else 1
    except ValueError:
        every = 1
    return Phi1FrozenJacobianPolicy(mode=jac_mode, use_cache=use_cache, every=max(1, int(every)))


@dataclass(frozen=True)
class Phi1LineSearchPolicy:
    step_scale: float
    factor: float | None
    c1: float
    mode: str
    maxiter: int


def phi1_line_search_policy(
    *,
    use_frozen_linearization: bool,
    include_phi1: bool,
) -> Phi1LineSearchPolicy:
    step_scale_env = os.environ.get("SFINCS_JAX_PHI1_STEP_SCALE", "").strip()
    try:
        step_scale = float(step_scale_env) if step_scale_env else 1.0
    except ValueError:
        step_scale = 1.0

    ls_factor_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_FACTOR", "").strip()
    ls_c1_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_C1", "").strip()
    try:
        ls_factor = float(ls_factor_env) if ls_factor_env else None
    except ValueError:
        ls_factor = None
    try:
        ls_c1 = float(ls_c1_env) if ls_c1_env else 1.0e-4
    except ValueError:
        ls_c1 = 1.0e-4

    ls_mode_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MODE", "").strip().lower()
    if ls_mode_env:
        ls_mode = ls_mode_env
    else:
        ls_mode = "petsc" if (bool(use_frozen_linearization) and bool(include_phi1)) else "best"

    max_ls_env = os.environ.get("SFINCS_JAX_PHI1_LINESEARCH_MAXITER", "").strip()
    try:
        max_ls = int(max_ls_env) if max_ls_env else (40 if ls_mode == "petsc" else 12)
    except ValueError:
        max_ls = 40 if ls_mode == "petsc" else 12

    return Phi1LineSearchPolicy(
        step_scale=float(step_scale),
        factor=ls_factor,
        c1=float(ls_c1),
        mode=str(ls_mode),
        maxiter=int(max_ls),
    )
