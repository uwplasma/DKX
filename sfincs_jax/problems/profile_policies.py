"""RHSMode=1 profile-response solve-routing policy helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers import path_policy as _solver_path_policy
from ..solvers.preconditioner_pas_policy import (
    pas_fast_accept as _pas_fast_accept_metric,
    rhs1_pas_small_near_zero_er_kind,
)
from sfincs_jax.solvers.path_policy import (
    SolverAcceptanceCriteria,
    SolverCandidateGate,
    SolverCandidateMetrics,
    solver_candidate_gate,
)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off"}
TRUE_ENV_VALUES = {"1", "true", "t", "yes", "on", ".true.", ".t."}
FALSE_ENV_VALUES = {"0", "false", "f", "no", "off", ".false.", ".f."}

def resolve_use_implicit(*, differentiable: bool | None = None) -> bool:
    """Resolve whether to use the implicit/differentiable linear-solve path."""

    if differentiable is not None:
        return bool(differentiable)
    implicit_env = os.environ.get("SFINCS_JAX_IMPLICIT_SOLVE", "").strip().lower()
    return implicit_env not in _FALSE_VALUES

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

_read_bool_env = read_bool_env
_read_int_env = read_int_env
_read_float_env = read_float_env

_DEFAULT_ACTIVE_AUTO_CANDIDATES = (
    "active_fortran_v3_reduced_lu",
    "active_schwarz_sparse_coarse",
    "active_global_field_split_schur",
    "active_xblock_ell_band_schur",
    "active_ell_band_schur",
    "active_bounded_native_stack",
    "active_xblock",
    "active_diagonal_schur",
    "active_spilu",
    "jacobi",
)

_DEFAULT_ACTIVE_LARGE_AUTO_CANDIDATES = (
    "active_fortran_v3_reduced_lu",
    "active_coupled_kinetic_field_split_sparse_coarse",
    "active_schwarz_sparse_coarse",
    "active_spilu",
)

_ACTIVE_LARGE_FALLBACK_CANDIDATES = frozenset(
    {
        "active_diagonal_schur",
        "active_diag_schur",
        "active_tail_schur",
        "active_constraint_tail_schur",
        "active_field_split",
        "active_field_split_tail",
        "jacobi",
        "diagonal",
    }
)

@dataclass(frozen=True)
class ActiveProjectedPreconditionerAutoPolicy:
    """Resolved auto-policy for active projected RHSMode=1 preconditioners."""

    candidates: tuple[str, ...]
    candidates_requested: tuple[str, ...]
    skipped_large_fallbacks: tuple[str, ...]
    large_fallback_size: int
    large_default_used: bool
    log_progress: bool

def _parse_active_candidates(candidate_env: str) -> tuple[str, ...]:
    return tuple(
        item.strip().lower().replace("-", "_")
        for item in str(candidate_env).split(",")
        if item.strip()
    )

def resolve_active_projected_preconditioner_auto_policy(
    *,
    matrix_size: int,
    env: Mapping[str, str] | None = None,
) -> ActiveProjectedPreconditionerAutoPolicy:
    """Resolve the active-system preconditioner auto ladder."""

    env_map = os.environ if env is None else env
    size = int(matrix_size)
    large_fallback_size = max(
        1,
        read_int_env(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE",
            default=300000,
            env=env_map,
        ),
    )
    candidate_env_override = env_map.get("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES")
    candidate_env = (
        candidate_env_override
        if candidate_env_override is not None
        else ",".join(_DEFAULT_ACTIVE_AUTO_CANDIDATES)
    )
    large_default_used = False
    if candidate_env_override is None and size >= int(large_fallback_size):
        candidate_env = env_map.get(
            "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES",
            ",".join(_DEFAULT_ACTIVE_LARGE_AUTO_CANDIDATES),
        )
        large_default_used = True

    candidates = _parse_active_candidates(candidate_env)
    if not candidates:
        candidates = ("active_diagonal_schur", "jacobi")
    candidates_requested = tuple(candidates)

    allow_large_fallback = read_bool_env(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_ALLOW_LARGE_DIAGONAL_FALLBACK",
        default=False,
        env=env_map,
    )
    skipped_large_fallbacks: tuple[str, ...] = ()
    if size >= int(large_fallback_size) and not bool(allow_large_fallback):
        skipped_large_fallbacks = tuple(
            candidate for candidate in candidates if candidate in _ACTIVE_LARGE_FALLBACK_CANDIDATES
        )
        candidates = tuple(
            candidate for candidate in candidates if candidate not in _ACTIVE_LARGE_FALLBACK_CANDIDATES
        )

    log_progress = read_bool_env(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_PROGRESS",
        default=bool(large_default_used) or size >= int(large_fallback_size),
        env=env_map,
    )
    return ActiveProjectedPreconditionerAutoPolicy(
        candidates=tuple(candidates),
        candidates_requested=tuple(candidates_requested),
        skipped_large_fallbacks=tuple(skipped_large_fallbacks),
        large_fallback_size=int(large_fallback_size),
        large_default_used=bool(large_default_used),
        log_progress=bool(log_progress),
    )

def parse_rhs1_pas_tz_guarded_structured_levels(raw: str) -> tuple[str, ...]:
    """Parse low-memory coarse levels for guarded PAS-TZ fallback trials."""

    normalized = str(raw or "").strip().lower().replace("-", "_")
    if normalized in {"", "0", "false", "no", "off", "none"}:
        return ()
    for sep in ("+", ";", ":", "|"):
        normalized = normalized.replace(sep, ",")
    aliases = {
        "x": "xmg",
        "x_grid": "xmg",
        "xmultigrid": "xmg",
        "x_mg": "xmg",
        "coll": "collision",
        "collisions": "collision",
        "collision_diag": "collision",
        "collision_diagonal": "collision",
        "diag": "collision",
        "xmg_collision": "xmg,collision",
        "collision_xmg": "collision,xmg",
        "structured": "xmg,collision",
        "default": "xmg,collision",
    }
    expanded_tokens: list[str] = []
    for token in normalized.replace(" ", ",").split(","):
        token = token.strip("_ ")
        if not token:
            continue
        expanded = aliases.get(token, token)
        expanded_tokens.extend(
            part.strip("_ ") for part in expanded.split(",") if part.strip("_ ")
        )

    levels: list[str] = []
    for token in expanded_tokens:
        if token not in {"xmg", "collision"}:
            continue
        if token not in levels:
            levels.append(token)
    return tuple(levels)

def rhs1_xblock_fallback_initial_guess(
    *,
    candidate: np.ndarray,
    original_x0: jnp.ndarray | None,
    rhs_shape: tuple[int, ...],
    candidate_residual_norm: float,
    rhs_norm: float,
    precondition_side: str,
) -> tuple[jnp.ndarray | None, bool, bool]:
    """Return a safe initial guess for x-block Krylov rescues.

    Non-GMRES host Krylov methods can produce a useful physical-space state
    before failing a strict residual gate. Reusing that state is safe only for
    left/no preconditioning when it improves over the zero-state RHS norm.
    Right-preconditioned iteration states are rejected because SciPy stores
    them in preconditioned coordinates.
    """

    candidate_improved_rhs = bool(
        np.isfinite(float(candidate_residual_norm))
        and np.isfinite(float(rhs_norm))
        and float(candidate_residual_norm) < float(rhs_norm)
    )
    if (not candidate_improved_rhs) or str(
        precondition_side
    ).strip().lower() == "right":
        return original_x0, False, candidate_improved_rhs
    try:
        candidate_x0 = jnp.asarray(candidate, dtype=jnp.float64)
        if candidate_x0.shape == tuple(rhs_shape) and bool(
            jnp.all(jnp.isfinite(candidate_x0))
        ):
            return candidate_x0, True, candidate_improved_rhs
    except Exception:
        pass
    return original_x0, False, candidate_improved_rhs

# From sfincs_jax.problems.profile_policies
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
        ratio=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_RATIO", 100.0),
        abs_floor=_env_float("SFINCS_JAX_PAS_FAST_ACCEPT_ABS", 1e-07),
    )

def rhs1_host_factor_probe_ok(*, factor: object | None, block_size: int) -> bool:
    """Return whether a host factor solve passes a bounded unit-vector probe."""
    if factor is None or int(block_size) <= 0:
        return False
    probe_max = max(
        _env_float("SFINCS_JAX_RHSMODE1_XBLOCK_FACTOR_PROBE_MAX", 100000000.0), 1.0
    )
    probe = np.ones((int(block_size),), dtype=np.float64)
    try:
        solved = np.asarray(factor.solve(probe), dtype=np.float64).reshape((-1,))
    except Exception:
        return False
    if solved.shape != probe.shape or not np.all(np.isfinite(solved)):
        return False
    ratio = float(np.linalg.norm(solved)) / max(float(np.linalg.norm(probe)), 1e-300)
    return np.isfinite(ratio) and ratio <= probe_max

# From sfincs_jax.problems.profile_policies
@dataclass(frozen=True)
class RHS1Constraint0PETScCompatConfig:
    """Host sparse-ILU controls for the constraint-scheme-0 PETSc lane."""

    drop_tol: float
    fill: float
    diag_pivot: float
    restart: int
    maxiter: int

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

def rhs1_constraint0_petsc_compat_config_from_env(
    *,
    restart: int,
    maxiter: int | None,
) -> RHS1Constraint0PETScCompatConfig:
    """Parse constraint-scheme-0 PETSc-compatible sparse solve controls."""

    default_restart = max(int(restart), 2000)
    default_maxiter = max(int(maxiter or 1), 1)
    return RHS1Constraint0PETScCompatConfig(
        drop_tol=float(
            _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DROP_TOL", 1.0e-4)
        ),
        fill=float(_env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_FILL", 10.0)),
        diag_pivot=float(
            _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_DIAG_PIVOT", 0.0)
        ),
        restart=int(
            _env_int("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_RESTART", default_restart)
        ),
        maxiter=int(
            _env_int("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_MAXITER", default_maxiter)
        ),
    )

def rhs1_constraint0_petsc_compat_regularization(*, max_abs: float) -> float:
    """Parse/floor the PETSc-compatible diagonal regularization."""

    default_reg = 1.0e-12 * float(max_abs)
    regularization = _env_float("SFINCS_JAX_RHSMODE1_CS0_PETSC_COMPAT_REG", default_reg)
    return max(0.0, float(regularization))

# From sfincs_jax.problems.profile_policies
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
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_max = _env_int("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAX", 200000)
    if int(polish_max) > 0 and int(active_size) > int(polish_max):
        return False
    polish_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RATIO", 1000.0
    )
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_ABS", 1e-06)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)

@dataclass(frozen=True)
class RHS1FastPostXBlockPolishControls:
    """Bounded Krylov controls for the fast post-xblock polish pass."""

    restart: int
    maxiter: int
    tol: float

def rhs1_fast_post_xblock_polish_controls_from_env(
    *,
    restart: int,
    maxiter: int | None,
    tol: float,
) -> RHS1FastPostXBlockPolishControls:
    """Parse fast post-xblock polish controls with legacy bounds."""

    restart_use = _env_int(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_RESTART",
        min(int(restart), 40),
    )
    maxiter_use = _env_int(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_MAXITER",
        min(max(40, int(maxiter or 80)), 80),
    )
    tol_use = _env_float(
        "SFINCS_JAX_RHSMODE1_FAST_POST_XBLOCK_POLISH_TOL",
        min(float(tol), 1.0e-10),
    )
    return RHS1FastPostXBlockPolishControls(
        restart=max(5, int(restart_use)),
        maxiter=max(5, int(maxiter_use)),
        tol=float(tol_use),
    )

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
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    if str(rhs1_precond_kind) != "xmg":
        return False
    polish_min = _env_int("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_MIN", 12000)
    if int(active_size) < max(1, int(polish_min)):
        return False
    polish_ratio = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_RATIO", 10.0)
    polish_abs = _env_float("SFINCS_JAX_RHSMODE1_FP_TARGETED_POLISH_ABS", 1e-09)
    threshold = max(float(polish_abs), float(target) * max(1.0, float(polish_ratio)))
    return float(residual_norm) > float(threshold)

@dataclass(frozen=True)
class RHS1FPResidualPolishControls:
    """Controls for the cheap damped FP residual-polish pass."""

    min_size: int
    steps: int
    hybrid: bool
    omega: float
    backtrack: int

@dataclass(frozen=True)
class RHS1FPLowLPolishControls:
    """Controls for the FP low-L angular/x polish pass."""

    lmax_default: int
    block_max: int
    restart: int
    maxiter: int

@dataclass(frozen=True)
class RHS1FPL1PolishControls:
    """Controls for the FP L=1 projected residual-polish pass."""

    enabled: bool
    restart: int
    maxiter: int
    ratio: float
    abs_threshold: float
    tol: float
    full_accept_ratio: float

@dataclass(frozen=True)
class RHS1FPGlobalLowLPolishControls:
    """Controls for the FP global low-L projected residual-polish pass."""

    enabled: bool
    lmax: int
    max_size: int
    ratio: float
    restart: int
    maxiter: int
    abs_threshold: float
    full_accept_ratio: float
    tol: float = 1.0e-10
    threshold_ratio: float = 2.0

@dataclass(frozen=True)
class RHS1FPBiCGStabPolishControls:
    """Controls for the optional FP BiCGStab residual-polish pass."""

    enabled: bool
    min_size: int
    maxiter: int
    tol: float
    atol: float

@dataclass(frozen=True)
class RHS1ScipyRescueControls:
    """Controls for the host-only RHSMode=1 SciPy rescue pass."""

    enabled: bool
    ratio: float
    restart: int
    maxiter: int
    use_strong: bool
    method: str

@dataclass(frozen=True)
class RHS1BiCGStabFallbackControls:
    """Controls for strict BiCGStab-to-GMRES fallback."""

    strict: bool

@dataclass(frozen=True)
class RHS1BiCGStabFallbackDecision:
    """Resolved target and admission flag for BiCGStab-to-GMRES fallback."""

    target: float
    run_fallback: bool

@dataclass(frozen=True)
class RHS1KrylovRoutingControls:
    """Shared Krylov routing controls for RHSMode=1 profile-response solves."""

    gmres_precondition_side: str
    distributed_auto_solver: str

def rhs1_fp_residual_polish_controls_from_env() -> RHS1FPResidualPolishControls:
    """Parse damped FP residual-polish controls with bounded defaults."""

    hybrid_env = _env_token("SFINCS_JAX_RHSMODE1_FP_POLISH_HYBRID")
    omega = _env_float("SFINCS_JAX_RHSMODE1_FP_POLISH_OMEGA", 1.0)
    backtrack = _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_BACKTRACK", 3)
    return RHS1FPResidualPolishControls(
        min_size=max(1, _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_MIN", 80000)),
        steps=max(0, min(_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_STEPS", 2), 6)),
        hybrid=hybrid_env not in _FALSE_VALUES,
        omega=max(1.0e-3, min(float(omega), 1.5)),
        backtrack=max(0, min(int(backtrack), 6)),
    )

def rhs1_fp_l1_polish_controls_from_env() -> RHS1FPL1PolishControls:
    """Parse FP L=1 projected-polish controls with legacy bounds."""

    enabled_env = _env_token("SFINCS_JAX_RHSMODE1_FP_L1_POLISH")
    maxiter = max(
        5,
        min(_env_int("SFINCS_JAX_RHSMODE1_FP_L1_POLISH_MAXITER", 120), 200),
    )
    restart = max(
        5,
        min(_env_int("SFINCS_JAX_RHSMODE1_FP_L1_POLISH_RESTART", 80), max(5, maxiter)),
    )
    return RHS1FPL1PolishControls(
        enabled=enabled_env not in _FALSE_VALUES,
        restart=int(restart),
        maxiter=int(maxiter),
        ratio=_env_float("SFINCS_JAX_RHSMODE1_FP_L1_POLISH_RATIO", 2.0),
        abs_threshold=_env_float("SFINCS_JAX_RHSMODE1_FP_L1_POLISH_ABS", 1.0e-8),
        tol=_env_float("SFINCS_JAX_RHSMODE1_FP_L1_POLISH_TOL", 1.0e-10),
        full_accept_ratio=_env_float(
            "SFINCS_JAX_RHSMODE1_FP_L1_POLISH_FULL_RATIO",
            1.2,
        ),
    )

def rhs1_fp_low_l_polish_controls_from_env(
    *,
    has_fp: bool,
    has_pas: bool,
    n_theta: int,
    n_zeta: int,
) -> RHS1FPLowLPolishControls:
    """Parse FP low-L polish controls and the small-angular-grid default bump."""

    lmax_env = _env_token("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX")
    lmax_default = _env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX", 2)
    if (
        not lmax_env
        and bool(has_fp)
        and not bool(has_pas)
        and int(n_theta) * int(n_zeta) <= 256
    ):
        lmax_default = max(int(lmax_default), 6)
    return RHS1FPLowLPolishControls(
        lmax_default=int(lmax_default),
        block_max=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_BLOCK_MAX", 1500),
        restart=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_RESTART", 80),
        maxiter=_env_int("SFINCS_JAX_RHSMODE1_FP_POLISH_LMAX_MAXITER", 120),
    )

def rhs1_fp_global_low_l_polish_controls_from_env(
    *,
    n_xi: int,
) -> RHS1FPGlobalLowLPolishControls:
    """Parse FP global low-L projected-polish controls with legacy bounds."""

    enabled_env = _env_token("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_L_POLISH")
    low_lmax = _env_int("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_LMAX", 6)
    low_max_size = _env_int("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_MAX_SIZE", 8000)
    low_ratio = _env_float("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_RATIO", 1.0e4)
    maxiter = max(
        5,
        min(_env_int("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_MAXITER", 120), 250),
    )
    restart = max(
        5,
        min(_env_int("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_RESTART", 80), max(5, maxiter)),
    )
    full_ratio = _env_float("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_FULL_RATIO", 1.2)
    return RHS1FPGlobalLowLPolishControls(
        enabled=enabled_env in _TRUE_VALUES,
        lmax=max(0, min(int(low_lmax), int(n_xi) - 1)),
        max_size=max(0, int(low_max_size)),
        ratio=max(1.0, float(low_ratio)),
        restart=int(restart),
        maxiter=int(maxiter),
        abs_threshold=_env_float("SFINCS_JAX_RHSMODE1_FP_GLOBAL_LOW_ABS", 1.0e-8),
        full_accept_ratio=max(1.0, float(full_ratio)),
    )

def rhs1_fp_bicgstab_polish_controls_from_env(
    *,
    tol: float,
    atol: float,
) -> RHS1FPBiCGStabPolishControls:
    """Parse optional FP BiCGStab polish controls with legacy bounds."""

    enabled_env = _env_token("SFINCS_JAX_RHSMODE1_FP_BICGSTAB_POLISH")
    default_tol = min(float(tol), 1.0e-10)
    maxiter = _env_int("SFINCS_JAX_RHSMODE1_FP_BICGSTAB_MAXITER", 120)
    return RHS1FPBiCGStabPolishControls(
        enabled=enabled_env in _TRUE_VALUES,
        min_size=max(1, _env_int("SFINCS_JAX_RHSMODE1_FP_BICGSTAB_MIN", 80000)),
        maxiter=max(5, min(int(maxiter), 400)),
        tol=_env_float("SFINCS_JAX_RHSMODE1_FP_BICGSTAB_TOL", default_tol),
        atol=_env_float("SFINCS_JAX_RHSMODE1_FP_BICGSTAB_ATOL", float(atol)),
    )

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
    if env in _FALSE_VALUES:
        return False
    if not bool(used_large_cpu_xblock_shortcut) or not bool(
        used_explicit_fp_xblock_seed
    ):
        return False
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return False
    skip_min = _env_int(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_MIN", 12000
    )
    if int(active_size) < max(1, int(skip_min)):
        return False
    skip_ratio = _env_float(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_RATIO", 50000.0
    )
    skip_abs = _env_float(
        "SFINCS_JAX_RHSMODE1_SKIP_GLOBAL_SPARSE_AFTER_XBLOCK_ABS", 0.0005
    )
    threshold = max(float(skip_abs), float(target) * max(1.0, float(skip_ratio)))
    return float(residual_norm) <= float(threshold)

def rhs1_scipy_rescue_abs_floor_after_xblock(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> float:
    """Return an absolute residual floor below which CPU SciPy rescue is skipped.

    Large explicit FP runs can reach a physically tight residual after the
    x-block seed/refinement while still missing an over-tight relative target
    caused by a small RHS norm.  In that case a full SciPy rescue tends to chase
    roundoff for minutes.  The floor is intentionally limited to the same
    large-CPU, explicit-FP, post-x-block path and remains user-overridable.
    """
    env = _env_token("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS")
    if env:
        try:
            return max(0.0, float(env))
        except ValueError:
            return 0.0
    if not bool(used_large_cpu_xblock_shortcut) or not bool(
        used_explicit_fp_xblock_seed
    ):
        return 0.0
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return 0.0
    floor_min = _env_int("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_ABS_MIN", 12000)
    if int(active_size) < max(1, int(floor_min)):
        return 0.0
    return 1e-09

def rhs1_scipy_rescue_active_size_allowed(
    *,
    op: Any,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
    backend: str,
) -> bool:
    """Return whether CPU SciPy rescue may run for this active-system size.

    Production-resolution explicit FP VMEC systems can reach the SciPy rescue
    branch with a very poor seed and then spend minutes in host Krylov without
    producing output.  Keep that rescue for moderate systems and for successful
    x-block-seed follow-up, but make the no-seed large-CPU shortcut fail fast by
    default.  A non-positive max-active override restores the historical
    unbounded behavior.
    """
    if not bool(used_large_cpu_xblock_shortcut):
        return True
    if bool(used_explicit_fp_xblock_seed):
        return True
    if not _is_explicit_cpu_rhs1_fp_only(
        op=op, use_implicit=use_implicit, backend=backend
    ):
        return True
    max_active = _env_int("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAX_ACTIVE", 250000)
    if int(max_active) <= 0:
        return True
    return int(active_size) <= int(max_active)

def rhs1_scipy_rescue_controls_from_env(
    *,
    restart: int,
    maxiter: int | None,
) -> RHS1ScipyRescueControls:
    """Parse host SciPy rescue controls with legacy defaults and bounds."""

    enabled_env = _env_token("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE")
    default_restart = max(int(restart), 120)
    default_maxiter = max(int(maxiter or 400), 600)
    method = _env_token("SFINCS_JAX_RHSMODE1_SCIPY_RESCUE_METHOD")
    if method not in {"gmres", "bicgstab"}:
        method = "auto"
    return RHS1ScipyRescueControls(
        enabled=enabled_env not in _FALSE_VALUES,
        ratio=max(
            1.0,
            _env_float("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_RATIO", 1.0e3),
        ),
        restart=max(
            5,
            _env_int(
                "SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_RESTART",
                default_restart,
            ),
        ),
        maxiter=max(
            5,
            _env_int(
                "SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_MAXITER",
                default_maxiter,
            ),
        ),
        use_strong=_env_token("SFINCS_JAX_RHSMODE1_SCIPY_GMRES_RESCUE_USE_STRONG")
        not in _FALSE_VALUES,
        method=method,
    )

def rhs1_pas_source_zero_tolerance_from_env() -> float:
    """Parse the tiny PAS source cleanup tolerance shared by RHSMode=1 exits."""

    return _env_float("SFINCS_JAX_PAS_SOURCE_ZERO_TOL", 2.0e-9)

def rhs1_bicgstab_preconditioner_kind(
    *,
    env_value: str,
    tokamak_pas: bool,
    has_fp: bool,
    has_pas: bool,
    use_dkes: bool,
    rhs1_precond_kind: str | None,
) -> str | None:
    """Resolve the first-attempt BiCGStab preconditioner for RHSMode=1.

    The default path mirrors the historical driver behavior: try a cheap
    collision preconditioner unless the problem family is known to waste time
    before falling back to GMRES.  FP and PAS+DKES systems with a stronger RHS1
    preconditioner should reuse that stronger path so the first attempt is a
    meaningful numerical probe rather than an avoidable setup cost.
    """

    token = str(env_value).strip().lower()
    if token in _FALSE_VALUES:
        kind = None
    elif token in {"rhs1", "same", "preconditioner"}:
        kind = "rhs1"
    elif token in {"", "auto", *_TRUE_VALUES, "collision", "diag"}:
        kind = "collision"
    else:
        kind = None
    if bool(tokamak_pas) and token in {"", "auto"}:
        return None
    if kind == "collision" and bool(has_fp) and rhs1_precond_kind not in {None, "collision"}:
        return "rhs1"
    if (
        kind == "collision"
        and bool(has_pas)
        and bool(use_dkes)
        and rhs1_precond_kind not in {None, "collision"}
    ):
        return "rhs1"
    return kind

def rhs1_bicgstab_fallback_controls_from_env(
    *,
    pas_large_bicgstab_fastpath: bool,
) -> RHS1BiCGStabFallbackControls:
    """Parse strict BiCGStab fallback controls with the PAS-large fastpath exception."""

    env_value = _env_token("SFINCS_JAX_BICGSTAB_FALLBACK")
    if env_value in _FALSE_VALUES:
        strict = False
    elif env_value in _TRUE_VALUES or env_value == "strict":
        strict = True
    else:
        strict = True
    if bool(pas_large_bicgstab_fastpath) and env_value == "":
        strict = False
    return RHS1BiCGStabFallbackControls(strict=bool(strict))

def rhs1_bicgstab_fallback_target_from_env(
    *,
    target: float,
    distributed_axis: str | None,
    has_pas: bool,
    include_phi1: bool,
) -> float:
    """Return the strict BiCGStab fallback target including distributed PAS floor."""

    default_floor = 0.0
    if distributed_axis is not None and bool(has_pas) and not bool(include_phi1):
        default_floor = 1.0e-7
    floor = _env_float("SFINCS_JAX_BICGSTAB_FALLBACK_ABS_FLOOR", default_floor)
    return max(float(target), max(0.0, float(floor)))

def rhs1_bicgstab_fallback_decision(
    *,
    solver_kind: str,
    cpu_large_sparse_shortcut: bool,
    result_is_finite: bool,
    residual_norm: float,
    strict: bool,
    target: float,
    distributed_axis: str | None,
    has_pas: bool,
    include_phi1: bool,
) -> RHS1BiCGStabFallbackDecision:
    """Resolve whether a BiCGStab result should fall back to GMRES."""

    fallback_target = float(target)
    if bool(strict):
        fallback_target = rhs1_bicgstab_fallback_target_from_env(
            target=float(target),
            distributed_axis=distributed_axis,
            has_pas=bool(has_pas),
            include_phi1=bool(include_phi1),
        )
    run_fallback = bool(
        (not bool(cpu_large_sparse_shortcut))
        and str(solver_kind) == "bicgstab"
        and (
            (not bool(result_is_finite))
            or (bool(strict) and float(residual_norm) > float(fallback_target))
        )
    )
    return RHS1BiCGStabFallbackDecision(
        target=float(fallback_target),
        run_fallback=bool(run_fallback),
    )

def rhs1_gmres_precondition_side_from_env() -> str:
    """Return the validated GMRES precondition side with legacy left default."""

    side = _env_token("SFINCS_JAX_GMRES_PRECONDITION_SIDE")
    if side not in {"", "left", "right", "none"}:
        side = ""
    return side or "left"

def rhs1_distributed_auto_solver_from_env() -> str:
    """Return the distributed Krylov solver family used for sharded matvecs."""

    env_value = _env_token("SFINCS_JAX_DISTRIBUTED_KRYLOV")
    if env_value in {
        "",
        "auto",
        "comm_reduced",
        "short_recurrence",
        "bicgstab",
        "bicgstab_jax",
    }:
        return "bicgstab"
    if env_value in {"gmres", "incremental", "batched"}:
        return "gmres"
    return "bicgstab"

def rhs1_krylov_routing_controls_from_env() -> RHS1KrylovRoutingControls:
    """Parse shared RHSMode=1 Krylov routing controls."""

    return RHS1KrylovRoutingControls(
        gmres_precondition_side=rhs1_gmres_precondition_side_from_env(),
        distributed_auto_solver=rhs1_distributed_auto_solver_from_env(),
    )

# From sfincs_jax.problems.profile_policies
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
    allow_pas_full = bool(has_pas) and (
        bool(full_precond_requested) or env in _TRUE_VALUES
    )
    if not has_fp and (not allow_pas_full):
        return False
    if str(solve_method_kind).strip().lower() in {"dense", "dense_ksp"}:
        return False
    backend_name = str(backend).strip().lower()
    exact_default = 6000 if backend_name == "cpu" else 12000
    exact_max = max(
        0, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_MAX", exact_default)
    )
    if int(active_size) > int(exact_max):
        return False
    if env in _TRUE_VALUES:
        return True
    accel_small_max = _env_int(
        "SFINCS_JAX_RHSMODE1_SPARSE_EXACT_LU_ACCEL_SMALL_MAX", 4000
    )
    accel_small_case = backend_name != "cpu" and int(active_size) <= max(
        0, int(accel_small_max)
    )
    return int(preconditioner_x) == 0 or (
        backend_name != "cpu" and (bool(use_dkes) or bool(accel_small_case))
    )

def rhs1_prefer_sparse_over_dense_shortcut(
    *, op: Any, active_size: int, sparse_max_size: int, use_implicit: bool
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
    min_size = max(
        1, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_PREFER_OVER_DENSE_SHORTCUT_MIN", 2000)
    )
    return int(active_size) >= int(min_size)

def rhs1_sparse_prefer_skips_stage2(
    *, sparse_prefer_over_dense_shortcut: bool, sparse_precond_mode: str
) -> bool:
    """Return whether sparse-prefer routing should skip the stage-2 fallback."""
    env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_SKIP_STAGE2")
    if env in _FALSE_VALUES:
        return False
    return (
        bool(sparse_prefer_over_dense_shortcut)
        and str(sparse_precond_mode).strip().lower() != "off"
    )

# From sfincs_jax.problems.profile_policies
@dataclass(frozen=True)
class RHS1SparseRescueOrdering:
    """Resolved sparse-rescue ordering state for one solve branch."""

    enabled: bool
    kind_use: str
    xblock_rescue_active: bool = False
    prefer_sparse_exact_over_dense_shortcut: bool = False
    reason_dense_shortcut_skip: bool = False
    reason_size_disabled: bool = False
    reason_size_large_cpu: bool = False
    reason_size_exact_direct: bool = False
    reason_size_targeted: bool = False
    reason_sparse_jax_mem_disabled: bool = False
    reason_large_cpu_exact_skips_targeted: bool = False
    reason_pas_fast_accept: bool = False
    reason_gpu_sparse_skip: bool = False

@dataclass(frozen=True)
class RHS1SparseRescuePolicySetup:
    """Complete sparse-rescue policy setup shared by full and active systems."""

    enabled: bool
    kind_use: str
    ordering: RHS1SparseRescueOrdering
    sparse_jax_est_mb: float | None = None
    sparse_jax_memory_disabled_message: str | None = None

@dataclass(frozen=True)
class RHS1FullSparseRescueSetupContext:
    """Inputs for full-system sparse-rescue setup and message emission."""

    sparse_precond_mode: str
    sparse_precond_kind: str
    has_fp: bool
    has_pas: bool
    residual_norm: float
    target: float
    rhs_mode: int
    include_phi1: bool
    size: int
    sparse_max_size: int
    precond_dtype: Any
    sparse_exact_lu: bool
    use_implicit: bool
    large_cpu_sparse_rescue: bool
    sparse_jax_max_mb: float
    pas_fast_accept: bool
    gpu_sparse_skip: bool
    rhs1_precond_kind: str
    emit: Any = None
    large_cpu_label: str = "large CPU sparse"
    host_sparse_direct_allowed: Any = None
    large_cpu_sparse_exact_lu_allowed: Any = None

@dataclass(frozen=True)
class RHS1FullSparseRescueSetupResult:
    """Full-system sparse-rescue setup state handed back to the driver."""

    policy: RHS1SparseRescuePolicySetup
    ordering: RHS1SparseRescueOrdering
    enabled: bool
    kind_use: str
    sparse_exact_direct: bool
    sparse_exact_lu: bool
    large_cpu_sparse_rescue: bool

@dataclass(frozen=True)
class RHS1SparseJAXConfig:
    """Environment-controlled sparse-JAX retry controls for RHSMode=1."""

    max_mb: float
    sweeps: int
    omega: float
    reg: float

@dataclass(frozen=True)
class RHS1SparsePreconditionerConfig:
    """Environment-controlled sparse preconditioner policy for RHSMode=1."""

    precond_mode: str
    precond_kind: str
    allow_nondiff: bool
    use_matvec: bool
    operator_mode: str
    max_size: int
    pas_sparse_min: int
    drop_tol: float
    drop_rel: float
    ilu_drop_tol: float
    ilu_fill: float
    ilu_dense_max: int
    dense_cache_max: int

@dataclass(frozen=True)
class RHS1SparseOperatorAdmission:
    """Admission result for replacing a reduced matvec with a sparse operator."""

    use_sparse_operator: bool
    messages: tuple[tuple[int, str], ...] = ()

def rhs1_sparse_enabled_initial(
    *,
    sparse_precond_mode: str,
    has_fp: bool,
    has_pas: bool,
    residual_norm: float,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
) -> bool:
    """Resolve the initial sparse-rescue enable bit before ordering/skip rules."""
    enabled = False
    if sparse_precond_mode == "on":
        enabled = True
    elif sparse_precond_mode == "auto":
        enabled = bool(has_fp) or (
            bool(has_pas) and float(residual_norm) > float(target)
        )
    if enabled:
        enabled = int(rhs_mode) == 1 and (not bool(include_phi1))
    return bool(enabled)

def rhs1_sparse_preconditioner_config_from_env(
    *,
    has_pas: bool,
    use_dkes: bool,
    active_size: int,
    backend: str,
) -> RHS1SparsePreconditionerConfig:
    """Parse RHSMode=1 sparse-preconditioner controls with legacy defaults."""

    sparse_precond_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_PRECOND")
    if sparse_precond_env in {"jax", "jax_native", "native"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "jax"
    elif sparse_precond_env in {"scipy", "ilu", "spilu"}:
        sparse_precond_mode = "on"
        sparse_precond_kind = "scipy"
    elif sparse_precond_env in _TRUE_VALUES:
        sparse_precond_mode = "on"
        sparse_precond_kind = "auto"
    elif sparse_precond_env in _FALSE_VALUES:
        sparse_precond_mode = "off"
        sparse_precond_kind = "auto"
    else:
        sparse_precond_mode = "auto"
        sparse_precond_kind = "auto"

    sparse_allow_nondiff = (
        _env_token("SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF") in _TRUE_VALUES
    )
    sparse_matvec_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_MATVEC")
    if sparse_matvec_env in _TRUE_VALUES:
        sparse_use_matvec = True
    elif sparse_matvec_env in _FALSE_VALUES:
        sparse_use_matvec = False
    else:
        sparse_use_matvec = False

    sparse_operator_env = _env_token("SFINCS_JAX_RHSMODE1_SPARSE_OPERATOR")
    if sparse_operator_env in _TRUE_VALUES:
        sparse_operator_mode = "on"
    elif sparse_operator_env in _FALSE_VALUES:
        sparse_operator_mode = "off"
    else:
        sparse_operator_mode = "auto"

    default_sparse_max = 60000 if bool(has_pas) and bool(use_dkes) else 6000
    sparse_max_size = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_MAX", default_sparse_max)
    pas_sparse_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_SPARSE_ILU_MIN", 2000)
    if bool(has_pas) and int(active_size) < max(0, int(pas_sparse_min)):
        sparse_precond_mode = "off"

    default_sparse_ilu_dense_max = 3000 if str(backend).lower() != "cpu" else 2500
    return RHS1SparsePreconditionerConfig(
        precond_mode=str(sparse_precond_mode),
        precond_kind=str(sparse_precond_kind),
        allow_nondiff=bool(sparse_allow_nondiff),
        use_matvec=bool(sparse_use_matvec),
        operator_mode=str(sparse_operator_mode),
        max_size=int(sparse_max_size),
        pas_sparse_min=int(pas_sparse_min),
        drop_tol=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_DROP_TOL", 0.0)),
        drop_rel=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_DROP_REL", 1.0e-8)),
        ilu_drop_tol=float(
            _env_float("SFINCS_JAX_RHSMODE1_SPARSE_ILU_DROP_TOL", 1.0e-4)
        ),
        ilu_fill=float(_env_float("SFINCS_JAX_RHSMODE1_SPARSE_ILU_FILL_FACTOR", 10.0)),
        ilu_dense_max=int(
            _env_int(
                "SFINCS_JAX_RHSMODE1_SPARSE_ILU_DENSE_MAX",
                default_sparse_ilu_dense_max,
            )
        ),
        dense_cache_max=int(_env_int("SFINCS_JAX_RHSMODE1_SPARSE_DENSE_CACHE_MAX", 3000)),
    )

def rhs1_sparse_operator_admission(
    *,
    operator_mode: str,
    use_matvec: bool,
    has_fp: bool,
    rhs_mode: int,
    include_phi1: bool,
    use_implicit: bool,
    allow_nondiff: bool,
    active_size: int,
    sparse_max_size: int,
) -> RHS1SparseOperatorAdmission:
    """Decide whether to materialize the reduced sparse operator matvec."""

    use_sparse_operator = False
    if str(operator_mode) == "on":
        use_sparse_operator = True
    elif str(operator_mode) == "auto":
        use_sparse_operator = bool(use_matvec) and bool(has_fp)
    if use_sparse_operator:
        use_sparse_operator = int(rhs_mode) == 1 and (not bool(include_phi1))

    messages: list[tuple[int, str]] = []
    if use_sparse_operator:
        if bool(use_implicit) and not bool(allow_nondiff):
            use_sparse_operator = False
            messages.append(
                (
                    1,
                    "sparse_operator: disabled for implicit solves "
                    "(set SFINCS_JAX_RHSMODE1_SPARSE_ALLOW_NONDIFF=1 to override)",
                )
            )
        elif int(active_size) > int(sparse_max_size):
            use_sparse_operator = False
            messages.append(
                (
                    1,
                    f"sparse_operator: disabled (size={int(active_size)} > max={int(sparse_max_size)})",
                )
            )

    return RHS1SparseOperatorAdmission(
        use_sparse_operator=bool(use_sparse_operator),
        messages=tuple(messages),
    )

def rhs1_sparse_jax_config_from_env() -> RHS1SparseJAXConfig:
    """Parse sparse-JAX retry controls with bounded, stable defaults."""

    max_mb = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_MAX_MB", 128.0)
    sweeps = max(1, _env_int("SFINCS_JAX_RHSMODE1_SPARSE_JAX_SWEEPS", 2))
    omega = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_OMEGA", 0.8)
    reg = _env_float("SFINCS_JAX_RHSMODE1_SPARSE_JAX_REG", 1.0e-10)
    return RHS1SparseJAXConfig(
        max_mb=float(max_mb),
        sweeps=int(sweeps),
        omega=float(omega),
        reg=float(reg),
    )

def rhs1_sparse_kind_use(*, sparse_precond_kind: str) -> str:
    """Resolve the concrete sparse backend kind used for rescue."""
    return "scipy" if str(sparse_precond_kind) == "auto" else str(sparse_precond_kind)

def rhs1_resolved_sparse_rescue_ordering(
    *,
    sparse_enabled: bool,
    sparse_kind_use: str,
    dense_shortcut: bool = False,
    sparse_exact_direct: bool = False,
    size: int,
    sparse_max_size: int,
    large_cpu_sparse_rescue: bool = False,
    sparse_xblock_rescue_active: bool = False,
    sparse_jax_est_mb: float | None = None,
    sparse_jax_max_mb: float = 0.0,
    pas_fast_accept: bool = False,
    gpu_sparse_skip: bool = False,
) -> RHS1SparseRescueOrdering:
    """Apply sparse-rescue ordering and skip decisions without side effects."""
    enabled = bool(sparse_enabled)
    kind_use = rhs1_sparse_kind_use(sparse_precond_kind=str(sparse_kind_use))
    xblock_active = bool(sparse_xblock_rescue_active)
    prefer_sparse_exact_over_dense_shortcut = False
    reason_dense_shortcut_skip = False
    reason_size_disabled = False
    reason_size_large_cpu = False
    reason_size_exact_direct = False
    reason_size_targeted = False
    reason_sparse_jax_mem_disabled = False
    reason_large_cpu_exact_skips_targeted = False
    reason_pas_fast_accept = False
    reason_gpu_sparse_skip = False
    if enabled and bool(dense_shortcut):
        if bool(sparse_exact_direct):
            prefer_sparse_exact_over_dense_shortcut = True
        else:
            enabled = False
            reason_dense_shortcut_skip = True
    if enabled and int(size) > int(sparse_max_size):
        if bool(large_cpu_sparse_rescue):
            reason_size_large_cpu = True
        elif bool(sparse_exact_direct):
            reason_size_exact_direct = True
        elif xblock_active:
            reason_size_targeted = True
        else:
            enabled = False
            reason_size_disabled = True
    if enabled and str(kind_use) == "jax" and (sparse_jax_est_mb is not None):
        if float(sparse_jax_max_mb) > 0.0 and float(sparse_jax_est_mb) > float(
            sparse_jax_max_mb
        ):
            enabled = False
            reason_sparse_jax_mem_disabled = True
    if bool(large_cpu_sparse_rescue) and bool(sparse_exact_direct):
        xblock_active = False
        reason_large_cpu_exact_skips_targeted = True
    if bool(pas_fast_accept):
        enabled = False
        reason_pas_fast_accept = True
    if bool(gpu_sparse_skip):
        enabled = False
        reason_gpu_sparse_skip = True
    return RHS1SparseRescueOrdering(
        enabled=bool(enabled),
        kind_use=str(kind_use),
        xblock_rescue_active=bool(xblock_active),
        prefer_sparse_exact_over_dense_shortcut=bool(
            prefer_sparse_exact_over_dense_shortcut
        ),
        reason_dense_shortcut_skip=bool(reason_dense_shortcut_skip),
        reason_size_disabled=bool(reason_size_disabled),
        reason_size_large_cpu=bool(reason_size_large_cpu),
        reason_size_exact_direct=bool(reason_size_exact_direct),
        reason_size_targeted=bool(reason_size_targeted),
        reason_sparse_jax_mem_disabled=bool(reason_sparse_jax_mem_disabled),
        reason_large_cpu_exact_skips_targeted=bool(
            reason_large_cpu_exact_skips_targeted
        ),
        reason_pas_fast_accept=bool(reason_pas_fast_accept),
        reason_gpu_sparse_skip=bool(reason_gpu_sparse_skip),
    )

def rhs1_sparse_rescue_policy_setup(
    *,
    sparse_precond_mode: str,
    sparse_precond_kind: str,
    has_fp: bool,
    has_pas: bool,
    residual_norm: float,
    target: float,
    rhs_mode: int,
    include_phi1: bool,
    size: int,
    sparse_max_size: int,
    precond_dtype: Any,
    dense_shortcut: bool = False,
    sparse_exact_direct: bool = False,
    large_cpu_sparse_rescue: bool = False,
    sparse_xblock_rescue_active: bool = False,
    sparse_jax_max_mb: float = 0.0,
    pas_fast_accept: bool = False,
    gpu_sparse_skip: bool = False,
) -> RHS1SparseRescuePolicySetup:
    """Resolve sparse-rescue policy and its JAX dense-memory admission estimate."""

    sparse_enabled = rhs1_sparse_enabled_initial(
        sparse_precond_mode=sparse_precond_mode,
        has_fp=bool(has_fp),
        has_pas=bool(has_pas),
        residual_norm=float(residual_norm),
        target=float(target),
        rhs_mode=int(rhs_mode),
        include_phi1=bool(include_phi1),
    )
    sparse_kind_use = rhs1_sparse_kind_use(sparse_precond_kind=sparse_precond_kind)
    sparse_jax_est_mb: float | None = None
    if sparse_enabled and sparse_kind_use == "jax" and int(size) <= int(sparse_max_size):
        bytes_per = float(np.dtype(precond_dtype).itemsize)
        sparse_jax_est_mb = (int(size) ** 2) * bytes_per / 1.0e6

    ordering = rhs1_resolved_sparse_rescue_ordering(
        sparse_enabled=bool(sparse_enabled),
        sparse_kind_use=sparse_kind_use,
        dense_shortcut=bool(dense_shortcut),
        sparse_exact_direct=bool(sparse_exact_direct),
        size=int(size),
        sparse_max_size=int(sparse_max_size),
        large_cpu_sparse_rescue=bool(large_cpu_sparse_rescue),
        sparse_xblock_rescue_active=bool(sparse_xblock_rescue_active),
        sparse_jax_est_mb=sparse_jax_est_mb,
        sparse_jax_max_mb=float(sparse_jax_max_mb),
        pas_fast_accept=bool(pas_fast_accept),
        gpu_sparse_skip=bool(gpu_sparse_skip),
    )
    sparse_jax_memory_disabled_message: str | None = None
    if ordering.reason_sparse_jax_mem_disabled and sparse_jax_est_mb is not None:
        sparse_jax_memory_disabled_message = (
            "sparse_jax: disabled "
            f"(est_mem={sparse_jax_est_mb:.1f} MB > max_mb={float(sparse_jax_max_mb):.1f})"
        )
    return RHS1SparseRescuePolicySetup(
        enabled=bool(ordering.enabled),
        kind_use=str(ordering.kind_use),
        ordering=ordering,
        sparse_jax_est_mb=sparse_jax_est_mb,
        sparse_jax_memory_disabled_message=sparse_jax_memory_disabled_message,
    )

def rhs1_sparse_rescue_initial_messages(
    *,
    ordering: RHS1SparseRescueOrdering,
    size: int,
    sparse_max_size: int,
    sparse_jax_memory_disabled_message: str | None = None,
    large_cpu_sparse_exact_lu: bool | None = None,
    large_cpu_label: str = "large CPU sparse",
    targeted_rescue_kind: str | None = None,
) -> tuple[tuple[int, str], ...]:
    """Format initial sparse-rescue policy progress messages without side effects."""

    messages: list[tuple[int, str]] = []
    if ordering.prefer_sparse_exact_over_dense_shortcut:
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: preferring sparse exact rescue over dense shortcut",
            )
        )
    if ordering.reason_size_large_cpu:
        sparse_exact_lu = bool(large_cpu_sparse_exact_lu)
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: "
                f"{large_cpu_label} {'LU' if sparse_exact_lu else 'ILU'} rescue "
                f"(size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_exact_direct:
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: exact sparse LU rescue "
                f"(size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_targeted:
        rescue_kind = str(targeted_rescue_kind or "targeted")
        messages.append(
            (
                0,
                "solve_v3_full_system_linear_gmres: targeted sparse "
                f"{rescue_kind} rescue (size={int(size)} > max={int(sparse_max_size)})",
            )
        )
    elif ordering.reason_size_disabled:
        messages.append(
            (1, f"sparse_ilu: disabled (size={int(size)} > max={int(sparse_max_size)})")
        )
    if sparse_jax_memory_disabled_message is not None:
        messages.append((1, sparse_jax_memory_disabled_message))
    return tuple(messages)

def rhs1_sparse_rescue_tail_skip_messages(
    *,
    ordering: RHS1SparseRescueOrdering,
    residual_norm: float,
    rhs1_precond_kind: str,
) -> tuple[tuple[int, str], ...]:
    """Format sparse-rescue tail skip messages without moving driver control flow."""

    messages: list[tuple[int, str]] = []
    if ordering.reason_large_cpu_exact_skips_targeted:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: exact large-CPU sparse LU selected "
                "-> skipping targeted sparse xblock/sxblock rescue",
            )
        )
    if ordering.reason_pas_fast_accept:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: PAS fast-accept "
                f"(residual={float(residual_norm):.3e}) -> skip sparse rescue tail",
            )
        )
    if ordering.reason_gpu_sparse_skip:
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: GPU sparse fallback skipped after "
                f"{rhs1_precond_kind} accept (residual={float(residual_norm):.3e})",
            )
        )
    return tuple(messages)

def rhs1_full_sparse_rescue_setup(
    context: RHS1FullSparseRescueSetupContext,
) -> RHS1FullSparseRescueSetupResult:
    """Resolve full-system sparse-rescue policy and emit progress messages."""

    if context.host_sparse_direct_allowed is None:
        sparse_exact_direct = bool(context.sparse_exact_lu) and not bool(
            context.use_implicit
        )
    else:
        sparse_exact_direct = bool(
            context.host_sparse_direct_allowed(
                sparse_exact_lu=bool(context.sparse_exact_lu),
                use_implicit=bool(context.use_implicit),
            )
        )
    policy = rhs1_sparse_rescue_policy_setup(
        sparse_precond_mode=context.sparse_precond_mode,
        sparse_precond_kind=context.sparse_precond_kind,
        has_fp=bool(context.has_fp),
        has_pas=bool(context.has_pas),
        residual_norm=float(context.residual_norm),
        target=float(context.target),
        rhs_mode=int(context.rhs_mode),
        include_phi1=bool(context.include_phi1),
        size=int(context.size),
        sparse_max_size=int(context.sparse_max_size),
        precond_dtype=context.precond_dtype,
        sparse_exact_direct=bool(sparse_exact_direct),
        large_cpu_sparse_rescue=bool(context.large_cpu_sparse_rescue),
        sparse_jax_max_mb=float(context.sparse_jax_max_mb),
        pas_fast_accept=bool(context.pas_fast_accept),
        gpu_sparse_skip=bool(context.gpu_sparse_skip),
    )
    ordering = policy.ordering
    sparse_exact_lu = bool(context.sparse_exact_lu)
    if ordering.reason_size_large_cpu:
        if context.large_cpu_sparse_exact_lu_allowed is None:
            sparse_exact_lu = bool(context.sparse_exact_lu)
        else:
            sparse_exact_lu = bool(
                context.large_cpu_sparse_exact_lu_allowed(active_size=int(context.size))
            )
    if context.emit is not None:
        for level, message in rhs1_sparse_rescue_initial_messages(
            ordering=ordering,
            size=int(context.size),
            sparse_max_size=int(context.sparse_max_size),
            sparse_jax_memory_disabled_message=policy.sparse_jax_memory_disabled_message,
            large_cpu_sparse_exact_lu=bool(sparse_exact_lu),
            large_cpu_label=str(context.large_cpu_label),
        ):
            context.emit(level, message)
        for level, message in rhs1_sparse_rescue_tail_skip_messages(
            ordering=ordering,
            residual_norm=float(context.residual_norm),
            rhs1_precond_kind=str(context.rhs1_precond_kind),
        ):
            context.emit(level, message)
    return RHS1FullSparseRescueSetupResult(
        policy=policy,
        ordering=ordering,
        enabled=bool(policy.enabled),
        kind_use=str(policy.kind_use),
        sparse_exact_direct=bool(sparse_exact_direct),
        sparse_exact_lu=bool(sparse_exact_lu),
        large_cpu_sparse_rescue=bool(context.large_cpu_sparse_rescue),
    )

# From sfincs_jax.problems.profile_policies
def rhs1_polish_enabled(*, env_name: str) -> bool:
    """Return whether a polish stage is enabled by its boolean-like env var."""
    env = os.environ.get(env_name, "").strip().lower()
    return env not in {"0", "false", "no", "off"}

def rhs1_parse_accept_ratio(*, env_name: str, default: float) -> float:
    """Parse an acceptance ratio with a floor of 1."""
    env = os.environ.get(env_name, "").strip()
    try:
        value = float(env) if env else float(default)
    except ValueError:
        value = float(default)
    return max(1.0, float(value))

def rhs1_parse_polish_gmres_config(
    *,
    restart_env_name: str,
    maxiter_env_name: str,
    default_restart: int,
    default_maxiter: int,
    min_restart: int = 5,
    min_maxiter: int = 5,
    active_size: int | None = None,
    large_active_min_env_name: str = "",
    large_default_restart_env_name: str = "",
    large_default_maxiter_env_name: str = "",
    default_large_restart: int | None = None,
    default_large_maxiter: int | None = None,
) -> tuple[int, int]:
    """Parse bounded restart/maxiter settings for a short GMRES polish."""
    restart_env = os.environ.get(restart_env_name, "").strip()
    maxiter_env = os.environ.get(maxiter_env_name, "").strip()
    default_restart_use = int(default_restart)
    default_maxiter_use = int(default_maxiter)
    if active_size is not None and (
        default_large_restart is not None or default_large_maxiter is not None
    ):
        large_min_env = (
            os.environ.get(large_active_min_env_name, "").strip()
            if large_active_min_env_name
            else ""
        )
        large_restart_env = (
            os.environ.get(large_default_restart_env_name, "").strip()
            if large_default_restart_env_name
            else ""
        )
        large_default_env = (
            os.environ.get(large_default_maxiter_env_name, "").strip()
            if large_default_maxiter_env_name
            else ""
        )
        try:
            large_min = int(large_min_env) if large_min_env else 200000
        except ValueError:
            large_min = 200000
        if int(active_size) >= max(1, int(large_min)):
            if default_large_restart is not None and (not restart_env):
                try:
                    large_restart = (
                        int(large_restart_env)
                        if large_restart_env
                        else int(default_large_restart)
                    )
                except ValueError:
                    large_restart = int(default_large_restart)
                default_restart_use = min(
                    int(default_restart_use), max(1, int(large_restart))
                )
            if default_large_maxiter is not None and (not maxiter_env):
                try:
                    large_default = (
                        int(large_default_env)
                        if large_default_env
                        else int(default_large_maxiter)
                    )
                except ValueError:
                    large_default = int(default_large_maxiter)
                default_maxiter_use = min(
                    int(default_maxiter_use), max(1, int(large_default))
                )
    try:
        restart = int(restart_env) if restart_env else int(default_restart_use)
    except ValueError:
        restart = int(default_restart_use)
    try:
        maxiter = int(maxiter_env) if maxiter_env else int(default_maxiter_use)
    except ValueError:
        maxiter = int(default_maxiter_use)
    return (max(int(min_restart), int(restart)), max(int(min_maxiter), int(maxiter)))

# From sfincs_jax.problems.profile_policies
_PAS_STAGE2_SKIP_BASE_KINDS = frozenset(
    {"pas_lite", "pas_hybrid", "pas_tz", "pas_schur", "pas_tokamak_theta"}
)

_PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS = frozenset(
    {"pas_ilu", "schur", "xblock_tz", "xblock_tz_lmax"}
)

_PAS_STAGE2_WEAK_SKIP_KINDS = frozenset({"collision", "point", "xmg"})

@dataclass(frozen=True)
class RHS1Stage2RetryControls:
    """Restart/maxiter/method controls for a Stage-2 RHSMode=1 Krylov retry."""

    restart: int
    maxiter: int
    method: str

@dataclass(frozen=True)
class RHS1Stage2AdmissionControls:
    """Admission and elapsed-time budget controls for Stage-2 fallback solves."""

    enabled: bool
    time_cap_s: float

@dataclass(frozen=True)
class RHS1Stage2TriggerDecision:
    """Resolved Stage-2 trigger state and progress messages."""

    stage2_trigger: bool
    fp_force_stage2: bool
    messages: tuple[tuple[int, str], ...] = ()

@dataclass(frozen=True)
class RHS1Stage2RetryAdmissionDecision:
    """Resolved Stage-2 retry execution gate and progress messages."""

    run_retry: bool
    messages: tuple[tuple[int, str], ...] = ()

def rhs1_stage2_ratio(*, use_dkes: bool) -> float:
    """Return the stage-2 residual-ratio trigger with DKES tightening."""
    stage2_ratio_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RATIO", "").strip()
    try:
        stage2_ratio = float(stage2_ratio_env) if stage2_ratio_env else 100.0
    except ValueError:
        stage2_ratio = 100.0
    if use_dkes:
        stage2_ratio = min(float(stage2_ratio), 1.0)
    return float(stage2_ratio)

def rhs1_stage2_trigger(*, res_ratio: float, use_dkes: bool) -> bool:
    """Return whether stage-2 should be considered from the residual ratio."""
    ratio = rhs1_stage2_ratio(use_dkes=use_dkes)
    return bool(res_ratio > ratio) if ratio > 0 else True

def rhs1_stage2_admission_controls_from_env(
    *,
    rhs_mode: int,
    include_phi1: bool,
    solver_kind_default: str,
    pas_large_bicgstab_fastpath: bool,
    tokamak_pas: bool,
    has_fp: bool,
    use_dkes: bool,
    total_size: int,
) -> RHS1Stage2AdmissionControls:
    """Resolve whether Stage-2 is available and how long it may run."""

    stage2_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2", "").strip().lower()
    if stage2_env in {"0", "false", "no", "off"}:
        enabled = False
    elif stage2_env in {"1", "true", "yes", "on"}:
        enabled = True
    else:
        enabled = (
            int(rhs_mode) == 1
            and (not bool(include_phi1))
            and str(solver_kind_default) in {"gmres", "bicgstab"}
        )
    if pas_large_bicgstab_fastpath and stage2_env == "":
        enabled = False

    # Stage-2 is a stronger fallback solve for difficult cases. The default
    # time cap must be large enough to include one-time preconditioner setup,
    # while remaining bounded for interactive use and CI.
    time_cap_s = float(os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", "30.0"))
    if tokamak_pas and time_cap_s < 120.0:
        time_cap_s = 120.0
    if has_fp and use_dkes and time_cap_s < 120.0:
        time_cap_s = 120.0
    if has_fp and int(total_size) >= 300000 and time_cap_s < 1200.0:
        time_cap_s = 1200.0
    if has_fp and int(total_size) >= 600000 and time_cap_s < 2400.0:
        time_cap_s = 2400.0
    return RHS1Stage2AdmissionControls(enabled=bool(enabled), time_cap_s=float(time_cap_s))

def rhs1_stage2_retry_controls_from_env(
    *,
    restart: int,
    maxiter: int | None,
    active_size: int,
    has_fp: bool,
    has_pas: bool,
    tokamak_pas: bool = False,
) -> RHS1Stage2RetryControls:
    """Resolve Stage-2 retry Krylov bounds without changing legacy defaults."""

    maxiter_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_MAXITER", "").strip()
    restart_env = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_RESTART", "").strip()
    maxiter_use = int(maxiter_env or str(max(600, int(maxiter or 400) * 2)))
    restart_use = int(restart_env or str(max(120, int(restart))))

    if tokamak_pas and (not maxiter_env):
        maxiter_use = max(int(maxiter_use), 2000)
    if tokamak_pas and (not restart_env):
        restart_use = max(int(restart_use), 160)
    if has_fp and (not has_pas) and int(active_size) >= 300000 and (not maxiter_env):
        # Large FP systems often need Stage-2 to close diagnostics, but the
        # historical maxiter=800 default is unnecessarily expensive in practice.
        maxiter_use = min(int(maxiter_use), 600)
    if has_fp and (not has_pas) and int(active_size) >= 300000 and (not restart_env):
        restart_use = min(max(80, int(restart_use)), 100)

    method = os.environ.get("SFINCS_JAX_LINEAR_STAGE2_METHOD", "incremental").strip().lower()
    if method not in {"batched", "incremental", "dense"}:
        method = "incremental"
    return RHS1Stage2RetryControls(
        restart=int(restart_use),
        maxiter=int(maxiter_use),
        method=method,
    )

def rhs1_fp_force_stage2(
    *, has_fp: bool, include_phi1: bool, residual_norm: float
) -> bool:
    """Return whether FP runs should force a stage-2 polish based on absolute residual."""
    fp_stage2_abs_env = os.environ.get("SFINCS_JAX_FP_STAGE2_ABS", "").strip()
    try:
        fp_stage2_abs = float(fp_stage2_abs_env) if fp_stage2_abs_env else 1e-06
    except ValueError:
        fp_stage2_abs = 1e-06
    return bool(
        has_fp and (not include_phi1) and (float(residual_norm) > float(fp_stage2_abs))
    )

def rhs1_pas_stage2_skip(
    *, has_pas: bool, rhs1_precond_kind: str | None, res_ratio: float
) -> bool:
    """Return whether PAS runs should skip stage-2 and move to later rescue logic.

    Stage-2 GMRES is useful as a polish when the first residual is close enough
    to target. For the historical PAS-lite/hybrid/tz family, very large
    residual ratios should move directly to later rescue logic. Broader skips
    for Schur/xblock/PAS-ILU routes are opt-in only because production-floor
    tests show they can produce faster but non-parity-clean completed outputs.
    """
    if not has_pas:
        return False
    if rhs1_precond_kind not in _PAS_STAGE2_SKIP_BASE_KINDS:
        if rhs1_precond_kind in _PAS_STAGE2_WEAK_SKIP_KINDS:
            weak_skip_env = os.environ.get(
                "SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO", ""
            ).strip()
            try:
                weak_skip_ratio = (
                    float(weak_skip_env) if weak_skip_env else 1000000000000.0
                )
            except ValueError:
                weak_skip_ratio = 1000000000000.0
            if weak_skip_ratio <= 0.0:
                return False
            return float(res_ratio) >= float(weak_skip_ratio)
        extended_env = (
            os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED", "").strip().lower()
        )
        if extended_env not in {"1", "true", "yes", "on"}:
            return False
        if rhs1_precond_kind not in _PAS_STAGE2_EXTENDED_SKIP_BASE_KINDS:
            return False
    pas_stage2_skip_env = os.environ.get("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "").strip()
    try:
        pas_stage2_skip_ratio = (
            float(pas_stage2_skip_env) if pas_stage2_skip_env else 1000000.0
        )
    except ValueError:
        pas_stage2_skip_ratio = 1000000.0
    return float(res_ratio) >= float(pas_stage2_skip_ratio)

def rhs1_pas_tz_guarded_stage2_retry() -> bool:
    """Return whether guarded PAS-TZ fallbacks should attempt stage-2 GMRES.

    Guarded PAS-TZ fallbacks are selected after the dense structured PAS-TZ
    builder is rejected by the memory gate. Their purpose is to keep the run
    bounded and diagnostic-rich; strict stage-2 retries can turn an otherwise
    bounded fallback into the same long-running solver-path problem the guard is
    meant to avoid. Users can still opt in when profiling a candidate fallback.
    """
    env = (
        os.environ.get("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY", "")
        .strip()
        .lower()
    )
    return env in {"1", "true", "yes", "on"}

def rhs1_stage2_trigger_decision(
    *,
    res_ratio: float,
    use_dkes: bool,
    has_fp: bool,
    include_phi1: bool,
    residual_norm: float,
    cpu_large_xblock_shortcut: bool,
    cpu_large_sparse_shortcut: bool,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    pas_tz_guarded_fallback: bool,
    pas_tz_guarded_retry: bool,
) -> RHS1Stage2TriggerDecision:
    """Resolve Stage-2 trigger/skip state after the primary residual is known."""

    stage2_trigger = rhs1_stage2_trigger(res_ratio=float(res_ratio), use_dkes=bool(use_dkes))
    fp_force_stage2 = rhs1_fp_force_stage2(
        has_fp=bool(has_fp),
        include_phi1=bool(include_phi1),
        residual_norm=float(residual_norm),
    )
    if fp_force_stage2:
        stage2_trigger = True
    messages: list[tuple[int, str]] = []
    if cpu_large_xblock_shortcut:
        stage2_trigger = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: CPU large FP x-block shortcut "
                "skipping stage2 GMRES and proceeding directly to x-block rescue",
            )
        )
    if cpu_large_sparse_shortcut:
        stage2_trigger = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: CPU large FP sparse-LU shortcut "
                "skipping stage2 GMRES and proceeding directly to sparse rescue",
            )
        )
    if rhs1_pas_stage2_skip(
        has_pas=bool(has_pas),
        rhs1_precond_kind=rhs1_precond_kind,
        res_ratio=float(res_ratio),
    ):
        stage2_trigger = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: PAS stage2 skipped "
                f"(residual ratio={float(res_ratio):.3e}; set the relevant PAS stage2 skip ratio to 0 to retry)",
            )
        )
    if bool(pas_tz_guarded_fallback) and bool(stage2_trigger) and not bool(pas_tz_guarded_retry):
        stage2_trigger = False
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: stage2 reduced GMRES skipped "
                "after guarded PAS-TZ fallback; set "
                "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY=1 to retry",
            )
        )
    return RHS1Stage2TriggerDecision(
        stage2_trigger=bool(stage2_trigger),
        fp_force_stage2=bool(fp_force_stage2),
        messages=tuple(messages),
    )

def rhs1_stage2_retry_admission_decision(
    *,
    residual_norm: float,
    target: float,
    fp_force_stage2: bool,
    stage2_enabled: bool,
    stage2_trigger: bool,
    early_dense_shortcut: bool,
    gpu_dkes_sparse_shortcut: bool,
    sparse_prefer_skips_stage2: bool,
    elapsed_s: float,
    time_cap_s: float,
) -> RHS1Stage2RetryAdmissionDecision:
    """Resolve whether the Stage-2 retry should run after trigger policy."""

    residual_above_target = float(residual_norm) > float(target)
    common_admitted = bool(stage2_enabled) and bool(stage2_trigger) and (
        not bool(early_dense_shortcut)
    ) and (not bool(gpu_dkes_sparse_shortcut))
    messages: list[tuple[int, str]] = []
    if (
        bool(sparse_prefer_skips_stage2)
        and residual_above_target
        and common_admitted
    ):
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: stage2 reduced GMRES skipped "
                "(preferring sparse rescue first)",
            )
        )
    run_retry = bool(
        (residual_above_target or bool(fp_force_stage2))
        and common_admitted
        and (not bool(sparse_prefer_skips_stage2))
        and float(elapsed_s) < float(time_cap_s)
    )
    return RHS1Stage2RetryAdmissionDecision(
        run_retry=bool(run_retry),
        messages=tuple(messages),
    )

# Consolidated host/dense/sparse policy section

def _env_bool(name: str) -> bool | None:
    env = str(os.environ.get(name, "")).strip().lower()
    if env in _TRUE_VALUES:
        return True
    if env in _FALSE_VALUES:
        return False
    return None

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
    """Allow bounded accelerator full-FP systems to use host dense LU directly.

    GPU RHSMode=1 full-FP systems below the dense fallback budget are often
    faster if they go straight to the host dense solve rather than first paying
    for Krylov/preconditioner probes on the accelerator. The explicit shortcut
    cap still lets memory-constrained users lower or disable this path.
    """
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
    shortcut_default = min(
        int(dense_fallback_max),
        rhs1_dense_auto_fp_cutoff(dense_active_cutoff=int(dense_fallback_max)),
    )
    shortcut_max = _env_int(
        "SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX",
        max(0, int(shortcut_default)),
    )
    dense_cap = min(max(0, int(shortcut_max)), max(0, int(dense_fallback_max)))
    if dense_cap <= 0:
        return False
    if int(active_size) > dense_cap:
        return False
    max_bytes = rhs1_host_dense_shortcut_max_bytes()
    if max_bytes > 0 and rhs1_host_dense_shortcut_estimated_nbytes(active_size) > max_bytes:
        return False
    return True

def rhs1_host_dense_shortcut_max_bytes() -> int:
    """Return the host-dense shortcut memory admission ceiling in bytes.

    The route is intended for bounded accelerator systems where avoiding GPU
    dense scratch and preconditioner probes is faster. Keep a separate memory
    ceiling so raising the active-size cutoff cannot silently turn the shortcut
    into a large host factorization path. Set the environment value to ``0`` to
    disable the byte ceiling for one-off diagnostics.
    """

    return _env_int("SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_MAX_BYTES", 1_500_000_000)

def rhs1_host_dense_shortcut_estimated_nbytes(active_size: int) -> int:
    """Estimate host dense matrix/factor/work storage for shortcut admission."""

    n = max(0, int(active_size))
    matrix_nbytes = n * n * 8
    factor_overhead = _env_float(
        "SFINCS_JAX_RHSMODE1_HOST_DENSE_SHORTCUT_FACTOR_OVERHEAD",
        2.5,
    )
    work_nbytes = n * 8 * 6
    return int(np.ceil(matrix_nbytes * max(1.0, float(factor_overhead)) + work_nbytes))

def rhs1_dense_fallback_max(op: Any) -> int:
    """Resolve the RHSMode=1 dense fallback active-size ceiling.

    Full Fokker-Planck systems use a larger conservative default because dense
    fallback is often the cheapest robust path for small/medium FP systems. PAS
    systems are stricter: dense fallback can drift away from PETSc-style
    approximate branches, so PAS is disabled by default except for
    ``constraintScheme=0`` or explicit user opt-in.
    """
    base_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX", 400)
    if op.fblock.fp is None:
        dense_pas_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", "")).strip()
        if dense_pas_raw:
            dense_pas_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_PAS_MAX", base_max)
            if dense_pas_max <= 0:
                return 0
            return max(base_max, dense_pas_max)
        if int(op.constraint_scheme) != 0:
            return 0
        dense_pas_max = 5000
        if dense_pas_max <= 0:
            return base_max
        return max(base_max, dense_pas_max)

    dense_fp_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", "")).strip()
    dense_fp_cutoff_raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "")).strip()
    if dense_fp_raw:
        dense_fp_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_MAX", base_max)
    elif dense_fp_cutoff_raw:
        dense_fp_max = _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", base_max)
    else:
        dense_fp_max = 8000
    if dense_fp_max <= 0:
        return base_max
    return max(base_max, dense_fp_max)

def rhs1_dense_auto_fp_cutoff(*, dense_active_cutoff: int) -> int:
    """Resolve the initial dense-solve cutoff for full-FP RHSMode=1 systems.

    This is the pre-Krylov auto-selection threshold used by the CLI/output
    writer. It intentionally matches the default full-FP dense fallback budget
    (8000 active unknowns) so moderate FP systems do not first run through the
    expensive Krylov/strong/sparse rescue ladder. Users may still disable the
    initial dense path with ``SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF=0`` or lower it
    for memory-constrained hosts.
    """
    raw = str(os.environ.get("SFINCS_JAX_RHSMODE1_DENSE_FP_CUTOFF", "")).strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return min(max(0, int(dense_active_cutoff)), 8000)

def rhs1_dense_auto_fp_accelerator_min() -> int:
    """Minimum active size for default accelerator dense auto-selection.

    Tiny GPU full-FP systems are usually faster on the existing matrix-free path
    because dense assembly/solver setup dominates. Moderate systems can avoid the
    expensive Krylov/preconditioner ladder, so enable accelerator dense auto only
    above this floor unless the user explicitly overrides the solve method.
    """
    return max(0, _env_int("SFINCS_JAX_RHSMODE1_DENSE_FP_ACCELERATOR_MIN", 1000))

def rhs1_dense_auto_fp_allowed(
    *,
    backend: str,
    active_size: int,
    dense_active_cutoff: int,
) -> bool:
    """Return whether full-FP RHSMode=1 auto mode should start with dense LU.

    CPU defaults use the dense path for all systems below the FP cutoff. On
    accelerators, the default dense route is the explicit host-dense shortcut
    because it avoids XLA dense-solve scratch allocations and the Krylov/probe
    ladder. Users can still opt into accelerator dense linear algebra with
    ``SFINCS_JAX_RHSMODE1_DENSE_ALLOW_ACCELERATOR=1``.
    """
    cutoff = rhs1_dense_auto_fp_cutoff(dense_active_cutoff=dense_active_cutoff)
    if cutoff <= 0 or int(active_size) > int(cutoff):
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm == "cpu":
        return True
    if not rhs1_dense_backend_allowed(backend=backend_norm):
        return False
    return int(active_size) >= int(rhs1_dense_auto_fp_accelerator_min())

@dataclass(frozen=True)
class RHS1FullFPDenseAutoRouteDecision:
    """Route decision for bounded full-FP RHSMode=1 dense auto solves."""

    solve_method: str
    solve_method_kind: str
    selected: bool
    cutoff: int
    messages: tuple[tuple[int, str], ...] = ()

def resolve_rhs1_full_fp_dense_auto_route(
    *,
    solve_method: str,
    solve_method_kind: str,
    use_implicit: bool,
    has_fp: bool,
    has_pas: bool,
    include_phi1: bool,
    rhs_mode: int,
    active_size: int,
    dense_active_cutoff: int,
    backend: str,
) -> RHS1FullFPDenseAutoRouteDecision:
    """Resolve the first full-FP RHSMode=1 auto route before Krylov setup.

    The dense route is intentionally a policy decision rather than solve logic:
    it prevents moderate full-FP systems from paying the Krylov/preconditioner
    setup cost when the bounded dense path is admitted by the memory policy.
    """

    method = str(solve_method)
    method_kind = str(solve_method_kind).strip().lower()
    cutoff = rhs1_dense_auto_fp_cutoff(dense_active_cutoff=int(dense_active_cutoff))
    selected = bool(
        method_kind in {"auto", "default", "incremental"}
        and (not bool(use_implicit))
        and bool(has_fp)
        and (not bool(has_pas))
        and (not bool(include_phi1))
        and int(rhs_mode) == 1
        and rhs1_dense_auto_fp_allowed(
            backend=str(backend),
            active_size=int(active_size),
            dense_active_cutoff=int(dense_active_cutoff),
        )
    )
    if not selected:
        return RHS1FullFPDenseAutoRouteDecision(
            solve_method=method,
            solve_method_kind=method_kind,
            selected=False,
            cutoff=int(cutoff),
        )
    return RHS1FullFPDenseAutoRouteDecision(
        solve_method="dense",
        solve_method_kind="dense",
        selected=True,
        cutoff=int(cutoff),
        messages=(
            (
                1,
                "solve_v3_full_system_linear_gmres: auto-selected dense "
                f"full-FP solve (size={int(active_size)} <= cutoff={int(cutoff)})",
            ),
        ),
    )

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

@dataclass(frozen=True)
class RHS1InitialSparseShortcutRouteDecision:
    """Initial sparse-route decision and preconditioner-state updates."""

    gpu_dkes_sparse_shortcut: bool
    cs0_sparse_first: bool
    cs0_petsc_compat: bool
    cs0_dense_fallback_allowed: bool
    rhs1_precond_kind: str | None
    rhs1_precond_enabled: bool
    rhs1_bicgstab_kind: str | None
    messages: tuple[tuple[int, str], ...] = ()

def resolve_rhs1_initial_sparse_shortcut_route(
    *,
    op: Any,
    rhs1_precond_env_user: str,
    rhs1_bicgstab_env_user: str,
    rhs1_precond_kind: str | None,
    rhs1_precond_enabled: bool,
    rhs1_bicgstab_kind: str | None,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
    use_dkes: bool,
    backend: str,
) -> RHS1InitialSparseShortcutRouteDecision:
    """Resolve sparse-first lanes that bypass the default preconditioner build."""

    method_kind = str(solve_method_kind).strip().lower()
    backend_norm = str(backend).strip().lower()
    cs0_sparse_first = rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind=method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        backend=backend_norm,
    )
    cs0_petsc_compat = rhs1_constraint0_petsc_compat(
        op=op,
        solve_method_kind=method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
    )
    gpu_dkes_sparse_shortcut = bool(
        str(rhs1_precond_env_user).strip().lower() in {"", "auto"}
        and str(rhs1_bicgstab_env_user).strip().lower() in {"", "auto"}
        and method_kind not in {"dense", "dense_ksp"}
        and backend_norm != "cpu"
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and bool(use_dkes)
        and str(sparse_precond_mode).strip().lower() != "off"
        and int(active_size) <= int(sparse_max_size)
    )
    messages: list[tuple[int, str]] = []
    precond_kind_out = rhs1_precond_kind
    precond_enabled_out = bool(rhs1_precond_enabled)
    bicgstab_kind_out = rhs1_bicgstab_kind
    if cs0_petsc_compat:
        precond_kind_out = None
        precond_enabled_out = False
        bicgstab_kind_out = None
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: constraintScheme=0 PETSc-compat auto mode "
                "-> dedicated sparse ILU path",
            )
        )
    if gpu_dkes_sparse_shortcut:
        precond_kind_out = None
        precond_enabled_out = False
        bicgstab_kind_out = None
        messages.append(
            (
                1,
                "solve_v3_full_system_linear_gmres: GPU DKES auto mode -> sparse ILU shortcut "
                f"(size={int(active_size)})",
            )
        )
    return RHS1InitialSparseShortcutRouteDecision(
        gpu_dkes_sparse_shortcut=bool(gpu_dkes_sparse_shortcut),
        cs0_sparse_first=bool(cs0_sparse_first),
        cs0_petsc_compat=bool(cs0_petsc_compat),
        cs0_dense_fallback_allowed=rhs1_constraint0_dense_fallback_allowed(op),
        rhs1_precond_kind=precond_kind_out,
        rhs1_precond_enabled=bool(precond_enabled_out),
        rhs1_bicgstab_kind=bicgstab_kind_out,
        messages=tuple(messages),
    )

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

def rhs1_constrained_pas_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
) -> bool:
    """Return whether large constrained-PAS RHSMode=1 should start sparse-PC GMRES.

    The matrix-free PAS path is robust for small examples, but production-sized
    finite-beta profile-current decks can spend many minutes in Krylov fallback
    and still stall at a large true residual.  The host sparse-PC branch builds
    the same explicit operator sparsity used for diagnostics, factors the
    RHSMode=1 preconditioner, and then polishes the true residual with GMRES.

    Keep this as a narrow non-differentiable policy: it is a CLI/production
    solve path, not the JAX-native autodiff route.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC")
    if env is False:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MIN", 30_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_CONSTRAINED_PAS_SPARSE_PC_MAX", 300_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_tokamak_pas_er_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak PAS+Er should start the host sparse-PC lane.

    Production-floor tokamak PAS+Er full-trajectory cases at
    ``25 x 1 x 8 x 100`` stall in the matrix-free PAS-ILU/Schur fallback ladder
    but are parity-clean with the non-differentiable sparse-PC GMRES route using
    a tiny diagonal shift and relaxed SuperLU pivoting. Keep this policy narrow:
    CPU/GPU only, no Phi1, pure PAS, axisymmetric, electric-field trajectory
    terms enabled, and active size in the measured window. The sparse
    factorization is still hosted, but the matrix-vector probes follow the
    active JAX backend; keep accelerators outside CPU/GPU opt-in until tested.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_ER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_tokamak_pas_noer_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
) -> bool:
    """Return whether tokamak PAS no-Er should start the host sparse-PC lane.

    The measured production-floor ``tokamak_*species_PASCollisions_noEr`` cases
    at ``25 x 1 x (4|8) x 100`` spend most of their default matrix-free memory
    or GPU wall time inside the Krylov solve even though a host sparse-PC solve
    reaches the same Fortran output. Keep this policy scoped to the validated
    non-differentiable axisymmetric no-Er PAS window so geometry-rich and Phi1
    systems continue to use their own policies.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 2:
        return False
    if op.fblock.fp is not None or op.fblock.pas is None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    if abs(float(er_abs)) > 0.0:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MIN", 5_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_PAS_NOER_SPARSE_PC_MAX", 750_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_tokamak_fp_er_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak full-FP + Er should start sparse-PC GMRES.

    Production-floor CPU/GPU probes at ``25 x 1 x 8 x 100`` show the
    matrix-free FP+Er routes can either stall before the strong fallback or pay
    for large generic XLA solves. The x-block sparse-PC route is parity-clean
    and materially faster in this measured axisymmetric window, so it is the
    default for both CPU and GPU non-differentiable output/CLI lanes.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    allowed_backends = {"cpu", "gpu", "cuda"}
    if backend_norm not in allowed_backends:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_ER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_tokamak_fp_noer_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    er_abs: float,
    use_dkes: bool,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether tokamak full-FP no-Er should start sparse-PC GMRES.

    The production-floor ``tokamak_1species_FPCollisions_noEr`` GPU row at
    ``25 x 1 x 8 x 100`` can exit the matrix-free XMG/strong-preconditioner
    ladder with a small-but-physics-visible residual.  Sparse-PC GMRES is slower
    than the memory-heavy theta-line route, but it is parity-clean against the
    Fortran direct solve and has substantially lower peak memory.  Keep this
    default GPU-only and constrained to the measured axisymmetric no-Er window.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC")
    if env is False:
        return False
    backend_norm = str(backend).strip().lower()
    allowed_backends = {"gpu", "cuda"} if env is not True else {"cpu", "gpu", "cuda"}
    if backend_norm not in allowed_backends:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 0:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MIN", 10_000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_FP_NOER_SPARSE_PC_MAX", 60_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_fp_3d_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    eparallel_abs: float = 0.0,
) -> bool:
    """Return whether 3D full-FP RHSMode=1 should start sparse-PC GMRES.

    Bounded HSX and geometryScheme11 FP probes show that the host sparse-PC
    branch can beat the dense FP shortcut on runtime and memory for some
    geometry-rich systems while preserving Fortran parity. Keep the promotion
    narrow and CPU-only, and do not take it for small or low-pitch-resolution
    systems by default: dense LU is more robust for large 3D full-FP smoke decks and
    avoids sparse-factorization failures before a solver trace is written.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC")
    if env is False:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    if int(getattr(op, "n_zeta", 1)) <= 1:
        return False
    if abs(float(eparallel_abs)) > 0.0:
        return False

    min_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN_NXI", 50)
    if int(getattr(op, "n_xi", max(0, int(min_nxi)))) < max(0, int(min_nxi)):
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MIN", 5001)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_SPARSE_PC_MAX", 20000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_fp_3d_xblock_sparse_pc_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    eparallel_abs: float = 0.0,
) -> bool:
    """Return whether 3D full-FP RHSMode=1 should use x-block sparse-PC GMRES.

    Bounded 3D full-FP CPU/GPU ladders and measured finite-beta two-species QA
    electron-root decks are too large for dense fallback but converge with
    host-assembled x-block sparse LU as a right preconditioner. Keep this as a
    bounded non-differentiable output/CLI route: small systems still use dense
    LU, GPU users get a checked non-dense route in the medium window, and
    production-floor systems remain deferred until larger non-dense ladders are
    measured.
    """

    env = _env_bool("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC")
    if env is False:
        return False
    if str(backend).strip().lower() not in {"cpu", "gpu", "cuda"}:
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) != 1:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    n_species = int(getattr(op, "n_species", 1))
    if n_species not in {1, 2}:
        return False
    if int(getattr(op, "n_zeta", 1)) <= 1:
        return False
    if bool(getattr(op, "point_at_x0", False)):
        return False
    if abs(float(eparallel_abs)) > 0.0:
        return False

    if n_species == 1:
        min_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN_NXI", 50)
        max_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MAX_NXI", 10_000_000)
        min_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MIN", 30_000)
        max_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MAX", 45_000)
        if env is True:
            min_size = 0
    else:
        multispecies_env = _env_bool("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES")
        if multispecies_env is False:
            return False
        min_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN_NXI", 12)
        max_nxi = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX_NXI", 16)
        min_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MIN", 30_000)
        max_size = _env_int("SFINCS_JAX_RHSMODE1_FP3D_XBLOCK_SPARSE_PC_MULTISPECIES_MAX", 100_000)
        if env is True or multispecies_env is True:
            min_size = 0

    n_xi = int(getattr(op, "n_xi", max(0, int(min_nxi))))
    if n_xi < max(0, int(min_nxi)):
        return False
    if int(max_nxi) > 0 and n_xi > int(max_nxi):
        return False
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_structured_full_csr_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    eparallel_abs: float = 0.0,
) -> bool:
    """Return whether ``auto`` should try the no-probe full-CSR host lane.

    This is intentionally a non-differentiable CLI/output opt-in policy. The
    assembled full-CSR path preserves the production matrix-vector product, but
    Zenodo QA/QH finite-beta probes showed that it can spend minutes before
    falling back when the current x-block Schur preconditioner is insufficient.
    Therefore the public ``auto`` default does not try this route unless
    ``SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO=1`` is set by an expert benchmark.
    The actual CSR/factor memory gates still live in the structured solve
    itself; this helper only decides whether an opted-in ``auto`` run is allowed
    to try that route before falling back. Keep a finite default size ceiling
    because production QH finite-beta audits showed that full-CSR assembly can
    exceed 50 GB before the memory gate is reached; experts can set
    ``SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MAX_SIZE=0`` for deliberate large
    benchmarks.
    """

    env = _env_bool("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO")
    if env is not True:
        return False
    if bool(use_implicit):
        return False
    if str(backend).strip().lower() not in {"cpu", "gpu", "cuda"}:
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(op.constraint_scheme) not in {1, 2}:
        return False
    if op.fblock.fp is None or op.fblock.pas is not None:
        return False
    n_species = int(getattr(op, "n_species", 1))
    if n_species > 2:
        return False
    if int(getattr(op, "n_zeta", 1)) <= 1:
        return False
    if bool(getattr(op, "point_at_x0", False)):
        return False
    if abs(float(eparallel_abs)) > 0.0:
        return False

    min_nxi = _env_int("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", 12)
    if int(getattr(op, "n_xi", max(0, int(min_nxi)))) < max(0, int(min_nxi)):
        return False

    min_size = _env_int("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", 10_000)
    max_size = _env_int("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MAX_SIZE", 100_000)
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    return int(active_size) >= max(0, int(min_size))

def rhs1_tokamak_er_dense_auto_allowed(
    *,
    op: Any,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
    backend: str,
    use_dkes: bool,
    er_abs: float,
    include_xdot: bool,
    include_electric_field_xi: bool,
) -> bool:
    """Return whether bounded tokamak electric-field RHSMode=1 may use dense LU.

    Production-resolution tokamak Er probes show that the matrix-free
    Krylov/strong/sparse-rescue ladder can spend O(100 s) on systems just above
    the generic dense cutoff, while dense LU solves the same algebraic problem in
    a few seconds with Fortran-clean diagnostics. Keep this CPU-only and
    size-bounded because dense LU has a larger transient RSS footprint.
    """
    env = _env_bool("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE")
    if env is False:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if bool(use_implicit):
        return False
    if str(solve_method_kind).strip().lower().replace("-", "_") not in {"auto", "default", "incremental"}:
        return False
    if int(op.rhs_mode) != 1 or bool(op.include_phi1):
        return False
    if int(getattr(op, "n_zeta", 1)) != 1:
        return False
    if op.fblock.fp is None and op.fblock.pas is None:
        return False
    has_er_drive = abs(float(er_abs)) > 0.0 and (
        bool(use_dkes) or bool(include_xdot) or bool(include_electric_field_xi)
    )
    if not has_er_drive:
        return False

    min_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MIN", 5000)
    max_size = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX", 6500)
    max_dense_bytes = _env_int("SFINCS_JAX_RHSMODE1_TOKAMAK_ER_DENSE_MAX_BYTES", 350_000_000)
    if env is True:
        min_size = 0
    if int(max_size) > 0 and int(active_size) > int(max_size):
        return False
    dense_bytes = int(active_size) * int(active_size) * 8
    if int(max_dense_bytes) > 0 and dense_bytes > int(max_dense_bytes):
        return False
    return int(active_size) >= max(0, int(min_size))

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

# Consolidated large explicit-FP host-rescue policy section

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

    skip_primary_max = _env_int("SFINCS_JAX_RHSMODE1_SPARSE_LARGE_CPU_SKIP_PRIMARY_MAX", 30000)
    rescue_min_default = max(int(sparse_max_size) + 1, 12000, int(skip_primary_max) + 1)
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

# Consolidated automatic preconditioner-routing policy section

PAS_AUTO_STRONG_BASE_KINDS = frozenset(
    {
        "schur",
        "xblock_tz",
        "xblock_tz_lmax",
        "sxblock_tz",
        "species_block",
        "theta_zeta",
        "pas_lite",
        "pas_hybrid",
        "pas_schur",
        "pas_tz",
        "pas_tzfft",
        "pas_tokamak_theta",
    }
)

PAS_WEAK_AUTO_OVERRIDE_KINDS = frozenset(
    {
        None,
        "collision",
        "point",
        "xmg",
        "theta_line",
        "zeta_line",
        "theta_zeta",
        "xblock_tz",
        "xblock_tz_lmax",
        "theta_line_xdiag",
    }
)

FP_FORCE_XMG_WEAK_KINDS = frozenset(
    {
        None,
        "collision",
        "point",
        "theta_line",
        "zeta_line",
        "theta_schwarz",
        "zeta_schwarz",
    }
)

_RHS1_PRECONDITIONER_KIND_ALIASES = {
    "0": None,
    "false": None,
    "no": None,
    "off": None,
    "theta": "theta_line",
    "theta_line": "theta_line",
    "line_theta": "theta_line",
    "theta_dd": "theta_dd",
    "theta_block": "theta_dd",
    "dd_theta": "theta_dd",
    "dd_t": "theta_dd",
    "theta_schwarz": "theta_schwarz",
    "schwarz_theta": "theta_schwarz",
    "ras_theta": "theta_schwarz",
    "theta_ras": "theta_schwarz",
    "theta_line_xdiag": "theta_line_xdiag",
    "theta_xdiag": "theta_line_xdiag",
    "theta_line_diagx": "theta_line_xdiag",
    "xdiag": "point_xdiag",
    "point_xdiag": "point_xdiag",
    "block_xdiag": "point_xdiag",
    "species": "species_block",
    "species_block": "species_block",
    "speciesblock": "species_block",
    "sxblock": "sxblock",
    "species_xblock": "sxblock",
    "species_x": "sxblock",
    "sxblock_tz": "sxblock_tz",
    "sxblock_theta_zeta": "sxblock_tz",
    "species_xblock_tz": "sxblock_tz",
    "sx_tz": "sxblock_tz",
    "xblock_tz_lmax": "xblock_tz_lmax",
    "xblock_tz_trunc": "xblock_tz_lmax",
    "xblock_tz_cut": "xblock_tz_lmax",
    "xblock_tz": "xblock_tz",
    "xblock": "xblock_tz",
    "x_tz": "xblock_tz",
    "xtz": "xblock_tz",
    "xblock_theta_zeta": "xblock_tz",
    "xmg": "xmg",
    "multigrid": "xmg",
    "x_coarse": "xmg",
    "coarse_x": "xmg",
    "pas_lite": "pas_lite",
    "pas_light": "pas_lite",
    "pas_xmg": "pas_lite",
    "pas_xmg_lite": "pas_lite",
    "pas_hybrid": "pas_hybrid",
    "pas_xline_xcoarse": "pas_hybrid",
    "pas_line_xcoarse": "pas_hybrid",
    "pas_xcoarse_line": "pas_hybrid",
    "pas_schur": "pas_schur",
    "pas_block_schur": "pas_schur",
    "pas_xmg_l": "pas_schur",
    "pas_tz": "pas_tz",
    "pas_3d": "pas_tz",
    "pas_tz_l": "pas_tz",
    "pas_tzfft": "pas_tzfft",
    "pas_fft": "pas_tzfft",
    "pas_stream_fft": "pas_tzfft",
    "pas_streaming_fft": "pas_tzfft",
    "pas_ilu": "pas_ilu",
    "pas_block_ilu": "pas_ilu",
    "pas_xblock_ilu": "pas_ilu",
    "block_ilu": "pas_ilu",
    "theta_zeta": "theta_zeta",
    "theta_zeta_line": "theta_zeta",
    "tz": "theta_zeta",
    "tz_line": "theta_zeta",
    "zeta": "zeta_line",
    "zeta_line": "zeta_line",
    "line_zeta": "zeta_line",
    "zeta_dd": "zeta_dd",
    "zeta_block": "zeta_dd",
    "dd_zeta": "zeta_dd",
    "dd_z": "zeta_dd",
    "zeta_schwarz": "zeta_schwarz",
    "schwarz_zeta": "zeta_schwarz",
    "ras_zeta": "zeta_schwarz",
    "zeta_ras": "zeta_schwarz",
    "adi": "adi",
    "adi_line": "adi",
    "line_adi": "adi",
    "zeta_theta": "adi",
    "1": "point",
    "true": "point",
    "yes": "point",
    "on": "point",
    "point": "point",
    "point_block": "point",
    "schur": "schur",
    "schur_complement": "schur",
    "constraint_schur": "schur",
    "collision": "collision",
    "diag": "collision",
    "collision_diag": "collision",
    "structured_fblock": "structured_fblock_jacobi",
    "structured_fblock_jacobi": "structured_fblock_jacobi",
    "fblock_jacobi": "structured_fblock_jacobi",
    "block_coo_jacobi": "structured_fblock_jacobi",
    "structured_fblock_angular": "structured_fblock_angular_jacobi",
    "structured_fblock_angular_jacobi": "structured_fblock_angular_jacobi",
    "fblock_angular_jacobi": "structured_fblock_angular_jacobi",
    "block_coo_angular_jacobi": "structured_fblock_angular_jacobi",
    "structured_fblock_xi_angular": "structured_fblock_xi_angular_jacobi",
    "structured_fblock_xi_angular_jacobi": "structured_fblock_xi_angular_jacobi",
    "fblock_xi_angular_jacobi": "structured_fblock_xi_angular_jacobi",
    "block_coo_xi_angular_jacobi": "structured_fblock_xi_angular_jacobi",
    "structured_fblock_fp_radial": "structured_fblock_fp_radial_jacobi",
    "structured_fblock_fp_radial_jacobi": "structured_fblock_fp_radial_jacobi",
    "fblock_fp_radial_jacobi": "structured_fblock_fp_radial_jacobi",
    "fblock_species_x_jacobi": "structured_fblock_fp_radial_jacobi",
    "block_coo_fp_radial_jacobi": "structured_fblock_fp_radial_jacobi",
    "structured_fblock_fp_lowmode_schur": "structured_fblock_fp_lowmode_schur",
    "fblock_fp_lowmode_schur": "structured_fblock_fp_lowmode_schur",
    "fblock_fp_galerkin": "structured_fblock_fp_lowmode_schur",
    "block_coo_fp_lowmode_schur": "structured_fblock_fp_lowmode_schur",
    "structured_fblock_fp_moment_schur": "structured_fblock_fp_moment_schur",
    "fblock_fp_moment_schur": "structured_fblock_fp_moment_schur",
    "fblock_fp_moment_galerkin": "structured_fblock_fp_moment_schur",
    "block_coo_fp_moment_schur": "structured_fblock_fp_moment_schur",
    "structured_fblock_fp_coupled_moment_schur": "structured_fblock_fp_coupled_moment_schur",
    "fblock_fp_coupled_moment_schur": "structured_fblock_fp_coupled_moment_schur",
    "fblock_fp_coupled_galerkin": "structured_fblock_fp_coupled_moment_schur",
    "block_coo_fp_coupled_moment_schur": "structured_fblock_fp_coupled_moment_schur",
    "structured_fblock_fp_tail_coupled_schur": "structured_fblock_fp_tail_coupled_schur",
    "fblock_fp_tail_coupled_schur": "structured_fblock_fp_tail_coupled_schur",
    "fblock_fp_tail_coupled_minres": "structured_fblock_fp_tail_coupled_schur",
    "fblock_fp_tail_minres": "structured_fblock_fp_tail_coupled_schur",
    "block_coo_fp_tail_coupled_schur": "structured_fblock_fp_tail_coupled_schur",
    "block_coo_fp_tail_coupled_minres": "structured_fblock_fp_tail_coupled_schur",
}

@dataclass(frozen=True)
class RHS1DefaultPreconditionerSelectionContext:
    """Driver scope for automatic RHSMode-1 preconditioner selection."""

    values: Mapping[str, Any]

@dataclass(frozen=True)
class RHS1PreconditionerRouteSetupContext:
    """Driver scope for RHSMode-1 preconditioner routing setup."""

    values: Mapping[str, Any]

def resolve_rhs1_default_preconditioner_selection(
    context: RHS1DefaultPreconditionerSelectionContext,
) -> dict[str, Any]:
    """Resolve the default RHSMode-1 preconditioner policy branch."""

    _canonical_rhs1_preconditioner_kind = context.values['_canonical_rhs1_preconditioner_kind']
    _matvec_shard_axis = context.values['_matvec_shard_axis']
    _pas_tz_preconditioner_applicable = context.values['_pas_tz_preconditioner_applicable']
    _rhs1_fp_dkes_default_kind = context.values['_rhs1_fp_dkes_default_kind']
    _rhs1_pas_auto_large_base_kind = context.values['_rhs1_pas_auto_large_base_kind']
    _rhs1_pas_dkes_pas_tz_preferred = context.values['_rhs1_pas_dkes_pas_tz_preferred']
    _rhs1_pas_dkes_xblock_allowed = context.values['_rhs1_pas_dkes_xblock_allowed']
    _rhs1_pas_small_near_zero_er_kind = context.values['_rhs1_pas_small_near_zero_er_kind']
    _rhs1_pas_tokamak_gpu_theta_allowed = context.values['_rhs1_pas_tokamak_gpu_theta_allowed']
    _rhs1_pas_tokamak_gpu_xblock_preferred = context.values['_rhs1_pas_tokamak_gpu_xblock_preferred']
    active_size = context.values['active_size']
    emit = context.values['emit']
    full_precond_requested = context.values['full_precond_requested']
    geom_scheme = context.values['geom_scheme']
    jax = context.values['jax']
    nml = context.values['nml']
    np = context.values['np']
    op = context.values['op']
    os = context.values['os']
    pre_theta = context.values['pre_theta']
    pre_zeta = context.values['pre_zeta']
    er_abs = context.values['er_abs']
    rhs1_gpu_tokamak_pas_tight_gmres = context.values[
        'rhs1_gpu_tokamak_pas_tight_gmres'
    ]
    rhs1_precond_env = context.values['rhs1_precond_env']
    rhs1_xblock_tz_lmax = context.values['rhs1_xblock_tz_lmax']
    schur_er_min = context.values['schur_er_min']
    use_dkes = context.values['use_dkes']

    if rhs1_precond_env:
        rhs1_precond_kind = _canonical_rhs1_preconditioner_kind(rhs1_precond_env)
    else:
        # Default to v3-like preconditioner options: when preconditioner_theta/zeta are 0,
        # use point-block Jacobi. Enable line preconditioning only when explicitly requested.
        if int(op.rhs_mode) == 1 and (not bool(op.include_phi1)):
            if pre_theta == 0 and pre_zeta == 0:
                tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "").strip()
                try:
                    tz_max = int(tz_max_env) if tz_max_env else 128
                except ValueError:
                    tz_max = 128
                xblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "").strip()
                default_xblock_tz_max = 1200
                if op.fblock.pas is not None and geom_scheme == 1:
                    default_xblock_tz_max = 6000
                elif op.fblock.pas is not None:
                    default_xblock_tz_max = 2000
                try:
                    xblock_tz_max = int(xblock_tz_max_env) if xblock_tz_max_env else default_xblock_tz_max
                except ValueError:
                    xblock_tz_max = default_xblock_tz_max
                xblock_tz_lmax_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "").strip()
                try:
                    xblock_tz_lmax_override = int(xblock_tz_lmax_env) if xblock_tz_lmax_env else 0
                except ValueError:
                    xblock_tz_lmax_override = 0
                species_block_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SPECIES_BLOCK_MAX", "").strip()
                try:
                    species_block_max = int(species_block_max_env) if species_block_max_env else 1600
                except ValueError:
                    species_block_max = 1600
                nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
                max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                lmax_auto = 0
                if int(op.n_theta) > 0 and int(op.n_zeta) > 0:
                    lmax_auto = int(xblock_tz_max // (int(op.n_theta) * int(op.n_zeta)))
                lmax_auto = max(0, min(max_l, lmax_auto))
                local_per_species = int(np.sum(nxi_for_x))
                dke_size = int(local_per_species * int(op.n_theta) * int(op.n_zeta))
                line_size = int(local_per_species * int(op.n_theta))
                sxblock_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_MAX", "").strip()
                try:
                    sxblock_max = int(sxblock_max_env) if sxblock_max_env else 64
                except ValueError:
                    sxblock_max = 64
                sxblock_tz_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_MAX", "").strip()
                try:
                    sxblock_tz_max = int(sxblock_tz_max_env) if sxblock_tz_max_env else 0
                except ValueError:
                    sxblock_tz_max = 0
                sxblock_tz_active_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_SXBLOCK_TZ_ACTIVE_MAX", "").strip()
                try:
                    sxblock_tz_active_max = int(sxblock_tz_active_max_env) if sxblock_tz_active_max_env else 20000
                except ValueError:
                    sxblock_tz_active_max = 20000
                if sxblock_tz_max == 0 and op.fblock.fp is not None and (
                    int(op.n_theta) > 1 or int(op.n_zeta) > 1
                ):
                    # Allow a modest FP sxblock_tz preconditioner in multi-angle FP cases
                    # to avoid RHSMode=1 stagnation without large dense fallbacks.
                    sxblock_tz_max = 2000
                sxblock_size = int(int(op.n_species) * local_per_species)
                sxblock_tz_size = int(int(op.n_species) * int(op.n_x) * int(op.n_theta) * int(op.n_zeta))
                schur_auto = False
                if (
                    int(op.constraint_scheme) == 2
                    and int(op.extra_size) > 0
                    and op.fblock.pas is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                ):
                    schur_auto_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_AUTO_MIN", "").strip()
                    try:
                        schur_auto_min = int(schur_auto_min_env) if schur_auto_min_env else 2500
                    except ValueError:
                        schur_auto_min = 2500
                    schur_auto = int(op.total_size) >= schur_auto_min
                phys_params = nml.group("physicsParameters")
                er_val = phys_params.get("ER", phys_params.get("Er", phys_params.get("er", None)))
                er_abs = 0.0
                if er_val is not None:
                    try:
                        er_abs = float(er_val)
                    except (TypeError, ValueError):
                        er_abs = 0.0
                er_abs = abs(er_abs)
                epar_val = phys_params.get("EPARALLELHAT", phys_params.get("EParallelHat", None))
                try:
                    epar_abs = abs(float(epar_val)) if epar_val is not None else 0.0
                except (TypeError, ValueError):
                    epar_abs = 0.0
                if epar_abs > 0.0 and sxblock_tz_max == 0:
                    sxblock_tz_max = 2000
                schur_er_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_ER_ABS_MIN", "").strip()
                try:
                    schur_er_min = float(schur_er_env) if schur_er_env else 1.0e-12
                except ValueError:
                    schur_er_min = 1.0e-12
                pas_dkes_gpu_xblock_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_DKES_GPU_XBLOCK_TZ_MAX", "").strip()
                try:
                    pas_dkes_gpu_xblock_max = int(pas_dkes_gpu_xblock_env) if pas_dkes_gpu_xblock_env else 2500
                except ValueError:
                    pas_dkes_gpu_xblock_max = 2500
                pas_xdiag_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_XDIAG_MIN", "").strip()
                try:
                    pas_xdiag_min = int(pas_xdiag_env) if pas_xdiag_env else 1000000000
                except ValueError:
                    pas_xdiag_min = 1000000000
                pas_xmg_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", "").strip()
                try:
                    pas_xmg_min = int(pas_xmg_env) if pas_xmg_env else 80000
                except ValueError:
                    pas_xmg_min = 80000
                fp_xmg_env = os.environ.get("SFINCS_JAX_RHSMODE1_FP_XMG_MAX", "").strip()
                try:
                    # Keep xmg as the default for larger FP systems as long as we are still
                    # in the matrix-free Krylov regime; this avoids expensive Schwarz builds
                    # that can dominate runtime in high-resolution single-RHS runs.
                    fp_xmg_max = int(fp_xmg_env) if fp_xmg_env else 200000
                except ValueError:
                    fp_xmg_max = 200000
                schur_tokamak_env = os.environ.get("SFINCS_JAX_RHSMODE1_SCHUR_TOKAMAK", "").strip().lower()
                schur_tokamak = schur_tokamak_env in {"1", "true", "yes", "on"}
                tokamak_like = int(op.n_zeta) == 1 or geom_scheme == 1
                if full_precond_requested and int(op.constraint_scheme) == 2 and int(op.extra_size) > 0:
                    if tokamak_like and schur_tokamak and er_abs <= schur_er_min:
                        rhs1_precond_kind = "schur"
                    elif tokamak_like and (not schur_tokamak) and er_abs <= schur_er_min:
                        if op.fblock.pas is not None:
                            # For tiny tokamak PAS systems, prefer the xblock_tz preconditioner
                            # (matches legacy fixtures). For larger systems, keep the lighter
                            # PAS hybrid to avoid expensive dense angular blocks.
                            xblock_small_env = os.environ.get("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", "").strip()
                            try:
                                xblock_small_max = int(xblock_small_env) if xblock_small_env else 4000
                            except ValueError:
                                xblock_small_max = 4000
                            if (
                                int(op.total_size) <= max(1, int(xblock_small_max))
                                and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= max(1, int(xblock_tz_max))
                            ):
                                rhs1_precond_kind = "xblock_tz"
                            else:
                                # Tokamak-like PAS systems benefit from the PAS hybrid (line + x-coarse)
                                # preconditioner; avoid the expensive global Schur/xblock path here.
                                rhs1_precond_kind = "pas_hybrid"
                        else:
                            pas_schur_small_env = os.environ.get("SFINCS_JAX_RHSMODE1_PAS_SCHUR_SMALL_MAX", "").strip()
                            try:
                                pas_schur_small_max = int(pas_schur_small_env) if pas_schur_small_env else 20000
                            except ValueError:
                                pas_schur_small_max = 20000
                            if int(op.total_size) <= max(1, int(pas_schur_small_max)):
                                rhs1_precond_kind = "schur"
                            elif (
                                int(op.n_theta) > 1
                                and xblock_tz_max > 0
                                and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                            ):
                                rhs1_precond_kind = "xblock_tz"
                            else:
                                rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                    else:
                        if (
                            op.fblock.pas is not None
                            and er_abs <= schur_er_min
                            and (not schur_tokamak)
                            and int(op.total_size) < pas_xmg_min
                        ):
                            # For constrained PAS near-zero-Er systems below the Schur regime,
                            # prefer a lightweight PAS preconditioner when angular blocks are
                            # modest; otherwise fall back to x-coarsening to avoid expensive
                            # global Schur setup while retaining good Krylov convergence.
                            rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                                pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                                tz_size=int(op.n_theta) * int(op.n_zeta),
                                active_size=int(active_size),
                            )
                        elif (
                            op.fblock.fp is not None
                            and er_abs <= schur_er_min
                            and int(op.total_size) < fp_xmg_max
                        ):
                            rhs1_precond_kind = "xmg"
                        elif _rhs1_pas_tokamak_gpu_xblock_preferred(
                            has_pas=op.fblock.pas is not None,
                            has_fp=op.fblock.fp is not None,
                            backend=jax.default_backend(),
                            tokamak_like=tokamak_like,
                            active_size=int(active_size),
                            er_abs=float(er_abs),
                            schur_er_min=float(schur_er_min),
                            has_magdrift=(
                                op.fblock.magdrift_theta is not None
                                or op.fblock.magdrift_zeta is not None
                                or op.fblock.magdrift_xidot is not None
                            ),
                            has_collisionless=op.fblock.collisionless is not None,
                            n_theta=int(op.n_theta),
                            n_zeta=int(op.n_zeta),
                            max_l=int(max_l),
                            xblock_tz_limit=max(int(xblock_tz_max), int(pas_dkes_gpu_xblock_max)),
                        ):
                            rhs1_precond_kind = "xblock_tz"
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                                    "auto -> xblock_tz preconditioner",
                                )
                        elif _rhs1_pas_tokamak_gpu_theta_allowed(
                            has_pas=op.fblock.pas is not None,
                            has_fp=op.fblock.fp is not None,
                            backend=jax.default_backend(),
                            tokamak_like=tokamak_like,
                            active_size=int(active_size),
                            er_abs=float(er_abs),
                            schur_er_min=float(schur_er_min),
                            has_magdrift=(
                                op.fblock.magdrift_theta is not None
                                or op.fblock.magdrift_zeta is not None
                                or op.fblock.magdrift_xidot is not None
                            ),
                            has_collisionless=op.fblock.collisionless is not None,
                        ):
                            rhs1_precond_kind = None
                            rhs1_gpu_tokamak_pas_tight_gmres = True
                            if emit is not None:
                                emit(
                                    1,
                                    "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                                    "auto -> tight unpreconditioned GMRES",
                                )
                        elif op.fblock.pas is not None and int(op.total_size) >= pas_xmg_min:
                            # Large constrained PAS+Er systems need stronger x/L coupling than
                            # collision/point/xmg alone, but a global Schur setup can dominate
                            # wall time. Keep the auto path in the PAS-native family here so
                            # the later tokamak/3D refinements can promote to pas_tz/pas_ilu.
                            rhs1_precond_kind = _rhs1_pas_auto_large_base_kind(active_size=int(active_size))
                        elif op.fblock.pas is not None and int(op.total_size) >= pas_xdiag_min:
                            lmax_use = xblock_tz_lmax_override if xblock_tz_lmax_override > 0 else lmax_auto
                            if lmax_use >= 1:
                                rhs1_precond_kind = "xblock_tz_lmax"
                                rhs1_xblock_tz_lmax = int(lmax_use)
                            else:
                                rhs1_precond_kind = "point_xdiag"
                        else:
                            if op.fblock.pas is not None:
                                rhs1_precond_kind = _rhs1_pas_auto_large_base_kind(active_size=int(active_size))
                            else:
                                rhs1_precond_kind = "schur"
                elif full_precond_requested and (int(op.n_theta) > 1 or int(op.n_zeta) > 1):
                    if (
                        op.fblock.pas is not None
                        and er_abs <= schur_er_min
                        and (not schur_tokamak)
                        and int(op.total_size) < pas_xmg_min
                    ):
                        rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                            pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                            tz_size=int(op.n_theta) * int(op.n_zeta),
                            active_size=int(active_size),
                        )
                    elif (
                        op.fblock.fp is not None
                        and er_abs <= schur_er_min
                        and int(op.total_size) < fp_xmg_max
                    ):
                        rhs1_precond_kind = "xmg"
                    else:
                        rhs1_precond_kind = "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
                elif schur_auto:
                    # For sharded multi-device PAS near-zero-Er runs, Schur can become
                    # communication-dominated as device count increases. Prefer x-coarsening
                    # here to keep Krylov/preconditioner cost closer to shard-local.
                    shard_axis_auto = _matvec_shard_axis(op)
                    if (
                        op.fblock.pas is not None
                        and (not bool(op.include_phi1))
                        and float(er_abs) <= float(schur_er_min)
                        and shard_axis_auto in {"theta", "zeta"}
                        and jax.device_count() > 1
                        and int(op.total_size) <= max(1, int(pas_xmg_min))
                    ):
                        rhs1_precond_kind = "xmg"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: sharded PAS near-zero-Er "
                                "schur_auto -> xmg preconditioner",
                            )
                    elif _rhs1_pas_dkes_pas_tz_preferred(
                        has_pas=op.fblock.pas is not None,
                        use_dkes=bool(use_dkes),
                        backend=jax.default_backend(),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        active_size=int(active_size),
                    ):
                        rhs1_precond_kind = "pas_tz"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: PAS DKES "
                                "schur_auto -> pas_tz preconditioner",
                            )
                    elif _rhs1_pas_dkes_xblock_allowed(
                        has_pas=op.fblock.pas is not None,
                        use_dkes=bool(use_dkes),
                        backend=jax.default_backend(),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        xblock_tz_limit=max(int(xblock_tz_max), int(pas_dkes_gpu_xblock_max)),
                    ):
                        rhs1_precond_kind = "xblock_tz"
                        if emit is not None:
                            emit(
                                1,
                                "solve_v3_full_system_linear_gmres: GPU PAS DKES "
                                "schur_auto -> xblock_tz preconditioner",
                            )
                    else:
                        rhs1_precond_kind = "schur"
                elif op.fblock.fp is not None and use_dkes:
                    # DKES-trajectory FP cases can stagnate with collision-only
                    # preconditioners. Prefer a lightweight xmg/sxblock_tz path for
                    # small/medium systems, and fall back to collision for larger sizes
                    # to avoid expensive block builds.
                    max_l = int(np.max(nxi_for_x)) if nxi_for_x.size else 0
                    rhs1_precond_kind = _rhs1_fp_dkes_default_kind(
                        active_size=int(active_size),
                        n_theta=int(op.n_theta),
                        n_zeta=int(op.n_zeta),
                        max_l=int(max_l),
                        xblock_tz_limit=int(xblock_tz_max),
                    )
                    if rhs1_precond_kind == "xblock_tz" and not rhs1_precond_env:
                        rhs1_precond_env = "xblock_tz"
                elif (
                    op.fblock.fp is not None
                    and er_abs <= schur_er_min
                    and int(active_size) < fp_xmg_max
                ):
                    # For moderate-size FP systems at near-zero Er, x-coarsened preconditioning
                    # is typically much cheaper than global (S,X,theta,zeta) blocks and
                    # preserves parity for RHSMode=1.
                    rhs1_precond_kind = "xmg"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and sxblock_tz_max > 0
                    and int(op.total_size) <= max(1, int(sxblock_tz_active_max))
                    and sxblock_tz_size <= sxblock_tz_max
                ):
                    rhs1_precond_kind = "sxblock_tz"
                elif (
                    op.fblock.fp is not None
                    and (int(op.n_theta) > 1 or int(op.n_zeta) > 1)
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                elif op.fblock.fp is not None and sxblock_max > 0 and sxblock_size <= sxblock_max:
                    rhs1_precond_kind = "sxblock"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and species_block_max > 0
                    and dke_size <= species_block_max
                ):
                    rhs1_precond_kind = "species_block"
                elif _rhs1_pas_dkes_pas_tz_preferred(
                    has_pas=op.fblock.pas is not None,
                    use_dkes=bool(use_dkes),
                    backend=jax.default_backend(),
                    n_theta=int(op.n_theta),
                    n_zeta=int(op.n_zeta),
                    max_l=int(max_l),
                    active_size=int(active_size),
                ):
                    rhs1_precond_kind = "pas_tz"
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: PAS DKES "
                            "auto -> pas_tz preconditioner",
                        )
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and xblock_tz_max > 0
                    and int(max_l) * int(op.n_theta) * int(op.n_zeta) <= xblock_tz_max
                ):
                    rhs1_precond_kind = "xblock_tz"
                elif (
                    op.fblock.pas is not None
                    and int(op.n_theta) > 1
                    and int(op.n_zeta) > 1
                    and int(op.n_theta) * int(op.n_zeta) <= tz_max
                ):
                    rhs1_precond_kind = "theta_zeta"
                elif (
                    op.fblock.pas is not None
                    and int(active_size) >= pas_xmg_min
                ):
                    # Large PAS systems tend to be x/L-coupling dominated. Prefer the
                    # x-coarsened PAS preconditioner over weak collision/point
                    # preconditioners that often trigger expensive fallback branches.
                    rhs1_precond_kind = "xmg"
                else:
                    if (
                        op.fblock.pas is not None
                        and er_abs <= schur_er_min
                        and int(active_size) < pas_xmg_min
                    ):
                        rhs1_precond_kind = _rhs1_pas_small_near_zero_er_kind(
                            pas_tz_applicable=_pas_tz_preconditioner_applicable(op),
                            tz_size=int(op.n_theta) * int(op.n_zeta),
                            active_size=int(active_size),
                        )
                    else:
                        collision_precond_min_env = os.environ.get("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_MIN", "").strip()
                        try:
                            collision_precond_min = int(collision_precond_min_env) if collision_precond_min_env else 600
                        except ValueError:
                            collision_precond_min = 600
                        use_collision_precond = (
                            (op.fblock.fp is not None or op.fblock.pas is not None)
                            and int(op.total_size) >= collision_precond_min
                        )
                        if (
                            use_collision_precond
                            and full_precond_requested
                            and op.fblock.pas is not None
                            and int(op.total_size) >= pas_xdiag_min
                        ):
                            rhs1_precond_kind = "point_xdiag"
                        else:
                            # Last-resort auto mode: collision-only preconditioning is cheap but
                            # can be too weak/unstable for FP systems at nonzero Er. Prefer xmg
                            # when the (x,theta,zeta) grid is still moderate to improve robustness.
                            if (
                                op.fblock.fp is not None
                                and op.fblock.pas is None
                                and int(active_size) < fp_xmg_max
                            ):
                                rhs1_precond_kind = "xmg"
                            else:
                                rhs1_precond_kind = "collision" if use_collision_precond else "point"
                theta_line_max_env = os.environ.get("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", "").strip()
                try:
                    theta_line_max = int(theta_line_max_env) if theta_line_max_env else 0
                except ValueError:
                    theta_line_max = 0
                if rhs1_precond_kind == "theta_line" and theta_line_max > 0 and line_size > theta_line_max:
                    rhs1_precond_kind = "theta_line_xdiag"
            elif pre_theta > 0 and pre_zeta > 0:
                rhs1_precond_kind = "adi"
            elif pre_theta > 0:
                rhs1_precond_kind = "theta_line"
            elif pre_zeta > 0:
                rhs1_precond_kind = "zeta_line"
            else:
                rhs1_precond_kind = "point"
        else:
            rhs1_precond_kind = None
    result: dict[str, Any] = {}
    if 'er_abs' in locals():
        result['er_abs'] = er_abs
    if 'lmax_use' in locals():
        result['lmax_use'] = lmax_use
    if 'max_l' in locals():
        result['max_l'] = max_l
    if 'nxi_for_x' in locals():
        result['nxi_for_x'] = nxi_for_x
    if 'rhs1_gpu_tokamak_pas_tight_gmres' in locals():
        result['rhs1_gpu_tokamak_pas_tight_gmres'] = rhs1_gpu_tokamak_pas_tight_gmres
    if 'rhs1_precond_env' in locals():
        result['rhs1_precond_env'] = rhs1_precond_env
    if 'rhs1_precond_kind' in locals():
        result['rhs1_precond_kind'] = rhs1_precond_kind
    if 'rhs1_xblock_tz_lmax' in locals():
        result['rhs1_xblock_tz_lmax'] = rhs1_xblock_tz_lmax
    if 'schur_er_min' in locals():
        result['schur_er_min'] = schur_er_min
    if 'tokamak_like' in locals():
        result['tokamak_like'] = tokamak_like
    if 'use_collision_precond' in locals():
        result['use_collision_precond'] = use_collision_precond
    if 'xblock_tz_max' in locals():
        result['xblock_tz_max'] = xblock_tz_max
    return result

def resolve_rhs1_preconditioner_route_setup(
    context: RHS1PreconditionerRouteSetupContext,
) -> dict[str, Any]:
    """Resolve RHSMode=1 preconditioner routing before Krylov branch setup.

    This function owns the policy-only part of RHSMode=1 route selection:
    environment parsing, automatic FP/PAS preconditioner selection, sharded-run
    downgrades, GPU PAS tolerance tightening, and the final policy-hint update.
    It intentionally receives callbacks through ``context`` to avoid importing
    solver builders or operator modules into the policy owner.
    """

    values = context.values
    jax_mod = values.get("jax", jax)
    np_mod = values.get("np", np)
    os_mod = values.get("os", os)
    op = values["op"]
    nml = values["nml"]
    emit = values.get("emit")
    precond_opts = values["precond_opts"]
    use_dkes = bool(values["use_dkes"])
    active_size = int(values["active_size"])
    geom_scheme = int(values["geom_scheme"])
    max_l = int(values["max_l"])
    nxi_for_x = np_mod.asarray(values["nxi_for_x"], dtype=np.int32)
    tol = float(values["tol"])
    restart = int(values["restart"])
    maxiter = values["maxiter"]
    solve_method = str(values["solve_method"])
    use_pas_projection = bool(values["use_pas_projection"])

    matvec_shard_axis = values["_matvec_shard_axis"]
    pas_tz_preconditioner_applicable = values["_pas_tz_preconditioner_applicable"]
    pas_tokamak_theta_preconditioner_applicable = values[
        "_pas_tokamak_theta_preconditioner_applicable"
    ]
    estimate_rhs1_pas_tz_build_bytes = values["_estimate_rhs1_pas_tz_build_bytes"]
    rhs1_pas_tz_max_bytes = values["_rhs1_pas_tz_max_bytes"]
    set_precond_policy_hints = values["_set_precond_policy_hints"]
    resolve_domain_decomposition_setup = values[
        "resolve_rhs1_domain_decomposition_setup"
    ]

    rhs1_precond_env = (
        os_mod.environ.get("SFINCS_JAX_RHSMODE1_PRECONDITIONER", "")
        .strip()
        .lower()
    )
    rhs1_precond_env_user = rhs1_precond_env
    rhs1_bicgstab_env = (
        os_mod.environ.get("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND", "")
        .strip()
        .lower()
    )
    rhs1_bicgstab_env_user = rhs1_bicgstab_env
    rhs1_precond_env = rhs1_fp_dkes_env_preconditioner_kind(
        rhs1_precond_env=rhs1_precond_env,
        rhs_mode=int(op.rhs_mode),
        include_phi1=bool(op.include_phi1),
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        total_size=int(op.total_size),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        max_l=int(np_mod.max(nxi_for_x)) if nxi_for_x.size else 0,
    )
    try:
        pre_theta = int(precond_opts.get("PRECONDITIONER_THETA", 0) or 0)
    except (TypeError, ValueError):
        pre_theta = 0
    try:
        pre_zeta = int(precond_opts.get("PRECONDITIONER_ZETA", 0) or 0)
    except (TypeError, ValueError):
        pre_zeta = 0
    rhs1_precond_kind: str | None = None
    rhs1_xblock_tz_lmax: int | None = None
    rhs1_gpu_tokamak_pas_tight_gmres = False
    rhs1_dd_setup = resolve_domain_decomposition_setup(
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        sum_nxi=int(np_mod.sum(nxi_for_x)) if nxi_for_x.size else 1,
        distributed_env=os_mod.environ.get("SFINCS_JAX_GMRES_DISTRIBUTED", ""),
        device_count=int(jax_mod.device_count()),
        auto_axis=matvec_shard_axis(op),
        theta_block_env=os_mod.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_T", ""),
        zeta_block_env=os_mod.environ.get("SFINCS_JAX_RHSMODE1_DD_BLOCK_Z", ""),
        theta_overlap_env=os_mod.environ.get(
            "SFINCS_JAX_RHSMODE1_DD_OVERLAP_T", ""
        ),
        zeta_overlap_env=os_mod.environ.get("SFINCS_JAX_RHSMODE1_DD_OVERLAP_Z", ""),
        overlap_env=os_mod.environ.get("SFINCS_JAX_RHSMODE1_DD_OVERLAP", ""),
        patch_dof_target_env=os_mod.environ.get(
            "SFINCS_JAX_RHSMODE1_SCHWARZ_PATCH_DOF_TARGET", ""
        ),
    )

    pas_auto_strong_ratio_env = os_mod.environ.get(
        "SFINCS_JAX_PAS_AUTO_STRONG_RATIO", ""
    ).strip()
    try:
        pas_auto_strong_ratio = (
            float(pas_auto_strong_ratio_env) if pas_auto_strong_ratio_env else 10.0
        )
    except ValueError:
        pas_auto_strong_ratio = 10.0
    er_abs = 0.0
    schur_er_min = 1.0e-12

    selection_values = dict(values)
    selection_values.update(locals())
    selection_values.update(
        {
            "_canonical_rhs1_preconditioner_kind": canonical_rhs1_preconditioner_kind,
            "_matvec_shard_axis": matvec_shard_axis,
            "_pas_tz_preconditioner_applicable": pas_tz_preconditioner_applicable,
            "_rhs1_fp_dkes_default_kind": rhs1_fp_dkes_default_kind,
            "_rhs1_pas_auto_large_base_kind": rhs1_pas_auto_large_base_kind,
            "_rhs1_pas_dkes_pas_tz_preferred": rhs1_pas_dkes_pas_tz_preferred,
            "_rhs1_pas_dkes_xblock_allowed": rhs1_pas_dkes_xblock_allowed,
            "_rhs1_pas_small_near_zero_er_kind": rhs1_pas_small_near_zero_er_kind,
            "_rhs1_pas_tokamak_gpu_theta_allowed": (
                rhs1_pas_tokamak_gpu_theta_allowed
            ),
            "_rhs1_pas_tokamak_gpu_xblock_preferred": (
                rhs1_pas_tokamak_gpu_xblock_preferred
            ),
            "jax": jax_mod,
            "np": np_mod,
            "os": os_mod,
        }
    )
    rhs1_default_precond_selection = resolve_rhs1_default_preconditioner_selection(
        RHS1DefaultPreconditionerSelectionContext(selection_values)
    )
    if "er_abs" in rhs1_default_precond_selection:
        er_abs = rhs1_default_precond_selection["er_abs"]
    if "lmax_use" in rhs1_default_precond_selection:
        lmax_use = rhs1_default_precond_selection["lmax_use"]
    if "max_l" in rhs1_default_precond_selection:
        max_l = rhs1_default_precond_selection["max_l"]
    if "nxi_for_x" in rhs1_default_precond_selection:
        nxi_for_x = rhs1_default_precond_selection["nxi_for_x"]
    if "rhs1_gpu_tokamak_pas_tight_gmres" in rhs1_default_precond_selection:
        rhs1_gpu_tokamak_pas_tight_gmres = rhs1_default_precond_selection[
            "rhs1_gpu_tokamak_pas_tight_gmres"
        ]
    if "rhs1_precond_env" in rhs1_default_precond_selection:
        rhs1_precond_env = rhs1_default_precond_selection["rhs1_precond_env"]
    if "rhs1_precond_kind" in rhs1_default_precond_selection:
        rhs1_precond_kind = rhs1_default_precond_selection["rhs1_precond_kind"]
    if "rhs1_xblock_tz_lmax" in rhs1_default_precond_selection:
        rhs1_xblock_tz_lmax = rhs1_default_precond_selection["rhs1_xblock_tz_lmax"]
    if "schur_er_min" in rhs1_default_precond_selection:
        schur_er_min = rhs1_default_precond_selection["schur_er_min"]
    if "tokamak_like" in rhs1_default_precond_selection:
        tokamak_like = rhs1_default_precond_selection["tokamak_like"]
    if "use_collision_precond" in rhs1_default_precond_selection:
        use_collision_precond = rhs1_default_precond_selection["use_collision_precond"]
    if "xblock_tz_max" in rhs1_default_precond_selection:
        xblock_tz_max = rhs1_default_precond_selection["xblock_tz_max"]
    if (
        (not rhs1_precond_env)
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and rhs1_precond_kind
        in {
            None,
            "collision",
            "point",
            "xmg",
            "theta_line",
            "zeta_line",
            "theta_zeta",
            "xblock_tz",
            "xblock_tz_lmax",
            "theta_line_xdiag",
        }
        and not rhs1_gpu_tokamak_pas_tight_gmres
    ):
        max_l_local = int(np_mod.max(nxi_for_x)) if nxi_for_x.size else 0
        rhs1_precond_kind = rhs1_pas_weak_auto_override_kind(
            rhs1_precond_env=rhs1_precond_env,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_pas=op.fblock.pas is not None,
            current_kind=rhs1_precond_kind,
            active_size=int(active_size),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l_local),
        )
        if rhs1_precond_kind == "xblock_tz" and rhs1_pas_dkes_pas_tz_preferred(
            has_pas=op.fblock.pas is not None,
            use_dkes=bool(use_dkes),
            backend=jax_mod.default_backend(),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l_local),
            active_size=int(active_size),
        ):
            rhs1_precond_kind = "pas_tz"
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: PAS DKES "
                    "weak-auto override -> pas_tz preconditioner",
                )
    tokamak_like = bool(geom_scheme == 1 or int(op.n_zeta) <= 5)
    rhs1_precond_kind = rhs1_pas_family_refinement_kind(
        rhs1_precond_env=rhs1_precond_env,
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        current_kind=rhs1_precond_kind,
        active_size=int(active_size),
        n_zeta=int(op.n_zeta),
        geom_scheme=int(geom_scheme),
        pas_tz_applicable=pas_tz_preconditioner_applicable(op),
        pas_tokamak_theta_applicable=pas_tokamak_theta_preconditioner_applicable(op),
    )
    if (
        rhs1_precond_env in {"", "auto", "default"}
        and rhs1_precond_kind == "schur"
        and rhs1_pas_full_pas_tz_preferred(
            has_pas=op.fblock.pas is not None,
            has_fp=op.fblock.fp is not None,
            use_dkes=bool(use_dkes),
            backend=jax_mod.default_backend(),
            geom_scheme=int(geom_scheme),
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l),
            active_size=int(active_size),
            pas_tz_applicable=pas_tz_preconditioner_applicable(op),
        )
    ):
        rhs1_precond_kind = "pas_tz"
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: full-trajectory PAS "
                "auto -> pas_tz preconditioner",
            )
    if rhs1_geometry4_pas_memory_pas_tz_preferred(
        rhs1_precond_env=rhs1_precond_env,
        current_kind=rhs1_precond_kind,
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        use_dkes=bool(use_dkes),
        geom_scheme=int(geom_scheme),
        n_theta=int(op.n_theta),
        n_zeta=int(op.n_zeta),
        max_l=int(max_l),
        active_size=int(active_size),
        er_abs=float(er_abs),
        schur_er_min=float(schur_er_min),
        pas_tz_applicable=pas_tz_preconditioner_applicable(op),
    ):
        rhs1_precond_kind = "pas_tz"
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: geometry4 PAS memory "
                "auto -> pas_tz preconditioner",
            )
    if (
        tokamak_like
        and rhs1_precond_env in {"", "auto", "default"}
        and rhs1_precond_kind
        in {
            "pas_lite",
            "pas_hybrid",
            "pas_tokamak_theta",
            "pas_tz",
            "xmg",
            "collision",
            "point",
        }
        and op.fblock.pas is not None
        and op.fblock.fp is None
        and (not pas_tz_preconditioner_applicable(op))
        and (not pas_tokamak_theta_preconditioner_applicable(op))
        and (
            not rhs1_pas_tokamak_gpu_theta_allowed(
                has_pas=op.fblock.pas is not None,
                has_fp=op.fblock.fp is not None,
                backend=jax_mod.default_backend(),
                tokamak_like=tokamak_like,
                active_size=int(active_size),
                er_abs=float(er_abs),
                schur_er_min=float(schur_er_min),
                has_magdrift=(
                    op.fblock.magdrift_theta is not None
                    or op.fblock.magdrift_zeta is not None
                    or op.fblock.magdrift_xidot is not None
                ),
                has_collisionless=op.fblock.collisionless is not None,
            )
        )
    ):
        if rhs1_pas_tokamak_cpu_xblock_preferred(
            has_pas=op.fblock.pas is not None,
            has_fp=op.fblock.fp is not None,
            backend=jax_mod.default_backend(),
            tokamak_like=tokamak_like,
            active_size=int(active_size),
            er_abs=float(er_abs),
            schur_er_min=float(schur_er_min),
            has_magdrift=(
                op.fblock.magdrift_theta is not None
                or op.fblock.magdrift_zeta is not None
                or op.fblock.magdrift_xidot is not None
            ),
            has_collisionless=op.fblock.collisionless is not None,
            n_theta=int(op.n_theta),
            n_zeta=int(op.n_zeta),
            max_l=int(max_l),
            xblock_tz_limit=max(1, int(xblock_tz_max)),
        ):
            rhs1_precond_kind = "xblock_tz"
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: CPU PAS tokamak "
                    "auto -> xblock_tz preconditioner",
                )
        else:
            rhs1_precond_kind = "pas_schur"
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and float(er_abs) <= float(schur_er_min)
    ):
        rhs1_precond_kind_override = rhs1_large_fp_near_zero_er_override_kind(
            rhs1_precond_env=rhs1_precond_env,
            rhs_mode=int(op.rhs_mode),
            include_phi1=bool(op.include_phi1),
            has_fp=op.fblock.fp is not None,
            has_pas=op.fblock.pas is not None,
            current_kind=rhs1_precond_kind,
            total_size=int(op.total_size),
            er_abs=float(er_abs),
            schur_er_min=float(schur_er_min),
        )
        if rhs1_precond_kind_override != rhs1_precond_kind:
            rhs1_precond_kind = rhs1_precond_kind_override
            if emit is not None:
                emit(
                    1,
                    "solve_v3_full_system_linear_gmres: large FP near-zero-Er "
                    "auto override -> xmg preconditioner",
                )
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and op.fblock.pas is not None
        and rhs1_precond_kind in {None, "collision", "point"}
        and not rhs1_gpu_tokamak_pas_tight_gmres
    ):
        pas_strong_max_env = os_mod.environ.get(
            "SFINCS_JAX_PAS_STRONG_MAX", ""
        ).strip()
        try:
            pas_strong_max = int(pas_strong_max_env) if pas_strong_max_env else 25000
        except ValueError:
            pas_strong_max = 25000
        if int(active_size) <= max(1, int(pas_strong_max)):
            rhs1_precond_kind = "pas_hybrid"
    if (
        rhs1_precond_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.pas is not None
        and int(op.n_species) >= 2
        and (geom_scheme == 1 or int(op.n_zeta) <= 9)
        and rhs1_precond_kind in {"theta_line", "zeta_line", "theta_zeta"}
    ):
        rhs1_precond_kind = "schur"
    rhs1_precond_kind_requested = rhs1_precond_kind
    if rhs1_precond_env == "" and rhs1_precond_kind == "point" and use_pas_projection:
        rhs1_precond_kind = (
            "theta_line" if int(op.n_theta) >= int(op.n_zeta) else "zeta_line"
        )
    if rhs1_precond_env == "":
        shard_axis = matvec_shard_axis(op)
        if shard_axis in {"theta", "zeta"} and jax_mod.device_count() > 1:
            pas_tz_estimate = estimate_rhs1_pas_tz_build_bytes(op)
            pas_tz_max_bytes = rhs1_pas_tz_max_bytes()
            pas_shard_xmg_min_env = os_mod.environ.get(
                "SFINCS_JAX_RHSMODE1_PAS_SHARD_XMG_MIN", ""
            ).strip()
            try:
                pas_shard_xmg_min = (
                    int(pas_shard_xmg_min_env) if pas_shard_xmg_min_env else 80000
                )
            except ValueError:
                pas_shard_xmg_min = 80000
            fp_shard_xmg_min_env = os_mod.environ.get(
                "SFINCS_JAX_RHSMODE1_FP_SHARD_XMG_MIN", ""
            ).strip()
            try:
                fp_shard_xmg_min = (
                    int(fp_shard_xmg_min_env) if fp_shard_xmg_min_env else 120000
                )
            except ValueError:
                fp_shard_xmg_min = 120000
            keep_xmg_for_large_pas_er = bool(
                rhs1_precond_kind == "xmg"
                and op.fblock.pas is not None
                and int(op.total_size) >= max(1, int(pas_shard_xmg_min))
            )
            keep_xmg_for_large_fp = bool(
                rhs1_precond_kind == "xmg"
                and op.fblock.fp is not None
                and int(op.total_size) >= max(1, int(fp_shard_xmg_min))
            )
            schwarz_auto_min_env = os_mod.environ.get(
                "SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", ""
            ).strip()
            try:
                schwarz_auto_min = (
                    int(schwarz_auto_min_env) if schwarz_auto_min_env else 120000
                )
            except ValueError:
                schwarz_auto_min = 120000
            force_schwarz = bool(schwarz_auto_min_env) and int(schwarz_auto_min) <= 0
            if force_schwarz:
                rhs1_precond_kind = (
                    "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                )
            elif rhs1_sharded_line_override_allowed(rhs1_precond_kind):
                if (
                    rhs1_precond_kind in {"theta_line", "zeta_line"}
                    and rhs1_precond_kind != f"{shard_axis}_line"
                ):
                    pass
                elif keep_xmg_for_large_pas_er or keep_xmg_for_large_fp:
                    pass
                elif op.fblock.pas is not None and pas_tz_estimate > max(
                    0, int(pas_tz_max_bytes)
                ):
                    rhs1_precond_kind = (
                        "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                    )
                    if emit is not None:
                        emit(
                            1,
                            "solve_v3_full_system_linear_gmres: sharded PAS large pas_tz "
                            f"(est={pas_tz_estimate / 2**30:.2f} GiB > cap={pas_tz_max_bytes / 2**30:.2f} GiB) -> "
                            f"{rhs1_precond_kind}",
                        )
                elif int(op.total_size) >= max(1, int(schwarz_auto_min)):
                    rhs1_precond_kind = (
                        "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                    )
                else:
                    rhs1_precond_kind = (
                        "theta_line" if shard_axis == "theta" else "zeta_line"
                    )
    if (
        rhs1_precond_env == ""
        and rhs1_precond_kind == "schur"
        and op.fblock.pas is not None
        and (not bool(op.include_phi1))
    ):
        shard_axis = matvec_shard_axis(op)
        if shard_axis in {"theta", "zeta"} and jax_mod.device_count() > 1:
            schur_shard_max_env = os_mod.environ.get(
                "SFINCS_JAX_RHSMODE1_SCHUR_SHARD_MAX", ""
            ).strip()
            try:
                schur_shard_max = (
                    int(schur_shard_max_env) if schur_shard_max_env else 30000
                )
            except ValueError:
                schur_shard_max = 30000
            schur_shard_er_env = os_mod.environ.get(
                "SFINCS_JAX_RHSMODE1_SCHUR_SHARD_ER_MAX", ""
            ).strip()
            try:
                schur_shard_er_max = (
                    float(schur_shard_er_env) if schur_shard_er_env else 1.0e-8
                )
            except ValueError:
                schur_shard_er_max = 1.0e-8
            if int(op.total_size) <= max(1, int(schur_shard_max)) and float(
                er_abs
            ) <= max(0.0, float(schur_shard_er_max)):
                rhs1_precond_kind = (
                    "theta_schwarz" if shard_axis == "theta" else "zeta_schwarz"
                )
                if emit is not None:
                    emit(
                        1,
                        "solve_v3_full_system_linear_gmres: sharded PAS near-zero-Er -> "
                        f"{rhs1_precond_kind} preconditioner",
                    )
    if str(solve_method).strip().lower() in {"dense", "dense_ksp", "dense_row_scaled"}:
        rhs1_precond_kind = None
    pas_tokamak_gpu_tol = rhs1_pas_tokamak_gpu_tight_tol(
        enabled=rhs1_gpu_tokamak_pas_tight_gmres
        or rhs1_precond_kind == "pas_tokamak_theta",
        has_pas=op.fblock.pas is not None,
        has_fp=op.fblock.fp is not None,
        backend=jax_mod.default_backend(),
        tokamak_like=tokamak_like,
        active_size=int(active_size),
        er_abs=float(er_abs),
        schur_er_min=float(schur_er_min),
        has_magdrift=(
            op.fblock.magdrift_theta is not None
            or op.fblock.magdrift_zeta is not None
            or op.fblock.magdrift_xidot is not None
        ),
        has_collisionless=op.fblock.collisionless is not None,
    )
    if pas_tokamak_gpu_tol is not None:
        tol_old = float(tol)
        tol = min(float(tol), float(pas_tokamak_gpu_tol))
        if emit is not None and float(tol) < tol_old:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: GPU PAS tokamak "
                f"tol tightened {tol_old:.1e} -> {float(tol):.1e}",
            )
    maxiter_env = os_mod.environ.get("SFINCS_JAX_GMRES_MAXITER", "").strip()
    if (
        (not rhs1_precond_env)
        and maxiter_env == ""
        and int(op.rhs_mode) == 1
        and (not bool(op.include_phi1))
        and op.fblock.fp is not None
        and op.fblock.pas is None
        and rhs1_precond_kind == "xmg"
        and int(op.total_size) >= 120000
    ):
        fp_auto_maxiter = 800
        maxiter = min(int(maxiter if maxiter is not None else 400), int(fp_auto_maxiter))
        fp_auto_restart_max = 160
        restart = max(80, min(int(restart), int(fp_auto_restart_max)))
        if emit is not None:
            emit(
                1,
                "solve_v3_full_system_linear_gmres: large FP auto-tune "
                f"(precond=xmg restart={int(restart)} maxiter={int(maxiter)})",
            )
    structured_fblock_precond_requested = str(rhs1_precond_kind or "").startswith(
        "structured_fblock_"
    )
    rhs1_precond_enabled = (
        rhs1_precond_kind is not None
        and int(op.rhs_mode) == 1
        and ((not bool(op.include_phi1)) or bool(structured_fblock_precond_requested))
    )
    set_precond_policy_hints(
        geom_scheme=geom_scheme,
        use_dkes=bool(use_dkes),
        rhs1_precond_kind=rhs1_precond_kind,
        has_pas=getattr(op.fblock, "pas", None) is not None,
        has_fp=getattr(op.fblock, "fp", None) is not None,
        include_phi1=bool(op.include_phi1),
        rhs_mode=int(op.rhs_mode),
        er_abs=float(er_abs),
    )

    result: dict[str, Any] = {
        "er_abs": er_abs,
        "max_l": max_l,
        "nxi_for_x": nxi_for_x,
        "pas_auto_strong_ratio": pas_auto_strong_ratio,
        "pre_theta": pre_theta,
        "pre_zeta": pre_zeta,
        "restart": restart,
        "rhs1_bicgstab_env": rhs1_bicgstab_env,
        "rhs1_bicgstab_env_user": rhs1_bicgstab_env_user,
        "rhs1_dd_setup": rhs1_dd_setup,
        "rhs1_gpu_tokamak_pas_tight_gmres": rhs1_gpu_tokamak_pas_tight_gmres,
        "rhs1_precond_enabled": rhs1_precond_enabled,
        "rhs1_precond_env": rhs1_precond_env,
        "rhs1_precond_env_user": rhs1_precond_env_user,
        "rhs1_precond_kind": rhs1_precond_kind,
        "rhs1_precond_kind_requested": rhs1_precond_kind_requested,
        "rhs1_xblock_tz_lmax": rhs1_xblock_tz_lmax,
        "schur_er_min": schur_er_min,
        "structured_fblock_precond_requested": structured_fblock_precond_requested,
        "tokamak_like": tokamak_like,
        "tol": tol,
        "maxiter": maxiter,
    }
    if "lmax_use" in locals():
        result["lmax_use"] = lmax_use
    if "use_collision_precond" in locals():
        result["use_collision_precond"] = use_collision_precond
    if "xblock_tz_max" in locals():
        result["xblock_tz_max"] = xblock_tz_max
    return result

def canonical_rhs1_preconditioner_kind(raw: str | None) -> str | None:
    """Canonicalize ``SFINCS_JAX_RHSMODE1_PRECONDITIONER`` aliases.

    Unknown non-empty aliases intentionally return ``None`` so unrecognized
    values fail closed without changing default solver selection.
    """
    key = str(raw or "").strip().lower()
    if not key:
        return None
    return _RHS1_PRECONDITIONER_KIND_ALIASES.get(key)

def rhs1_measured_auto_promotion_gate(
    *,
    current_kind: str | None,
    candidate_kind: str | None,
    candidate_metrics: SolverCandidateMetrics | None = None,
    baseline_metrics: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> SolverCandidateGate:
    """Gate an automatic RHSMode=1 preconditioner promotion with measured data.

    Existing heuristic policy calls pass no metrics, so they remain unchanged.
    Once a dispatch point has residual/runtime/memory data for a candidate, this
    helper enforces the common rule: do not promote a new automatic path over a
    clean incumbent unless the candidate is residual-clean and provides a real
    measured runtime or memory win.
    """
    if candidate_kind == current_kind:
        return SolverCandidateGate(accepted=True, reasons=("same_kind",))
    if candidate_metrics is None:
        return SolverCandidateGate(accepted=True, reasons=("unmeasured_historical_policy",))
    return solver_candidate_gate(candidate_metrics, baseline=baseline_metrics, criteria=criteria)

def rhs1_pas_auto_large_base_kind(*, active_size: int) -> str:
    """Keep large auto-selected PAS solves in the PAS-native preconditioner family."""
    pas_lite_min = _env_int("SFINCS_JAX_PAS_LITE_MIN", 20000)
    if int(active_size) >= max(1, int(pas_lite_min)):
        return "pas_lite"
    return "pas_hybrid"

def rhs1_pas_weak_auto_override_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    current_kind: str | None,
    active_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    candidate_metrics: SolverCandidateMetrics | None = None,
    baseline_metrics: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> str | None:
    """Promote weak default PAS preconditioners to PAS-aware defaults.

    This mirrors the driver auto-policy used before expensive PAS fallback
    attempts: small angular blocks may use ``xblock_tz``; larger systems stay in
    the PAS-native lite/hybrid family.
    """
    if str(rhs1_precond_env or "").strip().lower():
        return current_kind
    if int(rhs_mode) != 1 or bool(include_phi1) or not has_pas:
        return current_kind
    if current_kind not in PAS_WEAK_AUTO_OVERRIDE_KINDS:
        return current_kind

    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", 1200)
    xblock_small_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_SMALL_MAX", 4000)
    if (
        int(active_size) <= max(1, int(xblock_small_max))
        and int(xblock_tz_max) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
    ):
        candidate_kind = "xblock_tz"
    else:
        candidate_kind = rhs1_pas_auto_large_base_kind(active_size=int(active_size))
    gate = rhs1_measured_auto_promotion_gate(
        current_kind=current_kind,
        candidate_kind=candidate_kind,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        criteria=criteria,
    )
    return candidate_kind if gate.accepted else current_kind

def rhs1_measured_auto_promotion_allowed(
    *,
    current_kind: str | None,
    candidate_kind: str | None,
    candidate_metrics: SolverCandidateMetrics | None = None,
    baseline_metrics: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> bool:
    """Return whether measured data allow an RHSMode=1 automatic promotion."""
    return rhs1_measured_auto_promotion_gate(
        current_kind=current_kind,
        candidate_kind=candidate_kind,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
        criteria=criteria,
    ).accepted

def rhs1_pas_family_refinement_kind(
    *,
    rhs1_precond_env: str,
    has_pas: bool,
    has_fp: bool,
    current_kind: str | None,
    active_size: int,
    n_zeta: int,
    geom_scheme: int,
    pas_tz_applicable: bool,
    pas_tokamak_theta_applicable: bool,
) -> str | None:
    """Refine automatic PAS lite/hybrid selections to specialized PAS builders."""
    result = current_kind
    env = str(rhs1_precond_env or "").strip().lower()
    tokamak_geometry = int(n_zeta) == 1 or int(geom_scheme) == 1
    tokamak_like = int(geom_scheme) == 1 or int(n_zeta) <= 5

    if result == "pas_lite" and has_pas and tokamak_geometry:
        # GeometryScheme=1 tokamak PAS runs need stronger angular/L coupling
        # than pas_lite provides, but can still avoid the most expensive global
        # blocks by staying in the hybrid family.
        result = "pas_hybrid"
    if env in {"", "auto", "default"} and result in {"pas_lite", "pas_hybrid"} and pas_tokamak_theta_applicable:
        return "pas_tokamak_theta"
    if (
        env in {"", "auto", "default"}
        and result in {"pas_lite", "pas_hybrid"}
        and pas_tz_applicable
        and (not pas_tokamak_theta_applicable)
    ):
        return "pas_tz"
    if tokamak_like and result in {"pas_lite", "pas_hybrid"} and pas_tz_applicable:
        return "pas_tz"
    if env == "" and has_pas and tokamak_like and result in {"pas_lite", "pas_hybrid"}:
        pas_ilu_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_ILU_MIN", 12000)
        if int(active_size) >= max(1, int(pas_ilu_min)):
            return "pas_ilu"
    return result

def rhs1_fp_dkes_env_preconditioner_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_fp: bool,
    use_dkes: bool,
    total_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
) -> str:
    """Return an early env override for bounded FP DKES xblock_tz solves."""
    env = str(rhs1_precond_env or "").strip().lower()
    if env:
        return env
    if int(rhs_mode) != 1 or bool(include_phi1) or (not has_fp) or (not use_dkes):
        return env

    fp_dkes_max = _env_int("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", 20000)
    if int(total_size) > max(1, int(fp_dkes_max)):
        return env

    xblock_tz_max = _env_int("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", 1200)
    if (
        int(n_theta) > 1
        and int(xblock_tz_max) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_max)
    ):
        return "xblock_tz"
    return env

def rhs1_fp_dkes_default_kind(
    *,
    active_size: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> str:
    """Select the default RHSMode=1 preconditioner for FP DKES trajectory cases."""
    fp_dkes_strong_max = _env_int("SFINCS_JAX_RHSMODE1_FP_DKES_STRONG_MAX", 20000)
    if int(active_size) > max(1, int(fp_dkes_strong_max)):
        return "collision"
    if (
        int(n_theta) > 1
        and int(xblock_tz_limit) > 0
        and int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)
    ):
        return "xblock_tz"
    return "xmg"

def rhs1_large_fp_near_zero_er_override_kind(
    *,
    rhs1_precond_env: str,
    rhs_mode: int,
    include_phi1: bool,
    has_fp: bool,
    has_pas: bool,
    current_kind: str | None,
    total_size: int,
    er_abs: float,
    schur_er_min: float,
) -> str | None:
    """Force large near-zero-Er FP-only systems from weak line/point blocks to xmg."""
    if str(rhs1_precond_env or "").strip().lower():
        return current_kind
    if int(rhs_mode) != 1 or bool(include_phi1) or (not has_fp) or has_pas:
        return current_kind
    if float(er_abs) > float(schur_er_min):
        return current_kind
    if current_kind not in FP_FORCE_XMG_WEAK_KINDS:
        return current_kind

    fp_force_xmg_min = _env_int("SFINCS_JAX_RHSMODE1_FP_FORCE_XMG_MIN", 120000)
    if int(total_size) >= max(1, int(fp_force_xmg_min)):
        return "xmg"
    return current_kind

def pas_auto_skip_strong_retry(
    *,
    has_pas: bool,
    strong_precond_env: str,
    rhs1_precond_kind: str | None,
    residual_norm: float,
    target: float,
    ratio: float,
) -> bool:
    """Skip PAS strong retry when the current strong base already met the relaxed target."""
    if not has_pas or ratio <= 0.0:
        return False
    if strong_precond_env not in {"", "auto"}:
        return False
    if rhs1_precond_kind not in PAS_AUTO_STRONG_BASE_KINDS:
        return False
    return float(residual_norm) <= float(target) * float(ratio)

def rhs1_pas_dkes_xblock_allowed(
    *,
    has_pas: bool,
    use_dkes: bool,
    backend: str,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Return whether bounded PAS DKES runs may use dense xblock_tz preconditioning."""
    if not has_pas or not use_dkes:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu", "tpu"}:
        return False
    if int(n_theta) <= 1:
        return False
    if int(xblock_tz_limit) <= 0:
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)

def rhs1_pas_dkes_pas_tz_preferred(
    *,
    has_pas: bool,
    use_dkes: bool,
    backend: str,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
) -> bool:
    """Return whether PAS DKES auto-selection should prefer ``pas_tz``.

    Dense x-blocks are robust for small DKES angular blocks, but on the HSX DKES
    benchmark the structured PAS angular block is faster and lower-memory on
    both CPU and GPU once the angular block reaches O(10^3) DOFs.
    """
    if not has_pas or not use_dkes:
        return False
    backend_norm = str(backend).strip().lower()
    if backend_norm not in {"cpu", "gpu"}:
        return False
    if int(n_theta) <= 1 or int(n_zeta) <= 1:
        return False
    backend_key = backend_norm.upper()
    min_block = _env_int(f"SFINCS_JAX_RHSMODE1_PAS_DKES_{backend_key}_PAS_TZ_MIN", 950)
    max_active = _env_int(f"SFINCS_JAX_RHSMODE1_PAS_DKES_{backend_key}_PAS_TZ_ACTIVE_MAX", 15000)
    block_size = int(max_l) * int(n_theta) * int(n_zeta)
    return block_size >= max(1, int(min_block)) and int(active_size) <= max(1, int(max_active))

def rhs1_pas_dkes_cpu_pas_tz_preferred(
    *,
    has_pas: bool,
    use_dkes: bool,
    backend: str,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
) -> bool:
    """Backward-compatible CPU PAS-DKES alias for older policy tests/callers."""
    return rhs1_pas_dkes_pas_tz_preferred(
        has_pas=has_pas,
        use_dkes=use_dkes,
        backend=backend,
        n_theta=n_theta,
        n_zeta=n_zeta,
        max_l=max_l,
        active_size=active_size,
    )

def rhs1_pas_full_cpu_pas_tz_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    use_dkes: bool,
    backend: str,
    geom_scheme: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
    pas_tz_applicable: bool,
) -> bool:
    """Return whether bounded CPU full-trajectory PAS should prefer ``pas_tz``.

    This targets bounded geometryScheme=11 full-trajectory PAS cases where
    ``pas_tz`` is faster and much lower-memory than the default Schur block on
    CPU.  The backend-general wrapper below adds the corresponding bounded GPU
    gate using separate environment controls.
    """
    if not has_pas or has_fp or use_dkes:
        return False
    if not pas_tz_applicable:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if int(geom_scheme) != 11:
        return False
    if int(n_theta) <= 1 or int(n_zeta) <= 1:
        return False
    max_zeta = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_NZETA_MAX", 19)
    min_block = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_MIN", 950)
    max_active = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_CPU_PAS_TZ_ACTIVE_MAX", 15000)
    block_size = int(max_l) * int(n_theta) * int(n_zeta)
    return (
        int(n_zeta) <= max(1, int(max_zeta))
        and block_size >= max(1, int(min_block))
        and int(active_size) <= max(1, int(max_active))
    )

def rhs1_pas_full_pas_tz_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    use_dkes: bool,
    backend: str,
    geom_scheme: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
    pas_tz_applicable: bool,
) -> bool:
    """Return whether bounded full-trajectory PAS should prefer ``pas_tz``.

    CPU behavior is delegated to the existing CPU policy. GPU behavior is
    admitted only for the measured geometryScheme=11 full-trajectory PAS lane
    where ``pas_tz`` lowers RSS and wall time relative to Schur while preserving
    parity. The bounds deliberately mirror the CPU lane and can be tightened or
    disabled with backend-specific environment variables.
    """
    backend_key = str(backend).strip().lower()
    if backend_key == "cpu":
        return rhs1_pas_full_cpu_pas_tz_preferred(
            has_pas=has_pas,
            has_fp=has_fp,
            use_dkes=use_dkes,
            backend=backend,
            geom_scheme=geom_scheme,
            n_theta=n_theta,
            n_zeta=n_zeta,
            max_l=max_l,
            active_size=active_size,
            pas_tz_applicable=pas_tz_applicable,
        )
    if backend_key not in {"gpu", "cuda"}:
        return False
    mode = _env_token("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ")
    if mode in _FALSE_VALUES:
        return False
    if not has_pas or has_fp or use_dkes:
        return False
    if not pas_tz_applicable:
        return False
    if int(geom_scheme) != 11:
        return False
    if int(n_theta) <= 1 or int(n_zeta) <= 1:
        return False
    max_zeta = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_NZETA_MAX", 19)
    min_block = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_MIN", 950)
    max_active = _env_int("SFINCS_JAX_RHSMODE1_PAS_FULL_GPU_PAS_TZ_ACTIVE_MAX", 15000)
    block_size = int(max_l) * int(n_theta) * int(n_zeta)
    return (
        int(n_zeta) <= max(1, int(max_zeta))
        and block_size >= max(1, int(min_block))
        and int(active_size) <= max(1, int(max_active))
    )

def rhs1_geometry4_pas_memory_pas_tz_preferred(
    *,
    rhs1_precond_env: str | None,
    current_kind: str | None,
    has_pas: bool,
    has_fp: bool,
    use_dkes: bool,
    geom_scheme: int,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    pas_tz_applicable: bool,
) -> bool:
    """Return whether geometryScheme=4 PAS should use memory-oriented ``pas_tz``.

    This targets the bounded no-Er geometry4 PAS offender where direct top-level
    ``pas_tz`` is parity-clean and materially lower-memory than wrapping the same
    angular block inside the constraint-Schur preconditioner.
    """
    mode = _env_token("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ")
    if mode in _FALSE_VALUES:
        return False
    if (rhs1_precond_env or "").strip().lower() not in {"", "auto", "default"}:
        return False
    if current_kind != "schur":
        return False
    if not has_pas or has_fp or use_dkes:
        return False
    if not pas_tz_applicable:
        return False
    if int(geom_scheme) != 4:
        return False
    if int(n_theta) <= 1 or int(n_zeta) <= 1:
        return False
    if float(er_abs) > float(schur_er_min):
        return False
    block_size = int(max_l) * int(n_theta) * int(n_zeta)
    min_block = _env_int("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_MIN", 1500)
    min_active = _env_int("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MIN", 8000)
    max_active = _env_int("SFINCS_JAX_RHSMODE1_GEOM4_PAS_MEMORY_PAS_TZ_ACTIVE_MAX", 25000)
    return (
        block_size >= max(1, int(min_block))
        and int(active_size) >= max(1, int(min_active))
        and int(active_size) <= max(1, int(max_active))
    )

def rhs1_pas_tokamak_gpu_theta_allowed(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
) -> bool:
    """Return whether the bounded GPU tokamak PAS theta/L path is eligible."""
    if not has_pas or has_fp:
        return False
    if str(backend).strip().lower() == "cpu":
        return False
    if not tokamak_like or not has_collisionless:
        return False
    if float(er_abs) <= float(schur_er_min):
        return False
    if has_magdrift:
        return False
    theta_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_MAX", 8000)
    return int(active_size) <= max(1, int(theta_max))

def rhs1_pas_tokamak_gpu_xblock_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Return whether GPU tokamak PAS+Er should opt into ``xblock_tz``.

    Very small one-GPU tokamak PAS+Er cases are fastest with the tightened
    unpreconditioned route because setup dominates. Medium cases need the
    structured xblock_tz preconditioner to avoid expensive sparse-LU fallbacks.
    """
    if not rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=has_pas,
        has_fp=has_fp,
        backend=backend,
        tokamak_like=tokamak_like,
        active_size=active_size,
        er_abs=er_abs,
        schur_er_min=schur_er_min,
        has_magdrift=has_magdrift,
        has_collisionless=has_collisionless,
    ):
        return False
    if int(n_theta) <= 1 or int(xblock_tz_limit) <= 0:
        return False
    prefer_min = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MIN", 1000)
    prefer_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_XBLOCK_ACTIVE_MAX", 8000)
    if int(active_size) < max(1, int(prefer_min)):
        return False
    if int(prefer_max) <= 0:
        return False
    if int(active_size) > int(prefer_max):
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)

def rhs1_pas_tokamak_gpu_tight_tol(
    *,
    enabled: bool,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
) -> float | None:
    """Return the auto-tightened GPU tokamak PAS tolerance, if applicable."""
    if not enabled:
        return None
    if not rhs1_pas_tokamak_gpu_theta_allowed(
        has_pas=has_pas,
        has_fp=has_fp,
        backend=backend,
        tokamak_like=tokamak_like,
        active_size=active_size,
        er_abs=er_abs,
        schur_er_min=schur_er_min,
        has_magdrift=has_magdrift,
        has_collisionless=has_collisionless,
    ):
        return None
    raw = _env_token("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_TOL")
    if not raw:
        raw = _env_token("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_GPU_THETA_TOL")
    if raw in _FALSE_VALUES:
        return None
    try:
        tol = float(raw) if raw else 1.0e-8
    except ValueError:
        tol = 1.0e-8
    return tol if tol > 0.0 else None

def rhs1_pas_tokamak_cpu_xblock_preferred(
    *,
    has_pas: bool,
    has_fp: bool,
    backend: str,
    tokamak_like: bool,
    active_size: int,
    er_abs: float,
    schur_er_min: float,
    has_magdrift: bool,
    has_collisionless: bool,
    n_theta: int,
    n_zeta: int,
    max_l: int,
    xblock_tz_limit: int,
) -> bool:
    """Prefer xblock_tz for bounded CPU tokamak PAS+Er branches before pas_schur."""
    if not has_pas or has_fp:
        return False
    if str(backend).strip().lower() != "cpu":
        return False
    if not tokamak_like or not has_collisionless:
        return False
    if float(er_abs) <= float(schur_er_min) and (not has_magdrift):
        return False
    if int(n_theta) <= 1 or int(xblock_tz_limit) <= 0:
        return False
    prefer_max = _env_int("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_CPU_XBLOCK_ACTIVE_MAX", 4000)
    if int(active_size) > max(1, int(prefer_max)):
        return False
    return int(max_l) * int(n_theta) * int(n_zeta) <= int(xblock_tz_limit)

def rhs1_gpu_sparse_fallback_skip_allowed(
    *,
    backend: str,
    rhs_mode: int,
    include_phi1: bool,
    has_pas: bool,
    rhs1_precond_kind: str | None,
    use_active_dof_mode: bool,
    residual_norm: float,
    target: float,
) -> bool:
    """Return whether a GPU PAS sparse fallback can be skipped after Schur acceptance."""
    if str(backend).strip().lower() == "cpu":
        return False
    if not bool(use_active_dof_mode):
        return False
    if int(rhs_mode) != 1 or bool(include_phi1):
        return False
    if not has_pas:
        return False
    if str(rhs1_precond_kind or "").strip().lower() not in {"schur", "pas_schur"}:
        return False
    skip_ratio = _env_float("SFINCS_JAX_RHSMODE1_GPU_SPARSE_SKIP_RATIO", 10.0)
    if skip_ratio <= 0.0:
        return False
    return float(residual_norm) <= float(skip_ratio) * max(float(target), 1.0e-300)

def rhs1_gpu_sparse_fallback_skip_allowed_current_backend(
    *,
    op: object,
    rhs1_precond_kind: str | None,
    use_active_dof_mode: bool,
    residual_norm: float,
    target: float,
) -> bool:
    """Evaluate the GPU sparse-fallback skip policy on the active JAX backend."""

    return rhs1_gpu_sparse_fallback_skip_allowed(
        backend=jax.default_backend(),
        rhs_mode=int(getattr(op, "rhs_mode", 0) or 0),
        include_phi1=bool(getattr(op, "include_phi1", False)),
        has_pas=getattr(getattr(op, "fblock", None), "pas", None) is not None,
        rhs1_precond_kind=rhs1_precond_kind,
        use_active_dof_mode=bool(use_active_dof_mode),
        residual_norm=float(residual_norm),
        target=float(target),
    )

def rhsmode1_host_dense_shortcut_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    use_implicit: bool,
    solve_method_kind: str,
) -> bool:
    """Evaluate the host-dense shortcut policy using current backend defaults."""

    return rhs1_host_dense_shortcut_allowed(
        op=op,
        active_size=active_size,
        use_implicit=use_implicit,
        solve_method_kind=solve_method_kind,
        backend=jax.default_backend(),
        dense_fallback_max=rhs1_dense_fallback_max(op),
    )

def rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend(
    *,
    op: object,
    sparse_exact_lu: bool,
    host_sparse_direct_wanted: bool,
) -> bool:
    """Evaluate the Fortran-like sparse-preconditioned rescue policy."""

    return rhs1_sparse_operator_preconditioned_rescue_allowed(
        op=op,
        sparse_exact_lu=sparse_exact_lu,
        host_sparse_direct_wanted=host_sparse_direct_wanted,
        backend=jax.default_backend(),
    )

def host_sparse_factor_dtype_current_backend(
    *,
    size: int,
    factorization: str,
    use_implicit: bool,
) -> np.dtype:
    """Return the host sparse factor dtype for the current backend."""

    return host_sparse_factor_dtype(
        size=size,
        factorization=factorization,
        use_implicit=use_implicit,
        backend=jax.default_backend(),
    )

def rhsmode1_sparse_pc_default_permc_spec(
    *,
    constrained_pas_pc: bool,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> str:
    """Return the measured SuperLU column-ordering default for sparse-PC RHSMode=1."""

    return _solver_path_policy.rhsmode1_sparse_pc_default_permc_spec(
        constrained_pas_pc=constrained_pas_pc,
        tokamak_pas_er_pc=tokamak_pas_er_pc,
        n_species=n_species,
    )

def rhsmode1_sparse_pc_default_restart(
    *,
    requested_restart: int,
    restart_env_value: str,
    tokamak_pas_er_pc: bool,
    n_species: int,
) -> int:
    """Return the sparse-PC GMRES restart after scoped production caps."""

    return _solver_path_policy.rhsmode1_sparse_pc_default_restart(
        requested_restart=requested_restart,
        restart_env_value=restart_env_value,
        tokamak_pas_er_pc=tokamak_pas_er_pc,
        n_species=n_species,
    )

def rhsmode1_pas_fast_accept_current_backend(
    *,
    op: object,
    active_size: int,
    residual_norm: float,
    target: float,
    use_implicit: bool,
) -> bool:
    """Evaluate the PAS fast-accept policy on the current backend."""

    return rhs1_pas_fast_accept(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_constraint0_sparse_first_current_backend(
    *,
    op: object,
    solve_method_kind: str,
    sparse_precond_mode: str,
    active_size: int,
    sparse_max_size: int,
) -> bool:
    """Evaluate the constraintScheme=0 sparse-first policy on the current backend."""

    return rhs1_constraint0_sparse_first(
        op=op,
        solve_method_kind=solve_method_kind,
        sparse_precond_mode=sparse_precond_mode,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        backend=jax.default_backend(),
    )

def rhsmode1_sparse_exact_lu_requested_current_backend(
    *,
    op: object,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    full_precond_requested: bool = False,
    preconditioner_x: int,
    use_dkes: bool,
) -> bool:
    """Evaluate whether RHSMode-1 should request exact sparse LU."""

    return rhs1_sparse_exact_lu_requested(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        full_precond_requested=bool(full_precond_requested),
        preconditioner_x=int(preconditioner_x),
        use_dkes=bool(use_dkes),
        backend=jax.default_backend(),
    )

def rhsmode1_large_cpu_sparse_rescue_allowed_current_backend(
    *,
    op: object,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    residual_norm: float,
    target: float,
) -> bool:
    """Evaluate large-CPU sparse rescue admission on the current backend."""

    return rhs1_large_cpu_sparse_rescue_allowed(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_x=int(preconditioner_x),
        residual_norm=float(residual_norm),
        target=float(target),
        backend=jax.default_backend(),
    )

def rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend(
    *,
    op: object,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    use_implicit: bool,
) -> bool:
    """Evaluate large-CPU sparse skip-primary admission on the current backend."""

    return rhs1_large_cpu_sparse_skip_primary_allowed(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    preconditioner_x: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    xblock_seed_residual: float,
    xblock_seed_improvement_ratio: float,
    use_implicit: bool,
) -> bool:
    """Evaluate whether the sparse exact-LU x-block stage is allowed."""

    return rhs1_large_cpu_sparse_exact_lu_xblock_allowed(
        op=op,
        active_size=int(active_size),
        preconditioner_x=int(preconditioner_x),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        xblock_seed_residual=float(xblock_seed_residual),
        xblock_seed_improvement_ratio=float(xblock_seed_improvement_ratio),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_sparse_xblock_rescue_allowed_current_backend(
    *,
    op: object,
    solve_method_kind: str,
    active_size: int,
    sparse_max_size: int,
    preconditioner_x: int,
    pre_theta: int,
    pre_zeta: int,
    residual_norm: float,
    target: float,
) -> bool:
    """Evaluate sparse x-block rescue admission on the current backend."""

    return rhs1_sparse_xblock_rescue_allowed(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_x=int(preconditioner_x),
        pre_theta=int(pre_theta),
        pre_zeta=int(pre_zeta),
        residual_norm=float(residual_norm),
        target=float(target),
        backend=jax.default_backend(),
    )

def rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend(
    *,
    op: object,
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
) -> bool:
    """Evaluate large-CPU x-block skip-primary admission on the current backend."""

    return rhs1_large_cpu_xblock_skip_primary_allowed(
        op=op,
        solve_method_kind=solve_method_kind,
        active_size=int(active_size),
        sparse_max_size=int(sparse_max_size),
        preconditioner_species=int(preconditioner_species),
        preconditioner_x=int(preconditioner_x),
        preconditioner_xi=int(preconditioner_xi),
        pre_theta=int(pre_theta),
        pre_zeta=int(pre_zeta),
        use_implicit=bool(use_implicit),
        rhs1_precond_env=rhs1_precond_env,
        backend=jax.default_backend(),
    )

def rhsmode1_fast_post_xblock_polish_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    use_implicit: bool,
) -> bool:
    """Evaluate the fast post-x-block polish policy on the current backend."""

    return rhs1_fast_post_xblock_polish_allowed(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_fp_targeted_polish_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    residual_norm: float,
    target: float,
    rhs1_precond_kind: str,
    use_implicit: bool,
) -> bool:
    """Evaluate the FP targeted-polish policy on the current backend."""

    return rhs1_fp_targeted_polish_allowed(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        rhs1_precond_kind=rhs1_precond_kind,
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    residual_norm: float,
    target: float,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> bool:
    """Evaluate whether global sparse rescue should be skipped after x-block."""

    return rhs1_skip_global_sparse_after_xblock_allowed(
        op=op,
        active_size=int(active_size),
        residual_norm=float(residual_norm),
        target=float(target),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend(
    *,
    op: object,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> float:
    """Evaluate the SciPy rescue absolute residual floor after x-block."""

    return rhs1_scipy_rescue_abs_floor_after_xblock(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhsmode1_scipy_rescue_active_size_allowed_current_backend(
    *,
    op: object,
    active_size: int,
    used_large_cpu_xblock_shortcut: bool,
    used_explicit_fp_xblock_seed: bool,
    use_implicit: bool,
) -> bool:
    """Evaluate SciPy rescue size admission after x-block on the current backend."""

    return rhs1_scipy_rescue_active_size_allowed(
        op=op,
        active_size=int(active_size),
        used_large_cpu_xblock_shortcut=bool(used_large_cpu_xblock_shortcut),
        used_explicit_fp_xblock_seed=bool(used_explicit_fp_xblock_seed),
        use_implicit=bool(use_implicit),
        backend=jax.default_backend(),
    )

def rhs1_sharded_line_override_allowed(rhs1_precond_kind: str | None) -> bool:
    """Return whether sharded auto-selection may demote the current preconditioner to line DD."""
    return rhs1_precond_kind in {
        None,
        "point",
        "point_xdiag",
        "theta_line",
        "theta_line_xdiag",
        "zeta_line",
        "xmg",
        "collision",
        "pas_lite",
        "pas_hybrid",
    }

__all__ = (
    "host_sparse_factor_dtype_current_backend",
    "host_sparse_direct_refine_steps",
    "host_sparse_factor_dtype",
    "rhsmode1_constraint0_sparse_first_current_backend",
    "rhsmode1_fast_post_xblock_polish_allowed_current_backend",
    "rhsmode1_fp_targeted_polish_allowed_current_backend",
    "rhsmode1_host_dense_shortcut_allowed_current_backend",
    "rhsmode1_large_cpu_sparse_exact_lu_xblock_allowed_current_backend",
    "rhsmode1_large_cpu_sparse_rescue_allowed_current_backend",
    "rhsmode1_large_cpu_sparse_skip_primary_allowed_current_backend",
    "rhsmode1_large_cpu_xblock_skip_primary_allowed_current_backend",
    "rhsmode1_pas_fast_accept_current_backend",
    "rhsmode1_scipy_rescue_abs_floor_after_xblock_current_backend",
    "rhsmode1_scipy_rescue_active_size_allowed_current_backend",
    "rhsmode1_skip_global_sparse_after_xblock_allowed_current_backend",
    "rhsmode1_sparse_exact_lu_requested_current_backend",
    "rhsmode1_sparse_operator_preconditioned_rescue_allowed_current_backend",
    "rhsmode1_sparse_pc_default_permc_spec",
    "rhsmode1_sparse_pc_default_restart",
    "rhsmode1_sparse_xblock_rescue_allowed_current_backend",
    "rhs1_dense_backend_allowed",
    "rhs1_dense_auto_fp_cutoff",
    "rhs1_dense_auto_fp_allowed",
    "rhs1_dense_auto_fp_accelerator_min",
    "RHS1FullFPDenseAutoRouteDecision",
    "resolve_rhs1_full_fp_dense_auto_route",
    "RHS1InitialSparseShortcutRouteDecision",
    "resolve_rhs1_initial_sparse_shortcut_route",
    "rhs1_dense_fallback_max",
    "rhs1_dense_krylov_allowed",
    "rhs1_explicit_sparse_host_direct_allowed",
    "rhs1_host_dense_fallback_allowed",
    "rhs1_host_dense_shortcut_allowed",
    "rhs1_host_dense_shortcut_estimated_nbytes",
    "rhs1_host_dense_shortcut_max_bytes",
    "rhs1_host_sparse_direct_allowed",
    "rhs1_host_sparse_skip_dense_ratio",
    "rhs1_constrained_pas_sparse_pc_auto_allowed",
    "rhs1_fp_3d_sparse_pc_auto_allowed",
    "rhs1_fp_3d_xblock_sparse_pc_auto_allowed",
    "rhs1_tokamak_er_dense_auto_allowed",
    "rhs1_tokamak_fp_er_sparse_pc_auto_allowed",
    "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed",
    "rhs1_tokamak_pas_er_sparse_pc_auto_allowed",
    "rhs1_tokamak_pas_noer_sparse_pc_auto_allowed",
    "rhs1_sparse_operator_preconditioned_rescue_allowed",
    "rhs1_fp_xblock_assembled_host_allowed",
    "rhs1_large_cpu_sparse_exact_lu_allowed",
    "rhs1_large_cpu_sparse_exact_lu_xblock_allowed",
    "rhs1_large_cpu_sparse_rescue_allowed",
    "rhs1_large_cpu_sparse_rescue_first",
    "rhs1_large_cpu_sparse_skip_primary_allowed",
    "rhs1_large_cpu_xblock_skip_primary_allowed",
    "rhs1_sparse_xblock_rescue_allowed",
    "RHS1DefaultPreconditionerSelectionContext",
    "RHS1PreconditionerRouteSetupContext",
    "PAS_AUTO_STRONG_BASE_KINDS",
    "FP_FORCE_XMG_WEAK_KINDS",
    "PAS_WEAK_AUTO_OVERRIDE_KINDS",
    "canonical_rhs1_preconditioner_kind",
    "pas_auto_skip_strong_retry",
    "resolve_rhs1_default_preconditioner_selection",
    "resolve_rhs1_preconditioner_route_setup",
    "rhs1_measured_auto_promotion_allowed",
    "rhs1_measured_auto_promotion_gate",
    "rhs1_fp_dkes_default_kind",
    "rhs1_fp_dkes_env_preconditioner_kind",
    "rhs1_geometry4_pas_memory_pas_tz_preferred",
    "rhs1_large_fp_near_zero_er_override_kind",
    "rhs1_pas_family_refinement_kind",
    "rhs1_gpu_sparse_fallback_skip_allowed",
    "rhs1_pas_auto_large_base_kind",
    "rhs1_pas_dkes_cpu_pas_tz_preferred",
    "rhs1_pas_dkes_pas_tz_preferred",
    "rhs1_pas_dkes_xblock_allowed",
    "rhs1_pas_full_pas_tz_preferred",
    "rhs1_pas_full_cpu_pas_tz_preferred",
    "rhs1_pas_weak_auto_override_kind",
    "rhs1_pas_tokamak_cpu_xblock_preferred",
    "rhs1_pas_tokamak_gpu_theta_allowed",
    "rhs1_pas_tokamak_gpu_tight_tol",
    "rhs1_pas_tokamak_gpu_xblock_preferred",
    "rhs1_sharded_line_override_allowed",
    "ActiveProjectedPreconditionerAutoPolicy",
    "RHS1Constraint0PETScCompatConfig",
    "RHS1BiCGStabFallbackControls",
    "RHS1BiCGStabFallbackDecision",
    "RHS1FastPostXBlockPolishControls",
    "RHS1FPBiCGStabPolishControls",
    "RHS1FPGlobalLowLPolishControls",
    "RHS1FPL1PolishControls",
    "RHS1FPLowLPolishControls",
    "RHS1FPResidualPolishControls",
    "RHS1KrylovRoutingControls",
    "RHS1ScipyRescueControls",
    "RHS1FullSparseRescueSetupContext",
    "RHS1FullSparseRescueSetupResult",
    "RHS1SparseJAXConfig",
    "RHS1SparseOperatorAdmission",
    "RHS1SparsePreconditionerConfig",
    "RHS1SparseRescueOrdering",
    "RHS1SparseRescuePolicySetup",
    "RHS1Stage2AdmissionControls",
    "RHS1Stage2RetryAdmissionDecision",
    "RHS1Stage2RetryControls",
    "RHS1Stage2TriggerDecision",
    "parse_rhs1_pas_tz_guarded_structured_levels",
    "read_bool_env",
    "read_float_env",
    "read_int_env",
    "resolve_active_projected_preconditioner_auto_policy",
    "rhs1_bicgstab_fallback_controls_from_env",
    "rhs1_bicgstab_fallback_decision",
    "rhs1_bicgstab_fallback_target_from_env",
    "rhs1_bicgstab_preconditioner_kind",
    "rhs1_constraint0_dense_fallback_allowed",
    "rhs1_constraint0_petsc_compat",
    "rhs1_constraint0_petsc_compat_config_from_env",
    "rhs1_constraint0_petsc_compat_regularization",
    "rhs1_constraint0_sparse_first",
    "rhs1_fast_post_xblock_polish_allowed",
    "rhs1_fast_post_xblock_polish_controls_from_env",
    "rhs1_fp_force_stage2",
    "rhs1_fp_bicgstab_polish_controls_from_env",
    "rhs1_fp_global_low_l_polish_controls_from_env",
    "rhs1_fp_l1_polish_controls_from_env",
    "rhs1_fp_low_l_polish_controls_from_env",
    "rhs1_fp_residual_polish_controls_from_env",
    "rhs1_fp_targeted_polish_allowed",
    "rhs1_distributed_auto_solver_from_env",
    "rhs1_gpu_sparse_fallback_skip_allowed_current_backend",
    "rhs1_gmres_precondition_side_from_env",
    "rhs1_host_factor_probe_ok",
    "rhs1_krylov_routing_controls_from_env",
    "rhs1_parse_accept_ratio",
    "rhs1_parse_polish_gmres_config",
    "rhs1_pas_source_zero_tolerance_from_env",
    "rhs1_pas_fast_accept",
    "rhs1_pas_stage2_skip",
    "rhs1_pas_tz_guarded_stage2_retry",
    "rhs1_polish_enabled",
    "rhs1_prefer_sparse_over_dense_shortcut",
    "rhs1_resolved_sparse_rescue_ordering",
    "rhs1_full_sparse_rescue_setup",
    "rhs1_scipy_rescue_abs_floor_after_xblock",
    "rhs1_scipy_rescue_active_size_allowed",
    "rhs1_scipy_rescue_controls_from_env",
    "rhs1_skip_global_sparse_after_xblock_allowed",
    "rhs1_sparse_enabled_initial",
    "rhs1_sparse_exact_lu_requested",
    "rhs1_sparse_jax_config_from_env",
    "rhs1_sparse_operator_admission",
    "rhs1_sparse_rescue_initial_messages",
    "rhs1_sparse_kind_use",
    "rhs1_sparse_prefer_skips_stage2",
    "rhs1_sparse_preconditioner_config_from_env",
    "rhs1_sparse_rescue_policy_setup",
    "rhs1_sparse_rescue_tail_skip_messages",
    "rhs1_stage2_ratio",
    "rhs1_stage2_admission_controls_from_env",
    "rhs1_stage2_retry_admission_decision",
    "rhs1_stage2_retry_controls_from_env",
    "rhs1_stage2_trigger",
    "rhs1_stage2_trigger_decision",
    "rhs1_xblock_fallback_initial_guess",
)
