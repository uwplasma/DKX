"""Pure solver/preconditioner path policy helpers for the v3 driver.

The v3 driver keeps runtime state such as the active JAX backend and cached
operator-size hints.  This module holds the small policy decisions underneath
that state so they can be tested without importing the full driver.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import math
import os

import numpy as np


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


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    return value_f if np.isfinite(value_f) else None


@dataclass(frozen=True)
class SolverCandidateMetrics:
    """Measured diagnostics for one solver/preconditioner candidate."""

    name: str
    residual_norm: float | None = None
    target: float | None = None
    setup_s: float | None = None
    solve_s: float | None = None
    peak_rss_mb: float | None = None
    active_rss_mb: float | None = None
    device_peak_mb: float | None = None
    compiled_temp_mb: float | None = None
    finite: bool = True
    parity_failures: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict, compare=False)

    @property
    def total_s(self) -> float | None:
        """Return measured setup+solve time when either part is available."""

        setup = _finite_or_none(self.setup_s)
        solve = _finite_or_none(self.solve_s)
        if setup is None and solve is None:
            return None
        return float(setup or 0.0) + float(solve or 0.0)

    @property
    def residual_ratio(self) -> float | None:
        """Return ``||r|| / target`` when both values are meaningful."""

        residual = _finite_or_none(self.residual_norm)
        target = _finite_or_none(self.target)
        if residual is None or target is None or target <= 0.0:
            return None
        return residual / max(target, 1.0e-300)


@dataclass(frozen=True)
class SolverAcceptanceCriteria:
    """Numerical and performance bounds for accepting a candidate path."""

    max_residual_ratio: float = 1.0
    max_runtime_factor_vs_baseline: float = 1.10
    max_memory_factor_vs_baseline: float = 1.05
    min_runtime_speedup_for_promotion: float = 1.05
    min_memory_reduction_for_promotion: float = 1.05
    allow_unknown_runtime_when_baseline_failed: bool = True
    allow_unknown_memory_when_baseline_failed: bool = True


@dataclass(frozen=True)
class SolverCandidateGate:
    """Decision and diagnostics for one candidate acceptance check."""

    accepted: bool
    reasons: tuple[str, ...]
    residual_ratio: float | None = None
    runtime_ratio: float | None = None
    memory_ratio: float | None = None
    memory_metric: str | None = None


def _passes_residual(candidate: SolverCandidateMetrics, criteria: SolverAcceptanceCriteria) -> bool:
    ratio = candidate.residual_ratio
    return ratio is not None and ratio <= float(criteria.max_residual_ratio)


def _paired_memory_values(
    candidate: SolverCandidateMetrics,
    baseline: SolverCandidateMetrics,
) -> tuple[float | None, float | None, str | None]:
    """Return comparable memory values and the metric name.

    Candidate gates should not compare device memory against process RSS. Prefer
    the most specific paired metric available, then fall back to legacy peak RSS.
    """

    for attr in ("device_peak_mb", "active_rss_mb", "compiled_temp_mb", "peak_rss_mb"):
        cand = _finite_or_none(getattr(candidate, attr))
        base = _finite_or_none(getattr(baseline, attr))
        if cand is not None and base is not None:
            return cand, base, attr
    return None, None, None


def _single_memory_value(candidate: SolverCandidateMetrics) -> float | None:
    """Return the best available memory metric for tie-breaking."""

    for attr in ("device_peak_mb", "active_rss_mb", "compiled_temp_mb", "peak_rss_mb"):
        value = _finite_or_none(getattr(candidate, attr))
        if value is not None:
            return value
    return None


def solver_candidate_gate(
    candidate: SolverCandidateMetrics,
    *,
    baseline: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> SolverCandidateGate:
    """Return whether ``candidate`` is safe to auto-select.

    If the baseline is already residual-clean, a new candidate must be
    residual-clean and must provide a measured runtime or memory win. If the
    baseline failed, a residual-clean candidate is accepted even if it is slower,
    because correctness takes priority over performance.
    """

    criteria = criteria or SolverAcceptanceCriteria()
    reasons: list[str] = []

    if not candidate.finite:
        reasons.append("nonfinite_candidate")
    if candidate.parity_failures is not None and int(candidate.parity_failures) > 0:
        reasons.append("parity_failures")

    residual_ratio = candidate.residual_ratio
    if not _passes_residual(candidate, criteria):
        reasons.append("residual_not_clean")

    runtime_ratio: float | None = None
    memory_ratio: float | None = None
    memory_metric: str | None = None
    if baseline is not None:
        baseline_clean = baseline.finite and _passes_residual(baseline, criteria)
        cand_time = candidate.total_s
        base_time = baseline.total_s
        if cand_time is not None and base_time is not None and base_time > 0.0:
            runtime_ratio = cand_time / base_time
        cand_mem, base_mem, memory_metric = _paired_memory_values(candidate, baseline)
        if cand_mem is not None and base_mem is not None and base_mem > 0.0:
            memory_ratio = cand_mem / base_mem

        if baseline_clean:
            faster = runtime_ratio is not None and runtime_ratio <= 1.0 / float(
                criteria.min_runtime_speedup_for_promotion
            )
            lower_memory = memory_ratio is not None and memory_ratio <= 1.0 / float(
                criteria.min_memory_reduction_for_promotion
            )
            if not (faster or lower_memory):
                reasons.append("no_measured_promotion_win")
            if runtime_ratio is not None and runtime_ratio > float(criteria.max_runtime_factor_vs_baseline):
                reasons.append("runtime_regression")
            if memory_ratio is not None and memory_ratio > float(criteria.max_memory_factor_vs_baseline):
                reasons.append("memory_regression")
        else:
            if candidate.total_s is None and not criteria.allow_unknown_runtime_when_baseline_failed:
                reasons.append("missing_runtime")
            if _single_memory_value(candidate) is None and not criteria.allow_unknown_memory_when_baseline_failed:
                reasons.append("missing_memory")

    return SolverCandidateGate(
        accepted=not reasons,
        reasons=tuple(reasons),
        residual_ratio=residual_ratio,
        runtime_ratio=runtime_ratio,
        memory_ratio=memory_ratio,
        memory_metric=memory_metric,
    )


def choose_solver_candidate(
    candidates: list[SolverCandidateMetrics],
    *,
    baseline: SolverCandidateMetrics | None = None,
    criteria: SolverAcceptanceCriteria | None = None,
) -> SolverCandidateMetrics | None:
    """Choose the fastest accepted candidate, breaking ties by lower memory."""

    accepted: list[tuple[float, float, SolverCandidateMetrics]] = []
    for candidate in candidates:
        gate = solver_candidate_gate(candidate, baseline=baseline, criteria=criteria)
        if not gate.accepted:
            continue
        total_s = candidate.total_s
        memory_mb = _single_memory_value(candidate)
        accepted.append(
            (
                float(total_s) if total_s is not None else float("inf"),
                float(memory_mb) if memory_mb is not None else float("inf"),
                candidate,
            )
        )
    if not accepted:
        return None
    accepted.sort(key=lambda item: (item[0], item[1], item[2].name))
    return accepted[0][2]


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
