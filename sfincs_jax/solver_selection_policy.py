"""Measured solver-candidate gates for automatic route selection.

The routines in this module are intentionally independent of SFINCS matrix
assembly. They encode the common acceptance rules that should guard future
automatic solver-path promotions: a candidate must be finite, parity-clean,
residual-clean, and either faster or lower-memory than the incumbent unless the
incumbent failed to converge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np


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


def _passes_residual(candidate: SolverCandidateMetrics, criteria: SolverAcceptanceCriteria) -> bool:
    ratio = candidate.residual_ratio
    return ratio is not None and ratio <= float(criteria.max_residual_ratio)


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
    baseline_clean = False
    if baseline is not None:
        baseline_clean = baseline.finite and _passes_residual(baseline, criteria)
        cand_time = candidate.total_s
        base_time = baseline.total_s
        if cand_time is not None and base_time is not None and base_time > 0.0:
            runtime_ratio = cand_time / base_time
        cand_mem = _finite_or_none(candidate.peak_rss_mb)
        base_mem = _finite_or_none(baseline.peak_rss_mb)
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
            if candidate.peak_rss_mb is None and not criteria.allow_unknown_memory_when_baseline_failed:
                reasons.append("missing_memory")

    return SolverCandidateGate(
        accepted=not reasons,
        reasons=tuple(reasons),
        residual_ratio=residual_ratio,
        runtime_ratio=runtime_ratio,
        memory_ratio=memory_ratio,
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
        peak_rss_mb = _finite_or_none(candidate.peak_rss_mb)
        accepted.append(
            (
                float(total_s) if total_s is not None else float("inf"),
                float(peak_rss_mb) if peak_rss_mb is not None else float("inf"),
                candidate,
            )
        )
    if not accepted:
        return None
    accepted.sort(key=lambda item: (item[0], item[1], item[2].name))
    return accepted[0][2]


__all__ = [
    "SolverAcceptanceCriteria",
    "SolverCandidateGate",
    "SolverCandidateMetrics",
    "choose_solver_candidate",
    "solver_candidate_gate",
]
