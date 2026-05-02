"""Diagnostics for constrained-PAS nullspace branch sensitivity.

RHSMode=1 PAS systems with ``constraintScheme=2`` can be singular or nearly
singular in the flow/current gauge.  Exact residual solves, PETSc-compatible
minimum-norm solves, and preconditioned-residual Fortran references can then
land on different diagnostic branches even when the kinetic operator itself is
the same.  These helpers keep that state explicit in tests, reports, and future
solver-policy experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class ConstrainedPASBranchRecord:
    """One solver/reference branch for a constrained-PAS diagnostic."""

    label: str
    observable: float
    residual_norm: float | None = None
    residual_target: float | None = None
    criterion: str = "unknown"
    accepted: bool = True

    @property
    def residual_ratio(self) -> float:
        """Return ``residual_norm / residual_target`` when both are meaningful."""
        if self.residual_norm is None or self.residual_target is None:
            return math.inf
        target = float(self.residual_target)
        if not math.isfinite(target) or target <= 0.0:
            return math.inf
        residual = float(self.residual_norm)
        if not math.isfinite(residual):
            return math.inf
        return residual / target

    def true_residual_converged(self, *, slack: float = 1.0) -> bool:
        """Return whether the branch satisfies its true-residual target."""
        return self.residual_ratio <= float(slack)


@dataclass(frozen=True)
class ConstrainedPASBranchSummary:
    """Compact classification of branch spread and reference quality."""

    reference_label: str | None
    branch_sensitive: bool
    max_relative_spread: float
    weak_reference_labels: tuple[str, ...]
    recommendation: str

    @property
    def has_reference_quality_blocker(self) -> bool:
        """Return whether a weak true-residual reference affects the comparison."""
        return bool(self.weak_reference_labels)


def summarize_constrained_pas_branches(
    records: Iterable[ConstrainedPASBranchRecord],
    *,
    residual_slack: float = 1.0,
    weak_residual_ratio: float = 10.0,
    branch_relative_gate: float = 1.0e-3,
) -> ConstrainedPASBranchSummary:
    """Classify constrained-PAS branch spread from solver/reference records.

    The reference branch is the accepted record with the smallest true-residual
    ratio.  Weak references are records whose true-residual ratio exceeds
    ``weak_residual_ratio``.  A case is branch-sensitive when the observable
    spread exceeds ``branch_relative_gate`` relative to the selected reference
    observable.  The function is intentionally physics-agnostic: callers decide
    whether the observable is ``FSABjHat``, ``FSABjHatOverRootFSAB2``, or another
    flow/current diagnostic.
    """
    rows = tuple(records)
    if not rows:
        return ConstrainedPASBranchSummary(
            reference_label=None,
            branch_sensitive=False,
            max_relative_spread=0.0,
            weak_reference_labels=(),
            recommendation="no_branch_records",
        )

    accepted_rows = tuple(row for row in rows if row.accepted)
    converged_rows = tuple(row for row in accepted_rows if row.true_residual_converged(slack=residual_slack))
    candidate_rows = converged_rows or accepted_rows or rows
    reference = min(candidate_rows, key=lambda row: row.residual_ratio)

    scale = max(abs(float(reference.observable)), 1.0e-300)
    spreads = [abs(float(row.observable) - float(reference.observable)) / scale for row in rows]
    max_spread = max(spreads, default=0.0)
    weak = tuple(row.label for row in rows if row.residual_ratio > float(weak_residual_ratio))
    branch_sensitive = max_spread > float(branch_relative_gate)

    if not converged_rows:
        recommendation = "needs_true_residual_reference"
    elif branch_sensitive and weak:
        recommendation = "pin_gauge_before_parity_claim"
    elif branch_sensitive:
        recommendation = "branch_sensitive_even_with_converged_records"
    else:
        recommendation = "converged_branch_consistent"

    return ConstrainedPASBranchSummary(
        reference_label=reference.label,
        branch_sensitive=bool(branch_sensitive),
        max_relative_spread=float(max_spread),
        weak_reference_labels=weak,
        recommendation=recommendation,
    )


__all__ = [
    "ConstrainedPASBranchRecord",
    "ConstrainedPASBranchSummary",
    "summarize_constrained_pas_branches",
]
