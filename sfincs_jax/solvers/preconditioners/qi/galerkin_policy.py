"""Policy helpers for opt-in RHSMode=1 QI Galerkin preconditioners.

The production driver should not keep using an experimental coarse
preconditioner when its own cheap probe shows a worse true residual.  These
helpers keep that decision pure and directly testable outside ``v3_driver.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


VALID_QI_GALERKIN_MODES = ("additive", "multiplicative")


@dataclass(frozen=True)
class RHS1QIGalerkinProbeCandidate:
    """Measured residual for one Galerkin preconditioner candidate."""

    mode: str
    damping: float
    residual_norm: float
    improvement_ratio: float | None
    reduced: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "damping": float(self.damping),
            "residual_norm": float(self.residual_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "reduced": bool(self.reduced),
        }


@dataclass(frozen=True)
class RHS1QIGalerkinProbeSelection:
    """Fail-closed selection result for candidate Galerkin preconditioners."""

    accepted: bool
    reason: str
    selected_index: int | None
    selected_mode: str | None
    selected_damping: float | None
    residual_before_norm: float
    residual_after_norm: float | None
    improvement_ratio: float | None
    candidates: tuple[RHS1QIGalerkinProbeCandidate, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "accepted": bool(self.accepted),
            "reason": self.reason,
            "selected_index": self.selected_index,
            "selected_mode": self.selected_mode,
            "selected_damping": None if self.selected_damping is None else float(self.selected_damping),
            "residual_before_norm": float(self.residual_before_norm),
            "residual_after_norm": None if self.residual_after_norm is None else float(self.residual_after_norm),
            "improvement_ratio": None if self.improvement_ratio is None else float(self.improvement_ratio),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def parse_rhs1_qi_galerkin_modes(raw: str | None, *, default: str = "auto") -> tuple[str, ...]:
    """Parse a mode list for the QI Galerkin probe.

    ``auto`` tries both supported compositions.  Explicit lists such as
    ``additive,multiplicative`` are accepted so bounded GPU campaigns can use
    one environment variable instead of rerunning the whole process per mode.
    Invalid tokens are ignored; if nothing valid remains the result is
    ``("additive",)``.
    """

    text = (default if raw is None or not str(raw).strip() else str(raw)).strip().lower().replace("-", "_")
    tokens = [token.strip() for token in text.split(",") if token.strip()]
    if not tokens or "auto" in tokens:
        return VALID_QI_GALERKIN_MODES
    modes: list[str] = []
    for token in tokens:
        if token in VALID_QI_GALERKIN_MODES and token not in modes:
            modes.append(token)
    return tuple(modes) if modes else ("additive",)


def parse_rhs1_qi_galerkin_dampings(
    raw: str | None,
    *,
    default: float = 1.0,
    auto_defaults: Sequence[float] = (1.0, 0.5, 0.25),
) -> tuple[float, ...]:
    """Parse positive damping candidates for the QI Galerkin probe."""

    if raw is None or not str(raw).strip():
        values = tuple(float(value) for value in auto_defaults) if float(default) == 1.0 else (float(default),)
    else:
        parsed: list[float] = []
        for token in str(raw).replace(";", ",").split(","):
            if not token.strip():
                continue
            try:
                parsed.append(float(token))
            except ValueError:
                continue
        values = tuple(parsed)
    cleaned: list[float] = []
    for value in values:
        if np.isfinite(value) and value >= 0.0 and value not in cleaned:
            cleaned.append(float(value))
    return tuple(cleaned) if cleaned else (max(0.0, float(default)),)


def select_rhs1_qi_galerkin_probe_candidate(
    residual_before_norm: float,
    candidates: Sequence[dict[str, Any] | RHS1QIGalerkinProbeCandidate],
    *,
    min_relative_improvement: float = 0.0,
    acceptance_atol: float = 0.0,
) -> RHS1QIGalerkinProbeSelection:
    """Select the best candidate only if it reduces the true residual.

    The gate is intentionally conservative.  Non-finite residuals are recorded
    but cannot be selected; a candidate must beat ``residual_before_norm`` by at
    least the requested relative or absolute margin.
    """

    before = float(residual_before_norm)
    records: list[RHS1QIGalerkinProbeCandidate] = []
    for candidate in candidates:
        if isinstance(candidate, RHS1QIGalerkinProbeCandidate):
            record = candidate
        else:
            residual = float(candidate.get("residual_norm", np.inf))
            ratio = residual / before if before > 0.0 and np.isfinite(residual) else None
            record = RHS1QIGalerkinProbeCandidate(
                mode=str(candidate.get("mode", "additive")),
                damping=float(candidate.get("damping", 1.0)),
                residual_norm=residual,
                improvement_ratio=ratio,
                reduced=bool(np.isfinite(residual) and residual < before),
            )
        records.append(record)

    if not records:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=(),
        )

    finite = [(idx, record) for idx, record in enumerate(records) if np.isfinite(float(record.residual_norm))]
    if not finite:
        return RHS1QIGalerkinProbeSelection(
            accepted=False,
            reason="no_finite_probe_candidates",
            selected_index=None,
            selected_mode=None,
            selected_damping=None,
            residual_before_norm=before,
            residual_after_norm=None,
            improvement_ratio=None,
            candidates=tuple(records),
        )

    best_index, best = min(finite, key=lambda item: float(item[1].residual_norm))
    required_drop = max(float(acceptance_atol), before * max(0.0, float(min_relative_improvement)))
    accepted = bool(float(best.residual_norm) < before - required_drop)
    return RHS1QIGalerkinProbeSelection(
        accepted=accepted,
        reason="probe_reduced" if accepted else "probe_not_reduced",
        selected_index=int(best_index) if accepted else None,
        selected_mode=best.mode if accepted else None,
        selected_damping=float(best.damping) if accepted else None,
        residual_before_norm=before,
        residual_after_norm=float(best.residual_norm),
        improvement_ratio=best.improvement_ratio,
        candidates=tuple(records),
    )


__all__ = [
    "RHS1QIGalerkinProbeCandidate",
    "RHS1QIGalerkinProbeSelection",
    "VALID_QI_GALERKIN_MODES",
    "parse_rhs1_qi_galerkin_dampings",
    "parse_rhs1_qi_galerkin_modes",
    "select_rhs1_qi_galerkin_probe_candidate",
]
