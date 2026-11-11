"""Publication figure panels assembled from committed evidence.

Builds the Simakov--Helander high-collisionality panel, including the provenance and deferral bookkeeping a
caption has to be able to state.

Split out of ``artifacts.py``. These builders are leaves, used by the figure
and docs-claim tests and by nothing else in the package.
"""

from __future__ import annotations
import contextlib
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field as dataclass_field
import json
import math
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any, Iterator, Protocol
import numpy as np


SIMAKOV_HELANDER_PROVENANCE_FIELDS = (
    "geometry_source",
    "scan_source",
    "analytic_limit_reference",
    "normalization_reference",
)


def build_simakov_helander_high_nu_panel(
    payload: Mapping[str, object],
    *,
    source_artifact: str | Path | None = None,
    required_provenance_fields: Sequence[str] = SIMAKOV_HELANDER_PROVENANCE_FIELDS,
    min_high_nuprime_for_literature: float = 100.0,
    min_high_nu_points: int = 3,
    min_high_nu_decades: float = 1.0,
    n_tail_fit: int = 3,
    max_asymptotic_slope: float = -0.5,
    monotonic_tolerance: float = 1.0e-12,
) -> dict[str, object]:
    """Build deterministic panel data for the deferred Simakov-Helander lane.

    The payload is intentionally small and plot-ready: each run supplies
    ``nuprime``, a computed scalar ``value``, and its ``analytic_limit``. The
    helper records asymptotic approach metadata while keeping the literature gate
    closed unless the scan reaches a configured high-``nu`` range and comes from
    a complete, Git-tracked artifact.
    """

    rows = _simakov_helander_runs(payload)
    nuprime = np.asarray([row["nuprime"] for row in rows], dtype=np.float64)
    values = np.asarray([row["value"] for row in rows], dtype=np.float64)
    analytic_limits = np.asarray([row["analytic_limit"] for row in rows], dtype=np.float64)
    relative_error = np.abs(values - analytic_limits) / np.maximum(
        np.abs(analytic_limits),
        np.finfo(float).tiny,
    )

    high_nu_mask = nuprime >= float(min_high_nuprime_for_literature)
    high_nu_count = int(np.count_nonzero(high_nu_mask))
    high_nu_min = None if high_nu_count == 0 else float(np.min(nuprime[high_nu_mask]))
    high_nu_max = None if high_nu_count == 0 else float(np.max(nuprime[high_nu_mask]))
    high_nu_decades = (
        0.0
        if high_nu_min is None or high_nu_max is None or high_nu_min <= 0.0
        else float(np.log10(high_nu_max / high_nu_min))
    )

    n_tail_fit = int(n_tail_fit)
    if n_tail_fit < 2:
        raise ValueError("n_tail_fit must be at least 2.")
    tail = _tail_asymptotic_metadata(
        nuprime=nuprime,
        relative_error=relative_error,
        n_tail_fit=n_tail_fit,
        monotonic_tolerance=float(monotonic_tolerance),
    )
    provenance = _provenance_summary(payload, required_fields=required_provenance_fields)
    artifact = _simakov_helander_artifact_summary(source_artifact, payload=payload)

    finite_scan = bool(
        nuprime.size >= 2
        and np.all(np.isfinite(nuprime))
        and np.all(nuprime > 0.0)
        and np.all(np.isfinite(values))
    )
    finite_limits = bool(
        analytic_limits.size == nuprime.size
        and np.all(np.isfinite(analytic_limits))
        and np.all(np.abs(analytic_limits) > 0.0)
    )
    finite_errors = bool(relative_error.size == nuprime.size and np.all(np.isfinite(relative_error)))
    gates = {
        "finite_positive_nuprime_scan": finite_scan,
        "finite_nonzero_analytic_limits": finite_limits,
        "finite_asymptotic_error": finite_errors,
        "enough_tail_points_for_slope": bool(tail["fit_point_count"] >= n_tail_fit),
        "asymptotic_error_decreases": bool(tail["relative_error_nonincreasing"]),
        "asymptotic_slope_matches_expected": bool(
            tail["slope"] is not None
            and np.isfinite(float(tail["slope"]))
            and float(tail["slope"]) <= float(max_asymptotic_slope)
        ),
        "high_nu_range_reaches_threshold": bool(
            float(np.max(nuprime)) >= float(min_high_nuprime_for_literature)
        ),
        "high_nu_tail_has_enough_points": bool(high_nu_count >= int(min_high_nu_points)),
        "high_nu_tail_spans_required_decades": bool(high_nu_decades >= float(min_high_nu_decades)),
        "provenance_complete": bool(provenance["complete"]),
        "source_artifact_checked_in": bool(artifact["checked_in"]),
    }
    ready = bool(all(gates.values()))
    gates["ready_for_literature_claim"] = ready
    gates["checked_in_converged_artifact"] = ready
    validation_state = "artifact_backed_literature_ready" if ready else "scaffold_deferred"
    deferred_reasons = _simakov_helander_deferred_reasons(
        gates=gates,
        provenance=provenance,
        source_artifact=artifact,
    )

    source_metadata = payload.get("metadata", {})
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "simakov_helander_high_nu_panel",
            "manuscript_lane": "simakov_helander_high_collisionality_limit",
            "validation_state": validation_state,
            "figure_label": (
                "ARTIFACT-BACKED SIMAKOV-HELANDER HIGH-NU LIMIT"
                if ready
                else "DEFERRED SCAFFOLD: SIMAKOV-HELANDER HIGH-NU LIMIT"
            ),
            "source_kind": source_metadata.get("kind"),
            "source_validation_scope": source_metadata.get("validation_scope"),
            "publication_figure": _publication_figure_claim_metadata(
                ready=ready,
                ready_artifact_class="checked_in_simakov_helander_high_nu_artifact",
                deferred_artifact_class="deferred_simakov_helander_high_nu_scaffold",
                closed_claim="checked-in Simakov-Helander high-nu analytic-limit panel",
                deferred_claim="deferred Simakov-Helander high-nu scaffold",
            ),
            "notes": [
                "This panel is an analytic-limit scaffold unless ready_for_literature_claim is true.",
                "A literature-ready label requires complete provenance, a matching Git-tracked source artifact, and a wide high-nu scan.",
            ],
            "deferred_reason_codes": [str(reason["code"]) for reason in deferred_reasons],
        },
        "scan": {
            "nuprime": [float(value) for value in nuprime.tolist()],
            "value": [float(value) for value in values.tolist()],
            "analytic_limit": [float(value) for value in analytic_limits.tolist()],
            "relative_error": [float(value) for value in relative_error.tolist()],
            "nuprime_min": float(np.min(nuprime)),
            "nuprime_max": float(np.max(nuprime)),
        },
        "asymptotic_approach": tail,
        "high_nu_range": {
            "min_high_nuprime_for_literature": float(min_high_nuprime_for_literature),
            "min_high_nu_points": int(min_high_nu_points),
            "min_high_nu_decades": float(min_high_nu_decades),
            "observed_high_nu_points": high_nu_count,
            "observed_high_nu_min": high_nu_min,
            "observed_high_nu_max": high_nu_max,
            "observed_high_nu_decades": high_nu_decades,
        },
        "provenance": provenance,
        "source_artifact": artifact,
        "gates": gates,
        "deferred_reasons": deferred_reasons,
    }


def _simakov_helander_runs(payload: Mapping[str, object]) -> list[dict[str, float]]:
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("Simakov-Helander payload must contain a 'runs' sequence.")

    runs: list[dict[str, float]] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            raise ValueError("Each Simakov-Helander run must be a mapping.")
        try:
            nuprime = float(raw["nuprime"])
            value = float(raw["value"])
            analytic_limit = float(raw["analytic_limit"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "Each Simakov-Helander run needs finite nuprime, value, and analytic_limit fields."
            ) from exc
        if (
            not np.isfinite(nuprime)
            or nuprime <= 0.0
            or not np.isfinite(value)
            or not np.isfinite(analytic_limit)
            or analytic_limit == 0.0
        ):
            raise ValueError(
                "Each Simakov-Helander run needs positive finite nuprime, finite value, and nonzero analytic_limit fields."
            )
        runs.append(
            {
                "nuprime": nuprime,
                "value": value,
                "analytic_limit": analytic_limit,
            }
        )

    if len(runs) < 2:
        raise ValueError("Need at least two Simakov-Helander runs to build a high-nu panel.")
    runs.sort(key=lambda row: row["nuprime"])
    nuprime_values = np.asarray([row["nuprime"] for row in runs], dtype=np.float64)
    if np.any(np.diff(nuprime_values) <= 0.0):
        raise ValueError("Simakov-Helander nuprime scan points must be distinct.")
    return runs


def _publication_figure_claim_metadata(
    *,
    ready: bool,
    ready_artifact_class: str,
    deferred_artifact_class: str,
    closed_claim: str,
    deferred_claim: str,
) -> dict[str, object]:
    claim_status = "checked_in_converged_artifact" if bool(ready) else "proxy_or_deferred"
    return {
        "claim_status": claim_status,
        "artifact_class": str(ready_artifact_class if ready else deferred_artifact_class),
        "checked_in_converged_artifact": bool(ready),
        "ready_for_physics_validation_claim": bool(ready),
        "manuscript_label": str(closed_claim if ready else deferred_claim),
    }


def _provenance_summary(
    payload: Mapping[str, object],
    *,
    required_fields: Sequence[str],
) -> dict[str, object]:
    raw_provenance = payload.get("provenance", {})
    if not isinstance(raw_provenance, Mapping):
        raw_provenance = {}
    fields = {str(key): raw_provenance.get(str(key), "") for key in required_fields}
    missing = [key for key, value in fields.items() if not str(value).strip()]
    total = len(fields)
    present = total - len(missing)
    return {
        "required_fields": [str(key) for key in required_fields],
        "fields": fields,
        "missing_fields": missing,
        "present_required_fields": int(present),
        "total_required_fields": int(total),
        "completeness_score": float(1.0 if total == 0 else present / total),
        "complete": bool(not missing),
    }


def _tail_asymptotic_metadata(
    *,
    nuprime: np.ndarray,
    relative_error: np.ndarray,
    n_tail_fit: int,
    monotonic_tolerance: float,
) -> dict[str, object]:
    fit_count = min(int(n_tail_fit), int(nuprime.size), int(relative_error.size))
    if fit_count < 2:
        tail_nu = np.asarray([], dtype=np.float64)
        tail_error = np.asarray([], dtype=np.float64)
        slope = None
        intercept = None
    else:
        tail_nu = nuprime[-fit_count:]
        tail_error = relative_error[-fit_count:]
        safe_error = np.maximum(tail_error, np.finfo(float).tiny)
        slope_value, intercept_value = np.polyfit(np.log(tail_nu), np.log(safe_error), 1)
        slope = float(slope_value)
        intercept = float(intercept_value)

    if tail_error.size < 2:
        nonincreasing = False
    else:
        nonincreasing = bool(np.all(np.diff(tail_error) <= float(monotonic_tolerance)))
    tail_first = None if tail_error.size == 0 else float(tail_error[0])
    tail_last = None if tail_error.size == 0 else float(tail_error[-1])
    reduction_factor = (
        None
        if tail_first is None or tail_last is None or tail_last <= 0.0
        else float(tail_first / tail_last)
    )
    return {
        "n_tail_fit": int(n_tail_fit),
        "fit_point_count": int(fit_count),
        "tail_nuprime": [float(value) for value in tail_nu.tolist()],
        "tail_relative_error": [float(value) for value in tail_error.tolist()],
        "slope": slope,
        "intercept": intercept,
        "relative_error_nonincreasing": nonincreasing,
        "tail_error_first": tail_first,
        "tail_error_last": tail_last,
        "tail_error_reduction_factor": reduction_factor,
    }


def _simakov_helander_deferred_reasons(
    *,
    gates: Mapping[str, object],
    provenance: Mapping[str, object],
    source_artifact: Mapping[str, object],
) -> list[dict[str, object]]:
    if bool(gates.get("ready_for_literature_claim")):
        return []

    reasons: list[dict[str, object]] = []
    reason_specs = [
        (
            "nonfinite_or_underresolved_nuprime_scan",
            "finite_positive_nuprime_scan",
            "The nuprime scan must contain at least two positive finite points.",
        ),
        (
            "missing_or_zero_analytic_limits",
            "finite_nonzero_analytic_limits",
            "Each scan point must include a finite nonzero analytic-limit value.",
        ),
        (
            "nonfinite_asymptotic_error",
            "finite_asymptotic_error",
            "The normalized distance to the analytic limit must be finite.",
        ),
        (
            "insufficient_tail_points_for_slope",
            "enough_tail_points_for_slope",
            "The high-collisionality tail does not contain enough points for a slope fit.",
        ),
        (
            "asymptotic_error_not_monotone",
            "asymptotic_error_decreases",
            "The tail does not approach the analytic limit monotonically.",
        ),
        (
            "asymptotic_slope_outside_expected_range",
            "asymptotic_slope_matches_expected",
            "The fitted tail slope is not negative enough for an asymptotic approach claim.",
        ),
        (
            "high_nu_threshold_not_reached",
            "high_nu_range_reaches_threshold",
            "The scan does not reach the configured high-nu threshold.",
        ),
        (
            "insufficient_high_nu_points",
            "high_nu_tail_has_enough_points",
            "The scan does not contain enough points above the high-nu threshold.",
        ),
        (
            "high_nu_span_too_narrow",
            "high_nu_tail_spans_required_decades",
            "The high-nu tail does not span the required number of decades.",
        ),
    ]
    for code, gate, message in reason_specs:
        if not bool(gates.get(gate)):
            reasons.append({"code": code, "gate": gate, "message": message})
    if not bool(gates.get("provenance_complete")):
        reasons.append(
            {
                "code": "incomplete_provenance",
                "gate": "provenance_complete",
                "missing_fields": list(provenance.get("missing_fields", [])),
                "completeness_score": float(provenance.get("completeness_score", 0.0)),
                "message": "Required Simakov-Helander provenance fields are incomplete.",
            }
        )
    if not bool(gates.get("source_artifact_checked_in")):
        reasons.append(
            {
                "code": "source_artifact_not_checked_in",
                "gate": "source_artifact_checked_in",
                "artifact_status": source_artifact.get("status", "unknown"),
                "message": "The source JSON is not a matching Git-tracked Simakov-Helander artifact.",
            }
        )
    return reasons


def _simakov_helander_artifact_summary(
    source_artifact: str | Path | None,
    *,
    payload: Mapping[str, object],
) -> dict[str, object]:
    if source_artifact is None:
        return {
            "path": None,
            "exists": False,
            "tracked": False,
            "looks_like_simakov_helander_artifact": False,
            "payload_matches": False,
            "checked_in": False,
            "status": "missing",
        }
    path = Path(source_artifact).expanduser()
    exists = path.is_file()
    tracked = bool(exists and _is_git_tracked_file(path))
    lower_name = path.name.lower()
    looks_like_simakov_helander_artifact = bool(
        path.suffix.lower() == ".json"
        and "simakov" in lower_name
        and "helander" in lower_name
    )
    payload_matches = bool(
        exists
        and looks_like_simakov_helander_artifact
        and _json_payload_matches(path=path, payload=payload)
    )
    checked_in = bool(tracked and looks_like_simakov_helander_artifact and payload_matches)
    if checked_in:
        status = "checked_in"
    elif tracked and looks_like_simakov_helander_artifact and not payload_matches:
        status = "tracked_simakov_helander_artifact_payload_mismatch"
    elif tracked:
        status = "tracked_non_simakov_helander_artifact"
    elif exists:
        status = "untracked_or_outside_git"
    else:
        status = "missing"
    return {
        "path": str(path),
        "exists": bool(exists),
        "tracked": tracked,
        "looks_like_simakov_helander_artifact": looks_like_simakov_helander_artifact,
        "payload_matches": payload_matches,
        "checked_in": checked_in,
        "status": status,
    }


def _json_payload_matches(*, path: Path, payload: Mapping[str, object]) -> bool:
    try:
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        normalized_payload = json.loads(json.dumps(payload, sort_keys=True))
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(on_disk == normalized_payload)


def _is_git_tracked_file(path: Path) -> bool:
    resolved = path.resolve()
    git_root = _find_git_root(resolved)
    if git_root is None:
        return False
    try:
        relative = resolved.relative_to(git_root)
    except ValueError:
        return False
    try:
        result = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", str(relative)],
            cwd=git_root,
            capture_output=True,
            check=False,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(result.returncode == 0)


def _find_git_root(path: Path) -> Path | None:
    start = path if path.is_dir() else path.parent
    for parent in (start, *start.parents):
        if (parent / ".git").exists():
            return parent
    return None


