from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import subprocess

import numpy as np


W7X_AMBIPOLAR_PROVENANCE_FIELDS = (
    "equilibrium_source",
    "profile_source",
    "configuration_or_shot",
    "literature_reference",
)


def build_w7x_ambipolar_root_provenance_panel(
    payload: Mapping[str, object],
    *,
    source_artifact: str | Path | None = None,
    required_provenance_fields: Sequence[str] = W7X_AMBIPOLAR_PROVENANCE_FIELDS,
    root_er_tolerance: float = 1.0e-9,
    current_zero_tolerance: float = 1.0e-12,
    min_abs_current_slope: float = 1.0e-12,
) -> dict[str, object]:
    """Build deterministic panel data for the deferred W7-X ambipolar-root lane.

    The returned payload is intentionally conservative: it is labelled
    ``scaffold_deferred`` unless the summary has complete provenance and
    ``source_artifact`` points at a matching Git-tracked W7-X ambipolar JSON
    artifact. This prevents synthetic scans and ad-hoc reruns from being mistaken
    for a closed literature claim.
    """

    runs = _scan_runs(payload)
    er = np.asarray([row["er"] for row in runs], dtype=np.float64)
    radial_current = np.asarray([row["radial_current"] for row in runs], dtype=np.float64)
    brackets = _zero_crossing_brackets(
        er=er,
        radial_current=radial_current,
        current_zero_tolerance=float(current_zero_tolerance),
    )
    roots = _roots(payload)
    root_types = _root_types(payload, roots.size)
    root_rows = _root_rows(
        roots=roots,
        root_types=root_types,
        er_min=float(np.min(er)),
        er_max=float(np.max(er)),
        brackets=brackets,
        root_er_tolerance=float(root_er_tolerance),
    )
    provenance = _provenance_summary(payload, required_fields=required_provenance_fields)
    artifact = _artifact_summary(source_artifact, payload=payload)

    finite_series = bool(er.size >= 2 and np.all(np.isfinite(er)) and np.all(np.isfinite(radial_current)))
    finite_roots = bool(roots.size > 0 and np.all(np.isfinite(roots)))
    root_inside_scan_range = bool(
        finite_roots
        and all(bool(row["inside_scanned_er_range"]) for row in root_rows)
    )
    root_consistent = bool(
        finite_roots
        and bool(brackets)
        and all(row["matching_bracket_index"] is not None for row in root_rows)
    )
    resolved_slopes = [
        row["local_radial_current_slope"]
        for row in root_rows
        if row["matching_bracket_index"] is not None
    ]
    slope_resolved = bool(
        resolved_slopes
        and all(
            value is not None
            and np.isfinite(float(value))
            and abs(float(value)) >= float(min_abs_current_slope)
            for value in resolved_slopes
        )
    )
    ion_root_candidate = bool(
        any("ion" in str(root_type).lower() for root_type in root_types)
        or np.any(roots < 0.0)
    )
    gates = {
        "finite_er_current_series": finite_series,
        "radial_current_brackets_zero": bool(brackets),
        "finite_ambipolar_roots": finite_roots,
        "root_inside_scanned_er_range": root_inside_scan_range,
        "root_consistent_with_sign_change": root_consistent,
        "ambipolar_root_slope_resolved": slope_resolved,
        "ion_root_candidate": ion_root_candidate,
        "current_trend_monotonic": _is_monotonic(radial_current),
        "provenance_complete": bool(provenance["complete"]),
        "source_artifact_checked_in": bool(artifact["checked_in"]),
    }
    ready = bool(
        gates["finite_er_current_series"]
        and gates["radial_current_brackets_zero"]
        and gates["finite_ambipolar_roots"]
        and gates["root_inside_scanned_er_range"]
        and gates["root_consistent_with_sign_change"]
        and gates["ambipolar_root_slope_resolved"]
        and gates["ion_root_candidate"]
        and gates["provenance_complete"]
        and gates["source_artifact_checked_in"]
    )
    gates["ready_for_literature_claim"] = ready
    validation_state = "artifact_backed_literature_ready" if ready else "scaffold_deferred"

    source_metadata = payload.get("metadata", {})
    if not isinstance(source_metadata, Mapping):
        source_metadata = {}
    return {
        "metadata": {
            "schema_version": 1,
            "kind": "w7x_ambipolar_root_provenance_panel",
            "manuscript_lane": "w7x_ambipolar_er_validation",
            "validation_state": validation_state,
            "figure_label": (
                "ARTIFACT-BACKED W7-X AMBIPOLAR ROOT"
                if ready
                else "DEFERRED SCAFFOLD: W7-X AMBIPOLAR ROOT"
            ),
            "source_kind": source_metadata.get("kind"),
            "source_validation_scope": source_metadata.get("validation_scope"),
            "notes": [
                "This panel is a provenance and numerical-gate scaffold unless ready_for_literature_claim is true.",
                "A literature-ready label requires complete W7-X provenance and a matching Git-tracked W7-X ambipolar JSON artifact.",
            ],
        },
        "scan": {
            "er": [float(value) for value in er.tolist()],
            "radial_current": [float(value) for value in radial_current.tolist()],
            "radial_current_min": float(np.min(radial_current)),
            "radial_current_max": float(np.max(radial_current)),
        },
        "zero_crossing_brackets": brackets,
        "roots": root_rows,
        "provenance": provenance,
        "source_artifact": artifact,
        "gates": gates,
    }


def _scan_runs(payload: Mapping[str, object]) -> list[dict[str, float]]:
    raw_runs = payload.get("runs")
    if not isinstance(raw_runs, Sequence) or isinstance(raw_runs, (str, bytes)):
        raise ValueError("W7-X ambipolar payload must contain a 'runs' sequence.")

    runs: list[dict[str, float]] = []
    for raw in raw_runs:
        if not isinstance(raw, Mapping):
            raise ValueError("Each W7-X ambipolar run must be a mapping.")
        try:
            er = float(raw["er"])
            radial_current = float(raw["radial_current"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Each W7-X ambipolar run needs finite er and radial_current fields.") from exc
        if not np.isfinite(er) or not np.isfinite(radial_current):
            raise ValueError("Each W7-X ambipolar run needs finite er and radial_current fields.")
        runs.append({"er": er, "radial_current": radial_current})

    if len(runs) < 2:
        raise ValueError("Need at least two W7-X ambipolar runs to build a root provenance panel.")
    runs.sort(key=lambda row: row["er"])
    er_values = np.asarray([row["er"] for row in runs], dtype=np.float64)
    if np.any(np.diff(er_values) <= 0.0):
        raise ValueError("W7-X ambipolar Er scan points must be distinct.")
    return runs


def _roots(payload: Mapping[str, object]) -> np.ndarray:
    ambipolar = payload.get("ambipolar")
    if not isinstance(ambipolar, Mapping):
        raise ValueError("W7-X ambipolar payload must contain an 'ambipolar' mapping.")
    raw_roots = ambipolar.get("roots_er", [])
    if not isinstance(raw_roots, Sequence) or isinstance(raw_roots, (str, bytes)):
        raise ValueError("ambipolar.roots_er must be a sequence.")
    try:
        roots = np.asarray([float(value) for value in raw_roots], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("ambipolar.roots_er must contain numeric values.") from exc
    return roots


def _root_types(payload: Mapping[str, object], n_roots: int) -> list[str]:
    ambipolar = payload.get("ambipolar")
    if not isinstance(ambipolar, Mapping):
        return ["unknown"] * int(n_roots)
    raw_types = ambipolar.get("root_types", [])
    if not isinstance(raw_types, Sequence) or isinstance(raw_types, (str, bytes)):
        return ["unknown"] * int(n_roots)
    root_types = [str(value) for value in raw_types]
    if len(root_types) < int(n_roots):
        root_types.extend(["unknown"] * (int(n_roots) - len(root_types)))
    return root_types[: int(n_roots)]


def _zero_crossing_brackets(
    *,
    er: np.ndarray,
    radial_current: np.ndarray,
    current_zero_tolerance: float,
) -> list[dict[str, float | int | None]]:
    brackets: list[dict[str, float | int | None]] = []
    for idx in range(er.size - 1):
        er_left = float(er[idx])
        er_right = float(er[idx + 1])
        current_left = float(radial_current[idx])
        current_right = float(radial_current[idx + 1])
        if not _crosses_zero(current_left, current_right, tolerance=float(current_zero_tolerance)):
            continue
        delta_current = current_right - current_left
        slope = delta_current / (er_right - er_left)
        if abs(delta_current) <= float(current_zero_tolerance):
            linear_root = None
        else:
            linear_root = er_left - current_left * (er_right - er_left) / delta_current
        brackets.append(
            {
                "index": int(idx),
                "er_min": float(min(er_left, er_right)),
                "er_max": float(max(er_left, er_right)),
                "current_left": current_left,
                "current_right": current_right,
                "linear_interpolated_root_er": None if linear_root is None else float(linear_root),
                "local_radial_current_slope": float(slope),
            }
        )
    return brackets


def _crosses_zero(left: float, right: float, *, tolerance: float) -> bool:
    if abs(float(left)) <= float(tolerance) or abs(float(right)) <= float(tolerance):
        return True
    return bool((left < 0.0 < right) or (right < 0.0 < left))


def _root_rows(
    *,
    roots: np.ndarray,
    root_types: Sequence[str],
    er_min: float,
    er_max: float,
    brackets: Sequence[Mapping[str, object]],
    root_er_tolerance: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx, root in enumerate(roots):
        root_value = float(root)
        matching = _matching_bracket(root_value, brackets=brackets, tolerance=float(root_er_tolerance))
        linear_root = None if matching is None else matching.get("linear_interpolated_root_er")
        root_to_linear_delta = (
            None
            if linear_root is None
            else float(root_value - float(linear_root))
        )
        root_type = str(root_types[idx]) if idx < len(root_types) else "unknown"
        rows.append(
            {
                "root_index": int(idx),
                "er": root_value,
                "root_type": root_type,
                "is_finite": bool(np.isfinite(root_value)),
                "inside_scanned_er_range": bool(
                    np.isfinite(root_value)
                    and er_min - float(root_er_tolerance) <= root_value <= er_max + float(root_er_tolerance)
                ),
                "ion_root_candidate": bool("ion" in root_type.lower() or root_value < 0.0),
                "matching_bracket_index": None if matching is None else int(matching["index"]),
                "bracket_er_min": None if matching is None else float(matching["er_min"]),
                "bracket_er_max": None if matching is None else float(matching["er_max"]),
                "linear_interpolated_root_er": None if linear_root is None else float(linear_root),
                "root_to_linear_delta": root_to_linear_delta,
                "local_radial_current_slope": (
                    None
                    if matching is None
                    else float(matching["local_radial_current_slope"])
                ),
            }
        )
    return rows


def _matching_bracket(
    root: float,
    *,
    brackets: Sequence[Mapping[str, object]],
    tolerance: float,
) -> Mapping[str, object] | None:
    matches = [
        bracket
        for bracket in brackets
        if float(bracket["er_min"]) - float(tolerance)
        <= float(root)
        <= float(bracket["er_max"]) + float(tolerance)
    ]
    if not matches:
        return None

    def _distance(bracket: Mapping[str, object]) -> float:
        linear_root = bracket.get("linear_interpolated_root_er")
        if linear_root is not None:
            return abs(float(root) - float(linear_root))
        center = 0.5 * (float(bracket["er_min"]) + float(bracket["er_max"]))
        return abs(float(root) - center)

    return min(matches, key=_distance)


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
    return {
        "required_fields": [str(key) for key in required_fields],
        "fields": fields,
        "missing_fields": missing,
        "complete": bool(not missing),
    }


def _artifact_summary(source_artifact: str | Path | None, *, payload: Mapping[str, object]) -> dict[str, object]:
    if source_artifact is None:
        return {
            "path": None,
            "exists": False,
            "tracked": False,
            "looks_like_w7x_ambipolar_artifact": False,
            "payload_matches": False,
            "checked_in": False,
            "status": "missing",
        }
    path = Path(source_artifact).expanduser()
    exists = path.is_file()
    tracked = bool(exists and _is_git_tracked_file(path))
    looks_like_w7x_ambipolar_artifact = bool(
        path.suffix.lower() == ".json"
        and "w7x" in path.name.lower()
        and "ambipolar" in path.name.lower()
    )
    payload_matches = bool(
        exists
        and looks_like_w7x_ambipolar_artifact
        and _json_payload_matches(path=path, payload=payload)
    )
    checked_in = bool(tracked and looks_like_w7x_ambipolar_artifact and payload_matches)
    if checked_in:
        status = "checked_in"
    elif tracked and looks_like_w7x_ambipolar_artifact and not payload_matches:
        status = "tracked_w7x_ambipolar_artifact_payload_mismatch"
    elif tracked:
        status = "tracked_non_w7x_ambipolar_artifact"
    elif exists:
        status = "untracked_or_outside_git"
    else:
        status = "missing"
    return {
        "path": str(path),
        "exists": bool(exists),
        "tracked": tracked,
        "looks_like_w7x_ambipolar_artifact": looks_like_w7x_ambipolar_artifact,
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


def _is_monotonic(values: np.ndarray) -> bool:
    deltas = np.diff(np.asarray(values, dtype=np.float64))
    return bool(np.all(deltas >= 0.0) or np.all(deltas <= 0.0))
