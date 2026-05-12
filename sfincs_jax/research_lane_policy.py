"""CI-fast policy checks for evidence-backed research-lane completion."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
import json
import math


VALID_LANE_STATUSES = frozenset({"active", "evidence_ready", "closed", "deferred"})
DEFAULT_MIN_SUBSTANTIAL_DELTA_PERCENT = 10.0


class ResearchLanePolicyError(ValueError):
    """Raised when the research-lane completion manifest is inconsistent."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def research_lane_completion_errors(
    payload: object,
    *,
    source: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Return schema and consistency errors for a research-lane manifest.

    The manifest is a lightweight release-management artifact: it records the
    open research/performance lanes, the evidence backing each completion
    estimate, and whether a lane made a substantial jump during the current
    push.  The check is intentionally CI-fast and does not launch simulations.
    """

    prefix = f"{source}: " if source is not None else ""
    root = Path(repo_root) if repo_root is not None else _default_repo_root()
    errors: list[str] = []

    def add(message: str) -> None:
        errors.append(prefix + message)

    if not isinstance(payload, Mapping):
        add("manifest must be a JSON object")
        return errors

    schema_version = _as_number(payload.get("schema_version"))
    if schema_version is None or schema_version < 1:
        add("field schema_version must be a number >= 1")

    min_delta = _as_number(
        payload.get("minimum_substantial_delta_percent", DEFAULT_MIN_SUBSTANTIAL_DELTA_PERCENT)
    )
    if min_delta is None or min_delta < 0:
        add("field minimum_substantial_delta_percent must be a non-negative number")
        min_delta = DEFAULT_MIN_SUBSTANTIAL_DELTA_PERCENT

    lanes = payload.get("lanes")
    if not isinstance(lanes, Sequence) or isinstance(lanes, (str, bytes)):
        add("field lanes must be a non-empty list")
        return errors
    if not lanes:
        add("field lanes must be a non-empty list")
        return errors

    seen_ids: set[str] = set()
    substantial_count = 0
    for index, lane in enumerate(lanes):
        if not isinstance(lane, Mapping):
            add(f"lanes[{index}] must be a JSON object")
            continue
        lane_id = _nonempty_string(lane.get("id"))
        label = lane_id or f"lanes[{index}]"
        if lane_id is None:
            add(f"lanes[{index}].id must be a non-empty string")
        elif lane_id in seen_ids:
            add(f"duplicate lane id {lane_id!r}")
        else:
            seen_ids.add(lane_id)

        title = _nonempty_string(lane.get("title"))
        if title is None:
            add(f"{label}: field title must be a non-empty string")

        status = _nonempty_string(lane.get("status"))
        if status not in VALID_LANE_STATUSES:
            add(f"{label}: field status must be one of {sorted(VALID_LANE_STATUSES)}")

        before = _percent_field(lane, "before_percent", label, add)
        current = _percent_field(lane, "current_percent", label, add)
        target = _percent_field(lane, "target_percent", label, add)
        if before is not None and current is not None:
            if current < before:
                add(f"{label}: current_percent must be >= before_percent")
            elif _lane_delta_satisfies_push_gate(before, current, target, min_delta):
                substantial_count += 1
            elif status not in {"closed", "deferred"}:
                required_delta = _required_lane_delta(before, target, min_delta)
                add(
                    f"{label}: active/evidence_ready lane delta must be >= {required_delta:g} "
                    "percentage points or saturate target_percent"
                )
        if current is not None and target is not None and current > target:
            add(f"{label}: current_percent must be <= target_percent")
        if status == "closed" and current is not None and current < 90:
            add(f"{label}: closed lanes must be at least 90% complete")

        _check_evidence(lane.get("evidence"), label, root, add)
        _check_nonempty_list(lane.get("gates"), f"{label}: field gates", add)
        if status in {"active", "evidence_ready"}:
            _check_nonempty_list(lane.get("next_actions"), f"{label}: field next_actions", add)
        if status == "deferred":
            reason = _nonempty_string(lane.get("deferred_reason"))
            if reason is None:
                add(f"{label}: deferred lanes require deferred_reason")

    if substantial_count == 0:
        add("manifest must record at least one substantial lane-completion increase")
    return errors


def validate_research_lane_completion(
    payload: object,
    *,
    source: str | Path | None = None,
    repo_root: str | Path | None = None,
) -> None:
    """Raise ``ResearchLanePolicyError`` if a lane manifest is invalid."""

    errors = research_lane_completion_errors(payload, source=source, repo_root=repo_root)
    if errors:
        raise ResearchLanePolicyError(errors)


def check_research_lane_completion_file(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> list[str]:
    """Load one research-lane JSON file and return policy errors."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return [f"{path}: could not read JSON file: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"]
    return research_lane_completion_errors(payload, source=path, repo_root=repo_root)


def validate_research_lane_completion_file(
    path: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> None:
    """Raise ``ResearchLanePolicyError`` for an invalid lane manifest file."""

    errors = check_research_lane_completion_file(path, repo_root=repo_root)
    if errors:
        raise ResearchLanePolicyError(errors)


def _default_repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _nonempty_string(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _as_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        number = float(value)
        if math.isfinite(number):
            return number
    return None


def _percent_field(
    lane: Mapping[str, object],
    field: str,
    label: str,
    add: Callable[[str], None],
) -> float | None:
    value = _as_number(lane.get(field))
    if value is None or not 0.0 <= value <= 100.0:
        add(f"{label}: field {field} must be a finite percentage in [0, 100]")
        return None
    return value


def _required_lane_delta(before: float, target: float | None, min_delta: float) -> float:
    """Return the target-capped lane movement required by the manifest gate."""

    if target is None:
        return min_delta
    return min(min_delta, max(0.0, target - before))


def _lane_delta_satisfies_push_gate(
    before: float,
    current: float,
    target: float | None,
    min_delta: float,
) -> bool:
    """Return whether a lane made a substantial or target-saturating push.

    Several lanes may have fewer percentage points remaining than the current
    large-push target.  Those lanes must reach ``target_percent`` rather than
    being allowed to overclaim beyond the target just to satisfy the requested
    absolute delta.
    """

    required_delta = _required_lane_delta(before, target, min_delta)
    return current - before >= required_delta


def _check_nonempty_list(value: object, label: str, add: Callable[[str], None]) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        add(f"{label} must be a non-empty list")
        return
    for index, entry in enumerate(value):
        if not _nonempty_string(entry):
            add(f"{label}[{index}] must be a non-empty string")


def _check_evidence(
    value: object,
    label: str,
    repo_root: Path,
    add: Callable[[str], None],
) -> None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        add(f"{label}: field evidence must be a non-empty list")
        return
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            add(f"{label}: evidence[{index}] must be a JSON object")
            continue
        path = _nonempty_string(entry.get("path"))
        claim = _nonempty_string(entry.get("claim"))
        if path is None:
            add(f"{label}: evidence[{index}].path must be a non-empty string")
            continue
        if claim is None:
            add(f"{label}: evidence[{index}].claim must be a non-empty string")
        if not (repo_root / path).exists():
            add(f"{label}: evidence[{index}].path does not exist: {path}")
