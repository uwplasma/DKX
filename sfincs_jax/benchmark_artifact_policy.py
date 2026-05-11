from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
import json
from pathlib import Path


OK_RESULT_STATUSES = frozenset({"ok"})
TAIL_RESULT_STATUSES = frozenset({"ok", "timeout", "error"})
ErrorSink = Callable[[str], None]


class BenchmarkArtifactPolicyError(ValueError):
    """Raised when a benchmark artifact violates the reproducibility policy."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


def benchmark_artifact_policy_errors(
    payload: object,
    *,
    source: str | Path | None = None,
) -> list[str]:
    """Return policy errors for a PAS benchmark JSON payload.

    The policy is intentionally small and fast: checked-in PAS benchmark
    artifacts must identify the schema version, the planned variant methods,
    and the provenance metadata needed to distinguish defaults from opt-in
    probes. Completed rows also need solver and phase metadata.
    """

    prefix = f"{source}: " if source is not None else ""
    errors: list[str] = []

    def add(message: str) -> None:
        errors.append(prefix + message)

    if not isinstance(payload, Mapping):
        add("artifact must be a JSON object")
        return errors

    _check_schema_version(payload, add)
    _check_plan(payload, add)
    _check_results(payload, add)
    return errors


def validate_benchmark_artifact(
    payload: object,
    *,
    source: str | Path | None = None,
) -> None:
    """Raise ``BenchmarkArtifactPolicyError`` if ``payload`` violates policy."""

    errors = benchmark_artifact_policy_errors(payload, source=source)
    if errors:
        raise BenchmarkArtifactPolicyError(errors)


def check_benchmark_artifact_file(path: str | Path) -> list[str]:
    """Load one JSON file and return benchmark artifact policy errors."""

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        return [f"{path}: could not read JSON file: {exc}"]
    except json.JSONDecodeError as exc:
        return [f"{path}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}"]
    return benchmark_artifact_policy_errors(payload, source=path)


def check_benchmark_artifact_files(paths: Iterable[str | Path]) -> list[str]:
    """Return all policy errors across one or more benchmark JSON files."""

    errors: list[str] = []
    for path in paths:
        errors.extend(check_benchmark_artifact_file(path))
    return errors


def validate_benchmark_artifact_file(path: str | Path) -> None:
    """Raise ``BenchmarkArtifactPolicyError`` for an invalid JSON artifact file."""

    errors = check_benchmark_artifact_file(path)
    if errors:
        raise BenchmarkArtifactPolicyError(errors)


def _check_schema_version(payload: Mapping[object, object], add: ErrorSink) -> None:
    if "schema_version" not in payload:
        add("missing field schema_version")
        return

    schema_version = payload["schema_version"]
    if isinstance(schema_version, bool) or not isinstance(schema_version, int | float):
        add(f"field schema_version must be a number >= 2, got {schema_version!r}")
        return
    if schema_version < 2:
        add(f"field schema_version must be >= 2, got {schema_version!r}")


def _check_plan(payload: Mapping[object, object], add: ErrorSink) -> None:
    if "plan" not in payload:
        add("missing field plan")
        return

    plan = payload["plan"]
    if not isinstance(plan, Mapping):
        add("field plan must be a JSON object")
        return

    if "variant_methods" not in plan:
        add("missing field plan.variant_methods")
        return

    variant_methods = plan["variant_methods"]
    if not isinstance(variant_methods, list):
        add("field plan.variant_methods must be a list")


def _check_results(payload: Mapping[object, object], add: ErrorSink) -> None:
    if "results" not in payload:
        add("missing field results")
        return

    results = payload["results"]
    if not isinstance(results, list):
        add("field results must be a list")
        return

    for idx, row in enumerate(results):
        _check_result_row(idx, row, add)


def _check_result_row(idx: int, row: object, add: ErrorSink) -> None:
    row_path = f"results[{idx}]"
    if not isinstance(row, Mapping):
        add(f"{row_path} must be a JSON object")
        return

    _require_mapping_field(row, "variant_provenance", f"{row_path}.variant_provenance", add)

    status = row.get("status")
    if status in OK_RESULT_STATUSES:
        _require_mapping_field(row, "solver_provenance", f"{row_path}.solver_provenance", add)
        _require_list_field(row, "phase_metadata", f"{row_path}.phase_metadata", add)
    if status in TAIL_RESULT_STATUSES:
        _require_mapping_field(row, "tail_metadata", f"{row_path}.tail_metadata", add)


def _require_mapping_field(
    row: Mapping[object, object],
    key: str,
    field_path: str,
    add: ErrorSink,
) -> None:
    if key not in row:
        add(f"missing field {field_path}")
        return
    if not isinstance(row[key], Mapping):
        add(f"field {field_path} must be a JSON object")


def _require_list_field(
    row: Mapping[object, object],
    key: str,
    field_path: str,
    add: ErrorSink,
) -> None:
    if key not in row:
        add(f"missing field {field_path}")
        return
    if not isinstance(row[key], list):
        add(f"field {field_path} must be a list")
