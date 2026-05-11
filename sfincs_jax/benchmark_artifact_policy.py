from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path


OK_RESULT_STATUSES = frozenset({"ok"})
TAIL_RESULT_STATUSES = frozenset({"ok", "timeout", "error"})
ErrorSink = Callable[[str], None]

ARTIFACT_CLASS_SCHEMA_V2 = "schema-v2-compliant"
ARTIFACT_CLASS_LEGACY = "legacy-schema-v1-historical"
ARTIFACT_CLASS_NON_PAS = "non-pas-unrelated"
ARTIFACT_CLASS_RELEASE_BLOCKING = "release-blocking"
ARTIFACT_CLASSES = (
    ARTIFACT_CLASS_SCHEMA_V2,
    ARTIFACT_CLASS_LEGACY,
    ARTIFACT_CLASS_NON_PAS,
    ARTIFACT_CLASS_RELEASE_BLOCKING,
)


class BenchmarkArtifactPolicyError(ValueError):
    """Raised when a benchmark artifact violates the reproducibility policy."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class BenchmarkArtifactIndexEntry:
    """Release-gating classification for one selected JSON artifact."""

    path: Path
    classification: str
    errors: tuple[str, ...] = ()

    @property
    def release_blocking(self) -> bool:
        return self.classification == ARTIFACT_CLASS_RELEASE_BLOCKING


@dataclass(frozen=True)
class BenchmarkArtifactIndex:
    """Summary of benchmark artifact classifications."""

    entries: tuple[BenchmarkArtifactIndexEntry, ...]

    @property
    def counts(self) -> dict[str, int]:
        counts = dict.fromkeys(ARTIFACT_CLASSES, 0)
        for entry in self.entries:
            counts[entry.classification] += 1
        return counts

    @property
    def release_blocking(self) -> tuple[BenchmarkArtifactIndexEntry, ...]:
        return tuple(entry for entry in self.entries if entry.release_blocking)


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


def classify_benchmark_artifact_file(path: str | Path) -> BenchmarkArtifactIndexEntry:
    """Classify one selected JSON file for release benchmark gating.

    Historical schema-v1 artifacts are indexed but excluded from release
    blocking so checked-in benchmark history can coexist with v2 release
    artifacts. Malformed JSON and schema-v2 PAS artifacts that fail the policy
    are release-blocking.
    """

    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except OSError as exc:
        return BenchmarkArtifactIndexEntry(
            path,
            ARTIFACT_CLASS_RELEASE_BLOCKING,
            (f"{path}: could not read JSON file: {exc}",),
        )
    except json.JSONDecodeError as exc:
        return BenchmarkArtifactIndexEntry(
            path,
            ARTIFACT_CLASS_RELEASE_BLOCKING,
            (f"{path}: invalid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",),
        )

    if not isinstance(payload, Mapping):
        return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_NON_PAS)

    schema_version = payload.get("schema_version")
    if _is_legacy_schema_version(schema_version):
        return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_LEGACY)

    if not _looks_like_pas_benchmark_artifact(payload):
        return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_NON_PAS)

    errors = tuple(benchmark_artifact_policy_errors(payload, source=path))
    if errors:
        return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_RELEASE_BLOCKING, errors)
    return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_SCHEMA_V2)


def index_benchmark_artifact_files(paths: Iterable[str | Path]) -> BenchmarkArtifactIndex:
    """Classify selected JSON files for release benchmark gating."""

    return BenchmarkArtifactIndex(
        tuple(classify_benchmark_artifact_file(path) for path in paths)
    )


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


def _is_legacy_schema_version(schema_version: object) -> bool:
    return (
        not isinstance(schema_version, bool)
        and isinstance(schema_version, int | float)
        and schema_version < 2
    )


def _looks_like_pas_benchmark_artifact(payload: Mapping[object, object]) -> bool:
    kind = payload.get("kind")
    if isinstance(kind, str) and kind.startswith("pas_") and "benchmark" in kind:
        return True

    plan = payload.get("plan")
    return isinstance(plan, Mapping) and "variant_methods" in plan and "results" in payload


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
        return

    seen_variants: dict[str, int] = {}
    for idx, method in enumerate(variant_methods):
        method_path = f"plan.variant_methods[{idx}]"
        if not isinstance(method, Mapping):
            add(f"{method_path} must be a JSON object")
            continue
        variant = method.get("variant")
        if not isinstance(variant, str) or not variant:
            add(f"field {method_path}.variant must be a non-empty string")
            continue
        if variant in seen_variants:
            add(
                f"duplicate variant {variant!r} in plan.variant_methods "
                f"at indexes {seen_variants[variant]} and {idx}"
            )
            continue
        seen_variants[variant] = idx


def _check_results(payload: Mapping[object, object], add: ErrorSink) -> None:
    if "results" not in payload:
        add("missing field results")
        return

    results = payload["results"]
    if not isinstance(results, list):
        add("field results must be a list")
        return

    seen_variants: dict[str, int] = {}
    for idx, row in enumerate(results):
        _check_result_row(idx, row, add)
        if not isinstance(row, Mapping):
            continue
        variant = row.get("variant")
        if not isinstance(variant, str) or not variant:
            add(f"field results[{idx}].variant must be a non-empty string")
            continue
        if variant in seen_variants:
            add(
                f"duplicate variant {variant!r} in results "
                f"at indexes {seen_variants[variant]} and {idx}"
            )
            continue
        seen_variants[variant] = idx


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
