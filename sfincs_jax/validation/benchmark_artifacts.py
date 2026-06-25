"""Fast benchmark-artifact schema and release-gate policy helpers."""

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
ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY = "fortran-suite-summary"
ARTIFACT_CLASS_NON_PAS = "non-pas-unrelated"
ARTIFACT_CLASS_RELEASE_BLOCKING = "release-blocking"
ARTIFACT_CLASSES = (
    ARTIFACT_CLASS_SCHEMA_V2,
    ARTIFACT_CLASS_LEGACY,
    ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY,
    ARTIFACT_CLASS_NON_PAS,
    ARTIFACT_CLASS_RELEASE_BLOCKING,
)
FORTRAN_SUITE_SUMMARY_KIND = "fortran_v3_suite_benchmark_summary"
FORTRAN_SUITE_MIN_RUNTIME_GATE_S = 10.0
FORTRAN_SUITE_BACKENDS = ("cpu", "gpu")
WARM_RUNTIME_SOURCES = frozenset({"jax_runtime_s_warm", "jax_logged_elapsed_s"})


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

    if _looks_like_fortran_suite_benchmark_summary(payload):
        return fortran_suite_benchmark_summary_errors(payload, source=source)

    _check_schema_version(payload, add)
    _check_plan(payload, add)
    _check_results(payload, add)
    return errors


def fortran_suite_benchmark_summary_errors(
    payload: object,
    *,
    source: str | Path | None = None,
) -> list[str]:
    """Return release-gate errors for the Fortran suite benchmark summary."""

    prefix = f"{source}: " if source is not None else ""
    errors: list[str] = []

    def add(message: str) -> None:
        errors.append(prefix + message)

    if not isinstance(payload, Mapping):
        add("Fortran suite benchmark summary must be a JSON object")
        return errors

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        add("missing field metadata")
        return errors

    if metadata.get("kind") != FORTRAN_SUITE_SUMMARY_KIND:
        add(f"field metadata.kind must be {FORTRAN_SUITE_SUMMARY_KIND!r}")

    min_runtime = _finite_number(metadata.get("min_fortran_runtime_s"))
    if min_runtime is None:
        add("field metadata.min_fortran_runtime_s must be a finite number")
        min_runtime = FORTRAN_SUITE_MIN_RUNTIME_GATE_S
    elif min_runtime < FORTRAN_SUITE_MIN_RUNTIME_GATE_S:
        add(
            "field metadata.min_fortran_runtime_s must be "
            f">= {FORTRAN_SUITE_MIN_RUNTIME_GATE_S:g}, got {min_runtime:g}"
        )

    _check_excluded_fortran_runtime_rows(metadata, min_runtime, add)
    _check_fortran_summary_reports(payload, metadata, min_runtime, add)
    _check_fortran_summary_canonical_rows(payload, min_runtime, add)
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

    if _looks_like_fortran_suite_benchmark_summary(payload):
        errors = tuple(fortran_suite_benchmark_summary_errors(payload, source=path))
        if errors:
            return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_RELEASE_BLOCKING, errors)
        return BenchmarkArtifactIndexEntry(path, ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY)

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


def _looks_like_fortran_suite_benchmark_summary(payload: Mapping[object, object]) -> bool:
    metadata = payload.get("metadata")
    return isinstance(metadata, Mapping) and metadata.get("kind") == FORTRAN_SUITE_SUMMARY_KIND


def _finite_number(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _int_value(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _numbers_close(left: float, right: float) -> bool:
    return abs(left - right) <= max(1.0e-9, 1.0e-12 * max(abs(left), abs(right), 1.0))


def _check_excluded_fortran_runtime_rows(
    metadata: Mapping[object, object],
    min_runtime: float,
    add: ErrorSink,
) -> None:
    excluded = metadata.get("excluded_low_fortran_runtime_cases", [])
    if not isinstance(excluded, list):
        add("field metadata.excluded_low_fortran_runtime_cases must be a list")
        return
    for idx, row in enumerate(excluded):
        row_path = f"metadata.excluded_low_fortran_runtime_cases[{idx}]"
        if not isinstance(row, Mapping):
            add(f"{row_path} must be a JSON object")
            continue
        case = row.get("case")
        if not isinstance(case, str) or not case:
            add(f"field {row_path}.case must be a non-empty string")
        runtime = _finite_number(row.get("fortran_runtime_s"))
        if runtime is not None and runtime >= min_runtime:
            add(
                f"field {row_path}.fortran_runtime_s must be below "
                f"{min_runtime:g}, got {runtime:g}"
            )


def _check_fortran_summary_reports(
    payload: Mapping[object, object],
    metadata: Mapping[object, object],
    min_runtime: float,
    add: ErrorSink,
) -> None:
    reports = payload.get("reports")
    if not isinstance(reports, Mapping):
        add("missing field reports")
        return

    reported_counts = metadata.get("reported_case_counts")
    if not isinstance(reported_counts, Mapping):
        add("missing field metadata.reported_case_counts")
        reported_counts = {}

    for backend in FORTRAN_SUITE_BACKENDS:
        report_path = f"reports.{backend}"
        report = reports.get(backend)
        if not isinstance(report, Mapping):
            add(f"missing field {report_path}")
            continue
        total_cases = _int_value(report.get("total_cases"))
        if total_cases is None or total_cases < 1:
            add(f"field {report_path}.total_cases must be a positive integer")
            total_cases = 0
        expected_count = _int_value(reported_counts.get(backend))
        if expected_count is not None and total_cases != expected_count:
            add(
                f"field {report_path}.total_cases must match "
                f"metadata.reported_case_counts.{backend}"
            )
        parity_ok = _int_value(report.get("parity_ok_cases"))
        if parity_ok != total_cases:
            add(f"field {report_path}.parity_ok_cases must equal total_cases")
        strict_mismatches = _int_value(report.get("strict_mismatch_total"))
        if strict_mismatches != 0:
            add(f"field {report_path}.strict_mismatch_total must be 0")
        _check_summary_count(report, "cold_runtime_ratio_summary", total_cases, report_path, add)
        _check_summary_count(
            report,
            "warm_or_logged_runtime_ratio_summary",
            total_cases,
            report_path,
            add,
        )
        _check_summary_count(report, "active_memory_ratio_summary", total_cases, report_path, add)
        _check_warm_source_counts(report, total_cases, report_path, add)
        _check_sorted_summary_rows(
            report,
            "fastest_jax_vs_fortran_cases",
            "runtime_ratio",
            reverse=False,
            report_path=report_path,
            min_runtime=min_runtime,
            add=add,
        )
        _check_sorted_summary_rows(
            report,
            "slowest_jax_vs_fortran_cases",
            "runtime_ratio",
            reverse=True,
            report_path=report_path,
            min_runtime=min_runtime,
            add=add,
        )
        _check_sorted_summary_rows(
            report,
            "highest_active_jax_memory_cases",
            "active_jax_memory_mb",
            reverse=True,
            report_path=report_path,
            min_runtime=min_runtime,
            add=add,
        )


def _check_summary_count(
    report: Mapping[object, object],
    key: str,
    total_cases: int,
    report_path: str,
    add: ErrorSink,
) -> None:
    summary = report.get(key)
    if not isinstance(summary, Mapping):
        add(f"missing field {report_path}.{key}")
        return
    count = _int_value(summary.get("count"))
    if count != total_cases:
        add(f"field {report_path}.{key}.count must equal total_cases")


def _check_warm_source_counts(
    report: Mapping[object, object],
    total_cases: int,
    report_path: str,
    add: ErrorSink,
) -> None:
    counts = report.get("warm_or_logged_runtime_source_counts")
    if not isinstance(counts, Mapping):
        add(f"missing field {report_path}.warm_or_logged_runtime_source_counts")
        return
    total = 0
    for source, count in counts.items():
        if source not in WARM_RUNTIME_SOURCES:
            add(f"field {report_path}.warm_or_logged_runtime_source_counts has unknown source {source!r}")
        value = _int_value(count)
        if value is None or value < 0:
            add(f"field {report_path}.warm_or_logged_runtime_source_counts.{source} must be non-negative")
            continue
        total += value
    if total != total_cases:
        add(f"field {report_path}.warm_or_logged_runtime_source_counts must sum to total_cases")


def _check_sorted_summary_rows(
    report: Mapping[object, object],
    list_key: str,
    sort_key: str,
    *,
    reverse: bool,
    report_path: str,
    min_runtime: float,
    add: ErrorSink,
) -> None:
    rows = report.get(list_key)
    if not isinstance(rows, list):
        add(f"missing field {report_path}.{list_key}")
        return

    values: list[float] = []
    for idx, row in enumerate(rows):
        row_path = f"{report_path}.{list_key}[{idx}]"
        if not isinstance(row, Mapping):
            add(f"{row_path} must be a JSON object")
            continue
        _check_fortran_summary_metric_row(row, row_path, min_runtime, add)
        value = _finite_number(row.get(sort_key))
        if value is None:
            add(f"field {row_path}.{sort_key} must be a finite number")
            continue
        values.append(value)

    if values != sorted(values, reverse=reverse):
        direction = "descending" if reverse else "ascending"
        add(f"field {report_path}.{list_key} must be sorted {direction} by {sort_key}")


def _check_fortran_summary_metric_row(
    row: Mapping[object, object],
    row_path: str,
    min_runtime: float,
    add: ErrorSink,
) -> None:
    case = row.get("case")
    if not isinstance(case, str) or not case:
        add(f"field {row_path}.case must be a non-empty string")

    status = row.get("status")
    if status != "parity_ok":
        add(f"field {row_path}.status must be 'parity_ok'")

    runtime = _finite_number(row.get("fortran_runtime_s"))
    if runtime is None:
        add(f"field {row_path}.fortran_runtime_s must be a finite number")
    elif runtime < min_runtime:
        add(f"field {row_path}.fortran_runtime_s must be >= {min_runtime:g}, got {runtime:g}")

    cold_runtime = _finite_number(row.get("jax_runtime_s_cold"))
    if cold_runtime is None or cold_runtime <= 0.0:
        add(f"field {row_path}.jax_runtime_s_cold must be a positive finite number")

    warm_runtime = _finite_number(row.get("warm_or_logged_runtime_s"))
    if warm_runtime is None or warm_runtime <= 0.0:
        add(f"field {row_path}.warm_or_logged_runtime_s must be a positive finite number")

    source = row.get("warm_or_logged_runtime_source")
    if source not in WARM_RUNTIME_SOURCES:
        add(f"field {row_path}.warm_or_logged_runtime_source must be one of {sorted(WARM_RUNTIME_SOURCES)}")
    elif source == "jax_runtime_s_warm" and _finite_number(row.get("jax_runtime_s_warm")) is None:
        add(f"field {row_path}.jax_runtime_s_warm must be present when selected as warm source")
    elif source == "jax_logged_elapsed_s" and _finite_number(row.get("jax_logged_elapsed_s")) is None:
        add(f"field {row_path}.jax_logged_elapsed_s must be present when selected as warm source")

    active = _finite_number(row.get("active_jax_memory_mb"))
    if active is None or active <= 0.0:
        add(f"field {row_path}.active_jax_memory_mb must be a positive finite number")
    incremental = _finite_number(row.get("jax_incremental_max_rss_mb"))
    peak = _finite_number(row.get("jax_max_rss_mb"))
    expected_active = incremental if incremental is not None else peak
    if active is not None and expected_active is not None and not _numbers_close(active, expected_active):
        add(
            f"field {row_path}.active_jax_memory_mb must use "
            "jax_incremental_max_rss_mb when present, otherwise jax_max_rss_mb"
        )


def _check_fortran_summary_canonical_rows(
    payload: Mapping[object, object],
    min_runtime: float,
    add: ErrorSink,
) -> None:
    canonical_rows = payload.get("canonical_rows")
    if canonical_rows is None:
        return
    if not isinstance(canonical_rows, Mapping):
        add("field canonical_rows must be a JSON object")
        return

    cases_by_backend: dict[str, list[str]] = {}
    for backend in FORTRAN_SUITE_BACKENDS:
        rows = canonical_rows.get(backend)
        if not isinstance(rows, list):
            add(f"field canonical_rows.{backend} must be a list")
            continue
        cases: list[str] = []
        seen: set[str] = set()
        for idx, row in enumerate(rows):
            row_path = f"canonical_rows.{backend}[{idx}]"
            if not isinstance(row, Mapping):
                add(f"{row_path} must be a JSON object")
                continue
            _check_fortran_summary_metric_row(row, row_path, min_runtime, add)
            case = row.get("case")
            if isinstance(case, str):
                if case in seen:
                    add(f"duplicate case {case!r} in canonical_rows.{backend}")
                seen.add(case)
                cases.append(case)
        cases_by_backend[backend] = cases

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        return
    case_order = metadata.get("canonical_case_order")
    if case_order is None:
        add("missing field metadata.canonical_case_order when canonical_rows is present")
        return
    if not isinstance(case_order, list) or not all(isinstance(case, str) for case in case_order):
        add("field metadata.canonical_case_order must be a list of strings")
        return
    if len(case_order) != len(set(case_order)):
        add("field metadata.canonical_case_order must not contain duplicates")
    for backend, cases in cases_by_backend.items():
        ordered_subset = [case for case in case_order if case in set(cases)]
        if ordered_subset != cases:
            add(f"field canonical_rows.{backend} must follow metadata.canonical_case_order")


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

    _check_default_promotion_plan(plan, add)


def _check_default_promotion_plan(plan: Mapping[object, object], add: ErrorSink) -> None:
    gates = plan.get("gates", {})
    if not isinstance(gates, Mapping):
        return
    if gates.get("default_promotion_required") is not True:
        return

    baseline_elapsed = _finite_number(gates.get("baseline_elapsed_s"))
    if baseline_elapsed is None or baseline_elapsed <= 0.0:
        add("field plan.gates.baseline_elapsed_s must be a positive finite number when default promotion is required")
    baseline_rss = _finite_number(gates.get("baseline_rss_mb"))
    if baseline_rss is None or baseline_rss <= 0.0:
        add("field plan.gates.baseline_rss_mb must be a positive finite number when default promotion is required")
    min_runtime_speedup = _finite_number(gates.get("min_runtime_speedup"))
    if min_runtime_speedup is None or min_runtime_speedup < 1.0:
        add("field plan.gates.min_runtime_speedup must be a finite number >= 1 when default promotion is required")
    min_memory_reduction = _finite_number(gates.get("min_memory_reduction"))
    if min_memory_reduction is None or min_memory_reduction < 1.0:
        add("field plan.gates.min_memory_reduction must be a finite number >= 1 when default promotion is required")


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

    _check_default_promotion_results(payload, results, add)


def _default_promotion_required(payload: Mapping[object, object]) -> bool:
    plan = payload.get("plan")
    if not isinstance(plan, Mapping):
        return False
    gates = plan.get("gates")
    return isinstance(gates, Mapping) and gates.get("default_promotion_required") is True


def _check_default_promotion_results(
    payload: Mapping[object, object],
    results: list[object],
    add: ErrorSink,
) -> None:
    if not _default_promotion_required(payload):
        return
    summary = payload.get("summary", {})
    if isinstance(summary, Mapping) and summary.get("all_gates_passed") is not True:
        add("field summary.all_gates_passed must be true when default promotion is required")
    for idx, row in enumerate(results):
        if not isinstance(row, Mapping) or row.get("status") not in OK_RESULT_STATUSES:
            continue
        row_path = f"results[{idx}]"
        gates = row.get("gates")
        if not isinstance(gates, Mapping):
            add(f"missing field {row_path}.gates")
            continue
        for gate_name in ("stall", "residual", "memory", "solver_path", "default_promotion"):
            gate = gates.get(gate_name)
            if not isinstance(gate, Mapping):
                add(f"missing field {row_path}.gates.{gate_name}")
                continue
            if gate.get("status") != "pass":
                add(f"field {row_path}.gates.{gate_name}.status must be 'pass' when default promotion is required")


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
