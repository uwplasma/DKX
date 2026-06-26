from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_benchmark_artifacts import main as check_benchmark_artifacts_main
from scripts.benchmark_artifact_index import main as benchmark_artifact_index_main
from sfincs_jax.validation.artifacts import (
    ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY,
    ARTIFACT_CLASS_LEGACY,
    ARTIFACT_CLASS_NON_PAS,
    ARTIFACT_CLASS_RELEASE_BLOCKING,
    ARTIFACT_CLASS_SCHEMA_V2,
    BenchmarkArtifactPolicyError,
    benchmark_artifact_policy_errors,
    classify_benchmark_artifact_file,
    fortran_suite_benchmark_summary_errors,
    index_benchmark_artifact_files,
    validate_benchmark_artifact,
)


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 2,
        "kind": "pas_tz_memory_fallback_benchmark",
        "plan": {
            "variants": ["zeta", "theta"],
            "variant_methods": [
                {"variant": "zeta", "realized_solve_method": "incremental"},
                {"variant": "theta", "realized_solve_method": "incremental"},
            ],
        },
        "results": [
            {
                "variant": "zeta",
                "status": "ok",
                "variant_provenance": {"variant": "zeta"},
                "solver_provenance": {
                    "requested_solve_method": "incremental",
                    "realized_solve_method": "incremental",
                },
                "phase_metadata": [{"name": "solve", "status": "ok", "elapsed_s": 0.1}],
                "tail_metadata": {"messages_tail_limit": 40, "messages_tail_count": 3},
            },
            {
                "variant": "theta",
                "status": "timeout",
                "variant_provenance": {"variant": "theta"},
                "tail_metadata": {"tail_limit_chars": 4000, "stdout_tail_chars": 120},
            },
        ],
    }


def _fortran_metric_row(
    case: str,
    *,
    fortran_runtime_s: float,
    jax_runtime_s: float,
    jax_runtime_s_warm: float | None,
    jax_logged_elapsed_s: float | None,
    jax_max_rss_mb: float,
    jax_incremental_max_rss_mb: float | None,
) -> dict[str, object]:
    warm_or_logged = jax_runtime_s_warm if jax_runtime_s_warm is not None else jax_logged_elapsed_s
    warm_source = "jax_runtime_s_warm" if jax_runtime_s_warm is not None else "jax_logged_elapsed_s"
    active = jax_incremental_max_rss_mb if jax_incremental_max_rss_mb is not None else jax_max_rss_mb
    assert warm_or_logged is not None
    return {
        "case": case,
        "status": "parity_ok",
        "blocker_type": "none",
        "fortran_runtime_s": fortran_runtime_s,
        "jax_runtime_s": jax_runtime_s,
        "jax_runtime_s_cold": jax_runtime_s,
        "jax_runtime_s_warm": jax_runtime_s_warm,
        "jax_logged_elapsed_s": jax_logged_elapsed_s,
        "warm_or_logged_runtime_s": warm_or_logged,
        "warm_or_logged_runtime_source": warm_source,
        "fortran_max_rss_mb": 100.0,
        "jax_max_rss_mb": jax_max_rss_mb,
        "jax_incremental_max_rss_mb": jax_incremental_max_rss_mb,
        "jax_rss_baseline_mb": 300.0,
        "jax_memory_metric_source": "drss_mb",
        "active_jax_memory_mb": active,
        "runtime_ratio": jax_runtime_s / fortran_runtime_s,
        "cold_runtime_ratio": jax_runtime_s / fortran_runtime_s,
        "warm_or_logged_runtime_ratio": warm_or_logged / fortran_runtime_s,
        "active_memory_ratio": active / 100.0,
        "strict_mismatches": 0,
    }


def _valid_fortran_suite_summary_payload() -> dict[str, object]:
    fast = _fortran_metric_row(
        "fast_case",
        fortran_runtime_s=20.0,
        jax_runtime_s=2.0,
        jax_runtime_s_warm=1.0,
        jax_logged_elapsed_s=9.0,
        jax_max_rss_mb=500.0,
        jax_incremental_max_rss_mb=50.0,
    )
    slow = _fortran_metric_row(
        "slow_case",
        fortran_runtime_s=30.0,
        jax_runtime_s=15.0,
        jax_runtime_s_warm=None,
        jax_logged_elapsed_s=10.0,
        jax_max_rss_mb=120.0,
        jax_incremental_max_rss_mb=None,
    )
    report = {
        "total_cases": 2,
        "parity_ok_cases": 2,
        "strict_mismatch_total": 0,
        "cold_runtime_ratio_summary": {"count": 2},
        "warm_or_logged_runtime_ratio_summary": {"count": 2},
        "active_memory_ratio_summary": {"count": 2},
        "warm_or_logged_runtime_source_counts": {
            "jax_logged_elapsed_s": 1,
            "jax_runtime_s_warm": 1,
        },
        "fastest_jax_vs_fortran_cases": [fast, slow],
        "slowest_jax_vs_fortran_cases": [slow, fast],
        "highest_active_jax_memory_cases": [slow, fast],
    }
    return {
        "metadata": {
            "kind": "fortran_v3_suite_benchmark_summary",
            "min_fortran_runtime_s": 10.0,
            "reported_case_counts": {"cpu": 2, "gpu": 2},
            "excluded_low_fortran_runtime_cases": [
                {"case": "short_case", "fortran_runtime_s": 2.0}
            ],
            "canonical_case_order": ["fast_case", "slow_case"],
        },
        "canonical_rows": {
            "cpu": [fast, slow],
            "gpu": [fast, slow],
        },
        "reports": {
            "cpu": copy.deepcopy(report),
            "gpu": copy.deepcopy(report),
        },
    }


def test_valid_payload_passes_policy() -> None:
    payload = _valid_payload()

    assert benchmark_artifact_policy_errors(payload) == []
    validate_benchmark_artifact(payload)


def test_old_schema_version_is_rejected() -> None:
    payload = _valid_payload()
    payload["schema_version"] = 1

    errors = benchmark_artifact_policy_errors(payload, source="artifact.json")

    assert errors == ["artifact.json: field schema_version must be >= 2, got 1"]
    with pytest.raises(BenchmarkArtifactPolicyError, match="schema_version.*>= 2"):
        validate_benchmark_artifact(payload)


def test_missing_plan_variant_methods_is_rejected() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    del plan["variant_methods"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["missing field plan.variant_methods"]


def test_duplicate_plan_variant_methods_are_rejected() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    variant_methods = plan["variant_methods"]
    assert isinstance(variant_methods, list)
    variant_methods.append({"variant": "zeta", "realized_solve_method": "lgmres"})

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["duplicate variant 'zeta' in plan.variant_methods at indexes 0 and 2"]


def test_plan_variant_method_requires_variant_label() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    variant_methods = plan["variant_methods"]
    assert isinstance(variant_methods, list)
    variant_methods[1] = {"realized_solve_method": "incremental"}

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["field plan.variant_methods[1].variant must be a non-empty string"]


def test_missing_variant_provenance_names_result_index() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[1]
    assert isinstance(row, dict)
    del row["variant_provenance"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["missing field results[1].variant_provenance"]


def test_duplicate_result_variants_are_rejected() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = copy.deepcopy(results[1])
    assert isinstance(row, dict)
    row["variant"] = "zeta"
    row["variant_provenance"] = {"variant": "zeta"}
    results.append(row)

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["duplicate variant 'zeta' in results at indexes 0 and 2"]


def test_result_row_requires_variant_label() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[1]
    assert isinstance(row, dict)
    del row["variant"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["field results[1].variant must be a non-empty string"]


def test_missing_solver_provenance_for_ok_row_names_result_index() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[0]
    assert isinstance(row, dict)
    del row["solver_provenance"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["missing field results[0].solver_provenance"]


def test_ok_row_requires_phase_and_tail_metadata() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[0]
    assert isinstance(row, dict)
    del row["phase_metadata"]
    del row["tail_metadata"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == [
        "missing field results[0].phase_metadata",
        "missing field results[0].tail_metadata",
    ]


def test_timeout_row_does_not_require_solver_or_phase_metadata() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[1]
    assert isinstance(row, dict)
    row.pop("solver_provenance", None)
    row.pop("phase_metadata", None)

    assert benchmark_artifact_policy_errors(payload) == []


def test_default_promotion_plan_requires_quantitative_baselines() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    plan["gates"] = {
        "default_promotion_required": True,
        "baseline_elapsed_s": 0.0,
        "baseline_rss_mb": "missing",
        "min_runtime_speedup": 0.75,
        "min_memory_reduction": 0.5,
    }

    errors = benchmark_artifact_policy_errors(payload)

    assert "field plan.gates.baseline_elapsed_s must be a positive finite number when default promotion is required" in errors
    assert "field plan.gates.baseline_rss_mb must be a positive finite number when default promotion is required" in errors
    assert "field plan.gates.min_runtime_speedup must be a finite number >= 1 when default promotion is required" in errors
    assert "field plan.gates.min_memory_reduction must be a finite number >= 1 when default promotion is required" in errors


def test_default_promotion_result_gates_must_pass_for_completed_rows() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    plan["gates"] = {
        "default_promotion_required": True,
        "baseline_elapsed_s": 10.0,
        "baseline_rss_mb": 2000.0,
        "min_runtime_speedup": 1.2,
        "min_memory_reduction": 1.1,
    }

    errors = benchmark_artifact_policy_errors(payload)

    assert "field summary.all_gates_passed must be true when default promotion is required" in errors
    assert "missing field results[0].gates" in errors
    assert not any("results[1].gates" in error for error in errors)


def test_default_promotion_artifact_passes_with_complete_quantitative_gates() -> None:
    payload = _valid_payload()
    plan = payload["plan"]
    assert isinstance(plan, dict)
    plan["gates"] = {
        "default_promotion_required": True,
        "baseline_elapsed_s": 10.0,
        "baseline_rss_mb": 2000.0,
        "min_runtime_speedup": 1.2,
        "min_memory_reduction": 1.1,
    }
    payload["summary"] = {"all_gates_passed": True}
    results = payload["results"]
    assert isinstance(results, list)
    ok_row = results[0]
    assert isinstance(ok_row, dict)
    ok_row["gates"] = {
        "stall": {"status": "pass"},
        "residual": {"status": "pass"},
        "memory": {"status": "pass"},
        "solver_path": {"status": "pass"},
        "default_promotion": {"status": "pass"},
    }

    assert benchmark_artifact_policy_errors(payload) == []
    validate_benchmark_artifact(payload)


def test_fortran_suite_summary_payload_passes_release_policy() -> None:
    payload = _valid_fortran_suite_summary_payload()

    assert fortran_suite_benchmark_summary_errors(payload) == []
    assert benchmark_artifact_policy_errors(payload) == []


def test_fortran_suite_summary_enforces_minimum_fortran_runtime_policy() -> None:
    payload = _valid_fortran_suite_summary_payload()
    canonical_rows = payload["canonical_rows"]
    assert isinstance(canonical_rows, dict)
    cpu_rows = canonical_rows["cpu"]
    assert isinstance(cpu_rows, list)
    cpu_row = cpu_rows[0]
    assert isinstance(cpu_row, dict)
    cpu_row["fortran_runtime_s"] = 9.0

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field canonical_rows.cpu[0].fortran_runtime_s must be >= 10, got 9" in errors


def test_fortran_suite_summary_rejects_lowered_public_runtime_gate() -> None:
    payload = _valid_fortran_suite_summary_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["min_fortran_runtime_s"] = 5.0

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field metadata.min_fortran_runtime_s must be >= 10, got 5" in errors


def test_fortran_suite_summary_rejects_bad_warm_runtime_source() -> None:
    payload = _valid_fortran_suite_summary_payload()
    canonical_rows = payload["canonical_rows"]
    assert isinstance(canonical_rows, dict)
    gpu_rows = canonical_rows["gpu"]
    assert isinstance(gpu_rows, list)
    gpu_row = gpu_rows[1]
    assert isinstance(gpu_row, dict)
    gpu_row["warm_or_logged_runtime_source"] = "jax_runtime_s_warm"

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field canonical_rows.gpu[1].jax_runtime_s_warm must be present" in "\n".join(errors)


def test_fortran_suite_summary_rejects_active_memory_that_ignores_incremental_field() -> None:
    payload = _valid_fortran_suite_summary_payload()
    canonical_rows = payload["canonical_rows"]
    assert isinstance(canonical_rows, dict)
    cpu_rows = canonical_rows["cpu"]
    assert isinstance(cpu_rows, list)
    cpu_row = cpu_rows[0]
    assert isinstance(cpu_row, dict)
    cpu_row["active_jax_memory_mb"] = cpu_row["jax_max_rss_mb"]

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "active_jax_memory_mb must use jax_incremental_max_rss_mb" in "\n".join(errors)


def test_fortran_suite_summary_rejects_included_low_runtime_exclusion_rows() -> None:
    payload = _valid_fortran_suite_summary_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    excluded = metadata["excluded_low_fortran_runtime_cases"]
    assert isinstance(excluded, list)
    excluded[0]["fortran_runtime_s"] = 12.0

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field metadata.excluded_low_fortran_runtime_cases[0].fortran_runtime_s must be below 10, got 12" in errors


def test_fortran_suite_summary_rejects_unsorted_summary_rows() -> None:
    payload = _valid_fortran_suite_summary_payload()
    reports = payload["reports"]
    assert isinstance(reports, dict)
    cpu_report = reports["cpu"]
    assert isinstance(cpu_report, dict)
    rows = cpu_report["fastest_jax_vs_fortran_cases"]
    assert isinstance(rows, list)
    rows.reverse()

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "reports.cpu.fastest_jax_vs_fortran_cases must be sorted ascending by runtime_ratio" in "\n".join(errors)


def test_fortran_suite_summary_rejects_canonical_order_mismatches() -> None:
    payload = _valid_fortran_suite_summary_payload()
    canonical_rows = payload["canonical_rows"]
    assert isinstance(canonical_rows, dict)
    cpu_rows = canonical_rows["cpu"]
    assert isinstance(cpu_rows, list)
    cpu_rows.reverse()

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field canonical_rows.cpu must follow metadata.canonical_case_order" in errors


def test_fortran_suite_summary_rejects_duplicate_canonical_case_order() -> None:
    payload = _valid_fortran_suite_summary_payload()
    metadata = payload["metadata"]
    assert isinstance(metadata, dict)
    metadata["canonical_case_order"] = ["fast_case", "fast_case"]

    errors = fortran_suite_benchmark_summary_errors(payload)

    assert "field metadata.canonical_case_order must not contain duplicates" in errors


def test_cli_reports_success_for_valid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "valid.json"
    artifact.write_text(json.dumps(_valid_payload()) + "\n")

    rc = check_benchmark_artifacts_main([str(artifact)])

    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == f"{artifact}: ok\n"
    assert captured.err == ""


def test_cli_reports_failure_for_invalid_file(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    payload = copy.deepcopy(_valid_payload())
    results = payload["results"]
    assert isinstance(results, list)
    row = results[1]
    assert isinstance(row, dict)
    del row["variant_provenance"]
    artifact = tmp_path / "invalid.json"
    artifact.write_text(json.dumps(payload) + "\n")

    rc = check_benchmark_artifacts_main([str(artifact)])

    captured = capsys.readouterr()
    assert rc == 1
    assert captured.out == ""
    assert captured.err == f"{artifact}: missing field results[1].variant_provenance\n"


def test_release_index_classifies_schema_v2_compliant_file(tmp_path: Path) -> None:
    artifact = tmp_path / "pas_release_v2.json"
    artifact.write_text(json.dumps(_valid_payload()) + "\n")

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_SCHEMA_V2
    assert entry.errors == ()
    assert not entry.release_blocking


def test_release_index_excludes_legacy_schema_v1_from_gate(tmp_path: Path) -> None:
    payload = copy.deepcopy(_valid_payload())
    payload["schema_version"] = 1
    artifact = tmp_path / "pas_historical_v1.json"
    artifact.write_text(json.dumps(payload) + "\n")

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_LEGACY
    assert entry.errors == ()
    assert not entry.release_blocking


def test_release_index_classifies_non_pas_unrelated_file(tmp_path: Path) -> None:
    artifact = tmp_path / "package_metadata.json"
    artifact.write_text(json.dumps({"name": "sfincs-jax", "schema_version": 2}) + "\n")

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_NON_PAS
    assert entry.errors == ()
    assert not entry.release_blocking


def test_release_index_classifies_fortran_suite_summary(tmp_path: Path) -> None:
    artifact = tmp_path / "sfincs_jax_fortran_suite_benchmark_summary.json"
    artifact.write_text(json.dumps(_valid_fortran_suite_summary_payload()) + "\n")

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY
    assert entry.errors == ()
    assert not entry.release_blocking


def test_checked_in_fortran_suite_summary_is_release_indexed() -> None:
    artifact = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "publication_figures"
        / "artifacts"
        / "sfincs_jax_fortran_suite_benchmark_summary.json"
    )

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY
    assert entry.errors == ()
    assert not entry.release_blocking


def test_release_index_treats_malformed_json_as_release_blocking(tmp_path: Path) -> None:
    artifact = tmp_path / "broken.json"
    artifact.write_text('{"schema_version": 2,')

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_RELEASE_BLOCKING
    assert entry.release_blocking
    assert entry.errors == (
        f"{artifact}: invalid JSON: Expecting property name enclosed in double quotes at line 1 column 22",
    )


def test_release_index_treats_duplicate_v2_variants_as_release_blocking(tmp_path: Path) -> None:
    payload = copy.deepcopy(_valid_payload())
    plan = payload["plan"]
    assert isinstance(plan, dict)
    variant_methods = plan["variant_methods"]
    assert isinstance(variant_methods, list)
    variant_methods.append({"variant": "zeta", "realized_solve_method": "lgmres"})
    artifact = tmp_path / "pas_release_duplicate_variants.json"
    artifact.write_text(json.dumps(payload) + "\n")

    entry = classify_benchmark_artifact_file(artifact)

    assert entry.classification == ARTIFACT_CLASS_RELEASE_BLOCKING
    assert entry.release_blocking
    assert entry.errors == (
        f"{artifact}: duplicate variant 'zeta' in plan.variant_methods at indexes 0 and 2",
    )


def test_release_index_summary_counts(tmp_path: Path) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_payload()) + "\n")
    legacy_payload = copy.deepcopy(_valid_payload())
    legacy_payload["schema_version"] = 1
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps(legacy_payload) + "\n")
    unrelated = tmp_path / "unrelated.json"
    unrelated.write_text(json.dumps({"tool": "pytest"}) + "\n")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")

    index = index_benchmark_artifact_files([valid, legacy, unrelated, malformed])

    assert index.counts == {
        ARTIFACT_CLASS_SCHEMA_V2: 1,
        ARTIFACT_CLASS_LEGACY: 1,
        ARTIFACT_CLASS_FORTRAN_SUITE_SUMMARY: 0,
        ARTIFACT_CLASS_NON_PAS: 1,
        ARTIFACT_CLASS_RELEASE_BLOCKING: 1,
    }
    assert [entry.path for entry in index.release_blocking] == [malformed]


def test_release_index_cli_reports_counts_and_fails_for_blockers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_payload()) + "\n")
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{")

    rc = benchmark_artifact_index_main([str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 1
    assert f"{valid}: schema-v2-compliant\n" in captured.out
    assert f"{malformed}: release-blocking\n" in captured.out
    assert "summary: total=2" in captured.out
    assert "schema-v2-compliant=1" in captured.out
    assert "release-blocking=1" in captured.out
    assert f"{malformed}: invalid JSON:" in captured.err
    assert "release gate: fail (1 blocking artifact(s))" in captured.err
