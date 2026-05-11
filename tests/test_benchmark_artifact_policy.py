from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_benchmark_artifacts import main as check_benchmark_artifacts_main
from scripts.benchmark_artifact_index import main as benchmark_artifact_index_main
from sfincs_jax.benchmark_artifact_policy import (
    ARTIFACT_CLASS_LEGACY,
    ARTIFACT_CLASS_NON_PAS,
    ARTIFACT_CLASS_RELEASE_BLOCKING,
    ARTIFACT_CLASS_SCHEMA_V2,
    BenchmarkArtifactPolicyError,
    benchmark_artifact_policy_errors,
    classify_benchmark_artifact_file,
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
