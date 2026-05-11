from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from scripts.check_benchmark_artifacts import main as check_benchmark_artifacts_main
from sfincs_jax.benchmark_artifact_policy import (
    BenchmarkArtifactPolicyError,
    benchmark_artifact_policy_errors,
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


def test_missing_variant_provenance_names_result_index() -> None:
    payload = _valid_payload()
    results = payload["results"]
    assert isinstance(results, list)
    row = results[1]
    assert isinstance(row, dict)
    del row["variant_provenance"]

    errors = benchmark_artifact_policy_errors(payload)

    assert errors == ["missing field results[1].variant_provenance"]


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
