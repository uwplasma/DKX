from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dkx.validation.release import check_research_lanes_main
from dkx.validation.artifacts import (
    ResearchLanePolicyError,
    check_research_lane_completion_file,
    research_lane_completion_errors,
    validate_research_lane_completion,
    validate_research_lane_completion_file,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "docs" / "_static" / "research_lane_completion_2026_05_12.json"


def _valid_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "minimum_substantial_delta_percent": 10,
        "lanes": [
            {
                "id": "pas_geometry_runtime_memory",
                "title": "PAS geometry-rich runtime and memory",
                "status": "evidence_ready",
                "before_percent": 70,
                "current_percent": 82,
                "target_percent": 95,
                "evidence": [
                    {
                        "path": "docs/performance.rst",
                        "claim": "Documents the production PAS memory strategy.",
                    }
                ],
                "gates": ["Residual and parity gates remain required before closure."],
                "next_actions": ["Run the next production-floor PAS GPU campaign."],
            }
        ],
    }


def test_research_lane_policy_accepts_valid_payload() -> None:
    errors = research_lane_completion_errors(_valid_payload(), repo_root=_repo_root())

    assert errors == []


def test_research_lane_policy_rejects_overclaiming() -> None:
    payload = copy.deepcopy(_valid_payload())
    lane = payload["lanes"][0]  # type: ignore[index]
    lane["current_percent"] = 75  # type: ignore[index]
    lane["evidence"] = []  # type: ignore[index]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert (
        "pas_geometry_runtime_memory: active/evidence_ready lane delta must be >= 10 "
        "percentage points or saturate target_percent"
    ) in errors
    assert "pas_geometry_runtime_memory: field evidence must be a non-empty list" in errors


def test_research_lane_policy_rejects_manifest_shape_and_bad_scalars() -> None:
    assert research_lane_completion_errors(["not", "an", "object"]) == [
        "manifest must be a JSON object"
    ]

    payload = copy.deepcopy(_valid_payload())
    payload["schema_version"] = 0
    payload["minimum_substantial_delta_percent"] = -1
    payload["lanes"] = []

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert "field schema_version must be a number >= 1" in errors
    assert "field minimum_substantial_delta_percent must be a non-negative number" in errors
    assert "field lanes must be a non-empty list" in errors


def test_research_lane_policy_reports_lane_schema_errors() -> None:
    payload = copy.deepcopy(_valid_payload())
    payload["lanes"] = [
        "not-a-lane",
        {
            "id": "duplicate",
            "title": "",
            "status": "unknown",
            "before_percent": 90,
            "current_percent": 80,
            "target_percent": 110,
            "evidence": [{"path": "", "claim": ""}],
            "gates": ["valid gate"],
            "next_actions": ["valid action"],
        },
        {
            "id": "duplicate",
            "title": "Closed but too low",
            "status": "closed",
            "before_percent": 50,
            "current_percent": 80,
            "target_percent": 100,
            "evidence": [{"path": "docs/performance.rst", "claim": "ok"}],
            "gates": ["valid gate"],
        },
    ]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert "lanes[0] must be a JSON object" in errors
    assert "duplicate: field title must be a non-empty string" in errors
    assert any("field status must be one of" in error for error in errors)
    assert "duplicate: current_percent must be >= before_percent" in errors
    assert "duplicate: field target_percent must be a finite percentage in [0, 100]" in errors
    assert "duplicate: evidence[0].path must be a non-empty string" in errors
    assert "duplicate lane id 'duplicate'" in errors
    assert "duplicate: closed lanes must be at least 90% complete" in errors


def test_research_lane_policy_requires_deferred_reason_and_next_actions() -> None:
    payload = copy.deepcopy(_valid_payload())
    active = payload["lanes"][0]  # type: ignore[index]
    active["next_actions"] = [""]  # type: ignore[index]
    deferred = copy.deepcopy(active)
    deferred["id"] = "deferred_lane"
    deferred["title"] = "Deferred lane"
    deferred["status"] = "deferred"
    deferred["before_percent"] = 40
    deferred["current_percent"] = 40
    deferred["target_percent"] = 90
    deferred.pop("deferred_reason", None)
    payload["lanes"].append(deferred)  # type: ignore[union-attr]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert "pas_geometry_runtime_memory: field next_actions[0] must be a non-empty string" in errors
    assert "deferred_lane: deferred lanes require deferred_reason" in errors


def test_research_lane_policy_checks_evidence_path_and_claim() -> None:
    payload = copy.deepcopy(_valid_payload())
    lane = payload["lanes"][0]  # type: ignore[index]
    lane["evidence"] = [  # type: ignore[index]
        "not-an-object",
        {"path": "docs/does_not_exist.rst", "claim": ""},
    ]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert "pas_geometry_runtime_memory: evidence[0] must be a JSON object" in errors
    assert "pas_geometry_runtime_memory: evidence[1].claim must be a non-empty string" in errors
    assert any("evidence[1].path does not exist: docs/does_not_exist.rst" in error for error in errors)


def test_research_lane_policy_accepts_target_saturation_for_large_push() -> None:
    payload = copy.deepcopy(_valid_payload())
    payload["minimum_substantial_delta_percent"] = 15
    lane = payload["lanes"][0]  # type: ignore[index]
    lane["before_percent"] = 85  # type: ignore[index]
    lane["current_percent"] = 93  # type: ignore[index]
    lane["target_percent"] = 93  # type: ignore[index]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert errors == []


def test_research_lane_policy_rejects_unfinished_target_saturation() -> None:
    payload = copy.deepcopy(_valid_payload())
    payload["minimum_substantial_delta_percent"] = 15
    lane = payload["lanes"][0]  # type: ignore[index]
    lane["before_percent"] = 85  # type: ignore[index]
    lane["current_percent"] = 92  # type: ignore[index]
    lane["target_percent"] = 93  # type: ignore[index]

    errors = research_lane_completion_errors(payload, repo_root=_repo_root())

    assert (
        "pas_geometry_runtime_memory: active/evidence_ready lane delta must be >= 8 "
        "percentage points or saturate target_percent"
    ) in errors


def test_checked_in_research_lane_manifest_is_consistent() -> None:
    payload = json.loads(_manifest_path().read_text(encoding="utf-8"))

    assert research_lane_completion_errors(payload, source=_manifest_path(), repo_root=_repo_root()) == []


def test_checked_in_research_lane_manifest_records_computed_overall_average() -> None:
    payload = json.loads(_manifest_path().read_text(encoding="utf-8"))
    lanes = payload["lanes"]
    expected = round(sum(float(lane["current_percent"]) for lane in lanes) / len(lanes), 1)

    assert payload["overall_average_percent"] == expected
    assert payload["overall_average_percent"] > 90.0


def test_research_lane_checker_cli_passes_current_manifest() -> None:
    assert check_research_lanes_main([]) == 0


def test_research_lane_validation_wrappers_raise_structured_errors(tmp_path: Path) -> None:
    bad_payload = {"schema_version": 1, "lanes": []}

    with pytest.raises(ResearchLanePolicyError) as excinfo:
        validate_research_lane_completion(bad_payload, source="unit.json")
    assert excinfo.value.errors == ("unit.json: field lanes must be a non-empty list",)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{not-json", encoding="utf-8")
    errors = check_research_lane_completion_file(bad_json, repo_root=_repo_root())
    assert errors == [f"{bad_json}: invalid JSON: Expecting property name enclosed in double quotes at line 1 column 2"]

    missing = tmp_path / "missing.json"
    assert "could not read JSON file" in check_research_lane_completion_file(missing)[0]

    valid = tmp_path / "valid.json"
    valid.write_text(json.dumps(_valid_payload()), encoding="utf-8")
    validate_research_lane_completion_file(valid, repo_root=_repo_root())
