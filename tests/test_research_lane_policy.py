from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.check_research_lanes import main as check_research_lanes_main
from sfincs_jax.research_lane_policy import research_lane_completion_errors


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
                        "path": "docs/performance_techniques.rst",
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
