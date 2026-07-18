from __future__ import annotations

import copy
import json
from pathlib import Path

from dkx.validation.release import release_gate_errors


_DEFERRED_LANE_IDS = {
    "sfincs2014_fig3_high_collisionality_limit",
    "w7x_ambipolar_er_validation",
    "monkes_monoenergetic_overlap",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "examples" / "publication_figures" / "validation_manifest.json"


def _manifest() -> list[dict[str, object]]:
    payload = json.loads(_manifest_path().read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    return payload


def _record(records: list[dict[str, object]], lane_id: str) -> dict[str, object]:
    return next(record for record in records if str(record["id"]) == lane_id)


def test_deferred_validation_lanes_are_closed_but_not_untracked() -> None:
    records = _manifest()
    assert release_gate_errors(docs_paths=()) == []

    deferred = {str(record["id"]): record for record in records if str(record["status"]) == "deferred_post_release"}
    assert set(deferred) == _DEFERRED_LANE_IDS
    for lane_id, record in deferred.items():
        gate = record["release_gate"]
        assert isinstance(gate, dict)
        assert gate["claim_status"] == "closed_deferred", lane_id
        assert gate["blocks_current_release"] is False, lane_id
        assert "post-release" in str(gate["closed_or_deferred_reason"]).lower(), lane_id
        assert str(gate["promotion_gate"]).startswith("Promote only after"), lane_id
        joined_claims = " ".join(str(claim).lower() for claim in record["claims"])
        assert "closed" in joined_claims and "post-release" in joined_claims, lane_id


def test_release_gate_rejects_stale_deferred_lane_paths(tmp_path: Path) -> None:
    records = copy.deepcopy(_manifest())
    lane = _record(records, "w7x_ambipolar_er_validation")
    lane["source_code"] = ["dkx/does_not_exist_deferred_source.py"]
    lane["tests"] = ["tests/test_missing_deferred_validation_gate.py"]
    lane["scripts"] = ["examples/publication_figures/missing_w7x_validation_script.py --fast"]
    lane["artifacts"] = ["docs/_static/figures/paper/missing_w7x_validation_artifact.png"]
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    errors = release_gate_errors(manifest_path, docs_paths=())

    assert (
        "w7x_ambipolar_er_validation: source_code path does not exist: "
        "dkx/does_not_exist_deferred_source.py"
    ) in errors
    assert (
        "w7x_ambipolar_er_validation: tests path does not exist: "
        "tests/test_missing_deferred_validation_gate.py"
    ) in errors
    assert (
        "w7x_ambipolar_er_validation: script path does not exist: "
        "examples/publication_figures/missing_w7x_validation_script.py"
    ) in errors
    assert (
        "w7x_ambipolar_er_validation: artifacts path does not exist: "
        "docs/_static/figures/paper/missing_w7x_validation_artifact.png"
    ) in errors


def test_release_gate_rejects_unreviewable_validation_record_shape(tmp_path: Path) -> None:
    records = copy.deepcopy(_manifest())
    lane = _record(records, "sfincs2014_fig1_lhd_collisionality")
    lane["status"] = "open"
    lane["kind"] = "plot_only"
    lane["claims"] = []
    lane["acceptance_gates"] = [""]
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    errors = release_gate_errors(manifest_path, docs_paths=())

    assert "sfincs2014_fig1_lhd_collisionality: status must be one of" in "\n".join(errors)
    assert "sfincs2014_fig1_lhd_collisionality: kind must be one of" in "\n".join(errors)
    assert (
        "sfincs2014_fig1_lhd_collisionality: field claims must be a non-empty list of strings"
        in errors
    )
    assert (
        "sfincs2014_fig1_lhd_collisionality: field acceptance_gates must be a non-empty list of strings"
        in errors
    )
