from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dkx.validation.release import check_release_gates_main
from dkx.validation.release import release_gate_errors


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _manifest_path() -> Path:
    return _repo_root() / "examples" / "publication_figures" / "validation_manifest.json"


def _manifest() -> list[dict[str, object]]:
    return json.loads(_manifest_path().read_text(encoding="utf-8"))


def test_release_gate_metadata_is_closed_for_current_release() -> None:
    assert release_gate_errors() == []


def test_release_gate_checker_cli_passes_current_manifest() -> None:
    assert check_release_gates_main([]) == 0


def test_release_gate_rejects_open_deferred_lane(tmp_path: Path) -> None:
    records = copy.deepcopy(_manifest())
    records[0]["release_gate"]["blocks_current_release"] = True  # type: ignore[index]
    records[4]["release_gate"]["claim_status"] = "bounded_proxy"  # type: ignore[index]
    records[4]["release_gate"]["closed_or_deferred_reason"] = "needs more work"  # type: ignore[index]
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    errors = release_gate_errors(manifest_path, docs_paths=())

    assert "sfincs2014_fig1_lhd_collisionality: no manifest lane may block the current release" in errors
    assert (
        "sfincs2014_fig3_high_collisionality_limit: deferred lanes must use "
        "claim_status='closed_deferred'"
    ) in errors
    assert (
        "sfincs2014_fig3_high_collisionality_limit: deferred lanes must record "
        "a closed post-release reason"
    ) in errors


def test_release_gate_rejects_malformed_implemented_lane(tmp_path: Path) -> None:
    records = copy.deepcopy(_manifest())
    record = records[0]
    record_id = str(record["id"])
    record["release_gate"] = {  # type: ignore[index]
        "claim_status": "closed_deferred",
        "blocks_current_release": "no",
        "evidence": "",
        "promotion_gate": " ",
    }
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")

    errors = release_gate_errors(manifest_path, docs_paths=())

    assert f"{record_id}: release_gate.blocks_current_release must be a bool" in errors
    assert f"{record_id}: release_gate.evidence must be a non-empty string" in errors
    assert f"{record_id}: release_gate.promotion_gate must be a non-empty string" in errors
    assert (
        f"{record_id}: implemented lanes must use an implemented claim status, "
        "got 'closed_deferred'"
    ) in errors


def test_release_gate_rejects_missing_gate_invalid_status_and_docs(tmp_path: Path) -> None:
    records = copy.deepcopy(_manifest())
    records[0].pop("release_gate")
    records[1]["release_gate"]["claim_status"] = "unknown"  # type: ignore[index]
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps(records), encoding="utf-8")
    docs_path = tmp_path / "validation_matrix.rst"
    docs_path.write_text("Release claim gate metadata\nrelease_ready\n", encoding="utf-8")

    errors = release_gate_errors(manifest_path, docs_paths=(docs_path,))

    assert "sfincs2014_fig1_lhd_collisionality: missing release_gate object" in errors
    assert any("release_gate.claim_status must be one of" in error for error in errors)
    assert any("implemented lanes must use an implemented claim status" in error for error in errors)
    assert f"{docs_path}: missing release-gate docs phrase 'regression_scaffold'" in errors


def test_release_gate_manifest_must_be_a_list(tmp_path: Path) -> None:
    manifest_path = tmp_path / "validation_manifest.json"
    manifest_path.write_text(json.dumps({"records": []}), encoding="utf-8")

    with pytest.raises(TypeError) as exc_info:
        release_gate_errors(manifest_path, docs_paths=())

    assert f"{manifest_path}: manifest must be a list" in str(exc_info.value)
