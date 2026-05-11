from __future__ import annotations

import copy
import json
from pathlib import Path

from scripts.check_release_gates import main as check_release_gates_main
from scripts.check_release_gates import release_gate_errors


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
