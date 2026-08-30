from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation/w7x_admitted_grid_uniform_probe_no_go_v1.json"


def test_uniform_admitted_grid_no_go_is_bounded_and_claim_scoped() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["schema"] == "dkx.w7x_admitted_grid_uniform_probe_no_go.v1"
    assert payload["preflight"] == {
        "hierarchy_points": 33,
        "max_evaluations_per_surface": 825,
        "max_profile_evaluations": 1650,
        "retained_profile_bytes": 1531200,
    }
    measurement = payload["measurement"]
    assert measurement["completed_surfaces"] == 0
    assert measurement["result_written"] is False
    assert measurement["wall_seconds"] == 2551.33
    assert measurement["maximum_rss_bytes"] == 10205478912
    assert measurement["peak_process_footprint_bytes"] == 21883225584

    route = payload["route_diagnosis"]
    assert route["reusable_dense_coarse_bands_bytes"] > 3 * 24 * 1024**3
    assert route["schur_lu_factors_bytes"] > 24 * 1024**3
    assert route["checkpointed_dense_factors_per_subsystem_bytes"] < 1024**3

    outcome = payload["outcome"]
    assert outcome["uniform_high_grid_launch_admitted"] is False
    assert outcome["numerical_failure_claimed"] is False
    assert "no-root evidence" in payload["claim_exclusions"]
    assert len(payload["source"]["input_sha256"]) == 64
    assert len(payload["source"]["log_sha256"]) == 64
