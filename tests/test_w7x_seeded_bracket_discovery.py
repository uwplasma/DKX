from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation/w7x_seeded_bracket_discovery_v1.json"
SCRIPT = ROOT / "tools/paper_benchmarks/audit_w7x_seeded_bracket_discovery.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_w7x_seeded_bracket_discovery", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_seeded_bracket_discovery_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["topology"] == [1, 3]
    assert report["endpoint_count"] == 4
    assert report["admitted_grid_promotion_ready"] is False


def test_every_candidate_interval_retains_a_strict_sign_change() -> None:
    endpoints = _payload()["seeded_replay"]["endpoints"]
    assert len(endpoints) == 4
    assert all(
        endpoint["left_current_A_m2"] * endpoint["right_current_A_m2"] < 0.0
        for endpoint in endpoints
    )


def test_discovery_does_not_promote_the_admitted_grid() -> None:
    payload = _payload()
    assert payload["outcome"]["discovery_brackets_replayed"] is True
    assert payload["outcome"]["admitted_grid_promotion_ready"] is False
    assert "admitted-grid ambipolar roots" in payload["claim_exclusions"]
