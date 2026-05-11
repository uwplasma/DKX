from __future__ import annotations

import json
from pathlib import Path


def test_qi_seed_smoke_artifact_records_passing_default_cli_run() -> None:
    path = Path("docs/_static/qi_seed_robustness_smoke.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["lane"] == "qi_seed_robustness"
    assert payload["passed"] == 1
    assert payload["failed"] == 0
    assert payload["public_cli_default_path"] is True
    assert payload["output_exists"] is True

    trace = payload["solver_trace_summary"]
    assert trace["readable"] is True
    assert trace["converged"] is True
    assert trace["solve_method"] == "dense"
    assert trace["residual_norm"] < trace["residual_target"]
    assert trace["residual_ratio"] < 1.0
