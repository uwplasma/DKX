from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "scripts" / "audit_suite_runtime_drift.py"
    spec = importlib.util.spec_from_file_location("audit_suite_runtime_drift", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_audit_suite_runtime_drift_flags_only_cases_above_threshold(tmp_path: Path) -> None:
    mod = _load_module()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(
        json.dumps(
            [
                {"case": "slow_case", "jax_runtime_s": 2.0},
                {"case": "tiny_case", "jax_runtime_s": 0.2},
                {"case": "stable_case", "jax_runtime_s": 5.0},
            ]
        ),
        encoding="utf-8",
    )
    candidate.write_text(
        json.dumps(
            [
                {"case": "slow_case", "jax_runtime_s": 3.0},
                {"case": "tiny_case", "jax_runtime_s": 0.5},
                {"case": "stable_case", "jax_runtime_s": 5.4},
            ]
        ),
        encoding="utf-8",
    )

    flagged = mod.audit_suite_runtime_drift(
        baseline_report=baseline,
        candidate_report=candidate,
        threshold_ratio=1.25,
        min_baseline_runtime_s=1.0,
    )
    assert [(item.case, round(item.ratio, 2)) for item in flagged] == [("slow_case", 1.5)]


def test_audit_suite_runtime_drift_cli_can_fail(tmp_path: Path) -> None:
    mod = _load_module()
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text(json.dumps([{"case": "case", "jax_runtime_s": 1.0}]), encoding="utf-8")
    candidate.write_text(json.dumps([{"case": "case", "jax_runtime_s": 1.4}]), encoding="utf-8")
    rc = mod.main(
        [
            "--baseline-report",
            str(baseline),
            "--candidate-report",
            str(candidate),
            "--threshold-ratio",
            "1.25",
            "--fail-on-drift",
        ]
    )
    assert rc == 1
