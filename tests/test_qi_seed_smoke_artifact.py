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


def test_qi_seed_multiseed_artifact_records_passing_default_cli_run() -> None:
    path = Path("docs/_static/qi_seed_robustness_multiseed.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["lane"] == "qi_seed_robustness"
    assert payload["case_count"] == 3
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["gates"]["max_residual_ratio"] == 1.0
    assert payload["gates"]["require_converged"] is True
    assert payload["execution_summary"]["process_passed"] == 3
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["converged"] == 3
    assert payload["execution_summary"]["max_residual_ratio"] < 1.0
    assert len(payload["seeds"]) == 3
    assert {seed["seed"] for seed in payload["seeds"]} == {0, 1, 2}


def test_qi_seed_multiseed_gpu_artifact_records_passing_default_cli_run() -> None:
    path = Path("docs/_static/qi_seed_robustness_multiseed_gpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["lane"] == "qi_seed_robustness"
    assert payload["case_count"] == 3
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["gates"]["max_residual_ratio"] == 1.0
    assert payload["gates"]["require_converged"] is True
    assert payload["execution_summary"]["backends"] == ["gpu"]
    assert payload["execution_summary"]["process_passed"] == 3
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["converged"] == 3
    assert payload["execution_summary"]["max_residual_ratio"] < 1.0
    assert payload["execution_summary"]["timed_out"] == 0
    assert len(payload["seeds"]) == 3
    assert {seed["seed"] for seed in payload["seeds"]} == {0, 1, 2}


def test_qi_seed_scale035_cpu_gpu_artifact_records_host_sparse_gpu_fix() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale035_cpu_gpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["lane"] == "qi_seed_robustness"
    assert payload["active_size"] == 13169
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["runs"]["cpu_after_patch"]["converged"] is True
    assert payload["runs"]["gpu_after_patch"]["converged"] is True
    assert payload["runs"]["gpu_before_patch"]["process_passed"] is False
    assert payload["runs"]["gpu_before_patch"]["residual_ratio_from_stdout"] > 1.0
    assert payload["runs"]["gpu_after_patch"]["residual_ratio"] < 1.0
    assert payload["runs"]["gpu_after_patch"]["elapsed_s"] < payload["runs"]["gpu_before_patch"]["elapsed_s"]
    assert payload["solver_policy"]["accelerator_host_sparse_rescue_default_max_active_size"] == 30000
