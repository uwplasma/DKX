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


def test_qi_seed_multiseed5_cpu_artifact_records_passing_manifest_gate() -> None:
    path = Path("docs/_static/qi_seed_robustness_multiseed5_cpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["case_count"] == 5
    assert payload["public_cli_default_path"] is True
    assert payload["resolution"] == {"NTHETA": 7, "NZETA": 13, "NX": 4, "NXI": 25}
    assert payload["total_size_estimate"] == 9102

    gates = payload["gates"]
    assert gates["passed"] is True
    assert gates["max_residual_ratio"] == 1.0
    assert gates["require_converged"] is True

    summary = payload["execution_summary"]
    assert summary["attempted"] == 5
    assert summary["process_passed"] == 5
    assert summary["process_failed"] == 0
    assert summary["timed_out"] == 0
    assert summary["outputs_written"] == 5
    assert summary["solver_traces_written"] == 5
    assert summary["converged"] == 5
    assert summary["backends"] == ["cpu"]
    assert summary["solve_methods"] == ["auto"]
    assert summary["selected_paths"] == ["rhsmode1_solution"]
    assert summary["max_residual_ratio"] < 1.0

    seeds = payload["seeds"]
    assert {seed["seed"] for seed in seeds} == {0, 1, 2, 3, 4}
    assert all(seed["returncode"] == 0 for seed in seeds)
    assert all(seed["timed_out"] is False for seed in seeds)
    assert all(seed["output_exists"] is True for seed in seeds)
    assert all(seed["solver_trace_exists"] is True for seed in seeds)
    assert all(seed["backend"] == "cpu" for seed in seeds)
    assert all(seed["solve_method"] == "auto" for seed in seeds)
    assert all(seed["converged"] is True for seed in seeds)
    assert max(seed["residual_ratio"] for seed in seeds) == summary["max_residual_ratio"]


def test_qi_seed_scale050_cpu_probe_artifact_records_timeout_blocker() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale050_cpu_probe.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["case_count"] == 1
    assert payload["public_cli_default_path"] is True
    assert payload["resolution"] == {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50}
    assert payload["total_size_estimate"] == 70202
    assert "blocker evidence" in payload["evidence_note"]

    gates = payload["gates"]
    assert gates["passed"] is False
    assert {failure["reason"] for failure in gates["failures"]} == {"process_failed"}

    summary = payload["execution_summary"]
    assert summary["attempted"] == 1
    assert summary["process_passed"] == 0
    assert summary["process_failed"] == 1
    assert summary["timed_out"] == 1
    assert summary["outputs_written"] == 0
    assert summary["solver_traces_written"] == 0
    assert summary["converged"] == 0
    assert summary["backends"] == []
    assert summary["max_residual_ratio"] is None

    seed = payload["seeds"][0]
    assert seed["seed"] == 0
    assert seed["returncode"] == 124
    assert seed["timed_out"] is True
    assert seed["output_exists"] is False
    assert seed["solver_trace_exists"] is False
    assert seed["converged"] is None
    assert seed["residual_ratio"] is None


def test_qi_seed_scale045_cpu_probe_artifact_records_passing_larger_cpu_gate() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale045_cpu_probe.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 2
    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["public_cli_default_path"] is True
    assert payload["resolution"] == {"NTHETA": 11, "NZETA": 23, "NX": 4, "NXI": 45}
    assert payload["total_size_estimate"] == 45542
    assert payload["gates"]["passed"] is True

    summary = payload["execution_summary"]
    assert summary["attempted"] == 1
    assert summary["process_passed"] == 1
    assert summary["process_failed"] == 0
    assert summary["timed_out"] == 0
    assert summary["outputs_written"] == 1
    assert summary["solver_traces_written"] == 1
    assert summary["converged"] == 1
    assert summary["backends"] == ["cpu"]
    assert summary["max_residual_ratio"] < 1.0e-6
    assert summary["max_elapsed_s"] < 240.0


def test_qi_seed_evidence_manifest_tracks_production_gap_and_gates() -> None:
    path = Path("docs/_static/qi_seed_robustness_evidence_manifest.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["artifact_kind"] == "qi_seed_production_gate_manifest"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["release_gate"] == "bounded_proxy"
    assert payload["production_target"]["resolution"] == {
        "NTHETA": 25,
        "NZETA": 51,
        "NX": 8,
        "NXI": 100,
    }
    assert payload["production_target"]["total_size_estimate"] == 1020002
    assert payload["production_target"]["seed_count"] == 5
    assert payload["production_target"]["required_backends"] == ["cpu", "gpu"]

    current = payload["current_evidence"]
    assert current["artifact_count"] == len(payload["source_artifacts"]) == 7
    assert current["passing_artifact_count"] == 6
    assert current["nonpassing_artifact_count"] == 1
    assert current["checked_backends"] == ["cpu", "gpu"]
    assert current["max_checked_active_size"] == 13169
    assert current["max_checked_total_size"] == 45542
    assert current["largest_attempted_total_size"] == 70202
    assert current["largest_nonpassing_total_size"] == 70202
    assert current["max_checked_total_size_fraction"] < 0.05
    assert current["max_checked_per_axis_resolution_fraction"] == 0.44
    assert current["bounded_lane_completion_estimate_percent"] == 44.0
    assert current["completion_estimate_basis"] == "largest passing measured artifact only"
    assert current["production_total_size_uncovered_percent"] > 95.0

    source_paths = {artifact["path"] for artifact in payload["source_artifacts"]}
    assert {
        "docs/_static/qi_seed_robustness_smoke.json",
        "docs/_static/qi_seed_robustness_multiseed.json",
        "docs/_static/qi_seed_robustness_multiseed_gpu.json",
        "docs/_static/qi_seed_robustness_scale035_cpu_gpu.json",
        "docs/_static/qi_seed_robustness_multiseed5_cpu.json",
        "docs/_static/qi_seed_robustness_scale045_cpu_probe.json",
        "docs/_static/qi_seed_robustness_scale050_cpu_probe.json",
    } == source_paths

    gates = payload["acceptance_gates"]
    assert gates["public_cli_default_path"] is True
    assert gates["solve_method"] == "auto"
    assert gates["process_failed"] == 0
    assert gates["timed_out"] == 0
    assert gates["outputs_written"] == 5
    assert gates["solver_traces_written"] == 5
    assert gates["converged"] == 5
    assert gates["max_residual_ratio"] == 1.0
    assert gates["required_backends"] == ["cpu", "gpu"]

    commands = payload["regeneration_commands"]
    assert "--summarize-artifacts-only" in commands["refresh_evidence_manifest"]
    assert "JAX_PLATFORM_NAME=cpu" in commands["production_cpu_seed_ladder"]
    assert "JAX_PLATFORM_NAME=gpu" in commands["production_gpu0_seed_ladder"]
    assert "--resolution-scale 1.0" in commands["production_cpu_seed_ladder"]
    assert "--resolution-scale 1.0" in commands["production_gpu0_seed_ladder"]
