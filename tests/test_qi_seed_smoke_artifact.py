from __future__ import annotations

import json
from pathlib import Path

import pytest


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


def test_qi_seed_scale050_solver_matrix_records_failed_diagnostic_routes() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale050_solver_matrix_2026_05_12.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["artifact_kind"] == "qi_seed_solver_comparison_matrix"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50}
    assert payload["total_size_estimate"] == 70202
    assert payload["gates"]["passed"] is False
    assert payload["execution_summary"]["process_passed"] == 0
    assert payload["execution_summary"]["process_failed"] == 8

    runs = payload["runs"]
    assert set(runs) == {
        "auto_360s",
        "sparse_host_safe",
        "sparse_lsmr",
        "xblock_sparse_pc_gmres",
        "xblock_sparse_pc_gmres_initial_seed_optin",
        "xblock_sparse_pc_lgmres_optin",
        "xblock_sparse_pc_gmres_post_coarse1_optin",
        "xblock_sparse_pc_gmres_post_minres4_optin",
    }
    assert runs["auto_360s"]["timed_out"] is True
    assert any("explicit FP x-block seed" in event for event in runs["auto_360s"]["progress_events"])
    assert any("Host sparse factorization failed" in event for event in runs["sparse_host_safe"]["progress_events"])
    assert runs["sparse_lsmr"]["residual_ratio_or_last_reported"] > 1.0e5
    assert runs["xblock_sparse_pc_gmres"]["residual_ratio_or_last_reported"] > 1.0e5
    assert runs["xblock_sparse_pc_gmres_initial_seed_optin"]["elapsed_s"] < runs["xblock_sparse_pc_gmres"][
        "elapsed_s"
    ]
    assert any(
        "initial x-block seed rejected" in event
        for event in runs["xblock_sparse_pc_gmres_initial_seed_optin"]["progress_events"]
    )
    assert runs["xblock_sparse_pc_gmres_post_minres4_optin"]["elapsed_s"] < runs["xblock_sparse_pc_gmres"][
        "elapsed_s"
    ]
    assert runs["xblock_sparse_pc_gmres_post_minres4_optin"]["residual_ratio_or_last_reported"] > 1.0e5
    assert (
        0.0
        < runs["xblock_sparse_pc_gmres_post_minres4_optin"]["post_minres_relative_improvement"]
        < 1.0e-2
    )
    assert any(
        "post-minres improved residual" in event
        for event in runs["xblock_sparse_pc_gmres_post_minres4_optin"]["progress_events"]
    )
    assert runs["xblock_sparse_pc_gmres_post_coarse1_optin"]["elapsed_s"] < runs["xblock_sparse_pc_gmres"][
        "elapsed_s"
    ]
    assert runs["xblock_sparse_pc_gmres_post_coarse1_optin"]["residual_ratio_or_last_reported"] > 1.0e5
    assert (
        0.0
        < runs["xblock_sparse_pc_gmres_post_coarse1_optin"]["post_coarse_relative_improvement"]
        < 1.0e-2
    )
    assert runs["xblock_sparse_pc_gmres_post_coarse1_optin"]["post_coarse_direction_count"] == 10
    assert any(
        "post-coarse improved residual" in event
        for event in runs["xblock_sparse_pc_gmres_post_coarse1_optin"]["progress_events"]
    )
    assert runs["xblock_sparse_pc_lgmres_optin"]["elapsed_s"] > runs["xblock_sparse_pc_gmres"]["elapsed_s"]
    assert runs["xblock_sparse_pc_lgmres_optin"]["residual_ratio_or_last_reported"] > 1.0e5
    assert (
        runs["xblock_sparse_pc_lgmres_optin"]["lgmres_residual_before_gmres_fallback"]
        > runs["xblock_sparse_pc_gmres"]["residual_norm_or_last_reported"]
    )
    assert any(
        "lgmres residual" in event and "falling back to gmres" in event
        for event in runs["xblock_sparse_pc_lgmres_optin"]["progress_events"]
    )
    assert "closed on CPU and one GPU" in payload["conclusion"]["scale050_status"]
    assert (
        payload["conclusion"]["successor_cpu_artifact"]
        == "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json"
    )
    assert (
        payload["conclusion"]["successor_gpu_artifact"]
        == "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json"
    )


def test_qi_seed_scale050_xblock_lu_right_cpu_artifact_passes() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50}
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_residual_ratio"] < 0.05
    assert payload["execution_summary"]["max_elapsed_s"] < 30.0

    seed = payload["seeds"][0]
    assert seed["accepted_converged"] is True
    assert seed["converged"] is True
    assert seed["residual_norm"] < seed["residual_target"]
    assert seed["solver_elapsed_s"] < 15.0
    assert any("sparse_lu:" in event for event in seed["progress_events"])
    assert any("precondition_side=right" in event for event in seed["progress_events"]) or seed["solver_trace_exists"]

    trace_path = Path("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu_solver_trace.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    solver_metadata = trace["metadata"]["solver_metadata"]
    assert solver_metadata["precondition_side"] == "right"
    assert solver_metadata["default_right_preconditioned"] is True
    assert solver_metadata["default_short_restart_capped"] is False
    assert solver_metadata["gmres_restart"] == 80
    assert solver_metadata["iterations"] == 81
    assert solver_metadata["matvecs"] == 85
    assert solver_metadata["accepted_converged"] is True


def test_qi_seed_scale055_xblock_lu_right_cpu_artifact_passes() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale055_xblock_lu_right_cpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 29, "NX": 4, "NXI": 55}
    assert payload["active_size"] == 52637
    assert payload["total_size_estimate"] == 95702
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == ["cpu"]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["timed_out"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_residual_ratio"] < 0.01
    assert payload["execution_summary"]["max_elapsed_s"] < 30.0

    seed = payload["seeds"][0]
    assert seed["active_size"] == 52637.0
    assert seed["total_size"] == 95702.0
    assert seed["accepted_converged"] is True
    assert seed["converged"] is True
    assert seed["residual_norm"] < seed["residual_target"]
    assert seed["solve_method"] == "xblock_sparse_pc_gmres"
    assert any("sparse_lu: nnz=637603" in event for event in seed["progress_events"])


def test_qi_seed_scale055_adaptive_side_hard_cpu_seed_artifact_passes() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 29, "NX": 4, "NXI": 55}
    assert payload["active_size"] == 52637
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == ["cpu"]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["timed_out"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_residual_ratio"] < 0.01
    assert payload["execution_summary"]["max_elapsed_s"] < 60.0

    seed = payload["seeds"][0]
    assert seed["seed"] == 3
    assert seed["precondition_side"] == "left"
    assert seed["default_right_preconditioned"] is False
    assert seed["gmres_restart"] == 80
    assert seed["iterations"] == 720
    assert seed["matvecs"] == 731
    assert seed["residual_norm"] < seed["residual_target"]


def test_qi_seed_scale050_xblock_lu_right_gpu_artifact_passes() -> None:
    path = Path("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json")
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50}
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == ["gpu"]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_residual_ratio"] < 1.0
    assert payload["execution_summary"]["max_elapsed_s"] < 60.0

    seed = payload["seeds"][0]
    assert seed["backend"] == "gpu"
    assert seed["accepted_converged"] is True
    assert seed["residual_norm"] < seed["residual_target"]
    assert any("sparse_lu:" in event for event in seed["progress_events"])

    trace_path = Path("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu_solver_trace.json")
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    solver_metadata = trace["metadata"]["solver_metadata"]
    assert trace["backend"] == "gpu"
    assert solver_metadata["precondition_side"] == "right"
    assert solver_metadata["default_right_preconditioned"] is True
    assert solver_metadata["default_short_restart_capped"] is False
    assert solver_metadata["gmres_restart"] == 80
    assert solver_metadata["iterations"] == 69
    assert solver_metadata["matvecs"] == 72
    assert solver_metadata["accepted_converged"] is True


@pytest.mark.parametrize(
    ("artifact", "backend", "max_elapsed_s"),
    [
        ("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json", "cpu", 15.0),
        ("docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json", "gpu", 45.0),
    ],
)
def test_qi_seed_scale050_xblock_lu_right_multiseed5_artifacts_pass(
    artifact: str,
    backend: str,
    max_elapsed_s: float,
) -> None:
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 13, "NZETA": 27, "NX": 4, "NXI": 50}
    assert payload["public_cli_default_path"] is True
    assert payload["solve_method_request"] == "auto"
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == [backend]
    assert payload["execution_summary"]["attempted"] == 5
    assert payload["execution_summary"]["process_passed"] == 5
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["timed_out"] == 0
    assert payload["execution_summary"]["outputs_written"] == 5
    assert payload["execution_summary"]["solver_traces_written"] == 5
    assert payload["execution_summary"]["accepted_converged"] == 5
    assert payload["execution_summary"]["max_residual_ratio"] < 1.0
    assert payload["execution_summary"]["max_elapsed_s"] < max_elapsed_s

    seeds = payload["seeds"]
    assert len(seeds) == 5
    assert {seed["seed"] for seed in seeds} == {0, 1, 2, 3, 4}
    for seed in seeds:
        assert seed["backend"] == backend
        assert seed["accepted_converged"] is True
        assert seed["converged"] is True
        assert seed["residual_norm"] < seed["residual_target"]
        assert seed["solver_trace_exists"] is True


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
    assert current["artifact_count"] == len(payload["source_artifacts"]) == 15
    assert current["passing_artifact_count"] == 12
    assert current["nonpassing_artifact_count"] == 3
    assert current["checked_backends"] == ["cpu", "gpu"]
    assert current["max_checked_active_size"] == 52637
    assert current["max_checked_total_size"] == 95702
    assert current["largest_attempted_total_size"] == 95702
    assert current["largest_nonpassing_total_size"] == 95702
    assert current["max_checked_total_size_fraction"] < 0.10
    assert current["max_checked_per_axis_resolution_fraction"] == 0.5
    assert current["bounded_lane_completion_estimate_percent"] == 50.0
    assert current["completion_estimate_basis"] == "largest passing measured artifact only"
    assert current["production_total_size_uncovered_percent"] > 90.0

    source_paths = {artifact["path"] for artifact in payload["source_artifacts"]}
    assert {
        "docs/_static/qi_seed_robustness_smoke.json",
        "docs/_static/qi_seed_robustness_multiseed.json",
        "docs/_static/qi_seed_robustness_multiseed_gpu.json",
        "docs/_static/qi_seed_robustness_scale035_cpu_gpu.json",
        "docs/_static/qi_seed_robustness_multiseed5_cpu.json",
        "docs/_static/qi_seed_robustness_scale045_cpu_probe.json",
        "docs/_static/qi_seed_robustness_scale050_cpu_probe.json",
        "docs/_static/qi_seed_robustness_scale050_solver_matrix_2026_05_12.json",
        "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_cpu.json",
        "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_gpu.json",
        "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_cpu.json",
        "docs/_static/qi_seed_robustness_scale050_xblock_lu_right_multiseed5_gpu.json",
        "docs/_static/qi_seed_robustness_scale055_auto_cpu_blocker.json",
        "docs/_static/qi_seed_robustness_scale055_xblock_lu_right_cpu.json",
        "docs/_static/qi_seed_robustness_scale055_xblock_auto_side_seed3_cpu.json",
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
