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


@pytest.mark.parametrize(
    ("artifact", "backend", "max_elapsed_s"),
    [
        ("docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json", "cpu", 60.0),
        ("docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json", "gpu", 240.0),
    ],
)
def test_qi_seed_scale055_adaptive_side_multiseed5_artifacts_pass(
    artifact: str,
    backend: str,
    max_elapsed_s: float,
) -> None:
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 29, "NX": 4, "NXI": 55}
    assert payload["active_size"] == 52637
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == [backend]
    assert payload["execution_summary"]["attempted"] == 5
    assert payload["execution_summary"]["process_passed"] == 5
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["timed_out"] == 0
    assert payload["execution_summary"]["outputs_written"] == 5
    assert payload["execution_summary"]["solver_traces_written"] == 5
    assert payload["execution_summary"]["accepted_converged"] == 5
    assert payload["execution_summary"]["max_residual_ratio"] < 0.01
    assert payload["execution_summary"]["max_elapsed_s"] < max_elapsed_s

    seeds = payload["seeds"]
    assert {seed["seed"] for seed in seeds} == {0, 1, 2, 3, 4}
    for seed in seeds:
        assert seed["backend"] == backend
        assert seed["precondition_side"] == "left"
        assert seed["default_right_preconditioned"] is False
        assert seed["gmres_restart"] == 80
        assert seed["accepted_converged"] is True
        assert seed["converged"] is True
        assert seed["residual_norm"] < seed["residual_target"]
        assert seed["solver_trace_exists"] is True


@pytest.mark.parametrize(
    ("artifact", "backend", "max_elapsed_s"),
    [
        ("docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json", "cpu", 60.0),
        ("docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json", "gpu", 180.0),
    ],
)
def test_qi_seed_scale060_adaptive_side_seed0_artifacts_pass(
    artifact: str,
    backend: str,
    max_elapsed_s: float,
) -> None:
    payload = json.loads(Path(artifact).read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NX": 5, "NXI": 60}
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    assert payload["public_cli_default_path"] is True
    assert payload["gates"]["passed"] is True
    assert payload["execution_summary"]["backends"] == [backend]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["process_failed"] == 0
    assert payload["execution_summary"]["timed_out"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_residual_ratio"] < 0.01
    assert payload["execution_summary"]["max_elapsed_s"] < max_elapsed_s

    seed = payload["seeds"][0]
    assert seed["backend"] == backend
    assert seed["precondition_side"] == "left"
    assert seed["default_right_preconditioned"] is False
    assert seed["accepted_converged"] is True
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


def test_qi_seed_scale060_rejected_solver_probe_artifact_records_blockers() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_rejected_solver_probe_summary"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NXI": 60, "NX": 5}
    assert payload["active_size"] == 81377
    rejected = payload["rejected_probes"]
    assert any(probe["backend"] == "gpu" and probe["outcome"] == "timeout" for probe in rejected)
    assert any("JAX-factor" in probe["policy"] for probe in rejected)
    assert "default-off blocker evidence" in payload["code_rejected"][0]["reason"]


def test_qi_seed_scale060_global_coupling_rejected_artifact_records_blockers() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_rejected_solver_probe_summary"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NXI": 60, "NX": 5}
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    assert payload["conclusion"]["defaults_changed"] is False
    assert payload["conclusion"]["hard_seed_closed"] is False
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING" in payload["implemented_opt_in_controls"]
    gpu_probes = [probe for probe in payload["probes"] if probe.get("backend") == "gpu"]
    assert gpu_probes
    assert all(probe["timed_out"] is True for probe in gpu_probes)
    assert any("keep_global_coupling" in " ".join(probe["observed_progress"]) for probe in gpu_probes)


def test_qi_seed_scale060_device_krylov_rejected_artifact_records_blockers() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_rejected_solver_probe_summary"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NXI": 60, "NX": 5}
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    assert payload["conclusion"]["defaults_changed"] is False
    assert payload["conclusion"]["hard_seed_closed"] is False
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV=fgmres" in payload["implemented_opt_in_controls"]
    probes = payload["probes"]
    assert {probe["name"] for probe in probes} == {
        "gpu_device_fgmres_right_gc24",
        "gpu_device_gmresjax_left_gc24",
    }
    assert all(probe["timed_out"] is False for probe in probes)
    assert all(probe["accepted_converged"] is False for probe in probes)
    assert any("method=fgmres_jax" in " ".join(probe["observed_progress"]) for probe in probes)


def test_qi_seed_scale060_device_operator_rejected_artifact_records_blocker() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_device_operator_rejected_probe"
    assert payload["lane"] == "qi_seed_robustness"
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NXI": 60, "NX": 5}
    assert payload["active_size"] == 81377
    assert payload["total_size"] == 139502
    probe = payload["device_operator_probe"]
    assert probe["backend"] == "gpu"
    assert probe["operator_device_resident"] is True
    assert probe["assembled_operator_nnz"] == 2884321
    assert probe["assembled_operator_setup_s"] == pytest.approx(83.587)
    assert probe["matvec_progress"][-1] == {"matvecs": 400, "elapsed_s": pytest.approx(505.695)}
    assert probe["output_written"] is False
    assert probe["solver_trace_written"] is False
    assert probe["promotion_decision"] == "rejected_timeout_after_device_operator_build"
    short_recurrence = payload["bicgstab_device_probe"]
    assert short_recurrence["backend"] == "gpu"
    assert short_recurrence["operator_device_resident"] is True
    assert short_recurrence["assembled_operator_nnz"] == 2884321
    assert short_recurrence["assembled_operator_setup_s"] == pytest.approx(84.767)
    assert short_recurrence["output_written"] is False
    assert short_recurrence["solver_trace_written"] is False
    assert short_recurrence["promotion_decision"] == "rejected_divergent_short_recurrence"
    assert short_recurrence["reported_residual_norm"] > 1.0e100
    assert short_recurrence["memory_reduction_vs_device_fgmres_percent"] > 60.0
    tfqmr = payload["tfqmr_device_probe"]
    assert tfqmr["backend"] == "gpu"
    assert tfqmr["operator_device_resident"] is True
    assert tfqmr["assembled_operator_nnz"] == 2884321
    assert tfqmr["krylov_method"] == "tfqmr_jax"
    assert tfqmr["residual_replacement_interval"] == 0
    assert tfqmr["reported_iterations"] == 401
    assert tfqmr["output_written"] is False
    assert tfqmr["promotion_decision"] == "rejected_divergent_tfqmr"
    assert tfqmr["reported_residual_norm"] > 1.0e100
    tfqmr_replaced = payload["tfqmr_replacement20_device_probe"]
    assert tfqmr_replaced["backend"] == "gpu"
    assert tfqmr_replaced["operator_device_resident"] is True
    assert tfqmr_replaced["krylov_method"] == "tfqmr_jax"
    assert tfqmr_replaced["residual_replacement_interval"] == 20
    assert tfqmr_replaced["reported_iterations"] == 21
    assert tfqmr_replaced["outer_elapsed_s"] < tfqmr["outer_elapsed_s"]
    assert tfqmr_replaced["output_written"] is False
    assert tfqmr_replaced["promotion_decision"] == "rejected_divergent_tfqmr_residual_replacement"
    assert tfqmr_replaced["reported_residual_norm"] > 1.0e100
    qr_global = payload["device_global_qr_fgmres_probe"]
    assert qr_global["backend"] == "gpu"
    assert qr_global["operator_device_resident"] is True
    assert qr_global["coarse_solver"] == "qr"
    assert qr_global["global_coupling_rank"] == 20
    assert qr_global["global_coupling_basis_size"] == 20
    assert qr_global["reported_matvecs"] == 475
    assert qr_global["timed_out"] is True
    assert qr_global["output_written"] is False
    assert qr_global["promotion_decision"] == "rejected_timeout_device_global_qr_fgmres"
    qr_identity = payload["device_global_qr_identity_probe_coarse_probe"]
    assert qr_identity["backend"] == "gpu"
    assert qr_identity["operator_device_resident"] is True
    assert qr_identity["coarse_solver"] == "qr"
    assert qr_identity["global_coupling_smoother"] == "identity"
    assert qr_identity["global_coupling_rank"] == 30
    assert qr_identity["probe_coarse_seed_initialized"] is True
    assert qr_identity["probe_coarse_residual_after"] < qr_identity["probe_coarse_residual_before"]
    assert qr_identity["reported_matvecs"] == 400
    assert qr_identity["timed_out"] is True
    assert qr_identity["output_written"] is False
    assert qr_identity["promotion_decision"] == "rejected_timeout_identity_global_qr_probe_coarse_fgmres"
    restarted = payload["fgmres_restart20_device_probe"]
    assert restarted["backend"] == "gpu"
    assert restarted["operator_device_resident"] is True
    assert restarted["assembled_operator_nnz"] == 2884321
    assert restarted["assembled_operator_setup_s"] == pytest.approx(83.809)
    assert restarted["krylov_method"] == "fgmres_jax"
    assert restarted["restart"] == 20
    assert restarted["reported_matvecs"] == 500
    assert restarted["output_written"] is False
    assert restarted["solver_trace_written"] is False
    assert restarted["promotion_decision"] == "rejected_timeout_restart20_host_memory_regression"
    assert restarted["memory_change_vs_device_fgmres_percent"] > 0.0
    synchronized = payload["fgmres_restart20_cycle_sync_device_probe"]
    assert synchronized["backend"] == "gpu"
    assert synchronized["operator_device_resident"] is True
    assert synchronized["block_between_cycles"] is True
    assert synchronized["assembled_operator_nnz"] == 2884321
    assert synchronized["assembled_operator_setup_s"] == pytest.approx(82.525)
    assert synchronized["krylov_method"] == "fgmres_jax"
    assert synchronized["restart"] == 20
    assert synchronized["reported_matvecs"] == 500
    assert synchronized["output_written"] is False
    assert synchronized["solver_trace_written"] is False
    assert synchronized["promotion_decision"] == "rejected_timeout_cycle_sync_no_memory_gain"
    assert synchronized["memory_change_vs_restart20_percent"] > 0.0
    compact = payload["device_compact_csr_exact_xblock_probe"]
    assert compact["backend"] == "gpu"
    assert compact["factor_format"] == "compact_csr"
    assert compact["krylov_method"] == "gmres_jax"
    assert compact["precondition_side"] == "left"
    assert compact["xblock_lu_max"] == 30000
    assert compact["xblock_factor_row_cap"] == 0
    assert compact["lower_factor_nnz"] == 110063331
    assert compact["upper_factor_nnz"] == 119280365
    assert compact["factor_nbytes_estimate"] == 2755472392
    assert compact["assembled_operator_reuse"] == "requested_but_rejected_max_colors_512"
    assert compact["timed_out"] is True
    assert compact["output_written"] is False
    assert compact["solver_trace_written"] is False
    assert compact["reported_residual_norm"] is None
    assert compact["promotion_decision"] == "rejected"
    assert "memory improvement, not QI closure" in compact["conclusion"]
    assert payload["execution_summary"]["timed_out"] == 1
    assert any("assembled operator built location=device" in line for line in payload["observed_progress"])
    assert any("method=bicgstab_jax" in line for line in payload["observed_progress"])
    assert any("method=tfqmr_jax" in line for line in payload["observed_progress"])
    assert any("coarse_solver=qr" in line for line in payload["observed_progress"])
    assert any("cycle-boundary synchronization enabled" in line for line in payload["observed_progress"])


def test_qi_seed_scale060_no_lgmres_gpu_timeout_artifact_records_blocker() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["gates"]["passed"] is False
    assert payload["resolution"] == {"NTHETA": 15, "NZETA": 31, "NXI": 60, "NX": 5}
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    seed = payload["seeds"][0]
    assert seed["timed_out"] is True
    assert seed["heartbeat_count"] == 31
    assert seed["last_matvecs"] == 900
    assert seed["last_matvec_elapsed_s"] == pytest.approx(412.719)
    assert seed["output_exists"] is False
    assert seed["solver_trace_exists"] is False
    assert any("side probe switch side=left->right method=gmres->gmres" in line for line in seed["progress_events"])
    assert any("probe-coarse improved seed residual" in line for line in seed["progress_events"])


def test_qi_seed_scale060_angular_residual_cpu_artifact_passes() -> None:
    payload = json.loads(
        Path(
            "docs/_static/"
            "qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json"
        ).read_text(encoding="utf-8")
    )

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["gates"]["passed"] is True
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    assert payload["execution_summary"]["backends"] == ["cpu"]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_elapsed_s"] < 180.0
    assert payload["execution_summary"]["max_residual_ratio"] < 0.003
    seed = payload["seeds"][0]
    assert seed["last_matvecs"] == 2024
    assert seed["xblock_side_probe_selected_method"] == "lgmres"
    assert seed["xblock_lgmres_rescue_status"] == "used"
    assert any("probe-coarse improved seed residual" in line for line in seed["progress_events"])


def test_qi_seed_scale060_device_host_fallback_cpu_artifact_passes() -> None:
    payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json").read_text(
            encoding="utf-8"
        )
    )

    assert payload["artifact_kind"] == "qi_seed_execution_summary"
    assert payload["gates"]["passed"] is True
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502
    assert payload["execution_summary"]["backends"] == ["cpu"]
    assert payload["execution_summary"]["process_passed"] == 1
    assert payload["execution_summary"]["accepted_converged"] == 1
    assert payload["execution_summary"]["max_elapsed_s"] < 180.0
    assert payload["execution_summary"]["max_residual_ratio"] < 0.01
    seed = payload["seeds"][0]
    assert seed["last_matvecs"] == 1789
    assert seed["xblock_side_probe_selected_method"] == "lgmres"
    assert seed["xblock_lgmres_rescue_status"] == "used"
    assert seed["xblock_device_host_fallback_used"] is True
    assert seed["xblock_device_host_fallback_reason"] == "large-qi-full-fp-3d"
    assert seed["xblock_device_host_fallback_requested_method"] == "gmres_jax"
    assert seed["xblock_device_host_fallback_effective_krylov_env_value"] == "auto"
    assert seed["xblock_device_host_fallback_non_autodiff"] is True
    assert any("using non-autodiff host x-block fallback" in line for line in seed["progress_events"])


def test_qi_seed_scale060_lower_fill_artifacts_record_memory_convergence_tradeoff() -> None:
    for artifact in (
        "docs/_static/qi_seed_robustness_scale060_lower_fill_seed3_cpu_rejected_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_lower_fill8_seed3_cpu_rejected_2026_05_14.json",
    ):
        payload = json.loads(Path(artifact).read_text(encoding="utf-8"))
        assert payload["artifact_kind"] == "qi_seed_execution_summary"
        assert payload["gates"]["passed"] is False
        assert payload["active_size"] == 81377
        assert payload["total_size_estimate"] == 139502
        seed = payload["seeds"][0]
        assert seed["returncode"] == 2
        assert seed["output_exists"] is False
        assert seed["solver_trace_exists"] is False
        assert seed["last_matvecs"] == 14835
        assert seed["xblock_lgmres_rescue_status"] == "used"
        assert any("sparse_ilu:" in line for line in seed["progress_events"])
        assert any("Refusing to write nonconverged" in line for line in seed["progress_events"])


def test_qi_seed_scale060_enriched_and_device_krylov_artifacts_remain_rejected() -> None:
    enriched_cpu = json.loads(
        Path(
            "docs/_static/qi_seed_robustness_scale060_enriched_qi_coarse_seed3_cpu_rejected_2026_05_14.json"
        ).read_text(encoding="utf-8")
    )
    assert enriched_cpu["gates"]["passed"] is False
    assert "performance_regression" in enriched_cpu["performance_rejection"]
    assert enriched_cpu["execution_summary"]["process_passed"] == 1
    assert enriched_cpu["execution_summary"]["max_elapsed_s"] > 250.0
    assert enriched_cpu["seeds"][0]["last_matvecs"] == 3298

    gpu_no_lgmres = json.loads(
        Path(
            "docs/_static/"
            "qi_seed_robustness_scale060_enriched_angular_seed3_gpu0_no_lgmres_timeout_2026_05_14.json"
        ).read_text(encoding="utf-8")
    )
    assert gpu_no_lgmres["gates"]["passed"] is False
    assert gpu_no_lgmres["active_size"] == 81377
    assert gpu_no_lgmres["seeds"][0]["timed_out"] is True
    assert gpu_no_lgmres["seeds"][0]["last_matvecs"] == 925
    assert gpu_no_lgmres["seeds"][0]["xblock_lgmres_rescue_status"] == "not_selected"

    device_krylov = json.loads(
        Path(
            "docs/_static/qi_seed_robustness_scale060_device_krylov_enriched_seed3_gpu1_timeout_2026_05_14.json"
        ).read_text(encoding="utf-8")
    )
    assert device_krylov["gates"]["passed"] is False
    assert device_krylov["active_size"] == 81377
    assert device_krylov["seeds"][0]["timed_out"] is True
    assert device_krylov["seeds"][0]["last_matvecs"] is None
    assert any("QI coarse seed improved residual" in line for line in device_krylov["seeds"][0]["progress_events"])


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
    assert current["artifact_count"] == len(payload["source_artifacts"]) == 103
    assert current["passing_artifact_count"] == 32
    assert current["nonpassing_artifact_count"] == 71
    assert current["checked_backends"] == ["cpu", "gpu"]
    assert current["max_checked_active_size"] == 81377
    assert current["max_checked_total_size"] == 139502
    assert current["largest_attempted_total_size"] == 139502
    assert current["largest_nonpassing_total_size"] == 139502
    assert current["max_checked_total_size_fraction"] > 0.13
    assert current["max_checked_per_axis_resolution_fraction"] == 0.6
    assert current["bounded_lane_completion_estimate_percent"] == 60.0
    assert current["completion_estimate_basis"] == "largest passing measured artifact only"
    assert 85.0 < current["production_total_size_uncovered_percent"] < 90.0

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
        "docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_cpu.json",
        "docs/_static/qi_seed_robustness_scale055_xblock_auto_side_multiseed5_gpu.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_cpu.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_auto_side_seed0_gpu.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_multiseed5_cpu.json",
        "docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_cpu.json",
        "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_early_qi_skipstrong_skipglobal_seed3_cpu_2026_05_19.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_qi_two_level_smoothed_load_seed3_cpu_2026_05_16.json",
        "docs/_static/qi_seed_robustness_scale060_moment_schur_probe_seed3_cpu_2026_05_16.json",
        "docs/_static/qi_seed_robustness_scale060_qi_deflated_seed3_cpu_2026_05_16.json",
        "docs/_static/qi_seed_robustness_scale060_qi_deflated_cycles8_seed3_cpu_2026_05_17.json",
        "docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_seed3_cpu_2026_05_17.json",
        "docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_seed3_gpu0_2026_05_17.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_deflated_minres8_lgmres_forced_seed3_gpu0_2026_05_17.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_qi_device_preconditioner_seed3_cpu_2026_05_19.json",
        "docs/_static/qi_seed_robustness_scale060_qi_device_matrixfree_seed3_cpu_2026_05_19.json",
        "docs/_static/qi_seed_robustness_scale060_enriched_qi_coarse_seed3_cpu_rejected_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_lower_fill_seed3_cpu_rejected_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_lower_fill8_seed3_cpu_rejected_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_gpu0_timeout.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu_timeout.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_right_gmres_seed3_gpu_timeout.json",
        "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_enriched_angular_seed3_gpu0_no_lgmres_timeout_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_device_krylov_enriched_seed3_gpu1_timeout_2026_05_14.json",
        "docs/_static/qi_seed_robustness_scale060_device_krylov_skip_side_probe_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_krylov_compact_no_moment_seed3_gpu1_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_krylov_compact_right_restart20_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_restart20_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_restart4_diag_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_factor_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_exact_lu_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_exact_cap16_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_left_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_left_gmres_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_galerkin_failclosed_seed3_gpu0_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_galerkin_forced_xblock_seed3_gpu1_2026_05_15.json",
        "docs/_static/qi_seed_robustness_scale060_xblock_lgmres_rescue_seed3_gpu0_2026_05_15_retry.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_early_seed3_gpu1_reduced_not_output_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_xblock_hostfallback_seed3_gpu1_timeout_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_rank48_depth2_seed3_gpu1_timeout_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_minres_cycles4_rank27_seed3_gpu0_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_fixed_cycles4_rank27_seed3_gpu0_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_krylov_rank27_seed3_gpu0_timeout_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_recycle_cycles2_rank32_seed3_gpu0_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_mfminres_sweeps2_seed3_gpu1_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_mfminres_sweeps8_seed3_gpu1_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_sweeps1_seed3_cpu_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_cpu_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_cpu_2026_05_19.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups48_sweeps2_cycles8_seed3_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups48_sweeps4_cycles12_minres_seed3_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_operator_krylov_depth64_blockminres_hybrid_seed3_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_gpu0_public_auto_after_transpose_fixes_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_operator_krylov_device_qi_gpu0_fgmres_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_operator_krylov_multilevel_device_qi_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_operator_krylov_pitch_multilevel_device_qi_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_current_constraint_device_qi_gpu0_2026_05_20.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_augmented_krylov_device_qi_cpu_2026_05_20.json",
        "docs/_static/qi_seed_robustness_scale060_augmented_krylov_device_qi_gpu0_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_recycled_augmented_deep_device_qi_gpu0_2026_05_20.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_coarse_residual_device_qi_cpu_2026_05_20.json",
        "docs/_static/qi_seed_robustness_scale060_residual_snapshot_device_qi_cpu_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_residual_snapshot_equation_device_qi_cpu_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_global_moment_closure_device_qi_cpu_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_global_moment_closure_device_qi_gpu0_2026_05_20.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_residual_galerkin_device_qi_cpu_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_residual_galerkin_device_qi_gpu1_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_phase_space_coarse_reuse_device_qi_gpu0.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_block_schur_device_qi_cpu_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_block_schur_bestof_device_qi_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_adaptive_residual_device_qi_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_composite_closure_device_qi_gpu1_2026_05_20.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_adjoint_krylov_device_qi_cpu_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_operator_krylov_composite_device_qi_gpu0_2026_05_20.json"
        ),
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_device_global_coupling_gpu0_2026_05_20.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_rankdef_schur_gpu0_2026_05_20.json",
        (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_krylov_nohost_recycle_seed3_gpu0_2026_05_19.json"
        ),
        "docs/_static/qi_seed_robustness_scale060_gpu_rejected_solver_probes_2026_05_13.json",
        "docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json",
        "docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json",
        "docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json",
    } == source_paths

    recycle = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_recycle_cycles2_rank32_seed3_gpu0_2026_05_19.json"
    )
    assert recycle["passed"] is False
    assert recycle["active_size"] == 81377
    assert recycle["total_size"] == 139502
    mfminres = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_mfminres_sweeps2_seed3_gpu1_2026_05_19.json"
    )
    assert mfminres["passed"] is False
    assert mfminres["active_size"] == 81377
    assert mfminres["total_size"] == 139502
    assert mfminres["max_elapsed_s"] < 90.0
    assert mfminres["last_reported_residual_norm"] < recycle["last_reported_residual_norm"]
    mfminres8 = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_mfminres_sweeps8_seed3_gpu1_2026_05_19.json"
    )
    assert mfminres8["passed"] is False
    assert mfminres8["active_size"] == 81377
    assert mfminres8["total_size"] == 139502
    assert mfminres8["max_elapsed_s"] < 90.0
    assert mfminres8["last_reported_residual_norm"] < mfminres["last_reported_residual_norm"]
    blockminres = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_blockminres_sweeps1_seed3_cpu_2026_05_19.json"
    )
    assert blockminres["passed"] is True
    assert blockminres["active_size"] == 81377
    assert blockminres["total_size"] == 139502
    assert blockminres["max_elapsed_s"] < 320.0
    assert blockminres["max_residual_ratio"] < 30.0
    blockminres_groups32 = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_cpu_2026_05_19.json"
    )
    assert blockminres_groups32["passed"] is True
    assert blockminres_groups32["active_size"] == 81377
    assert blockminres_groups32["total_size"] == 139502
    assert blockminres_groups32["max_elapsed_s"] < blockminres["max_elapsed_s"]
    assert blockminres_groups32["max_residual_ratio"] == pytest.approx(blockminres["max_residual_ratio"])
    blockminres_hybrid = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_cpu_2026_05_19.json"
        )
    )
    assert blockminres_hybrid["passed"] is True
    assert blockminres_hybrid["active_size"] == 81377
    assert blockminres_hybrid["total_size"] == 139502
    assert blockminres_hybrid["max_elapsed_s"] > blockminres_groups32["max_elapsed_s"]
    assert blockminres_hybrid["max_residual_ratio"] == pytest.approx(blockminres_groups32["max_residual_ratio"])
    blockminres_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_groups32_sweeps1_seed3_gpu0_2026_05_20.json"
        )
    )
    blockminres_hybrid_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups32_sweeps1_seed3_gpu0_2026_05_20.json"
        )
    )
    blockminres_hybrid_gpu_deeper = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_blockminres_hybrid_groups48_sweeps4_cycles12_minres_seed3_gpu0_2026_05_20.json"
        )
    )
    assert blockminres_gpu["passed"] is False
    assert blockminres_gpu["active_size"] == 81377
    assert blockminres_gpu["total_size"] == 139502
    assert blockminres_gpu["last_reported_residual_norm"] == pytest.approx(7.004256e-07)
    assert blockminres_hybrid_gpu["last_reported_residual_norm"] < blockminres_gpu[
        "last_reported_residual_norm"
    ]
    assert blockminres_hybrid_gpu_deeper["last_reported_residual_norm"] < blockminres_hybrid_gpu[
        "last_reported_residual_norm"
    ]
    assert blockminres_hybrid_gpu_deeper["last_reported_residual_norm"] == pytest.approx(4.689216e-07)
    operator_krylov_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_qi_device_operator_krylov_depth64_blockminres_hybrid_seed3_gpu0_2026_05_20.json"
        )
    )
    assert operator_krylov_gpu["passed"] is False
    assert operator_krylov_gpu["active_size"] == 81377
    assert operator_krylov_gpu["total_size"] == 139502
    assert operator_krylov_gpu["last_reported_residual_norm"] == pytest.approx(3.626870e-07)
    assert operator_krylov_gpu["last_reported_residual_norm"] < blockminres_hybrid_gpu_deeper[
        "last_reported_residual_norm"
    ]
    pitch_multilevel_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_operator_krylov_pitch_multilevel_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert pitch_multilevel_gpu["passed"] is False
    assert pitch_multilevel_gpu["last_reported_residual_norm"] == pytest.approx(2.306911e-05)
    assert pitch_multilevel_gpu["max_elapsed_s"] < 300.0
    current_constraint_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_current_constraint_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert current_constraint_gpu["passed"] is False
    assert current_constraint_gpu["last_reported_residual_norm"] == pytest.approx(2.339521e-05)
    assert current_constraint_gpu["last_reported_residual_norm"] > pitch_multilevel_gpu[
        "last_reported_residual_norm"
    ]
    assert current_constraint_gpu["max_elapsed_s"] < 320.0
    augmented_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_augmented_krylov_device_qi_cpu_2026_05_20.json"
    )
    augmented_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_augmented_krylov_device_qi_gpu0_2026_05_20.json"
    )
    assert augmented_cpu["passed"] is False
    assert augmented_gpu["passed"] is False
    assert augmented_cpu["evidence_classes"] == ["device_qi_augmented_krylov_coarse_reuse"]
    assert augmented_gpu["evidence_classes"] == ["device_qi_augmented_krylov_coarse_reuse"]
    assert "observed_augmented_krylov" in augmented_cpu["evidence_tags"]
    assert "observed_augmented_krylov" in augmented_gpu["evidence_tags"]
    assert augmented_cpu["last_reported_residual_norm"] == pytest.approx(2.218300e-05)
    assert augmented_gpu["last_reported_residual_norm"] == pytest.approx(2.218202e-05)
    assert augmented_gpu["last_reported_residual_norm"] < pitch_multilevel_gpu[
        "last_reported_residual_norm"
    ]
    assert augmented_gpu["max_elapsed_s"] < 170.0
    recycled_augmented_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_recycled_augmented_deep_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert recycled_augmented_gpu["passed"] is False
    assert recycled_augmented_gpu["evidence_classes"] == ["device_qi_augmented_krylov_coarse_reuse"]
    assert "observed_augmented_krylov" in recycled_augmented_gpu["evidence_tags"]
    assert recycled_augmented_gpu["last_reported_residual_norm"] == pytest.approx(7.336295e-06)
    assert recycled_augmented_gpu["last_reported_residual_norm"] < augmented_gpu[
        "last_reported_residual_norm"
    ]
    assert recycled_augmented_gpu["max_elapsed_s"] < 170.0
    phase_space_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_phase_space_coarse_reuse_device_qi_gpu0.json"
    )
    assert phase_space_gpu["passed"] is False
    assert phase_space_gpu["evidence_classes"] == [
        "device_qi_phase_space_residual_equation_coarse_reuse"
    ]
    assert "observed_phase_space_residual_equation" in phase_space_gpu["evidence_tags"]
    assert phase_space_gpu["last_reported_residual_norm"] == pytest.approx(
        recycled_augmented_gpu["last_reported_residual_norm"]
    )
    assert phase_space_gpu["max_elapsed_s"] > recycled_augmented_gpu["max_elapsed_s"]
    assert phase_space_gpu["max_elapsed_s"] < 260.0
    coarse_residual_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_coarse_residual_device_qi_cpu_2026_05_20.json"
    )
    assert coarse_residual_cpu["passed"] is False
    assert coarse_residual_cpu["evidence_classes"] == ["device_qi_multilevel_residual_equation"]
    assert "observed_multilevel_residual_equation" in coarse_residual_cpu["evidence_tags"]
    assert coarse_residual_cpu["last_reported_residual_norm"] == pytest.approx(2.306911e-05)
    assert coarse_residual_cpu["last_reported_residual_norm"] > augmented_cpu["last_reported_residual_norm"]
    assert coarse_residual_cpu["max_elapsed_s"] < 300.0
    residual_snapshot_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_residual_snapshot_device_qi_cpu_2026_05_20.json"
    )
    assert residual_snapshot_cpu["passed"] is False
    assert residual_snapshot_cpu["evidence_classes"] == ["device_qi_residual_snapshot_coarse_reuse"]
    assert "observed_residual_snapshot" in residual_snapshot_cpu["evidence_tags"]
    assert residual_snapshot_cpu["last_reported_residual_norm"] == pytest.approx(2.103015e-05)
    assert residual_snapshot_cpu["last_reported_residual_norm"] < augmented_cpu[
        "last_reported_residual_norm"
    ]
    assert residual_snapshot_cpu["max_elapsed_s"] < 300.0
    residual_snapshot_equation_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_residual_snapshot_equation_device_qi_cpu_2026_05_20.json"
        )
    )
    assert residual_snapshot_equation_cpu["passed"] is False
    assert residual_snapshot_equation_cpu["evidence_classes"] == [
        "device_qi_residual_snapshot_residual_equation_coarse_reuse"
    ]
    assert "observed_residual_snapshot_residual_equation" in residual_snapshot_equation_cpu["evidence_tags"]
    assert residual_snapshot_equation_cpu["last_reported_residual_norm"] == pytest.approx(2.320763e-05)
    assert residual_snapshot_equation_cpu["last_reported_residual_norm"] > residual_snapshot_cpu[
        "last_reported_residual_norm"
    ]
    assert residual_snapshot_equation_cpu["max_elapsed_s"] < 300.0
    global_moment_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_global_moment_closure_device_qi_cpu_2026_05_20.json"
        )
    )
    assert global_moment_cpu["passed"] is False
    assert global_moment_cpu["evidence_classes"] == [
        "device_qi_global_moment_residual_equation_coarse_reuse"
    ]
    assert "observed_global_moment_residual_equation" in global_moment_cpu["evidence_tags"]
    assert global_moment_cpu["last_reported_residual_norm"] == pytest.approx(2.420524e-05)
    assert global_moment_cpu["last_reported_residual_norm"] > residual_snapshot_cpu[
        "last_reported_residual_norm"
    ]
    assert global_moment_cpu["max_elapsed_s"] < 300.0
    global_moment_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_global_moment_closure_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert global_moment_gpu["passed"] is False
    assert global_moment_gpu["evidence_classes"] == global_moment_cpu["evidence_classes"]
    assert "observed_global_moment_residual_equation" in global_moment_gpu["evidence_tags"]
    assert global_moment_gpu["last_reported_residual_norm"] == pytest.approx(2.420524e-05)
    assert global_moment_gpu["max_elapsed_s"] < 320.0
    residual_galerkin_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_residual_galerkin_device_qi_cpu_2026_05_20.json"
    )
    assert residual_galerkin_cpu["passed"] is False
    assert residual_galerkin_cpu["evidence_classes"] == [
        "device_qi_residual_galerkin_equation_coarse_reuse"
    ]
    assert "observed_residual_galerkin_equation" in residual_galerkin_cpu["evidence_tags"]
    assert residual_galerkin_cpu["last_reported_residual_norm"] == pytest.approx(2.632208e-05)
    assert residual_galerkin_cpu["last_reported_residual_norm"] > residual_snapshot_cpu[
        "last_reported_residual_norm"
    ]
    assert residual_galerkin_cpu["max_elapsed_s"] < 300.0
    residual_galerkin_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_residual_galerkin_device_qi_gpu1_2026_05_20.json"
        )
    )
    assert residual_galerkin_gpu["passed"] is False
    assert residual_galerkin_gpu["evidence_classes"] == residual_galerkin_cpu["evidence_classes"]
    assert "observed_residual_galerkin_equation" in residual_galerkin_gpu["evidence_tags"]
    assert residual_galerkin_gpu["last_reported_residual_norm"] == pytest.approx(2.632208e-05)
    assert residual_galerkin_gpu["max_elapsed_s"] < 330.0
    block_schur_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_block_schur_device_qi_cpu_2026_05_20.json"
    )
    assert block_schur_cpu["passed"] is False
    assert block_schur_cpu["evidence_classes"] == ["device_qi_block_schur_residual_coarse_reuse"]
    assert "observed_block_schur_residual" in block_schur_cpu["evidence_tags"]
    assert block_schur_cpu["last_reported_residual_norm"] == pytest.approx(2.275188e-05)
    assert block_schur_cpu["last_reported_residual_norm"] > residual_snapshot_cpu[
        "last_reported_residual_norm"
    ]
    assert block_schur_cpu["max_elapsed_s"] < 300.0
    block_schur_bestof_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_block_schur_bestof_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert block_schur_bestof_gpu["passed"] is False
    assert block_schur_bestof_gpu["evidence_classes"] == ["device_qi_block_schur_residual_coarse_reuse"]
    assert "observed_block_schur_residual" in block_schur_bestof_gpu["evidence_tags"]
    assert block_schur_bestof_gpu["last_reported_residual_norm"] == pytest.approx(1.992464e-05)
    assert block_schur_bestof_gpu["last_reported_residual_norm"] < residual_snapshot_cpu[
        "last_reported_residual_norm"
    ]
    assert block_schur_bestof_gpu["max_elapsed_s"] < 320.0
    adaptive_residual_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_adaptive_residual_device_qi_gpu0_2026_05_20.json"
        )
    )
    assert adaptive_residual_gpu["passed"] is False
    assert adaptive_residual_gpu["evidence_classes"] == ["device_qi_block_schur_residual_coarse_reuse"]
    assert "observed_block_schur_residual" in adaptive_residual_gpu["evidence_tags"]
    assert adaptive_residual_gpu["last_reported_residual_norm"] == pytest.approx(2.307995e-05)
    assert adaptive_residual_gpu["last_reported_residual_norm"] > block_schur_bestof_gpu[
        "last_reported_residual_norm"
    ]
    assert adaptive_residual_gpu["max_elapsed_s"] < 300.0
    composite_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == (
            "docs/_static/"
            "qi_seed_robustness_scale060_composite_closure_device_qi_gpu1_2026_05_20.json"
        )
    )
    assert composite_gpu["passed"] is False
    assert composite_gpu["evidence_classes"] == ["device_qi_residual_galerkin_equation_coarse_reuse"]
    assert "observed_residual_snapshot" in composite_gpu["evidence_tags"]
    assert "observed_residual_galerkin_equation" in composite_gpu["evidence_tags"]
    assert "observed_block_schur_residual" in composite_gpu["evidence_tags"]
    assert composite_gpu["last_reported_residual_norm"] == pytest.approx(2.305955e-05)
    assert composite_gpu["last_reported_residual_norm"] > block_schur_bestof_gpu[
        "last_reported_residual_norm"
    ]
    assert composite_gpu["max_elapsed_s"] < 330.0
    adjoint_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_adjoint_krylov_device_qi_cpu_2026_05_20.json"
    )
    assert adjoint_cpu["passed"] is False
    assert adjoint_cpu["last_reported_residual_norm"] == pytest.approx(2.48643e-05)
    assert adjoint_cpu["last_reported_residual_norm"] > pitch_multilevel_gpu["last_reported_residual_norm"]
    assert adjoint_cpu["max_elapsed_s"] < 650.0
    early_payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_early_qi_skipstrong_skipglobal_seed3_cpu_2026_05_19.json")
        .read_text(encoding="utf-8")
    )
    block_payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_qi_device_blockminres_sweeps1_seed3_cpu_2026_05_19.json")
        .read_text(encoding="utf-8")
    )
    assert (
        block_payload["seeds"][0]["xblock_qi_device_preconditioner_residual_after"]
        < early_payload["seeds"][0]["xblock_qi_device_preconditioner_residual_after"]
    )
    nohost = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_krylov_nohost_recycle_seed3_gpu0_2026_05_19.json"
    )
    assert nohost["passed"] is False
    assert nohost["active_size"] == 81377
    assert nohost["total_size"] == 139502

    rejected_global = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_global_coupling_rejected_2026_05_13.json"
    )
    assert rejected_global["passed"] is False
    assert rejected_global["active_size"] == 81377
    assert rejected_global["total_size"] == 139502
    assert rejected_global["resolution_fractions"]["NTHETA"] == 0.6
    rejected_device = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_device_krylov_rejected_2026_05_13.json"
    )
    assert rejected_device["passed"] is False
    assert rejected_device["backends"] == ["gpu"]
    assert rejected_device["max_elapsed_s"] > 250.0
    rejected_device_operator = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_device_operator_rejected_2026_05_13.json"
    )
    assert rejected_device_operator["passed"] is False
    assert rejected_device_operator["backends"] == ["gpu"]
    assert rejected_device_operator["active_size"] == 81377
    assert rejected_device_operator["total_size"] == 139502
    assert rejected_device_operator["max_elapsed_s"] > 500.0
    probe_coarse_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_cpu.json"
    )
    assert probe_coarse_cpu["passed"] is True
    assert probe_coarse_cpu["backends"] == ["cpu"]
    assert probe_coarse_cpu["active_size"] == 81377
    assert probe_coarse_cpu["total_size"] == 139502
    assert probe_coarse_cpu["max_residual_ratio"] < 0.01
    qi_coarse_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_cpu_2026_05_14.json"
    )
    assert qi_coarse_cpu["passed"] is True
    assert qi_coarse_cpu["backends"] == ["cpu"]
    assert qi_coarse_cpu["active_size"] == 81377
    assert qi_coarse_cpu["total_size"] == 139502
    assert qi_coarse_cpu["max_residual_ratio"] < 0.01
    angular_residual_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_probe_coarse_angular_residual_seed3_cpu_2026_05_14.json"
    )
    assert angular_residual_cpu["passed"] is True
    assert angular_residual_cpu["active_size"] == 81377
    assert angular_residual_cpu["total_size"] == 139502
    assert angular_residual_cpu["max_elapsed_s"] < qi_coarse_cpu["max_elapsed_s"]
    host_fallback_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json"
    )
    assert host_fallback_cpu["passed"] is True
    assert host_fallback_cpu["backends"] == ["cpu"]
    assert host_fallback_cpu["active_size"] == 81377
    assert host_fallback_cpu["total_size"] == 139502
    assert host_fallback_cpu["max_elapsed_s"] < 180.0
    assert host_fallback_cpu["max_residual_ratio"] < 0.01
    smoothed_load_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_two_level_smoothed_load_seed3_cpu_2026_05_16.json"
    )
    assert smoothed_load_cpu["passed"] is True
    assert smoothed_load_cpu["active_size"] == 81377
    assert smoothed_load_cpu["total_size"] == 139502
    assert smoothed_load_cpu["max_elapsed_s"] > host_fallback_cpu["max_elapsed_s"]
    moment_schur_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_moment_schur_probe_seed3_cpu_2026_05_16.json"
    )
    assert moment_schur_cpu["passed"] is True
    assert moment_schur_cpu["active_size"] == 81377
    assert moment_schur_cpu["total_size"] == 139502
    assert moment_schur_cpu["max_elapsed_s"] > host_fallback_cpu["max_elapsed_s"]
    qi_deflated_minres_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_seed3_cpu_2026_05_17.json"
    )
    assert qi_deflated_minres_cpu["passed"] is True
    assert qi_deflated_minres_cpu["backends"] == ["cpu"]
    assert qi_deflated_minres_cpu["active_size"] == 81377
    assert qi_deflated_minres_cpu["total_size"] == 139502
    assert qi_deflated_minres_cpu["max_residual_ratio"] < 0.01
    qi_device_full_csr_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_preconditioner_seed3_cpu_2026_05_19.json"
    )
    assert qi_device_full_csr_cpu["passed"] is True
    assert qi_device_full_csr_cpu["active_size"] == 81377
    assert qi_device_full_csr_cpu["total_size"] == 139502
    assert qi_device_full_csr_cpu["max_residual_ratio"] < 0.01
    qi_device_matrixfree_cpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_device_matrixfree_seed3_cpu_2026_05_19.json"
    )
    assert qi_device_matrixfree_cpu["passed"] is True
    assert qi_device_matrixfree_cpu["active_size"] == 81377
    assert qi_device_matrixfree_cpu["total_size"] == 139502
    assert qi_device_matrixfree_cpu["max_residual_ratio"] < 0.01
    matrixfree_payload = json.loads(
        Path("docs/_static/qi_seed_robustness_scale060_qi_device_matrixfree_seed3_cpu_2026_05_19.json").read_text(
            encoding="utf-8"
        )
    )
    matrixfree_seed = matrixfree_payload["seeds"][0]
    assert matrixfree_seed["xblock_qi_device_preconditioner_built"] is True
    assert matrixfree_seed["xblock_qi_device_preconditioner_used"] is False
    assert matrixfree_seed["xblock_qi_device_preconditioner_reason"] == "residual_not_reduced"
    assert matrixfree_seed["xblock_qi_device_preconditioner_metadata"]["operator_source"] == "matrix_free"
    assert matrixfree_seed["xblock_qi_device_preconditioner_metadata"]["local_smoother_kind"] == "none"
    qi_deflated_gpu_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_seed3_gpu0_2026_05_17.json"
    )
    assert qi_deflated_gpu_timeout["passed"] is False
    assert qi_deflated_gpu_timeout["total_size"] == 139502
    assert qi_deflated_gpu_timeout["max_elapsed_s"] > 420.0
    qi_deflated_lgmres_gpu_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_deflated_minres8_lgmres_forced_seed3_gpu0_2026_05_17.json"
    )
    assert qi_deflated_lgmres_gpu_timeout["passed"] is False
    assert qi_deflated_lgmres_gpu_timeout["total_size"] == 139502
    assert qi_deflated_lgmres_gpu_timeout["max_elapsed_s"] > 420.0
    lower_fill = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_lower_fill_seed3_cpu_rejected_2026_05_14.json"
    )
    assert lower_fill["passed"] is False
    assert lower_fill["active_size"] == 81377
    assert lower_fill["total_size"] == 139502
    probe_coarse_gpu = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"] == "docs/_static/qi_seed_robustness_scale060_probe_coarse_seed3_gpu0_timeout.json"
    )
    assert probe_coarse_gpu["passed"] is False
    assert probe_coarse_gpu["total_size"] == 139502
    assert probe_coarse_gpu["max_elapsed_s"] > 350.0
    qi_coarse_gpu_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_heartbeat_timeout_2026_05_14.json"
    )
    assert qi_coarse_gpu_timeout["passed"] is False
    assert qi_coarse_gpu_timeout["active_size"] == 81377
    assert qi_coarse_gpu_timeout["total_size"] == 139502
    assert qi_coarse_gpu_timeout["max_elapsed_s"] > 420.0
    qi_coarse_gpu_no_lgmres_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_qi_coarse_seed3_gpu0_no_lgmres_timeout_2026_05_14.json"
    )
    assert qi_coarse_gpu_no_lgmres_timeout["passed"] is False
    assert qi_coarse_gpu_no_lgmres_timeout["active_size"] == 81377
    assert qi_coarse_gpu_no_lgmres_timeout["total_size"] == 139502
    assert qi_coarse_gpu_no_lgmres_timeout["max_elapsed_s"] > 420.0
    enriched_gpu_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_enriched_angular_seed3_gpu0_no_lgmres_timeout_2026_05_14.json"
    )
    assert enriched_gpu_timeout["passed"] is False
    assert enriched_gpu_timeout["active_size"] == 81377
    assert enriched_gpu_timeout["total_size"] == 139502
    compact_no_moment_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_device_krylov_compact_no_moment_seed3_gpu1_2026_05_15.json"
    )
    assert compact_no_moment_timeout["passed"] is False
    assert compact_no_moment_timeout["active_size"] == 81377
    assert compact_no_moment_timeout["total_size"] == 139502
    assert compact_no_moment_timeout["max_elapsed_s"] > 470.0
    cycle_restart4_timeout = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_restart4_diag_seed3_gpu0_2026_05_15.json"
    )
    assert cycle_restart4_timeout["passed"] is False
    assert cycle_restart4_timeout["active_size"] == 81377
    assert cycle_restart4_timeout["total_size"] == 139502
    assert cycle_restart4_timeout["max_elapsed_s"] > 350.0
    diag_factor_rejected = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_diag_factor_seed3_gpu0_2026_05_15.json"
    )
    assert diag_factor_rejected["passed"] is False
    assert diag_factor_rejected["active_size"] == 81377
    assert diag_factor_rejected["total_size"] == 139502
    assert 120.0 < diag_factor_rejected["max_elapsed_s"] < 180.0
    exact_cap16_rejected = next(
        artifact
        for artifact in payload["source_artifacts"]
        if artifact["path"]
        == "docs/_static/qi_seed_robustness_scale060_device_cycle_jit_exact_cap16_seed3_gpu0_2026_05_15.json"
    )
    assert exact_cap16_rejected["passed"] is False
    assert exact_cap16_rejected["active_size"] == 81377
    assert exact_cap16_rejected["total_size"] == 139502
    assert exact_cap16_rejected["max_elapsed_s"] > diag_factor_rejected["max_elapsed_s"]

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
    assert "--probe-preset block-schur-device-qi" in commands["block_schur_device_qi_gpu0_probe"]
    assert (
        "--probe-preset global-moment-closure-device-qi"
        in commands["global_moment_closure_device_qi_gpu0_probe"]
    )
    assert (
        "--probe-preset residual-galerkin-device-qi"
        in commands["residual_galerkin_device_qi_gpu0_probe"]
    )
    assert (
        "--probe-preset residual-snapshot-equation-device-qi"
        in commands["residual_snapshot_equation_device_qi_gpu0_probe"]
    )
    assert (
        "--probe-preset recycled-augmented-device-qi"
        in commands["recycled_augmented_device_qi_gpu0_probe"]
    )
    assert (
        "--probe-preset phase-space-coarse-reuse-device-qi"
        in commands["phase_space_coarse_reuse_device_qi_gpu0_probe"]
    )
    assert "--probe-preset adaptive-residual-device-qi" in commands["adaptive_residual_device_qi_gpu0_probe"]

    residual_snapshot_equation_preset = payload["probe_presets"]["residual-snapshot-equation-device-qi"]
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "48"
    )
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER"
        ]
        == "action_lstsq"
    )
    assert (
        residual_snapshot_equation_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_INCLUDE_GLOBAL"
        ]
        == "1"
    )
    assert "fail-closed" in residual_snapshot_equation_preset["description"]

    composite_closure_preset = payload["probe_presets"]["composite-closure-device-qi"]
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_ENRICHMENT"
        ]
        == "1"
    )
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"
        ]
        == "1"
    )
    assert (
        composite_closure_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert "composite residual-snapshot" in composite_closure_preset["description"]

    global_moment_preset = payload["probe_presets"]["global-moment-closure-device-qi"]
    assert (
        global_moment_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert "fail-closed" in global_moment_preset["description"]

    residual_galerkin_preset = payload["probe_presets"]["residual-galerkin-device-qi"]
    assert (
        residual_galerkin_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION"
        ]
        == "1"
    )
    assert (
        residual_galerkin_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_GALERKIN_EQUATION_MAX_STAGE_RANK"
        ]
        == "8"
    )
    assert "fail-closed" in residual_galerkin_preset["description"]
    phase_space_preset = payload["probe_presets"]["phase-space-coarse-reuse-device-qi"]
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        phase_space_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_PHASE_SPACE_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "32"
    )
    assert "fail-closed" in phase_space_preset["description"]

    block_schur_preset = payload["probe_presets"]["block-schur-device-qi"]
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION"
        ]
        == "1"
    )
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK"
        ]
        == "64"
    )
    assert (
        block_schur_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER"
        ]
        == "galerkin"
    )
    assert "fail-closed" in block_schur_preset["description"]
    recycled_preset = payload["probe_presets"]["recycled-augmented-device-qi"]
    assert recycled_preset["env"]["SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_OUTER_K"] == "32"
    assert recycled_preset["env"]["SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER"] == "960"
    assert "best checked residual" in recycled_preset["description"]
    adaptive_preset = payload["probe_presets"]["adaptive-residual-device-qi"]
    assert (
        adaptive_preset["env"][
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER"
        ]
        == "adaptive_residual_equation"
    )
    assert "negative-evidence" in adaptive_preset["description"]
