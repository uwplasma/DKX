from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT = (
    Path("docs/_static/figures/optimization")
    / "qi_nfp2_electron_root_res13_single_point_probe.json"
)

CPU_SCAN_ARTIFACT = (
    Path("docs/_static/figures/optimization")
    / "qi_nfp2_electron_root_res13_cpu_sparse_skip.json"
)


def test_qi_res13_single_point_probe_stays_bounded_and_fail_scoped() -> None:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_nfp2_kinetic_single_point_probe"
    assert payload["status"] == "pass_bounded_single_point"
    assert "not a full electron-root scan" in payload["claim_boundary"]
    assert payload["resolution"] == {
        "Ntheta": 13,
        "Nzeta": 13,
        "Nxi": 15,
        "Nx": 4,
        "solverTolerance": "1d-6",
    }

    result = payload["result"]
    assert result["active_size"] == 11496
    assert result["total_size"] == 20284
    assert result["converged"] is True
    assert result["residual_norm"] < result["residual_target"]
    assert result["solver_elapsed_s"] == pytest.approx(35.277955916011706)
    assert result["peak_rss_mb"] == pytest.approx(1872.109375)

    comparison = payload["performance_comparison"]
    assert comparison["baseline_auto_solver_elapsed_s"] == pytest.approx(107.87407708284445)
    assert comparison["sparse_skip_speedup"] > 3.0
    assert comparison["max_key_observable_abs_difference"] == 0.0
    assert comparison["max_key_observable_rel_difference"] == 0.0

    policy = payload["policy_result"]
    assert policy["selected_route"] == "auto -> active sparse-LU rescue"
    assert "stage2 GMRES" in policy["skipped_routes"]
    assert policy["top_level_sharding_preserved"] is True
    assert policy["transformed_matvec_path"] == "local_unsharded_jit"

    rejected = {record["route"]: record for record in payload["failed_or_rejected_routes"]}
    assert "one_device_unsharded_cpu" in rejected
    assert "sparse_host" in rejected
    assert "sparse_pc_gmres" in rejected
    assert "residual" in rejected["one_device_unsharded_cpu"]["reason"]
    assert "SuperLU factorization failed" in rejected["sparse_host"]["reason"]


def test_qi_res13_cpu_ladder_artifact_stays_residual_clean_and_scoped() -> None:
    payload = json.loads(CPU_SCAN_ARTIFACT.read_text(encoding="utf-8"))

    assert payload["artifact_kind"] == "qi_nfp2_kinetic_res13_cpu_sparse_skip_scan"
    assert payload["status"] == "pass_bounded_cpu_ladder"
    assert "GPU and Fortran-v3 fixed-resolution evidence are still required" in payload[
        "claim_boundary"
    ]
    assert payload["backend"] == "cpu"
    assert payload["resolution"] == {
        "Ntheta": 13,
        "Nzeta": 13,
        "Nxi": 15,
        "NL": 4,
        "Nx": 4,
        "solverTolerance": "1d-6",
    }

    gate = payload["promotion_gate"]
    assert gate["gate_status"] == "pass"
    assert gate["failures"] == []
    assert gate["selected_root"]["root_type"] == "electron"
    assert gate["selected_root"]["bracket"] == [2.0, 3.0]
    assert gate["selected_root"]["er"] == pytest.approx(2.2153427466642333)

    performance = payload["performance_summary"]
    assert performance["single_point_sparse_skip_speedup"] > 3.0
    assert performance["total_scan_elapsed_s"] < 300.0
    assert performance["max_solver_elapsed_s"] < 40.0
    assert performance["mean_solver_elapsed_s"] == pytest.approx(32.84566542171524)

    runs = payload["runs"]
    assert len(runs) == 8
    assert [run["er"] for run in runs] == [-0.3, -0.1, 0.0, 0.1, 0.3, 1.0, 2.0, 3.0]
    assert all(run["residual_norm"] < run["residual_target"] for run in runs)

    radial_currents = [run["radial_current"] for run in runs]
    assert min(radial_currents) < 0.0
    assert max(radial_currents) > 0.0

    policy = payload["policy_result"]
    assert policy["selected_route"] == "auto -> active sparse-LU rescue"
    assert "primary Krylov" in policy["skipped_routes"]
    assert policy["top_level_sharding_preserved"] is True
