from __future__ import annotations

import json
from pathlib import Path

import pytest


ARTIFACT = (
    Path("docs/_static/figures/optimization")
    / "qi_nfp2_electron_root_res13_single_point_probe.json"
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
