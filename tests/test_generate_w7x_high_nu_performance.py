from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_w7x_high_nu_performance.py"
    spec = importlib.util.spec_from_file_location("generate_w7x_high_nu_performance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_w7x_high_nu_summary_tracks_residual_and_setup_gates() -> None:
    mod = _load_module()
    records = [
        {
            "id": "bounded_30k_krylov",
            "label": "failed",
            "status": "failed_residual_gate",
            "elapsed_s": 400.0,
            "sparse_factorizations": 0,
            "relative_residuals": [0.7, 0.9, 1.0],
            "max_rss_mb": 1300.0,
        },
        {
            "id": "sparse_lu_no_reuse",
            "label": "no reuse",
            "status": "passed",
            "elapsed_s": 2000.0,
            "sparse_factorizations": 3,
            "max_relative_residual": 8.0e-7,
            "relative_residuals": [8.0e-7, 7.0e-9],
            "max_rss_mb": 19000.0,
        },
        {
            "id": "sparse_lu_factor_reuse",
            "label": "reuse",
            "status": "passed",
            "elapsed_s": 800.0,
            "sparse_factorizations": 1,
            "max_relative_residual": 7.0e-7,
            "relative_residuals": [8.0e-7, 7.0e-9],
            "max_rss_mb": 15000.0,
        },
    ]

    payload = mod.build_w7x_high_nu_performance_summary(records, residual_gate=1.0e-6)

    assert payload["metadata"]["kind"] == "w7x_high_nu_preconditioning_performance"
    assert payload["metadata"]["publication_figure"]["ready_for_physics_validation_claim"] is False
    assert payload["gates"]["failed_route_rejected"] is True
    assert payload["gates"]["factor_reuse_residual_clean"] is True
    assert payload["gates"]["factor_reuse_fewer_factorizations"] is True
    assert payload["gates"]["factor_reuse_residuals_match_no_reuse"] is True
    assert payload["gates"]["declared_passed_routes_residual_clean"] is True
    assert payload["gates"]["single_point_performance_claim_supported"] is True
    assert payload["gates"]["checked_in_converged_artifact"] is False
    assert payload["gates"]["ready_for_physics_validation_claim"] is False
    assert payload["gates"]["factor_reuse_speedup_vs_no_reuse"] == 2.5
    assert payload["gates"]["factor_reuse_wall_time_saved_s"] == 1200.0

    checked_payload = mod.build_w7x_high_nu_performance_summary(
        records,
        residual_gate=1.0e-6,
        records_source="checked_in_default_records",
        summary_artifact_checked_in=True,
    )
    assert checked_payload["metadata"]["publication_figure"]["claim_status"] == "checked_in_converged_artifact"
    assert checked_payload["gates"]["checked_in_converged_artifact"] is True
    assert checked_payload["gates"]["ready_for_physics_validation_claim"] is False


def test_generate_w7x_high_nu_performance_writes_figure_and_summary(tmp_path: Path) -> None:
    mod = _load_module()
    out_dir = tmp_path / "figures"
    summary_json = tmp_path / "summary.json"

    rc = mod.main(
        [
            "--out-dir",
            str(out_dir),
            "--summary-json",
            str(summary_json),
            "--stem",
            "w7x_high_nu_test",
        ]
    )

    assert rc == 0
    assert (out_dir / "w7x_high_nu_test.png").exists()
    assert (out_dir / "w7x_high_nu_test.pdf").exists()
    payload = json.loads(summary_json.read_text())
    assert payload["metadata"]["kind"] == "w7x_high_nu_preconditioning_performance"
    assert payload["metadata"]["records_source"] == "checked_in_default_records"
    assert payload["metadata"]["summary_artifact_checked_in"] is False
    assert payload["metadata"]["validation_state"] == "performance_claim_deferred_or_external_records"
    assert payload["metadata"]["publication_figure"]["checked_in_converged_artifact"] is False
    assert payload["metadata"]["publication_figure"]["ready_for_physics_validation_claim"] is False
    assert payload["gates"]["factor_reuse_present"] is True
    assert payload["gates"]["single_point_performance_claim_supported"] is True
    assert payload["gates"]["checked_in_converged_artifact"] is False
    assert payload["gates"]["ready_for_physics_validation_claim"] is False
