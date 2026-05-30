from __future__ import annotations

import json
from pathlib import Path


ARTIFACT_DIR = Path("docs/_static/figures/optimization")
CONFIG = ARTIFACT_DIR / "qi_nfp2_electron_root_ladder_config.json"
SUMMARY = ARTIFACT_DIR / "qi_nfp2_electron_root_convergence_ladder.json"


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_qi_nfp2_electron_root_ladder_rollup_is_fail_closed() -> None:
    config = _read(CONFIG)
    summary = _read(SUMMARY)

    assert config["workflow"] == "sfincs_jax_qi_nfp2_electron_root_convergence_ladder_config"
    assert summary["workflow"] == "sfincs_jax_qi_nfp2_electron_root_convergence_ladder"
    assert summary["status"] == "deferred"
    assert "not a production-resolution QI convergence" in summary["claim_boundary"]
    assert summary["production_target"] == "25 x 51 x 100 x 4 CPU and GPU QI ladder"
    assert config["production_floor"] == summary["production_floor"]

    tiers = summary["tiers"]
    assert [tier["name"] for tier in tiers] == [tier["name"] for tier in config["tiers"]]
    assert all(tier["production_floor_met"] is False for tier in tiers)
    assert all((ARTIFACT_DIR / source).exists() for tier in tiers for source in tier["source_artifacts"])
    assert summary["gates"]["production_resolution_qi_convergence"] == "open"
    assert summary["gates"]["res15_gpu_promotion"] == "pass_bounded"


def test_qi_nfp2_electron_root_ladder_values_match_source_artifacts() -> None:
    summary = _read(SUMMARY)
    tier_by_name = {tier["name"]: tier for tier in summary["tiers"]}

    low_cpu = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_lowres_cpu.json")
    res9_gpu = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res9_gpu.json")
    res11_fortran = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res11_fortran.json")
    res13_comparison = _read(
        ARTIFACT_DIR / "qi_nfp2_electron_root_res13_reference_tolerance_comparison_sparse_skip.json"
    )
    res15_comparison = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res15_cpu_fortran_sparse_skip.json")
    res15_gpu = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res15_gpu_campaign.json")

    assert tier_by_name["low_7x7x7"]["lanes"]["cpu"]["selected_root_er"] == low_cpu["selected_root"]["er"]
    assert tier_by_name["res9_9x9x11"]["lanes"]["gpu"]["selected_root_er"] == res9_gpu["selected_root"]["er"]
    assert tier_by_name["res11_11x11x13"]["lanes"]["fortran_v3"]["selected_root_er"] == res11_fortran["selected_root"]["er"]
    assert (
        tier_by_name["res13_13x13x15"]["lanes"]["gpu"]["selected_root_er"]
        == res13_comparison["fixed_resolution_roots"]["gpu"]["er"]
    )
    assert (
        tier_by_name["res15_15x15x17"]["lanes"]["cpu"]["selected_root_er"]
        == res15_comparison["fixed_resolution_roots"]["cpu"]["er"]
    )
    assert (
        tier_by_name["res15_15x15x17"]["lanes"]["gpu"]["selected_root_er"]
        == res15_gpu["gpu_selected_root"]["er"]
    )
    assert tier_by_name["res15_15x15x17"]["gpu_campaign_gate"]["status"] == "pass_bounded_gpu_res15"


def test_qi_nfp2_ladder_convergence_trend_is_improving_but_not_promoted() -> None:
    summary = _read(SUMMARY)
    drifts = [float(tier["root_drift_from_previous"]) for tier in summary["tiers"][1:]]

    assert drifts == sorted(drifts, reverse=True)
    assert drifts[-1] < summary["tolerances"]["latest_root_drift_atol_for_trend_only"]
    assert summary["status"] == "deferred"
    assert any("No checked tier meets" in blocker for blocker in summary["blockers"])


def test_qi_nfp2_res15_gpu_campaign_artifact_passes_fixed_resolution_gate() -> None:
    campaign = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res15_gpu_campaign.json")
    promotion = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res15_gpu.json")
    campaign_summary = _read(ARTIFACT_DIR / "qi_nfp2_electron_root_res15_gpu_campaign_summary.json")

    assert campaign["workflow"] == "sfincs_jax_qi_nfp2_res15_gpu_campaign_evidence"
    assert campaign["status"] == "pass_bounded_gpu_res15"
    assert campaign["failures"] == []
    assert campaign["gates"]["gpu_residuals"] == "pass"
    assert campaign["gates"]["gpu_cpu_root_agreement"] == "pass"
    assert campaign["gates"]["gpu_fortran_v3_root_agreement"] == "pass"
    assert campaign["gates"]["production_resolution_qi_convergence"] == "open"
    assert campaign["residual_summary"]["run_count"] == 8
    assert campaign["residual_summary"]["max_residual_ratio"] < 1.0e-5
    assert campaign["root_differences"]["gpu_minus_cpu_abs"] < 1.0e-12
    assert campaign["root_differences"]["gpu_minus_fortran_v3_abs"] < 3.0e-6

    assert promotion["gate_status"] == "pass"
    assert promotion["selected_root"]["root_type"] == "electron"
    assert promotion["selected_root"]["er"] == campaign["gpu_selected_root"]["er"]
    assert campaign_summary["campaign_status"] == "pass"
