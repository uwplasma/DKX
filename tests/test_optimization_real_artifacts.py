from __future__ import annotations

import json
from pathlib import Path

from sfincs_jax.optimization_ladder import estimate_rhs1_active_size


_REPO = Path(__file__).resolve().parents[1]
_OPT_FIGURES = _REPO / "docs" / "_static" / "figures" / "optimization"


def _load(name: str) -> dict[str, object]:
    return json.loads((_OPT_FIGURES / name).read_text(encoding="utf-8"))


def _promotion_root(payload: dict[str, object]) -> dict[str, object]:
    root = payload["selected_root"]
    assert isinstance(root, dict)
    assert root["root_type"] == "electron"
    assert root["bracket"] == [0.25, 0.5]
    assert root["er"] > 0.0
    assert root["slope"] > 0.0
    return root


def test_finite_beta_electron_root_promotion_artifact_passes() -> None:
    comparison = _load("qa_nfp2_finite_beta_electron_root_promotion_comparison.json")
    cpu = _load("qa_nfp2_finite_beta_electron_root_cpu_promotion.json")
    gpu = _load("qa_nfp2_finite_beta_electron_root_gpu_promotion.json")
    fortran = _load("qa_nfp2_finite_beta_electron_root_fortran_v3_promotion.json")

    assert comparison["status"] == "pass"
    assert comparison["failures"] == []
    assert comparison["tolerances"] == {
        "bootstrap_objective_rtol": 1.0e-5,
        "flux_objective_total_rtol": 1.0e-6,
        "selected_root_er_atol": 1.0e-7,
        "selected_root_er_rtol": 1.0e-7,
    }

    for payload in (cpu, gpu, fortran):
        assert payload["gate_status"] == "pass"
        _promotion_root(payload)

    assert abs(cpu["selected_root"]["er"] - gpu["selected_root"]["er"]) < 1.0e-12
    assert abs(cpu["selected_root"]["er"] - fortran["selected_root"]["er"]) < 1.0e-7


def test_finite_beta_electron_root_ladder_config_resolves_checked_artifacts() -> None:
    config = _load("qa_nfp2_finite_beta_electron_root_ladder_config.json")
    summary = _load("qa_nfp2_finite_beta_electron_root_convergence_ladder.json")

    assert config["production_floor"] == summary["production_floor"]
    assert config["workflow"] == "sfincs_jax_finite_beta_electron_root_convergence_ladder_config"
    assert summary["workflow"] == "sfincs_jax_finite_beta_electron_root_convergence_ladder"

    config_tiers = config["tiers"]
    summary_tiers = summary["tiers"]
    assert isinstance(config_tiers, list)
    assert isinstance(summary_tiers, list)
    assert len(config_tiers) == len(summary_tiers)

    previous_active_size = 0
    for config_tier, summary_tier in zip(config_tiers, summary_tiers, strict=True):
        assert isinstance(config_tier, dict)
        assert isinstance(summary_tier, dict)
        assert summary_tier["name"] == config_tier["name"]
        assert summary_tier["resolution"] == config_tier["resolution"]
        assert summary_tier["r_n"] == config_tier["r_n"]
        assert summary_tier["status"] == "pass"
        assert summary_tier["production_floor_met"] is False

        resolution = config_tier["resolution"]
        assert isinstance(resolution, dict)
        expected_active_size = estimate_rhs1_active_size(
            ntheta=resolution["Ntheta"],
            nzeta=resolution["Nzeta"],
            nxi=resolution["Nxi"],
            nx=resolution["Nx"],
            n_species=2,
        )
        assert summary_tier["active_size_estimate"] == expected_active_size
        assert expected_active_size > previous_active_size
        previous_active_size = expected_active_size

        promotions = config_tier["promotions"]
        lanes = summary_tier["lanes"]
        assert isinstance(promotions, dict)
        assert isinstance(lanes, dict)
        assert sorted(promotions) == ["cpu", "fortran_v3", "gpu"]
        assert sorted(lanes) == sorted(promotions)

        for lane_name, artifact_name in promotions.items():
            assert isinstance(artifact_name, str)
            payload = _load(artifact_name)
            root = _promotion_root(payload)
            lane = lanes[lane_name]
            assert isinstance(lane, dict)
            assert payload["gate_status"] == "pass"
            assert lane["gate_status"] == "pass"
            assert lane["root_type"] == "electron"
            assert lane["root_bracket"] == root["bracket"]
            assert lane["selected_root_er"] == root["er"]
            assert lane["root_slope"] == root["slope"]


def test_finite_beta_electron_root_ladder_is_deferred_not_failed() -> None:
    summary = _load("qa_nfp2_finite_beta_electron_root_convergence_ladder.json")

    assert summary["status"] == "deferred"
    assert summary["failures"] == []
    assert summary["tiers"][0]["name"] == "low_7x7x5"
    assert summary["tiers"][1]["name"] == "mid_9x9x7"
    assert summary["tiers"][1]["convergence_gate"]["status"] == "pass"
    assert not summary["tiers"][1]["production_floor_met"]
    assert any("production floor" in blocker for blocker in summary["blockers"])


def test_finite_beta_electron_root_xblock_policy_probe_is_bounded_and_clean() -> None:
    summary = _load("qa_nfp2_finite_beta_electron_root_xblock_policy_probe.json")

    assert summary["status"] == "pass_bounded"
    assert summary["resolution"] == {"Ntheta": 17, "Nzeta": 21, "Nxi": 12, "NL": 4, "Nx": 4}
    assert summary["active_size"] == 34276
    assert summary["jax_auto"]["linear_solver_method"] == "xblock_sparse_pc_gmres"
    assert summary["jax_auto"]["residual_norm"] < summary["jax_auto"]["residual_target"]
    assert summary["jax_auto"]["wrapper_elapsed_s"] < 10.0
    assert summary["jax_vs_fortran_v3"]["FSABFlow_max_rel"] < 2.0e-6
    assert summary["production_floor"]["active_size"] > 1_000_000


def test_finite_beta_electron_root_xblock_21x25x14_probe_is_cpu_gpu_fortran_clean() -> None:
    summary = _load("qa_nfp2_finite_beta_electron_root_xblock_policy_probe_21x25x14.json")

    assert summary["status"] == "pass_bounded"
    assert summary["active_size"] == 58804
    assert summary["jax_cpu_forced_xblock"]["residual_norm"] < summary["jax_cpu_forced_xblock"]["residual_target"]
    assert summary["jax_gpu_forced_xblock"]["residual_norm"] < summary["jax_gpu_forced_xblock"]["residual_target"]
    assert summary["jax_cpu_vs_gpu"]["FSABjHat_max_rel"] < 1.0e-8
    assert summary["jax_gpu_vs_fortran_v3"]["FSABFlow_max_rel"] < 1.0e-6
    assert summary["policy_after_probe"]["default_multispecies_active_size_max"] == 60000
    assert summary["production_floor"]["active_size"] > summary["active_size"] * 10


def test_finite_beta_electron_root_xblock_25x31x16_probe_is_bounded_backend_clean() -> None:
    summary = _load("qa_nfp2_finite_beta_electron_root_xblock_policy_probe_25x31x16.json")

    assert summary["status"] == "pass_bounded"
    assert summary["active_size"] == 99_204
    assert summary["jax_cpu_forced_xblock"]["residual_norm"] < summary["jax_cpu_forced_xblock"]["residual_target"]
    assert summary["jax_cpu_auto"]["linear_solver_method"] == "xblock_sparse_pc_lgmres"
    assert summary["jax_cpu_auto"]["residual_norm"] < summary["jax_cpu_auto"]["residual_target"]
    assert summary["jax_gpu_forced_xblock"]["residual_norm"] < summary["jax_gpu_forced_xblock"]["residual_target"]
    assert summary["jax_cpu_forced_xblock"]["wrapper_elapsed_s"] < 90.0
    assert summary["jax_gpu_forced_xblock"]["wrapper_elapsed_s"] < 300.0
    assert summary["jax_cpu_vs_gpu"]["FSABjHat_max_rel"] < 1.0e-8
    assert summary["jax_cpu_vs_gpu"]["heatFlux_vm_psiHat_max_rel"] < 5.0e-8
    assert summary["jax_gpu_vs_fortran_v3"]["particleFlux_vm_psiHat_max_rel"] < 1.0e-6
    assert summary["policy_after_probe"]["default_multispecies_active_size_max"] == 100000
    assert summary["policy_after_probe"]["default_multispecies_nxi_max"] == 16
    assert summary["production_floor"]["active_size"] > summary["active_size"] * 10


def test_finite_beta_xblock_policy_probe_window_stays_below_production_floor() -> None:
    probe_names = [
        "qa_nfp2_finite_beta_electron_root_xblock_policy_probe.json",
        "qa_nfp2_finite_beta_electron_root_xblock_policy_probe_21x25x14.json",
        "qa_nfp2_finite_beta_electron_root_xblock_policy_probe_25x31x16.json",
    ]
    probes = [_load(name) for name in probe_names]

    active_sizes = [probe["active_size"] for probe in probes]
    nxi_values = [probe["resolution"]["Nxi"] for probe in probes]
    production_floor = probes[-1]["production_floor"]

    assert active_sizes == sorted(active_sizes)
    assert nxi_values == sorted(nxi_values)
    assert all(probe["status"] == "pass_bounded" for probe in probes)
    assert all(probe["n_species"] == 2 for probe in probes)
    assert all(probe["production_floor"] == production_floor for probe in probes)
    assert all(probe["active_size"] < production_floor["active_size"] for probe in probes)
    assert active_sizes[-1] < production_floor["active_size"] / 10

    for probe in probes[1:]:
        policy = probe["policy_after_probe"]
        assert policy["default_multispecies_active_size_min"] == 30_000
        assert policy["default_multispecies_active_size_max"] >= probe["active_size"]
        assert policy["default_multispecies_nxi_min"] == 12
        assert policy["default_multispecies_nxi_max"] >= probe["resolution"]["Nxi"]


def test_qi_electron_root_nfp_screen_recommends_qi_nfp2_without_kinetic_claim() -> None:
    payload = _load("qi_electron_root_nfp_screen.json")

    assert payload["workflow"] == "sfincs_jax_qi_qa_electron_root_nfp_screening_proxy"
    assert "not a kinetic SFINCS transport claim" in payload["claim_boundary"]
    best = payload["recommended_candidate"]
    assert isinstance(best, dict)
    assert best["candidate"] == "qi:nfp2"
    assert best["symmetry"] == "qi"
    assert best["nfp"] == 2
    assert best["screening_gate"]["status"] == "defer"

    candidates = payload["candidates"]
    assert isinstance(candidates, list)
    labels = {row["candidate"] for row in candidates}
    assert {"qa:nfp2", "qi:nfp1", "qi:nfp2", "qi:nfp5"}.issubset(labels)
    qi_candidates = [row for row in candidates if row["symmetry"] == "qi"]
    assert qi_candidates
    assert all(row["screening_gate"]["claim_boundary"].startswith("proxy screening") for row in qi_candidates)
    assert all(row["final_components"]["electron_root_drive"] > 0.0 for row in qi_candidates)
    assert all(row["final_components"]["symmetry_regularization"] >= 0.0 for row in qi_candidates)

    plan = payload["promotion_plan"]
    assert plan["recommended_candidate"] == "qi:nfp2"
    assert any("scan-er" in command for command in plan["next_commands"])
    assert "positive ambipolar root bracket from completed kinetic scan" in plan["required_gates"]


def test_qa_bootstrap_current_comparison_artifact_is_vmec_backed_and_finite_iota() -> None:
    payload = _load("qa_nfp2_bootstrap_current_comparison.json")

    assert payload["workflow"] == "sfincs_jax_vmec_jax_qa_optimization_current_diagnostic"
    assert "vmec_jax QA_optimization.py outputs" in payload["claim_boundary"]
    assert "not a completed high-fidelity sfincs_jax kinetic bootstrap-current claim" in payload["claim_boundary"]
    assert payload["targets"] == {"aspect_ratio": 5.0, "iota": 0.41}
    assert payload["comparison"]["status"] == "baseline_only"
    qa = payload["qa_optimization"]
    assert qa["gate"]["status"] == "pass"
    assert abs(qa["metrics"]["aspect_ratio"] - 5.0) < 5.0e-3
    assert abs(qa["metrics"]["mean_iota"] - 0.41) < 2.0e-2
    assert qa["metrics"]["jdotb_over_root_bdotb_rms"] > 0.0
    assert "preserve finite target iota" in " ".join(payload["promotion_plan"]["required_gates"])
    assert "FSABjHatOverRootFSAB2" in " ".join(payload["promotion_plan"]["required_gates"])


def test_qi_nfp2_lowres_kinetic_electron_root_artifacts_pass_cpu_gpu_and_reference_gates() -> None:
    cpu = _load("qi_nfp2_electron_root_lowres_cpu.json")
    gpu = _load("qi_nfp2_electron_root_lowres_gpu.json")
    fortran = _load("qi_nfp2_electron_root_lowres_fortran.json")
    comparison = _load("qi_nfp2_electron_root_lowres_reference_tolerance_comparison.json")

    for payload in (cpu, gpu, fortran):
        assert payload["workflow"] == "sfincs_jax_optimization_high_fidelity_promotion"
        assert payload["gate_status"] == "pass"
        assert payload["failures"] == []
        root = payload["selected_root"]
        assert root["root_type"] == "electron"
        assert root["bracket"] == [2.0, 3.0]
        assert 2.43 < root["er"] < 2.45
        assert root["slope"] > 0.0
        assert len(payload["runs"]) == 8

    assert abs(cpu["selected_root"]["er"] - gpu["selected_root"]["er"]) < 1.0e-10
    assert abs(cpu["selected_root"]["er"] - fortran["selected_root"]["er"]) < 2.0e-6

    for run in cpu["runs"] + gpu["runs"]:
        assert run["residual_norm"] <= run["residual_target"]
        assert run["residual_gate"]["status"] == "pass"

    assert comparison["status"] == "pass"
    assert comparison["failures"] == []
    assert comparison["tolerances"] == {
        "bootstrap_objective_rtol": 1.0e-3,
        "flux_objective_total_rtol": 1.0e-5,
        "selected_root_er_atol": 1.0e-10,
        "selected_root_er_rtol": 1.0e-6,
    }
    assert comparison["comparisons"]["cpu_gpu"]["status"] == "pass"
    assert comparison["comparisons"]["sfincs_jax_fortran_v3"]["status"] == "pass"


def test_qi_nfp2_first_refined_resolution_cpu_gpu_fortran_artifacts_pass_but_remain_unconverged() -> None:
    lowres = _load("qi_nfp2_electron_root_lowres_cpu.json")
    cpu = _load("qi_nfp2_electron_root_res9_cpu.json")
    gpu = _load("qi_nfp2_electron_root_res9_gpu.json")
    fortran = _load("qi_nfp2_electron_root_res9_fortran.json")
    cpu_gpu_comparison = _load("qi_nfp2_electron_root_res9_cpu_gpu.json")
    reference_comparison = _load("qi_nfp2_electron_root_res9_reference_tolerance_comparison.json")

    for payload in (cpu, gpu, fortran):
        assert payload["workflow"] == "sfincs_jax_optimization_high_fidelity_promotion"
        assert payload["gate_status"] == "pass"
        assert payload["failures"] == []
        assert payload["flux_objective"] is None
        root = payload["selected_root"]
        assert root["root_type"] == "electron"
        assert root["bracket"] == [2.0, 3.0]
        assert 2.28 < root["er"] < 2.29
        assert root["slope"] > 0.0
        assert len(payload["runs"]) == 8
        if payload in (cpu, gpu):
            for run in payload["runs"]:
                assert run["residual_norm"] <= run["residual_target"]
                assert run["residual_gate"]["status"] == "pass"

    assert cpu_gpu_comparison["status"] == "pass"
    assert cpu_gpu_comparison["failures"] == []
    assert cpu_gpu_comparison["comparisons"]["cpu_gpu"]["status"] == "pass"
    assert abs(cpu["selected_root"]["er"] - gpu["selected_root"]["er"]) < 1.0e-10
    assert reference_comparison["status"] == "pass"
    assert reference_comparison["failures"] == []
    assert reference_comparison["tolerances"] == {
        "bootstrap_objective_rtol": 1.0e-4,
        "flux_objective_total_rtol": 1.0e-6,
        "selected_root_er_atol": 1.0e-10,
        "selected_root_er_rtol": 2.0e-6,
    }
    assert reference_comparison["comparisons"]["cpu_gpu"]["status"] == "pass"
    assert reference_comparison["comparisons"]["sfincs_jax_fortran_v3"]["status"] == "pass"
    assert abs(cpu["selected_root"]["er"] - fortran["selected_root"]["er"]) < 5.0e-6

    root_drift = abs(cpu["selected_root"]["er"] - lowres["selected_root"]["er"])
    assert root_drift > 1.0e-1


def test_qi_nfp2_second_refined_resolution_dense8000_artifacts_pass_but_remain_unconverged() -> None:
    lowres = _load("qi_nfp2_electron_root_lowres_cpu.json")
    res9 = _load("qi_nfp2_electron_root_res9_cpu.json")
    cpu = _load("qi_nfp2_electron_root_res11_cpu_dense8000_default.json")
    gpu = _load("qi_nfp2_electron_root_res11_gpu_dense8000_default.json")
    fortran = _load("qi_nfp2_electron_root_res11_fortran.json")
    reference_comparison = _load(
        "qi_nfp2_electron_root_res11_reference_tolerance_comparison_dense8000_default.json"
    )

    for payload in (cpu, gpu, fortran):
        assert payload["workflow"] == "sfincs_jax_optimization_high_fidelity_promotion"
        assert payload["gate_status"] == "pass"
        assert payload["failures"] == []
        assert payload["flux_objective"] is None
        root = payload["selected_root"]
        assert root["root_type"] == "electron"
        assert root["bracket"] == [2.0, 3.0]
        assert 2.22 < root["er"] < 2.23
        assert root["slope"] > 0.0
        assert len(payload["runs"]) == 8

    for payload in (cpu, gpu):
        for run in payload["runs"]:
            assert run["residual_norm"] <= run["residual_target"]
            assert run["residual_gate"]["status"] == "pass"

    assert abs(cpu["selected_root"]["er"] - gpu["selected_root"]["er"]) < 1.0e-10
    assert abs(cpu["selected_root"]["er"] - fortran["selected_root"]["er"]) < 5.0e-6

    assert reference_comparison["status"] == "pass"
    assert reference_comparison["failures"] == []
    assert reference_comparison["tolerances"] == {
        "bootstrap_objective_rtol": 1.0e-4,
        "flux_objective_total_rtol": 1.0e-6,
        "selected_root_er_atol": 1.0e-10,
        "selected_root_er_rtol": 2.0e-6,
    }
    assert reference_comparison["comparisons"]["cpu_gpu"]["status"] == "pass"
    assert reference_comparison["comparisons"]["sfincs_jax_fortran_v3"]["status"] == "pass"

    root_drift_from_res9 = abs(cpu["selected_root"]["er"] - res9["selected_root"]["er"])
    root_drift_from_lowres = abs(cpu["selected_root"]["er"] - lowres["selected_root"]["er"])
    assert root_drift_from_res9 > 5.0e-2
    assert root_drift_from_lowres > 2.0e-1
