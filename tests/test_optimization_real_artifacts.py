from __future__ import annotations

import json
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]
_OPT_FIGURES = _REPO / "docs" / "_static" / "figures" / "optimization"


def _load(name: str) -> dict[str, object]:
    return json.loads((_OPT_FIGURES / name).read_text(encoding="utf-8"))


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
        root = payload["selected_root"]
        assert payload["gate_status"] == "pass"
        assert isinstance(root, dict)
        assert root["root_type"] == "electron"
        assert root["bracket"] == [0.25, 0.5]

    assert abs(cpu["selected_root"]["er"] - gpu["selected_root"]["er"]) < 1.0e-12
    assert abs(cpu["selected_root"]["er"] - fortran["selected_root"]["er"]) < 1.0e-7


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
