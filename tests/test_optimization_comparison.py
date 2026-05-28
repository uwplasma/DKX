from __future__ import annotations

import json
from pathlib import Path

import pytest

from sfincs_jax.optimization_comparison import (
    PromotionComparisonTolerances,
    compare_cpu_gpu_promotions,
    compare_fortran_promotion,
    compare_optimization_promotions,
    load_promotion_payload,
)


def _promotion_payload(
    *,
    er: float = 0.42,
    bootstrap_objective: float = 2.5e-4,
    flux_total: float = 1.25e-3,
    gate_status: str = "pass",
) -> dict[str, object]:
    return {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
        "selected_root": {
            "er": er,
            "root_type": "electron",
            "bracket": [0.25, 0.75],
            "slope": 1.1,
        },
        "bootstrap_objective": bootstrap_objective,
        "flux_objective": {
            "total": flux_total,
            "main_particle": 1.0e-4,
            "main_heat": 2.0e-4,
            "impurity_penalty": 0.0,
        },
        "gate_status": gate_status,
        "failures": [] if gate_status == "pass" else ["synthetic failure"],
    }


def test_compare_cpu_gpu_promotions_passes_within_tolerances() -> None:
    tolerances = PromotionComparisonTolerances(
        selected_root_er_rtol=1.0e-5,
        selected_root_er_atol=1.0e-8,
        bootstrap_objective_rtol=1.0e-5,
        flux_objective_total_rtol=1.0e-5,
    )
    cpu = _promotion_payload()
    gpu = _promotion_payload(
        er=0.4200005,
        bootstrap_objective=2.50001e-4,
        flux_total=1.250005e-3,
    )

    result = compare_cpu_gpu_promotions(cpu, gpu, tolerances=tolerances)

    assert result["status"] == "pass"
    assert result["failures"] == []
    assert result["metrics"]["selected_root_er"]["status"] == "pass"
    assert result["metrics"]["bootstrap_objective"]["rel_diff"] > 0.0
    assert result["metrics"]["flux_objective_total"]["status"] == "pass"


def test_compare_cpu_gpu_promotions_reports_numeric_and_gate_failures() -> None:
    tolerances = PromotionComparisonTolerances(
        selected_root_er_rtol=1.0e-6,
        selected_root_er_atol=0.0,
        bootstrap_objective_rtol=1.0e-6,
        flux_objective_total_rtol=1.0e-6,
    )
    cpu = _promotion_payload()
    gpu = _promotion_payload(
        er=0.45,
        bootstrap_objective=3.0e-4,
        flux_total=1.5e-3,
        gate_status="fail",
    )

    result = compare_cpu_gpu_promotions(cpu, gpu, tolerances=tolerances)

    assert result["status"] == "fail"
    assert result["metrics"]["gate_status"]["status"] == "fail"
    assert result["metrics"]["selected_root_er"]["status"] == "fail"
    assert result["metrics"]["bootstrap_objective"]["status"] == "fail"
    assert result["metrics"]["flux_objective_total"]["status"] == "fail"
    assert any("gpu gate_status" in failure for failure in result["failures"])
    assert any("selected_root_er differs" in failure for failure in result["failures"])


def test_compare_cpu_gpu_promotions_reports_missing_and_bad_payload_keys() -> None:
    cpu = {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
        "bootstrap_objective": 2.5e-4,
        "flux_objective": {"total": 1.25e-3},
    }
    gpu = {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
        "gate_status": "pass",
        "selected_root": {"er": "not-a-number"},
        "bootstrap_objective": float("inf"),
        "flux_objective": {"total": 1.25e-3},
    }

    result = compare_cpu_gpu_promotions(cpu, gpu)

    assert result["status"] == "fail"
    assert result["metrics"]["gate_status"]["status"] == "fail"
    assert result["metrics"]["selected_root_er"]["status"] == "fail"
    assert result["metrics"]["bootstrap_objective"]["status"] == "fail"
    assert result["metrics"]["flux_objective_total"]["status"] == "pass"
    assert any("cpu gate_status is None" in failure for failure in result["failures"])
    assert any("cpu is missing finite selected_root_er" in failure for failure in result["failures"])
    assert any("gpu is missing finite bootstrap_objective" in failure for failure in result["failures"])


def test_load_promotion_payload_rejects_non_object_json(tmp_path: Path) -> None:
    payload = tmp_path / "promotion.json"
    payload.write_text("[]\n", encoding="utf-8")

    with pytest.raises(TypeError, match="must decode to an object"):
        load_promotion_payload(payload)


def test_compare_fortran_promotion_accepts_paths_and_flat_payload_keys(tmp_path: Path) -> None:
    tolerances = PromotionComparisonTolerances(
        selected_root_er_rtol=1.0e-4,
        selected_root_er_atol=1.0e-8,
        bootstrap_objective_rtol=1.0e-4,
        flux_objective_total_rtol=1.0e-4,
    )
    jax_path = tmp_path / "jax_promotion.json"
    fortran_path = tmp_path / "fortran_promotion.json"
    jax_path.write_text(json.dumps(_promotion_payload()), encoding="utf-8")
    fortran_path.write_text(
        json.dumps(
            {
                "selected_root_er": 0.420001,
                "bootstrap_objective": 2.5001e-4,
                "flux_objective_total": 1.2501e-3,
                "gate_status": "pass",
            }
        ),
        encoding="utf-8",
    )

    result = compare_fortran_promotion(jax_path, fortran_path, tolerances=tolerances)

    assert result["status"] == "pass"
    assert result["reference_label"] == "sfincs_jax"
    assert result["candidate_label"] == "fortran_v3"
    assert result["metrics"]["selected_root_er"]["candidate"] == 0.420001


def test_compare_optimization_promotions_includes_optional_fortran() -> None:
    tolerances = PromotionComparisonTolerances(
        selected_root_er_rtol=1.0e-4,
        selected_root_er_atol=1.0e-8,
        bootstrap_objective_rtol=1.0e-4,
        flux_objective_total_rtol=1.0e-4,
    )
    cpu = _promotion_payload()
    gpu = _promotion_payload(er=0.420001)
    fortran = _promotion_payload(er=0.420002)

    result = compare_optimization_promotions(
        cpu,
        gpu,
        fortran_v3_payload=fortran,
        tolerances=tolerances,
    )

    assert result["status"] == "pass"
    assert set(result["comparisons"]) == {"cpu_gpu", "sfincs_jax_fortran_v3"}
    assert result["comparisons"]["cpu_gpu"]["status"] == "pass"
    assert result["comparisons"]["sfincs_jax_fortran_v3"]["status"] == "pass"


def test_compare_cpu_gpu_promotions_can_skip_missing_flux_objective() -> None:
    cpu = _promotion_payload()
    gpu = _promotion_payload()
    cpu["flux_objective"] = None
    gpu["flux_objective"] = None

    default_result = compare_cpu_gpu_promotions(cpu, gpu)
    skipped_result = compare_cpu_gpu_promotions(cpu, gpu, require_flux_objective=False)

    assert default_result["status"] == "fail"
    assert any("flux_objective_total" in failure for failure in default_result["failures"])
    assert skipped_result["status"] == "pass"
    assert "flux_objective_total" not in skipped_result["metrics"]
