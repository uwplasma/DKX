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

