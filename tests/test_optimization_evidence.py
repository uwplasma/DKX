from __future__ import annotations

import json
from pathlib import Path

from sfincs_jax.workflows.optimization import (
    build_promotion_evidence_plan,
    prepare_fortran_er_scan_inputs,
)


_REPO = Path(__file__).resolve().parents[1]
_INPUT = _REPO / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme11.input.namelist"


def test_promotion_evidence_plan_builds_cpu_gpu_fortran_commands(tmp_path: Path) -> None:
    plan = build_promotion_evidence_plan(
        input_namelist=_INPUT,
        out_dir=tmp_path / "campaign",
        er_values=(-1.0, 0.0, 1.0),
        include_cpu=True,
        include_gpu=True,
        include_fortran=True,
        fortran_exe=tmp_path / "sfincs",
        gpu_device="1",
        jobs=2,
        require_electron_root=False,
        impurity_species_index=2,
        target_impurity_flux=0.03,
    )
    payload = plan.as_dict()

    assert payload["workflow"] == "sfincs_jax_optimization_promotion_evidence_plan"
    assert [lane["label"] for lane in payload["lanes"]] == ["cpu", "gpu", "fortran_v3"]
    assert payload["lanes"][0]["env"] == {"JAX_PLATFORM_NAME": "cpu"}
    assert payload["lanes"][1]["env"] == {
        "CUDA_VISIBLE_DEVICES": "1",
        "JAX_PLATFORM_NAME": "gpu",
    }
    assert payload["lanes"][2]["scan_command"] is None
    assert "--allow-missing-residuals" in payload["lanes"][2]["promotion_command"]
    assert payload["comparison_command"] is not None
    assert "--fortran" in payload["comparison_command"]
    assert "--allow-missing-flux" not in payload["comparison_command"]
    assert "--jobs 2" in payload["lanes"][0]["scan_command_string"]
    assert "--allow-no-electron-root" in payload["lanes"][0]["promotion_command"]


def test_promotion_evidence_plan_allows_missing_flux_when_no_impurity_objective(tmp_path: Path) -> None:
    plan = build_promotion_evidence_plan(
        input_namelist=_INPUT,
        out_dir=tmp_path / "campaign",
        er_values=(-1.0, 1.0),
        include_cpu=True,
        include_gpu=True,
        require_electron_root=True,
        impurity_species_index=None,
    )
    payload = plan.as_dict()

    assert payload["comparison_command"] is not None
    assert "--allow-missing-flux" in payload["comparison_command"]
    assert "--impurity-species-index" not in payload["lanes"][0]["promotion_command"]


def test_prepare_fortran_er_scan_inputs_matches_scan_directory_contract(tmp_path: Path) -> None:
    inputs = prepare_fortran_er_scan_inputs(
        input_namelist=_INPUT,
        out_dir=tmp_path / "fortran_scan",
        values=(-0.5, 0.5),
    )

    assert [path.parent.name for path in inputs] == ["Er0.5", "Er-0.5"]
    assert (tmp_path / "fortran_scan" / "input.namelist").read_text(encoding="utf-8").endswith(
        "!ss NErs = 2\n!ss ErMin = -0.5\n!ss ErMax = 0.5\n"
    )
    first = inputs[0].read_text(encoding="utf-8")
    second = inputs[1].read_text(encoding="utf-8")
    assert "Er = 0.5" in first
    assert "Er = -0.5" in second
    assert "equilibriumFile" in first
