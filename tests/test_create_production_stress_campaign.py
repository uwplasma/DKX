from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "create_production_stress_campaign.py"
sys.path.insert(0, str(_SCRIPT_PATH.parent))
_SPEC = importlib.util.spec_from_file_location("create_production_stress_campaign", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
campaign = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = campaign
_SPEC.loader.exec_module(campaign)


def test_campaign_plan_lists_exact_unique_public_performance_cases() -> None:
    plan = campaign.build_campaign_plan(
        fortran_exe="/opt/sfincs/bin/sfincs",
        gpu_device="1",
    )

    assert plan["kind"] == "sfincs_jax_production_stress_campaign_plan"
    assert plan["case_count"] == 31
    cases = {row["case"]: row for row in plan["cases"]}
    assert len(cases) == 31
    assert "additional_examples" in cases
    assert "transportMatrix_geometryScheme2" in cases
    assert "HSX_FPCollisions_DKESTrajectories" in cases

    additional = cases["additional_examples"]
    assert additional["reasons"] == ["below_public_benchmark_resolution_floor"]
    assert "--pattern '^additional_examples$'" in additional["commands"]["cpu"]
    assert "--examples-root benchmarks/production_resolution_inputs_2026-05-04/inputs" in additional["commands"]["cpu"]
    assert "--production-manifest benchmarks/production_resolution_inputs_2026-05-04/manifest.json" in additional["commands"]["cpu"]
    assert "CUDA_VISIBLE_DEVICES=1 JAX_PLATFORM_NAME=gpu" in additional["commands"]["gpu"]
    assert "--fortran-exe /opt/sfincs/bin/sfincs" in additional["commands"]["gpu"]

    hsx = cases["HSX_FPCollisions_DKESTrajectories"]
    assert hsx["timeout_s"] >= additional["timeout_s"]
    assert hsx["total_unknowns_estimate"] > additional["total_unknowns_estimate"]


def test_campaign_writer_emits_json_and_shell_index(tmp_path: Path) -> None:
    assert campaign.main(["--out-root", str(tmp_path), "--fortran-exe", "/opt/sfincs", "--json"]) == 0

    plan_path = tmp_path / "campaign_plan.json"
    commands_path = tmp_path / "commands.sh"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    commands = commands_path.read_text(encoding="utf-8")

    assert plan["case_count"] == 31
    assert commands_path.exists()
    assert "Run selected lines manually" in commands
    assert "# CPU:" in commands
    assert "# GPU:" in commands
