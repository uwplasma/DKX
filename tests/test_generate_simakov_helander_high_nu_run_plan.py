from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    repo = Path(__file__).resolve().parents[1]
    path = repo / "examples" / "publication_figures" / "generate_simakov_helander_high_nu_run_plan.py"
    spec = importlib.util.spec_from_file_location("generate_simakov_helander_high_nu_run_plan", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_high_nu_run_plan_uses_audit_extension_grid(tmp_path: Path) -> None:
    mod = _load_module()
    audit = {
        "cases": {
            "lhd": {
                "max_nuprime": 10.0,
                "recommended_high_nuprime_extension": [17.8, 31.6, 56.2, 100.0],
            },
            "w7x": {
                "max_nuprime": 10.0,
                "recommended_high_nuprime_extension": [],
            },
        }
    }

    plan = mod.build_high_nu_run_plan(
        audit,
        work_root=tmp_path / "work",
        summary_root=tmp_path / "summary",
        timeout_s=120.0,
        collision_operators="0",
        python_executable="python3",
        transport_workers=3,
        transport_parallel_backend="cpu",
        transport_sparse_direct_max=30000,
        transport_maxiter=1200,
        max_transport_residual=1.0e-7,
        max_transport_relative_residual=1.0e-8,
    )

    assert plan["metadata"]["kind"] == "simakov_helander_high_nu_run_plan"
    assert plan["metadata"]["validation_state"] == "deferred_run_plan"
    assert plan["metadata"]["publication_figure"]["claim_status"] == "proxy_or_deferred"
    assert plan["metadata"]["publication_figure"]["checked_in_converged_artifact"] is False
    assert plan["ready_to_run"] is True
    assert plan["gates"]["ready_to_run"] is True
    assert plan["gates"]["commands_require_residual_gates"] is True
    assert plan["gates"]["source_audit_keeps_full_reproduction_closed"] is False
    assert plan["gates"]["run_plan_only_not_completed_validation"] is True
    assert plan["gates"]["checked_in_converged_artifact"] is False
    assert plan["gates"]["ready_for_literature_claim"] is False
    assert [reason["code"] for reason in plan["deferred_reasons"]] == [
        "run_plan_not_completed_validation",
        "source_audit_closure_gate_missing_or_open",
    ]
    assert len(plan["runs"]) == 1
    run = plan["runs"][0]
    assert run["case"] == "lhd"
    assert run["nuprime_min"] == 17.8
    assert run["nuprime_max"] == 100.0
    assert run["n_points"] == 4
    assert "--nuprime-min" in run["command"]
    assert "--scan-only" in run["command"]
    assert "--collision-operators" in run["command"]
    assert run["command"][0] == "python3"
    assert run["command"][run["command"].index("--transport-workers") + 1] == "3"
    assert run["command"][run["command"].index("--transport-parallel-backend") + 1] == "cpu"
    assert run["command"][run["command"].index("--transport-sparse-direct-max") + 1] == "30000"
    assert run["command"][run["command"].index("--transport-maxiter") + 1] == "1200"
    assert run["command"][run["command"].index("--max-transport-residual") + 1] == "1e-07"
    assert run["command"][run["command"].index("--max-transport-relative-residual") + 1] == "1e-08"
    assert "--require-residuals" in run["command"]
    assert run["pilot_command"][0] == "python3"
    assert run["pilot_command"][run["pilot_command"].index("--collision-operators") + 1] == "0"
    assert run["pilot_command"][run["pilot_command"].index("--n-points") + 1] == "1"
    assert run["pilot_command"][run["pilot_command"].index("--nuprime-max") + 1] == "17.8"
    assert run["pilot_command"][run["pilot_command"].index("--transport-sparse-direct-max") + 1] == "30000"
    assert run["pilot_command"][run["pilot_command"].index("--transport-maxiter") + 1] == "1200"
    assert "--require-residuals" in run["pilot_command"]
    assert plan["configuration"]["transport_workers"] == 3
    assert plan["configuration"]["transport_parallel_backend"] == "cpu"
    assert plan["configuration"]["transport_sparse_direct_max"] == 30000
    assert plan["configuration"]["transport_maxiter"] == 1200
    assert plan["configuration"]["require_residuals"] is True
    assert plan["configuration"]["max_transport_residual"] == 1.0e-7
    assert plan["configuration"]["max_transport_relative_residual"] == 1.0e-8


def test_main_writes_high_nu_run_plan_from_checked_in_audit(tmp_path: Path) -> None:
    mod = _load_module()
    repo = Path(__file__).resolve().parents[1]
    audit_json = repo / "examples" / "publication_figures" / "artifacts" / "sfincs_jax_simakov_helander_limit_audit_summary.json"
    out_json = tmp_path / "plan.json"

    rc = mod.main(
        [
            "--audit-json",
            str(audit_json),
            "--out-json",
            str(out_json),
            "--work-root",
            str(tmp_path / "work"),
            "--summary-root",
            str(tmp_path / "summary"),
            "--timeout-s",
            "60",
            "--transport-workers",
            "2",
            "--transport-parallel-backend",
            "gpu",
            "--transport-sparse-direct-max",
            "30000",
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text())
    assert payload["ready_to_run"] is True
    assert {run["case"] for run in payload["runs"]} == {"lhd", "w7x"}
    assert all(run["nuprime_max"] >= 99.0 for run in payload["runs"])
    assert payload["configuration"]["transport_workers"] == 2
    assert payload["configuration"]["transport_parallel_backend"] == "gpu"
    assert payload["configuration"]["transport_sparse_direct_max"] == 30000
    assert payload["configuration"]["require_residuals"] is True
    assert payload["gates"]["commands_require_residual_gates"] is True
    assert payload["gates"]["source_audit_keeps_full_reproduction_closed"] is True
    assert payload["gates"]["ready_for_literature_claim"] is False
    assert payload["deferred_reasons"] == [
        {
            "code": "run_plan_not_completed_validation",
            "gate": "run_plan_only_not_completed_validation",
            "message": "This artifact contains commands for future scans, not checked converged scan outputs.",
        }
    ]
