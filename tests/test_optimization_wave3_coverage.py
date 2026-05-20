from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


_REPO = Path(__file__).resolve().parents[1]


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_optional_objective_summary_covers_missing_and_deferred_backend_branches() -> None:
    mod = _load_script(
        _REPO / "examples" / "optimization" / "benchmark_optional_eqx_jaxopt_scheme4_gate.py",
        "wave3_eqx_jaxopt_gate",
    )

    eqx_bad = mod.ObjectiveGateResult(
        case="scheme4_geometry_fit",
        backend="equinox_wrapper",
        status="ok",
        initial_loss=1.0,
        final_loss=1.0,
        loss_ratio=1.0,
        directional_grad=2.0,
        finite_difference_grad=1.0,
        directional_grad_abs_error=1.0,
        final_param_error=0.0,
        elapsed_s=0.01,
    )
    summary = mod.summarize_gate_results([eqx_bad])

    assert summary["rows"] == 1
    assert summary["measured_rows"] == 1
    assert summary["backends"] == ["equinox_wrapper"]
    assert summary["adoption_decision"]["equinox"] == "defer_gradient_gate_not_clean"
    assert summary["adoption_decision"]["jaxopt"] == "not_evaluated"
    assert summary["adoption_decision"]["hard_dependency"] is False

    jaxopt_bad = mod.ObjectiveGateResult(
        case="scheme4_geometry_fit",
        backend="jaxopt_gradient_descent",
        status="ok",
        initial_loss=4.0,
        final_loss=2.0,
        loss_ratio=0.5,
        directional_grad=None,
        finite_difference_grad=None,
        directional_grad_abs_error=None,
        final_param_error=1.0,
        elapsed_s=0.02,
    )
    summary = mod.summarize_gate_results([jaxopt_bad])

    assert summary["adoption_decision"]["equinox"] == "not_evaluated"
    assert summary["adoption_decision"]["jaxopt"] == "defer_optimization_gate_not_clean"
    assert summary["evidence"]["jaxopt_loss_ratio"] == 0.5
    assert summary["evidence"]["jaxopt_final_param_error"] == 1.0


def test_vmec_workflow_status_helpers_are_skip_safe_and_strict(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    mod = _load_script(
        _REPO / "examples" / "optimization" / "vmec_jax_workflow_status.py",
        "wave3_vmec_jax_workflow_status",
    )

    proxy_summary = tmp_path / "proxy summary.json"
    command = mod._proxy_gate_command(
        wout=None,
        proxy_summary_json=proxy_summary,
        steps=3,
    )
    assert "/path/to/wout_circular_tokamak.nc" in command
    assert "--steps 3" in command
    assert str(proxy_summary) in command

    def fake_backend_report() -> dict[str, object]:
        return {"backends": {"booz_xform_jax": True, "vmec_jax": False}}

    def fake_summary(*, backend_status):
        return {
            "workflow": "vmec_jax_to_boozer_sfincs_geometry_proxy",
            "workflow_contract": {
                "differentiated_graph": ["booz_xform_jax"],
                "outside_differentiated_graph": ["SFINCS kinetic transport solve"],
            },
            "no_overclaim_gate": {"status": "pass"},
            "kinetic_transport_scalar_contract": {"status": "pass"},
            "claims": {
                "not_claimed": "full VMEC-boundary-to-SFINCS kinetic transport gradients",
            },
            "backend_status": dict(backend_status),
        }

    def fake_readiness_gate() -> dict[str, object]:
        return {
            "status": "pass",
            "optional_dependencies_required": False,
            "max_gradient_abs_error": 0.0,
            "gradient_tolerance": 1.0e-6,
        }

    monkeypatch.setattr(mod, "optional_jax_geometry_backend_report", fake_backend_report)
    monkeypatch.setattr(mod, "geometry_proxy_workflow_summary", fake_summary)
    monkeypatch.setattr(
        mod,
        "geometry_proxy_no_solve_provenance_gate",
        lambda summary: {
            "status": "pass",
            "kinetic_solve_executed": False,
            "kinetic_transport_scalar_contract_gate": {"status": "pass"},
            "summary_workflow": summary["workflow"],
        },
    )
    monkeypatch.setattr(mod, "boozer_spectrum_proxy_transport_gradient_gate", fake_readiness_gate)

    status = mod.build_status(wout=Path("/tmp/wout.nc"), proxy_summary_json=proxy_summary, steps=3)
    assert status["status"] == "skipped"
    assert status["skip_reason"] == "missing optional backends: vmec_jax"
    assert status["backend_readiness_gate"]["optional_dependencies_required"] is False
    assert status["no_solve_provenance_gate"]["kinetic_solve_executed"] is False
    assert "--wout /tmp/wout.nc" in status["commands"]["proxy_gradient_gate"]
    assert status["differentiability_contract"]["not_claimed"].startswith("full VMEC-boundary")

    out_json = tmp_path / "status.json"
    rc = mod.main(["--json", "--out-json", str(out_json), "--strict"])
    captured = capsys.readouterr()
    persisted = json.loads(out_json.read_text(encoding="utf-8"))

    assert rc == 2
    assert json.loads(captured.out) == persisted
    assert persisted["status"] == "skipped"


def test_vmec_workflow_status_human_output_handles_ready_payload(monkeypatch, capsys) -> None:
    mod = _load_script(
        _REPO / "examples" / "optimization" / "vmec_jax_workflow_status.py",
        "wave3_vmec_jax_workflow_status_ready",
    )
    ready_payload = {
        "workflow": "vmec_jax_to_boozer_sfincs_geometry_proxy",
        "status": "ready",
        "skip_reason": None,
        "optional_backends": {"booz_xform_jax": True, "vmec_jax": True},
        "backend_readiness_gate": {
            "status": "pass",
            "optional_dependencies_required": False,
            "max_gradient_abs_error": 0.0,
            "gradient_tolerance": 1.0e-6,
        },
        "no_solve_provenance_gate": {
            "status": "pass",
            "kinetic_solve_executed": False,
            "kinetic_transport_scalar_contract_gate": {"status": "pass"},
        },
        "differentiability_contract": {
            "differentiated_graph": ["vmec_jax", "booz_xform_jax"],
            "outside_differentiated_graph": ["SFINCS kinetic transport solve"],
            "no_overclaim_gate": {"status": "pass"},
            "not_claimed": "full transport gradients",
        },
        "commands": {"preflight": "python example.py"},
    }
    monkeypatch.setattr(mod, "build_status", lambda **kwargs: ready_payload)

    assert mod.main([]) == 0
    output = capsys.readouterr().out
    assert "VMEC JAX workflow status: ready" in output
    assert "booz_xform_jax: available" in output
    assert "no-overclaim gate: pass" in output
