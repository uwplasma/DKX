from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


_REPO = Path(__file__).resolve().parents[1]
_STATUS_SCRIPT = _REPO / "examples" / "optimization" / "vmec_jax_workflow_status.py"
_PIPELINE_SCRIPT = _REPO / "examples" / "autodiff" / "vmec_jax_to_boozer_sfincs_pipeline.py"


def test_vmec_jax_workflow_status_scaffold_is_skip_safe(tmp_path: Path) -> None:
    out_json = tmp_path / "status.json"
    proxy_summary = tmp_path / "proxy-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_STATUS_SCRIPT),
            "--json",
            "--out-json",
            str(out_json),
            "--wout",
            "/tmp/wout_circular_tokamak.nc",
            "--proxy-summary-json",
            str(proxy_summary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    persisted = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload == persisted
    assert payload["workflow"] == "vmec_jax_to_boozer_sfincs_geometry_proxy"
    assert payload["status"] in {"ready", "skipped"}
    assert set(payload["optional_backends"]) == {"vmec_jax", "booz_xform_jax"}
    assert payload["default_ci_requires_optional_backends"] is False
    assert payload["no_solve_provenance_gate"]["status"] == "pass"
    assert payload["no_solve_provenance_gate"]["kinetic_solve_executed"] is False
    assert payload["no_solve_provenance_gate"]["requires_file_provenance"] is False
    assert (
        payload["no_solve_provenance_gate"]["kinetic_transport_scalar_contract_gate"]["status"]
        == "pass"
    )
    assert "linear_kinetic_solve" in payload["no_solve_provenance_gate"][
        "required_kinetic_transport_scalar_stages"
    ]
    assert "kinetic_operator_assembly" in payload["no_solve_provenance_gate"][
        "differentiability_boundary"
    ]["not_covered_stage_names"]
    assert "booz_xform_jax" in payload["differentiability_contract"]["differentiated_graph"]
    assert "SFINCS kinetic transport solve" in payload["differentiability_contract"]["outside_differentiated_graph"]
    assert payload["differentiability_contract"]["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert (
        payload["differentiability_contract"]["kinetic_transport_scalar_contract"][
            "no_overclaim_gate"
        ]["status"]
        == "pass"
    )
    assert payload["differentiability_contract"]["not_claimed"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )
    assert "--wout /tmp/wout_circular_tokamak.nc" in payload["commands"]["proxy_gradient_gate"]
    assert f"--summary-json {proxy_summary}" in payload["commands"]["proxy_gradient_gate"]
    if payload["status"] == "skipped":
        assert payload["skip_reason"].startswith("missing optional backends:")


def test_vmec_jax_workflow_docs_are_indexed_and_command_complete() -> None:
    index = (_REPO / "docs" / "index.rst").read_text(encoding="utf-8")
    page = (_REPO / "docs" / "vmec_jax_workflow.rst").read_text(encoding="utf-8")

    assert "vmec_jax_workflow" in index
    assert "python examples/optimization/vmec_jax_workflow_status.py --json" in page
    assert "python examples/autodiff/vmec_jax_to_boozer_sfincs_pipeline.py" in page
    assert "--check-backends" in page
    assert "--summary-json workflow-summary.json" in page
    assert "no_solve_provenance_gate" in page
    assert "kinetic_transport_scalar_contract" in page
    assert "required_kinetic_transport_scalar_stages" in page
    assert "no full VMEC-boundary-to-SFINCS kinetic transport gradients" in page
    assert "tests/test_optional_ecosystem_gates.py" in page


def _optional_wout_fixture() -> Path | None:
    candidates = [
        Path("/Users/rogeriojorge/local/vmec_jax/examples/data/wout_circular_tokamak.nc"),
        Path("/Users/rogeriojorge/local/booz_xform_jax/tests/test_files/wout_circular_tokamak.nc"),
        _REPO / "tests" / "ref" / "wout_w7x_standardConfig.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def test_vmec_jax_boozer_proxy_gate_passes_or_skips_cleanly(tmp_path: Path) -> None:
    if importlib.util.find_spec("vmec_jax") is None:
        pytest.skip("vmec_jax is not installed")
    if importlib.util.find_spec("booz_xform_jax") is None:
        pytest.skip("booz_xform_jax is not installed")
    fixture = _optional_wout_fixture()
    if fixture is None:
        pytest.skip("optional VMEC wout fixture not found")

    summary_path = tmp_path / "workflow-summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(_PIPELINE_SCRIPT),
            "--wout",
            str(fixture),
            "--mboz",
            "3",
            "--nboz",
            "3",
            "--surface",
            "0.5",
            "--n-theta",
            "8",
            "--n-zeta",
            "6",
            "--steps",
            "0",
            "--summary-json",
            str(summary_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "numerical gradient gate: pass" in result.stdout
    assert summary["numerical_gradient_gate"]["status"] == "pass"
    assert summary["no_solve_provenance_gate"]["status"] == "pass"
    assert summary["no_solve_provenance_gate"]["kinetic_solve_executed"] is False
    assert summary["no_solve_provenance_gate"]["missing_file_provenance_fields"] == []
    assert (
        summary["no_solve_provenance_gate"]["kinetic_transport_scalar_contract_gate"]["status"]
        == "pass"
    )
    assert "gradient_validation" in summary["no_solve_provenance_gate"][
        "required_kinetic_transport_scalar_stages"
    ]
    assert set(summary["no_solve_provenance_gate"]["present_file_provenance_fields"]) == {
        "source",
        "selected_surface",
        "boozer_resolution",
        "grid_shape",
        "scale",
    }
    assert summary["no_overclaim_gate"]["full_transport_gradients_claimed"] is False
    assert summary["claims"]["not_claimed"] == (
        "full VMEC-boundary-to-SFINCS kinetic transport gradients"
    )
