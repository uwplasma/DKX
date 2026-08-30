from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation/w7x_fixed_field_resolution_referee_v1.json"
SCRIPT = ROOT / "tools/paper_benchmarks/audit_w7x_fixed_field_resolution_referee.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_w7x_fixed_field_resolution_referee", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_w7x_fixed_field_resolution_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["external_results_verified"] is False
    assert report["maximum_cross_code_scaled_error"] == pytest.approx(
        0.0026879401726477422
    )


def test_independent_pitch_ladder_and_high_rung_parity_are_retained() -> None:
    payload = _payload()
    pitch = payload["comparisons"]["sfincs_pitch"]
    assert [item["label"] for item in pitch] == [
        "sfincs_pitch_52_to_70",
        "sfincs_pitch_70_to_90",
        "sfincs_pitch_90_to_120",
        "sfincs_pitch_120_to_150",
    ]
    assert pitch[0]["heat_flux_max_scaled_movement"] > 0.16
    assert (
        max(
            pitch[-1]["particle_flux_max_scaled_movement"],
            pitch[-1]["heat_flux_max_scaled_movement"],
            pitch[-1]["parallel_current_scaled_movement"],
        )
        < 0.002
    )
    assert (
        payload["comparisons"]["dkx_sfincs_pitch150_zeta37_parity"][
            "maximum_scaled_error"
        ]
        < 0.005
    )


def test_joint_transport_flux_axes_pass_without_promoting_current() -> None:
    payload = _payload()
    outcome = payload["outcome"]
    assert outcome["sfincs_pitch_converged"] is True
    assert outcome["dkx_pitch_converged_at_zeta85"] is True
    assert outcome["dkx_zeta85_flux_converged_against_zeta109"] is True
    assert outcome["dkx_speed8_flux_converged_against_speed10"] is True
    assert outcome["dkx_theta15_flux_converged_against_theta19"] is True
    assert outcome["transport_flux_fixed_field_admitted"] is True
    assert outcome["parallel_current_theta_converged"] is False
    assert outcome["parallel_current_status"] == "refinement_exhausted"
    assert outcome["whole_profile_admitted"] is False


def test_theta_current_failure_remains_visible() -> None:
    theta = _payload()["comparisons"]["dkx_theta"]
    assert all(item["parallel_current_scaled_movement"] > 0.05 for item in theta)
    assert all(item["particle_flux_max_scaled_movement"] < 0.01 for item in theta)
    assert all(item["heat_flux_max_scaled_movement"] < 0.011 for item in theta)


def test_audit_rejects_false_current_promotion(tmp_path: Path) -> None:
    payload = _payload()
    payload["outcome"]["parallel_current_theta_converged"] = True
    artifact = tmp_path / "tampered.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.audit(artifact)
    assert report["pass"] is False
    assert any(
        "parallel_current_theta_converged" in error for error in report["errors"]
    )
