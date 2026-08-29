from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "validation" / "full_kinetic_sfincs_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_full_kinetic_artifact_pins_matched_equations_and_rungs() -> None:
    payload = _payload()
    equations = payload["equations"]
    assert isinstance(equations, dict)
    assert payload["schema"] == "dkx.full_kinetic_sfincs.v1"
    assert payload["claim_scope"] == (
        "matched_zero_field_axisymmetric_full_fokker_planck_surface_profile"
    )
    assert equations["collision_operator"] == 0
    assert equations["collision_model"] == "full_linearized_fokker_planck"
    assert equations["electric_field"] == 0.0
    assert equations["constraint_input"] == -1
    assert equations["constraint_resolved"] == 1
    assert equations["include_x_dot"] is True
    assert equations["include_electric_field_term_in_xi_dot"] is True
    assert equations["use_dkes_exb_drift"] is False

    rungs = payload["rungs"]
    assert isinstance(rungs, list)
    assert [rung["id"] for rung in rungs] == ["high", "ultra"]
    assert [rung["resolution"]["matrix_size"] for rung in rungs] == [6887, 12509]


def test_full_kinetic_artifact_has_strict_residual_parity_and_convergence_gates() -> None:
    acceptance = _payload()["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["all_gates_pass"] is True
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        2.6843339656885753e-10
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.002797721820022167
    )
    assert acceptance["measured_max_near_zero_absolute_value"] == pytest.approx(
        3.625981422302594e-13
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(1.8153328e-11)
    assert (
        acceptance["measured_max_cross_code_scaled_error"]
        < acceptance["max_cross_code_scaled_error"]
    )
    assert (
        acceptance["measured_max_high_to_ultra_scaled_movement"]
        < acceptance["max_high_to_ultra_scaled_movement"]
    )


def test_full_kinetic_audit_recomputes_all_checked_gates() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py"],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["external_results_verified"] is False


def test_full_kinetic_audit_rejects_tampered_compact_output(tmp_path: Path) -> None:
    payload = _payload()
    payload["rungs"][1]["dkx"]["outputs"]["FSABFlow"] *= 1.01
    artifact = tmp_path / "tampered.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py",
            "--artifact",
            str(artifact),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert report["pass"] is False
    assert "ultra: dkx compact-output checksum mismatch" in report["errors"]


def test_full_kinetic_claim_keeps_later_validation_gates_explicit() -> None:
    exclusions = set(_payload()["exclusions"])
    assert {
        "not_multispecies_validation",
        "not_stellarator_full_fokker_planck_validation",
        "not_finite_er_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    } <= exclusions
