from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "validation" / "full_kinetic_sfincs_finite_er_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_finite_er_artifact_pins_matched_full_kinetic_equations() -> None:
    payload = _payload()
    equations = payload["equations"]
    assert isinstance(equations, dict)
    assert payload["schema"] == "dkx.full_kinetic_sfincs_finite_er.v1"
    assert payload["claim_scope"] == (
        "matched_finite_er_axisymmetric_full_fokker_planck_surface_profile"
    )
    assert equations["collision_operator"] == 0
    assert equations["electric_field"] == -30.0
    assert equations["constraint_input"] == -1
    assert equations["constraint_resolved"] == 1
    assert equations["include_x_dot"] is True
    assert equations["include_electric_field_term_in_xi_dot"] is True
    assert equations["use_dkes_exb_drift"] is False
    assert equations["solver_tolerance"] == pytest.approx(1e-13)


def test_finite_er_artifact_has_resolution_parity_and_residual_gates() -> None:
    payload = _payload()
    rungs = payload["rungs"]
    assert isinstance(rungs, list)
    assert [rung["id"] for rung in rungs] == ["high", "ultra"]
    assert [rung["resolution"]["sfincs_matrix_size"] for rung in rungs] == [
        6887,
        12509,
    ]
    assert [rung["resolution"]["dkx_matrix_size"] for rung in rungs] == [
        10532,
        18614,
    ]

    acceptance = payload["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["all_gates_pass"] is True
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        1.8756861003267208e-9
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.0032524453653068005
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(
        5.2475734e-11
    )


def test_shared_full_kinetic_audit_accepts_finite_er_artifact() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py",
            "--artifact",
            str(_ARTIFACT),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["measured"]["max_cross_code_scaled_error"] < 1e-8


def test_finite_er_claim_keeps_later_gates_explicit() -> None:
    exclusions = set(_payload()["exclusions"])
    assert {
        "not_multispecies_validation",
        "not_stellarator_full_fokker_planck_validation",
        "not_er_scan_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    } <= exclusions
