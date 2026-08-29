from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation/native_physical_flux_sfincs_v1.json"
SCRIPT = ROOT / "tools/paper_benchmarks/audit_native_physical_flux_sfincs.py"
SPEC = importlib.util.spec_from_file_location("audit_native_physical_flux_sfincs", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_native_physical_flux_sfincs_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["external_results_verified"] is False
    assert report["max_cross_code_scaled_error"] == pytest.approx(
        0.0031635807589954893
    )


def test_certificate_pins_the_matched_pas_dkes_case() -> None:
    payload = _payload()
    case = payload["case"]
    assert payload["schema"] == "dkx.native_physical_flux_sfincs.v1"
    assert case["collision_operator"] == "pitch_angle_scattering"
    assert case["use_dkes_exb_drift"] is True
    assert case["phi1"] is False
    assert case["electric_field_kV_m"] == 8.55
    assert case["resolution"]["pitch_modes_by_speed"] == [6, 11, 19, 30, 42, 52, 52, 52]


def test_certificate_uses_the_documented_flux_factor_not_its_inverse() -> None:
    conversion = _payload()["conversion"]
    assert conversion["correct_d_dr_hat_to_d_dpsi_hat"] == pytest.approx(
        -1.3949598653433055
    )
    assert conversion["previous_inverse_d_dpsi_hat_to_d_dr_hat"] == pytest.approx(
        -0.7168665026458634
    )
    assert conversion["correct_to_previous_ratio"] == pytest.approx(1.9459130259186133)


def test_audit_rejects_a_tampered_radial_factor(tmp_path: Path) -> None:
    payload = _payload()
    payload["conversion"]["correct_d_dr_hat_to_d_dpsi_hat"] *= 0.5
    artifact = tmp_path / "tampered.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    report = MODULE.audit(artifact)
    assert report["pass"] is False
    assert any("correct radial factor mismatch" in error for error in report["errors"])


def test_claim_exclusions_prevent_overpromotion() -> None:
    payload = _payload()
    exclusions = set(payload["claim_exclusions"])
    assert {
        "whole-profile phase-space convergence",
        "ambipolar root agreement",
        "full Fokker-Planck collisions",
        "Phi1",
        "experimental validation",
        "cross-code performance equivalence",
    } <= exclusions
    supersession = payload["historical_artifact_supersession"]
    assert supersession["absolute_physical_flux_values_promoted"] is False
    assert "validation/native_ambipolar_profile_v1.json" in supersession["artifacts"]
    assert "validation/ambipolar_joint_speed_zeta_tail_v1.json" in supersession["artifacts"]
