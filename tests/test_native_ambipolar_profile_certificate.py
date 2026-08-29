from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "native_ambipolar_profile_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_native_profile_certificate_pins_case_and_claim_boundary() -> None:
    payload = _payload()
    case = payload["case"]
    source = payload["source"]
    assert isinstance(case, dict)
    assert isinstance(source, dict)
    assert payload["schema"] == "dkx.native_ambipolar_profile.v1"
    assert payload["claim_scope"] == (
        "native_w7x_pas_dkes_whole_profile_workflow_certificate"
    )
    assert source["dkx_commit"] == "cb1ed04ca7e7f320b9ebbf066157c703a27cde29"
    assert source["geometry_sha256"] == (
        "81c686e5a5bd8f38d8b1f754ebe2910951f20094bab35d73d2827d9875bb6062"
    )
    assert case["case_id"] == (
        "f284407a441b7e06c4f3a24a0b46e80676f60e42d66cf6bce84113c1d1f096bf"
    )
    assert case["physics"] == {
        "collisions": "pitch_angle_scattering",
        "magnetic_drifts": "dkes",
        "model": "full_local",
        "phi1": "off",
    }
    assert case["solver"]["relative_tolerance"] == pytest.approx(1e-9)


def test_native_profile_certificate_retains_all_roots_and_recovery() -> None:
    payload = _payload()
    acceptance = payload["acceptance"]
    profile = payload["profile"]
    assert isinstance(acceptance, dict)
    assert isinstance(profile, dict)
    assert acceptance["all_gates_pass"] is True
    assert acceptance["measured_root_counts"] == [1, 1, 3, 1, 1]
    assert acceptance["measured_max_final_bracket_width_kV_m"] == pytest.approx(
        0.0048828125
    )
    assert acceptance["measured_max_selected_primal_residual"] == pytest.approx(
        5.365974684738137e-15
    )
    assert acceptance["measured_max_root_current_bracket_fraction"] < 0.5
    assert acceptance["measured_automatic_recovery_count"] == 1
    assert acceptance["measured_scientific_array_differences"] == []
    assert profile["attempts"]["attempt_count"] == 222
    assert profile["attempts"]["executed_route_counts"] == {
        "block_tridiagonal_truncated": 221,
        "gmres": 1,
    }
    recovery = profile["attempts"]["recoveries"][0]
    assert recovery["surface_index"] == 4
    assert recovery["electric_field_kV_m"] == 0.0
    assert [attempt["accepted"] for attempt in recovery["attempts"]] == [False, True]
    assert recovery["attempts"][0]["residual"] == pytest.approx(
        8.234475195278555e-13
    )
    assert recovery["attempts"][1]["residual"] == pytest.approx(
        1.9323484429235053e-13
    )


def test_native_profile_certificate_audit_recomputes_all_gates() -> None:
    completed = subprocess.run(
        [sys.executable, "tools/paper_benchmarks/audit_native_ambipolar_profile.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["measured"]["all_surfaces_bracketed"] is True
    assert report["measured"]["all_refinement_hierarchies_resolved"] is True


def test_native_profile_certificate_audit_rejects_compact_corruption(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["profile"]["surfaces"][0]["roots"][0]["electric_field_kV_m"] = 0.0
    artifact = tmp_path / "corrupt.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_native_ambipolar_profile.py",
            "--artifact",
            str(artifact),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert "compact profile checksum mismatch" in report["errors"]


def test_native_profile_certificate_keeps_promotion_gaps_explicit() -> None:
    assert {
        "not_phase_space_convergence_validation",
        "not_continuous_branch_event_localization",
        "not_experimental_validation",
        "not_full_fokker_planck_ambipolar_validation",
        "not_phi1_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_second_stellarator_family_validation",
        "not_cross_code_performance_validation",
    } <= set(_payload()["exclusions"])
