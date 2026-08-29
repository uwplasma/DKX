from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "ambipolar_phase_space_ladder_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_phase_space_ladder_pins_bounded_claim_and_inputs() -> None:
    payload = _payload()
    source = payload["source"]
    assert isinstance(source, dict)
    assert payload["schema"] == "dkx.ambipolar_phase_space_ladder.v1"
    assert payload["claim_scope"] == (
        "bounded_w7x_pas_dkes_phase_space_nonconvergence_evidence"
    )
    assert source["dkx_commit"] == ("92d1823a09ed9e59a6497e0965e7a04896ffc208")
    assert source["geometry_sha256"] == (
        "81c686e5a5bd8f38d8b1f754ebe2910951f20094bab35d73d2827d9875bb6062"
    )
    assert [rung["resolution"] for rung in payload["rungs"]] == [
        {"theta": 13, "zeta": 31, "pitch": 32, "speed": 5},
        {"theta": 15, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 17, "zeta": 37, "pitch": 40, "speed": 6},
    ]


def test_phase_space_ladder_records_every_root_and_selected_observable() -> None:
    payload = _payload()
    assert [
        [surface["root_count"] for surface in rung["surfaces"]]
        for rung in payload["rungs"]
    ] == [[1, 1, 3, 1, 1]] * 3
    for rung in payload["rungs"]:
        assert sum(len(surface["roots"]) for surface in rung["surfaces"]) == 7
        for surface in rung["surfaces"]:
            selected = surface["roots"][surface["selected_root_index"]]
            assert selected["selected"] is True
            assert len(selected["particle_flux_m2_s"]) == 2
            assert len(selected["heat_flux_W_m2"]) == 2
    assert all(
        len(comparison["root_movements"]) == 7
        and len(comparison["selected_movements"]) == 5
        for comparison in payload["comparisons"]
    )


def test_phase_space_ladder_fails_admission_without_relaxing_gates() -> None:
    payload = _payload()
    admission = payload["admission"]
    latest = payload["comparisons"][-1]["summary"]
    assert admission["status"] == "refinement_exhausted"
    assert admission["admission_pass"] is False
    assert admission["limits"] == {
        "max_all_root_electric_field_movement_kV_m": 0.005,
        "max_selected_particle_flux_scaled_movement": 0.02,
        "max_selected_heat_flux_scaled_movement": 0.02,
        "max_accepted_true_residual": 1e-12,
    }
    assert latest["topology_stable"] is True
    assert latest["max_all_root_electric_field_movement_kV_m"] == pytest.approx(
        1.6259765625
    )
    assert latest["max_selected_particle_flux_scaled_movement"] == pytest.approx(
        0.040755305441301674
    )
    assert latest["max_selected_heat_flux_scaled_movement"] == pytest.approx(
        0.0781298871962563
    )
    assert admission["measured_max_accepted_true_residual"] == pytest.approx(
        3.917314213777927e-13
    )
    assert admission["failed_gates"] == [
        "all_root_electric_field_movement",
        "selected_particle_flux_movement",
        "selected_heat_flux_movement",
    ]


def test_phase_space_ladder_audit_recomputes_truthful_failure() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_phase_space_ladder.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["admission_status"] == "refinement_exhausted"
    assert report["admission_pass"] is False


def test_phase_space_ladder_audit_rejects_compact_corruption(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["comparisons"][-1]["root_movements"][0]["electric_field_movement_kV_m"] = (
        0.0
    )
    artifact = tmp_path / "corrupt.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_phase_space_ladder.py",
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
    assert "compact ladder checksum mismatch" in report["errors"]
    assert "stored comparison arithmetic mismatch" in report["errors"]


def test_phase_space_ladder_keeps_promotion_gaps_explicit() -> None:
    assert {
        "not_phase_space_convergence_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_full_fokker_planck_ambipolar_validation",
        "not_experimental_validation",
        "not_phi1_validation",
        "not_cross_code_performance_validation",
        "fine_rung_does_not_refine_zeta_or_speed_beyond_reference",
    } == set(_payload()["exclusions"])
