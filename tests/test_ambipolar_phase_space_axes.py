from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "ambipolar_phase_space_axes_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_axis_ladder_pins_claim_and_resolutions() -> None:
    payload = _payload()
    assert payload["schema"] == "dkx.ambipolar_phase_space_axes.v1"
    assert payload["claim_scope"] == (
        "bounded_w7x_pas_dkes_theta_pitch_nonconvergence_diagnosis"
    )
    assert payload["source"]["dkx_commit"] == (
        "a042bd58e95a86667185945ab85327241f47eb16"
    )
    assert [rung["resolution"] for rung in payload["rungs"]] == [
        {"theta": 15, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 17, "zeta": 37, "pitch": 36, "speed": 6},
        {"theta": 15, "zeta": 37, "pitch": 40, "speed": 6},
        {"theta": 15, "zeta": 37, "pitch": 44, "speed": 6},
    ]


def test_axis_ladder_preserves_topology_and_every_root() -> None:
    payload = _payload()
    for rung in payload["rungs"]:
        assert [surface["root_count"] for surface in rung["surfaces"]] == [
            1,
            1,
            3,
            1,
            1,
        ]
        assert sum(len(surface["roots"]) for surface in rung["surfaces"]) == 7
    for comparison in payload["comparisons"]:
        assert comparison["summary"]["topology_stable"] is True
        assert len(comparison["root_movements"]) == 7
        assert len(comparison["selected_movements"]) == 5


def test_axis_ladder_identifies_pitch_without_promoting_it() -> None:
    outcome = _payload()["outcome"]
    diagnosis = outcome["diagnosis"]
    assert outcome["status"] == "refinement_exhausted"
    assert outcome["admission_pass"] is False
    assert outcome["theta17_vs_reference"]["failed_gates"] == [
        "all_root_electric_field_movement"
    ]
    assert outcome["pitch44_vs_pitch40"]["failed_gates"] == [
        "all_root_electric_field_movement",
        "selected_particle_flux_movement",
        "selected_heat_flux_movement",
    ]
    assert diagnosis["dominant_failed_direction"] == "pitch"
    assert diagnosis["theta_max_root_movement_kV_m"] == pytest.approx(0.1611328125)
    assert diagnosis["pitch40_max_root_movement_kV_m"] == pytest.approx(1.7333984375)
    assert diagnosis["pitch40_to_pitch44_approaches_gate"] is False
    assert diagnosis["pitch48_bruteforce_admitted"] is False
    assert outcome["measured_max_accepted_true_residual"] == pytest.approx(
        3.75131427218295e-13
    )


def test_axis_ladder_audit_recomputes_negative_outcome() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_phase_space_axes.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["status"] == "refinement_exhausted"
    assert report["admission_pass"] is False
    assert report["dominant_failed_direction"] == "pitch"


def test_axis_ladder_audit_rejects_corruption(tmp_path: Path) -> None:
    payload = _payload()
    payload["comparisons"][-1]["selected_movements"][0]["heat_flux_scaled_movement"] = (
        0.0
    )
    artifact = tmp_path / "corrupt.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_phase_space_axes.py",
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
    assert "compact axes checksum mismatch" in report["errors"]
    assert "stored axis comparison arithmetic mismatch" in report["errors"]
