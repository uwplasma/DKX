from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation" / "ambipolar_pitch_budget_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_pitch_budget_pins_exact_route_parity() -> None:
    payload = _payload()
    assert payload["schema"] == "dkx.ambipolar_pitch_budget.v1"
    assert payload["source"]["bounded_dkx_commit"] == (
        "f08fd7a1c802b0d860a2d694924d33fd2e52cec0"
    )
    parity = payload["route_parity"]
    assert parity["status"] == "resolved"
    assert parity["admission_pass"] is True
    assert parity["metrics"]["roots_and_brackets_exact"] is True
    assert parity["metrics"]["full_executed_routes"] == {"block_tridiagonal": 139}
    assert parity["metrics"]["bounded_executed_routes"] == {
        "block_tridiagonal_truncated": 139
    }
    assert parity["metrics"]["max_evaluation_heat_flux_relative_difference"] < 2.0e-9
    assert parity["footprint_reduction_fraction"] == pytest.approx(0.9082292155037494)
    assert parity["warm_speedup_claim"] is False


def test_uniform_pitch_ladder_retains_every_changed_topology() -> None:
    payload = _payload()
    rungs = payload["rungs"]
    assert [rung["resolution"]["pitch"] for rung in rungs] == [22, 26, 30]
    assert [rung["resolution"]["pitch_speed_ramp"] for rung in rungs] == [0, 0, 0]
    assert [
        [surface["root_count"] for surface in rung["surfaces"]] for rung in rungs
    ] == [[3, 1, 1], [1, 1, 1], [1, 3, 1]]
    assert [
        [
            surface["roots"][surface["selected_root_index"]]["electric_field_kV_m"]
            for surface in rung["surfaces"]
        ]
        for rung in rungs
    ] == [
        [-0.361328125, -1.5966796875, -3.4375],
        [9.23828125, -0.9521484375, -3.18359375],
        [10.5615234375, 6.748046875, -2.9736328125],
    ]


def test_uniform_pitch_ladder_fails_unchanged_gates() -> None:
    payload = _payload()
    outcome = payload["outcome"]
    assert outcome["status"] == "refinement_exhausted"
    assert outcome["admission_pass"] is False
    assert outcome["uniform34_or_higher_admitted"] is False
    assert outcome["measured_max_accepted_true_residual"] == pytest.approx(
        5.641144886776424e-14
    )
    for comparison in outcome["comparisons"]:
        assert comparison["failed_gates"] == [
            "topology_stable",
            "selected_electric_field_movement",
            "selected_particle_flux_movement",
            "selected_heat_flux_movement",
        ]
    summaries = [comparison["summary"] for comparison in payload["comparisons"]]
    assert summaries[0]["max_selected_electric_field_movement_kV_m"] == pytest.approx(
        9.599609375
    )
    assert summaries[1]["max_selected_electric_field_movement_kV_m"] == pytest.approx(
        7.7001953125
    )
    assert summaries[0]["max_selected_heat_flux_scaled_movement"] > 0.55
    assert summaries[1]["max_selected_heat_flux_scaled_movement"] > 0.45


def test_pitch_budget_audit_recomputes_truthful_failure() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_pitch_budget.py",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["route_parity_pass"] is True
    assert report["status"] == "refinement_exhausted"
    assert report["admission_pass"] is False


def test_pitch_budget_audit_rejects_corruption(tmp_path: Path) -> None:
    payload = _payload()
    payload["comparisons"][0]["selected_movements"][0]["heat_flux_scaled_movement"] = (
        0.0
    )
    artifact = tmp_path / "corrupt.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_ambipolar_pitch_budget.py",
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
    assert "compact checksum mismatch" in report["errors"]
    assert "stored comparison arithmetic mismatch" in report["errors"]


def test_pitch_budget_keeps_claim_exclusions_explicit() -> None:
    exclusions = set(_payload()["exclusions"])
    assert {
        "not_phase_space_convergence_validation",
        "not_speed_or_zeta_convergence_validation",
        "not_independent_cross_code_ambipolar_validation",
        "not_full_fokker_planck_or_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
    } <= exclusions
