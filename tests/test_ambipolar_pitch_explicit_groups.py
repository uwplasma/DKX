from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "paper_benchmarks"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "audit_ambipolar_pitch_explicit_groups",
    TOOLS / "audit_ambipolar_pitch_explicit_groups.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ARTIFACT = ROOT / "validation" / "ambipolar_pitch_explicit_groups_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_explicit_group_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["admission_pass"] is True
    assert report["status"] == "refinement_exhausted"


def test_total_and_high_speed_work_are_exactly_fixed() -> None:
    rungs = _payload()["rungs"]
    assert [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs] == [
        129,
        129,
        129,
    ]
    assert [rung["allocation"]["groups"]["high_speed_nodes_4_5"] for rung in rungs] == [
        72,
        72,
        72,
    ]
    assert [rung["allocation"]["groups"] for rung in rungs] == [
        {
            "low_speed_nodes_0_1": 13,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 24,
            "intermediate_speed_nodes_2_3": 33,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 8,
            "intermediate_speed_nodes_2_3": 49,
            "high_speed_nodes_4_5": 72,
        },
    ]


def test_topology_is_stable_but_movement_gates_fail() -> None:
    payload = _payload()
    assert [
        [surface["root_count"] for surface in rung["surfaces"]]
        for rung in payload["rungs"]
    ] == [[1, 3], [1, 3], [1, 3]]
    outcome = payload["outcome"]
    assert outcome["phase_space_converged"] is False
    assert outcome["maximum_selected_movements"] == {
        "electric_field_kV_m": 1.064453125,
        "particle_flux_scaled": 0.09894109254425403,
        "heat_flux_scaled": 0.09077156078426843,
    }


def test_intermediate_replay_is_scientifically_exact() -> None:
    assert _payload()["intermediate_cold_warm_parity"] == {
        "scientific_arrays_exact": True,
        "ignored_timing_arrays": ["solve_time_s"],
        "mismatches": [],
    }


def test_residual_and_memory_gates_pass() -> None:
    payload = _payload()
    assert (
        max(
            rung["attempts"]["maximum_accepted_true_residual"]
            for rung in payload["rungs"]
        )
        <= 1.0e-12
    )
    assert (
        max(row["peak_footprint_bytes"] for row in payload["measurements"].values())
        < 24 * 2**30
    )


def test_claim_boundary_stays_explicit() -> None:
    payload = _payload()
    assert payload["claim_scope"] == "fixed_high_work_low_intermediate_pitch_diagnosis"
    assert "not_phase_space_convergence_validation" in payload["exclusions"]
    assert "not_cross_allocation_runtime_comparison" in payload["exclusions"]
