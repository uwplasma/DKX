from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools" / "paper_benchmarks"
sys.path.insert(0, str(TOOLS))
SPEC = importlib.util.spec_from_file_location(
    "audit_ambipolar_pitch_speed_groups",
    TOOLS / "audit_ambipolar_pitch_speed_groups.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

ARTIFACT = ROOT / "validation" / "ambipolar_pitch_speed_groups_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_pitch_speed_group_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["admission_pass"] is True
    assert report["status"] == "diagnostic_complete"


def test_allocations_are_fixed_work_and_grouped() -> None:
    rungs = _payload()["rungs"]
    assert [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs] == [
        132,
        129,
        133,
    ]
    assert [rung["allocation"]["groups"] for rung in rungs] == [
        {
            "low_speed_nodes_0_1": 44,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 44,
        },
        {
            "low_speed_nodes_0_1": 13,
            "intermediate_speed_nodes_2_3": 44,
            "high_speed_nodes_4_5": 72,
        },
        {
            "low_speed_nodes_0_1": 9,
            "intermediate_speed_nodes_2_3": 36,
            "high_speed_nodes_4_5": 88,
        },
    ]


def test_root_topology_changes_at_fixed_work() -> None:
    rungs = _payload()["rungs"]
    assert [
        [surface["root_count"] for surface in rung["surfaces"]] for rung in rungs
    ] == [
        [3, 1],
        [1, 3],
        [1, 1],
    ]
    outcome = _payload()["outcome"]
    assert outcome["phase_space_converged"] is False
    assert outcome["topology_changing_comparisons"] == [
        "uniform22_to_linear36",
        "linear36_to_quadratic44",
    ]


def test_quadratic_replay_is_scientifically_exact() -> None:
    parity = _payload()["quadratic_cold_warm_parity"]
    assert parity == {
        "scientific_arrays_exact": True,
        "ignored_timing_arrays": ["solve_time_s"],
        "mismatches": [],
    }


def test_every_retained_result_meets_residual_and_memory_gates() -> None:
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


def test_artifact_keeps_claim_boundary_explicit() -> None:
    payload = _payload()
    assert payload["claim_scope"] == "fixed_work_speed_local_pitch_allocation_diagnosis"
    assert "not_phase_space_convergence_validation" in payload["exclusions"]
    assert "not_cross_allocation_runtime_comparison" in payload["exclusions"]
