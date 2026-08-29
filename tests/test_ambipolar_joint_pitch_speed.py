from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/paper_benchmarks/audit_ambipolar_joint_pitch_speed.py"
SPEC = importlib.util.spec_from_file_location("audit_ambipolar_joint_pitch_speed", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / "validation/ambipolar_joint_pitch_speed_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_joint_pitch_speed_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["phase_space_converged"] is False


def test_speed_and_pitch_gates_both_fail() -> None:
    gates = _payload()["outcome"]["gates"]
    assert gates["speed_particle_below_2_percent"] is False
    assert gates["speed_heat_below_2_percent"] is False
    assert gates["pitch_particle_below_2_percent"] is False
    assert gates["pitch_heat_below_2_percent"] is False


def test_modal_tail_claim_is_route_aware() -> None:
    payload = _payload()
    assert payload["analytic_full_state_oracle"]["diagnostic_status"] == (
        "retained_full_state_relative_l2"
    )
    assert all(
        rung["diagnostic_status"] == "unavailable_on_zero_padded_truncated_state"
        for rung in payload["rungs"]
    )
