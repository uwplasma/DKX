from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/paper_benchmarks/audit_ambipolar_speed_local_pitch.py"
SPEC = importlib.util.spec_from_file_location("audit_ambipolar_speed_local_pitch", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / "validation/ambipolar_speed_local_pitch_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_speed_local_pitch_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["phase_space_converged"] is False


def test_node3_localization_and_failed_ceiling_gates_are_retained() -> None:
    payload = _payload()
    shares = payload["comparisons"][0]["node3_absolute_delta_share"]
    assert min(shares["particle"] + shares["heat"]) > 0.96
    gates = payload["outcome"]["gates"]
    assert gates["node3_dominates_initial_delta"] is True
    assert gates["node3_33_to_36_particle_below_2_percent"] is False
    assert gates["node3_33_to_36_heat_below_2_percent"] is False
    assert gates["pitch36_to_44_particle_below_2_percent"] is False
    assert gates["pitch36_to_44_heat_below_2_percent"] is False


def test_speed_local_claim_boundary_is_explicit() -> None:
    payload = _payload()
    assert payload["claim_scope"] == "common_field_speed_local_pitch_diagnosis"
    assert "not_phase_space_convergence_validation" in payload["exclusions"]
    assert "not_full_profile_root_validation" in payload["exclusions"]
