from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/paper_benchmarks/audit_ambipolar_joint_speed_zeta_tail.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_ambipolar_joint_speed_zeta_tail", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / "validation/ambipolar_joint_speed_zeta_tail_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_joint_speed_zeta_tail_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["phase_space_converged"] is False


def test_speed_and_zeta_observable_gates_fail() -> None:
    comparisons = {row["name"]: row for row in _payload()["comparisons"]}
    assert all(
        row["particle_flux_scaled_movement"] > 0.02
        and row["heat_flux_scaled_movement"] > 0.02
        for row in comparisons.values()
    )


def test_tail_bound_is_not_promoted_as_a_convergence_oracle() -> None:
    payload = _payload()
    tail_maxima = [
        rung["maximum_selected_tail_bound"] for rung in payload["rungs"]
    ]
    assert max(tail_maxima) > 0.09
    assert tail_maxima[3] < tail_maxima[2]
    assert payload["outcome"]["phase_space_converged"] is False
    assert payload["outcome"]["whole_profile_escalation_admitted"] is False
