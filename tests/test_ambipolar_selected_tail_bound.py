from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/paper_benchmarks/audit_ambipolar_selected_tail_bound.py"
SPEC = importlib.util.spec_from_file_location(
    "audit_ambipolar_selected_tail_bound", SCRIPT
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / "validation/ambipolar_selected_tail_bound_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_selected_tail_bound_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["phase_space_converged"] is False


def test_selected_tail_evidence_is_bounded_and_sparse() -> None:
    payload = _payload()
    case = payload["case"]
    bound = np.asarray(
        case["selected_tail_bound_by_surface_speed_species"], dtype=np.float64
    )
    assert bound.shape == (2, 8, 2)
    assert np.all((bound >= 0.0) & (bound <= 1.0))
    assert np.max(bound) == case["maximum_tail_bound"]
    assert case["finite_tail_values"] == bound.size
    assert case["diagnostic_replays"] == 2


def test_selected_tail_claim_does_not_promote_phase_space() -> None:
    payload = _payload()
    assert payload["case"]["diagnostic_status"] == (
        "retained_selected_tail_relative_l2_upper_bound"
    )
    assert payload["outcome"]["phase_space_converged"] is False
    assert payload["outcome"]["whole_profile_escalation_admitted"] is False
