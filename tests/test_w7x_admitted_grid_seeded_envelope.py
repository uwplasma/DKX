from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT = ROOT / "validation/w7x_admitted_grid_seeded_envelope_v1.json"
SCRIPT = ROOT / "tools/paper_benchmarks/audit_w7x_admitted_grid_seeded_envelope.py"
SPEC = importlib.util.spec_from_file_location("audit_w7x_admitted_grid_seeded", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_admitted_grid_seeded_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["topology"] == [1, 1]
    assert report["roots_kV_m"] == [12.681640625, 11.533203125]
    assert report["global_all_root_claim"] is False


def test_final_replay_has_strict_endpoint_signs_and_exact_parity() -> None:
    final = _payload()["final_replay"]
    assert all(left * right < 0.0 for left, right in final["endpoint_currents_A_m2"])
    assert final["cold_warm_arrays_exact_except_solve_time_s"] is True
    assert final["maximum_primal_residual"] < 1.0e-12


def test_failed_lower_surface_one_envelopes_remain_explicit() -> None:
    payload = _payload()
    assert payload["envelope_run"]["statuses"][1] == "seeded_bracket_failed"
    assert all(
        value < 0.0
        for value in payload["envelope_run"][
            "surface_1_endpoint_currents_A_m2"
        ].values()
    )
    assert payload["case"]["unsampled_crossings_excluded"] is False
