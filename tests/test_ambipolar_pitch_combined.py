from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/paper_benchmarks/audit_ambipolar_pitch_combined.py"
SPEC = importlib.util.spec_from_file_location("audit_ambipolar_pitch_combined", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
ARTIFACT = ROOT / "validation/ambipolar_pitch_combined_v1.json"


def _payload() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_compact_combined_pitch_audit_passes() -> None:
    report = MODULE.audit(ARTIFACT)
    assert report["pass"] is True
    assert report["status"] == "refinement_exhausted"
    assert report["phase_space_converged"] is False


def test_combined_ladder_retains_failed_observable_gates() -> None:
    payload = _payload()
    assert [rung["root_counts"] for rung in payload["rungs"]] == [[1, 3]] * 3
    assert payload["outcome"]["gates"] == {
        "topology_stable": True,
        "maximum_true_residual_below_1e-12": True,
        "all_process_footprints_below_24_gib": True,
        "combined_cold_warm_scientific_arrays_exact": True,
        "electric_field_movement_below_0_005_kV_m": False,
        "particle_flux_movement_below_0_02": False,
        "heat_flux_movement_below_0_02": False,
    }


def test_source_review_and_claim_boundary_are_explicit() -> None:
    payload = _payload()
    assert {row["code"] for row in payload["source"]["inspiration_review"]} == {
        "SFINCS",
        "YANCC",
        "MONKES",
        "STELLOPT/PENTA",
    }
    assert "not_phase_space_convergence_validation" in payload["exclusions"]
    assert "not_independent_cross_code_ambipolar_validation" in payload["exclusions"]
