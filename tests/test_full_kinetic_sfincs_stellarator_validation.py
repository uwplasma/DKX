from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ARTIFACT = _ROOT / "validation" / "full_kinetic_sfincs_stellarator_v1.json"


def _payload() -> dict[str, object]:
    payload = json.loads(_ARTIFACT.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_stellarator_artifact_pins_full_kinetic_equations_and_geometry() -> None:
    payload = _payload()
    equations = payload["equations"]
    source = payload["source"]
    assert isinstance(equations, dict)
    assert isinstance(source, dict)
    assert payload["schema"] == "dkx.full_kinetic_sfincs_stellarator.v1"
    assert payload["claim_scope"] == (
        "matched_zero_field_stellarator_full_fokker_planck_surface_profile"
    )
    assert equations["geometry"] == "W7-X SC1 Boozer surface"
    assert equations["geometry_scheme"] == 11
    assert equations["collision_operator"] == 0
    assert equations["electric_field"] == 0.0
    assert equations["constraint_input"] == -1
    assert equations["constraint_resolved"] == 1
    assert equations["include_x_dot"] is True
    assert equations["include_electric_field_term_in_xi_dot"] is True
    assert equations["use_dkes_exb_drift"] is False
    assert equations["solver_tolerance"] == pytest.approx(1e-12)
    assert source["sfincs_equilibrium"] == "equilibria/w7x-sc1.bc"
    assert source["sfincs_equilibrium_sha256"] == (
        "1d096d5ad8104750fcc787ef226b2fbc8a82bcd3774fbab41a2f87dcb04ce831"
    )


def test_stellarator_artifact_has_converged_resolution_and_residual_gates() -> None:
    payload = _payload()
    rungs = payload["rungs"]
    assert isinstance(rungs, list)
    assert [rung["id"] for rung in rungs] == ["high", "ultra"]
    assert [rung["resolution"]["sfincs_matrix_size"] for rung in rungs] == [
        54407,
        98126,
    ]
    assert [rung["resolution"]["dkx_matrix_size"] for rung in rungs] == [
        87887,
        155994,
    ]

    acceptance = payload["acceptance"]
    assert isinstance(acceptance, dict)
    assert acceptance["all_gates_pass"] is True
    assert acceptance["measured_max_cross_code_scaled_error"] == pytest.approx(
        1.3644269489024203e-8
    )
    assert acceptance["measured_max_high_to_ultra_scaled_movement"] == pytest.approx(
        0.004436383524571227
    )
    assert acceptance["measured_max_near_zero_absolute_value"] == pytest.approx(
        1.8892805190711152e-21
    )
    assert acceptance["measured_max_completed_true_residual"] == pytest.approx(
        1.8182906e-12
    )


def test_shared_full_kinetic_audit_accepts_stellarator_artifact() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py",
            "--artifact",
            str(_ARTIFACT),
        ],
        cwd=_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert report["pass"] is True
    assert report["errors"] == []
    assert report["measured"]["max_cross_code_scaled_error"] < 2e-8


def test_external_audit_rejects_wrong_stellarator_geometry(tmp_path: Path) -> None:
    payload = _payload()
    source = payload["source"]
    source["sfincs_equilibrium_sha256"] = hashlib.sha256(b"expected").hexdigest()
    source["sfincs_equilibrium_bytes"] = len(b"expected")
    artifact = tmp_path / "stellarator.json"
    artifact.write_text(json.dumps(payload), encoding="utf-8")

    results = tmp_path / "results"
    for rung in ("high", "ultra"):
        for code in ("sfincs", "dkx"):
            directory = results / rung / code
            directory.mkdir(parents=True)
            (directory / "w7x-sc1.bc").write_bytes(b"wrong")

    completed = subprocess.run(
        [
            sys.executable,
            "tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py",
            "--artifact",
            str(artifact),
            "--results-root",
            str(results),
        ],
        cwd=_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)
    assert completed.returncode == 1
    assert "high: external sfincs equilibrium checksum mismatch" in report["errors"]
    assert "ultra: external dkx equilibrium checksum mismatch" in report["errors"]


def test_stellarator_claim_keeps_later_gates_explicit() -> None:
    exclusions = set(_payload()["exclusions"])
    assert {
        "not_multispecies_validation",
        "not_finite_er_validation",
        "not_er_scan_validation",
        "not_ambipolar_profile_validation",
        "not_phi1_validation",
        "not_experimental_validation",
        "not_cross_code_performance_validation",
        "not_second_stellarator_family_full_fokker_planck_validation",
    } <= exclusions
