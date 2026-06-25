from __future__ import annotations

import json
from pathlib import Path

from sfincs_jax.validation.qi_device import (
    check_qi_device_artifact_file,
    qi_device_artifact_errors,
)


ARTIFACT = (
    Path("docs/_static/figures/optimization")
    / "qi_nfp2_electron_root_res13_gpu_operator_reuse_coupled_failclosed.json"
)


def test_checked_gpu_operator_reuse_artifact_satisfies_qi_device_policy() -> None:
    check = check_qi_device_artifact_file(ARTIFACT)

    assert check.relevant is True
    assert check.passed is True
    assert check.errors == ()


def test_qi_device_policy_rejects_overpromoted_nonconverged_gpu_artifact() -> None:
    payload = {
        "artifact_kind": "qi_device_gpu_probe",
        "status": "fail_closed_nonconverged_device_qi",
        "claim_boundary": "fail-closed route evidence only",
        "backend": "gpu",
        "route": {
            "operator_reuse_enabled": True,
            "host_fallback": True,
            "local_xblock_preconditioner_built": True,
        },
        "result": {
            "converged": False,
            "output_refused": False,
        },
        "promotion_gates": {
            "residual_convergence": "fail",
            "production_gpu_qi_performance": "pass",
        },
    }

    errors = qi_device_artifact_errors(payload, path="bad.json")

    assert any("host_fallback=false" in error for error in errors)
    assert any("local x-block factors skipped" in error for error in errors)
    assert any("output_refused=true" in error for error in errors)
    assert any("production GPU QI performance cannot pass" in error for error in errors)


def test_qi_device_policy_accepts_legacy_fail_closed_gpu_artifact_without_backend() -> None:
    payload = {
        "artifact_kind": "qi_seed_execution_summary",
        "probe_preset": "post-residual-equation-device-qi",
        "evidence_note": "legacy fail-closed GPU blocker artifact",
        "execution_summary": {
            "accepted_converged": 0,
            "outputs_written": 0,
        },
        "gates": {"passed": False},
    }

    assert qi_device_artifact_errors(payload, path="qi_seed_gpu0_device_qi.json") == []


def test_qi_device_policy_still_rejects_claimed_gpu_artifact_without_backend() -> None:
    payload = {
        "artifact_kind": "qi_seed_execution_summary",
        "probe_preset": "post-residual-equation-device-qi",
        "evidence_note": "claimed GPU output without backend provenance",
        "execution_summary": {
            "accepted_converged": 1,
            "outputs_written": 1,
        },
        "gates": {"passed": True},
    }

    errors = qi_device_artifact_errors(payload, path="qi_seed_gpu0_device_qi.json")

    assert any("GPU QI artifact must record backend='gpu'" in error for error in errors)


def test_qi_device_artifact_file_classifies_unrelated_json_as_irrelevant(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.json"
    path.write_text(json.dumps({"artifact_kind": "pas_benchmark"}), encoding="utf-8")

    check = check_qi_device_artifact_file(path)

    assert check.relevant is False
    assert check.passed is True
