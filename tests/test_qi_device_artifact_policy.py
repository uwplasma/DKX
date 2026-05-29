from __future__ import annotations

import json
from pathlib import Path

from sfincs_jax.qi_device_artifact_policy import (
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


def test_qi_device_artifact_file_classifies_unrelated_json_as_irrelevant(tmp_path: Path) -> None:
    path = tmp_path / "unrelated.json"
    path.write_text(json.dumps({"artifact_kind": "pas_benchmark"}), encoding="utf-8")

    check = check_qi_device_artifact_file(path)

    assert check.relevant is False
    assert check.passed is True
