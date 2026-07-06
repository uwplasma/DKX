from __future__ import annotations

import json
from pathlib import Path

import pytest

import sfincs_jax.validation.qi_device as qi_device
from sfincs_jax.validation.qi_device import (
    check_qi_device_artifact_files,
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


def test_qi_device_policy_requires_claim_metadata_and_fail_closed_consistency() -> None:
    payload = {
        "route": {"operator_reuse_enabled": True},
        "status": "fail_closed_nonconverged_device_qi",
        "backend": "gpu",
        "result": {"converged": True, "output_refused": True},
        "promotion_gates": {"residual_convergence": "pass"},
        "execution_summary": {"accepted_converged": 0, "outputs_written": 1},
    }

    errors = qi_device_artifact_errors(payload, path="operator_reuse_qi.json")

    assert any("missing artifact_kind" in error for error in errors)
    assert any("evidence_note or claim_boundary" in error for error in errors)
    assert any("converged=false" in error for error in errors)
    assert any("residual_convergence='fail'" in error for error in errors)
    assert any("must not count written outputs" in error for error in errors)


def test_qi_device_policy_ignores_unrelated_payloads() -> None:
    assert qi_device_artifact_errors({"artifact_kind": "pas_benchmark"}, path="pas.json") == []


def test_qi_device_artifact_file_reports_io_json_and_scalar_inputs(tmp_path: Path) -> None:
    invalid_json = tmp_path / "device_qi_invalid.json"
    invalid_json.write_text("{not json", encoding="utf-8")
    invalid_check = check_qi_device_artifact_file(invalid_json)
    assert invalid_check.relevant is True
    assert invalid_check.passed is False
    assert "invalid JSON" in invalid_check.errors[0]

    scalar_json = tmp_path / "device_qi_scalar.json"
    scalar_json.write_text("[1, 2, 3]", encoding="utf-8")
    scalar_check = check_qi_device_artifact_file(scalar_json)
    assert scalar_check.relevant is False
    assert scalar_check.passed is True

    missing_check, scalar_check_again = check_qi_device_artifact_files([tmp_path / "missing.json", scalar_json])
    assert missing_check.relevant is True
    assert missing_check.passed is False
    assert "could not read" in missing_check.errors[0]
    assert scalar_check_again.relevant is False


def test_private_qi_device_artifact_helpers_cover_text_and_legacy_paths(tmp_path: Path) -> None:
    assert qi_device._looks_like_qi_device_artifact("operator_reuse qa qi", {}) is True
    assert qi_device._looks_like_qi_device_artifact("", {"route": {"operator_reuse_enabled": True}}) is True
    assert qi_device._looks_like_qi_device_artifact("unrelated", {"route": {}}) is False

    assert qi_device._legacy_fail_closed_gpu_artifact("cpu device_qi", {}) is False
    assert qi_device._legacy_fail_closed_gpu_artifact(
        "gpu device_qi",
        {"result": {"output_refused": True, "converged": False}},
    ) is True
    assert qi_device._legacy_fail_closed_gpu_artifact("gpu device_qi", {"result": {"converged": True}}) is False
    assert qi_device._legacy_fail_closed_gpu_artifact("gpu device_qi", {}) is False

    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        qi_device.load_json_object(scalar_json)
