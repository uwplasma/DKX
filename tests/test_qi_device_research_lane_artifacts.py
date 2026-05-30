from __future__ import annotations

import json
from pathlib import Path

import pytest


STATIC_DIR = Path("docs/_static")


def _load_static_json(name: str) -> dict[str, object]:
    return json.loads((STATIC_DIR / name).read_text(encoding="utf-8"))


def _first_seed(payload: dict[str, object]) -> dict[str, object]:
    seeds = payload["seeds"]
    assert isinstance(seeds, list)
    assert len(seeds) == 1
    seed = seeds[0]
    assert isinstance(seed, dict)
    return seed


def test_true_device_qi_manifest_stays_closed_deferred_until_production_gate_passes() -> None:
    payload = _load_static_json("qi_seed_robustness_evidence_manifest.json")

    assert payload["artifact_kind"] == "qi_seed_production_gate_manifest"
    assert payload["lane"] == "qi_seed_robustness"

    release_claims = payload["release_claims"]
    assert isinstance(release_claims, dict)
    true_device_qi = release_claims["true_device_qi"]
    assert isinstance(true_device_qi, dict)
    assert true_device_qi["claim_status"] == "closed_deferred"
    assert true_device_qi["blocks_current_release"] is False
    assert "recycled augmented-Krylov" in true_device_qi["closed_or_deferred_reason"]

    production_target = payload["production_target"]
    assert isinstance(production_target, dict)
    assert production_target["resolution"] == {"NTHETA": 25, "NZETA": 51, "NXI": 100, "NX": 4}
    assert production_target["seed_count"] == 5

    blockers = payload["open_blockers"]
    assert isinstance(blockers, list)
    assert any("scale-0.60 GPU hard-seed" in str(blocker) for blocker in blockers)
    assert any("non-autodiff host fallback" in str(blocker) for blocker in blockers)


def test_best_checked_one_gpu_device_qi_artifact_is_still_fail_closed() -> None:
    payload = _load_static_json("qi_seed_robustness_scale060_recycled_augmented_deep_device_qi_gpu0_2026_05_20.json")

    assert payload["gates"]["passed"] is False
    assert payload["execution_summary"]["outputs_written"] == 0
    assert payload["execution_summary"]["solver_traces_written"] == 0
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502

    seed = _first_seed(payload)
    assert seed["output_exists"] is False
    assert seed["solver_trace_exists"] is False
    assert seed["xblock_qi_device_preconditioner_used"] is True
    assert "observed_augmented_krylov" in seed["evidence_tags"]
    assert "observed_installed_krylov" in seed["evidence_tags"]
    assert seed["last_progress_residual_norm"] == pytest.approx(7.336295e-6)
    assert seed["last_progress_residual_target"] == pytest.approx(3.021487e-11)
    assert seed["last_progress_residual_ratio"] > 2.0e5


@pytest.mark.parametrize(
    "artifact_name, expected_residual",
    [
        ("qi_seed_robustness_scale060_post_residual_equation_device_qi_cpu_2026_05_22.json", 2.105918e-5),
        ("qi_seed_robustness_scale060_post_residual_equation_device_qi_gpu1_2026_05_22.json", 2.142936e-5),
    ],
)
def test_post_residual_equation_artifacts_record_research_evidence_not_promotion(
    artifact_name: str,
    expected_residual: float,
) -> None:
    payload = _load_static_json(artifact_name)

    assert payload["gates"]["passed"] is False
    assert payload["execution_summary"]["outputs_written"] == 0
    assert payload["execution_summary"]["solver_traces_written"] == 0
    assert payload["execution_summary"]["accepted_converged"] == 0
    assert payload["active_size"] == 81377
    assert payload["total_size_estimate"] == 139502

    seed = _first_seed(payload)
    assert seed["output_exists"] is False
    assert seed["solver_trace_exists"] is False
    assert seed["last_progress_residual_norm"] == pytest.approx(expected_residual)
    assert seed["last_progress_residual_target"] == pytest.approx(3.021487e-11)
    assert seed["last_progress_residual_ratio"] > 6.0e5
    assert "observed_post_residual_equation" in seed["evidence_tags"]
    assert "requested_post_residual_equation" in seed["evidence_tags"]
