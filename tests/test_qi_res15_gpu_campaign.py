from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import sfincs_jax.validation.qi_device as qi_device
from sfincs_jax.validation.qi_device import (
    evaluate_qi_res15_gpu_campaign,
    evaluate_qi_res15_gpu_campaign_files,
    load_json_object,
)


_REPO = Path(__file__).resolve().parents[1]


def _reference() -> dict[str, object]:
    return {
        "artifact_kind": "synthetic_res15_cpu_fortran_reference",
        "workflow": "sfincs_jax_qi_nfp2_kinetic_res15_reference_tolerance_comparison",
        "resolution": {"Ntheta": 15, "Nzeta": 15, "Nxi": 17, "NL": 4, "Nx": 4},
        "fixed_resolution_roots": {
            "cpu": {"er": 2.213238923947695, "root_type": "electron"},
            "fortran_v3": {"er": 2.213236890600478, "root_type": "electron"},
        },
    }


def _promotion(*, er: float = 2.213238923947695, residual_status: str = "pass") -> dict[str, object]:
    return {
        "workflow": "sfincs_jax_optimization_high_fidelity_promotion",
        "gate_status": "pass",
        "failures": [],
        "selected_root": {
            "er": er,
            "root_type": "electron",
            "bracket": [2.0, 3.0],
            "slope": 1.0e-8,
        },
        "runs": [
            {
                "er": -0.3,
                "residual_norm": 1.0e-18 if residual_status == "pass" else 1.0e-3,
                "residual_target": 1.0e-11,
                "residual_gate": {"status": residual_status},
            }
        ],
    }


def _write_json(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_qi_res15_gpu_campaign_ingestion_accepts_clean_gpu_promotion(tmp_path: Path) -> None:
    promotion_path = _write_json(tmp_path / "gpu_promotion" / "qi_gpu.json", _promotion())
    campaign_path = _write_json(
        tmp_path / "promotion_evidence_campaign.json",
        {
            "workflow": "sfincs_jax_optimization_promotion_evidence_plan",
            "campaign_status": "pass",
            "lane_results": [
                {
                    "label": "gpu",
                    "backend": "sfincs_jax",
                    "status": "pass",
                    "promotion_json": str(promotion_path),
                }
            ],
        },
    )
    reference_path = _write_json(tmp_path / "reference.json", _reference())

    artifact = evaluate_qi_res15_gpu_campaign_files(
        campaign_path=campaign_path,
        reference_path=reference_path,
    )

    assert artifact["status"] == "pass_bounded_gpu_res15"
    assert artifact["failures"] == []
    assert artifact["gates"]["gpu_cpu_root_agreement"] == "pass"
    assert artifact["gates"]["production_resolution_qi_convergence"] == "open"
    assert artifact["residual_summary"]["failed_count"] == 0


def test_qi_res15_gpu_campaign_mapping_api_and_json_loader(tmp_path: Path) -> None:
    promotion_path = _write_json(tmp_path / "gpu_promotion" / "qi_gpu.json", _promotion())
    campaign = {
        "workflow": "sfincs_jax_optimization_promotion_evidence_plan",
        "campaign_status": "pass",
        "lane_results": [
            {
                "label": "gpu",
                "backend": "sfincs_jax",
                "status": "pass",
                "promotion_json": str(promotion_path),
            }
        ],
    }
    reference = _reference()

    artifact = evaluate_qi_res15_gpu_campaign(campaign, reference, campaign_dir=tmp_path)

    assert artifact["status"] == "pass_bounded_gpu_res15"
    assert load_json_object(promotion_path)["selected_root"]["root_type"] == "electron"
    scalar_path = tmp_path / "bad.json"
    scalar_path.write_text("[1, 2, 3]", encoding="utf-8")
    try:
        load_json_object(scalar_path)
    except TypeError as exc:
        assert "JSON object" in str(exc)
    else:  # pragma: no cover - defensive assertion.
        raise AssertionError("load_json_object accepted a non-object JSON payload")


def test_qi_res15_gpu_campaign_ingestion_fails_closed_on_root_or_residual_mismatch(
    tmp_path: Path,
) -> None:
    promotion_path = _write_json(
        tmp_path / "gpu_promotion" / "qi_gpu.json",
        _promotion(er=2.3, residual_status="fail"),
    )
    campaign_path = _write_json(
        tmp_path / "promotion_evidence_campaign.json",
        {
            "campaign_status": "pass",
            "lane_results": [
                {
                    "label": "gpu",
                    "backend": "sfincs_jax",
                    "status": "pass",
                    "promotion_json": str(promotion_path),
                }
            ],
        },
    )
    reference_path = _write_json(tmp_path / "reference.json", _reference())

    artifact = evaluate_qi_res15_gpu_campaign_files(
        campaign_path=campaign_path,
        reference_path=reference_path,
    )

    assert artifact["status"] == "fail_closed"
    assert artifact["gates"]["gpu_residuals"] == "fail"
    assert artifact["gates"]["gpu_cpu_root_agreement"] == "fail"
    assert any("selected-root difference" in failure for failure in artifact["failures"])
    assert any("failed residual gates" in failure for failure in artifact["failures"])


def test_public_qi_res15_gpu_campaign_ingestion_cli_writes_artifact(tmp_path: Path) -> None:
    promotion_path = _write_json(tmp_path / "gpu_promotion" / "qi_gpu.json", _promotion())
    campaign_path = _write_json(
        tmp_path / "promotion_evidence_campaign.json",
        {
            "campaign_status": "pass",
            "lane_results": [
                {
                    "label": "gpu",
                    "backend": "sfincs_jax",
                    "status": "pass",
                    "promotion_json": str(promotion_path),
                }
            ],
        },
    )
    reference_path = _write_json(tmp_path / "reference.json", _reference())
    script = _REPO / "examples" / "optimization" / "ingest_qi_res15_gpu_campaign.py"
    out_dir = tmp_path / "out"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "--campaign",
            str(campaign_path),
            "--reference",
            str(reference_path),
            "--out-dir",
            str(out_dir),
            "--stem",
            "checked",
        ],
        cwd=_REPO,
        check=True,
        timeout=20,
    )

    artifact = json.loads((out_dir / "checked.json").read_text(encoding="utf-8"))
    assert artifact["workflow"] == "sfincs_jax_qi_nfp2_res15_gpu_campaign_evidence"
    assert artifact["status"] == "pass_bounded_gpu_res15"


def test_qi_res15_gpu_campaign_fails_closed_when_lane_or_promotion_is_missing(tmp_path: Path) -> None:
    artifact = evaluate_qi_res15_gpu_campaign(
        {"campaign_status": "fail", "lane_results": "not a list"},
        {"fixed_resolution_roots": "not a mapping"},
        campaign_dir=tmp_path,
    )

    assert artifact["status"] == "fail_closed"
    assert artifact["gpu_lane_status"] is None
    assert artifact["source_promotion"] is None
    assert artifact["gates"]["campaign_passed"] == "fail"
    assert artifact["gates"]["gpu_promotion_gate"] == "fail"
    assert artifact["gates"]["gpu_cpu_root_agreement"] == "fail"
    assert any("no gpu lane" in failure for failure in artifact["failures"])
    assert any("campaign_status" in failure for failure in artifact["failures"])
    assert any("promotion JSON" in failure for failure in artifact["failures"])
    assert any("selected root is not an electron root" in failure for failure in artifact["failures"])
    assert any("selected root is missing" in failure for failure in artifact["failures"])


def test_qi_res15_gpu_campaign_fails_closed_for_bad_promotion_and_reference(tmp_path: Path) -> None:
    promotion_path = _write_json(
        tmp_path / "gpu_promotion" / "qi_gpu.json",
        {
            "gate_status": "fail",
            "failures": ["manual residual gate failure"],
            "selected_root": {"er": "not finite", "root_type": "ion"},
            "runs": "not a residual run list",
        },
    )
    campaign = {
        "campaign_status": "pass",
        "lane_results": [
            {"label": "warm-gpu", "backend": "sfincs_jax", "status": "fail", "promotion_json": promotion_path.name},
            "not a mapping",
        ],
    }
    artifact = evaluate_qi_res15_gpu_campaign(campaign, {}, campaign_dir=tmp_path)

    assert artifact["status"] == "fail_closed"
    assert artifact["gpu_lane_status"] == "fail"
    assert artifact["residual_summary"] == {"run_count": 0, "failed_count": 1, "max_residual_ratio": None}
    assert any("gpu lane status" in failure for failure in artifact["failures"])
    assert any("selected root is not an electron root" in failure for failure in artifact["failures"])
    assert any("selected root is missing" in failure for failure in artifact["failures"])
    assert any("gate_status" in failure for failure in artifact["failures"])
    assert any("promotion records failures" in failure for failure in artifact["failures"])
    assert all("missing cpu selected root" not in failure for failure in artifact["failures"])


def test_qi_res15_gpu_campaign_fails_closed_for_missing_reference_roots(tmp_path: Path) -> None:
    promotion_path = _write_json(tmp_path / "gpu_promotion" / "qi_gpu.json", _promotion())
    campaign = {
        "campaign_status": "pass",
        "lane_results": [
            {"label": "gpu", "backend": "sfincs_jax", "status": "pass", "promotion_json": promotion_path.name}
        ],
    }

    artifact = evaluate_qi_res15_gpu_campaign(campaign, {}, campaign_dir=tmp_path)

    assert artifact["status"] == "fail_closed"
    assert any("missing cpu selected root" in failure for failure in artifact["failures"])
    assert any("missing fortran_v3 selected root" in failure for failure in artifact["failures"])


def test_private_qi_res15_campaign_helpers_cover_absent_and_malformed_paths(tmp_path: Path) -> None:
    assert qi_device._find_gpu_lane({"lane_results": [{"label": "cpu", "backend": "sfincs_jax"}, "bad"]}) is None
    assert qi_device._resolve_promotion_json({"promotion_json": "missing.json"}, campaign_dir=tmp_path) is None
    assert qi_device._reference_roots({"fixed_resolution_roots": []}) == {"cpu": None, "fortran_v3": None}
    assert qi_device._selected_root(None) == {}
    assert qi_device._selected_root({"selected_root": []}) == {}
    assert qi_device._finite_float("not finite") is None
    assert qi_device._residual_summary({"runs": [{"residual_gate": {"status": "fail"}}]}) == {
        "run_count": 1,
        "failed_count": 1,
        "max_residual_ratio": None,
    }
    assert qi_device._residual_summary({"runs": ["not a mapping"]}) == {
        "run_count": 1,
        "failed_count": 1,
        "max_residual_ratio": None,
    }

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(TypeError, match="JSON object"):
        load_json_object(scalar_path)
