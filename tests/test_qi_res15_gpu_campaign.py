from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from sfincs_jax.workflows.qi_res15_gpu_campaign import evaluate_qi_res15_gpu_campaign_files


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
