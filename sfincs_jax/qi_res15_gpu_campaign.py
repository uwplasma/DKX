"""Claim-safe ingestion of the QI ``15x`` GPU promotion campaign.

The bounded GPU campaign writes an execution summary first and a promotion JSON
only if the scan itself completes.  This module performs the second gate before
any result is folded into the checked QI convergence ladder: the campaign must
have passed, the GPU promotion must have passed, residuals must be clean, and
the selected electron root must agree with the checked CPU/Fortran ``15x``
reference artifact.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_GPU_CPU_ROOT_ATOL = 1.0e-6
DEFAULT_GPU_FORTRAN_ROOT_ATOL = 3.0e-6


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON object from ``path``."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must decode to a JSON object")
    return payload


def evaluate_qi_res15_gpu_campaign(
    campaign: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    campaign_dir: str | Path | None = None,
    gpu_cpu_root_atol: float = DEFAULT_GPU_CPU_ROOT_ATOL,
    gpu_fortran_root_atol: float = DEFAULT_GPU_FORTRAN_ROOT_ATOL,
) -> dict[str, Any]:
    """Evaluate whether a bounded QI ``15x`` GPU campaign can be promoted.

    A passing result is still a fixed-resolution bounded artifact, not a
    production-resolution QI claim.  The production floor remains the
    ``25 x 51 x 100 x 4`` CPU/GPU ladder.
    """

    root = Path(campaign_dir).resolve() if campaign_dir is not None else Path.cwd()
    failures: list[str] = []
    lane = _find_gpu_lane(campaign)
    if lane is None:
        failures.append("campaign has no gpu lane result")
        lane = {}

    if campaign.get("campaign_status") != "pass":
        failures.append(f"campaign_status is {campaign.get('campaign_status')!r}, expected 'pass'")
    if lane.get("status") != "pass":
        failures.append(f"gpu lane status is {lane.get('status')!r}, expected 'pass'")

    promotion_payload: dict[str, Any] | None = None
    promotion_path: Path | None = _resolve_promotion_json(lane, campaign_dir=root)
    if promotion_path is None:
        failures.append("gpu promotion JSON could not be resolved from campaign summary")
    else:
        promotion_payload = load_json_object(promotion_path)

    reference_roots = _reference_roots(reference)
    gpu_root = _selected_root(promotion_payload) if promotion_payload is not None else {}
    gpu_er = _finite_float(gpu_root.get("er")) if gpu_root else None
    if gpu_root.get("root_type") != "electron":
        failures.append("gpu selected root is not an electron root")
    if gpu_er is None:
        failures.append("gpu selected root is missing or non-finite")

    if promotion_payload is not None:
        if promotion_payload.get("gate_status") != "pass":
            failures.append(f"gpu promotion gate_status is {promotion_payload.get('gate_status')!r}, expected 'pass'")
        promotion_failures = promotion_payload.get("failures", [])
        if promotion_failures:
            failures.append(f"gpu promotion records failures: {promotion_failures}")

    residual_summary = _residual_summary(promotion_payload)
    if residual_summary["failed_count"] > 0:
        failures.append(f"gpu promotion has {residual_summary['failed_count']} failed residual gates")

    root_differences: dict[str, float | None] = {
        "gpu_minus_cpu_abs": None,
        "gpu_minus_fortran_v3_abs": None,
    }
    if gpu_er is not None:
        cpu_er = reference_roots.get("cpu")
        fortran_er = reference_roots.get("fortran_v3")
        if cpu_er is None:
            failures.append("reference artifact is missing cpu selected root")
        else:
            root_differences["gpu_minus_cpu_abs"] = abs(float(gpu_er) - float(cpu_er))
            if root_differences["gpu_minus_cpu_abs"] > float(gpu_cpu_root_atol):
                failures.append(
                    "gpu/cpu selected-root difference "
                    f"{root_differences['gpu_minus_cpu_abs']:.6g} exceeds {float(gpu_cpu_root_atol):.6g}"
                )
        if fortran_er is None:
            failures.append("reference artifact is missing fortran_v3 selected root")
        else:
            root_differences["gpu_minus_fortran_v3_abs"] = abs(float(gpu_er) - float(fortran_er))
            if root_differences["gpu_minus_fortran_v3_abs"] > float(gpu_fortran_root_atol):
                failures.append(
                    "gpu/fortran_v3 selected-root difference "
                    f"{root_differences['gpu_minus_fortran_v3_abs']:.6g} exceeds "
                    f"{float(gpu_fortran_root_atol):.6g}"
                )

    status = "pass_bounded_gpu_res15" if not failures else "fail_closed"
    return {
        "workflow": "sfincs_jax_qi_nfp2_res15_gpu_campaign_evidence",
        "status": status,
        "claim_boundary": (
            "This artifact can close only the fixed-resolution QI nfp=2 15x GPU "
            "promotion rung. It is not a production-resolution QI convergence "
            "or performance claim until the 25 x 51 x 100 x 4 CPU/GPU ladder "
            "passes."
        ),
        "source_campaign": str(root / "promotion_evidence_campaign.json"),
        "source_promotion": None if promotion_path is None else str(promotion_path),
        "source_reference": str(reference.get("artifact_kind", "qi_nfp2_res15_cpu_fortran_reference")),
        "resolution": reference.get("resolution", {}),
        "campaign_status": campaign.get("campaign_status"),
        "gpu_lane_status": lane.get("status"),
        "gpu_selected_root": gpu_root,
        "reference_roots": reference_roots,
        "root_differences": root_differences,
        "tolerances": {
            "gpu_cpu_root_atol": float(gpu_cpu_root_atol),
            "gpu_fortran_root_atol": float(gpu_fortran_root_atol),
        },
        "residual_summary": residual_summary,
        "gates": {
            "campaign_passed": "pass" if campaign.get("campaign_status") == "pass" else "fail",
            "gpu_promotion_gate": (
                "pass" if promotion_payload is not None and promotion_payload.get("gate_status") == "pass" else "fail"
            ),
            "gpu_residuals": "pass" if residual_summary["failed_count"] == 0 else "fail",
            "gpu_cpu_root_agreement": (
                "pass"
                if root_differences["gpu_minus_cpu_abs"] is not None
                and root_differences["gpu_minus_cpu_abs"] <= float(gpu_cpu_root_atol)
                else "fail"
            ),
            "gpu_fortran_v3_root_agreement": (
                "pass"
                if root_differences["gpu_minus_fortran_v3_abs"] is not None
                and root_differences["gpu_minus_fortran_v3_abs"] <= float(gpu_fortran_root_atol)
                else "fail"
            ),
            "production_resolution_qi_convergence": "open",
        },
        "failures": failures,
    }


def evaluate_qi_res15_gpu_campaign_files(
    *,
    campaign_path: str | Path,
    reference_path: str | Path,
    gpu_cpu_root_atol: float = DEFAULT_GPU_CPU_ROOT_ATOL,
    gpu_fortran_root_atol: float = DEFAULT_GPU_FORTRAN_ROOT_ATOL,
) -> dict[str, Any]:
    """Load campaign/reference JSON files and evaluate the QI ``15x`` GPU gate."""

    campaign_file = Path(campaign_path).resolve()
    reference_file = Path(reference_path).resolve()
    return evaluate_qi_res15_gpu_campaign(
        load_json_object(campaign_file),
        load_json_object(reference_file),
        campaign_dir=campaign_file.parent,
        gpu_cpu_root_atol=gpu_cpu_root_atol,
        gpu_fortran_root_atol=gpu_fortran_root_atol,
    ) | {
        "source_campaign": str(campaign_file),
        "source_reference": str(reference_file),
    }


def _find_gpu_lane(campaign: Mapping[str, Any]) -> Mapping[str, Any] | None:
    lanes = campaign.get("lane_results", [])
    if not isinstance(lanes, list):
        return None
    for lane in lanes:
        if not isinstance(lane, Mapping):
            continue
        label = str(lane.get("label", "")).lower()
        backend = str(lane.get("backend", "")).lower()
        if label == "gpu" or ("gpu" in label and backend == "sfincs_jax"):
            return lane
    return None


def _resolve_promotion_json(lane: Mapping[str, Any], *, campaign_dir: Path) -> Path | None:
    raw = lane.get("promotion_json")
    candidates: list[Path] = []
    if raw:
        path = Path(str(raw))
        candidates.append(path)
        candidates.append(campaign_dir / path.name)
    candidates.extend(sorted((campaign_dir / "gpu_promotion").glob("*.json")))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _reference_roots(reference: Mapping[str, Any]) -> dict[str, float | None]:
    roots = reference.get("fixed_resolution_roots", {})
    if not isinstance(roots, Mapping):
        return {"cpu": None, "fortran_v3": None}
    out: dict[str, float | None] = {}
    for name in ("cpu", "fortran_v3"):
        raw = roots.get(name, {})
        out[name] = _finite_float(raw.get("er")) if isinstance(raw, Mapping) else None
    return out


def _selected_root(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    if payload is None:
        return {}
    root = payload.get("selected_root", {})
    return dict(root) if isinstance(root, Mapping) else {}


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _residual_summary(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    runs = payload.get("runs", []) if payload is not None else []
    if not isinstance(runs, list):
        return {"run_count": 0, "failed_count": 1, "max_residual_ratio": None}
    failed = 0
    ratios: list[float] = []
    for run in runs:
        if not isinstance(run, Mapping):
            failed += 1
            continue
        gate = run.get("residual_gate", {})
        status = gate.get("status") if isinstance(gate, Mapping) else None
        residual = _finite_float(run.get("residual_norm"))
        target = _finite_float(run.get("residual_target"))
        if residual is not None and target is not None and target > 0:
            ratios.append(residual / target)
        if status == "pass":
            continue
        if residual is None or target is None or residual > target:
            failed += 1
    return {
        "run_count": len(runs),
        "failed_count": failed,
        "max_residual_ratio": max(ratios) if ratios else None,
    }
