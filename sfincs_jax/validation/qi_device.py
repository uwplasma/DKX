"""Offline policy checks for QI device/GPU evidence artifacts.

The benchmark artifact policy is intentionally PAS-focused, so QI device
artifacts need a separate CI-fast gate.  These checks do not prove a solve is
converged; they make overclaiming hard by requiring fail-closed metadata for
nonconverged device-QI runs and backend/provenance fields for GPU-route claims.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_GPU_CPU_ROOT_ATOL = 1.0e-6
DEFAULT_GPU_FORTRAN_ROOT_ATOL = 3.0e-6

__all__ = (
    "DEFAULT_GPU_CPU_ROOT_ATOL",
    "DEFAULT_GPU_FORTRAN_ROOT_ATOL",
    "QIDeviceArtifactCheck",
    "check_qi_device_artifact_file",
    "check_qi_device_artifact_files",
    "evaluate_qi_res15_gpu_campaign",
    "evaluate_qi_res15_gpu_campaign_files",
    "load_json_object",
    "qi_device_artifact_errors",
)


@dataclass(frozen=True)
class QIDeviceArtifactCheck:
    """Result for one checked JSON artifact."""

    path: Path
    relevant: bool
    errors: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Return whether this artifact satisfies the QI-device policy."""

        return not self.errors


def qi_device_artifact_errors(
    payload: Mapping[str, Any],
    *,
    path: str | Path = "<artifact>",
) -> list[str]:
    """Return policy errors for one QI device artifact payload."""

    source = str(path)
    errors: list[str] = []
    artifact_kind = str(payload.get("artifact_kind", ""))
    status = str(payload.get("status", ""))
    probe_preset = str(payload.get("probe_preset", ""))
    route = payload.get("route")
    route_map = route if isinstance(route, Mapping) else {}
    text = " ".join(
        (
            source,
            artifact_kind,
            status,
            probe_preset,
            str(payload.get("workflow", "")),
        )
    ).lower()
    if not _looks_like_qi_device_artifact(text, payload):
        return errors

    if not artifact_kind:
        errors.append(f"{source}: missing artifact_kind")
    if not (payload.get("evidence_note") or payload.get("claim_boundary")):
        errors.append(f"{source}: QI device artifact must include evidence_note or claim_boundary")

    backend = str(payload.get("backend", "")).lower()
    probe_env = payload.get("probe_env") if isinstance(payload.get("probe_env"), Mapping) else {}
    jax_platform = str(probe_env.get("JAX_PLATFORM_NAME", "")).lower()
    legacy_fail_closed_gpu = _legacy_fail_closed_gpu_artifact(text, payload)
    if "gpu" in text and backend != "gpu" and jax_platform != "gpu" and not legacy_fail_closed_gpu:
        errors.append(f"{source}: GPU QI artifact must record backend='gpu' or probe_env JAX_PLATFORM_NAME=gpu")

    if bool(route_map.get("operator_reuse_enabled")):
        if route_map.get("host_fallback") is not False:
            errors.append(f"{source}: operator-reuse QI route must record host_fallback=false")
        if route_map.get("local_xblock_preconditioner_built") is not False:
            errors.append(f"{source}: operator-reuse QI route must record local x-block factors skipped")

    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
    gates = payload.get("promotion_gates") if isinstance(payload.get("promotion_gates"), Mapping) else {}
    if "fail_closed" in status:
        if result.get("output_refused") is not True:
            errors.append(f"{source}: fail-closed QI artifact must record output_refused=true")
        if result.get("converged") is not False:
            errors.append(f"{source}: fail-closed QI artifact must record converged=false")
        if gates and gates.get("residual_convergence") != "fail":
            errors.append(f"{source}: fail-closed QI artifact must keep residual_convergence='fail'")

    production_gate = str(gates.get("production_gpu_qi_performance", "")).lower()
    residual_gate = str(gates.get("residual_convergence", "")).lower()
    if production_gate == "pass" and residual_gate != "pass":
        errors.append(f"{source}: production GPU QI performance cannot pass while residual gate is not pass")

    execution_summary = (
        payload.get("execution_summary")
        if isinstance(payload.get("execution_summary"), Mapping)
        else {}
    )
    if execution_summary:
        accepted = int(execution_summary.get("accepted_converged", 0) or 0)
        outputs = int(execution_summary.get("outputs_written", 0) or 0)
        if accepted == 0 and outputs > 0:
            errors.append(f"{source}: non-accepted QI device summary must not count written outputs")

    return errors


def check_qi_device_artifact_file(path: str | Path) -> QIDeviceArtifactCheck:
    """Check one JSON artifact file and classify non-QI files as irrelevant."""

    artifact_path = Path(path)
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except OSError as exc:
        return QIDeviceArtifactCheck(artifact_path, True, (f"{artifact_path}: could not read: {exc}",))
    except json.JSONDecodeError as exc:
        return QIDeviceArtifactCheck(
            artifact_path,
            True,
            (f"{artifact_path}: invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}",),
        )
    if not isinstance(payload, Mapping):
        return QIDeviceArtifactCheck(artifact_path, False, ())

    text = " ".join(
        (
            str(artifact_path),
            str(payload.get("artifact_kind", "")),
            str(payload.get("status", "")),
            str(payload.get("probe_preset", "")),
            str(payload.get("workflow", "")),
        )
    ).lower()
    relevant = _looks_like_qi_device_artifact(text, payload)
    errors = qi_device_artifact_errors(payload, path=artifact_path) if relevant else []
    return QIDeviceArtifactCheck(artifact_path, relevant, tuple(errors))


def check_qi_device_artifact_files(paths: Iterable[str | Path]) -> tuple[QIDeviceArtifactCheck, ...]:
    """Check a collection of JSON artifact files."""

    return tuple(check_qi_device_artifact_file(path) for path in paths)


def load_json_object(path: str | Path) -> dict[str, Any]:
    """Load a JSON file that must contain one JSON object."""

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
    """Gate a bounded QI ``15x`` GPU campaign before it is used as evidence.

    This validation belongs with QI device-artifact policy rather than the
    reusable workflow layer: it checks provenance, residual gates, and
    CPU/Fortran root agreement for one fixed-resolution evidence rung.
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
    """Load campaign/reference JSON files and gate the QI ``15x`` GPU artifact."""

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


def _looks_like_qi_device_artifact(text: str, payload: Mapping[str, Any]) -> bool:
    if ("device_qi" in text or "device-qi" in text) and "qi" in text:
        return True
    if "operator_reuse" in text and "qi" in text:
        return True
    route = payload.get("route")
    if isinstance(route, Mapping) and route.get("operator_reuse_enabled") is True:
        return True
    return False


def _legacy_fail_closed_gpu_artifact(text: str, payload: Mapping[str, Any]) -> bool:
    """Return true for old no-output GPU blocker artifacts that predate backend fields."""

    if "gpu" not in text:
        return False
    execution_summary = (
        payload.get("execution_summary")
        if isinstance(payload.get("execution_summary"), Mapping)
        else {}
    )
    if execution_summary:
        outputs = int(execution_summary.get("outputs_written", 0) or 0)
        accepted = int(execution_summary.get("accepted_converged", 0) or 0)
        passed = payload.get("gates", {}).get("passed") if isinstance(payload.get("gates"), Mapping) else None
        return outputs == 0 and accepted == 0 and passed is False
    result = payload.get("result") if isinstance(payload.get("result"), Mapping) else {}
    if result:
        return result.get("output_refused") is True and result.get("converged") is False
    return False


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
    return out if math.isfinite(out) else None


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
