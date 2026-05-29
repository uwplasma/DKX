"""Offline policy checks for QI device/GPU evidence artifacts.

The benchmark artifact policy is intentionally PAS-focused, so QI device
artifacts need a separate CI-fast gate.  These checks do not prove a solve is
converged; they make overclaiming hard by requiring fail-closed metadata for
nonconverged device-QI runs and backend/provenance fields for GPU-route claims.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    if "gpu" in text and backend != "gpu" and jax_platform != "gpu":
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


def _looks_like_qi_device_artifact(text: str, payload: Mapping[str, Any]) -> bool:
    if ("device_qi" in text or "device-qi" in text) and "qi" in text:
        return True
    if "operator_reuse" in text and "qi" in text:
        return True
    route = payload.get("route")
    if isinstance(route, Mapping) and route.get("operator_reuse_enabled") is True:
        return True
    return False
