#!/usr/bin/env python3
"""Audit bounded joint pitch-speed and route-aware tail evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/ambipolar_joint_pitch_speed_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(left: Any, right: Any) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), 1.0e-300)
    return float(np.max(np.abs(right - left) / scale))


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    rungs = {rung["name"]: rung for rung in payload["rungs"]}
    case_ids = {
        "analytic_uniform": payload["analytic_full_state_oracle"]["case_id"],
        **{name: rung["case_id"] for name, rung in rungs.items()},
    }
    for name, record in payload["source"]["inputs"].items():
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            errors.append(f"input provenance mismatch: {name}")
            continue
        case = dkx.Case.from_file(path)
        if case.case_id != case_ids[name]:
            errors.append(f"input case ID mismatch: {name}")
        # Compact CI checks the tracked decks without cloning the reference-code
        # workspace.  Require the sibling geometry only for the optional raw-result
        # audit, where the external files and their originating input must coexist.
        if (
            results_root is not None
            and case.geometry.format == "boozer"
            and not case.geometry_path.is_file()
        ):
            errors.append(f"fresh-workspace geometry path is not executable: {name}")

    pairs = (
        ("nx6_pitch44", "nx8_pitch44", "speed6_to_speed8_at_pitch44"),
        ("nx8_pitch44", "nx8_pitch52", "pitch44_to_pitch52_at_speed8"),
    )
    for left, right, comparison_name in pairs:
        stored = next(
            row for row in payload["comparisons"] if row["name"] == comparison_name
        )
        expected = {
            "particle_flux_scaled_movement": _relative(
                rungs[left]["particle_flux_m2_s"], rungs[right]["particle_flux_m2_s"]
            ),
            "heat_flux_scaled_movement": _relative(
                rungs[left]["heat_flux_W_m2"], rungs[right]["heat_flux_W_m2"]
            ),
            "radial_current_absolute_change_A_m2": abs(
                rungs[right]["radial_current_A_m2"]
                - rungs[left]["radial_current_A_m2"]
            ),
        }
        for key, value in expected.items():
            if stored[key] != value:
                errors.append(f"comparison arithmetic mismatch: {comparison_name}/{key}")

    outcome = payload["outcome"]
    if outcome["phase_space_converged"] or outcome["status"] != "diagnostic_complete":
        errors.append("unconverged outcome is not explicit")
    if not all(
        rung["diagnostic_status"] == "unavailable_on_zero_padded_truncated_state"
        for rung in rungs.values()
    ):
        errors.append("truncated-route tail unavailability is not explicit")

    external_verified = results_root is not None
    if results_root is not None:
        for name, record in payload["source"]["external_results"].items():
            path = results_root / record["file"]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                errors.append(f"external result provenance mismatch: {name}")
                continue
            result = dkx.Result.load(path)
            if result.case_id != case_ids[name]:
                errors.append(f"external result case mismatch: {name}")
                continue
            if name == "analytic_uniform":
                if result.metadata["legendre_tail_diagnostic"] != (
                    "retained_full_state_relative_l2"
                ) or "evaluation_legendre_tail_relative_l2" not in result.arrays:
                    errors.append("full-state analytic tail is not retained")
                else:
                    actual = np.nanmax(
                        result.evaluation_legendre_tail_relative_l2[
                            np.isfinite(result.evaluation_electric_field_kV_m)
                        ],
                        axis=0,
                    )
                    expected = payload["analytic_full_state_oracle"][
                        "tail_max_by_speed_species"
                    ]
                    if not np.array_equal(actual, np.asarray(expected)):
                        errors.append("full-state analytic tail mismatch")
                continue
            if "evaluation_legendre_tail_relative_l2" in result.arrays or result.metadata[
                "legendre_tail_diagnostic"
            ] != "unavailable_on_zero_padded_truncated_state":
                errors.append(f"truncated result presents a false modal tail: {name}")
            fields = np.asarray(result.evaluation_electric_field_kV_m[1])
            index = int(np.nanargmin(np.abs(fields - 8.55)))
            if fields[index] != 8.55:
                errors.append(f"common field missing: {name}")
                continue
            for array_name, compact_name in (
                ("evaluation_particle_flux_m2_s", "particle_flux_m2_s"),
                ("evaluation_heat_flux_W_m2", "heat_flux_W_m2"),
            ):
                if not np.array_equal(
                    np.asarray(getattr(result, array_name)[1, index]),
                    np.asarray(rungs[name][compact_name]),
                ):
                    errors.append(f"external compact value mismatch: {name}/{compact_name}")

    return {
        "schema": "dkx.ambipolar_joint_pitch_speed.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": external_verified,
        "phase_space_converged": outcome["phase_space_converged"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
