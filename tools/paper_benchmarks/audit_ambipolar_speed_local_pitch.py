#!/usr/bin/env python3
"""Audit common-field speed-local pitch evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/ambipolar_speed_local_pitch_v1.json"


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


def _comparison(left: dict[str, Any], right: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "name": name,
        "particle_flux_scaled_movement": _relative(
            left["particle_flux_m2_s"], right["particle_flux_m2_s"]
        ),
        "heat_flux_scaled_movement": _relative(
            left["heat_flux_W_m2"], right["heat_flux_W_m2"]
        ),
        "radial_current_absolute_change_A_m2": abs(
            right["radial_current_A_m2"] - left["radial_current_A_m2"]
        ),
    }


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    names = ["combined", "intermediate28", "node3_33", "node3_36", "pitch44"]
    for name in names:
        record = payload["source"]["inputs"][name]
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            errors.append(f"input provenance mismatch: {name}")
        if dkx.Case.from_file(path).case_id != next(
            rung["case_id"] for rung in payload["rungs"] if rung["name"] == name
        ):
            errors.append(f"input case ID mismatch: {name}")

    rungs = {rung["name"]: rung for rung in payload["rungs"]}
    pairs = [
        ("combined", "intermediate28", "combined_to_intermediate28"),
        ("combined", "node3_33", "combined_to_node3_33"),
        ("node3_33", "node3_36", "node3_33_to_node3_36"),
        ("node3_36", "pitch44", "node3_36_to_pitch44"),
    ]
    for left, right, name in pairs:
        expected = _comparison(rungs[left], rungs[right], name)
        stored = next(row for row in payload["comparisons"] if row["name"] == name)
        for key, value in expected.items():
            if key != "name" and stored[key] != value:
                errors.append(f"comparison arithmetic mismatch: {name}/{key}")

    outcome = payload["outcome"]
    gates = outcome["gates"]
    if not all(
        rung["maximum_accepted_true_residual"] <= 1.0e-12
        for rung in rungs.values()
    ) or not gates["all_true_residuals_below_1e-12"]:
        errors.append("true-residual gate mismatch")
    if not all(rung["peak_host_memory_bytes"] < 2 * 2**30 for rung in rungs.values()) or not gates[
        "all_peak_host_memory_below_2_gib"
    ]:
        errors.append("memory gate mismatch")
    if outcome["phase_space_converged"] or outcome["status"] != "diagnostic_complete":
        errors.append("unconverged outcome is not explicit")

    external_verified = results_root is not None
    if results_root is not None:
        loaded = {}
        for name in names:
            record = payload["source"]["external_results"][name]
            path = results_root / record["file"]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                errors.append(f"external result provenance mismatch: {name}")
                continue
            result = dkx.Result.load(path)
            loaded[name] = result
            fields = np.asarray(result.evaluation_electric_field_kV_m[1])
            index = int(np.nanargmin(np.abs(fields - 8.55)))
            if fields[index] != 8.55 or result.case_id != rungs[name]["case_id"]:
                errors.append(f"external common-field identity mismatch: {name}")
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
            if float(result.radial_current_A_m2[1, index]) != rungs[name]["radial_current_A_m2"]:
                errors.append(f"external radial-current mismatch: {name}")
        if {"combined", "intermediate28"} <= loaded.keys():
            shares = {}
            for array_name, label in (
                ("evaluation_particle_flux_m2_s_vs_speed", "particle"),
                ("evaluation_heat_flux_W_m2_vs_speed", "heat"),
            ):
                values = []
                for name in ("combined", "intermediate28"):
                    result = loaded[name]
                    fields = np.asarray(result.evaluation_electric_field_kV_m[1])
                    index = int(np.nanargmin(np.abs(fields - 8.55)))
                    values.append(np.asarray(getattr(result, array_name)[1, index]))
                delta = np.abs(values[1] - values[0])
                shares[label] = (delta[3] / np.sum(delta, axis=0)).tolist()
            stored = payload["comparisons"][0]["node3_absolute_delta_share"]
            if shares != stored:
                errors.append("external node-3 localization mismatch")

    return {
        "schema": "dkx.ambipolar_speed_local_pitch.audit.v1",
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
