#!/usr/bin/env python3
"""Audit the bounded combined low/intermediate pitch diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_pitch_combined_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _movement(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    def relative(name: str) -> float:
        a = np.asarray(left[name], dtype=np.float64)
        b = np.asarray(right[name], dtype=np.float64)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)
        return float(np.max(np.abs(a - b) / scale))

    return {
        "electric_field_kV_m": float(
            np.max(
                np.abs(
                    np.asarray(left["selected_electric_field_kV_m"])
                    - np.asarray(right["selected_electric_field_kV_m"])
                )
            )
        ),
        "particle_flux_scaled": relative("selected_particle_flux_m2_s"),
        "heat_flux_scaled": relative("selected_heat_flux_W_m2"),
    }


def audit(
    artifact: Path,
    *,
    results_root: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = payload["source"]
    for record in source["inputs"].values():
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            errors.append(f"input provenance mismatch: {record['path']}")
    prior = source["prior_artifact"]
    if _sha256(ROOT / prior["path"]) != prior["sha256"]:
        errors.append("prior artifact checksum mismatch")

    rungs = payload["rungs"]
    expected_allocations = [
        [4, 4, 24, 25, 36, 36],
        [12, 12, 24, 25, 36, 36],
        [12, 12, 28, 29, 36, 36],
    ]
    if [rung["allocation"] for rung in rungs] != expected_allocations:
        errors.append("allocation ladder mismatch")
    if any(rung["root_counts"] != [1, 3] for rung in rungs):
        errors.append("root topology is not the retained [1,3] topology")
    for index, comparison in enumerate(payload["comparisons"]):
        recomputed = _movement(rungs[index], rungs[index + 1])
        if recomputed != comparison["maximum_selected_movement"]:
            errors.append(f"stored movement arithmetic mismatch: {comparison['name']}")

    outcome = payload["outcome"]
    last = payload["comparisons"][-1]["maximum_selected_movement"]
    expected_gates = {
        "topology_stable": True,
        "maximum_true_residual_below_1e-12": max(
            rung["maximum_accepted_true_residual"] for rung in rungs
        )
        <= 1.0e-12,
        "all_process_footprints_below_24_gib": max(
            row["peak_footprint_bytes"] for row in payload["measurements"].values()
        )
        < 24 * 2**30,
        "combined_cold_warm_scientific_arrays_exact": payload[
            "combined_cold_warm_parity"
        ]["scientific_arrays_exact"],
        "electric_field_movement_below_0_005_kV_m": last["electric_field_kV_m"]
        <= 0.005,
        "particle_flux_movement_below_0_02": last["particle_flux_scaled"] <= 0.02,
        "heat_flux_movement_below_0_02": last["heat_flux_scaled"] <= 0.02,
    }
    if outcome["gates"] != expected_gates:
        errors.append("stored outcome gates disagree with recomputed gates")
    if outcome["status"] != "refinement_exhausted" or outcome[
        "phase_space_converged"
    ]:
        errors.append("failed convergence was not retained explicitly")

    external_verified = results_root is not None
    if results_root is not None:
        names = {
            "combined_cold": "combined_pair_bounded.nc",
            "combined_warm": "combined_pair_bounded_warm.nc",
            "intermediate28_cold": "intermediate28_pair_bounded.nc",
        }
        for key, name in names.items():
            path = results_root / name
            record = source["external_results"][key]
            if (
                not path.is_file()
                or path.stat().st_size != record["bytes"]
                or _sha256(path) != record["sha256"]
            ):
                errors.append(f"external result provenance mismatch: {name}")

    return {
        "schema": "dkx.ambipolar_pitch_combined.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": external_verified,
        "status": outcome["status"],
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
