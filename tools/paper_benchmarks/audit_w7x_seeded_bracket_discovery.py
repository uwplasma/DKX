"""Audit bounded W7-X discovery and seeded-endpoint replay evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import dkx
from dkx.workflows.ambipolar_native import preflight_ambipolar_case


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/w7x_seeded_bracket_discovery_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten_brackets(value: list[list[list[float]]]) -> list[list[float]]:
    return [bracket for surface in value for bracket in surface]


def audit(artifact_path: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if artifact.get("schema") != "dkx.w7x_seeded_bracket_discovery.v1":
        errors.append("unexpected artifact schema")

    source = artifact["source"]
    for key in ("checked_discovery_input", "checked_seeded_input"):
        path = ROOT / source[key]
        expected = source[f"{key}_sha256"]
        if not path.is_file() or _sha256(path) != expected:
            errors.append(f"checked input provenance mismatch: {key}")

    discovery_case = dkx.Case.from_file(ROOT / source["checked_discovery_input"])
    seeded_case = dkx.Case.from_file(ROOT / source["checked_seeded_input"])
    discovery_preflight = preflight_ambipolar_case(discovery_case)
    seeded_preflight = preflight_ambipolar_case(seeded_case)
    recorded_case = artifact["case"]
    discovery = artifact["discovery"]
    seeded = artifact["seeded_replay"]

    if list(discovery_case.geometry.surfaces) != recorded_case["surfaces_psi_N"]:
        errors.append("checked discovery surfaces do not match the artifact")
    if discovery_preflight.hierarchy_points != recorded_case["hierarchy_points"]:
        errors.append("discovery hierarchy preflight mismatch")
    if discovery_case.electric_field.search_strategy != "uniform":
        errors.append("discovery input is not a uniform global search")
    if seeded_case.electric_field.search_strategy != "seeded_brackets":
        errors.append("replay input is not seeded_brackets")
    if seeded_case.convergence.enabled:
        errors.append("seeded replay incorrectly enables global refinement")
    if list(seeded_preflight.search_points_by_surface) != [2, 6]:
        errors.append("seeded endpoint preflight mismatch")
    if seeded_preflight.evaluations_per_surface != 51:
        errors.append("seeded evaluation bound mismatch")
    if seeded["evaluation_budgets"] != [17, 51]:
        errors.append("recorded per-surface evaluation bounds mismatch")

    checked_brackets = [
        [list(bracket) for bracket in surface]
        for surface in seeded_case.electric_field.seed_brackets_kV_m or ()
    ]
    if checked_brackets != discovery["brackets_kV_m"]:
        errors.append("checked seeds do not exactly match discovery brackets")
    if (
        seeded["topology"] != discovery["topology"]
        or not seeded["roots_and_brackets_exact"]
    ):
        errors.append("seeded replay does not retain exact discovery topology")
    if seeded["search_scope"] != "explicit_seeded_intervals_only":
        errors.append("seeded replay scope is not interval-limited")
    if discovery["refinement_status"] != ["resolved", "resolved"]:
        errors.append("discovery refinement is not resolved")
    if not discovery["cold_warm_science_exact_except_timing"]:
        errors.append("cold/warm discovery parity is not retained")
    if (
        max(discovery["maximum_primal_residual"], seeded["maximum_primal_residual"])
        > 1.0e-12
    ):
        errors.append("accepted primal residual exceeds 1e-12")

    endpoints = seeded["endpoints"]
    if len(endpoints) != len(_flatten_brackets(discovery["brackets_kV_m"])):
        errors.append("not every discovery bracket has endpoint evidence")
    for endpoint in endpoints:
        if endpoint["left_current_A_m2"] * endpoint["right_current_A_m2"] >= 0.0:
            errors.append(
                "endpoint pair does not retain a strict sign change: "
                f"surface {endpoint['surface_index']} bracket {endpoint['bracket_index']}"
            )

    external_verified = False
    if results_root is not None:
        results_root = results_root.resolve()
        external_verified = True
        for name, expected in source["external_files"].items():
            path = results_root / name
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"external provenance mismatch: {name}")
                external_verified = False
        if external_verified:
            cold = dkx.Result.load(results_root / "discovery-cold.nc")
            warm = dkx.Result.load(results_root / "discovery-warm.nc")
            replay = dkx.Result.load(results_root / "seeded-replay.nc")
            if cold.case_id != discovery["external_case_id"]:
                errors.append("external discovery case ID mismatch")
            if replay.case_id != seeded["external_case_id"]:
                errors.append("external seeded case ID mismatch")
            if not np.array_equal(
                cold.ambipolar_root_kV_m,
                warm.ambipolar_root_kV_m,
                equal_nan=True,
            ):
                errors.append("external cold/warm roots differ")
            if not np.array_equal(
                replay.ambipolar_root_bracket_kV_m,
                cold.ambipolar_root_bracket_kV_m,
                equal_nan=True,
            ):
                errors.append("external seeded brackets differ from discovery")
            if replay.solver_iterations.tolist() != seeded["evaluations"]:
                errors.append("external seeded evaluation count mismatch")
            if set(replay.ambipolar_search_scope.tolist()) != {
                "explicit_seeded_intervals_only"
            }:
                errors.append("external seeded scope mismatch")

    return {
        "schema": "dkx.w7x_seeded_bracket_discovery.audit.v1",
        "artifact": str(artifact_path),
        "external_results_verified": external_verified,
        "topology": discovery["topology"],
        "endpoint_count": len(endpoints),
        "admitted_grid_promotion_ready": artifact["outcome"][
            "admitted_grid_promotion_ready"
        ],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifact", nargs="?", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
