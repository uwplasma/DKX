"""Audit admitted-grid W7-X interval discovery, extension, and replay."""

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
DEFAULT_ARTIFACT = ROOT / "validation/w7x_admitted_grid_seeded_envelope_v1.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _finite_roots(result: dkx.Result) -> list[float]:
    return [
        float(value)
        for value in result.ambipolar_root_kV_m.ravel()
        if np.isfinite(value)
    ]


def _arrays_equal_except_solve_time(left: dkx.Result, right: dkx.Result) -> bool:
    for name, first in left.arrays.items():
        if name == "solve_time_s":
            continue
        second = right.arrays[name]
        if np.issubdtype(first.dtype, np.number):
            if not np.array_equal(first, second, equal_nan=True):
                return False
        elif not np.array_equal(first, second):
            return False
    return True


def audit(artifact_path: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if artifact.get("schema") != "dkx.w7x_admitted_grid_seeded_envelope.v1":
        errors.append("unexpected artifact schema")

    checked_cases: dict[str, dkx.Case] = {}
    for name, record in artifact["source"]["checked_inputs"].items():
        path = ROOT / record["path"]
        if not path.is_file() or _sha256(path) != record["sha256"]:
            errors.append(f"checked input provenance mismatch: {name}")
            continue
        case = dkx.Case.from_file(path)
        checked_cases[name] = case
        if case.case_id != record["case_id"]:
            errors.append(f"checked input case ID mismatch: {name}")
        if case.electric_field.search_strategy != "seeded_brackets":
            errors.append(f"checked input is not seeded: {name}")
        if case.convergence.enabled:
            errors.append(f"checked input enables global convergence: {name}")

    if "envelopes" in checked_cases:
        preflight = preflight_ambipolar_case(checked_cases["envelopes"])
        if list(preflight.search_points_by_surface) != [2, 5]:
            errors.append("envelope endpoint preflight mismatch")
        if preflight.profile_evaluations != 55:
            errors.append("envelope profile bound mismatch")
    if "final" in checked_cases:
        seeds = checked_cases["final"].electric_field.seed_brackets_kV_m
        flat_seeds = [list(surface[0]) for surface in seeds or ()]
        if flat_seeds != artifact["final_replay"]["brackets_kV_m"]:
            errors.append("final checked seeds do not match promoted brackets")

    final = artifact["final_replay"]
    if final["topology"] != [1, 1] or final["evaluations"] != [2, 2]:
        errors.append("final replay topology or evaluation count mismatch")
    if final["maximum_primal_residual"] > 1.0e-12:
        errors.append("final replay residual exceeds 1e-12")
    if not all(left * right < 0.0 for left, right in final["endpoint_currents_A_m2"]):
        errors.append("a final bracket lacks a strict endpoint sign change")
    if not final["cold_warm_arrays_exact_except_solve_time_s"]:
        errors.append("cold/warm final replay parity is not retained")
    if artifact["case"]["unsampled_crossings_excluded"]:
        errors.append("artifact incorrectly excludes unsampled crossings")
    if artifact["outcome"]["global_all_root_claim"]:
        errors.append("artifact incorrectly promotes a global all-root claim")
    if not all(
        value < 0.0
        for value in artifact["envelope_run"][
            "surface_1_endpoint_currents_A_m2"
        ].values()
    ):
        errors.append("failed surface-1 envelope signs are not all retained")

    external_verified = False
    if results_root is not None:
        results_root = results_root.resolve()
        external_verified = True
        for name, expected in artifact["source"]["external_files"].items():
            path = results_root / name
            if not path.is_file() or _sha256(path) != expected:
                errors.append(f"external provenance mismatch: {name}")
                external_verified = False
        if external_verified:
            envelope = dkx.Result.load(results_root / "result.nc")
            extension = dkx.Result.load(results_root / "extension.nc")
            cold = dkx.Result.load(results_root / "final-cold.nc")
            warm = dkx.Result.load(results_root / "final-warm.nc")
            if envelope.ambipolar_status.tolist() != [
                "bracketed_root",
                "seeded_bracket_failed",
            ]:
                errors.append("external envelope status mismatch")
            if _finite_roots(extension) != final["roots_kV_m"]:
                errors.append("external extension roots mismatch")
            if _finite_roots(cold) != final["roots_kV_m"]:
                errors.append("external final roots mismatch")
            if not _arrays_equal_except_solve_time(cold, warm):
                errors.append("external cold/warm arrays differ beyond solve time")
            if set(warm.ambipolar_search_scope.tolist()) != {
                "explicit_seeded_intervals_only"
            }:
                errors.append("external final search scope mismatch")
            if float(np.nanmax(warm.evaluation_primal_residual)) > 1.0e-12:
                errors.append("external final residual exceeds 1e-12")

    return {
        "schema": "dkx.w7x_admitted_grid_seeded_envelope.audit.v1",
        "artifact": str(artifact_path),
        "external_results_verified": external_verified,
        "topology": final["topology"],
        "roots_kV_m": final["roots_kV_m"],
        "global_all_root_claim": artifact["outcome"]["global_all_root_claim"],
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
