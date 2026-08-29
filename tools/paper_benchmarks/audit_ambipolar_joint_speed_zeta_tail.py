#!/usr/bin/env python3
"""Audit bounded joint speed/zeta selected-tail evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/ambipolar_joint_speed_zeta_tail_v1.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    scale = np.maximum(
        np.maximum(np.abs(left_array), np.abs(right_array)), 1.0e-300
    )
    return float(np.max(np.abs(right_array - left_array) / scale))


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    rungs = {rung["name"]: rung for rung in payload["rungs"]}

    for name, record in payload["source"]["inputs"].items():
        path = ROOT / record["path"]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            errors.append(f"input provenance mismatch: {name}")
            continue
        case = dkx.Case.from_file(path)
        if case.case_id != rungs[name]["case_id"]:
            errors.append(f"input case ID mismatch: {name}")
        if not case.convergence.retain_legendre_tail:
            errors.append(f"tail retention is not requested: {name}")
        if results_root is not None and not case.geometry_path.is_file():
            errors.append(f"fresh-workspace geometry path is not executable: {name}")

    pairs = (
        ("nx8_zeta37", "nx10_zeta37", "speed8_to_speed10_at_zeta37"),
        ("nx8_zeta45", "nx10_zeta45", "speed8_to_speed10_at_zeta45"),
        ("nx8_zeta37", "nx8_zeta45", "zeta37_to_zeta45_at_speed8"),
        ("nx10_zeta37", "nx10_zeta45", "zeta37_to_zeta45_at_speed10"),
    )
    comparisons = {row["name"]: row for row in payload["comparisons"]}
    for left, right, comparison_name in pairs:
        for compact_name, movement_name in (
            ("particle_flux_m2_s", "particle_flux_scaled_movement"),
            ("heat_flux_W_m2", "heat_flux_scaled_movement"),
        ):
            actual = _relative(rungs[left][compact_name], rungs[right][compact_name])
            if comparisons[comparison_name][movement_name] != actual:
                errors.append(
                    f"comparison arithmetic mismatch: {comparison_name}/{movement_name}"
                )

    outcome = payload["outcome"]
    if (
        outcome["status"] != "diagnostic_complete"
        or outcome["phase_space_converged"]
        or outcome["whole_profile_escalation_admitted"]
    ):
        errors.append("unconverged outcome is not explicit")
    if not any(
        row["particle_flux_scaled_movement"] > 0.02
        and row["heat_flux_scaled_movement"] > 0.02
        for row in comparisons.values()
    ):
        errors.append("failed observable gates are not retained")
    tail_maxima = [rung["maximum_selected_tail_bound"] for rung in rungs.values()]
    if not all(np.isfinite(tail_maxima)) or max(tail_maxima) < 0.09:
        errors.append("selected-tail conditioning signal is not retained")

    external_verified = results_root is not None
    if results_root is not None:
        for name, versions in payload["source"]["external_results"].items():
            loaded = []
            for temperature in ("cold", "warm"):
                record = versions[temperature]
                path = results_root / record["file"]
                if (
                    not path.is_file()
                    or path.stat().st_size != record["bytes"]
                    or _sha256(path) != record["sha256"]
                ):
                    errors.append(
                        f"external result provenance mismatch: {name}/{temperature}"
                    )
                    continue
                result = dkx.Result.load(path)
                loaded.append(result)
                rung = rungs[name]
                if result.case_id != rung["case_id"]:
                    errors.append(f"external result case mismatch: {name}/{temperature}")
                    continue
                if "evaluation_legendre_tail_relative_l2" in result.arrays:
                    errors.append(f"truncated result presents an exact tail: {name}")
                if result.metadata["legendre_tail_diagnostic"] != (
                    "retained_selected_tail_relative_l2_upper_bound"
                ):
                    errors.append(f"selected-tail status mismatch: {name}")
                if not np.array_equal(
                    np.asarray(result.ambipolar_status),
                    np.asarray(["no_bracketed_root", "no_bracketed_root"]),
                ):
                    errors.append(f"narrow-window status mismatch: {name}")

                fields = np.asarray(result.evaluation_electric_field_kV_m)
                particle = []
                heat = []
                selected_tail = []
                tail = np.asarray(
                    result.evaluation_legendre_tail_relative_l2_upper_bound
                )
                for surface_index, selected_field in enumerate(
                    np.asarray(result.electric_field_kV_m)
                ):
                    common_index = int(
                        np.nanargmin(np.abs(fields[surface_index] - 8.55))
                    )
                    selected_index = int(
                        np.nanargmin(
                            np.abs(fields[surface_index] - selected_field)
                        )
                    )
                    if fields[surface_index, common_index] != 8.55:
                        errors.append(f"common field missing: {name}")
                    particle.append(
                        result.evaluation_particle_flux_m2_s[
                            surface_index, common_index
                        ]
                    )
                    heat.append(
                        result.evaluation_heat_flux_W_m2[surface_index, common_index]
                    )
                    selected_tail.append(tail[surface_index, selected_index])
                if not np.array_equal(np.asarray(particle), rung["particle_flux_m2_s"]):
                    errors.append(f"compact particle flux mismatch: {name}")
                if not np.array_equal(np.asarray(heat), rung["heat_flux_W_m2"]):
                    errors.append(f"compact heat flux mismatch: {name}")
                if float(np.max(selected_tail)) != rung["maximum_selected_tail_bound"]:
                    errors.append(f"compact tail maximum mismatch: {name}")
                accepted = np.asarray(result.evaluation_solver_attempt_accepted).astype(
                    bool
                )
                residuals = np.asarray(result.evaluation_solver_attempt_residual)
                if (
                    int(np.count_nonzero(accepted)) != rung["accepted_attempt_count"]
                    or float(np.max(residuals[accepted]))
                    != rung["maximum_accepted_true_residual"]
                ):
                    errors.append(f"accepted-solve evidence mismatch: {name}")

            if len(loaded) == 2:
                for array_name in loaded[0].arrays:
                    if array_name == "solve_time_s":
                        continue
                    left = np.asarray(loaded[0].arrays[array_name])
                    right = np.asarray(loaded[1].arrays[array_name])
                    equal = (
                        np.array_equal(left, right)
                        if left.dtype.kind in "OUS"
                        else np.array_equal(left, right, equal_nan=True)
                    )
                    if not equal:
                        errors.append(
                            f"cold/warm scientific array mismatch: {name}/{array_name}"
                        )

    return {
        "schema": "dkx.ambipolar_joint_speed_zeta_tail.audit.v1",
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
