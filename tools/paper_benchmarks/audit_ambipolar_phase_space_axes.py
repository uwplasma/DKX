#!/usr/bin/env python3
"""Build or audit the bounded theta/pitch ambipolar resolution diagnosis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 floor
    import tomli as tomllib

import numpy as np

from audit_ambipolar_phase_space_ladder import (
    _canonical_sha256,
    _compact_rung,
    _compare_rungs,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_phase_space_axes_v1.json"
NAMES = ("reference", "theta17", "pitch40", "pitch44")
INPUTS = {
    "reference": ROOT / "validation/inputs/w7x_standard_native_ambipolar_profile.toml",
    "theta17": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_profile_theta17.toml",
    "pitch40": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_profile_pitch40.toml",
    "pitch44": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_profile_pitch44.toml",
}
MEASUREMENTS = {
    "reference": {
        "process_wall_s": 301.64,
        "command_stopwatch_s": 299.5713892089989,
        "maximum_rss_bytes": 13_588_021_248,
        "peak_footprint_bytes": 17_728_371_400,
    },
    "theta17": {
        "process_wall_s": 405.03,
        "command_stopwatch_s": 403.1844017500007,
        "maximum_rss_bytes": 13_031_489_536,
        "peak_footprint_bytes": 21_386_708_200,
    },
    "pitch40": {
        "process_wall_s": 326.91,
        "command_stopwatch_s": 325.23692429199946,
        "maximum_rss_bytes": 6_147_063_808,
        "peak_footprint_bytes": 5_633_497_568,
    },
    "pitch44": {
        "process_wall_s": 395.70,
        "command_stopwatch_s": 393.843974833002,
        "maximum_rss_bytes": 13_231_325_184,
        "peak_footprint_bytes": 22_275_409_800,
    },
}


def _named_comparison(name: str, left: Any, right: Any) -> dict[str, Any]:
    return {"name": name, **_compare_rungs(left, right)}


def _gate(comparison: dict[str, Any], max_residual: float) -> dict[str, Any]:
    summary = comparison["summary"]
    gates = {
        "topology_stable": summary["topology_stable"],
        "all_root_electric_field_movement": (
            summary["max_all_root_electric_field_movement_kV_m"] <= 0.005
        ),
        "selected_particle_flux_movement": (
            summary["max_selected_particle_flux_scaled_movement"] <= 0.02
        ),
        "selected_heat_flux_movement": (
            summary["max_selected_heat_flux_scaled_movement"] <= 0.02
        ),
        "accepted_true_residual": max_residual <= 1.0e-12,
    }
    return {
        "status": "resolved" if all(gates.values()) else "refinement_exhausted",
        "admission_pass": all(gates.values()),
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
    }


def _outcome(
    rungs: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    max_residual = max(
        rung["attempts"]["maximum_accepted_true_residual"] for rung in rungs
    )
    theta_gate = _gate(comparisons[0], max_residual)
    pitch40_gate = _gate(comparisons[1], max_residual)
    pitch44_gate = _gate(comparisons[2], max_residual)
    theta = comparisons[0]["summary"]
    pitch = comparisons[1]["summary"]
    return {
        "status": "refinement_exhausted",
        "admission_pass": False,
        "limits": {
            "max_all_root_electric_field_movement_kV_m": 0.005,
            "max_selected_particle_flux_scaled_movement": 0.02,
            "max_selected_heat_flux_scaled_movement": 0.02,
            "max_accepted_true_residual": 1.0e-12,
        },
        "measured_max_accepted_true_residual": max_residual,
        "theta17_vs_reference": theta_gate,
        "pitch40_vs_reference": pitch40_gate,
        "pitch44_vs_pitch40": pitch44_gate,
        "diagnosis": {
            "dominant_failed_direction": "pitch",
            "theta_max_root_movement_kV_m": theta[
                "max_all_root_electric_field_movement_kV_m"
            ],
            "pitch40_max_root_movement_kV_m": pitch[
                "max_all_root_electric_field_movement_kV_m"
            ],
            "theta_max_selected_flux_scaled_movement": max(
                theta["max_selected_particle_flux_scaled_movement"],
                theta["max_selected_heat_flux_scaled_movement"],
            ),
            "pitch40_max_selected_flux_scaled_movement": max(
                pitch["max_selected_particle_flux_scaled_movement"],
                pitch["max_selected_heat_flux_scaled_movement"],
            ),
            "pitch40_to_pitch44_approaches_gate": False,
            "pitch48_bruteforce_admitted": False,
            "reason": (
                "Pitch40-to-pitch44 movements remain above every scientific "
                "gate and several observables move farther from the reference; "
                "the pitch44 process reached a 22275409800-byte footprint on "
                "the 24 GiB host."
            ),
        },
    }


def build_artifact(result_paths: dict[str, Path], geometry: Path) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    resolutions = {
        name: {
            key: int(value)
            for key, value in tomllib.loads(path.read_text(encoding="utf-8"))[
                "resolution"
            ].items()
        }
        for name, path in INPUTS.items()
    }
    results = {name: dkx.Result.load(result_paths[name]) for name in NAMES}
    rungs = [_compact_rung(results[name], resolutions[name]) for name in NAMES]
    comparisons = [
        _named_comparison("reference_to_theta17", rungs[0], rungs[1]),
        _named_comparison("reference_to_pitch40", rungs[0], rungs[2]),
        _named_comparison("pitch40_to_pitch44", rungs[2], rungs[3]),
    ]
    compact = {"rungs": rungs, "comparisons": comparisons}
    reference = results["reference"]
    return {
        "schema": "dkx.ambipolar_phase_space_axes.v1",
        "created_utc": "2026-08-29T13:02:00Z",
        "claim_scope": "bounded_w7x_pas_dkes_theta_pitch_nonconvergence_diagnosis",
        "summary": (
            "Separate theta and pitch rungs identify pitch as the dominant "
            "failed resolution direction; pitch36-to-40-to-44 does not approach "
            "the unchanged root and selected-flux gates."
        ),
        "source": {
            "dkx_commit": "a042bd58e95a86667185945ab85327241f47eb16",
            "dkx_version": reference.metadata["dkx_version"],
            "geometry": "SFINCS/equilibria/w7x_standardConfig.bc",
            "geometry_sha256": _sha256(geometry),
            "geometry_bytes": geometry.stat().st_size,
            "inputs": {
                name: {
                    "path": str(path.relative_to(ROOT)),
                    "sha256": _sha256(path),
                }
                for name, path in INPUTS.items()
            },
            "results": {
                name: {
                    "sha256": _sha256(result_paths[name]),
                    "bytes": result_paths[name].stat().st_size,
                }
                for name in NAMES
            },
            "compact_axes_sha256": _canonical_sha256(compact),
        },
        "environment": {
            "host": "Mac mini, Apple M4, 10 CPU cores, 24 GB RAM",
            "os": reference.metadata["platform"],
            "python": reference.metadata["python_version"],
            "jax": reference.metadata["jax_version"],
            "jaxlib": reference.metadata["jaxlib_version"],
            "numpy": np.__version__,
            "dkx_cores": 8,
            "cache_state": "fresh_process_with_initially_empty_external_cache",
            "timing_note": "Timings and memory are provenance, not a performance claim.",
        },
        "measurements": MEASUREMENTS,
        **compact,
        "outcome": _outcome(rungs, comparisons),
        "exclusions": [
            "not_phase_space_convergence_validation",
            "not_zeta_or_speed_convergence_validation",
            "not_independent_cross_code_ambipolar_validation",
            "not_full_fokker_planck_ambipolar_validation",
            "not_experimental_validation",
            "not_cross_code_performance_validation",
        ],
    }


def audit(
    artifact: Path,
    *,
    result_paths: dict[str, Path] | None = None,
    geometry: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = payload["source"]
    for record in source["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            errors.append(f"input checksum mismatch: {record['path']}")
    compact = {"rungs": payload["rungs"], "comparisons": payload["comparisons"]}
    if _canonical_sha256(compact) != source["compact_axes_sha256"]:
        errors.append("compact axes checksum mismatch")
    rungs = payload["rungs"]
    recomputed = [
        _named_comparison("reference_to_theta17", rungs[0], rungs[1]),
        _named_comparison("reference_to_pitch40", rungs[0], rungs[2]),
        _named_comparison("pitch40_to_pitch44", rungs[2], rungs[3]),
    ]
    if recomputed != payload["comparisons"]:
        errors.append("stored axis comparison arithmetic mismatch")
    outcome = _outcome(rungs, recomputed)
    if outcome != payload["outcome"]:
        errors.append("stored axis outcome disagrees with recomputed gates")

    if result_paths is not None:
        import dkx  # noqa: PLC0415

        for index, name in enumerate(NAMES):
            path = result_paths[name]
            record = source["results"][name]
            if _sha256(path) != record["sha256"]:
                errors.append(f"external {name} result checksum mismatch")
                continue
            if path.stat().st_size != record["bytes"]:
                errors.append(f"external {name} result size mismatch")
            resolution = rungs[index]["resolution"]
            if _compact_rung(dkx.Result.load(path), resolution) != rungs[index]:
                errors.append(f"external {name} compact rung mismatch")
        if geometry is None:
            errors.append("external geometry path missing")
        elif _sha256(geometry) != source["geometry_sha256"]:
            errors.append("external geometry checksum mismatch")

    return {
        "schema": "dkx.ambipolar_phase_space_axes.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": result_paths is not None,
        "status": outcome["status"],
        "admission_pass": outcome["admission_pass"],
        "dominant_failed_direction": outcome["diagnosis"]["dominant_failed_direction"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--build", action="store_true")
    for name in NAMES:
        parser.add_argument(f"--{name}-result", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    result_paths = {name: getattr(args, f"{name}_result") for name in NAMES}
    provided = [path is not None for path in result_paths.values()]
    if args.build:
        if not all(provided) or args.geometry is None:
            parser.error("--build requires all four results and --geometry")
        payload = build_artifact(result_paths, args.geometry)
        args.artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"artifact": str(args.artifact), "outcome": payload["outcome"]},
                indent=2,
            )
        )
        return 0
    external = None
    if any(provided):
        if not all(provided):
            parser.error("external audit requires all four result paths")
        external = result_paths
    report = audit(args.artifact, result_paths=external, geometry=args.geometry)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
