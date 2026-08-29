#!/usr/bin/env python3
"""Build or audit the bounded native ambipolar phase-space ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_phase_space_ladder_v1.json"
INPUTS = {
    "coarse": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_profile_coarse.toml",
    "reference": ROOT / "validation/inputs/w7x_standard_native_ambipolar_profile.toml",
    "fine": ROOT / "validation/inputs/w7x_standard_native_ambipolar_profile_fine.toml",
}
MEASUREMENTS = {
    "coarse": {
        "process_wall_s": 123.33,
        "command_stopwatch_s": 121.9769,
        "maximum_rss_bytes": 5_612_601_344,
        "peak_footprint_bytes": 5_226_239_928,
        "cache_state": "empty_external_cache",
    },
    "reference": {
        "process_wall_s": 301.64,
        "command_stopwatch_s": 299.5713892089989,
        "maximum_rss_bytes": 13_588_021_248,
        "peak_footprint_bytes": 17_728_371_400,
        "cache_state": "empty_external_cache",
    },
    "fine": {
        "process_wall_s": 440.57,
        "command_stopwatch_s": 438.8634540419989,
        "maximum_rss_bytes": 13_234_585_600,
        "peak_footprint_bytes": 23_418_227_384,
        "cache_state": "empty_external_cache",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scaled_movement(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)
    return float(np.max(np.abs(a - b) / scale))


def _attempt_summary(result: Any) -> dict[str, Any]:
    metadata = result.metadata["ambipolar_solver_attempts"]
    return {
        "attempt_count": int(metadata["attempt_count"]),
        "executed_route_counts": {
            str(key): int(value)
            for key, value in metadata["executed_route_counts"].items()
        },
        "automatic_true_residual_recovery_count": int(
            metadata["automatic_true_residual_recovery_count"]
        ),
        "maximum_accepted_true_residual": float(result.metadata["residual_norm"]),
    }


def _compact_rung(result: Any, resolution: dict[str, int]) -> dict[str, Any]:
    arrays = result.arrays
    surfaces: list[dict[str, Any]] = []
    for surface_index, count_value in enumerate(arrays["ambipolar_root_count"]):
        roots: list[dict[str, Any]] = []
        valid = np.flatnonzero(
            np.isfinite(arrays["evaluation_electric_field_kV_m"][surface_index])
        )
        selected_index = int(arrays["selected_ambipolar_root"][surface_index])
        for root_index in range(int(count_value)):
            electric_field = float(
                arrays["ambipolar_root_kV_m"][surface_index, root_index]
            )
            matches = valid[
                np.isclose(
                    arrays["evaluation_electric_field_kV_m"][surface_index, valid],
                    electric_field,
                    rtol=0.0,
                    atol=1.0e-14,
                )
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"surface {surface_index} root {root_index} has "
                    f"{len(matches)} retained evaluations"
                )
            evaluation_index = int(matches[0])
            roots.append(
                {
                    "index": root_index,
                    "classification": str(
                        arrays["ambipolar_root_type"][surface_index, root_index]
                    ),
                    "branch_id": str(
                        arrays["ambipolar_root_branch_id"][surface_index, root_index]
                    ),
                    "selected": root_index == selected_index,
                    "electric_field_kV_m": electric_field,
                    "particle_flux_m2_s": [
                        float(value)
                        for value in arrays["evaluation_particle_flux_m2_s"][
                            surface_index, evaluation_index
                        ]
                    ],
                    "heat_flux_W_m2": [
                        float(value)
                        for value in arrays["evaluation_heat_flux_W_m2"][
                            surface_index, evaluation_index
                        ]
                    ],
                    "parallel_current_A_T_m2": float(
                        arrays["evaluation_parallel_current_A_T_m2"][
                            surface_index, evaluation_index
                        ]
                    ),
                    "primal_residual": float(
                        arrays["evaluation_primal_residual"][
                            surface_index, evaluation_index
                        ]
                    ),
                    "final_bracket_width_kV_m": float(
                        arrays["ambipolar_root_final_bracket_width_kV_m"][
                            surface_index, root_index
                        ]
                    ),
                }
            )
        surfaces.append(
            {
                "surface_index": surface_index,
                "psi_N": float(arrays["surface"][surface_index]),
                "root_count": len(roots),
                "roots": roots,
                "selected_root_index": selected_index,
                "selected_branch_id": str(
                    arrays["selected_ambipolar_branch"][surface_index]
                ),
            }
        )
    return {
        "case_id": result.case_id,
        "resolution": resolution,
        "surfaces": surfaces,
        "attempts": _attempt_summary(result),
    }


def _compare_rungs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    topology_stable = True
    root_movements: list[dict[str, Any]] = []
    selected_movements: list[dict[str, Any]] = []
    for left_surface, right_surface in zip(
        left["surfaces"], right["surfaces"], strict=True
    ):
        left_roots = left_surface["roots"]
        right_roots = right_surface["roots"]
        if len(left_roots) != len(right_roots):
            topology_stable = False
            continue
        for left_root, right_root in zip(left_roots, right_roots, strict=True):
            identity_stable = (
                left_root["classification"] == right_root["classification"]
                and left_root["branch_id"] == right_root["branch_id"]
            )
            topology_stable &= identity_stable
            root_movements.append(
                {
                    "surface_index": left_surface["surface_index"],
                    "root_index": left_root["index"],
                    "classification": left_root["classification"],
                    "branch_id": left_root["branch_id"],
                    "identity_stable": identity_stable,
                    "left_electric_field_kV_m": left_root["electric_field_kV_m"],
                    "right_electric_field_kV_m": right_root["electric_field_kV_m"],
                    "electric_field_movement_kV_m": abs(
                        right_root["electric_field_kV_m"]
                        - left_root["electric_field_kV_m"]
                    ),
                    "particle_flux_scaled_movement": _scaled_movement(
                        left_root["particle_flux_m2_s"],
                        right_root["particle_flux_m2_s"],
                    ),
                    "heat_flux_scaled_movement": _scaled_movement(
                        left_root["heat_flux_W_m2"], right_root["heat_flux_W_m2"]
                    ),
                    "parallel_current_scaled_movement": _scaled_movement(
                        left_root["parallel_current_A_T_m2"],
                        right_root["parallel_current_A_T_m2"],
                    ),
                }
            )
        left_selected = left_roots[left_surface["selected_root_index"]]
        right_selected = right_roots[right_surface["selected_root_index"]]
        selected_movements.append(
            {
                "surface_index": left_surface["surface_index"],
                "branch_stable": (
                    left_surface["selected_branch_id"]
                    == right_surface["selected_branch_id"]
                ),
                "electric_field_movement_kV_m": abs(
                    right_selected["electric_field_kV_m"]
                    - left_selected["electric_field_kV_m"]
                ),
                "particle_flux_scaled_movement": _scaled_movement(
                    left_selected["particle_flux_m2_s"],
                    right_selected["particle_flux_m2_s"],
                ),
                "heat_flux_scaled_movement": _scaled_movement(
                    left_selected["heat_flux_W_m2"],
                    right_selected["heat_flux_W_m2"],
                ),
                "parallel_current_scaled_movement": _scaled_movement(
                    left_selected["parallel_current_A_T_m2"],
                    right_selected["parallel_current_A_T_m2"],
                ),
            }
        )
    topology_stable &= all(row["branch_stable"] for row in selected_movements)
    summary = {
        "topology_stable": topology_stable,
        "max_all_root_electric_field_movement_kV_m": max(
            row["electric_field_movement_kV_m"] for row in root_movements
        ),
        "max_all_root_particle_flux_scaled_movement": max(
            row["particle_flux_scaled_movement"] for row in root_movements
        ),
        "max_all_root_heat_flux_scaled_movement": max(
            row["heat_flux_scaled_movement"] for row in root_movements
        ),
        "max_all_root_parallel_current_scaled_movement": max(
            row["parallel_current_scaled_movement"] for row in root_movements
        ),
        "max_selected_electric_field_movement_kV_m": max(
            row["electric_field_movement_kV_m"] for row in selected_movements
        ),
        "max_selected_particle_flux_scaled_movement": max(
            row["particle_flux_scaled_movement"] for row in selected_movements
        ),
        "max_selected_heat_flux_scaled_movement": max(
            row["heat_flux_scaled_movement"] for row in selected_movements
        ),
        "max_selected_parallel_current_scaled_movement": max(
            row["parallel_current_scaled_movement"] for row in selected_movements
        ),
    }
    return {
        "left": left["resolution"],
        "right": right["resolution"],
        "root_movements": root_movements,
        "selected_movements": selected_movements,
        "summary": summary,
    }


def _admission(
    rungs: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    latest = comparisons[-1]["summary"]
    max_residual = max(
        rung["attempts"]["maximum_accepted_true_residual"] for rung in rungs
    )
    gates = {
        "topology_stable": latest["topology_stable"],
        "all_root_electric_field_movement": (
            latest["max_all_root_electric_field_movement_kV_m"] <= 0.005
        ),
        "selected_particle_flux_movement": (
            latest["max_selected_particle_flux_scaled_movement"] <= 0.02
        ),
        "selected_heat_flux_movement": (
            latest["max_selected_heat_flux_scaled_movement"] <= 0.02
        ),
        "accepted_true_residual": max_residual <= 1.0e-12,
    }
    return {
        "status": "resolved" if all(gates.values()) else "refinement_exhausted",
        "admission_pass": all(gates.values()),
        "gates": gates,
        "limits": {
            "max_all_root_electric_field_movement_kV_m": 0.005,
            "max_selected_particle_flux_scaled_movement": 0.02,
            "max_selected_heat_flux_scaled_movement": 0.02,
            "max_accepted_true_residual": 1.0e-12,
        },
        "measured_max_accepted_true_residual": max_residual,
        "failed_gates": [name for name, passed in gates.items() if not passed],
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
    results = {name: dkx.Result.load(path) for name, path in result_paths.items()}
    rungs = [
        _compact_rung(results[name], resolutions[name])
        for name in ("coarse", "reference", "fine")
    ]
    comparisons = [
        _compare_rungs(rungs[0], rungs[1]),
        _compare_rungs(rungs[1], rungs[2]),
    ]
    compact = {"rungs": rungs, "comparisons": comparisons}
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    first = results["reference"]
    return {
        "schema": "dkx.ambipolar_phase_space_ladder.v1",
        "created_utc": "2026-08-29T12:11:46Z",
        "claim_scope": "bounded_w7x_pas_dkes_phase_space_nonconvergence_evidence",
        "summary": (
            "A bounded coarse/reference/fine kinetic-grid ladder preserves the "
            "all-root topology but exhausts the configured root and selected-flux "
            "movement gates; it is evidence against promotion, not a convergence "
            "certificate."
        ),
        "source": {
            "dkx_commit": commit,
            "dkx_version": first.metadata["dkx_version"],
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
                    "sha256": _sha256(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in result_paths.items()
            },
            "compact_ladder_sha256": _canonical_sha256(compact),
        },
        "environment": {
            "host": "Mac mini, Apple M4, 10 CPU cores, 24 GB RAM",
            "os": first.metadata["platform"],
            "python": first.metadata["python_version"],
            "jax": first.metadata["jax_version"],
            "jaxlib": first.metadata["jaxlib_version"],
            "numpy": np.__version__,
            "dkx_cores": 8,
            "timing_note": (
                "Each rung used a fresh process and initially empty matching "
                "external compilation cache. Timings are provenance, not a "
                "performance comparison."
            ),
        },
        "measurements": MEASUREMENTS,
        **compact,
        "admission": _admission(rungs, comparisons),
        "exclusions": [
            "not_phase_space_convergence_validation",
            "not_independent_cross_code_ambipolar_validation",
            "not_full_fokker_planck_ambipolar_validation",
            "not_experimental_validation",
            "not_phi1_validation",
            "not_cross_code_performance_validation",
            "fine_rung_does_not_refine_zeta_or_speed_beyond_reference",
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
    if _canonical_sha256(compact) != source["compact_ladder_sha256"]:
        errors.append("compact ladder checksum mismatch")
    recomputed_comparisons = [
        _compare_rungs(payload["rungs"][0], payload["rungs"][1]),
        _compare_rungs(payload["rungs"][1], payload["rungs"][2]),
    ]
    if recomputed_comparisons != payload["comparisons"]:
        errors.append("stored comparison arithmetic mismatch")
    measured_admission = _admission(payload["rungs"], recomputed_comparisons)
    if measured_admission != payload["admission"]:
        errors.append("stored admission status disagrees with recomputed gates")

    if result_paths is not None:
        import dkx  # noqa: PLC0415

        for index, name in enumerate(("coarse", "reference", "fine")):
            path = result_paths[name]
            record = source["results"][name]
            if _sha256(path) != record["sha256"]:
                errors.append(f"external {name} result checksum mismatch")
                continue
            if path.stat().st_size != record["bytes"]:
                errors.append(f"external {name} result size mismatch")
            resolution = payload["rungs"][index]["resolution"]
            if (
                _compact_rung(dkx.Result.load(path), resolution)
                != payload["rungs"][index]
            ):
                errors.append(f"external {name} compact rung mismatch")
        if geometry is None:
            errors.append("external geometry path missing")
        elif _sha256(geometry) != source["geometry_sha256"]:
            errors.append("external geometry checksum mismatch")
        elif geometry.stat().st_size != source["geometry_bytes"]:
            errors.append("external geometry size mismatch")

    return {
        "schema": "dkx.ambipolar_phase_space_ladder.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": result_paths is not None,
        "admission_status": measured_admission["status"],
        "admission_pass": measured_admission["admission_pass"],
        "failed_scientific_gates": measured_admission["failed_gates"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--coarse-result", type=Path)
    parser.add_argument("--reference-result", type=Path)
    parser.add_argument("--fine-result", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    provided = (args.coarse_result, args.reference_result, args.fine_result)
    if args.build:
        if any(path is None for path in (*provided, args.geometry)):
            parser.error("--build requires all three results and --geometry")
        payload = build_artifact(
            {
                "coarse": args.coarse_result,
                "reference": args.reference_result,
                "fine": args.fine_result,
            },
            args.geometry,
        )
        args.artifact.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {"artifact": str(args.artifact), "admission": payload["admission"]},
                indent=2,
            )
        )
        return 0
    result_paths = None
    if any(path is not None for path in provided):
        if any(path is None for path in provided):
            parser.error("external audit requires all three result paths")
        result_paths = {
            "coarse": args.coarse_result,
            "reference": args.reference_result,
            "fine": args.fine_result,
        }
    report = audit(args.artifact, result_paths=result_paths, geometry=args.geometry)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
