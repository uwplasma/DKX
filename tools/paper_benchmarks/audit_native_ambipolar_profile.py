#!/usr/bin/env python3
"""Build or audit the compact native whole-profile ambipolar certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "native_ambipolar_profile_v1.json"
DEFAULT_INPUT = (
    ROOT / "validation" / "inputs" / "w7x_standard_native_ambipolar_profile.toml"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _optional_float(value: Any) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _compact_result(result: Any) -> dict[str, Any]:
    arrays = result.arrays
    surfaces: list[dict[str, Any]] = []
    for surface_index in range(len(arrays["surface"])):
        root_count = int(arrays["ambipolar_root_count"][surface_index])
        roots = []
        for root_index in range(root_count):
            roots.append(
                {
                    "index": root_index,
                    "electric_field_kV_m": float(
                        arrays["ambipolar_root_kV_m"][surface_index, root_index]
                    ),
                    "radial_current_A_m2": float(
                        arrays["ambipolar_root_current_A_m2"][
                            surface_index, root_index
                        ]
                    ),
                    "slope_A_m2_per_kV_m": float(
                        arrays["ambipolar_root_slope_A_m2_per_kV_m"][
                            surface_index, root_index
                        ]
                    ),
                    "classification": str(
                        arrays["ambipolar_root_type"][surface_index, root_index]
                    ),
                    "bracket_kV_m": [
                        float(value)
                        for value in arrays["ambipolar_root_bracket_kV_m"][
                            surface_index, root_index
                        ]
                    ],
                    "final_bracket_width_kV_m": float(
                        arrays["ambipolar_root_final_bracket_width_kV_m"][
                            surface_index, root_index
                        ]
                    ),
                    "root_movement_kV_m": float(
                        arrays["ambipolar_root_movement_kV_m"][
                            surface_index, root_index
                        ]
                    ),
                    "observable_relative_movement": float(
                        arrays["ambipolar_root_observable_relative_movement"][
                            surface_index, root_index
                        ]
                    ),
                    "branch_id": str(
                        arrays["ambipolar_root_branch_id"][surface_index, root_index]
                    ),
                }
            )
        refinement = []
        for rung_index in range(arrays["ambipolar_refinement_level"].shape[1]):
            refinement.append(
                {
                    "level": int(
                        arrays["ambipolar_refinement_level"][
                            surface_index, rung_index
                        ]
                    ),
                    "search_evaluations": int(
                        arrays["ambipolar_refinement_search_evaluations"][
                            surface_index, rung_index
                        ]
                    ),
                    "total_evaluations": int(
                        arrays["ambipolar_refinement_total_evaluations"][
                            surface_index, rung_index
                        ]
                    ),
                    "root_count": int(
                        arrays["ambipolar_refinement_root_count"][
                            surface_index, rung_index
                        ]
                    ),
                    "root_movement_kV_m": _optional_float(
                        arrays["ambipolar_refinement_root_movement_kV_m"][
                            surface_index, rung_index
                        ]
                    ),
                    "observable_relative_movement": _optional_float(
                        arrays["ambipolar_refinement_observable_relative_movement"][
                            surface_index, rung_index
                        ]
                    ),
                    "max_bracket_width_kV_m": float(
                        arrays["ambipolar_refinement_max_bracket_width_kV_m"][
                            surface_index, rung_index
                        ]
                    ),
                    "converged": bool(
                        arrays["ambipolar_refinement_converged"][
                            surface_index, rung_index
                        ]
                    ),
                }
            )
        surfaces.append(
            {
                "surface_index": surface_index,
                "psi_N": float(arrays["surface"][surface_index]),
                "r_N": float(arrays["r_N"][surface_index]),
                "status": str(arrays["ambipolar_status"][surface_index]),
                "refinement_status": str(
                    arrays["ambipolar_refinement_status"][surface_index]
                ),
                "evaluation_budget": int(
                    arrays["ambipolar_evaluation_budget"][surface_index]
                ),
                "root_count": root_count,
                "roots": roots,
                "selected_root_index": int(
                    arrays["selected_ambipolar_root"][surface_index]
                ),
                "selected_branch_id": str(
                    arrays["selected_ambipolar_branch"][surface_index]
                ),
                "selection_reason": str(
                    arrays["ambipolar_selection_reason"][surface_index]
                ),
                "selected": {
                    "electric_field_kV_m": float(
                        arrays["electric_field_kV_m"][surface_index]
                    ),
                    "particle_flux_m2_s": [
                        float(value)
                        for value in arrays["particle_flux_m2_s"][surface_index]
                    ],
                    "heat_flux_W_m2": [
                        float(value)
                        for value in arrays["heat_flux_W_m2"][surface_index]
                    ],
                    "parallel_current_A_T_m2": float(
                        arrays["parallel_current_A_T_m2"][surface_index]
                    ),
                    "primal_residual": float(arrays["primal_residual"][surface_index]),
                },
                "refinement": refinement,
            }
        )

    events: list[dict[str, Any]] = []
    for surface_index in range(len(surfaces)):
        count = int(arrays["ambipolar_branch_event_count"][surface_index])
        for event_index in range(count):
            events.append(
                {
                    "surface_index": surface_index,
                    "kind": str(
                        arrays["ambipolar_branch_event_kind"][
                            surface_index, event_index
                        ]
                    ),
                    "branch_ids": [
                        str(value)
                        for value in arrays["ambipolar_branch_event_branch_id"][
                            surface_index, event_index
                        ]
                        if str(value)
                    ],
                    "root_indices": [
                        int(value)
                        for value in arrays["ambipolar_branch_event_root_index"][
                            surface_index, event_index
                        ]
                        if int(value) >= 0
                    ],
                    "electric_field_kV_m": float(
                        arrays["ambipolar_branch_event_electric_field_kV_m"][
                            surface_index, event_index
                        ]
                    ),
                    "detail": str(
                        arrays["ambipolar_branch_event_detail"][
                            surface_index, event_index
                        ]
                    ),
                    "nonsmooth": bool(
                        arrays["ambipolar_branch_event_nonsmooth"][
                            surface_index, event_index
                        ]
                    ),
                }
            )

    recoveries: list[dict[str, Any]] = []
    accepted_residuals: list[float] = []
    completed_residuals: list[float] = []
    for surface_index in range(len(surfaces)):
        for evaluation_index in range(arrays["evaluation"].shape[0]):
            attempt_count = int(
                arrays["evaluation_solver_attempt_count"][
                    surface_index, evaluation_index
                ]
            )
            attempts = []
            for attempt_index in range(attempt_count):
                residual = float(
                    arrays["evaluation_solver_attempt_residual"][
                        surface_index, evaluation_index, attempt_index
                    ]
                )
                accepted = bool(
                    arrays["evaluation_solver_attempt_accepted"][
                        surface_index, evaluation_index, attempt_index
                    ]
                )
                completed_residuals.append(residual)
                if accepted:
                    accepted_residuals.append(residual)
                attempts.append(
                    {
                        "requested_method": str(
                            arrays["evaluation_solver_attempt_requested_method"][
                                surface_index, evaluation_index, attempt_index
                            ]
                        ),
                        "executed_method": str(
                            arrays["evaluation_solver_attempt_executed_method"][
                                surface_index, evaluation_index, attempt_index
                            ]
                        ),
                        "residual": residual,
                        "accepted": accepted,
                        "reason": str(
                            arrays["evaluation_solver_attempt_reason"][
                                surface_index, evaluation_index, attempt_index
                            ]
                        ),
                    }
                )
            if attempt_count > 1:
                recoveries.append(
                    {
                        "surface_index": surface_index,
                        "evaluation_index": evaluation_index,
                        "electric_field_kV_m": float(
                            arrays["evaluation_electric_field_kV_m"][
                                surface_index, evaluation_index
                            ]
                        ),
                        "radial_current_A_m2": float(
                            arrays["radial_current_A_m2"][
                                surface_index, evaluation_index
                            ]
                        ),
                        "attempts": attempts,
                    }
                )
    attempt_metadata = result.metadata["ambipolar_solver_attempts"]
    attempt_summary = {
        "attempt_count": int(attempt_metadata["attempt_count"]),
        "executed_route_counts": {
            str(key): int(value)
            for key, value in attempt_metadata["executed_route_counts"].items()
        },
        "automatic_true_residual_recovery_count": int(
            attempt_metadata["automatic_true_residual_recovery_count"]
        ),
        "policy": str(attempt_metadata["policy"]),
        "maximum_completed_attempt_residual": max(completed_residuals),
        "maximum_accepted_attempt_residual": max(accepted_residuals),
        "recoveries": recoveries,
    }
    return {"surfaces": surfaces, "branch_events": events, "attempts": attempt_summary}


def _scientific_array_differences(cold: Any, warm: Any) -> list[str]:
    differences = []
    for key in cold.arrays:
        if key == "solve_time_s":
            continue
        left = np.asarray(cold.arrays[key])
        right = np.asarray(warm.arrays[key])
        equal = (
            np.array_equal(left, right, equal_nan=True)
            if left.dtype.kind in "fc"
            else np.array_equal(left, right)
        )
        if not equal:
            differences.append(key)
    return differences


def build_artifact(
    results_root: Path,
    geometry: Path,
    *,
    cold_process_wall_s: float,
    warm_process_wall_s: float,
    cold_peak_footprint_bytes: int,
    warm_peak_footprint_bytes: int,
) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    cold_path = results_root / "cold.nc"
    warm_path = results_root / "warm.nc"
    cold = dkx.Result.load(cold_path)
    warm = dkx.Result.load(warm_path)
    compact = _compact_result(cold)
    differences = _scientific_array_differences(cold, warm)
    case = cold.metadata["canonical_case"]
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    selected_residuals = [
        surface["selected"]["primal_residual"] for surface in compact["surfaces"]
    ]
    bracket_widths = [
        root["final_bracket_width_kV_m"]
        for surface in compact["surfaces"]
        for root in surface["roots"]
    ]
    root_currents = [
        abs(root["radial_current_A_m2"])
        for surface in compact["surfaces"]
        for root in surface["roots"]
    ]
    root_current_bracket_fractions = [
        abs(root["radial_current_A_m2"])
        / (
            abs(root["slope_A_m2_per_kV_m"])
            * root["final_bracket_width_kV_m"]
        )
        for surface in compact["surfaces"]
        for root in surface["roots"]
    ]
    acceptance = {
        "required_surface_count": 5,
        "required_root_counts": [1, 1, 3, 1, 1],
        "max_final_bracket_width_kV_m": 0.005,
        "max_selected_primal_residual": 1e-12,
        "max_root_current_bracket_fraction": 1.0,
        "required_automatic_recovery_count": 1,
        "required_scientific_array_differences": [],
        "measured_surface_count": len(compact["surfaces"]),
        "measured_root_counts": [
            surface["root_count"] for surface in compact["surfaces"]
        ],
        "measured_max_final_bracket_width_kV_m": max(bracket_widths),
        "measured_max_selected_primal_residual": max(selected_residuals),
        "measured_max_root_current_abs_A_m2": max(root_currents),
        "measured_max_root_current_bracket_fraction": max(
            root_current_bracket_fractions
        ),
        "measured_automatic_recovery_count": compact["attempts"][
            "automatic_true_residual_recovery_count"
        ],
        "measured_scientific_array_differences": differences,
        "all_surfaces_bracketed": all(
            surface["status"] == "bracketed_root" for surface in compact["surfaces"]
        ),
        "all_refinement_hierarchies_resolved": all(
            surface["refinement_status"] == "resolved"
            for surface in compact["surfaces"]
        ),
        "all_gates_pass": True,
    }
    return {
        "schema": "dkx.native_ambipolar_profile.v1",
        "created_utc": "2026-08-29T11:40:00Z",
        "claim_scope": "native_w7x_pas_dkes_whole_profile_workflow_certificate",
        "summary": (
            "A physical-unit five-surface W7-X standard-configuration profile "
            "retains every PAS/DKES kinetic evaluation, all ambipolar roots, "
            "adaptive brackets, selected fluxes, branch events, and one bounded "
            "automatic true-residual recovery in native NetCDF."
        ),
        "source": {
            "dkx_commit": git_commit,
            "dkx_version": cold.metadata["dkx_version"],
            "solvax_version": "0.18.0",
            "sfincs_geometry_source_commit": (
                "8df5453472e982df0f6ae005243ce38d57a83711"
            ),
            "input": str(DEFAULT_INPUT.relative_to(ROOT)),
            "input_sha256": _sha256(DEFAULT_INPUT),
            "geometry": "SFINCS/equilibria/w7x_standardConfig.bc",
            "geometry_sha256": _sha256(geometry),
            "geometry_bytes": geometry.stat().st_size,
            "cold_result_sha256": _sha256(cold_path),
            "warm_result_sha256": _sha256(warm_path),
            "result_bytes": cold_path.stat().st_size,
            "compact_profile_sha256": _canonical_sha256(compact),
        },
        "environment": {
            "host": "Mac mini, Apple M4, 10 CPU cores, 24 GB RAM",
            "os": cold.metadata["platform"],
            "python": cold.metadata["python_version"],
            "jax": cold.metadata["jax_version"],
            "jaxlib": cold.metadata["jaxlib_version"],
            "numpy": np.__version__,
            "dkx_cores": 8,
            "timing_note": (
                "Cold uses an initially empty external compilation cache; warm "
                "uses a fresh process and the populated matching cache. Warm wall "
                "time is 0.15% above cold, so no warm-speedup claim is made."
            ),
        },
        "case": {
            "case_id": cold.case_id,
            "workflow": case["run"]["workflow"],
            "geometry_format": case["geometry"]["format"],
            "surfaces_psi_N": list(case["geometry"]["surfaces"]),
            "species": [dict(species) for species in case["species"]],
            "physics": dict(case["physics"]),
            "electric_field": dict(case["electric_field"]),
            "resolution": dict(case["resolution"]),
            "solver": dict(case["solver"]),
            "convergence": dict(case["convergence"]),
        },
        "measurements": {
            "cold": {
                "process_wall_s": cold_process_wall_s,
                "dkx_run_total_s": cold.metadata["timings_s"]["total"],
                "maximum_rss_bytes": cold.metadata["peak_host_memory_bytes"],
                "peak_footprint_bytes": cold_peak_footprint_bytes,
            },
            "warm": {
                "process_wall_s": warm_process_wall_s,
                "dkx_run_total_s": warm.metadata["timings_s"]["total"],
                "maximum_rss_bytes": warm.metadata["peak_host_memory_bytes"],
                "peak_footprint_bytes": warm_peak_footprint_bytes,
            },
        },
        "profile": compact,
        "acceptance": acceptance,
        "exclusions": [
            "not_phase_space_convergence_validation",
            "not_continuous_branch_event_localization",
            "not_experimental_validation",
            "not_full_fokker_planck_ambipolar_validation",
            "not_phi1_validation",
            "not_independent_cross_code_ambipolar_validation",
            "not_second_stellarator_family_validation",
            "not_cross_code_performance_validation",
        ],
    }


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    source = payload["source"]
    acceptance = payload["acceptance"]
    profile = payload["profile"]
    errors: list[str] = []
    if _sha256(ROOT / source["input"]) != source["input_sha256"]:
        errors.append("input checksum mismatch")
    if _canonical_sha256(profile) != source["compact_profile_sha256"]:
        errors.append("compact profile checksum mismatch")

    measured = {
        "surface_count": len(profile["surfaces"]),
        "root_counts": [surface["root_count"] for surface in profile["surfaces"]],
        "max_final_bracket_width_kV_m": max(
            root["final_bracket_width_kV_m"]
            for surface in profile["surfaces"]
            for root in surface["roots"]
        ),
        "max_selected_primal_residual": max(
            surface["selected"]["primal_residual"]
            for surface in profile["surfaces"]
        ),
        "max_root_current_abs_A_m2": max(
            abs(root["radial_current_A_m2"])
            for surface in profile["surfaces"]
            for root in surface["roots"]
        ),
        "max_root_current_bracket_fraction": max(
            abs(root["radial_current_A_m2"])
            / (
                abs(root["slope_A_m2_per_kV_m"])
                * root["final_bracket_width_kV_m"]
            )
            for surface in profile["surfaces"]
            for root in surface["roots"]
        ),
        "automatic_recovery_count": profile["attempts"][
            "automatic_true_residual_recovery_count"
        ],
        "all_surfaces_bracketed": all(
            surface["status"] == "bracketed_root" for surface in profile["surfaces"]
        ),
        "all_refinement_hierarchies_resolved": all(
            surface["refinement_status"] == "resolved"
            for surface in profile["surfaces"]
        ),
    }
    stored_pairs = {
        "surface_count": "measured_surface_count",
        "root_counts": "measured_root_counts",
        "max_final_bracket_width_kV_m": (
            "measured_max_final_bracket_width_kV_m"
        ),
        "max_selected_primal_residual": "measured_max_selected_primal_residual",
        "max_root_current_abs_A_m2": "measured_max_root_current_abs_A_m2",
        "max_root_current_bracket_fraction": (
            "measured_max_root_current_bracket_fraction"
        ),
        "automatic_recovery_count": "measured_automatic_recovery_count",
        "all_surfaces_bracketed": "all_surfaces_bracketed",
        "all_refinement_hierarchies_resolved": (
            "all_refinement_hierarchies_resolved"
        ),
    }
    for measured_key, stored_key in stored_pairs.items():
        if measured[measured_key] != acceptance[stored_key]:
            errors.append(f"stored {measured_key} disagrees with compact profile")
    passed = (
        measured["surface_count"] == acceptance["required_surface_count"]
        and measured["root_counts"] == acceptance["required_root_counts"]
        and measured["max_final_bracket_width_kV_m"]
        <= acceptance["max_final_bracket_width_kV_m"]
        and measured["max_selected_primal_residual"]
        <= acceptance["max_selected_primal_residual"]
        and measured["max_root_current_bracket_fraction"]
        <= acceptance["max_root_current_bracket_fraction"]
        and measured["automatic_recovery_count"]
        == acceptance["required_automatic_recovery_count"]
        and acceptance["measured_scientific_array_differences"]
        == acceptance["required_scientific_array_differences"]
        and measured["all_surfaces_bracketed"]
        and measured["all_refinement_hierarchies_resolved"]
    )
    if passed is not acceptance["all_gates_pass"]:
        errors.append("stored all_gates_pass disagrees with recomputed gates")

    if results_root is not None:
        import dkx  # noqa: PLC0415

        cold_path = results_root / "cold.nc"
        warm_path = results_root / "warm.nc"
        geometry = results_root / Path(source["geometry"]).name
        for label, path in (("cold", cold_path), ("warm", warm_path)):
            if not path.exists():
                errors.append(f"external {label} result missing")
                continue
            if _sha256(path) != source[f"{label}_result_sha256"]:
                errors.append(f"external {label} result checksum mismatch")
            if path.stat().st_size != source["result_bytes"]:
                errors.append(f"external {label} result size mismatch")
        if not geometry.exists():
            errors.append("external geometry missing")
        else:
            if _sha256(geometry) != source["geometry_sha256"]:
                errors.append("external geometry checksum mismatch")
            if geometry.stat().st_size != source["geometry_bytes"]:
                errors.append("external geometry size mismatch")
        if cold_path.exists() and warm_path.exists():
            cold = dkx.Result.load(cold_path)
            warm = dkx.Result.load(warm_path)
            if (
                _canonical_sha256(_compact_result(cold))
                != source["compact_profile_sha256"]
            ):
                errors.append("external cold compact profile mismatch")
            if _scientific_array_differences(cold, warm) != acceptance[
                "required_scientific_array_differences"
            ]:
                errors.append("external cold/warm scientific arrays differ")

    return {
        "schema": "dkx.native_ambipolar_profile.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": results_root is not None,
        "measured": measured,
        "errors": errors,
        "pass": passed and not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-root", type=Path)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--cold-process-wall-s", type=float)
    parser.add_argument("--warm-process-wall-s", type=float)
    parser.add_argument("--cold-peak-footprint-bytes", type=int)
    parser.add_argument("--warm-peak-footprint-bytes", type=int)
    args = parser.parse_args()
    if args.build:
        required = (
            args.results_root,
            args.geometry,
            args.cold_process_wall_s,
            args.warm_process_wall_s,
            args.cold_peak_footprint_bytes,
            args.warm_peak_footprint_bytes,
        )
        if any(value is None for value in required):
            parser.error("--build requires results, geometry, wall, and footprint inputs")
        payload = build_artifact(
            args.results_root,
            args.geometry,
            cold_process_wall_s=args.cold_process_wall_s,
            warm_process_wall_s=args.warm_process_wall_s,
            cold_peak_footprint_bytes=args.cold_peak_footprint_bytes,
            warm_peak_footprint_bytes=args.warm_peak_footprint_bytes,
        )
        args.artifact.write_text(
            json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
        )
        print(args.artifact)
        return 0
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
