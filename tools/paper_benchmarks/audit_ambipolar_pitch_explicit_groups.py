#!/usr/bin/env python3
"""Build or audit the fixed-high-work explicit pitch allocation diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from audit_ambipolar_phase_space_ladder import (
    _canonical_sha256,
    _compact_rung,
    _sha256,
)
from audit_ambipolar_pitch_budget import _compare
from audit_ambipolar_pitch_speed_groups import _scientific_arrays_equal

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_pitch_explicit_groups_v1.json"
NAMES = ("linear36", "low_heavy", "intermediate_heavy")
INPUTS = {
    "linear36": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_ramp36_triplet.toml",
    "low_heavy": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_pitch_low_heavy_pair_bounded.toml",
    "intermediate_heavy": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_pitch_intermediate_heavy_pair_bounded.toml",
}
RESOLUTIONS = {
    "linear36": {
        "theta": 15,
        "zeta": 37,
        "pitch": 36,
        "speed": 6,
        "pitch_speed_ramp": 1,
    },
    "low_heavy": {
        "theta": 15,
        "zeta": 37,
        "pitch": 36,
        "speed": 6,
        "pitch_modes_by_speed": [12, 12, 16, 17, 36, 36],
    },
    "intermediate_heavy": {
        "theta": 15,
        "zeta": 37,
        "pitch": 36,
        "speed": 6,
        "pitch_modes_by_speed": [4, 4, 24, 25, 36, 36],
    },
}
MEASUREMENTS = {
    "linear36": {
        "surface_count": 3,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 184.37,
        "dkx_total_s": 184.3672281250001,
        "maximum_rss_bytes": 4_556_554_240,
        "peak_footprint_bytes": 4_130_181_920,
    },
    "low_heavy_cold": {
        "surface_count": 2,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 148.25,
        "dkx_total_s": 146.4549990000014,
        "maximum_rss_bytes": 4_076_601_344,
        "peak_footprint_bytes": 3_666_596_184,
    },
    "intermediate_heavy_cold": {
        "surface_count": 2,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 149.17,
        "dkx_total_s": 147.2118450420021,
        "maximum_rss_bytes": 4_020_191_232,
        "peak_footprint_bytes": 3_570_897_288,
    },
    "intermediate_heavy_warm": {
        "surface_count": 2,
        "cache_state": "fresh_process_with_populated_external_cache",
        "process_wall_s": 148.36,
        "dkx_total_s": 146.38495766700362,
        "maximum_rss_bytes": 3_375_595_520,
        "peak_footprint_bytes": 4_001_649_080,
    },
}


def _allocation(result: Any) -> dict[str, Any]:
    phase_space = result.metadata["phase_space"]
    modes = [int(value) for value in phase_space["active_pitch_modes_by_speed"]]
    if len(modes) != 6:
        raise ValueError(f"expected six speed nodes, got {len(modes)}")
    return {
        "source": str(phase_space.get("pitch_allocation_source", "pitch_speed_ramp")),
        "pitch_speed_ramp": phase_space["pitch_speed_ramp"],
        "active_pitch_modes_by_speed": modes,
        "active_pitch_mode_sum": int(phase_space["active_pitch_mode_sum"]),
        "groups": {
            "low_speed_nodes_0_1": sum(modes[0:2]),
            "intermediate_speed_nodes_2_3": sum(modes[2:4]),
            "high_speed_nodes_4_5": sum(modes[4:6]),
        },
    }


def _compact_pair(result: Any, name: str) -> dict[str, Any]:
    compact = _compact_rung(result, RESOLUTIONS[name])
    compact["surfaces"] = compact["surfaces"][:2]
    compact["allocation"] = _allocation(result)
    return compact


def _outcome(
    rungs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    warm_parity: bool,
    measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    totals = [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs]
    high_groups = [
        rung["allocation"]["groups"]["high_speed_nodes_4_5"] for rung in rungs
    ]
    topologies = [
        [surface["root_count"] for surface in rung["surfaces"]] for rung in rungs
    ]
    residual_pass = (
        max(rung["attempts"]["maximum_accepted_true_residual"] for rung in rungs)
        <= 1.0e-12
    )
    bounded_memory = (
        max(row["peak_footprint_bytes"] for row in measurements.values()) < 24 * 2**30
    )
    maximum_movements = {
        "electric_field_kV_m": max(
            row["summary"]["max_selected_electric_field_movement_kV_m"]
            for row in comparisons
        ),
        "particle_flux_scaled": max(
            row["summary"]["max_selected_particle_flux_scaled_movement"]
            for row in comparisons
        ),
        "heat_flux_scaled": max(
            row["summary"]["max_selected_heat_flux_scaled_movement"]
            for row in comparisons
        ),
    }
    gates = {
        "exact_total_active_work_129": totals == [129, 129, 129],
        "exact_high_speed_group_work_72": high_groups == [72, 72, 72],
        "topology_stable_1_3": topologies == [[1, 3], [1, 3], [1, 3]],
        "maximum_true_residual_below_1e-12": residual_pass,
        "all_process_footprints_below_24_gib": bounded_memory,
        "intermediate_cold_warm_scientific_arrays_exact": warm_parity,
        "allocation_sensitivity_observed": (
            maximum_movements["electric_field_kV_m"] > 0.005
            or maximum_movements["particle_flux_scaled"] > 0.02
            or maximum_movements["heat_flux_scaled"] > 0.02
        ),
    }
    return {
        "status": "refinement_exhausted",
        "admission_pass": all(gates.values()),
        "gates": gates,
        "maximum_selected_movements": maximum_movements,
        "phase_space_converged": False,
        "interpretation": (
            "Holding total and high-speed pitch work exactly fixed preserves "
            "root topology, but low/intermediate redistribution still fails "
            "root and selected-flux movement gates."
        ),
        "next_action": (
            "Use the bounded pair to raise low and intermediate pitch work "
            "together while holding the admitted high-speed group fixed; do not "
            "promote any current allocation as phase-space converged."
        ),
    }


def build_artifact(
    result_paths: dict[str, Path], warm_result: Path, geometry: Path
) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    results = {name: dkx.Result.load(path) for name, path in result_paths.items()}
    warm = dkx.Result.load(warm_result)
    rungs = [_compact_pair(results[name], name) for name in NAMES]
    comparisons = [
        _compare(rungs[0], rungs[1], "linear36_to_low_heavy"),
        _compare(rungs[0], rungs[2], "linear36_to_intermediate_heavy"),
        _compare(rungs[1], rungs[2], "low_heavy_to_intermediate_heavy"),
    ]
    warm_equal, warm_mismatches = _scientific_arrays_equal(
        results["intermediate_heavy"], warm
    )
    parity = {
        "scientific_arrays_exact": warm_equal,
        "ignored_timing_arrays": ["solve_time_s"],
        "mismatches": warm_mismatches,
    }
    compact = {
        "measurements": MEASUREMENTS,
        "rungs": rungs,
        "comparisons": comparisons,
        "intermediate_cold_warm_parity": parity,
    }
    reference = results["intermediate_heavy"]
    return {
        "schema": "dkx.ambipolar_pitch_explicit_groups.v1",
        "created_utc": "2026-08-29T18:12:00Z",
        "claim_scope": "fixed_high_work_low_intermediate_pitch_diagnosis",
        "summary": (
            "Three 129-mode allocations hold the high-speed group at exactly "
            "72 modes. Root topology stays [1,3], but roots and selected fluxes "
            "remain sensitive to low/intermediate redistribution."
        ),
        "source": {
            "dkx_commit": "0a793931eccb8dc8c4fd488fe04d642e08fe0cc8",
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
                **{
                    name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                    for name, path in result_paths.items()
                },
                "intermediate_heavy_warm": {
                    "sha256": _sha256(warm_result),
                    "bytes": warm_result.stat().st_size,
                },
            },
            "compact_sha256": _canonical_sha256(compact),
        },
        "environment": {
            "host": "Mac mini, Apple M4, 10 CPU cores, 24 GiB RAM",
            "os": reference.metadata["platform"],
            "python": reference.metadata["python_version"],
            "jax": reference.metadata["jax_version"],
            "jaxlib": reference.metadata["jaxlib_version"],
            "numpy": np.__version__,
            "timing_note": (
                "The retained linear result has three surfaces; the two explicit "
                "results contain the minimum valid pair. Runtime is provenance, "
                "not a cross-allocation speed comparison."
            ),
        },
        **compact,
        "outcome": _outcome(rungs, comparisons, warm_equal, MEASUREMENTS),
        "exclusions": [
            "not_phase_space_convergence_validation",
            "not_cross_allocation_runtime_comparison",
            "not_speed_or_zeta_convergence_validation",
            "not_independent_cross_code_ambipolar_validation",
            "not_full_fokker_planck_or_phi1_validation",
            "not_experimental_validation",
        ],
    }


def audit(
    artifact: Path,
    *,
    result_paths: dict[str, Path] | None = None,
    warm_result: Path | None = None,
    geometry: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = payload["source"]
    for record in source["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            errors.append(f"input checksum mismatch: {record['path']}")
    compact = {
        "measurements": payload["measurements"],
        "rungs": payload["rungs"],
        "comparisons": payload["comparisons"],
        "intermediate_cold_warm_parity": payload["intermediate_cold_warm_parity"],
    }
    if _canonical_sha256(compact) != source["compact_sha256"]:
        errors.append("compact checksum mismatch")
    recomputed = [
        _compare(payload["rungs"][0], payload["rungs"][1], "linear36_to_low_heavy"),
        _compare(
            payload["rungs"][0],
            payload["rungs"][2],
            "linear36_to_intermediate_heavy",
        ),
        _compare(
            payload["rungs"][1],
            payload["rungs"][2],
            "low_heavy_to_intermediate_heavy",
        ),
    ]
    if recomputed != payload["comparisons"]:
        errors.append("stored comparison arithmetic mismatch")
    outcome = _outcome(
        payload["rungs"],
        recomputed,
        payload["intermediate_cold_warm_parity"]["scientific_arrays_exact"],
        payload["measurements"],
    )
    if outcome != payload["outcome"]:
        errors.append("stored outcome disagrees with recomputed gates")

    external_verified = result_paths is not None
    if external_verified:
        import dkx  # noqa: PLC0415

        assert result_paths is not None
        for index, name in enumerate(NAMES):
            path = result_paths[name]
            record = source["results"][name]
            if _sha256(path) != record["sha256"]:
                errors.append(f"external {name} result checksum mismatch")
                continue
            if _compact_pair(dkx.Result.load(path), name) != payload["rungs"][index]:
                errors.append(f"external {name} compact rung mismatch")
        if warm_result is None:
            errors.append("external intermediate warm result missing")
        else:
            record = source["results"]["intermediate_heavy_warm"]
            if _sha256(warm_result) != record["sha256"]:
                errors.append("external intermediate warm result checksum mismatch")
            else:
                equal, mismatches = _scientific_arrays_equal(
                    dkx.Result.load(result_paths["intermediate_heavy"]),
                    dkx.Result.load(warm_result),
                )
                if {
                    "scientific_arrays_exact": equal,
                    "ignored_timing_arrays": ["solve_time_s"],
                    "mismatches": mismatches,
                } != payload["intermediate_cold_warm_parity"]:
                    errors.append("external intermediate cold/warm parity mismatch")
        if geometry is None or _sha256(geometry) != source["geometry_sha256"]:
            errors.append("external geometry checksum mismatch")

    return {
        "schema": "dkx.ambipolar_pitch_explicit_groups.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": external_verified,
        "status": outcome["status"],
        "admission_pass": outcome["admission_pass"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--build", action="store_true")
    for name in NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}-result", type=Path)
    parser.add_argument("--intermediate-heavy-warm-result", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    result_paths = {name: getattr(args, f"{name}_result") for name in NAMES}
    provided = [path is not None for path in result_paths.values()]
    if args.build:
        if (
            not all(provided)
            or args.intermediate_heavy_warm_result is None
            or args.geometry is None
        ):
            parser.error("--build requires all four results and --geometry")
        payload = build_artifact(
            result_paths, args.intermediate_heavy_warm_result, args.geometry
        )
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
            parser.error("external audit requires all three allocation results")
        external = result_paths
    report = audit(
        args.artifact,
        result_paths=external,
        warm_result=args.intermediate_heavy_warm_result,
        geometry=args.geometry,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
