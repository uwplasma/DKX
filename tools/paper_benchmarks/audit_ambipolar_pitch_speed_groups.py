#!/usr/bin/env python3
"""Build or audit the fixed-work pitch-by-speed allocation diagnostic."""

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
    _sha256,
)
from audit_ambipolar_pitch_budget import _compare

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_pitch_speed_groups_v1.json"
NAMES = ("uniform22", "linear36", "quadratic44")
INPUTS = {
    "uniform22": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_uniform22_bounded.toml",
    "linear36": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_ramp36_triplet.toml",
    "quadratic44": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_quadratic44_pair_bounded.toml",
}
MEASUREMENTS = {
    "uniform22": {
        "surface_count": 3,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 176.33,
        "dkx_total_s": 175.07368329200108,
        "maximum_rss_bytes": 3_093_299_200,
        "peak_footprint_bytes": 2_830_077_576,
    },
    "linear36": {
        "surface_count": 3,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 184.37,
        "dkx_total_s": 184.3672281250001,
        "maximum_rss_bytes": 4_556_554_240,
        "peak_footprint_bytes": 4_130_181_920,
    },
    "quadratic44_cold": {
        "surface_count": 2,
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 140.74,
        "dkx_total_s": 139.13717291699868,
        "maximum_rss_bytes": 2_852_945_920,
        "peak_footprint_bytes": 2_398_850_384,
        "reporting_wrapper_exit": 1,
        "reporting_wrapper_note": (
            "The result was saved after a successful solve; a post-save reporting "
            "lookup used timings instead of timings_s and raised KeyError."
        ),
    },
    "quadratic44_warm": {
        "surface_count": 2,
        "cache_state": "fresh_process_with_populated_external_cache",
        "process_wall_s": 148.75,
        "dkx_total_s": 146.8235591249977,
        "maximum_rss_bytes": 2_946_809_856,
        "peak_footprint_bytes": 2_745_945_472,
        "reporting_wrapper_exit": 0,
    },
}


def _resolution(name: str) -> dict[str, int]:
    raw = tomllib.loads(INPUTS[name].read_text(encoding="utf-8"))["resolution"]
    return {str(key): int(value) for key, value in raw.items()}


def _allocation(result: Any) -> dict[str, Any]:
    phase_space = result.metadata["phase_space"]
    modes = [int(value) for value in phase_space["active_pitch_modes_by_speed"]]
    if len(modes) != 6:
        raise ValueError(f"expected six speed nodes, got {len(modes)}")
    return {
        "pitch_speed_ramp": int(phase_space["pitch_speed_ramp"]),
        "active_pitch_modes_by_speed": modes,
        "active_pitch_mode_sum": int(phase_space["active_pitch_mode_sum"]),
        "groups": {
            "low_speed_nodes_0_1": sum(modes[0:2]),
            "intermediate_speed_nodes_2_3": sum(modes[2:4]),
            "high_speed_nodes_4_5": sum(modes[4:6]),
        },
    }


def _compact_pair(result: Any, name: str) -> dict[str, Any]:
    compact = _compact_rung(result, _resolution(name))
    compact["surfaces"] = compact["surfaces"][:2]
    compact["allocation"] = _allocation(result)
    return compact


def _scientific_arrays_equal(left: Any, right: Any) -> tuple[bool, list[str]]:
    ignored = {"solve_time_s"}
    mismatches: list[str] = []
    for key in sorted(set(left.arrays) | set(right.arrays)):
        if key in ignored:
            continue
        if key not in left.arrays or key not in right.arrays:
            mismatches.append(key)
            continue
        a = np.asarray(left.arrays[key])
        b = np.asarray(right.arrays[key])
        if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
            equal = np.array_equal(a, b)
        else:
            equal = np.array_equal(a, b, equal_nan=True)
        if not equal:
            mismatches.append(key)
    return not mismatches, mismatches


def _outcome(
    rungs: list[dict[str, Any]],
    comparisons: list[dict[str, Any]],
    warm_parity: bool,
    measurements: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    totals = [rung["allocation"]["active_pitch_mode_sum"] for rung in rungs]
    fixed_work = max(totals) - min(totals) <= 4
    residual_pass = (
        max(rung["attempts"]["maximum_accepted_true_residual"] for rung in rungs)
        <= 1.0e-12
    )
    bounded_memory = (
        max(row["peak_footprint_bytes"] for row in measurements.values()) < 24 * 2**30
    )
    topology_changes = [
        comparison["name"]
        for comparison in comparisons
        if not comparison["summary"]["topology_stable"]
    ]
    return {
        "status": "diagnostic_complete",
        "admission_pass": fixed_work
        and residual_pass
        and bounded_memory
        and warm_parity
        and bool(topology_changes),
        "gates": {
            "fixed_work_within_four_active_modes": fixed_work,
            "maximum_true_residual_below_1e-12": residual_pass,
            "all_process_footprints_below_24_gib": bounded_memory,
            "quadratic_cold_warm_scientific_arrays_exact": warm_parity,
            "allocation_sensitivity_observed": bool(topology_changes),
        },
        "topology_changing_comparisons": topology_changes,
        "phase_space_converged": False,
        "interpretation": (
            "At nearly fixed active-mode work, moving pitch resolution from low "
            "toward high speed changes root topology on both retained surfaces. "
            "None of the three allocations is admitted as phase-space converged."
        ),
        "next_action": (
            "Introduce a narrowly scoped explicit six-node allocation diagnostic "
            "to separate low from intermediate speed sensitivity while holding "
            "high-speed work fixed; keep the two-surface bounded scope."
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
        _compare(rungs[0], rungs[1], "uniform22_to_linear36"),
        _compare(rungs[1], rungs[2], "linear36_to_quadratic44"),
    ]
    warm_equal, warm_mismatches = _scientific_arrays_equal(results["quadratic44"], warm)
    compact = {
        "rungs": rungs,
        "comparisons": comparisons,
        "quadratic_cold_warm_parity": {
            "scientific_arrays_exact": warm_equal,
            "ignored_timing_arrays": ["solve_time_s"],
            "mismatches": warm_mismatches,
        },
    }
    reference = results["quadratic44"]
    return {
        "schema": "dkx.ambipolar_pitch_speed_groups.v1",
        "created_utc": "2026-08-29T17:36:00Z",
        "claim_scope": "fixed_work_speed_local_pitch_allocation_diagnosis",
        "summary": (
            "Three supported pitch-by-speed rules retain 129, 132, and 133 "
            "active modes but produce different ambipolar root topology on a "
            "common two-surface W7-X profile pair."
        ),
        "source": {
            "dkx_commit": "6c35aa1c482a2b4dc1e914052dd836852d29ef43",
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
                "quadratic44_warm": {
                    "sha256": _sha256(warm_result),
                    "bytes": warm_result.stat().st_size,
                },
            },
            "compact_sha256": _canonical_sha256(
                {"measurements": MEASUREMENTS, **compact}
            ),
        },
        "environment": {
            "host": "Mac mini, Apple M4, 10 CPU cores, 24 GiB RAM",
            "os": reference.metadata["platform"],
            "python": reference.metadata["python_version"],
            "jax": reference.metadata["jax_version"],
            "jaxlib": reference.metadata["jaxlib_version"],
            "numpy": np.__version__,
            "timing_note": (
                "The older uniform/linear measurements contain three surfaces; "
                "the new quadratic measurement contains the minimum valid pair. "
                "Runtime is provenance, not a cross-allocation speed comparison."
            ),
        },
        "measurements": MEASUREMENTS,
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
        "quadratic_cold_warm_parity": payload["quadratic_cold_warm_parity"],
    }
    if _canonical_sha256(compact) != source["compact_sha256"]:
        errors.append("compact checksum mismatch")
    recomputed = [
        _compare(payload["rungs"][0], payload["rungs"][1], "uniform22_to_linear36"),
        _compare(payload["rungs"][1], payload["rungs"][2], "linear36_to_quadratic44"),
    ]
    if recomputed != payload["comparisons"]:
        errors.append("stored comparison arithmetic mismatch")
    outcome = _outcome(
        payload["rungs"],
        recomputed,
        payload["quadratic_cold_warm_parity"]["scientific_arrays_exact"],
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
            errors.append("external quadratic warm result missing")
        else:
            record = source["results"]["quadratic44_warm"]
            if _sha256(warm_result) != record["sha256"]:
                errors.append("external quadratic44_warm result checksum mismatch")
            else:
                equal, mismatches = _scientific_arrays_equal(
                    dkx.Result.load(result_paths["quadratic44"]),
                    dkx.Result.load(warm_result),
                )
                if {
                    "scientific_arrays_exact": equal,
                    "ignored_timing_arrays": ["solve_time_s"],
                    "mismatches": mismatches,
                } != payload["quadratic_cold_warm_parity"]:
                    errors.append("external quadratic cold/warm parity mismatch")
        if geometry is None or _sha256(geometry) != source["geometry_sha256"]:
            errors.append("external geometry checksum mismatch")

    return {
        "schema": "dkx.ambipolar_pitch_speed_groups.audit.v1",
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
        parser.add_argument(f"--{name}-result", type=Path)
    parser.add_argument("--quadratic44-warm-result", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    result_paths = {name: getattr(args, f"{name}_result") for name in NAMES}
    provided = [path is not None for path in result_paths.values()]
    if args.build:
        if (
            not all(provided)
            or args.quadratic44_warm_result is None
            or args.geometry is None
        ):
            parser.error("--build requires all four results and --geometry")
        payload = build_artifact(
            result_paths, args.quadratic44_warm_result, args.geometry
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
        warm_result=args.quadratic44_warm_result,
        geometry=args.geometry,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
