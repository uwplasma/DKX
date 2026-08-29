#!/usr/bin/env python3
"""Build or audit the bounded uniform-pitch ambipolar diagnostic."""

from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import numpy as np

from audit_ambipolar_phase_space_ladder import (
    _canonical_sha256,
    _compact_rung,
    _scaled_movement,
    _sha256,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "ambipolar_pitch_budget_v1.json"
NAMES = ("uniform22_bounded", "uniform26_bounded", "uniform30_bounded")
INPUTS = {
    "uniform22_full": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_uniform22_full.toml",
    "uniform22_bounded": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_uniform22_bounded.toml",
    "uniform26_bounded": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_uniform26_bounded.toml",
    "uniform30_bounded": ROOT
    / "validation/inputs/w7x_standard_native_ambipolar_uniform30_bounded.toml",
}
MEASUREMENTS = {
    "uniform22_full": {
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 341.72,
        "maximum_rss_bytes": 13_068_550_144,
        "peak_footprint_bytes": 31_859_925_880,
    },
    "uniform22_bounded_cold": {
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 176.33,
        "dkx_total_s": 175.07368329200108,
        "maximum_rss_bytes": 3_093_299_200,
        "peak_footprint_bytes": 2_830_077_576,
    },
    "uniform22_bounded_warm": {
        "cache_state": "fresh_process_with_populated_external_cache",
        "process_wall_s": 184.81,
        "dkx_total_s": 183.16227945800347,
        "maximum_rss_bytes": 3_055_878_144,
        "peak_footprint_bytes": 2_923_810_392,
    },
    "uniform26_bounded": {
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 226.04,
        "dkx_total_s": 224.3405415840025,
        "maximum_rss_bytes": 1_742_225_408,
        "peak_footprint_bytes": 2_143_816_560,
    },
    "uniform30_bounded": {
        "cache_state": "fresh_process_with_initially_empty_external_cache",
        "process_wall_s": 296.89,
        "dkx_total_s": 294.6967267500004,
        "maximum_rss_bytes": 2_593_325_056,
        "peak_footprint_bytes": 2_861_944_480,
    },
}


def _maximum_relative(left: Any, right: Any) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    finite = np.isfinite(a) & np.isfinite(b)
    scale = np.maximum(np.abs(a), np.abs(b))
    active = finite & (scale > 0.0)
    if not np.any(active):
        return 0.0
    return float(np.max(np.abs(a[active] - b[active]) / scale[active]))


def _route_parity(full: Any, bounded: Any) -> dict[str, Any]:
    fa = full.arrays
    ba = bounded.arrays
    roots_equal = all(
        np.array_equal(fa[name], ba[name], equal_nan=True)
        for name in (
            "ambipolar_root_count",
            "ambipolar_root_kV_m",
            "ambipolar_root_bracket_kV_m",
            "selected_ambipolar_root",
            "electric_field_kV_m",
        )
    )
    metrics = {
        "roots_and_brackets_exact": roots_equal,
        "max_selected_particle_flux_relative_difference": _maximum_relative(
            fa["particle_flux_m2_s"], ba["particle_flux_m2_s"]
        ),
        "max_selected_heat_flux_relative_difference": _maximum_relative(
            fa["heat_flux_W_m2"], ba["heat_flux_W_m2"]
        ),
        "max_evaluation_particle_flux_relative_difference": _maximum_relative(
            fa["evaluation_particle_flux_m2_s"],
            ba["evaluation_particle_flux_m2_s"],
        ),
        "max_evaluation_heat_flux_relative_difference": _maximum_relative(
            fa["evaluation_heat_flux_W_m2"], ba["evaluation_heat_flux_W_m2"]
        ),
        "full_maximum_accepted_true_residual": float(full.metadata["residual_norm"]),
        "bounded_maximum_accepted_true_residual": float(
            bounded.metadata["residual_norm"]
        ),
        "full_executed_routes": dict(
            full.metadata["ambipolar_solver_attempts"]["executed_route_counts"]
        ),
        "bounded_executed_routes": dict(
            bounded.metadata["ambipolar_solver_attempts"]["executed_route_counts"]
        ),
    }
    gates = {
        "roots_and_brackets_exact": metrics["roots_and_brackets_exact"],
        "selected_particle_flux": metrics[
            "max_selected_particle_flux_relative_difference"
        ]
        <= 2.0e-9,
        "selected_heat_flux": metrics["max_selected_heat_flux_relative_difference"]
        <= 2.0e-9,
        "retained_evaluation_particle_flux": metrics[
            "max_evaluation_particle_flux_relative_difference"
        ]
        <= 2.0e-9,
        "retained_evaluation_heat_flux": metrics[
            "max_evaluation_heat_flux_relative_difference"
        ]
        <= 2.0e-9,
        "bounded_true_residual": metrics["bounded_maximum_accepted_true_residual"]
        <= 1.0e-12,
        "bounded_host_footprint": MEASUREMENTS["uniform22_bounded_warm"][
            "peak_footprint_bytes"
        ]
        < 24 * 2**30,
    }
    return {
        "status": "resolved" if all(gates.values()) else "failed",
        "admission_pass": all(gates.values()),
        "gates": gates,
        "metrics": metrics,
        "footprint_reduction_fraction": 1.0
        - MEASUREMENTS["uniform22_bounded_warm"]["peak_footprint_bytes"]
        / MEASUREMENTS["uniform22_full"]["peak_footprint_bytes"],
        "runtime_reduction_fraction": 1.0
        - MEASUREMENTS["uniform22_bounded_cold"]["process_wall_s"]
        / MEASUREMENTS["uniform22_full"]["process_wall_s"],
        "warm_speedup_claim": False,
    }


def _selected(surface: dict[str, Any]) -> dict[str, Any]:
    return surface["roots"][surface["selected_root_index"]]


def _compare(left: dict[str, Any], right: dict[str, Any], name: str) -> dict[str, Any]:
    topology_stable = True
    common_root_movements: list[dict[str, Any]] = []
    selected_movements: list[dict[str, Any]] = []
    for ls, rs in zip(left["surfaces"], right["surfaces"], strict=True):
        lr = ls["roots"]
        rr = rs["roots"]
        same_count = len(lr) == len(rr)
        topology_stable &= same_count
        if same_count:
            for lroot, rroot in zip(lr, rr, strict=True):
                identity_stable = (
                    lroot["classification"] == rroot["classification"]
                    and lroot["branch_id"] == rroot["branch_id"]
                )
                topology_stable &= identity_stable
                common_root_movements.append(
                    {
                        "surface_index": ls["surface_index"],
                        "root_index": lroot["index"],
                        "identity_stable": identity_stable,
                        "electric_field_movement_kV_m": abs(
                            rroot["electric_field_kV_m"] - lroot["electric_field_kV_m"]
                        ),
                    }
                )
        lsel = _selected(ls)
        rsel = _selected(rs)
        selected_movements.append(
            {
                "surface_index": ls["surface_index"],
                "left_root_count": len(lr),
                "right_root_count": len(rr),
                "electric_field_movement_kV_m": abs(
                    rsel["electric_field_kV_m"] - lsel["electric_field_kV_m"]
                ),
                "particle_flux_scaled_movement": _scaled_movement(
                    lsel["particle_flux_m2_s"], rsel["particle_flux_m2_s"]
                ),
                "heat_flux_scaled_movement": _scaled_movement(
                    lsel["heat_flux_W_m2"], rsel["heat_flux_W_m2"]
                ),
            }
        )
    return {
        "name": name,
        "topology_stable": topology_stable,
        "common_root_movements": common_root_movements,
        "selected_movements": selected_movements,
        "summary": {
            "topology_stable": topology_stable,
            "max_selected_electric_field_movement_kV_m": max(
                row["electric_field_movement_kV_m"] for row in selected_movements
            ),
            "max_selected_particle_flux_scaled_movement": max(
                row["particle_flux_scaled_movement"] for row in selected_movements
            ),
            "max_selected_heat_flux_scaled_movement": max(
                row["heat_flux_scaled_movement"] for row in selected_movements
            ),
        },
    }


def _outcome(
    rungs: list[dict[str, Any]], comparisons: list[dict[str, Any]]
) -> dict[str, Any]:
    gates = []
    for comparison in comparisons:
        summary = comparison["summary"]
        passed = {
            "topology_stable": summary["topology_stable"],
            "selected_electric_field_movement": summary[
                "max_selected_electric_field_movement_kV_m"
            ]
            <= 0.005,
            "selected_particle_flux_movement": summary[
                "max_selected_particle_flux_scaled_movement"
            ]
            <= 0.02,
            "selected_heat_flux_movement": summary[
                "max_selected_heat_flux_scaled_movement"
            ]
            <= 0.02,
        }
        gates.append(
            {
                "comparison": comparison["name"],
                "status": "resolved"
                if all(passed.values())
                else "refinement_exhausted",
                "gates": passed,
                "failed_gates": [key for key, value in passed.items() if not value],
            }
        )
    max_residual = max(
        rung["attempts"]["maximum_accepted_true_residual"] for rung in rungs
    )
    return {
        "status": "refinement_exhausted",
        "admission_pass": False,
        "limits": {
            "max_selected_electric_field_movement_kV_m": 0.005,
            "max_selected_particle_flux_scaled_movement": 0.02,
            "max_selected_heat_flux_scaled_movement": 0.02,
            "max_accepted_true_residual": 1.0e-12,
        },
        "measured_max_accepted_true_residual": max_residual,
        "comparisons": gates,
        "uniform34_or_higher_admitted": False,
        "next_action": (
            "Use the admitted budget propagation to isolate speed-node groups at "
            "fixed bounded work; do not run another uniform whole-profile or pitch48 rung."
        ),
    }


def build_artifact(
    result_paths: dict[str, Path], full_result: Path, geometry: Path
) -> dict[str, Any]:
    import dkx  # noqa: PLC0415

    results = {name: dkx.Result.load(result_paths[name]) for name in NAMES}
    full = dkx.Result.load(full_result)
    resolutions = {
        name: {
            key: int(value)
            for key, value in tomllib.loads(INPUTS[name].read_text(encoding="utf-8"))[
                "resolution"
            ].items()
        }
        for name in NAMES
    }
    rungs = [_compact_rung(results[name], resolutions[name]) for name in NAMES]
    comparisons = [
        _compare(rungs[0], rungs[1], "uniform22_to_uniform26"),
        _compare(rungs[1], rungs[2], "uniform26_to_uniform30"),
    ]
    route_parity = _route_parity(full, results["uniform22_bounded"])
    compact = {
        "route_parity": route_parity,
        "rungs": rungs,
        "comparisons": comparisons,
    }
    all_results = {"uniform22_full": full_result, **result_paths}
    reference = results["uniform22_bounded"]
    return {
        "schema": "dkx.ambipolar_pitch_budget.v1",
        "created_utc": "2026-08-29T17:10:00Z",
        "claim_scope": "bounded_uniform_pitch_route_parity_and_nonconvergence",
        "summary": (
            "The native batch memory budget now controls both chunking and each "
            "solver route. It preserves the uniform22 roots and observables while "
            "cutting process footprint, but uniform22-to-26-to-30 changes root "
            "topology and fails every phase-space movement gate."
        ),
        "source": {
            "bounded_dkx_commit": "f08fd7a1c802b0d860a2d694924d33fd2e52cec0",
            "prior_full_dkx_commit": "2df599f",
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
                name: {"sha256": _sha256(path), "bytes": path.stat().st_size}
                for name, path in all_results.items()
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
                "Cold/warm timings are provenance. The warm uniform22 run was "
                "slower, so no cache-speedup claim is made."
            ),
        },
        "measurements": MEASUREMENTS,
        **compact,
        "outcome": _outcome(rungs, comparisons),
        "exclusions": [
            "not_phase_space_convergence_validation",
            "not_speed_or_zeta_convergence_validation",
            "not_independent_cross_code_ambipolar_validation",
            "not_full_fokker_planck_or_phi1_validation",
            "not_experimental_validation",
            "not_cross_code_performance_validation",
        ],
    }


def audit(
    artifact: Path,
    *,
    result_paths: dict[str, Path] | None = None,
    full_result: Path | None = None,
    geometry: Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    source = payload["source"]
    for record in source["inputs"].values():
        if _sha256(ROOT / record["path"]) != record["sha256"]:
            errors.append(f"input checksum mismatch: {record['path']}")
    compact = {
        "route_parity": payload["route_parity"],
        "rungs": payload["rungs"],
        "comparisons": payload["comparisons"],
    }
    if _canonical_sha256(compact) != source["compact_sha256"]:
        errors.append("compact checksum mismatch")
    recomputed = [
        _compare(payload["rungs"][0], payload["rungs"][1], "uniform22_to_uniform26"),
        _compare(payload["rungs"][1], payload["rungs"][2], "uniform26_to_uniform30"),
    ]
    if recomputed != payload["comparisons"]:
        errors.append("stored comparison arithmetic mismatch")
    outcome = _outcome(payload["rungs"], recomputed)
    if outcome != payload["outcome"]:
        errors.append("stored outcome disagrees with recomputed gates")

    external_verified = result_paths is not None
    if external_verified:
        import dkx  # noqa: PLC0415

        assert result_paths is not None
        if full_result is None:
            errors.append("external full result path missing")
        else:
            record = source["results"]["uniform22_full"]
            if _sha256(full_result) != record["sha256"]:
                errors.append("external uniform22_full result checksum mismatch")
        for index, name in enumerate(NAMES):
            path = result_paths[name]
            record = source["results"][name]
            if _sha256(path) != record["sha256"]:
                errors.append(f"external {name} result checksum mismatch")
                continue
            if (
                _compact_rung(
                    dkx.Result.load(path), payload["rungs"][index]["resolution"]
                )
                != payload["rungs"][index]
            ):
                errors.append(f"external {name} compact rung mismatch")
        if full_result is not None:
            parity = _route_parity(
                dkx.Result.load(full_result),
                dkx.Result.load(result_paths["uniform22_bounded"]),
            )
            if parity != payload["route_parity"]:
                errors.append("external route parity mismatch")
        if geometry is None or _sha256(geometry) != source["geometry_sha256"]:
            errors.append("external geometry checksum mismatch")

    return {
        "schema": "dkx.ambipolar_pitch_budget.audit.v1",
        "artifact": str(artifact),
        "external_results_verified": external_verified,
        "route_parity_pass": payload["route_parity"]["admission_pass"],
        "status": outcome["status"],
        "admission_pass": outcome["admission_pass"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--uniform22-full-result", type=Path)
    for name in NAMES:
        parser.add_argument(f"--{name.replace('_', '-')}-result", type=Path)
    parser.add_argument("--geometry", type=Path)
    args = parser.parse_args()
    result_paths = {name: getattr(args, f"{name}_result") for name in NAMES}
    provided = [path is not None for path in result_paths.values()]
    if args.build:
        if (
            not all(provided)
            or args.uniform22_full_result is None
            or args.geometry is None
        ):
            parser.error("--build requires all four results and --geometry")
        payload = build_artifact(
            result_paths, args.uniform22_full_result, args.geometry
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
            parser.error("external audit requires all three bounded results")
        external = result_paths
    report = audit(
        args.artifact,
        result_paths=external,
        full_result=args.uniform22_full_result,
        geometry=args.geometry,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
