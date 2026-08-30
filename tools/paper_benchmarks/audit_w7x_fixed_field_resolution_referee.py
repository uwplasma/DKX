#!/usr/bin/env python3
"""Audit the bounded W7-X fixed-field phase-space/SFINCS referee."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX, flux_psi_hat_to_r_hat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/w7x_fixed_field_resolution_referee_v1.json"
MOVEMENT_KEYS = (
    "particle_flux_max_scaled_movement",
    "heat_flux_max_scaled_movement",
    "parallel_current_scaled_movement",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaled_error(left: Any, right: Any) -> np.ndarray:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    scale = np.maximum(np.maximum(np.abs(left_array), np.abs(right_array)), 1.0e-300)
    return np.abs(left_array - right_array) / scale


def _movement(left: Any, right: Any) -> float:
    return float(np.max(_scaled_error(left, right)))


def _require_close(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if not np.allclose(actual, expected, rtol=2.0e-12, atol=1.0e-14):
        errors.append(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def _compare(
    errors: list[str],
    record: dict[str, Any],
    left: dict[str, Any],
    right: dict[str, Any],
) -> None:
    measured = {
        "particle_flux_max_scaled_movement": _movement(
            left["particle_flux_m2_s"], right["particle_flux_m2_s"]
        ),
        "heat_flux_max_scaled_movement": _movement(
            left["heat_flux_W_m2"], right["heat_flux_W_m2"]
        ),
        "parallel_current_scaled_movement": _movement(
            left["parallel_current_A_T_m2"], right["parallel_current_A_T_m2"]
        ),
    }
    for key, value in measured.items():
        _require_close(errors, f"{record['label']} {key}", value, record[key])


def _check_file(
    errors: list[str], label: str, path: Path, record: dict[str, Any]
) -> None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return
    if path.stat().st_size != int(record["bytes"]):
        errors.append(f"{label} byte count mismatch")
    if _sha256(path) != record["sha256"]:
        errors.append(f"{label} checksum mismatch")


def audit(artifact_path: Path, results_root: Path | None = None) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    errors: list[str] = []

    if artifact.get("schema") != "dkx.w7x_fixed_field_resolution_referee.v1":
        errors.append("unexpected artifact schema")

    for name, record in artifact["source"]["inputs"].items():
        _check_file(errors, name, ROOT / record["path"], record)

    case = artifact["case"]
    factor = flux_psi_hat_to_r_hat(
        psi_a_hat=-0.38493580592574961,
        a_hat=0.53697,
        r_n=float(case["r_N"]),
    )
    _require_close(errors, "radial flux factor", factor, case["radial_flux_factor"])

    sfincs = {int(item["pitch"]): item for item in artifact["sfincs_pitch_rungs"]}
    dkx = {item["name"]: item for item in artifact["dkx_rungs"]}
    comparisons = artifact["comparisons"]

    for record, pair in zip(
        comparisons["sfincs_pitch"], ((52, 70), (70, 90), (90, 120), (120, 150))
    ):
        _compare(errors, record, sfincs[pair[0]], sfincs[pair[1]])

    _compare(
        errors,
        comparisons["dkx_pitch_at_admitted_zeta"],
        dkx["pitch150_zeta85"],
        dkx["pitch180_zeta85"],
    )
    for record, pair in zip(
        comparisons["dkx_zeta"], ((37, 45), (45, 61), (61, 85), (85, 109))
    ):
        _compare(
            errors,
            record,
            dkx[f"pitch150_zeta{pair[0]}"],
            dkx[f"pitch150_zeta{pair[1]}"],
        )
    _compare(
        errors,
        comparisons["dkx_speed"],
        dkx["pitch150_zeta85"],
        dkx["speed10_pitch150_zeta85"],
    )
    theta_names = (
        ("pitch150_zeta85", "theta19_pitch150_zeta85"),
        ("theta19_pitch150_zeta85", "theta23_pitch150_zeta85"),
        ("theta23_pitch150_zeta85", "theta29_pitch150_zeta85"),
    )
    for record, names in zip(comparisons["dkx_theta"], theta_names):
        _compare(errors, record, dkx[names[0]], dkx[names[1]])

    parity = comparisons["dkx_sfincs_pitch150_zeta37_parity"]
    dkx_surface = dkx["pitch150_zeta37"]
    sfincs_high = sfincs[150]
    particle_parity = _scaled_error(
        dkx_surface["particle_flux_m2_s"][1], sfincs_high["particle_flux_m2_s"]
    )
    heat_parity = _scaled_error(
        dkx_surface["heat_flux_W_m2"][1], sfincs_high["heat_flux_W_m2"]
    )
    current_parity = _movement(
        dkx_surface["parallel_current_A_T_m2"][1],
        sfincs_high["parallel_current_A_T_m2"],
    )
    _require_close(
        errors, "particle parity", particle_parity, parity["particle_flux_scaled_error"]
    )
    _require_close(errors, "heat parity", heat_parity, parity["heat_flux_scaled_error"])
    _require_close(
        errors,
        "current parity",
        current_parity,
        parity["parallel_current_scaled_error"],
    )
    measured_parity = float(
        max(np.max(particle_parity), np.max(heat_parity), current_parity)
    )
    _require_close(
        errors, "maximum parity", measured_parity, parity["maximum_scaled_error"]
    )

    gate = float(artifact["acceptance"]["max_observable_scaled_movement"])
    parity_gate = float(artifact["acceptance"]["max_cross_code_scaled_error"])
    residual_gate = float(artifact["acceptance"]["max_dkx_primal_residual"])
    sfincs_residual_gate = float(artifact["acceptance"]["max_sfincs_ksp_residual"])
    outcome = artifact["outcome"]
    sfincs_pitch_pass = (
        max(comparisons["sfincs_pitch"][-1][key] for key in MOVEMENT_KEYS) <= gate
    )
    joint_pitch_pass = (
        max(comparisons["dkx_pitch_at_admitted_zeta"][key] for key in MOVEMENT_KEYS)
        <= gate
    )
    zeta_flux_pass = (
        max(
            comparisons["dkx_zeta"][-1]["particle_flux_max_scaled_movement"],
            comparisons["dkx_zeta"][-1]["heat_flux_max_scaled_movement"],
        )
        <= gate
    )
    speed_flux_pass = (
        max(
            comparisons["dkx_speed"]["particle_flux_max_scaled_movement"],
            comparisons["dkx_speed"]["heat_flux_max_scaled_movement"],
        )
        <= gate
    )
    theta_flux_pass = (
        max(
            comparisons["dkx_theta"][0]["particle_flux_max_scaled_movement"],
            comparisons["dkx_theta"][0]["heat_flux_max_scaled_movement"],
        )
        <= gate
    )
    current_theta_pass = (
        comparisons["dkx_theta"][-1]["parallel_current_scaled_movement"] <= gate
    )
    expected = {
        "sfincs_pitch_converged": sfincs_pitch_pass,
        "dkx_pitch_converged_at_zeta85": joint_pitch_pass,
        "dkx_zeta85_flux_converged_against_zeta109": zeta_flux_pass,
        "dkx_speed8_flux_converged_against_speed10": speed_flux_pass,
        "dkx_theta15_flux_converged_against_theta19": theta_flux_pass,
        "parallel_current_theta_converged": current_theta_pass,
    }
    for key, value in expected.items():
        if bool(outcome[key]) != value:
            errors.append(f"stored outcome mismatch for {key}")
    if not all(
        (
            sfincs_pitch_pass,
            joint_pitch_pass,
            zeta_flux_pass,
            speed_flux_pass,
            theta_flux_pass,
        )
    ):
        errors.append("fixed-field transport-flux convergence gate failed")
    if (
        current_theta_pass
        or outcome["parallel_current_status"] != "refinement_exhausted"
    ):
        errors.append("parallel-current negative result was not retained")
    if (
        not outcome["transport_flux_fixed_field_admitted"]
        or outcome["whole_profile_admitted"]
    ):
        errors.append("claim boundary mismatch")
    if measured_parity > parity_gate:
        errors.append("cross-code parity gate failed")
    if max(item["maximum_primal_residual"] for item in dkx.values()) > residual_gate:
        errors.append("DKX residual gate failed")
    if (
        max(item["ksp_final_residual"] for item in sfincs.values())
        > sfincs_residual_gate
    ):
        errors.append("SFINCS residual gate failed")

    external_verified = False
    if results_root is not None:
        import h5py
        import netCDF4

        results_root = results_root.resolve()
        for item in sfincs.values():
            for key in ("raw_output", "raw_log"):
                record = item[key]
                _check_file(
                    errors,
                    f"SFINCS pitch {item['pitch']} {key}",
                    results_root / record["relative_path"],
                    record,
                )
            path = results_root / item["raw_output"]["relative_path"]
            if path.is_file():
                with h5py.File(path, "r") as handle:
                    particle_psi = np.asarray(
                        handle["particleFlux_vm_psiHat"], dtype=np.float64
                    ).reshape(-1)
                    heat_psi = np.asarray(
                        handle["heatFlux_vm_psiHat"], dtype=np.float64
                    ).reshape(-1)
                    current_hat = float(
                        np.asarray(handle["FSABjHat"], dtype=np.float64).reshape(-1)[0]
                    )
                    _require_close(
                        errors,
                        "raw SFINCS particle",
                        particle_psi,
                        item["particle_flux_psi_hat"],
                    )
                    _require_close(
                        errors,
                        "raw SFINCS heat",
                        heat_psi,
                        item["heat_flux_psi_hat"],
                    )
                    _require_close(
                        errors,
                        "raw SFINCS particle SI",
                        particle_psi * factor * PARTICLE_FLUX,
                        item["particle_flux_m2_s"],
                    )
                    _require_close(
                        errors,
                        "raw SFINCS heat SI",
                        heat_psi * factor * HEAT_FLUX,
                        item["heat_flux_W_m2"],
                    )
                    _require_close(
                        errors,
                        "raw SFINCS current SI",
                        current_hat * PARALLEL_CURRENT,
                        item["parallel_current_A_T_m2"],
                    )
        for item in dkx.values():
            for key in ("raw_output", "raw_log", "runtime_input"):
                record = item[key]
                _check_file(
                    errors,
                    f"DKX {item['name']} {key}",
                    results_root / record["relative_path"],
                    record,
                )
            path = results_root / item["raw_output"]["relative_path"]
            if path.is_file():
                with netCDF4.Dataset(path) as handle:
                    _require_close(
                        errors,
                        "raw DKX particle",
                        np.asarray(handle["particle_flux_m2_s"][:]),
                        item["particle_flux_m2_s"],
                    )
                    _require_close(
                        errors,
                        "raw DKX heat",
                        np.asarray(handle["heat_flux_W_m2"][:]),
                        item["heat_flux_W_m2"],
                    )
                    _require_close(
                        errors,
                        "raw DKX current",
                        np.asarray(handle["parallel_current_A_T_m2"][:]),
                        item["parallel_current_A_T_m2"],
                    )
        external_verified = True

    return {
        "schema": "dkx.w7x_fixed_field_resolution_referee.audit.v1",
        "artifact": str(artifact_path),
        "external_results_verified": external_verified,
        "maximum_cross_code_scaled_error": measured_parity,
        "transport_flux_fixed_field_admitted": bool(
            outcome["transport_flux_fixed_field_admitted"]
        ),
        "parallel_current_status": outcome["parallel_current_status"],
        "errors": errors,
        "pass": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--results-root", type=Path)
    args = parser.parse_args()
    result = audit(args.artifact, args.results_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
