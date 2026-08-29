#!/usr/bin/env python3
"""Audit the matched native physical-flux / SFINCS W7-X certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from dkx.constants import RadialCoordinates
from dkx.units import HEAT_FLUX, PARALLEL_CURRENT, PARTICLE_FLUX, flux_psi_hat_to_r_hat


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation/native_physical_flux_sfincs_v1.json"


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


def _require_close(errors: list[str], label: str, actual: Any, expected: Any) -> None:
    if not np.allclose(actual, expected, rtol=2.0e-12, atol=1.0e-14):
        errors.append(f"{label} mismatch: actual={actual!r}, expected={expected!r}")


def audit(artifact_path: Path, results_root: Path | None = None) -> dict[str, Any]:
    artifact_path = artifact_path.resolve()
    artifact = json.loads(artifact_path.read_text())
    errors: list[str] = []

    if artifact.get("schema") != "dkx.native_physical_flux_sfincs.v1":
        errors.append("unexpected artifact schema")

    for key in ("native_input", "sfincs_input"):
        record = artifact["source"][key]
        path = ROOT / record["path"]
        if not path.is_file():
            errors.append(f"missing {key}: {path}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"{key} byte count mismatch")
        if _sha256(path) != record["sha256"]:
            errors.append(f"{key} checksum mismatch")

    case = artifact["case"]
    radial = RadialCoordinates(
        psi_a_hat=float(case["psi_a_hat"]),
        a_hat=float(case["a_hat"]),
        r_n=float(case["r_n"]),
    )
    conversion = artifact["conversion"]
    correct = flux_psi_hat_to_r_hat(
        psi_a_hat=radial.psi_a_hat,
        a_hat=radial.a_hat,
        r_n=radial.r_n,
    )
    _require_close(
        errors,
        "correct radial factor",
        correct,
        conversion["correct_d_dr_hat_to_d_dpsi_hat"],
    )
    _require_close(
        errors,
        "previous inverse factor",
        radial.d_dpsi_hat_to_d_dr_hat,
        conversion["previous_inverse_d_dpsi_hat_to_d_dr_hat"],
    )
    if np.isclose(correct, radial.d_dpsi_hat_to_d_dr_hat, rtol=1.0e-12, atol=0.0):
        errors.append("radial conversion regression: correct factor equals its inverse")

    sfincs = artifact["sfincs"]
    dkx = artifact["dkx"]
    comparison = artifact["comparison"]
    particle_error = _scaled_error(dkx["particle_flux_m2_s"], sfincs["particle_flux_m2_s"])
    heat_error = _scaled_error(dkx["heat_flux_W_m2"], sfincs["heat_flux_W_m2"])
    current_error = float(
        _scaled_error(dkx["parallel_current_A_T_m2"], sfincs["parallel_current_A_T_m2"])
    )
    _require_close(errors, "particle scaled error", particle_error, comparison["particle_flux_scaled_error"])
    _require_close(errors, "heat scaled error", heat_error, comparison["heat_flux_scaled_error"])
    _require_close(errors, "parallel current scaled error", current_error, comparison["parallel_current_scaled_error"])
    measured_max = float(max(np.max(particle_error), np.max(heat_error)))
    _require_close(errors, "maximum scaled error", measured_max, comparison["max_scaled_error"])

    acceptance = artifact["acceptance"]
    gates = {
        "cross_code": measured_max <= float(acceptance["max_cross_code_scaled_error"]),
        "parallel_current": current_error
        <= float(acceptance["max_parallel_current_scaled_error"]),
        "dkx_residual": float(dkx["primal_residual"])
        <= float(acceptance["max_dkx_primal_residual"]),
        "sfincs_residual": float(sfincs["ksp_final_residual"])
        <= float(acceptance["max_sfincs_ksp_residual"]),
    }
    if not all(gates.values()):
        errors.append(f"acceptance gate failure: {gates}")
    if bool(acceptance["all_gates_pass"]) != all(gates.values()):
        errors.append("stored all_gates_pass does not match recomputed gates")

    external_verified = False
    if results_root is not None:
        import h5py
        import netCDF4

        results_root = results_root.resolve()
        sfincs_h5 = results_root / sfincs["raw_output"]["relative_path"]
        sfincs_log = results_root / sfincs["raw_log"]["relative_path"]
        dkx_nc = results_root / dkx["raw_output"]["relative_path"]
        for label, path, record in (
            ("SFINCS output", sfincs_h5, sfincs["raw_output"]),
            ("SFINCS log", sfincs_log, sfincs["raw_log"]),
            ("DKX result", dkx_nc, dkx["raw_output"]),
        ):
            if not path.is_file():
                errors.append(f"missing external {label}: {path}")
                continue
            if path.stat().st_size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
                errors.append(f"external {label} checksum/size mismatch")

        if sfincs_h5.is_file() and dkx_nc.is_file():
            with h5py.File(sfincs_h5, "r") as handle:
                particle_psi = np.asarray(handle["particleFlux_vm_psiHat"][...], dtype=np.float64).reshape(-1)
                heat_psi = np.asarray(handle["heatFlux_vm_psiHat"][...], dtype=np.float64).reshape(-1)
                current_hat = float(np.asarray(handle["FSABjHat"][...], dtype=np.float64).reshape(-1)[0])
            _require_close(errors, "raw SFINCS particle psiHat", particle_psi, sfincs["particle_flux_psi_hat"])
            _require_close(errors, "raw SFINCS heat psiHat", heat_psi, sfincs["heat_flux_psi_hat"])
            _require_close(errors, "raw SFINCS particle SI", particle_psi * correct * PARTICLE_FLUX, sfincs["particle_flux_m2_s"])
            _require_close(errors, "raw SFINCS heat SI", heat_psi * correct * HEAT_FLUX, sfincs["heat_flux_W_m2"])
            _require_close(errors, "raw SFINCS current SI", current_hat * PARALLEL_CURRENT, sfincs["parallel_current_A_T_m2"])

            with netCDF4.Dataset(dkx_nc) as handle:
                _require_close(errors, "raw DKX particle SI", np.asarray(handle["particle_flux_m2_s"][1]), dkx["particle_flux_m2_s"])
                _require_close(errors, "raw DKX heat SI", np.asarray(handle["heat_flux_W_m2"][1]), dkx["heat_flux_W_m2"])
                _require_close(errors, "raw DKX current SI", float(handle["parallel_current_A_T_m2"][1]), dkx["parallel_current_A_T_m2"])
                _require_close(errors, "raw DKX residual", float(handle["primal_residual"][1]), dkx["primal_residual"])
            external_verified = True

    return {
        "schema": "dkx.native_physical_flux_sfincs.audit.v1",
        "artifact": str(artifact_path),
        "external_results_verified": external_verified,
        "max_cross_code_scaled_error": measured_max,
        "gates": gates,
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
