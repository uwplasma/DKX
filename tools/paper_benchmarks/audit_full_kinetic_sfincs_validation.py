#!/usr/bin/env python3
"""Audit the pinned full-kinetic DKX/SFINCS validation rung.

The expensive external-code and JAX runs are not CI dependencies. This script
recomputes every checked acceptance value from the compact outputs, verifies
the exact DKX-owned decks, and can additionally verify the external run tree:

    python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py
    python tools/paper_benchmarks/audit_full_kinetic_sfincs_validation.py \
        --results-root ../runtime/evidence/full-fp
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACT = ROOT / "validation" / "full_kinetic_sfincs_v1.json"
NONZERO_SCALARS = ("FSABFlow", "heatFlux_vm_psiHat")
SPECTRA = (
    "FSABFlow_vs_x",
    "particleFlux_vm_psiHat_vs_x",
    "heatFlux_vm_psiHat_vs_x",
)
NEAR_ZERO_SCALARS = ("particleFlux_vm_psiHat", "NTV")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _scaled_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right))


def _spectrum_error(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("matched cross-code spectra must have equal lengths")
    numerator = max(abs(a - b) for a, b in zip(left, right, strict=True))
    denominator = max(*(abs(value) for value in left), *(abs(value) for value in right))
    return numerator / denominator


def _verify_external_results(
    payload: dict[str, Any], results_root: Path, errors: list[str]
) -> None:
    for rung in payload["rungs"]:
        rung_id = str(rung["id"])
        paths = {
            "sfincs.raw_output_sha256": results_root
            / rung_id
            / "sfincs"
            / "sfincsOutput.h5",
            "sfincs.raw_log_sha256": results_root / rung_id / "sfincs" / "sfincs.log",
            "dkx.raw_output_sha256": results_root / rung_id / "dkx" / "sfincsOutput.h5",
            "dkx.cold_log_sha256": results_root / rung_id / "dkx" / "dkx-cold.log",
            "dkx.warm_log_sha256": results_root / rung_id / "dkx" / "dkx-warm.log",
        }
        if "warm_output_sha256" in rung["dkx"]:
            paths["dkx.warm_output_sha256"] = (
                results_root / rung_id / "dkx" / "sfincsOutput_warm.h5"
            )
        for key, path in paths.items():
            owner, checksum_key = key.split(".")
            expected = str(rung[owner][checksum_key])
            if not path.exists() or _sha256(path) != expected:
                errors.append(f"{rung_id}: external {key} checksum mismatch")


def audit(artifact: Path, *, results_root: Path | None = None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    acceptance = payload["acceptance"]
    nonzero_scalars = tuple(
        acceptance.get(
            "nonzero_scalar_observables",
            acceptance.get("accepted_nonzero_observables", NONZERO_SCALARS),
        )
    )
    spectra = tuple(acceptance.get("spectrum_observables", SPECTRA))
    near_zero_scalars = tuple(
        acceptance.get("near_zero_observables", NEAR_ZERO_SCALARS)
    )
    convergence_observables = tuple(
        acceptance.get("convergence_observables", nonzero_scalars)
    )
    errors: list[str] = []
    cross_code_errors: list[float] = []
    near_zero_values: list[float] = []
    residuals: list[float] = []
    by_id: dict[str, dict[str, Any]] = {}

    for rung in payload["rungs"]:
        rung_id = str(rung["id"])
        by_id[rung_id] = rung
        deck = ROOT / str(rung["input"])
        if not deck.exists() or _sha256(deck) != rung["input_sha256"]:
            errors.append(f"{rung_id}: input checksum mismatch")

        outputs: dict[str, dict[str, Any]] = {}
        for code in ("sfincs", "dkx"):
            code_output = rung[code]["outputs"]
            outputs[code] = code_output
            if _canonical_sha256(code_output) != rung[code]["compact_output_sha256"]:
                errors.append(f"{rung_id}: {code} compact-output checksum mismatch")
            residuals.append(float(rung[code]["completed_true_residual"]))

        for observable in nonzero_scalars:
            cross_code_errors.append(
                _scaled_error(
                    float(outputs["sfincs"][observable]),
                    float(outputs["dkx"][observable]),
                )
            )
        for observable in spectra:
            cross_code_errors.append(
                _spectrum_error(
                    [float(value) for value in outputs["sfincs"][observable]],
                    [float(value) for value in outputs["dkx"][observable]],
                )
            )
        for observable in near_zero_scalars:
            near_zero_values.extend(
                abs(float(outputs[code][observable])) for code in ("sfincs", "dkx")
            )

    if set(by_id) != {"high", "ultra"}:
        errors.append("artifact must contain exactly the high and ultra accepted rungs")
        movement = float("inf")
    else:
        movement = max(
            _scaled_error(
                float(by_id["high"][code]["outputs"][observable]),
                float(by_id["ultra"][code]["outputs"][observable]),
            )
            for code in ("sfincs", "dkx")
            for observable in convergence_observables
        )

    measured = {
        "max_cross_code_scaled_error": max(cross_code_errors),
        "max_high_to_ultra_scaled_movement": movement,
        "max_near_zero_absolute_value": max(near_zero_values),
        "max_completed_true_residual": max(residuals),
    }
    for key, value in measured.items():
        stored = float(acceptance[f"measured_{key}"])
        if abs(value - stored) > max(5e-16, 2e-12 * abs(stored)):
            errors.append(f"stored {key} disagrees with recomputed value")

    passed = (
        measured["max_cross_code_scaled_error"]
        <= float(acceptance["max_cross_code_scaled_error"])
        and measured["max_high_to_ultra_scaled_movement"]
        <= float(acceptance["max_high_to_ultra_scaled_movement"])
        and measured["max_near_zero_absolute_value"]
        <= float(acceptance["max_near_zero_absolute_value"])
        and measured["max_completed_true_residual"]
        <= float(acceptance["max_completed_true_residual"])
    )
    if passed is not bool(acceptance["all_gates_pass"]):
        errors.append("stored all_gates_pass disagrees with recomputed gates")
    if results_root is not None:
        _verify_external_results(payload, results_root, errors)

    return {
        "schema": "dkx.full_kinetic_sfincs.audit.v1",
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
    args = parser.parse_args()
    report = audit(args.artifact, results_root=args.results_root)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
