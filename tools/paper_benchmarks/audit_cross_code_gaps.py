"""Audit the production-resolution DKX-vs-SFINCS gap cells from their sealed rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

FLUXES = ("particleFlux_vm_psiHat", "heatFlux_vm_psiHat")


def _scaled(s: Any, d: Any) -> float:
    s, d = np.atleast_1d(np.asarray(s, float)), np.atleast_1d(np.asarray(d, float))
    return float(np.max(np.abs(d - s)) / np.max(np.abs(s)))


def _relative(s: Any, d: Any) -> float:
    s, d = np.asarray(s, float), np.asarray(d, float)
    return float(np.max(np.abs(d - s) / np.abs(s)))


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    gate = payload["acceptance"]
    errors: list[str] = []
    table: dict[str, float] = {}
    for cell in payload["cells"]:
        if cell["diagnostic"]:
            continue
        s, d, name = cell["sfincs"], cell["dkx"], cell["id"]
        if "transportMatrix" in s:
            checks = {"transportMatrix": (_relative(s["transportMatrix"], d["transportMatrix"]),
                                          gate["transport_matrix_relative"])}
        else:
            checks = {k: (_scaled(s[k], d[k]), gate["flux_scaled"]) for k in FLUXES}
            checks["FSABFlow"] = (_scaled(s["FSABFlow"], d["FSABFlow"]), gate["flow_scaled"])
        for key, (err, limit) in checks.items():
            table[f"{name}.{key}"] = err
            if not err <= limit:
                errors.append(f"{name} {key}: {err:.2e} > {limit:.0e}")
    for row in payload["ambipolar"]:
        # Radial current Z_i Gamma_i - Gamma_e at the SFINCS root and 0.05 above it.
        j0 = row["dkx_flux_at_root"][0] - row["dkx_flux_at_root"][1]
        j1 = row["dkx_flux_at_root_plus_0p05"][0] - row["dkx_flux_at_root_plus_0p05"][1]
        shift = abs(j0 / ((j1 - j0) / 0.05) / row["sfincs_root_er"])
        table[f"ambipolar_rN{row['rN']}"] = shift
        if not shift <= gate["ambipolar_root_relative"]:
            errors.append(f"ambipolar rN={row['rN']}: root shift {shift:.2e}")
    if len(payload["ambipolar"]) < 2:
        errors.append("ambipolar needs at least two radii")
    return {"pass": not errors, "errors": errors, "errors_by_cell": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
