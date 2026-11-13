"""Audit the W7-X full-Fokker-Planck ambipolar root agreement between DKX and SFINCS v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
ROOT_GATE = 0.02
FLUX_GATE = 0.01
# Absolute floor of the relative difference, in the artifact's units (kV/m, SI fluxes);
# two values that are both essentially zero then differ by 0 instead of dividing by zero.
FLOOR = 1e-12


def _rel(a: float, b: float) -> float:
    """Symmetric, zero-safe relative difference |a - b| / max(|a|, |b|, FLOOR)."""
    return abs(a - b) / max(abs(a), abs(b), FLOOR)


def _pairs(case: dict[str, Any]) -> list[tuple[str, float, float]]:
    dkx, sfincs = case["dkx"], case["sfincs"]
    out = [("root", dkx["Er"], sfincs["Er"])]
    for name in ("particle_flux", "heat_flux"):
        for species, a, b in zip(case["species"], dkx[name], sfincs[name]):
            out.append((f"{name}.{species}", a, b))
    return out


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    deck = ROOT / payload["source"]["deck"]["path"]
    if not deck.is_file():
        errors.append(f"missing deck {deck.name}")
    elif hashlib.sha256(deck.read_bytes()).hexdigest() != payload["source"]["deck"]["sha256"]:
        errors.append("deck checksum mismatch")

    # The accepted rung is the finest grid in the artifact, recomputed here; the recorded
    # label is only checked against it.
    rungs = payload["rungs"]
    recorded = payload["gate"]
    finest = max(rungs, key=lambda label: (math.prod(rungs[label]["grid"]), label))
    if recorded["accepted_rung"] != finest:
        errors.append(f"recorded accepted rung {recorded['accepted_rung']} is not the finest rung {finest}")
    accepted = rungs[finest]
    top = {"root": 0.0, "flux": 0.0}
    for name, a, b in _pairs(accepted):
        kind = "root" if name == "root" else "flux"
        change = _rel(a, b)
        top[kind] = max(top[kind], change)
        if change > (ROOT_GATE if kind == "root" else FLUX_GATE):
            errors.append(f"{finest} {name}: {change:.4f} exceeds the gate")
    for kind, key in (("root", "max_root_difference"), ("flux", "max_flux_difference")):
        if abs(recorded[key] - top[kind]) > 1e-12:
            errors.append(f"recorded {key} {recorded[key]} differs from recomputed {top[kind]}")
    if recorded["root_gate"] != ROOT_GATE or recorded["flux_gate"] != FLUX_GATE:
        errors.append("recorded gate thresholds differ from the audited thresholds")
    for code in ("dkx", "sfincs"):
        for label, case in payload["rungs"].items():
            if not abs(case[code]["radial_current_at_root"]) <= case[code]["current_scale"] * 1e-4:
                errors.append(f"{label} {code}: radial current at the root is not small against its scale")
    scan = {tuple(row["grid"]): row["Er"] for row in payload["sfincs_resolution_scan"]}
    if tuple(accepted["grid"]) not in scan:
        errors.append("the accepted grid is missing from the SFINCS resolution scan")
    elif _rel(scan[tuple(accepted["grid"])], accepted["sfincs"]["Er"]) > 1e-4:
        errors.append("the scan root on the accepted grid differs from the SFINCS rung root")
    return {"pass": not errors, "errors": errors, "max_root_difference": top["root"], "max_flux_difference": top["flux"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
