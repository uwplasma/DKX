"""Audit the Phi1 DKX-vs-Fortran-SFINCS ladder artifact from its sealed rows."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# SFINCS ran at solverTolerance 1e-6; agreement must sit within ten times that.
CODE_GATE = 1.0e-5
# Main-deck observables between the deck resolution (B) and the next rung (C).
RESOLUTION_GATE = 1.0e-2


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(b)


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    rows = {row["case"]: row for row in payload["rows"]}
    errors: list[str] = []
    compared = 0
    for name, row in rows.items():
        if "sfincs_particleFlux_vm_psiHat" not in row:
            continue
        compared += 1
        pairs = list(zip(row["dkx_particleFlux_vm_psiHat"], row["sfincs_particleFlux_vm_psiHat"]))
        pairs.append((row["dkx_phi1_rms"], row["sfincs_phi1_rms"]))
        worst = max(_rel(a, b) for a, b in pairs)
        if not worst <= CODE_GATE:
            errors.append(f"{name}: DKX vs SFINCS {worst:.2e} > {CODE_GATE:.0e}")
    for name in ("main_B", "main_C", "imp_B"):
        if "sfincs_particleFlux_vm_psiHat" not in rows.get(name, {}):
            errors.append(f"{name}: the deck-resolution ladder needs a SFINCS comparison")
    b, c = rows["main_B"], rows["main_C"]
    values_b = b["dkx_particleFlux_vm_psiHat"] + [b["dkx_phi1_rms"]]
    values_c = c["dkx_particleFlux_vm_psiHat"] + [c["dkx_phi1_rms"]]
    drift = max(_rel(x, y) for x, y in zip(values_b, values_c))
    if not drift <= RESOLUTION_GATE:
        errors.append(f"main deck B->C drift {drift:.2e} > {RESOLUTION_GATE:.0e}")
    return {"pass": not errors, "errors": errors, "compared_rungs": compared,
            "main_resolution_drift": drift}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
