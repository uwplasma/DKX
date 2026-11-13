"""Audit an ambipolar rung gate (HSX, W7-X) from its sealed roots."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

# Every root and selected flux may move by at most this between consecutive rungs.
GATE = 0.02
# Absolute floor of the relative difference, in the artifact's own units (Er in kV/m,
# fluxes in SI). It only matters when both values are essentially zero, where it makes
# the difference 0 instead of dividing by zero.
FLOOR = 1e-12


def rel(a: float, b: float) -> float:
    """Symmetric, zero-safe relative difference |a - b| / max(|a|, |b|, FLOOR)."""
    return abs(a - b) / max(abs(a), abs(b), FLOOR)


def grid_size(resolution: dict[str, int]) -> int:
    """Number of phase-space grid points of a rung, the key rungs are ordered by."""
    return math.prod(int(v) for v in resolution.values())


def _changes(a: dict[str, Any], b: dict[str, Any]) -> float:
    pairs = [(a["Er"], b["Er"])] + list(zip(a["gamma"] + a["q"], b["gamma"] + b["q"]))
    return max(rel(u, v) for u, v in pairs)


def audit(artifact_path: Path, geometry: Path | None = None) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    rungs = payload["rungs"]
    names = sorted(rungs, key=lambda name: (grid_size(rungs[name]["resolution"]), name))
    worst = 0.0
    for lo, hi in zip(names, names[1:]):
        a, b = rungs[lo]["roots"], rungs[hi]["roots"]
        if set(a) != set(b):
            errors.append(f"{lo}->{hi}: root sets differ {sorted(a)} {sorted(b)}")
            continue
        for key in a:
            change = _changes(a[key], b[key])
            worst = max(worst, change)
            if not change <= GATE:
                errors.append(f"{lo}->{hi} {key}: {change:.4f} > {GATE}")
    recorded = payload["gate"]["max_relative_difference"]
    if abs(recorded - worst) > 1e-12:
        errors.append(f"recorded max {recorded} differs from recomputed {worst}")
    sha = payload.get("geometry_sha256")
    if sha is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", str(sha)):
            errors.append("geometry_sha256 is not a sha256 digest")
        elif geometry is not None and hashlib.sha256(Path(geometry).read_bytes()).hexdigest() != sha:
            errors.append("geometry checksum mismatch")
    return {
        "pass": not errors,
        "errors": errors,
        "rungs": names,
        "max_relative_difference": worst,
        "geometry_checked": sha is not None and geometry is not None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, help="VMEC file to check against geometry_sha256")
    args = parser.parse_args()
    report = audit(args.artifact, args.geometry)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
