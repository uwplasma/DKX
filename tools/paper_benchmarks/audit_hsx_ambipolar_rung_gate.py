"""Audit the HSX ambipolar rung gate from its sealed roots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Every root and selected flux may move by at most this between consecutive rungs.
GATE = 0.02


def _rel(a: float, b: float) -> float:
    return abs(a - b) / abs(a)


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    names = sorted(payload["rungs"])
    errors: list[str] = []
    worst = 0.0
    for lo, hi in zip(names, names[1:]):
        a, b = payload["rungs"][lo]["roots"], payload["rungs"][hi]["roots"]
        if set(a) != set(b):
            errors.append(f"{lo}->{hi}: root sets differ {sorted(a)} {sorted(b)}")
            continue
        for key in a:
            ra, rb = a[key], b[key]
            pairs = [(ra["Er"], rb["Er"])] + list(zip(ra["gamma"] + ra["q"], rb["gamma"] + rb["q"]))
            change = max(_rel(u, v) for u, v in pairs)
            worst = max(worst, change)
            if not change <= GATE:
                errors.append(f"{lo}->{hi} {key}: {change:.4f} > {GATE}")
    recorded = payload["gate"]["max_relative_difference"]
    if abs(recorded - worst) > 1e-12:
        errors.append(f"recorded max {recorded} differs from recomputed {worst}")
    return {"pass": not errors, "errors": errors, "rungs": names, "max_relative_difference": worst}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
