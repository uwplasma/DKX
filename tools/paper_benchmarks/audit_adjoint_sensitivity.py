"""Audit the DKX adjoint-sensitivity record: reverse-mode derivatives against central differences."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROW_GATE = (
    1.0e-6  # best step: worst error of a row, relative to that row's largest derivative
)
ENTRY_GATE = 1.0e-5  # best step: worst error of an entry, relative to the entry
ORDER_GATE = 50.0  # error at the largest step over error at one tenth of it (second order gives 100)


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    errors: list[str] = []
    table: dict[str, float] = {}
    steps = [f"{h:g}" for h in payload["steps"]]
    if not payload["cases"]:
        errors.append("no case")
    for case in payload["cases"]:
        name = case["id"]
        adjoint = np.asarray(case["adjoint"], dtype=float)
        shape = (len(case["observables"]), len(case["parameters"]))
        if adjoint.shape != shape or not np.all(np.isfinite(adjoint)):
            errors.append(f"{name}: adjoint is not a finite {shape} array")
            continue
        row_scale = np.abs(adjoint).max(axis=1, keepdims=True)
        row_err, entry_err = {}, {}
        for step in steps:
            central = np.asarray(case["central_difference"][step], dtype=float)
            if central.shape != shape:
                errors.append(
                    f"{name}: central differences at {step} have shape {central.shape}"
                )
                continue
            row_err[step] = float((np.abs(central - adjoint) / row_scale).max())
            entry_err[step] = float((np.abs(central - adjoint) / np.abs(adjoint)).max())
        if len(row_err) != len(steps):
            continue
        best = min(steps, key=row_err.get)
        table[f"{name}.row_error_best"] = row_err[best]
        table[f"{name}.entry_error_best"] = min(entry_err.values())
        if row_err[best] > ROW_GATE:
            errors.append(
                f"{name}: best row-scaled error {row_err[best]:.2e} above {ROW_GATE:.0e}"
            )
        if min(entry_err.values()) > ENTRY_GATE:
            errors.append(
                f"{name}: best entry error {min(entry_err.values()):.2e} above {ENTRY_GATE:.0e}"
            )
        order = row_err[steps[0]] / row_err[steps[1]]
        table[f"{name}.order_ratio"] = order
        if order < ORDER_GATE:
            errors.append(
                f"{name}: error falls only {order:.0f}x for a tenth of the step, not second order"
            )
    return {"pass": not errors, "errors": errors, "errors_by_cell": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
