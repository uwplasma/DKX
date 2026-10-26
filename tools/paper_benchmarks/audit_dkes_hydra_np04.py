#!/usr/bin/env python3
"""Audit the sealed DKX versus DKES comparison on the HELIAS hydra_Np04_20190108 reactor.

The solves need the 19 MB Boozer file, which is not in the repository, and the DKES
values cannot be redistributed, so this audit does not rerun the solves and the record holds
agreement statistics, not DKES values. It checks the gates on the stored per-regime
statistics (``cmul >= 1e-3``). With ``DKX_DKES_HYDRA_DIR`` pointing at the private
``DKES_rho_<rho>.h5`` files it also recomputes every statistic from the stored DKX values:

    python tools/paper_benchmarks/audit_dkes_hydra_np04.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.paper_benchmarks.run_dkes_hydra_np04 import agreement, dkes_dir, reference_rows  # noqa: E402

DEFAULT_ARTIFACT = ROOT / "validation" / "dkes_hydra_np04_v1.json"


def audit(artifact: Path = DEFAULT_ARTIFACT, dkes_directory: Path | str | None = None) -> dict[str, Any]:
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    errors: list[str] = []
    gates = payload["gates"]
    excluded = {(e["rho"], "zero" if e["efield"] == 0.0 else "finite") for e in payload["excluded"]}
    worst = {"D11": 0.0, "D33": 0.0, "D31": 0.0}
    counted = 0
    for stat in payload["agreement"]:
        if stat["regime"] == "low_nu":
            continue  # reported, not gated
        where = f"{stat['rho']} {stat['efield']} {stat['regime']}"
        counted += stat["points"]
        if (stat["rho"], stat["efield"]) not in excluded:
            worst["D11"] = max(worst["D11"], stat["d11_max_pct"] / 100)
            if stat["d11_max_pct"] / 100 > gates["d11_rel"]:
                errors.append(f"{where}: D11* differs by {stat['d11_max_pct']}%")
        worst["D33"] = max(worst["D33"], stat["d33_max_pct"] / 100)
        if stat["d33_max_pct"] / 100 > gates["d33_rel"]:
            errors.append(f"{where}: D33* differs by {stat['d33_max_pct']}%")
        if stat["d31_max_abs"] is not None:
            worst["D31"] = max(worst["D31"], stat["d31_max_abs"])
            if stat["d31_max_abs"] > gates["d31_abs"]:
                errors.append(f"{where}: D31* differs by {stat['d31_max_abs']}")
    if counted == 0:
        errors.append("no gated rows")
    checked = None
    if dkes_dir(dkes_directory) is not None:
        reference = reference_rows(dkes_directory)
        checked = agreement(payload["columns"], reference) == payload["agreement"]
        if not checked:
            errors.append("stored agreement statistics differ from the private DKES data")
    return {"schema": "dkx.dkes_hydra_np04.audit.v1", "gated_rows": counted, "worst": worst,
            "recomputed_from_dkes": checked, "errors": errors, "pass": not errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=DEFAULT_ARTIFACT)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
