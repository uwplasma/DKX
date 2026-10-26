"""Audit the 633,604-unknown HSX-like gap deck records: DKX routes against the benchmark-row schema."""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = ROOT / "validation" / "benchmark_schema.toml"
DECK_TOLERANCE = 1.0e-8  # tolerance requested of the DKX rows (the deck says 1e-7)
AGREE_GATE = 1.0e-6  # two passed DKX routes agree on the solution to this relative norm


def audit(artifact_path: Path) -> dict[str, Any]:
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    schema = tomllib.loads(SCHEMA.read_text(encoding="utf-8"))
    errors: list[str] = []
    table: dict[str, float] = {}
    statuses = set(schema["outcome"]["allowed_status"])
    deck_sha = payload["deck"]["input_sha256"]
    for row in payload["records"]:
        name = row["artifact_id"]
        for key in schema["required_top_level"]:
            if key not in row:
                errors.append(f"{name}: missing {key}")
        for section in ("repository", "software", "hardware", "case", "model", "resolution", "route", "timing_s", "memory_bytes", "accuracy", "outcome"):
            for key in schema.get(section, {}).get("required", []):
                if key not in row.get(section, {}):
                    errors.append(f"{name}: missing {section}.{key}")
        status = row["outcome"]["status"]
        if status not in statuses:
            errors.append(f"{name}: unknown status {status}")
        if status != "passed":
            for key in schema["outcome"]["required_when_not_passed"]:
                if not row["outcome"].get(key):
                    errors.append(f"{name}: a {status} row needs {key}")
        if row["case"]["input_sha256"] != deck_sha:
            errors.append(f"{name}: input_sha256 differs from the deck")
        if row["resolution"]["unknowns"] != payload["deck"]["unknowns"]:
            errors.append(f"{name}: unknowns differ from the deck")
        if status == "passed":
            residual = row["accuracy"]["true_residual"]
            table[f"{name}.true_residual"] = residual
            if not residual <= DECK_TOLERANCE:
                errors.append(f"{name}: true residual {residual:.2e} above the deck tolerance {DECK_TOLERANCE:.0e}")
    passed = [r["artifact_id"] for r in payload["records"] if r["outcome"]["status"] == "passed" and r["artifact_id"].startswith("dkx")]
    if not passed:
        errors.append("no passed DKX row")
    for pair in payload.get("solution_agreement", []):
        table[f"agreement.{pair['a']}.{pair['b']}"] = pair["relative_difference"]
        if not pair["relative_difference"] <= AGREE_GATE:
            errors.append(f"solutions of {pair['a']} and {pair['b']} differ by {pair['relative_difference']:.2e}")
        for key in ("a", "b"):
            if pair[key] not in passed:
                errors.append(f"agreement names {pair[key]}, which is not a passed DKX row")
    return {"pass": not errors, "errors": errors, "errors_by_cell": table}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    report = audit(parser.parse_args().artifact)
    print(json.dumps(report, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
