#!/usr/bin/env python
"""Validate the research-lane completion manifest without launching solves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sfincs_jax.research_lane_policy import check_research_lane_completion_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO_ROOT / "docs" / "_static" / "research_lane_completion_2026_05_12.json"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the machine-readable open-lane completion manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Research-lane completion JSON to validate.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = check_research_lane_completion_file(args.manifest, repo_root=REPO_ROOT)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    lanes = payload["lanes"]
    avg_current = sum(float(lane["current_percent"]) for lane in lanes) / len(lanes)
    avg_delta = sum(
        float(lane["current_percent"]) - float(lane["before_percent"]) for lane in lanes
    ) / len(lanes)
    print(
        f"{args.manifest}: research lanes ok "
        f"(lanes={len(lanes)}, avg_current={avg_current:.1f}%, avg_delta={avg_delta:.1f}pp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
