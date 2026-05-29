#!/usr/bin/env python
"""Check QI device/GPU evidence artifacts without launching solves.

This validates provenance, route metadata, and fail-closed output behavior for
QI device/operator-reuse artifacts. It does not certify residual convergence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from sfincs_jax.qi_device_artifact_policy import check_qi_device_artifact_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        type=Path,
        help="JSON artifact files or directories to scan recursively.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable output.",
    )
    parser.add_argument(
        "--min-relevant",
        type=int,
        default=1,
        help="Minimum number of QI device artifacts expected in the scan.",
    )
    return parser


def _expand(paths: list[Path]) -> list[Path]:
    expanded: list[Path] = []
    for path in paths:
        if path.is_dir():
            expanded.extend(sorted(candidate for candidate in path.rglob("*.json") if candidate.is_file()))
        else:
            expanded.append(path)
    return expanded


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    checks = check_qi_device_artifact_files(_expand(args.paths))
    relevant = [check for check in checks if check.relevant]
    failing = [check for check in relevant if not check.passed]
    if len(relevant) < max(0, int(args.min_relevant)):
        missing_error = f"expected at least {args.min_relevant} QI device artifact(s), found {len(relevant)}"
    else:
        missing_error = ""

    if args.json:
        print(
            json.dumps(
                {
                    "checked": len(checks),
                    "relevant": len(relevant),
                    "failed": len(failing) + int(bool(missing_error)),
                    "missing_error": missing_error,
                    "artifacts": [
                        {
                            "path": str(check.path),
                            "relevant": check.relevant,
                            "passed": check.passed,
                            "errors": list(check.errors),
                        }
                        for check in checks
                    ],
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        for check in relevant:
            print(f"{check.path}: {'pass' if check.passed else 'fail'}")
            for error in check.errors:
                print(f"  {error}", file=sys.stderr)
        if missing_error:
            print(missing_error, file=sys.stderr)
        print(f"summary: checked={len(checks)}, relevant={len(relevant)}, failed={len([check for check in relevant if not check.passed])}")

    return 1 if failing or missing_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
