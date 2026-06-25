#!/usr/bin/env python
"""Check benchmark JSON artifacts for reproducibility metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from sfincs_jax.validation.benchmark_artifacts import check_benchmark_artifact_files


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate benchmark JSON artifacts against the reproducibility policy."
    )
    parser.add_argument("paths", nargs="+", type=Path, help="Benchmark JSON artifact path(s) to check.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    errors = check_benchmark_artifact_files(args.paths)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    for path in args.paths:
        print(f"{path}: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
