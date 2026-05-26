#!/usr/bin/env python3
"""Fail if large tracked files have not been explicitly reviewed.

This is a lightweight repository hygiene gate. It does not rewrite Git history;
it prevents future accidental additions of large benchmark outputs, build
artifacts, or copied equilibria.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


DEFAULT_THRESHOLD_MIB = 2.0

# Files above the default threshold that are intentionally kept in the current
# tree. Each entry needs a short reason so the audit is reviewable.
REVIEWED_LARGE_FILES: dict[str, str] = {
    "sfincs_jax/data/equilibria/hsx3free.bc": "canonical HSX Boozer equilibrium used by public geometryScheme=11 examples",
    "sfincs_jax/data/equilibria/w7x_standardConfig.bc": "canonical W7-X Boozer equilibrium used by geometryScheme=11 examples",
    "sfincs_jax/data/equilibria/w7x-sc1.bc": "small W7-X Boozer equilibrium used by paper/example parity cases",
    "sfincs_jax/data/equilibria/wout_w7x_standardConfig.nc": "canonical W7-X VMEC netCDF equilibrium used by geometryScheme=5 examples",
    "sfincs_jax/data/equilibria/wout_w7x_standardConfig.txt": "ASCII VMEC fixture retained for geometryScheme=5 ASCII compatibility",
    "examples/additional_examples/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc": "self-contained QI VMEC example input",
    "benchmarks/production_resolution_inputs_2026-04-30/inputs/additional_examples/wout_QI_nfp2_stable_Er_006_000043_hires_scaled.nc": "self-contained production-resolution QI benchmark input",
}


def _repo_root() -> Path:
    raw = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True)
    return Path(raw.strip())


def _tracked_files(root: Path) -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=root)
    return [root / item.decode() for item in raw.split(b"\0") if item]


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def large_tracked_files(*, root: Path, threshold_bytes: int) -> dict[str, int]:
    """Return tracked files larger than ``threshold_bytes``."""

    large: dict[str, int] = {}
    for path in _tracked_files(root):
        if not path.exists():
            continue
        size = path.stat().st_size
        if size > threshold_bytes:
            large[_relative(path, root)] = int(size)
    return large


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold-mib",
        type=float,
        default=DEFAULT_THRESHOLD_MIB,
        help="Tracked-file review threshold in MiB.",
    )
    args = parser.parse_args(argv)

    root = _repo_root()
    threshold_bytes = int(float(args.threshold_mib) * 1024 * 1024)
    large = large_tracked_files(root=root, threshold_bytes=threshold_bytes)

    missing = sorted(path for path in large if path not in REVIEWED_LARGE_FILES)
    stale = sorted(path for path in REVIEWED_LARGE_FILES if path not in large)

    if missing or stale:
        print("Repository size audit failed.", file=sys.stderr)
        if missing:
            print("\nTracked files above threshold without review:", file=sys.stderr)
            for path in missing:
                print(f"  {large[path] / 1024 / 1024:7.2f} MiB  {path}", file=sys.stderr)
        if stale:
            print("\nReviewed-large-file entries that no longer exist or are below threshold:", file=sys.stderr)
            for path in stale:
                print(f"  {path}", file=sys.stderr)
        return 1

    print(f"Repository size audit passed: {len(large)} reviewed files above {args.threshold_mib:g} MiB.")
    for path in sorted(large):
        print(f"  {large[path] / 1024 / 1024:7.2f} MiB  {path} - {REVIEWED_LARGE_FILES[path]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
