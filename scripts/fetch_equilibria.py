#!/usr/bin/env python3
"""Fetch release-hosted SFINCS-JAX equilibrium fixtures into the local cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true", help="Suppress download progress output.")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(_repo_root()))
    from sfincs_jax.data_fetch import ensure_external_equilibrium_data, external_data_manifest

    target = ensure_external_equilibrium_data(quiet=args.quiet)
    manifest = external_data_manifest()
    if not args.quiet:
        print(f"SFINCS-JAX external data {manifest['version']} ready in {target}")
        for item in manifest["files"]:
            print(f"  {item['path']} ({int(item['size']) / 1024 / 1024:.2f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
