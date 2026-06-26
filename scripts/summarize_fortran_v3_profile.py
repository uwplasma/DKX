#!/usr/bin/env python3
"""Summarize SFINCS Fortran v3 PETSc/MUMPS logs as compact JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sfincs_jax.validation.fortran import parse_fortran_v3_profile_file


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Fortran v3 stdout/PETSc log to parse")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args(argv)

    summary = parse_fortran_v3_profile_file(args.log)
    summary["source_log"] = str(args.log)
    payload = json.dumps(summary, indent=2, sort_keys=True, default=_json_default)
    if args.out is None:
        print(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
