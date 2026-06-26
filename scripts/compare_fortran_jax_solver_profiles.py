#!/usr/bin/env python3
"""Compare compact SFINCS Fortran-v3 and SFINCS-JAX solver profiles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from sfincs_jax.solvers.diagnostics import compare_solver_profile_files


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fortran-profile", required=True, type=Path, help="Fortran profile JSON")
    parser.add_argument("--jax-profile", required=True, type=Path, help="SFINCS-JAX campaign or solver-trace JSON")
    parser.add_argument("--case-index", type=int, default=0, help="Campaign case index when --jax-profile is a report")
    parser.add_argument("--out", type=Path, default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    comparison = compare_solver_profile_files(
        fortran_profile_path=args.fortran_profile,
        jax_profile_path=args.jax_profile,
        case_index=int(args.case_index),
    )
    comparison["source_fortran_profile"] = str(args.fortran_profile)
    comparison["source_jax_profile"] = str(args.jax_profile)
    payload = json.dumps(comparison, indent=2, sort_keys=True, default=_json_default)
    if args.out is None:
        print(payload)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
