#!/usr/bin/env python
"""Bounded benchmark for RHSMode=1 structured f-block CSR reuse."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_response.kinetic import (
    clear_structured_rhs1_fblock_csr_cache,
    select_structured_rhs1_fblock_csr_operator,
)
from sfincs_jax.v3_fblock import apply_v3_fblock_operator, fblock_operator_from_namelist


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
DEFAULT_OUT = _REPO_ROOT / "outputs" / "rhs1_fblock_csr_reuse_benchmark.json"


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--identity-shift", type=float, default=0.5)
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument("--max-csr-mb", type=float, default=512.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the benchmark JSON payload.")
    return parser


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the bounded CSR reuse benchmark and return JSON-friendly metrics."""

    input_path = args.input.expanduser().resolve()
    payload: dict[str, Any] = {
        "kind": "rhs1_fblock_csr_reuse_benchmark",
        "input": str(input_path),
        "identity_shift": float(args.identity_shift),
        "repeats": int(args.repeats),
        "max_csr_mb": float(args.max_csr_mb),
        "dry_run": bool(args.dry_run),
    }
    if bool(args.dry_run):
        payload["status"] = "planned"
        return payload

    nml = read_sfincs_input(input_path)
    t0 = time.perf_counter()
    op = fblock_operator_from_namelist(nml=nml, identity_shift=float(args.identity_shift))
    operator_build_s = time.perf_counter() - t0

    rhs = jnp.linspace(-0.25, 0.75, int(op.flat_size), dtype=jnp.float64)
    expected = np.asarray(apply_v3_fblock_operator(op, rhs.reshape(op.f_shape))).reshape((-1,))
    jax.block_until_ready(rhs)

    clear_structured_rhs1_fblock_csr_cache()
    rows: list[dict[str, Any]] = []
    max_csr_nbytes = int(max(0.0, float(args.max_csr_mb)) * 1024.0 * 1024.0)
    for repeat in range(int(args.repeats)):
        build_start = time.perf_counter()
        selected = select_structured_rhs1_fblock_csr_operator(op, max_csr_nbytes=max_csr_nbytes)
        select_s = time.perf_counter() - build_start
        if not bool(selected.selected):
            rows.append(
                {
                    "repeat": int(repeat),
                    "selected": False,
                    "reason": str(selected.reason),
                    "cache_hit": bool(selected.cache_hit),
                    "select_s": float(select_s),
                }
            )
            continue
        matvec_start = time.perf_counter()
        got = selected.matvec(np.asarray(rhs))
        matvec_s = time.perf_counter() - matvec_start
        rows.append(
            {
                "repeat": int(repeat),
                "selected": True,
                "reason": str(selected.reason),
                "cache_hit": bool(selected.cache_hit),
                "select_s": float(select_s),
                "matvec_s": float(matvec_s),
                "max_abs_error": float(np.max(np.abs(got - expected))),
                "max_rel_error": float(np.max(np.abs(got - expected)) / max(float(np.max(np.abs(expected))), 1.0e-300)),
                "metadata": selected.metadata,
            }
        )

    ok_rows = [row for row in rows if bool(row.get("selected", False))]
    payload.update(
        {
            "status": "ok" if len(ok_rows) == len(rows) else "failed",
            "operator_build_s": float(operator_build_s),
            "rows": rows,
            "max_abs_error": max((float(row["max_abs_error"]) for row in ok_rows), default=None),
            "max_rel_error": max((float(row["max_rel_error"]) for row in ok_rows), default=None),
            "cache_hits": int(sum(1 for row in ok_rows if bool(row.get("cache_hit", False)))),
        }
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    payload = run_benchmark(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))
    print(f"Wrote {args.out}")
    return 0 if payload.get("status") in {"ok", "planned"} else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
