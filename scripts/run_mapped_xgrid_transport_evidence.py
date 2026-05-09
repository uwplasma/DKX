#!/usr/bin/env python3
"""Run mapped x-grid transport evidence scans and write reviewer artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfincs_jax.mapped_xgrid_transport_evidence import (  # noqa: E402
    copy_namelist_with_resolution,
    run_rational_tail_transport_comparison,
    write_transport_evidence_csv,
    write_transport_evidence_json,
)
from sfincs_jax.namelist import read_sfincs_input  # noqa: E402
from sfincs_jax.v3_driver import solve_v3_transport_matrix_linear_gmres  # noqa: E402


def _parse_log_lengths(text: str) -> tuple[float, ...]:
    values = [token.strip() for token in str(text).replace(",", " ").split()]
    if not values:
        raise argparse.ArgumentTypeError("at least one log-length value is required")
    try:
        return tuple(float(value) for value in values)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid log-length list: {text!r}") from exc


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_namelist", type=Path, help="Base SFINCS input.namelist for mapped candidates.")
    parser.add_argument("--json-out", type=Path, required=True, help="Output JSON evidence path.")
    parser.add_argument("--csv-out", type=Path, required=True, help="Output CSV evidence path.")
    parser.add_argument(
        "--log-lengths",
        type=_parse_log_lengths,
        default=_parse_log_lengths("-1.0,-0.5,0.0,0.5,1.0"),
        help="Comma- or space-separated rational-tail log lengths.",
    )
    parser.add_argument("--reference-namelist", type=Path, default=None, help="Optional high-resolution reference input.")
    parser.add_argument("--reference-nx", type=int, default=None, help="Optional reference Nx built from the base input.")
    parser.add_argument("--eta-kind", default="gauss", choices=("gauss", "uniform"))
    parser.add_argument("--derivative", default="barycentric")
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--restart", type=int, default=80)
    parser.add_argument("--maxiter", type=int, default=400)
    return parser.parse_args(argv)


def _metadata(args: argparse.Namespace) -> dict[str, object]:
    return {
        "input_namelist": str(args.input_namelist),
        "reference_namelist": None if args.reference_namelist is None else str(args.reference_namelist),
        "reference_nx": args.reference_nx,
        "log_lengths": list(args.log_lengths),
        "eta_kind": str(args.eta_kind),
        "derivative": str(args.derivative),
        "eps": float(args.eps),
        "tol": float(args.tol),
        "atol": float(args.atol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
    }


def run_from_args(
    args: argparse.Namespace,
    *,
    solve_fn: Callable[..., object] = solve_v3_transport_matrix_linear_gmres,
) -> int:
    """Execute one evidence scan from parsed arguments."""

    if args.reference_namelist is not None and args.reference_nx is not None:
        raise ValueError("--reference-namelist and --reference-nx are mutually exclusive")

    nml = read_sfincs_input(args.input_namelist)
    if args.reference_namelist is not None:
        reference_nml = read_sfincs_input(args.reference_namelist)
    elif args.reference_nx is not None:
        reference_nml = copy_namelist_with_resolution(nml, nx=int(args.reference_nx))
    else:
        reference_nml = None

    report = run_rational_tail_transport_comparison(
        nml,
        log_length_values=args.log_lengths,
        reference_nml=reference_nml,
        solve_fn=solve_fn,
        eta_kind=str(args.eta_kind),
        derivative=str(args.derivative),
        eps=float(args.eps),
        solve_kwargs={
            "tol": float(args.tol),
            "atol": float(args.atol),
            "restart": int(args.restart),
            "maxiter": int(args.maxiter),
        },
    )
    write_transport_evidence_json(report, args.json_out, metadata=_metadata(args))
    write_transport_evidence_csv(report, args.csv_out)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    print(f"best_by_transport_error_log_length={report.best_by_transport_error.log_length:.8g}")
    return 0


def main(
    argv: list[str] | None = None,
    *,
    solve_fn: Callable[..., object] = solve_v3_transport_matrix_linear_gmres,
) -> int:
    args = _parse_args(argv)
    return run_from_args(args, solve_fn=solve_fn)


if __name__ == "__main__":
    raise SystemExit(main())
