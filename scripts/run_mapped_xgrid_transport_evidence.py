#!/usr/bin/env python3
"""Run mapped x-grid transport evidence scans and write reviewer artifacts."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sfincs_jax.workflows.mapped_xgrid import (  # noqa: E402
    copy_namelist_with_resolution,
    run_rational_tail_transport_comparison,
    write_transport_evidence_csv,
    write_transport_evidence_json,
)
from sfincs_jax.namelist import Namelist, read_sfincs_input  # noqa: E402
from sfincs_jax.problems.transport_solve import solve_v3_transport_matrix_linear_gmres  # noqa: E402


DEFAULT_LOG_LENGTHS = (-1.0, -0.5, 0.0, 0.5, 1.0)


@dataclass(frozen=True)
class CasePreset:
    """Documented evidence-lane preset resolved by the CLI."""

    description: str
    input_namelist: Path
    log_lengths: tuple[float, ...]
    candidate_resolution: dict[str, int]
    reference_nx: int
    group_updates: dict[str, dict[str, object]]


CASE_PRESETS: dict[str, CasePreset] = {
    "tiny_pas_rhsmode2_scheme2": CasePreset(
        description=(
            "Original tiny PAS RHSMode=2 geometryScheme=2 transport fixture; useful for "
            "fast provenance checks but intentionally low-resolution."
        ),
        input_namelist=REPO_ROOT / "tests" / "ref" / "transportMatrix_PAS_tiny_rhsMode2_scheme2.input.namelist",
        log_lengths=(-0.5, 0.0, 0.5),
        candidate_resolution={},
        reference_nx=5,
        group_updates={},
    ),
    "reduced_pas_tokamak_rhsmode2": CasePreset(
        description=(
            "Small real PAS transport-matrix case derived from the reduced tokamak PAS "
            "fixture by switching to RHSMode=2 and using DKES trajectories. The preset "
            "runs mapped candidates at Nx=7 against an Nx=13 reference with active-DOF "
            "reduction, leaving the full reduced Nx=50 case easy to request explicitly."
        ),
        input_namelist=REPO_ROOT / "tests" / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr.input.namelist",
        log_lengths=DEFAULT_LOG_LENGTHS,
        candidate_resolution={"nx": 7, "ntheta": 11, "nzeta": 1, "nxi": 10, "nl": 3},
        reference_nx=13,
        group_updates={
            "general": {"RHSMode": 2},
            "physicsParameters": {
                "Er": 0.0,
                "includeXDotTerm": False,
                "includeElectricFieldTermInXiDot": False,
                "useDKESExBDrift": True,
                "includePhi1": False,
                "collisionOperator": 1,
            },
            "otherNumericalParameters": {"Nxi_for_x_option": 0},
        },
    ),
}


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
    parser.add_argument("input_namelist", type=Path, nargs="?", help="Base SFINCS input.namelist for mapped candidates.")
    parser.add_argument(
        "--case",
        "--preset",
        dest="case",
        choices=tuple(sorted(CASE_PRESETS)),
        help="Named evidence preset. Use --list-cases to see the documented options.",
    )
    parser.add_argument("--list-cases", action="store_true", help="Print available presets and exit.")
    parser.add_argument("--json-out", type=Path, help="Output JSON evidence path.")
    parser.add_argument("--csv-out", type=Path, help="Output CSV evidence path.")
    parser.add_argument(
        "--log-lengths",
        type=_parse_log_lengths,
        default=None,
        help="Comma- or space-separated rational-tail log lengths.",
    )
    parser.add_argument("--reference-namelist", type=Path, default=None, help="Optional high-resolution reference input.")
    parser.add_argument("--reference-nx", type=int, default=None, help="Optional reference Nx built from the base input.")
    parser.add_argument("--candidate-nx", type=int, default=None, help="Optional mapped-candidate Nx override.")
    parser.add_argument("--candidate-nxi", type=int, default=None, help="Optional mapped-candidate Nxi override.")
    parser.add_argument("--candidate-nl", type=int, default=None, help="Optional mapped-candidate NL override.")
    parser.add_argument("--candidate-ntheta", type=int, default=None, help="Optional mapped-candidate Ntheta override.")
    parser.add_argument("--candidate-nzeta", type=int, default=None, help="Optional mapped-candidate Nzeta override.")
    parser.add_argument("--eta-kind", default="gauss", choices=("gauss", "uniform"))
    parser.add_argument("--derivative", default="barycentric")
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--restart", type=int, default=80)
    parser.add_argument("--maxiter", type=int, default=400)
    parser.add_argument("--solve-method", default=None, help="Optional solve_method override passed to the transport solver.")
    args = parser.parse_args(argv)
    if args.list_cases:
        return args
    if args.case is not None and args.input_namelist is not None:
        parser.error("input_namelist cannot be combined with --case/--preset")
    if args.input_namelist is None and args.case is None:
        parser.error("either input_namelist or --case/--preset is required")
    if args.json_out is None or args.csv_out is None:
        parser.error("--json-out and --csv-out are required unless --list-cases is used")
    return args


def _casefold_set(mapping: dict[str, object], key: str, value: object) -> None:
    key_lower = key.lower()
    for existing_key in list(mapping):
        if existing_key.lower() == key_lower:
            mapping[existing_key] = value
            return
    mapping[key] = value


def _copy_namelist_with_group_updates(
    nml: Namelist,
    updates: dict[str, dict[str, object]],
) -> Namelist:
    groups = {group_name: dict(values) for group_name, values in nml.groups.items()}
    for group_name, values in updates.items():
        target_name = next((name for name in groups if name.lower() == group_name.lower()), group_name.lower())
        group = dict(groups.get(target_name, {}))
        for key, value in values.items():
            _casefold_set(group, key, value)
        groups[target_name] = group
    indexed = {
        group_name: {key: dict(values) for key, values in group.items()}
        for group_name, group in nml.indexed.items()
    }
    return Namelist(groups=groups, indexed=indexed, source_path=nml.source_path, source_text=nml.source_text)


def _resolved_case(args: argparse.Namespace) -> CasePreset | None:
    if args.case is None:
        return None
    return CASE_PRESETS[str(args.case)]


def _resolved_input_path(args: argparse.Namespace, preset: CasePreset | None) -> Path:
    if preset is not None:
        return preset.input_namelist
    assert args.input_namelist is not None
    return args.input_namelist


def _resolved_log_lengths(args: argparse.Namespace, preset: CasePreset | None) -> tuple[float, ...]:
    if args.log_lengths is not None:
        return tuple(float(value) for value in args.log_lengths)
    if preset is not None:
        return preset.log_lengths
    return DEFAULT_LOG_LENGTHS


def _resolved_candidate_resolution(args: argparse.Namespace, preset: CasePreset | None) -> dict[str, int]:
    resolution = dict(preset.candidate_resolution) if preset is not None else {}
    explicit = {
        "nx": args.candidate_nx,
        "nxi": args.candidate_nxi,
        "nl": args.candidate_nl,
        "ntheta": args.candidate_ntheta,
        "nzeta": args.candidate_nzeta,
    }
    for key, value in explicit.items():
        if value is not None:
            resolution[key] = int(value)
    return resolution


def _resolved_reference_nx(args: argparse.Namespace, preset: CasePreset | None) -> int | None:
    if args.reference_nx is not None:
        return int(args.reference_nx)
    if args.reference_namelist is not None:
        return None
    if preset is not None:
        return int(preset.reference_nx)
    return None


def _portable_path(path: Path | None) -> str | None:
    if path is None:
        return None
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _metadata(
    args: argparse.Namespace,
    *,
    preset: CasePreset | None,
    input_path: Path,
    log_lengths: tuple[float, ...],
    candidate_resolution: dict[str, int],
    reference_nx: int | None,
) -> dict[str, object]:
    return {
        "case": None if args.case is None else str(args.case),
        "case_description": None if preset is None else preset.description,
        "input_namelist": _portable_path(input_path),
        "reference_namelist": _portable_path(args.reference_namelist),
        "reference_nx": reference_nx,
        "candidate_resolution": dict(candidate_resolution),
        "log_lengths": list(log_lengths),
        "eta_kind": str(args.eta_kind),
        "derivative": str(args.derivative),
        "eps": float(args.eps),
        "tol": float(args.tol),
        "atol": float(args.atol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
        "solve_method": None if args.solve_method is None else str(args.solve_method),
    }


def run_from_args(
    args: argparse.Namespace,
    *,
    solve_fn: Callable[..., object] = solve_v3_transport_matrix_linear_gmres,
) -> int:
    """Execute one evidence scan from parsed arguments."""

    if args.list_cases:
        for name, preset in CASE_PRESETS.items():
            print(f"{name}: {preset.description}")
        return 0

    if args.reference_namelist is not None and args.reference_nx is not None:
        raise ValueError("--reference-namelist and --reference-nx are mutually exclusive")

    preset = _resolved_case(args)
    input_path = _resolved_input_path(args, preset)
    log_lengths = _resolved_log_lengths(args, preset)
    candidate_resolution = _resolved_candidate_resolution(args, preset)
    reference_nx = _resolved_reference_nx(args, preset)

    nml = read_sfincs_input(input_path)
    if preset is not None and preset.group_updates:
        nml = _copy_namelist_with_group_updates(nml, preset.group_updates)
    if candidate_resolution:
        nml = copy_namelist_with_resolution(nml, **candidate_resolution)

    if args.reference_namelist is not None:
        reference_nml = read_sfincs_input(args.reference_namelist)
    elif reference_nx is not None:
        reference_nml = copy_namelist_with_resolution(nml, nx=int(reference_nx))
    else:
        reference_nml = None

    solve_kwargs: dict[str, object] = {
        "tol": float(args.tol),
        "atol": float(args.atol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
    }
    if args.solve_method is not None:
        solve_kwargs["solve_method"] = str(args.solve_method)

    report = run_rational_tail_transport_comparison(
        nml,
        log_length_values=log_lengths,
        reference_nml=reference_nml,
        solve_fn=solve_fn,
        eta_kind=str(args.eta_kind),
        derivative=str(args.derivative),
        eps=float(args.eps),
        solve_kwargs=solve_kwargs,
    )
    metadata = _metadata(
        args,
        preset=preset,
        input_path=input_path,
        log_lengths=log_lengths,
        candidate_resolution=candidate_resolution,
        reference_nx=reference_nx,
    )
    write_transport_evidence_json(report, args.json_out, metadata=metadata)
    write_transport_evidence_csv(report, args.csv_out)
    print(f"wrote {args.json_out}")
    print(f"wrote {args.csv_out}")
    print(f"best_by_transport_error_log_length={report.best_by_transport_error.log_length:.8g}")
    print(f"best_by_moment_log_length={report.best_by_moment.log_length:.8g}")
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
