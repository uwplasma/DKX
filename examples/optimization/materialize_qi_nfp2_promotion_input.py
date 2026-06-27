#!/usr/bin/env python
"""Materialize a low-resolution two-species QI nfp=2 promotion input."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.namelist import read_sfincs_input  # noqa: E402

_DEFAULT_SOURCE = _REPO_ROOT / "examples" / "data" / "qi_nfp2_reference.input.namelist"
_DEFAULT_STEM = "qi_nfp2_two_species_lowres_promotion"
_CLAIM_BOUNDARY = (
    "This is a low-resolution two-species QI nfp=2 kinetic promotion candidate, "
    "not electron-root evidence. A kinetic electron-root claim requires completed "
    "CPU/GPU/Fortran Er scans plus residual, ambipolar-root, backend-comparison, "
    "and resolution-ladder gates."
)


def _group_bounds(txt: str, group: str) -> tuple[int, int]:
    start = re.search(rf"(?im)^\s*&{re.escape(group)}\s*$", txt)
    if start is None:
        raise ValueError(f"Missing namelist group &{group}")
    end = re.search(r"(?m)^\s*/\s*$", txt[start.end() :])
    if end is None:
        raise ValueError(f"Missing '/' terminator for &{group}")
    return start.end(), start.end() + end.start()


def _patch_assignment(txt: str, *, group: str, key: str, value: str) -> str:
    start, end = _group_bounds(txt, group)
    body = txt[start:end]
    line = f"  {key} = {value}"
    pat = re.compile(rf"(?im)^[ \t]*{re.escape(key)}[ \t]*=[^\n\r]*$")
    match = pat.search(body)
    if match is None:
        if not body.endswith("\n"):
            body += "\n"
        body = body + line + "\n"
    else:
        body = body[: match.start()] + line + body[match.end() :]
    return txt[:start] + body + txt[end:]


def _patch_scan_comments(txt: str) -> str:
    txt = re.sub(r"(?im)^!ss\s+scanType\s*=.*$", "!ss scanType = 1", txt, count=1)
    txt = re.sub(r"(?im)^!ss\s+runSpecFile\s*=.*\n?", "", txt, count=1)
    if not re.search(r"(?im)^!ss\s+scanType\s*=", txt):
        txt = "!ss scanType = 1\n" + txt
    return txt


def materialize_input_text(
    source_text: str,
    *,
    n_theta: int,
    n_zeta: int,
    n_xi: int,
    n_x: int,
    solver_tolerance: str,
    er: float,
    equilibrium_file: str | None = None,
) -> str:
    """Return the bounded low-resolution two-species promotion input text."""

    txt = _patch_scan_comments(source_text)
    if equilibrium_file is not None:
        txt = _patch_assignment(
            txt,
            group="geometryParameters",
            key="equilibriumFile",
            value=json.dumps(str(equilibrium_file)),
        )

    species = {
        "Zs": "1.0d+0 -1.0d+0",
        "mHats": "1.0d+0 5.446170214d-4",
        "nHats": "1.0d+0 1.0d+0",
        "THats": "1.0d+0 1.0d+0",
        "dnHatdrHats": "1.0d+0 1.0d+0",
        "dTHatdrHats": "1.0d+0 1.0d+0",
    }
    for key, value in species.items():
        txt = _patch_assignment(txt, group="speciesParameters", key=key, value=value)

    txt = _patch_assignment(txt, group="physicsParameters", key="Er", value=f"{float(er):.16g}")
    resolution = {
        "Ntheta": str(int(n_theta)),
        "Nzeta": str(int(n_zeta)),
        "Nxi": str(int(n_xi)),
        "Nx": str(int(n_x)),
        "solverTolerance": str(solver_tolerance),
    }
    for key, value in resolution.items():
        txt = _patch_assignment(txt, group="resolutionParameters", key=key, value=value)

    if not txt.endswith("\n"):
        txt += "\n"
    banner = (
        "! Materialized by examples/optimization/materialize_qi_nfp2_promotion_input.py\n"
        f"! Claim boundary: {_CLAIM_BOUNDARY}\n"
    )
    return banner + txt


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=_DEFAULT_SOURCE, help="Source QI nfp=2 input.namelist.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for the generated input and JSON.")
    parser.add_argument("--stem", default=_DEFAULT_STEM, help="Output stem.")
    parser.add_argument(
        "--equilibrium-file",
        help="Optional equilibriumFile override. By default the source QI nfp=2 reference is preserved.",
    )
    parser.add_argument("--er", type=float, default=0.0, help="Nominal Er value before scan-er patches scan points.")
    parser.add_argument("--n-theta", type=int, default=7, help="Low-resolution Ntheta.")
    parser.add_argument("--n-zeta", type=int, default=7, help="Low-resolution Nzeta.")
    parser.add_argument("--n-xi", type=int, default=7, help="Low-resolution Nxi.")
    parser.add_argument("--n-x", type=int, default=4, help="Low-resolution Nx.")
    parser.add_argument("--solver-tolerance", default="1d-6", help="Low-resolution solverTolerance.")
    parser.add_argument("--json", action="store_true", help="Print the JSON summary.")
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    for attr in ("n_theta", "n_zeta", "n_xi", "n_x"):
        if int(getattr(args, attr)) < 1:
            raise ValueError(f"{attr.replace('_', '-')} must be >= 1")
    if not args.source.exists():
        raise FileNotFoundError(str(args.source))


def _summary(
    *,
    source: Path,
    input_path: Path,
    summary_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "workflow": "sfincs_jax_qi_nfp2_kinetic_promotion_input",
        "claim_boundary": _CLAIM_BOUNDARY,
        "candidate": {
            "symmetry": "QI",
            "nfp": 2,
            "species": "ion_electron",
            "resolution": "low",
        },
        "source_input": str(source.resolve()),
        "input_namelist": str(input_path.resolve()),
        "summary_json": str(summary_path.resolve()),
        "resolution": {
            "Ntheta": int(args.n_theta),
            "Nzeta": int(args.n_zeta),
            "Nxi": int(args.n_xi),
            "Nx": int(args.n_x),
            "solverTolerance": str(args.solver_tolerance),
        },
        "species_parameters": {
            "Zs": [1.0, -1.0],
            "mHats": [1.0, 0.000545509],
            "nHats": [1.0, 1.0],
            "THats": [1.0, 1.0],
            "dnHatdrHats": [1.0, 1.0],
            "dTHatdrHats": [1.0, 1.0],
        },
        "next_commands": [
            (
                "python examples/optimization/run_promotion_evidence_campaign.py "
                f"--input {input_path.resolve()} --out-dir <campaign-dir> "
                "--values -0.3 -0.1 0 0.1 0.3 1 2 3 --run-cpu --run-gpu --run-fortran --dry-run"
            ),
            (
                "python examples/optimization/evaluate_sfincs_jax_promotion_scan.py "
                "--scan-dir <completed-scan-dir> --out-dir <audit-dir> "
                "--stem qi_nfp2_promotion --require-electron-root"
            ),
        ],
        "required_gates": [
            "completed CPU scan-er campaign",
            "completed GPU scan-er campaign",
            "completed Fortran-v3 Er scan over the same points",
            "linear residual and ambipolar electron-root gates",
            "CPU/GPU/Fortran comparison and resolution-ladder gates",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_args(args)
    source = args.source.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    input_path = out_dir / f"{args.stem}.input.namelist"
    summary_path = out_dir / f"{args.stem}.json"

    text = materialize_input_text(
        source.read_text(encoding="utf-8"),
        n_theta=int(args.n_theta),
        n_zeta=int(args.n_zeta),
        n_xi=int(args.n_xi),
        n_x=int(args.n_x),
        solver_tolerance=str(args.solver_tolerance),
        er=float(args.er),
        equilibrium_file=args.equilibrium_file,
    )
    input_path.write_text(text, encoding="utf-8")
    read_sfincs_input(input_path)

    payload = _summary(source=source, input_path=input_path, summary_path=summary_path, args=args)
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("QI nfp=2 promotion input materialized")
        print(f"  input:   {input_path}")
        print(f"  summary: {summary_path}")
        print(f"  claim boundary: {_CLAIM_BOUNDARY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
