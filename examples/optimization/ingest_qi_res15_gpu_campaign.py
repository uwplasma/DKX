#!/usr/bin/env python
"""Validate a bounded QI ``15x`` GPU campaign before ladder promotion."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.qi_res15_gpu_campaign import (  # noqa: E402
    DEFAULT_GPU_CPU_ROOT_ATOL,
    DEFAULT_GPU_FORTRAN_ROOT_ATOL,
    evaluate_qi_res15_gpu_campaign_files,
)


_DEFAULT_REFERENCE = (
    _REPO_ROOT
    / "docs"
    / "_static"
    / "figures"
    / "optimization"
    / "qi_nfp2_electron_root_res15_cpu_fortran_sparse_skip.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        required=True,
        help="Path to promotion_evidence_campaign.json from run_promotion_evidence_campaign.py.",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=_DEFAULT_REFERENCE,
        help="Checked res15 CPU/Fortran reference artifact.",
    )
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for the gated artifact.")
    parser.add_argument("--stem", default="qi_nfp2_electron_root_res15_gpu_campaign", help="Output JSON stem.")
    parser.add_argument("--gpu-cpu-root-atol", type=float, default=DEFAULT_GPU_CPU_ROOT_ATOL)
    parser.add_argument("--gpu-fortran-root-atol", type=float, default=DEFAULT_GPU_FORTRAN_ROOT_ATOL)
    parser.add_argument("--json", action="store_true", help="Print the gated artifact JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    artifact = evaluate_qi_res15_gpu_campaign_files(
        campaign_path=args.campaign,
        reference_path=args.reference,
        gpu_cpu_root_atol=float(args.gpu_cpu_root_atol),
        gpu_fortran_root_atol=float(args.gpu_fortran_root_atol),
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"{args.stem}.json"
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("QI res15 GPU campaign ingestion")
    print(f"  status: {artifact['status']}")
    print(f"  output: {output}")
    if artifact.get("failures"):
        print("  failures:")
        for failure in artifact["failures"]:
            print(f"    - {failure}")
    if args.json:
        print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if artifact["status"] == "pass_bounded_gpu_res15" else 2


if __name__ == "__main__":
    raise SystemExit(main())
