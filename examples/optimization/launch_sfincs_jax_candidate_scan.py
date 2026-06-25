#!/usr/bin/env python
"""Create or execute a SFINCS_JAX scan plan for an optimized candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.workflows.optimization_workflow import (  # noqa: E402
    build_candidate_scan_plan,
    er_values_from_bounds,
    load_proxy_summary,
    write_candidate_scan_plan,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy-summary", type=Path, required=True, help="Proxy optimization JSON summary.")
    parser.add_argument("--input", type=Path, required=True, help="SFINCS input.namelist template for scan-er.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for scan-er outputs.")
    parser.add_argument("--er-min", type=float, default=-3.0, help="Minimum Er scan value.")
    parser.add_argument("--er-max", type=float, default=3.0, help="Maximum Er scan value.")
    parser.add_argument("--n-er", type=int, default=7, help="Number of Er points.")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel scan worker processes.")
    parser.add_argument("--no-compute-solution", action="store_true", help="Do not request solution-derived fields.")
    parser.add_argument("--compute-transport-matrix", action="store_true", help="Also compute transport matrix outputs.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not reuse existing scan outputs.")
    parser.add_argument("--allow-no-electron-root", action="store_true", help="Do not require an electron root in promotion.")
    parser.add_argument("--impurity-species-index", type=int, help="Impurity species index for flux-selectivity audit.")
    parser.add_argument("--target-impurity-flux", type=float, default=0.0, help="Outward impurity flux target.")
    parser.add_argument("--promotion-stem", default="candidate_promotion", help="Promotion audit output stem.")
    parser.add_argument("--plan-json", type=Path, help="Plan JSON path. Defaults to OUT_DIR/candidate_scan_plan.json.")
    parser.add_argument("--execute", action="store_true", help="Execute scan-er after writing the plan.")
    parser.add_argument("--execute-promotion", action="store_true", help="Execute the promotion audit after scan-er.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    proxy_payload = load_proxy_summary(args.proxy_summary)
    er_values = er_values_from_bounds(er_min=args.er_min, er_max=args.er_max, n=args.n_er)
    plan = build_candidate_scan_plan(
        proxy_summary=args.proxy_summary,
        input_namelist=args.input,
        out_dir=args.out_dir,
        er_values=er_values,
        compute_solution=not bool(args.no_compute_solution),
        compute_transport_matrix=bool(args.compute_transport_matrix),
        jobs=int(args.jobs),
        skip_existing=not bool(args.no_skip_existing),
        promotion_stem=args.promotion_stem,
        require_electron_root=not bool(args.allow_no_electron_root),
        impurity_species_index=args.impurity_species_index,
        target_impurity_flux=float(args.target_impurity_flux),
    )
    plan_path = args.plan_json or (args.out_dir / "candidate_scan_plan.json")
    write_candidate_scan_plan(plan_path, plan, proxy_payload=proxy_payload)

    print("sfincs_jax candidate scan plan written")
    print(f"  plan:      {Path(plan_path).resolve()}")
    print(f"  scan:      {plan.as_dict()['scan_command_string']}")
    print(f"  promotion: {plan.as_dict()['promotion_command_string']}")

    if args.execute:
        subprocess.run(plan.scan_command, cwd=_REPO_ROOT, check=True)
    if args.execute_promotion:
        subprocess.run(plan.promotion_command, cwd=_REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
