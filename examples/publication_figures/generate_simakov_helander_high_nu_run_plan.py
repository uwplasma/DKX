#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AUDIT_JSON = (
    _REPO_ROOT
    / "examples"
    / "publication_figures"
    / "artifacts"
    / "sfincs_jax_simakov_helander_limit_audit_summary.json"
)
DEFAULT_PLAN_JSON = (
    _REPO_ROOT
    / "examples"
    / "publication_figures"
    / "artifacts"
    / "sfincs_jax_simakov_helander_high_nu_run_plan.json"
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_simakov_helander_high_nu_run_plan",
        description="Write concrete high-nu scan commands from the Simakov-Helander audit.",
    )
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_PLAN_JSON)
    parser.add_argument(
        "--work-root",
        type=Path,
        default=_REPO_ROOT / "examples" / "publication_figures" / "output" / "simakov_helander_high_nu",
    )
    parser.add_argument(
        "--summary-root",
        type=Path,
        default=_REPO_ROOT / "examples" / "publication_figures" / "artifacts" / "simakov_helander_high_nu",
    )
    parser.add_argument("--timeout-s", type=float, default=3600.0)
    parser.add_argument("--collision-operators", default="0,1")
    parser.add_argument("--python", default="python")
    parser.add_argument(
        "--transport-workers",
        type=int,
        default=2,
        help="Process workers for independent whichRHS transport solves in the generated commands.",
    )
    parser.add_argument(
        "--transport-parallel-backend",
        choices=("auto", "cpu", "gpu"),
        default="gpu",
        help="Parallel backend for generated high-nu commands.",
    )
    parser.add_argument(
        "--transport-sparse-direct-max",
        type=int,
        default=30000,
        help=(
            "Sparse-direct size cap for high-nu transport solves. The default keeps "
            "the LHD pilot on the accurate direct path while preventing W7-X FP "
            "from silently entering an oversized host sparse-LU factorization."
        ),
    )
    parser.add_argument(
        "--transport-maxiter",
        type=int,
        default=0,
        help="Optional SFINCS_JAX_TRANSPORT_MAXITER override for generated commands.",
    )
    parser.add_argument(
        "--max-transport-residual",
        type=float,
        default=1.0e-6,
        help="Residual gate used when generated commands reuse existing outputs.",
    )
    parser.add_argument(
        "--max-transport-relative-residual",
        type=float,
        default=1.0e-6,
        help="Relative residual gate used when generated commands reuse existing outputs.",
    )
    return parser


def _rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(path)


def build_high_nu_run_plan(
    audit_payload: dict[str, object],
    *,
    work_root: Path,
    summary_root: Path,
    timeout_s: float,
    collision_operators: str = "0,1",
    python_executable: str = "python",
    transport_workers: int = 2,
    transport_parallel_backend: str = "gpu",
    transport_sparse_direct_max: int = 30000,
    transport_maxiter: int = 0,
    max_transport_residual: float = 1.0e-6,
    max_transport_relative_residual: float = 1.0e-6,
) -> dict[str, object]:
    cases = audit_payload.get("cases", {})
    if not isinstance(cases, dict):
        raise ValueError("Audit payload must contain a 'cases' object.")

    runs: list[dict[str, object]] = []
    for case_name in ("lhd", "w7x"):
        case_payload = cases.get(case_name, {})
        if not isinstance(case_payload, dict):
            continue
        extension = [float(v) for v in case_payload.get("recommended_high_nuprime_extension", [])]
        if not extension:
            continue
        nuprime_min = float(extension[0])
        nuprime_max = float(extension[-1])
        n_points = len(extension)
        work_dir = Path(work_root) / case_name
        summary_dir = Path(summary_root) / case_name
        command = [
            python_executable,
            "examples/publication_figures/generate_sfincs_paper_figs.py",
            "--case",
            case_name,
            "--collision-operators",
            str(collision_operators),
            "--nuprime-min",
            f"{nuprime_min:.16g}",
            "--nuprime-max",
            f"{nuprime_max:.16g}",
            "--n-points",
            str(n_points),
            "--work-dir",
            _rel(work_dir),
            "--summary-dir",
            _rel(summary_dir),
            "--timeout-s",
            f"{float(timeout_s):.16g}",
            "--transport-workers",
            str(max(1, int(transport_workers))),
            "--transport-parallel-backend",
            str(transport_parallel_backend),
            "--transport-sparse-direct-max",
            str(max(0, int(transport_sparse_direct_max))),
            "--require-residuals",
            "--max-transport-residual",
            f"{float(max_transport_residual):.16g}",
            "--max-transport-relative-residual",
            f"{float(max_transport_relative_residual):.16g}",
            "--skip-existing",
            "--scan-only",
        ]
        if int(transport_maxiter) > 0:
            command.extend(["--transport-maxiter", str(int(transport_maxiter))])
        pilot_command = [
            python_executable,
            "examples/publication_figures/generate_sfincs_paper_figs.py",
            "--case",
            case_name,
            "--collision-operators",
            "0",
            "--nuprime-min",
            f"{nuprime_min:.16g}",
            "--nuprime-max",
            f"{nuprime_min:.16g}",
            "--n-points",
            "1",
            "--work-dir",
            _rel(work_dir / "pilot_fp"),
            "--summary-dir",
            _rel(summary_dir / "pilot_fp"),
            "--timeout-s",
            f"{float(timeout_s):.16g}",
            "--transport-workers",
            str(max(1, int(transport_workers))),
            "--transport-parallel-backend",
            str(transport_parallel_backend),
            "--transport-sparse-direct-max",
            str(max(0, int(transport_sparse_direct_max))),
            "--require-residuals",
            "--max-transport-residual",
            f"{float(max_transport_residual):.16g}",
            "--max-transport-relative-residual",
            f"{float(max_transport_relative_residual):.16g}",
            "--skip-existing",
            "--scan-only",
        ]
        if int(transport_maxiter) > 0:
            pilot_command.extend(["--transport-maxiter", str(int(transport_maxiter))])
        runs.append(
            {
                "case": case_name,
                "current_max_nuprime": float(case_payload.get("max_nuprime", 0.0)),
                "recommended_high_nuprime_extension": extension,
                "nuprime_min": nuprime_min,
                "nuprime_max": nuprime_max,
                "n_points": n_points,
                "work_dir": _rel(work_dir),
                "summary_dir": _rel(summary_dir),
                "command": command,
                "shell_command": " ".join(command),
                "pilot_command": pilot_command,
                "pilot_shell_command": " ".join(pilot_command),
            }
        )

    return {
        "metadata": {
            "schema_version": 1,
            "kind": "simakov_helander_high_nu_run_plan",
            "source_audit": _rel(DEFAULT_AUDIT_JSON),
            "notes": [
                "This is an executable run plan, not a completed validation artifact.",
                "Run these scan-only commands on a workstation or nightly lane, then regenerate the Simakov-Helander audit.",
                "Start with each pilot_command to estimate wall time before launching the full FP/PAS extension.",
                "The residual gates are intentionally strict; if W7-X FP high-nu fails, treat it as a preconditioner lane, not a completed validation.",
            ],
        },
        "configuration": {
            "timeout_s": float(timeout_s),
            "collision_operators": str(collision_operators),
            "transport_workers": max(1, int(transport_workers)),
            "transport_parallel_backend": str(transport_parallel_backend),
            "transport_sparse_direct_max": max(0, int(transport_sparse_direct_max)),
            "transport_maxiter": max(0, int(transport_maxiter)),
            "require_residuals": True,
            "max_transport_residual": float(max_transport_residual),
            "max_transport_relative_residual": float(max_transport_relative_residual),
        },
        "runs": runs,
        "ready_to_run": bool(runs),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    audit_json = Path(args.audit_json)
    payload = json.loads(audit_json.read_text())
    plan = build_high_nu_run_plan(
        payload,
        work_root=Path(args.work_root),
        summary_root=Path(args.summary_root),
        timeout_s=float(args.timeout_s),
        collision_operators=str(args.collision_operators),
        python_executable=str(args.python),
        transport_workers=int(args.transport_workers),
        transport_parallel_backend=str(args.transport_parallel_backend),
        transport_sparse_direct_max=int(args.transport_sparse_direct_max),
        transport_maxiter=int(args.transport_maxiter),
        max_transport_residual=float(args.max_transport_residual),
        max_transport_relative_residual=float(args.max_transport_relative_residual),
    )
    plan["metadata"]["source_audit"] = _rel(audit_json)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print(f"Wrote Simakov-Helander high-nu run plan to {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
