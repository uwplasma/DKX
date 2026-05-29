#!/usr/bin/env python
"""Run CPU/GPU/Fortran promotion evidence for an optimization candidate."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.optimization_evidence import (  # noqa: E402
    PromotionEvidenceLane,
    build_promotion_evidence_plan,
    run_fortran_er_scan,
    write_promotion_evidence_plan,
)
from sfincs_jax.optimization_workflow import er_values_from_bounds  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="SFINCS input.namelist template.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Campaign output directory.")
    parser.add_argument("--er-min", type=float, default=-3.0, help="Minimum Er scan value.")
    parser.add_argument("--er-max", type=float, default=3.0, help="Maximum Er scan value.")
    parser.add_argument("--n-er", type=int, default=7, help="Number of Er scan points.")
    parser.add_argument("--values", type=float, nargs="+", help="Explicit Er scan values; overrides bounds.")
    parser.add_argument("--jobs", type=int, default=1, help="JAX scan worker processes.")
    parser.add_argument("--run-cpu", action="store_true", help="Run the sfincs_jax CPU lane.")
    parser.add_argument("--run-gpu", action="store_true", help="Run the sfincs_jax GPU lane.")
    parser.add_argument("--gpu-device", help="CUDA_VISIBLE_DEVICES value for the GPU lane.")
    parser.add_argument("--run-fortran", action="store_true", help="Run the SFINCS Fortran v3 lane.")
    parser.add_argument("--fortran-exe", type=Path, help="SFINCS Fortran v3 executable.")
    parser.add_argument("--fortran-timeout-s", type=float, default=600.0, help="Per-point Fortran timeout.")
    parser.add_argument(
        "--jax-scan-timeout-s",
        type=float,
        help="Wall-time timeout for each sfincs_jax CPU/GPU scan lane subprocess.",
    )
    parser.add_argument(
        "--promotion-timeout-s",
        type=float,
        help="Wall-time timeout for each promotion-audit subprocess.",
    )
    parser.add_argument(
        "--continue-on-lane-failure",
        action="store_true",
        help="Keep running independent lanes after one lane fails; campaign exits nonzero and records failures.",
    )
    parser.add_argument(
        "--require-fortran-residuals",
        action="store_true",
        help=(
            "Require linear residual datasets in Fortran-v3 promotion outputs. "
            "By default these are allowed to be absent because upstream v3 HDF5 "
            "files often do not contain JAX residual diagnostics."
        ),
    )
    parser.add_argument("--no-compute-solution", action="store_true", help="Do not request solution outputs.")
    parser.add_argument("--compute-transport-matrix", action="store_true", help="Request transport matrix outputs.")
    parser.add_argument("--no-skip-existing", action="store_true", help="Do not reuse existing scan outputs.")
    parser.add_argument("--allow-no-electron-root", action="store_true", help="Do not require an electron root.")
    parser.add_argument("--impurity-species-index", type=int, help="Impurity species index for flux-selectivity audit.")
    parser.add_argument("--target-impurity-flux", type=float, default=0.0, help="Outward impurity flux target.")
    parser.add_argument("--promotion-stem", default="candidate_promotion", help="Promotion output stem.")
    parser.add_argument("--comparison-stem", default="candidate_promotion_comparison", help="Comparison output stem.")
    parser.add_argument("--dry-run", action="store_true", help="Only write the JSON command plan.")
    parser.add_argument("--no-compare", action="store_true", help="Skip the CPU/GPU/Fortran comparison step.")
    parser.add_argument("--json", action="store_true", help="Print the final campaign JSON.")
    return parser


def _emit(message: str) -> None:
    print(message, flush=True)


class LaneCommandError(RuntimeError):
    """Wrap one failed lane subprocess with the campaign stage that failed."""

    def __init__(self, stage: str, original: BaseException) -> None:
        self.stage = str(stage)
        self.original = original
        super().__init__(f"{self.stage}: {type(original).__name__}: {original}")


def _run_command(
    command: tuple[str, ...],
    *,
    env_delta: dict[str, str],
    timeout_s: float | None = None,
) -> None:
    env = os.environ.copy()
    env.update(env_delta)
    subprocess.run(command, cwd=_REPO_ROOT, env=env, check=True, timeout=timeout_s)


def _run_jax_lane(
    lane: PromotionEvidenceLane,
    *,
    promotion_stem: str,
    scan_timeout_s: float | None,
    promotion_timeout_s: float | None,
) -> dict[str, object]:
    if lane.scan_command is None:
        raise ValueError(f"{lane.label} lane has no JAX scan command")
    _emit(f"[{lane.label}] scan command: {' '.join(lane.scan_command)}")
    try:
        _run_command(lane.scan_command, env_delta=lane.env, timeout_s=scan_timeout_s)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        raise LaneCommandError("scan", exc) from exc
    _emit(f"[{lane.label}] promotion command: {' '.join(lane.promotion_command)}")
    try:
        _run_command(lane.promotion_command, env_delta=lane.env, timeout_s=promotion_timeout_s)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        raise LaneCommandError("promotion", exc) from exc
    return _lane_result(lane, promotion_stem=promotion_stem)


def _run_fortran_lane(
    lane: PromotionEvidenceLane,
    *,
    input_namelist: Path,
    er_values: tuple[float, ...],
    fortran_exe: Path | None,
    timeout_s: float,
    promotion_timeout_s: float | None,
    skip_existing: bool,
    promotion_stem: str,
) -> dict[str, object]:
    _emit(f"[{lane.label}] running Fortran v3 scan in {lane.scan_dir}")
    run_fortran_er_scan(
        input_namelist=input_namelist,
        out_dir=lane.scan_dir,
        values=er_values,
        exe=fortran_exe,
        timeout_s=float(timeout_s),
        skip_existing=skip_existing,
        emit=lambda _level, msg: _emit(msg),
    )
    _emit(f"[{lane.label}] promotion command: {' '.join(lane.promotion_command)}")
    try:
        _run_command(lane.promotion_command, env_delta=lane.env, timeout_s=promotion_timeout_s)
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, OSError) as exc:
        raise LaneCommandError("promotion", exc) from exc
    return _lane_result(lane, promotion_stem=promotion_stem)


def _lane_result(lane: PromotionEvidenceLane, *, promotion_stem: str) -> dict[str, object]:
    promotion_json = lane.promotion_dir / f"{promotion_stem}.json"
    if not promotion_json.exists():
        matches = sorted(lane.promotion_dir.glob("*.json"))
        promotion_json = matches[0] if matches else promotion_json
    payload = json.loads(promotion_json.read_text(encoding="utf-8"))
    return {
        "label": lane.label,
        "backend": lane.backend,
        "status": "pass",
        "promotion_json": str(promotion_json.resolve()),
        "gate_status": payload.get("gate_status"),
        "failures": payload.get("failures", []),
        "selected_root": payload.get("selected_root"),
    }


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def _lane_failure_result(
    lane: PromotionEvidenceLane,
    *,
    stage: str,
    exc: BaseException,
) -> dict[str, object]:
    command = getattr(exc, "cmd", None)
    result: dict[str, object] = {
        "label": lane.label,
        "backend": lane.backend,
        "status": "fail",
        "stage": str(stage),
        "error_type": type(exc).__name__,
        "error": str(exc),
        "command": [str(part) for part in command] if isinstance(command, (list, tuple)) else command,
    }
    if isinstance(exc, subprocess.TimeoutExpired):
        result["failure_kind"] = "timeout"
        result["timeout_s"] = float(exc.timeout) if exc.timeout is not None else None
        result["stdout_tail"] = _text(exc.stdout)[-4000:]
        result["stderr_tail"] = _text(exc.stderr)[-4000:]
    elif isinstance(exc, subprocess.CalledProcessError):
        result["failure_kind"] = "returncode"
        result["returncode"] = int(exc.returncode)
        result["stdout_tail"] = _text(exc.stdout)[-4000:]
        result["stderr_tail"] = _text(exc.stderr)[-4000:]
    else:
        result["failure_kind"] = "exception"
    return result


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    run_cpu = bool(args.run_cpu)
    run_gpu = bool(args.run_gpu)
    run_fortran = bool(args.run_fortran)
    if not (run_cpu or run_gpu or run_fortran):
        run_cpu = True

    er_values = (
        tuple(float(value) for value in args.values)
        if args.values is not None
        else er_values_from_bounds(er_min=args.er_min, er_max=args.er_max, n=args.n_er)
    )
    plan = build_promotion_evidence_plan(
        input_namelist=args.input,
        out_dir=args.out_dir,
        er_values=er_values,
        include_cpu=run_cpu,
        include_gpu=run_gpu,
        include_fortran=run_fortran,
        fortran_exe=args.fortran_exe,
        gpu_device=args.gpu_device,
        jobs=int(args.jobs),
        compute_solution=not bool(args.no_compute_solution),
        compute_transport_matrix=bool(args.compute_transport_matrix),
        skip_existing=not bool(args.no_skip_existing),
        require_electron_root=not bool(args.allow_no_electron_root),
        impurity_species_index=args.impurity_species_index,
        target_impurity_flux=float(args.target_impurity_flux),
        require_fortran_residuals=bool(args.require_fortran_residuals),
        promotion_stem=args.promotion_stem,
        compare_stem=args.comparison_stem,
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    plan_path = write_promotion_evidence_plan(out_dir / "promotion_evidence_plan.json", plan)
    _emit(f"promotion evidence plan written: {plan_path}")

    result: dict[str, object] = {
        **plan.as_dict(),
        "plan_json": str(plan_path),
        "executed": not bool(args.dry_run),
        "campaign_status": "planned" if bool(args.dry_run) else "running",
        "jax_scan_timeout_s": None if args.jax_scan_timeout_s is None else float(args.jax_scan_timeout_s),
        "promotion_timeout_s": None if args.promotion_timeout_s is None else float(args.promotion_timeout_s),
        "continue_on_lane_failure": bool(args.continue_on_lane_failure),
        "lane_results": [],
        "campaign_failures": [],
        "comparison_result": None,
    }
    if args.dry_run:
        if args.json:
            print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    lane_results: list[dict[str, object]] = []
    campaign_failures: list[dict[str, object]] = []
    for lane in plan.lanes:
        try:
            if lane.backend == "fortran_v3":
                lane_results.append(
                    _run_fortran_lane(
                        lane,
                        input_namelist=args.input.resolve(),
                        er_values=er_values,
                        fortran_exe=args.fortran_exe,
                        timeout_s=float(args.fortran_timeout_s),
                        promotion_timeout_s=args.promotion_timeout_s,
                        skip_existing=not bool(args.no_skip_existing),
                        promotion_stem=args.promotion_stem,
                    )
                )
            else:
                lane_results.append(
                    _run_jax_lane(
                        lane,
                        promotion_stem=args.promotion_stem,
                        scan_timeout_s=args.jax_scan_timeout_s,
                        promotion_timeout_s=args.promotion_timeout_s,
                    )
                )
        except LaneCommandError as exc:
            failure = _lane_failure_result(lane, stage=exc.stage, exc=exc.original)
            lane_results.append(failure)
            campaign_failures.append(failure)
            _emit(f"[{lane.label}] failed during {exc.stage}: {type(exc.original).__name__}: {exc.original}")
            if not bool(args.continue_on_lane_failure):
                break
        except (OSError, RuntimeError, ValueError) as exc:
            failure = _lane_failure_result(lane, stage="setup", exc=exc)
            lane_results.append(failure)
            campaign_failures.append(failure)
            _emit(f"[{lane.label}] failed during setup: {type(exc).__name__}: {exc}")
            if not bool(args.continue_on_lane_failure):
                break
    result["lane_results"] = lane_results
    result["campaign_failures"] = campaign_failures

    if plan.comparison_command is not None and not bool(args.no_compare) and not campaign_failures:
        _emit(f"[comparison] command: {' '.join(plan.comparison_command)}")
        completed = subprocess.run(
            plan.comparison_command,
            cwd=_REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        result["comparison_result"] = {
            "returncode": int(completed.returncode),
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        print(completed.stdout, end="")
        print(completed.stderr, end="", file=sys.stderr)
        if completed.returncode != 0:
            _emit("[comparison] failed; keeping lane artifacts for inspection")
    elif campaign_failures and plan.comparison_command is not None and not bool(args.no_compare):
        result["comparison_result"] = {
            "returncode": None,
            "skipped": True,
            "reason": "lane_failure",
        }

    result["campaign_status"] = "fail" if campaign_failures else "pass"

    summary_path = out_dir / "promotion_evidence_campaign.json"
    summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _emit(f"promotion evidence campaign summary: {summary_path}")
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    comparison = result.get("comparison_result")
    comparison_returncode = comparison.get("returncode", 0) if isinstance(comparison, dict) else 0
    if comparison_returncode is not None and int(comparison_returncode) != 0:
        return 2
    if campaign_failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
