#!/usr/bin/env python
"""Bounded benchmark harness for structured RHSMode=1 f-block preconditioners."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import sys
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = _REPO_ROOT / "outputs" / "structured_fblock_preconditioner_benchmark.json"
RESULT_MARKER = "__SFINCS_JAX_STRUCTURED_FBLOCK_RESULT__="

CASES = {
    "fp_phi1_tiny": _REPO_ROOT
    / "tests"
    / "ref"
    / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist",
    "quick_fp": _REPO_ROOT / "tests" / "ref" / "quick_2species_FPCollisions_noEr.input.namelist",
    "tokamak_fp_phi1_medium": _REPO_ROOT
    / "examples"
    / "sfincs_examples"
    / "tokamak_1species_FPCollisions_noEr_withPhi1InDKE"
    / "input.namelist",
}

PRECONDITIONERS = {
    "fp_radial": "structured_fblock_fp_radial_jacobi",
    "fp_lowmode_schur": "structured_fblock_fp_lowmode_schur",
    "fp_moment_schur": "structured_fblock_fp_moment_schur",
    "fp_coupled_moment_schur": "structured_fblock_fp_coupled_moment_schur",
    "fp_tail_coupled_schur": "structured_fblock_fp_tail_coupled_schur",
}


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be finite and > 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--cases", nargs="+", choices=tuple(CASES), default=["fp_phi1_tiny", "quick_fp"])
    parser.add_argument(
        "--preconditioners",
        nargs="+",
        choices=tuple(PRECONDITIONERS),
        default=["fp_radial", "fp_lowmode_schur"],
    )
    parser.add_argument(
        "--solve-cases",
        nargs="*",
        choices=tuple(CASES),
        default=["fp_phi1_tiny"],
        help="Cases that also run a bounded Krylov solve. Empty disables solve rows.",
    )
    parser.add_argument("--timeout-s", type=_positive_float, default=45.0)
    parser.add_argument("--identity-shift", type=float, default=0.5)
    parser.add_argument("--tol", type=_positive_float, default=1.0e-8)
    parser.add_argument("--restart", type=_positive_int, default=20)
    parser.add_argument("--maxiter", type=_positive_int, default=40)
    parser.add_argument("--solve-method", default="incremental")
    parser.add_argument(
        "--warm-repeats",
        type=_positive_int,
        default=2,
        help="Number of same-process warm preconditioner applications to time after the cold build/apply.",
    )
    parser.add_argument(
        "--solve-repeats",
        type=_positive_int,
        default=1,
        help="Number of same-process solve repeats for cases listed in --solve-cases.",
    )
    parser.add_argument("--max-solve-residual", type=_positive_float, default=1.0e-8)
    parser.add_argument("--min-dke-improvement", type=_positive_float, default=1.05)
    parser.add_argument(
        "--max-warm-runtime-ratio",
        type=_positive_float,
        default=1.0,
        help="Candidate warm runtime must be at or below this baseline ratio for auto-promotion.",
    )
    parser.add_argument(
        "--max-rss-ratio",
        type=_positive_float,
        default=1.25,
        help="Candidate peak RSS must be at or below this baseline ratio for auto-promotion.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=tuple(CASES), help=argparse.SUPPRESS)
    parser.add_argument("--preconditioner", choices=tuple(PRECONDITIONERS), help=argparse.SUPPRESS)
    parser.add_argument("--run-solve", action="store_true", help=argparse.SUPPRESS)
    return parser


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def _preconditioner_env(preconditioner: str) -> dict[str, str]:
    return {
        "SFINCS_JAX_FORTRAN_STDOUT": "0",
        "SFINCS_JAX_SOLVER_ITER_STATS": "0",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND": "0",
        "SFINCS_JAX_RHSMODE1_DENSE_FALLBACK_MAX": "0",
        "SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND": "off",
        "SFINCS_JAX_RHSMODE1_PRECONDITIONER": PRECONDITIONERS[str(preconditioner)],
    }


def _residual_ratios(op, rhs_full, candidate_full) -> tuple[float, float]:
    import jax.numpy as jnp

    from sfincs_jax.operators.profile_response.system import apply_v3_full_system_operator

    rhs_full = jnp.asarray(rhs_full, dtype=jnp.float64)
    candidate_full = jnp.asarray(candidate_full, dtype=jnp.float64)
    residual = rhs_full - apply_v3_full_system_operator(op, candidate_full)
    dke_denom = jnp.linalg.norm(rhs_full[: op.f_size])
    full_denom = jnp.linalg.norm(rhs_full)
    dke_ratio = jnp.linalg.norm(residual[: op.f_size]) / (dke_denom + 1.0e-300)
    full_ratio = jnp.linalg.norm(residual) / (full_denom + 1.0e-300)
    return float(dke_ratio), float(full_ratio)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _build_preconditioner(op, preconditioner: str):
    import sfincs_jax.v3_driver as vd

    if preconditioner == "fp_radial":
        return vd._build_rhsmode1_structured_fblock_fp_radial_jacobi_preconditioner(op=op)
    if preconditioner == "fp_lowmode_schur":
        return vd._build_rhsmode1_structured_fblock_fp_lowmode_schur_preconditioner(op=op)
    if preconditioner == "fp_moment_schur":
        return vd._build_rhsmode1_structured_fblock_fp_moment_schur_preconditioner(op=op)
    if preconditioner == "fp_coupled_moment_schur":
        return vd._build_rhsmode1_structured_fblock_fp_coupled_moment_schur_preconditioner(op=op)
    if preconditioner == "fp_tail_coupled_schur":
        return vd._build_rhsmode1_structured_fblock_fp_tail_coupled_schur_preconditioner(op=op)
    raise ValueError(f"unknown preconditioner {preconditioner!r}")


def run_child_payload(args: argparse.Namespace) -> dict[str, Any]:
    import jax
    import jax.numpy as jnp

    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.operators.profile_response.system import full_system_operator_from_namelist
    from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres

    case = str(args.case)
    preconditioner = str(args.preconditioner)
    input_path = CASES[case]
    for key, value in _preconditioner_env(preconditioner).items():
        os.environ[key] = value

    nml = read_sfincs_input(input_path)
    build_operator_start = time.perf_counter()
    op = full_system_operator_from_namelist(nml=nml, identity_shift=float(args.identity_shift))
    jax.block_until_ready(jnp.asarray(float(op.total_size)))
    operator_build_s = time.perf_counter() - build_operator_start

    build_precond_start = time.perf_counter()
    precond = _build_preconditioner(op, preconditioner)
    metadata = getattr(precond, "_sfincs_jax_structured_fblock_metadata", {})
    rhs_f = jnp.linspace(-0.35, 0.65, int(op.f_size), dtype=jnp.float64)
    zeros_tail = jnp.zeros((int(op.total_size) - int(op.f_size),), dtype=jnp.float64)
    rhs_full = jnp.concatenate([rhs_f, zeros_tail], axis=0)
    candidate = precond(rhs_full)
    jax.block_until_ready(candidate)
    preconditioner_build_apply_s = time.perf_counter() - build_precond_start

    one_step_start = time.perf_counter()
    dke_ratio, full_ratio = _residual_ratios(op, rhs_full, candidate)
    one_step_s = time.perf_counter() - one_step_start

    warm_apply_times_s: list[float] = []
    warm_dke_ratio: float | None = None
    for repeat_index in range(int(args.warm_repeats)):
        repeat_offset = jnp.asarray(1.0e-3 * float(repeat_index + 1), dtype=jnp.float64)
        warm_rhs_f = rhs_f + repeat_offset
        warm_rhs_full = jnp.concatenate([warm_rhs_f, zeros_tail], axis=0)
        warm_apply_start = time.perf_counter()
        warm_candidate = precond(warm_rhs_full)
        jax.block_until_ready(warm_candidate)
        warm_apply_times_s.append(time.perf_counter() - warm_apply_start)
        if repeat_index == int(args.warm_repeats) - 1:
            warm_dke_ratio, _ = _residual_ratios(op, warm_rhs_full, warm_candidate)

    solve_payload: dict[str, Any] | None = None
    if bool(args.run_solve):
        solve_repeats: list[dict[str, Any]] = []
        selected_logs: list[str] = []
        result = None
        for repeat_index in range(int(args.solve_repeats)):
            logs: list[str] = []
            solve_start = time.perf_counter()
            result = solve_v3_full_system_linear_gmres(
                nml=nml,
                tol=float(args.tol),
                restart=int(args.restart),
                maxiter=int(args.maxiter),
                solve_method=str(args.solve_method),
                identity_shift=float(args.identity_shift),
                emit=lambda level, msg: logs.append(str(msg)) if level <= 1 else None,
            )
            jax.block_until_ready((result.x, result.residual_norm))
            solve_repeats.append(
                {
                    "repeat_index": int(repeat_index),
                    "elapsed_s": time.perf_counter() - solve_start,
                    "residual_norm": float(result.residual_norm),
                    "structured_selected": bool(
                        (result.metadata or {}).get("structured_fblock_preconditioner_enabled", False)
                    ),
                }
            )
            selected_logs.extend(msg for msg in logs if "building RHSMode=1 preconditioner" in msg)
        assert result is not None
        first = solve_repeats[0]
        last = solve_repeats[-1]
        solve_payload = {
            "elapsed_s": float(first["elapsed_s"]),
            "warm_elapsed_s": float(last["elapsed_s"]) if len(solve_repeats) > 1 else None,
            "residual_norm": float(last["residual_norm"]),
            "structured_selected": bool(last["structured_selected"]),
            "repeats": solve_repeats,
            "selected_logs": selected_logs,
        }

    return {
        "case": case,
        "input": str(input_path.relative_to(_REPO_ROOT)),
        "preconditioner": preconditioner,
        "canonical_preconditioner": PRECONDITIONERS[preconditioner],
        "backend": jax.default_backend(),
        "device_count": int(jax.device_count()),
        "operator_shape": {
            "total_size": int(op.total_size),
            "f_size": int(op.f_size),
            "n_species": int(op.n_species),
            "n_x": int(op.n_x),
            "n_xi": int(op.n_xi),
            "n_theta": int(op.n_theta),
            "n_zeta": int(op.n_zeta),
            "include_phi1": bool(op.include_phi1),
        },
        "operator_build_s": float(operator_build_s),
        "preconditioner_build_apply_s": float(preconditioner_build_apply_s),
        "one_step_s": float(one_step_s),
        "dke_residual_ratio": float(dke_ratio),
        "full_residual_ratio": float(full_ratio),
        "warm_preconditioner_apply_s": [float(value) for value in warm_apply_times_s],
        "warm_preconditioner_apply_s_min": float(min(warm_apply_times_s)) if warm_apply_times_s else None,
        "warm_preconditioner_apply_s_median": _median(warm_apply_times_s),
        "warm_dke_residual_ratio": float(warm_dke_ratio) if warm_dke_ratio is not None else None,
        "metadata": _json_safe(metadata),
        "solve": _json_safe(solve_payload),
        "max_rss_mb": _rss_mb(),
        "ok": math.isfinite(float(dke_ratio)),
    }


def _run_child(args: argparse.Namespace, *, case: str, preconditioner: str, run_solve: bool) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--child",
        "--case",
        case,
        "--preconditioner",
        preconditioner,
        "--identity-shift",
        str(float(args.identity_shift)),
        "--tol",
        str(float(args.tol)),
        "--restart",
        str(int(args.restart)),
        "--maxiter",
        str(int(args.maxiter)),
        "--solve-method",
        str(args.solve_method),
        "--warm-repeats",
        str(int(args.warm_repeats)),
        "--solve-repeats",
        str(int(args.solve_repeats)),
    ]
    if run_solve:
        cmd.append("--run-solve")
    env = os.environ.copy()
    env.update(_preconditioner_env(preconditioner))
    env.setdefault("PYTHONPATH", str(_REPO_ROOT))
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(args.timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else str(exc.stdout or "")
        stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else str(exc.stderr or "")
        return {
            "case": case,
            "preconditioner": preconditioner,
            "returncode": -9,
            "ok": False,
            "error": "timeout",
            "timeout_s": float(args.timeout_s),
            "stdout_tail": stdout[-4000:],
            "stderr_tail": stderr[-4000:],
        }
    marker_line = ""
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARKER):
            marker_line = line[len(RESULT_MARKER) :]
    if proc.returncode == 0 and marker_line:
        row = json.loads(marker_line)
        row["returncode"] = int(proc.returncode)
        return row
    return {
        "case": case,
        "preconditioner": preconditioner,
        "returncode": int(proc.returncode),
        "ok": False,
        "error": "child_failed",
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
    }


def _solve_residual_ok(row: dict[str, Any], *, max_solve_residual: float) -> bool:
    solve = row.get("solve")
    if solve is None:
        return True
    if not isinstance(solve, dict):
        return False
    residual = solve.get("residual_norm")
    try:
        residual_f = float(residual)
    except (TypeError, ValueError):
        return False
    return math.isfinite(residual_f) and residual_f <= float(max_solve_residual)


def _warm_runtime_s(row: dict[str, Any]) -> float:
    solve = row.get("solve")
    if isinstance(solve, dict):
        for key in ("warm_elapsed_s", "elapsed_s"):
            value = solve.get(key)
            if value is None:
                continue
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed) and parsed >= 0.0:
                return parsed
    for key in ("warm_preconditioner_apply_s_median", "warm_preconditioner_apply_s_min"):
        try:
            parsed = float(row.get(key))
        except (TypeError, ValueError):
            continue
        if math.isfinite(parsed) and parsed >= 0.0:
            return parsed
    return float("inf")


def _rss_mb_value(row: dict[str, Any]) -> float:
    try:
        parsed = float(row.get("max_rss_mb"))
    except (TypeError, ValueError):
        return float("inf")
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else float("inf")


def _promotion_comparisons(
    results: list[dict[str, Any]],
    *,
    min_dke_improvement: float,
    max_solve_residual: float,
    max_warm_runtime_ratio: float,
    max_rss_ratio: float,
) -> dict[str, dict[str, Any]]:
    comparisons: dict[str, dict[str, Any]] = {}
    cases = sorted({str(row.get("case")) for row in results if bool(row.get("ok", False))})
    for case in cases:
        by_precond = {
            str(row.get("preconditioner")): row
            for row in results
            if bool(row.get("ok", False)) and str(row.get("case")) == case
        }
        radial = by_precond.get("fp_radial")
        if radial is None:
            continue
        radial_ratio = float(radial.get("dke_residual_ratio", float("inf")))
        radial_full_ratio = float(radial.get("full_residual_ratio", float("inf")))
        for candidate_name in (
            "fp_lowmode_schur",
            "fp_moment_schur",
            "fp_coupled_moment_schur",
            "fp_tail_coupled_schur",
        ):
            candidate = by_precond.get(candidate_name)
            if candidate is None:
                continue
            candidate_ratio = float(candidate.get("dke_residual_ratio", float("inf")))
            candidate_full_ratio = float(candidate.get("full_residual_ratio", float("inf")))
            improvement = radial_ratio / candidate_ratio if candidate_ratio > 0.0 else float("inf")
            full_improvement = (
                radial_full_ratio / candidate_full_ratio
                if math.isfinite(radial_full_ratio)
                and math.isfinite(candidate_full_ratio)
                and candidate_full_ratio > 0.0
                else None
            )
            candidate_meta = candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {}
            coarse = candidate_meta.get("coarse") if isinstance(candidate_meta, dict) else {}
            basis_storage = coarse.get("basis_storage_nbytes", -1) if isinstance(coarse, dict) else -1
            try:
                basis_storage_i = int(basis_storage)
            except (TypeError, ValueError):
                basis_storage_i = -1
            matrix_free = isinstance(coarse, dict) and basis_storage_i == 0
            radial_warm_runtime_s = _warm_runtime_s(radial)
            candidate_warm_runtime_s = _warm_runtime_s(candidate)
            runtime_ratio = (
                candidate_warm_runtime_s / radial_warm_runtime_s if radial_warm_runtime_s > 0.0 else float("inf")
            )
            runtime_gate = (
                math.isfinite(runtime_ratio)
                and candidate_warm_runtime_s >= 0.0
                and runtime_ratio <= float(max_warm_runtime_ratio)
            )
            radial_rss_mb = _rss_mb_value(radial)
            candidate_rss_mb = _rss_mb_value(candidate)
            rss_ratio = candidate_rss_mb / radial_rss_mb if radial_rss_mb > 0.0 else float("inf")
            rss_gate = math.isfinite(rss_ratio) and rss_ratio <= float(max_rss_ratio)
            residual_gate = bool(
                _solve_residual_ok(radial, max_solve_residual=float(max_solve_residual))
                and _solve_residual_ok(candidate, max_solve_residual=float(max_solve_residual))
            )
            promotion_ready = bool(
                math.isfinite(improvement)
                and improvement >= float(min_dke_improvement)
                and residual_gate
                and matrix_free
                and runtime_gate
                and rss_gate
            )
            comparison_key = case if candidate_name == "fp_lowmode_schur" else f"{case}:{candidate_name}"
            comparisons[comparison_key] = {
                "case": case,
                "baseline": "fp_radial",
                "candidate": candidate_name,
                "baseline_dke_residual_ratio": radial_ratio,
                "candidate_dke_residual_ratio": candidate_ratio,
                "dke_improvement": improvement,
                "baseline_full_residual_ratio": radial_full_ratio,
                "candidate_full_residual_ratio": candidate_full_ratio,
                "full_residual_improvement": full_improvement,
                "min_dke_improvement": float(min_dke_improvement),
                "max_solve_residual": float(max_solve_residual),
                "solve_residual_gate_ok": residual_gate,
                "matrix_free_storage_gate_ok": bool(matrix_free),
                "baseline_warm_runtime_s": radial_warm_runtime_s,
                "candidate_warm_runtime_s": candidate_warm_runtime_s,
                "warm_runtime_ratio": runtime_ratio,
                "max_warm_runtime_ratio": float(max_warm_runtime_ratio),
                "warm_runtime_gate_ok": bool(runtime_gate),
                "baseline_rss_mb": radial_rss_mb,
                "candidate_rss_mb": candidate_rss_mb,
                "rss_ratio": rss_ratio,
                "max_rss_ratio": float(max_rss_ratio),
                "rss_gate_ok": bool(rss_gate),
                "promotion_ready": promotion_ready,
            }
    return comparisons


def _summarize(
    results: list[dict[str, Any]],
    *,
    min_dke_improvement: float = 1.05,
    max_solve_residual: float = 1.0e-8,
    max_warm_runtime_ratio: float = 1.0,
    max_rss_ratio: float = 1.25,
) -> dict[str, Any]:
    ok_rows = [row for row in results if bool(row.get("ok", False))]
    solved = [row for row in ok_rows if isinstance(row.get("solve"), dict)]
    comparisons = _promotion_comparisons(
        results,
        min_dke_improvement=float(min_dke_improvement),
        max_solve_residual=float(max_solve_residual),
        max_warm_runtime_ratio=float(max_warm_runtime_ratio),
        max_rss_ratio=float(max_rss_ratio),
    )
    return {
        "result_count": len(results),
        "ok_count": len(ok_rows),
        "solve_count": len(solved),
        "all_ok": len(results) > 0 and len(ok_rows) == len(results),
        "best_one_step_by_case": {
            case: min(
                (row for row in ok_rows if row.get("case") == case),
                key=lambda row: float(row.get("dke_residual_ratio", float("inf"))),
                default={},
            ).get("preconditioner")
            for case in sorted({str(row.get("case")) for row in ok_rows})
        },
        "promotion_comparisons": comparisons,
        "promotion_ready_cases": [
            case for case, comparison in comparisons.items() if bool(comparison.get("promotion_ready", False))
        ],
    }


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "cases": list(args.cases),
        "preconditioners": list(args.preconditioners),
        "solve_cases": list(args.solve_cases or []),
        "timeout_s": float(args.timeout_s),
        "identity_shift": float(args.identity_shift),
        "tol": float(args.tol),
        "restart": int(args.restart),
        "maxiter": int(args.maxiter),
        "solve_method": str(args.solve_method),
        "warm_repeats": int(args.warm_repeats),
        "solve_repeats": int(args.solve_repeats),
        "max_solve_residual": float(args.max_solve_residual),
        "min_dke_improvement": float(args.min_dke_improvement),
        "max_warm_runtime_ratio": float(args.max_warm_runtime_ratio),
        "max_rss_ratio": float(args.max_rss_ratio),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if bool(args.child):
        payload = run_child_payload(args)
        print(RESULT_MARKER + json.dumps(_json_safe(payload), sort_keys=True))
        return 0

    plan = build_plan(args)
    results: list[dict[str, Any]] = []
    if not bool(args.dry_run):
        solve_cases = set(args.solve_cases or [])
        for case in args.cases:
            for preconditioner in args.preconditioners:
                results.append(
                    _run_child(
                        args,
                        case=str(case),
                        preconditioner=str(preconditioner),
                        run_solve=str(case) in solve_cases,
                    )
                )

    payload = {
        "schema_version": 1,
        "kind": "structured_fblock_preconditioner_benchmark",
        "plan": plan,
        "summary": _summarize(
            results,
            min_dke_improvement=float(args.min_dke_improvement),
            max_solve_residual=float(args.max_solve_residual),
            max_warm_runtime_ratio=float(args.max_warm_runtime_ratio),
            max_rss_ratio=float(args.max_rss_ratio),
        ),
        "results": _json_safe(results),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"], sort_keys=True))
    return 0 if payload["summary"]["all_ok"] or bool(args.dry_run) else 1


if __name__ == "__main__":
    raise SystemExit(main())
