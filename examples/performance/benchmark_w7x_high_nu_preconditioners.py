#!/usr/bin/env python
"""Benchmark W7-X high-nu FP transport preconditioner candidates.

This script is intentionally a *single-RHS* harness. It isolates candidate
preconditioners for the unresolved W7-X high-collisionality FP lane without
launching the full publication scan. Each candidate runs in a fresh Python
process so JAX compilation, device memory, and environment variables are not
cross-contaminated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_INPUT = REPO_ROOT / "examples" / "sfincs_examples" / "transportMatrix_geometryScheme11" / "input.namelist"
DEFAULT_WORK_DIR = REPO_ROOT / "examples" / "performance" / "output" / "w7x_high_nu_preconditioners"
W7X_NUPRIME_FACTOR = 0.172714565


def _parse_csv_ints(text: str) -> list[int]:
    values = [int(part.strip()) for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one whichRHS value is required")
    return sorted(set(values))


def _parse_csv_strings(text: str) -> list[str]:
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("at least one preconditioner is required")
    return values


def _strip_scan_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.strip().lower().startswith("!ss")) + "\n"


def _set_group_assignment(text: str, group: str, key: str, value: str) -> str:
    group_pattern = re.compile(
        rf"(^\s*&{re.escape(group)}\b.*?$)(.*?)(^\s*/\s*$)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    match = group_pattern.search(text)
    if not match:
        raise ValueError(f"Could not find &{group} group in input text")
    group_header, group_body, group_end = match.groups()
    assign_pattern = re.compile(rf"(^\s*{re.escape(key)}\s*=\s*)([^!\n\r]*)(.*)$", re.IGNORECASE | re.MULTILINE)
    if assign_pattern.search(group_body):
        new_body = assign_pattern.sub(lambda m: f"{m.group(1)}{value}{m.group(3)}", group_body, count=1)
    else:
        if group_body and not group_body.endswith("\n"):
            group_body += "\n"
        new_body = group_body + f"  {key} = {value}\n"
    return text[: match.start()] + group_header + new_body + group_end + text[match.end() :]


def write_w7x_high_nu_input(
    *,
    destination: Path,
    nuprime: float,
    collision_operator: int,
    reduced_resolution: bool,
) -> Path:
    """Write one direct W7-X high-nu input from the publication base fixture."""
    text = _strip_scan_lines(BASE_INPUT.read_text())
    nu_n = float(nuprime) * W7X_NUPRIME_FACTOR
    text = _set_group_assignment(text, "physicsParameters", "nu_n", f"{nu_n:.12e}")
    text = _set_group_assignment(text, "physicsParameters", "collisionOperator", str(int(collision_operator)))
    if reduced_resolution:
        for key, value in (
            ("Ntheta", "5"),
            ("Nzeta", "7"),
            ("Nxi", "6"),
            ("Nx", "3"),
            ("solverTolerance", "1e-5"),
        ):
            text = _set_group_assignment(text, "resolutionParameters", key, value)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text)
    return destination


def candidate_environment(
    *,
    preconditioner: str,
    sparse_direct_max: int,
    sparse_factor_dtype: str,
    maxiter: int,
    fp_tzfft_max_mb: float,
) -> dict[str, str]:
    """Return isolated environment overrides for one candidate process."""
    env = {
        "JAX_ENABLE_X64": "True",
        "PYTHONUNBUFFERED": "1",
        "SFINCS_JAX_IMPLICIT_SOLVE": "0",
        "SFINCS_JAX_TRANSPORT_PRECOND": str(preconditioner),
        "SFINCS_JAX_WRITE_SOLVER_DIAGNOSTICS": "1",
        "SFINCS_JAX_TRANSPORT_LOW_MEMORY": "1",
        "SFINCS_JAX_TRANSPORT_STORE_STATE": "0",
        "SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS": "1",
    }
    if int(sparse_direct_max) > 0:
        env["SFINCS_JAX_TRANSPORT_SPARSE_DIRECT_MAX"] = str(int(sparse_direct_max))
    if str(sparse_factor_dtype).strip():
        env["SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE"] = str(sparse_factor_dtype).strip()
    if int(maxiter) > 0:
        env["SFINCS_JAX_TRANSPORT_MAXITER"] = str(int(maxiter))
    if float(fp_tzfft_max_mb) > 0.0:
        env["SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB"] = f"{float(fp_tzfft_max_mb):.16g}"
    return env


def _rss_mb() -> float:
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss / 1.0e6 if sys.platform == "darwin" else rss / 1024.0


def _run_one(args: argparse.Namespace) -> int:
    env = candidate_environment(
        preconditioner=str(args.preconditioner),
        sparse_direct_max=int(args.sparse_direct_max),
        sparse_factor_dtype=str(args.sparse_factor_dtype),
        maxiter=int(args.maxiter),
        fp_tzfft_max_mb=float(args.fp_tzfft_max_mb),
    )
    os.environ.update(env)

    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.problems.transport_matrix.solve import solve_v3_transport_matrix_linear_gmres

    rhs_values = _parse_csv_ints(args.which_rhs)
    messages: list[str] = []

    def _emit(level: int, message: str) -> None:
        line = f"[candidate={args.preconditioner} level={level}] {message}"
        messages.append(line)
        if int(args.verbose) > 0:
            print(line, flush=True)

    t0 = time.perf_counter()
    result = solve_v3_transport_matrix_linear_gmres(
        nml=read_sfincs_input(Path(args.input)),
        tol=float(args.tol),
        atol=float(args.atol),
        restart=int(args.restart),
        maxiter=int(args.maxiter) if int(args.maxiter) > 0 else None,
        solve_method=str(args.solve_method),
        differentiable=False,
        which_rhs_values=rhs_values,
        force_stream_diagnostics=True,
        force_store_state=False,
        collect_transport_output_fields=False,
        emit=_emit,
    )
    elapsed = float(time.perf_counter() - t0)
    residuals = {int(k): float(np.asarray(v, dtype=np.float64)) for k, v in result.residual_norms_by_rhs.items()}
    rhs_norms = {int(k): float(np.asarray(v, dtype=np.float64)) for k, v in result.rhs_norms_by_rhs.items()}
    relative = {
        int(k): float(residuals[int(k)] / rhs_norms[int(k)])
        for k in residuals
        if int(k) in rhs_norms and np.isfinite(rhs_norms[int(k)]) and rhs_norms[int(k)] > 0.0
    }
    payload = {
        "status": "ok",
        "preconditioner": str(args.preconditioner),
        "input": str(Path(args.input).resolve()),
        "which_rhs": rhs_values,
        "residual_norms_by_rhs": residuals,
        "rhs_norms_by_rhs": rhs_norms,
        "relative_residuals_by_rhs": relative,
        "elapsed_s": elapsed,
        "elapsed_time_by_rhs_s": [float(v) for v in np.asarray(result.elapsed_time_s, dtype=np.float64).reshape((-1,))],
        "max_rss_mb": _rss_mb(),
        "environment": env,
        "last_messages": messages[-40:],
    }
    print("SFINCS_JAX_W7X_PRECOND_RESULT " + json.dumps(payload, sort_keys=True), flush=True)
    return 0


def _extract_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith("SFINCS_JAX_W7X_PRECOND_RESULT "):
            return json.loads(line.split(" ", 1)[1])
    return None


def _run_candidate(args: argparse.Namespace, *, input_path: Path, preconditioner: str) -> dict[str, Any]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--run-one",
        "--input",
        str(input_path),
        "--preconditioner",
        str(preconditioner),
        "--which-rhs",
        str(args.which_rhs),
        "--tol",
        str(args.tol),
        "--atol",
        str(args.atol),
        "--restart",
        str(args.restart),
        "--maxiter",
        str(args.maxiter),
        "--solve-method",
        str(args.solve_method),
        "--sparse-direct-max",
        str(args.sparse_direct_max),
        "--sparse-factor-dtype",
        str(args.sparse_factor_dtype),
        "--fp-tzfft-max-mb",
        str(args.fp_tzfft_max_mb),
        "--verbose",
        str(args.child_verbose),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}" if env.get("PYTHONPATH") else str(REPO_ROOT)
    log_path = Path(args.work_dir) / f"candidate_{preconditioner}.log"
    t0 = time.perf_counter()
    try:
        completed = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(args.timeout_s) if float(args.timeout_s) > 0.0 else None,
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        log_path.write_text(stdout + ("\nSTDERR:\n" + stderr if stderr else ""))
        parsed = _extract_result(stdout)
        if completed.returncode == 0 and parsed is not None:
            parsed["log_path"] = str(log_path)
            parsed["returncode"] = int(completed.returncode)
            return parsed
        return {
            "status": "failed",
            "preconditioner": str(preconditioner),
            "returncode": int(completed.returncode),
            "elapsed_s": float(time.perf_counter() - t0),
            "log_path": str(log_path),
            "stdout_tail": stdout.splitlines()[-40:],
            "stderr_tail": stderr.splitlines()[-40:],
            "command": cmd,
        }
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout.decode() if exc.stdout else "")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr.decode() if exc.stderr else "")
        log_path.write_text(stdout + ("\nSTDERR:\n" + stderr if stderr else ""))
        return {
            "status": "timeout",
            "preconditioner": str(preconditioner),
            "elapsed_s": float(time.perf_counter() - t0),
            "timeout_s": float(args.timeout_s),
            "log_path": str(log_path),
            "stdout_tail": stdout.splitlines()[-40:],
            "stderr_tail": stderr.splitlines()[-40:],
            "command": cmd,
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, default=None, help="Existing input.namelist to benchmark.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Benchmark output directory.")
    parser.add_argument("--summary", type=Path, default=None, help="Output summary JSON path.")
    parser.add_argument("--nuprime", type=float, default=17.78332923601508, help="W7-X normalized collisionality.")
    parser.add_argument("--collision-operator", type=int, default=0, help="0=FP, 1=PAS.")
    parser.add_argument("--reduced-resolution", action="store_true", help="Use a small generated input for quick tests.")
    parser.add_argument("--preconditioners", default="auto,fp_tzfft,xmg,theta_schwarz", help="CSV candidate list.")
    parser.add_argument("--preconditioner", default="auto", help=argparse.SUPPRESS)
    parser.add_argument("--which-rhs", default="2", help="CSV whichRHS subset; RHS2 is the known W7-X blocker.")
    parser.add_argument("--tol", default="1e-10")
    parser.add_argument("--atol", default="0.0")
    parser.add_argument("--restart", default="120")
    parser.add_argument("--maxiter", type=int, default=800)
    parser.add_argument("--solve-method", default="incremental")
    parser.add_argument("--sparse-direct-max", type=int, default=30000)
    parser.add_argument(
        "--sparse-factor-dtype",
        default="",
        help="Optional SFINCS_JAX_TRANSPORT_SPARSE_FACTOR_DTYPE override, e.g. float32.",
    )
    parser.add_argument("--fp-tzfft-max-mb", type=float, default=384.0)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--dry-run", action="store_true", help="Write input and commands but do not run candidates.")
    parser.add_argument("--child-verbose", type=int, default=1, help="Verbosity passed to candidate subprocesses.")
    parser.add_argument("--verbose", type=int, default=0, help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.run_one:
        if args.input is None:
            raise ValueError("--input is required with --run-one")
        return _run_one(args)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    input_path = Path(args.input) if args.input is not None else work_dir / "input.w7x_high_nu.namelist"
    if args.input is None:
        write_w7x_high_nu_input(
            destination=input_path,
            nuprime=float(args.nuprime),
            collision_operator=int(args.collision_operator),
            reduced_resolution=bool(args.reduced_resolution),
        )
    summary_path = Path(args.summary) if args.summary is not None else work_dir / "w7x_high_nu_preconditioner_benchmark.json"
    preconditioners = _parse_csv_strings(args.preconditioners)
    rhs_values = _parse_csv_ints(args.which_rhs)
    commands = []
    for precond in preconditioners:
        commands.append(
            {
                "preconditioner": precond,
                "environment": candidate_environment(
                    preconditioner=precond,
                    sparse_direct_max=int(args.sparse_direct_max),
                    sparse_factor_dtype=str(args.sparse_factor_dtype),
                    maxiter=int(args.maxiter),
                    fp_tzfft_max_mb=float(args.fp_tzfft_max_mb),
                ),
            }
        )
    if args.dry_run:
        payload = {
            "status": "dry_run",
            "input": str(input_path.resolve()),
            "which_rhs": rhs_values,
            "preconditioners": preconditioners,
            "commands": commands,
        }
        summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    results = [_run_candidate(args, input_path=input_path, preconditioner=precond) for precond in preconditioners]
    payload = {
        "status": "complete",
        "input": str(input_path.resolve()),
        "which_rhs": rhs_values,
        "preconditioners": preconditioners,
        "results": results,
    }
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
