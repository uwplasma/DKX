#!/usr/bin/env python3
"""Bounded RHSMode=1 solver-stack audit helper for QA/QH experiments.

The helper intentionally stays outside solver code.  It runs one bounded
``sfincs_jax solve-v3`` command per case/preconditioner pair, captures log lines
that mention setup/preflight/GMRES activity, and writes a compact JSON summary.
Use ``--dry-run`` first when pointing at production QA/QH inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "outputs" / "rhs1_solver_stack_audit.json"
DEFAULT_WORK_DIR = REPO_ROOT / "outputs" / "rhs1_solver_stack_audit_work"
DEFAULT_TIMEOUT_S = 120.0
MAX_DEFAULT_TIMEOUT_S = 600.0

PROFILE_EVENT_RE = re.compile(r"\bprofiling:\s*(?P<event>\S+)(?P<tail>.*)$", re.IGNORECASE)
FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eEdD][-+]?\d+)?"
KSP_ITER_RE = re.compile(
    r"\bksp_iterations=(?P<iterations>\d+)\s+solver=(?P<solver>[A-Za-z0-9_+-]+)",
    re.IGNORECASE,
)
GMRES_TOKEN_RE = re.compile(r"\b[A-Za-z0-9_+-]*gmres[A-Za-z0-9_+-]*\b", re.IGNORECASE)
ITER_VALUE_RE = re.compile(r"\b(?:iters|iterations)=(?P<value>\d+)", re.IGNORECASE)
MATVECS_VALUE_RE = re.compile(r"\bmatvecs=(?P<value>\d+)", re.IGNORECASE)
RESIDUAL_RE = re.compile(
    rf"\b(?P<name>ksp_residual|residual_norm|true_residual|final_residual)=\s*(?P<value>{FLOAT_RE})",
    re.IGNORECASE,
)
STRUCTURED_SOLVE_RE = re.compile(
    rf"solve_v3_full_system_structured_csr:\s+converged=(?P<converged>True|False|true|false)"
    rf"\s+residual=(?P<residual>{FLOAT_RE})\s+solve_s=(?P<solve_s>{FLOAT_RE})",
    re.IGNORECASE,
)
STRUCTURED_PC_RE = re.compile(
    r"solve_v3_full_system_structured_csr:\s+pc_kind=(?P<kind>\S+)"
    r"\s+pc_selected=(?P<selected>True|False|true|false)"
    r"\s+pc_reason=(?P<reason>\S+)"
    rf"\s+pc_setup_s=(?P<setup_s>{FLOAT_RE})"
    r"\s+pc_factor_nbytes=(?P<factor_nbytes>\S+)"
    r"\s+pc_permc=(?P<permc>\S+)"
    r"\s+pc_superlu_permc=(?P<superlu_permc>\S+)",
    re.IGNORECASE,
)
DIRECT_TAIL_PC_RE = re.compile(
    r"fortran_reduced direct-tail structured preconditioner selected kind=(?P<kind>\S+)"
    rf"\s+setup_s=(?P<setup_s>{FLOAT_RE})"
    rf"\s+elapsed_s=(?P<elapsed_s>{FLOAT_RE})"
    r"\s+reason=(?P<reason>\S+)"
    r"\s+cache_hit=(?P<cache_hit>True|False|true|false)"
    r"(?:\s+factor_nbytes=(?P<factor_nbytes>\S+)\s+permc=(?P<permc>\S+)\s+superlu_permc=(?P<superlu_permc>\S+))?",
    re.IGNORECASE,
)
PRECONDITIONER_RE = re.compile(
    r"\bbuilding RHSMode=1 preconditioner=(?P<preconditioner>\S+)",
    re.IGNORECASE,
)
SETUP_RE = re.compile(
    r"(setup|factor|assemble|building RHSMode=1 preconditioner|profiling:.*(?:build|setup|factor|assemble))",
    re.IGNORECASE,
)
PREFLIGHT_RE = re.compile(
    r"(preflight|budget|gate|candidate|reject|fallback|promotion)",
    re.IGNORECASE,
)
GMRES_RE = re.compile(
    r"(gmres|lgmres|fgmres|ksp_|ksp iterations|ksp_residual|matvecs)",
    re.IGNORECASE,
)

RunCallable = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class AuditCase:
    label: str
    input_path: Path


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


def _parse_env_assignment(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {spec!r}")
    key, value = spec.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"empty env key in {spec!r}")
    return key, value.strip()


def _parse_case_spec(spec: str) -> AuditCase:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"expected LABEL=PATH, got {spec!r}")
    label, path = spec.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError(f"empty case label in {spec!r}")
    return AuditCase(label=label, input_path=Path(path).expanduser())


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _tail_text(value: str | bytes | None, limit: int) -> str:
    if value is None:
        return ""
    text = value.decode(errors="replace") if isinstance(value, bytes) else str(value)
    return text[-int(limit) :]


def _float_or_none(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(str(value).replace("D", "E").replace("d", "e"))
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _parse_profile_events(text: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in str(text).splitlines():
        match = PROFILE_EVENT_RE.search(line)
        if match is None:
            continue
        fields: dict[str, str] = {}
        for part in match.group("tail").strip().split():
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        total_s = _float_or_none(fields.get("total_s"))
        delta_s = _float_or_none(fields.get("delta_s", fields.get("dt_s")))
        if total_s is None or delta_s is None:
            continue
        rss_raw = fields.get("rss_mb", "na")
        rss_mb = None if str(rss_raw).lower() == "na" else _float_or_none(rss_raw)
        events.append(
            {
                "event": match.group("event"),
                "total_s": total_s,
                "delta_s": delta_s,
                "rss_mb": rss_mb,
            }
        )
    return events


def _profile_stage_durations(events: Sequence[dict[str, Any]]) -> dict[str, float]:
    starts: dict[str, float] = {}
    durations: dict[str, float] = {}
    for event in events:
        name = str(event.get("event", ""))
        total_s = _float_or_none(str(event.get("total_s", "")))
        if total_s is None:
            continue
        if name.endswith("_start"):
            starts[name[: -len("_start")]] = total_s
        elif name.endswith("_done"):
            base = name[: -len("_done")]
            start = starts.pop(base, None)
            if start is not None:
                durations[base] = round(durations.get(base, 0.0) + max(0.0, total_s - start), 6)
    return durations


def _capture_matching_lines(text: str, pattern: re.Pattern[str], *, max_lines: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(str(text).splitlines(), start=1):
        if pattern.search(line):
            rows.append({"line": line_no, "text": line})
    if len(rows) > int(max_lines):
        return rows[: int(max_lines)]
    return rows


def _last_gmres_token(line: str) -> str | None:
    tokens = [match.group(0).lower() for match in GMRES_TOKEN_RE.finditer(line)]
    if not tokens:
        return None
    if len(tokens) > 1 and tokens[-1] == "gmres" and tokens[0].endswith("linear_gmres"):
        return tokens[-1]
    return tokens[-1]


def parse_audit_text(text: str, *, max_lines: int = 80) -> dict[str, Any]:
    """Extract bounded setup/preflight/GMRES evidence from solver stdout/stderr."""
    text = str(text)
    profile_events = _parse_profile_events(text)
    structured_solves = [
        {
            "converged": str(match.group("converged")).lower() == "true",
            "residual": _float_or_none(match.group("residual")),
            "solve_s": _float_or_none(match.group("solve_s")),
        }
        for match in STRUCTURED_SOLVE_RE.finditer(text)
    ]
    structured_preconditioners = []
    for match in STRUCTURED_PC_RE.finditer(text):
        factor_nbytes_raw = str(match.group("factor_nbytes"))
        structured_preconditioners.append(
            {
                "kind": match.group("kind"),
                "selected": str(match.group("selected")).lower() == "true",
                "reason": match.group("reason"),
                "setup_s": _float_or_none(match.group("setup_s")),
                "factor_nbytes": None if factor_nbytes_raw.lower() == "na" else _float_or_none(factor_nbytes_raw),
                "permc_spec": match.group("permc"),
                "superlu_permc_spec": match.group("superlu_permc"),
            }
        )
    direct_tail_preconditioners = []
    for match in DIRECT_TAIL_PC_RE.finditer(text):
        factor_nbytes_raw = match.group("factor_nbytes")
        direct_tail_preconditioners.append(
            {
                "kind": match.group("kind"),
                "setup_s": _float_or_none(match.group("setup_s")),
                "elapsed_s": _float_or_none(match.group("elapsed_s")),
                "reason": match.group("reason"),
                "cache_hit": str(match.group("cache_hit")).lower() == "true",
                "factor_nbytes": (
                    None
                    if factor_nbytes_raw is None or str(factor_nbytes_raw).lower() == "na"
                    else _float_or_none(factor_nbytes_raw)
                ),
                "permc_spec": match.group("permc"),
                "superlu_permc_spec": match.group("superlu_permc"),
            }
        )
    preconditioners = [match.group("preconditioner") for match in PRECONDITIONER_RE.finditer(text)]
    ksp_iterations = [
        {"solver": match.group("solver").lower(), "iterations": int(match.group("iterations"))}
        for match in KSP_ITER_RE.finditer(text)
    ]
    seen_iter: set[tuple[str, int]] = {(row["solver"], int(row["iterations"])) for row in ksp_iterations}
    gmres_matvecs: list[dict[str, Any]] = []
    for line in text.splitlines():
        solver = _last_gmres_token(line)
        if solver is None:
            continue
        iter_match = ITER_VALUE_RE.search(line)
        if iter_match is not None:
            key = (solver, int(iter_match.group("value")))
            if key not in seen_iter:
                seen_iter.add(key)
                ksp_iterations.append({"solver": key[0], "iterations": key[1]})
        matvecs_match = MATVECS_VALUE_RE.search(line)
        if matvecs_match is not None:
            gmres_matvecs.append({"solver": solver, "matvecs": int(matvecs_match.group("value"))})
    residuals = [
        {"name": match.group("name").lower(), "value": _float_or_none(match.group("value"))}
        for match in RESIDUAL_RE.finditer(text)
    ]
    setup_lines = _capture_matching_lines(text, SETUP_RE, max_lines=max_lines)
    preflight_lines = _capture_matching_lines(text, PREFLIGHT_RE, max_lines=max_lines)
    gmres_lines = _capture_matching_lines(text, GMRES_RE, max_lines=max_lines)
    selected_line_keys: set[tuple[int, str]] = set()
    selected_lines: list[dict[str, Any]] = []
    for rows in (setup_lines, preflight_lines, gmres_lines):
        for row in rows:
            key = (int(row["line"]), str(row["text"]))
            if key in selected_line_keys:
                continue
            selected_line_keys.add(key)
            selected_lines.append(row)
    selected_lines.sort(key=lambda row: int(row["line"]))
    if len(selected_lines) > int(max_lines):
        selected_lines = selected_lines[: int(max_lines)]
    rss_values = [float(event["rss_mb"]) for event in profile_events if event.get("rss_mb") is not None]
    return {
        "setup_lines": setup_lines,
        "preflight_lines": preflight_lines,
        "gmres_lines": gmres_lines,
        "selected_lines": selected_lines,
        "rhs1_preconditioners": preconditioners,
        "last_rhs1_preconditioner": preconditioners[-1] if preconditioners else None,
        "profile_events": profile_events[: int(max_lines)],
        "profile_stage_durations_s": _profile_stage_durations(profile_events),
        "profile_peak_rss_mb": max(rss_values) if rss_values else None,
        "structured_solves": structured_solves,
        "last_structured_solve": structured_solves[-1] if structured_solves else None,
        "structured_preconditioners": structured_preconditioners,
        "last_structured_preconditioner": (
            structured_preconditioners[-1] if structured_preconditioners else None
        ),
        "direct_tail_preconditioners": direct_tail_preconditioners,
        "last_direct_tail_preconditioner": (
            direct_tail_preconditioners[-1] if direct_tail_preconditioners else None
        ),
        "ksp_iterations": ksp_iterations,
        "gmres_matvecs": gmres_matvecs,
        "residuals": residuals,
    }


def _rhs_mode_from_input(path: Path) -> int | None:
    try:
        from sfincs_jax.namelist import read_sfincs_input  # noqa: PLC0415

        nml = read_sfincs_input(path)
        return int(nml.group("general").get("RHSMODE", 1))
    except Exception:  # noqa: BLE001
        return None


def _validate_cases(cases: Sequence[AuditCase], *, skip_rhsmode_check: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        input_path = case.input_path.resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"missing input for {case.label}: {input_path}")
        rhs_mode = None if skip_rhsmode_check else _rhs_mode_from_input(input_path)
        if rhs_mode is not None and int(rhs_mode) != 1:
            raise ValueError(f"{case.label} is RHSMode={rhs_mode}, expected RHSMode=1: {input_path}")
        rows.append({"label": case.label, "input": str(input_path), "rhs_mode": rhs_mode})
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        type=_parse_case_spec,
        help="Case spec LABEL=/path/to/input.namelist. Repeat for QA and QH.",
    )
    parser.add_argument("--qa-input", type=Path, default=None, help="Convenience alias for --case QA=PATH.")
    parser.add_argument("--qh-input", type=Path, default=None, help="Convenience alias for --case QH=PATH.")
    parser.add_argument(
        "--preconditioner",
        action="append",
        default=[],
        help="Preconditioner value to force for each case. Repeatable. Default: auto.",
    )
    parser.add_argument(
        "--preconditioner-env-key",
        default=None,
        help=(
            "Env var used for --preconditioner. Default is SFINCS_JAX_RHSMODE1_PRECONDITIONER, "
            "or SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER for host_structured_csr solves."
        ),
    )
    parser.add_argument("--env", action="append", default=[], type=_parse_env_assignment, help="Extra KEY=VALUE for child runs.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="JSON summary path.")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR, help="Directory for per-row state vectors.")
    parser.add_argument("--timeout-s", type=_positive_float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--allow-long-run", action="store_true", help="Allow --timeout-s above the default 600s cap.")
    parser.add_argument("--solve-method", default="auto", help="Forwarded to sfincs_jax solve-v3 --solve-method.")
    parser.add_argument("--tol", default="1e-8", help="Forwarded GMRES relative tolerance.")
    parser.add_argument("--atol", default="0.0", help="Forwarded GMRES absolute tolerance.")
    parser.add_argument("--restart", type=_positive_int, default=20, help="Forwarded GMRES restart.")
    parser.add_argument("--maxiter", type=_positive_int, default=12, help="Forwarded GMRES maxiter.")
    parser.add_argument("--cores", type=_positive_int, default=1, help="Forwarded sfincs_jax --cores value.")
    parser.add_argument(
        "--command",
        default=None,
        help=(
            "Optional shell-like command template. Available placeholders: {input}, {out_state}, "
            "{case}, {preconditioner}. If omitted, a bounded sfincs_jax solve-v3 command is used."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Write planned commands without running them.")
    parser.add_argument("--skip-rhsmode-check", action="store_true", help="Do not validate that inputs have RHSMode=1.")
    parser.add_argument("--capture-lines", type=_positive_int, default=80, help="Max lines retained per capture group.")
    parser.add_argument("--tail-chars", type=_positive_int, default=4000, help="Max stdout/stderr tail characters per row.")
    parser.add_argument("--print-json", action="store_true", help="Also print the JSON payload.")
    return parser


def _cases_from_args(args: argparse.Namespace) -> list[AuditCase]:
    cases = list(args.case or [])
    if args.qa_input is not None:
        cases.append(AuditCase(label="QA", input_path=Path(args.qa_input).expanduser()))
    if args.qh_input is not None:
        cases.append(AuditCase(label="QH", input_path=Path(args.qh_input).expanduser()))
    if not cases:
        raise ValueError("provide at least one --case LABEL=PATH, --qa-input, or --qh-input")
    labels = [case.label for case in cases]
    duplicates = sorted({label for label in labels if labels.count(label) > 1})
    if duplicates:
        raise ValueError(f"duplicate case labels: {', '.join(duplicates)}")
    return cases


def _preconditioners_from_args(args: argparse.Namespace) -> list[str]:
    values = [str(value).strip() for value in (args.preconditioner or []) if str(value).strip()]
    return values or ["auto"]


def _base_env(repo: Path, args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo) if not existing_pythonpath else f"{repo}{os.pathsep}{existing_pythonpath}"
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")
    env.update(
        {
            "SFINCS_JAX_FORTRAN_STDOUT": "0",
            "SFINCS_JAX_SOLVER_ITER_STATS": "1",
            "SFINCS_JAX_PROFILE": "1",
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART": str(int(args.restart)),
            "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER": str(int(args.maxiter)),
        }
    )
    if str(args.solve_method).strip().lower() in {"host_structured_csr", "structured_csr", "structured_full_csr"}:
        env.setdefault("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "gmres")
        env.setdefault("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "1")
        env.setdefault("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", "512")
        env.setdefault("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", "256")
    for key, value in args.env:
        env[key] = value
    return env


def _selected_env_for_row(preconditioner: str, args: argparse.Namespace) -> dict[str, str]:
    selected: dict[str, str] = {
        "SFINCS_JAX_FORTRAN_STDOUT": "0",
        "SFINCS_JAX_SOLVER_ITER_STATS": "1",
        "SFINCS_JAX_PROFILE": "1",
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART": str(int(args.restart)),
        "SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER": str(int(args.maxiter)),
    }
    if str(args.solve_method).strip().lower() in {"host_structured_csr", "structured_csr", "structured_full_csr"}:
        selected.update(
            {
                "SFINCS_JAX_RHS1_FULL_CSR_KRYLOV": "gmres",
                "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF": "1",
                "SFINCS_JAX_RHS1_FULL_CSR_MAX_MB": "512",
                "SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB": "256",
            }
        )
    for key, value in args.env:
        selected[key] = value
    env_key = args.preconditioner_env_key
    if env_key is None:
        env_key = (
            "SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER"
            if str(args.solve_method).strip().lower()
            in {"host_structured_csr", "structured_csr", "structured_full_csr"}
            else "SFINCS_JAX_RHSMODE1_PRECONDITIONER"
        )
    if preconditioner and preconditioner != "auto":
        selected[str(env_key)] = str(preconditioner)
    return selected


def _out_state_path(work_dir: Path, case: AuditCase, preconditioner: str) -> Path:
    safe_preconditioner = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(preconditioner))
    safe_case = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(case.label))
    return work_dir / f"{safe_case}.{safe_preconditioner}.stateVector.npy"


def _build_command(case: AuditCase, preconditioner: str, args: argparse.Namespace) -> list[str]:
    out_state = _out_state_path(Path(args.work_dir), case, preconditioner)
    if args.command:
        formatted = str(args.command).format(
            input=str(case.input_path.resolve()),
            out_state=str(out_state),
            case=str(case.label),
            preconditioner=str(preconditioner),
        )
        return shlex.split(formatted)
    verbose_flag = "-vv"
    return [
        sys.executable,
        "-m",
        "sfincs_jax",
        verbose_flag,
        "--no-fortran-stdout",
        "--cores",
        str(int(args.cores)),
        "solve-v3",
        "--input",
        str(case.input_path.resolve()),
        "--out-state",
        str(out_state),
        "--tol",
        str(args.tol),
        "--atol",
        str(args.atol),
        "--restart",
        str(int(args.restart)),
        "--maxiter",
        str(int(args.maxiter)),
        "--solve-method",
        str(args.solve_method),
    ]


def _row_status(returncode: int | None, timed_out: bool) -> str:
    if timed_out:
        return "timeout"
    if returncode == 0:
        return "ok"
    return "error"


def _run_one(
    *,
    repo: Path,
    case: AuditCase,
    preconditioner: str,
    args: argparse.Namespace,
    run_fn: RunCallable | None = None,
) -> dict[str, Any]:
    run_fn = subprocess.run if run_fn is None else run_fn
    command = _build_command(case, preconditioner, args)
    selected_env = _selected_env_for_row(preconditioner, args)
    env = _base_env(repo, args)
    env.update(selected_env)
    out_state = _out_state_path(Path(args.work_dir), case, preconditioner)
    row: dict[str, Any] = {
        "case": case.label,
        "input": str(case.input_path.resolve()),
        "preconditioner": preconditioner,
        "preconditioner_env": selected_env,
        "command": command,
        "out_state": str(out_state),
        "timeout_s": float(args.timeout_s),
    }
    if bool(args.dry_run):
        row.update({"status": "planned", "returncode": None, "timed_out": False, "wall_s": 0.0})
        return row

    out_state.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        proc = run_fn(
            command,
            cwd=str(repo),
            env=env,
            text=True,
            capture_output=True,
            timeout=float(args.timeout_s),
        )
        wall_s = time.perf_counter() - t0
        combined = f"{proc.stdout}\n{proc.stderr}"
        row.update(
            {
                "status": _row_status(int(proc.returncode), False),
                "returncode": int(proc.returncode),
                "timed_out": False,
                "wall_s": round(wall_s, 3),
                "stdout_tail": _tail_text(proc.stdout, int(args.tail_chars)),
                "stderr_tail": _tail_text(proc.stderr, int(args.tail_chars)),
                "captures": parse_audit_text(combined, max_lines=int(args.capture_lines)),
            }
        )
    except subprocess.TimeoutExpired as exc:
        wall_s = time.perf_counter() - t0
        combined = f"{_tail_text(exc.stdout, int(args.tail_chars))}\n{_tail_text(exc.stderr, int(args.tail_chars))}"
        row.update(
            {
                "status": "timeout",
                "returncode": None,
                "timed_out": True,
                "wall_s": round(wall_s, 3),
                "stdout_tail": _tail_text(exc.stdout, int(args.tail_chars)),
                "stderr_tail": _tail_text(exc.stderr, int(args.tail_chars)),
                "captures": parse_audit_text(combined, max_lines=int(args.capture_lines)),
            }
        )
    return row


def _summarize_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    case_counts: dict[str, dict[str, int]] = {}
    for row in rows:
        status = str(row.get("status", "unknown"))
        case = str(row.get("case", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        case_counts.setdefault(case, {})
        case_counts[case][status] = case_counts[case].get(status, 0) + 1
    return {
        "row_count": len(rows),
        "status_counts": status_counts,
        "case_status_counts": case_counts,
        "any_timeout": bool(status_counts.get("timeout", 0)),
        "any_error": bool(status_counts.get("error", 0)),
    }


def build_payload(
    *,
    cases: Sequence[AuditCase],
    preconditioners: Sequence[str],
    args: argparse.Namespace,
    run_fn: RunCallable | None = None,
) -> dict[str, Any]:
    if float(args.timeout_s) > MAX_DEFAULT_TIMEOUT_S and not bool(args.allow_long_run):
        raise ValueError(
            f"timeout {float(args.timeout_s):.1f}s exceeds the default {MAX_DEFAULT_TIMEOUT_S:.0f}s cap; "
            "pass --allow-long-run for explicit longer audits"
        )
    repo = REPO_ROOT.resolve()
    case_plan = _validate_cases(cases, skip_rhsmode_check=bool(args.skip_rhsmode_check))
    rows: list[dict[str, Any]] = []
    for case in cases:
        for preconditioner in preconditioners:
            rows.append(_run_one(repo=repo, case=case, preconditioner=preconditioner, args=args, run_fn=run_fn))
    return {
        "kind": "rhs1_solver_stack_audit",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "repo": str(repo),
        "plan": {
            "cases": case_plan,
            "preconditioners": list(preconditioners),
            "timeout_s": float(args.timeout_s),
            "timeout_cap_s": MAX_DEFAULT_TIMEOUT_S,
            "solve_method": str(args.solve_method),
            "tol": str(args.tol),
            "atol": str(args.atol),
            "restart": int(args.restart),
            "maxiter": int(args.maxiter),
            "cores": int(args.cores),
            "dry_run": bool(args.dry_run),
            "command_template": args.command,
        },
        "rows": rows,
        "summary": _summarize_rows(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cases = _cases_from_args(args)
        preconditioners = _preconditioners_from_args(args)
        payload = build_payload(cases=cases, preconditioners=preconditioners, args=args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, default=_json_default)
    args.out.write_text(text + "\n", encoding="utf-8")
    if bool(args.print_json):
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
