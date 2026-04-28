#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sfincs_jax.compare import compare_sfincs_outputs
from sfincs_jax.namelist import read_sfincs_input


def _rhs_mode_from_input(path: Path) -> int:
    """Return the input RHSMode, defaulting to the SFINCS v3 full-system mode."""
    try:
        nml = read_sfincs_input(path)
        general = nml.group("general")
        return int(general.get("RHSMODE", general.get("rhsMode", 1)) or 1)
    except Exception:  # noqa: BLE001
        return 1


def _parse_variant(spec: str) -> tuple[str, dict[str, str]]:
    if "=" not in spec:
        return spec.strip(), {}
    name, rest = spec.split("=", 1)
    env: dict[str, str] = {}
    if rest.strip():
        for part in rest.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Invalid variant env assignment {part!r}")
            key, value = part.split("=", 1)
            env[key.strip()] = value.strip()
    return name.strip(), env


def _mismatch_summary(a: Path, b: Path, *, rtol: float, atol: float) -> dict[str, object]:
    diffs = compare_sfincs_outputs(a_path=a, b_path=b, rtol=rtol, atol=atol)
    bad = [d.key for d in diffs if not d.ok]
    return {"count": len(bad), "sample": bad[:8]}


def _tail_text(value: str | bytes | None, limit: int) -> str:
    """Return a JSON-safe tail for subprocess output.

    ``subprocess.TimeoutExpired`` can expose captured streams as bytes even when
    ``subprocess.run(..., text=True)`` was requested, so normalize here before
    serializing benchmark rows.
    """
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = value
    return text[-int(limit) :]


def _last_rhs1_preconditioner(stdout: str) -> str | None:
    preconditioners = _rhs1_preconditioners(stdout)
    return preconditioners[-1] if preconditioners else None


def _rhs1_preconditioners(stdout: str) -> list[str]:
    marker = "building RHSMode=1 preconditioner="
    values: list[str] = []
    for line in str(stdout).splitlines():
        if marker not in line:
            continue
        value = line.split(marker, 1)[1].strip()
        if value:
            values.append(value.split()[0])
    return values


def _parse_profile_events(stdout: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for line in str(stdout).splitlines():
        if "profiling:" not in line:
            continue
        head, tail = line.split("profiling:", 1)
        del head
        parts = tail.strip().split()
        if not parts:
            continue
        event_name = parts[0]
        fields: dict[str, str] = {}
        for part in parts[1:]:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            fields[key.strip()] = value.strip()
        total_raw = fields.get("total_s")
        delta_raw = fields.get("delta_s", fields.get("dt_s"))
        rss_raw = fields.get("rss_mb", "na")
        if total_raw is None or delta_raw is None:
            continue
        event: dict[str, object] = {
            "event": event_name,
            "total_s": float(total_raw.replace("D", "E").replace("d", "e")),
            "delta_s": float(delta_raw.replace("D", "E").replace("d", "e")),
            "rss_mb": None if rss_raw.lower() == "na" else float(rss_raw.replace("D", "E").replace("d", "e")),
        }
        events.append(event)
    return events


def _profile_stage_durations(events: list[dict[str, object]]) -> dict[str, float]:
    starts: dict[str, float] = {}
    durations: dict[str, float] = {}
    for event in events:
        name = str(event["event"])
        total_s = float(event["total_s"])
        if name.endswith("_start"):
            starts[name[: -len("_start")]] = total_s
        elif name.endswith("_done"):
            base = name[: -len("_done")]
            start = starts.pop(base, None)
            if start is not None:
                durations[base] = round(durations.get(base, 0.0) + max(0.0, total_s - start), 6)
    return durations


def _solver_path_summary(stdout: str) -> dict[str, object]:
    events = _parse_profile_events(stdout)
    text = str(stdout)
    return {
        "preconditioners": _rhs1_preconditioners(text),
        "profile_stage_durations_s": _profile_stage_durations(events),
        "profile_peak_rss_mb": max(
            (float(event["rss_mb"]) for event in events if event["rss_mb"] is not None),
            default=None,
        ),
        "profile_events": events,
        "used_dense_auto": "FP RHSMode=1 small system -> using dense solve" in text,
        "used_dense_fallback": "rhs1_dense_fallback" in text,
        "used_sparse_fallback": "rhs1_sparse_precond" in text or "host sparse LU direct fallback" in text,
    }


def _resource_maxrss_mb(raw_value: int, *, platform: str = sys.platform) -> float:
    """Convert ``resource.ru_maxrss`` to MB on Linux and macOS."""
    if str(platform).startswith("darwin"):
        return float(raw_value) / (1024.0 * 1024.0)
    return float(raw_value) / 1024.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark sfincs_jax solver variants on a frozen case directory.")
    parser.add_argument("--case-dir", type=Path, required=True, help="Case directory with input.namelist and optional fortran_run/sfincsOutput.h5")
    parser.add_argument(
        "--variant",
        action="append",
        default=[],
        help="Variant spec: name=ENV1=VALUE1,ENV2=VALUE2. Repeatable. 'default' is implied if omitted.",
    )
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--timeout-s", type=float, default=900.0)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument(
        "--differentiable",
        action="store_true",
        help="Use the differentiable implicit path. Default mirrors the CLI fast path.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable sfincs_jax profiling marks and include solver-stage timings/RSS in the JSON rows.",
    )
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case_dir = args.case_dir.resolve()
    case_input = case_dir / "input.namelist"
    if not case_input.exists():
        raise FileNotFoundError(f"Missing {case_input}")
    rhs_mode = _rhs_mode_from_input(case_input)
    compute_transport_matrix = int(rhs_mode) in {2, 3}
    reference = case_dir / "fortran_run" / "sfincsOutput.h5"
    have_reference = reference.exists()

    variants = [_parse_variant(spec) for spec in args.variant] if args.variant else [("default", {})]
    if not any(name == "default" for name, _ in variants):
        variants = [("default", {})] + variants

    child = r"""
import json, os, resource, sys, time
from pathlib import Path
from sfincs_jax.io import write_sfincs_jax_output_h5

case = Path(os.environ["CASE_INPUT"])
out = Path(os.environ["CASE_OUTPUT"])
t0 = time.perf_counter()
write_sfincs_jax_output_h5(
    input_namelist=case,
    output_path=out,
    compute_solution=True,
    compute_transport_matrix=bool(int(os.environ["COMPUTE_TRANSPORT_MATRIX"])),
    differentiable=bool(int(os.environ["DIFFERENTIABLE"])),
)
elapsed = time.perf_counter() - t0
raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
rss_mb = raw_rss / (1024.0 * 1024.0) if sys.platform.startswith("darwin") else raw_rss / 1024.0
print("@@RESULT@@" + json.dumps({"elapsed_s": elapsed, "ru_maxrss_raw": raw_rss, "ru_maxrss_mb": rss_mb}))
"""

    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": str(repo),
            "SFINCS_JAX_FORTRAN_STDOUT": "0",
            "SFINCS_JAX_SOLVER_ITER_STATS": "1",
            "SFINCS_JAX_IMPLICIT_SOLVE": "1" if args.differentiable else "0",
        }
    )
    if args.profile:
        base_env["SFINCS_JAX_PROFILE"] = "1"

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=f"{case_dir.name}_bench_") as td:
        tmp = Path(td)
        outputs: dict[str, Path] = {}
        for name, extra_env in variants:
            env = base_env.copy()
            env.update(extra_env)
            env["CASE_INPUT"] = str(case_input)
            env["CASE_OUTPUT"] = str(tmp / f"{name}.h5")
            env["COMPUTE_TRANSPORT_MATRIX"] = "1" if compute_transport_matrix else "0"
            env["DIFFERENTIABLE"] = "1" if args.differentiable else "0"
            print(f"## running {name}", flush=True)
            t0 = time.perf_counter()
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", child],
                    cwd=repo,
                    env=env,
                    text=True,
                    capture_output=True,
                    timeout=float(args.timeout_s),
                )
            except subprocess.TimeoutExpired as exc:
                rows.append(
                    {
                        "variant": name,
                        "wall_s": round(time.perf_counter() - t0, 3),
                        "returncode": None,
                        "env": extra_env,
                        "status": "timeout",
                        "stdout_tail": _tail_text(exc.stdout, 2500),
                        "stderr_tail": _tail_text(exc.stderr, 2000),
                    }
                )
                continue
            wall = time.perf_counter() - t0
            row: dict[str, object] = {
                "variant": name,
                "wall_s": round(wall, 3),
                "returncode": proc.returncode,
                "env": extra_env,
            }
            if proc.returncode != 0:
                row["status"] = "error"
                row["stderr_tail"] = _tail_text(proc.stderr, 2000)
                row["stdout_tail"] = _tail_text(proc.stdout, 2500)
                rows.append(row)
                continue
            marker = [line for line in proc.stdout.splitlines() if line.startswith("@@RESULT@@")][-1]
            result = json.loads(marker[len("@@RESULT@@") :])
            out_path = tmp / f"{name}.h5"
            outputs[name] = out_path
            combined_log = f"{proc.stdout}\n{proc.stderr}"
            row.update(
                {
                    "status": "ok",
                    "elapsed_s": round(float(result["elapsed_s"]), 3),
                    "ru_maxrss_raw": int(result.get("ru_maxrss_raw", 0)),
                    "ru_maxrss_mb": round(
                        float(
                            result.get(
                                "ru_maxrss_mb",
                                _resource_maxrss_mb(int(result.get("ru_maxrss_kb", 0))),
                            )
                        ),
                        3,
                    ),
                    "rhs1_preconditioner": _last_rhs1_preconditioner(combined_log),
                    "solver_path": _solver_path_summary(combined_log),
                    "used_adaptive_pas_smoother": "adaptive PAS smoother" in combined_log,
                    "used_pas_tokamak_theta": "preconditioner=pas_tokamak_theta" in combined_log,
                    "used_lgmres": "solve method forced by env -> lgmres" in combined_log,
                    "used_explicit_sparse_helper": "explicit sparse helper" in combined_log,
                }
            )
            if have_reference:
                row["vs_fortran"] = _mismatch_summary(reference, out_path, rtol=float(args.rtol), atol=float(args.atol))
            rows.append(row)
        default_out = outputs.get("default")
        if default_out is not None:
            for row in rows:
                path = outputs.get(str(row["variant"]))
                row["vs_default"] = (
                    {"count": 0, "sample": []}
                    if path is None or row["variant"] == "default"
                    else _mismatch_summary(default_out, path, rtol=float(args.rtol), atol=float(args.atol))
                )

    text = json.dumps(rows, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
