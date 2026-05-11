#!/usr/bin/env python
"""Bounded PAS-TZ memory-fallback benchmark harness.

The production-resolution geometry-rich PAS lane should not be promoted by
heuristics. This script forces the matrix-free RHSMode=1 solver through the
``pas_tz`` memory-fallback path in short-lived subprocesses, so slow
preconditioner builds are recorded as bounded timeouts instead of hanging a
developer shell or CI job.
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
import tempfile
import time
from typing import Any


_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = _REPO_ROOT / "examples" / "sfincs_examples" / "geometryScheme4_2species_PAS_noEr" / "input.namelist"
DEFAULT_OUT = _REPO_ROOT / "examples" / "performance" / "output" / "pas_tz_memory_fallback_benchmark.json"
RESULT_MARKER = "__SFINCS_JAX_PAS_TZ_RESULT__="
GRID_OVERRIDE_KEYS = ("Ntheta", "Nzeta", "Nxi", "Nx")


def _tail_text(value: str | bytes | None, n: int = 4000) -> str:
    """Return a JSON-serializable tail from subprocess output."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        text = value.decode(errors="replace")
    else:
        text = str(value)
    return text[-int(n) :]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark forced PAS-TZ memory fallback variants with hard timeouts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--variants", nargs="+", default=["collision", "hybrid", "zeta", "theta", "tzfft"])
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--maxiter", type=int, default=8)
    parser.add_argument("--restart", type=int, default=12)
    parser.add_argument("--tol", type=float, default=1.0e-6)
    parser.add_argument("--solve-method", default="incremental", help="Krylov solve method passed to the child solve.")
    parser.add_argument("--block", type=int, default=3)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--Ntheta", "--ntheta", dest="Ntheta", type=int, help="Override resolutionParameters.Ntheta.")
    parser.add_argument("--Nzeta", "--nzeta", dest="Nzeta", type=int, help="Override resolutionParameters.Nzeta.")
    parser.add_argument("--Nxi", "--nxi", dest="Nxi", type=int, help="Override resolutionParameters.Nxi.")
    parser.add_argument("--Nx", "--nx", dest="Nx", type=int, help="Override resolutionParameters.Nx.")
    parser.add_argument("--dry-run", action="store_true", help="Write planned variants without running subprocesses.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser


def _grid_overrides(args: argparse.Namespace) -> dict[str, int]:
    """Return requested positive grid overrides keyed by namelist variable."""
    overrides: dict[str, int] = {}
    for key in GRID_OVERRIDE_KEYS:
        value = getattr(args, key, None)
        if value is None:
            continue
        value_i = int(value)
        if value_i <= 0:
            raise ValueError(f"{key} override must be positive, got {value_i}")
        overrides[key] = value_i
    return overrides


def _input_record(input_path: Path) -> str:
    try:
        return str(input_path.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(input_path)


def _override_namelist_text(text: str, overrides: dict[str, int]) -> str:
    """Apply simple scalar grid overrides to an existing SFINCS namelist."""
    updated = text
    for key, value in overrides.items():
        pattern = re.compile(rf"^(\s*{re.escape(key)}\s*=\s*)[^\n!]*?(\s*(?:!.*)?)$", re.MULTILINE)
        updated, count = pattern.subn(rf"\g<1>{value}\2", updated, count=1)
        if count != 1:
            raise ValueError(f"Could not find active namelist assignment for {key}")
    return updated


def _write_child_input(input_path: Path, work_dir: Path, overrides: dict[str, int]) -> Path:
    """Write a temporary child input namelist with requested grid overrides."""
    child_input = work_dir / "input.namelist"
    text = input_path.read_text()
    child_input.write_text(_override_namelist_text(text, overrides))
    return child_input


def _variant_env(variant: str, *, block: int, overlap: int, maxiter: int, restart: int) -> dict[str, str]:
    """Return environment overrides for one forced PAS-TZ fallback variant."""
    variant_l = str(variant).strip().lower().replace("-", "_")
    variant_core = variant_l.removesuffix("_lgmres")
    fallback_variant = variant_core
    structured_levels = ""
    if variant_core in {"collision_tzfft", "collision_tzfft_correction", "tzfft_correction"}:
        fallback_variant = "collision"
    elif variant_core in {"tzfft_structured", "tzfft_structured_default", "tzfft_xmg_collision"}:
        fallback_variant = "tzfft"
        structured_levels = "xmg,collision"
    elif variant_core in {"tzfft_xmg", "tzfft_structured_xmg"}:
        fallback_variant = "tzfft"
        structured_levels = "xmg"
    elif variant_core in {"tzfft_collision", "tzfft_structured_collision"}:
        fallback_variant = "tzfft"
        structured_levels = "collision"
    env = {
        "SFINCS_JAX_FORTRAN_STDOUT": "0",
        "SFINCS_JAX_SOLVER_ITER_STATS": "1",
        "SFINCS_JAX_RHSMODE1_PRECONDITIONER": "pas_tz",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES": "1",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK": fallback_variant,
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK": str(int(block)),
        "SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP": str(int(overlap)),
        "SFINCS_JAX_GMRES_MAXITER": str(int(maxiter)),
        "SFINCS_JAX_GMRES_RESTART": str(int(restart)),
    }
    if variant_core in {"collision_tzfft", "collision_tzfft_correction", "tzfft_correction"}:
        env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_CORRECTION"] = "tzfft"
    if structured_levels:
        env["SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS"] = structured_levels
    return env


def _variant_solve_method(variant: str, default: str) -> str:
    """Return the child solve method for a variant name."""
    variant_l = str(variant).strip().lower().replace("-", "_")
    if variant_l.endswith("_lgmres") or variant_l == "lgmres":
        return "lgmres"
    return str(default)


def _child_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Run one forced fallback solve in the current process and return metrics."""
    from sfincs_jax.namelist import read_sfincs_input
    from sfincs_jax.profiling import _resource_maxrss_to_mb
    from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres

    messages: list[str] = []

    def emit(_level: int, msg: str) -> None:
        msg_s = str(msg)
        messages.append(msg_s)
        if "preconditioner" in msg_s or "GMRES" in msg_s or "complete" in msg_s:
            print(msg_s, flush=True)

    t0 = time.perf_counter()
    nml = read_sfincs_input(args.input)
    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        tol=float(args.tol),
        maxiter=int(args.maxiter),
        restart=int(args.restart),
        solve_method=str(args.solve_method),
        emit=emit,
    )
    elapsed_s = time.perf_counter() - t0
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_rss_mb = _resource_maxrss_to_mb(float(usage.ru_maxrss))
    metadata = dict(result.metadata or {})
    return {
        "status": "ok",
        "elapsed_s": float(elapsed_s),
        "max_rss_mb": max_rss_mb,
        "residual_norm": float(result.residual_norm),
        "metadata": metadata,
        "messages_tail": messages[-40:],
    }


def _run_child(args: argparse.Namespace, variant: str) -> dict[str, Any]:
    """Run one variant in a subprocess and return a bounded result row."""
    env = os.environ.copy()
    env.update(
        _variant_env(
            variant,
            block=int(args.block),
            overlap=int(args.overlap),
            maxiter=int(args.maxiter),
            restart=int(args.restart),
        )
    )
    input_path = Path(args.input)
    overrides = _grid_overrides(args)
    tmp_ctx = tempfile.TemporaryDirectory(prefix="sfincs-jax-pas-tz-") if overrides else None
    t0 = time.perf_counter()
    try:
        if tmp_ctx is not None:
            input_path = _write_child_input(input_path, Path(tmp_ctx.name), overrides)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--input",
            str(input_path),
            "--tol",
            str(args.tol),
            "--maxiter",
            str(args.maxiter),
            "--restart",
            str(args.restart),
            "--solve-method",
            _variant_solve_method(str(variant), str(args.solve_method)),
        ]
        try:
            completed = subprocess.run(
                cmd,
                env=env,
                text=True,
                capture_output=True,
                timeout=float(args.timeout_s),
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "variant": str(variant),
                "status": "timeout",
                "elapsed_s": float(time.perf_counter() - t0),
                "timeout_s": float(args.timeout_s),
                "stdout_tail": _tail_text(exc.stdout),
                "stderr_tail": _tail_text(exc.stderr),
            }
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    payload: dict[str, Any] | None = None
    for line in completed.stdout.splitlines()[::-1]:
        if line.startswith(RESULT_MARKER):
            payload = json.loads(line[len(RESULT_MARKER) :])
            break
    if payload is None:
        payload = {
            "status": "error",
            "returncode": int(completed.returncode),
            "stdout_tail": completed.stdout[-4000:],
            "stderr_tail": completed.stderr[-4000:],
        }
    payload["variant"] = str(variant)
    payload["returncode"] = int(completed.returncode)
    payload.setdefault("elapsed_s", float(time.perf_counter() - t0))
    if completed.returncode != 0 and payload.get("status") == "ok":
        payload["status"] = "error"
    return payload


def build_plan(args: argparse.Namespace) -> dict[str, Any]:
    """Build the benchmark plan payload."""
    input_path = Path(args.input)
    overrides = _grid_overrides(args)
    return {
        "input": _input_record(input_path),
        "input_overrides": overrides,
        "timeout_s": float(args.timeout_s),
        "tol": float(args.tol),
        "solve_method": str(args.solve_method),
        "maxiter": int(args.maxiter),
        "restart": int(args.restart),
        "block": int(args.block),
        "overlap": int(args.overlap),
        "variants": list(args.variants),
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.child:
        payload = _child_payload(args)
        print(RESULT_MARKER + json.dumps(payload, sort_keys=True))
        return 0

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "pas_tz_memory_fallback_benchmark",
        "plan": build_plan(args),
        "results": [],
    }
    if not args.dry_run:
        for variant in args.variants:
            row = _run_child(args, str(variant))
            payload["results"].append(row)
            print(f"{variant}: {row.get('status')} elapsed={float(row.get('elapsed_s', 0.0)):.2f}s", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
