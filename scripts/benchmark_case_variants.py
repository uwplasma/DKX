#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from sfincs_jax.compare import compare_sfincs_outputs


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
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    case_dir = args.case_dir.resolve()
    case_input = case_dir / "input.namelist"
    if not case_input.exists():
        raise FileNotFoundError(f"Missing {case_input}")
    reference = case_dir / "fortran_run" / "sfincsOutput.h5"
    have_reference = reference.exists()

    variants = [_parse_variant(spec) for spec in args.variant] if args.variant else [("default", {})]
    if not any(name == "default" for name, _ in variants):
        variants = [("default", {})] + variants

    child = r"""
import json, os, resource, time
from pathlib import Path
from sfincs_jax.io import write_sfincs_jax_output_h5

case = Path(os.environ["CASE_INPUT"])
out = Path(os.environ["CASE_OUTPUT"])
t0 = time.perf_counter()
write_sfincs_jax_output_h5(input_namelist=case, output_path=out, compute_solution=True)
elapsed = time.perf_counter() - t0
print("@@RESULT@@" + json.dumps({"elapsed_s": elapsed, "ru_maxrss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}))
"""
    differentiable_arg = "True" if args.differentiable else "False"
    child = child.replace(
        'write_sfincs_jax_output_h5(input_namelist=case, output_path=out, compute_solution=True)',
        'write_sfincs_jax_output_h5(input_namelist=case, output_path=out, compute_solution=True, differentiable='
        + differentiable_arg
        + ')',
    )

    base_env = os.environ.copy()
    base_env.update(
        {
            "PYTHONPATH": str(repo),
            "SFINCS_JAX_FORTRAN_STDOUT": "0",
            "SFINCS_JAX_SOLVER_ITER_STATS": "1",
            "SFINCS_JAX_IMPLICIT_SOLVE": "1" if args.differentiable else "0",
        }
    )

    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix=f"{case_dir.name}_bench_") as td:
        tmp = Path(td)
        outputs: dict[str, Path] = {}
        for name, extra_env in variants:
            env = base_env.copy()
            env.update(extra_env)
            env["CASE_INPUT"] = str(case_input)
            env["CASE_OUTPUT"] = str(tmp / f"{name}.h5")
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
                        "stdout_tail": (exc.stdout or "")[-2500:],
                        "stderr_tail": (exc.stderr or "")[-2000:],
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
                row["stderr_tail"] = proc.stderr[-2000:]
                row["stdout_tail"] = proc.stdout[-2500:]
                rows.append(row)
                continue
            marker = [line for line in proc.stdout.splitlines() if line.startswith("@@RESULT@@")][-1]
            result = json.loads(marker[len("@@RESULT@@") :])
            out_path = tmp / f"{name}.h5"
            outputs[name] = out_path
            row.update(
                {
                    "status": "ok",
                    "elapsed_s": round(float(result["elapsed_s"]), 3),
                    "ru_maxrss_kb": int(result["ru_maxrss_kb"]),
                    "used_adaptive_pas_smoother": "adaptive PAS smoother" in proc.stdout,
                    "used_pas_tokamak_theta": "preconditioner=pas_tokamak_theta" in proc.stdout,
                    "used_lgmres": "solve method forced by env -> lgmres" in proc.stdout,
                    "used_explicit_sparse_helper": "explicit sparse helper" in proc.stdout,
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
        args.json_out.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
