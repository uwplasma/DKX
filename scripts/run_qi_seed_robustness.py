#!/usr/bin/env python
"""Build and optionally run deterministic QI seed-robustness cases.

The checked-in quasi-isodynamic VMEC example is expensive at authored
resolution, so this lane creates reproducible neighboring smoke decks around
that input. By default it only writes inputs and a manifest; pass ``--execute``
to run each seed through ``sfincs_jax write-output``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QI_INPUT = REPO_ROOT / "examples" / "additional_examples" / "input.namelist"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "qi_seed_robustness"
RESOLUTION_KEYS = ("NTHETA", "NZETA", "NX", "NXI")


def _read_resolution(text: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in RESOLUTION_KEYS:
        value = _read_number_parameter(text, key)
        if value is not None:
            out[key] = int(round(float(value)))
    return out


def _read_number_parameter(text: str, key: str) -> float | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([-+0-9.eEdD]+)", text)
    if match is None:
        return None
    try:
        return float(match.group(1).replace("D", "E").replace("d", "e"))
    except ValueError:
        return None


def _read_string_parameter(text: str, key: str) -> str | None:
    match = re.search(rf"(?im)^\s*{re.escape(key)}\s*=\s*([^!\n]+)", text)
    if match is None:
        return None
    value = match.group(1).strip().rstrip(",").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _replace_or_append_parameter(text: str, *, group: str, key: str, value: str) -> str:
    pattern = re.compile(rf"(?im)^(\s*{re.escape(key)}\s*=\s*)([^!\n]*?)(\s*(?:!.*)?)$")
    if pattern.search(text):
        return pattern.sub(rf"\g<1>{value}\3", text, count=1)

    group_pattern = re.compile(rf"(?ims)(^\s*&{re.escape(group)}\b.*?)(^\s*/\s*$)")
    group_match = group_pattern.search(text)
    if group_match is not None:
        return text[: group_match.start(2)] + f"  {key} = {value}\n" + text[group_match.start(2) :]

    return text.rstrip() + f"\n\n&{group}\n  {key} = {value}\n/\n"


def _normalize_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip() + "\n"


def _hash_unit(seed: int, label: str) -> float:
    digest = hashlib.sha256(f"{int(seed)}:{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def _signed_jitter(seed: int, label: str) -> float:
    return 2.0 * _hash_unit(seed, label) - 1.0


def _scaled_resolution(
    resolution: dict[str, int],
    *,
    scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
) -> dict[str, int]:
    def scaled_value(key: str, minimum: int) -> int:
        source = int(resolution.get(key, minimum))
        return max(int(minimum), int(round(source * float(scale))))

    out = {
        "NTHETA": scaled_value("NTHETA", min_ntheta),
        "NZETA": scaled_value("NZETA", min_nzeta),
        "NX": scaled_value("NX", min_nx),
        "NXI": scaled_value("NXI", min_nxi),
    }
    for key in ("NTHETA", "NZETA"):
        if int(resolution.get(key, out[key])) % 2 == 1 and out[key] % 2 == 0:
            out[key] += 1
    return out


def _resolve_equilibrium(input_path: Path, text: str) -> Path | None:
    raw = _read_string_parameter(text, "equilibriumFile")
    if not raw:
        return None
    candidate = Path(raw)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    relative = (input_path.parent / candidate).resolve()
    if relative.exists():
        return relative
    by_basename = input_path.parent / candidate.name
    if by_basename.exists():
        return by_basename.resolve()
    return None


def _case_command(case_dir: Path, *, solve_method: str) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "sfincs_jax",
        "write-output",
        "--input",
        str(case_dir / "input.namelist"),
        "--out",
        str(case_dir / "sfincsOutput_jax.h5"),
        "--solver-trace",
        str(case_dir / "sfincsOutput_jax.solver_trace.json"),
    ]
    if str(solve_method).strip().lower() not in {"", "auto", "default"}:
        command.extend(["--solve-method", str(solve_method)])
    return command


def _materialize_case(
    *,
    seed: int,
    source_input: Path,
    source_text: str,
    source_resolution: dict[str, int],
    source_equilibrium: Path | None,
    out_root: Path,
    resolution_scale: float,
    min_ntheta: int,
    min_nzeta: int,
    min_nx: int,
    min_nxi: int,
    nu_jitter: float,
    er_jitter: float,
    solve_method: str,
) -> dict[str, object]:
    case_name = f"qi_seed_{int(seed):04d}"
    case_dir = out_root / case_name
    case_dir.mkdir(parents=True, exist_ok=True)

    text = source_text
    resolution = _scaled_resolution(
        source_resolution,
        scale=resolution_scale,
        min_ntheta=min_ntheta,
        min_nzeta=min_nzeta,
        min_nx=min_nx,
        min_nxi=min_nxi,
    )
    for key, value in resolution.items():
        text = _replace_or_append_parameter(text, group="resolutionParameters", key=key, value=str(int(value)))

    base_nu = _read_number_parameter(source_text, "nu_n")
    base_er = _read_number_parameter(source_text, "Er")
    nu_factor = 1.0 + float(nu_jitter) * _signed_jitter(seed, "nu_n")
    er_delta = float(er_jitter) * _signed_jitter(seed, "Er")
    nu_value = None if base_nu is None else float(base_nu) * nu_factor
    er_value = None if base_er is None else float(base_er) + er_delta
    if nu_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="nu_n", value=f"{nu_value:.12g}")
    if er_value is not None:
        text = _replace_or_append_parameter(text, group="physicsParameters", key="Er", value=f"{er_value:.12g}")

    copied_equilibrium = None
    if source_equilibrium is not None:
        copied_equilibrium = case_dir / source_equilibrium.name
        if source_equilibrium.resolve() != copied_equilibrium.resolve():
            shutil.copy2(source_equilibrium, copied_equilibrium)
        text = _replace_or_append_parameter(
            text,
            group="geometryParameters",
            key="equilibriumFile",
            value=f"'{copied_equilibrium.name}'",
        )

    input_path = case_dir / "input.namelist"
    input_path.write_text(_normalize_text(text), encoding="utf-8")
    (case_dir / "input.source.namelist").write_text(_normalize_text(source_text), encoding="utf-8")
    command = _case_command(case_dir, solve_method=solve_method)
    return {
        "case": case_name,
        "seed": int(seed),
        "input": str(input_path.relative_to(out_root)),
        "output": str((case_dir / "sfincsOutput_jax.h5").relative_to(out_root)),
        "solver_trace": str((case_dir / "sfincsOutput_jax.solver_trace.json").relative_to(out_root)),
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "copied_equilibrium": str(copied_equilibrium.relative_to(out_root)) if copied_equilibrium is not None else None,
        "solve_method": str(solve_method),
        "resolution": resolution,
        "perturbations": {
            "nu_n": nu_value,
            "nu_factor": nu_factor if nu_value is not None else None,
            "Er": er_value,
            "Er_delta": er_delta if er_value is not None else None,
        },
        "command": command,
    }


def _finite_float_or_none(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if out != out or out in {float("inf"), float("-inf")}:
        return None
    return out


def _solver_trace_summary(trace_path: Path) -> dict[str, object] | None:
    if not trace_path.exists():
        return None
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"path": str(trace_path), "readable": False}

    residual_norm = _finite_float_or_none(payload.get("residual_norm"))
    residual_target = _finite_float_or_none(payload.get("residual_target"))
    residual_ratio = None
    if residual_norm is not None and residual_target is not None and residual_target > 0.0:
        residual_ratio = residual_norm / residual_target
    metadata = payload.get("metadata")
    solver_metadata = metadata.get("solver_metadata", {}) if isinstance(metadata, dict) else {}
    return {
        "path": str(trace_path),
        "readable": True,
        "solve_method": payload.get("solve_method"),
        "selected_path": payload.get("selected_path"),
        "backend": payload.get("backend"),
        "elapsed_s": _finite_float_or_none(payload.get("elapsed_s")),
        "residual_norm": residual_norm,
        "residual_target": residual_target,
        "residual_ratio": residual_ratio,
        "converged": payload.get("converged"),
        "accepted_converged": solver_metadata.get("accepted_converged"),
        "acceptance_criterion": solver_metadata.get("acceptance_criterion"),
        "iterations": solver_metadata.get("iterations"),
        "solver_kind": solver_metadata.get("solver_kind"),
    }


def _execute_cases(out_root: Path, cases: Iterable[dict[str, object]], *, timeout_s: float, fail_fast: bool) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        command = [str(part) for part in case["command"]]  # type: ignore[index]
        case_dir = out_root / str(case["case"])
        stdout_path = case_dir / "sfincs_jax.stdout.log"
        stderr_path = case_dir / "sfincs_jax.stderr.log"
        trace_path = case_dir / "sfincsOutput_jax.solver_trace.json"
        start = time.perf_counter()
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            try:
                completed = subprocess.run(
                    command,
                    cwd=REPO_ROOT,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=float(timeout_s),
                    check=False,
                )
                returncode = int(completed.returncode)
                timed_out = False
            except subprocess.TimeoutExpired:
                stderr.write(f"\nQI seed execution timed out after {float(timeout_s):.3f} s.\n")
                returncode = 124
                timed_out = True
        elapsed_s = time.perf_counter() - start
        result = {
            "case": case["case"],
            "seed": case["seed"],
            "returncode": returncode,
            "timed_out": timed_out,
            "elapsed_s": elapsed_s,
            "stdout": str(stdout_path.relative_to(out_root)),
            "stderr": str(stderr_path.relative_to(out_root)),
            "output_exists": (case_dir / "sfincsOutput_jax.h5").exists(),
            "solver_trace_exists": trace_path.exists(),
            "solver_trace_summary": _solver_trace_summary(trace_path),
        }
        results.append(result)
        if returncode != 0 and fail_fast:
            break
    return results


def _execution_summary(results: Iterable[dict[str, object]]) -> dict[str, object]:
    """Return compact aggregate diagnostics for an executed seed ladder."""
    result_list = list(results)
    trace_summaries = [
        result.get("solver_trace_summary")
        for result in result_list
        if isinstance(result.get("solver_trace_summary"), dict)
    ]
    residual_ratios = [
        float(summary["residual_ratio"])
        for summary in trace_summaries
        if _finite_float_or_none(summary.get("residual_ratio")) is not None
    ]
    elapsed_values = [
        float(result["elapsed_s"])
        for result in result_list
        if _finite_float_or_none(result.get("elapsed_s")) is not None
    ]
    return {
        "attempted": len(result_list),
        "process_passed": sum(1 for result in result_list if int(result["returncode"]) == 0),
        "process_failed": sum(1 for result in result_list if int(result["returncode"]) != 0),
        "timed_out": sum(1 for result in result_list if bool(result.get("timed_out"))),
        "outputs_written": sum(1 for result in result_list if bool(result.get("output_exists"))),
        "solver_traces_written": sum(1 for result in result_list if bool(result.get("solver_trace_exists"))),
        "converged": sum(1 for summary in trace_summaries if summary.get("converged") is True),
        "accepted_converged": sum(1 for summary in trace_summaries if summary.get("accepted_converged") is True),
        "max_residual_ratio": max(residual_ratios) if residual_ratios else None,
        "max_elapsed_s": max(elapsed_values) if elapsed_values else None,
        "backends": sorted({str(summary.get("backend")) for summary in trace_summaries if summary.get("backend")}),
        "solve_methods": sorted(
            {str(summary.get("solve_method")) for summary in trace_summaries if summary.get("solve_method")}
        ),
        "selected_paths": sorted(
            {str(summary.get("selected_path")) for summary in trace_summaries if summary.get("selected_path")}
        ),
    }


def _evaluate_execution_gates(
    results: Iterable[dict[str, object]],
    *,
    max_residual_ratio: float | None,
    require_converged: bool,
    require_accepted_converged: bool,
) -> dict[str, object]:
    """Evaluate optional seed-ladder promotion gates against executed cases."""
    failures: list[dict[str, object]] = []
    for result in results:
        case_name = str(result.get("case"))
        returncode = int(result.get("returncode", 1))
        if returncode != 0:
            failures.append({"case": case_name, "reason": "process_failed", "returncode": returncode})
            continue

        summary = result.get("solver_trace_summary")
        if (max_residual_ratio is not None or require_converged or require_accepted_converged) and not isinstance(
            summary, dict
        ):
            failures.append({"case": case_name, "reason": "missing_solver_trace_summary"})
            continue

        if isinstance(summary, dict) and max_residual_ratio is not None:
            residual_ratio = _finite_float_or_none(summary.get("residual_ratio"))
            if residual_ratio is None:
                failures.append({"case": case_name, "reason": "missing_residual_ratio"})
            elif residual_ratio > float(max_residual_ratio):
                failures.append(
                    {
                        "case": case_name,
                        "reason": "residual_ratio_exceeded",
                        "residual_ratio": residual_ratio,
                        "max_residual_ratio": float(max_residual_ratio),
                    }
                )

        if isinstance(summary, dict) and bool(require_converged) and summary.get("converged") is not True:
            failures.append({"case": case_name, "reason": "not_converged"})

        if (
            isinstance(summary, dict)
            and bool(require_accepted_converged)
            and summary.get("accepted_converged") is not True
        ):
            failures.append({"case": case_name, "reason": "not_accepted_converged"})

    return {
        "passed": not failures,
        "failures": failures,
        "max_residual_ratio": max_residual_ratio,
        "require_converged": bool(require_converged),
        "require_accepted_converged": bool(require_accepted_converged),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_QI_INPUT, help="Base QI input.namelist.")
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="Directory for generated seed cases.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2], help="Deterministic seed ids to materialize.")
    parser.add_argument("--resolution-scale", type=float, default=0.25, help="Scale applied to NTHETA/NZETA/NX/NXI.")
    parser.add_argument("--min-ntheta", type=int, default=7)
    parser.add_argument("--min-nzeta", type=int, default=11)
    parser.add_argument("--min-nx", type=int, default=4)
    parser.add_argument("--min-nxi", type=int, default=16)
    parser.add_argument("--nu-jitter", type=float, default=0.05, help="Relative symmetric nu_n jitter per seed.")
    parser.add_argument("--er-jitter", type=float, default=0.02, help="Additive symmetric Er jitter per seed.")
    parser.add_argument(
        "--solve-method",
        default="auto",
        help=(
            "RHSMode=1 solve method passed to sfincs_jax write-output when --execute is set. "
            "The default is auto, which exercises the public CLI solver policy. Pass an explicit "
            "method such as dense or sparse_lsmr only for diagnostic probes."
        ),
    )
    parser.add_argument("--execute", action="store_true", help="Run each generated seed through sfincs_jax write-output.")
    parser.add_argument("--timeout-s", type=float, default=300.0, help="Per-seed execution timeout.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop executing after the first failed seed.")
    parser.add_argument(
        "--max-residual-ratio",
        type=float,
        default=None,
        help="Optional promotion gate: every solver trace residual_norm/residual_target must be at or below this value.",
    )
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Optional promotion gate: require every solver trace to report converged=true.",
    )
    parser.add_argument(
        "--require-accepted-converged",
        action="store_true",
        help="Optional promotion gate: require every solver trace metadata to report accepted_converged=true.",
    )
    parser.add_argument("--clean", action="store_true", help="Remove --out-root before materializing cases.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_input = Path(args.input).resolve()
    if not source_input.exists():
        raise FileNotFoundError(source_input)

    out_root = Path(args.out_root).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    source_text = source_input.read_text(encoding="utf-8")
    source_resolution = _read_resolution(source_text)
    source_equilibrium = _resolve_equilibrium(source_input, source_text)
    cases = [
        _materialize_case(
            seed=int(seed),
            source_input=source_input,
            source_text=source_text,
            source_resolution=source_resolution,
            source_equilibrium=source_equilibrium,
            out_root=out_root,
            resolution_scale=float(args.resolution_scale),
            min_ntheta=int(args.min_ntheta),
            min_nzeta=int(args.min_nzeta),
            min_nx=int(args.min_nx),
            min_nxi=int(args.min_nxi),
            nu_jitter=float(args.nu_jitter),
            er_jitter=float(args.er_jitter),
            solve_method=str(args.solve_method),
        )
        for seed in args.seeds
    ]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "lane": "qi_seed_robustness",
        "source_input": str(source_input),
        "source_equilibrium": str(source_equilibrium) if source_equilibrium is not None else None,
        "resolution_scale": float(args.resolution_scale),
        "nu_jitter": float(args.nu_jitter),
        "er_jitter": float(args.er_jitter),
        "solve_method": str(args.solve_method),
        "case_count": len(cases),
        "cases": cases,
    }
    if bool(args.execute):
        results = _execute_cases(out_root, cases, timeout_s=float(args.timeout_s), fail_fast=bool(args.fail_fast))
        gates = _evaluate_execution_gates(
            results,
            max_residual_ratio=args.max_residual_ratio,
            require_converged=bool(args.require_converged),
            require_accepted_converged=bool(args.require_accepted_converged),
        )
        manifest["execution"] = {
            "timeout_s": float(args.timeout_s),
            "fail_fast": bool(args.fail_fast),
            "results": results,
            "passed": sum(1 for result in results if int(result["returncode"]) == 0),
            "failed": sum(1 for result in results if int(result["returncode"]) != 0),
            "summary": _execution_summary(results),
            "gates": gates,
        }

    manifest_path = out_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Cases: {len(cases)}")
    if bool(args.execute):
        execution = manifest["execution"]  # type: ignore[index]
        print(f"Executed: {execution['passed']} passed, {execution['failed']} failed")
        gates = execution["gates"]
        if not gates["passed"]:
            print(f"Gates failed: {len(gates['failures'])}")
        return 0 if int(execution["failed"]) == 0 and bool(gates["passed"]) else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
