#!/usr/bin/env python3
"""Audit selected Zenodo VMEC cases and optionally compare cached JAX outputs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from sfincs_jax.validation.h5_parity import compare_h5_outputs
from sfincs_jax.input_compat import effective_equilibrium_file
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.artifacts import PhaseTimer

DEFAULT_ZENODO_ROOT = Path(
    os.environ.get(
        "SFINCS_JAX_QS_ZENODO_ROOT",
        "/Users/rogeriojorge/local/20220708-01-zenodo_for_QS_optimization_with_self_consistent_bootstrap_current",
    )
)
DEFAULT_SELECTION = Path("docs/_static/figures/vmec_jax_finite_beta/zenodo_vmec_benchmark_selection.json")
DEFAULT_KEYS = (
    "FSABjHat",
    "FSABjHatOverRootFSAB2",
    "NIterations",
    "FSABFlow",
    "particleFlux_vm_psiHat",
    "heatFlux_vm_psiHat",
    "particleFlux_vm_psiN",
    "heatFlux_vm_psiN",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _safe_token(value: Any) -> str:
    text = str(value)
    return (
        text.replace("/", "__")
        .replace(" ", "_")
        .replace(".", "p")
        .replace("-", "m")
        .replace("+", "p")
        .replace(":", "_")
    )


def _numeric_h5_summary(path: Path, *, keys: tuple[str, ...]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "exists": path.exists(),
        "path": str(path),
    }
    if not path.exists():
        return summary

    summary["size_bytes"] = path.stat().st_size
    dataset_count = 0
    numeric_count = 0
    selected: dict[str, Any] = {}
    with h5py.File(path, "r") as h5:

        def visit(name: str, obj: Any) -> None:
            nonlocal dataset_count, numeric_count
            if not isinstance(obj, h5py.Dataset):
                return
            dataset_count += 1
            dtype_kind = np.dtype(obj.dtype).kind
            if dtype_kind in {"b", "i", "u", "f", "c"}:
                numeric_count += 1

        h5.visititems(visit)
        for key in keys:
            item: dict[str, Any] = {"exists": key in h5}
            if key in h5:
                data = np.asarray(h5[key][...])
                item["shape"] = list(data.shape)
                item["dtype"] = str(data.dtype)
                item["numeric"] = data.dtype.kind in {"b", "i", "u", "f", "c"}
                if item["numeric"] and data.size == 1:
                    item["value"] = data.reshape(-1)[0].item()
            selected[key] = item
    summary["dataset_count"] = dataset_count
    summary["numeric_dataset_count"] = numeric_count
    summary["selected_datasets"] = selected
    return summary


def _load_candidate_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    if isinstance(payload, list):
        out: dict[str, str] = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            key = row.get("input") or row.get("case_id") or row.get("case")
            value = row.get("candidate_h5") or row.get("jax_h5") or row.get("output")
            if key is not None and value is not None:
                out[str(key)] = str(value)
        return out
    raise TypeError(f"Unsupported candidate-map JSON type: {type(payload).__name__}")


def _candidate_format_fields(case: dict[str, Any]) -> dict[str, Any]:
    resolution = case.get("resolution", {}) or {}
    surface = case.get("surface_s")
    return {
        "input": case.get("input"),
        "input_token": _safe_token(case.get("input")),
        "family": case.get("family"),
        "rung": case.get("rung"),
        "surface_role": case.get("surface_role"),
        "surface_s": surface,
        "surface_token": _safe_token(surface),
        "resolution_label": resolution.get("label"),
        "resolution_token": _safe_token(resolution.get("label")),
        "case": case.get("case"),
        "case_token": _safe_token(case.get("case")),
    }


def _candidate_path(
    case: dict[str, Any],
    *,
    candidate_root: Path | None,
    candidate_template: str | None,
    candidate_map: dict[str, str],
) -> Path | None:
    for key in (str(case.get("input")), str(case.get("case")), str(case.get("case_id"))):
        if key in candidate_map:
            return Path(candidate_map[key]).expanduser()
    if candidate_root is None or candidate_template is None:
        return None
    fields = _candidate_format_fields(case)
    return candidate_root.expanduser() / candidate_template.format(**fields)


def _solve_case_dir(run_root: Path, case: dict[str, Any], index: int) -> Path:
    resolution = case.get("resolution", {}) or {}
    parts = [
        f"{int(index):03d}",
        _safe_token(case.get("family")),
        _safe_token(case.get("rung")),
        _safe_token(case.get("surface_role")),
        _safe_token(resolution.get("label")),
    ]
    return run_root / "_".join(parts)


def _build_equilibrium_index(search_root: Path | None) -> dict[str, list[Path]]:
    if search_root is None:
        return {}
    root = search_root.expanduser()
    if not root.exists():
        return {}
    index: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.nc")):
        index.setdefault(path.name, []).append(path)
    return index


def _equilibrium_override_for_input(input_path: Path, equilibrium_index: dict[str, list[Path]]) -> Path | None:
    try:
        nml = read_sfincs_input(input_path)
    except Exception:  # noqa: BLE001
        return None
    raw = effective_equilibrium_file(geom_params=nml.group("geometryParameters"))
    if raw is None:
        return None
    text = str(raw).strip().strip('"').strip("'")
    direct = Path(text).expanduser()
    if direct.exists():
        return direct
    basename = direct.name
    candidates = equilibrium_index.get(basename, [])
    return candidates[0] if candidates else None


def _tail(text: str | bytes | None, limit: int) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return str(text)[-int(limit) :]


def _parse_solve_result_marker(stdout: str) -> dict[str, Any] | None:
    marker = "@@SFINCS_JAX_CAMPAIGN_RESULT@@"
    for line in reversed(str(stdout).splitlines()):
        if marker not in line:
            continue
        _, payload = line.split(marker, 1)
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return None
    return None


def _extract_solve_progress(stdout: str | bytes | None, stderr: str | bytes | None) -> dict[str, Any]:
    combined = f"{_tail(stdout, 20000)}\n{_tail(stderr, 10000)}"
    progress_markers = (
        "profiling:",
        "timing:",
        "VMEC:",
        "The matrix is",
        "Runtime hint:",
        "Solver note:",
        "solve_v3_full_system_linear_gmres:",
        "Entering main solver loop",
        "write_sfincs_jax_output_h5:",
        "building RHSMode=1 preconditioner=",
    )
    progress_lines = [line for line in combined.splitlines() if any(marker in line for marker in progress_markers)]
    profile_labels: list[str] = []
    for line in progress_lines:
        if "profiling:" not in line:
            continue
        tail = line.split("profiling:", 1)[1].strip()
        if tail:
            profile_labels.append(tail.split()[0])
    krylov_progress: list[dict[str, Any]] = []
    for line in progress_lines:
        if "solve_v3_full_system_linear_gmres:" not in line:
            continue
        count_match = re.search(r"\b(?P<kind>iters|matvecs)=(?P<count>\d+)\b", line)
        if count_match is None:
            continue
        item: dict[str, Any] = {
            "line": line,
            "kind": count_match.group("kind"),
            "count": int(count_match.group("count")),
        }
        residual_match = re.search(r"\b(?:ksp_residual|residual)=(?P<residual>[-+0-9.eE]+)\b", line)
        if residual_match is not None:
            item["residual"] = float(residual_match.group("residual"))
        elapsed_match = re.search(r"\belapsed_s=(?P<elapsed>[-+0-9.eE]+)\b", line)
        if elapsed_match is not None:
            item["elapsed_s"] = float(elapsed_match.group("elapsed"))
        krylov_progress.append(item)
    residuals = [float(item["residual"]) for item in krylov_progress if "residual" in item]
    return {
        "progress_line_count": len(progress_lines),
        "profile_label_count": len(profile_labels),
        "profile_labels": profile_labels,
        "krylov_progress_count": len(krylov_progress),
        "last_krylov_progress": krylov_progress[-1] if krylov_progress else None,
        "max_krylov_count": max((int(item["count"]) for item in krylov_progress), default=None),
        "last_krylov_residual": residuals[-1] if residuals else None,
        "min_krylov_residual": min(residuals, default=None),
        "last_progress_lines": progress_lines[-20:],
    }


def _jsonable_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    return None


def _compact_solver_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Keep high-value solver-path fields without copying long histories."""

    prefixes = (
        "scipy_rescue_",
        "sparse_xblock_rescue_",
        "fp_xblock_global_correction_",
        "fp_xblock_highx_residual_correction_",
        "xblock_assembled_operator_",
        "xblock_row_equilibration_",
        "xblock_col_equilibration_",
    )
    exact_keys = {
        "solver_kind",
        "resolved_solver_kind",
        "realized_solver_kind",
        "preconditioner_kind",
        "xblock_active_dof",
        "xblock_linear_size",
        "used_explicit_fp_xblock_seed",
        "used_large_cpu_xblock_shortcut",
        "failure_reason",
        "output_refused",
    }
    out: dict[str, Any] = {}
    for key, value in sorted(metadata.items()):
        key_s = str(key)
        if key_s not in exact_keys and not any(key_s.startswith(prefix) for prefix in prefixes):
            continue
        scalar = _jsonable_scalar(value)
        if scalar is not None or value is None:
            out[key_s] = scalar
        elif isinstance(value, (list, tuple)):
            out[f"{key_s}_len"] = len(value)
    return out


def _solver_trace_summary(trace_path: Path) -> dict[str, Any]:
    """Return a compact campaign-facing summary of a solver-trace sidecar."""

    summary: dict[str, Any] = {"exists": trace_path.exists(), "path": str(trace_path)}
    if not trace_path.exists():
        return summary
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        summary.update({"readable": False, "error": f"{type(exc).__name__}: {exc}"})
        return summary

    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    solver_metadata = metadata.get("solver_metadata") if isinstance(metadata.get("solver_metadata"), dict) else {}
    profile_entries = metadata.get("profile_entries") if isinstance(metadata.get("profile_entries"), list) else []
    profile_labels: list[str] = []
    max_profile_label = None
    max_profile_dt_s = None
    for entry in profile_entries:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", ""))
        if label:
            profile_labels.append(label)
        try:
            dt_s = float(entry.get("dt_s", 0.0))
        except (TypeError, ValueError):
            continue
        if max_profile_dt_s is None or dt_s > max_profile_dt_s:
            max_profile_dt_s = dt_s
            max_profile_label = label or None

    residual_norm = payload.get("residual_norm")
    residual_target = payload.get("residual_target")
    try:
        residual_ratio = float(residual_norm) / max(float(residual_target), 1.0e-300)
    except (TypeError, ValueError):
        residual_ratio = None

    candidate_decisions = payload.get("candidate_decisions")
    if not isinstance(candidate_decisions, list):
        candidate_decisions = []
    accepted_candidates = [
        str(item.get("name"))
        for item in candidate_decisions
        if isinstance(item, dict) and bool(item.get("accepted")) and item.get("name") is not None
    ]

    summary.update(
        {
            "readable": True,
            "schema_version": payload.get("schema_version"),
            "backend": payload.get("backend"),
            "rhs_mode": payload.get("rhs_mode"),
            "geometry_scheme": payload.get("geometry_scheme"),
            "solve_method": payload.get("solve_method"),
            "selected_path": payload.get("selected_path"),
            "converged": payload.get("converged"),
            "total_size": payload.get("total_size"),
            "active_size": payload.get("active_size"),
            "residual_norm": residual_norm,
            "residual_target": residual_target,
            "residual_ratio": residual_ratio,
            "elapsed_s": payload.get("elapsed_s"),
            "setup_s": payload.get("setup_s"),
            "solve_s": payload.get("solve_s"),
            "peak_rss_mb": payload.get("peak_rss_mb"),
            "active_rss_mb": payload.get("active_rss_mb"),
            "device_peak_mb": payload.get("device_peak_mb"),
            "estimated_dense_nbytes": payload.get("estimated_dense_nbytes"),
            "estimated_csr_nbytes": payload.get("estimated_csr_nbytes"),
            "estimated_gmres_basis_nbytes": payload.get("estimated_gmres_basis_nbytes"),
            "profile_entry_count": len(profile_entries),
            "profile_labels": profile_labels,
            "max_profile_label": max_profile_label,
            "max_profile_dt_s": max_profile_dt_s,
            "accepted_candidates": accepted_candidates,
            "solver_metadata": _compact_solver_metadata(solver_metadata),
            "failure_reason": metadata.get("failure_reason"),
            "output_refused": metadata.get("output_refused"),
        }
    )
    return summary


def _solve_child_script() -> str:
    return r"""
import json
import os
import resource
import sys
import time
import traceback

from sfincs_jax.cli import main

args = json.loads(os.environ["SFINCS_JAX_CAMPAIGN_CLI_ARGS"])
t0 = time.perf_counter()
try:
    rc = int(main(args))
except SystemExit as exc:
    try:
        rc = int(exc.code)
    except Exception:
        rc = 1
except Exception:  # noqa: BLE001
    traceback.print_exc()
    rc = 1
elapsed = time.perf_counter() - t0
raw_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
if sys.platform.startswith("darwin"):
    rss_mb = raw_rss / (1024.0 * 1024.0)
else:
    rss_mb = raw_rss / 1024.0
print(
    "@@SFINCS_JAX_CAMPAIGN_RESULT@@"
    + json.dumps({"elapsed_s": elapsed, "ru_maxrss_raw": raw_rss, "ru_maxrss_mb": rss_mb, "returncode": rc}),
    flush=True,
)
raise SystemExit(rc)
"""


def _solve_cli_args(
    *,
    input_path: Path,
    output_path: Path,
    trace_path: Path,
    equilibrium_path: Path | None,
    solve_method: str,
    quiet: bool,
) -> list[str]:
    args = []
    args.extend(["-q"] if quiet else ["-v"])
    args.extend(
        [
            "write-output",
            "--input",
            str(input_path),
            "--out",
            str(output_path),
            "--solver-trace",
            str(trace_path),
            "--compute-solution",
            "--solve-method",
            str(solve_method),
        ]
    )
    if equilibrium_path is not None:
        args.extend(["--wout-path", str(equilibrium_path)])
    return args


def _run_solve(
    *,
    cli_args: list[str],
    cwd: Path,
    timeout_s: float,
    python_executable: str,
    extra_env: dict[str, str],
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(extra_env)
    env["SFINCS_JAX_CAMPAIGN_CLI_ARGS"] = json.dumps(cli_args)
    try:
        proc = subprocess.run(
            [python_executable, "-c", _solve_child_script()],
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(timeout_s),
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "returncode": None,
            "timeout_s": float(timeout_s),
            "stdout_tail": _tail(exc.stdout, 5000),
            "stderr_tail": _tail(exc.stderr, 5000),
            "progress": _extract_solve_progress(exc.stdout, exc.stderr),
            "result": None,
        }
    return {
        "status": "completed" if proc.returncode == 0 else "error",
        "returncode": int(proc.returncode),
        "timeout_s": float(timeout_s),
        "stdout_tail": _tail(proc.stdout, 5000),
        "stderr_tail": _tail(proc.stderr, 5000),
        "progress": _extract_solve_progress(proc.stdout, proc.stderr),
        "result": _parse_solve_result_marker(proc.stdout),
    }


def _case_status(
    *,
    mode: str,
    reference: dict[str, Any],
    candidate: dict[str, Any] | None,
    parity: dict[str, Any] | None,
    solve: dict[str, Any] | None = None,
) -> str:
    if not reference.get("exists"):
        return "reference_missing"
    if mode == "reference-only":
        return "reference_ok"
    if mode == "solve" and solve is not None:
        if solve.get("status") == "dry_run":
            return "solve_dry_run"
        if solve.get("status") == "timeout":
            return "solve_timeout"
        if solve.get("status") == "error":
            return "solve_error"
    if candidate is None or not candidate.get("exists"):
        return "candidate_missing"
    if parity is None:
        return "candidate_present"
    return "parity_pass" if parity.get("overall_status") == "pass" else "parity_fail"


def _solve_trace_campaign_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    traces: list[dict[str, Any]] = []
    for row in rows:
        solve = row.get("solve")
        if not isinstance(solve, dict):
            continue
        trace = solve.get("solver_trace")
        if isinstance(trace, dict):
            traces.append(trace)

    selected_paths = Counter(str(trace.get("selected_path")) for trace in traces if trace.get("selected_path") is not None)
    backends = Counter(str(trace.get("backend")) for trace in traces if trace.get("backend") is not None)
    rescue_skip_reasons = Counter()
    sparse_xblock_reasons = Counter()
    fp_xblock_global_correction_reasons = Counter()
    fp_xblock_global_correction_accepted = 0
    fp_xblock_highx_residual_correction_reasons = Counter()
    fp_xblock_highx_residual_correction_accepted = 0
    xblock_assembled_built = 0
    xblock_assembled_device_resident = 0
    converged = 0
    output_refused = 0
    residual_ratios: list[float] = []
    for trace in traces:
        if bool(trace.get("converged")):
            converged += 1
        if bool(trace.get("output_refused")):
            output_refused += 1
        ratio = trace.get("residual_ratio")
        if ratio is not None:
            try:
                ratio_f = float(ratio)
            except (TypeError, ValueError):
                ratio_f = np.nan
            if np.isfinite(ratio_f):
                residual_ratios.append(ratio_f)
        metadata = trace.get("solver_metadata")
        if not isinstance(metadata, dict):
            continue
        reason = metadata.get("scipy_rescue_skip_reason")
        if reason is not None:
            rescue_skip_reasons[str(reason)] += 1
        sparse_xblock_reason = metadata.get("sparse_xblock_rescue_reason")
        if sparse_xblock_reason is not None:
            sparse_xblock_reasons[str(sparse_xblock_reason)] += 1
        correction_reason = metadata.get("fp_xblock_global_correction_reason")
        if correction_reason is not None:
            fp_xblock_global_correction_reasons[str(correction_reason)] += 1
        if bool(metadata.get("fp_xblock_global_correction_accepted")):
            fp_xblock_global_correction_accepted += 1
        highx_reason = metadata.get("fp_xblock_highx_residual_correction_reason")
        if highx_reason is not None:
            fp_xblock_highx_residual_correction_reasons[str(highx_reason)] += 1
        if bool(metadata.get("fp_xblock_highx_residual_correction_accepted")):
            fp_xblock_highx_residual_correction_accepted += 1
        if bool(metadata.get("xblock_assembled_operator_built")):
            xblock_assembled_built += 1
        if bool(metadata.get("xblock_assembled_operator_device_resident")):
            xblock_assembled_device_resident += 1

    return {
        "solver_traces_written": sum(1 for trace in traces if bool(trace.get("exists"))),
        "solver_traces_readable": sum(1 for trace in traces if bool(trace.get("readable"))),
        "converged": int(converged),
        "output_refused": int(output_refused),
        "selected_paths": dict(sorted(selected_paths.items())),
        "backends": dict(sorted(backends.items())),
        "scipy_rescue_skip_reasons": dict(sorted(rescue_skip_reasons.items())),
        "sparse_xblock_rescue_reasons": dict(sorted(sparse_xblock_reasons.items())),
        "fp_xblock_global_correction_reasons": dict(sorted(fp_xblock_global_correction_reasons.items())),
        "fp_xblock_global_correction_accepted": int(fp_xblock_global_correction_accepted),
        "fp_xblock_highx_residual_correction_reasons": dict(
            sorted(fp_xblock_highx_residual_correction_reasons.items())
        ),
        "fp_xblock_highx_residual_correction_accepted": int(
            fp_xblock_highx_residual_correction_accepted
        ),
        "xblock_assembled_operator_built": int(xblock_assembled_built),
        "xblock_assembled_operator_device_resident": int(xblock_assembled_device_resident),
        "max_residual_ratio": max(residual_ratios, default=None),
    }


def run_campaign(
    *,
    selection_path: Path,
    zenodo_root: Path,
    mode: str,
    keys: tuple[str, ...] = DEFAULT_KEYS,
    candidate_root: Path | None = None,
    candidate_template: str | None = None,
    candidate_map_path: Path | None = None,
    run_root: Path | None = None,
    timeout_s: float = 600.0,
    max_cases: int | None = None,
    solve_method: str = "auto",
    equilibrium_search_root: Path | None = None,
    dry_run: bool = False,
    quiet_solve: bool = False,
    python_executable: str = sys.executable,
    extra_env: dict[str, str] | None = None,
    atol: float = 1.0e-12,
    rtol: float = 1.0e-12,
) -> dict[str, Any]:
    """Run a bounded parity campaign audit from a benchmark selection JSON."""
    timer = PhaseTimer()
    with timer.phase("load_selection", selection_path=str(selection_path)):
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        cases = list(selection.get("cases", []))
        if max_cases is not None:
            cases = cases[: max(0, int(max_cases))]
        candidate_map = _load_candidate_map(candidate_map_path)
        if mode == "solve" and equilibrium_search_root is None:
            equilibrium_search_root = zenodo_root
        equilibrium_index = _build_equilibrium_index(equilibrium_search_root if mode == "solve" else None)

    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for index, case in enumerate(cases):
        case_timer = PhaseTimer()
        reference_path = zenodo_root.expanduser() / str(case["fortran_output"])
        input_path = zenodo_root.expanduser() / str(case["input"])
        candidate_h5 = _candidate_path(
            case,
            candidate_root=candidate_root,
            candidate_template=candidate_template,
            candidate_map=candidate_map,
        )
        solve_report: dict[str, Any] | None = None
        if mode == "solve":
            if run_root is None:
                raise ValueError("--run-root is required for --mode solve")
            run_dir = _solve_case_dir(run_root.expanduser(), case, index)
            candidate_h5 = run_dir / "sfincsOutput.h5"
            trace_path = run_dir / "solver_trace.json"
            run_dir.mkdir(parents=True, exist_ok=True)
            with case_timer.phase("solve_prepare", case_index=index):
                equilibrium_path = _equilibrium_override_for_input(input_path, equilibrium_index)
                cli_args = _solve_cli_args(
                    input_path=input_path,
                    output_path=candidate_h5,
                    trace_path=trace_path,
                    equilibrium_path=equilibrium_path,
                    solve_method=solve_method,
                    quiet=quiet_solve,
                )
            solve_report = {
                "run_dir": str(run_dir),
                "trace_path": str(trace_path),
                "cli_args": cli_args,
                "equilibrium_path": None if equilibrium_path is None else str(equilibrium_path),
            }
            if dry_run:
                solve_report.update({"status": "dry_run", "returncode": None, "timeout_s": float(timeout_s)})
            else:
                with case_timer.phase("solve_subprocess", case_index=index):
                    solve_report.update(
                        _run_solve(
                            cli_args=cli_args,
                            cwd=Path(__file__).resolve().parents[1],
                            timeout_s=float(timeout_s),
                            python_executable=python_executable,
                            extra_env=extra_env or {},
                        )
                    )
                with case_timer.phase("solver_trace_audit", case_index=index):
                    solve_report["solver_trace"] = _solver_trace_summary(trace_path)

        with case_timer.phase("reference_h5_audit", case_index=index):
            reference = _numeric_h5_summary(reference_path, keys=keys)

        candidate: dict[str, Any] | None = None
        parity: dict[str, Any] | None = None
        if mode in {"cached-compare", "solve"}:
            if candidate_h5 is not None:
                with case_timer.phase("candidate_h5_audit", case_index=index):
                    candidate = _numeric_h5_summary(candidate_h5, keys=keys)
                if reference.get("exists") and candidate.get("exists"):
                    with case_timer.phase("h5_parity", case_index=index):
                        parity = compare_h5_outputs(
                            reference_path=reference_path,
                            candidate_path=candidate_h5,
                            keys=keys,
                            atol=atol,
                            rtol=rtol,
                        )
            else:
                candidate = {"exists": False, "path": None, "reason": "no_candidate_path"}

        status = _case_status(mode=mode, reference=reference, candidate=candidate, parity=parity, solve=solve_report)
        status_counts[status] += 1
        rows.append(
            {
                "case_index": index,
                "status": status,
                "selection": case,
                "reference_h5": str(reference_path),
                "candidate_h5": None if candidate_h5 is None else str(candidate_h5),
                "reference": reference,
                "candidate": candidate,
                "parity": parity,
                "solve": solve_report,
                "timing": case_timer.summary(),
            }
        )

    return {
        "schema_version": 1,
        "mode": mode,
        "selection_path": str(selection_path),
        "zenodo_root": str(zenodo_root),
        "selected_count": len(cases),
        "keys": list(keys),
        "status_counts": dict(sorted(status_counts.items())),
        "selection_summary": {
            "source_manifest_input_count": selection.get("source_manifest_input_count"),
            "source_manifest_with_fortran_output_count": selection.get("source_manifest_with_fortran_output_count"),
            "counts_by_family": selection.get("counts_by_family"),
            "counts_by_rung": selection.get("counts_by_rung"),
        },
        "solve_settings": None
        if mode != "solve"
        else {
            "run_root": None if run_root is None else str(run_root),
            "timeout_s": float(timeout_s),
            "max_cases": max_cases,
            "solve_method": str(solve_method),
            "equilibrium_search_root": None if equilibrium_search_root is None else str(equilibrium_search_root),
            "dry_run": bool(dry_run),
            "quiet_solve": bool(quiet_solve),
        },
        "cases": rows,
        "solve_trace_summary": None if mode != "solve" else _solve_trace_campaign_summary(rows),
        "timing": timer.summary(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--zenodo-root", type=Path, default=DEFAULT_ZENODO_ROOT)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("reference-only", "cached-compare", "solve"), default="reference-only")
    parser.add_argument("--key", action="append", default=None, help="Selected numeric HDF5 key to audit/compare; repeatable")
    parser.add_argument("--candidate-root", type=Path, default=None)
    parser.add_argument(
        "--candidate-template",
        default=None,
        help=(
            "Candidate path template relative to --candidate-root. Supported fields include "
            "{family}, {rung}, {surface_token}, {resolution_token}, and {input_token}."
        ),
    )
    parser.add_argument("--candidate-map", type=Path, default=None, help="JSON map from selected input/case to candidate H5 path")
    parser.add_argument("--run-root", type=Path, default=None, help="Output root for --mode solve")
    parser.add_argument("--timeout-s", type=float, default=600.0, help="Per-case subprocess timeout for --mode solve")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit selected cases, useful for bounded probes")
    parser.add_argument("--solve-method", default="auto")
    parser.add_argument(
        "--equilibrium-search-root",
        type=Path,
        default=None,
        help="Root searched by basename for VMEC/Boozer files in --mode solve; defaults to --zenodo-root.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Construct solve commands without running them")
    parser.add_argument("--quiet-solve", action="store_true", help="Use quiet sfincs_jax solve subprocesses")
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument(
        "--extra-env",
        action="append",
        default=[],
        help="Extra solve subprocess environment assignment KEY=VALUE; may be repeated.",
    )
    parser.add_argument("--atol", type=float, default=1.0e-12)
    parser.add_argument("--rtol", type=float, default=1.0e-12)
    args = parser.parse_args(argv)
    extra_env: dict[str, str] = {}
    for assignment in args.extra_env:
        if "=" not in assignment:
            parser.error(f"--extra-env expects KEY=VALUE, got {assignment!r}")
        key, value = assignment.split("=", 1)
        extra_env[key] = value

    payload = run_campaign(
        selection_path=args.selection,
        zenodo_root=args.zenodo_root,
        mode=args.mode,
        keys=tuple(args.key or DEFAULT_KEYS),
        candidate_root=args.candidate_root,
        candidate_template=args.candidate_template,
        candidate_map_path=args.candidate_map,
        run_root=args.run_root,
        timeout_s=float(args.timeout_s),
        max_cases=args.max_cases,
        solve_method=str(args.solve_method),
        equilibrium_search_root=args.equilibrium_search_root or args.zenodo_root,
        dry_run=bool(args.dry_run),
        quiet_solve=bool(args.quiet_solve),
        python_executable=str(args.python_executable),
        extra_env=extra_env,
        atol=float(args.atol),
        rtol=float(args.rtol),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(f"Wrote Zenodo VMEC parity campaign report: {args.out} ({payload['status_counts']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
