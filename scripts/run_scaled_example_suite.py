#!/usr/bin/env python
from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import asdict
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
from pathlib import Path

from audit_suite_output_keys import audit_suite_output_keys
from audit_suite_runtime_drift import audit_suite_runtime_drift
from sfincs_jax.io import localize_equilibrium_file_in_place

from run_reduced_upstream_suite import (
    CaseResult,
    REPO_ROOT,
    _executable_metadata,
    _estimate_active_size_from_namelist,
    _iter_inputs,
    _load_existing_results,
    _replace_resolution_values_in_text,
    _repo_rel,
    _resolution_from_namelist,
    _rhs_mode_from_namelist,
    _run_case,
    _sanitize_resolution,
    _status_for_mode,
    _write_rst,
)


_RUN_RECOMMENDATION_ORDER = {
    "bounded_local_ok": 0,
    "bounded_remote": 1,
    "remote_or_cluster_only": 2,
}


def _gather_jax_env() -> dict[str, object]:
    info: dict[str, object] = {
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
    }
    try:
        import jax

        info["jax_version"] = getattr(jax, "__version__", "unknown")
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [f"{d.platform}:{getattr(d, 'device_kind', d)}" for d in jax.devices()]
    except Exception as exc:  # noqa: BLE001
        info["jax_error"] = f"{type(exc).__name__}: {exc}"
    return info


def _auto_production_manifest_path(examples_root: Path, explicit_manifest: Path | None) -> Path | None:
    if explicit_manifest is not None:
        return explicit_manifest
    candidate = examples_root.parent / "manifest.json"
    if examples_root.name == "inputs" and candidate.exists():
        return candidate
    return None


def _load_production_manifest_cases(
    manifest_path: Path | None,
) -> tuple[Path | None, dict[Path, dict[str, object]], dict[str, object]]:
    if manifest_path is None:
        return None, {}, {}
    resolved_manifest = manifest_path.resolve()
    if not resolved_manifest.exists():
        raise FileNotFoundError(f"production manifest does not exist: {resolved_manifest}")
    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    manifest_root = resolved_manifest.parent
    by_input: dict[Path, dict[str, object]] = {}
    for case in payload.get("cases", []):
        if not isinstance(case, dict):
            continue
        rel_input = case.get("input")
        if not rel_input:
            continue
        by_input[(manifest_root / str(rel_input)).resolve()] = case
    return resolved_manifest, by_input, payload


def _run_recommendation_allowed(recommendation: str | None, max_recommendation: str) -> bool:
    if max_recommendation == "all":
        return True
    if recommendation is None:
        return False
    rec_rank = _RUN_RECOMMENDATION_ORDER.get(str(recommendation))
    max_rank = _RUN_RECOMMENDATION_ORDER.get(str(max_recommendation))
    if rec_rank is None or max_rank is None:
        return False
    return rec_rank <= max_rank


def _filter_inputs_by_production_recommendation(
    *,
    inputs: list[Path],
    case_names: dict[Path, str],
    manifest_cases_by_input: dict[Path, dict[str, object]],
    max_run_recommendation: str,
) -> tuple[list[Path], list[dict[str, object]]]:
    if not manifest_cases_by_input:
        return inputs, []
    kept: list[Path] = []
    skipped: list[dict[str, object]] = []
    for input_path in inputs:
        case_name = case_names[input_path]
        manifest_case = manifest_cases_by_input.get(input_path.resolve())
        size_estimate = manifest_case.get("size_estimate", {}) if isinstance(manifest_case, dict) else {}
        recommendation = (
            size_estimate.get("run_recommendation")
            if isinstance(size_estimate, dict)
            else None
        )
        if _run_recommendation_allowed(
            str(recommendation) if recommendation is not None else None,
            max_run_recommendation,
        ):
            kept.append(input_path)
            continue
        skipped.append(
            {
                "case": case_name,
                "input": str(input_path),
                "run_recommendation": recommendation,
                "max_run_recommendation": max_run_recommendation,
                "reason": "production_run_recommendation_guard",
            }
        )
    return kept, skipped


def _case_names_for_inputs(inputs: list[Path], *, base_root: Path | None = None) -> dict[Path, str]:
    parent_counts: dict[str, int] = {}
    for input_path in inputs:
        parent_counts[input_path.parent.name] = parent_counts.get(input_path.parent.name, 0) + 1

    names: dict[Path, str] = {}
    for input_path in inputs:
        parent_name = input_path.parent.name
        if parent_counts[parent_name] == 1:
            names[input_path] = parent_name
            continue
        if base_root is not None:
            try:
                rel = input_path.parent.resolve().relative_to(base_root.resolve())
                names[input_path] = "__".join(rel.parts)
                continue
            except Exception:  # noqa: BLE001
                pass
        names[input_path] = "__".join(input_path.parent.parts[-2:])
    return names


def _write_lane_summary(rows: list[CaseResult], out_path: Path) -> None:
    def _fmt(v: float | None, *, places: int = 3) -> str:
        if v is None:
            return "-"
        return f"{float(v):.{places}f}"

    def _jax_perf_runtime(row: CaseResult) -> float | None:
        return row.jax_logged_elapsed_s if row.jax_logged_elapsed_s is not None else row.jax_runtime_s

    practical_counts: dict[str, int] = {}
    strict_counts: dict[str, int] = {}
    for row in rows:
        practical_counts[row.status] = practical_counts.get(row.status, 0) + 1
        strict_status = _status_for_mode(row, strict=True)
        strict_counts[strict_status] = strict_counts.get(strict_status, 0) + 1

    offenders_runtime = sorted(
        (row for row in rows if _jax_perf_runtime(row) is not None),
        key=lambda row: float(_jax_perf_runtime(row)),
        reverse=True,
    )[:10]
    offenders_runtime_ratio = sorted(
        (
            (row, float(_jax_perf_runtime(row)) / float(row.fortran_runtime_s))
            for row in rows
            if _jax_perf_runtime(row) is not None and row.fortran_runtime_s not in (None, 0.0)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    offenders_memory = sorted(
        (row for row in rows if row.jax_max_rss_mb is not None),
        key=lambda row: float(row.jax_max_rss_mb),
        reverse=True,
    )[:10]
    offenders_memory_ratio = sorted(
        (
            (row, float(row.jax_max_rss_mb) / float(row.fortran_max_rss_mb))
            for row in rows
            if row.jax_max_rss_mb is not None and row.fortran_max_rss_mb not in (None, 0.0)
        ),
        key=lambda item: item[1],
        reverse=True,
    )[:10]
    mismatch_rows = [row for row in rows if row.n_mismatch_common > 0 or row.strict_n_mismatch_common > 0]
    reference_quality_rows = [
        row
        for row in mismatch_rows
        if str(row.blocker_type).strip().lower() == "reference solver quality"
    ]
    print_gap_rows = [row for row in rows if row.print_parity_total > 0 and row.print_parity_signals < row.print_parity_total]
    failure_rows = [row for row in rows if row.status not in {"parity_ok", "parity_mismatch"}]

    lines: list[str] = []
    lines.append("# Scaled Example Suite Summary\n\n")
    lines.append(f"- Cases: {len(rows)}\n")
    lines.append(
        "- Practical status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(practical_counts.items()))
        + "\n"
    )
    lines.append(
        "- Strict status counts: "
        + ", ".join(f"{k}={v}" for k, v in sorted(strict_counts.items()))
        + "\n\n"
    )

    lines.append("## Runtime offenders (absolute JAX time)\n\n")
    for row in offenders_runtime:
        jax_perf = _jax_perf_runtime(row)
        lines.append(
            f"- {row.case}: jax={_fmt(jax_perf)}s fortran={_fmt(row.fortran_runtime_s)}s "
            f"ratio={_fmt((float(jax_perf) / float(row.fortran_runtime_s)) if jax_perf is not None and row.fortran_runtime_s not in (None, 0.0) else None)} "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Runtime offenders (JAX/Fortran ratio)\n\n")
    for row, ratio in offenders_runtime_ratio:
        jax_perf = _jax_perf_runtime(row)
        lines.append(
            f"- {row.case}: ratio={_fmt(ratio)} jax={_fmt(jax_perf)}s fortran={_fmt(row.fortran_runtime_s)}s "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Memory offenders (absolute JAX RSS)\n\n")
    for row in offenders_memory:
        lines.append(
            f"- {row.case}: jax={_fmt(row.jax_max_rss_mb, places=1)}MB "
            f"fortran={_fmt(row.fortran_max_rss_mb, places=1)}MB "
            f"ratio={_fmt((float(row.jax_max_rss_mb) / float(row.fortran_max_rss_mb)) if row.fortran_max_rss_mb not in (None, 0.0) else None)} "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Memory offenders (JAX/Fortran ratio)\n\n")
    for row, ratio in offenders_memory_ratio:
        lines.append(
            f"- {row.case}: ratio={_fmt(ratio)} jax={_fmt(row.jax_max_rss_mb, places=1)}MB "
            f"fortran={_fmt(row.fortran_max_rss_mb, places=1)}MB "
            f"res={row.final_resolution} status={row.status}\n"
        )
    lines.append("\n## Mismatches\n\n")
    if mismatch_rows:
        for row in sorted(
            mismatch_rows,
            key=lambda item: (item.n_mismatch_common, item.strict_n_mismatch_common),
            reverse=True,
        ):
            lines.append(
                f"- {row.case}: practical={row.n_mismatch_common}/{row.n_common_keys} "
                f"strict={row.strict_n_mismatch_common}/{row.strict_n_common_keys} "
                f"solver={row.n_mismatch_solver} physics={row.n_mismatch_physics} "
                f"blocker={row.blocker_type or '-'} "
                f"sample={','.join(row.mismatch_keys_sample[:4]) or '-'}\n"
            )
    else:
        lines.append("- None\n")
    lines.append("\n## Reference-quality rows\n\n")
    if reference_quality_rows:
        for row in sorted(reference_quality_rows, key=lambda item: item.case):
            lines.append(
                f"- {row.case}: strict={row.strict_n_mismatch_common}/{row.strict_n_common_keys} "
                f"sample={','.join(row.strict_mismatch_keys_sample[:4]) or '-'} "
                f"note={row.message}\n"
            )
    else:
        lines.append("- None\n")
    lines.append("\n## Print parity gaps\n\n")
    if print_gap_rows:
        for row in sorted(print_gap_rows, key=lambda item: (item.print_parity_total - item.print_parity_signals), reverse=True):
            lines.append(
                f"- {row.case}: {row.print_parity_signals}/{row.print_parity_total} "
                f"missing={','.join(row.print_missing_signals[:6]) or '-'}\n"
            )
    else:
        lines.append("- None\n")
    lines.append("\n## Failures and blockers\n\n")
    if failure_rows:
        for row in sorted(failure_rows, key=lambda item: (item.status, item.case)):
            lines.append(
                f"- {row.case}: status={row.status} blocker={row.blocker_type} "
                f"attempts={row.attempts} reductions={row.reductions} note={row.message}\n"
            )
    else:
        lines.append("- None\n")

    out_path.write_text("".join(lines), encoding="utf-8")


def _write_suite_outputs(rows: list[CaseResult], out_root: Path) -> None:
    ordered = [rows_by_case for rows_by_case in sorted(rows, key=lambda row: row.case)]
    report_json = out_root / "suite_report.json"
    report_json.write_text(json.dumps([asdict(row) for row in ordered], indent=2), encoding="utf-8")

    report_json_strict = out_root / "suite_report_strict.json"
    strict_rows = []
    for row in ordered:
        row_dict = asdict(row)
        row_dict["status"] = _status_for_mode(row, strict=True)
        row_dict["n_common_keys"] = row.strict_n_common_keys
        row_dict["n_mismatch_common"] = row.strict_n_mismatch_common
        row_dict["mismatch_keys_sample"] = list(row.strict_mismatch_keys_sample)
        row_dict["n_mismatch_solver"] = row.strict_n_mismatch_solver
        row_dict["n_mismatch_physics"] = row.strict_n_mismatch_physics
        row_dict["mismatch_solver_sample"] = list(row.strict_mismatch_solver_sample)
        row_dict["mismatch_physics_sample"] = list(row.strict_mismatch_physics_sample)
        row_dict["max_abs_mismatch"] = row.strict_max_abs_mismatch
        row_dict["compare_mode"] = "strict"
        strict_rows.append(row_dict)
    report_json_strict.write_text(json.dumps(strict_rows, indent=2), encoding="utf-8")

    report_rst = out_root / "suite_status.rst"
    report_rst_strict = out_root / "suite_status_strict.rst"
    _write_rst(ordered, report_rst, strict=False)
    _write_rst(ordered, report_rst_strict, strict=True)
    _write_lane_summary(ordered, out_root / "summary.md")


def _resolve_repo_or_abs(path: Path | None) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else (REPO_ROOT / path)


def _write_suite_audits(
    *,
    out_root: Path,
    runtime_baseline_report: Path | None,
    runtime_drift_threshold_ratio: float,
    runtime_drift_min_baseline_runtime_s: float,
) -> dict[str, object]:
    coverage = audit_suite_output_keys(suite_root=out_root)
    (out_root / "suite_output_key_coverage.json").write_text(
        json.dumps([asdict(item) for item in coverage], indent=2),
        encoding="utf-8",
    )
    output_summary = {
        "cases": len(coverage),
        "audited_cases": sum(1 for item in coverage if not item.skipped),
        "skipped_cases": sum(1 for item in coverage if item.skipped),
        "missing_total": sum(len(item.missing_in_jax) for item in coverage),
        "extra_total": sum(len(item.extra_in_jax) for item in coverage),
        "cases_with_missing": [item.case for item in coverage if item.missing_in_jax],
        "cases_with_extra": [item.case for item in coverage if item.extra_in_jax],
        "cases_skipped": [
            {"case": item.case, "reason": item.skip_reason}
            for item in coverage
            if item.skipped
        ],
    }
    (out_root / "suite_output_key_coverage_summary.json").write_text(
        json.dumps(output_summary, indent=2),
        encoding="utf-8",
    )

    runtime_summary: dict[str, object] | None = None
    baseline_path = _resolve_repo_or_abs(runtime_baseline_report)
    if baseline_path is not None:
        if not baseline_path.exists():
            raise FileNotFoundError(f"Runtime baseline report does not exist: {baseline_path}")
        flagged = audit_suite_runtime_drift(
            baseline_report=baseline_path,
            candidate_report=out_root / "suite_report.json",
            threshold_ratio=float(runtime_drift_threshold_ratio),
            min_baseline_runtime_s=float(runtime_drift_min_baseline_runtime_s),
        )
        (out_root / "suite_runtime_drift.json").write_text(
            json.dumps([asdict(item) for item in flagged], indent=2),
            encoding="utf-8",
        )
        runtime_summary = {
            "baseline_report": _repo_rel(baseline_path),
            "candidate_report": _repo_rel(out_root / "suite_report.json"),
            "threshold_ratio": float(runtime_drift_threshold_ratio),
            "min_baseline_runtime_s": float(runtime_drift_min_baseline_runtime_s),
            "flagged_cases": len(flagged),
            "cases": [item.case for item in flagged],
        }
        (out_root / "suite_runtime_drift_summary.json").write_text(
            json.dumps(runtime_summary, indent=2),
            encoding="utf-8",
        )

    return {
        "output_key_coverage": output_summary,
        "runtime_drift": runtime_summary,
    }


def _scaled_resolution_from_reference(*, reference_input: Path, runtime_input: Path, scale_factor: float) -> dict[str, int]:
    reference_res = _resolution_from_namelist(reference_input)
    rhs_mode = _rhs_mode_from_namelist(runtime_input)
    if not reference_res:
        return {}
    if float(scale_factor) <= 0.0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor!r}")
    updates: dict[str, int] = {}
    for key, val in reference_res.items():
        if key == "NZETA" and int(val) <= 1 and float(scale_factor) > 1.0:
            updates[key] = int(val)
            continue
        scaled_float = float(val) * float(scale_factor)
        if float(scale_factor) >= 1.0:
            scaled = int(math.ceil(scaled_float))
        else:
            scaled = int(math.floor(scaled_float))
        updates[key] = scaled
    return _sanitize_resolution(updates, current=reference_res, rhs_mode=rhs_mode)


def _prepare_scaled_seed(
    *,
    source_input: Path,
    reference_input: Path,
    case_out_dir: Path,
    scale_factor: float,
) -> tuple[Path, dict[str, int], dict[str, int], dict[str, int]]:
    case_out_dir.mkdir(parents=True, exist_ok=True)
    original_copy = case_out_dir / "input.example.namelist"
    shutil.copyfile(source_input, original_copy)
    scaled_seed = case_out_dir / "input.scale_seed.namelist"
    shutil.copyfile(source_input, scaled_seed)
    source_res = _resolution_from_namelist(source_input)
    reference_res = _resolution_from_namelist(reference_input)
    scaled_res = _scaled_resolution_from_reference(
        reference_input=reference_input,
        runtime_input=source_input,
        scale_factor=float(scale_factor),
    )
    if scaled_res:
        scaled_seed.write_text(
            _replace_resolution_values_in_text(scaled_seed.read_text(), updates=scaled_res),
            encoding="utf-8",
        )
    prev_equilibria_dirs = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "")
    os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = (
        str(source_input.parent)
        if not prev_equilibria_dirs
        else os.pathsep.join((str(source_input.parent), prev_equilibria_dirs))
    )
    try:
        localize_equilibrium_file_in_place(input_namelist=scaled_seed, overwrite=True)
    finally:
        if prev_equilibria_dirs:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = prev_equilibria_dirs
        else:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)
    scaled_res = _resolution_from_namelist(scaled_seed)
    return scaled_seed, source_res, reference_res, scaled_res


def _stage_reference_fortran_artifacts(
    *,
    case_name: str,
    case_input: Path,
    case_out_dir: Path,
    reference_results_root: Path | None,
) -> tuple[bool, Path]:
    if reference_results_root is None:
        return False, case_input
    ref_case_dir = reference_results_root / case_name
    h5_candidates = (
        ref_case_dir / "last_success" / "sfincsOutput_fortran.h5",
        ref_case_dir / "fortran_run" / "sfincsOutput.h5",
    )
    log_candidates = (
        ref_case_dir / "last_success" / "sfincs_fortran.log",
        ref_case_dir / "fortran_run" / "sfincs.log",
    )
    # Prefer the input that was actually used to produce the staged H5 artifact.
    # Suite roots often keep a promoted `input.namelist` at the case root while the
    # frozen Fortran artifact under `fortran_run/` was produced at a different
    # reduced resolution. Using the root input here can stage an H5/input pair with
    # mismatched grids and generate false geometry mismatches.
    input_candidates = (
        ref_case_dir / "fortran_run" / "input.namelist",
        ref_case_dir / "input.namelist",
    )
    ref_h5 = next((path for path in h5_candidates if path.exists()), None)
    ref_log = next((path for path in log_candidates if path.exists()), None)
    ref_input = next((path for path in input_candidates if path.exists()), None)
    if ref_h5 is None:
        return False, case_input
    effective_input = case_input
    if ref_input is not None:
        case_text = case_input.read_text(encoding="utf-8")
        ref_text = ref_input.read_text(encoding="utf-8")
        if ref_text != case_text:
            ref_res = _resolution_from_namelist(ref_input)
            case_res = _resolution_from_namelist(case_input)
            if ref_res != case_res:
                # Reuse the exact reduced resolution from the frozen reference root
                # while preserving the current lane's localized equilibrium paths.
                effective_input = case_out_dir / "input.reference_seed.namelist"
                effective_input.parent.mkdir(parents=True, exist_ok=True)
                effective_input.write_text(
                    _replace_resolution_values_in_text(case_text, updates=ref_res),
                    encoding="utf-8",
                )
    staged_dir = case_out_dir / "fortran_run"
    staged_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(effective_input, staged_dir / "input.namelist")
    # Match the same localization pass that `_run_case()` applies to `dst_input` before
    # checking whether a staged Fortran artifact can be reused. Without this, analytic
    # geometry / equilibrium-path localization can make the text differ even when the
    # staged H5 already matches the intended frozen-reference resolution. Reuse the
    # current case directory as an additional equilibrium search root so extra examples
    # with locally copied VMEC files can be localized before `_run_prepared_case()`
    # installs its usual `SFINCS_JAX_EQUILIBRIA_DIRS` override.
    prev_equilibria_dirs = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "")
    os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = (
        str(case_input.parent)
        if not prev_equilibria_dirs
        else os.pathsep.join((str(case_input.parent), prev_equilibria_dirs))
    )
    try:
        localize_equilibrium_file_in_place(input_namelist=staged_dir / "input.namelist", overwrite=False)
    finally:
        if prev_equilibria_dirs:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = prev_equilibria_dirs
        else:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)
    shutil.copyfile(ref_h5, staged_dir / "sfincsOutput.h5")
    if ref_log is not None:
        shutil.copyfile(ref_log, staged_dir / "sfincs.log")
    return True, effective_input


def _load_reference_case_metrics(reference_results_root: Path | None, case_name: str) -> dict[str, object]:
    """Load measured Fortran metadata from a staged reference suite report."""
    if reference_results_root is None:
        return {}
    for report_name in ("suite_report.json", "suite_report_strict.json"):
        report_path = reference_results_root / report_name
        if not report_path.exists():
            continue
        try:
            rows = json.loads(report_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        if isinstance(rows, dict):
            rows = rows.get("cases", [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and row.get("case") == case_name:
                return row
    return {}


def _run_prepared_case(
    *,
    case_name: str,
    case_input: Path,
    reference_input: Path,
    case_out_dir: Path,
    fortran_exe: Path | None,
    timeout_s: float,
    rtol: float,
    atol: float,
    max_attempts: int,
    reuse_fortran: bool,
    collect_iterations: bool,
    jax_repeats: int,
    jax_cache_dir: Path | None,
    jax_profile_mode: str,
    equilibria_search_dir: Path | None,
    reference_results_root: Path | None,
    target_runtime_s: float | None,
    target_runtime_max_s: float | None,
    target_runtime_max_iters: int,
    target_runtime_basis: str,
) -> CaseResult:
    staged_reference, effective_case_input = _stage_reference_fortran_artifacts(
        case_name=case_name,
        case_input=case_input,
        case_out_dir=case_out_dir,
        reference_results_root=reference_results_root,
    )
    if fortran_exe is None and not staged_reference:
        raise FileNotFoundError(
            f"No staged Fortran reference available for case {case_name}, and --fortran-exe was not provided."
        )
    prev_equilibria_dirs = os.environ.get("SFINCS_JAX_EQUILIBRIA_DIRS", "")
    if equilibria_search_dir is not None:
        os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = (
            str(equilibria_search_dir)
            if not prev_equilibria_dirs
            else os.pathsep.join((str(equilibria_search_dir), prev_equilibria_dirs))
        )
    try:
        result = _run_case(
            case_name=case_name,
            case_input=effective_case_input,
            reference_input=reference_input,
            case_out_dir=case_out_dir,
            fortran_exe=fortran_exe if fortran_exe is not None else (case_out_dir / "__unused_sfincs__"),
            timeout_s=timeout_s,
            rtol=rtol,
            atol=atol,
            max_attempts=max_attempts,
            target_runtime_s=target_runtime_s,
            target_runtime_max_s=target_runtime_max_s,
            target_runtime_max_iters=target_runtime_max_iters,
            target_runtime_basis=target_runtime_basis,
            use_seed_resolution=True,
            reuse_fortran=bool(reuse_fortran or staged_reference),
            collect_iterations=collect_iterations,
            jax_repeats=jax_repeats,
            jax_cache_dir=jax_cache_dir,
            jax_profile_mode=jax_profile_mode,
        )
        if staged_reference:
            reference_metrics = _load_reference_case_metrics(reference_results_root, case_name)
            for attr in ("fortran_runtime_s", "fortran_max_rss_mb"):
                value = reference_metrics.get(attr)
                if value is not None:
                    setattr(result, attr, float(value))
        return result
    finally:
        if prev_equilibria_dirs:
            os.environ["SFINCS_JAX_EQUILIBRIA_DIRS"] = prev_equilibria_dirs
        else:
            os.environ.pop("SFINCS_JAX_EQUILIBRIA_DIRS", None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the vendored SFINCS example suite at scaled resolution.")
    parser.add_argument(
        "--examples-root",
        type=Path,
        default=Path("examples") / "sfincs_examples",
        help="Root containing upstream-style example directories.",
    )
    parser.add_argument(
        "--extra-input",
        action="append",
        default=[str(Path("examples") / "additional_examples" / "input.namelist")],
        help="Extra input.namelist to include outside --examples-root. Repeatable.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=1.0,
        help="Scale applied to upstream reference NTHETA/NZETA/NX/NXI before running.",
    )
    parser.add_argument(
        "--resolution-reference-root",
        type=Path,
        default=None,
        help=(
            "Optional root containing canonical upstream example inputs used only for "
            "NTHETA/NZETA/NX/NXI. Matching is by case directory name."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("tests") / "scaled_example_suite",
        help="Output directory for per-case runs and reports.",
    )
    parser.add_argument(
        "--fortran-exe",
        type=Path,
        default=None,
        help="Path to the Fortran SFINCS v3 executable.",
    )
    parser.add_argument(
        "--reference-results-root",
        type=Path,
        default=None,
        help=(
            "Optional existing suite root containing per-case Fortran artifacts to reuse "
            "for comparison instead of re-running Fortran on this lane."
        ),
    )
    parser.add_argument(
        "--runtime-baseline-report",
        type=Path,
        default=None,
        help=(
            "Optional baseline suite_report.json used to audit candidate JAX runtime drift "
            "for this lane."
        ),
    )
    parser.add_argument(
        "--runtime-drift-threshold-ratio",
        type=float,
        default=1.25,
        help="Flag candidate/base JAX runtime ratios above this threshold in suite_runtime_drift.json.",
    )
    parser.add_argument(
        "--runtime-drift-min-baseline-runtime-s",
        type=float,
        default=1.0,
        help="Ignore baseline cases faster than this threshold when auditing runtime drift.",
    )
    parser.add_argument(
        "--fail-on-missing-output-keys",
        action="store_true",
        help="Exit nonzero if any Fortran top-level sfincsOutput.h5 key is missing in JAX output.",
    )
    parser.add_argument(
        "--fail-on-runtime-drift",
        action="store_true",
        help="Exit nonzero if any case exceeds --runtime-drift-threshold-ratio against the baseline report.",
    )
    parser.add_argument("--timeout-s", type=float, default=900.0, help="Per-attempt timeout in seconds.")
    parser.add_argument(
        "--fortran-min-runtime-s",
        type=float,
        default=None,
        help=(
            "Minimum per-case benchmark runtime target. If the selected runtime basis falls below "
            "this after downscaling, the runner will try to scale back up, capped at the original "
            "reference resolution. Defaults to the production manifest target when a generated "
            "production input tree is used, otherwise 1.0 s."
        ),
    )
    parser.add_argument(
        "--fortran-max-runtime-s",
        type=float,
        default=None,
        help=(
            "Optional per-case runtime cap for the selected benchmark basis. Use this to downscale "
            "only the largest original-resolution cases while keeping smaller cases near their "
            "original v3 resolution."
        ),
    )
    parser.add_argument(
        "--runtime-adjustment-iters",
        type=int,
        default=3,
        help="Maximum per-case runtime-guided resolution adjustments.",
    )
    parser.add_argument(
        "--runtime-target-basis",
        choices=("fortran", "max"),
        default="fortran",
        help=(
            "Runtime metric used to adjust resolution. 'fortran' keeps example benchmarking tied "
            "to the original v3 runtime window; 'max' uses max(Fortran, JAX)."
        ),
    )
    parser.add_argument("--rtol", type=float, default=5e-4, help="Relative tolerance for H5 comparison.")
    parser.add_argument("--atol", type=float, default=1e-9, help="Absolute tolerance for H5 comparison.")
    parser.add_argument("--max-attempts", type=int, default=3, help="Maximum attempts per case.")
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="Regex filter applied to case name and source path.",
    )
    parser.add_argument(
        "--reuse-fortran",
        action="store_true",
        help="Reuse a matching per-case Fortran result when present.",
    )
    parser.add_argument(
        "--reset-report",
        action="store_true",
        help="Overwrite suite_report.json instead of merging with prior results.",
    )
    parser.add_argument(
        "--jax-cache-dir",
        type=Path,
        default=None,
        help="Optional persistent JAX compilation cache directory.",
    )
    parser.add_argument(
        "--jax-repeats",
        type=int,
        default=1,
        help="Number of sfincs_jax repeats per case.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of cases to run in parallel.",
    )
    parser.add_argument("--case-index", type=int, default=None, help="0-based job-array index.")
    parser.add_argument("--case-stride", type=int, default=1, help="Job-array stride.")
    parser.add_argument(
        "--no-collect-iterations",
        action="store_true",
        help="Disable solver-iteration parsing from sfincs_jax logs.",
    )
    parser.add_argument(
        "--jax-profile-marks",
        choices=("off", "on", "full"),
        default="off",
        help=(
            "sfincs_jax profiling mode for suite subprocesses. "
            "Use 'off' for runtime audits and opt into 'on'/'full' only for targeted profiling lanes."
        ),
    )
    parser.add_argument(
        "--production-manifest",
        type=Path,
        default=None,
        help=(
            "Optional manifest.json generated by create_production_benchmark_inputs.py. "
            "If omitted and --examples-root points at a generated inputs/ directory, the sibling "
            "manifest.json is used automatically."
        ),
    )
    parser.add_argument(
        "--max-run-recommendation",
        choices=("bounded_local_ok", "bounded_remote", "remote_or_cluster_only", "all"),
        default="bounded_local_ok",
        help=(
            "Largest production size recommendation to launch when a production manifest is active. "
            "Default keeps local runs bounded; use 'bounded_remote', 'remote_or_cluster_only', or "
            "'all' only on explicitly budgeted remote/cluster lanes."
        ),
    )
    args = parser.parse_args()

    examples_root = Path(args.examples_root)
    reference_root = Path(args.resolution_reference_root) if args.resolution_reference_root is not None else None
    out_root = Path(args.out_root)
    fortran_exe = Path(args.fortran_exe) if args.fortran_exe is not None else None
    reference_results_root = Path(args.reference_results_root) if args.reference_results_root is not None else None
    if not examples_root.exists():
        raise SystemExit(f"examples root does not exist: {examples_root}")
    if reference_root is not None and not reference_root.exists():
        raise SystemExit(f"resolution reference root does not exist: {reference_root}")
    if reference_results_root is not None and not reference_results_root.exists():
        raise SystemExit(f"reference results root does not exist: {reference_results_root}")
    if fortran_exe is None and reference_results_root is None:
        raise SystemExit("Either --fortran-exe or --reference-results-root must be provided.")
    if fortran_exe is not None and not fortran_exe.exists():
        raise SystemExit(f"Fortran executable does not exist: {fortran_exe}")
    out_root.mkdir(parents=True, exist_ok=True)
    production_manifest_path = _auto_production_manifest_path(examples_root, args.production_manifest)
    (
        production_manifest_path,
        production_manifest_cases_by_input,
        production_manifest_payload,
    ) = _load_production_manifest_cases(production_manifest_path)
    if args.fortran_min_runtime_s is None:
        manifest_runtime_target = production_manifest_payload.get("target_fortran_min_runtime_s")
        args.fortran_min_runtime_s = (
            float(manifest_runtime_target) if manifest_runtime_target is not None else 1.0
        )

    inputs = _iter_inputs(examples_root)
    extra_inputs: list[Path] = []
    for raw in args.extra_input:
        path = Path(raw)
        if path.exists():
            extra_inputs.append(path)
    deduped_inputs: list[Path] = []
    seen_inputs: set[Path] = set()
    for input_path in [*inputs, *extra_inputs]:
        resolved = input_path.resolve()
        if resolved in seen_inputs:
            continue
        seen_inputs.add(resolved)
        deduped_inputs.append(input_path)
    inputs = deduped_inputs
    case_names = _case_names_for_inputs(inputs, base_root=examples_root.parent)

    if args.pattern:
        rx = re.compile(str(args.pattern), flags=re.IGNORECASE)
        inputs = [
            path
            for path in inputs
            if rx.search(case_names[path]) or rx.search(str(path.parent)) or rx.search(str(path))
        ]
    if not inputs:
        raise SystemExit("No input.namelist files matched.")

    inputs, skipped_by_recommendation = _filter_inputs_by_production_recommendation(
        inputs=inputs,
        case_names=case_names,
        manifest_cases_by_input=production_manifest_cases_by_input,
        max_run_recommendation=str(args.max_run_recommendation),
    )
    if skipped_by_recommendation:
        print(
            "Production manifest guard: "
            f"skipping {len(skipped_by_recommendation)} cases above "
            f"--max-run-recommendation={args.max_run_recommendation}"
        )
        for item in skipped_by_recommendation:
            print(
                f"  skip {item['case']}: "
                f"recommendation={item['run_recommendation']} "
                f"limit={item['max_run_recommendation']}"
            )
    if not inputs:
        raise SystemExit("No input.namelist files remained after production recommendation filtering.")

    stride_val = max(1, int(args.case_stride))
    if args.case_index is not None:
        idx = int(args.case_index)
        if idx < 0 or idx >= stride_val:
            raise SystemExit(f"--case-index={idx} out of range for --case-stride={stride_val}")
        inputs = [path for i, path in enumerate(inputs) if i % stride_val == idx]
        if not inputs:
            raise SystemExit("No inputs matched after case-index filtering.")

    report_json = out_root / "suite_report.json"
    merged_results: dict[str, CaseResult] = {} if args.reset_report else _load_existing_results(report_json)
    current_run_results: list[CaseResult] = []

    manifest_cases: list[dict[str, object]] = []
    prepared: list[tuple[int, str, Path, Path, Path, dict[str, int], dict[str, int], dict[str, int]]] = []
    for input_path in inputs:
        case = case_names[input_path]
        case_out = out_root / case
        reference_input = input_path
        if reference_root is not None:
            candidate = reference_root / case / "input.namelist"
            if candidate.exists():
                reference_input = candidate
        seed_input, source_res, reference_res, scaled_res = _prepare_scaled_seed(
            source_input=input_path,
            reference_input=reference_input,
            case_out_dir=case_out,
            scale_factor=float(args.scale_factor),
        )
        est_size = int(_estimate_active_size_from_namelist(seed_input) or 0)
        manifest_cases.append(
            {
                "case": case,
                "source_input": _repo_rel(input_path),
                "source_input_abs": str(input_path.resolve()),
                "reference_input": _repo_rel(reference_input),
                "reference_input_abs": str(reference_input.resolve()),
                "source_resolution": source_res,
                "reference_resolution": reference_res,
                "scaled_seed_resolution": scaled_res,
                "estimated_active_size": est_size,
                "scale_factor": float(args.scale_factor),
            }
        )
        prepared.append((est_size, case, input_path, reference_input, seed_input, source_res, reference_res, scaled_res))

    prepared.sort(key=lambda item: (item[0], item[1]))

    manifest = {
        "resolution_policy": (
            "reference_first_runtime_window"
            if args.fortran_min_runtime_s is not None or args.fortran_max_runtime_s is not None
            else "reference_scale_only"
        ),
        "scale_factor": float(args.scale_factor),
        "timeout_s": float(args.timeout_s),
        "fortran_min_runtime_s": float(args.fortran_min_runtime_s) if args.fortran_min_runtime_s is not None else None,
        "fortran_max_runtime_s": float(args.fortran_max_runtime_s) if args.fortran_max_runtime_s is not None else None,
        "runtime_adjustment_iters": int(args.runtime_adjustment_iters),
        "runtime_target_basis": str(args.runtime_target_basis),
        "rtol": float(args.rtol),
        "atol": float(args.atol),
        "max_attempts": int(args.max_attempts),
        "jax_repeats": int(args.jax_repeats),
        "jax_profile_marks": str(args.jax_profile_marks),
        "jobs": int(args.jobs),
        "resolution_reference_root": _repo_rel(reference_root) if reference_root is not None else None,
        "reference_results_root": _repo_rel(reference_results_root) if reference_results_root is not None else None,
        "runtime_baseline_report": _repo_rel(_resolve_repo_or_abs(args.runtime_baseline_report))
        if args.runtime_baseline_report is not None
        else None,
        "runtime_drift_threshold_ratio": float(args.runtime_drift_threshold_ratio),
        "runtime_drift_min_baseline_runtime_s": float(args.runtime_drift_min_baseline_runtime_s),
        "production_manifest": _repo_rel(production_manifest_path) if production_manifest_path is not None else None,
        "max_run_recommendation": str(args.max_run_recommendation),
        "skipped_by_recommendation": skipped_by_recommendation,
        "fortran_exe": _repo_rel(fortran_exe) if fortran_exe is not None else None,
        "fortran_executable": _executable_metadata(fortran_exe),
        "environment": _gather_jax_env(),
        "cases": manifest_cases,
    }
    (out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _handle_result(result: CaseResult) -> None:
        prev = merged_results.get(result.case)
        if prev is not None:
            for attr in (
                "fortran_runtime_s",
                "jax_runtime_s",
                "jax_runtime_s_cold",
                "jax_runtime_s_warm",
                "fortran_max_rss_mb",
                "jax_max_rss_mb",
            ):
                if getattr(result, attr) is None and getattr(prev, attr) is not None:
                    setattr(result, attr, getattr(prev, attr))
        current_run_results.append(result)
        merged_results[result.case] = result
        _write_suite_outputs(list(merged_results.values()), out_root)
        _write_suite_audits(
            out_root=out_root,
            runtime_baseline_report=args.runtime_baseline_report,
            runtime_drift_threshold_ratio=float(args.runtime_drift_threshold_ratio),
            runtime_drift_min_baseline_runtime_s=float(args.runtime_drift_min_baseline_runtime_s),
        )
        print(
            f"  status={result.status} attempts={result.attempts} reductions={result.reductions} "
            f"res={result.final_resolution} mismatch={result.n_mismatch_common}/{result.n_common_keys} "
            f"strict={result.strict_n_mismatch_common}/{result.strict_n_common_keys} "
            f"printParity={result.print_parity_signals}/{result.print_parity_total} blocker={result.blocker_type}"
        )

    jobs = max(1, int(args.jobs))
    if jobs <= 1:
        for index, (est_size, case, input_path, reference_input, seed_input, _source_res, _reference_res, scaled_res) in enumerate(prepared, start=1):
            print(f"[{index}/{len(prepared)}] {case}")
            print(f"  source={input_path}")
            print(f"  reference={reference_input}")
            print(f"  scaled_seed={scaled_res} est_size={est_size}")
            case_out = out_root / case
            result = _run_prepared_case(
                case_name=case,
                case_input=seed_input,
                reference_input=reference_input,
                case_out_dir=case_out,
                fortran_exe=fortran_exe,
                timeout_s=float(args.timeout_s),
                rtol=float(args.rtol),
                atol=float(args.atol),
                max_attempts=int(args.max_attempts),
                target_runtime_s=float(args.fortran_min_runtime_s) if args.fortran_min_runtime_s is not None else None,
                target_runtime_max_s=(
                    float(args.fortran_max_runtime_s) if args.fortran_max_runtime_s is not None else None
                ),
                target_runtime_max_iters=int(args.runtime_adjustment_iters),
                target_runtime_basis=str(args.runtime_target_basis),
                reuse_fortran=bool(args.reuse_fortran),
                collect_iterations=not bool(args.no_collect_iterations),
                jax_repeats=int(args.jax_repeats),
                jax_cache_dir=(REPO_ROOT / args.jax_cache_dir) if args.jax_cache_dir is not None else None,
                jax_profile_mode=str(args.jax_profile_marks),
                equilibria_search_dir=seed_input.parent,
                reference_results_root=reference_results_root,
            )
            _handle_result(result)
    else:
        print(f"Running {len(prepared)} cases with --jobs={jobs}")
        futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            for index, (est_size, case, input_path, reference_input, seed_input, _source_res, _reference_res, scaled_res) in enumerate(prepared, start=1):
                print(f"[{index}/{len(prepared)}] {case}")
                print(f"  source={input_path}")
                print(f"  reference={reference_input}")
                print(f"  scaled_seed={scaled_res} est_size={est_size}")
                case_out = out_root / case
                futures.append(
                    pool.submit(
                        _run_prepared_case,
                        case_name=case,
                        case_input=seed_input,
                        reference_input=reference_input,
                        case_out_dir=case_out,
                        fortran_exe=fortran_exe,
                        timeout_s=float(args.timeout_s),
                        rtol=float(args.rtol),
                        atol=float(args.atol),
                        max_attempts=int(args.max_attempts),
                        target_runtime_s=(
                            float(args.fortran_min_runtime_s) if args.fortran_min_runtime_s is not None else None
                        ),
                        target_runtime_max_s=(
                            float(args.fortran_max_runtime_s) if args.fortran_max_runtime_s is not None else None
                        ),
                        target_runtime_max_iters=int(args.runtime_adjustment_iters),
                        target_runtime_basis=str(args.runtime_target_basis),
                        reuse_fortran=bool(args.reuse_fortran),
                        collect_iterations=not bool(args.no_collect_iterations),
                        jax_repeats=int(args.jax_repeats),
                        jax_cache_dir=(REPO_ROOT / args.jax_cache_dir) if args.jax_cache_dir is not None else None,
                        jax_profile_mode=str(args.jax_profile_marks),
                        equilibria_search_dir=seed_input.parent,
                        reference_results_root=reference_results_root,
                    )
                )
            for fut in concurrent.futures.as_completed(futures):
                result = fut.result()
                _handle_result(result)

    ordered = [merged_results[key] for key in sorted(merged_results)]
    _write_suite_outputs(ordered, out_root)
    audit_summary = _write_suite_audits(
        out_root=out_root,
        runtime_baseline_report=args.runtime_baseline_report,
        runtime_drift_threshold_ratio=float(args.runtime_drift_threshold_ratio),
        runtime_drift_min_baseline_runtime_s=float(args.runtime_drift_min_baseline_runtime_s),
    )

    print(f"Wrote {report_json}")
    print(f"Wrote {out_root / 'suite_report_strict.json'}")
    print(f"Wrote {out_root / 'suite_status.rst'}")
    print(f"Wrote {out_root / 'suite_status_strict.rst'}")
    print(f"Wrote {out_root / 'summary.md'}")
    print(f"Wrote {out_root / 'suite_output_key_coverage.json'}")
    print(f"Wrote {out_root / 'suite_output_key_coverage_summary.json'}")
    output_summary = audit_summary["output_key_coverage"]
    print(
        "Output-key coverage: "
        f"missing_total={output_summary['missing_total']} "
        f"extra_total={output_summary['extra_total']} "
        f"audited_cases={output_summary['audited_cases']} "
        f"skipped_cases={output_summary['skipped_cases']}"
    )
    runtime_summary = audit_summary["runtime_drift"]
    if runtime_summary is not None:
        print(f"Wrote {out_root / 'suite_runtime_drift.json'}")
        print(f"Wrote {out_root / 'suite_runtime_drift_summary.json'}")
        print(
            "Runtime drift audit: "
            f"flagged_cases={runtime_summary['flagged_cases']} "
            f"threshold_ratio={runtime_summary['threshold_ratio']} "
            f"min_baseline_runtime_s={runtime_summary['min_baseline_runtime_s']}"
        )

    exit_code = 0
    if bool(args.fail_on_missing_output_keys) and int(output_summary["missing_total"]) > 0:
        exit_code = 1
    if (
        bool(args.fail_on_runtime_drift)
        and runtime_summary is not None
        and int(runtime_summary["flagged_cases"]) > 0
    ):
        exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
