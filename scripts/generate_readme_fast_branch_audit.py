#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
README = REPO_ROOT / "README.md"
DEFAULT_OUT_ROOT = REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak"
DEFAULT_GPU_OUT_ROOT = REPO_ROOT / "tests" / "scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas"
BASELINE_REPORT = REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak" / "suite_report.json"
EXAMPLES_ROOT = REPO_ROOT / "examples" / "sfincs_examples"
EXTRA_INPUT = REPO_ROOT / "examples" / "data" / "qi_nfp2_reference.input.namelist"
EXTRA_CASE_NAME = "additional_examples"
DEFAULT_PUBLIC_MIN_FORTRAN_RUNTIME_S = 10.0

BEGIN = "<!-- BEGIN EXAMPLE_SUITE_AUDIT -->"
END = "<!-- END EXAMPLE_SUITE_AUDIT -->"
LEGACY_BEGIN = "<!-- BEGIN FAST_BRANCH_AUDIT -->"
LEGACY_END = "<!-- END FAST_BRANCH_AUDIT -->"


def _load_json(path: Path) -> object:
    return json.loads(path.read_text())


def _load_optional_json(path: Path) -> object | None:
    if not path.exists():
        return None
    return _load_json(path)


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:  # noqa: BLE001
        return str(path)


def _case_names_for_inputs(inputs: list[Path], *, base_root: Path | None = None) -> dict[Path, str]:
    parent_counts: dict[str, int] = {}
    for input_path in inputs:
        parent_counts[input_path.parent.name] = parent_counts.get(input_path.parent.name, 0) + 1

    names: dict[Path, str] = {}
    for input_path in inputs:
        try:
            if input_path.resolve() == EXTRA_INPUT.resolve():
                names[input_path] = EXTRA_CASE_NAME
                continue
        except Exception:  # noqa: BLE001
            pass
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


def _expected_cases() -> list[str]:
    inputs = sorted(EXAMPLES_ROOT.rglob("input.namelist"))
    if EXTRA_INPUT.exists():
        inputs.append(EXTRA_INPUT)
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    case_names = _case_names_for_inputs(deduped, base_root=EXAMPLES_ROOT.parent)
    return sorted(case_names.values())


def _fmt_float(value: object | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_runtime_ratio(row: dict[str, object]) -> str:
    jax = row.get("jax_runtime_s")
    fort = row.get("fortran_runtime_s")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _warm_or_logged_runtime(row: dict[str, object] | None) -> object | None:
    if row is None:
        return None
    warm = row.get("jax_runtime_s_warm")
    if warm is not None:
        return warm
    return row.get("jax_logged_elapsed_s")


def _fmt_memory_ratio(row: dict[str, object]) -> str:
    jax = _jax_public_memory(row)
    fort = row.get("fortran_max_rss_mb")
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _jax_public_memory(row: dict[str, object] | None) -> object | None:
    if row is None:
        return None
    active = row.get("jax_incremental_max_rss_mb")
    if active is not None:
        return active
    return row.get("jax_max_rss_mb")


def _fmt_ratio(jax: object | None, fort: object | None) -> str:
    if jax is None or fort in (None, 0):
        return "-"
    return f"{float(jax) / float(fort):.2f}x"


def _fmt_status(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    status = str(row.get("status", "")).strip()
    return status or "-"


def _fmt_mismatch_pair(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    return (
        f"{int(row.get('n_mismatch_common', 0))}/{int(row.get('n_common_keys', 0))}"
        f" (strict {int(row.get('strict_n_mismatch_common', 0))}/{int(row.get('strict_n_common_keys', 0))})"
    )


def _fmt_print_parity(row: dict[str, object] | None) -> str:
    if row is None:
        return "-"
    total = int(row.get("print_parity_total", 0))
    if total <= 0:
        return "-"
    return f"{int(row.get('print_parity_signals', 0))}/{total}"


def _reference_fortran_runtime(
    case: str,
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
) -> float | None:
    row = rows_by_case.get(case) or gpu_rows_by_case.get(case)
    if row is None:
        return None
    value = row.get("fortran_runtime_s")
    if value is None:
        return None
    return float(value)


def _public_comparison_cases(
    case_order: list[str],
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
    *,
    min_fortran_runtime_s: float,
) -> tuple[list[str], list[dict[str, object]]]:
    included: list[str] = []
    excluded: list[dict[str, object]] = []
    threshold = float(min_fortran_runtime_s)
    for case in case_order:
        runtime = _reference_fortran_runtime(case, rows_by_case, gpu_rows_by_case)
        if runtime is not None and runtime >= threshold:
            included.append(case)
        else:
            excluded.append({"case": case, "fortran_runtime_s": runtime})
    return included, excluded


def _format_excluded_public_cases(excluded_cases: list[dict[str, object]]) -> str:
    if not excluded_cases:
        return "none"
    return ", ".join(
        f"`{row['case']}` ({_fmt_float(row.get('fortran_runtime_s'), 3)}s)" for row in excluded_cases
    )


def _status_counts(rows: list[dict[str, object]], prefix: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get(prefix, row.get("status", "unknown")))] += 1
    return counts


def _strict_status_counts(rows: list[dict[str, object]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for row in rows:
        strict_mismatch = int(row.get("strict_n_mismatch_common", 0))
        strict_common = int(row.get("strict_n_common_keys", 0))
        strict_status = "parity_ok"
        if strict_common > 0 and strict_mismatch > 0:
            strict_status = "parity_mismatch"
        elif row.get("status") not in {"parity_ok", "parity_mismatch"}:
            strict_status = str(row.get("status"))
        counts[str(strict_status)] += 1
    return counts


def _top_rows(
    rows: list[dict[str, object]],
    *,
    key: str,
    limit: int = 5,
) -> list[dict[str, object]]:
    valid = [row for row in rows if row.get(key) is not None]
    return sorted(valid, key=lambda row: float(row[key]), reverse=True)[:limit]


def _format_row_summary(row: dict[str, object], *, metric_key: str, digits: int = 1) -> str:
    value = _fmt_float(row.get(metric_key), digits)
    fort_key = "fortran_runtime_s" if metric_key == "jax_runtime_s" else "fortran_max_rss_mb"
    ratio = _fmt_runtime_ratio(row) if metric_key == "jax_runtime_s" else _fmt_memory_ratio(row)
    final_resolution = row.get("final_resolution")
    res_str = f", res={final_resolution}" if final_resolution else ""
    return (
        f"- `{row['case']}`: jax={value}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"fortran={_fmt_float(row.get(fort_key), digits)}"
        f"{'s' if metric_key == 'jax_runtime_s' else ' MB'} "
        f"ratio={ratio} status={row.get('status', '-')}{res_str}"
    )


def _format_mismatch(row: dict[str, object]) -> str:
    return (
        f"- `{row['case']}`: status={row.get('status', '-')}, "
        f"practical={row.get('n_mismatch_common', 0)}/{row.get('n_common_keys', 0)}, "
        f"strict={row.get('strict_n_mismatch_common', 0)}/{row.get('strict_n_common_keys', 0)}, "
        f"sample={','.join(row.get('mismatch_keys_sample', [])[:4]) or '-'}"
    )


def _format_case_table(
    case_order: list[str],
    rows_by_case: dict[str, dict[str, object]],
    gpu_rows_by_case: dict[str, dict[str, object]],
) -> list[str]:
    lines = [
        "| Case | Fortran CPU(s) | JAX CPU cold(s) | CPU cold x | JAX CPU warm/logged(s) | CPU warm/logged x | JAX GPU cold(s) | GPU cold x | JAX GPU warm/logged(s) | GPU warm/logged x | Fortran MB | JAX CPU active MB | CPU MB x | JAX GPU active MB | GPU MB x | CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for case in case_order:
        cpu_row = rows_by_case.get(case)
        gpu_row = gpu_rows_by_case.get(case)
        if cpu_row is None and gpu_row is None:
            continue
        reference_row = cpu_row or gpu_row
        fort_runtime = reference_row.get("fortran_runtime_s") if reference_row else None
        fort_memory = reference_row.get("fortran_max_rss_mb") if reference_row else None
        cpu_runtime = cpu_row.get("jax_runtime_s") if cpu_row else None
        gpu_runtime = gpu_row.get("jax_runtime_s") if gpu_row else None
        cpu_warm_runtime = _warm_or_logged_runtime(cpu_row)
        gpu_warm_runtime = _warm_or_logged_runtime(gpu_row)
        cpu_memory = _jax_public_memory(cpu_row)
        gpu_memory = _jax_public_memory(gpu_row)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{case}`",
                    _fmt_float(fort_runtime, 3),
                    _fmt_float(cpu_runtime, 3),
                    _fmt_ratio(cpu_runtime, fort_runtime),
                    _fmt_float(cpu_warm_runtime, 3),
                    _fmt_ratio(cpu_warm_runtime, fort_runtime),
                    _fmt_float(gpu_runtime, 3),
                    _fmt_ratio(gpu_runtime, fort_runtime),
                    _fmt_float(gpu_warm_runtime, 3),
                    _fmt_ratio(gpu_warm_runtime, fort_runtime),
                    _fmt_float(fort_memory, 1),
                    _fmt_float(cpu_memory, 1),
                    _fmt_ratio(cpu_memory, fort_memory),
                    _fmt_float(gpu_memory, 1),
                    _fmt_ratio(gpu_memory, fort_memory),
                    _fmt_mismatch_pair(cpu_row),
                    _fmt_mismatch_pair(gpu_row),
                    _fmt_print_parity(cpu_row),
                    _fmt_print_parity(gpu_row),
                    _fmt_status(cpu_row),
                    _fmt_status(gpu_row),
                ]
            )
            + " |"
        )
    return lines


def _format_improvement(
    current_row: dict[str, object],
    baseline_row: dict[str, object],
    *,
    metric_key: str,
    digits: int = 1,
) -> str:
    if metric_key == "jax_incremental_max_rss_mb":
        current_value = _jax_public_memory(current_row)
        baseline_value = _jax_public_memory(baseline_row)
    else:
        current_value = current_row[metric_key]
        baseline_value = baseline_row[metric_key]
    current = float(current_value)
    baseline = float(baseline_value)
    delta = baseline - current
    unit = "s" if metric_key == "jax_runtime_s" else " MB"
    return (
        f"- `{current_row['case']}`: "
        f"{_fmt_float(baseline, digits)}{unit} -> {_fmt_float(current, digits)}{unit} "
        f"(delta={_fmt_float(delta, digits)}{unit})"
    )


def _format_runtime_drift_summary(prefix: str, summary: dict[str, object]) -> str:
    """Format only same-resolution runtime drift as a gate."""
    status = str(summary.get("status", "")).strip().lower()
    if status in {"not_applicable", "skipped"}:
        reason = str(summary.get("reason", "")).strip()
        for stale_reason in (
            "production-floor reruns are not same-resolution with the older frozen smoke baseline",
            "production-floor reruns are not same-resolution with the frozen smoke baseline",
        ):
            reason = reason.replace(
                stale_reason,
                "suite rows are not same-resolution with the optional runtime baseline",
            )
        suffix = f": {reason}" if reason else ""
        return f"- {prefix} runtime drift gate: not applicable{suffix}"

    flagged = int(summary.get("flagged_cases", 0))
    threshold = summary.get("threshold_ratio", "-")
    baseline = summary.get("baseline_report", "-")
    cases = [str(case) for case in summary.get("cases", [])]
    if flagged > 0:
        return (
            f"- {prefix} runtime drift gate vs `{baseline}`: "
            f"`{flagged}` cases above `{threshold}x` "
            f"({', '.join(cases[:4])})"
        )
    return f"- {prefix} runtime drift gate vs `{baseline}`: none"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update README full example-suite audit block.")
    parser.add_argument(
        "--out-root",
        type=Path,
        default=DEFAULT_OUT_ROOT,
        help="Suite output root containing suite_report.json and run_manifest.json.",
    )
    parser.add_argument(
        "--baseline-report",
        type=Path,
        default=BASELINE_REPORT,
        help="Optional baseline report used for improvement summaries.",
    )
    parser.add_argument(
        "--gpu-out-root",
        type=Path,
        default=DEFAULT_GPU_OUT_ROOT,
        help="Optional GPU suite output root containing suite_report.json.",
    )
    parser.add_argument(
        "--min-fortran-runtime-s",
        type=float,
        default=DEFAULT_PUBLIC_MIN_FORTRAN_RUNTIME_S,
        help="Minimum Fortran v3 runtime for public runtime/memory comparison rows.",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    report_path = out_root / "suite_report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing report: {report_path}")
    manifest_path = out_root / "run_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    gpu_out_root = Path(args.gpu_out_root)
    gpu_report_path = gpu_out_root / "suite_report.json"
    gpu_rows = list(_load_json(gpu_report_path)) if gpu_report_path.exists() else []
    cpu_key_summary = _load_optional_json(out_root / "suite_output_key_coverage_summary.json")
    gpu_key_summary = _load_optional_json(gpu_out_root / "suite_output_key_coverage_summary.json")
    cpu_runtime_drift_summary = _load_optional_json(out_root / "suite_runtime_drift_summary.json")
    gpu_runtime_drift_summary = _load_optional_json(gpu_out_root / "suite_runtime_drift_summary.json")

    rows = list(_load_json(report_path))
    case_order = _expected_cases()
    total_cases = len(case_order)
    rows_by_case = {str(row["case"]): row for row in rows}
    missing_cases = [case for case in case_order if case not in rows_by_case]

    strict_counts = _strict_status_counts(rows)
    gpu_status_counts = Counter(str(row.get("status", "unknown")) for row in gpu_rows)
    gpu_strict_counts = _strict_status_counts(gpu_rows)

    status_counts = Counter(str(row.get("status", "unknown")) for row in rows)
    mismatches = [row for row in rows if str(row.get("status")) != "parity_ok"]
    gpu_mismatches = [row for row in gpu_rows if str(row.get("status")) != "parity_ok"]

    improvements_runtime: list[str] = []
    improvements_memory: list[str] = []
    baseline_report = Path(args.baseline_report)
    if baseline_report.exists():
        baseline_rows = {str(row["case"]): row for row in _load_json(baseline_report)}
        paired_runtime = []
        paired_memory = []
        for case, row in rows_by_case.items():
            reference_runtime = _reference_fortran_runtime(case, rows_by_case, {})
            if reference_runtime is None or reference_runtime < float(args.min_fortran_runtime_s):
                continue
            base = baseline_rows.get(case)
            if not base:
                continue
            if row.get("jax_runtime_s") is not None and base.get("jax_runtime_s") is not None:
                paired_runtime.append((float(base["jax_runtime_s"]) - float(row["jax_runtime_s"]), row, base))
            if row.get("jax_max_rss_mb") is not None and base.get("jax_max_rss_mb") is not None:
                paired_memory.append((float(base["jax_max_rss_mb"]) - float(row["jax_max_rss_mb"]), row, base))
        for _delta, row, base in sorted(paired_runtime, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_runtime.append(_format_improvement(row, base, metric_key="jax_runtime_s", digits=1))
        for _delta, row, base in sorted(paired_memory, key=lambda item: item[0], reverse=True)[:5]:
            if _delta > 0:
                improvements_memory.append(_format_improvement(row, base, metric_key="jax_max_rss_mb", digits=1))

    lines = [
        BEGIN,
        f"CPU audit source: `{_repo_rel(out_root)}`.",
        (
            f"GPU audit source: `{_repo_rel(gpu_out_root)}`."
            if gpu_rows
            else "GPU audit source: not available."
        ),
        "",
        f"- Recorded cases: `{len(rows)}/{total_cases}`",
        f"- Practical status counts: `{', '.join(f'{k}={status_counts[k]}' for k in sorted(status_counts))}`",
        f"- Strict status counts: `{', '.join(f'{k}={strict_counts[k]}' for k in sorted(strict_counts))}`",
    ]
    if gpu_rows:
        lines.append(
            f"- GPU practical status counts: `{', '.join(f'{k}={gpu_status_counts[k]}' for k in sorted(gpu_status_counts))}`"
        )
        lines.append(
            f"- GPU strict status counts: `{', '.join(f'{k}={gpu_strict_counts[k]}' for k in sorted(gpu_strict_counts))}`"
        )
    if cpu_key_summary is not None:
        lines.append(
            "- CPU output-key coverage: "
            f"`missing_total={cpu_key_summary.get('missing_total') or 0}, "
            f"extra_total={cpu_key_summary.get('extra_total') or '-'}, "
            f"audited_cases={cpu_key_summary.get('audited_cases') or '-'}, "
            f"skipped_cases={cpu_key_summary.get('skipped_cases') or 0}`"
        )
    if gpu_key_summary is not None:
        lines.append(
            "- GPU output-key coverage: "
            f"`missing_total={gpu_key_summary.get('missing_total') or 0}, "
            f"extra_total={gpu_key_summary.get('extra_total') or '-'}, "
            f"audited_cases={gpu_key_summary.get('audited_cases') or '-'}, "
            f"skipped_cases={gpu_key_summary.get('skipped_cases') or 0}`"
        )
    if cpu_runtime_drift_summary is not None:
        lines.append(_format_runtime_drift_summary("CPU", cpu_runtime_drift_summary))
    if gpu_runtime_drift_summary is not None:
        lines.append(_format_runtime_drift_summary("GPU", gpu_runtime_drift_summary))
    if manifest:
        resolution_policy = manifest.get("resolution_policy")
        scale_factor = manifest.get("scale_factor")
        runtime_basis = manifest.get("runtime_target_basis")
        runtime_floor = manifest.get("fortran_min_runtime_s")
        runtime_cap = manifest.get("fortran_max_runtime_s")
        adjust_iters = manifest.get("runtime_adjustment_iters")
        lines.append(
            "- Resolution policy: "
            f"`{resolution_policy}, scale_factor={scale_factor}, runtime_basis={runtime_basis}, "
            f"fortran_min={runtime_floor}, fortran_max={runtime_cap}, adjust_iters={adjust_iters}`"
        )
    if missing_cases:
        lines.append(f"- Remaining cases: `{', '.join(missing_cases)}`")
    else:
        lines.append("- Remaining cases: none")
    cpu_additional = rows_by_case.get("additional_examples")
    gpu_rows_by_case = {str(row["case"]): row for row in gpu_rows}
    public_case_order, excluded_public_cases = _public_comparison_cases(
        case_order,
        rows_by_case,
        gpu_rows_by_case,
        min_fortran_runtime_s=float(args.min_fortran_runtime_s),
    )
    gpu_additional = gpu_rows_by_case.get("additional_examples")
    if cpu_additional is not None and gpu_additional is not None:
        lines.append(
            f"- Additional example: `{cpu_additional.get('status', '-')}` on CPU and `{gpu_additional.get('status', '-')}` on GPU"
        )

    gpu_rows_by_case = {str(row["case"]): row for row in gpu_rows}

    if mismatches:
        lines.extend(
            [
                "",
                "Mismatches:",
                *[_format_mismatch(row) for row in mismatches],
            ]
        )
    elif gpu_rows:
        lines.extend(
            [
                "",
                "Mismatches:",
                "- CPU practical mismatches: none",
                (
                    "- CPU strict-only survivor: "
                    f"`{next(row['case'] for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}` "
                    f"(`{next(int(row.get('strict_n_mismatch_common', 0)) for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}/"
                    f"{next(int(row.get('strict_n_common_keys', 0)) for row in rows if int(row.get('strict_n_mismatch_common', 0)) > 0)}`)"
                    if any(int(row.get("strict_n_mismatch_common", 0)) > 0 for row in rows)
                    else "- CPU strict mismatches: none"
                ),
                "- GPU practical/strict mismatches: none" if not gpu_mismatches else _format_mismatch(gpu_mismatches[0]),
            ]
        )

    lines.extend(
        [
            "",
            "Runtime columns match the summary plot: cold is `jax_runtime_s`; warm/logged is "
            "`jax_runtime_s_warm` when available, otherwise `jax_logged_elapsed_s`. "
            "The JAX memory columns match the plot and use profiler active RSS deltas "
            "(`jax_incremental_max_rss_mb`) when present; full process peak RSS remains "
            "available as `jax_max_rss_mb` in the merged JSON reports.",
            "The benchmark summary JSON records production-resolution floor violations for "
            "frozen reference rows, so the table is a reference-runtime-window comparison "
            "unless a row is also marked as satisfying the production-resolution floor.",
            (
                f"The public runtime/memory table is restricted to cases where the "
                f"SFINCS Fortran v3 reference runtime is at least `{float(args.min_fortran_runtime_s):g} s`. "
                f"Excluded lower-resolution CI parity/smoke rows: "
                f"{_format_excluded_public_cases(excluded_public_cases)}."
            ),
            "",
            "Full per-case runtime / memory table:",
            *_format_case_table(public_case_order, rows_by_case, gpu_rows_by_case),
        ]
    )

    if improvements_runtime:
        lines.extend(
            [
                "",
                f"Largest CPU runtime improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_runtime,
            ]
        )
    if improvements_memory:
        lines.extend(
            [
                "",
                f"Largest CPU process peak-RSS improvements vs `{_repo_rel(baseline_report)}`:",
                *improvements_memory,
            ]
        )

    lines.append(END)

    readme = README.read_text()
    begin = BEGIN if BEGIN in readme else LEGACY_BEGIN
    end = END if END in readme else LEGACY_END
    if begin not in readme or end not in readme:
        raise SystemExit("README example-suite audit markers not found.")
    prefix, rest = readme.split(begin, 1)
    _old, suffix = rest.split(end, 1)
    README.write_text(prefix + "\n".join(lines) + suffix)
    print("Updated README full example-suite audit block.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
