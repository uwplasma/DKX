#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from collections.abc import Mapping, Sequence
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit("This example requires matplotlib. Install with: pip install matplotlib") from exc

import numpy as np

from sfincs_jax.validation.artifacts import (
    SuiteCaseMetric,
    build_fortran_suite_benchmark_summary,
    filter_suite_metrics_by_fortran_runtime,
    load_suite_report,
    suite_case_metrics,
)


DEFAULT_CPU_REPORT = (
    _REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_2026-05-08_production_tokamak" / "suite_report.json"
)
DEFAULT_GPU_REPORT = (
    _REPO_ROOT
    / "tests"
    / "scaled_example_suite_gpu_bounded_default_2026-05-08_lu3000_pas"
    / "suite_report.json"
)
DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_STEM = "sfincs_jax_fortran_suite_benchmark_summary"
DEFAULT_MIN_FORTRAN_RUNTIME_S = 10.0
CANONICAL_ROWS_KEY = "canonical_rows"
CANONICAL_CASE_ORDER_KEY = "canonical_case_order"
README_TABLE_HEADER = (
    "| Case | Fortran CPU(s) | JAX CPU cold(s) | CPU cold x | "
    "JAX CPU warm/logged(s) | CPU warm/logged x | JAX GPU cold(s) | "
    "GPU cold x | JAX GPU warm/logged(s) | GPU warm/logged x | Fortran MB | "
    "JAX CPU active MB | CPU MB x | JAX GPU active MB | GPU MB x | "
    "CPU mismatch | GPU mismatch | CPU print | GPU print | CPU status | GPU status |"
)
README_TABLE_SEPARATOR = (
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | "
    "---: | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |"
)


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "grid.linestyle": "-",
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_fortran_suite_benchmark_summary",
        description="Generate CPU/GPU runtime and memory bars from frozen Fortran v3 suite comparisons.",
    )
    parser.add_argument("--cpu-report", type=Path, default=DEFAULT_CPU_REPORT)
    parser.add_argument("--gpu-report", type=Path, default=DEFAULT_GPU_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}.json",
    )
    parser.add_argument(
        "--min-fortran-runtime-s",
        type=float,
        default=DEFAULT_MIN_FORTRAN_RUNTIME_S,
        help=(
            "Minimum SFINCS Fortran v3 runtime for README-facing performance rows. "
            "Use 0 to regenerate the all-case smoke/regression comparison."
        ),
    )
    parser.add_argument(
        "--enforce-public-resolution-floor",
        action="store_true",
        help=(
            "Fail if any plotted row is below the documented production-resolution floor. "
            "By default, legacy frozen-report regeneration is allowed and the floor "
            "violations are recorded in the summary JSON."
        ),
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser


def write_benchmark_summary(
    *,
    cpu_report: Path,
    gpu_report: Path,
    summary_json: Path,
    min_fortran_runtime_s: float | None = DEFAULT_MIN_FORTRAN_RUNTIME_S,
    enforce_public_resolution_floor: bool = False,
) -> dict[str, object]:
    payload = build_fortran_suite_benchmark_summary(
        cpu_report=Path(cpu_report),
        gpu_report=Path(gpu_report),
        min_fortran_runtime_s=min_fortran_runtime_s,
        enforce_public_resolution_floor=enforce_public_resolution_floor,
    )
    _attach_canonical_rows(
        payload,
        cpu_report=Path(cpu_report),
        gpu_report=Path(gpu_report),
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _case_label(case: str) -> str:
    """Return a compact case label that still identifies every suite row."""

    replacements = (
        ("sfincsPaperFigure3_", "paperFig3_"),
        ("filteredW7XNetCDF", "W7X"),
        ("geometryScheme", "geom"),
        ("monoenergetic", "mono"),
        ("transportMatrix", "transport"),
        ("FPCollisions", "FP"),
        ("PASCollisions", "PAS"),
        ("DKESTrajectories", "DKES"),
        ("fullTrajectories", "full"),
        ("magneticDrifts", "magDrift"),
        ("tokamak_", "tok_"),
        ("additional_examples", "additional"),
    )
    label = case
    for old, new in replacements:
        label = label.replace(old, new)
    return "\n".join(textwrap.wrap(label, width=32, break_long_words=False, break_on_hyphens=False))


def _metric_by_case(metrics: list[SuiteCaseMetric]) -> dict[str, SuiteCaseMetric]:
    return {metric.case: metric for metric in metrics}


def _case_order(cpu_metrics: list[SuiteCaseMetric], gpu_metrics: list[SuiteCaseMetric]) -> list[str]:
    """Put cases with a faster warm JAX path first, ordered by largest speedup."""

    by_case = _metric_by_case(cpu_metrics) | _metric_by_case(gpu_metrics)
    cpu_by_case = _metric_by_case(cpu_metrics)
    gpu_by_case = _metric_by_case(gpu_metrics)

    def best_warm_speedup(case: str) -> float:
        reference_metric = cpu_by_case.get(case) or gpu_by_case.get(case)
        fortran_runtime = reference_metric.fortran_runtime_s if reference_metric else None
        warm_times = [
            metric.warm_or_logged_runtime_s
            for metric in (cpu_by_case.get(case), gpu_by_case.get(case))
            if metric is not None and metric.warm_or_logged_runtime_s is not None
        ]
        finite_warm_times = [float(value) for value in warm_times if np.isfinite(float(value)) and float(value) > 0.0]
        if fortran_runtime is None or not np.isfinite(float(fortran_runtime)) or float(fortran_runtime) <= 0.0:
            return float("nan")
        if not finite_warm_times:
            return float("nan")
        return float(fortran_runtime) / min(finite_warm_times)

    def sort_key(case: str) -> tuple[int, float, str]:
        speedup = best_warm_speedup(case)
        if not np.isfinite(speedup):
            return (2, 0.0, case)
        if speedup >= 1.0:
            return (0, -speedup, case)
        return (1, -speedup, case)

    return sorted(by_case, key=sort_key)


def _canonical_metric_row(metric: SuiteCaseMetric, raw_row: Mapping[str, object]) -> dict[str, object]:
    """Return the canonical public runtime/memory row used by plots and README checks."""

    row: dict[str, object] = {
        "case": metric.case,
        "status": metric.status,
        "blocker_type": metric.blocker_type,
        "fortran_runtime_s": metric.fortran_runtime_s,
        "jax_runtime_s": metric.jax_runtime_s,
        "jax_runtime_s_cold": metric.jax_runtime_s_cold,
        "jax_runtime_s_warm": metric.jax_runtime_s_warm,
        "jax_logged_elapsed_s": metric.jax_logged_elapsed_s,
        "warm_or_logged_runtime_s": metric.warm_or_logged_runtime_s,
        "warm_or_logged_runtime_source": metric.warm_or_logged_runtime_source,
        "fortran_max_rss_mb": metric.fortran_max_rss_mb,
        "jax_max_rss_mb": metric.jax_max_rss_mb,
        "jax_incremental_max_rss_mb": metric.jax_incremental_max_rss_mb,
        "jax_rss_baseline_mb": metric.jax_rss_baseline_mb,
        "jax_memory_metric_source": metric.jax_memory_metric_source,
        "active_jax_memory_mb": metric.active_jax_memory_mb,
        "runtime_ratio": metric.runtime_ratio,
        "cold_runtime_ratio": metric.cold_runtime_ratio,
        "warm_runtime_ratio": metric.warm_runtime_ratio,
        "warm_or_logged_runtime_ratio": metric.warm_or_logged_runtime_ratio,
        "logged_runtime_ratio": metric.logged_runtime_ratio,
        "memory_ratio": metric.memory_ratio,
        "active_memory_ratio": metric.active_memory_ratio,
        "n_mismatch_common": raw_row.get("n_mismatch_common", 0),
        "n_common_keys": raw_row.get("n_common_keys", 0),
        "strict_n_mismatch_common": raw_row.get("strict_n_mismatch_common", 0),
        "strict_n_common_keys": raw_row.get("strict_n_common_keys", 0),
        "print_parity_signals": raw_row.get("print_parity_signals", 0),
        "print_parity_total": raw_row.get("print_parity_total", 0),
    }
    final_resolution = raw_row.get("final_resolution")
    if isinstance(final_resolution, Mapping):
        row["final_resolution"] = dict(final_resolution)
    return row


def _canonical_rows_for_metrics(
    metrics: Sequence[SuiteCaseMetric],
    raw_rows: Sequence[Mapping[str, object]],
    *,
    case_order: Sequence[str],
) -> list[dict[str, object]]:
    metrics_by_case = _metric_by_case(list(metrics))
    raw_by_case = {str(row.get("case", "")): row for row in raw_rows}
    rows: list[dict[str, object]] = []
    for case in case_order:
        metric = metrics_by_case.get(case)
        raw_row = raw_by_case.get(case)
        if metric is None or raw_row is None:
            continue
        rows.append(_canonical_metric_row(metric, raw_row))
    return rows


def canonical_benchmark_rows_from_reports(
    *,
    cpu_report: Path,
    gpu_report: Path,
    min_fortran_runtime_s: float | None = DEFAULT_MIN_FORTRAN_RUNTIME_S,
) -> dict[str, object]:
    """Build the canonical filtered rows for release plot/table artifacts.

    The CPU/GPU suite reports remain the measured data source. This function
    applies the public Fortran-runtime policy once, computes the public
    cold/warm/runtime/memory fields once, and returns the row order used by the
    runtime/memory plot.
    """

    raw_cpu_rows = load_suite_report(Path(cpu_report))
    raw_gpu_rows = load_suite_report(Path(gpu_report))
    raw_cpu_metrics = suite_case_metrics(raw_cpu_rows)
    raw_gpu_metrics = suite_case_metrics(raw_gpu_rows)
    cpu_metrics, gpu_metrics, excluded_cases = filter_suite_metrics_by_fortran_runtime(
        raw_cpu_metrics,
        raw_gpu_metrics,
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    case_order = _case_order(cpu_metrics, gpu_metrics)
    return {
        "metadata": {
            "min_fortran_runtime_s": None if min_fortran_runtime_s is None else float(min_fortran_runtime_s),
            "excluded_low_fortran_runtime_cases": excluded_cases,
            CANONICAL_CASE_ORDER_KEY: case_order,
        },
        "rows": {
            "cpu": _canonical_rows_for_metrics(cpu_metrics, raw_cpu_rows, case_order=case_order),
            "gpu": _canonical_rows_for_metrics(gpu_metrics, raw_gpu_rows, case_order=case_order),
        },
    }


def _attach_canonical_rows(
    payload: dict[str, object],
    *,
    cpu_report: Path,
    gpu_report: Path,
    min_fortran_runtime_s: float | None,
) -> None:
    canonical = canonical_benchmark_rows_from_reports(
        cpu_report=Path(cpu_report),
        gpu_report=Path(gpu_report),
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    rows = canonical.get("rows")
    if isinstance(rows, Mapping):
        payload[CANONICAL_ROWS_KEY] = rows
    metadata = payload.setdefault("metadata", {})
    if isinstance(metadata, dict):
        canonical_metadata = canonical.get("metadata")
        if isinstance(canonical_metadata, Mapping):
            metadata[CANONICAL_CASE_ORDER_KEY] = list(canonical_metadata.get(CANONICAL_CASE_ORDER_KEY, []))
            metadata["canonical_row_source"] = "filtered_cpu_gpu_suite_reports"


def _summary_rows_by_backend(payload: Mapping[str, object]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    canonical_rows = payload.get(CANONICAL_ROWS_KEY)
    if not isinstance(canonical_rows, Mapping):
        raise ValueError(
            f"benchmark summary is missing {CANONICAL_ROWS_KEY!r}; regenerate with this script"
        )
    cpu_rows = canonical_rows.get("cpu")
    gpu_rows = canonical_rows.get("gpu")
    if not isinstance(cpu_rows, list) or not isinstance(gpu_rows, list):
        raise ValueError(f"benchmark summary field {CANONICAL_ROWS_KEY!r} must contain cpu/gpu row lists")
    return (
        [dict(row) for row in cpu_rows if isinstance(row, Mapping)],
        [dict(row) for row in gpu_rows if isinstance(row, Mapping)],
    )


def _case_order_from_summary(payload: Mapping[str, object]) -> list[str]:
    metadata = payload.get("metadata")
    if isinstance(metadata, Mapping):
        case_order = metadata.get(CANONICAL_CASE_ORDER_KEY)
        if isinstance(case_order, list) and all(isinstance(case, str) for case in case_order):
            return list(case_order)

    cpu_rows, gpu_rows = _summary_rows_by_backend(payload)
    return _case_order(suite_case_metrics(cpu_rows), suite_case_metrics(gpu_rows))


def _value_or_nan(value: float | None) -> float:
    if value is None or not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(value)


def _fmt_float(value: object | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{float(value):.{digits}f}"


def _fmt_ratio(numerator: object | None, denominator: object | None) -> str:
    if numerator is None or denominator in (None, 0):
        return "-"
    return f"{float(numerator) / float(denominator):.2f}x"


def _fmt_mismatch_pair(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "-"
    return (
        f"{int(row.get('n_mismatch_common', 0))}/{int(row.get('n_common_keys', 0))}"
        f" (strict {int(row.get('strict_n_mismatch_common', 0))}/"
        f"{int(row.get('strict_n_common_keys', 0))})"
    )


def _fmt_print_parity(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "-"
    total = int(row.get("print_parity_total", 0))
    if total <= 0:
        return "-"
    return f"{int(row.get('print_parity_signals', 0))}/{total}"


def _fmt_status(row: Mapping[str, object] | None) -> str:
    if row is None:
        return "-"
    status = str(row.get("status", "")).strip()
    return status or "-"


def readme_benchmark_table_lines_from_payload(payload: Mapping[str, object]) -> list[str]:
    """Format the README-facing runtime/memory table from canonical summary rows."""

    cpu_rows, gpu_rows = _summary_rows_by_backend(payload)
    cpu_by_case = {str(row.get("case", "")): row for row in cpu_rows}
    gpu_by_case = {str(row.get("case", "")): row for row in gpu_rows}
    lines = [README_TABLE_HEADER, README_TABLE_SEPARATOR]
    for case in _case_order_from_summary(payload):
        cpu_row = cpu_by_case.get(case)
        gpu_row = gpu_by_case.get(case)
        if cpu_row is None and gpu_row is None:
            continue
        reference_row = cpu_row or gpu_row
        fort_runtime = reference_row.get("fortran_runtime_s") if reference_row else None
        fort_memory = reference_row.get("fortran_max_rss_mb") if reference_row else None
        cpu_runtime = cpu_row.get("jax_runtime_s") if cpu_row else None
        gpu_runtime = gpu_row.get("jax_runtime_s") if gpu_row else None
        cpu_warm_runtime = cpu_row.get("warm_or_logged_runtime_s") if cpu_row else None
        gpu_warm_runtime = gpu_row.get("warm_or_logged_runtime_s") if gpu_row else None
        cpu_memory = cpu_row.get("active_jax_memory_mb") if cpu_row else None
        gpu_memory = gpu_row.get("active_jax_memory_mb") if gpu_row else None
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


def _plot_grouped_bar_panel(
    ax,
    *,
    cases: list[str],
    cpu_by_case: dict[str, SuiteCaseMetric],
    gpu_by_case: dict[str, SuiteCaseMetric],
    quantity: str,
    title: str,
    xlabel: str,
    show_ylabels: bool,
) -> None:
    y = np.arange(len(cases), dtype=float)
    height = 0.13
    series = (
        ("SFINCS Fortran v3", "#2f3437", 2.0),
        ("sfincs_jax CPU cold", "#1f4f8a", 1.0),
        ("sfincs_jax CPU warm", "#7fb3e8", 0.0),
        ("sfincs_jax GPU cold", "#b45309", -1.0),
        ("sfincs_jax GPU warm", "#f6ad55", -2.0),
    )
    values_by_label: dict[str, list[float]] = {label: [] for label, _, _ in series}
    for case in cases:
        cpu_metric = cpu_by_case.get(case)
        gpu_metric = gpu_by_case.get(case)
        reference_metric = cpu_metric or gpu_metric
        if quantity == "runtime":
            values_by_label["SFINCS Fortran v3"].append(
                _value_or_nan(reference_metric.fortran_runtime_s if reference_metric else None)
            )
            values_by_label["sfincs_jax CPU cold"].append(
                _value_or_nan(cpu_metric.jax_runtime_s_cold if cpu_metric else None)
            )
            values_by_label["sfincs_jax CPU warm"].append(_jax_warm_runtime_for_plot(cpu_metric))
            values_by_label["sfincs_jax GPU cold"].append(
                _value_or_nan(gpu_metric.jax_runtime_s_cold if gpu_metric else None)
            )
            values_by_label["sfincs_jax GPU warm"].append(_jax_warm_runtime_for_plot(gpu_metric))
        elif quantity == "memory":
            values_by_label["SFINCS Fortran v3"].append(
                _value_or_nan(reference_metric.fortran_max_rss_mb if reference_metric else None)
            )
            # JAX reports keep full process RSS for audit, but the public plot
            # uses profiler-derived active solver memory when it is available so
            # fixed Python/JAX/XLA runtime overhead does not dominate every bar.
            values_by_label["sfincs_jax CPU cold"].append(
                _value_or_nan(cpu_metric.active_jax_memory_mb if cpu_metric else None)
            )
            values_by_label["sfincs_jax CPU warm"].append(
                _value_or_nan(cpu_metric.active_jax_memory_mb if cpu_metric else None)
            )
            values_by_label["sfincs_jax GPU cold"].append(
                _value_or_nan(gpu_metric.active_jax_memory_mb if gpu_metric else None)
            )
            values_by_label["sfincs_jax GPU warm"].append(
                _value_or_nan(gpu_metric.active_jax_memory_mb if gpu_metric else None)
            )
        else:  # pragma: no cover - caller bug
            raise ValueError(f"Unknown quantity: {quantity}")

    for label, color, offset in series:
        ax.barh(y + offset * height, values_by_label[label], height=height, label=label, color=color)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(True, which="both", axis="x", alpha=0.25)
    ax.grid(False, axis="y")
    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([_case_label(case) for case in cases], fontsize=6.4)
    else:
        ax.tick_params(axis="y", labelleft=False, length=0)


def _jax_warm_runtime_for_plot(metric: SuiteCaseMetric | None) -> float:
    """Return true warm-rerun time when present, else the logged in-process elapsed time."""

    if metric is None:
        return float("nan")
    return _value_or_nan(metric.warm_or_logged_runtime_s)


def plot_benchmark_summary(
    *,
    cpu_report: Path,
    gpu_report: Path,
    out_dir: Path,
    stem: str,
    min_fortran_runtime_s: float | None = DEFAULT_MIN_FORTRAN_RUNTIME_S,
) -> None:
    payload = build_fortran_suite_benchmark_summary(
        cpu_report=Path(cpu_report),
        gpu_report=Path(gpu_report),
        min_fortran_runtime_s=min_fortran_runtime_s,
        enforce_public_resolution_floor=False,
    )
    _attach_canonical_rows(
        payload,
        cpu_report=Path(cpu_report),
        gpu_report=Path(gpu_report),
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    plot_benchmark_summary_from_payload(payload=payload, out_dir=out_dir, stem=stem)


def plot_benchmark_summary_from_payload(
    *,
    payload: Mapping[str, object],
    out_dir: Path,
    stem: str,
) -> None:
    cpu_rows, gpu_rows = _summary_rows_by_backend(payload)
    cpu_metrics = suite_case_metrics(cpu_rows)
    gpu_metrics = suite_case_metrics(gpu_rows)
    if not cpu_metrics and not gpu_metrics:
        metadata = payload.get("metadata", {})
        min_fortran_runtime_s = (
            metadata.get("min_fortran_runtime_s") if isinstance(metadata, Mapping) else None
        )
        raise SystemExit(
            "No benchmark rows remain after applying the summary JSON Fortran-runtime filter "
            f"({min_fortran_runtime_s}). Lower the threshold or regenerate production-resolution reports."
        )
    cpu_by_case = _metric_by_case(cpu_metrics)
    gpu_by_case = _metric_by_case(gpu_metrics)
    cases = _case_order_from_summary(payload)
    metadata = payload.get("metadata", {})
    excluded_cases = (
        metadata.get("excluded_low_fortran_runtime_cases", []) if isinstance(metadata, Mapping) else []
    )
    min_fortran_runtime_s = (
        metadata.get("min_fortran_runtime_s") if isinstance(metadata, Mapping) else None
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_height = max(14.5, 0.40 * len(cases) + 2.0)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14.5, fig_height),
        sharey=True,
        gridspec_kw={"width_ratios": [1.0, 1.0]},
    )
    _plot_grouped_bar_panel(
        axes[0],
        cases=cases,
        cpu_by_case=cpu_by_case,
        gpu_by_case=gpu_by_case,
        quantity="runtime",
        title="A. Wall-clock runtime",
        xlabel="seconds (log scale)",
        show_ylabels=True,
    )
    _plot_grouped_bar_panel(
        axes[1],
        cases=cases,
        cpu_by_case=cpu_by_case,
        gpu_by_case=gpu_by_case,
        quantity="memory",
        title="B. Active solver memory",
        xlabel="MB (Fortran max RSS; JAX active RSS delta, log scale)",
        show_ylabels=False,
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.27, right=0.985, top=0.915, bottom=0.075, wspace=0.04)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.61, 0.972))
    if min_fortran_runtime_s is None or float(min_fortran_runtime_s) <= 0.0:
        scope = "all audited example-suite cases"
    else:
        scope = f"reference-runtime rows; Fortran v3 >= {float(min_fortran_runtime_s):g} s"
    fig.suptitle(
        f"SFINCS Fortran v3 vs sfincs_jax cold/warm CPU/GPU: {scope}",
        fontsize=13.0,
        y=0.998,
    )
    fig.text(
        0.63,
        0.027,
        "Cold = first external suite command. Warm runtime = jax_runtime_s_warm when present, otherwise CLI "
        "jax_logged_elapsed_s.\n"
        f"Excluded low-reference-runtime CI/smoke rows: {len(excluded_cases)}. "
        "Resolution-floor audit is recorded in JSON. JAX memory bars use profiler active RSS deltas.",
        ha="center",
        va="bottom",
        fontsize=8.0,
    )
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    payload = write_benchmark_summary(
        cpu_report=Path(args.cpu_report),
        gpu_report=Path(args.gpu_report),
        summary_json=Path(args.summary_json),
        min_fortran_runtime_s=float(args.min_fortran_runtime_s),
        enforce_public_resolution_floor=bool(args.enforce_public_resolution_floor),
    )
    plot_benchmark_summary_from_payload(
        payload=payload,
        out_dir=Path(args.out_dir),
        stem=str(args.stem),
    )
    print(f"Wrote suite benchmark summary to {Path(args.summary_json)}")
    print(f"Wrote suite benchmark figures to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
