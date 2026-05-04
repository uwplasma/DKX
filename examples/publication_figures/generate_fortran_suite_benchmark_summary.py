#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
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

from sfincs_jax.validation_artifacts import (
    SuiteCaseMetric,
    build_fortran_suite_benchmark_summary,
    filter_suite_metrics_by_fortran_runtime,
    load_suite_report,
    suite_case_metrics,
)


DEFAULT_CPU_REPORT = (
    _REPO_ROOT / "tests" / "scaled_example_suite_release_cpu_frozen_2026-04-25_v106" / "suite_report.json"
)
DEFAULT_GPU_REPORT = (
    _REPO_ROOT
    / "tests"
    / "scaled_example_suite_gpu_bounded_default_2026-04-28"
    / "suite_report.json"
)
DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_STEM = "sfincs_jax_fortran_suite_benchmark_summary"
DEFAULT_MIN_FORTRAN_RUNTIME_S = 10.0


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
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser


def write_benchmark_summary(
    *,
    cpu_report: Path,
    gpu_report: Path,
    summary_json: Path,
    min_fortran_runtime_s: float | None = DEFAULT_MIN_FORTRAN_RUNTIME_S,
) -> dict[str, object]:
    payload = build_fortran_suite_benchmark_summary(
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


def _value_or_nan(value: float | None) -> float:
    if value is None or not np.isfinite(value) or value <= 0.0:
        return float("nan")
    return float(value)


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
            # The frozen suite records peak RSS per external command. If future
            # reports add separate warm-run RSS fields, this is the only place
            # that needs to split the cold/warm memory bars.
            values_by_label["sfincs_jax CPU cold"].append(
                _value_or_nan(cpu_metric.jax_max_rss_mb if cpu_metric else None)
            )
            values_by_label["sfincs_jax CPU warm"].append(
                _value_or_nan(cpu_metric.jax_max_rss_mb if cpu_metric else None)
            )
            values_by_label["sfincs_jax GPU cold"].append(
                _value_or_nan(gpu_metric.jax_max_rss_mb if gpu_metric else None)
            )
            values_by_label["sfincs_jax GPU warm"].append(
                _value_or_nan(gpu_metric.jax_max_rss_mb if gpu_metric else None)
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
    cpu_metrics, gpu_metrics, excluded_cases = filter_suite_metrics_by_fortran_runtime(
        suite_case_metrics(load_suite_report(Path(cpu_report))),
        suite_case_metrics(load_suite_report(Path(gpu_report))),
        min_fortran_runtime_s=min_fortran_runtime_s,
    )
    if not cpu_metrics and not gpu_metrics:
        raise SystemExit(
            "No benchmark rows remain after applying --min-fortran-runtime-s="
            f"{min_fortran_runtime_s}. Lower the threshold or regenerate production-resolution reports."
        )
    cpu_by_case = _metric_by_case(cpu_metrics)
    gpu_by_case = _metric_by_case(gpu_metrics)
    cases = _case_order(cpu_metrics, gpu_metrics)

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
        title="B. Peak memory",
        xlabel="maximum RSS, MB (log scale)",
        show_ylabels=False,
    )
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.27, right=0.985, top=0.915, bottom=0.075, wspace=0.04)
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.61, 0.972))
    if min_fortran_runtime_s is None or float(min_fortran_runtime_s) <= 0.0:
        scope = "all audited example-suite cases"
    else:
        scope = f"production-scale rows; Fortran v3 >= {float(min_fortran_runtime_s):g} s"
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
        "Warm memory uses the recorded peak-RSS field unless a future report provides separate warm RSS.",
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
    write_benchmark_summary(
        cpu_report=Path(args.cpu_report),
        gpu_report=Path(args.gpu_report),
        summary_json=Path(args.summary_json),
        min_fortran_runtime_s=float(args.min_fortran_runtime_s),
    )
    plot_benchmark_summary(
        cpu_report=Path(args.cpu_report),
        gpu_report=Path(args.gpu_report),
        out_dir=Path(args.out_dir),
        stem=str(args.stem),
        min_fortran_runtime_s=float(args.min_fortran_runtime_s),
    )
    print(f"Wrote suite benchmark summary to {Path(args.summary_json)}")
    print(f"Wrote suite benchmark figures to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
