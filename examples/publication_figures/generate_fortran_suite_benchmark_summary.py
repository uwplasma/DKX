#!/usr/bin/env python
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import os
import sys
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
        description="Generate a CPU/GPU benchmark dashboard from frozen Fortran v3 suite comparisons.",
    )
    parser.add_argument("--cpu-report", type=Path, default=DEFAULT_CPU_REPORT)
    parser.add_argument("--gpu-report", type=Path, default=DEFAULT_GPU_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}.json",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    return parser


def _finite_values(metrics: list[SuiteCaseMetric], attr: str) -> np.ndarray:
    values = [getattr(metric, attr) for metric in metrics]
    return np.asarray([value for value in values if value is not None and np.isfinite(value)], dtype=np.float64)


def write_benchmark_summary(*, cpu_report: Path, gpu_report: Path, summary_json: Path) -> dict[str, object]:
    payload = build_fortran_suite_benchmark_summary(cpu_report=Path(cpu_report), gpu_report=Path(gpu_report))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _plot_status_panel(ax, summary: dict[str, object]) -> None:
    labels = ["CPU", "GPU"]
    totals = [summary["reports"][key]["total_cases"] for key in ("cpu", "gpu")]
    parity = [summary["reports"][key]["parity_ok_cases"] for key in ("cpu", "gpu")]
    errors = [
        summary["reports"][key]["jax_error_cases"] + summary["reports"][key]["max_attempts_cases"]
        for key in ("cpu", "gpu")
    ]
    x = np.arange(len(labels))
    ax.bar(x, totals, color="#d8dee9", label="audited cases")
    ax.bar(x, parity, color="#1b9e77", label="parity_ok")
    for idx, (ok, total, err) in enumerate(zip(parity, totals, errors)):
        ax.text(idx, total + 0.9, f"{ok}/{total}\nerrors {err}", ha="center", va="bottom", fontsize=9.0)
    ax.set_xticks(x, labels)
    ax.set_ylim(0.0, max(totals) * 1.22)
    ax.set_ylabel("case count")
    ax.set_title("A. Frozen suite parity gate")
    ax.legend(loc="lower right")


def _plot_ratio_distribution(ax, metrics_by_backend: dict[str, list[SuiteCaseMetric]], *, attr: str, title: str) -> None:
    colors = {"CPU": "#0f4c81", "GPU": "#d95f02"}
    labels = list(metrics_by_backend)
    positions = np.arange(len(labels), dtype=float)
    for x0, label in zip(positions, labels):
        values = _finite_values(metrics_by_backend[label], attr)
        jitter = np.linspace(-0.13, 0.13, max(len(values), 1))
        ax.scatter(
            np.full_like(values, x0) + jitter[: len(values)],
            values,
            s=22,
            color=colors[label],
            alpha=0.68,
            edgecolor="white",
            linewidth=0.3,
        )
        if values.size:
            median = float(np.median(values))
            ax.hlines(median, x0 - 0.28, x0 + 0.28, color="black", linewidth=1.8)
            ax.text(x0, median * 1.08, f"median {median:.2g}x", ha="center", va="bottom", fontsize=8.2)
    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_yscale("log")
    ax.set_xticks(positions, labels)
    ax.set_ylabel("JAX / Fortran v3")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.24)


def _plot_runtime_memory_scatter(ax, metrics_by_backend: dict[str, list[SuiteCaseMetric]]) -> None:
    colors = {"CPU": "#0f4c81", "GPU": "#d95f02"}
    for label, metrics in metrics_by_backend.items():
        x_values: list[float] = []
        y_values: list[float] = []
        for metric in metrics:
            if metric.runtime_ratio is None or metric.memory_ratio is None:
                continue
            x_values.append(float(metric.runtime_ratio))
            y_values.append(float(metric.memory_ratio))
        ax.scatter(
            x_values,
            y_values,
            s=34,
            color=colors[label],
            label=label,
            alpha=0.72,
            edgecolor="white",
            linewidth=0.35,
        )
    ax.axvline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.axhline(1.0, color="black", linewidth=0.9, linestyle="--", alpha=0.55)
    ax.text(
        0.04,
        0.94,
        "Top runtime and memory cases are\nlisted in the summary JSON.",
        transform=ax.transAxes,
        va="top",
        fontsize=8.0,
        bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("runtime ratio")
    ax.set_ylabel("memory ratio")
    ax.set_title("D. Runtime-memory trade space")
    ax.legend(loc="best")
    ax.grid(True, which="both", alpha=0.24)


def plot_benchmark_summary(*, cpu_report: Path, gpu_report: Path, out_dir: Path, stem: str) -> None:
    cpu_metrics = suite_case_metrics(load_suite_report(Path(cpu_report)))
    gpu_metrics = suite_case_metrics(load_suite_report(Path(gpu_report)))
    summary = build_fortran_suite_benchmark_summary(cpu_report=Path(cpu_report), gpu_report=Path(gpu_report))
    metrics_by_backend = {"CPU": cpu_metrics, "GPU": gpu_metrics}

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.5), constrained_layout=True)
    _plot_status_panel(axes[0, 0], summary)
    _plot_ratio_distribution(
        axes[0, 1],
        metrics_by_backend,
        attr="runtime_ratio",
        title="B. Wall-clock runtime ratios",
    )
    _plot_ratio_distribution(
        axes[1, 0],
        metrics_by_backend,
        attr="memory_ratio",
        title="C. Maximum-RSS memory ratios",
    )
    _plot_runtime_memory_scatter(axes[1, 1], metrics_by_backend)
    fig.suptitle("sfincs_jax vs SFINCS Fortran v3: frozen CPU/GPU suite benchmark", fontsize=12.5)
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
    )
    plot_benchmark_summary(
        cpu_report=Path(args.cpu_report),
        gpu_report=Path(args.gpu_report),
        out_dir=Path(args.out_dir),
        stem=str(args.stem),
    )
    print(f"Wrote suite benchmark summary to {Path(args.summary_json)}")
    print(f"Wrote suite benchmark figures to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
