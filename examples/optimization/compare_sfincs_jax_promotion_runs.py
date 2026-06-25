#!/usr/bin/env python
"""Compare CPU/GPU and optional Fortran-v3 optimization promotion summaries."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path("/tmp") / "mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.workflows.optimization_comparison import (  # noqa: E402
    PromotionComparisonTolerances,
    compare_optimization_promotions,
    load_promotion_payload,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu", type=Path, required=True, help="CPU promotion JSON payload.")
    parser.add_argument("--gpu", type=Path, required=True, help="GPU promotion JSON payload.")
    parser.add_argument("--fortran", type=Path, help="Optional Fortran-v3-derived promotion JSON payload.")
    parser.add_argument("--out-dir", type=Path, default=Path.cwd(), help="Directory for JSON/PNG/PDF outputs.")
    parser.add_argument("--stem", default="optimization_promotion_comparison", help="Output file stem.")
    parser.add_argument("--root-rtol", type=float, default=1.0e-7)
    parser.add_argument("--root-atol", type=float, default=1.0e-10)
    parser.add_argument("--bootstrap-rtol", type=float, default=1.0e-6)
    parser.add_argument("--flux-rtol", type=float, default=1.0e-6)
    parser.add_argument("--allow-missing-flux", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print summary JSON.")
    return parser


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 11.0,
            "legend.fontsize": 8.6,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.grid": True,
            "grid.alpha": 0.24,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _lookup(payload: dict, *keys: str):
    current = payload
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def _metrics(payload: dict) -> dict[str, float | None]:
    return {
        "root Er": _lookup(payload, "selected_root", "er"),
        "bootstrap J": payload.get("bootstrap_objective"),
        "flux J": _lookup(payload, "flux_objective", "total"),
    }


def _plot(path_png: Path, path_pdf: Path, *, payloads: list[tuple[str, dict]], comparison: dict) -> None:
    _setup_mpl()
    names = [name for name, _ in payloads]
    metric_names = ["root Er", "bootstrap J", "flux J"]
    values = np.asarray(
        [
            [
                np.nan if _metrics(payload)[metric] is None else float(_metrics(payload)[metric])
                for metric in metric_names
            ]
            for _, payload in payloads
        ],
        dtype=float,
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4), constrained_layout=True)

    ax = axes[0]
    width = 0.75 / max(len(names), 1)
    x = np.asarray([0.0])
    for idx, name in enumerate(names):
        ax.bar(x + (idx - (len(names) - 1) / 2.0) * width, values[idx, 0], width=width, label=name)
    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["selected root"])
    ax.set_title("A. Ambipolar root")
    ax.set_ylabel(r"$E_r$")
    ax.legend(loc="best")

    ax = axes[1]
    objective_names = metric_names[1:]
    objective_values = np.where(values[:, 1:] > 0.0, values[:, 1:], np.nan)
    x = np.arange(len(objective_names), dtype=float)
    width = 0.75 / max(len(names), 1)
    for idx, name in enumerate(names):
        ax.bar(
            x + (idx - (len(names) - 1) / 2.0) * width,
            objective_values[idx],
            width=width,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(objective_names, rotation=18, ha="right")
    ax.set_yscale("log")
    ax.set_title("B. Promotion objectives")
    ax.set_ylabel("objective value")

    ax = axes[2]
    labels = []
    abs_diffs = []
    limits = []
    for comparison_name, item in comparison["comparisons"].items():
        for metric_name, metric in item["metrics"].items():
            if metric_name == "gate_status" or metric.get("abs_diff") is None:
                continue
            labels.append(f"{comparison_name}\n{metric_name}")
            abs_diffs.append(float(metric["abs_diff"]))
            limits.append(float(metric["limit"]))
    if labels:
        xx = np.arange(len(labels))
        ax.bar(xx, abs_diffs, color="#0f4c81", label="abs diff")
        ax.plot(xx, limits, color="#d95f02", marker="o", lw=2.0, label="gate limit")
        ax.set_yscale("log")
        ax.set_xticks(xx)
        ax.set_xticklabels(labels, rotation=28, ha="right")
    ax.set_title(f"C. Comparison gates: {comparison['status']}")
    ax.set_ylabel("absolute difference")
    ax.legend(loc="best")

    path_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_png)
    fig.savefig(path_pdf)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    tolerances = PromotionComparisonTolerances(
        selected_root_er_rtol=float(args.root_rtol),
        selected_root_er_atol=float(args.root_atol),
        bootstrap_objective_rtol=float(args.bootstrap_rtol),
        flux_objective_total_rtol=float(args.flux_rtol),
    )
    comparison = compare_optimization_promotions(
        args.cpu,
        args.gpu,
        fortran_v3_payload=args.fortran,
        tolerances=tolerances,
        require_flux_objective=not bool(args.allow_missing_flux),
    )
    cpu_payload = load_promotion_payload(args.cpu)
    gpu_payload = load_promotion_payload(args.gpu)
    payloads = [("CPU", cpu_payload), ("GPU", gpu_payload)]
    if args.fortran is not None:
        payloads.append(("Fortran v3", load_promotion_payload(args.fortran)))

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.stem}.json"
    png_path = out_dir / f"{args.stem}.png"
    pdf_path = out_dir / f"{args.stem}.pdf"
    output = {
        **comparison,
        "inputs": {
            "cpu": str(args.cpu.resolve()),
            "gpu": str(args.gpu.resolve()),
            "fortran": None if args.fortran is None else str(args.fortran.resolve()),
        },
    }
    json_path.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _plot(png_path, pdf_path, payloads=payloads, comparison=comparison)

    if args.json:
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print("sfincs_jax optimization promotion comparison complete")
        print(f"  status:  {comparison['status']}")
        print(f"  summary: {json_path}")
        print(f"  figure:  {png_path}")
        print(f"  pdf:     {pdf_path}")
    return 0 if comparison["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
