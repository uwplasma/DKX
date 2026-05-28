#!/usr/bin/env python
"""Summarize finite-beta QA electron-root promotion convergence ladders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.optimization_ladder import evaluate_promotion_ladder, load_ladder_config  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Promotion ladder config JSON.")
    parser.add_argument("--out-dir", type=Path, required=True, help="Directory for JSON/PNG/PDF outputs.")
    parser.add_argument("--stem", default="finite_beta_electron_root_ladder", help="Output file stem.")
    parser.add_argument("--backend-root-atol", type=float, default=1.0e-6)
    parser.add_argument("--root-drift-atol", type=float, default=2.0e-2)
    parser.add_argument("--json", action="store_true", help="Print the summary JSON.")
    return parser


def _plot(summary: dict[str, object], *, out_dir: Path, stem: str) -> tuple[Path, Path]:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np

    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    tiers = list(summary["tiers"])  # type: ignore[index]
    names = [str(tier["name"]) for tier in tiers]
    active = np.asarray([float(tier["active_size_estimate"]) for tier in tiers], dtype=float)
    x = np.arange(len(tiers), dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.2), constrained_layout=True)
    fig.suptitle(f"Finite-beta QA electron-root convergence ladder: {summary['status']}", fontweight="bold")

    for lane, marker in (("cpu", "o"), ("gpu", "s"), ("fortran_v3", "^")):
        roots: list[float] = []
        present_x: list[float] = []
        for i, tier in enumerate(tiers):
            lanes = tier["lanes"]
            if lane in lanes:
                present_x.append(x[i])
                roots.append(float(lanes[lane]["selected_root_er"]))
        if roots:
            axes[0, 0].plot(present_x, roots, marker=marker, label=lane)
    axes[0, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 0].set_ylabel(r"selected electron-root $E_r$")
    axes[0, 0].set_title("A. Root location by backend")
    axes[0, 0].legend()

    drift = [float(tier["convergence_gate"]["root_drift_from_previous"]) for tier in tiers]
    drift_limit = float(summary["tolerances"]["root_drift_atol"])  # type: ignore[index]
    axes[0, 1].bar(x, drift, color="#4477aa")
    axes[0, 1].axhline(drift_limit, color="#cc3311", linestyle="--", label="gate")
    axes[0, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[0, 1].set_yscale("symlog", linthresh=1.0e-12)
    axes[0, 1].set_ylabel(r"$|\Delta E_r|$ vs previous tier")
    axes[0, 1].set_title("B. Resolution drift")
    axes[0, 1].legend()

    max_backend = [
        max([0.0, *(float(v) for v in tier["backend_root_diffs"].values())])
        for tier in tiers
    ]
    backend_limit = float(summary["tolerances"]["backend_root_atol"])  # type: ignore[index]
    axes[1, 0].bar(x, max_backend, color="#228833")
    axes[1, 0].axhline(backend_limit, color="#cc3311", linestyle="--", label="gate")
    axes[1, 0].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 0].set_yscale("symlog", linthresh=1.0e-14)
    axes[1, 0].set_ylabel(r"max backend $|\Delta E_r|$")
    axes[1, 0].set_title("C. Backend/reference agreement")
    axes[1, 0].legend()

    dense_gib = np.asarray([float(tier["dense_matrix_gib"]) for tier in tiers], dtype=float)
    axes[1, 1].bar(x, dense_gib, color="#aa3377", label="dense matrix")
    axes[1, 1].set_xticks(x, names, rotation=15, ha="right")
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_ylabel("estimated dense matrix GiB")
    axes[1, 1].set_title("D. Cost growth")
    for i, tier in enumerate(tiers):
        axes[1, 1].text(
            x[i],
            dense_gib[i],
            f"N={int(active[i])}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    note = "; ".join(str(item) for item in summary.get("blockers", [])) or "all gates passed"
    fig.text(0.01, 0.01, note, fontsize=8.5)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    config = load_ladder_config(args.config)
    summary = evaluate_promotion_ladder(
        config,
        base_dir=args.config.parent,
        backend_root_atol=float(args.backend_root_atol),
        root_drift_atol=float(args.root_drift_atol),
    )
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / f"{args.stem}.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    png, pdf = _plot(summary, out_dir=out_dir, stem=args.stem)
    print("finite-beta electron-root ladder summary")
    print(f"  status:  {summary['status']}")
    print(f"  summary: {summary_path}")
    print(f"  figure:  {png}")
    print(f"  pdf:     {pdf}")
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["status"] in {"pass", "deferred"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
