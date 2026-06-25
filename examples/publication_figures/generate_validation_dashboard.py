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

from sfincs_jax.validation.artifacts import (
    DEFAULT_PUBLICATION_ARTIFACTS,
    build_publication_validation_summary,
    load_collisionality_records,
    load_er_sweep_records,
    l11_abs_series,
)


DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_STEM = "sfincs_jax_publication_validation_dashboard"


def _setup_mpl() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 11.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linestyle": "-",
            "lines.linewidth": 2.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="generate_validation_dashboard",
        description="Generate the publication validation dashboard from checked-in sfincs_jax artifacts.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR,
        help="Directory containing publication summary artifacts.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for publication-style figures.",
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}_summary.json",
        help="Machine-readable dashboard summary path.",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM, help="Figure stem without extension.")
    return parser


def _plot_collisionality_panel(ax, records, *, title: str) -> None:
    colors = {"Fokker-Planck": "#0f4c81", "PAS": "#d95f02"}
    markers = {"Fokker-Planck": "o", "PAS": "s"}
    for label in ("Fokker-Planck", "PAS"):
        nuprime, l11 = l11_abs_series(records, label=label)
        ax.loglog(
            nuprime,
            l11,
            marker=markers[label],
            color=colors[label],
            label=label,
        )
    ax.set_title(title)
    ax.set_xlabel(r"normalized collisionality $\nu'$")
    ax.set_ylabel(r"$|L_{11}|$")
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="best")


def _plot_er_panel(ax, records, *, field: str, title: str, ylabel: str) -> None:
    colors = {
        "dkes": "#0f4c81",
        "partial": "#1b9e77",
        "full": "#d95f02",
    }
    markers = {"dkes": "o", "partial": "^", "full": "s"}
    by_model: dict[str, list[object]] = {}
    for record in records:
        by_model.setdefault(record.model, []).append(record)
    for model in ("dkes", "partial", "full"):
        model_records = sorted(by_model[model], key=lambda record: record.er)
        x = np.asarray([record.er for record in model_records], dtype=float)
        y = np.asarray([float(getattr(record, field)) for record in model_records], dtype=float)
        ax.plot(
            x,
            y,
            marker=markers[model],
            color=colors[model],
            label=model_records[0].label.replace(" trajectories", ""),
        )
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.55)
    ax.set_title(title)
    ax.set_xlabel(r"$E_r$")
    ax.set_ylabel(ylabel)
    ax.grid(True, which="both", alpha=0.24)
    ax.legend(loc="best")


def write_dashboard_summary(*, artifact_dir: Path, summary_json: Path) -> dict[str, object]:
    payload = build_publication_validation_summary(artifact_dir=Path(artifact_dir))
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def plot_dashboard(*, artifact_dir: Path, out_dir: Path, stem: str) -> None:
    artifact_dir = Path(artifact_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lhd = load_collisionality_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["w7x_collisionality"])
    tokamak = load_er_sweep_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["tokamak_er_sweep"])
    stellarator = load_er_sweep_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["stellarator_er_sweep"])

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.4), constrained_layout=True)
    _plot_collisionality_panel(axes[0, 0], lhd, title="A. LHD collision-operator scan")
    _plot_collisionality_panel(axes[0, 1], w7x, title="B. W7-X collision-operator scan")
    _plot_er_panel(
        axes[1, 0],
        tokamak,
        field="fsab_flow",
        title="C. Tokamak-like trajectory sweep",
        ylabel=r"$\langle B u_\parallel \rangle$",
    )
    _plot_er_panel(
        axes[1, 1],
        stellarator,
        field="fsab_jhat",
        title="D. Stellarator-like trajectory sweep",
        ylabel=r"$\langle J\cdot B\rangle$",
    )
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.out_dir)
    summary_json = Path(args.summary_json)
    stem = str(args.stem)
    write_dashboard_summary(artifact_dir=artifact_dir, summary_json=summary_json)
    plot_dashboard(artifact_dir=artifact_dir, out_dir=out_dir, stem=stem)
    print(f"Wrote dashboard summary to {summary_json}")
    print(f"Wrote dashboard figures to {out_dir / (stem + '.png')} and {out_dir / (stem + '.pdf')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
