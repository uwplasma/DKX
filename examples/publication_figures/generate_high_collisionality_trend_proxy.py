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

from sfincs_jax.validation_artifacts import (
    DEFAULT_PUBLICATION_ARTIFACTS,
    TRANSPORT_ELEMENTS,
    build_high_collisionality_trend_proxy_summary,
    collisionality_power_law_slope,
    load_collisionality_records,
    transport_element_abs_series,
)


DEFAULT_ARTIFACT_DIR = _REPO_ROOT / "examples" / "publication_figures" / "artifacts"
DEFAULT_OUT_DIR = _REPO_ROOT / "docs" / "_static" / "figures" / "paper"
DEFAULT_STEM = "sfincs_jax_high_collisionality_trend_proxy"


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
        prog="generate_high_collisionality_trend_proxy",
        description="Generate a high-collisionality trend proxy from checked-in collisionality artifacts.",
    )
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / f"{DEFAULT_STEM}_summary.json",
    )
    parser.add_argument("--stem", default=DEFAULT_STEM)
    parser.add_argument("--n-fit", type=int, default=3, help="Number of high-nu points used for slope fits.")
    return parser


def _plot_case_element(ax, records, *, case_label: str, element_name: str, n_fit: int) -> None:
    element = TRANSPORT_ELEMENTS[element_name]
    colors = {"Fokker-Planck": "#0f4c81", "PAS": "#d95f02"}
    markers = {"Fokker-Planck": "o", "PAS": "s"}
    slope_lines: list[str] = []
    for label in ("Fokker-Planck", "PAS"):
        nuprime, values = transport_element_abs_series(records, label=label, element=element)
        slope = collisionality_power_law_slope(records, label=label, element=element, n_fit=n_fit)
        ax.loglog(nuprime, values, marker=markers[label], color=colors[label], label=label)
        tail_nu = nuprime[-int(n_fit) :]
        tail_values = values[-int(n_fit) :]
        ref = tail_values[-1] * (tail_nu / tail_nu[-1]) ** slope
        ax.loglog(tail_nu, ref, linestyle="--", color=colors[label], alpha=0.72)
        slope_lines.append(f"{label}: slope {slope:+.2f}")
    ax.set_title(f"{case_label} {element_name}")
    ax.set_xlabel(r"normalized collisionality $\nu'$")
    ax.set_ylabel(rf"$|{element_name}|$")
    ax.text(
        0.04,
        0.05,
        "\n".join(slope_lines),
        transform=ax.transAxes,
        fontsize=8.0,
        bbox={"facecolor": "white", "alpha": 0.76, "edgecolor": "none"},
    )
    ax.grid(True, which="both", alpha=0.24)


def write_trend_summary(*, artifact_dir: Path, summary_json: Path, n_fit: int) -> dict[str, object]:
    payload = build_high_collisionality_trend_proxy_summary(
        artifact_dir=Path(artifact_dir),
        n_fit=int(n_fit),
    )
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def plot_trend_proxy(*, artifact_dir: Path, out_dir: Path, stem: str, n_fit: int) -> None:
    artifact_dir = Path(artifact_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lhd = load_collisionality_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["lhd_collisionality"])
    w7x = load_collisionality_records(artifact_dir / DEFAULT_PUBLICATION_ARTIFACTS["w7x_collisionality"])

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.4), constrained_layout=True)
    _plot_case_element(axes[0, 0], lhd, case_label="LHD", element_name="L11", n_fit=n_fit)
    _plot_case_element(axes[0, 1], lhd, case_label="LHD", element_name="L12", n_fit=n_fit)
    _plot_case_element(axes[1, 0], w7x, case_label="W7-X", element_name="L11", n_fit=n_fit)
    _plot_case_element(axes[1, 1], w7x, case_label="W7-X", element_name="L12", n_fit=n_fit)
    axes[0, 0].legend(loc="upper left")
    fig.savefig(out_dir / f"{stem}.png", dpi=180)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    _setup_mpl()
    args = _build_parser().parse_args(argv)
    write_trend_summary(
        artifact_dir=Path(args.artifact_dir),
        summary_json=Path(args.summary_json),
        n_fit=int(args.n_fit),
    )
    plot_trend_proxy(
        artifact_dir=Path(args.artifact_dir),
        out_dir=Path(args.out_dir),
        stem=str(args.stem),
        n_fit=int(args.n_fit),
    )
    print(f"Wrote high-collisionality trend summary to {Path(args.summary_json)}")
    print(f"Wrote high-collisionality trend figures to {Path(args.out_dir) / (str(args.stem) + '.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
