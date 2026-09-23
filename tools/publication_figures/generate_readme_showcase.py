"""The two README showcase figures, drawn from recorded measurements.

Every number below is copied from the experiment record named beside it; the
script runs no solve, so it regenerates the figures in seconds on any machine::

    python tools/publication_figures/generate_readme_showcase.py

``sfincs_reference_limits.png``
    Left: the HSX bootstrap-current gap between released SFINCS v3 and DKX, and
    what is left of it once SFINCS's ``1d-12`` matrix-entry cutoff is set to
    zero (docs/experiments/2026-09-13-sfincs-sparsify-threshold.md).  Right:
    the HSX-like gap deck, 633,604 unknowns, where every SFINCS route and the
    DKX Krylov route stop short of the requested tolerance on a 36 GiB host
    (docs/experiments/2026-09-19-sfincs-on-the-gap-deck.md).

``factor_reuse.png``
    Wall time of one factorization serving many solves, on the structured and
    sparse direct routes (docs/experiments/2026-09-20-one-factorization-many-solves.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "readme"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
DKX = "#2a78d6"
SFINCS = "#eb6834"
MUTED = "#a8a69f"

# --- 2026-09-13-sfincs-sparsify-threshold.md --------------------------------
# Bootstrap current FSABjHat, relative difference to DKX (the follow-up table:
# released SFINCS reproduces the refined reference; cutoff 0 agrees with DKX).
THRESHOLD_POINTS = ("point A\nrN = 0.187", "point B\nrN = 0.367")
RELEASED_GAP = (0.19, 0.12)  # |J_SFINCS - J_DKX| / |J_SFINCS|, "12-19%"
CUTOFF_ZERO_GAP = (6.6e-11, 1.4e-11)  # "Relative to DKX" column

# --- 2026-09-19-sfincs-on-the-gap-deck.md ------------------------------------
# Relative residual ||r||/||b|| each route reached, or None when the run was
# killed for memory before returning one.  Tolerance requested: 1e-10.
GAP_DECK_ROUTES = (
    ("SFINCS direct (MUMPS)", None, "killed for memory, 22.5 min"),
    ("SFINCS GMRES, preconditioner_x = 1", 0.9955, "0.9955: stagnant, 66 iterations, 60 min"),
    ("SFINCS preconditioner_x = 2", None, "killed for memory, 30.6 min"),
    ("DKX recycled Krylov, coarse", 2.5e-5, "2.5e-5 after 20,000 iterations"),
)
GAP_DECK_TOLERANCE = 1e-10

# --- 2026-09-20-one-factorization-many-solves.md ----------------------------
# Median seconds over nine interleaved repeats, and factorizations counted.
REUSE_ARMS = (
    "one RHS",
    "3 RHS,\none call",
    "3 RHS,\n3 calls",
    "3 RHS, 3 calls,\nstored factors",
    "primal + adjoint,\nstored factors",
)
REUSE_ROUTES = {
    "structured direct, 16,230 unknowns": ((0.3355, 0.3279, 0.9593, 0.2822, 0.4406), (1, 1, 3, 0, 1)),
    "sparse direct, 1,962 unknowns": ((0.3557, 0.3540, 1.1931, 0.2779, 0.4090), (1, 1, 3, 0, 1)),
}


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(axis="x" if ax.get_label() == "h" else "y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _save(fig, name: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.savefig(path, dpi=110, facecolor=SURFACE, metadata={"Software": None})
    plt.close(fig)
    return path


def reference_limits(out_dir: Path) -> Path:
    fig, (left, right) = plt.subplots(
        1, 2, figsize=(10.0, 3.8), gridspec_kw={"width_ratios": [1.0, 1.35]}
    )
    fig.patch.set_facecolor(SURFACE)

    _style(left)
    xs = range(len(THRESHOLD_POINTS))
    width = 0.36
    left.bar([x - width / 2 for x in xs], RELEASED_GAP, width - 0.03, color=SFINCS,
             label="SFINCS as released (cutoff 1e-12)")
    left.bar([x + width / 2 for x in xs], CUTOFF_ZERO_GAP, width - 0.03, color=DKX,
             label="SFINCS with cutoff 0")
    for x, value in zip(xs, RELEASED_GAP):
        left.annotate(f"{value:.0%}", (x - width / 2, value), ha="center", va="bottom",
                      fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")
    for x, value in zip(xs, CUTOFF_ZERO_GAP):
        left.annotate(f"{value:.1e}", (x + width / 2, value), ha="center", va="bottom",
                      fontsize=8, color=INK, xytext=(0, 2), textcoords="offset points")
    left.set_yscale("log")
    left.set_ylim(1e-12, 3.0)
    left.set_xticks(list(xs), THRESHOLD_POINTS)
    left.set_ylabel("bootstrap current,\nrelative difference to DKX", fontsize=8, color=INK_2)
    left.set_title("HSX: a matrix-entry cutoff changes the answer", fontsize=9, color=INK, loc="left")
    left.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=1, labelcolor=INK_2)

    right.set_label("h")
    _style(right)
    labels = [route for route, _, _ in GAP_DECK_ROUTES]
    ys = list(range(len(GAP_DECK_ROUTES)))[::-1]
    for y, (route, residual, note) in zip(ys, GAP_DECK_ROUTES):
        colour = DKX if route.startswith("DKX") else SFINCS
        if residual is None:
            right.scatter([1.0], [y], marker="x", s=40, color=colour, linewidths=2)
            right.annotate(note, (1.0, y), xytext=(-8, 0), textcoords="offset points",
                           ha="right", va="center", fontsize=7.5, color=INK_2)
        else:
            right.barh(y, residual, height=0.5, color=colour, left=GAP_DECK_TOLERANCE)
            inside = residual > 1e-2
            right.annotate(note, (residual, y), xytext=(-4 if inside else 4, 0),
                           textcoords="offset points", ha="right" if inside else "left",
                           va="center", fontsize=7.5, color=SURFACE if inside else INK_2)
    right.axvline(GAP_DECK_TOLERANCE, color=INK, linewidth=1.0, linestyle="--")
    right.annotate("requested 1e-10", (GAP_DECK_TOLERANCE, -0.45), xytext=(3, 0),
                   textcoords="offset points", fontsize=7.5, color=INK)
    right.set_xscale("log")
    right.set_ylim(-0.6, len(ys) - 0.5)
    right.set_xlim(3e-11, 30.0)
    right.set_yticks(ys, labels)
    right.set_xlabel("relative residual reached, ||r|| / ||b||", fontsize=8, color=INK_2)
    right.set_title("HSX-like gap deck, 633,604 unknowns: open in both codes",
                    fontsize=9, color=INK, loc="left")
    fig.tight_layout()
    return _save(fig, "sfincs_reference_limits.png", out_dir)


def factor_reuse(out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (route, (seconds, factorizations)) in zip(axes, REUSE_ROUTES.items()):
        _style(ax)
        xs = range(len(REUSE_ARMS))
        colours = [DKX if count <= 1 else MUTED for count in factorizations]
        ax.bar(list(xs), seconds, 0.62, color=colours)
        for x, value, count in zip(xs, seconds, factorizations):
            word = "factorization" if count == 1 else "factorizations"
            ax.annotate(f"{value:.2f} s\n{count} {word}", (x, value), ha="center",
                        va="bottom", fontsize=7, color=INK, xytext=(0, 2),
                        textcoords="offset points")
        ax.set_xticks(list(xs), REUSE_ARMS, fontsize=7)
        ax.set_ylim(0, 1.55)
        ax.set_title(route, fontsize=9, color=INK, loc="left")
    axes[0].set_ylabel("wall time, median of 9 (s)", fontsize=8, color=INK_2)
    fig.tight_layout()
    return _save(fig, "factor_reuse.png", out_dir)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args(argv)
    for path in (reference_limits(args.out_dir), factor_reuse(args.out_dir)):
        print(f"{path.name}: {path.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
