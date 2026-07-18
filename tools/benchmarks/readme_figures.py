"""Generate the README head-to-head and parity figures from measured values.

Every number in this file is a recorded measurement; nothing is estimated.
The two figures land in ``docs/_static/figures/readme/`` and are referenced by
``README.md`` and ``docs/performance.rst``.

Provenance
----------
Runtime/memory bars (``tier1_hsx_runtime_memory.png``):
  ``docs/dev/failure_analysis.md`` sections "Phase 5.1 Fortran strong-scaling
  baseline" and "Phase-4 head-to-head" — ``HSX_PASCollisions_DKESTrajectories``
  at Ntheta=25, Nzeta=51, Nxi=100, Nx=5 (744,610 unknowns), RHSMode=1.
  - dkx tier-1 truncated Legendre elimination (canonical stack), warm
    solve: 27.2 s with the Nxi-for-x ramp discretization (0.93 GB peak RSS) and
    44.3 s with uniform Nxi (1.16 GB), MacBook M4 CPU, JAX x64.
  - dkx on an RTX A4000 GPU: 45.0 s (the L-scan is serial and the A4000
    runs FP64 at 1/32 rate, so GPU ~= M4 CPU for this case).
  - SFINCS Fortran v3 (conda PETSc 3.23 + MUMPS 5.8.2), same deck, same
    machine: 463.6 s / 3.98 GB at 1 MPI rank and 229.5 s / 2.86 GB at 2 ranks
    (its measured parallel floor; 4 and 8 ranks are slower).

Parity chart (``canonical_parity.png``):
  Deletion-pass referee tests that gated each canonical-stack slice:
  - RHSMode=1 output tables vs the Fortran-parity legacy writer: max scaled
    difference 8e-14 (``tests/test_run_rhsmode1.py``).
  - Tier-1 state vectors vs recorded reference state vectors: 1e-11
    (``tests/test_run_transport.py``, frozen Fortran/legacy state fixtures).
  - RHSMode=2/3 transport matrices vs Fortran golden data: 6e-13 .. 9e-9
    depending on case/conditioning (``tests/test_run_transport.py``; the
    scheme-1 monoenergetic [0,1] element is pinned to upstream's expected
    value because that element is tolerance-unstable in the Fortran build
    itself, see ``docs/dev/failure_analysis.md``).

Run:
  python tools/benchmarks/readme_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "readme"

# Palette (colorblind-validated defaults; color follows the code, not the bar).
BLUE = "#2a78d6"  # dkx
ORANGE = "#eb6834"  # SFINCS Fortran v3
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

# --- Measured head-to-head values (see module docstring for provenance) -----
CASE_LABEL = "HSX PAS/DKES, RHSMode=1, 25x51x100x5 (744,610 unknowns)"
ROWS = (
    # label, runtime seconds, peak RSS GB, is_fortran
    ("dkx\nM4, ramp", 27.2, 0.93, False),
    ("dkx\nM4, uniform", 44.3, 1.16, False),
    ("dkx\nA4000 GPU", 45.0, 1.88, False),
    ("Fortran v3\n1 rank", 463.6, 3.98, True),
    ("Fortran v3\n2 ranks", 229.5, 2.86, True),
)

# --- Measured parity envelopes (see module docstring for provenance) --------
PARITY = (
    # label, low, high (equal -> single point), referee
    ("RHSMode=1 output tables\nvs Fortran-parity writer", 8e-14, 8e-14,
     "tests/test_run_rhsmode1.py"),
    ("State vectors (tier 1)\nvs recorded references", 1e-11, 1e-11,
     "tests/test_run_transport.py"),
    ("Transport matrices (RHSMode=2/3)\nvs Fortran golden data", 6e-13, 9e-9,
     "tests/test_run_transport.py"),
)


def _style_axes(ax: plt.Axes) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=8.5)


def runtime_memory_figure(path: Path) -> None:
    labels = [row[0] for row in ROWS]
    runtimes = np.array([row[1] for row in ROWS])
    memories = np.array([row[2] for row in ROWS])
    colors = [ORANGE if row[3] else BLUE for row in ROWS]

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(9.2, 3.4))
    x = np.arange(len(ROWS))
    for ax, values, unit, title in (
        (ax_t, runtimes, "s", "Warm solve wall time"),
        (ax_m, memories, "GB", "Peak process RSS"),
    ):
        ax.bar(x, values, width=0.62, color=colors, zorder=3)
        ax.set_xticks(x, labels, fontsize=7.6, color=INK)
        ax.set_title(title, fontsize=10.5, color=INK)
        ax.set_ylabel(unit, fontsize=9, color=INK_2)
        ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
        _style_axes(ax)
        for xi, value in zip(x, values):
            ax.annotate(
                f"{value:g}", (xi, value), textcoords="offset points",
                xytext=(0, 3), ha="center", fontsize=8, color=INK,
            )
        ax.set_ylim(0, 1.15 * values.max())
    fig.suptitle(CASE_LABEL, fontsize=9.5, color=INK_2, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def parity_figure(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 2.7))
    y = np.arange(len(PARITY))[::-1]
    for yi, (label, low, high, _referee) in zip(y, PARITY):
        if high > low:
            ax.plot([low, high], [yi, yi], color=BLUE, linewidth=3,
                    solid_capstyle="round", zorder=3)
            ax.scatter([low, high], [yi, yi], s=42, color=BLUE, zorder=4)
            text = f"{low:.0e} .. {high:.0e}"
            anchor = np.sqrt(low * high)
        else:
            ax.scatter([low], [yi], s=52, color=BLUE, zorder=4)
            text = f"{low:.0e}"
            anchor = low
        ax.annotate(text, (anchor, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, [row[0] for row in PARITY], fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-15, 1e-6)
    ax.set_ylim(-0.6, len(PARITY) - 0.3)
    ax.set_xlabel("max scaled difference vs reference (pinned by CI referee tests)",
                  fontsize=9, color=INK_2)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)
    _style_axes(ax)
    ax.set_title("Canonical stack vs SFINCS Fortran v3 / recorded references",
                 fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runtime_path = OUT_DIR / "tier1_hsx_runtime_memory.png"
    parity_path = OUT_DIR / "canonical_parity.png"
    runtime_memory_figure(runtime_path)
    parity_figure(parity_path)
    for out in (runtime_path, parity_path):
        size_kb = out.stat().st_size / 1024.0
        print(f"wrote {out.relative_to(REPO_ROOT)} ({size_kb:.1f} kB)")
        if size_kb > 150.0:
            raise SystemExit(f"{out} exceeds the 150 kB README figure budget")
