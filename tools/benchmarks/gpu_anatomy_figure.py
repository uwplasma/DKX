"""Render the RTX A4000 GPU anatomy / memory-headroom figure for the docs.

Two panels, both measured on one office host (RTX A4000 16 GB, 12.56 GB usable
device budget, 36-core CPU), warm best-of-N ``block_tridiagonal_truncated``
solves of HSX-family decks:

* left  -- single-GPU memory ladder: device peak (GB) versus unknown count,
  against the 12.56 GB device budget, showing that the truncated tier-1
  working set keeps a 2.525M-unknown solve at 2.21 GB;
* right -- 36-core CPU single-solve thread scaling on the mid deck (336,610
  unknowns): warm solve versus pinned core count, with the 8-core optimum and
  the inversion past it.

The measured values are transcribed from the GPU-campaign JSON records (the
memory ladder in ``docs/performance.rst`` and the CPU thread-scaling prose in
``docs/parallelism.rst``); this generator embeds them so the figure is
reproducible from the repository. Regenerate with::

    python tools/benchmarks/gpu_anatomy_figure.py

Output: ``docs/_static/figures/gpu_anatomy_memory.png`` (palette-quantized).
"""

from __future__ import annotations

import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT = REPO_ROOT / "docs" / "_static" / "figures" / "gpu_anatomy_memory.png"

DEVICE_BUDGET_GB = 12.56

# Single RTX A4000 memory ladder: (unknowns, device peak [GB], warm solve [s]).
LADDER = (
    (336_610, 0.18, 3.7),
    (1_275_010, 0.61, 25.6),
    (2_025_010, 1.42, 85.3),
    (2_525_010, 2.21, 157.6),
)

# Office 36-core CPU thread scaling, mid deck (336,610 unknowns): (cores, warm [s]).
CPU_SCALING = (
    (1, 9.69),
    (2, 7.80),
    (4, 5.57),
    (8, 4.87),
    (16, 12.25),
    (32, 56.64),
    (36, 29.35),
)

INK = "#1b2733"
ACCENT = "#2f6f8f"
BUDGET = "#b0552d"
OPTIMUM = "#2f8f5b"


def render() -> None:
    fig, (ax_mem, ax_cpu) = plt.subplots(1, 2, figsize=(9.0, 3.5))
    fig.patch.set_facecolor("white")

    # --- Left: GPU memory ladder ---
    dofs = [row[0] / 1e6 for row in LADDER]
    peak = [row[1] for row in LADDER]
    ax_mem.plot(dofs, peak, marker="o", color=ACCENT, linewidth=2.0, zorder=3)
    ax_mem.axhline(
        DEVICE_BUDGET_GB, color=BUDGET, linestyle="--", linewidth=1.4, zorder=2
    )
    ax_mem.text(
        dofs[0],
        DEVICE_BUDGET_GB,
        " 12.56 GB device budget",
        color=BUDGET,
        va="bottom",
        ha="left",
        fontsize=8.5,
    )
    ax_mem.annotate(
        "2.53M unknowns in 2.21 GB\n(full-band charge ~208 GB)",
        xy=(dofs[-1], peak[-1]),
        xytext=(dofs[0] + 0.05, 4.6),
        color=INK,
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.0},
    )
    ax_mem.set_ylim(0, 14)
    ax_mem.set_xlabel("Unknowns [millions]")
    ax_mem.set_ylabel("Device peak in use [GB]")
    ax_mem.set_title("GPU memory ladder (RTX A4000)", fontsize=10, color=INK)

    # --- Right: CPU thread scaling ---
    cores = [row[0] for row in CPU_SCALING]
    warm = [row[1] for row in CPU_SCALING]
    ax_cpu.plot(cores, warm, marker="o", color=ACCENT, linewidth=2.0, zorder=3)
    opt_cores, opt_warm = 8, 4.87
    ax_cpu.scatter([opt_cores], [opt_warm], color=OPTIMUM, s=70, zorder=4)
    ax_cpu.annotate(
        "8-core optimum\n4.87 s (1.99x, 25% eff)",
        xy=(opt_cores, opt_warm),
        xytext=(9, 26),
        color=OPTIMUM,
        fontsize=8.5,
        arrowprops={"arrowstyle": "->", "color": OPTIMUM, "linewidth": 1.0},
    )
    ax_cpu.set_xscale("log", base=2)
    # 32 and 36 sit almost on top of each other on a log2 axis; label up to 32
    # and let the full-pool (36-core) marker plot without a colliding tick.
    tick_cores = [c for c in cores if c != 36]
    ax_cpu.set_xticks(tick_cores)
    ax_cpu.set_xticklabels([str(c) for c in tick_cores])
    ax_cpu.minorticks_off()
    ax_cpu.set_ylim(0, 60)
    ax_cpu.set_xlabel("Pinned CPU cores")
    ax_cpu.set_ylabel("Warm solve [s]")
    ax_cpu.set_title(
        "CPU single-solve scaling (mid deck, 36-core)", fontsize=10, color=INK
    )

    for ax in (ax_mem, ax_cpu):
        ax.grid(True, color="#dbe1e6", linewidth=0.7)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()

    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, facecolor="white")
    plt.close(fig)
    buffer.seek(0)

    image = Image.open(buffer).convert("RGB")
    quantized = image.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    quantized.save(OUTPUT, format="PNG", optimize=True)

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({size_kb:.1f} KB)")


if __name__ == "__main__":
    render()
