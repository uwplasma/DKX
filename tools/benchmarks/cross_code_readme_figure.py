"""Generate the README cross-code validation figure from the sealed artifacts.

Nothing here is typed in by hand: every number is read at run time out of the
checked-in validation artifacts under ``validation/``, so the figure cannot
drift away from the evidence that gates the release. If an artifact moves, this
script fails rather than drawing a stale number.

The figure answers the question a reader actually has -- "is this thing right?"
-- with the two different answers it deserves, because the two comparisons mean
different things:

Left panel, DKX against SFINCS Fortran v3. Same equations, same discretization,
matched decks, so the only thing being measured is whether the JAX
reimplementation reproduces the Fortran arithmetic. Agreement is at 1e-10 to
1e-8 in scaled error -- the level where the two codes differ by solver
tolerance and floating-point summation order, not by physics. The W7-X row is
the one exception and is the strictest kind of test in the set: it converts to
physical flux units through the native path, where 0.3% is the honest number.

Right panel, DKX against MONKES and YANCC. Independent codes with independent
discretizations solving the same monoenergetic equation, compared through the
Beidler normalization. Percent-level agreement is what a cross-code comparison
of this kind produces; the 6% band is the release gate.

The scope line under the right panel is not decoration. The cross-code rung is
bounded to matched zero-field monoenergetic PAS/DKES equations, and the artifact
records five explicit exclusions. A figure that showed the agreement without the
bound would be claiming more than the evidence supports.

Run:
  python tools/benchmarks/cross_code_readme_figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATION = REPO_ROOT / "validation"
OUT_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "readme"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#2e9e6b"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BAND = "#f2c14e"

# Left panel: the four SFINCS Fortran v3 comparisons, in the order a reader
# should meet them -- simplest physics first, hardest last.
SFINCS_ARTIFACTS = (
    ("full_kinetic_sfincs_v1.json", "Tokamak\nfull Fokker-Planck, $E_r=0$"),
    ("full_kinetic_sfincs_finite_er_v1.json", "Tokamak\nfull Fokker-Planck, finite $E_r$"),
    ("full_kinetic_sfincs_stellarator_v1.json", "Stellarator\nfull Fokker-Planck, $E_r=0$"),
    ("native_physical_flux_sfincs_v1.json", "W7-X\nphysical flux units"),
)

COEFFS = ("D11_star", "D31_star", "D13_star", "D33_star")
COEFF_LABEL = {
    "D11_star": r"$D_{11}^{*}$",
    "D31_star": r"$D_{31}^{*}$",
    "D13_star": r"$D_{13}^{*}$",
    "D33_star": r"$D_{33}^{*}$",
}


def _load(name: str) -> dict:
    path = VALIDATION / name
    if not path.is_file():
        raise SystemExit(f"missing validation artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _sfincs_rows() -> list[tuple[str, float, float]]:
    """(label, measured scaled error, gate) for each SFINCS comparison."""
    rows = []
    for name, label in SFINCS_ARTIFACTS:
        acc = _load(name)["acceptance"]
        measured = acc["measured_max_cross_code_scaled_error"]
        gate = acc["max_cross_code_scaled_error"]
        rows.append((label, float(measured), float(gate)))
    return rows


def _cross_code() -> tuple[list[dict], float, list[str]]:
    doc = _load("independent_cross_code_v1.json")
    tol = float(doc["acceptance"]["coefficient_relative_tolerance"])
    cases = [
        {
            "id": case["id"],
            "reference": case["reference"]["code"],
            "family": case["family"],
            "nu_prime": float(case["normalization"]["nu_prime"]),
            "error": case["comparison"]["relative_error"],
        }
        for case in doc["cases"]
    ]
    return cases, tol, list(doc["exclusions"])


def main() -> None:
    sfincs = _sfincs_rows()
    cases, tol, _exclusions = _cross_code()

    fig = plt.figure(figsize=(13.0, 6.1))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.05), wspace=0.30,
                            left=0.185, right=0.985, top=0.815, bottom=0.275)

    # ---------------- left: agreement with SFINCS Fortran v3 ---------------
    ax = fig.add_subplot(grid[0, 0])
    labels = [row[0] for row in sfincs]
    measured = np.array([row[1] for row in sfincs])
    gates = np.array([row[2] for row in sfincs])
    y = np.arange(len(labels))[::-1]

    ax.barh(y, measured, height=0.55, color=BLUE, zorder=3)
    for yi, gate in zip(y, gates, strict=True):
        ax.plot([gate, gate], [yi - 0.34, yi + 0.34], color=ORANGE, lw=2.2,
                solid_capstyle="butt", zorder=4)
    for yi, value in zip(y, measured, strict=True):
        ax.text(value * 1.5, yi, f"{value:.1e}", va="center", ha="left",
                fontsize=9.5, color=INK, zorder=5)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    ax.set_xscale("log")
    ax.set_xlim(1e-11, 8e-1)
    ax.set_xlabel("max scaled difference from SFINCS Fortran v3", fontsize=10)
    ax.set_title("Same equations: agreement is at solver tolerance",
                 fontsize=11.5, color=INK, pad=10)
    ax.xaxis.grid(True, color=GRID, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK_2)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], color=ORANGE, lw=2.2, label="release gate")
    ax.bar(0, 0, color=BLUE, label="measured")
    ax.legend(frameon=False, fontsize=9, loc="upper center",
              bbox_to_anchor=(0.5, -0.175), ncol=2)

    # ---------------- right: agreement with MONKES and YANCC ---------------
    ax2 = fig.add_subplot(grid[0, 1])
    n_case, n_coef = len(cases), len(COEFFS)
    width = 0.2
    base = np.arange(n_case)
    palette = (BLUE, ORANGE, GREEN, "#8a5fd6")

    ax2.set_xlim(-0.62, n_case - 0.38)
    ax2.axhspan(tol * 100.0, 45.0, color="#d94f4f", alpha=0.09, zorder=0)
    # Data coordinates rather than axhline so the rule ends exactly at the
    # axis box, level with the shaded region it bounds.
    ax2.plot([-0.62, n_case - 0.38], [tol * 100.0] * 2, color="#d94f4f",
             lw=1.5, ls="--", zorder=1)
    ax2.text(n_case - 0.5, tol * 100.0 * 1.18, f"{tol * 100:.0f}% release gate",
             fontsize=9, color="#a83232", ha="right", va="bottom", zorder=5)

    for j, coeff in enumerate(COEFFS):
        offs = (j - (n_coef - 1) / 2.0) * width
        vals = [max(case["error"][coeff] * 100.0, 1e-4) for case in cases]
        ax2.bar(base + offs, vals, width=width * 0.92, color=palette[j],
                label=COEFF_LABEL[coeff], zorder=3)

    ax2.set_xticks(base)
    ax2.set_xticklabels(
        [f"{case['id'].split('_')[0].upper()}\nvs {case['reference']}"
         f"\n$\\nu'={case['nu_prime']:.3g}$" for case in cases],
        fontsize=9.5,
    )
    ax2.set_yscale("log")
    ax2.set_ylim(1e-4, 45.0)
    ax2.set_ylabel("relative difference (%)", fontsize=10)
    ax2.set_title("Independent codes: agreement is at the percent level",
                  fontsize=11.5, color=INK, pad=10)
    ax2.yaxis.grid(True, color=GRID, zorder=0)
    ax2.set_axisbelow(True)
    for side in ("top", "right"):
        ax2.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax2.spines[side].set_color(INK_2)
    ax2.legend(frameon=False, fontsize=9.5, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.175), columnspacing=2.4)

    fig.suptitle("DKX against SFINCS Fortran v3, MONKES and YANCC",
                 fontsize=13.5, color=INK, x=0.03, ha="left", y=0.955)
    fig.text(0.03, 0.893,
             "every value read from the sealed artifacts in validation/ that gate the release",
             fontsize=9.5, color=INK_2, ha="left")
    fig.text(0.03, 0.022,
             "Left: matched decks, full Fokker-Planck collisions. Right: matched zero-field "
             "monoenergetic PAS/DKES equations in the Beidler normalization -- the cross-code "
             "rung is bounded to those\nequations and is not a finite-$E_r$, ambipolar-profile, "
             "experimental, or performance comparison.",
             fontsize=8.6, color=INK_2, ha="left")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "cross_code_validation.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
