"""What decides whether dkx beats the Fortran reference: the solver route.

Across the upstream suite the outcome is not spread out.  It splits on whether
``dkx`` has a structured *direct* solver for the deck or falls back to
preconditioned Krylov:

===============================  ==================
route                            faster than SFINCS
===============================  ==================
structured direct                9 of 9
recycled Krylov                  7 of 23
===============================  ==================

That is the finding worth plotting.  A win/lose colouring would show 16 of 32
and hide the mechanism; colouring by route shows that the losses are not spread
thinly over the suite but concentrated exactly where the block-tridiagonal
structure in the Legendre index is broken -- by Fokker-Planck collisions, by
tangential magnetic drifts, by the ``E_r`` ``xDot``/``xiDot`` terms, and by the
``Phi1`` Newton iteration.

Memory is the weaker axis and is shown as measured: ``dkx`` is lighter on 3 of
the 32 decks it completed.  Below ~10k unknowns the JAX runtime floor dominates
a Fortran process that runs in 0.1 GB; above ~1M the Krylov preconditioner's
dense ``(Ntheta*Nzeta)`` bands do.  The memory panel shows only completed runs:
a killed process's peak RSS records where it died, not what the solve costs.

Six decks did not complete.  Five were killed by the operating system while the
Krylov preconditioner allocated its bands -- ``dkx.solve`` refuses those up
front instead, naming the fill-reducing route that does fit.  The sixth wanted
the LIBSTELL text form of a VMEC ``wout``, which ``dkx`` could not read at the
time of this sweep and now can.

Data is the upstream example suite run end to end through both codes by
``tools/benchmarks/parity_performance_matrix.py`` -- geometry schemes
1/2/4/5/11, the filtered W7-X netCDF equilibria, pitch-angle and Fokker-Planck
collisions, zero and finite ``Er``, ``Phi1`` on and off, tangential magnetic
drifts, one to three species, 651 to 1.9M unknowns.

The comparison is warm ``dkx`` against the Fortran binary's wall time.  Cold
``dkx`` includes JIT compilation and is the honest number for a single one-shot
solve; warm is the honest number for the scan and optimization workloads the
code exists for.  Both are in the JSONL; the figure states which it uses.

Reproduce (from the repo root):

  python tools/benchmarks/parity_performance_matrix.py \\
      --examples /path/to/sfincs/fortran/version3/examples \\
      --fortran-binary /path/to/sfincs \\
      --fortran-launcher 'micromamba run -n sfincs-fortran' \\
      --fortran-residual --out campaign.jsonl

  python tools/paper_benchmarks/cross_code_matrix.py --results campaign.jsonl

Outputs ``docs/_static/figures/paper_benchmarks/cross_code_matrix.png``.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_PATH = (
    REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
    / "cross_code_matrix.png"
)

DKX_COLOR = "#2b6cb0"
REF_COLOR = "#c05621"
DEAD_COLOR = "#9b2c2c"


def load(path: Path) -> list[dict]:
    """One row per case that the reference completed."""
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        fortran = (record.get("fortran") or {}).get("1", {}) or {}
        if not fortran.get("succeeded"):
            continue
        dkx = record.get("dkx") or {}
        parity = [v for v in (record.get("parity") or {}).values() if isinstance(v, float)]
        method = str(dkx.get("method") or "")
        rows.append(
            {
                "case": record["case"],
                "dof": int(record.get("dof") or 0),
                "dkx_s": dkx.get("warm_s"),
                "ref_s": fortran.get("wall_s"),
                "dkx_gb": dkx.get("peak_rss_gb"),
                "ref_gb": fortran.get("peak_rss_gb"),
                "parity": max(parity) if parity else None,
                "finished": bool(dkx.get("warm_s")),
                "direct": method.startswith("block_tridiagonal"),
                "method": method,
            }
        )
    return sorted(rows, key=lambda r: r["dof"])


def plot(rows: list[dict], path: Path) -> None:
    """Speed, memory and completion against problem size."""
    fig, (speed, memory) = plt.subplots(
        1, 2, figsize=(12.4, 5.0), constrained_layout=True
    )

    done = [r for r in rows if r["finished"]]
    dead = [r for r in rows if not r["finished"]]

    # --- left: speed-up, coloured by which solver dkx could use -------------
    direct = [r for r in done if r["direct"]]
    krylov = [r for r in done if not r["direct"]]
    d_win = sum(1 for r in direct if r["ref_s"] / r["dkx_s"] >= 1.0)
    k_win = sum(1 for r in krylov if r["ref_s"] / r["dkx_s"] >= 1.0)

    speed.axhspan(1.0, 40.0, color=DKX_COLOR, alpha=0.05, zorder=0)
    speed.axhline(1.0, color="0.5", lw=1.0, ls=(0, (5, 4)), zorder=1)
    for group, color, marker, label in (
        (direct, DKX_COLOR, "o",
         f"structured direct — faster on {d_win} of {len(direct)}"),
        (krylov, REF_COLOR, "s",
         f"recycled Krylov — faster on {k_win} of {len(krylov)}"),
    ):
        speed.scatter([r["dof"] for r in group],
                      [r["ref_s"] / r["dkx_s"] for r in group],
                      s=62, color=color, marker=marker, zorder=3,
                      edgecolor="white", linewidth=0.9, label=label)

    floor_y = 0.02
    if dead:
        speed.scatter([r["dof"] for r in dead], [floor_y] * len(dead), s=110,
                      color=DEAD_COLOR, marker="X", zorder=4,
                      edgecolor="white", linewidth=0.8,
                      label=f"dkx did not complete ({len(dead)})")
    speed.set_xscale("log")
    speed.set_yscale("log")
    speed.set_ylim(floor_y / 1.8, 40.0)
    speed.set_xlabel("unknowns")
    speed.set_ylabel("speed-up   (SFINCS wall time / dkx warm solve)")
    speed.set_title(
        "The solver route decides the outcome, not the problem size",
        fontsize=11.5,
    )
    speed.grid(True, which="major", alpha=0.22, lw=0.6)
    speed.legend(fontsize=8, loc="upper right", framealpha=0.96)
    speed.annotate("dkx faster above this line", (1.1e3, 1.25), fontsize=8,
                   color="0.4")

    # --- right: peak memory, both codes ------------------------------------
    all_rows = [r for r in rows if r["finished"] and r["dkx_gb"] and r["ref_gb"]]
    d = np.array([r["dof"] for r in all_rows], dtype=float)
    memory.plot(d, [r["ref_gb"] for r in all_rows], "s", ms=6, color=REF_COLOR,
                mec="white", mew=0.8, label="SFINCS Fortran v3", zorder=3)
    memory.plot(d, [r["dkx_gb"] for r in all_rows], "o", ms=6, color=DKX_COLOR,
                mec="white", mew=0.8, label="dkx", zorder=3)
    lighter = sum(1 for r in all_rows if r["dkx_gb"] < r["ref_gb"])
    floor = min(r["dkx_gb"] for r in all_rows)
    memory.axhline(floor, color=DKX_COLOR, lw=1.0, ls=(0, (2, 3)), alpha=0.7, zorder=1)
    memory.annotate(
        f"JAX runtime floor, ~{floor:.1f} GB:\npaid on every solve, however small",
        (1.05e3, floor * 1.12), fontsize=8, color=DKX_COLOR, va="bottom",
    )
    memory.set_xscale("log")
    memory.set_yscale("log")
    memory.set_xlabel("unknowns")
    memory.set_ylabel("peak resident memory [GB]")
    memory.set_title(
        f"Memory is the weak axis: lighter on {lighter} of the {len(all_rows)} "
        "decks it completed",
        fontsize=11.5,
    )
    memory.grid(True, which="major", alpha=0.22, lw=0.6)
    memory.legend(fontsize=8.5, loc="upper left", framealpha=0.95)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=FIG_PATH)
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no cases with a completed reference in {args.results}")

    done = [r for r in rows if r["finished"]]
    faster = [r for r in done if r["ref_s"] / r["dkx_s"] >= 1.0]
    lighter = [r for r in done if r["dkx_gb"] and r["ref_gb"] and r["dkx_gb"] < r["ref_gb"]]
    parities = [r["parity"] for r in done if r["parity"] is not None]

    print(f"{len(rows)} cases with a completed reference")
    print(f"  dkx completed      {len(done)}")
    print(f"  dkx faster         {len(faster)}/{len(done)}")
    print(f"  dkx lighter        {len(lighter)}/{len(done)}")
    if parities:
        print(f"  median parity      {np.median(parities):.1e}")
        print(f"  worst parity       {max(parities):.1e}")
    best = max(done, key=lambda r: r["ref_s"] / r["dkx_s"])
    worst = min(done, key=lambda r: r["ref_s"] / r["dkx_s"])
    print(f"  best speed-up      {best['ref_s'] / best['dkx_s']:.1f}x  ({best['case']})")
    print(f"  worst speed-up     {worst['ref_s'] / worst['dkx_s']:.2f}x  ({worst['case']})")

    plot(rows, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
