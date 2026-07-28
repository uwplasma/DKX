"""Cost of a gradient: implicit differentiation against finite differences.

A finite-difference gradient of ``N`` parameters costs ``2N`` converged solves
with central differences.  Implicit differentiation costs one transposed solve
whatever ``N`` is, because the adjoint is defined by the linear equation at the
converged solution rather than by differentiating the iteration.

That is a statement about scaling, and the measurements here are deliberately
at small ``N`` where it is *least* favourable: with two parameters a finite
difference needs four solves, which is the same order as one forward plus one
adjoint, so the wall-time advantage is small.  The advantage is in the slope,
not the intercept, and profile or geometry optimization runs at ``N`` in the
tens.

Accuracy is the other half.  Finite differences have no exact answer to
converge to: the step size trades truncation error against solver noise, and
the reference solve's own residual sets a floor below which the difference is
noise rather than signal.  The per-case step sweep in
``tools/benchmarks/ad_vs_fortran_fd.py`` reports that floor instead of quoting
one lucky step.

Decks are drawn from upstream's example suite and restricted to those whose
*reference* solve is well converged in the true residual (see
``reference_convergence.py``): a finite-difference gradient built from an
under-converged reference measures the reference's noise, not its derivative.

Reproduce (from the repo root), one JSON per deck:

  for D in tokamak_1species_PASCollisions_noEr \\
           tokamak_1species_FPCollisions_withEr_fullTrajectories \\
           tokamak_2species_PASCollisions_noEr \\
           tokamak_2species_PASCollisions_withEr_fullTrajectories; do
    python tools/benchmarks/ad_vs_fortran_fd.py \\
        --deck /path/to/sfincs/fortran/version3/examples/$D \\
        --fortran-binary /path/to/sfincs \\
        --fortran-launcher 'micromamba run -n sfincs-fortran' \\
        --steps 1e-2 1e-3 1e-4 --out ad/$D.json
  done

  python examples/paper_benchmarks/gradient_cost_scaling.py --results ad/

Outputs ``docs/_static/figures/paper_benchmarks/gradient_cost_scaling.png``.
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
    / "gradient_cost_scaling.png"
)

#: How far to project the finite-difference cost line.  Beyond the measured
#: points this is arithmetic on the measured per-solve cost, not a measurement,
#: and the figure says so.
PROJECT_TO = 24


def load(directory: Path) -> list[dict]:
    """One record per ``ad_vs_fortran_fd`` JSON in ``directory``."""
    rows = []
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text())
        summary = report["summary"]
        rows.append(
            {
                "case": report["case"],
                "n": int(summary["n_parameters"]),
                "ad_s": float(summary["dkx_warm_s"]),
                "fd_s": float(summary["fortran_wall_s"]),
                "fd_solves": int(summary["fortran_solves"]),
                "agreement": float(summary["best_agreement_rel"]),
            }
        )
    return sorted(rows, key=lambda r: (r["n"], r["case"]))


def plot(rows: list[dict], path: Path) -> None:
    """Wall time against parameter count, with the finite-difference slope."""
    n = np.array([r["n"] for r in rows], dtype=float)
    ad = np.array([r["ad_s"] for r in rows])
    fd = np.array([r["fd_s"] for r in rows])

    fig, (left, right) = plt.subplots(
        1, 2, figsize=(11.0, 4.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    # Cost per solve, averaged over the measured cases, drives the projection.
    per_solve = float(np.mean(fd / np.array([r["fd_solves"] for r in rows])))
    grid = np.arange(1, PROJECT_TO + 1)
    left.plot(grid, 2.0 * grid * per_solve, color="#c05621", lw=1.4, ls=(0, (5, 4)),
              zorder=1, label=f"finite differences, $2N$ solves (projected at "
                             f"{per_solve:.1f} s/solve)")
    left.axhline(float(np.mean(ad)), color="#2b6cb0", lw=1.4, ls=(0, (5, 4)),
                 zorder=1, label="implicit differentiation, one adjoint (mean)")
    left.scatter(n, fd, s=64, color="#c05621", marker="s", zorder=3,
                 edgecolor="white", linewidth=0.9, label="finite differences, measured")
    left.scatter(n, ad, s=64, color="#2b6cb0", zorder=3,
                 edgecolor="white", linewidth=0.9, label="implicit diff., measured")

    left.axvspan(float(n.max()) + 0.5, PROJECT_TO + 0.5, color="0.94", zorder=0)
    left.annotate("projected", (PROJECT_TO * 0.72, left.get_ylim()[1] * 0.06),
                  fontsize=8.5, color="0.4", ha="center")
    left.set_xlabel("number of parameters $N$")
    left.set_ylabel("wall time for one gradient [s]")
    left.set_xlim(0.5, PROJECT_TO + 0.5)
    left.set_title("Cost scales with $N$ only for finite differences", fontsize=11)
    left.grid(True, alpha=0.22, lw=0.6)
    left.legend(fontsize=8, loc="upper left", framealpha=0.95)

    agreement = np.array([r["agreement"] for r in rows])
    order = np.argsort(agreement)
    labels = [rows[i]["case"].replace("tokamak_", "").replace("Collisions", "")[:30]
              for i in order]
    # Markers, not bars: on a log axis a bar's length carries no meaning.
    positions = np.arange(len(order))
    right.hlines(positions, 1e-11, agreement[order], color="0.85", lw=1.0, zorder=1)
    right.scatter(agreement[order], positions, s=62, color="#2b6cb0", zorder=3,
                  edgecolor="white", linewidth=0.9)
    right.set_yticks(positions)
    right.set_yticklabels(labels, fontsize=8)
    right.set_ylim(-0.6, len(order) - 0.4)
    right.set_xscale("log")
    right.set_xlim(1e-11, 1e-5)
    right.set_xlabel("relative difference,\nimplicit differentiation vs finite differences")
    right.set_title("Agreement across configurations", fontsize=11)
    right.grid(True, axis="x", alpha=0.22, lw=0.6)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, required=True,
                        help="directory of ad_vs_fortran_fd JSON reports")
    parser.add_argument("--out", type=Path, default=FIG_PATH)
    args = parser.parse_args()

    rows = load(args.results)
    if not rows:
        raise SystemExit(f"no ad_vs_fortran_fd JSON reports found in {args.results}")

    print(f"{'case':<52}{'N':>3}{'AD [s]':>9}{'FD [s]':>9}{'solves':>8}{'agreement':>12}")
    for row in rows:
        print(f"{row['case'][:51]:<52}{row['n']:>3}{row['ad_s']:>9.2f}"
              f"{row['fd_s']:>9.2f}{row['fd_solves']:>8}{row['agreement']:>12.2e}")

    worst = max(rows, key=lambda r: r["agreement"])
    print(f"\n{len(rows)} configurations; worst agreement {worst['agreement']:.2e} "
          f"({worst['case']})")
    print("finite differences used exactly 2N solves in every case: "
          + ", ".join(f"{r['fd_solves']}@N={r['n']}" for r in rows))

    plot(rows, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
