"""How reference convergence limits a cross-code comparison.

Two independent codes solving the same discretised system can only be expected
to agree to the accuracy each one actually reaches.  That accuracy is not
always the number in the input file.

SFINCS solves its linear system with PETSc, whose default Krylov convergence
test measures the *preconditioned* residual ``||M^-1 (A x - b)||`` rather than
the true residual ``||A x - b||`` (PETSc manual, ``KSPSetNormType``;
``KSP_NORM_PRECONDITIONED`` is the default for left-preconditioned methods).
SFINCS preconditions with a simplified operator, assembled separately from the
full one, so the two norms can differ substantially.  Where they do, a run
reports success at its requested ``solverTolerance`` while the state it returns
still leaves a large true residual.

This script measures that directly and plots it against the observed
cross-code difference.  Everything it uses comes from SFINCS's own binary
output -- its matrix, its right-hand side, its state vector -- so the
convergence measurement involves no dkx quantity at all:

    ||A x - b|| / ||b||        A = whichMatrix_3, x = stateVector, b = -residual

The comparison is restricted to linear runs.  With ``Phi1`` the problem is a
Newton iteration and the ``iteration_000`` matrix is the Jacobian at the
initial guess, so pairing it with a later state measures nothing.

The residual bounds how closely two solutions of the same system can agree in
the *state vector*.  It is not a bound on the output moments: those are
contractions of the state, so a residual can either cancel out of them or be
amplified by them, and the measured points fall on both sides of equality.
What the data does show is that every large cross-code difference here comes
with a large reference residual.

Nothing about this is specific to SFINCS: a preconditioned-norm convergence
test is standard practice and usually adequate.  The narrow point is that when
two codes disagree, the reference's own residual is worth checking first.

Inputs: a JSONL file from ``tools/benchmarks/parity_performance_matrix.py``
run with ``--fortran-residual``.

Reproduce (from the repo root):

  python tools/benchmarks/parity_performance_matrix.py \
      --examples /path/to/sfincs/fortran/version3/examples \
      --fortran-binary /path/to/sfincs \
      --fortran-launcher 'micromamba run -n sfincs-fortran' \
      --fortran-residual --out residual_sweep.jsonl

  python tools/paper_benchmarks/reference_convergence.py \
      --results residual_sweep.jsonl

Outputs ``docs/_static/figures/paper_benchmarks/reference_convergence.png``
and prints the table.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = REPO_ROOT / "docs" / "_static" / "figures" / "paper_benchmarks"
FIG_PATH = FIG_DIR / "reference_convergence.png"

# Below this the reference has converged in the true norm and the comparison is
# limited only by discretisation and round-off.
CONVERGED = 1e-8


def load(results: Path) -> list[dict]:
    """Linear cases that produced both a reference residual and a parity number."""
    rows = []
    for line in results.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        fortran = (record.get("fortran") or {}).get("1", {})
        parity = record.get("parity", {})
        residual = fortran.get("true_residual")
        values = [v for v in parity.values() if isinstance(v, float)]
        if residual is None or not values:
            continue
        rows.append(
            {
                "case": record["case"],
                "dof": record.get("dof", 0),
                "residual": float(residual),
                "parity": max(values),
            }
        )
    return sorted(rows, key=lambda r: r["residual"])


def plot(rows: list[dict], path: Path) -> None:
    """Reference residual against cross-code difference, one point per case."""
    residual = np.array([max(r["residual"], 1e-16) for r in rows])
    parity = np.array([max(r["parity"], 1e-16) for r in rows])
    converged = residual < CONVERGED

    fig, ax = plt.subplots(figsize=(7.6, 5.2), constrained_layout=True)
    limits = (1e-15, 3e0)

    ax.plot(limits, limits, color="0.72", lw=1.0, ls=(0, (5, 4)), zorder=1)
    ax.annotate("equal magnitude", (2e-4, 2e-4), rotation=39, fontsize=8,
                color="0.45", ha="center", va="bottom",
                rotation_mode="anchor")

    ax.scatter(residual[converged], parity[converged], s=58, zorder=3,
               color="#2b6cb0", edgecolor="white", linewidth=0.9,
               label=f"reference residual < {CONVERGED:g}  ({int(converged.sum())} cases)")
    ax.scatter(residual[~converged], parity[~converged], s=58, zorder=3,
               color="#c05621", edgecolor="white", linewidth=0.9, marker="s",
               label=f"reference residual $\\geq$ {CONVERGED:g}  ({int((~converged).sum())} cases)")

    # Label only the three largest reference residuals; more than that collides
    # at this aspect ratio and the individual names are not the point.
    offsets = [(-14, -20), (-14, 16), (-16, -22)]
    for offset, index in zip(offsets, np.argsort(residual)[-3:]):
        ax.annotate(
            rows[index]["case"].replace("geometryScheme", "geoScheme")[:32],
            (residual[index], parity[index]),
            textcoords="offset points", xytext=offset, ha="right",
            fontsize=8, color="#7b341e",
            arrowprops=dict(arrowstyle="-", color="#c05621", lw=0.7,
                            shrinkA=0, shrinkB=4),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_xlabel(
        r"reference true residual  $\|Ax-b\|/\|b\|$"
        "\n(computed from SFINCS's own matrix, state vector and right-hand side)"
    )
    ax.set_ylabel("largest relative difference\nin shared output moments")
    ax.set_title(
        "Every large cross-code difference comes with a large reference residual",
        fontsize=11.5, pad=10,
    )
    ax.grid(True, which="major", alpha=0.22, lw=0.6)
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.95, borderpad=0.7)
    ax.set_aspect("equal")

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
        raise SystemExit(
            "no linear cases carried both a reference residual and a parity number; "
            "was the sweep run with --fortran-residual?"
        )

    print(f"{'case':<48}{'dof':>9}{'ref residual':>15}{'difference':>13}")
    for row in rows:
        print(f"{row['case'][:47]:<48}{row['dof']:>9}"
              f"{row['residual']:>15.3e}{row['parity']:>13.3e}")

    unconverged = [r for r in rows if r["residual"] >= CONVERGED]
    print(f"\n{len(rows)} linear cases; {len(unconverged)} where the reference's own "
          f"true residual is >= {CONVERGED:g}")
    if unconverged:
        worst = max(unconverged, key=lambda r: r["residual"])
        print(f"largest: {worst['case']} at {worst['residual']:.3e}, "
              f"cross-code difference {worst['parity']:.3e}")

    plot(rows, args.out)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
