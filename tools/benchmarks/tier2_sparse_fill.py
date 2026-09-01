"""How much fill the Krylov preconditioner elimination order costs, per deck.

The classical Krylov preconditioner (:func:`dkx.coarse_precond.build_coarse_preconditioner`)
eliminates the Legendre index first, which fills the ``(theta, zeta)`` blocks in
completely: the Schur complement ``D_l - L_l D_{l-1}^{-1} U_{l-1}`` is dense even
though every input block carries only the 3- or 5-point ``createGrids.F90``
stencils.  :mod:`dkx.sparse_precond` inverts the same operator in a
fill-reducing order instead.

This script measures the difference in the one currency that does not depend on
the machine or on how loaded it is: **nonzeros**.  For each deck it reports the
dense bands the classical route allocates, the nonzeros of the sparse assembly,
and the fill after SuperLU factors one subsystem.  Factorization time is
deliberately *not* reported here -- it belongs in a controlled measurement, not
in a structural audit.

Reproduce (from the repo root):

  python tools/benchmarks/tier2_sparse_fill.py \\
      --examples /path/to/sfincs/fortran/version3/examples \\
      --decks tokamak_2species_PASCollisions_withEr_fullTrajectories \\
              geometryScheme4_2species_withEr_fullTrajectories \\
              sfincsPaperFigure3_geometryScheme11_PASCollisions_2Species_fullTrajectories

Prints one row per deck and, with ``--json``, writes the same numbers out.
"""

import argparse
import json
from pathlib import Path

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from dkx.sparse_precond import assemble_simplified


def measure(deck: Path) -> dict:
    """Structural numbers for one deck; factors a single subsystem."""
    import scipy.sparse.linalg as spla

    op = KineticOperator.from_namelist(read_sfincs_input(deck))
    assembled = assemble_simplified(op)
    n_tz = assembled.n_tz
    width = assembled.n_xi * n_tz
    n_sub = len(assembled.matrices)

    # Every subsystem shares one sparsity pattern, so one factorization measures
    # the fill of all of them.
    lu = spla.splu(assembled.matrices[0].tocsc())
    fill_one = int(lu.L.nnz + lu.U.nnz)

    dense_entries = assembled.dense_band_bytes / 8.0
    return {
        "case": deck.parent.name,
        "n_species": int(op.n_species),
        "n_x": int(op.n_x),
        "n_xi": int(op.n_xi),
        "n_theta": int(op.n_theta),
        "n_zeta": int(op.n_zeta),
        "n_tz": n_tz,
        "subsystems": n_sub,
        "subsystem_width": width,
        "dense_band_gb": assembled.dense_band_bytes / 2**30,
        # The classical route's factorization cost, in the same units the
        # module docstring quotes: Nxi * Nspecies * Nx * (Ntheta Nzeta)^3.
        "dense_factor_gflop": n_sub * assembled.n_xi * n_tz**3 / 1e9,
        "sparse_nnz": assembled.nnz,
        "sparse_fraction_of_dense": assembled.nnz / dense_entries,
        "lu_fill_gb": fill_one * n_sub * 12.0 / 2**30,  # 8 B value + 4 B index
        "lu_fill_over_assembly": fill_one / (assembled.matrices[0].nnz or 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--decks", nargs="+", required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = []
    header = (
        f"{'case':<52}{'TZ':>6}{'dense GB':>10}{'dense Gflop':>13}"
        f"{'nnz/dense':>11}{'LU fill GB':>12}"
    )
    print(header)
    for name in args.decks:
        row = measure(args.examples / name / "input.namelist")
        rows.append(row)
        print(
            f"{row['case'][:51]:<52}{row['n_tz']:>6}{row['dense_band_gb']:>10.2f}"
            f"{row['dense_factor_gflop']:>13.0f}"
            f"{row['sparse_fraction_of_dense']:>11.4f}{row['lu_fill_gb']:>12.2f}"
        )

    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
