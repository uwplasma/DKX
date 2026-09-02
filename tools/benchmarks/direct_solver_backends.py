#!/usr/bin/env python
"""scipy SuperLU against MUMPS, on SFINCS's own matrix.

The question this settles: DKX's sparse-direct referee route refuses at any
useful size, and when forced it ran 75 minutes on a 66004-unknown deck without
finishing. Is direct factorization simply the wrong tool for these operators,
or is scipy's SuperLU the wrong *implementation* of it?

PETSc's event log says MUMPS factorizes the same class of operator as a small
slice of a SFINCS run. MUMPS is multifrontal with a fill-reducing ordering;
scipy's SuperLU is neither, so the gap could be large. If it is, DKX's tier-3
becomes a real referee rather than a route that declines, and stalls like the
large-|Er| HSX scan may have a deterministic answer instead of needing a
resolution change.

Method. SFINCS is asked to dump its assembled matrix
(``saveMatricesAndVectorsInBinary``), which is read back with DKX's own PETSc
binary reader, so both backends factorize a byte-identical operator. MUMPS's
number comes from PETSc's ``MatLUFactorNum`` event on the run that produced the
matrix; SuperLU's is timed here.

**Run this on an idle machine.** The same deck measured ``MatLUFactorNum`` at
0.025 s idle and 3.25 s under a concurrent sweep, a factor of 130.

Usage:
  python tools/benchmarks/direct_solver_backends.py WORKDIR [WORKDIR ...]

where each WORKDIR is a completed SFINCS run directory containing both
``sfincsBinary_iteration_000_whichMatrix_3`` and a ``-log_view`` dump saved as
``petsc_log.txt``.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path


def petsc_lu_seconds(log: Path) -> float | None:
    """MUMPS numeric factorization time, from PETSc's own event table."""
    if not log.is_file():
        return None
    for line in log.read_text(errors="replace").splitlines():
        if line.strip().startswith("MatLUFactorNum"):
            fields = line.split()
            try:
                return float(fields[3])
            except (IndexError, ValueError):
                return None
    return None


def petsc_total_seconds(log: Path) -> float | None:
    if not log.is_file():
        return None
    match = re.search(
        r"^Time \(sec\):\s+(\S+)", log.read_text(errors="replace"), re.MULTILINE
    )
    return float(match.group(1)) if match else None


def superlu_seconds(matrix_path: Path) -> tuple[float, int, int]:
    """Time scipy's SuperLU on the same matrix; returns (seconds, n, fill)."""
    from scipy.sparse import csc_matrix
    from scipy.sparse.linalg import splu

    from dkx.validation.fortran import read_petsc_mat_aij

    aij = read_petsc_mat_aij(matrix_path)
    operator = csc_matrix((aij.data, aij.col_ind, aij.row_ptr), shape=aij.shape)
    started = time.perf_counter()
    factors = splu(operator)
    elapsed = time.perf_counter() - started
    fill = int(factors.L.nnz + factors.U.nnz)
    del factors
    return elapsed, int(operator.shape[0]), fill


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("workdirs", nargs="+", type=Path)
    parser.add_argument(
        "--timeout-note",
        action="store_true",
        help="only report sizes and MUMPS times, without running SuperLU",
    )
    args = parser.parse_args()

    print(
        f"{'run':<40s} {'n':>8s} {'nnz(LU)':>12s} "
        f"{'MUMPS [s]':>10s} {'SuperLU [s]':>12s} {'ratio':>8s}"
    )
    for work in args.workdirs:
        matrix = work / "sfincsBinary_iteration_000_whichMatrix_3"
        if not matrix.is_file():
            print(f"{work.name[:40]:<40s} {'-':>8s} (no matrix dump)")
            continue
        mumps = petsc_lu_seconds(work / "petsc_log.txt")
        if args.timeout_note:
            print(
                f"{work.name[:40]:<40s} {'?':>8s} {'?':>12s} "
                f"{(mumps if mumps is not None else float('nan')):10.4f}"
            )
            continue
        seconds, size, fill = superlu_seconds(matrix)
        ratio = (seconds / mumps) if mumps else float("nan")
        print(
            f"{work.name[:40]:<40s} {size:8d} {fill:12d} "
            f"{(mumps if mumps is not None else float('nan')):10.4f} "
            f"{seconds:12.4f} {ratio:7.1f}x"
        )


if __name__ == "__main__":
    main()
