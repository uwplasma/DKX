#!/usr/bin/env python
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_system import (
    apply_v3_full_system_operator_cached,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
)


@dataclass(frozen=True)
class VectorSummary:
    norm2: float
    max_abs: float
    argmax_1based: int | None


def summarize_vector(values: np.ndarray) -> VectorSummary:
    """Return stable diagnostics for a vector difference or residual."""
    arr = np.asarray(values, dtype=np.float64).reshape((-1,))
    if arr.size == 0:
        return VectorSummary(norm2=0.0, max_abs=0.0, argmax_1based=None)
    idx = int(np.argmax(np.abs(arr)))
    return VectorSummary(norm2=float(np.linalg.norm(arr)), max_abs=float(np.max(np.abs(arr))), argmax_1based=idx + 1)


def parse_petsc_matlab_vector(path: Path) -> np.ndarray:
    """Parse a PETSc ASCII MATLAB vector dump produced by SFINCS v3."""
    values: list[float] = []
    in_data = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.endswith("= ["):
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("];"):
            break
        if not stripped or stripped.startswith("%"):
            continue
        try:
            values.append(float(stripped.replace("D", "E").replace("d", "e")))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64)


def parse_petsc_matlab_sparse(path: Path) -> sp.csr_matrix:
    """Parse a PETSc ASCII MATLAB sparse matrix dump into CSR form."""
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    shape: tuple[int, int] | None = None
    in_data = False
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = line.strip()
        if stripped.startswith("% Size ="):
            parts = stripped.replace("% Size =", "", 1).split()
            if len(parts) >= 2:
                shape = (int(parts[0]), int(parts[1]))
            continue
        if stripped == "zzz = [":
            in_data = True
            continue
        if not in_data:
            continue
        if stripped.startswith("];"):
            break
        parts = stripped.split()
        if len(parts) != 3:
            continue
        rows.append(int(parts[0]) - 1)
        cols.append(int(parts[1]) - 1)
        vals.append(float(parts[2].replace("D", "E").replace("d", "e")))
    if shape is None:
        raise ValueError(f"Could not find matrix shape in {path}")
    return sp.coo_matrix((np.asarray(vals), (np.asarray(rows), np.asarray(cols))), shape=shape).tocsr()


def _maybe_parse_vector(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return parse_petsc_matlab_vector(path)


def _matrix_path(dump_dir: Path, *, prefix: str, iteration: int, which_matrix: int) -> Path:
    return dump_dir / f"{prefix}_iteration_{int(iteration):03d}_whichMatrix_{int(which_matrix)}.m"


def _vector_path(dump_dir: Path, *, prefix: str, iteration: int, name: str) -> Path:
    return dump_dir / f"{prefix}_iteration_{int(iteration):03d}_{name}.m"


def solve_sparse_system_report(
    matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    reference_state: np.ndarray | None = None,
) -> dict[str, object]:
    """Solve a dumped sparse system with SciPy and return residual diagnostics."""

    matrix_csc = matrix.tocsc()
    rhs_arr = np.asarray(rhs, dtype=np.float64).reshape((-1,))
    t0 = time.perf_counter()
    x = np.asarray(spla.spsolve(matrix_csc, rhs_arr), dtype=np.float64).reshape((-1,))
    elapsed_s = time.perf_counter() - t0
    residual = np.asarray(matrix @ x, dtype=np.float64).reshape((-1,)) - rhs_arr
    report: dict[str, object] = {
        "elapsed_s": float(elapsed_s),
        "solution_norm": asdict(summarize_vector(x)),
        "true_residual": asdict(summarize_vector(residual)),
    }
    if reference_state is not None:
        ref = np.asarray(reference_state, dtype=np.float64).reshape((-1,))
        if ref.shape == x.shape:
            report["state_difference"] = asdict(summarize_vector(x - ref))
    return report


def audit_operator_dump(
    *,
    input_path: Path,
    dump_dir: Path,
    prefix: str,
    matrix_iteration: int = 0,
    which_matrix: int = 1,
    state_iteration: int = 0,
    residual_iterations: Iterable[int] = (0, 1),
    solve_sparse: bool = False,
) -> dict[str, object]:
    """Compare a Fortran v3 dumped matrix/vector against the JAX operator.

    This is intended for targeted parity triage. It does not run the solver; it
    checks whether the assembled JAX operator and RHS reproduce the dumped
    Fortran linear system and whether the saved Fortran state is a true residual
    clean solution of that system.
    """
    input_path = Path(input_path)
    dump_dir = Path(dump_dir)
    matrix_file = _matrix_path(
        dump_dir,
        prefix=prefix,
        iteration=int(matrix_iteration),
        which_matrix=int(which_matrix),
    )
    state_file = _vector_path(dump_dir, prefix=prefix, iteration=int(state_iteration), name="stateVector")

    matrix = parse_petsc_matlab_sparse(matrix_file)
    state = _maybe_parse_vector(state_file)
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml)
    rhs = np.asarray(rhs_v3_full_system(op), dtype=np.float64)

    report: dict[str, object] = {
        "input": str(input_path.resolve()),
        "dump_dir": str(dump_dir.resolve()),
        "matrix_file": str(matrix_file.resolve()),
        "matrix_shape": list(matrix.shape),
        "matrix_nnz": int(matrix.nnz),
        "operator_total_size": int(op.total_size),
        "rhs_norm": asdict(summarize_vector(rhs)),
        "residual_dumps": {},
    }
    if int(matrix.shape[0]) != int(op.total_size) or int(matrix.shape[1]) != int(op.total_size):
        report["status"] = "shape_mismatch"
        return report

    for iteration in residual_iterations:
        residual_file = _vector_path(dump_dir, prefix=prefix, iteration=int(iteration), name="residual")
        residual = _maybe_parse_vector(residual_file)
        if residual is not None:
            report["residual_dumps"][str(int(iteration))] = asdict(summarize_vector(residual))

    if state is not None:
        y_fortran = np.asarray(matrix @ state, dtype=np.float64)
        y_jax = np.asarray(apply_v3_full_system_operator_cached(op, state), dtype=np.float64)
        report.update(
            {
                "state_file": str(state_file.resolve()),
                "state_norm": asdict(summarize_vector(state)),
                "operator_on_state_diff": asdict(summarize_vector(y_jax - y_fortran)),
                "fortran_matrix_true_residual": asdict(summarize_vector(y_fortran - rhs)),
                "jax_operator_true_residual": asdict(summarize_vector(y_jax - rhs)),
            }
        )

    if solve_sparse:
        report["scipy_sparse_solve"] = solve_sparse_system_report(matrix, rhs, reference_state=state)

    report["status"] = "ok"
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a SFINCS v3 MATLAB matrix dump against sfincs_jax.")
    parser.add_argument("--input", required=True, type=Path, help="input.namelist used for the run.")
    parser.add_argument("--dump-dir", required=True, type=Path, help="Directory containing MATLAB dumps.")
    parser.add_argument("--prefix", default="sfincsMatrices", help="MatlabOutputFilename prefix used by v3.")
    parser.add_argument("--matrix-iteration", type=int, default=0)
    parser.add_argument("--which-matrix", type=int, default=1)
    parser.add_argument("--state-iteration", type=int, default=0)
    parser.add_argument(
        "--solve-sparse",
        action="store_true",
        help="Also solve the dumped sparse matrix with SciPy and report residual/time.",
    )
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    report = audit_operator_dump(
        input_path=args.input,
        dump_dir=args.dump_dir,
        prefix=str(args.prefix),
        matrix_iteration=int(args.matrix_iteration),
        which_matrix=int(args.which_matrix),
        state_iteration=int(args.state_iteration),
        solve_sparse=bool(args.solve_sparse),
    )
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if report.get("status") == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
