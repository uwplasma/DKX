"""Parity-check the matrix-free drift-kinetic matvec against Fortran PETSc binaries.

Applies the canonical `KineticOperator` to a frozen Fortran v3 `stateVector`
and compares against the sparse matvec of the frozen `whichMatrix_3` matrix
(the full v3 solver matrix), printing a short summary instead of asserting.

By default it uses the repository fixture in `tests/ref/quick_2species_FPCollisions_noEr.*`.
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from scipy.sparse import csr_matrix

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_mat_aij, read_petsc_vec


def _default_prefix() -> Path:
    return Path(__file__).parents[2] / "tests" / "ref" / "quick_2species_FPCollisions_noEr"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=None, help="Path to input.namelist (default: fixture)")
    p.add_argument("--mat", default=None, help="Path to whichMatrix_3 petscbin (default: fixture)")
    p.add_argument("--vec", default=None, help="Path to stateVector petscbin (default: fixture)")
    args = p.parse_args()

    prefix = _default_prefix()
    input_path = Path(args.input) if args.input else Path(str(prefix) + ".input.namelist")
    mat_path = Path(args.mat) if args.mat else Path(str(prefix) + ".whichMatrix_3.petscbin")
    vec_path = Path(args.vec) if args.vec else Path(str(prefix) + ".stateVector.petscbin")

    nml = read_sfincs_input(input_path)
    op = kinetic_operator_from_namelist(nml)

    a = read_petsc_mat_aij(mat_path)
    x_ref = read_petsc_vec(vec_path).values
    if x_ref.size != op.total_size:
        raise SystemExit(f"State size mismatch: fixture {x_ref.size} vs operator {op.total_size}")

    y_jax = np.asarray(op.apply(jnp.asarray(x_ref)))
    y_ref = csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape).dot(x_ref)

    scale = max(1.0, float(np.max(np.abs(y_ref))))
    max_abs = float(np.max(np.abs(y_jax - y_ref)))
    print(f"n = {x_ref.size}")
    print(f"max |jax - fortran|          = {max_abs:.3e}")
    print(f"max |jax - fortran| / scale  = {max_abs / scale:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
