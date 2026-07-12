"""Build v3 grids + simplified Boozer geometry (geometryScheme=4).

This example does *not* require the Fortran code. It demonstrates the supported sfincs_jax
Python API surface:

- Parse an `input.namelist`
- Build the canonical drift-kinetic operator (grids + geometry in one call)
- Inspect the theta/zeta/x grids and the W7-X Boozer fields used by Fortran v3
  `geometryScheme=4`
"""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import jax.numpy as jnp

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.drift_kinetic import kinetic_operator_from_namelist
from sfincs_jax.namelist import read_sfincs_input


def _default_input() -> Path:
    return Path(__file__).parents[1] / "data" / "geometryScheme4_quick_2species.input.namelist"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input",
        default=str(_default_input()),
        help="Path to SFINCS input.namelist (default: examples/data/*)",
    )
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    op = kinetic_operator_from_namelist(nml)

    print("Grid sizes:")
    print(f"  Ntheta = {op.n_theta}")
    print(f"  Nzeta  = {op.n_zeta}")
    print(f"  Nx     = {op.n_x}")

    print("\nGeometry arrays (internal layout is (Ntheta, Nzeta)):")
    print(f"  BHat.shape          = {tuple(op.b_hat.shape)}")
    print(f"  dBHatdtheta.shape   = {tuple(op.db_hat_dtheta.shape)}")
    print(f"  dBHatdzeta.shape    = {tuple(op.db_hat_dzeta.shape)}")
    print(f"  DHat.shape          = {tuple(op.d_hat.shape)}")

    # A simple flux-surface average using the weights that appear in the v3 constraints:
    # <g> ~ sum(thetaWeights * zetaWeights / DHat * g) / sum(thetaWeights * zetaWeights / DHat)
    w = op.theta_weights[:, None] * op.zeta_weights[None, :] / op.d_hat
    bhat_fsa = jnp.sum(w * op.b_hat) / jnp.sum(w)
    print(f"\nFlux-surface average <BHat> ≈ {float(bhat_fsa):.14e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
