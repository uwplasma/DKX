"""Apply the v3 drift-kinetic operator matrix-free in JAX.

This example shows:
- Building the canonical `KineticOperator` from an `input.namelist`
- Running a non-jitted and jitted matvec and confirming they match

The matvec includes every v3 term the deck selects (streaming, mirror,
collisions, constraints) without forming a sparse matrix.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402


def _default_input() -> Path:
    return Path(__file__).parents[1] / "data" / "geometryScheme4_quick_2species.input.namelist"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default=str(_default_input()))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    nml = read_sfincs_input(Path(args.input))
    op = kinetic_operator_from_namelist(nml)

    key = jax.random.key(args.seed)
    v = jax.random.normal(key, shape=(op.total_size,), dtype=jnp.float64)

    y = op.apply(v)
    apply_jit = jax.jit(lambda operator, x: operator.apply(x))
    y_jit = apply_jit(op, v).block_until_ready()
    err = np.max(np.abs(np.asarray(y_jit) - np.asarray(y)))

    print(f"state size = {op.total_size}")
    print(f"out.shape  = {tuple(y.shape)}")
    print(f"max |jit - nojit| = {err:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
