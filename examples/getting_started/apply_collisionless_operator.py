"""Apply the v3 drift-kinetic operator matrix-free in JAX.

What this example teaches:
  - how to build the canonical ``KineticOperator`` from an ``input.namelist``,
  - how the operator applies matrix-free (``op.apply``): it computes every v3
    term the deck selects (streaming, mirror force, collisions, constraints)
    on a state vector without ever forming a sparse matrix,
  - how ``jax.jit`` compiles that same matvec and reproduces the eager result
    to machine precision.

Physics context: SFINCS discretizes the drift-kinetic operator on the
theta/zeta/xi/x phase-space grid; the matrix-free action is what the iterative
(Krylov) solvers need, and keeping it in JAX is what makes the whole operator
differentiable [M. Landreman et al., Phys. Plasmas 21, 042503 (2014); SFINCS
technical documentation, https://github.com/landreman/sfincs].  Applying the
operator to a random state is a cheap way to exercise the full matvec.

Run:
  python examples/getting_started/apply_collisionless_operator.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.namelist import read_sfincs_input

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# A tiny geometryScheme=4 two-species deck that ships in examples/data/.
INPUT_NAMELIST = Path(__file__).resolve().parents[1] / "data" / "geometryScheme4_quick_2species.input.namelist"
SEED = 0  # PRNG seed for the random state vector

# ----------------------------------------------------------------------------
# 1) Build the operator
# ----------------------------------------------------------------------------
print("=== examples/getting_started/apply_collisionless_operator.py ===")
print(f"Step 1: building the kinetic operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op = kinetic_operator_from_namelist(nml)
print(f"  state size = {op.total_size}")

# ----------------------------------------------------------------------------
# 2) Apply the operator eagerly and under jit, and compare
# ----------------------------------------------------------------------------
print("Step 2: applying the matvec eagerly and under jax.jit")
key = jax.random.key(SEED)
v = jax.random.normal(key, shape=(op.total_size,), dtype=jnp.float64)

y = op.apply(v)
apply_jit = jax.jit(lambda operator, x: operator.apply(x))
y_jit = apply_jit(op, v).block_until_ready()
err = float(np.max(np.abs(np.asarray(y_jit) - np.asarray(y))))

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  out.shape         = {tuple(y.shape)}")
print(f"  max |jit - nojit| = {err:.3e}")
print("Done: examples/getting_started/apply_collisionless_operator.py")
