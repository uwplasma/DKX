"""Matrix-free residual and Jacobian-vector products (JVP) for the v3 F-block.

What this example teaches:
  - how to express the discrete drift-kinetic problem as a residual
    ``r(x) = A x - b`` and apply the Jacobian matrix-free with ``jax.jvp``,
  - how reverse-mode AD (``jax.grad``) differentiates a scalar built on the
    residual, ``phi(x) = 0.5 * ||r(x)||^2``, with no sparse assembly,
  - how the autodiff directional derivative matches a centered finite
    difference.

Physics context: the base drift-kinetic system is linear in the distribution
``x``, so ``r(x)`` is linear and ``J v`` is just ``A v``; the same JVP machinery
extends to the nonlinear ``includePhi1`` Newton solve, keeping the Jacobian
application matrix-free throughout [M. Landreman et al., Phys. Plasmas 21,
042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].

Run:
  python examples/autodiff/matrix_free_residual_and_jvp.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# A tiny single-species PAS deck from the test suite.
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
SEED = 0  # PRNG seed for the random state and direction vectors
FD_EPS = 1e-6  # centered finite-difference step for the directional-derivative check

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "matrix_free_residual_and_jvp"
PLOT_PATH = OUTPUT_DIR / "matrix_free_residual_and_jvp.png"

# ----------------------------------------------------------------------------
# 1) Build the operator and set up the residual
# ----------------------------------------------------------------------------
print("=== examples/autodiff/matrix_free_residual_and_jvp.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Step 1: building the operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op = kinetic_operator_from_namelist(nml)

rng = np.random.default_rng(SEED)
x0 = jnp.asarray(rng.normal(size=(op.total_size,)).astype(np.float64))
v = jnp.asarray(rng.normal(size=(op.total_size,)).astype(np.float64))
b = op.apply(x0)  # choose b so the "true" solution is x0 (for demonstration)


def residual(x: jnp.ndarray) -> jnp.ndarray:
    return op.apply(x) - b


def phi(x: jnp.ndarray) -> jnp.ndarray:
    r = residual(x)
    return 0.5 * jnp.vdot(r, r)


# ----------------------------------------------------------------------------
# 2) Residual, JVP, and the reverse-mode gradient
# ----------------------------------------------------------------------------
print("Step 2: evaluating the residual, a JVP, and grad(phi)")
r0, jv = jax.jvp(residual, (x0,), (v,))
grad_phi = jax.grad(phi)(x0)

# Finite-difference check of the directional derivative of phi along v.
fd = float((phi(x0 + FD_EPS * v) - phi(x0 - FD_EPS * v)) / (2 * FD_EPS))
ad = float(jnp.vdot(grad_phi, v))

# ----------------------------------------------------------------------------
# 3) Plot phi(x0 + t v) to visualize the smooth, differentiable objective
# ----------------------------------------------------------------------------
print("Step 3: plotting phi(x0 + t v)")
ts = np.linspace(-5e-3, 5e-3, 101)
vals = np.array([float(phi(x0 + float(t) * v)) for t in ts])
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.plot(ts, vals, lw=2, color="tab:blue")
ax.set_xlabel("t")
ax.set_ylabel(r"$\phi(x_0 + t v)$")
ax.set_title("Matrix-free residual objective (autodiff-ready)")
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  n = {op.total_size}")
print(f"  ||r(x0)||_2 = {float(jnp.linalg.norm(r0)):.3e}  (should be ~0)")
print(f"  ||J v||_2   = {float(jnp.linalg.norm(jv)):.3e}")
print(f"  directional derivative: finite-diff={fd:.6e}  autodiff={ad:.6e}  abs_err={abs(fd - ad):.3e}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/autodiff/matrix_free_residual_and_jvp.py")
