"""Differentiate a geometry scalar with respect to |B| harmonic amplitudes.

What this example teaches:
  - how the canonical differentiable geometry constructor
    ``FluxSurfaceGeometry.from_fourier`` (the ``geometryScheme=13``
    |B|-spectrum path used by optimization loops) builds the on-surface field
    from a Boozer |B| spectrum,
  - how ``jax.grad`` (and its jitted form) differentiate a scalar geometry
    objective w.r.t. the ripple-harmonic amplitudes,
  - that the eager and jitted gradients agree to machine precision.

Physics context: stellarator optimization tunes the |B| Fourier spectrum to
shape neoclassical transport; being able to differentiate on-surface geometry
quantities w.r.t. the spectrum is what lets a gradient-based optimizer move the
harmonics [M. Landreman et al., Phys. Plasmas 21, 042503 (2014); SFINCS
technical documentation, https://github.com/landreman/sfincs].  The harmonics
here are the W7-X standard-configuration modes used by ``geometryScheme=4``.

Run:
  python examples/autodiff/differentiable_geometry_gradients.py
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# W7-X standard (geometryScheme=4): (m, n) modes and flux functions.
B0_OVER_BBAR = 3.089
M_MODES = jnp.asarray([0.0, 0.0, 1.0, 1.0], dtype=jnp.float64)
N_MODES = jnp.asarray([0.0, 1.0, 1.0, 0.0], dtype=jnp.float64)
N_PERIODS = 5
IOTA = 0.8700
G_HAT = -17.885
I_HAT = 0.0

# Ripple-harmonic amplitudes (relative to B0OverBBar) we differentiate against.
AMPS0 = jnp.asarray([0.04645, -0.04351, -0.01902], dtype=jnp.float64)

# On-surface grid resolution.
N_THETA = 31
N_ZETA = 31

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "differentiable_geometry_gradients"
PLOT_PATH = OUTPUT_DIR / "differentiable_geometry_gradients.png"

theta = jnp.linspace(0.0, 2 * jnp.pi, N_THETA, endpoint=False, dtype=jnp.float64)
zeta = jnp.linspace(0.0, 2 * jnp.pi / N_PERIODS, N_ZETA, endpoint=False, dtype=jnp.float64)


def build_geometry(harmonics_amp0: jnp.ndarray) -> FluxSurfaceGeometry:
    # bmnc holds the (0,0) mode (B0OverBBar) followed by the ripple harmonics.
    bmnc = jnp.concatenate([jnp.asarray([B0_OVER_BBAR]), harmonics_amp0 * B0_OVER_BBAR])
    return FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=bmnc, m=M_MODES, n=N_MODES,
        n_periods=N_PERIODS, iota=IOTA, g_hat=G_HAT, i_hat=I_HAT,
    )  # fmt: skip


def objective(harmonics_amp0: jnp.ndarray) -> jnp.ndarray:
    # A simple scalar objective: mean(BHat^2).
    return jnp.mean(build_geometry(harmonics_amp0).b_hat ** 2)


# ----------------------------------------------------------------------------
# 1) Gradient of the objective (eager and jitted)
# ----------------------------------------------------------------------------
print("=== examples/autodiff/differentiable_geometry_gradients.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print("Step 1: differentiating mean(BHat^2) w.r.t. the ripple amplitudes")
grad = jax.grad(objective)(AMPS0)
grad_jit = jax.jit(jax.grad(objective))(AMPS0)
err = float(jnp.max(jnp.abs(grad - grad_jit)))

# ----------------------------------------------------------------------------
# 2) Plot the field on the surface at the base amplitudes
# ----------------------------------------------------------------------------
print("Step 2: plotting BHat on the surface at the base amplitudes")
geom = build_geometry(AMPS0)
b_hat = np.asarray(geom.b_hat)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
pcm = ax.pcolormesh(np.asarray(zeta), np.asarray(theta), b_hat, shading="auto", cmap="viridis")
fig.colorbar(pcm, ax=ax, label=r"$\hat{B}$")
ax.set_xlabel(r"$\zeta$ (one field period)")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"from_fourier |B|-spectrum: $\hat B(\theta,\zeta)$")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  amps0            = {np.asarray(AMPS0)}")
print(f"  objective(amps0) = {float(objective(AMPS0)):.6e}")
print(f"  grad             = {np.asarray(grad)}")
print(f"  max |grad - grad(jit)| = {err:.3e}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/autodiff/differentiable_geometry_gradients.py")
