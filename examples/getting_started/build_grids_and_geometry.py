"""Build v3 grids and simplified Boozer geometry (``geometryScheme=4``).

What this example teaches:
  - how ``dkx.namelist.read_sfincs_input`` parses a Fortran ``input.namelist``,
  - how ``dkx.drift_kinetic.kinetic_operator_from_namelist`` builds the grids
    and magnetic geometry (theta/zeta/x grids, ``BHat`` and its derivatives,
    the Jacobian factor ``DHat``) in one call, without the Fortran code,
  - how to form a flux-surface average with the v3 quadrature weights and
    plot the field strength on the surface.

Physics context: SFINCS works in Boozer-like coordinates; ``geometryScheme=4``
supplies the simplified W7-X Boozer field the Fortran v3 tests use.  The
flux-surface average <g> = sum(w g) / sum(w) with weights
w = thetaWeights * zetaWeights / DHat is the same average that enters the v3
density/flow constraints [M. Landreman et al., Phys. Plasmas 21, 042503 (2014);
SFINCS technical documentation, https://github.com/landreman/sfincs].

Run:
  python examples/getting_started/build_grids_and_geometry.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.drift_kinetic import kinetic_operator_from_namelist  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ on sys.path
from _example_utils import output_dir  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
# A tiny geometryScheme=4 two-species deck that ships in examples/data/.
INPUT_NAMELIST = Path(__file__).resolve().parents[1] / "data" / "geometryScheme4_quick_2species.input.namelist"

OUTPUT_DIR = output_dir(__file__)
PLOT_PATH = OUTPUT_DIR / "build_grids_and_geometry.png"

# ----------------------------------------------------------------------------
# 1) Parse the namelist and build the operator (grids + geometry)
# ----------------------------------------------------------------------------
print("=== examples/getting_started/build_grids_and_geometry.py ===")
print(f"Step 1: building the kinetic operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op = kinetic_operator_from_namelist(nml)
print(f"  grid sizes: Ntheta={op.n_theta} Nzeta={op.n_zeta} Nx={op.n_x}")

# ----------------------------------------------------------------------------
# 2) Inspect the geometry arrays (internal layout is (Ntheta, Nzeta))
# ----------------------------------------------------------------------------
print("Step 2: inspecting the geometry arrays")
print(f"  BHat.shape        = {tuple(op.b_hat.shape)}")
print(f"  dBHatdtheta.shape = {tuple(op.db_hat_dtheta.shape)}")
print(f"  dBHatdzeta.shape  = {tuple(op.db_hat_dzeta.shape)}")
print(f"  DHat.shape        = {tuple(op.d_hat.shape)}")

# Flux-surface average of BHat using the v3 constraint weights.
w = op.theta_weights[:, None] * op.zeta_weights[None, :] / op.d_hat
bhat_fsa = float(jnp.sum(w * op.b_hat) / jnp.sum(w))

# ----------------------------------------------------------------------------
# 3) Plot the field strength on the surface
# ----------------------------------------------------------------------------
print("Step 3: plotting BHat on the flux surface")
# The v3 theta grid is uniform on [0, 2*pi); the zeta grid is uniform over one
# field period.  Reconstruct both from the grid sizes for the axes.
theta = np.linspace(0.0, 2.0 * np.pi, op.n_theta, endpoint=False)
zeta_index = np.arange(op.n_zeta)
b_hat = np.asarray(op.b_hat)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
pcm = ax.pcolormesh(zeta_index, theta, b_hat, shading="auto", cmap="viridis")
fig.colorbar(pcm, ax=ax, label=r"$\hat{B}$")
ax.set_xlabel(r"$\zeta$ grid point (one field period)")
ax.set_ylabel(r"$\theta$")
ax.set_title(r"geometryScheme=4: $\hat B(\theta,\zeta)$")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  flux-surface average <BHat> = {bhat_fsa:.14e}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/getting_started/build_grids_and_geometry.py")
