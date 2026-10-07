"""Compute v3 transport matrices (RHSMode=2 and RHSMode=3) with the JAX driver.

What this example teaches:
  - how ``dkx.run.run_transport_matrix`` runs the SFINCS v3 ``whichRHS`` loop
    (one solve per right-hand side) and assembles the ``transportMatrix``,
  - the two transport-matrix conventions: RHSMode=2 builds the 3x3
    energy-integrated matrix, RHSMode=3 the 2x2 monoenergetic matrix,
  - how to render each matrix as a heatmap.

Physics context: the transport matrix relates the thermodynamic forces
(density/temperature gradients and the parallel electric field) to the fluxes
(radial particle/heat flux and the parallel bootstrap current) on one flux
surface; it is the compact, force-free summary of the neoclassical solve
[M. Landreman, H. M. Smith, A. Mollen and P. Helander, Phys. Plasmas 21,
042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].  The RHSMode=3 monoenergetic form is the
one used in the ICNTS benchmark database.  Both runs here use tiny grids so
they finish in a couple of seconds -- they teach the workflow, not production
physics.

Run:
  python examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from dkx.run import run_transport_matrix  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
SOLVER_TOLERANCE = 1e-10  # Krylov tolerance for each whichRHS solve

# RHSMode=2 (3x3) energy-integrated transport matrix, simplified LHD model
# (geometryScheme=2).
DECK_RHSMODE2 = """\
&general
  RHSMode = 2
/
&geometryParameters
  geometryScheme = 2
/
&speciesParameters
  Zs = 1
  mHats = 1
  nHats = 1.0d+0
  THats = 1.0d+0
/
&physicsParameters
  Delta = 4.5694d-3
  alpha = 1.0d+0
  nu_n = 0.15d+0
  Er = 0.0d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 9
  Nzeta = 9
  Nxi = 6
  NL = 3
  Nx = 3
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""

# RHSMode=3 (2x2) monoenergetic transport matrix, 3-helicity analytic model
# (geometryScheme=1).
DECK_RHSMODE3 = """\
&general
  RHSMode = 3
/
&geometryParameters
  geometryScheme = 1
  epsilon_t = -0.07053d+0
  epsilon_h = 0.05067d+0
  iota = 0.4542d+0
  GHat = 3.7481d+0
  IHat = 0d+0
  helicity_l = 2
  helicity_n = 10
  B0OverBBar = 1d+0
/
&physicsParameters
  nuPrime = 1.0d+0
  EStar = 0.1d+0
  collisionOperator = 1
  includeXDotTerm = .false.
  includeElectricFieldTermInXiDot = .false.
  useDKESExBDrift = .true.
  includePhi1 = .false.
/
&resolutionParameters
  Ntheta = 9
  Nzeta = 9
  Nxi = 6
  NL = 3
  Nx = 1
  solverTolerance = 1d-10
/
&otherNumericalParameters
  Nxi_for_x_option = 0
/
"""

CASES = (("RHSMode=2", DECK_RHSMODE2), ("RHSMode=3", DECK_RHSMODE3))

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "transport_matrix_rhsmode2_and_rhsmode3"

# ----------------------------------------------------------------------------
# 1) Run both transport-matrix cases and collect the matrices
# ----------------------------------------------------------------------------
print("=== examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
matrices: dict[str, np.ndarray] = {}
for label, deck_text in CASES:
    print(f"Step 1[{label}]: writing the deck and running the whichRHS loop")
    deck_path = OUTPUT_DIR / f"{label.replace('=', '').lower()}.input.namelist"
    deck_path.write_text(deck_text, encoding="utf-8")
    result = run_transport_matrix(deck_path, tol=SOLVER_TOLERANCE, emit=None)
    tm = np.asarray(result.transport_matrix)
    matrices[label] = tm
    print(f"  {label} transportMatrix ({tm.shape[0]}x{tm.shape[1]}, mathematical row/col order):")
    print("   " + np.array2string(tm, prefix="   "))

# ----------------------------------------------------------------------------
# 2) Plot each transport matrix as a heatmap
# ----------------------------------------------------------------------------
print("Step 2: plotting the transport matrices")
fig, axes = plt.subplots(1, len(CASES), figsize=(9.0, 3.8), constrained_layout=True)
for ax, (label, _) in zip(np.atleast_1d(axes), CASES):
    tm = matrices[label]
    im = ax.imshow(tm, cmap="coolwarm", interpolation="nearest")
    ax.set_title(f"{label} transportMatrix")
    ax.set_xlabel("column (whichRHS)")
    ax.set_ylabel("row")
    fig.colorbar(im, ax=ax, shrink=0.85)
PLOT_PATH = OUTPUT_DIR / "transport_matrix_rhsmode2_and_rhsmode3.png"
fig.savefig(PLOT_PATH, dpi=150)
plt.close(fig)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
for label, _ in CASES:
    tm = matrices[label]
    print(f"  {label}: L11 (D11-like) = {float(tm[0, 0]):.6e}, shape = {tm.shape}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/transport/transport_matrix_rhsmode2_and_rhsmode3.py")
