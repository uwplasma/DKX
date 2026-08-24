"""The 3x3 transport matrix, and the symmetry that tells you if it is converged.

``RHSMode=2`` solves the three drives at once and returns the transport matrix
relating the fluxes to the thermodynamic forces.  ``dkx.run_transport_matrix``
does it in one call.

Onsager symmetry is what makes it worth running.  ``L_ij = L_ji`` follows from
time-reversal symmetry of the kinetic equation and is not imposed by the
solver, so the asymmetry that comes out is a free error estimate -- a visibly
asymmetric matrix is under-resolved, whatever its residual says.

Physics: the tokamak of ``run_tokamak.py``.  Row/column 1 is the particle
channel, 2 the heat channel, 3 the parallel-current channel.

Expected runtime: ~20 s on a laptop CPU.

Achieved: L12 and L21 agree to 3.9e-14 -- the particle-heat block is exact to
round-off.  The current-coupled pairs differ by 1.3e-04 and 1.1e-04, so those
entries carry error in their fourth digit and the script reports three as the
conservative floor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
CASE = dict(
    RHSMode=2,                       # 2 = transport matrix; 1 = profiles; 3 = monoenergetic
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    Ntheta=13, Nzeta=1, Nxi=24, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=0.01,
)
LABELS = ("particle", "heat", "current")
OUT = Path(__file__).resolve().parent / "output" / "transport_matrix.png"
# ----------------------------- end of parameters ----------------------------

run = dkx.run_transport_matrix(dkx.SfincsInput.from_params(**CASE), emit=None)
matrix = np.asarray(run.transport_matrix)

print("transport matrix L:")
for row, label in zip(matrix, LABELS):
    print(f"  {label:>8}  " + "  ".join(f"{value:+.8e}" for value in row))

# Onsager: the solver never imposes this, so the residual is a measurement.
print("\nOnsager symmetry (L_ij vs L_ji) -- not imposed, so this measures error:")
asymmetry = np.zeros_like(matrix)
for i in range(3):
    for j in range(i + 1, 3):
        scale = max(abs(matrix[i, j]), abs(matrix[j, i]))
        relative = abs(matrix[i, j] - matrix[j, i]) / scale if scale else 0.0
        asymmetry[i, j] = asymmetry[j, i] = relative
        print(f"  L{i + 1}{j + 1} vs L{j + 1}{i + 1}: {relative:.2e} relative")

worst = asymmetry.max()
print(f"\nworst asymmetry {worst:.2e} -> trust about "
      f"{max(0, int(-np.log10(worst))) if worst else 15} digits of this matrix")
assert worst < 1e-2, "matrix is visibly asymmetric: raise Nxi/Ntheta before using it"

figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)

# The entries span four decades and both signs, so the colour scale has to be
# symmetric-log.  Not sign(L)*log10|L|: that inverts the ordering below |L|=1,
# where log10 goes negative, and paints a small negative entry the same shade
# as a large positive one -- which is exactly what L11 and L33 are here.
norm = colors.SymLogNorm(linthresh=1e-2, vmin=-np.abs(matrix).max(),
                         vmax=np.abs(matrix).max(), base=10)
image = left.imshow(matrix, cmap="RdBu_r", norm=norm)
left.set_title(r"$L$ (symmetric-log colour scale)")
figure.colorbar(image, ax=left, fraction=0.046)
for i in range(3):
    for j in range(3):
        left.text(j, i, f"{matrix[i, j]:+.2e}", ha="center", va="center", fontsize=7)

symmetry_image = right.imshow(asymmetry, cmap="viridis")
right.set_title("relative asymmetry (0 = converged)")
figure.colorbar(symmetry_image, ax=right, fraction=0.046)
for i in range(3):
    for j in range(3):
        right.text(j, i, "—" if i == j else f"{asymmetry[i, j]:.1e}",
                   ha="center", va="center", fontsize=8, color="w")

for axis in (left, right):
    axis.set_xticks(range(3), LABELS, fontsize=8)
    axis.set_yticks(range(3), LABELS, fontsize=8)

figure.suptitle("Transport matrix and its Onsager residual")
OUT.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(OUT, dpi=150)
print(f"wrote {OUT}")
