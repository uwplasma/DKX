"""Tutorial 10 -- transport matrices: SFINCS ``RHSMode = 2`` and ``RHSMode = 3``.

Because the drift-kinetic equation is linear in its drives, the fluxes are a
matrix times the thermodynamic forces.  With RHSMode = 2 DKX solves once per
force (density gradient, temperature gradient, inductive E_par) and assembles
the 3x3 energy-integrated matrix

    [ particle flux ]   [L11 L12 L13] [ A1 (dn/dpsi, dPhi/dpsi) ]
    [ heat flux     ] = [L21 L22 L23] [ A2 (dT/dpsi)            ]
    [ parallel flow ]   [L31 L32 L33] [ A3 (<E_par B>)          ]

so any gradient combination is a matrix-vector product rather than a new
solve.  Onsager symmetry (L12 = L21 up to normalization) is a check on the
solution.  RHSMode = 3 is the 2x2 *monoenergetic* matrix at one (nu', E*):
L11 ~ D11*, L12 ~ L21 ~ D31*, L22 ~ D33* (tutorial 09 scans it).
Both runs pass SFINCS parameters as keywords to ``dkx.run`` -- no deck file.
Geometry: geometryScheme = 2 (a simplified LHD model) and = 1 (analytic
helical field).

Run:              python examples/tutorials/10_transport_matrix.py
CLI:              dkx sfincs transport-matrix-v3 --input input.namelist
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/10_transport_matrix.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
GRID = dict(Ntheta=9, Nzeta=9, Nxi=6, NL=3) if SMOKE else dict(Ntheta=15, Nzeta=15, Nxi=24, NL=4)
CASES = {
    "RHSMode=2 (3x3, LHD model)": dict(
        RHSMode=2, geometryScheme=2, Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
        Delta=4.5694e-3, alpha=1.0, nu_n=0.15, Er=0.0, collisionOperator=1,
        includeXDotTerm=False, includeElectricFieldTermInXiDot=False, useDKESExBDrift=True,
        Nx=3 if SMOKE else 5, solverTolerance=1e-10, **GRID),
    "RHSMode=3 (2x2, monoenergetic)": dict(
        RHSMode=3, geometryScheme=1, epsilon_t=-0.07053, epsilon_h=0.05067, iota=0.4542,
        GHat=3.7481, IHat=0.0, helicity_l=2, helicity_n=10, B0OverBBar=1.0,
        nuPrime=1.0, EStar=0.1, collisionOperator=1, includeXDotTerm=False,
        includeElectricFieldTermInXiDot=False, useDKESExBDrift=True, Nx=1,
        solverTolerance=1e-10, **GRID),
}

# 3. Run -- one solve per right-hand side, progress printed
matrices = {}
for label, parameters in CASES.items():
    print(f"\n--- {label} ---", flush=True)
    matrices[label] = np.asarray(dkx.run(emit=print, **parameters).transport_matrix, dtype=float)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
for axis, (label, matrix) in zip(axes, matrices.items()):
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-abs(matrix).max(), vmax=abs(matrix).max())
    for (i, j), value in np.ndenumerate(matrix):
        axis.text(j, i, f"{value:.2e}", ha="center", va="center", fontsize=7)
    axis.set_title(label, fontsize=9)
    axis.set_xlabel("drive (whichRHS)")
    axis.set_ylabel("flux")
    fig.colorbar(image, ax=axis, shrink=0.8)
fig.savefig(OUT / "transport_matrix.png", dpi=120)
np.savez(OUT / "transport_matrix.npz", **{k.split()[0].replace("=", ""): v for k, v in matrices.items()})
print(f"\nsaved {OUT / 'transport_matrix.npz'}\nsaved {OUT / 'transport_matrix.png'}")
print("\n=== Summary ===")
for label, matrix in matrices.items():
    print(f"  {label}:\n    " + np.array2string(matrix, precision=4, prefix="    "))
    print(f"    Onsager check |L12 - L21| / |L12| = {abs(matrix[0, 1] - matrix[1, 0]) / abs(matrix[0, 1]):.2e}")
