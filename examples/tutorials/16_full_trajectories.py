"""Tutorial 16 -- full trajectories against DKES trajectories, with a radial electric field.

The drift-kinetic equation has two common forms.  The DKES trajectory model
drops the E_r-dependent terms that change a particle's energy and pitch while
it drifts (``includeXDotTerm``, ``includeElectricFieldTermInXiDot``) and uses
a simplified E x B drift (``useDKESExBDrift``).  The full-trajectory model,
SFINCS's default, keeps them.  At E_r = 0 the two agree; as |E_r| grows the
extra terms matter, most for the trapped particles that carry the transport.

A circular tokamak with a small helical ripple, one ion species and
pitch-angle scattering is solved at E_r = 0 and at a finite field with each
model.  The summary prints the ion particle flux and the parallel flow, and
the full-to-DKES ratio at each field.

Run:              python examples/tutorials/16_full_trajectories.py
Runtime:          about 152 s on a loaded 36-core CPU node, compilation included.
"""

# 1. Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
ER = (0.0, -30.0)  # radial electric field in SFINCS units (d Phi / d psi_N scale)
MODELS = {  # trajectory model -> (includeXDotTerm, includeElectricFieldTermInXiDot, useDKESExBDrift)
    "DKES": (False, False, True),
    "full": (True, True, False),
}
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,  # analytic field at r/a = 0.3
    B0OverBBar=1.0, GHat=1.0, IHat=0.0, iota=1.31, epsilon_t=-0.1, epsilon_h=0.0,
    helicity_l=1, helicity_n=1, psiAHat=0.045, aHat=0.1,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[0.5], dNHatdrHats=[-6.0], dTHatdrHats=[-3.0],
    Delta=4.5694e-3, alpha=1.0, nu_n=8.4774e-3, collisionOperator=1,  # pitch-angle scattering
    Ntheta=11, Nzeta=1, Nxi=12, NL=4, Nx=4,  # coarse grid: seconds, not converged
)

# 3. Run -- one solve per (trajectory model, E_r)
runs = {}
for name, (x_dot, xi_dot, dkes_exb) in MODELS.items():
    for er in ER:
        print(f"\n--- {name} trajectories, Er = {er:g} ---", flush=True)
        runs[name, er] = dkx.run(**PARAMETERS, Er=er, includeXDotTerm=x_dot,
                                 includeElectricFieldTermInXiDot=xi_dot, useDKESExBDrift=dkes_exb, emit=print)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
flux = {key: float(np.asarray(run.moments["particleFlux_vm_psiHat"]).ravel()[0]) for key, run in runs.items()}
flow = {key: float(np.asarray(run.moments["FSABFlow"]).ravel()[0]) for key, run in runs.items()}
fig, axes = plt.subplots(1, 2, figsize=(8, 3.3), constrained_layout=True)
for axis, (values, title) in zip(axes, ((flux, "ion particle flux"), (flow, r"$\langle B V_\parallel\rangle$"))):
    for k, name in enumerate(MODELS):
        axis.bar(np.arange(len(ER)) + 0.2 * (2 * k - 1), [values[name, er] for er in ER], 0.4, label=name)
    axis.set_xticks(range(len(ER)), [f"Er = {er:g}" for er in ER])
    axis.set_title(title + " (SFINCS units)", fontsize=9)
    axis.axhline(0.0, color="k", lw=0.5)
axes[0].legend(fontsize=8)
fig.savefig(OUT / "trajectories.png", dpi=120)
print(f"\nsaved {OUT / 'trajectories.png'}")
print("\n=== Summary ===")
for er in ER:
    print(f"  Er = {er:+6.1f}: particle flux DKES {flux['DKES', er]:+.4e}  full {flux['full', er]:+.4e}"
          f"  full/DKES = {flux['full', er] / flux['DKES', er]:.3f}")
    print(f"  {'':12s}  flow          DKES {flow['DKES', er]:+.4e}  full {flow['full', er]:+.4e}")
