"""Tutorial 12 -- the kinetic bootstrap current against the Redl analytic fit.

The bootstrap current <j.B> is a parallel current driven by the pressure
gradient through trapped-passing friction.  Equilibrium codes usually take it
from an analytic fit; the standard one (Redl et al., Phys. Plasmas 28,
022502, 2021) refits Sauter's tokamak formula,

    <j.B> = -I(psi) p [ L31 dln n/dpsi + (L31 + L32) dln Te/dpsi + ... ],

with L31, L32 functions of the trapped fraction f_t and collisionality only.
For a quasisymmetric stellarator the same formula applies with the helicity
substitution, which is what VMEX/simsopt optimizers use.  DKX computes the
current from the drift-kinetic solution instead, with no such assumptions.

The surface: s = 0.46 of the two-field-period VMEX equilibrium of tutorial
04, Te = Ti, n = n0 (1 - s^5), T = T0 (1 - s), with Redl's value recorded in
tests/ref/boozer_route_sign_vmex_seed_nfp2.json.  That equilibrium is an
unoptimized seed, *not* quasisymmetric, so Redl is an extrapolation here and
the two need not agree: read the ratio as how far the fit is from the kinetic
answer on this surface.  PAS lacks momentum restoration, so only the
Fokker-Planck value is the kinetic reference; rerun at higher resolution to
see how much of the gap is discretization.

Run:              python examples/tutorials/12_bootstrap_vs_redl.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/12_bootstrap_vs_redl.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import json
import lzma
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import dkx
from dkx.units import PARALLEL_CURRENT

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
REF = Path(__file__).resolve().parents[2] / "tests" / "ref"
FIXTURE = json.loads((REF / "boozer_route_sign_vmex_seed_nfp2.json").read_text())
S = FIXTURE["s"]  # normalized toroidal flux of the surface
N0, T0 = 1.0e19, 2.0e3  # m^-3, eV
N_HAT, T_HAT = N0 / 1e20 * (1 - S**5), T0 / 1e3 * (1 - S)  # SFINCS units: 1e20 m^-3, keV
PARAMETERS = dict(
    geometryScheme=5, VMECRadialOption=0, inputRadialCoordinate=1, inputRadialCoordinateForGradients=1,
    psiN_wish=S, Zs=[1.0, -1.0], mHats=[1.0, 5.446170214e-4], nHats=[N_HAT, N_HAT], THats=[T_HAT, T_HAT],
    dNHatdpsiNs=[-5 * N0 / 1e20 * S**4] * 2, dTHatdpsiNs=[-T0 / 1e3] * 2,
    Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3, Er=0.0,
    # Coarser than this the current changes sign: the floor, not a converged grid.
    **(dict(Ntheta=11, Nzeta=11, Nxi=16, NL=4, Nx=4) if SMOKE else dict(Ntheta=17, Nzeta=17, Nxi=32, NL=4, Nx=5)),
)
OPERATORS = {"Fokker-Planck": 0, "pitch-angle scattering": 1}  # SFINCS collisionOperator

# 3. Run
wout = REF / FIXTURE["wout"]
if not wout.exists():
    with lzma.open(wout.with_name(wout.name + ".xz")) as src, open(wout, "wb") as dst:
        shutil.copyfileobj(src, dst)
current = {}
for name, operator in OPERATORS.items():
    print(f"\n--- {name} ---", flush=True)
    run = dkx.run(equilibriumFile=str(wout), collisionOperator=operator, emit=print, **PARAMETERS)
    current[name] = float(np.asarray(run.moments["FSABjHat"]).reshape(())) * PARALLEL_CURRENT  # A T m^-2
redl = FIXTURE["redl_jdotB_A_T_per_m2"]

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
labels = ["Redl fit", *current]
values = np.array([redl, *current.values()]) / 1e6
fig, axis = plt.subplots(figsize=(5.5, 3.5))
axis.bar(labels, values, color=["0.5", "tab:blue", "tab:orange"])
axis.set_ylabel(r"$\langle j\cdot B\rangle$ [MA T m$^{-2}$]")
axis.set_title(f"bootstrap current at s = {S:.2f}")
fig.tight_layout()
fig.savefig(OUT / "bootstrap.png", dpi=120)
(OUT / "bootstrap.json").write_text(json.dumps({"s": S, "redl": redl, **current}, indent=1))
print(f"\nsaved {OUT / 'bootstrap.json'}\nsaved {OUT / 'bootstrap.png'}")
print("\n=== Summary ===")
print(f"  Redl fit               <j.B> = {redl:+.4e} A T m^-2")
for name, value in current.items():
    print(f"  DKX {name:22s} <j.B> = {value:+.4e} A T m^-2  ratio to Redl = {value / redl:.3f}")
