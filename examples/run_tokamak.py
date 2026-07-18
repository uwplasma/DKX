"""Single-species pitch-angle-scattering tokamak run, from a namelist built in Python.

What this example teaches:
  - how to create a SFINCS ``input.namelist`` from scratch as a Python dict,
  - how to run the canonical RHSMode=1 driver (``dkx.run.run_profile``),
  - how to read the per-species results table and the moments dictionary,
  - how to write ``sfincsOutput`` in both HDF5 and NetCDF and read it back,
  - how to plot f-independent basics (B on the surface, fluxes per radial coordinate).

Physics context: SFINCS solves the radially local, linearized drift-kinetic
equation for the non-Maxwellian part of the distribution function on one flux
surface, giving neoclassical particle/heat fluxes and parallel flows [M.
Landreman, H. M. Smith, A. Mollen and P. Helander, "Comparison of particle
trajectories and collision operators for collisional transport in
nonaxisymmetric plasmas", Phys. Plasmas 21, 042503 (2014); SFINCS technical
documentation, https://github.com/landreman/sfincs].  Here the geometry is a
concentric circular-cross-section tokamak (geometryScheme=1 with zero helical
ripple, so Nzeta=1), one ion species, and pitch-angle-scattering collisions.
In an axisymmetric field, neoclassical theory in the banana regime predicts a
small radial particle flux driven by the density/temperature gradients and a
parallel (bootstrap-like) flow; the printed table shows both.

Expected runtime: ~5 s on a laptop CPU (a few seconds of JAX compilation
dominates the tiny solve itself).

Run:
  python examples/run_tokamak.py
"""

import os
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dkx.run import run_profile

# ----------------------------------------------------------------------------
# Parameters (all inputs live at the top of the script)
# ----------------------------------------------------------------------------
CI = os.environ.get("DKX_CI") == "1"  # shrink resolution for CI

# Resolution (Fortran names; Nzeta=1 because the field is axisymmetric).
N_THETA = 9 if CI else 15
N_XI = 6 if CI else 8
N_L = 2 if CI else 4
N_X = 4 if CI else 6

# Tokamak surface: geometryScheme=1 is the SFINCS three-helicity model
# BHat = B0OverBBar * (1 + epsilon_t cos(theta) + epsilon_h cos(l theta - n N zeta));
# epsilon_h = 0 makes it a pure circular tokamak with inverse aspect ratio 0.1.
GEOMETRY = {
    "geometryScheme": 1,
    "inputRadialCoordinate": 3,  # pick the surface by rN = r/a
    "rN_wish": 0.3,
    "B0OverBBar": 1.0,
    "GHat": 1.0,  # toroidal covariant field component (B_zeta)
    "IHat": 0.0,  # poloidal covariant field component (B_theta)
    "iota": 1.31,  # rotational transform
    "epsilon_t": 0.1,  # toroidal ripple (inverse aspect ratio)
    "epsilon_h": 0.0,  # no helical ripple: a tokamak
    "helicity_l": 1,
    "helicity_n": 1,
    "psiAHat": 0.045,
    "aHat": 0.1,
}

# One hydrogen-like ion species with normalized gradients.
SPECIES = {
    "Zs": [1],
    "mHats": [1.0],
    "nHats": [1.0],
    "THats": [0.5],
    "dNHatdrHats": [-6.0],
    "dTHatdrHats": [-3.0],
}

PHYSICS = {
    "Delta": 4.5694e-3,  # rho* at the reference values
    "alpha": 1.0,
    "nu_n": 8.4774e-3,  # normalized collisionality
    "Er": 0.0,
    "collisionOperator": 1,  # 1 = pure pitch-angle scattering (PAS)
    "includeXDotTerm": True,
    "includeElectricFieldTermInXiDot": True,
    "useDKESExBDrift": False,
    "includePhi1": False,
}

OUT_DIR = Path(__file__).parent / "output"
DECK_PATH = OUT_DIR / "run_tokamak.input.namelist"
H5_PATH = OUT_DIR / "run_tokamak.sfincsOutput.h5"
NC_PATH = OUT_DIR / "run_tokamak.sfincsOutput.nc"
PLOT_PATH = OUT_DIR / "run_tokamak.png"

# ----------------------------------------------------------------------------
# 1) Build the input.namelist from the dicts above
# ----------------------------------------------------------------------------
print("=== examples/run_tokamak.py ===")
print("Step 1: building a SFINCS input.namelist from Python dicts")
print(f"  resolution: Ntheta={N_THETA} Nzeta=1 Nxi={N_XI} NL={N_L} Nx={N_X} (CI={CI})")
print(f"  geometry:   circular tokamak, epsilon_t={GEOMETRY['epsilon_t']}, "
      f"iota={GEOMETRY['iota']}, rN={GEOMETRY['rN_wish']}")
print(f"  species:    Z={SPECIES['Zs'][0]} nHat={SPECIES['nHats'][0]} THat={SPECIES['THats'][0]} "
      f"dnHat/drHat={SPECIES['dNHatdrHats'][0]} dTHat/drHat={SPECIES['dTHatdrHats'][0]}")
print(f"  physics:    PAS collisions, nu_n={PHYSICS['nu_n']}, Er={PHYSICS['Er']}")


def fortran_value(val) -> str:
    """Render one Python value in Fortran-namelist syntax."""
    if isinstance(val, bool):
        return ".true." if val else ".false."
    if isinstance(val, list):
        return " ".join(fortran_value(v) for v in val)
    return repr(val)


groups = {
    "general": {},
    "geometryParameters": GEOMETRY,
    "speciesParameters": SPECIES,
    "physicsParameters": PHYSICS,
    "resolutionParameters": {
        "Ntheta": N_THETA, "Nzeta": 1, "Nxi": N_XI, "NL": N_L, "Nx": N_X,
        "solverTolerance": 1e-10,
    },
    "otherNumericalParameters": {"Nxi_for_x_option": 0},
    "preconditionerOptions": {},
}
deck_lines = ["! SFINCS input generated by examples/run_tokamak.py"]
for group, params in groups.items():
    deck_lines.append(f"&{group}")
    deck_lines.extend(f"  {key} = {fortran_value(val)}" for key, val in params.items())
    deck_lines.append("/")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DECK_PATH.write_text("\n".join(deck_lines) + "\n")
print(f"  wrote namelist: {DECK_PATH}")

# ----------------------------------------------------------------------------
# 2) Solve the drift-kinetic equation (run_profile prints the Fortran-parity
#    console flow: banner, grids, solve progress, per-species results table)
# ----------------------------------------------------------------------------
print("Step 2: solving the drift-kinetic equation with run_profile()")
run = run_profile(DECK_PATH, solve_method="auto", out_path=H5_PATH)
print(f"  solver tier used: {run.solve_result.method} "
      f"(residual norm {float(np.max(np.asarray(run.solve_result.residual_norms))):.3e})")

# Also write NetCDF: the writer picks the format from the file suffix.
run_nc = run_profile(DECK_PATH, solve_method="auto", out_path=NC_PATH, emit=None)
print(f"  wrote outputs: {run.output_path} and {run_nc.output_path}")

# ----------------------------------------------------------------------------
# 3) Read the HDF5 output back and plot f-independent basics
# ----------------------------------------------------------------------------
print("Step 3: reading the HDF5 output back and plotting")
with h5py.File(H5_PATH, "r") as f:
    theta = np.asarray(f["theta"][...], dtype=np.float64)
    b_hat = np.asarray(f["BHat"][...], dtype=np.float64).reshape(theta.size, -1)
    # Radial fluxes are stored in all four radial-coordinate conventions:
    # the flux of a species through grad(y) scales by dy/dpsiHat.
    flux_variants = {
        coord: float(np.asarray(f[f"particleFlux_vm_{coord}"][...]).reshape(-1)[0])
        for coord in ("psiHat", "psiN", "rHat", "rN")
    }
    fsab_flow = float(np.asarray(f["FSABFlow"][...]).reshape(-1)[0])
    heat_flux = float(np.asarray(f["heatFlux_vm_psiHat"][...]).reshape(-1)[0])
print(f"  read back from h5: FSABFlow = {fsab_flow:.6e}")
print(f"  read back from h5: heatFlux_vm_psiHat = {heat_flux:.6e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax1.plot(theta, b_hat[:, 0], "o-", color="tab:blue")
ax1.set_xlabel(r"$\theta$")
ax1.set_ylabel(r"$\hat{B}$")
ax1.set_title(r"$\hat B(\theta)$: tokamak $1+\epsilon_t\cos\theta$ well")
names = list(flux_variants)
ax2.bar(names, [abs(flux_variants[n]) for n in names], color="tab:orange")
ax2.set_yscale("log")
ax2.set_ylabel(r"$|\Gamma|$ (normalized)")
ax2.set_title("particle flux per radial coordinate")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Final results
# ----------------------------------------------------------------------------
gamma = float(np.asarray(run.moments["particleFlux_vm_psiHat"])[0])
q_flux = float(np.asarray(run.moments["heatFlux_vm_psiHat"])[0])
print("=== Final results ===")
print(f"  particleFlux_vm_psiHat = {gamma:.6e}")
print(f"  heatFlux_vm_psiHat     = {q_flux:.6e}")
print(f"  FSABFlow               = {float(np.asarray(run.moments['FSABFlow'])[0]):.6e}")
print(f"  FSABjHat               = {float(np.asarray(run.moments['FSABjHat'])):.6e}")
print(f"  Saved plot: {PLOT_PATH}")
print(f"  Wrote output files: {H5_PATH.name}, {NC_PATH.name}")
print("Done: examples/run_tokamak.py")
