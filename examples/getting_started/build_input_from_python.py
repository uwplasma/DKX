"""Build a typed SFINCS input in Python and solve it in memory — no input file.

What this example teaches:
  - how to build a typed input deck programmatically with
    ``SfincsInput.from_params`` (flat Fortran parameter names such as
    ``Ntheta``, ``geometryScheme``, ``nHats`` — matched case-insensitively),
  - how to serialize it back to a Fortran-readable ``input.namelist`` with
    ``to_namelist()``/``write()`` (compact by default; ``include_defaults=True``
    spells out every typed field), and that the round trip is lossless,
  - how to pick solver tuning with the typed ``SolverOptions`` knob set and run
    ``run_profile`` directly on the in-memory input (no temp files),
  - that the in-memory run and the written-deck run agree exactly.

Physics context: SFINCS solves the radially local, linearized drift-kinetic
equation for the non-Maxwellian part of the distribution function on one flux
surface, giving neoclassical particle/heat fluxes and parallel flows [M.
Landreman, H. M. Smith, A. Mollen and P. Helander, Phys. Plasmas 21, 042503
(2014)].  Here the geometry is a concentric circular-cross-section tokamak
(geometryScheme=1 with zero helical ripple, so Nzeta=1), one ion species, and
pitch-angle-scattering collisions.  Scanning the normalized collisionality
``nu_n`` moves the surface across collisionality regimes: the radial particle
flux and the parallel (bootstrap-like) flow both respond, and the plot shows
the resulting trends.

Expected runtime: ~10 s on a laptop CPU (a few seconds of JAX compilation,
then a handful of tiny solves that reuse the compiled kernels).

Run:
  python examples/getting_started/build_input_from_python.py
"""

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dkx import SfincsInput, SolverOptions, run_profile
from dkx.inputs import parse_sfincs_input_text, sfincs_input_from_raw

# ----------------------------------------------------------------------------
# Parameters (all inputs live at the top of the script)
# ----------------------------------------------------------------------------
CI = os.environ.get("DKX_CI") == "1"  # shrink resolution for CI

# Resolution (Fortran names; Nzeta=1 because the field is axisymmetric).
N_THETA = 9 if CI else 15
N_XI = 6 if CI else 8
N_L = 2 if CI else 4
N_X = 4 if CI else 6

# Normalized collisionality scan (nu_n at the reference parameters).
NU_N_SCAN = [3.0e-3, 3.0e-2] if CI else [3.0e-3, 1.0e-2, 3.0e-2, 1.0e-1]

# Solver tuning: the typed knob set threaded to dkx.solve.solve.  "auto"
# picks the tier-1 structured direct path for this PAS deck; tol is the
# relative residual tolerance.  (SolverOptions also carries atol/restart/
# recycle_dim/max_restarts/differentiable/use_preconditioner/device/
# memory_budget_gb for the Krylov and memory-budget paths.)
SOLVER = SolverOptions(method="auto", tol=1.0e-10)

OUT_DIR = Path(__file__).resolve().parents[1] / "output"
DECK_PATH = OUT_DIR / "build_input_from_python.input.namelist"
PLOT_PATH = OUT_DIR / "build_input_from_python.png"


def make_input(nu_n: float) -> SfincsInput:
    """One typed tokamak deck at collisionality ``nu_n`` (flat Fortran names)."""
    return SfincsInput.from_params(
        # Geometry: circular tokamak, |B| = B0*(1 + epsilon_t*cos(theta)).
        geometryScheme=1,
        inputRadialCoordinate=3,  # pick the surface by rN = r/a
        rN_wish=0.3,
        B0OverBBar=1.0,
        GHat=1.0,  # toroidal covariant field component (B_zeta)
        IHat=0.0,  # poloidal covariant field component (B_theta)
        iota=1.31,  # rotational transform
        epsilon_t=0.1,  # inverse aspect ratio (toroidal ripple)
        epsilon_h=0.0,  # no helical ripple: a tokamak
        helicity_l=1,
        helicity_n=1,
        psiAHat=0.045,
        aHat=0.1,
        # One hydrogen-like ion species with normalized gradients.
        Zs=[1.0],
        mHats=[1.0],
        nHats=[1.0],
        THats=[0.5],
        dNHatdrHats=[-6.0],
        dTHatdrHats=[-3.0],
        # Physics: pure pitch-angle-scattering collisions, no radial E field.
        nu_n=nu_n,
        Er=0.0,
        collisionOperator=1,
        # Resolution.
        Ntheta=N_THETA,
        Nzeta=1,
        Nxi=N_XI,
        NL=N_L,
        Nx=N_X,
        solverTolerance=1.0e-10,
        Nxi_for_x_option=0,
    )


# ----------------------------------------------------------------------------
# 1) Build the typed input and check the namelist round trip
# ----------------------------------------------------------------------------
print("=== examples/getting_started/build_input_from_python.py ===")
print("Step 1: building a typed SfincsInput with from_params (flat Fortran names)")
inp = make_input(NU_N_SCAN[0])
print(f"  resolution: Ntheta={inp.resolution.n_theta} Nzeta={inp.resolution.n_zeta} "
      f"Nxi={inp.resolution.n_xi} NL={inp.resolution.n_l} Nx={inp.resolution.n_x} (CI={CI})")
print(f"  species:    Z={inp.species.z_s[0]:g} nHat={inp.species.n_hats[0]:g} "
      f"THat={inp.species.t_hats[0]:g}")

# to_namelist() writes only the non-default fields (compact); parsing the text
# back reproduces every typed field, so the serializer is the parser's inverse.
text = inp.to_namelist()
round_tripped = sfincs_input_from_raw(parse_sfincs_input_text(text))
assert round_tripped.geometry == inp.geometry
assert round_tripped.species == inp.species
assert round_tripped.physics == inp.physics
assert round_tripped.resolution == inp.resolution
print(f"  compact namelist: {len(text.splitlines())} lines; SfincsInput round trip verified")

OUT_DIR.mkdir(parents=True, exist_ok=True)
inp.write(DECK_PATH)
print(f"  wrote namelist: {DECK_PATH}")

# ----------------------------------------------------------------------------
# 2) Solve in memory (no input file is read) and cross-check against the file
# ----------------------------------------------------------------------------
print("Step 2: run_profile on the in-memory input with SolverOptions")
run_mem = run_profile(inp, solver=SOLVER, emit=None)
print(f"  solver tier used: {run_mem.solve_result.method} "
      f"(residual norm {float(np.max(np.asarray(run_mem.solve_result.residual_norms))):.3e})")

run_file = run_profile(DECK_PATH, solver=SOLVER, emit=None)
gamma_mem = float(np.asarray(run_mem.moments["particleFlux_vm_psiHat"])[0])
gamma_file = float(np.asarray(run_file.moments["particleFlux_vm_psiHat"])[0])
assert np.isclose(gamma_mem, gamma_file, rtol=1e-12, atol=0.0)
print(f"  in-memory run matches the file-based run: particleFlux_vm_psiHat = {gamma_mem:.6e}")

# ----------------------------------------------------------------------------
# 3) Collisionality scan: rebuild the typed input per point, solve, collect
# ----------------------------------------------------------------------------
print("Step 3: nu_n scan (each point is a fresh SfincsInput, solved in memory)")
particle_flux = []
fsab_flow = []
for nu_n in NU_N_SCAN:
    run = run_profile(make_input(nu_n), solver=SOLVER, emit=None)
    particle_flux.append(float(np.asarray(run.moments["particleFlux_vm_psiHat"])[0]))
    fsab_flow.append(float(np.asarray(run.moments["FSABFlow"])[0]))
    print(f"  nu_n = {nu_n:.1e}:  particleFlux_vm_psiHat = {particle_flux[-1]:+.6e}  "
          f"FSABFlow = {fsab_flow[-1]:+.6e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.0, 3.6))
ax1.loglog(NU_N_SCAN, np.abs(particle_flux), "o-", color="tab:blue")
ax1.set_xlabel(r"$\nu_n$")
ax1.set_ylabel(r"$|\Gamma_{\psi}|$ (normalized)")
ax1.set_title("radial particle flux vs collisionality")
ax2.semilogx(NU_N_SCAN, fsab_flow, "s-", color="tab:orange")
ax2.set_xlabel(r"$\nu_n$")
ax2.set_ylabel(r"$\langle B V_\parallel \rangle$ (normalized)")
ax2.set_title("parallel flow vs collisionality")
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Final results
# ----------------------------------------------------------------------------
print("=== Final results ===")
for nu_n, gamma, flow in zip(NU_N_SCAN, particle_flux, fsab_flow):
    print(f"  nu_n = {nu_n:.1e}:  particleFlux_vm_psiHat = {gamma:+.6e}  FSABFlow = {flow:+.6e}")
print(f"  Saved plot: {PLOT_PATH}")
print(f"  Wrote namelist: {DECK_PATH.name}")
print("Done: examples/getting_started/build_input_from_python.py")
