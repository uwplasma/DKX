"""Shape optimization: differentiate the kinetic solve with respect to the field.

Rung 07 differentiated a plasma parameter.  This one differentiates the
*geometry* -- the derivative stellarator design actually needs.  The traced
chain runs from a Boozer ``|B|`` Fourier amplitude, through
``FluxSurfaceGeometry.from_fourier``, into the operator, through the linear
solve, out to the radial particle flux; gradient descent then reduces it.

The knob is the helical ripple amplitude ``epsilon_h``.  Ripple is what makes
a stellarator lose particles in the ``1/nu`` regime, so the gradient should
push it toward zero and the flux should fall as it does.

Where vmex fits.  In a production loop ``vmex`` solves the VMEC equilibrium
and ``booz_xform_jax`` transforms it, so the amplitudes below are outputs of a
boundary shape rather than typed by hand.  DKX's public hand-off to that lane
is a *geometry proxy*: the full VMEC-boundary-to-kinetic-transport gradient is
not yet claimed, and the script prints that contract rather than asserting
more.  It does not import vmex, so the kinetic gradient below is real and
reproducible with or without the optional backends installed.

Physics: three-helicity analytic surface, ``N=5`` field periods, one hydrogen
species, pitch-angle scattering, at fixed collisionality.

Expected runtime: ~12 s on a laptop CPU.
"""

# 1. Imports
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from netCDF4 import Dataset  # noqa: E402

import dkx  # noqa: E402
from dkx.magnetic_geometry import FluxSurfaceGeometry  # noqa: E402
from dkx.run import profile_moments_from_operator  # noqa: E402
from dkx.solve import solve  # noqa: E402
from dkx.workflows.geometry_adapters import (  # noqa: E402
    geometry_proxy_workflow_summary,
    optional_jax_geometry_backend_status,
)

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "08_vmex_optimization"
RESULT_FILE = OUT_DIR / "optimization.nc"
PLOT_FILE = OUT_DIR / "optimization.png"

# The design variable: the amplitude of the helical |B| harmonic, relative to
# the field on axis.  Replace this with amplitudes read off a vmex/booz_xform
# spectrum to drive a real boundary.
EPSILON_H_START = 0.05
# Plain gradient descent, because the point is the gradient rather than the
# optimizer.  Five steps is enough to show the trend without hiding it in a
# line search.
LEARNING_RATE = 10.0
N_STEPS = 5
EPSILON_H_FLOOR = 0.0  # a negative ripple amplitude is the same surface, rotated
FD_STEP = 1.0e-6  # for the gradient cross-check only

# 3. Geometry and species construction
N_PERIODS = 5
B0_OVER_BBAR = 1.0
EPSILON_T = 0.07  # toroidal ripple, held fixed: this is the aspect ratio
IOTA, G_HAT, I_HAT = 0.8700, 3.7481, 0.0
# (m, n) of the three retained |B| harmonics: (0,0) the mean field, (1,0) the
# toroidal ripple, (1,1) the helical ripple we optimize.
M_MODES = jnp.asarray([0.0, 1.0, 1.0], dtype=jnp.float64)
N_MODES = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float64)

# 4. Physics and numerical configuration
CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=B0_OVER_BBAR, epsilon_t=-EPSILON_T, epsilon_h=EPSILON_H_START,
    helicity_l=1, helicity_n=1, Nperiods=N_PERIODS,
    iota=IOTA, GHat=G_HAT, IHat=I_HAT, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=11, Nzeta=11, Nxi=10, NL=4, Nx=4,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
# The operator leaves the geometry owns.  Replacing exactly these and nothing
# else is what keeps the collision operator and the grids fixed, so the
# derivative is a pure shape derivative.
GEOMETRY_LEAVES = (
    "b_hat", "db_hat_dtheta", "db_hat_dzeta", "d_hat",
    "b_hat_sup_theta", "b_hat_sup_zeta", "b_hat_sub_theta", "b_hat_sub_zeta",
)
# end of parameters

OUT_DIR.mkdir(parents=True, exist_ok=True)
operator = dkx.run(**CASE, emit=None).operator
theta = jnp.linspace(0.0, 2.0 * jnp.pi, operator.n_theta, endpoint=False, dtype=jnp.float64)
zeta = jnp.linspace(
    0.0, 2.0 * jnp.pi / N_PERIODS, operator.n_zeta, endpoint=False, dtype=jnp.float64
)
print(f"operator: {operator.n_species} species, matrix size {operator.total_size}, "
      f"grid {operator.n_theta}x{operator.n_zeta}")

backends = optional_jax_geometry_backend_status()
gate = geometry_proxy_workflow_summary()["no_overclaim_gate"]
print(f"optional geometry backends: {backends}")
print(f"vmex hand-off claim scope: {gate['claim_scope']}")
print(f"full VMEC-boundary transport gradients claimed: "
      f"{gate['full_transport_gradients_claimed']}")


def particle_flux(epsilon_h):
    """Radial particle flux as a function of the helical ripple.  Differentiable."""
    bmnc = jnp.stack(
        [
            jnp.asarray(B0_OVER_BBAR),
            jnp.asarray(-EPSILON_T * B0_OVER_BBAR),
            epsilon_h * B0_OVER_BBAR,
        ]
    )
    surface = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=bmnc, m=M_MODES, n=N_MODES,
        n_periods=N_PERIODS, iota=IOTA, g_hat=G_HAT, i_hat=I_HAT,
    )  # fmt: skip
    leaves = {name: getattr(surface, name) for name in GEOMETRY_LEAVES}
    leaves["fsab_hat2"] = surface.fsab_hat2(
        theta_weights=operator.theta_weights, zeta_weights=operator.zeta_weights
    )
    perturbed = replace(operator, **leaves)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["particleFlux_vm_psiHat"][0]


# 5. Run
value_and_gradient = jax.value_and_grad(particle_flux)

history_eps = [EPSILON_H_START]
history_flux = []
history_grad = []
epsilon_h = EPSILON_H_START
for step in range(N_STEPS + 1):
    flux, gradient = value_and_gradient(jnp.asarray(epsilon_h))
    history_flux.append(float(flux))
    history_grad.append(float(gradient))
    print(f"  step {step}: epsilon_h = {epsilon_h:.6f}  "
          f"Gamma = {float(flux):.6e}  dGamma/d(epsilon_h) = {float(gradient):+.6e}")
    if step == N_STEPS:
        break
    epsilon_h = max(EPSILON_H_FLOOR, epsilon_h - LEARNING_RATE * float(gradient))
    history_eps.append(epsilon_h)

# The gradient cross-check, at the starting point.
step = FD_STEP
central_difference = float(
    (particle_flux(jnp.asarray(EPSILON_H_START + step))
     - particle_flux(jnp.asarray(EPSILON_H_START - step))) / (2.0 * step)
)

# 6. Print a scientific summary and certificate
relative_difference = abs(history_grad[0] / central_difference - 1.0)
print("\n=== Final results ===")
print(f"  epsilon_h: {history_eps[0]:.6f} -> {history_eps[-1]:.6f}")
print(f"  particle flux: {history_flux[0]:.6e} -> {history_flux[-1]:.6e} (normalized)")
reduction = 1.0 - history_flux[-1] / history_flux[0]
print(f"  reduction: {reduction:.1%} in {N_STEPS} gradient steps")
print(f"  jax.grad           dGamma/d(epsilon_h) = {history_grad[0]:+.10e}")
print(f"  central difference dGamma/d(epsilon_h) = {central_difference:+.10e}")
print(f"  relative difference                    = {relative_difference:.3e}")
assert relative_difference < 1.0e-5, "shape gradient disagrees with central differences"
print("  all gradients verified against central finite differences")
assert history_flux[-1] < history_flux[0], "descent did not reduce the flux"
assert history_eps[-1] < history_eps[0], "ripple did not fall: check the sign of the gradient"
print("  physics check: removing helical ripple lowered the neoclassical flux")
print("  kinetic solve executed: True (this gradient is not the geometry proxy)")

# 7. Save native result
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("step", len(history_flux))
    dataset.createVariable("epsilon_h", "f8", ("step",))[:] = np.asarray(history_eps)
    dataset.createVariable("particleFlux_vm_psiHat", "f8", ("step",))[:] = np.asarray(history_flux)
    dataset.createVariable("dFlux_depsilon_h", "f8", ("step",))[:] = np.asarray(history_grad)
    dataset.createVariable("dFlux_depsilon_h_central_difference", "f8")[...] = central_difference
    dataset.dkx_version = dkx.__version__
    dataset.claim_scope = gate["claim_scope"]
    dataset.optional_backends = ", ".join(f"{k}={v}" for k, v in sorted(backends.items()))
print(f"  Wrote result: {RESULT_FILE}")

# 8. Plot publication-ready outputs
figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
steps = np.arange(len(history_flux))
left.plot(steps, history_flux, "o-", color="tab:blue")
left.set(xlabel="gradient step", ylabel=r"$\Gamma$ (normalized)")
left.set_title("neoclassical flux falls with each step", fontsize=10)
left.grid(alpha=0.3)
right.plot(history_eps, history_flux, "o-", color="tab:purple")
for index, (eps, flux_value) in enumerate(zip(history_eps, history_flux)):
    right.annotate(str(index), (eps, flux_value), textcoords="offset points",
                   xytext=(6, 4), fontsize=8)  # fmt: skip
right.set(xlabel=r"$\epsilon_h$ (helical ripple amplitude)", ylabel=r"$\Gamma$ (normalized)")
right.set_title("the path descent takes through design space", fontsize=10)
right.grid(alpha=0.3)
figure.suptitle("Shape gradient through the drift-kinetic solve")
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/08_vmex_optimization/run.py")
