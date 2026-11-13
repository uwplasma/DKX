"""Shape |B| to push a trace carbon impurity outward: temperature screening by optimization.

A highly charged impurity (C6+) tends to accumulate in a stellarator core,
pulled in by the main-ion density gradient.  Temperature screening is the
competing effect, in which the main-ion temperature gradient drives it out
(Helander & Sigmar 2002; Newton et al., J. Plasma Phys. 83, 2017).  Which one
wins depends on the magnetic geometry, so it is an optimization target.

Three species (deuterium, electrons, C6+) with the linearized Fokker-Planck
operator couple the impurity flux to inter-species momentum exchange.  The
chain is helical |B| amplitude -> operator -> differentiable solve -> impurity
flux Gamma_z, as in tutorial 15, so ``jax.value_and_grad`` gives the gradient
of Gamma_z in the shape; normalized gradient ascent then raises it (Gamma_z > 0
is outward).  The script checks the first gradient against a central
difference and prints the temperature-screening coefficient, the sensitivity
of Gamma_z to the main-ion temperature gradient, also from autodiff.

The electron-root counterpart, shaping |B| to steer the ambipolar E_r, is the
differentiable root of ``ambipolar_bootstrap_dkx.py`` and ``dkx.er``.

Run:              python examples/advanced/impurity_screening_optimization.py
Runtime:          about 258 s on a loaded 36-core CPU node, compilation included.
"""

# 1. Imports
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import dkx
from dkx.magnetic_geometry import FluxSurfaceGeometry
from dkx.run import profile_moments_from_operator
from dkx.solve import solve

# 2. Input parameters
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
EPS_H_START = 0.05  # initial helical ripple (the design variable)
STEP = 0.01  # change of eps_h per ascent step (the gradient sets only the sign)
N_STEPS = 2  # ascent steps; the coarse-grid Krylov adjoint stops converging beyond eps_h = 0.07
FD_STEP = 1.0e-6  # central-difference step for the gradient check
IMPURITY = 2  # species index of C6+
N_PERIODS, EPS_T, IOTA, G_HAT = 5, 0.07, 0.87, 3.7481  # field periods, toroidal ripple, transform, G
M_MODES, N_MODES = jnp.array([0.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 1.0])  # (0,0), (1,0), (1,1)
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, B0OverBBar=1.0, epsilon_t=-EPS_T,
    epsilon_h=EPS_H_START, helicity_l=1, helicity_n=1, Nperiods=N_PERIODS, iota=IOTA, GHat=G_HAT,
    IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0, -1.0, 6.0], mHats=[2.0, 5.446170214e-4, 12.0], nHats=[1.0, 1.06, 0.01], THats=[1.0, 1.0, 1.0],
    dNHatdrHats=[-1.0, -1.0, -0.5], dTHatdrHats=[-2.0, -2.0, -2.0],  # drives; the ion one is screened
    collisionOperator=0, Er=-1.0, Delta=4.5694e-3, alpha=1.0, nu_n=0.03,  # Fokker-Planck at a fixed ion-root field
    Ntheta=9, Nzeta=9, Nxi=10, NL=4, Nx=4,  # coarse grid: seconds, not converged
)
# Operator fields the geometry owns; replacing only these keeps the collision
# operator and grids fixed, so the derivative is a pure shape derivative.
GEOMETRY_LEAVES = ("b_hat", "db_hat_dtheta", "db_hat_dzeta", "d_hat",
                   "b_hat_sup_theta", "b_hat_sup_zeta", "b_hat_sub_theta", "b_hat_sub_zeta")

# 3. Run
print("building the operator", flush=True)
operator = dkx.run(**PARAMETERS, emit=print).operator
theta = jnp.linspace(0.0, 2 * jnp.pi, operator.n_theta, endpoint=False)
zeta = jnp.linspace(0.0, 2 * jnp.pi / N_PERIODS, operator.n_zeta, endpoint=False)


def impurity_flux(eps_h, temperature_gradient_scale=1.0):
    """Normalized C6+ radial particle flux as a differentiable function of eps_h."""
    surface = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=jnp.stack([jnp.asarray(1.0), jnp.asarray(-EPS_T), eps_h]),
        m=M_MODES, n=N_MODES, n_periods=N_PERIODS, iota=IOTA, g_hat=G_HAT, i_hat=0.0)
    leaves = {name: getattr(surface, name) for name in GEOMETRY_LEAVES}
    leaves["fsab_hat2"] = surface.fsab_hat2(theta_weights=operator.theta_weights,
                                            zeta_weights=operator.zeta_weights)
    leaves["dt_hat_dpsi_hat"] = operator.dt_hat_dpsi_hat.at[0].multiply(temperature_gradient_scale)
    perturbed = replace(operator, **leaves)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True,
                   tol=1e-9, tier1_keep_lowest=perturbed.n_xi)
    return profile_moments_from_operator(perturbed, solved.x.reshape(-1))["particleFlux_vm_psiHat"][IMPURITY]


value_and_grad = jax.jit(jax.value_and_grad(impurity_flux))
history = []  # (eps_h, Gamma_z, dGamma_z/deps_h)
eps_h = EPS_H_START
for step in range(N_STEPS + 1):
    flux, gradient = (float(v) for v in value_and_grad(jnp.asarray(eps_h)))
    history.append((eps_h, flux, gradient))
    print(f"step {step}/{N_STEPS}: eps_h={eps_h:.4f}  Gamma_z={flux:+.5e}  dGamma_z/deps_h={gradient:+.4e}", flush=True)
    eps_h = max(0.0, eps_h + STEP * np.sign(gradient))
print("central-difference check of the first gradient", flush=True)
fd = float(impurity_flux(EPS_H_START + FD_STEP) - impurity_flux(EPS_H_START - FD_STEP)) / (2 * FD_STEP)
print("temperature-screening coefficient from jax.grad", flush=True)
screening = float(jax.grad(lambda scale: impurity_flux(jnp.asarray(EPS_H_START), scale))(1.0))

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
eps, flux, grads = (np.array(column) for column in zip(*history))
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
left.plot(flux, "o-")
left.set(xlabel="ascent step", ylabel=r"$\Gamma_z$ (C6+, normalized)", title="objective (up = more screening)")
right.plot(eps, flux, "o-", color="tab:purple")
right.set(xlabel=r"$\epsilon_h$", ylabel=r"$\Gamma_z$", title="path through design space")
fig.savefig(OUT / "impurity_screening.png", dpi=120)
np.savez(OUT / "impurity_screening.npz", eps_h=eps, flux=flux, gradient=grads, fd_gradient=fd, screening=screening)
print(f"\nsaved {OUT / 'impurity_screening.png'}")
print("\n=== Summary ===")
print(f"  eps_h: {eps[0]:.4f} -> {eps[-1]:.4f}   Gamma_z: {flux[0]:+.4e} -> {flux[-1]:+.4e}"
      f"  ({'outward' if flux[-1] > 0 else 'inward'} at the end)")
print(f"  first gradient: jax.grad {grads[0]:+.8e}  central difference {fd:+.8e}"
      f"  relative difference {abs(grads[0] - fd) / abs(fd):.1e}")
print(f"  temperature-screening coefficient d Gamma_z / d(scale on the ion dT/dr) = {screening:+.4e}")
