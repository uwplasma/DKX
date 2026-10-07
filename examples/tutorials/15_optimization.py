"""Tutorial 15 -- a small optimization loop: reshape |B| to lower the neoclassical flux.

Tutorial 14 differentiated a plasma parameter; stellarator design needs the
derivative with respect to the *magnetic geometry*.  Here the surface is a
three-harmonic Boozer spectrum

    B(theta, zeta) = B0 [1 - eps_t cos(theta) + eps_h cos(theta - N zeta)],

and the design variable is the helical ripple eps_h.  The traced chain is
eps_h -> |B| and metric on the grid (``FluxSurfaceGeometry.from_fourier``) ->
drift-kinetic operator -> linear solve -> radial particle flux Gamma, so
``jax.value_and_grad`` returns dGamma/deps_h at the cost of about two solves.
Plain gradient descent, eps_h <- eps_h - lr * dGamma/deps_h, then lowers
the flux: removing helical ripple shrinks the helically trapped population
that drives 1/nu transport.  The first gradient is checked against a central
difference.  In a real design loop the amplitudes come from a VMEC/VMEX
boundary through ``booz_xform_jax``; ``examples/advanced/`` does that.

Run:              python examples/tutorials/15_optimization.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/15_optimization.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
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
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
EPS_H_START = 0.05  # initial helical ripple (the design variable)
LEARNING_RATE = 10.0
N_STEPS = 2 if SMOKE else 8
FD_STEP = 1.0e-6
N_PERIODS, EPS_T, IOTA, G_HAT = 5, 0.07, 0.87, 3.7481
M_MODES, N_MODES = jnp.array([0.0, 1.0, 1.0]), jnp.array([0.0, 0.0, 1.0])  # (0,0), (1,0), (1,1)
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, B0OverBBar=1.0, epsilon_t=-EPS_T,
    epsilon_h=EPS_H_START, helicity_l=1, helicity_n=1, Nperiods=N_PERIODS, iota=IOTA, GHat=G_HAT,
    IHat=0.0, psiAHat=0.15596, aHat=0.5585, Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0], collisionOperator=1, Delta=4.5694e-3, alpha=1.0,
    nu_n=8.330e-3, Ntheta=11, Nzeta=11, Nxi=10 if SMOKE else 16, NL=4, Nx=4,
)
# Operator fields that the geometry owns; replacing only these keeps the
# collision operator and grids fixed, so the derivative is a pure shape derivative.
GEOMETRY_LEAVES = ("b_hat", "db_hat_dtheta", "db_hat_dzeta", "d_hat",
                   "b_hat_sup_theta", "b_hat_sup_zeta", "b_hat_sub_theta", "b_hat_sub_zeta")

# 3. Run
print("building the operator", flush=True)
operator = dkx.run(**PARAMETERS, emit=print).operator
theta = jnp.linspace(0.0, 2 * jnp.pi, operator.n_theta, endpoint=False)
zeta = jnp.linspace(0.0, 2 * jnp.pi / N_PERIODS, operator.n_zeta, endpoint=False)


def particle_flux(eps_h):
    """Normalized radial particle flux as a differentiable function of eps_h."""
    surface = FluxSurfaceGeometry.from_fourier(
        theta=theta, zeta=zeta, bmnc=jnp.stack([jnp.asarray(1.0), jnp.asarray(-EPS_T), eps_h]),
        m=M_MODES, n=N_MODES, n_periods=N_PERIODS, iota=IOTA, g_hat=G_HAT, i_hat=0.0)
    leaves = {name: getattr(surface, name) for name in GEOMETRY_LEAVES}
    leaves["fsab_hat2"] = surface.fsab_hat2(theta_weights=operator.theta_weights,
                                            zeta_weights=operator.zeta_weights)
    perturbed = replace(operator, **leaves)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True,
                   tol=1e-10, tier1_keep_lowest=perturbed.n_xi)
    return profile_moments_from_operator(perturbed, solved.x.reshape(-1))["particleFlux_vm_psiHat"][0]


value_and_grad = jax.jit(jax.value_and_grad(particle_flux))
history = []  # (eps_h, Gamma, dGamma/deps_h)
eps_h = EPS_H_START
for step in range(N_STEPS + 1):
    flux, gradient = (float(v) for v in value_and_grad(jnp.asarray(eps_h)))
    history.append((eps_h, flux, gradient))
    print(f"step {step}/{N_STEPS}: eps_h={eps_h:.5f}  Gamma={flux:.6e}  dGamma/deps_h={gradient:+.4e}", flush=True)
    eps_h = max(0.0, eps_h - LEARNING_RATE * gradient)
print("central-difference check of the first gradient", flush=True)
fd = float(particle_flux(EPS_H_START + FD_STEP) - particle_flux(EPS_H_START - FD_STEP)) / (2 * FD_STEP)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
eps, flux, grads = (np.array(column) for column in zip(*history))
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 3.8), constrained_layout=True)
left.plot(flux, "o-")
left.set(xlabel="gradient step", ylabel=r"$\Gamma$ (normalized)", title="objective")
right.plot(eps, flux, "o-", color="tab:purple")
right.set(xlabel=r"$\epsilon_h$", ylabel=r"$\Gamma$ (normalized)", title="path through design space")
fig.savefig(OUT / "optimization.png", dpi=120)
np.savez(OUT / "optimization.npz", eps_h=eps, flux=flux, gradient=grads, fd_gradient=fd)
print(f"\nsaved {OUT / 'optimization.npz'}\nsaved {OUT / 'optimization.png'}")
print("\n=== Summary ===")
print(f"  eps_h: {eps[0]:.5f} -> {eps[-1]:.5f}   Gamma: {flux[0]:.5e} -> {flux[-1]:.5e}"
      f"  ({1 - flux[-1] / flux[0]:.1%} lower)")
print(f"  first gradient: jax.grad {grads[0]:+.8e}  central difference {fd:+.8e}"
      f"  relative difference {abs(grads[0] - fd) / abs(fd):.1e}")
