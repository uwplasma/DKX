"""Tutorial 14 -- exact derivatives through the kinetic solve with ``jax.grad``.

DKX is written in JAX, so an observable O(p) = M[f1(p)] that depends on a
parameter p through the linear solve  A(p) f1 = b(p)  can be differentiated
exactly.  Reverse mode uses the adjoint: one extra *transposed* solve
A^T lambda = dM/df1 gives

    dO/dp = dM/dp + lambda^T (db/dp - dA/dp f1),

at the cost of about one solve however many parameters p there are -- the
reason gradient-based optimization (tutorial 15) is affordable.  Here O is
the bootstrap current <j.B> of a single-species PAS tokamak case and p the
temperature T; the pitch-angle collision frequency nu_D(T) is rebuilt inside
the function, so dO/dT includes the collisionality change.  The gradient is
checked against central finite differences at three step sizes, and the
figure draws the jax.grad tangent on a T sweep.

Run:              python examples/tutorials/14_gradients.py
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/14_gradients.py
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
from dkx.collisions import make_pitch_angle_scattering_v3_operator
from dkx.run import profile_moments_from_operator
from dkx.solve import solve

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
PARAMETERS = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3, B0OverBBar=1.0, epsilon_t=-0.07,
    epsilon_h=0.0, iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0], dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
    Ntheta=11, Nzeta=1, Nxi=12 if SMOKE else 24, NL=4, Nx=5,
)
FD_STEPS = 1.0e-5 * np.array([1.0, 3.0, 10.0])  # central-difference steps in THat
SWEEP = np.array([0.8, 1.0, 1.2] if SMOKE else [0.8, 0.9, 1.0, 1.1, 1.2])  # THat values

# 3. Run
print("building the operator", flush=True)
operator = dkx.run(**PARAMETERS, emit=print).operator


@jax.jit
def bootstrap_current(t_hat):
    """<j.B> (normalized) as a differentiable function of the temperature."""
    temperature = jnp.reshape(t_hat, (1,))
    pas = make_pitch_angle_scattering_v3_operator(
        x=operator.x, z_s=operator.z_s, m_hats=operator.m_hat, n_hats=operator.n_hat,
        t_hats=temperature, nu_n=operator.pas.nu_n, krook=operator.pas.krook,
        n_xi_for_x=operator.n_xi_for_x, n_xi=operator.n_xi)
    perturbed = replace(operator, t_hat=temperature, pas=pas)
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]


t0 = float(operator.t_hat[0])
print("jax.grad (forward solve + one adjoint solve)", flush=True)
value, gradient = (float(v) for v in jax.value_and_grad(bootstrap_current)(jnp.asarray(t0)))
print("central finite differences", flush=True)
fd = np.array([float(bootstrap_current(t0 + h) - bootstrap_current(t0 - h)) / (2 * h) for h in FD_STEPS])
print("temperature sweep", flush=True)
sweep = np.array([float(bootstrap_current(t)) for t in SWEEP * t0])

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
fig, axis = plt.subplots(figsize=(6, 3.8), constrained_layout=True)
axis.plot(SWEEP * t0, sweep, "o-", label=r"$\langle j\cdot B\rangle(\hat T)$")
axis.plot(SWEEP * t0, value + gradient * (SWEEP * t0 - t0), "--", label="tangent from jax.grad")
axis.set(xlabel=r"$\hat T$", ylabel=r"$\langle j\cdot B\rangle$ (normalized)")
axis.legend()
fig.savefig(OUT / "gradients.png", dpi=120)
np.savez(OUT / "gradients.npz", t_hat=SWEEP * t0, current=sweep, gradient=gradient, fd_steps=FD_STEPS, fd=fd)
print(f"\nsaved {OUT / 'gradients.npz'}\nsaved {OUT / 'gradients.png'}")
relative = np.max(np.abs(gradient - fd) / np.abs(fd))
print("\n=== Summary ===")
print(f"  <j.B> at THat={t0:.3f}: {value:+.8e}")
print(f"  jax.grad            d<j.B>/dTHat = {gradient:+.10e}")
print(f"  central differences d<j.B>/dTHat = {np.array2string(fd, precision=10)}")
print(f"  max relative difference          = {relative:.2e}")
