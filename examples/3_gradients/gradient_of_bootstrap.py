"""d(bootstrap current)/d(temperature), exactly, for the price of one solve.

The solve is differentiable, so ``jax.grad`` gives an exact derivative of any
output with respect to any input.  The cost is what makes it worth doing: the
backward pass is one transposed solve reusing the factors already computed, so
the gradient of N parameters costs about the same as one -- while finite
differences cost 2N solves and give you a step-size problem for free.

The pattern is three lines: take ``run.operator``, ``dataclasses.replace`` the
leaf you want to differentiate, and solve with ``differentiable=True``.  The
operator is a registered JAX pytree, so everything downstream -- drive, matrix
apply, solve, moment integrals -- is traced.

One honesty note: the PAS collision matrices are built when the operator is
constructed, so replacing ``t_hat`` afterwards does not rebuild them.  Both
``jax.grad`` and the finite difference below differentiate that same
fixed-collisionality function, which is what makes them comparable at all.

Expected runtime: ~30 s on a laptop CPU.

Achieved: d<j.B>/dTHat = +1.2755377070e-02 at THat = 1, against
+1.2755377054e-02 from central differences -- agreement to 1.3e-09, which is
about where the difference stops being autodiff's and starts being the finite
difference's own truncation error.
"""

from dataclasses import replace

import jax
import jax.numpy as jnp

import dkx
from dkx.run import profile_moments_from_operator
from dkx.solve import solve

# --------------------------- parameters -------------------------------------
CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
    Ntheta=13, Nzeta=1, Nxi=16, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
FD_STEP = 1e-5   # central-difference step, for the check only
# ----------------------------- end of parameters ----------------------------

operator = dkx.run(**CASE).operator
print(f"operator: {operator.n_species} species, matrix size {operator.total_size}")


def bootstrap_current(t_hat):
    """<j.B> as a function of the species temperature.  Differentiable."""
    perturbed = replace(operator, t_hat=jnp.reshape(t_hat, (1,)))
    # method="auto" picks the tier the deck needs.  Naming a tier by hand is
    # how you meet "tier-1 requires uniform Nxi_for_x": this deck ramps Nxi
    # with speed, so it belongs on the truncated kernel, and "auto" knows that.
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]


t_hat_0 = float(operator.t_hat[0])
value = float(bootstrap_current(jnp.asarray(t_hat_0)))
print(f"FSABjHat at THat = {t_hat_0} is {value:+.8e}")

gradient = float(jax.grad(bootstrap_current)(jnp.asarray(t_hat_0)))

# The check.  Finite differences are not the ground truth here -- they carry
# their own truncation and round-off error -- so this establishes that the two
# agree, not that autodiff has been "validated" against something better.
step = FD_STEP * max(1.0, abs(t_hat_0))
finite_difference = float(
    (bootstrap_current(jnp.asarray(t_hat_0 + step))
     - bootstrap_current(jnp.asarray(t_hat_0 - step))) / (2.0 * step)
)

print(f"\n  jax.grad           d<j.B>/dTHat = {gradient:+.10e}")
print(f"  central difference d<j.B>/dTHat = {finite_difference:+.10e}")
print(f"  relative difference             = {abs(gradient / finite_difference - 1.0):.3e}")

# Sanity in physical terms rather than numerical: a hotter plasma at fixed
# gradients is less collisional, and the bootstrap current grows with it.
print(f"\n  sign check: d<j.B>/dTHat is {'positive' if gradient > 0 else 'negative'}, "
      f"and <j.B> is {'positive' if value > 0 else 'negative'}")
print("Scale this to many parameters and the cost argument is the whole point:")
print("  reverse mode costs one transposed solve regardless of how many there are.")
