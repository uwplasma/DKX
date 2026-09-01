"""Exact derivatives through the solve: ``jax.grad``, not finite differences.

The drift-kinetic solve is differentiable, so the derivative of any output with
respect to any input is available for about the cost of one extra solve --
reverse mode costs one transposed solve no matter how many parameters there
are, against 2N solves for central differences, plus a step size you have to
guess.  That is what makes gradient-based stellarator optimization affordable.

The pattern is three lines: take ``run.operator``, ``dataclasses.replace`` the
leaf you are differentiating, and solve with ``differentiable=True``.  The
operator is a JAX pytree, so the drive, the matrix apply, the solve and the
moments are all traced.

Two honesty notes.  Gradients live on the operator lane, which is why this
rung does not use a ``dkx.Case``: ``dkx.run(case)`` is a NumPy host loop over
surfaces and is not traceable.  And the collision matrices are built with the
operator, so replacing ``t_hat`` does not rebuild them -- autodiff and the
finite difference differentiate the same fixed-collisionality function, which
is exactly what makes them comparable.

Physics: the circular tokamak of rung 01 as an SFINCS deck; the output is the
flux-surface-averaged bootstrap current ``<j.B>``, the input its temperature.

Expected runtime: ~17 s on a laptop CPU.
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
from dkx.run import profile_moments_from_operator  # noqa: E402
from dkx.solve import solve  # noqa: E402

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "07_gradients"
RESULT_FILE = OUT_DIR / "gradients.nc"
PLOT_FILE = OUT_DIR / "gradients.png"

# The step for the finite-difference cross-check.  Finite differences are not
# ground truth here: they carry their own truncation and round-off error, so
# this establishes that the two agree, not that autodiff has been "validated".
FD_STEP = 1.0e-5
# Temperatures at which to evaluate <j.B> for the figure, as multiples of the
# base THat.  Each is one extra solve.
SWEEP_FRACTIONS = (0.80, 0.90, 1.00, 1.10, 1.20)

# 3. Geometry and species construction
CASE = dict(
    geometryScheme=1, inputRadialCoordinate=3, rN_wish=0.3,
    B0OverBBar=1.0, epsilon_t=-0.07, epsilon_h=0.0,
    iota=0.4542, GHat=3.7481, IHat=0.0, psiAHat=0.15596, aHat=0.5585,
    Zs=[1.0], mHats=[1.0], nHats=[1.0], THats=[1.0],
    dNHatdrHats=[-0.5], dTHatdrHats=[-1.0],
)

# 4. Physics and numerical configuration
NUMERICS = dict(
    Ntheta=11, Nzeta=1, Nxi=12, NL=4, Nx=5,
    collisionOperator=1, Delta=4.5694e-3, alpha=1.0, nu_n=8.330e-3,
)
# end of parameters

OUT_DIR.mkdir(parents=True, exist_ok=True)
operator = dkx.run(**CASE, **NUMERICS, emit=None).operator
print(f"operator: {operator.n_species} species, matrix size {operator.total_size}")


def bootstrap_current(t_hat):
    """<j.B> as a function of the species temperature.  Differentiable."""
    perturbed = replace(operator, t_hat=jnp.reshape(t_hat, (1,)))
    # method="auto" picks the route this operator needs.  Naming a route by
    # hand is how you meet "the full-band structured direct factorization
    # requires uniform Nxi_for_x": this deck ramps Nxi with speed, so it
    # belongs on the truncated kernel, and "auto" knows that.
    solved = solve(perturbed, perturbed.rhs(), method="auto", differentiable=True)
    return profile_moments_from_operator(perturbed, solved.x)["FSABjHat"]


# 5. Run
t_hat_0 = float(operator.t_hat[0])
value = float(bootstrap_current(jnp.asarray(t_hat_0)))
gradient = float(jax.grad(bootstrap_current)(jnp.asarray(t_hat_0)))

step = FD_STEP * max(1.0, abs(t_hat_0))
finite_difference = float(
    (bootstrap_current(jnp.asarray(t_hat_0 + step))
     - bootstrap_current(jnp.asarray(t_hat_0 - step))) / (2.0 * step)
)

sweep_t_hat = np.array([fraction * t_hat_0 for fraction in SWEEP_FRACTIONS], dtype=float)
sweep_current = np.array(
    [float(bootstrap_current(jnp.asarray(t))) for t in sweep_t_hat], dtype=float
)

# 6. Print a scientific summary and certificate
relative_difference = abs(gradient / finite_difference - 1.0)
print("\n=== Final results ===")
print(f"  <j.B> at THat = {t_hat_0:.4f}      = {value:+.8e} (normalized)")
print(f"  jax.grad           d<j.B>/dTHat = {gradient:+.10e}")
print(f"  central difference d<j.B>/dTHat = {finite_difference:+.10e}")
print(f"  relative difference             = {relative_difference:.3e}")
assert relative_difference < 1.0e-5, "autodiff and central differences disagree"
print("  all gradients verified against central finite differences")
# Sanity in physical terms rather than numerical: at fixed normalized gradients
# a hotter plasma is less collisional, and the bootstrap current grows with it.
print(f"  sign check: d<j.B>/dTHat is {'positive' if gradient > 0 else 'negative'}, "
      f"and <j.B> is {'positive' if value > 0 else 'negative'}")
print("  cost: reverse mode is one transposed solve regardless of how many "
      "parameters are differentiated")

# 7. Save native result
with Dataset(RESULT_FILE, "w", format="NETCDF4") as dataset:
    dataset.createDimension("sweep", sweep_t_hat.size)
    dataset.createVariable("THat", "f8", ("sweep",))[:] = sweep_t_hat
    dataset.createVariable("FSABjHat", "f8", ("sweep",))[:] = sweep_current
    dataset.createVariable("dFSABjHat_dTHat", "f8")[...] = gradient
    dataset.createVariable("dFSABjHat_dTHat_central_difference", "f8")[...] = finite_difference
    dataset.dkx_version = dkx.__version__
    dataset.autodiff = "jax.grad through dkx.solve(differentiable=True)"
print(f"  Wrote result: {RESULT_FILE}")

# 8. Plot publication-ready outputs
figure, (left, right) = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
left.plot(sweep_t_hat, sweep_current, "o-", color="tab:blue", label=r"$\langle j\cdot B\rangle$")
tangent_t = np.linspace(sweep_t_hat.min(), sweep_t_hat.max(), 2)
left.plot(tangent_t, value + gradient * (tangent_t - t_hat_0), "--", color="tab:red",
          label="tangent from jax.grad")  # fmt: skip
left.plot([t_hat_0], [value], "x", color="tab:red", ms=11, mew=2.5)
left.set(xlabel=r"$\hat T$", ylabel=r"$\langle j\cdot B\rangle$ (normalized)")
left.set_title("one solve gives the value and the slope", fontsize=10)
left.grid(alpha=0.3)
left.legend(fontsize=9)

right.bar(["jax.grad", "central difference"], [gradient, finite_difference],
          color=["tab:red", "tab:grey"])  # fmt: skip
right.set_ylabel(r"$d\langle j\cdot B\rangle/d\hat T$")
right.set_title(f"agreement: {relative_difference:.1e} relative", fontsize=10)
right.grid(alpha=0.3, axis="y")
figure.suptitle("Exact derivatives through the drift-kinetic solve")
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/07_gradients/run.py")
