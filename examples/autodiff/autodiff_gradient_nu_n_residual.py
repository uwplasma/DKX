"""Differentiate a parity residual objective w.r.t. the collisionality ``nu_n``.

What this example teaches:
  - how to treat a physics parameter (the normalized collision frequency
    ``nu_n``) as a differentiable JAX scalar,
  - how to differentiate a matrix-free residual objective
    ``0.5 * || r(nu_n; x_ref) - r_ref ||^2`` with ``jax.value_and_grad`` --
    no optimization loop, no optional dependencies, no sparse matrices,
  - how the PAS collision operator rebuilds cheaply at a new ``nu_n`` because
    its coefficient is linear in ``nu_n``.

Physics context: differentiating a physics objective through the drift-kinetic
residual is the building block for gradient-based neoclassical optimization;
here ``x_ref`` and ``r_ref`` come from a frozen SFINCS Fortran v3 fixture, so
the objective is anchored to a real reference solve [M. Landreman et al., Phys.
Plasmas 21, 042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].

Run:
  python examples/autodiff/autodiff_gradient_nu_n_residual.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from dkx.drift_kinetic import KineticOperator, kinetic_operator_from_namelist  # noqa: E402
from dkx.namelist import read_sfincs_input  # noqa: E402
from dkx.validation.fortran import read_petsc_vec  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen Fortran v3 PAS fixture (single species, no Er, scheme 5).
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
STATEVECTOR = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"
RESIDUAL = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.residual.petscbin"

# nu_n scan (relative to the fixture value) for the loss-curve plot.
SCAN_FRACTIONS = np.linspace(0.5, 1.5, 21)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "autodiff_gradient_nu_n_residual"
PLOT_PATH = OUTPUT_DIR / "autodiff_gradient_nu_n_residual.png"


def with_nu_n(op: KineticOperator, nu_n: jnp.ndarray) -> KineticOperator:
    """Rebuild the PAS collision operator at a new ``nu_n`` (coef is linear in it)."""

    pas = op.pas
    scale = jnp.asarray(nu_n, dtype=jnp.float64) / pas.nu_n
    pas2 = replace(pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64), coef=pas.coef * scale)
    return replace(op, pas=pas2)


# ----------------------------------------------------------------------------
# 1) Build the operator and load the frozen reference vectors
# ----------------------------------------------------------------------------
print("=== examples/autodiff/autodiff_gradient_nu_n_residual.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Step 1: building the operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op = kinetic_operator_from_namelist(nml)
if op.pas is None:
    raise RuntimeError("Expected collisionOperator=1 (PAS) fixture.")

x_ref = jnp.asarray(read_petsc_vec(STATEVECTOR).values)
r_ref = jnp.asarray(read_petsc_vec(RESIDUAL).values)
nu0 = jnp.asarray(op.pas.nu_n, dtype=jnp.float64)
rhs = op.rhs()


def loss(nu_n: jnp.ndarray) -> jnp.ndarray:
    op2 = with_nu_n(op, nu_n)
    d = (op2.apply(x_ref) - rhs) - r_ref
    return 0.5 * jnp.vdot(d, d)


# ----------------------------------------------------------------------------
# 2) Value and gradient at the fixture nu_n
# ----------------------------------------------------------------------------
print("Step 2: differentiating the residual objective with jax.value_and_grad")
val, grad = jax.value_and_grad(loss)(nu0)

# ----------------------------------------------------------------------------
# 3) Plot the loss over a small nu_n scan with the AD tangent at nu0
# ----------------------------------------------------------------------------
print("Step 3: plotting the loss curve and the autodiff tangent")
nu_scan = np.asarray(SCAN_FRACTIONS) * float(nu0)
loss_scan = np.array([float(loss(jnp.asarray(nu))) for nu in nu_scan])
fig, ax = plt.subplots(figsize=(5.8, 4.0))
ax.plot(nu_scan, loss_scan, "o-", ms=4, color="tab:blue", label="loss(nu_n)")
tangent = float(val) + float(grad) * (nu_scan - float(nu0))
ax.plot(nu_scan, tangent, "--", color="tab:red", label="autodiff tangent at nu_n0")
ax.set_xlabel(r"$\nu_n$")
ax.set_ylabel(r"$0.5\,\|r(\nu_n) - r_{\mathrm{ref}}\|^2$")
ax.set_title("Differentiable residual objective")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(PLOT_PATH, dpi=140)
plt.close(fig)

# ----------------------------------------------------------------------------
# 4) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  nu_n0            = {float(nu0):.6g}")
print(f"  loss(nu_n0)      = {float(val):.6e}")
print(f"  d(loss)/d(nu_n)  = {float(grad):.6e}")
print(f"  Saved plot: {PLOT_PATH.name}")
print("Done: examples/autodiff/autodiff_gradient_nu_n_residual.py")
