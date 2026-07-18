"""Autodiff vs finite-difference sensitivity of the residual to ``nu_n``.

What this example teaches:
  - how to differentiate the full-system residual objective
    ``0.5 * || r(nu_n; x0) ||^2`` with respect to the normalized collisionality
    ``nu_n`` using ``jax.grad`` -- straight through the matrix-free operator, no
    sparse assembly (VMEC ``geometryScheme=5``),
  - how to validate the autodiff gradient against a centered finite difference,
  - how a perturbed state ``x0`` is built from a frozen Fortran v3
    ``stateVector`` fixture.

Physics context: automatic differentiation through the discrete drift-kinetic
operator is a core advantage of the JAX port -- exact gradients of neoclassical
objectives with no hand-derived adjoints [M. Landreman et al., Phys. Plasmas
21, 042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].

Run:
  python examples/autodiff/autodiff_sensitivity_nu_n_scheme5.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from dkx.drift_kinetic import KineticOperator, kinetic_operator_from_namelist
from dkx.namelist import read_sfincs_input
from dkx.validation.fortran import read_petsc_vec

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen Fortran v3 PAS fixture (single species, no Er, scheme 5).
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
STATEVECTOR = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"

SEED = 0  # PRNG seed for the state-vector perturbation
NOISE = 1e-3  # additive perturbation scale applied to x_ref
FD_EPS = 1e-5  # centered finite-difference step for the gradient check


def with_nu_n(op: KineticOperator, nu_n: jnp.ndarray) -> KineticOperator:
    """Rebuild the PAS collision operator at a new ``nu_n`` (coef is linear in it)."""

    pas = op.pas
    scale = jnp.asarray(nu_n, dtype=jnp.float64) / pas.nu_n
    pas2 = replace(pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64), coef=pas.coef * scale)
    return replace(op, pas=pas2)


# ----------------------------------------------------------------------------
# 1) Build the operator and a perturbed state vector
# ----------------------------------------------------------------------------
print("=== examples/autodiff/autodiff_sensitivity_nu_n_scheme5.py ===")
print(f"Step 1: building the operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op = kinetic_operator_from_namelist(nml)
if op.pas is None:
    raise RuntimeError("This example expects collisionOperator=1 (PAS), but op.pas is None.")

x_ref = jnp.asarray(read_petsc_vec(STATEVECTOR).values)
rng = np.random.default_rng(SEED)
x0 = x_ref + jnp.asarray(NOISE * rng.normal(size=(x_ref.size,)).astype(np.float64))
nu0 = jnp.asarray(op.pas.nu_n, dtype=jnp.float64)
rhs = op.rhs()


def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
    op2 = with_nu_n(op, nu_n)
    r = op2.apply(x0) - rhs
    return 0.5 * jnp.vdot(r, r)


# ----------------------------------------------------------------------------
# 2) Autodiff gradient vs centered finite difference
# ----------------------------------------------------------------------------
print("Step 2: comparing jax.grad against a centered finite difference")
grad = jax.grad(objective)(nu0)
fd = (float(objective(nu0 + FD_EPS)) - float(objective(nu0 - FD_EPS))) / (2.0 * FD_EPS)
abs_err = abs(float(grad) - fd)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  nu_n0                          = {float(nu0):.6g}")
print(f"  objective(nu_n0)               = {float(objective(nu0)):.6e}")
print(f"  d objective / d nu_n (autodiff)     = {float(grad):.6e}")
print(f"  d objective / d nu_n (finite-diff)  = {fd:.6e}")
print(f"  abs_err                        = {abs_err:.3e}")
print("Done: examples/autodiff/autodiff_sensitivity_nu_n_scheme5.py")
