"""Implicit differentiation through a full-system Krylov solve (VMEC scheme 5).

What this example teaches:
  - how to differentiate *through a linear solve* ``A(nu_n) x = rhs(nu_n)`` with
    implicit differentiation instead of backpropagating through the Krylov
    iterations,
  - how ``dkx.solve.solve(..., differentiable=True)`` wires the adjoint/transpose
    solve so ``jax.grad`` of ``0.5 * ||x(nu_n)||^2`` is exact and cheap,
  - how the implicit-diff gradient matches a centered finite difference.

Physics context: implicit differentiation via the adjoint identity
``dL/dp = lambda^T (db/dp - (dA/dp) x)`` with ``A^T lambda = dL/dx`` is the
standard way to get gradients through an iterative solver without unrolling it
-- a key ingredient for gradient-based stellarator/neoclassical optimization
[M. Landreman et al., Phys. Plasmas 21, 042503 (2014); SFINCS technical
documentation, https://github.com/landreman/sfincs].  The ``auto`` solve policy
routes this PAS deck to the structured direct tier.

Run:
  python examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from dkx.drift_kinetic import KineticOperator, kinetic_operator_from_namelist
from dkx.namelist import read_sfincs_input
from dkx.solve import solve

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Frozen Fortran v3 PAS fixture (single species, no Er, scheme 5).
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"

# Solve tier: "auto" routes this PAS deck to the structured direct tier;
# "gmres" forces the recycled-Krylov tier; "block_tridiagonal" the direct tier.
SOLVE_METHOD = "auto"
SOLVER_TOLERANCE = 1e-12
FD_EPS = 1e-5  # centered finite-difference step for the gradient check


def with_nu_n(op: KineticOperator, nu_n: jnp.ndarray) -> KineticOperator:
    """Rebuild the PAS collision operator at a new ``nu_n`` (coef is linear in it)."""

    pas = op.pas
    scale = jnp.asarray(nu_n, dtype=jnp.float64) / pas.nu_n
    pas2 = replace(pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64), coef=pas.coef * scale)
    return replace(op, pas=pas2)


# ----------------------------------------------------------------------------
# 1) Build the operator
# ----------------------------------------------------------------------------
print("=== examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py ===")
print(f"Step 1: building the operator from {INPUT_NAMELIST.name}")
nml = read_sfincs_input(INPUT_NAMELIST)
op0 = kinetic_operator_from_namelist(nml)
if op0.pas is None:
    raise RuntimeError("This example expects collisionOperator=1 (PAS), but op.pas is None.")
nu0 = jnp.asarray(op0.pas.nu_n, dtype=jnp.float64)


def objective(nu_n: jnp.ndarray) -> jnp.ndarray:
    op = with_nu_n(op0, nu_n)
    b = op.rhs()
    x = solve(op, b, method=SOLVE_METHOD, tol=SOLVER_TOLERANCE, differentiable=True).x
    x_flat = jnp.reshape(x, (-1,))
    return 0.5 * jnp.vdot(x_flat, x_flat)


# ----------------------------------------------------------------------------
# 2) Implicit-diff gradient vs centered finite difference
# ----------------------------------------------------------------------------
print(f"Step 2: differentiating through the '{SOLVE_METHOD}' solve (implicit diff)")
grad = jax.grad(objective)(nu0)
fd = (float(objective(nu0 + FD_EPS)) - float(objective(nu0 - FD_EPS))) / (2.0 * FD_EPS)
abs_err = abs(float(grad) - fd)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  nu_n0                              = {float(nu0):.6g}")
print(f"  objective(nu_n0)                   = {float(objective(nu0)):.6e}")
print(f"  d objective / d nu_n (implicit-diff)  = {float(grad):.6e}")
print(f"  d objective / d nu_n (finite-diff)    = {fd:.6e}")
print(f"  abs_err                            = {abs_err:.3e}")
print("Done: examples/autodiff/implicit_diff_through_gmres_solve_scheme5.py")
