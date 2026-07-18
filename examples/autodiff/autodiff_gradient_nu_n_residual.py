"""
Autodiff demo: gradient of a parity objective w.r.t. the collision frequency `nu_n`.

This example is a lightweight differentiable residual check:

- no optimization loop
- no optional dependencies
- no generated benchmark fixtures

It demonstrates a key `dkx` capability for "design/optimization-style" workflows:
differentiate a physics objective through the (matrix-free) residual evaluation.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp

from dkx.drift_kinetic import KineticOperator, kinetic_operator_from_namelist
from dkx.namelist import read_sfincs_input
from dkx.validation.fortran import read_petsc_vec


def _with_nu_n(op: KineticOperator, nu_n: jnp.ndarray) -> KineticOperator:
    """Rebuild the PAS collision operator at a new `nu_n` (coef is linear in it)."""
    pas = op.pas
    scale = jnp.asarray(nu_n, dtype=jnp.float64) / pas.nu_n
    pas2 = replace(pas, nu_n=jnp.asarray(nu_n, dtype=jnp.float64), coef=pas.coef * scale)
    return replace(op, pas=pas2)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    input_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.input.namelist"
    x_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.stateVector.petscbin"
    r_path = repo_root / "tests" / "ref" / "pas_1species_PAS_noEr_tiny_scheme5.residual.petscbin"

    nml = read_sfincs_input(input_path)
    op = kinetic_operator_from_namelist(nml)
    if op.pas is None:
        raise RuntimeError("Expected collisionOperator=1 (PAS) fixture.")

    x_ref = jnp.asarray(read_petsc_vec(x_path).values)
    r_ref = jnp.asarray(read_petsc_vec(r_path).values)
    nu0 = jnp.asarray(op.pas.nu_n, dtype=jnp.float64)
    rhs = op.rhs()

    def loss(nu_n: jnp.ndarray) -> jnp.ndarray:
        op2 = _with_nu_n(op, nu_n)
        r = op2.apply(x_ref) - rhs
        d = r - r_ref
        return 0.5 * jnp.vdot(d, d)

    val, g = jax.value_and_grad(loss)(nu0)
    print("nu_n:", float(nu0))
    print("loss:", float(val))
    print("d(loss)/d(nu_n):", float(g))


if __name__ == "__main__":
    main()
