from __future__ import annotations

from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_vec
from sfincs_jax.operators.profile_system import (
    V3FullLinearSystem,
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    jacobian_matvec_v3_full_system_jit,
    residual_v3_full_system_jit,
)


def test_full_system_residual_and_jvp_pas_tiny() -> None:
    """Residual and JVP are consistent for the full operator in a tiny PAS case."""
    here = Path(__file__).parent
    input_path = here / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    vec_path = here / "ref" / "pas_1species_PAS_noEr_tiny.stateVector.petscbin"

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    x_ref = jnp.asarray(read_petsc_vec(vec_path).values)
    b = apply_v3_full_system_operator(op, x_ref)
    sys = V3FullLinearSystem(op=op, b_full=b)

    r = np.asarray(sys.residual(x_ref))
    np.testing.assert_allclose(r, 0.0, rtol=0, atol=1e-12)

    key = jax.random.key(0)
    v = jax.random.normal(key, shape=(op.total_size,), dtype=jnp.float64)
    r0, jvp = sys.jvp(x_ref, v)

    np.testing.assert_allclose(np.asarray(r0), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(jvp), np.asarray(sys.jacobian_matvec(v)), rtol=0, atol=1e-12)

    children, aux = sys.tree_flatten()
    rebuilt_direct = V3FullLinearSystem.tree_unflatten(aux, children)
    rebuilt_tree = jtu.tree_unflatten(jtu.tree_structure(sys), jtu.tree_leaves(sys))

    np.testing.assert_allclose(np.asarray(rebuilt_direct.residual(x_ref)), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(rebuilt_tree.jacobian_matvec(v)), np.asarray(jvp), rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(residual_v3_full_system_jit(sys, x_ref)), 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        np.asarray(jacobian_matvec_v3_full_system_jit(sys, v)),
        np.asarray(jvp),
        rtol=0,
        atol=1e-12,
    )
