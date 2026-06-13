from __future__ import annotations

import jax.numpy as jnp
from jax import tree_util as jtu
import numpy as np

from sfincs_jax.solver import GMRESSolveResult
from sfincs_jax.v3_results import (
    V3LinearSolveResult,
    V3NewtonKrylovResult,
    V3TransportMatrixSolveResult,
)


def test_v3_linear_solve_result_properties_and_pytree_metadata() -> None:
    gmres = GMRESSolveResult(
        x=jnp.asarray([1.0, -2.0], dtype=jnp.float64),
        residual_norm=jnp.asarray(1.0e-8, dtype=jnp.float64),
    )
    result = V3LinearSolveResult(op=None, rhs=jnp.ones((2,), dtype=jnp.float64), gmres=gmres, metadata={"path": "unit"})

    np.testing.assert_allclose(np.asarray(result.x), np.asarray([1.0, -2.0]))
    assert float(result.residual_norm) == 1.0e-8

    children, treedef = jtu.tree_flatten(result)
    rebuilt = jtu.tree_unflatten(treedef, children)
    assert rebuilt.metadata == {"path": "unit"}
    np.testing.assert_allclose(np.asarray(rebuilt.x), np.asarray(result.x))


def test_v3_newton_krylov_result_pytree_roundtrip() -> None:
    result = V3NewtonKrylovResult(
        op=None,
        x=jnp.asarray([0.5], dtype=jnp.float64),
        residual_norm=jnp.asarray(2.0e-9, dtype=jnp.float64),
        n_newton=3,
        last_linear_residual_norm=jnp.asarray(4.0e-10, dtype=jnp.float64),
    )

    children, treedef = jtu.tree_flatten(result)
    rebuilt = jtu.tree_unflatten(treedef, children)

    assert rebuilt.n_newton == 3
    assert float(rebuilt.residual_norm) == 2.0e-9
    assert float(rebuilt.last_linear_residual_norm) == 4.0e-10


def test_v3_transport_matrix_result_keeps_diagnostics_and_metadata() -> None:
    result = V3TransportMatrixSolveResult(
        op0=None,
        transport_matrix=jnp.eye(2, dtype=jnp.float64),
        state_vectors_by_rhs={1: jnp.ones((2,), dtype=jnp.float64)},
        residual_norms_by_rhs={1: jnp.asarray(1.0e-12, dtype=jnp.float64)},
        fsab_flow=jnp.ones((1, 2), dtype=jnp.float64),
        particle_flux_vm_psi_hat=jnp.zeros((1, 2), dtype=jnp.float64),
        heat_flux_vm_psi_hat=jnp.zeros((1, 2), dtype=jnp.float64),
        elapsed_time_s=jnp.asarray([0.1, 0.2], dtype=jnp.float64),
        preconditioner_kind="auto",
    )

    assert result.transport_matrix.shape == (2, 2)
    assert result.preconditioner_kind == "auto"
    assert set(result.state_vectors_by_rhs) == {1}
