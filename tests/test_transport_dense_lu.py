from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.problems.transport_matrix.solve import (
    dense_preconditioner_for_matvec,
    dense_solver_for_matvec,
)


def test_dense_solver_for_matvec_solves_and_reuses_cache() -> None:
    a = jnp.asarray([[3.0, 1.0], [1.0, 2.0]], dtype=jnp.float64)

    def mv(x):
        return a @ x

    cache = {}
    solve = dense_solver_for_matvec(matvec_fn=mv, n=2, dtype=jnp.float64, cache=cache, key=("a", 2))
    x = solve(jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    np.testing.assert_allclose(a @ x, jnp.asarray([1.0, 2.0]), rtol=1e-12, atol=1e-12)
    assert dense_solver_for_matvec(matvec_fn=mv, n=2, dtype=jnp.float64, cache=cache, key=("a", 2)) is solve


def test_dense_preconditioner_for_matvec_solves_and_reuses_cache() -> None:
    a = jnp.asarray([[4.0, 0.0], [0.0, 5.0]], dtype=jnp.float64)

    def mv(x):
        return a @ x

    cache = {}
    precond = dense_preconditioner_for_matvec(matvec_fn=mv, n=2, dtype=jnp.float64, cache=cache, key=("p", 2))
    y = precond(jnp.asarray([8.0, 10.0], dtype=jnp.float64))
    np.testing.assert_allclose(y, jnp.asarray([2.0, 2.0]), rtol=1e-12, atol=1e-12)
    assert dense_preconditioner_for_matvec(matvec_fn=mv, n=2, dtype=jnp.float64, cache=cache, key=("p", 2)) is precond
