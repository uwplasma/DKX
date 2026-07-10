from __future__ import annotations

import jax
from jax import config as jax_config
import numpy as np
import jax.numpy as jnp

jax_config.update("jax_enable_x64", True)

from sfincs_jax.solvers.krylov import recycled_initial_guess, small_regularized_lstsq  # noqa: E402


def test_small_regularized_lstsq_matches_numpy_tall_system() -> None:
    a = jnp.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.0, 3.0, 1.0],
            [1.0, 1.0, -2.0],
            [4.0, 0.5, 1.5],
            [-1.0, 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, -2.0, 0.5, 3.0, -1.5], dtype=jnp.float64)

    coeff = np.asarray(small_regularized_lstsq(a, b))
    coeff_ref, *_ = np.linalg.lstsq(np.asarray(a), np.asarray(b), rcond=None)

    assert np.allclose(coeff, coeff_ref, rtol=1e-9, atol=1e-9)


def test_small_regularized_lstsq_handles_near_rank_deficiency() -> None:
    a = jnp.asarray(
        [
            [1.0, 1.0],
            [2.0, 2.0 + 1e-10],
            [3.0, 3.0 - 1e-10],
            [4.0, 4.0 + 2e-10],
        ],
        dtype=jnp.float64,
    )
    b = jnp.asarray([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    coeff = np.asarray(small_regularized_lstsq(a, b))
    residual = np.linalg.norm(np.asarray(a) @ coeff - np.asarray(b))

    assert np.all(np.isfinite(coeff))
    assert residual < 1e-8


def test_small_regularized_lstsq_handles_empty_basis_and_gradients() -> None:
    empty = small_regularized_lstsq(jnp.zeros((4, 0), dtype=jnp.float64), jnp.ones((4,), dtype=jnp.float64))
    assert empty.shape == (0,)

    b = jnp.asarray([1.0, -2.0, 0.5, 3.0, -1.5], dtype=jnp.float64)

    def loss(flat_a: jnp.ndarray) -> jnp.ndarray:
        a = flat_a.reshape((5, 3))
        coeff = small_regularized_lstsq(a, b)
        return jnp.sum(coeff * coeff)

    a0 = jnp.asarray(
        [
            [2.0, -1.0, 0.5],
            [0.0, 3.0, 1.0],
            [1.0, 1.0, -2.0],
            [4.0, 0.5, 1.5],
            [-1.0, 2.0, 0.0],
        ],
        dtype=jnp.float64,
    )
    grad = jax.grad(loss)(a0.reshape((-1,)))

    assert grad.shape == (15,)
    assert np.all(np.isfinite(np.asarray(grad)))


def test_recycled_initial_guess_minimizes_residual_over_basis() -> None:
    basis = [
        jnp.asarray([1.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float64),
    ]
    basis_au = [
        jnp.asarray([2.0, 0.0, 0.0], dtype=jnp.float64),
        jnp.asarray([0.0, 3.0, 0.0], dtype=jnp.float64),
    ]
    rhs = jnp.asarray([4.0, 9.0, 1.0], dtype=jnp.float64)

    x0 = recycled_initial_guess(rhs, basis, basis_au)

    assert x0 is not None
    assert np.allclose(np.asarray(x0), np.asarray([2.0, 3.0, 0.0]), rtol=1e-9, atol=1e-9)


def test_recycled_initial_guess_handles_empty_or_nonfinite_basis() -> None:
    rhs = jnp.asarray([1.0, 2.0], dtype=jnp.float64)

    assert recycled_initial_guess(rhs, [], []) is None
    assert recycled_initial_guess(
        rhs,
        [jnp.asarray([jnp.nan, 0.0], dtype=jnp.float64)],
        [jnp.asarray([1.0, 0.0], dtype=jnp.float64)],
    ) is None
