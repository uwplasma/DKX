from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_operators import block_diagonal_only, diagonal_only


def test_diagonal_only_preserves_only_point_coupling() -> None:
    matrix = jnp.asarray(
        [
            [3.0, 1.0, -2.0],
            [4.0, 5.0, 6.0],
            [-1.0, 2.0, 7.0],
        ],
        dtype=jnp.float64,
    )

    reduced = np.asarray(diagonal_only(matrix))
    np.testing.assert_allclose(reduced, np.diag([3.0, 5.0, 7.0]))


def test_block_diagonal_only_preserves_local_block_coupling() -> None:
    matrix = jnp.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ],
        dtype=jnp.float64,
    )
    expected = np.asarray(
        [
            [1.0, 2.0, 0.0, 0.0],
            [5.0, 6.0, 0.0, 0.0],
            [0.0, 0.0, 11.0, 12.0],
            [0.0, 0.0, 15.0, 16.0],
        ]
    )

    np.testing.assert_allclose(np.asarray(block_diagonal_only(matrix, block=2)), expected)
    np.testing.assert_allclose(np.asarray(block_diagonal_only(matrix, block=1)), np.asarray(diagonal_only(matrix)))
