from __future__ import annotations

import numpy as np
import pytest
import jax.numpy as jnp

from sfincs_jax.problems.profile_response.dense import (
    HostDenseReducedSolveContext,
    solve_host_dense_reduced,
)


def test_host_dense_reduced_row_scaled_lu_solves_square_system() -> None:
    a_np = np.asarray([[2.0, 0.0], [0.0, 4.0]])
    rhs = jnp.asarray([2.0, 8.0])
    result = solve_host_dense_reduced(
        context=HostDenseReducedSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            active_size=2,
            constraint_scheme=0,
            has_fp=False,
            dense_matrix_cache=a_np,
        ),
        x0=jnp.zeros(2),
    )

    assert result.x.tolist() == pytest.approx([1.0, 2.0])
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)


def test_host_dense_reduced_lstsq_handles_rectangular_cache() -> None:
    a_np = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rhs = jnp.asarray([1.0, 2.0, 3.0])
    result = solve_host_dense_reduced(
        context=HostDenseReducedSolveContext(
            matvec=lambda x: jnp.asarray(a_np) @ x,
            rhs=rhs,
            active_size=2,
            constraint_scheme=2,
            has_fp=False,
            dense_matrix_cache=a_np,
        )
    )

    expected = np.linalg.lstsq(a_np, np.asarray(rhs), rcond=None)[0]
    assert result.x.tolist() == pytest.approx(expected.tolist())
    assert float(result.residual_norm) == pytest.approx(0.0, abs=1.0e-12)
