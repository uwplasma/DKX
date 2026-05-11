from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from sfincs_jax.host_refinement import (
    host_direct_solve_with_refinement,
    host_sparse_direct_solve_with_refinement,
)


class _HalfSolve:
    def solve(self, rhs):  # noqa: ANN001
        return 0.5 * np.asarray(rhs, dtype=np.float64)


def test_direct_and_sparse_refinement_share_monotone_residual_behavior() -> None:
    rhs = np.asarray([2.0, -4.0], dtype=np.float64)
    ident = np.eye(2, dtype=np.float64)

    x_direct, rn_direct = host_direct_solve_with_refinement(
        factor_solve=lambda v: 0.5 * np.asarray(v, dtype=np.float64),
        operator_matrix=ident,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )
    x_sparse, rn_sparse = host_sparse_direct_solve_with_refinement(
        ilu=_HalfSolve(),
        a_csr_full=sparse.csr_matrix(ident),
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=3,
    )

    np.testing.assert_allclose(x_direct, np.asarray([1.875, -3.75]))
    np.testing.assert_allclose(x_sparse, x_direct)
    assert rn_sparse == pytest.approx(rn_direct)
    assert rn_direct < np.linalg.norm(rhs - 0.5 * rhs)


def test_refinement_stops_when_trial_residual_worsens() -> None:
    rhs = np.asarray([1.0, 2.0], dtype=np.float64)
    ident = np.eye(2, dtype=np.float64)
    calls = 0

    def bad_correction(rhs_in: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        if calls == 1:
            return np.zeros_like(rhs_in)
        return -np.asarray(rhs_in, dtype=np.float64)

    x, rn = host_direct_solve_with_refinement(
        factor_solve=bad_correction,
        operator_matrix=ident,
        rhs_vec=rhs,
        factor_dtype=np.dtype(np.float64),
        refine_steps=5,
    )

    np.testing.assert_allclose(x, np.zeros_like(rhs))
    assert rn == pytest.approx(np.linalg.norm(rhs))
    assert calls == 2
