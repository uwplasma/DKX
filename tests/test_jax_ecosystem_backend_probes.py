from __future__ import annotations

# ruff: noqa: E402

from importlib.util import find_spec
from pathlib import Path

from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_full_system import clear_structured_rhs1_full_csr_cache, select_structured_rhs1_full_csr_operator
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist


REF = Path(__file__).parent / "ref"


def test_jax_sparse_bcoo_matches_rhs1_structured_csr_matvec() -> None:
    from jax.experimental import sparse as jsparse

    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None

    matrix = selection.matrix.tocsr()
    sparse_matrix = jsparse.BCOO.from_scipy_sparse(matrix)
    vector = jnp.asarray(np.sin(0.11 * np.arange(matrix.shape[1], dtype=np.float64)))

    expected = np.asarray(matrix @ np.asarray(vector))
    actual = np.asarray(sparse_matrix @ vector)
    actual_jit = np.asarray(jax.jit(lambda x: sparse_matrix @ x)(vector))

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)
    np.testing.assert_allclose(actual_jit, expected, rtol=1.0e-12, atol=1.0e-12)


@pytest.mark.skipif(find_spec("lineax") is None, reason="lineax is optional")
def test_optional_lineax_gmres_matches_jax_dense_solve() -> None:
    import lineax as lx

    matrix = jnp.asarray(
        [
            [4.0, 0.3, -0.2, 0.1],
            [0.1, 3.2, 0.4, -0.3],
            [-0.2, 0.5, 2.8, 0.2],
            [0.3, -0.1, 0.2, 2.5],
        ],
        dtype=jnp.float64,
    )
    rhs = jnp.asarray([1.0, -0.25, 0.5, 2.0], dtype=jnp.float64)
    operator = lx.MatrixLinearOperator(matrix)
    solver = lx.GMRES(rtol=1.0e-12, atol=1.0e-12, max_steps=32, restart=8)

    solution = lx.linear_solve(operator, rhs, solver, throw=True)

    np.testing.assert_allclose(solution.value, jnp.linalg.solve(matrix, rhs), rtol=1.0e-10, atol=1.0e-10)
    assert int(solution.stats["num_steps"]) <= 8


@pytest.mark.skipif(find_spec("interpax") is None, reason="interpax is optional")
def test_optional_interpax_profile_interpolation_matches_expected_linear_profile() -> None:
    import interpax

    radius = jnp.linspace(0.0, 1.0, 6, dtype=jnp.float64)
    density = 1.25 - 0.4 * radius
    query = jnp.asarray([0.05, 0.25, 0.55, 0.85], dtype=jnp.float64)

    actual = interpax.interp1d(query, radius, density, method="linear")
    expected = 1.25 - 0.4 * query

    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    gradient = jax.grad(lambda q: jnp.sum(interpax.interp1d(q, radius, density, method="linear")))(query)
    np.testing.assert_allclose(gradient, -0.4 * jnp.ones_like(query), rtol=1.0e-12, atol=1.0e-12)
