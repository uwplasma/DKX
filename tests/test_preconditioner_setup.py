from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.preconditioner_setup import hash_array, matvec_submatrix, precond_chunk_cols


def test_precond_chunk_cols_respects_explicit_column_override():
    env = {"SFINCS_JAX_PRECOND_CHUNK": "3", "SFINCS_JAX_PRECOND_MAX_MB": "1e-9"}
    assert precond_chunk_cols(1000, 10, environ=env) == 3
    assert precond_chunk_cols(1000, 2, environ=env) == 2


def test_precond_chunk_cols_uses_memory_budget_and_safe_fallbacks():
    assert precond_chunk_cols(1000, 100, environ={"SFINCS_JAX_PRECOND_MAX_MB": "0.016"}) == 2
    assert precond_chunk_cols(1000, 100, environ={"SFINCS_JAX_PRECOND_MAX_MB": "bad"}) == 100
    assert precond_chunk_cols(0, 7, environ={}) == 7
    assert precond_chunk_cols(1000, 7, environ={"SFINCS_JAX_PRECOND_MAX_MB": "0"}) == 7


def test_hash_array_is_stable_for_array_like_inputs():
    values = np.asarray([1.0, 2.0, 3.5])
    assert hash_array(values) == hash_array(jnp.asarray(values))
    assert hash_array(values) != hash_array(np.asarray([1.0, 2.0, 3.6]))


def test_matvec_submatrix_uses_injected_unsharded_operator_and_chunks():
    calls: list[tuple[bool, bool, tuple[int, ...]]] = []

    def _apply(_op, vector, *, include_jacobian_terms=True, allow_sharding=True):
        calls.append((include_jacobian_terms, allow_sharding, tuple(vector.shape)))
        return 3.0 * vector + jnp.arange(vector.shape[0], dtype=vector.dtype)

    submatrix = matvec_submatrix(
        SimpleNamespace(),
        col_idx=np.asarray([0, 2, 3], dtype=np.int32),
        row_idx=np.asarray([0, 2], dtype=np.int32),
        total_size=4,
        chunk_cols=2,
        apply_operator_fn=_apply,
    )

    np.testing.assert_allclose(
        submatrix,
        np.asarray(
            [
                [3.0, 2.0],
                [0.0, 5.0],
                [0.0, 2.0],
            ]
        ),
    )
    assert calls == [(True, False, (4,)), (True, False, (4,))]
