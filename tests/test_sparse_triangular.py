import jax.numpy as jnp
import numpy as np

from sfincs_jax.sparse_triangular import (
    inverse_permutation,
    triangular_solve_lower_csr_rows,
    triangular_solve_lower_padded,
    triangular_solve_upper_csr_rows,
    triangular_solve_upper_padded,
)


def test_inverse_permutation_round_trips_indices():
    permutation = np.asarray([2, 0, 3, 1], dtype=np.int32)
    inverse = inverse_permutation(permutation)
    assert np.array_equal(permutation[inverse], np.arange(4, dtype=np.int32))
    assert np.array_equal(inverse[permutation], np.arange(4, dtype=np.int32))


def test_padded_triangular_solves_match_dense_reference():
    lower_idx = jnp.asarray([[-1, -1], [0, -1], [0, 1]], dtype=jnp.int32)
    lower_val = jnp.asarray([[0.0, 0.0], [2.0, 0.0], [-1.0, 0.5]])
    b_lower = jnp.asarray([1.0, 5.0, 2.0])
    y = triangular_solve_lower_padded(lower_idx=lower_idx, lower_val=lower_val, b=b_lower)
    lower_dense = np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.5, 1.0]])
    np.testing.assert_allclose(lower_dense @ np.asarray(y), np.asarray(b_lower), rtol=1e-12, atol=1e-12)

    upper_idx = jnp.asarray([[1, 2], [2, -1], [-1, -1]], dtype=jnp.int32)
    upper_val = jnp.asarray([[1.5, -0.5], [3.0, 0.0], [0.0, 0.0]])
    upper_diag = jnp.asarray([2.0, -1.0, 4.0])
    b_upper = jnp.asarray([7.0, -2.0, 8.0])
    x = triangular_solve_upper_padded(
        upper_idx=upper_idx,
        upper_val=upper_val,
        upper_diag=upper_diag,
        b=b_upper,
    )
    upper_dense = np.asarray([[2.0, 1.5, -0.5], [0.0, -1.0, 3.0], [0.0, 0.0, 4.0]])
    np.testing.assert_allclose(upper_dense @ np.asarray(x), np.asarray(b_upper), rtol=1e-12, atol=1e-12)


def test_csr_triangular_solves_match_dense_reference():
    lower_indptr = jnp.asarray([0, 0, 1, 3], dtype=jnp.int32)
    lower_indices = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    lower_data = jnp.asarray([2.0, -1.0, 0.5])
    b_lower = jnp.asarray([1.0, 5.0, 2.0])
    y = triangular_solve_lower_csr_rows(
        indptr=lower_indptr,
        indices=lower_indices,
        data=lower_data,
        b=b_lower,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )
    lower_dense = np.asarray([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.5, 1.0]])
    np.testing.assert_allclose(lower_dense @ np.asarray(y), np.asarray(b_lower), rtol=1e-12, atol=1e-12)

    upper_indptr = jnp.asarray([0, 2, 3, 3], dtype=jnp.int32)
    upper_indices = jnp.asarray([1, 2, 2], dtype=jnp.int32)
    upper_data = jnp.asarray([1.5, -0.5, 3.0])
    upper_diag = jnp.asarray([2.0, -1.0, 4.0])
    b_upper = jnp.asarray([7.0, -2.0, 8.0])
    x = triangular_solve_upper_csr_rows(
        indptr=upper_indptr,
        indices=upper_indices,
        data=upper_data,
        upper_diag=upper_diag,
        b=b_upper,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )
    upper_dense = np.asarray([[2.0, 1.5, -0.5], [0.0, -1.0, 3.0], [0.0, 0.0, 4.0]])
    np.testing.assert_allclose(upper_dense @ np.asarray(x), np.asarray(b_upper), rtol=1e-12, atol=1e-12)


def test_empty_sparse_rows_use_diagonal_shortcuts():
    b = jnp.asarray([2.0, -4.0])
    assert np.array_equal(
        np.asarray(
            triangular_solve_lower_padded(
                lower_idx=jnp.zeros((2, 0), dtype=jnp.int32),
                lower_val=jnp.zeros((2, 0)),
                b=b,
            )
        ),
        np.asarray(b),
    )

    diag = jnp.asarray([2.0, -2.0])
    upper = triangular_solve_upper_csr_rows(
        indptr=jnp.asarray([0, 0, 0], dtype=jnp.int32),
        indices=jnp.asarray([], dtype=jnp.int32),
        data=jnp.asarray([]),
        upper_diag=diag,
        b=b,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )
    np.testing.assert_allclose(np.asarray(upper), np.asarray(b / diag), rtol=0.0, atol=0.0)
