from __future__ import annotations

import numpy as np

from sfincs_jax.transport_parallel_runtime import (
    merge_transport_parallel_results,
    partition_transport_rhs,
)


def test_partition_transport_rhs_round_robin() -> None:
    chunks = partition_transport_rhs([1, 2, 3, 4, 5], 3)
    assert chunks == [[1, 4], [2, 5], [3]]


def test_merge_transport_parallel_results_handles_subset_elapsed_layouts() -> None:
    results = [
        {
            "which_rhs_values": [1, 3],
            "state_vectors_by_rhs": {1: np.array([1.0]), 3: np.array([3.0])},
            "residual_norms_by_rhs": {1: 1.0e-8, 3: 3.0e-8},
            "elapsed_time_s": np.asarray([0.1, 0.3], dtype=np.float64),
        },
        {
            "which_rhs_values": [2],
            "state_vectors_by_rhs": {2: np.array([2.0])},
            "residual_norms_by_rhs": {2: 2.0e-8},
            "elapsed_time_s": np.asarray([0.2], dtype=np.float64),
        },
    ]

    state_vectors, residual_norms, elapsed_s = merge_transport_parallel_results(n_rhs=3, results=results)

    assert set(state_vectors) == {1, 2, 3}
    assert set(residual_norms) == {1, 2, 3}
    np.testing.assert_allclose(state_vectors[1], np.array([1.0]))
    np.testing.assert_allclose(state_vectors[2], np.array([2.0]))
    np.testing.assert_allclose(state_vectors[3], np.array([3.0]))
    np.testing.assert_allclose(elapsed_s, np.array([0.1, 0.2, 0.3]))


def test_merge_transport_parallel_results_handles_indexed_elapsed_layout() -> None:
    results = [
        {
            "which_rhs_values": [2, 4],
            "state_vectors_by_rhs": {2: np.array([2.0]), 4: np.array([4.0])},
            "residual_norms_by_rhs": {2: 2.0e-8, 4: 4.0e-8},
            "elapsed_time_s": np.asarray([9.0, 0.2, 9.0, 0.4], dtype=np.float64),
        }
    ]

    state_vectors, residual_norms, elapsed_s = merge_transport_parallel_results(n_rhs=4, results=results)

    assert set(state_vectors) == {2, 4}
    assert set(residual_norms) == {2, 4}
    np.testing.assert_allclose(elapsed_s, np.array([0.0, 0.2, 0.0, 0.4]))
