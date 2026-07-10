from __future__ import annotations

import numpy as np
import pytest

from sfincs_jax.problems.transport_parallel_runtime import (
    validate_complete_transport_worker_rhs_coverage,
    validate_distinct_transport_worker_rhs,
    validate_gpu_transport_worker_arrays,
    validate_transport_worker_result_payload,
)


def _complete_worker_result() -> dict[str, object]:
    return {
        "state_vectors_by_rhs": {1: np.array([1.0]), 2: np.array([2.0])},
        "residual_norms_by_rhs": {1: 1.0e-8, 2: 2.0e-8},
        "rhs_norms_by_rhs": {1: 1.0, 2: 2.0},
    }


def test_validate_distinct_transport_worker_rhs_rejects_duplicate_rhs() -> None:
    with pytest.raises(ValueError, match=r"duplicate whichRHS values \[2\]"):
        validate_distinct_transport_worker_rhs(rhs_values=[1, 2], seen_rhs={2, 3})


def test_validate_transport_worker_result_payload_rejects_missing_mapping_keys() -> None:
    result = _complete_worker_result()
    result["rhs_norms_by_rhs"] = {1: 1.0}

    with pytest.raises(ValueError, match=r"missing rhs_norms_by_rhs entries for whichRHS=\[2\]"):
        validate_transport_worker_result_payload(rhs_values=[1, 2], result=result, n_rhs=2)


def test_validate_transport_worker_result_payload_rejects_non_mapping_result_field() -> None:
    result = _complete_worker_result()
    result["residual_norms_by_rhs"] = [1.0e-8, 2.0e-8]

    with pytest.raises(ValueError, match=r"field 'residual_norms_by_rhs' must be a mapping"):
        validate_transport_worker_result_payload(rhs_values=[1, 2], result=result, n_rhs=2)


def test_validate_transport_worker_result_payload_rejects_out_of_range_rhs() -> None:
    with pytest.raises(ValueError, match=r"out-of-range whichRHS values \[3\] for n_rhs=2"):
        validate_transport_worker_result_payload(
            rhs_values=[1, 3],
            result=_complete_worker_result(),
            n_rhs=2,
        )


def test_validate_transport_worker_result_payload_rejects_zero_rhs() -> None:
    with pytest.raises(ValueError, match=r"invalid whichRHS values \[0\]"):
        validate_transport_worker_result_payload(
            rhs_values=[0, 1],
            result=_complete_worker_result(),
            n_rhs=2,
        )


def test_validate_complete_transport_worker_rhs_coverage_rejects_missing_rhs() -> None:
    with pytest.raises(ValueError, match=r"missing whichRHS values \[2\]"):
        validate_complete_transport_worker_rhs_coverage(seen_rhs={1, 3}, n_rhs=3)


def test_validate_complete_transport_worker_rhs_coverage_rejects_extra_rhs() -> None:
    with pytest.raises(ValueError, match=r"out-of-range whichRHS values \[4\] for n_rhs=3"):
        validate_complete_transport_worker_rhs_coverage(seen_rhs={1, 2, 3, 4}, n_rhs=3)


def test_validate_gpu_transport_worker_arrays_rejects_rhs_coverage_mismatch() -> None:
    with pytest.raises(RuntimeError, match=r"unexpected whichRHS coverage .*requested=\[1, 2\] returned=\[2, 1\]"):
        validate_gpu_transport_worker_arrays(
            requested_rhs_values=[1, 2],
            output_rhs_values=[2, 1],
            state_vectors=np.ones((2, 3), dtype=np.float64),
            residual_norms=np.ones((2,), dtype=np.float64),
            rhs_norms=np.ones((2,), dtype=np.float64),
            elapsed_time_s=np.ones((2,), dtype=np.float64),
            gpu_id="0",
        )


def test_validate_gpu_transport_worker_arrays_rejects_length_mismatch() -> None:
    with pytest.raises(RuntimeError, match=r"inconsistent result array lengths .*elapsed_time_s=1"):
        validate_gpu_transport_worker_arrays(
            requested_rhs_values=[1, 2],
            output_rhs_values=[1, 2],
            state_vectors=np.ones((2, 3), dtype=np.float64),
            residual_norms=np.ones((2,), dtype=np.float64),
            rhs_norms=np.ones((2,), dtype=np.float64),
            elapsed_time_s=np.ones((1,), dtype=np.float64),
            gpu_id="1",
        )


def test_validate_gpu_transport_worker_arrays_accepts_matching_arrays() -> None:
    validate_gpu_transport_worker_arrays(
        requested_rhs_values=[1, 2],
        output_rhs_values=[1, 2],
        state_vectors=np.ones((2, 3), dtype=np.float64),
        residual_norms=np.ones((2,), dtype=np.float64),
        rhs_norms=np.ones((2,), dtype=np.float64),
        elapsed_time_s=np.ones((2,), dtype=np.float64),
        gpu_id="0",
    )
