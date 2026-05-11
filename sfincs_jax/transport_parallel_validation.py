from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def format_transport_rhs_list(values: Sequence[int]) -> str:
    return "[" + ", ".join(str(v) for v in values) + "]"


def validate_distinct_transport_worker_rhs(
    *,
    rhs_values: Sequence[int],
    seen_rhs: set[int],
) -> None:
    duplicate_rhs = [int(rhs) for rhs in rhs_values if int(rhs) in seen_rhs]
    if duplicate_rhs:
        raise ValueError(
            "transport parallel worker results contain duplicate whichRHS values "
            f"{format_transport_rhs_list(duplicate_rhs)}"
        )


def validate_transport_worker_result_payload(
    *,
    rhs_values: Sequence[int],
    result: Mapping[str, object],
    n_rhs: int | None,
) -> None:
    rhs_values = [int(rhs) for rhs in rhs_values]
    if any(rhs < 1 for rhs in rhs_values):
        invalid = [rhs for rhs in rhs_values if rhs < 1]
        raise ValueError(
            "transport parallel worker reported invalid whichRHS values "
            f"{format_transport_rhs_list(invalid)}"
        )
    if n_rhs is not None and any(rhs > int(n_rhs) for rhs in rhs_values):
        invalid = [rhs for rhs in rhs_values if rhs > int(n_rhs)]
        raise ValueError(
            "transport parallel worker reported out-of-range whichRHS values "
            f"{format_transport_rhs_list(invalid)} for n_rhs={int(n_rhs)}"
        )

    required_maps = ("state_vectors_by_rhs", "residual_norms_by_rhs", "rhs_norms_by_rhs")
    for key in required_maps:
        value = result.get(key, {})
        if not isinstance(value, dict):
            raise ValueError(f"transport parallel worker result field {key!r} must be a mapping")
        present = {int(k) for k in value}
        missing = [rhs for rhs in rhs_values if rhs not in present]
        if missing:
            raise ValueError(
                "transport parallel worker result is missing "
                f"{key} entries for whichRHS={format_transport_rhs_list(missing)}"
            )


def validate_gpu_transport_worker_arrays(
    *,
    requested_rhs_values: Sequence[int],
    output_rhs_values: Sequence[int],
    state_vectors: np.ndarray,
    residual_norms: np.ndarray,
    rhs_norms: np.ndarray,
    elapsed_time_s: np.ndarray,
    gpu_id: str,
) -> None:
    requested_rhs_values = [int(rhs) for rhs in requested_rhs_values]
    output_rhs_values = [int(rhs) for rhs in output_rhs_values]
    if output_rhs_values != requested_rhs_values:
        raise RuntimeError(
            "GPU transport worker returned unexpected whichRHS coverage "
            f"(gpu={gpu_id} requested={requested_rhs_values} returned={output_rhs_values})"
        )
    expected = len(output_rhs_values)
    lengths = {
        "state_vectors": int(state_vectors.shape[0]) if state_vectors.ndim > 0 else 0,
        "residual_norms": int(residual_norms.shape[0]) if residual_norms.ndim > 0 else 0,
        "rhs_norms": int(rhs_norms.shape[0]) if rhs_norms.ndim > 0 else 0,
        "elapsed_time_s": int(elapsed_time_s.shape[0]) if elapsed_time_s.ndim > 0 else 0,
    }
    bad_lengths = {key: length for key, length in lengths.items() if length != expected}
    if bad_lengths:
        details = ", ".join(f"{key}={length}" for key, length in sorted(bad_lengths.items()))
        raise RuntimeError(
            "GPU transport worker returned inconsistent result array lengths "
            f"(gpu={gpu_id} whichRHS={output_rhs_values} expected={expected}; {details})"
        )


__all__ = [
    "format_transport_rhs_list",
    "validate_distinct_transport_worker_rhs",
    "validate_gpu_transport_worker_arrays",
    "validate_transport_worker_result_payload",
]
