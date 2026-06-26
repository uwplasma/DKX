from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.transport_parallel_runtime import (
    pack_transport_parallel_result,
    solve_transport_parallel_payload,
    transport_parallel_result_to_npz_arrays,
)


def test_solve_transport_parallel_payload_normalizes_kwargs_and_child_env(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    captured: dict[str, object] = {}

    def _read_input(path: Path) -> object:
        captured["read_path"] = path
        return {"input": "model"}

    def _solve_transport(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            state_vectors_by_rhs={2: np.asarray([1.0, 2.0])},
            residual_norms_by_rhs={2: np.asarray(3.0e-12)},
            rhs_norms_by_rhs={2: np.asarray(4.0)},
            elapsed_time_s=np.asarray([9.0, 0.25], dtype=np.float64),
        )

    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL", raising=False)
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_PARALLEL_CHILD", raising=False)
    result = solve_transport_parallel_payload(
        {
            "input_path": str(input_path),
            "which_rhs_values": [2],
            "tol": "1e-8",
            "atol": "1e-12",
            "restart": "17",
            "maxiter": 23,
            "solve_method": "xblock",
            "identity_shift": "0.125",
            "differentiable": False,
            "phi1_hat_base": [0.0, 1.0],
        },
        read_input=_read_input,
        solve_transport=_solve_transport,
    )

    assert captured["read_path"] == input_path
    assert captured["nml"] == {"input": "model"}
    assert captured["which_rhs_values"] == [2]
    assert captured["tol"] == 1.0e-8
    assert captured["atol"] == 1.0e-12
    assert captured["restart"] == 17
    assert captured["maxiter"] == 23
    assert captured["solve_method"] == "xblock"
    assert captured["identity_shift"] == 0.125
    assert captured["differentiable"] is False
    assert captured["force_stream_diagnostics"] is True
    assert captured["force_store_state"] is True
    assert captured["collect_transport_output_fields"] is False
    assert captured["parallel_workers"] == 1
    np.testing.assert_allclose(np.asarray(captured["phi1_hat_base"]), np.asarray([0.0, 1.0]))
    assert result["which_rhs_values"] == [2]
    np.testing.assert_allclose(result["elapsed_time_s"], np.asarray([9.0, 0.25]))
    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL"] == "off"
    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] == "1"


def test_pack_transport_parallel_result_preserves_optional_rhs_norms() -> None:
    result = SimpleNamespace(
        state_vectors_by_rhs={1: np.asarray([1.0])},
        residual_norms_by_rhs={1: np.asarray(2.0e-11)},
        elapsed_time_s=np.asarray([0.5], dtype=np.float64),
    )

    packed = pack_transport_parallel_result(which_rhs_values=[1], result=result)

    assert packed["which_rhs_values"] == [1]
    assert packed["rhs_norms_by_rhs"] == {}
    np.testing.assert_allclose(packed["state_vectors_by_rhs"][1], np.asarray([1.0]))
    assert packed["residual_norms_by_rhs"][1] == 2.0e-11


def test_transport_parallel_result_to_npz_arrays_indexes_full_elapsed_vector() -> None:
    arrays = transport_parallel_result_to_npz_arrays(
        {
            "which_rhs_values": [2, 4],
            "state_vectors_by_rhs": {
                2: np.asarray([2.0, 20.0]),
                4: np.asarray([4.0, 40.0]),
            },
            "residual_norms_by_rhs": {2: 2.0e-12, 4: 4.0e-12},
            "rhs_norms_by_rhs": {2: 2.0, 4: 4.0},
            "elapsed_time_s": np.asarray([99.0, 0.2, 99.0, 0.4], dtype=np.float64),
        }
    )

    np.testing.assert_array_equal(arrays["which_rhs_values"], np.asarray([2, 4], dtype=np.int32))
    np.testing.assert_allclose(
        arrays["state_vectors"],
        np.asarray([[2.0, 20.0], [4.0, 40.0]], dtype=np.float64),
    )
    np.testing.assert_allclose(arrays["residual_norms"], np.asarray([2.0e-12, 4.0e-12]))
    np.testing.assert_allclose(arrays["rhs_norms"], np.asarray([2.0, 4.0]))
    np.testing.assert_allclose(arrays["elapsed_time_s"], np.asarray([0.2, 0.4]))


def test_transport_parallel_result_to_npz_arrays_handles_empty_payload() -> None:
    arrays = transport_parallel_result_to_npz_arrays({"which_rhs_values": []})

    assert arrays["which_rhs_values"].shape == (0,)
    assert arrays["state_vectors"].shape == (0, 0)
    assert arrays["residual_norms"].shape == (0,)
    assert arrays["rhs_norms"].shape == (0,)
    assert arrays["elapsed_time_s"].shape == (0,)
