from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sfincs_jax.namelist as namelist_module
import sfincs_jax.problems.transport_parallel_runtime as worker_cli
import sfincs_jax.problems.transport_solve as transport_solve


def _result_fixture() -> SimpleNamespace:
    return SimpleNamespace(
        state_vectors_by_rhs={
            1: np.asarray([1.0, 2.0], dtype=np.float64),
            3: np.asarray([3.0, 4.0], dtype=np.float64),
        },
        residual_norms_by_rhs={
            1: np.asarray(1.0e-11, dtype=np.float64),
            3: np.asarray(3.0e-11, dtype=np.float64),
        },
        rhs_norms_by_rhs={
            1: np.asarray(10.0, dtype=np.float64),
            3: np.asarray(30.0, dtype=np.float64),
        },
        elapsed_time_s=np.asarray([0.1, 0.2, 0.3], dtype=np.float64),
    )


def test_transport_parallel_worker_main_writes_npz_schema_and_creates_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.namelist"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "nested" / "worker_output.npz"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    payload_path.write_text(
        json.dumps(
            {
                "input_path": str(input_path),
                "which_rhs_values": [3, 1],
                "tol": 1.0e-9,
                "atol": 1.0e-12,
                "restart": 17,
                "maxiter": 23,
                "solve_method": "auto",
                "identity_shift": 0.125,
                "differentiable": False,
            }
        ),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}

    def _read_input(path: Path) -> object:
        captured["read_input"] = path
        return {"nml": "fixture"}

    def _solve_transport(**kwargs):
        captured.update(kwargs)
        return _result_fixture()

    monkeypatch.setattr(worker_cli, "_read_worker_input", _read_input)
    monkeypatch.setattr(worker_cli, "_solve_worker_transport", _solve_transport)
    monkeypatch.setattr(
        sys,
        "argv",
        ["transport_parallel_worker", "--payload", str(payload_path), "--output", str(output_path)],
    )

    assert worker_cli.main() == 0

    assert captured["read_input"] == input_path
    assert captured["which_rhs_values"] == [3, 1]
    assert captured["tol"] == 1.0e-9
    assert captured["atol"] == 1.0e-12
    assert captured["restart"] == 17
    assert captured["maxiter"] == 23
    assert captured["identity_shift"] == 0.125
    assert captured["differentiable"] is False
    assert output_path.exists()
    with np.load(output_path) as data:
        np.testing.assert_array_equal(data["which_rhs_values"], np.asarray([3, 1], dtype=np.int32))
        np.testing.assert_allclose(data["state_vectors"], np.asarray([[3.0, 4.0], [1.0, 2.0]]))
        np.testing.assert_allclose(data["residual_norms"], np.asarray([3.0e-11, 1.0e-11]))
        np.testing.assert_allclose(data["rhs_norms"], np.asarray([30.0, 10.0]))
        np.testing.assert_allclose(data["elapsed_time_s"], np.asarray([0.3, 0.1]))


def test_transport_parallel_worker_module_entrypoint_exits_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "input.namelist"
    payload_path = tmp_path / "payload.json"
    output_path = tmp_path / "worker_output.npz"
    input_path.write_text("&general\n/\n", encoding="utf-8")
    payload_path.write_text(
        json.dumps({"input_path": str(input_path), "which_rhs_values": [1]}),
        encoding="utf-8",
    )

    monkeypatch.setattr(namelist_module, "read_sfincs_input", lambda _path: {"nml": "fixture"})
    monkeypatch.setattr(
        transport_solve,
        "solve_v3_transport_matrix_linear_gmres",
        lambda **_kwargs: _result_fixture(),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["transport_parallel_worker", "--payload", str(payload_path), "--output", str(output_path)],
    )
    monkeypatch.delitem(sys.modules, "sfincs_jax.problems.transport_parallel_runtime", raising=False)

    with pytest.raises(SystemExit) as exc:
        runpy.run_module("sfincs_jax.problems.transport_parallel_runtime", run_name="__main__")

    assert exc.value.code == 0
    with np.load(output_path) as data:
        np.testing.assert_array_equal(data["which_rhs_values"], np.asarray([1], dtype=np.int32))
        np.testing.assert_allclose(data["rhs_norms"], np.asarray([10.0]))
