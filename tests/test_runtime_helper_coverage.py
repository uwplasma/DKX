from __future__ import annotations

import numpy as np
import pytest

import sfincs_jax
from sfincs_jax.compare import _as_numpy, _center_fsa, _merge_tolerance_floor


def test_initialize_distributed_runtime_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    import jax.distributed as jax_distributed

    calls: list[dict[str, int | str]] = []

    def _fake_initialize(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(jax_distributed, "initialize", _fake_initialize)
    monkeypatch.setattr(sfincs_jax, "_distributed_runtime_initialized", False)

    monkeypatch.delenv("SFINCS_JAX_DISTRIBUTED", raising=False)
    assert sfincs_jax.initialize_distributed_runtime_from_env() is False

    monkeypatch.setenv("SFINCS_JAX_DISTRIBUTED", "1")
    monkeypatch.delenv("SFINCS_JAX_COORDINATOR_ADDRESS", raising=False)
    monkeypatch.setattr(sfincs_jax, "_distributed_runtime_initialized", False)
    assert sfincs_jax.initialize_distributed_runtime_from_env() is False

    monkeypatch.setenv("SFINCS_JAX_COORDINATOR_ADDRESS", "127.0.0.1")
    monkeypatch.setenv("SFINCS_JAX_COORDINATOR_PORT", "2345")
    monkeypatch.setenv("SFINCS_JAX_PROCESS_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_PROCESS_ID", "1")
    monkeypatch.setattr(sfincs_jax, "_distributed_runtime_initialized", False)
    assert sfincs_jax.initialize_distributed_runtime_from_env() is True
    assert sfincs_jax.initialize_distributed_runtime_from_env() is True
    assert len(calls) == 1
    assert calls[0] == {
        "coordinator_address": "127.0.0.1",
        "coordinator_port": 2345,
        "num_processes": 2,
        "process_id": 1,
    }


def test_compare_helper_functions() -> None:
    assert _as_numpy(np.asarray([1.0])).shape == (1,)
    assert _as_numpy(3.0).shape == ()
    assert _as_numpy("x") is None
    assert _as_numpy(np.asarray(["x"], dtype=object)) is None

    arr4 = np.arange(2 * 3 * 4 * 2, dtype=np.float64).reshape(2, 3, 4, 2)
    centered4 = _center_fsa(arr4)
    np.testing.assert_allclose(centered4.mean(axis=(1, 2)), 0.0)

    arr3 = np.arange(3 * 4 * 2, dtype=np.float64).reshape(3, 4, 2)
    centered3 = _center_fsa(arr3)
    np.testing.assert_allclose(centered3.mean(axis=(0, 1)), 0.0)

    tolerances: dict[str, dict[str, float | bool]] = {"A": {"rtol": 1e-8, "ignore": False}}
    _merge_tolerance_floor(tolerances, "A", {"rtol": 1e-6, "atol": 1e-12, "ignore": True, "center_fsa": True})
    assert tolerances["A"]["rtol"] == pytest.approx(1e-6)
    assert tolerances["A"]["atol"] == pytest.approx(1e-12)
    assert tolerances["A"]["ignore"] is True
    assert tolerances["A"]["center_fsa"] is True
