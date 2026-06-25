from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sfincs_jax
from sfincs_jax.compare import _as_numpy, _center_fsa, _merge_tolerance_floor
from sfincs_jax.solvers.state import (
    _op_signature,
    operator_shape_signature,
    operator_shape_signature_dict,
    read_krylov_state_signature,
    load_krylov_state,
    save_krylov_state,
)


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


def test_solver_state_round_trip_and_signature_guard(tmp_path: Path) -> None:
    op = SimpleNamespace(
        rhs_mode=1,
        total_size=5,
        n_species=2,
        n_x=3,
        n_xi=4,
        n_theta=5,
        n_zeta=6,
        constraint_scheme=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        quasineutrality_option=2,
    )
    path = tmp_path / "state.npz"
    x_full = np.asarray([1.0, 2.0, 3.0])
    x_by_rhs = {3: np.asarray([4.0, 5.0]), 1: np.asarray([6.0, 7.0])}
    x_history = [np.asarray([1.0, 1.0]), np.asarray([2.0, 2.0])]

    sig = _op_signature(op)
    assert sig.tolist() == [1, 5, 2, 3, 4, 5, 6, 1, 1, 0, 2]
    assert operator_shape_signature(op) == tuple(sig.tolist())
    assert operator_shape_signature_dict(op) == {
        "rhs_mode": 1,
        "total_size": 5,
        "n_species": 2,
        "n_x": 3,
        "n_xi": 4,
        "n_theta": 5,
        "n_zeta": 6,
        "constraint_scheme": 1,
        "include_phi1": 1,
        "include_phi1_in_kinetic": 0,
        "quasineutrality_option": 2,
    }

    save_krylov_state(path=path, op=op, x_full=x_full, x_by_rhs=x_by_rhs, x_history=x_history)
    assert read_krylov_state_signature(path) == tuple(sig.tolist())
    loaded = load_krylov_state(path=path, op=op)
    assert loaded is not None
    np.testing.assert_allclose(loaded["x_full"], x_full)
    assert sorted(loaded["x_by_rhs"]) == [1, 3]
    np.testing.assert_allclose(loaded["x_by_rhs"][1], np.asarray([6.0, 7.0]))
    assert len(loaded["x_history"]) == 2

    mismatched = SimpleNamespace(**{**op.__dict__, "n_theta": 99})
    assert load_krylov_state(path=path, op=mismatched) is None
    assert load_krylov_state(path=tmp_path / "missing.npz", op=op) is None


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
