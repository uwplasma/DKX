from __future__ import annotations

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np

from sfincs_jax.transport_parallel_execution import (
    build_transport_parallel_payloads,
    run_transport_parallel_payloads,
    should_run_transport_parallel,
)


def test_should_run_transport_parallel_requires_real_parallel_context(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    assert should_run_transport_parallel(
        parallel_child=False,
        parallel_workers=2,
        which_rhs_values=[1, 2],
        input_namelist=input_path,
    )
    assert not should_run_transport_parallel(
        parallel_child=True,
        parallel_workers=2,
        which_rhs_values=[1, 2],
        input_namelist=input_path,
    )
    assert not should_run_transport_parallel(
        parallel_child=False,
        parallel_workers=1,
        which_rhs_values=[1, 2],
        input_namelist=input_path,
    )


def test_build_transport_parallel_payloads_preserves_solver_inputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    payloads = build_transport_parallel_payloads(
        chunks=[[1, 3], [2]],
        input_namelist=input_path,
        tol=1.0e-10,
        atol=1.0e-14,
        restart=80,
        maxiter=400,
        solve_method="auto",
        identity_shift=0.25,
        collect_transport_output_fields=False,
        phi1_hat_base=np.array([1.0, 2.0]),
        differentiable=False,
    )
    assert [p["which_rhs_values"] for p in payloads] == [[1, 3], [2]]
    assert payloads[0]["input_path"] == str(input_path)
    assert payloads[0]["differentiable"] is False
    np.testing.assert_allclose(payloads[0]["phi1_hat_base"], np.array([1.0, 2.0]))


def test_run_transport_parallel_payloads_uses_gpu_runner() -> None:
    payloads = [{"which_rhs_values": [1]}]

    def _gpu(**kwargs):
        assert kwargs["payloads"] == payloads
        return [{"which_rhs_values": [1], "state_vectors_by_rhs": {}, "residual_norms_by_rhs": {}, "elapsed_time_s": np.array([0.1])}]

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=2,
        parallel_backend="gpu",
        run_gpu_subprocesses=_gpu,
        persistent_pool_enabled=False,
        get_pool=lambda **_kwargs: None,
        shutdown_pool=lambda: None,
        worker=lambda payload: payload,
        worker_env=lambda _n: _DummyEnv(),  # unreachable on GPU branch
        executor_class=None,
        executor_kwargs=lambda **_kwargs: {},
        emit=None,
    )
    assert results[0]["which_rhs_values"] == [1]


class _DummyPool:
    def __init__(self, results):
        self._results = results

    def submit(self, worker, payload):
        fut = concurrent.futures.Future()
        fut.set_result(worker(payload))
        return fut


def test_run_transport_parallel_payloads_retries_broken_pool_then_falls_back() -> None:
    payloads = [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]
    shutdown_calls: list[str] = []
    get_calls = {"n": 0}

    def _get_pool(**_kwargs):
        get_calls["n"] += 1
        if get_calls["n"] == 1:
            raise BrokenProcessPool("boom")
        raise RuntimeError("still broken")

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=2,
        parallel_backend="cpu",
        run_gpu_subprocesses=lambda **_kwargs: [],
        persistent_pool_enabled=True,
        get_pool=_get_pool,
        shutdown_pool=lambda: shutdown_calls.append("x"),
        worker=lambda payload: {"which_rhs_values": payload["which_rhs_values"]},
        worker_env=lambda _n: _DummyEnv(),  # unreachable on persistent branch
        executor_class=None,
        executor_kwargs=lambda **_kwargs: {},
        emit=None,
    )
    assert shutdown_calls == ["x"]
    assert [res["which_rhs_values"] for res in results] == [[1], [2]]


class _DummyContextPool(_DummyPool):
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _DummyEnv:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


def test_run_transport_parallel_payloads_nonpersistent_process_pool() -> None:
    payloads = [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]
    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=2,
        parallel_backend="cpu",
        run_gpu_subprocesses=lambda **_kwargs: [],
        persistent_pool_enabled=False,
        get_pool=lambda **_kwargs: None,
        shutdown_pool=lambda: None,
        worker=lambda payload: {"which_rhs_values": payload["which_rhs_values"]},
        worker_env=lambda _n: _DummyEnv(),
        executor_class=lambda **_kwargs: _DummyContextPool(payloads),
        executor_kwargs=lambda **_kwargs: {},
        emit=None,
    )
    assert sorted((tuple(res["which_rhs_values"]) for res in results)) == [(1,), (2,)]
