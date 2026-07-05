from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import sfincs_jax.problems.transport_parallel_runtime as transport_parallel_runtime
from sfincs_jax.problems.transport_parallel_runtime import (
    TransportParallelPoolCache,
    plan_transport_parallel_gpu_subprocesses,
    merge_transport_parallel_results,
    partition_transport_rhs,
    run_transport_parallel_gpu_subprocesses,
    run_transport_parallel_payloads,
    summarize_transport_worker_output,
    transport_parallel_result_to_npz_arrays,
    transport_worker_subprocess_env,
    validate_complete_transport_worker_rhs_coverage,
    validate_transport_worker_result_payload,
)


def test_partition_transport_rhs_round_robin() -> None:
    chunks = partition_transport_rhs([1, 2, 3, 4, 5], 3)
    assert chunks == [[1, 4], [2, 5], [3]]


def test_partition_transport_rhs_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="transport parallel worker count must be >= 1; got 0"):
        partition_transport_rhs([1, 2], 0)


def test_merge_transport_parallel_results_handles_subset_elapsed_layouts() -> None:
    results = [
        {
            "which_rhs_values": [1, 3],
            "state_vectors_by_rhs": {1: np.array([1.0]), 3: np.array([3.0])},
            "residual_norms_by_rhs": {1: 1.0e-8, 3: 3.0e-8},
            "rhs_norms_by_rhs": {1: 1.0, 3: 3.0},
            "elapsed_time_s": np.asarray([0.1, 0.3], dtype=np.float64),
        },
        {
            "which_rhs_values": [2],
            "state_vectors_by_rhs": {2: np.array([2.0])},
            "residual_norms_by_rhs": {2: 2.0e-8},
            "rhs_norms_by_rhs": {2: 2.0},
            "elapsed_time_s": np.asarray([0.2], dtype=np.float64),
        },
    ]

    state_vectors, residual_norms, rhs_norms, elapsed_s = merge_transport_parallel_results(n_rhs=3, results=results)

    assert set(state_vectors) == {1, 2, 3}
    assert set(residual_norms) == {1, 2, 3}
    assert rhs_norms == {1: 1.0, 2: 2.0, 3: 3.0}
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
            "rhs_norms_by_rhs": {2: 2.0, 4: 4.0},
            "elapsed_time_s": np.asarray([9.0, 0.2, 9.0, 0.4], dtype=np.float64),
        }
    ]

    state_vectors, residual_norms, rhs_norms, elapsed_s = merge_transport_parallel_results(n_rhs=4, results=results)

    assert set(state_vectors) == {2, 4}
    assert set(residual_norms) == {2, 4}
    assert rhs_norms == {2: 2.0, 4: 4.0}
    np.testing.assert_allclose(elapsed_s, np.array([0.0, 0.2, 0.0, 0.4]))


def test_transport_parallel_result_to_npz_arrays_covers_elapsed_layouts() -> None:
    empty = transport_parallel_result_to_npz_arrays({})
    assert empty["which_rhs_values"].shape == (0,)
    assert empty["state_vectors"].shape == (0, 0)

    scalar_elapsed = transport_parallel_result_to_npz_arrays(
        {
            "which_rhs_values": [2, 4],
            "state_vectors_by_rhs": {2: np.asarray([2.0]), 4: np.asarray([4.0])},
            "residual_norms_by_rhs": {2: 2.0e-12, 4: 4.0e-12},
            "rhs_norms_by_rhs": {2: 2.0, 4: 4.0},
            "elapsed_time_s": np.asarray(0.75),
        }
    )
    np.testing.assert_allclose(scalar_elapsed["elapsed_time_s"], np.asarray([0.75, 0.75]))
    np.testing.assert_allclose(scalar_elapsed["rhs_norms"], np.asarray([2.0, 4.0]))

    indexed_elapsed = transport_parallel_result_to_npz_arrays(
        {
            "which_rhs_values": [2, 4],
            "state_vectors_by_rhs": {2: np.asarray([2.0]), 4: np.asarray([4.0])},
            "residual_norms_by_rhs": {2: 2.0e-12, 4: 4.0e-12},
            "elapsed_time_s": np.asarray([9.0, 0.2, 9.0, 0.4], dtype=np.float64),
        }
    )
    np.testing.assert_allclose(indexed_elapsed["elapsed_time_s"], np.asarray([0.2, 0.4]))
    np.testing.assert_allclose(indexed_elapsed["rhs_norms"], np.asarray([np.nan, np.nan]))

    partial_elapsed = transport_parallel_result_to_npz_arrays(
        {
            "which_rhs_values": [3, 1],
            "state_vectors_by_rhs": {1: np.asarray([1.0]), 3: np.asarray([3.0])},
            "residual_norms_by_rhs": {1: 1.0e-12, 3: 3.0e-12},
            "elapsed_time_s": np.asarray([0.3], dtype=np.float64),
        }
    )
    np.testing.assert_allclose(partial_elapsed["elapsed_time_s"], np.asarray([0.3, 0.0]))


def test_merge_transport_parallel_results_rejects_duplicate_rhs_coverage() -> None:
    results = [
        {
            "which_rhs_values": [1],
            "state_vectors_by_rhs": {1: np.array([1.0])},
            "residual_norms_by_rhs": {1: 1.0e-8},
            "rhs_norms_by_rhs": {1: 1.0},
        },
        {
            "which_rhs_values": [1],
            "state_vectors_by_rhs": {1: np.array([2.0])},
            "residual_norms_by_rhs": {1: 2.0e-8},
            "rhs_norms_by_rhs": {1: 2.0},
        },
    ]

    with pytest.raises(ValueError, match=r"duplicate whichRHS values \[1\]"):
        merge_transport_parallel_results(n_rhs=2, results=results)


def test_merge_transport_parallel_results_rejects_missing_payload_entries() -> None:
    results = [
        {
            "which_rhs_values": [1, 2],
            "state_vectors_by_rhs": {1: np.array([1.0])},
            "residual_norms_by_rhs": {1: 1.0e-8, 2: 2.0e-8},
            "rhs_norms_by_rhs": {1: 1.0, 2: 2.0},
        },
    ]

    with pytest.raises(ValueError, match=r"missing state_vectors_by_rhs entries for whichRHS=\[2\]"):
        merge_transport_parallel_results(n_rhs=2, results=results)


def test_merge_transport_parallel_results_rejects_incomplete_rhs_coverage() -> None:
    results = [
        {
            "which_rhs_values": [1, 3],
            "state_vectors_by_rhs": {1: np.array([1.0]), 3: np.array([3.0])},
            "residual_norms_by_rhs": {1: 1.0e-8, 3: 3.0e-8},
            "rhs_norms_by_rhs": {1: 1.0, 3: 3.0},
        },
    ]

    with pytest.raises(ValueError, match=r"missing whichRHS values \[2\]"):
        merge_transport_parallel_results(n_rhs=3, results=results, require_complete_coverage=True)


def test_run_transport_parallel_gpu_subprocesses_collects_completed_workers(
    monkeypatch,
) -> None:
    messages: list[str] = []
    launched_envs: list[dict[str, str]] = []
    launched_cmds: list[list[str]] = []

    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            launched_cmds.append(list(cmd))
            launched_envs.append(dict(_kwargs["env"]))
            payload_path = Path(cmd[cmd.index("--payload") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            payload = json.loads(payload_path.read_text())
            rhs_values = np.asarray(payload["which_rhs_values"], dtype=np.int32)
            state_vectors = np.ones((len(rhs_values), 2), dtype=np.float64)
            residual_norms = np.full((len(rhs_values),), 1.0e-12, dtype=np.float64)
            rhs_norms = np.full((len(rhs_values),), 1.0, dtype=np.float64)
            elapsed_time_s = np.full((len(rhs_values),), 0.25, dtype=np.float64)
            np.savez(
                output_path,
                which_rhs_values=rhs_values,
                state_vectors=state_vectors,
                residual_norms=residual_norms,
                rhs_norms=rhs_norms,
                elapsed_time_s=elapsed_time_s,
            )

        def poll(self):
            return self.returncode

        def communicate(self):
            return "whichRHS=1/3: assembling+solving (rhs_norm=1.000000e+00)\n", ""

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    results = run_transport_parallel_gpu_subprocesses(
        payloads=[{"which_rhs_values": [1, 3]}, {"which_rhs_values": [2]}],
        parallel_workers=2,
        visible_gpu_ids=lambda _workers: ["0", "1"],
        gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        emit=lambda _level, msg: messages.append(msg),
    )

    assert [res["which_rhs_values"] for res in results] == [[1, 3], [2]]
    assert any("GPU transport worker done" in msg for msg in messages)
    assert any("GPU transport worker log" in msg and "rhs_norm=1.000000e+00" in msg for msg in messages)
    assert any("GPU transport worker result" in msg and "relative_residual=1.000000e-12" in msg for msg in messages)
    assert all("sfincs_jax" in env["PYTHONPATH"] for env in launched_envs)
    assert all(cmd[1:3] == ["-m", "sfincs_jax.problems.transport_parallel_runtime"] for cmd in launched_cmds)


def test_run_transport_parallel_gpu_subprocesses_rejects_mismatched_worker_rhs(
    monkeypatch,
) -> None:
    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            output_path = Path(cmd[cmd.index("--output") + 1])
            np.savez(
                output_path,
                which_rhs_values=np.asarray([2], dtype=np.int32),
                state_vectors=np.ones((1, 2), dtype=np.float64),
                residual_norms=np.full((1,), 1.0e-12, dtype=np.float64),
                rhs_norms=np.ones((1,), dtype=np.float64),
                elapsed_time_s=np.full((1,), 0.25, dtype=np.float64),
            )

        def poll(self):
            return self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    with pytest.raises(RuntimeError, match=r"unexpected whichRHS coverage .*requested=\[1\] returned=\[2\]"):
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1]}],
            parallel_workers=1,
            visible_gpu_ids=lambda _workers: ["0"],
            gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )


def test_run_transport_parallel_gpu_subprocesses_rejects_short_worker_arrays(
    monkeypatch,
) -> None:
    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            output_path = Path(cmd[cmd.index("--output") + 1])
            np.savez(
                output_path,
                which_rhs_values=np.asarray([1, 2], dtype=np.int32),
                state_vectors=np.ones((2, 2), dtype=np.float64),
                residual_norms=np.full((1,), 1.0e-12, dtype=np.float64),
                rhs_norms=np.ones((2,), dtype=np.float64),
                elapsed_time_s=np.full((2,), 0.25, dtype=np.float64),
            )

        def poll(self):
            return self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    with pytest.raises(RuntimeError, match=r"inconsistent result array lengths .*residual_norms=1"):
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1, 2]}],
            parallel_workers=1,
            visible_gpu_ids=lambda _workers: ["0"],
            gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )


def test_run_transport_parallel_gpu_subprocesses_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="GPU transport worker count must be >= 1; got 0"):
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1]}],
            parallel_workers=0,
            visible_gpu_ids=lambda _workers: ["0"],
            gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )


def test_gpu_subprocesses_deduplicates_visible_ids_and_reports_plan_cap(
    monkeypatch,
) -> None:
    messages: list[str] = []
    launched_envs: list[dict[str, str]] = []

    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            launched_envs.append(dict(_kwargs["env"]))
            payload_path = Path(cmd[cmd.index("--payload") + 1])
            output_path = Path(cmd[cmd.index("--output") + 1])
            payload = json.loads(payload_path.read_text())
            rhs_values = np.asarray(payload["which_rhs_values"], dtype=np.int32)
            np.savez(
                output_path,
                which_rhs_values=rhs_values,
                state_vectors=np.ones((len(rhs_values), 2), dtype=np.float64),
                residual_norms=np.full((len(rhs_values),), 1.0e-12, dtype=np.float64),
                rhs_norms=np.ones((len(rhs_values),), dtype=np.float64),
                elapsed_time_s=np.full((len(rhs_values),), 0.25, dtype=np.float64),
            )

        def poll(self):
            return self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    results = run_transport_parallel_gpu_subprocesses(
        payloads=[{"which_rhs_values": [1]}, {"which_rhs_values": [2]}, {"which_rhs_values": [3]}],
        parallel_workers=3,
        visible_gpu_ids=lambda _workers: ["0", "0", "1"],
        gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        emit=lambda _level, msg: messages.append(msg),
    )

    assert [res["which_rhs_values"] for res in results] == [[1, 3], [2]]
    state_vectors, residual_norms, rhs_norms, elapsed_s = merge_transport_parallel_results(
        n_rhs=3,
        results=results,
    )
    assert set(state_vectors) == {1, 2, 3}
    assert set(residual_norms) == {1, 2, 3}
    assert set(rhs_norms) == {1, 2, 3}
    np.testing.assert_allclose(elapsed_s, np.asarray([0.25, 0.25, 0.25]))
    assert [env["CUDA_VISIBLE_DEVICES"] for env in launched_envs] == ["0", "1"]
    assert any(
        "GPU transport worker plan capped" in msg
        and "active=2 requested=3" in msg
        and "unique visible GPU ids=2" in msg
        for msg in messages
    )


def test_transport_worker_subprocess_env_prepends_repo_root() -> None:
    env = transport_worker_subprocess_env({"PYTHONPATH": "existing"})

    parts = env["PYTHONPATH"].split(":")
    assert (Path(parts[0]) / "sfincs_jax").is_dir()
    assert parts[1] == "existing"


def test_summarize_transport_worker_output_filters_noise() -> None:
    text = "\n".join(
        [
            "import chatter",
            "whichRHS=2/3: assembling+solving (rhs_norm=2.0e-04)",
            "solve_v3_transport_matrix_linear_gmres: preconditioner=sxblock strong=xmg",
            "whichRHS=2: residual_norm=1.0e-10 elapsed_s=0.2",
        ]
    )

    lines = summarize_transport_worker_output(text)

    assert lines == [
        "whichRHS=2/3: assembling+solving (rhs_norm=2.0e-04)",
        "solve_v3_transport_matrix_linear_gmres: preconditioner=sxblock strong=xmg",
        "whichRHS=2: residual_norm=1.0e-10 elapsed_s=0.2",
    ]


def test_summarize_transport_worker_output_truncates_long_logs() -> None:
    text = "\n".join(
        f"whichRHS={idx}: residual_norm={idx}.0e-12 elapsed_s=0.{idx}"
        for idx in range(1, 9)
    )

    lines = summarize_transport_worker_output(text, max_lines=5)

    assert lines[0].startswith("whichRHS=1")
    assert lines[1] == "..."
    assert lines[-1].startswith("whichRHS=8")
    assert len(lines) == 5


def test_gpu_subprocess_plan_handles_empty_payloads_and_rejects_missing_gpus() -> None:
    empty = plan_transport_parallel_gpu_subprocesses(
        payloads=[],
        parallel_workers=4,
        visible_gpu_ids=[],
    )
    assert empty["active_workers"] == 0
    assert empty["worker_assignments"] == []
    assert empty["capped"] is False

    with pytest.raises(RuntimeError, match="no visible GPU ids"):
        plan_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1]}],
            parallel_workers=1,
            visible_gpu_ids=["", " "],
        )


def test_transport_worker_payload_validators_fail_closed() -> None:
    with pytest.raises(ValueError, match="invalid whichRHS"):
        validate_transport_worker_result_payload(
            rhs_values=[0],
            result={
                "state_vectors_by_rhs": {0: np.asarray([0.0])},
                "residual_norms_by_rhs": {0: 0.0},
                "rhs_norms_by_rhs": {0: 1.0},
            },
            n_rhs=2,
        )

    with pytest.raises(ValueError, match="out-of-range whichRHS"):
        validate_transport_worker_result_payload(
            rhs_values=[3],
            result={
                "state_vectors_by_rhs": {3: np.asarray([0.0])},
                "residual_norms_by_rhs": {3: 0.0},
                "rhs_norms_by_rhs": {3: 1.0},
            },
            n_rhs=2,
        )

    with pytest.raises(ValueError, match="must be a mapping"):
        validate_transport_worker_result_payload(
            rhs_values=[1],
            result={
                "state_vectors_by_rhs": [],
                "residual_norms_by_rhs": {1: 0.0},
                "rhs_norms_by_rhs": {1: 1.0},
            },
            n_rhs=2,
        )

    with pytest.raises(ValueError, match="out-of-range whichRHS values \\[3\\]"):
        validate_complete_transport_worker_rhs_coverage(seen_rhs={1, 2, 3}, n_rhs=2)


def test_run_transport_parallel_payloads_falls_back_when_persistent_pool_breaks() -> None:
    messages: list[str] = []
    calls = {"get_pool": 0, "worker": 0, "shutdown": 0}
    payloads = [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]

    def _get_pool(**_kwargs):
        calls["get_pool"] += 1
        if calls["get_pool"] == 1:
            raise transport_parallel_runtime.BrokenProcessPool("broken")
        raise OSError("retry unavailable")

    def _worker(payload):
        calls["worker"] += 1
        return {"which_rhs_values": payload["which_rhs_values"]}

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=2,
        parallel_backend="cpu",
        run_gpu_subprocesses=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GPU path")),
        persistent_pool_enabled=True,
        get_pool=_get_pool,
        shutdown_pool=lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
        worker=_worker,
        worker_env=lambda _workers: (_ for _ in ()).throw(AssertionError("persistent pool should not use worker_env")),
        executor_class=object,
        executor_kwargs=lambda **_kwargs: {},
        emit=lambda _level, message: messages.append(message),
    )

    assert results == [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]
    assert calls == {"get_pool": 2, "worker": 2, "shutdown": 1}
    assert any("persistent transport pool broke" in message for message in messages)
    assert any("falling back to sequential whichRHS" in message for message in messages)


def test_transport_parallel_pool_cache_reuses_and_replaces_matching_keys() -> None:
    cache = TransportParallelPoolCache()
    shutdowns: list[str] = []

    class _Pool:
        def __init__(self, label: str):
            self.label = label

        def shutdown(self, **_kwargs):
            shutdowns.append(self.label)

    class _WorkerEnv:
        def __init__(self, workers: int):
            self.workers = workers

        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    created: list[str] = []

    def _executor_class(**kwargs):
        label = f"pool-{kwargs['label']}"
        created.append(label)
        return _Pool(label)

    pool1 = cache.get(
        parallel_workers=2,
        key_fn=lambda workers: ("key", workers),
        worker_env=lambda workers: _WorkerEnv(workers),
        executor_kwargs=lambda **_kwargs: {"label": "a"},
        executor_class=_executor_class,
    )
    pool2 = cache.get(
        parallel_workers=2,
        key_fn=lambda workers: ("key", workers),
        worker_env=lambda workers: _WorkerEnv(workers),
        executor_kwargs=lambda **_kwargs: {"label": "b"},
        executor_class=_executor_class,
    )
    pool3 = cache.get(
        parallel_workers=3,
        key_fn=lambda workers: ("key", workers),
        worker_env=lambda workers: _WorkerEnv(workers),
        executor_kwargs=lambda **_kwargs: {"label": "c"},
        executor_class=_executor_class,
    )
    cache.shutdown()

    assert pool1 is pool2
    assert pool3 is not pool1
    assert created == ["pool-a", "pool-c"]
    assert shutdowns == ["pool-a", "pool-c"]


def test_gpu_subprocesses_abort_pending_workers_on_residual_gate(
    monkeypatch,
) -> None:
    terminated: list[int] = []

    class _FakeProc:
        _counter = 0

        def __init__(self, cmd, **_kwargs):
            self.idx = _FakeProc._counter
            _FakeProc._counter += 1
            self.poll_count = 0
            self.returncode = 0 if self.idx == 0 else None
            output_path = Path(cmd[cmd.index("--output") + 1])
            if self.idx == 0:
                np.savez(
                    output_path,
                    which_rhs_values=np.asarray([2], dtype=np.int32),
                    state_vectors=np.ones((1, 2), dtype=np.float64),
                    residual_norms=np.asarray([1.0e-3], dtype=np.float64),
                    rhs_norms=np.asarray([1.0e-3], dtype=np.float64),
                    elapsed_time_s=np.asarray([1.0], dtype=np.float64),
                )
            else:
                np.savez(
                    output_path,
                    which_rhs_values=np.asarray([1, 3], dtype=np.int32),
                    state_vectors=np.ones((2, 2), dtype=np.float64),
                    residual_norms=np.asarray([1.0e-12, 1.0e-12], dtype=np.float64),
                    rhs_norms=np.asarray([1.0, 1.0], dtype=np.float64),
                    elapsed_time_s=np.asarray([1.0, 1.0], dtype=np.float64),
                )

        def poll(self):
            if self.idx != 0 and self.returncode is None:
                self.poll_count += 1
                if self.poll_count > 1:
                    self.returncode = 0
            return self.returncode

        def communicate(self):
            return "", ""

        def terminate(self):
            terminated.append(self.idx)
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RESIDUAL", "1e-6")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_ABORT_MAX_RELATIVE_RESIDUAL", "1e-6")
    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    try:
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [2]}, {"which_rhs_values": [1, 3]}],
            parallel_workers=2,
            visible_gpu_ids=lambda _workers: ["0", "1"],
            gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
    except RuntimeError as exc:
        assert "residual gate failed" in str(exc)
        assert "whichRHS=2" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected residual-gate failure")

    assert terminated == [1]


def test_gpu_subprocesses_classify_worker_residual_gate_exit(
    monkeypatch,
) -> None:
    terminated: list[int] = []

    class _FakeProc:
        _counter = 0

        def __init__(self, _cmd, **_kwargs):
            self.idx = _FakeProc._counter
            _FakeProc._counter += 1
            self.poll_count = 0
            self.returncode = 1 if self.idx == 0 else None

        def poll(self):
            if self.idx != 0 and self.returncode is None:
                return None
            return self.returncode

        def communicate(self):
            if self.idx == 0:
                return (
                    "solve_v3_transport_matrix_linear_gmres: transport residual gate failed; "
                    "aborting remaining whichRHS solves "
                    "(whichRHS=1 residual_norm=1.0e-04 rhs_norm=1.0e-04 relative_residual=1.0e+00)\n",
                    "RuntimeError: transport residual gate failed: whichRHS=1 "
                    "residual_norm=1.0e-04 rhs_norm=1.0e-04 relative_residual=1.0e+00\n",
                )
            return "", ""

        def terminate(self):
            terminated.append(self.idx)
            self.returncode = -15

        def kill(self):
            self.returncode = -9

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    try:
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1]}, {"which_rhs_values": [2]}],
            parallel_workers=2,
            visible_gpu_ids=lambda _workers: ["0", "1"],
            gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        )
    except RuntimeError as exc:
        assert "GPU transport worker residual gate failed" in str(exc)
        assert "whichRHS=1" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected residual-gate failure")

    assert terminated == [1]
