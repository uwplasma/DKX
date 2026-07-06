from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import sfincs_jax.problems.transport_parallel_runtime as transport_parallel_runtime
from sfincs_jax.problems.transport_parallel_runtime import (
    TransportParallelPoolCache,
    plan_transport_parallel_gpu_subprocesses,
    merge_transport_parallel_results,
    partition_transport_rhs,
    pack_transport_parallel_result,
    run_transport_parallel_gpu_subprocesses,
    run_transport_parallel_gpu_subprocesses_with_policy,
    run_transport_parallel_payloads,
    solve_transport_parallel_payload,
    summarize_transport_worker_output,
    transport_parallel_result_to_npz_arrays,
    transport_worker_subprocess_env,
    validate_complete_transport_worker_rhs_coverage,
    validate_gpu_transport_worker_arrays,
    validate_transport_worker_result_payload,
    validate_transport_parallel_worker_count,
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


def test_run_transport_parallel_gpu_subprocesses_reports_periodic_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    messages: list[str] = []
    perf_values = iter([0.0, 0.2, 0.4])

    class _FakeProc:
        returncode = 0

        def __init__(self, cmd, **_kwargs):
            self.poll_count = 0
            output_path = Path(cmd[cmd.index("--output") + 1])
            np.savez(
                output_path,
                which_rhs_values=np.asarray([1], dtype=np.int32),
                state_vectors=np.ones((1, 2), dtype=np.float64),
                residual_norms=np.full((1,), 1.0e-12, dtype=np.float64),
                rhs_norms=np.ones((1,), dtype=np.float64),
                elapsed_time_s=np.full((1,), 0.25, dtype=np.float64),
            )

        def poll(self):
            self.poll_count += 1
            return None if self.poll_count == 1 else self.returncode

        def communicate(self):
            return "", ""

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_STATUS_INTERVAL", "0.1")
    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)
    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.time.sleep", lambda _seconds: None)
    monkeypatch.setattr(
        "sfincs_jax.problems.transport_parallel_runtime.time.perf_counter",
        lambda: next(perf_values),
    )

    results = run_transport_parallel_gpu_subprocesses(
        payloads=[{"which_rhs_values": [1]}],
        parallel_workers=1,
        visible_gpu_ids=lambda _workers: ["0"],
        gpu_worker_env=lambda gpu_id: {"CUDA_VISIBLE_DEVICES": str(gpu_id)},
        emit=lambda _level, message: messages.append(message),
    )

    assert results[0]["which_rhs_values"] == [1]
    assert any("GPU transport workers running" in message for message in messages)


def test_run_transport_parallel_gpu_subprocesses_reports_non_residual_worker_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeProc:
        returncode = 2

        def __init__(self, _cmd, **_kwargs):
            pass

        def poll(self):
            return self.returncode

        def communicate(self):
            return "ordinary stdout", "ordinary stderr"

    monkeypatch.setattr("sfincs_jax.problems.transport_parallel_runtime.subprocess.Popen", _FakeProc)

    with pytest.raises(RuntimeError, match="GPU transport worker failed .*code=2"):
        run_transport_parallel_gpu_subprocesses(
            payloads=[{"which_rhs_values": [1]}],
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


def test_transport_parallel_validation_helpers_reject_bad_scalars() -> None:
    with pytest.raises(ValueError, match="worker count must be an integer"):
        validate_transport_parallel_worker_count("bad")

    with pytest.raises(RuntimeError, match="state_vectors=0"):
        validate_gpu_transport_worker_arrays(
            requested_rhs_values=[1, 2],
            output_rhs_values=[1, 2],
            state_vectors=np.asarray(1.0),
            residual_norms=np.ones((2,), dtype=np.float64),
            rhs_norms=np.ones((2,), dtype=np.float64),
            elapsed_time_s=np.ones((2,), dtype=np.float64),
            gpu_id="0",
        )


def test_gpu_subprocess_policy_wrapper_wires_standard_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, object]] = []

    def _fake_gpu_runner(**kwargs):
        calls.append(kwargs)
        assert kwargs["visible_gpu_ids"] is transport_parallel_runtime.transport_parallel_visible_gpu_ids
        assert kwargs["gpu_worker_env"] is transport_parallel_runtime.transport_parallel_gpu_worker_env
        return [{"which_rhs_values": [1], "state_vectors_by_rhs": {}, "residual_norms_by_rhs": {}}]

    monkeypatch.setattr(
        transport_parallel_runtime,
        "run_transport_parallel_gpu_subprocesses",
        _fake_gpu_runner,
    )

    result = run_transport_parallel_gpu_subprocesses_with_policy(
        payloads=[{"which_rhs_values": [1]}],
        parallel_workers=2,
        emit=lambda _level, _message: None,
    )

    assert result == [{"which_rhs_values": [1], "state_vectors_by_rhs": {}, "residual_norms_by_rhs": {}}]
    assert calls[0]["parallel_workers"] == 2


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


def test_transport_scaling_audit_rejects_malformed_evidence() -> None:
    with pytest.raises(ValueError, match="release_scaling_claim must be a boolean"):
        transport_parallel_runtime.audit_transport_parallel_scaling_summary(
            {
                "backend": "cpu",
                "rhs_count": 2,
                "release_scaling_claim": "maybe",
                "results": [
                    {"workers": 1, "mean_s": 2.0},
                    {"workers": 2, "mean_s": 1.0},
                ],
                "payloads": [
                    {"which_rhs_values": [1]},
                    {"which_rhs_values": [2]},
                ],
            }
        )

    with pytest.raises(ValueError, match="payloads\\[0\\] must include which_rhs_values"):
        transport_parallel_runtime.audit_transport_parallel_scaling_summary(
            {
                "backend": "cpu",
                "rhs_count": 2,
                "results": [
                    {"workers": 1, "mean_s": 2.0},
                    {"workers": 2, "mean_s": 1.0},
                ],
                "payloads": [{}],
            }
        )

    with pytest.raises(ValueError, match="compile_amortization_gate.notes"):
        transport_parallel_runtime.audit_transport_parallel_scaling_summary(
            {
                "backend": "cpu",
                "rhs_count": 2,
                "timing_semantics": "warm",
                "results": [
                    {"workers": 1, "mean_s": 2.0},
                    {"workers": 2, "mean_s": 1.0},
                ],
                "payloads": [
                    {"which_rhs_values": [1]},
                    {"which_rhs_values": [2]},
                ],
                "compile_amortization_gate": {
                    "passes": True,
                    "timing_semantics": "warm",
                    "notes": 1,
                },
            }
        )


def test_rewrite_xla_flags_replaces_stale_worker_limits() -> None:
    rewritten = transport_parallel_runtime.rewrite_xla_flags(
        "--foo=1 --xla_cpu_multi_thread_eigen_num_threads=99 "
        "--xla_cpu_parallelism_threads=8 --xla_force_host_platform_device_count=8",
        cpu_threads=3,
        host_devices=2,
    )

    assert "--foo=1" in rewritten
    assert "--xla_cpu_multi_thread_eigen=true" in rewritten
    assert "--xla_cpu_multi_thread_eigen_num_threads=3" in rewritten
    assert "--xla_force_host_platform_device_count=2" in rewritten
    assert "--xla_cpu_parallelism_threads=8" not in rewritten


def test_transport_scaling_audit_accepts_release_quality_worker_summary() -> None:
    audit = transport_parallel_runtime.audit_transport_parallel_scaling_summary(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "gpu",
            "rhs_count": 4,
            "device_ids": ["0", "1"],
            "timing_semantics": "warm",
            "release_scaling_claim": True,
            "results": [
                {"workers": 1, "mean_s": 4.0},
                {"workers": 2, "mean_s": 2.0},
            ],
            "payloads_by_workers": {
                "2": [
                    {"which_rhs_values": [1, 3]},
                    {"which_rhs_values": [2, 4]},
                ]
            },
            "deterministic_output_check": True,
            "compile_amortization_gate": {
                "passes": True,
                "timing_semantics": "warm",
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "timed_repeats": 2,
                "min_timed_repeats": 2,
                "notes": ["cache warmed"],
            },
        }
    )

    assert audit.release_scaling_claim
    assert audit.claim_speedup == pytest.approx(2.0)
    assert audit.claim_efficiency == pytest.approx(1.0)
    assert audit.deterministic_payload_coverage
    assert audit.compile_amortization_gate
    assert not audit.failures


@pytest.mark.parametrize(
    ("summary", "expected_scope", "supported"),
    (
        (
            {
                "benchmark_kind": "transport_worker_scaling",
                "backend": "cpu",
                "rhs_count": 2,
                "workers": [1, 2],
                "results": [
                    {"workers": 1, "mean_s": 2.0},
                    {"workers": 2, "mean_s": 1.0},
                ],
            },
            "independent_transport_worker_throughput",
            True,
        ),
        (
            {
                "benchmark_kind": "single_case_sharded_solve",
                "backend": "gpu",
                "experimental_single_case_scaling": True,
                "devices": [1, 2],
                "results": [
                    {"devices": 1, "mean_s": 2.0},
                    {"devices": 2, "mean_s": 1.5},
                ],
            },
            "single_case_sharded_solve_experimental",
            False,
        ),
        (
            {
                "benchmark_kind": "multi_gpu_case_throughput",
                "backend": "gpu",
                "required_gpu_count": 2,
            },
            "independent_case_throughput_non_release",
            False,
        ),
    ),
)
def test_parallel_scaling_claim_scope_classifies_artifact_families(
    summary: dict[str, object],
    expected_scope: str,
    supported: bool,
) -> None:
    audit = transport_parallel_runtime.audit_parallel_scaling_claim_scope(summary)

    assert audit.claim_scope == expected_scope
    assert audit.release_scaling_supported is supported
    assert audit.parallel_count >= 2


def test_multi_gpu_case_throughput_audit_accepts_non_release_evidence() -> None:
    audit = transport_parallel_runtime.audit_multi_gpu_case_throughput_summary(
        {
            "benchmark_kind": "multi_gpu_case_throughput",
            "backend": "gpu",
            "timing_semantics": "warm",
            "required_gpu_count": 2,
            "sequential_one_gpu": {"wall_s": 12.0},
            "parallel_two_gpu": {"wall_s": 6.0},
        },
        min_throughput_speedup=1.5,
    )

    assert audit.ci_gate_pass
    assert audit.throughput_speedup == pytest.approx(2.0)
    assert not audit.release_scaling_claim
    assert any("not single-case strong scaling" in note for note in audit.notes)


def test_sharded_solve_scaling_audit_accepts_experimental_gate_schema() -> None:
    audit = transport_parallel_runtime.audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "device_ids": ["0", "1"],
            "timing_semantics": "warm",
            "release_scaling_claim": False,
            "experimental_single_case_scaling": True,
            "results": [
                {"devices": 1, "mean_s": 10.0},
                {"devices": 2, "mean_s": 7.5},
            ],
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "warm",
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "timed_repeats": 2,
                "min_timed_repeats": 2,
            },
            "deterministic_output_gate": {
                "passes": True,
                "residual_tolerance": 1.0e-10,
                "max_relative_residual_norm": 1.0e-12,
                "output_digest": "abc",
                "evidence_source": "unit_fixture",
            },
        }
    )

    assert audit.ci_gate_pass
    assert audit.experimental_single_case_scaling
    assert audit.operator_reuse_gate
    assert audit.deterministic_output_gate
    assert not audit.release_promotion_supported


def test_single_case_sharding_plans_and_gates_are_fail_closed() -> None:
    plan = transport_parallel_runtime.plan_single_case_sharded_solve(
        requested_devices=4,
        backend="gpu",
        available_device_ids="0,1",
        rhs_mode=1,
        shard_axis="theta",
        shard_axis_size=5,
        experimental_single_case_scaling=True,
    )

    assert plan.active_devices == 2
    assert plan.capped
    assert plan.eligible_for_single_case_sharding
    assert plan.balance_diagnostics.max_to_mean_ratio == pytest.approx(1.2)
    assert plan.to_dict()["benchmark_kind"] == "single_case_sharded_solve"

    amortization = transport_parallel_runtime.estimate_sharded_solve_amortization(
        active_devices=2,
        serial_work_units=100.0,
        setup_work_units=2.0,
        krylov_iterations=3,
        collectives_per_iteration=2,
        collective_latency_units=0.1,
        halo_bytes_per_iteration=10.0,
        bandwidth_bytes_per_unit=100.0,
    )
    assert amortization.release_scaling_supported
    assert amortization.predicted_speedup > 1.0

    compile_gate = transport_parallel_runtime.plan_compiled_sharded_operator_reuse(
        benchmark_kind="single_case_sharded_solve",
        timing_semantics="warm",
        inner_warmup_runs=1,
        timed_repeats=2,
        min_timed_repeats=2,
        work_units_per_sample=4,
    )
    assert compile_gate.passes
    assert compile_gate.strategy == "inner_warmup"

    deterministic_gate = transport_parallel_runtime.plan_sharded_solve_deterministic_output_gate(
        comparison_devices=2,
        max_relative_residual_norm=1.0e-12,
        residual_tolerance=1.0e-10,
        baseline_output_digest="same",
        comparison_output_digest="same",
    )
    assert deterministic_gate.passes
    assert deterministic_gate.output_digest_match

    reuse_plan = transport_parallel_runtime.plan_single_case_operator_coarse_reuse(
        active_devices=2,
        backend="gpu",
        operator_reuse_gate_pass=True,
        deterministic_output_gate_pass=True,
        measured_hot_speedup=1.3,
        memory_growth_fraction=0.0,
        coarse_levels=1,
        max_coarse_rank=4,
    )
    assert reuse_plan.plan_valid
    assert reuse_plan.promotion_ready
    assert reuse_plan.operator_build_scope == "once_per_child_process"


def test_transport_parallel_environment_helpers_are_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_MP_START_METHOD", "forkserver")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND", "gpu_process")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,2,1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PIN_THREADS", "1")
    monkeypatch.setenv("SFINCS_JAX_CORES", "8")

    assert transport_parallel_runtime.transport_parallel_start_method() == "forkserver"
    assert transport_parallel_runtime.transport_parallel_backend() == "gpu"
    assert transport_parallel_runtime.transport_parallel_visible_gpu_ids(4) == ["1", "2"]
    assert transport_parallel_runtime.transport_parallel_pool_key(2)[:3] == (
        2,
        "gpu",
        "forkserver",
    )

    gpu_env = transport_parallel_runtime.transport_parallel_gpu_worker_env(gpu_id="2")
    assert gpu_env["CUDA_VISIBLE_DEVICES"] == "2"
    assert gpu_env["SFINCS_JAX_TRANSPORT_PARALLEL_CHILD"] == "1"

    messages: list[str] = []
    kwargs = transport_parallel_runtime.transport_parallel_pool_executor_kwargs(
        parallel_workers=2,
        get_context=lambda method: (_ for _ in ()).throw(ValueError(method))
        if method == "forkserver"
        else SimpleNamespace(method=method),
        emit=lambda _level, message: messages.append(message),
    )
    assert kwargs["max_workers"] == 2
    assert kwargs["mp_context"].method == "spawn"
    assert any("using 'spawn'" in message for message in messages)


def test_transport_parallel_worker_env_sets_and_restores_thread_caps(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_CORES", "8")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PIN_THREADS", "1")
    monkeypatch.setenv("XLA_FLAGS", "--foo=1 --xla_force_host_platform_device_count=8")
    monkeypatch.setenv("OMP_NUM_THREADS", "old")

    with transport_parallel_runtime.transport_parallel_worker_env(2):
        assert os.environ["SFINCS_JAX_SHARD"] == "0"
        assert os.environ["SFINCS_JAX_CPU_DEVICES"] == "1"
        assert os.environ["OMP_NUM_THREADS"] == "4"
        assert "--foo=1" in os.environ["XLA_FLAGS"]
        assert "--xla_force_host_platform_device_count=1" in os.environ["XLA_FLAGS"]

    assert os.environ["OMP_NUM_THREADS"] == "old"
    assert os.environ["XLA_FLAGS"] == "--foo=1 --xla_force_host_platform_device_count=8"


def test_transport_parallel_payload_policy_helpers_are_pure(tmp_path: Path) -> None:
    input_path = tmp_path / "input.namelist"
    payloads = transport_parallel_runtime.build_transport_parallel_payloads(
        chunks=[[1, 3], [2]],
        input_namelist=input_path,
        tol=1.0e-10,
        atol=1.0e-14,
        restart=20,
        maxiter=40,
        solve_method="auto",
        identity_shift=1.0e-8,
        collect_transport_output_fields=True,
        phi1_hat_base=np.asarray([0.0, 1.0]),
        differentiable=False,
    )

    assert transport_parallel_runtime.should_run_transport_parallel(
        parallel_child=False,
        parallel_workers=2,
        which_rhs_values=[1, 2, 3],
        input_namelist=input_path,
    )
    assert not transport_parallel_runtime.should_run_transport_parallel(
        parallel_child=True,
        parallel_workers=2,
        which_rhs_values=[1, 2],
        input_namelist=input_path,
    )
    assert payloads[0]["which_rhs_values"] == [1, 3]
    np.testing.assert_allclose(payloads[0]["phi1_hat_base"], [0.0, 1.0])
    assert payloads[1]["maxiter"] == 40


def test_run_transport_parallel_payloads_uses_gpu_dispatch() -> None:
    calls: list[dict[str, object]] = []

    result = run_transport_parallel_payloads(
        payloads=[{"which_rhs_values": [1]}],
        parallel_workers=2,
        parallel_backend="gpu",
        run_gpu_subprocesses=lambda **kwargs: calls.append(kwargs) or [{"which_rhs_values": [1]}],
        persistent_pool_enabled=False,
        get_pool=lambda **_kwargs: pytest.fail("CPU pool was not expected"),
        shutdown_pool=lambda: pytest.fail("shutdown was not expected"),
        worker=lambda _payload: pytest.fail("CPU worker was not expected"),
        worker_env=lambda _workers: pytest.fail("worker env was not expected"),
        executor_class=object,
        executor_kwargs=lambda **_kwargs: {},
        emit=None,
    )

    assert result == [{"which_rhs_values": [1]}]
    assert calls[0]["parallel_workers"] == 2


def test_sharded_planning_gates_record_failures_without_running_solves() -> None:
    plan = transport_parallel_runtime.plan_single_case_sharded_solve(
        requested_devices=2,
        backend="cpu",
        available_device_count=0,
        rhs_mode=2,
        shard_axis="bad",
        task_count=2,
        release_scaling_claim=True,
        experimental_single_case_scaling=False,
    )
    assert not plan.eligible_for_single_case_sharding
    assert any("RHSMode=1" in failure for failure in plan.failures)
    assert plan.balance_diagnostics.idle_device_count == 0

    amortization = transport_parallel_runtime.estimate_sharded_solve_amortization(
        active_devices=1,
        serial_work_units=4.0,
        setup_work_units=10.0,
        krylov_iterations=10,
        collectives_per_iteration=2,
        collective_latency_units=1.0,
        min_speedup=2.0,
        min_efficiency=0.9,
        max_communication_fraction=0.1,
    )
    assert not amortization.release_scaling_supported
    assert any("at least 2 active devices" in failure for failure in amortization.failures)
    assert any("setup cost exceeds" in note for note in amortization.notes)

    compile_gate = transport_parallel_runtime.plan_compiled_sharded_operator_reuse(
        benchmark_kind="bad_kind",
        timing_semantics="cold_start",
        global_warmup_runs=1,
        timed_repeats=1,
        min_timed_repeats=2,
        persistent_compile_cache=False,
        compile_in_timed_region=True,
    )
    assert not compile_gate.passes
    assert any("unsupported" in failure for failure in compile_gate.failures)
    assert any("compile_cache_dir" in failure for failure in compile_gate.failures)

    deterministic_gate = transport_parallel_runtime.plan_sharded_solve_deterministic_output_gate(
        comparison_devices=1,
        max_relative_residual_norm=1.0e-3,
        residual_tolerance=1.0e-6,
        baseline_output_digest="a",
        comparison_output_digest="b",
    )
    assert not deterministic_gate.passes
    assert deterministic_gate.output_digest_match is False

    reuse_plan = transport_parallel_runtime.plan_single_case_operator_coarse_reuse(
        active_devices=1,
        rhs_mode=2,
        shard_axis="bad",
        operator_reuse_enabled=False,
        measured_hot_speedup=1.0,
        memory_growth_fraction=0.5,
    )
    assert not reuse_plan.plan_valid
    assert not reuse_plan.promotion_ready
    assert "compiled operator-reuse gate has not passed" in reuse_plan.promotion_blockers


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


def test_run_transport_parallel_payloads_falls_back_when_one_shot_pool_unavailable() -> None:
    messages: list[str] = []
    payloads = [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]

    class _Env:
        def __enter__(self):
            return None

        def __exit__(self, *_exc):
            return False

    def _executor_class(**_kwargs):
        raise OSError("process pool unavailable")

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=2,
        parallel_backend="cpu",
        run_gpu_subprocesses=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected GPU path")),
        persistent_pool_enabled=False,
        get_pool=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("unexpected persistent pool")),
        shutdown_pool=lambda: None,
        worker=lambda payload: {"which_rhs_values": payload["which_rhs_values"]},
        worker_env=lambda _workers: _Env(),
        executor_class=_executor_class,
        executor_kwargs=lambda **_kwargs: {},
        emit=lambda level, message: messages.append(f"{level}:{message}"),
    )

    assert results == [{"which_rhs_values": [1]}, {"which_rhs_values": [2]}]
    assert any("process parallelism unavailable" in message for message in messages)


def test_solve_transport_parallel_payload_packs_stubbed_result(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.namelist"
    input_path.write_text("&general\n/\n")
    emitted: list[str] = []
    calls: list[dict[str, object]] = []

    def _read_input(path: Path):
        assert path == input_path
        return {"input": str(path)}

    def _solve_transport(**kwargs):
        calls.append(kwargs)
        assert kwargs["nml"] == {"input": str(input_path)}
        assert kwargs["which_rhs_values"] == [1, 2]
        assert kwargs["collect_transport_output_fields"] is False
        assert kwargs["parallel_workers"] == 1
        assert kwargs["force_store_state"] is True
        kwargs["emit"](0, "worker progress")
        return SimpleNamespace(
            state_vectors_by_rhs={1: np.asarray([1.0, 0.0]), 2: np.asarray([0.0, 1.0])},
            residual_norms_by_rhs={1: 1.0e-12, 2: 2.0e-12},
            rhs_norms_by_rhs={1: 1.0, 2: 2.0},
            elapsed_time_s=np.asarray([0.1, 0.2]),
        )

    result = solve_transport_parallel_payload(
        {
            "input_path": str(input_path),
            "which_rhs_values": [1, 2],
            "tol": 1.0e-11,
            "atol": 1.0e-14,
            "restart": 20,
            "maxiter": 40,
            "solve_method": "auto",
            "identity_shift": 0.0,
            "phi1_hat_base": [0.0, 0.0],
            "differentiable": False,
        },
        read_input=_read_input,
        solve_transport=_solve_transport,
        emit=lambda _level, message: emitted.append(message),
    )

    assert result["which_rhs_values"] == [1, 2]
    np.testing.assert_allclose(result["state_vectors_by_rhs"][1], np.asarray([1.0, 0.0]))
    assert result["residual_norms_by_rhs"] == {1: 1.0e-12, 2: 2.0e-12}
    assert result["rhs_norms_by_rhs"] == {1: 1.0, 2: 2.0}
    assert emitted == ["worker progress"]
    assert calls[0]["tol"] == 1.0e-11
    assert calls[0]["differentiable"] is False


def test_pack_transport_parallel_result_tolerates_missing_rhs_norms() -> None:
    packed = pack_transport_parallel_result(
        which_rhs_values=[3],
        result=SimpleNamespace(
            state_vectors_by_rhs={3: np.asarray([3.0])},
            residual_norms_by_rhs={3: np.asarray(3.0e-12)},
            elapsed_time_s=np.asarray([0.3]),
        ),
    )

    assert packed["which_rhs_values"] == [3]
    assert packed["rhs_norms_by_rhs"] == {}
    np.testing.assert_allclose(packed["state_vectors_by_rhs"][3], np.asarray([3.0]))


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
