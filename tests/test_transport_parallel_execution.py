from __future__ import annotations

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import numpy as np
import pytest

from sfincs_jax.problems.transport_matrix.parallel.runtime import (
    audit_parallel_scaling_claim_scope,
    audit_sharded_solve_scaling_summary,
    audit_transport_parallel_scaling_summary,
)
from sfincs_jax.problems.transport_matrix.parallel.runtime import (
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


def test_audit_transport_parallel_scaling_summary_accepts_release_grade_gpu_worker_claim() -> None:
    audit = audit_transport_parallel_scaling_summary(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "gpu",
            "rhs_count": 3,
            "gpu_device_count": 2,
            "timing_semantics": "cache_warm",
            "deterministic_output_check": True,
            "workers": [1, 2],
            "ideal_speedup_finite_rhs": [1.0, 1.5],
            "payloads": [
                {"which_rhs_values": [1, 3]},
                {"which_rhs_values": [2]},
            ],
            "results": [
                {"workers": 1, "mean_s": 351.05, "speedup": 1.0},
                {"workers": 2, "mean_s": 237.75, "speedup": 1.4766},
            ],
        }
    )

    assert audit.release_scaling_claim
    assert audit.backend == "gpu"
    assert audit.task_count == 3
    assert audit.device_count == 2
    assert audit.claim_workers == 2
    assert audit.claim_speedup == pytest.approx(1.4766)
    assert audit.claim_efficiency == pytest.approx(0.7383)
    assert audit.claim_finite_task_ideal_speedup == pytest.approx(1.5)
    assert audit.deterministic_payload_coverage
    assert audit.deterministic_output_check
    assert audit.timing_semantics == "cache_warm"
    assert audit.failures == ()


def test_parallel_scaling_claim_scope_distinguishes_throughput_from_single_case() -> None:
    transport_scope = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "cpu",
            "rhs_count": 3,
            "workers": [1, 2],
        }
    )
    assert transport_scope.claim_scope == "independent_transport_worker_throughput"
    assert transport_scope.claim_scope_release_eligible
    assert not transport_scope.release_scaling_supported
    assert not transport_scope.unsupported_single_case_strong_scaling
    assert transport_scope.release_gate_required == "audit_transport_parallel_scaling_summary"

    sharded_scope = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "scaling_status": "experimental_single_case_sharding",
            "release_scaling_claim": True,
            "devices": [1, 2],
        }
    )
    assert sharded_scope.claim_scope == "single_case_sharded_solve_experimental"
    assert not sharded_scope.release_scaling_supported
    assert sharded_scope.unsupported_single_case_strong_scaling
    assert any("release_scaling_claim=true" in failure for failure in sharded_scope.failures)

    throughput_scope = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "multi_gpu_case_throughput",
            "backend": "gpu",
            "required_gpu_count": 2,
            "release_scaling_claim": False,
        }
    )
    assert throughput_scope.claim_scope == "independent_case_throughput_non_release"
    assert not throughput_scope.release_scaling_supported
    assert not throughput_scope.unsupported_single_case_strong_scaling


def test_parallel_scaling_claim_scope_fails_closed_for_ambiguous_or_overclaimed_artifacts() -> None:
    ambiguous = audit_parallel_scaling_claim_scope(
        {
            "backend": "cpu",
            "rhs_count": 3,
            "workers": [1, 2],
        }
    )
    assert not ambiguous.release_scaling_supported
    assert any("explicit benchmark_kind" in failure for failure in ambiguous.failures)

    plan_overclaim = audit_parallel_scaling_claim_scope(
        {
            "artifact_kind": "benchmark_plan",
            "benchmark_kind": "transport_worker_scaling",
            "backend": "cpu",
            "launches_solves": False,
            "release_scaling_claim": True,
            "rhs_count": 3,
            "workers": [1, 2],
        }
    )
    assert plan_overclaim.plan_only_scope_evidence
    assert not plan_overclaim.measured_results_present
    assert not plan_overclaim.claim_scope_release_eligible
    assert not plan_overclaim.release_scaling_supported
    assert any("plan-only" in failure for failure in plan_overclaim.failures)
    assert any("measured timing results" in failure for failure in plan_overclaim.failures)

    conflicting_scope = audit_parallel_scaling_claim_scope(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "claim_scope": "independent_transport_worker_throughput",
            "scaling_status": "experimental_single_case_sharding",
            "release_scaling_claim": False,
            "devices": [1, 2],
        }
    )
    assert conflicting_scope.claim_scope == "single_case_sharded_solve_experimental"
    assert any("conflicts" in failure for failure in conflicting_scope.failures)


def test_audit_transport_parallel_scaling_summary_reports_weak_claim_gates() -> None:
    audit = audit_transport_parallel_scaling_summary(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "gpu",
            "rhs_count": 3,
            "gpu_device_count": 1,
            "timing_semantics": "cache_warm",
            "payloads": [
                {"which_rhs_values": [1, 1]},
                {"which_rhs_values": [2]},
            ],
            "results": [
                {"workers": 1, "mean_s": 100.0},
                {"workers": 2, "mean_s": 120.0},
            ],
        }
    )

    assert not audit.release_scaling_claim
    assert audit.claim_speedup == pytest.approx(100.0 / 120.0)
    assert not audit.deterministic_payload_coverage
    assert any("GPU devices" in failure for failure in audit.failures)
    assert any("speedup" in failure for failure in audit.failures)
    assert any("efficiency" in failure for failure in audit.failures)
    assert any("deterministic payload coverage" in failure for failure in audit.failures)
    assert any("payload RHS coverage" in note for note in audit.notes)


def test_audit_transport_parallel_scaling_summary_rejects_task_overclaim_and_cold_timing() -> None:
    audit = audit_transport_parallel_scaling_summary(
        {
            "benchmark_kind": "transport_worker_scaling",
            "backend": "cpu",
            "rhs_count": 3,
            "timing_semantics": "cold_start",
            "deterministic_output_check": True,
            "payloads": [
                {"which_rhs_values": [1]},
                {"which_rhs_values": [2]},
                {"which_rhs_values": [3]},
            ],
            "results": [
                {"workers": 1, "mean_s": 300.0, "speedup": 1.0},
                {"workers": 4, "mean_s": 100.0, "speedup": 3.0},
            ],
        }
    )

    assert not audit.release_scaling_claim
    assert any("4 workers cannot be claimed for only 3 independent transport tasks" in failure for failure in audit.failures)
    assert any("cold_start" in failure for failure in audit.failures)


def test_audit_transport_parallel_scaling_summary_rejects_sharded_solve_payload() -> None:
    with pytest.raises(ValueError, match="single-case sharded-solve summaries must use"):
        audit_transport_parallel_scaling_summary(
            {
                "benchmark_kind": "single_case_sharded_solve",
                "task_count": 1,
                "devices": [1, 2],
                "results": [
                    {"devices": 1, "mean_s": 10.0},
                    {"devices": 2, "mean_s": 7.0},
                ],
            }
        )


def test_audit_sharded_solve_scaling_summary_accepts_honest_experimental_payload() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "scaling_status": "experimental_single_case_sharding",
            "experimental_single_case_scaling": True,
            "release_scaling_claim": False,
            "backend": "gpu",
            "gpu_device_count": 2,
            "timing_semantics": "hot_solve",
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "hot_solve",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "persistent_compile_cache": True,
                "compile_cache_dir": "examples/performance/output/cache",
            },
            "deterministic_output_check": False,
            "devices": [1, 2],
            "results": [
                {"devices": 1, "mean_s": 10.0, "speedup": 1.0},
                {"devices": 2, "mean_s": 8.0, "speedup": 1.25},
            ],
        }
    )

    assert audit.ci_gate_pass
    assert not audit.release_scaling_claim
    assert audit.experimental_single_case_scaling
    assert audit.operator_reuse_gate
    assert not audit.deterministic_output_gate
    assert audit.claim_speedup == pytest.approx(1.25)
    assert not audit.release_promotion_supported
    assert any("single-case sharded solve" in blocker for blocker in audit.release_promotion_blockers)
    assert any("not a release scaling claim" in note for note in audit.notes)


def test_audit_sharded_solve_scaling_summary_requires_operator_reuse_gate() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "scaling_status": "experimental_single_case_sharding",
            "experimental_single_case_scaling": True,
            "release_scaling_claim": False,
            "backend": "gpu",
            "gpu_device_count": 2,
            "timing_semantics": "hot_solve",
            "devices": [1, 2],
            "results": [
                {"devices": 1, "mean_s": 10.0, "speedup": 1.0},
                {"devices": 2, "mean_s": 8.0, "speedup": 1.25},
            ],
        }
    )

    assert not audit.ci_gate_pass
    assert not audit.operator_reuse_gate
    assert any("operator-reuse gate metadata" in failure for failure in audit.failures)


def test_audit_sharded_solve_scaling_summary_rejects_timed_out_samples() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "scaling_status": "experimental_single_case_sharding",
            "experimental_single_case_scaling": True,
            "release_scaling_claim": False,
            "backend": "gpu",
            "gpu_device_count": 2,
            "timing_semantics": "hot_solve",
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "hot_solve",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "persistent_compile_cache": True,
                "compile_cache_dir": "examples/performance/output/cache",
            },
            "deterministic_output_check": False,
            "devices": [1, 2],
            "results": [
                {"devices": 1, "mean_s": 4.0, "speedup": 1.0, "samples": [4.0]},
                {
                    "devices": 2,
                    "mean_s": 300.0,
                    "speedup": 4.0 / 300.0,
                    "samples": [],
                    "timed_out": True,
                    "sample_failures": ["repeat 1/1 failed: RuntimeError: Timed out after 300.0s"],
                },
            ],
        }
    )

    assert not audit.ci_gate_pass
    assert any("devices=2 recorded failed/timed-out" in failure for failure in audit.failures)
    assert any("devices=2 recorded failed/timed-out" in blocker for blocker in audit.release_promotion_blockers)


def test_audit_sharded_solve_scaling_summary_fails_requested_output_probe() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "scaling_status": "experimental_single_case_sharding",
            "experimental_single_case_scaling": True,
            "release_scaling_claim": False,
            "backend": "gpu",
            "gpu_device_count": 2,
            "timing_semantics": "hot_solve",
            "operator_reuse_gate": {
                "passes": True,
                "timing_semantics": "hot_solve",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": False,
                "warm_run_amortization_pass": True,
                "persistent_compile_cache": True,
                "compile_cache_dir": "examples/performance/output/cache",
            },
            "deterministic_output_probe_requested": True,
            "deterministic_output_gate": {
                "passes": False,
                "status": "probe_failed",
                "failures": ["deterministic output probe failed"],
            },
            "devices": [1, 2],
            "results": [
                {"devices": 1, "mean_s": 10.0, "speedup": 1.0},
                {"devices": 2, "mean_s": 8.0, "speedup": 1.25},
            ],
        }
    )

    assert not audit.ci_gate_pass
    assert any("requested deterministic output probe" in failure for failure in audit.failures)
    assert any("requested deterministic output probe" in blocker for blocker in audit.release_promotion_blockers)


def test_audit_sharded_solve_scaling_summary_rejects_release_overclaim() -> None:
    audit = audit_sharded_solve_scaling_summary(
        {
            "benchmark_kind": "single_case_sharded_solve",
            "backend": "gpu",
            "gpu_device_count": 1,
            "timing_semantics": "cache_warm",
            "release_scaling_claim": True,
            "devices": [1, 2],
            "results": [
                {"devices": 1, "mean_s": 10.0},
                {"devices": 2, "mean_s": 5.0},
            ],
        }
    )

    assert not audit.ci_gate_pass
    assert any("must not set release_scaling_claim=true" in failure for failure in audit.failures)
    assert any("must be marked experimental" in failure for failure in audit.failures)
    assert any("only 1 GPU devices" in failure for failure in audit.failures)


def test_audit_transport_parallel_scaling_summary_rejects_malformed_summaries() -> None:
    with pytest.raises(ValueError, match="1-worker baseline"):
        audit_transport_parallel_scaling_summary(
            {
                "rhs_count": 3,
                "results": [{"workers": 2, "mean_s": 10.0}],
            }
        )

    with pytest.raises(ValueError, match=r"results\[1\]\.mean_s"):
        audit_transport_parallel_scaling_summary(
            {
                "rhs_count": 3,
                "results": [
                    {"workers": 1, "mean_s": 10.0},
                    {"workers": 2, "mean_s": 0.0},
                ],
            }
        )


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


def test_run_transport_parallel_payloads_rejects_invalid_worker_count() -> None:
    with pytest.raises(ValueError, match="transport parallel worker count must be >= 1; got 0"):
        run_transport_parallel_payloads(
            payloads=[{"which_rhs_values": [1]}],
            parallel_workers=0,
            parallel_backend="cpu",
            run_gpu_subprocesses=lambda **_kwargs: [],
            persistent_pool_enabled=False,
            get_pool=lambda **_kwargs: None,
            shutdown_pool=lambda: None,
            worker=lambda payload: payload,
            worker_env=lambda _n: None,
            executor_class=None,
            executor_kwargs=lambda **_kwargs: {},
            emit=None,
        )


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


def test_run_transport_parallel_payloads_preserves_payload_order_when_cpu_futures_finish_out_of_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = [
        {"which_rhs_values": [1]},
        {"which_rhs_values": [2]},
        {"which_rhs_values": [3]},
    ]

    monkeypatch.setattr(
        "sfincs_jax.problems.transport_matrix.parallel.runtime.concurrent.futures.as_completed",
        lambda futures: reversed(list(futures)),
    )

    results = run_transport_parallel_payloads(
        payloads=payloads,
        parallel_workers=3,
        parallel_backend="cpu",
        run_gpu_subprocesses=lambda **_kwargs: [],
        persistent_pool_enabled=True,
        get_pool=lambda **_kwargs: _DummyPool(payloads),
        shutdown_pool=lambda: None,
        worker=lambda payload: {"which_rhs_values": payload["which_rhs_values"]},
        worker_env=lambda _n: _DummyEnv(),
        executor_class=None,
        executor_kwargs=lambda **_kwargs: {},
        emit=None,
    )

    assert [res["which_rhs_values"] for res in results] == [[1], [2], [3]]


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
