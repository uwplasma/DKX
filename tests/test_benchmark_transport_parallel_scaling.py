from __future__ import annotations

import os
from pathlib import Path

from examples.performance.benchmark_transport_parallel_scaling import (
    _build_transport_benchmark_plan,
    _configure_backend_env,
    _payloads_for_workers,
    _timing_semantics,
    _write_scaling_figure,
)
from sfincs_jax.problems.transport_parallel_runtime import audit_transport_parallel_scaling_summary
from sfincs_jax.problems.transport_parallel_runtime import plan_transport_parallel_gpu_subprocesses


def test_configure_backend_env_cpu() -> None:
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    _configure_backend_env(workers=2, backend="cpu")
    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND"] == "cpu"
    assert os.environ["SFINCS_JAX_CPU_DEVICES"] == "1"
    assert "CUDA_VISIBLE_DEVICES" not in os.environ


def test_configure_backend_env_gpu() -> None:
    _configure_backend_env(workers=2, backend="gpu")
    assert os.environ["SFINCS_JAX_TRANSPORT_PARALLEL_BACKEND"] == "gpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert os.environ["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"


def test_payloads_for_workers_caps_at_rhs_count() -> None:
    assert _payloads_for_workers(rhs_count=3, workers=4) == [
        {"which_rhs_values": [1]},
        {"which_rhs_values": [2]},
        {"which_rhs_values": [3]},
    ]


def test_timing_semantics_labels_warm_and_cold_modes() -> None:
    assert _timing_semantics(global_warmup=1, per_worker_warmup=0) == "cache_warm"
    assert _timing_semantics(global_warmup=0, per_worker_warmup=1) == "hot_solve"
    assert _timing_semantics(global_warmup=0, per_worker_warmup=0) == "cold_start"


def test_write_scaling_figure_from_payload(tmp_path) -> None:
    payload = {
        "case": "unit_case",
        "backend": "gpu",
        "ideal_speedup_finite_rhs": [1.0, 1.5],
        "results": [
            {"workers": 1, "mean_s": 10.0, "speedup": 1.0},
            {"workers": 2, "mean_s": 7.0, "speedup": 10.0 / 7.0},
        ],
    }

    path = _write_scaling_figure(payload, tmp_path)

    assert path == tmp_path / "transport_parallel_scaling.png"
    assert path.exists()
    assert path.stat().st_size > 0


def test_build_transport_benchmark_plan_records_capped_workers_and_gate_semantics(tmp_path) -> None:
    input_path = Path("examples/performance/transport_parallel_2min.input.namelist")

    plan = _build_transport_benchmark_plan(
        input_path=input_path,
        rhs_mode=2,
        rhs_count=3,
        requested_workers=[4],
        repeats=1,
        warmup=0,
        global_warmup=1,
        precond="xmg",
        backend="gpu",
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        figure_name="transport_parallel_scaling.png",
        audit=True,
    )

    assert plan["artifact_kind"] == "benchmark_plan"
    assert plan["launches_solves"] is False
    assert plan["workers"] == [1, 3]
    assert plan["skipped_workers"] == [4]
    assert plan["timing_semantics"] == "cache_warm"
    assert plan["compile_amortization_gate"]["passes"] is True
    assert plan["compile_amortization_gate"]["persistent_compile_cache"] is True
    assert plan["compile_amortization_gate"]["compile_in_timed_region"] is False
    assert plan["estimated_transport_solve_calls"] == 3
    assert plan["payloads"] == [
        {"which_rhs_values": [1]},
        {"which_rhs_values": [2]},
        {"which_rhs_values": [3]},
    ]
    assert plan["release_gate_semantics"]["cold_start_rejected"] is True
    assert plan["release_gate_semantics"]["requires_compile_amortization_gate"] is True
    assert plan["memory_gate_semantics"]["gpu_preallocation_disabled"] is True
    assert plan["parallel_claim_scope"]["claim_scope"] == "independent_transport_worker_throughput"
    assert plan["parallel_claim_scope"]["claim_scope_release_eligible"] is True
    assert plan["parallel_claim_scope"]["release_scaling_supported"] is False
    assert plan["parallel_claim_scope"]["unsupported_single_case_strong_scaling"] is False
    assert plan["parallel_claim_scope"]["backend"] == "gpu"
    assert plan["parallel_claim_scope"]["artifact_kind"] == "benchmark_plan"
    assert plan["parallel_claim_scope"]["launches_solves"] is False
    assert plan["parallel_claim_scope"]["plan_only_scope_evidence"] is True
    assert plan["parallel_claim_scope"]["measured_results_present"] is False
    assert plan["parallel_claim_scope"]["release_gate_required"] == "audit_transport_parallel_scaling_summary"
    assert "--audit" in plan["benchmark_command"]


def test_transport_scaling_audit_rejects_cold_or_weak_speedup_payloads() -> None:
    payload = {
        "benchmark_kind": "transport_worker_scaling",
        "backend": "cpu",
        "rhs_count": 3,
        "which_rhs_values": [1, 2, 3],
        "workers": [1, 2],
        "timing_semantics": "cache_warm",
        "deterministic_payload_coverage": True,
        "deterministic_output_check": True,
        "payloads": [
            {"which_rhs_values": [1, 3]},
            {"which_rhs_values": [2]},
        ],
        "results": [
            {"workers": 1, "mean_s": 10.0, "speedup": 1.0},
            {"workers": 2, "mean_s": 7.0, "speedup": 10.0 / 7.0},
        ],
    }

    assert audit_transport_parallel_scaling_summary(payload).release_scaling_claim is True

    cold_payload = dict(payload, timing_semantics="cold_start")
    cold_audit = audit_transport_parallel_scaling_summary(cold_payload)
    assert cold_audit.release_scaling_claim is False
    assert any("cold" in failure for failure in cold_audit.failures)

    weak_payload = {
        **payload,
        "results": [
            {"workers": 1, "mean_s": 10.0, "speedup": 1.0},
            {"workers": 2, "mean_s": 9.5, "speedup": 10.0 / 9.5},
        ],
    }
    weak_audit = audit_transport_parallel_scaling_summary(weak_payload)
    assert weak_audit.release_scaling_claim is False
    assert any("speedup" in failure for failure in weak_audit.failures)


def test_transport_scaling_audit_validates_compile_amortization_gate() -> None:
    payload = {
        "benchmark_kind": "transport_worker_scaling",
        "backend": "cpu",
        "rhs_count": 3,
        "which_rhs_values": [1, 2, 3],
        "workers": [1, 2],
        "timing_semantics": "cache_warm",
        "deterministic_payload_coverage": True,
        "deterministic_output_check": True,
        "payloads": [
            {"which_rhs_values": [1, 3]},
            {"which_rhs_values": [2]},
        ],
        "results": [
            {"workers": 1, "mean_s": 10.0, "speedup": 1.0},
            {"workers": 2, "mean_s": 7.0, "speedup": 10.0 / 7.0},
        ],
    }

    explicit_release = audit_transport_parallel_scaling_summary({**payload, "release_scaling_claim": True})
    assert explicit_release.release_scaling_claim is False
    assert any("compile-amortization gate metadata" in failure for failure in explicit_release.failures)

    failing_gate = audit_transport_parallel_scaling_summary(
        {
            **payload,
            "compile_amortization_gate": {
                "passes": False,
                "timing_semantics": "cache_warm",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": True,
            },
        }
    )
    assert failing_gate.release_scaling_claim is False
    assert failing_gate.compile_amortization_gate is False
    assert any("did not pass" in failure for failure in failing_gate.failures)
    assert any("inside the timed region" in failure for failure in failing_gate.failures)

    passing_gate = audit_transport_parallel_scaling_summary(
        {
            **payload,
            "release_scaling_claim": True,
            "compile_amortization_gate": {
                "passes": True,
                "timing_semantics": "cache_warm",
                "timed_repeats": 1,
                "min_timed_repeats": 1,
                "compile_in_timed_region": False,
            },
        }
    )
    assert passing_gate.release_scaling_claim is True
    assert passing_gate.compile_amortization_gate is True


def test_gpu_transport_runtime_plan_deduplicates_and_coalesces_workers() -> None:
    plan = plan_transport_parallel_gpu_subprocesses(
        payloads=[
            {"which_rhs_values": [1]},
            {"which_rhs_values": [2]},
            {"which_rhs_values": [3]},
        ],
        parallel_workers=3,
        visible_gpu_ids=["0", "0", "1"],
    )

    assert plan["requested_workers"] == 3
    assert plan["active_workers"] == 2
    assert plan["capped"] is True
    assert plan["cap_reasons"] == ["unique visible GPU ids=2"]
    assert [
        assignment["which_rhs_values"] for assignment in plan["worker_assignments"]
    ] == [[1, 3], [2]]
    assert [assignment["gpu_id"] for assignment in plan["worker_assignments"]] == ["0", "1"]
