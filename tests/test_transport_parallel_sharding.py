from __future__ import annotations

import pytest

from sfincs_jax.problems.transport_matrix.parallel.runtime import (
    estimate_sharded_solve_amortization,
    plan_compiled_sharded_operator_reuse,
    plan_sharded_solve_deterministic_output_gate,
    plan_single_case_operator_coarse_reuse,
    plan_single_case_sharded_solve,
)


def test_single_case_sharding_plan_cpu_one_device_fails_closed() -> None:
    plan = plan_single_case_sharded_solve(
        requested_devices=1,
        backend="cpu",
        available_device_count=1,
        rhs_mode=1,
        shard_axis="theta",
        experimental_single_case_scaling=True,
    )

    assert plan.backend == "cpu"
    assert plan.requested_devices == 1
    assert plan.active_devices == 1
    assert plan.capped is False
    assert plan.release_scaling_claim is False
    assert plan.release_scaling_supported is False
    assert plan.eligible_for_single_case_sharding is False
    assert any("only 1 active devices" in failure for failure in plan.failures)
    assert plan.balance_diagnostics.total_work_units == 1
    assert plan.balance_diagnostics.idle_device_count == 0


def test_single_case_sharding_plan_caps_and_balances_simulated_devices() -> None:
    plan = plan_single_case_sharded_solve(
        requested_devices=4,
        backend="gpu",
        available_device_ids=["0", "1", "1", "2"],
        rhs_mode=1,
        shard_axis="zeta",
        shard_axis_size=10,
        experimental_single_case_scaling=True,
    )

    assert plan.requested_devices == 4
    assert plan.available_device_count == 3
    assert plan.available_device_ids == ("0", "1", "2")
    assert plan.active_devices == 3
    assert plan.capped is True
    assert plan.cap_reasons == ("available devices=3",)
    assert plan.eligible_for_single_case_sharding is True
    assert [assignment.device_id for assignment in plan.device_assignments] == ["0", "1", "2"]
    assert [
        (assignment.shard_start, assignment.shard_stop, assignment.work_units)
        for assignment in plan.device_assignments
    ] == [(0, 4, 4), (4, 7, 3), (7, 10, 3)]
    assert plan.balance_diagnostics.total_work_units == 10
    assert plan.balance_diagnostics.imbalance_units == 1
    assert plan.balance_diagnostics.max_to_mean_ratio == pytest.approx(1.2)


def test_single_case_sharding_plan_claim_gating_fails_closed() -> None:
    plan = plan_single_case_sharded_solve(
        requested_devices=2,
        backend="gpu",
        available_device_ids=["0", "1"],
        rhs_mode=2,
        shard_axis="flat",
        benchmark_kind="transport_worker_scaling",
        task_count=3,
        release_scaling_claim=True,
        experimental_single_case_scaling=False,
    )

    assert plan.active_devices == 2
    assert plan.release_scaling_claim is False
    assert plan.release_scaling_supported is False
    assert plan.eligible_for_single_case_sharding is False
    assert any("benchmark_kind='single_case_sharded_solve'" in failure for failure in plan.failures)
    assert any("task_count=1" in failure for failure in plan.failures)
    assert any("RHSMode=1" in failure for failure in plan.failures)
    assert any("shard_axis" in failure for failure in plan.failures)
    assert any("release_scaling_claim=true" in failure for failure in plan.failures)
    assert any("marked experimental" in failure for failure in plan.failures)


def test_single_case_sharding_plan_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError, match="requested_devices must be a positive integer"):
        plan_single_case_sharded_solve(requested_devices=0)


def test_compiled_sharded_operator_reuse_gate_accepts_hot_inner_warmup() -> None:
    gate = plan_compiled_sharded_operator_reuse(
        benchmark_kind="single_case_sharded_solve",
        timing_semantics="hot_solve",
        inner_warmup_runs=1,
        timed_repeats=2,
        work_units_per_sample=4,
        compile_cache_dir="examples/performance/output/cache",
        persistent_compile_cache=True,
    )

    assert gate.passes is True
    assert gate.warm_run_amortization_pass is True
    assert gate.strategy == "inner_warmup"
    assert gate.compile_in_timed_region is False
    assert gate.failures == ()


def test_compiled_sharded_operator_reuse_gate_fails_closed_without_cache() -> None:
    gate = plan_compiled_sharded_operator_reuse(
        benchmark_kind="single_case_sharded_solve",
        timing_semantics="cache_warm",
        global_warmup_runs=1,
        timed_repeats=1,
        compile_cache_dir=None,
        persistent_compile_cache=False,
    )

    assert gate.passes is False
    assert gate.strategy == "global_persistent_compile_cache"
    assert any("persistent_compile_cache=true" in failure for failure in gate.failures)
    assert any("compile_cache_dir" in failure for failure in gate.failures)


def test_sharded_solve_deterministic_output_gate_schema_is_fail_closed() -> None:
    missing = plan_sharded_solve_deterministic_output_gate(comparison_devices=2)
    assert missing.passes is False
    assert missing.status == "not_measured"
    assert any("max_relative_residual_norm" in failure for failure in missing.failures)
    assert any("output digest" in failure for failure in missing.failures)

    measured = plan_sharded_solve_deterministic_output_gate(
        comparison_devices=2,
        max_relative_residual_norm=2.0e-12,
        output_digest="abc123",
    )
    assert measured.passes is True
    assert measured.status == "pass"

    measured_pair = plan_sharded_solve_deterministic_output_gate(
        comparison_devices=2,
        max_relative_residual_norm=2.0e-12,
        baseline_output_digest="baseline",
        comparison_output_digest="comparison",
        evidence_source="measured_solve_output_digest",
    )
    assert measured_pair.passes is True
    assert measured_pair.output_digest == "comparison"
    assert measured_pair.output_digest_match is False
    assert measured_pair.evidence_source == "measured_solve_output_digest"
    assert any("digests differ" in note for note in measured_pair.notes)


def test_sharded_solve_amortization_rejects_communication_dominated_claim() -> None:
    diagnostics = estimate_sharded_solve_amortization(
        active_devices=2,
        serial_work_units=100.0,
        setup_work_units=40.0,
        krylov_iterations=80,
        collectives_per_iteration=2,
        collective_latency_units=0.4,
        halo_bytes_per_iteration=2.0e8,
        bandwidth_bytes_per_unit=2.0e9,
        min_work_units_per_device=20.0,
    )

    assert diagnostics.release_scaling_supported is False
    assert diagnostics.predicted_speedup < 1.25
    assert diagnostics.communication_fraction > 0.35
    assert any("communication fraction" in failure for failure in diagnostics.failures)
    assert "communication cost exceeds per-device compute work" in diagnostics.notes


def test_sharded_solve_amortization_accepts_compute_dominated_claim() -> None:
    diagnostics = estimate_sharded_solve_amortization(
        active_devices=2,
        serial_work_units=1000.0,
        setup_work_units=20.0,
        krylov_iterations=25,
        collectives_per_iteration=1,
        collective_latency_units=0.05,
        halo_bytes_per_iteration=1.0e6,
        bandwidth_bytes_per_unit=1.0e9,
        min_work_units_per_device=100.0,
        min_speedup=1.7,
        min_efficiency=0.8,
        max_communication_fraction=0.05,
    )

    assert diagnostics.release_scaling_supported is True
    assert diagnostics.predicted_speedup == pytest.approx(1.9175, rel=5.0e-4)
    assert diagnostics.parallel_efficiency > 0.9
    assert diagnostics.communication_fraction < 0.01
    assert diagnostics.failures == ()


def test_operator_coarse_reuse_plan_records_required_promotion_blockers() -> None:
    plan = plan_single_case_operator_coarse_reuse(
        active_devices=2,
        backend="gpu",
        rhs_mode=1,
        shard_axis="theta",
        operator_reuse_enabled=True,
        operator_reuse_gate_pass=True,
        deterministic_output_gate_pass=False,
        coarse_strategy="replicated_schur_schwarz",
        coarse_levels=2,
        max_coarse_rank=32,
    )

    assert plan.plan_valid is True
    assert plan.promotion_ready is False
    assert plan.backend == "gpu"
    assert plan.operator_build_scope == "once_per_child_process"
    assert plan.operator_action_scope == "compiled_sharded_device_function"
    assert plan.preconditioner_scope == "local_theta_slab_apply"
    assert plan.coarse_operator_scope == "replicated_small_dense_operator"
    assert "compiled_operator_reuse_gate" in plan.required_runtime_gates
    assert "deterministic 1-vs-N output gate has not passed" in plan.promotion_blockers
    assert "hot 1-vs-N speedup has not been measured" in plan.promotion_blockers
    assert "1-vs-N peak-memory growth has not been measured" in plan.promotion_blockers


def test_operator_coarse_reuse_plan_can_be_promotion_ready_after_measured_gates() -> None:
    plan = plan_single_case_operator_coarse_reuse(
        active_devices=2,
        backend="cpu",
        rhs_mode=1,
        shard_axis="zeta",
        operator_reuse_enabled=True,
        operator_reuse_gate_pass=True,
        deterministic_output_gate_pass=True,
        measured_hot_speedup=1.25,
        min_hot_speedup=1.15,
        memory_growth_fraction=-0.05,
        coarse_levels=1,
    )

    assert plan.plan_valid is True
    assert plan.promotion_ready is True
    assert plan.failures == ()
    assert plan.promotion_blockers == ()
    assert plan.per_device_components[0] == "zeta_slab_state"
    assert "projected_coarse_operator" in plan.replicated_components


def test_operator_coarse_reuse_plan_fails_closed_for_non_rhs1_or_bad_axis() -> None:
    plan = plan_single_case_operator_coarse_reuse(
        active_devices=1,
        backend="cpu",
        rhs_mode=2,
        shard_axis="flat",
        experimental_single_case_scaling=False,
    )

    assert plan.plan_valid is False
    assert plan.promotion_ready is False
    assert any("RHSMode=1" in failure for failure in plan.failures)
    assert any("shard_axis" in failure for failure in plan.failures)
    assert any("at least two active devices" in failure for failure in plan.failures)
    assert any("marked experimental" in failure for failure in plan.failures)
