from __future__ import annotations

import pytest

from sfincs_jax.transport_parallel_sharding import plan_single_case_sharded_solve


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
