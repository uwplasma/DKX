from __future__ import annotations

from pathlib import Path

from examples.performance.benchmark_multi_gpu_case_throughput import (
    _base_env,
    _build_case_throughput_plan,
    _case_run_once_command,
)


def test_base_env_sets_gpu_benchmark_defaults(tmp_path: Path) -> None:
    env = _base_env(tmp_path / "jax_cache", "theta_schwarz", 2)
    assert env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "theta_schwarz"
    assert env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS"] == "2"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "jax_cache")


def test_case_run_once_command_records_internal_sharded_solve_invocation() -> None:
    cmd = _case_run_once_command(input_path=Path("case.input.namelist"), nsolve=3)

    assert cmd[1].endswith("benchmark_sharded_solve_scaling.py")
    assert "--run-once" in cmd
    assert cmd[cmd.index("--input") + 1] == "case.input.namelist"
    assert cmd[cmd.index("--nsolve") + 1] == "3"


def test_case_throughput_plan_records_gpu_allocation_and_non_release_speedup_gate(tmp_path: Path) -> None:
    plan = _build_case_throughput_plan(
        input_path=Path("examples/performance/rhsmode1_sharded_scaling.input.namelist"),
        nsolve=4,
        rhs1_precond="theta_schwarz",
        coarse_levels=2,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )

    assert plan["artifact_kind"] == "benchmark_plan"
    assert plan["launches_solves"] is False
    assert plan["required_gpu_count"] == 2
    assert plan["timing_semantics"] == "cache_warm"
    assert [entry["visible_devices"] for entry in plan["warmup_plan"]] == ["0", "1"]
    assert [entry["visible_devices"] for entry in plan["sequential_one_gpu_plan"]] == ["0", "0"]
    assert [entry["visible_devices"] for entry in plan["parallel_two_gpu_plan"]] == ["0", "1"]
    assert plan["speedup_gate_semantics"]["release_gate"] is False
    assert plan["memory_gate_semantics"]["gpu_preallocation_disabled"] is True
