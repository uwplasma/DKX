from __future__ import annotations

from pathlib import Path

from examples.performance.benchmark_multi_gpu_case_throughput import (
    _base_env,
    _build_case_throughput_plan,
    _case_run_once_command,
    _run_case_once,
)
from sfincs_jax.transport_parallel_policy import audit_multi_gpu_case_throughput_summary


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
        sample_timeout_s=120.0,
    )

    assert plan["artifact_kind"] == "benchmark_plan"
    assert plan["launches_solves"] is False
    assert plan["required_gpu_count"] == 2
    assert plan["timing_semantics"] == "cache_warm"
    assert plan["sample_timeout_s"] == 120.0
    assert [entry["visible_devices"] for entry in plan["warmup_plan"]] == ["0", "1"]
    assert [entry["visible_devices"] for entry in plan["sequential_one_gpu_plan"]] == ["0", "0"]
    assert [entry["visible_devices"] for entry in plan["parallel_two_gpu_plan"]] == ["0", "1"]
    assert plan["speedup_gate_semantics"]["release_gate"] is False
    assert plan["speedup_gate_semantics"]["evaluated_by"] == "audit_multi_gpu_case_throughput_summary"
    assert plan["memory_gate_semantics"]["gpu_preallocation_disabled"] is True
    assert plan["memory_gate_semantics"]["child_process_timeout_enabled"] is True


def test_run_case_once_applies_child_timeout(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str], float | None]] = []

    def fake_check_output(cmd, *, env, text, timeout):  # noqa: ANN001
        calls.append((list(cmd), dict(env), timeout))
        return "0.25\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    dt = _run_case_once(
        input_path=Path("case.input.namelist"),
        visible_devices="1",
        nsolve=2,
        env={"A": "B"},
        sample_timeout_s=45.0,
    )

    assert dt == 0.25
    cmd, env, timeout = calls[0]
    assert cmd[1].endswith("benchmark_sharded_solve_scaling.py")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert timeout == 45.0


def test_multi_gpu_case_throughput_audit_rejects_non_improving_or_release_payloads() -> None:
    payload = {
        "benchmark_kind": "multi_gpu_case_throughput",
        "backend": "gpu",
        "required_gpu_count": 2,
        "release_scaling_claim": False,
        "timing_semantics": "cache_warm",
        "sequential_one_gpu": {"wall_s": 20.0},
        "parallel_two_gpu": {"wall_s": 10.0},
        "throughput_speedup": 2.0,
    }

    passing = audit_multi_gpu_case_throughput_summary(payload)
    assert passing.ci_gate_pass is True
    assert passing.release_scaling_claim is False
    assert passing.throughput_speedup == 2.0

    weak = audit_multi_gpu_case_throughput_summary({**payload, "parallel_two_gpu": {"wall_s": 25.0}})
    assert weak.ci_gate_pass is False
    assert any("throughput speedup" in failure for failure in weak.failures)

    release = audit_multi_gpu_case_throughput_summary({**payload, "release_scaling_claim": True})
    assert release.release_scaling_claim is False
    assert release.ci_gate_pass is False
    assert any("release_scaling_claim=true" in failure for failure in release.failures)
