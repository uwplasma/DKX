from __future__ import annotations

from pathlib import Path

from examples.performance.benchmark_sharded_matvec_scaling import _build_sharded_matvec_benchmark_plan
from examples.performance.benchmark_sharded_solve_scaling import (
    _build_sharded_solve_benchmark_plan,
    _configure_backend_env,
    _configure_benchmark_subprocess_env,
    _configure_solver_env,
    _run_once_command,
    _run_once_subprocess,
    _timing_semantics,
)
from sfincs_jax.transport_parallel_policy import audit_sharded_solve_scaling_summary


def test_configure_backend_env_cpu() -> None:
    env = {"CUDA_VISIBLE_DEVICES": "0,1"}
    _configure_backend_env(env=env, devices=4, backend="cpu")
    assert env["SFINCS_JAX_CPU_DEVICES"] == "4"
    assert "CUDA_VISIBLE_DEVICES" not in env


def test_configure_backend_env_gpu() -> None:
    env = {"SFINCS_JAX_CPU_DEVICES": "8"}
    _configure_backend_env(env=env, devices=2, backend="gpu")
    assert "SFINCS_JAX_CPU_DEVICES" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert env["SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR"] == "1"


def test_configure_backend_env_auto_defaults_to_cpu_devices() -> None:
    env: dict[str, str] = {}
    _configure_backend_env(env=env, devices=8, backend="auto")
    assert env["SFINCS_JAX_CPU_DEVICES"] == "8"


def test_configure_solver_env_sets_multilevel_schwarz_controls() -> None:
    env: dict[str, str] = {}
    _configure_solver_env(
        env=env,
        shard_axis="theta",
        gmres_distributed="0",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        rhs1_precond="theta_schwarz",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=1,
        schwarz_coarse_damp=0.8,
    )
    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "theta_schwarz"
    assert env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS"] == "2"
    assert env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_STEPS"] == "1"
    assert env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_DAMP"] == "0.8"


def test_configure_solver_env_enables_accelerator_distributed_gmres_for_sharded_path() -> None:
    env: dict[str, str] = {}
    _configure_solver_env(
        env=env,
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        rhs1_precond="theta_schwarz",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=1,
        schwarz_coarse_damp=0.8,
    )
    assert env["SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR"] == "1"


def test_configure_benchmark_subprocess_env_quiets_jax_runtime_logs() -> None:
    env: dict[str, str] = {}
    _configure_benchmark_subprocess_env(env)
    assert env["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert env["GLOG_minloglevel"] == "2"
    assert env["ABSL_MIN_LOG_LEVEL"] == "2"


def test_timing_semantics_labels_sharded_hot_and_cold_modes() -> None:
    assert _timing_semantics(global_warmup=0, per_device_warmup=0, inner_warmup_solves=1) == "hot_solve"
    assert _timing_semantics(global_warmup=1, per_device_warmup=0, inner_warmup_solves=0) == "cache_warm"
    assert _timing_semantics(global_warmup=0, per_device_warmup=0, inner_warmup_solves=0) == "cold_start"


def test_run_once_subprocess_passes_recorded_solver_options(monkeypatch, tmp_path) -> None:
    calls: list[tuple[list[str], dict[str, str], float | None]] = []

    def fake_check_output(cmd, *, env, text, timeout):  # noqa: ANN001
        calls.append((list(cmd), dict(env), timeout))
        return "0.125\n"

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    dt = _run_once_subprocess(
        input_path=Path("case.input.namelist"),
        devices=2,
        cache_dir=tmp_path / "cache",
        shard_axis="zeta",
        gmres_distributed="0",
        distributed_krylov="off",
        periodic_stencil_on_sharded="off",
        nsolve=3,
        inner_warmup_solves=1,
        sample_timeout_s=45.0,
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=1,
        schwarz_coarse_damp=0.75,
    )

    assert dt == 0.125
    cmd, env, timeout = calls[0]
    assert "--shard-axis" in cmd and cmd[cmd.index("--shard-axis") + 1] == "zeta"
    assert "--gmres-distributed" in cmd and cmd[cmd.index("--gmres-distributed") + 1] == "0"
    assert "--distributed-krylov" in cmd and cmd[cmd.index("--distributed-krylov") + 1] == "off"
    assert "--inner-warmup-solves" in cmd
    assert cmd[cmd.index("--inner-warmup-solves") + 1] == "1"
    assert "--periodic-stencil-on-sharded" in cmd
    assert cmd[cmd.index("--periodic-stencil-on-sharded") + 1] == "off"
    assert "--rhs1-precond" in cmd and cmd[cmd.index("--rhs1-precond") + 1] == "theta_schwarz"
    assert "--backend" in cmd and cmd[cmd.index("--backend") + 1] == "gpu"
    assert "--schwarz-coarse-levels" in cmd and cmd[cmd.index("--schwarz-coarse-levels") + 1] == "2"
    assert "--schwarz-coarse-steps" in cmd and cmd[cmd.index("--schwarz-coarse-steps") + 1] == "1"
    assert "--schwarz-coarse-damp" in cmd and cmd[cmd.index("--schwarz-coarse-damp") + 1] == "0.75"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache")
    assert env["TF_CPP_MIN_LOG_LEVEL"] == "2"
    assert timeout == 45.0


def test_run_once_command_records_all_sharded_solver_options() -> None:
    cmd = _run_once_command(
        input_path=Path("case.input.namelist"),
        shard_axis="zeta",
        gmres_distributed="0",
        distributed_krylov="off",
        periodic_stencil_on_sharded="off",
        nsolve=3,
        inner_warmup_solves=1,
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=1,
        schwarz_coarse_damp=0.75,
    )

    assert cmd[1].endswith("benchmark_sharded_solve_scaling.py")
    assert "--run-once" in cmd
    assert cmd[cmd.index("--shard-axis") + 1] == "zeta"
    assert cmd[cmd.index("--gmres-distributed") + 1] == "0"
    assert cmd[cmd.index("--distributed-krylov") + 1] == "off"
    assert cmd[cmd.index("--periodic-stencil-on-sharded") + 1] == "off"
    assert cmd[cmd.index("--rhs1-precond") + 1] == "theta_schwarz"
    assert cmd[cmd.index("--backend") + 1] == "gpu"
    assert cmd[cmd.index("--schwarz-coarse-levels") + 1] == "2"


def test_sharded_solve_plan_records_hot_timing_timeout_and_non_release_gate(tmp_path: Path) -> None:
    plan = _build_sharded_solve_benchmark_plan(
        input_path=Path("examples/performance/rhsmode1_sharded_scaling.input.namelist"),
        devices=[2, 1],
        warmup=0,
        repeats=1,
        nsolve=4,
        inner_warmup_solves=1,
        sample_timeout_s=120.0,
        global_warmup=0,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=None,
        schwarz_coarse_damp=None,
        audit=True,
    )

    assert plan["artifact_kind"] == "benchmark_plan"
    assert plan["launches_solves"] is False
    assert plan["devices"] == [1, 2]
    assert plan["timing_semantics"] == "hot_solve"
    assert plan["release_scaling_claim"] is False
    assert plan["speedup_gate_semantics"]["gate_scope"] == "schema_and_honesty_only"
    assert plan["memory_gate_semantics"]["child_process_timeout_enabled"] is True
    assert plan["parallel_claim_scope"]["claim_scope"] == "single_case_sharded_solve_experimental"
    assert plan["parallel_claim_scope"]["claim_scope_release_eligible"] is True
    assert plan["parallel_claim_scope"]["release_scaling_supported"] is False
    assert plan["parallel_claim_scope"]["unsupported_single_case_strong_scaling"] is True
    assert plan["parallel_claim_scope"]["backend"] == "gpu"
    assert plan["parallel_claim_scope"]["artifact_kind"] == "benchmark_plan"
    assert plan["parallel_claim_scope"]["launches_solves"] is False
    assert plan["parallel_claim_scope"]["plan_only_scope_evidence"] is True
    assert plan["parallel_claim_scope"]["measured_results_present"] is False
    assert plan["parallel_claim_scope"]["release_gate_required"] is None
    assert plan["device_plan"][1]["env"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert plan["device_plan"][1]["sharding_plan"]["requested_devices"] == 2
    assert plan["device_plan"][1]["sharding_plan"]["active_devices"] == 2
    assert plan["device_plan"][1]["sharding_plan"]["eligible_for_single_case_sharding"] is True
    assert plan["device_plan"][1]["sharding_plan"]["release_scaling_supported"] is False
    assert plan["device_plan"][1]["sharding_plan"]["balance_diagnostics"]["imbalance_units"] == 0
    assert "--audit" in plan["benchmark_command"]


def test_sharded_solve_audit_keeps_single_case_scaling_non_release() -> None:
    payload = {
        "benchmark_kind": "single_case_sharded_solve",
        "scaling_status": "experimental_single_case_sharding",
        "experimental_single_case_scaling": True,
        "release_scaling_claim": True,
        "backend": "cpu",
        "timing_semantics": "hot_solve",
        "deterministic_output_check": False,
        "results": [
            {"devices": 1, "mean_s": 10.0, "speedup": 1.0},
            {"devices": 2, "mean_s": 4.0, "speedup": 2.5},
        ],
    }

    audit = audit_sharded_solve_scaling_summary(payload)

    assert audit.release_scaling_claim is False
    assert audit.ci_gate_pass is False
    assert any("release_scaling_claim=true" in failure for failure in audit.failures)


def test_sharded_matvec_plan_records_compiled_hot_loop_and_padding(tmp_path: Path) -> None:
    plan = _build_sharded_matvec_benchmark_plan(
        input_path=Path("examples/performance/transport_parallel_sharded.input.namelist"),
        devices=[2, 1],
        nrep=20,
        repeats=2,
        global_warmup=1,
        axis="theta",
        pad=True,
        out_dir=tmp_path / "out",
        cache_dir=tmp_path / "cache",
    )

    assert plan["artifact_kind"] == "benchmark_plan"
    assert plan["benchmark_kind"] == "sharded_matvec_scaling"
    assert plan["devices"] == [1, 2]
    assert plan["timing_semantics"] == "compiled_matvec_hot_loop"
    assert plan["memory_gate_semantics"]["padding_enabled"] is True
    assert plan["estimated_child_process_samples"] == 5
