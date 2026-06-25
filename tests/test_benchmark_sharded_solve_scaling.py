from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.performance.benchmark_sharded_matvec_scaling import _build_sharded_matvec_benchmark_plan
from examples.performance.benchmark_sharded_solve_scaling import (
    _build_sharded_solve_benchmark_plan,
    _configure_backend_env,
    _configure_benchmark_subprocess_env,
    _configure_solver_env,
    _measure_deterministic_output_gate,
    _operator_reuse_enabled,
    _run_once,
    _run_once_command,
    _run_once_output_digest_command,
    _run_once_output_digest_subprocess,
    _run_once_subprocess,
    _timing_semantics,
)
from sfincs_jax.problems.transport_matrix.parallel.policy import audit_sharded_solve_scaling_summary


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


def test_operator_reuse_auto_preserves_cold_start_and_enables_hot_reuse() -> None:
    assert _operator_reuse_enabled(mode="auto", nsolve=1, inner_warmup_solves=0) is False
    assert _operator_reuse_enabled(mode="auto", nsolve=2, inner_warmup_solves=0) is True
    assert _operator_reuse_enabled(mode="auto", nsolve=1, inner_warmup_solves=1) is True
    assert _operator_reuse_enabled(mode="on", nsolve=1, inner_warmup_solves=0) is True
    assert _operator_reuse_enabled(mode="off", nsolve=8, inner_warmup_solves=8) is False


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
    assert "--operator-reuse" in cmd and cmd[cmd.index("--operator-reuse") + 1] == "auto"
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
    assert cmd[cmd.index("--operator-reuse") + 1] == "auto"
    assert cmd[cmd.index("--schwarz-coarse-levels") + 1] == "2"


def test_run_once_output_digest_command_records_vector_artifact_path() -> None:
    cmd = _run_once_output_digest_command(
        input_path=Path("case.input.namelist"),
        output_vector_path=Path("out/vector.npy"),
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        inner_warmup_solves=1,
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=None,
        schwarz_coarse_damp=None,
    )

    assert cmd[1].endswith("benchmark_sharded_solve_scaling.py")
    assert "--run-once-output-digest" in cmd
    assert cmd[cmd.index("--output-vector-path") + 1] == "out/vector.npy"
    assert cmd[cmd.index("--inner-warmup-solves") + 1] == "1"
    assert cmd[cmd.index("--backend") + 1] == "gpu"
    assert cmd[cmd.index("--operator-reuse") + 1] == "auto"


def test_run_once_reuses_full_system_operator_for_hot_child_process(monkeypatch) -> None:
    nml = object()
    op = object()
    operator_builds: list[object] = []
    solves: list[dict[str, object]] = []

    monkeypatch.setattr(
        "examples.performance.benchmark_sharded_solve_scaling.read_sfincs_input",
        lambda _path: nml,
    )

    def fake_build_operator(*, nml):  # noqa: ANN001
        operator_builds.append(nml)
        return op

    def fake_solve(**kwargs):  # noqa: ANN003
        solves.append(dict(kwargs))
        return SimpleNamespace(x=np.asarray([1.0]))

    monkeypatch.setattr(
        "examples.performance.benchmark_sharded_solve_scaling.full_system_operator_from_namelist",
        fake_build_operator,
    )
    monkeypatch.setattr(
        "examples.performance.benchmark_sharded_solve_scaling.solve_v3_full_system_linear_gmres",
        fake_solve,
    )
    monkeypatch.setattr(
        "examples.performance.benchmark_sharded_solve_scaling.jax.block_until_ready",
        lambda x: x,
    )

    _run_once(
        Path("case.input.namelist"),
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        nsolve=2,
        inner_warmup_solves=1,
        rhs1_precond="theta_schwarz",
        backend="cpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=None,
        schwarz_coarse_damp=None,
        operator_reuse="auto",
    )

    assert operator_builds == [nml]
    assert len(solves) == 3
    assert all(call["nml"] is nml for call in solves)
    assert all(call["op"] is op for call in solves)


def test_run_once_output_digest_subprocess_records_backend_env(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], dict[str, str], float | None]] = []

    def fake_check_output(cmd, *, env, text, timeout):  # noqa: ANN001
        calls.append((list(cmd), dict(env), timeout))
        return '{"output_digest_algorithm":"sha256","output_digest":"abc","solve_s":0.5}\n'

    monkeypatch.setattr("subprocess.check_output", fake_check_output)

    payload = _run_once_output_digest_subprocess(
        input_path=Path("case.input.namelist"),
        devices=2,
        cache_dir=tmp_path / "cache",
        output_vector_path=tmp_path / "vector.npy",
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        inner_warmup_solves=1,
        sample_timeout_s=42.0,
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=None,
        schwarz_coarse_damp=None,
    )

    assert payload["output_digest"] == "abc"
    cmd, env, timeout = calls[0]
    assert "--run-once-output-digest" in cmd
    assert env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "cache")
    assert timeout == 42.0


def test_measure_deterministic_output_gate_builds_residual_digest_schema(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def fake_digest_subprocess(*, devices, output_vector_path, **_kwargs):  # noqa: ANN001
        values = np.array([1.0, 2.0], dtype=np.float64)
        if int(devices) == 2:
            values = np.array([1.0, 2.0 + 1.0e-12], dtype=np.float64)
        np.save(output_vector_path, values)
        return {
            "output_digest_algorithm": "sha256",
            "output_digest": f"digest-devices-{devices}",
            "solve_s": 0.25,
        }

    monkeypatch.setattr(
        "examples.performance.benchmark_sharded_solve_scaling._run_once_output_digest_subprocess",
        fake_digest_subprocess,
    )

    gate = _measure_deterministic_output_gate(
        input_path=Path("case.input.namelist"),
        devices=[1, 2],
        cache_dir=tmp_path / "cache",
        deterministic_output_dir=tmp_path / "deterministic",
        shard_axis="theta",
        gmres_distributed="1",
        distributed_krylov="auto",
        periodic_stencil_on_sharded="auto",
        inner_warmup_solves=1,
        sample_timeout_s=60.0,
        rhs1_precond="theta_schwarz",
        backend="gpu",
        schwarz_coarse_levels=2,
        schwarz_coarse_steps=None,
        schwarz_coarse_damp=None,
        residual_tolerance=1.0e-9,
        repo_root=tmp_path,
    )

    assert gate["passes"] is True
    assert gate["status"] == "pass"
    assert gate["baseline_output_digest"] == "digest-devices-1"
    assert gate["comparison_output_digest"] == "digest-devices-2"
    assert gate["output_digest_match"] is False
    assert gate["max_relative_residual_norm"] < 1.0e-9
    assert gate["baseline_probe"]["devices"] == 1
    assert gate["comparison_probe"]["devices"] == 2


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
    assert plan["operator_reuse_gate"]["passes"] is True
    assert plan["operator_reuse_gate"]["strategy"] == "inner_warmup"
    assert plan["operator_reuse_gate"]["persistent_compile_cache"] is True
    assert plan["operator_reuse_gate"]["compile_in_timed_region"] is False
    assert plan["operator_reuse"] == "auto"
    assert plan["assembled_operator_reuse"]["enabled"] is True
    assert plan["assembled_operator_reuse"]["scope"] == "child_process_full_system_operator"
    assert plan["operator_coarse_reuse_plan"]["plan_valid"] is True
    assert plan["operator_coarse_reuse_plan"]["promotion_ready"] is False
    assert plan["operator_coarse_reuse_plan"]["operator_reuse_gate_pass"] is True
    assert plan["operator_coarse_reuse_plan"]["deterministic_output_gate_pass"] is False
    assert plan["operator_coarse_reuse_plan"]["coarse_levels"] == 2
    assert plan["operator_coarse_reuse_plan"]["coarse_operator_scope"] == "replicated_small_dense_operator"
    blockers = plan["operator_coarse_reuse_plan"]["promotion_blockers"]
    assert "deterministic 1-vs-N output gate has not passed" in blockers
    assert plan["deterministic_output_gate"]["passes"] is False
    assert plan["deterministic_output_gate"]["status"] == "not_measured"
    assert plan["deterministic_output_gate"]["evidence_source"] == "not_measured"
    assert plan["deterministic_output_probe_requested"] is False
    assert plan["release_scaling_claim"] is False
    assert plan["speedup_gate_semantics"]["gate_scope"] == "schema_and_honesty_only"
    assert plan["speedup_gate_semantics"]["requires_operator_reuse_gate"] is True
    assert plan["speedup_gate_semantics"]["requires_deterministic_output_gate_for_claim"] is True
    assert plan["speedup_gate_semantics"]["requires_operator_coarse_reuse_plan"] is True
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
    assert plan["operator_reuse_gate"]["passes"] is True
    assert plan["operator_reuse_gate"]["strategy"] == "inner_warmup"
    assert plan["operator_reuse_gate"]["work_units_per_sample"] == 20
    assert plan["memory_gate_semantics"]["padding_enabled"] is True
    assert plan["estimated_child_process_samples"] == 5
