from __future__ import annotations

from examples.performance.benchmark_sharded_solve_scaling import _configure_backend_env, _configure_solver_env


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
