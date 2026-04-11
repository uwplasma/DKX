from __future__ import annotations

from examples.performance.benchmark_sharded_solve_scaling import _configure_backend_env


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
