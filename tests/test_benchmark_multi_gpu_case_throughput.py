from __future__ import annotations

from pathlib import Path

from examples.performance.benchmark_multi_gpu_case_throughput import _base_env


def test_base_env_sets_gpu_benchmark_defaults(tmp_path: Path) -> None:
    env = _base_env(tmp_path / "jax_cache", "theta_schwarz", 2)
    assert env["TF_GPU_ALLOCATOR"] == "cuda_malloc_async"
    assert env["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert env["SFINCS_JAX_RHSMODE1_PRECONDITIONER"] == "theta_schwarz"
    assert env["SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS"] == "2"
    assert env["JAX_COMPILATION_CACHE_DIR"] == str(tmp_path / "jax_cache")
