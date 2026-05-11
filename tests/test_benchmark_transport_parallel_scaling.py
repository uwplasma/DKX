from __future__ import annotations

import os

from examples.performance.benchmark_transport_parallel_scaling import _configure_backend_env, _write_scaling_figure


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
