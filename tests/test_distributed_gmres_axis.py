from __future__ import annotations

from types import SimpleNamespace

import pytest

import sfincs_jax.solvers.krylov_dispatch as krylov_dispatch


def test_resolve_distributed_gmres_axis_allows_nondivisible_with_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(krylov_dispatch.jax, "local_device_count", lambda: 8)
    monkeypatch.setattr(krylov_dispatch.jax, "default_backend", lambda: "cpu")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert krylov_dispatch.resolve_distributed_gmres_axis(op=op, emit=None) == "theta"


def test_resolve_distributed_gmres_axis_disables_auto_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr(krylov_dispatch.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(krylov_dispatch.jax, "default_backend", lambda: "gpu")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert (
        krylov_dispatch.resolve_distributed_gmres_axis(
            op=op,
            emit=None,
            matvec_shard_axis_fn=lambda _op: "zeta",
        )
        is None
    )


def test_resolve_distributed_gmres_axis_allows_explicit_gpu_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(krylov_dispatch.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(krylov_dispatch.jax, "default_backend", lambda: "gpu")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert krylov_dispatch.resolve_distributed_gmres_axis(op=op, emit=None) == "theta"
