from __future__ import annotations

from types import SimpleNamespace

import pytest

import sfincs_jax.problems.profile_solve as profile_solve


def test_resolve_distributed_gmres_axis_allows_nondivisible_with_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(profile_solve.jax, "local_device_count", lambda: 8)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "cpu")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert profile_solve._resolve_distributed_gmres_axis(op=op, emit=None) == "theta"


def test_resolve_distributed_gmres_axis_disables_auto_on_gpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr(profile_solve.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(profile_solve, "_matvec_shard_axis", lambda op: "zeta")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert profile_solve._resolve_distributed_gmres_axis(op=op, emit=None) is None


def test_resolve_distributed_gmres_axis_allows_explicit_gpu_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "theta")
    monkeypatch.setattr(profile_solve.jax, "local_device_count", lambda: 2)
    monkeypatch.setattr(profile_solve.jax, "default_backend", lambda: "gpu")
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    assert profile_solve._resolve_distributed_gmres_axis(op=op, emit=None) == "theta"
