from __future__ import annotations

from types import SimpleNamespace

import pytest

from sfincs_jax.solvers import krylov_dispatch as kd


def test_host_scipy_krylov_requested_and_labels() -> None:
    assert kd.host_scipy_krylov_requested("lgmres")
    assert kd.host_scipy_krylov_requested(" LGMRES_SCIPY ")
    assert not kd.host_scipy_krylov_requested("incremental")
    assert kd.ksp_iteration_solver_label(solver_kind="gmres", solve_method="lgmres") == "lgmres"
    assert kd.ksp_iteration_solver_label(solver_kind="gmres", solve_method="auto") == "gmres"
    assert kd.ksp_iteration_solver_label(solver_kind="bicgstab", solve_method="lgmres") == "bicgstab"
    assert kd.solver_kind_for_label("bicgstab_jax") == ("bicgstab", "batched")
    assert kd.solver_kind_for_label("default") == ("gmres", "incremental")


def test_gmres_dispatch_uses_host_only_path_and_rejects_distributed() -> None:
    assert kd.gmres_solve_dispatch(
        solve_method="lgmres",
        distributed_axis=None,
        gmres_solve_fn=lambda **kwargs: ("host", kwargs["solve_method"]),
        distributed_gmres_enabled_fn=lambda: False,
    ) == ("host", "lgmres")

    with pytest.raises(ValueError, match="host-only"):
        kd.gmres_solve_dispatch(
            solve_method="lgmres",
            distributed_axis="theta",
            distributed_gmres_enabled_fn=lambda: False,
        )


def test_gmres_dispatch_selects_plain_jit_and_distributed_paths() -> None:
    assert kd.gmres_solve_dispatch(
        solve_method="incremental",
        distributed_axis=None,
        size_hint=20,
        gmres_solve_fn=lambda **kwargs: "plain",
        gmres_solve_jit_fn=lambda **kwargs: "jit",
        distributed_gmres_enabled_fn=lambda: False,
        use_solver_jit_fn=lambda size_hint: False,
    ) == "plain"
    assert kd.gmres_solve_dispatch(
        solve_method="incremental",
        distributed_axis=None,
        size_hint=20,
        gmres_solve_fn=lambda **kwargs: "plain",
        gmres_solve_jit_fn=lambda **kwargs: "jit",
        distributed_gmres_enabled_fn=lambda: False,
        use_solver_jit_fn=lambda size_hint: True,
    ) == "jit"
    assert kd.gmres_solve_dispatch(
        solve_method="incremental",
        distributed_axis="zeta",
        gmres_solve_distributed_fn=lambda **kwargs: ("distributed", kwargs["axis_name"]),
    ) == ("distributed", "zeta")


def test_gmres_with_residual_dispatch_uses_injected_paths() -> None:
    assert kd.gmres_solve_with_residual_dispatch(
        solve_method="lgmres_scipy",
        distributed_axis=None,
        gmres_solve_with_residual_fn=lambda **kwargs: ("host_residual", kwargs["solve_method"]),
        distributed_gmres_enabled_fn=lambda: False,
    ) == ("host_residual", "lgmres_scipy")
    assert kd.gmres_solve_with_residual_dispatch(
        solve_method="incremental",
        distributed_axis=None,
        gmres_solve_with_residual_fn=lambda **kwargs: "plain",
        gmres_solve_with_residual_jit_fn=lambda **kwargs: "jit",
        distributed_gmres_enabled_fn=lambda: False,
        use_solver_jit_fn=lambda size_hint: True,
    ) == "jit"


def test_rhs_krylov_method_for_context_downgrades_host_only_cases() -> None:
    assert (
        kd.rhs_krylov_method_for_context(
            gmres_method="lgmres",
            use_implicit=False,
            distributed_axis=None,
            solver_jit=False,
        )
        == "lgmres"
    )
    assert (
        kd.rhs_krylov_method_for_context(
            gmres_method="lgmres",
            use_implicit=True,
            distributed_axis=None,
            solver_jit=False,
        )
        == "incremental"
    )
    assert (
        kd.rhs_krylov_method_for_context(
            gmres_method="lgmres_scipy",
            use_implicit=False,
            distributed_axis="theta",
            solver_jit=False,
        )
        == "incremental"
    )


def test_resolve_distributed_gmres_axis_env_and_backend_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    op = SimpleNamespace(n_theta=65, n_zeta=65)
    notes: list[str] = []

    def emit(level: int, message: str) -> None:
        notes.append(f"{level}:{message}")

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "bad-token")
    assert kd.resolve_distributed_gmres_axis(op=op, emit=emit, matvec_shard_axis_fn=lambda op: "theta") is None
    assert "not recognized" in notes[-1]

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED", "auto")
    monkeypatch.delenv("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", raising=False)
    monkeypatch.setattr(kd.jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(kd.jax, "local_device_count", lambda: 2)
    assert kd.resolve_distributed_gmres_axis(op=op, emit=emit, matvec_shard_axis_fn=lambda op: "theta") is None

    monkeypatch.setenv("SFINCS_JAX_GMRES_DISTRIBUTED_ALLOW_ACCELERATOR", "1")
    assert kd.resolve_distributed_gmres_axis(op=op, emit=emit, matvec_shard_axis_fn=lambda op: "theta") == "theta"
