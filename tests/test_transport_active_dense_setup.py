from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.problems.transport_matrix.linear_system import resolve_transport_active_dense_setup


def _op(*, total_size: int = 2000, n_xi: int = 4, nxi_for_x=(4, 2)):
    return SimpleNamespace(
        total_size=total_size,
        n_x=1,
        n_xi=n_xi,
        fblock=SimpleNamespace(
            fp=None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray(nxi_for_x, dtype=np.int32)),
        ),
    )


def _clear_env(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_TRANSPORT_LOW_MEMORY",
        "SFINCS_JAX_TRANSPORT_STREAM_DIAGNOSTICS",
        "SFINCS_JAX_TRANSPORT_STORE_STATE",
        "SFINCS_JAX_TRANSPORT_FORCE_KRYLOV",
        "SFINCS_JAX_TRANSPORT_FORCE_DENSE",
        "SFINCS_JAX_TRANSPORT_DENSE_FALLBACK",
        "SFINCS_JAX_TRANSPORT_DENSE_FALLBACK_MAX",
        "SFINCS_JAX_TRANSPORT_DENSE_RETRY_MAX",
        "SFINCS_JAX_TRANSPORT_DENSE_MAX_MB",
        "SFINCS_JAX_TRANSPORT_GMRES_RESTART",
        "SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY",
        "SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MIN",
        "SFINCS_JAX_TRANSPORT_GEOM5_MONO_LOW_MEMORY_MAX",
        "SFINCS_JAX_TRANSPORT_ACTIVE_DOF",
        "SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX",
        "SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB",
    ):
        monkeypatch.delenv(name, raising=False)


def _resolve(monkeypatch, *, op=None, active_dof_env: str | None = ""):
    _clear_env(monkeypatch)
    op = _op() if op is None else op
    return resolve_transport_active_dense_setup(
        op=op,
        rhs_mode=2,
        n_rhs=2,
        solve_method="auto",
        restart=60,
        maxiter=None,
        backend="cpu",
        geometry_scheme=1,
        dense_accelerator_auto_allowed=False,
        dense_backend_policy_allowed=True,
        state_out_requested=False,
        force_stream_diagnostics=None,
        force_store_state=None,
        subset_mode=False,
        active_dof_indices=lambda _op: np.arange(20, dtype=np.int32),
        active_dof_env=active_dof_env,
    )


def test_active_dense_setup_compacts_active_dofs_and_auto_selects_dense(monkeypatch) -> None:
    setup = _resolve(monkeypatch)

    assert setup.use_active_dof_mode
    assert setup.active_size == 20
    assert setup.active_idx_np.tolist() == list(range(20))
    assert setup.full_to_active_jnp.shape == (2000,)
    assert setup.solve_method_use == "dense"
    assert setup.gmres_restart == 40
    assert not setup.low_memory_outputs
    assert any("active-DOF mode enabled" in message for _, message in setup.active_notes)
    assert any("auto dense solve for RHSMode=2" in message for _, message in setup.dense_notes)


def test_active_dense_setup_reports_disabled_active_hint(monkeypatch) -> None:
    setup = _resolve(monkeypatch, op=_op(total_size=100, nxi_for_x=(4, 4)))

    assert not setup.use_active_dof_mode
    assert setup.active_size == 100
    assert setup.active_idx_np is None
    assert any("active-DOF mode disabled" in message for _, message in setup.active_notes)


def test_active_dense_setup_reports_dense_preconditioner_memory_guard(monkeypatch) -> None:
    _clear_env(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FORCE_KRYLOV", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX", "5000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_DENSE_PRECOND_MAX_MB", "1")

    setup = resolve_transport_active_dense_setup(
        op=_op(total_size=2000, nxi_for_x=(4, 4)),
        rhs_mode=2,
        n_rhs=2,
        solve_method="incremental",
        restart=60,
        maxiter=None,
        backend="cpu",
        geometry_scheme=1,
        dense_accelerator_auto_allowed=False,
        dense_backend_policy_allowed=True,
        state_out_requested=False,
        force_stream_diagnostics=None,
        force_store_state=None,
        subset_mode=False,
        active_dof_indices=lambda _op: np.arange(2000, dtype=np.int32),
        active_dof_env="0",
    )

    assert setup.solve_method_use == "incremental"
    assert not setup.dense_precond_enabled
    assert any("dense preconditioner disabled" in message for _, message in setup.dense_notes)
