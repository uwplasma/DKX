from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp

from sfincs_jax.problems.profile_response.sparse_pc import (
    SparsePCGMRESContext,
    SparsePCPostMinresContext,
    apply_sparse_pc_post_minres,
    resolve_sparse_pc_entry_policy,
    resolve_xblock_sparse_pc_setup,
    run_sparse_pc_gmres_once,
)


def _identity(v: jnp.ndarray) -> jnp.ndarray:
    return v


def _op(*, fp=False, pas=False, constraint_scheme=1, n_zeta=1, n_species=1) -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=constraint_scheme,
        include_phi1=False,
        n_zeta=n_zeta,
        n_species=n_species,
        point_at_x0=False,
        fblock=SimpleNamespace(
            fp=object() if fp else None,
            pas=object() if pas else None,
        ),
    )


def test_sparse_pc_entry_policy_classifies_pas_er_and_active_dof() -> None:
    def parse_config(**kwargs):
        assert kwargs["default_restart"] == 50
        assert kwargs["default_maxiter"] == 100
        return 50, 100

    setup = resolve_sparse_pc_entry_policy(
        op=_op(pas=True, constraint_scheme=2, n_zeta=1, n_species=1),
        solve_method_kind="sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=False,
        xblock_active_dof_requested=False,
        active_maps_available=False,
        use_dkes=True,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=True,
        er_abs_sparse_pc=0.2,
        restart=50,
        maxiter=80,
        parse_polish_gmres_config=parse_config,
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"] - 5,
        env={"SFINCS_JAX_RHSMODE1_SPARSE_PC_FP_DENSE_VELOCITY_BLOCK": "1"},
    )

    assert setup.constrained_pas_pc
    assert setup.tokamak_pas_er_pc
    assert not setup.tokamak_pas_noer_pc
    assert setup.sparse_pc_use_active_dof
    assert setup.sparse_pc_fp_dense_velocity_block is True
    assert setup.pc_restart == 45
    assert setup.pc_maxiter == 100


def test_sparse_pc_entry_policy_classifies_xblock_active_maps() -> None:
    setup = resolve_sparse_pc_entry_policy(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3, n_species=2),
        solve_method_kind="xblock_sparse_pc_gmres",
        has_reduced_modes=True,
        use_active_dof_mode=True,
        xblock_active_dof_requested=True,
        active_maps_available=True,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        er_abs_sparse_pc=0.0,
        restart=10,
        maxiter=None,
        parse_polish_gmres_config=lambda **_kwargs: (20, 400),
        sparse_pc_default_restart=lambda **kwargs: kwargs["requested_restart"],
        env={},
    )

    assert setup.xblock_sparse_pc
    assert setup.xblock_use_active_dof
    assert not setup.sparse_pc_use_active_dof
    assert setup.pc_restart == 20
    assert setup.pc_maxiter == 400


def test_xblock_sparse_pc_setup_resolves_host_assembly_and_device_fallback() -> None:
    fallback_calls: list[dict[str, object]] = []

    def fallback_decision(**kwargs):
        fallback_calls.append(kwargs)
        return SimpleNamespace(
            used=True,
            ignored_env=False,
            mode="host",
            reason="forced",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value="auto",
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=True,
        )

    setup = resolve_xblock_sparse_pc_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        preconditioner_species=0,
        preconditioner_xi=0,
        active_size=1000,
        lower_fill_mode=lambda value: ("force", value == "bad"),
        species_decoupled_for_host_assembly=lambda **_kwargs: True,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: ("gmres_jax" if value == "gmres_jax" else "gmres", False),
        device_host_fallback_decision=fallback_decision,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_DROP_TOL": "1e-5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL": "force",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "gmres_jax",
            "SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK": "host",
        },
    )

    assert setup.xblock_drop_tol == pytest.approx(1.0e-5)
    assert setup.xblock_lower_fill_mode == "force"
    assert setup.xblock_preconditioner_xi == 1
    assert setup.force_assembled_host_fp
    assert setup.xblock_assembled_host_fp
    assert setup.xblock_krylov_env_requested == "gmres_jax"
    assert setup.xblock_krylov_env == "auto"
    assert setup.xblock_krylov_requested == "gmres"
    assert not setup.xblock_device_krylov_requested
    assert setup.xblock_device_host_fallback_decision.used
    assert any("non-autodiff host x-block fallback" in message for _, message in setup.messages)
    assert fallback_calls[0]["requested_krylov_method"] == "gmres_jax"


def test_xblock_sparse_pc_setup_disables_auto_host_fallback_for_qi_device_request() -> None:
    def fallback_decision(**kwargs):
        assert kwargs["env_value"] == "off"
        return SimpleNamespace(
            used=False,
            ignored_env=False,
            mode="disabled",
            reason="disabled",
            requested_method=kwargs["requested_krylov_method"],
            effective_krylov_env_value=kwargs["env_value"],
            min_active_size=1,
            qi_like_full_fp_3d=False,
            non_autodiff=False,
        )

    setup = resolve_xblock_sparse_pc_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        preconditioner_species=1,
        preconditioner_xi=1,
        active_size=2000,
        lower_fill_mode=lambda _value: ("off", False),
        species_decoupled_for_host_assembly=lambda **_kwargs: False,
        assembled_host_allowed=lambda **_kwargs: False,
        krylov_method=lambda value: ("gmres_jax" if value == "gmres_jax" else "gmres", False),
        device_host_fallback_decision=fallback_decision,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV": "gmres_jax",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
        },
    )

    assert setup.xblock_device_krylov_requested
    assert setup.xblock_device_host_fallback_auto_disabled_by_qi_device
    assert setup.qi_device_preconditioner_requested_for_fallback
    assert any("fallback disabled by explicit matrix-free" in message for _, message in setup.messages)


def test_sparse_pc_gmres_once_explicit_left_recomputes_true_residual() -> None:
    messages: list[str] = []
    times = iter((0.0, 0.25, 0.5, 0.75))

    def explicit_left_solver(**kwargs):
        kwargs["progress_callback"](2, 4.0e-1)
        return np.asarray([0.25, 0.75]), 99.0, 0.5, (1.0, 0.4)

    result = run_sparse_pc_gmres_once(
        context=SparsePCGMRESContext(
            matvec=lambda x: 2.0 * x,
            rhs=jnp.asarray([1.0, 1.0]),
            preconditioner=_identity,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            restart=7,
            tol=1.0e-8,
            atol=0.0,
            precondition_side="left",
            factor_dtype=np.dtype(np.float32),
            progress_every=2,
            stagnation_abort=False,
            stagnation_min_iter=10,
            stagnation_window=10,
            stagnation_rel_improvement=1.0e-3,
            explicit_left_solver=explicit_left_solver,
            gmres_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("wrong solver")),
        ),
        x0=None,
        maxiter=3,
    )

    assert result.x.tolist() == [0.25, 0.75]
    assert result.preconditioned_residual_norm == pytest.approx(0.5)
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.5, -0.5]))
    assert result.history == (1.0, 0.4)
    assert any("factor_dtype=float32" in msg for msg in messages)
    assert any("iters=2" in msg for msg in messages)


def test_sparse_pc_gmres_once_stagnation_guard_raises() -> None:
    def gmres_solver(**kwargs):
        progress = kwargs["progress_callback"]
        progress(1, 1.0)
        progress(2, 1.0)
        return np.ones(2), 1.0, (1.0,)

    with pytest.raises(RuntimeError, match="sparse_pc_gmres stagnation detected"):
        run_sparse_pc_gmres_once(
            context=SparsePCGMRESContext(
                matvec=_identity,
                rhs=jnp.ones(2),
                preconditioner=_identity,
                emit=None,
                elapsed_s=lambda: 0.0,
                pc_form="right",
                restart=5,
                tol=1.0e-8,
                atol=0.0,
                precondition_side="right",
                factor_dtype=np.dtype(np.float64),
                progress_every=0,
                stagnation_abort=True,
                stagnation_min_iter=2,
                stagnation_window=1,
                stagnation_rel_improvement=1.0e-3,
                explicit_left_solver=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("wrong solver")),
                gmres_solver=gmres_solver,
            ),
            x0=None,
            maxiter=10,
        )


def test_sparse_pc_post_minres_accepts_improved_residual_and_recomputes_pc_norm() -> None:
    messages: list[str] = []
    times = iter((1.0, 1.4))

    def minres_correction(**_kwargs):
        return (
            jnp.asarray([0.5, 0.5]),
            jnp.asarray([0.1, 0.2]),
            (0.9, 0.25),
            (0.75,),
        )

    result = apply_sparse_pc_post_minres(
        context=SparsePCPostMinresContext(
            matvec=_identity,
            rhs=jnp.zeros(2),
            preconditioner=lambda v: 0.5 * v,
            emit=lambda _level, msg: messages.append(msg),
            elapsed_s=lambda: next(times),
            pc_form="explicit_left",
            steps=2,
            alpha_clip=10.0,
            min_improvement=0.0,
            minres_correction=minres_correction,
        ),
        x=np.zeros(2),
        residual_norm=1.0,
        preconditioned_residual_norm=float("nan"),
    )

    assert result.x.tolist() == [0.5, 0.5]
    assert result.residual_norm == pytest.approx(np.linalg.norm([0.1, 0.2]))
    assert result.preconditioned_residual_norm == pytest.approx(np.linalg.norm([-0.25, -0.25]))
    assert result.history == (0.9, 0.25)
    assert result.alphas == (0.75,)
    assert result.error is None
    assert result.solve_s == pytest.approx(0.4)
    assert any("post-minres improved residual" in msg for msg in messages)
