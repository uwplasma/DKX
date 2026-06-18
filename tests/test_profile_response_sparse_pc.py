from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import jax.numpy as jnp
from scipy import sparse as scipy_sparse

from sfincs_jax.problems.profile_response.diagnostics import (
    fp_xblock_global_correction_metadata,
    fp_xblock_highx_residual_correction_metadata,
    sparse_rescue_tail_metadata,
    sparse_xblock_rescue_metadata,
    xblock_assembled_operator_diagnostics,
    xblock_coarse_correction_diagnostics,
    xblock_device_krylov_diagnostics,
    xblock_qi_deflated_preconditioner_diagnostics,
    xblock_qi_device_preconditioner_diagnostics,
    xblock_qi_seed_preconditioner_diagnostics,
    xblock_sparse_pc_core_diagnostics,
    xblock_side_probe_diagnostics,
)
from sfincs_jax.problems.profile_response.sparse_pc import (
    SparsePCGMRESContext,
    SparsePCPostMinresContext,
    XBlockAssembledPreflightError,
    apply_sparse_pc_post_minres,
    build_xblock_assembled_equilibration_setup,
    build_xblock_assembled_device_setup,
    build_xblock_assembled_matvec_setup,
    build_xblock_assembled_operator_preflight_setup,
    build_xblock_krylov_matvec_setup,
    evaluate_xblock_moment_schur_probe_result,
    failed_xblock_global_coupling_metadata,
    failed_xblock_two_level_metadata,
    failed_xblock_moment_schur_metadata,
    finalize_xblock_global_coupling_metadata,
    finalize_xblock_two_level_metadata,
    finalize_xblock_moment_schur_metadata,
    prepare_xblock_initial_guess,
    resolve_sparse_pc_entry_policy,
    resolve_xblock_qi_device_admission_setup,
    resolve_xblock_qi_device_base_config_setup,
    resolve_xblock_qi_device_enrichment_config_setup,
    resolve_xblock_qi_device_multilevel_config_setup,
    resolve_xblock_qi_device_operator_reuse_setup,
    resolve_xblock_qi_galerkin_policy_setup,
    resolve_xblock_qi_seed_policy_setup,
    resolve_xblock_qi_two_level_policy_setup,
    resolve_xblock_global_coupling_policy_setup,
    resolve_xblock_moment_schur_policy_setup,
    resolve_xblock_seed_policy_setup,
    resolve_xblock_sparse_pc_setup,
    resolve_xblock_sparse_pc_side_policy_setup,
    resolve_xblock_two_level_policy_setup,
    run_sparse_pc_gmres_once,
    finalize_xblock_assembled_operator_metadata,
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


def _sparse_rescue_scope() -> dict[str, object]:
    return {
        "sparse_xblock_rescue_active": 1,
        "sparse_xblock_rescue_attempted": True,
        "sparse_xblock_rescue_built": False,
        "sparse_xblock_rescue_error": None,
        "sparse_xblock_rescue_reason": "seed_rejected",
        "sparse_xblock_rescue_assembled_host_fp": True,
        "sparse_xblock_rescue_preconditioner_xi": 2,
        "sparse_xblock_rescue_seed_residual": 1.25,
        "sparse_xblock_rescue_seed_improvement_ratio": 4.0,
        "sparse_xblock_rescue_seed_accept_ratio": 0.2,
        "sparse_xblock_rescue_seed_refine_steps": 3,
        "sparse_xblock_rescue_seed_refines_performed": 2,
        "sparse_xblock_rescue_candidate_residual": 0.5,
        "sparse_xblock_rescue_candidate_accepted": 1,
        "fp_xblock_global_correction_allowed": True,
        "fp_xblock_global_correction_attempted": True,
        "fp_xblock_global_correction_accepted": False,
        "fp_xblock_global_correction_reason": "no_improvement",
        "fp_xblock_global_correction_error": "none",
        "fp_xblock_global_correction_preconditioner": "xblock",
        "fp_xblock_global_correction_steps": 4,
        "fp_xblock_global_correction_accepted_steps": 1,
        "fp_xblock_global_correction_residual_before": 2.0,
        "fp_xblock_global_correction_residual_after": 1.5,
        "fp_xblock_global_correction_improvement_ratio": 1.25,
        "fp_xblock_global_correction_elapsed_s": 0.125,
        "fp_xblock_highx_residual_correction_allowed": True,
        "fp_xblock_highx_residual_correction_attempted": False,
        "fp_xblock_highx_residual_correction_accepted": False,
        "fp_xblock_highx_residual_correction_reason": "policy_guard",
        "fp_xblock_highx_residual_correction_error": None,
        "fp_xblock_highx_residual_correction_residual_before": None,
        "fp_xblock_highx_residual_correction_residual_after": None,
        "fp_xblock_highx_residual_correction_improvement_ratio": None,
        "fp_xblock_highx_residual_correction_elapsed_s": None,
        "fp_xblock_highx_residual_correction_direction_count": 0,
        "fp_xblock_highx_residual_correction_direction_names": ["a", "b"],
    }


def test_sparse_rescue_metadata_helpers_preserve_driver_keys_and_types() -> None:
    scope = _sparse_rescue_scope()

    sparse_meta = sparse_xblock_rescue_metadata(scope)
    global_meta = fp_xblock_global_correction_metadata(scope)
    highx_meta = fp_xblock_highx_residual_correction_metadata(scope)
    combined = sparse_rescue_tail_metadata(scope)

    assert sparse_meta["sparse_xblock_rescue_active"] is True
    assert sparse_meta["sparse_xblock_rescue_reason"] == "seed_rejected"
    assert sparse_meta["sparse_xblock_rescue_candidate_accepted"] is True
    assert global_meta["fp_xblock_global_correction_reason"] == "no_improvement"
    assert global_meta["fp_xblock_global_correction_accepted"] is False
    assert highx_meta["fp_xblock_highx_residual_correction_allowed"] is True
    assert highx_meta["fp_xblock_highx_residual_correction_direction_names"] == (
        "a",
        "b",
    )
    assert combined == {**sparse_meta, **global_meta, **highx_meta}


def _qi_device_preconditioner_scope() -> dict[str, object]:
    return {
        "qi_device_preconditioner_enabled": 1,
        "qi_device_preconditioner_built": True,
        "qi_device_preconditioner_used": False,
        "qi_device_preconditioner_used_in_krylov": True,
        "qi_device_preconditioner_reason": "probe_reject",
        "qi_device_preconditioner_rank": 7,
        "qi_device_preconditioner_candidate_count": 11,
        "qi_device_preconditioner_coarse_shape": (7, 7),
        "qi_device_preconditioner_operator_on_basis_shape": (13, 7),
        "qi_device_preconditioner_coarse_norm": 2.5,
        "qi_device_preconditioner_operator_on_basis_norm": 3.5,
        "qi_device_preconditioner_residual_before": 4.0,
        "qi_device_preconditioner_residual_after": 1.0,
        "qi_device_preconditioner_improvement_ratio": 4.0,
        "qi_device_preconditioner_setup_s": 0.25,
        "qi_device_preconditioner_min_improvement": 0.05,
        "qi_device_preconditioner_use_in_krylov": True,
        "qi_device_augmented_krylov_requested": True,
        "qi_device_augmented_krylov_used": False,
        "qi_device_augmented_krylov_rank": 3,
        "qi_device_augmented_krylov_reason": "seed_only",
        "qi_device_augmented_krylov_mode": "right",
        "qi_device_augmented_seed_requested": True,
        "qi_device_augmented_seed_available": True,
        "qi_device_augmented_seed_used": False,
        "qi_device_augmented_seed_rank": 2,
        "qi_device_augmented_seed_max_rank": 5,
        "qi_device_augmented_seed_reason": "accepted",
        "qi_device_augmented_seed_projection_residual": 1.0e-3,
        "qi_device_augmented_seed_labels": ("constant", "current"),
        "qi_device_stats": {"applies": 9},
        "qi_device_preconditioner_metadata": {
            "operator_krylov_enrichment_enabled": True,
            "multilevel_coarse_enabled": True,
            "global_moment_residual_equation_enabled": True,
            "global_moment_residual_equation_solver": "galerkin",
            "global_moment_residual_equation_rank": 4,
            "global_moment_residual_equation_candidate_count": 6,
            "global_moment_residual_equation_condition_estimate": 12.0,
            "phase_space_residual_equation_enabled": True,
            "phase_space_residual_equation_max_rank_requested": 8,
            "phase_space_residual_equation_solver": "action_lstsq",
            "phase_space_residual_equation_rank": 5,
            "phase_space_residual_equation_candidate_count": 10,
            "phase_space_residual_equation_stage_count": 1,
            "phase_space_residual_equation_condition_estimate": 6.0,
            "phase_space_residual_equation_include_global": True,
            "phase_space_residual_equation_trapped_boundary_fraction": 0.25,
            "residual_region_bounce_coarse_enabled": True,
            "residual_region_bounce_coarse_max_rank": 9,
            "residual_region_bounce_coarse_rank": 3,
            "residual_region_bounce_coarse_candidate_count": 4,
            "active_pattern_coarse_enabled": True,
            "active_pattern_coarse_max_rank_requested": 12,
            "active_pattern_coarse_candidate_count": 7,
            "coupled_residual_equation_enabled": True,
            "coupled_residual_equation_max_rank_requested": 14,
            "coupled_residual_equation_rank": 6,
            "coupled_residual_equation_candidate_count": 16,
            "coupled_residual_equation_source_stage_ranks": (1, 2, 3),
            "coupled_residual_equation_solver": "action_lstsq",
            "coupled_residual_equation_include_flat": True,
            "coupled_residual_equation_min_relative_improvement_requested": 0.1,
            "coupled_residual_equation_install_in_krylov_on_reject_requested": True,
            "seed_probe_accepted": False,
            "installed_in_krylov_after_seed_reject": True,
            "coupled_residual_equation_accepted": False,
            "coupled_residual_equation_reason": "insufficient_improvement",
            "block_schur_residual_enrichment_enabled": True,
        },
    }


def test_xblock_qi_device_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_device_preconditioner_diagnostics(
        _qi_device_preconditioner_scope()
    )

    assert metadata["xblock_qi_device_preconditioner_enabled"] is True
    assert metadata["xblock_qi_device_preconditioner_used"] is False
    assert metadata["xblock_qi_device_preconditioner_rank"] == 7
    assert metadata["xblock_qi_device_preconditioner_applies"] == 9
    assert metadata["xblock_qi_device_preconditioner_augmented_seed_labels"] == (
        "constant",
        "current",
    )
    assert metadata["xblock_qi_device_preconditioner_global_moment_residual_equation"] is True
    assert metadata["xblock_qi_device_preconditioner_global_moment_residual_equation_rank"] == 4
    assert metadata["xblock_qi_device_preconditioner_phase_space_residual_equation_max_rank"] == 8
    assert metadata["xblock_qi_device_preconditioner_residual_region_bounce_coarse_max_rank"] == 9
    assert metadata["xblock_qi_device_preconditioner_active_pattern_coarse_max_rank"] == 12
    assert metadata["xblock_qi_device_preconditioner_coupled_residual_equation_rank"] == 6
    assert (
        metadata[
            "xblock_qi_device_preconditioner_coupled_residual_equation_install_in_krylov_on_reject"
        ]
        is True
    )
    assert metadata["xblock_qi_device_preconditioner_block_schur_residual_enrichment"] is True


def _qi_deflated_preconditioner_scope() -> dict[str, object]:
    return {
        "qi_deflated_preconditioner_enabled": 1,
        "qi_deflated_preconditioner_built": True,
        "qi_deflated_preconditioner_used": False,
        "qi_deflated_preconditioner_used_in_krylov": True,
        "qi_deflated_preconditioner_reason": "seed_rejected",
        "qi_deflated_preconditioner_rank": 5,
        "qi_deflated_preconditioner_candidate_count": 8,
        "qi_deflated_preconditioner_residual_before": 3.0,
        "qi_deflated_preconditioner_residual_after": 1.5,
        "qi_deflated_preconditioner_improvement_ratio": 2.0,
        "qi_deflated_preconditioner_setup_s": 0.125,
        "qi_deflated_stats": {"applies": 4, "local_applies": 6},
        "qi_deflated_preconditioner_metadata": {
            "correction_cycles": 3,
            "seed_solver": "minres",
            "cycle_residual_history": (3.0, 2.0, 1.5),
            "cycle_coefficients": (0.25, 0.5),
        },
    }


def test_xblock_qi_deflated_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_deflated_preconditioner_diagnostics(
        _qi_deflated_preconditioner_scope()
    )

    assert metadata["xblock_qi_deflated_preconditioner_enabled"] is True
    assert metadata["xblock_qi_deflated_preconditioner_used"] is False
    assert metadata["xblock_qi_deflated_preconditioner_use_in_krylov"] is True
    assert metadata["xblock_qi_deflated_preconditioner_rank"] == 5
    assert metadata["xblock_qi_deflated_preconditioner_candidate_count"] == 8
    assert metadata["xblock_qi_deflated_preconditioner_setup_s"] == 0.125
    assert metadata["xblock_qi_deflated_preconditioner_applies"] == 4
    assert metadata["xblock_qi_deflated_preconditioner_local_applies"] == 6
    assert metadata["xblock_qi_deflated_preconditioner_cycles"] == 3
    assert metadata["xblock_qi_deflated_preconditioner_seed_solver"] == "minres"
    assert metadata["xblock_qi_deflated_preconditioner_cycle_residual_history"] == (
        3.0,
        2.0,
        1.5,
    )
    assert metadata["xblock_qi_deflated_preconditioner_cycle_coefficients"] == (
        0.25,
        0.5,
    )


def _xblock_side_probe_scope() -> dict[str, object]:
    return {
        "xblock_side_probe_enabled": 1,
        "xblock_side_probe_used": True,
        "xblock_side_probe_switched": False,
        "xblock_side_probe_switch_suppressed_by_global_coupling": True,
        "xblock_side_probe_switch_suppressed_by_explicit_side": False,
        "xblock_side_probe_physical_seed_preserved_after_switch": True,
        "xblock_side_probe_seed_used": 1,
        "xblock_side_probe_seed_residual_norm": 1.0e-4,
        "xblock_side_probe_initial_side": "left",
        "xblock_side_probe_selected_side": "right",
        "xblock_side_probe_initial_method": "gmres",
        "xblock_side_probe_selected_method": "lgmres",
        "xblock_side_probe_lgmres_rescue": True,
        "xblock_lgmres_rescue_maxiter_capped": 1,
        "xblock_lgmres_rescue_outer_k": 12,
        "xblock_side_probe_residual_norm": 2.0e-5,
        "xblock_side_probe_residual_ratio": 3.5,
        "xblock_side_probe_iterations": np.int64(9),
        "xblock_side_probe_matvecs": np.int64(14),
        "xblock_side_probe_s": np.float64(0.375),
    }


def test_xblock_side_probe_diagnostics_preserve_payload() -> None:
    metadata = xblock_side_probe_diagnostics(_xblock_side_probe_scope())

    assert metadata["xblock_side_probe_enabled"] is True
    assert metadata["xblock_side_probe_switched"] is False
    assert metadata["xblock_side_probe_switch_suppressed_by_global_coupling"] is True
    assert metadata["xblock_side_probe_physical_seed_preserved_after_switch"] is True
    assert metadata["xblock_side_probe_seed_used"] is True
    assert metadata["xblock_side_probe_initial_side"] == "left"
    assert metadata["xblock_side_probe_selected_side"] == "right"
    assert metadata["xblock_side_probe_selected_method"] == "lgmres"
    assert metadata["xblock_side_probe_lgmres_rescue"] is True
    assert metadata["xblock_lgmres_rescue_maxiter_capped"] is True
    assert metadata["xblock_lgmres_rescue_outer_k"] == 12
    assert metadata["xblock_side_probe_residual_ratio"] == 3.5
    assert metadata["xblock_side_probe_iterations"] == 9
    assert metadata["xblock_side_probe_matvecs"] == 14
    assert metadata["xblock_side_probe_s"] == 0.375


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


def test_xblock_sparse_pc_side_policy_parses_jax_factors_and_forces_fgmres_right_pc() -> None:
    def side_policy(**kwargs):
        assert kwargs["krylov_env_value"] == "fgmres_jax"
        assert kwargs["full_fp_3d_pc"] is True
        return SimpleNamespace(
            precondition_side="left",
            default_right_preconditioned=False,
            krylov_method="fgmres_jax",
            gmres_restart=33,
            restart_capped=True,
            ignored_krylov_env=True,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=7, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        xblock_krylov_env="fgmres_jax",
        pc_restart=50,
        pc_restart_env="50",
        tokamak_fp_er_pc=False,
        active_size=4000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT": "compact-csr",
            "SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY": "jacobi",
        },
    )

    assert setup.xblock_jax_factors
    assert not setup.xblock_jax_factors_requested
    assert setup.xblock_device_krylov_forced_jax_factors
    assert setup.xblock_jax_factor_format == "csr"
    assert setup.xblock_jax_factor_apply == "diagonal"
    assert setup.precondition_side == "right"
    assert setup.xblock_device_fgmres_forced_right_pc
    assert setup.pc_restart == 33
    assert setup.xblock_default_restart_capped
    assert any("ignoring unknown" in message for _, message in setup.messages)


def test_xblock_sparse_pc_side_policy_uses_host_factors_when_fallback_is_used() -> None:
    def side_policy(**kwargs):
        return SimpleNamespace(
            precondition_side="right",
            default_right_preconditioned=True,
            krylov_method=kwargs["krylov_env_value"],
            gmres_restart=kwargs["requested_restart"],
            restart_capped=False,
            ignored_krylov_env=False,
        )

    setup = resolve_xblock_sparse_pc_side_policy_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        xblock_device_krylov_requested=True,
        xblock_device_host_fallback_decision=SimpleNamespace(used=True),
        xblock_krylov_env="gmres",
        pc_restart=20,
        pc_restart_env="",
        tokamak_fp_er_pc=False,
        active_size=2000,
        use_dkes=False,
        include_xdot_sparse_pc=False,
        include_electric_field_xi_sparse_pc=False,
        resolve_xblock_policy=side_policy,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_JAX_FACTORS": "1"},
    )

    assert setup.xblock_jax_factors_requested
    assert not setup.xblock_jax_factors
    assert setup.xblock_jax_factor_format == "padded"
    assert setup.xblock_jax_factor_apply == "exact"
    assert any("requires host sparse factors" in message for _, message in setup.messages)


def test_xblock_qi_device_operator_reuse_setup_skips_local_factors() -> None:
    calls: list[dict[str, object]] = []

    def reuse_decision(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(skip_xblock_factors=True)

    setup = resolve_xblock_qi_device_operator_reuse_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=5, n_species=1),
        xblock_krylov_method="fgmres_jax",
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        qi_device_preconditioner_requested=True,
        qi_device_matrix_free_requested=True,
        qi_device_use_in_krylov_requested=True,
        precondition_side="right",
        xblock_jax_factors=True,
        xblock_device_krylov_forced_jax_factors=True,
        xblock_preconditioner_xi=3,
        reuse_decision=reuse_decision,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_QI_DEVICE_OPERATOR_REUSE": "auto"},
    )

    assert setup.skip_xblock_factors
    assert not setup.xblock_jax_factors
    assert not setup.xblock_device_krylov_forced_jax_factors
    assert calls[0]["env_value"] == "auto"
    assert calls[0]["requested_krylov_method"] == "fgmres_jax"
    assert any("skipping local x-block factors" in message for _, message in setup.messages)


def test_xblock_qi_device_operator_reuse_setup_reports_factor_build_route() -> None:
    def reuse_decision(**_kwargs):
        return SimpleNamespace(skip_xblock_factors=False)

    setup = resolve_xblock_qi_device_operator_reuse_setup(
        op=_op(fp=True, constraint_scheme=1, n_zeta=3, n_species=1),
        xblock_krylov_method="gmres_jax",
        xblock_device_host_fallback_decision=SimpleNamespace(used=False),
        qi_device_preconditioner_requested=False,
        qi_device_matrix_free_requested=False,
        qi_device_use_in_krylov_requested=False,
        precondition_side="right",
        xblock_jax_factors=True,
        xblock_device_krylov_forced_jax_factors=True,
        xblock_preconditioner_xi=1,
        reuse_decision=reuse_decision,
        env={},
    )

    assert not setup.skip_xblock_factors
    assert setup.xblock_jax_factors
    assert setup.factor_backend == "jax"
    assert setup.factor_reason == " device-krylov"
    assert any("building jax x-block preconditioner" in message for _, message in setup.messages)


def test_xblock_krylov_matvec_setup_reduces_active_dofs_and_counts_progress() -> None:
    messages: list[str] = []
    active_idx = jnp.asarray([0, 2], dtype=jnp.int32)
    full_to_active = jnp.asarray([0, -1, 1, -1], dtype=jnp.int32)
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=4),
        rhs=jnp.asarray([1.0, 2.0, 3.0, 4.0]),
        xblock_use_active_dof=True,
        active_idx=active_idx,
        full_to_active=full_to_active,
        reduce_full_with_indices=lambda v, idx: v[idx],
        expand_reduced_with_map=lambda v, fmap: jnp.where(fmap >= 0, v[jnp.maximum(fmap, 0)], 0.0),
        operator_matvec=lambda v: 2.0 * v,
        elapsed_s=lambda: 12.5,
        emit=lambda _level, msg: messages.append(msg),
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "2"},
    )

    assert setup.xblock_linear_size == 2
    assert setup.xblock_active_idx_np.tolist() == [0, 2]
    assert setup.xblock_rhs.tolist() == [1.0, 3.0]
    assert setup.matvec_no_count(jnp.asarray([5.0, 7.0])).tolist() == [10.0, 14.0]
    assert int(setup.mv_count) == 0
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [2.0, 4.0]
    assert int(setup.mv_count) == 1
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [4.0, 6.0]
    assert int(setup.mv_count) == 2
    assert any("active-DOF reduction" in message for _, message in setup.messages)
    assert any("matvecs=2" in message for message in messages)


def test_xblock_krylov_matvec_setup_full_space_is_identity_mapping() -> None:
    setup = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=3),
        rhs=jnp.asarray([1.0, 2.0, 3.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        operator_matvec=lambda v: v + 1.0,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={"SFINCS_JAX_SPARSE_PC_PROGRESS_EVERY": "bad"},
    )

    assert setup.progress_every == 25
    assert setup.xblock_linear_size == 3
    assert setup.xblock_active_idx_np is None
    assert setup.reduce_full(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.expand_reduced(jnp.asarray([1.0, 2.0])).tolist() == [1.0, 2.0]
    assert setup.matvec(jnp.asarray([1.0, 2.0, 3.0])).tolist() == [2.0, 3.0, 4.0]
    assert int(setup.mv_count) == 1


def test_xblock_assembled_equilibration_setup_builds_row_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, -1.0], [0.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 3.0,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE": "1"},
    )

    assert setup.row_enabled
    assert setup.row_built
    assert not setup.col_enabled
    assert not setup.col_built
    assert setup.row_metadata["norm"] == "linf"
    assert np.asarray(setup.row_scale).tolist() == pytest.approx([0.5, 0.25])
    assert np.asarray(setup.inv_row_scale).tolist() == pytest.approx([2.0, 4.0])
    assert any("assembled row equilibration built" in message for _, message in setup.messages)


def test_xblock_assembled_equilibration_setup_builds_row_and_column_scales() -> None:
    matrix = scipy_sparse.csr_matrix([[2.0, 0.0], [1.0, 4.0]])
    setup = build_xblock_assembled_equilibration_setup(
        assembled_matrix=matrix,
        xblock_linear_size=2,
        elapsed_s=lambda: 5.0,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM": "l1",
        },
    )

    assert setup.row_enabled
    assert setup.row_built
    assert setup.col_enabled
    assert setup.col_built
    assert setup.row_metadata["norm"] == "l1"
    assert setup.col_metadata["norm"] == "l1"
    assert setup.row_metadata["column_equilibration"] is True
    assert np.all(np.isfinite(np.asarray(setup.col_scale)))
    assert np.all(np.asarray(setup.col_scale) > 0.0)
    assert any("assembled column equilibration built" in message for _, message in setup.messages)


def test_xblock_assembled_operator_preflight_uses_full_pattern_when_under_budget() -> None:
    full_pattern = object()
    full_summary = SimpleNamespace(nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0)

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=None,
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres_jax",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: full_pattern,
        active_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
        summarize_pattern=lambda _op, pattern: full_summary if pattern is full_pattern else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is full_pattern
    assert setup.summary is full_summary
    assert setup.device_enabled
    assert not setup.metadata["preflight_rejected"]
    assert setup.metadata["preflight_scope"] == "full"


def test_xblock_assembled_operator_preflight_uses_active_pattern_scope() -> None:
    full_summary = SimpleNamespace(nnz=1000, shape=(100, 100), max_row_nnz=20, avg_row_nnz=10.0)
    active_summary = SimpleNamespace(nnz=4, shape=(2, 2), max_row_nnz=2, avg_row_nnz=2.0)
    active_pattern = object()

    setup = build_xblock_assembled_operator_preflight_setup(
        op=SimpleNamespace(),
        xblock_active_idx_np=np.asarray([0, 2], dtype=np.int32),
        sparse_pc_fp_dense_velocity_block=False,
        xblock_krylov_method="gmres",
        estimate_summary=lambda *_args, **_kwargs: full_summary,
        full_pattern=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("unused")),
        active_pattern=lambda *_args, **_kwargs: active_pattern,
        summarize_pattern=lambda _op, pattern: active_summary if pattern is active_pattern else None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "1"},
    )

    assert setup.pattern is active_pattern
    assert setup.summary is active_summary
    assert setup.metadata["preflight_scope"] == "active_dof"
    assert setup.metadata["preflight_active_pattern_nnz_estimate"] == 4
    assert not setup.device_enabled


def test_xblock_assembled_operator_preflight_rejection_carries_metadata() -> None:
    summary = SimpleNamespace(nnz=10, shape=(3, 3), max_row_nnz=4, avg_row_nnz=3.0)
    with pytest.raises(XBlockAssembledPreflightError) as excinfo:
        build_xblock_assembled_operator_preflight_setup(
            op=SimpleNamespace(),
            xblock_active_idx_np=None,
            sparse_pc_fp_dense_velocity_block=False,
            xblock_krylov_method="gmres",
            estimate_summary=lambda *_args, **_kwargs: summary,
            full_pattern=lambda *_args, **_kwargs: object(),
            active_pattern=lambda *_args, **_kwargs: object(),
            summarize_pattern=lambda _op, _pattern: summary,
            env={"SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB": "0"},
        )

    assert excinfo.value.metadata["preflight_rejected"] is True
    assert excinfo.value.metadata["preflight_pattern_nnz_estimate"] == 10
    assert "non-positive CSR memory budget" in str(excinfo.value)


def test_xblock_assembled_device_setup_builds_and_validates_operator() -> None:
    device_operator = SimpleNamespace(nnz=2, nbytes_estimate=64)

    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1024,
        device_enabled=True,
        device_required=False,
        validation_samples=2,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: device_operator,
        validate_device_csr_matvec=lambda *_args, **_kwargs: (0.0, 1.0e-12),
    )

    assert setup.device_operator is device_operator
    assert setup.device_resident
    assert setup.validation_errors == (0.0, 1.0e-12)
    assert setup.error is None


def test_xblock_assembled_device_setup_optional_failure_returns_message() -> None:
    setup = build_xblock_assembled_device_setup(
        assembled_matrix=object(),
        assembled_matvec=lambda x: x,
        csr_cap_nbytes=1,
        device_enabled=True,
        device_required=False,
        validation_samples=1,
        validation_tol=1.0e-8,
        device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("too large")),
        validate_device_csr_matvec=lambda *_args, **_kwargs: (),
    )

    assert setup.device_operator is None
    assert not setup.device_resident
    assert "MemoryError" in str(setup.error)
    assert any("disabled after build failure" in message for _, message in setup.messages)


def test_xblock_assembled_device_setup_required_failure_raises() -> None:
    with pytest.raises(RuntimeError, match="device CSR operator failed"):
        build_xblock_assembled_device_setup(
            assembled_matrix=object(),
            assembled_matvec=lambda x: x,
            csr_cap_nbytes=1,
            device_enabled=True,
            device_required=True,
            validation_samples=1,
            validation_tol=1.0e-8,
            device_csr_from_matrix=lambda *_args, **_kwargs: (_ for _ in ()).throw(MemoryError("too large")),
            validate_device_csr_matvec=lambda *_args, **_kwargs: (),
        )


def test_xblock_assembled_matvec_setup_host_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda x: 3.0 * x,
        device_operator=None,
        mv_count=counter,
        progress_every=2,
        elapsed_s=lambda: 4.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "host"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [3.0, 6.0]
    assert setup.matvec(jnp.asarray([2.0, 3.0])).tolist() == [6.0, 9.0]
    assert int(counter) == 2
    assert any("assembled_host_matvecs=2" in message for message in messages)


def test_xblock_assembled_matvec_setup_device_counts_progress() -> None:
    messages: list[str] = []
    counter = build_xblock_krylov_matvec_setup(
        op=SimpleNamespace(total_size=2),
        rhs=jnp.asarray([0.0, 0.0]),
        xblock_use_active_dof=False,
        active_idx=None,
        full_to_active=None,
        reduce_full_with_indices=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        expand_reduced_with_map=lambda _v, _idx: (_ for _ in ()).throw(AssertionError("unused")),
        operator_matvec=lambda v: v,
        elapsed_s=lambda: 0.0,
        emit=None,
        env={},
    ).mv_count
    device_operator = SimpleNamespace(jitted_matvec=lambda: (lambda v: 5.0 * v))
    setup = build_xblock_assembled_matvec_setup(
        assembled_matvec=lambda _x: (_ for _ in ()).throw(AssertionError("unused")),
        device_operator=device_operator,
        mv_count=counter,
        progress_every=1,
        elapsed_s=lambda: 7.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert setup.location == "device"
    assert setup.matvec(jnp.asarray([1.0, 2.0])).tolist() == [5.0, 10.0]
    assert int(counter) == 1
    assert any("assembled_device_matvecs=1" in message for message in messages)


def test_finalize_xblock_assembled_operator_metadata_normalizes_fields() -> None:
    metadata = finalize_xblock_assembled_operator_metadata(
        metadata={"preflight_scope": "full"},
        setup_s=1.25,
        assembled_matrix=scipy_sparse.csr_matrix([[1.0, 0.0], [2.0, 3.0]]),
        assembled_summary=SimpleNamespace(nnz=3, avg_row_nnz=1.5, max_row_nnz=2),
        assembled_bundle_metadata=SimpleNamespace(
            storage_kind="csr",
            reason="materialized",
            csr_nbytes_estimate=128,
        ),
        max_colors=4,
        validation_errors=(1.0e-12,),
        device_enabled=True,
        device_required=False,
        device_resident=True,
        device_operator=SimpleNamespace(nnz=3, nbytes_estimate=96),
        device_validation_errors=(2.0e-12,),
        device_error=None,
    )

    assert metadata["preflight_scope"] == "full"
    assert metadata["matrix_nnz"] == 3
    assert metadata["pattern_avg_row_nnz"] == pytest.approx(1.5)
    assert metadata["device_nnz"] == 3
    assert metadata["device_validation_rel_errors"] == (2.0e-12,)


def test_xblock_assembled_operator_diagnostics_preserve_payload() -> None:
    metadata = xblock_assembled_operator_diagnostics(
        {
            "assembled_operator_enabled": 1,
            "assembled_operator_built": True,
            "assembled_operator_metadata": {
                "active_dof": True,
                "preflight_scope": "active",
                "setup_s": 1.25,
                "preflight_rejected": False,
                "preflight_pattern_nnz_estimate": 50,
                "preflight_peak_nbytes_estimate": 2048,
                "preflight_full_csr_nbytes_estimate": 4096,
                "preflight_active_csr_nbytes_estimate": 1024,
                "pattern_nnz": 45,
                "matrix_nnz": 43,
                "csr_nbytes_estimate": 512,
                "device_enabled": True,
                "device_required": False,
                "device_resident": True,
                "device_nnz": 43,
                "device_csr_nbytes_estimate": 400,
                "device_validation_rel_errors": (1.0e-12,),
                "device_error": None,
                "max_colors": 7,
                "validation_rel_errors": (2.0e-12,),
                "error": None,
            },
            "xblock_row_equilibration_enabled": 1,
            "xblock_row_equilibration_built": True,
            "xblock_row_equilibration_metadata": {
                "norm": "inf",
                "setup_s": 0.1,
                "zero_or_tiny_rows": 2,
                "row_norm_min": 0.5,
                "row_norm_max": 4.0,
                "row_scale_min": 0.25,
                "row_scale_max": 2.0,
            },
            "xblock_col_equilibration_enabled": 1,
            "xblock_col_equilibration_built": False,
            "xblock_col_equilibration_metadata": {
                "norm": "two",
                "setup_s": 0.2,
                "zero_or_tiny_columns": 3,
                "col_norm_min": 0.25,
                "col_norm_max": 5.0,
                "col_scale_min": 0.2,
                "col_scale_max": 4.0,
            },
        }
    )

    assert metadata["xblock_assembled_operator_enabled"] is True
    assert metadata["xblock_assembled_operator_built"] is True
    assert metadata["xblock_assembled_operator_active_dof"] is True
    assert metadata["xblock_assembled_operator_preflight_scope"] == "active"
    assert metadata["xblock_assembled_operator_preflight_pattern_nnz_estimate"] == 50
    assert metadata["xblock_assembled_operator_matrix_nnz"] == 43
    assert metadata["xblock_assembled_operator_device_enabled"] is True
    assert metadata["xblock_assembled_operator_device_resident"] is True
    assert metadata["xblock_assembled_operator_device_validation_rel_errors"] == (
        1.0e-12,
    )
    assert metadata["xblock_assembled_operator_row_equilibration_enabled"] is True
    assert metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert metadata["xblock_assembled_operator_row_equilibration_norm"] == "inf"
    assert metadata["xblock_assembled_operator_row_equilibration_zero_or_tiny_rows"] == 2
    assert metadata["xblock_assembled_operator_col_equilibration_enabled"] is True
    assert metadata["xblock_assembled_operator_col_equilibration_built"] is False
    assert metadata["xblock_assembled_operator_col_equilibration_norm"] == "two"
    assert metadata["xblock_assembled_operator_col_equilibration_zero_or_tiny_columns"] == 3
    assert metadata["xblock_assembled_operator_max_colors"] == 7
    assert metadata["xblock_assembled_operator_validation_rel_errors"] == (2.0e-12,)


def test_xblock_coarse_correction_diagnostics_preserve_payload() -> None:
    metadata = xblock_coarse_correction_diagnostics(
        {
            "moment_schur_enabled": 1,
            "moment_schur_built": True,
            "moment_schur_used": False,
            "moment_schur_reason": "compact_factor_guard",
            "moment_schur_default_blocked_by_compact_factors": 1,
            "moment_schur_probe_residual_before": 2.0,
            "moment_schur_probe_residual_after": 0.5,
            "moment_schur_probe_improvement_ratio": 4.0,
            "moment_schur_metadata": {
                "mode": "constraint",
                "rank": 3,
                "extra_size": 2,
                "setup_s": 0.1,
                "expected_size": 5,
                "rcond": 1.0e-8,
                "singular_value_proxy": (1.0, 0.1),
                "device_resident": True,
                "error": None,
            },
            "moment_schur_stats": {"applies": 6, "base_applies": 4},
            "two_level_enabled": 1,
            "two_level_built": True,
            "two_level_metadata": {
                "mode": "seed",
                "basis_size": 4,
                "rank": 4,
                "setup_s": 0.2,
                "rcond": 1.0e-7,
                "basis_names": ("density", "flow"),
                "active_projected": True,
                "expected_size": 8,
                "error": None,
            },
            "two_level_stats": {"applies": 8, "coarse_applies": 3},
            "global_coupling_enabled": 1,
            "global_coupling_built": True,
            "global_coupling_metadata": {
                "mode": "load",
                "load_basis_size": 5,
                "basis_size": 7,
                "rank": 6,
                "setup_s": 0.3,
                "setup_budget_s": 2.0,
                "setup_budget_reached": False,
                "rcond": 1.0e-6,
                "coarse_solver": "pinv",
                "smoother": "block",
                "ridge": 1.0e-10,
                "singular_values": (3.0, 1.0),
                "device_resident": True,
                "fsavg_lmax": 2,
                "angular_lmax": 3,
                "basis_names": ("source", "constraint"),
                "error": None,
            },
            "global_coupling_stats": {"applies": 9, "coarse_applies": 5},
        }
    )

    assert metadata["xblock_moment_schur_enabled"] is True
    assert metadata["xblock_moment_schur_used"] is False
    assert metadata["xblock_moment_schur_default_blocked_by_compact_factors"] is True
    assert metadata["xblock_moment_schur_rank"] == 3
    assert metadata["xblock_moment_schur_device_resident"] is True
    assert metadata["xblock_moment_schur_applies"] == 6
    assert metadata["xblock_two_level_enabled"] is True
    assert metadata["xblock_two_level_basis_names"] == ("density", "flow")
    assert metadata["xblock_two_level_active_projected"] is True
    assert metadata["xblock_two_level_coarse_applies"] == 3
    assert metadata["xblock_global_coupling_enabled"] is True
    assert metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert metadata["xblock_global_coupling_coarse_solver"] == "pinv"
    assert metadata["xblock_global_coupling_singular_values"] == (3.0, 1.0)
    assert metadata["xblock_global_coupling_device_resident"] is True
    assert metadata["xblock_global_coupling_coarse_applies"] == 5


def test_xblock_qi_seed_preconditioner_diagnostics_preserve_payload() -> None:
    metadata = xblock_qi_seed_preconditioner_diagnostics(
        {
            "xblock_initial_seed_used": 1,
            "xblock_initial_seed_residual_norm": 1.0e-3,
            "xblock_initial_seed_residual_ratio": 0.5,
            "moment_schur_seed_enabled": 1,
            "moment_schur_seed_used": False,
            "moment_schur_seed_residual_norm": 2.0e-3,
            "moment_schur_seed_residual_ratio": 0.75,
            "qi_coarse_seed_enabled": 1,
            "qi_coarse_seed_used": True,
            "qi_coarse_seed_residual_before": 3.0,
            "qi_coarse_seed_residual_after": 1.0,
            "qi_coarse_seed_improvement_ratio": 3.0,
            "qi_coarse_seed_rank": 4,
            "qi_coarse_seed_candidate_count": 9,
            "qi_coarse_seed_reason": "accepted",
            "qi_coarse_seed_labels": ("flat", "current"),
            "qi_coarse_seed_s": 0.25,
            "qi_seed_basis_kind": "load",
            "qi_seed_max_candidates": 12,
            "qi_seed_max_angular_mode": 3,
            "qi_galerkin_preconditioner_enabled": 1,
            "qi_galerkin_preconditioner_built": True,
            "qi_galerkin_preconditioner_used": False,
            "qi_galerkin_preconditioner_reason": "probe_rejected",
            "qi_galerkin_preconditioner_mode": "coarse",
            "qi_galerkin_preconditioner_rank": 5,
            "qi_galerkin_preconditioner_candidate_count": 10,
            "qi_galerkin_preconditioner_coarse_shape": (5, 5),
            "qi_galerkin_preconditioner_coarse_norm": 2.5,
            "qi_galerkin_preconditioner_rcond": 1.0e-6,
            "qi_galerkin_preconditioner_damping": 1.0e-4,
            "qi_galerkin_preconditioner_basis_reused_from_seed": True,
            "qi_galerkin_preconditioner_residual_before": 4.0,
            "qi_galerkin_preconditioner_residual_after": 2.0,
            "qi_galerkin_preconditioner_improvement_ratio": 2.0,
            "qi_galerkin_preconditioner_probe_reduced": True,
            "qi_galerkin_preconditioner_probe_candidates": (0, 2),
            "qi_galerkin_preconditioner_selected_index": 1,
            "qi_galerkin_preconditioner_setup_s": 0.5,
            "qi_galerkin_stats": {
                "applies": 6,
                "coarse_applies": 4,
                "base_applies": 2,
            },
            "qi_two_level_preconditioner_enabled": 1,
            "qi_two_level_preconditioner_built": True,
            "qi_two_level_preconditioner_used": True,
            "qi_two_level_preconditioner_reason": "accepted",
            "qi_two_level_preconditioner_rank": 6,
            "qi_two_level_preconditioner_candidate_count": 11,
            "qi_two_level_preconditioner_coarse_shape": (6, 6),
            "qi_two_level_preconditioner_coarse_norm": 3.5,
            "qi_two_level_preconditioner_operator_on_basis_shape": (20, 6),
            "qi_two_level_preconditioner_operator_on_basis_norm": 4.5,
            "qi_two_level_preconditioner_coarse_solver": "pinv",
            "qi_two_level_preconditioner_residual_augmented": True,
            "qi_two_level_preconditioner_rank_before_augmentation": 4,
            "qi_two_level_preconditioner_augmentation_labels": ("r0", "r1"),
            "qi_two_level_preconditioner_residual_augment_max_extra": 3,
            "qi_two_level_preconditioner_residual_augment_steps": 2,
            "qi_two_level_preconditioner_residual_augment_include_residuals": True,
            "qi_two_level_preconditioner_smoothed_load_basis": True,
            "qi_two_level_preconditioner_smoothed_load_metadata": {"rank": 2},
            "qi_two_level_preconditioner_rcond": 1.0e-7,
            "qi_two_level_preconditioner_damping": 1.0e-5,
            "qi_two_level_preconditioner_basis_reused_from_seed": True,
            "qi_two_level_preconditioner_residual_before": 5.0,
            "qi_two_level_preconditioner_residual_after": 1.0,
            "qi_two_level_preconditioner_improvement_ratio": 5.0,
            "qi_two_level_preconditioner_probe_candidates": (1, 3),
            "qi_two_level_preconditioner_selected_index": 0,
            "qi_two_level_preconditioner_setup_s": 0.75,
            "qi_two_level_stats": {"applies": 8, "local_applies": 5},
        }
    )

    assert metadata["xblock_initial_seed_used"] is True
    assert metadata["xblock_moment_schur_seed_used"] is False
    assert metadata["xblock_qi_coarse_seed_used"] is True
    assert metadata["xblock_qi_coarse_seed_rank"] == 4
    assert metadata["xblock_qi_coarse_seed_labels"] == ("flat", "current")
    assert metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert metadata["xblock_qi_galerkin_preconditioner_used"] is False
    assert metadata["xblock_qi_galerkin_preconditioner_rank"] == 5
    assert metadata["xblock_qi_galerkin_preconditioner_applies"] == 6
    assert metadata["xblock_qi_galerkin_preconditioner_base_applies"] == 2
    assert metadata["xblock_qi_two_level_preconditioner_enabled"] is True
    assert metadata["xblock_qi_two_level_preconditioner_used"] is True
    assert metadata["xblock_qi_two_level_preconditioner_rank"] == 6
    assert metadata["xblock_qi_two_level_preconditioner_residual_augmented"] is True
    assert metadata["xblock_qi_two_level_preconditioner_augmentation_labels"] == (
        "r0",
        "r1",
    )
    assert metadata["xblock_qi_two_level_preconditioner_smoothed_load_metadata"] == {
        "rank": 2
    }
    assert metadata["xblock_qi_two_level_preconditioner_local_applies"] == 5


def test_xblock_device_krylov_diagnostics_preserve_transfer_free_logic() -> None:
    fallback = SimpleNamespace(
        mode="auto",
        used=False,
        reason="device_ok",
        requested_method="fgmres_jax",
        effective_krylov_env_value="fgmres_jax",
        min_active_size=128,
        qi_like_full_fp_3d=True,
        ignored_env=False,
        non_autodiff=False,
    )
    operator_reuse = SimpleNamespace(
        enabled=True,
        reason="shape_reusable",
        skip_xblock_factors=True,
        to_metadata=lambda: {"enabled": True, "reason": "shape_reusable"},
    )

    metadata = xblock_device_krylov_diagnostics(
        {
            "xblock_krylov_method": "fgmres_jax",
            "xblock_device_krylov_methods": {"gmres_jax", "fgmres_jax"},
            "xblock_jax_factors": True,
            "assembled_operator_built": True,
            "assembled_operator_device_resident": True,
            "two_level_built": False,
            "global_coupling_built": True,
            "global_coupling_metadata": {"device_resident": True},
            "xblock_device_host_fallback_decision": fallback,
            "xblock_krylov_env_requested": "auto",
            "xblock_device_host_fallback_auto_disabled_by_qi_device": True,
            "xblock_qi_device_operator_reuse_decision": operator_reuse,
            "xblock_device_fgmres_jit": True,
            "xblock_device_fgmres_jit_mode": "cycle",
            "xblock_device_fgmres_jit_outer_k": 7,
            "qi_device_augmented_krylov_requested": True,
            "qi_device_augmented_krylov_used": False,
            "qi_device_augmented_krylov_rank": 3,
            "qi_device_augmented_krylov_reason": "seed_only",
            "qi_device_augmented_krylov_mode": "right",
            "qi_device_augmented_seed_requested": True,
            "qi_device_augmented_seed_available": True,
            "qi_device_augmented_seed_used": False,
            "qi_device_augmented_seed_rank": 2,
            "qi_device_augmented_seed_max_rank": 5,
            "qi_device_augmented_seed_reason": "accepted",
            "qi_device_augmented_seed_projection_residual": 1.0e-4,
            "qi_device_augmented_seed_labels": ("constant", "current"),
            "tfqmr_replacement_interval": 11,
            "xblock_device_krylov_forced_jax_factors": True,
            "xblock_device_fgmres_forced_right_pc": True,
            "fgmres_block_between_cycles": True,
            "xblock_estimated_gmres_basis_nbytes": 100,
            "xblock_estimated_bicgstab_work_nbytes": 200,
            "xblock_estimated_tfqmr_work_nbytes": 300,
        }
    )

    assert metadata["xblock_device_krylov_method"] == "fgmres_jax"
    assert metadata["xblock_device_host_fallback_mode"] == "auto"
    assert metadata["xblock_device_host_fallback_used"] is False
    assert metadata["xblock_device_host_fallback_auto_disabled_by_qi_device"] is True
    assert metadata["xblock_qi_device_operator_reuse"] == {
        "enabled": True,
        "reason": "shape_reusable",
    }
    assert metadata["xblock_qi_device_operator_reuse_skip_xblock_factors"] is True
    assert metadata["xblock_device_fgmres_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_enabled"] is True
    assert metadata["xblock_device_fgmres_jit_mode"] == "cycle"
    assert metadata["xblock_device_fgmres_jit_outer_k"] == 7
    assert metadata["xblock_device_fgmres_qi_augmented_seed_labels"] == (
        "constant",
        "current",
    )
    assert metadata["xblock_device_tfqmr_replacement_interval"] == 11
    assert metadata["xblock_device_fgmres_forced_jax_factors"] is True
    assert metadata["xblock_estimated_tfqmr_work_nbytes"] == 300
    assert metadata["xblock_device_krylov_host_transfer_free"] is True
    assert metadata["xblock_device_fgmres_host_transfer_free"] is True
    assert metadata["xblock_device_bicgstab_host_transfer_free"] is False
    assert metadata["xblock_device_tfqmr_host_transfer_free"] is False


def test_xblock_sparse_pc_core_diagnostics_preserve_payload() -> None:
    timer = SimpleNamespace(elapsed_s=lambda: 12.5)
    metadata = xblock_sparse_pc_core_diagnostics(
        {
            "xblock_solver_kind": "sparse_pc_gmres",
            "accepted_converged_xblock": True,
            "reported_iterations": np.int64(12),
            "reported_matvecs": np.int64(34),
            "mv_count": np.int64(56),
            "device_krylov_estimated_matvecs": np.int64(78),
            "xblock_krylov_method": "fgmres_jax",
            "candidate_krylov_method": "gmres_jax",
            "candidate_iterations": 9,
            "candidate_matvecs": 10,
            "candidate_residual_norm": 1.0e-5,
            "fallback_started_from_candidate": True,
            "fallback_candidate_improved_rhs": False,
            "precondition_side": "right",
            "xblock_default_right_pc": True,
            "xblock_default_restart_capped": False,
            "pc_restart": 40,
            "pc_maxiter": 80,
            "setup_s": 0.5,
            "solve_s": 1.5,
            "sparse_timer": timer,
            "pc_factor_s": 0.25,
            "xblock_preconditioner_xi": 3,
            "xblock_preconditioner_built": True,
            "xblock_assembled_host_fp": False,
            "xblock_jax_factors": True,
            "xblock_jax_factor_format": "padded",
            "xblock_jax_factor_apply": "vmap",
            "xblock_lower_fill_mode": "probe",
            "xblock_lower_fill_ignored_env": True,
        }
    )

    assert metadata["solver_kind"] == "sparse_pc_gmres"
    assert metadata["residual_kind"] == "true_residual"
    assert metadata["accepted_converged"] is True
    assert metadata["iterations"] == 12
    assert metadata["python_matvecs"] == 56
    assert metadata["device_cycle_estimated_matvecs"] == 78
    assert metadata["fallback_from_krylov_method"] == "gmres_jax"
    assert metadata["fallback_started_from_candidate"] is True
    assert metadata["precondition_side"] == "right"
    assert metadata["elapsed_s"] == 12.5
    assert metadata["sparse_pc_xblock_preconditioner_xi"] == 3
    assert metadata["sparse_pc_xblock_jax_factor_format"] == "padded"
    assert metadata["xblock_lower_fill_requested"] is True
    assert metadata["xblock_lower_fill_ignored_env"] is True


def test_xblock_moment_schur_policy_defaults_on_for_constraint1_device_krylov() -> None:
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="right",
        env={},
    )

    assert setup.default_candidate
    assert setup.enabled
    assert not setup.default_blocked_by_compact_factors
    assert any("moment-Schur build start" in message for _, message in setup.messages)


def test_xblock_moment_schur_policy_blocks_compact_csr_default_but_allows_force() -> None:
    op = SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0)
    blocked = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={},
    )
    forced = resolve_xblock_moment_schur_policy_setup(
        op=op,
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=True,
        xblock_jax_factor_format="csr",
        precondition_side="right",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT": "0.25",
        },
    )

    assert blocked.default_blocked_by_compact_factors
    assert not blocked.enabled
    assert any("default disabled" in message for _, message in blocked.messages)
    assert forced.enabled
    assert forced.rcond == pytest.approx(1.0e-9)
    assert forced.probe_enabled
    assert forced.probe_min_improvement == pytest.approx(0.25)


def test_xblock_moment_schur_policy_does_not_emit_build_for_no_preconditioner_side() -> None:
    setup = resolve_xblock_moment_schur_policy_setup(
        op=SimpleNamespace(rhs_mode=1, constraint_scheme=1, extra_size=2, phi1_size=0),
        xblock_krylov_method="gmres_jax",
        xblock_jax_factors=False,
        xblock_jax_factor_format="padded",
        precondition_side="none",
        env={},
    )

    assert setup.enabled
    assert not setup.messages


def test_xblock_moment_schur_probe_result_accepts_sufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=7.0,
        min_improvement=0.2,
    )

    assert result.used
    assert result.reason == "probe_reduced"
    assert result.improvement_ratio == pytest.approx(0.7)
    assert any("accepted" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_rejects_insufficient_reduction() -> None:
    result = evaluate_xblock_moment_schur_probe_result(
        residual_before=10.0,
        residual_after=9.0,
        min_improvement=0.2,
    )

    assert not result.used
    assert result.reason == "probe_not_reduced"
    assert result.improvement_ratio == pytest.approx(0.9)
    assert any("rejected" in message for _, message in result.messages)


def test_xblock_moment_schur_probe_result_handles_zero_rhs_norm() -> None:
    zero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=0.0,
        min_improvement=0.5,
    )
    nonzero = evaluate_xblock_moment_schur_probe_result(
        residual_before=0.0,
        residual_after=1.0,
        min_improvement=0.5,
    )

    assert zero.used
    assert zero.improvement_ratio == 0.0
    assert not nonzero.used
    assert np.isinf(nonzero.improvement_ratio)


def test_xblock_moment_schur_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_moment_schur_metadata(
        metadata={"rank": 3},
        setup_s=1.5,
    )
    failure = failed_xblock_moment_schur_metadata(
        exc=ValueError("bad factor"),
        setup_s=2.5,
    )

    assert success == {"rank": 3, "setup_s": 1.5}
    assert failure["setup_s"] == 2.5
    assert failure["error"] == "ValueError: bad factor"


def test_xblock_two_level_policy_defaults_off_and_honors_disabled_side() -> None:
    off = resolve_xblock_two_level_policy_setup(precondition_side="right", env={})
    no_side = resolve_xblock_two_level_policy_setup(
        precondition_side="none",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert off.mode == "additive"
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_two_level_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_two_level_policy_setup(
        precondition_side="left",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_EXTRA_UNITS": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_INCLUDE_RHS": "0",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 7
    assert setup.fsavg_lmax == 3
    assert setup.max_extra_units == 2
    assert setup.rcond == pytest.approx(1.0e-8)
    assert not setup.include_rhs


def test_xblock_two_level_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_two_level_metadata(metadata={"mode": "additive"}, setup_s=0.25)
    failure = failed_xblock_two_level_metadata(exc=RuntimeError("bad coarse"), setup_s=0.5)

    assert success == {"mode": "additive", "setup_s": 0.25}
    assert failure == {"error": "RuntimeError: bad coarse", "setup_s": 0.5}


def test_xblock_global_coupling_policy_defaults_off_and_selects_builder_defaults() -> None:
    off = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres",
        env={},
    )
    device = resolve_xblock_global_coupling_policy_setup(
        precondition_side="right",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )
    no_side = resolve_xblock_global_coupling_policy_setup(
        precondition_side="none",
        xblock_krylov_method="gmres_jax",
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert not off.use_device_builder
    assert off.setup_max_s == 0.0
    assert device.enabled
    assert device.should_build
    assert device.use_device_builder
    assert device.setup_max_s == pytest.approx(180.0)
    assert no_side.enabled
    assert not no_side.should_build


def test_xblock_global_coupling_policy_parses_build_parameters() -> None:
    setup = resolve_xblock_global_coupling_policy_setup(
        precondition_side="left",
        xblock_krylov_method="bicgstab_jax",
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS": "9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_EXTRA_UNITS": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_RCOND": "1e-7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_INCLUDE_RHS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S": "12.5",
        },
    )

    assert setup.enabled
    assert setup.should_build
    assert setup.use_device_builder
    assert setup.mode == "multiplicative"
    assert setup.max_directions == 9
    assert setup.fsavg_lmax == 3
    assert setup.angular_lmax == 4
    assert setup.max_extra_units == 5
    assert setup.rcond == pytest.approx(1.0e-7)
    assert not setup.include_rhs
    assert setup.setup_max_s == pytest.approx(12.5)


def test_xblock_global_coupling_metadata_helpers_normalize_success_and_failure() -> None:
    success = finalize_xblock_global_coupling_metadata(metadata={"mode": "additive"}, setup_s=0.75)
    failure = failed_xblock_global_coupling_metadata(exc=RuntimeError("timeout"), setup_s=1.5)

    assert success == {"mode": "additive", "setup_s": 0.75}
    assert failure == {"error": "RuntimeError: timeout", "setup_s": 1.5}


def test_prepare_xblock_initial_guess_accepts_reduced_and_full_active_shapes() -> None:
    reduced = jnp.asarray([1.0, 2.0])
    full = jnp.asarray([10.0, 11.0, 12.0, 13.0])
    rhs_reduced = jnp.zeros(2)
    rhs_full = jnp.zeros(4)

    reduced_result = prepare_xblock_initial_guess(
        x0=reduced,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )
    full_result = prepare_xblock_initial_guess(
        x0=full,
        xblock_rhs=rhs_reduced,
        full_rhs=rhs_full,
        xblock_use_active_dof=True,
        reduce_full=lambda v: v[jnp.asarray([0, 2])],
    )

    assert reduced_result.messages == ()
    assert jnp.asarray(reduced_result.x0_full).tolist() == [1.0, 2.0]
    assert full_result.messages == ()
    assert jnp.asarray(full_result.x0_full).tolist() == [10.0, 12.0]


def test_prepare_xblock_initial_guess_rejects_incompatible_shape_with_message() -> None:
    result = prepare_xblock_initial_guess(
        x0=jnp.ones(3),
        xblock_rhs=jnp.zeros(2),
        full_rhs=jnp.zeros(4),
        xblock_use_active_dof=True,
        reduce_full=lambda v: v,
    )

    assert result.x0_full is None
    assert len(result.messages) == 1
    assert "ignoring incompatible x0 shape=(3,)" in result.messages[0][1]
    assert "expected=(2,) or (4,)" in result.messages[0][1]


def test_xblock_seed_policy_defaults_and_env_overrides() -> None:
    default = resolve_xblock_seed_policy_setup(moment_schur_used=True, env={})
    disabled = resolve_xblock_seed_policy_setup(
        moment_schur_used=True,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED": "0",
        },
    )

    assert not default.initial_seed_enabled
    assert default.moment_schur_seed_enabled
    assert disabled.initial_seed_enabled
    assert not disabled.moment_schur_seed_enabled


def test_xblock_qi_seed_policy_defaults_off_without_shared_basis() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(env={})

    assert not setup.coarse_seed_enabled
    assert not setup.galerkin_preconditioner_enabled
    assert not setup.two_level_preconditioner_enabled
    assert not setup.device_preconditioner_enabled
    assert not setup.deflated_preconditioner_enabled
    assert not setup.shared_basis_required
    assert setup.max_rank == 0
    assert setup.basis_kind is None


def test_xblock_qi_seed_policy_deflated_only_does_not_parse_shared_basis() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEFLATED_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK": "48",
        },
    )

    assert setup.deflated_preconditioner_enabled
    assert not setup.shared_basis_required
    assert setup.max_rank == 0
    assert setup.max_candidates == 0


def test_xblock_qi_seed_policy_parses_shared_basis_parameters() -> None:
    setup = resolve_xblock_qi_seed_policy_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK": "10",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES": "24",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_ANGULAR_MODE": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RANK_RTOL": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MIN_IMPROVEMENT": "0.15",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_ANGULAR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_BLOCKS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_RADIAL_ANGULAR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_CONSTRAINT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_INCLUDE_SCHUR": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS": "Residual-Enriched",
        },
    )

    assert setup.coarse_seed_enabled
    assert setup.galerkin_preconditioner_enabled
    assert setup.two_level_preconditioner_enabled
    assert setup.device_preconditioner_enabled
    assert setup.shared_basis_required
    assert setup.max_rank == 10
    assert setup.max_candidates == 24
    assert setup.max_angular_mode == 4
    assert setup.rank_rtol == pytest.approx(1.0e-8)
    assert setup.min_improvement == pytest.approx(0.15)
    assert setup.rcond == pytest.approx(1.0e-9)
    assert not setup.include_angular
    assert not setup.include_blocks
    assert not setup.include_radial
    assert not setup.include_radial_angular
    assert not setup.include_constraint_moments
    assert not setup.include_schur
    assert setup.basis_kind == "residual_enriched"


def test_xblock_qi_galerkin_policy_handles_disabled_and_fallback_cases() -> None:
    def parse_modes(raw, *, default="auto"):
        return ("additive", "multiplicative") if (raw or default) == "auto" else (raw,)

    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0, 0.5, 0.25)):
        return tuple(auto_defaults) if not raw else tuple(float(v) for v in str(raw).split(","))

    off = resolve_xblock_qi_galerkin_policy_setup(
        enabled=False,
        host_fallback_used=False,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={},
    )
    fallback = resolve_xblock_qi_galerkin_policy_setup(
        enabled=True,
        host_fallback_used=True,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={},
    )

    assert not off.should_build
    assert off.reason is None
    assert fallback.enabled
    assert not fallback.should_build
    assert fallback.reason == "disabled_by_device_host_fallback"
    assert any("device-host fallback" in message for _level, message in fallback.messages)


def test_xblock_qi_galerkin_policy_parses_build_parameters() -> None:
    def parse_modes(raw, *, default="auto"):
        return tuple(str(raw or default).split(","))

    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0, 0.5, 0.25)):
        return tuple(auto_defaults) if not raw else tuple(float(v) for v in str(raw).split(","))

    setup = resolve_xblock_qi_galerkin_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="right",
        parse_modes=parse_modes,
        parse_dampings=parse_dampings,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_MODE": "multiplicative",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPING": "0.6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPINGS": "0.6,0.3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_PROBE": "0",
        },
    )

    assert setup.should_build
    assert setup.preconditioner_mode == "multiplicative"
    assert setup.candidate_modes == ("multiplicative",)
    assert setup.rcond == pytest.approx(1.0e-8)
    assert setup.damping == pytest.approx(0.6)
    assert setup.candidate_dampings == (0.6, 0.3)
    assert not setup.probe_enabled


def test_xblock_qi_two_level_policy_handles_disabled_and_side_none_cases() -> None:
    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0,)):
        return tuple(auto_defaults) if not raw else tuple(float(v) for v in str(raw).split(","))

    off = resolve_xblock_qi_two_level_policy_setup(
        enabled=False,
        host_fallback_used=False,
        precondition_side="right",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={},
    )
    side_none = resolve_xblock_qi_two_level_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="none",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={},
    )

    assert not off.should_build
    assert off.smoothed_load_max_rank == 8
    assert not side_none.should_build
    assert side_none.reason == "disabled_by_precondition_side_none"


def test_xblock_qi_two_level_policy_parses_build_parameters() -> None:
    def parse_dampings(raw, *, default=1.0, auto_defaults=(1.0,)):
        return tuple(auto_defaults) if not raw else tuple(float(v) for v in str(raw).split(","))

    setup = resolve_xblock_qi_two_level_policy_setup(
        enabled=True,
        host_fallback_used=False,
        precondition_side="right",
        seed_max_rank=8,
        parse_dampings=parse_dampings,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RCOND": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPING": "0.7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_DAMPINGS": "0.7,0.35",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_MIN_IMPROVEMENT": "0.2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_COARSE_SOLVER": "Action-Lstsq",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_MAX_EXTRA": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_STEPS": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_INCLUDE_RESIDUALS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS_COMBINE": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_DIRECTIONS": "12",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_RANK": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_FSAVG_LMAX": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_ANGULAR_LMAX": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_EXTRA_UNITS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_INCLUDE_RHS": "0",
        },
    )

    assert setup.should_build
    assert setup.rcond == pytest.approx(1.0e-9)
    assert setup.damping == pytest.approx(0.7)
    assert setup.candidate_dampings == (0.7, 0.35)
    assert setup.min_improvement == pytest.approx(0.2)
    assert setup.coarse_solver == "action_lstsq"
    assert setup.residual_augment
    assert setup.residual_augment_max_extra == 2
    assert setup.residual_augment_steps == 3
    assert not setup.residual_augment_include_residuals
    assert setup.smoothed_load_basis
    assert not setup.smoothed_load_basis_combine
    assert setup.smoothed_load_max_directions == 12
    assert setup.smoothed_load_max_rank == 5
    assert setup.smoothed_load_fsavg_lmax == 2
    assert setup.smoothed_load_angular_lmax == 3
    assert setup.smoothed_load_max_extra_units == 4
    assert not setup.smoothed_load_include_rhs


def test_xblock_qi_device_admission_defaults_off_and_handles_host_fallback() -> None:
    off = resolve_xblock_qi_device_admission_setup(
        enabled=False,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=False,
        assembled_operator_built=False,
        assembled_operator_device_resident=False,
        assembled_operator_device_error=None,
        env={},
    )
    fallback = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=True,
        assembled_device_operator_available=True,
        assembled_operator_enabled=True,
        assembled_operator_built=True,
        assembled_operator_device_resident=True,
        assembled_operator_device_error=None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1"},
    )

    assert not off.enabled
    assert not off.should_build
    assert fallback.enabled
    assert not fallback.should_build
    assert fallback.matrix_free_enabled
    assert fallback.reason == "disabled_by_device_host_fallback"
    assert any("device-host fallback" in message for _level, message in fallback.messages)


def test_xblock_qi_device_admission_records_missing_device_metadata() -> None:
    setup = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=True,
        assembled_operator_built=True,
        assembled_operator_device_resident=False,
        assembled_operator_device_error="validation failed",
        env={},
    )

    assert not setup.should_build
    assert setup.reason == "disabled_missing_assembled_device_operator"
    assert setup.metadata["assembled_operator_enabled"] is True
    assert setup.metadata["assembled_operator_built"] is True
    assert setup.metadata["assembled_operator_device_resident"] is False
    assert setup.metadata["assembled_operator_device_error"] == "validation failed"
    assert "SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR=1" in setup.metadata["requires"]
    assert any("no assembled device CSR operator" in message for _level, message in setup.messages)


def test_xblock_qi_device_admission_allows_matrix_free_without_device_operator() -> None:
    setup = resolve_xblock_qi_device_admission_setup(
        enabled=True,
        host_fallback_used=False,
        assembled_device_operator_available=False,
        assembled_operator_enabled=False,
        assembled_operator_built=False,
        assembled_operator_device_resident=False,
        assembled_operator_device_error=None,
        env={"SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE": "1"},
    )

    assert setup.should_build
    assert setup.matrix_free_enabled
    assert setup.reason is None
    assert setup.metadata == {}


def test_xblock_qi_device_base_config_defaults_with_device_operator() -> None:
    setup = resolve_xblock_qi_device_base_config_setup(
        matrix_free_enabled=False,
        assembled_device_operator_available=True,
        precondition_side="right",
        probe_uses_minres_step=lambda: True,
        env={},
    )

    assert setup.rcond == pytest.approx(1.0e-12)
    assert setup.damping == pytest.approx(1.0)
    assert setup.jacobi_damping == pytest.approx(0.7)
    assert setup.jacobi_sweeps == 1
    assert setup.jacobi_floor == pytest.approx(1.0e-14)
    assert setup.jacobi_require_all_diagonal
    assert setup.local_smoother_kind == "auto"
    assert setup.matrix_free_smoother_sweeps == 1
    assert setup.matrix_free_smoother_damping == pytest.approx(1.0)
    assert setup.matrix_free_smoother_step_policy == "residual_minimizing"
    assert setup.matrix_free_block_smoother_max_groups == 32
    assert setup.matrix_free_block_smoother_include_tail
    assert setup.matrix_free_block_smoother_grouping == "contiguous"
    assert setup.jacobi_step_policy == "stationary"
    assert setup.coarse_solver == "action_lstsq"
    assert setup.min_improvement == pytest.approx(0.05)
    assert setup.cycles == 1
    assert not setup.augmented_seed_requested
    assert setup.augmented_seed_max_rank == 1
    assert setup.minres_step
    assert setup.alpha_clip == pytest.approx(10.0)
    assert setup.use_in_krylov_requested
    assert setup.use_in_krylov
    assert not setup.compose_with_base
    assert setup.compose_mode == "multiplicative"


def test_xblock_qi_device_base_config_parses_matrix_free_and_composition_settings() -> None:
    setup = resolve_xblock_qi_device_base_config_setup(
        matrix_free_enabled=True,
        assembled_device_operator_available=False,
        precondition_side="none",
        probe_uses_minres_step=lambda: False,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RCOND": "1e-8",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_DAMPING": "0.6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DAMPING": "0.4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_SWEEPS": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_DIAGONAL_FLOOR": "1e-9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_REQUIRE_ALL_DIAGONAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_LOCAL_SMOOTHER": "matrix-free-block-minres",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_SWEEPS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_DAMPING": "0.75",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_STEP_POLICY": "Fixed",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_SMOOTHER_ALPHA_CLIP": "2.5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_MAX_GROUPS": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_INCLUDE_TAIL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_RCOND": "1e-7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MATRIX_FREE_BLOCK_SMOOTHER_GROUPING": "block-x-species",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_JACOBI_STEP_POLICY": "Residual-Minimizing",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COARSE_SOLVER": "Galerkin",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MIN_IMPROVEMENT": "0.2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_CYCLES": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_AUGMENTED_SEED_MAX_RANK": "9",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ALPHA_CLIP": "3.5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_USE_IN_KRYLOV": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_WITH_BASE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COMPOSE_MODE": "invalid",
        },
    )

    assert setup.rcond == pytest.approx(1.0e-8)
    assert setup.damping == pytest.approx(0.6)
    assert setup.jacobi_damping == pytest.approx(0.4)
    assert setup.jacobi_sweeps == 3
    assert setup.jacobi_floor == pytest.approx(1.0e-9)
    assert not setup.jacobi_require_all_diagonal
    assert setup.local_smoother_kind == "matrix_free_block_minres"
    assert setup.matrix_free_smoother_sweeps == 4
    assert setup.matrix_free_smoother_damping == pytest.approx(0.75)
    assert setup.matrix_free_smoother_step_policy == "fixed"
    assert setup.matrix_free_smoother_alpha_clip == pytest.approx(2.5)
    assert setup.matrix_free_block_smoother_max_groups == 7
    assert not setup.matrix_free_block_smoother_include_tail
    assert setup.matrix_free_block_smoother_rcond == pytest.approx(1.0e-7)
    assert setup.matrix_free_block_smoother_grouping == "block_x_species"
    assert setup.jacobi_step_policy == "residual_minimizing"
    assert setup.coarse_solver == "galerkin"
    assert setup.min_improvement == pytest.approx(0.2)
    assert setup.cycles == 5
    assert setup.augmented_seed_requested
    assert setup.augmented_seed_max_rank == 9
    assert not setup.minres_step
    assert setup.alpha_clip == pytest.approx(3.5)
    assert setup.use_in_krylov_requested
    assert not setup.use_in_krylov
    assert setup.compose_with_base
    assert setup.compose_mode == "multiplicative"


def test_xblock_qi_device_enrichment_config_defaults_follow_matrix_free() -> None:
    off = resolve_xblock_qi_device_enrichment_config_setup(matrix_free_enabled=False, env={})
    matrix_free = resolve_xblock_qi_device_enrichment_config_setup(matrix_free_enabled=True, env={})

    assert not off.residual_enrichment
    assert off.residual_enrichment_depth == 0
    assert matrix_free.residual_enrichment
    assert matrix_free.residual_enrichment_depth == 2
    assert matrix_free.residual_enrichment_include_residual
    assert not matrix_free.recycle_enrichment
    assert matrix_free.recycle_cycles == 0
    assert not matrix_free.operator_krylov_enrichment
    assert matrix_free.operator_krylov_depth == 0
    assert not matrix_free.adjoint_krylov_enrichment
    assert matrix_free.adjoint_krylov_depth == 0
    assert matrix_free.adjoint_krylov_transpose_source == "autodiff"
    assert not matrix_free.operator_action_enrichment
    assert matrix_free.operator_action_depth == 0


def test_xblock_qi_device_enrichment_config_parses_explicit_settings() -> None:
    setup = resolve_xblock_qi_device_enrichment_config_setup(
        matrix_free_enabled=False,
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_DEPTH": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_ENRICHMENT_INCLUDE_RESIDUAL": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RECYCLE_CYCLES": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_KRYLOV_DEPTH": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_DEPTH": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ADJOINT_KRYLOV_TRANSPOSE": "Finite-Difference",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_ENRICHMENT": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_OPERATOR_ACTION_DEPTH": "6",
        },
    )

    assert setup.residual_enrichment
    assert setup.residual_enrichment_depth == 5
    assert not setup.residual_enrichment_include_residual
    assert setup.recycle_enrichment
    assert setup.recycle_cycles == 3
    assert setup.operator_krylov_enrichment
    assert setup.operator_krylov_depth == 2
    assert setup.adjoint_krylov_enrichment
    assert setup.adjoint_krylov_depth == 4
    assert setup.adjoint_krylov_transpose_source == "finite_difference"
    assert setup.operator_action_enrichment
    assert setup.operator_action_depth == 6


def test_xblock_qi_device_multilevel_config_defaults_disabled() -> None:
    setup = resolve_xblock_qi_device_multilevel_config_setup(env={})

    assert not setup.multilevel_coarse
    assert setup.multilevel_max_levels == 1
    assert setup.multilevel_aggregate_factor == 2
    assert setup.multilevel_max_angular_mode == 1
    assert setup.multilevel_max_radial_degree == 2
    assert setup.multilevel_max_pitch_degree == 0
    assert not setup.multilevel_current_moments
    assert setup.multilevel_species_current_moments
    assert setup.multilevel_radial_current_moments
    assert setup.multilevel_tail_constraint_moments
    assert setup.multilevel_current_max_pitch_degree == 1
    assert not setup.multilevel_residual_equation
    assert setup.multilevel_residual_equation_max_level_rank == 16
    assert setup.multilevel_residual_equation_order == "coarse_to_fine"
    assert setup.multilevel_residual_equation_solver == "action_lstsq"
    assert setup.multilevel_residual_equation_include_global


def test_xblock_qi_device_multilevel_config_reuses_coarse_operator_alias() -> None:
    alias = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
        }
    )
    explicit_off = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_REUSE_COARSE_OPERATOR": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE": "0",
        }
    )

    assert alias.multilevel_coarse
    assert alias.multilevel_max_levels == 3
    assert not explicit_off.multilevel_coarse
    assert explicit_off.multilevel_max_levels == 1


def test_xblock_qi_device_multilevel_config_parses_explicit_controls() -> None:
    setup = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_COARSE": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_LEVELS": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_AGGREGATE_FACTOR": "5",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_ANGULAR_MODE": "3",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_RADIAL_DEGREE": "6",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_MAX_PITCH_DEGREE": "2",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MOMENTS": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_SPECIES_CURRENT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RADIAL_CURRENT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_TAIL_CONSTRAINT_MOMENTS": "0",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_CURRENT_MAX_PITCH_DEGREE": "4",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION": "1",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_MAX_LEVEL_RANK": "7",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER": "fine-to-coarse",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "qtaq",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_INCLUDE_GLOBAL": "0",
        }
    )

    assert setup.multilevel_coarse
    assert setup.multilevel_max_levels == 4
    assert setup.multilevel_aggregate_factor == 5
    assert setup.multilevel_max_angular_mode == 3
    assert setup.multilevel_max_radial_degree == 6
    assert setup.multilevel_max_pitch_degree == 2
    assert setup.multilevel_current_moments
    assert not setup.multilevel_species_current_moments
    assert not setup.multilevel_radial_current_moments
    assert not setup.multilevel_tail_constraint_moments
    assert setup.multilevel_current_max_pitch_degree == 4
    assert setup.multilevel_residual_equation
    assert setup.multilevel_residual_equation_max_level_rank == 7
    assert setup.multilevel_residual_equation_order == "fine_to_coarse"
    assert setup.multilevel_residual_equation_solver == "galerkin"
    assert not setup.multilevel_residual_equation_include_global


def test_xblock_qi_device_multilevel_config_normalizes_invalid_residual_controls() -> None:
    setup = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_ORDER": "inside-out",
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "least-squares",
        }
    )
    invalid_solver = resolve_xblock_qi_device_multilevel_config_setup(
        env={
            "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_MULTILEVEL_RESIDUAL_EQUATION_SOLVER": "unknown",
        }
    )

    assert setup.multilevel_residual_equation_order == "coarse_to_fine"
    assert setup.multilevel_residual_equation_solver == "action_lstsq"
    assert invalid_solver.multilevel_residual_equation_solver == "action_lstsq"


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
