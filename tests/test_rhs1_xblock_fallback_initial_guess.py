from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax import v3_driver as vd
from sfincs_jax.problems.profile_response.policies import (
    parse_rhs1_pas_tz_guarded_structured_levels,
    rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_extra_coarse_metadata,
    rhs1_qi_device_extra_coarse_setup_kwargs,
    rhs1_qi_device_coupled_install_on_reject_requested,
    rhs1_qi_device_probe_uses_minres_step,
    rhs1_qi_device_progress_messages,
    rhs1_qi_device_rank_budget,
    rhs1_qi_device_residual_correction_controls,
    rhs1_qi_device_residual_correction_metadata,
    rhs1_qi_device_residual_correction_setup_kwargs,
    rhs1_qi_device_setup_summary,
    rhs1_qi_device_status_fields,
    rhs1_qi_device_tail_block_required,
    rhs1_xblock_fallback_initial_guess,
)


def test_driver_private_policy_helpers_alias_canonical_profile_response_helpers():
    assert (
        vd._rhs1_pas_tz_guarded_structured_levels
        is parse_rhs1_pas_tz_guarded_structured_levels
    )
    assert (
        vd._rhs1_qi_device_extra_coarse_controls is rhs1_qi_device_extra_coarse_controls
    )
    assert (
        vd._rhs1_qi_device_extra_coarse_setup_kwargs
        is rhs1_qi_device_extra_coarse_setup_kwargs
    )
    assert (
        vd._rhs1_qi_device_extra_coarse_metadata
        is rhs1_qi_device_extra_coarse_metadata
    )
    assert (
        vd._rhs1_qi_device_coupled_install_on_reject_requested
        is rhs1_qi_device_coupled_install_on_reject_requested
    )
    assert (
        vd._rhs1_qi_device_probe_uses_minres_step
        is rhs1_qi_device_probe_uses_minres_step
    )
    assert vd._rhs1_qi_device_progress_messages is rhs1_qi_device_progress_messages
    assert (
        vd._rhs1_qi_device_residual_correction_controls
        is rhs1_qi_device_residual_correction_controls
    )
    assert (
        vd._rhs1_qi_device_residual_correction_setup_kwargs
        is rhs1_qi_device_residual_correction_setup_kwargs
    )
    assert (
        vd._rhs1_qi_device_residual_correction_metadata
        is rhs1_qi_device_residual_correction_metadata
    )
    assert vd._rhs1_qi_device_setup_summary is rhs1_qi_device_setup_summary
    assert vd._rhs1_qi_device_status_fields is rhs1_qi_device_status_fields
    assert vd._rhs1_qi_device_tail_block_required is rhs1_qi_device_tail_block_required
    assert vd._rhs1_qi_device_rank_budget is rhs1_qi_device_rank_budget
    assert vd._rhs1_xblock_fallback_initial_guess is rhs1_xblock_fallback_initial_guess


def test_fallback_initial_guess_reuses_left_candidate_that_improves_rhs():
    original = jnp.array([0.0, 0.0])
    candidate = np.array([1.0, -2.0])

    x0, started_from_candidate, improved_rhs = rhs1_xblock_fallback_initial_guess(
        candidate=candidate,
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="left",
    )

    assert started_from_candidate is True
    assert improved_rhs is True
    np.testing.assert_allclose(np.asarray(x0), candidate)


def test_fallback_initial_guess_rejects_candidate_that_does_not_improve_rhs():
    original = jnp.array([3.0, 4.0])

    x0, started_from_candidate, improved_rhs = rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=2.0,
        rhs_norm=1.0,
        precondition_side="left",
    )

    assert started_from_candidate is False
    assert improved_rhs is False
    assert x0 is original


def test_fallback_initial_guess_rejects_right_preconditioned_candidate():
    original = jnp.array([3.0, 4.0])

    x0, started_from_candidate, improved_rhs = rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="right",
    )

    assert started_from_candidate is False
    assert improved_rhs is True
    assert x0 is original


def test_fallback_initial_guess_rejects_bad_shape_or_nonfinite_candidate():
    original = jnp.array([3.0, 4.0])

    x0_bad_shape, started_bad_shape, _ = rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0, 3.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="none",
    )
    x0_nonfinite, started_nonfinite, _ = rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, np.nan]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=0.1,
        rhs_norm=1.0,
        precondition_side="none",
    )

    assert started_bad_shape is False
    assert x0_bad_shape is original
    assert started_nonfinite is False
    assert x0_nonfinite is original


@pytest.mark.parametrize(
    ("candidate_residual_norm", "rhs_norm"),
    [
        (np.inf, 1.0),
        (np.nan, 1.0),
        (0.1, np.inf),
        (0.1, np.nan),
        (1.0, 1.0),
    ],
)
def test_fallback_initial_guess_requires_finite_strict_rhs_improvement(
    candidate_residual_norm, rhs_norm
):
    original = jnp.array([3.0, 4.0])

    x0, started_from_candidate, improved_rhs = rhs1_xblock_fallback_initial_guess(
        candidate=np.array([1.0, -2.0]),
        original_x0=original,
        rhs_shape=(2,),
        candidate_residual_norm=candidate_residual_norm,
        rhs_norm=rhs_norm,
        precondition_side="left",
    )

    assert started_from_candidate is False
    assert improved_rhs is False
    assert x0 is original


def test_guarded_structured_levels_parse_aliases_from_canonical_module() -> None:
    assert parse_rhs1_pas_tz_guarded_structured_levels("") == ()
    assert parse_rhs1_pas_tz_guarded_structured_levels("off") == ()
    assert parse_rhs1_pas_tz_guarded_structured_levels("structured") == (
        "xmg",
        "collision",
    )
    assert parse_rhs1_pas_tz_guarded_structured_levels("x+coll+x") == (
        "xmg",
        "collision",
    )
    assert parse_rhs1_pas_tz_guarded_structured_levels("unknown,collision_diag") == (
        "collision",
    )


@pytest.mark.parametrize(
    "value",
    ["minres", "line-search", "linesearch", "residual_minimizing"],
)
def test_qi_device_probe_minres_policy_accepts_aliases(monkeypatch, value) -> None:
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY",
        value,
    )

    assert rhs1_qi_device_probe_uses_minres_step() is True


def test_qi_device_probe_minres_policy_rejects_fixed_step(monkeypatch) -> None:
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_STEP_POLICY",
        "fixed",
    )

    assert rhs1_qi_device_probe_uses_minres_step() is False


def test_qi_device_extra_coarse_controls_parse_bounded_overrides(monkeypatch) -> None:
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION",
        "yes",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_MAX_RANK",
        "7",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_GLOBAL_MOMENT_RESIDUAL_EQUATION_SOLVER",
        "least-squares",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_ACTIVE_PATTERN_COARSE_MAX_RANK",
        "-2",
    )

    controls = rhs1_qi_device_extra_coarse_controls()

    assert controls["global_moment_residual_equation"] is True
    assert controls["global_moment_residual_equation_max_rank"] == 7
    assert controls["global_moment_residual_equation_solver"] == "action_lstsq"
    assert controls["active_pattern_coarse_max_rank"] == 1
    assert controls["multilevel_species_current_moments"] is True


def test_qi_device_residual_correction_controls_parse_bounded_overrides(monkeypatch) -> None:
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION",
        "yes",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_EQUATION_MAX_RANK",
        "-4",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_SOLVER",
        "schur",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_COUPLED_RESIDUAL_EQUATION_MIN_RELATIVE_IMPROVEMENT",
        "-0.5",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_RESIDUAL_SNAPSHOT_RESIDUAL_EQUATION_SOLVER",
        "least-squares",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_DEVICE_PRECONDITIONER_BLOCK_SCHUR_RESIDUAL_INCLUDE_AGGREGATES",
        "0",
    )

    controls = rhs1_qi_device_residual_correction_controls()

    assert controls["block_schur_residual_equation"] is True
    assert controls["block_schur_residual_equation_max_rank"] == 1
    assert controls["coupled_residual_equation_solver"] == "galerkin"
    assert controls["coupled_residual_equation_min_improvement"] == pytest.approx(0.0)
    assert controls["residual_snapshot_residual_equation_solver"] == "action_lstsq"
    assert controls["block_schur_residual_include_aggregates"] is False
    assert controls["residual_snapshot_include_primal"] is True


def _rank_budget(**overrides):
    defaults = {
        "seed_max_rank": 4,
        "n_species": 2,
        "residual_enrichment": False,
        "residual_enrichment_depth": 2,
        "residual_enrichment_include_residual": True,
        "recycle_enrichment": False,
        "recycle_cycles": 1,
        "operator_krylov_enrichment": False,
        "operator_krylov_depth": 3,
        "adjoint_krylov_enrichment": False,
        "adjoint_krylov_depth": 3,
        "operator_action_enrichment": False,
        "operator_action_depth": 1,
        "multilevel_coarse": False,
        "multilevel_max_rank": None,
        "multilevel_current_moments": False,
        "multilevel_current_max_pitch_degree": 1,
        "multilevel_residual_equation": False,
        "multilevel_residual_equation_max_level_rank": 5,
        "multilevel_max_levels": 2,
        "global_moment_residual_equation": False,
        "global_moment_residual_equation_max_rank": 7,
        "residual_galerkin_equation": False,
        "residual_galerkin_equation_max_rank": 11,
        "phase_space_residual_equation": False,
        "phase_space_residual_equation_max_rank": 13,
        "residual_region_bounce_coarse": False,
        "residual_region_bounce_coarse_max_rank": 17,
        "active_pattern_coarse": False,
        "active_pattern_coarse_max_rank": 19,
        "block_schur_residual_equation": False,
        "block_schur_residual_equation_max_rank": 23,
        "coupled_residual_equation": False,
        "coupled_residual_equation_max_rank": 29,
        "residual_snapshot_enrichment": False,
        "residual_snapshot_max_rank": 31,
        "residual_snapshot_residual_equation": False,
        "residual_snapshot_residual_equation_max_rank": 37,
        "block_schur_residual_enrichment": False,
        "block_schur_residual_max_rank": 41,
        "max_rank_env_value": "",
    }
    defaults.update(overrides)
    return rhs1_qi_device_rank_budget(**defaults)


def test_qi_device_rank_budget_keeps_no_cap_without_active_enrichment() -> None:
    setup = _rank_budget()

    assert setup.rank_budget == 4
    assert setup.max_rank is None


def test_qi_device_rank_budget_tracks_active_controls_and_explicit_cap() -> None:
    setup = _rank_budget(
        residual_enrichment=True,
        recycle_enrichment=True,
        operator_krylov_enrichment=True,
        operator_action_enrichment=True,
        multilevel_coarse=True,
        multilevel_max_rank=6,
        multilevel_current_moments=True,
        global_moment_residual_equation=True,
        coupled_residual_equation=True,
        max_rank_env_value="9",
    )

    assert setup.rank_budget == 72
    assert setup.max_rank == 9


def test_qi_device_rank_budget_invalid_cap_falls_back_to_budget() -> None:
    setup = _rank_budget(
        residual_snapshot_enrichment=True,
        block_schur_residual_enrichment=True,
        max_rank_env_value="bad",
    )

    assert setup.rank_budget == 76
    assert setup.max_rank == 76


def test_qi_device_rank_budget_preserves_adjoint_only_no_cap_behavior() -> None:
    setup = _rank_budget(adjoint_krylov_enrichment=True)

    assert setup.rank_budget == 8
    assert setup.max_rank is None


def _progress_messages(**overrides) -> tuple[str, ...]:
    defaults = {
        "assembled_device_operator_available": True,
        "residual_enrichment": False,
        "residual_enrichment_depth": 2,
        "operator_action_enrichment": False,
        "operator_action_depth": 1,
        "operator_krylov_enrichment": False,
        "operator_krylov_depth": 3,
        "adjoint_krylov_enrichment": False,
        "adjoint_krylov_depth": 4,
        "adjoint_krylov_transpose_source": "csr",
        "max_rank": None,
        "multilevel_coarse": False,
        "multilevel_max_levels": 2,
        "multilevel_aggregate_factor": 3,
        "multilevel_max_pitch_degree": 1,
        "multilevel_current_moments": False,
        "multilevel_max_rank": None,
        "multilevel_residual_equation": False,
        "multilevel_residual_equation_max_level_rank": 5,
        "multilevel_residual_equation_order": "coarse_to_fine",
        "multilevel_residual_equation_solver": "galerkin",
        "multilevel_residual_equation_include_global": True,
        "global_moment_residual_equation": False,
        "global_moment_residual_equation_max_rank": 7,
        "global_moment_residual_equation_solver": "action_lstsq",
        "global_moment_residual_equation_include_profile": True,
        "global_moment_residual_equation_include_current": True,
        "global_moment_residual_equation_include_tail": False,
        "residual_galerkin_equation": False,
        "residual_galerkin_equation_max_stages": 2,
        "residual_galerkin_equation_max_stage_rank": 3,
        "residual_galerkin_equation_max_rank": 11,
        "residual_galerkin_equation_solver": "galerkin",
        "residual_galerkin_equation_include_global_residual": True,
        "residual_galerkin_equation_include_block_residuals": True,
        "residual_galerkin_equation_include_operator_images": False,
        "phase_space_residual_equation": False,
        "phase_space_residual_equation_max_rank": 13,
        "phase_space_residual_equation_solver": "action_lstsq",
        "phase_space_residual_equation_boundary": 0.35,
        "phase_space_residual_equation_include_global": False,
        "phase_space_residual_equation_include_radial": True,
        "phase_space_residual_equation_include_species": True,
        "residual_region_bounce_coarse": False,
        "residual_region_bounce_coarse_max_rank": 17,
        "residual_region_bounce_coarse_solver": "galerkin",
        "residual_region_bounce_coarse_boundary": 0.25,
        "residual_region_bounce_coarse_min_energy": 0.02,
        "residual_region_bounce_coarse_include_global": True,
        "residual_region_bounce_coarse_include_radial": False,
        "residual_region_bounce_coarse_include_species": True,
        "residual_region_bounce_coarse_region_bands": "bounce,trapped",
        "active_pattern_coarse": False,
        "active_pattern_coarse_max_rank": 19,
        "active_pattern_coarse_max_candidates": 23,
        "active_pattern_coarse_solver": "action_lstsq",
        "active_pattern_coarse_min_chunk_energy": 0.03,
        "active_pattern_coarse_include_global": True,
        "block_schur_residual_equation": False,
        "block_schur_residual_equation_max_rank": 29,
        "block_schur_residual_equation_include_global": True,
        "block_schur_residual_equation_include_blocks": True,
        "block_schur_residual_equation_include_aggregates": False,
        "coupled_residual_equation": False,
        "coupled_residual_equation_max_rank": 31,
        "coupled_residual_equation_solver": "galerkin",
        "coupled_residual_equation_include_flat": True,
        "coupled_residual_equation_install_on_reject": False,
        "coupled_residual_equation_min_improvement": 0.1,
        "residual_snapshot_enrichment": False,
        "residual_snapshot_max_rank": 37,
        "residual_snapshot_include_primal": True,
        "residual_snapshot_use_adjoint": True,
        "residual_snapshot_include_global": True,
        "residual_snapshot_include_blocks": False,
        "residual_snapshot_include_aggregates": True,
        "residual_snapshot_residual_equation": False,
        "residual_snapshot_residual_equation_max_rank": 41,
        "residual_snapshot_residual_equation_solver": "action_lstsq",
        "residual_snapshot_residual_equation_include_global": True,
        "block_schur_residual_enrichment": False,
        "block_schur_residual_max_rank": 43,
        "block_schur_residual_include_global": True,
        "block_schur_residual_include_blocks": True,
        "block_schur_residual_include_aggregates": True,
    }
    defaults.update(overrides)
    return rhs1_qi_device_progress_messages(**defaults)


def test_qi_device_progress_messages_stays_quiet_without_active_features() -> None:
    assert _progress_messages() == ()


def test_qi_device_progress_messages_reports_matrix_free_fallback() -> None:
    messages = _progress_messages(assembled_device_operator_available=False)

    assert len(messages) == 1
    assert "matrix-free coarse-only operator-on-basis fallback" in messages[0]


def test_qi_device_progress_messages_reports_enabled_feature_parameters() -> None:
    messages = _progress_messages(
        residual_enrichment=True,
        residual_enrichment_depth=5,
        max_rank=9,
        coupled_residual_equation=True,
        coupled_residual_equation_max_rank=15,
        coupled_residual_equation_include_flat=False,
        coupled_residual_equation_install_on_reject=True,
        coupled_residual_equation_min_improvement=0.25,
        residual_snapshot_residual_equation=True,
        residual_snapshot_residual_equation_max_rank=21,
        residual_snapshot_include_blocks=True,
    )

    assert len(messages) == 3
    assert "residual enrichment (depth=5 max_rank=9)" in messages[0]
    assert (
        "coupled residual equation (max_rank=15 solver=galerkin include_flat=0 "
        "install_on_reject=1 min_improvement=2.500e-01)"
    ) in messages[1]
    assert (
        "residual-snapshot residual equation (max_rank=21 solver=action_lstsq "
        "include_global=1 include_primal=1 use_adjoint=1 include_blocks=1 "
        "include_aggregates=1)"
    ) in messages[2]


def _enrichment_config(**overrides):
    defaults = {
        "residual_enrichment": False,
        "residual_enrichment_depth": 2,
        "residual_enrichment_include_residual": True,
        "recycle_enrichment": False,
        "recycle_cycles": 1,
        "operator_krylov_enrichment": False,
        "operator_krylov_depth": 3,
        "adjoint_krylov_enrichment": False,
        "adjoint_krylov_depth": 4,
        "adjoint_krylov_transpose_source": "csr",
        "operator_action_enrichment": False,
        "operator_action_depth": 1,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _multilevel_config(**overrides):
    defaults = {
        "multilevel_coarse": False,
        "multilevel_max_levels": 2,
        "multilevel_aggregate_factor": 2,
        "multilevel_max_pitch_degree": 1,
        "multilevel_current_moments": False,
        "multilevel_current_max_pitch_degree": 1,
        "multilevel_residual_equation": False,
        "multilevel_residual_equation_max_level_rank": 5,
        "multilevel_residual_equation_order": "coarse_to_fine",
        "multilevel_residual_equation_solver": "galerkin",
        "multilevel_residual_equation_include_global": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _extra_coarse_controls_for_summary(**overrides):
    controls = rhs1_qi_device_extra_coarse_controls()
    controls.update(
        {
            "global_moment_residual_equation": False,
            "global_moment_residual_equation_max_rank": 7,
            "global_moment_residual_equation_solver": "action_lstsq",
            "global_moment_residual_equation_include_profile": True,
            "global_moment_residual_equation_include_current": True,
            "global_moment_residual_equation_include_tail": False,
            "residual_galerkin_equation": False,
            "residual_galerkin_equation_max_stages": 2,
            "residual_galerkin_equation_max_stage_rank": 3,
            "residual_galerkin_equation_max_rank": 11,
            "residual_galerkin_equation_solver": "galerkin",
            "residual_galerkin_equation_include_global_residual": True,
            "residual_galerkin_equation_include_block_residuals": True,
            "residual_galerkin_equation_include_operator_images": False,
            "phase_space_residual_equation": False,
            "phase_space_residual_equation_max_rank": 13,
            "phase_space_residual_equation_solver": "action_lstsq",
            "phase_space_residual_equation_boundary": 0.35,
            "phase_space_residual_equation_include_global": False,
            "phase_space_residual_equation_include_radial": True,
            "phase_space_residual_equation_include_species": True,
            "residual_region_bounce_coarse": False,
            "residual_region_bounce_coarse_max_rank": 17,
            "residual_region_bounce_coarse_solver": "galerkin",
            "residual_region_bounce_coarse_boundary": 0.25,
            "residual_region_bounce_coarse_min_energy": 0.02,
            "residual_region_bounce_coarse_include_global": True,
            "residual_region_bounce_coarse_include_radial": False,
            "residual_region_bounce_coarse_include_species": True,
            "residual_region_bounce_coarse_region_bands": "bounce,trapped",
            "active_pattern_coarse": False,
            "active_pattern_coarse_max_rank": 19,
            "active_pattern_coarse_max_candidates": 23,
            "active_pattern_coarse_solver": "action_lstsq",
            "active_pattern_coarse_min_chunk_energy": 0.03,
            "active_pattern_coarse_include_global": True,
        }
    )
    controls.update(overrides)
    return controls


def _residual_controls_for_summary(**overrides):
    controls = rhs1_qi_device_residual_correction_controls()
    controls.update(
        {
            "block_schur_residual_equation": False,
            "block_schur_residual_equation_max_rank": 29,
            "block_schur_residual_equation_include_global": True,
            "block_schur_residual_equation_include_blocks": True,
            "block_schur_residual_equation_include_aggregates": False,
            "coupled_residual_equation": False,
            "coupled_residual_equation_max_rank": 31,
            "coupled_residual_equation_solver": "galerkin",
            "coupled_residual_equation_include_flat": True,
            "coupled_residual_equation_install_on_reject": False,
            "coupled_residual_equation_min_improvement": 0.1,
            "residual_snapshot_enrichment": False,
            "residual_snapshot_max_rank": 37,
            "residual_snapshot_include_primal": True,
            "residual_snapshot_use_adjoint": True,
            "residual_snapshot_include_global": True,
            "residual_snapshot_include_blocks": False,
            "residual_snapshot_include_aggregates": True,
            "residual_snapshot_residual_equation": False,
            "residual_snapshot_residual_equation_max_rank": 41,
            "residual_snapshot_residual_equation_solver": "action_lstsq",
            "residual_snapshot_residual_equation_include_global": True,
            "block_schur_residual_enrichment": False,
            "block_schur_residual_max_rank": 43,
            "block_schur_residual_include_global": True,
            "block_schur_residual_include_blocks": True,
            "block_schur_residual_include_aggregates": True,
        }
    )
    controls.update(overrides)
    return controls


def test_qi_device_setup_summary_matches_rank_progress_and_seed_policy() -> None:
    summary = rhs1_qi_device_setup_summary(
        seed_max_rank=4,
        n_species=2,
        assembled_device_operator_available=False,
        enrichment_config=_enrichment_config(
            residual_enrichment=True,
            residual_enrichment_depth=3,
        ),
        multilevel_config=_multilevel_config(),
        multilevel_max_rank=None,
        extra_coarse_controls=_extra_coarse_controls_for_summary(),
        residual_correction_controls=_residual_controls_for_summary(),
        max_rank_env_value="",
    )

    assert summary.rank_budget == 8
    assert summary.max_rank == 8
    assert summary.residual_seed_required is True
    assert len(summary.progress_messages) == 2
    assert "matrix-free coarse-only" in summary.progress_messages[0]
    assert "residual enrichment (depth=3 max_rank=8)" in summary.progress_messages[1]


def test_qi_device_setup_summary_preserves_adjoint_only_compatibility() -> None:
    summary = rhs1_qi_device_setup_summary(
        seed_max_rank=4,
        n_species=1,
        assembled_device_operator_available=True,
        enrichment_config=_enrichment_config(adjoint_krylov_enrichment=True),
        multilevel_config=_multilevel_config(),
        multilevel_max_rank=None,
        extra_coarse_controls=_extra_coarse_controls_for_summary(),
        residual_correction_controls=_residual_controls_for_summary(),
        max_rank_env_value="",
    )

    assert summary.rank_budget == 9
    assert summary.max_rank is None
    assert summary.residual_seed_required is False
    assert len(summary.progress_messages) == 1
    assert "adjoint-normal Krylov coarse enrichment" in summary.progress_messages[0]


def test_qi_device_extra_coarse_setup_kwargs_map_solver_parameter_names() -> None:
    kwargs = rhs1_qi_device_extra_coarse_setup_kwargs(
        _extra_coarse_controls_for_summary(
            phase_space_residual_equation_boundary=0.125,
            residual_region_bounce_coarse_boundary=0.25,
            residual_region_bounce_coarse_min_energy=0.03125,
            active_pattern_coarse_min_chunk_energy=0.0625,
        )
    )

    assert kwargs["phase_space_residual_equation_trapped_boundary_fraction"] == 0.125
    assert kwargs["residual_region_bounce_coarse_trapped_boundary_fraction"] == 0.25
    assert (
        kwargs["residual_region_bounce_coarse_min_region_energy_fraction"]
        == 0.03125
    )
    assert kwargs["active_pattern_coarse_min_chunk_energy_fraction"] == 0.0625
    assert kwargs["global_moment_residual_equation_solver"] == "action_lstsq"


def test_qi_device_residual_correction_setup_kwargs_map_solver_parameters() -> None:
    kwargs = rhs1_qi_device_residual_correction_setup_kwargs(
        _residual_controls_for_summary(
            coupled_residual_equation=True,
            coupled_residual_equation_min_improvement=0.2,
            residual_snapshot_residual_equation=True,
        )
    )

    assert kwargs["coupled_residual_equation"] is True
    assert kwargs["coupled_residual_equation_min_relative_improvement"] == 0.2
    assert kwargs["residual_snapshot_residual_equation"] is True
    assert "coupled_residual_equation_install_on_reject" not in kwargs


def test_qi_device_extra_coarse_metadata_uses_requested_key_names() -> None:
    metadata = rhs1_qi_device_extra_coarse_metadata(
        _extra_coarse_controls_for_summary(
            global_moment_residual_equation=True,
            residual_region_bounce_coarse_min_energy=0.04,
            active_pattern_coarse_include_species=False,
        )
    )

    assert metadata["global_moment_residual_equation_requested"] is True
    assert (
        metadata["residual_region_bounce_coarse_min_region_energy_fraction_requested"]
        == 0.04
    )
    assert metadata["active_pattern_coarse_include_species_requested"] is False


def test_qi_device_residual_correction_metadata_uses_requested_key_names() -> None:
    metadata = rhs1_qi_device_residual_correction_metadata(
        _residual_controls_for_summary(
            coupled_residual_equation_install_on_reject=True,
            residual_snapshot_use_adjoint=False,
            block_schur_residual_include_aggregates=False,
        )
    )

    assert (
        metadata["coupled_residual_equation_install_in_krylov_on_reject_requested"]
        is True
    )
    assert metadata["residual_snapshot_use_adjoint_requested"] is False
    assert metadata["block_schur_residual_include_aggregates_requested"] is False


def test_qi_device_tail_block_required_tracks_tail_moment_controls() -> None:
    assert rhs1_qi_device_tail_block_required(
        multilevel_coarse=True,
        extra_coarse_controls=_extra_coarse_controls_for_summary(),
    )
    assert rhs1_qi_device_tail_block_required(
        multilevel_coarse=False,
        extra_coarse_controls=_extra_coarse_controls_for_summary(
            global_moment_residual_equation=True,
            global_moment_residual_equation_include_tail=True,
        ),
    )
    assert not rhs1_qi_device_tail_block_required(
        multilevel_coarse=False,
        extra_coarse_controls=_extra_coarse_controls_for_summary(
            global_moment_residual_equation=True,
            global_moment_residual_equation_include_tail=False,
        ),
    )


def test_qi_device_coupled_install_on_reject_uses_grouped_control() -> None:
    assert rhs1_qi_device_coupled_install_on_reject_requested(
        _residual_controls_for_summary(
            coupled_residual_equation_install_on_reject=True
        )
    )
    assert not rhs1_qi_device_coupled_install_on_reject_requested(
        _residual_controls_for_summary(
            coupled_residual_equation_install_on_reject=False
        )
    )


def test_qi_device_status_fields_format_controls_and_metadata() -> None:
    fields = rhs1_qi_device_status_fields(
        extra_coarse_controls=_extra_coarse_controls_for_summary(
            global_moment_residual_equation=True,
            residual_region_bounce_coarse=True,
            active_pattern_coarse=True,
        ),
        residual_correction_controls=_residual_controls_for_summary(
            coupled_residual_equation=True,
            residual_snapshot_enrichment=True,
        ),
        metadata={
            "global_moment_residual_equation_rank": 2,
            "global_moment_residual_equation_candidate_count": 3,
            "global_moment_residual_equation_condition_estimate": 4.5,
            "residual_region_bounce_coarse_rank": 5,
            "active_pattern_coarse_candidate_count": 7,
            "coupled_residual_equation_rank": 11,
        },
    )

    assert "global_moment_equation=1" in fields
    assert "global_moment_rank=2" in fields
    assert "global_moment_cond=4.500000e+00" in fields
    assert "residual_region_bounce=1" in fields
    assert "active_pattern_candidates=7" in fields
    assert "coupled_equation=1" in fields
    assert "coupled_rank=11" in fields
    assert "residual_snapshot=1" in fields


def test_qi_device_status_fields_allow_seed_only_residual_controls() -> None:
    fields = rhs1_qi_device_status_fields(
        extra_coarse_controls=_extra_coarse_controls_for_summary(
            phase_space_residual_equation=True,
            active_pattern_coarse=True,
        ),
        residual_correction_controls={},
        metadata={
            "phase_space_residual_equation_rank": 5,
            "phase_space_residual_equation_candidate_count": 9,
            "active_pattern_coarse_rank": 3,
            "active_pattern_coarse_candidate_count": 4,
        },
    )

    assert "phase_space_equation=1" in fields
    assert "phase_space_rank=5" in fields
    assert "active_pattern_coarse=1" in fields
    assert "active_pattern_rank=3" in fields
    assert "block_schur_equation=0" in fields
    assert "coupled_equation=0" in fields
    assert "residual_snapshot=0" in fields
