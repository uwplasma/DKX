import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax import v3_driver as vd
from sfincs_jax.problems.profile_response.policies import (
    parse_rhs1_pas_tz_guarded_structured_levels,
    rhs1_qi_device_extra_coarse_controls,
    rhs1_qi_device_probe_uses_minres_step,
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
        vd._rhs1_qi_device_probe_uses_minres_step
        is rhs1_qi_device_probe_uses_minres_step
    )
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
