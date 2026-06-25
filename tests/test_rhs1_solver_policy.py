from __future__ import annotations

from sfincs_jax.problems.profile_response.solver_policy import (
    RHS1PostSolveCorrectionPolicy,
    read_bool_env,
    read_float_env,
    read_int_env,
    read_post_minres_policy,
    read_post_residual_equation_policy,
    read_post_solve_correction_policy,
    read_probe_coarse_policy,
)


def test_rhs1_solver_policy_env_parsers_match_driver_semantics() -> None:
    env = {
        "BOOL_TRUE": ".true.",
        "BOOL_FALSE": "off",
        "BOOL_BAD": "maybe",
        "INT_OK": "7",
        "INT_BAD": "bad",
        "INT_LOW": "-3",
        "FLOAT_OK": "2.5e-2",
        "FLOAT_BAD": "bad",
        "FLOAT_LOW": "-1.0",
    }

    assert read_bool_env("BOOL_TRUE", env=env) is True
    assert read_bool_env("BOOL_FALSE", default=True, env=env) is False
    assert read_bool_env("BOOL_BAD", default=True, env=env) is True
    assert read_int_env("INT_OK", default=1, minimum=0, env=env) == 7
    assert read_int_env("INT_BAD", default=4, minimum=0, env=env) == 4
    assert read_int_env("INT_LOW", default=4, minimum=2, env=env) == 2
    assert read_float_env("FLOAT_OK", default=1.0, minimum=0.0, env=env) == 2.5e-2
    assert read_float_env("FLOAT_BAD", default=1.5, minimum=0.0, env=env) == 1.5
    assert read_float_env("FLOAT_LOW", default=1.5, minimum=0.25, env=env) == 0.25


def test_probe_coarse_policy_is_disabled_by_default() -> None:
    policy = read_probe_coarse_policy(env={})

    assert policy.steps_requested == 0
    assert policy.max_directions == 16
    assert policy.max_extra_units == 8
    assert policy.fsavg_lmax == 2
    assert policy.angular_lmax == -1
    assert policy.include_angular_residual is False
    assert policy.include_raw is True
    assert policy.alpha_clip == 0.0
    assert policy.rcond == 1.0e-12
    assert policy.min_improvement == 0.0


def test_post_residual_equation_policy_reads_opt_in_controls() -> None:
    env = {
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION": "1",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_STEPS": "2",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_MAX_DIRECTIONS": "96",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_MAX_EXTRA_UNITS": "12",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_FSAVG_LMAX": "5",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_ANGULAR_LMAX": "3",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_ANGULAR_RESIDUAL": "0",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_INCLUDE_RAW": "false",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_INCLUDE_POST_COARSE": "0",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_INCLUDE_QI_BASIS": "no",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_ALPHA_CLIP": "4.0",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_RCOND": "1e-10",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_MIN_IMPROVEMENT": "1e-3",
    }

    policy = read_post_residual_equation_policy(env=env)

    assert policy.steps_requested == 2
    assert policy.max_directions == 96
    assert policy.max_extra_units == 12
    assert policy.fsavg_lmax == 5
    assert policy.angular_lmax == 3
    assert policy.include_angular_residual is False
    assert policy.include_raw is False
    assert policy.include_post_coarse is False
    assert policy.include_qi_basis is False
    assert policy.alpha_clip == 4.0
    assert policy.rcond == 1.0e-10
    assert policy.min_improvement == 1.0e-3


def test_post_minres_policy_invalid_values_fail_to_defaults_or_bounds() -> None:
    env = {
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS": "bad",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_ALPHA_CLIP": "-2.0",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_MIN_IMPROVEMENT": "bad",
    }

    policy = read_post_minres_policy(env=env)

    assert policy.steps_requested == 0
    assert policy.alpha_clip == 0.0
    assert policy.min_improvement == 0.0


def test_post_solve_policy_groups_post_correction_controls() -> None:
    env = {
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS": "3",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE": "yes",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE_STEPS": "2",
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION": "true",
    }

    policy = read_post_solve_correction_policy(env=env)

    assert isinstance(policy, RHS1PostSolveCorrectionPolicy)
    assert policy.post_minres.steps_requested == 3
    assert policy.post_coarse.steps_requested == 2
    assert policy.post_coarse.max_directions == 16
    assert policy.post_residual_equation.steps_requested == 1
    assert policy.post_residual_equation.max_directions == 64
