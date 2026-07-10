from __future__ import annotations

from sfincs_jax.problems.profile_policies import (
    read_bool_env,
    read_float_env,
    read_int_env,
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
