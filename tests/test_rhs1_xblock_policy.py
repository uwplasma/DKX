from __future__ import annotations

import pytest

from sfincs_jax.rhs1_xblock_policy import (
    DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER,
    DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K,
    DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE,
    DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE,
    resolve_rhs1_xblock_sparse_pc_policy,
    rhs1_xblock_gmres_restart,
    rhs1_xblock_krylov_method,
    rhs1_xblock_lgmres_rescue_enabled,
    rhs1_xblock_lgmres_rescue_maxiter,
    rhs1_xblock_lgmres_rescue_outer_k,
    rhs1_xblock_precondition_side,
    rhs1_xblock_side_probe_enabled,
    rhs1_xblock_side_probe_should_switch,
)


@pytest.mark.parametrize("env_value", ["left", " LEFT ", "right", "none"])
def test_precondition_side_respects_explicit_env_overrides(env_value: str) -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value=env_value,
        tokamak_fp_er_pc=True,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )

    assert side == env_value.strip().lower()
    assert not auto_right


def test_precondition_side_defaults_right_only_for_measured_full_fp_er_path() -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        use_dkes=False,
        include_xdot=False,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("right", True)

    churn_guards = (
        {
            "tokamak_fp_er_pc": False,
            "use_dkes": False,
            "include_xdot": True,
            "include_electric_field_xi": False,
        },
        {
            "tokamak_fp_er_pc": True,
            "use_dkes": True,
            "include_xdot": True,
            "include_electric_field_xi": True,
        },
        {
            "tokamak_fp_er_pc": True,
            "use_dkes": False,
            "include_xdot": False,
            "include_electric_field_xi": False,
        },
    )
    for kwargs in churn_guards:
        side, auto_right = rhs1_xblock_precondition_side(env_value="", **kwargs)
        assert (side, auto_right) == ("left", False)


def test_precondition_side_uses_size_window_for_full_fp_3d_path() -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE + 1,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)


def test_precondition_side_allows_full_fp_3d_size_window_override() -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=52_637,
        full_fp_3d_right_pc_max_env_value="70000",
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=39_314,
        full_fp_3d_right_pc_max_env_value="not-an-int",
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("right", True)


def test_precondition_side_keeps_tokamak_er_right_pc_independent_of_size_window() -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=False,
        active_size=1_000_000,
        full_fp_3d_right_pc_max_env_value="0",
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("right", True)


def test_side_probe_defaults_only_for_large_full_fp_3d_auto_gmres() -> None:
    assert rhs1_xblock_side_probe_enabled(
        env_value="",
        explicit_side_env_value="",
        full_fp_3d_pc=True,
        active_size=DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE,
        min_active_size_env_value="",
        krylov_method="gmres",
        precondition_side="left",
    )

    blocked_cases = (
        {"active_size": DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE - 1},
        {"full_fp_3d_pc": False},
        {"explicit_side_env_value": "right"},
        {"krylov_method": "lgmres"},
        {"precondition_side": "none"},
    )
    base = {
        "env_value": "",
        "explicit_side_env_value": "",
        "full_fp_3d_pc": True,
        "active_size": DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE,
        "min_active_size_env_value": "",
        "krylov_method": "gmres",
        "precondition_side": "left",
    }
    for override in blocked_cases:
        kwargs = {**base, **override}
        assert not rhs1_xblock_side_probe_enabled(**kwargs)


def test_side_probe_respects_env_overrides_and_switch_threshold() -> None:
    assert rhs1_xblock_side_probe_enabled(
        env_value="1",
        explicit_side_env_value="",
        full_fp_3d_pc=False,
        active_size=1,
        min_active_size_env_value="999999",
        krylov_method="gmres",
        precondition_side="left",
    )
    assert not rhs1_xblock_side_probe_enabled(
        env_value="0",
        explicit_side_env_value="",
        full_fp_3d_pc=True,
        active_size=1_000_000,
        min_active_size_env_value="",
        krylov_method="gmres",
        precondition_side="left",
    )
    assert rhs1_xblock_side_probe_should_switch(residual_ratio=5_001.0, switch_ratio_env_value="")
    assert not rhs1_xblock_side_probe_should_switch(residual_ratio=4_999.0, switch_ratio_env_value="")
    assert rhs1_xblock_side_probe_should_switch(residual_ratio=11.0, switch_ratio_env_value="10")
    assert not rhs1_xblock_side_probe_should_switch(residual_ratio=None, switch_ratio_env_value="10")


def test_lgmres_rescue_respects_explicit_krylov_method_and_caps_maxiter() -> None:
    assert rhs1_xblock_lgmres_rescue_enabled(env_value="", krylov_env_value="")
    assert rhs1_xblock_lgmres_rescue_enabled(env_value="", krylov_env_value="auto")
    assert rhs1_xblock_lgmres_rescue_enabled(env_value="1", krylov_env_value="gmres")
    assert not rhs1_xblock_lgmres_rescue_enabled(env_value="", krylov_env_value="gmres")
    assert not rhs1_xblock_lgmres_rescue_enabled(env_value="0", krylov_env_value="")

    selected, capped = rhs1_xblock_lgmres_rescue_maxiter("", current_maxiter=400)
    assert selected == DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER
    assert capped
    selected, capped = rhs1_xblock_lgmres_rescue_maxiter("96", current_maxiter=400)
    assert selected == 96
    assert capped
    selected, capped = rhs1_xblock_lgmres_rescue_maxiter("", current_maxiter=32)
    assert selected == 32
    assert not capped
    assert rhs1_xblock_lgmres_rescue_outer_k("") == DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K
    assert rhs1_xblock_lgmres_rescue_outer_k("12") == 12
    assert rhs1_xblock_lgmres_rescue_outer_k("bad") == DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K


def test_invalid_precondition_side_falls_back_to_default_policy() -> None:
    side, auto_right = rhs1_xblock_precondition_side(
        env_value="bogus",
        tokamak_fp_er_pc=True,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )

    assert (side, auto_right) == ("right", True)


@pytest.mark.parametrize(
    ("env_value", "expected"),
    [
        ("", "gmres"),
        ("default", "gmres"),
        (" AUTO ", "gmres"),
        ("gmres", "gmres"),
        ("lgmres", "lgmres"),
        ("lgmres-scipy", "lgmres"),
        ("short-recurrence", "bicgstab"),
        ("shortrecurrence", "bicgstab"),
        ("BiCGStab", "bicgstab"),
    ],
)
def test_krylov_method_canonicalizes_supported_aliases(env_value: str, expected: str) -> None:
    method, ignored = rhs1_xblock_krylov_method(env_value)

    assert method == expected
    assert not ignored


def test_krylov_method_reports_unknown_values_without_changing_default() -> None:
    method, ignored = rhs1_xblock_krylov_method("tfqmr")

    assert method == "gmres"
    assert ignored


def test_gmres_restart_caps_only_auto_right_preconditioned_gmres() -> None:
    restart, capped = rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (20, True)

    restart, capped = rhs1_xblock_gmres_restart(
        requested_restart=12,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (12, False)

    for kwargs in (
        {
            "restart_env_value": "40",
            "krylov_method": "gmres",
            "default_right_preconditioned": True,
        },
        {
            "restart_env_value": "",
            "krylov_method": "lgmres",
            "default_right_preconditioned": True,
        },
        {
            "restart_env_value": "",
            "krylov_method": "gmres",
            "default_right_preconditioned": False,
        },
    ):
        restart, capped = rhs1_xblock_gmres_restart(requested_restart=80, **kwargs)
        assert (restart, capped) == (80, False)


@pytest.mark.parametrize("requested_restart", [0, -5])
def test_gmres_restart_enforces_positive_floor_before_policy(requested_restart: int) -> None:
    restart, capped = rhs1_xblock_gmres_restart(
        requested_restart=requested_restart,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )

    assert (restart, capped) == (1, False)


def test_resolve_xblock_sparse_pc_policy_combines_driver_decisions() -> None:
    policy = resolve_rhs1_xblock_sparse_pc_policy(
        precondition_side_env_value="",
        krylov_env_value="short-recurrence",
        requested_restart=80,
        restart_env_value="",
        tokamak_fp_er_pc=True,
        active_size=1_000_000,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )

    assert policy.precondition_side == "right"
    assert policy.default_right_preconditioned
    assert policy.krylov_method == "bicgstab"
    assert not policy.ignored_krylov_env
    assert policy.gmres_restart == 80
    assert not policy.restart_capped


def test_resolve_xblock_sparse_pc_policy_preserves_unknown_krylov_warning_bit() -> None:
    policy = resolve_rhs1_xblock_sparse_pc_policy(
        precondition_side_env_value="right",
        krylov_env_value="tfqmr",
        requested_restart=80,
        restart_env_value="",
        tokamak_fp_er_pc=False,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )

    assert policy.precondition_side == "right"
    assert not policy.default_right_preconditioned
    assert policy.krylov_method == "gmres"
    assert policy.ignored_krylov_env
    assert policy.gmres_restart == 80
    assert not policy.restart_capped
