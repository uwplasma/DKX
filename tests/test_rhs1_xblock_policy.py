from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_qi_seed_robustness import build_evidence_manifest
from sfincs_jax.rhs1_xblock_policy import (
    DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE,
    DEFAULT_FULL_FP_3D_LGMRES_RESCUE_MAXITER,
    DEFAULT_FULL_FP_3D_LGMRES_RESCUE_OUTER_K,
    DEFAULT_FULL_FP_3D_RIGHT_PC_MAX_ACTIVE_SIZE,
    DEFAULT_FULL_FP_3D_SIDE_PROBE_MIN_ACTIVE_SIZE,
    DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP,
    DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL,
    DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL,
    DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR,
    DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL,
    DEFAULT_RHS1_XBLOCK_LOCAL_ROW_NNZ_CAP,
    DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO,
    DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT,
    DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR,
    DEFAULT_RHS1_XBLOCK_LOWER_FILL_ILU_DROP_TOL,
    DEFAULT_RHS1_XBLOCK_LOWER_FILL_ROW_NNZ_CAP,
    resolve_rhs1_xblock_sparse_pc_policy,
    rhs1_xblock_device_host_fallback_decision,
    rhs1_xblock_gmres_restart,
    rhs1_xblock_krylov_method,
    rhs1_xblock_local_solve_candidate,
    rhs1_xblock_local_solve_metadata_label,
    rhs1_xblock_lower_fill_acceptance_decision,
    rhs1_xblock_lower_fill_mode,
    rhs1_xblock_lgmres_rescue_backend_allowed,
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
        {"krylov_method": "gmres_jax"},
        {"krylov_method": "fgmres_jax"},
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
    assert rhs1_xblock_side_probe_enabled(
        env_value="1",
        explicit_side_env_value="right",
        full_fp_3d_pc=False,
        active_size=1,
        min_active_size_env_value="999999",
        krylov_method="fgmres_jax",
        precondition_side="right",
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


def test_lgmres_rescue_backend_guard_keeps_default_off_gpu() -> None:
    assert rhs1_xblock_lgmres_rescue_backend_allowed(backend="cpu", env_value="")
    assert not rhs1_xblock_lgmres_rescue_backend_allowed(backend="gpu", env_value="")
    assert not rhs1_xblock_lgmres_rescue_backend_allowed(backend="cuda", env_value="0")
    assert rhs1_xblock_lgmres_rescue_backend_allowed(backend="gpu", env_value="1")


def test_device_host_fallback_defaults_to_large_qi_device_requests_only() -> None:
    decision = rhs1_xblock_device_host_fallback_decision(
        env_value="",
        requested_krylov_method="fgmres_jax",
        active_size=DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE,
        min_active_size_env_value="",
        rhs_mode=1,
        constraint_scheme=1,
        include_phi1=False,
        has_fp=True,
        has_pas=False,
        n_zeta=11,
    )

    assert decision.used
    assert decision.reason == "large-qi-full-fp-3d"
    assert decision.effective_krylov_env_value == "auto"
    assert decision.non_autodiff
    assert decision.to_metadata()["used"] is True

    blocked = rhs1_xblock_device_host_fallback_decision(
        env_value="",
        requested_krylov_method="gmres",
        active_size=1_000_000,
        min_active_size_env_value="",
        rhs_mode=1,
        constraint_scheme=1,
        include_phi1=False,
        has_fp=True,
        has_pas=False,
        n_zeta=11,
    )
    assert not blocked.used
    assert blocked.reason == "not-device-krylov"

    small = rhs1_xblock_device_host_fallback_decision(
        env_value="",
        requested_krylov_method="gmres_jax",
        active_size=DEFAULT_FULL_FP_3D_DEVICE_HOST_FALLBACK_MIN_ACTIVE_SIZE - 1,
        min_active_size_env_value="",
        rhs_mode=1,
        constraint_scheme=1,
        include_phi1=False,
        has_fp=True,
        has_pas=False,
        n_zeta=11,
    )
    assert not small.used
    assert small.reason == "below-active-size-floor"


def test_device_host_fallback_respects_force_disable_and_invalid_env() -> None:
    base = {
        "requested_krylov_method": "gmres_jax",
        "active_size": 1,
        "min_active_size_env_value": "1000000",
        "rhs_mode": 1,
        "constraint_scheme": 1,
        "include_phi1": False,
        "has_fp": True,
        "has_pas": False,
        "n_zeta": 3,
    }

    forced = rhs1_xblock_device_host_fallback_decision(env_value="host", **base)
    assert forced.used
    assert forced.mode == "force"
    assert forced.reason == "forced"

    disabled = rhs1_xblock_device_host_fallback_decision(env_value="0", **base)
    assert not disabled.used
    assert disabled.mode == "off"
    assert disabled.reason == "disabled"

    ignored = rhs1_xblock_device_host_fallback_decision(
        env_value="surprising",
        active_size=1_000_000,
        min_active_size_env_value="",
        requested_krylov_method="tfqmr_jax",
        rhs_mode=1,
        constraint_scheme=1,
        include_phi1=False,
        has_fp=True,
        has_pas=False,
        n_zeta=3,
    )
    assert ignored.used
    assert ignored.mode == "auto"
    assert ignored.ignored_env


@pytest.mark.parametrize(
    ("env_value", "expected_mode", "expected_ignored"),
    [
        ("", "off", False),
        ("default", "off", False),
        ("0", "off", False),
        ("legacy", "off", False),
        ("1", "probe", False),
        ("lower-fill", "probe", False),
        ("bounded-ilu", "probe", False),
        ("force", "force", False),
        ("required", "force", False),
        ("surprising", "off", True),
    ],
)
def test_lower_fill_mode_parses_supported_env_tokens(
    env_value: str,
    expected_mode: str,
    expected_ignored: bool,
) -> None:
    assert rhs1_xblock_lower_fill_mode(env_value) == (expected_mode, expected_ignored)


def test_local_solve_candidate_preserves_legacy_defaults_and_labels() -> None:
    exact_candidate = rhs1_xblock_local_solve_candidate(block_size=2_000, lu_max=3_000)

    assert exact_candidate.factorization == "lu"
    assert exact_candidate.exact_lu
    assert not exact_candidate.lower_fill
    assert exact_candidate.selection_reason == "legacy-exact-lu"
    assert exact_candidate.metadata_label == "rhs1_xblock_local_exact_lu"
    assert exact_candidate.tuning.drop_tol == DEFAULT_RHS1_XBLOCK_LOCAL_DROP_TOL
    assert exact_candidate.tuning.drop_rel == DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL
    assert exact_candidate.tuning.ilu_drop_tol == DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL
    assert exact_candidate.tuning.fill_factor == DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR
    assert exact_candidate.tuning.row_nnz_cap == DEFAULT_RHS1_XBLOCK_LOCAL_ROW_NNZ_CAP
    assert exact_candidate.tuning.compact_row_nnz_cap == DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP

    ilu_candidate = rhs1_xblock_local_solve_candidate(block_size=4_000, lu_max=3_000)
    metadata = ilu_candidate.to_metadata()

    assert ilu_candidate.factorization == "ilu"
    assert not ilu_candidate.exact_lu
    assert not ilu_candidate.lower_fill
    assert ilu_candidate.selection_reason == "legacy-ilu"
    assert metadata["metadata_label"] == "rhs1_xblock_local_ilu"
    assert metadata["fill_factor"] == DEFAULT_RHS1_XBLOCK_LOCAL_FILL_FACTOR
    assert rhs1_xblock_local_solve_metadata_label(factorization="lu", lower_fill=False) == (
        "rhs1_xblock_local_exact_lu"
    )


def test_lower_fill_candidate_uses_bounded_env_overrides() -> None:
    candidate = rhs1_xblock_local_solve_candidate(
        block_size=4_000,
        lu_max=3_000,
        lower_fill_env_value="1",
        lower_fill_drop_tol_env_value="1e-7",
        lower_fill_drop_rel_env_value="2e-8",
        lower_fill_ilu_drop_tol_env_value="5e-3",
        lower_fill_factor_env_value="2.5",
        lower_fill_row_nnz_cap_env_value="24",
        lower_fill_compact_row_nnz_cap_env_value="12",
    )

    assert candidate.mode == "probe"
    assert candidate.factorization == "ilu"
    assert not candidate.exact_lu
    assert candidate.lower_fill
    assert candidate.lower_fill_requested
    assert candidate.selection_reason == "lower-fill-requested"
    assert candidate.metadata_label == "rhs1_xblock_local_lower_fill_ilu"
    assert candidate.tuning.drop_tol == pytest.approx(1.0e-7)
    assert candidate.tuning.drop_rel == pytest.approx(2.0e-8)
    assert candidate.tuning.ilu_drop_tol == pytest.approx(5.0e-3)
    assert candidate.tuning.fill_factor == pytest.approx(2.5)
    assert candidate.tuning.row_nnz_cap == 24
    assert candidate.tuning.compact_row_nnz_cap == 12


def test_lower_fill_candidate_keeps_exact_lu_window_unless_forced() -> None:
    candidate = rhs1_xblock_local_solve_candidate(
        block_size=2_000,
        lu_max=3_000,
        lower_fill_env_value="1",
    )

    assert candidate.factorization == "lu"
    assert candidate.exact_lu
    assert not candidate.lower_fill
    assert candidate.lower_fill_requested
    assert candidate.selection_reason == "exact-lu-within-lu-max"

    forced = rhs1_xblock_local_solve_candidate(
        block_size=2_000,
        lu_max=3_000,
        lower_fill_env_value="force",
    )

    assert forced.factorization == "ilu"
    assert not forced.exact_lu
    assert forced.lower_fill
    assert forced.selection_reason == "lower-fill-forced"
    assert forced.tuning.ilu_drop_tol == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ILU_DROP_TOL
    assert forced.tuning.fill_factor == DEFAULT_RHS1_XBLOCK_LOWER_FILL_FACTOR
    assert forced.tuning.row_nnz_cap == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ROW_NNZ_CAP


def test_local_solve_candidate_handles_invalid_values_and_caps() -> None:
    ignored = rhs1_xblock_local_solve_candidate(
        block_size=4_000,
        lu_max=3_000,
        lower_fill_env_value="unknown",
        drop_tol_env_value="-1",
        drop_rel_env_value="bad",
        ilu_drop_tol_env_value="nan",
        fill_factor_env_value="0.5",
        row_nnz_cap_env_value="-9",
        compact_row_nnz_cap_env_value="bad",
    )

    assert ignored.mode == "off"
    assert ignored.ignored_lower_fill_env
    assert ignored.factorization == "ilu"
    assert ignored.selection_reason == "lower-fill-env-ignored"
    assert ignored.tuning.drop_tol == 0.0
    assert ignored.tuning.drop_rel == DEFAULT_RHS1_XBLOCK_LOCAL_DROP_REL
    assert ignored.tuning.ilu_drop_tol == DEFAULT_RHS1_XBLOCK_LOCAL_ILU_DROP_TOL
    assert ignored.tuning.fill_factor == 1.0
    assert ignored.tuning.row_nnz_cap == 0
    assert ignored.tuning.compact_row_nnz_cap == DEFAULT_RHS1_XBLOCK_LOCAL_COMPACT_ROW_NNZ_CAP

    capped = rhs1_xblock_local_solve_candidate(
        block_size=250_001,
        lu_max=3_000,
        lower_fill_env_value="1",
        lower_fill_max_block_size_env_value="250000",
    )

    assert not capped.lower_fill
    assert capped.lower_fill_block_size_capped
    assert capped.selection_reason == "lower-fill-block-size-cap-exceeded"

    uncapped = rhs1_xblock_local_solve_candidate(
        block_size=250_001,
        lu_max=3_000,
        lower_fill_env_value="1",
        lower_fill_max_block_size_env_value="0",
        lower_fill_factor_env_value="0.25",
        lower_fill_row_nnz_cap_env_value="-4",
    )

    assert uncapped.lower_fill
    assert not uncapped.lower_fill_block_size_capped
    assert uncapped.tuning.fill_factor == 1.0
    assert uncapped.tuning.row_nnz_cap == 0


def test_lower_fill_acceptance_accepts_bounded_candidate_and_metadata() -> None:
    decision = rhs1_xblock_lower_fill_acceptance_decision(
        factorization_ok=True,
        residual_norm=5.0e-7,
        target=1.0e-8,
        baseline_residual_norm=1.0e-6,
        factor_probe_ratio=10.0,
    )
    metadata = decision.to_metadata()

    assert decision.accepted
    assert decision.reason == "accepted"
    assert decision.residual_ratio == pytest.approx(50.0)
    assert decision.improvement == pytest.approx(2.0)
    assert decision.factor_probe_ratio == pytest.approx(10.0)
    assert decision.max_residual_ratio == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO
    assert decision.min_improvement == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT
    assert metadata["metadata_label"] == "rhs1_xblock_lower_fill_acceptance"
    assert metadata["candidate_metadata_label"] == "rhs1_xblock_local_lower_fill_ilu"


def test_lower_fill_acceptance_rejects_failed_or_unsafe_candidates() -> None:
    assert (
        rhs1_xblock_lower_fill_acceptance_decision(
            factorization_ok=False,
            residual_norm=1.0e-9,
            target=1.0e-8,
        ).reason
        == "factorization-failed"
    )
    assert (
        rhs1_xblock_lower_fill_acceptance_decision(
            factorization_ok=True,
            residual_norm=float("nan"),
            target=1.0e-8,
        ).reason
        == "nonfinite-residual"
    )
    assert (
        rhs1_xblock_lower_fill_acceptance_decision(
            factorization_ok=True,
            residual_norm=2.0e-6,
            target=1.0e-8,
        ).reason
        == "residual-ratio-limit-exceeded"
    )
    assert (
        rhs1_xblock_lower_fill_acceptance_decision(
            factorization_ok=True,
            residual_norm=8.0e-7,
            target=1.0e-8,
            baseline_residual_norm=1.0e-6,
            min_improvement_env_value="2",
        ).reason
        == "insufficient-improvement"
    )
    assert (
        rhs1_xblock_lower_fill_acceptance_decision(
            factorization_ok=True,
            residual_norm=1.0e-9,
            target=1.0e-8,
            factor_probe_ratio=101.0,
            factor_probe_max_env_value="100",
        ).reason
        == "factor-probe-limit-exceeded"
    )


def test_lower_fill_acceptance_invalid_gate_env_values_fall_back_to_defaults() -> None:
    decision = rhs1_xblock_lower_fill_acceptance_decision(
        factorization_ok=True,
        residual_norm=5.0e-7,
        target=1.0e-8,
        baseline_residual_norm=1.0e-6,
        factor_probe_ratio=10.0,
        max_residual_ratio_env_value="bad",
        min_improvement_env_value="bad",
        factor_probe_max_env_value="bad",
    )

    assert decision.accepted
    assert decision.max_residual_ratio == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MAX_RESIDUAL_RATIO
    assert decision.min_improvement == DEFAULT_RHS1_XBLOCK_LOWER_FILL_ACCEPT_MIN_IMPROVEMENT


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
        ("gmres-jax", "gmres_jax"),
        ("device-gmres", "gmres_jax"),
        ("bicgstab-jax", "bicgstab_jax"),
        ("device-bicgstab", "bicgstab_jax"),
        ("short-recurrence-jax", "bicgstab_jax"),
        ("tfqmr", "tfqmr_jax"),
        ("tfqmr-jax", "tfqmr_jax"),
        ("device-tfqmr", "tfqmr_jax"),
        ("transpose-free-qmr", "tfqmr_jax"),
        ("fgmres", "fgmres_jax"),
        ("flexible-gmres", "fgmres_jax"),
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
    method, ignored = rhs1_xblock_krylov_method("idr")

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
        krylov_env_value="idr",
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


def test_qi_evidence_manifest_separates_host_fallback_from_true_device_qi(tmp_path: Path) -> None:
    source_input = tmp_path / "input.namelist"
    source_input.write_text(
        (
            "&resolutionParameters\n"
            "  Ntheta = 25\n"
            "  Nzeta = 51\n"
            "  Nx = 8\n"
            "  Nxi = 100\n"
            "/\n"
        ),
        encoding="utf-8",
    )
    host_fallback = tmp_path / "qi_seed_robustness_scale060_device_host_fallback_seed3_cpu_2026_05_15.json"
    host_fallback.write_text(
        json.dumps(
            {
                "artifact_kind": "qi_seed_execution_summary",
                "schema_version": 2,
                "lane": "qi_seed_robustness",
                "case_count": 1,
                "public_cli_default_path": False,
                "resolution": {"NTHETA": 15, "NZETA": 31, "NX": 5, "NXI": 60},
                "total_size_estimate": 139502,
                "active_size": 81377,
                "execution_summary": {
                    "backends": ["cpu"],
                    "max_elapsed_s": 156.8,
                    "max_residual_ratio": 0.0034,
                    "process_failed": 0,
                    "timed_out": 0,
                },
                "gates": {"passed": True, "failures": []},
            }
        ),
        encoding="utf-8",
    )
    device_timeout = tmp_path / "qi_seed_robustness_scale060_galerkin_forced_xblock_seed3_gpu1_2026_05_15.json"
    device_timeout.write_text(
        json.dumps(
            {
                "artifact_kind": "qi_seed_execution_summary",
                "schema_version": 2,
                "lane": "qi_seed_robustness",
                "case_count": 1,
                "public_cli_default_path": False,
                "resolution": {"NTHETA": 15, "NZETA": 31, "NX": 5, "NXI": 60},
                "total_size_estimate": 139502,
                "active_size": 81377,
                "execution_summary": {
                    "backends": [],
                    "max_elapsed_s": 600.3,
                    "max_residual_ratio": None,
                    "process_failed": 1,
                    "timed_out": 1,
                },
                "gates": {"passed": False, "failures": [{"reason": "process_failed"}]},
            }
        ),
        encoding="utf-8",
    )

    manifest = build_evidence_manifest(
        artifact_paths=[host_fallback, device_timeout],
        source_input=source_input,
        production_seed_count=5,
        production_timeout_s=3600.0,
    )

    assert manifest["release_gate"] == "bounded_proxy"
    claims = manifest["release_claims"]
    assert claims["production_non_autodiff_host_fallback"]["claim_status"] == "release_ready"
    assert claims["production_non_autodiff_host_fallback"]["blocks_current_release"] is False
    assert "not a differentiable or true device-resident" in claims["production_non_autodiff_host_fallback"]["scope"]
    assert claims["true_device_qi"]["claim_status"] == "closed_deferred"
    assert claims["true_device_qi"]["blocks_current_release"] is False
    assert "post-release" in claims["true_device_qi"]["closed_or_deferred_reason"]
    assert str(device_timeout.name) in claims["true_device_qi"]["evidence"][0]
    assert claims["public_auto_production_ladder"]["claim_status"] == "bounded_proxy"
    assert any("Do not use the non-autodiff host fallback" in blocker for blocker in manifest["open_blockers"])
