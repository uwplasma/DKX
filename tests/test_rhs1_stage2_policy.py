from __future__ import annotations

from sfincs_jax.rhs1_stage2_policy import (
    RHS1Stage2AdmissionControls,
    RHS1Stage2RetryControls,
    rhs1_fp_force_stage2,
    rhs1_pas_stage2_skip,
    rhs1_pas_tz_guarded_stage2_retry,
    rhs1_stage2_admission_controls_from_env,
    rhs1_stage2_ratio,
    rhs1_stage2_retry_controls_from_env,
    rhs1_stage2_trigger,
)


def test_rhs1_stage2_ratio_handles_invalid_env_and_dkes_tightening(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_RATIO", "bad")
    assert rhs1_stage2_ratio(use_dkes=False) == 1.0e2
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_RATIO", "50")
    assert rhs1_stage2_ratio(use_dkes=False) == 50.0
    assert rhs1_stage2_ratio(use_dkes=True) == 1.0


def test_rhs1_stage2_trigger_uses_ratio_policy(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_RATIO", "10")
    assert rhs1_stage2_trigger(res_ratio=11.0, use_dkes=False)
    assert not rhs1_stage2_trigger(res_ratio=9.0, use_dkes=False)
    assert rhs1_stage2_trigger(res_ratio=1.1, use_dkes=True)
    assert not rhs1_stage2_trigger(res_ratio=0.9, use_dkes=True)


def test_rhs1_stage2_admission_controls_preserve_default_gate(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2", raising=False)
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", raising=False)

    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ) == RHS1Stage2AdmissionControls(enabled=True, time_cap_s=30.0)

    assert not rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=True,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).enabled

    assert not rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="dense",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).enabled


def test_rhs1_stage2_admission_controls_respect_env_and_fastpath(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2", "off")
    assert not rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).enabled

    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2", "on")
    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=2,
        include_phi1=True,
        solver_kind_default="dense",
        pas_large_bicgstab_fastpath=True,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).enabled

    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2", raising=False)
    assert not rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="bicgstab",
        pas_large_bicgstab_fastpath=True,
        tokamak_pas=False,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).enabled


def test_rhs1_stage2_admission_controls_raise_time_budget_floors(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", "10")

    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=True,
        has_fp=False,
        use_dkes=False,
        total_size=1000,
    ).time_cap_s == 120.0

    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=True,
        use_dkes=True,
        total_size=1000,
    ).time_cap_s == 120.0

    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=True,
        use_dkes=False,
        total_size=300000,
    ).time_cap_s == 1200.0

    assert rhs1_stage2_admission_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        solver_kind_default="gmres",
        pas_large_bicgstab_fastpath=False,
        tokamak_pas=False,
        has_fp=True,
        use_dkes=False,
        total_size=600000,
    ).time_cap_s == 2400.0


def test_rhs1_stage2_admission_controls_preserve_invalid_time_cap_error(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_MAX_ELAPSED_S", "bad")

    try:
        rhs1_stage2_admission_controls_from_env(
            rhs_mode=1,
            include_phi1=False,
            solver_kind_default="gmres",
            pas_large_bicgstab_fastpath=False,
            tokamak_pas=False,
            has_fp=False,
            use_dkes=False,
            total_size=1000,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - defensive clarity for this legacy contract
        raise AssertionError("invalid Stage-2 elapsed-time env should still raise ValueError")


def test_rhs1_stage2_retry_controls_preserve_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_MAXITER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_METHOD", raising=False)

    assert rhs1_stage2_retry_controls_from_env(
        restart=80,
        maxiter=300,
        active_size=1000,
        has_fp=False,
        has_pas=False,
    ) == RHS1Stage2RetryControls(restart=120, maxiter=600, method="incremental")

    assert rhs1_stage2_retry_controls_from_env(
        restart=160,
        maxiter=500,
        active_size=1000,
        has_fp=False,
        has_pas=False,
    ) == RHS1Stage2RetryControls(restart=160, maxiter=1000, method="incremental")


def test_rhs1_stage2_retry_controls_tokamak_pas_and_large_fp_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_MAXITER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_RESTART", raising=False)

    assert rhs1_stage2_retry_controls_from_env(
        restart=80,
        maxiter=300,
        active_size=1000,
        has_fp=False,
        has_pas=True,
        tokamak_pas=True,
    ) == RHS1Stage2RetryControls(restart=160, maxiter=2000, method="incremental")

    assert rhs1_stage2_retry_controls_from_env(
        restart=160,
        maxiter=500,
        active_size=300000,
        has_fp=True,
        has_pas=False,
    ) == RHS1Stage2RetryControls(restart=100, maxiter=600, method="incremental")


def test_rhs1_stage2_retry_controls_respect_user_caps_and_method(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_MAXITER", "77")
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_RESTART", "33")
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_METHOD", "dense")

    assert rhs1_stage2_retry_controls_from_env(
        restart=80,
        maxiter=300,
        active_size=300000,
        has_fp=True,
        has_pas=False,
        tokamak_pas=True,
    ) == RHS1Stage2RetryControls(restart=33, maxiter=77, method="dense")

    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_METHOD", "bad")
    assert (
        rhs1_stage2_retry_controls_from_env(
            restart=80,
            maxiter=300,
            active_size=1000,
            has_fp=False,
            has_pas=False,
        ).method
        == "incremental"
    )


def test_rhs1_stage2_retry_controls_preserve_invalid_integer_errors(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_LINEAR_STAGE2_MAXITER", "bad")
    monkeypatch.delenv("SFINCS_JAX_LINEAR_STAGE2_RESTART", raising=False)

    try:
        rhs1_stage2_retry_controls_from_env(
            restart=80,
            maxiter=300,
            active_size=1000,
            has_fp=False,
            has_pas=False,
        )
    except ValueError:
        pass
    else:  # pragma: no cover - defensive clarity for this legacy contract
        raise AssertionError("invalid Stage-2 maxiter env should still raise ValueError")


def test_rhs1_fp_force_stage2_respects_abs_floor_and_include_phi1(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_FP_STAGE2_ABS", "bad")
    assert rhs1_fp_force_stage2(has_fp=True, include_phi1=False, residual_norm=2.0e-6)
    assert not rhs1_fp_force_stage2(has_fp=False, include_phi1=False, residual_norm=1.0)
    assert not rhs1_fp_force_stage2(has_fp=True, include_phi1=True, residual_norm=1.0)
    monkeypatch.setenv("SFINCS_JAX_FP_STAGE2_ABS", "1e-3")
    assert not rhs1_fp_force_stage2(has_fp=True, include_phi1=False, residual_norm=1.0e-4)


def test_rhs1_pas_stage2_skip_respects_kind_and_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "bad")
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="pas_lite", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="pas_ilu", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="schur", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="xblock_tz", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="xblock_tz_lmax", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="theta_line", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=False, rhs1_precond_kind="pas_lite", res_ratio=1.0e7)
    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "10")
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="pas_hybrid", res_ratio=11.0)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="pas_hybrid", res_ratio=9.0)


def test_rhs1_pas_stage2_skip_extended_is_opt_in(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_SKIP_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_SKIP_EXTENDED", "1")
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="pas_ilu", res_ratio=1.0e7)
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="schur", res_ratio=1.0e7)
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="xblock_tz", res_ratio=1.0e7)
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="xblock_tz_lmax", res_ratio=1.0e7)
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="theta_line", res_ratio=1.0e7)


def test_rhs1_pas_stage2_skip_weak_preconditioners_only_for_huge_ratios(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO", raising=False)
    for kind in ("collision", "point", "xmg"):
        assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind=kind, res_ratio=1.0e7)
        assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind=kind, res_ratio=1.0e13)

    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO", "0")
    assert not rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="collision", res_ratio=1.0e99)

    monkeypatch.setenv("SFINCS_JAX_PAS_STAGE2_WEAK_SKIP_RATIO", "bad")
    assert rhs1_pas_stage2_skip(has_pas=True, rhs1_precond_kind="xmg", res_ratio=1.0e13)


def test_rhs1_pas_tz_guarded_stage2_retry_is_explicit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY", raising=False)
    assert not rhs1_pas_tz_guarded_stage2_retry()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY", "1")
    assert rhs1_pas_tz_guarded_stage2_retry()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STAGE2_RETRY", "false")
    assert not rhs1_pas_tz_guarded_stage2_retry()
