from __future__ import annotations

from sfincs_jax.problems.profile_response.preconditioner_build import (
    RHS1FPStrongSizeGuard,
    RHS1StrongRetryControls,
    RHS1StrongTriggerControls,
    rhs1_collision_retry_allowed,
    rhs1_fp_strong_size_guard_from_env,
    rhs1_pas_force_strong_ratio_from_env,
    rhs1_resolved_strong_preconditioner_control,
    rhs1_strong_preconditioner_env_from_env,
    rhs1_strong_preconditioner_control_messages,
    rhs1_strong_retry_controls_from_env,
    rhs1_strong_preconditioner_min_size,
    rhs1_strong_trigger_controls_from_env,
)


def test_rhs1_strong_preconditioner_min_size_handles_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "bad")
    assert rhs1_strong_preconditioner_min_size() == 800
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "1200")
    assert rhs1_strong_preconditioner_min_size() == 1200


def test_rhs1_strong_preconditioner_env_and_pas_force_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", raising=False)
    assert rhs1_strong_preconditioner_env_from_env() == ""
    assert rhs1_pas_force_strong_ratio_from_env() == 50.0

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", " Theta_Line ")
    monkeypatch.setenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "12.5")
    assert rhs1_strong_preconditioner_env_from_env() == "theta_line"
    assert rhs1_pas_force_strong_ratio_from_env() == 12.5

    monkeypatch.setenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "bad")
    assert rhs1_pas_force_strong_ratio_from_env() == 50.0


def test_rhs1_strong_trigger_controls_preserve_default_ratio(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_FP_STRONG_ABS", raising=False)

    assert rhs1_strong_trigger_controls_from_env(
        residual_norm=2.0,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=False,
        rhs1_precond_kind="point",
        delay_pas_base_retries=False,
    ) == RHS1StrongTriggerControls(
        res_ratio=2.0,
        ratio_threshold=1.0,
        trigger=True,
        fp_force=False,
        fp_abs_threshold=1.0e-6,
    )


def test_rhs1_strong_trigger_controls_delay_pas_base_retries(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", raising=False)

    pas_controls = rhs1_strong_trigger_controls_from_env(
        residual_norm=50.0,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="pas_hybrid",
        delay_pas_base_retries=True,
    )
    assert pas_controls.ratio_threshold == 1.0e2
    assert not pas_controls.trigger

    tokamak_controls = rhs1_strong_trigger_controls_from_env(
        residual_norm=999.0,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="pas_tokamak_theta",
        delay_pas_base_retries=True,
    )
    assert tokamak_controls.ratio_threshold == 1.0e4
    assert not tokamak_controls.trigger


def test_rhs1_strong_trigger_controls_respect_explicit_and_invalid_ratio(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "0")
    assert rhs1_strong_trigger_controls_from_env(
        residual_norm=0.1,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="pas_hybrid",
        delay_pas_base_retries=True,
    ).trigger

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "bad")
    controls = rhs1_strong_trigger_controls_from_env(
        residual_norm=0.5,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="pas_hybrid",
        delay_pas_base_retries=True,
    )
    assert controls.ratio_threshold == 1.0
    assert not controls.trigger


def test_rhs1_strong_trigger_controls_fp_force(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_FP_STRONG_ABS", "bad")
    assert rhs1_strong_trigger_controls_from_env(
        residual_norm=2.0e-6,
        target=1.0,
        has_fp=True,
        include_phi1=False,
        has_pas=False,
        rhs1_precond_kind="point",
        delay_pas_base_retries=False,
    ).fp_force

    monkeypatch.setenv("SFINCS_JAX_FP_STRONG_ABS", "1e-3")
    assert not rhs1_strong_trigger_controls_from_env(
        residual_norm=2.0e-6,
        target=1.0,
        has_fp=True,
        include_phi1=False,
        has_pas=False,
        rhs1_precond_kind="point",
        delay_pas_base_retries=False,
    ).fp_force

    assert not rhs1_strong_trigger_controls_from_env(
        residual_norm=1.0,
        target=1.0,
        has_fp=True,
        include_phi1=True,
        has_pas=False,
        rhs1_precond_kind="point",
        delay_pas_base_retries=False,
    ).fp_force


def test_rhs1_strong_retry_controls_preserve_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", raising=False)

    assert rhs1_strong_retry_controls_from_env(
        restart=80,
        maxiter=300,
    ) == RHS1StrongRetryControls(restart=120, maxiter=800)

    assert rhs1_strong_retry_controls_from_env(
        restart=160,
        maxiter=500,
    ) == RHS1StrongRetryControls(restart=160, maxiter=1000)


def test_rhs1_strong_retry_controls_respect_env_and_invalid_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "44")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "88")
    assert rhs1_strong_retry_controls_from_env(
        restart=80,
        maxiter=300,
    ) == RHS1StrongRetryControls(restart=44, maxiter=88)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "bad")
    assert rhs1_strong_retry_controls_from_env(
        restart=80,
        maxiter=None,
    ) == RHS1StrongRetryControls(restart=120, maxiter=800)


def test_rhs1_collision_retry_allowed_requires_point_rhs1_nonphi1_and_trigger() -> None:
    kwargs = dict(
        residual_norm=2.0,
        target=1.0,
        rhs_mode=1,
        include_phi1=False,
        rhs1_precond_kind="point",
        has_fp=True,
        has_pas=False,
        strong_precond_trigger=True,
    )
    assert rhs1_collision_retry_allowed(**kwargs)
    assert not rhs1_collision_retry_allowed(**{**kwargs, "residual_norm": 0.5})
    assert not rhs1_collision_retry_allowed(**{**kwargs, "rhs_mode": 2})
    assert not rhs1_collision_retry_allowed(**{**kwargs, "include_phi1": True})
    assert not rhs1_collision_retry_allowed(**{**kwargs, "rhs1_precond_kind": "xmg"})
    assert not rhs1_collision_retry_allowed(**{**kwargs, "has_fp": False, "has_pas": False})
    assert not rhs1_collision_retry_allowed(**{**kwargs, "strong_precond_trigger": False})
    assert rhs1_collision_retry_allowed(**{**kwargs, "has_fp": False, "has_pas": True})


def test_rhs1_fp_strong_size_guard_preserves_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FP_STRONG_PRECOND_MAX", raising=False)

    assert rhs1_fp_strong_size_guard_from_env(
        active_size=120000,
        strong_precond_kind="xblock_tz",
        has_fp=True,
        has_pas=False,
    ) == RHS1FPStrongSizeGuard(skip=False, max_active_size=120000)
    assert rhs1_fp_strong_size_guard_from_env(
        active_size=120001,
        strong_precond_kind="xblock_tz",
        has_fp=True,
        has_pas=False,
    ) == RHS1FPStrongSizeGuard(skip=True, max_active_size=120000)


def test_rhs1_fp_strong_size_guard_respects_env_and_problem_guards(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_STRONG_PRECOND_MAX", "42")

    kwargs = dict(
        active_size=43,
        strong_precond_kind="theta_zeta",
        has_fp=True,
        has_pas=False,
    )
    assert rhs1_fp_strong_size_guard_from_env(**kwargs) == RHS1FPStrongSizeGuard(
        skip=True,
        max_active_size=42,
    )
    assert not rhs1_fp_strong_size_guard_from_env(**{**kwargs, "active_size": 42}).skip
    assert not rhs1_fp_strong_size_guard_from_env(**{**kwargs, "strong_precond_kind": "point"}).skip
    assert not rhs1_fp_strong_size_guard_from_env(**{**kwargs, "has_fp": False}).skip
    assert not rhs1_fp_strong_size_guard_from_env(**{**kwargs, "has_pas": True}).skip

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_STRONG_PRECOND_MAX", "bad")
    assert rhs1_fp_strong_size_guard_from_env(**kwargs) == RHS1FPStrongSizeGuard(
        skip=False,
        max_active_size=120000,
    )


def test_rhs1_resolved_strong_preconditioner_control_enables_auto_on_default_problem_families() -> None:
    control = rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        size=5000,
        n_theta=9,
        n_zeta=5,
    )
    assert not control.disabled
    assert control.auto


def test_rhs1_resolved_strong_preconditioner_control_tracks_disable_reasons(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "10")
    control = rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=5000,
        n_theta=9,
        n_zeta=5,
        cs0_sparse_first=True,
        large_cpu_sparse_rescue_first=True,
        pas_auto_skip=True,
        pas_fast_accept=True,
        pas_precond_force_collision=True,
        residual_norm=1.0,
        target=1.0,
    )
    assert control.disabled
    assert not control.auto
    assert control.reason_cs0_sparse_first
    assert control.reason_large_cpu_sparse_first
    assert control.reason_pas_auto_skip
    assert control.reason_pas_fast_accept
    assert control.reason_collision_probe_skip


def test_rhs1_resolved_strong_preconditioner_control_does_not_skip_collision_probe_above_ratio(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "10")
    control = rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=5000,
        n_theta=9,
        n_zeta=5,
        pas_precond_force_collision=True,
        residual_norm=11.0,
        target=1.0,
    )
    assert not control.reason_collision_probe_skip
    assert control.auto


def test_rhs1_strong_preconditioner_control_messages_report_all_gates() -> None:
    control = rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=5000,
        n_theta=9,
        n_zeta=5,
        cs0_sparse_first=True,
        large_cpu_sparse_rescue_first=True,
        pas_auto_skip=True,
        pas_fast_accept=True,
        pas_precond_force_collision=True,
        residual_norm=1.0,
        target=1.0,
    )

    messages = rhs1_strong_preconditioner_control_messages(
        control,
        residual_norm=1.0,
        target=1.0,
        rhs1_precond_kind="pas_lite",
        pas_auto_strong_ratio=4.0,
        sparse_rescue_label="gpu host-sparse",
    )

    joined = "\n".join(messages)
    assert "constraintScheme=0 sparse-first" in joined
    assert "gpu host-sparse rescue-first" in joined
    assert "PAS auto strong preconditioner skipped after base=pas_lite" in joined
    assert "PAS fast-accept" in joined
    assert "PAS collision probe disabled strong preconditioner auto" in joined


def test_rhs1_strong_preconditioner_control_messages_reports_collision_probe_allow() -> None:
    messages = rhs1_strong_preconditioner_control_messages(
        rhs1_resolved_strong_preconditioner_control(
            strong_precond_env="",
            has_extra_constraint_block=False,
            has_fp=False,
            has_pas=True,
            size=5000,
            n_theta=9,
            n_zeta=5,
            residual_norm=20.0,
            target=1.0,
        ),
        residual_norm=20.0,
        target=1.0,
        rhs1_precond_kind="pas_lite",
        pas_auto_strong_ratio=4.0,
        pas_collision_probe_allows_strong=True,
        pas_force_strong_ratio=10.0,
    )

    assert messages == (
        "solve_v3_full_system_linear_gmres: PAS collision probe allows strong "
        "preconditioner (residual=2.000e+01 > 10.0x target)",
    )
