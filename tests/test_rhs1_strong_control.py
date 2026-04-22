from __future__ import annotations

from sfincs_jax.rhs1_strong_control import (
    rhs1_resolved_strong_preconditioner_control,
    rhs1_strong_preconditioner_min_size,
)


def test_rhs1_strong_preconditioner_min_size_handles_invalid_env(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "bad")
    assert rhs1_strong_preconditioner_min_size() == 800
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "1200")
    assert rhs1_strong_preconditioner_min_size() == 1200


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
