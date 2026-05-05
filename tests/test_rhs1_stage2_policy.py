from __future__ import annotations

from sfincs_jax.rhs1_stage2_policy import (
    rhs1_fp_force_stage2,
    rhs1_pas_stage2_skip,
    rhs1_stage2_ratio,
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
