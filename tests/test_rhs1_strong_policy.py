from __future__ import annotations

from sfincs_jax.rhs1_strong_policy import (
    requested_rhs1_strong_preconditioner_kind,
    rhs1_pas_weak_minres_steps,
    rhs1_pas_weak_strong_retry_skip,
)


def test_requested_rhs1_strong_preconditioner_kind_reduced_mode_extended_aliases() -> None:
    assert requested_rhs1_strong_preconditioner_kind("theta", mode="reduced") == "theta_line"
    assert requested_rhs1_strong_preconditioner_kind("point_xdiag", mode="reduced") == "point_xdiag"
    assert requested_rhs1_strong_preconditioner_kind("xblock_tz_lmax", mode="reduced") == "xblock_tz_lmax"
    assert requested_rhs1_strong_preconditioner_kind("pas_tz", mode="reduced") == "pas_tz"
    assert requested_rhs1_strong_preconditioner_kind("theta_zeta", mode="reduced") == "theta_zeta"
    assert requested_rhs1_strong_preconditioner_kind("adi_line", mode="reduced") == "adi"


def test_requested_rhs1_strong_preconditioner_kind_full_mode_preserves_existing_behavior() -> None:
    assert requested_rhs1_strong_preconditioner_kind("theta", mode="full") == "theta_line"
    assert requested_rhs1_strong_preconditioner_kind("theta_zeta", mode="full") == "adi"
    assert requested_rhs1_strong_preconditioner_kind("adi", mode="full") == "adi"
    assert requested_rhs1_strong_preconditioner_kind("point_xdiag", mode="full") is None
    assert requested_rhs1_strong_preconditioner_kind("xblock_tz_lmax", mode="full") is None
    assert requested_rhs1_strong_preconditioner_kind("pas_tz", mode="full") is None
    assert requested_rhs1_strong_preconditioner_kind("auto", mode="full") is None
    assert requested_rhs1_strong_preconditioner_kind("unknown", mode="full") is None


def test_rhs1_pas_weak_strong_retry_skip_only_for_huge_ratios(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", raising=False)
    for kind in ("collision", "point", "xmg"):
        assert not rhs1_pas_weak_strong_retry_skip(has_pas=True, rhs1_precond_kind=kind, res_ratio=1.0e7)
        assert rhs1_pas_weak_strong_retry_skip(has_pas=True, rhs1_precond_kind=kind, res_ratio=1.0e13)

    assert not rhs1_pas_weak_strong_retry_skip(has_pas=False, rhs1_precond_kind="collision", res_ratio=1.0e99)
    assert not rhs1_pas_weak_strong_retry_skip(has_pas=True, rhs1_precond_kind="pas_lite", res_ratio=1.0e99)

    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", "0")
    assert not rhs1_pas_weak_strong_retry_skip(has_pas=True, rhs1_precond_kind="collision", res_ratio=1.0e99)

    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", "bad")
    assert rhs1_pas_weak_strong_retry_skip(has_pas=True, rhs1_precond_kind="xmg", res_ratio=1.0e13)


def test_rhs1_pas_weak_minres_steps_only_for_large_weak_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", raising=False)

    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="collision", res_ratio=1.0e5) == 0
    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="collision", res_ratio=1.0e7) == 2
    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="pas_lite", res_ratio=1.0e99) == 0
    assert rhs1_pas_weak_minres_steps(has_pas=False, rhs1_precond_kind="xmg", res_ratio=1.0e99) == 0

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", "0")
    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="point", res_ratio=1.0e99) == 0

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", "4")
    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="xmg", res_ratio=1.0e7) == 4

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", "bad")
    assert rhs1_pas_weak_minres_steps(has_pas=True, rhs1_precond_kind="xmg", res_ratio=1.0e7) == 2
