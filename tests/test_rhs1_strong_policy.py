from __future__ import annotations

from sfincs_jax.rhs1_strong_policy import requested_rhs1_strong_preconditioner_kind


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
