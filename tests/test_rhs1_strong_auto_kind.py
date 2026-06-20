from __future__ import annotations

from sfincs_jax.rhs1_strong_auto_kind import (
    RHS1StrongPreconditionerControl,
    adjust_rhs1_reduced_auto_kind,
    adjust_rhs1_theta_line_auto_kind,
    auto_rhs1_full_strong_kind,
    auto_rhs1_reduced_strong_kind,
    resolve_rhs1_full_strong_preconditioner_selection,
    resolve_rhs1_reduced_strong_preconditioner_selection,
    rhs1_reduced_strong_selection_skip_messages,
)


def test_auto_rhs1_reduced_strong_kind_fp_prefers_xblock_tz_lmax_when_full_xblock_is_too_large(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "64")
    selection = auto_rhs1_reduced_strong_kind(
        has_pas=False,
        has_fp=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=4000,
        strong_precond_min=800,
        n_theta=4,
        n_zeta=4,
        max_l=8,
        shard_axis=None,
        device_count=1,
    )
    assert selection.kind == "xblock_tz_lmax"
    assert selection.xblock_tz_lmax == 4


def test_auto_rhs1_reduced_strong_kind_pas_defaults_to_pas_hybrid_below_lite_threshold() -> None:
    selection = auto_rhs1_reduced_strong_kind(
        has_pas=True,
        has_fp=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=5000,
        strong_precond_min=800,
        n_theta=9,
        n_zeta=5,
        max_l=6,
        shard_axis=None,
        device_count=1,
    )
    assert selection.kind == "pas_hybrid"


def test_adjust_rhs1_reduced_auto_kind_promotes_tokamak_pas_to_xblock_tz_lmax_when_triggered(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "64")
    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_LMAX", "3")
    selection = adjust_rhs1_reduced_auto_kind(
        kind="pas_hybrid",
        has_pas=True,
        geom_scheme=1,
        n_zeta=1,
        strong_precond_trigger=True,
        max_l=8,
        n_theta=12,
    )
    assert selection.kind == "xblock_tz_lmax"
    assert selection.xblock_tz_lmax == 3


def test_adjust_rhs1_theta_line_auto_kind_promotes_to_xdiag_when_line_too_large(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", "100")
    selection = adjust_rhs1_theta_line_auto_kind(
        kind="theta_line",
        n_theta=12,
        nxi_for_x_sum=10,
    )
    assert selection.kind == "theta_line_xdiag"


def test_auto_rhs1_full_strong_kind_pas_point_still_tracks_existing_pas_hybrid_preference(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", "2000")
    selection = auto_rhs1_full_strong_kind(
        has_pas=True,
        has_fp=False,
        rhs1_precond_kind="point",
        total_size=5000,
        strong_precond_min=800,
        n_theta=9,
        n_zeta=5,
        max_l=6,
        shard_axis=None,
        device_count=1,
    )
    assert selection.kind == "pas_hybrid"


def test_auto_rhs1_full_strong_kind_fp_prefers_sharded_schwarz_before_line(
    monkeypatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "16")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "1000")
    selection = auto_rhs1_full_strong_kind(
        has_pas=False,
        has_fp=True,
        rhs1_precond_kind="block",
        total_size=4000,
        strong_precond_min=800,
        n_theta=16,
        n_zeta=16,
        max_l=8,
        shard_axis="theta",
        device_count=2,
    )
    assert selection.kind == "theta_schwarz"


def test_resolve_rhs1_reduced_strong_selection_preserves_explicit_request() -> None:
    selection = resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="theta",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=False),
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="point",
        res_ratio=10.0,
        pas_tz_guarded_fallback=False,
        pas_tz_guarded_strong_retry=False,
        qi_device_skip_strong=False,
    )

    assert selection.kind == "theta_line"
    assert selection.candidate_kind_before_skips == "theta_line"
    assert selection.trigger


def test_resolve_rhs1_reduced_strong_selection_extra_constraint_uses_schur() -> None:
    selection = resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=True),
        has_extra_constraint_block=True,
        has_fp=True,
        has_pas=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="point",
        res_ratio=10.0,
        pas_tz_guarded_fallback=False,
        pas_tz_guarded_strong_retry=False,
        qi_device_skip_strong=False,
    )

    assert selection.kind == "schur"
    assert selection.candidate_kind_before_skips == "schur"


def test_resolve_rhs1_reduced_strong_selection_tracks_pas_and_device_skip_gates(
    monkeypatch,
) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", raising=False)
    selection = resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="theta",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=False),
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="collision",
        res_ratio=1.0e13,
        pas_tz_guarded_fallback=True,
        pas_tz_guarded_strong_retry=False,
        qi_device_skip_strong=True,
    )

    assert selection.kind is None
    assert selection.candidate_kind_before_skips == "theta_line"
    assert not selection.trigger
    assert selection.skipped_weak_pas
    assert selection.skipped_guarded_pas_tz
    assert selection.skipped_qi_device
    messages = rhs1_reduced_strong_selection_skip_messages(selection)
    assert messages == (
        "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
        "after weak PAS base residual exceeded skip threshold; set "
        "SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO=0 to retry",
        "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
        "after guarded PAS-TZ fallback; set "
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRONG_RETRY=1 to retry",
        "solve_v3_full_system_linear_gmres: skipping strong preconditioner "
        "for QI device preconditioner experiment",
    )


def test_resolve_rhs1_reduced_strong_selection_allows_guarded_pas_retry() -> None:
    selection = resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="theta",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=False),
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="pas_lite",
        res_ratio=10.0,
        pas_tz_guarded_fallback=True,
        pas_tz_guarded_strong_retry=True,
        qi_device_skip_strong=False,
    )

    assert selection.kind == "theta_line"
    assert selection.trigger
    assert not selection.skipped_guarded_pas_tz


def test_resolve_rhs1_full_strong_selection_preserves_full_mode_alias() -> None:
    selection = resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="theta_zeta",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=False),
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        rhs1_precond_kind="point",
        geom_scheme=5,
        total_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )

    assert selection.kind == "adi"


def test_resolve_rhs1_full_strong_selection_extra_constraint_uses_schur() -> None:
    selection = resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=True),
        has_extra_constraint_block=True,
        has_fp=True,
        has_pas=False,
        rhs1_precond_kind="point",
        geom_scheme=5,
        total_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )

    assert selection.kind == "schur"


def test_resolve_rhs1_full_strong_selection_auto_uses_full_size() -> None:
    selection = resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="",
        control=RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=True),
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        rhs1_precond_kind="block",
        geom_scheme=5,
        total_size=5000,
        n_theta=9,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )

    assert selection.kind == "xblock_tz"
