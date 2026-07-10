from __future__ import annotations

from dataclasses import dataclass

from sfincs_jax.problems.profile_preconditioner_build import (
    RHS1MinresCorrectionControls,
    RHS1PostPrimaryMinresCorrectionContext,
    adjust_rhs1_pas_schur_strong_kind_from_env,
    requested_rhs1_strong_preconditioner_kind,
    rhs1_pas_tz_guarded_minres_controls_from_env,
    rhs1_pas_weak_minres_controls_from_env,
    rhs1_pas_weak_minres_steps,
    rhs1_pas_weak_strong_retry_skip,
    run_rhs1_post_primary_minres_corrections,
)


@dataclass(frozen=True)
class _DummyResult:
    x: object
    residual_norm: float


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


def test_rhs1_pas_tz_guarded_minres_controls_from_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT", raising=False)
    assert rhs1_pas_tz_guarded_minres_controls_from_env() == RHS1MinresCorrectionControls(
        steps=2,
        alpha_clip=10.0,
        min_improvement=0.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS", "5")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP", "2.5")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT", "0.1")
    assert rhs1_pas_tz_guarded_minres_controls_from_env() == RHS1MinresCorrectionControls(
        steps=5,
        alpha_clip=2.5,
        min_improvement=0.1,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT", "bad")
    assert rhs1_pas_tz_guarded_minres_controls_from_env() == RHS1MinresCorrectionControls(
        steps=2,
        alpha_clip=10.0,
        min_improvement=0.0,
    )


def test_rhs1_pas_weak_minres_controls_from_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT", raising=False)
    assert rhs1_pas_weak_minres_controls_from_env(steps=3) == RHS1MinresCorrectionControls(
        steps=3,
        alpha_clip=10.0,
        min_improvement=0.0,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP", "4.0")
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT", "0.25")
    assert rhs1_pas_weak_minres_controls_from_env(steps=7) == RHS1MinresCorrectionControls(
        steps=7,
        alpha_clip=4.0,
        min_improvement=0.25,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT", "bad")
    assert rhs1_pas_weak_minres_controls_from_env(steps=2) == RHS1MinresCorrectionControls(
        steps=2,
        alpha_clip=10.0,
        min_improvement=0.0,
    )


def test_post_primary_minres_corrections_accepts_guarded_tzfft_path() -> None:
    messages: list[str] = []
    built: list[str] = []
    wrapped: list[object] = []

    def base_preconditioner(vector):
        return ("base", vector)

    def tzfft_preconditioner(vector):
        return ("tzfft", vector)

    def build_tzfft_preconditioner():
        built.append("tzfft")
        return tzfft_preconditioner

    def wrap_pas_preconditioner(preconditioner):
        wrapped.append(preconditioner)

        def wrapped_preconditioner(vector):
            return ("wrapped", preconditioner(vector))

        return wrapped_preconditioner

    def minres_correction(**kwargs):
        assert kwargs["preconditioner"](("probe",))[0] == "wrapped"
        return "x1", "residual_vec1", (10.0, 2.0), (0.25,)

    metadata: dict[str, object] = {}
    outcome = run_rhs1_post_primary_minres_corrections(
        RHS1PostPrimaryMinresCorrectionContext(
            result=_DummyResult(x="x0", residual_norm=10.0),
            residual_vec="residual_vec0",
            residual_norm_true=10.0,
            target=1.0,
            matvec=lambda vector: vector,
            rhs="rhs",
            preconditioner=base_preconditioner,
            has_pas=True,
            rhs1_precond_kind="pas_lite",
            pas_tz_guarded_fallback=True,
            pas_tz_guarded_axis="theta",
            pas_tz_guarded_stream_requested=True,
            use_pas_projection=True,
            metadata=metadata,
            requested_guarded_correction="tzfft",
            build_tzfft_preconditioner=build_tzfft_preconditioner,
            wrap_pas_preconditioner=wrap_pas_preconditioner,
            minres_correction=minres_correction,
            result_factory=lambda x, residual_norm: _DummyResult(
                x=x,
                residual_norm=residual_norm,
            ),
            resolve_guarded_correction_kind=lambda *, requested: requested,
            guarded_controls_factory=lambda: RHS1MinresCorrectionControls(
                steps=2,
                alpha_clip=10.0,
                min_improvement=0.0,
            ),
            weak_steps_policy=lambda **_kwargs: 2,
            weak_controls_factory=lambda *, steps: RHS1MinresCorrectionControls(
                steps=steps,
                alpha_clip=10.0,
                min_improvement=0.0,
            ),
        ),
        emit=lambda _level, message: messages.append(message),
    )

    assert outcome.accepted_guarded
    assert not outcome.accepted_weak
    assert outcome.result == _DummyResult(x="x1", residual_norm=2.0)
    assert outcome.residual_vec == "residual_vec1"
    assert outcome.residual_norm_true == 2.0
    assert built == ["tzfft"]
    assert wrapped == [tzfft_preconditioner]
    assert metadata["pas_tz_guarded_correction_kind"] == "tzfft"
    assert metadata["pas_tz_guarded_correction_full_update_materialized"] is True
    assert metadata["pas_tz_guarded_correction_minres_steps"] == 2
    assert metadata["pas_tz_guarded_correction_stream_blocker"] == (
        "production-pas-tz-minres-correction-requires-full-residual-direction"
    )
    assert any("PAS-TZ guarded streamed correction requested" in message for message in messages)
    assert any("PAS-TZ guarded minres correction accepted" in message for message in messages)


def test_post_primary_minres_corrections_accepts_weak_pas_path() -> None:
    messages: list[str] = []

    def minres_correction(**_kwargs):
        return "x1", "residual_vec1", (20.0, 5.0), (0.5,)

    outcome = run_rhs1_post_primary_minres_corrections(
        RHS1PostPrimaryMinresCorrectionContext(
            result=_DummyResult(x="x0", residual_norm=20.0),
            residual_vec="residual_vec0",
            residual_norm_true=20.0,
            target=1.0,
            matvec=lambda vector: vector,
            rhs="rhs",
            preconditioner=lambda vector: vector,
            has_pas=True,
            rhs1_precond_kind="collision",
            pas_tz_guarded_fallback=False,
            pas_tz_guarded_axis=None,
            pas_tz_guarded_stream_requested=False,
            use_pas_projection=False,
            metadata={},
            requested_guarded_correction="",
            build_tzfft_preconditioner=lambda: (lambda vector: vector),
            wrap_pas_preconditioner=lambda preconditioner: preconditioner,
            minres_correction=minres_correction,
            result_factory=lambda x, residual_norm: _DummyResult(
                x=x,
                residual_norm=residual_norm,
            ),
            resolve_guarded_correction_kind=lambda *, requested: None,
            guarded_controls_factory=lambda: RHS1MinresCorrectionControls(
                steps=0,
                alpha_clip=10.0,
                min_improvement=0.0,
            ),
            weak_steps_policy=lambda **_kwargs: 3,
            weak_controls_factory=lambda *, steps: RHS1MinresCorrectionControls(
                steps=steps,
                alpha_clip=4.0,
                min_improvement=0.0,
            ),
        ),
        emit=lambda _level, message: messages.append(message),
    )

    assert not outcome.accepted_guarded
    assert outcome.accepted_weak
    assert outcome.result == _DummyResult(x="x1", residual_norm=5.0)
    assert outcome.residual_vec == "residual_vec1"
    assert outcome.residual_norm_true == 5.0
    assert any("weak PAS minres correction accepted" in message for message in messages)


def test_adjust_rhs1_pas_schur_strong_kind_from_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_SCHUR_SMALL_MAX", raising=False)
    kwargs = dict(
        kind="schur",
        has_pas=True,
        base_kind="pas_lite",
        residual_norm=1.0,
        active_size=2001,
    )
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**kwargs) == "pas_hybrid"
    assert (
        adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "active_size": 2000})
        == "schur"
    )
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "kind": "theta_line"}) == "theta_line"
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "has_pas": False}) == "schur"
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "base_kind": "point"}) == "schur"
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "residual_norm": float("inf")}) == "schur"

    monkeypatch.setenv("SFINCS_JAX_PAS_SCHUR_SMALL_MAX", "10")
    assert adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "active_size": 11}) == "pas_hybrid"

    monkeypatch.setenv("SFINCS_JAX_PAS_SCHUR_SMALL_MAX", "bad")
    assert (
        adjust_rhs1_pas_schur_strong_kind_from_env(**{**kwargs, "active_size": 2001})
        == "pas_hybrid"
    )
