from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_pas_policy import (
    build_pas_tz_memory_fallback,
    estimate_pas_tz_schwarz_fallback_work,
    pas_tz_schwarz_fallback_guard,
    pas_tz_schwarz_fallback_memory_safe,
    preferred_pas_tz_schwarz_axis,
    resolve_pas_tz_cheap_fallback_kind,
    resolve_pas_tz_guarded_correction_kind,
    resolve_pas_tz_memory_fallback_axis,
    rhs1_pas_adaptive_smoother_allowed,
)


def _op(
    *,
    has_pas: bool = True,
    has_phi1: bool = False,
    n_species: int = 1,
    n_theta: int = 25,
    n_zeta: int = 51,
    n_x: int = 2,
    n_xi: int = 6,
):
    return SimpleNamespace(
        include_phi1=has_phi1,
        n_species=n_species,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_x=n_x,
        n_xi=n_xi,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=tuple([int(n_xi)] * int(n_x))),
            pas=object() if has_pas else None,
        ),
    )


def test_pas_adaptive_smoother_allowed_for_large_explicit_pas(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", raising=False)
    assert rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )


def test_pas_adaptive_smoother_respects_problem_and_residual_guards(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", raising=False)
    kwargs = dict(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "op": _op(has_phi1=True)})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "op": _op(has_pas=False)})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "residual_norm": 1.0e-10})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "active_size": 1000})
    assert not rhs1_pas_adaptive_smoother_allowed(**{**kwargs, "use_implicit": True})


def test_pas_adaptive_smoother_env_controls_and_invalid_min(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "0")
    assert not rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER", "1")
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_MIN", "bad")
    assert rhs1_pas_adaptive_smoother_allowed(
        op=_op(),
        active_size=4000,
        residual_norm=1.0e-2,
        target=1.0e-8,
        use_implicit=False,
    )


def test_pas_tz_memory_fallback_axis_preserves_default_behavior() -> None:
    assert preferred_pas_tz_schwarz_axis(_op()) == "zeta"
    assert (
        resolve_pas_tz_memory_fallback_axis(
            op=_op(),
            requested="",
            shard_axis=None,
            n_devices=1,
        )
        is None
    )
    assert (
        resolve_pas_tz_memory_fallback_axis(
            op=_op(),
            requested="",
            shard_axis="theta",
            n_devices=2,
        )
        == "theta"
    )


def test_pas_tz_memory_fallback_axis_supports_opt_in_structured_schwarz() -> None:
    assert (
        resolve_pas_tz_memory_fallback_axis(
            op=_op(),
            requested="schwarz",
            shard_axis=None,
            n_devices=1,
        )
        == "zeta"
    )
    assert (
        resolve_pas_tz_memory_fallback_axis(
            op=SimpleNamespace(n_theta=64, n_zeta=8),
            requested="structured",
            shard_axis=None,
            n_devices=1,
        )
        == "theta"
    )
    assert (
        resolve_pas_tz_memory_fallback_axis(
            op=_op(),
            requested="hybrid",
            shard_axis="zeta",
            n_devices=8,
        )
        is None
    )


def test_pas_tz_cheap_fallback_kind_defaults_to_collision_with_hybrid_override() -> None:
    assert resolve_pas_tz_cheap_fallback_kind(requested="") == "collision"
    assert resolve_pas_tz_cheap_fallback_kind(requested="collision") == "collision"
    assert resolve_pas_tz_cheap_fallback_kind(requested="zeta") == "collision"
    assert resolve_pas_tz_cheap_fallback_kind(requested="hybrid") == "hybrid"
    assert resolve_pas_tz_cheap_fallback_kind(requested="pas-hybrid") == "hybrid"
    assert resolve_pas_tz_cheap_fallback_kind(requested="tzfft") == "tzfft"
    assert resolve_pas_tz_cheap_fallback_kind(requested="pas-stream-fft") == "tzfft"


def test_pas_tz_guarded_correction_kind_is_explicit() -> None:
    assert resolve_pas_tz_guarded_correction_kind(requested="") is None
    assert resolve_pas_tz_guarded_correction_kind(requested="off") is None
    assert resolve_pas_tz_guarded_correction_kind(requested="tzfft") == "tzfft"
    assert resolve_pas_tz_guarded_correction_kind(requested="collision-tzfft-correction") == "tzfft"
    assert resolve_pas_tz_guarded_correction_kind(requested="unknown") is None


def test_build_pas_tz_memory_fallback_can_force_zeta_schwarz(monkeypatch) -> None:
    calls: list[tuple[str, int, int]] = []

    def theta_builder(**kwargs):
        calls.append(("theta", int(kwargs["block"]), int(kwargs["overlap"])))
        return "theta-preconditioner"

    def zeta_builder(**kwargs):
        calls.append(("zeta", int(kwargs["block"]), int(kwargs["overlap"])))
        return "zeta-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append(("hybrid", 0, 0))
        return "hybrid-preconditioner"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "zeta")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK", "7")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP", "2")
    result = build_pas_tz_memory_fallback(
        op=_op(),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=theta_builder,
        zeta_schwarz_builder=zeta_builder,
        hybrid_builder=hybrid_builder,
    )
    assert result == "zeta-preconditioner"
    assert calls == [("zeta", 7, 2)]


def test_pas_tz_schwarz_fallback_work_estimate_flags_research_grid() -> None:
    work = estimate_pas_tz_schwarz_fallback_work(
        _op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        axis="zeta",
        block=3,
        overlap=1,
    )
    assert work["patch_count"] == 850
    assert work["max_patch_unknowns"] == 2000
    assert work["inverse_entries"] == 3_400_000_000
    assert not pas_tz_schwarz_fallback_memory_safe(
        _op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        axis="zeta",
        block=3,
        overlap=1,
    )
    guard = pas_tz_schwarz_fallback_guard(
        _op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        axis="zeta",
        block=3,
        overlap=1,
    )
    assert guard["safe"] is False
    assert "max-inverse-entries-exceeded" in guard["reason"]
    assert guard["work"]["inverse_entries"] == 3_400_000_000


def test_build_pas_tz_memory_fallback_uses_collision_when_schwarz_guard_fails(monkeypatch) -> None:
    calls: list[str] = []

    def unused_builder(**_kwargs):
        calls.append("schwarz")
        return "schwarz-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "zeta")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK", "3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP", "1")
    result = build_pas_tz_memory_fallback(
        op=_op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
    )
    assert result == "collision-preconditioner"
    assert calls == ["collision"]


def test_build_pas_tz_memory_fallback_prefers_tzfft_when_schwarz_guard_fails(monkeypatch) -> None:
    calls: list[str] = []

    def unused_builder(**_kwargs):
        calls.append("schwarz")
        return "schwarz-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    def tzfft_builder(**_kwargs):
        calls.append("tzfft")

        def _apply(value):
            return value

        return _apply

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "zeta")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK", "3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP", "1")
    result = build_pas_tz_memory_fallback(
        op=_op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
        tzfft_builder=tzfft_builder,
    )

    assert calls == ["tzfft"]
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_fallback") is True
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_axis") == "tzfft"
    metadata = getattr(result, "_sfincs_jax_pas_tz_guarded_metadata")
    assert metadata["requested_axis"] == "zeta"
    assert "using tzfft" in metadata["reason"]


def test_build_pas_tz_memory_fallback_keeps_implicit_sharded_default_bounded(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", raising=False)
    calls: list[str] = []

    def unused_builder(**_kwargs):
        calls.append("schwarz")
        return "schwarz-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    def tzfft_builder(**_kwargs):
        calls.append("tzfft")

        def _apply(value):
            return value

        return _apply

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_BLOCK", "3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_OVERLAP", "1")
    result = build_pas_tz_memory_fallback(
        op=_op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100),
        matvec_shard_axis=lambda _op: "zeta",
        device_count=lambda: 2,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
        tzfft_builder=tzfft_builder,
    )

    assert result == "collision-preconditioner"
    assert calls == ["collision"]


def test_build_pas_tz_memory_fallback_defaults_to_collision_on_single_device(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", raising=False)
    calls: list[str] = []

    def unused_builder(**_kwargs):
        calls.append("schwarz")
        return "schwarz-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    result = build_pas_tz_memory_fallback(
        op=_op(),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
    )
    assert result == "collision-preconditioner"
    assert calls == ["collision"]


def test_build_pas_tz_memory_fallback_can_force_legacy_hybrid(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "hybrid")
    calls: list[str] = []

    def unused_builder(**_kwargs):
        calls.append("schwarz")
        return "schwarz-preconditioner"

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    result = build_pas_tz_memory_fallback(
        op=_op(),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
    )
    assert result == "hybrid-preconditioner"
    assert calls == ["hybrid"]


def test_build_pas_tz_memory_fallback_can_force_tzfft_candidate(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", "tzfft")
    calls: list[str] = []

    def unused_builder(**_kwargs):
        raise AssertionError("tzfft request must not build structured Schwarz")

    def hybrid_builder(**_kwargs):
        calls.append("hybrid")
        return "hybrid-preconditioner"

    def collision_builder(**_kwargs):
        calls.append("collision")
        return "collision-preconditioner"

    def tzfft_builder(**_kwargs):
        calls.append("tzfft")

        def _apply(value):
            return value

        return _apply

    result = build_pas_tz_memory_fallback(
        op=_op(),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
        tzfft_builder=tzfft_builder,
    )
    assert calls == ["tzfft"]
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_fallback") is True
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_axis") == "tzfft"


def test_build_pas_tz_memory_fallback_marks_default_collision_guarded(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MEMORY_FALLBACK", raising=False)

    def unused_builder(**_kwargs):
        raise AssertionError("single-device default must not build structured Schwarz")

    def hybrid_builder(**_kwargs):
        raise AssertionError("single-device default must prefer cheap collision when available")

    def collision_builder(**_kwargs):
        def _apply(value):
            return value

        return _apply

    result = build_pas_tz_memory_fallback(
        op=_op(),
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=unused_builder,
        zeta_schwarz_builder=unused_builder,
        hybrid_builder=hybrid_builder,
        collision_builder=collision_builder,
    )

    assert getattr(result, "_sfincs_jax_pas_tz_guarded_fallback") is True
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_axis") == "collision"
    assert getattr(result, "_sfincs_jax_pas_tz_guarded_metadata")["reason"] == "cheap-collision"
