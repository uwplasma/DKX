from __future__ import annotations

from types import SimpleNamespace

from sfincs_jax.rhs1_pas_policy import (
    RHS1PASAdaptiveSmootherControls,
    RHS1PASPreconditionerProbeConfig,
    RHS1PASSchurRescueControls,
    build_pas_tz_memory_fallback,
    estimate_rhs1_pas_tz_build_bytes,
    estimate_rhs1_pas_tz_build_memory,
    estimate_pas_tz_schwarz_fallback_work,
    pas_tz_preconditioner_memory_safe,
    pas_tz_schwarz_fallback_guard,
    pas_tz_schwarz_fallback_memory_safe,
    preferred_pas_tz_schwarz_axis,
    resolve_pas_tz_cheap_fallback_kind,
    resolve_pas_tz_guarded_correction_kind,
    resolve_pas_tz_memory_fallback_axis,
    rhs1_pas_adaptive_smoother_allowed,
    rhs1_pas_adaptive_smoother_controls_from_env,
    rhs1_pas_default_preconditioner_kind,
    rhs1_pas_preconditioner_probe_admitted,
    rhs1_pas_preconditioner_probe_config_from_env,
    rhs1_pas_preconditioner_probe_large_collision_skip,
    rhs1_pas_preconditioner_probe_uses_collision,
    rhs1_pas_schur_rescue_controls_from_env,
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


def _pas_tz_op(
    *,
    rhs_mode: int = 1,
    has_pas: bool = True,
    has_fp: bool = False,
    n_species: int = 2,
    n_theta: int = 25,
    n_zeta: int = 51,
    n_x: int = 4,
    n_xi: int = 100,
):
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        include_phi1=False,
        n_species=n_species,
        n_theta=n_theta,
        n_zeta=n_zeta,
        n_x=n_x,
        n_xi=n_xi,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=tuple([int(n_xi)] * int(n_x))),
            pas=object() if has_pas else None,
            fp=object() if has_fp else None,
            fp_phi1=None,
            exb_theta=None,
            exb_zeta=None,
            magdrift_theta=None,
            magdrift_zeta=None,
            magdrift_xidot=None,
            er_xdot=None,
            er_xidot=None,
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


def test_pas_tz_build_memory_preflight_rejects_productionish_grid(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES", raising=False)

    op = _pas_tz_op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100)
    metadata = estimate_rhs1_pas_tz_build_memory(op)

    assert metadata["applicable"] is True
    assert metadata["safe"] is False
    assert metadata["reason"] == "pas-tz-build-memory-limit-exceeded"
    assert metadata["active_unknowns"] == 2 * 25 * 51 * 4 * 100
    assert metadata["lmax"] == 6
    assert metadata["lmax_source"] == "default"
    assert metadata["total_nbytes"] == estimate_rhs1_pas_tz_build_bytes(op)
    assert metadata["max_nbytes"] == 2 * 1024 * 1024 * 1024
    assert not pas_tz_preconditioner_memory_safe(op)


def test_pas_tz_build_memory_preflight_tracks_env_limit(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES", str(4 * 1024 * 1024 * 1024))

    metadata = estimate_rhs1_pas_tz_build_memory(
        _pas_tz_op(n_species=2, n_theta=25, n_zeta=51, n_x=4, n_xi=100)
    )

    assert metadata["safe"] is True
    assert metadata["reason"] == "within-pas-tz-build-memory-limit"
    assert metadata["max_nbytes"] == 4 * 1024 * 1024 * 1024


def test_pas_tz_build_memory_preflight_explains_inapplicable_ops(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_MAX_BYTES", raising=False)
    metadata = estimate_rhs1_pas_tz_build_memory(_pas_tz_op(rhs_mode=2))

    assert metadata == {
        "applicable": False,
        "safe": True,
        "reason": "pas-tz-inapplicable",
        "total_nbytes": 0,
        "max_nbytes": 2 * 1024 * 1024 * 1024,
    }


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


def test_pas_adaptive_smoother_controls_from_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_SWEEPS", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_OMEGA", raising=False)
    assert rhs1_pas_adaptive_smoother_controls_from_env() == RHS1PASAdaptiveSmootherControls(
        max_sweeps=3,
        omega=1.0,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_SWEEPS", "5")
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_OMEGA", "0.75")
    assert rhs1_pas_adaptive_smoother_controls_from_env() == RHS1PASAdaptiveSmootherControls(
        max_sweeps=5,
        omega=0.75,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_SWEEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_ADAPTIVE_SMOOTHER_OMEGA", "bad")
    assert rhs1_pas_adaptive_smoother_controls_from_env() == RHS1PASAdaptiveSmootherControls(
        max_sweeps=3,
        omega=1.0,
    )


def test_pas_schur_rescue_controls_trigger_and_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RATIO", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAXITER", raising=False)

    assert rhs1_pas_schur_rescue_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        n_species=2,
        residual_norm=2.0e-2,
        target=1.0e-8,
        active_size=4000,
        restart=80,
        maxiter=300,
    ) == RHS1PASSchurRescueControls(
        run=True,
        ratio=1.0e4,
        max_active_size=90000,
        restart=120,
        maxiter=1200,
    )


def test_pas_schur_rescue_controls_respect_guards(monkeypatch) -> None:
    kwargs = dict(
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        n_species=2,
        residual_norm=2.0e-2,
        target=1.0e-8,
        active_size=4000,
        restart=80,
        maxiter=300,
    )
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "rhs_mode": 2}).run
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "include_phi1": True}).run
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "has_pas": False}).run
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "n_species": 1}).run
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "residual_norm": float("inf")}).run
    assert not rhs1_pas_schur_rescue_controls_from_env(**{**kwargs, "active_size": 100000}).run

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RATIO", "0")
    assert not rhs1_pas_schur_rescue_controls_from_env(**kwargs).run


def test_pas_schur_rescue_controls_env_overrides_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RATIO", "25")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAX", "5000")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RESTART", "44")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAXITER", "88")
    assert rhs1_pas_schur_rescue_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        n_species=2,
        residual_norm=3.0e-7,
        target=1.0e-8,
        active_size=4000,
        restart=80,
        maxiter=300,
    ) == RHS1PASSchurRescueControls(
        run=True,
        ratio=25.0,
        max_active_size=5000,
        restart=44,
        maxiter=88,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_SCHUR_RESCUE_MAXITER", "bad")
    assert rhs1_pas_schur_rescue_controls_from_env(
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        n_species=2,
        residual_norm=2.0e-2,
        target=1.0e-8,
        active_size=4000,
        restart=80,
        maxiter=None,
    ) == RHS1PASSchurRescueControls(
        run=True,
        ratio=1.0e4,
        max_active_size=90000,
        restart=120,
        maxiter=1200,
    )


def test_pas_preconditioner_probe_config_preserves_defaults(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_PRECOND_PROBE", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_PRECOND_PROBE_REL_MAX", raising=False)
    monkeypatch.delenv("SFINCS_JAX_PAS_PRECOND_BUILD_MAX", raising=False)

    assert rhs1_pas_preconditioner_probe_config_from_env() == RHS1PASPreconditionerProbeConfig(
        enabled=True,
        rel_max=0.9,
        build_max=20000,
    )


def test_pas_preconditioner_probe_config_respects_env_and_invalid_values(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_PROBE", "off")
    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_PROBE_REL_MAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_BUILD_MAX", "bad")

    assert rhs1_pas_preconditioner_probe_config_from_env() == RHS1PASPreconditionerProbeConfig(
        enabled=False,
        rel_max=0.9,
        build_max=20000,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_PROBE_REL_MAX", "0.25")
    monkeypatch.setenv("SFINCS_JAX_PAS_PRECOND_BUILD_MAX", "1234")
    assert rhs1_pas_preconditioner_probe_config_from_env() == RHS1PASPreconditionerProbeConfig(
        enabled=True,
        rel_max=0.25,
        build_max=1234,
    )


def test_pas_default_preconditioner_kind_prefers_schur_for_tokamak_like_multispecies() -> None:
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="auto",
            current_kind="theta_line",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            n_species=2,
            n_zeta=1,
            geom_scheme=1,
        )
        == "schur"
    )
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="default",
            current_kind="theta_line",
            rhs_mode=1,
            include_phi1=False,
            has_pas=True,
            n_species=2,
            n_zeta=9,
            geom_scheme=5,
        )
        == "schur"
    )


def test_pas_default_preconditioner_kind_preserves_user_and_non_pas_choices() -> None:
    kwargs = dict(
        current_kind="theta_line",
        rhs_mode=1,
        include_phi1=False,
        has_pas=True,
        n_species=2,
        n_zeta=1,
        geom_scheme=1,
    )
    assert rhs1_pas_default_preconditioner_kind(requested_env="xblock_tz", **kwargs) == "theta_line"
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="auto",
            **{**kwargs, "include_phi1": True},
        )
        == "theta_line"
    )
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="auto",
            **{**kwargs, "has_pas": False},
        )
        == "theta_line"
    )
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="auto",
            **{**kwargs, "n_species": 1},
        )
        == "theta_line"
    )
    assert (
        rhs1_pas_default_preconditioner_kind(
            requested_env="auto",
            **{**kwargs, "geom_scheme": 5, "n_zeta": 17},
        )
        == "theta_line"
    )


def test_pas_preconditioner_probe_admission_guards_heavy_paths() -> None:
    config = RHS1PASPreconditionerProbeConfig(enabled=True, rel_max=0.9, build_max=20000)
    kwargs = dict(
        config=config,
        preconditioner_kind="schur",
        preconditioner_enabled=True,
        solve_method_kind="gmres",
        has_pas=True,
        use_dkes=False,
    )

    assert rhs1_pas_preconditioner_probe_admitted(**kwargs)
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "config": RHS1PASPreconditionerProbeConfig(enabled=False, rel_max=0.9, build_max=20000)})
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "preconditioner_kind": "collision"})
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "preconditioner_enabled": False})
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "solve_method_kind": "dense"})
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "has_pas": False})
    assert not rhs1_pas_preconditioner_probe_admitted(**{**kwargs, "use_dkes": True})


def test_pas_preconditioner_probe_large_collision_skip_preserves_constraint_tail_guard() -> None:
    config = RHS1PASPreconditionerProbeConfig(enabled=True, rel_max=0.9, build_max=100)

    decision, message = rhs1_pas_preconditioner_probe_large_collision_skip(
        config=config,
        cached_decision=None,
        total_size=101,
        constraint_scheme=0,
        extra_size=0,
    )
    assert decision is True
    assert message == "solve_v3_full_system_linear_gmres: PAS precond skip (size=101 >= 100) -> collision"

    assert rhs1_pas_preconditioner_probe_large_collision_skip(
        config=config,
        cached_decision=False,
        total_size=101,
        constraint_scheme=0,
        extra_size=0,
    ) == (False, None)
    assert rhs1_pas_preconditioner_probe_large_collision_skip(
        config=config,
        cached_decision=None,
        total_size=99,
        constraint_scheme=0,
        extra_size=0,
    ) == (None, None)
    assert rhs1_pas_preconditioner_probe_large_collision_skip(
        config=config,
        cached_decision=None,
        total_size=101,
        constraint_scheme=2,
        extra_size=1,
    ) == (None, None)


def test_pas_preconditioner_probe_residual_decision_uses_threshold() -> None:
    assert rhs1_pas_preconditioner_probe_uses_collision(probe_rel=0.9, rel_max=0.9)
    assert not rhs1_pas_preconditioner_probe_uses_collision(probe_rel=0.9001, rel_max=0.9)


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


def test_build_pas_tz_memory_fallback_prefers_tzfft_when_available(monkeypatch) -> None:
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

    assert callable(result)
    assert calls == ["tzfft"]


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
