from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

import sfincs_jax.problems.profile_preconditioner_build as pb


@dataclass(frozen=True)
class FakeFBlock:
    pas: object | None = None


@dataclass(frozen=True)
class FakeOperator:
    fblock: FakeFBlock


class FakeDDSetup:
    def block(self, axis: str) -> int:
        return 7 if axis == "theta" else 9

    def overlap(self, axis: str, *, default: int) -> int:
        return default + (10 if axis == "theta" else 20)


def _identity(v: jnp.ndarray) -> jnp.ndarray:
    return v


def _context(**overrides) -> pb.RHS1ReducedPreconditionerBuildContext:
    calls = overrides.pop("calls", {})

    def build_from_kind(**kwargs):
        calls["build_from_kind"] = kwargs
        precond = overrides.get("precond", _identity)
        if overrides.get("guarded", False):
            setattr(precond, "_sfincs_jax_pas_tz_guarded_fallback", True)
            setattr(precond, "_sfincs_jax_pas_tz_guarded_axis", "theta")
        if overrides.get("raise_build", False):
            raise RuntimeError("resource exhausted")
        return precond

    def build_collision(**kwargs):
        calls.setdefault("collision", []).append(kwargs)
        return _identity

    return pb.RHS1ReducedPreconditionerBuildContext(
        op=overrides.get("op", FakeOperator(fblock=FakeFBlock())),
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
        mv_reduced=lambda x: x,
        emit=overrides.get("emit"),
        mark=overrides.get("mark", lambda name: calls.setdefault("marks", []).append(name)),
        progress_preconditioner_build=overrides.get(
            "progress",
            lambda kind: calls.setdefault("progress", []).append(kind),
        ),
        record_structured_metadata=overrides.get(
            "record",
            lambda precond: calls.setdefault("recorded", []).append(precond),
        ),
        wrap_pas_preconditioner=overrides.get("wrap", lambda precond: precond),
        dd_setup=FakeDDSetup(),
        use_pas_projection=bool(overrides.get("use_pas_projection", False)),
        preconditioner_species=1,
        preconditioner_x=2,
        preconditioner_xi=3,
        build_from_kind=build_from_kind,
        build_collision=build_collision,
        build_xmg=build_collision,
        compose_residual_correction=overrides.get("compose_residual", lambda **kwargs: kwargs["base"]),
        compose_multilevel_residual_correction=overrides.get("compose_multilevel", lambda **kwargs: kwargs["base"]),
        compose_multilevel_minres_correction=overrides.get("compose_minres", lambda **kwargs: kwargs["base"]),
        parse_guarded_structured_levels=overrides.get("parse_levels", lambda _raw: ()),
        resource_exhausted_error=overrides.get("resource_exhausted", lambda _exc: False),
    )


def _full_context(**overrides) -> pb.RHS1FullPreconditionerBuildContext:
    calls = overrides.pop("calls", {})

    def build_from_kind(**kwargs):
        calls["build_from_kind"] = kwargs
        return overrides.get("precond", _identity)

    return pb.RHS1FullPreconditionerBuildContext(
        op=overrides.get("op", FakeOperator(fblock=FakeFBlock())),
        emit=overrides.get("emit"),
        mark=overrides.get("mark", lambda name: calls.setdefault("marks", []).append(name)),
        progress_preconditioner_build=overrides.get(
            "progress",
            lambda kind: calls.setdefault("progress", []).append(kind),
        ),
        record_structured_metadata=overrides.get(
            "record",
            lambda precond: calls.setdefault("recorded", []).append(precond),
        ),
        dd_setup=FakeDDSetup(),
        preconditioner_species=4,
        preconditioner_x=5,
        preconditioner_xi=6,
        build_from_kind=build_from_kind,
    )


def test_reduced_preconditioner_build_passes_policy_inputs() -> None:
    calls: dict[str, object] = {}
    result = pb.build_rhs1_reduced_preconditioner(
        context=_context(calls=calls, emit=lambda _level, _msg: None),
        rhs1_precond_kind="theta_schwarz",
        rhs1_xblock_tz_lmax=None,
    )

    kwargs = calls["build_from_kind"]
    assert kwargs["preconditioner_species"] == 1
    assert kwargs["preconditioner_x"] == 2
    assert kwargs["preconditioner_xi"] == 3
    assert kwargs["dd_block_theta"] == 7
    assert kwargs["dd_overlap_theta"] == 11
    assert kwargs["dd_block_zeta"] == 9
    assert kwargs["dd_overlap_zeta"] == 20
    assert calls["marks"] == ["rhs1_precond_build_start", "rhs1_precond_build_done"]
    assert calls["progress"] == ["theta_schwarz"]
    assert result.preconditioner is _identity
    assert not result.pas_tz_guarded_fallback


def test_full_preconditioner_build_passes_policy_inputs(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "3")

    result = pb.build_rhs1_full_preconditioner(
        context=_full_context(calls=calls, emit=lambda _level, _msg: None),
        rhs1_precond_kind="xblock_tz_lmax",
        rhs1_xblock_tz_lmax=None,
    )

    kwargs = calls["build_from_kind"]
    assert kwargs["preconditioner_species"] == 4
    assert kwargs["preconditioner_x"] == 5
    assert kwargs["preconditioner_xi"] == 6
    assert kwargs["rhs1_xblock_tz_lmax"] == 3
    assert kwargs["dd_block_theta"] == 7
    assert kwargs["dd_overlap_theta"] == 10
    assert kwargs["dd_block_zeta"] == 9
    assert kwargs["dd_overlap_zeta"] == 20
    assert kwargs["adi_sweeps"] == 4
    assert calls["marks"] == ["rhs1_precond_build_start", "rhs1_precond_build_done"]
    assert calls["progress"] == ["xblock_tz_lmax"]
    assert calls["recorded"] == [_identity]
    assert result is _identity


def test_strong_preconditioner_family_builds_full_through_dispatch(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, object] = {}

    def dispatch(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "6")
    family = pb.RHS1StrongPreconditionerFamilyBuilders(dispatch_builder=dispatch)

    kind, preconditioner = family.build_full_from_kind(
        op=FakeOperator(fblock=FakeFBlock()),
        strong_precond_kind="theta_schwarz",
        base_preconditioner_kind="point",
        residual_norm=1.0,
        dd_block_theta=17,
        dd_overlap_theta=3,
        dd_block_zeta=19,
        dd_overlap_zeta=4,
    )

    assert kind == "theta_schwarz"
    assert preconditioner is sentinel
    assert seen["rhs1_precond_kind"] == "theta_schwarz"
    assert seen["dd_block_theta"] == 17
    assert seen["dd_overlap_theta"] == 3
    assert seen["dd_block_zeta"] == 19
    assert seen["dd_overlap_zeta"] == 4
    assert seen["adi_sweeps"] == 6


def test_strong_preconditioner_family_downgrades_pas_schur_full_build() -> None:
    seen: dict[str, object] = {}

    def dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    family = pb.RHS1StrongPreconditionerFamilyBuilders(dispatch_builder=dispatch)
    kind, preconditioner = family.build_full_from_kind(
        op=FakeOperator(fblock=FakeFBlock(pas=object())),
        strong_precond_kind="schur",
        base_preconditioner_kind="pas_lite",
        residual_norm=1.0,
    )

    assert kind == "pas_hybrid"
    assert preconditioner is not None
    assert seen["rhs1_precond_kind"] == "pas_hybrid"


def test_strong_preconditioner_family_builds_reduced_with_lmax_and_adi_env(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, object] = {}

    def dispatch(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "6")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "4")
    family = pb.RHS1StrongPreconditionerFamilyBuilders(dispatch_builder=dispatch)

    preconditioner = family.build_reduced_from_kind(
        op=FakeOperator(fblock=FakeFBlock()),
        strong_precond_kind="xblock_tz_lmax",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )

    assert preconditioner is sentinel
    assert seen["rhs1_precond_kind"] == "xblock_tz_lmax"
    assert seen["rhs1_xblock_tz_lmax"] == 6
    assert seen["adi_sweeps"] == 4


def test_strong_preconditioner_family_reduced_unknown_kind_uses_adi(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "bad")
    family = pb.RHS1StrongPreconditionerFamilyBuilders(dispatch_builder=dispatch)

    family.build_reduced_from_kind(
        op=FakeOperator(fblock=FakeFBlock()),
        strong_precond_kind="not-a-kind",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )

    assert seen["rhs1_precond_kind"] == "adi"
    assert seen["adi_sweeps"] == 2


def test_preconditioner_threshold_readers_use_defaults_overrides_and_invalid_fallbacks(monkeypatch) -> None:
    keys = (
        "SFINCS_JAX_PAS_LITE_MIN",
        "SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX",
        "SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN",
        "SFINCS_JAX_RHSMODE1_PAS_XMG_MIN",
        "SFINCS_JAX_RHSMODE1_THETA_LINE_MAX",
        "SFINCS_JAX_PAS_STRONG_LMAX",
    )
    for key in keys:
        monkeypatch.delenv(key, raising=False)

    assert pb.rhs1_pas_lite_min() == 20000
    assert pb.rhs1_tz_precond_max() == 128
    assert pb.rhs1_xblock_tz_max(default=321) == 321
    assert pb.rhs1_schwarz_auto_min() == 4000
    assert pb.rhs1_pas_xmg_min() == 50000
    assert pb.rhs1_theta_line_max() == 0
    assert pb.rhs1_pas_strong_lmax() == 2

    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_MIN", "123")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "456")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "789")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN", "111")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_XMG_MIN", "222")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", "333")
    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_LMAX", "4")

    assert pb.rhs1_pas_lite_min() == 123
    assert pb.rhs1_tz_precond_max() == 456
    assert pb.rhs1_xblock_tz_max(default=321) == 789
    assert pb.rhs1_schwarz_auto_min() == 111
    assert pb.rhs1_pas_xmg_min() == 222
    assert pb.rhs1_theta_line_max() == 333
    assert pb.rhs1_pas_strong_lmax() == 4

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "not-an-int")
    assert pb.rhs1_xblock_tz_max(default=654) == 654


def test_direct_strong_preconditioner_build_functions_cover_none_and_pas_adjustment(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def dispatch(**kwargs):
        seen.update(kwargs)
        return "preconditioner"

    assert (
        pb.resolve_rhs1_strong_preconditioner_kind_for_build(
            "schur",
            has_pas=True,
            base_preconditioner_kind="pas_hybrid",
            residual_norm=1.0,
        )
        == "pas_hybrid"
    )
    assert (
        pb.resolve_rhs1_strong_preconditioner_kind_for_build(
            "schur",
            has_pas=True,
            base_preconditioner_kind="point",
            residual_norm=1.0,
        )
        == "schur"
    )

    assert (
        pb.build_rhs1_strong_preconditioner_reduced_from_kind(
            op=FakeOperator(fblock=FakeFBlock()),
            strong_precond_kind=None,
            reduce_full=lambda x: x,
            expand_reduced=lambda x: x,
            rhs1_xblock_tz_lmax=None,
            dd_block_theta=1,
            dd_overlap_theta=2,
            dd_block_zeta=3,
            dd_overlap_zeta=4,
            dispatch_builder=dispatch,
        )
        is None
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "5")
    reduced = pb.build_rhs1_strong_preconditioner_reduced_from_kind(
        op=FakeOperator(fblock=FakeFBlock()),
        strong_precond_kind="xblock_tz_lmax",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
        rhs1_xblock_tz_lmax=None,
        dd_block_theta=1,
        dd_overlap_theta=2,
        dd_block_zeta=3,
        dd_overlap_zeta=4,
        dispatch_builder=dispatch,
    )
    assert reduced == "preconditioner"
    assert seen["rhs1_precond_kind"] == "xblock_tz_lmax"
    assert seen["rhs1_xblock_tz_lmax"] == 5
    assert seen["dd_block_theta"] == 1
    assert seen["dd_overlap_zeta"] == 4

    seen.clear()
    kind, full = pb.build_rhs1_strong_preconditioner_full_from_kind(
        op=FakeOperator(fblock=FakeFBlock(pas=object())),
        strong_precond_kind="schur",
        base_preconditioner_kind="pas_lite",
        residual_norm=1.0,
        rhs1_xblock_tz_lmax=7,
        dd_block_theta=8,
        dd_overlap_theta=9,
        dd_block_zeta=10,
        dd_overlap_zeta=11,
        dispatch_builder=dispatch,
        adi_sweeps=0,
    )
    assert (kind, full) == ("pas_hybrid", "preconditioner")
    assert seen["rhs1_precond_kind"] == "pas_hybrid"
    assert seen["rhs1_xblock_tz_lmax"] == 7
    assert seen["adi_sweeps"] == 1

    assert pb.build_rhs1_strong_preconditioner_full_from_kind(
        op=FakeOperator(fblock=FakeFBlock()),
        strong_precond_kind=None,
        base_preconditioner_kind=None,
        residual_norm=1.0,
        rhs1_xblock_tz_lmax=None,
        dd_block_theta=0,
        dd_overlap_theta=0,
        dd_block_zeta=0,
        dd_overlap_zeta=0,
        dispatch_builder=dispatch,
    ) == (None, None)


def test_pas_tz_guarded_overlay_uses_structured_correction(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS", "collision")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_MODE", "fixed")

    def compose_multilevel(**kwargs):
        calls["compose_multilevel"] = kwargs
        return kwargs["base"]

    result = pb.build_rhs1_reduced_preconditioner(
        context=_context(
            calls=calls,
            guarded=True,
            parse_levels=lambda _raw: ("collision",),
            compose_multilevel=compose_multilevel,
        ),
        rhs1_precond_kind="pas_tz",
        rhs1_xblock_tz_lmax=None,
    )

    assert result.pas_tz_guarded_fallback
    assert result.pas_tz_guarded_axis == "theta"
    assert "compose_multilevel" in calls
    assert len(calls["collision"]) == 1


def test_reduced_preconditioner_fallback_switches_to_collision_on_accelerator_oom(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setattr(pb.jax, "default_backend", lambda: "gpu")

    result = pb.build_rhs1_reduced_preconditioner_with_fallback(
        context=_context(
            calls=calls,
            op=FakeOperator(fblock=FakeFBlock(pas=object())),
            raise_build=True,
            resource_exhausted=lambda _exc: True,
        ),
        rhs1_precond_kind="pas_tz",
        rhs1_xblock_tz_lmax=None,
        rhs1_bicgstab_kind="rhs1",
    )

    assert result.rhs1_precond_kind == "collision"
    assert result.pas_precond_force_collision
    assert result.bicgstab_preconditioner is result.preconditioner
    assert len(calls["collision"]) == 1


def _full_base_setup_context(**overrides) -> pb.RHS1FullBasePreconditionerSetupContext:
    calls = overrides.pop("calls", {})
    rhs = overrides.get("rhs", jnp.asarray([1.0, 2.0], dtype=jnp.float64))
    rhs1_precond = overrides.get("rhs1_precond", _identity)
    collision_precond = overrides.get("collision_precond", lambda v: 0.5 * v)

    def build_rhs1():
        calls.setdefault("build_rhs1", 0)
        calls["build_rhs1"] += 1
        return rhs1_precond

    def build_collision():
        calls.setdefault("build_collision", 0)
        calls["build_collision"] += 1
        return collision_precond

    return pb.RHS1FullBasePreconditionerSetupContext(
        rhs=rhs,
        rhs1_precond_enabled=bool(overrides.get("rhs1_precond_enabled", True)),
        host_dense_shortcut=bool(overrides.get("host_dense_shortcut", False)),
        rhs1_bicgstab_kind=overrides.get("rhs1_bicgstab_kind"),
        rhs1_precond_kind=overrides.get("rhs1_precond_kind", "point"),
        solve_method=overrides.get("solve_method", "gmres"),
        solve_method_kind=overrides.get("solve_method_kind", "gmres"),
        emit=overrides.get("emit"),
        solver_kind=overrides.get("solver_kind", lambda _method: ("gmres", "incremental")),
        build_rhs1_preconditioner=build_rhs1,
        build_collision_preconditioner=build_collision,
    )


def test_full_base_preconditioner_setup_builds_rhs1_for_gmres() -> None:
    calls: dict[str, int] = {}

    result = pb.setup_rhs1_full_base_preconditioner(
        _full_base_setup_context(calls=calls, rhs1_precond_kind="point")
    )

    assert result.preconditioner is _identity
    assert result.bicgstab_preconditioner is None
    assert calls == {"build_rhs1": 1}


def test_full_base_preconditioner_setup_skips_when_host_dense_shortcut() -> None:
    calls: dict[str, int] = {}

    result = pb.setup_rhs1_full_base_preconditioner(
        _full_base_setup_context(calls=calls, host_dense_shortcut=True)
    )

    assert result.preconditioner is None
    assert result.bicgstab_preconditioner is None
    assert calls == {}


def test_full_base_preconditioner_setup_uses_collision_for_bicgstab_kind() -> None:
    calls: dict[str, int] = {}

    result = pb.setup_rhs1_full_base_preconditioner(
        _full_base_setup_context(
            calls=calls,
            rhs1_precond_enabled=False,
            rhs1_bicgstab_kind="collision",
        )
    )

    assert result.preconditioner is result.bicgstab_preconditioner
    assert calls == {"build_collision": 1}


def test_full_base_preconditioner_setup_falls_back_for_nonfinite_pas_probe() -> None:
    calls: dict[str, int] = {}
    messages: list[str] = []

    def nonfinite_precond(v: jnp.ndarray) -> jnp.ndarray:
        return jnp.full_like(v, jnp.inf)

    result = pb.setup_rhs1_full_base_preconditioner(
        _full_base_setup_context(
            calls=calls,
            rhs1_precond=nonfinite_precond,
            rhs1_bicgstab_kind="rhs1",
            rhs1_precond_kind="pas_tz",
            emit=lambda _level, msg: messages.append(msg),
        )
    )

    assert result.preconditioner is result.bicgstab_preconditioner
    assert calls == {"build_rhs1": 1, "build_collision": 1}
    assert any("PAS precond non-finite" in message for message in messages)


@dataclass(frozen=True)
class FakeSolveResult:
    residual_norm: float


@dataclass(frozen=True)
class FakeProfileResult:
    x: jnp.ndarray
    residual_norm: float


def _clear_strong_policy_env(monkeypatch) -> None:
    for name in (
        "SFINCS_JAX_FP_STRONG_ABS",
        "SFINCS_JAX_PAS_FORCE_STRONG_RATIO",
        "SFINCS_JAX_PAS_LITE_MIN",
        "SFINCS_JAX_PAS_SCHUR_SMALL_MAX",
        "SFINCS_JAX_PAS_STRONG_LMAX",
        "SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO",
        "SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP",
        "SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT",
        "SFINCS_JAX_PAS_WEAK_MINRES_RATIO",
        "SFINCS_JAX_PAS_WEAK_MINRES_STEPS",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT",
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS",
        "SFINCS_JAX_RHSMODE1_SCHWARZ_AUTO_MIN",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO",
        "SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART",
        "SFINCS_JAX_RHSMODE1_THETA_LINE_MAX",
        "SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX",
        "SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX",
    ):
        monkeypatch.delenv(name, raising=False)


def test_strong_preconditioner_aliases_cover_modes_and_env(monkeypatch) -> None:
    _clear_strong_policy_env(monkeypatch)

    cases = {
        "theta_schwarz": "theta_schwarz",
        "theta_xdiag": "theta_line_xdiag",
        "species": "species_block",
        "species_x": "sxblock",
        "sx_tz": "sxblock_tz",
        "line_zeta": "zeta_line",
        "ras_zeta": "zeta_schwarz",
        "xtz": "xblock_tz",
        "multigrid": "xmg",
        "pas_light": "pas_lite",
        "pas_line_xcoarse": "pas_hybrid",
        "constraint_schur": "schur",
        "auto": None,
    }
    for token, expected in cases.items():
        assert pb.requested_rhs1_strong_preconditioner_kind(token, mode="full") == expected

    reduced_only = {
        "point_xdiag": "point_xdiag",
        "xblock_tz_cut": "xblock_tz_lmax",
        "pas_3d": "pas_tz",
        "tz": "theta_zeta",
        "zeta_theta": "adi",
    }
    for token, expected in reduced_only.items():
        assert pb.requested_rhs1_strong_preconditioner_kind(token, mode="reduced") == expected

    assert pb.requested_rhs1_strong_preconditioner_kind("zeta_theta", mode="full") == "adi"
    assert pb.requested_rhs1_strong_preconditioner_kind("unknown", mode="reduced") is None

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND", "  XMG ")
    assert pb.rhs1_strong_preconditioner_env_from_env() == "xmg"


def test_strong_trigger_retry_and_weak_controls_cover_env_edges(monkeypatch) -> None:
    _clear_strong_policy_env(monkeypatch)

    delayed = pb.rhs1_strong_trigger_controls_from_env(
        residual_norm=20.0,
        target=1.0,
        has_fp=False,
        include_phi1=False,
        has_pas=True,
        rhs1_precond_kind="pas_tokamak_theta",
        delay_pas_base_retries=True,
    )
    assert delayed.ratio_threshold == 1.0e4
    assert not delayed.trigger

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_FP_STRONG_ABS", "bad")
    fp_forced = pb.rhs1_strong_trigger_controls_from_env(
        residual_norm=2.0e-6,
        target=1.0e-9,
        has_fp=True,
        include_phi1=False,
        has_pas=False,
        rhs1_precond_kind="point",
        delay_pas_base_retries=False,
    )
    assert fp_forced.ratio_threshold == 1.0
    assert fp_forced.fp_abs_threshold == 1.0e-6
    assert fp_forced.fp_force

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", "bad")
    retry = pb.rhs1_strong_retry_controls_from_env(restart=7, maxiter=9)
    assert retry.restart == 120
    assert retry.maxiter == 800

    monkeypatch.setenv("SFINCS_JAX_PAS_FORCE_STRONG_RATIO", "bad")
    assert pb.rhs1_pas_force_strong_ratio_from_env() == 50.0

    assert pb.rhs1_collision_retry_allowed(
        residual_norm=2.0,
        target=1.0,
        rhs_mode=1,
        include_phi1=False,
        rhs1_precond_kind="point",
        has_fp=True,
        has_pas=False,
        strong_precond_trigger=True,
    )
    assert not pb.rhs1_collision_retry_allowed(
        residual_norm=2.0,
        target=1.0,
        rhs_mode=2,
        include_phi1=False,
        rhs1_precond_kind="point",
        has_fp=True,
        has_pas=False,
        strong_precond_trigger=True,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", "bad")
    assert pb.rhs1_pas_weak_strong_retry_skip(
        has_pas=True,
        rhs1_precond_kind="collision",
        res_ratio=1.0e13,
    )
    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_WEAK_SKIP_RATIO", "0")
    assert not pb.rhs1_pas_weak_strong_retry_skip(
        has_pas=True,
        rhs1_precond_kind="collision",
        res_ratio=1.0e13,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_STEPS", "bad")
    assert (
        pb.rhs1_pas_weak_minres_steps(
            has_pas=True,
            rhs1_precond_kind="collision",
            res_ratio=1.0e7,
        )
        == 2
    )
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_RATIO", "-1")
    assert (
        pb.rhs1_pas_weak_minres_steps(
            has_pas=True,
            rhs1_precond_kind="collision",
            res_ratio=1.0e7,
        )
        == 0
    )
    assert (
        pb.rhs1_pas_weak_minres_steps(
            has_pas=False,
            rhs1_precond_kind="collision",
            res_ratio=1.0e7,
        )
        == 0
    )


def test_minres_control_readers_and_messages_cover_policy_edges(monkeypatch) -> None:
    _clear_strong_policy_env(monkeypatch)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_STEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_ALPHA_CLIP", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_MINRES_MIN_IMPROVEMENT", "bad")
    guarded = pb.rhs1_pas_tz_guarded_minres_controls_from_env()
    assert guarded == pb.RHS1MinresCorrectionControls(
        steps=2,
        alpha_clip=10.0,
        min_improvement=0.0,
    )

    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_ALPHA_CLIP", "bad")
    monkeypatch.setenv("SFINCS_JAX_PAS_WEAK_MINRES_MIN_IMPROVEMENT", "bad")
    weak = pb.rhs1_pas_weak_minres_controls_from_env(steps=3)
    assert weak == pb.RHS1MinresCorrectionControls(
        steps=3,
        alpha_clip=10.0,
        min_improvement=0.0,
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MIN", "bad")
    assert pb.rhs1_strong_preconditioner_min_size() == 800

    control = pb.RHS1StrongPreconditionerControl(
        min_size=800,
        disabled=True,
        auto=False,
        reason_large_cpu_sparse_first=True,
        reason_collision_probe_skip=True,
    )
    messages = pb.rhs1_strong_preconditioner_control_messages(
        control,
        residual_norm=2.0,
        target=1.0,
        rhs1_precond_kind="collision",
        pas_auto_strong_ratio=50.0,
        sparse_rescue_label="production CPU",
    )
    assert any("production CPU rescue-first" in message for message in messages)
    assert any("collision probe disabled" in message for message in messages)

    allowed = pb.rhs1_strong_preconditioner_control_messages(
        pb.RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=True),
        residual_norm=101.0,
        target=1.0,
        rhs1_precond_kind="collision",
        pas_auto_strong_ratio=50.0,
        pas_collision_probe_allows_strong=True,
        pas_force_strong_ratio=100.0,
    )
    assert any("collision probe allows" in message for message in allowed)


def test_resolved_strong_control_and_auto_selection_helpers(monkeypatch) -> None:
    _clear_strong_policy_env(monkeypatch)

    cs0 = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="auto",
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        size=900,
        n_theta=5,
        n_zeta=1,
        cs0_sparse_first=True,
    )
    assert cs0.disabled
    assert cs0.reason_cs0_sparse_first

    fastpath = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=900,
        n_theta=5,
        n_zeta=1,
        pas_large_bicgstab_fastpath=True,
    )
    assert fastpath.disabled
    assert not fastpath.auto

    sparse_first = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        size=900,
        n_theta=5,
        n_zeta=1,
        large_cpu_sparse_rescue_first=True,
    )
    assert sparse_first.disabled
    assert sparse_first.reason_large_cpu_sparse_first

    pas_skip = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="theta",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=900,
        n_theta=5,
        n_zeta=1,
        pas_auto_skip=True,
    )
    assert pas_skip.disabled
    assert pas_skip.reason_pas_auto_skip

    pas_fast = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="auto",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=900,
        n_theta=5,
        n_zeta=1,
        pas_fast_accept=True,
    )
    assert pas_fast.disabled
    assert pas_fast.reason_pas_fast_accept

    collision_probe = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        size=900,
        n_theta=5,
        n_zeta=1,
        pas_precond_force_collision=True,
        residual_norm=20.0,
        target=1.0,
    )
    assert collision_probe.disabled
    assert collision_probe.reason_collision_probe_skip

    constraint_auto = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=True,
        has_fp=False,
        has_pas=False,
        size=10,
        n_theta=1,
        n_zeta=1,
    )
    assert constraint_auto.auto

    fp_auto = pb.rhs1_resolved_strong_preconditioner_control(
        strong_precond_env="",
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        size=900,
        n_theta=5,
        n_zeta=1,
    )
    assert fp_auto.auto

    assert pb.auto_rhs1_reduced_strong_kind(
        has_pas=True,
        has_fp=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=30000,
        strong_precond_min=800,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        shard_axis=None,
        device_count=1,
    ).kind == "pas_lite"

    assert pb.auto_rhs1_reduced_strong_kind(
        has_pas=True,
        has_fp=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=10,
        strong_precond_min=800,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        shard_axis=None,
        device_count=1,
    ).kind == "pas_hybrid"

    fp_xblock = pb.auto_rhs1_reduced_strong_kind(
        has_pas=False,
        has_fp=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=3000,
        strong_precond_min=800,
        n_theta=5,
        n_zeta=5,
        max_l=2,
        shard_axis=None,
        device_count=1,
    )
    assert fp_xblock.kind == "xblock_tz"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "60")
    fp_lmax = pb.auto_rhs1_reduced_strong_kind(
        has_pas=False,
        has_fp=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=3000,
        strong_precond_min=800,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        shard_axis=None,
        device_count=1,
    )
    assert fp_lmax.kind == "xblock_tz_lmax"
    assert fp_lmax.xblock_tz_lmax == 2

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "100")
    assert (
        pb.auto_rhs1_reduced_strong_kind(
            has_pas=False,
            has_fp=True,
            geom_scheme=5,
            use_dkes=False,
            active_size=3000,
            strong_precond_min=800,
            n_theta=5,
            n_zeta=5,
            max_l=8,
            shard_axis=None,
            device_count=1,
        ).kind
        == "theta_zeta"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "1")
    assert (
        pb.auto_rhs1_reduced_strong_kind(
            has_pas=False,
            has_fp=True,
            geom_scheme=5,
            use_dkes=False,
            active_size=5000,
            strong_precond_min=800,
            n_theta=5,
            n_zeta=9,
            max_l=8,
            shard_axis="zeta",
            device_count=2,
        ).kind
        == "zeta_schwarz"
    )

    assert (
        pb.auto_rhs1_reduced_strong_kind(
            has_pas=False,
            has_fp=True,
            geom_scheme=5,
            use_dkes=False,
            active_size=5000,
            strong_precond_min=800,
            n_theta=3,
            n_zeta=9,
            max_l=8,
            shard_axis=None,
            device_count=1,
        ).kind
        == "zeta_line"
    )

    assert (
        pb.auto_rhs1_reduced_strong_kind(
            has_pas=False,
            has_fp=False,
            geom_scheme=5,
            use_dkes=False,
            active_size=1,
            strong_precond_min=800,
            n_theta=1,
            n_zeta=1,
            max_l=1,
            shard_axis=None,
            device_count=1,
        ).kind
        is None
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_TZ_PRECOND_MAX", "1")
    assert (
        pb.auto_rhs1_full_strong_kind(
            has_pas=True,
            has_fp=False,
            rhs1_precond_kind="point",
            total_size=10,
            strong_precond_min=800,
            n_theta=5,
            n_zeta=5,
            max_l=8,
            shard_axis=None,
            device_count=1,
        ).kind
        == "pas_hybrid"
    )
    assert (
        pb.auto_rhs1_full_strong_kind(
            has_pas=False,
            has_fp=True,
            rhs1_precond_kind="point",
            total_size=5000,
            strong_precond_min=800,
            n_theta=5,
            n_zeta=5,
            max_l=8,
            shard_axis="theta",
            device_count=2,
        ).kind
        == "theta_schwarz"
    )
    assert (
        pb.auto_rhs1_full_strong_kind(
            has_pas=False,
            has_fp=True,
            rhs1_precond_kind="point",
            total_size=5000,
            strong_precond_min=800,
            n_theta=3,
            n_zeta=7,
            max_l=8,
            shard_axis=None,
            device_count=1,
        ).kind
        == "zeta_line"
    )


def test_adjust_and_resolve_strong_selection_helpers(monkeypatch) -> None:
    _clear_strong_policy_env(monkeypatch)

    tokamak = pb.adjust_rhs1_reduced_auto_kind(
        kind="pas_lite",
        has_pas=True,
        geom_scheme=1,
        n_zeta=1,
        strong_precond_trigger=False,
        max_l=8,
        n_theta=5,
    )
    assert tokamak.kind == "pas_hybrid"

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "1")
    monkeypatch.setenv("SFINCS_JAX_PAS_STRONG_LMAX", "3")
    truncated = pb.adjust_rhs1_reduced_auto_kind(
        kind="pas_hybrid",
        has_pas=True,
        geom_scheme=1,
        n_zeta=1,
        strong_precond_trigger=True,
        max_l=8,
        n_theta=5,
    )
    assert truncated.kind == "xblock_tz_lmax"
    assert truncated.xblock_tz_lmax == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_MAX", "2000")
    full_xblock = pb.adjust_rhs1_reduced_auto_kind(
        kind="pas_hybrid",
        has_pas=True,
        geom_scheme=1,
        n_zeta=1,
        strong_precond_trigger=True,
        max_l=8,
        n_theta=5,
    )
    assert full_xblock.kind == "xblock_tz"

    monkeypatch.setenv("SFINCS_JAX_PAS_SCHUR_SMALL_MAX", "bad")
    assert (
        pb.adjust_rhs1_pas_schur_strong_kind_from_env(
            kind="schur",
            has_pas=True,
            base_kind="pas_lite",
            residual_norm=1.0,
            active_size=3000,
        )
        == "pas_hybrid"
    )
    assert (
        pb.adjust_rhs1_pas_schur_strong_kind_from_env(
            kind="schur",
            has_pas=True,
            base_kind="pas_lite",
            residual_norm=1.0,
            active_size=10,
        )
        == "schur"
    )

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_THETA_LINE_MAX", "10")
    assert (
        pb.adjust_rhs1_theta_line_auto_kind(
            kind="theta_line",
            n_theta=5,
            nxi_for_x_sum=3,
        ).kind
        == "theta_line_xdiag"
    )

    control = pb.RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=True)
    reduced = pb.resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="",
        control=control,
        has_extra_constraint_block=True,
        has_fp=False,
        has_pas=False,
        geom_scheme=5,
        use_dkes=False,
        active_size=3000,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="point",
        res_ratio=2.0,
        pas_tz_guarded_fallback=False,
        pas_tz_guarded_strong_retry=False,
        qi_device_skip_strong=False,
    )
    assert reduced.kind == "schur"
    assert reduced.trigger

    skipped = pb.resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="theta",
        control=control,
        has_extra_constraint_block=False,
        has_fp=False,
        has_pas=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=3000,
        n_theta=5,
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
    assert skipped.kind is None
    assert skipped.skipped_weak_pas
    assert skipped.skipped_guarded_pas_tz
    assert skipped.skipped_qi_device
    messages = pb.rhs1_reduced_strong_selection_skip_messages(skipped)
    assert len(messages) == 3
    assert pb.rhs1_reduced_strong_selection_skip_messages(
        pb.RHS1ReducedStrongPreconditionerSelection(
            kind=None,
            candidate_kind_before_skips=None,
            xblock_tz_lmax=None,
            trigger=False,
        )
    ) == ()

    reduced_auto_pas = pb.resolve_rhs1_reduced_strong_preconditioner_selection(
        strong_precond_env="",
        control=control,
        has_extra_constraint_block=True,
        has_fp=False,
        has_pas=True,
        geom_scheme=5,
        use_dkes=False,
        active_size=30000,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
        strong_precond_trigger=True,
        rhs1_precond_kind="point",
        res_ratio=2.0,
        pas_tz_guarded_fallback=False,
        pas_tz_guarded_strong_retry=False,
        qi_device_skip_strong=False,
    )
    assert reduced_auto_pas.kind == "pas_lite"

    full = pb.resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="",
        control=control,
        has_extra_constraint_block=True,
        has_fp=False,
        has_pas=False,
        rhs1_precond_kind="point",
        geom_scheme=5,
        total_size=3000,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )
    assert full.kind == "schur"

    full_auto_pas = pb.resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="",
        control=control,
        has_extra_constraint_block=True,
        has_fp=False,
        has_pas=True,
        rhs1_precond_kind="point",
        geom_scheme=5,
        total_size=30000,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )
    assert full_auto_pas.kind == "pas_lite"

    full_auto_fp = pb.resolve_rhs1_full_strong_preconditioner_selection(
        strong_precond_env="",
        control=control,
        has_extra_constraint_block=False,
        has_fp=True,
        has_pas=False,
        rhs1_precond_kind="point",
        geom_scheme=5,
        total_size=3000,
        n_theta=5,
        n_zeta=5,
        max_l=8,
        nxi_for_x_sum=20,
        shard_axis=None,
        device_count=1,
    )
    assert full_auto_fp.kind in {"xblock_tz", "xblock_tz_lmax", "theta_zeta"}


def _post_primary_context(**overrides) -> pb.RHS1PostPrimaryMinresCorrectionContext:
    calls = overrides.pop("calls", {})
    metadata = overrides.get("metadata", {})
    result = overrides.get(
        "result",
        FakeProfileResult(
            x=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
            residual_norm=10.0,
        ),
    )

    def minres_correction(**kwargs):
        calls.setdefault("minres", []).append(kwargs)
        return (
            kwargs["x0"] * 0.5,
            jnp.asarray([0.25, -0.25], dtype=jnp.float64),
            tuple(overrides.get("history", (5.0,))),
            tuple(overrides.get("alphas", (0.75,))),
        )

    def result_factory(x, residual_norm):
        calls.setdefault("factory", []).append((x, residual_norm))
        return FakeProfileResult(x=x, residual_norm=float(residual_norm))

    return pb.RHS1PostPrimaryMinresCorrectionContext(
        result=result,
        residual_vec=overrides.get("residual_vec", jnp.asarray([9.0, -9.0], dtype=jnp.float64)),
        residual_norm_true=float(overrides.get("residual_norm_true", result.residual_norm)),
        target=float(overrides.get("target", 1.0)),
        matvec=overrides.get("matvec", _identity),
        rhs=overrides.get("rhs", jnp.asarray([1.0, -1.0], dtype=jnp.float64)),
        preconditioner=overrides.get("preconditioner", _identity),
        has_pas=bool(overrides.get("has_pas", True)),
        rhs1_precond_kind=overrides.get("rhs1_precond_kind", "collision"),
        pas_tz_guarded_fallback=bool(overrides.get("pas_tz_guarded_fallback", False)),
        pas_tz_guarded_axis=overrides.get("pas_tz_guarded_axis"),
        pas_tz_guarded_stream_requested=bool(
            overrides.get("pas_tz_guarded_stream_requested", False)
        ),
        use_pas_projection=bool(overrides.get("use_pas_projection", False)),
        metadata=metadata,
        requested_guarded_correction=overrides.get("requested_guarded_correction", "base"),
        build_tzfft_preconditioner=overrides.get(
            "build_tzfft",
            lambda: calls.setdefault("tzfft", _identity),
        ),
        wrap_pas_preconditioner=overrides.get(
            "wrap",
            lambda precond: calls.setdefault("wrapped", precond),
        ),
        minres_correction=overrides.get("minres_correction", minres_correction),
        result_factory=overrides.get("result_factory", result_factory),
        resolve_guarded_correction_kind=overrides.get(
            "resolve_guarded",
            lambda *, requested: requested,
        ),
        guarded_controls_factory=overrides.get(
            "guarded_controls",
            lambda: pb.RHS1MinresCorrectionControls(
                steps=2,
                alpha_clip=10.0,
                min_improvement=0.0,
            ),
        ),
        weak_steps_policy=overrides.get("weak_steps_policy", lambda **_kwargs: 0),
        weak_controls_factory=overrides.get(
            "weak_controls",
            lambda *, steps: pb.RHS1MinresCorrectionControls(
                steps=int(steps),
                alpha_clip=10.0,
                min_improvement=0.0,
            ),
        ),
    )


def test_post_primary_minres_accepts_guarded_tzfft_stream_fallback() -> None:
    calls: dict[str, object] = {}
    metadata: dict[str, object] = {}
    messages: list[str] = []

    outcome = pb.run_rhs1_post_primary_minres_corrections(
        _post_primary_context(
            calls=calls,
            metadata=metadata,
            pas_tz_guarded_fallback=True,
            pas_tz_guarded_axis="theta",
            pas_tz_guarded_stream_requested=True,
            requested_guarded_correction="tzfft",
            use_pas_projection=True,
        ),
        emit=lambda _level, msg: messages.append(msg),
    )

    assert outcome.accepted_guarded
    assert not outcome.accepted_weak
    assert outcome.residual_norm_true == 5.0
    assert metadata["pas_tz_guarded_correction_kind"] == "tzfft"
    assert metadata["pas_tz_guarded_correction_stream_blocker"].startswith(
        "production-pas-tz"
    )
    assert metadata["pas_tz_guarded_correction_full_update_materialized"] is True
    assert calls["wrapped"] is calls["tzfft"]
    assert any("matrix-free correction=tzfft" in message for message in messages)
    assert any("streamed correction requested" in message for message in messages)


def test_post_primary_minres_uses_base_when_tzfft_correction_fails() -> None:
    messages: list[str] = []

    outcome = pb.run_rhs1_post_primary_minres_corrections(
        _post_primary_context(
            pas_tz_guarded_fallback=True,
            pas_tz_guarded_axis="theta",
            requested_guarded_correction="tzfft",
            build_tzfft=lambda: (_ for _ in ()).throw(RuntimeError("no tzfft")),
        ),
        emit=lambda _level, msg: messages.append(msg),
    )

    assert outcome.accepted_guarded
    assert any("tzfft unavailable" in message for message in messages)


def test_post_primary_minres_accepts_weak_pas_correction() -> None:
    calls: dict[str, object] = {}

    outcome = pb.run_rhs1_post_primary_minres_corrections(
        _post_primary_context(
            calls=calls,
            weak_steps_policy=lambda **_kwargs: 2,
        )
    )

    assert outcome.accepted_weak
    assert not outcome.accepted_guarded
    assert outcome.residual_norm_true == 5.0
    assert len(calls["minres"]) == 1


def test_pas_tz_guarded_overlay_poly_and_minres_paths(monkeypatch) -> None:
    calls: dict[str, object] = {}
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_STEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_DAMPING", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS", "xmg,unknown")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_STEPS", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_DAMPING", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_ALPHA_CLIP", "bad")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_MIN_IMPROVEMENT",
        "bad",
    )

    def compose_minres(**kwargs):
        calls["compose_minres"] = kwargs
        return kwargs["base"]

    result = pb.build_rhs1_reduced_preconditioner(
        context=_context(
            calls=calls,
            guarded=True,
            emit=lambda _level, msg: messages.append(msg),
            parse_levels=lambda _raw: ("xmg", "unknown"),
            compose_minres=compose_minres,
        ),
        rhs1_precond_kind="pas_tz",
        rhs1_xblock_tz_lmax=None,
    )

    assert result.pas_tz_guarded_fallback
    assert len(calls["collision"]) == 1
    assert "compose_minres" in calls
    assert any("mode=minres" in message for message in messages)


def test_pas_tz_guarded_overlay_poly_only_returns_when_no_structured_levels(
    monkeypatch,
) -> None:
    calls: dict[str, object] = {}
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_STEPS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_POLY_DAMPING", "bad")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS", raising=False)

    def compose_residual(**kwargs):
        calls["compose_residual"] = kwargs
        return kwargs["base"]

    result = pb.build_rhs1_reduced_preconditioner(
        context=_context(
            calls=calls,
            guarded=True,
            emit=lambda _level, msg: messages.append(msg),
            parse_levels=lambda _raw: (),
            compose_residual=compose_residual,
        ),
        rhs1_precond_kind="pas_tz",
        rhs1_xblock_tz_lmax=None,
    )

    assert result.pas_tz_guarded_fallback
    assert calls["compose_residual"]["steps"] == 2
    assert calls["compose_residual"]["damping"] == 0.5
    assert any("polynomial correction" in message for message in messages)


def test_pas_tz_guarded_overlay_structured_steps_zero_returns_base(monkeypatch) -> None:
    calls: dict[str, object] = {}
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_LEVELS", "collision")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_GUARDED_STRUCTURED_STEPS", "0")

    result = pb.build_rhs1_reduced_preconditioner(
        context=_context(
            calls=calls,
            guarded=True,
            parse_levels=lambda _raw: ("collision",),
        ),
        rhs1_precond_kind="pas_tz",
        rhs1_xblock_tz_lmax=None,
    )

    assert result.pas_tz_guarded_fallback
    assert len(calls["collision"]) == 1
    assert "compose_multilevel" not in calls

def _full_strong_stage_context(**overrides) -> pb.RHS1FullStrongRetryStageContext:
    calls = overrides.pop("calls", {})
    rhs = overrides.get("rhs", jnp.asarray([1.0, -2.0], dtype=jnp.float64))
    current_result = overrides.get("current_result", FakeSolveResult(residual_norm=10.0))
    precond = overrides.get("precond", _identity)

    def build_strong(kind: str):
        calls["build_strong"] = kind
        return kind, precond

    def run_measured_candidate(**kwargs):
        calls["run_measured"] = kwargs
        return (
            overrides.get("candidate_result", FakeSolveResult(residual_norm=0.1)),
            overrides.get("candidate_residual_vec", jnp.zeros_like(rhs)),
            True,
            0.25,
        )

    return pb.RHS1FullStrongRetryStageContext(
        strong_precond_env=overrides.get("strong_precond_env", "theta"),
        strong_control=overrides.get(
            "strong_control",
            pb.RHS1StrongPreconditionerControl(min_size=800, disabled=False, auto=False),
        ),
        has_extra_constraint_block=bool(overrides.get("has_extra_constraint_block", False)),
        has_fp=bool(overrides.get("has_fp", True)),
        has_pas=bool(overrides.get("has_pas", False)),
        rhs1_precond_kind=overrides.get("rhs1_precond_kind", "point"),
        geom_scheme=int(overrides.get("geom_scheme", 5)),
        total_size=int(overrides.get("total_size", 5000)),
        n_theta=int(overrides.get("n_theta", 9)),
        n_zeta=int(overrides.get("n_zeta", 5)),
        max_l=int(overrides.get("max_l", 8)),
        nxi_for_x_sum=int(overrides.get("nxi_for_x_sum", 20)),
        shard_axis=overrides.get("shard_axis"),
        device_count=int(overrides.get("device_count", 1)),
        pas_auto_strong_ratio=float(overrides.get("pas_auto_strong_ratio", 50.0)),
        current_result=current_result,
        current_residual_vec=overrides.get("current_residual_vec"),
        matvec=overrides.get("matvec", _identity),
        rhs=rhs,
        tol=float(overrides.get("tol", 1.0e-9)),
        atol=float(overrides.get("atol", 1.0e-12)),
        restart=int(overrides.get("restart", 10)),
        maxiter=overrides.get("maxiter", 20),
        precondition_side=overrides.get("precondition_side", "left"),
        solver_kind=overrides.get("solver_kind", "gmres"),
        target=float(overrides.get("target", 1.0)),
        peak_rss_mb=float(overrides.get("peak_rss_mb", 123.0)),
        emit=overrides.get("emit"),
        mark=overrides.get("mark", lambda name: calls.setdefault("marks", []).append(name)),
        replay_state=overrides.get("replay_state", object()),
        build_strong_preconditioner=overrides.get("build_strong", build_strong),
        run_measured_candidate=overrides.get("run_measured", run_measured_candidate),
        solve_linear=overrides.get("solve_linear", lambda **_kwargs: None),
    )


def _reduced_strong_stage_context(**overrides) -> pb.RHS1ReducedStrongRetryStageContext:
    calls = overrides.pop("calls", {})
    rhs = overrides.get("rhs", jnp.asarray([1.0, -2.0], dtype=jnp.float64))
    current_result = overrides.get("current_result", FakeSolveResult(residual_norm=10.0))
    precond = overrides.get("precond", _identity)

    def build_strong(kind: str, lmax: int | None):
        calls["build_strong"] = (kind, lmax)
        return precond

    def run_measured_candidate(**kwargs):
        calls["run_measured"] = kwargs
        return (
            overrides.get("candidate_result", FakeSolveResult(residual_norm=0.1)),
            overrides.get("candidate_residual_vec", None),
            True,
            0.5,
        )

    def wrap(preconditioner):
        calls["wrapped"] = preconditioner
        return overrides.get("wrapped_precond", preconditioner)

    return pb.RHS1ReducedStrongRetryStageContext(
        strong_precond_kind=overrides.get("strong_precond_kind", "theta_line"),
        strong_xblock_tz_lmax=overrides.get("strong_xblock_tz_lmax", 3),
        rescue_needed=bool(overrides.get("rescue_needed", True)),
        strong_precond_trigger=bool(overrides.get("strong_precond_trigger", True)),
        early_dense_shortcut=bool(overrides.get("early_dense_shortcut", False)),
        active_size=int(overrides.get("active_size", 5000)),
        has_fp=bool(overrides.get("has_fp", True)),
        has_pas=bool(overrides.get("has_pas", False)),
        rhs1_precond_kind=overrides.get("rhs1_precond_kind", "point"),
        current_result=current_result,
        current_residual_vec=overrides.get("current_residual_vec"),
        matvec=overrides.get("matvec", _identity),
        rhs=rhs,
        tol=float(overrides.get("tol", 1.0e-9)),
        atol=float(overrides.get("atol", 1.0e-12)),
        restart=int(overrides.get("restart", 10)),
        maxiter=overrides.get("maxiter", 20),
        precondition_side=overrides.get("precondition_side", "left"),
        solver_kind=overrides.get("solver_kind", "gmres"),
        target=float(overrides.get("target", 1.0)),
        peak_rss_mb=float(overrides.get("peak_rss_mb", 321.0)),
        emit=overrides.get("emit"),
        mark=overrides.get("mark", lambda name: calls.setdefault("marks", []).append(name)),
        replay_state=overrides.get("replay_state", object()),
        build_strong_preconditioner=overrides.get("build_strong", build_strong),
        wrap_pas_preconditioner=overrides.get("wrap", wrap),
        use_pas_projection=bool(overrides.get("use_pas_projection", False)),
        run_measured_candidate=overrides.get("run_measured", run_measured_candidate),
        solve_linear=overrides.get("solve_linear", lambda **_kwargs: None),
        result_ready=overrides.get("result_ready", lambda _result: True),
    )


def test_reduced_strong_retry_stage_builds_and_runs_explicit_request(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", raising=False)
    calls: dict[str, object] = {}
    messages: list[str] = []

    outcome = pb.run_rhs1_reduced_strong_retry_stage(
        _reduced_strong_stage_context(
            calls=calls,
            emit=lambda _level, msg: messages.append(msg),
            use_pas_projection=True,
        )
    )

    assert calls["build_strong"] == ("theta_line", 3)
    assert calls["marks"] == ["rhs1_strong_precond_build_start", "rhs1_strong_precond_build_done"]
    assert calls["wrapped"] is _identity
    measured = calls["run_measured"]
    assert measured["precond_fn"] is _identity
    assert measured["restart"] == 120
    assert measured["maxiter"] == 800
    assert measured["candidate_name"] == "strong_reduced"
    assert measured["returns_residual_vec"] is False
    assert outcome.accepted
    assert outcome.selected_kind == "theta_line"
    assert any("strong preconditioner fallback kind=theta_line" in message for message in messages)


def test_reduced_strong_retry_stage_respects_rescue_gate() -> None:
    calls: dict[str, object] = {}
    current = FakeSolveResult(residual_norm=0.5)

    outcome = pb.run_rhs1_reduced_strong_retry_stage(
        _reduced_strong_stage_context(
            calls=calls,
            current_result=current,
            rescue_needed=False,
        )
    )

    assert outcome.result is current
    assert outcome.residual_vec is None
    assert not outcome.accepted
    assert outcome.selected_kind == "theta_line"
    assert outcome.preconditioner is None
    assert calls == {}


def test_reduced_strong_retry_stage_applies_fp_size_guard(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_STRONG_PRECOND_MAX", "4")
    calls: dict[str, object] = {}
    messages: list[str] = []

    outcome = pb.run_rhs1_reduced_strong_retry_stage(
        _reduced_strong_stage_context(
            calls=calls,
            active_size=5,
            has_fp=True,
            has_pas=False,
            strong_precond_kind="xblock_tz",
            emit=lambda _level, msg: messages.append(msg),
        )
    )

    assert not outcome.accepted
    assert outcome.selected_kind is None
    assert outcome.preconditioner is None
    assert calls == {}
    assert any("fp_max=4" in message for message in messages)


def test_full_strong_retry_stage_builds_and_runs_explicit_request(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_RESTART", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_STRONG_PRECOND_MAXITER", raising=False)
    calls: dict[str, object] = {}
    messages: list[str] = []

    outcome = pb.run_rhs1_full_strong_retry_stage(
        _full_strong_stage_context(calls=calls, emit=lambda _level, msg: messages.append(msg))
    )

    assert calls["build_strong"] == "theta_line"
    assert calls["marks"] == ["rhs1_strong_precond_build_start", "rhs1_strong_precond_build_done"]
    measured = calls["run_measured"]
    assert measured["precond_fn"] is _identity
    assert measured["restart"] == 120
    assert measured["maxiter"] == 800
    assert measured["candidate_name"] == "strong_full"
    assert measured["returns_residual_vec"] is True
    assert outcome.accepted
    assert outcome.selected_kind == "theta_line"
    assert outcome.preconditioner is _identity
    assert any("strong preconditioner fallback kind=theta_line" in message for message in messages)


def test_full_strong_retry_stage_respects_residual_gate() -> None:
    calls: dict[str, object] = {}
    current = FakeSolveResult(residual_norm=0.5)

    outcome = pb.run_rhs1_full_strong_retry_stage(
        _full_strong_stage_context(calls=calls, current_result=current, target=1.0)
    )

    assert outcome.result is current
    assert outcome.residual_vec is None
    assert not outcome.accepted
    assert outcome.selected_kind == "theta_line"
    assert outcome.preconditioner is None
    assert calls == {}


def test_full_strong_retry_stage_emits_policy_skip_messages() -> None:
    messages: list[str] = []
    control = pb.RHS1StrongPreconditionerControl(
        min_size=800,
        disabled=True,
        auto=False,
        reason_cs0_sparse_first=True,
        reason_pas_auto_skip=True,
        reason_pas_fast_accept=True,
    )

    outcome = pb.run_rhs1_full_strong_retry_stage(
        _full_strong_stage_context(
            strong_control=control,
            has_fp=False,
            has_pas=True,
            rhs1_precond_kind="pas_lite",
            pas_auto_strong_ratio=25.0,
            emit=lambda _level, msg: messages.append(msg),
        )
    )

    assert outcome.selected_kind is None
    assert any("constraintScheme=0 sparse-first" in message for message in messages)
    assert any("25.0x target" in message for message in messages)
    assert any("PAS fast-accept" in message for message in messages)
