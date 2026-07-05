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
