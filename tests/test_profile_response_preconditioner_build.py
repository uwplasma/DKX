from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from sfincs_jax.problems.profile_response import preconditioner_build as pb


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
