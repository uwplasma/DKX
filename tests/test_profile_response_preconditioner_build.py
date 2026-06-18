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
