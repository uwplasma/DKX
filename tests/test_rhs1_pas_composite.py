from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

from sfincs_jax.solvers.preconditioner_pas_composite import (
    RHS1PasCompositeBuilders,
    compose_preconditioners,
    build_rhs1_pas_hybrid_preconditioner,
    build_rhs1_pas_lite_preconditioner,
    build_rhs1_pas_schur_preconditioner,
)
from sfincs_jax.solvers.preconditioner_dispatch import (
    RHS1PreconditionerDispatchBuilders,
    build_rhs1_preconditioner_from_kind,
)


def _op(
    *,
    n_theta: int = 8,
    n_zeta: int = 1,
    has_er_xdot: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        n_theta=n_theta,
        n_zeta=n_zeta,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=jnp.asarray([2, 2], dtype=jnp.int32)),
            pas=object(),
            fp=None,
            er_xdot=object() if has_er_xdot else None,
            er_xidot=None,
            exb_theta=None,
            exb_zeta=None,
        ),
    )


def _builder(transform):
    def _build(**_kwargs):
        def _apply(v):
            return transform(v)

        return _apply

    return _build


def _builders(
    *,
    tokamak_applicable: bool = False,
    tz_applicable: bool = False,
) -> RHS1PasCompositeBuilders:
    return RHS1PasCompositeBuilders(
        pas_tokamak_theta_applicable=lambda _op: tokamak_applicable,
        pas_tz_applicable=lambda _op: tz_applicable,
        pas_tokamak_theta_builder=_builder(lambda v: v + 1.0),
        pas_tz_builder=_builder(lambda v: v + 2.0),
        theta_line_builder=_builder(lambda v: v + 3.0),
        zeta_line_builder=_builder(lambda v: v + 4.0),
        xblock_tz_lmax_builder=_builder(lambda v: v + 5.0),
        xmg_builder=_builder(lambda v: 2.0 * v),
        xupwind_builder=_builder(lambda v: 3.0 * v),
        collision_builder=_builder(lambda v: v - 7.0),
    )


def test_compose_preconditioners_applies_second_after_first() -> None:
    composed = compose_preconditioners(lambda v: v + 2.0, lambda v: 3.0 * v)

    result = composed(jnp.asarray([4.0]))

    assert float(result[0]) == 18.0


def test_pas_lite_composes_angular_xcoarse_and_collision() -> None:
    precond = build_rhs1_pas_lite_preconditioner(
        op=_op(n_theta=8, n_zeta=1),
        builders=_builders(tokamak_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xmg(tokamak_theta(v))) = (5 + 1) * 2 - 7.
    assert float(result[0]) == 5.0


def test_pas_lite_respects_tz_size_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_TZ_MAX", "10")
    precond = build_rhs1_pas_lite_preconditioner(
        op=_op(n_theta=20, n_zeta=20),
        builders=_builders(tz_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # The theta-zeta angular stage is skipped, so only xmg and collision apply.
    assert float(result[0]) == 3.0


def test_pas_lite_invalid_tz_gate_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_LITE_TZ_MAX", "not-an-int")
    precond = build_rhs1_pas_lite_preconditioner(
        op=_op(n_theta=8, n_zeta=3),
        builders=_builders(tz_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # Invalid user input keeps the conservative default cap and admits PAS TZ.
    assert float(result[0]) == 7.0


def test_pas_hybrid_uses_line_then_xupwind_for_er_xdot_pas() -> None:
    precond = build_rhs1_pas_hybrid_preconditioner(
        op=_op(n_theta=8, n_zeta=1, has_er_xdot=True),
        builders=_builders(),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xupwind(theta_line(v))) = (5 + 3) * 3 - 7.
    assert float(result[0]) == 17.0


def test_pas_hybrid_uses_zeta_line_for_zeta_dominated_grids() -> None:
    precond = build_rhs1_pas_hybrid_preconditioner(
        op=_op(n_theta=4, n_zeta=8),
        builders=_builders(),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xmg(zeta_line(v))) = (5 + 4) * 2 - 7.
    assert float(result[0]) == 11.0


def test_pas_hybrid_uses_xblock_lmax_when_line_gate_rejects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_HYBRID_LINE_MAX", "0")
    monkeypatch.setenv("SFINCS_JAX_PAS_HYBRID_LMAX", "3")
    monkeypatch.setenv("SFINCS_JAX_PAS_HYBRID_XBLOCK_MAX", "1000")
    precond = build_rhs1_pas_hybrid_preconditioner(
        op=_op(n_theta=8, n_zeta=3, has_er_xdot=True),
        builders=_builders(),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xupwind(xblock_tz_lmax(v))) = (5 + 5) * 3 - 7.
    assert float(result[0]) == 23.0


def test_pas_schur_uses_pas_tz_then_xcoarse_and_collision() -> None:
    precond = build_rhs1_pas_schur_preconditioner(
        op=_op(n_theta=8, n_zeta=3),
        builders=_builders(tz_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xmg(pas_tz(v))) = (5 + 2) * 2 - 7.
    assert float(result[0]) == 7.0


def test_pas_schur_falls_back_to_hybrid_when_no_angular_model_applies() -> None:
    precond = build_rhs1_pas_schur_preconditioner(
        op=_op(n_theta=8, n_zeta=1),
        builders=_builders(),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # angular fallback = collision(xmg(theta_line(v))) = 9, then xmg and collision.
    assert float(result[0]) == 11.0


def _dispatch_builder_bundle(calls: list[tuple[str, dict[str, object]]]) -> RHS1PreconditionerDispatchBuilders:
    def _transform(name: str):
        if name in {"theta_line", "theta_line_xdiag"}:
            return lambda v: v + 1.0
        if name == "zeta_line":
            return lambda v: 3.0 * v
        if name == "collision":
            return lambda v: 2.0 * v
        return lambda v: v

    def _builder(name: str):
        def _build(**kwargs):
            calls.append((name, dict(kwargs)))
            return _transform(name)

        return _build

    names = (
        "theta_line_builder",
        "theta_dd_builder",
        "theta_schwarz_builder",
        "theta_line_xdiag_builder",
        "block_xdiag_builder",
        "species_block_builder",
        "sxblock_builder",
        "sxblock_tz_builder",
        "xblock_tz_builder",
        "xblock_tz_lmax_builder",
        "theta_zeta_builder",
        "xmg_builder",
        "pas_lite_builder",
        "pas_hybrid_builder",
        "pas_schur_builder",
        "pas_tz_builder",
        "pas_tzfft_builder",
        "pas_tokamak_theta_builder",
        "pas_ilu_builder",
        "zeta_line_builder",
        "zeta_dd_builder",
        "zeta_schwarz_builder",
        "schur_builder",
        "collision_builder",
        "structured_fblock_jacobi_builder",
        "structured_fblock_angular_jacobi_builder",
        "structured_fblock_xi_angular_jacobi_builder",
        "structured_fblock_fp_radial_jacobi_builder",
        "structured_fblock_fp_lowmode_schur_builder",
        "structured_fblock_fp_moment_schur_builder",
        "structured_fblock_fp_coupled_moment_schur_builder",
        "structured_fblock_fp_tail_coupled_schur_builder",
        "block_builder",
    )
    kwargs = {field: _builder(field.removesuffix("_builder")) for field in names}
    kwargs["compose_preconditioners"] = compose_preconditioners
    return RHS1PreconditionerDispatchBuilders(**kwargs)


def _dispatch_op(*, with_fp: bool = False, with_pas: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        fblock=SimpleNamespace(
            fp=object() if with_fp else None,
            pas=object() if with_pas else None,
        )
    )


def test_direct_dispatch_routes_domain_decomposition_overlap_policy() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    builders = _dispatch_builder_bundle(calls)

    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="theta_dd",
        builders=builders,
        dd_block_theta=11,
        dd_overlap_theta=0,
    )
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="theta_dd",
        builders=builders,
        dd_block_theta=13,
        dd_overlap_theta=2,
    )
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="zeta_dd",
        builders=builders,
        dd_block_zeta=7,
        dd_overlap_zeta=0,
    )
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="zeta_dd",
        builders=builders,
        dd_block_zeta=9,
        dd_overlap_zeta=3,
    )

    assert [(name, kwargs.get("block"), kwargs.get("overlap")) for name, kwargs in calls] == [
        ("theta_dd", 11, None),
        ("theta_schwarz", 13, 2),
        ("zeta_dd", 7, None),
        ("zeta_schwarz", 9, 3),
    ]


def test_direct_dispatch_emits_schwarz_progress_messages() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    messages: list[tuple[int, str]] = []

    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="theta_schwarz",
        builders=_dispatch_builder_bundle(calls),
        dd_block_theta=5,
        dd_overlap_theta=2,
        emit=lambda level, message: messages.append((level, message)),
    )
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="zeta_schwarz",
        builders=_dispatch_builder_bundle(calls),
        dd_block_zeta=6,
        dd_overlap_zeta=1,
        emit=lambda level, message: messages.append((level, message)),
    )

    assert [name for name, _kwargs in calls] == ["theta_schwarz", "zeta_schwarz"]
    assert "theta_schwarz (block=5, overlap=2)" in messages[0][1]
    assert "zeta_schwarz (block=6, overlap=1)" in messages[1][1]


def test_direct_dispatch_composes_collision_for_theta_line_xdiag_pas() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    precond = build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(with_pas=True),
        rhs1_precond_kind="theta_line_xdiag",
        builders=_dispatch_builder_bundle(calls),
    )

    result = precond(jnp.asarray([5.0]))

    assert [name for name, _kwargs in calls] == ["theta_line_xdiag", "collision"]
    assert float(result[0]) == 11.0


def test_direct_dispatch_repeats_adi_sweeps() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    precond = build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="adi",
        builders=_dispatch_builder_bundle(calls),
        adi_sweeps=2,
    )

    result = precond(jnp.asarray([2.0]))

    assert [name for name, _kwargs in calls] == ["theta_line", "zeta_line"]
    assert float(result[0]) == 30.0


def test_direct_dispatch_default_forwards_block_dimensions() -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(),
        rhs1_precond_kind="unknown",
        builders=_dispatch_builder_bundle(calls),
        preconditioner_species=2,
        preconditioner_x=3,
        preconditioner_xi=4,
    )

    assert calls[0][0] == "block"
    assert calls[0][1]["preconditioner_species"] == 2
    assert calls[0][1]["preconditioner_x"] == 3
    assert calls[0][1]["preconditioner_xi"] == 4


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("point_xdiag", "block_xdiag"),
        ("xblock_tz_lmax", "xblock_tz_lmax"),
        ("pas_lite", "pas_lite"),
        ("pas_hybrid", "pas_hybrid"),
        ("pas_schur", "pas_schur"),
        ("pas_tzfft", "pas_tzfft"),
        ("structured_fblock_fp_tail_coupled_schur", "structured_fblock_fp_tail_coupled_schur"),
    ],
)
def test_direct_dispatch_routes_named_preconditioner_families(kind: str, expected: str) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    build_rhs1_preconditioner_from_kind(
        op=_dispatch_op(with_fp=True, with_pas=True),
        rhs1_precond_kind=kind,
        builders=_dispatch_builder_bundle(calls),
        rhs1_xblock_tz_lmax=6,
        preconditioner_xi=5,
    )

    assert calls[0][0] == expected
    if kind == "xblock_tz_lmax":
        assert calls[0][1]["lmax"] == 6
    if kind == "point_xdiag":
        assert calls[0][1]["preconditioner_xi"] == 5
