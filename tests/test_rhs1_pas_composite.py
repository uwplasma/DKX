from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp

from sfincs_jax.solvers.preconditioners.pas import (
    RHS1PasCompositeBuilders,
    build_rhs1_pas_hybrid_preconditioner,
    build_rhs1_pas_lite_preconditioner,
    build_rhs1_pas_schur_preconditioner,
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


def test_pas_lite_composes_angular_xcoarse_and_collision() -> None:
    precond = build_rhs1_pas_lite_preconditioner(
        op=_op(n_theta=8, n_zeta=1),
        builders=_builders(tokamak_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xmg(tokamak_theta(v))) = (5 + 1) * 2 - 7.
    assert float(result[0]) == 5.0


def test_pas_hybrid_uses_line_then_xupwind_for_er_xdot_pas() -> None:
    precond = build_rhs1_pas_hybrid_preconditioner(
        op=_op(n_theta=8, n_zeta=1, has_er_xdot=True),
        builders=_builders(),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xupwind(theta_line(v))) = (5 + 3) * 3 - 7.
    assert float(result[0]) == 17.0


def test_pas_schur_uses_pas_tz_then_xcoarse_and_collision() -> None:
    precond = build_rhs1_pas_schur_preconditioner(
        op=_op(n_theta=8, n_zeta=3),
        builders=_builders(tz_applicable=True),
        safe=False,
    )

    result = precond(jnp.asarray([5.0]))

    # collision(xmg(pas_tz(v))) = (5 + 2) * 2 - 7.
    assert float(result[0]) == 7.0
