from __future__ import annotations

from types import SimpleNamespace

import sfincs_jax.v3_driver as vd


def _op(*, with_pas: bool = False, with_fp: bool = False):
    return SimpleNamespace(
        fblock=SimpleNamespace(
            pas=object() if with_pas else None,
            fp=object() if with_fp else None,
        )
    )


def test_rhs1_dispatch_theta_dd_uses_dd_without_overlap(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["block"] = kwargs["block"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_theta_dd_preconditioner", _builder)
    assert (
        vd._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="theta_dd",
            dd_block_theta=11,
            dd_overlap_theta=0,
        )
        is sentinel
    )
    assert seen == {"block": 11}


def test_rhs1_dispatch_theta_dd_uses_schwarz_with_overlap(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["block"] = kwargs["block"]
        seen["overlap"] = kwargs["overlap"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_theta_schwarz_preconditioner", _builder)
    assert (
        vd._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="theta_dd",
            dd_block_theta=13,
            dd_overlap_theta=2,
        )
        is sentinel
    )
    assert seen == {"block": 13, "overlap": 2}


def test_rhs1_dispatch_point_xdiag_forwards_preconditioner_xi(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["preconditioner_xi"] = kwargs["preconditioner_xi"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_block_preconditioner_xdiag", _builder)
    assert (
        vd._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="point_xdiag",
            preconditioner_xi=7,
        )
        is sentinel
    )
    assert seen == {"preconditioner_xi": 7}


def test_rhs1_dispatch_xblock_tz_lmax_forwards_lmax(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["lmax"] = kwargs["lmax"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_xblock_tz_lmax_preconditioner", _builder)
    assert (
        vd._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="xblock_tz_lmax",
            rhs1_xblock_tz_lmax=5,
        )
        is sentinel
    )
    assert seen == {"lmax": 5}


def test_rhs1_dispatch_theta_line_xdiag_composes_collision_for_pas(monkeypatch) -> None:
    line = object()
    collision = object()
    sentinel = object()

    monkeypatch.setattr(vd, "_build_rhsmode1_theta_line_xdiag_preconditioner", lambda **kwargs: line)
    monkeypatch.setattr(vd, "_build_rhsmode1_collision_preconditioner", lambda **kwargs: collision)
    monkeypatch.setattr(vd, "_compose_preconditioners", lambda a, b: sentinel if (a, b) == (collision, line) else None)

    assert vd._build_rhs1_preconditioner_from_kind(op=_op(with_pas=True), rhs1_precond_kind="theta_line_xdiag") is sentinel


def test_rhs1_dispatch_default_falls_back_to_block_preconditioner(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _builder(**kwargs):
        seen["species"] = kwargs["preconditioner_species"]
        seen["x"] = kwargs["preconditioner_x"]
        seen["xi"] = kwargs["preconditioner_xi"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_block_preconditioner", _builder)
    assert (
        vd._build_rhs1_preconditioner_from_kind(
            op=_op(),
            rhs1_precond_kind="unknown-kind",
            preconditioner_species=2,
            preconditioner_x=3,
            preconditioner_xi=4,
        )
        is sentinel
    )
    assert seen == {"species": 2, "x": 3, "xi": 4}


def test_rhs1_dkes_gmres_budget_respects_explicit_limits() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=20,
        restart_forced=True,
        maxiter_forced=True,
        restart_cap_env="100",
    )

    assert (restart, maxiter) == (20, 20)
    assert restart_defaulted is False
    assert maxiter_defaulted is False


def test_rhs1_dkes_gmres_budget_applies_defaults_when_unforced() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=20,
        maxiter=20,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="90",
    )

    assert (restart, maxiter) == (80, 600)
    assert restart_defaulted is True
    assert maxiter_defaulted is True


def test_rhs1_dkes_gmres_budget_caps_unforced_restart() -> None:
    restart, maxiter, restart_defaulted, maxiter_defaulted = vd._rhs1_dkes_gmres_budget(
        restart=200,
        maxiter=None,
        restart_forced=False,
        maxiter_forced=False,
        restart_cap_env="100",
    )

    assert (restart, maxiter) == (100, 600)
    assert restart_defaulted is True
    assert maxiter_defaulted is True


def test_rhs1_pas_tz_guarded_structured_levels_parse_aliases() -> None:
    assert vd._rhs1_pas_tz_guarded_structured_levels("") == ()
    assert vd._rhs1_pas_tz_guarded_structured_levels("off") == ()
    assert vd._rhs1_pas_tz_guarded_structured_levels("structured") == ("xmg", "collision")
    assert vd._rhs1_pas_tz_guarded_structured_levels("x+coll+x") == ("xmg", "collision")
    assert vd._rhs1_pas_tz_guarded_structured_levels("unknown,collision_diag") == ("collision",)
