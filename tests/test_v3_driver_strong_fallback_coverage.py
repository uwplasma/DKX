from __future__ import annotations

from types import SimpleNamespace

import sfincs_jax.v3_driver as vd


def _op(*, with_pas: bool = False):
    return SimpleNamespace(
        fblock=SimpleNamespace(
            pas=object() if with_pas else None,
        )
    )


def test_build_rhs1_strong_preconditioner_full_from_kind_forwards_dispatch_args(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)
    kind, preconditioner = vd._build_rhs1_strong_preconditioner_full_from_kind(
        op=_op(),
        strong_precond_kind="theta_schwarz",
        rhs1_precond_kind="point",
        residual_norm=1.0,
        dd_block_theta=17,
        dd_overlap_theta=3,
        dd_block_zeta=19,
        dd_overlap_zeta=4,
        adi_sweeps=5,
    )

    assert kind == "theta_schwarz"
    assert preconditioner is sentinel
    assert seen["rhs1_precond_kind"] == "theta_schwarz"
    assert seen["dd_block_theta"] == 17
    assert seen["dd_overlap_theta"] == 3
    assert seen["dd_block_zeta"] == 19
    assert seen["dd_overlap_zeta"] == 4
    assert seen["adi_sweeps"] == 5


def test_build_rhs1_strong_preconditioner_full_from_kind_downgrades_pas_schur_to_hybrid(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)
    kind, _preconditioner = vd._build_rhs1_strong_preconditioner_full_from_kind(
        op=_op(with_pas=True),
        strong_precond_kind="schur",
        rhs1_precond_kind="pas_lite",
        residual_norm=1.0,
    )

    assert kind == "pas_hybrid"
    assert seen["rhs1_precond_kind"] == "pas_hybrid"


def test_build_rhs1_strong_preconditioner_full_from_kind_returns_none_when_disabled(monkeypatch) -> None:
    called = {"value": False}

    def _dispatch(**kwargs):
        called["value"] = True
        return object()

    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)
    kind, preconditioner = vd._build_rhs1_strong_preconditioner_full_from_kind(
        op=_op(),
        strong_precond_kind=None,
        rhs1_precond_kind="point",
        residual_norm=1.0,
    )

    assert kind is None
    assert preconditioner is None
    assert not called["value"]
