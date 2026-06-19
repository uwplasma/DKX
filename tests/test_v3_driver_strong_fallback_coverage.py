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


def test_build_rhs1_strong_preconditioner_full_from_kind_uses_env_adi_sweeps(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "6")
    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)

    vd._build_rhs1_strong_preconditioner_full_from_kind(
        op=_op(),
        strong_precond_kind="adi",
        rhs1_precond_kind="point",
        residual_norm=1.0,
    )

    assert seen["rhs1_precond_kind"] == "adi"
    assert seen["adi_sweeps"] == 6

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "bad")
    vd._build_rhs1_strong_preconditioner_full_from_kind(
        op=_op(),
        strong_precond_kind="adi",
        rhs1_precond_kind="point",
        residual_norm=1.0,
    )
    assert seen["adi_sweeps"] == 2


def test_build_rhs1_strong_preconditioner_reduced_from_kind_forwards_dispatch_args(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return sentinel

    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)
    preconditioner = vd._build_rhs1_strong_preconditioner_reduced_from_kind(
        op=_op(),
        strong_precond_kind="theta_schwarz",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
        dd_block_theta=17,
        dd_overlap_theta=3,
        dd_block_zeta=19,
        dd_overlap_zeta=4,
    )

    assert preconditioner is sentinel
    assert seen["rhs1_precond_kind"] == "theta_schwarz"
    assert seen["dd_block_theta"] == 17
    assert seen["dd_overlap_theta"] == 3
    assert seen["dd_block_zeta"] == 19
    assert seen["dd_overlap_zeta"] == 4


def test_build_rhs1_strong_preconditioner_reduced_from_kind_uses_env_controls(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_TZ_LMAX", "6")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "4")
    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)

    vd._build_rhs1_strong_preconditioner_reduced_from_kind(
        op=_op(),
        strong_precond_kind="xblock_tz_lmax",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )

    assert seen["rhs1_precond_kind"] == "xblock_tz_lmax"
    assert seen["rhs1_xblock_tz_lmax"] == 6
    assert seen["adi_sweeps"] == 4


def test_build_rhs1_strong_preconditioner_reduced_from_kind_preserves_legacy_adi_fallback(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _dispatch(**kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ADI_SWEEPS", "bad")
    monkeypatch.setattr(vd, "_build_rhs1_preconditioner_from_kind", _dispatch)

    vd._build_rhs1_strong_preconditioner_reduced_from_kind(
        op=_op(),
        strong_precond_kind="unknown",
        reduce_full=lambda x: x,
        expand_reduced=lambda x: x,
    )

    assert seen["rhs1_precond_kind"] == "adi"
    assert seen["adi_sweeps"] == 2


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
