from __future__ import annotations

from types import SimpleNamespace

import numpy as np

import sfincs_jax.rhs1_pas_policy as pas_policy
import sfincs_jax.v3_driver as vd


def _pas_tokamak_like_op(*, n_zeta: int = 3, zeta_varying: bool = False, rhs_mode: int = 1):
    theta = 4
    zeta = n_zeta
    base = np.asarray(
        [
            [1.0, 1.0, 1.0],
            [1.1, 1.1, 1.1],
            [1.2, 1.2, 1.2],
            [1.3, 1.3, 1.3],
        ],
        dtype=np.float64,
    )[:, :zeta]
    if zeta_varying and zeta > 1:
        base = base.copy()
        base[0, -1] += 1.0e-2
    collisionless = SimpleNamespace(
        b_hat=base,
        b_hat_sup_theta=2.0 * base,
        b_hat_sup_zeta=3.0 * base,
        db_hat_dtheta=0.5 * base,
        db_hat_dzeta=np.zeros_like(base),
    )
    fblock = SimpleNamespace(
        collisionless=collisionless,
        pas=object(),
        fp=None,
        fp_phi1=None,
        exb_theta=None,
        exb_zeta=None,
        magdrift_theta=None,
        magdrift_zeta=None,
        magdrift_xidot=None,
        er_xdot=None,
        er_xidot=None,
    )
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        n_zeta=zeta,
        n_theta=theta,
        n_xi=4,
        n_x=2,
        n_species=1,
        fblock=fblock,
    )


def _pas_tz_op(
    *,
    rhs_mode: int = 1,
    n_theta: int = 9,
    n_zeta: int = 9,
    n_xi: int = 4,
    with_pas: bool = True,
    with_fp: bool = False,
    with_drift: bool = False,
):
    collisionless = SimpleNamespace(n_xi_for_x=np.full((2,), n_xi, dtype=np.int32))
    fblock = SimpleNamespace(
        collisionless=collisionless,
        pas=object() if with_pas else None,
        fp=object() if with_fp else None,
        fp_phi1=None,
        exb_theta=object() if with_drift else None,
        exb_zeta=None,
        magdrift_theta=None,
        magdrift_zeta=None,
        magdrift_xidot=None,
        er_xdot=None,
        er_xidot=None,
    )
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        n_species=1,
        n_x=2,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        include_phi1=False,
        fblock=fblock,
    )


def test_pas_tokamak_theta_preconditioner_applicable_on_zeta_invariant_multizeta(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_PAS_TOKAMAK_TZ_TOL", raising=False)
    assert vd._pas_tokamak_theta_preconditioner_applicable(_pas_tokamak_like_op(n_zeta=3, zeta_varying=False))


def test_pas_tokamak_theta_preconditioner_applicable_rejects_zeta_variation_and_drifts(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_TZ_TOL", "bad")
    assert not vd._pas_tokamak_theta_preconditioner_applicable(_pas_tokamak_like_op(n_zeta=3, zeta_varying=True))

    op = _pas_tokamak_like_op(n_zeta=1, zeta_varying=False)
    op.fblock.exb_theta = object()
    assert not vd._pas_tokamak_theta_preconditioner_applicable(op)


def test_pas_tokamak_theta_builder_falls_back_to_block_preconditioner(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(vd, "_build_rhsmode1_block_preconditioner", lambda **kwargs: sentinel)
    op = _pas_tokamak_like_op(n_zeta=3, zeta_varying=True)
    assert vd._build_rhsmode1_pas_tokamak_theta_preconditioner(op=op) is sentinel


def test_pas_tz_preconditioner_applicable_positive_and_negative_cases() -> None:
    assert vd._pas_tz_preconditioner_applicable(_pas_tz_op())
    assert not vd._pas_tz_preconditioner_applicable(_pas_tz_op(rhs_mode=2))
    assert not vd._pas_tz_preconditioner_applicable(_pas_tz_op(n_theta=4, n_zeta=4))
    assert not vd._pas_tz_preconditioner_applicable(_pas_tz_op(n_xi=1))
    assert not vd._pas_tz_preconditioner_applicable(_pas_tz_op(with_pas=False))
    assert not vd._pas_tz_preconditioner_applicable(_pas_tz_op(with_fp=True))
    assert vd._pas_tz_preconditioner_applicable(_pas_tz_op(with_drift=True))


def test_rhs1_pas_tz_max_bytes_invalid_env_falls_back() -> None:
    assert vd._rhs1_pas_tz_max_bytes() > 0


def test_pas_tz_build_bytes_zero_for_inapplicable_and_positive_for_valid(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", raising=False)
    assert vd._estimate_rhs1_pas_tz_build_bytes(_pas_tz_op(rhs_mode=2)) == 0
    assert vd._estimate_rhs1_pas_tz_build_bytes(_pas_tz_op()) > 0


def test_pas_tz_memory_safe_respects_env_override(monkeypatch) -> None:
    op = _pas_tz_op(n_theta=17, n_zeta=17, n_xi=6)
    monkeypatch.setattr(pas_policy, "rhs1_pas_tz_max_bytes", lambda: 1)
    assert not vd._pas_tz_preconditioner_memory_safe(op)
    monkeypatch.setattr(pas_policy, "rhs1_pas_tz_max_bytes", lambda: 10**12)
    assert vd._pas_tz_preconditioner_memory_safe(op)


def test_pas_tz_builder_falls_back_to_hybrid_when_inapplicable(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(vd, "_build_rhsmode1_pas_hybrid_preconditioner", lambda **kwargs: sentinel)
    op = _pas_tz_op(rhs_mode=2)
    assert vd._build_rhsmode1_pas_tz_preconditioner(op=op) is sentinel


def test_pas_tz_builder_falls_back_to_hybrid_when_memory_unsafe_without_sharding(monkeypatch) -> None:
    sentinel = object()
    monkeypatch.setattr(vd, "_build_rhsmode1_pas_hybrid_preconditioner", lambda **kwargs: sentinel)
    monkeypatch.setattr(pas_policy, "estimate_rhs1_pas_tz_build_bytes", lambda _op: 10 * 2**30)
    monkeypatch.setattr(pas_policy, "rhs1_pas_tz_max_bytes", lambda: 2 * 2**30)
    monkeypatch.setattr(vd, "_matvec_shard_axis", lambda _op: None)
    monkeypatch.setattr(vd.jax, "device_count", lambda: 1)
    assert vd._build_rhsmode1_pas_tz_preconditioner(op=_pas_tz_op(n_theta=17, n_zeta=17, n_xi=6)) is sentinel


def test_pas_tz_builder_falls_back_to_theta_schwarz_when_memory_unsafe_and_theta_sharded(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _theta_builder(**kwargs):
        seen["block"] = kwargs["block"]
        seen["overlap"] = kwargs["overlap"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_theta_schwarz_preconditioner", _theta_builder)
    monkeypatch.setattr(pas_policy, "estimate_rhs1_pas_tz_build_bytes", lambda _op: 10 * 2**30)
    monkeypatch.setattr(pas_policy, "rhs1_pas_tz_max_bytes", lambda: 2 * 2**30)
    monkeypatch.setattr(vd, "_matvec_shard_axis", lambda _op: "theta")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_THETA_DD_BLOCK", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_THETA_DD_OVERLAP", raising=False)

    assert vd._build_rhsmode1_pas_tz_preconditioner(op=_pas_tz_op(n_theta=17, n_zeta=17, n_xi=6)) is sentinel
    assert seen == {"block": 64, "overlap": 1}


def test_pas_tz_builder_falls_back_to_zeta_schwarz_when_memory_unsafe_and_zeta_sharded(monkeypatch) -> None:
    sentinel = object()
    seen: dict[str, int] = {}

    def _zeta_builder(**kwargs):
        seen["block"] = kwargs["block"]
        seen["overlap"] = kwargs["overlap"]
        return sentinel

    monkeypatch.setattr(vd, "_build_rhsmode1_zeta_schwarz_preconditioner", _zeta_builder)
    monkeypatch.setattr(pas_policy, "estimate_rhs1_pas_tz_build_bytes", lambda _op: 10 * 2**30)
    monkeypatch.setattr(pas_policy, "rhs1_pas_tz_max_bytes", lambda: 2 * 2**30)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_PATCH_UNKNOWNS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_SCHWARZ_MAX_INVERSE_ENTRIES", "0")
    monkeypatch.setattr(vd, "_matvec_shard_axis", lambda _op: "zeta")
    monkeypatch.setattr(vd.jax, "device_count", lambda: 2)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ZETA_DD_BLOCK", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_ZETA_DD_OVERLAP", "bad")

    assert vd._build_rhsmode1_pas_tz_preconditioner(op=_pas_tz_op(n_theta=17, n_zeta=17, n_xi=6)) is sentinel
    assert seen == {"block": 64, "overlap": 1}
