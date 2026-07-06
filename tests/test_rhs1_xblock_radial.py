from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.solvers.preconditioner_xblock_radial as radial
from sfincs_jax.solvers.preconditioning import (
    _RHSMODE1_XMG_PRECOND_CACHE,
    _RHSMODE1_XUPWIND_PRECOND_CACHE,
)


def _base_cache_fields(
    *,
    n_species: int,
    n_x: int,
    n_l: int,
    n_theta: int,
    n_zeta: int,
) -> dict[str, object]:
    shape_tz = (n_theta, n_zeta)
    return {
        "rhs_mode": 1,
        "n_species": n_species,
        "n_x": n_x,
        "n_xi": n_l,
        "n_theta": n_theta,
        "n_zeta": n_zeta,
        "constraint_scheme": 1,
        "quasineutrality_option": 1,
        "include_phi1": False,
        "include_phi1_in_kinetic": False,
        "with_adiabatic": False,
        "alpha": 1.0,
        "delta": 1.0,
        "dphi_hat_dpsi_hat": 0.0,
        "adiabatic_z": np.zeros((1,), dtype=np.float64),
        "adiabatic_nhat": np.zeros((1,), dtype=np.float64),
        "adiabatic_that": np.zeros((1,), dtype=np.float64),
        "z_s": np.ones((n_species,), dtype=np.float64),
        "m_hat": np.ones((n_species,), dtype=np.float64),
        "t_hat": np.ones((n_species,), dtype=np.float64),
        "n_hat": np.ones((n_species,), dtype=np.float64),
        "theta_weights": np.ones((n_theta,), dtype=np.float64) / float(n_theta),
        "zeta_weights": np.ones((n_zeta,), dtype=np.float64) / float(n_zeta),
        "b_hat": np.ones(shape_tz, dtype=np.float64),
        "d_hat": np.ones(shape_tz, dtype=np.float64),
        "b_hat_sub_theta": np.ones(shape_tz, dtype=np.float64),
        "b_hat_sub_zeta": np.zeros(shape_tz, dtype=np.float64),
        "x": np.linspace(0.25, 1.0, n_x, dtype=np.float64),
        "x_weights": np.ones((n_x,), dtype=np.float64) / float(n_x),
        "phi1_size": 0,
    }


def _fp_matrix(*, n_species: int, n_x: int, n_l: int) -> np.ndarray:
    mat = np.zeros((n_species, n_species, n_l, n_x, n_x), dtype=np.float64)
    for s in range(n_species):
        for ell in range(n_l):
            for ix in range(n_x):
                mat[s, s, ell, ix, ix] = 2.0 + s + ell + 0.25 * ix
    return mat


def _fp_op() -> SimpleNamespace:
    n_species, n_x, n_l, n_theta, n_zeta = 1, 4, 3, 2, 1
    f_shape = (n_species, n_x, n_l, n_theta, n_zeta)
    f_size = int(np.prod(f_shape))
    fields = _base_cache_fields(
        n_species=n_species,
        n_x=n_x,
        n_l=n_l,
        n_theta=n_theta,
        n_zeta=n_zeta,
    )
    fields["dphi_hat_dpsi_hat"] = -0.25
    fblock = SimpleNamespace(
        f_shape=f_shape,
        identity_shift=0.5,
        fp=SimpleNamespace(mat=jnp.asarray(_fp_matrix(n_species=n_species, n_x=n_x, n_l=n_l))),
        pas=None,
        er_xdot=None,
        collisionless=SimpleNamespace(n_xi_for_x=np.asarray([3, 3, 2, 1], dtype=np.int32)),
    )
    return SimpleNamespace(
        **fields,
        f_size=f_size,
        total_size=f_size + 2,
        extra_size=2,
        fblock=fblock,
    )


def _pas_er_op() -> SimpleNamespace:
    n_species, n_x, n_l, n_theta, n_zeta = 1, 4, 5, 2, 2
    f_shape = (n_species, n_x, n_l, n_theta, n_zeta)
    f_size = int(np.prod(f_shape))
    fields = _base_cache_fields(
        n_species=n_species,
        n_x=n_x,
        n_l=n_l,
        n_theta=n_theta,
        n_zeta=n_zeta,
    )
    x = np.linspace(0.2, 1.0, n_x, dtype=np.float64)
    fields["x"] = x
    ddx = np.tril(np.ones((n_x, n_x), dtype=np.float64), k=0)
    shape_tz = (n_theta, n_zeta)
    er_xdot = SimpleNamespace(
        alpha=1.0,
        delta=1.0,
        dphi_hat_dpsi_hat=-0.25,
        d_hat=np.ones(shape_tz, dtype=np.float64),
        b_hat=np.ones(shape_tz, dtype=np.float64),
        b_hat_sub_theta=np.ones(shape_tz, dtype=np.float64),
        b_hat_sub_zeta=np.zeros(shape_tz, dtype=np.float64),
        db_dtheta=np.zeros(shape_tz, dtype=np.float64),
        db_dzeta=np.ones(shape_tz, dtype=np.float64),
        db_hat_dtheta=np.zeros(shape_tz, dtype=np.float64),
        db_hat_dzeta=np.ones(shape_tz, dtype=np.float64),
        x=x,
        ddx_plus=ddx,
    )
    fblock = SimpleNamespace(
        f_shape=f_shape,
        identity_shift=0.25,
        fp=None,
        pas=SimpleNamespace(
            nu_n=0.3,
            krook=0.1,
            nu_d_hat=np.ones((n_species, n_x), dtype=np.float64),
        ),
        er_xdot=er_xdot,
        collisionless=SimpleNamespace(n_xi_for_x=np.asarray([5, 5, 3, 2], dtype=np.int32)),
    )
    return SimpleNamespace(
        **fields,
        f_size=f_size,
        total_size=f_size + 1,
        extra_size=1,
        fblock=fblock,
    )


def _er_xdot_for_op(op: SimpleNamespace) -> SimpleNamespace:
    """Build a tiny valid Er xDot fixture matching an existing test operator."""

    x = np.asarray(op.x, dtype=np.float64)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    shape_tz = (n_theta, n_zeta)
    ddx = np.tril(np.ones((int(op.n_x), int(op.n_x)), dtype=np.float64), k=0)
    return SimpleNamespace(
        alpha=1.0,
        delta=0.75,
        dphi_hat_dpsi_hat=-0.5,
        d_hat=np.ones(shape_tz, dtype=np.float64),
        b_hat=np.full(shape_tz, 1.25, dtype=np.float64),
        b_hat_sub_theta=np.full(shape_tz, 0.9, dtype=np.float64),
        b_hat_sub_zeta=np.full(shape_tz, -0.2, dtype=np.float64),
        db_dtheta=np.full(shape_tz, 0.15, dtype=np.float64),
        db_dzeta=np.full(shape_tz, 0.4, dtype=np.float64),
        db_hat_dtheta=np.full(shape_tz, 0.15, dtype=np.float64),
        db_hat_dzeta=np.full(shape_tz, 0.4, dtype=np.float64),
        x=x,
        ddx_plus=ddx,
    )


def test_xmg_radial_preconditioner_builds_cached_fp_coarse_factor(monkeypatch) -> None:
    op = _fp_op()
    _RHSMODE1_XMG_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "1e-6")

    vector = jnp.linspace(1.0, 2.0, op.total_size, dtype=jnp.float64)
    precond = radial.build_rhs1_xmg_preconditioner(op=op, stride_override=2)
    result = np.asarray(precond(vector))

    assert len(_RHSMODE1_XMG_PRECOND_CACHE) == 1
    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[op.f_size :], np.asarray(vector[op.f_size :]))

    second = radial.build_rhs1_xmg_preconditioner(op=op, stride_override=2)
    assert len(_RHSMODE1_XMG_PRECOND_CACHE) == 1
    np.testing.assert_allclose(np.asarray(second(vector)), result)


def test_xmg_radial_preconditioner_fp_with_er_coupling_is_finite(monkeypatch) -> None:
    """FP xMG should include valid Er xDot metadata without changing tail data."""

    op = _fp_op()
    op.fblock.er_xdot = _er_xdot_for_op(op)
    _RHSMODE1_XMG_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XMG_PINV_RCOND", "not-a-float")

    vector = jnp.linspace(-2.0, 2.0, op.total_size, dtype=jnp.float64)
    precond = radial.build_rhs1_xmg_preconditioner(op=op, stride_override=2)
    result = np.asarray(precond(vector))
    cached = next(iter(_RHSMODE1_XMG_PRECOND_CACHE.values()))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[op.f_size :], np.asarray(vector[op.f_size :]))
    assert cached.coarse_inv_lblock is None
    assert cached.lblock == 0
    assert cached.coarse_inv.shape[-2:] == (2, 2)
    assert np.any(np.abs(np.asarray(cached.coarse_inv)) > 0.0)


def test_xmg_radial_preconditioner_supports_reduced_vectors() -> None:
    op = _fp_op()
    _RHSMODE1_XMG_PRECOND_CACHE.clear()
    vector = jnp.linspace(0.25, 1.25, op.f_size, dtype=jnp.float64)

    def expand_reduced(reduced: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([reduced, jnp.zeros((op.extra_size,), dtype=jnp.float64)])

    def reduce_full(full: jnp.ndarray) -> jnp.ndarray:
        return full[: op.f_size]

    precond = radial.build_rhs1_xmg_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        stride_override=3,
    )
    reduced = np.asarray(precond(vector))

    assert reduced.shape == (op.f_size,)
    assert np.all(np.isfinite(reduced))


def test_xmg_radial_preconditioner_invalid_stride_env_falls_back_to_two(monkeypatch) -> None:
    op = _fp_op()
    _RHSMODE1_XMG_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XMG_STRIDE", "not-an-int")

    precond = radial.build_rhs1_xmg_preconditioner(op=op)
    result = np.asarray(precond(jnp.ones((op.total_size,), dtype=jnp.float64)))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    assert next(iter(_RHSMODE1_XMG_PRECOND_CACHE))[0] == "xmg_2"


def test_xmg_radial_preconditioner_large_fp_default_uses_coarser_stride() -> None:
    op = _fp_op()
    # The production-size decision uses the operator-reported full size. Keep
    # the fixture array small so the test checks admission logic, not runtime.
    op.total_size = 120_000
    _RHSMODE1_XMG_PRECOND_CACHE.clear()

    precond = radial.build_rhs1_xmg_preconditioner(op=op)
    result = np.asarray(precond(jnp.ones((op.f_size + op.extra_size,), dtype=jnp.float64)))

    assert result.shape == (op.f_size + op.extra_size,)
    assert next(iter(_RHSMODE1_XMG_PRECOND_CACHE))[0] == "xmg_4"


def test_xmg_radial_preconditioner_handles_pas_without_er_coupling() -> None:
    op = _pas_er_op()
    op.fblock.er_xdot = None
    _RHSMODE1_XMG_PRECOND_CACHE.clear()

    precond = radial.build_rhs1_xmg_preconditioner(op=op, stride_override=2)
    result = np.asarray(precond(jnp.linspace(0.1, 0.9, op.total_size, dtype=jnp.float64)))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    assert len(_RHSMODE1_XMG_PRECOND_CACHE) == 1


def test_xmg_radial_preconditioner_fail_closes_malformed_er_xdot_metadata() -> None:
    op = _fp_op()
    op.fblock.er_xdot = SimpleNamespace(alpha=object())
    _RHSMODE1_XMG_PRECOND_CACHE.clear()

    precond = radial.build_rhs1_xmg_preconditioner(op=op, stride_override=2)
    result = np.asarray(precond(jnp.ones((op.total_size,), dtype=jnp.float64)))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))


def test_pas_er_xmg_dispatches_to_xupwind_and_preserves_tail(monkeypatch) -> None:
    op = _pas_er_op()
    _RHSMODE1_XUPWIND_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XUPWIND_LBLOCK", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "1e-5")

    vector = jnp.linspace(-1.0, 1.0, op.total_size, dtype=jnp.float64)
    precond = radial.build_rhs1_xmg_preconditioner(op=op)
    result = np.asarray(precond(vector))

    assert len(_RHSMODE1_XUPWIND_PRECOND_CACHE) >= 1
    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    np.testing.assert_allclose(result[op.f_size :], np.asarray(vector[op.f_size :]))


def test_xupwind_radial_preconditioner_falls_back_to_xmg_for_fp_operator() -> None:
    op = _fp_op()
    _RHSMODE1_XMG_PRECOND_CACHE.clear()

    precond = radial.build_rhs1_xupwind_preconditioner(op=op)
    result = np.asarray(precond(jnp.ones((op.total_size,), dtype=jnp.float64)))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    assert len(_RHSMODE1_XMG_PRECOND_CACHE) == 1


def test_xupwind_radial_preconditioner_invalid_envs_remain_finite(monkeypatch) -> None:
    op = _pas_er_op()
    _RHSMODE1_XUPWIND_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XUPWIND_LBLOCK", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XUPWIND_PINV_RCOND", "bad")

    precond = radial.build_rhs1_xupwind_preconditioner(op=op)
    result = np.asarray(precond(jnp.linspace(-0.5, 0.5, op.total_size, dtype=jnp.float64)))
    cached = next(iter(_RHSMODE1_XUPWIND_PRECOND_CACHE.values()))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    assert cached.lblock == op.n_xi
    assert cached.block_inv is not None


def test_xupwind_radial_preconditioner_fail_closes_malformed_er_metadata() -> None:
    op = _pas_er_op()
    op.fblock.er_xdot = SimpleNamespace(
        alpha=object(),
        x=np.linspace(0.2, 1.0, op.n_x, dtype=np.float64),
    )
    _RHSMODE1_XUPWIND_PRECOND_CACHE.clear()

    precond = radial.build_rhs1_xupwind_preconditioner(op=op)
    result = np.asarray(precond(jnp.ones((op.total_size,), dtype=jnp.float64)))
    cached = next(iter(_RHSMODE1_XUPWIND_PRECOND_CACHE.values()))

    assert result.shape == (op.total_size,)
    assert np.all(np.isfinite(result))
    assert cached.block_inv is None


def test_xupwind_radial_preconditioner_can_apply_reduced_vector(monkeypatch) -> None:
    op = _pas_er_op()
    _RHSMODE1_XUPWIND_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XUPWIND_LBLOCK", "0")
    vector = jnp.linspace(1.0, 3.0, op.f_size, dtype=jnp.float64)

    def expand_reduced(reduced: jnp.ndarray) -> jnp.ndarray:
        return jnp.concatenate([reduced, jnp.zeros((op.extra_size,), dtype=jnp.float64)])

    def reduce_full(full: jnp.ndarray) -> jnp.ndarray:
        return full[: op.f_size]

    precond = radial.build_rhs1_xupwind_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    reduced = np.asarray(precond(vector))

    assert reduced.shape == (op.f_size,)
    assert np.all(np.isfinite(reduced))
