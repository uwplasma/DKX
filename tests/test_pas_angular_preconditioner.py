from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.solvers.preconditioner_pas_angular as pa


def _periodic_derivative(n: int, scale: float) -> np.ndarray:
    derivative = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return derivative
    for i in range(n):
        derivative[i, (i + 1) % n] = 0.5 * scale
        derivative[i, (i - 1) % n] = -0.5 * scale
    return derivative


def _pas_operator(*, n_zeta: int, n_theta: int = 3, n_l: int = 4) -> SimpleNamespace:
    n_species = 1
    n_x = 2
    f_shape = (n_species, n_x, n_l, n_theta, n_zeta)
    f_size = int(np.prod(f_shape))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    b_hat = 1.0 + 0.04 * np.cos(theta)[:, None] + 0.03 * np.sin(zeta)[None, :]
    b_sup_theta = 0.35 + 0.02 * np.sin(theta)[:, None] + np.zeros((n_theta, n_zeta))
    b_sup_zeta = 0.12 + 0.01 * np.cos(zeta)[None, :] + np.zeros((n_theta, n_zeta))
    db_dtheta = -0.04 * np.sin(theta)[:, None] + np.zeros((n_theta, n_zeta))
    db_dzeta = 0.03 * np.cos(zeta)[None, :] + np.zeros((n_theta, n_zeta))
    pas_coef = np.zeros((n_species, n_x, n_l), dtype=np.float64)
    for ix in range(n_x):
        pas_coef[0, ix, :] = 0.8 + 0.15 * ix + 0.25 * np.arange(n_l)
    collisionless = SimpleNamespace(
        x=np.asarray([0.35, 0.8], dtype=np.float64),
        ddtheta=_periodic_derivative(n_theta, 1.0),
        ddzeta=_periodic_derivative(n_zeta, 0.7),
        b_hat=b_hat,
        b_hat_sup_theta=b_sup_theta,
        b_hat_sup_zeta=b_sup_zeta,
        db_hat_dtheta=db_dtheta,
        db_hat_dzeta=db_dzeta,
        t_hats=np.asarray([1.4], dtype=np.float64),
        m_hats=np.asarray([2.0], dtype=np.float64),
        n_xi_for_x=np.asarray([n_l, n_l - 1], dtype=np.int32),
    )
    fblock = SimpleNamespace(
        f_shape=f_shape,
        identity_shift=0.6,
        pas=SimpleNamespace(coef=jnp.asarray(pas_coef, dtype=jnp.float64)),
        collisionless=collisionless,
        exb_theta=None,
        exb_zeta=None,
        magdrift_theta=None,
        magdrift_zeta=None,
        magdrift_xidot=None,
        er_xdot=None,
        er_xidot=None,
    )
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_l,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=f_size + 3,
        fblock=fblock,
    )


def _vector(op: SimpleNamespace) -> jnp.ndarray:
    return jnp.cos(0.09 * jnp.arange(op.total_size, dtype=jnp.float64)) - 0.2


def _inactive_indices(op: SimpleNamespace) -> np.ndarray:
    indices: list[int] = []
    n_l = int(op.n_xi)
    n_theta = int(op.n_theta)
    n_zeta = int(op.n_zeta)
    for ix, n_active in enumerate(np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)):
        for ell in range(int(n_active), n_l):
            for it in range(n_theta):
                for iz in range(n_zeta):
                    indices.append(((((0 * op.n_x + ix) * n_l + ell) * n_theta + it) * n_zeta + iz))
    return np.asarray(indices, dtype=np.int32)


def _fallback_builder(scale: float):
    def _builder(**_kwargs):
        return lambda vector: scale * vector

    return _builder


def test_pas_tokamak_theta_preconditioner_masks_inactive_pitch_and_reuses_cache(monkeypatch) -> None:
    op = _pas_operator(n_zeta=1)
    monkeypatch.setattr(pa, "_rhsmode1_precond_cache_key", lambda _op, kind: ("tokamak", kind, id(_op)))
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_LMAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TOKAMAK_REG", "bad")
    monkeypatch.delenv("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", raising=False)
    pa._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.clear()

    preconditioner = pa.build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        block_preconditioner_builder=_fallback_builder(9.0),
        pas_tokamak_theta_applicable=lambda _op: True,
    )
    vector = _vector(op)
    result = preconditioner(vector)

    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    np.testing.assert_allclose(np.asarray(result[op.f_size :]), np.asarray(vector[op.f_size :]))
    np.testing.assert_allclose(np.asarray(result)[_inactive_indices(op)], 0.0, atol=1e-12)
    assert len(pa._RHSMODE1_PAS_TOKAMAK_THETA_CACHE) == 1

    second = pa.build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        block_preconditioner_builder=_fallback_builder(9.0),
        pas_tokamak_theta_applicable=lambda _op: True,
    )
    assert len(pa._RHSMODE1_PAS_TOKAMAK_THETA_CACHE) == 1
    np.testing.assert_allclose(np.asarray(second(vector)), np.asarray(result))


def test_pas_tokamak_theta_structured_tail_and_reduced_application(monkeypatch) -> None:
    op = _pas_operator(n_zeta=1)
    monkeypatch.setattr(pa, "_rhsmode1_precond_cache_key", lambda _op, kind: ("tokamak-structured", kind, id(_op)))
    monkeypatch.setenv("SFINCS_JAX_PAS_TOKAMAK_STRUCTURED", "1")
    pa._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.clear()

    full_preconditioner = pa.build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        block_preconditioner_builder=_fallback_builder(5.0),
        pas_tokamak_theta_applicable=lambda _op: True,
    )
    cache = next(iter(pa._RHSMODE1_PAS_TOKAMAK_THETA_CACHE.values()))
    assert cache.tail_factors is not None
    assert cache.tail_factors[0][0] is not None

    active = jnp.arange(op.total_size, dtype=jnp.int32)[::5]

    def reduce_full(vector: jnp.ndarray) -> jnp.ndarray:
        return vector[active]

    def expand_reduced(vector: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(vector)

    reduced_preconditioner = pa.build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        block_preconditioner_builder=_fallback_builder(5.0),
        pas_tokamak_theta_applicable=lambda _op: True,
    )
    reduced_rhs = jnp.sin(0.21 * jnp.arange(active.size, dtype=jnp.float64))

    expected = reduce_full(full_preconditioner(expand_reduced(reduced_rhs)))
    np.testing.assert_allclose(np.asarray(reduced_preconditioner(reduced_rhs)), np.asarray(expected))


def test_pas_tokamak_theta_falls_back_when_not_applicable() -> None:
    op = _pas_operator(n_zeta=1)
    preconditioner = pa.build_rhs1_pas_tokamak_theta_preconditioner(
        op=op,
        block_preconditioner_builder=_fallback_builder(4.0),
        pas_tokamak_theta_applicable=lambda _op: False,
    )
    vector = _vector(op)
    np.testing.assert_allclose(np.asarray(preconditioner(vector)), np.asarray(4.0 * vector))


def test_pas_tz_preconditioner_masks_inactive_pitch_and_reuses_cache(monkeypatch) -> None:
    op = _pas_operator(n_zeta=2, n_theta=2)
    monkeypatch.setattr(pa, "_rhsmode1_precond_cache_key", lambda _op, kind: ("tz", kind, id(_op)))
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_LMAX", "bad")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PAS_TZ_REG", "bad")
    pa._RHSMODE1_PAS_TZ_CACHE.clear()

    preconditioner = pa.build_rhs1_pas_tz_preconditioner(
        op=op,
        pas_tz_applicable=lambda _op: True,
        pas_tz_memory_safe=lambda _op: True,
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=_fallback_builder(2.0),
        zeta_schwarz_builder=_fallback_builder(3.0),
        pas_hybrid_builder=_fallback_builder(4.0),
        collision_builder=_fallback_builder(5.0),
        tzfft_builder=_fallback_builder(6.0),
    )
    vector = _vector(op)
    result = preconditioner(vector)

    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    np.testing.assert_allclose(np.asarray(result[op.f_size :]), np.asarray(vector[op.f_size :]))
    np.testing.assert_allclose(np.asarray(result)[_inactive_indices(op)], 0.0, atol=1e-12)
    assert len(pa._RHSMODE1_PAS_TZ_CACHE) == 1

    second = pa.build_rhs1_pas_tz_preconditioner(
        op=op,
        pas_tz_applicable=lambda _op: True,
        pas_tz_memory_safe=lambda _op: True,
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=_fallback_builder(2.0),
        zeta_schwarz_builder=_fallback_builder(3.0),
        pas_hybrid_builder=_fallback_builder(4.0),
        collision_builder=_fallback_builder(5.0),
        tzfft_builder=_fallback_builder(6.0),
    )
    assert len(pa._RHSMODE1_PAS_TZ_CACHE) == 1
    np.testing.assert_allclose(np.asarray(second(vector)), np.asarray(result))


def test_pas_tz_fallback_branches(monkeypatch) -> None:
    op = _pas_operator(n_zeta=2, n_theta=2)
    vector = _vector(op)

    not_applicable = pa.build_rhs1_pas_tz_preconditioner(
        op=op,
        pas_tz_applicable=lambda _op: False,
        pas_tz_memory_safe=lambda _op: True,
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=_fallback_builder(2.0),
        zeta_schwarz_builder=_fallback_builder(3.0),
        pas_hybrid_builder=_fallback_builder(4.0),
        collision_builder=_fallback_builder(5.0),
        tzfft_builder=_fallback_builder(6.0),
    )
    np.testing.assert_allclose(np.asarray(not_applicable(vector)), np.asarray(4.0 * vector))

    calls: list[str] = []

    def memory_fallback(**_kwargs):
        calls.append("memory")
        return lambda rhs: 7.0 * rhs

    monkeypatch.setattr(pa, "build_pas_tz_memory_fallback", memory_fallback)
    unsafe = pa.build_rhs1_pas_tz_preconditioner(
        op=op,
        pas_tz_applicable=lambda _op: True,
        pas_tz_memory_safe=lambda _op: False,
        matvec_shard_axis=lambda _op: None,
        device_count=lambda: 1,
        theta_schwarz_builder=_fallback_builder(2.0),
        zeta_schwarz_builder=_fallback_builder(3.0),
        pas_hybrid_builder=_fallback_builder(4.0),
        collision_builder=_fallback_builder(5.0),
        tzfft_builder=_fallback_builder(6.0),
    )

    assert calls == ["memory"]
    np.testing.assert_allclose(np.asarray(unsafe(vector)), np.asarray(7.0 * vector))
