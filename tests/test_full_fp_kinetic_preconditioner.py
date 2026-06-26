from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioner_full_fp_kinetic import build_rhs1_collision_preconditioner
from sfincs_jax.solvers.preconditioning import (
    _RHSMODE1_DIAG_PRECOND_CACHE,
    _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE,
    _RHSMODE1_SXBLOCK_PRECOND_CACHE,
    _RHSMODE1_XBLOCK_PRECOND_CACHE,
)


def _clear_collision_caches() -> None:
    _RHSMODE1_DIAG_PRECOND_CACHE.clear()
    _RHSMODE1_XBLOCK_PRECOND_CACHE.clear()
    _RHSMODE1_SXBLOCK_PRECOND_CACHE.clear()
    _RHSMODE1_SXBLOCK_LR_PRECOND_CACHE.clear()


def _fp_matrix(*, n_species: int = 2, n_x: int = 3, n_l: int = 3) -> np.ndarray:
    matrix = np.zeros((n_species, n_species, n_l, n_x, n_x), dtype=np.float64)
    for ell in range(n_l):
        for species in range(n_species):
            diagonal = 3.0 + 0.7 * species + 0.4 * ell + 0.11 * np.arange(n_x)
            block = np.diag(diagonal)
            block += 0.015 * (np.ones((n_x, n_x), dtype=np.float64) - np.eye(n_x))
            matrix[species, species, ell, :, :] = block
        if n_species > 1:
            cross = 0.02 * (ell + 1.0) * np.eye(n_x, dtype=np.float64)
            matrix[0, 1, ell, :, :] = cross
            matrix[1, 0, ell, :, :] = cross
    return matrix


def _fp_operator(*, include_pas: bool = True) -> SimpleNamespace:
    n_species = 2
    n_x = 3
    n_l = 3
    n_theta = 2
    n_zeta = 2
    f_shape = (n_species, n_x, n_l, n_theta, n_zeta)
    f_size = int(np.prod(f_shape))
    fp = SimpleNamespace(mat=jnp.asarray(_fp_matrix(n_species=n_species, n_x=n_x, n_l=n_l)))
    pas = None
    if include_pas:
        pas = SimpleNamespace(
            nu_n=0.35,
            krook=0.2,
            nu_d_hat=jnp.asarray(
                [[1.0, 1.15, 1.3], [0.85, 1.05, 1.2]],
                dtype=jnp.float64,
            ),
        )
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_l,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=f_size + 2,
        fblock=SimpleNamespace(
            f_shape=f_shape,
            identity_shift=0.5,
            fp=fp,
            pas=pas,
            er_xdot=None,
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([3, 2, 1], dtype=np.int32)),
        ),
    )


def _test_vector(op: SimpleNamespace) -> jnp.ndarray:
    return jnp.sin(0.17 * jnp.arange(op.total_size, dtype=jnp.float64)) + 0.1


def test_collision_diag_preconditioner_respects_fp_pas_and_inactive_pitch(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "diag")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "0")
    _clear_collision_caches()
    op = _fp_operator()

    preconditioner = build_rhs1_collision_preconditioner(op=op)
    vector = _test_vector(op)
    result = np.asarray(preconditioner(vector))

    f = np.asarray(vector[: op.f_size].reshape(op.fblock.f_shape))
    mat = np.asarray(op.fblock.fp.mat)
    pas = op.fblock.pas
    expected_diag = np.zeros(op.fblock.f_shape, dtype=np.float64)
    for species in range(op.n_species):
        for ix in range(op.n_x):
            for ell in range(op.n_xi):
                if ell >= op.fblock.collisionless.n_xi_for_x[ix]:
                    diag = 1.0
                else:
                    factor_l = 0.5 * (ell * (ell + 1.0) + 2.0 * float(pas.krook))
                    diag = (
                        float(op.fblock.identity_shift)
                        + mat[species, species, ell, ix, ix]
                        + float(pas.nu_n) * np.asarray(pas.nu_d_hat)[species, ix] * factor_l
                    )
                expected_diag[species, ix, ell, :, :] = diag
    expected = np.concatenate([(f / expected_diag).reshape((-1,)), np.asarray(vector[op.f_size :])])

    np.testing.assert_allclose(result, expected, rtol=2e-6, atol=2e-6)
    assert len(_RHSMODE1_DIAG_PRECOND_CACHE) == 1


def test_collision_xblock_and_sxblock_build_finite_cached_factors(monkeypatch) -> None:
    op = _fp_operator()
    vector = _test_vector(op)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_BICGSTAB_PRECOND_REG", "not-a-float")
    _clear_collision_caches()
    xblock_preconditioner = build_rhs1_collision_preconditioner(op=op)
    xblock_result = xblock_preconditioner(vector)
    assert xblock_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(xblock_result)))
    assert len(_RHSMODE1_XBLOCK_PRECOND_CACHE) == 1

    second_xblock = build_rhs1_collision_preconditioner(op=op)
    assert len(_RHSMODE1_XBLOCK_PRECOND_CACHE) == 1
    np.testing.assert_allclose(np.asarray(second_xblock(vector)), np.asarray(xblock_result))

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "sxblock")
    _clear_collision_caches()
    sxblock_preconditioner = build_rhs1_collision_preconditioner(op=op)
    sxblock_result = sxblock_preconditioner(vector)
    assert sxblock_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(sxblock_result)))
    assert len(_RHSMODE1_SXBLOCK_PRECOND_CACHE) == 1


def test_collision_sxblock_low_rank_and_auto_threshold_parsing(monkeypatch) -> None:
    op = _fp_operator(include_pas=False)
    vector = _test_vector(op)

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_SXBLOCK_MAX", "bad-int")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_XBLOCK_MAX", "bad-int")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FP_LOW_RANK_K", "1")
    _clear_collision_caches()

    preconditioner = build_rhs1_collision_preconditioner(op=op)
    result = preconditioner(vector)

    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    assert len(_RHSMODE1_SXBLOCK_LR_PRECOND_CACHE) == 1


def test_collision_reduced_application_matches_projected_full_application(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_COLLISION_PRECOND_KIND", "sxblock")
    _clear_collision_caches()
    op = _fp_operator()
    active = jnp.arange(op.total_size, dtype=jnp.int32)[1::4]

    def reduce_full(vector: jnp.ndarray) -> jnp.ndarray:
        return vector[active]

    def expand_reduced(vector: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(vector)

    full_preconditioner = build_rhs1_collision_preconditioner(op=op)
    reduced_preconditioner = build_rhs1_collision_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )

    reduced_rhs = jnp.cos(0.13 * jnp.arange(active.size, dtype=jnp.float64))
    expected = reduce_full(full_preconditioner(expand_reduced(reduced_rhs)))

    np.testing.assert_allclose(np.asarray(reduced_preconditioner(reduced_rhs)), np.asarray(expected))
