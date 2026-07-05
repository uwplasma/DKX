from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import sfincs_jax.solvers.preconditioner_pas_xblock_ilu as pas_xblock_ilu
import sfincs_jax.solvers.preconditioner_symbolic_host as symbolic_host
import sfincs_jax.solvers.preconditioner_xblock_tz_sparse as tz_sparse


class _DiagonalFactor:
    def __init__(self, diagonal: np.ndarray):
        self._diagonal = np.asarray(diagonal, dtype=np.float64)

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return np.asarray(rhs, dtype=np.float64) / self._diagonal


def _op() -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        total_size=5,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        constraint_scheme=1,
        quasineutrality_option=2,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        with_adiabatic=False,
        alpha=0.0,
        delta=0.0,
        dphi_hat_dpsi_hat=0.0,
        point_at_x0=False,
        adiabatic_z=np.asarray([], dtype=np.float64),
        adiabatic_nhat=np.asarray([], dtype=np.float64),
        adiabatic_that=np.asarray([], dtype=np.float64),
        z_s=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([1.0], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        n_hat=np.asarray([1.0], dtype=np.float64),
        theta_weights=np.ones((1,), dtype=np.float64),
        zeta_weights=np.ones((1,), dtype=np.float64),
        b_hat=np.ones((1, 1), dtype=np.float64),
        d_hat=np.ones((1, 1), dtype=np.float64),
        b_hat_sub_theta=np.zeros((1, 1), dtype=np.float64),
        b_hat_sub_zeta=np.zeros((1, 1), dtype=np.float64),
        x=np.asarray([0.1, 0.9], dtype=np.float64),
        x_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 1], dtype=np.int32)),
            fp=None,
            pas=None,
        ),
    )


def _cache_key_op() -> SimpleNamespace:
    return SimpleNamespace(
        rhs_mode=1,
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=3,
        n_zeta=1,
        constraint_scheme=1,
        quasineutrality_option=2,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        with_adiabatic=False,
        alpha=0.0,
        delta=0.0,
        dphi_hat_dpsi_hat=0.0,
        adiabatic_z=np.asarray([], dtype=np.float64),
        adiabatic_nhat=np.asarray([], dtype=np.float64),
        adiabatic_that=np.asarray([], dtype=np.float64),
        z_s=np.asarray([1.0], dtype=np.float64),
        m_hat=np.asarray([1.0], dtype=np.float64),
        t_hat=np.asarray([1.0], dtype=np.float64),
        n_hat=np.asarray([1.0], dtype=np.float64),
        theta_weights=np.ones((3,), dtype=np.float64),
        zeta_weights=np.ones((1,), dtype=np.float64),
        b_hat=np.ones((3, 1), dtype=np.float64),
        d_hat=np.ones((3, 1), dtype=np.float64),
        b_hat_sub_theta=np.zeros((3, 1), dtype=np.float64),
        b_hat_sub_zeta=np.zeros((3, 1), dtype=np.float64),
        x=np.asarray([0.1, 0.9], dtype=np.float64),
        x_weights=np.asarray([0.5, 0.5], dtype=np.float64),
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 1], dtype=np.int32))),
    )


def _patch_sparse_host(monkeypatch, diag_by_index: dict[int, float]) -> None:
    tz_sparse._RHSMODE1_SPARSE_ILU_CACHE.clear()
    tz_sparse._RHSMODE1_SPARSE_SXBLOCK_HOST_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SXBLOCK_SPARSE_LU_MAX", "16")
    monkeypatch.setattr(tz_sparse, "rhsmode1_precond_cache_key", lambda _op, kind: ("unit", kind, id(_op)))
    monkeypatch.setattr(tz_sparse, "rhsmode1_host_factor_probe_ok", lambda *, factor, block_size: True)

    def _probe(_op, *, col_idx, row_idx, total_size, chunk_cols):
        col_idx = np.asarray(col_idx, dtype=np.int32)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        out = np.zeros((col_idx.size, row_idx.size), dtype=np.float64)
        for i, col in enumerate(col_idx):
            for j, row in enumerate(row_idx):
                if int(col) == int(row):
                    out[i, j] = float(diag_by_index[int(col)])
        return out

    def _factorize(*, a_csr_full, cache_key, **_kwargs):
        diagonal = np.asarray(a_csr_full.diagonal(), dtype=np.float64)
        tz_sparse._RHSMODE1_SPARSE_ILU_CACHE[cache_key] = SimpleNamespace(
            ilu=_DiagonalFactor(diagonal)
        )

    monkeypatch.setattr(tz_sparse, "matvec_submatrix_v3_unsharded", _probe)
    monkeypatch.setattr(tz_sparse, "factorize_sparse_matrix_csr_host", _factorize)


def test_rhsmode1_sparse_cache_key_wrappers_use_kind_and_dtype() -> None:
    op = _cache_key_op()

    pas_key = pas_xblock_ilu.rhsmode1_pas_xblock_precond_cache_key(op)
    tz_key = tz_sparse.rhsmode1_precond_cache_key(op, "xblock_tz_sparse")

    assert pas_key[0] == "pas_xblock_ilu"
    assert tz_key[0] == "xblock_tz_sparse"
    assert pas_key[2:] == tz_key[2:]
    assert pas_key != tz_key


def test_symbolic_host_factorize_sparse_matrix_csr_host_reuses_cache() -> None:
    symbolic_host._RHSMODE1_SPARSE_ILU_CACHE.clear()
    matrix = sp.csr_matrix(np.asarray([[4.0, 1.0], [0.5, 3.0]], dtype=np.float64))

    first_full, first_drop, first_factor = symbolic_host.factorize_sparse_matrix_csr_host(
        a_csr_full=matrix,
        cache_key=("unit-symbolic-host-factor",),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        factorization="lu",
    )
    second_full, second_drop, second_factor = symbolic_host.factorize_sparse_matrix_csr_host(
        a_csr_full=matrix,
        cache_key=("unit-symbolic-host-factor",),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        factorization="lu",
    )

    assert first_factor is second_factor
    assert first_full.shape == (2, 2)
    assert second_drop.shape == first_drop.shape


def test_xblock_tz_sparse_builder_direct_host_skip_path_is_bounded(monkeypatch) -> None:
    op = _op()
    op.total_size = 4
    op.extra_size = 0
    op.fblock.collisionless.n_xi_for_x = np.asarray([2, 2], dtype=np.int32)
    tz_sparse._RHSMODE1_SPARSE_XBLOCK_HOST_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_HOST_BLOCK_MAX", "1")

    preconditioner = tz_sparse.build_rhs1_xblock_tz_sparse_preconditioner(
        op=op,
        build_jax_factors=False,
        preconditioner_species=1,
        preconditioner_xi=1,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
    )
    rhs = jnp.asarray([1.0, -2.0, 3.0, -4.0], dtype=jnp.float64)

    np.testing.assert_allclose(np.asarray(preconditioner(rhs)), np.asarray(rhs), rtol=0.0, atol=0.0)


def test_sxblock_sparse_host_builder_solves_active_l_blocks_and_extra(monkeypatch) -> None:
    op = _op()
    _patch_sparse_host(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0, 4: 20.0})

    precond = tz_sparse.build_rhs1_sxblock_tz_sparse_host_preconditioner(
        op=op,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
    )
    result = precond(jnp.asarray([2.0, 4.0, 8.0, 160.0, 20.0]))

    assert np.allclose(np.asarray(result), np.asarray([1.0, 1.0, 1.0, 160.0, 1.0]))


def test_sxblock_sparse_host_builder_supports_reduced_vectors(monkeypatch) -> None:
    op = _op()
    _patch_sparse_host(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0, 4: 20.0})
    active = jnp.asarray([0, 2, 4], dtype=jnp.int32)

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    full_precond = tz_sparse.build_rhs1_sxblock_tz_sparse_host_preconditioner(
        op=op,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
    )
    reduced_precond = tz_sparse.build_rhs1_sxblock_tz_sparse_host_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
    )
    rhs_reduced = reduce_full(jnp.asarray([2.0, 4.0, 8.0, 160.0, 20.0]))

    expected = reduce_full(full_precond(expand_reduced(rhs_reduced)))
    np.testing.assert_allclose(np.asarray(reduced_precond(rhs_reduced)), np.asarray(expected))


def test_sxblock_sparse_host_builder_fails_closed_when_block_factors_fail(monkeypatch) -> None:
    op = _op()
    _patch_sparse_host(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0, 4: 20.0})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SXBLOCK_SPARSE_LU_MAX", "bad")

    def _raise_factorization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic block failure")

    messages: list[str] = []
    monkeypatch.setattr(tz_sparse, "factorize_sparse_matrix_csr_host", _raise_factorization)
    precond = tz_sparse.build_rhs1_sxblock_tz_sparse_host_preconditioner(
        op=op,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
        emit=lambda _level, message: messages.append(str(message)),
    )
    result = precond(jnp.asarray([2.0, 4.0, 8.0, 160.0, 20.0]))

    np.testing.assert_allclose(np.asarray(result), np.asarray([2.0, 4.0, 8.0, 160.0, 1.0]))
    assert any("factorization failed" in message for message in messages)


def test_sxblock_sparse_host_seed_drops_one_shot_factors(monkeypatch) -> None:
    op = _op()
    _patch_sparse_host(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0, 4: 20.0})

    seed = tz_sparse.compute_rhs1_sxblock_tz_sparse_host_seed(
        op=op,
        rhs_reduced=jnp.asarray([2.0, 4.0, 8.0, 160.0, 20.0]),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
    )

    assert np.allclose(np.asarray(seed), np.asarray([1.0, 1.0, 1.0, 160.0, 1.0]))
    assert tz_sparse._RHSMODE1_SPARSE_ILU_CACHE == {}


def test_sxblock_sparse_host_seed_fails_closed_and_supports_reduced_vectors(monkeypatch) -> None:
    op = _op()
    _patch_sparse_host(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0, 4: 20.0})
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SXBLOCK_CHUNK_COLS", "bad")

    def _raise_factorization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic seed failure")

    active = jnp.asarray([0, 2, 4], dtype=jnp.int32)

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    messages: list[str] = []
    monkeypatch.setattr(tz_sparse, "factorize_sparse_matrix_csr_host", _raise_factorization)
    seed = tz_sparse.compute_rhs1_sxblock_tz_sparse_host_seed(
        op=op,
        rhs_reduced=reduce_full(jnp.asarray([2.0, 4.0, 8.0, 160.0, 20.0])),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=1.0,
        emit=lambda _level, message: messages.append(str(message)),
    )

    np.testing.assert_allclose(np.asarray(seed), np.asarray([2.0, 8.0, 1.0]))
    assert any("block solve failed" in message for message in messages)
    assert tz_sparse._RHSMODE1_SPARSE_ILU_CACHE == {}
