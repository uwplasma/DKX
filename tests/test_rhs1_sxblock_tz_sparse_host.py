from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.xblock import tz_sparse


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
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 1], dtype=np.int32)),
        ),
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
