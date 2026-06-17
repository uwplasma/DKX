from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.xblock import block_jacobi as xb


def _op() -> SimpleNamespace:
    return SimpleNamespace(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        total_size=4,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 1], dtype=np.int32)),
            pas=None,
        ),
    )


def _patch_cache_and_probe(monkeypatch, diag_by_index: dict[int, float]) -> None:
    xb._RHSMODE1_PRECOND_LIST_CACHE.clear()
    xb._RHSMODE1_PRECOND_IDX_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")
    monkeypatch.setattr(xb, "_cache_key", lambda _op, kind: ("unit", kind, id(_op)))

    def _probe(_op, *, col_idx, row_idx, total_size, chunk_cols):
        col_idx = np.asarray(col_idx, dtype=np.int32)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        out = np.zeros((col_idx.size, row_idx.size), dtype=np.float64)
        for i, col in enumerate(col_idx):
            for j, row in enumerate(row_idx):
                if int(col) == int(row):
                    out[i, j] = float(diag_by_index[int(col)])
        return out

    monkeypatch.setattr(xb, "matvec_submatrix_v3_unsharded", _probe)


def test_xblock_tz_builder_inverts_each_species_x_block(monkeypatch) -> None:
    op = _op()
    _patch_cache_and_probe(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0})

    precond = xb.build_rhs1_xblock_tz_preconditioner(op=op)
    result = precond(jnp.asarray([2.0, 4.0, 8.0, 16.0]))

    assert np.allclose(np.asarray(result), np.asarray([1.0, 1.0, 1.0, 0.0]))


def test_xblock_tz_lmax_builder_leaves_high_l_modes_as_identity(monkeypatch) -> None:
    op = _op()
    _patch_cache_and_probe(monkeypatch, {0: 2.0, 2: 8.0})

    precond = xb.build_rhs1_xblock_tz_lmax_preconditioner(op=op, lmax=1)
    result = precond(jnp.asarray([2.0, 40.0, 8.0, 160.0]))

    assert np.allclose(np.asarray(result), np.asarray([1.0, 40.0, 1.0, 160.0]))


def test_sxblock_tz_builder_inverts_each_pitch_block(monkeypatch) -> None:
    op = _op()
    _patch_cache_and_probe(monkeypatch, {0: 2.0, 1: 4.0, 2: 8.0, 3: 16.0})

    precond = xb.build_rhs1_sxblock_tz_preconditioner(op=op)
    result = precond(jnp.asarray([2.0, 4.0, 8.0, 16.0]))

    assert np.allclose(np.asarray(result), np.asarray([1.0, 1.0, 1.0, 0.0]))
