from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.solvers.preconditioner_full_fp_species as sb


def _op(
    *,
    n_species: int,
    n_theta: int,
) -> SimpleNamespace:
    f_size = int(n_species * n_theta)
    return SimpleNamespace(
        n_species=n_species,
        n_x=1,
        n_xi=1,
        n_theta=n_theta,
        n_zeta=1,
        total_size=f_size,
        f_size=f_size,
        phi1_size=0,
        extra_size=0,
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1], dtype=np.int32))),
    )


def _patch_cache_and_probe(monkeypatch, diag_by_index: dict[int, float]) -> None:
    sb._RHSMODE1_PRECOND_CACHE.clear()
    sb._RHSMODE1_PRECOND_GLOBAL_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")
    monkeypatch.setattr(sb, "_cache_key", lambda _op, kind: ("unit", kind, id(_op)))

    def _probe(_op, *, col_idx, row_idx, total_size, chunk_cols):
        col_idx = np.asarray(col_idx, dtype=np.int32)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        out = np.zeros((col_idx.size, row_idx.size), dtype=np.float64)
        for i, col in enumerate(col_idx):
            for j, row in enumerate(row_idx):
                if int(col) == int(row):
                    out[i, j] = float(diag_by_index[int(col)])
        return out

    monkeypatch.setattr(sb, "matvec_submatrix_v3_unsharded", _probe)


def test_species_block_builder_inverts_each_species_block(monkeypatch) -> None:
    op = _op(n_species=2, n_theta=1)
    _patch_cache_and_probe(monkeypatch, {0: 2.0, 1: 4.0})

    precond = sb.build_rhs1_species_block_preconditioner(op=op)
    result = precond(jnp.asarray([1.0, 2.0]))

    assert np.allclose(np.asarray(result), np.asarray([0.5, 0.5]))


def test_species_xblock_builder_reuses_species_x_factor_at_each_angle(monkeypatch) -> None:
    op = _op(n_species=2, n_theta=2)
    _patch_cache_and_probe(monkeypatch, {0: 2.0, 2: 4.0})

    precond = sb.build_rhs1_species_xblock_preconditioner(op=op)
    result = precond(jnp.asarray([2.0, 4.0, 8.0, 12.0]))

    assert np.allclose(np.asarray(result), np.asarray([1.0, 2.0, 2.0, 3.0]))
