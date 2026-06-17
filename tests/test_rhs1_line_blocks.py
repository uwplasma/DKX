from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioners.domain_decomposition import line_blocks


def _op() -> SimpleNamespace:
    return SimpleNamespace(
        n_species=1,
        n_x=1,
        n_xi=1,
        n_theta=2,
        n_zeta=2,
        total_size=4,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1], dtype=np.int32)),
        ),
    )


def test_zeta_line_builder_inverts_each_theta_line(monkeypatch) -> None:
    op = _op()
    line_blocks._RHSMODE1_PRECOND_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")
    monkeypatch.setattr(line_blocks, "_cache_key", lambda _op, kind: ("unit", kind, id(_op)))
    monkeypatch.setattr(line_blocks, "_build_rhsmode1_preconditioner_operator_zeta_line", lambda _op: _op)

    def _probe(_op, *, col_idx, row_idx, total_size, chunk_cols):
        col_idx = np.asarray(col_idx, dtype=np.int32)
        row_idx = np.asarray(row_idx, dtype=np.int32)
        out = np.zeros((col_idx.size, row_idx.size), dtype=np.float64)
        for i, col in enumerate(col_idx):
            for j, row in enumerate(row_idx):
                if int(col) == int(row):
                    out[i, j] = float(int(col) + 2)
        return out

    monkeypatch.setattr(line_blocks, "matvec_submatrix_v3_unsharded", _probe)

    precond = line_blocks.build_rhs1_zeta_line_preconditioner(op=op)
    result = precond(jnp.asarray([2.0, 3.0, 4.0, 5.0]))

    assert np.allclose(np.asarray(result), np.ones((4,)))
