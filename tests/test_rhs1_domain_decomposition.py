from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

import sfincs_jax.solvers.preconditioner_domain_decomposition as dd
from sfincs_jax.solvers.preconditioner_domain_decomposition import (
    _axis_line_index_map,
    _dd_core_patch_ranges,
    _extra_inverse,
    _regularization_from_env,
    _rhs1_dd_auto_block_size,
    _rhs1_dd_coarse_block_size,
    _rhs1_dd_coarse_block_sizes,
    _rhs1_dd_coarse_level_count,
    build_rhs1_theta_line_preconditioner,
    build_rhs1_theta_line_xdiag_preconditioner,
    build_rhs1_theta_schwarz_preconditioner,
    build_rhs1_theta_zeta_preconditioner,
    build_rhs1_zeta_line_preconditioner,
    build_rhs1_zeta_schwarz_preconditioner,
)


def _toy_op(*, n_theta: int = 2, n_zeta: int = 2, extra_size: int = 1):
    f_size = int(n_theta * n_zeta)
    return SimpleNamespace(
        n_species=1,
        n_x=1,
        n_xi=1,
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        f_size=f_size,
        phi1_size=0,
        extra_size=int(extra_size),
        total_size=f_size + int(extra_size),
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1], dtype=np.int32))),
    )


def _install_diagonal_toy_operator(monkeypatch, *, diagonal: float = 2.0) -> None:
    dd._RHSMODE1_PRECOND_CACHE.clear()
    dd._RHSMODE1_SCHWARZ_PRECOND_CACHE.clear()
    dd._RHSMODE1_THETA_LINE_DIAGX_CACHE.clear()
    monkeypatch.setattr(dd, "_cache_key", lambda op, kind: (id(op), kind))
    monkeypatch.setattr(dd, "_build_rhsmode1_preconditioner_operator_theta_line", lambda op: op)
    monkeypatch.setattr(dd, "_build_rhsmode1_preconditioner_operator_theta_dd", lambda op, block: op)
    monkeypatch.setattr(dd, "_build_rhsmode1_preconditioner_operator_zeta_line", lambda op: op)
    monkeypatch.setattr(dd, "_build_rhsmode1_preconditioner_operator_zeta_dd", lambda op, block: op)
    monkeypatch.setattr(dd, "precond_chunk_cols", lambda _total_size, n_cols: max(1, int(n_cols)))

    def fake_submatrix(_op, *, col_idx, row_idx, total_size, chunk_cols):
        del total_size, chunk_cols
        row = np.asarray(row_idx, dtype=np.int32)
        col = np.asarray(col_idx, dtype=np.int32)
        out = np.zeros((col.size, row.size), dtype=np.float64)
        for j, col_value in enumerate(col):
            for i, row_value in enumerate(row):
                if int(col_value) == int(row_value):
                    out[j, i] = float(diagonal)
        return out

    monkeypatch.setattr(dd, "matvec_submatrix_v3_unsharded", fake_submatrix)


def _assert_half_inverse(preconditioner, *, n: int) -> None:
    residual = jnp.arange(1, n + 1, dtype=jnp.float64)
    actual = np.asarray(preconditioner(residual), dtype=np.float64)
    np.testing.assert_allclose(actual, 0.5 * np.asarray(residual), rtol=1.0e-9, atol=1.0e-9)


def test_dd_core_patch_ranges_cover_domain_with_overlap() -> None:
    assert _dd_core_patch_ranges(n=10, block=4, overlap=1) == [
        (0, 4, 0, 5),
        (4, 8, 3, 9),
        (8, 10, 7, 10),
    ]


def test_dd_core_patch_ranges_clamp_to_domain_edges() -> None:
    assert _dd_core_patch_ranges(n=5, block=2, overlap=10) == [
        (0, 2, 0, 5),
        (2, 4, 0, 5),
        (4, 5, 0, 5),
    ]


def test_rhs1_dd_auto_block_size_spans_more_than_one_local_shard() -> None:
    block = _rhs1_dd_auto_block_size(n=31, n_dev=8, sum_nxi=144, dof_target=1200)

    assert block == 12
    assert block > 4


def test_rhs1_dd_auto_block_size_respects_global_extent() -> None:
    block = _rhs1_dd_auto_block_size(n=31, n_dev=2, sum_nxi=144, dof_target=1200)

    assert block == 24
    assert block <= 31


def test_rhs1_dd_coarse_block_size_widens_local_patch() -> None:
    coarse = _rhs1_dd_coarse_block_size(n=31, block=12, overlap=1)

    assert coarse == 20
    assert coarse > 12


def test_rhs1_dd_coarse_level_count_auto_and_env(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", raising=False)
    assert _rhs1_dd_coarse_level_count(n_dev=2) == 0
    assert _rhs1_dd_coarse_level_count(n_dev=4) == 1
    assert _rhs1_dd_coarse_level_count(n_dev=8) == 2

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "3")
    assert _rhs1_dd_coarse_level_count(n_dev=2) == 3

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE_LEVELS", "bad")
    assert _rhs1_dd_coarse_level_count(n_dev=8) == 1


def test_rhs1_dd_coarse_block_sizes_build_and_stop_at_global_extent() -> None:
    assert _rhs1_dd_coarse_block_sizes(n=63, block=12, overlap=1, levels=2) == (20, 30)
    assert _rhs1_dd_coarse_block_sizes(n=25, block=12, overlap=4, levels=3) == (20, 25)


def test_axis_line_index_maps_match_v3_layout_and_reject_unknown_axis() -> None:
    op = _toy_op(n_theta=2, n_zeta=3, extra_size=0)

    theta_map, theta_line_size = _axis_line_index_map(op=op, axis="theta")
    zeta_map, zeta_line_size = _axis_line_index_map(op=op, axis="zeta")

    assert theta_line_size == 2
    assert zeta_line_size == 3
    np.testing.assert_array_equal(theta_map[0, 0], np.asarray([0, 3], dtype=np.int32))
    np.testing.assert_array_equal(zeta_map[0, 0], np.asarray([0, 1, 2], dtype=np.int32))
    with np.testing.assert_raises(ValueError):
        _axis_line_index_map(op=op, axis="radial")


def test_regularization_and_extra_inverse_follow_documented_policy(monkeypatch) -> None:
    op = _toy_op(extra_size=1)
    _install_diagonal_toy_operator(monkeypatch, diagonal=2.0)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", raising=False)

    extra_idx, extra_inv = _extra_inverse(
        op_pc=op,
        op=op,
        total_size=op.total_size,
        dtype=jnp.float64,
        reg=_regularization_from_env(),
    )

    np.testing.assert_array_equal(np.asarray(extra_idx), np.asarray([4], dtype=np.int32))
    np.testing.assert_allclose(np.asarray(extra_inv), np.asarray([[0.5]]), rtol=1.0e-10, atol=1.0e-10)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0.25")
    assert float(_regularization_from_env()) == 0.25


def test_theta_and_zeta_line_preconditioners_apply_block_inverse_and_cache(monkeypatch) -> None:
    op = _toy_op()
    _install_diagonal_toy_operator(monkeypatch, diagonal=2.0)

    theta = build_rhs1_theta_line_preconditioner(op=op)
    zeta = build_rhs1_zeta_line_preconditioner(op=op)
    _assert_half_inverse(theta, n=op.total_size)
    _assert_half_inverse(zeta, n=op.total_size)

    cache_size = len(dd._RHSMODE1_PRECOND_CACHE)
    theta_again = build_rhs1_theta_line_preconditioner(op=op)
    _assert_half_inverse(theta_again, n=op.total_size)
    assert len(dd._RHSMODE1_PRECOND_CACHE) == cache_size


def test_line_preconditioner_reduced_apply_uses_expand_reduce_contract(monkeypatch) -> None:
    op = _toy_op()
    _install_diagonal_toy_operator(monkeypatch, diagonal=2.0)
    active = np.asarray([0, 2, 4], dtype=np.int32)

    def expand_reduced(vec):
        full = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return full.at[jnp.asarray(active)].set(vec)

    def reduce_full(vec):
        return vec[jnp.asarray(active)]

    preconditioner = build_rhs1_theta_line_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )

    residual = jnp.asarray([2.0, 4.0, 6.0], dtype=jnp.float64)
    np.testing.assert_allclose(np.asarray(preconditioner(residual)), np.asarray([1.0, 2.0, 3.0]))


def test_schwarz_preconditioners_apply_restricted_additive_inverse(monkeypatch) -> None:
    op = _toy_op()
    _install_diagonal_toy_operator(monkeypatch, diagonal=2.0)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SCHWARZ_COARSE", "0")

    theta = build_rhs1_theta_schwarz_preconditioner(op=op, block=1, overlap=1)
    zeta = build_rhs1_zeta_schwarz_preconditioner(op=op, block=1, overlap=1)

    _assert_half_inverse(theta, n=op.total_size)
    _assert_half_inverse(zeta, n=op.total_size)


def test_theta_line_xdiag_and_theta_zeta_preconditioners_apply_inverse(monkeypatch) -> None:
    op = _toy_op()
    _install_diagonal_toy_operator(monkeypatch, diagonal=2.0)

    xdiag = build_rhs1_theta_line_xdiag_preconditioner(op=op)
    angular = build_rhs1_theta_zeta_preconditioner(op=op)

    _assert_half_inverse(xdiag, n=op.total_size)
    _assert_half_inverse(angular, n=op.total_size)


def test_domain_preconditioners_fail_closed_on_singular_local_blocks(monkeypatch) -> None:
    op = _toy_op(extra_size=0)
    _install_diagonal_toy_operator(monkeypatch, diagonal=0.0)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_PRECOND_REG", "0")

    preconditioner = build_rhs1_theta_zeta_preconditioner(op=op)
    residual = jnp.arange(1, op.total_size + 1, dtype=jnp.float64)
    actual = np.asarray(preconditioner(residual), dtype=np.float64)

    np.testing.assert_allclose(actual, np.zeros((op.total_size,)), rtol=0.0, atol=0.0)
    assert np.all(np.isfinite(actual))
