from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers import preconditioner_transport_matrix as tm
from sfincs_jax.solvers.preconditioning import (
    _RHSMODE23_PRECOND_CACHE,
    _RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE,
    _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE,
    _TRANSPORT_FP_TZFFT_PRECOND_CACHE,
    _TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE,
    _TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE,
    _TRANSPORT_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE,
    _TRANSPORT_SXBLOCK_PRECOND_CACHE,
    _TRANSPORT_TZFFT_PRECOND_CACHE,
    _TRANSPORT_XMG_PRECOND_CACHE,
)


def _periodic_derivative(n: int, scale: float) -> np.ndarray:
    derivative = np.zeros((n, n), dtype=np.float64)
    if n <= 1:
        return derivative
    for i in range(n):
        derivative[i, (i + 1) % n] = 0.5 * scale
        derivative[i, (i - 1) % n] = -0.5 * scale
    return derivative


def _fp_matrix(*, n_species: int, n_x: int, n_l: int) -> np.ndarray:
    matrix = np.zeros((n_species, n_species, n_l, n_x, n_x), dtype=np.float64)
    for ell in range(n_l):
        for species in range(n_species):
            diagonal = 2.2 + 0.4 * species + 0.3 * ell + 0.08 * np.arange(n_x)
            matrix[species, species, ell] = np.diag(diagonal)
            matrix[species, species, ell] += 0.01 * (np.ones((n_x, n_x)) - np.eye(n_x))
        if n_species > 1:
            cross = 0.015 * (ell + 1.0) * np.eye(n_x)
            matrix[0, 1, ell] = cross
            matrix[1, 0, ell] = cross
    return matrix


def _transport_operator(*, include_fp: bool = True) -> SimpleNamespace:
    n_species = 2
    n_x = 3
    n_l = 3
    n_theta = 2
    n_zeta = 2
    f_shape = (n_species, n_x, n_l, n_theta, n_zeta)
    f_size = int(np.prod(f_shape))
    theta = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    zeta = np.linspace(0.0, 2.0 * np.pi, n_zeta, endpoint=False)
    b_hat = 1.1 + 0.03 * np.cos(theta)[:, None] + 0.02 * np.sin(zeta)[None, :]
    b_sup_theta = 0.25 + 0.01 * np.sin(theta)[:, None] + np.zeros((n_theta, n_zeta))
    b_sup_zeta = 0.15 + 0.02 * np.cos(zeta)[None, :] + np.zeros((n_theta, n_zeta))
    db_dtheta = -0.03 * np.sin(theta)[:, None] + np.zeros((n_theta, n_zeta))
    db_dzeta = 0.02 * np.cos(zeta)[None, :] + np.zeros((n_theta, n_zeta))
    nxi_for_x = np.asarray([3, 2, 1], dtype=np.int32)
    collisionless = SimpleNamespace(
        x=np.asarray([0.25, 0.55, 0.9], dtype=np.float64),
        ddtheta=_periodic_derivative(n_theta, 1.0),
        ddzeta=_periodic_derivative(n_zeta, 0.7),
        n_xi_for_x=nxi_for_x,
        b_hat=b_hat,
        b_hat_sup_theta=b_sup_theta,
        b_hat_sup_zeta=b_sup_zeta,
        db_hat_dtheta=db_dtheta,
        db_hat_dzeta=db_dzeta,
        t_hats=np.asarray([1.3, 0.9], dtype=np.float64),
        m_hats=np.asarray([2.0, 1.0], dtype=np.float64),
    )
    fp = None
    if include_fp:
        fp = SimpleNamespace(mat=jnp.asarray(_fp_matrix(n_species=n_species, n_x=n_x, n_l=n_l)))
    pas = SimpleNamespace(
        nu_n=0.4,
        krook=0.15,
        nu_d_hat=jnp.asarray(
            [[1.0, 1.1, 1.25], [0.9, 1.05, 1.18]],
            dtype=jnp.float64,
        ),
    )
    fblock = SimpleNamespace(
        f_shape=f_shape,
        identity_shift=0.45,
        fp=fp,
        pas=pas,
        collisionless=collisionless,
        exb_theta=None,
        exb_zeta=None,
        er_xdot=None,
        er_xidot=None,
        magdrift_theta=None,
        magdrift_zeta=None,
        magdrift_xidot=None,
    )
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_l,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        extra_size=2,
        total_size=f_size + 2,
        fblock=fblock,
        theta_weights=jnp.ones((n_theta,), dtype=jnp.float64),
        zeta_weights=jnp.ones((n_zeta,), dtype=jnp.float64),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat=jnp.asarray(b_hat, dtype=jnp.float64),
        b_hat_sup_theta=jnp.asarray(b_sup_theta, dtype=jnp.float64),
        b_hat_sup_zeta=jnp.asarray(b_sup_zeta, dtype=jnp.float64),
        db_hat_dtheta=jnp.asarray(db_dtheta, dtype=jnp.float64),
        db_hat_dzeta=jnp.asarray(db_dzeta, dtype=jnp.float64),
        b_hat_sub_theta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        b_hat_sub_zeta=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        fsab_hat2=1.0,
        x=jnp.asarray(collisionless.x, dtype=jnp.float64),
        x_weights=jnp.asarray([0.2, 0.5, 0.3], dtype=jnp.float64),
        t_hat=jnp.asarray([1.3, 0.9], dtype=jnp.float64),
        m_hat=jnp.asarray([2.0, 1.0], dtype=jnp.float64),
        alpha=1.0,
        delta=0.4,
        dphi_hat_dpsi_hat=0.0,
        rhs_mode=2,
        constraint_scheme=1,
        quasineutrality_option=1,
        point_at_x0=False,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        with_adiabatic=False,
        adiabatic_z=jnp.zeros((0,), dtype=jnp.float64),
        adiabatic_nhat=jnp.zeros((0,), dtype=jnp.float64),
        adiabatic_that=jnp.zeros((0,), dtype=jnp.float64),
        z_s=jnp.asarray([1.0, -1.0], dtype=jnp.float64),
        n_hat=jnp.asarray([1.0, 0.7], dtype=jnp.float64),
    )


def _attach_exb_terms(op: SimpleNamespace, *, use_dkes: bool) -> None:
    shape = (int(op.n_theta), int(op.n_zeta))
    ones = jnp.ones(shape, dtype=jnp.float64)
    exb_common = dict(
        use_dkes_exb_drift=bool(use_dkes),
        d_hat=op.d_hat,
        b_hat=op.b_hat,
        b_hat_sub_theta=op.b_hat_sub_theta,
        b_hat_sub_zeta=op.b_hat_sub_zeta,
        fsab_hat2=jnp.asarray(op.fsab_hat2, dtype=jnp.float64),
        alpha=jnp.asarray(1.2, dtype=jnp.float64),
        delta=jnp.asarray(0.35, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(0.8, dtype=jnp.float64),
    )
    op.fblock.exb_theta = SimpleNamespace(**exb_common)
    op.fblock.exb_zeta = SimpleNamespace(**exb_common)
    op.fblock.exb_theta.b_hat_sub_zeta = 0.9 * ones
    op.fblock.exb_zeta.b_hat_sub_theta = 1.1 * ones
    op.alpha = 1.2
    op.delta = 0.35
    op.dphi_hat_dpsi_hat = 0.8


def _clear_transport_caches() -> None:
    _RHSMODE1_FP_XBLOCK_ASSEMBLED_HOST_CACHE.clear()
    _RHSMODE23_PRECOND_CACHE.clear()
    _TRANSPORT_PRECOND_CACHE.clear()
    _TRANSPORT_SXBLOCK_PRECOND_CACHE.clear()
    _TRANSPORT_SXBLOCK_LR_PRECOND_CACHE.clear()
    _TRANSPORT_XMG_PRECOND_CACHE.clear()
    _TRANSPORT_TZFFT_PRECOND_CACHE.clear()
    _TRANSPORT_FP_TZFFT_PRECOND_CACHE.clear()
    _TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE.clear()
    _TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE.clear()
    _TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE.clear()
    _TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE.clear()
    _TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE.clear()
    _TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE.clear()


def _vector(op: SimpleNamespace) -> jnp.ndarray:
    return jnp.sin(0.11 * jnp.arange(op.total_size, dtype=jnp.float64)) + 0.05


def _reduction_pair(op: SimpleNamespace, stride: int = 3):
    active = jnp.arange(op.total_size, dtype=jnp.int32)[1::stride]

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    return active, reduce_full, expand_reduced


def test_transport_collision_diag_matches_fp_pas_formula_and_masks_inactive_l(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_REG", "0")
    _clear_transport_caches()
    op = _transport_operator()

    preconditioner = tm.build_rhsmode23_collision_preconditioner(op=op)
    vector = _vector(op)
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
    assert len(_TRANSPORT_PRECOND_CACHE) == 1


def test_transport_preconditioner_reduced_views_match_full_projection(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_REG", "0.2")
    _clear_transport_caches()
    op = _transport_operator()
    vector = _vector(op)
    _, reduce_full, expand_reduced = _reduction_pair(op)
    reduced_rhs = reduce_full(vector)

    builders = (
        tm.build_rhsmode23_collision_preconditioner,
        tm.build_rhsmode23_sxblock_preconditioner,
        tm.build_rhsmode23_tzfft_preconditioner,
        tm.build_rhsmode23_fp_tzfft_line_preconditioner,
    )
    for builder in builders:
        full = builder(op=op)
        reduced = builder(op=op, reduce_full=reduce_full, expand_reduced=expand_reduced)
        expected = reduce_full(full(expand_reduced(reduced_rhs)))
        np.testing.assert_allclose(np.asarray(reduced(reduced_rhs)), np.asarray(expected), rtol=3e-6, atol=3e-6)


def test_transport_sxblock_exact_low_rank_and_no_fp_fallback(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_REG", "not-a-float")
    _clear_transport_caches()
    exact = tm.build_rhsmode23_sxblock_preconditioner(op=op)
    exact_result = exact(vector)
    assert exact_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(exact_result)))
    assert len(_TRANSPORT_SXBLOCK_PRECOND_CACHE) == 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "1")
    _clear_transport_caches()
    low_rank = tm.build_rhsmode23_sxblock_preconditioner(op=op)
    low_rank_result = low_rank(vector)
    assert low_rank_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(low_rank_result)))
    assert len(_TRANSPORT_SXBLOCK_LR_PRECOND_CACHE) == 1

    no_fp = _transport_operator(include_fp=False)
    collision = tm.build_rhsmode23_collision_preconditioner(op=no_fp)
    fallback = tm.build_rhsmode23_sxblock_preconditioner(op=no_fp)
    no_fp_vector = _vector(no_fp)
    np.testing.assert_allclose(np.asarray(fallback(no_fp_vector)), np.asarray(collision(no_fp_vector)))


def test_transport_singular_collision_blocks_use_pseudoinverse_fallback(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOW_RANK_K", "0")
    _clear_transport_caches()
    op = _transport_operator()
    op.fblock.identity_shift = 0.0
    op.fblock.pas = None
    op.fblock.fp = SimpleNamespace(mat=jnp.zeros_like(op.fblock.fp.mat))
    vector = _vector(op)

    sxblock = tm.build_rhsmode23_sxblock_preconditioner(op=op)
    sxblock_result = np.asarray(sxblock(vector))
    assert np.all(np.isfinite(sxblock_result))
    np.testing.assert_allclose(sxblock_result[op.f_size :], np.asarray(vector[op.f_size :]))

    real_inv = np.linalg.inv

    def raise_for_coarse_blocks(matrix: np.ndarray) -> np.ndarray:
        if np.asarray(matrix).shape == (2, 2):
            raise np.linalg.LinAlgError("synthetic coarse singularity")
        return real_inv(matrix)

    op.fblock.identity_shift = 0.5
    monkeypatch.setattr(tm.np.linalg, "inv", raise_for_coarse_blocks)
    _clear_transport_caches()
    xmg = tm.build_rhsmode23_xmg_preconditioner(op=op)
    xmg_result = np.asarray(xmg(vector))
    assert np.all(np.isfinite(xmg_result))
    np.testing.assert_allclose(xmg_result[op.f_size :], np.asarray(vector[op.f_size :]))


def test_transport_xmg_preconditioner_uses_coarse_grid_and_reduced_projection(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_XMG_STRIDE", "bad")
    _clear_transport_caches()
    op = _transport_operator()
    vector = _vector(op)
    full_preconditioner = tm.build_rhsmode23_xmg_preconditioner(op=op)
    full_result = full_preconditioner(vector)

    assert full_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(full_result)))
    cache = next(iter(_TRANSPORT_XMG_PRECOND_CACHE.values()))
    np.testing.assert_array_equal(np.asarray(cache.coarse_idx), np.asarray([0, 2], dtype=np.int32))

    active = jnp.arange(op.total_size, dtype=jnp.int32)[2::5]

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    reduced_preconditioner = tm.build_rhsmode23_xmg_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    reduced_rhs = jnp.cos(0.17 * jnp.arange(active.size, dtype=jnp.float64))
    expected = reduce_full(full_preconditioner(expand_reduced(reduced_rhs)))
    np.testing.assert_allclose(np.asarray(reduced_preconditioner(reduced_rhs)), np.asarray(expected))


def test_transport_block_preconditioner_assembles_active_local_and_tail_blocks(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_PRECOND_BLOCK_REG", "0")
    _clear_transport_caches()
    op = _transport_operator()
    vector = _vector(op)
    calls: list[tuple[np.ndarray, np.ndarray, int, int]] = []

    def fake_matvec_submatrix(op_pc, *, col_idx, row_idx, total_size, chunk_cols):
        del op_pc
        col = np.asarray(col_idx, dtype=np.int32)
        row = np.asarray(row_idx, dtype=np.int32)
        calls.append((col, row, int(total_size), int(chunk_cols)))
        scale = 3.0 if bool(np.all(col >= op.f_size)) else 2.0
        return scale * np.eye(col.size, dtype=np.float64)

    monkeypatch.setattr(tm, "_build_transport_preconditioner_operator_point", lambda op_arg: op_arg)
    monkeypatch.setattr(tm, "_matvec_submatrix", fake_matvec_submatrix)

    preconditioner = tm.build_rhsmode23_block_preconditioner(op=op)
    result = np.asarray(preconditioner(vector))

    expected_f = np.zeros(op.fblock.f_shape, dtype=np.float64)
    vector_f = np.asarray(vector[: op.f_size].reshape(op.fblock.f_shape))
    nxi_for_x = np.asarray(op.fblock.collisionless.n_xi_for_x, dtype=np.int32)
    for species in range(op.n_species):
        for ix in range(op.n_x):
            for ell in range(int(nxi_for_x[ix])):
                expected_f[species, ix, ell, :, :] = 0.5 * vector_f[species, ix, ell, :, :]
    expected = np.zeros((op.total_size,), dtype=np.float64)
    expected[: op.f_size] = expected_f.reshape((-1,))
    expected[op.f_size :] = np.asarray(vector[op.f_size :]) / 3.0

    np.testing.assert_allclose(result, expected)
    assert len(_RHSMODE23_PRECOND_CACHE) == 1
    assert len(calls) == op.n_species + 1
    for col, row, total_size, chunk_cols in calls:
        np.testing.assert_array_equal(col, row)
        assert total_size == op.total_size
        assert chunk_cols >= 1


def test_transport_tzfft_and_fp_tzfft_preconditioners_are_finite_and_cached(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_REG", "not-a-float")
    _clear_transport_caches()
    tzfft = tm.build_rhsmode23_tzfft_preconditioner(op=op)
    tzfft_result = tzfft(vector)
    assert tzfft_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(tzfft_result)))
    assert len(_TRANSPORT_TZFFT_PRECOND_CACHE) == 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_PINV_RCOND", "not-a-float")
    fp_tzfft = tm.build_rhsmode23_fp_tzfft_preconditioner(op=op)
    fp_tzfft_result = fp_tzfft(vector)
    assert fp_tzfft_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(fp_tzfft_result)))
    assert len(_TRANSPORT_FP_TZFFT_PRECOND_CACHE) == 1


def test_transport_fft_preconditioners_include_exb_and_dkes_branches(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_REG", "0.25")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "float64")

    for use_dkes in (False, True):
        _clear_transport_caches()
        op = _transport_operator()
        _attach_exb_terms(op, use_dkes=use_dkes)
        vector = _vector(op)

        tzfft = tm.build_rhsmode23_tzfft_preconditioner(op=op)
        fp_tzfft = tm.build_rhsmode23_fp_tzfft_preconditioner(op=op)
        fp_line = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)

        for preconditioner in (tzfft, fp_tzfft, fp_line):
            result = preconditioner(vector)
            assert result.shape == vector.shape
            assert bool(jnp.all(jnp.isfinite(result)))


def test_transport_fp_preconditioners_respect_memory_caps_and_fallbacks(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_MAX_MB", "1e-12")
    _clear_transport_caches()
    fp_tzfft = tm.build_rhsmode23_fp_tzfft_preconditioner(op=op)
    np.testing.assert_allclose(
        np.asarray(fp_tzfft(vector)),
        np.asarray(tm.build_rhsmode23_sxblock_preconditioner(op=op)(vector)),
    )
    assert len(_TRANSPORT_FP_TZFFT_PRECOND_CACHE) == 0
    assert len(_TRANSPORT_SXBLOCK_PRECOND_CACHE) == 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_MAX_MB", "1e-12")
    _clear_transport_caches()
    line = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    np.testing.assert_allclose(
        np.asarray(line(vector)),
        np.asarray(tm.build_rhsmode23_sxblock_preconditioner(op=op)(vector)),
    )
    assert len(_TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE) == 0
    assert len(_TRANSPORT_SXBLOCK_PRECOND_CACHE) == 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_MAX_MB", "1e-12")
    _clear_transport_caches()
    local_line = tm.build_rhsmode23_fp_local_geom_line_preconditioner(op=op)
    np.testing.assert_allclose(
        np.asarray(local_line(vector)),
        np.asarray(tm.build_rhsmode23_sxblock_preconditioner(op=op)(vector)),
    )
    assert len(_TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE) == 0
    assert len(_TRANSPORT_SXBLOCK_PRECOND_CACHE) == 1


def test_transport_fp_builders_without_fp_route_to_collisionless_tzfft(monkeypatch) -> None:
    """FP-specific transport preconditioners degrade to the collisionless angular factor."""
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_TZFFT_REG", "0.2")
    op = _transport_operator(include_fp=False)
    vector = _vector(op)

    _clear_transport_caches()
    expected = np.asarray(tm.build_rhsmode23_tzfft_preconditioner(op=op)(vector))
    builders = (
        tm.build_rhsmode23_fp_tzfft_preconditioner,
        tm.build_rhsmode23_fp_tzfft_line_preconditioner,
        tm.build_rhsmode23_fp_tzfft_line_schur_preconditioner,
        tm.build_rhsmode23_fp_local_geom_line_preconditioner,
        tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner,
        tm.build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner,
        tm.build_rhsmode23_fp_structured_fblock_lu_preconditioner,
    )
    for builder in builders:
        _clear_transport_caches()
        result = np.asarray(builder(op=op)(vector))
        np.testing.assert_allclose(result, expected, rtol=3e-6, atol=3e-6)


def test_transport_fp_line_and_local_geometry_factors_are_finite_and_cached(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "float64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_PINV_RCOND", "not-a-float")
    _clear_transport_caches()
    line = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    line_result = line(vector)
    assert line_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(line_result)))
    assert len(_TRANSPORT_FP_TZFFT_LINE_PRECOND_CACHE) == 1

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_DTYPE", "float32")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_LOCAL_GEOM_LINE_PINV_RCOND", "not-a-float")
    _clear_transport_caches()
    local_line = tm.build_rhsmode23_fp_local_geom_line_preconditioner(op=op)
    local_result = local_line(vector)
    assert local_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(local_result)))
    assert len(_TRANSPORT_FP_LOCAL_GEOM_LINE_PRECOND_CACHE) == 1


def test_transport_fp_schur_wrappers_bypass_for_phi1_reduced_views(monkeypatch) -> None:
    """Schur coarse wrappers fall back to their base factors for unsupported Phi1 solves."""
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_DTYPE", "float64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "128")
    op = _transport_operator()
    op.include_phi1 = True
    op.point_at_x0 = True
    vector = _vector(op)
    _, reduce_full, expand_reduced = _reduction_pair(op, stride=4)
    reduced_rhs = reduce_full(vector)

    _clear_transport_caches()
    base_line = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    line_schur = tm.build_rhsmode23_fp_tzfft_line_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    expected_line = reduce_full(base_line(expand_reduced(reduced_rhs)))
    np.testing.assert_allclose(np.asarray(line_schur(reduced_rhs)), np.asarray(expected_line), rtol=3e-6, atol=3e-6)
    assert len(_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE) == 0

    _clear_transport_caches()
    base_xblock = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    xblock_schur = tm.build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    expected_xblock = reduce_full(base_xblock(expand_reduced(reduced_rhs)))
    np.testing.assert_allclose(np.asarray(xblock_schur(reduced_rhs)), np.asarray(expected_xblock), rtol=3e-6, atol=3e-6)
    assert len(_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE) == 0


def test_transport_fp_xblock_tz_lu_factor_is_finite_cached_and_reduced(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_REG", "not-a-float")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR", "unknown")
    _clear_transport_caches()

    full_preconditioner = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    full_result = full_preconditioner(vector)

    assert full_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(full_result)))
    assert len(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE) == 1
    cache = next(iter(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE.values()))
    assert cache.metadata["kind"] == "fp_xblock_tz_lu"
    assert cache.metadata["factor_kind"] == "lu"
    assert cache.metadata["block_failures"] == 0
    assert cache.metadata["n_species"] == op.n_species

    active = jnp.arange(op.total_size, dtype=jnp.int32)[::4]

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    reduced_preconditioner = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    reduced_rhs = reduce_full(vector)
    expected = reduce_full(full_preconditioner(expand_reduced(reduced_rhs)))
    np.testing.assert_allclose(np.asarray(reduced_preconditioner(reduced_rhs)), np.asarray(expected))
    assert len(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE) == 1


def test_transport_fp_xblock_tz_lu_factor_solve_failure_keeps_bounded_identity_fallback(
    monkeypatch,
) -> None:
    """A per-block host-factor failure during apply does not poison the full transport solve."""
    op = _transport_operator()
    vector = _vector(op)

    class _RaisingFactor:
        factor_nbytes_estimate = 16
        factor_nnz_estimate = 4

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            del rhs
            raise RuntimeError("synthetic block solve failure")

    monkeypatch.setattr(tm, "factorize_host_sparse_operator", lambda *args, **kwargs: _RaisingFactor())
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "128")
    _clear_transport_caches()

    preconditioner = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    result = np.asarray(preconditioner(vector))

    np.testing.assert_allclose(result, np.asarray(vector), rtol=0.0, atol=0.0)
    cache = next(iter(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE.values()))
    assert cache.metadata["block_failures"] == 0


def test_transport_fp_xblock_tz_lu_uses_diagonal_and_memory_fallbacks(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    def fail_factorization(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic factorization failure")

    monkeypatch.setattr(tm, "factorize_host_sparse_operator", fail_factorization)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "128")
    _clear_transport_caches()
    diagonal_fallback = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    diagonal_result = diagonal_fallback(vector)
    assert diagonal_result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(diagonal_result)))
    cache = next(iter(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE.values()))
    assert cache.metadata["block_failures"] == op.n_species * op.n_x
    assert cache.metadata["block_diagonal_fallbacks"] == op.n_species * op.n_x

    monkeypatch.setattr(tm, "factorize_host_sparse_operator", lambda *args, **kwargs: None)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_MAX_MB", "1e-12")
    _clear_transport_caches()
    capped = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    expected = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    np.testing.assert_allclose(np.asarray(capped(vector)), np.asarray(expected(vector)))
    assert len(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE) == 0


def test_transport_fp_schur_wrappers_disable_cleanly_and_support_reduced_view(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)
    active = jnp.arange(op.total_size, dtype=jnp.int32)[1::3]

    def reduce_full(candidate: jnp.ndarray) -> jnp.ndarray:
        return candidate[active]

    def expand_reduced(candidate: jnp.ndarray) -> jnp.ndarray:
        expanded = jnp.zeros((op.total_size,), dtype=jnp.float64)
        return expanded.at[active].set(candidate)

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_COLS", "0")
    _clear_transport_caches()
    base = tm.build_rhsmode23_fp_tzfft_line_preconditioner(op=op)
    schur_reduced = tm.build_rhsmode23_fp_tzfft_line_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    rhs_reduced = reduce_full(vector)
    expected = reduce_full(base(expand_reduced(rhs_reduced)))
    np.testing.assert_allclose(np.asarray(schur_reduced(rhs_reduced)), np.asarray(expected))
    assert len(_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE) == 0

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_COLS", "0")
    _clear_transport_caches()
    op.point_at_x0 = True
    base_xblock = tm.build_rhsmode23_fp_xblock_tz_lu_preconditioner(op=op)
    xblock_schur_reduced = tm.build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner(
        op=op,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
    )
    expected_xblock = reduce_full(base_xblock(expand_reduced(rhs_reduced)))
    np.testing.assert_allclose(np.asarray(xblock_schur_reduced(rhs_reduced)), np.asarray(expected_xblock))
    assert len(_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE) == 0


def test_transport_fp_tzfft_line_schur_builds_true_action_coarse_space(monkeypatch) -> None:
    op = _transport_operator()
    vector = _vector(op)

    monkeypatch.setattr(tm, "apply_v3_full_system_operator_cached", lambda _op, x: jnp.asarray(x, dtype=jnp.float64))
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_MAX_COLS", "6")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_RESTRICTION", "tail_galerkin")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_DTYPE", "float32")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_TZFFT_LINE_SCHUR_REG", "bad")
    _clear_transport_caches()

    schur = tm.build_rhsmode23_fp_tzfft_line_schur_preconditioner(op=op)
    result = schur(vector)

    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    cache = next(iter(_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE.values()))
    assert 0 < cache.n_columns <= 6
    assert cache.restriction_kind == "tail_galerkin"
    assert cache.restrict_basis is not None
    assert any(label.startswith("tail_") or "constraint1" in label for label in cache.basis_labels)


def test_transport_fp_xblock_tz_lu_schur_builds_kinetic_error_columns(monkeypatch) -> None:
    op = _transport_operator()
    op.point_at_x0 = True
    vector = _vector(op)

    monkeypatch.setattr(tm, "apply_v3_full_system_operator_cached", lambda _op, x: jnp.asarray(x, dtype=jnp.float64))
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_FACTOR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_MAX_COLS", "8")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RESTRICTION", "tail_galerkin")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_KINETIC_RESIDUAL", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_RHS_RESIDUAL", "0")
    _clear_transport_caches()

    schur = tm.build_rhsmode23_fp_xblock_tz_lu_schur_preconditioner(op=op)
    result = schur(vector)

    assert result.shape == vector.shape
    assert bool(jnp.all(jnp.isfinite(result)))
    cache = next(iter(_TRANSPORT_FP_XBLOCK_TZ_LU_SCHUR_PRECOND_CACHE.values()))
    assert 0 < cache.n_columns <= 8
    assert cache.restriction_kind == "tail_galerkin"
    assert any(label.endswith("_xblock_residual_error") for label in cache.basis_labels)


def test_transport_structured_fblock_lu_uses_factor_metadata_and_memory_fallback(monkeypatch) -> None:
    import scipy.sparse as sp

    op = _transport_operator()
    vector = _vector(op)

    class _Selection:
        selected = True
        matrix = sp.eye(op.f_size, dtype=np.float64, format="csr")

        def to_dict(self) -> dict[str, object]:
            return {"selected": True, "nnz": int(self.matrix.nnz)}

    class _Factor:
        factor_nbytes_estimate = 512
        factor_nnz_estimate = op.f_size
        factor_s = 1.5e-3

        def solve(self, rhs: np.ndarray) -> np.ndarray:
            return 0.5 * np.asarray(rhs, dtype=np.float64)

    monkeypatch.setattr(tm, "select_structured_rhs1_fblock_csr_operator", lambda *args, **kwargs: _Selection())
    monkeypatch.setattr(tm, "factorize_host_sparse_operator", lambda *args, **kwargs: _Factor())
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_REG", "bad")
    _clear_transport_caches()

    preconditioner = tm.build_rhsmode23_fp_structured_fblock_lu_preconditioner(op=op)
    result = np.asarray(preconditioner(vector))
    np.testing.assert_allclose(result[: op.f_size], 0.5 * np.asarray(vector[: op.f_size]))
    np.testing.assert_allclose(result[op.f_size :], np.asarray(vector[op.f_size :]))
    cache = next(iter(_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE.values()))
    assert cache.metadata["selection"] == {"selected": True, "nnz": op.f_size}
    assert cache.metadata["factor_nbytes_estimate"] == 512

    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_FACTOR_MAX_MB", "1e-12")
    _clear_transport_caches()
    fallback = tm.build_rhsmode23_fp_structured_fblock_lu_preconditioner(op=op)
    np.testing.assert_allclose(
        np.asarray(fallback(vector)),
        np.asarray(tm.build_rhsmode23_sxblock_preconditioner(op=op)(vector)),
    )
    assert len(_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE) == 0

    class _RejectedSelection:
        selected = False
        matrix = None

        def to_dict(self) -> dict[str, object]:
            return {"selected": False}

    monkeypatch.setattr(tm, "select_structured_rhs1_fblock_csr_operator", lambda *args, **kwargs: _RejectedSelection())
    _clear_transport_caches()
    rejected = tm.build_rhsmode23_fp_structured_fblock_lu_preconditioner(op=op)
    np.testing.assert_allclose(
        np.asarray(rejected(vector)),
        np.asarray(tm.build_rhsmode23_sxblock_preconditioner(op=op)(vector)),
    )
    assert len(_TRANSPORT_FP_STRUCTURED_FBLOCK_LU_PRECOND_CACHE) == 0
