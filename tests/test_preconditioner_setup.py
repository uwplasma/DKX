from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from sfincs_jax.solvers.preconditioning import (
    hash_array,
    matvec_submatrix,
    precond_chunk_cols,
    rhs_mode1_precond_cache_key,
    rhs_mode1_structured_fblock_cache_key,
    transport_precond_cache_key,
)


def _cache_key_operator(*, pas: bool = True, fp: bool = True, phi1: bool = True):
    collisionless = SimpleNamespace(n_xi_for_x=np.asarray([2, 3, 1], dtype=np.int32))
    pas_obj = (
        SimpleNamespace(
            nu_n=0.25,
            krook=0.125,
            nu_d_hat=np.asarray([[1.0, 1.5, 2.0], [2.5, 3.0, 3.5]]),
        )
        if pas
        else None
    )
    fp_obj = (
        SimpleNamespace(mat=np.arange(2 * 2 * 3 * 3 * 3, dtype=np.float64).reshape(2, 2, 3, 3, 3))
        if fp
        else None
    )
    fblock = SimpleNamespace(
        collisionless=collisionless,
        pas=pas_obj,
        fp=fp_obj,
        identity_shift=0.75,
    )
    return SimpleNamespace(
        rhs_mode=1,
        n_species=2,
        n_x=3,
        n_xi=4,
        n_theta=5,
        n_zeta=6,
        constraint_scheme=2,
        quasineutrality_option=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        with_adiabatic=True,
        alpha=0.5,
        delta=0.125,
        dphi_hat_dpsi_hat=-0.25,
        adiabatic_z=np.asarray([1.0, -1.0]),
        adiabatic_nhat=np.asarray([0.1, 0.2]),
        adiabatic_that=np.asarray([1.5, 2.5]),
        z_s=np.asarray([1.0, -1.0]),
        m_hat=np.asarray([2.0, 1.0 / 1836.0]),
        t_hat=np.asarray([1.2, 1.8]),
        n_hat=np.asarray([0.8, 0.9]),
        theta_weights=np.linspace(0.1, 0.5, 5),
        zeta_weights=np.linspace(0.2, 0.7, 6),
        b_hat=np.arange(30, dtype=np.float64).reshape(5, 6) / 10.0,
        d_hat=np.arange(30, dtype=np.float64).reshape(5, 6) / 20.0,
        b_hat_sub_theta=np.arange(30, dtype=np.float64).reshape(5, 6) / 30.0,
        b_hat_sub_zeta=np.arange(30, dtype=np.float64).reshape(5, 6) / 40.0,
        x=np.asarray([0.1, 0.7, 1.6]),
        x_weights=np.asarray([0.2, 0.3, 0.5]),
        phi1_hat_base=np.arange(30, dtype=np.float64).reshape(5, 6) / 50.0 if phi1 else None,
        fblock=fblock,
    )


def test_precond_chunk_cols_respects_explicit_column_override():
    env = {"SFINCS_JAX_PRECOND_CHUNK": "3", "SFINCS_JAX_PRECOND_MAX_MB": "1e-9"}
    assert precond_chunk_cols(1000, 10, environ=env) == 3
    assert precond_chunk_cols(1000, 2, environ=env) == 2


def test_precond_chunk_cols_uses_memory_budget_and_safe_fallbacks():
    assert precond_chunk_cols(1000, 100, environ={"SFINCS_JAX_PRECOND_MAX_MB": "0.016"}) == 2
    assert precond_chunk_cols(1000, 100, environ={"SFINCS_JAX_PRECOND_MAX_MB": "bad"}) == 100
    assert precond_chunk_cols(0, 7, environ={}) == 7
    assert precond_chunk_cols(1000, 7, environ={"SFINCS_JAX_PRECOND_MAX_MB": "0"}) == 7


def test_hash_array_is_stable_for_array_like_inputs():
    values = np.asarray([1.0, 2.0, 3.5])
    assert hash_array(values) == hash_array(jnp.asarray(values))
    assert hash_array(values) != hash_array(np.asarray([1.0, 2.0, 3.6]))


def test_rhs_mode1_cache_key_tracks_operator_signature():
    op = _cache_key_operator()
    key = rhs_mode1_precond_cache_key(op, "xblock", precond_dtype=np.float32)

    assert key[0] == "xblock"
    assert key[1] == str(np.float32)
    assert key[-1] == (2, 3, 1)

    same_key = rhs_mode1_precond_cache_key(op, "xblock", precond_dtype=np.float32)
    assert same_key == key

    op_with_different_density = _cache_key_operator()
    op_with_different_density.n_hat = np.asarray([0.8, 1.1])
    assert rhs_mode1_precond_cache_key(op_with_different_density, "xblock", precond_dtype=np.float32) != key
    assert rhs_mode1_precond_cache_key(op, "xblock", precond_dtype=np.float64) != key


def test_structured_rhs_mode1_cache_key_adds_phi1_and_params():
    op = _cache_key_operator(phi1=True)
    key = rhs_mode1_structured_fblock_cache_key(
        op,
        "xline",
        precond_dtype="float64",
        params=("damping", 0.5),
    )

    assert key[0] == "structured_fblock_xline"
    assert key[-3] == hash_array(op.phi1_hat_base)
    assert key[-2:] == ("damping", 0.5)

    op_without_phi1 = _cache_key_operator(phi1=False)
    no_phi1_key = rhs_mode1_structured_fblock_cache_key(
        op_without_phi1,
        "xline",
        precond_dtype="float64",
    )
    assert no_phi1_key[-1] is None


def test_transport_cache_key_tracks_pas_and_fp_signatures():
    op = _cache_key_operator(pas=True, fp=True)
    key = transport_precond_cache_key(op, "collision_diag", precond_dtype=jnp.float64)

    assert key[0] == "collision_diag"
    assert key[1] == str(jnp.float64)
    assert key[7] == 0.75
    assert key[8] is True
    assert key[9] == 0.25
    assert key[10] == 0.125
    assert key[12] is True
    assert key[-1] == (2, 3, 1)

    no_pas_no_fp_key = transport_precond_cache_key(
        _cache_key_operator(pas=False, fp=False),
        "collision_diag",
        precond_dtype=jnp.float64,
    )
    assert no_pas_no_fp_key[8:14] == (False, None, None, None, False, None)
    assert no_pas_no_fp_key != key


def test_matvec_submatrix_uses_injected_unsharded_operator_and_chunks():
    calls: list[tuple[bool, bool, tuple[int, ...]]] = []

    def _apply(_op, vector, *, include_jacobian_terms=True, allow_sharding=True):
        calls.append((include_jacobian_terms, allow_sharding, tuple(vector.shape)))
        return 3.0 * vector + jnp.arange(vector.shape[0], dtype=vector.dtype)

    submatrix = matvec_submatrix(
        SimpleNamespace(),
        col_idx=np.asarray([0, 2, 3], dtype=np.int32),
        row_idx=np.asarray([0, 2], dtype=np.int32),
        total_size=4,
        chunk_cols=2,
        apply_operator_fn=_apply,
    )

    np.testing.assert_allclose(
        submatrix,
        np.asarray(
            [
                [3.0, 2.0],
                [0.0, 5.0],
                [0.0, 2.0],
            ]
        ),
    )
    assert calls == [(True, False, (4,)), (True, False, (4,))]
