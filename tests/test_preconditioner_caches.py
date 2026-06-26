import jax.numpy as jnp
import numpy as np
import pytest

import sfincs_jax.solvers.preconditioning as pc
import sfincs_jax.v3_driver as vd


@pytest.mark.parametrize(
    ("driver_name", "module_name"),
    [
        ("_RHSMODE1_PRECOND_CACHE", "_RHSMODE1_PRECOND_CACHE"),
        ("_RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE", "_RHSMODE1_STRUCTURED_FBLOCK_PRECOND_CACHE"),
        ("_RHSMODE1_SPARSE_ILU_CACHE", "_RHSMODE1_SPARSE_ILU_CACHE"),
        ("_RHSMODE1_PAS_TOKAMAK_THETA_CACHE", "_RHSMODE1_PAS_TOKAMAK_THETA_CACHE"),
        ("_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE", "_TRANSPORT_FP_TZFFT_LINE_SCHUR_PRECOND_CACHE"),
        ("_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE", "_TRANSPORT_FP_XBLOCK_TZ_LU_PRECOND_CACHE"),
    ],
)
def test_v3_driver_reexports_preconditioner_cache_registries_by_identity(driver_name, module_name):
    assert getattr(vd, driver_name) is getattr(pc, module_name)


def test_rhsmode1_cache_containers_are_lightweight_data_models():
    identity = jnp.eye(2)
    idx = jnp.arange(2)

    cache = pc._RHSMode1PrecondCache(
        idx_map_jnp=idx,
        flat_idx_jnp=idx,
        block_inv_jnp=identity,
        extra_idx_jnp=idx,
        extra_inv_jnp=None,
    )
    assert cache.block_inv_jnp.shape == (2, 2)
    assert cache.extra_inv_jnp is None

    structured = pc._RHSMode1StructuredFBlockPrecondCache(
        operator=object(),
        metadata={"kind": "structured"},
        factor=None,
        coarse=None,
        base_preconditioner=lambda v: v,
    )
    assert structured.metadata["kind"] == "structured"
    assert structured.base_preconditioner is not None

    sparse_ilu = pc._SparseILUCache(
        a_csr_full=object(),
        a_csr_drop=object(),
        ilu=None,
        a_dense=np.eye(2),
        l_dense=np.eye(2),
        u_dense=np.eye(2),
        l_unit_diag=True,
        perm_r=idx,
        inv_perm_c=idx,
        lower_idx=jnp.zeros((2, 0), dtype=jnp.int32),
        lower_val=jnp.zeros((2, 0)),
        upper_idx=jnp.zeros((2, 0), dtype=jnp.int32),
        upper_val=jnp.zeros((2, 0)),
        upper_diag=jnp.ones(2),
    )
    assert sparse_ilu.l_unit_diag is True
    assert sparse_ilu.upper_diag is not None


def test_transport_and_pas_cache_containers_are_lightweight_data_models():
    one = jnp.ones((1,))
    matrix = jnp.eye(2)

    transport = pc._TransportPrecondCache(inv_diag_f=one)
    assert transport.inv_diag_f.shape == (1,)

    schur = pc._TransportFpTzFftLineSchurPrecondCache(
        basis=matrix,
        action=matrix,
        normal_inv=matrix,
        restrict_basis=None,
        damping=0.5,
        tail0=1,
        n_columns=2,
        restriction_kind="tail",
        basis_labels=("density", "current"),
    )
    assert schur.basis_labels == ("density", "current")
    assert schur.restrict_basis is None

    pas_theta = pc._PasTokamakThetaPrecondCache(
        inv_a01=jnp.zeros((1, 1, 2, 2)),
        g01=jnp.zeros((1, 1, 2, 1)),
        inv_a=jnp.zeros((1, 1, 1, 1, 1)),
        g=jnp.zeros((1, 1, 0, 1, 1)),
        c_stream=jnp.ones((1, 3)),
        c_mirror=jnp.ones((1, 3)),
        m_theta=matrix.reshape((1, 2, 2)),
        mirror_factor=jnp.ones((1, 2)),
        mask_active=jnp.ones((1, 3)),
        n_l_build=3,
    )
    assert pas_theta.n_l_build == 3
    assert pas_theta.tail_factors is None
