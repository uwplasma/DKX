from __future__ import annotations

from pathlib import Path

import numpy as np
import jax.numpy as jnp

from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_fblock import (
    collisionless_operator_from_namelist,
    fblock_operator_from_namelist,
    fokker_planck_collision_operator_from_namelist,
    fokker_planck_collision_operator_with_phi1_from_namelist,
    matvec_v3_fblock_flat,
    pas_collision_operator_from_namelist,
    solve_v3_fblock_gmres,
)
from sfincs_jax.operators.profile_kinetic import rhs1_fblock_layout_from_operator


def test_v3_fblock_matvec_and_gmres_smoke() -> None:
    """Smoke-test a matrix-free solve for the (partial) v3 F-block operator."""
    input_path = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny.input.namelist"
    nml = read_sfincs_input(input_path)

    op = fblock_operator_from_namelist(nml=nml, identity_shift=1.0)
    rng = np.random.default_rng(0)

    x_true = jnp.asarray(rng.normal(size=(op.flat_size,)).astype(np.float64))
    b = matvec_v3_fblock_flat(op, x_true)

    result = solve_v3_fblock_gmres(op=op, b_flat=b, tol=1e-12, restart=40, maxiter=60)
    x = np.asarray(result.x)

    np.testing.assert_allclose(x, np.asarray(x_true), rtol=1e-7, atol=1e-7)
    assert float(result.residual_norm) < 1e-7


def test_profile_fblock_from_namelist_builders_and_layout_adapter() -> None:
    pas_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_PASCollisions_noEr.input.namelist"
    fp_path = Path(__file__).parent / "reduced_inputs" / "tokamak_1species_FPCollisions_noEr.input.namelist"
    phi1_path = (
        Path(__file__).parent
        / "reduced_inputs"
        / "tokamak_1species_FPCollisions_noEr_withPhi1InDKE.input.namelist"
    )
    pas_nml = read_sfincs_input(pas_path)
    fp_nml = read_sfincs_input(fp_path)
    phi1_nml = read_sfincs_input(phi1_path)

    pas_grids = grids_from_namelist(pas_nml)
    pas_geom = geometry_from_namelist(nml=pas_nml, grids=pas_grids)
    collisionless = collisionless_operator_from_namelist(nml=pas_nml, grids=pas_grids, geom=pas_geom)
    pas = pas_collision_operator_from_namelist(nml=pas_nml, grids=pas_grids)

    fp_grids = grids_from_namelist(fp_nml)
    fp = fokker_planck_collision_operator_from_namelist(nml=fp_nml, grids=fp_grids)

    phi1_grids = grids_from_namelist(phi1_nml)
    fp_phi1 = fokker_planck_collision_operator_with_phi1_from_namelist(
        nml=phi1_nml,
        grids=phi1_grids,
        alpha=1.0,
    )
    fblock = fblock_operator_from_namelist(nml=fp_nml, identity_shift=0.0)
    layout = rhs1_fblock_layout_from_operator(fblock)

    assert collisionless.n_xi_for_x.shape == pas_grids.n_xi_for_x.shape
    assert pas.n_xi_for_x.shape == pas_grids.n_xi_for_x.shape
    assert pas.coef.shape[-1] == int(pas_grids.n_xi)
    assert fp.mat.shape[2] == int(fp_grids.n_xi)
    assert fp.mask_xi.shape[-1] == int(fp_grids.n_xi)
    assert fp_phi1.nl == int(phi1_grids.n_l)
    assert layout.f_size == fblock.flat_size
    assert layout.total_size == fblock.flat_size
    assert layout.include_phi1 is False
