from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.physics.collisions import (
    apply_fokker_planck_v3,
    apply_fokker_planck_v3_phi1,
    apply_pitch_angle_scattering_v3,
    make_pitch_angle_scattering_v3_operator,
)
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.operators.profile_collisions import (
    build_fokker_planck_collision_f_block_operator,
    build_fokker_planck_phi1_collision_f_block_operator,
    build_pas_collision_f_block_operator,
)
from sfincs_jax.operators.profile_fblock import fblock_operator_from_namelist


def _layout(*, n_species: int = 1, n_x: int = 2, n_xi: int = 4, n_theta: int = 3, n_zeta: int = 2) -> RHS1BlockLayout:
    f_size = int(n_species * n_x * n_xi * n_theta * n_zeta)
    return RHS1BlockLayout.from_operator(
        SimpleNamespace(
            n_species=n_species,
            n_x=n_x,
            n_xi=n_xi,
            n_theta=n_theta,
            n_zeta=n_zeta,
            f_size=f_size,
            phi1_size=0,
            extra_size=0,
            total_size=f_size,
            constraint_scheme=1,
            include_phi1=False,
            include_phi1_in_kinetic=False,
            rhs_mode=1,
        )
    )


def _pas_operator():
    return make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray([0.35, 0.9], dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        m_hats=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([1.0], dtype=jnp.float64),
        t_hats=jnp.asarray([1.0], dtype=jnp.float64),
        nu_n=0.7,
        krook=0.0,
        n_xi_for_x=jnp.asarray([4, 2], dtype=jnp.int32),
        n_xi=4,
    )


def _fp_operator():
    mat = np.zeros((1, 1, 4, 2, 2), dtype=np.float64)
    mat[0, 0, 0] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mat[0, 0, 1] = np.asarray([[0.5, -1.0], [2.0, 0.25]])
    mat[0, 0, 2] = np.asarray([[2.0, 0.0], [0.0, -1.0]])
    mat[0, 0, 3] = np.asarray([[-0.5, 0.25], [1.5, -2.0]])
    return SimpleNamespace(
        mat=jnp.asarray(mat, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([4, 2], dtype=jnp.int32),
        mask_xi=jnp.asarray([[True, True, True, True], [True, True, False, False]]),
    )


def _fp_phi1_operator():
    return SimpleNamespace(
        nu_n=jnp.asarray(0.7, dtype=jnp.float64),
        krook=jnp.asarray(0.05, dtype=jnp.float64),
        alpha=jnp.asarray(1.3, dtype=jnp.float64),
        z_s=jnp.asarray([1.0], dtype=jnp.float64),
        n_hats=jnp.asarray([2.0], dtype=jnp.float64),
        t_hats=jnp.asarray([3.0], dtype=jnp.float64),
        nl=3,
        k_nu=jnp.asarray([[[0.2, 0.5]]], dtype=jnp.float64),
        k_cd=jnp.asarray([[[[1.0, 0.25], [-0.5, 0.75]]]], dtype=jnp.float64),
        k_ce=jnp.asarray([[[[0.1, -0.3], [0.4, 0.2]]]], dtype=jnp.float64),
        k_rosen=jnp.asarray(
            [
                [
                    [
                        [[0.2, 0.1], [-0.15, 0.05]],
                        [[-0.1, 0.25], [0.35, -0.2]],
                        [[0.05, -0.04], [0.02, 0.03]],
                    ]
                ]
            ],
            dtype=jnp.float64,
        ),
        n_xi_for_x=jnp.asarray([4, 2], dtype=jnp.int32),
    )


def test_pas_collision_f_block_operator_matches_existing_apply() -> None:
    layout = _layout()
    pas = _pas_operator()
    block_op = build_pas_collision_f_block_operator(layout=layout, pas_operator=pas)
    rng = np.random.default_rng(20260603)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_pitch_angle_scattering_v3(pas, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)
    assert block_op.shape == (layout.f_size, layout.f_size)
    assert block_op.block_size == layout.n_zeta
    assert block_op.nnz_blocks == layout.n_species * (4 + 2) * layout.n_theta


def test_pas_collision_f_block_operator_keeps_inactive_legendre_slots_zero() -> None:
    layout = _layout()
    pas = _pas_operator()
    block_op = build_pas_collision_f_block_operator(layout=layout, pas_operator=pas)
    f = np.ones(layout.f_shape, dtype=np.float64)
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    np.testing.assert_allclose(got[:, 1, 2:, :, :], 0.0, rtol=0.0, atol=0.0)


def test_fokker_planck_collision_f_block_operator_matches_existing_apply() -> None:
    layout = _layout()
    fp = _fp_operator()
    block_op = build_fokker_planck_collision_f_block_operator(layout=layout, fp_operator=fp)
    rng = np.random.default_rng(2026060316)
    f = rng.normal(size=layout.f_shape).astype(np.float64)
    # This padded input slot is invalid for x=1,L=2 but should still contribute
    # to active x=0,L=2 rows, matching apply_fokker_planck_v3.
    f[0, 1, 2, :, :] = 100.0

    expected = np.asarray(apply_fokker_planck_v3(fp, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)
    assert block_op.shape == (layout.f_size, layout.f_size)
    assert block_op.block_size == layout.n_zeta


def test_parsed_fokker_planck_collision_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "quick_2species_FPCollisions_noEr.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.fp is not None
    layout = _layout(
        n_species=op.n_species,
        n_x=op.n_x,
        n_xi=op.n_xi,
        n_theta=op.n_theta,
        n_zeta=op.n_zeta,
    )
    block_op = build_fokker_planck_collision_f_block_operator(layout=layout, fp_operator=op.fp)
    rng = np.random.default_rng(2026060317)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_fokker_planck_v3(op.fp, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_fokker_planck_phi1_collision_f_block_operator_matches_existing_apply() -> None:
    layout = _layout()
    fp_phi1 = _fp_phi1_operator()
    phi1_hat_base = np.asarray([[0.0, 0.2], [-0.1, 0.4], [0.3, -0.2]], dtype=np.float64)
    block_op = build_fokker_planck_phi1_collision_f_block_operator(
        layout=layout,
        fp_phi1_operator=fp_phi1,
        phi1_hat_base=phi1_hat_base,
    )
    rng = np.random.default_rng(2026060318)
    f = rng.normal(size=layout.f_shape).astype(np.float64)
    f[0, 1, 2, :, :] = 75.0

    expected = np.asarray(apply_fokker_planck_v3_phi1(fp_phi1, jnp.asarray(f), phi1_hat=jnp.asarray(phi1_hat_base)))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_parsed_fokker_planck_phi1_collision_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.fp_phi1 is not None
    layout = _layout(
        n_species=op.n_species,
        n_x=op.n_x,
        n_xi=op.n_xi,
        n_theta=op.n_theta,
        n_zeta=op.n_zeta,
    )
    theta_phase = np.linspace(0.0, np.pi, op.n_theta, dtype=np.float64)
    zeta_phase = np.linspace(0.0, 2.0 * np.pi, op.n_zeta, endpoint=False, dtype=np.float64)
    phi1_hat_base = 0.05 * np.sin(theta_phase)[:, None] * np.cos(zeta_phase)[None, :]
    block_op = build_fokker_planck_phi1_collision_f_block_operator(
        layout=layout,
        fp_phi1_operator=op.fp_phi1,
        phi1_hat_base=phi1_hat_base,
    )
    rng = np.random.default_rng(2026060319)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_fokker_planck_v3_phi1(op.fp_phi1, jnp.asarray(f), phi1_hat=jnp.asarray(phi1_hat_base)))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=2.0e-13, atol=2.0e-13)


def test_pas_collision_f_block_operator_validates_layout_and_operator_shapes() -> None:
    layout = _layout()
    pas = _pas_operator()

    bad_layout = RHS1BlockLayout(
        **{
            **layout.to_dict(),
            "f_size": layout.f_size + 1,
        }
    )
    with pytest.raises(ValueError, match="layout f_size"):
        build_pas_collision_f_block_operator(layout=bad_layout, pas_operator=pas)

    bad_coef = SimpleNamespace(coef=jnp.zeros((1, 2, 3), dtype=jnp.float64), mask_xi=pas.mask_xi)
    with pytest.raises(ValueError, match="pas_operator.coef"):
        build_pas_collision_f_block_operator(layout=layout, pas_operator=bad_coef)

    bad_mask = SimpleNamespace(coef=pas.coef, mask_xi=jnp.ones((2, 3), dtype=bool))
    with pytest.raises(ValueError, match="pas_operator.mask_xi"):
        build_pas_collision_f_block_operator(layout=layout, pas_operator=bad_mask)


def test_fokker_planck_collision_f_block_operator_validates_layout_and_operator_shapes() -> None:
    layout = _layout()
    fp = _fp_operator()

    bad_layout = RHS1BlockLayout(
        **{
            **layout.to_dict(),
            "f_size": layout.f_size + 1,
        }
    )
    with pytest.raises(ValueError, match="layout f_size"):
        build_fokker_planck_collision_f_block_operator(layout=bad_layout, fp_operator=fp)

    bad_mat = SimpleNamespace(mat=jnp.zeros((1, 1, 3, 2, 2), dtype=jnp.float64), n_xi_for_x=fp.n_xi_for_x)
    with pytest.raises(ValueError, match="fp_operator.mat"):
        build_fokker_planck_collision_f_block_operator(layout=layout, fp_operator=bad_mat)

    bad_nxi = SimpleNamespace(mat=fp.mat, n_xi_for_x=jnp.asarray([4, 5], dtype=jnp.int32))
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_fokker_planck_collision_f_block_operator(layout=layout, fp_operator=bad_nxi)


def test_fokker_planck_phi1_collision_f_block_operator_validates_shapes() -> None:
    layout = _layout()
    fp_phi1 = _fp_phi1_operator()
    phi1_hat_base = np.zeros((layout.n_theta, layout.n_zeta), dtype=np.float64)

    with pytest.raises(ValueError, match="phi1_hat_base"):
        build_fokker_planck_phi1_collision_f_block_operator(
            layout=layout,
            fp_phi1_operator=fp_phi1,
            phi1_hat_base=np.zeros((layout.n_theta + 1, layout.n_zeta), dtype=np.float64),
        )

    bad_k_nu = SimpleNamespace(**{**fp_phi1.__dict__, "k_nu": jnp.zeros((1, 1, 3), dtype=jnp.float64)})
    with pytest.raises(ValueError, match="k_nu"):
        build_fokker_planck_phi1_collision_f_block_operator(
            layout=layout,
            fp_phi1_operator=bad_k_nu,
            phi1_hat_base=phi1_hat_base,
        )

    bad_nxi = SimpleNamespace(**{**fp_phi1.__dict__, "n_xi_for_x": jnp.asarray([4, 5], dtype=jnp.int32)})
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_fokker_planck_phi1_collision_f_block_operator(
            layout=layout,
            fp_phi1_operator=bad_nxi,
            phi1_hat_base=phi1_hat_base,
        )
