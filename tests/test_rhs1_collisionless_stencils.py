from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from sfincs_jax.collisionless import CollisionlessV3Operator, apply_collisionless_v3
from sfincs_jax.collisionless_er import apply_er_xdot_v3, apply_er_xidot_v3
from sfincs_jax.collisionless_exb import ExBThetaV3Operator, ExBZetaV3Operator, apply_exb_theta_v3, apply_exb_zeta_v3
from sfincs_jax.collisions import apply_pitch_angle_scattering_v3, make_pitch_angle_scattering_v3_operator
from sfincs_jax.magnetic_drifts import apply_magnetic_drift_theta_v3, apply_magnetic_drift_xidot_v3, apply_magnetic_drift_zeta_v3
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.rhs1_block_operator import RHS1BlockLayout
from sfincs_jax.rhs1_collisionless_stencils import (
    build_collisionless_f_block_operator,
    build_er_xdot_f_block_operator,
    build_er_xidot_f_block_operator,
    build_exb_theta_f_block_operator,
    build_exb_zeta_f_block_operator,
    build_magnetic_drift_theta_f_block_operator,
    build_magnetic_drift_xidot_f_block_operator,
    build_magnetic_drift_zeta_f_block_operator,
)
from sfincs_jax.rhs1_collision_stencils import build_pas_collision_f_block_operator
from sfincs_jax.v3_fblock import fblock_operator_from_namelist


def _layout(
    *,
    n_species: int = 2,
    n_x: int = 2,
    n_xi: int = 3,
    n_theta: int = 3,
    n_zeta: int = 2,
) -> RHS1BlockLayout:
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


def _layout_from_fblock_operator(op) -> RHS1BlockLayout:
    return RHS1BlockLayout.from_operator(
        SimpleNamespace(
            n_species=op.n_species,
            n_x=op.n_x,
            n_xi=op.n_xi,
            n_theta=op.n_theta,
            n_zeta=op.n_zeta,
            f_size=op.flat_size,
            phi1_size=0,
            extra_size=0,
            total_size=op.flat_size,
            constraint_scheme=1,
            include_phi1=False,
            include_phi1_in_kinetic=False,
            rhs_mode=1,
        )
    )


def _exb_theta_operator(*, use_dkes_exb_drift: bool = False) -> ExBThetaV3Operator:
    ddtheta = jnp.asarray(
        [
            [0.0, 1.5, -0.5],
            [-1.0, 0.0, 0.75],
            [0.25, -1.25, 0.0],
        ],
        dtype=jnp.float64,
    )
    d_hat = jnp.asarray(
        [
            [1.0, 1.1],
            [0.9, 1.2],
            [1.3, 0.8],
        ],
        dtype=jnp.float64,
    )
    b_hat = jnp.asarray(
        [
            [2.0, 2.1],
            [1.8, 2.2],
            [2.3, 1.9],
        ],
        dtype=jnp.float64,
    )
    b_hat_sub_zeta = jnp.asarray(
        [
            [0.4, -0.2],
            [0.3, 0.5],
            [-0.1, 0.25],
        ],
        dtype=jnp.float64,
    )
    return ExBThetaV3Operator(
        alpha=jnp.asarray(1.7, dtype=jnp.float64),
        delta=jnp.asarray(0.03, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(-2.5, dtype=jnp.float64),
        ddtheta=ddtheta,
        d_hat=d_hat,
        b_hat=b_hat,
        b_hat_sub_zeta=b_hat_sub_zeta,
        use_dkes_exb_drift=bool(use_dkes_exb_drift),
        fsab_hat2=jnp.asarray(3.4, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
    )


def _collisionless_operator() -> CollisionlessV3Operator:
    ddtheta = jnp.asarray(
        [
            [0.0, 1.5, -0.5],
            [-1.0, 0.0, 0.75],
            [0.25, -1.25, 0.0],
        ],
        dtype=jnp.float64,
    )
    ddzeta = jnp.asarray(
        [
            [0.0, 1.25],
            [-0.75, 0.0],
        ],
        dtype=jnp.float64,
    )
    b_hat = jnp.asarray(
        [
            [2.0, 2.1],
            [1.8, 2.2],
            [2.3, 1.9],
        ],
        dtype=jnp.float64,
    )
    return CollisionlessV3Operator(
        x=jnp.asarray([0.35, 0.9], dtype=jnp.float64),
        ddtheta=ddtheta,
        ddzeta=ddzeta,
        b_hat=b_hat,
        b_hat_sup_theta=jnp.asarray(
            [
                [0.7, -0.2],
                [0.4, 0.5],
                [-0.3, 0.6],
            ],
            dtype=jnp.float64,
        ),
        b_hat_sup_zeta=jnp.asarray(
            [
                [-0.5, 0.3],
                [0.2, -0.4],
                [0.55, 0.1],
            ],
            dtype=jnp.float64,
        ),
        db_hat_dtheta=jnp.asarray(
            [
                [0.15, -0.08],
                [0.04, 0.11],
                [-0.07, 0.05],
            ],
            dtype=jnp.float64,
        ),
        db_hat_dzeta=jnp.asarray(
            [
                [-0.03, 0.09],
                [0.12, -0.06],
                [0.02, 0.07],
            ],
            dtype=jnp.float64,
        ),
        t_hats=jnp.asarray([1.0, 1.5], dtype=jnp.float64),
        m_hats=jnp.asarray([1.0, 4.0], dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
    )


def _exb_zeta_operator(*, use_dkes_exb_drift: bool = False) -> ExBZetaV3Operator:
    ddzeta = jnp.asarray(
        [
            [0.0, 1.25],
            [-0.75, 0.0],
        ],
        dtype=jnp.float64,
    )
    d_hat = jnp.asarray(
        [
            [1.0, 1.1],
            [0.9, 1.2],
            [1.3, 0.8],
        ],
        dtype=jnp.float64,
    )
    b_hat = jnp.asarray(
        [
            [2.0, 2.1],
            [1.8, 2.2],
            [2.3, 1.9],
        ],
        dtype=jnp.float64,
    )
    b_hat_sub_theta = jnp.asarray(
        [
            [-0.35, 0.15],
            [0.45, -0.25],
            [0.2, 0.55],
        ],
        dtype=jnp.float64,
    )
    return ExBZetaV3Operator(
        alpha=jnp.asarray(1.7, dtype=jnp.float64),
        delta=jnp.asarray(0.03, dtype=jnp.float64),
        dphi_hat_dpsi_hat=jnp.asarray(-2.5, dtype=jnp.float64),
        ddzeta=ddzeta,
        d_hat=d_hat,
        b_hat=b_hat,
        b_hat_sub_theta=b_hat_sub_theta,
        use_dkes_exb_drift=bool(use_dkes_exb_drift),
        fsab_hat2=jnp.asarray(3.4, dtype=jnp.float64),
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
    )


def _pas_operator():
    return make_pitch_angle_scattering_v3_operator(
        x=jnp.asarray([0.35, 0.9], dtype=jnp.float64),
        z_s=jnp.asarray([1.0, 2.0], dtype=jnp.float64),
        m_hats=jnp.asarray([1.0, 4.0], dtype=jnp.float64),
        n_hats=jnp.asarray([1.0, 0.25], dtype=jnp.float64),
        t_hats=jnp.asarray([1.0, 1.5], dtype=jnp.float64),
        nu_n=0.7,
        krook=0.0,
        n_xi_for_x=jnp.asarray([3, 2], dtype=jnp.int32),
        n_xi=3,
    )


def test_collisionless_f_block_operator_matches_existing_apply() -> None:
    layout = _layout()
    collisionless = _collisionless_operator()
    block_op = build_collisionless_f_block_operator(layout=layout, collisionless_operator=collisionless)
    rng = np.random.default_rng(2026060306)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_collisionless_v3(collisionless, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)
    assert block_op.shape == (layout.f_size, layout.f_size)
    assert block_op.block_size == layout.n_zeta


@pytest.mark.parametrize("use_dkes_exb_drift", (False, True))
def test_exb_theta_f_block_operator_matches_existing_apply(use_dkes_exb_drift: bool) -> None:
    layout = _layout()
    exb = _exb_theta_operator(use_dkes_exb_drift=use_dkes_exb_drift)
    block_op = build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=exb)
    rng = np.random.default_rng(2026060301 + int(use_dkes_exb_drift))
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_exb_theta_v3(exb, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)
    assert block_op.shape == (layout.f_size, layout.f_size)
    assert block_op.block_size == layout.n_zeta


@pytest.mark.parametrize("use_dkes_exb_drift", (False, True))
def test_exb_zeta_f_block_operator_matches_existing_apply(use_dkes_exb_drift: bool) -> None:
    layout = _layout()
    exb = _exb_zeta_operator(use_dkes_exb_drift=use_dkes_exb_drift)
    block_op = build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=exb)
    rng = np.random.default_rng(2026060304 + int(use_dkes_exb_drift))
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_exb_zeta_v3(exb, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)
    assert block_op.shape == (layout.f_size, layout.f_size)
    assert block_op.block_size == layout.n_zeta


def test_pas_collision_plus_exb_theta_zeta_block_sum_matches_existing_apply_sum() -> None:
    layout = _layout()
    pas = _pas_operator()
    exb_theta = _exb_theta_operator()
    exb_zeta = _exb_zeta_operator()
    pas_block = build_pas_collision_f_block_operator(layout=layout, pas_operator=pas)
    collisionless = _collisionless_operator()
    collisionless_block = build_collisionless_f_block_operator(layout=layout, collisionless_operator=collisionless)
    exb_theta_block = build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=exb_theta)
    exb_zeta_block = build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=exb_zeta)
    rng = np.random.default_rng(2026060303)
    f = rng.normal(size=layout.f_shape).astype(np.float64)
    flat = jnp.asarray(f.reshape((-1,)))

    expected = np.asarray(
        apply_pitch_angle_scattering_v3(pas, jnp.asarray(f))
        + apply_collisionless_v3(collisionless, jnp.asarray(f))
        + apply_exb_theta_v3(exb_theta, jnp.asarray(f))
        + apply_exb_zeta_v3(exb_zeta, jnp.asarray(f))
    ).reshape((-1,))
    got = np.asarray(
        pas_block.matvec(flat)
        + collisionless_block.matvec(flat)
        + exb_theta_block.matvec(flat)
        + exb_zeta_block.matvec(flat)
    )

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_3d_pas_er_fblock_terms_assemble_to_existing_apply_sum(tmp_path: Path) -> None:
    source = Path(__file__).parent / "ref" / "pas_1species_PAS_noEr_tiny_scheme12.input.namelist"
    input_path = tmp_path / "pas_er_scheme12.input.namelist"
    input_path.write_text(source.read_text().replace("Er = 0.0d+0", "Er = 0.5d+0"))

    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.pas is not None
    assert op.exb_theta is not None
    assert op.exb_zeta is not None

    layout = _layout_from_fblock_operator(op)
    pas_block = build_pas_collision_f_block_operator(layout=layout, pas_operator=op.pas)
    exb_theta_block = build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=op.exb_theta)
    exb_zeta_block = build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=op.exb_zeta)
    collisionless_block = build_collisionless_f_block_operator(layout=layout, collisionless_operator=op.collisionless)

    rng = np.random.default_rng(2026060305)
    f = rng.normal(size=layout.f_shape).astype(np.float64)
    flat = jnp.asarray(f.reshape((-1,)))
    pas_expected = apply_pitch_angle_scattering_v3(op.pas, jnp.asarray(f))
    collisionless_expected = apply_collisionless_v3(op.collisionless, jnp.asarray(f))
    exb_theta_expected = apply_exb_theta_v3(op.exb_theta, jnp.asarray(f))
    exb_zeta_expected = apply_exb_zeta_v3(op.exb_zeta, jnp.asarray(f))

    for term in (pas_expected, collisionless_expected, exb_theta_expected, exb_zeta_expected):
        assert np.linalg.norm(np.asarray(term)) > 0.0

    expected = np.asarray(pas_expected + collisionless_expected + exb_theta_expected + exb_zeta_expected).reshape((-1,))
    got = np.asarray(
        pas_block.matvec(flat)
        + collisionless_block.matvec(flat)
        + exb_theta_block.matvec(flat)
        + exb_zeta_block.matvec(flat)
    )

    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_er_xidot_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "er_xidot_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.er_xidot is not None
    layout = _layout_from_fblock_operator(op)
    block_op = build_er_xidot_f_block_operator(layout=layout, er_xidot_operator=op.er_xidot)

    rng = np.random.default_rng(2026060309)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_er_xidot_v3(op.er_xidot, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_er_xdot_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "er_xdot_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.er_xdot is not None
    layout = _layout_from_fblock_operator(op)
    block_op = build_er_xdot_f_block_operator(layout=layout, er_xdot_operator=op.er_xdot)

    rng = np.random.default_rng(2026060310)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_er_xdot_v3(op.er_xdot, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_magnetic_drift_xidot_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "magdrift_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.magdrift_xidot is not None
    layout = _layout_from_fblock_operator(op)
    block_op = build_magnetic_drift_xidot_f_block_operator(layout=layout, magdrift_xidot_operator=op.magdrift_xidot)

    rng = np.random.default_rng(2026060312)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_magnetic_drift_xidot_v3(op.magdrift_xidot, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_magnetic_drift_theta_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "magdrift_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.magdrift_theta is not None
    layout = _layout_from_fblock_operator(op)
    block_op = build_magnetic_drift_theta_f_block_operator(layout=layout, magdrift_theta_operator=op.magdrift_theta)

    rng = np.random.default_rng(2026060314)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_magnetic_drift_theta_v3(op.magdrift_theta, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_parsed_magnetic_drift_zeta_f_block_operator_matches_existing_apply() -> None:
    input_path = Path(__file__).parent / "ref" / "magdrift_1species_tiny.input.namelist"
    op = fblock_operator_from_namelist(nml=read_sfincs_input(input_path), identity_shift=0.0)
    assert op.magdrift_zeta is not None
    layout = _layout_from_fblock_operator(op)
    block_op = build_magnetic_drift_zeta_f_block_operator(layout=layout, magdrift_zeta_operator=op.magdrift_zeta)

    rng = np.random.default_rng(2026060315)
    f = rng.normal(size=layout.f_shape).astype(np.float64)

    expected = np.asarray(apply_magnetic_drift_zeta_v3(op.magdrift_zeta, jnp.asarray(f))).reshape((-1,))
    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,)))))

    assert np.linalg.norm(expected) > 0.0
    np.testing.assert_allclose(got, expected, rtol=1.0e-13, atol=1.0e-13)


def test_collisionless_f_block_operator_keeps_inactive_legendre_slots_zero() -> None:
    layout = _layout()
    collisionless = _collisionless_operator()
    block_op = build_collisionless_f_block_operator(layout=layout, collisionless_operator=collisionless)
    f = np.ones(layout.f_shape, dtype=np.float64)

    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    np.testing.assert_allclose(got[:, 1, 2:, :, :], 0.0, rtol=0.0, atol=0.0)


def test_exb_theta_f_block_operator_keeps_inactive_legendre_slots_zero() -> None:
    layout = _layout()
    exb = _exb_theta_operator()
    block_op = build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=exb)
    f = np.ones(layout.f_shape, dtype=np.float64)

    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    np.testing.assert_allclose(got[:, 1, 2:, :, :], 0.0, rtol=0.0, atol=0.0)


def test_exb_zeta_f_block_operator_keeps_inactive_legendre_slots_zero() -> None:
    layout = _layout()
    exb = _exb_zeta_operator()
    block_op = build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=exb)
    f = np.ones(layout.f_shape, dtype=np.float64)

    got = np.asarray(block_op.matvec(jnp.asarray(f.reshape((-1,))))).reshape(layout.f_shape)

    np.testing.assert_allclose(got[:, 1, 2:, :, :], 0.0, rtol=0.0, atol=0.0)


def test_collisionless_f_block_operator_validates_inputs() -> None:
    layout = _layout()
    collisionless = _collisionless_operator()

    bad_layout = RHS1BlockLayout(
        **{
            **layout.to_dict(),
            "f_size": layout.f_size + 1,
        }
    )
    with pytest.raises(ValueError, match="layout f_size"):
        build_collisionless_f_block_operator(layout=bad_layout, collisionless_operator=collisionless)

    bad_ddtheta = SimpleNamespace(
        **{
            **collisionless.__dict__,
            "ddtheta": jnp.zeros((2, 2), dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="ddtheta"):
        build_collisionless_f_block_operator(layout=layout, collisionless_operator=bad_ddtheta)

    bad_nxi = SimpleNamespace(
        **{
            **collisionless.__dict__,
            "n_xi_for_x": jnp.asarray([3, 4], dtype=jnp.int32),
        }
    )
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_collisionless_f_block_operator(layout=layout, collisionless_operator=bad_nxi)

    bad_b = SimpleNamespace(
        **{
            **collisionless.__dict__,
            "b_hat": jnp.zeros_like(collisionless.b_hat),
        }
    )
    with pytest.raises(ValueError, match="b_hat"):
        build_collisionless_f_block_operator(layout=layout, collisionless_operator=bad_b)

    bad_species = SimpleNamespace(
        **{
            **collisionless.__dict__,
            "m_hats": jnp.asarray([1.0], dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="m_hats"):
        build_collisionless_f_block_operator(layout=layout, collisionless_operator=bad_species)


def test_exb_theta_f_block_operator_validates_inputs() -> None:
    layout = _layout()
    exb = _exb_theta_operator()

    bad_layout = RHS1BlockLayout(
        **{
            **layout.to_dict(),
            "f_size": layout.f_size + 1,
        }
    )
    with pytest.raises(ValueError, match="layout f_size"):
        build_exb_theta_f_block_operator(layout=bad_layout, exb_theta_operator=exb)

    bad_ddtheta = SimpleNamespace(
        **{
            **exb.__dict__,
            "ddtheta": jnp.zeros((2, 2), dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="ddtheta"):
        build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=bad_ddtheta)

    bad_nxi = SimpleNamespace(
        **{
            **exb.__dict__,
            "n_xi_for_x": jnp.asarray([3, 4], dtype=jnp.int32),
        }
    )
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=bad_nxi)

    bad_b = SimpleNamespace(
        **{
            **exb.__dict__,
            "b_hat": jnp.zeros_like(exb.b_hat),
        }
    )
    with pytest.raises(ValueError, match="b_hat"):
        build_exb_theta_f_block_operator(layout=layout, exb_theta_operator=bad_b)


def test_exb_zeta_f_block_operator_validates_inputs() -> None:
    layout = _layout()
    exb = _exb_zeta_operator()

    bad_layout = RHS1BlockLayout(
        **{
            **layout.to_dict(),
            "f_size": layout.f_size + 1,
        }
    )
    with pytest.raises(ValueError, match="layout f_size"):
        build_exb_zeta_f_block_operator(layout=bad_layout, exb_zeta_operator=exb)

    bad_ddzeta = SimpleNamespace(
        **{
            **exb.__dict__,
            "ddzeta": jnp.zeros((3, 3), dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="ddzeta"):
        build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=bad_ddzeta)

    bad_nxi = SimpleNamespace(
        **{
            **exb.__dict__,
            "n_xi_for_x": jnp.asarray([3, 4], dtype=jnp.int32),
        }
    )
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=bad_nxi)

    bad_b = SimpleNamespace(
        **{
            **exb.__dict__,
            "b_hat": jnp.zeros_like(exb.b_hat),
        }
    )
    with pytest.raises(ValueError, match="b_hat"):
        build_exb_zeta_f_block_operator(layout=layout, exb_zeta_operator=bad_b)


def test_er_f_block_operators_validate_inputs() -> None:
    xidot_input = Path(__file__).parent / "ref" / "er_xidot_1species_tiny.input.namelist"
    xdot_input = Path(__file__).parent / "ref" / "er_xdot_1species_tiny.input.namelist"
    xidot = fblock_operator_from_namelist(nml=read_sfincs_input(xidot_input), identity_shift=0.0).er_xidot
    xdot = fblock_operator_from_namelist(nml=read_sfincs_input(xdot_input), identity_shift=0.0).er_xdot
    assert xidot is not None
    assert xdot is not None
    layout = _layout_from_fblock_operator(fblock_operator_from_namelist(nml=read_sfincs_input(xidot_input), identity_shift=0.0))

    bad_nxi = SimpleNamespace(
        **{
            **xidot.__dict__,
            "n_xi_for_x": jnp.asarray([layout.n_xi + 1] * layout.n_x, dtype=jnp.int32),
        }
    )
    with pytest.raises(ValueError, match="n_xi_for_x"):
        build_er_xidot_f_block_operator(layout=layout, er_xidot_operator=bad_nxi)

    bad_b = SimpleNamespace(
        **{
            **xidot.__dict__,
            "b_hat": jnp.zeros_like(xidot.b_hat),
        }
    )
    with pytest.raises(ValueError, match="b_hat"):
        build_er_xidot_f_block_operator(layout=layout, er_xidot_operator=bad_b)

    bad_ddx = SimpleNamespace(
        **{
            **xdot.__dict__,
            "ddx_plus": jnp.zeros((layout.n_x - 1, layout.n_x - 1), dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="ddx_plus"):
        build_er_xdot_f_block_operator(layout=layout, er_xdot_operator=bad_ddx)

    mag_input = Path(__file__).parent / "ref" / "magdrift_1species_tiny.input.namelist"
    mag_xidot = fblock_operator_from_namelist(nml=read_sfincs_input(mag_input), identity_shift=0.0).magdrift_xidot
    assert mag_xidot is not None
    mag_layout = _layout_from_fblock_operator(fblock_operator_from_namelist(nml=read_sfincs_input(mag_input), identity_shift=0.0))
    bad_mag_x = SimpleNamespace(
        **{
            **mag_xidot.__dict__,
            "x": jnp.zeros((mag_layout.n_x - 1,), dtype=jnp.float64),
        }
    )
    with pytest.raises(ValueError, match="x"):
        build_magnetic_drift_xidot_f_block_operator(layout=mag_layout, magdrift_xidot_operator=bad_mag_x)
