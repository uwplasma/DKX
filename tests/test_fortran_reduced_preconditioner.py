from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.solvers.explicit_sparse import factorize_host_sparse_operator
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.solvers import preconditioning as po
import sfincs_jax.problems.profile_sparse_direct as sparse_direct
from sfincs_jax.operators.profile_system import full_system_operator_from_namelist
import sfincs_jax.problems.profile_solve as vd


@dataclass(frozen=True)
class _Collisionless:
    ddtheta: jnp.ndarray
    ddzeta: jnp.ndarray


@dataclass(frozen=True)
class _RadialDrift:
    ddx_plus: jnp.ndarray
    ddx_minus: jnp.ndarray
    drop_l2_couplings: bool = False


@dataclass(frozen=True)
class _DropL2Term:
    drop_l2_couplings: bool = False


@dataclass(frozen=True)
class _FP:
    mat: jnp.ndarray


@dataclass(frozen=True)
class _FBlock:
    collisionless: _Collisionless
    er_xdot: _RadialDrift | None
    exb_theta: object
    exb_zeta: object
    magdrift_theta: object
    magdrift_zeta: object
    fp: object | None = None
    pas: object | None = None
    magdrift_xidot: object | None = None
    er_xidot: object | None = None


@dataclass(frozen=True)
class _Op:
    rhs_mode: int
    fblock: _FBlock
    n_species: int = 1
    n_x: int = 3


def _matrix(values: list[list[float]]) -> jnp.ndarray:
    return jnp.asarray(values, dtype=jnp.float64)


def _fake_rhs1_op() -> _Op:
    return _Op(
        rhs_mode=1,
        fblock=_FBlock(
            collisionless=_Collisionless(
                ddtheta=_matrix([[1.0, 2.0, 0.0], [3.0, 4.0, 5.0], [0.0, 6.0, 7.0]]),
                ddzeta=_matrix([[8.0, 0.0, 9.0], [10.0, 11.0, 0.0], [12.0, 13.0, 14.0]]),
            ),
            er_xdot=_RadialDrift(
                ddx_plus=_matrix([[1.0, 20.0, 0.0], [30.0, 4.0, 50.0], [0.0, 60.0, 7.0]]),
                ddx_minus=_matrix([[2.0, -2.0, 0.0], [-3.0, 5.0, -5.0], [0.0, -6.0, 8.0]]),
            ),
            exb_theta=None,
            exb_zeta=None,
            magdrift_theta=None,
            magdrift_zeta=None,
            fp=object(),
        ),
    )


def test_fortran_reduced_operator_diagonalizes_only_radial_x_drift() -> None:
    op = _fake_rhs1_op()

    assert vd._build_rhsmode1_preconditioner_operator_fortran_reduced is (
        po._build_rhsmode1_preconditioner_operator_fortran_reduced
    )
    reduced = po._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    assert reduced is not op
    assert reduced.fblock is not op.fblock
    assert reduced.fblock.er_xdot is not op.fblock.er_xdot
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.er_xdot.ddx_plus),
        np.diag(np.diag(np.asarray(op.fblock.er_xdot.ddx_plus))),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.er_xdot.ddx_minus),
        np.diag(np.diag(np.asarray(op.fblock.er_xdot.ddx_minus))),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddtheta),
        np.asarray(op.fblock.collisionless.ddtheta),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddzeta),
        np.asarray(op.fblock.collisionless.ddzeta),
    )
    assert reduced.fblock.exb_theta is op.fblock.exb_theta
    assert reduced.fblock.exb_zeta is op.fblock.exb_zeta
    assert reduced.fblock.magdrift_theta is op.fblock.magdrift_theta
    assert reduced.fblock.magdrift_zeta is op.fblock.magdrift_zeta
    assert reduced.fblock.fp is op.fblock.fp
    assert reduced.fblock.pas is op.fblock.pas


def test_fortran_reduced_operator_respects_preconditioner_x_gate() -> None:
    op = _fake_rhs1_op()

    reduced = vd._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=0,
        preconditioner_xi=0,
        preconditioner_species=0,
    )

    assert reduced.fblock.er_xdot is op.fblock.er_xdot
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.er_xdot.ddx_plus),
        np.asarray(op.fblock.er_xdot.ddx_plus),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.er_xdot.ddx_minus),
        np.asarray(op.fblock.er_xdot.ddx_minus),
    )
    assert reduced.fblock.er_xdot.drop_l2_couplings is False


def test_fortran_reduced_operator_drops_l2_terms_when_preconditioner_xi_is_enabled() -> None:
    op = _Op(
        rhs_mode=1,
        fblock=_FBlock(
            collisionless=_Collisionless(ddtheta=jnp.eye(2), ddzeta=jnp.eye(2)),
            er_xdot=_RadialDrift(ddx_plus=jnp.eye(3), ddx_minus=jnp.eye(3)),
            exb_theta=object(),
            exb_zeta=object(),
            magdrift_theta=_DropL2Term(),
            magdrift_zeta=_DropL2Term(),
            magdrift_xidot=_DropL2Term(),
            er_xidot=_DropL2Term(),
            fp=None,
        ),
    )

    reduced = vd._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    assert reduced.fblock.magdrift_theta.drop_l2_couplings is True
    assert reduced.fblock.magdrift_zeta.drop_l2_couplings is True
    assert reduced.fblock.magdrift_xidot.drop_l2_couplings is True
    assert reduced.fblock.er_xidot.drop_l2_couplings is True
    assert reduced.fblock.er_xdot.drop_l2_couplings is True

    unreduced_xi = vd._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=0,
        preconditioner_species=1,
    )

    assert unreduced_xi.fblock.magdrift_theta.drop_l2_couplings is False
    assert unreduced_xi.fblock.magdrift_zeta.drop_l2_couplings is False
    assert unreduced_xi.fblock.magdrift_xidot.drop_l2_couplings is False
    assert unreduced_xi.fblock.er_xidot.drop_l2_couplings is False
    assert unreduced_xi.fblock.er_xdot.drop_l2_couplings is False


def test_fortran_reduced_operator_respects_preconditioner_x_min_l_for_fp_tensor() -> None:
    op = _fake_rhs1_op()
    fp_mat = jnp.ones((1, 1, 3, 3, 3), dtype=jnp.float64)
    fp_op = _Op(
        rhs_mode=1,
        fblock=_FBlock(
            collisionless=op.fblock.collisionless,
            er_xdot=None,
            exb_theta=op.fblock.exb_theta,
            exb_zeta=op.fblock.exb_zeta,
            magdrift_theta=op.fblock.magdrift_theta,
            magdrift_zeta=op.fblock.magdrift_zeta,
            fp=_FP(mat=fp_mat),
        ),
    )

    reduced = vd._build_rhsmode1_preconditioner_operator_fortran_reduced(
        fp_op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=1,
    )

    reduced_mat = np.asarray(reduced.fblock.fp.mat)[0, 0]
    np.testing.assert_allclose(reduced_mat[0], np.ones((3, 3)))
    np.testing.assert_allclose(reduced_mat[1], np.eye(3))
    np.testing.assert_allclose(reduced_mat[2], np.eye(3))


def test_fortran_reduced_operator_is_rhs1_only() -> None:
    op = _fake_rhs1_op()
    rhs2_op = _Op(rhs_mode=2, fblock=op.fblock)

    assert vd._build_rhsmode1_preconditioner_operator_fortran_reduced(rhs2_op) is rhs2_op


def test_transport_fortran_reduced_operator_applies_to_rhs2_and_keeps_default_angular() -> None:
    op = _fake_rhs1_op()
    rhs2_op = _Op(rhs_mode=2, fblock=op.fblock)

    reduced = vd._build_transport_preconditioner_operator_fortran_reduced(rhs2_op)

    assert reduced is not rhs2_op
    assert reduced.rhs_mode == 2
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.er_xdot.ddx_plus),
        np.diag(np.diag(np.asarray(rhs2_op.fblock.er_xdot.ddx_plus))),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddtheta),
        np.asarray(rhs2_op.fblock.collisionless.ddtheta),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddzeta),
        np.asarray(rhs2_op.fblock.collisionless.ddzeta),
    )


def test_transport_fortran_reduced_operator_can_drop_angular_couplings() -> None:
    op = _fake_rhs1_op()
    rhs3_op = _Op(rhs_mode=3, fblock=op.fblock)

    reduced = vd._build_transport_preconditioner_operator_fortran_reduced(
        rhs3_op,
        keep_theta_zeta=False,
    )

    assert reduced.rhs_mode == 3
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddtheta),
        np.diag(np.diag(np.asarray(rhs3_op.fblock.collisionless.ddtheta))),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddzeta),
        np.diag(np.diag(np.asarray(rhs3_op.fblock.collisionless.ddzeta))),
    )


def test_host_sparse_builder_accepts_symbolic_frontal_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", raising=False)
    a = np.asarray(
        [
            [4.0, 1.0, 0.5, 0.0],
            [1.0, 3.5, 0.0, 0.25],
            [0.5, 0.0, 3.0, 1.0],
            [0.0, 0.25, 1.0, 2.5],
        ],
        dtype=np.float64,
    )

    def _matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(a) @ x

    _operator, factor = vd._build_host_sparse_direct_factor_from_matvec(
        matvec=_matvec,
        n=4,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="symbolic_frontal_schur_lu",
        default_symbolic_ordering_kind="natural",
        default_symbolic_block_size=2,
        default_symbolic_frontal_max_separator_cols=2,
        default_symbolic_frontal_boundary_width=0,
        default_symbolic_frontal_high_degree_cols=0,
        default_symbolic_frontal_max_superblock_size=2,
        default_symbolic_frontal_max_superblock_blocks=1,
        default_symbolic_frontal_min_cross_separator_fraction=1.0,
        default_symbolic_frontal_regularization_rel=0.0,
    )

    rhs = np.asarray([1.0, -2.0, 0.5, 3.0], dtype=np.float64)
    assert factor.kind == "symbolic_frontal_schur_lu"
    np.testing.assert_allclose(a @ factor.solve(rhs), rhs, rtol=1e-11, atol=1e-11)


def test_host_sparse_builder_env_accepts_symbolic_superblock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "symbolic_superblock_lu")
    a = np.asarray(
        [
            [5.0, 1.0, 0.25, 0.0],
            [1.0, 4.0, 0.0, 0.25],
            [0.25, 0.0, 3.0, 1.0],
            [0.0, 0.25, 1.0, 2.0],
        ],
        dtype=np.float64,
    )

    def _matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(a) @ x

    _operator, factor = vd._build_host_sparse_direct_factor_from_matvec(
        matvec=_matvec,
        n=4,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="jacobi",
        default_symbolic_ordering_kind="natural",
        default_symbolic_block_size=2,
        default_symbolic_superblock_max_size=4,
        default_symbolic_superblock_max_blocks=2,
        default_symbolic_superblock_min_retained_cross_fraction=1.0,
        default_symbolic_superblock_regularization_rel=0.0,
    )

    rhs = np.asarray([2.0, -1.0, 1.5, -0.5], dtype=np.float64)
    assert factor.kind == "symbolic_superblock_lu"
    np.testing.assert_allclose(a @ factor.solve(rhs), rhs, rtol=1e-11, atol=1e-11)


def test_host_sparse_builder_env_accepts_symbolic_nd_frontal(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", "nested_dissection_frontal_schur_lu")
    n = 12
    a = sp.diags(
        [
            -0.35 * np.ones(n - 1),
            5.0 + 0.1 * np.arange(n),
            -0.6 * np.ones(n - 1),
        ],
        offsets=[-1, 0, 1],
        format="csr",
    ).toarray()

    def _matvec(x: jnp.ndarray) -> jnp.ndarray:
        return jnp.asarray(a) @ x

    _operator, factor = vd._build_host_sparse_direct_factor_from_matvec(
        matvec=_matvec,
        n=n,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        default_factor_kind="jacobi",
        default_symbolic_ordering_kind="natural",
        default_symbolic_block_size=3,
        default_symbolic_nd_max_leaf_size=3,
        default_symbolic_nd_max_terminal_factor_size=12,
        default_symbolic_nd_max_depth=4,
        default_symbolic_nd_separator_width=2,
        default_symbolic_nd_max_separator_cols=3,
        default_symbolic_nd_high_degree_cols=0,
        default_symbolic_nd_regularization_rel=0.0,
        default_symbolic_nd_max_setup_s=60.0,
    )

    rhs = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    assert factor.kind == "symbolic_nd_frontal_schur_lu"
    assert factor.factor.node_count > 1
    assert factor.factor.metadata["max_terminal_factor_size"] == 12
    assert factor.factor.metadata["max_setup_s"] == 60.0
    np.testing.assert_allclose(a @ factor.solve(rhs), rhs, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize(
    "input_path",
    (
        Path("tests/reduced_inputs/transportMatrix_geometryScheme2.input.namelist"),
        Path("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist"),
    ),
)
def test_transport_direct_reduced_pmat_matches_matrix_free_active_operator(input_path: Path) -> None:
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op_pc = vd._build_transport_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        keep_theta_zeta=True,
    )
    active = vd._transport_active_dof_indices(op_pc)
    shift = 1.0e-10

    result = vd._try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op_pc,
        active_indices=active,
        factor_dtype=np.dtype(np.float64),
        pc_shift=shift,
        emit=None,
    )

    assert result is not None
    bundle, metadata = result
    assert metadata["direct_pmat"] is True
    assert bundle.matrix is not None
    assert bundle.matrix.shape == (active.size, active.size)
    assert bundle.matrix.nnz > op.total_size

    rng = np.random.default_rng(1234)
    for _ in range(3):
        x_reduced = rng.normal(size=int(active.size))
        x_full = np.zeros((int(op_pc.total_size),), dtype=np.float64)
        x_full[np.asarray(active, dtype=np.int64)] = x_reduced
        y_true_full = vd.apply_v3_full_system_operator_cached(op_pc, jnp.asarray(x_full, dtype=jnp.float64))
        y_true = np.asarray(y_true_full, dtype=np.float64)[np.asarray(active, dtype=np.int64)] + shift * x_reduced
        y_direct = bundle.matvec(x_reduced)
        np.testing.assert_allclose(y_direct, y_true, rtol=2e-11, atol=2e-10)


def test_transport_fortran_reduced_lu_attaches_symbolic_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_block_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "0")

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    symbolic = metadata.get("symbolic")

    assert metadata["factor_kind"] == "symbolic_block_lu"
    assert metadata["factor_nbytes_estimate"] > 0
    assert metadata["direct_pmat_enabled"] is True
    assert metadata["direct_pmat"] is True
    assert isinstance(symbolic, dict)
    assert symbolic["ordering_kind"] == "natural"
    assert symbolic["block_size_target"] == 128
    assert symbolic["nnz"] == metadata["direct_pmat_nnz"]
    assert symbolic["diagonal_missing"] == 0
    assert symbolic["pattern_hash"]
    assert metadata["symbolic_cache_key"][2] == symbolic["pattern_hash"]


def test_transport_fortran_reduced_lu_defaults_to_mumps_like_symbolic_ordering(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_block_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "0")

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    symbolic = metadata.get("symbolic")

    assert metadata["factor_kind"] == "symbolic_block_lu"
    assert metadata["symbolic_ordering"] == "mumps_like"
    assert isinstance(symbolic, dict)
    assert symbolic["ordering_kind"] == "nested_dissection"


def test_transport_fortran_reduced_lu_accepts_symbolic_block_schur_metadata(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_block_schur_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS", "32")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_BOUNDARY_WIDTH", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_HIGH_DEGREE_COLS", "8")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "0")

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    symbolic = metadata.get("symbolic")

    assert metadata["factor_kind"] == "symbolic_block_schur_lu"
    assert metadata["direct_pmat"] is True
    assert metadata["symbolic_schur_max_separator_cols"] == 32
    assert metadata["symbolic_schur_boundary_width"] == 1
    assert metadata["symbolic_schur_high_degree_cols"] == 8
    assert metadata["symbolic_factor_coarse_size"] > 0
    assert isinstance(symbolic, dict)
    assert symbolic["ordering_kind"] == "natural"
    assert symbolic["block_size_target"] == 128
    assert symbolic["nnz"] == metadata["direct_pmat_nnz"]


def test_transport_fortran_reduced_symbolic_prefill_guard_skips_factorization(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_block_lu_coarse")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", "0.001")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR", "1")

    def fail_factorize(*_args, **_kwargs):
        raise AssertionError("symbolic prefill guard should reject before factorization")

    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fail_factorize)

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )

    rhs = jnp.ones((int(active.size),), dtype=jnp.float64)
    actual = preconditioner(rhs)
    assert actual.shape == rhs.shape
    assert np.all(np.isfinite(np.asarray(actual)))
    assert not hasattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata")


def test_transport_fortran_reduced_nd_setup_guard_skips_pattern_probe(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_nd_frontal_schur_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR", "1")

    def fail_factorize(*_args, **_kwargs):
        raise RuntimeError(
            "Host sparse factorization failed. Underlying factorization error: "
            "symbolic_nd_frontal_schur_lu setup time budget exceeded "
            "(92.0s>90.0s; stage=separator_update; node_size=55796 depth=3)"
        )

    def fail_pattern(*_args, **_kwargs):
        raise AssertionError("ND setup-budget rejection should skip the pattern-probe fallback")

    monkeypatch.setattr(sparse_direct, "factorize_host_sparse_operator", fail_factorize)
    monkeypatch.setattr(sparse_direct, "build_operator_from_pattern", fail_pattern)

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )

    rhs = jnp.ones((int(active.size),), dtype=jnp.float64)
    actual = preconditioner(rhs)
    assert actual.shape == rhs.shape
    assert np.all(np.isfinite(np.asarray(actual)))
    assert not hasattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata")


def test_transport_fortran_reduced_auto_exact_rescue_after_symbolic_prefill_guard(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MONOLITHIC_AUTO_MAX_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", "0.001")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR", "64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_SIZE", "10000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT_ADMISSION", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL", "1e-2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT", "10")

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    admission = metadata.get("direct_admission", {})

    assert metadata["factor_kind"] == "lu"
    assert metadata["auto_exact_rescue_selected"] is True
    assert metadata["auto_exact_rescue_max_mb"] == 256.0
    assert metadata["auto_exact_rescue_max_size"] == 10000
    assert metadata["effective_factor_max_mb"] == 256.0
    assert metadata["direct_pmat_auto_exact_rescue_selected"] is True
    assert admission["accepted"] is True
    assert admission["max_relative_residual"] < 1.0e-2
    assert metadata["factor_nbytes_estimate"] > 0


def test_transport_fortran_reduced_auto_exact_rescue_factor_entry_guard(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.delenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", raising=False)
    monkeypatch.delenv("SFINCS_JAX_EXPLICIT_SPARSE_FACTOR_KIND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_MONOLITHIC_AUTO_MAX_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR_MAX_MB", "0.001")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_PREFILL_SAFETY_FACTOR", "64")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_SIZE", "10000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_AUTO_EXACT_RESCUE_MAX_FACTOR_ENTRIES", "1")

    def fail_factor_build(*_args, **_kwargs):
        raise AssertionError("factor-entry guard should reject exact rescue before factorization")

    monkeypatch.setattr(vd, "_build_host_sparse_direct_factor_from_matvec", fail_factor_build)

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )

    rhs = jnp.ones((int(active.size),), dtype=jnp.float64)
    actual = preconditioner(rhs)
    assert actual.shape == rhs.shape
    assert np.all(np.isfinite(np.asarray(actual)))
    assert not hasattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata")


def test_transport_fortran_reduced_lu_rescues_rejected_symbolic_factor_with_direct_lu(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_block_schur_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "rcm")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", "512")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_SCHUR_MAX_SEPARATOR_COLS", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL", "1e-2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT", "10")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU_MAX_MB", "256")

    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    rescue = metadata.get("symbolic_admission_rescue_lu_admission", {})

    assert metadata["factor_kind"] == "lu"
    assert metadata["symbolic_admission"]["accepted"] is False
    assert metadata["symbolic_admission_rescue_lu"] is True
    assert rescue["accepted"] is True
    assert rescue["max_relative_residual"] < 1.0e-2
    assert metadata["factor_nbytes_estimate"] > 0


@pytest.mark.parametrize(
    "input_path",
    (
        Path("tests/reduced_inputs/transportMatrix_geometryScheme2.input.namelist"),
        Path("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist"),
    ),
)
def test_transport_fortran_reduced_lu_admits_blr_frontal_schur_on_reduced_geometry_rich_cases(
    monkeypatch,
    input_path: Path,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_blr_frontal_schur_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL", "1e-2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT", "10")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "rcm")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLOCK_SIZE", "384")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS", "2400")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH", "2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE", "4096")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS", "2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_FRONTAL_MAX_DENSE_RHS_ENTRIES", "400000000")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_MAX_RANK", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_TOL", "1e-6")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_RANK", "4096")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_BLR_FRONTAL_WOODBURY_MAX_CONDITION", "1")

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    admission = metadata.get("symbolic_admission", {})
    factor_metadata = metadata.get("symbolic_factor_metadata", {})

    assert metadata["factor_kind"] == "symbolic_blr_frontal_schur_lu"
    assert admission["accepted"] is True
    assert admission["max_relative_residual"] < 1.0e-2
    assert admission["min_improvement_vs_identity"] > 10.0
    assert factor_metadata["blr_update_count"] > 0
    assert factor_metadata["blr_rank_total"] > 0
    assert factor_metadata["blr_error_estimate_max"] < 1.0e-5


@pytest.mark.parametrize(
    "input_path",
    (
        Path("tests/reduced_inputs/transportMatrix_geometryScheme2.input.namelist"),
        Path("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist"),
    ),
)
def test_transport_fortran_reduced_lu_admits_nd_frontal_residual_polish_on_reduced_geometry_rich_cases(
    monkeypatch,
    input_path: Path,
) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_FACTOR", "symbolic_nd_frontal_schur_lu")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_DIRECT", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MAX_REL", "1e-2")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_MIN_IMPROVEMENT", "10")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ADMISSION_RESCUE_LU", "0")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ORDERING", "rcm")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_LEAF_SIZE", "384")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_TERMINAL_FACTOR_SIZE", "4096")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DEPTH", "1")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_SEPARATOR_WIDTH", "128")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_SEPARATOR_COLS", "4096")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_HIGH_DEGREE_COLS", "256")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES", "400000000")
    monkeypatch.setenv(
        "SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_DENSE_RHS_ENTRIES_PER_CHILD",
        "200000000",
    )
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_MAX_SETUP_S", "60")
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_FORTRAN_REDUCED_LU_SYMBOLIC_ND_RESIDUAL_POLISH_STEPS", "2")

    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    preconditioner = vd._build_rhsmode23_fp_fortran_reduced_lu_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_fortran_reduced_lu_metadata", {})
    admission = metadata.get("symbolic_admission", {})
    factor_metadata = metadata.get("symbolic_factor_metadata", {})

    assert metadata["factor_kind"] == "symbolic_nd_frontal_schur_lu"
    assert admission["accepted"] is True
    assert admission["max_relative_residual"] < 1.0e-2
    assert admission["min_improvement_vs_identity"] > 10.0
    assert factor_metadata["architecture"] == "symbolic_nd_frontal_schur_lu"
    assert factor_metadata["max_terminal_factor_size"] == 4096
    assert factor_metadata["max_setup_s"] == 60.0
    assert factor_metadata["max_dense_rhs_entries_per_child"] == 200000000
    assert factor_metadata["separator_update_mode"] == "csc_column_chunks"
    assert factor_metadata["separator_update_chunks"] > 0
    assert metadata["symbolic_nd_max_terminal_factor_size"] == 4096
    assert metadata["symbolic_nd_max_dense_rhs_entries_per_child"] == 200000000
    assert metadata["symbolic_nd_max_setup_s"] == 60.0
    assert factor_metadata["node_count"] >= 3
    assert factor_metadata["residual_polish_steps"] == 2


def test_transport_direct_pmat_physics_coarse_basis_includes_constraint_modes() -> None:
    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op_pc = vd._build_transport_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        keep_theta_zeta=True,
    )
    active = np.asarray(vd._transport_active_dof_indices(op_pc), dtype=np.int64)
    result = vd._try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op_pc,
        active_indices=active,
        factor_dtype=np.dtype(np.float64),
        pc_shift=1.0e-10,
        emit=None,
    )
    assert result is not None
    bundle, _metadata = result
    base_factor = factorize_host_sparse_operator(
        bundle,
        kind="symbolic_block_lu",
        symbolic_ordering_kind="natural",
        symbolic_block_size=128,
    )

    basis, names = vd._build_rhsmode23_direct_pmat_physics_coarse_basis(
        op=op_pc,
        active_indices=active,
        max_cols=64,
        base_factor_bundle=base_factor,
    )

    assert basis is not None
    assert basis.shape[0] == active.size
    assert basis.shape[1] == len(names)
    assert basis.nnz > basis.shape[1]
    assert any("tail_unit" in name for name in names)
    assert any("source_shape" in name or "l0_fsavg" in name for name in names)
    assert any("tail_schur_response" in name for name in names)


@pytest.mark.parametrize(
    "input_path",
    (
        Path("tests/reduced_inputs/transportMatrix_geometryScheme2.input.namelist"),
        Path("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist"),
    ),
)
def test_transport_direct_active_true_operator_matches_matrix_free_active_operator(input_path: Path) -> None:
    nml = read_sfincs_input(input_path)
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = vd._transport_active_dof_indices(op)

    result = vd._try_build_rhsmode23_fp_direct_active_operator_bundle(
        op=op,
        active_indices=active,
        factor_dtype=np.dtype(np.float64),
        emit=None,
    )

    assert result is not None
    bundle, metadata = result
    assert metadata["direct_true_operator"] is True
    assert bundle.matrix is not None
    assert bundle.matrix.shape == (active.size, active.size)
    assert bundle.matrix.nnz > op.total_size

    rng = np.random.default_rng(5678)
    active_i64 = np.asarray(active, dtype=np.int64)
    for _ in range(3):
        x_reduced = rng.normal(size=int(active.size))
        x_full = np.zeros((int(op.total_size),), dtype=np.float64)
        x_full[active_i64] = x_reduced
        y_true_full = vd.apply_v3_full_system_operator_cached(op, jnp.asarray(x_full, dtype=jnp.float64))
        y_true = np.asarray(y_true_full, dtype=np.float64)[active_i64]
        y_direct = bundle.matvec(x_reduced)
        np.testing.assert_allclose(y_direct, y_true, rtol=2e-11, atol=2e-10)


def test_transport_direct_active_block_schur_closes_true_tail_residual(monkeypatch) -> None:
    monkeypatch.setenv("SFINCS_JAX_TRANSPORT_FP_DIRECT_ACTIVE_BLOCK_SCHUR_ADMISSION", "0")
    nml = read_sfincs_input("tests/reduced_inputs/transportMatrix_geometryScheme11.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active = np.asarray(vd._transport_active_dof_indices(op), dtype=np.int64)
    active_jnp = jnp.asarray(active, dtype=jnp.int32)
    full_to_active = np.zeros((int(op.total_size) + 1,), dtype=np.int32)
    full_to_active[active + 1] = np.arange(1, int(active.size) + 1, dtype=np.int32)
    full_to_active_jnp = jnp.asarray(full_to_active, dtype=jnp.int32)

    def _reduce_full(v_full: jnp.ndarray) -> jnp.ndarray:
        return v_full[active_jnp]

    def _expand_reduced(v_reduced: jnp.ndarray) -> jnp.ndarray:
        padded = jnp.concatenate([jnp.zeros((1,), dtype=v_reduced.dtype), v_reduced], axis=0)
        return padded[full_to_active_jnp[1:]]

    result = vd._try_build_rhsmode23_fp_direct_active_operator_bundle(
        op=op,
        active_indices=active,
        factor_dtype=np.dtype(np.float64),
        emit=None,
    )
    assert result is not None
    bundle, metadata = result
    tail_size = int(metadata["direct_pmat_tail_size"])
    assert tail_size > 0

    preconditioner = vd._build_rhsmode23_fp_direct_active_block_schur_preconditioner(
        op=op,
        reduce_full=_reduce_full,
        expand_reduced=_expand_reduced,
        active_indices_np=active,
        emit=None,
    )
    pc_metadata = getattr(preconditioner, "_sfincs_jax_transport_fp_direct_active_block_schur_metadata", {})
    assert pc_metadata["tail_size"] == tail_size
    assert pc_metadata["schur_reason"] == "dense_schur"

    rng = np.random.default_rng(44)
    rhs = rng.normal(size=int(active.size))
    correction = np.asarray(preconditioner(jnp.asarray(rhs, dtype=jnp.float64)), dtype=np.float64)
    residual = np.asarray(bundle.matrix @ correction - rhs, dtype=np.float64)
    tail_residual = residual[-tail_size:]
    assert np.linalg.norm(tail_residual) <= 1.0e-7 * max(1.0, np.linalg.norm(rhs[-tail_size:]))


def test_fortran_reduced_pc_gmres_aliases_are_classified_as_sparse_host_pc() -> None:
    expected_aliases = {
        "fortran_reduced_pc_gmres",
        "fortran_reduced_sparse_pc_gmres",
        "fortran_like_pc_gmres",
        "petsc_like_pc_gmres",
    }

    assert expected_aliases == vd._SPARSE_HOST_FORTRAN_REDUCED_PC_GMRES_SOLVE_METHODS
    assert expected_aliases <= vd._SPARSE_HOST_PC_GMRES_SOLVE_METHODS
