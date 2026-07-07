from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import sfincs_jax.io as io_module
import sfincs_jax.operators.profile_full_system as profile_full_system_module
import sfincs_jax.operators.profile_sparse_pattern as sparse_pattern_module
import sfincs_jax.problems.profile_policies as profile_policies_module
import sfincs_jax.solvers.preconditioner_symbolic_host as symbolic_host_module
import sfincs_jax.solvers.preconditioning as preconditioning_module
from sfincs_jax.solvers import preconditioner_xblock_policy as rhs1_xblock_policy_module
import sfincs_jax.problems.profile_solve as profile_solve_module
from sfincs_jax.solvers.explicit_sparse import build_operator_from_pattern
from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.validation.fortran import read_petsc_mat_aij
from sfincs_jax.problems.profile_residual import (
    apply_device_subspace_residual_equation_correction,
    build_rhs1_xblock_post_coarse_directions,
)
from sfincs_jax.problems.transport_linear_system import transport_active_dof_indices
from sfincs_jax.solvers.preconditioner_xblock_policy import resolve_rhs1_xblock_sparse_pc_policy
from sfincs_jax.solver import FlexibleGMRESSolveResult
from sfincs_jax.operators.profile_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary,
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
    v3_full_system_fortran_reduced_preconditioner_sparsity_pattern,
)
from sfincs_jax.problems.profile_solve import (
    _rhs1_xblock_gmres_restart,
    _rhs1_xblock_precondition_side,
    _triangular_solve_lower_csr_rows,
    _triangular_solve_upper_csr_rows,
    solve_v3_full_system_linear_gmres,
)
from sfincs_jax.operators.profile_system import apply_v3_full_system_operator, full_system_operator_from_namelist


def _csr_from_petsc(path: Path) -> sp.csr_matrix:
    a = read_petsc_mat_aij(path)
    return sp.csr_matrix((a.data, a.col_ind, a.row_ptr), shape=a.shape)


def _assert_pattern_covers_matrix(pattern: sp.spmatrix, matrix: sp.spmatrix) -> None:
    pattern_bool = pattern.tocsr().astype(bool)
    matrix_bool = matrix.tocsr().astype(bool)
    covered = matrix_bool.multiply(pattern_bool)
    missing = matrix_bool.astype(np.int8) - covered.astype(np.int8)
    missing.eliminate_zeros()
    assert missing.nnz == 0


def _synthetic_sparse_pattern_op(
    *,
    n_species: int = 1,
    n_x: int = 2,
    n_xi: int = 2,
    n_theta: int = 2,
    n_zeta: int = 2,
    include_phi1: bool = False,
    include_phi1_in_kinetic: bool = False,
    constraint_scheme: int = 0,
    point_at_x0: bool = False,
    fp: object | None = None,
    pas: object | None = None,
    er_xdot: object | None = None,
    magdrift_xidot: object | None = None,
) -> SimpleNamespace:
    f_size = int(n_species * n_x * n_xi * n_theta * n_zeta)
    phi1_size = int(n_theta * n_zeta + 1) if include_phi1 else 0
    if int(constraint_scheme) == 2:
        extra_size = int(n_species * n_x)
    elif int(constraint_scheme) in {1, 3, 4}:
        extra_size = int(2 * n_species)
    else:
        extra_size = 0
    ddtheta = np.eye(int(n_theta), dtype=np.float64)
    if int(n_theta) > 1:
        ddtheta[0, 1] = 0.25
    ddzeta = np.eye(int(n_zeta), dtype=np.float64)
    if int(n_zeta) > 1:
        ddzeta[0, 1] = -0.5
    return SimpleNamespace(
        n_species=int(n_species),
        n_x=int(n_x),
        n_xi=int(n_xi),
        n_theta=int(n_theta),
        n_zeta=int(n_zeta),
        f_size=f_size,
        phi1_size=phi1_size,
        total_size=int(f_size + phi1_size + extra_size),
        include_phi1=bool(include_phi1),
        include_phi1_in_kinetic=bool(include_phi1_in_kinetic),
        constraint_scheme=int(constraint_scheme),
        point_at_x0=bool(point_at_x0),
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(ddtheta=ddtheta, ddzeta=ddzeta),
            fp=fp,
            fp_phi1=None,
            pas=pas,
            er_xdot=er_xdot,
            magdrift_xidot=magdrift_xidot,
        ),
    )


def _fast_device_krylov_result(**kwargs):
    """Return a converged device-Krylov result for solver-path metadata tests."""

    b = jnp.asarray(kwargs["b"], dtype=jnp.float64)
    x = jnp.zeros_like(b)
    history = jnp.asarray([jnp.linalg.norm(b), 0.0], dtype=jnp.float64)
    return (
        FlexibleGMRESSolveResult(
            x=x,
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
            residual_history=history,
            n_iterations=jnp.asarray(1, dtype=jnp.int32),
            n_restarts=jnp.asarray(0, dtype=jnp.int32),
            converged=jnp.asarray(True),
        ),
        jnp.zeros_like(b),
    )


def _fast_device_cycle_krylov_result(**kwargs):
    """Return a converged cycle-JIT FGMRES result with many internal iterations."""

    b = jnp.asarray(kwargs["b"], dtype=jnp.float64)
    x = jnp.zeros_like(b)
    history = jnp.asarray([jnp.linalg.norm(b), 1.0e-6, 0.0], dtype=jnp.float64)
    return (
        FlexibleGMRESSolveResult(
            x=x,
            residual_norm=jnp.asarray(0.0, dtype=jnp.float64),
            residual_history=history,
            n_iterations=jnp.asarray(80, dtype=jnp.int32),
            n_restarts=jnp.asarray(1, dtype=jnp.int32),
            converged=jnp.asarray(True),
        ),
        jnp.zeros_like(b),
    )


def test_sparse_host_ilu_escalates_regularization_after_singular_factor(monkeypatch) -> None:
    preconditioning_module._RHSMODE1_SPARSE_ILU_CACHE.clear()
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_REG", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_ILU_ATTEMPTS", "2")
    messages: list[str] = []

    _a_full, _a_drop, ilu = symbolic_host_module.factorize_sparse_matrix_csr_host(
        a_csr_full=sp.csr_matrix([[0.0, 0.0], [0.0, 1.0]]),
        cache_key=("singular-regularization-test",),
        drop_tol=0.0,
        drop_rel=0.0,
        ilu_drop_tol=0.0,
        fill_factor=10.0,
        factorization="ilu",
        emit=lambda _level, msg: messages.append(msg),
    )

    assert ilu is not None
    assert any("increasing diagonal regularization" in msg for msg in messages)


def test_xblock_precondition_side_defaults_right_only_for_full_fp_er() -> None:
    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=False,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=39_314,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=52_637,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=False,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="left",
        tokamak_fp_er_pc=True,
        full_fp_3d_pc=True,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("left", False)


def test_xblock_gmres_restart_caps_only_auto_right_preconditioned_path() -> None:
    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (20, True)

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="40",
        krylov_method="gmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (80, False)

    policy = resolve_rhs1_xblock_sparse_pc_policy(
        precondition_side_env_value="",
        krylov_env_value="",
        requested_restart=80,
        restart_env_value="",
        tokamak_fp_er_pc=False,
        full_fp_3d_pc=True,
        active_size=39_314,
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=False,
    )
    assert policy.precondition_side == "right"
    assert policy.default_right_preconditioned is True
    assert policy.gmres_restart == 80
    assert policy.restart_capped is False

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="lgmres",
        default_right_preconditioned=True,
    )
    assert (restart, capped) == (80, False)

    restart, capped = _rhs1_xblock_gmres_restart(
        requested_restart=80,
        restart_env_value="",
        krylov_method="gmres",
        default_right_preconditioned=False,
    )
    assert (restart, capped) == (80, False)


def test_compact_csr_triangular_solves_match_dense_reference() -> None:
    lower_indptr = jnp.asarray([0, 0, 1, 3], dtype=jnp.int32)
    lower_indices = jnp.asarray([0, 0, 1], dtype=jnp.int32)
    lower_data = jnp.asarray([2.0, -1.0, 0.5], dtype=jnp.float64)
    upper_indptr = jnp.asarray([0, 2, 3, 3], dtype=jnp.int32)
    upper_indices = jnp.asarray([1, 2, 2], dtype=jnp.int32)
    upper_data = jnp.asarray([-0.25, 0.5, 1.5], dtype=jnp.float64)
    upper_diag = jnp.asarray([4.0, -3.0, 2.0], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -2.0, 0.25], dtype=jnp.float64)

    y = _triangular_solve_lower_csr_rows(
        indptr=lower_indptr,
        indices=lower_indices,
        data=lower_data,
        b=rhs,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )
    z = _triangular_solve_upper_csr_rows(
        indptr=upper_indptr,
        indices=upper_indices,
        data=upper_data,
        upper_diag=upper_diag,
        b=y,
        row_base=jnp.asarray(0, dtype=jnp.int32),
    )

    lower = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0], [-1.0, 0.5, 1.0]])
    upper = np.array([[4.0, -0.25, 0.5], [0.0, -3.0, 1.5], [0.0, 0.0, 2.0]])
    expected = np.linalg.solve(upper, np.linalg.solve(lower, np.asarray(rhs)))
    np.testing.assert_allclose(np.asarray(z), expected, rtol=1.0e-12, atol=1.0e-12)


def test_sparse_pattern_support_helpers_encode_fortran_style_reduced_blocks() -> None:
    op = SimpleNamespace(
        n_species=2,
        n_x=4,
        n_xi=5,
        n_theta=3,
        n_zeta=2,
        f_size=2 * 4 * 5 * 3 * 2,
        phi1_size=3 * 2,
        constraint_scheme=2,
        include_phi1=True,
        fblock=SimpleNamespace(fp=object(), pas=object()),
    )

    assert sparse_pattern_module._f_index(op, 1, 2, 3, 1, 0) == (((1 * 4 + 2) * 5 + 3) * 3 + 1) * 2
    assert sparse_pattern_module._phi1_index(op, 2, 1) == op.f_size + 5
    assert sparse_pattern_module._lambda_index(op) == op.f_size + op.n_theta * op.n_zeta
    assert sparse_pattern_module._extra_index(op, 3) == op.f_size + op.phi1_size + 3

    supports = sparse_pattern_module._matrix_row_supports(
        np.asarray([[0.0, 1.0e-13, 0.0], [-2.0, 0.0, 3.0]]),
        threshold=1.0e-14,
    )
    np.testing.assert_array_equal(supports[0], np.asarray([1], dtype=np.int32))
    np.testing.assert_array_equal(supports[1], np.asarray([0, 2], dtype=np.int32))
    with pytest.raises(ValueError, match="2D derivative matrix"):
        sparse_pattern_module._matrix_row_supports(np.ones(3))

    assert list(sparse_pattern_module._nearby_l(0, 5, radius=2)) == [0, 1, 2]
    assert list(sparse_pattern_module._nearby_l(4, 5, radius=2)) == [2, 3, 4]

    assert [list(r) for r in sparse_pattern_module._x_supports(op, dense=True)] == [list(range(4))] * 4
    assert [list(r) for r in sparse_pattern_module._x_supports(op, dense=False)] == [[0], [1], [2], [3]]

    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=0)] == [
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
        [0, 1, 2, 3],
    ]
    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=1)] == [
        [0],
        [1],
        [2],
        [3],
    ]
    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=2)] == [
        [0, 1, 2, 3],
        [1, 2, 3],
        [2, 3],
        [3],
    ]
    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=3)] == [
        [0, 1],
        [0, 1, 2],
        [1, 2, 3],
        [2, 3],
    ]
    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=4)] == [
        [0, 1],
        [1, 2],
        [2, 3],
        [3],
    ]
    assert [list(r) for r in sparse_pattern_module._fortran_reduced_x_supports(op, preconditioner_x=99)] == [
        [0],
        [1],
        [2],
        [3],
    ]

    low_l_supports = sparse_pattern_module._fortran_reduced_x_supports_for_l(
        op,
        ell=1,
        preconditioner_x=1,
        preconditioner_x_min_l=2,
    )
    high_l_supports = sparse_pattern_module._fortran_reduced_x_supports_for_l(
        op,
        ell=2,
        preconditioner_x=1,
        preconditioner_x_min_l=2,
    )
    assert [list(r) for r in low_l_supports] == [list(range(4))] * 4
    assert [list(r) for r in high_l_supports] == [[0], [1], [2], [3]]
    assert list(sparse_pattern_module._fortran_reduced_l_supports(2, 5, preconditioner_xi=0)) == [0, 1, 2, 3, 4]
    assert list(sparse_pattern_module._fortran_reduced_l_supports(2, 5, preconditioner_xi=1)) == [1, 2, 3]


def test_sparse_pattern_summary_dict_preserves_operator_metadata() -> None:
    op = SimpleNamespace(
        include_phi1=True,
        constraint_scheme=2,
        fblock=SimpleNamespace(fp=object(), pas=None),
    )
    pattern = sp.csr_matrix(np.asarray([[1.0, 0.0, 2.0], [0.0, 0.0, 0.0], [3.0, 4.0, 0.0]]))

    summary = sparse_pattern_module.summarize_v3_sparse_pattern(op, pattern)

    assert summary.shape == (3, 3)
    assert summary.nnz == 4
    assert summary.avg_row_nnz == pytest.approx(4.0 / 3.0)
    assert summary.max_row_nnz == 2
    assert summary.include_phi1 is True
    assert summary.constraint_scheme == 2
    assert summary.has_fp is True
    assert summary.has_pas is False
    assert summary.to_dict() == {
        "shape": (3, 3),
        "nnz": 4,
        "avg_row_nnz": pytest.approx(4.0 / 3.0),
        "max_row_nnz": 2,
        "include_phi1": True,
        "constraint_scheme": 2,
        "has_fp": True,
        "has_pas": False,
    }


def test_sparse_pattern_empty_and_invalid_active_sets_fail_closed() -> None:
    empty = _synthetic_sparse_pattern_op(n_x=0, n_xi=0, n_theta=0, n_zeta=0)
    estimate = estimate_v3_full_system_conservative_sparsity_summary(empty)
    assert estimate.shape == (0, 0)
    assert estimate.nnz == 0
    assert v3_full_system_conservative_sparsity_pattern(empty).shape == (0, 0)
    assert sparse_pattern_module.v3_full_system_conservative_sparsity_pattern_for_indices(
        empty,
        np.asarray([], dtype=np.int32),
    ).shape == (0, 0)

    op = _synthetic_sparse_pattern_op()
    bad_active = np.asarray([0, op.total_size], dtype=np.int32)
    with pytest.raises(ValueError, match="outside the full-system vector"):
        sparse_pattern_module.v3_full_system_conservative_sparsity_pattern_for_indices(op, bad_active)
    with pytest.raises(ValueError, match="outside the full-system vector"):
        sparse_pattern_module.v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
            op,
            bad_active,
        )


def test_sparse_pattern_phi1_in_kinetic_adds_local_angular_phi1_columns() -> None:
    op = _synthetic_sparse_pattern_op(
        n_x=1,
        n_xi=2,
        include_phi1=True,
        include_phi1_in_kinetic=True,
    )
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    row = sparse_pattern_module._f_index(op, 0, 0, 0, 0, 0)
    cols = set(pattern.getrow(row).indices.tolist())

    assert sparse_pattern_module._phi1_index(op, 0, 0) in cols
    assert sparse_pattern_module._phi1_index(op, 1, 0) in cols
    assert sparse_pattern_module._phi1_index(op, 0, 1) in cols


def test_sparse_pattern_constraint_scheme2_point_at_x0_matches_fortran_source_rows() -> None:
    op = _synthetic_sparse_pattern_op(
        n_x=2,
        n_xi=1,
        n_theta=1,
        n_zeta=1,
        constraint_scheme=2,
        point_at_x0=True,
    )
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    f_x0 = sparse_pattern_module._f_index(op, 0, 0, 0, 0, 0)
    f_x1 = sparse_pattern_module._f_index(op, 0, 1, 0, 0, 0)
    source_x0 = sparse_pattern_module._extra_index(op, 0)
    source_x1 = sparse_pattern_module._extra_index(op, 1)

    assert source_x0 not in set(pattern.getrow(f_x0).indices.tolist())
    assert source_x1 in set(pattern.getrow(f_x1).indices.tolist())
    assert f_x0 in set(pattern.getrow(source_x0).indices.tolist())
    assert source_x0 in set(pattern.getrow(source_x0).indices.tolist())


def test_active_sparse_pattern_fallback_matches_full_slice_for_partial_angular_set() -> None:
    op = _synthetic_sparse_pattern_op(
        n_x=2,
        n_xi=2,
        include_phi1=True,
        include_phi1_in_kinetic=True,
        constraint_scheme=2,
        point_at_x0=True,
        fp=object(),
        er_xdot=object(),
    )
    active = np.asarray(
        [
            sparse_pattern_module._f_index(op, 0, 0, 0, 0, 0),
            sparse_pattern_module._f_index(op, 0, 1, 0, 0, 0),
            sparse_pattern_module._phi1_index(op, 0, 0),
            sparse_pattern_module._extra_index(op, 0),
            sparse_pattern_module._extra_index(op, 1),
        ],
        dtype=np.int32,
    )

    full = v3_full_system_conservative_sparsity_pattern(op)[active, :][:, active].tocsr()
    active_pattern = sparse_pattern_module.v3_full_system_conservative_sparsity_pattern_for_indices(op, active)
    reduced = sparse_pattern_module.v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
        op,
        active,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=0,
    )

    assert (active_pattern != full).nnz == 0
    assert reduced.shape == active_pattern.shape
    assert reduced.nnz <= active_pattern.nnz


def test_active_sparse_pattern_structured_path_matches_full_slice_for_complete_blocks() -> None:
    op = _synthetic_sparse_pattern_op(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=2,
        n_zeta=2,
        include_phi1=True,
        include_phi1_in_kinetic=True,
        constraint_scheme=1,
        fp=object(),
        er_xdot=object(),
    )
    n_tz = op.n_theta * op.n_zeta
    active_velocity_blocks = np.asarray([0, 1, 3], dtype=np.int32)
    active_f = (
        active_velocity_blocks[:, None].astype(np.int64) * n_tz
        + np.arange(n_tz, dtype=np.int64)[None, :]
    ).reshape((-1,))
    active_tail = np.arange(op.f_size, op.total_size, dtype=np.int64)
    active = np.concatenate([active_f, active_tail]).astype(np.int32)

    full_slice = v3_full_system_conservative_sparsity_pattern(op)[active, :][:, active].tocsr()
    active_pattern = v3_full_system_conservative_sparsity_pattern_for_indices(op, active)

    assert active_pattern.shape == full_slice.shape
    assert (active_pattern != full_slice).nnz == 0
    assert active_pattern.nnz > 0


def test_fortran_reduced_active_sparse_pattern_matches_full_reduced_slice_for_complete_blocks() -> None:
    op = _synthetic_sparse_pattern_op(
        n_species=2,
        n_x=2,
        n_xi=4,
        n_theta=2,
        n_zeta=2,
        include_phi1=True,
        include_phi1_in_kinetic=True,
        constraint_scheme=2,
        point_at_x0=True,
        fp=object(),
        er_xdot=object(),
    )
    n_tz = op.n_theta * op.n_zeta
    active_velocity_blocks = np.asarray([0, 1, 5, 8], dtype=np.int32)
    active_f = (
        active_velocity_blocks[:, None].astype(np.int64) * n_tz
        + np.arange(n_tz, dtype=np.int64)[None, :]
    ).reshape((-1,))
    active_tail = np.arange(op.f_size, op.total_size, dtype=np.int64)
    active = np.concatenate([active_f, active_tail]).astype(np.int32)

    kwargs = dict(
        preconditioner_x=3,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=2,
    )
    full_reduced = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(op, **kwargs)
    active_reduced = sparse_pattern_module.v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
        op,
        active,
        **kwargs,
    )
    conservative_active = v3_full_system_conservative_sparsity_pattern_for_indices(op, active)

    assert active_reduced.shape == (active.size, active.size)
    assert (active_reduced != full_reduced[active, :][:, active].tocsr()).nnz == 0
    assert active_reduced.nnz <= conservative_active.nnz


def test_sparse_pattern_unsupported_constraint_scheme_fails_closed() -> None:
    op = _synthetic_sparse_pattern_op(
        n_x=1,
        n_xi=1,
        n_theta=1,
        n_zeta=1,
        constraint_scheme=9,
    )
    active = np.arange(op.total_size, dtype=np.int32)

    with pytest.raises(NotImplementedError, match="constraintScheme=9"):
        v3_full_system_conservative_sparsity_pattern(op)
    with pytest.raises(NotImplementedError, match="constraintScheme=9"):
        v3_full_system_conservative_sparsity_pattern_for_indices(op, active)
    with pytest.raises(NotImplementedError, match="constraintScheme=9"):
        sparse_pattern_module.v3_full_system_fortran_reduced_preconditioner_sparsity_pattern_for_indices(
            op,
            active,
        )


def test_conservative_sparse_pattern_covers_pas_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.nnz == pattern.nnz
    assert summary.has_pas
    assert not summary.has_fp


def test_conservative_sparse_pattern_covers_fp_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "quick_2species_FPCollisions_noEr.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.has_fp
    assert summary.avg_row_nnz > 0.0


def test_conservative_sparse_pattern_preflight_estimate_bounds_materialized_pattern() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    summary = summarize_v3_sparse_pattern(op, pattern)
    estimate = estimate_v3_full_system_conservative_sparsity_summary(op)

    assert estimate.shape == summary.shape
    assert estimate.nnz >= summary.nnz
    assert estimate.max_row_nnz >= summary.max_row_nnz
    assert estimate.has_fp is True


def test_fp_sparse_pc_can_use_local_velocity_pattern() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    dense_velocity = v3_full_system_conservative_sparsity_pattern(op, fp_dense_velocity_block=True)
    local_velocity = v3_full_system_conservative_sparsity_pattern(op, fp_dense_velocity_block=False)

    assert local_velocity.shape == dense_velocity.shape == (op.total_size, op.total_size)
    assert local_velocity.nnz < dense_velocity.nnz
    assert summarize_v3_sparse_pattern(op, local_velocity).has_fp


def test_conservative_sparse_pattern_covers_phi1_fortran_matrix() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_withPhi1_linear.whichMatrix_3.petscbin")

    assert pattern.shape == matrix.shape == (op.total_size, op.total_size)
    _assert_pattern_covers_matrix(pattern, matrix)
    summary = summarize_v3_sparse_pattern(op, pattern)
    assert summary.include_phi1
    assert summary.max_row_nnz >= op.n_theta * op.n_zeta


def test_pattern_probe_recovers_pas_tiny_matrix_free_operator() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    pattern = v3_full_system_conservative_sparsity_pattern(op)
    fortran_matrix = _csr_from_petsc(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.whichMatrix_3.petscbin")

    def mv(x):
        return apply_v3_full_system_operator(op, jnp.asarray(x, dtype=jnp.float64))

    bundle = build_operator_from_pattern(mv, pattern=pattern, backend="cpu")

    assert sp.isspmatrix_csr(bundle.matrix)
    assert bundle.metadata.block_cols < op.total_size
    np.testing.assert_allclose(bundle.matrix.toarray(), fortran_matrix.toarray(), rtol=0, atol=3e-12)


def test_sparse_host_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_host",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-10
    assert any("sparse_host complete" in msg for msg in messages)


def test_sparse_pc_gmres_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_pc_gmres",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-10
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "sparse_pc_gmres"
    assert result.metadata["setup_s"] >= 0.0
    assert result.metadata["solve_s"] >= 0.0
    assert result.metadata["elapsed_s"] >= result.metadata["setup_s"]
    assert result.metadata["sparse_pc_factor_dtype"] == "float64"
    assert result.metadata["sparse_pc_initial_factor_dtype"] == "float64"
    assert result.metadata["sparse_pc_factor_dtype_retry"] is None
    assert result.metadata["sparse_pc_first_attempt_maxiter"] == result.metadata["gmres_maxiter"]
    assert result.metadata["sparse_pc_permc_spec"] in {"COLAMD", "MMD_ATA"}
    assert result.metadata["sparse_pc_default_permc_spec"] in {"COLAMD", "MMD_ATA"}
    assert result.metadata["sparse_pattern_nnz"] > 0
    assert result.metadata["sparse_pattern_max_row_nnz"] > 0
    assert any("sparse_pc_gmres complete" in msg for msg in messages)


def test_fortran_reduced_pc_operator_preserves_angular_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    point = preconditioning_module._build_rhsmode1_preconditioner_operator_point(op)
    reduced = preconditioning_module._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddtheta),
        np.asarray(op.fblock.collisionless.ddtheta),
    )
    np.testing.assert_allclose(
        np.asarray(reduced.fblock.collisionless.ddzeta),
        np.asarray(op.fblock.collisionless.ddzeta),
    )
    np.testing.assert_allclose(
        np.asarray(point.fblock.collisionless.ddtheta),
        np.diag(np.diag(np.asarray(op.fblock.collisionless.ddtheta))),
    )
    np.testing.assert_allclose(
        np.asarray(point.fblock.collisionless.ddzeta),
        np.diag(np.diag(np.asarray(op.fblock.collisionless.ddzeta))),
    )


def test_fortran_reduced_pc_pattern_keeps_global_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    point = preconditioning_module._build_rhsmode1_preconditioner_operator_point(op)
    reduced = preconditioning_module._build_rhsmode1_preconditioner_operator_fortran_reduced(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    )

    true_pattern = v3_full_system_conservative_sparsity_pattern(op).astype(bool).tocsr()
    point_pattern = v3_full_system_conservative_sparsity_pattern(point).astype(bool).tocsr()
    reduced_pattern = v3_full_system_conservative_sparsity_pattern(reduced).astype(bool).tocsr()

    missing_from_true = reduced_pattern.astype(np.int8) - reduced_pattern.multiply(true_pattern).astype(np.int8)
    missing_from_true.eliminate_zeros()

    assert point_pattern.nnz < reduced_pattern.nnz
    assert reduced_pattern.nnz <= true_pattern.nnz
    assert missing_from_true.nnz == 0


def test_fortran_reduced_structural_pattern_drops_fp_x_species_coupling() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    conservative = v3_full_system_conservative_sparsity_pattern(op).astype(bool).tocsr()
    reduced = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
    ).astype(bool).tocsr()

    def idx(s: int, x: int, ell: int, theta: int, zeta: int) -> int:
        return (((s * op.n_x + x) * op.n_xi + ell) * op.n_theta + theta) * op.n_zeta + zeta

    row = idx(0, 0, 0, 2, 2)
    off_species_col = idx(1, 0, 0, 2, 2)
    off_x_col = idx(0, 1, 0, 2, 2)
    theta_coupled_col = idx(0, 0, 1, 1, 2)
    zeta_coupled_col = idx(0, 0, 1, 2, 1)
    off_xi2_col = idx(0, 0, 2, 2, 2)

    assert reduced.shape == conservative.shape
    assert reduced.nnz < conservative.nnz
    assert conservative[row, off_species_col]
    assert conservative[row, off_x_col]
    assert not reduced[row, off_species_col]
    assert not reduced[row, off_x_col]
    assert not reduced[row, off_xi2_col]
    assert reduced[row, theta_coupled_col]
    assert reduced[row, zeta_coupled_col]


def test_fortran_reduced_structural_pattern_respects_preconditioner_x_min_l() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3

    op = full_system_operator_from_namelist(nml=nml)
    reduced = v3_full_system_fortran_reduced_preconditioner_sparsity_pattern(
        op,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=1,
    ).astype(bool).tocsr()

    def idx(s: int, x: int, ell: int, theta: int, zeta: int) -> int:
        return (((s * op.n_x + x) * op.n_xi + ell) * op.n_theta + theta) * op.n_zeta + zeta

    low_l_row = idx(0, 0, 0, 2, 2)
    high_l_row = idx(0, 0, 1, 2, 2)
    low_l_off_x = idx(0, 1, 0, 2, 2)
    high_l_off_x = idx(0, 1, 1, 2, 2)

    assert reduced[low_l_row, low_l_off_x]
    assert not reduced[high_l_row, high_l_off_x]


def test_fortran_reduced_pc_gmres_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "64")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced"] is True
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_global"
    assert result.metadata["sparse_pc_fortran_reduced_keeps_theta_zeta"] is True
    assert result.metadata["sparse_pc_fortran_reduced_preconditioner_x"] == 1
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_full"
    assert result.metadata["sparse_pattern_nnz"] > 0
    assert any("fortran_reduced_pc_gmres using global angular-coupled" in msg for msg in messages)
    assert any("sparse_pc_gmres complete" in msg for msg in messages)


def test_auto_selects_fortran_reduced_pc_gmres_for_large_full_fp_rhs1(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_AUTO_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_EXPLICIT_SPARSE_CSR_MAX_MB", "64")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="auto",
        differentiable=False,
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert result.metadata is not None
    assert result.metadata["auto_solver_selected"] is True
    assert result.metadata["auto_solver_policy"] == "fortran_reduced_pc_gmres"
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced"] is True
    assert any("auto selecting Fortran-reduced sparse-PC GMRES" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_enabled"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_error"] is None
    assert "structured direct-tail CSR" in result.metadata[
        "sparse_pc_fortran_reduced_direct_tail_operator_reason"
    ]
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] > 0
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_csr_nbytes_estimate"] > 0
    assert result.metadata["sparse_pc_operator_nnz_estimate"] > 0
    assert result.metadata["sparse_pc_operator_csr_nbytes_estimate"] > 0
    assert result.metadata["sparse_pc_factor_elapsed_s"] >= 0.0
    assert result.metadata["sparse_pc_residual_target"] > 0.0
    assert result.metadata["sparse_pc_residual_ratio_to_target"] < 1.0
    assert result.metadata["sparse_pc_factor_quality_rejected"] is False
    assert result.metadata["sparse_pc_factor_preflight_enabled"] is True
    assert result.metadata["sparse_pc_factor_preflight_required"] is False
    assert result.metadata["sparse_pc_factor_preflight_residual_before"] > 0.0
    assert result.metadata["sparse_pc_factor_preflight_residual_after"] >= 0.0
    assert result.metadata["sparse_pc_factor_preflight_improvement_ratio"] is not None
    assert result.metadata["sparse_pc_factor_preflight_target_ratio"] is not None
    assert any("fortran_reduced direct-tail structured csr built" in msg for msg in messages)
    assert any("explicit_sparse: factorization complete" in msg for msg in messages)
    assert any("sparse_pc_gmres factor preflight" in msg for msg in messages)


def test_sparse_pc_post_minres_records_true_residual_improvement(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_POST_MINRES_STEPS", "2")

    def fake_gmres_solve_with_history_scipy(**kwargs):
        b = np.asarray(kwargs["b"], dtype=np.float64)
        x = np.zeros_like(b)
        return x, float(np.linalg.norm(b)), [float(np.linalg.norm(b))]

    monkeypatch.setattr(
        profile_solve_module,
        "gmres_solve_with_history_scipy",
        fake_gmres_solve_with_history_scipy,
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-12,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.metadata["sparse_pc_post_minres_steps_requested"] == 2
    assert result.metadata["sparse_pc_post_minres_steps_accepted"] > 0
    assert result.metadata["sparse_pc_post_minres_error"] is None
    assert result.metadata["sparse_pc_post_minres_residual_before"] is not None
    assert result.metadata["sparse_pc_post_minres_residual_after"] is not None
    assert (
        result.metadata["sparse_pc_post_minres_residual_after"]
        < result.metadata["sparse_pc_post_minres_residual_before"]
    )
    assert any("sparse_pc_gmres post-minres improved residual" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_can_fallback_to_pattern_probe(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert "direct-tail materialization" in result.metadata[
        "sparse_pc_fortran_reduced_direct_tail_operator_reason"
    ]
    assert any("fortran_reduced direct-tail materialization csr built" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_whichmatrix0_active_terms_solve_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "whichMatrix0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    reason = result.metadata["sparse_pc_fortran_reduced_direct_tail_operator_reason"]
    assert "whichMatrix=0 active term-level" in reason
    assert "no kinetic probing" in reason
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_nnz"] > 0
    assert any("whichMatrix=0 active term CSR built" in msg for msg in messages)


def test_fortran_reduced_direct_tail_auto_preconditioner_uses_active_ladder(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES", "jacobi")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    # CI/Linux JAX releases can differ at the last Krylov iteration by a few
    # ulps around the requested tolerance. This test is about auto path
    # selection, so keep the residual gate tight without making it bit-fragile.
    assert float(result.residual_norm) < 1.2e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "auto_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "auto"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "jacobi"
    assert structured_metadata["metadata"]["auto_selected_kind"] == "jacobi"


def test_fortran_reduced_direct_pmat_preconditioner_skips_active_csr_materialization(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_direct_pmat_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "256")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_XI", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_SPECIES", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X_MIN_L", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_pmat_requested"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is False
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_fortran_v3_reduced_direct_pmat_lu"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["metadata"]["direct_reduced_pmat_emission"] is True
    assert structured_metadata["metadata"]["direct_reduced_pmat_avoids_full_active_true_csr"] is True
    assert any("materialization skipped; direct reduced-Pmat preconditioner requested" in msg for msg in messages)
    assert not any("whichMatrix=0 active term CSR built" in msg for msg in messages)


def test_fortran_reduced_direct_tail_auto_retries_active_lu_after_native_preflight_failure(
    monkeypatch,
) -> None:
    profile_policies_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES",
        "active_fortran_v3_reduced_native_stack,active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "auto_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "auto"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"]
        == "auto_retry_selected_no_required_preflight:complete"
    )
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    pc_metadata = structured_metadata["metadata"]
    assert pc_metadata["auto_preflight_retry_selected"] is True
    attempts = pc_metadata["auto_preflight_retry_attempts"]
    assert attempts[0]["kind"] == "active_fortran_v3_reduced_lu"
    assert attempts[0]["preflight_required"] is False
    assert attempts[0]["preflight_passed"] is False
    assert attempts[0]["preflight_policy_passed"] is True
    assert any(
        "structured preconditioner selected kind=active_fortran_v3_reduced_native_stack" in msg
        for msg in messages
    )
    assert any(
        "auto preflight retry accepted kind=active_fortran_v3_reduced_lu required=False" in msg
        for msg in messages
    )


def test_fortran_reduced_direct_tail_large_auto_fails_closed_before_host_factor_fallback(
    monkeypatch,
) -> None:
    profile_policies_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER", "auto")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "1e-6")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_FAIL_CLOSED_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES", "active_fortran_v3_reduced_lu")
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="direct-tail structured preconditioner was explicitly requested"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )

    assert any("structured preconditioner not selected" in msg for msg in messages)
    assert not any("sparse_pc_gmres host sparse factor built" in msg for msg in messages)


def test_structured_direct_tail_uses_actual_csr_budget_instead_of_preflight() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    op = full_system_operator_from_namelist(nml=nml)
    messages: list[str] = []

    bundle = profile_full_system_module._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=None,
        csr_max_mb=1.0e-4,
        drop_tol=0.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert bundle is None
    assert any("rejected actual CSR budget" in msg for msg in messages)
    assert not any("csr_budget_preflight_exceeded" in msg for msg in messages)


def test_structured_direct_tail_skips_large_project_after_build(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    op = full_system_operator_from_namelist(nml=nml)
    active = np.arange(int(op.total_size) - 1, dtype=np.int32)
    messages: list[str] = []
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_PROJECT_AFTER_BUILD_MAX_SIZE", "1")

    bundle = profile_full_system_module._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=active,
        csr_max_mb=100.0,
        drop_tol=0.0,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert bundle is None
    assert any("skipped full build before active projection" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_diagonal_schur_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_diagonal_schur",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "active_diagonal_schur"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_diagonal_schur"
    assert structured_metadata["metadata"]["tail_size"] > 0
    assert structured_metadata["metadata"]["factor_nbytes_actual"] > 0
    assert result.metadata["sparse_pc_factor_elapsed_s"] >= 0.0
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_diagonal_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_global_field_split_schur_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_global_field_split_schur",
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_global_field_split_schur"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_global_field_split_schur"
    assert structured_metadata["metadata"]["architecture"] == "active_kinetic_global_field_split_schur"
    assert structured_metadata["metadata"]["tail_size"] > 0
    assert structured_metadata["metadata"]["factor_nbytes_actual"] > 0
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_global_field_split_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_fortran_v3_reduced_ilu_fails_fast_when_preflight_worsens(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_ilu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FILL_FACTOR", "8")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DROP_TOL", "0")
    with pytest.raises(RuntimeError, match="direct-tail structured preconditioner preflight failed") as excinfo:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )

    message = str(excinfo.value)
    assert "active_fortran_v3_reduced_ilu" in message
    assert "target_ratio=" in message


def test_fortran_reduced_pc_gmres_direct_tail_active_xblock_ilu_low_l_schur_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_xblock_ilu_low_l_schur",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_ALLOW_SINGULAR_FALLBACK", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_DROP_TOL", "1e-3")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=120,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_xblock_ilu_low_l_schur"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_reason"] == "complete"
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_low_l_schur"
    assert structured_metadata["metadata"]["factor_kind"] == "spilu"
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["factor_kind"] == "spilu"
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["allow_block_fallback"] is True
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == structured_metadata["metadata"]["factor_nbytes_actual"]
    assert any("structured preconditioner selected kind=active_low_l_schur" in msg for msg in messages)
    assert not any("explicit_sparse: factorization complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_direct_tail_active_native_xell_coarse_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_native_xell_field_split_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "128")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_native_xell_field_split_sparse_coarse"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_native_xell_field_split_sparse_coarse"
    assert (
        structured_metadata["metadata"]["architecture"]
        == "active_native_xell_global_field_split_sparse_coarse"
    )
    assert structured_metadata["metadata"]["base_preconditioner"]["metadata"]["requested_base_kind"] == "active_native_xell"
    assert structured_metadata["metadata"]["coarse_size"] > 0
    assert structured_metadata["metadata"]["az_basis_nnz"] > 0
    assert any(
        "structured preconditioner selected kind=active_native_xell_field_split_sparse_coarse" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_angular_line_coarse_solves_tiny_rhs1_system(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_angular_line_field_split_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_angular_line_field_split_sparse_coarse"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_angular_line_field_split_sparse_coarse"
    assert (
        structured_metadata["metadata"]["architecture"]
        == "active_angular_line_global_field_split_sparse_coarse"
    )
    assert structured_metadata["metadata"]["requested_base_kind"] == "active_angular_line"
    assert structured_metadata["metadata"]["adaptive_residual_basis_enabled"] is True
    assert any(
        "structured preconditioner selected kind=active_angular_line_field_split_sparse_coarse" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_bounded_native_stack_fails_fast_when_preflight_worsens(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_bounded_native_stack",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    messages: list[str] = []

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_bounded_native_stack" in str(exc)
    else:  # pragma: no cover - the bounded stack must not enter GMRES if it worsens residual.
        raise AssertionError("bounded native stack should fail fast when preflight worsens the residual")
    assert any(
        "structured preconditioner selected kind=active_bounded_native_stack" in msg
        for msg in messages
    )


def test_fortran_reduced_pc_gmres_direct_tail_active_native_stack_production_alias_fails_fast(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_native_stack",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_COARSE_MAX_SIZE", "128")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_TARGET_RATIO",
        "1",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_AUTO_MIN_SIZE",
        "1",
    )
    messages: list[str] = []

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_fortran_v3_reduced_native_stack" in str(exc)
    else:  # pragma: no cover - the production alias must stay residual-gated.
        raise AssertionError("production native stack should fail fast when preflight worsens the residual")
    assert any(
        "structured preconditioner selected kind=active_fortran_v3_reduced_native_stack" in msg
        for msg in messages
    )


def test_fortran_reduced_direct_tail_structured_pc_preflight_can_fail_fast(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_diagonal_schur",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )

    try:
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )
    except RuntimeError as exc:
        assert "direct-tail structured preconditioner preflight failed" in str(exc)
        assert "active_diagonal_schur" in str(exc)
    else:  # pragma: no cover - the preflight must reject this weak one-step factor.
        raise AssertionError("structured direct-tail preflight should fail fast when explicitly required")


def test_fortran_reduced_pc_gmres_direct_tail_sparse_coarse_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_tail_sparse_coarse",
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"] == "active_tail_sparse_coarse"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_tail_sparse_coarse"
    assert structured_metadata["metadata"]["base_kind"] == "active_diagonal_schur"
    assert structured_metadata["metadata"]["coarse_size"] > 0
    assert structured_metadata["metadata"]["az_basis_nnz"] > 0


def test_fortran_reduced_direct_tail_structured_pc_cache_reuses_candidate(monkeypatch) -> None:
    profile_policies_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_schwarz_sparse_coarse",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", "32")

    first = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )
    second = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    first_metadata = first.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    second_metadata = second.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert first_metadata["metadata"]["direct_tail_structured_pc_cache_hit"] is False
    assert second_metadata["metadata"]["direct_tail_structured_pc_cache_hit"] is True
    assert first_metadata["metadata"]["direct_tail_structured_pc_cache_key_digest"] == second_metadata["metadata"][
        "direct_tail_structured_pc_cache_key_digest"
    ]
    assert second_metadata["metadata"]["architecture"] == "additive_schwarz_global_sparse_coarse"


def test_fortran_reduced_direct_tail_active_lu_preflight_stays_diagnostic_under_size_gate(
    monkeypatch,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_PREFLIGHT_REQUIRED_MIN_SIZE",
        "1",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_TRUE_COUPLED_COARSE_AUTO", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    assert result.metadata["sparse_pc_factor_preflight_enabled"] is True
    assert result.metadata["sparse_pc_factor_preflight_required"] is False
    assert result.metadata["sparse_pc_direct_tail_structured_pc_preflight_required"] is False
    assert result.metadata["sparse_pc_factor_preflight_residual_before"] > 0.0
    assert result.metadata["sparse_pc_factor_preflight_residual_after"] >= 0.0
    assert any("sparse_pc_gmres factor preflight" in msg for msg in messages)


def test_fortran_reduced_direct_tail_pc_default_cap_is_adaptive_for_active_lu(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_BASE_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MAX_MB", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_AUTO_MB_PER_UNKNOWN", raising=False)

    small = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=604,
    )
    mid = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=110_000,
    )
    production = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=900_000,
    )
    fullgrid_qa_qh = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=507_004,
    )
    upper_midgrid = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=169_264,
    )
    auto_upper_midgrid = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="auto",
        active_size=169_264,
    )
    non_exact = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_xblock",
        active_size=900_000,
    )

    assert small == pytest.approx(521.664)
    assert mid == pytest.approx(2272.0)
    assert upper_midgrid == pytest.approx(3220.224)
    assert auto_upper_midgrid == pytest.approx(3220.224)
    assert fullgrid_qa_qh == pytest.approx(14708.112)
    assert production == pytest.approx(16384.0)
    assert non_exact == pytest.approx(512.0)


def test_fortran_reduced_direct_tail_explicit_structured_pc_rejection_is_fast(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "global")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_xblock",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PC_MAX_MB", "0")
    messages: list[str] = []

    with pytest.raises(RuntimeError, match="explicitly requested but not selected"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="fortran_reduced_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
            emit=lambda _level, msg: messages.append(msg),
        )

    assert not any("explicit_sparse: factorization start" in msg for msg in messages)


def test_fortran_reduced_direct_tail_required_pc_forces_global_backend(monkeypatch) -> None:
    profile_policies_module._DIRECT_TAIL_STRUCTURED_PC_CACHE.clear()
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_CONSTRAINT_TAIL", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_PRECONDITIONER",
        "active_fortran_v3_reduced_lu",
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_PC_REQUIRED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "global"
    assert result.metadata["sparse_pc_backend_reason"] == "required_direct_tail_structured_pc"
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_global"
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_built"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_required"] is True
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb_auto"] is True
    expected_cap_mb = profile_policies_module._rhsmode1_fortran_reduced_direct_tail_pc_default_max_mb(
        requested_kind="active_fortran_v3_reduced_lu",
        active_size=int(result.metadata["sparse_pc_linear_size"]),
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_max_mb"] == pytest.approx(
        expected_cap_mb
    )
    assert (
        result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_requested"]
        == "active_fortran_v3_reduced_lu"
    )
    assert result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_selected"] is True
    structured_metadata = result.metadata["sparse_pc_fortran_reduced_direct_tail_structured_pc_metadata"]
    assert structured_metadata["kind"] == "active_fortran_v3_reduced_lu"
    assert structured_metadata["reason"] == "complete"
    pc_metadata = structured_metadata["metadata"]
    assert pc_metadata["factor_kind"] == "lu"
    assert pc_metadata["requires_preflight"] is False
    assert pc_metadata["max_factor_nbytes"] == int(expected_cap_mb * 1024.0 * 1024.0)
    assert pc_metadata["permc_spec_requested"] == "AUTO"
    assert not any("using x-block backend instead of monolithic CSR factor" in msg for msg in messages)


def test_sparse_pc_gmres_stagnation_guard_aborts_mocked_krylov(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_ABORT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_MIN_ITER", "3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_WINDOW", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_STAGNATION_REL_IMPROVEMENT", "0")

    def _stagnating_gmres(*, b, progress_callback, **_kwargs):
        for iteration in range(1, 5):
            progress_callback(iteration, 1.0)
        return np.zeros_like(np.asarray(b, dtype=np.float64)), float(np.linalg.norm(b)), [1.0] * 4

    monkeypatch.setattr(profile_solve_module, "gmres_solve_with_history_scipy", _stagnating_gmres)

    with pytest.raises(RuntimeError, match="sparse_pc_gmres stagnation detected"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="sparse_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )


def test_fortran_reduced_pc_gmres_xblock_backend_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    messages: list[str] = []

    def _forbidden_global_pattern(*_args, **_kwargs):
        raise AssertionError("x-block backend must not build the monolithic Fortran-reduced pattern")

    monkeypatch.setattr(
        profile_solve_module,
        "v3_full_system_fortran_reduced_preconditioner_sparsity_pattern",
        _forbidden_global_pattern,
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_preconditioner_operator"] == "fortran_reduced_xblock"
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_xblock_no_global_pattern"
    assert result.metadata["sparse_pattern_nnz"] == 0
    assert result.metadata["sparse_pc_xblock_moment_schur_enabled"] is False
    assert result.metadata["sparse_pc_xblock_global_coupling_enabled"] is False
    assert any("using x-block backend instead of monolithic CSR factor" in msg for msg in messages)
    assert any("fortran_reduced_pc_gmres xblock complete" in msg for msg in messages)


def test_fortran_reduced_pc_gmres_xblock_backend_accepts_lgmres(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_KRYLOV", "lgmres")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_xblock_krylov_method"] == "lgmres"


def test_fortran_reduced_pc_gmres_xblock_backend_moment_schur_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NZETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 4
    nml.group("resolutionParameters")["NX"] = 3
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_PC_BACKEND", "xblock")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_XBLOCK_MOMENT_SCHUR", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["sparse_pc_backend"] == "xblock"
    assert result.metadata["sparse_pc_xblock_moment_schur_enabled"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_built"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_used"] is True
    assert result.metadata["sparse_pc_xblock_moment_schur_mode"] == "constraint1_moment_schur"
    assert result.metadata["sparse_pc_xblock_moment_schur_extra_size"] == 4
    assert result.metadata["sparse_pc_xblock_moment_schur_rank"] == 4
    assert result.metadata["sparse_pc_xblock_moment_schur_base_applies"] >= (
        2 * result.metadata["sparse_pc_xblock_moment_schur_applies"]
    )
    assert any("fortran_reduced_pc_gmres xblock constraint1 moment-Schur" in msg for msg in messages)


def test_sparse_pc_gmres_active_dof_reduces_truncated_pas_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 8
    nml.group("resolutionParameters")["NL"] = 4
    nml.group("resolutionParameters")["NX"] = 4
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    nml.group("physicsParameters")["ER"] = 0.1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert result.metadata is not None
    assert result.metadata["sparse_pc_active_dof"] is True
    assert result.metadata["sparse_pattern_scope"] == "active_dof"
    assert result.metadata["sparse_pc_linear_size"] < result.metadata["sparse_pc_full_size"]
    active_idx = transport_active_dof_indices(result.op)
    inactive_idx = np.setdiff1d(np.arange(int(result.op.total_size), dtype=np.int32), active_idx)
    assert np.allclose(np.asarray(result.x)[inactive_idx], 0.0)
    residual = result.rhs[active_idx] - apply_v3_full_system_operator(result.op, result.x)[active_idx]
    target = 1.0e-8 * float(jnp.linalg.norm(result.rhs[active_idx]))
    assert float(jnp.linalg.norm(residual)) <= target
    assert any("active-DOF reduction enabled" in msg for msg in messages)


def test_fortran_reduced_pc_auto_uses_active_dof_for_truncated_modes(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    nml.group("resolutionParameters")["NTHETA"] = 5
    nml.group("resolutionParameters")["NXI"] = 8
    nml.group("resolutionParameters")["NL"] = 4
    nml.group("resolutionParameters")["NX"] = 4
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    nml.group("physicsParameters")["ER"] = 0.1
    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_ACTIVE_DOF", raising=False)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="fortran_reduced_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "fortran_reduced_pc_gmres"
    assert result.metadata["sparse_pc_active_dof"] is True
    assert result.metadata["sparse_pattern_scope"] == "fortran_reduced_active_dof"
    assert result.metadata["sparse_pc_linear_size"] < result.metadata["sparse_pc_full_size"]


def test_xblock_sparse_pc_gmres_solve_method_solves_fp_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-2
    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["setup_s"] >= 0.0
    assert result.metadata["solve_s"] >= 0.0
    assert result.metadata["elapsed_s"] >= result.metadata["setup_s"]
    assert result.metadata["xblock_initial_seed_used"] in {True, False}
    assert result.metadata["xblock_initial_seed_residual_norm"] >= 0.0
    assert any("initial x-block seed" in msg for msg in messages)
    assert any("xblock_sparse_pc_gmres complete" in msg for msg in messages)


def test_xblock_sparse_pc_gmres_initial_seed_can_be_disabled(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_INITIAL_SEED", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_MINRES_STEPS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_COARSE", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-2
    assert result.metadata["xblock_initial_seed_used"] is False
    assert result.metadata["xblock_initial_seed_residual_norm"] is None
    assert result.metadata["xblock_initial_seed_residual_ratio"] is None
    assert result.metadata["xblock_post_minres_steps_requested"] == 2
    assert result.metadata["xblock_post_minres_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_steps_requested"] == 1
    assert result.metadata["xblock_post_coarse_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_direction_count"] == 0


def test_xblock_sparse_pc_post_residual_equation_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres_jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_MAX_DIRECTIONS", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_FSAVG_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_POST_RESIDUAL_EQUATION_ANGULAR_LMAX", "-1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-12,
        maxiter=2,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.metadata["xblock_post_residual_equation_steps_requested"] == 1
    assert result.metadata["xblock_post_residual_equation_residual_before"] is not None
    assert result.metadata["xblock_post_residual_equation_residual_after"] is not None
    assert (
        result.metadata["xblock_post_residual_equation_residual_after"]
        < result.metadata["xblock_post_residual_equation_residual_before"]
    )
    assert result.metadata["xblock_post_residual_equation_direction_count"] > 0
    assert any("post-residual-equation improved residual" in msg for msg in messages)


def test_xblock_sparse_pc_constraint1_moment_schur_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_SEED", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_moment_schur_enabled"] is True
    assert result.metadata["xblock_moment_schur_built"] is True
    assert result.metadata["xblock_moment_schur_mode"] == "constraint1_moment_schur"
    assert result.metadata["xblock_moment_schur_extra_size"] == 4
    assert result.metadata["xblock_moment_schur_rank"] == 4
    assert result.metadata["xblock_moment_schur_device_resident"] is True
    assert result.metadata["xblock_moment_schur_used"] is True
    assert result.metadata["xblock_moment_schur_reason"] == "built"
    assert result.metadata["xblock_moment_schur_base_applies"] == 2 * result.metadata["xblock_moment_schur_applies"]
    assert result.metadata["xblock_moment_schur_seed_residual_norm"] is not None
    assert any("constraint1 moment-Schur built" in msg for msg in messages)


def test_xblock_sparse_pc_constraint1_moment_schur_probe_fails_closed(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_MOMENT_SCHUR_MIN_IMPROVEMENT", "1.0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_moment_schur_enabled"] is True
    assert result.metadata["xblock_moment_schur_built"] is True
    assert result.metadata["xblock_moment_schur_used"] is False
    assert result.metadata["xblock_moment_schur_reason"] == "probe_not_reduced"
    assert result.metadata["xblock_moment_schur_probe_residual_before"] is not None
    assert result.metadata["xblock_moment_schur_probe_residual_after"] is not None
    assert result.metadata["xblock_moment_schur_probe_improvement_ratio"] is not None
    assert result.metadata["xblock_moment_schur_seed_used"] is False
    assert any("constraint1 moment-Schur rejected" in msg for msg in messages)


def test_xblock_sparse_pc_preflight_required_rejects_weak_seed(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_MIN_IMPROVEMENT", "0.9")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PREFLIGHT_REQUIRED", "1")

    with pytest.raises(RuntimeError, match="preflight gate failed"):
        solve_v3_full_system_linear_gmres(
            nml=nml,
            solve_method="xblock_sparse_pc_gmres",
            tol=1.0e-8,
            maxiter=80,
        )


def test_xblock_sparse_pc_assembled_operator_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_matrix_nnz"] > 0
    assert result.metadata["xblock_assembled_operator_error"] is None
    assert any("assembled operator built" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_can_use_device_csr(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_enabled"] is True
    assert result.metadata["xblock_assembled_operator_device_required"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_assembled_operator_device_nnz"] == result.metadata["xblock_assembled_operator_matrix_nnz"]
    assert result.metadata["xblock_assembled_operator_device_csr_nbytes_estimate"] > 0
    assert result.metadata["xblock_assembled_operator_device_error"] is None
    assert any("assembled operator built location=device" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_row_equilibration_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_ROW_EQUILIBRATE_NORM", "linf")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_enabled"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_row_equilibration_norm"] == "linf"
    assert result.metadata["xblock_assembled_operator_row_equilibration_setup_s"] >= 0.0
    assert result.metadata["xblock_assembled_operator_row_equilibration_scale_min"] > 0.0
    assert result.metadata["xblock_assembled_operator_row_equilibration_scale_max"] > 0.0
    assert any("assembled row equilibration built" in msg for msg in messages)
    assert any("using row-equilibrated assembled operator" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_row_col_equilibration_maps_solution(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_COL_EQUILIBRATE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_row_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_enabled"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_built"] is True
    assert result.metadata["xblock_assembled_operator_col_equilibration_scale_min"] > 0.0
    assert result.metadata["xblock_assembled_operator_col_equilibration_scale_max"] > 0.0
    assert any("assembled column equilibration built" in msg for msg in messages)
    assert any("using row/column-equilibrated assembled operator" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_records_budget_rejection(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is False
    assert "MemoryError" in str(result.metadata["xblock_assembled_operator_error"])
    assert result.metadata["xblock_assembled_operator_preflight_rejected"] is True
    assert result.metadata["xblock_assembled_operator_preflight_pattern_nnz_estimate"] > 0
    assert any("assembled operator disabled after build failure" in msg for msg in messages)


def test_xblock_sparse_pc_active_dof_opt_in_records_reduced_size(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_linear_size"] < result.metadata["xblock_full_size"]
    assert result.gmres.x.shape == result.rhs.shape


def test_xblock_sparse_pc_probe_coarse_uses_active_projected_directions(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_FSAVG_LMAX", "2")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_probe_coarse_steps_requested"] == 1
    assert result.metadata["xblock_probe_coarse_steps_accepted"] == 1
    assert result.metadata["xblock_probe_coarse_direction_count"] == 8
    assert result.metadata["xblock_probe_coarse_angular_lmax"] == -1
    assert result.metadata["xblock_probe_coarse_seed_initialized"] is True
    assert result.metadata["xblock_probe_coarse_residual_after"] < result.metadata["xblock_probe_coarse_residual_before"]
    assert any("probe-coarse improved seed residual" in msg for msg in messages)


def test_xblock_post_coarse_directions_can_include_angular_modes() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    residual = jnp.ones((op.total_size,), dtype=jnp.float64)

    directions = build_rhs1_xblock_post_coarse_directions(
        op=op,
        residual=residual,
        preconditioner=lambda v: jnp.asarray(v, dtype=jnp.float64),
        include_raw=False,
        fsavg_lmax=0,
        angular_lmax=1,
        max_extra_units=0,
        max_directions=16,
    )

    names = tuple(name for name, _direction in directions)
    assert any(name.startswith("fsavg_l") for name in names)
    assert any(name.startswith("angular_") for name in names)
    assert len(directions) <= 16


def test_xblock_post_coarse_directions_can_include_residual_weighted_angular_modes() -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    theta = jnp.arange(int(op.n_theta), dtype=jnp.float64)
    zeta = jnp.arange(int(op.n_zeta), dtype=jnp.float64)
    pattern = jnp.cos(2.0 * jnp.pi * theta[:, None] / float(op.n_theta)) + 0.25 * jnp.sin(
        2.0 * jnp.pi * zeta[None, :] / float(op.n_zeta)
    )
    f_res = jnp.zeros(op.fblock.f_shape, dtype=jnp.float64)
    f_res = f_res.at[0, :, 0, :, :].set(pattern[None, :, :])
    residual = jnp.concatenate(
        [f_res.reshape((-1,)), jnp.zeros((int(op.total_size) - int(op.f_size),), dtype=jnp.float64)]
    )

    directions = build_rhs1_xblock_post_coarse_directions(
        op=op,
        residual=residual,
        preconditioner=lambda v: jnp.asarray(v, dtype=jnp.float64),
        include_raw=False,
        fsavg_lmax=0,
        angular_lmax=0,
        include_angular_residual=True,
        max_extra_units=0,
        max_directions=16,
    )

    names = tuple(name for name, _direction in directions)
    assert any(name.startswith("angular_residual_") for name in names)
    assert len(directions) <= 16


def test_device_subspace_residual_equation_reuses_cached_operator_basis() -> None:
    operator_matrix = jnp.asarray([[1.0, 1.0], [0.0, 1.0]], dtype=jnp.float64)
    rhs = jnp.asarray([0.0, 1.0], dtype=jnp.float64)
    cached_basis = jnp.asarray([[1.0], [0.0]], dtype=jnp.float64)
    cached_action = operator_matrix @ cached_basis

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    def direction_builder(_residual):
        return (("missing_mode", jnp.asarray([0.0, 1.0], dtype=jnp.float64)),)

    x, residual, history, counts, names = apply_device_subspace_residual_equation_correction(
        matvec=matvec,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        direction_builder=direction_builder,
        steps=1,
        max_directions=2,
        cached_basis=cached_basis,
        cached_operator_on_basis=cached_action,
        cached_labels=("flat_x0",),
        rcond=0.0,
    )

    np.testing.assert_allclose(matvec(x), rhs, rtol=1.0e-12, atol=1.0e-12)
    assert float(jnp.linalg.norm(residual)) < 1.0e-12
    assert history[-1] < 1.0e-12
    assert counts == (2,)
    assert names == ("cached_qi:flat_x0", "missing_mode")


def test_device_subspace_residual_equation_fails_closed_without_improvement() -> None:
    operator_matrix = jnp.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=jnp.float64)
    rhs = jnp.asarray([1.0, -1.0], dtype=jnp.float64)

    def matvec(x):
        return operator_matrix @ jnp.asarray(x, dtype=jnp.float64)

    def direction_builder(_residual):
        return (("zero_mode", jnp.zeros_like(rhs)),)

    x, residual, history, counts, names = apply_device_subspace_residual_equation_correction(
        matvec=matvec,
        rhs=rhs,
        x0=jnp.zeros_like(rhs),
        direction_builder=direction_builder,
        steps=1,
        max_directions=4,
    )

    np.testing.assert_allclose(x, jnp.zeros_like(rhs), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(residual, rhs, rtol=1.0e-12, atol=1.0e-12)
    assert len(history) == 1
    assert history[0] == pytest.approx(float(jnp.linalg.norm(rhs)))
    assert counts == ()
    assert names == ()


def test_xblock_sparse_pc_probe_coarse_records_angular_mode_usage(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_FSAVG_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_PROBE_COARSE_ANGULAR_RESIDUAL", "1")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_probe_coarse_angular_lmax"] == 1
    assert result.metadata["xblock_probe_coarse_angular_residual"] is True
    assert any(
        str(name).startswith("angular_")
        for name in result.metadata["xblock_probe_coarse_direction_names"]
    )


def test_xblock_sparse_pc_lower_fill_local_policy_is_wired(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL", "force")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_FACTOR", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ILU_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LOWER_FILL_ROW_NNZ_MAX", "16")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-4,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-4
    assert result.metadata["xblock_lower_fill_mode"] == "force"
    assert result.metadata["xblock_lower_fill_requested"] is True
    assert result.metadata["xblock_lower_fill_ignored_env"] is False
    assert any("lower-fill local factor" in msg for msg in messages)


def test_xblock_side_probe_switch_preserves_physical_seed_for_right_pc(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("physicsParameters")["includeXDotTerm"] = False
    nml.group("physicsParameters")["includeElectricFieldTermInXiDot"] = False
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_SIDE_PROBE_RESTART", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_LGMRES_RESCUE", "0")
    monkeypatch.setattr(
        rhs1_xblock_policy_module,
        "rhs1_xblock_side_probe_should_switch",
        lambda *, residual_ratio, switch_ratio_env_value: True,
    )
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_side_probe_used"] is True
    assert result.metadata["xblock_side_probe_switched"] is True
    assert result.metadata["xblock_side_probe_initial_side"] == "left"
    assert result.metadata["xblock_side_probe_selected_side"] == "right"
    assert result.metadata["xblock_side_probe_physical_seed_preserved_after_switch"] is True
    assert result.metadata["xblock_side_probe_seed_used"] is True
    assert any("preserved_physical_seed=1" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_active_dof_uses_sliced_budget(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active_idx = transport_active_dof_indices(op)
    full_summary = estimate_v3_full_system_conservative_sparsity_summary(op)
    full_csr_nbytes = int(full_summary.nnz * 12 + (full_summary.shape[0] + 1) * 4)
    full_sliced_pattern = v3_full_system_conservative_sparsity_pattern(op)[active_idx, :][:, active_idx].tocsr()
    active_pattern = v3_full_system_conservative_sparsity_pattern_for_indices(op, active_idx)
    assert (active_pattern != full_sliced_pattern).nnz == 0
    active_csr_nbytes = int(active_pattern.nnz * 12 + (active_pattern.shape[0] + 1) * 4)
    assert active_csr_nbytes < full_csr_nbytes

    cap_mb = 1.2 * active_csr_nbytes / 1.0e6
    assert full_csr_nbytes > cap_mb * 1.0e6
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", f"{cap_mb:.6f}")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_assembled_operator_enabled"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_active_dof"] is True
    assert result.metadata["xblock_assembled_operator_preflight_scope"] == "active_dof"
    assert result.metadata["xblock_assembled_operator_preflight_active_csr_nbytes_estimate"] <= cap_mb * 1.0e6
    assert result.metadata["xblock_assembled_operator_preflight_full_csr_nbytes_estimate"] > cap_mb * 1.0e6
    assert result.metadata["xblock_assembled_operator_error"] is None


@pytest.mark.parametrize(
    ("method", "expected_solver_kind"),
    [
        ("gmres", "xblock_sparse_pc_gmres"),
        ("lgmres", "xblock_sparse_pc_lgmres"),
    ],
)
def test_xblock_sparse_pc_gmres_opt_in_krylov_method_records_realized_solver(
    monkeypatch,
    method: str,
    expected_solver_kind: str,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", method)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == expected_solver_kind
    assert result.metadata["krylov_method"] == method
    assert result.metadata["candidate_krylov_method"] == method
    assert result.metadata["fallback_from_krylov_method"] is None
    assert result.metadata["matvecs"] >= result.metadata["candidate_matvecs"]

def test_xblock_sparse_pc_device_cycle_jit_reports_internal_iterations(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_DEVICE_JIT_MODE", "cycle")

    monkeypatch.setattr(
        profile_solve_module,
        "fgmres_cycle_jit_solve_with_residual",
        _fast_device_cycle_krylov_result,
    )

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_fgmres_jax"
    assert result.metadata["iterations"] == 80
    assert result.metadata["matvecs"] >= 82
    assert result.metadata["device_cycle_estimated_matvecs"] == result.metadata["matvecs"]
    assert result.metadata["python_matvecs"] < result.metadata["matvecs"]
    assert result.metadata["candidate_iterations"] == 80
    assert result.metadata["candidate_matvecs"] == result.metadata["matvecs"]


def test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "csr")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "100000")
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e3,
        maxiter=1,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["sparse_pc_xblock_jax_factor_format"] == "csr"
    assert result.metadata["xblock_moment_schur_default_blocked_by_compact_factors"] is True
    assert result.metadata["xblock_moment_schur_built"] is False
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert any("xblock_sparse_csr: built compact JAX factors" in msg for msg in messages)
    assert any("moment-Schur default disabled for compact JAX factors" in msg for msg in messages)


def test_xblock_sparse_pc_device_krylov_can_use_compact_diagonal_factor_apply(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "csr")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_APPLY", "diagonal")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "100000")
    messages: list[str] = []

    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["sparse_pc_xblock_jax_factor_format"] == "csr"
    assert result.metadata["sparse_pc_xblock_jax_factor_apply"] == "diagonal"
    assert any("xblock_sparse_csr: using approximate compact JAX factor apply mode=diagonal" in msg for msg in messages)


def test_xblock_sparse_pc_device_krylov_with_device_assembled_operator_is_transfer_free(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_RESTART", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_FGMRES_BLOCK_BETWEEN_CYCLES", "1")
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual", _fast_device_krylov_result)
    monkeypatch.setattr(profile_solve_module, "fgmres_solve_with_residual_jit", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_fgmres_jax"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_block_between_cycles"] is True


def test_xblock_sparse_pc_device_bicgstab_uses_device_assembled_operator(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "bicgstab-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setattr(profile_solve_module, "bicgstab_solve_with_residual", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_bicgstab_jax"
    assert result.metadata["xblock_device_krylov_method"] == "bicgstab_jax"
    assert result.metadata["xblock_device_bicgstab_enabled"] is True
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_bicgstab_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False
    assert result.metadata["candidate_iterations"] >= 1
    assert result.metadata["xblock_estimated_bicgstab_work_nbytes"] < result.metadata["xblock_estimated_gmres_basis_nbytes"]


def test_xblock_sparse_pc_device_tfqmr_uses_device_assembled_operator(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "tfqmr-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_DEVICE_REQUIRED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_CSR_MAX_MB", "64")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ASSEMBLED_OPERATOR_MAX_COLORS", "4096")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_SPARSE_PC_GMRES_MAXITER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TFQMR_REPLACE_INTERVAL", "2")
    monkeypatch.setattr(profile_solve_module, "tfqmr_solve_with_residual", _fast_device_krylov_result)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-2,
        maxiter=2,
    )

    assert np.isfinite(float(result.residual_norm))
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_tfqmr_jax"
    assert result.metadata["xblock_device_krylov_method"] == "tfqmr_jax"
    assert result.metadata["xblock_device_tfqmr_enabled"] is True
    assert result.metadata["xblock_device_tfqmr_replacement_interval"] == 2
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_assembled_operator_built"] is True
    assert result.metadata["xblock_assembled_operator_device_resident"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is True
    assert result.metadata["xblock_device_tfqmr_host_transfer_free"] is True
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False
    assert result.metadata["candidate_iterations"] >= 1
    assert result.metadata["xblock_estimated_tfqmr_work_nbytes"] < result.metadata["xblock_estimated_gmres_basis_nbytes"]


def test_xblock_sparse_pc_candidate_falls_back_to_gmres_when_residual_is_bad(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "bicgstab")

    def fake_bicgstab(*, b, **_kwargs):
        return np.zeros(int(b.size), dtype=np.float64), float("inf"), [float("inf")]

    monkeypatch.setattr(profile_solve_module, "bicgstab_solve_with_history_scipy", fake_bicgstab)

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["krylov_method"] == "gmres"
    assert result.metadata["candidate_krylov_method"] == "bicgstab"
    assert result.metadata["fallback_from_krylov_method"] == "bicgstab"
    assert result.metadata["candidate_residual_norm"] > 1.0e-8
    assert result.metadata["candidate_iterations"] == 1


def test_sparse_lsmr_solve_method_solves_tiny_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="sparse_lsmr",
        tol=1.0e-10,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert result.x.shape == (result.op.total_size,)
    assert float(result.residual_norm) < 1.0e-8
    assert any("sparse_lsmr complete" in msg for msg in messages)


def test_petsc_compat_solve_method_labels_minimum_norm_branch(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="petsc_compat",
        tol=1.0e-10,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "sparse_lsmr"
    assert result.metadata["petsc_compat_requested"] is True
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["acceptance_criterion"] in {
        "true_residual",
        "petsc_compatible_minimum_norm",
    }


def test_write_output_preserves_explicit_sparse_host_solve_method(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method="sparse_host",
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "sparse_host"
    assert trace["converged"] is True


def test_write_output_preserves_sparse_pc_gmres_solve_method(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method="sparse_pc_gmres",
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "sparse_pc_gmres"
    assert trace["converged"] is True
    assert trace["setup_s"] is not None
    assert trace["solve_s"] is not None
    assert trace["metadata"]["solver_metadata"]["sparse_pattern_nnz"] > 0


def test_write_output_auto_tokamak_fp_noer_policy_uses_xblock_sparse_pc(monkeypatch, tmp_path: Path) -> None:
    here = Path(__file__).parent
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_ASSEMBLED_HOST", "1")
    monkeypatch.setattr(io_module, "rhs1_tokamak_fp_noer_sparse_pc_auto_allowed", lambda **_kwargs: True)
    trace_path = tmp_path / "solver_trace.json"

    write_sfincs_jax_output_h5(
        input_namelist=here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist",
        output_path=tmp_path / "sfincsOutput.h5",
        compute_solution=True,
        solve_method=None,
        solver_trace_path=trace_path,
        verbose=False,
    )

    trace = json.loads(trace_path.read_text())
    assert trace["solve_method"] == "xblock_sparse_pc_gmres"
    assert trace["converged"] is True
    assert trace["metadata"]["solver_metadata"]["solver_kind"] == "xblock_sparse_pc_gmres"
    assert trace["metadata"]["solver_metadata"]["sparse_pc_xblock_preconditioner_xi"] == 1
