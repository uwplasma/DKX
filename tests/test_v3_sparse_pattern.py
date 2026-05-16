from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import sfincs_jax.io as io_module
import sfincs_jax.v3_driver as v3_driver_module
from sfincs_jax.explicit_sparse import build_operator_from_pattern
from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij
from sfincs_jax.rhs1_xblock_policy import resolve_rhs1_xblock_sparse_pc_policy
from sfincs_jax.v3_sparse_pattern import (
    estimate_v3_full_system_conservative_sparsity_summary,
    summarize_v3_sparse_pattern,
    v3_full_system_conservative_sparsity_pattern,
    v3_full_system_conservative_sparsity_pattern_for_indices,
)
from sfincs_jax.v3_driver import (
    _rhs1_xblock_gmres_restart,
    _rhs1_xblock_precondition_side,
    _rhs1_xblock_post_coarse_directions,
    _triangular_solve_lower_csr_rows,
    _triangular_solve_upper_csr_rows,
    solve_v3_full_system_linear_gmres,
)
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist


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
    assert result.metadata["sparse_pc_linear_size"] < result.metadata["sparse_pc_full_size"]
    active_idx = v3_driver_module._transport_active_dof_indices(result.op)
    inactive_idx = np.setdiff1d(np.arange(int(result.op.total_size), dtype=np.int32), active_idx)
    assert np.allclose(np.asarray(result.x)[inactive_idx], 0.0)
    residual = result.rhs[active_idx] - apply_v3_full_system_operator(result.op, result.x)[active_idx]
    target = 1.0e-8 * float(jnp.linalg.norm(result.rhs[active_idx]))
    assert float(jnp.linalg.norm(residual)) <= target
    assert any("active-DOF reduction enabled" in msg for msg in messages)


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
    assert float(result.residual_norm) < 1.0e-8
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

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_initial_seed_used"] is False
    assert result.metadata["xblock_initial_seed_residual_norm"] is None
    assert result.metadata["xblock_initial_seed_residual_ratio"] is None
    assert result.metadata["xblock_post_minres_steps_requested"] == 2
    assert result.metadata["xblock_post_minres_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_steps_requested"] == 1
    assert result.metadata["xblock_post_coarse_steps_accepted"] == 0
    assert result.metadata["xblock_post_coarse_direction_count"] == 0


def test_xblock_sparse_pc_two_level_preconditioner_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "10")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX", "2")
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
    assert result.metadata["xblock_two_level_enabled"] is True
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_two_level_mode"] == "additive"
    assert 1 <= result.metadata["xblock_two_level_rank"] <= result.metadata["xblock_two_level_basis_size"] <= 10
    assert result.metadata["xblock_two_level_applies"] > 0
    assert result.metadata["xblock_two_level_coarse_applies"] == result.metadata["xblock_two_level_applies"]
    assert any("two-level coarse built" in msg for msg in messages)


def test_xblock_sparse_pc_global_coupling_preconditioner_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX", "1")
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
    assert result.metadata["xblock_global_coupling_enabled"] is True
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_mode"] == "additive"
    assert 1 <= result.metadata["xblock_global_coupling_rank"] <= result.metadata["xblock_global_coupling_basis_size"] <= 12
    assert result.metadata["xblock_global_coupling_smoother"] == "base"
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert result.metadata["xblock_global_coupling_applies"] > 0
    assert result.metadata["xblock_global_coupling_coarse_applies"] == result.metadata["xblock_global_coupling_applies"]
    assert any("global-coupling built" in msg for msg in messages)


def test_xblock_sparse_pc_global_coupling_setup_budget_uses_partial_basis(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "12")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_ANGULAR_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_SETUP_MAX_S", "1e-12")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_smoother"] == "base"
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == pytest.approx(1.0e-12)
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is True
    assert result.metadata["xblock_global_coupling_basis_size"] < result.metadata["xblock_global_coupling_load_basis_size"]
    assert any("global-coupling setup budget reached" in msg for msg in messages)


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
    assert result.metadata["xblock_moment_schur_base_applies"] == 2 * result.metadata["xblock_moment_schur_applies"]
    assert result.metadata["xblock_moment_schur_seed_residual_norm"] is not None
    assert any("constraint1 moment-Schur built" in msg for msg in messages)


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

    directions = _rhs1_xblock_post_coarse_directions(
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

    directions = _rhs1_xblock_post_coarse_directions(
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


def test_xblock_sparse_pc_qi_coarse_seed_records_residual_reduction(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "10")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_coarse_seed_enabled"] is True
    assert result.metadata["xblock_qi_coarse_seed_used"] is True
    assert result.metadata["xblock_qi_coarse_seed_rank"] > 0
    assert result.metadata["xblock_qi_coarse_seed_residual_after"] < result.metadata[
        "xblock_qi_coarse_seed_residual_before"
    ]
    assert result.metadata["xblock_qi_coarse_seed_basis"] == "enriched"
    assert result.metadata["xblock_qi_coarse_seed_candidate_count"] <= 24
    assert result.metadata["xblock_qi_coarse_seed_max_candidates"] == 24
    assert result.metadata["xblock_qi_coarse_seed_max_angular_mode"] == 2
    assert "global" in result.metadata["xblock_qi_coarse_seed_labels"]
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is False
    assert any("QI coarse seed improved residual" in msg for msg in messages)


def test_xblock_sparse_pc_qi_galerkin_preconditioner_fails_closed_when_probe_worsens(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_MODE", "multiplicative")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER_DAMPINGS", "1.0")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_used"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_reason"] == "probe_not_reduced"
    assert result.metadata["xblock_qi_galerkin_preconditioner_mode"] == "multiplicative"
    assert result.metadata["xblock_qi_galerkin_preconditioner_basis_reused_from_seed"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_rank"] > 0
    assert result.metadata["xblock_qi_galerkin_preconditioner_candidate_count"] <= 24
    assert result.metadata["xblock_qi_galerkin_preconditioner_coarse_operator_shape"][0] == result.metadata[
        "xblock_qi_galerkin_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_galerkin_preconditioner_coarse_applies"] == 0
    assert result.metadata["xblock_qi_galerkin_preconditioner_base_applies"] == 0
    assert np.isfinite(float(result.metadata["xblock_qi_galerkin_preconditioner_residual_before"]))
    assert np.isfinite(float(result.metadata["xblock_qi_galerkin_preconditioner_residual_after"]))
    assert result.metadata["xblock_qi_galerkin_preconditioner_probe_reduced"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_selected_index"] is None
    assert result.metadata["xblock_qi_galerkin_preconditioner_probe_candidates"]
    assert all(
        candidate["residual_norm"] >= result.metadata["xblock_qi_galerkin_preconditioner_residual_before"]
        for candidate in result.metadata["xblock_qi_galerkin_preconditioner_probe_candidates"]
    )
    assert any("QI Galerkin preconditioner built" in msg for msg in messages)


def test_xblock_sparse_pc_qi_two_level_preconditioner_fails_closed_when_probe_worsens(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_used"] is False
    assert result.metadata["xblock_qi_two_level_preconditioner_reason"] == "residual_not_reduced"
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] > 0
    assert result.metadata["xblock_qi_two_level_preconditioner_candidate_count"] <= 24
    assert result.metadata["xblock_qi_two_level_preconditioner_coarse_solver"] == "action_lstsq"
    assert result.metadata["xblock_qi_two_level_preconditioner_coarse_operator_shape"][0] == result.metadata[
        "xblock_qi_two_level_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_two_level_preconditioner_operator_on_basis_shape"][1] == result.metadata[
        "xblock_qi_two_level_preconditioner_rank"
    ]
    assert result.metadata["xblock_qi_two_level_preconditioner_probe_candidates"]
    assert result.metadata["xblock_qi_two_level_preconditioner_selected_index"] is not None
    assert result.metadata["xblock_qi_two_level_preconditioner_improvement_ratio"] >= 0.95
    assert result.metadata["xblock_qi_two_level_preconditioner_applies"] == 0
    assert result.metadata["xblock_qi_two_level_preconditioner_local_applies"] >= 1
    assert any("QI two-level preconditioner rejected" in msg for msg in messages)


def test_xblock_sparse_pc_qi_two_level_residual_augmentation_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_RESIDUAL_AUGMENT_MAX_EXTRA", "2")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "24")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augmented"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_rank_before_augmentation"] > 0
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_max_extra"] == 2
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_steps"] == 1
    assert result.metadata["xblock_qi_two_level_preconditioner_residual_augment_include_residuals"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_augmentation_labels"]
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] >= result.metadata[
        "xblock_qi_two_level_preconditioner_rank_before_augmentation"
    ]


def test_xblock_sparse_pc_qi_two_level_smoothed_load_basis_records_metadata(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS", "1"
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_BASIS_COMBINE", "0"
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_RANK", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_FSAVG_LMAX", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_TWO_LEVEL_PRECONDITIONER_SMOOTHED_LOAD_ANGULAR_LMAX", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_BASIS", "enriched")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_RANK", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_COARSE_SEED_MAX_CANDIDATES", "12")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=80,
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_qi_two_level_preconditioner_built"] is True
    assert result.metadata["xblock_qi_two_level_preconditioner_smoothed_load_basis"] is True
    smoothed = result.metadata["xblock_qi_two_level_preconditioner_smoothed_load_metadata"]
    assert smoothed["smoothed_candidate_count"] > 0
    assert smoothed["rank"] == result.metadata["xblock_qi_two_level_preconditioner_rank"]
    assert result.metadata["xblock_qi_two_level_preconditioner_rank"] <= 4


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
        v3_driver_module._rhs1_xblock_policy,
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


def test_xblock_sparse_pc_two_level_active_dof_projects_coarse_basis(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_ACTIVE_DOF", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_FSAVG_LMAX", "2")
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
    assert result.metadata["xblock_two_level_enabled"] is True
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_two_level_active_projected"] is True
    assert result.metadata["xblock_two_level_expected_size"] == result.metadata["xblock_linear_size"]
    assert result.metadata["xblock_two_level_applies"] > 0
    assert any("two-level coarse built" in msg for msg in messages)


def test_xblock_sparse_pc_assembled_operator_active_dof_uses_sliced_budget(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    nml.group("otherNumericalParameters")["NXI_FOR_X_OPTION"] = 1

    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    active_idx = v3_driver_module._transport_active_dof_indices(op)
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


@pytest.mark.parametrize(
    ("method", "expected_kind", "expected_metadata_key"),
    [
        ("fgmres", "xblock_sparse_pc_fgmres_jax", "xblock_device_fgmres_enabled"),
        ("gmres-jax", "xblock_sparse_pc_gmres_jax", "xblock_device_gmres_enabled"),
    ],
)
def test_xblock_sparse_pc_device_krylov_records_experimental_metadata(
    monkeypatch,
    method: str,
    expected_kind: str,
    expected_metadata_key: str,
) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", method)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "4")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    expected_method = {
        "fgmres": "fgmres_jax",
        "gmres-jax": "gmres_jax",
    }[method]
    assert result.metadata["solver_kind"] == expected_kind
    assert result.metadata["krylov_method"] == expected_method
    assert result.metadata["candidate_krylov_method"] == expected_method
    assert result.metadata["sparse_pc_xblock_jax_factors"] is True
    assert result.metadata["xblock_device_krylov_method"] == expected_method
    assert result.metadata[expected_metadata_key] is True
    assert result.metadata["xblock_device_fgmres_forced_jax_factors"] is True
    if method == "fgmres":
        assert result.metadata["precondition_side"] == "right"
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_device_resident"] is True
    assert result.metadata["xblock_global_coupling_coarse_solver"] == "qr"
    assert result.metadata["xblock_global_coupling_smoother"] == "identity"
    assert result.metadata["xblock_global_coupling_ridge"] == 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 180.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False
    assert len(result.metadata["xblock_global_coupling_singular_values"]) >= 1


def test_xblock_sparse_pc_device_host_fallback_records_non_autodiff_host_policy(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_DEVICE_HOST_FALLBACK", "force")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_QI_GALERKIN_PRECONDITIONER", "1")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-8,
        maxiter=40,
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["solver_kind"] == "xblock_sparse_pc_gmres"
    assert result.metadata["krylov_method"] == "gmres"
    assert result.metadata["sparse_pc_xblock_jax_factors"] is False
    assert result.metadata["xblock_device_krylov_method"] is None
    assert result.metadata["xblock_device_host_fallback_used"] is True
    assert result.metadata["xblock_device_host_fallback_reason"] == "forced"
    assert result.metadata["xblock_device_host_fallback_requested_method"] == "gmres_jax"
    assert result.metadata["xblock_device_host_fallback_effective_krylov_env_value"] == "auto"
    assert result.metadata["xblock_device_host_fallback_non_autodiff"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_enabled"] is True
    assert result.metadata["xblock_qi_galerkin_preconditioner_built"] is False
    assert result.metadata["xblock_qi_galerkin_preconditioner_reason"] == "disabled_by_device_host_fallback"
    assert any("using non-autodiff host x-block fallback" in msg for msg in messages)


def test_xblock_sparse_pc_device_krylov_can_use_compact_csr_factors(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_JAX_FACTOR_FORMAT", "csr")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_SPARSE_LU_MAX", "100000")
    messages: list[str] = []

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
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


def test_xblock_sparse_pc_device_global_coupling_can_use_normal_equations(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "gmres-jax")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_MAX_DIRECTIONS", "4")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_GLOBAL_COUPLING_DEVICE_SOLVER", "normal-equations")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["xblock_global_coupling_built"] is True
    assert result.metadata["xblock_global_coupling_device_resident"] is True
    assert result.metadata["xblock_global_coupling_coarse_solver"] == "normal_equations"
    assert result.metadata["xblock_global_coupling_smoother"] == "identity"
    assert result.metadata["xblock_global_coupling_ridge"] > 0.0
    assert result.metadata["xblock_global_coupling_setup_budget_s"] == 180.0
    assert result.metadata["xblock_global_coupling_setup_budget_reached"] is False


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


def test_xblock_sparse_pc_device_krylov_marks_host_two_level_transfer(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "fgmres")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_TWO_LEVEL_MAX_DIRECTIONS", "2")

    result = solve_v3_full_system_linear_gmres(
        nml=nml,
        solve_method="xblock_sparse_pc_gmres",
        tol=1.0e-3,
        maxiter=20,
    )

    assert float(result.residual_norm) < 1.0e-3
    assert result.metadata["xblock_device_krylov_method"] == "fgmres_jax"
    assert result.metadata["xblock_two_level_built"] is True
    assert result.metadata["xblock_device_krylov_host_transfer_free"] is False
    assert result.metadata["xblock_device_fgmres_host_transfer_free"] is False


def test_xblock_sparse_pc_candidate_falls_back_to_gmres_when_residual_is_bad(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_XBLOCK_PC_KRYLOV", "bicgstab")

    def fake_bicgstab(*, b, **_kwargs):
        return np.zeros(int(b.size), dtype=np.float64), float("inf"), [float("inf")]

    monkeypatch.setattr(v3_driver_module, "bicgstab_solve_with_history_scipy", fake_bicgstab)

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
