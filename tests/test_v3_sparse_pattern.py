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
    assert result.metadata["xblock_global_coupling_applies"] > 0
    assert result.metadata["xblock_global_coupling_coarse_applies"] == result.metadata["xblock_global_coupling_applies"]
    assert any("global-coupling built" in msg for msg in messages)


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
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
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
        x0=jnp.zeros((op.total_size,)),
        emit=lambda _level, msg: messages.append(msg),
    )

    assert float(result.residual_norm) < 1.0e-8
    assert result.metadata["xblock_active_dof"] is True
    assert result.metadata["xblock_probe_coarse_steps_requested"] == 1
    assert result.metadata["xblock_probe_coarse_steps_accepted"] == 1
    assert result.metadata["xblock_probe_coarse_direction_count"] == 8
    assert result.metadata["xblock_probe_coarse_residual_after"] < result.metadata["xblock_probe_coarse_residual_before"]
    assert any("probe-coarse improved seed residual" in msg for msg in messages)


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
    expected_method = "fgmres_jax" if method == "fgmres" else "gmres_jax"
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
