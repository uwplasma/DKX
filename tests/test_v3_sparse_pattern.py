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
from sfincs_jax.v3_sparse_pattern import summarize_v3_sparse_pattern, v3_full_system_conservative_sparsity_pattern
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
        use_dkes=False,
        include_xdot=True,
        include_electric_field_xi=True,
    )
    assert (side, auto_right) == ("right", True)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="",
        tokamak_fp_er_pc=True,
        use_dkes=True,
        include_xdot=False,
        include_electric_field_xi=False,
    )
    assert (side, auto_right) == ("left", False)

    side, auto_right = _rhs1_xblock_precondition_side(
        env_value="left",
        tokamak_fp_er_pc=True,
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
