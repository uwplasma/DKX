from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import sfincs_jax.io as io_module
from sfincs_jax.explicit_sparse import build_operator_from_pattern
from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.petsc_binary import read_petsc_mat_aij
from sfincs_jax.v3_sparse_pattern import summarize_v3_sparse_pattern, v3_full_system_conservative_sparsity_pattern
from sfincs_jax.v3_driver import solve_v3_full_system_linear_gmres
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
    assert result.metadata["sparse_pattern_nnz"] > 0
    assert result.metadata["sparse_pattern_max_row_nnz"] > 0
    assert any("sparse_pc_gmres complete" in msg for msg in messages)


def test_xblock_sparse_pc_gmres_solve_method_solves_fp_rhs1_system(monkeypatch) -> None:
    here = Path(__file__).parent
    nml = read_sfincs_input(here / "ref" / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_ACTIVE_DOF", "0")
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
    assert any("xblock_sparse_pc_gmres complete" in msg for msg in messages)


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
