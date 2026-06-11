from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp

import sfincs_jax.rhs1_full_assembly as rfa
import sfincs_jax.v3_driver as vd
from sfincs_jax.namelist import read_sfincs_input
from sfincs_jax.rhs1_full_assembly import (
    build_active_projected_rhs1_full_csr_preconditioner,
    build_structured_rhs1_full_csr_preconditioner,
    clear_structured_rhs1_full_csr_cache,
    select_active_fortran_v3_reduced_support_mode_preconditioner,
    select_structured_rhs1_full_csr_operator,
    solve_structured_rhs1_full_csr,
)
from sfincs_jax.rhs1_block_operator import RHS1BlockLayout
from sfincs_jax.rhs1_full_csr_kinetic_pc import rhs1_full_csr_x_ell_block_indices
from sfincs_jax.v3_system import apply_v3_full_system_operator, full_system_operator_from_namelist, rhs_v3_full_system


REF = Path(__file__).parent / "ref"


def _deterministic_vector(size: int) -> np.ndarray:
    idx = np.arange(int(size), dtype=np.float64)
    return np.sin(0.17 * idx) + 0.25 * np.cos(0.31 * idx)


def test_structured_full_csr_matches_constraint_scheme1_full_operator() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)

    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected, selection.to_dict()
    assert selection.matrix is not None
    assert selection.matrix.shape == (op.total_size, op.total_size)
    assert selection.metadata["tail_nnz"] > 0

    x = _deterministic_vector(op.total_size)
    expected = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x)))
    actual = selection.matvec(x)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_structured_full_csr_matches_phi1_constraint_scheme2_operator_and_reuses_cache() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "include_phi1_linear_subset_tiny.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.25)

    first = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    second = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert first.selected, first.to_dict()
    assert second.selected, second.to_dict()
    assert second.cache_hit
    assert second.metadata["object_cache_hit"] is True

    x = _deterministic_vector(op.total_size)
    expected = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x)))
    actual = first.matvec(x)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_structured_full_csr_rejects_memory_budget_fail_closed() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)

    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=1)
    assert not selection.selected
    assert selection.reason.startswith(("csr_budget_preflight_exceeded:", "csr_budget_exceeded:"))
    assert selection.to_dict()["fblock_selection"]["selected"] is False


def test_structured_full_csr_rejects_phi1_in_kinetic_fail_closed() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)

    selection = select_structured_rhs1_full_csr_operator(op)
    assert not selection.selected
    assert selection.reason == "unsupported_phi1_in_kinetic"


def test_driver_structured_full_csr_bundle_matches_full_and_active_operator() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)

    full_bundle = vd._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=None,
        csr_max_mb=100.0,
        drop_tol=0.0,
        emit=None,
    )
    assert full_bundle is not None
    x = _deterministic_vector(op.total_size)
    expected_full = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x)))
    np.testing.assert_allclose(full_bundle.matvec(x), expected_full, rtol=1.0e-12, atol=1.0e-12)

    active = vd._transport_active_dof_indices(op)
    active_bundle = vd._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=active,
        csr_max_mb=100.0,
        drop_tol=0.0,
        emit=None,
    )
    assert active_bundle is not None
    x_active = _deterministic_vector(int(active.size))
    x_full = np.zeros((op.total_size,), dtype=np.float64)
    x_full[active] = x_active
    expected_active = np.asarray(apply_v3_full_system_operator(op, jnp.asarray(x_full)))[active]
    np.testing.assert_allclose(active_bundle.matvec(x_active), expected_active, rtol=1.0e-12, atol=1.0e-12)


def test_structured_full_csr_host_gmres_solve_reaches_true_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = rhs_v3_full_system(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="gmres",
        preconditioner="diagonal_schur",
        max_csr_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    true_norm = float(np.linalg.norm(true_residual))
    assert true_norm <= 1.0e-8 * max(float(np.linalg.norm(np.asarray(rhs))), 1.0)
    np.testing.assert_allclose(true_norm, result.residual_norm, rtol=1.0e-10, atol=1.0e-12)
    preconditioner = result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "diagonal_schur"
    assert preconditioner["metadata"]["tail_size"] == op.extra_size


def test_structured_full_csr_native_xell_preconditioner_matches_block_reference() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    layout = RHS1BlockLayout.from_operator(op)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None

    preconditioner = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=layout,
        kind="native_xell",
        max_block_inverse_nbytes=100_000_000,
        regularization=0.0,
    )
    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.kind == "native_xell"
    assert preconditioner.metadata["native_factor_available"] is True
    assert preconditioner.metadata["backend"] == "jax_native_x_ell"
    assert preconditioner.operator is not None

    rhs = _deterministic_vector(op.total_size)
    actual = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    expected = np.zeros_like(rhs)
    matrix = selection.matrix.tocsr()
    for indices in rhs1_full_csr_x_ell_block_indices(layout):
        expected[indices] = np.linalg.solve(matrix[indices[:, None], indices].toarray(), rhs[indices])
    tail_diag = matrix.diagonal()[int(layout.f_size) :].copy()
    floor = float(preconditioner.metadata["tail_diagonal_floor"])
    small = np.abs(tail_diag) <= floor
    tail_diag[small] = np.where(tail_diag[small] < 0.0, -floor, floor)
    expected[int(layout.f_size) :] = rhs[int(layout.f_size) :] / tail_diag

    np.testing.assert_allclose(actual, expected, rtol=1.0e-12, atol=1.0e-12)


def test_structured_full_csr_native_xell_tail_schur_gmres_reaches_true_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = rhs_v3_full_system(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="gmres",
        preconditioner="native_xell_tail_schur",
        max_csr_nbytes=100_000_000,
        preconditioner_max_block_inverse_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    assert result.metadata["preconditioner"]["kind"] == "native_xell_tail_schur"
    pc_meta = result.metadata["preconditioner"]["metadata"]
    assert pc_meta["backend"] == "jax_native_x_ell_tail_schur"
    assert pc_meta["native_factor_available"] is True
    assert pc_meta["schur_nbytes"] > 0
    assert pc_meta["factor_nbytes_actual"] <= pc_meta["max_factor_nbytes"]
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_structured_full_csr_native_xell_tail_schur_preconditioner_is_memory_gated() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    layout = RHS1BlockLayout.from_operator(op)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None

    schur_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=layout,
        kind="native_xell_tail_schur",
        max_schur_size=1,
        max_block_inverse_nbytes=100_000_000,
    )
    assert schur_rejected.selected is False
    assert schur_rejected.reason == "schur_tail_size_exceeded:4>1"

    budget_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=layout,
        kind="native_xell_tail_schur",
        max_schur_size=2048,
        max_block_inverse_nbytes=1,
    )
    assert budget_rejected.selected is False
    assert "budget_exceeded" in budget_rejected.reason
    assert budget_rejected.metadata["max_factor_nbytes"] == 1


def test_structured_full_csr_active_direct_solve_reaches_true_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)
    active = vd._transport_active_dof_indices(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        method="direct",
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    assert result.metadata["method"] == "direct"
    assert result.metadata["active_dof"] is True
    assert result.metadata["active_size"] == int(active.size)
    assert result.metadata["factor_kind"] == "splu"
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_active_projected_spilu_preconditioner_is_memory_gated() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        kind="active_ilu",
        max_factor_nbytes=1,
    )
    assert rejected.selected is False
    assert rejected.reason.startswith("active_spilu_budget_exceeded:")

    auto_fallback = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        kind="auto",
        max_factor_nbytes=1,
    )
    assert auto_fallback.selected is True
    assert auto_fallback.kind == "jacobi"
    assert auto_fallback.reason.startswith("auto_selected:")
    assert auto_fallback.metadata["auto_rejected_candidates"]

    accepted = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        kind="active_ilu",
        max_factor_nbytes=100_000_000,
    )
    assert accepted.selected is True
    assert accepted.kind == "active_spilu"
    assert accepted.metadata["factor_nbytes_actual"] > 0

    native_full_kind = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="native_xell_tail_schur",
        max_factor_nbytes=100_000_000,
    )
    assert native_full_kind.selected is False
    assert native_full_kind.reason == "unsupported_active_projected_preconditioner"

    coarse_rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_coarse",
        max_factor_nbytes=1,
    )
    assert coarse_rejected.selected is False
    assert coarse_rejected.reason.startswith("active_coarse_budget_exceeded:")

    low_l_rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_low_l_schur",
        max_factor_nbytes=1,
    )
    assert low_l_rejected.selected is False
    assert low_l_rejected.reason.startswith("active_low_l_schur_budget_exceeded:")

    xblock_rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xblock",
        max_factor_nbytes=1,
    )
    assert xblock_rejected.selected is False
    assert xblock_rejected.reason.startswith("active_xblock_budget_exceeded:")

    xblock_accepted = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xblock",
        max_factor_nbytes=100_000_000,
    )
    assert xblock_accepted.selected is True
    assert xblock_accepted.kind == "active_xblock"
    assert xblock_accepted.metadata["block_count"] > 0
    assert xblock_accepted.metadata["factor_nbytes_actual"] > 0


def test_active_scaled_ilu_preconditioner_equilibrates_and_solves(monkeypatch) -> None:
    matrix = sp.csr_matrix(
        [
            [1.0e-6, 2.0e-6, 0.0],
            [0.0, 1.0, -0.25],
            [0.0, 2.0e5, 1.0e6],
        ],
        dtype=np.float64,
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_DIAGONAL_SHIFT", "0")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="active_scaled_ilu",
        max_factor_nbytes=10_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_scaled_lu"
    assert pc.metadata["row_scaling"]["scale_max"] > pc.metadata["row_scaling"]["scale_min"]
    rhs = np.asarray([1.0, -2.0, 3.0], dtype=np.float64)
    solved = pc.operator.matvec(rhs)
    np.testing.assert_allclose(matrix @ solved, rhs, rtol=1.0e-10, atol=1.0e-10)


def test_active_fortran_v3_reduced_lu_preconditioner_drops_default_couplings(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "1e-12")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=100_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_fortran_v3_reduced_lu"
    assert pc.metadata["architecture"] == "fortran_v3_reduced_active_pc_matrix"
    assert pc.metadata["reduced_matrix_nnz"] < pc.metadata["matrix_nnz"]
    assert pc.metadata["dropped_entries"]["x_nonlocal"] > 0
    assert "ell_two" in pc.metadata["dropped_entries"]
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_fortran_v3_reduced_lu_defaults_to_natural_ordering(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [4.0, 0.3, 0.0, 0.1],
                [0.2, 3.5, 0.4, 0.0],
                [0.0, 0.2, 3.8, 0.5],
                [0.1, 0.0, 0.3, 3.2],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", raising=False)

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.metadata["permc_spec"] == "NATURAL"
    assert pc.metadata["permc_spec_requested"] == "AUTO"
    assert pc.metadata["permc_spec_candidates"] == ("NATURAL", "COLAMD")


def test_active_fortran_v3_reduced_lu_respects_explicit_ordering(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr", dtype=np.float64) * 3.0
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", "COLAMD")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.metadata["permc_spec"] == "COLAMD"
    assert pc.metadata["permc_spec_requested"] == "COLAMD"
    assert pc.metadata["permc_spec_candidates"] == ("COLAMD",)


def test_active_fortran_v3_reduced_lu_supports_petsc_style_rcm_ordering(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=0,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    dense = np.asarray(
        [
            [5.0, 0.1, 0.0, 0.0, 0.2, 0.0],
            [0.2, 4.8, 0.3, 0.0, 0.0, 0.0],
            [0.0, 0.4, 4.5, 0.2, 0.0, 0.1],
            [0.0, 0.0, 0.2, 4.2, 0.5, 0.0],
            [0.1, 0.0, 0.0, 0.4, 4.7, 0.3],
            [0.0, 0.0, 0.2, 0.0, 0.1, 5.2],
        ],
        dtype=np.float64,
    )
    matrix = sp.csr_matrix(dense)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", "RCM")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
        preconditioner_x=0,
        preconditioner_xi=0,
        preconditioner_species=0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.metadata["permc_spec"] == "RCM"
    assert pc.metadata["superlu_permc_spec"] == "NATURAL"
    assert pc.metadata["explicit_symmetric_ordering"] is True
    rhs = _deterministic_vector(layout.total_size)
    solution = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert np.linalg.norm(matrix @ solution - rhs) < 1.0e-10


def test_active_fortran_v3_reduced_lu_falls_back_when_natural_fails(monkeypatch) -> None:
    import scipy.sparse.linalg as spla

    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr", dtype=np.float64) * 2.0
    original_splu = spla.splu
    calls: list[str] = []

    def flaky_splu(matrix_arg, *args, **kwargs):
        permc = str(kwargs.get("permc_spec", ""))
        calls.append(permc)
        if permc == "NATURAL":
            raise RuntimeError("forced natural ordering failure")
        return original_splu(matrix_arg, *args, **kwargs)

    monkeypatch.setattr(spla, "splu", flaky_splu)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PERMC_SPEC", raising=False)

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert calls[:2] == ["NATURAL", "COLAMD"]
    assert pc.metadata["permc_spec"] == "COLAMD"
    assert pc.metadata["permc_failures"][0]["permc_spec"] == "NATURAL"


def test_active_fortran_v3_reduced_matrix_respects_support_modes(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=2,
        n_x=3,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=24,
        phi1_size=0,
        extra_size=0,
        total_size=24,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    dense = np.ones((layout.total_size, layout.total_size), dtype=np.float64)
    dense += 10.0 * np.eye(layout.total_size, dtype=np.float64)
    matrix = sp.csr_matrix(dense)
    active = np.arange(layout.total_size, dtype=np.int64)
    env_keys = [
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_XI",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_SPECIES",
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X_MIN_L",
    ]

    def reduced_nnz(**settings: str) -> tuple[int, dict[str, object]]:
        for key in env_keys:
            monkeypatch.delenv(key, raising=False)
        for suffix, value in settings.items():
            monkeypatch.setenv(f"SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_{suffix}", value)
        reduced, metadata = rfa._active_fortran_v3_reduced_preconditioner_matrix(
            matrix=matrix,
            layout=layout,
            active_indices=active,
            regularization=0.0,
        )
        return int(reduced.nnz), metadata

    default_nnz, default_meta = reduced_nnz()
    x0_nnz, x0_meta = reduced_nnz(X="0")
    x3_nnz, x3_meta = reduced_nnz(X="3")
    xi0_nnz, xi0_meta = reduced_nnz(XI="0")
    species0_nnz, species0_meta = reduced_nnz(SPECIES="0")
    xmin_l_nnz, xmin_l_meta = reduced_nnz(X_MIN_L="2")

    assert default_meta["fortran_reduced_filter"] == "layout_decoded_supports"
    assert default_meta["preconditioner_x"] == 1
    assert default_meta["preconditioner_xi"] == 1
    assert default_meta["preconditioner_species"] == 1
    assert default_meta["dropped_entries"]["x_nonlocal"] > 0
    assert default_meta["dropped_entries"]["species_cross"] > 0
    assert default_meta["dropped_entries"]["ell_two"] > 0
    assert default_meta["dropped_entries"]["ell_outside_support"] >= default_meta["dropped_entries"]["ell_two"]

    assert x0_meta["preconditioner_x"] == 0
    assert x0_nnz > default_nnz
    assert x3_meta["preconditioner_x"] == 3
    assert x3_nnz > default_nnz
    assert xi0_meta["preconditioner_xi"] == 0
    assert xi0_nnz > default_nnz
    assert species0_meta["preconditioner_species"] == 0
    assert species0_nnz > default_nnz
    assert xmin_l_meta["preconditioner_x_min_l"] == 2
    assert xmin_l_nnz > default_nnz


def test_active_fortran_v3_reduced_builder_prefers_explicit_support_modes(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=9,
        phi1_size=0,
        extra_size=0,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    dense = np.ones((layout.total_size, layout.total_size), dtype=np.float64)
    dense += 8.0 * np.eye(layout.total_size, dtype=np.float64)
    matrix = sp.csr_matrix(dense)
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_PRECONDITIONER_X", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")

    explicit = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=0,
    )
    env_driven = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert explicit.selected, explicit.to_dict()
    assert env_driven.selected, env_driven.to_dict()
    assert explicit.metadata["preconditioner_x"] == 1
    assert env_driven.metadata["preconditioner_x"] == 0
    assert env_driven.metadata["reduced_matrix_nnz"] > explicit.metadata["reduced_matrix_nnz"]


def test_active_fortran_v3_reduced_lu_prefill_gate_rejects_before_factorization(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=9,
        phi1_size=0,
        extra_size=0,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr", dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_PREFILL_SAFETY_FACTOR", "1e9")

    rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert rejected.selected is False
    assert rejected.reason.startswith("active_fortran_v3_pc_matrix_lu_prefill_budget_exceeded:")
    assert rejected.metadata["factor_kind"] == "lu"
    assert rejected.metadata["factor_nbytes_estimate"] < 1_000_000
    assert rejected.metadata["factor_nbytes_prefill_estimate"] > 1_000_000
    assert rejected.metadata["lu_prefill_safety_factor"] == 1.0e9


def test_active_fortran_v3_reduced_lu_large_default_prefill_rejects_observed_production_estimate(
    monkeypatch,
) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=9,
        phi1_size=0,
        extra_size=0,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr", dtype=np.float64)
    observed_symbolic_estimate = int(1687.0 * 1024.0 * 1024.0)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_SIZE", "1")
    monkeypatch.delenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LU_LARGE_PREFILL_SAFETY_FACTOR",
        raising=False,
    )
    monkeypatch.setattr(
        rfa,
        "_estimate_spilu_factor_nbytes",
        lambda *, matrix, fill_factor: observed_symbolic_estimate,
    )

    rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=int(45.0 * 1024.0 * 1024.0 * 1024.0),
        regularization=0.0,
    )

    assert rejected.selected is False
    assert rejected.reason.startswith("active_fortran_v3_pc_matrix_lu_prefill_budget_exceeded:")
    assert rejected.metadata["factor_kind"] == "lu"
    assert rejected.metadata["factor_nbytes_estimate"] == observed_symbolic_estimate + 2 * layout.total_size * 8
    assert rejected.metadata["lu_prefill_safety_factor"] == 32.0
    assert rejected.metadata["factor_nbytes_prefill_estimate"] > int(45.0 * 1024.0 * 1024.0 * 1024.0)


def test_active_fortran_v3_support_mode_preflight_selects_lower_true_residual(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=0,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    dense = np.asarray(
        [
            [5.0, 0.3, 1.2, -0.4, 0.7, 0.2],
            [0.2, 4.5, -0.5, 1.0, -0.3, 0.9],
            [1.1, -0.2, 5.3, 0.4, 1.4, -0.6],
            [-0.3, 0.8, 0.5, 4.8, -0.7, 1.2],
            [0.6, -0.4, 1.3, -0.5, 5.5, 0.1],
            [0.1, 0.7, -0.4, 1.1, 0.2, 4.9],
        ],
        dtype=np.float64,
    )
    matrix = sp.csr_matrix(dense)
    active = np.arange(layout.total_size, dtype=np.int64)
    x_true = np.asarray([0.2, -0.5, 0.7, -0.3, 0.4, 0.9], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")

    pc, metadata = select_active_fortran_v3_reduced_support_mode_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        requested_kind="active_fortran_v3_reduced_lu",
        regularization=0.0,
        max_factor_nbytes=2_000_000,
        rhs=rhs,
        true_matvec=lambda vector: np.asarray(matrix @ np.asarray(vector, dtype=np.float64)),
        candidates="current,x0",
        max_candidates=2,
        min_improvement_ratio=1.01,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=0,
    )

    assert pc.selected, pc.to_dict()
    assert metadata["selected_candidate"] == "x0"
    assert metadata["accepted_nonbaseline"] is True
    assert pc.metadata["preconditioner_x"] == 0
    assert metadata["baseline_residual_after"] > 1.0e-3
    assert metadata["best_residual_after"] < 1.0e-10
    recovered = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(recovered, x_true, rtol=1.0e-10, atol=1.0e-10)


def test_active_fortran_v3_support_mode_preflight_skips_relaxed_candidates_when_current_is_complete(
    monkeypatch,
) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=3,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=0,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    local_block = sp.csr_matrix(np.asarray([[3.0, 0.2], [-0.1, 3.2]], dtype=np.float64))
    matrix = sp.block_diag((local_block, local_block, local_block), format="csr")
    active = np.arange(layout.total_size, dtype=np.int64)
    rhs = np.asarray([1.0, -0.2, 0.4, 0.7, -0.5, 0.3], dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_DIAGONAL_SHIFT", "0")

    pc, metadata = select_active_fortran_v3_reduced_support_mode_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        requested_kind="active_fortran_v3_reduced_lu",
        regularization=0.0,
        max_factor_nbytes=2_000_000,
        rhs=rhs,
        true_matvec=lambda vector: np.asarray(matrix @ np.asarray(vector, dtype=np.float64)),
        candidates="current,x0,xmin_l2,species0",
        max_candidates=4,
        min_improvement_ratio=1.01,
        preconditioner_x=1,
        preconditioner_xi=1,
        preconditioner_species=1,
        preconditioner_x_min_l=0,
    )

    assert pc.selected, pc.to_dict()
    assert metadata["selected_candidate"] == "current"
    assert metadata["accepted_nonbaseline"] is False
    assert metadata["early_stop_reason"] == "current_support_dropped_no_entries"
    assert len(metadata["candidates"]) == 1


def test_direct_tail_cache_key_includes_fortran_support_modes() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr", dtype=np.float64)
    active = np.arange(layout.total_size, dtype=np.int64)

    key_default = vd._direct_tail_structured_pc_cache_key(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=1.0e-12,
        support_modes=(1, 1, 1, 0),
    )
    key_radial_dense = vd._direct_tail_structured_pc_cache_key(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1_000_000,
        regularization=1.0e-12,
        support_modes=(0, 1, 1, 0),
    )

    assert key_default != key_radial_dense


def test_active_fortran_v3_reduced_preconditioner_is_memory_gated(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_FACTOR_KIND", "lu")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_fortran_v3_reduced_lu",
        max_factor_nbytes=1,
    )

    assert pc.selected is False
    assert pc.reason.startswith("active_fortran_v3_pc_matrix_budget_exceeded:")
    assert pc.metadata["reduced_matrix_nnz"] < pc.metadata["matrix_nnz"]


def test_active_xblock_preconditioner_can_scale_local_blocks(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_SCALE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", "3")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xblock",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_xblock"
    assert pc.metadata["block_scaling_enabled"] is True
    assert pc.metadata["block_scale_nbytes_actual"] > 0
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_xblock_ilu_preconditioner_builds_bounded_partial_blocks(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FACTOR_KIND", "ilu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_DROP_TOL", "1e-3")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_FILL_FACTOR", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_ALLOW_SINGULAR_FALLBACK", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", "3")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xblock_ilu",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_xblock"
    assert pc.metadata["factor_kind"] == "spilu"
    assert pc.metadata["allow_block_fallback"] is True
    assert pc.metadata["block_count"] > 0
    assert pc.metadata["covered_fraction"] > 0.0
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_ell_band_schur_preconditioner_builds_coupled_pitch_band(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_CENTER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_HALF_WIDTH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", "3")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_ell_band_schur",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_ell_band_schur"
    assert pc.metadata["ell_min"] == 2
    assert pc.metadata["ell_max"] == 2
    assert pc.metadata["band_size"] > 0
    assert pc.metadata["band_factor_nbytes_actual"] > 0
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_projected_auto_ladder_can_select_ell_band(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES",
        "active_ell_band_schur,jacobi",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_CENTER", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ELL_BAND_HALF_WIDTH", "0")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="auto",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_ell_band_schur"
    assert pc.reason.startswith("auto_selected:")
    assert pc.metadata["auto_selected_kind"] == "active_ell_band_schur"
    assert pc.metadata["auto_candidates"] == ["active_ell_band_schur", "jacobi"]


def test_active_projected_default_auto_ladder_avoids_tail_sparse_path(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES", raising=False)

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="auto",
        max_factor_nbytes=1,
    )

    assert pc.selected, pc.to_dict()
    assert "active_global_field_split_schur" in pc.metadata["auto_candidates"]
    assert "active_tail_sparse_coarse" not in pc.metadata["auto_candidates"]


def test_active_projected_auto_ladder_skips_large_diagonal_fallbacks(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    matrix = sp.eye(12, format="csc")

    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES",
        "active_diagonal_schur,jacobi",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", "10")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=None,
        active_indices=None,
        kind="auto",
        max_factor_nbytes=100_000_000,
    )

    assert not pc.selected
    assert pc.reason == "active_auto_no_safe_large_candidate_selected"
    assert pc.metadata["auto_candidates"] == []
    assert pc.metadata["auto_candidates_requested"] == ["active_diagonal_schur", "jacobi"]
    assert pc.metadata["auto_skipped_large_fallbacks"] == ("active_diagonal_schur", "jacobi")


def test_active_projected_auto_ladder_uses_large_default_candidates(monkeypatch, capsys) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    matrix = sp.eye(12, format="csc")

    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_CANDIDATES", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_LARGE_FALLBACK_SIZE", "10")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=None,
        active_indices=None,
        kind="auto",
        max_factor_nbytes=100_000_000,
    )
    out = capsys.readouterr().out

    assert not pc.selected
    assert pc.reason == "active_auto_no_safe_large_candidate_selected"
    assert pc.metadata["auto_candidates"] == [
        "active_fortran_v3_reduced_native_stack",
        "active_symbolic_frontal_schur_lu",
        "active_symbolic_superblock_lu",
        "active_coupled_kinetic_field_split_sparse_coarse",
        "active_symbolic_block_schur_lu",
        "active_fortran_v3_reduced_lu",
    ]
    assert pc.metadata["auto_large_default_used"] is True
    assert pc.metadata["auto_rejected_candidates"][0]["kind"] == "active_fortran_v3_reduced_native_stack"
    assert pc.metadata["auto_rejected_candidates"][1]["kind"] == "active_symbolic_frontal_schur_lu"
    assert pc.metadata["auto_rejected_candidates"][2]["kind"] == "active_symbolic_superblock_lu"
    assert pc.metadata["auto_rejected_candidates"][3]["kind"] == "active_coupled_kinetic_field_split_sparse_coarse"
    assert pc.metadata["auto_rejected_candidates"][4]["kind"] == "active_symbolic_block_schur_lu"
    assert pc.metadata["auto_rejected_candidates"][5]["kind"] == "active_fortran_v3_reduced_lu"
    assert "auto candidate start kind=active_fortran_v3_reduced_native_stack" in out
    assert "auto candidate done kind=active_fortran_v3_reduced_native_stack" in out
    assert "auto candidate start kind=active_symbolic_frontal_schur_lu" in out
    assert "auto candidate done kind=active_symbolic_frontal_schur_lu" in out
    assert "auto candidate start kind=active_symbolic_superblock_lu" in out
    assert "auto candidate done kind=active_symbolic_superblock_lu" in out
    assert "auto candidate start kind=active_coupled_kinetic_field_split_sparse_coarse" in out
    assert "auto candidate done kind=active_coupled_kinetic_field_split_sparse_coarse" in out
    assert "auto candidate start kind=active_symbolic_block_schur_lu" in out
    assert "auto candidate done kind=active_symbolic_block_schur_lu" in out
    assert "auto candidate start kind=active_fortran_v3_reduced_lu" in out
    assert "auto candidate done kind=active_fortran_v3_reduced_lu" in out


def test_active_schwarz_sparse_coarse_preconditioner_builds_two_level_candidate(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", "32")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_schwarz_sparse_coarse",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_tail_sparse_coarse"
    assert pc.metadata["architecture"] == "additive_schwarz_global_sparse_coarse"
    assert pc.metadata["requested_base_kind"] == "active_overlap_schwarz"
    assert pc.metadata["base_kind"] == "active_overlap_schwarz"
    assert pc.metadata["base_preconditioner"]["metadata"]["patch_count"] > 0
    assert pc.metadata["coarse_size"] > 0
    assert pc.metadata["az_basis_nnz"] > 0
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_projected_auto_ladder_can_select_schwarz_sparse_coarse(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_AUTO_CANDIDATES",
        "active_schwarz_sparse_coarse,jacobi",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SPARSE_COARSE_MAX_SIZE", "32")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="auto",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.reason.startswith("auto_selected:")
    assert pc.metadata["auto_selected_kind"] == "active_tail_sparse_coarse"
    assert pc.metadata["requested_kind"] == "active_schwarz_sparse_coarse"
    assert pc.metadata["architecture"] == "additive_schwarz_global_sparse_coarse"
    assert pc.metadata["auto_candidates"] == ["active_schwarz_sparse_coarse", "jacobi"]


def test_active_global_sparse_lu_preconditioner_solves_active_csr(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FACTOR_MAX_SIZE", "1000000")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_global_sparse_lu",
        max_factor_nbytes=100_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_sparse_factor"
    assert pc.metadata["architecture"] == "global_active_sparse_factor"
    assert pc.metadata["factor_kind"] == "lu"
    probe = _deterministic_vector(active_matrix.shape[0])
    rhs = np.asarray(active_matrix @ probe, dtype=np.float64).reshape((-1,))
    recovered = np.asarray(pc.operator.matvec(rhs), dtype=np.float64).reshape((-1,))
    residual = np.linalg.norm(np.asarray(active_matrix @ recovered, dtype=np.float64).reshape((-1,)) - rhs)
    assert residual < 1.0e-8


def test_active_global_sparse_factor_is_memory_gated(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FACTOR_MAX_SIZE", "1000000")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_global_sparse_ilu",
        max_factor_nbytes=1,
    )

    assert not pc.selected
    assert pc.kind == "active_global_sparse_factor"
    assert pc.reason.startswith("active_global_sparse_factor_budget_exceeded:")
    assert pc.metadata["factor_kind"] == "ilu"


def test_active_xell_window_lsq_schur_preconditioner_builds_and_applies(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_SPEC", "0:0:2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_ELL_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_BASE", "jacobi")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xell_window_lsq_schur",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_xell_window_lsq_schur"
    assert pc.metadata["window_size"] > 0
    assert pc.metadata["a_window_nnz"] > 0
    assert pc.metadata["normal_nbytes"] > 0
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_xell_window_lsq_schur_preconditioner_is_memory_gated(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_SPEC", "0:0:2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_BASE", "jacobi")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_xell_window_lsq_schur",
        max_factor_nbytes=1,
    )

    assert pc.selected is False
    assert pc.reason.startswith("active_xell_window_lsq_schur_budget_exceeded:")


def test_active_xell_window_lsq_schur_solves_global_residual_equation(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(3, dtype=np.int64)
    matrix = sp.csr_matrix(
        [
            [1.0, 8.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, -4.0, 1.0],
        ],
        dtype=np.float64,
    )
    rhs = np.asarray([1.0, 0.25, -0.5], dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_SPEC", "0:0:1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_ELL_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_INCLUDE_TAIL", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XELL_WINDOW_BASE", "jacobi")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_xell_window_lsq_schur",
        max_factor_nbytes=1_000_000,
    )

    assert pc.selected, pc.to_dict()
    jacobi = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="jacobi",
        max_factor_nbytes=1_000_000,
    )
    residual_jacobi = rhs - np.asarray(matrix @ jacobi.operator.matvec(rhs), dtype=np.float64)
    residual_lsq = rhs - np.asarray(matrix @ pc.operator.matvec(rhs), dtype=np.float64)
    assert np.linalg.norm(residual_lsq) < np.linalg.norm(residual_jacobi)


def test_active_projected_diagonal_schur_solves_diagonal_kinetic_tail_system() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=6,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=2,
        total_size=8,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.asarray([0, 2, 4, 6, 7], dtype=np.int64)
    diag = np.asarray([2.0, 3.0, 5.0], dtype=np.float64)
    u = np.asarray([[0.25, -0.10], [0.0, 0.15], [-0.35, 0.20]], dtype=np.float64)
    v = np.asarray([[0.10, -0.25, 0.30], [-0.20, 0.05, 0.40]], dtype=np.float64)
    w = np.asarray([[1.40, 0.08], [-0.12, 1.15]], dtype=np.float64)
    matrix = sp.bmat([[sp.diags(diag), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]], format="csr")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_diagonal_schur",
        max_factor_nbytes=1_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_diagonal_schur"
    assert pc.metadata["tail_size"] == 2
    assert pc.metadata["kinetic_size"] == 3
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-12, atol=1.0e-12)


def test_active_projected_diagonal_schur_rejects_noncontiguous_tail() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=3,
        phi1_size=0,
        extra_size=2,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.asarray([0, 3, 1, 4], dtype=np.int64)
    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sp.eye(active.size, format="csr"),
        layout=layout,
        active_indices=active,
        kind="active_diagonal_schur",
    )
    assert pc.selected is False
    assert pc.reason == "active_tail_not_contiguous"


def test_active_global_field_split_schur_solves_block_system_with_xblock_base() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    kinetic = np.asarray(
        [
            [3.0, 0.4, -0.2, 0.1],
            [0.1, 2.4, 0.3, -0.2],
            [-0.1, 0.2, 2.8, 0.5],
            [0.0, -0.3, 0.2, 3.2],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.20, -0.10], [0.00, 0.15], [-0.30, 0.25], [0.10, 0.05]], dtype=np.float64)
    v = np.asarray([[0.10, -0.20, 0.30, 0.15], [-0.25, 0.05, 0.10, 0.40]], dtype=np.float64)
    w = np.asarray([[1.30, 0.12], [-0.08, 1.55]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_field_split_schur"
    assert pc.metadata["architecture"] == "active_kinetic_global_field_split_schur"
    assert pc.metadata["base_kind"] == "active_xblock"
    assert pc.metadata["tail_size"] == 2
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5, -0.1], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-10, atol=1.0e-10)


def test_active_global_field_split_schur_can_use_native_xell_base(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    kinetic = np.asarray(
        [
            [3.0, 0.4, -0.2, 0.1],
            [0.1, 2.4, 0.3, -0.2],
            [-0.1, 0.2, 2.8, 0.5],
            [0.0, -0.3, 0.2, 3.2],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.20, -0.10], [0.00, 0.15], [-0.30, 0.25], [0.10, 0.05]], dtype=np.float64)
    v = np.asarray([[0.10, -0.20, 0.30, 0.15], [-0.25, 0.05, 0.10, 0.40]], dtype=np.float64)
    w = np.asarray([[1.30, 0.12], [-0.08, 1.55]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_BASE", "native_xell")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_field_split_schur"
    assert pc.metadata["requested_base_kind"] == "native_xell"
    assert pc.metadata["base_kind"] == "native_xell"
    assert pc.metadata["base_preconditioner"]["metadata"]["backend"] == "jax_native_x_ell"
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5, -0.1], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-10, atol=1.0e-10)


def test_active_fortran_v3_reduced_ilu_uses_large_matrix_safe_defaults(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [3.0, 0.2, 0.0, 0.0, 0.1, 0.0],
                [0.1, 2.7, 0.4, 0.0, 0.0, 0.2],
                [0.0, 0.3, 3.4, 0.5, 0.2, 0.0],
                [0.0, 0.0, 0.2, 2.8, 0.0, 0.1],
                [0.1, 0.0, 0.2, 0.0, 1.6, 0.3],
                [0.0, 0.2, 0.0, 0.1, 0.4, 1.8],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FORTRAN_V3_PC_LARGE_SIZE", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_fortran_v3_reduced_ilu",
        max_factor_nbytes=1_000_000,
        regularization=1.0e-12,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_fortran_v3_reduced_ilu"
    assert pc.metadata["architecture"] == "fortran_v3_reduced_active_pc_matrix"
    assert pc.metadata["large_matrix_defaults"] is True
    assert pc.metadata["fill_factor"] == 1.2
    assert pc.metadata["drop_tol"] == 5.0e-2
    assert pc.metadata["requires_preflight"] is True


def test_active_global_field_split_schur_is_memory_gated(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    matrix = sp.eye(layout.total_size, format="csr")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_BASE", "jacobi")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=1,
    )

    assert pc.selected is False
    assert pc.kind == "active_global_field_split_schur"
    assert pc.reason.startswith("active_global_field_split_budget_exceeded:")
    assert pc.metadata["tail_size"] == 2


def test_active_global_field_split_schur_builds_and_applies_on_quick_active_csr() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsc()

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=100_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_field_split_schur"
    assert pc.metadata["kinetic_size"] > 0
    assert pc.metadata["tail_size"] == op.extra_size
    rhs = _deterministic_vector(active_matrix.shape[0])
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_global_field_split_schur_uses_active_native_xell_base_on_reduced_kinetic_set(
    monkeypatch,
) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=3,
        n_theta=1,
        n_zeta=1,
        f_size=6,
        phi1_size=0,
        extra_size=2,
        total_size=8,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.asarray([0, 1, 3, 4, 6, 7], dtype=np.int64)
    kinetic = np.asarray(
        [
            [3.0, 0.4, -0.2, 0.1],
            [0.1, 2.4, 0.3, -0.2],
            [-0.1, 0.2, 2.8, 0.5],
            [0.0, -0.3, 0.2, 3.2],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.20, -0.10], [0.00, 0.15], [-0.30, 0.25], [0.10, 0.05]], dtype=np.float64)
    v = np.asarray([[0.10, -0.20, 0.30, 0.15], [-0.25, 0.05, 0.10, 0.40]], dtype=np.float64)
    w = np.asarray([[1.30, 0.12], [-0.08, 1.55]], dtype=np.float64)
    active_matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_BASE", "active_native_xell")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=layout,
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=100_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_global_field_split_schur"
    assert pc.metadata["base_kind"] == "active_native_xell"
    assert pc.metadata["base_preconditioner"]["metadata"]["backend"] == "python_native_active_x_ell_line_inverse"
    assert pc.metadata["base_preconditioner"]["metadata"]["block_size_max"] == 4
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5, -0.1], dtype=np.float64)
    rhs = np.asarray(active_matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-10, atol=1.0e-10)


def test_active_angular_line_preconditioner_solves_angular_blocks() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    first = np.asarray([[3.0, 0.4], [0.2, 2.7]], dtype=np.float64)
    second = np.asarray([[2.5, -0.3], [0.1, 3.2]], dtype=np.float64)
    matrix = sp.block_diag((first, second, np.asarray([[4.0]], dtype=np.float64)), format="csr")
    active = np.arange(layout.total_size, dtype=np.int64)

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_angular_line",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_angular_line"
    assert pc.metadata["backend"] == "python_native_active_angular_line_inverse"
    assert pc.metadata["block_size_max"] == 2
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-12, atol=1.0e-12)


def test_active_native_indexed_schwarz_reduces_multiline_residual(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=0,
        extra_size=1,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    dense = 4.0 * np.eye(layout.total_size, dtype=np.float64)
    # Couplings at fixed angular location exercise the x-ell line blocks.
    for group in ([0, 2, 4, 6], [1, 3, 5, 7]):
        for left, right in zip(group[:-1], group[1:], strict=False):
            dense[left, right] = 0.35
            dense[right, left] = -0.20
    # Couplings at fixed (x, ell) exercise the angular-line blocks.
    for group in ([0, 1], [2, 3], [4, 5], [6, 7]):
        dense[group[0], group[1]] += 0.55
        dense[group[1], group[0]] -= 0.30
    dense[8, 8] = 2.5
    matrix = sp.csr_matrix(dense)
    active = np.arange(layout.total_size, dtype=np.int64)
    rhs = _deterministic_vector(layout.total_size)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MAX_BLOCK_SIZE", "4")

    jacobi = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="jacobi",
        max_factor_nbytes=1_000_000,
    )
    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_native_indexed_schwarz",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_native_indexed_schwarz"
    assert pc.metadata["architecture"] == "jax_native_overlapping_indexed_line_blocks"
    assert pc.metadata["block_family_counts"] == {"angular": 4, "xell": 2}
    assert pc.metadata["normalize_overlap"] is True
    jacobi_residual = rhs - np.asarray(matrix @ np.asarray(jacobi.operator.matvec(rhs), dtype=np.float64))
    schwarz_residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(schwarz_residual) < np.linalg.norm(jacobi_residual)


def test_active_native_indexed_schwarz_is_memory_gated(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=0,
        extra_size=1,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, format="csr")
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MAX_BLOCK_SIZE", "4")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_native_indexed_schwarz",
        max_factor_nbytes=1,
        regularization=0.0,
    )

    assert pc.selected is False
    assert pc.kind == "active_native_indexed_schwarz"
    assert pc.reason.startswith("active_native_indexed_schwarz_budget_exceeded:")


def test_active_native_xell_field_split_sparse_coarse_reduces_true_residual(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.0, 0.2, 0.7, -0.3],
            [0.1, 2.6, -0.4, 0.6],
            [0.5, -0.2, 3.4, 0.1],
            [-0.3, 0.4, 0.2, 2.9],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.15, -0.10], [0.05, 0.12], [-0.20, 0.18], [0.11, -0.04]], dtype=np.float64)
    v = np.asarray([[0.08, -0.12, 0.20, -0.15], [-0.18, 0.07, -0.09, 0.22]], dtype=np.float64)
    w = np.asarray([[1.4, 0.05], [-0.03, 1.7]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_GLOBAL_FIELD_SPLIT_BASE", "active_native_xell")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_SOLVER", "least_squares")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_MAX_SIZE", "64")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_SPECS", "0:0:1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_XELL_COARSE_WINDOW_ELL_RADIUS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_MAX_COLUMNS", "8")

    base = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_global_field_split_schur",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )
    coarse = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_native_xell_field_split_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert base.selected, base.to_dict()
    assert coarse.selected, coarse.to_dict()
    assert coarse.kind == "active_native_xell_field_split_sparse_coarse"
    assert coarse.metadata["architecture"] == "active_native_xell_global_field_split_sparse_coarse"
    assert coarse.metadata["base_preconditioner"]["metadata"]["requested_base_kind"] == "active_native_xell"
    assert coarse.metadata["coarse_equation"] == "least_squares"
    assert coarse.metadata["window_basis_requested"] is True
    assert coarse.metadata["window_basis_columns"] > 0
    assert coarse.metadata["adaptive_residual_basis_enabled"] is True
    assert coarse.metadata["adaptive_residual_basis_columns"] > 0
    rhs = _deterministic_vector(layout.total_size)
    base_residual = rhs - np.asarray(matrix @ np.asarray(base.operator.matvec(rhs), dtype=np.float64))
    coarse_residual = rhs - np.asarray(matrix @ np.asarray(coarse.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(base_residual) > 1.0e-4
    assert np.linalg.norm(coarse_residual) < 1.0e-10


def test_active_angular_line_field_split_sparse_coarse_builds_coupled_path(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.0, 0.2, 0.7, -0.3],
            [0.1, 2.6, -0.4, 0.6],
            [0.5, -0.2, 3.4, 0.1],
            [-0.3, 0.4, 0.2, 2.9],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.15, -0.10], [0.05, 0.12], [-0.20, 0.18], [0.11, -0.04]], dtype=np.float64)
    v = np.asarray([[0.08, -0.12, 0.20, -0.15], [-0.18, 0.07, -0.09, 0.22]], dtype=np.float64)
    w = np.asarray([[1.4, 0.05], [-0.03, 1.7]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_angular_line_field_split_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_angular_line_field_split_sparse_coarse"
    assert pc.metadata["architecture"] == "active_angular_line_global_field_split_sparse_coarse"
    assert pc.metadata["requested_base_kind"] == "active_angular_line"
    assert pc.metadata["base_preconditioner"]["metadata"]["base_preconditioner"]["kind"] == "active_angular_line"
    assert pc.metadata["adaptive_residual_basis_enabled"] is True
    assert pc.metadata["adaptive_residual_basis_columns"] > 0
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_multiline_field_split_sparse_coarse_builds_residual_composed_path(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.0, 0.2, 0.7, -0.3],
            [0.1, 2.6, -0.4, 0.6],
            [0.5, -0.2, 3.4, 0.1],
            [-0.3, 0.4, 0.2, 2.9],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.15, -0.10], [0.05, 0.12], [-0.20, 0.18], [0.11, -0.04]], dtype=np.float64)
    v = np.asarray([[0.08, -0.12, 0.20, -0.15], [-0.18, 0.07, -0.09, 0.22]], dtype=np.float64)
    w = np.asarray([[1.4, 0.05], [-0.03, 1.7]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_multiline_field_split_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_multiline_field_split_sparse_coarse"
    assert pc.metadata["architecture"] == "active_multiline_xell_angular_global_field_split_sparse_coarse"
    assert pc.metadata["requested_base_kind"] == "active_multiline_xell_angular"
    base_metadata = pc.metadata["base_preconditioner"]["metadata"]
    assert base_metadata["architecture"] == "active_multiline_xell_angular_field_split_residual"
    assert base_metadata["xell_preconditioner"]["metadata"]["requested_base_kind"] == "active_native_xell"
    assert base_metadata["angular_preconditioner"]["metadata"]["requested_base_kind"] == "active_angular_line"
    assert base_metadata["mode"] == "multiplicative"
    assert pc.metadata["adaptive_residual_basis_enabled"] is True
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_bounded_native_stack_builds_line_patch_coarse_path(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.0, 0.2, 0.7, -0.3],
            [0.1, 2.6, -0.4, 0.6],
            [0.5, -0.2, 3.4, 0.1],
            [-0.3, 0.4, 0.2, 2.9],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.15, -0.10], [0.05, 0.12], [-0.20, 0.18], [0.11, -0.04]], dtype=np.float64)
    v = np.asarray([[0.08, -0.12, 0.20, -0.15], [-0.18, 0.07, -0.09, 0.22]], dtype=np.float64)
    w = np.asarray([[1.4, 0.05], [-0.03, 1.7]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_bounded_native_stack",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_bounded_native_stack"
    assert pc.metadata["architecture"] == "active_bounded_native_line_schwarz_coupled_coarse"
    assert pc.metadata["line_base_preconditioner"]["kind"] == "active_multiline_field_split_base"
    assert pc.metadata["schwarz_requested"] is True
    assert pc.metadata["adaptive_residual_operator"] == "line_base_plus_schwarz"
    assert pc.metadata["note"] == "no_global_serial_sparse_factor"
    assert pc.metadata["requires_preflight"] is True
    assert pc.metadata["coarse_size"] > 0
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_fortran_v3_reduced_native_stack_alias_uses_bounded_components(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.0, 0.2, 0.7, -0.3],
            [0.1, 2.6, -0.4, 0.6],
            [0.5, -0.2, 3.4, 0.1],
            [-0.3, 0.4, 0.2, 2.9],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.15, -0.10], [0.05, 0.12], [-0.20, 0.18], [0.11, -0.04]], dtype=np.float64)
    v = np.asarray([[0.08, -0.12, 0.20, -0.15], [-0.18, 0.07, -0.09, 0.22]], dtype=np.float64)
    w = np.asarray([[1.4, 0.05], [-0.03, 1.7]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_ADAPTIVE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_STACK_SCHWARZ", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_fortran_v3_reduced_native_stack",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_fortran_v3_reduced_native_stack"
    assert pc.metadata["architecture"] == "fortran_v3_reduced_active_native_line_schwarz_coupled_coarse"
    assert pc.metadata["base_architecture"] == "active_bounded_native_line_schwarz_coupled_coarse"
    assert pc.metadata["no_global_serial_sparse_factor"] is True
    assert pc.metadata["exact_serial_lu_factor"] is False
    assert pc.metadata["production_candidate"] is True
    assert pc.metadata["requires_preflight"] is True
    assert pc.metadata["line_base_preconditioner"]["kind"] == "active_multiline_field_split_base"
    assert pc.metadata["schwarz_requested"] is True
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_coupled_kinetic_block_retains_true_offdiagonal_couplings(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.2, -0.7, 0.9, 0.3],
            [0.6, 2.9, -0.8, 0.7],
            [0.8, -0.5, 3.4, -0.9],
            [-0.4, 0.6, 0.5, 2.8],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.20, -0.15], [0.08, 0.17], [-0.22, 0.19], [0.12, -0.09]], dtype=np.float64)
    v = np.asarray([[0.11, -0.16, 0.24, -0.18], [-0.21, 0.09, -0.13, 0.25]], dtype=np.float64)
    w = np.asarray([[1.6, 0.07], [-0.05, 1.8]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FACTOR_KIND", "splu")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_coupled_kinetic_block",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_coupled_kinetic_block"
    assert pc.metadata["architecture"] == "active_dominant_kinetic_sparse_coupled_factor"
    assert pc.metadata["block_covers_active"] is True
    assert pc.metadata["kinetic_selected"] == 4
    assert pc.metadata["tail_selected"] == 2
    assert pc.metadata["base_kind"] == "zero_coupled_kinetic_base"
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_probe_residual_basis_adds_true_action_columns_and_reduces_residual(monkeypatch) -> None:
    from scipy.sparse.linalg import LinearOperator

    matrix = sp.csr_matrix(
        [
            [4.0, 1.5, 0.0],
            [-0.7, 3.2, 0.9],
            [0.0, -1.1, 2.8],
        ],
        dtype=np.float64,
    )
    diagonal = matrix.diagonal()
    base = LinearOperator(
        matrix.shape,
        matvec=lambda x: np.asarray(x, dtype=np.float64).reshape((-1,)) / diagonal,
        dtype=np.float64,
    )
    basis0 = sp.csc_matrix((matrix.shape[0], 0), dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_BASIS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_MAX_COLUMNS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_PROBES", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_PROBE_RESIDUAL_DROP_REL", "0")

    basis, metadata = rfa._append_probe_residual_basis_csc(
        matrix=matrix,
        base_operator=base,
        basis=basis0,
        max_total_columns=2,
        enabled_default=False,
    )

    assert metadata["probe_residual_basis_enabled"] is True
    assert metadata["probe_residual_basis_columns"] > 0
    assert basis.shape[1] <= 2
    rhs = _deterministic_vector(matrix.shape[0])
    y0 = np.asarray(base.matvec(rhs), dtype=np.float64)
    residual0 = rhs - np.asarray(matrix @ y0, dtype=np.float64)
    az_basis = matrix @ basis
    coarse = np.asarray((az_basis.T @ az_basis).toarray(), dtype=np.float64)
    coarse_rhs = np.asarray(az_basis.T @ residual0, dtype=np.float64).reshape((-1,))
    coeff = np.linalg.solve(coarse + 1.0e-14 * np.eye(coarse.shape[0]), coarse_rhs)
    y1 = y0 + np.asarray(basis @ coeff, dtype=np.float64).reshape((-1,))
    residual1 = rhs - np.asarray(matrix @ y1, dtype=np.float64)
    assert np.linalg.norm(residual1) < np.linalg.norm(residual0)


def test_active_filtered_sparse_factor_retains_selected_offdiagonal_couplings(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=0,
        extra_size=1,
        total_size=9,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    rows = list(range(layout.total_size))
    cols = list(range(layout.total_size))
    data = [4.0] * layout.total_size
    near_a = layout.kinetic_flat_index(species=0, x=0, ell=0, theta=0, zeta=0)
    near_b = layout.kinetic_flat_index(species=0, x=1, ell=0, theta=0, zeta=0)
    far_a = layout.kinetic_flat_index(species=0, x=0, ell=0, theta=0, zeta=0)
    far_b = layout.kinetic_flat_index(species=0, x=1, ell=1, theta=1, zeta=0)
    tail = layout.f_size
    rows.extend([near_a, near_b, far_a, far_b, near_a, tail])
    cols.extend([near_b, near_a, far_b, far_a, tail, near_a])
    data.extend([-0.4, 0.3, 0.25, -0.2, 0.1, -0.15])
    matrix = sp.coo_matrix((data, (rows, cols)), shape=(layout.total_size, layout.total_size)).tocsr()

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_X_RADIUS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ELL_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_THETA_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ZETA_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_KIND", "splu")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_filtered_sparse_factor",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_filtered_sparse_factor"
    assert pc.metadata["architecture"] == "active_physics_filtered_sparse_factor"
    assert pc.metadata["filtered_nnz"] < matrix.nnz
    assert pc.metadata["physical_band_nnz"] >= 2
    assert pc.metadata["include_tail_couplings"] is True
    rhs = _deterministic_vector(layout.total_size)
    applied = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    assert applied.shape == rhs.shape
    assert np.all(np.isfinite(applied))


def test_active_coupled_kinetic_sparse_coarse_admits_true_coupled_factor(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.2, -0.7, 0.9, 0.3],
            [0.6, 2.9, -0.8, 0.7],
            [0.8, -0.5, 3.4, -0.9],
            [-0.4, 0.6, 0.5, 2.8],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.20, -0.15], [0.08, 0.17], [-0.22, 0.19], [0.12, -0.09]], dtype=np.float64)
    v = np.asarray([[0.11, -0.16, 0.24, -0.18], [-0.21, 0.09, -0.13, 0.25]], dtype=np.float64)
    w = np.asarray([[1.6, 0.07], [-0.05, 1.8]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FACTOR_KIND", "splu")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-11",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_coupled_kinetic_field_split_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_coupled_kinetic_field_split_sparse_coarse"
    assert pc.metadata["architecture"] == "active_coupled_kinetic_true_action_sparse_coarse"
    assert pc.metadata["base_preconditioner"]["kind"] == "active_coupled_kinetic_block"
    assert pc.metadata["admission"]["accepted"] is True
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_coupled_kinetic_sparse_coarse_rejects_weak_true_action(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [3.0, 0.4, 1.6, 0.0, 0.2, 0.0],
            [0.1, 2.7, 0.0, -1.2, 0.0, 0.2],
            [1.1, 0.0, 3.4, 0.5, 0.2, 0.0],
            [0.0, -0.9, 0.2, 2.8, 0.0, 0.1],
            [0.1, 0.0, 0.2, 0.0, 1.6, 0.3],
            [0.0, 0.2, 0.0, 0.1, 0.4, 1.8],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_TAIL", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_FACTOR_KIND", "splu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_MAX_SIZE", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_COARSE_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-6",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_coupled_kinetic_field_split_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.kind == "active_coupled_kinetic_field_split_sparse_coarse"
    assert pc.reason.startswith("active_coupled_kinetic_sparse_coarse_admission_failed:")
    assert "max_rel=" in pc.reason
    assert "min_improvement=" in pc.reason
    assert pc.metadata["admission"]["accepted"] is False
    assert pc.metadata["requires_preflight"] is True


def test_active_symbolic_frontal_schur_lu_solves_separator_coupled_active_system(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "2e-12",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_frontal_schur_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_symbolic_frontal_schur_lu"
    assert pc.metadata["architecture"] == "active_true_operator_symbolic_frontal_schur_lu"
    assert pc.metadata["symbolic_factor_kind"] == "symbolic_frontal_schur_lu"
    assert pc.metadata["separator_count"] == 2
    assert pc.metadata["selected_cross_nnz"] == pc.metadata["total_cross_nnz"]
    assert pc.metadata["cross_separator_fraction"] == 1.0
    assert pc.metadata["admission"]["accepted"] is True
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    recovered = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(recovered, np.linalg.solve(matrix.toarray(), rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_symbolic_frontal_schur_lu_rejects_insufficient_separator_coverage(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 25.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [30.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SUPERBLOCK_BLOCKS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MAX_SEPARATOR_COLS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_BOUNDARY_WIDTH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_HIGH_DEGREE_COLS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_FRONTAL_MIN_CROSS_SEPARATOR_FRACTION", "1")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_frontal_schur_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.kind == "active_symbolic_frontal_schur_lu"
    assert pc.reason.startswith("active_symbolic_frontal_schur_lu_factor_failed:")
    assert "selected insufficient cross-block separator coverage" in pc.reason
    assert pc.metadata["min_cross_separator_fraction"] == 1.0


def test_active_symbolic_superblock_lu_solves_coupled_active_blocks(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 2.5, 0.0],
            [2.0, 3.0, 0.0, -1.0],
            [3.0, 0.0, 5.0, -1.0],
            [0.0, -0.7, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_SIZE", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS", "2")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-12",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_superblock_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_symbolic_superblock_lu"
    assert pc.metadata["architecture"] == "active_true_operator_symbolic_superblock_lu"
    assert pc.metadata["symbolic_factor_kind"] == "symbolic_superblock_lu"
    assert pc.metadata["superblock_count"] == 1
    assert pc.metadata["retained_cross_nnz"] > 0
    assert pc.metadata["dropped_cross_nnz"] == 0
    assert pc.metadata["admission"]["accepted"] is True
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    recovered = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(recovered, np.linalg.solve(matrix.toarray(), rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_symbolic_superblock_lu_rejects_size_limited_missing_coupling(
    monkeypatch,
) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=0,
        total_size=4,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 30.0, 0.0],
            [2.0, 3.0, 0.0, 0.0],
            [25.0, 0.0, 5.0, -1.0],
            [0.0, 0.0, 1.0, 2.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_MAX_BLOCKS", "1")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-2",
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SUPERBLOCK_ADMISSION_MIN_IMPROVEMENT", "10")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_superblock_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.kind == "active_symbolic_superblock_lu"
    assert pc.reason.startswith("active_symbolic_superblock_lu_admission_failed:")
    assert pc.metadata["superblock_count"] == 2
    assert pc.metadata["retained_cross_nnz"] == 0
    assert pc.metadata["dropped_cross_nnz"] > 0
    assert pc.metadata["admission"]["accepted"] is False
    assert pc.metadata["requires_preflight"] is True


def test_active_filtered_sparse_factor_prefill_gate_rejects_before_factorization(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, dtype=np.float64, format="csr")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_LARGE_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_PREFILL_SAFETY_FACTOR", "10")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_filtered_sparse_factor",
        max_factor_nbytes=1000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.reason.startswith("active_filtered_sparse_factor_prefill_budget_exceeded:")
    assert pc.metadata["factor_nbytes_estimate"] < pc.metadata["max_factor_nbytes"]
    assert pc.metadata["factor_nbytes_prefill_estimate"] > pc.metadata["max_factor_nbytes"]
    assert pc.metadata["prefill_safety_factor"] == 10.0


def test_active_filtered_sparse_factor_sparse_coarse_wraps_true_residual(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.2, -0.35, 0.18, 0.00],
            [0.24, 2.9, -0.31, 0.12],
            [0.10, -0.22, 3.4, -0.28],
            [0.00, 0.16, 0.25, 2.8],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.10, -0.05], [0.02, 0.08], [-0.12, 0.14], [0.07, -0.03]], dtype=np.float64)
    v = np.asarray([[0.04, -0.09, 0.11, -0.10], [-0.13, 0.02, -0.06, 0.16]], dtype=np.float64)
    w = np.asarray([[1.7, 0.05], [-0.04, 1.5]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_KIND", "splu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_X_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ELL_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_THETA_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_FILTERED_FACTOR_ZETA_RADIUS", "0")

    base = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_filtered_sparse_factor",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )
    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_filtered_sparse_factor_sparse_coarse",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert base.selected, base.to_dict()
    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_filtered_sparse_coarse"
    assert pc.metadata["architecture"] == "filtered_sparse_factor_global_sparse_coarse"
    assert pc.metadata["base_kind"] == "active_filtered_sparse_factor"
    assert pc.metadata["coarse_equation"] == "least_squares"
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    base_residual = rhs - np.asarray(matrix @ np.asarray(base.operator.matvec(rhs), dtype=np.float64))
    pc_residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(pc_residual) <= 1.0e-10 * max(np.linalg.norm(base_residual), 1.0)


def test_active_symbolic_block_schur_lu_solves_separator_coupled_active_system(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 0.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0, -1.0],
            [0.0, 0.0, 5.0, -1.0, 1.5],
            [0.0, 0.0, -1.0, 2.0, 0.5],
            [3.0, -2.0, 1.0, 1.0, 7.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_SEPARATOR_COLS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BOUNDARY_WIDTH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_HIGH_DEGREE_COLS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_REGULARIZATION_REL", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-12",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_block_schur_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_symbolic_block_schur_lu"
    assert pc.metadata["architecture"] == "active_true_operator_symbolic_separator_schur_lu"
    assert pc.metadata["symbolic_factor_kind"] == "symbolic_block_schur_lu"
    assert pc.metadata["separator_size"] == 1
    assert pc.metadata["tail_size"] == 1
    assert pc.metadata["admission"]["accepted"] is True
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    recovered = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(recovered, np.linalg.solve(matrix.toarray(), rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_symbolic_block_schur_lu_admission_rejects_missing_interior_coupling(
    monkeypatch,
) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.csr_matrix(
        [
            [4.0, 1.0, 30.0, 0.0, 2.0],
            [1.0, 3.0, 0.0, 0.0, -1.0],
            [25.0, 0.0, 5.0, -1.0, 1.5],
            [0.0, 0.0, -1.0, 2.0, 0.5],
            [3.0, -2.0, 1.0, 1.0, 7.0],
        ],
        dtype=np.float64,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ORDERING", "natural")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BLOCK_SIZE", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_MAX_SEPARATOR_COLS", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_BOUNDARY_WIDTH", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_HIGH_DEGREE_COLS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_REGULARIZATION_REL", "0")
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MAX_RELATIVE_RESIDUAL",
        "1e-2",
    )
    monkeypatch.setenv(
        "SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_BLOCK_SCHUR_ADMISSION_MIN_IMPROVEMENT",
        "10",
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_block_schur_lu",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.kind == "active_symbolic_block_schur_lu"
    assert pc.reason.startswith("active_symbolic_block_schur_lu_admission_failed:")
    assert "max_rel=" in pc.reason
    assert "min_improvement=" in pc.reason
    assert pc.metadata["admission"]["accepted"] is False
    assert pc.metadata["admission"]["max_relative_residual"] > 1.0e-2
    assert pc.metadata["requires_preflight"] is True


def test_active_symbolic_coupled_schur_can_use_coupled_kinetic_factor_base(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.2, -0.3, 0.4, 0.1],
            [0.2, 2.8, -0.2, 0.5],
            [0.6, 0.1, 3.5, -0.4],
            [-0.1, 0.3, 0.2, 2.7],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.12, -0.08], [0.05, 0.11], [-0.18, 0.16], [0.10, -0.06]], dtype=np.float64)
    v = np.asarray([[0.07, -0.10, 0.18, -0.14], [-0.15, 0.05, -0.07, 0.20]], dtype=np.float64)
    w = np.asarray([[1.5, 0.04], [-0.02, 1.6]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_BASE", "active_coupled_kinetic_block")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_COARSE_SIZE", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_IDENTITY_COLUMNS", "8")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_coupled_schur",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_symbolic_coupled_schur"
    assert pc.metadata["base_kind"] == "active_coupled_kinetic_block"
    base_metadata = pc.metadata["base_preconditioner"]["metadata"]
    assert base_metadata["architecture"] == "active_dominant_kinetic_sparse_coupled_factor"
    assert base_metadata["block_covers_active"] is True
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-10


def test_active_symbolic_coupled_schur_uses_symbolic_kinetic_and_tail_space(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    kinetic = np.asarray(
        [
            [3.2, -0.3, 0.4, 0.1],
            [0.2, 2.8, -0.2, 0.5],
            [0.6, 0.1, 3.5, -0.4],
            [-0.1, 0.3, 0.2, 2.7],
        ],
        dtype=np.float64,
    )
    u = np.asarray([[0.12, -0.08], [0.05, 0.11], [-0.18, 0.16], [0.10, -0.06]], dtype=np.float64)
    v = np.asarray([[0.07, -0.10, 0.18, -0.14], [-0.15, 0.05, -0.07, 0.20]], dtype=np.float64)
    w = np.asarray([[1.5, 0.04], [-0.02, 1.6]], dtype=np.float64)
    matrix = sp.bmat(
        [[sp.csr_matrix(kinetic), sp.csr_matrix(u)], [sp.csr_matrix(v), sp.csr_matrix(w)]],
        format="csr",
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_COARSE_SIZE", "16")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_IDENTITY_COLUMNS", "8")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_symbolic_coupled_schur",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_symbolic_coupled_schur"
    assert pc.metadata["architecture"] == "active_symbolic_native_base_true_lsq_schur"
    assert pc.metadata["symbolic_ordering"]["kinetic_size"] == 4
    assert pc.metadata["symbolic_kinetic_basis_columns"] == 4
    assert pc.metadata["requires_preflight"] is True
    rhs = _deterministic_vector(layout.total_size)
    residual = rhs - np.asarray(matrix @ np.asarray(pc.operator.matvec(rhs), dtype=np.float64))
    assert np.linalg.norm(residual) < 1.0e-8


def test_active_symbolic_coupled_schur_large_size_gate_fails_fast(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=2,
        total_size=6,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sp.eye(layout.total_size, dtype=np.float64, format="csr")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SYMBOLIC_SCHUR_MAX_ACTIVE_SIZE", "5")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(layout.total_size, dtype=np.int64),
        kind="active_symbolic_coupled_schur",
        max_factor_nbytes=2_000_000,
        regularization=0.0,
    )

    assert not pc.selected
    assert pc.reason == "active_symbolic_coupled_schur_size_exceeded:6>5"
    assert pc.metadata["active_size"] == 6
    assert pc.metadata["max_active_size"] == 5


def test_structured_full_csr_active_global_field_split_schur_gmres_reaches_true_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = rhs_v3_full_system(op)
    active = vd._transport_active_dof_indices(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=30,
        method="gmres",
        preconditioner="active_global_field_split_schur",
        preconditioner_max_block_inverse_nbytes=100_000_000,
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    preconditioner = result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "active_global_field_split_schur"
    assert preconditioner["metadata"]["architecture"] == "active_kinetic_global_field_split_schur"
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_active_projected_sparse_tail_coarse_solves_spanned_residual_equation() -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2.0, 0.5, 0.0, 0.0, 0.1],
                [0.2, 3.0, 0.4, 0.0, 0.0],
                [0.0, 0.3, 4.0, 0.1, -0.2],
                [0.1, 0.0, 0.2, 5.0, 0.3],
                [0.1, -0.2, 0.3, 0.4, 1.5],
            ],
            dtype=np.float64,
        )
    )

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_tail_sparse_coarse",
        max_factor_nbytes=1_000_000,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_tail_sparse_coarse"
    assert pc.metadata["base_kind"] == "active_diagonal_schur"
    assert pc.metadata["coarse_size"] == layout.total_size
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-11, atol=1.0e-11)

    xblock_pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_xblock_sparse_coarse",
        max_factor_nbytes=1_000_000,
    )
    assert xblock_pc.selected, xblock_pc.to_dict()
    assert xblock_pc.metadata["base_kind"] == "active_xblock"
    xblock_actual = np.asarray(xblock_pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(xblock_actual, x_true, rtol=1.0e-11, atol=1.0e-11)


def test_active_scaled_ilu_sparse_coarse_solves_spanned_residual_equation(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=4,
        n_theta=1,
        n_zeta=1,
        f_size=4,
        phi1_size=0,
        extra_size=1,
        total_size=5,
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    active = np.arange(layout.total_size, dtype=np.int64)
    matrix = sp.csr_matrix(
        np.asarray(
            [
                [2.0e-3, 0.5e-3, 0.0, 0.0, 0.1],
                [0.2e-1, 3.0e-1, 0.4e-1, 0.0, 0.0],
                [0.0, 0.3, 4.0, 0.1, -0.2],
                [0.1, 0.0, 0.2, 5.0e2, 0.3],
                [0.1, -0.2, 0.3, 0.4, 1.5e3],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_FACTOR_KIND", "lu")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCALED_ILU_DIAGONAL_SHIFT", "0")

    pc = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active,
        kind="active_scaled_ilu_sparse_coarse",
        max_factor_nbytes=1_000_000,
        regularization=0.0,
    )

    assert pc.selected, pc.to_dict()
    assert pc.kind == "active_tail_sparse_coarse"
    assert pc.metadata["requested_base_kind"] == "active_scaled_ilu"
    assert pc.metadata["architecture"] == "scaled_ilu_global_sparse_coarse"
    assert pc.metadata["base_preconditioner"]["kind"] == "active_scaled_lu"
    x_true = np.asarray([0.2, -0.4, 0.7, -0.3, 0.5], dtype=np.float64)
    rhs = np.asarray(matrix @ x_true, dtype=np.float64)
    x_actual = np.asarray(pc.operator.matvec(rhs), dtype=np.float64)
    np.testing.assert_allclose(x_actual, x_true, rtol=1.0e-11, atol=1.0e-11)


def test_active_projected_coarse_residual_improves_physical_one_step_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = np.asarray(rhs_v3_full_system(op), dtype=np.float64)
    active = vd._transport_active_dof_indices(op)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active_matrix = selection.matrix.tocsr()[active[:, None], active].tocsr()
    active_rhs = rhs[active]
    layout = RHS1BlockLayout.from_operator(op)

    jacobi = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=layout,
        active_indices=active,
        kind="jacobi",
        max_factor_nbytes=100_000_000,
    )
    coarse = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=active_matrix,
        layout=layout,
        active_indices=active,
        kind="active_coarse",
        max_factor_nbytes=100_000_000,
    )

    assert jacobi.selected and jacobi.operator is not None
    assert coarse.selected and coarse.operator is not None
    residual_jacobi = float(np.linalg.norm(active_rhs - np.asarray(active_matrix @ jacobi.operator.matvec(active_rhs))))
    residual_coarse = float(np.linalg.norm(active_rhs - np.asarray(active_matrix @ coarse.operator.matvec(active_rhs))))
    assert residual_coarse < 0.1 * residual_jacobi
    assert coarse.metadata["coarse_size"] > op.extra_size
    assert coarse.metadata["factor_nbytes_actual"] > 0


def test_structured_full_csr_active_ilu_and_coarse_lgmres_reach_shifted_true_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = rhs_v3_full_system(op)
    active = vd._transport_active_dof_indices(op)

    ilu_result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="lgmres",
        preconditioner="active_ilu",
        preconditioner_max_block_inverse_nbytes=100_000_000,
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert ilu_result.converged, ilu_result.to_dict()
    assert ilu_result.metadata["method"] == "lgmres"
    assert ilu_result.metadata["active_dof"] is True
    preconditioner = ilu_result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "active_spilu"
    assert preconditioner["metadata"]["factor_nbytes_actual"] > 0
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(ilu_result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(ilu_result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    coarse_result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="lgmres",
        preconditioner="active_coarse",
        preconditioner_max_block_inverse_nbytes=100_000_000,
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert coarse_result.converged, coarse_result.to_dict()
    preconditioner = coarse_result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "active_coarse"
    assert preconditioner["metadata"]["coarse_size"] > op.extra_size
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(coarse_result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(coarse_result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_structured_full_csr_active_low_l_schur_gmres_reaches_physical_true_residual(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_LOW_L_SCHUR_LMAX", "4")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs = rhs_v3_full_system(op)
    active = vd._transport_active_dof_indices(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="gmres",
        preconditioner="active_low_l_schur",
        preconditioner_max_block_inverse_nbytes=100_000_000,
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    preconditioner = result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "active_low_l_schur"
    assert preconditioner["metadata"]["lmax"] == 4
    assert preconditioner["metadata"]["coarse_size"] > 0
    assert preconditioner["metadata"]["factor_nbytes_actual"] <= 100_000_000
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_structured_full_csr_active_overlap_schwarz_gmres_reaches_shifted_true_residual(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "1")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = rhs_v3_full_system(op)
    active = vd._transport_active_dof_indices(op)

    result = solve_structured_rhs1_full_csr(
        op,
        rhs,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        method="gmres",
        preconditioner="active_overlap_schwarz",
        preconditioner_max_block_inverse_nbytes=100_000_000,
        active_indices=active,
        max_csr_nbytes=100_000_000,
    )

    assert result.converged, result.to_dict()
    preconditioner = result.metadata["preconditioner"]
    assert preconditioner["selected"] is True
    assert preconditioner["kind"] == "active_overlap_schwarz"
    assert preconditioner["metadata"]["lmax"] == 4
    assert preconditioner["metadata"]["radius"] == 1
    assert preconditioner["metadata"]["patch_count"] == op.n_species * op.n_x
    assert preconditioner["metadata"]["factor_nbytes_actual"] <= 100_000_000
    true_residual = np.asarray(rhs) - np.asarray(apply_v3_full_system_operator(op, jnp.asarray(result.x)))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_structured_full_csr_active_overlap_schwarz_is_memory_gated(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", "4")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "1")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    active = vd._transport_active_dof_indices(op)

    rejected = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=selection.matrix[active[:, None], active],
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_overlap_schwarz",
        max_factor_nbytes=1,
    )
    assert rejected.selected is False
    assert rejected.reason.startswith("active_overlap_schwarz_budget_exceeded:")

    accepted = build_active_projected_rhs1_full_csr_preconditioner(
        matrix=selection.matrix[active[:, None], active],
        layout=RHS1BlockLayout.from_operator(op),
        active_indices=active,
        kind="active_overlap_schwarz",
        max_factor_nbytes=100_000_000,
    )
    assert accepted.selected is True
    assert accepted.kind == "active_overlap_schwarz"
    assert accepted.metadata["patch_count"] == op.n_species * op.n_x
    assert accepted.metadata["factor_nbytes_actual"] > 0


def test_structured_full_csr_preconditioner_falls_back_to_jacobi_when_tail_is_large() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "include_phi1_linear_subset_tiny.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.25)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None

    preconditioner = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="auto",
        max_schur_size=1,
    )

    assert preconditioner.selected is True
    assert preconditioner.kind == "jacobi"
    assert preconditioner.reason == "auto_fallback_jacobi"


def test_structured_full_csr_block_schur_preconditioner_is_memory_gated() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None

    rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="block_schur",
        max_block_inverse_nbytes=1,
    )
    assert rejected.selected is False
    assert rejected.reason.startswith("block_schur_budget_exceeded:")

    accepted = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="block_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    assert accepted.selected is True
    assert accepted.kind == "block_schur"
    assert accepted.metadata["tail_size"] == op.extra_size
    assert accepted.metadata["block_inverse_nbytes_actual"] > 0

    xi_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xi_block_schur",
        max_block_inverse_nbytes=1,
    )
    assert xi_rejected.selected is False
    assert xi_rejected.reason.startswith("xi_block_schur_budget_exceeded:")

    xi_accepted = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xi_block_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    assert xi_accepted.selected is True
    assert xi_accepted.kind == "xi_block_schur"
    assert xi_accepted.metadata["tail_size"] == op.extra_size
    assert xi_accepted.metadata["block_inverse_nbytes_actual"] > 0

    x_xi_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="x_xi_block_schur",
        max_block_inverse_nbytes=1,
    )
    assert x_xi_rejected.selected is False
    assert x_xi_rejected.reason.startswith("x_xi_block_schur_budget_exceeded:")

    x_xi_accepted = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="x_xi_block_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    assert x_xi_accepted.selected is True
    assert x_xi_accepted.kind == "x_xi_block_schur"
    assert x_xi_accepted.metadata["tail_size"] == op.extra_size
    assert x_xi_accepted.metadata["block_inverse_nbytes_actual"] > 0

    xblock_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xblock_tz_low_l_schur",
        max_block_inverse_nbytes=1,
    )
    assert xblock_rejected.selected is False
    assert xblock_rejected.reason.startswith("xblock_tz_low_l_schur_budget_exceeded:")

    xblock_accepted = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xblock_tz_low_l_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    assert xblock_accepted.selected is True
    assert xblock_accepted.kind == "xblock_tz_low_l_schur"
    assert xblock_accepted.metadata["tail_size"] == op.extra_size
    assert xblock_accepted.metadata["selected_blocks"] > 0

    coarse_rejected = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xblock_tz_low_l_coarse_schur",
        max_block_inverse_nbytes=1,
    )
    assert coarse_rejected.selected is False
    assert coarse_rejected.reason.startswith("xblock_tz_low_l_coarse_schur_budget_exceeded:")

    coarse_accepted = build_structured_rhs1_full_csr_preconditioner(
        matrix=selection.matrix,
        layout=RHS1BlockLayout.from_operator(op),
        kind="xblock_tz_low_l_coarse_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    assert coarse_accepted.selected is True
    assert coarse_accepted.kind == "xblock_tz_low_l_coarse_schur"
    assert coarse_accepted.metadata["tail_size"] == op.extra_size
    assert coarse_accepted.metadata["coarse_size"] > op.extra_size
    assert coarse_accepted.metadata["coarse_basis"] == "flux_surface_low_l_angular_plus_tail"
    assert coarse_accepted.metadata["coarse_surface_mode_count"] > 1
    assert "constant" in coarse_accepted.metadata["coarse_surface_modes"]
    assert coarse_accepted.metadata["base_preconditioner"]["kind"] == "xblock_tz_low_l_schur"


def test_xblock_coarse_residual_preconditioner_improves_one_step_residual() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    op = full_system_operator_from_namelist(nml=nml, identity_shift=0.5)
    rhs = np.asarray(rhs_v3_full_system(op), dtype=np.float64)
    selection = select_structured_rhs1_full_csr_operator(op, max_csr_nbytes=100_000_000)
    assert selection.selected and selection.matrix is not None
    matrix = selection.matrix.tocsr()
    layout = RHS1BlockLayout.from_operator(op)

    base = build_structured_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="xblock_tz_low_l_schur",
        max_block_inverse_nbytes=100_000_000,
    )
    coarse = build_structured_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="xblock_tz_low_l_coarse_schur",
        max_block_inverse_nbytes=100_000_000,
    )

    assert base.selected and base.operator is not None
    assert coarse.selected and coarse.operator is not None
    x_base = np.asarray(base.operator.matvec(rhs), dtype=np.float64)
    x_coarse = np.asarray(coarse.operator.matvec(rhs), dtype=np.float64)
    residual_base = float(np.linalg.norm(rhs - np.asarray(matrix @ x_base, dtype=np.float64)))
    residual_coarse = float(np.linalg.norm(rhs - np.asarray(matrix @ x_coarse, dtype=np.float64)))
    assert residual_coarse <= residual_base * (1.0 + 1.0e-10)


def test_driver_structured_csr_solve_returns_standard_result() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_structured_csr(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        preconditioner="diagonal_schur",
        max_csr_nbytes=100_000_000,
    )

    assert result.metadata is not None
    assert result.metadata["solver_path"] == "structured_full_csr_host_gmres"
    assert float(result.gmres.residual_norm) <= 1.0e-10
    true_residual = np.asarray(result.rhs) - np.asarray(apply_v3_full_system_operator(result.op, result.x))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.gmres.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_driver_linear_gmres_structured_csr_solve_method_routes_to_host_csr() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="structured_csr",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["solver_path"] == "structured_full_csr_host_gmres"
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["solve_method_requested"] == "structured_csr"
    assert result.metadata["requested_solve_method"] == "structured_csr"
    assert result.metadata["differentiable"] is False
    assert result.metadata["residual_kind"] == "true_residual"
    assert result.metadata["acceptance_criterion"] == "true_residual"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] in {"diagonal_schur", "xblock_tz_low_l_schur"}
    assert result.metadata["csr_nnz"] > 0
    assert result.metadata["csr_operator_nbytes"] > 0
    assert float(result.gmres.residual_norm) <= 1.0e-10


def test_driver_linear_gmres_structured_csr_native_xell_env_probe(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "gmres")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER", "native_xell")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", "128")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="host_structured_csr",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] == "native_xell"
    assert result.metadata["structured_full_csr_env"]["krylov"] == "gmres"
    assert result.metadata["structured_full_csr_env"]["active_dof"] is False
    pc = result.metadata["structured_full_csr"]["metadata"]["preconditioner"]
    assert pc["kind"] == "native_xell"
    assert pc["metadata"]["native_factor_available"] is True
    assert pc["metadata"]["backend"] == "jax_native_x_ell"
    true_residual = np.asarray(result.rhs) - np.asarray(apply_v3_full_system_operator(result.op, result.x))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.gmres.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_driver_linear_gmres_structured_csr_native_xell_tail_schur_env_probe(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "gmres")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER", "native_xell_tail_schur")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", "128")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="host_structured_csr",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] == "native_xell_tail_schur"
    assert result.metadata["structured_full_csr_env"]["krylov"] == "gmres"
    assert result.metadata["structured_full_csr_env"]["active_dof"] is False
    pc = result.metadata["structured_full_csr"]["metadata"]["preconditioner"]
    assert pc["kind"] == "native_xell_tail_schur"
    assert pc["metadata"]["native_factor_available"] is True
    assert pc["metadata"]["backend"] == "jax_native_x_ell_tail_schur"
    true_residual = np.asarray(result.rhs) - np.asarray(apply_v3_full_system_operator(result.op, result.x))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.gmres.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_driver_linear_gmres_structured_csr_active_direct_env_reports_factor_metadata(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", "direct")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", "1")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.0,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="host_structured_csr",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] == "none"
    assert result.metadata["structured_active_dof"] is True
    assert result.metadata["structured_active_size"] <= result.metadata["structured_full_size"]
    assert result.metadata["structured_active_size"] > 0
    assert result.metadata["direct_factor_s"] is not None
    assert result.metadata["direct_factor_nbytes_actual"] is not None
    assert result.metadata["sparse_pc_factor_nbytes_estimate"] == result.metadata["direct_factor_nbytes_actual"]
    assert result.metadata["structured_full_csr_env"]["krylov"] == "direct"
    assert result.metadata["structured_full_csr_env"]["active_dof"] is True
    true_residual = np.asarray(result.rhs) - np.asarray(apply_v3_full_system_operator(result.op, result.x))
    np.testing.assert_allclose(
        float(np.linalg.norm(true_residual)),
        float(result.gmres.residual_norm),
        rtol=1.0e-10,
        atol=1.0e-12,
    )


def test_driver_linear_gmres_structured_csr_physical_default_is_active_direct(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_KRYLOV", raising=False)
    monkeypatch.delenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_DOF", raising=False)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.0,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="host_structured_csr",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] == "none"
    assert result.metadata["structured_active_dof"] is True
    assert result.metadata["structured_full_csr_env"]["krylov"] == "direct"
    assert result.metadata["structured_full_csr_env"]["active_dof"] is True


def test_driver_linear_gmres_auto_selects_structured_csr_when_policy_allows(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_PRECONDITIONER_MAX_MB", "128")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="auto",
        differentiable=False,
    )

    assert result.metadata is not None
    assert result.metadata["auto_solver_selected"] is True
    assert result.metadata["auto_solver_policy"] == "structured_full_csr"
    assert result.metadata["solver_path"] == "structured_full_csr_host_gmres"
    assert result.metadata["solver_kind"] == "structured_full_csr"
    assert result.metadata["solve_method_requested"] == "auto"
    assert result.metadata["accepted_converged"] is True
    assert result.metadata["preconditioner_kind"] in {"diagonal_schur", "xblock_tz_low_l_schur"}
    assert result.metadata["csr_nnz"] > 0
    assert result.metadata["csr_operator_nbytes"] > 0
    assert float(result.gmres.residual_norm) <= 1.0e-10


def test_driver_linear_gmres_auto_leaves_structured_csr_disabled_by_default(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.delenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", "1")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="auto",
        differentiable=False,
    )

    assert result.metadata is None or result.metadata.get("solver_path") != "structured_full_csr_host_gmres"
    assert float(result.gmres.residual_norm) <= 1.0e-8 * max(float(np.linalg.norm(np.asarray(result.rhs))), 1.0)


def test_driver_linear_gmres_auto_structured_csr_memory_cap_falls_back(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_STRUCTURED_CSR_AUTO_MIN_NXI", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", "0")
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    result = vd.solve_v3_full_system_linear_gmres(
        nml=nml,
        identity_shift=0.5,
        tol=1.0e-8,
        atol=1.0e-10,
        restart=80,
        maxiter=20,
        solve_method="auto",
        differentiable=False,
    )

    assert result.metadata is None or result.metadata.get("solver_path") != "structured_full_csr_host_gmres"
    assert float(result.gmres.residual_norm) <= 1.0e-8 * max(float(np.linalg.norm(np.asarray(result.rhs))), 1.0)


def test_driver_linear_gmres_structured_csr_rejects_differentiable_true() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")

    try:
        vd.solve_v3_full_system_linear_gmres(
            nml=nml,
            identity_shift=0.5,
            solve_method="structured_csr",
            differentiable=True,
        )
    except ValueError as exc:
        assert "host-only/non-differentiable" in str(exc)
    else:  # pragma: no cover - keeps the failure message explicit.
        raise AssertionError("structured_csr must reject differentiable=True")


def test_driver_linear_gmres_structured_csr_memory_cap_fails_closed(monkeypatch) -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_MAX_MB", "0")

    try:
        vd.solve_v3_full_system_linear_gmres(
            nml=nml,
            identity_shift=0.5,
            solve_method="structured_csr",
            differentiable=False,
        )
    except RuntimeError as exc:
        message = str(exc)
        assert "structured full CSR solve was not selected" in message
        assert ("csr_budget_preflight_exceeded:" in message) or ("csr_budget_exceeded:" in message)
    else:  # pragma: no cover - keeps the failure message explicit.
        raise AssertionError("explicit structured_csr must fail closed when the CSR cap is exceeded")


def test_driver_structured_csr_solve_rejects_unsupported_phi1_in_kinetic() -> None:
    clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    nml = read_sfincs_input(REF / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist")

    try:
        vd.solve_v3_full_system_structured_csr(
            nml=nml,
            tol=1.0e-8,
            atol=1.0e-10,
            maxiter=1,
            max_csr_nbytes=100_000_000,
        )
    except RuntimeError as exc:
        assert "unsupported_phi1_in_kinetic" in str(exc)
    else:  # pragma: no cover - keeps the failure message explicit.
        raise AssertionError("unsupported Phi1-in-kinetic input should fail closed")
