from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp

import sfincs_jax.operators.profile_full_system as legacy
import sfincs_jax.solvers.preconditioner_xblock_active as active_projected
import sfincs_jax.solvers.preconditioner_xblock_policy as xblock_policy
from sfincs_jax.operators.profile_layout import RHS1BlockLayout


def _layout() -> RHS1BlockLayout:
    n_species = 1
    n_x = 2
    n_xi = 2
    n_theta = 2
    n_zeta = 1
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    return RHS1BlockLayout(
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


def _block_matrix(layout: RHS1BlockLayout) -> sp.csr_matrix:
    block_size = int(layout.n_xi * layout.n_theta * layout.n_zeta)
    blocks = []
    for x in range(int(layout.n_x)):
        dense = np.diag(2.0 + 0.25 * x + 0.1 * np.arange(block_size, dtype=np.float64))
        dense += 0.03 * np.tril(np.ones((block_size, block_size), dtype=np.float64), k=-1)
        dense += 0.02 * np.triu(np.ones((block_size, block_size), dtype=np.float64), k=1)
        blocks.append(sp.csc_matrix(dense))
    return sp.block_diag(blocks, format="csr")


def test_rhs1_full_assembly_keeps_legacy_aliases_to_active_xblock_owner() -> None:
    assert (
        legacy._build_active_projected_xblock_preconditioner
        is active_projected.build_active_projected_xblock_preconditioner
    )
    assert (
        legacy._build_active_projected_overlap_schwarz_preconditioner
        is active_projected.build_active_projected_overlap_schwarz_preconditioner
    )
    assert legacy._active_positions_for_full_indices is active_projected.active_positions_for_full_indices


def test_active_positions_for_full_indices_deduplicates_and_maps_unsorted_active_indices() -> None:
    active = np.asarray([10, 4, 7, 2, 9], dtype=np.int64)
    full = np.asarray([9, 4, 9, 11, 2], dtype=np.int64)

    positions = active_projected.active_positions_for_full_indices(
        active_indices=active,
        full_indices=full,
    )

    np.testing.assert_array_equal(positions, np.asarray([1, 3, 4], dtype=np.int64))


def test_active_positions_for_full_indices_handles_empty_and_absent_matches() -> None:
    np.testing.assert_array_equal(
        active_projected.active_positions_for_full_indices(
            active_indices=[],
            full_indices=[1, 2],
        ),
        np.zeros((0,), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        active_projected.active_positions_for_full_indices(
            active_indices=[3, 5, 7],
            full_indices=[],
        ),
        np.zeros((0,), dtype=np.int64),
    )
    np.testing.assert_array_equal(
        active_projected.active_positions_for_full_indices(
            active_indices=[3, 5, 7],
            full_indices=[1, 2, 4],
        ),
        np.zeros((0,), dtype=np.int64),
    )


def test_active_projected_low_level_policy_helpers_fail_closed(monkeypatch) -> None:
    matrix = sp.csr_matrix(np.asarray([[1.0, 0.0], [2.0, 3.0]], dtype=np.float64))
    expected_nbytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    assert active_projected._scipy_csr_nbytes(matrix) == expected_nbytes

    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_INT", "bad")
    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_FLOAT", "bad")
    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_BOOL", "off")
    assert active_projected._env_int("SFINCS_JAX_UNIT_TEST_INT", 17) == 17
    assert active_projected._env_float("SFINCS_JAX_UNIT_TEST_FLOAT", 2.5) == 2.5
    assert active_projected._env_bool("SFINCS_JAX_UNIT_TEST_BOOL", True) is False

    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_INT", "23")
    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_FLOAT", "1.25")
    monkeypatch.setenv("SFINCS_JAX_UNIT_TEST_BOOL", "yes")
    assert active_projected._env_int("SFINCS_JAX_UNIT_TEST_INT", 17) == 23
    assert active_projected._env_float("SFINCS_JAX_UNIT_TEST_FLOAT", 2.5) == 1.25
    assert active_projected._env_bool("SFINCS_JAX_UNIT_TEST_BOOL", False) is True

    monkeypatch.delenv("SFINCS_JAX_UNIT_TEST_BOOL", raising=False)
    assert active_projected._env_bool("SFINCS_JAX_UNIT_TEST_BOOL", True) is True


def test_xblock_policy_tuning_parsers_fail_closed_and_accept_overrides() -> None:
    tuning = xblock_policy.rhs1_xblock_local_solve_tuning(
        drop_tol_env_value="bad",
        drop_rel_env_value="0.125",
        ilu_drop_tol_env_value="1e-4",
        fill_factor_env_value="0.25",
        row_nnz_cap_env_value="-1",
        compact_row_nnz_cap_env_value="17",
    )
    lower = xblock_policy.rhs1_xblock_lower_fill_local_solve_tuning(
        drop_tol_env_value="2e-3",
        drop_rel_env_value="bad",
        ilu_drop_tol_env_value="3e-4",
        fill_factor_env_value="2.5",
        row_nnz_cap_env_value="19",
        compact_row_nnz_cap_env_value="bad",
    )

    assert tuning.drop_rel == 0.125
    assert tuning.ilu_drop_tol == 1.0e-4
    assert tuning.fill_factor >= 1.0
    assert tuning.row_nnz_cap >= 0
    assert tuning.compact_row_nnz_cap == 17
    assert lower.drop_tol == 2.0e-3
    assert lower.ilu_drop_tol == 3.0e-4
    assert lower.fill_factor == 2.5
    assert lower.row_nnz_cap == 19
    assert xblock_policy.rhs1_xblock_side_probe_min_active_size("") > 0
    assert xblock_policy.rhs1_xblock_side_probe_min_active_size("bad") > 0
    assert xblock_policy.rhs1_xblock_side_probe_min_active_size("-5") == 0
    assert xblock_policy.rhs1_xblock_side_probe_min_active_size("42") == 42


def test_active_projected_xblock_solves_exact_block_system(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    rhs = np.linspace(-0.5, 0.75, int(layout.total_size), dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_xblock_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_xblock",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "active_xblock"
    assert preconditioner.metadata["base_mode"] == "zero"
    assert preconditioner.metadata["block_count"] == layout.n_species * layout.n_x
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_projected_xblock_rejects_too_small_factor_budget(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_XBLOCK_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_xblock_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_xblock",
        regularization=0.0,
        max_factor_nbytes=1,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason.startswith("active_xblock_budget_exceeded:")
    assert preconditioner.metadata["max_factor_nbytes"] == 1


def test_active_projected_overlap_schwarz_solves_exact_radius_zero_blocks(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    rhs = np.cos(np.arange(int(layout.total_size), dtype=np.float64))
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_overlap_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_overlap_schwarz",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "active_overlap_schwarz"
    assert preconditioner.metadata["patch_count"] == layout.n_species * layout.n_x
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-11, atol=1.0e-11)


def test_active_projected_overlap_schwarz_rejects_too_small_factor_budget(monkeypatch) -> None:
    layout = _layout()
    matrix = _block_matrix(layout)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_RADIUS", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_SCHWARZ_LMAX", str(layout.n_xi))

    preconditioner = active_projected.build_active_projected_overlap_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_overlap_schwarz",
        regularization=0.0,
        max_factor_nbytes=1,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason.startswith("active_overlap_schwarz_budget_exceeded:")
    assert preconditioner.metadata["max_factor_nbytes"] == 1


def test_active_field_split_stack_builders_fail_closed_on_active_size_mismatch() -> None:
    layout = _layout()
    matrix = sp.eye(2, format="csr", dtype=np.float64)
    active_indices = np.asarray([0], dtype=np.int64)

    global_schur = active_projected.build_active_projected_global_field_split_schur_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_global_field_split_schur",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    diagonal_schur = active_projected.build_active_projected_diagonal_schur_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_diagonal_schur",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    multiline = active_projected.build_active_projected_multiline_field_split_base_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    bounded_stack = active_projected.build_active_projected_bounded_native_stack_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_bounded_native_stack",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    fortran_named_stack = active_projected.build_active_fortran_v3_reduced_native_stack_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=active_indices,
        requested_kind="active_fortran_v3_reduced_native_stack",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )

    assert global_schur.selected is False
    assert global_schur.reason == "active_index_size_mismatch"
    assert diagonal_schur.selected is False
    assert diagonal_schur.reason == "active_index_size_mismatch"
    assert multiline.selected is False
    assert multiline.reason.startswith("active_multiline_base_failed:")
    assert bounded_stack.selected is False
    assert bounded_stack.reason == "active_index_size_mismatch"
    assert fortran_named_stack.selected is False
    assert fortran_named_stack.reason == "active_index_size_mismatch"
    assert fortran_named_stack.metadata["production_candidate"] is True


def test_active_line_builders_solve_identity_system_and_reject_bad_active_size() -> None:
    layout = _layout()
    matrix = sp.eye(int(layout.total_size), format="csr", dtype=np.float64)
    active_indices = np.arange(int(layout.total_size), dtype=np.int64)
    rhs = np.linspace(-1.0, 1.0, int(layout.total_size), dtype=np.float64)

    xell = active_projected.build_active_projected_xell_kinetic_line_preconditioner(
        matrix=matrix,
        layout=layout,
        active_kinetic_indices=active_indices,
        requested_kind="active_native_xell",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    angular = active_projected.build_active_projected_angular_line_preconditioner(
        matrix=matrix,
        layout=layout,
        active_kinetic_indices=active_indices,
        requested_kind="active_angular_line",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )

    assert xell.selected, xell.to_dict()
    assert angular.selected, angular.to_dict()
    assert xell.operator is not None
    assert angular.operator is not None
    np.testing.assert_allclose(xell.operator.matvec(rhs), rhs, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(angular.operator.matvec(rhs), rhs, rtol=0.0, atol=0.0)
    assert xell.metadata["fixed_axes"] == ("species", "theta", "zeta")
    assert angular.metadata["fixed_axes"] == ("species", "x", "ell")

    bad_indices = np.asarray([0], dtype=np.int64)
    xell_bad = active_projected.build_active_projected_xell_kinetic_line_preconditioner(
        matrix=sp.eye(2, format="csr", dtype=np.float64),
        layout=layout,
        active_kinetic_indices=bad_indices,
        requested_kind="active_native_xell",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    angular_bad = active_projected.build_active_projected_angular_line_preconditioner(
        matrix=sp.eye(2, format="csr", dtype=np.float64),
        layout=layout,
        active_kinetic_indices=bad_indices,
        requested_kind="active_angular_line",
        regularization=0.0,
        max_factor_nbytes=1_000_000,
        t0=time.perf_counter(),
    )
    assert xell_bad.selected is False
    assert xell_bad.reason == "active_kinetic_index_size_mismatch"
    assert angular_bad.selected is False
    assert angular_bad.reason == "active_index_size_mismatch"


def test_active_native_indexed_schwarz_solves_diagonal_active_system(monkeypatch) -> None:
    layout = _layout()
    diagonal = 2.0 + 0.25 * np.arange(int(layout.total_size), dtype=np.float64)
    matrix = sp.diags(diagonal, format="csr")
    rhs = np.linspace(0.5, 1.5, int(layout.total_size), dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MIN_BLOCK_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_MAX_BLOCK_SIZE", "8")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_NORMALIZE_OVERLAP", "true")

    preconditioner = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected, preconditioner.to_dict()
    assert preconditioner.operator is not None
    assert preconditioner.kind == "active_native_indexed_schwarz"
    assert preconditioner.metadata["architecture"] == "jax_native_overlapping_indexed_line_blocks"
    assert preconditioner.metadata["block_family_counts"] == {"angular": 4, "xell": 2}
    solution = np.asarray(preconditioner.operator.matvec(rhs), dtype=np.float64)
    residual = rhs - np.asarray(matrix @ solution, dtype=np.float64)
    np.testing.assert_allclose(residual, np.zeros_like(rhs), rtol=1.0e-12, atol=1.0e-12)


def test_active_native_indexed_schwarz_rejection_paths_are_fail_closed() -> None:
    layout = _layout()

    nonsquare = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=sp.csr_matrix((2, 3), dtype=np.float64),
        layout=layout,
        active_indices=np.arange(2, dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=1000,
        t0=time.perf_counter(),
    )
    assert nonsquare.selected is False
    assert nonsquare.reason == "matrix_not_square"

    size_mismatch = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=sp.eye(2, format="csr", dtype=np.float64),
        layout=layout,
        active_indices=np.asarray([0], dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=1000,
        t0=time.perf_counter(),
    )
    assert size_mismatch.selected is False
    assert size_mismatch.reason == "active_index_size_mismatch"

    empty = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=sp.csr_matrix((0, 0), dtype=np.float64),
        layout=layout,
        active_indices=np.zeros((0,), dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=1000,
        t0=time.perf_counter(),
    )
    assert empty.selected is False
    assert empty.reason == "empty_active_space"

    out_of_range = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=sp.eye(1, format="csr", dtype=np.float64),
        layout=layout,
        active_indices=np.asarray([int(layout.total_size)], dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=1000,
        t0=time.perf_counter(),
    )
    assert out_of_range.selected is False
    assert out_of_range.reason == "active_indices_outside_full_vector"


def test_active_native_indexed_schwarz_empty_block_space_and_local_dispatch(monkeypatch) -> None:
    layout = _layout()
    matrix = sp.eye(int(layout.total_size), format="csr", dtype=np.float64)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_INCLUDE_XELL", "false")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_NATIVE_INDEXED_INCLUDE_ANGULAR", "false")

    preconditioner = active_projected.build_active_projected_native_indexed_schwarz_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        requested_kind="active_native_indexed_schwarz",
        regularization=0.0,
        max_factor_nbytes=100_000_000,
        t0=time.perf_counter(),
    )

    assert preconditioner.selected is False
    assert preconditioner.reason == "empty_active_native_indexed_block_space"
    assert preconditioner.metadata["include_xell"] is False
    assert preconditioner.metadata["include_angular"] is False

    dispatched = active_projected._build_active_projected_local_base_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        kind="active_native_indexed_lines",
        max_factor_nbytes=100_000_000,
        regularization=0.0,
        t0=time.perf_counter(),
    )
    assert dispatched.reason == "empty_active_native_indexed_block_space"

    unsupported = active_projected._build_active_projected_local_base_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.arange(int(layout.total_size), dtype=np.int64),
        kind="not_a_real_base",
        max_factor_nbytes=1000,
        regularization=0.0,
        t0=time.perf_counter(),
    )
    assert unsupported.selected is False
    assert unsupported.reason == "unsupported_local_base_kind"
