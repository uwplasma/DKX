from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
import time

import numpy as np
import scipy.sparse as sp

import sfincs_jax.operators.profile_full_system as rhs1_full_assembly
from sfincs_jax.solvers.preconditioner_schur_profile import (
    RHS1StructuredFullCSRPreconditioner,
    build_block_schur_preconditioner,
    build_diagonal_schur_preconditioner,
    build_jacobi_preconditioner,
    build_x_xi_diagonal_inverse_blocks,
    build_x_xi_block_schur_preconditioner,
    build_xi_diagonal_inverse_blocks,
    build_xi_block_schur_preconditioner,
    build_zeta_diagonal_inverse_blocks,
    estimate_x_xi_block_inverse_nbytes,
    estimate_xi_block_inverse_nbytes,
    estimate_zeta_block_inverse_nbytes,
    safe_inverse_diagonal,
)


@dataclass(frozen=True)
class TinyRHS1Layout:
    n_species: int = 1
    n_x: int = 2
    n_xi: int = 2
    n_theta: int = 1
    n_zeta: int = 2

    @property
    def f_size(self) -> int:
        return int(self.n_species * self.n_x * self.n_xi * self.n_theta * self.n_zeta)

    @property
    def total_size(self) -> int:
        return int(self.f_size + 2)

    def kinetic_flat_index(self, *, species: int, x: int, ell: int, theta: int, zeta: int) -> int:
        return int(((((species * self.n_x + x) * self.n_xi + ell) * self.n_theta + theta) * self.n_zeta) + zeta)


class TailFreeTinyRHS1Layout(TinyRHS1Layout):
    @property
    def total_size(self) -> int:
        return int(self.f_size)


def _tail_coupled_diagonal_kinetic_matrix(layout: TinyRHS1Layout) -> sp.csr_matrix:
    diag_f = np.linspace(3.0, 6.5, layout.f_size)
    u = np.zeros((layout.f_size, 2), dtype=np.float64)
    v = np.zeros((2, layout.f_size), dtype=np.float64)
    u[0, 0] = 0.20
    u[3, 1] = -0.15
    u[6, 0] = 0.05
    v[0, 1] = -0.10
    v[1, 4] = 0.25
    v[1, 7] = -0.05
    w = np.array([[2.4, 0.10], [0.20, 1.9]], dtype=np.float64)
    return sp.bmat(
        [
            [sp.diags(diag_f, format="csr"), sp.csr_matrix(u)],
            [sp.csr_matrix(v), sp.csr_matrix(w)],
        ],
        format="csr",
    )


def _tail_only_coupled_matrix(layout: TinyRHS1Layout) -> sp.csr_matrix:
    diag_f = np.linspace(3.0, 6.5, layout.f_size)
    u = np.zeros((layout.f_size, 2), dtype=np.float64)
    v = np.zeros((2, layout.f_size), dtype=np.float64)
    v[0, 1] = -0.10
    v[1, 4] = 0.25
    w = np.array([[2.4, 0.10], [0.20, 1.9]], dtype=np.float64)
    return sp.bmat(
        [
            [sp.diags(diag_f, format="csr"), sp.csr_matrix(u)],
            [sp.csr_matrix(v), sp.csr_matrix(w)],
        ],
        format="csr",
    )


def test_safe_inverse_diagonal_regularizes_small_entries() -> None:
    inv, metadata = safe_inverse_diagonal(np.array([0.0, 2.0, -4.0]), regularization=0.1)

    np.testing.assert_allclose(inv, np.array([2.5, 0.5, -0.25]))
    assert metadata["diagonal_size"] == 3
    assert metadata["diagonal_floor"] == 0.4
    assert metadata["diagonal_regularized_count"] == 1


def test_jacobi_preconditioner_records_regularized_diagonal_metadata() -> None:
    matrix = sp.diags([0.0, 2.0, -4.0], format="csr")
    pc = build_jacobi_preconditioner(
        matrix=matrix,
        requested_kind="unit_test_jacobi",
        regularization=0.1,
        t0=time.perf_counter(),
        reason="unit_test",
    )

    assert pc.selected
    assert pc.kind == "jacobi"
    assert pc.metadata["requested_kind"] == "unit_test_jacobi"
    assert pc.metadata["diagonal_regularized_count"] == 1
    np.testing.assert_allclose(pc.operator.matvec(np.ones(3)), np.array([2.5, 0.5, -0.25]))


def test_structured_full_csr_preconditioner_to_dict_is_json_friendly() -> None:
    pc = build_jacobi_preconditioner(
        matrix=sp.diags([1.0, 2.0], format="csr"),
        requested_kind="unit_test",
        regularization=0.0,
        t0=time.perf_counter(),
        reason="metadata_smoke",
    )

    payload = pc.to_dict()

    assert payload["selected"] is True
    assert payload["kind"] == "jacobi"
    assert payload["reason"] == "metadata_smoke"
    assert isinstance(payload["setup_s"], float)
    assert payload["metadata"]["requested_kind"] == "unit_test"


def test_diagonal_schur_without_tail_degrades_to_jacobi() -> None:
    matrix = sp.diags([1.0, 0.0, -4.0], format="csr")
    pc = build_diagonal_schur_preconditioner(
        matrix=matrix,
        layout=SimpleNamespace(f_size=3, total_size=3),
        requested_kind="unit_test_no_tail",
        regularization=0.25,
        t0=time.perf_counter(),
    )

    assert pc.kind == "jacobi"
    assert pc.reason == "no_global_tail"
    assert pc.metadata["requested_kind"] == "unit_test_no_tail"
    np.testing.assert_allclose(pc.operator.matvec(np.ones(3)), np.array([1.0, 1.0, -0.25]))


def test_full_csr_schur_builders_are_exact_for_diagonal_kinetic_block() -> None:
    layout = TinyRHS1Layout()
    matrix = _tail_coupled_diagonal_kinetic_matrix(layout)
    rhs = np.linspace(-0.3, 0.8, layout.total_size)
    expected = np.linalg.solve(matrix.toarray(), rhs)

    builders = [
        (build_diagonal_schur_preconditioner, "diagonal_schur"),
        (build_block_schur_preconditioner, "block_schur"),
        (build_xi_block_schur_preconditioner, "xi_block_schur"),
        (build_x_xi_block_schur_preconditioner, "x_xi_block_schur"),
    ]
    for builder, expected_kind in builders:
        pc = builder(
            matrix=matrix,
            layout=layout,
            requested_kind=f"unit_test_{expected_kind}",
            regularization=0.0,
            t0=time.perf_counter(),
        )
        assert pc.selected
        assert pc.kind == expected_kind
        assert pc.metadata["tail_size"] == 2
        assert pc.metadata["kinetic_size"] == layout.f_size
        np.testing.assert_allclose(pc.operator.matvec(rhs), expected, rtol=1e-12, atol=1e-12)


def test_full_csr_schur_builders_skip_empty_u_columns_and_regularize_tail() -> None:
    layout = TinyRHS1Layout()
    matrix = _tail_only_coupled_matrix(layout)
    rhs = np.linspace(-0.3, 0.8, layout.total_size)

    builders = [
        (build_diagonal_schur_preconditioner, "diagonal_schur"),
        (build_block_schur_preconditioner, "block_schur"),
        (build_xi_block_schur_preconditioner, "xi_block_schur"),
        (build_x_xi_block_schur_preconditioner, "x_xi_block_schur"),
    ]
    for builder, expected_kind in builders:
        pc = builder(
            matrix=matrix,
            layout=layout,
            requested_kind=f"unit_test_{expected_kind}_regularized",
            regularization=1.0e-3,
            t0=time.perf_counter(),
        )
        assert pc.selected
        assert pc.kind == expected_kind
        assert pc.metadata["tail_size"] == 2
        assert pc.metadata["schur_regularization"] > 0.0
        if expected_kind != "diagonal_schur":
            assert pc.metadata["active_u_columns"] == 0
        assert np.all(np.isfinite(pc.operator.matvec(rhs)))


def test_structured_block_schur_builders_apply_kinetic_inverse_without_tail() -> None:
    layout = TailFreeTinyRHS1Layout()
    diagonal = np.linspace(2.0, 9.0, layout.total_size)
    matrix = sp.diags(diagonal, format="csr")
    rhs = np.linspace(-0.4, 0.9, layout.total_size)
    expected = rhs / diagonal

    builders = [
        (build_block_schur_preconditioner, "block_schur"),
        (build_xi_block_schur_preconditioner, "xi_block_schur"),
        (build_x_xi_block_schur_preconditioner, "x_xi_block_schur"),
    ]
    for builder, expected_kind in builders:
        pc = builder(
            matrix=matrix,
            layout=layout,
            requested_kind=f"unit_test_{expected_kind}",
            regularization=0.0,
            t0=time.perf_counter(),
        )
        assert pc.selected
        assert pc.kind == expected_kind
        assert pc.reason == "no_global_tail"
        assert pc.metadata["tail_size"] == 0
        assert pc.metadata["kinetic_size"] == layout.f_size
        np.testing.assert_allclose(pc.operator.matvec(rhs), expected, rtol=1e-12, atol=1e-12)


def test_full_csr_block_memory_estimates_match_layout_sizes() -> None:
    layout = TinyRHS1Layout()

    assert estimate_zeta_block_inverse_nbytes(layout) == layout.f_size // layout.n_zeta * layout.n_zeta**2 * 8
    assert (
        estimate_xi_block_inverse_nbytes(layout)
        == layout.n_species * layout.n_x * layout.n_theta * layout.n_zeta * layout.n_xi**2 * 8
    )
    assert (
        estimate_x_xi_block_inverse_nbytes(layout)
        == layout.n_species * layout.n_theta * layout.n_zeta * (layout.n_x * layout.n_xi) ** 2 * 8
    )


def test_full_csr_block_inverse_builders_fail_closed_on_singular_blocks() -> None:
    layout = TinyRHS1Layout()
    singular_matrix = sp.csr_matrix((layout.total_size, layout.total_size), dtype=np.float64)

    zeta_inv, zeta_meta = build_zeta_diagonal_inverse_blocks(
        matrix=singular_matrix,
        n_f=layout.f_size,
        block_size=layout.n_zeta,
        regularization=0.0,
    )
    xi_inv, xi_indices, xi_meta = build_xi_diagonal_inverse_blocks(
        matrix=singular_matrix,
        layout=layout,
        regularization=0.0,
    )
    x_xi_inv, x_xi_indices, x_xi_meta = build_x_xi_diagonal_inverse_blocks(
        matrix=singular_matrix,
        layout=layout,
        regularization=0.0,
    )

    assert zeta_meta["block_inverse_singular_count"] == zeta_inv.shape[0]
    assert xi_meta["block_inverse_singular_count"] == xi_inv.shape[0]
    assert x_xi_meta["block_inverse_singular_count"] == x_xi_inv.shape[0]
    np.testing.assert_allclose(zeta_inv, np.zeros_like(zeta_inv))
    np.testing.assert_allclose(xi_inv, np.zeros_like(xi_inv))
    np.testing.assert_allclose(x_xi_inv, np.zeros_like(x_xi_inv))
    assert xi_indices.shape == (layout.n_species * layout.n_x * layout.n_theta * layout.n_zeta, layout.n_xi)
    assert x_xi_indices.shape == (layout.n_species * layout.n_theta * layout.n_zeta, layout.n_x * layout.n_xi)


def test_full_csr_block_inverse_builders_regularize_rank_deficient_blocks() -> None:
    layout = TinyRHS1Layout()
    singular_matrix = sp.csr_matrix((layout.total_size, layout.total_size), dtype=np.float64)

    zeta_inv, zeta_meta = build_zeta_diagonal_inverse_blocks(
        matrix=singular_matrix,
        n_f=layout.f_size,
        block_size=layout.n_zeta,
        regularization=1.0e-3,
    )
    xi_inv, _xi_indices, xi_meta = build_xi_diagonal_inverse_blocks(
        matrix=singular_matrix,
        layout=layout,
        regularization=1.0e-3,
    )
    x_xi_inv, _x_xi_indices, x_xi_meta = build_x_xi_diagonal_inverse_blocks(
        matrix=singular_matrix,
        layout=layout,
        regularization=1.0e-3,
    )

    assert zeta_meta["block_inverse_regularized_count"] == zeta_inv.shape[0]
    assert xi_meta["block_inverse_regularized_count"] == xi_inv.shape[0]
    assert x_xi_meta["block_inverse_regularized_count"] == x_xi_inv.shape[0]
    assert zeta_meta["block_inverse_singular_count"] == 0
    assert xi_meta["block_inverse_singular_count"] == 0
    assert x_xi_meta["block_inverse_singular_count"] == 0
    np.testing.assert_allclose(zeta_inv[0], np.eye(layout.n_zeta) / 1.0e-3)
    np.testing.assert_allclose(xi_inv[0], np.eye(layout.n_xi) / 1.0e-3)
    np.testing.assert_allclose(x_xi_inv[0], np.eye(layout.n_x * layout.n_xi) / 1.0e-3)


def test_rhs1_full_assembly_keeps_legacy_schur_aliases() -> None:
    assert rhs1_full_assembly.RHS1StructuredFullCSRPreconditioner is RHS1StructuredFullCSRPreconditioner
    assert rhs1_full_assembly._build_jacobi_preconditioner is build_jacobi_preconditioner
    assert rhs1_full_assembly._build_diagonal_schur_preconditioner is build_diagonal_schur_preconditioner
    assert rhs1_full_assembly._build_block_schur_preconditioner is build_block_schur_preconditioner
