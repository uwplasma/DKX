from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from sfincs_jax.problems.transport_linear_system import (
    admit_active_block_schur_factor,
    build_active_block_ordering,
    build_active_block_schur_factor,
    build_active_block_schur_residual_coarse_factor,
    deterministic_probe_matrix,
)


def test_active_block_ordering_supports_reusable_layouts() -> None:
    zeta = build_active_block_ordering(
        kinetic_size=24,
        tail_size=2,
        n_theta=3,
        n_zeta=4,
        block_kind="zeta_line",
        max_block_size=4,
    )
    assert zeta.block_kind == "zeta_line"
    assert len(zeta.blocks) == 6
    assert zeta.block_size_max == 4

    theta = build_active_block_ordering(
        kinetic_size=24,
        tail_size=2,
        n_theta=3,
        n_zeta=4,
        block_kind="theta_line",
        max_block_size=3,
    )
    assert theta.block_kind == "theta_line"
    assert len(theta.blocks) == 8
    np.testing.assert_array_equal(theta.blocks[0], np.asarray([0, 4, 8], dtype=np.int64))

    plane = build_active_block_ordering(
        kinetic_size=24,
        tail_size=2,
        n_theta=3,
        n_zeta=4,
        block_kind="ell_band",
        ell_block=2,
        max_block_size=24,
    )
    assert plane.block_kind == "ell_band"
    assert len(plane.blocks) == 1
    assert plane.block_size_max == 24


def test_active_block_ordering_rejects_oversized_blocks() -> None:
    with pytest.raises(MemoryError):
        build_active_block_ordering(
            kinetic_size=24,
            tail_size=0,
            n_theta=3,
            n_zeta=4,
            block_kind="angular_plane",
            max_block_size=8,
        )


def test_active_block_ordering_rejects_invalid_layout_contracts() -> None:
    invalid_cases = [
        dict(kinetic_size=0, tail_size=0, n_theta=1, n_zeta=1, block_kind="zeta_line"),
        dict(kinetic_size=4, tail_size=0, n_theta=0, n_zeta=1, block_kind="zeta_line"),
        dict(kinetic_size=5, tail_size=0, n_theta=2, n_zeta=3, block_kind="zeta_line"),
        dict(kinetic_size=5, tail_size=0, n_theta=2, n_zeta=3, block_kind="theta_line"),
        dict(kinetic_size=5, tail_size=0, n_theta=2, n_zeta=3, block_kind="angular_plane"),
        dict(kinetic_size=6, tail_size=0, n_theta=2, n_zeta=3, block_kind="unknown"),
    ]

    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            build_active_block_ordering(max_block_size=8, **kwargs)


def test_active_block_schur_factor_solves_exact_block_tail_system() -> None:
    k = np.asarray(
        [
            [4.0, 0.2, 0.0, 0.0],
            [0.1, 3.0, 0.0, 0.0],
            [0.0, 0.0, 5.0, 0.3],
            [0.0, 0.0, 0.2, 4.0],
        ],
        dtype=np.float64,
    )
    b = np.asarray([[1.0], [0.5], [-0.2], [0.3]], dtype=np.float64)
    c = np.asarray([[0.4, -0.1, 0.2, 0.3]], dtype=np.float64)
    d = np.asarray([[2.0]], dtype=np.float64)
    matrix = sp.bmat([[sp.csr_matrix(k), sp.csr_matrix(b)], [sp.csr_matrix(c), sp.csr_matrix(d)]], format="csr")
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=1,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )
    factor = build_active_block_schur_factor(matrix, ordering, reg=0.0, max_mb=1.0)

    rhs = np.asarray([1.0, 2.0, -1.0, 0.5, 3.0], dtype=np.float64)
    solution = factor.apply(rhs)
    np.testing.assert_allclose(matrix @ solution, rhs, rtol=1.0e-12, atol=1.0e-12)

    admission = admit_active_block_schur_factor(
        matrix,
        factor,
        deterministic_probe_matrix(active_size=5, kinetic_size=4, tail_size=1, count=3),
        max_relative_residual=1.0e-10,
        min_improvement_vs_identity=1.0,
    )
    assert admission.accepted
    assert admission.max_relative_residual < 1.0e-10


def test_active_block_schur_factor_rejects_bad_shape_and_memory_budget() -> None:
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=1,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )

    with pytest.raises(ValueError, match="matrix shape"):
        build_active_block_schur_factor(sp.eye(4, format="csr"), ordering, max_mb=1.0)

    with pytest.raises(MemoryError, match="active block-Schur factor estimate"):
        build_active_block_schur_factor(sp.eye(5, format="csr"), ordering, max_mb=1.0e-9)


def test_active_block_schur_admission_rejects_missing_strong_offblock_couplings() -> None:
    k = np.asarray(
        [
            [3.0, 0.0, 2.5, 0.0],
            [0.0, 3.0, 0.0, 2.5],
            [2.5, 0.0, 3.0, 0.0],
            [0.0, 2.5, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    matrix = sp.csr_matrix(k)
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=0,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )
    factor = build_active_block_schur_factor(matrix, ordering, reg=0.0, max_mb=1.0)
    probes = np.eye(4, dtype=np.float64)
    admission = admit_active_block_schur_factor(
        matrix,
        factor,
        probes,
        max_relative_residual=1.0e-3,
        min_improvement_vs_identity=1.0e6,
    )

    assert not admission.accepted
    assert admission.reason == "relative_residual_gate"
    assert admission.max_relative_residual > 1.0e-1


def test_active_block_schur_admission_rejects_insufficient_improvement() -> None:
    matrix = sp.eye(4, format="csr")
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=0,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )
    factor = build_active_block_schur_factor(matrix, ordering, reg=0.0, max_mb=1.0)
    admission = admit_active_block_schur_factor(
        matrix,
        factor,
        deterministic_probe_matrix(active_size=4, kinetic_size=4, tail_size=0, count=2),
        max_relative_residual=1.0e-10,
        min_improvement_vs_identity=2.0,
    )

    assert not admission.accepted
    assert admission.reason == "improvement_gate"


def test_residual_coarse_factor_repairs_ranked_offblock_residuals() -> None:
    k = np.asarray(
        [
            [3.0, 0.0, 2.5, 0.0],
            [0.0, 3.0, 0.0, 2.5],
            [2.5, 0.0, 3.0, 0.0],
            [0.0, 2.5, 0.0, 3.0],
        ],
        dtype=np.float64,
    )
    matrix = sp.csr_matrix(k)
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=0,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )
    base = build_active_block_schur_factor(matrix, ordering, reg=0.0, max_mb=1.0)
    probes = np.eye(4, dtype=np.float64)
    base_admission = admit_active_block_schur_factor(
        matrix,
        base,
        probes,
        max_relative_residual=1.0e-3,
        min_improvement_vs_identity=1.0,
    )
    assert not base_admission.accepted

    coarse = build_active_block_schur_residual_coarse_factor(
        matrix,
        base,
        probes,
        max_cols=4,
        regularization_rel=1.0e-14,
        max_mb=1.0,
    )
    admission = admit_active_block_schur_factor(
        matrix,
        coarse,
        probes,
        max_relative_residual=1.0e-10,
        min_improvement_vs_identity=1.0,
    )

    assert admission.accepted
    assert admission.max_relative_residual < 1.0e-10
    assert coarse.metadata["residual_coarse_cols"] == 4


def test_residual_coarse_factor_rejects_bad_or_rank_deficient_probes() -> None:
    matrix = sp.eye(4, format="csr")
    ordering = build_active_block_ordering(
        kinetic_size=4,
        tail_size=0,
        n_theta=1,
        n_zeta=2,
        block_kind="zeta_line",
        max_block_size=2,
    )
    factor = build_active_block_schur_factor(matrix, ordering, reg=0.0, max_mb=1.0)

    with pytest.raises(ValueError, match="probe length"):
        build_active_block_schur_residual_coarse_factor(
            matrix,
            factor,
            np.ones((3, 1), dtype=np.float64),
            max_mb=1.0,
        )

    with pytest.raises(ValueError, match="no finite candidate"):
        build_active_block_schur_residual_coarse_factor(
            matrix,
            factor,
            np.eye(4, dtype=np.float64),
            max_mb=1.0,
        )
