from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
import scipy.sparse as sp

import sfincs_jax.problems.transport_linear_system as tls
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


class _ToyProjectedFBlock:
    def __init__(self, *, n_zeta: int):
        self.n_zeta = int(n_zeta)

    def project_block_indices(self, active_blocks: np.ndarray) -> "_ToyProjectedFBlockMatrix":
        size = int(np.asarray(active_blocks).size) * int(self.n_zeta)
        return _ToyProjectedFBlockMatrix(size=size)


class _ToyProjectedFBlockMatrix:
    def __init__(self, *, size: int):
        self.size = int(size)

    def to_scipy_csr_matrix(self):
        diagonal = np.linspace(3.0, 4.0, self.size, dtype=np.float64)
        return sp.diags(diagonal, 0, format="csr")


def _toy_transport_op(*, constraint_scheme: int = 2, rhs_mode: int = 2) -> SimpleNamespace:
    n_species = 1
    n_x = 2
    n_xi = 1
    n_theta = 1
    n_zeta = 2
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    extra_size = 2
    return SimpleNamespace(
        rhs_mode=rhs_mode,
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=0,
        extra_size=extra_size,
        total_size=f_size + extra_size,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        constraint_scheme=constraint_scheme,
        point_at_x0=False,
        theta_weights=jnp.ones((n_theta,), dtype=jnp.float64),
        zeta_weights=jnp.asarray([0.25, 0.75], dtype=jnp.float64),
        d_hat=jnp.ones((n_theta, n_zeta), dtype=jnp.float64),
        x=jnp.asarray([0.5, 1.0], dtype=jnp.float64),
        x_weights=jnp.asarray([0.4, 0.6], dtype=jnp.float64),
        fblock=SimpleNamespace(fp=object()),
    )


@pytest.mark.parametrize("constraint_scheme", [1, 2])
def test_direct_reduced_pmat_emits_term_level_operator_for_constraint_schemes(
    monkeypatch: pytest.MonkeyPatch,
    constraint_scheme: int,
) -> None:
    op = _toy_transport_op(constraint_scheme=constraint_scheme)

    def fake_select(_fblock, *, include_identity_shift: bool, require_complete: bool):
        assert include_identity_shift
        assert require_complete
        return SimpleNamespace(
            selected=True,
            assembly=SimpleNamespace(
                operator=_ToyProjectedFBlock(n_zeta=op.n_zeta),
                included_terms=("identity", "fp_collision"),
            ),
        )

    messages: list[str] = []
    monkeypatch.setattr(tls, "select_structured_rhs1_fblock_operator", fake_select)

    result = tls._try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op,
        active_indices=np.arange(op.total_size, dtype=np.int64),
        factor_dtype=np.dtype(np.float64),
        pc_shift=0.125,
        emit=lambda _level, message: messages.append(str(message)),
    )

    assert result is not None
    bundle, metadata = result
    assert bundle.matrix.shape == (op.total_size, op.total_size)
    assert metadata["direct_pmat"] is True
    assert metadata["direct_pmat_reason"] == "term_level_reduced_fortran_pmat"
    assert metadata["direct_pmat_active_size"] == op.total_size
    assert metadata["direct_pmat_kinetic_size"] == op.f_size
    assert metadata["direct_pmat_tail_size"] == op.extra_size
    assert metadata["direct_pmat_kinetic_nnz"] == op.f_size
    assert metadata["direct_pmat_source_nnz"] > 0
    assert metadata["direct_pmat_constraint_nnz"] > 0
    assert metadata["direct_pmat_included_terms"] == ("identity", "fp_collision")
    assert any("direct reduced Pmat selected" in message for message in messages)

    x = np.arange(1, op.total_size + 1, dtype=np.float64)
    np.testing.assert_allclose(bundle.operator @ x, np.asarray(bundle.matrix @ x).reshape((-1,)))

    true_result = tls._try_build_rhsmode23_fp_direct_active_operator_bundle(
        op=op,
        active_indices=np.arange(op.total_size, dtype=np.int64),
        factor_dtype=np.dtype(np.float64),
        emit=lambda _level, message: messages.append(str(message)),
    )
    assert true_result is not None
    true_bundle, true_metadata = true_result
    assert true_bundle.metadata.reason == "direct term-level active true FP operator emission"
    assert true_metadata["direct_true_operator"] is True
    assert true_metadata["direct_true_operator_active_size"] == op.total_size
    assert any("direct active true FP operator selected" in message for message in messages)


@pytest.mark.parametrize(
    "op_mutation, active_indices",
    [
        ({"rhs_mode": 1}, None),
        ({"include_phi1": True}, None),
        ({"include_phi1_in_kinetic": True}, None),
        ({"constraint_scheme": 9}, None),
        ({}, np.asarray([], dtype=np.int64)),
        ({}, np.asarray([0, 0, 4, 5], dtype=np.int64)),
        ({}, np.asarray([0, 1, 2, 99], dtype=np.int64)),
        ({}, np.asarray([0, 1, 2, 3], dtype=np.int64)),
        ({}, np.asarray([1, 2, 4, 5], dtype=np.int64)),
    ],
)
def test_direct_reduced_pmat_fails_closed_for_incompatible_layouts(
    op_mutation: dict[str, object],
    active_indices: np.ndarray | None,
) -> None:
    op = _toy_transport_op()
    for key, value in op_mutation.items():
        setattr(op, key, value)

    result = tls._try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op,
        active_indices=active_indices,
        factor_dtype=np.dtype(np.float64),
        pc_shift=0.0,
        emit=None,
    )

    assert result is None


def test_direct_reduced_pmat_reports_structured_fblock_selection_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    op = _toy_transport_op()
    messages: list[str] = []

    def fail_select(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic structured selection failure")

    monkeypatch.setattr(tls, "select_structured_rhs1_fblock_operator", fail_select)

    result = tls._try_build_rhsmode23_fp_fortran_reduced_direct_pmat_bundle(
        op_pc=op,
        active_indices=np.arange(op.total_size, dtype=np.int64),
        factor_dtype=np.dtype(np.float64),
        pc_shift=0.0,
        emit=lambda _level, message: messages.append(str(message)),
    )

    assert result is None
    assert any("direct reduced Pmat unavailable" in message for message in messages)
