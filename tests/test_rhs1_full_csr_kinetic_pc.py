from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from jax import jit
import jax.numpy as jnp

import sfincs_jax.solvers.preconditioner_full_fp_csr as full_csr
from sfincs_jax.solvers.native_block_factor import (
    apply_native_x_ell_kinetic_factor,
    build_native_x_ell_kinetic_factor,
)
from sfincs_jax.operators.profile_layout import RHS1BlockLayout
from sfincs_jax.solvers.preconditioner_full_fp_csr import (
    build_rhs1_full_csr_kinetic_preconditioner,
    estimate_rhs1_full_csr_kinetic_preconditioner_nbytes,
    rhs1_full_csr_x_ell_block_indices,
)


def _tiny_layout(*, rhs_mode: int = 1) -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=2,
        n_xi=2,
        n_theta=2,
        n_zeta=1,
        f_size=8,
        phi1_size=1,
        extra_size=1,
        total_size=10,
        constraint_scheme=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        rhs_mode=rhs_mode,
    )


def _layout_variant(**overrides) -> RHS1BlockLayout:
    data = {
        "n_species": 1,
        "n_x": 2,
        "n_xi": 2,
        "n_theta": 2,
        "n_zeta": 1,
        "f_size": 8,
        "phi1_size": 1,
        "extra_size": 1,
        "total_size": 10,
        "constraint_scheme": 1,
        "include_phi1": True,
        "include_phi1_in_kinetic": False,
        "rhs_mode": 1,
    }
    data.update(overrides)
    return RHS1BlockLayout(**data)


def _tiny_full_csr(layout: RHS1BlockLayout) -> sp.csr_matrix:
    dense = np.zeros((int(layout.total_size), int(layout.total_size)), dtype=np.float64)
    base_block = np.asarray(
        [
            [4.0, 0.30, -0.20, 0.10],
            [0.25, 3.5, 0.40, -0.15],
            [-0.10, 0.35, 5.0, 0.20],
            [0.05, -0.25, 0.30, 4.5],
        ],
        dtype=np.float64,
    )
    for block_id, indices in enumerate(rhs1_full_csr_x_ell_block_indices(layout)):
        dense[np.ix_(indices, indices)] = base_block + 0.2 * block_id * np.eye(4, dtype=np.float64)

    tail = int(layout.f_size)
    dense[tail, tail] = 3.0
    dense[tail + 1, tail + 1] = -5.0
    dense[tail, 0] = 0.7
    dense[1, tail + 1] = -0.4
    return sp.csr_matrix(dense)


def test_native_x_ell_factor_builder_applies_blocks_and_tail() -> None:
    factor = build_native_x_ell_kinetic_factor(
        block_inverses=jnp.asarray(
            [
                [[0.5, 0.0], [0.0, 0.25]],
                [[1.0 / 3.0, 0.0], [0.0, 0.2]],
            ],
            dtype=jnp.float64,
        ),
        block_indices=jnp.asarray([[0, 1], [2, 3]], dtype=jnp.int32),
        f_size=4,
        total_size=6,
        inv_tail=jnp.asarray([0.1, -0.5], dtype=jnp.float64),
    )

    rhs = jnp.asarray([2.0, 4.0, 9.0, 10.0, 30.0, -8.0], dtype=jnp.float64)
    expected = jnp.asarray([1.0, 1.0, 3.0, 2.0, 3.0, 4.0], dtype=jnp.float64)

    np.testing.assert_allclose(
        np.asarray(apply_native_x_ell_kinetic_factor(factor, rhs)),
        np.asarray(expected),
        rtol=0.0,
        atol=1.0e-14,
    )


def test_x_ell_preconditioner_applies_dense_kinetic_lines_and_tail_jacobi() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)
    rhs = np.linspace(-0.75, 1.25, int(layout.total_size), dtype=np.float64)

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
        regularization=0.0,
        build_native_factor=True,
    )

    assert preconditioner.selected is True, preconditioner.to_dict()
    assert preconditioner.reason == "complete"
    assert preconditioner.kind == "x_ell"
    assert preconditioner.operator is not None

    def expected_from_blocks(rhs_like: np.ndarray) -> np.ndarray:
        arr = np.asarray(rhs_like, dtype=np.float64)
        squeeze = arr.ndim == 1
        arr_2d = arr[:, None] if squeeze else arr
        out = np.zeros_like(arr_2d)
        for indices in rhs1_full_csr_x_ell_block_indices(layout):
            out[indices, :] = np.linalg.solve(matrix[indices[:, None], indices].toarray(), arr_2d[indices, :])
        out[int(layout.f_size) :, :] = arr_2d[int(layout.f_size) :, :] / matrix.diagonal()[
            int(layout.f_size) :
        ][:, None]
        return out[:, 0] if squeeze else out

    expected = expected_from_blocks(rhs)

    actual = preconditioner.apply(rhs)
    np.testing.assert_allclose(actual, expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(preconditioner.operator.matvec(rhs), expected, rtol=1.0e-13, atol=1.0e-13)
    np.testing.assert_allclose(preconditioner.apply_native(rhs), expected, rtol=1.0e-13, atol=1.0e-13)
    assert preconditioner.native_factor is not None
    np.testing.assert_allclose(
        jit(lambda vec: apply_native_x_ell_kinetic_factor(preconditioner.native_factor, vec))(rhs),
        expected,
        rtol=1.0e-13,
        atol=1.0e-13,
    )
    rhs_cols = np.stack([rhs, 0.25 * rhs + 0.1], axis=1)
    expected_cols = expected_from_blocks(rhs_cols)
    np.testing.assert_allclose(preconditioner.apply_native(rhs_cols), expected_cols, rtol=1.0e-13, atol=1.0e-13)

    scalar_jacobi = rhs / matrix.diagonal()
    assert not np.allclose(actual[: int(layout.f_size)], scalar_jacobi[: int(layout.f_size)])
    metadata = preconditioner.metadata
    assert metadata["line_axes"] == ("x", "ell")
    assert metadata["fixed_axes"] == ("species", "theta", "zeta")
    assert metadata["n_blocks"] == 2
    assert metadata["block_size"] == 4
    assert metadata["tail_policy"] == "jacobi"
    assert metadata["native_factor_available"] is True
    assert metadata["candidate_nbytes_actual"] == metadata["candidate_nbytes_estimate"]
    assert metadata["candidate_nbytes_estimate"] == estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout)


def test_x_ell_preconditioner_fails_closed_when_memory_budget_is_exceeded() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)
    estimate = estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout)

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=estimate - 1,
    )

    assert preconditioner.selected is False
    assert preconditioner.operator is None
    assert preconditioner.reason == f"kinetic_pc_budget_exceeded:{estimate}>{estimate - 1}"
    assert preconditioner.metadata["candidate_nbytes_estimate"] == estimate
    assert preconditioner.metadata["max_candidate_nbytes"] == estimate - 1
    with pytest.raises(RuntimeError, match="was not selected"):
        preconditioner.apply(np.ones((int(layout.total_size),), dtype=np.float64))


def test_x_ell_preconditioner_regularizes_singular_line_blocks() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout).tolil()
    first_indices = rhs1_full_csr_x_ell_block_indices(layout)[0]
    matrix[np.ix_(first_indices, first_indices)] = 1.0
    matrix = matrix.tocsr()

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
        regularization=1.0e-8,
    )

    assert preconditioner.selected is True, preconditioner.to_dict()
    assert preconditioner.metadata["block_inverse_regularized_count"] == 2
    actual = preconditioner.apply(np.ones((int(layout.total_size),), dtype=np.float64))
    assert np.all(np.isfinite(actual))


def test_x_ell_preconditioner_identity_tail_policy_does_not_store_tail_jacobi() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)
    rhs = np.linspace(-0.75, 1.25, int(layout.total_size), dtype=np.float64)

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
        tail_policy="identity",
        build_native_factor=True,
    )

    assert preconditioner.selected is True, preconditioner.to_dict()
    assert preconditioner.metadata["tail_policy"] == "identity"
    assert preconditioner.metadata["tail_inverse_nbytes_actual"] == 0
    assert preconditioner.metadata["native_factor_tail_inverse_nbytes"] == 0
    actual = preconditioner.apply(rhs)
    np.testing.assert_allclose(actual[int(layout.f_size) :], rhs[int(layout.f_size) :])
    np.testing.assert_allclose(preconditioner.apply_native(rhs), actual, rtol=1.0e-13, atol=1.0e-13)


def test_x_ell_preconditioner_raises_for_wrong_rhs_shape_and_missing_native_factor() -> None:
    layout = _tiny_layout()
    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=_tiny_full_csr(layout),
        layout=layout,
        max_candidate_nbytes=10_000,
    )

    assert preconditioner.selected
    with pytest.raises(ValueError, match="dimension mismatch"):
        preconditioner.apply(np.ones((int(layout.total_size) + 1,), dtype=np.float64))
    assert preconditioner.operator is not None
    with pytest.raises(ValueError, match="rhs must have shape"):
        preconditioner.operator._matvec(
            np.ones((int(layout.total_size) + 1,), dtype=np.float64)
        )
    with pytest.raises(RuntimeError, match="native kinetic factor is unavailable"):
        preconditioner.apply_native(np.ones((int(layout.total_size),), dtype=np.float64))


def test_x_ell_preconditioner_tail_free_application_returns_kinetic_block_only() -> None:
    layout = _layout_variant(phi1_size=0, extra_size=0, total_size=8, include_phi1=False)
    diagonal = np.linspace(2.0, 9.0, int(layout.total_size), dtype=np.float64)
    matrix = sp.diags(diagonal, format="csr")
    rhs = np.linspace(-0.75, 1.25, int(layout.total_size), dtype=np.float64)

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
        regularization=0.0,
    )

    assert preconditioner.selected
    actual = preconditioner.apply(rhs)
    np.testing.assert_allclose(actual, rhs / diagonal, rtol=1.0e-13, atol=1.0e-13)


def test_x_ell_preconditioner_aliases_and_disabled_paths_are_fail_closed() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)
    baseline = estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout)

    assert estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout, kind="auto") == baseline
    assert estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout, kind="disabled") == 0
    assert (
        estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout, tail_policy="diag")
        == baseline
    )
    assert (
        estimate_rhs1_full_csr_kinetic_preconditioner_nbytes(layout, tail_policy="off")
        == baseline - 2 * np.dtype(np.float64).itemsize
    )

    disabled = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="off",
    )
    unsupported = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="block-jacobi",
    )
    unsupported_tail = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        tail_policy="solve_tail",
    )

    assert disabled.reason == "disabled"
    assert unsupported.reason == "unsupported_preconditioner"
    assert unsupported_tail.reason == "unsupported_tail_policy"
    assert unsupported_tail.metadata["tail_policy"] == "solve_tail"


def test_x_ell_preconditioner_rejects_memory_and_nonfinite_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)

    block_too_large = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_block_size=1,
    )
    assert block_too_large.reason == "kinetic_pc_block_size_exceeded:4>1"

    nonfinite_matrix = matrix.tolil()
    nonfinite_matrix[0, 0] = np.nan
    nonfinite_matrix = nonfinite_matrix.tocsr()
    nonfinite = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=nonfinite_matrix,
        layout=layout,
    )
    assert nonfinite.reason == "matrix_data_nonfinite"

    monkeypatch.setattr(
        full_csr,
        "estimate_rhs1_full_csr_kinetic_preconditioner_nbytes",
        lambda *_args, **_kwargs: 1,
    )
    actual_too_large = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=2,
    )
    assert actual_too_large.reason.startswith("kinetic_pc_budget_exceeded_actual:")


def test_x_ell_preconditioner_rejects_nonfinite_inverse_blocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)

    def fake_inverse(block):
        return np.full_like(block, np.inf, dtype=np.float64)

    monkeypatch.setattr(full_csr.np.linalg, "inv", fake_inverse)

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
    )

    assert preconditioner.selected is False
    assert preconditioner.reason == "block_inverse_nonfinite"
    assert preconditioner.metadata["block_inverse_nonfinite_count"] == 2


def test_x_ell_preconditioner_uses_pseudoinverse_for_singular_line_blocks() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout).tolil()
    first_indices = rhs1_full_csr_x_ell_block_indices(layout)[0]
    matrix[np.ix_(first_indices, first_indices)] = 1.0
    matrix = matrix.tocsr()

    preconditioner = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=layout,
        max_candidate_nbytes=10_000,
        regularization=0.0,
    )

    assert preconditioner.selected is True, preconditioner.to_dict()
    assert preconditioner.metadata["block_inverse_singular_count"] == 1
    assert np.all(np.isfinite(preconditioner.apply(np.ones(int(layout.total_size)))))


def test_x_ell_preconditioner_rejects_non_rhs1_and_shape_mismatch() -> None:
    layout = _tiny_layout()
    matrix = _tiny_full_csr(layout)

    wrong_mode = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=matrix,
        layout=_tiny_layout(rhs_mode=2),
    )
    assert wrong_mode.selected is False
    assert wrong_mode.reason == "unsupported_rhs_mode:2"

    wrong_shape = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=sp.eye(int(layout.total_size) - 1, format="csr"),
        layout=layout,
    )
    assert wrong_shape.selected is False
    assert wrong_shape.reason == "layout_size_mismatch"


def test_x_ell_preconditioner_rejects_invalid_layout_and_matrix_contracts() -> None:
    layout = _tiny_layout()

    bad_f_size = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=sp.eye(int(layout.total_size), format="csr"),
        layout=_layout_variant(f_size=7),
    )
    bad_total = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=sp.eye(7, format="csr"),
        layout=_layout_variant(total_size=7, phi1_size=0, extra_size=-1),
    )
    dense_matrix = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=np.eye(int(layout.total_size)),
        layout=layout,
    )
    nonsquare = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=sp.csr_matrix((int(layout.total_size), int(layout.total_size) - 1)),
        layout=layout,
    )
    empty_layout = _layout_variant(
        n_x=0,
        f_size=0,
        phi1_size=0,
        extra_size=0,
        total_size=0,
        include_phi1=False,
    )
    empty = build_rhs1_full_csr_kinetic_preconditioner(
        matrix=sp.csr_matrix((0, 0)),
        layout=empty_layout,
    )

    assert bad_f_size.reason == "layout_f_size_mismatch"
    assert bad_total.reason == "layout_total_size_mismatch"
    assert dense_matrix.reason == "matrix_not_sparse"
    assert nonsquare.reason == "matrix_not_square"
    assert empty.reason == "empty_kinetic_layout"
