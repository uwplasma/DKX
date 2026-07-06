from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import sfincs_jax.operators.profile_full_system as profile_full_system


def _fake_op(*, rhs_mode: int = 1, total_size: int = 4, f_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        n_species=1,
        n_x=1,
        n_xi=max(1, int(f_size)),
        n_theta=1,
        n_zeta=1,
        f_size=int(f_size),
        phi1_size=0,
        extra_size=max(0, int(total_size) - int(f_size)),
        total_size=int(total_size),
        constraint_scheme=1,
        include_phi1=False,
        include_phi1_in_kinetic=False,
        rhs_mode=int(rhs_mode),
        fblock=SimpleNamespace(label="fblock", fp=None, pas=None),
        phi1_hat_base=None,
    )


def _fblock_selection(*, selected: bool = True, matrix: sparse.spmatrix | None = None, reason: str = "complete"):
    if matrix is None and selected:
        matrix = sparse.eye(2, format="csr", dtype=np.float64)
    return profile_full_system.RHS1StructuredFBlockCSRSelection(
        selection=SimpleNamespace(to_dict=lambda: {"selected": bool(selected), "reason": reason}),
        matrix=matrix,
        selected=bool(selected),
        reason=str(reason),
        cache_hit=False,
        build_s=0.0,
        metadata={
            "selected": bool(selected),
            "reason": str(reason),
            "csr_nbytes_actual": 0 if matrix is None else profile_full_system._scipy_csr_nbytes(matrix.tocsr()),
        },
    )


def test_structured_full_csr_selection_cache_budget_and_fblock_fail_closed(monkeypatch) -> None:
    profile_full_system.clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    op = _fake_op(total_size=4, f_size=2)
    f_matrix = sparse.csr_matrix(np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float64))
    f_selection = _fblock_selection(matrix=f_matrix)

    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_fblock_csr_operator",
        lambda *_args, **_kwargs: f_selection,
    )
    monkeypatch.setattr(
        profile_full_system,
        "_assemble_full_tail_csr",
        lambda **_kwargs: sparse.csr_matrix((4, 4), dtype=np.float64),
    )

    selected = profile_full_system.select_structured_rhs1_full_csr_operator(
        op,
        max_csr_nbytes=None,
        use_cache=True,
    )
    assert selected.selected
    np.testing.assert_allclose(selected.matvec(np.asarray([1.0, 2.0, 3.0, 4.0])), np.asarray([2.0, 6.0, 0.0, 0.0]))
    assert selected.to_dict()["selected"] is True

    rejected_from_cache = profile_full_system.select_structured_rhs1_full_csr_operator(
        op,
        max_csr_nbytes=1,
        use_cache=True,
    )
    assert not rejected_from_cache.selected
    assert rejected_from_cache.cache_hit
    assert rejected_from_cache.reason.startswith("csr_budget_exceeded:")

    failed_fblock = _fblock_selection(selected=False, matrix=None, reason="fblock_not_complete")
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_fblock_csr_operator",
        lambda *_args, **_kwargs: failed_fblock,
    )
    not_selected = profile_full_system.select_structured_rhs1_full_csr_operator(
        _fake_op(total_size=4, f_size=2),
        use_cache=False,
    )
    assert not not_selected.selected
    assert not_selected.reason == "fblock_not_complete"
    with pytest.raises(RuntimeError, match="structured full CSR operator was not selected"):
        not_selected.matvec(np.ones(4))


def test_structured_full_csr_preflight_budget_avoids_fblock_assembly(monkeypatch) -> None:
    op = _fake_op(total_size=6, f_size=4)
    summary = SimpleNamespace(
        shape=(6, 6),
        nnz=10_000,
        to_dict=lambda: {"shape": (6, 6), "nnz": 10_000},
    )
    monkeypatch.setattr(
        profile_full_system,
        "estimate_v3_full_system_conservative_sparsity_summary",
        lambda _op: summary,
    )
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_fblock_csr_operator",
        lambda *_args, **_kwargs: pytest.fail("f-block assembly should be skipped by preflight"),
    )

    selected = profile_full_system.select_structured_rhs1_full_csr_operator(
        op,
        max_csr_nbytes=16,
        use_cache=False,
    )

    assert not selected.selected
    assert selected.reason.startswith("csr_budget_preflight_exceeded:")
    assert selected.metadata["sparsity_summary"]["nnz"] == 10_000
    assert selected.fblock_selection.reason == "not_built_due_to_full_csr_preflight"


def test_structured_full_csr_operator_bundle_active_projection_and_budget_paths(monkeypatch) -> None:
    profile_full_system.clear_structured_rhs1_full_csr_cache(clear_fblock_cache=True)
    messages: list[str] = []
    assert profile_full_system._try_build_structured_rhs1_full_csr_operator_bundle(
        op=_fake_op(rhs_mode=2, total_size=4, f_size=2),
        active_indices=None,
        csr_max_mb=1.0,
        drop_tol=0.0,
    ) is None

    op = _fake_op(total_size=4, f_size=2)
    active = np.asarray([0, 2], dtype=np.int32)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_PROJECT_AFTER_BUILD_MAX_SIZE", "1")
    assert profile_full_system._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=active,
        csr_max_mb=1.0,
        drop_tol=0.0,
        emit=lambda level, message: messages.append(f"{level}:{message}"),
    ) is None
    assert any("skipped full build before active projection" in message for message in messages)

    full_matrix = sparse.csr_matrix(
        np.asarray(
            [
                [3.0, 1.0e-12, 0.2, 0.0],
                [0.0, 4.0, 0.0, 0.0],
                [0.2, 0.0, 5.0, 0.0],
                [0.0, 0.0, 0.0, 6.0],
            ],
            dtype=np.float64,
        )
    )
    selected = profile_full_system.RHS1StructuredFullCSRSelection(
        fblock_selection=_fblock_selection(),
        matrix=full_matrix,
        selected=True,
        reason="complete",
        cache_hit=False,
        build_s=0.0,
        metadata={"tail_nnz": 1, "fblock_csr_nbytes_actual": 16},
    )
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_STRUCTURED_FULL_CSR_ALLOW_PROJECT_AFTER_BUILD", "1")
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_full_csr_operator",
        lambda *_args, **_kwargs: selected,
    )
    bundle = profile_full_system._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=active,
        csr_max_mb=1.0,
        drop_tol=1.0e-10,
        emit=lambda level, message: messages.append(f"{level}:{message}"),
    )
    assert bundle is not None
    assert bundle.matrix.shape == (2, 2)
    assert bundle.metadata.shape == (2, 2)
    np.testing.assert_allclose(bundle.matvec(np.asarray([1.0, 2.0])), np.asarray([3.4, 10.2]))
    assert any("assembly complete" in message for message in messages)

    too_large = profile_full_system._try_build_structured_rhs1_full_csr_operator_bundle(
        op=op,
        active_indices=None,
        csr_max_mb=1.0e-5,
        drop_tol=0.0,
        emit=lambda level, message: messages.append(f"{level}:{message}"),
    )
    assert too_large is None
    assert any("rejected actual CSR budget" in message for message in messages)


def test_structured_full_csr_solve_error_gates_and_lgmres_branch(monkeypatch) -> None:
    op = _fake_op(total_size=3, f_size=3)
    matrix = sparse.csr_matrix(
        np.asarray(
            [
                [4.0, 0.1, 0.0],
                [0.0, 3.0, 0.2],
                [0.1, 0.0, 2.5],
            ],
            dtype=np.float64,
        )
    )
    selection = profile_full_system.RHS1StructuredFullCSRSelection(
        fblock_selection=_fblock_selection(matrix=matrix),
        matrix=matrix,
        selected=True,
        reason="complete",
        cache_hit=False,
        build_s=0.0,
        metadata={"tail_nnz": 0, "fblock_csr_nbytes_actual": profile_full_system._scipy_csr_nbytes(matrix)},
    )
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_full_csr_operator",
        lambda *_args, **_kwargs: selection,
    )

    with pytest.raises(ValueError, match="rhs must have shape"):
        profile_full_system.solve_structured_rhs1_full_csr(op, np.ones(2))
    with pytest.raises(ValueError, match="active_indices must not be empty"):
        profile_full_system.solve_structured_rhs1_full_csr(op, np.ones(3), active_indices=[])
    with pytest.raises(ValueError, match="outside the full system"):
        profile_full_system.solve_structured_rhs1_full_csr(op, np.ones(3), active_indices=[0, 9])
    with pytest.raises(ValueError, match="x0 must have shape"):
        profile_full_system.solve_structured_rhs1_full_csr(op, np.ones(3), x0=np.ones(2))
    with pytest.raises(ValueError, match="method must be"):
        profile_full_system.solve_structured_rhs1_full_csr(op, np.ones(3), method="bad")

    rhs = np.asarray([1.0, -2.0, 0.5])
    result = profile_full_system.solve_structured_rhs1_full_csr(
        op,
        rhs,
        method="lgmres",
        preconditioner="none",
        tol=1.0e-12,
        atol=1.0e-12,
        maxiter=20,
    )
    np.testing.assert_allclose(matrix @ result.x, rhs, rtol=1.0e-9, atol=1.0e-9)
    assert result.converged
    assert result.to_dict()["metadata"]["method"] == "lgmres"

    direct = profile_full_system.solve_structured_rhs1_full_csr(
        op,
        rhs,
        method="direct",
        preconditioner="none",
        tol=1.0e-12,
        atol=1.0e-12,
    )
    assert direct.converged
    assert direct.metadata["factor_kind"] == "splu"
