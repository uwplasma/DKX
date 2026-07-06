from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import sparse

import sfincs_jax.operators.profile_full_system as profile_full_system
from sfincs_jax.operators.profile_layout import RHS1BlockLayout


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


def _layout(total_size: int = 3) -> RHS1BlockLayout:
    return RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=1,
        n_theta=1,
        n_zeta=1,
        f_size=1,
        phi1_size=1 if total_size > 1 else 0,
        extra_size=max(0, int(total_size) - 2),
        total_size=int(total_size),
        constraint_scheme=1,
        include_phi1=total_size > 1,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )


def _selected_preconditioner(kind: str = "sentinel") -> profile_full_system.RHS1StructuredFullCSRPreconditioner:
    return profile_full_system.RHS1StructuredFullCSRPreconditioner(
        operator=object(),
        selected=True,
        kind=kind,
        reason="complete",
        setup_s=0.0,
        metadata={"builder": kind},
    )


def test_active_projected_preconditioner_dispatch_routes_supported_aliases(monkeypatch) -> None:
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    layout = _layout(total_size=3)
    active = np.arange(3, dtype=np.int32)
    calls: list[tuple[str, str]] = []

    def fake_builder(tag: str):
        def _builder(**kwargs):
            calls.append((tag, str(kwargs["requested_kind"])))
            return _selected_preconditioner(kind=tag)

        return _builder

    dispatch_cases = [
        ("active_diagonal_schur", "_build_active_projected_diagonal_schur_preconditioner", "diag"),
        ("active_tail_sparse_coarse", "_build_active_projected_sparse_coarse_residual_preconditioner", "sparse-coarse"),
        ("active_fortran_v3_reduced_lu", "_build_active_fortran_v3_reduced_sparse_factor_preconditioner", "fortran-reduced"),
        ("active_coarse", "_build_active_projected_coarse_residual_preconditioner", "coarse"),
        ("active_low_l_schur", "_build_active_projected_low_l_schur_preconditioner", "low-l"),
        ("active_ell_band_schur", "_build_active_projected_ell_band_schur_preconditioner", "ell-band"),
        ("active_xell_window_lsq_schur", "_build_active_projected_xell_window_lsq_schur_preconditioner", "xell-window"),
        ("active_coupled_kinetic_block", "_build_active_projected_coupled_kinetic_block_preconditioner", "coupled-kinetic"),
        ("active_filtered_sparse_factor", "_build_active_projected_filtered_sparse_factor_preconditioner", "filtered"),
        ("active_symbolic_frontal_schur_lu", "_build_active_projected_symbolic_frontal_schur_lu_preconditioner", "frontal"),
        ("active_symbolic_superblock_lu", "_build_active_projected_symbolic_superblock_lu_preconditioner", "superblock"),
        ("active_symbolic_block_schur_lu", "_build_active_projected_symbolic_block_schur_lu_preconditioner", "block-schur"),
        ("active_symbolic_coupled_schur", "_build_active_projected_symbolic_coupled_schur_preconditioner", "coupled-schur"),
        ("active_bounded_native_stack", "_build_active_projected_bounded_native_stack_preconditioner", "native-stack"),
        ("active_fortran_v3_reduced_native_stack", "_build_active_fortran_v3_reduced_native_stack_preconditioner", "v3-native-stack"),
        (
            "active_native_xell_field_split_sparse_coarse",
            "_build_active_projected_native_xell_field_split_sparse_coarse_preconditioner",
            "native-xell-coarse",
        ),
        ("active_global_field_split_schur", "_build_active_projected_global_field_split_schur_preconditioner", "global-schur"),
        ("active_overlap_schwarz", "_build_active_projected_overlap_schwarz_preconditioner", "schwarz"),
        ("active_angular_line", "_build_active_projected_angular_line_preconditioner", "angular-line"),
        ("active_native_indexed_schwarz", "_build_active_projected_native_indexed_schwarz_preconditioner", "indexed-schwarz"),
        ("active_xblock", "_build_active_projected_xblock_preconditioner", "xblock"),
        ("active_ilu_coarse", "_build_active_projected_coarse_residual_preconditioner", "ilu-coarse"),
        ("active_global_sparse_factor", "_build_active_global_sparse_factor_preconditioner", "global-factor"),
        ("active_scaled_ilu", "_build_active_scaled_sparse_factor_preconditioner", "scaled-factor"),
    ]
    for kind, builder_name, tag in dispatch_cases:
        monkeypatch.setattr(profile_full_system, builder_name, fake_builder(tag))
        result = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
            matrix=matrix,
            layout=layout,
            active_indices=active,
            kind=kind,
            max_factor_nbytes=4096,
        )
        assert result.selected
        assert result.kind == tag

    assert calls == [(tag, kind) for kind, _builder_name, tag in dispatch_cases]


@pytest.mark.parametrize(
    ("kind", "expected_kind"),
    [
        ("active_diagonal_schur", "active_diagonal_schur"),
        ("active_tail_sparse_coarse", "active_tail_sparse_coarse"),
        ("active_fortran_v3_reduced_lu", "active_fortran_v3_pc_matrix"),
        ("active_coarse", "active_coarse"),
        ("active_low_l_schur", "active_low_l_schur"),
        ("active_ell_band_schur", "active_ell_band_schur"),
        ("active_xell_window_lsq_schur", "active_xell_window_lsq_schur"),
        ("active_coupled_kinetic_block", "active_coupled_kinetic_block"),
        ("active_filtered_sparse_factor", "active_filtered_sparse_factor"),
        ("active_symbolic_frontal_schur_lu", "active_symbolic_frontal_schur_lu"),
        ("active_symbolic_superblock_lu", "active_symbolic_superblock_lu"),
        ("active_symbolic_block_schur_lu", "active_symbolic_block_schur_lu"),
        ("active_symbolic_coupled_schur", "active_symbolic_coupled_schur"),
        ("active_bounded_native_stack", "active_bounded_native_stack"),
        ("active_fortran_v3_reduced_native_stack", "active_fortran_v3_reduced_native_stack"),
        ("active_native_xell_field_split_sparse_coarse", "active_native_xell_field_split_sparse_coarse"),
        ("active_angular_line_field_split_sparse_coarse", "active_angular_line_field_split_sparse_coarse"),
        ("active_multiline_field_split_sparse_coarse", "active_multiline_field_split_sparse_coarse"),
        ("active_global_field_split_schur", "active_global_field_split_schur"),
        ("active_overlap_schwarz", "active_overlap_schwarz"),
        ("active_angular_line", "active_angular_line"),
        ("active_native_indexed_schwarz", "active_native_indexed_schwarz"),
        ("active_xblock", "active_xblock"),
        ("active_ilu_coarse", "active_ilu_coarse"),
    ],
)
def test_active_projected_preconditioner_dispatch_fails_closed_without_layout(
    kind: str,
    expected_kind: str,
) -> None:
    result = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(3, format="csr", dtype=np.float64),
        layout=None,
        active_indices=None,
        kind=kind,
        max_factor_nbytes=4096,
    )

    assert not result.selected
    assert result.kind == expected_kind
    assert result.reason == "missing_active_layout"


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


@pytest.mark.parametrize(
    ("kind", "expected_kind"),
    [
        ("active_diagonal_schur", "active_diagonal_schur"),
        ("active_schwarz_sparse_coarse", "active_tail_sparse_coarse"),
        ("active_fortran_v3_reduced_lu", "active_fortran_v3_pc_matrix"),
        ("active_coarse", "active_coarse"),
        ("active_low_l_schur", "active_low_l_schur"),
        ("active_ell_band_schur", "active_ell_band_schur"),
        ("active_xell_window_lsq_schur", "active_xell_window_lsq_schur"),
        ("active_coupled_kinetic_block", "active_coupled_kinetic_block"),
        ("active_filtered_sparse_factor", "active_filtered_sparse_factor"),
        ("active_symbolic_frontal_schur_lu", "active_symbolic_frontal_schur_lu"),
        ("active_symbolic_superblock_lu", "active_symbolic_superblock_lu"),
        ("active_symbolic_block_schur_lu", "active_symbolic_block_schur_lu"),
        ("active_symbolic_coupled_schur", "active_symbolic_coupled_schur"),
        ("active_bounded_native_stack", "active_bounded_native_stack"),
        ("active_fortran_v3_reduced_native_stack", "active_fortran_v3_reduced_native_stack"),
        ("active_native_xell_field_split_sparse_coarse", "active_native_xell_field_split_sparse_coarse"),
        ("active_angular_line_field_split_sparse_coarse", "active_angular_line_field_split_sparse_coarse"),
        ("active_multiline_field_split_sparse_coarse", "active_multiline_field_split_sparse_coarse"),
        ("active_coupled_kinetic_field_split_sparse_coarse", "active_coupled_kinetic_field_split_sparse_coarse"),
        ("active_global_field_split_schur", "active_global_field_split_schur"),
        ("active_overlap_schwarz", "active_overlap_schwarz"),
        ("active_angular_line", "active_angular_line"),
        ("active_native_indexed_schwarz", "active_native_indexed_schwarz"),
        ("active_xblock", "active_xblock"),
        ("active_ilu_coarse", "active_ilu_coarse"),
    ],
)
def test_active_projected_preconditioner_aliases_fail_closed_without_layout(kind: str, expected_kind: str) -> None:
    matrix = sparse.eye(3, format="csr", dtype=np.float64)

    pc = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=None,
        active_indices=None,
        kind=kind,
    )

    assert not pc.selected
    assert pc.kind == expected_kind
    assert pc.reason == "missing_active_layout"


@pytest.mark.parametrize(
    ("kind", "builder_attr", "active_required"),
    [
        ("active_diagonal_schur", "_build_active_projected_diagonal_schur_preconditioner", False),
        ("active_schwarz_sparse_coarse", "_build_active_projected_sparse_coarse_residual_preconditioner", False),
        ("active_fortran_v3_reduced_lu", "_build_active_fortran_v3_reduced_sparse_factor_preconditioner", False),
        ("active_coarse", "_build_active_projected_coarse_residual_preconditioner", False),
        ("active_low_l_schur", "_build_active_projected_low_l_schur_preconditioner", False),
        ("active_ell_band_schur", "_build_active_projected_ell_band_schur_preconditioner", True),
        ("active_xell_window_lsq_schur", "_build_active_projected_xell_window_lsq_schur_preconditioner", True),
        ("active_coupled_kinetic_block", "_build_active_projected_coupled_kinetic_block_preconditioner", False),
        ("active_filtered_sparse_factor", "_build_active_projected_filtered_sparse_factor_preconditioner", False),
        ("active_symbolic_frontal_schur_lu", "_build_active_projected_symbolic_frontal_schur_lu_preconditioner", False),
        ("active_symbolic_superblock_lu", "_build_active_projected_symbolic_superblock_lu_preconditioner", False),
        ("active_symbolic_block_schur_lu", "_build_active_projected_symbolic_block_schur_lu_preconditioner", False),
        ("active_symbolic_coupled_schur", "_build_active_projected_symbolic_coupled_schur_preconditioner", False),
        ("active_bounded_native_stack", "_build_active_projected_bounded_native_stack_preconditioner", False),
        ("active_fortran_v3_reduced_native_stack", "_build_active_fortran_v3_reduced_native_stack_preconditioner", False),
        (
            "active_native_xell_field_split_sparse_coarse",
            "_build_active_projected_native_xell_field_split_sparse_coarse_preconditioner",
            False,
        ),
        ("active_global_field_split_schur", "_build_active_projected_global_field_split_schur_preconditioner", False),
        ("active_overlap_schwarz", "_build_active_projected_overlap_schwarz_preconditioner", False),
        ("active_angular_line", "_build_active_projected_angular_line_preconditioner", True),
        ("active_native_indexed_schwarz", "_build_active_projected_native_indexed_schwarz_preconditioner", True),
        ("active_xblock", "_build_active_projected_xblock_preconditioner", False),
        ("active_global_sparse_factor", "_build_active_global_sparse_factor_preconditioner", False),
        ("active_scaled_ilu", "_build_active_scaled_sparse_factor_preconditioner", False),
    ],
)
def test_active_projected_preconditioner_aliases_route_to_builder(
    monkeypatch,
    kind: str,
    builder_attr: str,
    active_required: bool,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_builder(**kwargs):
        calls.append(kwargs)
        return _selected_preconditioner(kind=f"built:{builder_attr}")

    monkeypatch.setattr(profile_full_system, builder_attr, fake_builder)
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    active = np.arange(3, dtype=np.int64) if active_required else None

    pc = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=_layout(3),
        active_indices=active,
        kind=kind,
        max_factor_nbytes=12345,
        regularization=1.0e-9,
    )

    assert pc.selected
    assert pc.kind == f"built:{builder_attr}"
    assert len(calls) == 1
    assert calls[0]["requested_kind"] == kind
    assert calls[0]["max_factor_nbytes"] == 12345
    if active_required:
        active_key = "active_kinetic_indices" if kind == "active_angular_line" else "active_indices"
        np.testing.assert_array_equal(calls[0][active_key], active)


def test_active_projected_preconditioner_ilu_budget_and_unsupported_paths(monkeypatch) -> None:
    matrix = sparse.eye(4, format="csr", dtype=np.float64)
    monkeypatch.setattr(
        profile_full_system,
        "_estimate_spilu_factor_nbytes",
        lambda **_kwargs: 1024,
    )

    explicit = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="active_ilu",
        max_factor_nbytes=16,
    )
    assert not explicit.selected
    assert explicit.kind == "active_spilu"
    assert explicit.reason == "active_spilu_budget_exceeded:1024>16"
    assert explicit.metadata["factor_nbytes_estimate"] == 1024

    auto = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="auto",
        max_factor_nbytes=16,
    )
    assert auto.selected
    assert auto.kind == "jacobi"
    assert auto.reason.startswith("auto_selected:")

    unsupported = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="not_a_preconditioner",
    )
    assert not unsupported.selected
    assert unsupported.reason == "unsupported_active_projected_preconditioner"


def test_active_projected_preconditioner_fail_closed_contract_edges(monkeypatch) -> None:
    rectangular = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sparse.csr_matrix((2, 3), dtype=np.float64),
        kind="jacobi",
    )
    assert not rectangular.selected
    assert rectangular.reason == "matrix_not_square"
    assert rectangular.metadata["shape"] == (2, 3)

    disabled = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(3, format="csr", dtype=np.float64),
        kind="off",
    )
    assert not disabled.selected
    assert disabled.kind == "none"
    assert disabled.reason == "disabled"

    monkeypatch.setattr(
        profile_full_system,
        "resolve_active_projected_preconditioner_auto_policy",
        lambda *, matrix_size: SimpleNamespace(
            candidates=("auto",),
            candidates_requested=("auto",),
            skipped_large_fallbacks=(),
            large_fallback_size=100,
            large_default_used=False,
            log_progress=False,
        ),
    )
    auto_fallback = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sparse.diags([2.0, 4.0, 8.0], format="csr", dtype=np.float64),
        kind="auto",
        regularization=0.0,
    )
    assert auto_fallback.selected
    assert auto_fallback.kind == "jacobi"
    assert auto_fallback.reason == "active_auto_no_candidate_selected"
    assert auto_fallback.metadata["auto_candidates"] == ["auto"]

    monkeypatch.setattr(
        profile_full_system,
        "resolve_active_projected_preconditioner_auto_policy",
        lambda *, matrix_size: SimpleNamespace(
            candidates=(),
            candidates_requested=("jacobi",),
            skipped_large_fallbacks=("jacobi",),
            large_fallback_size=1,
            large_default_used=True,
            log_progress=False,
        ),
    )
    skipped_large = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(3, format="csr", dtype=np.float64),
        kind="auto",
    )
    assert not skipped_large.selected
    assert skipped_large.reason == "active_auto_no_safe_large_candidate_selected"
    assert skipped_large.metadata["auto_skipped_large_fallbacks"] == ("jacobi",)


def test_active_projected_preconditioner_explicit_spilu_failure(monkeypatch) -> None:
    import scipy.sparse.linalg as scipy_sparse_linalg

    matrix = sparse.eye(4, format="csc", dtype=np.float64)
    monkeypatch.setattr(profile_full_system, "_estimate_spilu_factor_nbytes", lambda **_kwargs: 128)

    def fail_spilu(*_args, **_kwargs):
        raise RuntimeError("synthetic factorization failure")

    monkeypatch.setattr(scipy_sparse_linalg, "spilu", fail_spilu)
    failed = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        kind="active_ilu",
        max_factor_nbytes=1024,
    )

    assert not failed.selected
    assert failed.kind == "active_spilu"
    assert failed.reason == "active_spilu_failed:RuntimeError"
    assert failed.metadata["factor_nbytes_estimate"] == 128


def test_direct_reduced_pmat_preconditioner_admission_and_metadata(monkeypatch) -> None:
    op = _fake_op(total_size=4, f_size=2)
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_DIRECT_REDUCED_PMAT_EMISSION_MAX_SIZE", "2")
    too_large = profile_full_system.build_direct_active_fortran_v3_reduced_pmat_preconditioner(
        op=op,
        active_indices=np.arange(4, dtype=np.int64),
    )
    assert not too_large.selected
    assert too_large.reason == "direct_reduced_pmat_emission_size_exceeded:4>2"
    assert too_large.metadata["direct_reduced_pmat_emission"] is False

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_DIRECT_REDUCED_PMAT_EMISSION_MAX_SIZE", "10")

    def fail_emission(**_kwargs):
        raise ValueError("unsupported synthetic contract")

    monkeypatch.setattr(
        profile_full_system,
        "_direct_active_fortran_v3_reduced_pmat_input_matrix",
        fail_emission,
    )
    failed = profile_full_system.build_direct_active_fortran_v3_reduced_pmat_preconditioner(
        op=op,
        active_indices=np.arange(3, dtype=np.int64),
    )
    assert not failed.selected
    assert failed.reason == "direct_reduced_pmat_emission_failed"
    assert failed.metadata["error"] == "unsupported synthetic contract"

    layout = _layout(3)
    active = np.arange(3, dtype=np.int64)
    pmat = sparse.diags([2.0, 3.0, 5.0], format="csr", dtype=np.float64)
    monkeypatch.setattr(
        profile_full_system,
        "_direct_active_fortran_v3_reduced_pmat_input_matrix",
        lambda **_kwargs: (
            pmat,
            layout,
            active,
            {
                "direct_reduced_pmat_emission": True,
                "direct_reduced_pmat_nnz": int(pmat.nnz),
            },
        ),
    )

    def fake_factor_builder(**kwargs):
        assert kwargs["matrix"] is pmat
        assert kwargs["layout"] is layout
        np.testing.assert_array_equal(kwargs["active_indices"], active)
        return profile_full_system.RHS1StructuredFullCSRPreconditioner(
            operator=object(),
            selected=True,
            kind="active_fortran_v3_pc_matrix",
            reason="complete",
            setup_s=0.0,
            metadata={"factor_builder": "synthetic"},
        )

    monkeypatch.setattr(
        profile_full_system,
        "_build_active_fortran_v3_reduced_sparse_factor_preconditioner",
        fake_factor_builder,
    )
    selected = profile_full_system.build_direct_active_fortran_v3_reduced_pmat_preconditioner(
        op=op,
        active_indices=active,
        requested_kind="active_fortran_v3_reduced_direct_pmat_lu",
    )

    assert selected.selected
    assert selected.kind == "active_fortran_v3_pc_matrix"
    assert selected.metadata["factor_builder"] == "synthetic"
    assert selected.metadata["direct_reduced_pmat_emission"] is True
    assert selected.metadata["direct_reduced_pmat_nnz"] == 3


def test_direct_reduced_pmat_input_matrix_uses_structured_fblock_and_tail(monkeypatch) -> None:
    op = _fake_op(total_size=3, f_size=2)
    f_matrix = sparse.diags([2.0, 3.0], format="csr", dtype=np.float64)
    tail = sparse.csr_matrix(
        np.asarray(
            [
                [0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0],
                [0.25, 0.0, 5.0],
            ],
            dtype=np.float64,
        )
    )
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_fblock_csr_operator",
        lambda *_args, **_kwargs: _fblock_selection(matrix=f_matrix),
    )
    monkeypatch.setattr(profile_full_system, "_assemble_full_tail_csr", lambda **_kwargs: tail)

    direct, layout, active, metadata = profile_full_system._direct_active_fortran_v3_reduced_pmat_input_matrix(
        op=op,
        active_indices=None,
        max_csr_nbytes=1024,
    )

    assert layout.total_size == 3
    np.testing.assert_array_equal(active, np.asarray([0, 1, 2], dtype=np.int64))
    np.testing.assert_allclose(
        direct.toarray(),
        np.asarray(
            [
                [2.0, 0.0, 0.5],
                [0.0, 3.0, 0.0],
                [0.25, 0.0, 5.0],
            ],
            dtype=np.float64,
        ),
    )
    assert metadata["direct_reduced_pmat_emission"] is True
    assert metadata["direct_reduced_pmat_kinetic_nnz"] == 2
    assert metadata["direct_reduced_pmat_tail_nnz"] == 3

    with pytest.raises(ValueError, match="direct_reduced_pmat_budget_exceeded"):
        profile_full_system._direct_active_fortran_v3_reduced_pmat_input_matrix(
            op=op,
            active_indices=None,
            max_csr_nbytes=1,
        )


def test_direct_reduced_pmat_input_matrix_fails_closed_when_fblock_missing(monkeypatch) -> None:
    op = _fake_op(total_size=3, f_size=2)
    monkeypatch.setattr(
        profile_full_system,
        "select_structured_rhs1_fblock_csr_operator",
        lambda *_args, **_kwargs: _fblock_selection(selected=False, matrix=None, reason="budget"),
    )

    with pytest.raises(ValueError, match="fblock_not_selected:budget"):
        profile_full_system._direct_active_fortran_v3_reduced_pmat_input_matrix(
            op=op,
            active_indices=None,
        )


def test_active_coupled_kinetic_block_admission_and_zero_base(monkeypatch) -> None:
    layout = RHS1BlockLayout(
        n_species=1,
        n_x=1,
        n_xi=2,
        n_theta=1,
        n_zeta=1,
        f_size=2,
        phi1_size=1,
        extra_size=0,
        total_size=3,
        constraint_scheme=1,
        include_phi1=True,
        include_phi1_in_kinetic=False,
        rhs_mode=1,
    )
    matrix = sparse.diags([2.0, 3.0, 5.0], format="csr", dtype=np.float64)

    mismatch = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=np.asarray([0, 1], dtype=np.int64),
        kind="active_coupled_kinetic_block",
    )
    assert not mismatch.selected
    assert mismatch.reason == "active_index_size_mismatch"

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_INCLUDE_TAIL", "0")
    empty = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=None,
        kind="active_coupled_kinetic_block",
    )
    assert not empty.selected
    assert empty.reason == "empty_coupled_kinetic_block"

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_X_COUNT", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_INCLUDE_TAIL", "0")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_ELL_COUNT", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "2")
    too_many = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=None,
        kind="active_coupled_kinetic_block",
    )
    assert not too_many.selected
    assert too_many.reason == "active_coupled_kinetic_block_size_exceeded:2>1"

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_BLOCK_SIZE", "8")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_MAX_POSITIONS", "2")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_INCLUDE_TAIL", "1")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_BASE", "zero")
    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_ACTIVE_COUPLED_KINETIC_SCALE", "0")
    selected = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=None,
        kind="active_coupled_kinetic_block",
        regularization=0.0,
        max_factor_nbytes=1024 * 1024,
    )
    assert selected.selected
    assert selected.kind == "active_coupled_kinetic_block"
    assert selected.metadata["requested_base_kind"] == "zero"
    assert selected.metadata["block_size"] == 3
    np.testing.assert_allclose(selected.operator @ np.asarray([2.0, 6.0, 10.0]), np.asarray([1.0, 2.0, 2.0]))

    budget = profile_full_system.build_active_projected_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        active_indices=None,
        kind="active_coupled_kinetic_block",
        max_factor_nbytes=1,
    )
    assert not budget.selected
    assert budget.reason.startswith("active_coupled_kinetic_budget_exceeded:")


def test_full_csr_preconditioner_rejects_bad_contracts_and_disabled_path() -> None:
    layout = _layout(3)

    rectangular = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=sparse.csr_matrix((2, 3), dtype=np.float64),
        layout=layout,
        kind="jacobi",
    )
    assert not rectangular.selected
    assert rectangular.reason == "matrix_not_square"
    assert rectangular.metadata["shape"] == (2, 3)

    mismatch = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(2, format="csr", dtype=np.float64),
        layout=layout,
        kind="jacobi",
    )
    assert not mismatch.selected
    assert mismatch.reason == "layout_size_mismatch"
    assert mismatch.metadata["layout_total_size"] == 3

    disabled = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(3, format="csr", dtype=np.float64),
        layout=layout,
        kind="off",
    )
    assert not disabled.selected
    assert disabled.kind == "none"
    assert disabled.reason == "disabled"

    unsupported = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=sparse.eye(3, format="csr", dtype=np.float64),
        layout=layout,
        kind="does_not_exist",
    )
    assert not unsupported.selected
    assert unsupported.reason == "unsupported_preconditioner"


def test_full_csr_preconditioner_auto_prefers_safe_xblock_candidate(monkeypatch) -> None:
    layout = _layout(3)
    matrix = sparse.eye(3, format="csr", dtype=np.float64)
    captured: dict[str, object] = {}

    def fake_xblock(**kwargs):
        captured.update(kwargs)
        return _selected_preconditioner(kind="xblock_tz_low_l_schur")

    monkeypatch.setenv("SFINCS_JAX_RHS1_FULL_CSR_XBLOCK_AUTO_MIN_SIZE", "0")
    monkeypatch.setattr(
        profile_full_system,
        "_build_xblock_tz_low_l_schur_preconditioner",
        fake_xblock,
    )

    pc = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="auto",
        max_schur_size=8,
        max_block_inverse_nbytes=1024 * 1024,
    )

    assert pc.selected
    assert pc.kind == "xblock_tz_low_l_schur"
    assert captured["requested_kind"] == "auto"
    assert captured["layout"] is layout
    assert captured["config"]["lmax"] >= 0


@pytest.mark.parametrize(
    ("kind", "expected_kind"),
    [
        ("xi_block_schur", "xi_block_schur"),
        ("x_xi_block_schur", "x_xi_block_schur"),
        ("xblock_tz_low_l_schur", "xblock_tz_low_l_schur"),
        ("xblock_tz_low_l_coarse_schur", "xblock_tz_low_l_coarse_schur"),
        ("block_schur", "block_schur"),
        ("diagonal_schur", "diagonal_schur"),
    ],
)
def test_full_csr_preconditioner_reports_tail_budget_rejections(kind: str, expected_kind: str) -> None:
    layout = _layout(4)
    matrix = sparse.eye(4, format="csr", dtype=np.float64)

    pc = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        kind=kind,
        max_schur_size=0,
        max_block_inverse_nbytes=1024 * 1024,
    )

    assert not pc.selected
    assert pc.kind == expected_kind
    assert pc.reason == "schur_tail_size_exceeded:3>0"
    assert pc.metadata["tail_size"] == 3
    assert pc.metadata["max_schur_size"] == 0


def test_full_csr_preconditioner_jacobi_path_is_finite_and_metadata_rich() -> None:
    layout = _layout(3)
    matrix = sparse.diags([2.0, 4.0, 8.0], format="csr", dtype=np.float64)

    pc = profile_full_system.build_structured_rhs1_full_csr_preconditioner(
        matrix=matrix,
        layout=layout,
        kind="jacobi",
        regularization=0.0,
    )

    assert pc.selected
    assert pc.kind == "jacobi"
    np.testing.assert_allclose(pc.operator @ np.asarray([2.0, 8.0, 24.0]), np.asarray([1.0, 2.0, 3.0]))
    assert pc.metadata["diagonal_size"] == 3
