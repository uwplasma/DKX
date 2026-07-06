from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator

from sfincs_jax.operators import profile_reduced_tail
from sfincs_jax.solvers.explicit_sparse import SparseDecision, SparseOperatorBundle


def _tiny_constraint1_op(*, rhs_mode: int = 1, extra_size: int = 2, phi1_size: int = 0) -> SimpleNamespace:
    n_species = 1
    n_x = 1
    n_xi = 1
    n_theta = 1
    n_zeta = 1
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    total_size = f_size + int(phi1_size) + int(extra_size)
    return SimpleNamespace(
        rhs_mode=int(rhs_mode),
        constraint_scheme=1,
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        phi1_size=int(phi1_size),
        extra_size=int(extra_size),
        total_size=total_size,
        include_phi1=bool(phi1_size),
        include_phi1_in_kinetic=False,
        theta_weights=np.asarray([1.0], dtype=np.float64),
        zeta_weights=np.asarray([1.0], dtype=np.float64),
        d_hat=np.asarray([[1.0]], dtype=np.float64),
        x=np.asarray([0.5], dtype=np.float64),
        x_weights=np.asarray([2.0], dtype=np.float64),
        fblock=SimpleNamespace(collisionless=SimpleNamespace(n_xi_for_x=np.asarray([1], dtype=np.int32))),
    )


def _constraint1_pattern_op() -> SimpleNamespace:
    n_species = 1
    n_x = 2
    n_xi = 2
    n_theta = 1
    n_zeta = 1
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    extra_size = 2
    return SimpleNamespace(
        rhs_mode=1,
        constraint_scheme=1,
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
        point_at_x0=False,
        theta_weights=np.asarray([1.0], dtype=np.float64),
        zeta_weights=np.asarray([1.0], dtype=np.float64),
        d_hat=np.asarray([[2.0]], dtype=np.float64),
        x=np.asarray([0.5, 1.5], dtype=np.float64),
        x_weights=np.asarray([0.25, 0.75], dtype=np.float64),
        fblock=SimpleNamespace(
            f_shape=(n_species, n_x, n_xi, n_theta, n_zeta),
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray([2, 2], dtype=np.int32)),
        ),
    )


def _identity_reduced_maps(size: int):
    def reduce_full(vec):
        return jnp.asarray(vec, dtype=jnp.float64).reshape((size,))

    def expand_reduced(vec):
        return jnp.asarray(vec, dtype=jnp.float64).reshape((size,))

    return reduce_full, expand_reduced


def _active_reduced_maps(total_size: int, active_indices: np.ndarray):
    active = np.asarray(active_indices, dtype=np.int32)

    def reduce_full(vec):
        return jnp.asarray(vec, dtype=jnp.float64).reshape((total_size,))[jnp.asarray(active)]

    def expand_reduced(vec):
        out = jnp.zeros((total_size,), dtype=jnp.float64)
        return out.at[jnp.asarray(active)].set(jnp.asarray(vec, dtype=jnp.float64).reshape((active.size,)))

    return reduce_full, expand_reduced


def _structured_bundle(matrix: sp.spmatrix) -> SparseOperatorBundle:
    matrix = matrix.tocsr()
    decision = SparseDecision(
        storage_kind="csr",
        reason="unit structured callback",
        backend="cpu",
        shape=tuple(int(v) for v in matrix.shape),
        dense_nbytes=int(np.prod(matrix.shape) * np.dtype(np.float64).itemsize),
        csr_nbytes_estimate=int(matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes),
        nnz_estimate=int(matrix.nnz),
        block_cols=0,
        drop_tol=0.0,
    )
    operator = LinearOperator(
        matrix.shape,
        matvec=lambda x: np.asarray(matrix @ np.asarray(x, dtype=np.float64)),
        dtype=np.float64,
    )
    return SparseOperatorBundle(matrix=matrix, operator=operator, metadata=decision)


class _ProjectedFBlock:
    def __init__(self, matrix: sp.spmatrix, *, nnz_blocks: int | None = None) -> None:
        self._matrix = matrix.tocsr()
        self.shape = tuple(int(v) for v in self._matrix.shape)
        self.nnz_blocks = int(self._matrix.nnz if nnz_blocks is None else nnz_blocks)

    def to_scipy_csr_matrix(self):
        return self._matrix


class _StructuredFBlockOperator:
    block_size = 2

    def __init__(self, *, nnz_blocks: int | None = None) -> None:
        self.projected_blocks: list[int] = []
        self._nnz_blocks = nnz_blocks

    def project_block_indices(self, active_blocks):
        blocks = np.asarray(active_blocks, dtype=np.int64).reshape((-1,))
        self.projected_blocks = [int(v) for v in blocks]
        size = int(blocks.size) * int(self.block_size)
        diag = np.arange(11.0, 11.0 + size, dtype=np.float64)
        return _ProjectedFBlock(sp.diags(diag, format="csr"), nnz_blocks=self._nnz_blocks)


def _patch_structured_fblock_selection(monkeypatch, operator: _StructuredFBlockOperator) -> None:
    selection = SimpleNamespace(
        selected=True,
        assembly=SimpleNamespace(operator=operator),
        reason="unit structured f-block",
    )
    monkeypatch.setattr(profile_reduced_tail, "select_structured_rhs1_fblock_operator", lambda *_args, **_kwargs: selection)


def test_fortran_reduced_direct_tail_rejects_non_constraint1_layouts() -> None:
    pattern = sp.eye(3, format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _identity_reduced_maps(3)

    kwargs = dict(
        op_pc=_tiny_constraint1_op(),
        pattern=pattern,
        active_indices=np.arange(3, dtype=np.int32),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.0,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=1.0,
        drop_tol=0.0,
        color_batch=1,
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **_kwargs: None,
    )

    assert profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=_tiny_constraint1_op(rhs_mode=2),
        **kwargs,
    ) is None
    assert profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=_tiny_constraint1_op(extra_size=1),
        **kwargs,
    ) is None
    assert profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=_tiny_constraint1_op(phi1_size=1),
        **kwargs,
    ) is None
    assert profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=_tiny_constraint1_op(),
        **{**kwargs, "pattern": sp.csr_matrix(np.ones((2, 3), dtype=np.float64))},
    ) is None
    assert profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=_tiny_constraint1_op(),
        **{**kwargs, "active_indices": np.arange(2, dtype=np.int32)},
    ) is None


def test_fortran_reduced_direct_tail_accepts_structured_csr_callback(monkeypatch) -> None:
    op = _tiny_constraint1_op()
    pattern = sp.eye(op.total_size, format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _identity_reduced_maps(op.total_size)
    emitted: list[tuple[int, str]] = []

    monkeypatch.delenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", raising=False)
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "1")

    def _callback(**kwargs):
        assert kwargs["active_indices"].tolist() == [0, 1, 2]
        assert kwargs["csr_max_mb"] == 4.0
        assert kwargs["drop_tol"] == 0.0
        matrix = sp.diags([2.0, 3.0, 4.0], format="csr", dtype=np.float64)
        return _structured_bundle(matrix)

    bundle = profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=op,
        op_pc=op,
        pattern=pattern,
        active_indices=np.arange(op.total_size, dtype=np.int32),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.5,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=4.0,
        drop_tol=0.0,
        color_batch=3,
        emit=lambda level, message: emitted.append((int(level), str(message))),
        build_structured_rhs1_full_csr_operator_bundle_callback=_callback,
    )

    assert bundle is not None
    assert bundle.matrix is not None
    np.testing.assert_allclose(bundle.matrix.diagonal(), [2.5, 3.5, 4.5])
    np.testing.assert_allclose(bundle.matvec(np.asarray([1.0, 2.0, 3.0])), [2.5, 7.0, 13.5])
    assert bundle.metadata.shape == (op.total_size, op.total_size)
    assert bundle.metadata.block_cols == 0
    assert "structured direct-tail CSR" in bundle.metadata.reason
    assert any("structured csr built" in message for _level, message in emitted)


def test_fortran_reduced_direct_tail_pattern_fallback_builds_source_and_moment_blocks(
    monkeypatch,
) -> None:
    op = _constraint1_pattern_op()
    pattern = sp.eye(op.total_size, format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _identity_reduced_maps(op.total_size)
    emitted: list[str] = []

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "pattern")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")

    diag = jnp.asarray([2.0, 3.0, 4.0, 5.0, 0.0, 0.0], dtype=jnp.float64)

    def _apply(_op, x_full):
        return diag * jnp.asarray(x_full, dtype=jnp.float64)

    monkeypatch.setattr(profile_reduced_tail, "apply_v3_full_system_operator_cached", _apply)

    bundle = profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=op,
        op_pc=op,
        pattern=pattern,
        active_indices=None,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.25,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=4.0,
        drop_tol=0.0,
        color_batch=2,
        emit=lambda _level, message: emitted.append(str(message)),
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **_kwargs: None,
    )

    assert bundle is not None
    assert bundle.matrix is not None
    matrix = bundle.matrix.tocsr()
    np.testing.assert_allclose(matrix.diagonal(), [2.25, 3.25, 4.25, 5.25, 0.25, 0.25])
    assert matrix[: op.f_size, op.f_size :].nnz > 0
    assert matrix[op.f_size :, : op.f_size].nnz > 0
    np.testing.assert_allclose(bundle.matvec(np.zeros(op.total_size)), np.zeros(op.total_size))
    assert bundle.metadata.shape == (op.total_size, op.total_size)
    assert "direct-tail materialization" in bundle.metadata.reason
    assert any("kinetic_pattern_nnz" in message for message in emitted)
    assert any("source_nnz" in message and "moment_nnz" in message for message in emitted)


def test_fortran_reduced_direct_tail_active_term_rejects_incomplete_fblock_blocks(
    monkeypatch,
) -> None:
    op = _constraint1_pattern_op()
    active_indices = np.asarray([0, 2, 4, 5], dtype=np.int32)
    pattern = sp.eye(int(active_indices.size), format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _active_reduced_maps(op.total_size, active_indices)
    emitted: list[str] = []

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "whichmatrix0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    _patch_structured_fblock_selection(monkeypatch, _StructuredFBlockOperator())

    diag = jnp.arange(1.0, float(op.total_size) + 1.0, dtype=jnp.float64)

    def _apply(_op, x_full):
        return diag * jnp.asarray(x_full, dtype=jnp.float64)

    monkeypatch.setattr(profile_reduced_tail, "apply_v3_full_system_operator_cached", _apply)

    bundle = profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=op,
        op_pc=op,
        pattern=pattern,
        active_indices=active_indices,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.0,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=4.0,
        drop_tol=0.0,
        color_batch=2,
        emit=lambda _level, message: emitted.append(str(message)),
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **_kwargs: None,
    )

    assert bundle is not None
    assert bundle.matrix is not None
    assert "direct-tail materialization" in bundle.metadata.reason
    assert any("active_indices_do_not_form_complete_fblock_blocks" in message for message in emitted)


def test_fortran_reduced_direct_tail_active_term_falls_back_when_projected_csr_exceeds_budget(
    monkeypatch,
) -> None:
    op = _constraint1_pattern_op()
    active_indices = np.asarray([0, 1, 4, 5], dtype=np.int32)
    pattern = sp.eye(int(active_indices.size), format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _active_reduced_maps(op.total_size, active_indices)
    emitted: list[str] = []
    fake_operator = _StructuredFBlockOperator(nnz_blocks=32)

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "whichmatrix0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_WHICHMATRIX0_FBLOCK_MAX_MB", "0.000001")
    _patch_structured_fblock_selection(monkeypatch, fake_operator)

    diag = jnp.arange(2.0, float(op.total_size) + 2.0, dtype=jnp.float64)

    def _apply(_op, x_full):
        return diag * jnp.asarray(x_full, dtype=jnp.float64)

    monkeypatch.setattr(profile_reduced_tail, "apply_v3_full_system_operator_cached", _apply)

    bundle = profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=op,
        op_pc=op,
        pattern=pattern,
        active_indices=active_indices,
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.0,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=4.0,
        drop_tol=0.0,
        color_batch=2,
        emit=lambda _level, message: emitted.append(str(message)),
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **_kwargs: None,
    )

    assert bundle is not None
    assert bundle.matrix is not None
    assert fake_operator.projected_blocks == [0]
    assert "direct-tail materialization" in bundle.metadata.reason
    assert any("projected_csr_budget_exceeded" in message for message in emitted)


def test_fortran_reduced_direct_tail_active_term_uses_projected_fblock_without_pattern_probe(
    monkeypatch,
) -> None:
    op = _constraint1_pattern_op()
    pattern = sp.eye(op.total_size, format="csr", dtype=np.float64)
    reduce_full, expand_reduced = _identity_reduced_maps(op.total_size)
    emitted: list[str] = []
    fake_operator = _StructuredFBlockOperator()

    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_ASSEMBLY", "whichmatrix0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_DIRECT_TAIL_STRUCTURED_CSR", "0")
    monkeypatch.setenv("SFINCS_JAX_RHSMODE1_FORTRAN_REDUCED_WHICHMATRIX0_FBLOCK_MAX_MB", "4")
    _patch_structured_fblock_selection(monkeypatch, fake_operator)

    def _unexpected_pattern_probe(*_args, **_kwargs):
        raise AssertionError("active term-level path should not pattern-probe the f-block")

    monkeypatch.setattr(profile_reduced_tail, "build_operator_from_pattern", _unexpected_pattern_probe)

    bundle = profile_reduced_tail._try_build_fortran_reduced_constraint1_direct_tail_bundle(
        op=op,
        op_pc=op,
        pattern=pattern,
        active_indices=np.arange(op.total_size, dtype=np.int32),
        reduce_full=reduce_full,
        expand_reduced=expand_reduced,
        pc_shift=0.5,
        dtype=jnp.float64,
        factor_dtype=np.dtype(np.float64),
        csr_max_mb=4.0,
        drop_tol=0.0,
        color_batch=2,
        emit=lambda _level, message: emitted.append(str(message)),
        build_structured_rhs1_full_csr_operator_bundle_callback=lambda **_kwargs: None,
    )

    assert bundle is not None
    assert bundle.matrix is not None
    assert fake_operator.projected_blocks == [0, 1]
    matrix = bundle.matrix.tocsr()
    np.testing.assert_allclose(matrix.diagonal()[: op.f_size], [11.5, 12.5, 13.5, 14.5])
    np.testing.assert_allclose(matrix.diagonal()[op.f_size :], [0.5, 0.5])
    assert matrix[: op.f_size, op.f_size :].nnz > 0
    assert matrix[op.f_size :, : op.f_size].nnz > 0
    assert "whichMatrix=0 active term-level" in bundle.metadata.reason
    assert any("whichMatrix=0 active term CSR built" in message for message in emitted)
