"""Assembling the operator from products with it, instead of column by column.

The sparse direct route needs a matrix and the operator is matrix-free. Reading
one column per product costs ``total_size`` products, which is why that route is
confined to small decks. These tests pin the alternative: the couplings are
known, so a product with a whole group of columns carries all of them.

What must not happen is a pattern that misses a coupling. The recovered matrix
would then be wrong in entries the pattern *does* contain, factor successfully,
and answer a different question, so the check against the operator is part of
the assembly rather than an option.
"""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as np
import pytest

from dkx.assembly import assemble_operator, f_block_groups, f_block_pattern
from dkx.drift_kinetic import kinetic_operator_from_namelist
from dkx.inputs import load_sfincs_input

DECK = """&general
/
&geometryParameters
 geometryScheme=1
 inputRadialCoordinate=3
 rN_wish=0.3
 B0OverBBar=1.0
 GHat=1.0
 IHat=0.0
 iota=1.31
 epsilon_t=0.1
 epsilon_h=0.05
 helicity_l=2
 helicity_n=5
 psiAHat=0.045
 aHat=0.1
/
&speciesParameters
 Zs=1 -1
 mHats=1.0 5.4465e-4
 nHats=1.0 1.0
 THats=1.0 1.0
 dNHatdrHats=-0.5 -0.5
 dTHatdrHats=-1.0 -1.0
/
&physicsParameters
 Delta=4.5694d-3
 alpha=1.0
 nu_n=8.4774d-3
 Er=15.0
 collisionOperator=0
 includeXDotTerm=.true.
 includeElectricFieldTermInXiDot=.true.
 useDKESExBDrift=.false.
 includePhi1=.false.
/
&resolutionParameters
 Ntheta=5
 Nzeta=5
 Nxi=6
 NL=4
 Nx=4
 solverTolerance=1d-10
/
&otherNumericalParameters
/
&preconditionerOptions
/
"""


@pytest.fixture(scope="module")
def operator(tmp_path_factory):
    path = tmp_path_factory.mktemp("assembly") / "input.namelist"
    path.write_text(DECK)
    return kinetic_operator_from_namelist(load_sfincs_input(path).raw)


def _sampled(op, *, pinned: bool = True):
    """The matrix by its definition: one product per column."""
    import jax.numpy as jnp

    from dkx.solve import _pinned_matvecs
    from solvax.native_eigen import sparse_operator_matrix

    apply = _pinned_matvecs(op)[0] if pinned else op.apply
    matrix = sparse_operator_matrix(
        jax.jit(lambda v: apply(jnp.asarray(v))), np.zeros(op.total_size), batch_size=64
    ).tocsr()
    matrix.eliminate_zeros()
    return matrix


def test_it_recovers_the_matrix_the_operator_defines(operator) -> None:
    """Entry for entry against sampling every column, not merely in norm."""
    result = assemble_operator(operator)
    reference = _sampled(operator)
    difference = result.matrix - reference
    difference.eliminate_zeros()
    assert difference.nnz == 0
    assert result.matrix.nnz == reference.nnz
    assert result.relative_error < 1e-12


def test_the_truncated_rows_are_pinned_so_the_matrix_can_be_factored(operator) -> None:
    """The rectangular layout keeps ``l >= Nxi_for_x(x)`` as exact zero rows, so
    the raw operator is structurally singular: no factorization exists. The
    solver poses the pinned system instead, and so does the assembly."""
    mask = operator.active_dof_mask()
    if mask is None:
        pytest.skip("this deck truncates nothing")
    raw = assemble_operator(operator, pinned=False).matrix
    pinned = assemble_operator(operator).matrix
    truncated = np.flatnonzero(np.asarray(mask) == 0.0)
    assert truncated.size
    assert np.count_nonzero(np.abs(raw[truncated].toarray()).sum(axis=1)) == 0
    np.testing.assert_allclose(pinned[truncated].toarray().sum(axis=1), 1.0)
    difference = pinned - _sampled(operator)
    difference.eliminate_zeros()
    assert difference.nnz == 0


def test_it_costs_far_fewer_products_than_the_matrix_has_columns(operator) -> None:
    result = assemble_operator(operator)
    assert result.products < operator.total_size
    # The bound no grouping can beat: the densest row of the pattern.
    assert result.products >= int(np.diff(f_block_pattern(operator).indptr).max())


def test_no_group_holds_two_columns_that_share_a_row(operator) -> None:
    """The property that makes one product carry a whole group."""
    pattern = f_block_pattern(operator).tocsc()
    for group in f_block_groups(operator):
        rows = np.concatenate(
            [pattern.indices[pattern.indptr[j] : pattern.indptr[j + 1]] for j in group]
        )
        assert np.unique(rows).size == rows.size


def test_a_pattern_that_misses_a_coupling_is_refused(operator, monkeypatch) -> None:
    """The failure with no other symptom: a matrix that is not the operator."""
    import dkx.assembly as assembly

    narrow = assembly.f_block_pattern(operator, l_bandwidth=0)
    monkeypatch.setattr(assembly, "f_block_pattern", lambda op, **kwargs: narrow)
    with pytest.raises(RuntimeError, match="does not reproduce the operator"):
        assembly.assemble_operator(operator)


def test_the_border_survives_the_assembly(operator) -> None:
    """The border is probed, not recovered; it must still be exact."""
    from dkx.coarse_precond import _materialize_borders

    result = assemble_operator(operator)
    b_cols, c_rows = _materialize_borders(operator)
    f_size, extra = operator.f_size, operator.extra_size
    assert extra > 0
    matrix = result.matrix.toarray()
    np.testing.assert_allclose(matrix[:f_size, f_size:], np.asarray(b_cols), atol=0.0)
    np.testing.assert_allclose(matrix[f_size:, :f_size], np.asarray(c_rows), atol=0.0)
