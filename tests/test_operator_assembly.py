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

import itertools
from pathlib import Path
from types import SimpleNamespace

import jax
import numpy as np
import pytest

from dkx.assembly import (
    _angular_classes,
    assemble_operator,
    f_block_groups,
    f_block_pattern,
)
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


def test_pattern_and_groups_include_the_wider_magnetic_upwind_stencil() -> None:
    """Magnetic-drift upwinding can reach farther than the centred derivative."""

    def cyclic_stencil(n: int, offsets: tuple[int, ...]) -> np.ndarray:
        matrix = np.zeros((n, n))
        rows = np.arange(n)
        for offset in offsets:
            matrix[rows, (rows + offset) % n] = 1.0
        return matrix

    n_theta, n_zeta = 5, 9
    op = SimpleNamespace(
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_shape=(1, 1, 3, n_theta, n_zeta),
        with_magnetic_drifts=True,
        ddtheta=cyclic_stencil(n_theta, (-1, 1)),
        ddzeta=cyclic_stencil(n_zeta, (-1, 1)),
        ddtheta_magdrift_plus=None,
        ddtheta_magdrift_minus=None,
        ddzeta_magdrift_plus=cyclic_stencil(n_zeta, (-3, -2, -1, 0, 1, 2)),
        ddzeta_magdrift_minus=cyclic_stencil(n_zeta, (-2, -1, 0, 1, 2, 3)),
    )
    op.f_size = int(np.prod(op.f_shape))
    index = np.arange(op.f_size).reshape(op.f_shape)
    pattern = f_block_pattern(op).tocsc()

    row = index[0, 0, 1, 2, 0]
    plus_column = index[0, 0, 1, 2, 3]
    minus_column = index[0, 0, 1, 2, 6]
    assert pattern[row, plus_column]
    assert pattern[row, minus_column]

    for group in f_block_groups(op):
        rows = np.concatenate(
            [pattern.indices[pattern.indptr[j] : pattern.indptr[j + 1]] for j in group]
        )
        assert np.unique(rows).size == rows.size

    op.with_magnetic_drifts = False
    centred_pattern = f_block_pattern(op)
    assert not centred_pattern[row, plus_column]
    assert not centred_pattern[row, minus_column]


@pytest.mark.parametrize(
    "n, radius", [(11, 2), (15, 2), (5, 2), (12, 1), (7, 3), (16, 2), (13, 1)]
)
def test_every_angular_class_clears_the_stencil_both_ways(n: int, radius: int) -> None:
    """The property the grouping rests on, including around the wrap."""
    classes = _angular_classes(n, radius)
    covered = np.concatenate(classes)
    assert np.array_equal(np.sort(covered), np.arange(n)), "must be a partition"
    for members in classes:
        for a, b in itertools.combinations(members.tolist(), 2):
            gap = abs(a - b)
            assert min(gap, n - gap) > 2 * radius, (n, radius, members)


def test_the_classes_beat_a_dividing_stride_where_the_grid_is_unhelpful() -> None:
    """A separation that must divide the grid is the wasteful part.

    At ``n = 11`` with radius 2 the only divisor above 4 is 11, so a dividing
    stride puts every point in its own class. ``{0, 5}`` is legal -- its cyclic
    gaps are 5 and 6 -- and pairing points that way is one product saved per
    pair on the assembly.
    """
    classes = _angular_classes(11, 2)
    assert len(classes) == 6, [c.tolist() for c in classes]
    assert sorted(len(c) for c in classes) == [1, 2, 2, 2, 2, 2]


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


def test_public_w7x_full_fp_operator_assembles_with_magnetic_drifts() -> None:
    """The checked reduced W7-X deck exercises the real upwind drift support."""
    from dkx.validation.data_fetch import resolve_external_equilibrium

    equilibrium = resolve_external_equilibrium(
        "wout_w7x_standardConfig.nc", fetch=False
    )
    if equilibrium is None:
        pytest.skip("optional public W7-X equilibrium data is not cached")

    deck = (
        Path(__file__).parent
        / "reduced_inputs"
        / "filteredW7XNetCDF_2species_magneticDrifts_withEr.input.namelist"
    )
    raw = load_sfincs_input(deck).raw
    raw.group("geometryParameters")["EQUILIBRIUMFILE"] = str(equilibrium)
    public_operator = kinetic_operator_from_namelist(raw)

    result = assemble_operator(public_operator)
    reference = _sampled(public_operator)
    difference = result.matrix - reference

    assert result.relative_error < 1e-12
    assert result.matrix.nnz == reference.nnz
    assert np.max(np.abs(difference.data), initial=0.0) < 1e-15
