from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from sfincs_jax.operators.profile_response.compressed_layout import build_rhs1_compressed_pitch_layout
from sfincs_jax.solvers.preconditioners.symbolic_sparse.rhs1_fortran_reduced import build_rhs1_reduced_pmat_elimination_plan


def _op(*, nxi_for_x: list[int] | None = None, total_tail: int = 5) -> SimpleNamespace:
    n_species = 2
    n_x = 3
    n_xi = 4
    n_theta = 2
    n_zeta = 3
    f_size = n_species * n_x * n_xi * n_theta * n_zeta
    if nxi_for_x is None:
        nxi_for_x = [4, 2, 1]
    return SimpleNamespace(
        n_species=n_species,
        n_x=n_x,
        n_xi=n_xi,
        n_theta=n_theta,
        n_zeta=n_zeta,
        f_size=f_size,
        total_size=f_size + total_tail,
        fblock=SimpleNamespace(
            collisionless=SimpleNamespace(n_xi_for_x=np.asarray(nxi_for_x, dtype=np.int32)),
        ),
    )


def test_reduced_pmat_plan_builds_complete_permutation_with_tail_root() -> None:
    layout = build_rhs1_compressed_pitch_layout(_op(total_tail=4))
    plan = build_rhs1_reduced_pmat_elimination_plan(
        layout,
        separator_ells=(0,),
        max_interior_group_size=7,
        max_separator_size=10_000,
    )

    assert plan.layout is layout
    assert plan.selected_separator_x_indices == (0, 1, 2)
    assert plan.selected_separator_ells == (0,)
    assert plan.tail_group.indices.tolist() == list(range(layout.tail_reduced_start, layout.reduced_size))
    assert plan.root_group.indices.tolist() == plan.separator_group.indices.tolist() + plan.tail_group.indices.tolist()
    assert plan.permutation.size == layout.reduced_size
    assert np.unique(plan.permutation).size == layout.reduced_size
    np.testing.assert_array_equal(plan.inverse_permutation[plan.permutation], np.arange(layout.reduced_size))

    separator_set = set(int(v) for v in plan.separator_group.indices)
    assert layout.reduced_kinetic_index(0, 0, 0, 0, 0) in separator_set
    assert layout.reduced_kinetic_index(1, 2, 0, 1, 2) in separator_set
    assert layout.reduced_kinetic_index(0, 0, 1, 0, 0) not in separator_set
    assert all(group.size <= 7 for group in plan.interior_groups)

    metadata = plan.metadata()
    assert metadata["interior_size"] + metadata["root_size"] == layout.reduced_size
    assert metadata["tail_size"] == 4
    assert metadata["root_dense_nbytes_estimate"] == plan.root_size * plan.root_size * 8


def test_reduced_pmat_plan_bounds_selected_x_separators() -> None:
    layout = build_rhs1_compressed_pitch_layout(_op(total_tail=0))
    plan = build_rhs1_reduced_pmat_elimination_plan(
        layout,
        separator_ells=(0, 1),
        max_interior_group_size=100,
        max_separator_size=layout.n_theta * layout.n_zeta * 2,
    )

    assert plan.selected_separator_x_indices == (0,)
    # Two species, one selected x, two selected ell values, and one theta-zeta
    # plane per retained pitch mode.
    assert plan.separator_size == 2 * 2 * layout.n_theta * layout.n_zeta
    assert plan.tail_size == 0
    assert plan.root_size == plan.separator_size


def test_reduced_pmat_plan_ignores_inactive_separator_ells_and_can_use_no_separator() -> None:
    layout = build_rhs1_compressed_pitch_layout(_op(nxi_for_x=[1, 0, 1], total_tail=2))

    no_separator = build_rhs1_reduced_pmat_elimination_plan(
        layout,
        separator_ells=(5,),
        max_interior_group_size=100,
        max_separator_size=100,
    )
    assert no_separator.separator_size == 0
    assert no_separator.root_size == layout.tail_size
    assert no_separator.interior_size == layout.kinetic_active_size

    ell0 = build_rhs1_reduced_pmat_elimination_plan(
        layout,
        separator_ells=(0,),
        max_interior_group_size=100,
        max_separator_size=100,
    )
    assert ell0.selected_separator_x_indices == (0, 1, 2)
    # x=1 has no active pitch modes, so it contributes no separator rows.
    assert ell0.separator_size == 2 * 2 * layout.n_theta * layout.n_zeta
