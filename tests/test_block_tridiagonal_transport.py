"""Focused tests for the RHSMode=3 solvax block-tridiagonal (block-Thomas) direct path."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("solvax", reason="the block-Thomas path requires solvax")

from sfincs_jax.namelist import read_sfincs_input  # noqa: E402
from sfincs_jax.operators.profile_system import (  # noqa: E402
    apply_v3_full_system_operator,
    full_system_operator_from_namelist,
    rhs_v3_full_system,
    with_transport_rhs_settings,
)
from sfincs_jax.problems.transport_solve import solve_v3_transport_matrix_linear_gmres  # noqa: E402
from sfincs_jax.solvers.block_tridiagonal_transport import (  # noqa: E402
    build_rhs3_block_thomas_solver,
    extract_rhs3_legendre_blocks,
    rhs3_block_tridiagonal_requested,
    solve_rhs3_block_tridiagonal,
    solve_rhs3_block_tridiagonal_truncated,
)

_SMALL_CASES = (
    "monoenergetic_PAS_tiny_scheme1",
    "monoenergetic_PAS_tiny_scheme11",
)


def _nml(base: str):
    return read_sfincs_input(Path(__file__).parent / "ref" / f"{base}.input.namelist")


@pytest.mark.parametrize("base", _SMALL_CASES)
def test_block_tridiagonal_matches_existing_path(base: str) -> None:
    """The direct block-Thomas path must reproduce the existing RHSMode=3 solve."""
    nml = _nml(base)
    res_bt = solve_v3_transport_matrix_linear_gmres(nml=nml, solve_method="block_tridiagonal")
    res_ref = solve_v3_transport_matrix_linear_gmres(nml=nml, solve_method="auto")

    # The switch must actually have been exercised.
    assert set(res_bt.solver_kinds_by_rhs.values()) == {"block_tridiagonal"}
    assert set(res_bt.solve_methods_by_rhs.values()) == {"block_tridiagonal"}

    tm_bt = np.asarray(res_bt.transport_matrix, dtype=np.float64)
    tm_ref = np.asarray(res_ref.transport_matrix, dtype=np.float64)
    np.testing.assert_allclose(tm_bt, tm_ref, rtol=1e-8, atol=0.0)

    # The direct solve is essentially exact: true residuals far below the GMRES target.
    for which_rhs, res_norm in res_bt.residual_norms_by_rhs.items():
        rhs_norm = float(res_bt.rhs_norms_by_rhs[which_rhs])
        assert float(res_norm) <= 1e-9 * max(rhs_norm, 1.0)

    # State vectors (including the constraint/source unknown) agree between paths.
    for which_rhs, x_bt in res_bt.state_vectors_by_rhs.items():
        x_ref = res_ref.state_vectors_by_rhs[which_rhs]
        scale = float(jnp.linalg.norm(x_ref))
        assert float(jnp.linalg.norm(x_bt - x_ref)) <= 1e-8 * max(scale, 1.0)


def test_env_switch_enables_block_tridiagonal(monkeypatch) -> None:
    """SFINCS_JAX_RHS3_BLOCK_THOMAS=1 activates the path without a solve_method change."""
    monkeypatch.setenv("SFINCS_JAX_RHS3_BLOCK_THOMAS", "1")
    assert rhs3_block_tridiagonal_requested(solve_method="auto", rhs_mode=3)
    assert not rhs3_block_tridiagonal_requested(solve_method="auto", rhs_mode=1)
    res = solve_v3_transport_matrix_linear_gmres(
        nml=_nml("monoenergetic_PAS_tiny_scheme1"), solve_method="auto"
    )
    assert set(res.solver_kinds_by_rhs.values()) == {"block_tridiagonal"}


def test_requested_flag_off_by_default(monkeypatch) -> None:
    monkeypatch.delenv("SFINCS_JAX_RHS3_BLOCK_THOMAS", raising=False)
    assert not rhs3_block_tridiagonal_requested(solve_method="auto", rhs_mode=3)
    assert rhs3_block_tridiagonal_requested(solve_method="block_tridiagonal", rhs_mode=3)
    assert rhs3_block_tridiagonal_requested(solve_method="BLOCK_THOMAS", rhs_mode=3)
    assert not rhs3_block_tridiagonal_requested(solve_method="block_tridiagonal", rhs_mode=2)


def test_nullspace_handling_solution_independent_of_regularization() -> None:
    """The rank-one null-space regularization must not affect the solution.

    ``A~ = A + gamma B C`` is exact for any nonzero ``gamma`` (the analog of the
    arbitrary pinned-row choice in classic null-space fixes): the solved state,
    including the source unknown, must be unchanged when ``gamma`` varies by
    orders of magnitude.
    """
    nml = _nml("monoenergetic_PAS_tiny_scheme1")
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    ops = [with_transport_rhs_settings(op0, which_rhs=w) for w in (1, 2)]
    rhs_mat = jnp.stack([rhs_v3_full_system(op) for op in ops], axis=1)

    blocks = extract_rhs3_legendre_blocks(op0)
    solver_default = build_rhs3_block_thomas_solver(blocks)
    gamma0 = float(solver_default.gamma)
    x_default = np.asarray(solver_default.solve(rhs_mat))

    for gamma in (gamma0 * 1e2, gamma0 * 1e-2):
        x_gamma = np.asarray(build_rhs3_block_thomas_solver(blocks, gamma=gamma).solve(rhs_mat))
        np.testing.assert_allclose(
            x_gamma, x_default, rtol=0.0, atol=1e-9 * max(float(np.abs(x_default).max()), 1.0)
        )

    # The constraint row (flux-surface average of the l=0 component) is satisfied
    # and the full bordered system is solved to near machine precision.
    m = int(op0.n_theta) * int(op0.n_zeta)
    for idx, op in enumerate(ops):
        x = jnp.asarray(x_default[:, idx])
        c_dot_f0 = float(jnp.dot(blocks.constraint_row, x[:m]))
        assert abs(c_dot_f0) <= 1e-12 * max(float(jnp.linalg.norm(x[:m])), 1.0)
        residual = float(jnp.linalg.norm(apply_v3_full_system_operator(op, x) - rhs_mat[:, idx]))
        rhs_norm = float(jnp.linalg.norm(rhs_mat[:, idx]))
        assert residual <= 1e-9 * max(rhs_norm, 1.0)


@pytest.mark.parametrize("base", _SMALL_CASES)
def test_truncated_solve_matches_full_solve_on_low_blocks(base: str) -> None:
    """The memory-lean truncated solve reproduces the l<=2 blocks and the source.

    RHSMode=3 right-hand sides vanish for l >= 3 and all transport moments touch
    only l <= 2, so ``block_thomas_truncated_fn`` with on-the-fly probed blocks
    must give the same transport matrix as the full band solve.
    """
    from sfincs_jax.discretization.v3 import geometry_from_namelist, grids_from_namelist
    from sfincs_jax.problems.transport_diagnostics import v3_transport_matrix_from_state_vectors

    nml = _nml(base)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    rhs_mat = jnp.stack(
        [rhs_v3_full_system(with_transport_rhs_settings(op0, which_rhs=w)) for w in (1, 2)],
        axis=1,
    )
    # RHS support really is l <= 2 (density drive: l=0,2; E_parallel drive: l=1).
    m = int(op0.n_theta) * int(op0.n_zeta)
    rhs_f = np.asarray(rhs_mat)[: int(op0.f_size)].reshape(int(op0.n_xi), m, 2)
    assert np.max(np.abs(rhs_f[3:])) == 0.0

    x_full = solve_rhs3_block_tridiagonal(op=op0, rhs_columns=rhs_mat)
    x_tr = solve_rhs3_block_tridiagonal_truncated(op=op0, rhs_columns=rhs_mat, keep_lowest=3)

    scale = float(jnp.max(jnp.abs(x_full[: 3 * m])))
    np.testing.assert_allclose(
        np.asarray(x_tr[: 3 * m]), np.asarray(x_full[: 3 * m]), rtol=0.0, atol=1e-9 * max(scale, 1.0)
    )
    np.testing.assert_allclose(
        float(x_tr[int(op0.f_size), 0]), float(x_full[int(op0.f_size), 0]), rtol=1e-8, atol=1e-14
    )

    grids = grids_from_namelist(nml)
    geom = geometry_from_namelist(nml=nml, grids=grids)
    tm_full = np.asarray(
        v3_transport_matrix_from_state_vectors(
            op0=op0, geom=geom, state_vectors_by_rhs={1: x_full[:, 0], 2: x_full[:, 1]}
        )
    )
    tm_tr = np.asarray(
        v3_transport_matrix_from_state_vectors(
            op0=op0, geom=geom, state_vectors_by_rhs={1: x_tr[:, 0], 2: x_tr[:, 1]}
        )
    )
    np.testing.assert_allclose(tm_tr, tm_full, rtol=1e-10, atol=0.0)


def test_direct_solve_matches_frozen_fortran_state_vector() -> None:
    """One-shot direct solve reproduces the frozen v3 PETSc state vector."""
    from sfincs_jax.validation.fortran import read_petsc_vec

    here = Path(__file__).parent
    base = "monoenergetic_PAS_tiny_scheme1"
    nml = _nml(base)
    op0 = full_system_operator_from_namelist(nml=nml, identity_shift=0.0)
    op = with_transport_rhs_settings(op0, which_rhs=1)
    rhs = rhs_v3_full_system(op)

    x = solve_rhs3_block_tridiagonal(op=op0, rhs_columns=rhs)
    x_ref = read_petsc_vec(here / "ref" / f"{base}.whichRHS1.stateVector.petscbin").values
    np.testing.assert_allclose(np.asarray(x), x_ref, rtol=0, atol=1e-9)
