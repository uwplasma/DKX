"""Structured direct defect correction is the library recurrence, unchanged.

The structured direct factor solve is followed by iterative refinement because the
block-Thomas elimination, though backward-stable, can leave the true relative
residual a small multiple of eps above the production gate.  That recurrence
moved from a hand-rolled line in :mod:`dkx.solve` to
:func:`solvax.refine.iterative_refinement`, which owns it, reports the residual
of every sweep, and carries the float32-factor variant.

What has to hold is that the move changed nothing: one sweep of the library
recurrence is the pass it replaced, to the bit.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from solvax.refine import iterative_refinement

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from dkx.solve import _TIER1_REFINEMENT_SWEEPS, build_tier1_solver, solve

REF = Path(__file__).parent / "ref"


def _op() -> KineticOperator:
    return KineticOperator.from_namelist(
        read_sfincs_input(REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
    )


@pytest.mark.parametrize("n_rhs", [None, 3])
def test_one_sweep_reproduces_the_hand_rolled_pass(n_rhs):
    """``x + solve(b - A x)`` is exactly what one sweep computes.

    Checked for a single vector and for a right-hand-side matrix, because the
    replaced code branched on ``b.ndim`` to decide whether to vmap the operator
    and a library that quietly disagreed there would be invisible in RHSMode 1.
    """
    op = _op()
    solver = build_tier1_solver(op)
    rng = np.random.default_rng(0)
    shape = (op.total_size,) if n_rhs is None else (op.total_size, n_rhs)
    b = jnp.asarray(rng.standard_normal(shape))
    apply2d = op.apply if b.ndim == 1 else jax.vmap(op.apply, in_axes=1, out_axes=1)

    x0 = solver.solve(b)
    hand_rolled = x0 + solver.solve(b - apply2d(x0))
    library, _ = iterative_refinement(apply2d, b, solver.solve, iterations=1)

    assert np.array_equal(np.asarray(library), np.asarray(hand_rolled))


def test_the_sweep_count_is_what_the_solver_uses():
    """A silent bump would change the cost of every structured direct solve."""
    assert _TIER1_REFINEMENT_SWEEPS == 1


def test_refinement_reports_a_decreasing_residual():
    """The per-sweep norms are the diagnostic the hand-rolled pass discarded.

    Refinement that is not converging is worth knowing about; before the move
    there was nothing to inspect.
    """
    op = _op()
    solver = build_tier1_solver(op)
    b = jnp.asarray(np.random.default_rng(1).standard_normal(op.total_size))
    _, residuals = iterative_refinement(op.apply, b, solver.solve, iterations=2)
    residuals = np.asarray(residuals)
    assert residuals.shape == (3,)
    assert residuals[-1] <= residuals[0]


def test_the_tier1_solve_still_meets_its_residual_gate():
    """End to end: the property refinement exists to deliver."""
    op = _op()
    rhs = op.rhs()
    result = solve(op, rhs, method="block_tridiagonal", tol=1e-10)
    assert result.converged
    scale = max(float(jnp.linalg.norm(rhs)), 1.0)
    assert float(jnp.linalg.norm(op.apply(result.x) - rhs)) / scale < 1e-12


def test_transposed_solve_is_refined_too():
    """The adjoint path shares the recurrence; an unrefined transpose would
    only show up as a slightly wrong gradient, which is the hardest failure to
    notice."""
    op = _op()
    solver = build_tier1_solver(op)
    from dkx.solve import _transposed_apply

    apply_t = _transposed_apply(op)
    b = jnp.asarray(np.random.default_rng(2).standard_normal(op.total_size))
    x, residuals = iterative_refinement(
        apply_t, b, lambda r: solver.solve(r, transpose=True), iterations=1
    )
    assert float(jnp.linalg.norm(apply_t(x) - b)) / float(jnp.linalg.norm(b)) < 1e-10
    assert np.asarray(residuals)[-1] <= np.asarray(residuals)[0]
