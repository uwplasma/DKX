"""The sparse preconditioner is the coarse one in a different order.

:mod:`dkx.sparse_precond` exists for speed and memory, so the property that has
to hold is that it changes *nothing else*: same simplified operator, same
regularization, same border elimination, and therefore the same linear map up
to factorization round-off.  These tests pin that equivalence, and the one
structural fact the module is for — that the angular blocks keep the
``createGrids.F90`` stencils instead of being filled in by eliminating ``L``
first.  Timings live in ``docs/performance.rst``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import build_coarse_preconditioner, build_tier2_preconditioner, solve
from dkx.sparse_precond import (
    assemble_simplified,
    build_sparse_preconditioner,
    simplified_subsystem_csr,
)

pytest.importorskip("scipy.sparse.linalg")

REF = Path(__file__).parent / "ref"


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _sugama_op() -> KineticOperator:
    text = (REF / "quick_2species_FPCollisions_noEr.input.namelist").read_text()
    return KineticOperator.from_namelist(
        parse_sfincs_input_text(text.replace("collisionOperator = 0", "collisionOperator = 3"))
    )


# Each entry exercises a different way the two routes could diverge: the
# collision reduction (fokker_planck, improved_sugama), the ``E_r`` terms that
# force recycled Krylov in the first place (er_xdot), and the bordered constraint rows.
CASES = {
    "pas": lambda: _load_op("pas_1species_PAS_noEr_tiny_scheme1"),
    "fokker_planck": lambda: _load_op("quick_2species_FPCollisions_noEr"),
    "improved_sugama": _sugama_op,
    "er_xdot": lambda: _load_op("er_xdot_1species_tiny"),
    "phi1_in_collision": lambda: _load_op(
        "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision"
    ),
}


def _agreement(op: KineticOperator, seed: int = 0) -> float:
    """Worst relative difference between the two preconditioners, both directions."""
    coarse, coarse_t = build_coarse_preconditioner(op)
    sparse, sparse_t = build_sparse_preconditioner(op)
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(3):
        v = jnp.asarray(rng.standard_normal(op.total_size))
        for exact, cheap in ((coarse, sparse), (coarse_t, sparse_t)):
            a = np.asarray(exact(v))
            b = np.asarray(cheap(v))
            worst = max(worst, float(np.linalg.norm(a - b) / np.linalg.norm(a)))
    return worst


# ``er_xdot`` pairs the Er xiDot term with a per-speed constraint border and is
# numerically singular -- ``dkx.solve``'s residual guard rejects its adjoint
# outright, and ``tests/test_multigrid.py`` excludes it from solution-vector
# comparisons for the same reason.  Two exact factorizations of a near-singular
# matrix agree everywhere except along its near-null direction, so comparing
# preconditioner *outputs* there measures the conditioning, not the assembly.
# It is checked below by the well-posed statement instead.
WELL_CONDITIONED = sorted(set(CASES) - {"er_xdot"})


@pytest.mark.parametrize("case", WELL_CONDITIONED)
def test_sparse_preconditioner_is_the_same_map_as_the_coarse_one(case: str) -> None:
    """Same simplified operator, same regularization — only the elimination order.

    The bound is loose against round-off because the two routes are genuinely
    different factorizations of the same matrix; it is tight enough to catch a
    dropped term, a mis-sized floor or a transposed block.
    """
    assert _agreement(CASES[case]()) < 1e-8


# The Phi1 deck's solve is a Newton iteration owned by ``dkx.phi1.solve_phi1``
# and its right-hand side is the quasineutrality residual, not ``op.rhs()``.
# Its preconditioner path -- the generalized bordered Schur complement -- is
# covered by the equivalence test above, which is the part this module changes.
LINEAR_SOLVE_CASES = sorted(set(CASES) - {"phi1_in_collision"})


@pytest.mark.parametrize("case", LINEAR_SOLVE_CASES)
def test_sparse_preconditioner_does_not_change_the_answer(case: str) -> None:
    """A preconditioner may change the iteration count and nothing else.

    Stated as a residual so it stays meaningful on the singular deck: what a
    solver owes is a small residual, and the solution vector is compared only
    where the operator determines it.
    """
    op = CASES[case]()
    rhs = op.rhs()
    reference = solve(op, rhs, method="gmres", tol=1e-10, preconditioner="coarse")
    sparse = solve(op, rhs, method="gmres", tol=1e-10, preconditioner="sparse")
    assert reference.converged and sparse.converged

    scale = max(1.0, float(jnp.linalg.norm(rhs)))
    for result in (reference, sparse):
        residual = float(jnp.linalg.norm(op.apply(result.x) - rhs)) / scale
        assert residual < 1e-6, (case, residual)

    if case != "er_xdot":
        span = max(1.0, float(jnp.max(jnp.abs(reference.x))))
        assert float(jnp.max(jnp.abs(sparse.x - reference.x))) / span < 1e-6


def test_assembly_keeps_the_angular_stencils_sparse() -> None:
    """The point of the module, and not implied by the agreement tests above.

    A dense assembly would pass every equivalence check here while costing
    exactly what the classical route costs.  ``quick_2species`` is only ``5 x 7``
    in angle, so the ratio it can reach is modest; the production decks reach
    5-8%.  Pin a bound that a dense assembly (ratio 1) cannot meet.
    """
    assembled = assemble_simplified(CASES["fokker_planck"]())
    dense_entries = assembled.dense_band_bytes / 8.0
    assert assembled.nnz < 0.6 * dense_entries


def test_subsystems_are_uncoupled_and_square() -> None:
    """One factorization per ``(species, x)``, each ``Nxi * Ntheta * Nzeta`` wide."""
    op = CASES["fokker_planck"]()
    assembled = assemble_simplified(op)
    width = op.n_xi * op.n_theta * op.n_zeta
    assert len(assembled.matrices) == op.n_species * op.n_x
    assert all(m.shape == (width, width) for m in assembled.matrices)


def test_single_subsystem_helper_agrees_with_the_batch() -> None:
    op = CASES["pas"]()
    one = simplified_subsystem_csr(op, 0, 0)
    batch = assemble_simplified(op).matrices[0]
    assert abs(one - batch).max() == 0.0


def test_l_coupling_can_be_dropped() -> None:
    """``drop_l_coupling`` (the Fortran ``preconditioner_xi=1`` knob) is block diagonal."""
    op = CASES["pas"]()
    n_tz = op.n_theta * op.n_zeta
    mat = assemble_simplified(op, drop_l_coupling=True).matrices[0].tocoo()
    assert np.array_equal(mat.row // n_tz, mat.col // n_tz)


def test_refuses_traced_operator_leaves() -> None:
    """The host assembly has no values to read under ``jit``/``grad``.

    Refusing with a message that names the working alternative is the contract:
    silently falling back to the dense route would hide a large cost change.
    """
    op = CASES["pas"]()

    with pytest.raises(NotImplementedError, match="cannot run with traced"):
        jax.jit(lambda leaf: build_sparse_preconditioner(replace(op, x=leaf)))(op.x)


def test_solve_accepts_the_sparse_route() -> None:
    """``build_tier2_preconditioner`` dispatches the name ``solve`` validates."""
    op = CASES["pas"]()
    precond, precond_t = build_tier2_preconditioner(op, "sparse")
    v = jnp.asarray(np.random.default_rng(1).standard_normal(op.total_size))
    assert np.all(np.isfinite(np.asarray(precond(v))))
    assert np.all(np.isfinite(np.asarray(precond_t(v))))
