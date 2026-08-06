"""The L-diagonal magnetic-drift blocks must equal what the operator applies.

:meth:`KineticOperator.magnetic_drift_diagonal_parts` rebuilds, as matrices, the
part of :meth:`_magnetic_drifts` that lands on the same Legendre row it came
from.  It exists so the tier-2 coarse preconditioner can carry the drifts the
way Fortran's ``preconditioner_magnetic_drifts_max_L`` does instead of dropping
them --- worth 51 -> 30 GCROT iterations on ``magdrift_1species_tiny`` and 0-1
on the other eight drift schemes, measured through the production path.

An earlier 6000 -> 7 figure for the same change was a harness artifact: it came
from a dense pseudo-inverse of the *unpinned* stripped operator, so what it
measured was the missing ``l = 0`` and mask pins, not the missing drifts.  Real
DKX takes 51 on that deck, not 6000.  Preconditioner experiments that rebuild
the coarse operator outside :func:`dkx.coarse_precond.build_coarse_preconditioner`
measure the pins unless they reproduce them.

A preconditioner is allowed to be an approximation of the operator, so a wrong
block here would not fail loudly; it would just cost iterations, and be blamed
on the physics.  Hence exactness is checked rather than assumed.  Two bugs were
caught this way and neither was visible by inspection:

* the zeta matrix was assembled as ``(S, T, T, Z, Z)`` where the flat ``(t, z)``
  index needs ``(S, T, Z, T, Z)`` --- the theta delta belongs *between* the two
  zeta axes;
* the upwind selector must come from ``gf1`` for every factor, not from the
  factor being multiplied, which is what ``_magnetic_drifts`` does and which
  only shows up where ``gf2``/``gf3`` differ from ``gf1`` in sign.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.inputs import read_sfincs_input

REF = Path(__file__).parent / "ref"

#: Every magnetic-drift scheme that ships a tiny fixture.
SCHEMES = [
    p.name.removesuffix(".input.namelist")
    for p in sorted(REF.glob("magdrift_1species_tiny*.input.namelist"))
]


def _load(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _diagonal_block(op: KineticOperator, parts: dict, s: int, x: int, ell: int):
    """The (TZ, TZ) L-diagonal drift block for one ``(species, x, L)``."""
    c = op.magnetic_drift_diagonal_coefficients(ell)
    return (op.x[x] ** 2) * (
        c["c1"] * (parts["mt1"][s] + parts["mz1"][s])
        + c["c2"] * (parts["mt2"][s] + parts["mz2"][s])
        + c["c3"] * (parts["mt3"][s] + parts["mz3"][s])
        + c["xi"] * jnp.diag(parts["xi"][s])
    )


@pytest.mark.parametrize("name", SCHEMES)
def test_the_diagonal_blocks_reproduce_the_operator(name: str):
    """Feed one Legendre row: the output on that row *is* the diagonal block.

    The L+-2 couplings land on rows ``l0 +- 2``, so row ``l0`` isolates the
    diagonal exactly -- no fitting, no tolerance-shopping.
    """
    op = _load(name)
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    parts = op.magnetic_drift_diagonal_parts()
    assert parts is not None, "fixture carries no drifts"
    mask = op._mask()

    worst = 0.0
    row = jax.random.normal(jax.random.PRNGKey(0), (n_s, n_x, n_t, n_z))
    for ell in range(n_xi):
        f = jnp.zeros(op.f_shape).at[:, :, ell, :, :].set(row)
        applied = op._magnetic_drifts(f)[:, :, ell, :, :]
        for s in range(n_s):
            for x in range(n_x):
                block = _diagonal_block(op, parts, s, x, ell)
                v = (row[s, x] * mask[x, ell]).reshape(n_t * n_z)
                mine = (block @ v).reshape(n_t, n_z) * mask[x, ell]
                worst = max(worst, float(jnp.max(jnp.abs(mine - applied[s, x]))))

    scale = float(jnp.max(jnp.abs(op._magnetic_drifts(jnp.ones(op.f_shape)))))
    assert worst < 1e-11 * scale, f"relative error {worst / scale:.2e}"


def test_no_parts_when_the_operator_has_no_drifts():
    op = _load("pas_1species_PAS_noEr_tiny_scheme1")
    assert op.magnetic_drift_diagonal_parts() is None


@pytest.mark.parametrize("name", SCHEMES[:1])
def test_what_the_parts_hold_does_not_grow_with_x_or_l(name: str):
    """Six ``(TZ, TZ)`` matrices per species, and that is the whole point.

    Materializing the per-``L`` blocks instead would cost
    ``S * X * Nxi * TZ^2`` --- the size of a whole band, which is exactly what
    the coarse route exists to avoid on these decks.  Holding ``L``-invariant
    pieces makes the cost independent of ``Nx`` and ``Nxi``, so the saving grows
    with resolution: 6/(Nx*Nxi), which is 25% on this tiny fixture and about 1%
    at production ``Nx=6, Nxi=100``.  Asserting the exact formula pins the
    independence rather than a fraction that only holds at one resolution.
    """
    op = _load(name)
    parts = op.magnetic_drift_diagonal_parts()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    tz = n_t * n_z
    held = sum(int(v.size) for v in parts.values())
    assert held == 6 * n_s * tz * tz + n_s * tz
    per_l_blocks = n_s * n_x * n_xi * tz * tz
    assert held < per_l_blocks  # strictly cheaper at every resolution
