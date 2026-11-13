"""The coarse Legendre-row elimination is a scan, so its compile cost is flat in ``Nxi``.

``_factor_coarse_rows`` runs one ``lax.scan`` per run of rows with the same active chains.
Its value against the unrolled loop is the compiled program size, checked on the lowered
module; its exactness is checked against the Schur-LU route in
``tests/test_coarse_ragged_chains.py`` and, here, through the adjoint identity of the
sweeps that consume it and a finite difference on the pin strength.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dkx.coarse_precond as cp
from tests.test_coarse_ragged_chains import _RAMP, _edited

FACTOR = cp._factor_coarse_rows
UNIFORM = ("ref/quick_2species_FPCollisions_noEr", lambda n: [("Nxi = 8", f"Nxi = {n}")])
# the ramp scales with Nxi: 16 and 32 rows
RAMPED = ("ref/pas_1species_PAS_noEr_tiny_scheme1", lambda n: [(_RAMP[0][0], f"Nxi = {2 * n}"), *_RAMP[1:]])


def _captured(name, *edits, monkeypatch):
    """The arguments ``build_coarse_preconditioner`` hands to ``_factor_coarse_rows``."""
    seen: list = []
    monkeypatch.setattr(cp, "_factor_coarse_rows", lambda *a, **k: (seen.append((a, k)), FACTOR(*a, **k))[1])
    cp.build_coarse_preconditioner(_edited(name, *edits))
    return seen[-1]


def _ops(lowered) -> int:
    return sum(" = " in line or line.strip().startswith(("return", "stablehlo.return")) for line in lowered.as_text().splitlines())


@pytest.mark.parametrize("case", (UNIFORM, RAMPED), ids=("uniform", "ramped"))
def test_program_size_is_independent_of_nxi(case, monkeypatch):
    name, edits = case
    sizes = []
    for n_xi in (8, 16):
        args, kwargs = _captured(name, *edits(n_xi), monkeypatch=monkeypatch)
        sizes.append(_ops(FACTOR.lower(*args, **kwargs)))
    assert sizes[1] <= 1.05 * sizes[0]


def test_scanned_factors_give_the_adjoint_pair_and_are_differentiable(monkeypatch):
    name, edits = RAMPED
    args, kwargs = _captured(name, *edits(8), monkeypatch=monkeypatch)
    coef, subs, floor, gamma = args
    rows = kwargs["rows"]
    assert len(cp._row_segments(rows[1])) > 1  # a real ramp: the padded run transitions are exercised
    n_tz = coef["b0"].shape[0]
    v = jnp.asarray(np.random.default_rng(0).normal(size=(floor.shape[0], kwargs["n_xi"], n_tz)))
    w = jnp.asarray(np.random.default_rng(1).normal(size=v.shape))

    def solve(g, v, transpose=False):
        segments, _ = FACTOR(coef, subs, floor, g, **{**kwargs, "inverse": False})
        return cp._apply_dense_coarse_factors(cp._unstack(list(segments)), floor, v, transpose=transpose, rows=rows)

    forward, backward = solve(gamma, v), solve(gamma, w, True)
    assert abs(jnp.vdot(w, forward) - jnp.vdot(backward, v)) <= 1e-10 * abs(jnp.vdot(w, forward))
    grad = jax.grad(lambda g: jnp.vdot(w, solve(g, v)))(gamma)
    direction = jnp.ones_like(gamma)
    step = 1e-4 * float(jnp.abs(gamma).max())
    fd = (jnp.vdot(w, solve(gamma + step * direction, v)) - jnp.vdot(w, solve(gamma - step * direction, v))) / (2 * step)
    assert abs(jnp.vdot(grad, direction) - fd) <= 1e-3 * abs(fd)  # the ramped chains have condition numbers near 1e8
