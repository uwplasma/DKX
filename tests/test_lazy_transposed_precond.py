"""The transposed coarse preconditioner is built on first use, and is the same map.

``build_coarse_preconditioner`` returns ``(precond, precond_t)``. Building the
bordered projection applies the coarse inverse to every border column, and for
``precond_t`` a non-differentiable solve never needs that work
(docs/experiments/2026-09-13-preconditioner-cost-anatomy.md).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import dkx.coarse_precond as coarse_precond
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from dkx.solve import build_coarse_preconditioner, solve

REF = Path(__file__).parent / "ref"
DECK = "quick_2species_FPCollisions_noEr"


def _load_op(name: str = DECK) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _vectors(op: KineticOperator, count: int = 2) -> list[jnp.ndarray]:
    rng = np.random.default_rng(20260913)
    return [jnp.asarray(rng.standard_normal(op.total_size)) for _ in range(count)]


def _count_projections(monkeypatch) -> list[tuple]:
    calls: list[tuple] = []
    real = coarse_precond.schur_projected_precond

    def counted(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(coarse_precond, "schur_projected_precond", counted)
    return calls


def test_a_non_differentiable_solve_never_builds_the_transposed_projection(monkeypatch):
    op = _load_op()
    assert op.extra_size > 0  # the bordered route, where the projection exists
    calls = _count_projections(monkeypatch)

    result = solve(op, op.rhs(), method="gmres", tol=1e-10)

    assert result.converged
    assert len(calls) == 1
    precond_t = result.precond[1]
    (u,) = _vectors(op, 1)
    precond_t(u)
    precond_t(u)
    assert len(calls) == 2


def test_the_lazy_transpose_is_the_transpose_of_the_preconditioner():
    op = _load_op()
    precond, precond_t = build_coarse_preconditioner(op)
    u, v = _vectors(op)

    lhs = float(jnp.vdot(u, precond(v)))
    rhs = float(jnp.vdot(precond_t(u), v))

    scale = float(jnp.linalg.norm(u) * jnp.linalg.norm(precond(v)))
    assert abs(lhs - rhs) <= 1e-12 * scale


def test_the_lazy_transpose_matches_the_eager_construction(monkeypatch):
    op = _load_op()
    (u,) = _vectors(op, 1)
    _, lazy = build_coarse_preconditioner(op)
    monkeypatch.setattr(coarse_precond, "_any_traced", lambda *trees: True)
    _, eager = build_coarse_preconditioner(op)

    np.testing.assert_allclose(np.asarray(lazy(u)), np.asarray(eager(u)), rtol=1e-13, atol=0.0)


def test_a_first_application_inside_jit_caches_no_tracer():
    op = _load_op()
    _, precond_t = build_coarse_preconditioner(op)
    (u,) = _vectors(op, 1)

    inside = jax.jit(precond_t)(u)
    outside = precond_t(u)
    doubled = jax.jit(lambda w: 2.0 * precond_t(w))(u)

    np.testing.assert_allclose(np.asarray(inside), np.asarray(outside), rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(np.asarray(doubled), 2.0 * np.asarray(outside), rtol=1e-13, atol=0.0)


def test_traced_operator_leaves_still_build_the_transpose():
    op = _load_op()
    (u,) = _vectors(op, 1)
    eager = build_coarse_preconditioner(op)[1](u)

    traced = jax.jit(lambda o, w: build_coarse_preconditioner(o)[1](w))(op, u)

    # Tracing the whole elimination lets XLA fuse it differently from the eager
    # build, which moves the last digits of this near-singular chain: 6e-11 on
    # this deck with the eager transpose as well, and 7e-12 on the forward map.
    # Per entry the rounding is unbounded where an entry is tiny: with one BLAS
    # thread a CI runner moved one of 2804 entries by 3.7e-12, 1.2e-9 of its
    # value. A dropped term or a wrong transpose moves the whole vector, so the
    # comparison is on the norm.
    traced, eager = np.asarray(traced), np.asarray(eager)
    assert np.linalg.norm(traced - eager) <= 1e-9 * np.linalg.norm(eager)
