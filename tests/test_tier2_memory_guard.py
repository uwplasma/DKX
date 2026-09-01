"""The coarse preconditioner generates what it cannot afford to store.

Its dense ``(Ntheta*Nzeta)`` bands are sized exactly by the grid, so a solve
that cannot fit them is knowable before any work happens.  Five upstream decks
need 42.9-53.3 GB of them on a 24 GB machine; left to allocate, the process is
killed by the OS part way through and produces nothing -- no traceback, no
partial output, and a return code indistinguishable from any other crash.  On
the 2026-08-01 upstream campaign that cost those decks 55-90 s each before
dying.

The route out is not a smaller preconditioner but a generated one: the same
simplified operator, the same pins, eliminated from block rows built on demand
so that no band is materialized.  There are two of those, and which one runs is
the difference between a preconditioner and a solver.

``solvax.direct.block_thomas_factor_fn(..., store_offdiagonals=False)`` keeps
only the Schur LU --- a third of the bands, a sixth with a float32 LU --- and
regenerates the two off-diagonal blocks during each substitution sweep.  The
elimination runs once, so the factors are *reusable* across the tens of Krylov
applications a solve makes.  This is the route taken wherever those factors fit.
It is not yet shown to make the 42.9-53.3 GB decks complete --- they are still
killed, for a reason that is about tracing rather than sizing
(docs/performance.rst, "Not yet demonstrated at production scale").

``solvax.direct.block_thomas_checkpointed_fn`` retains no band-sized state at
all and re-eliminates on every application, so it is much slower and is taken
only where even the Schur LU does not fit.  It is not dead: that regime is
reachable, and it is the one thing the reusable route cannot do.

Thresholds and their evidence live in
:func:`dkx.coarse_precond._coarse_bands_fit` and
:func:`dkx.coarse_precond._coarse_factors_fit`; these tests pin the behaviour,
not the constants.
"""

from __future__ import annotations

import warnings
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from dkx.coarse_precond import (
    _coarse_bands_fit,
    _coarse_factor_dtype,
    _coarse_factors_fit,
    _coarse_generated_fallback_message,
    _coarse_generated_peak_bytes,
    _coarse_reusable_fallback_message,
    _host_memory_bytes,
    build_coarse_preconditioner,
    coarse_preconditioner_band_bytes,
    coarse_preconditioner_factor_bytes,
)
import dkx.coarse_precond as coarse_precond
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import build_tier2_preconditioner, solve

REF = Path(__file__).parent / "ref"

# Host sizes that select each route for a given deck.  ``_coarse_bands_fit``
# compares the band bytes against RAM and ``_coarse_factors_fit`` the Schur-LU
# bytes, so half the band size sits between them by construction: too small for
# the bands, ample for a third of them.
def _ram_for_reusable(op: KineticOperator) -> float:
    return 0.5 * coarse_preconditioner_band_bytes(op)


def _small_op() -> KineticOperator:
    return KineticOperator.from_namelist(
        read_sfincs_input(REF / "er_xdot_1species_tiny.input.namelist")
    )


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _ramped_op() -> KineticOperator:
    """The tiny PAS fixture rescaled so ``Nxi_for_x_option=1`` gives a real ramp.

    The identity rows the ramp's truncated ``(x, l)`` pairs need are one of the
    two pins that make the coarse chain invertible, so the generated route has
    to reproduce them exactly and this deck is where that shows.
    """
    text = (
        (REF / "pas_1species_PAS_noEr_tiny_scheme1.input.namelist")
        .read_text()
        .replace("Nxi = 4", "Nxi = 16")
        .replace("Nx = 3", "Nx = 5")
        .replace("Nxi_for_x_option = 0", "Nxi_for_x_option = 1")
    )
    op = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    assert int(np.min(np.asarray(op.n_xi_for_x))) < op.n_xi
    return op


def test_band_size_is_the_exact_allocation():
    """Three dense TZ blocks per (species, x, L) — an allocation, not a guess."""
    op = _small_op()
    n_s, n_x, n_xi, n_t, n_z = op.f_shape
    assert coarse_preconditioner_band_bytes(op) == pytest.approx(
        3.0 * n_s * n_x * n_xi * (n_t * n_z) ** 2 * 8.0
    )


def test_band_size_grows_quadratically_in_the_angular_grid():
    """The quadratic in ``Ntheta*Nzeta`` is what makes big decks unrunnable."""
    op = _small_op()
    doubled = replace(op, n_theta=2 * op.n_theta)
    ratio = coarse_preconditioner_band_bytes(doubled) / coarse_preconditioner_band_bytes(op)
    assert ratio == pytest.approx(4.0)


def test_generating_the_rows_stores_an_order_of_magnitude_less():
    """The reason the fallback exists at all, stated as the two sizes.

    The bands hold ``3 Nxi`` dense blocks per subsystem; the checkpointed
    elimination holds ``Nxi/cs + 3 cs`` with ``cs ~ sqrt(Nxi)``, which is what
    turns 53.3 GB into something a 24 GB machine can run.
    """
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    tall = replace(op, n_xi=400)
    per_subsystem = coarse_preconditioner_band_bytes(tall) / (tall.n_species * tall.n_x)
    assert _coarse_generated_peak_bytes(tall) < 0.1 * per_subsystem


def test_keeping_only_the_schur_lu_is_a_third_of_the_bands():
    """The sizing the reusable route is chosen on: one of three arrays, not three.

    Stated as a ratio rather than a byte count so it pins the storage *policy*.
    One of the three ``(Nxi, m, m)`` float64 arrays survives, plus the ``(Nxi, m)``
    int32 pivots both policies keep, so the ratio is ``1/3 + 1/(6m)`` exactly ---
    which is a third to within 0.05% at a production angular grid and visibly
    above it on a tiny fixture, and the difference is the pivots rather than
    slack in the estimate.
    """
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    tall = replace(op, n_xi=400)
    m = tall.n_theta * tall.n_zeta
    ratio = coarse_preconditioner_factor_bytes(tall) / coarse_preconditioner_band_bytes(tall)
    assert ratio == pytest.approx(1.0 / 3.0 + 1.0 / (6.0 * m))

    # The HSX deck this route exists for: 25x51 angular grid, where it is a third.
    production = replace(tall, n_theta=25, n_zeta=51)
    assert coarse_preconditioner_factor_bytes(production) / coarse_preconditioner_band_bytes(
        production
    ) == pytest.approx(1.0 / 3.0, rel=5e-4)


def test_a_float32_schur_lu_halves_the_factors_again():
    """The knob that exists for the decks a third is still not enough for.

    Only the LU halves; the int32 pivots are the same either way, so the ratio
    is ``(m + 1) / (2m + 1)`` --- a half to within 0.02% at a production angular
    grid, and visibly above it on a tiny fixture.
    """
    op = _load_op("pas_1species_PAS_noEr_tiny_scheme1")
    m = op.n_theta * op.n_zeta
    ratio = coarse_preconditioner_factor_bytes(
        op, jnp.float32
    ) / coarse_preconditioner_factor_bytes(op)
    assert ratio == pytest.approx((m + 1.0) / (2.0 * m + 1.0))

    production = replace(op, n_theta=25, n_zeta=51)
    assert coarse_preconditioner_factor_bytes(
        production, jnp.float32
    ) / coarse_preconditioner_factor_bytes(production) == pytest.approx(0.5, rel=1e-3)


def test_the_factor_precision_knob_rejects_a_value_it_does_not_understand(monkeypatch):
    """A misspelled precision must not silently become the default.

    It selects how the preconditioner is factored, so a typo that fell through
    to float64 would be a silent performance and memory difference rather than
    an error.
    """
    monkeypatch.setenv("DKX_COARSE_FACTOR_DTYPE", "float16")
    with pytest.raises(ValueError, match="not a recognized precision"):
        _coarse_factor_dtype()


def test_a_deck_that_fits_keeps_the_dense_route(monkeypatch):
    """The fallback must stay out of the way wherever the fast route works."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    assert _coarse_bands_fit(_small_op())


def test_a_deck_that_does_not_fit_is_routed_to_the_generated_one(monkeypatch):
    """And it must divert wherever the bands would be allocated into a kill."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    assert not _coarse_bands_fit(_small_op())


def test_a_deck_that_misses_the_bands_by_under_3x_keeps_reusable_factors(monkeypatch):
    """The routing decision the reusable route exists to make.

    Between the two guards there is a band of machine sizes where the three
    bands do not fit but a third of them does. A deck landing there must keep
    reusable factors rather than fall all the way to the one-shot route, because
    that is the difference between amortizing the elimination over a Krylov
    solve and repeating it every application.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    op = _small_op()
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: _ram_for_reusable(op))
    assert not _coarse_bands_fit(op)
    assert _coarse_factors_fit(op)


def test_a_deck_too_small_even_for_the_schur_lu_falls_all_the_way(monkeypatch):
    """The regime the checkpointed route is kept for, and the only one."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    op = _small_op()
    assert not _coarse_bands_fit(op)
    assert not _coarse_factors_fit(op)


def test_the_warning_states_the_size_and_the_cost(monkeypatch):
    """A solve that silently got an order of magnitude slower is a bug report.

    The message has to say what did not fit, what runs instead, and how to get
    the fast route back.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    op = _small_op()
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    message = _coarse_generated_fallback_message(op)
    assert f"{op.n_theta}x{op.n_zeta}" in message  # the size that did not fit
    assert "block_thomas_checkpointed_fn" in message  # what runs instead
    assert "It completes; it is not fast." in message  # what to expect of it
    assert "reduce Ntheta/Nzeta or Nxi" in message  # how to get the fast route back
    assert "DKX_TIER2_MEMORY_GUARD=off" in message


def test_the_reusable_warning_says_what_it_keeps_and_what_it_costs(monkeypatch):
    """The route most oversized decks take must not read like the one-shot one.

    It has to say what it retains, that the factors are *reused* (which is the
    whole reason it is preferred), and how to shrink them further.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    op = _small_op()
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: _ram_for_reusable(op))
    message = _coarse_reusable_fallback_message(op)
    assert f"{op.n_theta}x{op.n_zeta}" in message  # the size that did not fit
    assert "store_offdiagonals=False" in message  # what runs instead
    assert "reused across Krylov applications" in message  # why it is preferred
    assert "It completes; it is not fast." not in message  # not the one-shot claim
    assert "DKX_COARSE_FACTOR_DTYPE=float32" in message  # how to shrink it further


@pytest.mark.parametrize(
    "message_fn", [_coarse_generated_fallback_message, _coarse_reusable_fallback_message]
)
def test_no_fallback_warning_recommends_a_route_that_also_fails(message_fn, monkeypatch):
    """Regression guard on the correction the message already carries.

    An early version of this message told users to switch to
    ``preconditioner='sparse'``; the experiment that followed found it rescues
    none of the five decks — killed on three, timed out on two.  Neither the
    generated fallback nor the reusable-factor route changed that measurement,
    so re-adding the recommendation without new measurement should still fail
    here — on both messages, since they now share the paragraph.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    assert "'sparse' stores far less but was measured killed" in message_fn(_small_op())


def test_the_oversized_deck_builds_and_warns(monkeypatch):
    """It fires through the route callers actually take, and it produces a solver."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    with pytest.warns(RuntimeWarning, match="block_thomas_checkpointed_fn"):
        precond, precond_t = build_tier2_preconditioner(_small_op(), "coarse")
    assert callable(precond) and callable(precond_t)


def test_the_reusable_route_is_what_a_near_miss_deck_builds(monkeypatch):
    """A deck that misses the bands by under 3x must warn about *that* route."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    op = _small_op()
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: _ram_for_reusable(op))
    with pytest.warns(RuntimeWarning, match="store_offdiagonals=False"):
        precond, precond_t = build_tier2_preconditioner(op, "coarse")
    assert callable(precond) and callable(precond_t)


def test_the_sparse_route_is_not_gated(monkeypatch):
    """The other preconditioner routes stay reachable whatever the bands cost."""
    pytest.importorskip("scipy.sparse.linalg")
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    precond, _ = build_tier2_preconditioner(_small_op(), "sparse")
    assert callable(precond)


def test_the_routing_can_be_switched_off(monkeypatch):
    """Someone who knows their machine better than ``sysconf`` keeps control."""
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    monkeypatch.setenv("DKX_TIER2_MEMORY_GUARD", "off")
    assert _coarse_bands_fit(_small_op())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_coarse_preconditioner(_small_op())  # the dense route, unwarned


def test_unknown_host_memory_does_not_cost_the_fast_route(monkeypatch):
    """A platform that cannot report its RAM must not be demoted on a guess."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: None)
    assert _coarse_bands_fit(_small_op())


def test_host_memory_is_readable_here():
    """Guards the assumption the routing rests on, on the platforms we run."""
    total = _host_memory_bytes()
    assert total is None or total > 2**30


# ---------------------------------------------------------------------------
# The three routes must be the same linear map.  The coarse chain is near-singular
# by construction -- its ``l = 0`` block annihilates a distribution constant on
# the flux surface, which is why the rank-one pin exists -- so two exact
# factorizations of it differ along that direction, and comparing their outputs
# on the singular deck would measure the conditioning rather than the code.  The
# equivalence is therefore stated twice: directly on the preconditioner outputs
# where the chain is well conditioned, and as a solve *residual* everywhere,
# which is well posed on every deck.  ``tests/test_multigrid.py`` and
# ``tests/test_sparse_precond.py`` split their equivalences the same way.
# ---------------------------------------------------------------------------

AGREEMENT_CASES = {
    # Each branch of the coarse operator's collision reduction.
    "pas": lambda: _load_op("pas_1species_PAS_noEr_tiny_scheme1"),
    "fokker_planck": lambda: _load_op("quick_2species_FPCollisions_noEr"),
    "phi1_in_collision": lambda: _load_op(
        "fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision"
    ),
    # The Er term that forces recycled Krylov in the first place, and the mask pin.
    "er_xdot": _small_op,
    "ramped_nxi_for_x": _ramped_op,
}


# The two generated routes, keyed by the RAM size that selects each.  Both are
# parametrized over everywhere the checkpointed one used to be tested alone: they
# eliminate the same pinned chain and differ only in what survives it, so a
# defect in the pins or the mask would show in either.
FALLBACK_ROUTES = {
    "reusable_factors": _ram_for_reusable,
    "checkpointed": lambda op: 1.0,
}


def _generated(op: KineticOperator, monkeypatch, route: str = "checkpointed"):
    """One generated route's ``(precond, precond_t)`` for the same operator."""
    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget", lambda: FALLBACK_ROUTES[route](op)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return build_coarse_preconditioner(op)


# ``er_xdot`` pairs the Er xiDot term with a per-speed constraint border and is
# numerically singular -- ``dkx.solve``'s residual guard rejects its adjoint
# outright, and ``tests/test_sparse_precond.py`` excludes it from
# solution-vector comparisons for the same reason.  It is covered below by the
# well-posed statement instead.
WELL_CONDITIONED = sorted(set(AGREEMENT_CASES) - {"er_xdot"})


@pytest.mark.parametrize("route", sorted(FALLBACK_ROUTES))
@pytest.mark.parametrize("case", WELL_CONDITIONED)
def test_the_two_routes_are_one_map_forward_and_transposed(
    case: str, route: str, monkeypatch
):
    """Generating the rows changes where the blocks come from and nothing else.

    Both routes are exact factorizations of the same pinned coarse operator, so
    what has to hold is that they act the same on a vector — loosely enough for
    two genuinely different eliminations of a near-singular chain, tightly
    enough to catch a dropped mask, an unpinned ``l = 0`` block or a
    transposition.
    """
    op = AGREEMENT_CASES[case]()
    dense = build_coarse_preconditioner(op)
    generated = _generated(op, monkeypatch, route)
    rng = np.random.default_rng(0)
    for _ in range(3):
        v = jnp.asarray(rng.standard_normal(op.total_size))
        for exact, cheap in zip(dense, generated):
            a, b = np.asarray(exact(v)), np.asarray(cheap(v))
            assert np.all(np.isfinite(b))
            assert np.linalg.norm(a - b) / np.linalg.norm(a) < 1e-8


LINEAR_SOLVE_CASES = sorted(set(AGREEMENT_CASES) - {"phi1_in_collision"})


@pytest.mark.parametrize("route", sorted(FALLBACK_ROUTES))
@pytest.mark.parametrize("case", LINEAR_SOLVE_CASES)
def test_the_generated_route_does_not_change_the_answer(
    case: str, route: str, monkeypatch
):
    """A preconditioner may change the iteration count and nothing else.

    Stated as a residual so it stays meaningful on the near-singular decks:
    what a solver owes is a small residual, and the solution vector is compared
    only where the operator determines it.
    """
    op = AGREEMENT_CASES[case]()
    rhs = op.rhs()
    reference = solve(op, rhs, method="gmres", tol=1e-10, preconditioner="coarse")
    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget", lambda: FALLBACK_ROUTES[route](op)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        generated = solve(op, rhs, method="gmres", tol=1e-10, preconditioner="coarse")
    assert reference.converged and generated.converged

    scale = max(1.0, float(jnp.linalg.norm(rhs)))
    for result in (reference, generated):
        assert float(jnp.linalg.norm(op.apply(result.x) - rhs)) / scale < 1e-9


def test_the_generated_route_is_jit_safe_over_traced_operator_leaves(monkeypatch):
    """Building it under ``jax.jit`` must compile and agree with the eager build.

    The generated route closes over per-subsystem coefficient arrays and jits
    its own application — an application otherwise re-traces ``Nxi`` generated
    block rows per subsystem — so both have to survive the operator leaves
    arriving as tracers, which is how the differentiable solve and the ``Phi1``
    Newton loop reach it.  A ramped deck is used because its truncation mask is
    non-uniform, which is the part that used to host-materialize.
    """
    import jax  # noqa: PLC0415

    op = _ramped_op()
    leaves, treedef = jax.tree_util.tree_flatten(op)
    v = jnp.asarray(np.linspace(-1.0, 1.0, op.total_size), dtype=jnp.float64)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)

    def action(values: list) -> jnp.ndarray:
        precond, _ = build_coarse_preconditioner(jax.tree_util.tree_unflatten(treedef, values))
        return precond(v)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        jitted = np.asarray(jax.jit(action)(leaves))  # compiles
        reference = np.asarray(build_coarse_preconditioner(op)[0](v))
    # Compared in norm, as the dense route's jit-safety test is: XLA is free to
    # fuse the elimination differently inside jit, and a Krylov method consumes
    # the applied vector, not its individual components.
    difference = np.linalg.norm(jitted - reference)
    assert difference <= 1e-6 * max(1.0, float(np.linalg.norm(reference))), difference


@pytest.mark.parametrize("route", sorted(FALLBACK_ROUTES))
def test_the_generated_route_carries_the_transposed_solve(route: str, monkeypatch):
    """The adjoint runs on ``precond_t``, so it needs its own residual.

    ``SolveResult.adjoint`` records ``||A^T y - g|| / ||g||`` recomputed from
    the operator once the backward pass has executed, which is the transposed
    statement of the test above: the generated route has to reach the same
    transposed residual as the dense one, and produce the same gradient.  The
    scalar is threaded through ``THat`` and the cotangent is a fixed
    pseudo-random vector — a generic linear functional, which is the hardest
    case for the adjoint solve and the one a composed objective produces.
    """
    import jax  # noqa: PLC0415

    op0 = _load_op("quick_2species_FPCollisions_noEr")
    w = jnp.asarray(np.random.default_rng(11).standard_normal(op0.total_size))
    captured: dict[str, object] = {}

    def loss(scale: jnp.ndarray) -> jnp.ndarray:
        op = replace(op0, t_hat=op0.t_hat * scale)
        result = solve(
            op, op.rhs(), method="gmres", tol=1e-10, differentiable=True,
            preconditioner="coarse",
        )  # fmt: skip
        captured["result"] = result
        return jnp.dot(w, result.x)

    dense_grad = float(jax.grad(loss)(jnp.asarray(1.0)))
    dense = captured["result"].adjoint
    assert dense.checked and dense.converged
    dense_adjoint = dense.worst_relative_residual
    assert 0.0 < dense_adjoint < 1e-8

    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget", lambda: FALLBACK_ROUTES[route](op0)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        generated_grad = float(jax.grad(loss)(jnp.asarray(1.0)))
    generated = captured["result"].adjoint
    assert generated.checked and generated.converged
    generated_adjoint = generated.worst_relative_residual

    assert 0.0 < generated_adjoint < 1e-8
    assert abs(generated_adjoint - dense_adjoint) < 1e-8
    assert abs(generated_grad - dense_grad) <= 1e-6 * max(abs(dense_grad), 1.0)


def test_the_generated_route_solves_a_phi1_deck_to_the_same_state(monkeypatch):
    """The other border, on the decks that have one.

    A ``Phi1`` operator's border is the whole quasineutrality block with a
    *nonzero* border-border term, so ``schur_projected_precond`` eliminates
    something structurally different from the constraint-only border every other
    case here exercises -- and it wraps the f-block inverse, which is exactly the
    piece these routes replace.  A preconditioner may change the Krylov path and
    nothing else, so the Newton solve has to reach the same state to solver
    tolerance.

    Run on the reusable route only, and deliberately.  What the border cares
    about is the *interface* of the f-block inverse, which both generated routes
    present identically, so parametrizing this over both would buy no coverage
    the equivalence tests above do not already give -- and the checkpointed
    variant costs 55 s against 14 s here, on a file that is already among the
    slower ones.  The route tested is the one real decks of this size take.
    """
    route = "reusable_factors"
    from dkx.phi1 import operator_from_input, solve_phi1  # noqa: PLC0415

    op = operator_from_input(REF / "pas_1species_PAS_noEr_tiny_withPhi1_inKinetic_linear.input.namelist")
    reference = solve_phi1(op, tol=1e-9, use_preconditioner=True)
    assert reference.converged

    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget", lambda: FALLBACK_ROUTES[route](op)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        generated = solve_phi1(op, tol=1e-9, use_preconditioner=True)
    assert generated.converged
    assert float(generated.residual_norm) < 1e-9

    scale = max(1.0, float(np.linalg.norm(np.asarray(reference.x))))
    assert np.linalg.norm(np.asarray(generated.x) - np.asarray(reference.x)) / scale < 1e-7


# ---------------------------------------------------------------------------
# The budget is available memory, not physical RAM
# ---------------------------------------------------------------------------
# Sizing against physical RAM admits a factor set the machine cannot actually
# hold.  Measured on filteredW7XNetCDF_2species_magneticDrifts_noEr: 14.3 GB of
# float64 factors passes "14.3 <= 24.0" on a 24 GB machine, and then thrashes --
# six hours, resident set oscillating 3.2-8.0 GB, 8.4 of 9.2 GB of swap in use,
# nothing produced.  That is a worse failure than the OOM kill this guard exists
# to prevent, because a kill at least ends.


def test_the_budget_prefers_available_memory_over_physical_ram(monkeypatch):
    monkeypatch.setattr("dkx.coarse_precond._available_memory_bytes", lambda: 4.0e9)
    monkeypatch.setattr("dkx.coarse_precond._host_memory_bytes", lambda: 64.0e9)
    assert coarse_precond._coarse_memory_budget() == 4.0e9


def test_the_budget_falls_back_to_physical_ram_when_available_is_unreadable(monkeypatch):
    """A platform that hides the honest number must not lose the guard entirely."""
    monkeypatch.setattr("dkx.coarse_precond._available_memory_bytes", lambda: None)
    monkeypatch.setattr("dkx.coarse_precond._host_memory_bytes", lambda: 64.0e9)
    assert coarse_precond._coarse_memory_budget() == 64.0e9


def test_factors_that_only_fit_physical_ram_are_refused(monkeypatch):
    """The exact shape of the six-hour thrash: fits the box, not the free space."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    factors = coarse_precond.coarse_preconditioner_factor_bytes(
        op, coarse_precond._coarse_factor_dtype()
    )
    # Room for the factors themselves but not for what holding them really costs.
    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget", lambda: factors * 1.05
    )
    assert not coarse_precond._coarse_factors_fit(op)
    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget",
        lambda: factors * coarse_precond._COARSE_RESIDENT_OVERHEAD * 1.01,
    )
    assert coarse_precond._coarse_factors_fit(op)


def test_the_resident_overhead_matches_what_was_measured():
    """7.2 GB of float32 factors peaked at 8.87 GB resident: 1.23x."""
    assert 1.2 <= coarse_precond._COARSE_RESIDENT_OVERHEAD <= 1.35
    predicted = 7.2 * coarse_precond._COARSE_RESIDENT_OVERHEAD
    assert abs(predicted - 8.87) < 0.35, f"predicts {predicted:.2f} GB against 8.87 measured"


def test_the_fallback_names_float32_when_that_keeps_reusable_factors(monkeypatch):
    """Do not describe the one-shot route to someone who can avoid it."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    small = coarse_precond.coarse_preconditioner_factor_bytes(op, jnp.float32)
    monkeypatch.setattr(
        "dkx.coarse_precond._coarse_memory_budget",
        lambda: small * coarse_precond._COARSE_RESIDENT_OVERHEAD * 1.5,
    )
    monkeypatch.setenv("DKX_COARSE_FACTOR_DTYPE", "float64")
    message = coarse_precond._coarse_generated_fallback_message(op)
    assert message.startswith("DKX_COARSE_FACTOR_DTYPE=float32")
    assert "thrashed for six hours" in message


def test_the_float32_hint_is_absent_when_float32_would_not_help_either(monkeypatch):
    op = _load_op("quick_2species_FPCollisions_noEr")
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0)
    monkeypatch.setenv("DKX_COARSE_FACTOR_DTYPE", "float64")
    assert not coarse_precond._coarse_generated_fallback_message(op).startswith(
        "DKX_COARSE_FACTOR_DTYPE=float32"
    )


def test_the_float32_hint_is_absent_when_already_on_float32(monkeypatch):
    op = _load_op("quick_2species_FPCollisions_noEr")
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: 1.0e15)
    monkeypatch.setenv("DKX_COARSE_FACTOR_DTYPE", "float32")
    assert coarse_precond._coarse_downgrade_hint(op, jnp.float32) == ""


def test_the_band_route_carries_its_own_resident_overhead():
    """Bands are three arrays where the factors are one; one constant cannot serve both.

    Measured on a 62 GB box: the 42.9 GB-band decks each peaked at 58.5-58.9 GB
    and were OOM-killed 43 s in, because 42.9 * 1.25 = 53.6 had passed a ~54 GB
    budget.  Borrowing the factor route's overhead for the bands is exactly that
    bug.
    """
    assert (
        coarse_precond._COARSE_BAND_RESIDENT_OVERHEAD
        > coarse_precond._COARSE_RESIDENT_OVERHEAD
    )
    assert 42.9 * coarse_precond._COARSE_BAND_RESIDENT_OVERHEAD > 58.9, (
        "the band overhead must cover the 58.9 GB that was actually observed"
    )


def test_the_42gb_bands_are_refused_on_a_62gb_box(monkeypatch):
    """The exact configuration that was OOM-killed on office."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    bands = coarse_precond.coarse_preconditioner_band_bytes(op)
    # A budget that the raw band size clears but the real transient does not.
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: bands * 1.3)
    assert not coarse_precond._coarse_bands_fit(op)
    monkeypatch.setattr("dkx.coarse_precond._coarse_memory_budget", lambda: bands * 1.6)
    assert coarse_precond._coarse_bands_fit(op)
