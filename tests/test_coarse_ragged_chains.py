"""The dense coarse route factors generated rows, and only the rows ``Nxi_for_x`` keeps.

Two claims make that exact rather than approximate, and both are checked here
numerically instead of being read off the code.

The dense route and the reusable Schur-LU route factor the *same* pinned rows
from the *same* generator, so their f-block inverses agree to rounding, not to
the ``1e-8`` the equivalence tests in ``tests/test_tier2_memory_guard.py`` allow
between genuinely different eliminations.

A row past a subsystem's ``Nxi_for_x`` has no coupling in or out, and its
diagonal is ``(1 + floor) I``, where ``floor`` is the ``1e-8`` relative
invertibility floor.  It is not the bare identity, because the floor is added to
every row.  So the dense route returns ``r / (1 + floor)`` on those rows without
factoring them, which is what an elimination through them computes.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dkx.coarse_precond as cp
from dkx.drift_kinetic import KineticOperator
from dkx.namelist import parse_sfincs_input_text, read_sfincs_input
from dkx.solve import solve

TESTS = Path(__file__).parent


def _load(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(TESTS / f"{name}.input.namelist"))


def _edited(name: str, *edits: tuple[str, str]) -> KineticOperator:
    text = (TESTS / f"{name}.input.namelist").read_text()
    for old, new in edits:
        assert old in text
        text = text.replace(old, new)
    return KineticOperator.from_namelist(parse_sfincs_input_text(text))


_RAMP = (("Nxi = 4", "Nxi = 16"), ("Nx = 3", "Nx = 5"), ("Nxi_for_x_option = 0", "Nxi_for_x_option = 1"))
DECKS = {
    # Two species and a real Nxi_for_x ramp, [4, 4, 8, 12, 14] of 14.
    "pas_2species_ramp": lambda: _load("reduced_inputs/geometryScheme4_2species_PAS_noEr"),
    "fokker_planck": lambda: _load("ref/quick_2species_FPCollisions_noEr"),
    "improved_sugama": lambda: _edited(
        "ref/quick_2species_FPCollisions_noEr", ("collisionOperator = 0", "collisionOperator = 3")
    ),
    "phi1_in_collision": lambda: _load("ref/fp_1species_FPCollisions_noEr_tiny_withPhi1_inCollision"),
    "magnetic_drift": lambda: _load("ref/magdrift_1species_tiny"),
    # tests/test_tier2_memory_guard.py::_ramped_op, [4, 5, 9, 14, 16] of 16.
    "pas_ramp": lambda: _edited("ref/pas_1species_PAS_noEr_tiny_scheme1", *_RAMP),
}
TRUNCATING = ("pas_2species_ramp", "pas_ramp")
REUSABLE = {"_coarse_bands_fit": lambda _op, *_: False, "_coarse_factors_fit": lambda _op, *_: True}
#: The tests of one chain's rows keep it whole: the parity split of a symmetric deck
#: (tested on its own below) rotates every row, so its rounding is not bitwise.
UNSPLIT = {"_coarse_parity": lambda *_: None}


def _inverses(op: KineticOperator, monkeypatch, **patches):
    """``(a_inv, a_inv_t, precond, precond_t)`` of the route ``patches`` select.

    The f-block inverses are what this change touches; the border's Schur
    complement wraps them and amplifies rounding on the near-singular ramped
    decks (chain condition numbers near 1e8), so the bordered maps are compared
    too, but more loosely.
    """
    captured: dict[str, object] = {}
    real_schur, real_lazy = cp.schur_projected_precond, cp._lazy_projected_precond

    def schur(a_inv, *args, **kwargs):
        captured.setdefault("a_inv", a_inv)
        return real_schur(a_inv, *args, **kwargs)

    real_border = cp._bordered_precond

    def border(apply_fn, a_inv, *args, **kwargs):
        captured.setdefault("a_inv", a_inv)
        return real_border(apply_fn, a_inv, *args, **kwargs)

    def lazy(op_, build, a_inv_t, *args, **kwargs):
        captured["a_inv_t"] = a_inv_t
        return real_lazy(op_, build, a_inv_t, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, "schur_projected_precond", schur)
        patch.setattr(cp, "_lazy_projected_precond", lazy)
        patch.setattr(cp, "_bordered_precond", border)
        for name, value in patches.items():
            patch.setattr(cp, name, value)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            precond, precond_t = cp.build_coarse_preconditioner(op)
    assert op.extra_size > 0  # every deck here has a border, so both were captured
    return captured["a_inv"], captured["a_inv_t"], precond, precond_t


def _relative(a, b) -> float:
    a, b = np.asarray(a), np.asarray(b)
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _floor(op: KineticOperator) -> np.ndarray:
    coef = cp._truncated_coefficients(cp._strip_for_coarse(op))
    c0 = op._fs_average_factor().reshape(-1)
    return np.asarray(cp._coarse_generated_block_data(op, coef, op._mask(), None, c0, False)[1])


@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_dense_route_is_the_reusable_route_to_rounding(name: str, monkeypatch):
    op = DECKS[name]()
    dense = _inverses(op, monkeypatch)
    reusable = _inverses(op, monkeypatch, **REUSABLE)
    rng = np.random.default_rng(0)
    for _ in range(2):
        f = jnp.asarray(rng.standard_normal(op.f_size))
        v = jnp.asarray(rng.standard_normal(op.total_size))
        for got, ref in zip(dense[:2], reusable[:2], strict=True):  # forward, transposed
            # An explicit inverse loses cond * eps (cond <= 1e7 here), the LU does not.
            assert _relative(got(f), ref(f)) <= 1e-10
        for got, ref in zip(dense[2:], reusable[2:], strict=True):
            out = np.asarray(got(v))
            assert np.all(np.isfinite(out))
            assert _relative(out, ref(v)) <= 1e-9


@pytest.mark.parametrize("name", TRUNCATING)
def test_truncated_rows_are_an_uncoupled_scaled_identity(name: str):
    """The claim the ragged chains rest on, read off the generated rows themselves."""
    op = DECKS[name]()
    coef = cp._truncated_coefficients(cp._strip_for_coarse(op))
    c0 = op._fs_average_factor().reshape(-1)
    subs, floor, gamma = cp._coarse_generated_block_data(op, coef, op._mask(), None, c0, False)
    rows = cp._coarse_pinned_block_fns(coef, op.n_xi, subs, floor, gamma, False)
    eye = np.eye(op.n_theta * op.n_zeta)
    order, counts = cp._coarse_row_layout(op)
    length = {b: sum(b in order[:c] for c in counts) for b in order}
    assert len(set(length.values())) > 1  # the ramp really does shorten some chains
    for b, n_active in length.items():
        if n_active < op.n_xi:
            _, _, upper = rows[b](jnp.asarray(n_active - 1, dtype=jnp.int32))
            assert not np.any(np.asarray(upper))  # no coupling out of the active chain
        for j in range(n_active, op.n_xi):
            lower, diag, upper = (np.asarray(a) for a in rows[b](jnp.asarray(j, dtype=jnp.int32)))
            assert not np.any(lower) and not np.any(upper)
            np.testing.assert_array_equal(diag, (1.0 + float(floor[b])) * eye)


@pytest.mark.parametrize("name", TRUNCATING)
def test_truncated_rows_pass_through_scaled_by_the_floor(name: str, monkeypatch):
    op = DECKS[name]()
    dense = _inverses(op, monkeypatch, **UNSPLIT)
    reusable = _inverses(op, monkeypatch, **REUSABLE, **UNSPLIT)
    floor = _floor(op).reshape(op.n_species, op.n_x, 1, 1, 1)
    padded = np.broadcast_to(np.asarray(op._mask())[None, :, :, None, None] == 0.0, op.f_shape)
    assert padded.any()
    f = np.random.default_rng(3).standard_normal(op.f_shape)
    for apply, reference in zip(dense[:2], reusable[:2], strict=True):
        out = np.asarray(apply(jnp.asarray(f.reshape(-1)))).reshape(op.f_shape)[padded]
        ref = np.asarray(reference(jnp.asarray(f.reshape(-1)))).reshape(op.f_shape)[padded]
        # r / (1 + floor) to the last bit XLA's division rounds; the input to the floor.
        np.testing.assert_allclose(out, (f / (1.0 + floor))[padded], rtol=5e-16, atol=0.0)
        assert np.all(np.abs(out - f[padded]) <= 1.01 * floor.max() * np.abs(f[padded]))
        # and what eliminating through those rows, as the reusable route does, gives.
        np.testing.assert_allclose(out, ref, rtol=5e-16, atol=0.0)


@pytest.mark.parametrize("name", (*TRUNCATING, "fokker_planck"))
def test_ragged_and_rectangular_chains_are_one_map(name: str, monkeypatch):
    """The rectangular layout is the fallback for a traced ``Nxi_for_x``; it must agree."""
    op = DECKS[name]()
    batch = op.n_species * op.n_x
    ragged = _inverses(op, monkeypatch, **UNSPLIT)
    rectangular = _inverses(
        op, monkeypatch, _coarse_row_layout=lambda o, *_: (tuple(range(batch)), (batch,) * o.n_xi), **UNSPLIT
    )
    rng = np.random.default_rng(5)
    f = jnp.asarray(rng.standard_normal(op.f_size))
    v = jnp.asarray(rng.standard_normal(op.total_size))
    for got, ref in zip(ragged[:2], rectangular[:2], strict=True):
        assert _relative(got(f), ref(f)) <= 1e-12
    for got, ref in zip(ragged[2:], rectangular[2:], strict=True):
        assert _relative(got(v), ref(v)) <= 1e-9


@pytest.mark.parametrize("name", TRUNCATING)
def test_reusable_factors_keep_exactly_the_active_rows(name: str, monkeypatch):
    op = DECKS[name]()
    captured = []
    real = cp._factor_coarse_active_chains

    def factor(*args, **kwargs):
        result = real(*args, **kwargs)
        captured.append(result)
        return result

    monkeypatch.setattr(cp, "_factor_coarse_active_chains", factor)
    _inverses(op, monkeypatch, **REUSABLE, **UNSPLIT)
    (factors, lengths), = captured
    m = op.n_theta * op.n_zeta
    active = sum(lengths)
    stored = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(factors))
    assert active == sum(cp._coarse_row_layout(op)[1])
    assert stored == active * (m * m * 8 + m * 4)
    assert stored == cp.coarse_preconditioner_factor_bytes(op)


#: ``solve(op, op.rhs(), method="gmres", tol=1e-10)`` GCROT iterations recorded on
#: perf/lazy-transposed-precond (2f29f4b), whose dense route assembled and
#: factored the rectangular bands.
ITERATIONS_BEFORE = {"fokker_planck": 90, "pas_2species_ramp": 3, "pas_ramp": 2}


@pytest.mark.parametrize("name", sorted(ITERATIONS_BEFORE))
def test_gcrot_iterations_are_unchanged(name: str):
    op = DECKS[name]()
    result = solve(op, op.rhs(), method="gmres", tol=1e-10)
    assert result.converged
    assert result.iterations == ITERATIONS_BEFORE[name]


@pytest.mark.parametrize("name", (*TRUNCATING, "fokker_planck"))
def test_only_the_active_rows_are_stored(name: str, monkeypatch):
    op = DECKS[name]()
    captured = []
    real = cp._factor_coarse_rows

    def factor(*args, **kwargs):
        captured.append(real(*args, **kwargs))
        return captured[-1]

    monkeypatch.setattr(cp, "_factor_coarse_rows", factor)
    cp.build_coarse_preconditioner(op)
    chains, _ = captured[-1]  # the LU rebuild follows an inverse whose blocks are too ill-conditioned
    m = op.n_theta * op.n_zeta
    h = cp._parity_basis(op.n_theta, op.n_zeta)[0].size  # every deck here is symmetric: two sectors of h
    active = sum(min(max(int(op.n_xi_for_x[b % op.n_x]), 1), op.n_xi) for b in range(op.n_species * op.n_x))
    stored = sum(leaf.nbytes for leaf in jax.tree_util.tree_leaves(chains))
    assert sum(cp._coarse_row_layout(op)[1]) == active
    assert sum(cp._coarse_row_layout(op, 2)[1]) == 2 * active
    assert stored == 2 * active * (3 * h * h * 8 + (h * 4 if len(chains[0]) == 4 else 0))  # inverse or LU+pivots, and both bands
    rectangular = op.n_species * op.n_x * op.n_xi
    assert (active < rectangular) == (name in TRUNCATING)
    assert stored <= 0.65 * cp.coarse_preconditioner_band_bytes(op) + rectangular * m * 4


def _parity_inputs(op: KineticOperator) -> tuple:
    coef = cp._truncated_coefficients(cp._strip_for_coarse(op))
    c0 = op._fs_average_factor().reshape(-1)
    subs = cp._coarse_generated_block_data(op, coef, op._mask(), None, c0, False, stacked=True)[0]
    return coef, subs


def _reflect(op: KineticOperator, f: jnp.ndarray) -> jnp.ndarray:
    """``S f``: ``(-1)^l f(-theta, -zeta)`` on the f-block."""
    t, z = np.arange(op.n_theta), np.arange(op.n_zeta)
    g = f.reshape(op.f_shape)[..., (-t) % op.n_theta, :][..., (-z) % op.n_zeta]
    return (g * (1.0 - 2.0 * (np.arange(op.n_xi) % 2))[:, None, None]).reshape(f.shape)


SYMMETRIC = ("fokker_planck", "pas_2species_ramp", "pas_ramp")
DECKS["with_er"] = lambda: _load("reduced_inputs/geometryScheme4_1species_PAS_withEr_DKESTrajectories")


@pytest.mark.parametrize("name", SYMMETRIC)
def test_a_symmetric_deck_splits_into_two_parity_sectors_with_the_same_inverse(name: str, monkeypatch):
    """Streaming and mirror are odd under the angular reflection R and couple l to l +- 1,
    so the coarse and full operators commute with S = (-1)^l R and both drives are odd under S.
    The split chain is then the unsplit inverse on two half-size sectors, on every route."""
    op = DECKS[name]()
    assert cp._coarse_parity(op, *_parity_inputs(op)) is not None
    rng = np.random.default_rng(0)
    f = jnp.asarray(rng.standard_normal(op.f_size))
    for apply in (op.apply_f, cp._coarse_operator(op).apply_f):
        assert _relative(_reflect(op, apply(_reflect(op, f).reshape(op.f_shape)).reshape(-1)), apply(f.reshape(op.f_shape)).reshape(-1)) <= 1e-13
    rhs = op.rhs()[: op.f_size]
    assert _relative(_reflect(op, rhs), -rhs) <= 1e-13
    for route in ({}, REUSABLE):
        split, full = _inverses(op, monkeypatch, **route), _inverses(op, monkeypatch, **route, **UNSPLIT)
        for got, ref in zip(split[:2], full[:2], strict=True):  # forward, transposed
            assert _relative(got(f), ref(f)) <= 1e-7  # chain condition numbers near 1e8 on the PAS decks


@pytest.mark.parametrize("name", ("magnetic_drift", "with_er"))
def test_drifts_and_er_keep_the_unsplit_chain(name: str, monkeypatch):
    """Tangential drifts and ExB are odd under R and diagonal in l: they anticommute with S."""
    op = DECKS[name]()
    assert cp._coarse_parity(op, *_parity_inputs(op)) is None
    basis = cp._parity_basis(op.n_theta, op.n_zeta)
    forced = _inverses(op, monkeypatch, _coarse_parity=lambda *_: basis)
    full = _inverses(op, monkeypatch)
    f = jnp.asarray(np.random.default_rng(0).standard_normal(op.f_size))
    assert _relative(forced[0](f), full[0](f)) > 1e-6  # rounding is 1e-10 on the symmetric decks
