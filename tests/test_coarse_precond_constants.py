"""The reusable-factor coarse route must capture no arrays as XLA constants.

The route's whole claim is storage: it keeps one ``(Nxi, TZ, TZ)`` array per
subsystem instead of three, regenerating the two off-diagonal bands inside each
substitution sweep.  That claim is only true if the regeneration reads *traced*
arrays.  A concrete leaf reached from inside a lowering becomes a compile-time
constant of it, XLA holds a second copy, and the bands come back --- which is
exactly how ``filteredW7XNetCDF_2species_magneticDrifts_noEr`` was OOM-killed
at 640 s with JAX reporting ``15.52GB total`` of captured constants.

Two arrays make that happen if left closed over: the Schur LU factors, and the
``(TZ, TZ)`` ``stream``/``exb`` blocks the row generator rebuilds from.  The
second is the easy one to reintroduce by accident, because
``GeneratedBlockTridiagFactors`` carries its generator as a *static* field, so
a generator built outside the jit silently drags its closure in.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import pytest

import dkx.coarse_precond as cp
from dkx.coarse_precond import build_coarse_preconditioner
from dkx.drift_kinetic import KineticOperator
from dkx.inputs import read_sfincs_input

REF = Path(__file__).parent / "ref"

#: Two ``(TZ, TZ)`` blocks of the fixture below (9.6 KB each), so a captured
#: generator (ten of them, one per subsystem) trips it and an incidental index
#: array does not.
_CAPTURE_THRESHOLD_BYTES = 20_000


def _load_op(name: str) -> KineticOperator:
    return KineticOperator.from_namelist(read_sfincs_input(REF / f"{name}.input.namelist"))


def _captured_constant_warnings(apply, vector) -> list[str]:
    """Warnings JAX raises about constants captured while lowering ``apply``."""
    previous = jax.config.jax_captured_constants_warn_bytes
    jax.config.update("jax_captured_constants_warn_bytes", _CAPTURE_THRESHOLD_BYTES)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            jax.block_until_ready(apply(vector))
        return [str(w.message) for w in caught if "constants were captured" in str(w.message)]
    finally:
        jax.config.update("jax_captured_constants_warn_bytes", previous)


@pytest.fixture
def reusable_route(monkeypatch):
    """Force the route the large decks take, on a deck whose bands would fit."""
    monkeypatch.setattr(cp, "_coarse_bands_fit", lambda op: False)
    monkeypatch.setattr(cp, "_coarse_factors_fit", lambda op: True)


@pytest.mark.usefixtures("reusable_route")
def test_the_reusable_route_captures_no_constants():
    op = _load_op("quick_2species_FPCollisions_noEr")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # the route's own notice
        precond, precond_t = build_coarse_preconditioner(op)
    vector = jnp.zeros((op.total_size,), dtype=jnp.float64)

    assert _captured_constant_warnings(precond, vector) == []
    assert _captured_constant_warnings(precond_t, vector) == []


@pytest.mark.usefixtures("reusable_route")
def test_rebuilding_the_generator_outside_the_jit_is_what_captures_them():
    """The negative control: without this fix the same route does capture.

    Without it the test above could pass for the wrong reason --- a threshold
    nothing reaches, or a route that never regenerates a block at all.
    """
    op = _load_op("quick_2species_FPCollisions_noEr")
    coef = cp._truncated_coefficients(cp._strip_for_coarse(op))
    mask = op._mask()
    gen_data = cp._coarse_generated_block_data(
        op, coef, mask, None, op._fs_average_factor().reshape(-1), False
    )
    rows = cp._coarse_pinned_block_fns(coef, op.n_xi, *gen_data, False)

    # Concrete leaves, closed over rather than passed: one (TZ, TZ) block per
    # subsystem is enough to trip the threshold.
    @jax.jit
    def regenerate(index):
        return jnp.stack([row(index)[0] for row in rows])

    captured = _captured_constant_warnings(regenerate, jnp.asarray(1, dtype=jnp.int32))
    assert captured, "closing over the generator no longer captures constants"


@pytest.mark.usefixtures("reusable_route")
def test_the_rebuilt_generator_produces_identical_blocks():
    """Passing the arrays across the jit boundary must not change the operator."""
    op = _load_op("quick_2species_FPCollisions_noEr")
    coef = cp._truncated_coefficients(cp._strip_for_coarse(op))
    mask = op._mask()
    gen_data = cp._coarse_generated_block_data(
        op, coef, mask, None, op._fs_average_factor().reshape(-1), False
    )
    outside = cp._coarse_pinned_block_fns(coef, op.n_xi, *gen_data, False)

    @jax.jit
    def inside(coef_t, subs_t, floor_t, gamma_t, index):
        rows = cp._coarse_pinned_block_fns(coef_t, op.n_xi, subs_t, floor_t, gamma_t, False)
        return [jnp.stack(row(index)) for row in rows]

    index = jnp.asarray(1, dtype=jnp.int32)
    for expected, actual in zip(
        [jnp.stack(row(index)) for row in outside],
        inside(coef, *gen_data, index),
        strict=True,
    ):
        assert jnp.array_equal(expected, actual)
