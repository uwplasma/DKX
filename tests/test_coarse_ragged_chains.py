"""The dense coarse route factors generated rows.


The dense route and the reusable Schur-LU route factor the *same* pinned rows
from the *same* generator, so their f-block inverses agree to rounding, not to
the ``1e-8`` the equivalence tests in ``tests/test_tier2_memory_guard.py`` allow
between genuinely different eliminations.
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
REUSABLE = {"_coarse_bands_fit": lambda _op: False, "_coarse_factors_fit": lambda _op: True}
#: Relative agreement of the f-block inverses.  ``magnetic_drift`` is the exception
#: and the reason is measured, not assumed: its chains have condition numbers of
#: 2.2e8, every route leaves the same 3e-3 residual against the pinned operator,
#: and the substitutions of the two routes (unrolled against a scan) round
#: differently by 8e-9.
F_BLOCK_TOLERANCE = {"magnetic_drift": 1e-7}


def _inverses(op: KineticOperator, monkeypatch, **patches):
    """``(a_inv, a_inv_t, precond, precond_t)`` of the route ``patches`` select.

    The f-block inverses are what this change touches; the border's Schur
    complement wraps them and amplifies rounding on the near-singular decks, so
    the bordered maps are compared too, but more loosely.
    """
    captured: dict[str, object] = {}
    real_schur, real_lazy = cp.schur_projected_precond, cp._lazy_projected_precond

    def schur(a_inv, *args, **kwargs):
        captured.setdefault("a_inv", a_inv)
        return real_schur(a_inv, *args, **kwargs)

    def lazy(op_, a_inv_t, *args, **kwargs):
        captured["a_inv_t"] = a_inv_t
        return real_lazy(op_, a_inv_t, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cp, "schur_projected_precond", schur)
        patch.setattr(cp, "_lazy_projected_precond", lazy)
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



@pytest.mark.parametrize("name", sorted(DECKS))
def test_the_dense_route_is_the_reusable_route_to_rounding(name: str, monkeypatch):
    op = DECKS[name]()
    dense = _inverses(op, monkeypatch)
    reusable = _inverses(op, monkeypatch, **REUSABLE)
    tolerance = F_BLOCK_TOLERANCE.get(name, 1e-12)
    rng = np.random.default_rng(0)
    for _ in range(2):
        f = jnp.asarray(rng.standard_normal(op.f_size))
        v = jnp.asarray(rng.standard_normal(op.total_size))
        for got, ref in zip(dense[:2], reusable[:2], strict=True):  # forward, transposed
            assert _relative(got(f), ref(f)) <= tolerance
        for got, ref in zip(dense[2:], reusable[2:], strict=True):
            out = np.asarray(got(v))
            assert np.all(np.isfinite(out))
            assert _relative(out, ref(v)) <= 1e-7


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
