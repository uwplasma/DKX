"""The coarse tier-2 preconditioner refuses to allocate what cannot fit.

Its dense ``(Ntheta*Nzeta)`` bands are sized exactly by the grid, so a solve
that cannot fit them is knowable before any work happens.  Without the check
the process is killed by the OS part way through and produces nothing: no
traceback, no partial output, and a return code indistinguishable from any
other crash.  On the 2026-08-01 upstream campaign that cost five decks 55-90 s
each before dying.

Threshold and its evidence live in
:func:`dkx.solve._check_coarse_preconditioner_fits`; these tests pin the
behaviour, not the constant.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.namelist import read_sfincs_input
from dkx.solve import (
    _check_coarse_preconditioner_fits,
    _host_memory_bytes,
    build_tier2_preconditioner,
    coarse_preconditioner_band_bytes,
)

REF = Path(__file__).parent / "ref"


def _small_op() -> KineticOperator:
    return KineticOperator.from_namelist(
        read_sfincs_input(REF / "er_xdot_1species_tiny.input.namelist")
    )


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


def test_a_deck_that_fits_is_allowed(monkeypatch):
    """The guard must be silent wherever the classical route works."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    _check_coarse_preconditioner_fits(_small_op())  # must not raise


def test_the_refusal_states_the_size_and_the_options(monkeypatch):
    """The message has to leave the user somewhere to go.

    It deliberately does *not* recommend the sparse route.  That route stores
    far less, but measured on the five decks that trigger this guard it was
    killed on three and timed out on two, so recommending it would trade a
    fast failure for a slow one.  A guard that sends users down a road that
    also fails is worse than the kill it replaces.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    op = _small_op()
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: 1.0)
    with pytest.raises(MemoryError) as excinfo:
        _check_coarse_preconditioner_fits(op)
    message = str(excinfo.value)
    assert f"{op.n_theta}x{op.n_zeta}" in message  # the size that did not fit
    assert "reduce Ntheta/Nzeta or Nxi" in message  # what actually works
    assert "DKX_TIER2_MEMORY_GUARD=off" in message


def test_the_refusal_does_not_recommend_a_route_that_also_fails(monkeypatch):
    """Regression guard on the correction itself.

    The first version of this message told users to switch to
    ``preconditioner='sparse'``; the experiment that followed found it rescues
    none of the five decks.  Re-adding that recommendation without new
    measurement should fail here.
    """
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: 1.0)
    with pytest.raises(MemoryError) as excinfo:
        _check_coarse_preconditioner_fits(_small_op())
    assert "'sparse' stores far less but was measured killed" in str(excinfo.value)


def test_the_guard_gates_the_coarse_build(monkeypatch):
    """It fires through the route callers actually take."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: 1.0)
    with pytest.raises(MemoryError):
        build_tier2_preconditioner(_small_op(), "coarse")


def test_the_sparse_route_is_not_gated(monkeypatch):
    """Gating the alternative the message recommends would be a trap."""
    pytest.importorskip("scipy.sparse.linalg")
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: 1.0)
    precond, _ = build_tier2_preconditioner(_small_op(), "sparse")
    assert callable(precond)


def test_the_guard_can_be_switched_off(monkeypatch):
    """Someone who knows their machine better than ``sysconf`` keeps control."""
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: 1.0)
    monkeypatch.setenv("DKX_TIER2_MEMORY_GUARD", "off")
    _check_coarse_preconditioner_fits(_small_op())  # must not raise


def test_unknown_host_memory_does_not_block_a_solve(monkeypatch):
    """A platform that cannot report its RAM must not lose the solver."""
    monkeypatch.delenv("DKX_TIER2_MEMORY_GUARD", raising=False)
    monkeypatch.setattr("dkx.solve._host_memory_bytes", lambda: None)
    _check_coarse_preconditioner_fits(_small_op())  # must not raise


def test_host_memory_is_readable_here():
    """Guards the assumption the check rests on, on the platforms we run."""
    total = _host_memory_bytes()
    assert total is None or total > 2**30
