"""``tol=None`` means the deck's ``solverTolerance`` in every deck-driven entry point."""

from __future__ import annotations

from pathlib import Path

import pytest

import dkx.run as run_module

REF = Path(__file__).parent / "ref"


class _Stop(Exception):
    pass


@pytest.mark.parametrize(
    ("driver", "deck", "tol", "expected"),
    [
        ("run_profile", "er_xdot_1species_tiny.input.namelist", None, 1e-8),
        ("run_profile", "er_xdot_1species_tiny.input.namelist", 1e-5, 1e-5),
        ("run_transport_matrix", "output_scheme2_lhd_1species_tiny.input.namelist", None, 1e-8),
        ("run_from_namelist", "er_xdot_1species_tiny.input.namelist", None, 1e-8),
    ],
)
def test_none_tolerance_reads_the_deck(monkeypatch, driver, deck, tol, expected) -> None:
    seen = {}
    extra = {"out_path": "unused.h5"} if driver == "run_from_namelist" else {}

    def options(**kwargs):
        seen.update(kwargs)
        raise _Stop

    monkeypatch.setattr(run_module, "SolverOptions", options)
    with pytest.raises(_Stop):
        getattr(run_module, driver)(REF / deck, tol=tol, **extra)
    assert seen["tol"] == expected
