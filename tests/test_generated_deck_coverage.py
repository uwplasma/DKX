"""Every legal input combination must build, solve and write output.

Parity bugs in this code have not come from wrong formulas so much as from
*unvisited combinations*.  ``geometryScheme = 3`` was absent from the writer's
analytic-scheme set, so any scheme-3 deck that reached the output stage
demanded an ``equilibriumFile`` its analytic model does not have -- but the one
scheme-3 fixture in ``tests/ref`` is ``RHSMode=3``, which takes a different
path, so nothing noticed.  Likewise 57 of the 59 fixtures pin
``Nxi_for_x_option = 0``, and the default ramp was wrong for years.

Hand-written fixtures cannot close that gap: they are a sample, and the bugs
live where the sample is not.  This module instead drives
:mod:`tools.benchmarks.generate_deck_matrix` -- the same generator the parity
sweeps use -- and runs a *covering set*: the smallest selection of generated
decks in which every value of every axis appears at least once.

The covering set is computed from the generator's own axes, so adding an axis
there (a new geometry scheme, a new collision operator) automatically extends
what is exercised here.  No Fortran binary is required, which is what lets this
run in CI: it cannot check parity, but it can check that dkx *accepts and
completes* every combination SFINCS accepts, which is the failure mode that has
actually bitten.
"""

from __future__ import annotations

import sys
from dataclasses import fields
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tools" / "benchmarks"))

from generate_deck_matrix import Case, enumerate_cases, render  # noqa: E402

#: Axes that select genuinely different code paths.  ``resolution`` is excluded:
#: it changes cost, not the branch taken.
_COVERED_AXES = tuple(f.name for f in fields(Case) if f.name != "resolution")


def _covering_set(cases: list[Case]) -> list[Case]:
    """Fewest cases in which every (axis, value) pair appears at least once.

    A greedy set cover.  Exhaustively running the matrix would be far too slow
    for CI, and sampling it at random would make failures irreproducible; a
    covering set keeps every axis value exercised at a fixed, small cost.
    """
    required = {
        (axis, getattr(case, axis)) for case in cases for axis in _COVERED_AXES
    }
    chosen: list[Case] = []
    remaining = set(required)
    # Deterministic order so a failure is always reproducible.
    ordered = sorted(cases, key=lambda case: case.name)
    while remaining:
        best = max(
            ordered,
            key=lambda case: (
                len({(a, getattr(case, a)) for a in _COVERED_AXES} & remaining),
                case.name,
            ),
        )
        gained = {(a, getattr(best, a)) for a in _COVERED_AXES} & remaining
        if not gained:  # pragma: no cover - only if an axis value is unreachable
            break
        remaining -= gained
        chosen.append(best)
    return chosen


_CASES = _covering_set(
    enumerate_cases(
        geometries=(1, 2, 3, 4), collisions=(0, 1), resolutions=("tiny",)
    )[0]
)


def test_the_covering_set_reaches_every_axis_value() -> None:
    """The selection is a cover, not a sample -- otherwise the sweep proves little."""
    all_cases = enumerate_cases(
        geometries=(1, 2, 3, 4), collisions=(0, 1), resolutions=("tiny",)
    )[0]
    expected = {(a, getattr(c, a)) for c in all_cases for a in _COVERED_AXES}
    covered = {(a, getattr(c, a)) for c in _CASES for a in _COVERED_AXES}
    assert covered == expected
    # A cover of eight axes should not need anything like the full matrix.
    assert len(_CASES) < 0.2 * len(all_cases)


@pytest.mark.parametrize("case", _CASES, ids=lambda c: c.name)
def test_generated_deck_runs_end_to_end(case: Case, tmp_path: Path) -> None:
    """Build, solve and write output for one legal combination.

    Writing matters as much as solving: the scheme-3 bug was in the *writer*,
    so a test that stopped at the solve would have passed through it.
    """
    from dkx.inputs import load_sfincs_input
    from dkx.run import run_profile, run_transport_matrix

    deck = tmp_path / "input.namelist"
    deck.write_text(render(case))

    general = load_sfincs_input(deck).raw.group("general")
    rhs_mode = int(next((v for k, v in general.items() if k.lower() == "rhsmode"), 1))
    driver = run_profile if rhs_mode == 1 else run_transport_matrix

    run = driver(deck, out_path=tmp_path / "out.h5", emit=None)
    assert (tmp_path / "out.h5").exists()
    assert run.solve_result.converged


# ---------------------------------------------------------------------------
# Phi1 switch defaults
# ---------------------------------------------------------------------------


def test_phi1_switch_defaults_match_fortran_and_each_other() -> None:
    """``includePhi1InKineticEquation`` defaults TRUE, in both places dkx stores it.

    ``version3/globalVariables.F90`` sets ``includePhi1InKineticEquation =
    .true.`` (line 152) and ``includePhi1InCollisionOperator = .false.``
    (line 150).  dkx held both values in two places and they disagreed:
    :class:`dkx.inputs.SfincsInput` had the kinetic switch right, while the
    operator builder defaulted it to ``False``.  Any deck that enabled ``Phi1``
    without naming the switch therefore dropped the Phi1-in-kinetic coupling,
    an ``O(Phi1^2)`` error -- measured at 23% on the electron flow of a
    two-species deck, and invisible to every fixture because they all name the
    switch explicitly.

    Asserting the two dkx defaults against each other *and* against the Fortran
    values is what makes this a parity check rather than a restatement.
    """
    from dkx.drift_kinetic import KineticOperator
    from dkx.namelist import parse_sfincs_input_text

    # A deck that enables Phi1 and says nothing about the coupling switches.
    text = render(
        Case(
            geometry=1, collision=0, er=0.0, phi1=True,
            rhs_mode=1, nxi_ramp=0, species=2, resolution="tiny",
        )
    )
    assert "includePhi1InKineticEquation" not in text  # the deck stays silent
    operator = KineticOperator.from_namelist(parse_sfincs_input_text(text))
    assert operator.include_phi1_in_kinetic is True, (
        "includePhi1InKineticEquation must default TRUE (globalVariables.F90:152)"
    )
    # The collision coupling is carried by the ``fp_phi1`` operator, which stays
    # unbuilt while the switch is off.
    assert operator.fp_phi1 is None, (
        "includePhi1InCollisionOperator must default FALSE (globalVariables.F90:150)"
    )
