"""``preconditioner_xi`` drops L±2; ``drop_l_coupling`` drops L±1. Not the same knob.

Fortran's ``preconditioner_xi=1`` (the default, ``globalVariables.F90:209``) drops
the *off-by-2* diagonal terms in L -- every call site in ``populateMatrix.F90``
(lines 625, 772, 872, 933, 1046) carries the comment "Drop the off-by-2 diagonal
terms in L if this is the preconditioner".  DKX drops those unconditionally, so
the namelist key is accepted for compatibility and changes nothing.

DKX's ``drop_l_coupling`` drops the L±1 streaming and mirror coupling instead.
That is strictly more aggressive and removes the dominant term: measured, 6000
iterations to a residual of 0.77 where keeping the coupling converges in 19.
Documentation that equated the two would send a user chasing Fortran parity
straight into a solve that never converges, so the equation is pinned out here
rather than left to review.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest

from dkx.drift_kinetic import KineticOperator
from dkx.inputs import _validate, read_sfincs_input, sfincs_input_from_raw
from dkx.multigrid import simplified_operator

ROOT = Path(__file__).resolve().parents[1]
REF = Path(__file__).parent / "ref"

#: Two shapes the mislabel took.  The first is co-occurrence: the two names
#: within a couple of lines, as if interchangeable.  The second is the claim
#: itself -- ``preconditioner_xi`` said to drop the L±1 coupling -- which is the
#: one that survived several passes of the first, because it never mentions
#: ``drop_l_coupling`` at all.
_CONFLATION = re.compile(
    r"drop_l_coupling[^\n]{0,120}(\n[^\n]{0,120}){0,2}preconditioner_xi\s*=?\s*1?"
    r"|preconditioner_xi[^\n]{0,80}(\n[^\n]{0,80}){0,1}\bis\b[^\n]{0,40}drop_l_coupling"
    r"|preconditioner_xi[^\n]{0,60}(\n[^\n]{0,60}){0,2}drops? the L(±|\s*\+-\s*)1"
)


def _sources() -> list[Path]:
    return [
        path
        for folder in ("dkx", "docs")
        for path in (ROOT / folder).rglob("*")
        if path.suffix in {".py", ".rst"} and path.is_file()
    ]


#: Phrases that mark a passage naming both knobs in order to *distinguish* them.
#: Without these the corrected text would trip its own guard.
_DISAMBIGUATION = ("not Fortran", "not what", "which drops", "separate", "differ")


def test_no_dkx_text_calls_drop_l_coupling_the_preconditioner_xi_knob():
    offenders = []
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for match in _CONFLATION.finditer(text):
            line = text[: match.start()].count("\n") + 1
            # Read past the match: the disambiguation may follow the two names.
            window = text[match.start() : match.end() + 200]
            if any(phrase in window for phrase in _DISAMBIGUATION):
                continue
            offenders.append(f"{path.relative_to(ROOT)}:{line}")
    assert not offenders, (
        "these places equate drop_l_coupling with Fortran's preconditioner_xi; "
        f"preconditioner_xi drops L+-2, drop_l_coupling drops L+-1: {offenders}"
    )


def test_the_docs_say_preconditioner_xi_drops_the_off_by_two_terms():
    text = (ROOT / "docs" / "inputs.rst").read_text()
    assert "L±2" in text and "populateMatrix.F90" in text
    assert "preconditioner_xi=1`` drops the **L±2** terms" in text


def _op() -> KineticOperator:
    return KineticOperator.from_namelist(
        read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    )


def test_drop_l_coupling_severs_the_l_plus_minus_one_bands():
    """The behavioural half: the flag zeroes the streaming/mirror coupling."""
    op = _op()
    kept = simplified_operator(op, drop_l_coupling=False)
    dropped = simplified_operator(op, drop_l_coupling=True)

    assert jnp.any(kept.xi_coupling_lower != 0.0)
    assert jnp.any(kept.xi_coupling_upper != 0.0)
    assert jnp.all(dropped.xi_coupling_lower == 0.0)
    assert jnp.all(dropped.xi_coupling_upper == 0.0)


def test_preconditioner_xi_is_parsed_validated_and_inert():
    """The other half of the docs claim: the key is accepted and does nothing.

    DKX drops the L±2 terms unconditionally, so there is no setting of
    ``preconditioner_xi`` for the solver to honour.  It is still parsed,
    range-checked and written back out, which is what "accepted for namelist
    compatibility" has to mean if it is to mean anything --- and it must not
    reach :class:`KineticOperator`, or the claim would be false.
    """
    raw = read_sfincs_input(REF / "quick_2species_FPCollisions_noEr.input.namelist")
    deck = sfincs_input_from_raw(raw)
    assert deck.preconditioner.preconditioner_xi == 1  # globalVariables.F90:209

    for bad in (-1, 2):
        broken = replace(
            deck,
            preconditioner=replace(deck.preconditioner, preconditioner_xi=bad),
        )
        with pytest.raises(ValueError, match="preconditioner_xi must be 0 or 1"):
            _validate(broken)

    # Inert: it is not a field of the operator the solver actually builds.
    assert not hasattr(KineticOperator.from_namelist(raw), "preconditioner_xi")
