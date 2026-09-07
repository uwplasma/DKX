"""The README budget, enforced.

The front page regrew from 158 lines to 250 three times between 2026-08-30 and
2026-09-05, each time by adding prose that belonged in the documentation. The
plan's working-method rule 12 sets a budget; this test is what makes it real.

The numbers sit just above what the page currently needs, because other
contracts in this repository force its shape: ``test_readme_quickstart_runs``
requires a self-contained quickstart that prints a flux and a solver route, and
``test_benchmark_doc_claims`` pins fourteen measured tokens, the capability
families and the figures. A budget the page cannot meet is not a budget.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"

MAX_LINES = 160
MAX_WORDS = 1_000
MAX_PYTHON_BLOCKS = 2
MAX_BLOCK_LINES = 16

#: Sentences that hedge a claim belong in the documentation quadrant Diataxis
#: assigns them, next to the evidence, not on the front page.
HEDGES = ("has not", "cannot", "does not yet", "is not converged")


def _text() -> str:
    return README.read_text(encoding="utf-8")


def _blocks(language: str) -> list[str]:
    return re.findall(rf"^```{language}\n(.*?)^```", _text(), re.S | re.M)


def test_the_readme_stays_within_its_line_and_word_budget() -> None:
    text = _text()
    lines, words = len(text.splitlines()), len(text.split())
    assert lines <= MAX_LINES, f"{lines} lines exceeds the {MAX_LINES}-line budget"
    assert words <= MAX_WORDS, f"{words} words exceeds the {MAX_WORDS}-word budget"


def test_the_readme_carries_at_most_two_short_python_blocks() -> None:
    blocks = _blocks("python")
    assert blocks, "the README must show how to run something"
    assert len(blocks) <= MAX_PYTHON_BLOCKS, (
        f"{len(blocks)} python blocks; a third belongs in docs/ or examples/"
    )
    for index, block in enumerate(blocks):
        length = len(block.rstrip("\n").splitlines())
        assert length <= MAX_BLOCK_LINES, (
            f"python block {index + 1} is {length} lines, over {MAX_BLOCK_LINES}"
        )


def test_the_first_block_runs_a_case_and_the_last_takes_a_gradient() -> None:
    """The two blocks earn their place by showing the two things DKX is for.

    The quickstart proves it computes transport; the second proves the outputs
    are differentiable, which is the claim competitors state without evidence.
    """
    blocks = _blocks("python")
    assert "dkx.run(" in blocks[0], "the first python block must run a case"
    gradient = ("jax.grad", "value_and_grad", "jacfwd", "jacrev")
    assert any(token in blocks[-1] for token in gradient), (
        "the last python block must end in a derivative"
    )


def test_the_readme_tells_a_reader_how_to_cite_it() -> None:
    assert _blocks("bibtex"), "a research code's front page needs a citation block"
    assert "CITATION.cff" in _text()


@pytest.mark.parametrize("hedge", HEDGES)
def test_the_readme_does_not_hedge_on_the_front_page(hedge: str) -> None:
    """Scope belongs beside the evidence it qualifies, not on the front page.

    This is not a licence to overclaim: the measured-results section states its
    scope in positive terms and links the validation matrix for the rest.
    """
    found = [
        line for line in _text().splitlines() if hedge in line.lower()
    ]
    assert not found, f"{hedge!r} on the front page: {found[:1]}"


def test_the_readme_shows_its_three_figures() -> None:
    """Hero, parity and cross-code. Each is pinned by the provenance contract."""
    text = _text()
    for figure in ("w7x_showcase.png", "canonical_parity.png", "cross_code_validation.png"):
        assert figure in text, f"{figure} is committed but not shown"
