"""The .pdf suffix pages only under style="summary"; the docs must say so.

An earlier docstring promised a "multi-page diagnostics book" from the .pdf
suffix without qualifying the style, and the default panels style writes one
page.  A user following that would get a single page and no error.
"""

import inspect
from pathlib import Path

from pypdf import PdfReader

from dkx.plotting import plot

REFERENCE = (
    Path(__file__).resolve().parents[1]
    / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
)


def test_panels_pdf_is_one_page_and_summary_pdf_is_not(tmp_path: Path) -> None:
    panels = PdfReader(str(plot(REFERENCE, out=tmp_path / "p.pdf")))
    summary = PdfReader(str(plot(REFERENCE, out=tmp_path / "s.pdf", style="summary")))
    assert panels.get_num_pages() == 1
    assert summary.get_num_pages() > 1


def test_the_docstring_does_not_promise_paging_without_naming_the_style() -> None:
    text = inspect.getdoc(plot).lower()
    if "page" not in text:
        return
    claim = text[: text.index("page")]
    assert "summary" in claim[-240:], (
        "the docstring must name style='summary' before promising multi-page "
        "output; the default panels style writes one page whatever the suffix"
    )
