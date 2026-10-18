from __future__ import annotations

import inspect
from pathlib import Path

from pypdf import PdfReader

from dkx.api import write_output
from dkx.plotting import plot, plot_sfincs_output_summary


def test_plot_sfincs_output_summary_writes_png(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    output_png = tmp_path / "summary.png"
    out_path = plot_sfincs_output_summary(input_h5=input_h5, output_png=output_png)
    assert out_path == output_png.resolve()
    assert output_png.exists()


def test_plot_sfincs_output_summary_writes_pdf_panel(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    output_pdf = tmp_path / "summary.pdf"
    out_path = plot_sfincs_output_summary(input_h5=input_h5, output_png=output_pdf)
    assert out_path == output_pdf.resolve()
    assert output_pdf.exists()
    assert output_pdf.stat().st_size > 1000


def test_plot_sfincs_output_summary_accepts_geometry_only_output(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_namelist = repo / "tests" / "ref" / "minimal_output_geometryScheme4.input.namelist"
    output_h5 = tmp_path / "geometry_only.h5"
    output_png = tmp_path / "geometry_only.png"
    write_output(input_namelist, output_h5)
    out_path = plot_sfincs_output_summary(input_h5=output_h5, output_png=output_png)
    assert out_path == output_png.resolve()
    assert output_png.exists()


# The .pdf suffix pages only under style="summary"; the docs must say so.


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
