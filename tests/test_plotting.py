from __future__ import annotations

from pathlib import Path

from sfincs_jax.io import write_sfincs_jax_output_h5
from sfincs_jax.plotting import plot_sfincs_output_summary


def test_plot_sfincs_output_summary_writes_png(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_h5 = repo / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"
    output_png = tmp_path / "summary.png"
    out_path = plot_sfincs_output_summary(input_h5=input_h5, output_png=output_png)
    assert out_path == output_png.resolve()
    assert output_png.exists()


def test_plot_sfincs_output_summary_accepts_geometry_only_output(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[1]
    input_namelist = repo / "examples" / "getting_started" / "input.namelist"
    output_h5 = tmp_path / "geometry_only.h5"
    output_png = tmp_path / "geometry_only.png"
    write_sfincs_jax_output_h5(
        input_namelist=input_namelist,
        output_path=output_h5,
        compute_solution=False,
    )
    out_path = plot_sfincs_output_summary(input_h5=output_h5, output_png=output_png)
    assert out_path == output_png.resolve()
    assert output_png.exists()
