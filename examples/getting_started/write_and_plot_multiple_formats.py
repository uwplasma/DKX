"""Write HDF5/NetCDF/NPZ outputs and create a PDF diagnostics panel.

Run from the repository root:

  python examples/getting_started/write_and_plot_multiple_formats.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sfincs_jax.io import read_sfincs_output_file, write_sfincs_jax_output_h5
from sfincs_jax.plotting import plot_sfincs_output_summary


_REPO_ROOT = Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "output",
        help="Directory for .h5/.nc/.npz outputs and the PDF panel.",
    )
    args = parser.parse_args(argv)
    input_path = _REPO_ROOT / "examples" / "getting_started" / "input.namelist"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for suffix in (".h5", ".nc", ".npz"):
        out_path = out_dir / f"sfincsOutput_getting_started{suffix}"
        write_sfincs_jax_output_h5(
            input_namelist=input_path,
            output_path=out_path,
            compute_solution=False,
            verbose=False,
        )
        data = read_sfincs_output_file(out_path)
        print(f"wrote {out_path} with {len(data)} datasets")

    pdf_path = out_dir / "sfincsOutput_getting_started_summary.pdf"
    plot_sfincs_output_summary(
        input_h5=out_dir / "sfincsOutput_getting_started.h5",
        output_png=pdf_path,
    )
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
