"""Write small tutorial outputs and a diagnostics panel.

Run from the repository root:

    python examples/tutorials/run_quick_output_and_plot.py --out-dir tutorial_output

The script is intentionally fast: it writes the standard output fields without
computing a full transport solve, then reads each format back and builds the same
summary panel used by ``sfincs_jax --plot``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sfincs_jax.api import write_output
from sfincs_jax.io import read_sfincs_output_file
from sfincs_jax.plotting import plot_sfincs_output_summary


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_NAMELIST = REPO_ROOT / "examples" / "getting_started" / "input.namelist"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path.cwd(),
        help="Directory where tutorial outputs should be written.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for suffix in (".h5", ".nc", ".npz"):
        path = out_dir / f"sfincsOutput_tutorial{suffix}"
        write_output(INPUT_NAMELIST, path)
        data = read_sfincs_output_file(path)
        print(f"{path.name}: {len(data)} output fields")
        written.append(path)

    pdf_path = out_dir / "sfincsOutput_tutorial_summary.pdf"
    plot_sfincs_output_summary(input_h5=written[0], output_png=pdf_path)
    print(f"summary panel: {pdf_path}")


if __name__ == "__main__":
    main()
