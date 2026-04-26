"""Plot diagnostics from ``sfincsOutput.h5``/``.nc``/``.npz``."""

from __future__ import annotations

import argparse
from pathlib import Path

from sfincs_jax.plotting import plot_sfincs_output_summary


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-h5",
        type=Path,
        default=repo_root / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5",
        help="Path to a sfincs_jax output file to visualize.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "output" / "sfincsOutput_summary.pdf",
        help="Output PDF/figure path.",
    )
    args = parser.parse_args(argv)
    out_path = plot_sfincs_output_summary(input_h5=args.input_h5, output_png=args.out)
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
