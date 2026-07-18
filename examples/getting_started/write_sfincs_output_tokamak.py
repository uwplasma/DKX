"""Write a tiny analytic-tokamak ``sfincsOutput.h5`` with the Python API.

This example uses the supported ``geometryScheme=1`` analytic tokamak input
path and writes a v3-style output file without requiring the Fortran
executable.

Run:
  python examples/getting_started/write_sfincs_output_tokamak.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dkx.api import write_output
from dkx.io import read_sfincs_h5


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "tests" / "ref" / "output_scheme1_tokamak_1species_tiny.input.namelist",
        help="Analytic tokamak input.namelist to run.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "output" / "sfincsOutput_tokamak.h5",
        help="Destination sfincsOutput.h5 path.",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_output(args.input, args.out)
    data = read_sfincs_h5(args.out)

    print(f"Wrote: {args.out}")
    print("Geometry summary:")
    for key in ["Ntheta", "Nzeta", "Nx", "Nxi", "FSABHat2", "VPrimeHat"]:
        print(f"  {key} = {np.asarray(data[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
