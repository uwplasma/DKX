"""Write a tiny VMEC ``sfincsOutput.h5`` with an explicit ``wout_path`` override.

This example demonstrates the supported ``geometryScheme=5`` workflow and the
``wout_path`` compatibility alias used by the CLI and Python API.

Run:
  python examples/getting_started/write_sfincs_output_vmec.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sfincs_jax.io import read_sfincs_h5, write_sfincs_jax_output_h5  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=_REPO_ROOT / "tests" / "ref" / "output_scheme5_1species_tiny.input.namelist",
        help="VMEC input.namelist to run.",
    )
    parser.add_argument(
        "--wout-path",
        type=Path,
        default=Path("wout_w7x_standardConfig.nc"),
        help="Explicit VMEC equilibrium override. Known public fixtures are fetched on demand.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).with_suffix("").parent / "output" / "sfincsOutput_vmec.h5",
        help="Destination sfincsOutput.h5 path.",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_sfincs_jax_output_h5(
        input_namelist=args.input,
        output_path=args.out,
        wout_path=args.wout_path,
    )
    data = read_sfincs_h5(args.out)

    print(f"Wrote: {args.out}")
    print(f"Using wout_path: {args.wout_path}")
    print("Geometry summary:")
    for key in ["Ntheta", "Nzeta", "Nx", "Nxi", "FSABHat2", "B0OverBBar"]:
        print(f"  {key} = {np.asarray(data[key])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
