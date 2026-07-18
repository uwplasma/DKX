"""Write a SFINCS-style ``sfincsOutput.h5`` with the dkx Python API.

What this example teaches:
  - how to parse a Fortran ``input.namelist`` and produce a v3-style
    ``sfincsOutput.h5`` with one call (``dkx.api.write_output``),
  - how to read the HDF5 output back into a plain dict (``dkx.io.read_sfincs_h5``)
    and inspect the grid sizes and a few normalized physics scalars.

Physics context: SFINCS solves the radially local, linearized drift-kinetic
equation on one flux surface; the ``sfincsOutput.h5`` file collects the grids,
the normalized geometry (e.g. the flux-surface average ``FSABHat2``), and the
neoclassical moments in the SFINCS v3 dataset layout [M. Landreman, H. M.
Smith, A. Mollen and P. Helander, Phys. Plasmas 21, 042503 (2014); SFINCS
technical documentation, https://github.com/landreman/sfincs].  This uses the
fast ``geometryScheme=4`` simplified-Boozer example so it runs in a second.

Run:
  python examples/getting_started/write_sfincs_output_python.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dkx.api import write_output
from dkx.io import read_sfincs_h5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ on sys.path
from _example_utils import output_dir, print_dataset_summary  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# A tiny geometryScheme=4 single-species deck that ships with the test suite.
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "output_scheme4_1species_tiny.input.namelist"

# Datasets to echo from the output file (grid sizes + a few normalized scalars).
SUMMARY_KEYS = ["Ntheta", "Nzeta", "Nx", "Delta", "Er", "FSABHat2"]

OUTPUT_DIR = output_dir(__file__)
OUT_PATH = OUTPUT_DIR / "sfincsOutput_python.h5"

# ----------------------------------------------------------------------------
# 1) Write the output file from the namelist
# ----------------------------------------------------------------------------
print("=== examples/getting_started/write_sfincs_output_python.py ===")
print(f"Step 1: writing {OUT_PATH.name} from {INPUT_NAMELIST.name}")
write_output(INPUT_NAMELIST, OUT_PATH)
print(f"  wrote: {OUT_PATH}")

# ----------------------------------------------------------------------------
# 2) Read it back and inspect a few datasets
# ----------------------------------------------------------------------------
print("Step 2: reading the HDF5 file back and inspecting datasets")
data = read_sfincs_h5(OUT_PATH)
print(f"  datasets in file: {len(data)}")

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print_dataset_summary(data, SUMMARY_KEYS)
print(f"  Wrote output file: {OUT_PATH.name}")
print("Done: examples/getting_started/write_sfincs_output_python.py")
