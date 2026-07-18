"""Write a tiny analytic-tokamak ``sfincsOutput.h5`` with the Python API.

What this example teaches:
  - how the supported ``geometryScheme=1`` analytic-tokamak input path builds
    its magnetic geometry directly from the namelist (no equilibrium file),
  - how ``dkx.api.write_output`` produces a v3-style output file without the
    Fortran executable,
  - how to read the file back and inspect grid sizes plus the geometry scalars
    ``FSABHat2`` and ``VPrimeHat``.

Physics context: ``geometryScheme=1`` is the SFINCS three-helicity analytic
model, here reduced to a concentric circular-cross-section tokamak; the
flux-surface average ``FSABHat2`` and the flux-surface volume element
``VPrimeHat`` are the normalized geometry factors that enter the neoclassical
transport coefficients [M. Landreman, H. M. Smith, A. Mollen and P. Helander,
Phys. Plasmas 21, 042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].

Run:
  python examples/getting_started/write_sfincs_output_tokamak.py
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

# Analytic tokamak (geometryScheme=1) single-species deck from the test suite.
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "output_scheme1_tokamak_1species_tiny.input.namelist"

# Datasets to echo (grid sizes + two normalized geometry scalars).
SUMMARY_KEYS = ["Ntheta", "Nzeta", "Nx", "Nxi", "FSABHat2", "VPrimeHat"]

OUTPUT_DIR = output_dir(__file__)
OUT_PATH = OUTPUT_DIR / "sfincsOutput_tokamak.h5"

# ----------------------------------------------------------------------------
# 1) Write the output file from the analytic-tokamak namelist
# ----------------------------------------------------------------------------
print("=== examples/getting_started/write_sfincs_output_tokamak.py ===")
print(f"Step 1: writing {OUT_PATH.name} from {INPUT_NAMELIST.name}")
write_output(INPUT_NAMELIST, OUT_PATH)
print(f"  wrote: {OUT_PATH}")

# ----------------------------------------------------------------------------
# 2) Read it back and inspect the geometry
# ----------------------------------------------------------------------------
print("Step 2: reading the HDF5 file back")
data = read_sfincs_h5(OUT_PATH)
print(f"  datasets in file: {len(data)}")

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print_dataset_summary(data, SUMMARY_KEYS)
print(f"  Wrote output file: {OUT_PATH.name}")
print("Done: examples/getting_started/write_sfincs_output_tokamak.py")
