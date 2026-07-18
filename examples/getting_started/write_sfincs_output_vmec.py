"""Write a tiny VMEC ``sfincsOutput.h5`` with an explicit ``wout_path`` override.

What this example teaches:
  - how the supported ``geometryScheme=5`` workflow reads magnetic geometry
    from a VMEC ``wout`` file,
  - how the ``wout_path`` compatibility alias (used by both the CLI and the
    Python API) points a run at a specific VMEC equilibrium; known public
    fixtures are fetched into the local data cache on demand,
  - how to read the resulting output file back and inspect grids + geometry.

Physics context: ``geometryScheme=5`` maps a VMEC MHD equilibrium onto the
flux surface SFINCS needs, so the neoclassical solve runs on realistic
stellarator geometry [M. Landreman, H. M. Smith, A. Mollen and P. Helander,
Phys. Plasmas 21, 042503 (2014); SFINCS technical documentation,
https://github.com/landreman/sfincs].  The tiny fixture here keeps the run to
a second while still exercising the full VMEC-to-SFINCS geometry path.

Run:
  python examples/getting_started/write_sfincs_output_vmec.py
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

# A tiny geometryScheme=5 VMEC deck plus the explicit equilibrium override.
INPUT_NAMELIST = REPO_ROOT / "tests" / "ref" / "output_scheme5_1species_tiny.input.namelist"
WOUT_PATH = Path("wout_w7x_standardConfig.nc")  # public fixture, fetched on demand

# Datasets to echo (grid sizes + two normalized geometry scalars).
SUMMARY_KEYS = ["Ntheta", "Nzeta", "Nx", "Nxi", "FSABHat2", "B0OverBBar"]

OUTPUT_DIR = output_dir(__file__)
OUT_PATH = OUTPUT_DIR / "sfincsOutput_vmec.h5"

# ----------------------------------------------------------------------------
# 1) Write the output file, pointing at the VMEC equilibrium
# ----------------------------------------------------------------------------
print("=== examples/getting_started/write_sfincs_output_vmec.py ===")
print(f"Step 1: writing {OUT_PATH.name} from {INPUT_NAMELIST.name}")
print(f"  wout_path: {WOUT_PATH}")
write_output(INPUT_NAMELIST, OUT_PATH, wout_path=WOUT_PATH)
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
print("Done: examples/getting_started/write_sfincs_output_vmec.py")
