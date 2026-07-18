"""Write a SFINCS-style ``sfincsOutput.h5`` by driving the ``dkx`` CLI.

What this example teaches:
  - how to invoke the ``dkx`` command-line interface from Python
    (``python -m dkx write-output``),
  - how the fast ``--geometry-only`` write path produces a valid output file
    without a full drift-kinetic solve (handy for smoke tests and CI),
  - how to load the resulting HDF5 file back with ``dkx.io.read_sfincs_h5``.

Physics context: the ``dkx`` CLI mirrors the SFINCS v3 workflow -- point it at
a Fortran ``input.namelist`` and it produces the same ``sfincsOutput.h5``
dataset layout the Python API writes [SFINCS technical documentation,
https://github.com/landreman/sfincs].  ``--geometry-only`` stops after building
the grids and normalized magnetic geometry (e.g. ``FSABHat2``), which is all we
inspect here.

Run:
  python examples/getting_started/write_sfincs_output_cli.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dkx.io import read_sfincs_h5

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ on sys.path
from _example_utils import output_dir, print_dataset_summary  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# The tiny getting-started deck (geometryScheme=4).
INPUT_NAMELIST = REPO_ROOT / "examples" / "getting_started" / "input.namelist"

# Datasets to echo from the geometry-only output file.
SUMMARY_KEYS = ["FSABHat2"]

OUTPUT_DIR = output_dir(__file__)
OUT_PATH = OUTPUT_DIR / "sfincsOutput_cli.h5"

# ----------------------------------------------------------------------------
# 1) Invoke the dkx CLI (geometry-only write path)
# ----------------------------------------------------------------------------
print("=== examples/getting_started/write_sfincs_output_cli.py ===")
cmd = [
    sys.executable, "-m", "dkx", "write-output",
    "--input", str(INPUT_NAMELIST),
    "--out", str(OUT_PATH),
    "--geometry-only",
]
print("Step 1: running the CLI:")
print("  " + " ".join(cmd))
subprocess.run(cmd, check=True)
print(f"  wrote: {OUT_PATH}")

# ----------------------------------------------------------------------------
# 2) Read the output back
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
print("Done: examples/getting_started/write_sfincs_output_cli.py")
