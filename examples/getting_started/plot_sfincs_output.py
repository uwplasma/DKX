"""Plot a diagnostics panel from a ``sfincsOutput.h5``/``.nc``/``.npz`` file.

What this example teaches:
  - how ``dkx.plotting.plot_sfincs_output_summary`` renders the same
    multi-panel diagnostics figure that ``dkx --plot`` produces,
  - how to point the plotter at any dkx output file and choose the output
    figure format from the file suffix (PDF here),
  - how the committed lzma-compressed reference fixtures are decompressed on
    demand so the example runs from a fresh checkout.

Physics context: the summary panel visualizes the f-independent basics of a
neoclassical solve -- the magnetic-field strength on the flux surface, the
speed grid, and the per-species fluxes and flows [SFINCS technical
documentation, https://github.com/landreman/sfincs].  Here it runs on a frozen
two-species ``geometryScheme=4`` reference output.

Run:
  python examples/getting_started/plot_sfincs_output.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from dkx.plotting import plot_sfincs_output_summary

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # examples/ on sys.path
from _example_utils import ensure_uncompressed, output_dir  # noqa: E402

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# A frozen two-species reference output (committed lzma-compressed).
INPUT_H5 = REPO_ROOT / "tests" / "ref" / "output_scheme4_2species_quick.sfincsOutput.h5"

OUTPUT_DIR = output_dir(__file__)
PLOT_PATH = OUTPUT_DIR / "sfincsOutput_summary.pdf"

# ----------------------------------------------------------------------------
# 1) Materialize the reference sample if only the compressed copy is present
# ----------------------------------------------------------------------------
print("=== examples/getting_started/plot_sfincs_output.py ===")
print(f"Step 1: preparing the input sample: {INPUT_H5.name}")
ensure_uncompressed(INPUT_H5)

# ----------------------------------------------------------------------------
# 2) Render the diagnostics panel
# ----------------------------------------------------------------------------
print("Step 2: rendering the diagnostics panel")
out_path = plot_sfincs_output_summary(input_h5=INPUT_H5, output_png=PLOT_PATH)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
print(f"  Saved plot: {out_path}")
print("Done: examples/getting_started/plot_sfincs_output.py")
