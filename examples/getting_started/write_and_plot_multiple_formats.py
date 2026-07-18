"""Write HDF5/NetCDF/NPZ outputs and render a PDF diagnostics panel.

What this example teaches:
  - how ``dkx.api.write_output`` picks the on-disk format from the file suffix
    (``.h5`` HDF5, ``.nc`` NetCDF, ``.npz`` NumPy archive) from the same solve,
  - how ``dkx.io.read_sfincs_output_file`` reads any of the three formats back
    into the same dict of datasets,
  - how ``dkx.plotting.plot_sfincs_output_summary`` builds the standard
    multi-panel diagnostics figure.

Physics context: all three formats carry the identical SFINCS v3 neoclassical
output (grids, normalized geometry, per-species moments); only the container
differs [SFINCS technical documentation, https://github.com/landreman/sfincs].
This runs on the fast getting-started ``geometryScheme=4`` deck.

Run:
  python examples/getting_started/write_and_plot_multiple_formats.py
"""

from __future__ import annotations

from pathlib import Path

from dkx.api import write_output
from dkx.io import read_sfincs_output_file
from dkx.plotting import plot_sfincs_output_summary

# ----------------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_NAMELIST = REPO_ROOT / "examples" / "getting_started" / "input.namelist"

# Output-file container formats to write from a single solve.
FORMATS = (".h5", ".nc", ".npz")

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "write_and_plot_multiple_formats"
STEM = "sfincsOutput_getting_started"

# ----------------------------------------------------------------------------
# 1) Write the same output in every container format and read each one back
# ----------------------------------------------------------------------------
print("=== examples/getting_started/write_and_plot_multiple_formats.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Step 1: writing {STEM} in {', '.join(FORMATS)}")
h5_path = OUTPUT_DIR / f"{STEM}.h5"
for suffix in FORMATS:
    out_path = OUTPUT_DIR / f"{STEM}{suffix}"
    write_output(INPUT_NAMELIST, out_path)
    data = read_sfincs_output_file(out_path)
    print(f"  wrote {out_path.name} with {len(data)} datasets")

# ----------------------------------------------------------------------------
# 2) Render the diagnostics panel from the HDF5 output
# ----------------------------------------------------------------------------
print("Step 2: rendering the diagnostics panel")
pdf_path = OUTPUT_DIR / f"{STEM}_summary.pdf"
plot_sfincs_output_summary(input_h5=h5_path, output_png=pdf_path)

# ----------------------------------------------------------------------------
# 3) Results
# ----------------------------------------------------------------------------
print("=== Final results ===")
for suffix in FORMATS:
    print(f"  Wrote output file: {STEM}{suffix}")
print(f"  Saved plot: {pdf_path.name}")
print("Done: examples/getting_started/write_and_plot_multiple_formats.py")
