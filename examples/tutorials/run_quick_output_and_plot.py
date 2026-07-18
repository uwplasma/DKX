"""Write small tutorial outputs and render a diagnostics panel.

What this example teaches:
  - how ``dkx.api.write_output`` writes the standard output fields quickly
    (geometry + moments in the SFINCS v3 layout) in three container formats,
  - how ``dkx.io.read_sfincs_output_file`` reads any of them back,
  - how ``dkx.plotting.plot_sfincs_output_summary`` builds the same summary
    panel that ``dkx --plot`` produces.

Physics context: this is the fast on-ramp to the tutorial notebooks -- it
exercises the full write/read/plot loop on the getting-started
``geometryScheme=4`` deck without a heavy transport solve [SFINCS technical
documentation, https://github.com/landreman/sfincs].

Run:
  python examples/tutorials/run_quick_output_and_plot.py
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

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output" / "run_quick_output_and_plot"
STEM = "sfincsOutput_tutorial"

# ----------------------------------------------------------------------------
# 1) Write the output in every container format and read each one back
# ----------------------------------------------------------------------------
print("=== examples/tutorials/run_quick_output_and_plot.py ===")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"Step 1: writing {STEM} in {', '.join(FORMATS)}")
h5_path = OUTPUT_DIR / f"{STEM}.h5"
for suffix in FORMATS:
    path = OUTPUT_DIR / f"{STEM}{suffix}"
    write_output(INPUT_NAMELIST, path)
    data = read_sfincs_output_file(path)
    print(f"  {path.name}: {len(data)} output fields")

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
print("Done: examples/tutorials/run_quick_output_and_plot.py")
