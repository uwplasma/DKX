"""What can I set, and what comes back?  Print it rather than guess.

Working in an IDE, the fastest way to learn an API is to print it.  This
script answers the three questions a new user actually has:

  1. which parameters can I pass to ``dkx.run``?
  2. what did the solve give me back?
  3. what can I plot?

Nothing here is special-cased for the example -- it reads the same metadata
the code uses, so it stays true as DKX changes.

Physics: none.  This is a tour of the interface, on the smallest case that
solves, so it is fast enough to re-run whenever you forget a name.

Expected runtime: ~5 s on a laptop CPU.

Achieved: lists 100+ settable parameters grouped by namelist section, and the
moments the run produced with their shapes.
"""

import dataclasses
from pathlib import Path

import numpy as np

import dkx

# --------------------------- parameters -------------------------------------
DECK = Path(__file__).resolve().parent / "input.namelist"
SHOW_PER_SECTION = 8  # parameters to list per section; raise to see them all
# ----------------------------- end of parameters ----------------------------

# 1. WHAT CAN I SET -----------------------------------------------------------
# Every input parameter is a field on one of the seven namelist sections, and
# each carries its Fortran spelling.  dkx.run accepts those spellings directly.
case = dkx.SfincsInput.from_params()
print("=" * 72)
print("SETTABLE PARAMETERS  (pass any of these to dkx.run as keywords)")
print("=" * 72)
for section_name in ("general", "geometry", "species", "physics", "resolution",
                     "other", "preconditioner"):  # fmt: skip
    section = getattr(case, section_name)
    fields = [f for f in dataclasses.fields(section) if f.metadata.get("nml")]
    print(f"\n&{section_name}  ({len(fields)} parameters)")
    for field_info in fields[:SHOW_PER_SECTION]:
        default = getattr(section, field_info.name)
        shown = f"{default!r}"
        print(f"    {field_info.metadata['nml']:<34s} = {shown[:30]}")
    if len(fields) > SHOW_PER_SECTION:
        print(f"    ... and {len(fields) - SHOW_PER_SECTION} more "
              f"(raise SHOW_PER_SECTION to see them)")  # fmt: skip

# 2. WHAT COMES BACK ----------------------------------------------------------
run = dkx.run(DECK)
print("\n" + "=" * 72)
print("WHAT THE RUN RETURNED")
print("=" * 72)
print(f"  run.moments       {len(run.moments)} entries, keyed by sfincsOutput names")
print(f"  run.solve_result  route={run.solve_result.route}, "
      f"converged={bool(run.solve_result.converged)}, "
      f"residual={float(run.solve_result.residual_norms[-1]):.2e}")  # fmt: skip
print(f"  run.input         the SfincsInput that was solved")
print(f"  run.operator      the drift-kinetic operator (differentiable)")
print(f"  run.state_vector  shape {np.shape(run.state_vector)}")

print("\n  moments (name, shape, value if scalar):")
for name in sorted(run.moments):
    value = np.asarray(run.moments[name])
    summary = f"{float(value.ravel()[0]):+.5e}" if value.size == 1 else f"shape {value.shape}"
    print(f"    {name:<40s} {summary}")

# 3. WHAT CAN I PLOT ----------------------------------------------------------
print("\n" + "=" * 72)
print("PLOTTING")
print("=" * 72)
print("""  dkx.plot(run_or_path, out="fig.png")             six-panel figure
  dkx.plot(..., style="summary")                   compact three-panel page
  dkx.plot(..., style="summary", out="fig.pdf")    four-page diagnostics book
  dkx.plot(..., show=True)                         also show it (Spyder/Jupyter)

  A .pdf only expands into pages under style="summary"; the default panels
  figure is one page whatever the suffix.

  For anything else, read run.moments and use matplotlib -- see
  plot_custom.py, plot_monoenergetic.py and plot_ambipolar_er.py.""")
