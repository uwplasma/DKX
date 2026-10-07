"""Tutorial 13 -- a small residual is not a converged answer: refine and look.

Every run so far printed a residual ~1e-15.  That says the *linear system*
was solved accurately; it says nothing about whether the *discretization*
(theta, zeta, pitch, speed grids) is fine enough.  This tutorial refines each
phase-space axis by a factor 1.5, then all axes together, and reports the
largest relative change of any observable,

    worst = max_observable |O_refined - O_base| / |O_base|,

against a tolerance (2% here).  The joint refinement matters: axes couple, so
four axes that each look settled are not evidence of convergence.  The base
case in ``13_convergence.toml`` is deliberately coarse, so expect "False".
``zeta`` stays 1 for this axisymmetric field (refining it measures nothing).

Run:              python examples/tutorials/13_convergence.py
CLI:              dkx converge examples/tutorials/13_convergence.toml
Smoke mode:       DKX_EXAMPLES_CI=1 python examples/tutorials/13_convergence.py
Expected runtime: see the table in examples/README.md.
"""

# 1. Imports
import os
from pathlib import Path

import matplotlib.pyplot as plt

import dkx
from dkx.workflows.converge import converge_case

# 2. Input parameters
SMOKE = os.environ.get("DKX_EXAMPLES_CI") == "1"
CASE_FILE = Path(__file__).with_suffix(".toml")
OUT = Path(__file__).resolve().parents[1] / "output" / Path(__file__).stem
AXES = ("theta", "pitch") if SMOKE else ("theta", "pitch", "speed")  # axes to refine
FACTOR = 1.5  # refinement factor per axis
TOLERANCE = 0.02  # allowed relative change; set it from the precision you need
OBSERVABLES = ("particle_flux_m2_s", "heat_flux_W_m2", "parallel_current_A_T_m2")

# 3. Run -- the base case, then one solve per refinement (each prints progress)
case = dkx.Case.from_file(CASE_FILE)
report = converge_case(case, axes=AXES, factor=FACTOR, tolerance=TOLERANCE, observables=OBSERVABLES,
                       joint=not SMOKE, emit=print)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
rows = [*report.refinements] + ([report.joint] if report.joint is not None else [])
fig, axis = plt.subplots(figsize=(6.5, 3.8), constrained_layout=True)
axis.bar([r.label for r in rows], [r.worst for r in rows],
         color=["tab:red" if r.worst > TOLERANCE else "tab:green" for r in rows])
axis.axhline(TOLERANCE, color="0.3", ls="--", label=f"tolerance {TOLERANCE:.0%}")
axis.set_yscale("log")
axis.set_ylabel("worst relative change")
axis.set_title("refine every axis, then all of them together")
axis.legend()
fig.savefig(OUT / "convergence.png", dpi=120)
print(f"\nsaved {OUT / 'convergence.png'}")
print("\n=== Summary ===")
for r in rows:
    grid = " ".join(f"{k}={v}" for k, v in sorted(r.resolution.items()))
    print(f"  {r.label:9s} {grid:36s} worst change {r.worst:9.1%}  {r.seconds:6.1f} s")
print(f"  converged at {report.tolerance:.0%}: {report.converged}")
