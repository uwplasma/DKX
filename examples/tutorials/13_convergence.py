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
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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

# 3b. A resolution ladder: one solve per rung, all axes refined together
LADDER = [(9, 8, 4), (17, 16, 5)] if SMOKE else [(9, 8, 4), (17, 16, 5), (25, 24, 6), (33, 32, 6), (41, 40, 7)]
ladder = []
for n_theta, n_pitch, n_speed in LADDER:
    grid = replace(case.resolution, theta=n_theta, pitch=n_pitch, speed=n_speed)
    arrays = dkx.run(replace(case, resolution=grid)).arrays
    ladder.append([float(arrays[k][0].sum()) for k in OBSERVABLES])
ladder = np.array(ladder)

# 4. Plot, save and print
OUT.mkdir(parents=True, exist_ok=True)
rows = [*report.refinements] + ([report.joint] if report.joint is not None else [])
fig, (axis, rung) = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
for column, name in enumerate(("particle flux", "heat flux", "current")):
    rung.plot(range(len(ladder)), ladder[:, column] / ladder[-1, column], "o-", label=name)
rung.set_xticks(range(len(ladder)), [f"{t}/{p}/{x}" for t, p, x in LADDER], fontsize=7)
rung.set_xlabel("theta / pitch / speed")
rung.set_ylabel("value / value on the finest rung")
rung.legend(fontsize=8)
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
print("  ladder (theta/pitch/speed): particle flux, heat flux, <j.B> at the first surface")
for (t, p, x), row in zip(LADDER, ladder):
    print(f"  {t:3d}/{p:2d}/{x}  {row[0]:+.4e}  {row[1]:+.4e}  {row[2]:+.4e}")
print(f"  converged at {report.tolerance:.0%}: {report.converged}")
