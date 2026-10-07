"""A converged residual is not a converged answer.  Refine every axis and look.

Rungs 01-05 all print ``residual norm ~ 1e-15``, which says the linear system
was solved accurately -- nothing at all about whether the *discretization* is
fine enough.  Those are different questions, and only the second one decides
whether a number is publishable.

This rung refines each phase-space axis in turn and then all of them together,
and reports how far the observables moved.  The joint run is the load-bearing
one: axes couple, so four axes that each look settled on their own are not
evidence that the case is converged.  The certificate that comes back with the
result records the solver route, the residual, and the exact software and
hardware the numbers came from.

Physics: the analytic tokamak of rung 01 at a deliberately coarse resolution,
so the refinement has something to find.

Expected runtime: ~12 s on a laptop CPU.  ``zeta`` does not appear in the
table: the field is axisymmetric, so that axis is already exact at one point
and refining it is not a measurement.

Equivalent CLI:
  dkx converge examples/06_convergence_certificate/case.toml
  dkx converge examples/06_convergence_certificate/case.toml --format json
"""

# 1. Imports
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import dkx  # noqa: E402
from dkx.workflows.converge import converge_case  # noqa: E402

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "06_convergence_certificate"
CASE_FILE = HERE / "case.toml"
RESULT_FILE = OUT_DIR / "result.nc"
PLOT_FILE = OUT_DIR / "convergence.png"

# Which phase-space axes to refine, and by how much.  A factor of 1.5 is enough
# to expose a trend without paying for a doubling on every axis.
AXES = ("theta", "zeta", "pitch", "speed")
FACTOR = 1.5
# The relative movement an observable is allowed before the case counts as
# unconverged.  2% is a reporting threshold, not a physical one -- set it from
# the precision your conclusion actually needs.
TOLERANCE = 0.02
OBSERVABLES = ("particle_flux_m2_s", "heat_flux_W_m2", "parallel_current_A_T_m2")
# end of parameters

# 3. Geometry and species construction
case = dkx.Case.from_file(CASE_FILE)
print(f"case id = {case.case_id[:12]}")
print(f"geometry: {case.geometry.format} '{case.geometry.file}' on {len(case.geometry.surfaces)} surfaces")
print(f"species:  {', '.join(s.name for s in case.species)}")

# 4. Physics and numerical configuration
print(f"baseline resolution: theta={case.resolution.theta} zeta={case.resolution.zeta} "
      f"pitch={case.resolution.pitch} speed={case.resolution.speed}")
print(f"refining {list(AXES)} by x{FACTOR}, tolerance {TOLERANCE:.1%}")

# 5. Run
OUT_DIR.mkdir(parents=True, exist_ok=True)
result = dkx.run(case)
report = converge_case(
    case,
    axes=AXES,
    factor=FACTOR,
    tolerance=TOLERANCE,
    observables=OBSERVABLES,
    joint=True,
)

# 6. Print a scientific summary and certificate
certificate = result.certificate()
print("\n=== Final results ===")
gamma = float(np.asarray(result.arrays["particle_flux_m2_s"])[0, 0])
heat = float(np.asarray(result.arrays["heat_flux_W_m2"])[0, 0])
print(f"  baseline at psi_N={float(result.arrays['surface'][0]):.2f}: "
      f"Gamma = {gamma:+.4e} m^-2 s^-1  Q = {heat:+.4e} W m^-2")
print(f"  baseline residual norm: {certificate['residual_norm']:.3e} "
      f"(the linear solve, not the discretization)")

rows = [*report.refinements] + ([report.joint] if report.joint is not None else [])
print(f"\n  {'refinement':<12} {'resolution':<28} {'worst change':>13} {'seconds':>8}")
for row in rows:
    resolution = " ".join(f"{key}={value}" for key, value in sorted(row.resolution.items()))
    print(f"  {row.label:<12} {resolution:<28} {row.worst:>12.1%} {row.seconds:>8.1f}")

print(f"\n  worst single-axis change: {report.per_axis_worst:.1%}")
if report.joint is not None:
    print(f"  joint change:             {report.joint.worst:.1%}")
print(f"  axes understate the joint change: {report.axes_understate_the_joint_change}")
print(f"  converged at {report.tolerance:.1%}: {report.converged}")
print(f"  provenance: dkx {certificate['dkx_version']}, jax {certificate['jax_version']}, "
      f"{certificate['device']}, {certificate['precision']}")

# 7. Save native result
saved = result.save(RESULT_FILE)
print(f"  Wrote result: {saved}")

# 8. Plot publication-ready outputs
labels = [row.label for row in rows]
worst = [row.worst for row in rows]
figure, axis = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
colors = ["tab:red" if value > TOLERANCE else "tab:green" for value in worst]
axis.bar(labels, worst, color=colors)
axis.axhline(TOLERANCE, color="0.3", ls="--", lw=1.0, label=f"tolerance {TOLERANCE:.0%}")
axis.set_yscale("log")
axis.set_ylabel("worst relative change in an observable")
axis.set_xlabel("refinement")
axis.set_title("Refine every axis, then all of them together")
axis.grid(alpha=0.3, axis="y", which="both")
axis.legend(fontsize=9)
figure.savefig(PLOT_FILE, dpi=150)
plt.close(figure)
print(f"  Saved plot: {PLOT_FILE}")
print("Done: examples/06_convergence_certificate/run.py")
