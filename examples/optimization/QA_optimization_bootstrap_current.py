"""QA optimization with an optional bootstrap-current objective.

This example intentionally mirrors ``vmex/examples/optimization/QA_optimization.py``
so the only conceptual change is easy to audit: set
``INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE`` below to compare the original
quasisymmetry/iota/aspect optimization against the same optimization with an
added VMEC ``JDotB`` current penalty.

Run from the ``dkx`` repository root:

    python examples/optimization/QA_optimization_bootstrap_current.py

Then flip ``INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE`` and rerun.  The two runs write
to separate output directories so their ``wout_final.nc``, ``history.json``, and
plots can be compared with
``qa_nfp2_bootstrap_current_comparison.py --comparison-result-dir ...``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# User parameters.  These are intended to be edited directly, matching the
# style of vmex/examples/optimization/QA_optimization.py.
# ---------------------------------------------------------------------------

# Local vmex checkout.  Override without editing this file with:
#   DKX_VMEX_ROOT=/path/to/vmex python examples/optimization/QA_optimization_bootstrap_current.py
VMEX_ROOT_HINT = Path(os.environ.get("DKX_VMEX_ROOT", "/Users/rogeriojorge/local/vmex"))

# Use the same QA seeds as vmex's public QA_optimization.py.
USE_SIMPLE_SEED = True  # Start from near-circular RBC(0,0), RBC(0,1), ZBS(0,1).
SIMPLE_SEED_PERTURBATION = 1.0e-5

# Keep this example fast enough for iteration.  The public vmex script uses
# MAX_MODE=5; use 3 here to compare current-objective behavior quickly.
MAX_MODE = 3
MIN_VMEC_MODE = MAX_MODE + 2
MAX_NFEV = 45
CONTINUATION_NFEV = 18
USE_MODE_CONTINUATION = not USE_SIMPLE_SEED

# Toggle this one line to compare QA-only and QA+current-objective runs.
INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE = False

# Output directories are separated automatically so the two runs do not
# overwrite each other.
OUTPUT_ROOT = Path("results/qa_opt_bootstrap_current_maxmode3")
OUTPUT_DIR = OUTPUT_ROOT / ("with_jdotb_current_objective" if INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE else "qa_only")

# Optimizer parameters.
METHOD = "scipy"  # Try also "auto", "gauss_newton", "scipy_matrix_free", "lbfgs_adjoint", or "scalar_trust".
SCIPY_TR_SOLVER = "exact"  # For METHOD="scipy": "lsmr" is memory-light; "exact" is dense.
SCIPY_LSMR_MAXITER = None
FTOL = 1.0e-5
GTOL = 1.0e-5
XTOL = 1.0e-6
INNER_MAX_ITER = 90
INNER_FTOL = 1.0e-9
TRIAL_MAX_ITER = 90
TRIAL_FTOL = 1.0e-9
SOLVER_DEVICE = None  # None uses JAX default; set "cpu" or "gpu" to force one backend.
USE_ESS = True
ALPHA = 1.2

# Output controls.
SAVE_STAGE_INPUTS = True
SAVE_STAGE_WOUTS = False
MAKE_PLOTS = True

# Physics targets and least-squares objective weights.  These are SIMSOPT-style
# tuple weights, so vmex minimizes sqrt(weight) * (J - target).
TARGET_ASPECT = 5.0
TARGET_IOTA = 0.41
HELICITY_M = 1
HELICITY_N = 0
SURFACES = np.arange(0.0, 1.01, 0.1)
ASPECT_WEIGHT = 1.0
IOTA_WEIGHT = 10_000.0
QS_WEIGHT = 1.0

# VMEC current objective.  This is an equilibrium-current diagnostic objective,
# not a completed kinetic dkx bootstrap-current objective.  It is the
# right fast VMEC-side knob for "small <J.B>" design scans; accepted candidates
# should still be promoted to dkx scans and checked with
# FSABjHatOverRootFSAB2.
JDOTB_SURFACES = (0.25, 0.50, 0.75)
JDOTB_NORMALIZATION = 50.0
JDOTB_WEIGHT = 1.0

# Optional finite-beta Redl bootstrap-current mismatch.  Leave this disabled for
# the vacuum-style QA seed above unless you intentionally move to a finite-beta
# input deck and review the profile coefficients.
INCLUDE_REDL_BOOTSTRAP_MISMATCH = False
REDL_BOOTSTRAP_WEIGHT = 1.0
BOOTSTRAP_SURFACES = (0.25, 0.50, 0.75)
NE_COEFFS = [3.0e20, 0.0, 0.0, 0.0, 0.0, -2.97e20]  # m^-3, polynomial in s.
TE_COEFFS = [15.0e3, -14.85e3]  # eV; Ti defaults to Te in vmex's residual.


def _import_vmex():
    """Import vmex from the requested checkout or an installed package."""

    if VMEX_ROOT_HINT.exists() and str(VMEX_ROOT_HINT) not in sys.path:
        sys.path.insert(0, str(VMEX_ROOT_HINT))
    import vmex as vj  # type: ignore[import-not-found]
    from vmex._compat import enable_x64

    root = VMEX_ROOT_HINT if VMEX_ROOT_HINT.exists() else Path(vj.__file__).resolve().parents[1]
    enable_x64(True)
    return vj, root


def _current_summary(wout) -> dict[str, float]:
    """Return a compact VMEC current diagnostic summary for final logging."""

    jdotb = np.asarray(getattr(wout, "jdotb", []), dtype=float)
    bdotb = np.asarray(getattr(wout, "bdotb", np.ones_like(jdotb)), dtype=float)
    if jdotb.size == 0:
        return {"jdotb_rms": float("nan"), "jdotb_over_root_bdotb_rms": float("nan")}
    root_b2 = np.sqrt(np.maximum(np.abs(bdotb), 1.0e-300))
    current = np.divide(jdotb, root_b2, out=np.zeros_like(jdotb), where=root_b2 > 0.0)
    active = slice(1, None)
    return {
        "jdotb_rms": float(np.sqrt(np.mean(jdotb[active] ** 2))),
        "jdotb_over_root_bdotb_rms": float(np.sqrt(np.mean(current[active] ** 2))),
    }


# ---------------------------------------------------------------------------
# Flat pipeline (runs on
# ``python examples/optimization/QA_optimization_bootstrap_current.py``).
# ---------------------------------------------------------------------------
vj, vmex_root = _import_vmex()
data_dir = vmex_root / "examples" / "data"
warm_start_input_file = data_dir / "input.nfp2_QA_omnigenity"
simple_seed_input_file = data_dir / "input.minimal_seed_nfp2"
input_file = simple_seed_input_file if USE_SIMPLE_SEED else warm_start_input_file

input_file = vj.prepare_simple_omnigenity_seed_input(
    input_file,
    OUTPUT_DIR,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    enabled=USE_SIMPLE_SEED,
    perturbation=SIMPLE_SEED_PERTURBATION,
)
stage_modes = vj.qs_stage_modes(
    max_mode=MAX_MODE,
    use_mode_continuation=USE_MODE_CONTINUATION,
    continuation_nfev=CONTINUATION_NFEV,
)

print("\nQA optimization with optional bootstrap-current objective")
print(f"  vmex root:       {vmex_root}")
print(f"  input file:          {input_file}")
print(f"  output dir:          {OUTPUT_DIR}")
print(f"  max mode:            {MAX_MODE}")
print(f"  bootstrap objective: {'enabled' if INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE else 'disabled'}")

vmec = vj.FixedBoundaryVMEC.from_input(
    input_file,
    max_mode=MAX_MODE,
    min_vmec_mode=MIN_VMEC_MODE,
    output_dir=OUTPUT_DIR,
)

aspect = vj.AspectRatio()
iota = vj.MeanIota()
qs = vj.QuasisymmetryRatioResidual(
    helicity_m=HELICITY_M,
    helicity_n=HELICITY_N,
    surfaces=SURFACES,
)

objective_tuples = [
    (aspect.J, TARGET_ASPECT, ASPECT_WEIGHT),
    (iota.J, TARGET_IOTA, IOTA_WEIGHT),
    (qs.J, 0.0, QS_WEIGHT),
]

if INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE:
    objective_tuples.append(
        (
            vj.JDotB(
                surfaces=JDOTB_SURFACES,
                normalize=JDOTB_NORMALIZATION,
            ).J,
            0.0,
            JDOTB_WEIGHT,
        )
    )

# If you prefer the upstream-file style, comment out the if-block above and
# uncomment this objective tuple for a hard-coded QA+current run:
#
# objective_tuples.append(
#     (vj.JDotB(surfaces=JDOTB_SURFACES, normalize=JDOTB_NORMALIZATION).J, 0.0, JDOTB_WEIGHT)
# )

if INCLUDE_REDL_BOOTSTRAP_MISMATCH:
    objective_tuples.append(
        (
            vj.RedlBootstrapMismatch(
                helicity_n=HELICITY_N,
                ne_coeffs=NE_COEFFS,
                Te_coeffs=TE_COEFFS,
                surfaces=BOOTSTRAP_SURFACES,
            ).J,
            0.0,
            REDL_BOOTSTRAP_WEIGHT,
        )
    )

problem = vj.LeastSquaresProblem.from_tuples(objective_tuples)

print("\nAssembled least-squares problem:")
print(f"  objectives: {', '.join(problem.objective_names)}")
print(f"  scalar terms: {problem.scalar_objective_names}")

result = vj.least_squares_solve(
    vmec,
    problem,
    stage_modes=stage_modes,
    max_nfev=MAX_NFEV,
    continuation_nfev=CONTINUATION_NFEV,
    method=METHOD,
    ftol=FTOL,
    gtol=GTOL,
    xtol=XTOL,
    use_ess=USE_ESS,
    ess_alpha=ALPHA,
    label=f"QA max_mode={MAX_MODE} {'+ JDotB' if INCLUDE_BOOTSTRAP_CURRENT_OBJECTIVE else 'QA-only'}",
    use_mode_continuation=USE_MODE_CONTINUATION,
    inner_max_iter=INNER_MAX_ITER,
    inner_ftol=INNER_FTOL,
    trial_max_iter=TRIAL_MAX_ITER,
    trial_ftol=TRIAL_FTOL,
    solver_device=SOLVER_DEVICE,
    scipy_tr_solver=SCIPY_TR_SOLVER,
    scipy_lsmr_maxiter=SCIPY_LSMR_MAXITER,
    save_stage_inputs=SAVE_STAGE_INPUTS,
    save_stage_wouts=SAVE_STAGE_WOUTS,
    save_final_outputs=False,
)

history = result.history
objective_history = result.objective_history
timing = result.timing_summary
result_summary = result.summary
saved_paths = vj.save_optimization_result(result, output_dir=OUTPUT_DIR)

print("\nFinal diagnostics from result.history:")
print(f"  stages:           {result_summary['stage_modes']}")
print(f"  aspect ratio:     {history['aspect_final']:.6g}")
print(f"  mean iota:        {history['iota_final']:.6g}")
print(f"  QS objective:     {history['qs_final']:.6e}")
print(f"  total objective:  {history['objective_final']:.6e}")
print(f"  wall time:        {timing['total_wall_time_s']:.2f} s")
print(f"  objective samples: {objective_history[:5]} ... {objective_history[-3:]}")

print("\nFiles saved from result objects:")
for name, path in saved_paths.as_dict().items():
    print(f"  {name}: {path}")

wout_final = vj.load_wout(saved_paths.final_wout)
current = _current_summary(wout_final)
print("\nVMEC current diagnostic from final wout:")
print(f"  rms(jdotb):                 {current['jdotb_rms']:.6e}")
print(f"  rms(jdotb/sqrt(bdotb)):     {current['jdotb_over_root_bdotb_rms']:.6e}")

theta, zeta, b_lcfs = vj.vmecplot2_bmag_grid(
    wout_final,
    s_index=-1,
    ntheta=64,
    nzeta=64,
    zeta_max=2.0 * np.pi / float(wout_final.nfp),
)
print("\nLCFS |B| data from vmecplot2_bmag_grid:")
print(f"  theta grid: {theta.shape}, zeta grid: {zeta.shape}, B grid: {b_lcfs.shape}")
print(f"  Bmin/Bmax:  {np.min(b_lcfs):.6g} / {np.max(b_lcfs):.6g}")

if MAKE_PLOTS:
    print("\nGenerating initial-vs-final plots:")
    plot_paths = {
        "boundary_comparison": vj.plot_3d_boundary_comparison(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "initial_vs_final_lcfs_boozer_bmag_contours": vj.plot_boozer_lcfs_bmag_comparison(
            saved_paths.initial_wout,
            saved_paths.final_wout,
            outdir=OUTPUT_DIR,
        ),
        "objective_history": vj.plot_objective_history(
            saved_paths.history,
            outdir=OUTPUT_DIR,
        ),
    }
    print("\nPlot files selected by this script:")
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

print("\nCompare the two modes with:")
print(
    "  python examples/optimization/qa_nfp2_bootstrap_current_comparison.py "
    f"--vmex-root {vmex_root} "
    f"--qa-result-dir {OUTPUT_ROOT / 'qa_only'} "
    f"--comparison-result-dir {OUTPUT_ROOT / 'with_jdotb_current_objective'}"
)
