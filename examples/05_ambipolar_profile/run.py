"""The radial electric field is not an input: solve for it from ambipolarity.

``E_r`` is fixed by the condition that no net charge leaves the flux surface,
``J_r(E_r) = sum_s Z_s Gamma_s = 0``.  Rungs 01-04 prescribed it; this one
searches for it.  Switching ``run.workflow`` to ``"ambipolar_profile"`` and
``electric_field.mode`` to ``"ambipolar"`` is the entire change -- DKX then
brackets the root on every surface, keeps *all* the roots it finds rather than
the first, classifies them (ion root, electron root, unstable), and records why
it selected the one it did.

Keeping every root matters because a stellarator can have three, and the
middle one is unstable: a solver that returns a single number cannot tell you
which branch you are on or when the profile jumped between branches.

Physics: the analytic tokamak of rung 01, one deuterium species, two surfaces,
searching +-5 kV/m.  Adaptive refinement is on, so the bracket is tightened
until the observables stop moving.

Expected runtime: ~8 s on a laptop CPU.

Equivalent CLI:
  dkx run examples/05_ambipolar_profile/case.toml --out examples/output/05_ambipolar_profile/result.nc
  dkx roots examples/output/05_ambipolar_profile/result.nc
  dkx validate examples/05_ambipolar_profile/w7x_case.toml   # the production-scale case
"""

# 1. Imports
from pathlib import Path

import numpy as np

import dkx

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "05_ambipolar_profile"
CASE_FILE = HERE / "case.toml"
SHOWCASE_CASE_FILE = HERE / "w7x_case.toml"
RESULT_FILE = OUT_DIR / "result.nc"
PLOT_FILE = OUT_DIR / "result.png"

SURFACES = (0.09, 0.16)

# 3. Geometry and species construction
GEOMETRY = {"format": "analytic", "file": "tokamak", "surfaces": list(SURFACES)}
SPECIES = [
    {
        "name": "deuterium",
        "charge": 1,
        "mass_amu": 2.014,
        "density_m3": [8.0e19, 7.0e19],
        "temperature_keV": [1.0, 0.8],
    },
]

# 4. Physics and numerical configuration
PHYSICS = {
    "model": "full_local",
    "collisions": "pitch_angle_scattering",
    "magnetic_drifts": "dkes",
    "phi1": "off",
}
# Widen search_kV_m if the run reports no bracketed root: a steeper profile
# pushes the root further out, and a bracket that misses it says "no root"
# rather than guessing.
ELECTRIC_FIELD = {
    "mode": "ambipolar",
    "search_kV_m": [-5.0, 5.0],
    "find_all_roots": True,
    "continue_branches": True,
    "search_points": 5,
    "root_tolerance_kV_m": 0.05,
    "max_root_iterations": 8,
}
RESOLUTION = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}
SOLVER = {"method": "auto", "relative_tolerance": 1.0e-8, "memory_fraction": 0.75, "reuse": "auto"}
CONVERGENCE = {
    "enabled": True,
    "observables": ["particle_flux", "heat_flux", "electric_field"],
    "relative_tolerance": 0.02,
    "max_refinements": 1,
}
# end of parameters

case = dkx.Case.from_mapping(
    {
        "schema": 1,
        "name": "analytic_ambipolar_profile",
        "run": {"workflow": "ambipolar_profile", "precision": "float64",
                "device": "auto", "progress": True},
        "geometry": GEOMETRY,
        "species": SPECIES,
        "physics": PHYSICS,
        "electric_field": ELECTRIC_FIELD,
        "resolution": RESOLUTION,
        "solver": SOLVER,
        "parallel": {"strategy": "auto"},
        "convergence": CONVERGENCE,
        "output": {"file": "outputs/analytic_ambipolar.nc", "plots": True},
    },
    source_path=CASE_FILE,
)
from_toml = dkx.Case.from_file(CASE_FILE)
assert case.case_id == from_toml.case_id, "run.py and case.toml have drifted apart"
print(f"case id = {case.case_id[:12]} (run.py and case.toml agree)")

# 5. Run
OUT_DIR.mkdir(parents=True, exist_ok=True)
result = dkx.run(case)

# 6. Print a scientific summary and certificate
certificate = result.certificate()
surfaces = np.asarray(result.arrays["surface"], dtype=float)
roots = np.asarray(result.arrays["ambipolar_root_kV_m"], dtype=float)
root_kinds = np.asarray(result.arrays["ambipolar_root_type"], dtype=object)
root_counts = np.asarray(result.arrays["ambipolar_root_count"], dtype=int)
root_currents = np.asarray(result.arrays["ambipolar_root_current_A_m2"], dtype=float)
slopes = np.asarray(result.arrays["ambipolar_root_slope_A_m2_per_kV_m"], dtype=float)
selected = np.asarray(result.arrays["selected_ambipolar_root"], dtype=int)
field = np.asarray(result.arrays["electric_field_kV_m"], dtype=float)

print("\n=== Final results ===")
for index, psi_n in enumerate(surfaces):
    print(f"  psi_N={psi_n:.2f}: {root_counts[index]} root(s), selected #{selected[index]}")
    for slot in range(int(root_counts[index])):
        print(
            f"    Er = {roots[index, slot]:+.4f} kV/m  "
            f"J_r = {root_currents[index, slot]:+.3e} A m^-2  "
            f"dJ_r/dEr = {slopes[index, slot]:+.3e} A m^-2 (kV/m)^-1  "
            f"[{root_kinds[index, slot]}]"
        )
    gamma = float(np.asarray(result.arrays["particle_flux_m2_s"])[index, 0])
    heat = float(np.asarray(result.arrays["heat_flux_W_m2"])[index, 0])
    print(
        f"    at the selected root: Er = {field[index]:+.4f} kV/m  "
        f"Gamma = {gamma:+.4e} m^-2 s^-1  Q = {heat:+.4e} W m^-2"
    )
print(f"  selection rule: {certificate['ambipolar_selection']}")
print(f"  all surfaces bracketed: {certificate['ambipolar_all_surfaces_bracketed']}")
print(f"  refinement: {certificate['ambipolar_refinement']}")
print(f"  converged: {certificate['converged']}")
print(f"  residual norm: {certificate['residual_norm']:.3e}")

# 7. Save native result
saved = result.save(RESULT_FILE)
print(f"  Wrote result: {saved}")

# 8. Plot publication-ready outputs
plotted = result.plot(PLOT_FILE)
print(f"  Saved plot: {plotted}")
print(f"  Production-scale schema showcase: {SHOWCASE_CASE_FILE.name} (validate it, do not run it)")
print("Done: examples/05_ambipolar_profile/run.py")
