"""Analytic tokamak: one native case, one solve, one certified result.  Start here.

The whole native workflow in one screen: describe the plasma as a ``dkx.Case``,
call ``dkx.run``, read SI moments off the ``dkx.Result``, print the certificate
that says whether to believe them, save NetCDF, plot.  Every later rung changes
one thing about this script and leaves the rest alone.

Physics: concentric circular-cross-section tokamak, one deuterium species,
pitch-angle-scattering collisions, three flux surfaces, no radial electric
field.  Axisymmetric, so a single toroidal grid point resolves it exactly.

Expected runtime: ~4 s on a laptop CPU, nearly all of it JAX compilation.

Equivalent CLI:
  dkx run examples/01_tokamak_profile/case.toml --out examples/output/01_tokamak_profile/result.nc
  dkx plot examples/output/01_tokamak_profile/result.nc
"""

# 1. Imports
from pathlib import Path

import numpy as np

import dkx

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "01_tokamak_profile"
CASE_FILE = HERE / "case.toml"
RESULT_FILE = OUT_DIR / "result.nc"
PLOT_FILE = OUT_DIR / "result.png"

# The surfaces are normalized toroidal flux, psi_N = psi/psi_edge, and the
# profiles below carry one value per surface in that same order.  Gradients are
# taken across them, so at least two are required.
SURFACES = (0.09, 0.16, 0.25)

# 3. Geometry and species construction
GEOMETRY = {
    # "analytic" takes a configuration *name*, not a path: tokamak,
    # lhd_standard, lhd_inward or w7x_standard.
    "format": "analytic",
    "file": "tokamak",
    "surfaces": list(SURFACES),
}
SPECIES = [
    {
        "name": "deuterium",
        "charge": 1,
        "mass_amu": 2.014,
        "density_m3": [8.0e19, 7.0e19, 5.8e19],
        "temperature_keV": [1.0, 0.8, 0.6],
    },
]

# 4. Physics and numerical configuration
PHYSICS = {
    "model": "full_local",
    # pitch_angle_scattering is cheap but has no momentum-restoring term; use
    # linearized_fokker_planck for anything whose headline number is a current.
    "collisions": "pitch_angle_scattering",
    "magnetic_drifts": "dkes",
    "phi1": "off",
}
ELECTRIC_FIELD = {"mode": "prescribed", "value_kV_m": 0.0}
# Deliberately small so this runs in seconds.  It is *not* converged: rung 06
# measures how far off it is.  Refine before quoting any number from it.
RESOLUTION = {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}
SOLVER = {"method": "auto", "relative_tolerance": 1.0e-8, "memory_fraction": 0.75, "reuse": "auto"}
# end of parameters

case = dkx.Case.from_mapping(
    {
        "schema": 1,
        "name": "analytic_tokamak_profile",
        "run": {"workflow": "profile", "progress": True},
        "geometry": GEOMETRY,
        "species": SPECIES,
        "physics": PHYSICS,
        "electric_field": ELECTRIC_FIELD,
        "resolution": RESOLUTION,
        "solver": SOLVER,
        "output": {"file": "analytic_tokamak_profile.nc", "plots": True},
    },
    source_path=CASE_FILE,
)

# The case ID is a hash of the case content, so this proves the dict above and
# case.toml beside it are the same physics -- the CLI line in the docstring
# solves exactly what this script does.
from_toml = dkx.Case.from_file(CASE_FILE)
print(f"case id (Python) = {case.case_id[:12]}")
print(f"case id (TOML)   = {from_toml.case_id[:12]}")
assert case.case_id == from_toml.case_id, "run.py and case.toml have drifted apart"

# 5. Run
OUT_DIR.mkdir(parents=True, exist_ok=True)
result = dkx.run(case)

# 6. Print a scientific summary and certificate
certificate = result.certificate()
print("\n=== Final results ===")
print(f"  workflow: {result.workflow} on {len(SURFACES)} surfaces, {len(SPECIES)} species")
for index, psi_n in enumerate(np.asarray(result.arrays["surface"], dtype=float)):
    gamma = float(np.asarray(result.arrays["particle_flux_m2_s"])[index, 0])
    heat = float(np.asarray(result.arrays["heat_flux_W_m2"])[index, 0])
    current = float(np.asarray(result.arrays["parallel_current_A_T_m2"])[index])
    print(
        f"  psi_N={psi_n:.2f}  Gamma = {gamma:+.4e} m^-2 s^-1"
        f"  Q = {heat:+.4e} W m^-2  <j.B> = {current:+.4e} A T m^-2"
    )
print(f"  converged: {certificate['converged']}")
print(f"  solver route: {certificate['solver_route']}")
print(f"  residual norm: {certificate['residual_norm']:.3e}")
print(f"  dkx {certificate['dkx_version']} on {certificate['device']}, {certificate['precision']}")

# 7. Save native result
saved = result.save(RESULT_FILE)
print(f"  Wrote result: {saved}")

# 8. Plot publication-ready outputs
plotted = result.plot(PLOT_FILE)
print(f"  Saved plot: {plotted}")
print("Done: examples/01_tokamak_profile/run.py")
