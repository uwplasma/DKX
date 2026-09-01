"""Boozer-coordinate geometry from a ``.bc`` file: the third geometry format.

Rung 02 with ``geometry.format = "boozer"``.  A ``.bc`` file already carries
|B| as a Boozer spectrum, so nothing has to be transformed on the way in --
this is the format most stellarator neoclassical work circulates in, and the
one to reach for when a collaborator sends you a configuration.

DKX auto-detects whether the file uses the cosine-only (stellarator-symmetric)
or the asymmetric column convention, so the case never names a geometry-scheme
number.  The bundled fixture is non-stellarator-symmetric, which exercises the
harder of the two.

Physics: one deuterium species, pitch-angle scattering, two surfaces, no radial
electric field.  Both angles are resolved (``theta`` and ``zeta`` are 5) because
a stellarator field varies along the field line as well as across it.

Expected runtime: ~4 s on a laptop CPU.

Equivalent CLI:
  dkx run examples/03_boozer_stellarator/case.toml --out examples/output/03_boozer_stellarator/result.nc
"""

# 1. Imports
from pathlib import Path

import numpy as np

import dkx

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "03_boozer_stellarator"
CASE_FILE = HERE / "case.toml"
RESULT_FILE = OUT_DIR / "result.nc"
PLOT_FILE = OUT_DIR / "result.png"

SURFACES = (0.20, 0.30)

# 3. Geometry and species construction
GEOMETRY = {
    "format": "boozer",
    "file": "../../tests/ref/nonStelSym_tiny_geometryScheme12.bc",
    "surfaces": list(SURFACES),
}
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
ELECTRIC_FIELD = {"mode": "prescribed", "value_kV_m": 0.0}
RESOLUTION = {"theta": 5, "zeta": 5, "pitch": 8, "speed": 4}
# A tighter tolerance than rungs 01-02: the Boozer operator here takes the
# Krylov route rather than the structured direct one, so the tolerance is what
# actually stops the iteration instead of being met by the factorization.
SOLVER = {"method": "auto", "relative_tolerance": 1.0e-10, "memory_fraction": 0.75, "reuse": "auto"}
# end of parameters

case = dkx.Case.from_mapping(
    {
        "schema": 1,
        "name": "native_boozer_profile",
        "run": {"workflow": "profile", "progress": True},
        "geometry": GEOMETRY,
        "species": SPECIES,
        "physics": PHYSICS,
        "electric_field": ELECTRIC_FIELD,
        "resolution": RESOLUTION,
        "solver": SOLVER,
        "output": {"file": "native_boozer_profile.nc", "plots": True},
    },
    source_path=CASE_FILE,
)
from_toml = dkx.Case.from_file(CASE_FILE)
assert case.case_id == from_toml.case_id, "run.py and case.toml have drifted apart"
print(f"case id = {case.case_id[:12]} (run.py and case.toml agree)")
print(f"equilibrium: {case.geometry_path}")

# 5. Run
OUT_DIR.mkdir(parents=True, exist_ok=True)
result = dkx.run(case)

# 6. Print a scientific summary and certificate
certificate = result.certificate()
print("\n=== Final results ===")
for index, psi_n in enumerate(np.asarray(result.arrays["surface"], dtype=float)):
    gamma = float(np.asarray(result.arrays["particle_flux_m2_s"])[index, 0])
    heat = float(np.asarray(result.arrays["heat_flux_W_m2"])[index, 0])
    current = float(np.asarray(result.arrays["parallel_current_A_T_m2"])[index])
    print(
        f"  psi_N={psi_n:.2f}  Gamma = {gamma:+.4e} m^-2 s^-1"
        f"  Q = {heat:+.4e} W m^-2  <j.B> = {current:+.4e} A T m^-2"
    )
# The route is chosen from the operator's structure, not requested by hand, and
# the certificate records both the choice and the reason for it.
print(f"  converged: {certificate['converged']}")
print(f"  solver route: {certificate['solver_route']}")
print(f"  route reason: {certificate['route_reason']}")
print(f"  residual norm: {certificate['residual_norm']:.3e}")

# 7. Save native result
saved = result.save(RESULT_FILE)
print(f"  Wrote result: {saved}")

# 8. Plot publication-ready outputs
plotted = result.plot(PLOT_FILE)
print(f"  Saved plot: {plotted}")
print("Done: examples/03_boozer_stellarator/run.py")
