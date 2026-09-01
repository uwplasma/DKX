"""Real geometry from a VMEC ``wout``: the same solve, one line different.

Rung 01 with ``geometry.format`` changed from ``"analytic"`` to ``"vmec"`` and
a file beside it.  That is the whole difference -- the species, physics and
solver blocks are untouched -- which is the point: the equilibrium is an input,
not a different code path.  Point ``GEOMETRY["file"]`` at your own
``wout_*.nc`` and rerun.

Physics: an up-down asymmetric tokamak equilibrium solved by VMEC, one
deuterium species, pitch-angle scattering, three surfaces, no radial electric
field.  ``zeta`` is now 9 rather than 1 because a VMEC field is resolved on a
toroidal grid even when the configuration is nearly axisymmetric.

Expected runtime: ~5 s on a laptop CPU.

Equivalent CLI:
  dkx run examples/02_vmec_stellarator/case.toml --out examples/output/02_vmec_stellarator/result.nc
  dkx inspect examples/output/02_vmec_stellarator/result.nc
"""

# 1. Imports
from pathlib import Path

import numpy as np

import dkx

# 2. User-editable parameters
HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "output" / "02_vmec_stellarator"
CASE_FILE = HERE / "case.toml"
RESULT_FILE = OUT_DIR / "result.nc"
PLOT_FILE = OUT_DIR / "result.png"

SURFACES = (0.16, 0.25, 0.36)

# 3. Geometry and species construction
GEOMETRY = {
    "format": "vmec",
    # Relative paths resolve beside the case file, not from the shell's working
    # directory, so the case is portable.  Swap in your own wout here.
    "file": "../../tests/ref/wout_up_down_asymmetric_tokamak.nc",
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
    "collisions": "pitch_angle_scattering",
    "magnetic_drifts": "dkes",
    "phi1": "off",
}
ELECTRIC_FIELD = {"mode": "prescribed", "value_kV_m": 0.0}
RESOLUTION = {"theta": 9, "zeta": 9, "pitch": 8, "speed": 4}
SOLVER = {"method": "auto", "relative_tolerance": 1.0e-8, "memory_fraction": 0.75, "reuse": "auto"}
# end of parameters

case = dkx.Case.from_mapping(
    {
        "schema": 1,
        "name": "native_vmec_profile",
        "run": {"workflow": "profile", "progress": True},
        "geometry": GEOMETRY,
        "species": SPECIES,
        "physics": PHYSICS,
        "electric_field": ELECTRIC_FIELD,
        "resolution": RESOLUTION,
        "solver": SOLVER,
        "output": {"file": "native_vmec_profile.nc", "plots": True},
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
print(f"  geometry sha256: {certificate['geometry_sha256'][:16]} (the equilibrium is pinned)")
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

# 7. Save native result
saved = result.save(RESULT_FILE)
print(f"  Wrote result: {saved}")

# 8. Plot publication-ready outputs
plotted = result.plot(PLOT_FILE)
print(f"  Saved plot: {plotted}")
print("Done: examples/02_vmec_stellarator/run.py")
