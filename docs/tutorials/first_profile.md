# Your first profile

This tutorial runs one native DKX case from a file, reads its outputs, and then
asks the question every result needs answered before it is used: is it
converged in resolution? It follows `examples/01_tokamak_profile` and
`examples/06_convergence_certificate`.

## The case file

A native case is a TOML (or JSON) file with one table per concern. This is
`examples/01_tokamak_profile/case.toml`, without its comments:

```toml
schema = 1
name = "analytic_tokamak_profile"

[run]
workflow = "profile"
progress = true

[geometry]
format = "analytic"
file = "tokamak"
surfaces = [0.09, 0.16, 0.25]

[[species]]
name = "deuterium"
charge = 1
mass_amu = 2.014
density_m3 = [8.0e19, 7.0e19, 5.8e19]
temperature_keV = [1.0, 0.8, 0.6]

[physics]
model = "full_local"
collisions = "pitch_angle_scattering"
magnetic_drifts = "dkes"
phi1 = "off"

[electric_field]
mode = "prescribed"
value_kV_m = 0.0

[resolution]
theta = 9
zeta = 1
pitch = 8
speed = 4

[solver]
method = "auto"
relative_tolerance = 1.0e-8
memory_fraction = 0.75
reuse = "auto"

[output]
file = "analytic_tokamak_profile.nc"
plots = true
```

Read it top to bottom:

- `geometry.surfaces` are values of normalized toroidal flux
  $\psi_N = \psi/\psi_{\mathrm{edge}}$. Profiles carry one value per surface in
  the same order, and radial gradients are taken across them, so at least two
  surfaces are needed. For `format = "analytic"`, `file` names a built-in
  configuration (`tokamak`, `lhd_standard`, `lhd_inward`, `w7x_standard`), not
  a path.
- `species` are in SI and keV. Add a second `[[species]]` table for a
  multi-species run.
- `physics.collisions = "pitch_angle_scattering"` is fast but has no momentum
  restoring term. Use `"linearized_fokker_planck"` for anything whose headline
  number is a current ({doc}`../physics/collisions`).
- `resolution` sets the grid in poloidal angle, toroidal angle, Legendre pitch
  modes and speed. `zeta = 1` is exact here only because the field is
  axisymmetric.
- `solver.method = "auto"` lets DKX pick the route from the operator's
  structure ({doc}`../numerics/solver_routes`).

Every field is documented in {doc}`../user_guide/inputs`; `dkx schema` prints a
complete example.

## Run it

```console
dkx validate examples/01_tokamak_profile/case.toml
dkx run examples/01_tokamak_profile/case.toml --out examples/output/01_tokamak_profile/result.nc
dkx inspect examples/output/01_tokamak_profile/result.nc
dkx plot examples/output/01_tokamak_profile/result.nc
```

`dkx validate` checks the file and prints its deterministic case ID, a hash of
the case content. The same run from Python:

```python
import numpy as np
import dkx

case = dkx.Case.from_file("examples/01_tokamak_profile/case.toml")
result = dkx.run(case)

for i, psi_n in enumerate(np.asarray(result.arrays["surface"])):
    gamma = float(result.arrays["particle_flux_m2_s"][i, 0])
    heat = float(result.arrays["heat_flux_W_m2"][i, 0])
    current = float(result.arrays["parallel_current_A_T_m2"][i])
    print(f"psi_N={psi_n:.2f}  Gamma={gamma:+.3e} m^-2 s^-1  Q={heat:+.3e} W m^-2  <j.B>={current:+.3e} A T m^-2")

result.save("result.nc")
result.plot("result.png")
```

Arrays indexed `[surface, species]` are per species; the parallel current is
summed over species. The full list of arrays and their units is in
{doc}`../user_guide/outputs`. The example script
`python examples/01_tokamak_profile/run.py` does all of this and also asserts
that its in-script case and `case.toml` have the same case ID.

## Read the certificate

```python
certificate = result.certificate()
print(certificate["converged"], certificate["solver_route"], certificate["residual_norm"])
print(certificate["dkx_version"], certificate["device"], certificate["precision"])
```

The certificate records whether the linear solve met its tolerance, which route
solved it, the residual of the original equation, and the software version,
device and precision the numbers came from. It says the linear system was
solved accurately. It says nothing about whether the grid is fine enough.

## Check convergence

```console
dkx converge examples/06_convergence_certificate/case.toml
dkx converge examples/06_convergence_certificate/case.toml --format json
```

`dkx converge` refines each phase-space axis by a factor (default `--factor 1.5`)
and then all axes together, and reports the relative change of each observable
against a tolerance (default `--tolerance 0.02`). The joint refinement is the one
that decides: axes couple, and four axes that each look settled alone are not
evidence that the case is converged.

For this example the answer is "not converged". The comments in `case.toml`
record that refining `theta` from 9 to 14 alone moves the particle flux by
1187%, and that at `pitch = 8` the `theta` axis looks settled to 0.2% while at
`pitch = 40` the same refinement moves the outputs by 74%. The example is sized
to run in seconds; it demonstrates the workflow, not the transport of this
configuration. Before quoting a number, raise the resolution until
`dkx converge` passes. The procedure and the options of the `[convergence]` table
are in {doc}`../user_guide/convergence`.

## Next

- Solve for $E_r$ instead of prescribing it: {doc}`ambipolar_er`.
- Use a real equilibrium: {doc}`stellarator_from_vmec`.
