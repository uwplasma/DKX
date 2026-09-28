# First run

This page runs one small case three ways: from a Python mapping, from a case
file on the command line, and through `dkx converge`. The case is an analytic
circular tokamak with one deuterium species and a prescribed zero radial
electric field (the `profile` workflow). The grid is a teaching grid: it runs
in seconds and it is **not converged**. The last section shows how to find
that out.

## From Python

```python
import dkx

case = dkx.Case.from_mapping({           # analytic tokamak, teaching grid: seconds, not converged
    "schema": 1, "name": "tokamak", "run": {"workflow": "profile", "progress": False},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"model": "full_local", "collisions": "pitch_angle_scattering",
                "magnetic_drifts": "dkes", "phi1": "off"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4},
    "solver": {"method": "auto", "relative_tolerance": 1e-8},
})
result = dkx.run(case)
print("solver route:", result.metadata["solver_route"])
print("particle flux:", float(result.arrays["particle_flux_m2_s"][1, 0]))
```

What each piece does:

- `Case.from_mapping` validates the mapping against case schema 1 and returns
  an immutable `Case`. Omitted optional tables (`[parallel]`, `[convergence]`,
  `[output]`) take their defaults. `case.case_id` is the SHA-256 of the
  normalized content, so the same physics always has the same ID.
- `surfaces` are values of normalized toroidal flux $\psi_N$. Density and
  temperature give one value per surface; DKX differentiates the profile to
  obtain the gradients that drive transport.
- `dkx.run(case)` solves the drift-kinetic equation on each surface and
  returns a `dkx.Result`. It keeps the result in memory; pass `out="result.nc"`
  to also write it.

On this case the route is `block_tridiagonal_truncated`, the memory-bounded
structured direct solve (measured with `JAX_PLATFORMS=cpu`; see
{doc}`../numerics/solver_routes`). The arrays of the result are:

| array | dimensions | shape here | unit |
|---|---|---|---|
| `surface` | (surface) | (3,) | $\psi_N$ |
| `r_N` | (surface) | (3,) | $r/a = \sqrt{\psi_N}$ |
| `species`, `charge_e`, `mass_amu` | (species) | (1,) | name, $e$, u |
| `density_m3`, `temperature_keV` | (surface, species) | (3, 1) | m$^{-3}$, keV |
| `electric_field_kV_m` | (surface) | (3,) | kV/m |
| `particle_flux_m2_s` | (surface, species) | (3, 1) | m$^{-2}$ s$^{-1}$ |
| `heat_flux_W_m2` | (surface, species) | (3, 1) | W m$^{-2}$ |
| `parallel_current_A_T_m2` | (surface) | (3,) | A T m$^{-2}$ |
| `primal_residual`, `primal_rhs_norm` | (surface) | (3,) | absolute $\ell_2$ norms |
| `solver_iterations`, `solve_time_s` | (surface) | (3,) | count, s |

`result.particle_flux_m2_s` is the same array as
`result.arrays["particle_flux_m2_s"]`. Arrays are read-only copies.

```python
result.print_summary()          # route, residual, wall time, array names
cert = result.certificate()     # compact provenance record (a dict)
result.save("tokamak.nc")       # NetCDF4, result schema 1
result.plot("tokamak.png")      # flux, heat flux, current, E_r against psi_N
```

`print_summary()` prints lines of this form (the residual and time depend on
the machine; the values below are illustrative):

```text
DKX result: tokamak (92e38763614d)
workflow: profile
converged: yes; residual: 2.8e-15
solver: block_tridiagonal_truncated; total time: 6.3 s
arrays: charge_e, density_m3, electric_field_kV_m, heat_flux_W_m2, ...
```

The first run of a session includes JAX compilation; later runs of the same
shapes reuse the persistent compilation cache.

## From a case file

The same kind of case is checked in as `examples/01_tokamak_profile/case.toml`
(its surfaces are 0.09, 0.16 and 0.25). From the repository root:

```console
dkx validate examples/01_tokamak_profile/case.toml
dkx run examples/01_tokamak_profile/case.toml --out result.nc
dkx inspect result.nc
dkx plot result.nc
```

- `dkx validate` parses the file, runs the executor's preflight without
  solving, and prints the `case_id`. It exits 2 with the field path, the
  supplied value, the expected form and a correction when anything is wrong.
- `dkx run` solves the case and prints a table with the workflow, the
  convergence flag, the true residual, the solver route, the wall time and the
  output path, followed by `print_summary()`. Without `--out` it writes the
  file named by `[output].file`, resolved beside the case file.
- `dkx inspect` lists the arrays of a saved result with their shapes and dtypes,
  without recomputing anything.
- `dkx plot` writes `result.png` next to the input: particle flux, heat flux,
  $\langle j_\parallel B\rangle$ and $E_r$ against $\psi_N$.

The same file loads in Python:

```python
import dkx

case = dkx.Case.from_file("examples/01_tokamak_profile/case.toml")
result = dkx.run(case, out="result.nc")
again = dkx.Result.load("result.nc")
```

## Is the answer resolved?

A result that solved to a small residual can still be far from the continuum
answer. `dkx converge` measures that:

```console
dkx converge examples/01_tokamak_profile/case.toml
```

It solves the case at its stated resolution, once with each of `theta`,
`zeta`, `pitch` and `speed` refined by a factor 1.5, and once with all of them
refined together. It reports the worst relative change of the particle flux,
heat flux and parallel current for each refinement, and exits zero only when
every change is below the tolerance (default 0.02) **and** every solve passed
its original-equation residual check. For an axisymmetric case (`zeta = 1`)
the `zeta` axis is skipped.

On this file it fails, as intended. The resolution comments in
`examples/01_tokamak_profile/case.toml` record what it reports: the outputs
move by more than 100% under refinement, and `theta` 9 → 14 alone moves the
particle flux by 1187%. They also record why the joint refinement matters: at
`pitch = 8` the `theta` axis looks settled to 0.2%, while at `pitch = 40` the
same `theta` refinement moves the outputs by 74%. Refining one axis at a time
would have called this case converged in `theta`.

{doc}`../user_guide/convergence` explains the report and how to choose a
resolution. {doc}`../user_guide/inputs` lists every field of the case file.
