# 1. Install and first run

This page takes you from an empty environment to a first result, in Python and
from the command line. Every output below was printed by the script named under
it (four CPU threads, `float64`); wall times differ between machines, the
numbers do not.

## What DKX computes

DKX solves the radially local, steady drift-kinetic equation of SFINCS v3 on
magnetic flux surfaces. For each species $s$ the distribution is a Maxwellian
plus a small correction, $f_s = f_{sM} + f_{s1}$, and $f_{s1}$ obeys

$$
v_\parallel \mathbf{b}\cdot\nabla f_{s1} + \mathbf{v}_E\cdot\nabla f_{s1}
- \sum_b C_{sb}[f_{s1}, f_{b1}] = -\,\mathbf{v}_{ms}\cdot\nabla\psi\,\frac{\partial f_{sM}}{\partial\psi}.
$$

Streaming, $E\times B$ drift and collisions on the left; on the right the
radial magnetic drift acting on the density, temperature and potential
gradients. The equation is linear in $f_{s1}$. Its velocity moments are the
outputs: the particle flux $\Gamma_s$, the heat flux $Q_s$ and the
parallel current $\langle j_\parallel B\rangle$ ({doc}`../physics/drift_kinetic_equation`).

## Install

```console
pip install dkx
dkx doctor
```

`dkx doctor` prints one row per check (Python, DKX, SOLVAX, JAX, `float64`,
devices) and exits non-zero if a blocking check fails. For an NVIDIA GPU add
`pip install -U "jax[cuda12]"`. From a clone: `pip install -e .`
({doc}`../getting_started/installation`).

## What a case is

A calculation is one **case**: geometry, species profiles, a physics model, an
electric field, a resolution and a solver choice. The same case can be a Python
mapping or a TOML file; both give the same case ID, a SHA-256 of the
normalized content.

## Run it from Python

`examples/tutorials/01_first_run.py`:

```python
import dkx

case = dkx.Case.from_mapping({
    "name": "first_run",
    "run": {"workflow": "profile", "progress": True},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.09, 0.16, 0.25]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 5.8e19], "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"collisions": "pitch_angle_scattering"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "solver": {"method": "auto"},
    "resolution": {"theta": 13, "zeta": 1, "pitch": 16, "speed": 5},
})
result = dkx.run(case)
result.save("result.nc")
result.plot("result.png")
```

The printed summary (22 s including JAX compilation):

```text
  psi_N=0.09  Gamma=+2.230e+20 m^-2 s^-1  Q=+1.309e+05 W m^-2  <j.B>=-2.695e+05 A T m^-2
  psi_N=0.16  Gamma=+1.481e+20 m^-2 s^-1  Q=+6.540e+04 W m^-2  <j.B>=-1.826e+05 A T m^-2
  psi_N=0.25  Gamma=+1.043e+20 m^-2 s^-1  Q=+3.359e+04 W m^-2  <j.B>=-1.222e+05 A T m^-2
  converged: True  route: block_tridiagonal_truncated  residual: 3.0e-14
```

The inputs, one by one:

- `surfaces` are values of the normalized toroidal flux $\psi_N$. Profiles give
  one value per surface and DKX differentiates them across the surfaces to get
  $dn/d\psi$ and $dT/d\psi$, so a case needs at least two surfaces.
- `resolution` is the grid: `theta`, `zeta` points in the two angles, `pitch`
  Legendre modes in $\xi = v_\parallel/v$ and `speed` nodes in $x = v/v_{th}$.
  These sizes are for speed, not accuracy: page 5 measures how far they are
  from converged.
- `solver.method = "auto"` picks the linear solver from the operator's
  structure; page 4 explains the choices.

## The same case from the command line

`examples/tutorials/02_cli_case.toml` holds the same physics with
`theta = 9, pitch = 8, speed = 4` (a smoke grid):

```toml
name = "analytic_tokamak_profile"

[run]
workflow = "profile"

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
collisions = "pitch_angle_scattering"

[electric_field]
mode = "prescribed"
value_kV_m = 0.0

[resolution]
theta = 9
zeta = 1
pitch = 8
speed = 4
```

(the file also spells out the defaults.) `dkx template > my_case.toml` prints
a commented template with every field.

```console
dkx validate examples/tutorials/02_cli_case.toml
dkx run examples/tutorials/02_cli_case.toml --out result.nc
dkx inspect result.nc
dkx plot result.nc
```

```text
valid DKX case: analytic_tokamak_profile
case_id: 2ca9e2e8db568238b5230334709e5141b9ca13149f63d0caeae86632f3a86343
workflow: profile; surfaces: 3; species: 1
```

`dkx run` ends with a summary table:

```text
│ workflow      │ profile                     │
│ surfaces      │ 3                           │
│ species       │ 1                           │
│ converged     │ yes                         │
│ true residual │ 4.274e-15                   │
│ solver route  │ block_tridiagonal_truncated │
│ wall time     │ 3.92 s                      │
```

`dkx inspect` lists the stored arrays and shapes without recomputing. In
Python the file loads with `dkx.Case.from_file("...toml")`; relative paths
inside a case resolve beside the case file.

## The first plot

```{figure} ../_static/figures/tutorials/first_run.png
:alt: Particle flux, heat flux, parallel current and radial electric field against normalized toroidal flux.
:width: 60%

`result.plot`: flux, heat flux, $\langle j\cdot B\rangle$ and $E_r$ against $\psi_N$
for the case above.
```

Fluxes are positive (outward, down the gradients) and fall with $\psi_N$ as the
gradients and the density fall; $E_r$ is zero because the case prescribes it.
The tokamak bootstrap current is negative in this sign convention.

## Next

{doc}`02_reading_results`: what each output means, its units, and what
"converged" does and does not say.
