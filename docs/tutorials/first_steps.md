# 1. First steps: install, run, read, check

This part of the track takes you from an empty environment to a checked result.
It uses one case throughout: an analytic tokamak with one deuterium species, no
radial electric field and pitch-angle-scattering collisions. Every command was
run as shown; the outputs below are copied from those runs (four CPU threads of
a shared Xeon workstation, `float64`). Your wall times will differ, the physics
numbers should not.

Scripts: `examples/tutorials/01_first_run.py`, `02_cli_case.py` and
`02_cli_case.toml`, `03_sfincs_namelist.py` and `03_sfincs_namelist.namelist`,
`13_convergence.py` and `13_convergence.toml`.

## What DKX solves

DKX is a radially local, steady, linearized drift-kinetic solver: the model of
SFINCS v3 (Landreman et al. 2014). On one flux surface it splits the
distribution of every species $s$ into a Maxwellian and a small correction,
$f_s = f_{sM} + f_{s1}$, and solves

$$
\underbrace{v_\parallel \mathbf{b}\cdot\nabla f_{s1}}_{\text{streaming}}
+ \underbrace{\mathbf{v}_E\cdot\nabla f_{s1}}_{E\times B\ \text{precession}}
- \underbrace{\sum_b C_{sb}[f_{s1}, f_{b1}]}_{\text{collisions}}
= -\,\underbrace{\mathbf{v}_{ms}\cdot\nabla\psi\,\frac{\partial f_{sM}}{\partial\psi}}_{\text{radial drift across the gradients}}
+ \underbrace{\frac{Z_s e\,v_\parallel B\,\langle E_\parallel B\rangle}{T_s\langle B^2\rangle} f_{sM}}_{\text{inductive field}} .
$$

The right-hand side is the drive. With $x_s = v/v_{\mathrm{th},s}$ and
$v_{\mathrm{th},s} = \sqrt{2T_s/m_s}$ the radial derivative of the Maxwellian is

$$
\frac{\partial f_{sM}}{\partial\psi}
= f_{sM}\left[\frac{1}{n_s}\frac{dn_s}{d\psi} + \frac{Z_s e}{T_s}\frac{d\Phi_0}{d\psi}
+ \left(x_s^2 - \frac{3}{2}\right)\frac{1}{T_s}\frac{dT_s}{d\psi}\right],
$$

so density gradients, temperature gradients and the radial electric field
$E_r = -d\Phi_0/dr$ are the thermodynamic forces. The equation is linear in
$f_{s1}$, and the outputs are its velocity moments, flux-surface averaged:

$$
\Gamma_s = \Bigl\langle \int d^3v\, f_{s1}\,\mathbf{v}_{ms}\cdot\nabla\psi \Bigr\rangle,
\qquad
Q_s = \Bigl\langle \int d^3v\, \tfrac{1}{2}m_s v^2 f_{s1}\,\mathbf{v}_{ms}\cdot\nabla\psi \Bigr\rangle,
\qquad
\langle j_\parallel B\rangle = \sum_s Z_s e \Bigl\langle B\int d^3v\, v_\parallel f_{s1}\Bigr\rangle .
$$

$\langle\cdot\rangle$ is the flux-surface average. The last one is the bootstrap
current when the inductive field is zero. Full term-by-term detail, including
the magnetic-drift and $E_r$ terms of the full trajectory models, is in
{doc}`../physics/drift_kinetic_equation`.

**Normalizations.** Inside the solver every quantity is a ratio to a fixed
reference set: $\bar n = 10^{20}\,$m$^{-3}$, $\bar T = 1\,$keV, $\bar m$ the
proton mass, $\bar B = 1\,$T, $\bar R = 1\,$m, $\bar v = \sqrt{2\bar T/\bar m}$.
Three dimensionless numbers carry the physics scales: the normalized gyroradius
$\Delta = \bar m\bar v/(e\bar B\bar R) = 4.57\times10^{-3}$, the normalized
collision frequency $\nu_n = \bar\nu\bar R/\bar v = 8.33\times10^{-3}$ at
$\ln\Lambda = 17$, and $\alpha = e\bar\Phi/\bar T = 1$. A native case is written
in SI units (m$^{-3}$, keV, amu, kV/m) and DKX converts in one place
(`dkx.units`); a SFINCS namelist is written directly in the hatted variables.
The full list is {doc}`../physics/normalizations`.

## Step 1: install

```console
pip install dkx
dkx doctor
```

`dkx doctor` prints one row per check (Python, DKX, SOLVAX, JAX, `float64`,
devices) and exits non-zero if a blocking check fails. Add
`pip install -U "jax[cuda12]"` for an NVIDIA GPU. From a clone:
`pip install -e .`. Details: {doc}`../getting_started/installation`.

## Step 2: a first run from Python

A calculation is one *case*: a mapping with a geometry, species profiles, a
physics model, an electric field, a resolution and a solver choice.

```python
import dkx

case = dkx.Case.from_mapping({
    "name": "tokamak", "run": {"workflow": "profile", "progress": False},
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
a = result.arrays
for i, psi in enumerate(a["surface"]):
    print(f"psi_N={float(psi):.2f}  Gamma={float(a['particle_flux_m2_s'][i, 0]):+.3e}"
          f"  Q={float(a['heat_flux_W_m2'][i, 0]):+.3e}"
          f"  <j.B>={float(a['parallel_current_A_T_m2'][i]):+.3e}")
result.print_summary()
```

Output:

```text
solver route: block_tridiagonal_truncated
psi_N=0.16  Gamma=+1.561e+20  Q=+7.625e+04  <j.B>=-2.124e+05
psi_N=0.25  Gamma=+9.041e+19  Q=+3.788e+04  <j.B>=-1.428e+05
psi_N=0.36  Gamma=+5.889e+19  Q=+2.011e+04  <j.B>=-9.570e+04
DKX result: tokamak (92e38763614d)
workflow: profile
converged: yes; residual: 3.895215628111529e-15
solver: block_tridiagonal_truncated; total time: 2.66 s
arrays: charge_e, density_m3, electric_field_kV_m, heat_flux_W_m2, ...
```

What happened:

- `surfaces` are values of the normalized toroidal flux $\psi_N$. Profiles give
  one value per surface, and DKX differentiates them across the surfaces to get
  $dn/d\psi$ and $dT/d\psi$, so a case needs at least two surfaces.
- On each surface DKX builds the operator, discretizes $\theta$ and $\zeta$ on a
  grid, the pitch $\xi = v_\parallel/v$ in Legendre modes $P_L(\xi)$ and the
  speed $x$ on a few nodes ({doc}`../numerics/discretization`).
- `method="auto"` chose the route from the operator's structure.
  Pitch-angle scattering with DKES trajectories couples each Legendre mode
  $L$ only to $L\pm1$ and leaves every (species, speed) pair independent, so the
  system is block tridiagonal and is solved by exact block elimination:
  `block_tridiagonal_truncated` keeps only the lowest Legendre blocks that the
  moments need ({doc}`../numerics/solver_routes`).
- The particle flux is outward (positive), the heat flux too; the bootstrap
  current $\langle j\cdot B\rangle$ is in A T m$^{-2}$. The residual
  $4\times10^{-15}$ is that of the original equation: the linear system was
  solved exactly.

Arrays indexed `[surface, species]` are per species; the parallel current is
summed over species. `result.save("tokamak.nc")` writes NetCDF,
`result.plot("tokamak.png")` draws flux, heat flux, current and $E_r$ against
$\psi_N$, and `result.certificate()` is the provenance record:

```text
{'converged': True, 'solver_route': 'block_tridiagonal_truncated',
 'residual_norm': 3.895215628111529e-15, 'device': 'cpu:cpu', 'precision': 'float64'}
```

Run the full script with `python examples/tutorials/01_first_run.py`.

```{note}
The first run of a session includes JAX compilation. Later runs of the same
shapes reuse the persistent compilation cache. On a CPU, DKX gives XLA
`min(8, cores)` threads and the host BLAS `min(4, cores)` threads when neither
`DKX_CORES` nor `NPROC` is set; `dkx --cores N` or `DKX_CORES=N` overrides it
before JAX starts ({doc}`../user_guide/scans_and_parallelism`).
```

## Step 3: the same case from a file and the command line

The case is a TOML file with one table per concern. `dkx template` prints a
commented template with every field and its default; edit it down.

```console
dkx template > my_case.toml
dkx validate examples/tutorials/02_cli_case.toml
dkx run examples/tutorials/02_cli_case.toml --out result.nc
dkx inspect result.nc
dkx plot result.nc
```

`dkx validate` parses the file, runs the preflight without solving and prints
the case ID, a SHA-256 of the normalized content (the same physics always has
the same ID):

```text
valid DKX case: analytic_tokamak_profile
case_id: 2ca9e2e8db568238b5230334709e5141b9ca13149f63d0caeae86632f3a86343
workflow: profile; surfaces: 3; species: 1
```

`dkx run` prints a summary table:

```text
│ workflow      │ profile                     │
│ surfaces      │ 3                           │
│ species       │ 1                           │
│ converged     │ yes                         │
│ true residual │ 4.090e-15                   │
│ solver route  │ block_tridiagonal_truncated │
│ wall time     │ 3.55 s                      │
```

and `dkx inspect` lists the stored arrays and shapes without recomputing
(`particle_flux_m2_s (3, 1)`, `parallel_current_A_T_m2 (3,)`, ...). The file in
Python is `dkx.Case.from_file("examples/tutorials/02_cli_case.toml")`.
Relative paths inside a case resolve beside the case file. Case files that still
say `schema = 1` load; the key is ignored. Every field
is in {doc}`../user_guide/inputs`; every command in {doc}`../user_guide/cli`.

## Step 4: a SFINCS v3 namelist

DKX reads SFINCS v3 `input.namelist` decks unchanged and writes
`sfincsOutput.h5` with SFINCS's variable names, so existing decks and
post-processing scripts keep working. A deck is written in the hatted
variables: `nHats`, `THats`, `dNHatdrHats`, `dTHatdrHats`, `Delta`, `nu_n`,
`Er` and the grid `Ntheta`, `Nzeta`, `Nxi`, `Nx`.

```console
dkx examples/tutorials/03_sfincs_namelist.namelist
dkx sfincs write-output --input examples/tutorials/03_sfincs_namelist.namelist --out sfincsOutput.h5
```

With no subcommand, `dkx` treats its argument as a deck and prints what the
Fortran code prints. The end of the log of the bundled `geometryScheme = 4`
(W7-X-like three-harmonic field) deck:

```text
    particleFlux_vm_psiHat     -9.7065631036157259E-021
    heatFlux_vm_psiHat         -2.3928340768017112E-020
 FSABjHat (bootstrap current):   -2.3421933067597170E-005
 Goodbye!
 wrote output -> sfincsOutput.h5
```

`FSABjHat` is $\langle j_\parallel B\rangle$ in SFINCS units. The deck sets
`nu_n = 0`, a collisionless deck: a smoke test of the I/O, not a physics case.

To leave the namelist world, convert the deck to a native case. The conversion
is a translation of units, not a renaming, and it fails, naming the key, on
anything the case format cannot carry:

```console
dkx convert input.namelist case.toml
dkx compare sfincsOutput_fortran.h5 sfincsOutput.h5 --rtol 1e-9
```

`dkx compare` reads DKX NetCDF and SFINCS HDF5 files and exits non-zero when any
array differs beyond the tolerance. Two differences from Fortran SFINCS matter
when comparing: DKX solves without PETSc (it picks its own route and reports the
true residual), and released SFINCS drops matrix entries with magnitude at or
below `1e-12`, which on cold-ion, hot-electron decks removes part of the
ion–electron collision coupling ({doc}`../benchmarks/sfincs`). The whole
namelist reference, including the `dkx sfincs` subcommands for transport
matrices, monoenergetic databases and $E_r$ scans, is
{doc}`../user_guide/sfincs_namelist`.

Upstream SFINCS plotting scripts read DKX output too:
`dkx sfincs postprocess-upstream --case-dir DIR --util sfincsScanPlot_1 -- pdf`.

```{figure} ../_static/figures/utils/sfincsScanPlot_1.png
:alt: Upstream sfincsScanPlot_1 output generated from DKX sfincsOutput.h5 files.
:width: 75%

An upstream `sfincsScanPlot_1` figure drawn from DKX output.
```

## Step 5: is the answer resolved?

A small residual says the discrete system was solved. It says nothing about
whether the grid is fine enough. `dkx converge` measures that:

```console
dkx converge examples/tutorials/13_convergence.toml
python examples/tutorials/13_convergence.py
```

It solves the case at its stated resolution, again with each of `theta`, `zeta`,
`pitch` and `speed` refined by 1.5×, and once with all of them refined
together. It exits zero only when the worst relative change of particle flux,
heat flux and current stays below the tolerance (default 0.02) **and** every
solve passed its residual check.

On the tutorial grid it fails, as intended. Refining `theta` from 9 to 14 alone
moves the particle flux by 1187%. At `pitch = 8` the `theta` axis looks settled
to 0.2%, while at `pitch = 40` the same refinement moves the outputs by 74%:
the axes couple, so the joint refinement is the one that decides. The tutorial
grids are sized to run in seconds, not to be quoted. How to choose a production
resolution: {doc}`../user_guide/convergence`.

## Next

{doc}`building_a_case`: real geometries, collision operators and more species.
