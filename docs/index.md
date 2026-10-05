# DKX

**Differentiable neoclassical transport for stellarators and tokamaks, in JAX.**

DKX solves the radially local, linearized drift-kinetic equation of SFINCS v3 on
toroidal flux surfaces. From a magnetic geometry and plasma profiles it computes
particle and heat fluxes, parallel flows, the bootstrap current, transport
matrices, monoenergetic coefficients and the ambipolar radial electric field, on
CPU or GPU. Every output can be differentiated with `jax.grad`, and SFINCS v3
input decks run unchanged.

## Quickstart

A calculation is one *case*: a versioned mapping with a geometry, species
profiles, a physics model, an electric field, a phase-space resolution and a
solver choice ({doc}`user_guide/inputs` lists every field). This one is an
analytic tokamak at a teaching resolution; it runs in seconds and is not
converged.

```python
import dkx

case = dkx.Case.from_mapping({
    "name": "tokamak", "run": {"workflow": "profile", "progress": False},
    "geometry": {"format": "analytic", "file": "tokamak", "surfaces": [0.16, 0.25, 0.36]},
    "species": [{"name": "deuterium", "charge": 1, "mass_amu": 2.014,
                 "density_m3": [8.0e19, 7.0e19, 6.0e19], "temperature_keV": [1.0, 0.8, 0.6]}],
    "physics": {"model": "full_local", "collisions": "pitch_angle_scattering", "magnetic_drifts": "dkes", "phi1": "off"},
    "electric_field": {"mode": "prescribed", "value_kV_m": 0.0},
    "resolution": {"theta": 9, "zeta": 1, "pitch": 8, "speed": 4}, "solver": {"method": "auto", "relative_tolerance": 1e-8},
})
result = dkx.run(case)
print("solver route:", result.metadata["solver_route"])
print(float(result.arrays["particle_flux_m2_s"][1, 0]))   # particle flux, m^-2 s^-1
```

The same case lives in `examples/01_tokamak_profile/case.toml`. Before quoting a
number, run `dkx converge examples/01_tokamak_profile/case.toml`: it refines every
phase-space axis and exits zero only when the observables stop moving and every
returned state satisfies the kinetic equation ({doc}`user_guide/convergence`).

## What DKX computes

| Quantity | Result array or SFINCS variable | Page |
| --- | --- | --- |
| particle and heat fluxes per species | `particle_flux_m2_s`, `heat_flux_W_m2` | {doc}`user_guide/outputs` |
| bootstrap current and parallel flow | `parallel_current_A_T_m2`, `FSABjHat` | {doc}`physics/drives_and_rhs_modes` |
| ambipolar $E_r$ and every root branch | `ambipolar_root_kV_m`, `ambipolar_root_type` | {doc}`physics/electric_field` |
| thermal transport matrix (RHSMode 2) | `transportMatrix` | {doc}`physics/drives_and_rhs_modes` |
| monoenergetic $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ (RHSMode 3) | `transportMatrix` | {doc}`physics/reduced_models` |
| in-surface potential $\Phi_1$ and impurity fluxes | `Phi1Hat`, per-species fluxes | {doc}`physics/phi1_and_impurities` |
| derivatives of any of the above | `jax.grad`, `jax.jacrev` | {doc}`numerics/differentiation` |

## How the documentation is organized

| Section | Read it to |
| --- | --- |
| {doc}`getting_started/installation`, {doc}`getting_started/first_run` | install DKX and run, read and check a first case |
| {doc}`user_guide/index` | look up every input field, output array, CLI command and Python entry point |
| {doc}`tutorials/index` | follow complete calculations: profiles, $E_r$ roots, transport matrices, VMEC stellarators, gradients, optimization, SFINCS decks |
| {doc}`how_it_works` | see the pipeline from a case to a result, module by module |
| {doc}`physics/index` | read the equations, collision operators, drives, electric field, $\Phi_1$ and geometry models |
| {doc}`numerics/index` | understand the discretization, the solver routes, preconditioners, factor reuse and adjoints |
| {doc}`benchmarks/index` | check the analytic limits, the SFINCS, MONKES and YANCC comparisons and the performance record |
| {doc}`examples/index` | browse the example ladder |
| {doc}`features`, {doc}`design_decisions` | see what the code can do and why it is built the way it is |
| {doc}`api`, {doc}`changelog` | the reference API and the release history |

## Evidence at a glance

```{figure} _static/figures/paper/dkx_fortran_suite_benchmark_summary.png
:alt: DKX against SFINCS v3 on the upstream example decks

Upstream decks whose SFINCS v3 reference runtime clears a `10 s`
reference-runtime-window, so process-launch and JIT-amortization noise does not
dominate the bars ({doc}`benchmarks/sfincs`).
```

The solver route decides the outcome: the structured direct route is faster than
SFINCS v3 on 9 of 9 decks, the recycled Krylov route on 7 of 23, and memory is the
weak axis. On the 744,610-unknown HSX pitch-angle-scattering case the warm solve
takes 27.2 s against 229.5 s for SFINCS on two ranks
({doc}`benchmarks/performance`). Every number in these pages names the test,
script or deck that produces it.

```{toctree}
:maxdepth: 2
:caption: Getting started
:hidden:

getting_started/installation
getting_started/first_run
```

```{toctree}
:maxdepth: 2
:caption: User guide
:hidden:

user_guide/index
tutorials/index
```

```{toctree}
:maxdepth: 2
:caption: How DKX works
:hidden:

how_it_works
physics/index
numerics/index
```

```{toctree}
:maxdepth: 2
:caption: Evidence and examples
:hidden:

benchmarks/index
examples/index
features
design_decisions
```

```{toctree}
:maxdepth: 1
:caption: Reference
:hidden:

api
changelog
references
contributing
```
