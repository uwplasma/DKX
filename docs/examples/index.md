# Examples

The examples are a ladder of nine numbered rungs. Each rung is one directory
with a `run.py` and, where the case can be written as a file, a `case.toml`
that the `dkx` command line accepts. Each rung changes one thing about the one
before it, so running them in order is the shortest route through the code.

Every script is flat and module-level: imports, then the editable parameters
above an `end of parameters` line, then geometry and species, physics and
numerics, run, a printed summary with the result certificate, a saved result,
and a plot. There is no argument parsing; to change a run, edit a constant and
rerun. Outputs are written to `examples/output/<rung>/`.

Run a rung from the repository root:

```console
python examples/01_tokamak_profile/run.py
```

Each rung ships at a resolution that runs in seconds, so CI runs the same code a
reader does. That resolution buys speed, not accuracy: rung 06 measures how far
the small cases are from converged, and the answer is "far". Refine before
quoting a number ({doc}`../user_guide/convergence`).

## The ladder

| Rung | What it teaches | Runtime on a laptop CPU |
| --- | --- | --- |
| `examples/01_tokamak_profile` | the whole native loop: `dkx.Case`, `dkx.run`, SI moments, certificate, NetCDF, plot | ~4 s |
| `examples/02_vmec_stellarator` | the same solve on a VMEC `wout`: one field of the case changes | ~5 s |
| `examples/03_boozer_stellarator` | Boozer `.bc` geometry, and the route the operator's structure picks | ~4 s |
| `examples/04_monoenergetic_scan` | $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ against collisionality | ~5 s |
| `examples/05_ambipolar_profile` | solving for $E_r$ from ambipolarity: every root, classified | seconds; depends on compilation and root branches |
| `examples/06_convergence_certificate` | refining every phase-space axis; a small residual is not a converged answer | ~12 s |
| `examples/07_gradients` | `jax.grad` through the solve, checked against central differences | tens of seconds |
| `examples/08_vmex_optimization` | a shape derivative: differentiate the solve with respect to the $\lvert B\rvert$ spectrum and descend | compilation-dominated, ~12 s |
| `examples/09_phi1_and_impurities` | multi-species impurity transport with and without $\Phi_1$ | ~12 s |

The runtimes are the ones each script states in its docstring.

### 01 Tokamak profile

`examples/01_tokamak_profile`. A concentric circular tokamak, one deuterium
species, pitch-angle scattering, three flux surfaces, no radial electric field.
The script builds the case in Python, checks that its case ID equals the one of
`case.toml` beside it, runs it, prints particle flux, heat flux and
$\langle j\cdot B\rangle$ per surface, prints the certificate (converged, solver
route, residual, version, device, precision), saves NetCDF and plots.

```console
python examples/01_tokamak_profile/run.py
dkx run examples/01_tokamak_profile/case.toml --out examples/output/01_tokamak_profile/result.nc
dkx plot examples/output/01_tokamak_profile/result.nc
```

Tutorial: {doc}`../tutorials/first_profile`.

### 02 VMEC stellarator

`examples/02_vmec_stellarator`. Rung 01 with `geometry.format = "vmec"` and a
`wout` file (`tests/ref/wout_up_down_asymmetric_tokamak.nc`). Species, physics
and solver are unchanged: the equilibrium is an input, not a different code
path. `zeta` becomes 9 because a VMEC field is resolved on a toroidal grid.

```console
python examples/02_vmec_stellarator/run.py
dkx inspect examples/output/02_vmec_stellarator/result.nc
```

Tutorial: {doc}`../tutorials/stellarator_from_vmec`.

### 03 Boozer stellarator

`examples/03_boozer_stellarator`. `geometry.format = "boozer"` with a `.bc` file
(`tests/ref/nonStelSym_tiny_geometryScheme12.bc`, non-stellarator-symmetric).
DKX detects the symmetric or asymmetric column convention itself, so the case
never names a geometry-scheme number.

```console
python examples/03_boozer_stellarator/run.py
```

### 04 Monoenergetic scan

`examples/04_monoenergetic_scan`. $D_{11}^*$, $D_{31}^*$, $D_{33}^*$ against
$\nu'$ at $E^* = 0$ and $0.1$ on the circular tokamak deck `input.namelist`,
through `dkx.run_monoenergetic_database`. The script asserts two properties of
the operator: $D_{11}^*$ rises monotonically in $\nu'$ at $E^* = 0$, and
$D_{33}^*$ approaches its collisional value 1 from below. There is no
`case.toml`: the native executor does not implement the monoenergetic workflow,
so this rung uses the SFINCS-deck entry point.

```console
python examples/04_monoenergetic_scan/run.py
```

```{figure} ../_static/figures/paper_benchmarks/monoenergetic_icnts_w7x.png
:alt: D11 and D31 against collisionality on W7-X at several E*, with SFINCS points.
:width: 80%

The same kind of scan at benchmark resolution on W7-X
({doc}`../benchmarks/cross_code`).
```

Tutorial: {doc}`../tutorials/transport_matrix`.

### 05 Ambipolar profile

`examples/05_ambipolar_profile`. Analytic W7-X, deuterium and electrons, full
linearized Fokker–Planck collisions, two surfaces. Setting
`run.workflow = "ambipolar_profile"` and `electric_field.mode = "ambipolar"` is
the whole change: DKX brackets the root on every surface, keeps every root it
finds, classifies each as ion, electron or unstable, and records which one it
selected and why. `w7x_case.toml` beside it is the production-scale case.

```console
python examples/05_ambipolar_profile/run.py
dkx roots examples/output/05_ambipolar_profile/result.nc
dkx validate examples/05_ambipolar_profile/w7x_case.toml
```

```{figure} ../_static/figures/docs/ambipolar_er_roots.png
:alt: Radial current against radial electric field with the ambipolar roots marked.
:width: 70%

Radial current against $E_r$ with its ambipolar roots.
```

Tutorial: {doc}`../tutorials/ambipolar_er`.

### 06 Convergence certificate

`examples/06_convergence_certificate`. Rung 01 at a deliberately coarse
resolution, refined along each axis and then all axes together, with the
movement of each observable reported. The joint refinement is the one that
counts, because the axes couple.

```console
python examples/06_convergence_certificate/run.py
dkx converge examples/06_convergence_certificate/case.toml
```

### 07 Gradients

`examples/07_gradients`. Temperature sensitivity of the pitch-angle-scattering
bootstrap current in a circular tokamak, with the collision coefficients rebuilt
in JAX for each temperature. `jax.grad` through the solve is compared with
central differences at three step sizes and must agree to 1e-5. Python only:
a derivative is not a case.

```console
python examples/07_gradients/run.py
```

Tutorial: {doc}`../tutorials/gradients`.

### 08 VMEX optimization

`examples/08_vmex_optimization`. A three-helicity analytic surface ($N = 5$
field periods). The traced chain runs from a Boozer $\lvert B\rvert$ Fourier
amplitude through `FluxSurfaceGeometry.from_fourier`, the operator and the solve
to the radial particle flux; gradient descent then reduces it. It does not
import `vmex`, so the kinetic gradient is reproducible with or without the
optional backends.

```console
python examples/08_vmex_optimization/run.py
```

Tutorial: {doc}`../tutorials/vmex_optimization`.

### 09 Phi1 and impurities

`examples/09_phi1_and_impurities`. Circular tokamak at $r/a = 0.3$, hydrogen,
fully ionized carbon and electrons, full linearized Fokker–Planck collisions,
run twice at identical resolution: with $\Phi_1$ off and on. With $\Phi_1$ on,
quasineutrality becomes an extra block of rows and the solve is a
Newton–Krylov iteration. Both legs use the SFINCS-parameter route, because the
native executor implements only `phi1 = "off"`.

```console
python examples/09_phi1_and_impurities/run.py
```

```{figure} ../_static/figures/paper_benchmarks/impurity_transport.png
:alt: C6+ impurity flux against collisionality on W7-X.
:width: 80%

Impurity flux at benchmark scale on W7-X (`tools/paper_benchmarks/impurity_transport.py`).
```

## Topic folders

These folders hold longer or more specialized scripts. They run and are
exercised by tests, but the teaching path is the ladder above.

| Folder | Contents |
| --- | --- |
| `examples/getting_started` | grids and geometry objects, writing `sfincsOutput.h5` from Python and from the CLI, analytic tokamak and VMEC decks, a typed `SfincsInput` built in Python, a two-species W7-X run |
| `examples/transport` | RHSMode=2 and RHSMode=3 transport matrices and transport coefficients on several geometry schemes |
| `examples/autodiff` | matrix-free residual and JVP, implicit differentiation through a Krylov solve, geometry gradients, `gradients_tour.py`, and the optional `vmex_to_boozer_sfincs_pipeline.py` |
| `examples/optimization` | differentiable-kinetic optimizations (`optimize_QA_bootstrap.py`, `optimize_QH_bootstrap.py`, `optimize_electron_root.py`, `optimize_impurity_screening.py`), VMEX QA/QH/QI bootstrap optimizations with a kinetic row, and `vmex_workflow_status.py` |
| `examples/vmex_finite_beta` | a finite-beta QA equilibrium from `vmex` handed to DKX as a `wout`, with ambipolar $E_r$ and bootstrap-current profiles and comparisons against the Redl formula |
| `examples/tutorials` | Jupyter notebooks covering the CLI, transport and autodiff, bootstrap current and optimization, geometry and performance |
| `examples/sfincs_examples` | the vendored upstream SFINCS v3 decks, used for parity and benchmark audits, and the upstream `utils` scan and plot scripts |
| `examples/data` | small shared input files |

```{figure} ../_static/figures/readme/QA_optimization_bootstrap_dkx.png
:alt: Bootstrap current profiles and objective history of the VMEX QA optimization with a DKX kinetic row.
:width: 85%

`examples/optimization/QA_optimization_bootstrap_dkx.py`: VMEX's QA
bootstrap-current optimization with the Redl row replaced by the DKX kinetic
one. At the committed parameters one run took 30 minutes and 4.6 GB on four
laptop CPU threads; the objective fell from 1.78 to 0.0061
(`examples/optimization/README.md`).
```


```{figure} ../_static/figures/finite_beta_vmex_sfincs_bootstrap_er.png
:alt: Ambipolar radial electric field and bootstrap current profiles for the finite-beta QA equilibrium.
:width: 85%

`examples/vmex_finite_beta/finite_beta_vmec_to_sfincs.py`: ambipolar $E_r$ and
bootstrap-current profiles on the finite-beta QA equilibrium. This is a primal
workflow: nothing in it is differentiated.
```

## Finding a workflow

`examples/workflow_catalog.json` lists every workflow with its entry point,
command, runtime budget and whether it needs a local SFINCS Fortran executable.

```console
python examples/list_workflows.py --list-topics
python examples/list_workflows.py --search "VMEC geometry"
```

Parity checks, benchmarks and figure generators live in `tools/`, not in
`examples/`; they regenerate the checked artifacts cited on the
{doc}`../benchmarks/index` pages.
