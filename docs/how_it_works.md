# How DKX works

A DKX run takes one case, builds one linear operator per flux surface, solves
it, and reduces the solution to transport moments. This page follows that
pipeline stage by stage and maps each stage to the module that owns it. The
physics behind each stage is in {doc}`physics/index`; the numerical methods are
in {doc}`numerics/index`.

## The pipeline

```text
case.toml / Case            input.namelist / SfincsInput
      |                              |
  config.Case                 inputs.load_sfincs_input
      |                              |
  execution.run_case          run.run_from_namelist
      \______________  _____________/
                     \/
   geometry        magnetic_geometry (+ vmec_ascii, workflows.geometry_adapters)
   grids           phase_space, xgrid
   species         species, constants, units
   collisions      collisions
   operator        drift_kinetic.KineticOperator
   route           solve.solve  (auto policy)
   solve           structured direct | recycled Krylov | sparse direct
   outer loops     er (ambipolar Er), phi1 (Newton on Phi1)
   moments         moments
   output          result.Result | writer (sfincsOutput) | console
```

### 1. Input

There are two front doors, and both end in the same kernels.

- **Native case.** `dkx.Case` (`dkx.config`) is an immutable, validated model
  in physical units: surfaces, species in m⁻³ and keV, electric field in kV/m,
  resolution, solver options. `Case.from_file` reads TOML or JSON, validates it,
  and gives it a deterministic case ID. `dkx.run(case)` dispatches to
  `dkx.execution.run_case`, which builds grids, geometry, species and operators
  directly from the physical fields without writing a namelist. See
  {doc}`user_guide/inputs`.
- **SFINCS namelist.** `dkx.run("input.namelist")` reads a SFINCS version 3 deck
  exactly as the Fortran code would (`dkx.namelist`, `dkx.inputs`), with the
  upstream defaults and validation, and dispatches by `RHSMode` to
  `run_profile`, `run_transport_matrix` or `run_geometry` in `dkx.run`. See
  {doc}`user_guide/sfincs_namelist`.

`dkx.runtime.configure()` runs before the first JAX import on either path: it
enables float64, sizes the XLA host threadpool (`--cores`), sets single-thread
host BLAS, and sets up the compilation cache.

### 2. Geometry

`dkx.magnetic_geometry` builds the flux-surface geometry for every SFINCS
`geometryScheme`: analytic models (1 to 4), VMEC `wout` files (5, with the text
form handled by `dkx.vmec_ascii`), Boozer `.bc` files (11, 12) and the
remaining upstream schemes. It returns $\hat B$, its derivatives and the
covariant and contravariant components on the $(\theta, \zeta)$ grid, truncated
to the modes the grid resolves. JAX-native producers (a `vmex` equilibrium
through `booz_xform_jax`) enter through `dkx.workflows.geometry_adapters` and
keep the geometry differentiable. See {doc}`physics/geometry`.

### 3. Grids

`dkx.phase_space.make_grids` builds the angular grids and differentiation
matrices, the Legendre coupling coefficients, the Landreman–Ernst speed grid
(`dkx.xgrid` holds the polynomial kernel) and the $N_\xi(x)$ ramp into one
`Grids` object. See {doc}`numerics/discretization`.

### 4. Species and collisions

`dkx.species` holds charges, masses, profiles, gradients and collisionality as
JAX pytrees, in the SFINCS normalization of `dkx.constants` (with SI
conversions in `dkx.units`). `dkx.collisions` builds the pitch-angle-scattering,
linearized Fokker–Planck or improved Sugama operator for those species on the
speed grid. See {doc}`physics/normalizations` and {doc}`physics/collisions`.

### 5. Operator assembly

`dkx.drift_kinetic.KineticOperator` combines geometry, grids, species,
collisions and drives into one immutable pytree. It never forms a matrix:
`apply` is the matrix-free action of the bordered drift-kinetic operator, `rhs`
the drives for the requested `RHSMode`, `legendre_blocks` and
`to_block_tridiagonal` the analytic Legendre blocks for the structured solver,
and `residual_phi1` the nonlinear residual when $\Phi_1$ is active. See
{doc}`physics/drift_kinetic_equation` and {doc}`physics/drives_and_rhs_modes`.

### 6. Route selection

`dkx.solve.solve(op, rhs, method="auto")` reads the operator's structure and
its memory estimates and picks one of three routes, printing a one-line reason:
the structured direct block elimination for the block-tridiagonal family (full
or truncated storage by memory), recycled Krylov for everything else, and the
sparse direct referee on request or as the last escalation rung. See
{doc}`numerics/solver_routes`.

### 7. Solve

- **Structured direct**: block-Thomas elimination along $L$ over independent
  $(s, x)$ chains, from `solvax.direct`, with one refinement sweep.
- **Recycled Krylov**: GCROT-recycled FGMRES preconditioned by an exact solve of
  a simplified operator (`dkx.coarse_precond`, with the alternatives
  `dkx.sparse_precond` and `dkx.multigrid`).
- **Sparse direct**: `dkx.assembly` recovers the sparse matrix from grouped
  operator products; SOLVAX equilibrates and factors it on the host.

Every route recomputes the original residual before a solution is accepted.
With `differentiable=True` the structured and Krylov routes are wrapped in an
implicit-function-theorem adjoint ({doc}`numerics/differentiation`).

Two outer loops call the linear solve repeatedly: `dkx.er` finds the ambipolar
$E_r$ root of the radial current, and `dkx.phi1` runs Newton iterations on
$\Phi_1$ with the linear solve as its inner step. Scans batch many solves
through `dkx.batch` ({doc}`numerics/compilation_and_parallelism`).

### 8. Moments

`dkx.moments` reduces the solved distribution to the SFINCS output quantities:
particle and heat fluxes, flows, the bootstrap current $\langle
\mathbf{j}\cdot\mathbf{B}\rangle$ (`FSABjHat`), transport-matrix entries, NTV
and classical fluxes, keyed by `sfincsOutput.h5` names.

### 9. Result and output

- The native path returns an immutable `dkx.Result` (`dkx.result`): named
  read-only arrays in SI units, a plain-text summary, and a `certificate()`
  holding the route, route reason, residual evidence, iterations, versions,
  precision, device, geometry checksum, timings and peak memory. It saves to a
  versioned NetCDF contract. See {doc}`user_guide/outputs`.
- The namelist path writes `sfincsOutput.h5` (or `.nc`, `.npz`) with the
  upstream dataset names through `dkx.writer`, and `dkx.console` prints the
  Fortran-parity stdout blocks.

## Package layout

The package is flat: canonical root modules named for the physics or numerics
they own, with at most one level of domain folders. There are two such folders,
``dkx/validation`` and ``dkx/workflows``, and neither contains a further
subpackage.

### Root modules

```{list-table}
:header-rows: 1
:widths: 24 76

* - Module
  - Role
* - `__init__`
  - Public exports; importing it does not import JAX or change the environment
* - `__main__`
  - `python -m dkx` entry point to the CLI
* - `_version`
  - Single source of the release version
* - `runtime`
  - `configure()`: numpy check, float64, XLA threadpool, BLAS threads, compilation cache, distributed bootstrap
* - `config`
  - Native `Case` model, validation, case IDs, template and JSON Schema export
* - `execution`
  - Native `Case` normalization and execution without a namelist
* - `namelist`
  - Minimal parser for SFINCS Fortran namelists
* - `inputs`
  - Typed SFINCS v3 namelist with upstream defaults and validation
* - `input_compat`
  - Option lookup across namelist and mapping forms, input aliases
* - `paths`
  - Resolution of data, equilibrium and cache paths
* - `constants`
  - SFINCS normalization constants and radial-coordinate conversions
* - `units`
  - SFINCS reference values and conversion of "Hat" outputs to SI
* - `species`
  - Species containers: charges, masses, profiles, gradients, collisionality
* - `magnetic_geometry`
  - Flux-surface geometry for every SFINCS geometry scheme; VMEC and Boozer readers
* - `vmec_ascii`
  - The LIBSTELL text form of a VMEC `wout`
* - `phase_space`
  - Angular, pitch and speed grids, derivative matrices, the $N_\xi(x)$ ramp
* - `xgrid`
  - SFINCS-compatible speed-grid quadrature and differentiation kernel
* - `collisions`
  - Pitch-angle scattering, linearized Fokker–Planck and improved Sugama kernels
* - `drift_kinetic`
  - `KineticOperator`: term assembly, matrix-free apply, Legendre blocks, drives, bordered constraints
* - `solve`
  - The three-route `auto` policy, stall escalation, factor reuse, implicit differentiation
* - `coarse_precond`
  - The coarse Krylov preconditioner, its pins and its three storage policies
* - `sparse_precond`
  - The same simplified operator inverted exactly in a fill-reducing order on the host
* - `multigrid`
  - Semicoarsened geometric multigrid preconditioner and pitch-basis diagnostics
* - `assembly`
  - Sparse matrix of the operator recovered from grouped operator products
* - `collocation`
  - Experimental pitch-collocation discretization with a multigrid-preconditioned solve
* - `phi1`
  - Nonlinear $\Phi_1$ quasineutrality Newton solve and its differentiable state
* - `er`
  - Radial current, Brent ambipolar root search, differentiable `ambipolar_er`
* - `ambipolar`
  - Ambipolar post-processing over precomputed scan directories
* - `batch`
  - Batched $E_r$ and surface scans over `jax.vmap`, memory-budgeted chunks, multi-device sharding
* - `moments`
  - Velocity-space moments and transport diagnostics of solved states
* - `monoenergetic`
  - Monoenergetic database scans and energy convolution to thermal transport matrices
* - `momentum_correction`
  - Sugama–Nishimura momentum correction of pitch-angle solves: particle, heat and higher Sonine flows, friction from the Fokker–Planck operator
* - `bounce_averaged`
  - Differentiable bounce-averaged $1/\nu$ transport and effective ripple
* - `shaing_callen`
  - Collisionless Shaing–Callen limit of the bootstrap coefficient
* - `impurity`
  - Classical impurity transport, screening and charge-state scans
* - `variational`
  - Variational bounds on the monoenergetic $D_{11}$ coefficient
* - `validity`
  - Local-validity diagnostics: orbit width, $E\times B$ resonances, collisionality regime, $E_*$
* - `sensitivity`
  - JVP, VJP, adjoint and algebraic-error helpers
* - `bootstrap`
  - Kinetic bootstrap current as a VMEX optimization objective term
* - `run`
  - End-to-end RHSMode 1/2/3 drivers and the `run` entry point
* - `api`
  - Stable public data contracts (`SolverOptions`, scan preparation, output helpers)
* - `result`
  - Immutable native `Result` and the version-1 NetCDF contract
* - `writer`
  - `sfincsOutput` writer for RHSMode 1/2/3 and geometry-only runs
* - `console`
  - Fortran-parity stdout blocks
* - `io`
  - Reading and serializing SFINCS output files (HDF5, NetCDF, NPZ)
* - `solver_trace`
  - Versioned solver-trace schema with JSON and HDF5 serialization
* - `profiling`
  - Opt-in runtime and memory profiling
* - `compare`
  - Output comparison, frozen-reference parity and benchmark tables
* - `plotting`
  - Output plots for the CLI and examples
* - `representative`
  - One-command representative run from an equilibrium to publication panels
* - `cli`
  - The `dkx` command line ({doc}`user_guide/cli`)
```

### Domain folders

```{list-table}
:header-rows: 1
:widths: 30 70

* - Folder
  - Contents
* - ``dkx/validation``
  - Fortran and PETSc fixture readers (`fortran`), release-hosted equilibrium fetching and checksums (`data_fetch`, `equilibria_manifest.json`)
* - ``dkx/workflows``
  - End-to-end workflows: native ambipolar scans (`ambipolar_native`), convergence studies with Richardson error bars (`converge`), JAX-native geometry adapters (`geometry_adapters`), optimization objectives and promotion gates (`optimization`), declarative parameter scans (`scan`) and $E_r$ scan drivers (`scans`)
```
