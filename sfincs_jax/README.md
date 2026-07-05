# sfincs_jax Source Layout

This directory contains the importable `sfincs_jax` package. The package uses a
small public API and a one-level domain structure: root modules are for
user-facing entry points, while domain folders own physics, geometry,
discretization, operators, solvers, outputs, validation, and research workflows.

## Where To Start

- `api.py`: high-level Python helpers for running solves from scripts or
  notebooks.
- `__init__.py`: stable package exports and compatibility aliases.
- `cli.py` and `__main__.py`: command-line entry points used by `sfincs_jax`.
- `solver.py`: public solve orchestration and solver-result metadata.
- `ambipolar.py`: public ambipolar electric-field workflows.
- `sensitivity.py`: differentiable JVP, VJP, adjoint, and implicit derivative
  helpers.
- `plotting.py`: output plotting utilities used by `sfincs_jax --plot`.
- `compare.py`: comparison utilities for frozen SFINCS Fortran v3 references,
  strict numeric HDF5 parity, regression fixtures, and benchmark summaries.
- `io.py`, `namelist.py`, `input_compat.py`, and `paths.py`: file I/O, input
  parsing, input compatibility, and cache or data-path helpers.
- `diagnostics.py` and `grids.py`: stable scientific helper APIs for
  flux-surface averages, `uHat`, and Fortran-v3 grid/stencil construction.
- `profiling.py`: lightweight timers and memory probes used by CLI, examples,
  and benchmark paths.

Normal users should use these public modules or the CLI. Implementation modules
inside domain folders are for contributors and advanced research workflows.

## Domain Folders

- `discretization/`: grids, differentiation stencils, active indices, and
  coordinate maps.
- `geometry/`: analytic magnetic geometries, VMEC `wout` loading, Boozer data,
  and JAX geometry adapters.
- `operators/`: drift-kinetic operator terms, profile-response layouts,
  matrix-free actions, sparse operator helpers, and full-system assembly.
  Profile-response owners use flat `profile_*.py` names.
- `physics/`: collision, classical-transport, bootstrap-current, and
  normalization formulas.
- `problems/`: physical problem owners, including flat RHSMode-1
  `profile_*.py` modules, flat RHSMode-2/3 `transport_*.py` modules, and
  ambipolar root solves.
- `solvers/`: Krylov dispatch, solver-path selection, sparse/native factors,
  memory models, and flat `preconditioner_*.py` modules.
- `outputs/`: HDF5/NetCDF/NPZ schemas, writer logic, and post-solve
  diagnostics.
- `validation/`: frozen-reference loading, Fortran/PETSc fixture readers,
  parity checks, release-data manifest/fetching, QI device evidence gates, and
  validation-figure helpers.
- `workflows/`: optional research workflows that combine public APIs into
  scans, optimization tasks, and reusable evidence-generation tasks.

## Main Implementation Owners

Use this map before adding a file or following an internal import:

- `operators/profile_system.py`: RHSMode-1 full-system operator, RHS assembly,
  matrix-free residual and JVP wrappers, and constraint-source moment kernels.
- `operators/profile_layout.py`: RHSMode-1 full, active, field-split, and
  Fortran-style compressed pitch-space layouts.
- `operators/profile_fblock.py`: kinetic distribution-function block assembly.
- `operators/profile_full_system.py`: explicit sparse/full-system assembly and
  reduced-Pmat helpers.
- `problems/profile_solve.py`: RHSMode-1 solve orchestration.
- `problems/profile_policies.py`: RHSMode-1 automatic solver, fallback, and
  environment-policy decisions.
- `problems/profile_residual.py`: RHSMode-1 post-Krylov residual correction and
  polish stages.
- `problems/transport_solve.py`: RHSMode-2/3 transport solve orchestration.
- `problems/transport_linear_system.py`: transport linear-system construction,
  batched RHS solves, and dense/host/JAX dispatch helpers.
- `problems/transport_parallel_runtime.py`: RHSMode-2/3 whichRHS
  parallelism, GPU subprocess worker CLI, worker payload schemas, and result
  merging.
- `solvers/preconditioning.py`: shared preconditioner caches, projection
  helpers, and RHSMode-1 preconditioner dispatch.
- `solvers/preconditioner_full_fp_kinetic.py`: full-FP RHSMode-1 kinetic,
  species-block, and collision preconditioners.
- `solvers/preconditioner_schur_profile.py`: RHSMode-1 Schur/coarse
  preconditioners and active sparse-coarse policies.
- `outputs/writer.py`: public output-write orchestration.
- `outputs/formats.py`: HDF5/NetCDF/NPZ schemas, output cache persistence, and
  `export_f` mapping helpers.
- `outputs/rhsmode1.py`: RHSMode-1 output diagnostics and output-path policy.
- `validation/artifacts.py`: validation artifact manifests and research-lane
  evidence gates.

Do not reintroduce helper-only modules for residual wrappers, constraint-source
kernels, compressed pitch layouts, full-FP species preconditioners, output
caches, preconditioner dispatch, or research-lane manifests. Those concepts are
owned by the canonical modules listed above and guarded by source-tree tests.

## Design Rules

- Keep root modules user-facing. New implementation code should go into a
  domain folder, not the package root. Root support modules are acceptable only
  when they are documented stable APIs such as `diagnostics.py`, `grids.py`, or
  `profiling.py`.
- Keep package depth shallow. The consolidation target is one folder below
  `sfincs_jax/`; deeper folders must be explicitly justified in
  `plan_final.md` and covered by import-structure tests.
- Prefer descriptive domain names over historical names. For example, use
  `profile_*`, `transport_*`, or `preconditioner_*` names only when they point
  to the physics or numerical role of the module.
- Preserve public imports through compatibility aliases only when they are
  documented user workflows. Internal imports should point at the canonical
  domain modules, and deleted non-root facades should not be reintroduced.
- Keep large validation data out of the git clone and wheel. Small frozen
  references can live in `tests/fixtures`; large equilibria or benchmark outputs
  should be fetched through `validation.data_fetch` from release assets. The
  embedded manifest lives in `validation/equilibria_manifest.json`; the large
  files named by that manifest remain release-hosted.

## Contributor Workflow

When moving code:

1. Add or update the import-structure test before moving files.
2. Move one coherent owner at a time, not one helper at a time.
3. Update internal imports, docs API references, and compatibility aliases in
   the same commit.
4. Run focused unit tests for the moved owner, then run the package import and
   docs checks.
5. Do not commit generated output, caches, local traces, or large benchmark
   artifacts.

The authoritative consolidation sequence lives in `../plan_final.md`.
