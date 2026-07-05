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
  regression fixtures, and benchmark summaries.
- `io.py`, `namelist.py`, `input_compat.py`, and `paths.py`: file I/O, input
  parsing, input compatibility, and cache or data-path helpers.
- `diagnostics.py` and `grids.py`: stable scientific helper APIs for
  flux-surface averages, `uHat`, and Fortran-v3 grid/stencil construction.
- `profiling.py`: lightweight timers and memory probes used by CLI, examples,
  and benchmark paths.

Compatibility facades are intentionally small:

- `v3_driver.py`: facade for historical monolithic imports; it should stay
  small and contain no physics or solver implementation.

Normal users should use these public modules or the CLI. Implementation modules
inside domain folders are for contributors and advanced research workflows.

## Domain Folders

- `discretization/`: grids, differentiation stencils, active indices, and
  coordinate maps.
- `geometry/`: analytic magnetic geometries, VMEC `wout` loading, Boozer data,
  and JAX geometry adapters.
- `operators/`: drift-kinetic operator terms, matrix-free actions, sparse
  operator helpers, and profile-response operator assembly. Profile-response
  owners use flat `profile_*.py` names.
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
