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
- `io.py`, `namelist.py`, and `paths.py`: file I/O, input parsing, and cache or
  data-path helpers.

Transitional root modules remain only where they are still documented or used by
compatibility tests:

- `diagnostics.py`: public flux-surface averages and `uHat` diagnostics used by
  output writers and API docs.
- `grids.py`: public grid helper retained while callers migrate to
  `discretization/`.
- `input_compat.py`: input-normalization helpers retained while namelist parsing
  and geometry defaults are consolidated.
- `profiling.py`: lightweight timers and memory probes used by CLI and
  benchmark paths.
- `v3_driver.py`: compatibility shim for former monolithic imports; it should
  stay small and contain no physics or solver implementation.

Normal users should use these public modules or the CLI. Implementation modules
inside domain folders are for contributors and advanced research workflows.

## Domain Folders

- `data/`: tiny manifest files for release-hosted equilibrium assets. Large
  `.bc`, `.nc`, and benchmark data are fetched on demand rather than stored in
  the git clone or wheel.
- `discretization/`: grids, differentiation stencils, active indices, and
  coordinate maps.
- `geometry/`: analytic magnetic geometries, VMEC `wout` loading, Boozer data,
  and JAX geometry adapters.
- `operators/`: drift-kinetic operator terms, matrix-free actions, sparse
  operator helpers, and profile-response operator assembly. Profile-response
  owners use flat `profile_*.py` names; `profile_response.py` is only a
  compatibility shim for the former nested import path.
- `physics/`: collision, classical-transport, bootstrap-current, and
  normalization formulas.
- `problems/`: physical problem owners, including flat RHSMode-1
  `profile_*.py` modules, flat RHSMode-2/3 `transport_*.py` modules, and
  ambipolar root solves. `profile_response.py` and `transport_matrix.py` are
  compatibility shims for former nested import paths.
- `solvers/`: Krylov dispatch, solver-path selection, sparse/native factors,
  memory models, and flat `preconditioner_*.py` modules. `preconditioners.py`
  is a compatibility index for former nested solver imports, not an
  implementation folder.
- `outputs/`: HDF5/NetCDF/NPZ schemas, writer logic, and post-solve
  diagnostics.
- `validation/`: frozen-reference loading, parity checks, release-data fetching,
  and validation-figure helpers.
- `workflows/`: optional research workflows that combine public APIs into
  scans, optimization tasks, and promotion campaigns.

## Design Rules

- Keep root modules user-facing. New implementation code should go into a
  domain folder, not the package root.
- Keep package depth shallow. The consolidation target is one folder below
  `sfincs_jax/`; deeper folders must be explicitly justified in
  `plan_final.md` and covered by import-structure tests.
- Prefer descriptive domain names over historical names. For example, use
  `profile_*`, `transport_*`, or `preconditioner_*` names only when they point
  to the physics or numerical role of the module.
- Preserve public imports through compatibility aliases during one release
  cycle, but move internal imports to the canonical modules as soon as a module
  is consolidated.
- Keep large validation data out of the git clone and wheel. Small frozen
  references can live in `tests/fixtures`; large equilibria or benchmark outputs
  should be fetched through `validation.data_fetch` from release assets.

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
