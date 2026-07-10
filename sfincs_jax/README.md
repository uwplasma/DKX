# sfincs_jax Source Layout

This directory contains the importable `sfincs_jax` package. The architecture
is the canonical stack of flat, physics-named root modules
(`plan_final.md`, "Source Structure Rules"): one input file plus one geometry
runs through `inputs -> drift_kinetic -> solve -> moments -> writer/console`,
and the public API/CLI route every supported case through that chain by
default. The remaining one-level domain packages are explicitly transitional:
they are the interim owners of the deferred features (Phi1/quasineutrality,
tangential magnetic drifts, constraint schemes 3/4, mapped speed grids,
export_f, non-stellarator-symmetric VMEC) plus a handful of workflow surfaces
(`.npz` output, solver traces, scan-er, ambipolar, sensitivity/compare/plot
utilities), and they shrink to zero as each vertical slice lands.

## The Canonical Stack (the architecture)

| Canonical owner | Purpose |
| --- | --- |
| `constants.py`, `species.py` | Normalizations, radial-coordinate Jacobians, species pytrees, collisionality. |
| `phase_space.py` | Theta/zeta grids and derivative matrices, Legendre pitch machinery, speed grid, Nxi-for-x ramps. |
| `magnetic_geometry.py` | All supported geometry schemes, VMEC/Boozer readers, differentiable Fourier path. |
| `collisions.py` | Pitch-angle scattering and full Fokker-Planck with Rosenbluth terms. |
| `drift_kinetic.py` | The `KineticOperator`: term assembly, matrix-free apply, analytic Legendre blocks, RHS drives, bordered constraints. |
| `solve.py` | Three-tier policy (structured block elimination, preconditioned recycled Krylov, host direct referee) on the optional `solvax` library; implicit differentiation. |
| `moments.py` | Velocity-space moments, flux families, transport matrices, NTV, classical transport, keyed by sfincsOutput.h5 names. |
| `inputs.py`, `console.py` | Typed namelist with Fortran-cited defaults/validation; byte-parity Fortran stdout blocks. |
| `run.py` | End-to-end RHSMode 1/2/3 drivers (`run_profile`, `run_transport_matrix`). |
| `writer.py` | Canonical `sfincsOutput.h5`/`.nc` writer for RHSMode 1/2/3. |
| `api.py`, `cli.py`, `__main__.py` | Thin public surface over the canonical modules. |

The CLI (`write-output` and the bare-run form) dispatches RHSMode 1/2/3 decks
through `run.py` unless `cli.deck_requires_legacy_pipeline` reports a deferred
feature (or a legacy-only CLI option such as `.npz` output, `--solver-trace`,
`--geometry-only`, `--no-fortran-layout`, `--no-overwrite` is requested), in
which case it prints the reason and falls back to the retained legacy owner
(`io.write_sfincs_jax_output_h5`).

## Other Root Modules

| Module | Role |
| --- | --- |
| `__init__.py` | Public package exports, JAX precision/cache setup, compatibility aliases. |
| `ambipolar.py` | Public ambipolar-root workflows (legacy-stack owner until the canonical `er.py` slice lands). |
| `sensitivity.py` | JVP/VJP, adjoint, and implicit differentiation helpers. |
| `plotting.py` | Output plotting used by the CLI and examples. |
| `compare.py` | HDF5 comparison, frozen-reference parity, benchmark-table utilities. |
| `io.py`, `namelist.py`, `input_compat.py`, `paths.py` | File formats, SFINCS-style namelist parsing, input aliases, data/cache paths. `io.py` is the retained legacy write entry. |
| `diagnostics.py`, `grids.py`, `profiling.py` | Stable diagnostics, v3 grid helpers, timers, memory probes. |

## Transitional Domain Packages

These packages are the legacy stack. They exist only until the deferred
features are consolidated into the canonical root modules; every slice that
promotes a case family deletes its superseded owners in the same series
(`plan_final.md`, "Repository-Wide Line Sweep"). Do not add new implementation
here unless it serves a deferred feature.

- `discretization/`: v3 grids, stencils, mapped-x-grid coordinate maps.
- `geometry/`: Boozer/VMEC readers and JAX geometry adapters used by the
  legacy stack and workflows.
- `operators/`: the legacy matrix-free profile-response operator
  (`profile_system.py`, `profile_fblock.py`, term modules, layouts).
- `physics/`: legacy collision/classical-transport formula owners.
- `problems/`: legacy RHSMode-1 (`profile_*.py`) and RHSMode-2/3
  (`transport_*.py`) solve orchestration plus the ambipolar Brent solve.
- `solvers/`: legacy Krylov wrappers, dispatch, and the retained
  `preconditioner_*.py` family the legacy auto policy can still select.
- `outputs/`: legacy HDF5/NetCDF/NPZ schemas, writer logic, export_f mapping,
  and post-solve diagnostics.
- `validation/`: frozen-reference loading, Fortran/PETSc fixture readers,
  release-data manifest/fetching, evidence gates.
- `workflows/`: scans, optimization support, and mapped-x-grid workflows.

The deleted sparse-direct/CSR-assembly solver families
(`operators/profile_full_system.py`, `solvers/explicit_sparse.py`, the
`profile_sparse_*` and `preconditioner_xblock_*_sparse` modules, and their
policy plumbing) must not be reintroduced; the retained legacy auto route is
the matrix-free Krylov policy with the dense/pas/collision preconditioner
families plus the SciPy rescue, and the canonical `solve.py` tiers own the
supported surface.

## Main Legacy Implementation Owners

- `operators/profile_system.py`: legacy RHSMode-1 full-system operator, RHS
  assembly, matrix-free residual/JVP wrappers, constraint-source kernels.
- `operators/profile_layout.py`: RHSMode-1 layout family.
- `operators/profile_fblock.py`: kinetic distribution-function block assembly.
- `problems/profile_solve.py`: legacy RHSMode-1 solve orchestration (the
  deferred-deck fallback).
- `problems/profile_policies.py`: legacy RHSMode-1 automatic solver policy.
- `problems/profile_phi1_newton.py`: the Phi1/quasineutrality Newton-Krylov
  owner (deferred slice).
- `problems/transport_solve.py`, `problems/transport_linear_system.py`,
  `problems/transport_parallel_runtime.py`: legacy RHSMode-2/3 orchestration,
  retained for `.npz`/solver-trace/export_f options and mapped-x-grid
  workflows.
- `solvers/krylov.py`: GMRES/FGMRES/BiCGStab/TFQMR wrappers, residual
  histories, distributed GMRES contracts.
- `solvers/preconditioning.py`: shared preconditioner caches and RHSMode-1
  preconditioner dispatch.
- `solvers/preconditioner_full_fp_kinetic.py`,
  `solvers/preconditioner_pas_*.py`, `solvers/preconditioner_schur_profile.py`,
  `solvers/preconditioner_xblock_{block_jacobi,radial}.py`,
  `solvers/preconditioner_domain_decomposition.py`,
  `solvers/preconditioner_transport_matrix.py`: the preconditioner kinds the
  legacy auto policy can select.
- `outputs/writer.py`, `outputs/formats.py`, `outputs/rhsmode1.py`,
  `outputs/transport.py`: legacy output writing and diagnostics.
- `validation/artifacts.py`: validation artifact manifests and evidence gates.

## Design Rules

- New stable code goes into a flat, physics-named canonical root module; the
  domain packages only shrink.
- Keep package depth shallow: one folder below `sfincs_jax/`, no nested
  packages.
- Stable file names describe physics or numerics — no version suffixes or
  experiment names (`plan_final.md`, "Source Structure Rules").
- No env-var-only solver routes in stable code; opt-in switches are namelist
  or API arguments with documented semantics.
- Keep large validation data out of the git clone and wheel; large equilibria
  are fetched through `validation.data_fetch` from release assets
  (`validation/equilibria_manifest.json`).

## Stability And Compatibility

The canonical root modules are the stable import surface. Legacy domain
modules are transitional: they may change or disappear as slices land, with
their public behavior preserved through the canonical replacements and the
parity gates. Compatibility aliases may remain in `__init__.py` only while a
documented workflow needs them.

## Generated Files Policy

Do not commit `__pycache__`, `.pyc`, profiling traces, HDF5/NetCDF/NPZ solve
outputs, XLA profiles, or large equilibrium files inside `sfincs_jax/`.

## Contributor Workflow

1. Start from the slice queue in `../plan_final.md`; delete or extract before
   moving code.
2. Update the source-tree manifest, this file, and the plan in the same
   commit as any module addition, rename, or deletion.
3. Run focused tests plus the import/compile guards before committing; full
   suite at slice milestones.
